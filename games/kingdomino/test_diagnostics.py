"""Tests for the phase-sliced calibration diagnostics (Milestone 2).

Synthetic Example objects drive the metrics directly; a tiny KingdominoNet (or a
mock returning fixed outputs) provides the forward pass.  No training run is
required — these validate the diagnostic maths and JSON-serialisability.
"""
import json

import numpy as np
import torch

from games.kingdomino.action_codec import NUM_JOINT_ACTIONS
from games.kingdomino.encoder import FLAT_LAYOUT, FLAT_SIZE
from games.kingdomino.network import KingdominoNet
from games.kingdomino.self_play import Example
from games.kingdomino.diagnostics import (
    _phase_mask,
    win_brier_by_phase,
    value_calibration_curve,
    policy_kl_by_phase,
    compute_all_diagnostics,
    check_alpha_transition,
    PHASE_THRESHOLDS,
)

_PROG = FLAT_LAYOUT["game_progress"]


def _make_example(progress, *, win_target=0.0, win_in_flat=None,
                  own_score=20.0, z=0.0, n_legal=6, uniform_policy=False,
                  rng=None):
    """Build one synthetic Example with the given game_progress in its flat vector."""
    rng = rng or np.random.default_rng(0)
    flat = rng.standard_normal(FLAT_SIZE).astype(np.float16)
    flat[_PROG] = np.float16(progress)
    if win_in_flat is not None:
        # Stash a target win_prob in flat[0] for mocks that read it back.
        flat[0] = np.float16(win_in_flat)
    my_board = rng.standard_normal((9, 13, 13)).astype(np.float16)
    opp_board = rng.standard_normal((9, 13, 13)).astype(np.float16)

    legal = rng.choice(NUM_JOINT_ACTIONS, size=n_legal, replace=False)
    legal_idx = np.sort(legal).astype(np.int32)
    if uniform_policy:
        policy_idx = legal_idx.copy()
        policy_val = np.full(n_legal, 1.0 / n_legal, dtype=np.float32)
    else:
        k = max(1, n_legal // 2)
        policy_idx = np.sort(legal_idx[:k]).astype(np.int32)
        w = rng.random(k) + 1e-3
        policy_val = (w / w.sum()).astype(np.float32)

    return Example(
        my_board=my_board, opp_board=opp_board, flat=flat,
        policy_idx=policy_idx, policy_val=policy_val, legal_idx=legal_idx,
        z=float(z), own_score=float(own_score), opp_score=float(20.0),
        win_target=float(win_target),
    )


class _MockNet:
    """Stand-in for KingdominoNet returning controllable head/logit outputs.

    win_from_flat: return flat[:, 0] as win_prob (lets a test pin exact values).
    win_const:     return a constant win_prob for every example.
    logits_value:  fill value for the (B, 3390) policy logits (0 ⇒ uniform).
    """
    def __init__(self, win_from_flat=False, win_const=None, logits_value=0.0):
        self.win_from_flat = win_from_flat
        self.win_const = win_const
        self.logits_value = logits_value

    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, mb, ob, flat):
        B = mb.shape[0]
        if self.win_from_flat:
            win = flat[:, 0].clamp(0.0, 1.0)
        elif self.win_const is not None:
            win = torch.full((B,), float(self.win_const))
        else:
            win = torch.zeros(B)
        own = torch.zeros(B)
        opp = torch.zeros(B)
        logits = torch.full((B, NUM_JOINT_ACTIONS), float(self.logits_value))
        return own, opp, win, logits


# ─── tests ─────────────────────────────────────────────────────────────────
def test_win_brier_by_phase_shape():
    rng = np.random.default_rng(1)
    progs = np.linspace(0.0, 1.0, 100)
    examples = [_make_example(float(p), win_target=float(rng.integers(0, 2)),
                              rng=rng)
                for p in progs]
    net = KingdominoNet(channels=8, blocks=1)
    out = win_brier_by_phase(examples, net, device="cpu", batch_size=32)

    for key in ("win_brier_opening", "win_brier_midgame", "win_brier_endgame",
                "n_opening", "n_midgame", "n_endgame", "baseline_brier_endgame"):
        assert key in out
    for phase in ("opening", "midgame", "endgame"):
        v = out[f"win_brier_{phase}"]
        assert v is None or (isinstance(v, float) and v >= 0.0)
        assert isinstance(out[f"n_{phase}"], int)


def test_phase_mask_coverage():
    examples = [_make_example(0.1), _make_example(0.5), _make_example(0.9)]
    m_open = _phase_mask(examples, "opening")
    m_mid = _phase_mask(examples, "midgame")
    m_end = _phase_mask(examples, "endgame")

    assert list(m_open) == [True, False, False]
    assert list(m_mid) == [False, True, False]
    assert list(m_end) == [False, False, True]
    # exact partition: every example in exactly one phase
    total = m_open.astype(int) + m_mid.astype(int) + m_end.astype(int)
    assert list(total) == [1, 1, 1]


def test_calibration_ece_perfect():
    rng = np.random.default_rng(2)
    examples = []
    for _ in range(200):
        p = float(rng.random())
        # win_target equals the win_prob the mock will report → per-bin
        # actual_rate == pred_mean → ECE ≈ 0.
        examples.append(_make_example(float(rng.random()), win_target=p,
                                      win_in_flat=p, rng=rng))
    net = _MockNet(win_from_flat=True)
    out = value_calibration_curve(examples, net, device="cpu")
    assert out["cal_ece"] < 0.01


def test_calibration_ece_overconfident():
    rng = np.random.default_rng(3)
    # Predicted 0.9 for all, but actual win rate ~0.6.
    targets = (rng.random(300) < 0.6).astype(float)
    examples = [_make_example(float(rng.random()), win_target=float(t), rng=rng)
                for t in targets]
    net = _MockNet(win_const=0.9)
    out = value_calibration_curve(examples, net, device="cpu")
    assert out["cal_ece"] > 0.2


def test_policy_kl_zero_for_matching_policy():
    rng = np.random.default_rng(4)
    # Uniform MCTS policy over legal actions; mock net returns flat (zero) logits
    # → masked softmax is also uniform over legal → KL == 0.
    examples = [_make_example(float(rng.random()), uniform_policy=True, rng=rng)
                for _ in range(60)]
    net = _MockNet(logits_value=0.0)
    out = policy_kl_by_phase(examples, net, device="cpu")
    assert out["policy_kl_overall"] is not None
    assert abs(out["policy_kl_overall"]) < 1e-6


def test_alpha_trigger_fires():
    rows = [{"win_brier_endgame": 0.08, "baseline_brier_endgame": 0.20}
            for _ in range(5)]   # ratio 0.4 < 0.5
    assert check_alpha_transition(rows) is True


def test_alpha_trigger_does_not_fire_early():
    rows = [{"win_brier_endgame": 0.08, "baseline_brier_endgame": 0.20}
            for _ in range(4)]
    assert check_alpha_transition(rows) is False


def test_compute_all_diagnostics_runs():
    rng = np.random.default_rng(5)
    examples = []
    for _ in range(200):
        examples.append(_make_example(
            float(rng.random()),
            win_target=float(rng.integers(0, 2)),
            own_score=float(rng.integers(0, 60)),
            z=float(rng.uniform(-1, 1)),
            rng=rng,
        ))
    net = KingdominoNet(channels=8, blocks=1)
    out = compute_all_diagnostics(examples, net, device="cpu", diag_n=128,
                                  batch_size=32)

    assert isinstance(out, dict)

    def _ok(v):
        if v is None or isinstance(v, (int, float)):
            return True
        if isinstance(v, list):
            return all(_ok(x) for x in v)
        return False

    for k, v in out.items():
        assert _ok(v), f"non-serialisable value for {k}: {v!r}"
    # round-trips through JSON without error
    json.dumps(out)
