"""Phase 4 plumbing: row-balanced schedule, buffer metadata and state,
per-variant generation, evaluation with seat cycles, the probe monitor,
the loop end to end, and an EXACT resume.

Everything runs tiny and on the CPU; the real run is launched by hand.

Run: python -m pytest games/cantstop/tests/test_phase4.py -q
"""

import json
import random

import numpy as np
import pytest
import torch

from games.cantstop import phase4
from games.cantstop.engine import ALL_RULESETS, RuleSet
from games.cantstop.encoder import FEATURE_SIZE
from games.cantstop.model import CantStopNet, NetEvaluator
from games.cantstop.portable_rng import PortableRng
from games.cantstop.rust_pool_equiv import MOCKS
from games.cantstop.schedule import SEED_ROWS_PER_GAME, RowSchedule, rules_key
from games.cantstop.train import ReplayBuffer, generate
from games.cantstop.variant_eval import ProbeSet, evaluate_variants

rust = pytest.importorskip("cantstop_rust", reason="run maturin develop first")

R2, R3 = RuleSet.make(2), RuleSet.make(3)


# ---- schedule ----

def test_seed_table_covers_every_rule_set():
    assert {rules_key(r) for r in ALL_RULESETS} == set(SEED_ROWS_PER_GAME)


def test_schedule_balances_rows_not_games():
    s = RowSchedule(ALL_RULESETS, rows_per_variant=4000)
    g = s.games()
    assert g[RuleSet.make(2)] == 409                       # 4000 / 9.8
    assert g[RuleSet.make(3, extended=True, blocking=True)] == 80   # / 50.0
    for r, n in g.items():
        assert n * SEED_ROWS_PER_GAME[rules_key(r)] >= 4000


def test_schedule_tracks_measured_lengths_and_round_trips():
    class Fake:
        def __init__(self, rules, rows):
            self.rules, self.rows = rules, rows

        def __len__(self):
            return self.rows

    s = RowSchedule([R2], rows_per_variant=100, smoothing=0.5)
    s.update([Fake(R2, 20), Fake(R2, 20)])                 # measured 20
    assert s.rows_per_game[rules_key(R2)] == pytest.approx(0.5 * 9.8 + 0.5 * 20)
    t = RowSchedule([R2], rows_per_variant=1)
    t.load(s.state())
    assert t.games() == s.games()


# ---- buffer metadata and state ----

def _rows(n, v):
    return (np.full((n, FEATURE_SIZE), v, np.float32),
            np.full((n, 4), 0.25, np.float32),
            {"variant": np.full(n, v, np.int16), "exact": np.zeros(n, np.int8)})


def test_buffer_keeps_meta_and_restores_exactly():
    b = ReplayBuffer(window_iterations=2)
    for v in range(3):
        b.add(*_rows(3 + v, v))
    assert b.iterations == 2 and len(b) == 4 + 5
    assert b.meta("variant").tolist() == [1] * 4 + [2] * 5
    c = ReplayBuffer.from_state(b.state())
    for (xa, ya), (xb, yb) in [(b.arrays(), c.arrays())]:
        assert np.array_equal(xa, xb) and np.array_equal(ya, yb)
    assert np.array_equal(b.meta("variant"), c.meta("variant"))


def test_buffer_rejects_misaligned_meta():
    x, y, m = _rows(3, 0)
    m["variant"] = m["variant"][:2]
    with pytest.raises(ValueError, match="meta"):
        ReplayBuffer(window_iterations=1).add(x, y, m)


# ---- per-variant generation ----

def test_generate_plays_the_scheduled_count_per_variant():
    res = generate([R2, R3], {R2: 3, R3: 2}, MOCKS["hashed"],
                   random.Random(1))
    assert [r.rules for r in res] == [R2] * 3 + [R3] * 2


# ---- evaluation and probes ----

def _net():
    torch.manual_seed(0)
    return NetEvaluator(CantStopNet(hidden=(16, 16)), device="cpu")


def test_evaluation_uses_complete_seat_cycles():
    out = evaluate_variants(_net(), [R3], games=4, seed=1)
    v = out[str(R3)]["vs_heuristic"]
    assert v["games"] == 6 and v["even_match"] == pytest.approx(1 / 3)


def test_probe_monitor_measures_every_variant():
    probes = ProbeSet.build([R2, R3], per_variant=4, seed=2, games=4)
    assert all(len(v) == 4 for v in probes.boards.values())
    m = probes.measure(_net(), alert=0.05)
    for key in (str(R2), str(R3)):
        assert m[key]["rmse"] >= 0 and isinstance(m[key]["alert"], bool)
    again = ProbeSet.from_state(probes.state()).measure(_net(), alert=0.05)
    assert again == m


# ---- the loop ----

KW = dict(rule_sets=(R2, R3), rows_per_variant=30, hidden=(16, 16),
          batch_size=32, replay_window=2, eval_every=2, eval_games=3,
          reference_iter=1, probes_per_variant=3, seed=5, device="cpu",
          conservative=0.25, aggressive=0.25, lr_schedule={2: 5e-4})


def test_loop_logs_per_variant_and_evaluates(tmp_path):
    phase4.run(tmp_path, iterations=2, **KW)
    rows = [json.loads(l) for l in open(tmp_path / "run.jsonl")]
    assert [r["iteration"] for r in rows] == [1, 2]
    assert set(rows[0]["variants"]) == {str(R2), str(R3)}
    assert rows[1]["lr"] == 5e-4 and rows[0]["lr"] == 1e-3
    assert "eval" not in rows[0] and set(rows[1]["eval"][str(R3)]) == {
        "vs_heuristic", "vs_iter0", "vs_iter1"}
    assert set(rows[1]["probes"]) == {str(R2), str(R3)}
    assert rows[1]["buffer_exact_share"] == 0.0
    assert (tmp_path / "run_meta.json").exists()
    assert rows[0]["attempted"] == rows[0]["games"]
    assert set(rows[0]["attempted"]) == {str(R2), str(R3)}


def test_exact_from_switches_targets_and_logs_the_mixture(tmp_path):
    phase4.run(tmp_path, iterations=2, exact_from=2, **KW)
    rows = [json.loads(l) for l in open(tmp_path / "run.jsonl")]
    assert [r["exact_targets"] for r in rows] == [False, True]
    assert 0.0 < rows[1]["buffer_exact_share"] < 1.0


def test_personas_from_delays_the_personas(tmp_path):
    phase4.run(tmp_path, iterations=2, personas_from=2, **KW)
    rows = [json.loads(l) for l in open(tmp_path / "run.jsonl")]
    assert [r["personas"] for r in rows] == [False, True]


def test_resume_continues_exactly(tmp_path):
    """3 iterations straight == 2, then --resume to 3: same weights, same
    log (timing fields aside)."""
    a, b = tmp_path / "a", tmp_path / "b"
    phase4.run(a, iterations=3, **KW)
    phase4.run(b, iterations=2, **KW)
    phase4.run(b, iterations=3, resume=True, **KW)
    wa = torch.load(a / "iter_0003.pt", weights_only=False)
    wb = torch.load(b / "iter_0003.pt", weights_only=False)
    sa, sb = wa["state_dict"], wb["state_dict"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    strip = lambda r: {k: v for k, v in r.items()
                       if not k.endswith("seconds")}
    la = [strip(json.loads(l)) for l in open(a / "run.jsonl")]
    lb = [strip(json.loads(l)) for l in open(b / "run.jsonl")]
    assert la == lb


def test_resume_refuses_a_changed_run(tmp_path):
    phase4.run(tmp_path, iterations=1, **KW)
    with pytest.raises(SystemExit, match="td_lambda"):
        phase4.run(tmp_path, iterations=2, resume=True,
                   **{**KW, "td_lambda": 0.5})
    with pytest.raises(SystemExit, match="exists"):
        phase4.run(tmp_path, iterations=2, **KW)


def test_capacity_probe_reads_the_phase4_buffer(tmp_path):
    from games.cantstop.capacity_probe import dataset_from_state
    phase4.run(tmp_path, iterations=2, **KW)
    xs, ys, vs, names = dataset_from_state(tmp_path / "state.pt")
    assert set(vs) == {0, 1} and names == [str(R2), str(R3)]
    assert sum(len(x) for x in xs) == sum(len(y) for y in ys)


def test_variant_selection_for_a_specialist():
    rs = phase4.parse_variants(["3:4:b", "2:3"])
    assert rs == (RuleSet.make(3, extended=True, blocking=True), RuleSet.make(2))
    assert phase4.parse_variants(None) == ALL_RULESETS
    with pytest.raises(SystemExit, match="PLAYERS:COLUMNS"):
        phase4.parse_variants(["3-4-b"])
    with pytest.raises(ValueError):
        phase4.parse_variants(["2:4"])                  # 2p plays to 3 or 5


# ---- unfinished-game guard ----

def _fake_results(rules, attempted, unfinished):
    from types import SimpleNamespace
    return [SimpleNamespace(rules=rules,winner=-1 if i<unfinished else 0)
            for i in range(attempted)]


def test_one_stalling_variant_cannot_hide_in_healthy_variants():
    results=[]
    for i,rules in enumerate(ALL_RULESETS):
        results += _fake_results(rules,20,2 if i==0 else 0)
    # 1% pooled, but 10% in the affected variant.
    with pytest.raises(phase4.TurnLimitExceeded) as error:
        phase4._check_unfinished(results,7)
    message=str(error.value)
    assert 'iteration 7' in message and str(ALL_RULESETS[0]) in message
    assert '2/20 (10.0%)' in message and '5%' in message


def test_unfinished_boundary_uses_actual_attempts_and_accepts_exactly_five_percent():
    results=_fake_results(R2,20,1)+_fake_results(R3,100,0)
    attempted,unfinished=phase4._check_unfinished(results,1)
    assert attempted=={str(R2):20,str(R3):100}
    assert unfinished=={str(R2):1}
    with pytest.raises(phase4.TurnLimitExceeded):
        phase4._check_unfinished(_fake_results(R2,19,1)+_fake_results(R3,100,0),1)


def test_unfinished_guard_rejects_empty_and_completely_stalled_generation():
    for results in ([],_fake_results(R2,3,3)):
        with pytest.raises(phase4.TurnLimitExceeded): phase4._check_unfinished(results,1)


def test_run_checks_variant_stalls_before_training(tmp_path,monkeypatch):
    results=_fake_results(R2,20,2)+_fake_results(R3,100,0)
    monkeypatch.setattr(phase4,'generate',lambda *args,**kwargs:results)
    def unexpected(*args,**kwargs): raise AssertionError('must stop before training')
    monkeypatch.setattr(phase4,'train_steps',unexpected)
    with pytest.raises(phase4.TurnLimitExceeded,match='per variant'):
        phase4.run(tmp_path,iterations=1,**KW)
    assert not (tmp_path/'iter_0001.pt').exists()
