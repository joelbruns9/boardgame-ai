"""Step 2b gate: the Rust `NnueEvaluator` (dense forward over the ported encoder)
must match the PyTorch net it was exported from, and the loader must reject
malformed artifacts.

Self-contained (no skips, exercises the REAL exporter): each run builds a tiny
deterministic net, exports it via nnue.export.export_checkpoint to a temp .knnue,
loads it in Rust, and compares on real encoded positions. Covers the reviewer's 2b
checklist: PyTorch-vs-Rust forward equivalence on real positions for BOTH actors;
actor->P0 frame equivalence; explicit 2*sigmoid-1 conversion; a separate margin
check; finite outputs; and malformed-binary rejection (bad magic / version /
encoder signature / dims / truncation / trailing / non-finite weights). Isolates the
FORWARD pass by feeding both sides the SAME Rust-encoded features (the encoder has
its own bit-exactness gate).
"""
from __future__ import annotations

import random
import struct

import numpy as np
import pytest
import torch

import kingdomino_rust as kr

from games.kingdomino.game import GameState, Phase
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.nnue.net import TwoHeadNNUE, config_of
from games.kingdomino.nnue.data import INPUT_DIM
from games.kingdomino.nnue.export import export_checkpoint, HEADER_SIZE

MARGIN_SCALE = 40.0


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    """A tiny deterministic net exported through the real exporter -> temp .knnue."""
    torch.manual_seed(0)
    net = TwoHeadNNUE(input_dim=INPUT_DIM, acc_width=16, tail_hidden=8)
    net.eval()
    d = tmp_path_factory.mktemp("nnue")
    ckpt = d / "fixture.pt"
    torch.save({"state_dict": net.state_dict(), "config": config_of(net),
                "margin_scale": MARGIN_SCALE}, ckpt)
    bin_path, _man = export_checkpoint(str(ckpt))
    return {"net": net, "knnue": str(bin_path), "bytes": bin_path.read_bytes()}


def _collect_states(want_per_actor=6):
    """PLACE_AND_SELECT positions with their current actor, ensuring BOTH actors."""
    by_actor = {0: [], 1: []}
    seed = 0
    while (len(by_actor[0]) < want_per_actor or len(by_actor[1]) < want_per_actor) and seed < 600:
        st = GameState.new(seed=seed)
        rng = random.Random(seed * 5 + 1)
        for _ in range(400):
            if st.phase == Phase.GAME_OVER:
                break
            if st.phase == Phase.PLACE_AND_SELECT and len(st.legal_actions()) >= 2:
                rs = _rust_state_from_python(st)
                if rs is not None:
                    actor = kr.SearchEngine(rs).current_actor()
                    if len(by_actor[actor]) < want_per_actor:
                        by_actor[actor].append((st, rs, actor))
            st = st.step(rng.choice(st.legal_actions()))
        seed += 1
    return by_actor


def test_rust_forward_matches_pytorch_both_actors(artifact):
    net = artifact["net"]
    ev = kr.NnueEvaluator(artifact["knnue"])
    by_actor = _collect_states()
    assert by_actor[0] and by_actor[1], "need positions with both actors to move"

    max_e = max_p = max_m = 0.0
    for actor in (0, 1):
        for _pystate, rs, a in by_actor[actor]:
            mb, ob, flat = rs.encode(a)  # same bit-exact encoder the Rust eval uses
            x = np.concatenate([mb.reshape(-1), ob.reshape(-1), flat]).astype(np.float32)
            with torch.no_grad():
                logit, margin_norm = net(torch.from_numpy(x)[None])
            expected_py = float(torch.sigmoid(logit))
            margin_pts_py = float(margin_norm) * MARGIN_SCALE
            actor_val = 2.0 * expected_py - 1.0
            p0_py = actor_val if a == 0 else -actor_val

            p0_r, expected_r, margin_pts_r = ev.evaluate(rs)
            assert np.isfinite([p0_r, expected_r, margin_pts_r]).all()
            max_e = max(max_e, abs(expected_r - expected_py))
            max_p = max(max_p, abs(p0_r - p0_py))
            max_m = max(max_m, abs(margin_pts_r - margin_pts_py))
    # Both sides do an f32 forward; agreement should be ~f32 epsilon, far tighter
    # than any tensor-order/layout bug (which would be >0.01).
    assert max_e < 1e-5, f"expected_score max diff {max_e:.2e}"
    assert max_p < 2e-5, f"p0 max diff {max_p:.2e}"
    assert max_m < 1e-3, f"margin max diff {max_m:.2e} pts"


def test_frame_flip_internal_consistency(artifact):
    """p0_value = (+1 if actor==P0 else -1) * (2*expected_score - 1), both actors."""
    ev = kr.NnueEvaluator(artifact["knnue"])
    by_actor = _collect_states()
    for actor in (0, 1):
        for _pystate, rs, a in by_actor[actor]:
            p0_r, expected_r, _ = ev.evaluate(rs)
            sign = 1.0 if a == 0 else -1.0
            # 2*expected-1 is computed in f32 then widened, so agreement is to f32
            # precision, not bit-for-bit.
            assert abs(p0_r - sign * (2.0 * expected_r - 1.0)) < 1e-6
            assert -1.0 <= p0_r <= 1.0 and 0.0 <= expected_r <= 1.0


def test_search_with_nnue_eval_runs(artifact):
    by_actor = _collect_states(want_per_actor=1)
    rs = (by_actor[0] or by_actor[1])[0][1]
    search = kr.RustSearch(depth=2, eval="nnue", nnue_path=artifact["knnue"])
    v = search.value(rs, 2)
    assert np.isfinite(v) and -1.5 <= v <= 1.5
    assert search.nodes > 0
    assert search.choose_action(rs) is not None


def test_constructor_arg_errors(artifact):
    with pytest.raises(Exception):
        kr.NnueEvaluator("does_not_exist.knnue")
    with pytest.raises(ValueError):
        kr.RustSearch(depth=2, eval="nnue")  # missing nnue_path
    with pytest.raises(ValueError):
        kr.RustSearch(depth=2, eval="pick_aware", nnue_path=artifact["knnue"])


# ── malformed .knnue rejection ───────────────────────────────────────────────

def _mutations(good: bytes):
    """(name, mutated_bytes) for each malformed-input case."""
    out = []

    b = bytearray(good); b[0] ^= 0xFF
    out.append(("bad_magic", bytes(b)))

    b = bytearray(good); b[4:8] = struct.pack("<I", 999)
    out.append(("bad_version", bytes(b)))

    b = bytearray(good); (sig,) = struct.unpack_from("<Q", b, 32); struct.pack_into("<Q", b, 32, sig ^ 1)
    out.append(("encoder_sig_mismatch", bytes(b)))

    b = bytearray(good); struct.pack_into("<I", b, 12, 0)  # acc_width = 0
    out.append(("zero_dim", bytes(b)))

    b = bytearray(good); struct.pack_into("<I", b, 12, 1 << 25)  # acc_width absurd
    out.append(("absurd_dim", bytes(b)))

    out.append(("truncated", good[: len(good) - 64]))
    out.append(("trailing_bytes", good + b"\x00\x00\x00\x00"))

    b = bytearray(good)  # first weight float -> NaN
    b[HEADER_SIZE:HEADER_SIZE + 4] = struct.pack("<f", float("nan"))
    out.append(("non_finite_weight", bytes(b)))

    return out


@pytest.mark.parametrize("name", [m[0] for m in _mutations(b"\0" * 200)])
def test_malformed_knnue_rejected(artifact, tmp_path, name):
    muts = dict(_mutations(artifact["bytes"]))
    p = tmp_path / f"{name}.knnue"
    p.write_bytes(muts[name])
    with pytest.raises(Exception):
        kr.NnueEvaluator(str(p))
