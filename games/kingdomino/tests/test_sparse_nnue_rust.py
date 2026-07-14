"""Sparse v3 export, Rust forward, and incremental-search equivalence gates."""
from __future__ import annotations

import random
import struct

import numpy as np
import pytest
import torch

import kingdomino_rust as kr

from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.game import GameState, Phase
from games.kingdomino.nnue.sparse_encoder import CORE_SIZE, core_schema_hash
from games.kingdomino.nnue.sparse_export import HEADER_SIZE, export_checkpoint
from games.kingdomino.nnue.sparse_net import SparseNNUE, sparse_config_of
from games.kingdomino.nnue.summary_encoder import SUMMARY_SIZE, summary_schema_hash

MARGIN_SCALE = 40.0


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    torch.manual_seed(707)
    net = SparseNNUE(CORE_SIZE, SUMMARY_SIZE, acc_width=16, tail_hidden=8).eval()
    root = tmp_path_factory.mktemp("sparse_nnue")
    checkpoint = root / "fixture.pt"
    torch.save(
        {
            "state_dict": net.state_dict(),
            "config": sparse_config_of(net),
            "margin_scale": MARGIN_SCALE,
            "core_schema_hash": core_schema_hash(),
            "summary_schema_hash": summary_schema_hash(),
        },
        checkpoint,
    )
    binary, manifest = export_checkpoint(checkpoint)
    return {
        "net": net,
        "binary": str(binary),
        "bytes": binary.read_bytes(),
        "manifest": manifest,
    }


def _states(count=12):
    out = []
    for seed in range(100):
        state = GameState.new(seed=seed)
        rng = random.Random(seed * 11 + 3)
        while state.phase != Phase.GAME_OVER:
            if state.phase in (Phase.PLACE_AND_SELECT, Phase.FINAL_PLACEMENT):
                rs = _rust_state_from_python(state)
                if rs is not None:
                    out.append(rs)
                    if len(out) >= count:
                        return out
            state = state.step(rng.choice(state.legal_actions()))
    raise AssertionError("not enough states")


def test_stateless_rust_forward_matches_pytorch(artifact):
    evaluator = kr.SparseNnueEvaluator(artifact["binary"])
    saw = set()
    max_expected = max_p0 = max_margin = 0.0
    for state in _states():
        actor = int(state.current_actor())
        saw.add(actor)
        indices, summary = state.nnue_features(actor)
        indices = torch.from_numpy(np.asarray(indices, dtype=np.int64))
        summary_t = torch.from_numpy(np.asarray(summary, dtype=np.float32)[None])
        with torch.no_grad():
            expected_py, margin_py = artifact["net"].evaluate(
                indices, torch.tensor([0, len(indices)]), summary_t
            )
        expected_py = float(expected_py[0])
        margin_py = float(margin_py[0]) * MARGIN_SCALE
        p0_py = (2 * expected_py - 1) * (1 if actor == 0 else -1)
        p0_r, expected_r, margin_r = evaluator.evaluate(state)
        max_expected = max(max_expected, abs(expected_r - expected_py))
        max_p0 = max(max_p0, abs(p0_r - p0_py))
        max_margin = max(max_margin, abs(margin_r - margin_py))
    assert saw == {0, 1}
    assert max_expected < 1e-5
    assert max_p0 < 2e-5
    assert max_margin < 1e-3


def test_incremental_search_matches_stateless_including_chance(artifact):
    states = _states(6)
    checks = 0
    for i, state in enumerate(states):
        # enum_cap=1 forces sampled chance whenever this horizon crosses a deal.
        ref = kr.RustSearch(
            depth=3, enum_cap=1, chance_samples=8, seed=19,
            eval="sparse_nnue_ref", nnue_path=artifact["binary"],
        )
        inc = kr.RustSearch(
            depth=3, enum_cap=1, chance_samples=8, seed=19,
            eval="sparse_nnue", nnue_path=artifact["binary"],
        )
        rv = ref.value(state, 3)
        iv = inc.value(state, 3)
        assert abs(rv - iv) < 2e-5, f"state {i}: {rv} != {iv}"
        assert ref.nodes == inc.nodes
        if len(state.legal_actions()) > 1:
            assert ref.choose_action(state, 7) == inc.choose_action(state, 7)
            assert ref.nodes == inc.nodes
        checks += 1
    assert checks == 6


def test_sparse_export_omits_aux_heads_and_documents_layout(artifact):
    import json

    manifest = json.loads(artifact["manifest"].read_text())
    assert manifest["format_version"] == 3
    assert manifest["accumulator_layout"].startswith("feature-major")
    assert manifest["omitted_training_heads"] == ["aux_score_head", "aux_bonus_head"]
    assert not any("aux" in name for name in manifest["tensor_order"])


def _mutations(good: bytes):
    out = {}
    b = bytearray(good); b[0] ^= 0xFF; out["magic"] = bytes(b)
    b = bytearray(good); struct.pack_into("<I", b, 4, 999); out["version"] = bytes(b)
    b = bytearray(good); struct.pack_into("<I", b, 8, CORE_SIZE - 1); out["features"] = bytes(b)
    b = bytearray(good); struct.pack_into("<I", b, 12, SUMMARY_SIZE - 1); out["summary"] = bytes(b)
    b = bytearray(good); struct.pack_into("<I", b, 16, 0); out["zero_width"] = bytes(b)
    b = bytearray(good); b[28] ^= 1; out["core_hash"] = bytes(b)
    b = bytearray(good); b[36] ^= 1; out["summary_hash"] = bytes(b)
    b = bytearray(good); b[HEADER_SIZE:HEADER_SIZE + 4] = struct.pack("<f", float("nan")); out["nan"] = bytes(b)
    out["truncated"] = good[:-8]
    out["trailing"] = good + b"\0\0\0\0"
    return out


@pytest.mark.parametrize(
    "name",
    ["magic", "version", "features", "summary", "zero_width", "core_hash",
     "summary_hash", "nan", "truncated", "trailing"],
)
def test_sparse_loader_rejects_malformed(artifact, tmp_path, name):
    path = tmp_path / f"{name}.knnue"
    path.write_bytes(_mutations(artifact["bytes"])[name])
    with pytest.raises(Exception):
        kr.SparseNnueEvaluator(str(path))


def test_sparse_constructor_validation(artifact):
    with pytest.raises(ValueError):
        kr.RustSearch(eval="sparse_nnue")
    with pytest.raises(ValueError):
        kr.RustSearch(eval="pick_aware", nnue_path=artifact["binary"])
