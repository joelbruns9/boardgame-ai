from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest

from games.kingdomino.denial_search import DenialSearch, SearchConfig, public_state_key
from games.kingdomino.denial_signal_sweep import file_sha256
from games.kingdomino.reply_pilot import (
    _training_root_filter,
    _validate_training_roots,
    decode_array_blob,
    merge_shards,
    reply_root_eligible,
    serialize_reply_example,
    split_artifact,
    validate_reply_example,
)
from games.kingdomino.tests.test_denial_search import _ConstantEvaluator, _round_start


def _eligible_state(start_seed: int = 300):
    for seed in range(start_seed, start_seed + 1000):
        state = _round_start(seed)
        if reply_root_eligible(state):
            return state
    raise AssertionError("could not construct eligible reply root")


def _example(tmp_path: Path, *, position_index: int, seed: int):
    state = _eligible_state(seed)
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    search = DenialSearch(
        _ConstantEvaluator(), checkpoint_path=str(checkpoint),
        config=SearchConfig(pick_plies=2, chance_k=1, root_search_sims=0,
                            placement_top_k=2),
    )
    root = search.search_position(state)
    label = search.extract_reply_labels(state, root_label=root)[0]
    return serialize_reply_example(
        label,
        position_index=position_index,
        root_state_key=public_state_key(state),
        source={"seed": seed},
        calibration=False,
        min_top_two_margin=0.0,
        max_mc_standard_error=1.0,
        max_target_entropy=10.0,
        reject_ties=False,
    )


def test_training_root_filter_excludes_held_out_and_duplicate_states():
    held_out = _eligible_state(300)
    candidate = _eligible_state(500)
    assert public_state_key(candidate) != public_state_key(held_out)

    blocked = {public_state_key(held_out): "held-out.jsonl"}
    position_filter, stats = _training_root_filter(blocked)

    assert not position_filter(held_out)
    assert position_filter(candidate)
    assert not position_filter(candidate.copy())
    assert stats == {
        "held_out_candidates_skipped": 1,
        "duplicate_candidates_skipped": 1,
    }

    _validate_training_roots([(candidate, {})], blocked)
    with pytest.raises(ValueError, match="leakage"):
        _validate_training_roots([(held_out, {})], blocked)
    with pytest.raises(ValueError, match="duplicate"):
        _validate_training_roots([(candidate, {}), (candidate.copy(), {})], blocked)


def test_reply_example_roundtrip_is_self_contained_and_valid(tmp_path):
    row = _example(tmp_path, position_index=0, seed=300)

    validate_reply_example(row)
    assert row["quality_accept"]
    assert decode_array_blob(row["encoded_state"]["flat"]).ndim == 1
    assert row["actor"] != row["root_actor"]

    corrupt = copy.deepcopy(row)
    corrupt["denial_policy_target"][0] += 0.25
    with pytest.raises(ValueError, match="normalized"):
        validate_reply_example(corrupt)


def test_calibration_examples_are_never_marked_accepted(tmp_path):
    state = _eligible_state(500)
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    search = DenialSearch(
        _ConstantEvaluator(), checkpoint_path=str(checkpoint),
        config=SearchConfig(pick_plies=2, chance_k=1, root_search_sims=0),
    )
    root = search.search_position(state)
    label = search.extract_reply_labels(state, root_label=root)[0]
    row = serialize_reply_example(
        label, position_index=0, root_state_key=public_state_key(state), source={},
        calibration=True, min_top_two_margin=None,
        max_mc_standard_error=None, max_target_entropy=None, reject_ties=False,
    )
    assert not row["quality_accept"]
    assert row["quality_rejection_reasons"] == ["calibration_only"]


def test_two_shard_merge_is_canonical_and_provenance_checked(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    examples = [
        _example(tmp_path, position_index=0, seed=700),
        _example(tmp_path, position_index=1, seed=900),
    ]
    for shard_index in (0, 1):
        stem = f"shard-{shard_index:04d}-of-0002"
        data_path = shard_dir / f"{stem}.jsonl"
        data_path.write_text(json.dumps({
            "position_index": shard_index,
            "root_state_key": examples[shard_index]["root_state_key"],
            "examples": [examples[shard_index]],
        }, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "kind": "kingdomino_opponent_reply_shard",
            "positions_sha256": "positions",
            "checkpoint_sha256": "checkpoint",
            "implementation_sha256": "implementation",
            "num_shards": 2,
            "shard_index": shard_index,
            "complete": True,
            "output": str(data_path),
            "output_sha256": file_sha256(data_path),
        }
        (shard_dir / f"{stem}.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")

    output = tmp_path / "merged.jsonl"
    result = merge_shards(Namespace(
        shards_dir=str(shard_dir), output=str(output), accepted_only=False))
    merged = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["position_index"] for row in merged] == [0, 1]
    assert result["reply_examples"] == 2
    assert result["output_sha256"] == file_sha256(output)

    bad = json.loads((shard_dir / "shard-0001-of-0002.manifest.json").read_text())
    bad["checkpoint_sha256"] = "different"
    (shard_dir / "shard-0001-of-0002.manifest.json").write_text(
        json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        merge_shards(Namespace(
            shards_dir=str(shard_dir), output=str(output), accepted_only=False))


def test_split_keeps_root_positions_disjoint(tmp_path):
    rows = [
        _example(tmp_path, position_index=0, seed=1100),
        _example(tmp_path, position_index=1, seed=1300),
    ]
    source = tmp_path / "merged.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"

    result = split_artifact(Namespace(
        input=str(source), train_output=str(train), validation_output=str(validation),
        validation_fraction=0.5, split_seed=77,
    ))

    train_rows = [json.loads(line) for line in train.read_text().splitlines()]
    validation_rows = [json.loads(line) for line in validation.read_text().splitlines()]
    assert {row["root_state_key"] for row in train_rows}.isdisjoint(
        {row["root_state_key"] for row in validation_rows})
    assert result["root_state_overlap"] == 0
    assert len(train_rows) == len(validation_rows) == 1
