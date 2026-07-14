from __future__ import annotations

import hashlib

import numpy as np
import pytest

from games.kingdomino.nnue.datagen import GenConfig, generate
from games.kingdomino.nnue.match import TimedBot, nnue_participant
from games.kingdomino.nnue.sparse_data import (
    ARTIFACT_VERSION,
    TARGET_SCHEMA,
    PackedSparseData,
    concatenate_packed,
)
from games.kingdomino.nnue.sparse_encoder import CORE_SIZE, core_schema_hash
from games.kingdomino.nnue.summary_encoder import SUMMARY_SIZE, summary_schema_hash
from games.kingdomino.nnue import datagen


def _part(source: str, outcome: float, actor: int) -> PackedSparseData:
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    meta = {
        "artifact": "kingdomino_sparse_nnue_csr",
        "artifact_version": ARTIFACT_VERSION,
        "core_size": CORE_SIZE,
        "summary_size": SUMMARY_SIZE,
        "core_schema_hash": core_schema_hash(),
        "summary_schema_hash": summary_schema_hash(),
        "datagen_engine_version": datagen.ENGINE_VERSION,
        "datagen_format_version": datagen.FORMAT_VERSION,
        "catalog_hash": datagen.catalog_hash(),
        "rules": {"harmony": True, "middle_kingdom": True},
        "game_count": 1,
        "position_count": 1,
        "source_records_sha256": source_hash,
        "source_git_commits": [source],
        "source_git_dirty": False,
        "source_seeds": [0, 0],
        "target_schema": TARGET_SCHEMA,
        "d4": "base orientation stored; one of 8 frozen permutations applied at batch time",
    }
    return PackedSparseData(
        indices=np.asarray([actor], dtype=np.int32),
        offsets=np.asarray([0, 1], dtype=np.int64),
        summaries=np.zeros((1, SUMMARY_SIZE), dtype=np.float32),
        outcome=np.asarray([outcome], dtype=np.float32),
        margin=np.zeros(1, dtype=np.float32),
        aux_scores=np.zeros((1, 6), dtype=np.float32),
        aux_bonus=np.zeros((1, 4), dtype=np.float32),
        actors=np.asarray([actor], dtype=np.uint8),
        game_index=np.asarray([0], dtype=np.int32),
        metadata=meta,
    )


def test_concatenate_packed_preserves_rows_and_unique_game_ids():
    merged = concatenate_packed([_part("a", 1.0, 0), _part("b", 0.0, 1)])
    assert len(merged) == 2
    assert merged.offsets.tolist() == [0, 1, 2]
    assert merged.game_index.tolist() == [0, 1]
    assert merged.outcome.tolist() == [1.0, 0.0]
    assert merged.metadata["game_count"] == 2
    assert len(merged.metadata["source_components"]) == 2


def test_datagen_requires_artifact_for_nnue(tmp_path):
    with pytest.raises(ValueError, match="requires nnue_path"):
        generate(0, str(tmp_path), GenConfig(eval="sparse_nnue_q"))


class _State:
    def __init__(self, n):
        self._actions = list(range(n))

    def legal_actions(self):
        return self._actions


class _Bot:
    def choose_action(self, state, actions=None, rng=None):
        return actions[0]


def test_timed_bot_separates_forced_and_decision_calls():
    bot = TimedBot(_Bot())
    assert bot.choose_action(_State(1)) == 0
    assert bot.choose_action(_State(3)) == 0
    summary = bot.summary()
    assert summary["forced_count"] == 1
    assert summary["decision_count"] == 1
    assert summary["decision_total_seconds"] >= 0.0


def test_nnue_participant_defaults_to_full_width_ordering(monkeypatch, tmp_path):
    captured = {}

    class FakeSearchBot:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "games.kingdomino.nnue.match.OperationalRustSearchBot", FakeSearchBot
    )
    participant = nnue_participant(
        "ordered",
        tmp_path / "model.knnue",
        move_secs=0.1,
    )
    participant.make_bot()
    assert captured["full_width_ordering"] is True
