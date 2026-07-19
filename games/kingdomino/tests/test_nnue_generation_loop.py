from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from games.kingdomino.action_codec import decode_action, encode_action
from games.kingdomino.nnue.datagen import GenConfig, generate
from games.kingdomino.nnue.competence_floor import CoherenceAggregator
from games.kingdomino.nnue.depth_conversion import (
    _fitness_verdict,
    _gap_closure,
    _pairing_summary,
)
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
from games.kingdomino.game import Phase, TurnAction
from games.kingdomino.round_robin_eval import play_game


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


def test_timed_bot_captures_and_aggregates_operational_reports():
    reports = iter([
        SimpleNamespace(
            completed_depth=3,
            timed_out=True,
            elapsed_secs=0.1,
            nodes=90,
            last_iteration_nodes=60,
            chance_nodes=10,
        ),
        SimpleNamespace(
            completed_depth=5,
            timed_out=False,
            elapsed_secs=0.2,
            nodes=180,
            last_iteration_nodes=120,
            chance_nodes=20,
        ),
    ])

    class ReportingBot:
        last_report = None

        def choose_action(self, state, actions=None, rng=None):
            if len(actions) > 1:
                self.last_report = next(reports)
            return actions[0]

    bot = TimedBot(ReportingBot())
    bot.choose_action(_State(3))
    bot.choose_action(_State(1))  # forced call must not duplicate stale telemetry
    bot.choose_action(_State(2))
    search = bot.summary()["search"]

    assert search["report_count"] == 2
    assert search["completed_depth_mean"] == 4.0
    assert search["completed_depth_median"] == 4.0
    assert search["completed_depth_min"] == 3
    assert search["completed_depth_max"] == 5
    assert search["completed_depth_histogram"] == {"3": 1, "5": 1}
    assert search["nodes_mean"] == 135.0
    assert search["last_iteration_nodes_mean"] == 90.0
    assert search["timeout_count"] == 1
    assert search["timeout_rate"] == 0.5
    assert search["chance_nodes_total"] == 30
    assert search["chance_share"] == pytest.approx(30 / 300)


def test_timed_bot_marks_chance_share_unavailable_for_old_report():
    class ReportingBot:
        last_report = None

        def choose_action(self, state, actions=None, rng=None):
            self.last_report = SimpleNamespace(
                completed_depth=2,
                timed_out=True,
                elapsed_secs=0.1,
                nodes=50,
                last_iteration_nodes=40,
            )
            return actions[0]

    bot = TimedBot(ReportingBot())
    bot.choose_action(_State(2))
    search = bot.summary()["search"]
    assert search["chance_nodes_total"] is None
    assert search["chance_share"] is None


def test_competence_coherence_aggregates_replay_phase_score_and_discards():
    observer = CoherenceAggregator()
    play_game("NNUE", _Bot(), "Stub", _Bot(), seed=731, observer=observer)
    summary = observer.summary()

    assert summary["completed_games"] == 1
    assert summary["legality_replay_integrity"]["total_failures"] == 0
    assert summary["scores"]["nnue_own"]["count"] == 1
    assert summary["scores"]["opponent"]["count"] == 1
    assert summary["phase_coverage"]["games_reaching_phase"] == {
        "INITIAL_SELECTION": 1,
        "PLACE_AND_SELECT": 1,
        "FINAL_PLACEMENT": 1,
        "GAME_OVER": 1,
    }
    assert set(summary["round_coverage"]["games_reaching_round"]) == {
        str(round_index) for round_index in range(13)
    }
    assert summary["progress_coverage"]["games_reaching_progress"] == {
        "opening": 1,
        "midgame": 1,
        "endgame": 1,
    }

    # A tiny stub stream isolates forced-discard location and failure counting.
    stub = SimpleNamespace(
        phase=Phase.PLACE_AND_SELECT,
        deck=[],
        current_actor=0,
        history=[None] * 40,
    )
    discarded = TurnAction(placement=None, pick_domino_id=1)
    forced = CoherenceAggregator()
    forced.on_game_start(99, "NNUE", "Stub", stub)
    forced.on_position(stub, "NNUE")
    forced.on_action(stub, "NNUE", discarded, [discarded])
    forced.on_failure("illegal_action", stub, "NNUE", ValueError("stub"))
    forced.on_game_aborted(99, "NNUE", "Stub", ValueError("stub"))
    forced_summary = forced.summary()
    assert forced_summary["forced_discards"]["nnue_count"] == 1
    assert forced_summary["forced_discards"]["nnue_by_round"] == {"11": 1}
    assert forced_summary["forced_discards"]["nnue_by_progress"] == {"endgame": 1}
    assert forced_summary["legality_replay_integrity"]["illegal_action"] == 1


def test_match_loop_maps_canonical_action_to_engine_legal_representative():
    class CanonicalActionBot:
        def choose_action(self, state, actions=None, rng=None):
            for legal in actions:
                decoded = decode_action(encode_action(legal, state), state)
                if decoded not in actions:
                    return decoded
            return actions[0]

    observer = CoherenceAggregator()
    play_game(
        "NNUE", CanonicalActionBot(), "Stub", _Bot(), seed=731, observer=observer
    )
    integrity = observer.summary()["legality_replay_integrity"]
    assert integrity["total_failures"] == 0
    assert integrity["successful_canonical_action_mappings"] > 0


def _fixed_depth_match(depth_a, depth_b, *, wins, losses, draws=0, margin=1.0):
    def timing(depth):
        return {
            "decision_mean_seconds": 0.01,
            "search": {
                "report_count": 10,
                "completed_depth_min": depth,
                "completed_depth_max": depth,
                "completed_depth_histogram": {str(depth): 10},
                "timeout_count": 0,
            },
        }

    games = wins + losses + draws
    return {
        "pair": {
            "a": f"pick_aware_d{depth_a}",
            "b": f"pick_aware_d{depth_b}",
            "games": games,
            "a_wins": wins,
            "b_wins": losses,
            "draws": draws,
            "a_points_rate": (wins + 0.5 * draws) / games,
            "avg_margin_a": margin,
        },
        "timing": {
            f"pick_aware_d{depth_a}": timing(depth_a),
            f"pick_aware_d{depth_b}": timing(depth_b),
        },
        "requested_games": games,
        "completed_games": games,
        "failed_games": [],
    }


def test_depth_conversion_summary_verdict_and_gap_closure():
    core = {
        "d2_vs_d1": _pairing_summary(
            _fixed_depth_match(2, 1, wins=30, losses=18, margin=4.0),
            higher_depth=2,
            lower_depth=1,
            paired_seeds=24,
        ),
        "d3_vs_d2": _pairing_summary(
            _fixed_depth_match(3, 2, wins=30, losses=18, margin=3.0),
            higher_depth=3,
            lower_depth=2,
            paired_seeds=24,
        ),
        "d3_vs_d1": _pairing_summary(
            _fixed_depth_match(3, 1, wins=36, losses=12, margin=8.0),
            higher_depth=3,
            lower_depth=1,
            paired_seeds=24,
        ),
    }
    verdict = _fitness_verdict(core)
    assert verdict["classification"] == "depth_converts"
    assert verdict["fixed_depth_control_valid"] is True
    assert core["d3_vs_d1"]["higher_depth_points_lcb_95"] > 0.5

    closure = _gap_closure(
        {
            "higher_depth_points_rate": 0.25,
            "avg_margin_higher_depth": -40.0,
        },
        {"pilot_points_rate": 0.0, "pilot_avg_margin_vs_az": -80.0},
    )
    assert closure["margin_improvement_points"] == 40.0
    assert closure["fraction_of_step1_margin_deficit_closed"] == 0.5


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
