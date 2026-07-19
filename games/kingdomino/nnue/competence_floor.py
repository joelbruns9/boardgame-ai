"""P0 step 2: competence floor for the pilot NNUE curriculum generator."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from games.kingdomino.bots import GreedyBot, RandomBot
from games.kingdomino.game import GameState, Phase, TurnAction, determine_winner
from games.kingdomino.nnue.match import _sha256, nnue_participant, run_paired, write_report
from games.kingdomino.promotion import wilson_lower_bound
from games.kingdomino.round_robin_eval import (
    GameResult,
    Participant,
    build_open_loop_checkpoint_participant,
    score_total,
)
from games.kingdomino.rust_expectiminimax import OperationalRustSearchBot


DEFAULT_NNUE = Path("runs/kingdomino/nnue_data/sparse_v3_pilot.knnue")
DEFAULT_MID_AZ = Path("runs/kingdomino/cloud_80x6_run1/iter_0020.pt")
DEFAULT_OUT = Path("runs/kingdomino/nnue_loop/competence_floor_p0_step2.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _round_index(state) -> int:
    if state.phase == Phase.INITIAL_SELECTION:
        return 0
    if state.phase in (Phase.FINAL_PLACEMENT, Phase.GAME_OVER):
        return 12
    # After the opening selection, the first placement round has 40 hidden
    # tiles remaining. Each subsequent public row consumes four more.
    return max(1, min(11, 1 + (40 - len(state.deck)) // 4))


def _progress_bin(round_index: int) -> str:
    if round_index <= 4:
        return "opening"
    if round_index <= 8:
        return "midgame"
    return "endgame"


def _state_fingerprint(state: GameState) -> tuple:
    boards = tuple(
        (board.terrain.tobytes(), board.crowns.tobytes(), board.domino_id.tobytes())
        for board in state.boards
    )
    return (
        int(state.phase),
        state.actor_index,
        state.initial_pick_count,
        state.start_player,
        tuple(state.deck),
        tuple(state.current_row),
        tuple((claim.player, claim.domino_id) for claim in state.pending_claims),
        tuple((claim.player, claim.domino_id) for claim in state.next_claims),
        tuple(state.discards),
        tuple(state.history),
        boards,
    )


def _distribution(values: list[int]) -> dict:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
            "histogram": {},
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": int(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "max": int(array.max()),
        "histogram": {
            str(value): count for value, count in sorted(Counter(values).items())
        },
    }


class CoherenceAggregator:
    """Observer for the existing match loop; it never chooses or changes moves."""

    def __init__(self, nnue_name: str = "NNUE"):
        self.nnue_name = nnue_name
        self.current = None
        self.completed_games = 0
        self.aborted_games = 0
        self.successful_action_mappings = 0
        self.failure_counts = Counter()
        self.failure_records = []
        self.own_scores: list[int] = []
        self.opp_scores: list[int] = []
        self.phase_positions = Counter()
        self.phase_game_coverage = Counter()
        self.round_positions = Counter()
        self.round_game_coverage = Counter()
        self.progress_positions = Counter()
        self.progress_game_coverage = Counter()
        self.nnue_decision_phases = Counter()
        self.nnue_decision_rounds = Counter()
        self.nnue_decisions = 0
        self.nnue_placement_decisions = 0
        self.all_placement_decisions = 0
        self.nnue_discard_events = []
        self.all_discard_events = []
        self.discard_counter_mismatches = 0
        self.replay_mismatch_records = []

    def on_game_start(self, seed, p0_name, p1_name, state) -> None:
        if self.current is not None:
            raise RuntimeError("observer received a new game before the prior game ended")
        if self.nnue_name not in (p0_name, p1_name):
            raise ValueError("competence observer requires NNUE in every game")
        self.current = {
            "seed": seed,
            "p0": p0_name,
            "p1": p1_name,
            "nnue_player": 0 if p0_name == self.nnue_name else 1,
            "actions": [],
            "phases": set(),
            "rounds": set(),
            "progress": set(),
            "nnue_discards": 0,
            "all_discards": [0, 0],
        }

    def _record_position(self, state, actor_name: str | None) -> None:
        phase = state.phase.name
        round_index = _round_index(state)
        progress = _progress_bin(round_index)
        self.phase_positions[phase] += 1
        self.round_positions[round_index] += 1
        self.progress_positions[progress] += 1
        if self.current is not None:
            self.current["phases"].add(phase)
            self.current["rounds"].add(round_index)
            self.current["progress"].add(progress)
        if actor_name == self.nnue_name:
            self.nnue_decision_phases[phase] += 1
            self.nnue_decision_rounds[round_index] += 1

    def on_position(self, state, actor_name) -> None:
        self._record_position(state, actor_name)

    def on_action(self, state, actor_name, action, legal_actions) -> None:
        if self.current is None:
            raise RuntimeError("action observed outside a game")
        self.current["actions"].append(action)
        is_nnue = actor_name == self.nnue_name
        if is_nnue:
            self.nnue_decisions += 1
        if isinstance(action, TurnAction):
            self.all_placement_decisions += 1
            if is_nnue:
                self.nnue_placement_decisions += 1
            if action.placement is None:
                event = {
                    "seed": self.current["seed"],
                    "game_key": (
                        f"{self.current['seed']}:{self.current['p0']}:{self.current['p1']}"
                    ),
                    "player": int(state.current_actor),
                    "actor": actor_name,
                    "phase": state.phase.name,
                    "round": _round_index(state),
                    "progress": _progress_bin(_round_index(state)),
                    "ply": len(state.history),
                    "deck_remaining": len(state.deck),
                }
                self.all_discard_events.append(event)
                self.current["all_discards"][state.current_actor] += 1
                if is_nnue:
                    self.nnue_discard_events.append(event)
                    self.current["nnue_discards"] += 1

    def on_transition(self, state, next_state, actor_name, action) -> None:
        return None

    def on_action_mapping(self, state, actor_name, target_index) -> None:
        self.successful_action_mappings += 1

    def on_failure(self, kind, state, actor_name, exc) -> None:
        self.failure_counts[kind] += 1
        message = str(exc).lower()
        if actor_name == self.nnue_name and (
            kind == "illegal_action"
            or (kind == "action_selection" and ("illegal" in message or "map" in message))
        ):
            self.failure_counts["operational_action_mapping"] += 1
        self.failure_records.append({
            "kind": kind,
            "actor": actor_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "seed": None if self.current is None else self.current["seed"],
            "phase": state.phase.name,
            "round": _round_index(state),
            "ply": len(state.history),
        })

    def on_game_aborted(self, seed, p0_name, p1_name, exc) -> None:
        self.aborted_games += 1
        self.current = None

    def _record_replay_mismatch(self, reason: str) -> None:
        self.failure_counts["replay_mismatch"] += 1
        self.replay_mismatch_records.append({
            "seed": self.current["seed"],
            "p0": self.current["p0"],
            "p1": self.current["p1"],
            "reason": reason,
        })

    def on_game_end(self, state: GameState, result: GameResult) -> None:
        if self.current is None:
            raise RuntimeError("game end observed without a current game")
        self._record_position(state, None)
        nnue_player = self.current["nnue_player"]
        scores = (result.score0, result.score1)
        self.own_scores.append(scores[nnue_player])
        self.opp_scores.append(scores[1 - nnue_player])
        if self.current["all_discards"] != list(state.discards):
            self.discard_counter_mismatches += 1
            self.failure_counts["discard_counter_mismatch"] += 1

        try:
            replay = GameState.new(seed=self.current["seed"])
            for action in self.current["actions"]:
                if action not in replay.legal_actions():
                    raise ValueError("recorded action is not legal during replay")
                replay = replay.step(action)
            if _state_fingerprint(replay) != _state_fingerprint(state):
                self._record_replay_mismatch("terminal state fingerprint differs")
            else:
                replay_scores = tuple(score_total(board) for board in replay.boards)
                replay_winner_index = determine_winner(replay)
                replay_winner = (
                    None
                    if replay_winner_index is None
                    else (result.p0, result.p1)[replay_winner_index]
                )
                if replay_scores != scores or replay_winner != result.winner:
                    self._record_replay_mismatch("terminal score/winner differs")
        except Exception as exc:
            self._record_replay_mismatch(f"{type(exc).__name__}: {exc}")

        for phase in self.current["phases"]:
            self.phase_game_coverage[phase] += 1
        for round_index in self.current["rounds"]:
            self.round_game_coverage[round_index] += 1
        for progress in self.current["progress"]:
            self.progress_game_coverage[progress] += 1
        self.completed_games += 1
        self.current = None

    def summary(self) -> dict:
        primary_failures = {
            key: int(self.failure_counts.get(key, 0))
            for key in (
                "illegal_action",
                "action_selection",
                "action_resolution",
                "replay_mismatch",
                "discard_counter_mismatch",
            )
        }
        return {
            "completed_games": self.completed_games,
            "aborted_games": self.aborted_games,
            "legality_replay_integrity": {
                **primary_failures,
                "operational_action_mapping_failures": int(
                    self.failure_counts.get("operational_action_mapping", 0)
                ),
                "successful_canonical_action_mappings": self.successful_action_mappings,
                "total_failures": sum(primary_failures.values()),
                "failure_records": self.failure_records,
                "replay_mismatch_records": self.replay_mismatch_records,
            },
            "scores": {
                "nnue_own": _distribution(self.own_scores),
                "opponent": _distribution(self.opp_scores),
                "margin": _distribution(
                    [own - opp for own, opp in zip(self.own_scores, self.opp_scores)]
                ),
            },
            "forced_discards": {
                "nnue_count": len(self.nnue_discard_events),
                "all_players_count": len(self.all_discard_events),
                "nnue_placement_decisions": self.nnue_placement_decisions,
                "nnue_frequency_per_placement": (
                    len(self.nnue_discard_events) / self.nnue_placement_decisions
                    if self.nnue_placement_decisions
                    else 0.0
                ),
                "nnue_games_with_discard": len({
                    event["game_key"] for event in self.nnue_discard_events
                }),
                "nnue_by_phase": dict(Counter(
                    event["phase"] for event in self.nnue_discard_events
                )),
                "nnue_by_round": {
                    str(key): value for key, value in sorted(Counter(
                        event["round"] for event in self.nnue_discard_events
                    ).items())
                },
                "nnue_by_progress": dict(Counter(
                    event["progress"] for event in self.nnue_discard_events
                )),
                "nnue_events": self.nnue_discard_events,
            },
            "phase_coverage": {
                "position_counts": dict(self.phase_positions),
                "games_reaching_phase": dict(self.phase_game_coverage),
                "nnue_decision_counts": dict(self.nnue_decision_phases),
            },
            "round_coverage": {
                "position_counts": {
                    str(key): value for key, value in sorted(self.round_positions.items())
                },
                "games_reaching_round": {
                    str(key): value for key, value in sorted(self.round_game_coverage.items())
                },
                "nnue_decision_counts": {
                    str(key): value for key, value in sorted(self.nnue_decision_rounds.items())
                },
            },
            "progress_coverage": {
                "position_counts": dict(self.progress_positions),
                "games_reaching_progress": dict(self.progress_game_coverage),
            },
        }


def _wilson_upper(points: float, games: int) -> float:
    return 1.0 - wilson_lower_bound(games - points, games)


def _series_summary(report: dict, paired_seeds: int, nnue_budget: float) -> dict:
    pair = report["pair"]
    points = pair["a_wins"] + 0.5 * pair["draws"]
    games = pair["games"]
    return {
        "paired_seeds": paired_seeds,
        "requested_games": report["requested_games"],
        "completed_games": report["completed_games"],
        "failed_games": report["failed_games"],
        "nnue_wins": pair["a_wins"],
        "draws": pair["draws"],
        "nnue_losses": pair["b_wins"],
        "nnue_points_rate": pair["a_points_rate"],
        "nnue_points_lcb_95": pair["a_points_lcb_95"],
        "nnue_points_ucb_95": _wilson_upper(points, games),
        "avg_margin_nnue": pair["avg_margin_a"],
        "nnue_generation_budget_seconds": nnue_budget,
        "nnue_timing": report["timing"]["NNUE"],
        "opponent_timing": next(
            timing for name, timing in report["timing"].items() if name != "NNUE"
        ),
    }


def _verdict(series: dict, coherence: dict) -> dict:
    policy = {
        "random_clear_lcb_min": 0.50,
        "greedy_clear_lcb_min": 0.50,
        "pick_aware_competitive_points_min": 0.45,
        "pick_aware_competitive_margin_min": -10.0,
        "mid_az_respectable_points_min": 0.25,
        "mid_az_respectable_margin_min": -25.0,
        "integrity_failures_required": 0,
        "required_progress_bins": ["opening", "midgame", "endgame"],
    }
    integrity = coherence["legality_replay_integrity"]["total_failures"] == 0
    complete = all(item["completed_games"] == item["requested_games"] for item in series.values())
    coverage = coherence["progress_coverage"]["games_reaching_progress"]
    broad = all(coverage.get(name, 0) == coherence["completed_games"] for name in policy["required_progress_bins"])
    random_clear = series["random"]["nnue_points_lcb_95"] > policy["random_clear_lcb_min"]
    greedy_clear = series["greedy"]["nnue_points_lcb_95"] > policy["greedy_clear_lcb_min"]
    pick_competitive = (
        series["pick_aware"]["nnue_points_rate"] >= policy["pick_aware_competitive_points_min"]
        and series["pick_aware"]["avg_margin_nnue"] >= policy["pick_aware_competitive_margin_min"]
    )
    mid_respectable = (
        series["mid_az"]["nnue_points_rate"] >= policy["mid_az_respectable_points_min"]
        and series["mid_az"]["avg_margin_nnue"] >= policy["mid_az_respectable_margin_min"]
    )
    checks = {
        "integrity": integrity,
        "all_requested_games_completed": complete,
        "broad_phase_coverage": broad,
        "clearly_beats_random": random_clear,
        "clearly_beats_greedy": greedy_clear,
        "competitive_with_pick_aware": pick_competitive,
        "respectable_against_mid_az": mid_respectable,
    }
    if all(checks.values()):
        classification = "competent_generator"
        fork = "Run P0 step 0 cascade alignment, then P0 step 3 fresh replayable AZ trajectories."
    elif (
        not integrity
        or not complete
        or not broad
        or series["random"]["nnue_points_rate"] <= 0.50
        or series["greedy"]["nnue_points_rate"] <= 0.50
        or series["pick_aware"]["nnue_points_rate"] < policy["pick_aware_competitive_points_min"]
    ):
        classification = "not_generation_ready"
        fork = (
            "Strengthen the pilot generator before P0 step 3; treat this as a "
            "training/evaluation-quality problem, not stochastic search allocation."
        )
    else:
        classification = "borderline"
        fork = (
            "Do not spend teacher compute yet; confirm the failed competence check "
            "or strengthen the pilot before P0 step 3."
        )
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "classification": classification,
        "checks": checks,
        "failed_checks": failed,
        "decision_policy": policy,
        "reasoning": (
            "All competence checks passed."
            if not failed
            else "Failed checks: " + ", ".join(failed) + "."
        ),
        "recommended_fork": fork,
        "do_not_act_in_this_step": True,
    }


def _pick_aware_participant(args) -> Participant:
    def make_bot():
        return OperationalRustSearchBot(
            max_secs=args.nnue_move_secs,
            max_depth=args.max_depth,
            chance_samples=args.chance_samples,
            eval="pick_aware",
            full_width_ordering=True,
            selective_width=None,
            selective_root_width=None,
        )

    return Participant("PickAware", make_bot, kind="operational_pick_aware")


def run_competence(args: argparse.Namespace) -> dict:
    if args.paired_seeds < 1:
        raise ValueError("paired seeds must be >= 1")
    if args.nnue_move_secs <= 0.0 or not math.isfinite(args.nnue_move_secs):
        raise ValueError("NNUE move seconds must be finite and positive")
    nnue_path = Path(args.nnue).resolve()
    mid_az_path = Path(args.mid_az_checkpoint).resolve()
    if not nnue_path.is_file() or not mid_az_path.is_file():
        raise FileNotFoundError("pilot NNUE and mid-strength AZ checkpoint must exist")

    nnue_hash = _sha256(nnue_path)
    if args.expected_nnue_sha256 and nnue_hash != args.expected_nnue_sha256:
        raise ValueError("pilot NNUE hash differs from the completed Step 1 artifact")
    mid_az_hash = _sha256(mid_az_path)
    rating_source = Path(args.mid_az_rating_source).resolve()
    if not rating_source.is_file():
        raise FileNotFoundError("mid-strength AZ rating source must exist")
    rating_source_hash = _sha256(rating_source)
    observer = CoherenceAggregator("NNUE")
    nnue = lambda: nnue_participant(
        "NNUE",
        nnue_path,
        move_secs=args.nnue_move_secs,
        max_depth=args.max_depth,
        chance_samples=args.chance_samples,
        full_width_ordering=True,
        selective_width=None,
        selective_root_width=None,
    )
    opponents = [
        ("random", Participant("Random", lambda: RandomBot(), kind="baseline_random"), {}),
        ("greedy", Participant("Greedy", lambda: GreedyBot(), kind="baseline_greedy"), {}),
        ("pick_aware", _pick_aware_participant(args), {
            "move_secs": args.nnue_move_secs,
            "max_depth": args.max_depth,
            "chance_samples": args.chance_samples,
            "full_width_ordering": True,
        }),
        ("mid_az", build_open_loop_checkpoint_participant(
            str(mid_az_path),
            name="MidAZ",
            device=args.device,
            sims=args.mid_az_sims,
            c_puct=args.c_puct,
            temperature=0.0,
            channels_override=None,
            blocks_override=None,
            bilinear_dim_override=None,
        ), {
            "checkpoint": str(mid_az_path),
            "checkpoint_sha256": mid_az_hash,
            "sims": args.mid_az_sims,
            "historical_rating": args.mid_az_historical_rating,
            "historical_rating_stderr": args.mid_az_historical_rating_stderr,
            "rating_source": str(rating_source),
            "rating_source_sha256": rating_source_hash,
        }),
    ]

    detailed = {}
    summaries = {}
    for key, opponent, opponent_settings in opponents:
        print(
            f"{key}: {args.paired_seeds} paired seeds, NNUE {args.nnue_move_secs:g}s/move",
            flush=True,
        )
        settings = {
            "kind": "nnue_competence_floor_p0_step2",
            "set_tag": "screening",
            "opponent_key": key,
            "opponent_kind": opponent.kind,
            "opponent_settings": opponent_settings,
            "seed_start": args.seed_start,
            "paired_seeds": args.paired_seeds,
            "nnue": str(nnue_path),
            "nnue_sha256": nnue_hash,
            "nnue_move_secs": args.nnue_move_secs,
            "max_depth": args.max_depth,
            "chance_samples": args.chance_samples,
            "full_width_ordering": True,
            "selective_width": None,
        }
        match = run_paired(
            nnue(),
            opponent,
            seed_start=args.seed_start,
            paired_seeds=args.paired_seeds,
            settings=settings,
            observer=observer,
            continue_on_game_error=True,
        )
        detailed[key] = match
        summaries[key] = _series_summary(match, args.paired_seeds, args.nnue_move_secs)
        item = summaries[key]
        print(
            f"  points={item['nnue_points_rate']:.3f} "
            f"[{item['nnue_points_lcb_95']:.3f}, {item['nnue_points_ucb_95']:.3f}] "
            f"margin={item['avg_margin_nnue']:+.2f} failures={len(item['failed_games'])}",
            flush=True,
        )

    coherence = observer.summary()
    verdict = _verdict(summaries, coherence)
    return {
        "schema": "kingdomino_nnue_competence_floor_p0_step2_v1",
        "created_utc": _utc_now(),
        "question": "At a generous offline budget, is the pilot NNUE a competent curriculum generator?",
        "verdict": verdict,
        "settings": {
            "nnue": str(nnue_path),
            "nnue_sha256": nnue_hash,
            "step1_expected_nnue_sha256": args.expected_nnue_sha256,
            "nnue_move_secs": args.nnue_move_secs,
            "max_depth": args.max_depth,
            "chance_samples": args.chance_samples,
            "full_width_ordering": True,
            "selective_width": None,
            "device": args.device,
            "seed_sets": {
                "screening": {
                    "seed_start": args.seed_start,
                    "paired_seeds_per_opponent": args.paired_seeds,
                },
                "confirmation": None,
            },
            "mid_az": opponents[-1][2],
            "reserved_test_split_opened": False,
            "current_best_ceiling_included": False,
            "current_best_ceiling_reason": (
                "Step 1 already established the mature-AZ ceiling; omitted here to focus compute on the competence bar."
            ),
        },
        "opponents": summaries,
        "coherence": coherence,
        "detailed_matches": detailed,
        "result_decides_only": verdict["recommended_fork"],
    }


def _print_summary(report: dict, out: Path) -> None:
    print(f"report: {out}")
    print("opponent    points (95% Wilson)      margin   NNUE mean  opp mean")
    for key, item in report["opponents"].items():
        print(
            f"{key:<11} {item['nnue_points_rate']:.3f} "
            f"[{item['nnue_points_lcb_95']:.3f}, {item['nnue_points_ucb_95']:.3f}] "
            f"{item['avg_margin_nnue']:+8.2f} "
            f"{item['nnue_timing']['decision_mean_seconds']:9.3f}s "
            f"{item['opponent_timing']['decision_mean_seconds']:8.3f}s"
        )
    integrity = report["coherence"]["legality_replay_integrity"]
    print(f"legality/replay failures: {integrity['total_failures']}")
    print(json.dumps(report["verdict"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nnue", default=str(DEFAULT_NNUE))
    parser.add_argument(
        "--expected-nnue-sha256",
        default="55c24ff2bfd7143a43921c7ee13b13d877be63c4c9c66d0971fd4f0cfb468302",
    )
    parser.add_argument("--mid-az-checkpoint", default=str(DEFAULT_MID_AZ))
    parser.add_argument("--mid-az-sims", type=int, default=400)
    parser.add_argument("--mid-az-historical-rating", type=float, default=1267.34)
    parser.add_argument("--mid-az-historical-rating-stderr", type=float, default=33.78)
    parser.add_argument(
        "--mid-az-rating-source",
        default="runs/kingdomino/elo_db_80x6_scratch.json",
    )
    parser.add_argument("--nnue-move-secs", type=float, default=2.0)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--chance-samples", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--paired-seeds", type=int, default=24)
    parser.add_argument("--seed-start", type=int, default=20262000)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    report = run_competence(args)
    out = write_report(report, args.out)
    _print_summary(report, out)


if __name__ == "__main__":
    main()
