"""Opponent pick disagreements with current_best across the frozen BGA corpus.

This is a descriptive, pick-only audit.  It reconstructs every clean single-pick
decision made by the strong-human opponent, groups the raw network policy over
placements by claimed tile, and joins the decision to the recorded final score.
It does not attribute an outcome to the move and it does not measure reveal luck.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from games.kingdomino.action_codec import encode_action
from games.kingdomino.bga_denial_anchor import _claims_by, _hero_player_index
from games.kingdomino.bga_reanalysis_corpus import (
    DEFAULT_MANIFEST,
    _load_manifest,
    _resolve_game_path,
    _selected_games,
    _verify_sha256,
)
from games.kingdomino.denial_search import AZBatchEvaluator, _pick_key, load_checkpoint_network
from games.kingdomino.promotion import DEFAULT_CURRENT_BEST
from games.kingdomino.web_app import state_from_debug_json


SCHEMA = "kingdomino-bga-opponent-pick-disagreement/v1"
SUMMARY_SCHEMA = "kingdomino-bga-opponent-pick-disagreement-summary/v1"
DEFAULT_OUTPUT = Path(
    "runs/kingdomino/placement_audit/bga_opponent_pick_disagreements_v1.jsonl"
)
DEFAULT_SUMMARY = Path(
    "runs/kingdomino/placement_audit/bga_opponent_pick_disagreements_summary_v1.json"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def claim_round(deck_count: int) -> int | None:
    """Claimed-tile round in a 12-round Mighty Duel game (1 through 12)."""
    deck_count = int(deck_count)
    if deck_count < 0 or deck_count > 44 or deck_count % 4:
        return None
    return 12 - deck_count // 4


def rank_pick_policy(
    by_pick: dict[int, float], human_pick: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if human_pick not in by_pick:
        raise ValueError(f"human pick {human_pick} missing from legal pick policy")
    ranked = sorted(by_pick.items(), key=lambda item: (-float(item[1]), int(item[0])))
    policy = [
        {"rank": rank, "domino_id": int(domino_id), "probability": float(probability)}
        for rank, (domino_id, probability) in enumerate(ranked, start=1)
    ]
    human = next(item for item in policy if item["domino_id"] == int(human_pick))
    model = policy[0]
    human_probability = float(human["probability"])
    model_probability = float(model["probability"])
    ratio = model_probability / max(human_probability, 1e-15)
    return policy, {
        "human_pick_domino_id": int(human_pick),
        "model_pick_domino_id": int(model["domino_id"]),
        "human_pick_rank": int(human["rank"]),
        "human_pick_probability": human_probability,
        "model_pick_probability": model_probability,
        "policy_probability_gap": model_probability - human_probability,
        "model_to_human_probability_ratio": ratio,
        "log_model_to_human_probability_ratio": math.log(ratio),
        "disagreement": int(human["rank"]) > 1,
    }


def game_outcome(
    players: dict[str, dict[str, Any]], hero_id: str, opponent_id: str
) -> dict[str, Any]:
    hero = players[str(hero_id)]
    opponent = players[str(opponent_id)]
    hero_score = int(hero["score"])
    opponent_score = int(opponent["score"])
    margin = opponent_score - hero_score
    result = "win" if margin > 0 else "loss" if margin < 0 else "tie"
    return {
        "opponent_name": opponent.get("name"),
        "viewer_name": hero.get("name"),
        "opponent_final_score": opponent_score,
        "viewer_final_score": hero_score,
        "opponent_score_margin": margin,
        "opponent_result": result,
        "winner_name": (
            opponent.get("name") if margin > 0 else hero.get("name") if margin < 0 else None
        ),
    }


def _pick_events(decisions: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Yield clean single-pick transitions with their source decision index."""
    for source_index in range(len(decisions) - 1):
        source_record = decisions[source_index]
        source = source_record.get("state", {})
        target = decisions[source_index + 1].get("state", {})
        actor = source.get("current_actor")
        if actor is None:
            continue
        actor = int(actor)
        available = {int(value) for value in source.get("current_row", [])}
        before = _claims_by(source).get(actor, set())
        after = _claims_by(target).get(actor, set())
        newly_claimed = sorted((after - before) & available)
        if len(newly_claimed) != 1:
            continue
        yield {
            "source_decision_index": source_index,
            "source_record": source_record,
            "actor": actor,
            "human_pick_domino_id": newly_claimed[0],
        }


def _current_domino_id(state: Any) -> int | None:
    if not state.pending_claims or state.actor_index >= len(state.pending_claims):
        return None
    return int(state.pending_claims[state.actor_index].domino_id)


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    disagreements = [row for row in rows if row["disagreement"]]
    rank_counts = Counter(int(row["human_pick_rank"]) for row in rows)
    result_counts = Counter(str(row["opponent_result"]) for row in rows)
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_game[str(row["table_id"])].append(row)
    games: list[dict[str, Any]] = []
    for table_id, values in sorted(by_game.items()):
        game_disagreements = [row for row in values if row["disagreement"]]
        first = values[0]
        games.append(
            {
                "table_id": table_id,
                "split": first["split"],
                "opponent_name": first["opponent_name"],
                "opponent_final_score": first["opponent_final_score"],
                "viewer_final_score": first["viewer_final_score"],
                "opponent_score_margin": first["opponent_score_margin"],
                "opponent_result": first["opponent_result"],
                "clean_picks": len(values),
                "disagreements": len(game_disagreements),
                "rank_3_or_4": sum(row["human_pick_rank"] >= 3 for row in game_disagreements),
                "human_probability_under_005": sum(
                    row["human_pick_probability"] < 0.05 for row in game_disagreements
                ),
                "aggregate_log_probability_ratio": sum(
                    float(row["log_model_to_human_probability_ratio"])
                    for row in game_disagreements
                ),
                "maximum_probability_ratio": max(
                    (
                        float(row["model_to_human_probability_ratio"])
                        for row in game_disagreements
                    ),
                    default=1.0,
                ),
            }
        )
    games.sort(
        key=lambda game: (
            -float(game["aggregate_log_probability_ratio"]), str(game["table_id"])
        )
    )
    top_moves = sorted(
        disagreements,
        key=lambda row: (
            -float(row["log_model_to_human_probability_ratio"]),
            str(row["table_id"]),
            int(row["source_decision_index"]),
        ),
    )[:30]
    top_move_fields = (
        "table_id",
        "split",
        "opponent_name",
        "source_decision_index",
        "claim_round",
        "deck_count",
        "available_pick_domino_ids",
        "current_domino_id",
        "human_pick_domino_id",
        "model_pick_domino_id",
        "human_pick_rank",
        "human_pick_probability",
        "model_pick_probability",
        "model_to_human_probability_ratio",
        "opponent_final_score",
        "viewer_final_score",
        "opponent_score_margin",
        "opponent_result",
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "games": len(games),
        "clean_opponent_picks": len(rows),
        "disagreements": len(disagreements),
        "disagreement_fraction": len(disagreements) / len(rows),
        "rank_counts": _counter_dict(rank_counts),
        "rank_3_or_4": sum(row["human_pick_rank"] >= 3 for row in rows),
        "human_probability_under_005": sum(
            row["human_pick_probability"] < 0.05 for row in rows
        ),
        "opponent_results_by_pick": _counter_dict(result_counts),
        "games_ranked_by_aggregate_disagreement": games,
        "top_30_disagreement_moves": [
            {field: row[field] for field in top_move_fields} for row in top_moves
        ],
    }


def run_audit(
    *,
    manifest_path: Path,
    checkpoint: Path,
    output_path: Path,
    summary_path: Path,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _load_manifest(manifest_path)
    prepared: list[dict[str, Any]] = []
    for game in _selected_games(manifest, "all"):
        table_id = str(game["table_id"])
        game_path = _resolve_game_path(manifest_path, str(game["path"]))
        _verify_sha256(game_path, str(game["sha256"]))
        records = _read_jsonl(game_path)
        decisions = [record for record in records if record.get("kind") == "decision"]
        final = next((record for record in records if record.get("kind") == "final"), None)
        if not decisions or final is None:
            raise ValueError(f"frozen game {table_id} lacks decisions or final record")
        players = final["final"]["players"]
        hero_id = str(decisions[0].get("viewer_id"))
        opponent_id = next((str(player_id) for player_id in players if str(player_id) != hero_id), None)
        if opponent_id is None:
            raise ValueError(f"could not identify opponent in game {table_id}")
        actor_by_player = _hero_player_index(decisions, hero_id)
        opponent_actor = actor_by_player.get(opponent_id)
        if opponent_actor is None:
            raise ValueError(f"could not identify opponent actor in game {table_id}")
        outcome = game_outcome(players, hero_id, opponent_id)
        for event in _pick_events(decisions):
            if int(event["actor"]) != int(opponent_actor):
                continue
            record = event["source_record"]
            try:
                state = state_from_debug_json(record["state"])
            except Exception:
                continue
            prepared.append(
                {
                    "table_id": table_id,
                    "split": str(game["split"]),
                    "source_path": str(game["path"]),
                    "source_sha256": str(game["sha256"]),
                    "source_decision_index": int(event["source_decision_index"]),
                    "captured_at": record.get("captured_at"),
                    "opponent_player_id": opponent_id,
                    "viewer_player_id": hero_id,
                    "opponent_actor": int(opponent_actor),
                    "state": state,
                    "human_pick_domino_id": int(event["human_pick_domino_id"]),
                    "outcome": outcome,
                }
            )

    net, config = load_checkpoint_network(checkpoint, device)
    evaluator = AZBatchEvaluator(
        net,
        device=device,
        batch_size=256,
        margin_gain=float(config.get("margin_gain", 2.0)),
        alpha=float(config.get("alpha", 0.5)),
    )
    evaluator.policies(item["state"] for item in prepared)

    rows: list[dict[str, Any]] = []
    for item in prepared:
        state = item.pop("state")
        by_pick: dict[int, float] = defaultdict(float)
        policy = evaluator.policy(state)
        for action in state.legal_actions():
            pick_id = _pick_key(action)
            if pick_id is None:
                continue
            by_pick[int(pick_id)] += float(policy[int(encode_action(action, state))])
        pick_policy, comparison = rank_pick_policy(
            dict(by_pick), int(item["human_pick_domino_id"])
        )
        deck_count = len(state.deck)
        row = {
            "schema": SCHEMA,
            **{key: value for key, value in item.items() if key != "outcome"},
            "phase": state.phase.name,
            "deck_count": deck_count,
            "claim_round": claim_round(deck_count),
            "current_domino_id": _current_domino_id(state),
            "available_pick_domino_ids": sorted(entry["domino_id"] for entry in pick_policy),
            "pick_policy": pick_policy,
            **comparison,
            **item["outcome"],
        }
        rows.append(row)
    rows.sort(key=lambda row: (str(row["table_id"]), int(row["source_decision_index"])))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    summary = summarize(rows)
    summary.update(
        {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_config": config,
            "output": str(output_path),
            "output_sha256": _sha256(output_path),
            "scope": {
                "players": "strong-human opponents only",
                "decision": "clean single-tile picks only",
                "model": "raw current_best policy grouped by pick tile",
                "outcome_use": "joined after disagreement measurement; no causal attribution",
                "luck_attribution": False,
            },
        }
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CURRENT_BEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    _rows, summary = run_audit(
        manifest_path=args.manifest,
        checkpoint=args.checkpoint,
        output_path=args.output,
        summary_path=args.summary,
        device=args.device,
    )
    print(json.dumps({key: summary[key] for key in (
        "games", "clean_opponent_picks", "disagreements", "disagreement_fraction",
        "rank_counts", "rank_3_or_4", "human_probability_under_005", "output_sha256",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
