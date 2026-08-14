"""Exact case studies for selected late BGA human/model pick disagreements."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from games.kingdomino.action_codec import encode_action
from games.kingdomino.bga_reanalysis_corpus import DEFAULT_CORPUS
from games.kingdomino.dominoes import DOMINOES
from games.kingdomino.web_app import RecommendRequest, recommend, state_from_debug_json


DEFAULT_CASES = (("881648336", 42), ("883162423", 37))
DEFAULT_OUTPUT = Path(
    "runs/kingdomino/placement_audit/bga_disagreement_case_studies_v1.json"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tile(domino_id: int) -> dict[str, Any]:
    domino = DOMINOES[int(domino_id)]
    return {
        "domino_id": int(domino_id),
        "a": {"terrain": domino.a.terrain.name, "crowns": int(domino.a.crowns)},
        "b": {"terrain": domino.b.terrain.name, "crowns": int(domino.b.crowns)},
    }


def _score(board: Any, state: Any) -> dict[str, Any]:
    return asdict(board.score(state.config.harmony, state.config.middle_kingdom)) | {
        "total": board.score(state.config.harmony, state.config.middle_kingdom).total
    }


def analyze_case(position: dict[str, Any], *, exact_max_secs: float) -> dict[str, Any]:
    state = state_from_debug_json(position["state"])
    actor = int(state.current_actor)
    human_action = position["human_action"]
    response = recommend(
        RecommendRequest(
            engine="exact",
            state=position["state"],
            top_k=100,
            exact_max_secs=float(exact_max_secs),
            exact_threads=0,
            swindle=False,
            seed=0,
        )
    )
    if response.get("engine") != "exact" or not response.get("exact", {}).get("solved"):
        raise RuntimeError(
            f"exact solve unavailable: engine={response.get('engine')} "
            f"reason={response.get('reason')}"
        )
    recommendations = response["recommendations"]
    by_pick: dict[int, list[dict[str, Any]]] = {}
    for rec in recommendations:
        pick = rec.get("pick_domino_id")
        if pick is not None:
            by_pick.setdefault(int(pick), []).append(rec)
    pick_values = []
    for pick, recs in by_pick.items():
        best = max(recs, key=lambda rec: float(rec["exact_margin_pts"]))
        pick_values.append(
            {
                "pick_domino_id": pick,
                "tile": _tile(pick),
                "best_exact_margin_actor": float(best["exact_margin_pts"]),
                "best_action_idx": int(best["action_idx"]),
                "best_placement": best["placement"],
                "legal_joint_actions": len(recs),
            }
        )
    pick_values.sort(key=lambda item: (-item["best_exact_margin_actor"], item["pick_domino_id"]))
    human_rec = next(
        rec
        for rec in recommendations
        if int(rec["action_idx"]) == int(human_action["action_idx"])
    )
    model_raw_pick = int(
        max(
            position["_pick_policy"],
            key=lambda item: float(item["probability"]),
        )["domino_id"]
    )
    human_pick = int(human_action["pick_domino_id"])
    best_margin = float(pick_values[0]["best_exact_margin_actor"])
    human_pick_best = next(item for item in pick_values if item["pick_domino_id"] == human_pick)
    model_pick_best = next(item for item in pick_values if item["pick_domino_id"] == model_raw_pick)
    played_action = next(
        action
        for action in state.legal_actions()
        if int(encode_action(action, state)) == int(human_action["action_idx"])
    )
    after_played = state.step(played_action)
    return {
        "table_id": position["table_id"],
        "source_decision_index": position["source_decision_index"],
        "position_id": position["position_id"],
        "opponent_name": position["actor_name"],
        "actor": actor,
        "deck_count": position["deck_count"],
        "current_domino": _tile(int(position["reconstruction"]["domino_id"])),
        "available_tiles": [_tile(value) for value in sorted(position["legal"]["pick_domino_ids"])],
        "raw_pick_policy": position["_pick_policy"],
        "human_action": human_action,
        "score_before": _score(state.boards[actor], state),
        "score_after_current_placement": _score(after_played.boards[actor], after_played),
        "board_before_pretty": state.boards[actor].pretty(),
        "board_after_current_placement_pretty": after_played.boards[actor].pretty(),
        "exact_root_margin_actor": float(response["root_margin_pts"]),
        "exact_pick_values": pick_values,
        "exact_optimal_pick_domino_id": int(pick_values[0]["pick_domino_id"]),
        "human_played_joint_exact_margin_actor": float(human_rec["exact_margin_pts"]),
        "human_played_joint_regret_points": best_margin - float(human_rec["exact_margin_pts"]),
        "human_pick_best_exact_margin_actor": human_pick_best["best_exact_margin_actor"],
        "human_pick_regret_points": best_margin - human_pick_best["best_exact_margin_actor"],
        "raw_model_pick_domino_id": model_raw_pick,
        "raw_model_pick_best_exact_margin_actor": model_pick_best["best_exact_margin_actor"],
        "raw_model_pick_regret_points": best_margin - model_pick_best["best_exact_margin_actor"],
        "recorded_final_scores": position["diagnostics_only"]["final_scores"],
        "exact_search_ms": int(response["search_ms"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exact-max-secs", type=float, default=600.0)
    parser.add_argument(
        "--case",
        choices=("all", "881648336", "883162423"),
        default="881648336",
    )
    args = parser.parse_args()
    positions = {
        (str(row["table_id"]), int(row["source_decision_index"])): row
        for row in _read_jsonl(args.corpus)
    }
    disagreement_rows = {
        (str(row["table_id"]), int(row["source_decision_index"])): row
        for row in _read_jsonl(
            Path("runs/kingdomino/placement_audit/bga_opponent_pick_disagreements_v1.jsonl")
        )
    }
    output = []
    selected = DEFAULT_CASES if args.case == "all" else tuple(
        key for key in DEFAULT_CASES if key[0] == args.case
    )
    for key in selected:
        position = dict(positions[key])
        position["_pick_policy"] = disagreement_rows[key]["pick_policy"]
        print(f"exact case {key[0]} d{key[1]}...", flush=True)
        output.append(analyze_case(position, exact_max_secs=args.exact_max_secs))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"schema": "kingdomino-bga-disagreement-case-study/v1", "cases": output},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "kingdomino-bga-disagreement-case-study/v1", "cases": output}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
