"""Exact viewer response if wishiwas takes tile 20 before tile 28."""
from __future__ import annotations

import json
from pathlib import Path

from games.kingdomino.action_codec import encode_action
from games.kingdomino.bga_reanalysis_corpus import DEFAULT_CORPUS
from games.kingdomino.deep_target_screen import _read_jsonl
from games.kingdomino.game import TurnAction
from games.kingdomino.web_app import RecommendRequest, recommend, state_from_debug_json, state_to_debug_json


OUTPUT = Path(
    "runs/kingdomino/placement_audit/bga_wishiwas_counterfactual_response_v1.json"
)


def main() -> int:
    position = next(
        row
        for row in _read_jsonl(DEFAULT_CORPUS)
        if str(row["table_id"]) == "883162423"
        and int(row["source_decision_index"]) == 37
    )
    state = state_from_debug_json(position["state"])
    human = position["human_action"]
    placement = human["placement"]
    counterfactual = next(
        action
        for action in state.legal_actions()
        if isinstance(action, TurnAction)
        and action.pick_domino_id == 20
        and action.placement is not None
        and (
            action.placement.x1,
            action.placement.y1,
            action.placement.x2,
            action.placement.y2,
            action.placement.flipped,
        )
        == (
            placement["x1"],
            placement["y1"],
            placement["x2"],
            placement["y2"],
            placement["flipped"],
        )
    )
    child = state.step(counterfactual)
    response = recommend(
        RecommendRequest(
            engine="exact",
            state=state_to_debug_json(child),
            top_k=100,
            exact_max_secs=600.0,
            swindle=False,
            seed=0,
        )
    )
    if response.get("engine") != "exact":
        raise RuntimeError(f"exact response unavailable: {response.get('reason')}")
    by_pick = {}
    for rec in response["recommendations"]:
        pick = rec.get("pick_domino_id")
        if pick is None:
            continue
        old = by_pick.get(int(pick))
        if old is None or rec["exact_margin_pts"] > old["exact_margin_pts"]:
            by_pick[int(pick)] = rec
    rows = [
        {
            "viewer_reply_pick_domino_id": pick,
            "best_exact_margin_viewer": rec["exact_margin_pts"],
            "best_action_idx": rec["action_idx"],
            "best_placement": rec["placement"],
        }
        for pick, rec in by_pick.items()
    ]
    rows.sort(key=lambda row: (-row["best_exact_margin_viewer"], row["viewer_reply_pick_domino_id"]))
    output = {
        "schema": "kingdomino-bga-wishiwas-counterfactual-response/v1",
        "counterfactual_first_pick": 20,
        "remaining_row_for_viewer": sorted(child.current_row),
        "exact_optimal_viewer_reply": rows[0]["viewer_reply_pick_domino_id"],
        "exact_reply_values": rows,
        "counterfactual_action_idx": int(encode_action(counterfactual, state)),
        "exact_root_margin_viewer": response["root_margin_pts"],
        "search_ms": response["search_ms"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
