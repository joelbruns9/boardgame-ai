"""Deep equal-budget tile comparison for the wishiwas BGA disagreement case."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import kingdomino_rust as kr

from games.kingdomino.action_codec import encode_action
from games.kingdomino.bga_reanalysis_corpus import DEFAULT_CORPUS
from games.kingdomino.deep_target_screen import DEFAULT_CHECKPOINT, _pick_key, _read_jsonl
from games.kingdomino.deep_target_stage2 import (
    _search_kwargs,
    aggregate_restricted_search,
    aggregate_search,
)
from games.kingdomino.denial_search import load_checkpoint_network
from games.kingdomino.dominoes import DOMINOES
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.late_model_policy_audit import prepare_bga_state
from games.kingdomino.self_play import make_rust_evaluator


TABLE_ID = "883162423"
SOURCE_INDEX = 37
HUMAN_PICK = 28
MODEL_PICK = 20
SEEDS = (5703456653659912300, 9389259382755257756)
OUTPUT = Path(
    "runs/kingdomino/placement_audit/bga_wishiwas_disagreement_deep_v1.json"
)


def _tile(domino_id: int):
    domino = DOMINOES[int(domino_id)]
    return {
        "domino_id": int(domino_id),
        "a": {"terrain": domino.a.terrain.name, "crowns": domino.a.crowns},
        "b": {"terrain": domino.b.terrain.name, "crowns": domino.b.crowns},
    }


def main() -> int:
    position = next(
        row
        for row in _read_jsonl(DEFAULT_CORPUS)
        if str(row["table_id"]) == TABLE_ID
        and int(row["source_decision_index"]) == SOURCE_INDEX
    )
    rules = position["state"].get("rules", {})
    state = prepare_bga_state(
        position["state"],
        harmony=bool(rules.get("harmony", True)),
        middle=bool(rules.get("middle_kingdom", True)),
    )
    net, config = load_checkpoint_network(DEFAULT_CHECKPOINT, "cuda")
    margin_gain = float(config.get("margin_gain", 2.0))
    alpha = float(config.get("alpha", 0.5))
    evaluator = make_rust_evaluator(
        net,
        device="cuda",
        amp=bool(config.get("inference_amp", False)),
        margin_gain=margin_gain,
        alpha=alpha,
    )
    kwargs = _search_kwargs(config, margin_gain, alpha)
    rust_state = _rust_state_from_python(state)
    action_indices = [int(encode_action(action, state)) for action in state.legal_actions()]
    picks = sorted({_pick_key(state, action_idx) for action_idx in action_indices})
    repeats = []
    started = time.perf_counter()
    for repeat, seed in enumerate(SEEDS):
        t0 = time.perf_counter()
        children, root_value_p0 = kr.advisor_open_loop_search(
            rust_state, evaluator, 30_000, seed=seed, **kwargs
        )
        ordinary = aggregate_search(
            state,
            children,
            root_value_p0=float(root_value_p0),
            elapsed_seconds=time.perf_counter() - t0,
            seed=seed,
        )
        groups = []
        for pick in picks:
            allowed = [
                action_idx
                for action_idx in action_indices
                if _pick_key(state, action_idx) == pick
            ]
            t1 = time.perf_counter()
            forced_children, forced_root_p0 = kr.advisor_open_loop_search(
                rust_state,
                evaluator,
                30_000,
                seed=seed,
                root_allowed_actions=allowed,
                **kwargs,
            )
            forced = aggregate_restricted_search(
                state,
                forced_children,
                allowed_actions=allowed,
                root_value_p0=float(forced_root_p0),
                elapsed_seconds=time.perf_counter() - t1,
                seed=seed,
                expected_visits=30_000,
            )
            groups.append(
                {
                    "pick_domino_id": pick,
                    "tile": _tile(pick),
                    "q_actor": forced["root_value_actor"],
                    "selected_action": forced["selected_action"],
                }
            )
        groups.sort(key=lambda group: (-group["q_actor"], group["pick_domino_id"]))
        values = {group["pick_domino_id"]: group["q_actor"] for group in groups}
        repeats.append(
            {
                "repeat": repeat,
                "seed": seed,
                "ordinary_30000_selected_pick": ordinary["selected_pick_domino_id"],
                "ordinary_30000_root_q_actor": ordinary["root_value_actor"],
                "matched_30000_best_pick": groups[0]["pick_domino_id"],
                "human_minus_model_q": values[HUMAN_PICK] - values[MODEL_PICK],
                "human_pick_q": values[HUMAN_PICK],
                "model_pick_q": values[MODEL_PICK],
                "pick_groups": groups,
            }
        )
        print(f"repeat {repeat + 1}/2 complete", flush=True)

    actor = int(state.current_actor)
    score = state.boards[actor].score(state.config.harmony, state.config.middle_kingdom)
    output = {
        "schema": "kingdomino-bga-wishiwas-deep-case/v1",
        "table_id": TABLE_ID,
        "source_decision_index": SOURCE_INDEX,
        "position_id": position["position_id"],
        "exact_attempt": "timed out at 600 seconds; these results are non-exact",
        "sims_per_ordinary_search": 30_000,
        "sims_per_forced_pick_group": 30_000,
        "repeats": repeats,
        "human_pick_domino_id": HUMAN_PICK,
        "model_raw_pick_domino_id": MODEL_PICK,
        "mean_human_minus_model_q": statistics.fmean(
            repeat["human_minus_model_q"] for repeat in repeats
        ),
        "human_better_than_model_in_both_repeats": all(
            repeat["human_minus_model_q"] > 0 for repeat in repeats
        ),
        "current_domino": _tile(int(position["reconstruction"]["domino_id"])),
        "remaining_hidden_tile_ids": sorted(position["state"]["debug"]["deck"]),
        "human_action": position["human_action"],
        "board_before_pretty": state.boards[actor].pretty(),
        "score_before": asdict(score) | {"total": score.total},
        "recorded_final_scores": position["diagnostics_only"]["final_scores"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
