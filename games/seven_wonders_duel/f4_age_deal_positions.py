"""Generate paired AgeDeal root-position rows against the paired-32 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .f4_quality import CONTRACT_PATH, _read_jsonl
from .game import Phase
from .phase_e import load_evaluator, reconstruct
from .rust_bridge import rust_flat_batch_adapter, rust_game_from_prefix


def run(args) -> dict:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    registered = contract["production_semantics"]["age_deal"]
    positions = _read_jsonl(args.corpus / "positions.jsonl")
    prepared = []
    for position in positions:
        state = reconstruct(position)
        if state.phase is not Phase.CHOOSE_NEXT_START_PLAYER:
            continue
        _, rust = rust_game_from_prefix(
            position["game_seed"], position["first_player"], position["prefix"]
        )
        prepared.append((position, rust))
    if not prepared:
        raise ValueError("corpus contains no choose-next-start-player AgeDeal roots")

    evaluator = load_evaluator(str(args.checkpoint), args.device)
    evaluator.max_batch = args.global_batch_cap
    adapter = rust_flat_batch_adapter(evaluator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import seven_wonders_rust as swr

    rows = []
    candidates = registered["candidate_counts"]
    reference_count = registered["diagnostic_reference_count"]
    for start in range(0, len(prepared), args.position_batch):
        batch = prepared[start : start + args.position_batch]
        games = [rust for _, rust in batch]
        seeds = [args.seed + start + index for index in range(len(batch))]
        reference = swr.search_many_flat_net(
            adapter,
            games,
            seeds,
            args.global_batch_cap,
            1,
            contract["production_semantics"]["full_sims"],
            contract["production_semantics"]["top_k"],
            force=True,
            age_deal_samples=reference_count,
        )
        for count in candidates:
            candidate = swr.search_many_flat_net(
                adapter,
                games,
                seeds,
                args.global_batch_cap,
                1,
                contract["production_semantics"]["full_sims"],
                contract["production_semantics"]["top_k"],
                force=True,
                age_deal_samples=count,
            )
            for (position, _), expected, actual, seed in zip(
                batch, reference, candidate, seeds
            ):
                rows.append(
                    {
                        "schema": "f4-age-deal-position-row-1",
                        "position_id": position["id"],
                        "search_seed": seed,
                        "sample_count": count,
                        "reference_sample_count": reference_count,
                        "reference_action": expected["action"],
                        "candidate_action": actual["action"],
                        "action_agreement": float(actual["action"] == expected["action"]),
                        "root_value_abs_error": abs(
                            float(actual["root_value"]) - float(expected["root_value"])
                        ),
                        "candidate_nn_work": actual["nn_work"],
                        "reference_nn_work": expected["nn_work"],
                    }
                )
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {"positions": len(prepared), "rows": len(rows), "output": str(args.output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--global-batch-cap", type=int, default=256)
    parser.add_argument("--position-batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=604000)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
