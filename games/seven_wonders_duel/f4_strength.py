"""Paired seat-swapped F4 WU-vs-sequential playing-strength gate.

The position gate chooses a candidate leaf batch first. This runner then plays
that candidate against the permanent leaf_batch=1 Rust/F3.3 oracle with paired
game and search seeds. Rows are append-only and resumable; the summary uses a
fixed-seed pair bootstrap and reports the one-sided Elo lower bound required by
``f4_contract_v2.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from .codec import decode_action
from .engine import apply_action
from .f4_quality import CONTRACT_PATH, _net_adapter
from .game import Phase, new_game
from .phase_e import load_evaluator, state_actor
from .portable_rng import PortableRng
from .rust_bridge import (
    rust_flat_batch_adapter,
    rust_game_for_self_play,
    rust_games_for_self_play,
)


SCHEMA = "f4-strength-row-1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score_to_elo(score: float) -> float:
    clipped = min(1.0 - 1e-9, max(1e-9, score))
    return 400.0 * math.log10(clipped / (1.0 - clipped))


def summarize(
    rows: list[dict], *, required_pairs: int, confidence: float, seed: int
) -> dict:
    by_pair: dict[int, list[float]] = {}
    for row in rows:
        by_pair.setdefault(int(row["pair_index"]), []).append(float(row["fast_score"]))
    complete = {
        pair: sum(scores) / 2.0
        for pair, scores in by_pair.items()
        if len(scores) == 2
    }
    pair_scores = list(complete.values())
    score = sum(pair_scores) / len(pair_scores) if pair_scores else 0.5
    bootstrap = []
    if pair_scores:
        rng = random.Random(seed)
        for _ in range(10_000):
            bootstrap.append(
                sum(rng.choice(pair_scores) for _ in pair_scores) / len(pair_scores)
            )
        bootstrap.sort()
        lower_index = max(0, math.floor((1.0 - confidence) * len(bootstrap)) - 1)
        score_lower = bootstrap[lower_index]
    else:
        score_lower = 0.0
    return {
        "complete_pairs": len(pair_scores),
        "games": len(pair_scores) * 2,
        "required_pairs": required_pairs,
        "score": score,
        "score_one_sided_lower": score_lower,
        "elo": _score_to_elo(score),
        "elo_one_sided_lower": _score_to_elo(score_lower),
        "sample_size_met": len(pair_scores) >= required_pairs,
    }


def _play_game(
    *,
    adapter,
    game_seed: int,
    first_player: int,
    fast_seat: int,
    leaf_batch: int,
    sims: int,
    top_k: int,
    force: bool,
) -> dict:
    state = new_game(game_seed, first_player=first_player)
    rust = rust_game_for_self_play(game_seed, first_player)
    seeds = PortableRng(game_seed ^ 0x8CB92BA72F3D8DD7)
    moves = 0
    while state.phase is not Phase.COMPLETE:
        actor = state_actor(state)
        batch = leaf_batch if actor == fast_seat else 1
        search_seed = seeds.getrandbits(63)
        result = rust.closed_search_batched_net(
            adapter,
            batch,
            sims,
            top_k,
            search_seed,
            force=force,
        )
        action = int(result[0])
        apply_action(state, decode_action(state, action))
        rust.apply_index(action)
        moves += 1
        if moves > 256:
            raise RuntimeError("strength game exceeded 256 moves")
    winner = state.winner
    fast_score = 0.5 if winner is None else float(winner == fast_seat)
    return {
        "winner": winner,
        "victory_type": state.victory_type.name.lower() if state.victory_type else None,
        "moves": moves,
        "fast_score": fast_score,
        "final_fingerprint": rust.fingerprint(),
    }


def run(args) -> dict:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    quality_contract = contract["fast_search_quality_gate"]
    required = {
        "sims": quality_contract["required_sims"],
        "top_k": quality_contract["required_top_k"],
        "force": quality_contract["required_force_expand_root_chance"],
    }
    actual = {"sims": args.sims, "top_k": args.top_k, "force": args.force}
    mismatches = [
        f"{field}={actual[field]!r} (required {value!r})"
        for field, value in required.items()
        if actual[field] != value
    ]
    if mismatches:
        raise ValueError(
            "strength run does not match f4-contract-2: " + ", ".join(mismatches)
        )
    strength = contract["fast_search_quality_gate"]["playing_strength"]
    required_pairs = strength["minimum_seat_swapped_pairs"]
    if args.pairs < required_pairs and not args.allow_underfilled:
        raise ValueError(f"--pairs must be at least {required_pairs} for an eligible run")
    evaluator = load_evaluator(str(args.checkpoint), args.device)
    evaluator.max_batch = args.global_batch_cap
    adapter = rust_flat_batch_adapter(evaluator)
    args.output.mkdir(parents=True, exist_ok=True)
    run_config = {
        "schema": "f4-strength-run-2",
        "contract_schema_version": contract["schema_version"],
        "contract_sha256": _sha256(CONTRACT_PATH),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "device": args.device,
        "leaf_batch": args.leaf_batch,
        "pairs": args.pairs,
        "sims": args.sims,
        "top_k": args.top_k,
        "seed": args.seed,
        "force": args.force,
        "age_deal_samples": args.age_deal_samples,
        "reference_age_deal_samples": args.reference_age_deal_samples,
        "slots": args.slots,
        "global_batch_cap": args.global_batch_cap,
        "max_inflight_batches": args.max_inflight_batches,
    }
    config_path = args.output / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != run_config:
            raise ValueError("existing strength output uses a different run configuration")
    else:
        config_path.write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    rows_path = args.output / "games.jsonl"
    rows = []
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                if row.get("schema") == SCHEMA:
                    rows.append(row)
                    done.add((row["pair_index"], row["leg"]))
    import seven_wonders_rust as swr

    with rows_path.open("a", encoding="utf-8", newline="\n") as handle:
        for leg, fast_seat in enumerate((0, 1)):
            pending = [pair for pair in range(args.pairs) if (pair, leg) not in done]
            for start in range(0, len(pending), args.slots):
                pair_indices = pending[start : start + args.slots]
                seeds = [args.seed + pair for pair in pair_indices]
                first_players = [pair % 2 for pair in pair_indices]
                records, _ = swr.self_play_many_flat_net(
                    adapter=adapter,
                    games=rust_games_for_self_play(seeds, first_players),
                    game_seeds=seeds,
                    global_batch_cap=args.global_batch_cap,
                    leaf_batch=args.leaf_batch,
                    leaf_batch_p0=args.leaf_batch if fast_seat == 0 else 1,
                    leaf_batch_p1=args.leaf_batch if fast_seat == 1 else 1,
                    deterministic_actions=True,
                    cheap_sims_min=args.sims,
                    cheap_sims_max=args.sims,
                    full_sims_min=args.sims,
                    full_sims_max=args.sims,
                    full_search_fraction=0.0,
                    top_k=args.top_k,
                    draft_prior=0.0,
                    iteration=-1,
                    force=args.force,
                    age_deal_samples=args.age_deal_samples,
                    age_deal_samples_p0=(
                        args.age_deal_samples
                        if fast_seat == 0
                        else args.reference_age_deal_samples
                    )
                    if args.reference_age_deal_samples is not None
                    else None,
                    age_deal_samples_p1=(
                        args.age_deal_samples
                        if fast_seat == 1
                        else args.reference_age_deal_samples
                    )
                    if args.reference_age_deal_samples is not None
                    else None,
                    max_inflight_batches=args.max_inflight_batches,
                )
                for pair_index, game_seed, first_player, record in zip(
                    pair_indices, seeds, first_players, records
                ):
                    winner = record["winner"]
                    fast_score = 0.5 if winner is None else float(winner == fast_seat)
                    row = {
                        "schema": SCHEMA,
                        "pair_index": pair_index,
                        "leg": leg,
                        "game_seed": game_seed,
                        "first_player": first_player,
                        "fast_seat": fast_seat,
                        "winner": winner,
                        "victory_type": record["victory_type"],
                        "moves": len(record["moves"]),
                        "fast_score": fast_score,
                        "final_fingerprint": record["final_fingerprint"],
                    }
                    rows.append(row)
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                print(
                    f"strength: leg {leg + 1}/2, {min(start + len(pair_indices), len(pending))}/{len(pending)} games",
                    flush=True,
                )
    summary = summarize(
        rows,
        required_pairs=required_pairs,
        confidence=strength["one_sided_confidence"],
        seed=args.seed,
    )
    summary["non_inferiority_margin_elo"] = strength["non_inferiority_margin_elo"]
    summary["eligible"] = (
        summary["sample_size_met"]
        and summary["games"] >= strength["minimum_games"]
        and summary["elo_one_sided_lower"] >= strength["non_inferiority_margin_elo"]
    )
    summary["manifest"] = {
        "contract_schema_version": contract["schema_version"],
        "contract_sha256": _sha256(CONTRACT_PATH),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint": str(args.checkpoint.resolve()),
        "pending_policy": "wu_incomplete_visits",
        "leaf_batch": args.leaf_batch,
        "sims": args.sims,
        "top_k": args.top_k,
        "force_expand_root_chance": args.force,
        "age_deal_sample_count": args.age_deal_samples,
        "reference_age_deal_sample_count": args.reference_age_deal_samples,
        "seed": args.seed,
        "device": args.device,
        "slots": args.slots,
        "global_batch_cap": args.global_batch_cap,
        "max_inflight_batches": args.max_inflight_batches,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--leaf-batch", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pairs", type=int, default=1200)
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=504000)
    parser.add_argument(
        "--force", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--age-deal-samples", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--reference-age-deal-samples", type=int, choices=(4, 8, 16, 32))
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--global-batch-cap", type=int, default=256)
    parser.add_argument("--max-inflight-batches", type=int, default=2)
    parser.add_argument("--allow-underfilled", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
