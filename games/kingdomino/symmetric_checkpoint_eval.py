"""Auditable symmetric checkpoint evaluation for Kingdomino.

Both checkpoints use the same search mechanism, simulation/NN-row budget,
seeds, and seat rotations.  The default treatment is the frozen progressive
chance search at deck counts 8 and 12; ``--search-mode open_loop`` provides the
incumbent-search control.  Promotion gates are deliberately not routed through
this CLI and retain their existing open-loop default.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from games.kingdomino.elo_rating import (
    EloConfig,
    play_rating_games_with_diagnostics,
)
from games.kingdomino.promotion import (
    _net_from_checkpoint,
    match_stats_from_pair,
    sha256_file,
    wilson_interval,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def _atomic_games_jsonl(path: Path, games: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for game in games:
            handle.write(json.dumps(asdict(game), sort_keys=True) + "\n")
    tmp.replace(path)


def _parse_decks(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(text).split(",")
                   if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one deck count is required")
    return values


def progressive_diagnostic_errors(
        diagnostics: dict[str, Any], decks: tuple[int, ...]) -> list[str]:
    """Return fail-closed reasons when requested progressive search did not run."""
    errors: list[str] = []
    for deck in decks:
        search_key = f"progressive_chance_search_count_deck{deck}"
        path_key = f"progressive_chance_path_count_deck{deck}"
        if int(diagnostics.get(search_key, 0)) <= 0:
            errors.append(f"no progressive searches recorded at deck {deck}")
        if int(diagnostics.get(path_key, 0)) <= 0:
            errors.append(f"no progressive paths crossed at deck {deck}")
    for key, label in (
        ("progressive_chance_admission_count", "admissions"),
        ("progressive_chance_bootstrap_rows", "bootstrap rows"),
        ("progressive_chance_width_sample_count", "width samples"),
    ):
        if int(diagnostics.get(key, 0)) <= 0:
            errors.append(f"no progressive {label} recorded")
    return errors


def run_symmetric_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    candidate = Path(args.candidate).resolve()
    baseline = Path(args.baseline).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"candidate checkpoint not found: {candidate}")
    if not baseline.is_file():
        raise FileNotFoundError(f"baseline checkpoint not found: {baseline}")
    if int(args.games) <= 0 or int(args.games) % 2:
        raise ValueError("--games must be a positive even number (seat pairs)")
    existing = [
        path for path in (
            output_dir / "manifest.json",
            output_dir / "games.jsonl",
            output_dir / "result.json",
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite evaluation artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    progressive = args.search_mode == "chance_progressive"
    decks = tuple(args.chance_progressive_decks) if progressive else ()
    cfg = EloConfig(
        games_per_anchor=int(args.games) // 2,
        sims=int(args.sims),
        device=str(args.device),
        n_slots=int(args.batch_slots),
        leaf_batch=int(args.leaf_batch),
        c_puct=float(args.c_puct),
        fpu=float(args.fpu),
        margin_gain=float(args.margin_gain),
        alpha=float(args.alpha),
        seed=int(args.seed),
        verbose=False,
        progressive_chance_decks=decks,
        progressive_chance_width_schedule=str(args.chance_width_schedule),
        progressive_chance_n_init=int(args.chance_n_init),
        progressive_chance_d_min=int(args.chance_d_min),
        progressive_chance_deck8_cap=int(args.chance_deck8_cap),
        progressive_chance_deck12_cap=int(args.chance_deck12_cap),
        progressive_chance_max_init_fraction=float(
            args.chance_max_init_fraction),
    )

    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": _now_iso(),
        "candidate": str(candidate),
        "candidate_sha256": sha256_file(candidate),
        "baseline": str(baseline),
        "baseline_sha256": sha256_file(baseline),
        "games": int(args.games),
        "paired_seeds": int(args.games) // 2,
        "seed_start": int(args.seed),
        "search_mode": str(args.search_mode),
        "search_is_symmetric": True,
        "parameters": {
            "sims": cfg.sims,
            "batch_slots": cfg.n_slots,
            "leaf_batch": cfg.leaf_batch,
            "c_puct": cfg.c_puct,
            "fpu": cfg.fpu,
            "margin_gain": cfg.margin_gain,
            "alpha": cfg.alpha,
            "progressive_chance_decks": list(decks),
            "progressive_chance_width_schedule":
                cfg.progressive_chance_width_schedule,
            "progressive_chance_n_init": cfg.progressive_chance_n_init,
            "progressive_chance_d_min": cfg.progressive_chance_d_min,
            "progressive_chance_deck8_cap":
                cfg.progressive_chance_deck8_cap,
            "progressive_chance_deck12_cap":
                cfg.progressive_chance_deck12_cap,
            "progressive_chance_max_init_fraction":
                cfg.progressive_chance_max_init_fraction,
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)

    print(
        f"Loading checkpoints and running {args.games} games "
        f"({args.games // 2} paired seeds) with {args.search_mode} search...",
        flush=True,
    )
    candidate_net = _net_from_checkpoint(candidate, cfg.device)
    baseline_net = _net_from_checkpoint(baseline, cfg.device)
    started = time.perf_counter()
    pair, games, diagnostics = play_rating_games_with_diagnostics(
        candidate_net,
        baseline_net,
        "candidate",
        "baseline",
        int(args.games) // 2,
        int(args.seed),
        cfg,
    )
    elapsed = time.perf_counter() - started
    stats = match_stats_from_pair(
        pair, games, candidate_name="candidate", z=float(args.z))
    pair_ci = wilson_interval(stats.pair_points, stats.pairs, z=float(args.z))
    errors = progressive_diagnostic_errors(diagnostics, decks) if progressive else []

    result = {
        **manifest,
        "status": "complete" if not errors else "invalid",
        "completed_at": _now_iso(),
        "elapsed_seconds": elapsed,
        "games_per_second": int(args.games) / elapsed if elapsed else 0.0,
        "valid": not errors,
        "errors": errors,
        "match": asdict(stats),
        "pair_score_wilson_interval": [pair_ci[0], pair_ci[1]],
        "raw_pair": asdict(pair),
        "progressive_diagnostics": diagnostics,
        "artifacts": {
            "manifest": "manifest.json",
            "games": "games.jsonl",
            "result": "result.json",
        },
    }
    _atomic_games_jsonl(output_dir / "games.jsonl", games)
    _atomic_json(output_dir / "result.json", result)
    print(json.dumps({
        "valid": result["valid"],
        "games": stats.games,
        "candidate_game_score_rate": stats.win_rate,
        "candidate_pair_score_rate": stats.pair_score_rate,
        "pair_score_wilson_interval": result["pair_score_wilson_interval"],
        "mean_margin": stats.mean_margin,
        "elapsed_seconds": elapsed,
        "progressive_diagnostics": diagnostics,
        "result": str(output_dir / "result.json"),
    }, indent=2, sort_keys=True), flush=True)
    if errors:
        raise RuntimeError("progressive evaluation invalid: " + "; ".join(errors))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Symmetric paired checkpoint evaluation with auditable search")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-slots", type=int, default=48)
    parser.add_argument("--leaf-batch", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20330000)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--fpu", type=float, default=-0.2)
    parser.add_argument("--margin-gain", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--z", type=float, default=1.96)
    parser.add_argument(
        "--search-mode", choices=("open_loop", "chance_progressive"),
        default="chance_progressive")
    parser.add_argument(
        "--chance-progressive-decks", type=_parse_decks, default=(8, 12))
    parser.add_argument("--chance-width-schedule", default="4,8,16,32,64,70")
    parser.add_argument("--chance-n-init", type=int, default=2)
    parser.add_argument("--chance-d-min", type=int, default=4)
    parser.add_argument("--chance-deck8-cap", type=int, default=16)
    parser.add_argument("--chance-deck12-cap", type=int, default=16)
    parser.add_argument("--chance-max-init-fraction", type=float, default=0.25)
    return parser


def main() -> None:
    run_symmetric_evaluation(build_parser().parse_args())


if __name__ == "__main__":
    main()
