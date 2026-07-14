"""Paired, seat-swapped NNUE evaluation with per-agent clock accounting.

The same runner is used for candidate promotion and the NNUE-vs-AlphaZero bar.
Wall time is measured only on non-forced decisions; forced moves are reported
separately so a superficially equal game clock cannot hide unequal search time.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from games.kingdomino.promotion import wilson_lower_bound
from games.kingdomino.round_robin_eval import (
    PairResult,
    Participant,
    build_open_loop_checkpoint_participant,
    play_game,
    update_pair,
)
from games.kingdomino.rust_expectiminimax import OperationalRustSearchBot


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class TimedBot:
    def __init__(self, bot):
        self.bot = bot
        self.decision_times: list[float] = []
        self.forced_times: list[float] = []

    def choose_action(self, state, actions=None, rng=None):
        legal = state.legal_actions() if actions is None else actions
        t0 = time.perf_counter()
        action = self.bot.choose_action(state, legal, rng=rng)
        elapsed = time.perf_counter() - t0
        (self.forced_times if len(legal) == 1 else self.decision_times).append(elapsed)
        return action

    def summary(self) -> dict:
        values = np.asarray(self.decision_times, dtype=np.float64)
        forced = np.asarray(self.forced_times, dtype=np.float64)
        return {
            "decision_count": int(len(values)),
            "decision_total_seconds": float(values.sum()) if len(values) else 0.0,
            "decision_mean_seconds": float(values.mean()) if len(values) else 0.0,
            "decision_p50_seconds": float(np.quantile(values, 0.50)) if len(values) else 0.0,
            "decision_p95_seconds": float(np.quantile(values, 0.95)) if len(values) else 0.0,
            "decision_max_seconds": float(values.max()) if len(values) else 0.0,
            "forced_count": int(len(forced)),
            "forced_total_seconds": float(forced.sum()) if len(forced) else 0.0,
        }


def nnue_participant(
    name: str,
    artifact: str | Path,
    *,
    move_secs: float,
    max_depth: int = 12,
    chance_samples: int = 16,
    eval_kind: str = "sparse_nnue_q",
    full_width_ordering: bool = True,
    selective_width: int | None = None,
    selective_root_width: int | None = None,
    selective_min_depth: int = 4,
) -> Participant:
    artifact = str(Path(artifact).resolve())

    def make_bot():
        return OperationalRustSearchBot(
            max_secs=move_secs,
            max_depth=max_depth,
            chance_samples=chance_samples,
            eval=eval_kind,
            nnue_path=artifact,
            full_width_ordering=full_width_ordering,
            selective_width=selective_width,
            selective_root_width=selective_root_width,
            selective_min_depth=selective_min_depth,
        )

    return Participant(name=name, make_bot=make_bot, kind=eval_kind, source=artifact)


def run_paired(
    a: Participant,
    b: Participant,
    *,
    seed_start: int,
    paired_seeds: int,
    settings: dict | None = None,
) -> dict:
    if paired_seeds < 1:
        raise ValueError("paired_seeds must be >= 1")
    bot_a = TimedBot(a.make_bot())
    bot_b = TimedBot(b.make_bot())
    pair = PairResult(a=a.name, b=b.name)
    games = []
    t0 = time.perf_counter()
    for i in range(paired_seeds):
        seed = seed_start + i
        for result in (
            play_game(a.name, bot_a, b.name, bot_b, seed=seed),
            play_game(b.name, bot_b, a.name, bot_a, seed=seed),
        ):
            update_pair(pair, result, a.name, b.name)
            games.append(asdict(result))
    wall = time.perf_counter() - t0
    pair.seconds = wall
    points = pair.a_wins + 0.5 * pair.draws
    return {
        "schema": "kingdomino_nnue_paired_match_v1",
        "settings": settings or {},
        "pair": {
            **asdict(pair),
            "a_points_rate": points / pair.games,
            "a_points_lcb_95": wilson_lower_bound(points, pair.games),
            "avg_margin_a": pair.avg_margin_a,
        },
        "timing": {a.name: bot_a.summary(), b.name: bot_b.summary()},
        "wall_seconds": wall,
        "games": games,
    }


def write_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _main_nnue(args) -> dict:
    candidate_width = (args.candidate_selective_width
                       if args.candidate_selective_width is not None
                       else args.selective_width)
    incumbent_width = (args.incumbent_selective_width
                       if args.incumbent_selective_width is not None
                       else args.selective_width)
    candidate_root_width = args.selective_root_width if candidate_width is not None else None
    incumbent_root_width = args.selective_root_width if incumbent_width is not None else None
    a = nnue_participant("candidate", args.candidate, move_secs=args.move_secs,
                         max_depth=args.max_depth, chance_samples=args.chance_samples,
                         full_width_ordering=args.candidate_full_width_ordering,
                         selective_width=candidate_width,
                         selective_root_width=candidate_root_width,
                         selective_min_depth=args.selective_min_depth)
    b = nnue_participant("incumbent", args.incumbent, move_secs=args.move_secs,
                         max_depth=args.max_depth, chance_samples=args.chance_samples,
                         full_width_ordering=args.incumbent_full_width_ordering,
                         selective_width=incumbent_width,
                         selective_root_width=incumbent_root_width,
                         selective_min_depth=args.selective_min_depth)
    return run_paired(
        a, b, seed_start=args.seed_start, paired_seeds=args.paired_seeds,
        settings={
            "kind": "nnue_promotion",
            "candidate": str(Path(args.candidate).resolve()),
            "candidate_sha256": _sha256(args.candidate),
            "incumbent": str(Path(args.incumbent).resolve()),
            "incumbent_sha256": _sha256(args.incumbent),
            "move_secs": args.move_secs,
            "max_depth": args.max_depth,
            "chance_samples": args.chance_samples,
            "candidate_full_width_ordering": args.candidate_full_width_ordering,
            "incumbent_full_width_ordering": args.incumbent_full_width_ordering,
            "selective_width": args.selective_width,
            "candidate_selective_width": candidate_width,
            "incumbent_selective_width": incumbent_width,
            "selective_root_width": args.selective_root_width,
            "selective_min_depth": args.selective_min_depth,
        },
    )


def _main_az(args) -> dict:
    nnue = nnue_participant("NNUE", args.nnue, move_secs=args.nnue_move_secs,
                            max_depth=args.max_depth,
                            chance_samples=args.chance_samples,
                            full_width_ordering=args.full_width_ordering,
                            selective_width=args.selective_width,
                            selective_root_width=args.selective_root_width,
                            selective_min_depth=args.selective_min_depth)
    az = build_open_loop_checkpoint_participant(
        args.az_checkpoint,
        name="AlphaZero",
        device=args.device,
        sims=args.az_sims,
        c_puct=args.c_puct,
        temperature=0.0,
        channels_override=None,
        blocks_override=None,
        bilinear_dim_override=None,
    )
    return run_paired(
        nnue, az, seed_start=args.seed_start, paired_seeds=args.paired_seeds,
        settings={
            "kind": "nnue_vs_alphazero_clock_characterized",
            "nnue": str(Path(args.nnue).resolve()),
            "nnue_sha256": _sha256(args.nnue),
            "nnue_move_secs": args.nnue_move_secs,
            "az_checkpoint": str(Path(args.az_checkpoint).resolve()),
            "az_checkpoint_sha256": _sha256(args.az_checkpoint),
            "az_sims": args.az_sims,
            "device": args.device,
            "max_depth": args.max_depth,
            "chance_samples": args.chance_samples,
            "full_width_ordering": args.full_width_ordering,
            "selective_width": args.selective_width,
            "selective_root_width": args.selective_root_width,
            "selective_min_depth": args.selective_min_depth,
        },
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("nnue", help="paired candidate-vs-incumbent promotion match")
    p.add_argument("--candidate", required=True)
    p.add_argument("--incumbent", required=True)
    p.add_argument("--move-secs", type=float, default=0.5)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--chance-samples", type=int, default=16)
    p.add_argument(
        "--candidate-full-width-ordering",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--incumbent-full-width-ordering",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--selective-width", type=int, default=None)
    p.add_argument("--selective-root-width", type=int, default=None)
    p.add_argument("--candidate-selective-width", type=int, default=None)
    p.add_argument("--incumbent-selective-width", type=int, default=None)
    p.add_argument("--selective-min-depth", type=int, default=4)
    p.add_argument("--paired-seeds", type=int, default=32)
    p.add_argument("--seed-start", type=int, default=20260714)
    p.add_argument("--out", required=True)

    p = sub.add_parser("az-baseline", help="paired NNUE-vs-AZ clock characterization")
    p.add_argument("--nnue", required=True)
    p.add_argument("--az-checkpoint", required=True)
    p.add_argument("--nnue-move-secs", type=float, default=0.5)
    p.add_argument("--az-sims", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--chance-samples", type=int, default=16)
    p.add_argument(
        "--full-width-ordering",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--selective-width", type=int, default=None)
    p.add_argument("--selective-root-width", type=int, default=None)
    p.add_argument("--selective-min-depth", type=int, default=4)
    p.add_argument("--paired-seeds", type=int, default=8)
    p.add_argument("--seed-start", type=int, default=20260714)
    p.add_argument("--out", required=True)

    args = ap.parse_args()
    report = _main_nnue(args) if args.command == "nnue" else _main_az(args)
    out = write_report(report, args.out)
    pair = report["pair"]
    print(json.dumps({"out": str(out), "pair": pair, "timing": report["timing"]}, indent=2))


if __name__ == "__main__":
    main()
