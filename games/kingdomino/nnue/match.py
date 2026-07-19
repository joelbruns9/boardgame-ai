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
    _REPORT_FIELDS = (
        "completed_depth",
        "timed_out",
        "elapsed_secs",
        "nodes",
        "last_iteration_nodes",
        "chance_nodes",
        "aspiration_researches",
        "star_cutoffs",
        "exact_extensions",
        "tt_hits",
        "tt_cutoffs",
        "ordering_evals",
        "ordering_actions",
        "selective_pruned",
        "selective",
    )

    def __init__(self, bot):
        self.bot = bot
        self.decision_times: list[float] = []
        self.forced_times: list[float] = []
        self.search_reports: list[dict] = []

    def choose_action(self, state, actions=None, rng=None):
        legal = state.legal_actions() if actions is None else actions
        t0 = time.perf_counter()
        action = self.bot.choose_action(state, legal, rng=rng)
        elapsed = time.perf_counter() - t0
        (self.forced_times if len(legal) == 1 else self.decision_times).append(elapsed)
        if len(legal) != 1:
            report = getattr(self.bot, "last_report", None)
            if report is not None:
                # Snapshot the PyO3 object immediately; OperationalRustSearchBot
                # replaces `last_report` on every move.
                self.search_reports.append({
                    field: getattr(report, field, None) for field in self._REPORT_FIELDS
                })
        return action

    def _search_summary(self) -> dict | None:
        if not self.search_reports:
            return None

        def values(field: str, dtype=np.float64) -> np.ndarray:
            return np.asarray(
                [report[field] for report in self.search_reports if report[field] is not None],
                dtype=dtype,
            )

        depth = values("completed_depth")
        nodes = values("nodes")
        last_nodes = values("last_iteration_nodes")
        timed_out = values("timed_out")
        chance = values("chance_nodes")
        chance_available = len(chance) == len(self.search_reports)
        node_total = float(nodes.sum()) if len(nodes) else 0.0
        chance_total = float(chance.sum()) if chance_available else None
        denominator = node_total + (chance_total or 0.0)
        chance_share = (
            chance_total / denominator
            if chance_total is not None and denominator > 0.0
            else None
        )
        return {
            "report_count": len(self.search_reports),
            "completed_depth_mean": float(depth.mean()) if len(depth) else None,
            "completed_depth_median": float(np.median(depth)) if len(depth) else None,
            "completed_depth_min": int(depth.min()) if len(depth) else None,
            "completed_depth_max": int(depth.max()) if len(depth) else None,
            "completed_depth_histogram": (
                {
                    str(int(value)): int((depth == value).sum())
                    for value in np.unique(depth)
                }
                if len(depth)
                else {}
            ),
            "nodes_mean": float(nodes.mean()) if len(nodes) else None,
            "nodes_total": int(nodes.sum()) if len(nodes) else None,
            "last_iteration_nodes_mean": (
                float(last_nodes.mean()) if len(last_nodes) else None
            ),
            "timeout_count": int(timed_out.sum()) if len(timed_out) else 0,
            "timeout_rate": float(timed_out.mean()) if len(timed_out) else None,
            "chance_nodes_mean": float(chance.mean()) if chance_available else None,
            "chance_nodes_total": int(chance_total) if chance_total is not None else None,
            "chance_share": chance_share,
            "chance_share_denominator": "chance_nodes / (nodes + chance_nodes)",
        }

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
            "search": self._search_summary(),
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
    observer=None,
    continue_on_game_error: bool = False,
) -> dict:
    if paired_seeds < 1:
        raise ValueError("paired_seeds must be >= 1")
    bot_a = TimedBot(a.make_bot())
    bot_b = TimedBot(b.make_bot())
    pair = PairResult(a=a.name, b=b.name)
    games = []
    failed_games = []
    t0 = time.perf_counter()
    for i in range(paired_seeds):
        seed = seed_start + i
        game_specs = (
            (a.name, bot_a, b.name, bot_b),
            (b.name, bot_b, a.name, bot_a),
        )
        for p0_name, p0_bot, p1_name, p1_bot in game_specs:
            try:
                result = play_game(
                    p0_name,
                    p0_bot,
                    p1_name,
                    p1_bot,
                    seed=seed,
                    observer=observer,
                )
            except Exception as exc:
                if not continue_on_game_error:
                    raise
                failure = {
                    "seed": seed,
                    "p0": p0_name,
                    "p1": p1_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failed_games.append(failure)
                callback = getattr(observer, "on_game_aborted", None)
                if callback is not None:
                    callback(seed, p0_name, p1_name, exc)
                continue
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
            "a_points_rate": points / pair.games if pair.games else 0.0,
            "a_points_lcb_95": wilson_lower_bound(points, pair.games),
            "avg_margin_a": pair.avg_margin_a,
        },
        "timing": {a.name: bot_a.summary(), b.name: bot_b.summary()},
        "wall_seconds": wall,
        "requested_games": paired_seeds * 2,
        "completed_games": pair.games,
        "failed_games": failed_games,
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
