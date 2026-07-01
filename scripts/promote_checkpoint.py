from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from games.kingdomino.promotion import (
    DEFAULT_BEST_DIR,
    DEFAULT_CURRENT_BEST,
    DEFAULT_FIXED_SUITE,
    compare_fixed_suite,
    decide_promotion,
    evaluate_checkpoint_match,
    fixed_suite_summary_for_checkpoint,
    promote_current_best,
    promotion_payload,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote a Kingdomino checkpoint to current_best.pt if it passes gates."
    )
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--current_best", type=Path, default=DEFAULT_CURRENT_BEST)
    parser.add_argument("--best_dir", type=Path, default=DEFAULT_BEST_DIR)
    parser.add_argument("--games", type=int, default=400,
                        help="Total head-to-head games; rounded up to paired seeds.")
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_slots", type=int, default=32)
    parser.add_argument("--leaf_batch", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--c_puct", type=float, default=1.5)
    parser.add_argument("--fpu", type=float, default=-0.2)
    parser.add_argument("--margin_gain", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--min_win_rate", type=float, default=0.55)
    parser.add_argument("--min_lcb", type=float, default=0.50)
    parser.add_argument("--confidence_z", type=float, default=1.96)
    parser.add_argument("--fixed_suite", type=Path, default=DEFAULT_FIXED_SUITE)
    parser.add_argument("--fixed_suite_tolerance", type=float, default=0.05)
    parser.add_argument("--skip_fixed_suite", action="store_true")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Allow promotion when current_best.pt does not exist.")
    parser.add_argument("--confirm", action="store_true",
                        help="Actually copy candidate to current_best.pt if gates pass.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Decision JSON path. Default: best_dir/promotion_candidate.json.")
    args = parser.parse_args()

    if not args.candidate.exists():
        raise SystemExit(f"candidate does not exist: {args.candidate}")

    has_current_best = args.current_best.exists()
    if not has_current_best and not args.bootstrap:
        raise SystemExit(
            f"current_best does not exist: {args.current_best}; use --bootstrap for first promotion"
        )

    match = None
    baseline_summary = None
    if has_current_best:
        print(
            f"Head-to-head: candidate vs current_best, games={args.games}, sims={args.sims}",
            flush=True,
        )
        match = evaluate_checkpoint_match(
            args.candidate,
            args.current_best,
            games=args.games,
            sims=args.sims,
            device=args.device,
            batch_slots=args.batch_slots,
            leaf_batch=args.leaf_batch,
            seed=args.seed,
            c_puct=args.c_puct,
            fpu=args.fpu,
            margin_gain=args.margin_gain,
            alpha=args.alpha,
            z=args.confidence_z,
        )
        print(
            f"  result: {match.wins}-{match.losses}-{match.draws} "
            f"points={match.points:.1f}/{match.games} "
            f"win_rate={match.win_rate:.1%} LCB={match.lower_confidence_bound:.1%} "
            f"margin={match.mean_margin:+.2f}",
            flush=True,
        )

    candidate_summary = None
    fixed_cmp = None
    if args.skip_fixed_suite:
        fixed_cmp = compare_fixed_suite(None, None, tolerance=args.fixed_suite_tolerance)
    else:
        print(f"Fixed suite: candidate on {args.fixed_suite}", flush=True)
        candidate_summary = fixed_suite_summary_for_checkpoint(
            args.candidate, suite=args.fixed_suite, device=args.device)
        if has_current_best:
            print("Fixed suite: current_best baseline", flush=True)
            baseline_summary = fixed_suite_summary_for_checkpoint(
                args.current_best, suite=args.fixed_suite, device=args.device)
        fixed_cmp = compare_fixed_suite(
            candidate_summary,
            baseline_summary,
            tolerance=args.fixed_suite_tolerance,
        )
        print(f"  fixed-suite gate: {fixed_cmp.reason}", flush=True)

    decision = decide_promotion(
        match,
        fixed_cmp,
        min_win_rate=args.min_win_rate,
        min_lcb=args.min_lcb,
        bootstrap=args.bootstrap and not has_current_best,
    )
    payload = promotion_payload(
        candidate=args.candidate,
        current_best=args.current_best,
        decision=decision,
        candidate_fixed_summary=candidate_summary,
        baseline_fixed_summary=baseline_summary,
        extra={
            "gate_config": {
                "games": args.games,
                "sims": args.sims,
                "device": args.device,
                "batch_slots": args.batch_slots,
                "leaf_batch": args.leaf_batch,
                "seed": args.seed,
                "min_win_rate": args.min_win_rate,
                "min_lcb": args.min_lcb,
                "confidence_z": args.confidence_z,
                "fixed_suite_tolerance": args.fixed_suite_tolerance,
                "skip_fixed_suite": args.skip_fixed_suite,
            }
        },
    )

    out = args.out or (args.best_dir / "promotion_candidate.json")
    write_json(out, payload)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True), flush=True)
    print(f"Decision JSON: {out}", flush=True)

    if not decision.passed:
        print("PROMOTION FAILED: current_best was not changed.", flush=True)
        return 1

    if not args.confirm:
        print("PROMOTION PASSED, but --confirm was not set; current_best was not changed.", flush=True)
        return 0

    promoted = promote_current_best(
        args.candidate,
        best_dir=args.best_dir,
        current_best=args.current_best,
        payload=payload,
    )
    print(f"PROMOTED: {args.candidate} -> {promoted}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
