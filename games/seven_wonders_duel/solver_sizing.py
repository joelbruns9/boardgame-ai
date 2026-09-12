"""Choose the solver's two node caps from what the rented box can actually do.

The split of labour, which is the whole reason this is cheap:

* **priced once, anywhere** -- how many nodes a position costs, and therefore
  what any pair of caps buys. `solver_corpus` holds it. Node counts are a
  property of positions, identical on every box.
* **measured on the box** -- how fast a core solves under contention, how many
  solver threads there are, and how long a generation iteration takes.

Cross the two and the caps fall out:

    budget = threads x generation_wall x rate x target_share
    demand(bar, max_nodes) = games x nodes_per_game(bar, max_nodes)
    choose the pair maximising proofs subject to demand <= budget

WHY A SHARE AND NOT THE WHOLE BUDGET. The solver's threads are cores generation
is not using, but the two are not independent: solves land unevenly, the
position mix drifts as the net strengthens, and a corpus built at one strength
under-describes another. Sizing to 100% of a measured capacity means the first
iteration whose endgames run rich is late. The share is the headroom for that.

WHY THE WALL CLOCK IS DERIVED AND NEVER CHOSEN. `--endgame-solver-max-secs` is a
guard against a hung position, not a budget. If it binds, node-censoring becomes
deadline-censoring, and whether a position got proved depends on how busy the box
was -- so the buffer stops being a function of its seeds. It is therefore set
from `max_nodes / rate` times a slack factor, and the factor is large on purpose.

WHAT THIS DOES NOT MODEL. A solve occupies a scheduler slot for its duration.
With `--exclude-parked-from-budget` the slot's token returns to the pool, so the
cost really is CPU and this arithmetic is the right one; without it the solve
also withholds concurrency from generation and every number here understates the
price. The launcher sets that flag on for exactly this reason, and
`parked_slot_fraction` in the sweep output is the check that it took effect.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .solver_corpus import admission_ceiling, price

#: Multiples of the attempt bar to consider as timeouts.
#:
#: Anchored on the bar rather than absolute, because the useful quantity is how
#: much room a solve gets beyond what the model predicted for it. The model's
#: residual on the censored tail measured +1.08 decades -- about 12x -- so a
#: timeout only 2-4x above the bar abandons most of the positions it mispriced.
TIMEOUT_MULTIPLES = (1, 2, 4, 8, 16, 32)

#: Slack between the node budget and the wall clock. Deliberately generous: the
#: clock must never be the thing that stops a solve.
CLOCK_SLACK = 5.0

#: Fraction of the DRAIN TAIL a single solve may occupy before it is flagged.
#:
#: The scheduler cannot end an iteration until every game finishes -- a slot
#: parked on a solve keeps `active_count` up -- so the worst-case stall is
#: `max_nodes / rate`, one thread running alone while the GPU idles.
#:
#: That sounds worse than it is, and the run's own counters say so. Across 97
#: cloud2 iterations the drain tail (below 25% of peak occupancy) was a median
#: 16.5% of generation wall, worst 23.0%, at a 40M cap -- games finishing
#: unevenly once the queue empties, nothing to do with solving. Idle after the
#: LAST NN batch, which is where a solve-induced stall would show since a parked
#: game submits none, was 3s median and 3s worst.
#:
#: So a long solve at the end lands INSIDE a window that is already idle rather
#: than extending the iteration. The bound that follows is "keep one solve
#: inside the drain you already have", not "keep it small".
STALL_FRACTION_OF_DRAIN = 0.5

#: Drain tail as a fraction of generation wall, when the caller does not supply
#: a measured one. cloud2's median across 97 iterations. It belongs to that
#: geometry -- 1,000 games over 256 slots -- and the box sweep should supply its
#: own, which is why this is a default and not a constant.
DEFAULT_DRAIN_FRACTION = 0.165


def candidates(
    corpus: dict,
    model: dict,
    *,
    games: int,
    bars: tuple[int, ...],
    multiples: tuple[int, ...] = TIMEOUT_MULTIPLES,
) -> list[dict]:
    """Every (bar, timeout) pair worth pricing, priced."""

    ceiling = admission_ceiling(corpus, model)
    out = []
    dropped = []
    for bar in bars:
        if bar > ceiling * (1.0 + 1e-9):
            # Not an error and not a zero result: the corpus simply has no rows
            # for what this bar would admit, because the run that produced it
            # refused them. Pricing it anyway would report the collecting bar's
            # numbers under a wider label.
            dropped.append(bar)
            continue
        for multiple in multiples:
            out.append(
                price(
                    corpus,
                    model,
                    attempt_nodes=bar,
                    max_nodes=bar * multiple,
                    games=games,
                )
            )
    if dropped:
        print(
            "not priceable from this corpus (above its admission ceiling of "
            f"{ceiling:,.0f} nodes, so the positions such a bar would admit are "
            "absent rather than expensive): "
            + ", ".join(f"{bar:,.0f}" for bar in dropped),
            flush=True,
        )
    if not out:
        raise SystemExit(
            f"every candidate bar is above this corpus's admission ceiling of "
            f"{ceiling:,.0f} nodes. Lower --bars, or build a corpus from a run "
            "that attempted more than this one did."
        )
    return out


def size(
    corpus: dict,
    model: dict,
    *,
    rate: float,
    threads: int,
    generation_wall_seconds: float,
    games: int,
    target_share: float,
    bars: tuple[int, ...],
    drain_fraction: float = DEFAULT_DRAIN_FRACTION,
) -> dict:
    """The best pair that fits the box's solver budget, and the runners-up.

    "Best" is most proofs, with ties broken toward fewer nodes. Proofs are the
    product -- a decided endgame replaces a sampled value target with an exact
    one -- and nodes are only what they cost.
    """

    if rate <= 0 or threads < 1 or generation_wall_seconds <= 0:
        raise SystemExit(
            "solver sizing needs a positive node rate, thread count and "
            "generation wall; one of them was not measured"
        )
    budget = threads * generation_wall_seconds * rate * target_share
    drain_seconds = generation_wall_seconds * drain_fraction
    priced = candidates(corpus, model, games=games, bars=bars)
    for row in priced:
        # `nodes_for_games` is already normalised by the corpus's own game
        # count and scaled to this iteration; `nodes_per_game * games` cancelled.
        row["demand_nodes"] = row["nodes_for_games"]
        row["fits"] = row["demand_nodes"] <= budget
        row["share_of_capacity"] = row["demand_nodes"] / budget * target_share
        # Thread-seconds the solver would spend per game, which is the number to
        # compare against a run's own profile once it exists.
        row["solver_seconds_per_game"] = row["nodes_per_game"] / rate
        row["proofs_expected"] = row["proofs_for_games"]
        row["max_secs"] = (row["max_nodes"] / rate) * CLOCK_SLACK
        # WORST-CASE STALL: one solve running the cap to exhaustion, alone,
        # while the iteration waits for its game to finish. Reported rather than
        # filtered on -- it is absorbed by the drain when it fits inside it, and
        # a filter would cost hard proofs to avoid a cost that is already idle.
        row["worst_stall_seconds"] = row["max_nodes"] / rate
        row["stall_share_of_drain"] = (
            row["worst_stall_seconds"] / drain_seconds if drain_seconds else None
        )
        row["stall_exceeds_drain"] = bool(
            drain_seconds
            and row["worst_stall_seconds"] > drain_seconds * STALL_FRACTION_OF_DRAIN
        )

    affordable = [row for row in priced if row["fits"]]
    if not affordable:
        cheapest = min(priced, key=lambda row: row["demand_nodes"])
        raise SystemExit(
            "no candidate fits the solver budget: the cheapest priced "
            f"{cheapest['demand_nodes'] / 1e9:.1f}B nodes/iteration against a "
            f"budget of {budget / 1e9:.1f}B. Lower the bars, raise "
            "--target-share, or accept that this box cannot solve at this "
            "generation rate."
        )
    best = max(affordable, key=lambda row: (row["proofs"], -row["nodes"]))
    return {
        "budget_nodes": budget,
        "drain_seconds": drain_seconds,
        "drain_fraction": drain_fraction,
        "rate_nodes_per_second_per_thread": rate,
        "solver_threads": threads,
        "generation_wall_seconds": generation_wall_seconds,
        "games_per_iteration": games,
        "target_share": target_share,
        "chosen": best,
        "considered": sorted(priced, key=lambda row: -row["proofs"]),
    }


def render_env(chosen: dict) -> str:
    """The chosen caps, as the environment the launcher reads.

    Emitted in the same shape `sweep_launch_env.py` uses so pass 2 sources one
    file. `max_secs` is included because it is DERIVED from the node budget and
    the box's rate -- re-deriving it from a stale rate elsewhere is how a clock
    ends up binding.
    """

    return "\n".join(
        [
            "# Solver caps, sized on this box (setup_cloud_7wd.sh stage 8c).",
            f"#   attempt bar   {chosen['attempt_nodes']:,} nodes "
            f"(effective {chosen['effective_bar_nodes']:,.0f} after the margin)",
            f"#   node timeout  {chosen['max_nodes']:,}",
            f"#   proofs/game   {chosen['proofs_per_game']:.2f} at "
            f"{chosen['nodes_per_game'] / 1e6:.1f}M nodes/game",
            f"#   wasted        {chosen['wasted_fraction']:.1%} of solver nodes",
            "# The clock is DERIVED from the node budget and this box's measured",
            "# rate, and is slack on purpose: if it binds, a node-censored",
            "# decline becomes a load-dependent one and the buffer stops being a",
            "# function of its seeds.",
            f"export ENDGAME_SOLVER_ATTEMPT_NODES={chosen['attempt_nodes']}",
            f"export ENDGAME_SOLVER_MAX_NODES={chosen['max_nodes']}",
            f"export ENDGAME_SOLVER_MAX_SECS={chosen['max_secs']:.0f}",
        ]
    ) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("games/seven_wonders_duel/endgame_cost_model.json"),
    )
    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="nodes/s/thread measured UNDER CONTENTION on this box "
        "(endgame_trigger_study.measure_node_rate_contended). The "
        "single-thread figure overstates capacity.",
    )
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument(
        "--generation-wall-seconds",
        type=float,
        required=True,
        help="seconds of GENERATION per iteration -- not total iteration wall. "
        "Solving happens during generation; training is not solver time.",
    )
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--target-share", type=float, default=0.80)
    parser.add_argument(
        "--drain-fraction",
        type=float,
        default=DEFAULT_DRAIN_FRACTION,
        help="the iteration's drain tail as a fraction of generation wall, used "
        "to judge whether one solve's worst-case stall is absorbed by idle time "
        "the run already has. Default is cloud2's median across 97 iterations "
        "(16.5%%, worst 23.0%%) and belongs to its geometry; read this box's own "
        "from the sweep's batch_live_slots/batch_submit_ns.",
    )
    parser.add_argument(
        "--bars",
        default="5000000,10000000,20000000,40000000,80000000",
        help="candidate attempt bars, comma separated",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    model = json.loads(args.model.read_text(encoding="utf-8"))
    result = size(
        corpus,
        model,
        rate=args.rate,
        threads=args.threads,
        generation_wall_seconds=args.generation_wall_seconds,
        games=args.games,
        target_share=args.target_share,
        bars=tuple(int(part) for part in args.bars.split(",") if part.strip()),
        drain_fraction=args.drain_fraction,
    )
    chosen = result["chosen"]
    print(
        f"solver budget {result['budget_nodes'] / 1e9:.1f}B nodes/iteration "
        f"({args.threads} threads x {args.generation_wall_seconds:,.0f}s x "
        f"{args.rate:,.0f}/s x {args.target_share:.0%})"
    )
    print(
        f"drain tail ~{result['drain_seconds']:,.0f}s "
        f"({result['drain_fraction']:.1%} of generation wall) -- a solve shorter "
        "than that lands in a window the scheduler is already idling through"
    )
    print(
        f"{'bar':>10} {'timeout':>11} {'proofs':>7} {'nodes/iter':>11} "
        f"{'wasted':>7} {'stall':>8} {'fits':>5}"
    )
    for row in result["considered"][:12]:
        stall = f"{row['worst_stall_seconds']:.0f}s"
        if row["stall_exceeds_drain"]:
            stall = "!" + stall
        print(
            f"{row['attempt_nodes'] / 1e6:>9.0f}M {row['max_nodes'] / 1e6:>10.0f}M "
            f"{row['proofs']:>7,} {row['demand_nodes'] / 1e9:>10.2f}B "
            f"{row['wasted_fraction']:>6.1%} {stall:>8} "
            f"{'yes' if row['fits'] else 'NO':>5}"
        )
    print(
        f"\nchosen: bar {chosen['attempt_nodes']:,} / timeout "
        f"{chosen['max_nodes']:,} / clock {chosen['max_secs']:.0f}s"
    )
    if chosen["stall_exceeds_drain"]:
        print(
            f"! worst-case stall {chosen['worst_stall_seconds']:,.0f}s against a "
            f"{result['drain_seconds']:,.0f}s drain tail. The scheduler cannot "
            "end an iteration while a game is parked on a solve, so one solve "
            "running this cap to exhaustion would hold it open past the window "
            "it would otherwise finish in. Lower --bars, or pass a measured "
            "--drain-fraction if this box drains more slowly than cloud2 did."
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_env(chosen), encoding="utf-8")
        print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
