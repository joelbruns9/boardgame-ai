"""Price the endgame solver's workload once, so a rented box never has to.

`endgame_trigger_study.measure_node_rate` states the asymmetry this file exists
to exploit:

    The only machine-dependent quantity in the whole calibration [is the node
    rate]. Everything else -- how many nodes a position needs, how the cost is
    distributed across card counts -- is a property of the POSITIONS, identical
    on every box.

So the expensive half of choosing `--endgame-solver-attempt-nodes` and
`--endgame-solver-max-nodes` can be done anywhere, once, and committed. What
travels is a table of (features, true node cost) for every position a real run
attempted. From it, `price()` computes exactly what any candidate pair of caps
would have bought -- proofs, nodes spent, nodes wasted -- with no solving at all.

WHERE THE TRUE COSTS COME FROM, and why two sources are needed. A run's own
buffer records `solver_nodes` for every attempted solve, which is the true cost
for the ones that ANSWERED. For the ones that hit the cap it is only a floor:
the position was right-censored at the budget, and the buffer cannot say by how
much. `resolve_censored.py` re-solves exactly those at a far larger budget, and
merging the two is what makes the tail real rather than a lower bound. Pricing a
wider cap off censored data alone would report every censored position as costing
exactly the old cap, which is the one answer that is certainly wrong.

FEATURES, not predictions. The model can be refit -- that is an open piece of
work, since its residual on the censored tail is +1.08 decades against a shipped
margin of 0.4 -- and a corpus storing predictions would silently price the old
model forever. Features are the durable thing; the prediction is derived.

WHAT THIS CORPUS IS NOT. It is one iteration of one run, so the position mix
belongs to that net's play. A weaker net reaches different endgames, and the
distribution drifts as a run strengthens. Treat a sizing off this corpus as a
starting point to be revisited, not a constant -- and prefer a corpus built from
the most recent iteration available.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SCHEMA = "solver_corpus/1"


def build(buffer_path: Path, study_path: Path | None) -> dict:
    """Replay a buffer and price every position a solve was attempted at.

    No solving happens here: the costs are already recorded (or supplied by the
    study), and the features are a pure function of the position. What this
    costs is one replay of the buffer.
    """

    import seven_wonders_rust as swr

    from .buffer import read_records, replay
    from .rust_bridge import rust_game_from_state

    true_cost: dict[tuple[int, int], float] = {}
    if study_path is not None:
        study = json.loads(Path(study_path).read_text(encoding="utf-8"))
        for row in study["rows"]:
            # `stop is None` means the re-solve finished, so `nodes` is the
            # position's true cost. Otherwise it is censored again, even at the
            # study's budget, and infinity is the honest entry: what is known is
            # only that it exceeds that budget, and pricing any finite cap
            # against it must count it as a decline.
            true_cost[(row["game_seed"], row["move_index"])] = (
                float(row["nodes"]) if row["stop"] is None else math.inf
            )

    rows: list[dict] = []
    censored_unresolved = 0
    for record in read_records(buffer_path):
        wanted = {
            move.i: move
            for move in record.moves
            if getattr(move, "solver_attempted", False)
        }
        if not wanted:
            continue

        def visit(game, move, _wanted=wanted, _record=record):
            nonlocal censored_unresolved
            target = _wanted.get(move.i)
            if target is None:
                return
            key = (_record.seed, move.i)
            declined = getattr(target, "solver_stop", None) is not None
            if key in true_cost:
                cost = true_cost[key]
            elif declined:
                # Censored, and the study did not cover it. The buffer's
                # `solver_nodes` is a FLOOR, not a cost, so recording it as one
                # would price this position as affordable at exactly the cap it
                # failed at.
                cost = math.inf
                censored_unresolved += 1
            else:
                cost = float(getattr(target, "solver_nodes", 0) or 0)
            rows.append(
                {
                    "features": [
                        float(value)
                        for value in swr.endgame_cost_features(rust_game_from_state(game))
                    ],
                    "true_nodes": cost if math.isfinite(cost) else None,
                    "declined": declined,
                }
            )

        replay(record, on_state=visit)

    return {
        "schema": SCHEMA,
        "source_buffer": str(buffer_path),
        "source_study": str(study_path) if study_path else None,
        "feature_names": list(swr.endgame_cost_model_features()),
        "positions": len(rows),
        "declined": sum(1 for row in rows if row["declined"]),
        # Censored rows the study did not resolve. `true_nodes` is null for
        # these, and `price` counts them as never completing -- correct, but it
        # makes the corpus pessimistic about large caps in proportion to this
        # number, so it is reported rather than buried.
        "censored_unresolved": censored_unresolved,
        "rows": rows,
    }


def predictions(corpus: dict, model: dict) -> list[float]:
    """`log10(nodes)` predicted for each row, under the given fitted model.

    Applied here rather than stored, so refitting the model reprices the corpus
    without rebuilding it.
    """

    names = corpus["feature_names"]
    if names != list(model.get("features", names)):
        raise SystemExit(
            "corpus feature order differs from the model's; one of them was "
            "built against a different `cost_model::features`"
        )
    weights = [model["coefficients"][name] for name in names]
    intercept = model["intercept"]
    return [
        intercept + sum(x * w for x, w in zip(row["features"], weights))
        for row in corpus["rows"]
    ]


def price(
    corpus: dict,
    model: dict,
    *,
    attempt_nodes: float,
    max_nodes: float,
    games: int,
) -> dict:
    """What one candidate pair of caps would have bought on this corpus.

    A position is ATTEMPTED when `predict + margin <= log10(attempt_nodes)`,
    exactly as `CostModel::affordable` decides it. An attempted position then
    either completes, costing its true node count, or exhausts `max_nodes` and
    costs exactly that -- a decline is not free, which is the whole reason the
    two caps are separate.

    A COUNTERFACTUAL, not a simulation. It holds the position set fixed, and a
    real run would not: a successful solve masks the policy target, which changes
    the move sampled, which changes every position after it. Measured on a toy
    grid, narrowing the bar 10x gave 57 attempts against 64 with one position
    unique to the narrow run. So read the columns as first-order, and never as a
    promise about a run that has not happened.
    """

    margin = float(model["margin_decades"])
    bar = math.log10(attempt_nodes) - margin
    proofs = attempts = 0
    nodes = wasted = 0.0
    for row, predicted in zip(corpus["rows"], predictions(corpus, model)):
        if predicted > bar:
            continue
        attempts += 1
        cost = row["true_nodes"]
        if cost is not None and cost <= max_nodes:
            proofs += 1
            nodes += cost
        else:
            nodes += max_nodes
            wasted += max_nodes
    return {
        "attempt_nodes": int(attempt_nodes),
        "max_nodes": int(max_nodes),
        "effective_bar_nodes": 10.0**bar,
        "attempts": attempts,
        "proofs": proofs,
        "nodes": nodes,
        "wasted_nodes": wasted,
        "wasted_fraction": wasted / nodes if nodes else 0.0,
        "nodes_per_game": nodes / games if games else 0.0,
        "proofs_per_game": proofs / games if games else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("buffer", type=Path, help="a run buffer, e.g. iter_0096.jsonl")
    parser.add_argument(
        "--study",
        type=Path,
        default=None,
        help="resolve_censored.py output for the SAME buffer. Without it every "
        "censored position is priced as never completing, because the buffer "
        "records only a floor on its cost.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    corpus = build(args.buffer, args.study)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(corpus), encoding="utf-8")
    print(
        f"{corpus['positions']:,} positions, {corpus['declined']} declined, "
        f"{corpus['censored_unresolved']} censored without a resolved cost"
    )
    print(f"written: {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
