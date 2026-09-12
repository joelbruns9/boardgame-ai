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


def build(
    buffer_path: Path,
    study_path: Path | None,
    *,
    collecting_attempt_nodes: int | None = None,
    collecting_model: dict | None = None,
) -> dict:
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
    seen_games: set = set()
    for record in read_records(buffer_path):
        wanted = {
            move.i: move
            for move in record.moves
            if getattr(move, "solver_attempted", False)
        }
        seen_games.add(record.seed)
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

    collecting_games = len(seen_games)
    return {
        "schema": SCHEMA,
        "source_buffer": str(buffer_path),
        "source_study": str(study_path) if study_path else None,
        # The model the COLLECTING run admitted positions with. Coverage is a
        # property of (bar, MODEL) jointly, not of the bar alone: a refit moves
        # each position's required bar, so the same 40M bar under a new model
        # admits a different set -- and the ones it newly admits are ABSENT
        # here, because the collecting run never attempted them.
        "collecting_model": collecting_model,
        # The bar the COLLECTING run used. `admission_ceiling` needs it exactly:
        # inferring it from the rows lands just under the true value and
        # excludes the run's own settings from being priced.
        "collecting_attempt_nodes": collecting_attempt_nodes,
        # GAMES the corpus was collected over. `price` normalises by this and
        # then scales to the target, because the two are different numbers:
        # dividing corpus nodes by the REQUESTED game count and multiplying by
        # it again cancels, so demand came out identical at 100, 1,000 and
        # 10,000 games while capacity grew with the iteration wall -- larger
        # iterations looked free.
        "collecting_games": collecting_games,
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


def admission_ceiling(corpus: dict, model: dict) -> float:
    """The largest attempt bar this corpus can HONESTLY price, in nodes.

    A corpus holds the positions a run ATTEMPTED, and a run attempts a position
    only when it clears that run's own bar. Everything the bar refused is
    therefore absent -- not recorded as expensive, absent -- so a candidate bar
    wider than the collecting run's admits positions the corpus has no rows for.

    Priced naively, such a candidate reports exactly the same proofs and nodes as
    the collecting bar, which reads as "widening buys nothing". The truth is
    "this corpus cannot see what widening would buy", and those are opposite
    conclusions: the first is a measurement, the second is missing data.

    RECORDED when the build knew it, because inferring it is off by exactly the
    amount that matters. The inference is `10**(max(prediction) + margin)` -- the
    smallest bar admitting every row present -- and the largest prediction in a
    corpus approaches the collecting bar from BELOW without reaching it. On
    cloud2 that inferred 39,975,202 against a true bar of 40,000,000, which
    excluded the run's own settings: the one candidate that must always be
    priceable, since it is the status quo every other option is compared to.
    """

    recorded = corpus.get("collecting_attempt_nodes")
    if not recorded:
        return 10.0 ** (max(predictions(corpus, model)) + float(model["margin_decades"]))
    ceiling = float(recorded)

    collecting = corpus.get("collecting_model")
    if not collecting or _same_model(collecting, model):
        return ceiling

    # A DIFFERENT MODEL. Coverage is a property of (bar, model) jointly: a refit
    # moves every position's required bar `10**(predict + margin)`, so the same
    # numeric bar admits a different set -- and whatever it newly admits is
    # absent here, because the collecting run never attempted it.
    #
    # Reproduced on the shipped corpus: seed 116260767 move 64 was NOT attempted
    # (required 48,162,395 under the collecting model, above the 40M bar), and
    # the refit puts it at 34,617,215 -- inside the bar, and missing.
    #
    # A BOUND, and a deliberately crude one: `ceiling * min(new/old)` assumes
    # every unobserved position moves as far as the worst observed one. On the
    # shipped corpus that worst ratio is 0.202, which collapses a 40M ceiling to
    # 8.09M -- while the set the refit actually newly admits at 40M is NINETEEN
    # positions out of 8,032, or 0.24%.
    #
    # So this is the fallback, not the answer. `uncovered_positions` enumerates
    # the gap exactly from the collecting buffer, and `price` reports it rather
    # than refusing; a bound that rejects a bar 99.76% covered is not caution,
    # it is a wrong answer with a safe-sounding shape.
    old_margin = float(collecting["margin_decades"])
    new_margin = float(model["margin_decades"])
    ratios = [
        10.0 ** ((new_p + new_margin) - (old_p + old_margin))
        for old_p, new_p in zip(predictions(corpus, collecting), predictions(corpus, model))
    ]
    return ceiling * min(ratios) if ratios else ceiling


def uncovered_positions(
    corpus: dict, model: dict, *, attempt_nodes: float, buffer_path: Path
) -> list[dict]:
    """Positions `attempt_nodes` admits under `model` that the corpus lacks.

    Enumerated, not bounded. The collecting run declined to ATTEMPT these, so
    they carry no true cost -- but they were still reached and played, so they
    are in the same buffer as ordinary moves and a replay finds them exactly.

    That is the whole reason the gap is cheap to close: widening coverage needs
    more offline SOLVING of positions that already exist, never new self-play.
    The positions themselves are irreplaceable -- they are what a strong net
    reaches -- and no rerun can produce them without that net.
    """

    import seven_wonders_rust as swr

    from .buffer import read_records, replay
    from .game import Phase
    from .rust_bridge import rust_game_from_state

    collecting = corpus.get("collecting_model")
    if not collecting:
        raise SystemExit(
            "this corpus records no collecting model, so what a different model "
            "would newly admit cannot be enumerated"
        )
    names = corpus["feature_names"]

    def required(features, fitted):
        weights = [fitted["coefficients"][name] for name in names]
        predicted = fitted["intercept"] + sum(
            x * w for x, w in zip(features, weights)
        )
        return 10.0 ** (predicted + float(fitted["margin_decades"]))

    bar = float(corpus["collecting_attempt_nodes"])
    out: list[dict] = []
    for record in read_records(Path(buffer_path)):
        moves = {move.i: move for move in record.moves}

        def visit(game, move, _moves=moves, _record=record):
            # The engine's own eligibility gate: `cost_model::eligible` is Age
            # III mid-play, and nothing outside it is ever a solve candidate.
            if game.phase is not Phase.PLAY_AGE or game.age != 3:
                return
            recorded = _moves.get(move.i)
            if recorded is None or recorded.solver_attempted:
                return
            features = [
                float(value)
                for value in swr.endgame_cost_features(rust_game_from_state(game))
            ]
            if required(features, collecting) > bar >= required(features, model):
                out.append(
                    {
                        "game_seed": _record.seed,
                        "move_index": move.i,
                        "features": features,
                    }
                )

        replay(record, on_state=visit)
    return out


def _same_model(left: dict, right: dict) -> bool:
    """Do these two describe the same admission decision?

    Compared on what `affordable` actually reads -- the intercept, the weights
    in feature order, and the margin -- rather than on the whole file, which
    carries a `fit` block that changes with every refit without changing a
    single decision.
    """

    if abs(float(left["intercept"]) - float(right["intercept"])) > 1e-12:
        return False
    if abs(float(left["margin_decades"]) - float(right["margin_decades"])) > 1e-12:
        return False
    names = set(left["coefficients"]) | set(right["coefficients"])
    return all(
        abs(float(left["coefficients"].get(n, 0.0))
            - float(right["coefficients"].get(n, 0.0))) <= 1e-12
        for n in names
    )


def price(
    corpus: dict,
    model: dict,
    *,
    attempt_nodes: float,
    max_nodes: float,
    games: int,
    uncovered: int | None = None,
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
    collecting_games = int(corpus.get("collecting_games") or 0)
    if collecting_games <= 0:
        raise SystemExit(
            "this corpus records no collecting game count, so per-game demand "
            "cannot be derived from it. Rebuild it with solver_corpus.py."
        )
    # COVERAGE, reported rather than refused when the gap is known.
    #
    # `uncovered` is the count of positions this bar admits that the corpus
    # lacks -- enumerated from the collecting buffer, not bounded. Supplied, it
    # is carried through to the caller as an uncertainty; absent, the crude
    # `admission_ceiling` bound applies and a bar beyond it is refused, because
    # then nothing knows how large the gap is.
    ceiling = admission_ceiling(corpus, model)
    if uncovered is None and attempt_nodes > ceiling * (1.0 + 1e-9):
        raise ValueError(
            f"attempt_nodes {attempt_nodes:,.0f} is above this corpus's "
            f"admission ceiling of {ceiling:,.0f}, and the size of the gap is "
            "unknown. Enumerate it with `uncovered_positions` against the "
            "collecting buffer and pass the count, or lower the bar."
        )
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
        # PER COLLECTING GAME, then scaled by the caller. Dividing by the
        # REQUESTED game count and multiplying by it again cancelled, so demand
        # was the corpus total whatever iteration size was asked for -- while
        # capacity grew with the wall, making bigger iterations look free.
        "collecting_games": collecting_games,
        "nodes_per_game": nodes / collecting_games,
        "proofs_per_game": proofs / collecting_games,
        "nodes_for_games": nodes / collecting_games * games,
        "proofs_for_games": proofs / collecting_games * games,
        # The gap, as an uncertainty rather than a refusal. A position the
        # corpus lacks has no true cost, so the honest bound is "it might fail
        # at the full timeout": proofs are understated by at most `uncovered`,
        # and nodes by at most `uncovered * max_nodes`. Demand is what decides
        # fits/does-not-fit, so the node figure is the one that can change an
        # answer -- proofs move by a fraction of a percent.
        "uncovered_positions": uncovered,
        "proofs_understated_by_at_most": uncovered or 0,
        "nodes_understated_by_at_most": (
            (uncovered or 0) / collecting_games * games * max_nodes
        ),
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
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="the collecting run's run_manifest.json. Supplies the attempt bar "
        "that run used, which bounds what this corpus can price -- inferring it "
        "from the rows lands just below the true value and excludes the run's "
        "own settings.",
    )
    parser.add_argument(
        "--collecting-attempt-nodes",
        type=int,
        default=None,
        help="override, when there is no manifest to read it from.",
    )
    parser.add_argument(
        "--collecting-model",
        type=Path,
        default=None,
        help="the cost model the COLLECTING run admitted positions with, when "
        "the manifest does not carry it. Coverage depends on the model as well "
        "as the bar: a refit moves each position's required bar, so the same "
        "bar admits a different set and whatever it newly admits is absent.",
    )
    args = parser.parse_args(argv)

    def find(node, key):
        """First occurrence of `key` anywhere in a nested manifest."""

        if isinstance(node, dict):
            if key in node:
                return node[key]
            for value in node.values():
                found = find(value, key)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = find(value, key)
                if found is not None:
                    return found
        return None

    bar = args.collecting_attempt_nodes
    if bar is None and args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        # A run predating the split recorded only one number, and it served as
        # both -- so the timeout IS that run's attempt bar.
        bar = find(manifest, "endgame_solver_attempt_nodes") or find(
            manifest, "endgame_solver_max_nodes"
        )
    if bar is None:
        print(
            "WARNING: no collecting bar recorded. It will be INFERRED from the "
            "rows, which lands just below the true value and will refuse to "
            "price the collecting run's own settings.",
            flush=True,
        )

    # The model the COLLECTING run used -- not whatever is shipped today. A
    # corpus that recorded today's model would claim coverage it does not have
    # the moment the model is refitted.
    collecting_model = None
    if args.collecting_model is not None:
        collecting_model = json.loads(
            args.collecting_model.read_text(encoding="utf-8")
        )
    elif args.manifest is not None:
        stored = find(json.loads(args.manifest.read_text(encoding="utf-8")),
                      "endgame_cost_model")
        if isinstance(stored, dict) and "coefficients" in stored:
            collecting_model = stored
    if collecting_model is None:
        print(
            "WARNING: no collecting model recorded. Coverage is a property of "
            "(bar, MODEL) jointly, so without it this corpus cannot tell a "
            "refit that its repricing is unsupported.",
            flush=True,
        )

    corpus = build(
        args.buffer, args.study,
        collecting_attempt_nodes=bar,
        collecting_model=collecting_model,
    )
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
