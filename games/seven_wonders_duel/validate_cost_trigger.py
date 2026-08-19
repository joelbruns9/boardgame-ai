"""Does the cost-predicted trigger's model hold on production self-play?

`endgame_trigger_study.py` fit `log10(nodes)` from cheap position features and
reported held-out R^2 0.904 -- on **40 games**, driven by a bot, with the solver
called on every eligible ply. The pre-retrain plan proposes replacing the card
cap with that model, which is a decision worth more than 40 games of evidence.

The 12-iteration shakedown produced 27,787 real solve attempts across 3,600
self-play games, and every one of them is reconstructible: a buffer record
carries `(seed, first_player, chance_log, actions)`, and `buffer.replay` walks
it with integrity checks, so features can be recomputed at exactly the position
the solver saw.

What this does and does not answer:

* It tests the **model form**, refit here and scored on held-out games. It does
  NOT test the study's published coefficients, because there are none to test:
  `endgame_trigger_study.py` computes rank correlations and budget simulations
  but contains no regression, so the plan's "held-out R^2 0.904" came from an
  ad-hoc fit that was never persisted and cannot be rerun. Treat 0.904 as
  unreproduced until this script's number either corroborates it or does not.

* **The holdout is by game, not by row.** Adjacent solved plies in one game
  share a position and a proof, so a row-wise split leaks the answer across it
  and would flatter any model.

* **Censoring is respected.** The shakedown ran with a binding 3-second wall
  clock (see PRE_RETRAIN_PLAN.md section 7), so declined solves stopped at an
  arbitrary, machine-load-dependent node count. Their true cost is unknown and
  only bounded below. They are scored as right-censored -- a prediction above
  the observed floor is not an error -- and never used to fit or to compute
  R^2, which would otherwise learn how busy the GPU was.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterator

from .buffer import GameRecord, read_records, replay
from .endgame_trigger_study import FEATURES, position_features

#: `FEATURES` carries one column a trigger cannot have. `moves_from_end` is only
#: known once the game is over, so a model using it would score well here and be
#: unusable in the decision it exists to make. Everything else comes straight out
#: of `position_features`, which is O(board) by construction.
UNAVAILABLE_AT_DECISION_TIME = frozenset({"moves_from_end"})

#: The four terms the plan proposed, kept as a named alternative because the
#: full set has 22 columns and the cloud corpus is small. Fitting 23 parameters
#: on a few hundred strong-play positions costs more in variance than the extra
#: columns buy in fit -- which is measurable, so it is measured rather than
#: argued. `legal` stands in for the plan's `log10(legal)`; `cards_x_logleg` is
#: already the interaction.
PLAN_FEATURES = ("cards_left", "unrevealed", "legal", "cards_x_logleg")

#: Features whose Python definitions need `_Derived` -- the observation's
#: reachability sets. Excluded from the shipped model because the trigger runs
#: inside Rust generation, and porting that machinery is a large surface for the
#: Python/Rust divergence this project has been bitten by. Dropping them is free:
#: measured on held-out cloud endgames, R2 is 0.939 either way and the trigger
#: makes the same 805 solves.
NEEDS_REACHABILITY = frozenset({"mil_win_feasible", "sci_win_feasible"})

TRIGGER_FEATURES = tuple(
    name for name in FEATURES if name not in UNAVAILABLE_AT_DECISION_TIME
)

#: What the Rust trigger computes. The shipped model is fit on exactly these.
RUST_FEATURES = tuple(
    name for name in TRIGGER_FEATURES if name not in NEEDS_REACHABILITY
)


def solved_positions(records: list[GameRecord]) -> Iterator[dict[str, Any]]:
    """Yield one feature row per solver attempt, at the position it saw.

    Replay verifies masks, actor and the final digest as it goes, so a row can
    only be produced from a game that reconstructs exactly.
    """

    for record in records:
        wanted = {move.i: move for move in record.moves if move.solver_attempted}
        if not wanted:
            continue

        def capture(game, move, wanted=wanted, record=record):
            attempt = wanted.get(move.i)
            if attempt is None:
                return
            row = position_features(game)
            row["nodes"] = attempt.solver_nodes
            # `solver_stop` is None exactly when the solve completed; a stop of
            # "budget" is a floor on cost, not a measurement of it.
            row["censored"] = attempt.solver_stop is not None
            row["stop"] = attempt.solver_stop
            row["regime"] = attempt.solver_regime
            row["seed"] = record.seed
            captured.append(row)

        captured: list[dict[str, Any]] = []
        replay(record, on_state=capture)
        yield from captured


def _design(row: dict[str, Any], features: tuple[str, ...]) -> list[float]:
    return [1.0] + [float(row[name]) for name in features]


def fit(rows: list[dict], features: tuple[str, ...]) -> list[float]:
    """Least squares for `log10(nodes)`, by normal equations with ridge.

    The ridge term is 1e-8: enough to survive a collinear pair (`cards_left` and
    `cards_x_logleg` are strongly related by construction), small enough not to
    shrink anything that matters.
    """

    columns = len(features) + 1
    ata = [[0.0] * columns for _ in range(columns)]
    atb = [0.0] * columns
    for row in rows:
        x = _design(row, features)
        y = math.log10(max(1, row["nodes"]))
        for i in range(columns):
            atb[i] += x[i] * y
            for j in range(columns):
                ata[i][j] += x[i] * x[j]
    for i in range(columns):
        ata[i][i] += 1e-8
    return _solve(ata, atb)


def fit_censored(
    rows: list[dict], features: tuple[str, ...], iterations: int = 50
) -> list[float]:
    """Least squares that uses the censored rows instead of discarding them.

    Dropping them is not neutral. A censored solve is one that ran out of budget,
    so the censored set IS the expensive tail, and a fit that sees only the rows
    that finished learns a cheaper world than the real one. That shows up
    directly: the uncensored-only fit underpredicts 51-55% of censored floors,
    and those are the positions a budget decision turns on.

    This is the EM algorithm for a Tobit model. A censored row contributes its
    floor when the current model already predicts above it (the observation is
    consistent, and the conditional expectation of a right-censored normal is
    approximated by the prediction itself), and is pulled up to the floor when
    the model predicts below it. Iterating to a fixed point is equivalent to
    imputing each censored value at `max(prediction, floor)`.

    The correction is **partial**. Imputing at `max(prediction, floor)`
    understates the conditional expectation of a right-censored normal, which
    lies above the prediction by an amount depending on the residual spread. On
    a synthetic truth with slope 0.5 it recovers 0.44 where the survivors-only
    fit gives 0.42 -- better, not unbiased. That is deliberate: the alternative
    is a full Tobit likelihood with a variance parameter, and the residual bias
    is small next to the safety margin the trigger applies anyway. Do not quote
    coefficients from this as unbiased estimates.
    """

    uncensored = [row for row in rows if not row["censored"]]
    censored = [row for row in rows if row["censored"]]
    if not censored:
        return fit(rows, features)

    coefficients = fit(uncensored, features)
    for _ in range(iterations):
        imputed = list(uncensored)
        for row in censored:
            floor = math.log10(max(1, row["nodes"]))
            predicted = predict(coefficients, row, features)
            filled = dict(row)
            filled["nodes"] = 10 ** max(predicted, floor)
            filled["censored"] = False
            imputed.append(filled)
        updated = fit(imputed, features)
        if all(abs(a - b) < 1e-9 for a, b in zip(coefficients, updated)):
            return updated
        coefficients = updated
    return coefficients


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("cost model design matrix is singular")
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def predict(coefficients: list[float], row: dict, features: tuple[str, ...]) -> float:
    return sum(c * x for c, x in zip(coefficients, _design(row, features)))


def score(
    coefficients: list[float], rows: list[dict], features: tuple[str, ...]
) -> dict[str, float]:
    """R^2 and residual spread on uncensored rows only.

    Censored rows are reported separately as a *violation rate*: the fraction
    whose prediction falls below the node count the solve had already reached
    when it was cut off. Those are unambiguous errors -- the true cost is at
    least the floor -- while a prediction above the floor is consistent with
    any true value and says nothing.
    """

    exact = [row for row in rows if not row["censored"]]
    if not exact:
        raise ValueError("no uncensored solves to score against")
    actual = [math.log10(max(1, row["nodes"])) for row in exact]
    predicted = [predict(coefficients, row, features) for row in exact]
    mean = sum(actual) / len(actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    ss_tot = sum((a - mean) ** 2 for a in actual)
    residuals = sorted(abs(a - p) for a, p in zip(actual, predicted))

    censored = [row for row in rows if row["censored"]]
    violations = sum(
        1
        for row in censored
        if predict(coefficients, row, features) < math.log10(max(1, row["nodes"]))
    )
    return {
        "n_uncensored": len(exact),
        "n_censored": len(censored),
        "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "residual_median_decades": residuals[len(residuals) // 2],
        "residual_p90_decades": residuals[int(0.9 * len(residuals))],
        "censored_underprediction_rate": violations / len(censored) if censored else 0.0,
    }


def compare_triggers(
    rows: list[dict], coefficients: list[float], features: tuple[str, ...], budget: int
) -> dict[str, Any]:
    """Card cap versus predicted cost, priced on the same positions.

    Both rules are scored on what they would have *spent* and what they would
    have *bought*, where buying means a solve that completes inside `budget`.
    Censored rows count as unbought at their observed floor, which is the
    charitable reading for the card cap: their true cost is higher still.
    """

    def outcome(row: dict) -> tuple[int, bool]:
        cost = max(1, row["nodes"])
        return cost, (not row["censored"]) and cost <= budget

    results: dict[str, Any] = {"budget": budget}
    caps = sorted({int(row["cards_left"]) for row in rows})
    best_cap = None
    for cap in caps:
        chosen = [row for row in rows if row["cards_left"] <= cap]
        spent = sum(outcome(row)[0] for row in chosen)
        bought = sum(1 for row in chosen if outcome(row)[1])
        entry = {"cap": cap, "attempts": len(chosen), "solved": bought, "nodes": spent}
        if best_cap is None or bought > best_cap["solved"]:
            best_cap = entry
    margin = math.log10(3.3)  # the study's ~0.51-decade residual
    chosen = [
        row
        for row in rows
        if predict(coefficients, row, features) + margin <= math.log10(budget)
    ]
    spent = sum(outcome(row)[0] for row in chosen)
    bought = sum(1 for row in chosen if outcome(row)[1])
    results["card_cap_best"] = best_cap
    results["cost_predicted"] = {
        "attempts": len(chosen),
        "solved": bought,
        "nodes": spent,
    }
    if best_cap and best_cap["solved"]:
        results["solved_ratio"] = bought / best_cap["solved"]
        results["cost_ratio"] = spent / best_cap["nodes"] if best_cap["nodes"] else None
    return results


def transfer(
    train_rows: list[dict], other_rows: list[dict], features: tuple[str, ...]
) -> dict[str, Any]:
    """Fit on one distribution, score on another.

    The question the shakedown cannot answer on its own: its endgames were
    reached by a 128x4 net playing itself, and the trigger will meet endgames
    reached by a trained one. If the cost model is really about board structure
    it should transfer; if it quietly learned "this is what a weak net's
    endgames look like", it will not.

    A model that scores well in-distribution and badly here is still useful --
    but only if refit per run, which is a different and more expensive proposal
    than the plan makes.
    """

    coefficients = fit([row for row in train_rows if not row["censored"]], features)
    return {
        "in_distribution": score(coefficients, train_rows, features),
        "transferred": score(coefficients, other_rows, features),
    }


def study_rows(path: Path) -> list[dict]:
    """Rows from `endgame_trigger_study --out`, normalised to this module's keys.

    The study writes `nodes: None` for a censored position; here `nodes` is
    always a number and `censored` says whether it is a cost or a floor.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    # A censored row's floor is the budget it exhausted, not zero. Normalising it
    # to zero (as this did at first) turns the most expensive positions in the
    # corpus into the cheapest: every "is it affordable" test then counts them as
    # trivially affordable, and the underprediction check can never flag one.
    budget = int(payload.get("study_nodes", 0))
    rows = []
    for row in payload["rows"]:
        censored = bool(row.get("censored"))
        if row.get("nodes") is None and not censored:
            continue
        normalised = dict(row)
        normalised["nodes"] = budget if row["nodes"] is None else row["nodes"]
        normalised["censored"] = censored
        rows.append(normalised)
    return rows


#: The shipped model, refit with `--fit-on` and rewritten by hand when the
#: corpus changes. Kept as data rather than constants so the Rust trigger and
#: the Python analysis cannot drift apart.
COST_MODEL_PATH = Path(__file__).with_name("endgame_cost_model.json")


def load_cost_model(path: Path | None = None) -> tuple[list[float], tuple[str, ...], float]:
    """Return `(coefficients, features, margin_decades)` for the shipped model.

    Coefficients are ordered intercept-first to match `_design`, so a caller can
    hand them straight to `predict` without knowing the file's layout.
    """

    payload = json.loads((path or COST_MODEL_PATH).read_text(encoding="utf-8"))
    features = tuple(payload["features"])
    coefficients = [float(payload["intercept"])] + [
        float(payload["coefficients"][name]) for name in features
    ]
    return coefficients, features, float(payload["margin_decades"])


def should_attempt(row: dict, budget: int, model=None) -> bool:
    """The trigger itself: is this position predicted to fit the budget?

    This is what replaces `cards_left <= cap`. On held-out cloud endgames it
    attempts 20% of 11-card positions and skips 4% of 8-card ones, which is the
    whole point -- a cap cannot do either.
    """

    coefficients, features, margin = model or load_cost_model()
    return predict(coefficients, row, features) + margin <= math.log10(budget)


def trigger_profile(
    rows: list[dict],
    coefficients: list[float],
    features: tuple[str, ...],
    budget: int,
    margin_decades: float,
) -> dict[str, Any]:
    """What the cost rule actually does, broken out by card count.

    The point of a cost-predicted trigger is not a better threshold on
    `cards_left` -- it is a rule that **crosses** card counts, attempting a cheap
    11 and skipping a dear 8. If it never does that, it is a card cap with extra
    steps and should be replaced by one.

    So this reports, per card count, the share attempted. A rule equivalent to a
    cap shows 100% down to some count and 0% below it; a genuinely variable rule
    shows intermediate shares on both sides of the boundary.
    """

    ceiling = math.log10(budget) - margin_decades
    by_cards: dict[int, dict[str, int]] = {}
    attempts = solved = spent = 0
    for row in rows:
        cards = int(row["cards_left"])
        bucket = by_cards.setdefault(cards, {"n": 0, "attempted": 0, "solved": 0})
        bucket["n"] += 1
        if predict(coefficients, row, features) > ceiling:
            continue
        bucket["attempted"] += 1
        attempts += 1
        cost = max(1, row["nodes"])
        # A censored row's true cost exceeds its floor, so it cannot have been
        # solved inside this budget unless the floor itself already fits.
        if not row["censored"] and cost <= budget:
            bucket["solved"] += 1
            solved += 1
            spent += cost
        else:
            spent += budget
    return {
        "margin_decades": margin_decades,
        "attempts": attempts,
        "solved": solved,
        "declines": attempts - solved,
        "decline_rate": (attempts - solved) / attempts if attempts else 0.0,
        "nodes": spent,
        "solves_per_mnode": solved / (spent / 1e6) if spent else 0.0,
        "by_cards": {
            str(cards): by_cards[cards] for cards in sorted(by_cards, reverse=True)
        },
    }


def best_cap(rows: list[dict], budget: int) -> dict[str, Any]:
    """The best fixed card cap on the same rows, priced the same way."""

    best = None
    for cap in sorted({int(row["cards_left"]) for row in rows}):
        chosen = [row for row in rows if row["cards_left"] <= cap]
        solved = spent = 0
        for row in chosen:
            cost = max(1, row["nodes"])
            if not row["censored"] and cost <= budget:
                solved += 1
                spent += cost
            else:
                spent += budget
        entry = {
            "cap": cap,
            "attempts": len(chosen),
            "solved": solved,
            "nodes": spent,
            "solves_per_mnode": solved / (spent / 1e6) if spent else 0.0,
        }
        if best is None or entry["solves_per_mnode"] > best["solves_per_mnode"]:
            best = entry
    return best


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "buffer", type=Path, nargs="?", help="a run's buffer_final.jsonl"
    )
    parser.add_argument("--budget", type=int, default=4_500_000)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--fit-on",
        type=Path,
        help="fit the model on an endgame_trigger_study --out file instead of a "
        "buffer, and report its coefficients. Use the cloud corpus here: the "
        "trigger meets endgames a trained net reaches, not a bot.",
    )
    parser.add_argument(
        "--transfer-to",
        type=Path,
        help="an endgame_trigger_study --out file whose positions the model, "
        "fit here, is then scored against. Use it to ask whether a cost model "
        "fit on self-play endgames predicts the endgames a trained net reaches.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.5,
        help="games held out by seed parity; the fit never sees them",
    )
    return parser


def report_fit(rows: list[dict], features: tuple[str, ...], holdout: float) -> dict:
    """Fit on held-out-by-game rows and report coefficients with their scores."""

    games = sorted({row.get("game", index) for index, row in enumerate(rows)})
    cut = int(len(games) * (1.0 - holdout))
    train_games = set(games[:cut])
    train = [row for row in rows if row.get("game") in train_games]
    test = [row for row in rows if row.get("game") not in train_games]
    coefficients = fit_censored(train, features)
    naive = fit([row for row in train if not row["censored"]], features)
    parsimonious = fit_censored(train, PLAN_FEATURES)
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "coefficients": dict(zip(("intercept",) + features, coefficients)),
        "censored_fit": score(coefficients, test, features),
        "survivors_only_fit": score(naive, test, features),
        "plan_four_features": {
            "coefficients": dict(zip(("intercept",) + PLAN_FEATURES, parsimonious)),
            **score(parsimonious, test, PLAN_FEATURES),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fit_on:
        rows = study_rows(args.fit_on)
        summary = {
            "source": str(args.fit_on),
            "rows": len(rows),
            "censored_fraction": sum(1 for r in rows if r["censored"]) / len(rows),
            **report_fit(rows, TRIGGER_FEATURES, args.holdout_fraction),
        }
        print(json.dumps(summary, indent=2))
        if args.out:
            args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0
    records = read_records(args.buffer)
    rows = list(solved_positions(records))
    if not rows:
        print("no solver attempts in this buffer")
        return 1

    # Split by GAME, not by row: adjacent solved plies in one game share a
    # position and a proof, so a row-wise split would leak the answer across it.
    seeds = sorted({row["seed"] for row in rows})
    cut = int(len(seeds) * (1.0 - args.holdout_fraction))
    train_seeds = set(seeds[:cut])
    train = [row for row in rows if row["seed"] in train_seeds]
    test = [row for row in rows if row["seed"] not in train_seeds]

    fit_rows = [row for row in train if not row["censored"]]
    coefficients = fit(fit_rows, TRIGGER_FEATURES)
    summary = {
        "buffer": str(args.buffer),
        "rows": len(rows),
        "games": len(seeds),
        "censored_fraction": sum(1 for row in rows if row["censored"]) / len(rows),
        "refit_on_shakedown": score(coefficients, test, TRIGGER_FEATURES),
        "triggers": compare_triggers(test, coefficients, TRIGGER_FEATURES, args.budget),
    }
    if args.transfer_to:
        other = study_rows(args.transfer_to)
        summary["transfer"] = {
            "source": str(args.transfer_to),
            "rows": len(other),
            **transfer(train, other, TRIGGER_FEATURES),
        }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
