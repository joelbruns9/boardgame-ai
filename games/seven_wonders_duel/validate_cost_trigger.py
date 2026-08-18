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
TRIGGER_FEATURES = tuple(
    name for name in FEATURES if name not in UNAVAILABLE_AT_DECISION_TIME
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
    rows = []
    for row in payload["rows"]:
        if row.get("nodes") is None and not row.get("censored"):
            continue
        normalised = dict(row)
        normalised["nodes"] = row["nodes"] if row["nodes"] is not None else 0
        normalised["censored"] = bool(row.get("censored"))
        rows.append(normalised)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("buffer", type=Path, help="a run's buffer_final.jsonl")
    parser.add_argument("--budget", type=int, default=4_500_000)
    parser.add_argument("--out", type=Path)
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
