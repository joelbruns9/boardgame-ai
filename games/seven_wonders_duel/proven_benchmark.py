"""A frozen set of endgame positions with PROVEN values, and a scorer for them.

Every value metric this project has is self-referential. Search agreement scores
the net against its own search; net-vs-prior KL scores it against an earlier
copy of itself; outcome accuracy scores it against games its own errors decided;
a gate scores it against another net that may be wrong the same way. None can
say the net is *wrong* -- only that it changed. That is why cloud6 could spend
38k games reaching no promotion and leave "optimization or capacity" open.

An exact endgame solve is a proof, so a solved position is a labelled test case
with no label noise at all -- the only ground truth available here. This module
freezes a set of them and scores any checkpoint against it with **one forward
pass per position**: no search, no games, seconds per evaluation.

Two design choices carry most of the value:

* **Paired.** Every net sees identical positions, so position difficulty cancels
  when arms are compared. Letting each net play its own games -- the obvious
  alternative -- destroys the pairing and most of the statistical power.
* **Prior-only.** No search runs. Measured 2026-08-17, search reaches proven
  values at 0.096 mean error against the raw net's 0.221, and 98% sign agreement
  against 92%: the search is not the weak link, the prior is. Scoring the prior
  alone measures exactly the term that is failing.

Sizing is not incidental. 133 positions with a residual p90 near 0.92 give a
standard error around 0.026, which cannot separate 0.221 from 0.19 -- the size of
difference actually at stake. A thousand brings that near 0.0095.

**Every position here is fully revealed.** That is forced, not chosen: only an
`exact` proof gives a value a three-class head can be scored against, and a solve
is exact exactly when it crosses no chance edge -- no card left face down. So the
benchmark measures the value head on *deterministic* endgames. Positions whose
outcome still depends on the deal are out of scope for any proven-value
instrument, and remain the harder case.

What this is NOT: a strength measurement. It scores value accuracy on endgames,
not play, and not the opening or midgame. `PRE_RETRAIN_PLAN.md` §C is explicit
that strength at a fixed deployment search budget is the only number that answers
the real question and everything else, this included, is a proxy for it.

Usage::

    # freeze a benchmark from strong-play buffers (needs no checkpoint)
    python -m games.seven_wonders_duel.proven_benchmark build \\
        --buffers "runs/cloud runs/cloud6_capture/cloud6/buffers" \\
        --target 1000 --out testdata/proven_endgames.jsonl

    # score any checkpoint against it
    python -m games.seven_wonders_duel.proven_benchmark score \\
        --benchmark testdata/proven_endgames.jsonl --checkpoint <path>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterator

import torch

from .buffer import GameRecord, ReplayMismatchError, read_records, replay
from .game import Phase

#: Positions are stored as `(seed, first_player, action prefix)` rather than a
#: serialised board: the engine reconstructs them exactly and the file stays
#: small enough to commit, which is what "frozen" requires. The source buffers
#: are gitignored and hundreds of megabytes.
SCHEMA = 1


def _instant_win_threat(game) -> bool:
    """Is either side one step from a military or scientific instant win?

    The bucket exists so the mean/max readout can be tested against its actual
    claim -- that pooling helps with *existentials*, "does a threat exist" -- and
    not merely on an aggregate where such positions are a minority and any effect
    washes out.
    """

    if abs(game.conflict_position) >= 8:
        return True
    return any(
        len({c.science for c in _buildings(game, seat) if c.science is not None}) >= 5
        for seat in (0, 1)
    )


def _buildings(game, seat: int):
    from .data import CARDS_BY_NAME

    return [CARDS_BY_NAME[name] for name in game.cities[seat].buildings]


def solved_endgames(
    records: list[GameRecord],
    *,
    source: str,
    max_nodes: int,
    max_secs: float,
    max_cards: int,
    per_game_cap: int,
    quota: int,
    found: list[dict],
    cost_model=None,
) -> dict[str, int]:
    """Solve Age III positions and bank the ones that come back proven.

    Only `exact` proofs are banked. An `exact_expectimax` solve returns a scalar
    `P(win) - P(loss)`, which a three-class value head cannot be scored against
    when draws exist -- the same reason those rows supply no value target during
    training. Banking them would put label noise into the one instrument that is
    supposed to have none.

    `per_game_cap` matters more than it looks: adjacent solved plies in one game
    share a position and a proof, so an uncapped scan would fill the benchmark
    with near-duplicates and report a sample size it does not have.
    """

    from .rust_bridge import rust_game_from_state

    stats = {
        "games": 0,
        "replay_mismatches": 0,
        "candidates": 0,
        "declined": 0,
        "skipped_by_cost_model": 0,
        "skipped_chance": 0,
    }
    for game_index, record in enumerate(records):
        if len(found) >= quota:
            break
        hits: list[dict] = []

        def on_state(game, move, _gi=game_index, _rec=record):
            if len(found) + len(hits) >= quota or len(hits) >= per_game_cap:
                return
            if game.phase is not Phase.PLAY_AGE or game.age != 3:
                return
            cards = [c for c in game.tableau.cards.values() if c.present]
            present = len(cards)
            if present > max_cards:
                return
            # Only `exact` proofs are bankable, and a solve is exact exactly when
            # it crosses no chance edge -- which means no card left face down.
            # Measured over 77 solved positions the separation is total: 26/26
            # with nothing unrevealed came back `exact`, and 0 of 51 with
            # something unrevealed did. Filtering here rather than discarding the
            # answer afterwards is what makes the build affordable: it was paying
            # a full solve for 45 expectimax results and 6 declines to bank 26.
            if any(not card.revealed for card in cards):
                stats["skipped_chance"] += 1
                return
            stats["candidates"] += 1
            # Pre-filter with the fitted cost model. Without it the build spends
            # most of its time on solves that then decline: at nine cards the
            # median position costs 668k nodes but the p90 costs 10.7M, so a
            # blind scan pays the full budget for the tail over and over. This is
            # exactly what the trigger exists to avoid, used here for its own
            # benefit.
            if cost_model is not None:
                from .endgame_trigger_study import position_features
                from .validate_cost_trigger import should_attempt

                if not should_attempt(position_features(game), max_nodes, cost_model):
                    stats["skipped_by_cost_model"] += 1
                    return
            answer = rust_game_from_state(game).solve_endgame(
                max_nodes, max_secs, "value_only", "star1"
            )
            if answer is None or answer["regime"] != "exact":
                stats["declined"] += 1
                return
            hits.append(
                {
                    "id": f"{source}:{_gi}:{move.i}",
                    "source": source,
                    "game_seed": _rec.seed,
                    "first_player": _rec.first_player,
                    "move_index": move.i,
                    "prefix": [m.action for m in _rec.moves[: move.i]],
                    "actor": move.actor,
                    "value": float(answer["root_value"]),
                    "cards_left": present,
                    "instant_win_threat": bool(_instant_win_threat(game)),
                    "nodes": int(answer["nodes"]),
                }
            )

        try:
            replay(record, on_state=on_state)
            stats["games"] += 1
        except ReplayMismatchError as error:
            # A FINAL-DIGEST mismatch is expected and harmless here: the
            # pre-military-fix buffers score their terminal state differently,
            # and every position banked above precedes it. A mask mismatch is
            # not -- the trajectory diverged, so the recorded actions after that
            # point were chosen for a position that no longer exists.
            #
            # Getting this wrong is silent and total. The first version treated
            # both alike, so every cloud game hit the `continue`, every solved
            # position was discarded after being paid for, and the build simply
            # never reached its quota.
            if "final digest" not in str(error):
                stats["replay_mismatches"] += 1
                continue
            stats["games"] += 1
        found.extend(hits[: max(0, quota - len(found))])
    return stats


#: The committed benchmark. Frozen on purpose: an instrument that moves cannot
#: compare two runs, and comparability across arms is most of its value.
BENCHMARK_PATH = Path(__file__).parent / "testdata" / "proven_endgames.jsonl"


def load_benchmark(path: Path | None = None) -> list[dict]:
    """Rows from a benchmark file, without its header line."""

    text = (path or BENCHMARK_PATH).read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [row for row in rows if "value" in row]


def _sign_agreement(values, truths, *, with_count: bool = False):
    """Fraction of DECIDED positions where the net has the winner right.

    Drawn proofs are excluded rather than scored. A draw has no sign, so
    counting one either way is an arbitrary hit or miss that moves the metric
    without measuring anything.
    """

    pairs = [(v, t) for v, t in zip(values, truths) if t != 0.0]
    if not pairs:
        return (float("nan"), 0) if with_count else float("nan")
    hits = sum(1 for value, truth in pairs if (value > 0) == (truth > 0))
    agreement = hits / len(pairs)
    return (agreement, len(pairs)) if with_count else agreement


def rebuild_position(row: dict):
    """Reconstruct a banked position from its `(seed, first_player, prefix)`.

    Verified rather than trusted: the actor is re-derived and checked, so a
    benchmark file that has drifted from the engine fails loudly instead of
    quietly scoring a different position than it was solved at.
    """

    from .buffer import new_game
    from .codec import decode_action
    from .engine import apply_action

    game = new_game(int(row["game_seed"]), first_player=int(row["first_player"]))
    for action_index in row["prefix"]:
        apply_action(game, decode_action(game, action_index))
    actor = (
        game.pending_choice.player
        if game.pending_choice is not None
        else game.active_player
    )
    if actor != row["actor"]:
        raise ValueError(
            f"position {row['id']} rebuilt with actor {actor}, banked as "
            f"{row['actor']}: the benchmark no longer matches this engine"
        )
    return game, actor


def score_checkpoint(
    rows: list[dict],
    checkpoint: Path,
    device: str,
    precision: str,
    batch: int = 256,
) -> dict[str, Any]:
    """Mean |error|, sign agreement and buckets, at one forward pass each.

    The value head is read for the position's ACTOR, and the proof is
    actor-relative, so no sign convention is inferred from the board.
    """

    from .inference import Evaluator
    from .train import load_checkpoint, model_from_config

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = model_from_config(payload.get("config", {}))
    load_checkpoint(checkpoint, model, checkpoint=payload)
    evaluator = Evaluator(model, device, precision=precision)

    # Batched: the whole point is that scoring costs one forward per position, so
    # feeding them one at a time would throw away most of the speed.
    games = [rebuild_position(row)[0] for row in rows]
    values: list[float] = []
    for start in range(0, len(games), batch):
        for evaluation in evaluator.evaluate_states(games[start : start + batch]):
            # The solver's scalar is actor-relative P(win) - P(loss); the value
            # head's W/D/L is actor-relative too, so this is the same quantity
            # and no sign convention has to be inferred from the board.
            values.append(float(evaluation.wdl[0] - evaluation.wdl[2]))

    truths = [float(row["value"]) for row in rows]
    agreement, decisive = _sign_agreement(values, truths, with_count=True)

    errors: list[float] = []
    buckets: dict[str, list[float]] = {}
    for row, value, truth in zip(rows, values, truths):
        error = abs(value - truth)
        errors.append(error)
        key = "threat" if row["instant_win_threat"] else "quiet"
        buckets.setdefault(key, []).append(error)
        buckets.setdefault(f"cards_{row['cards_left']}", []).append(error)

    ordered = sorted(errors)
    mean = sum(errors) / len(errors)
    variance = sum((e - mean) ** 2 for e in errors) / max(1, len(errors) - 1)
    return {
        "checkpoint": str(checkpoint),
        "positions": len(rows),
        "mean_abs_error": mean,
        # The number that decides whether an arm difference is resolvable.
        "standard_error": math.sqrt(variance / len(errors)),
        "p90_abs_error": ordered[int(0.9 * len(ordered))],
        "sign_agreement": agreement,
        "decisive_positions": decisive,
        "buckets": {
            name: {"n": len(values), "mean_abs_error": sum(values) / len(values)}
            for name, values in sorted(buckets.items())
        },
    }


def iter_buffer_files(paths: list[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.glob("*.jsonl"))
        else:
            yield path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="freeze a benchmark from buffer files")
    build.add_argument("--buffers", type=Path, nargs="+", required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--target", type=int, default=1000)
    build.add_argument("--max-cards", type=int, default=11)
    build.add_argument("--max-nodes", type=int, default=40_000_000)
    build.add_argument("--max-secs", type=float, default=600.0)
    build.add_argument("--no-cost-model", action="store_true")
    build.add_argument(
        "--cost-margin",
        type=float,
        default=0.1,
        help="safety margin for the pre-filter, in decades of predicted cost",
    )
    build.add_argument(
        "--per-game-cap",
        type=int,
        default=2,
        help="adjacent solved plies share one proof; more than a couple per game "
        "inflates the sample size without adding evidence",
    )

    score = sub.add_parser("score", help="score a checkpoint against a benchmark")
    score.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    score.add_argument("--precision", default="fp32")
    score.add_argument("--batch", type=int, default=256)
    score.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        cost_model = None
        if not args.no_cost_model:
            from .validate_cost_trigger import load_cost_model

            coefficients, features, margin = load_cost_model()
            # A tighter margin than generation uses: a build can afford to try a
            # few that fail, and every extra proof is a permanent asset, whereas
            # in generation a declined solve delays a game.
            cost_model = (coefficients, features, args.cost_margin)
            print(
                f"cost-model pre-filter on, margin {args.cost_margin} "
                f"(generation ships {margin})"
            )
        found: list[dict] = []
        totals: dict[str, int] = {}
        for path in iter_buffer_files(args.buffers):
            if len(found) >= args.target:
                break
            stats = solved_endgames(
                read_records(path),
                source=path.stem,
                cost_model=cost_model,
                max_nodes=args.max_nodes,
                max_secs=args.max_secs,
                max_cards=args.max_cards,
                per_game_cap=args.per_game_cap,
                quota=args.target,
                found=found,
            )
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + value
            print(f"{path.name}: {len(found)} banked  {json.dumps(stats)}", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"schema": SCHEMA, "stats": totals}) + "\n")
            for row in found:
                handle.write(json.dumps(row) + "\n")
        print(f"wrote {args.out} ({len(found)} positions)")
        return 0

    rows = load_benchmark(args.benchmark)
    summary = score_checkpoint(
        rows, args.checkpoint, args.device, args.precision, args.batch
    )
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
