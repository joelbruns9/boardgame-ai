"""How much of the cloud buffers still replays under the fixed engine?

The cloud runs are the only large corpus of 7WD endgames produced by something
stronger than a bot, which makes them the right place to price the endgame
solver -- bot endgames are not the endgames a trained net reaches.

They were generated before the military off-by-one fix, so they were played
under different rules and `buffer.replay` rejects them. The rejection is not
uniform, and the distinction decides whether anything is salvageable:

* **Final-digest-only divergence** -- every move's legal mask and actor matched
  the whole way through, and only the terminal state differs. The trajectory is
  reproducible; the recorded game is the game.
* **Mask-hash divergence** -- the legal move set differed at some ply, so the
  old engine offered a choice the new one does not (or vice versa). Every
  recorded action *after* that point was chosen for a position that no longer
  exists, so replaying past it produces legal-but-arbitrary play, not strong
  play. That is precisely the property the corpus was wanted for.

So a position is usable iff it precedes the first divergence. This measures how
many endgame positions survive that rule, because a salvage that keeps only
opening positions is no salvage at all -- the solver only runs near the end.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from .buffer import (
    FinalDigestMismatchError,
    GameRecord,
    ReplayMismatchError,
    StaleSpecVersionError,
    read_records,
    replay,
)


def divergence_point(record: GameRecord) -> tuple[int | None, int]:
    """Return `(first divergent move index, total moves)` for one record.

    `None` means every mask and actor matched to the end -- the final state may
    still differ, which does not affect any position before it.

    Implemented on top of `buffer.replay` rather than beside it. An earlier
    version walked the moves directly and reported divergence at ply 0 for every
    game, because it applied actions without injecting the recorded chance
    outcomes and so re-randomised the first deal. `replay` owns that logic; its
    `on_state` hook fires after each move's checks pass and before the move is
    applied, so the number of hook calls before it raises is the number of
    verified positions.
    """

    seen = [0]

    def count(game, move, seen=seen):
        seen[0] += 1

    try:
        replay(record, on_state=count)
    except FinalDigestMismatchError:
        # Trajectory intact; only the terminal score differs.
        return None, len(record.moves)
    except ReplayMismatchError:
        return seen[0], len(record.moves)
    except StaleSpecVersionError:
        return 0, len(record.moves)
    return None, len(record.moves)


def survey(records: list[GameRecord], solver_cards: int) -> dict[str, Any]:
    """Positions that survive, and how many of them are endgame positions.

    `solver_cards` is the card-count threshold the solver would fire at; a
    surviving position only matters here if the solver would have looked at it.
    """

    kinds: Counter[str] = Counter()
    usable_plies = 0
    total_plies = 0
    endgame_usable = 0
    endgame_total = 0
    divergence_depth: list[float] = []

    for record in records:
        first, total = divergence_point(record)
        total_plies += total
        # A move is an endgame move if it is within `solver_cards` plies of the
        # end -- a cheap proxy for cards remaining that needs no board walk, and
        # conservative: real endgames start no earlier than this.
        endgame_start = max(0, total - solver_cards)
        endgame_total += total - endgame_start
        if first is None:
            kinds["trajectory_intact"] += 1
            usable_plies += total
            endgame_usable += total - endgame_start
        else:
            kinds["diverged"] += 1
            usable_plies += first
            endgame_usable += max(0, first - endgame_start)
            divergence_depth.append(first / total if total else 0.0)

    depth = sorted(divergence_depth)
    return {
        "games": len(records),
        "trajectory_intact": kinds["trajectory_intact"],
        "diverged": kinds["diverged"],
        "intact_fraction": kinds["trajectory_intact"] / len(records) if records else 0,
        "plies_total": total_plies,
        "plies_usable": usable_plies,
        "endgame_plies_total": endgame_total,
        "endgame_plies_usable": endgame_usable,
        "endgame_survival": endgame_usable / endgame_total if endgame_total else 0.0,
        "divergence_depth_median": depth[len(depth) // 2] if depth else None,
        "divergence_depth_p10": depth[int(0.1 * len(depth))] if depth else None,
    }


def iter_buffers(paths: list[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.glob("*.jsonl"))
        else:
            yield path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("buffers", type=Path, nargs="+")
    parser.add_argument(
        "--solver-cards",
        type=int,
        default=10,
        help="plies from the end that count as endgame (the solver's trigger)",
    )
    parser.add_argument("--limit-per-file", type=int, default=200)
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = {}
    for path in iter_buffers(args.buffers):
        records = read_records(path)[: args.limit_per_file]
        if not records:
            continue
        results[path.name] = survey(records, args.solver_cards)
        print(f"{path.name}: {json.dumps(results[path.name])}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
