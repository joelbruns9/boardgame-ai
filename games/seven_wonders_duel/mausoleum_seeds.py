"""Seed the G12 restart archive with Mausoleum science positions from an
earlier run's buffers (owner decision 2026-10-08).

run07 inventory (all 101 buffers, 101k games): 320 Mausoleum science wins
(3.2 per 1,000, flat over the run; science specialists 3.8 vs general 3.0),
610 games reach a SETUP -- a player holds The Mausoleum unbuilt, five distinct
science symbols, and a card with the sixth in the discard. Self-play does not
make these positions more often as it improves, so a fresh run would see a few
per iteration at best. This builds one restart entry per setup game, a
uniformly drawn ``0..window`` plies before the game's first setup position (a
backward curriculum: before the threat exists as well as at it), with the
historical move there marked tried -- exactly what `restart_archive.harvest`
does for decisive positions.

Entries are ``seeded``: they never age out, and phase_d restarts at most
``--restart-seed-per-iteration`` of them per iteration so the pool lasts.

    python -m games.seven_wonders_duel.mausoleum_seeds \\
        runs/seven_wonders_duel/run07_bundle/buffers/iter_*.jsonl \\
        --out games/seven_wonders_duel/seeds/mausoleum_run07.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import random

from .buffer import GameRecord, read_records, replay
from .codec import legal_action_indices
from .data import CARDS_BY_NAME
from .engine import _science_symbols
from .game import GameState, Phase
from .restart_archive import DEFAULT_WINDOW, Archive, Entry

MAUSOLEUM = "The Mausoleum"


def sixth_symbol_discards(game: GameState, player: int) -> list[str]:
    """Discarded cards that would give ``player`` a science victory."""

    symbols = _science_symbols(game, player)
    if len(symbols) != 5:
        return []
    return [
        name for name in game.discard_pile
        if CARDS_BY_NAME[name].science is not None
        and CARDS_BY_NAME[name].science not in symbols
    ]


def is_setup(game: GameState) -> bool:
    """Either player holds The Mausoleum unbuilt with a sixth symbol in the discard."""

    if game.phase is not Phase.PLAY_AGE:
        return False
    for player in (0, 1):
        city = game.cities[player]
        if (
            MAUSOLEUM in city.wonders
            and MAUSOLEUM not in city.built_wonders
            and sixth_symbol_discards(game, player)
        ):
            return True
    return False


def seed_entry(record: GameRecord, *, window: int = DEFAULT_WINDOW, seed: int = 0) -> Entry | None:
    """The record's restart entry before its first setup position, or None."""

    if record.agents.get("opponent_type") == "bot" or record.restart_from:
        return None
    info: dict = {}
    first = {"setup": None}
    consumed = {"events": 0}

    def visit(state, move) -> None:
        info[move.i] = (
            consumed["events"],
            len(legal_action_indices(state)),
            state.phase is Phase.PLAY_AGE,
        )
        if first["setup"] is None and is_setup(state):
            first["setup"] = move.i

    def count(_move, events) -> None:
        consumed["events"] += len(events)

    replay(record, on_state=visit, on_events=count)
    setup = first["setup"]
    if setup is None:
        return None
    rng = random.Random(f"mausoleum:{seed}:{record.iteration}:{record.seed}")
    back = rng.randint(0, window)
    # Walk forward to the nearest ply with a choice to branch on (age play only).
    for ply in range(max(0, setup - back), setup + 1):
        events, legal, in_age = info[ply]
        if in_age and legal >= 2:
            break
    else:
        return None
    return Entry(
        seed=record.replay_seed,
        first_player=record.first_player,
        prefix=[(m.action, m.actor, m.mask_hash) for m in record.moves[:ply]],
        chance_prefix=[
            [kind, list(outcome) if isinstance(outcome, tuple) else outcome]
            for kind, outcome in record.chance_log[:events]
        ],
        tried=[record.moves[ply].action],
        born=0,
        source=[record.iteration, setup, setup - ply],
        legal_count=legal,
        family=(
            list(record.family) if record.family is not None
            else [record.iteration, record.seed]
        ),
        ancestor_winner=record.winner,
        seeded=True,
    )


def _scan(args) -> tuple[list[Entry], Counter]:
    path, window, seed = args
    entries, counts = [], Counter()
    for record in read_records(path):
        counts["games"] += 1
        entry = seed_entry(record, window=window, seed=seed)
        if entry is not None:
            entries.append(entry)
            counts["entries"] += 1
    return entries, counts


def build(paths, *, window: int = DEFAULT_WINDOW, seed: int = 0, workers: int = 8):
    entries: list[Entry] = []
    counts: Counter = Counter()
    jobs = [(Path(p), window, seed) for p in paths]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for found, c in pool.map(_scan, jobs):
            entries += found
            counts.update(c)
    return entries, counts


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("buffers", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    entries, counts = build(sorted(args.buffers), window=args.window, seed=args.seed,
                            workers=args.workers)
    archive = Archive(seed_source=Path(args.out).name)
    added = archive.add(entries)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    archive.save(out)
    print(f"{counts['games']} games -> {len(entries)} setup entries ({added} distinct) -> {out}")


if __name__ == "__main__":
    main()
