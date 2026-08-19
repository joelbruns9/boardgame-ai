"""Every Nth game searched entirely at the full budget.

A diversity lever. `full_search_fraction` scatters full moves as independent
coin flips, so a game is a patchwork: a three-move plan needs all three plies
full, which is 6% of the time at f=0.25. A wholly full game is coherent end to
end -- and because `forced_playout_k` and `dirichlet_epsilon` are both gated on
`full`, it also carries root noise and forced exploration on every ply.

The choice is keyed on the GAME SEED, not a dispatch counter, so a game's depth
is a property of the game and survives sharding, resumption and start order.
"""

from __future__ import annotations

import pytest

from .rust_bridge import rust_games_for_self_play
from .test_f4_scheduler import _common, _row_eval

swr = pytest.importorskip("seven_wonders_rust")

# 0 and 4 qualify at every=4; 1, 2, 3, 5 do not.
SEEDS = [0, 1, 2, 3, 4, 5]
FIRST = [i % 2 for i in range(len(SEEDS))]


def _records(every: int, *, full_fraction: float = 0.25):
    def adapter(rows):
        return [_row_eval(tokens, actor, legal) for tokens, actor, legal in rows]

    records, _ = swr.self_play_many_net(
        adapter=adapter,
        games=rust_games_for_self_play(SEEDS, FIRST),
        game_seeds=SEEDS,
        full_search_every_games=every,
        **(_common(leaf_batch=1, global_batch_cap=8)
           | {"full_search_fraction": full_fraction}),
    )
    return {record["seed"]: record for record in records}


def _searched(record):
    return [move for move in record["moves"] if move["sims"] > 0]


def test_off_by_default_leaves_the_schedule_alone():
    """0 must reproduce the behaviour of every run before this existed."""

    assert _records(0) == _records(0)
    mixed = _records(0)[0]
    depths = {move["sims"] for move in _searched(mixed)}
    assert len(depths) > 1, "a mixed game should contain both depths"


def test_a_qualifying_game_searches_every_move_at_the_full_budget():
    """The property, stated as the config claims it: EVERY move, not most."""

    records = _records(4)
    full_sims = _common()["full_sims_min"], _common()["full_sims_max"]
    for move in _searched(records[0]):
        assert full_sims[0] <= move["sims"] <= full_sims[1], (
            "a qualifying game must have no cheap plies at all; a patchwork is "
            "exactly what this setting exists to avoid"
        )


def test_a_non_qualifying_game_is_untouched():
    """Seed 1 is not a multiple of 4, so it must be byte-identical to a run
    with the setting off. Otherwise the lever changes games it does not select,
    and its cost is not what the arithmetic says."""

    assert _records(4)[1] == _records(0)[1]
    assert _records(4)[2] == _records(0)[2]


def test_selection_is_by_seed_not_by_dispatch_order():
    """Sharding, resumption and slot start order must not move which games are
    full. Keyed on the seed, a game carries its own depth."""

    for shards in (1, 3):
        def adapter(rows):
            return [_row_eval(tokens, actor, legal) for tokens, actor, legal in rows]

        records, _ = swr.self_play_many_net(
            adapter=adapter,
            games=rust_games_for_self_play(SEEDS, FIRST),
            game_seeds=SEEDS,
            full_search_every_games=4,
            scheduler_workers=shards,
            max_active_slots=3,
            **(_common(leaf_batch=1, global_batch_cap=8)
               | {"full_search_fraction": 0.25}),
        )
        by_seed = {r["seed"]: r for r in records}
        assert all(m["sims"] >= 2 for m in _searched(by_seed[0]))
        assert all(m["sims"] >= 2 for m in _searched(by_seed[4]))


def test_a_qualifying_game_records_more_policy_rows():
    """The point of the lever, in the units training sees.

    Cheap plies are `policy_excluded`, so a mixed game contributes only its full
    moves as policy labels. A wholly full game contributes all of them.
    """

    def policy_rows(record):
        return [m for m in record["moves"] if not m["policy_excluded"]]

    assert len(policy_rows(_records(4)[0])) > len(policy_rows(_records(0)[0]))
