"""A plain-data snapshot of a ``GameState``, shared with the Rust engine.

``cantstop_rust.GameState.snapshot()`` returns exactly this shape, so the M1
equivalence gate compares the two engines with a bare ``==``. It is also the
way a position crosses between them: ``from_snapshot`` here and
``cantstop_rust.GameState.from_snapshot`` load the same tuple.

Shape, one 8-tuple:

    ((num_players, columns_to_win, blocking),
     active_player,
     progress,      list per player of 11 ints, columns 2..12 in order
     claimed_by,    list of 11 ints, -1 for unclaimed
     runners,       list of (col, pos) tuples, ascending col
     dice,          list of 4 sorted ints, or None
     phase,         int, Phase's value
     winner)        int, -1 for none

Lists, not tuples, for every variable-length field: that is what pyo3 turns a
Rust ``Vec`` into, and ``[1, 2] == (1, 2)`` is False. Runners are sorted by
column because Python's dict order is insertion order and no rule depends on
it -- the Rust engine keeps no order at all.
"""

from games.cantstop.engine import COLUMNS, GameState, Phase, RuleSet


def snapshot(state):
    r = state.rules
    return (
        (r.num_players, r.columns_to_win, r.blocking),
        state.active_player,
        [[prog[c] for c in COLUMNS] for prog in state.progress],
        [-1 if state.claimed_by[c] is None else state.claimed_by[c]
         for c in COLUMNS],
        sorted(state.runners.items()),
        None if state.dice is None else list(state.dice),
        int(state.phase),
        -1 if state.winner is None else state.winner,
    )


def from_snapshot(snap):
    """Rebuild a ``GameState``. No reachability check: constructed positions
    for the gate are allowed to be ones no game would reach."""
    (n, cols, blocking), active, progress, claimed, runners, dice, phase, \
        winner = snap
    state = GameState(RuleSet(n, cols, bool(blocking)))
    state.active_player = active
    state.progress = [dict(zip(COLUMNS, row)) for row in progress]
    state.claimed_by = {c: (None if v == -1 else v)
                        for c, v in zip(COLUMNS, claimed)}
    state.runners = dict(runners)
    state.dice = None if dice is None else tuple(sorted(dice))
    state.phase = Phase(phase)
    state.winner = None if winner == -1 else winner
    return state
