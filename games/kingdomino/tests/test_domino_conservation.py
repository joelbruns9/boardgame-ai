"""48-domino conservation ledger (engine transition gate).

The sparse-accumulator delta design assumes dominoes are conserved: every one of
the 48 is, at every reachable state, in exactly one place -- hidden deck, current
row, a next-round claim, an unresolved current-round claim, placed on a board, or
discarded. If an ID could silently vanish or duplicate across a transition, the
add/subtract deltas would be ill-defined. This validates that invariant directly
on the engine (independent of any encoder), with disproportionate coverage of the
two danger zones the reviewer flagged: round boundaries and forced discards.

Two checks per state:
  * disjoint + complete: the visible ID sets are pairwise disjoint and
    |visible| + sum(discards) == 48.
  * trajectory ledger: an ID leaves the visible set ONLY by being discarded
    (placed IDs stay visible on the board), so the cumulative disappeared set
    must always equal the engine's total discard count.
"""
import random

import numpy as np
import pytest

from games.kingdomino.game import GameState, Phase
from games.kingdomino.dominoes import DOMINOES

ALL_IDS = set(DOMINOES.keys())
assert len(ALL_IDS) == 48


def _placed_ids(state) -> set[int]:
    ids: set[int] = set()
    for b in state.boards:
        grid = np.asarray(b.domino_id)
        ids.update(int(v) for v in np.unique(grid[grid > 0]))
    return ids


def _visible_sources(state) -> list[set[int]]:
    """The five mutually-exclusive places an unplaced-or-placed domino can be.

    pending_claims keeps RESOLVED entries (index < actor_index) whose tiles are
    already on the board, so only the unresolved tail [actor_index:] counts here
    (the resolved ones are captured by _placed_ids).
    """
    return [
        set(state.deck),
        set(state.current_row),
        {c.domino_id for c in state.next_claims},
        {c.domino_id for c in state.pending_claims[state.actor_index:]},
        _placed_ids(state),
    ]


def _check_state(state, cumulative_gone: set[int]):
    sources = _visible_sources(state)
    total = sum(len(s) for s in sources)
    visible = set().union(*sources)
    # disjoint: no ID appears in two places (a duplicate would make total > |union|)
    assert total == len(visible), (
        f"domino in two places at once: sum(sizes)={total} != |union|={len(visible)}"
    )
    n_disc = sum(state.discards)
    # complete: nothing has silently vanished; the only missing IDs are discards.
    assert len(visible) + n_disc == 48, (
        f"conservation broken: |visible|={len(visible)} + discards={n_disc} != 48"
    )
    # every visible ID is a real catalog ID
    assert visible <= ALL_IDS
    # trajectory ledger: disappeared-from-visible == discarded.
    gone = ALL_IDS - visible
    cumulative_gone |= gone
    assert cumulative_gone == gone, "an ID reappeared after leaving the visible set"
    assert len(gone) == n_disc, (
        f"ledger mismatch: {len(gone)} IDs gone but discards={n_disc}"
    )


def test_all_48_dominoes_conserved_across_transitions():
    saw_discard = False
    saw_round_boundary = False
    saw_final_phase = False
    n_states = 0

    for seed in range(120):
        st = GameState.new(seed=seed)
        rng = random.Random(seed * 11 + 5)
        cumulative_gone: set[int] = set()
        _check_state(st, cumulative_gone)  # initial state
        prev_discards = sum(st.discards)
        prev_phase = st.phase

        while st.phase != Phase.GAME_OVER:
            st = st.step(rng.choice(st.legal_actions()))
            _check_state(st, cumulative_gone)
            n_states += 1
            if sum(st.discards) > prev_discards:
                saw_discard = True
                prev_discards = sum(st.discards)
            if st.phase != prev_phase:
                if st.phase == Phase.PLACE_AND_SELECT:
                    saw_round_boundary = True
                if st.phase == Phase.FINAL_PLACEMENT:
                    saw_final_phase = True
                prev_phase = st.phase

    assert n_states > 1000
    # danger zones must actually be exercised, not just assumed.
    assert saw_round_boundary, "no round boundary (next->pending promotion) seen"
    assert saw_final_phase, "no FINAL_PLACEMENT phase seen"
    assert saw_discard, "no forced discard seen; discard conservation path untested"
    print(f"states checked={n_states}; round-boundaries, final-phase, discards all covered")


if __name__ == "__main__":
    test_all_48_dominoes_conserved_across_transitions()
    print("PASS: all 48 dominoes conserved across every transition.")
