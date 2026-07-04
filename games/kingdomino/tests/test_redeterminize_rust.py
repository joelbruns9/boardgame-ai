"""Redeterminization fires per search and leaks no hidden info — on the RUST
primitive the batched open-loop loop actually calls per simulation
(lib.rs: `slot.real_state.redeterminize(seed)`).

test_open_loop_mcts.py proves the PYTHON encoder is info-set safe across 100
determinizations. This checks the RUST path (RustGameState.redeterminize + Rust
encode) that production runs, and that redeterminize genuinely VARIES the deck
order (isn't a silent no-op).
"""
import random

import numpy as np
import pytest

from games.kingdomino.game import GameState, Phase
from games.kingdomino.endgame_solver import _rust_state_from_python


def _midgame_rust_state():
    st = GameState.new(seed=770001)
    rng = random.Random(7)
    for _ in range(14):
        if st.phase == Phase.GAME_OVER:
            break
        st = st.step(rng.choice(st.legal_actions()))
    return _rust_state_from_python(st)


def test_rust_redeterminize_fires_and_is_infoset_safe():
    pytest.importorskip("kingdomino_rust")
    rs = _midgame_rust_state()
    assert rs is not None

    actor = rs.current_actor()
    base_enc = [np.asarray(x).copy() for x in rs.encode(actor)]
    base_deck = list(rs.deck())
    assert len(base_deck) >= 4, "need a non-trivial hidden deck to test order"
    base_pub = (
        list(rs.current_row()),
        list(rs.board_terrain(0)), list(rs.board_terrain(1)),
        list(rs.board_crowns(0)), list(rs.board_crowns(1)),
        rs.scores(), rs.current_actor(), list(rs.pending_claims()),
    )

    N = 100
    order_changed = 0
    for seed in range(N):
        det = rs.redeterminize(seed)
        # A. encode byte-identical -> no hidden-order leak into the net input
        for a, b in zip(base_enc, det.encode(actor)):
            assert np.array_equal(a, np.asarray(b)), f"encode leaked at seed {seed}"
        # B. public bag multiset preserved
        d = list(det.deck())
        assert sorted(d) == sorted(base_deck)
        if d != base_deck:
            order_changed += 1
        # D. all public state unchanged
        det_pub = (
            list(det.current_row()),
            list(det.board_terrain(0)), list(det.board_terrain(1)),
            list(det.board_crowns(0)), list(det.board_crowns(1)),
            det.scores(), det.current_actor(), list(det.pending_claims()),
        )
        assert det_pub == base_pub

    # C. redeterminize actually fires (a frozen deck defeats info-set resampling)
    assert order_changed >= int(0.9 * N), f"deck order rarely changed ({order_changed}/{N})"


if __name__ == "__main__":
    test_rust_redeterminize_fires_and_is_infoset_safe()
    print("PASS: Rust redeterminize resamples the deck, fires, and leaks nothing.")
