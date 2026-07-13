"""Step 2 equivalence gate: the make/unmake-driven `RustExpectiminimax` must
return the SAME search value as the pure-Python `ExpectiminimaxBot` whenever all
chance nodes in the horizon are enumerated (deterministic). This validates the
whole engine — `make_with_row` chance expansion, `official_outcome` terminals,
alpha-beta over make/unmake — against the known-good reference, under real search
recursion rather than isolated unit assertions.

Late states (small deck) are used so C(n,4) <= enum_cap everywhere in-horizon and
neither searcher falls back to Monte-Carlo sampling.
"""
from __future__ import annotations

import random
from math import inf

import kingdomino_rust as kr

from games.kingdomino.game import GameState, Phase
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.expectiminimax import (
    ExpectiminimaxBot,
    tanh_margin_p0,
    pick_aware_p0,
)
from games.kingdomino.rust_expectiminimax import (
    RustExpectiminimax,
    tanh_margin as r_tanh_margin,
    pick_aware as r_pick_aware,
)

EVAL_PAIRS = [
    ("pick_blind", tanh_margin_p0, r_tanh_margin),
    ("pick_aware", pick_aware_p0, r_pick_aware),
]


def _wide_boundary_py(min_deck=8):
    """A Python PLACE_AND_SELECT round-boundary state (next move deals) with a bag
    large enough that C(deck,4) exceeds a small enum_cap → Monte-Carlo sampled."""
    for seed in range(400):
        st = GameState.new(seed=seed)
        rng = random.Random(seed * 3 + 2)
        for _ in range(400):
            if st.phase == Phase.GAME_OVER:
                break
            if (
                st.phase == Phase.PLACE_AND_SELECT
                and len(st.deck) >= min_deck
                and st.actor_index == len(st.pending_claims) - 1
            ):
                return st, st.legal_actions()[0]
            st = st.step(rng.choice(st.legal_actions()))
    raise AssertionError("no wide-boundary state found")


def test_sampled_chance_rows_match_reference():
    """P2: with chance forced to Monte-Carlo (enum_cap=1), the Rust-backed search
    samples byte-identical rows to the pure-Python reference — same blake2 stable
    seed and RNG-draw sequence. The enumerated equivalence test cannot see this;
    a direct sampled-row signature does."""
    st, action = _wide_boundary_py()
    rs = _rust_state_from_python(st)
    assert rs is not None
    for seed in (0, 1, 7):
        for cs in (4, 8):
            py_bot = ExpectiminimaxBot(depth=2, chance_samples=cs, enum_cap=1, seed=seed)
            r_bot = RustExpectiminimax(depth=2, chance_samples=cs, enum_cap=1, seed=seed)
            py_rows = sorted(
                tuple(child.current_row) for child, _ in py_bot._expand_chance(st, action)
            )
            r_rows = sorted(
                tuple(row) for row, _ in r_bot._chance_rows(kr.SearchEngine(rs))
            )
            assert len(py_rows) == cs and len(r_rows) == cs, "expected MC sampling"
            assert py_rows == r_rows, f"seed {seed} cs {cs}: sampled rows differ"


def _late_states(max_deck=8, count=8):
    """PLACE_AND_SELECT states with a small enough bag that every in-horizon
    chance node enumerates (C(<=8,4)=70 <= default enum_cap)."""
    out = []
    seed = 0
    while len(out) < count and seed < 400:
        st = GameState.new(seed=seed)
        rng = random.Random(seed * 7 + 1)
        for _ in range(400):
            if st.phase == Phase.GAME_OVER:
                break
            if (
                st.phase == Phase.PLACE_AND_SELECT
                and 4 <= len(st.deck) <= max_deck
                and len(st.legal_actions()) >= 2
            ):
                out.append(st)
                break
            st = st.step(rng.choice(st.legal_actions()))
        seed += 1
    return out


def test_search_unwinds_engine_on_evaluator_exception():
    """P2: a raising (caller-supplied) eval_fn must unwind the shared engine's
    undo stack via try/finally, leaving it at depth 0 and still usable — not
    stranded mid-mutation."""
    import pytest

    states = _late_states()
    rs = _rust_state_from_python(states[0])
    eng = kr.SearchEngine(rs)
    assert eng.depth() == 0

    def boom(_eng):
        raise RuntimeError("evaluator boom")

    bot = RustExpectiminimax(depth=3, enum_cap=128, eval_fn=boom)
    with pytest.raises(RuntimeError):
        bot.value(eng, 3)
    assert eng.depth() == 0, "engine left mutated after evaluator raised"

    # Engine is still usable afterwards.
    ok = RustExpectiminimax(depth=2, enum_cap=128, eval_fn=r_pick_aware)
    ok.value(eng, 2)
    assert eng.depth() == 0


def test_rust_search_value_matches_python_reference():
    states = _late_states()
    assert len(states) >= 5, f"too few late states ({len(states)})"
    checks = 0
    for pystate in states:
        rs = _rust_state_from_python(pystate)
        assert rs is not None
        for name, py_eval, r_eval in EVAL_PAIRS:
            for depth in (2, 3):
                py_bot = ExpectiminimaxBot(depth=depth, enum_cap=128, eval_fn=py_eval)
                r_bot = RustExpectiminimax(depth=depth, enum_cap=128, eval_fn=r_eval)
                pv = py_bot._value(pystate, depth, -inf, inf)
                eng = kr.SearchEngine(rs)
                rv = r_bot.value(eng, depth)
                assert abs(pv - rv) < 1e-9, (
                    f"{name} d{depth}: rust {rv} != python {pv} (delta {rv - pv:.2e})"
                )
                checks += 1
    assert checks >= 20, f"too few equivalence checks ({checks})"
