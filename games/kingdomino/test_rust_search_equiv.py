"""Step 3 equivalence gate: the Rust-HOSTED `kr.RustSearch` (recursion in Rust
over one mutable state) must return a search value NUMERICALLY IDENTICAL (within
1e-9) to the Python-hosted `RustExpectiminimax` — and hence the pure-Python
`ExpectiminimaxBot` — whenever every in-horizon chance node is enumerated
(deterministic). (The tolerance guards against last-ULP float summation-order
differences; in practice the values usually agree bit-for-bit.)

This is the same discipline as the make/unmake and encode_state milestones: a new
faster path is proven byte-identical to the known-good reference under real search
recursion, not just unit-tested in isolation. Wide (sampled) chance is validated
separately (reproducibility + it converges to the enumerated exact value as the
sample count grows), since sampled mode uses its own RNG and is a Monte-Carlo
estimate rather than a byte-for-byte match of CPython's Mersenne Twister.
"""
from __future__ import annotations

import random
from math import inf

import kingdomino_rust as kr

from games.kingdomino.game import GameState, Phase
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.rust_expectiminimax import (
    RustExpectiminimax,
    tanh_margin as r_tanh_margin,
    pick_aware as r_pick_aware,
)

# (RustSearch eval name, matching Python-hosted eval_fn)
EVAL_PAIRS = [
    ("pick_blind", r_tanh_margin),
    ("pick_aware", r_pick_aware),
]


def _late_states(max_deck=8, count=8):
    """PLACE_AND_SELECT states with a small enough bag that every in-horizon
    chance node enumerates (C(<=8,4)=70 <= enum_cap=128)."""
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


def _wide_boundary_state(min_deck=12):
    """A PLACE_AND_SELECT round-boundary state whose next move deals from a bag so
    large that C(deck,4) exceeds a small enum_cap → Monte-Carlo sampled."""
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
                return st
            st = st.step(rng.choice(st.legal_actions()))
    raise AssertionError("no wide-boundary state found")


def test_rust_hosted_value_matches_python_hosted():
    """The core gate: RustSearch.value == RustExpectiminimax.value on enumerated
    chance, both evals, depths 2 and 3."""
    states = _late_states()
    assert len(states) >= 5, f"too few late states ({len(states)})"
    checks = 0
    for pystate in states:
        rs = _rust_state_from_python(pystate)
        assert rs is not None
        for name, r_eval in EVAL_PAIRS:
            for depth in (2, 3):
                ref = RustExpectiminimax(depth=depth, enum_cap=128, eval_fn=r_eval)
                ref_v = ref.value(kr.SearchEngine(rs), depth)

                search = kr.RustSearch(depth=depth, enum_cap=128, eval=name)
                got_v = search.value(rs, depth)

                assert abs(ref_v - got_v) < 1e-9, (
                    f"{name} d{depth}: rust-hosted {got_v} != python-hosted {ref_v} "
                    f"(delta {got_v - ref_v:.2e})"
                )
                assert search.nodes > 0
                checks += 1
    assert checks >= 20, f"too few equivalence checks ({checks})"


def test_margin_weight_terminal_blend_matches():
    """The expected-outcome margin blend (margin_weight != 0) must also match the
    Python-hosted reference, which applies the same tanh-margin proxy at terminals."""
    states = _late_states()
    rs = _rust_state_from_python(states[0])
    for mw in (0.25, 1.0):
        ref = RustExpectiminimax(depth=3, enum_cap=128, eval_fn=r_pick_aware, margin_weight=mw)
        ref_v = ref.value(kr.SearchEngine(rs), 3)
        search = kr.RustSearch(depth=3, enum_cap=128, eval="pick_aware", margin_weight=mw)
        got_v = search.value(rs, 3)
        assert abs(ref_v - got_v) < 1e-9, f"margin_weight={mw}: {got_v} != {ref_v}"


def test_choose_action_returns_a_best_action():
    """choose_action must return an action whose searched value equals the optimal
    (max for player 0, min for player 1) among all root actions."""
    states = _late_states()
    for pystate in states[:4]:
        rs = _rust_state_from_python(pystate)
        search = kr.RustSearch(depth=2, enum_cap=128, eval="pick_aware")
        chosen = search.choose_action(rs)

        # Value of every root action at the same depth (fresh windows), then check
        # the chosen one is optimal for the side to move.
        eng = kr.SearchEngine(rs)
        actor = eng.current_actor()
        ref = RustExpectiminimax(depth=2, enum_cap=128, eval_fn=r_pick_aware)
        vals = {}
        for a in eng.legal_actions():
            vals[a] = ref._action_value(eng, a, 2, -inf, inf)
        best = max(vals.values()) if actor == 0 else min(vals.values())
        assert abs(vals[chosen] - best) < 1e-9, (
            f"chosen action value {vals[chosen]} != optimal {best} (actor {actor})"
        )


def test_sampled_mode_reproducible_and_converges():
    """Wide chance (enum_cap forcing MC sampling) is reproducible for a fixed seed,
    and as chance_samples grows the sampled value approaches the enumerated exact
    value (validating the Monte-Carlo estimator without byte-matching CPython's RNG)."""
    st = _wide_boundary_state()
    rs = _rust_state_from_python(st)

    # Reproducible: two identically-configured searches agree exactly.
    a = kr.RustSearch(depth=2, enum_cap=1, chance_samples=8, eval="pick_aware", seed=7)
    b = kr.RustSearch(depth=2, enum_cap=1, chance_samples=8, eval="pick_aware", seed=7)
    assert a.value(rs, 2) == b.value(rs, 2)

    # A different seed generally gives a different sampled estimate.
    c = kr.RustSearch(depth=2, enum_cap=1, chance_samples=8, eval="pick_aware", seed=99)
    # (not asserted equal/unequal — just must not error)
    c.value(rs, 2)

    # Convergence: enumerate the (single, wide) root chance node exactly, then check
    # a large sample lands near it. C(deck,4) with deck>=12 is >= 495, so enum_cap
    # above that enumerates; a big-sample estimate should be within a loose band.
    from math import comb

    exact_cap = comb(len(st.deck), 4) + 1
    exact = kr.RustSearch(depth=2, enum_cap=exact_cap, eval="pick_aware").value(rs, 2)
    approx = kr.RustSearch(
        depth=2, enum_cap=1, chance_samples=256, eval="pick_aware", seed=1
    ).value(rs, 2)
    assert abs(exact - approx) < 0.15, (
        f"sampled estimate {approx} far from exact {exact} (|Δ|={abs(exact - approx):.3f})"
    )


def test_constructor_validation():
    import pytest

    for bad in (dict(depth=0), dict(chance_samples=0), dict(enum_cap=0)):
        with pytest.raises(ValueError):
            kr.RustSearch(**bad)
    with pytest.raises(ValueError):
        kr.RustSearch(eval="nonsense")
    # A non-finite margin_weight would make terminal blends NaN and could leave
    # choose_action's best-action vector empty → an index panic. Reject at ctor.
    for bad_mw in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            kr.RustSearch(margin_weight=bad_mw)
