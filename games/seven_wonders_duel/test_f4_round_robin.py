"""Gates for the interleaved sequential-halving schedule.

`round_robin_candidates` issues one simulation per candidate and cycles, instead
of `per_action` consecutive simulations per candidate. Sequential halving fixes
only the per-round allocation, so both orders are faithful — but they visit
different leaves, so this is a *different search*, not a refactor. That means the
usual identity gates must be re-anchored rather than reused:

1. Rust round-robin must match the Python reference under round-robin, exactly
   as the blocked order does. Anything less and the port has diverged.
2. The Phase 2 exactness result must carry over: conflict-free waves at
   `leaf_batch > 1` must be bit-identical to `leaf_batch = 1` *under the new
   order*.
3. The allocation must be provably unchanged — each candidate still receives the
   same number of simulations per round — and the order must actually differ,
   or there is nothing to measure.
"""

from __future__ import annotations

import pytest

from .codec import decode_action, legal_action_indices
from .engine import apply_action
from .game import Phase, new_game
from .rust_bridge import rust_setup
from .search import GumbelMCTS, SearchConfig


def _mock_evaluate(state):
    """The deterministic oracle the cross-language gates use."""

    from .test_rust_engine_equiv import _mock_evaluate as reference

    return reference(state)


def _python_search(state, sims, top_k, seed, *, force, round_robin):
    mcts = GumbelMCTS(
        None,
        SearchConfig(
            sims=sims,
            top_k=top_k,
            mode="closed",
            seed=seed,
            force_expand_root_chance=force,
            round_robin_candidates=round_robin,
        ),
    )
    mcts._evaluate = _mock_evaluate  # type: ignore[method-assign]
    return mcts.search(state)


def _positions(count=4, skip=10):
    """Mid-Age positions with a wide-enough candidate set to interleave."""

    import random

    import seven_wonders_rust as swr

    found = []
    for game_seed in range(30):
        if len(found) >= count:
            break
        rng = random.Random(game_seed ^ 0xA11CE)
        python_game = new_game(game_seed, first_player=game_seed % 2)
        setup = rust_setup(python_game)
        library: list[list[str]] = []
        replay: list[int] = []
        for move in range(200):
            if python_game.phase is Phase.COMPLETE:
                break
            legal = list(legal_action_indices(python_game))
            if not legal:
                break
            if (
                move >= skip
                and python_game.phase is Phase.PLAY_AGE
                and python_game.pending_choice is None
                and len(legal) >= 6
            ):
                rust_game = swr.RustGame(
                    library_draws=[list(d) for d in library], **setup
                )
                for index in replay:
                    rust_game.apply_index(index)
                found.append((game_seed, python_game.clone(), rust_game))
                break
            choice = rng.choice(legal)
            action = decode_action(python_game, choice)
            apply_action(python_game, action)
            if action.wonder_name == "The Great Library":
                pending = python_game.pending_choice
                if pending is not None:
                    library.append(list(pending.options))
            replay.append(choice)
    assert len(found) >= count, f"only found {len(found)} positions"
    return found[:count]


@pytest.fixture(scope="module")
def positions():
    return _positions()


def test_rust_round_robin_matches_the_python_reference(positions):
    """Gate 1: re-anchor cross-language identity on the new order."""

    checked = 0
    for game_seed, python_game, rust_game in positions:
        for sims, top_k in ((32, 8), (64, 8), (48, 16)):
            seed = 91_000 + game_seed * 13 + sims
            for force in (False, True):
                expected = _python_search(
                    python_game, sims, top_k, seed, force=force, round_robin=True
                )
                got = rust_game.closed_search_resumable(
                    sims,
                    top_k,
                    seed,
                    force=force,
                    round_robin_candidates=True,
                )
                context = (
                    f"game {game_seed} sims={sims} top_k={top_k} force={force}"
                )
                assert got[0] == expected.action_index, f"{context}: action"
                legal = list(legal_action_indices(python_game))
                assert list(got[3]) == [
                    expected.visits.get(action, 0) for action in legal
                ], f"{context}: visits"
                assert list(got[5]) == list(expected.gumbel_topk), f"{context}: top-k"
                assert got[6] == expected.sims, f"{context}: sims"
                assert list(got[4]) == pytest.approx(
                    [expected.policy_target.get(a, 0.0) for a in legal],
                    rel=0,
                    abs=1e-9,
                ), f"{context}: policy"
                assert got[2] == pytest.approx(
                    expected.root_value, rel=0, abs=1e-9
                ), f"{context}: root value"
                checked += 1
    assert checked >= 12


def test_conflict_free_exactness_carries_over_to_round_robin(positions):
    """Gate 2: Phase 2's exactness must survive the reordering.

    This is the result that makes the new order safe to batch at all: without
    it, interleaving would trade a quality question for a throughput one.
    """

    for game_seed, _python_game, rust_game in positions:
        for sims in (32, 64, 128):
            for leaf_batch in (2, 8, 16):
                seed = 92_000 + game_seed * 7 + sims
                force = sims == 64
                baseline = rust_game.closed_search_batched(
                    1, sims, 16, seed, force=force,
                    conflict_free_waves=True, round_robin_candidates=True,
                )
                batched = rust_game.closed_search_batched(
                    leaf_batch, sims, 16, seed, force=force,
                    conflict_free_waves=True, round_robin_candidates=True,
                )
                context = (
                    f"game {game_seed} sims={sims} leaf_batch={leaf_batch} "
                    f"force={force}"
                )
                assert batched[0] == baseline[0], f"{context}: action"
                assert list(batched[3]) == list(baseline[3]), f"{context}: visits"
                assert list(batched[5]) == list(baseline[5]), f"{context}: top-k"
                assert batched[1] == pytest.approx(
                    baseline[1], rel=0, abs=0
                ), f"{context}: action value"
                assert batched[2] == pytest.approx(
                    baseline[2], rel=0, abs=0
                ), f"{context}: root value"
                assert list(batched[4]) == pytest.approx(
                    list(baseline[4]), rel=0, abs=0
                ), f"{context}: policy"
                assert list(batched[9]) == pytest.approx(
                    list(baseline[9]), rel=0, abs=0
                ), f"{context}: tree digest"


def test_allocation_is_unchanged_but_the_order_is_not(positions):
    """Gate 3: same visits per candidate, different search.

    Sequential halving is only faithful if the reordering preserves the
    per-round allocation. Total simulations and the surviving candidate sets are
    the observable consequence; the search *outputs* must nevertheless differ, or
    the flag does nothing and the measurement below is meaningless.
    """

    differed = 0
    for game_seed, _python_game, rust_game in positions:
        for sims in (32, 64, 128):
            seed = 93_000 + game_seed * 11 + sims
            blocked = rust_game.closed_search_batched(
                1, sims, 16, seed, force=False, conflict_free_waves=True
            )
            interleaved = rust_game.closed_search_batched(
                1, sims, 16, seed, force=False,
                conflict_free_waves=True, round_robin_candidates=True,
            )
            context = f"game {game_seed} sims={sims}"
            # Same budget, and the same total visit mass over the root.
            assert interleaved[6] == blocked[6] == sims, f"{context}: sims"
            assert sum(interleaved[3]) == sum(blocked[3]) == sims, f"{context}: visits"
            # Same Gumbel top-k: the reordering happens strictly inside a round.
            assert list(interleaved[5]) == list(blocked[5]), f"{context}: top-k"
            if list(interleaved[3]) != list(blocked[3]):
                differed += 1
    assert differed > 0, (
        "interleaving changed no search output, so it cannot change wave width "
        "either -- the flag is not doing what it claims"
    )


def test_round_robin_widens_waves(positions):
    """The point of the exercise, measured rather than predicted."""

    blocked_waves = interleaved_waves = 0
    blocked_paths = interleaved_paths = 0
    for game_seed, _python_game, rust_game in positions:
        for sims in (64, 128):
            seed = 94_000 + game_seed + sims
            blocked = rust_game.closed_search_batched(
                16, sims, 16, seed, force=False, conflict_free_waves=True
            )
            interleaved = rust_game.closed_search_batched(
                16, sims, 16, seed, force=False,
                conflict_free_waves=True, round_robin_candidates=True,
            )
            # metrics tuple: (scheduled, requested, unique, terminal, collisions,
            #                 waves, max_paths, max_unique)
            blocked_waves += blocked[7][5]
            interleaved_waves += interleaved[7][5]
            blocked_paths += blocked[7][1]
            interleaved_paths += interleaved[7][1]
    blocked_width = blocked_paths / blocked_waves
    interleaved_width = interleaved_paths / interleaved_waves
    print(
        f"mean wave width: blocked {blocked_width:.2f} -> "
        f"interleaved {interleaved_width:.2f} "
        f"({blocked_waves} -> {interleaved_waves} waves, "
        f"{blocked_waves / interleaved_waves:.2f}x fewer)"
    )
    assert interleaved_width > blocked_width
    assert interleaved_waves < blocked_waves
