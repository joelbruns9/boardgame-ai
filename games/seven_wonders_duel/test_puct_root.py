"""PUCT root selection for evaluation, and its Python/Rust equivalence.

The Gumbel root exists to make a small fixed simulation budget produce an
unbiased policy-improvement TARGET. Evaluation is not building a target, and
Gumbel keys perturb which candidates get searched at all -- so competitive play
should select at the root by PUCT, like every node below it, and play argmax
visits. That is already what the advisor does (`advisor_adapter` calls
`descend()`), so it is what a gate must measure for gate numbers to mean
advisor strength.

Self-play keeps the Gumbel root: it is the target generator.

The equivalence test reuses the F3 harness (`_mock_search` + `RustGame`),
because a second implementation of root selection is exactly where a subtle
divergence hides -- tie-breaking, fold order, the unvisited-edge Q fallback.
"""

from __future__ import annotations

import pytest

from .codec import decode_action, legal_action_indices
from .engine import apply_action
from .game import Phase, new_game
from .search import GumbelMCTS, SearchConfig
from .test_rust_engine_equiv import (
    _mock_evaluate,
    extract_setup,
    random_game,
    swr,
)


def _puct_search(state, sims, top_k, seed, force):
    mcts = GumbelMCTS(
        None,
        SearchConfig(
            sims=sims,
            top_k=top_k,
            mode="closed",
            seed=seed,
            force_expand_root_chance=force,
            root_selection="puct",
        ),
    )
    mcts._evaluate = _mock_evaluate  # type: ignore[method-assign]
    return mcts.search(state)


def test_gumbel_remains_the_default():
    """Self-play must keep the Gumbel root; only evaluation opts out."""

    assert SearchConfig().root_selection == "gumbel"


def test_puct_root_reports_no_gumbel_topk():
    """An invented top-k would let a record claim a candidate set that never was."""

    result = _puct_search(new_game(7), 16, 8, 3, False)
    assert result.gumbel_topk == ()


def test_puct_policy_target_is_the_visit_distribution():
    result = _puct_search(new_game(11), 32, 8, 3, False)
    total = sum(result.visits.values())
    assert total > 0
    assert sum(result.policy_target.values()) == pytest.approx(1.0, abs=1e-9)
    for action, visits in result.visits.items():
        assert result.policy_target[action] == pytest.approx(
            visits / total, abs=1e-12
        )


def test_puct_plays_the_most_visited_action():
    result = _puct_search(new_game(13), 32, 8, 3, False)
    assert result.visits[result.action_index] == max(result.visits.values())


@pytest.mark.parametrize("sims", [1, 7, 32, 50])
def test_puct_spends_its_whole_budget(sims: int):
    """Unlike sequential halving, every simulation is a root visit."""

    result = _puct_search(new_game(17), sims, 8, 3, False)
    assert result.sims == sims
    assert sum(result.visits.values()) == sims


def test_puct_and_gumbel_roots_differ():
    """Guards against the flag silently doing nothing."""

    game = new_game(19)
    gumbel = GumbelMCTS(
        None, SearchConfig(sims=32, top_k=8, mode="closed", seed=3)
    )
    gumbel._evaluate = _mock_evaluate  # type: ignore[method-assign]
    assert gumbel.search(game).policy_target != _puct_search(
        game, 32, 8, 3, False
    ).policy_target


def test_rust_puct_root_matches_python():
    """F3-style equivalence for the PUCT root, force-expansion off and on."""

    checked = 0
    for game_seed in range(4):
        first_player, actions, library = random_game(game_seed, game_seed % 2)
        py = new_game(game_seed, first_player=first_player)
        rg = swr.RustGame(
            library_draws=[list(d) for d in library], **extract_setup(py)
        )
        tested_here = False
        for i, idx in enumerate(actions):
            if (
                not tested_here
                and i >= 8
                and py.phase is Phase.PLAY_AGE
                and py.pending_choice is None
            ):
                legal = legal_action_indices(py)
                for sims in (16, 64):
                    for seed in (1, 5):
                        for force in (False, True):
                            result = _puct_search(py, sims, 8, seed, force)
                            act, av, rv, visits, policy, topk, rsims, _dig = (
                                rg.closed_search(
                                    sims, 8, seed, force=force, puct_root=True
                                )
                            )
                            ctx = (
                                f"game {game_seed} sims {sims} seed {seed} "
                                f"force {force}"
                            )
                            assert act == result.action_index, f"{ctx}: action"
                            assert rsims == result.sims, f"{ctx}: sims"
                            assert list(topk) == [], f"{ctx}: topk must be empty"
                            assert list(visits) == [
                                result.visits[a] for a in legal
                            ], f"{ctx}: visits"
                            assert av == pytest.approx(
                                result.action_value, abs=1e-9
                            ), f"{ctx}: action_value"
                            assert rv == pytest.approx(
                                result.root_value, abs=1e-9
                            ), f"{ctx}: root_value"
                            for j, a in enumerate(legal):
                                assert policy[j] == pytest.approx(
                                    result.policy_target[a], abs=1e-9
                                ), f"{ctx}: policy[{a}]"
                            checked += 1
                tested_here = True
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)
    assert checked >= 16, f"only {checked} positions compared"


def test_rust_batched_puct_root_matches_python():
    """The path evaluation ACTUALLY uses: search_many_flat_net -> tree_resumable.

    Part 1 ported `tree::puct_root` (scalar). Gates never call it -- they go
    through the resumable driver, which has its own root scheduling. A second
    port needs its own equivalence test or it is unverified code on the path
    that matters most.
    """

    checked = 0
    for game_seed in range(3):
        first_player, actions, library = random_game(game_seed, game_seed % 2)
        py = new_game(game_seed, first_player=first_player)
        rg = swr.RustGame(
            library_draws=[list(d) for d in library], **extract_setup(py)
        )
        tested_here = False
        for i, idx in enumerate(actions):
            if (
                not tested_here
                and i >= 8
                and py.phase is Phase.PLAY_AGE
                and py.pending_choice is None
            ):
                legal = legal_action_indices(py)
                for sims in (16, 64):
                    for seed in (1, 5):
                        result = _puct_search(py, sims, 8, seed, False)
                        batched = rg.closed_search_batched(
                            1, sims, 8, seed, force=False, puct_root=True
                        )
                        ctx = f"game {game_seed} sims {sims} seed {seed}"
                        assert batched[0] == result.action_index, f"{ctx}: action"
                        assert batched[6] == result.sims, f"{ctx}: sims"
                        assert list(batched[5]) == [], f"{ctx}: topk"
                        assert list(batched[3]) == [
                            result.visits[a] for a in legal
                        ], f"{ctx}: visits"
                        for j, a in enumerate(legal):
                            assert batched[4][j] == pytest.approx(
                                result.policy_target[a], abs=1e-9
                            ), f"{ctx}: policy[{a}]"
                        checked += 1
                tested_here = True
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)
    assert checked >= 8, f"only {checked} positions compared"


def test_batched_puct_root_rejects_leaf_batch_above_one():
    """WU virtual loss at the root is a different algorithm, not a knob."""

    first_player, _actions, library = random_game(0, 0)
    py = new_game(0, first_player=first_player)
    rg = swr.RustGame(library_draws=[list(d) for d in library], **extract_setup(py))
    with pytest.raises(Exception, match="leaf_batch"):
        rg.closed_search_batched(4, 8, 4, 1, force=False, puct_root=True)


# --- resumable PUCT handle (ADVISOR_RUST_UNIFICATION.md step 4) --------------


def test_resumable_handle_matches_the_one_shot_puct_search():
    """`RustPuctSearch` must reproduce the search `test_rust_puct_root_matches_python`
    already gates against Python -- otherwise the advisor would run a searcher
    whose equivalence is assumed rather than tested, which is the exact failure
    this track exists to remove.

    Chunking must not change the tree: the whole point of the handle is that N
    simulations arrive as several `advance` calls instead of one.
    """
    from .rust_bridge import rust_game_from_state

    compared = 0
    for game_seed in range(4):
        first_player, actions, library = random_game(game_seed, game_seed % 2)
        py = new_game(game_seed, first_player=first_player)
        rg = swr.RustGame(library_draws=[list(d) for d in library], **extract_setup(py))
        for i, idx in enumerate(actions):
            if i >= 8 and py.phase is Phase.PLAY_AGE and py.pending_choice is None:
                for sims, chunks in ((64, (64,)), (64, (16, 16, 32)), (48, (1, 47))):
                    for seed in (1, 5):
                        # force=False: the resumable path cannot force-expand the
                        # root chance layer (needs the F4.5 forced-child cache).
                        one_shot = rg.closed_search(
                            sims, 8, seed, force=False, puct_root=True
                        )
                        # open_mock, not open: the one-shot search above uses
                        # Rust's internal MockEval, which a Python adapter
                        # cannot reproduce byte-for-byte.
                        handle = swr.RustPuctSearch.open_mock(
                            rust_game_from_state(py), sims, seed, 1.5, 50.0, 0.1, 8
                        )
                        done = 0
                        for chunk in chunks:
                            done = handle.advance(chunk)
                        sims_done, _rv, _rvs, _actor, edges = handle.snapshot()
                        assert sims_done == done
                        ctx = f"game {game_seed} sims {sims} chunks {chunks} seed {seed}"
                        assert done == one_shot[6], f"{ctx}: sims"
                        by_action = {a: v for a, v, _vs, _p in edges}
                        legal = legal_action_indices(py)
                        assert [by_action[a] for a in legal] == list(
                            one_shot[3]
                        ), f"{ctx}: visits"
                        compared += 1
                break_outer = compared > 0
                if break_outer:
                    break
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)
    assert compared >= 20, f"only {compared} comparisons"
