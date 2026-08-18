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


def _puct_search(state, sims, top_k, seed, force, forced_playout_k=0.0):
    mcts = GumbelMCTS(
        None,
        SearchConfig(
            sims=sims,
            top_k=top_k,
            mode="closed",
            seed=seed,
            force_expand_root_chance=force,
            root_selection="puct",
            forced_playout_k=forced_playout_k,
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


def test_forced_playouts_match_between_python_and_rust():
    """KataGo forced playouts change WHICH simulations happen, so the two
    implementations must agree exactly or the buffers diverge silently.

    Three places implement this rule -- `search.py`, `tree.rs` and
    `tree_resumable.rs` -- and the one that most easily drifts is `N`: the root
    counts its own expansion in `node.visits`, so the quota is scaled by the sum
    of CHILD visits instead, which all three can compute identically.
    """

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
                    for k in (1.0, 2.0):
                        expected = _puct_search(py, sims, 8, 5, False, forced_playout_k=k)
                        act, av, _rv, visits, policy, _topk, _s, _d = rg.closed_search(
                            sims, 8, 5, force=False, puct_root=True, forced_playout_k=k
                        )
                        ctx = f"game {game_seed} sims {sims} k {k}"
                        assert act == expected.action_index, f"{ctx}: action"
                        assert list(visits) == [
                            expected.visits[a] for a in legal
                        ], f"{ctx}: visits"
                        assert av == pytest.approx(
                            expected.action_value, abs=1e-9
                        ), f"{ctx}: action_value"
                        for j, a in enumerate(legal):
                            assert policy[j] == pytest.approx(
                                expected.policy_target[a], abs=1e-9
                            ), f"{ctx}: policy[{a}]"
                        checked += 1
                tested_here = True
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)
    assert checked >= 8, f"only {checked} positions compared"


def test_forced_playouts_actually_change_the_search():
    """A knob that quietly did nothing would pass the equivalence test above."""

    state = new_game(7)
    for _ in range(10):
        apply_action(state, decode_action(state, legal_action_indices(state)[0]))
    plain = _puct_search(state, 64, 8, 3, False, forced_playout_k=0.0)
    forced = _puct_search(state, 64, 8, 3, False, forced_playout_k=2.0)
    assert plain.visits != forced.visits
    # Forcing spreads the budget: more distinct actions get looked at.
    assert sum(1 for v in forced.visits.values() if v > 0) >= sum(
        1 for v in plain.visits.values() if v > 0
    )


def test_resumable_forced_playouts_match_the_one_shot_search():
    """Three implementations of the forcing rule, so the third needs its own
    gate. `tree_resumable` is the searcher production self-play runs, and it
    accepted `forced_playout_k` before it honoured it -- a state where the
    config said one thing and the search did another.

    It selects on `visits + incomplete` rather than plain `visits`; those are the
    same quantity here because `puct_root` is only legal at `leaf_batch = 1`, so
    nothing is ever in flight while the root selects.
    """
    from .rust_bridge import rust_game_from_state

    compared = 0
    for game_seed in range(3):
        first_player, actions, library = random_game(game_seed, game_seed % 2)
        py = new_game(game_seed, first_player=first_player)
        rg = swr.RustGame(library_draws=[list(d) for d in library], **extract_setup(py))
        tested_here = False
        for i, idx in enumerate(actions):
            if (
                not tested_here
                and i >= 8
                and py.phase is Phase.PLAY_AGE
                and py.pending_choice is None
            ):
                for sims, chunks in ((64, (64,)), (64, (16, 48))):
                    one_shot = rg.closed_search(
                        sims, 8, 5, force=False, puct_root=True, forced_playout_k=2.0
                    )
                    handle = swr.RustPuctSearch.open_mock(
                        rust_game_from_state(py), sims, 5, 1.5, 50.0, 0.1, 8,
                        forced_playout_k=2.0,
                    )
                    done = 0
                    for chunk in chunks:
                        done = handle.advance(chunk)
                    _sims_done, _rv, _rvs, _actor, edges = handle.snapshot()
                    by_action = {a: v for a, v, _vs, _p in edges}
                    legal = legal_action_indices(py)
                    ctx = f"game {game_seed} sims {sims} chunks {chunks}"
                    assert done == one_shot[6], f"{ctx}: sims"
                    assert [by_action[a] for a in legal] == list(
                        one_shot[3]
                    ), f"{ctx}: visits"
                    compared += 1
                tested_here = True
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)
    assert compared >= 4, f"only {compared} comparisons"


def test_pruning_rule_matches_between_python_and_rust():
    """The rule itself, gated directly rather than only through a whole search.

    A disagreement here decides training labels, and through a search it would
    surface as a puzzling visit-count mismatch rather than naming the cause.
    """
    import random

    from .search import prune_policy_target as py_prune

    rng = random.Random(20260817)
    compared = 0
    for _ in range(300):
        n = rng.randint(2, 9)
        visits = [rng.choice([0, 1, 2, 5, 13, 40, 97]) for _ in range(n)]
        if sum(visits) == 0:
            continue
        raw = [rng.random() + 1e-3 for _ in range(n)]
        mass = sum(raw)
        priors = [p / mass for p in raw]
        q = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        for k in (0.0, 0.5, 2.0, 6.0):
            root_visits = sum(visits) + 1  # the root counts its own expansion
            expected = py_prune(visits, priors, q, 1.5, k, root_visits)
            got = swr.prune_policy_target(visits, priors, q, 1.5, k, root_visits)
            if expected is None:
                # Rust returns the raw distribution where Python returns None;
                # both mean "record the unpruned target".
                total = float(sum(visits))
                expected = [v / total for v in visits]
            assert len(got) == len(expected)
            for j, (a, b) in enumerate(zip(got, expected)):
                assert a == pytest.approx(b, abs=1e-12), f"n={n} k={k} index {j}"
            compared += 1
    assert compared > 500, compared


def test_pruning_changes_the_recorded_target_but_not_the_played_move():
    """The invariant the whole split exists for: forcing stays in the
    trajectory, and comes back out of the label."""

    state = new_game(11)
    for _ in range(12):
        apply_action(state, decode_action(state, legal_action_indices(state)[0]))
    forced = _puct_search(state, 64, 8, 3, False, forced_playout_k=2.0)
    assert forced.training_policy is not None
    total = sum(forced.visits.values())
    raw = {a: v / total for a, v in forced.visits.items()}
    assert forced.policy_target == pytest.approx(raw), "selection must be raw visits"
    assert forced.training_policy != pytest.approx(raw), "label must be pruned"
    # The played action is still argmax of the RAW visits, unaffected by pruning.
    assert forced.action_index == max(forced.visits, key=lambda a: forced.visits[a])


def _record_root_modes(records):
    """(full-move modes, cheap-move modes) as observed in the RECORD.

    `gumbel_topk` is the behavioural signature: the Gumbel root emits its
    candidate set, the PUCT root deliberately emits none rather than invent one.
    Asserting on this rather than on a config value is the point -- the defect
    this guards against is a config that says PUCT while the search runs Gumbel.
    """
    full, cheap = set(), set()
    for record in records:
        for move in record["moves"]:
            mode = "gumbel" if move["gumbel_topk"] else "puct"
            (full if move["full_search"] else cheap).add(mode)
    return full, cheap


def test_a_gate_shaped_run_is_puct_on_every_move():
    """The defect this exists for: gates set `full_search_fraction = 0.0` and
    take their strength from the CHEAP path, so any hybrid keyed off `full`
    would run the whole promotion gate under Gumbel while `--eval-search-mode
    puct` reported success."""

    from .rust_bridge import rust_games_for_self_play
    from .test_f4_scheduler import _common

    seeds = [20260817, 20260818, 20260819]
    records, _ = swr.self_play_many_mock(
        games=rust_games_for_self_play(seeds, [0, 1, 0]),
        game_seeds=seeds,
        force=True,
        puct_root=True,
        **(_common(leaf_batch=1, global_batch_cap=8) | {"full_search_fraction": 0.0}),
    )
    full, cheap = _record_root_modes(records)
    assert full == set(), "gate config should produce no full-budget moves"
    assert cheap == {"puct"}, f"gate ran something other than PUCT: {cheap}"


def test_the_hybrid_splits_full_and_cheap_moves():
    """Opt-in, and only when asked: PUCT where targets are produced, Gumbel on
    the cheap moves whose guarantee is built for tiny budgets."""

    from .rust_bridge import rust_games_for_self_play
    from .test_f4_scheduler import _common

    seeds = [20260820, 20260821, 20260822]
    records, _ = swr.self_play_many_mock(
        games=rust_games_for_self_play(seeds, [0, 1, 0]),
        game_seeds=seeds,
        force=True,
        puct_root=True,
        cheap_puct_root=False,
        **(_common(leaf_batch=1, global_batch_cap=8) | {"full_search_fraction": 0.5}),
    )
    full, cheap = _record_root_modes(records)
    assert full == {"puct"}, f"full moves not PUCT: {full}"
    assert cheap == {"gumbel"}, f"cheap moves not Gumbel: {cheap}"
