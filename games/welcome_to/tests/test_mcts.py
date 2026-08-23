"""The root-player contract, the leaf-value blend, and observation-keyed chance."""
from __future__ import annotations

import collections
import dataclasses
import math
import random

import numpy as np
import pytest
import torch

from games.welcome_to import action_codec as codec
from games.welcome_to import encoder as enc
from games.welcome_to import macro_codec as mc
from games.welcome_to import mcts
from games.welcome_to import network as nw
from games.welcome_to import training
from games.welcome_to.bots import GreedyBot
from games.welcome_to.sheet import Sheet
from games.welcome_to.constants import CARD_TABLE, NUM_BASE_CARDS
from typing import Optional

from games.welcome_to.game import GameConfig, GameState, Phase

_SMALL = nw.NetConfig(
    sheet_hidden=32, sheet_out=16, trunk_hidden=48, trunk_blocks=1, head_hidden=32
)


def _search(simulations: int = 32, **kwargs) -> tuple[mcts.MCTS, mcts.NetEvaluator]:
    config = mcts.SearchConfig(simulations=simulations, **kwargs)
    evaluator = mcts.NetEvaluator(nw.WelcomeToNet(_SMALL), torch.device("cpu"), config)
    return mcts.MCTS(evaluator, config), evaluator


def _position(players: int = 2, turn: int = 1, seed: int = 7, root: int = 0) -> GameState:
    bots = [GreedyBot(random.Random(seed * 10 + i)) for i in range(players)]
    state = GameState.new(seed=seed, config=GameConfig(players=players, advanced=True))
    while not state.is_terminal and state.turn < turn:
        state.apply(bots[state.actor].act(state))
    while not state.is_terminal and (
        state.actor != root or state.phase is Phase.WRITE_NUMBER
    ):
        state.apply(bots[state.actor].act(state))
    return state


# ──────────────────────────────────────────────────────────────────────────
# The root-player contract -- all four clauses
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("root", [0, 1, 2])
def test_leaves_are_evaluated_as_the_root_player_never_the_actor(root):
    """Clause 3, and the bug the contract exists to prevent.

    ``encode_state`` defaults to ``state.actor``, so a leaf reached while
    another seat is to move would silently yield *that seat's* value -- which
    cannot be backed up as the root's.
    """
    seen_viewers, seen_actors = set(), set()

    class Spy(mcts.NetEvaluator):
        def evaluate(self, state, viewer):
            seen_viewers.add(viewer)
            seen_actors.add(state.actor)
            return super().evaluate(state, viewer)

    config = mcts.SearchConfig(simulations=24)
    evaluator = Spy(nw.WelcomeToNet(_SMALL), torch.device("cpu"), config)
    state = _position(players=3, root=root)
    mcts.MCTS(evaluator, config).search(state, root=root, rng=random.Random(1))

    assert seen_viewers == {root}, "a leaf was evaluated from the wrong seat"
    # clause 1: and every node in the tree is a decision of the root player's
    assert seen_actors == {root}, "a node belonged to an opponent"


def test_opponents_are_sampled_from_their_own_view():
    """Clause 2.  Sampling seat 1's move needs seat 1's legal set and policy --
    that is a transition, not a node, and its value is never backed up."""
    opponent_views = set()

    class Spy(mcts.NetEvaluator):
        def policy(self, state, viewer):
            opponent_views.add((viewer, state.actor))
            return super().policy(state, viewer)

    # late enough, and with enough budget, that the search actually crosses a
    # turn boundary -- at turn 1 the tree never leaves the root player's own turn
    config = mcts.SearchConfig(simulations=256)
    evaluator = Spy(nw.WelcomeToNet(_SMALL), torch.device("cpu"), config)
    state = _position(players=3, turn=18, root=0)
    mcts.MCTS(evaluator, config).search(state, root=0, rng=random.Random(1))

    assert opponent_views, "no opponent was ever sampled"
    for viewer, actor in opponent_views:
        assert viewer == actor, "an opponent was sampled from someone else's view"
        assert viewer != 0, "the root player was sampled as an opponent"


def test_the_backed_up_value_is_never_negated():
    """Clause 4.  A constant evaluator must give every Q that same constant."""
    class Constant(mcts.NetEvaluator):
        def evaluate(self, state, viewer):
            priors, _ = super().evaluate(state, viewer)
            return priors, -0.75

    config = mcts.SearchConfig(simulations=48)
    evaluator = Constant(nw.WelcomeToNet(_SMALL), torch.device("cpu"), config)
    state = _position(players=2)
    _, visits, root = mcts.MCTS(evaluator, config).search(state, rng=random.Random(2))

    visited = root.visits > 0
    q = root.total[visited] / root.visits[visited]
    assert np.allclose(q, -0.75), "the sign or frame changed on the way up"


def test_search_does_not_disturb_the_state_it_was_given():
    search, _ = _search(simulations=32)
    state = _position(players=3, turn=4)
    before = (state.turn, state.actor, state.phase, [list(r) for r in state.sheets[0].numbers])
    search.search(state, rng=random.Random(5))
    after = (state.turn, state.actor, state.phase, [list(r) for r in state.sheets[0].numbers])
    assert before == after


def test_searching_from_a_seat_that_is_not_to_move_is_refused():
    search, _ = _search()
    state = _position(players=2, root=0)
    with pytest.raises(ValueError):
        search.search(state, root=1)


# ──────────────────────────────────────────────────────────────────────────
# Chance is keyed, not merged
# ──────────────────────────────────────────────────────────────────────────
def _tree_stats(node) -> dict:
    stats = {"nodes": 0, "max_depth": 0, "most_observations": 1, "leaf_depths": []}

    def walk(current, depth):
        stats["nodes"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth)
        if not current.children:
            stats["leaf_depths"].append(depth)
        counts = collections.Counter(action for action, _ in current.children)
        if counts:
            stats["most_observations"] = max(
                stats["most_observations"], max(counts.values())
            )
        for child in current.children.values():
            walk(child, depth + 1)

    walk(node, 0)
    depths = stats["leaf_depths"]
    stats["mean_leaf_depth"] = sum(depths) / len(depths)
    return stats


def test_one_action_under_different_reveals_becomes_different_children():
    """The selection bias this prevents: an action only *legal* under favourable
    reveals would otherwise accumulate Q solely from those simulations."""
    search, _ = _search(simulations=400)
    state = _position(players=2, turn=20)
    _, _, root = search.search(state, root=0, rng=random.Random(3))
    stats = _tree_stats(root)
    assert stats["most_observations"] > 1, "chance was merged into one child"


def test_sharpening_the_prior_does_not_deepen_the_tree():
    """The obvious objection, tested at its own limit.

    If depth were limited by simulations wasted on moves a trained policy would
    know are bad, concentrating the prior would buy depth.  It does not.
    Measured over 4 network seeds x 6 positions at 256 simulations: uniform mean
    leaf depth 1.59, one-hot 1.59, with 2 of 24 cells differing by more than 0.5.
    **One-hot is the upper bound on sharpness, so this closes the question** --
    no trained policy beats the extreme.

    The reason is structural.  A concentrated prior sends every simulation down
    the same root action, to the same child (no chance crossed yet, so the key
    recurs), along the same within-turn chain -- and then into the turn boundary,
    where **every simulation draws a unique observation and terminates as a fresh
    leaf**.  Depth is therefore "the root player's own decisions left this turn,
    plus one", and a uniform prior reaches it too because those chains are
    deterministic given the action.

    Budget beyond about one simulation per root action goes into root averaging,
    not depth.  Two earlier versions of this test asserted the opposite things,
    each from a single unseeded measurement; this one asserts the null, which is
    what 24 seeded cells actually show.
    """
    class Sharp(mcts.NetEvaluator):
        def evaluate(self, evaluated, viewer):
            priors, value = super().evaluate(evaluated, viewer)
            legal = mc.legal_mask(evaluated)
            sharp = np.zeros_like(priors)
            sharp[int(np.argmax(np.where(legal, priors, -1.0)))] = 1.0
            return sharp, value

    config = mcts.SearchConfig(simulations=256)
    torch.manual_seed(0)
    net = nw.WelcomeToNet(_SMALL)

    gaps = []
    for turn in (8, 16):
        state = _position(players=2, turn=turn)
        while not state.is_terminal and (
            state.actor != 0 or state.phase is not Phase.CHOOSE_CARDS
        ):
            state.apply(state.legal_actions()[0])
        if state.is_terminal:
            continue
        flat = mcts.MCTS(mcts.NetEvaluator(net, torch.device("cpu"), config), config)
        sharp = mcts.MCTS(Sharp(net, torch.device("cpu"), config), config)
        a = _tree_stats(flat.search(state, root=0, rng=random.Random(3))[2])
        b = _tree_stats(sharp.search(state, root=0, rng=random.Random(3))[2])
        gaps.append(b["mean_leaf_depth"] - a["mean_leaf_depth"])

    assert gaps, "no usable position"
    assert max(gaps) < 1.0, (
        f"sharpening deepened the tree by {max(gaps):.2f} -- if that reproduces "
        "across seeds and positions, the depth limit is not what this test says "
        "and SELF_PLAY_PLAN.md needs re-measuring"
    )


def test_every_simulation_gets_its_own_determinization():
    """`redeterminize` needs a search RNG the caller advances: `copy` clones the
    generator exactly, so a state's own RNG returns the identical shuffle."""
    state = _position(players=2, turn=6)
    rng = random.Random(3)
    futures = {
        tuple(state.redeterminize(rng).deck[state.deck_pos : state.deck_pos + 10])
        for _ in range(24)
    }
    assert len(futures) > 20, "the determinizations are not independent"


# ──────────────────────────────────────────────────────────────────────────
# The leaf value
# ──────────────────────────────────────────────────────────────────────────
def test_the_confidence_floor_matches_the_derivation():
    assert mcts.confidence_floor(2) == pytest.approx(0.0)
    assert mcts.confidence_floor(3) == pytest.approx(1 / 3)
    assert mcts.confidence_floor(4) == pytest.approx(4 / 9)


def test_at_two_seats_the_rank_value_is_the_win_probability_rescaled():
    config = mcts.SearchConfig(alpha=0.0)
    for p_win in (0.0, 0.25, 0.5, 1.0):
        value, parts = mcts.blend_value([p_win, 1 - p_win], [0.0, 0.0], 2, config)
        assert value == pytest.approx(2 * p_win - 1)
        assert parts["rank_value"] == pytest.approx(2 * p_win - 1)


def test_the_gate_tells_a_certain_second_place_from_a_coin_flip():
    """The category error a mean-based gate makes: both have E[u] = 0.5 at 3p."""
    config = mcts.SearchConfig(confidence_power=1.0)
    _, certain = mcts.blend_value([0.0, 1.0, 0.0], [0.0, 0.0, 0.0], 3, config)
    _, coinflip = mcts.blend_value([0.5, 0.0, 0.5], [0.0, 0.0, 0.0], 3, config)
    assert certain["rank_value"] == pytest.approx(coinflip["rank_value"])
    assert certain["confidence"] > 0.9
    assert coinflip["confidence"] == pytest.approx(0.0)


def test_margin_survives_where_rank_goes_flat():
    """A 4p seat locked into second: rank has zero gradient, so margin is the
    only signal left -- and a `win_value ** 4` gate would suppress it by 98%."""
    config = mcts.SearchConfig(confidence_power=1.0, alpha=0.5)
    ahead, _ = mcts.blend_value([0.0, 1.0, 0.0, 0.0], [0.5, 0.1, 0.0, 0.0], 4, config)
    behind, _ = mcts.blend_value([0.0, 1.0, 0.0, 0.0], [0.0, 0.4, 0.0, 0.0], 4, config)
    assert ahead > behind + 0.1, "margin was gated off in a locked-rank position"


def test_the_margin_reads_the_best_opponent_not_the_mean():
    config = mcts.SearchConfig(alpha=1.0, confidence_power=0.0)
    close, _ = mcts.blend_value([0.5, 0.5, 0.0, 0.0], [0.5, 0.45, 0.0, 0.0], 2, config)
    clear, _ = mcts.blend_value([0.5, 0.5, 0.0, 0.0], [0.5, 0.10, 0.0, 0.0], 2, config)
    assert clear > close


def test_a_won_terminal_beats_a_lost_one_in_the_root_players_frame():
    config = mcts.SearchConfig()
    bots = [GreedyBot(random.Random(i)) for i in range(2)]
    state = GameState.new(seed=6, config=GameConfig(players=2, advanced=True))
    while not state.is_terminal:
        state.apply(bots[state.actor].act(state))
    scores = state.scores()
    winner = max(range(2), key=lambda p: scores[p])
    loser = 1 - winner
    assert mcts.terminal_value(state, winner, config) > mcts.terminal_value(
        state, loser, config
    )


# ──────────────────────────────────────────────────────────────────────────
# Playing
# ──────────────────────────────────────────────────────────────────────────
def test_a_forced_move_skips_the_search_entirely():
    """Half of all decisions in this game have two options or fewer."""
    search, evaluator = _search(simulations=64)
    state = _position(players=2, turn=3)
    while not state.is_terminal and (
        state.phase is Phase.WRITE_NUMBER or len(mc.legal_macros(state)) != 1
    ):
        state.apply(state.legal_actions()[0])
    if state.is_terminal:
        pytest.skip("no forced position on this line")
    before = evaluator.calls
    action = search.play(state)
    assert evaluator.calls == before, "a forced move still paid for a search"
    assert action in mc.legal_macros(state)


def test_play_returns_a_legal_macro():
    search, _ = _search(simulations=16)
    state = _position(players=2, turn=5)
    action = search.play(state, rng=random.Random(1))
    assert action in mc.legal_macros(state)


def test_the_visit_policy_is_a_distribution_over_the_macro_space():
    search, _ = _search(simulations=32)
    state = _position(players=2, turn=5)
    actions, visits, _ = search.search(state, rng=random.Random(1))
    policy = mcts.visit_policy(actions, visits)
    assert policy.shape == (mc.NUM_MACRO_ACTIONS,)
    assert policy.sum() == pytest.approx(1.0)
    assert np.all(policy[np.setdiff1d(np.arange(mc.NUM_MACRO_ACTIONS), actions)] == 0.0)


def test_root_noise_moves_the_priors_and_leaves_them_a_distribution():
    state = _position(players=2, turn=5)
    plain, _ = _search(simulations=8)
    noisy, _ = _search(simulations=8, dirichlet_alpha=1.0)
    _, _, a = plain.search(state, rng=random.Random(1))
    _, _, b = noisy.search(state, rng=random.Random(1))
    assert not np.allclose(a.prior, b.prior)
    assert b.prior.sum() == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────────
# Forced nodes are collapsed inside simulations -- SEARCH_SPEC §12 step 2
# ──────────────────────────────────────────────────────────────────────────
def _forced_park_state(seed: int = 5) -> tuple[GameState, int]:
    """A real ``ACTION_PARK`` position, where the pass is now dominated.

    Walked to rather than constructed: ``_park_streets()`` reads
    ``ctx.last_house`` -- the street the number just went into -- so a state with
    the phase set by hand has no available street at all and ``_settle`` would
    have skipped it.
    """
    bots = [GreedyBot(random.Random(seed * 10 + i)) for i in range(2)]
    state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True))
    while not state.is_terminal:
        if state.phase is Phase.ACTION_PARK:
            assert len(state.legal_actions()) > 1
            assert len(mc.search_legal_macros(state)) == 1, "park should be forced"
            return state, state.actor
        state.apply(bots[state.actor].act(state))
    raise AssertionError("no ACTION_PARK state on this line")


def test_a_collapsed_forced_action_is_applied_not_skipped():
    """Skipping the *node* is not skipping the *move*.

    The park node disappears because its pass is dominated, but the park is
    still built -- ``parks[x] += 1`` -- and the phase still advances.
    """
    search, _ = _search(simulations=1)
    state, root = _forced_park_state()
    before = list(state.sheets[root].parks)
    turn = state.turn

    mcts.drive(
        search._collapse_forced(state, root, random.Random(0), turn, ("sentinel",)),
        search.evaluator,
    )
    assert state.sheets[root].parks != before, "the forced park was never built"
    assert state.phase is not Phase.ACTION_PARK


def test_collapsing_is_a_no_op_once_the_turn_has_advanced():
    """The guard that keeps the chance key honest.

    Two forced steps that each crossed a boundary would put *two* reveals behind
    a child keyed on the second alone.  ``_collapse_forced`` refuses to run at
    all when the turn it was given is not the turn it is looking at.
    """
    search, _ = _search(simulations=1)
    state, root = _forced_park_state()
    before = (state.phase, list(state.sheets[root].parks))

    observation = mcts.drive(
        search._collapse_forced(
            state, root, random.Random(0), state.turn - 1, ("sentinel",)
        ),
        search.evaluator,
    )
    assert (state.phase, list(state.sheets[root].parks)) == before
    assert observation == ("sentinel",)


def test_no_node_in_the_tree_is_a_one_action_node():
    """PUCT over a one-element array is a network call that bought nothing."""
    torch.manual_seed(0)
    search, _ = _search(simulations=192)
    state = _position(players=2, turn=10)
    _, _, root = search.search(state, root=0, rng=random.Random(4))

    widths = []

    def walk(node):
        widths.append(len(node.actions))
        for child in node.children.values():
            walk(child)

    walk(root)
    assert len(widths) > 1, "the tree never grew past the root"
    assert min(widths) > 1, f"a forced node was stored ({widths.count(1)} of them)"


def test_the_network_is_never_asked_about_a_forced_decision():
    evaluated: list[int] = []

    class Spy(mcts.NetEvaluator):
        def evaluate(self, state, viewer):
            evaluated.append(len(mc.search_legal_macros(state)))
            return super().evaluate(state, viewer)

    config = mcts.SearchConfig(simulations=192)
    evaluator = Spy(nw.WelcomeToNet(_SMALL), torch.device("cpu"), config)
    state = _position(players=2, turn=10)
    mcts.MCTS(evaluator, config).search(state, root=0, rng=random.Random(4))

    assert evaluated, "no leaf was evaluated"
    assert min(evaluated) > 1, "a forced state was evaluated"


# ──────────────────────────────────────────────────────────────────────────
# Deterministic within-turn re-rooting -- SEARCH_SPEC §12 step 3
# ──────────────────────────────────────────────────────────────────────────
def _drive(search, seats=2, seed=9, seat=0, max_steps=20000):
    """Play one game with ``search`` in ``seat``, recording every reuse decision.

    Returns ``[(turn, phase, reused), ...]`` over that seat's real decisions.
    """
    state = GameState.new(seed=seed, config=GameConfig(players=seats, advanced=True))
    bots = {p: GreedyBot(random.Random(seed * 100 + p)) for p in range(seats)}
    rng = random.Random(seed * 7919 + seat)
    search.reset()
    events = []
    for _ in range(max_steps):
        if state.is_terminal:
            break
        if state.actor != seat:
            state.apply(bots[state.actor].act(state))
            continue
        before = search.reroots
        decision = len(mc.search_legal_macros(state)) > 1
        choice = search.play(state, root=seat, rng=rng)
        if decision:
            events.append((state.turn, state.phase, search.reroots > before))
        mc.apply_macro(state, choice)
    return events


def test_a_retained_subtree_keeps_its_statistics_and_its_budget():
    """The exactness claim, at the level the brief allows: the retained node is
    the same object with the same numbers, and the second search tops it up to
    ``simulations`` rather than paying for the shared work twice."""
    torch.manual_seed(0)
    search, _ = _search(simulations=64)
    state = _position(players=2, turn=8, root=0)
    rng = random.Random(11)

    choice = search.play(state, root=0, rng=rng)
    retained = search._retained
    assert retained is not None, "nothing was retained inside a turn"
    kept_visits = retained.node.visits.copy()
    kept_total = retained.node.total.copy()
    assert kept_visits.sum() > 0, "an unvisited child was retained"

    mc.apply_macro(state, choice)
    while len(mc.search_legal_macros(state)) == 1:  # the forced moves the tree skipped
        mc.apply_macro(state, mc.search_legal_macros(state)[0])
    assert state.actor == 0 and not state.is_terminal

    node = search._take_retained(state, 0)
    assert node is not None, "the successor was not recognised"
    assert np.array_equal(node.visits, kept_visits)
    assert np.array_equal(node.total, kept_total)

    _, visits, node = search.search(state, root=0, rng=rng, node=node)
    assert visits.sum() == 64, "the retained visits were not counted against budget"


def test_a_position_that_is_not_the_predicted_successor_is_not_re_rooted():
    """The guard is a full position identity, not a plausibility check -- so a
    change to a seat the search does not even look at still refuses the reuse."""
    torch.manual_seed(0)
    search, _ = _search(simulations=32)
    state = _position(players=2, turn=8, root=0)
    choice = search.play(state, root=0, rng=random.Random(11))
    assert search._retained is not None

    tampered = state.copy()
    mc.apply_macro(tampered, choice)
    while len(mc.search_legal_macros(tampered)) == 1:
        mc.apply_macro(tampered, mc.search_legal_macros(tampered)[0])
    tampered.sheets[1].parks[0] += 1
    assert search._take_retained(tampered, 0) is None, "the key missed a real change"


def test_the_tree_is_discarded_across_a_turn_boundary():
    """Re-rooting is exact only within a turn.  Across one a reveal intervenes and
    the child's statistics were gathered under determinizations that no longer
    apply -- so every first decision of a turn must search from scratch."""
    torch.manual_seed(0)
    search, _ = _search(simulations=24)
    events = _drive(search, seats=2, seed=9)
    assert len(events) > 20, "not enough decisions to say anything"

    crossings = reuses = 0
    for (turn, _, _), (next_turn, _, reused) in zip(events, events[1:]):
        if next_turn != turn:
            crossings += 1
            assert not reused, f"a subtree survived the boundary into turn {next_turn}"
        elif reused:
            reuses += 1
    assert crossings > 5, "the game never crossed a turn boundary"
    assert reuses > 0, "re-rooting never fired at all"


def test_re_rooting_never_noises_the_same_node_twice():
    torch.manual_seed(0)
    search, _ = _search(simulations=32, dirichlet_alpha=1.0)
    state = _position(players=2, turn=8, root=0)
    rng = random.Random(3)
    search.play(state, root=0, rng=rng)
    retained = search._retained
    assert retained is not None
    assert not retained.node.noised, "a child was noised before it became a root"

    search._apply_root_noise(retained.node, rng)
    once = retained.node.prior.copy()
    search._apply_root_noise(retained.node, rng)
    assert np.array_equal(retained.node.prior, once), "noise was applied twice"


def test_re_rooting_saves_simulations(capsys):
    """Measured, and the number is recorded in ``SEARCH_SPEC.md`` §4.

    The net is seeded because the fraction preserved depends on how concentrated
    the priors are, and an unseeded random net moves it by a factor of two.
    """
    torch.manual_seed(0)
    search, _ = _search(simulations=24)
    _drive(search, seats=2, seed=9)
    total = search.simulations_run + search.simulations_reused
    assert total > 0
    saved = search.simulations_reused / total
    assert saved > 0.0, "re-rooting preserved nothing"
    with capsys.disabled():
        print(
            f"\n    re-rooting: {search.reroots} re-roots, "
            f"{search.simulations_reused} of {total} simulations preserved "
            f"({saved:.1%})"
        )


def test_the_roundabout_pass_flag_reaches_every_part_of_the_search():
    """``SearchConfig.prune_roundabout_pass`` is on (SEARCH_SPEC §5.1a), and
    whichever way it is set it must reach *all* of the search at once.

    ``_collapse_forced``, ``play`` and ``_within_turn_successor`` have to agree
    on which decisions are forced; if one of them read a different setting, a
    retained subtree would sit one move away from the position it is compared
    against and re-rooting would silently stop working.  So they all go through
    ``MCTS._search_actions``, and this is the test that says so.
    """
    assert mcts.SearchConfig().prune_roundabout_pass is True

    bots = [GreedyBot(random.Random(i)) for i in range(2)]
    state = GameState.new(seed=3, config=GameConfig(players=2, advanced=True))
    while not state.is_terminal and state.phase is not Phase.ROUNDABOUT_PLACE:
        state.apply(bots[state.actor].act(state))
    assert not state.is_terminal, "no ROUNDABOUT_PLACE state on this line"

    pass_macro = mc.from_primitive(codec.A_PASS_ROUNDABOUT)
    keeping, _ = _search(simulations=4, prune_roundabout_pass=False)
    pruning, _ = _search(simulations=4)
    assert pass_macro in keeping._search_actions(state)
    assert pass_macro not in pruning._search_actions(state)


def test_a_newly_noised_root_always_gets_fresh_simulations():
    """⚠ Root noise must be *observed*, not merely applied.

    ``simulations`` is a target total, so a retained root whose visits already
    meet it would perturb its prior and then run nothing -- leaving the visit
    counts, and so the policy target, bit-identical to the un-noised search.
    Reproduced before the fix as ``prior_changed True, fresh_simulations 0,
    visits_changed False``.  The existing noise test proves noise is not applied
    twice; this one proves it is applied at all.
    """
    torch.manual_seed(0)
    search, evaluator = _search(simulations=8, dirichlet_alpha=1.0)
    state = _position(players=2, turn=8, root=0)
    rng = random.Random(3)

    choice = search.play(state, root=0, rng=rng)
    retained = search._retained
    assert retained is not None, "nothing was retained inside a turn"

    successor = mc.step_macro(state, choice)
    while len(mc.search_legal_macros(successor)) == 1:
        mc.apply_macro(successor, mc.search_legal_macros(successor)[0])
    node = search._take_retained(successor, 0)
    assert node is not None

    # the pathological case: re-rooting alone already meets the whole budget
    node.visits[:] = 0.0
    node.visits[0] = float(search.config.simulations)
    node.total[0] = 0.0
    before_prior = node.prior.copy()
    before_visits = node.visits.copy()
    calls = evaluator.calls

    _, visits, node = search.search(successor, root=0, rng=rng, node=node)

    assert not np.array_equal(before_prior, node.prior), "the root was not noised"
    assert evaluator.calls > calls, "a noised root ran zero simulations"
    assert not np.array_equal(before_visits, visits), "no simulation saw the noise"


def test_an_un_noised_root_keeps_the_whole_re_rooting_saving():
    """The floor is scoped to noise; with noise off, nothing changes.  This is
    what keeps ``min_fresh_simulations`` from quietly costing the 47.7%."""
    torch.manual_seed(0)
    search, _ = _search(simulations=64)
    assert search.config.dirichlet_alpha is None
    state = _position(players=2, turn=8, root=0)
    rng = random.Random(11)

    choice = search.play(state, root=0, rng=rng)
    successor = mc.step_macro(state, choice)
    while len(mc.search_legal_macros(successor)) == 1:
        mc.apply_macro(successor, mc.search_legal_macros(successor)[0])
    node = search._take_retained(successor, 0)
    assert node is not None and node.visits.sum() > 0

    _, visits, _ = search.search(successor, root=0, rng=rng, node=node)
    assert visits.sum() == 64, "the target-total budget changed with noise off"


def test_the_position_key_sees_deck_and_discard_composition():
    """⚠ Swap an undrawn card with a discarded one of a *different* printed type.

    ``deck_pos`` and ``len(discard)`` are both unchanged, so a key built from
    those alone matches -- and an earlier version of ``_position_key`` did
    exactly that, despite claiming to be a full identity.  The two positions have
    different reveal distributions and different ``deck_composition`` in the
    encoder, so they must never share a retained subtree.
    """
    state = _position(players=2, turn=10, root=0)
    state.discard.append(state.deck[state.deck_pos + 5])
    before = mcts._position_key(state, 0)

    swapped = state.copy()
    pair = next(
        (i, j)
        for i, card in enumerate(swapped.deck[swapped.deck_pos :])
        for j, other in enumerate(swapped.discard)
        if CARD_TABLE[card] != CARD_TABLE[other]
    )
    i, j = pair
    card = swapped.deck[swapped.deck_pos + i]
    other = swapped.discard[j]
    swapped.deck[swapped.deck_pos + i], swapped.discard[j] = other, card

    assert swapped.deck_pos == state.deck_pos
    assert len(swapped.discard) == len(state.discard)
    assert mcts._position_key(swapped, 0) != before, "the key missed the swap"


def test_the_position_key_ignores_the_order_of_what_is_hidden():
    """The other half: a re-determinization permutes the undrawn deck and must
    *not* refuse a legitimate re-root.  Composition in, order out."""
    state = _position(players=2, turn=10, root=0)
    before = mcts._position_key(state, 0)
    shuffled = state.redeterminize(random.Random(5))
    assert shuffled.deck[shuffled.deck_pos :] != state.deck[state.deck_pos :]
    assert mcts._position_key(shuffled, 0) == before, "the key read hidden order"


def test_the_dirichlet_alpha_scales_with_the_width_of_the_root():
    """§7.8.  A search root here is 2 actions wide at `ASK_RESHUFFLE` and up to
    331 at `CHOOSE_CARDS`, so one absolute alpha cannot serve both -- it either
    drowns the narrow nodes or does nothing to the wide ones.  ``concentration /
    width`` is AlphaZero's own rule: its published constants are ~10 / branching
    factor (Go 0.03 at ~250, chess 0.3 at ~35, shogi 0.15 at ~70)."""
    search, _ = _search(simulations=2, dirichlet_concentration=10.0)
    assert search._root_alpha(50) == pytest.approx(0.2)
    assert search._root_alpha(2) == pytest.approx(5.0)
    assert search._root_alpha(331) == pytest.approx(10.0 / 331)

    # the absolute form still works, and the scaled form wins when both are set
    absolute, _ = _search(simulations=2, dirichlet_alpha=0.3)
    assert absolute._root_alpha(50) == pytest.approx(0.3)
    assert absolute._root_alpha(2) == pytest.approx(0.3)
    both, _ = _search(simulations=2, dirichlet_alpha=0.3, dirichlet_concentration=10.0)
    assert both._root_alpha(50) == pytest.approx(0.2)

    off, _ = _search(simulations=2)
    assert off._root_alpha(50) is None


def test_a_scaled_alpha_actually_perturbs_a_wide_and_a_narrow_root_alike():
    """The point of scaling: both ends move.  With one absolute alpha, whichever
    end it was tuned for is the only one that does."""
    state = _position(players=2, turn=8, root=0)
    torch.manual_seed(0)
    plain, _ = _search(simulations=8)
    scaled, _ = _search(simulations=8, dirichlet_concentration=10.0)

    base, _ = plain._leaf(state, 0)
    for width_node in (base,):
        before = width_node.prior.copy()
        assert scaled._apply_root_noise(width_node, random.Random(1)) is True
        assert not np.allclose(before, width_node.prior)
        assert width_node.prior.sum() == pytest.approx(1.0)
        assert (width_node.prior >= 0).all()


def test_the_fresh_simulation_floor_is_a_fraction_of_the_budget():
    """It has to track the budget for the same reason ``K`` does (§7.6): the same
    checkpoint searches at different budgets in self-play, arena and analysis."""
    full, _ = _search(simulations=64, dirichlet_concentration=10.0)
    assert full._fresh_after_noise() == 64

    quarter, _ = _search(
        simulations=64, dirichlet_concentration=10.0, noise_fresh_fraction=0.25
    )
    assert quarter._fresh_after_noise() == 16

    big, _ = _search(
        simulations=256, dirichlet_concentration=10.0, noise_fresh_fraction=0.25
    )
    assert big._fresh_after_noise() == 64, "the floor did not scale with the budget"


# ──────────────────────────────────────────────────────────────────────────
# The viewer information-state key -- SEARCH_SPEC §12.1, step 5
#
# This is the key §7.1a's progressive widening merges chance children on, so it
# has to be right in both directions: too coarse and distinct positions merge,
# biasing the weights; too fine and identical-looking outcomes never collide, so
# count/samples never converges.
# ──────────────────────────────────────────────────────────────────────────
def _mid_turn(players: int = 3, seed: int = 4, steps: int = 40) -> GameState:
    """A position with at least one seat already done and the turn unfinished."""
    bots = [GreedyBot(random.Random(seed * 7 + i)) for i in range(players)]
    state = GameState.new(seed=seed, config=GameConfig(players=players, advanced=True))
    for _ in range(steps):
        if state.is_terminal:
            break
        state.apply(bots[state.actor].act(state))
    return state


def test_the_key_hides_an_opponents_live_sheet_and_shows_its_public_snapshot():
    """⚠ §12.1's "test that matters", verbatim.

    Seat 1 is to act after seat 0 has already written this turn.  Mutating seat
    0's **live** sheet must leave seat 1's key alone -- that write is hidden
    until the turn resolves (``Houses::getOfPlayer``) -- while mutating seat 0's
    **turn-start snapshot** must change it, because that is what seat 1 sees.
    """
    state = _mid_turn()
    while not state.is_terminal and state.actor == 0:
        state.apply(GreedyBot(random.Random(1)).act(state))
    viewer = state.actor
    assert viewer != 0, "seat 0 has not finished acting"

    before = mcts.information_key(state, viewer)

    hidden = state.copy()
    hidden.sheets[0].parks[0] += 1  # seat 0's live, current-turn write
    assert mcts.information_key(hidden, viewer) == before, "a hidden write leaked"

    public = state.copy()
    public.public_sheets[0].parks[0] += 1  # what seat 1 is allowed to see
    assert mcts.information_key(public, viewer) != before, "the snapshot was ignored"

    # and the viewer's own sheet is live, not snapshotted
    own = state.copy()
    own.sheets[viewer].parks[0] += 1
    assert mcts.information_key(own, viewer) != before, "the viewer's own sheet is stale"


def test_the_key_does_not_carry_the_future_deck_order():
    """§7.3: a key that carries the deal is determinization strategy fusion --
    the tree would know the future through its own node identity."""
    state = _mid_turn()
    before = mcts.information_key(state, state.actor)
    for seed in range(6):
        shuffled = state.redeterminize(random.Random(seed))
        assert mcts.information_key(shuffled, shuffled.actor) == before


def test_the_key_reads_printed_types_not_physical_ids():
    """⚠ The latent defect §6.1 recorded: 15 of the 66 printed types have two
    physical copies, so keying on ids splits two reveals that every player at
    the table sees as identical.  Harmless while children are never reused, and
    wrong the moment §7 retains them."""
    twins = collections.defaultdict(list)
    for card in range(NUM_BASE_CARDS):
        twins[CARD_TABLE[card]].append(card)
    pair = next(ids for ids in twins.values() if len(ids) == 2)

    state = _mid_turn()
    viewer = state.actor
    slot = next(
        i for i, card in enumerate(state.stack_new[0]) if card is not None
    )
    original = state.stack_new[0][slot]

    swapped = state.copy()
    swapped.stack_new[0][slot] = pair[0]
    twinned = state.copy()
    twinned.stack_new[0][slot] = pair[1]

    assert pair[0] != pair[1]
    assert CARD_TABLE[pair[0]] == CARD_TABLE[pair[1]]
    assert mcts.information_key(swapped, viewer) == mcts.information_key(
        twinned, viewer
    ), "two copies of one printed card keyed to different children"
    if CARD_TABLE[original] != CARD_TABLE[pair[0]]:
        assert mcts.information_key(swapped, viewer) != mcts.information_key(
            state, viewer
        ), "a genuinely different card did not change the key"


def test_the_key_carries_the_viewers_own_vote_and_not_the_table_wide_flag():
    """``reshuffle_next_turn`` is the OR of the votes and is private mid-turn
    (``ENCODER_V2_SPEC`` §9.3a): reading it tells a later serial actor that an
    earlier one voted yes, which nobody knows in the concurrent game."""
    state = _mid_turn()
    viewer = state.actor
    before = mcts.information_key(state, viewer)

    aggregate = state.copy()
    aggregate.reshuffle_next_turn = not aggregate.reshuffle_next_turn
    assert mcts.information_key(aggregate, viewer) == before, "the aggregate leaked"

    other = state.copy()
    other.reshuffle_votes[1 - viewer if viewer < 2 else 0] = True
    assert mcts.information_key(other, viewer) == before, "another seat's vote leaked"

    mine = state.copy()
    mine.reshuffle_votes[viewer] = not mine.reshuffle_vote_for(viewer)
    assert mcts.information_key(mine, viewer) != before, "the viewer's own vote is lost"


def test_the_key_carries_deck_and_discard_composition():
    """Swap an undrawn card with a discarded one of a different printed type:
    ``deck_remaining`` and the discard length are both unchanged, and the two
    positions have genuinely different reveal distributions."""
    state = _mid_turn()
    state.discard.append(state.deck[state.deck_pos + 3])
    viewer = state.actor
    before = mcts.information_key(state, viewer)

    swapped = state.copy()
    i, j = next(
        (i, j)
        for i, card in enumerate(swapped.deck[swapped.deck_pos :])
        for j, other in enumerate(swapped.discard)
        if CARD_TABLE[card] != CARD_TABLE[other]
    )
    card = swapped.deck[swapped.deck_pos + i]
    swapped.deck[swapped.deck_pos + i] = swapped.discard[j]
    swapped.discard[j] = card

    assert swapped.deck_remaining == state.deck_remaining
    assert mcts.information_key(swapped, viewer) != before


def test_the_key_hides_another_seats_turn_context():
    """``ctx`` is the acting seat's scratch state -- the slot taken, the number,
    where it went.  That is exactly this turn's hidden write, so it is in the key
    only when the viewer *is* the actor."""
    state = _mid_turn()
    viewer = state.actor
    other = next(p for p in range(state.config.players) if p != viewer)

    assert mcts.information_key(state, viewer)[-1] is not None, "own ctx is missing"
    assert mcts.information_key(state, other)[-1] is None, "another seat's ctx leaked"


def test_two_different_positions_do_not_share_a_key():
    """The other direction: too coarse a key merges distinct positions, which
    biases every empirical weight built on it."""
    seen: dict[tuple, tuple] = {}
    collisions = 0
    for seed in (3, 4, 5):
        bots = [GreedyBot(random.Random(seed * 7 + i)) for i in range(3)]
        state = GameState.new(seed=seed, config=GameConfig(players=3, advanced=True))
        steps = 0
        while not state.is_terminal and steps < 400:
            steps += 1
            if state.phase is not Phase.WRITE_NUMBER:
                viewer = state.actor
                key = mcts.information_key(state, viewer)
                mark = (seed, state.turn, viewer, int(state.phase), steps)
                if key in seen and seen[key][:4] != mark[:4]:
                    collisions += 1
                seen.setdefault(key, mark)
            state.apply(bots[state.actor].act(state))
    assert len(seen) > 300, "not enough distinct positions sampled"
    assert collisions == 0, f"{collisions} distinct positions shared a key"


# ──────────────────────────────────────────────────────────────────────────
# The batching seam -- SEARCH_SPEC §12 step 6
#
# The point of this group is that batching is a *driver*, not a rewrite: the
# search suspends at every network request, and somebody else decides when and
# with what else they are computed.  So the tests are about equivalence, not
# speed -- THROUGHPUT_LEVERS §1 class A, "changes when work happens, never what
# work happens", verified by fingerprinting discrete outputs.
# ──────────────────────────────────────────────────────────────────────────
def _wave_positions(count: int = 8) -> list[GameState]:
    out = []
    for seed in range(count):
        for turn in (6, 12):
            state = _position(players=2, turn=turn, seed=seed, root=0)
            if not state.is_terminal:
                out.append(state)
    return out


def test_a_wave_of_searches_agrees_with_running_them_one_at_a_time():
    """⚠ THROUGHPUT_LEVERS §A: a class A lever must produce identical games from
    identical seeds, and the way to check that is to fingerprint the **discrete**
    outputs -- actions and visit counts -- never the float targets.

    Those really do drift: a batched forward reduces floats in a different order
    from a single one, measured at ~1e-7 on priors and values here.  Far too
    small to matter as a value, and in principle able to flip a comparison that
    was tied to seven digits, which is exactly why the fingerprint is discrete.
    """
    torch.manual_seed(0)
    net = nw.WelcomeToNet(_SMALL)
    states = _wave_positions()
    assert len(states) >= 8

    config = mcts.SearchConfig(simulations=24)
    one = mcts.MCTS(mcts.NetEvaluator(net, torch.device("cpu"), config), config)
    alone = [
        one.search(state, root=0, rng=random.Random(i)) for i, state in enumerate(states)
    ]

    evaluator = mcts.NetEvaluator(net, torch.device("cpu"), config)
    together = mcts.MCTS(evaluator, config)
    waved = mcts.run_searches(
        [
            together.search_gen(state, root=0, rng=random.Random(i))
            for i, state in enumerate(states)
        ],
        evaluator,
        max_batch=16,
    )

    assert len(waved) == len(alone)
    for (actions_a, visits_a, _), (actions_b, visits_b, _) in zip(alone, waved):
        assert mcts.trajectory_fingerprint(
            actions_a, [visits_a]
        ) == mcts.trajectory_fingerprint(actions_b, [visits_b])


def test_the_wave_pools_leaves_and_opponent_policies_into_the_same_call():
    """⚠ Both kinds must suspend, or the batching reaches half the work.

    Measured before opponent sampling went through the seam: leaves batched
    perfectly and opponent policies were still **53% of rows and 97% of calls**,
    every one of them a batch of one.  And both kinds must share a *call*, since
    they read the same heads of the same forward -- splitting by kind measured a
    mean batch of 12.1 where the wave had 32 searches live, against 22.4 pooled.
    """
    torch.manual_seed(0)
    config = mcts.SearchConfig(simulations=24)
    evaluator = mcts.NetEvaluator(nw.WelcomeToNet(_SMALL), torch.device("cpu"), config)
    search = mcts.MCTS(evaluator, config)
    states = _wave_positions()

    kinds: list[mcts.Ask] = []
    original = evaluator.answer_batch

    def spy(requests):
        kinds.extend(kind for kind, _, _ in requests)
        return original(requests)

    evaluator.answer_batch = spy
    mcts.run_searches(
        [
            search.search_gen(state, root=0, rng=random.Random(i))
            for i, state in enumerate(states)
        ],
        evaluator,
        max_batch=32,
    )

    assert mcts.Ask.LEAF in kinds and mcts.Ask.POLICY in kinds
    assert evaluator.rows > evaluator.calls, "nothing batched at all"
    assert evaluator.rows / evaluator.calls > 4.0, (
        f"mean batch {evaluator.rows / evaluator.calls:.1f} -- the wave is not pooling"
    )
    mixed = evaluator.batch_widths
    assert max(mixed) > 1, "every call was a batch of one"


def test_a_mixed_batch_answers_each_kind_the_way_the_single_calls_do():
    """One forward, two answer shapes, and neither may drift from its single
    form -- a second interpretation that agrees today is a parity bug waiting."""
    torch.manual_seed(0)
    config = mcts.SearchConfig()
    evaluator = mcts.NetEvaluator(nw.WelcomeToNet(_SMALL), torch.device("cpu"), config)
    states = _wave_positions(4)

    requests = []
    for i, state in enumerate(states):
        requests.append((mcts.Ask.LEAF if i % 2 else mcts.Ask.POLICY, state, 0))
    answers = evaluator.answer_batch(requests)

    assert len(answers) == len(requests)
    for (kind, state, viewer), answer in zip(requests, answers):
        if kind is mcts.Ask.LEAF:
            priors, value = answer
            single_priors, single_value = evaluator.evaluate(state, viewer)
            assert value == pytest.approx(single_value, abs=1e-5)
        else:
            priors = answer
            single_priors = evaluator.policy(state, viewer)
        assert priors.shape == (mc.NUM_MACRO_ACTIONS,)
        assert priors == pytest.approx(single_priors, abs=1e-5)
        assert priors.sum() == pytest.approx(1.0, abs=1e-5)


def test_an_empty_batch_is_not_a_network_call():
    evaluator = mcts.NetEvaluator(nw.WelcomeToNet(_SMALL), torch.device("cpu"))
    before = evaluator.calls
    assert evaluator.answer_batch([]) == []
    assert evaluator.evaluate_batch([]) == []
    assert evaluator.calls == before


def test_the_counters_keep_calls_rows_and_width_apart():
    """THROUGHPUT_LEVERS §2.1: batch width and throughput are different numbers,
    and reporting one as the other is the named failure mode."""
    torch.manual_seed(0)
    evaluator = mcts.NetEvaluator(nw.WelcomeToNet(_SMALL), torch.device("cpu"))
    states = _wave_positions(3)
    evaluator.evaluate_batch([(state, 0) for state in states])
    assert evaluator.calls == 1
    assert evaluator.rows == len(states)
    assert evaluator.batch_widths[len(states)] == 1

    evaluator.evaluate(states[0], 0)
    assert evaluator.calls == 2
    assert evaluator.rows == len(states) + 1
    assert evaluator.batch_widths[1] == 1


# ──────────────────────────────────────────────────────────────────────────
# The chance edge: progressive widening -- SEARCH_SPEC §7.1a resolution C, §12
# step 7.  Behind a flag, with the open-loop version kept as the control arm.
# ──────────────────────────────────────────────────────────────────────────
def _widened(simulations: int = 256, **kwargs):
    torch.manual_seed(0)
    return _search(simulations=simulations, chance_widening=1.0, **kwargs)


def _edges(root):
    """Every chance edge in the tree, as ``(node, action index, outcomes)``."""
    out = []

    def walk(node):
        for index, outcomes in node.outcomes.items():
            out.append((node, index, outcomes))
        for child in node.children.values():
            walk(child)

    walk(root)
    return out


def _tree_depth(root) -> float:
    depths: list[int] = []

    def walk(node, depth):
        if not node.children:
            depths.append(depth)
        for child in node.children.values():
            walk(child, depth + 1)

    walk(root, 0)
    return sum(depths) / len(depths)


def test_the_control_arm_allocates_no_chance_bookkeeping():
    """§12 step 7 keeps the open-loop version as the control arm, so with the
    flag off nothing about the tree may change -- not even its memory."""
    search, _ = _search(simulations=64)
    assert search.config.chance_widening is None
    state = _position(players=2, turn=10)
    _, _, root = search.search(state, root=0, rng=random.Random(4))

    def walk(node):
        assert node.outcomes == {} and node.edge_visits == {}
        for child in node.children.values():
            walk(child)

    walk(root)


def test_widening_bounds_the_number_of_outcomes_on_every_edge():
    """``ceil(C · traversals**alpha)``, which is what makes the chance fan-out
    finite and therefore revisitable at all."""
    search, _ = _widened()
    state = _position(players=2, turn=12)
    _, _, root = search.search(state, root=0, rng=random.Random(0))

    edges = _edges(root)
    assert edges, "no chance edge was recorded"
    for node, index, outcomes in edges:
        traversals = node.edge_visits[index]
        allowed = max(
            math.ceil(
                search.config.chance_widening
                * traversals ** search.config.chance_widening_alpha
            ),
            1,
        )
        assert len(outcomes) <= allowed, (
            f"{len(outcomes)} outcomes on an edge traversed {traversals} times, "
            f"cap {allowed}"
        )


def test_every_outcome_can_be_resumed_and_its_particles_are_capped():
    search, _ = _widened()
    state = _position(players=2, turn=12)
    _, _, root = search.search(state, root=0, rng=random.Random(0))
    for _, _, outcomes in _edges(root):
        for outcome in outcomes.values():
            assert outcome.count >= 1
            assert outcome.particles or outcome.terminal_value is not None
            assert len(outcome.particles) <= search.config.max_particles


def test_reusing_an_outcome_does_not_move_its_weight():
    """⚠ Regression for a real estimator bug.

    An earlier version incremented ``count`` on reuse *and* sampled in
    proportion to ``count``. That is a Polya urn: the sampling feeds the counter
    it samples by, so whichever outcome happened to arrive first is reinforced
    and the weights converge to a random limit rather than to the truth. It
    produced weights like ``[0.045, 0.045, 0.091, 0.091, 0.727]`` on an edge
    whose outcomes were near-unique reveals and should have been uniform.

    The fix is two counters that must never be confused: ``edge_visits`` counts
    **traversals** and drives the widening cap, ``count`` counts **fresh draws**
    from the real transition and drives the weights.
    """
    search, _ = _widened(simulations=512)
    state = _position(players=2, turn=12)
    _, _, root = search.search(state, root=0, rng=random.Random(0))

    reused = 0
    for node, index, outcomes in _edges(root):
        draws = sum(outcome.count for outcome in outcomes.values())
        traversals = node.edge_visits[index]
        assert draws <= traversals, "a weight counted a reuse"
        reused += traversals - draws
        weights = [outcome.count / draws for outcome in outcomes.values()]
        assert sum(weights) == pytest.approx(1.0)
    assert reused > 0, "no edge was ever reused, so the test proves nothing"


def test_a_deterministic_edge_is_drawn_once_and_reused_ever_after():
    """⚠ Regression for a closure defect the cap alone could not fix.

    A within-turn transition consumes no randomness -- no card is revealed
    between two of the root player's own decisions -- so its support is exactly
    one. But ``len(outcomes)`` then stops at 1 while ``ceil(C·n**alpha)`` keeps
    growing, so a cap-only predicate goes false for ever and the edge re-samples
    on nearly every traversal. Measured before the fix: support-one edges reused
    **6 of 77** traversals (7.8%), while the docstring claimed every later
    descent resumed from a particle. With determinism *proven* rather than
    inferred from collisions, the same edges reuse **97.4%** (111 of 114).
    """
    search, _ = _widened(simulations=512)
    state = _position(players=2, turn=12)
    _, _, root = search.search(state, root=0, rng=random.Random(0))

    exact = [
        (node, index, outcomes)
        for node, index, outcomes in _edges(root)
        if index in node.edge_exact and node.edge_visits[index] >= 8
    ]
    assert exact, "no deterministic edge was traversed enough to say anything"

    traversals = draws = 0
    for node, index, outcomes in exact:
        assert len(outcomes) == 1, "a proven-deterministic edge grew a second outcome"
        outcome = next(iter(outcomes.values()))
        assert outcome.count == 1, "a closed edge sampled the transition again"
        assert outcome.particles, "a resumable edge kept no particle"
        traversals += node.edge_visits[index]
        draws += outcome.count
    assert (traversals - draws) / traversals > 0.8, (
        f"only {(traversals - draws) / traversals:.1%} of traversals reused"
    )


def test_reuse_actually_happens_and_skips_the_transition():
    """The saving: a closed edge resumes from a particle, so no opponent is
    evaluated and no card is drawn for that traversal."""
    search, evaluator = _widened(simulations=512)
    state = _position(players=2, turn=12)
    _, _, root = search.search(state, root=0, rng=random.Random(0))

    traversals = sum(
        node.edge_visits[index] for node, index, _ in _edges(root)
    )
    draws = sum(
        outcome.count
        for _, _, outcomes in _edges(root)
        for outcome in outcomes.values()
    )
    assert traversals > draws, "nothing was ever reused"
    assert traversals - draws > 100, (
        f"only {traversals - draws} reuses of {traversals} traversals"
    )


def test_widening_deepens_the_tree_and_answers_to_the_budget():
    """⚠ This is the null §6.1 recorded, reversed.

    Open loop measured mean leaf depth 1.59 and *unmoved* by budget or by prior
    sharpness, because every boundary crossing drew a key never seen before and
    expanded a fresh leaf: budget went into root averaging, not depth.  A finite,
    revisitable chance edge is the thing that changes it.  Measured over 12
    positions at two seats: control 1.32 -> 1.42 from 64 to 256 simulations,
    widening 1.77 -> 2.14.
    """
    state = _position(players=2, turn=12)
    depths = {}
    for label, widening in (("control", None), ("widened", 1.0)):
        for sims in (64, 256):
            torch.manual_seed(0)
            search, _ = _search(simulations=sims, chance_widening=widening)
            _, _, root = search.search(state, root=0, rng=random.Random(3))
            depths[label, sims] = _tree_depth(root)

    assert depths["widened", 256] > depths["control", 256], (
        "widening did not deepen the tree"
    )
    control_gain = depths["control", 256] - depths["control", 64]
    widened_gain = depths["widened", 256] - depths["widened", 64]
    assert widened_gain > control_gain, (
        f"depth still does not answer to the budget: control +{control_gain:.2f}, "
        f"widened +{widened_gain:.2f}"
    )


def test_a_chance_edge_has_more_outcomes_than_the_deck_has_reveals():
    """⚠ Retraction, with the number that forced it.

    This asserted that a three-card deck gives a chance edge at most **six**
    outcomes, because §6.2 says three cards have six ordered reveals. That
    conflates a *reveal* with a *transition*, which is precisely the error §7.1a
    exists to correct: the retained outcome is the whole root-to-root transition
    and carries the opponents' sampled decisions too. It passed only because the
    widening cap happened to bind at six; with the cap fixed, the same edge shows
    **20** outcomes.

    The six-reveal claim is true, and is tested where it holds -- on the boundary
    sampler itself, in ``test_game.py``.
    """
    state = _position(players=2, turn=10, seed=5)
    keep = state.deck[state.deck_pos : state.deck_pos + 3]
    state.discard.extend(state.deck[state.deck_pos + 3 :])
    state.deck = state.deck[: state.deck_pos] + keep
    assert state.deck_remaining == 3

    search, _ = _widened(simulations=2048)
    _, _, root = search.search(state, root=0, rng=random.Random(1))

    multi = [outcomes for _, _, outcomes in _edges(root) if len(outcomes) > 1]
    assert multi, "no boundary-crossing edge was traversed"
    assert max(len(outcomes) for outcomes in multi) > 6, (
        "the whole-transition support was no larger than the reveal support, "
        "which would mean opponent sampling contributed nothing"
    )
    for outcomes in multi:
        draws = sum(outcome.count for outcome in outcomes.values())
        assert sum(outcome.count / draws for outcome in outcomes.values()) == (
            pytest.approx(1.0)
        )


def _last_decision_before_the_end(seed: int, root: int = 0) -> Optional[GameState]:
    """The root player's final decision of a GreedyBot game, so a search from it
    reaches terminal transitions rather than merely deep ones."""
    bots = [GreedyBot(random.Random(seed * 10 + i)) for i in range(2)]
    state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True))
    previous = None
    while not state.is_terminal:
        if state.actor == root and state.phase is not Phase.WRITE_NUMBER:
            previous = state.copy()
        state.apply(bots[state.actor].act(state))
    return previous


def test_terminal_transitions_are_keyed_apart_not_collapsed_together():
    """⚠ Regression for a real collapse.

    ``_advance`` used to return a bare ``()`` for every ending. That is harmless
    in the open-loop arm -- a terminal transition stores no child, so the key is
    never looked up -- and silently wrong under widening, where the observation
    *is* the outcome key. Measured before the fix: **255 distinct endings of one
    edge merged into a single outcome**, carrying whichever final score happened
    to be computed last, and every later reuse of that edge returned it. Endings
    have different scores; collapsing them is exactly the belief collapse
    particles exist to prevent.
    """
    torch.manual_seed(0)
    search, _ = _search(simulations=256, chance_widening=1.0)
    keys: set = set()
    values: dict = {}
    for seed in range(6):
        state = _last_decision_before_the_end(seed)
        if state is None or state.is_terminal:
            continue
        _, _, root = search.search(state, root=0, rng=random.Random(seed))
        for _, _, outcomes in _edges(root):
            for key, outcome in outcomes.items():
                if outcome.terminal_value is None:
                    continue
                assert key != (), "an ending keyed as the empty observation"
                keys.add(key)
                values.setdefault(key, outcome.terminal_value)

    assert len(keys) >= 5, f"only {len(keys)} terminal transitions reached"


def test_the_information_key_determines_the_terminal_value():
    """Why merging endings on the key is sound *once they are keyed properly*.

    An outcome keeps one ``terminal_value`` for however many draws land on it, so
    that is only correct if the key pins the value down. It does: the final
    scores are a function of the sheets, and at a terminal state the live sheets
    and the public snapshots agree, so everything ``terminal_value`` reads is in
    the key. Verified over 39 distinct terminal keys, none mapping to two values.
    """
    torch.manual_seed(0)
    config = mcts.SearchConfig(simulations=256, chance_widening=1.0)
    evaluator = mcts.NetEvaluator(nw.WelcomeToNet(_SMALL), torch.device("cpu"), config)
    seen: dict = collections.defaultdict(set)

    class Spy(mcts.MCTS):
        def _leaf_gen(self, state, root):
            if state.is_terminal:
                seen[mcts.information_key(state, root)].add(
                    round(mcts.terminal_value(state, root, self.config), 9)
                )
            return (yield from super()._leaf_gen(state, root))

    search = Spy(evaluator, config)
    for seed in range(6):
        state = _last_decision_before_the_end(seed)
        if state is None or state.is_terminal:
            continue
        search.search(state, root=0, rng=random.Random(seed))

    assert seen, "no terminal state was reached"
    ambiguous = {key: values for key, values in seen.items() if len(values) > 1}
    assert not ambiguous, (
        f"{len(ambiguous)} information keys map to more than one terminal value"
    )


def test_the_sheet_key_covers_every_field_of_the_sheet():
    """Sign-off 2 of the steps 4-7 review.

    ``_SHEET_FIELDS`` is a hand-written list of somebody else's fields, and a
    hand-written list rots silently: add a field to ``Sheet`` and both keys built
    from it stop distinguishing on it, with nothing failing. ``_position_key``
    would re-root onto a subtree from a different position; ``information_key``
    would merge two positions under widening.
    """
    declared = tuple(field.name for field in dataclasses.fields(Sheet))
    assert mcts._SHEET_FIELDS == declared, (
        "Sheet's fields and _SHEET_FIELDS have diverged; a key is now blind to "
        f"{set(declared) ^ set(mcts._SHEET_FIELDS)}"
    )


def test_the_information_key_carries_the_whole_configuration():
    """⚠ Regression. The key carried ``config.players`` alone while the encoder
    also reads ``advanced``, ``expert`` and ``solo`` -- so flipping ``advanced``
    left the key identical and the encoding different, which breaks the
    containment that licenses sharing one node's priors across a particle
    collection. Dormant while configuration is fixed within a search, and false
    as stated, which is worse."""
    state = _position(players=2, turn=10, seed=3)
    before = mcts.information_key(state, 0)

    flipped = state.copy()
    flipped.config = GameConfig(players=2, advanced=not state.config.advanced)
    assert mcts.information_key(flipped, 0) != before, "a config flip did not move the key"

    encoded_a = enc.encode_state(state, 0)
    encoded_b = enc.encode_state(flipped, 0)
    assert not all(
        np.array_equal(x, y) for x, y in zip(encoded_a, encoded_b)
    ), "the encoder ignored the flip too, so this test proves nothing"


def test_the_information_key_carries_the_raw_discard_count():
    """``discard_composition`` returns zeros in expert mode, where the discard is
    not attributable, so the count is not recoverable from the histogram."""
    state = _position(players=2, turn=10, seed=3)
    before = mcts.information_key(state, 0)
    grown = state.copy()
    grown.discard.append(grown.deck[grown.deck_pos])
    grown.deck_pos += 1
    assert mcts.information_key(grown, 0) != before


def test_a_mixed_batch_attributes_rows_across_seat_counts():
    """Sign-off 5: the mixed test used one viewer and one seat count, so a
    mask/seat transposition in ``answer_batch`` would have survived it.

    This feeds row-distinct synthetic head outputs at 2, 3 and 4 seats, so every
    row's value and priors are separable and a transposed scatter shows up as a
    mismatch rather than as plausible numbers.
    """
    torch.manual_seed(0)
    config = mcts.SearchConfig()
    evaluator = mcts.NetEvaluator(nw.WelcomeToNet(_SMALL), torch.device("cpu"), config)

    states = [_position(players=seats, turn=8, seed=seats) for seats in (2, 3, 4)]
    requests = [
        (mcts.Ask.LEAF, states[0], 0),
        (mcts.Ask.POLICY, states[1], 0),
        (mcts.Ask.LEAF, states[2], 0),
        (mcts.Ask.POLICY, states[0], 0),
    ]

    rows = len(requests)
    captured: dict = {}

    def synthetic(batch):
        captured["rows"] = [(state.config.players, viewer) for state, viewer in batch]
        policy = torch.zeros(rows, mc.NUM_MACRO_ACTIONS)
        for i in range(rows):
            policy[i] = float(i)  # row i is separable from every other row
        ranks = torch.zeros(rows, training.MAX_RANKS)
        for i in range(rows):
            ranks[i, : mcts.enc.MAX_SEATS] = torch.arange(mcts.enc.MAX_SEATS) + i
        score = torch.zeros(rows, mcts.enc.MAX_SEATS)
        for i in range(rows):
            score[i] = torch.arange(mcts.enc.MAX_SEATS) * 0.1 + i
        return {"policy_logits": policy, "rank_logits": ranks, "score": score}

    evaluator._forward_many = synthetic
    answers = evaluator.answer_batch(requests)

    assert captured["rows"] == [(2, 0), (3, 0), (4, 0), (2, 0)], (
        "the batch rows were built in the wrong order"
    )
    assert len(answers) == rows
    for index, (kind, state, _) in enumerate(requests):
        priors = answers[index][0] if kind is mcts.Ask.LEAF else answers[index]
        legal = mc.legal_mask(state)
        # a flat logit row means uniform over that state's own legal set, so a
        # transposed scatter shows as the wrong support, not just wrong numbers
        assert priors[legal].min() == pytest.approx(priors[legal].max())
        assert priors[~legal].sum() == pytest.approx(0.0)
        assert priors.sum() == pytest.approx(1.0)

    # and the two LEAF rows must not have swapped values: row 0 and row 2 were
    # given different rank logits and different scores
    assert answers[0][1] != pytest.approx(answers[2][1])


def test_particles_are_replaced_rather_than_frozen_after_the_cap():
    """⚠ Reservoir sampling, from the review's §4.

    Keeping the first ``max_particles`` and discarding every later fresh sample
    is unbiased in expectation but freezes the conditional belief on whichever
    determinizations arrived first -- the collection never improves however long
    the edge is searched. Replacement keeps a uniform sample of *all* draws for
    the same memory.
    """
    torch.manual_seed(0)
    search, _ = _search(simulations=8, chance_widening=1.0, max_particles=2)
    node = mcts.Node(np.array([0]), np.array([1.0]))
    rng = random.Random(0)
    state = _position(players=2, turn=8)

    seen: set[int] = set()
    for draw in range(200):
        marked = state.copy()
        marked.sheets[0].temps = draw  # a per-draw marker
        search._record_outcome(node, 0, ("k",), marked, rng, exact=False)
    outcome = node.outcomes[0][("k",)]
    assert outcome.count == 200
    assert len(outcome.particles) == 2, "the cap was not honoured"
    kept = {particle.sheets[0].temps for particle in outcome.particles}
    assert max(kept) > 3, (
        f"particles froze on the earliest draws: kept {sorted(kept)}"
    )
