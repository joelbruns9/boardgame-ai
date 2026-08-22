"""The root-player contract, the leaf-value blend, and observation-keyed chance."""
from __future__ import annotations

import collections
import random

import numpy as np
import pytest
import torch

from games.welcome_to import macro_codec as mc
from games.welcome_to import mcts
from games.welcome_to import network as nw
from games.welcome_to import training
from games.welcome_to.bots import GreedyBot
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


def test_the_tree_barely_deepens_however_large_the_budget():
    """MEASURED, and a real limit on what this search is.

    Every turn boundary reveals three fresh cards, so under exact observation
    keying a crossing almost always produces an unseen key and therefore a new
    leaf.  **Mean leaf depth plateaus at about 2 and stops responding to the
    budget** — measured over 2p and 3p positions at turns 4, 12 and 20:

    | sims | mean leaf depth (range over 6 positions) | max depth |
    |---|---|---|
    | 64 | 1.23 – 1.77 | 2 |
    | 256 | 1.26 – 2.00 | 2 – 4 |
    | 1024 | 1.34 – 2.00 | 3 – 4 |

    A rare line does reach depth 4, which is why this asserts the *mean* — the
    first version of this test asserted max depth from a single position and was
    simply wrong.

    So this is **one-turn lookahead over a value network**, not deep search: the
    root averages over many freshly-evaluated leaves rather than resolving a
    tree.  Strength has to come from the network.

    That is probably a property of the game rather than a defect to fix.  One
    turn out, a fixed ``macro_write(slot, 0, box)`` means writing any number
    from 1 to 15 — the index is ``(slot, delta, box)`` and the *move* is
    ``(number, box)``, and those only coincide where the table is known.  Depth
    through a boundary is therefore worth little however it is keyed.  See
    ``SELF_PLAY_PLAN.md`` before building machinery to deepen this.
    """
    for turn in (4, 12):
        state = _position(players=2, turn=turn)
        search, _ = _search(simulations=1024)
        _, _, root = search.search(state, root=0, rng=random.Random(3))
        stats = _tree_stats(root)
        assert stats["mean_leaf_depth"] < 3.0, (
            f"turn {turn}: mean leaf depth {stats['mean_leaf_depth']:.2f} -- if the "
            "tree now deepens, progressive widening may have landed; re-measure"
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
