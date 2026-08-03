"""Phase C gates (plan §5): chance-signature exactness over every legal
action, budget discipline, closed-mode expectimax equivalence (terminal AND
net-leaves variants, against an INDEPENDENT reference), open-mode agreement,
buffer round-trip of Gumbel targets, self-play smoke."""

import math
import random
from collections import Counter
from itertools import combinations

import pytest

torch = pytest.importorskip("torch")

from games.seven_wonders_duel.buffer import GameRecorder, from_json_line, to_json_line
from games.seven_wonders_duel.codec import decode_action, legal_action_indices
from games.seven_wonders_duel.data import (
    ALL_BUILDING_CARDS,
    BackType,
    PROGRESS_IDS,
    PROGRESS_TOKENS,
    TABLEAU_LAYOUTS,
    back_type_of,
)
from games.seven_wonders_duel.dataset import examples_from_record
from games.seven_wonders_duel.engine import Action, ActionUse, apply_action
from games.seven_wonders_duel.game import ChanceKind, Phase, new_game
from games.seven_wonders_duel.inference import Evaluator
from games.seven_wonders_duel.net import SWDNet
from games.seven_wonders_duel.pool import enumerate_card_reveal, unseen_pool
from games.seven_wonders_duel.search import (
    _Child,
    ChanceSpec,
    GumbelMCTS,
    SearchConfig,
    age_deal_key,
    balanced_double_reveal_chains,
    chance_signature,
    distinct_offsets,
    double_reveal_offset_seed,
    closed_root_exact_value,
    enumerate_chains,
    expand_exhaustive,
    fixed_support_index,
    state_actor,
)


@pytest.fixture(scope="module")
def evaluator():
    torch.manual_seed(7)
    return Evaluator(SWDNet(32, 1, 2))


def _play_random(seed, until=None, rng_seed=None):
    game = new_game(seed, first_player=seed % 2)
    rng = random.Random(rng_seed if rng_seed is not None else seed * 13 + 5)
    while game.phase is not Phase.COMPLETE:
        if until is not None and until(game):
            return game
        indices = legal_action_indices(game)
        apply_action(game, decode_action(game, rng.choice(indices)))
    return game


def _present_count(game):
    return sum(1 for card in game.tableau.cards.values() if card.present)


def _near_terminal_position(max_cards=3, need_hidden=True, skip=0):
    found = 0
    for seed in range(60):
        game = _play_random(
            seed,
            until=lambda g: (
                g.age == 3
                and g.pending_choice is None
                and g.phase is Phase.PLAY_AGE
                and _present_count(g) <= max_cards
            ),
        )
        if game.phase is Phase.COMPLETE:
            continue
        hidden = any(
            c.present and not c.revealed for c in game.tableau.cards.values()
        )
        if hidden or not need_hidden:
            if found == skip:
                return game
            found += 1
    raise AssertionError("no suitable near-terminal position found")


# --------------------------------------------------------------------------
# Independent chance reference: event kinds are probed from the engine
# (simulator clone), outcome spaces and probabilities are recomputed from raw
# data tables + the observation — no chance_signature, no enumerate_chains,
# no pool.py.
# --------------------------------------------------------------------------


def _independent_chains(state, action_index):
    probe = state.clone()
    probe.search_barrier = False  # test-side reference may probe the simulator
    events = apply_action(probe, decode_action(probe, action_index)).events

    observation = state.observation(0)
    visible = set(observation.discard_pile) | set(observation.buried_cards)
    for city in observation.cities:
        visible.update(city.buildings)
    for card in observation.tableau:
        if card.card_name is not None:
            visible.add(card.card_name)
    owned_progress = set(observation.available_progress_tokens)
    for city in observation.cities:
        owned_progress.update(city.progress_tokens)
    offboard = sorted(
        (t.name for t in PROGRESS_TOKENS if t.name not in owned_progress),
        key=PROGRESS_IDS.__getitem__,
    )

    def pool_for(back):
        return sorted(
            card.name
            for card in ALL_BUILDING_CARDS
            if back_type_of(card.name) is back and card.name not in visible
        )

    def recurse(index, used):
        if index == len(events):
            return [([], 1.0)]
        event = events[index]
        results = []
        if event.kind is ChanceKind.CARD_REVEAL:
            names = [n for n in pool_for(event.context[1]) if n not in used]
            for name in names:
                for tail, p in recurse(index + 1, used | {name}):
                    results.append(([name, *tail], p / len(names)))
        elif event.kind is ChanceKind.GREAT_LIBRARY_DRAW:
            subsets = list(combinations(offboard, 3))
            for subset in subsets:
                for tail, p in recurse(index + 1, used):
                    results.append(([tuple(subset), *tail], p / len(subsets)))
        else:
            raise AssertionError(f"gate position fired {event.kind}")
        return results

    return recurse(0, frozenset())


def _expectimax(game, evaluator, depth=None):
    """Independent reference: minimax over decisions, exact expectation over
    independently enumerated chance; terminal values, or net leaves when a
    depth cap is given. Player-0 perspective."""

    if game.phase is Phase.COMPLETE:
        if game.winner is None:
            return 0.0
        return 1.0 if game.winner == 0 else -1.0
    if depth is not None and depth <= 0:
        return GumbelMCTS(evaluator)._evaluate(game)[0]
    actor = state_actor(game)
    sign = 1.0 if actor == 0 else -1.0
    next_depth = None if depth is None else depth - 1
    best = -math.inf
    for index in legal_action_indices(game):
        expected = 0.0
        for outcomes, probability in _independent_chains(game, index):
            clone = game.clone()
            clone.search_barrier = True
            apply_action(
                clone, decode_action(clone, index), chance_outcomes=outcomes or None
            )
            expected += probability * _expectimax(clone, evaluator, next_depth)
        best = max(best, sign * expected)
    return sign * best


# --- chance signature: exact vs engine events, EVERY legal action -----------


def test_chance_signature_matches_engine_events_for_every_legal_action():
    checked = 0
    for seed in (2, 5):
        game = new_game(seed, first_player=seed % 2)
        rng = random.Random(seed)
        while game.phase is not Phase.COMPLETE:
            for index in legal_action_indices(game):
                clone = game.clone()
                action = decode_action(clone, index)
                specs = chance_signature(clone, action)
                result = apply_action(clone, action)
                assert len(result.events) == len(specs), (specs, result.events)
                for spec, event in zip(specs, result.events):
                    assert spec.kind is event.kind
                    if spec.kind in (ChanceKind.CARD_REVEAL, ChanceKind.AGE_DEAL):
                        assert spec.context == event.context
                checked += 1
            apply_action(
                game, decode_action(game, rng.choice(legal_action_indices(game)))
            )
    assert checked > 500  # every legal action at every state of two full games


def test_chance_signature_covers_great_library():
    game = _play_random(400, until=lambda g: g.phase is Phase.PLAY_AGE)
    for city in game.cities:
        if "The Great Library" in city.wonders:
            city.wonders.remove("The Great Library")
    game.cities[game.active_player].wonders[0:0] = ["The Great Library"]
    game.cities[game.active_player].coins = 100
    slot = game.tableau.accessible_slot_ids()[0]
    action = Action(slot, ActionUse.CONSTRUCT_WONDER, "The Great Library")
    specs = chance_signature(game, action)
    assert specs[-1].kind is ChanceKind.GREAT_LIBRARY_DRAW


# --- budget discipline ------------------------------------------------------


@pytest.mark.parametrize("sims", [5, 12, 20, 64])
def test_sequential_halving_never_exceeds_budget(evaluator, sims):
    game = _play_random(3, until=lambda g: g.phase is Phase.PLAY_AGE)
    config = SearchConfig(sims=sims, top_k=16, mode="closed", seed=1)
    result = GumbelMCTS(evaluator, config).search(game.clone())
    assert 1 <= result.sims <= sims
    assert result.action_index in set(legal_action_indices(game))


def test_invalid_config_rejected(evaluator):
    game = _play_random(3, until=lambda g: g.phase is Phase.PLAY_AGE)
    with pytest.raises(ValueError):
        GumbelMCTS(evaluator, SearchConfig(sims=0)).search(game)


# --- determinism ------------------------------------------------------------


@pytest.mark.parametrize("mode", ["closed", "open"])
def test_search_is_deterministic_given_seed(evaluator, mode):
    game = _play_random(3, until=lambda g: g.phase is Phase.PLAY_AGE)
    config = SearchConfig(sims=16, top_k=4, mode=mode, seed=11)
    first = GumbelMCTS(evaluator, config).search(game.clone())
    second = GumbelMCTS(
        evaluator, SearchConfig(sims=16, top_k=4, mode=mode, seed=11)
    ).search(game.clone())
    assert first.action_index == second.action_index
    assert first.visits == second.visits
    assert first.gumbel_topk == second.gumbel_topk
    assert first.root_value == pytest.approx(second.root_value)
    assert first.policy_target == pytest.approx(second.policy_target)


# --- closed mode == independent expectimax (terminal + net leaves) ----------


def test_closed_exact_value_matches_expectimax_to_1e6(evaluator):
    game = _near_terminal_position()
    mcts = GumbelMCTS(evaluator, SearchConfig(mode="closed", seed=5))
    root_state = game.clone()
    root_state.search_barrier = True
    root = mcts._make_closed_node(root_state)
    expand_exhaustive(mcts, root)
    exact = closed_root_exact_value(root)
    reference = _expectimax(game, evaluator)
    assert exact == pytest.approx(reference, abs=1e-6)


def test_closed_exact_value_with_net_leaves_matches_depth_limited_expectimax(
    evaluator,
):
    game = _near_terminal_position(max_cards=4)
    mcts = GumbelMCTS(evaluator, SearchConfig(mode="closed", seed=5))
    root_state = game.clone()
    root_state.search_barrier = True
    root = mcts._make_closed_node(root_state)
    expand_exhaustive(mcts, root, depth=2)
    exact = closed_root_exact_value(root)
    reference = _expectimax(game, evaluator, depth=2)
    assert exact == pytest.approx(reference, abs=1e-6)


def test_exact_value_rejects_partial_chance_mass(evaluator):
    game = _near_terminal_position()
    mcts = GumbelMCTS(evaluator, SearchConfig(mode="closed", seed=5))
    root_state = game.clone()
    root_state.search_barrier = True
    root = mcts._make_closed_node(root_state)
    expand_exhaustive(mcts, root)
    # Remove one outcome from the first chance edge -> mass < 1 -> must raise.
    for edge in root.edges:
        if len(edge.children) > 1 and all(
            child.probability is not None for child in edge.children.values()
        ):
            edge.children.pop(next(iter(edge.children)))
            break
    else:
        pytest.skip("position has no multi-outcome enumerable edge")
    with pytest.raises(ValueError, match="mass"):
        closed_root_exact_value(root)


def test_closed_search_converges_to_exact_on_small_position(evaluator):
    game = _near_terminal_position()
    reference = _expectimax(game, evaluator)
    config = SearchConfig(sims=400, top_k=8, mode="closed", seed=2)
    result = GumbelMCTS(evaluator, config).search(game)
    assert result.root_value == pytest.approx(reference, abs=0.2)
    actor = state_actor(game)
    sign = 1.0 if actor == 0 else -1.0
    action_values = {}
    for index in legal_action_indices(game):
        expected = 0.0
        for outcomes, probability in _independent_chains(game, index):
            clone = game.clone()
            clone.search_barrier = True
            apply_action(
                clone, decode_action(clone, index), chance_outcomes=outcomes or None
            )
            expected += probability * _expectimax(clone, evaluator)
        action_values[index] = sign * expected
    best_value = max(action_values.values())
    assert action_values[result.action_index] == pytest.approx(best_value, abs=1e-9)


def test_open_mode_agrees_on_small_positions(evaluator):
    """Convergence gate: at high sims the chosen action must be (near-)exact-
    optimal and its edge Q must match that action's independent exact value.
    (root_value is a mean over ALL descents including forced exploration of
    losing candidates, so it is not the convergence quantity.)"""

    for skip in range(3):
        game = _near_terminal_position(skip=skip)
        actor = state_actor(game)
        sign = 1.0 if actor == 0 else -1.0
        action_values = {}
        for index in legal_action_indices(game):
            expected = 0.0
            for outcomes, probability in _independent_chains(game, index):
                clone = game.clone()
                clone.search_barrier = True
                apply_action(
                    clone,
                    decode_action(clone, index),
                    chance_outcomes=outcomes or None,
                )
                expected += probability * _expectimax(clone, evaluator)
            action_values[index] = sign * expected
        best_value = max(action_values.values())

        config = SearchConfig(sims=1200, top_k=8, mode="open", seed=3 + skip)
        mcts = GumbelMCTS(evaluator, config)
        result = mcts.search(game)
        chosen = result.action_index
        assert action_values[chosen] >= best_value - 0.1
        root = mcts._open_root
        chosen_q = sign * (
            root.edge_value_p0[chosen] / root.edge_visits[chosen]
        )
        assert chosen_q == pytest.approx(action_values[chosen], abs=0.15)


# --- chance-layer structure -------------------------------------------------


def _double_uncover_state(seed=30):
    game = _play_random(seed, until=lambda g: g.phase is Phase.PLAY_AGE)
    apply_action(game, Action((4, 1), ActionUse.DISCARD_FOR_COINS))
    apply_action(game, Action((4, 5), ActionUse.DISCARD_FOR_COINS))
    return game


def test_sequential_reveal_chain_probabilities_sum_to_one():
    game = _double_uncover_state()
    action = Action((4, 3), ActionUse.DISCARD_FOR_COINS)
    specs = chance_signature(game, action)
    assert [s.kind for s in specs] == [ChanceKind.CARD_REVEAL] * 2
    chains = enumerate_chains(game, specs)
    assert len(chains) == 11 * 10
    total = sum(p for _, p, _ in chains)
    assert total == pytest.approx(1.0, abs=1e-12)
    assert all(p == pytest.approx(1 / 110) for _, p, _ in chains)


def test_age_deal_key_coalesces_equivalent_hidden_arrangements():
    layout = TABLEAU_LAYOUTS[1]
    names = [card.name for card in ALL_BUILDING_CARDS if card.age == 1][:20]
    face_down = [i for i, slot in enumerate(layout) if not slot.face_up]
    face_up = [i for i, slot in enumerate(layout) if slot.face_up]
    swapped = list(names)
    swapped[face_down[0]], swapped[face_down[1]] = (
        swapped[face_down[1]],
        swapped[face_down[0]],
    )
    assert age_deal_key(1, names) == age_deal_key(1, swapped)
    different = list(names)
    different[face_up[0]], different[face_down[0]] = (
        different[face_down[0]],
        different[face_up[0]],
    )
    assert age_deal_key(1, names) != age_deal_key(1, different)


def test_age_three_deal_samples_have_exactly_three_guilds():
    from games.seven_wonders_duel.search import sample_outcomes, ChanceSpec

    # The last take of Age II is where the Age III deal is now sampled: at the
    # CHOOSE_NEXT_START_PLAYER that follows, it has already happened.
    game = _play_random(
        11,
        until=lambda g: g.phase is Phase.PLAY_AGE
        and g.age == 2
        and _present_count(g) == 1,
    )
    assert game.age == 2
    specs = (ChanceSpec(ChanceKind.AGE_DEAL, (3,)),)
    rng = random.Random(0)
    for _ in range(10):
        outcomes, probability, key = sample_outcomes(game, specs, rng)
        deal = outcomes[0]
        assert probability is None
        assert len(deal) == 20 and len(set(deal)) == 20
        guilds = [n for n in deal if back_type_of(n) is BackType.GUILD]
        assert len(guilds) == 3


def test_closed_search_samples_hidden_boundaries_instead_of_reading_them(evaluator):
    # Final draft pick (initial Age I deal).
    game = new_game(9)
    for _ in range(7):
        apply_action(game, decode_action(game, legal_action_indices(game)[0]))
    mcts = GumbelMCTS(evaluator, SearchConfig(sims=24, top_k=4, mode="closed", seed=0))
    result = mcts.search(game)
    assert result.action_index in set(legal_action_indices(game))
    deal_edges = [
        edge
        for edge in mcts._closed_root.edges
        if any(spec.kind is ChanceKind.AGE_DEAL for spec in edge.specs)
    ]
    assert deal_edges
    assert max(len(edge.children) for edge in deal_edges) >= 2  # sampled worlds

    # Age boundary: the deal fires on the take that empties the pyramid, so
    # the chooser is asked with the new Age already on the table.
    last_take = _play_random(
        11,
        until=lambda g: g.phase is Phase.PLAY_AGE
        and g.age < 3
        and _present_count(g) == 1,
    )
    mcts = GumbelMCTS(evaluator, SearchConfig(sims=24, top_k=4, mode="closed", seed=1))
    result = mcts.search(last_take)
    assert result.action_index in set(legal_action_indices(last_take))
    deal_edges = [
        edge
        for edge in mcts._closed_root.edges
        if any(spec.kind is ChanceKind.AGE_DEAL for spec in edge.specs)
    ]
    assert deal_edges
    assert max(len(edge.children) for edge in deal_edges) >= 2

    # ... and the chooser itself now fires no chance at all.
    boundary = _play_random(
        11, until=lambda g: g.phase is Phase.CHOOSE_NEXT_START_PLAYER
    )
    mcts = GumbelMCTS(evaluator, SearchConfig(sims=24, top_k=4, mode="closed", seed=1))
    mcts.search(boundary)
    assert all(not edge.specs for edge in mcts._closed_root.edges)


def test_force_expand_root_chance_materializes_all_enumerable_children(evaluator):
    game = _double_uncover_state()
    config = SearchConfig(
        sims=8, top_k=4, mode="closed", seed=0, force_expand_root_chance=True
    )
    mcts = GumbelMCTS(evaluator, config)
    mcts.search(game)
    for edge in mcts._closed_root.edges:
        if edge.specs and not any(
            spec.kind is ChanceKind.AGE_DEAL for spec in edge.specs
        ):
            expected = len(enumerate_chains(game, edge.specs))
            assert len(edge.children) == expected
            for child in edge.children.values():
                assert child.node.visits >= 1  # evaluated at expansion


# --- approximate fixed-support chance edges (chance-enumeration Step 1) -----


def _truncated_fixed_support_root(evaluator, keep=8, weights=None, seed=0):
    """A forced root whose widest chance edge is truncated to `keep` outcomes,
    re-normalised and CLOSED — the third edge class, without yet having a
    production rule that builds one (that is Step 2)."""

    game = _double_uncover_state()
    mcts = GumbelMCTS(
        evaluator,
        SearchConfig(
            sims=8, top_k=4, mode="closed", seed=seed, force_expand_root_chance=True
        ),
    )
    mcts.search(game)
    root = mcts._closed_root
    edge = max(root.edges, key=lambda e: len(e.children))
    assert len(edge.children) > keep  # a real double reveal: 110 outcomes
    weights = weights if weights is not None else [1.0 / keep] * keep
    assert len(weights) == keep
    edge.children = dict(list(edge.children.items())[:keep])
    for child, weight in zip(edge.children.values(), weights, strict=True):
        child.samples = 0
        child.probability = weight
    edge.close_fixed_support()
    return mcts, root, edge


def test_fixed_support_index_matches_the_rust_golden_table():
    # Pinned identically in tree.rs::fixed_support_tests::golden_table_matches_python.
    uniform = [0.25, 0.25, 0.25, 0.25]
    skewed = [0.1, 0.7, 0.2]
    cases = [
        (uniform, 0.0, 0),
        (uniform, 0.249999, 0),
        (uniform, 0.25, 1),
        (uniform, 0.5, 2),
        (uniform, 0.999999, 3),
        (skewed, 0.09, 0),
        (skewed, 0.1, 1),
        (skewed, 0.799999, 1),
        (skewed, 0.8, 2),
        ([1.0], 0.0, 0),
        ([1.0], 0.999999, 0),
    ]
    for weights, target, expected in cases:
        assert fixed_support_index(weights, target) == expected, (weights, target)
    # A draw past the final cumulative sum (float residue) takes the last child.
    assert fixed_support_index([0.5, 0.5], 1.0) == 1
    with pytest.raises(ValueError, match="no children"):
        fixed_support_index([], 0.0)


def test_fixed_support_edge_never_grows_and_keeps_unit_mass(evaluator):
    mcts, root, edge = _truncated_fixed_support_root(evaluator)
    retained = tuple(edge.children)
    nodes = {id(child.node) for child in edge.children.values()}

    for _ in range(5000):
        node = mcts._closed_child(root, edge)
        assert id(node) in nodes  # only retained children are ever reached

    assert tuple(edge.children) == retained  # nothing materialized
    assert sum(child.probability for child in edge.children.values()) == 1.0
    assert sum(child.samples for child in edge.children.values()) == 5000
    assert all(child.samples > 0 for child in edge.children.values())
    edge.q_p0  # the mass invariant still holds after thousands of draws

    # The same through the integrated descent path, which also backs values up.
    for _ in range(200):
        mcts._descend_closed(root, forced_edge=edge)
    assert tuple(edge.children) == retained
    assert sum(child.probability for child in edge.children.values()) == 1.0


def test_fixed_support_draws_follow_the_renormalised_weights(evaluator):
    keep = 4
    weights = [0.7, 0.1, 0.1, 0.1]
    mcts, root, edge = _truncated_fixed_support_root(
        evaluator, keep=keep, weights=weights, seed=3
    )
    draws = 6000
    for _ in range(draws):
        mcts._closed_child(root, edge)
    empirical = [child.samples / draws for child in edge.children.values()]
    for observed, weight in zip(empirical, weights, strict=True):
        assert observed == pytest.approx(weight, abs=0.02)


def test_fixed_support_q_is_the_weighted_sum_over_retained_children(evaluator):
    keep = 8
    mcts, root, edge = _truncated_fixed_support_root(evaluator, keep=keep)
    values = [child.node.value_p0 for child in edge.children.values()]
    assert edge.q_p0 == pytest.approx(sum(values) / keep, abs=1e-12)

    skewed = [0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01]
    _, _, edge = _truncated_fixed_support_root(evaluator, keep=keep, weights=skewed)
    values = [child.node.value_p0 for child in edge.children.values()]
    hand = sum(w * v for w, v in zip(skewed, values, strict=True))
    assert edge.q_p0 == pytest.approx(hand, abs=1e-12)


def test_close_fixed_support_rejects_a_support_that_is_not_renormalised(evaluator):
    game = _double_uncover_state()
    mcts = GumbelMCTS(
        evaluator,
        SearchConfig(
            sims=8, top_k=4, mode="closed", seed=0, force_expand_root_chance=True
        ),
    )
    mcts.search(game)
    edge = max(mcts._closed_root.edges, key=lambda e: len(e.children))
    # Truncated but NOT re-normalised: the retained children hold mass < 1.
    edge.children = dict(list(edge.children.items())[:8])
    with pytest.raises(ValueError, match="mass"):
        edge.close_fixed_support()
    assert not edge.fixed_support  # the edge is never closed over partial mass


def test_exhaustive_expansion_refuses_to_complete_a_closed_support(evaluator):
    """`expand_exhaustive` appends every omitted outcome, which on a closed edge
    would push its re-normalised mass past 1 and corrupt the tree in place --
    several steps before `closed_root_exact_value` could refuse it."""

    mcts, root, edge = _truncated_fixed_support_root(evaluator)
    retained = tuple(edge.children)
    with pytest.raises(ValueError, match="fixed-support"):
        expand_exhaustive(mcts, root, depth=1)
    assert tuple(edge.children) == retained  # refused BEFORE mutating
    assert sum(child.probability for child in edge.children.values()) == 1.0


def test_forced_expansion_refuses_to_re_enter_a_closed_edge(evaluator):
    """A second forced expansion under a different seed would append members of
    a SECOND support and only notice at the closing mass check, by which point
    the tree is already mutated."""

    game = _double_uncover_state()
    config = SearchConfig(
        sims=8,
        top_k=4,
        mode="closed",
        seed=0,
        force_expand_root_chance=True,
        double_reveal_offsets=2,
    )
    mcts = GumbelMCTS(evaluator, config)
    root = mcts.make_root(game)
    capped = [edge for edge in root.edges if edge.fixed_support]
    assert capped
    sizes = [len(edge.children) for edge in root.edges]

    mcts.config.seed = 999  # a different support for the same signature
    with pytest.raises(RuntimeError, match="already-closed edge"):
        mcts._force_expand_root(root)
    assert [len(edge.children) for edge in root.edges] == sizes
    for edge in capped:
        assert sum(c.probability for c in edge.children.values()) == pytest.approx(
            1.0, abs=1e-9
        )


def test_forced_expansion_validates_mass_before_materializing(evaluator):
    """Transactional: a support whose mass is wrong must be rejected before any
    child is built, so an evaluator failure part-way cannot leave an edge whose
    mass is neither 1 nor recoverable."""

    game = _double_uncover_state()
    mcts = GumbelMCTS(
        evaluator,
        SearchConfig(sims=8, top_k=4, mode="closed", seed=0, force_expand_root_chance=True),
    )
    root_state = game.clone()
    root_state.search_barrier = True
    root = mcts._make_closed_node(root_state)
    mcts._expand_closed(root)
    target = next(
        edge
        for edge in root.edges
        if edge.specs and not any(s.kind is ChanceKind.AGE_DEAL for s in edge.specs)
    )
    # A child carrying mass that the enumeration will also offer: the union then
    # exceeds 1 and expansion must refuse without evaluating anything.
    outcomes, probability, key = enumerate_chains(game, target.specs)[0]
    clone = root.state.clone()
    clone.search_barrier = True
    apply_action(clone, decode_action(clone, target.action_index), chance_outcomes=outcomes)
    target.children[key] = _Child(probability=1.0, node=mcts._make_closed_node(clone))
    with pytest.raises(RuntimeError, match="would hold probability mass"):
        mcts._force_expand_root(root)
    assert len(target.children) == 1  # nothing else materialized


def test_exact_value_refuses_an_approximate_fixed_support_edge(evaluator):
    mcts, root, edge = _truncated_fixed_support_root(evaluator)
    root.edges = [edge]  # isolate the approximate edge from unexpanded siblings
    with pytest.raises(ValueError, match="fixed-support"):
        closed_root_exact_value(root)


# --- balanced double-reveal support (chance-enumeration Step 2) -------------


def _double_reveal_specs(game=None):
    game = game if game is not None else _double_uncover_state()
    action = Action((4, 3), ActionUse.DISCARD_FOR_COINS)
    specs = chance_signature(game, action)
    assert [s.kind for s in specs] == [ChanceKind.CARD_REVEAL] * 2
    assert specs[0].context[1] == specs[1].context[1]  # same back
    return game, specs


def _mixed_back_double_reveal():
    """A double reveal whose two slots carry DIFFERENT backs (Age III mixes
    guild backs in), where the pools are disjoint and the cycle runs over the
    second pool instead."""

    for seed in range(40):
        game = new_game(seed, first_player=seed % 2)
        rng = random.Random(seed * 7 + 1)
        while game.phase is not Phase.COMPLETE:
            for index in legal_action_indices(game):
                specs = chance_signature(game, decode_action(game, index))
                if (
                    len(specs) == 2
                    and all(s.kind is ChanceKind.CARD_REVEAL for s in specs)
                    and specs[0].context[1] != specs[1].context[1]
                ):
                    return game, specs
            apply_action(game, decode_action(game, rng.choice(legal_action_indices(game))))
    pytest.skip("no mixed-back double reveal found")


@pytest.mark.parametrize("offsets", [1, 2, 3])
def test_balanced_double_reveal_support_is_marginally_balanced(offsets):
    game, specs = _double_reveal_specs()
    full = enumerate_chains(game, specs)
    n = int(round(len(full) ** 0.5)) + 1  # |full| = n * (n - 1)
    assert len(full) == n * (n - 1) == 110

    chains = balanced_double_reveal_chains(game, specs, offsets, search_seed=17)
    assert len(chains) == n * offsets
    keys = [key for _, _, key in chains]
    assert len(set(keys)) == len(keys)  # every retained pair distinct
    assert set(keys) <= {key for _, _, key in full}  # and a real outcome
    assert sum(probability for _, probability, _ in chains) == pytest.approx(1.0, abs=1e-12)
    assert {probability for _, probability, _ in chains} == {1.0 / (n * offsets)}

    firsts = Counter(outcomes[0] for outcomes, _, _ in chains)
    seconds = Counter(outcomes[1] for outcomes, _, _ in chains)
    # Marginal coverage: every hidden card leads exactly one stratum (and so
    # appears `offsets` times as the first reveal) and lands exactly `offsets`
    # times in the second slot. Never paired with itself.
    assert len(firsts) == n and set(firsts.values()) == {offsets}
    assert len(seconds) == n and set(seconds.values()) == {offsets}
    assert all(outcomes[0] != outcomes[1] for outcomes, _, _ in chains)


def test_different_back_double_reveals_stay_exhaustive():
    """Two backs means two disjoint pools and a full n1 x n2 grid.

    A cyclic support over the second pool would be unbiased, but only its FIRST
    margin could be exact -- a subset balanced in both margins needs a size
    divisible by lcm(n1, n2), and these pool sizes are usually coprime, i.e. the
    grid itself. It would also make the retained count depend on which slot is
    listed first (board position). Measured: 2.9% of the cap's saving at 3-4x
    the Q error, so these edges keep exact chance."""

    game, specs = _mixed_back_double_reveal()
    full = enumerate_chains(game, specs)
    firsts = {outcomes[0] for outcomes, _, _ in full}
    seconds = {outcomes[1] for outcomes, _, _ in full}
    assert not (firsts & seconds)  # disjoint pools: no exclusion, a full grid
    assert len(full) == len(firsts) * len(seconds)
    for offsets in (1, 2, 3):
        assert balanced_double_reveal_chains(game, specs, offsets, 5) is None


def test_balanced_double_reveal_falls_back_when_it_cannot_shrink():
    game, specs = _double_reveal_specs()
    assert balanced_double_reveal_chains(game, specs, 0, 1) is None  # disabled
    # 11 unseen cards -> 10 directed distances; X = 10 IS the full space.
    assert balanced_double_reveal_chains(game, specs, 10, 1) is None
    assert balanced_double_reveal_chains(game, specs, 99, 1) is None
    # Not a pure double reveal: single reveal, and a reveal + Great Library.
    assert balanced_double_reveal_chains(game, specs[:1], 2, 1) is None
    library = ChanceSpec(ChanceKind.GREAT_LIBRARY_DRAW)
    assert balanced_double_reveal_chains(game, (specs[0], library), 2, 1) is None


def test_offset_seed_separates_search_seed_position_and_signature():
    game, specs = _double_reveal_specs()
    pool = [
        name
        for name, _ in enumerate_card_reveal(
            unseen_pool(game.observation(game.active_player)), specs[0].context[1]
        )
    ]
    base = double_reveal_offset_seed(11, specs, (pool, pool))
    assert base == double_reveal_offset_seed(11, specs, (pool, pool))  # reproducible
    assert base != double_reveal_offset_seed(12, specs, (pool, pool))  # search seed
    assert base != double_reveal_offset_seed(11, specs, (pool[:-1], pool))  # position
    assert base != double_reveal_offset_seed(
        11, (specs[1], specs[0]), (pool, pool)
    )  # chance signature (which slots are being revealed, in order)


def test_capped_edges_sharing_a_chance_signature_share_their_support(evaluator):
    """Common random numbers. Taking one card as a build, a discard, or a wonder
    fires the SAME reveals, so those edges must draw the same offsets — then
    comparing the three actions is a comparison over one common support and the
    offset noise cancels out of the comparison."""

    game = _double_uncover_state()
    mcts = GumbelMCTS(
        evaluator,
        SearchConfig(
            sims=8,
            top_k=4,
            mode="closed",
            seed=0,
            force_expand_root_chance=True,
            double_reveal_offsets=2,
        ),
    )
    mcts.search(game)
    by_signature = {}
    for edge in mcts._closed_root.edges:
        if edge.fixed_support:
            by_signature.setdefault(edge.specs, []).append(set(edge.children))
    assert by_signature
    shared = [group for group in by_signature.values() if len(group) > 1]
    assert shared, "expected several actions on one revealed slot"
    for group in shared:
        assert all(support == group[0] for support in group)


def test_distinct_offsets_are_uniform_over_subsets():
    modulus, count, draws = 5, 2, 4000
    seen = Counter()
    for seed in range(draws):
        chosen = distinct_offsets(modulus, count, seed)
        assert len(set(chosen)) == count
        assert chosen == sorted(chosen)
        assert all(0 <= value < modulus for value in chosen)
        seen[tuple(chosen)] += 1
    expected = draws / math.comb(modulus, count)
    assert len(seen) == math.comb(modulus, count)  # every subset reachable
    assert max(abs(hits - expected) for hits in seen.values()) < 0.25 * expected


def test_balanced_double_reveal_estimator_is_unbiased():
    """Mean-unbiasedness of the stratified estimator, run through the real
    construction: averaging the support mean of a fixed leaf oracle over seeds
    converges to the exhaustive expectation. (Not sufficient on its own —
    Step 4 measures per-position error, not just the mean.)"""

    game, specs = _double_reveal_specs()

    def value(outcomes):  # a fixed, deterministic mock evaluator over the pair
        return math.sin(float(hash((outcomes[0], outcomes[1])) % 100003))

    full = enumerate_chains(game, specs)
    exact = sum(probability * value(o) for o, probability, _ in full)
    for offsets in (1, 2):
        estimates = []
        for seed in range(400):
            chains = balanced_double_reveal_chains(game, specs, offsets, seed)
            estimates.append(sum(p * value(o) for o, p, _ in chains))
        assert sum(estimates) / len(estimates) == pytest.approx(exact, abs=0.03)


def test_force_expansion_caps_only_pure_double_reveals_and_closes_them(evaluator):
    game = _double_uncover_state()
    offsets = 2
    config = SearchConfig(
        sims=8,
        top_k=4,
        mode="closed",
        seed=0,
        force_expand_root_chance=True,
        double_reveal_offsets=offsets,
    )
    mcts = GumbelMCTS(evaluator, config)
    mcts.search(game)
    capped = 0
    for edge in mcts._closed_root.edges:
        if not edge.specs or any(
            spec.kind is ChanceKind.AGE_DEAL for spec in edge.specs
        ):
            continue
        balanced = balanced_double_reveal_chains(game, edge.specs, offsets, 0)
        if balanced is None:
            assert not edge.fixed_support
            assert len(edge.children) == len(enumerate_chains(game, edge.specs))
            continue
        capped += 1
        assert edge.fixed_support and edge.probability_weighted
        assert set(edge.children) == {key for _, _, key in balanced}
        assert sum(c.probability for c in edge.children.values()) == pytest.approx(
            1.0, abs=1e-12
        )
        # Seeded with one evaluation at expansion; descents may add visits.
        assert all(child.node.visits >= 1 for child in edge.children.values())
        # Closed: thousands of descents stay inside the retained support.
        keys = tuple(edge.children)
        for _ in range(2000):
            mcts._closed_child(mcts._closed_root, edge)
        assert tuple(edge.children) == keys
    assert capped > 0


# --- Gumbel root contract + buffer round trip -------------------------------


@pytest.mark.parametrize("mode", ["closed", "open"])
def test_gumbel_policy_target_is_a_distribution(evaluator, mode):
    game = _play_random(9, until=lambda g: g.phase is Phase.PLAY_AGE)
    result = GumbelMCTS(
        evaluator, SearchConfig(sims=24, top_k=8, mode=mode, seed=1)
    ).search(game)
    legal = set(legal_action_indices(game))
    assert result.action_index in legal
    assert set(result.policy_target) == legal
    assert set(result.gumbel_topk) <= legal
    assert 1 <= len(result.gumbel_topk) <= 8
    assert sum(result.policy_target.values()) == pytest.approx(1.0)
    assert all(p >= 0 for p in result.policy_target.values())
    assert sum(result.visits.values()) > 0


def test_search_results_flow_through_buffer_into_dataset_targets(evaluator):
    recorder = GameRecorder(31, agents={"p0": "search", "p1": "search"})
    rng = random.Random(0)
    searched = {}
    move_index = 0
    while recorder.game.phase is not Phase.COMPLETE:
        if move_index < 3:
            result = GumbelMCTS(
                evaluator, SearchConfig(sims=8, top_k=4, seed=move_index)
            ).search(recorder.game)
            recorder.play(
                result.action_index,
                visits=result.visits,
                policy_target=result.policy_target,
                root_value=result.root_value,
                sims=result.sims,
                mode=result.mode,
                gumbel_topk=result.gumbel_topk,
            )
            searched[move_index] = result
        else:
            recorder.play(rng.choice(legal_action_indices(recorder.game)))
        move_index += 1
    record = from_json_line(to_json_line(recorder.finish()))
    examples = examples_from_record(record)
    for index, result in searched.items():
        assert record.moves[index].gumbel_topk == result.gumbel_topk
        example = examples[index]
        legal = [int(a) for a in example.legal]
        for action, probability in result.policy_target.items():
            assert example.policy_target[legal.index(action)] == pytest.approx(
                probability, abs=1e-6
            )


# --- self-play smoke --------------------------------------------------------


def test_self_play_smoke_both_modes(evaluator):
    game = new_game(21)
    move = 0
    while game.phase is not Phase.COMPLETE and move < 90:
        mode = "closed" if move % 2 == 0 else "open"
        config = SearchConfig(sims=8, top_k=4, mode=mode, seed=move)
        result = GumbelMCTS(evaluator, config).search(game)
        assert result.action_index in set(legal_action_indices(game))
        assert result.sims <= config.sims
        apply_action(game, decode_action(game, result.action_index))
        move += 1
    assert game.phase is Phase.COMPLETE
    assert game.search_barrier is False


# -- sigma normalisation ----------------------------------------------------


def test_sigma_rescales_completed_q_to_the_unit_interval(evaluator):
    """The Gumbel-AlphaZero factor applies to a [0, 1]-rescaled Q.

    Applied to a raw actor-relative q in [-1, 1] with c_scale=1.0, sigma spanned
    +/-50 against log-prior differences of ~1-3, so the network's move
    preferences were erased from the improved policy.
    """

    mcts = GumbelMCTS(evaluator, SearchConfig(c_visit=50.0, c_scale=0.1))
    sigma = mcts._sigma({0: -1.0, 1: 0.0, 2: 1.0}, max_visits=0)
    # Worst action pinned to 0, best to the full (c_visit + max_visits)*c_scale.
    assert sigma[0] == pytest.approx(0.0)
    assert sigma[2] == pytest.approx(5.0)
    assert sigma[1] == pytest.approx(2.5)
    # Scale grows with visits, exactly as the paper's factor does.
    assert mcts._sigma({0: 0.0, 1: 1.0}, max_visits=50)[1] == pytest.approx(10.0)


def test_sigma_is_translation_and_scale_invariant(evaluator):
    """Only the ORDERING and relative spacing of completed Q may matter.

    An absolute-valued sigma made the prior's influence depend on how good the
    position happened to be, so identical decisions scored differently in a
    winning position than in a losing one.
    """

    mcts = GumbelMCTS(evaluator, SearchConfig())
    base = mcts._sigma({0: -0.5, 1: 0.0, 2: 0.25}, max_visits=3)
    shifted = mcts._sigma({0: 0.0, 1: 0.5, 2: 0.75}, max_visits=3)
    stretched = mcts._sigma({0: -1.0, 1: 0.0, 2: 0.5}, max_visits=3)
    for action in base:
        assert base[action] == pytest.approx(shifted[action])
        assert base[action] == pytest.approx(stretched[action])


def test_sigma_collapses_when_no_search_information_exists(evaluator):
    """All-equal completed Q must leave the improved policy equal to the prior."""

    mcts = GumbelMCTS(evaluator, SearchConfig())
    sigma = mcts._sigma({0: 0.3, 1: 0.3, 2: 0.3}, max_visits=0)
    assert set(sigma.values()) == {0.0}


def test_prior_survives_into_the_improved_policy(evaluator):
    """Regression: the improved policy must still reflect the network's prior.

    With unnormalised sigma the completed-Q term swamped log-prior by ~25x, so
    `policy_target` was effectively independent of the network's move
    preferences and search discarded most of what the net had learned.
    """

    game = new_game(11, first_player=0)
    for _ in range(12):
        if game.phase is Phase.COMPLETE or game.pending_choice is not None:
            break
        apply_action(game, decode_action(game, legal_action_indices(game)[0]))
    mcts = GumbelMCTS(evaluator, SearchConfig(sims=8, top_k=8, mode="closed", seed=3))
    result = mcts.search(game)
    legal = legal_action_indices(game)
    priors = mcts._evaluate(game)[1]
    target = result.policy_target
    # Unvisited actions differ only through their prior, so among the actions
    # the search never touched, prior order must be preserved in the target.
    unvisited = [a for a in legal if result.visits.get(a, 0) == 0]
    if len(unvisited) >= 2:
        by_prior = sorted(unvisited, key=lambda a: priors[a])
        by_target = sorted(unvisited, key=lambda a: target[a])
        assert by_prior == by_target
