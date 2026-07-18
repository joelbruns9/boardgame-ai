"""Phase C gates (plan §5): chance-signature exactness over every legal
action, budget discipline, closed-mode expectimax equivalence (terminal AND
net-leaves variants, against an INDEPENDENT reference), open-mode agreement,
buffer round-trip of Gumbel targets, self-play smoke."""

import math
import random
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
from games.seven_wonders_duel.search import (
    GumbelMCTS,
    SearchConfig,
    age_deal_key,
    chance_signature,
    closed_root_exact_value,
    enumerate_chains,
    expand_exhaustive,
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

    game = _play_random(
        11, until=lambda g: g.phase is Phase.CHOOSE_NEXT_START_PLAYER and g.age == 2
    )
    assert game.phase is Phase.CHOOSE_NEXT_START_PLAYER
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

    # Age boundary (next-age chooser).
    boundary = _play_random(
        11, until=lambda g: g.phase is Phase.CHOOSE_NEXT_START_PLAYER
    )
    mcts = GumbelMCTS(evaluator, SearchConfig(sims=24, top_k=4, mode="closed", seed=1))
    result = mcts.search(boundary)
    assert result.action_index in set(legal_action_indices(boundary))
    deal_edges = [
        edge
        for edge in mcts._closed_root.edges
        if any(spec.kind is ChanceKind.AGE_DEAL for spec in edge.specs)
    ]
    assert max(len(edge.children) for edge in deal_edges) >= 2


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
