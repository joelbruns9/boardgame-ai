"""Auxiliary targets and the self-play diversity meter."""
from __future__ import annotations

import random

import pytest

from games.welcome_to import training as tr
from games.welcome_to.bots import GreedyBot, play_match
from games.welcome_to.game import GameConfig, GameState


def _finished(players: int = 2, seed: int = 3, mirrored: bool = False) -> GameState:
    bots = [
        GreedyBot(random.Random(seed if mirrored else seed * 100 + i))
        for i in range(players)
    ]
    return play_match(bots, seed=seed, config=GameConfig(players=players))


# ──────────────────────────────────────────────────────────────────────────
# Auxiliary targets
# ──────────────────────────────────────────────────────────────────────────
def test_outcomes_need_a_finished_game():
    with pytest.raises(ValueError):
        tr.final_outcomes(GameState.new(seed=1, config=GameConfig(players=2)))


def test_outcomes_agree_with_the_final_state():
    state = _finished()
    outcomes = tr.final_outcomes(state)
    assert [o.player for o in outcomes] == [0, 1]
    assert [o.score for o in outcomes] == state.scores()
    winners = set(state.winners())
    assert [o.won for o in outcomes] == [p in winners for p in range(2)]
    for outcome in outcomes:
        sheet = state.sheets[outcome.player]
        assert outcome.permits == sheet.permits
        assert outcome.houses == sum(
            1 for row in sheet.numbers for n in row if n is not None
        )
        assert outcome.capacity_left == sum(sheet.placement_capacity())


def test_ranks_are_a_permutation():
    state = _finished(players=4, seed=9)
    ranks = sorted(o.rank for o in tr.final_outcomes(state))
    assert ranks == [0, 1, 2, 3]
    best = min(tr.final_outcomes(state), key=lambda o: o.rank)
    assert best.rank == 0


def test_plan_turns_match_the_recorded_validations():
    state = _finished(players=3, seed=11)
    for outcome in tr.final_outcomes(state):
        for slot in range(3):
            assert outcome.plan_turns[slot] == state.plan_turns[slot].get(outcome.player)
        completed = [t for t in outcome.plan_turns if t is not None]
        assert outcome.plans_completed == len(completed)
        assert len(outcome.plan_order) == len(completed)


def test_turns_to_plan_is_relative_and_masked():
    state = _finished(players=3, seed=11)
    for outcome in tr.final_outcomes(state):
        targets = tr.sample_targets(outcome, turn=4)
        for slot in range(3):
            mask = targets[f"turns_to_plan_{slot}_mask"]
            value = targets[f"turns_to_plan_{slot}"]
            if outcome.plan_turns[slot] is None:
                assert mask == 0.0 and value == tr.NEVER
            else:
                assert mask == 1.0
                assert value == max(0, outcome.plan_turns[slot] - 4)


def test_component_targets_reconstruct_the_score():
    state = _finished()
    for outcome in tr.final_outcomes(state):
        t = tr.sample_targets(outcome, turn=1)
        total = (
            t["score_plans"]
            + t["score_parks"]
            + t["score_pools"]
            + t["score_temp"]
            + t["score_estates"]
            - t["score_bis"]
            - t["score_permits"]
            - t["score_roundabouts"]
        )
        assert total == t["score"], "the aux heads must add back up to the value head"


def test_target_names_cover_every_key():
    state = _finished()
    targets = tr.sample_targets(tr.final_outcomes(state)[0], turn=2)
    assert set(targets) == set(tr.TARGET_NAMES)


def test_turns_left_never_goes_negative():
    state = _finished()
    outcome = tr.final_outcomes(state)[0]
    assert tr.sample_targets(outcome, turn=outcome.final_turn + 5)["turns_left"] == 0.0


# ──────────────────────────────────────────────────────────────────────────
# Divergence
# ──────────────────────────────────────────────────────────────────────────
def test_mirrored_seats_produce_one_sheet_between_them():
    """The degenerate case: same policy, same tie-breaks, shared stacks."""
    state = _finished(mirrored=True)
    assert tr.sheet_divergence(state) == 0.0
    assert tr.first_divergence_turn(state) is None
    assert state.scores()[0] == state.scores()[1]


def test_independent_sampling_diverges_immediately():
    state = _finished(mirrored=False)
    assert tr.sheet_divergence(state) > 0.5
    # The substantive claim is that divergence starts in the *opening* and then
    # self-amplifies, not that it lands on turn 1 for every seed -- which turn
    # exactly depends on the sampling stream.  Pinning it to 1 made this test
    # seed-fragile; the magnitude assertion above is what guards the mechanism.
    first = tr.first_divergence_turn(state)
    assert first is not None and first <= 3, "divergence is self-amplifying"


def test_a_single_seat_has_no_divergence_question():
    state = GameState.new(seed=1, config=GameConfig(players=1))
    assert tr.sheet_divergence(state) == 0.0
    assert tr.first_divergence_turn(state) is None


def test_the_diversity_report_flags_collapse():
    mirrored = [_finished(seed=s, mirrored=True) for s in range(4)]
    report = tr.diversity_report(mirrored)
    assert report["identical_games"] == 1.0
    assert report["mean_sheet_divergence"] == 0.0
    assert report["mean_score_spread"] == 0.0

    varied = [_finished(seed=s, mirrored=False) for s in range(4)]
    report = tr.diversity_report(varied)
    assert report["identical_games"] == 0.0
    assert report["mean_sheet_divergence"] > 0.5


def test_the_report_is_empty_for_no_games():
    assert tr.diversity_report([]) == {}


# ──────────────────────────────────────────────────────────────────────────
# One-seat modes.  Engine coverage only -- BGA offers them, so the engine has to
# be right about them, but neither is a training configuration: with one seat
# TEMP_SOLO_SCORE replaces the 7/4/1 ranking and every City Plan pays its
# first-place value.  See SELF_PLAY_PLAN.md.
# ──────────────────────────────────────────────────────────────────────────
def test_one_seat_standard_rules_is_not_real_solo():
    config = GameConfig(players=1, solo_rules=False)
    assert config.single_player and not config.solo
    assert config.standard, "three shared stacks, not the six solo pairs"
    assert config.choice_slots == 3

    state = GameState.new(seed=2, config=config)
    assert len([c for c in state.table_cards(0) if c is not None]) == 6
    assert all(e is not None for e in state.next_effects(0))
    # no solo marker card was shuffled in
    assert state.deck_remaining == 81 - 6


def test_real_solo_mode_is_unchanged():
    config = GameConfig(players=1)
    assert config.solo and not config.standard
    assert config.choice_slots == 6
    assert GameState.new(seed=2, config=config).deck_remaining == 82 - 3


def test_one_seat_games_terminate():
    state = GameState.new(seed=5, config=GameConfig(players=1, solo_rules=False))
    rng = random.Random(0)
    steps = 0
    while not state.is_terminal:
        state.apply(rng.choice(state.legal_actions()))
        steps += 1
        assert steps < 5000
    assert state.end_of_game_reason() is not None
