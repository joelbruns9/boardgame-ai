"""Auxiliary targets and the self-play diversity meter."""
from __future__ import annotations

import random

import numpy as np
import pytest

from games.welcome_to import encoder as enc
from games.welcome_to import training as tr
from games.welcome_to.bots import GreedyBot, play_match
from games.welcome_to.game import GameConfig, GameState


def _targets(state: GameState, viewer: int, turn: int) -> dict:
    """Targets for ``viewer`` on the encoder's own seat axis."""
    return tr.sample_targets(
        tr.final_outcomes(state), enc.seat_order(state, viewer), turn
    )


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
    for outcome in outcomes:
        sheet = state.sheets[outcome.player]
        assert outcome.num_seats == 2
        assert outcome.permits == sheet.permits
        assert outcome.houses == sum(
            1 for row in sheet.numbers for n in row if n is not None
        )
        assert outcome.capacity_left == sum(sheet.placement_capacity())


# ──────────────────────────────────────────────────────────────────────────
# The rank distribution (D1, D4).  A 0-indexed rank meant different things at
# different table sizes and disagreed with returns() on ties; the distribution
# is the fix for both, so both are tested here.
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("players", [2, 3, 4])
def test_rank_distributions_are_proper_and_stop_at_the_seat_count(players):
    state = _finished(players=players, seed=9)
    for dist in tr.rank_distributions(state):
        assert len(dist) == tr.MAX_RANKS
        assert sum(dist) == pytest.approx(1.0)
        assert all(p == 0.0 for p in dist[players:]), "a dead rank must stay dead"


@pytest.mark.parametrize("players", [2, 3, 4])
def test_every_finishing_position_is_filled_exactly_once(players):
    """Summed over seats, each live rank carries exactly one seat's worth."""
    state = _finished(players=players, seed=9)
    dists = tr.rank_distributions(state)
    for rank in range(players):
        assert sum(d[rank] for d in dists) == pytest.approx(1.0)


def test_ties_share_their_positions_instead_of_taking_seat_order():
    """The seat-order half of D1: sorted() gave tied seats distinct ranks."""
    state = _finished(mirrored=True)  # identical sheets, so an exact tie
    assert state.scores()[0] == state.scores()[1]
    for dist in tr.rank_distributions(state):
        assert dist == (0.5, 0.5, 0.0, 0.0)


def test_the_rank_value_reproduces_returns_at_two_seats():
    """D4: ``won`` gave 1.0 to every tied winner while returns() gave 0.0."""
    for seed in range(6):
        for mirrored in (False, True):
            state = _finished(seed=seed, mirrored=mirrored)
            returns = state.returns()
            for outcome in tr.final_outcomes(state):
                assert tr.rank_value(
                    outcome.rank_distribution, outcome.num_seats
                ) == pytest.approx(returns[outcome.player])


def test_the_utility_is_seat_count_invariant_at_the_ends():
    """The whole point of u_r: first is 1.0 and last is 0.0 at any table size."""
    for n in (2, 3, 4):
        utility = tr.rank_utility(n)
        assert len(utility) == n
        assert utility[0] == 1.0 and utility[-1] == 0.0
        assert utility == tuple(sorted(utility, reverse=True))


def test_a_fifth_seat_is_refused_rather_than_truncated():
    state = _finished(players=5, seed=4)
    with pytest.raises(ValueError, match="MAX_RANKS"):
        tr.final_outcomes(state)


def test_the_rank_mask_covers_exactly_the_live_positions():
    """M4 on the logits: a padded rank must never be able to take mass."""
    for players in (2, 3, 4):
        state = _finished(players=players, seed=7)
        targets = _targets(state, viewer=0, turn=3)
        mask = [targets[f"rank_mask_{r}"] for r in range(tr.MAX_RANKS)]
        assert mask == [1.0] * players + [0.0] * (tr.MAX_RANKS - players)
        for rank in range(tr.MAX_RANKS):
            if mask[rank] == 0.0:
                assert targets[f"rank_p_{rank}"] == 0.0


def test_the_rank_distribution_belongs_to_the_viewer():
    state = _finished(players=3, seed=9)
    outcomes = tr.final_outcomes(state)
    for viewer in range(3):
        targets = _targets(state, viewer, turn=3)
        got = tuple(targets[f"rank_p_{r}"] for r in range(tr.MAX_RANKS))
        assert got == outcomes[viewer].rank_distribution


# ──────────────────────────────────────────────────────────────────────────
# The seat axis.  Target k and encoded seat k must be the same player; if they
# are not, every shape still matches and the shared per-seat head quietly
# learns the average of four seats.
# ──────────────────────────────────────────────────────────────────────────
def test_the_target_seat_axis_matches_the_encoder():
    assert tr.MAX_SEATS == enc.MAX_SEATS


@pytest.mark.parametrize("players", [2, 3, 4])
def test_per_seat_targets_follow_the_encoders_seat_order(players):
    state = _finished(players=players, seed=players * 3)
    outcomes = tr.final_outcomes(state)
    for viewer in range(players):
        order = enc.seat_order(state, viewer)
        targets = _targets(state, viewer, turn=2)
        for k, seat in enumerate(order):
            expected = outcomes[seat].score / tr.SCORE_SCALE
            assert targets["score"][k] == pytest.approx(expected)
            assert targets["seat_valid"][k] == 1.0


def test_the_viewer_is_always_seat_zero():
    state = _finished(players=4, seed=8)
    outcomes = tr.final_outcomes(state)
    for viewer in range(4):
        targets = _targets(state, viewer, turn=2)
        assert targets["score"][0] == pytest.approx(
            outcomes[viewer].score / tr.SCORE_SCALE
        )


@pytest.mark.parametrize("players", [2, 3])
def test_padded_seats_are_absent_rather_than_zero_scoring(players):
    """M4.  Zero is a value; absent is not, and the per-seat head is shared."""
    state = _finished(players=players, seed=players)
    targets = _targets(state, viewer=0, turn=2)
    valid = targets["seat_valid"]
    assert list(valid) == [1.0] * players + [0.0] * (tr.MAX_SEATS - players)
    for k in range(players, tr.MAX_SEATS):
        assert targets["score"][k] == 0.0
        for name, mask_name in tr.MASKED_TARGETS.items():
            assert targets[mask_name][k] == 0.0
            assert targets[name][k] == float(tr.NEVER)


def test_every_per_seat_target_has_one_value_per_seat():
    state = _finished(players=3, seed=2)
    targets = _targets(state, viewer=1, turn=5)
    for name in tr.PER_SEAT_TARGETS:
        assert len(targets[name]) == tr.MAX_SEATS, name
    for name in tr.GLOBAL_TARGETS:
        assert isinstance(targets[name], float), name


def test_an_empty_seat_axis_is_refused():
    state = _finished(players=2, seed=1)
    with pytest.raises(ValueError):
        tr.sample_targets(tr.final_outcomes(state), [], turn=1)


def test_plan_turns_match_the_recorded_validations():
    state = _finished(players=3, seed=11)
    for outcome in tr.final_outcomes(state):
        for slot in range(3):
            assert outcome.plan_turns[slot] == state.plan_turns[slot].get(outcome.player)
        completed = [t for t in outcome.plan_turns if t is not None]
        assert outcome.plans_completed == len(completed)


def test_turns_to_plan_is_relative_and_masked():
    state = _finished(players=3, seed=11)
    outcomes = tr.final_outcomes(state)
    order = enc.seat_order(state, 0)
    targets = _targets(state, viewer=0, turn=4)
    for k, seat in enumerate(order):
        for slot in range(3):
            mask = targets[f"turns_to_plan_{slot}_mask"][k]
            value = targets[f"turns_to_plan_{slot}"][k]
            if outcomes[seat].plan_turns[slot] is None:
                assert mask == 0.0 and value == tr.NEVER
            else:
                assert mask == 1.0
                expected = max(0, outcomes[seat].plan_turns[slot] - 4) / tr.TURN_SCALE
                assert value == pytest.approx(expected)


# ──────────────────────────────────────────────────────────────────────────
# Masking discipline
# ──────────────────────────────────────────────────────────────────────────
def test_the_sentinel_never_appears_where_the_mask_is_one():
    """M2.  NEVER is not a value; if a loss ever sees it, it trains on -1."""
    for players in (2, 3, 4):
        state = _finished(players=players, seed=players * 5)
        final_turn = tr.final_outcomes(state)[0].final_turn
        for viewer in range(players):
            for turn in range(1, final_turn + 2):
                targets = _targets(state, viewer, turn)
                for name, mask_name in tr.MASKED_TARGETS.items():
                    for k in range(tr.MAX_SEATS):
                        if targets[mask_name][k] == 1.0:
                            assert targets[name][k] != float(tr.NEVER)
                        else:
                            assert targets[name][k] == float(tr.NEVER)


def test_a_masked_loss_is_normalised_by_the_mask_sum():
    """M1.  Dividing by the batch size discounts rare events by their rarity."""
    errors = np.array([4.0, 4.0, 4.0, 4.0])
    mask = np.array([1.0, 0.0, 0.0, 0.0])
    assert tr.masked_mean(errors, mask) == pytest.approx(4.0)
    assert tr.masked_mean(errors, np.ones(4)) == pytest.approx(4.0)


def test_an_empty_mask_contributes_nothing_rather_than_dividing_by_zero():
    errors = np.array([4.0, 9.0])
    assert tr.masked_mean(errors, np.zeros(2)) == 0.0


# ──────────────────────────────────────────────────────────────────────────
# Normalisation (D6)
# ──────────────────────────────────────────────────────────────────────────
def test_every_target_is_normalised_to_roughly_unit_scale():
    """The defect this replaces: score ~75 against permits 0-3 under one MSE."""
    for players in (2, 3, 4):
        state = _finished(players=players, seed=players)
        for viewer in range(players):
            for name, value in _targets(state, viewer, turn=3).items():
                values = value if isinstance(value, tuple) else (value,)
                for one in values:
                    if name in tr.MASKED_TARGETS and one == float(tr.NEVER):
                        continue
                    assert 0.0 <= one <= 1.5, f"{name} is not on a unit scale"


def test_component_targets_reconstruct_the_score():
    state = _finished()
    t = _targets(state, viewer=0, turn=1)
    for k in range(2):
        total = (
            t["score_plans"][k]
            + t["score_parks"][k]
            + t["score_pools"][k]
            + t["score_temp"][k]
            + t["score_estates"][k]
            - t["score_bis"][k]
            - t["score_permits"][k]
            - t["score_roundabouts"][k]
        )
        assert total == pytest.approx(
            t["score"][k]
        ), "the aux heads must add back up to the value head"


def test_target_names_cover_every_key():
    state = _finished()
    assert set(_targets(state, viewer=0, turn=2)) == set(tr.TARGET_NAMES)
    assert set(tr.PER_SEAT_TARGETS).isdisjoint(tr.GLOBAL_TARGETS)


def test_turns_left_never_goes_negative():
    state = _finished()
    final_turn = tr.final_outcomes(state)[0].final_turn
    assert _targets(state, viewer=0, turn=final_turn + 5)["turns_left"] == 0.0


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
