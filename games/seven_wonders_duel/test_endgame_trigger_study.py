"""The trigger study's arithmetic and its position features.

The study exists to decide where compute goes, so a quiet error here does not
raise -- it produces a plausible threshold that silently wastes budget or skips
the positions worth solving. Two things are pinned: that budget simulation is
exact (it is what lets one expensive pass stand in for every candidate budget),
and that the features mean what their names say on real positions.
"""

from __future__ import annotations

import math

import pytest

from .encoder_audit import DEFAULT_PAIRINGS, make_bot
from .endgame_trigger_study import (
    FEATURES,
    calibrate,
    position_features,
    report,
    simulate_budget,
)
from .engine import apply_action
from .game import Phase, new_game


def _row(nodes, cards=10, **kw):
    base = {f: 0 for f in FEATURES}
    base.update(
        {
            "game": 0,
            "move": 0,
            "cards_left": cards,
            "legal": 4,
            "nodes": nodes,
            "censored": nodes is None,
            "regime": "exact",
            "seconds": 0.0,
            "root_value": 1.0,
            "optimal": 2,
        }
    )
    base.update(kw)
    return base


# --- budget simulation ------------------------------------------------------


def test_a_solve_inside_the_budget_costs_what_it_needed_not_the_budget():
    """The whole measure-once-simulate-many design rests on this: a cheap
    position does not consume its allowance, it consumes its node count."""

    stats = simulate_budget([_row(100_000)], 5_000_000)
    assert stats["solved"] == 1
    assert stats["nodes_useful"] == 100_000
    assert stats["nodes_wasted"] == 0


def test_a_solve_over_the_budget_burns_exactly_the_budget():
    stats = simulate_budget([_row(9_000_000)], 5_000_000)
    assert stats["solved"] == 0
    assert stats["nodes_wasted"] == 5_000_000
    assert stats["nodes_useful"] == 0


def test_a_censored_position_declines_under_every_smaller_budget():
    """Censored means "needed more than the STUDY budget", so it is known to
    decline below that -- not unknown, and certainly not cheap."""

    stats = simulate_budget([_row(None)], 20_000_000)
    assert stats["solved"] == 0
    assert stats["nodes_wasted"] == 20_000_000


def test_the_trigger_rule_decides_what_is_even_attempted():
    rows = [_row(10, cards=6), _row(10_000_000, cards=12)]
    stats = simulate_budget(rows, 1_000_000, rule=lambda r: r["cards_left"] <= 8)
    assert stats["attempted"] == 1
    assert stats["solved"] == 1
    assert stats["nodes_wasted"] == 0


def test_solves_per_million_nodes_counts_wasted_nodes_too():
    """Otherwise a rule that declines constantly would score as efficient."""

    rows = [_row(1_000_000), _row(None)]
    stats = simulate_budget(rows, 1_000_000)
    # 1 solve for 2M nodes total, not for the 1M that produced it.
    assert stats["solves_per_million_nodes"] == pytest.approx(0.5)


def test_an_empty_attempt_set_reports_none_rather_than_dividing_by_zero():
    stats = simulate_budget([_row(10)], 1_000, rule=lambda r: False)
    assert stats["attempted"] == 0
    assert stats["decline_rate"] is None
    assert stats["solves_per_million_nodes"] is None


# --- calibration ------------------------------------------------------------


def test_calibration_never_recommends_more_than_the_time_allowance():
    rows = [_row(500_000, cards=8), _row(30_000_000, cards=12)]
    result = calibrate(rows, rate=1_000_000.0, seconds_per_game=1.0, games=1)
    assert result["recommended"]["seconds_per_game"] <= 1.0


def test_calibration_returns_nothing_when_the_allowance_buys_nothing():
    rows = [_row(50_000_000, cards=12)]
    result = calibrate(rows, rate=1_000.0, seconds_per_game=0.001, games=1)
    assert result["recommended"] is None


def test_a_faster_machine_can_afford_at_least_as_much():
    rows = [_row(500_000, cards=8), _row(8_000_000, cards=11)]
    slow = calibrate(rows, rate=1e6, seconds_per_game=1.0, games=1)["recommended"]
    fast = calibrate(rows, rate=1e8, seconds_per_game=1.0, games=1)["recommended"]
    assert fast["solves_per_game"] >= slow["solves_per_game"]


def test_report_separates_censored_positions_from_priced_ones():
    summary = report([_row(1_000), _row(None)], [5_000_000])
    assert summary["positions"] == 2
    assert summary["censored"] == 1
    # The median must come from the priced row alone; counting the censored one
    # as either cheap or absent would bias every threshold drawn off it.
    assert summary["by_cards_left"]["10"]["median_nodes"] == 1_000


# --- position features, on real positions -----------------------------------


def _age_three_position(cards: int):
    for seed in range(20):
        game = new_game(seed)
        bots = (
            make_bot(DEFAULT_PAIRINGS[0][0], seed),
            make_bot(DEFAULT_PAIRINGS[0][1], seed + 13),
        )
        while game.phase is not Phase.COMPLETE:
            if (
                game.phase is Phase.PLAY_AGE
                and game.age == 3
                and sum(1 for c in game.tableau.cards.values() if c.present) == cards
            ):
                return game
            actor = (
                game.pending_choice.player
                if game.pending_choice is not None
                else game.active_player
            )
            apply_action(game, bots[actor].select_action(game))
    pytest.skip(f"no Age III position with {cards} cards")


def test_every_declared_feature_is_actually_produced():
    """`FEATURES` drives the correlation report, so a name in one and not the
    other silently drops a variable from the analysis."""

    features = position_features(_age_three_position(10))
    missing = [name for name in FEATURES if name not in features and name != "moves_from_end"]
    assert not missing, missing


def test_cards_left_and_unrevealed_agree_with_the_board():
    game = _age_three_position(10)
    features = position_features(game)
    present = [c for c in game.tableau.cards.values() if c.present]
    assert features["cards_left"] == len(present) == 10
    assert features["unrevealed"] == sum(1 for c in present if not c.revealed)
    assert features["unrevealed"] <= features["cards_left"]


def test_chance_fanout_is_zero_exactly_when_nothing_is_face_down():
    """It is a sum of logs over face-down slots, so an all-revealed board must
    contribute nothing -- a nonzero floor there would mean minimax positions
    were being priced as if they had chance edges."""

    game = _age_three_position(4)
    features = position_features(game)
    if features["unrevealed"] == 0:
        assert features["chance_fanout"] == 0.0
        assert features["chance_fanout_max"] == 0
    else:
        assert features["chance_fanout"] > 0.0
        assert features["chance_fanout_max"] >= 1


def test_the_interaction_terms_match_their_definitions():
    game = _age_three_position(10)
    features = position_features(game)
    assert features["cards_x_logleg"] == pytest.approx(
        features["cards_left"] * math.log10(max(1, features["legal"]))
    )
    assert features["discard_x_revive"] == (
        features["discard"] * features["revive_wonders"]
    )


def test_military_to_win_is_the_distance_to_an_instant_win():
    game = _age_three_position(10)
    features = position_features(game)
    assert features["military"] + features["military_to_win"] == 9
    assert 0 <= features["military_to_win"] <= 9


def test_science_threat_is_one_symbol_from_the_instant_win():
    game = _age_three_position(10)
    features = position_features(game)
    assert features["science_threat"] == int(features["science_max"] >= 5)
    assert features["science_max"] <= 6


# --- pricing positions out of existing buffers ------------------------------


def _finished_record(seed: int = 7):
    """One complete bot game, recorded exactly as self-play would record it."""

    from .buffer import GameRecorder
    from .codec import encode_action

    recorder = GameRecorder(seed=seed, agents={"kind": "test"})
    bots = (
        make_bot(DEFAULT_PAIRINGS[0][0], seed),
        make_bot(DEFAULT_PAIRINGS[0][1], seed + 13),
    )
    while recorder.game.phase is not Phase.COMPLETE:
        game = recorder.game
        actor = (
            game.pending_choice.player
            if game.pending_choice is not None
            else game.active_player
        )
        recorder.play(encode_action(game, bots[actor].select_action(game)))
    return recorder.finish()


def test_pricing_refuses_a_record_whose_trajectory_diverges():
    """The tolerance must not extend to a mid-game divergence.

    A mask mismatch means the old engine offered a move this one does not, so
    every later recorded action was chosen for a position that no longer exists.
    Pricing those would quietly measure arbitrary play.
    """

    import dataclasses

    from .buffer import ReplayMismatchError
    from .endgame_trigger_study import price_records

    record = _finished_record()
    moves = list(record.moves)
    broken = dataclasses.replace(moves[2], mask_hash="sha256:not-the-real-mask")
    moves[2] = broken
    tampered = dataclasses.replace(record, moves=tuple(moves))

    with pytest.raises(ReplayMismatchError, match="mask hash"):
        price_records(
            [tampered],
            max_cards=2,
            study_nodes=1,
            study_secs=0.1,
            allow_final_digest_drift=True,
        )


def test_pricing_tolerates_only_a_terminal_score_difference():
    """The cloud buffers' exact shape: every position replays, the final
    state does not, because the military fix changed how the game is scored.

    Positions are captured before the terminal state, so none of them can be
    affected -- but the default must still refuse, since for a freshly
    generated record the same mismatch means the engine disagrees with itself.
    """

    import dataclasses

    from .buffer import ReplayMismatchError
    from .endgame_trigger_study import price_records

    record = _finished_record()
    tampered = dataclasses.replace(record, final_digest="sha256:different-score")

    with pytest.raises(ReplayMismatchError, match="final digest"):
        price_records([tampered], max_cards=0, study_nodes=1, study_secs=0.1)

    # Tolerated, and with max_cards=0 no position is priced, so this exercises
    # the replay path alone rather than the solver.
    assert (
        price_records(
            [tampered],
            max_cards=0,
            study_nodes=1,
            study_secs=0.1,
            allow_final_digest_drift=True,
        )
        == []
    )
