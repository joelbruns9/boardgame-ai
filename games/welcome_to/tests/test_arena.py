"""The permanent paired-seed harness."""
from __future__ import annotations

import random

import pytest
import torch

from games.welcome_to import arena
from games.welcome_to import macro_codec as mc
from games.welcome_to import mcts
from games.welcome_to import network as nw
from games.welcome_to.bots import GreedyBot
from games.welcome_to.game import GameConfig, GameState


def _greedy_subject(state: GameState, seat: int, rng: random.Random) -> None:
    state.apply(GreedyBot(rng).act(state))


def test_the_baseline_arm_is_a_plain_all_greedy_game():
    """If the two arms differ in anything but the substituted seat, the delta is
    not attributable to the subject.  So the baseline is rebuilt from scratch
    here and has to match to the point."""
    result = arena.paired(_greedy_subject, games=1, seats=2, seed=20_000)

    bots = [GreedyBot(random.Random(20_000 * 100 + p)) for p in range(2)]
    state = GameState.new(seed=20_000, config=GameConfig(players=2, advanced=True))
    while not state.is_terminal:
        state.apply(bots[state.actor].act(state))
    assert result.baseline_score == pytest.approx(float(state.scores()[0]))


def test_a_greedy_subject_scores_indistinguishably_from_the_baseline():
    """The null case.  GreedyBot against GreedyBot must land inside its own
    noise -- if it does not, the harness is biased, not the players."""
    result = arena.paired(_greedy_subject, games=24, seats=2)
    assert abs(result.score_gap) < 3.0 * result.score_gap_stderr


def test_the_harness_reports_its_own_noise():
    """A 4-point gate read off a sample whose stderr is 4 points means nothing."""
    result = arena.paired(_greedy_subject, games=24, seats=2)
    assert result.score_gap_stderr > 0.0
    assert result.games == 24


def test_it_is_reproducible():
    a = arena.paired(_greedy_subject, games=6, seats=2)
    b = arena.paired(_greedy_subject, games=6, seats=2)
    assert a == b


def test_the_evaluated_seat_rotates():
    """Otherwise the result is one seat's luck, repeated."""
    seats_used = []

    def spy(state: GameState, seat: int, rng: random.Random) -> None:
        seats_used.append(seat)
        state.apply(GreedyBot(rng).act(state))

    arena.paired(spy, games=4, seats=2)
    assert set(seats_used) == {0, 1}


def test_the_baseline_plan_rate_matches_the_documented_teacher():
    """GreedyBot completes ~0.42 plans a game; that is the number S1 must beat,
    and it is a property of the teacher, not of this harness."""
    result = arena.paired(_greedy_subject, games=40, seats=2)
    assert 0.1 < result.baseline_plans < 1.0


def test_the_gate_reads_both_conditions():
    perfect = arena.ArenaResult(
        games=10,
        score_gap=5.0,
        score_gap_stderr=0.5,
        subject_score=60.0,
        baseline_score=55.0,
        subject_plans=1.2,
        baseline_plans=0.4,
        subject_wins=0.7,
    )
    assert all(arena.s1_gate(perfect).values())

    # a score gain with no plans is exactly the case the plan number exists to
    # catch: better placement, still race-blind
    placement_only = arena.ArenaResult(
        games=10,
        score_gap=5.0,
        score_gap_stderr=0.5,
        subject_score=60.0,
        baseline_score=55.0,
        subject_plans=0.4,
        baseline_plans=0.4,
        subject_wins=0.7,
    )
    gate = arena.s1_gate(placement_only)
    assert gate["score_beats_greedy_by_4"] and not gate["completes_a_plan_per_game"]


def test_a_search_can_be_dropped_into_the_harness():
    """The adapter every later change goes through.

    Running a whole paired game is the assertion: ``apply_macro`` raises on any
    illegal step, so a completed pair means every searched move was legal end to
    end -- and it exercises the real integration path rather than a mock of it.
    """
    config = mcts.SearchConfig(simulations=2)
    net = nw.WelcomeToNet(
        nw.NetConfig(
            sheet_hidden=16, sheet_out=8, trunk_hidden=24, trunk_blocks=1, head_hidden=16
        )
    )
    search = mcts.MCTS(mcts.NetEvaluator(net, torch.device("cpu"), config), config)

    result = arena.paired(mcts.arena_player(search), games=2, seats=2)
    assert result.games == 2
    assert result.subject_score > 0.0
    assert 0.0 <= result.subject_wins <= 1.0


def test_the_search_root_is_the_arena_seat_not_the_default_actor():
    """Clause 3 one layer up: the arena substitutes a seat, so that seat is the
    search root.  Falling back to ``state.actor`` would work by accident here
    and break the moment the harness evaluates a seat other than the mover."""
    seen = set()

    class Spy(mcts.MCTS):
        def play(self, state, root=None, rng=None):
            seen.add(root)
            return super().play(state, root, rng)

    config = mcts.SearchConfig(simulations=1)
    net = nw.WelcomeToNet(
        nw.NetConfig(
            sheet_hidden=16, sheet_out=8, trunk_hidden=24, trunk_blocks=1, head_hidden=16
        )
    )
    search = Spy(mcts.NetEvaluator(net, torch.device("cpu"), config), config)
    arena.paired(mcts.arena_player(search), games=2, seats=2)
    assert None not in seen, "the root fell back to state.actor"
    assert seen == {0, 1}, "the root did not follow the rotating seat"
