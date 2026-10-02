"""Luck ledger: reconstruction from captures, and the ledger's accounting.

Games are simulated with the heuristic evaluator, every roll that leaves a
legal move captured exactly as the advisor would (AWAIT_MOVE, dice showing),
so the true moves are known and the reconstruction can be held to them.
"""

import random

import numpy as np
import pytest

from games.cantstop.advisor_adapter import wins_on_stop
from games.cantstop.engine import (GameState, Phase, RuleSet, apply_move,
                                   can_stop, legal_moves, random_dice, roll,
                                   stop)
from games.cantstop.luck import (DICE, Capture, Game, after_move, at,
                                 build_ledger, bust_probability, is_stale,
                                 reconstruct, seat_summary)
from games.cantstop.rust_solver import RustTurnSolver, sample_positions
from games.cantstop.solver import ProgressHeuristic

HEURISTIC = ProgressHeuristic()


def simulate(rules, seed, deviate=0.0):
    """One game; returns captures, the true (move, then) per capture, and the
    true end of each turn as (seat, 'stop' | 'bust' | 'win')."""
    rng = random.Random(seed)
    state = GameState(rules)
    caps, truth, ends = [], [], []
    turn = 0
    while not state.game_over:
        assert turn < 500
        solver = RustTurnSolver(state, HEURISTIC)
        seat = state.active_player
        while True:
            moves = roll(state, random_dice(rng))
            if not moves:
                ends.append((seat, "bust"))
                break
            caps.append(Capture(state.clone(), f"sim:{turn}"))
            m = (rng.choice(moves) if rng.random() < deviate
                 else solver.choose_move(state))
            apply_move(state, m)
            if wins_on_stop(state):
                then = "stop"
            elif not can_stop(state):
                then = "roll"
            elif rng.random() < deviate:
                then = rng.choice(["stop", "roll"])
            else:
                then = "stop" if solver.should_stop(state) else "roll"
            truth.append((m, then))
            if then == "stop":
                stop(state)
                ends.append((seat, "win" if state.game_over else "stop"))
                break
        turn += 1
    return caps, truth, ends, state.winner


def game_of(rules, caps, viewer=0, opponents_logged=True):
    ids = [f"p{i}" for i in range(rules.num_players)]
    return Game("sim", ids, viewer, opponents_logged, caps)


RULES = [RuleSet.make(2), RuleSet.make(3, extended=True, blocking=True)]


def test_bust_probability_matches_engine():
    for rules in RULES:
        for s in sample_positions(rules, 3, 12):
            if s.phase == Phase.AWAIT_MOVE:
                continue
            ordered = [(a, b, c, d) for a in range(1, 7) for b in range(1, 7)
                       for c in range(1, 7) for d in range(1, 7)]
            brute = sum(not legal_moves(s, d) for d in ordered) / 1296
            assert bust_probability(s, s.runners) == pytest.approx(brute, abs=1e-12)


def test_first_roll_before_value_is_the_exact_average():
    """The trap: a turn's first-roll 'before' must be the average over every
    roll, so first-roll luck averages zero. Checked against an explicit sum."""
    for rules in RULES:
        for s in sample_positions(rules, 5, 8):
            if s.phase != Phase.AWAIT_ROLL or s.runners:
                continue
            solver = RustTurnSolver(s, HEURISTIC)
            avg = np.zeros(rules.num_players)
            for d, w in DICE:
                if legal_moves(s, d):
                    avg += w / 1296 * np.asarray(solver.value(at(s, {}, Phase.AWAIT_MOVE, d)))
            avg += bust_probability(s, {}) * solver.bust_value
            np.testing.assert_allclose(solver.value(s), avg, atol=1e-12)


@pytest.mark.parametrize("rules", RULES)
@pytest.mark.parametrize("seed", [1, 2])
def test_reconstruction_recovers_the_true_moves(rules, seed):
    caps, truth, ends, winner = simulate(rules, seed, deviate=0.2)
    turns = reconstruct(game_of(rules, caps))
    # every captured roll lines up with its true move
    steps = [s for t in turns for s in t.steps if s.dice is not None]
    assert len(steps) == len(truth)
    for step, (m, then) in zip(steps, truth):
        if step.candidates is not None:     # the move before a bust: not observable
            assert then == "roll" and m in step.candidates
        else:
            assert (step.move, step.then) == (m, then)
    # every turn end, including first-roll busts (which leave no capture)
    assert [(t.seat, t.end) for t in turns] == ends
    assert turns[-1].end == "win" and turns[-1].seat == winner


def _captures_per_turn(caps, ends):
    counts = {}
    for c in caps:
        counts[c.turn_id] = counts.get(c.turn_id, 0) + 1
    return [counts.get(f"sim:{i}", 0) for i in range(len(ends))]


@pytest.mark.parametrize("rules", RULES)
def test_ledger_adds_up_to_the_result(rules):
    caps, truth, ends, winner = simulate(rules, 7, deviate=0.25)
    game = game_of(rules, caps)
    L = build_ledger(game, reconstruct(game), HEURISTIC)
    expected = np.zeros(rules.num_players)
    expected[winner] = 1.0
    np.testing.assert_allclose(L.outcome, expected)
    total = L.start + sum(e["delta"] for e in L.entries)
    np.testing.assert_allclose(total, L.outcome, atol=1e-9)
    for seat in range(rules.num_players):
        seat_summary(L, seat)          # asserts the headings add up
    kinds = {e["kind"] for e in L.entries}
    assert "gap" not in kinds and {"luck", "decision", "residual"} <= kinds


def test_decisions_cost_the_mover_only_when_they_deviate():
    rules = RULES[0]
    for deviate, seed in ((0.0, 11), (0.4, 12)):
        caps, truth, ends, winner = simulate(rules, seed, deviate=deviate)
        game = game_of(rules, caps)
        L = build_ledger(game, reconstruct(game), HEURISTIC)
        own = [e["delta"][e["actor"]] for e in L.entries if e["kind"] == "decision"]
        assert max(own) <= 1e-12               # never better than the best
        if deviate == 0.0:
            assert min(own) >= -1e-12          # following the solver costs nothing
        else:
            assert min(own) < -1e-3


def test_first_roll_bust_needs_the_turn_counter_to_agree():
    """A turn with no capture and no change reads as a first-roll bust only
    when the page's turn counter says exactly that many turns passed."""
    rules = RULES[1]
    caps, truth, ends, winner = simulate(rules, 7, deviate=0.25)
    counts = _captures_per_turn(caps, ends)
    full = reconstruct(game_of(rules, caps))
    assert all(t.link != "gap" for t in full[1:])
    # the first capture after an uncaptured turn
    silent = next(i for i, n in enumerate(counts) if n == 0)
    k = next(j for j, c in enumerate(caps) if int(c.turn_id[4:]) > silent)
    for relabel in (lambda c: "reload:" + c.turn_id[4:],           # page reloaded
                    lambda c: "sim:" + str(int(c.turn_id[4:]) + 5)):  # turns unaccounted for
        broken = caps[:k] + [Capture(c.state, relabel(c)) for c in caps[k:]]
        turns = reconstruct(game_of(rules, broken))
        assert sum(t.link == "gap" for t in turns) == 1
        assert len(turns) < len(full)       # that bust is no longer claimed


def test_a_missed_capture_becomes_a_gap_not_a_wrong_move():
    rules = RULES[0]
    caps, truth, ends, winner = simulate(rules, 3)
    # drop a capture strictly inside a turn
    i = next(i for i in range(1, len(caps) - 1)
             if caps[i - 1].turn_id == caps[i].turn_id == caps[i + 1].turn_id)
    game = game_of(rules, caps[:i] + caps[i + 1:])
    turns = reconstruct(game)
    assert any(s.dice is not None and s.move is None and s.candidates is None
               for t in turns for s in t.steps[:-1])
    L = build_ledger(game, turns, HEURISTIC)
    assert any(e["kind"] == "gap" for e in L.entries)
    np.testing.assert_allclose(L.start + sum(e["delta"] for e in L.entries), L.final, atol=1e-9)


def test_stale_capture_is_recognised():
    rules = RULES[0]
    caps, truth, ends, winner = simulate(rules, 4)
    i = next(i for i, (m, then) in enumerate(truth) if then == "roll"
             and caps[i + 1].state.dice != caps[i].state.dice)
    stale = after_move(caps[i].state, truth[i][0])
    stale.phase, stale.dice = Phase.AWAIT_MOVE, caps[i].state.dice
    assert is_stale(caps[i], Capture(stale, caps[i].turn_id))
    assert not is_stale(caps[i], caps[i + 1])


def test_viewer_only_logs_still_read_the_viewers_turns():
    """Older logs hold only the viewer's captures: opponents' turns (and
    their claims, which wipe the viewer's progress) happen unseen."""
    rules = RULES[0]
    for seed in (21, 22, 23):
        caps, truth, ends, winner = simulate(rules, seed)
        mine = [c for c in caps if c.state.active_player == 0]
        turns = reconstruct(game_of(rules, mine, opponents_logged=False))
        true_ends = [e for e, n in zip(ends, _captures_per_turn(caps, ends))
                     if n and e[0] == 0]
        observed = [(t.seat, t.end) for t in turns]
        # the last viewer turn has no next capture to explain it
        assert observed[:-1] == true_ends[:-1]
        assert all(t.link != "adjacent" for t in turns)


def test_every_roll_is_judged_against_the_exact_dice_average():
    """Each luck entry's 'before' is the average over all 1296 rolls of the
    after-roll value (bust included) -- recomputed here by explicit sum. The
    shortcut of the evaluator's own value of the board would fail this."""
    rules = RULES[1]
    caps, truth, ends, winner = simulate(rules, 9, deviate=0.2)
    game = game_of(rules, caps)
    turns = reconstruct(game)
    L = build_ledger(game, turns, HEURISTIC)
    luck = [e for e in L.entries if e["kind"] == "luck"]
    assert any(not e["runners"] for e in luck)
    for e in luck:
        start = turns[e["turn"]].start
        solver = RustTurnSolver(start, HEURISTIC)
        K = e["runners"]
        avg = bust_probability(start, K) * solver.bust_value
        for d, w in DICE:
            if legal_moves(at(start, K, Phase.AWAIT_MOVE), d):
                avg = avg + w / 1296 * np.asarray(solver.value(at(start, K, Phase.AWAIT_MOVE, d)))
        np.testing.assert_allclose(e["pre"], avg, atol=1e-12)
