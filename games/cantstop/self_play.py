"""Self-play: solver-driven games that produce value-net training rows.

One solve per turn. The turn solver builds its table rooted at the first
roll, and that one table answers every later decision in the same turn --
which move to take, and whether to stop or roll on -- so a turn costs exactly
one evaluator call no matter how long it runs.

One training row per **turn end**, holding the board as it stands after the
stop or the bust, labelled with the eventual winner of the game. That is
deliberately the same distribution the net is asked to score during search:
the solver only ever evaluates end-of-turn boards.

Boards that end the game are **not** rows. A winning stop is scored exactly
by the solver and never reaches the net, so training on terminal boards would
teach a distribution the net is never asked about.
"""

from dataclasses import dataclass, field

import numpy as np

from .encoder import FEATURE_SIZE, encode_board, seat_to_slot
from .engine import (
    GameState, apply_move, can_stop, random_dice, roll, stop,
)
from .solver import TurnSolver

# A game that has not finished by here is treated as a defect, not a draw:
# Can't Stop has no draws, so a game this long means the policy has collapsed
# into never banking progress.
DEFAULT_MAX_TURNS = 400


class TurnLimitExceeded(RuntimeError):
    """A game ran past ``max_turns`` without anyone winning."""


@dataclass
class GameResult:
    """Rows from one finished game, plus what it cost to produce them."""
    rules: object
    winner: int
    features: np.ndarray          # (rows, FEATURE_SIZE) float32
    winner_slots: np.ndarray      # (rows,) int64, encoding slots
    turns: int
    solves: int
    evaluator_rows: int
    turn_lengths: list = field(default_factory=list)
    boards: list = None           # only when play_game(keep_boards=True)

    def __len__(self):
        return len(self.winner_slots)


def play_turn(state, evaluate, rng):
    """Play the active player's whole turn. Returns (solves, decisions).

    ``state`` is mutated in place and left at the start of the next player's
    turn (or game over).
    """
    moves = roll(state, random_dice(rng))
    if not moves:
        return 0, 0               # busted on the opening roll; no solve needed

    solver = TurnSolver(state, evaluate)
    decisions = 0
    while True:
        apply_move(state, solver.choose_move(state))
        decisions += 1
        if can_stop(state) and solver.should_stop(state):
            stop(state)
            return 1, decisions
        moves = roll(state, random_dice(rng))
        if not moves:
            return 1, decisions   # busted; roll() already passed the turn


def play_game(rules, evaluate, rng, max_turns=DEFAULT_MAX_TURNS,
              keep_boards=False):
    """Play one game to completion and collect its training rows."""
    state = GameState(rules)
    boards, turns, solves, turn_lengths = [], 0, 0, []
    evaluator_rows_before = getattr(evaluate, "rows", 0)

    while not state.game_over:
        if turns >= max_turns:
            raise TurnLimitExceeded(
                f"{rules} reached {max_turns} turns with no winner; the "
                "policy is likely never banking progress")
        used, decisions = play_turn(state, evaluate, rng)
        turns += 1
        solves += used
        turn_lengths.append(decisions)
        if not state.game_over:
            boards.append(state.clone())

    winner = state.winner
    features = np.stack([encode_board(b) for b in boards]) if boards else \
        np.zeros((0, 0), dtype=np.float32)
    slots = np.array([seat_to_slot(b, winner) for b in boards], dtype=np.int64)

    return GameResult(
        rules=rules,
        winner=winner,
        features=features,
        winner_slots=slots,
        turns=turns,
        solves=solves,
        evaluator_rows=getattr(evaluate, "rows", 0) - evaluator_rows_before,
        turn_lengths=turn_lengths,
        boards=boards if keep_boards else None,
    )


def play_games(rules, evaluate, rng, games, max_turns=DEFAULT_MAX_TURNS):
    """Play ``games`` games of one rule set."""
    return [play_game(rules, evaluate, rng, max_turns) for _ in range(games)]


def stack_rows(results):
    """Concatenate several games' rows into one (features, slots) pair."""
    usable = [r for r in results if len(r)]
    if not usable:
        raise ValueError("no rows to stack")
    return (np.concatenate([r.features for r in usable]),
            np.concatenate([r.winner_slots for r in usable]))


def summarize(results):
    """Aggregate stats, including the turn-length distribution.

    Turn length is the collapse watch the plan asks for: a policy that has
    degenerated into always-stop shows length 1 everywhere, and one that has
    degenerated into always-roll shows the bust length. Either means the
    dice-variety-is-enough-exploration bet (D3) has failed.
    """
    lengths = [n for r in results for n in r.turn_lengths]
    wins = {}
    for r in results:
        wins[r.winner] = wins.get(r.winner, 0) + 1
    return {
        "games": len(results),
        "rows": sum(len(r) for r in results),
        "turns": sum(r.turns for r in results),
        "solves": sum(r.solves for r in results),
        "evaluator_rows": sum(r.evaluator_rows for r in results),
        "turns_per_game": (sum(r.turns for r in results) / len(results)
                           if results else 0.0),
        "mean_turn_length": float(np.mean(lengths)) if lengths else 0.0,
        "max_turn_length": max(lengths) if lengths else 0,
        "seat_wins": wins,
    }
