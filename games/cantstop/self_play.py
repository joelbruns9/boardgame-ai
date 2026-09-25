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


@dataclass(frozen=True)
class Search:
    """How each turn is searched. The defaults are the plain one-turn
    solve, rooted after the opening roll.

    exact_root        solve BEFORE the opening roll: the recorded turn value
                      becomes the exact expectation over every roll (the
                      exact TD backup, as PureTD's 1-ply target). Decisions
                      are identical -- one table answers them all.
    lookahead_k       selective 2-turn lookahead: refine this many leaves
                      (the bust board + the top k-1 stop leaves by reach) by
                      solving the next player's turn from them. 0 = off.
    lookahead_offset  shift unrefined leaves by the mean refinement.
    """
    exact_root: bool = False
    lookahead_k: int = 0
    lookahead_offset: bool = True


PLAIN = Search()


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
    # Per turn: the solver's absolute-seat value right after the opening
    # roll, or None when that roll busted. The TD bootstrap (td_targets).
    turn_values: list = field(default_factory=list)
    boards: list = None           # only when play_game(keep_boards=True)

    def __len__(self):
        return len(self.winner_slots)


def _solve(state, evaluate, search):
    solver = TurnSolver(state, evaluate)
    if search.lookahead_k:
        from .lookahead import refine
        refine(solver, state, evaluate, search.lookahead_k,
               search.lookahead_offset)
    return solver


def play_turn(state, evaluate, rng, values=None, search=PLAIN):
    """Play the active player's whole turn. Returns (solves, decisions).

    ``state`` is mutated in place and left at the start of the next player's
    turn (or game over). If ``values`` is a list, the solver's value at its
    root is appended: after the opening roll, or before it with
    ``search.exact_root`` (None for an opening bust with no solve).
    """
    if search.exact_root:
        solver = _solve(state, evaluate, search)
        if values is not None:
            values.append(solver.value(state).tolist())
        if not roll(state, random_dice(rng)):
            return 1, 0           # busted on the opening roll
    else:
        moves = roll(state, random_dice(rng))
        if not moves:
            if values is not None:
                values.append(None)
            return 0, 0           # busted on the opening roll; no solve needed
        solver = _solve(state, evaluate, search)
        if values is not None:
            values.append(solver.value(state).tolist())
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
              keep_boards=False, search=PLAIN):
    """Play one game to completion and collect its training rows."""
    state = GameState(rules)
    boards, turns, solves, turn_lengths, turn_values = [], 0, 0, [], []
    evaluator_rows_before = getattr(evaluate, "rows", 0)

    while not state.game_over:
        if turns >= max_turns:
            raise TurnLimitExceeded(
                f"{rules} reached {max_turns} turns with no winner; the "
                "policy is likely never banking progress")
        used, decisions = play_turn(state, evaluate, rng, turn_values,
                                    search)
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
        turn_values=turn_values,
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


def td_targets(result, lam):
    """Soft value targets (rows, MAX_SEATS) float32, in each row's encoding
    slots: the lambda-return over the game's turns.

    Row i is the board after turn i; turn i+1 starts from it, and the
    solver's value at that turn's opening roll (``turn_values[i + 1]``) is a
    one-turn-deeper estimate of the row's value, sampled over the opening
    roll. So, backwards from the final outcome z (one-hot winner):

        G_R = z;   G_i = (1 - lam) * v_{i+1} + lam * G_{i+1}

    ``lam = 1`` is the outcome-only label; ``lam = 0`` is one-step TD against
    the search. An opening bust has no solve, and its board is the same board
    with the turn passed on, so G passes through unchanged there.
    Absolute-seat vectors are mixed, then rotated to each row's slots.
    """
    from .encoder import MAX_SEATS
    rows = len(result)
    n = result.rules.num_players
    if rows == 0:
        return np.zeros((0, MAX_SEATS), dtype=np.float32)
    if len(result.turn_values) != rows + 1:
        raise ValueError(f"{rows} rows need {rows + 1} turn values, got "
                         f"{len(result.turn_values)}")
    # Row i's encoding puts seat (active_i + k) % n in slot k, and the
    # winner's slot is recorded; recover active_i from it.
    g = np.zeros(n)
    g[result.winner] = 1.0
    out = np.zeros((rows, MAX_SEATS), dtype=np.float32)
    for i in range(rows - 1, -1, -1):
        v = result.turn_values[i + 1]
        if v is not None:
            g = (1.0 - lam) * np.asarray(v, dtype=np.float64) + lam * g
        active = (result.winner - int(result.winner_slots[i])) % n
        out[i, :n] = [g[(active + k) % n] for k in range(n)]
    return out


def stack_training(results, lam):
    """(features, soft targets) for ``train_steps``; ``lam = 1`` reproduces
    the one-hot winner labels of ``stack_rows``."""
    usable = [r for r in results if len(r)]
    if not usable:
        raise ValueError("no rows to stack")
    return (np.concatenate([r.features for r in usable]),
            np.concatenate([td_targets(r, lam) for r in usable]))


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
