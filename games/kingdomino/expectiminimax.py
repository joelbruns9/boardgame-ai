"""Phase 0 of the NNUE project: a depth-limited expectiminimax search harness.

This is the search skeleton the NNUE eval will later plug into. It deliberately
uses a *trivial* evaluation (raw score margin) so the baseline it produces
measures the SEARCH, not any learned knowledge. See NNUE_PROJECT_PLAN.md.

Design decisions (several forced by the plan review, 2026-07-12):

* **Player-0 frame.** Every node value is in player-0's frame: a larger number is
  better for player 0. Decision nodes MAX when player 0 is to move and MIN when
  player 1 is. `choose_action` negates for player 1 so each actor picks the move
  best for itself.

* **Objective = expected match outcome.** Terminals return the official result
  via `determine_winner` (total -> largest-territory -> crowns cascade) as
  {+1, 0, -1} in player-0 frame, so the searcher maximizes expected win-minus-loss
  probability. The horizon (depth-cut) eval is a *bounded* proxy, `tanh(margin)`,
  strictly inside (-1, 1) so a proven terminal always dominates any heuristic leaf.
  NOTE (review 2026-07-12): this is NOT lexicographic under chance — a single
  scalar averaged over draws optimizes `1000·E[outcome] + E[margin]`, a weighted
  tradeoff, not "win-prob first, margin only to break exact ties". So the primary
  utility is outcome-only by default; `margin_weight > 0` re-introduces a
  deliberate outcome/margin *blend* (and forfeits strict terminal dominance),
  matching what the AlphaZero pipeline does on purpose via its alpha.

* **Correct chance handling (no strategy fusion).** Chance nodes are expanded
  INSIDE one public-state tree; the decision node above a chance node sees the
  probability-weighted average over draws, never a single realized future. Wide
  chance nodes (early game, up to C(44,4)=135,751 rows) are Monte-Carlo sampled;
  narrow ones (<= enum_cap, i.e. the last rounds) are enumerated exactly, so the
  endgame chance handling is exact for free. Draws are sampled from the KNOWN
  remaining bag (composition public, only order hidden), matching the engine's
  information-set-safe model.

* **Common random numbers.** The sampled draws at a given public chance node are
  seeded from the deck multiset, so identical chance nodes reached via different
  sibling lines draw identical futures. That makes sibling-move comparisons fair
  (noise cancels) and the whole search deterministic.

* **Alpha-beta on decision layers only.** Pruning is applied across consecutive
  MIN/MAX plies and passed through deterministic (single-child) transitions.
  Pruning THROUGH chance nodes (Star1/Star2 *-minimax) is a Phase-3 upgrade; here
  chance children are each evaluated with a full window and averaged.
"""
from __future__ import annotations

import hashlib
import itertools
import random
from math import comb, inf, tanh
from typing import Callable, Optional

from games.kingdomino.dominoes import DOMINOES
from games.kingdomino.game import GameState, Phase, determine_winner

# Margins live in ~[-80, 80]; this scales them into a sane tanh range so the
# bounded proxy neither saturates immediately nor stays near-linear everywhere.
SCORE_SCALE = 40.0


def margin_p0(state: GameState) -> float:
    """Raw score margin s0 - s1 in player-0 frame (unbounded)."""
    s = state.scores()
    return float(s[0] - s[1])


def tanh_margin_p0(state: GameState) -> float:
    """Bounded margin proxy in (-1, 1). The Phase-0 'dumb' horizon eval.

    Bounded so it can never outrank a proven terminal (±1); pick-blind by design
    (it reads the board score only, valuing the claimed-but-unplaced domino at 0)."""
    return tanh(margin_p0(state) / SCORE_SCALE)


def _claimed_crowns(state: GameState, player: int) -> int:
    """Crowns on `player`'s claimed-but-not-yet-placed dominoes (a pick signal)."""
    crowns = 0
    unplaced = list(state.next_claims)
    if state.phase != Phase.GAME_OVER:
        unplaced += state.pending_claims[state.actor_index:]
    for c in unplaced:
        if c.player == player:
            d = DOMINOES[c.domino_id]
            crowns += d.a.crowns + d.b.crowns
    return crowns


def pick_aware_p0(state: GameState, crown_weight: float = 4.0) -> float:
    """Control eval (review finding 3): margin PLUS a crude claimed-domino potential.

    Isolates whether cheap pick-awareness — not specifically a learned NNUE —
    closes the gap to GreedyBot. Still bounded via tanh so terminals dominate."""
    pot = crown_weight * (_claimed_crowns(state, 0) - _claimed_crowns(state, 1))
    return tanh((margin_p0(state) + pot) / SCORE_SCALE)


def outcome_p0(state: GameState) -> float:
    """Official match result at a terminal: +1 (P0), -1 (P1), 0 (true draw)."""
    winner = determine_winner(state)
    return 0.0 if winner is None else (1.0 if winner == 0 else -1.0)


def terminal_value_p0(state: GameState, margin_weight: float = 0.0) -> float:
    """Terminal utility in player-0 frame. Outcome-only by default.

    `margin_weight > 0` adds `margin_weight * tanh(margin)` — a deliberate
    outcome/margin BLEND (not a lexicographic tiebreak; see module docstring).
    Kept < 0.5 preserves win > draw > loss ordering among proven results."""
    v = outcome_p0(state)
    if margin_weight:
        v += margin_weight * tanh_margin_p0(state)
    return v


def _stable_seed(base: int, sorted_deck: list[int], drew: int) -> int:
    """Deterministic, cross-version-stable 64-bit seed from the public bag."""
    h = hashlib.blake2b(digest_size=8)
    h.update(int(base).to_bytes(8, "little", signed=True))
    h.update(int(drew).to_bytes(2, "little"))
    for d in sorted_deck:
        h.update(int(d).to_bytes(2, "little"))
    return int.from_bytes(h.digest(), "little")


def _replace_drawn_row(child: GameState, old_deck: list[int], row: tuple[int, ...]) -> GameState:
    """Install a specific public draw `row` as the child's current row, with the
    residual bag sorted (hidden order carries no information)."""
    out = child.copy()
    row_set = set(row)
    out.current_row = sorted(row)
    out.deck = sorted(d for d in old_deck if d not in row_set)
    return out


class ExpectiminimaxBot:
    """Depth-limited expectiminimax searcher usable as a bot_match player.

    Parameters
    ----------
    depth : decision plies to look ahead from the root.
    chance_samples : Monte-Carlo draws per WIDE chance node.
    enum_cap : enumerate a chance node exactly when its row count <= this
        (so C(8,4)=70 and C(4,4)=1 are exact by default; earlier rounds sampled).
    eval_fn : horizon evaluator, player-0 frame, ideally bounded in (-1, 1).
        Defaults to `tanh_margin_p0`.
    margin_weight : terminal outcome/margin blend (0 = pure expected outcome).
    seed : base seed for common-random-number draw sampling.
    """

    def __init__(
        self,
        depth: int = 4,
        chance_samples: int = 16,
        enum_cap: int = 128,
        eval_fn: Optional[Callable[[GameState], float]] = None,
        margin_weight: float = 0.0,
        seed: int = 0,
    ):
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if chance_samples < 1:
            raise ValueError(f"chance_samples must be >= 1, got {chance_samples}")
        if enum_cap < 1:
            raise ValueError(f"enum_cap must be >= 1, got {enum_cap}")
        self.depth = depth
        self.chance_samples = chance_samples
        self.enum_cap = enum_cap
        self.eval_fn = eval_fn or tanh_margin_p0
        self.margin_weight = margin_weight
        self.seed = seed
        self.nodes = 0  # instrumentation: nodes visited in the last choose_action

    # ---- chance expansion -------------------------------------------------

    def _expand_chance(self, state: GameState, action) -> list[tuple[GameState, float]]:
        """Apply `action`; if it revealed a new row from the deck, return the
        distribution over possible public draws. Otherwise a single certain child."""
        child = state.step(action)
        drew = len(state.deck) - len(child.deck)
        if drew <= 0:
            return [(child, 1.0)]

        # Sort the bag once: both enumeration and sampling MUST read a
        # canonical (deck-order-independent) population, or the search value
        # leaks hidden order — `random.sample` indexes into its population, so
        # a permuted deck would otherwise yield different draws for one seed.
        sorted_deck = sorted(state.deck)
        n_rows = comb(len(sorted_deck), drew)
        if n_rows <= self.enum_cap:
            p = 1.0 / n_rows
            return [
                (_replace_drawn_row(child, sorted_deck, row), p)
                for row in itertools.combinations(sorted_deck, drew)
            ]

        # Wide chance node: Monte-Carlo sample. Seed from the public deck multiset
        # so identical chance nodes draw identical futures (common random numbers).
        # Uses an EXPLICIT stable hash (not Python's tuple hash, which carries no
        # cross-version contract) so sampled draws are reproducible across machines
        # and Python upgrades — important if these values become training targets.
        local = random.Random(_stable_seed(self.seed, sorted_deck, drew))
        w = 1.0 / self.chance_samples
        return [
            (_replace_drawn_row(child, sorted_deck, tuple(sorted(local.sample(sorted_deck, drew)))), w)
            for _ in range(self.chance_samples)
        ]

    # ---- search -----------------------------------------------------------

    def _action_value(self, state: GameState, action, depth: int, alpha: float, beta: float) -> float:
        children = self._expand_chance(state, action)
        if len(children) == 1:
            # Deterministic transition: pass the window through to keep pruning.
            return self._value(children[0][0], depth - 1, alpha, beta)
        # Chance node: full expectation, each child a fresh window (Star1 deferred).
        expected = 0.0
        for child, p in children:
            expected += p * self._value(child, depth - 1, -inf, inf)
        return expected

    def _value(self, state: GameState, depth: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        if state.phase == Phase.GAME_OVER:
            return terminal_value_p0(state, self.margin_weight)
        if depth <= 0:
            return self.eval_fn(state)

        actions = state.legal_actions()
        actor = state.current_actor
        if actor == 0:  # maximize (player-0 frame)
            v = -inf
            for a in actions:
                v = max(v, self._action_value(state, a, depth, alpha, beta))
                alpha = max(alpha, v)
                if alpha >= beta:
                    break
            return v
        else:  # minimize
            v = inf
            for a in actions:
                v = min(v, self._action_value(state, a, depth, alpha, beta))
                beta = min(beta, v)
                if beta <= alpha:
                    break
            return v

    # ---- bot interface ----------------------------------------------------

    def choose_action(self, state, actions, rng=None):
        if len(actions) == 1:
            return actions[0]
        self.nodes = 0
        actor = state.current_actor
        best_score = None
        best_actions: list = []
        for action in actions:
            v = self._action_value(state, action, self.depth, -inf, inf)
            # Each actor maximizes its OWN outcome; v is player-0 frame.
            score = v if actor == 0 else -v
            if best_score is None or score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
        return (rng or random).choice(best_actions)
