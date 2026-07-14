"""Step 2: a make/unmake-driven expectiminimax over the Rust `SearchEngine`.

This is the Python-HOSTED search — the recursion lives in Python but walks a
single `SearchEngine` via `make`/`unmake` (and `make_with_row` for chance) rather
than cloning a state per node. It mirrors `expectiminimax.py`'s logic exactly
(same SCORE_SCALE, same eval formulas, same chance enumeration, same alpha-beta
on decision layers, outcome-only terminals) so that, with chance fully
enumerated, it returns byte-identical search VALUES to the pure-Python
`ExpectiminimaxBot`. See `test_rust_expectiminimax_equiv.py`.

Purpose: (a) validate the engine (`make_with_row`, `official_outcome`) under real
search recursion against the known-good reference; (b) an early nodes/s + feasible
-depth read before hosting the recursion in Rust (the perf destination). The
per-node FFI here caps throughput; that ceiling is the measurement, not a target.
"""
from __future__ import annotations

import itertools
import random
from math import comb, inf, tanh
from typing import Callable, Optional

import kingdomino_rust as kr

from games.kingdomino.expectiminimax import _stable_seed

SCORE_SCALE = 40.0  # must match expectiminimax.py

# Phase codes (mirror game.Phase / the Rust constants).
_INITIAL, _PLACE, _FINAL, _GAME_OVER = 0, 1, 2, 3


# ── leaf evaluators (read from the SearchEngine; player-0 frame) ─────────────

def _margin(eng) -> float:
    s0, s1 = eng.scores()
    return float(s0 - s1)


def tanh_margin(eng) -> float:
    """Bounded pick-blind margin proxy — the default 'dumb' horizon eval."""
    return tanh(_margin(eng) / SCORE_SCALE)


def _claimed_crowns(eng, player: int) -> int:
    """Crowns on `player`'s claimed-but-unplaced dominoes (mirrors
    expectiminimax._claimed_crowns, reading the engine's tuple claim lists)."""
    crowns = 0
    unplaced = list(eng.next_claims())
    if eng.phase != _GAME_OVER:
        pend = eng.pending_claims()
        unplaced += pend[eng.actor_index:]
    for pl, did in unplaced:
        if pl == player:
            _, ca, _, cb = kr.domino_halves(did)
            crowns += ca + cb
    return crowns


def pick_aware(eng, crown_weight: float = 4.0) -> float:
    """Control eval: margin plus a crude claimed-domino crown potential."""
    pot = crown_weight * (_claimed_crowns(eng, 0) - _claimed_crowns(eng, 1))
    return tanh((_margin(eng) + pot) / SCORE_SCALE)


def _terminal_value(eng, margin_weight: float) -> float:
    v = float(eng.official_outcome())
    if margin_weight:
        v += margin_weight * tanh_margin(eng)
    return v


class RustExpectiminimax:
    """Depth-limited expectiminimax over a Rust `SearchEngine` (make/unmake)."""

    def __init__(
        self,
        depth: int = 4,
        chance_samples: int = 16,
        enum_cap: int = 128,
        eval_fn: Optional[Callable] = None,
        margin_weight: float = 0.0,
        seed: int = 0,
    ):
        if depth < 1 or chance_samples < 1 or enum_cap < 1:
            raise ValueError("depth, chance_samples, enum_cap must all be >= 1")
        self.depth = depth
        self.chance_samples = chance_samples
        self.enum_cap = enum_cap
        self.eval_fn = eval_fn or tanh_margin
        self.margin_weight = margin_weight
        self.seed = seed
        self.nodes = 0

    # --- chance prediction & expansion ------------------------------------

    @staticmethod
    def _deals(eng) -> bool:
        """Does the next applied action reveal a new row (a chance node)?"""
        ph = eng.phase
        if ph == _INITIAL:
            return eng.initial_pick_count == 3  # the 4th pick deals the first row
        if ph == _PLACE:
            return len(eng.deck()) >= 4 and eng.actor_index == len(eng.pending_claims()) - 1
        return False  # FINAL_PLACEMENT never deals

    def _chance_rows(self, eng):
        """(row, weight) pairs for the pending deal: enumerated when
        C(n,4) <= enum_cap, else Monte-Carlo sampled.  Sampling uses the SAME
        blake2 stable seed and RNG-draw sequence as expectiminimax._expand_chance
        (drew=4), so wide chance nodes produce byte-identical sampled rows — the
        port mirrors the reference for sampled search too, not just enumerated."""
        deck = sorted(eng.deck())
        n_rows = comb(len(deck), 4)
        if n_rows <= self.enum_cap:
            p = 1.0 / n_rows
            return [(list(r), p) for r in itertools.combinations(deck, 4)]
        rng = random.Random(_stable_seed(self.seed, deck, 4))
        w = 1.0 / self.chance_samples
        return [(sorted(rng.sample(deck, 4)), w) for _ in range(self.chance_samples)]

    # --- search ------------------------------------------------------------

    def _action_value(self, eng, action, depth, alpha, beta) -> float:
        p, pk = action
        # try/finally so a raising eval_fn (caller-supplied) or any error unwinds
        # the shared engine's undo stack rather than leaving it mutated — the
        # SearchEngine is reused across value()/choose_action() calls.
        if not self._deals(eng):
            eng.make(p, pk)
            try:
                return self._value(eng, depth - 1, alpha, beta)
            finally:
                eng.unmake()
        # Chance node: probability-weighted average over drawn rows, each a fresh
        # window (Star1/Star2 pruning deferred).
        expected = 0.0
        for row, w in self._chance_rows(eng):
            eng.make_with_row(p, pk, row)
            try:
                expected += w * self._value(eng, depth - 1, -inf, inf)
            finally:
                eng.unmake()
        return expected

    def _value(self, eng, depth, alpha, beta) -> float:
        self.nodes += 1
        if eng.phase == _GAME_OVER:
            return _terminal_value(eng, self.margin_weight)
        if depth <= 0:
            return self.eval_fn(eng)

        actor = eng.current_actor()
        actions = eng.legal_actions()
        if actor == 0:  # maximize (player-0 frame)
            v = -inf
            for a in actions:
                v = max(v, self._action_value(eng, a, depth, alpha, beta))
                alpha = max(alpha, v)
                if alpha >= beta:
                    break
            return v
        else:  # minimize
            v = inf
            for a in actions:
                v = min(v, self._action_value(eng, a, depth, alpha, beta))
                beta = min(beta, v)
                if beta <= alpha:
                    break
            return v

    # --- entry points ------------------------------------------------------

    def value(self, eng, depth=None) -> float:
        """Root player-0-frame value of `eng` at `depth` (default self.depth)."""
        return self._value(eng, self.depth if depth is None else depth, -inf, inf)

    def choose_action(self, state, actions=None, rng=None):
        """Pick the best action for the side to move. `state` may be a
        RustGameState (wrapped fresh) or a SearchEngine (walked in place)."""
        eng = state if isinstance(state, kr.SearchEngine) else kr.SearchEngine(state)
        acts = actions if actions is not None else eng.legal_actions()
        if len(acts) == 1:
            return acts[0]
        self.nodes = 0
        actor = eng.current_actor()
        best_score, best_actions = None, []
        for a in acts:
            v = self._action_value(eng, a, self.depth, -inf, inf)
            score = v if actor == 0 else -v
            if best_score is None or score > best_score:
                best_score, best_actions = score, [a]
            elif score == best_score:
                best_actions.append(a)
        return (rng or random).choice(best_actions)


class OperationalRustSearchBot:
    """Bot adapter for RustSearch's deadline-safe operational path.

    The Rust search owns the clock, iterative deepening, TT/PV ordering, chance
    pruning, and exact deterministic-tail extension. The adapter only converts
    the public Python GameState/action objects at the bot_match boundary.
    """

    def __init__(
        self,
        *,
        max_secs: float = 1.0,
        max_depth: int = 8,
        chance_samples: int = 16,
        enum_cap: int = 128,
        eval: str = "pick_aware",
        nnue_path: str | None = None,
        margin_weight: float = 0.0,
        seed: int = 0,
        aspiration_window: float = 0.25,
        selective_width: int | None = None,
        selective_root_width: int | None = None,
        selective_min_depth: int = 4,
    ):
        if (max_secs <= 0 or max_depth < 1 or aspiration_window <= 0
                or (selective_width is not None and selective_width < 1)
                or (selective_root_width is not None and selective_root_width < 1)
                or (selective_root_width is not None and selective_width is None)
                or selective_min_depth < 1):
            raise ValueError("search limits and optional selective width must be positive")
        self.max_secs = float(max_secs)
        self.max_depth = int(max_depth)
        self.aspiration_window = float(aspiration_window)
        self.selective_width = selective_width
        self.selective_root_width = selective_root_width
        self.selective_min_depth = int(selective_min_depth)
        self.search = kr.RustSearch(
            depth=self.max_depth,
            chance_samples=chance_samples,
            enum_cap=enum_cap,
            eval=eval,
            margin_weight=margin_weight,
            seed=seed,
            nnue_path=nnue_path,
        )
        self.last_report = None
        self.nodes = 0

    @staticmethod
    def _placement_key(placement, domino_id):
        """Physical placement identity, independent of endpoint/flip spelling."""
        from games.kingdomino.dominoes import DOMINOES

        if placement is None:
            return None
        x1, y1, x2, y2, flipped = placement
        domino = DOMINOES[domino_id]
        h1, h2 = (domino.b, domino.a) if flipped else (domino.a, domino.b)
        cells = (
            (x1, y1, int(h1.terrain), int(h1.crowns)),
            (x2, y2, int(h2.terrain), int(h2.crowns)),
        )
        return tuple(sorted(cells))

    def choose_action(self, state, actions=None, rng=None):
        from games.kingdomino.endgame_solver import _rust_state_from_python

        rust_state = _rust_state_from_python(state)
        if rust_state is None:
            raise RuntimeError("could not convert GameState to RustGameState")
        report = self.search.choose_action_timed(
            rust_state,
            max_secs=self.max_secs,
            max_depth=self.max_depth,
            aspiration_window=self.aspiration_window,
            selective_width=self.selective_width,
            selective_root_width=self.selective_root_width,
            selective_min_depth=self.selective_min_depth,
        )
        self.last_report = report
        self.nodes = int(report.nodes)
        legal = state.legal_actions() if actions is None else actions
        rust_placement, rust_pick = report.action
        if state.phase == _INITIAL:
            for action in legal:
                if getattr(action, "domino_id", None) == rust_pick:
                    return action
        else:
            domino_id = state.pending_claims[state.actor_index].domino_id
            rust_key = self._placement_key(rust_placement, domino_id)
            for action in legal:
                placement = action.placement
                py_tuple = None if placement is None else (
                    placement.x1,
                    placement.y1,
                    placement.x2,
                    placement.y2,
                    placement.flipped,
                )
                if (
                    action.pick_domino_id == rust_pick
                    and self._placement_key(py_tuple, domino_id) == rust_key
                ):
                    return action
        raise RuntimeError(f"Rust operational search returned illegal action {report.action!r}")
