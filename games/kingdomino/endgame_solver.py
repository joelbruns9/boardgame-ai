from __future__ import annotations

import itertools
import math
import random
from typing import Protocol

from games.kingdomino.game import GameState, Phase


_COUNT_CAP = 1_000_000_000


class _StateLike(Protocol):
    phase: Phase
    current_actor: int
    deck: list[int]

    def legal_actions(self) -> list[object]: ...
    def step(self, action: object) -> "_StateLike": ...


def _chance_draw_happens(parent: _StateLike, child: _StateLike) -> bool:
    return (
        parent.phase == Phase.PLACE_AND_SELECT
        and child.phase == Phase.PLACE_AND_SELECT
        and bool(getattr(parent, "deck", []))
        and getattr(child, "actor_index", None) == 0
    )


def _replace_drawn_row(child: _StateLike, old_deck: list[int], row: tuple[int, ...]) -> _StateLike:
    out = child.copy() if hasattr(child, "copy") else child
    row_set = set(row)
    out.current_row = sorted(row)
    out.deck = sorted(d for d in old_deck if d not in row_set)
    return out


def _step_public_futures(state: _StateLike, action: object) -> list[tuple[_StateLike, float]]:
    """Step one action, expanding future row draws as public bag combinations.

    The normal engine draws the next row from `deck[:4]`. That is correct for a
    concrete determinization, but exact public endgame search must not depend on
    hidden deck order. When an action completes a round and reveals a new row,
    this helper enumerates every equally likely row combination from the old
    hidden bag and installs a sorted residual bag for future chance nodes.
    """
    old_deck = list(getattr(state, "deck", []))
    child = state.step(action)
    if not _chance_draw_happens(state, child):
        return [(child, 1.0)]

    draw_n = min(4, len(old_deck))
    rows = list(itertools.combinations(sorted(old_deck), draw_n))
    if not rows:
        return [(child, 1.0)]
    p = 1.0 / len(rows)
    return [(_replace_drawn_row(child, old_deck, row), p) for row in rows]


def _count_bounded(state: _StateLike, cap: int) -> int:
    if state.phase == Phase.GAME_OVER:
        return 0

    # Exact counting from a large hidden bag is itself the thing we are trying
    # to avoid. Returning cap+1 is conservative and forces network fallback.
    if len(getattr(state, "deck", [])) > 8:
        return cap + 1

    total = 1
    for action in state.legal_actions():
        for child, _p in _step_public_futures(state, action):
            child_count = _count_bounded(child, cap - total)
            total += max(1, child_count)
            if total > cap:
                return total
    return total


def _rust_state_from_python(state: GameState):
    try:
        import kingdomino_rust
    except Exception:
        return None
    if not hasattr(state, "boards") or len(getattr(state, "boards", [])) != 2:
        return None
    try:
        b0, b1 = state.boards
        castle_x, castle_y = b0.castle_pos
        return kingdomino_rust.RustGameState.from_parts(
            list(state.deck),
            list(state.current_row),
            [(int(c.player), int(c.domino_id)) for c in state.pending_claims],
            [(int(c.player), int(c.domino_id)) for c in state.next_claims],
            int(state.phase),
            int(state.actor_index),
            int(state.initial_pick_count),
            int(state.start_player),
            b0.terrain.astype("uint8", copy=False).ravel().tolist(),
            b0.crowns.astype("uint8", copy=False).ravel().tolist(),
            b1.terrain.astype("uint8", copy=False).ravel().tolist(),
            b1.crowns.astype("uint8", copy=False).ravel().tolist(),
            bool(state.config.harmony),
            bool(state.config.middle_kingdom),
            int(castle_x),
            int(castle_y),
        )
    except Exception:
        return None


def _is_rust_no_chance_state(state: GameState) -> bool:
    if state.phase == Phase.GAME_OVER:
        return True
    if state.phase == Phase.PLACE_AND_SELECT:
        return len(state.deck) in (0, 4)
    if state.phase == Phase.FINAL_PLACEMENT:
        return len(state.deck) == 0
    return False


def _rust_count_no_chance(state: GameState, max_nodes: int) -> int | None:
    if not _is_rust_no_chance_state(state):
        return None
    rs = _rust_state_from_python(state)
    if rs is None:
        return None
    try:
        import kingdomino_rust
        return int(kingdomino_rust.count_endgame_nodes_no_chance(rs, int(max_nodes)))
    except Exception:
        return None


def _rust_exact_no_chance(
    state: GameState,
    *,
    max_nodes: int,
    score_scale: float,
    margin_gain: float,
    alpha: float,
) -> tuple[float, bool] | None:
    if not _is_rust_no_chance_state(state):
        return None
    rs = _rust_state_from_python(state)
    if rs is None:
        return None
    try:
        import kingdomino_rust
        value0, solved, _nodes = kingdomino_rust.exact_endgame_value_no_chance(
            rs,
            int(max_nodes),
            float(score_scale),
            float(margin_gain),
            float(alpha),
        )
        return float(value0), bool(solved)
    except Exception:
        return None


def count_endgame_nodes(state: GameState, max_nodes: int = _COUNT_CAP) -> int:
    """Conservatively estimate exhaustive public-consistent endgame tree size.

    Returns 0 for GAME_OVER. For large hidden bags the function returns a large
    conservative sentinel rather than trying to enumerate a tree that should not
    be solved exactly.
    """
    if state.phase == Phase.GAME_OVER:
        return 0
    rust_count = _rust_count_no_chance(state, max_nodes)
    if rust_count is not None:
        return rust_count
    return _count_bounded(state, int(max_nodes))


def exact_endgame_value(
    state: GameState,
    *,
    max_nodes: int = 50_000,
    rng: random.Random,
    score_scale: float,
    margin_gain: float,
    alpha: float,
) -> tuple[float, bool]:
    """Return (value_player0, solved_exactly) for a public endgame state.

    The solver is pure expectiminimax: no priors, visits, PUCT, or neural
    network values. It enumerates legal actions with state.legal_actions() and
    terminal leaves with terminal_search_value().

    Hidden-order safety: callers may pass a concrete open-loop determinization,
    but the solver treats `state.deck` as an unordered public bag. Future row
    reveals are averaged uniformly over all row combinations from that bag, so
    two states with the same public information and different deck order produce
    the same value. `rng` is accepted for API symmetry and future tie-breaking;
    the current exhaustive solver is deterministic and does not consume it.
    """
    del rng
    if max_nodes < 1:
        return 0.0, False

    from games.kingdomino.mcts_az import terminal_search_value

    if state.phase == Phase.GAME_OVER:
        return (
            terminal_search_value(
                state,
                player=0,
                score_scale=score_scale,
                margin_gain=margin_gain,
                alpha=alpha,
            ),
            True,
        )

    rust_result = _rust_exact_no_chance(
        state,
        max_nodes=max_nodes,
        score_scale=score_scale,
        margin_gain=margin_gain,
        alpha=alpha,
    )
    if rust_result is not None:
        return rust_result

    node_count = _count_bounded(state, max_nodes)
    if node_count > max_nodes:
        return 0.0, False

    def solve(s: _StateLike) -> float:
        if s.phase == Phase.GAME_OVER:
            return terminal_search_value(
                s,
                player=0,
                score_scale=score_scale,
                margin_gain=margin_gain,
                alpha=alpha,
            )

        values = []
        for action in s.legal_actions():
            expected = 0.0
            for child, p in _step_public_futures(s, action):
                expected += p * solve(child)
            values.append(expected)

        if not values:
            raise ValueError(f"Non-terminal state has no legal actions: phase={s.phase.name}")
        if s.current_actor == 0:
            return max(values)
        return min(values)

    value0 = float(solve(state))
    if not math.isfinite(value0):
        raise FloatingPointError(f"exact_endgame_value produced non-finite value {value0}")
    return value0, True
