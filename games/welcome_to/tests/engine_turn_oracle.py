"""Test-only differential oracle: one turn, driven through the REAL engine.

`game.one_turn_sheets` is a hand-written model of what a single turn can do, and
`can_complete_this_turn` is asserted against it.  That assertion proves
*consistency*, not correctness: if the model omits a legal turn shape, or admits
an illegal one, both sides of the test agree on the wrong answer and it passes.

That is not hypothetical.  The model let a PARK mark land in any street, while
`Actions/Park` binds it to the street of the house just written -- so it built
illegal sheets and could report a completion the engine cannot reach.  The
bidirectional test was green throughout.

This module closes that hole by walking `legal_actions()` / `step()` from a real
`GameState`, so the reference comes from the engine rather than from a second
reading of the rules.

WHERE A TURN ENDS
─────────────────
Collection stops at ``CHOOSE_PLAN`` -- after the write and its effect, before any
plan validation.  That matches what `one_turn_sheets` models: validation is what
the *caller* then tests with `can_be_scored`, and letting the walk run through it
would fold the answer into the question.
"""
from __future__ import annotations

from games.welcome_to.game import GameState, Phase
from games.welcome_to.sheet import Sheet


class EngineWalkExhausted(RuntimeError):
    """The walk hit its cap.  Never a verdict -- see `plan_reachability`."""


def engine_one_turn_sheets(
    state: GameState, seat: int, *, cap: int = 300_000
) -> list[Sheet]:
    """Every sheet ``seat`` could hold at the end of its current turn.

    ``state`` must have ``seat`` to act.  The returned sheets are snapshots taken
    the moment the turn's decisions are spent, deduplicated.
    """
    if state.actor != seat:
        raise ValueError(f"seat {seat} is not the actor ({state.actor})")

    start_turn = state.turn
    seen: set[tuple] = set()
    out: list[Sheet] = []
    visited = 0

    def key(sheet: Sheet) -> tuple:
        return (
            tuple(tuple(r) for r in sheet.numbers),
            tuple(tuple(r) for r in sheet.is_bis),
            tuple(tuple(r) for r in sheet.fences),
            tuple(sheet.parks),
            tuple(sheet.pools),
            sheet.temps,
            sheet.bis_marks,
            sheet.roundabouts,
        )

    def record(sheet: Sheet) -> None:
        k = key(sheet)
        if k not in seen:
            seen.add(k)
            out.append(sheet.copy())

    def walk(current: GameState) -> None:
        nonlocal visited
        visited += 1
        if visited > cap:
            raise EngineWalkExhausted(f"engine walk exceeded {cap} states")

        turn_over = (
            current.actor != seat
            or current.turn != start_turn
            or current.is_terminal
            or current.phase is Phase.CHOOSE_PLAN
        )
        if turn_over:
            record(current.sheets[seat])
            return

        for action in current.legal_actions():
            walk(current.step(action))

    # ⚠ No unconditional record of the starting sheet: an unchanged sheet is a
    # turn OUTCOME only when a permit refusal was forced, and the walk reaches
    # that on its own.  Seeding it would make the reference generous in exactly
    # the direction this oracle exists to police.
    walk(state)
    return out
