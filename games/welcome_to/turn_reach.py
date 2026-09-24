"""
STATUS: UNWIRED HERE, AND ON THE WRONG BRANCH'S STEP 3.  Written 2026-09-14 on
`sevenwd-w9-prototype`, committed here 2026-09-24 to preserve it.  Nothing
imports it, and step 4 on THIS branch is still the `# Encoder v3:` block in
`game.py` -- which is what `plans.py` imports and `tests/test_plan_threat.py`
gates.  This module changes no behaviour.

⚠ IT DOES NOT IMPORT ON THIS BRANCH.  It needs
`deck_knowledge.ordered_draw_distribution` and
`deck_knowledge.boundary_pool_composition`, which exist only on
`sevenwd-w9-prototype` at commit 9d4cf9b ("encoder v3 step 3, exact
boundary-draw deck features").  Verified 2026-09-24: with 9d4cf9b's
`deck_knowledge.py` it imports and runs.

THERE ARE TWO COMPETING STEP 3+4 BUILDS, and this is half of the younger one:

  welcome-to-engine  5a73fe6 (Sep 1)   step 3 = number_prefix_sums,
                                       effect_supply_rate
                                       step 4 = game.py block + test_plan_threat
  w9-prototype       9d4cf9b (Sep 14)  step 3 = the above PLUS
                                       count_in_open_interval,
                                       boundary_pool_composition,
                                       ordered_draw_distribution
                                       step 4 = this file, untested

9d4cf9b branched from 4e65beb, not from 5a73fe6, so the two lines diverge.
Reconciling them is a pending decision, not a mechanical merge.

Why this rewrite is worth keeping -- three things `game.py`'s block lacks:

* **mid-turn staging.**  `game.py` has no phase logic and always enumerates a
  fresh whole turn from CHOOSE_CARDS, so it over-reports for the viewer who has
  already written.  `_stage_for` below walks only the rest of the turn.
* **the plan loop.**  `_plan_loop_can_validate` walks the real engine, because a
  half-validated plan's houses are not consumed yet and reading the sheet
  over-reports.  `game.py` has no equivalent.
* **roundabout legality.**  This honours `ctx.roundabout_declined` and
  `ctx.last_house is None`; `game.py` offers one whenever it is buildable.

And the exact next-turn draw: `game.py` enumerates 3,375 ordered triples and
returns `1 - miss/total`, renormalising over the triples it did not zero.  This
delegates to `ordered_draw_distribution`, which is what §10.2a actually asks for.

To finish it:

1. Get 9d4cf9b's `deck_knowledge.py` onto this branch.
2. `tests/test_turn_reach.py` -- the bidirectional walk against the engine's own
   `legal_actions` that the docstring below promises (§10.2).  Never written.
   `tests/engine_turn_oracle.py` already does this for `game.one_turn_sheets`
   and is the model to copy.
3. A differential test against `game.py`'s block, stating the three divergences
   above as EXPECTED, not as equality.

Dropped from `game.py` without justifying the drop:

* `_estate_houses_short`, the cheap estate pre-filter.  `game.py` keeps it
  because a near-empty sheet otherwise walks ~32,000 sheets and the sparse case
  is always a "no".  The content cache and first-success break below do not
  cover a NEGATIVE answer on a sparse sheet.  ⚠ MEASURED 2026-09-24 on w9:
  **3.45 s** for 2 seats x 3 slots at turn 6 of a 2p advanced game.  That is the
  predicted regression, and it is real.  Restore the pre-filter or justify it.
* `bis_usable`.  `tests/test_plan_threat.py:274` asserts on it.

One-turn reachability: what a seat could still do with this turn, or the next.

``ENCODER_V3_SPEC.md`` §6.4 and §8, build step 4.  Four facts per seat:

* :func:`one_turn_ceiling` -- the most one turn can advance a plan's ``steps_left``;
* :func:`can_complete_this_turn` -- could this seat score this plan this turn?
* :func:`p_complete_next_turn` -- the probability it could next turn, over the
  exact joint draw of next turn's three numbers;
* :func:`max_houses_this_turn` -- the most houses this turn can still place (0-3).

These answer "could they", never "did they" or "will they".  Whether an opponent
has already acted this turn is hidden, and correctly so.

WHOSE TURN IS BEING ASKED ABOUT
───────────────────────────────
Every seat is read through the viewer's information set:

* an **opponent** is asked about a fresh turn -- ``CHOOSE_CARDS``, nothing chosen --
  on their public sheet, facing the shared offer;
* the **viewer, on their own turn**, is asked about the *rest* of it, from the
  live phase and scratch state;
* the viewer after their own turn has finished can place nothing more.

At a turn boundary the first two coincide (live sheet = public snapshot, fresh
scratch state), which is what keeps the per-seat block symmetric (§10.5).

HOW THE TURN IS ENUMERATED
──────────────────────────
A turn is ``[roundabout] -> choose + write -> effect -> plans``, and no chance
occurs inside it.  :func:`_final_sheets` walks that chain on bare :class:`Sheet`
objects, replaying ``GameState._dispatch`` move for move.  It is not the engine,
so ``tests/test_turn_reach.py`` checks it against a brute-force walk of the
engine's own ``legal_actions`` -- in both directions (§10.2).

Two things are deliberately not walked:

* **the park, pool and estate passes.**  They are dominated (``SEARCH_SPEC.md``
  §5.1): the build consumes nothing, and every plan predicate reading parks or
  pools is monotone.  Estate-row marks are read by no plan at all.
* **the plan loop itself.**  Validating one plan only adds top fences, which can
  only *remove* scorable plans, so "some sequence scores plan *k*" is exactly
  "``can_be_scored`` holds when the plan phase is first entered".  A viewer
  already inside that loop is walked on the real engine instead (it is tiny).

The permit-refusal path is walked, because it leaves the sheet as it stands.

⚠ **Houses, not numbers** (§6.1).  A roundabout and a bis each put a house on a
box no drawn number can reach, and a fence makes new estates without a house.
All four are here because every one of them broke a feasibility rule in review.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from games.welcome_to import deck_knowledge as dk
from games.welcome_to.constants import (
    BIS_BOXES,
    CARD_NUMBERS,
    DECK_EFFECT_ORDER,
    MAX_NUMBER,
    MIN_NUMBER,
    NUMBER_INDEX,
    TEMP_BOXES,
    TEMP_DELTAS,
    Effect,
)
from games.welcome_to.game import GameState, Phase
from games.welcome_to.plans import PLANS, Plan, PlanKind, can_be_scored, progress
from games.welcome_to.sheet import Pos, Sheet

Combo = tuple[int, Effect]

#: §6.4 R1/R4, in ``progress()`` units.  ESTATE is filled in per plan: a single
#: fence can raise the estate match by two and a turn can supply three fences, so
#: its ceiling is ``len(required_sizes)`` -- "never exit early".
_CEILING: dict[PlanKind, int] = {
    PlanKind.FULL_STREET: 3,       # roundabout + write + bis
    PlanKind.EXTREMITIES: 3,       # same; a roundabout fills an extremity box
    PlanKind.FIVE_BIS: 1,          # one BIS mark per combination
    PlanKind.SEVEN_TEMP: 1,        # one TEMP mark per combination
    PlanKind.DECORATIVE: 1,        # one PARK or POOL mark per combination
    PlanKind.COMPLETE_STREET: 2,   # one mark, plus a roundabout in the same turn
}

#: A walk this large means something is wrong with the walk, not the sheet: the
#: measured maximum is a few tens of thousands of whole-turn sequences.  It
#: raises rather than answering ``False`` (§10.2 -- the cap must never bind).
MAX_FINAL_SHEETS: int = 1_000_000


class ReachExhausted(RuntimeError):
    """The enumeration cap bound.  Never a verdict."""


def one_turn_ceiling(plan: Plan) -> int:
    """The most one turn can reduce ``progress(plan, sheet)[1]``.  A bound, not a prediction."""
    if plan.kind is PlanKind.ESTATE:
        return len(plan.required_sizes)
    try:
        return _CEILING[plan.kind]
    except KeyError:
        raise NotImplementedError(f"plan kind {plan.kind} has no one-turn ceiling") from None


# ──────────────────────────────────────────────────────────────────────────
# The rest of a turn, as a sheet-level description
# ──────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _Stage:
    """Where in the turn the walk starts.  Exactly one of the three shapes is used.

    * ``combos`` set, ``chosen`` and ``pending`` unset -- at ``CHOOSE_CARDS``,
      with ``roundabout_ok`` saying whether a roundabout may still precede it;
    * ``chosen`` set -- at ``WRITE_NUMBER`` with that combination taken;
    * ``pending`` set -- the house is written and ``(effect, last_house)`` is owed.

    ``done`` means the sheet is final for this turn.
    """

    combos: tuple[Combo, ...] = ()
    roundabout_ok: bool = False
    chosen: Optional[Combo] = None
    pending: Optional[tuple[Effect, Pos]] = None
    done: bool = False


def _numbers_for(number: int, effect: Effect) -> list[int]:
    """``GameState.numbers_for``, which reads no state."""
    if effect is not Effect.TEMP:
        return [number]
    return [number] + [
        number + d for d in TEMP_DELTAS[1:] if MIN_NUMBER <= number + d <= MAX_NUMBER
    ]


def _playable(sheet: Sheet, combo: Combo) -> bool:
    return any(sheet.available_locations(n) for n in _numbers_for(*combo))


def _final_sheets(sheet: Sheet, stage: _Stage, advanced: bool) -> Iterator[Sheet]:
    """Every sheet this turn can end its effect phase on.  ``sheet`` is not mutated.

    Mirrors ``GameState.legal_actions`` / ``_dispatch`` phase by phase.  May yield
    the same sheet more than once; callers stop at the first sheet they need.
    """
    if stage.done:
        yield sheet
        return
    if stage.pending is not None:
        yield from _after_effect(sheet, *stage.pending)
        return
    if stage.chosen is not None:
        yield from _after_choice(sheet, stage.chosen)
        return

    yield from _after_cards(sheet, stage.combos)
    if (
        stage.roundabout_ok
        and advanced
        and sheet.can_build_roundabout()
        and sheet.has_free_box()
    ):
        for pos in sheet.available_locations(None):
            built = sheet.copy()
            built.build_roundabout(pos, turn=0)
            yield from _after_cards(built, stage.combos)


def _after_cards(sheet: Sheet, combos: tuple[Combo, ...]) -> Iterator[Sheet]:
    """``CHOOSE_CARDS`` with no roundabout left to open."""
    playable = [c for c in combos if _playable(sheet, c)]
    if not playable:
        if sheet.can_take_permit():
            yield sheet                      # the direct refusal
        return
    for combo in playable:
        yield from _after_choice(sheet, combo)


def _after_choice(sheet: Sheet, combo: Combo) -> Iterator[Sheet]:
    """``WRITE_NUMBER``: every legal write, and the refusal ``argWriteNumber`` keeps open."""
    number, effect = combo
    for n in _numbers_for(number, effect):
        for pos in sheet.available_locations(n):
            written = sheet.copy()
            written.write(n, pos, turn=0)
            if effect is Effect.TEMP:
                written.temps = min(written.temps + 1, TEMP_BOXES)
            yield from _after_effect(written, effect, pos)
    if not sheet.available_locations(number) and sheet.can_take_permit():
        yield sheet


def _after_effect(sheet: Sheet, effect: Effect, last: Pos) -> Iterator[Sheet]:
    """The effect phase.  The park, pool and estate passes are dominated and skipped."""
    if effect is Effect.SURVEYOR:
        yield sheet
        for x, j in sheet.surveyor_zones():
            fenced = sheet.copy()
            fenced.fences[x][j] = True
            yield fenced
    elif effect is Effect.BIS:
        yield sheet
        for x, y, number, _side in sheet.bis_candidates():
            bis = sheet.copy()
            bis.write(number, (x, y), turn=0, is_bis=True)
            bis.bis_marks = min(bis.bis_marks + 1, BIS_BOXES)
            yield bis
    elif effect is Effect.PARK and last[0] in sheet.park_streets():
        parked = sheet.copy()
        parked.parks[last[0]] += 1
        yield parked
    elif effect is Effect.POOL and sheet.can_build_pool_at(last):
        pooled = sheet.copy()
        pooled.pools[last[0]] += 1
        yield pooled
    else:
        # TEMP was marked at the write; ESTATE marks are read by no plan; a park
        # or pool that cannot be built resolves itself (``_settle``).
        yield sheet


# ──────────────────────────────────────────────────────────────────────────
# Reading a seat through the viewer's information set
# ──────────────────────────────────────────────────────────────────────────
_MID_EFFECT = {
    Phase.ACTION_SURVEYOR,
    Phase.ACTION_ESTATE,
    Phase.ACTION_PARK,
    Phase.ACTION_POOL,
    Phase.ACTION_BIS,
}
_PLAN_LOOP = {Phase.CHOOSE_PLAN, Phase.VALIDATE_PLAN, Phase.ASK_RESHUFFLE}


def _require_scope(state: GameState) -> None:
    if not state.config.standard or state.config.players < 2:
        raise ValueError(
            "turn reachability is defined for the 2+ player standard game only"
        )
    if state.boundary_prepared or state.is_terminal:
        raise ValueError("turn reachability reads a mid-turn state")


def _offer(state: GameState, viewer: int) -> tuple[Combo, ...]:
    return tuple(state.combination(slot, viewer) for slot in range(3))


def _stage_for(state: GameState, viewer: int, seat: int) -> _Stage:
    """The rest of ``seat``'s turn, as far as ``viewer`` may know it."""
    fresh = _Stage(combos=_offer(state, viewer), roundabout_ok=True)
    if seat != viewer:
        return fresh
    if seat != state.actor:
        return _Stage(done=True) if seat < state.actor else fresh
    ctx = state.ctx
    phase = state.phase
    if phase is Phase.CHOOSE_CARDS:
        return _Stage(
            combos=fresh.combos,
            roundabout_ok=ctx.last_house is None and not ctx.roundabout_declined,
        )
    if phase is Phase.ROUNDABOUT_PLACE:
        return _Stage(combos=fresh.combos, roundabout_ok=True)
    if phase is Phase.WRITE_NUMBER:
        assert ctx.number is not None and ctx.effect is not None
        return _Stage(chosen=(ctx.number, ctx.effect))
    if phase in _MID_EFFECT:
        assert ctx.effect is not None and ctx.last_house is not None
        return _Stage(pending=(ctx.effect, ctx.last_house))
    if phase in _PLAN_LOOP:
        return _Stage(done=True)
    raise AssertionError(f"unhandled phase {phase}")


def _open_slots(state: GameState, viewer: int, seat: int) -> list[int]:
    """Plan slots this seat has not banked, as the viewer can see (§6.4 R4)."""
    return [s for s in range(3) if seat not in state.plan_turns_for(viewer, s)]


# ──────────────────────────────────────────────────────────────────────────
# The enumeration, cached on content -- never on a seat
# ──────────────────────────────────────────────────────────────────────────
def _sheet_key(sheet: Sheet) -> tuple:
    return (
        tuple(map(tuple, sheet.numbers)),
        tuple(map(tuple, sheet.is_bis)),
        tuple(map(tuple, sheet.fences)),
        tuple(map(tuple, sheet.top_fences)),
        tuple(sheet.parks),
        tuple(sheet.pools),
        sheet.temps,
        sheet.bis_marks,
        sheet.permits,
        sheet.roundabouts,
    )


_CACHE: dict[tuple, frozenset[int]] = {}
_CACHE_LIMIT = 200_000


def _completable(
    sheet: Sheet, stage: _Stage, advanced: bool, plan_ids: tuple[int, ...]
) -> frozenset[int]:
    """Which of ``plan_ids`` some sequence of this turn leaves scorable.

    The key is the full sheet, the stage and the plans asked about: two seats
    with identical sheets facing an identical offer get the same answer, and no
    answer is ever filed under a seat or a viewer (§6.4's cache-key warning).
    Plans whose ``steps_left`` exceeds the one-turn ceiling are answered ``False``
    without walking (§6.4).
    """
    key = (_sheet_key(sheet), stage, advanced, plan_ids)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    wanted = [
        pid
        for pid in plan_ids
        if progress(PLANS[pid], sheet)[1] <= one_turn_ceiling(PLANS[pid])
    ]
    found: set[int] = set()
    if wanted:
        for count, final in enumerate(_final_sheets(sheet, stage, advanced)):
            if count >= MAX_FINAL_SHEETS:
                raise ReachExhausted(
                    f"more than {MAX_FINAL_SHEETS} end-of-turn sheets; the walk "
                    "is wrong, not the sheet"
                )
            for pid in wanted:
                if pid not in found and can_be_scored(PLANS[pid], final):
                    found.add(pid)
            if len(found) == len(wanted):
                break

    result = frozenset(found)
    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.clear()
    _CACHE[key] = result
    return result


def clear_cache() -> None:
    _CACHE.clear()


# ──────────────────────────────────────────────────────────────────────────
# Public predicates
# ──────────────────────────────────────────────────────────────────────────
def can_complete_this_turn(state: GameState, viewer: int, seat: int, slot: int) -> bool:
    """Could ``seat`` score plan ``slot`` this turn, as far as ``viewer`` can tell?

    ``False`` for a seat that has already banked the slot, before anything is
    walked: ``can_be_scored`` tests only the sheet, so a non-consuming plan like
    ``SEVEN_TEMP`` would otherwise read as a permanent threat (§6.4 R4).

    "Could they", not "did they": whether an opponent has acted this turn is
    hidden.
    """
    _require_scope(state)
    if seat in state.plan_turns_for(viewer, slot):
        return False
    if seat == viewer == state.actor and state.phase in _PLAN_LOOP:
        return _plan_loop_can_validate(state, slot)
    sheet = state.sheet_for(viewer, seat)
    pid = state.plan_ids[slot]
    stage = _stage_for(state, viewer, seat)
    if stage.done:
        return False
    return pid in _completable(sheet, stage, state.config.advanced, (pid,))


def _plan_loop_can_validate(state: GameState, slot: int) -> bool:
    """The viewer is already in the plan loop: walk the real engine to its end.

    Another plan may be half-validated, and its houses are not consumed until it
    finishes, so reading the sheet here could over-report.  The loop is a
    handful of choices, so the engine is cheap and exact.
    """
    actor = state.actor
    turn = state.turn
    stack = [state]
    while stack:
        current = stack.pop()
        for action in current.legal_actions():
            nxt = current.step(action)
            if nxt.plan_turns[slot].get(actor) == turn:
                return True
            if nxt.actor == actor and nxt.phase in _PLAN_LOOP and not nxt.boundary_prepared:
                stack.append(nxt)
    return False


def p_complete_next_turn(state: GameState, viewer: int, seat: int, slot: int) -> float:
    """P(some stack next turn lets ``seat`` complete plan ``slot``), exactly.

    Next turn's effects are printed and known, so only the three numbers are
    chance (§6.4 R2).  For each stack *i*:
    ``S_i = { n : a one-turn walk from combination (n, e_i) completes the plan }``,
    and ``P(none) = Σ joint[n1,n2,n3] · Π [n_i ∉ S_i]`` over
    :func:`deck_knowledge.ordered_draw_distribution` -- falling factorials, and the
    mid-draw reform when fewer than three cards remain.

    The sheet is the seat's sheet as the viewer sees it now; a roundabout may open
    the turn.  ``0.0`` for a slot already banked (§6.4 R4).

    ⚠ **The viewer's own yes vote to reshuffle** means next turn's effects are
    redrawn and unknown (§6.4 R5).  Then each stack's effect comes from one card
    of the reformed pool and its number from another, and this treats the ordered
    effect triple and the ordered number triple as independent draws from that
    pool's two marginals.  **That is the one approximation in this module**, and
    it applies only after the viewer has voted yes, at most once a game.
    """
    _require_scope(state)
    if seat in state.plan_turns_for(viewer, slot):
        return 0.0
    pid = state.plan_ids[slot]
    sheet = state.sheet_for(viewer, seat)
    advanced = state.config.advanced

    def supplies(effect: Effect) -> np.ndarray:
        hit = np.zeros(dk.NUM_NUMBERS, dtype=bool)
        for n in CARD_NUMBERS:
            stage = _Stage(combos=((n, effect),), roundabout_ok=True)
            hit[NUMBER_INDEX[n]] = pid in _completable(sheet, stage, advanced, (pid,))
        return hit

    if state.reshuffle_vote_for(viewer):
        pool = dk.after_reshuffle_composition(state, viewer)
        numbers = dk.ordered_draw_distribution(pool.sum(axis=1), np.zeros(dk.NUM_NUMBERS))
        effects = dk.ordered_draw_distribution(pool.sum(axis=0), np.zeros(dk.NUM_EFFECTS))
        miss = ~np.stack([supplies(e) for e in DECK_EFFECT_ORDER])      # (6, 15)
        none = 0.0
        for idx in zip(*np.nonzero(effects)):
            e1, e2, e3 = idx
            mask = np.multiply.outer(np.multiply.outer(miss[e1], miss[e2]), miss[e3])
            none += effects[idx] * float((numbers * mask).sum())
        return float(np.clip(1.0 - none, 0.0, 1.0))

    effects = state.next_effects(viewer)
    miss = [~supplies(e) for e in effects]
    joint = dk.ordered_draw_distribution(
        dk.deck_composition(state, viewer).sum(axis=1),
        dk.boundary_pool_composition(state, viewer).sum(axis=1),
    )
    mask = np.multiply.outer(np.multiply.outer(miss[0], miss[1]), miss[2])
    return float(np.clip(1.0 - (joint * mask).sum(), 0.0, 1.0))


def max_houses_this_turn(state: GameState, viewer: int, seat: int) -> int:
    """The most houses the rest of this turn can place, 0-3 (§8).

    ``roundabout -> choose + write -> bis`` is three.  Counted over the currently
    legal sequences, so a roundabout that would block the only playable box is
    counted as the trade it is, not as a free extra house.  How many are actually
    placed is behaviour; this is the ceiling.
    """
    _require_scope(state)
    stage = _stage_for(state, viewer, seat)
    sheet = state.sheet_for(viewer, seat)
    if stage.done:
        return 0
    if stage.pending is not None:
        effect, _ = stage.pending
        return int(effect is Effect.BIS and bool(sheet.bis_candidates()))
    if stage.chosen is not None:
        return _houses_after_choice(sheet, stage.chosen)

    best = _houses_after_cards(sheet, stage.combos)
    if (
        best < 3
        and stage.roundabout_ok
        and state.config.advanced
        and sheet.can_build_roundabout()
        and sheet.has_free_box()
    ):
        for pos in sheet.available_locations(None):
            built = sheet.copy()
            built.build_roundabout(pos, turn=0)
            best = max(best, 1 + _houses_after_cards(built, stage.combos))
            if best == 3:
                break
    return best


def _houses_after_cards(sheet: Sheet, combos: tuple[Combo, ...]) -> int:
    best = 0
    for combo in combos:
        best = max(best, _houses_after_choice(sheet, combo))
        if best == 2:
            break
    return best


def _houses_after_choice(sheet: Sheet, combo: Combo) -> int:
    number, effect = combo
    best = 0
    for n in _numbers_for(number, effect):
        for pos in sheet.available_locations(n):
            if effect is not Effect.BIS:
                return 1
            written = sheet.copy()
            written.write(n, pos, turn=0)
            if written.bis_candidates():
                return 2
            best = 1
    return best
