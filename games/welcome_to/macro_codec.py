"""
The macro action vocabulary -- 684 indices, frozen in ``ENCODER_V2_SPEC.md`` §10.6.

WHY A MACRO AT ALL
──────────────────
The primitive codec splits one decision into two: ``CHOOSE_STACK`` picks a
combination, then ``WRITE`` places it.  A correct deterministic tree can model
that split without a strength error: both nodes belong to the same player and,
with enough visits, UCT concentrates on the best continuation rather than
averaging the placements.  The macro is still the useful representation because
**you pick a combination *for* a placement**: it matches the semantic action,
shortens the finite-budget horizon, improves credit assignment, and removes the
otherwise mandatory ``WRITE_NUMBER`` network evaluation.  Measured in
``SEARCH_SPEC.md`` §3, that removes 28% of network evaluations.

So the whole ``CHOOSE_CARDS -> WRITE_NUMBER`` segment is one action::

    macro write     495   (slot, temp delta, box) = 3 x 5 x 33     at CHOOSE_CARDS
    macro refuse      3   (slot, PERMIT_REFUSAL)                   at CHOOSE_CARDS
    direct refuse     1   A_PERMIT_REFUSAL, no slot playable       at CHOOSE_CARDS
    roundabout open   1   A_ROUNDABOUT_OPEN                        at CHOOSE_CARDS
    primitives      184   the 357-slot codec minus the four above  at their phases
                    ---
                    684

**There are no ``WRITE_NUMBER`` network evaluations.** The macro applies both
engine steps, and that phase is where the branching lives (13.1 mean, 165 max),
so this is also the main saving.

THE TWO REFUSALS ARE DIFFERENT ACTIONS
──────────────────────────────────────
*Direct* refusal is "nothing on the table is playable at all".  *Macro* refusal
is "take slot ``s``, whose **printed** number has nowhere to go, and refuse" --
which is legal even when the slot is playable, because the temp agency could have
widened it and ``argWriteNumber`` refuses to force a player to spend the agency
merely to have somewhere to write.  Choosing which slot to burn is a real
decision: the slot you take is the one whose effect you forgo.

LEGALITY IS ENUMERATED, NEVER INTERSECTED
─────────────────────────────────────────
**A macro index is legal iff its full primitive sequence is legal end to end.**
Intersecting per-step masks -- "slot ``s`` is legal" AND "writing 7 in box 4 is
legal" -- would admit pairs that are jointly illegal, because ``WRITE`` legality
depends on *which slot was taken*.  So :func:`legal_macros` walks the engine: it
steps into each playable slot and reads that child's own ``legal_actions()``.
The engine stays the only source of truth about the rules, which is the same
contract ``action_codec`` has.

THE SEARCH HAS A SECOND, SMALLER ACTION SET
──────────────────────────────────────────
:func:`search_legal_macros` is :func:`legal_macros` minus four provably
dominated passes (``SEARCH_SPEC.md`` §5.1).  It exists **as a separate function
on purpose**: ``datagen.replay`` builds its training legal mask from
:func:`legal_mask`, and GreedyBot takes those passes 1,853 times in the recorded
corpus, so folding the pruning into :func:`legal_macros` would make 1,853
recorded labels illegal under their own mask.  Only :mod:`games.welcome_to.mcts`
calls the search version.

STANDARD MODE ONLY
──────────────────
Three choice slots, so expert and solo -- which have six ordered pairs -- have no
macro representation and are refused loudly rather than silently truncated.  They
are out of scope for training (``ENCODER_V2_SPEC.md`` target configuration) and
the engine still speaks primitives for them.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Optional

import numpy as np

from games.welcome_to import action_codec as codec
from games.welcome_to.constants import NUM_BOXES, TEMP_DELTAS, box_coords, box_index
from games.welcome_to.game import GameState, Phase

#: Choice slots a macro can name.  Standard mode has exactly three.
NUM_MACRO_SLOTS: int = 3
NUM_TEMP_DELTAS: int = len(TEMP_DELTAS)

#: The primitive actions the macro layer *subsumes*.  Everything else keeps its
#: own index and its own phase.
_SUBSUMED: tuple[tuple[int, int], ...] = (
    (codec.A_CHOOSE_STACK, codec.NUM_STACK_SLOTS),
    (codec.A_WRITE, NUM_TEMP_DELTAS * NUM_BOXES),
    (codec.A_PERMIT_REFUSAL, 1),
    (codec.A_ROUNDABOUT_OPEN, 1),
)


def _layout() -> tuple[dict[str, int], int]:
    subsumed = {i for base, size in _SUBSUMED for i in range(base, base + size)}
    primitives = tuple(i for i in range(codec.NUM_ACTIONS) if i not in subsumed)
    sizes = (
        ("MACRO_WRITE", NUM_MACRO_SLOTS * NUM_TEMP_DELTAS * NUM_BOXES),
        ("MACRO_REFUSE", NUM_MACRO_SLOTS),
        ("DIRECT_REFUSE", 1),
        ("ROUNDABOUT_OPEN", 1),
        ("PRIMITIVE", len(primitives)),
    )
    offsets: dict[str, int] = {}
    cursor = 0
    for name, size in sizes:
        offsets[name] = cursor
        cursor += size
    offsets["_primitives"] = primitives  # type: ignore[assignment]
    return offsets, cursor


_OFFSET, NUM_MACRO_ACTIONS = _layout()

M_WRITE: int = _OFFSET["MACRO_WRITE"]
M_REFUSE: int = _OFFSET["MACRO_REFUSE"]
M_DIRECT_REFUSE: int = _OFFSET["DIRECT_REFUSE"]
M_ROUNDABOUT_OPEN: int = _OFFSET["ROUNDABOUT_OPEN"]
M_PRIMITIVE: int = _OFFSET["PRIMITIVE"]

#: Primitive codec indices that survive into the macro space, in codec order.
PRIMITIVE_ACTIONS: tuple[int, ...] = _OFFSET["_primitives"]  # type: ignore[assignment]
_PRIMITIVE_TO_MACRO: dict[int, int] = {
    action: M_PRIMITIVE + i for i, action in enumerate(PRIMITIVE_ACTIONS)
}

assert NUM_MACRO_ACTIONS == 684, NUM_MACRO_ACTIONS
assert len(PRIMITIVE_ACTIONS) == 184, len(PRIMITIVE_ACTIONS)


# ──────────────────────────────────────────────────────────────────────────
# Index arithmetic
# ──────────────────────────────────────────────────────────────────────────
def macro_write(slot: int, delta_slot: int, x: int, y: int) -> int:
    """``(slot, temp delta, box)`` -- take a combination and place it."""
    if not 0 <= slot < NUM_MACRO_SLOTS:
        raise ValueError(f"slot {slot} is outside standard mode's three stacks")
    return M_WRITE + (slot * NUM_TEMP_DELTAS + delta_slot) * NUM_BOXES + box_index(x, y)


def decode_macro_write(index: int) -> tuple[int, int, int, int]:
    """``(slot, delta_slot, x, y)``."""
    offset = index - M_WRITE
    box = offset % NUM_BOXES
    rest = offset // NUM_BOXES
    x, y = box_coords(box)
    return rest // NUM_TEMP_DELTAS, rest % NUM_TEMP_DELTAS, x, y


def macro_refuse(slot: int) -> int:
    """Take ``slot``, whose printed number has nowhere to go, and refuse."""
    if not 0 <= slot < NUM_MACRO_SLOTS:
        raise ValueError(f"slot {slot} is outside standard mode's three stacks")
    return M_REFUSE + slot


def from_primitive(action: int) -> int:
    """The macro index of a primitive the macro layer does **not** subsume.

    Raises for a subsumed one rather than mapping it somewhere plausible: a
    bare ``WRITE`` has no macro meaning without the slot that preceded it, and
    quietly inventing one would put two different decisions on one logit.
    """
    try:
        return _PRIMITIVE_TO_MACRO[action]
    except KeyError:
        raise ValueError(
            f"primitive {action} ({codec.describe(action)}) is subsumed by the "
            "macro layer and has no standalone index"
        ) from None


def to_primitive(index: int) -> int:
    """The primitive behind a macro index, for the 184 that have exactly one."""
    if index < M_PRIMITIVE:
        raise ValueError(f"macro {index} ({describe(index)}) is a sequence, not one action")
    return PRIMITIVE_ACTIONS[index - M_PRIMITIVE]


def describe(index: int) -> str:
    if index < M_REFUSE:
        slot, delta_slot, x, y = decode_macro_write(index)
        return f"MACRO_WRITE(slot={slot}, delta={TEMP_DELTAS[delta_slot]:+d}, box=({x},{y}))"
    if index < M_DIRECT_REFUSE:
        return f"MACRO_REFUSE(slot={index - M_REFUSE})"
    if index < M_ROUNDABOUT_OPEN:
        return "DIRECT_REFUSE"
    if index < M_PRIMITIVE:
        return "ROUNDABOUT_OPEN"
    return codec.describe(to_primitive(index))


# ──────────────────────────────────────────────────────────────────────────
# Legality
# ──────────────────────────────────────────────────────────────────────────
def _is_choose_stack(action: int) -> bool:
    return (
        codec.A_CHOOSE_STACK <= action < codec.A_CHOOSE_STACK + codec.NUM_STACK_SLOTS
    )


def is_macro_root(state: GameState) -> bool:
    """Whether this is a state the macro layer makes a decision at.

    Everything except ``WRITE_NUMBER``, which the macro layer swallows.
    """
    return state.phase is not Phase.WRITE_NUMBER


def _require_standard(state: GameState) -> None:
    if not state.config.standard:
        raise ValueError(
            "the macro vocabulary covers standard mode only; expert and solo "
            "have six ordered pairs and no macro representation"
        )


def legal_macros(state: GameState) -> list[int]:
    """Every macro index whose **whole primitive sequence** is legal here.

    At ``CHOOSE_CARDS`` this steps into each playable slot and reads the child's
    own ``legal_actions()``, so the pairs it produces are exactly the pairs the
    engine will accept.  Anywhere else the phase's primitives map straight
    across.
    """
    if state.phase is Phase.GAME_OVER:
        return []
    if state.phase is Phase.WRITE_NUMBER:
        raise ValueError(
            "WRITE_NUMBER is inside a macro; the macro layer never decides here"
        )
    if state.phase is not Phase.CHOOSE_CARDS:
        return [from_primitive(a) for a in state.legal_actions()]

    _require_standard(state)
    out: list[int] = []
    for action in state.legal_actions():
        if action == codec.A_PERMIT_REFUSAL:
            out.append(M_DIRECT_REFUSE)
            continue
        if action == codec.A_ROUNDABOUT_OPEN:
            out.append(M_ROUNDABOUT_OPEN)
            continue
        slot = codec.decode_stack(action)
        child = state.step(action)
        for follow in child.legal_actions():
            if follow == codec.A_PERMIT_REFUSAL:
                out.append(macro_refuse(slot))
            else:
                delta_slot, x, y = codec.decode_write(follow)
                out.append(macro_write(slot, delta_slot, x, y))
    return out


def legal_mask(state: GameState) -> np.ndarray:
    mask = np.zeros(NUM_MACRO_ACTIONS, dtype=bool)
    idx = legal_macros(state)
    if idx:
        mask[np.asarray(idx, dtype=np.int64)] = True
    return mask


# ──────────────────────────────────────────────────────────────────────────
# The search's action set -- dominance pruning, SEARCH_SPEC.md §5.1
# ──────────────────────────────────────────────────────────────────────────
#: Passes the search never spends budget on, by the phase they are offered at.
#:
#: Each is **provably** dominated, not merely unpromising.  Park, pool and
#: estate advance a scoring track and nothing else -- ``parks[x] += 1``,
#: ``pools[street] += 1``, ``estate_marks[row] += 1`` -- consuming no box,
#: fence, number, turn or resource; ``PARK_SCORES`` and ``POOL_SCORES`` are
#: strictly increasing, and every plan predicate that reads them
#: (``DECORATIVE``, ``COMPLETE_STREET``) is monotone.  Plans are not
#: auto-validated (``CHOOSE_PLAN`` has ``PASS_PLAN``), so advancing a track can
#: never force an unwanted three-plan game end.  For the roundabout: opening and
#: then passing reaches the same ``CHOOSE_CARDS`` state as never opening, minus
#: the option, so not opening weakly dominates it.
#:
#: ⚠ All four are pruned.  ``PASS_ROUNDABOUT`` is the one sitting behind a
#: switch (``search_legal_macros``'s ``prune_roundabout_pass``, on) because it is
#: the one that interacts with the *bootstrap* prior -- SEARCH_SPEC §5.1a.
#:
#: ⚠ ``PASS_BIS`` and ``PASS_SURVEYOR`` are deliberately absent.  Bis calls
#: ``sheet.write(..., is_bis=True)``: it fills a box and takes a scoring
#: penalty.  The surveyor fence partitions a street into estates and can destroy
#: an ``EstatePlan``'s required sizes.  Both are genuine decisions.
_DOMINATED_PASS: dict[Phase, int] = {
    Phase.ACTION_PARK: from_primitive(codec.A_PASS_PARK),
    Phase.ACTION_POOL: from_primitive(codec.A_PASS_POOL),
    Phase.ACTION_ESTATE: from_primitive(codec.A_PASS_ESTATE),
    Phase.ROUNDABOUT_PLACE: from_primitive(codec.A_PASS_ROUNDABOUT),
}


def search_legal_macros(
    state: GameState, prune_roundabout_pass: bool = True
) -> list[int]:
    """:func:`legal_macros` minus the dominated passes -- **for the search only**.

    ⚠ **This is a search mask, not a rules change, and it must never move into**
    :func:`legal_macros`.  ``datagen.replay`` builds its training legal mask from
    :func:`legal_mask`, and the reference policy takes these actions anyway:
    measured over 75 GreedyBot games at 2/3/4 seats, ``PASS_ESTATE`` was taken 78
    times of 1435 offers-with-an-alternative and ``PASS_ROUNDABOUT`` 1775 times
    of 2028.  Pruning inside ``legal_macros`` would make 1,853 recorded labels
    illegal under their own mask and break replay.  ``GameState.legal_actions()``,
    the 684 indices and everything ``datagen`` touches stay untouched.

    A pass is dropped only when an alternative exists.  The pass is a single
    index, so ``len(macros) > 1`` *is* that condition -- and measured, it is
    load-bearing for exactly one phase.  ``ACTION_ESTATE`` is entered with
    ``estate_rows()`` empty, because nothing settles that phase away the way
    ``_settle()`` handles park and pool; ``ROUNDABOUT_PLACE`` is not, because it
    offers ``available_locations(None)``, which is the same ``has_free_box()``
    that put the roundabout on offer.

    ⚠ ``prune_roundabout_pass`` is on, and is a switch only because pruning it
    interacts badly with a **bootstrap** prior -- ``SEARCH_SPEC.md`` §5.1a, which
    is worth reading before trusting an early checkpoint's roundabout count.
    Once the pass is gone, ``ROUNDABOUT_OPEN`` means "build one, for -3 or -8
    points", and a policy cloned from GreedyBot arrives putting ~30% on it.
    That 30% is an artifact, not a preference: opening the prompt does not touch
    the sheet, so GreedyBot's one-ply evaluation scores it **identically** to
    doing nothing and it lands in the tied-best set 100% of the time (strictly
    best 4%), to be picked by a coin flip over ~3.5 tied actions.  The problem
    is therefore in the bootstrap data, not in this function.
    """
    macros = legal_macros(state)
    if not prune_roundabout_pass and state.phase is Phase.ROUNDABOUT_PLACE:
        return macros
    pruned = _DOMINATED_PASS.get(state.phase)
    if pruned is None or len(macros) < 2:
        return macros
    return [m for m in macros if m != pruned]


def search_legal_mask(
    state: GameState, prune_roundabout_pass: bool = True
) -> np.ndarray:
    """:func:`legal_mask`'s counterpart over :func:`search_legal_macros`."""
    mask = np.zeros(NUM_MACRO_ACTIONS, dtype=bool)
    idx = search_legal_macros(state, prune_roundabout_pass)
    if idx:
        mask[np.asarray(idx, dtype=np.int64)] = True
    return mask


# ──────────────────────────────────────────────────────────────────────────
# Applying
# ──────────────────────────────────────────────────────────────────────────
def primitives_for(index: int) -> tuple[int, ...]:
    """The primitive sequence a macro index stands for."""
    if index < M_REFUSE:
        slot, delta_slot, x, y = decode_macro_write(index)
        return (codec.choose_stack(slot), codec.write(delta_slot, x, y))
    if index < M_DIRECT_REFUSE:
        return (codec.choose_stack(index - M_REFUSE), codec.A_PERMIT_REFUSAL)
    if index < M_ROUNDABOUT_OPEN:
        return (codec.A_PERMIT_REFUSAL,)
    if index < M_PRIMITIVE:
        return (codec.A_ROUNDABOUT_OPEN,)
    return (to_primitive(index),)


def apply_macro(state: GameState, index: int) -> None:
    """Apply the whole sequence in place.  Raises if any step is illegal."""
    for action in primitives_for(index):
        state.apply(action)


def step_macro(state: GameState, index: int) -> GameState:
    """Apply the whole sequence to a copy and return it."""
    nxt = state.copy()
    apply_macro(nxt, index)
    return nxt


# ──────────────────────────────────────────────────────────────────────────
# Reading a primitive trajectory as macro labels
# ──────────────────────────────────────────────────────────────────────────
def collapse(state: GameState, actions: Iterable[int]) -> Iterator[tuple[GameState, int]]:
    """Walk a primitive trajectory, yielding ``(state, macro action)`` per decision.

    Deterministic and total: every primitive trajectory has exactly one macro
    reading.  ``CHOOSE_STACK`` is always followed immediately by that same
    actor's ``WRITE`` or ``PERMIT_REFUSAL`` -- turns are serialised, so nothing
    can interleave -- and the pair collapses to one label emitted at the
    ``CHOOSE_CARDS`` state.  **No sample is emitted at ``WRITE_NUMBER``**, which
    is the point: under this vocabulary the network is never asked there.

    The state is yielded *before* the action is applied and is not copied, so a
    consumer that needs to keep it must encode or copy it on the spot.
    """
    _require_standard(state)
    pending = iter(actions)
    for action in pending:
        if state.phase is Phase.CHOOSE_CARDS and _is_choose_stack(action):
            slot = codec.decode_stack(action)
            follow = next(pending, None)
            if follow is None:
                raise ValueError("trajectory ends inside a macro: CHOOSE_STACK unpaired")
            if follow == codec.A_PERMIT_REFUSAL:
                macro = macro_refuse(slot)
            else:
                delta_slot, x, y = codec.decode_write(follow)
                macro = macro_write(slot, delta_slot, x, y)
            yield state, macro
            state.apply(action)
            state.apply(follow)
            continue

        if action == codec.A_PERMIT_REFUSAL:
            macro = M_DIRECT_REFUSE
        elif action == codec.A_ROUNDABOUT_OPEN:
            macro = M_ROUNDABOUT_OPEN
        else:
            macro = from_primitive(action)
        yield state, macro
        state.apply(action)
