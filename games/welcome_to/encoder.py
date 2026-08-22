"""
State encoder for the Welcome To value/policy network.

DESIGN CONTRACT
───────────────
1. **Information-set safe.**  The encoder reads only what the acting player can
   legitimately know:

   * opponents' sheets come from :attr:`GameState.public_sheets`, the snapshot
     taken at the start of the turn, so nothing another player wrote *this*
     turn is visible;
   * City Plan completions from the current turn are hidden for everyone but
     the viewer (:meth:`GameState.plan_turns_for`);
   * the undrawn deck is never read.  Everything the player is told about what
     is coming is computed from public information by
     :mod:`games.welcome_to.deck_knowledge`.  Note how little is actually
     hidden: a card's number face prints its own effect, so the effect each
     stack offers *next* turn is a certainty, fed in as a one-hot, and the deck
     composition is exact bookkeeping rather than an estimate.  Only the next
     *number* is unknown, and even that has an exact distribution.

   Search must still call :meth:`GameState.redeterminize` at its root, passing a
   search RNG it advances between simulations; a clean encoder does not make a
   cheating rollout honest, and a repeated determinization is not a sample.

2. **Full symmetry across seats.**  Every seat -- the viewer and each opponent --
   is encoded by the *same function* into the *same* planes and scalars, and is
   meant to run through the same shared per-sheet weights.  ``encode_state``
   returns four arrays::

       sheet_planes    (4, 12, 3, 12)   one block per seat, identical function
       sheet_scalars   (4, 45)          one block per seat, identical function
       viewer_plane    (1, 3, 12)       phase scratch, viewer only
       global_scalars  (358,)           game-wide and viewer-relative

   Seats are padded to :data:`MAX_SEATS` and carry a validity flag, so one set
   of weights serves 2, 3 and 4 player games; the seat axis is the viewer first,
   then turn order (:func:`seat_order`).

   The reason for symmetry is that at the top of the field everyone is a good
   solitaire player.  What separates players is the other half of the game -- the
   plan races, reading what an opponent will finish and when -- and a model that
   represents its own sheet richly and its opponents' as two occupancy planes
   cannot compete on that axis.  Symmetry was never needed to make opponents
   *play* well: the encoder is viewer-relative and the viewer rotates.  It is
   needed to make the viewer's model *of* them as good as its model of itself.

3. **Sheet-shaped spatial planes.**  The three streets are laid out as a
   ``3 x 12`` grid, right-padded (street 0 is 10 long, street 1 is 11).  Plane 0
   is the validity mask for that padding.  There is no useful symmetry group
   here -- streets are not interchangeable and the left-to-right ascending rule
   breaks reflection -- so there is no augmentation hook, unlike Kingdomino.
   Weight sharing across *seats* is the sharing that pays in this game; weight
   sharing across positions is not.

4. **The two interactions are given first-class features.**  Welcome To is
   mostly played on your own sheet, and what is left is a *race*: whoever
   completes the three City Plans first usually wins, and the plan that pays 12
   to the first finisher pays 7 to everyone after.  Completion flags alone
   cannot express "two parks away", so each seat's block carries a
   distance-to-completion per plan slot, off that seat's public sheet
   (:func:`games.welcome_to.plans.progress`).  The temp-agency majority is the
   second interaction and is visible through the per-seat track and score
   blocks, which every seat now gets in full.

5. **Placement capacity is a feature, not something to be learned twice.**
   The strictly-ascending rule means a badly placed number destroys future
   placements wholesale -- writing 15 into the first box of a street takes its
   capacity from 10 to 2.  That quantity is what separates competent play from
   noise in this game, so :meth:`~games.welcome_to.sheet.Sheet.placement_capacity`,
   :meth:`~games.welcome_to.sheet.Sheet.box_spans` and
   :meth:`~games.welcome_to.sheet.Sheet.positional_fit` are all exposed directly,
   as a flat block and as spatial planes.  On a one-ply bot those three terms are
   jointly worth about twelve points a game over scoring alone, which is the
   evidence for handing them over rather than hoping the trunk rediscovers them.

6. **Training-isolated.**  Nothing here imports a heuristic evaluator.

``ENCODER_V2_SPEC.md`` is the spec of record.  Implemented so far: §12 steps 1
and 2 -- the dense plan one-hot and this seat-axis restructure.  The feature
blocks of steps 3-8 are not here yet, so the per-sheet width is 12 planes and 45
scalars rather than the 17 and 127 the spec ends at.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from games.welcome_to import deck_knowledge as dk
from games.welcome_to.constants import (
    BIS_BOXES,
    ESTATE_ROW_BOXES,
    MAX_STREET_LEN,
    NUM_BASE_CARDS,
    NUM_BOXES,
    NUM_NUMBER_VALUES,
    NUM_STREETS,
    PARK_BOXES,
    PERMIT_BOXES,
    POOL_BOXES,
    POOL_POSITION_SET,
    ROUNDABOUT,
    ROUNDABOUT_BOXES,
    STREET_SIZES,
    TEMP_BOXES,
    box_index,
)
from games.welcome_to.game import GameState, Phase
from games.welcome_to.plans import NUM_DEALT_PLANS, PLANS, dense_index, progress
from games.welcome_to.sheet import Sheet

#: Seats encoded individually.  A fifth seat is dropped from the seat axis
#: rather than pooled: pooling is seat-count-invariant but destroys identity,
#: and identity -- *which* opponent finishes plan 2 first -- is the whole point.
MAX_SEATS: int = 4
MAX_OPPONENTS: int = MAX_SEATS - 1
#: Width of the seat-index one-hot.
MAX_PLAYERS: int = 6

# -- per-sheet plane indices, public so tests cannot drift -----------------
P_VALID = 0          #: right-padding mask for the 3x12 grid
P_WRITTEN = 1        #: box holds a number or a roundabout
P_NUMBER = 2         #: that number / 17
P_BIS = 3
P_ROUNDABOUT = 4
P_TOP_FENCE = 5      #: house consumed by a completed plan
P_FENCE_RIGHT = 6
P_POOL = 7
P_WRITABLE = 8       #: writable with *any* currently-offered combination
P_ESTATE_SIZE = 9    #: estate size / 6
P_SPAN = 10          #: box_spans / 18
P_FIT = 11           #: positional_fit of the numbers on offer
SHEET_PLANES: int = 12

SHEET_PLANES_SHAPE: tuple[int, int, int, int] = (
    MAX_SEATS,
    SHEET_PLANES,
    NUM_STREETS,
    MAX_STREET_LEN,
)
VIEWER_PLANE_SHAPE: tuple[int, int, int] = (1, NUM_STREETS, MAX_STREET_LEN)

_NUM_EFFECTS = dk.NUM_EFFECTS   # 6
_NUM_NUMBERS = dk.NUM_NUMBERS   # 15
_EFFECT_INDEX = dk.EFFECT_INDEX

#: Named blocks of one seat's flat vector, in order.  Written by the same
#: function for every seat; anything that cannot be computed for an opponent
#: does not belong here.
SHEET_SCALAR_BLOCKS: tuple[tuple[str, int], ...] = (
    ("tracks", 20),
    ("score", 9),
    ("capacity", 4),
    ("plans", 3 * 3),
    ("free_boxes", 1),
    ("is_viewer", 1),
    ("seat_valid", 1),
)
NUM_SHEET_SCALAR: int = sum(size for _, size in SHEET_SCALAR_BLOCKS)

#: Named blocks of the game-wide flat vector, in order.  Viewer-relative is
#: fine here; per-*seat* is not -- that goes in the sheet block above.
GLOBAL_SCALAR_BLOCKS: tuple[tuple[str, int], ...] = (
    ("phase", len(Phase)),
    ("turn", 1),
    ("stacks", 3 * (NUM_NUMBER_VALUES + _NUM_EFFECTS) + 6),
    ("chosen_combination", NUM_NUMBER_VALUES + _NUM_EFFECTS + 1),
    ("last_house", NUM_BOXES + 1),
    ("pending_estate", 7),
    ("plan_identity", 3 * (NUM_DEALT_PLANS + 3)),
    ("reshuffle_race", 2),
    ("next_effects", 3 * _NUM_EFFECTS),
    ("deck", 2 + 3 * _NUM_NUMBERS + 2 * _NUM_EFFECTS + _NUM_NUMBERS),
    ("config", 4),
    ("seat", MAX_PLAYERS),
    ("seat_validity", MAX_SEATS),
)
NUM_GLOBAL_SCALAR: int = sum(size for _, size in GLOBAL_SCALAR_BLOCKS)


def _normalise(vector: np.ndarray) -> np.ndarray:
    total = float(vector.sum())
    if total <= 0.0:
        return np.zeros_like(vector)
    return (vector / total).astype(np.float32)


_TURN_SCALE = 30.0
_SCORE_SCALE = 50.0
_STEPS_SCALE = 12.0


def seat_order(state: GameState, viewer: int) -> list[int]:
    """The seat axis: the viewer, then turn order, capped at :data:`MAX_SEATS`.

    This is the ordering every per-seat array in the project is indexed by --
    the encoder's seat axis and the seat-indexed training targets alike -- so it
    lives here and callers ask for it instead of re-deriving it.
    """
    players = state.config.players
    return [viewer] + [(viewer + k) % players for k in range(1, players)][
        : MAX_SEATS - 1
    ]


# ──────────────────────────────────────────────────────────────────────────
# Spatial planes
# ──────────────────────────────────────────────────────────────────────────
def _offered_numbers(state: GameState, viewer: int) -> list[int]:
    """Every number writable from a combination currently on the table.

    Read from the **viewer's** stacks deliberately.  In standard mode the three
    stacks are shared (``_group_of`` returns 0 for everyone), so this is what
    every seat is offered; in expert mode it is not, and reading each seat's own
    stacks there would hand the viewer private information.

    Includes the temp-agency widening, because a combination's numbers are what
    it can actually be written as.
    """
    out: list[int] = []
    for number, effect in state.visible_cards(viewer):
        if number is None or effect is None:
            continue
        out.extend(state.numbers_for(number, effect))
    return out


def _estate_size_grid(sheet: Sheet) -> np.ndarray:
    grid = np.zeros((NUM_STREETS, MAX_STREET_LEN), dtype=np.float32)
    for x, start, size in sheet.estates():
        grid[x, start : start + size] = size
    return grid


def _sheet_planes(sheet: Sheet, offered: list[int], out: np.ndarray) -> None:
    """One seat's 12 planes.  Nothing here reads the viewer or the phase.

    That is the property the symmetry test checks, and it is what forced plane
    8 to be redefined: v1 read ``state.ctx.number``, the combination the viewer
    has *already locked in this turn*, which is viewer-only scratch state and
    would be all-zero for every opponent by construction.  "Writable with any
    currently-offered combination" is well defined for every seat -- and it is
    live at ``CHOOSE_CARDS``, where the v1 plane was blank and where the
    question "what are my options" is actually being asked.  The locked-in mask
    is not lost; it is the separate viewer plane.
    """
    writable: set[tuple[int, int]] = set()
    for n in offered:
        writable.update(sheet.available_locations(n))

    spans = sheet.box_spans()
    for x, size in enumerate(STREET_SIZES):
        for y in range(size):
            n = sheet.numbers[x][y]
            out[P_VALID, x, y] = 1.0
            if n is not None:
                out[P_WRITTEN, x, y] = 1.0
                if n == ROUNDABOUT:
                    out[P_ROUNDABOUT, x, y] = 1.0
                else:
                    out[P_NUMBER, x, y] = n / 17.0
            out[P_BIS, x, y] = float(sheet.is_bis[x][y])
            out[P_TOP_FENCE, x, y] = float(sheet.top_fences[x][y])
            if y < size - 1:
                out[P_FENCE_RIGHT, x, y] = float(sheet.fences[x][y])
            out[P_POOL, x, y] = float((x, y) in POOL_POSITION_SET)
            out[P_WRITABLE, x, y] = float((x, y) in writable)
            # how many numbers could still land here at all
            out[P_SPAN, x, y] = spans[x][y] / 18.0
            # and how well the numbers actually on offer would sit here.
            # `positional_fit` returns 0.0 for a PERFECT fit and None for no fit
            # at all, so the two must be told apart explicitly -- `fit or -99.0`
            # reads the best possible placement as the worst one.
            if n is None:
                fits = [
                    f
                    for f in (sheet.positional_fit(v, x, y) for v in offered)
                    if f is not None
                ]
                if fits:
                    out[P_FIT, x, y] = 1.0 / (1.0 - max(fits))
    out[P_ESTATE_SIZE] = _estate_size_grid(sheet) / 6.0


def _viewer_plane(state: GameState, viewer: int, out: np.ndarray) -> None:
    """Boxes legal for the combination the viewer has already locked in.

    The v1 plane-8 semantics, kept because it is effectively the legal-action
    mask in spatial form and the policy head benefits from the trunk seeing it.
    It sits outside the shared sheet encoder because it is phase scratch state,
    not a property of a sheet.
    """
    sheet = state.sheets[viewer]
    boxes: set[tuple[int, int]] = set()
    if state.phase is Phase.WRITE_NUMBER and state.ctx.number is not None:
        assert state.ctx.effect is not None
        for n in state.numbers_for(state.ctx.number, state.ctx.effect):
            boxes.update(sheet.available_locations(n))
    elif state.phase is Phase.ROUNDABOUT_PLACE:
        boxes.update(sheet.available_locations(None))
    for x, y in boxes:
        out[0, x, y] = 1.0


# ──────────────────────────────────────────────────────────────────────────
# Flat features
# ──────────────────────────────────────────────────────────────────────────
class _Writer:
    """Append-only cursor over a flat vector, checked against its block table."""

    def __init__(self, size: int) -> None:
        self.buf = np.zeros(size, dtype=np.float32)
        self.pos = 0

    def put(self, *values: float) -> None:
        for v in values:
            self.buf[self.pos] = v
            self.pos += 1

    def put_array(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float32).ravel()
        self.buf[self.pos : self.pos + flat.shape[0]] = flat
        self.pos += flat.shape[0]

    def one_hot(self, index: Optional[int], size: int) -> None:
        if index is not None and 0 <= index < size:
            self.buf[self.pos + index] = 1.0
        self.pos += size

    def skip(self, n: int) -> None:
        self.pos += n


def _sheet_scalars(state: GameState, viewer: int, seat: int) -> np.ndarray:
    """One seat's 45 flat features, by the same function for every seat.

    Every read goes through a viewer-safe accessor -- ``sheet_for``,
    ``score_breakdown(..., viewer=)``, ``plan_turns_for`` -- so this is symmetric
    *and* information-set safe by the same construction.  Reaching for
    ``state.sheets[seat]`` here would break both at once, which is exactly what
    the symmetry test is for.
    """
    sheet = state.sheet_for(viewer, seat)
    w = _Writer(NUM_SHEET_SCALAR)

    # tracks (20)
    for x in range(NUM_STREETS):
        w.put(sheet.parks[x] / PARK_BOXES[x])
    w.put(
        sheet.pool_count / POOL_BOXES,
        sheet.temps / TEMP_BOXES,
        sheet.bis_marks / BIS_BOXES,
        sheet.permits / PERMIT_BOXES,
        sheet.roundabouts / ROUNDABOUT_BOXES,
    )
    for i in range(6):
        w.put(sheet.estate_marks[i] / ESTATE_ROW_BOXES[i])
    for count in sheet.estate_size_counts():
        w.put(count / 4.0)

    # score components (9)
    breakdown = state.score_breakdown(seat, viewer=viewer)
    w.put(
        breakdown.parks / _SCORE_SCALE,
        breakdown.pools / _SCORE_SCALE,
        breakdown.estates / _SCORE_SCALE,
        breakdown.plans / _SCORE_SCALE,
        breakdown.temp / _SCORE_SCALE,
        breakdown.bis / _SCORE_SCALE,
        breakdown.permits / _SCORE_SCALE,
        breakdown.roundabouts / _SCORE_SCALE,
        breakdown.total / 100.0,
    )

    # placement capacity: what a careless write burns (4)
    capacity = sheet.placement_capacity()
    for x in range(NUM_STREETS):
        w.put(capacity[x] / STREET_SIZES[x])
    w.put(sum(capacity) / NUM_BOXES)

    # THE RACE: how far this seat is from each plan, and whether it banked it (9)
    for slot, plan_id in enumerate(state.plan_ids):
        fraction, steps = progress(PLANS[plan_id], sheet)
        banked = seat in state.plan_turns_for(viewer, slot)
        w.put(fraction, min(steps, _STEPS_SCALE) / _STEPS_SCALE, float(banked))

    # free boxes (1)
    written = sum(1 for row in sheet.numbers for n in row if n is not None)
    w.put((NUM_BOXES - written) / NUM_BOXES)

    w.put(1.0 if seat == viewer else 0.0)
    w.put(1.0)  # seat_valid; padded seats never reach this function

    assert w.pos == NUM_SHEET_SCALAR, f"wrote {w.pos}, expected {NUM_SHEET_SCALAR}"
    return w.buf


def _global_scalars(state: GameState, viewer: int) -> np.ndarray:
    cfg = state.config
    w = _Writer(NUM_GLOBAL_SCALAR)

    # phase, turn
    w.one_hot(int(state.phase), len(Phase))
    w.put(state.turn / _TURN_SCALE)

    # the three stacks as the viewer sees them, plus which choices are playable
    for number, effect in state.visible_cards(viewer):
        w.one_hot(number, NUM_NUMBER_VALUES)
        w.one_hot(_EFFECT_INDEX.get(effect) if effect is not None else None, _NUM_EFFECTS)
    playable = set(state.playable_slots(viewer)) if viewer == state.actor else set()
    for slot in range(6):
        w.put(1.0 if slot in playable else 0.0)

    # the combination the viewer locked in, if any
    ctx = state.ctx if viewer == state.actor else None
    if ctx is not None and ctx.number is not None:
        w.one_hot(ctx.number, NUM_NUMBER_VALUES)
        w.one_hot(_EFFECT_INDEX.get(ctx.effect), _NUM_EFFECTS)
        w.put(1.0)
    else:
        w.skip(NUM_NUMBER_VALUES + _NUM_EFFECTS + 1)

    # the house written this turn
    if ctx is not None and ctx.last_house is not None:
        w.one_hot(box_index(*ctx.last_house), NUM_BOXES)
        w.put(1.0)
    else:
        w.skip(NUM_BOXES + 1)

    # the estate size a plan validation is currently waiting on
    if ctx is not None and ctx.pending_sizes:
        w.one_hot(ctx.pending_sizes[0] - 1, 6)
        w.put(1.0)
    else:
        w.skip(7)

    # WHICH plans are in play and what they pay.  *Who* has banked them is a
    # per-seat fact and lives in the sheet block; what is left here is the one
    # genuinely shared thing -- whether the first-place value is still unclaimed.
    for slot, plan_id in enumerate(state.plan_ids):
        plan = PLANS[plan_id]
        w.one_hot(dense_index(plan_id), NUM_DEALT_PLANS)
        w.put(
            plan.scores[0] / 20.0,
            plan.scores[1] / 20.0,
            0.0 if state.plan_turns_for(viewer, slot) else 1.0,
        )

    # THE THIRD RACE: whoever finishes the first plan chooses the reshuffle.
    # The second flag is the viewer's OWN vote, not the table-wide
    # `reshuffle_next_turn`: turns are serialised, so the aggregate would tell a
    # later actor that an earlier one voted yes -- and therefore that they
    # completed a plan this turn, which `plan_turns_for` is at pains to hide.
    w.put(
        1.0 if state._may_ask_reshuffle() else 0.0,
        1.0 if state.reshuffle_vote_for(viewer) else 0.0,
    )

    # NEXT TURN'S EFFECTS: known now, one-hot per stack
    w.put_array(dk.known_next_effects(state, viewer))

    # WHAT IS COMING: the deck is exact bookkeeping, not an estimate
    deck = dk.deck_composition(state, viewer)
    discard = dk.discard_composition(state, viewer)
    reshuffled = dk.after_reshuffle_composition(state, viewer)
    w.put(state.deck_remaining / NUM_BASE_CARDS, len(state.discard) / NUM_BASE_CARDS)
    w.put_array(deck.sum(axis=1) / 9.0)
    w.put_array(deck.sum(axis=0) / 20.0)
    w.put_array(discard.sum(axis=1) / 9.0)
    w.put_array(discard.sum(axis=0) / 20.0)
    w.put_array(dk.next_number_distribution(state, viewer))
    w.put_array(_normalise(reshuffled.sum(axis=1)))

    # configuration and seat
    w.put(
        float(cfg.advanced),
        float(cfg.expert),
        float(cfg.solo),
        cfg.players / MAX_PLAYERS,
    )
    w.one_hot(viewer if viewer < MAX_PLAYERS else None, MAX_PLAYERS)
    seats = len(seat_order(state, viewer))
    for k in range(MAX_SEATS):
        w.put(1.0 if k < seats else 0.0)

    assert w.pos == NUM_GLOBAL_SCALAR, f"wrote {w.pos}, expected {NUM_GLOBAL_SCALAR}"
    return w.buf


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────
def encode_state(
    state: GameState, player: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode ``state`` from ``player``'s point of view.

    Returns ``(sheet_planes, sheet_scalars, viewer_plane, global_scalars)`` with
    shapes :data:`SHEET_PLANES_SHAPE`, ``(MAX_SEATS, NUM_SHEET_SCALAR)``,
    :data:`VIEWER_PLANE_SHAPE` and ``(NUM_GLOBAL_SCALAR,)``, all ``float32``.

    The seat axis is :func:`seat_order`: the viewer at index 0, then turn order.
    Unused seats are left zero, ``seat_valid`` included -- an absent seat
    contributes nothing, which is not the same as a seat that scored zero.
    """
    viewer = state.actor if player is None else player
    sheet_planes = np.zeros(SHEET_PLANES_SHAPE, dtype=np.float32)
    sheet_scalars = np.zeros((MAX_SEATS, NUM_SHEET_SCALAR), dtype=np.float32)
    viewer_plane = np.zeros(VIEWER_PLANE_SHAPE, dtype=np.float32)

    offered = _offered_numbers(state, viewer)
    for k, seat in enumerate(seat_order(state, viewer)):
        _sheet_planes(state.sheet_for(viewer, seat), offered, sheet_planes[k])
        sheet_scalars[k] = _sheet_scalars(state, viewer, seat)

    _viewer_plane(state, viewer, viewer_plane)
    return sheet_planes, sheet_scalars, viewer_plane, _global_scalars(state, viewer)


def encode_batch(
    states: list[GameState], players: Optional[list[int]] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack :func:`encode_state` over a list of states."""
    if players is None:
        players = [s.actor for s in states]
    n = len(states)
    sheet_planes = np.zeros((n, *SHEET_PLANES_SHAPE), dtype=np.float32)
    sheet_scalars = np.zeros((n, MAX_SEATS, NUM_SHEET_SCALAR), dtype=np.float32)
    viewer_plane = np.zeros((n, *VIEWER_PLANE_SHAPE), dtype=np.float32)
    global_scalars = np.zeros((n, NUM_GLOBAL_SCALAR), dtype=np.float32)
    for i, (state, player) in enumerate(zip(states, players)):
        (
            sheet_planes[i],
            sheet_scalars[i],
            viewer_plane[i],
            global_scalars[i],
        ) = encode_state(state, player)
    return sheet_planes, sheet_scalars, viewer_plane, global_scalars


def _build_block_index() -> dict[str, tuple[str, slice]]:
    index: dict[str, tuple[str, slice]] = {}
    for axis, table in (
        ("sheet", SHEET_SCALAR_BLOCKS),
        ("global", GLOBAL_SCALAR_BLOCKS),
    ):
        cursor = 0
        for name, size in table:
            assert name not in index, f"duplicate scalar block name {name!r}"
            index[name] = (axis, slice(cursor, cursor + size))
            cursor += size
    return index


_BLOCKS: dict[str, tuple[str, slice]] = _build_block_index()


def block_slice(name: str) -> slice:
    """Where a named scalar block lives along the last axis of its vector.

    Useful for probing a trained model ("does it use the next-reveal
    posterior?") and for ablations.  Block names are unique across the sheet and
    global tables; :func:`block_axis` says which of the two to index.
    """
    try:
        return _BLOCKS[name][1]
    except KeyError:
        raise KeyError(f"no scalar block named {name!r}") from None


def block_axis(name: str) -> str:
    """``"sheet"`` or ``"global"`` -- which vector :func:`block_slice` indexes."""
    try:
        return _BLOCKS[name][0]
    except KeyError:
        raise KeyError(f"no scalar block named {name!r}") from None
