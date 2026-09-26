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

       sheet_planes    (4, 22, 3, 12)   one block per seat, identical function
       sheet_scalars   (4, 196)         one block per seat, identical function
       viewer_plane    (1, 3, 12)       phase scratch, viewer only
       global_scalars  (367,)           game-wide and viewer-relative

   Seats are padded to :data:`MAX_SEATS` and carry a validity flag, so one set
   of weights serves 2, 3 and 4 player games; the seat axis is the viewer first,
   then turn order (:func:`seat_order`).

3. **Sheet-shaped spatial planes.**  The three streets are laid out as a
   ``3 x 12`` grid, right-padded (street 0 is 10 long, street 1 is 11).  Plane 0
   is the validity mask for that padding.  There is no useful symmetry group
   here -- streets are not interchangeable and the left-to-right ascending rule
   breaks reflection -- so there is no augmentation hook, unlike Kingdomino.

4. **The two interactions are given first-class features**: the plan race
   (§3, §6) and the temp-agency majority (§5).

5. **Placement capacity is a feature, not something to be learned twice**
   (§4, §7).

6. **Training-isolated.**  Nothing here imports a heuristic evaluator.

``ENCODER_V3_SPEC.md`` is the spec of record and this module implements its §12
step 5: every plane and block at **22 planes / 196 per-sheet / 367 global**,
ABI 1 -> 2.  Section references below are to that document.

⚠ **§6.4 DEMOTED (2026-09-25, user's call).**  ``can_complete_this_turn`` and
``p_complete_next_turn`` were 96.5% of encode time (25 ms median / 2.8 s max vs
4.3 ms without them) and are no longer encoded: the plan slot is 34 floats, not
36.  The net is expected to learn plan-completion threat from the requirement,
feasibility and rate fields that remain.  The predicates stay in :mod:`game`
(tested by ``test_plan_threat.py``) as an ablation to re-add, not as inputs.

⚠ **SCOPE (§0.5, decided 2026-09-14): the 2+ player STANDARD game only.**
Expert and one-seat play **raise**.  The fit
planes (§7.1, §7.5) and the boundary-draw rates (§9.3) all rest on "every seat
sees the same three stacks and opponents' sheets are public", which is
``config.standard``; and ``known_next_effects`` is all-zero outside it, so their
inputs do not exist.  A boundary **afterstate** also raises, via
:mod:`deck_knowledge`'s own guard -- its discard step has already run, so the
draw pool would silently include the cards just promoted.

⚠ **FOUR DEFINITIONAL CHOICES the spec leaves open** are resolved here and
marked ``SPEC GAP`` at the point of use.  They must be mirrored exactly in Rust
(§10.6), so they are stated as rules rather than left to the implementation:
``_SURVEYOR_DEMAND_SCALE``, the ``reshuffle_contraction`` duplication, the
"best" roundabout placement for §8, and the per-street -> per-effect reduction
in :func:`_effect_needs`.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from games.welcome_to import deck_knowledge as dk
from games.welcome_to.constants import (
    BIS_BOXES,
    CARD_NUMBERS,
    DECK_EFFECT_ORDER,
    EPS,
    ESTATE_ROW_BOXES,
    ESTATE_ROW_SCORES,
    MAX_NUMBER,
    MAX_STREET_LEN,
    MIN_NUMBER,
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
    TEMP_DELTAS,
    Effect,
    box_index,
)
from games.welcome_to.game import (
    GameState,
    Phase,
    bis_usable,
    max_houses_this_turn,
)
from games.welcome_to.plans import (
    NUM_DEALT_PLANS,
    PLANS,
    Plan,
    PlanKind,
    dense_index,
    feasible,
    progress,
    requirements,
    turns_lower_bound,
)
from games.welcome_to.sheet import Sheet

#: Seats encoded individually.  A fifth seat is dropped from the seat axis
#: rather than pooled: pooling is seat-count-invariant but destroys identity,
#: and identity -- *which* opponent finishes plan 2 first -- is the whole point.
MAX_SEATS: int = 4
MAX_OPPONENTS: int = MAX_SEATS - 1
# Cross-language layout contract. Increment this whenever the meaning, order,
# dtype, or shape of any encode_state output changes.
#
# 2: ENCODER_V3_SPEC.md step 5.  A hard break -- no checkpoint migration, no WTS
#    back-compatibility, no legacy head zero-fill (§0.4).
ENCODER_ABI_VERSION: int = 2
#: Width of the seat-index one-hot.
MAX_PLAYERS: int = 6

#: §9.4: every turn-valued feature is ``min(t, TURNS_CAP) / TURNS_CAP``, so a
#: supply of zero emits exactly 1.0 ("never") rather than ``inf``.
TURNS_CAP: float = 12.0

# -- per-sheet plane indices, public so tests cannot drift -----------------
# ⚠ §9.1: planes are RENUMBERED relative to v2, which put span_if_roundabout at
# 12.  Never refer to a plane by literal index outside these constants.
P_VALID = 0            #: right-padding mask for the 3x12 grid
P_WRITTEN = 1          #: box holds a number or a roundabout
P_NUMBER = 2           #: that number / 17
P_BIS = 3
P_ROUNDABOUT = 4
P_TOP_FENCE = 5        #: house consumed by a completed plan
P_FENCE_RIGHT = 6
P_POOL = 7
P_WRITABLE = 8         #: §5.2 writable_no_temp -- delta 0 only
P_ESTATE_SIZE = 9      #: estate size / 6
P_SPAN = 10            #: box_spans / 18
P_FIT = 11             #: positional_fit, §5.2 DELTA-0 NUMBERS ONLY
P_WRITABLE_TEMP = 12   #: §5.2 writable_temp_only -- reachable only via a temp
P_SPAN_ROUNDABOUT = 13  #: §4 span_if_roundabout / 18
P_FIT_DECK = 14        #: §7.1 P(fit | deck, no temp)
P_FIT_TEMP = 15        #: §7.1 P(fit | deck, temp +-2)
P_FIT_RESHUFFLE = 16   #: §7.1 P(fit | post-reshuffle pool, no temp)
P_FIT_ROUNDABOUT = 17  #: §4 plane 14 against plane 13's bounds
P_FIT_NEXT_TURN = 18   #: §7.5 P(some stack next turn supplies a fitting number)
P_PLAN_TARGET = (19, 20, 21)  #: §3.3 plan slot k still needs a house here
SHEET_PLANES: int = 22

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

#: §7.1 sentinels: exclusive gap bounds, a roundabout or street end removing one.
_LOW_SENTINEL = MIN_NUMBER - 1    # -1
_HIGH_SENTINEL = MAX_NUMBER + 1   # 18

#: §9.4: the empty-sheet maximum of ``total_span()`` -- NUM_BOXES x 18.
_TOTAL_SPAN_SCALE = float(NUM_BOXES * 18)
#: §7.2 R4: ``NUM_BOXES x max one-mark delta = 33 x 2``.  ``ESTATE_ROW_SCORES[0]``
#: is ``(1, 3)``, so the first mark of a size-1 row is worth 2 on every estate.
_ESTATE_DEMAND_SCALE = 66.0
#: §7.3: ``fit_rate`` divides by ``33 x sum(supply)``; 33 is ``NUM_BOXES``, the
#: most boxes a single card number could serve.
_FIT_RATE_BOX_SCALE = float(NUM_BOXES)

#: ⚠ **SPEC GAP 1.**  §7.2 defines ``effect_demand[e]`` as "marks of effect e
#: still wanted", normalised in §9.4 by "the track size of e".  SURVEYOR has no
#: track: its demand is plan-derived (fences serve estate plans), so there is no
#: structural denominator to quote.  Resolved here as the plan-side maximum --
#: 3 slots x ``MAX_ESTATE_SIZE`` steps -- which is the largest
#: ``estate_steps_left`` the three slots can jointly present.  Stated as a
#: constant so Rust can copy it rather than re-derive a different one.
_SURVEYOR_DEMAND_SCALE = 18.0

#: Per-effect track sizes for §7.2's ``effect_demand``.  ESTATE and SURVEYOR are
#: special-cased (see :func:`_effect_demand`).
_EFFECT_TRACK_SIZE: dict[Effect, float] = {
    Effect.PARK: float(sum(PARK_BOXES)),
    Effect.POOL: float(POOL_BOXES),
    Effect.TEMP: float(TEMP_BOXES),
    Effect.BIS: float(BIS_BOXES),
}

#: Named blocks of one seat's flat vector, in order (§9.2).  Written by the same
#: function for every seat; anything that cannot be computed for an opponent
#: does not belong here.
SHEET_SCALAR_BLOCKS: tuple[tuple[str, int], ...] = (
    ("tracks", 26),                 # 20 shipped + 6 free-estate size counts §2.3
    ("score", 9),
    ("capacity", 4),
    ("roundabout_repair", 3),       # §4
    ("total_span", 1),
    ("plans", 3 * 34),              # §3.4, minus the §6.4 threat pair
    ("demand", 24),                 # §7.2
    ("reshuffle_contraction", 8),   # §7.3
    ("refusal", 5),                 # §8
    ("houses_this_turn", 2),        # §8
    ("plan_conflict_seat", 9),      # §9.2a
    ("free_boxes", 1),
    ("is_viewer", 1),
    ("seat_valid", 1),
)
NUM_SHEET_SCALAR: int = sum(size for _, size in SHEET_SCALAR_BLOCKS)

#: Named blocks of the game-wide flat vector, in order (§9.3).  Viewer-relative
#: is fine here; per-*seat* is not -- that goes in the sheet block above.
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
    ("effect_supply_rate", _NUM_EFFECTS),   # §9.3
    ("temp_availability_rate", 1),          # §9.3
    ("bis_availability_rate", 1),           # §9.3
    ("turns_to_reform", 1),                 # §7.4
    ("config", 4),
    ("seat", MAX_PLAYERS),
    ("seat_validity", MAX_SEATS),
)
NUM_GLOBAL_SCALAR: int = sum(size for _, size in GLOBAL_SCALAR_BLOCKS)

#: Width of one plan slot's sub-block inside ``plans`` (§3.4).
PLAN_SLOT_WIDTH: int = 34


def _normalise(vector: np.ndarray) -> np.ndarray:
    total = float(vector.sum())
    if total <= 0.0:
        return np.zeros_like(vector)
    return (vector / total).astype(np.float32)


def _ratio(num: float, den: float) -> float:
    """§9.4: every quotient is ``num / max(den, EPS)``, clamped to [0, 1]."""
    return min(1.0, max(0.0, num / max(den, EPS)))


def _turns(value: float) -> float:
    """§9.4: ``min(t, TURNS_CAP) / TURNS_CAP``; zero supply lands on exactly 1.0."""
    if not np.isfinite(value):
        return 1.0
    return min(max(value, 0.0), TURNS_CAP) / TURNS_CAP


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


def _require_scope(state: GameState) -> None:
    """§0.5.  The encoder is defined for the 2+ player standard game only."""
    if not state.config.standard or state.config.players < 2:
        raise ValueError(
            "the v3 encoder is defined for the 2+ player standard game only, "
            "not expert or one-seat play (ENCODER_V3_SPEC.md §0.5)"
        )
    if state.boundary_prepared:
        raise ValueError(
            "the v3 encoder reads a mid-turn state; this is a prepared boundary "
            "afterstate, whose discard step has already run"
        )


# ──────────────────────────────────────────────────────────────────────────
# Per-state deck mathematics, built ONCE and shared by all four sheets
#
# §7.1: "Build the prefix sums once per state and reuse them across all four
# sheets -- this is what keeps symmetric encoding cheap."  Everything in this
# class is seat-independent by construction: one deck feeds every seat in
# standard mode, the premise §7.5 and §9.3 rest on.
# ──────────────────────────────────────────────────────────────────────────
class _DeckView:
    """Prefix sums, supplies and rates the whole state shares."""

    __slots__ = (
        "deck_prefix", "reshuffled_prefix", "deck_total", "reshuffled_total",
        "deck_numbers", "reform_numbers", "deck_effects", "reshuffled_numbers",
        "reshuffled_effects", "effect_rate", "next_effects", "next_is_temp",
        "joint", "deck_matrix", "reshuffled_matrix",
    )

    def __init__(self, state: GameState, viewer: int) -> None:
        deck_prefix, _reform_prefix, reshuffled_prefix = dk.number_prefix_sums(
            state, viewer
        )
        self.deck_prefix = deck_prefix
        self.reshuffled_prefix = reshuffled_prefix
        # ⚠ §7.1 R5: divide by the ACTUAL sum of the matrix being summed, never
        # by `deck_remaining`.  They differ whenever the undrawn deck holds a
        # card that is not a printed construction card.
        self.deck_total = float(deck_prefix[-1])
        self.reshuffled_total = float(reshuffled_prefix[-1])

        self.deck_matrix = dk.deck_composition(state, viewer)
        self.reshuffled_matrix = dk.after_reshuffle_composition(state, viewer)
        self.deck_numbers = self.deck_matrix.sum(axis=1)
        self.deck_effects = self.deck_matrix.sum(axis=0)
        self.reshuffled_numbers = self.reshuffled_matrix.sum(axis=1)
        self.reshuffled_effects = self.reshuffled_matrix.sum(axis=0)
        self.reform_numbers = dk.boundary_pool_composition(state, viewer).sum(axis=1)

        self.effect_rate = dk.effect_supply_rate(state, viewer)
        self.next_effects = list(state.next_effects(viewer))
        self.next_is_temp = tuple(e is Effect.TEMP for e in self.next_effects)

        # The ordered joint over next turn's three NUMBERS, reforming mid-draw,
        # as exact integer numerators over one denominator (see
        # `dk.ordered_draw_counts` for why never the float joint).  Built
        # lazily; §8's refusal block needs it on every sheet anyway.
        self.joint: Optional[tuple[np.ndarray, float]] = None

    def ordered_joint(self) -> tuple[np.ndarray, float]:
        if self.joint is None:
            self.joint = dk.ordered_draw_counts(
                self.deck_numbers, self.reform_numbers
            )
        return self.joint

    def draw_probability(self, masks) -> float:
        """P(stack ``i``'s next number is in ``masks[i]``, for all three)."""
        num, den = self.ordered_joint()
        return dk.draw_probability(num, den, masks)

    # -- interval lookups ---------------------------------------------------
    def fit_deck(self, low: int, high: int) -> float:
        """§7.1 plane 14/15: deck cards whose printed number is in ``(low, high)``."""
        count = dk.count_in_open_interval(self.deck_prefix, low, high)
        return _ratio(float(count), self.deck_total)

    def fit_reshuffled(self, low: int, high: int) -> float:
        """§7.1 plane 16, over ``deck + discard + asides``."""
        count = dk.count_in_open_interval(self.reshuffled_prefix, low, high)
        return _ratio(float(count), self.reshuffled_total)

    def deck_count(self, low: int, high: int) -> int:
        return dk.count_in_open_interval(self.deck_prefix, low, high)


def _p_fit_next_turn(view: _DeckView, low: int, high: int) -> float:
    """§7.5.  P(some stack next turn reveals a number fitting this gap).

    Next turn's three *effects* are printed and known, so the only chance is the
    three numbers.  Per stack ``i``, the fitting set is an interval::

        F_i = (low - 2, high + 2)   if stack i's known effect is TEMP
              (low,     high)       otherwise

    -- exclusive bounds, matching planes 15 and 14 exactly.  With ``x_i[n] = 1``
    when ``n`` misses ``F_i``, ``M_i = sum x_i[n] c[n]``,
    ``P_ij = sum x_i x_j c[n]`` and ``R = sum x_1 x_2 x_3 c[n]``::

        none = M1*M2*M3 - P12*M3 - P13*M2 - P23*M1 + 2R

    which is inclusion-exclusion over the three "two draws took the same card"
    events, and ``2R`` because subtracting the three pairs removes the all-equal
    case three times where it was counted once.  Falling factorials, never a
    product of marginals.

    The two intervals are NESTED (``F_notemp`` is a subset of ``F_temp``), so a
    union of any subset of them is simply the widest member -- which is what
    makes every ``P_ij`` and ``R`` one more prefix-sum lookup rather than a new
    enumeration.  O(1) per gap.

    ⚠ **R4: ``D < 3`` must NOT emit 0.0.**  The next reveal still produces three
    cards -- ``_draw`` reforms the discard mid-draw and carries on -- so the true
    probability is generally non-zero, and 0 would state a falsehood at exactly
    the moment a dead-looking gap comes back to life.  Below three cards this
    falls through to the literal boundary-draw enumeration, which reforms.
    """
    fit_notemp = view.deck_count(low, high)
    fit_temp = view.deck_count(low - 2, high + 2)
    fits = [fit_temp if temp else fit_notemp for temp in view.next_is_temp]

    total = view.deck_total
    if total >= 3.0:
        misses = [total - f for f in fits]
        # union of nested intervals == the wider one
        p12 = total - max(fits[0], fits[1])
        p13 = total - max(fits[0], fits[2])
        p23 = total - max(fits[1], fits[2])
        r = total - max(fits)
        none = (
            misses[0] * misses[1] * misses[2]
            - p12 * misses[2]
            - p13 * misses[1]
            - p23 * misses[0]
            + 2.0 * r
        )
        denom = total * (total - 1.0) * (total - 2.0)
        return min(1.0, max(0.0, 1.0 - none / max(denom, EPS)))

    # D < 3: the draw reforms mid-way, so enumerate it literally.
    masks = []
    for temp in view.next_is_temp:
        lo, hi = (low - 2, high + 2) if temp else (low, high)
        miss = np.ones(_NUM_NUMBERS, dtype=np.float64)
        for i, n in enumerate(CARD_NUMBERS):
            if lo < n < hi:
                miss[i] = 0.0
        masks.append(miss)
    return min(1.0, max(0.0, 1.0 - view.draw_probability(masks)))


# ──────────────────────────────────────────────────────────────────────────
# Spatial planes
# ──────────────────────────────────────────────────────────────────────────
def _offered(state: GameState, viewer: int) -> tuple[list[int], list[int]]:
    """``(delta-0 numbers, every reachable number)`` from the table's combinations.

    Read from the **viewer's** stacks deliberately: in standard mode the three
    stacks are shared, so this is what every seat is offered.

    ⚠ §5.1 -- the split is the point.  The shipped encoder unioned
    ``numbers_for`` into ONE writable set, so a box reachable only by spending a
    temp was marked identically to a box that is free.  That is a lie in the
    input, not an omission, and the temp is scarce, scoring and contested.
    """
    base: list[int] = []
    every: list[int] = []
    for number, effect in state.visible_cards(viewer):
        if number is None or effect is None:
            continue
        base.append(number)
        every.extend(state.numbers_for(number, effect))
    return base, every


def _estate_size_grid(sheet: Sheet) -> np.ndarray:
    grid = np.zeros((NUM_STREETS, MAX_STREET_LEN), dtype=np.float32)
    for x, start, size in sheet.estates():
        grid[x, start : start + size] = size
    return grid


def _roundabout_bounds(
    sheet: Sheet, x: int, y: int, available: bool
) -> tuple[int, int]:
    """The gap bounds box ``(x, y)`` would have under the best single roundabout.

    Mirrors :meth:`Sheet.span_if_roundabout` **exactly**, including its tie-break:
    the base interval first, then the left-bound-removed one, then the
    right-bound-removed one, each replacing the incumbent only on a strictly
    wider span.  Plane 17 is plane 14 evaluated against these bounds, so the two
    must select the same interval or ``plane 17 - plane 14`` stops being the
    option value of the banked roundabout.

    ⚠ §4 R5: when no roundabout remains -- base rules, or both already spent --
    this returns the plain bounds, so plane 13 collapses to ``box_spans`` and
    plane 17 to plane 14.  Reporting option value for an option that no longer
    exists is the same lie §5.2 deletes from plane 8.
    """
    bounds = sheet.gap_bounds(x, y)
    assert bounds is not None
    first, last, low, high = bounds
    if not available:
        return low, high

    best = (low, high)
    best_span = max(0, high - low - 1)
    if y > first:
        span = max(0, high - _LOW_SENTINEL - 1)
        if span > best_span:
            best, best_span = (_LOW_SENTINEL, high), span
    if y < last:
        span = max(0, _HIGH_SENTINEL - low - 1)
        if span > best_span:
            best, best_span = (low, _HIGH_SENTINEL), span
    return best


def _sheet_planes(
    state: GameState,
    viewer: int,
    seat: int,
    sheet: Sheet,
    base_numbers: list[int],
    all_numbers: list[int],
    view: _DeckView,
    out: np.ndarray,
) -> None:
    """One seat's 22 planes.

    Nothing here reads the viewer's private scratch state; that is the property
    the symmetry test (§10.5) checks, and it is what forced plane 8 to be
    defined over the *offer* rather than over the combination the viewer has
    already locked in -- which would be all-zero for every opponent by
    construction.  The locked-in mask lives in the separate viewer plane.
    """
    writable_base: set[tuple[int, int]] = set()
    for n in base_numbers:
        writable_base.update(sheet.available_locations(n))
    writable_any: set[tuple[int, int]] = set(writable_base)
    for n in all_numbers:
        writable_any.update(sheet.available_locations(n))
    writable_temp_only = writable_any - writable_base

    spans = sheet.box_spans()
    roundabout_open = (
        state.config.advanced and sheet.can_build_roundabout()
    )
    spans_ra = sheet.span_if_roundabout(available=roundabout_open)

    # §3.3: the plan-target planes, per slot.
    targets: list[set[tuple[int, int]]] = []
    for slot, plan_id in enumerate(state.plan_ids):
        if seat in state.plan_turns_for(viewer, slot):
            targets.append(set())
            continue
        targets.append(set(requirements(PLANS[plan_id], sheet).target_boxes))

    # §7.1: compute the fit planes ONCE PER GAP and broadcast.  A gap is a
    # maximal empty run; every box inside one has identical bounds.
    gap_cache: dict[tuple[int, int, int], tuple[float, float, float, float]] = {}

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
            out[P_WRITABLE, x, y] = float((x, y) in writable_base)
            out[P_WRITABLE_TEMP, x, y] = float((x, y) in writable_temp_only)
            out[P_SPAN, x, y] = spans[x][y] / 18.0
            out[P_SPAN_ROUNDABOUT, x, y] = spans_ra[x][y] / 18.0
            for k in range(3):
                out[P_PLAN_TARGET[k], x, y] = float((x, y) in targets[k])

            if n is not None:
                continue

            # ⚠ §5.2: positional_fit over the DELTA-0 numbers only.  The
            # temp-widened fit is recoverable from planes 14/15.
            #
            # `positional_fit` returns 0.0 for a PERFECT fit and None for no fit
            # at all, so the two must be told apart explicitly -- `fit or -99.0`
            # reads the best possible placement as the worst one.
            fits = [
                f
                for f in (sheet.positional_fit(v, x, y) for v in base_numbers)
                if f is not None
            ]
            if fits:
                out[P_FIT, x, y] = 1.0 / (1.0 - max(fits))

            bounds = sheet.gap_bounds(x, y)
            assert bounds is not None
            _first, _last, low, high = bounds
            key = (x, low, high)
            cached = gap_cache.get(key)
            if cached is None:
                cached = (
                    view.fit_deck(low, high),
                    view.fit_deck(low - 2, high + 2),
                    view.fit_reshuffled(low, high),
                    _p_fit_next_turn(view, low, high),
                )
                gap_cache[key] = cached
            out[P_FIT_DECK, x, y] = cached[0]
            out[P_FIT_TEMP, x, y] = cached[1]
            out[P_FIT_RESHUFFLE, x, y] = cached[2]
            out[P_FIT_NEXT_TURN, x, y] = cached[3]

            ra_low, ra_high = _roundabout_bounds(sheet, x, y, roundabout_open)
            out[P_FIT_ROUNDABOUT, x, y] = view.fit_deck(ra_low, ra_high)

    out[P_ESTATE_SIZE] = _estate_size_grid(sheet) / 6.0


def _viewer_plane(state: GameState, viewer: int, out: np.ndarray) -> None:
    """Boxes legal for the combination the viewer has already locked in.

    Effectively the legal-action mask in spatial form; the policy head benefits
    from the trunk seeing it.  It sits outside the shared sheet encoder because
    it is phase scratch state, not a property of a sheet.

    ⚠ **``state.ctx`` belongs to ``state.actor``, so this reads it only when the
    viewer *is* the actor.**  Without that guard the plane answers "where could
    the viewer write the number the *opponent* has just picked", which is both
    meaningless and a read of the one thing the information-set contract hides
    -- and it is invisible to ``mcts.information_key``, which carries ``ctx``
    only on the viewer's own turn, so two states the key merges could encode
    differently.
    """
    if viewer != state.actor:
        return
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


def _effect_needs(plan: Plan, req, steps_left: int) -> dict[Effect, int]:
    """Marks of each effect this plan still needs, for §6.3's rate terms.

    ⚠ **SPEC GAP 4.**  §6.3 asks for ``sum_e (marks of e still needed)`` but the
    §3.4 requirement vectors are **per-street alternatives**, not work that must
    all be done -- the mistake that made an earlier ``turns_lower_bound`` too
    high for every alternative-street plan and zero for an unfinished
    ``FIVE_BIS``.  Summing them here would repeat it.

    The rule, stated so Rust can copy it:

    * where ``progress()`` has already aggregated the alternatives -- the best
      two of three streets for a decorative plan -- use ``steps_left``, which
      carries that aggregation;
    * where the plan binds to ONE street, read that street's vectors at the
      **cheapest alive street**, tie-broken by lowest index;
    * house-driven plans (``FULL_STREET``, ``EXTREMITIES``) need no effect at
      all -- they are covered by the number-rate term instead.

    SURVEYOR takes ``estate_steps_left``.  ⚠ That is legitimate **here** and
    forbidden in §6.2: this is a rate estimate, not the hard bound, and §6.2's
    objection is that one fence can create two matches at once, which makes
    ``estate_steps_left`` unsound as a *lower* bound but fine as a descriptor.
    """
    kind = plan.kind
    if kind is PlanKind.SEVEN_TEMP:
        return {Effect.TEMP: req.temps_needed}
    if kind is PlanKind.ESTATE:
        return {Effect.SURVEYOR: req.estate_steps_left}
    if kind is PlanKind.FIVE_BIS:
        alive = [req.bis_needed[x] for x in range(NUM_STREETS) if req.street_serves[x]]
        return {Effect.BIS: min(alive)} if alive else {}
    if kind in (PlanKind.FULL_STREET, PlanKind.EXTREMITIES):
        return {}
    if kind is PlanKind.COMPLETE_STREET:
        alive = [x for x in range(NUM_STREETS) if req.street_serves[x]]
        if not alive:
            return {}
        best = min(
            alive,
            key=lambda x: (
                req.parks_needed[x] + req.pools_needed[x] + req.roundabout_needed[x],
                x,
            ),
        )
        return {Effect.PARK: req.parks_needed[best], Effect.POOL: req.pools_needed[best]}
    if kind is PlanKind.DECORATIVE:
        what = plan.params[0]
        if what == "pool&park":
            x = plan.params[1]
            return {
                Effect.PARK: req.parks_needed[x],
                Effect.POOL: req.pools_needed[x],
            }
        # `park` / `pool`: best two of three, already aggregated by progress().
        return {Effect.PARK if what == "park" else Effect.POOL: steps_left}
    return {}


def _plan_block(
    state: GameState,
    viewer: int,
    seat: int,
    sheet: Sheet,
    slot: int,
    view: _DeckView,
    w: _Writer,
) -> None:
    """One plan slot's 34 floats (§3.4, without the demoted §6.4 pair)."""
    plan = PLANS[state.plan_ids[slot]]
    fraction, steps = progress(plan, sheet)
    banked = seat in state.plan_turns_for(viewer, slot)
    req = requirements(plan, sheet)

    w.put(fraction, min(steps, _STEPS_SCALE) / _STEPS_SCALE, float(banked))

    # sheet-wide requirement (2)
    w.put(
        _ratio(float(req.temps_needed), 7.0),
        # §9.4 `fences_needed`: steps_left for ESTATE kinds / 6, else 0.  A
        # DESCRIPTOR of remaining work -- §6.2 forbids it as an operand of the
        # hard bound and it is never used as one.
        _ratio(float(req.estate_steps_left), 6.0)
        if plan.kind is PlanKind.ESTATE
        else 0.0,
    )

    # estate shortfall, sizes 1..6 (6) -- §2.3
    for s in range(6):
        w.put(_ratio(float(req.estate_shortfall[s]), 6.0))

    # per-street requirement, 3 x 6 (18) -- §3.2
    for x in range(NUM_STREETS):
        w.put(
            _ratio(float(req.parks_needed[x]), float(PARK_BOXES[x])),
            _ratio(float(req.pools_needed[x]), 3.0),
            _ratio(float(req.houses_needed[x]), float(STREET_SIZES[x])),
            _ratio(float(req.bis_needed[x]), 5.0),
            float(req.roundabout_needed[x]),
            float(req.street_serves[x]),
        )

    # feasibility and speed (5)
    alive = feasible(plan, sheet)
    w.put(1.0 if alive else 0.0)
    w.put(_turns(float(turns_lower_bound(plan, sheet))))

    needs = _effect_needs(plan, req, steps)
    effect_turns_raw = 0.0
    for effect, marks in needs.items():
        if marks <= 0:
            continue
        rate = float(view.effect_rate[_EFFECT_INDEX[effect]])
        effect_turns_raw += marks / max(rate, EPS)
    w.put(_turns(effect_turns_raw))

    # §6.3 `number_rate_supply`: the fraction of the deck supplying this plan's
    # number needs.  A plan's number needs are its target boxes' gaps; a plan
    # with no box targets has no number need and supplies nothing.
    houses_total = sum(req.houses_needed)
    supply_numbers: set[int] = set()
    for x, y in req.target_boxes:
        bounds = sheet.gap_bounds(x, y)
        if bounds is None:
            continue
        _f, _l, low, high = bounds
        for n in CARD_NUMBERS:
            if low < n < high:
                supply_numbers.add(n)
    supply = sum(
        float(view.deck_numbers[dk.NUMBER_INDEX[n]]) for n in supply_numbers
    )
    number_rate_supply = _ratio(supply, view.deck_total)
    w.put(number_rate_supply)

    # §9.4: `expected_turns_to_plan` is the MAX of the two rate terms, not their
    # sum -- effects and numbers are consumed by the SAME combination, so the
    # binding constraint is whichever is slower.
    number_turns_raw = (
        houses_total / max(number_rate_supply, EPS) if houses_total else 0.0
    )
    w.put(_turns(max(effect_turns_raw, number_turns_raw)))



def _number_demand(sheet: Sheet) -> np.ndarray:
    """§7.2: the count of legal ``(empty box, value)`` pairs, per value 0..17.

    Pure combinatorics, no judgement.  Computed per gap: within a maximal empty
    run every box is bounded identically and the other boxes of the run are
    empty, so each of its ``L`` boxes accepts every value in its open interval.
    """
    demand = np.zeros(NUM_NUMBER_VALUES, dtype=np.float64)
    for x, size in enumerate(STREET_SIZES):
        y = 0
        while y < size:
            if sheet.numbers[x][y] is not None:
                y += 1
                continue
            bounds = sheet.gap_bounds(x, y)
            assert bounds is not None
            first, last, low, high = bounds
            length = last - first + 1
            lo = max(low + 1, MIN_NUMBER)
            hi = min(high - 1, MAX_NUMBER)
            if hi >= lo:
                demand[lo : hi + 1] += length
            y = last + 1
    return demand


def _estate_demand(sheet: Sheet) -> float:
    """§7.2: the scoring value of one more ESTATE mark, summed over all estates.

    ``sum_i count[i] * delta(i)`` where ``delta`` is the next mark's score step.

    ⚠ **R1 -- ``estate_size_counts()``, NOT the free counts.**  ``estate_score``
    multiplies the counts of **all** estates; a top fence stops a City Plan
    reusing an estate, it does not remove its real-estate points.

    ⚠ **R5 -- the saturation guard is required, not defensive.**
    ``ESTATE_ROW_SCORES[i]`` has ``ESTATE_ROW_BOXES[i] + 1`` entries and
    ``marks[i]`` reaches ``ESTATE_ROW_BOXES[i]``, so the unguarded index raises
    on every row once it saturates -- for size 1 that is a single mark.
    """
    counts = sheet.estate_size_counts()
    total = 0.0
    for i in range(6):
        marks = sheet.estate_marks[i]
        if marks >= ESTATE_ROW_BOXES[i]:
            continue
        delta = ESTATE_ROW_SCORES[i][marks + 1] - ESTATE_ROW_SCORES[i][marks]
        total += counts[i] * delta
    return total


def _effect_demand(state: GameState, sheet: Sheet, viewer: int, seat: int) -> np.ndarray:
    """§7.2: marks of each effect still wanted, normalised by its track size."""
    out = np.zeros(_NUM_EFFECTS, dtype=np.float64)

    remaining = {
        Effect.PARK: float(sum(PARK_BOXES) - sum(sheet.parks)),
        Effect.POOL: float(POOL_BOXES - sheet.pool_count),
        Effect.TEMP: float(TEMP_BOXES - sheet.temps),
        Effect.BIS: float(BIS_BOXES - sheet.bis_marks),
    }
    for effect, value in remaining.items():
        out[_EFFECT_INDEX[effect]] = _ratio(
            max(0.0, value), _EFFECT_TRACK_SIZE[effect]
        )

    # SURVEYOR is plan-derived: fences make the estate sizes plans ask for.
    fences = 0
    for slot, plan_id in enumerate(state.plan_ids):
        if seat in state.plan_turns_for(viewer, slot):
            continue
        plan = PLANS[plan_id]
        if plan.kind is PlanKind.ESTATE:
            fences += requirements(plan, sheet).estate_steps_left
    out[_EFFECT_INDEX[Effect.SURVEYOR]] = _ratio(
        float(fences), _SURVEYOR_DEMAND_SCALE
    )

    out[_EFFECT_INDEX[Effect.ESTATE]] = _ratio(
        _estate_demand(sheet), _ESTATE_DEMAND_SCALE
    )
    return out


def _card_demand(sheet: Sheet, with_temp: bool) -> np.ndarray:
    """§7.3: per printed card number 1..15, how many empty BOXES it could serve.

    ⚠ Counts of **boxes**, not of ``(box, value)`` pairs.  A box takes one value,
    so a card either can or cannot serve it; counting pairs would double-count a
    card that reaches the same box through two different temp deltas.
    """
    out = np.zeros(_NUM_NUMBERS, dtype=np.float64)
    deltas = TEMP_DELTAS if with_temp else (0,)
    for x, size in enumerate(STREET_SIZES):
        y = 0
        while y < size:
            if sheet.numbers[x][y] is not None:
                y += 1
                continue
            bounds = sheet.gap_bounds(x, y)
            assert bounds is not None
            first, last, low, high = bounds
            length = last - first + 1
            for i, n in enumerate(CARD_NUMBERS):
                for d in deltas:
                    v = min(max(n + d, MIN_NUMBER), MAX_NUMBER)
                    if low < v < high:
                        out[i] += length
                        break
            y = last + 1
    return out


def _reshuffle_contraction(
    sheet: Sheet, effect_demand: np.ndarray, view: _DeckView, w: _Writer
) -> None:
    """§7.3's 8 floats: does the post-reshuffle deck fit my holes better?

    ⚠ **SPEC GAP 2.**  §7.3 lays the block out as
    ``notemp/temp x number/effect x deck/reshuffled = 8``, but ``eff_rate`` does
    not read the card-demand vector at all, so the two effect entries of the
    temp half are **equal by construction** to those of the no-temp half.  The
    literal layout is kept -- the block width and the field order are what Rust
    must match, and silently emitting 6 would desynchronise the two
    implementations -- but those two floats carry no information.  Worth
    revisiting when the block is next opened.
    """
    def fit_rate(demand: np.ndarray, supply: np.ndarray) -> float:
        total = float(supply.sum())
        num = float((demand * supply).sum())
        return _ratio(num, _FIT_RATE_BOX_SCALE * total)

    def eff_rate(supply: np.ndarray) -> float:
        # ⚠ §9.4 R4: NORM_EFF is `sum(supply) * sum(demand)`, NOT `81 * sum
        # (demand)`.  Dividing by the full deck size makes the same sheet
        # against the same PROPORTIONS score lower simply because the deck has
        # drained -- a deck-size signal leaking into a rate.
        #
        # ⚠ §10.6: `effect_demand` is NOT integer-valued, so these two sums are
        # order-sensitive.  Plain left-to-right loops, never `ndarray.sum()`
        # (pairwise) or builtin `sum()` (compensated since Python 3.12) -- Rust
        # mirrors this exact order.
        demand_total = 0.0
        weighted = 0.0
        for e in range(_NUM_EFFECTS):
            demand_total += float(effect_demand[e])
            weighted += float(effect_demand[e]) * float(supply[e])
        norm = float(supply.sum()) * demand_total
        return _ratio(weighted, norm)

    for with_temp in (False, True):
        demand = _card_demand(sheet, with_temp)
        for supply_numbers, supply_effects in (
            (view.deck_numbers, view.deck_effects),
            (view.reshuffled_numbers, view.reshuffled_effects),
        ):
            w.put(fit_rate(demand, supply_numbers), eff_rate(supply_effects))


def _playable_sets(
    state: GameState, sheet: Sheet, view: _DeckView
) -> tuple[list[np.ndarray], np.ndarray]:
    """Per stack, which printed numbers would give it a legal write next turn.

    Returns ``(per-stack masks, printed-only mask)``.  Each mask is over
    :data:`CARD_NUMBERS` and holds 1.0 where the number is **unplayable**, which
    is the form §8's probabilities contract against.
    """
    miss: list[np.ndarray] = []
    for effect in view.next_effects:
        m = np.ones(_NUM_NUMBERS, dtype=np.float64)
        if effect is not None:
            for i, n in enumerate(CARD_NUMBERS):
                for v in state.numbers_for(n, effect):
                    if sheet.available_locations(v):
                        m[i] = 0.0
                        break
        miss.append(m)

    printed = np.ones(_NUM_NUMBERS, dtype=np.float64)
    for i, n in enumerate(CARD_NUMBERS):
        if sheet.available_locations(n):
            printed[i] = 0.0
    return miss, printed


def _joint_miss(view: _DeckView, masks: list[np.ndarray]) -> float:
    """P(every stack's number falls in its own miss set), over the ordered draw."""
    return view.draw_probability(masks)


def _best_roundabout_sheet(state: GameState, sheet: Sheet) -> Optional[Sheet]:
    """The sheet after the "best" single roundabout, or ``None`` if none is legal.

    ⚠ **SPEC GAP 3.**  §8 says "given the best legal roundabout placement" and
    does not say best *for what*.  Choosing the placement that minimises the
    refusal probability itself would be an optimisation over ~33 sites, each
    costing a full joint contraction, on every sheet of every encoded state.
    Resolved as: **the placement maximising total placement capacity, tie-broken
    by lowest ``(street, box)``** -- the same quantity
    :meth:`Sheet.capacity_if_roundabout` already maximises, so the refusal block
    and the ``roundabout_repair`` scalars agree about which roundabout they are
    talking about.
    """
    if not (state.config.advanced and sheet.can_build_roundabout()):
        return None
    if not sheet.has_free_box():
        return None
    best: Optional[Sheet] = None
    best_key = -1
    for x, size in enumerate(STREET_SIZES):
        for y in range(size):
            if sheet.numbers[x][y] is not None:
                continue
            candidate = sheet.copy()
            candidate.build_roundabout((x, y), turn=0)
            total = sum(candidate.placement_capacity())
            if total > best_key:
                best, best_key = candidate, total
    return best


def _refusal_block(
    state: GameState, sheet: Sheet, view: _DeckView, w: _Writer
) -> None:
    """§8's 5 floats.  A refusal is a CHOICE, so these are its factual inputs.

    The first three are exact for next turn, because next turn's effects are
    printed and known; beyond that the steady-state term uses §9.3's rate.  Not
    derivable from planes 14/15: those are per-gap, these are a **union across
    gaps with overlapping number ranges**, needing inclusion-exclusion -- which
    is what the ordered joint supplies.
    """
    miss, printed_miss = _playable_sets(state, sheet, view)
    w.put(_joint_miss(view, miss))

    after = _best_roundabout_sheet(state, sheet)
    if after is None:
        p_after = _joint_miss(view, miss)
        rescue = 0.0
    else:
        miss_after, _ = _playable_sets(state, after, view)
        p_after = _joint_miss(view, miss_after)
        # "would change playable_slots()": some stack that had no legal write
        # now has one.
        rescue = float(
            any(
                bool((a < b).any())
                for a, b in zip(miss_after, miss)
            )
        )
    w.put(p_after)

    # P(at least one stack's PRINTED number has nowhere to go) -- what opens the
    # voluntary refusal.  The printed number carries no temp widening.
    all_placeable = _joint_miss(view, [1.0 - printed_miss] * 3)
    w.put(min(1.0, max(0.0, 1.0 - all_placeable)))

    w.put(rescue)

    # p_forced_refusal_steady: the post-roundabout refusal at the steady-state
    # deck, where next turn's effects are NOT known.  Each of the three cards is
    # then an ordinary draw carrying its own printed effect, so "all three
    # unplayable" is a hypergeometric over the set of unplayable CARDS -- exact
    # without replacement, and not the banned `1 - (1-p)**3`.
    steady_sheet = after if after is not None else sheet
    matrix = view.deck_matrix
    total = float(matrix.sum())
    if total < 3.0:
        matrix = view.reshuffled_matrix
        total = float(matrix.sum())
    unplayable = 0.0
    for i, n in enumerate(CARD_NUMBERS):
        for effect in DECK_EFFECT_ORDER:
            count = float(matrix[i, _EFFECT_INDEX[effect]])
            if count <= 0.0:
                continue
            if not any(
                steady_sheet.available_locations(v)
                for v in state.numbers_for(n, effect)
            ):
                unplayable += count
    if total < 3.0:
        w.put(0.0)
    else:
        num = unplayable * max(unplayable - 1.0, 0.0) * max(unplayable - 2.0, 0.0)
        den = total * (total - 1.0) * (total - 2.0)
        w.put(_ratio(num, den))


def _selected_estates(plan: Plan, sheet: Sheet) -> Optional[list[tuple[int, int]]]:
    """§9.2a's canonical satisfying selection, or ``None`` if none exists.

    Process required sizes **descending**; for each, take the eligible free
    estate with the lowest ``(street, start)``.  Deterministic, no optimisation,
    identical in Rust.

    ⚠ Descending order is a **tie-break**, not an optimisation, and is not
    claimed optimal.  ``feasible`` remains the authority on reachability; this
    only says which boxes a completion would consume *if it happened*.
    """
    free = sorted(sheet.free_estates())  # (street, start, size)
    taken: list[tuple[int, int]] = []
    used: set[tuple[int, int, int]] = set()
    for size in sorted(plan.required_sizes, reverse=True):
        for est in free:
            if est in used or est[2] != size:
                continue
            used.add(est)
            x, start, length = est
            taken.extend((x, start + k) for k in range(length))
            break
        else:
            return None
    return taken


def _target_boxes(state: GameState, viewer: int, seat: int, sheet: Sheet, slot: int):
    """``T(slot)`` -- the boxes completing this plan would consume (§9.2a)."""
    if seat in state.plan_turns_for(viewer, slot):
        return []
    plan = PLANS[state.plan_ids[slot]]
    if plan.kind is PlanKind.FULL_STREET:
        x = plan.params[0]
        return [(x, y) for y in range(STREET_SIZES[x])]
    if plan.kind is PlanKind.EXTREMITIES:
        from games.welcome_to.constants import EXTREMITY_POSITIONS

        return list(EXTREMITY_POSITIONS)
    if plan.kind is PlanKind.ESTATE:
        return _selected_estates(plan, sheet) or []
    return []


def _plan_conflict_seat(
    state: GameState, viewer: int, seat: int, sheet: Sheet, w: _Writer
) -> None:
    """§9.2a: 3 unordered overlaps + 6 directed kills.

    ⚠ ``kills`` is **directed** -- completing *a* may kill *b* while completing
    *b* leaves *a* alive -- which is why this is 9 and not 6.
    """
    targets = [_target_boxes(state, viewer, seat, sheet, s) for s in range(3)]
    sets = [set(t) for t in targets]

    for a, b in ((0, 1), (0, 2), (1, 2)):
        if not sets[a] or not sets[b]:
            w.put(0.0)
            continue
        inter = len(sets[a] & sets[b])
        w.put(_ratio(float(inter), float(min(len(sets[a]), len(sets[b])))))

    for a, b in ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)):
        if not sets[a]:
            w.put(0.0)
            continue
        plan_b = PLANS[state.plan_ids[b]]
        if seat in state.plan_turns_for(viewer, b):
            w.put(0.0)
            continue
        hypothetical = sheet.copy()
        for x, y in sets[a]:
            hypothetical.top_fences[x][y] = True
        w.put(0.0 if feasible(plan_b, hypothetical) else 1.0)


def _sheet_scalars(
    state: GameState, viewer: int, seat: int, view: _DeckView
) -> np.ndarray:
    """One seat's 196 flat features, by the same function for every seat.

    Every read goes through a viewer-safe accessor -- ``sheet_for``,
    ``score_breakdown(..., viewer=)``, ``plan_turns_for`` -- so this is symmetric
    *and* information-set safe by the same construction.  Reaching for
    ``state.sheets[seat]`` here would break both at once, which is exactly what
    the symmetry test is for.
    """
    sheet = state.sheet_for(viewer, seat)
    w = _Writer(NUM_SHEET_SCALAR)

    # tracks (26)
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
    # ⚠ §9.4 R5: BOTH estate count vectors are `count / 4.0` and NEITHER is
    # clamped.  Clamping only the new one breaks the differencing that justifies
    # the shared scale, and it breaks it at five size-1 estates -- routine, and
    # exactly where estate plans live.
    for count in sheet.estate_size_counts():
        w.put(count / 4.0)
    for count in sheet.free_estate_size_counts():
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

    # §4: what one roundabout could repair (3), and remaining freedom (1)
    roundabout_open = state.config.advanced and sheet.can_build_roundabout()
    repair = sheet.capacity_if_roundabout(available=roundabout_open)
    for x in range(NUM_STREETS):
        w.put(_ratio(float(repair[x]), float(STREET_SIZES[x])))
    w.put(_ratio(float(sheet.total_span()), _TOTAL_SPAN_SCALE))

    # THE RACE (102) -- §3.4
    for slot in range(3):
        _plan_block(state, viewer, seat, sheet, slot, view, w)

    # demand (24) -- §7.2
    w.put_array(_number_demand(sheet) / float(NUM_BOXES))
    effect_demand = _effect_demand(state, sheet, viewer, seat)
    w.put_array(effect_demand)

    # reshuffle contraction (8) -- §7.3
    _reshuffle_contraction(sheet, effect_demand, view, w)

    # refusal and blocking (5) -- §8
    _refusal_block(state, sheet, view, w)

    # houses this turn (2) -- §8
    w.put(
        max_houses_this_turn(state, viewer, seat) / 3.0,
        1.0 if bis_usable(state, viewer, seat) else 0.0,
    )

    # where the real conflict lives (9) -- §9.2a
    _plan_conflict_seat(state, viewer, seat, sheet, w)

    # free boxes (1)
    written = sum(1 for row in sheet.numbers for n in row if n is not None)
    w.put((NUM_BOXES - written) / NUM_BOXES)

    w.put(1.0 if seat == viewer else 0.0)
    w.put(1.0)  # seat_valid; padded seats never reach this function

    assert w.pos == NUM_SHEET_SCALAR, f"wrote {w.pos}, expected {NUM_SHEET_SCALAR}"
    return w.buf


def _global_scalars(state: GameState, viewer: int, view: _DeckView) -> np.ndarray:
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
    deck = view.deck_matrix
    discard = dk.discard_composition(state, viewer)
    w.put(state.deck_remaining / NUM_BASE_CARDS, len(state.discard) / NUM_BASE_CARDS)
    w.put_array(deck.sum(axis=1) / 9.0)
    w.put_array(deck.sum(axis=0) / 20.0)
    w.put_array(discard.sum(axis=1) / 9.0)
    w.put_array(discard.sum(axis=0) / 20.0)
    w.put_array(dk.next_number_distribution(state, viewer))
    w.put_array(_normalise(view.reshuffled_matrix.sum(axis=1)))

    # §9.3: the boundary-draw rates.  `temp_` and `bis_availability_rate` are
    # the same quantity surfaced separately because §8's two blocks each need
    # exactly one of them.
    w.put_array(view.effect_rate)
    w.put(float(view.effect_rate[_EFFECT_INDEX[Effect.TEMP]]))
    w.put(float(view.effect_rate[_EFFECT_INDEX[Effect.BIS]]))

    # §7.4: how long the current deck's fit probabilities stay meaningful.
    w.put(_turns(float(dk.reveals_to_reform(state))))

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

    Raises ``ValueError`` outside the 2+ player standard game (§0.5).
    """
    _require_scope(state)
    viewer = state.actor if player is None else player
    sheet_planes = np.zeros(SHEET_PLANES_SHAPE, dtype=np.float32)
    sheet_scalars = np.zeros((MAX_SEATS, NUM_SHEET_SCALAR), dtype=np.float32)
    viewer_plane = np.zeros(VIEWER_PLANE_SHAPE, dtype=np.float32)

    view = _DeckView(state, viewer)
    base_numbers, all_numbers = _offered(state, viewer)
    for k, seat in enumerate(seat_order(state, viewer)):
        sheet = state.sheet_for(viewer, seat)
        _sheet_planes(
            state, viewer, seat, sheet, base_numbers, all_numbers, view,
            sheet_planes[k],
        )
        sheet_scalars[k] = _sheet_scalars(state, viewer, seat, view)

    _viewer_plane(state, viewer, viewer_plane)
    return sheet_planes, sheet_scalars, viewer_plane, _global_scalars(
        state, viewer, view
    )


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


def plan_slot_slice(slot: int) -> slice:
    """Where plan ``slot``'s 34 floats sit inside the ``plans`` block."""
    if not 0 <= slot < 3:
        raise ValueError(f"plan slot {slot} out of range")
    block = block_slice("plans")
    start = block.start + slot * PLAN_SLOT_WIDTH
    return slice(start, start + PLAN_SLOT_WIDTH)
