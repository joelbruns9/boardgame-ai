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

2. **Fixed shape.**  ``encode_state`` always returns ``(SPATIAL_SHAPE,
   NUM_SCALAR)`` regardless of player count, so one set of weights serves 2, 3
   and 4 player games.  Seats beyond :data:`MAX_OPPONENTS` are dropped from the
   opponent block; the seat count itself is a feature so the net knows.

3. **Sheet-shaped spatial planes.**  The three streets are laid out as a
   ``3 x 12`` grid, right-padded (street 0 is 10 long, street 1 is 11).  Plane 0
   is the validity mask for that padding.  There is no useful symmetry group
   here — streets are not interchangeable and the left-to-right ascending rule
   breaks reflection — so there is no augmentation hook, unlike Kingdomino.

4. **The two interactions are given first-class features.**  Welcome To is
   mostly played on your own sheet, and what is left is a *race*: whoever completes the three
   City Plans first usually wins, and the plan that pays 12 to the first
   finisher pays 7 to everyone after.  Completion flags alone cannot express
   "two parks away", so the plan block carries a distance-to-completion for the
   viewer *and for every opponent*, off their public sheets
   (:func:`games.welcome_to.plans.progress`).  The temp-agency majority is the
   second interaction and rides in the opponent block.

5. **Placement capacity is a feature, not something to be learned twice.**
   The strictly-ascending rule means a badly placed number destroys future
   placements wholesale — writing 15 into the first box of a street takes its
   capacity from 10 to 2.  That quantity is what separates competent play from
   noise in this game, so :meth:`~games.welcome_to.sheet.Sheet.placement_capacity`,
   :meth:`~games.welcome_to.sheet.Sheet.box_spans` and
   :meth:`~games.welcome_to.sheet.Sheet.positional_fit` are all exposed directly,
   as a flat block and as spatial planes.  On a one-ply bot those three terms are
   jointly worth about twelve points a game over scoring alone, which is the
   evidence for handing them over rather than hoping the trunk rediscovers them.

6. **Training-isolated.**  Nothing here imports a heuristic evaluator.
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
from games.welcome_to.plans import NUM_PLANS, PLANS, progress
from games.welcome_to.sheet import Sheet

#: Opponents encoded individually; extra seats fold into the plan-race features
#: rather than getting their own block.
MAX_OPPONENTS: int = 3
#: Width of the seat-index one-hot.
MAX_PLAYERS: int = 6

#: Planes describing the viewer's own sheet; opponent planes start after them.
OWN_PLANES = 12
_OWN_PLANES = OWN_PLANES
_OPP_PLANES = 2
NUM_PLANES: int = _OWN_PLANES + MAX_OPPONENTS * _OPP_PLANES
SPATIAL_SHAPE: tuple[int, int, int] = (NUM_PLANES, NUM_STREETS, MAX_STREET_LEN)

_NUM_EFFECTS = dk.NUM_EFFECTS   # 6
_NUM_NUMBERS = dk.NUM_NUMBERS   # 15
_EFFECT_INDEX = dk.EFFECT_INDEX
_SEATS = 1 + MAX_OPPONENTS

#: Named blocks of the flat feature vector, in order.  Kept explicit so a change
#: to one block is a visible change to :data:`NUM_SCALAR`.
SCALAR_BLOCKS: tuple[tuple[str, int], ...] = (
    ("phase", len(Phase)),
    ("turn", 1),
    ("own_tracks", 20),
    ("own_score", 9),
    ("stacks", 3 * (NUM_NUMBER_VALUES + _NUM_EFFECTS) + 6),
    ("chosen_combination", NUM_NUMBER_VALUES + _NUM_EFFECTS + 1),
    ("last_house", 34),
    ("pending_estate", 7),
    ("plans", 3 * (NUM_PLANS + 5)),
    ("plan_race", 3 * _SEATS * 2),
    ("reshuffle_race", 4),
    ("capacity", 4),
    ("next_effects", 3 * _NUM_EFFECTS),
    ("deck", 2 + 3 * _NUM_NUMBERS + 2 * _NUM_EFFECTS + _NUM_NUMBERS),
    ("opponents", MAX_OPPONENTS * 9),
    ("config", 4),
    ("seat", MAX_PLAYERS),
)
NUM_SCALAR: int = sum(size for _, size in SCALAR_BLOCKS)

def _normalise(vector: np.ndarray) -> np.ndarray:
    total = float(vector.sum())
    if total <= 0.0:
        return np.zeros_like(vector)
    return (vector / total).astype(np.float32)


_TURN_SCALE = 30.0
_SCORE_SCALE = 50.0
_STEPS_SCALE = 12.0


# ──────────────────────────────────────────────────────────────────────────
# Spatial planes
# ──────────────────────────────────────────────────────────────────────────
def _estate_size_grid(sheet: Sheet) -> np.ndarray:
    grid = np.zeros((NUM_STREETS, MAX_STREET_LEN), dtype=np.float32)
    for x, start, size in sheet.estates():
        grid[x, start : start + size] = size
    return grid


def _own_planes(state: GameState, sheet: Sheet, out: np.ndarray) -> None:
    writable: set[tuple[int, int]] = set()
    if state.phase is Phase.WRITE_NUMBER and state.ctx.number is not None:
        assert state.ctx.effect is not None
        for n in state.numbers_for(state.ctx.number, state.ctx.effect):
            writable.update(sheet.available_locations(n))
    elif state.phase is Phase.ROUNDABOUT_PLACE:
        writable.update(sheet.available_locations(None))

    estate_grid = _estate_size_grid(sheet)
    spans = sheet.box_spans()
    # the numbers this player could actually write right now: the one already
    # locked in if the combination is chosen, otherwise everything on offer
    if state.phase is Phase.WRITE_NUMBER and state.ctx.number is not None:
        assert state.ctx.effect is not None
        offered = state.numbers_for(state.ctx.number, state.ctx.effect)
    else:
        offered = [n for n, _ in state.visible_cards() if n is not None]
    for x, size in enumerate(STREET_SIZES):
        for y in range(size):
            n = sheet.numbers[x][y]
            out[0, x, y] = 1.0
            if n is not None:
                out[1, x, y] = 1.0
                if n == ROUNDABOUT:
                    out[4, x, y] = 1.0
                else:
                    out[2, x, y] = n / 17.0
            out[3, x, y] = float(sheet.is_bis[x][y])
            out[5, x, y] = float(sheet.top_fences[x][y])
            if y < size - 1:
                out[6, x, y] = float(sheet.fences[x][y])
            out[7, x, y] = float((x, y) in POOL_POSITION_SET)
            out[8, x, y] = float((x, y) in writable)
            # how many numbers could still land here at all
            out[10, x, y] = spans[x][y] / 18.0
            # and how well the number actually on offer would sit here
            if sheet.numbers[x][y] is None:
                fit = max(
                    (
                        sheet.positional_fit(n, x, y) or -99.0
                        for n in offered
                    ),
                    default=None,
                )
                if fit is not None and fit > -99.0:
                    out[11, x, y] = 1.0 / (1.0 - fit)
    out[9] = estate_grid / 6.0


def _opponent_planes(sheet: Sheet, out: np.ndarray) -> None:
    for x, size in enumerate(STREET_SIZES):
        for y in range(size):
            n = sheet.numbers[x][y]
            if n is None:
                continue
            out[0, x, y] = 1.0
            out[1, x, y] = 0.0 if n == ROUNDABOUT else n / 17.0


# ──────────────────────────────────────────────────────────────────────────
# Flat features
# ──────────────────────────────────────────────────────────────────────────
class _Writer:
    """Append-only cursor over the flat vector, checked against SCALAR_BLOCKS."""

    def __init__(self) -> None:
        self.buf = np.zeros(NUM_SCALAR, dtype=np.float32)
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


def _seat_order(state: GameState, player: int) -> list[int]:
    """Opponents in turn order starting after ``player``, capped."""
    return [
        (player + k) % state.config.players for k in range(1, state.config.players)
    ][:MAX_OPPONENTS]


def _scalar_features(state: GameState, player: int) -> np.ndarray:
    cfg = state.config
    sheet = state.sheets[player]
    w = _Writer()

    # phase, turn
    w.one_hot(int(state.phase), len(Phase))
    w.put(state.turn / _TURN_SCALE)

    # own tracks (20)
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

    # own score components (9)
    breakdown = state.score_breakdown(player, viewer=player)
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

    # the three stacks as the player sees them, plus which choices are playable
    for number, effect in state.visible_cards(player):
        w.one_hot(number, NUM_NUMBER_VALUES)
        w.one_hot(_EFFECT_INDEX.get(effect) if effect is not None else None, _NUM_EFFECTS)
    playable = set(state.playable_slots(player)) if player == state.actor else set()
    for slot in range(6):
        w.put(1.0 if slot in playable else 0.0)

    # the combination this player locked in, if any
    ctx = state.ctx if player == state.actor else None
    if ctx is not None and ctx.number is not None:
        w.one_hot(ctx.number, NUM_NUMBER_VALUES)
        w.one_hot(_EFFECT_INDEX.get(ctx.effect), _NUM_EFFECTS)
        w.put(1.0)
    else:
        w.skip(NUM_NUMBER_VALUES + _NUM_EFFECTS + 1)

    # the house written this turn
    if ctx is not None and ctx.last_house is not None:
        w.one_hot(box_index(*ctx.last_house), 33)
        w.put(1.0)
    else:
        w.skip(34)

    # the estate size a plan validation is currently waiting on
    if ctx is not None and ctx.pending_sizes:
        w.one_hot(ctx.pending_sizes[0] - 1, 6)
        w.put(1.0)
    else:
        w.skip(7)

    # the three City Plans and who has already banked them
    for slot, plan_id in enumerate(state.plan_ids):
        plan = PLANS[plan_id]
        w.one_hot(plan_id, NUM_PLANS)
        visible = state.plan_turns_for(player, slot)
        mine = player in visible
        others = any(p != player and p >= 0 for p in visible)
        first_gone = bool(visible) and any(
            t == min(visible.values()) for p, t in visible.items() if p != player
        )
        w.put(
            plan.scores[0] / 20.0,
            plan.scores[1] / 20.0,
            float(mine),
            float(others),
            0.0 if first_gone else 1.0,
        )

    # THE RACE: how far every seat is from each plan, off public sheets
    opponents = _seat_order(state, player)
    for slot, plan_id in enumerate(state.plan_ids):
        plan = PLANS[plan_id]
        for k in range(_SEATS):
            if k == 0:
                target = player
            elif k - 1 < len(opponents):
                target = opponents[k - 1]
            else:
                w.skip(2)
                continue
            fraction, steps = progress(plan, state.sheet_for(player, target))
            w.put(fraction, min(steps, _STEPS_SCALE) / _STEPS_SCALE)

    # THE THIRD RACE: whoever finishes the first plan chooses the reshuffle
    reshuffle_open = state._may_ask_reshuffle()
    my_plans = sum(1 for slot in range(3) if player in state.plan_turns_for(player, slot))
    opponent_plans = max(
        (
            sum(1 for slot in range(3) if opp in state.plan_turns_for(player, slot))
            for opp in opponents
        ),
        default=0,
    )
    w.put(
        1.0 if reshuffle_open else 0.0,
        1.0 if state.reshuffle_next_turn else 0.0,
        my_plans / 3.0,
        opponent_plans / 3.0,
    )

    # PLACEMENT CAPACITY: what a careless write burns
    capacity = sheet.placement_capacity()
    for x in range(NUM_STREETS):
        w.put(capacity[x] / STREET_SIZES[x])
    w.put(sum(capacity) / 33.0)

    # NEXT TURN'S EFFECTS: known now, one-hot per stack
    w.put_array(dk.known_next_effects(state, player))

    # WHAT IS COMING: the deck is exact bookkeeping, not an estimate
    deck = dk.deck_composition(state, player)
    discard = dk.discard_composition(state, player)
    reshuffled = dk.after_reshuffle_composition(state, player)
    w.put(state.deck_remaining / NUM_BASE_CARDS, len(state.discard) / NUM_BASE_CARDS)
    w.put_array(deck.sum(axis=1) / 9.0)
    w.put_array(deck.sum(axis=0) / 20.0)
    w.put_array(discard.sum(axis=1) / 9.0)
    w.put_array(discard.sum(axis=0) / 20.0)
    w.put_array(dk.next_number_distribution(state, player))
    w.put_array(_normalise(reshuffled.sum(axis=1)))

    # opponents
    scores = state.scores(viewer=player)
    for k in range(MAX_OPPONENTS):
        if k >= len(opponents):
            w.skip(9)
            continue
        opp = opponents[k]
        view = state.sheet_for(player, opp)
        built = sum(1 for row in view.numbers for n in row if n is not None)
        plans_done = sum(
            1 for slot in range(3) if opp in state.plan_turns_for(player, slot)
        )
        w.put(
            1.0,
            scores[opp] / 100.0,
            view.temps / TEMP_BOXES,
            view.permits / PERMIT_BOXES,
            plans_done / 3.0,
            built / 33.0,
            view.pool_count / POOL_BOXES,
            sum(view.parks) / 12.0,
            view.bis_marks / BIS_BOXES,
        )

    # configuration and seat
    w.put(
        float(cfg.advanced),
        float(cfg.expert),
        float(cfg.solo),
        cfg.players / MAX_PLAYERS,
    )
    w.one_hot(player if player < MAX_PLAYERS else None, MAX_PLAYERS)

    assert w.pos == NUM_SCALAR, f"wrote {w.pos} features, expected {NUM_SCALAR}"
    return w.buf


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────
def encode_state(
    state: GameState, player: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Encode ``state`` from ``player``'s point of view.

    Returns ``(spatial, scalar)`` with shapes :data:`SPATIAL_SHAPE` and
    ``(NUM_SCALAR,)``, both ``float32``.
    """
    player = state.actor if player is None else player
    spatial = np.zeros(SPATIAL_SHAPE, dtype=np.float32)
    _own_planes(state, state.sheets[player], spatial[:_OWN_PLANES])

    for k, opp in enumerate(_seat_order(state, player)):
        base = _OWN_PLANES + k * _OPP_PLANES
        _opponent_planes(
            state.sheet_for(player, opp), spatial[base : base + _OPP_PLANES]
        )

    return spatial, _scalar_features(state, player)


def encode_batch(
    states: list[GameState], players: Optional[list[int]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Stack :func:`encode_state` over a list of states."""
    if players is None:
        players = [s.actor for s in states]
    spatial = np.zeros((len(states), *SPATIAL_SHAPE), dtype=np.float32)
    scalar = np.zeros((len(states), NUM_SCALAR), dtype=np.float32)
    for i, (state, player) in enumerate(zip(states, players)):
        spatial[i], scalar[i] = encode_state(state, player)
    return spatial, scalar


def block_slice(name: str) -> slice:
    """Where a named block of :data:`SCALAR_BLOCKS` lives in the flat vector.

    Useful for probing a trained model ("does it use the next-reveal posterior?")
    and for ablations.
    """
    cursor = 0
    for block, size in SCALAR_BLOCKS:
        if block == name:
            return slice(cursor, cursor + size)
        cursor += size
    raise KeyError(f"no scalar block named {name!r}")
