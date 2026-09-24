"""Feature encoding for the Can't Stop value net.

Only **end-of-turn boards** are encoded. That is all the net ever sees: the
turn solver hands it the board as it stands after a stop or a bust, so there
are never runners on the board and ``state.active_player`` is already the
seat about to move. See ``VARIANT_SOLVER_PLAN.md`` Phase 2.

Deliberately **no probability features**. The solver computes the dice
exactly over the 126 four-dice multisets, so handing the net an approximation
of its own job is duplication; and per-column difficulty (7 easy, 2 and 12
hard) is a *constant* for each feature slot, which the first layer absorbs
for free. The net carries strategy, the solver carries dice.

Layout, ``FEATURE_SIZE`` floats:

    per seat, 4 padded slots x PER_SEAT features, seat-relative
    (slot 0 is the seat to move, slot k is (active + k) % num_players):
        11  saved progress / column height
        11  claimed-by-this-seat flags
         1  columns still needed to win, / 5 (a CONSTANT, not / need)
         1  seat-present flag
    global:
         1  blocking flag
         1  columns_to_win / 5

Seat-relative padding is what lets one net cover 2, 3 and 4 players: an
absent seat is all zeros including its present-flag, so player count is
encoded by which slots are live rather than by a separate feature.
``columns still needed`` is what lets one net cover base and extended
columns: "two columns from winning" means the same thing whether the rule
set needs 3 or 5, so it is scaled by a constant rather than by the rule
set's own threshold (which would undo the bridging). The ``columns_to_win``
global carries the threshold itself.
"""

import numpy as np

from .engine import COLUMNS, COLUMN_HEIGHTS

MAX_SEATS = 4
NUM_COLUMNS = len(COLUMNS)

# per seat: progress, claimed, columns-needed, present
PER_SEAT = NUM_COLUMNS + NUM_COLUMNS + 1 + 1
NUM_GLOBAL = 2
FEATURE_SIZE = MAX_SEATS * PER_SEAT + NUM_GLOBAL

# Largest columns_to_win over every rule set. Both the columns-needed feature
# and the columns_to_win global are scaled by this CONSTANT, never by the rule
# set's own threshold: dividing by ``need`` would make "two columns from
# winning" encode as 2/3 in a base game and 2/5 in an extended one, which is
# precisely the cross-variant meaning the feature exists to preserve. The
# global below supplies the threshold separately, so the net can still tell
# "two away out of three" from "two away out of five".
_MAX_COLUMNS_TO_WIN = 5


def seat_order(state):
    """Absolute seat indices in encoding order: the seat to move first."""
    n = state.rules.num_players
    a = state.active_player
    return [(a + k) % n for k in range(n)]


def encode_board(state, out=None):
    """Encode one end-of-turn board into a float32 vector.

    ``out`` may be a writable row to fill in place (used by ``encode_batch``).

    Intermediate arithmetic is done in Python floats (float64) and only then
    assigned into the float32 array, mirroring numpy's float64 -> float32
    array-assignment cast. The Rust port must reproduce that same cast chain
    to stay bit-exact -- see the Kingdomino port notes.
    """
    if state.runners:
        raise ValueError(
            "encode_board expects an end-of-turn board (no runners); got "
            f"{len(state.runners)} runner(s). Encode after stop() or bust().")

    if out is None:
        out = np.zeros(FEATURE_SIZE, dtype=np.float32)
    else:
        out[:] = 0.0

    rules = state.rules
    need = rules.columns_to_win

    for slot, seat in enumerate(seat_order(state)):
        base = slot * PER_SEAT
        progress = state.progress[seat]
        claimed = 0
        for i, col in enumerate(COLUMNS):
            if state.claimed_by[col] == seat:
                out[base + NUM_COLUMNS + i] = 1.0
                claimed += 1
            # Progress is zeroed on claimed columns by stop(), so a claimed
            # column contributes only its flag. No branch needed here.
            pos = progress[col]
            if pos:
                out[base + i] = pos / COLUMN_HEIGHTS[col]
        out[base + 2 * NUM_COLUMNS] = (max(need - claimed, 0)
                                       / _MAX_COLUMNS_TO_WIN)
        out[base + 2 * NUM_COLUMNS + 1] = 1.0

    g = MAX_SEATS * PER_SEAT
    out[g] = 1.0 if rules.blocking else 0.0
    out[g + 1] = need / _MAX_COLUMNS_TO_WIN
    return out


def encode_batch(states):
    """Encode many boards into one (N, FEATURE_SIZE) float32 array."""
    batch = np.zeros((len(states), FEATURE_SIZE), dtype=np.float32)
    for i, s in enumerate(states):
        encode_board(s, out=batch[i])
    return batch


def to_absolute(relative, state):
    """Seat-relative win probabilities -> absolute seat order.

    The net emits MAX_SEATS values in encoding order (slot 0 = the seat to
    move). The solver's evaluator contract, which ``ProgressHeuristic``
    defines, is instead indexed by absolute seat over ``num_players``. This
    slices the live seats, puts them back in absolute order and renormalizes.

    Accepts one row (MAX_SEATS,) or a batch (N, MAX_SEATS); the batch form
    requires every state to share a rule set and seat to move, which is true
    of the leaves of any single turn solve.
    """
    relative = np.asarray(relative, dtype=np.float64)
    n = state.rules.num_players
    order = seat_order(state)

    single = relative.ndim == 1
    rel = relative[None, :] if single else relative
    if rel.shape[1] != MAX_SEATS:
        raise ValueError(f"expected {MAX_SEATS} seat outputs, got {rel.shape[1]}")

    out = np.empty((rel.shape[0], n), dtype=np.float64)
    for slot, seat in enumerate(order):
        out[:, seat] = rel[:, slot]
    total = out.sum(axis=1, keepdims=True)
    if not np.all(total > 0):
        raise ValueError("live seats carry no probability mass")
    out /= total
    return out[0] if single else out
