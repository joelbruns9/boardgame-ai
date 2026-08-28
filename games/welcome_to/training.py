"""
Self-play training support: auxiliary targets and diversity measurement.

This module holds nothing that plays the game.  It exists to answer the two
questions that make Welcome To awkward to train on.

────────────────────────────────────────────────────────────────────────────
1. EARLY PLACEMENTS, LATE CONSEQUENCES
────────────────────────────────────────────────────────────────────────────
Writing a 15 into the first box of a street on turn 3 costs nine future
placements.  Your score on turn 3 is unchanged.  The bill arrives around turn 20
as permit refusals and as parks, pools and estates you never got to build.  No
search reaches that far — a 25-turn horizon with chance at every turn boundary is
out of reach at any realistic simulation count — so the *value function* has to
carry it, and a single scalar "final score" target is a thin, high-variance way to
teach it.

The fix is to decompose the target.  :func:`final_outcomes` reads, off a finished
game, the things whose causes are local and whose values are late:

* ``permits`` — the direct, low-noise signature of capacity mismanagement.  A
  model that can predict "this sheet will end up taking two refusals" has learned
  the thing that is hard to learn, and it can learn it from turn 3 onwards.
* ``houses`` and ``capacity_left`` — did this sheet get filled or did it seize up?
* per-component scores — eight gradients instead of one, each attributable to a
  different part of the sheet.
* plan outcomes — whether each plan completes, whether it completes first, how
  long completion takes, and which independent terminal clause ends the game.

These are auxiliary heads, in the KataGo sense: they are never consulted at play
time, they exist to force the trunk to represent the long horizon.  Combined with
the capacity features the encoder already supplies, the network is asked to learn
what capacity is *worth* rather than to rediscover the arithmetic of the
ascending rule.

Two properties of the target set are load-bearing rather than incidental, and
both are there because the training corpus mixes 2-, 3- and 4-seat games.  The
outcome is a **distribution over finishing positions**, not a rank index, which
is the only framing that means the same thing at every table size; and every
target is **normalised**, so that a shared loss with per-group weights is
actually weighted by those weights and not by the accident that scores are two
orders of magnitude larger than permits.  ``AUX_TARGETS_SPEC.md`` is the spec of
record for both, and for the heads that read them.

────────────────────────────────────────────────────────────────────────────
2. SELF-PLAY DIVERGENCE
────────────────────────────────────────────────────────────────────────────
In standard mode every seat sees the same three combinations.  A deterministic
policy playing itself from identical empty sheets therefore produces identical
sheets, forever: the game degenerates and the seats are worth one sample between
them.

What saves it is that divergence is *self-amplifying* — one differing choice makes
the sheets differ, and every later decision is then taken in a different state, so
the trajectories never re-converge.  The whole problem is therefore concentrated
in the opening.  :func:`sheet_divergence` measures whether it is actually
happening, and should be logged every iteration; if it decays as the policy
sharpens, the sampling temperature is too low too early.

The other half of the answer is an opponent *pool*: two different policies cannot
mirror each other, so a learner playing frozen checkpoints or GreedyBot has no
collapse problem at all.  Symmetric self-play, where the same policy fills every
seat, is the only configuration that needs the guards above.

Dropping to a single seat is **not** an available answer, tempting as it looks:
one seat costs the same per learner trajectory as two, and it switches scoring
rules (``TEMP_SOLO_SCORE`` for the 7/4/1 ranking, every City Plan paying its
first-place value), so the data describes a different game.  See
``SELF_PLAY_PLAN.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence

from games.welcome_to.constants import NUM_BOXES, PERMIT_BOXES
from games.welcome_to.game import GameState
from games.welcome_to.sheet import SheetScore

#: Target value for "this player never completed that plan".  Pair every
#: turns-to-finish target with its mask rather than regressing on the sentinel.
NEVER: int = -1

#: Seats a sample carries targets for, padded.  Must equal
#: :data:`games.welcome_to.encoder.MAX_SEATS` -- the seat axis of the targets is
#: the seat axis of the encoding, or the per-seat head is reading one seat and
#: being scored against another.  A test asserts it; this module does not import
#: the encoder, because targets are a property of the game, not of an encoding.
MAX_SEATS: int = 4

#: How many finishing positions the rank distribution spans.  The same number by
#: construction: a table of ``n`` seats has exactly ``n`` finishing positions.
#: A fifth seat has nowhere to be represented on the input side either, so
#: :func:`final_outcomes` refuses the game rather than emitting a target the
#: network cannot be asked about.
MAX_RANKS: int = MAX_SEATS

# -- target scales -------------------------------------------------------
# Every emitted target is divided by one of these, so that a shared MSE with
# per-group weights means what the weights say it means.  Raw targets would let
# ``score`` (~75) dominate ``permits`` (0-3) by construction, and the permits
# head is the one this module exists for.  The divisors are round numbers just
# past the realistic maximum rather than measured maxima: a target that
# occasionally lands at 1.1 is harmless, a scale that shifts between runs is
# not.  A head reads its target back by multiplying.
SCORE_SCALE: float = 80.0
PERMIT_SCALE: float = 3.0
TURN_SCALE: float = 25.0
BOX_SCALE: float = float(NUM_BOXES)
PLAN_SCALE: float = 3.0


@dataclass(frozen=True, slots=True)
class PlayerOutcome:
    """Everything a finished game says about one seat."""

    player: int
    score: int
    components: SheetScore
    permits: int
    houses: int
    capacity_left: int
    #: Absolute turn on which this player completed each plan slot, or ``None``.
    plan_turns: tuple[Optional[int], ...]
    #: Whether each completed plan tied for the earliest completion turn.
    plan_first: tuple[bool, ...]
    plans_completed: int
    #: Independent terminal clauses. More than one may be true on the final turn.
    end_full_sheet: bool
    end_all_plans: bool
    end_max_permit: bool
    final_turn: int
    #: How many seats were at the table.  Carried on the outcome because the
    #: seat-count-dependent targets below need it and the state is gone by then.
    num_seats: int
    #: ``P(this seat finishes rank r)`` for ``r`` in ``0 .. MAX_RANKS - 1``,
    #: zero beyond the seat count.  See :func:`rank_distributions`.
    rank_distribution: tuple[float, ...]


def rank_distributions(state: GameState) -> list[tuple[float, ...]]:
    """Each seat's distribution over finishing positions, ties shared evenly.

    A **0-indexed rank is not seat-count-invariant**: ``1.0`` means "last of
    two" in a 2p game and "second of four" in a 4p game, and over a mixed-seat
    corpus a scalar head learns the average of two incompatible meanings.  A
    distribution over the four positions is invariant, collapses to a Bernoulli
    over win/lose at two seats, and -- the part a scalar cannot do -- carries
    its own uncertainty, which the value gate needs.

    Ties **share their positions evenly**: two seats tied for first in a 3p game
    each get ``(0.5, 0.5, 0.0, 0.0)``.  Sorting instead would hand rank 0 to
    whichever seat comes first in seat order, making the target depend on seat
    index, and it would disagree with :meth:`GameState.winners`, which returns
    every tied seat.  Under this convention ``2 * E[u] - 1`` reproduces
    :meth:`GameState.returns` exactly at two seats, ties included.
    """
    n = state.config.players
    if n > MAX_RANKS:
        raise ValueError(
            f"{n} seats exceeds MAX_RANKS={MAX_RANKS}; the encoder cannot "
            "represent that table either"
        )
    scores = state.scores()
    keys = [(scores[p], state.sheets[p].tiebreak_key()) for p in range(n)]
    order = sorted(range(n), key=lambda p: keys[p], reverse=True)

    out = [[0.0] * MAX_RANKS for _ in range(n)]
    lo = 0
    while lo < n:
        hi = lo
        while hi < n and keys[order[hi]] == keys[order[lo]]:
            hi += 1
        share = 1.0 / (hi - lo)
        for seat in order[lo:hi]:
            for rank in range(lo, hi):
                out[seat][rank] = share
        lo = hi
    return [tuple(row) for row in out]


def rank_utility(num_seats: int) -> tuple[float, ...]:
    """``u_r = (n - 1 - r) / (n - 1)`` -- 1.0 for first, 0.0 for last, any table.

    The utility is deliberately *not* baked into the head.  The head predicts
    what will happen; this says what we want, and applying it afterwards means
    the objective can be A/B'd on a frozen checkpoint -- ``[1, 0, 0, 0]`` for
    pure win probability, this for arena-rating-like linear rank -- instead of
    in two training runs.
    """
    if num_seats <= 1:
        return (1.0,)
    return tuple((num_seats - 1 - r) / (num_seats - 1) for r in range(num_seats))


def rank_value(distribution: Sequence[float], num_seats: int) -> float:
    """``2 * E[u] - 1``: the rank objective on the same [-1, 1] scale as a win.

    At two seats this is exactly :meth:`GameState.returns` for the seat, which
    is what makes the distribution a drop-in replacement for the old ``won``
    target rather than a second, disagreeing opinion about ties.
    """
    utility = rank_utility(num_seats)
    return 2.0 * sum(p * u for p, u in zip(distribution, utility)) - 1.0


def masked_mean(errors, mask):
    """``sum(mask * errors) / sum(mask)`` -- the only correct masked reduction.

    Normalising by the *batch* size instead silently discounts rare events in
    proportion to their rarity.  A target whose mask is 1 in 14% of samples then
    takes a hidden 7x penalty, which looks exactly like "that head just doesn't
    learn" while nothing anywhere reports an error.

    Written against ``*``, ``.sum()`` and broadcasting alone so that the torch
    loss and any numpy check are the *same* function rather than two
    implementations of one convention, free to drift apart.  The empty-mask case
    divides by one instead of branching, which keeps it safe under tracing.
    """
    count = mask.sum()
    return (mask * errors).sum() / (count + (count == 0))


def final_outcomes(state: GameState) -> list[PlayerOutcome]:
    """Read the per-seat training targets off a finished game."""
    if not state.is_terminal:
        raise ValueError("outcomes are only defined for a finished game")

    scores = state.scores()
    distributions = rank_distributions(state)
    first_plan_turns = tuple(
        min(turns.values()) if turns else None for turns in state.plan_turns
    )

    out: list[PlayerOutcome] = []
    for player in range(state.config.players):
        sheet = state.sheets[player]
        turns = tuple(state.plan_turns[slot].get(player) for slot in range(3))
        first = tuple(
            completed is not None and completed == first_plan_turns[slot]
            for slot, completed in enumerate(turns)
        )
        out.append(
            PlayerOutcome(
                player=player,
                score=scores[player],
                components=state.score_breakdown(player),
                permits=sheet.permits,
                houses=sum(1 for row in sheet.numbers for n in row if n is not None),
                capacity_left=sum(sheet.placement_capacity()),
                plan_turns=turns,
                plan_first=first,
                plans_completed=sum(1 for t in turns if t is not None),
                end_full_sheet=not sheet.has_free_box(),
                end_all_plans=all(t is not None for t in turns),
                end_max_permit=sheet.permits >= PERMIT_BOXES,
                final_turn=state.turn,
                num_seats=state.config.players,
                rank_distribution=distributions[player],
            )
        )
    return out


#: Masked targets, as ``target -> mask``.  Every loss over one of these reduces
#: with :func:`masked_mean`, and :data:`NEVER` must never reach it.
MASKED_TARGETS: dict[str, str] = {
    **{
        f"turns_to_plan_{slot}": f"turns_to_plan_{slot}_mask"
        for slot in range(3)
    },
    **{f"plan_{slot}_first": f"plan_{slot}_first_mask" for slot in range(3)},
}

#: Per-seat Bernoulli targets. Network outputs for these names are raw logits;
#: training uses BCE-with-logits and evaluation reports calibrated probabilities.
BINARY_TARGETS: tuple[str, ...] = (
    *(f"will_complete_plan_{slot}" for slot in range(3)),
    *(f"plan_{slot}_first" for slot in range(3)),
    "end_trigger_full_sheet",
    "end_trigger_all_plans",
    "end_trigger_max_permit",
)


def _seat_targets(outcome: PlayerOutcome, turn: int) -> dict[str, float]:
    """One seat's slice of the target set.

    Everything here is read off the *terminal* state, so it is a property of the
    seat rather than a viewer-relative one: the turn-start snapshot that hides an
    opponent's current turn has nothing left to hide in a finished game.
    """
    targets: dict[str, float] = {
        "score": outcome.score / SCORE_SCALE,
        "permits": outcome.permits / PERMIT_SCALE,
        "houses": outcome.houses / BOX_SCALE,
        "capacity_left": outcome.capacity_left / BOX_SCALE,
        "plans_completed": outcome.plans_completed / PLAN_SCALE,
        "score_parks": outcome.components.parks / SCORE_SCALE,
        "score_pools": outcome.components.pools / SCORE_SCALE,
        "score_estates": outcome.components.estates / SCORE_SCALE,
        "score_plans": outcome.components.plans / SCORE_SCALE,
        "score_temp": outcome.components.temp / SCORE_SCALE,
        "score_bis": outcome.components.bis / SCORE_SCALE,
        "score_permits": outcome.components.permits / SCORE_SCALE,
        "score_roundabouts": outcome.components.roundabouts / SCORE_SCALE,
    }
    for slot, completed_on in enumerate(outcome.plan_turns):
        if completed_on is None:
            targets[f"turns_to_plan_{slot}"] = float(NEVER)
            targets[f"turns_to_plan_{slot}_mask"] = 0.0
        else:
            targets[f"turns_to_plan_{slot}"] = max(0, completed_on - turn) / TURN_SCALE
            targets[f"turns_to_plan_{slot}_mask"] = 1.0
    for slot, completed_on in enumerate(outcome.plan_turns):
        targets[f"will_complete_plan_{slot}"] = float(completed_on is not None)
    for slot, completed_on in enumerate(outcome.plan_turns):
        if completed_on is None:
            targets[f"plan_{slot}_first"] = float(NEVER)
            targets[f"plan_{slot}_first_mask"] = 0.0
        else:
            targets[f"plan_{slot}_first"] = float(outcome.plan_first[slot])
            targets[f"plan_{slot}_first_mask"] = 1.0
    targets.update(
        {
            "end_trigger_full_sheet": float(outcome.end_full_sheet),
            "end_trigger_all_plans": float(outcome.end_all_plans),
            "end_trigger_max_permit": float(outcome.end_max_permit),
        }
    )
    #: 0.0 on a padded seat; see :data:`_PADDED_SEAT`.
    targets["seat_valid"] = 1.0
    return targets


_PROBE = PlayerOutcome(
    player=0,
    score=0,
    components=SheetScore(),
    permits=0,
    houses=0,
    capacity_left=0,
    plan_turns=(None, None, None),
    plan_first=(False, False, False),
    plans_completed=0,
    end_full_sheet=False,
    end_all_plans=False,
    end_max_permit=False,
    final_turn=1,
    num_seats=2,
    rank_distribution=(0.0,) * MAX_RANKS,
)

#: Targets carried once per seat, as a tuple along the seat axis.  Derived from
#: the builder rather than listed, so the two cannot drift.
PER_SEAT_TARGETS: tuple[str, ...] = tuple(_seat_targets(_PROBE, turn=1))

#: Version-1 WTS shards ended at ``seat_valid`` and predate the dense plan
#: outcome heads. Readers use this exact order to upgrade those immutable rows.
LEGACY_PER_SEAT_TARGETS: tuple[str, ...] = (
    "score",
    "permits",
    "houses",
    "capacity_left",
    "plans_completed",
    "score_parks",
    "score_pools",
    "score_estates",
    "score_plans",
    "score_temp",
    "score_bis",
    "score_permits",
    "score_roundabouts",
    "turns_to_plan_0",
    "turns_to_plan_0_mask",
    "turns_to_plan_1",
    "turns_to_plan_1_mask",
    "turns_to_plan_2",
    "turns_to_plan_2_mask",
    "seat_valid",
)
assert set(LEGACY_PER_SEAT_TARGETS).issubset(PER_SEAT_TARGETS)

#: An absent seat.  Every value zero **except** the plan sentinels, which stay
#: :data:`NEVER` behind their zero mask, and ``seat_valid``, which is the flag
#: saying none of it counts.  Zero is a value; absent is not, and a shared
#: per-seat head taught that a nonexistent player scores zero carries that error
#: into the real seats.
_PADDED_SEAT: dict[str, float] = {
    name: (float(NEVER) if name in MASKED_TARGETS else 0.0)
    for name in PER_SEAT_TARGETS
}

#: Targets carried once per sample.  ``rank_p_*`` and the per-seat ``score`` are
#: the real value heads; the rest are there to shape the trunk.
GLOBAL_TARGETS: tuple[str, ...] = ("turns_left",) + tuple(
    name
    for rank in range(MAX_RANKS)
    for name in (f"rank_p_{rank}", f"rank_mask_{rank}")
)

#: Every target name a sample carries, global first.
TARGET_NAMES: tuple[str, ...] = GLOBAL_TARGETS + PER_SEAT_TARGETS


def sample_targets(
    outcomes: Sequence[PlayerOutcome], order: Sequence[int], turn: int
) -> dict[str, float | tuple[float, ...]]:
    """Targets for one visited state, given the turn it was visited on.

    ``outcomes`` is the whole table, indexed by seat as :func:`final_outcomes`
    returns it.  ``order`` is the **seat axis** -- the viewer first, then turn
    order, exactly :func:`games.welcome_to.encoder.seat_order`.  Per-seat targets
    come back as tuples along that axis, padded to :data:`MAX_SEATS`, so that
    target ``k`` and encoded seat ``k`` are the same player.  Getting that
    alignment wrong is silent: every shape still matches, and the shared per-seat
    head simply learns the average of four seats.

    The viewer decides only the *order*, and which seat the global rank
    distribution belongs to.  The per-seat values themselves are not
    viewer-relative -- see :func:`_seat_targets`.

    ``turns_to_plan_*`` are relative to ``turn`` so the head predicts "how many
    turns from here", which is the quantity that is comparable across a game.
    Each comes with a ``_mask`` companion that is 0 when the plan was never
    completed; train the head only where the mask is 1, and reduce with
    :func:`masked_mean`.

    ``rank_p_*`` is a distribution over the **viewer's** finishing position, and
    its ``rank_mask_*`` companion is unlike the plan masks: it belongs on the
    **logits**, before the softmax, not on the loss term.  Softmaxing over four
    classes and zeroing the dead ones afterwards leaves the live classes summing
    to less than one, which corrupts the expected utility and its variance in the
    same direction -- so the damage would not even look like noise.
    """
    if not order:
        raise ValueError("the seat axis must contain at least the viewer")
    if len(order) > MAX_SEATS:
        raise ValueError(f"seat axis of {len(order)} exceeds MAX_SEATS={MAX_SEATS}")
    per_seat = [_seat_targets(outcomes[seat], turn) for seat in order]
    per_seat += [_PADDED_SEAT] * (MAX_SEATS - len(per_seat))

    viewer = outcomes[order[0]]
    targets: dict[str, float | tuple[float, ...]] = {
        "turns_left": max(0, viewer.final_turn - turn) / TURN_SCALE,
    }
    for rank in range(MAX_RANKS):
        targets[f"rank_p_{rank}"] = viewer.rank_distribution[rank]
        targets[f"rank_mask_{rank}"] = 1.0 if rank < viewer.num_seats else 0.0
    for name in PER_SEAT_TARGETS:
        targets[name] = tuple(seat[name] for seat in per_seat)
    return targets



# ──────────────────────────────────────────────────────────────────────────
# Diversity
# ──────────────────────────────────────────────────────────────────────────
def sheet_divergence(state: GameState) -> float:
    """Mean fraction of house boxes on which two seats differ.

    ``0.0`` means every seat wrote exactly the same sheet — the degenerate case
    that symmetric self-play falls into when the policy is confident and the
    sampling temperature is low.  Log this per self-play iteration; it is the
    canary for the whole problem.

    Returns ``0.0`` for a single-seat game, where the question does not arise.
    """
    players = state.config.players
    if players < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for a, b in combinations(range(players), 2):
        left, right = state.sheets[a], state.sheets[b]
        differing = sum(
            1
            for x in range(3)
            for y in range(len(left.numbers[x]))
            if left.numbers[x][y] != right.numbers[x][y]
        )
        total += differing / NUM_BOXES
        pairs += 1
    return total / pairs


def first_divergence_turn(state: GameState) -> Optional[int]:
    """The earliest turn on which any two seats wrote a different box.

    ``None`` if the seats never diverged.  Divergence is self-amplifying, so this
    is the number that matters: if it is not small, the opening temperature is
    doing its job.
    """
    players = state.config.players
    if players < 2:
        return None
    earliest: Optional[int] = None
    for a, b in combinations(range(players), 2):
        left, right = state.sheets[a], state.sheets[b]
        for x in range(3):
            for y in range(len(left.numbers[x])):
                if left.numbers[x][y] == right.numbers[x][y]:
                    continue
                turns = [
                    t
                    for t in (left.written_turn[x][y], right.written_turn[x][y])
                    if t >= 0
                ]
                if not turns:
                    continue
                candidate = min(turns)
                if earliest is None or candidate < earliest:
                    earliest = candidate
    return earliest


def diversity_report(states: list[GameState]) -> dict[str, float]:
    """Summarise divergence over a batch of finished self-play games."""
    if not states:
        return {}
    divergences = [sheet_divergence(s) for s in states]
    firsts = [first_divergence_turn(s) for s in states]
    seen = [t for t in firsts if t is not None]
    spreads = [max(s.scores()) - min(s.scores()) for s in states]
    return {
        "games": float(len(states)),
        "mean_sheet_divergence": sum(divergences) / len(divergences),
        "identical_games": sum(1 for d in divergences if d == 0.0) / len(divergences),
        "mean_first_divergence_turn": (sum(seen) / len(seen)) if seen else float("nan"),
        "mean_score_spread": sum(spreads) / len(spreads),
    }
