"""
Exact card counting.

WHAT IS ACTUALLY HIDDEN
───────────────────────
Less than it first appears. A construction card carries a number on one face and
an effect on the other, and the number face *also* prints the effect from its own
back in two corners (``.top-right-corner`` / ``.bottom-left-corner`` in
``wtoCards.scss``, keyed on ``data-action``; BGA sends the client full card rows
for both cards in a stack). So every card on the table is fully identified:

* the card flipped aside shows its effect, and showed its number last turn;
* the card on top of the stack shows its number, and prints its own effect.

Which means the effect each stack will offer **next** turn is not a posterior at
all — it is known, now, with certainty. :func:`known_next_effects` returns it and
the encoder feeds it to the network. The only thing genuinely unknown is what
number is coming, because that lives on the card underneath, still buried.

WHAT THAT LEAVES TO COUNT
─────────────────────────
The deck's *composition* is exact public bookkeeping: the printed 81 cards, minus
the discard pile, minus the six cards on the table. Every one of those has been
seen in full. So :func:`deck_composition` is not an estimate — it is the deck, as
a ``(number, effect)`` histogram — and :func:`next_number_distribution` is the
exact distribution of the next number each stack will show.

That distribution sharpens all game long, which is the edge: early on the next
number is nearly uniform over 1..15 weighted by the printed multiplicities (8 and
9 are the most common at nine copies each, 1 and 2 the rarest at three), and by
the back half of the deck a counting player knows which numbers are gone.

The joint histogram matters as well as the marginals, for two reasons. It is what
makes the numbers-versus-effects correlation usable — numbers 1, 2, 5, 11, 14 and
15 carry no POOL, TEMP or BIS card, 3 and 13 carry no PARK or ESTATE — and it is
what makes the reshuffle decision (see :func:`after_reshuffle_composition`)
answerable rather than a matter of taste.

RESHUFFLE
─────────
The first player to complete a City Plan may shuffle the discard back into the
deck. Reversing which cards are still to come is a large, one-off swing in the
number distribution, and :func:`after_reshuffle_composition` gives the exact pool
that choice would produce, so the decision can be computed instead of guessed.

Mind the ordering: the reshuffle resolves at the *next* turn boundary, after
this turn's aside cards have been discarded into it but before the number
cards beside them are.  The pool is ``deck + discard + aside``.

EXPERT MODE
───────────
``getAllDatas`` sends each BGA client only ``getForPlayer($pId)``, so in expert
mode a player never sees the opponents' cards and cannot attribute the shared
discard pile. Counting there would leak, so expert mode subtracts only the
player's own three cards. Standard and solo mode — where the stacks are shared and
everything discarded passed under the player's nose — are counted exactly.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from games.welcome_to.constants import (
    CARD_NUMBERS,
    CARD_TABLE,
    DECK_COUNTS,
    DECK_EFFECT_ORDER,
    EFFECT_INDEX,
    EPS,
    NUMBER_INDEX,
    Effect,
)
from games.welcome_to.game import GameState

NUM_NUMBERS: int = len(CARD_NUMBERS)          # 15
NUM_EFFECTS: int = len(DECK_EFFECT_ORDER)     # 6


def _deck_matrix() -> np.ndarray:
    matrix = np.zeros((NUM_NUMBERS, NUM_EFFECTS), dtype=np.float32)
    for number, counts in DECK_COUNTS.items():
        for effect, n in zip(DECK_EFFECT_ORDER, counts):
            matrix[NUMBER_INDEX[number], EFFECT_INDEX[effect]] = n
    return matrix


#: ``(15, 6)`` — how many copies of each (number, effect) card the printed deck
#: holds.  Sums to 81; the solo marker card is deliberately not in it.
DECK_MATRIX: np.ndarray = _deck_matrix()
DECK_SIZE: int = int(DECK_MATRIX.sum())  # 81


def _cell(card: int) -> Optional[tuple[int, int]]:
    number, effect = CARD_TABLE[card]
    if number is None or effect not in EFFECT_INDEX:
        return None  # the solo marker card is not a printed construction card
    return NUMBER_INDEX[number], EFFECT_INDEX[effect]


def _histogram(cards) -> np.ndarray:
    counts = np.zeros((NUM_NUMBERS, NUM_EFFECTS), dtype=np.float32)
    for card in cards:
        if card is None:
            continue
        cell = _cell(card)
        if cell is not None:
            counts[cell] += 1
    return counts


def known_cards(state: GameState, player: int) -> list[int]:
    """Cards this player has seen in full, and can therefore rule out of the deck."""
    table = [c for c in state.table_cards(player) if c is not None]
    if state.config.expert:
        return table  # the shared discard is not attributable in expert mode
    return list(state.discard) + table


def deck_composition(state: GameState, player: int) -> np.ndarray:
    """``(15, 6)`` — the exact composition of the undrawn deck.

    Sums to ``state.deck_remaining``, except in solo where the deck also holds the
    solo marker card, which is not a printed construction card and so is absent
    from :data:`DECK_MATRIX`.
    """
    counts = DECK_MATRIX - _histogram(known_cards(state, player))
    return np.maximum(counts, 0.0)


def discard_composition(state: GameState, player: int) -> np.ndarray:
    """``(15, 6)`` — what a reshuffle would put back into the deck."""
    if state.config.expert:
        return np.zeros((NUM_NUMBERS, NUM_EFFECTS), dtype=np.float32)
    return _histogram(state.discard)


def aside_composition(state: GameState, player: int) -> np.ndarray:
    """``(15, 6)`` — the three cards currently showing their EFFECT face.

    These join a reshuffle, and the discard does not account for them yet.  At
    the next turn boundary :meth:`GameState._begin_turn` runs ``_discard_step()``
    *before* ``_reshuffle_decks()``, so these cards are swept into the discard and
    then into the reformed deck.  The number cards beside them are discarded by
    the *second* ``_discard_step()``, which runs after ``_reform_deck()``, so they
    stay out of it.

    Zero outside standard mode, where there is no aside card.
    """
    if not state.config.standard:
        return np.zeros((NUM_NUMBERS, NUM_EFFECTS), dtype=np.float32)
    return _histogram(state.stack_old[0])


def after_reshuffle_composition(state: GameState, player: int) -> np.ndarray:
    """``(15, 6)`` — the pool the player would face if they took the reshuffle.

    The counterfactual behind the only genuinely strategic use of card counting in
    the base game: whoever completes the first City Plan chooses whether the
    discard goes back in.  Comparing this against :func:`deck_composition` turns
    that into arithmetic — do I want the low numbers back, given the gaps left on
    my sheet and the plans still open?

    **Three cards used to be missing from this.**  It returned ``deck + discard``,
    which is the pool as it stands *now* — but the reshuffle does not happen now,
    it happens at the next turn boundary, and by then this turn's three aside
    cards have already been discarded into it (see :func:`aside_composition`).
    Since this is the feature the card-counting edge rests on, undercounting the
    pool by three cards mattered.

    What the reshuffle then draws off the top — six cards, into the stacks — is a
    uniformly random subset of this pool, so it does not shift the composition the
    player should reason about.  This is the distribution the next numbers come
    from, which is what the encoder wants.
    """
    return (
        deck_composition(state, player)
        + discard_composition(state, player)
        + aside_composition(state, player)
    )


def next_number_distribution(state: GameState, player: int) -> np.ndarray:
    """``(15,)`` — the exact distribution of the next number a stack will show."""
    return _normalise(deck_composition(state, player).sum(axis=1))


def known_next_effects(state: GameState, player: int) -> np.ndarray:
    """``(3, 6)`` — the effect each stack will offer next turn, as one-hot rows.

    Certainty, not a posterior: the number face prints its own effect.  All-zero
    rows in expert and solo mode, where nothing carries over to the next turn.
    """
    out = np.zeros((3, NUM_EFFECTS), dtype=np.float32)
    for i, effect in enumerate(state.next_effects(player)):
        if effect is not None and effect in EFFECT_INDEX:
            out[i, EFFECT_INDEX[effect]] = 1.0
    return out


def effect_conditional_numbers(state: GameState, player: int) -> np.ndarray:
    """``(3, 15)`` — the number distribution *conditioned* on each stack's next effect.

    Not needed for the base game's reveal order, where the effect is known and the
    number is a plain deck draw, so this is the same marginal for all three
    stacks.  It is here because the correlation it encodes is what a *later* reveal
    tells you: once a number turns up, the effect that comes with it is
    constrained, and vice versa.  Cheap to compute, and it lets an ablation answer
    whether the network is using the joint histogram or only the marginals.
    """
    deck = deck_composition(state, player)
    out = np.zeros((3, NUM_NUMBERS), dtype=np.float32)
    marginal = _normalise(deck.sum(axis=1))
    for i, effect in enumerate(state.next_effects(player)):
        if effect is None or effect not in EFFECT_INDEX:
            out[i] = marginal
            continue
        column = deck[:, EFFECT_INDEX[effect]]
        out[i] = _normalise(column) if column.sum() > 0 else marginal
    return out


# ──────────────────────────────────────────────────────────────────────────
# Encoder v3 helpers -- ENCODER_V3_SPEC.md §7.1, §7.5, §9.3
# ──────────────────────────────────────────────────────────────────────────
def number_prefix_sums(
    state: GameState, player: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative counts over card numbers 1..15 for (deck, discard+aside, reshuffled).

    Each array is ``(16,)`` int64 with ``p[0] = 0`` and ``p[i]`` the number of
    cards printed ``1..i``, so any interval is two lookups (see
    :func:`count_in_open_interval`).  Built once per state and shared by every
    sheet: one deck feeds all seats in standard mode.

    ``reshuffled`` is the *immediate* reshuffle pool,
    :func:`after_reshuffle_composition` -- plane 16 is named for exactly that and
    is not the natural-reform pool (§7.4 R4).  ``p[-1]`` is the real denominator
    of each (§7.1 R5).
    """
    deck = deck_composition(state, player).sum(axis=1)
    pool = (
        discard_composition(state, player) + aside_composition(state, player)
    ).sum(axis=1)
    return _prefix(deck), _prefix(pool), _prefix(deck + pool)


def _prefix(counts: np.ndarray) -> np.ndarray:
    out = np.zeros(NUM_NUMBERS + 1, dtype=np.int64)
    np.cumsum(np.rint(counts).astype(np.int64), out=out[1:])
    return out


def count_in_open_interval(prefix: np.ndarray, low: int, high: int) -> int:
    """Cards whose printed number ``n`` satisfies ``low < n < high``.

    Exclusive bounds, matching ``gap_bounds`` and the §7.1 sentinels
    (``low = -1``, ``high = 18``); anything outside ``1..15`` is clipped, since
    no card prints it.  Card number ``n`` sits at ``prefix[n]`` because
    :data:`CARD_NUMBERS` is exactly ``1..15``.
    """
    lo = min(max(low, 0), NUM_NUMBERS)          # numbers <= lo are excluded
    hi = min(max(high - 1, 0), NUM_NUMBERS)     # numbers <= hi are included
    return int(prefix[hi] - prefix[lo]) if hi > lo else 0


def boundary_pool_composition(state: GameState, player: int) -> np.ndarray:
    """``(15, 6)`` -- the pool ``_reform_deck`` sees if the next boundary exhausts the deck.

    Not the discard pile as it stands: the boundary's ``_discard_step`` runs
    *before* the draw and sweeps the three **aside** cards into it, so they are
    already in the pool when ``_draw`` finds the deck empty (§7.5 R5 --
    undercounting this by three was a shipped bug once).  The number cards beside
    them are promoted, not discarded, and stay out.  So it is ``discard + aside``,
    the same pool a queued reshuffle reforms.

    Raises outside the 2+ player standard game (the only game this project trains
    or serves) and on a prepared boundary afterstate, where the discard step has
    already run and this would count the promoted number cards into a pool they
    never join.
    """
    _require_pre_boundary(state)
    return discard_composition(state, player) + aside_composition(state, player)


def _require_pre_boundary(state: GameState) -> None:
    if not state.config.standard or state.config.players < 2:
        raise ValueError(
            "boundary-draw features are defined for the 2+ player standard game "
            "only, not expert or one-seat play"
        )
    if state.boundary_prepared:
        raise ValueError(
            "boundary-draw features read a mid-turn state; this is a prepared "
            "boundary afterstate, whose discard step has already run"
        )


def ordered_draw_distribution(
    deck: np.ndarray, pool: np.ndarray, draws: int = 3
) -> np.ndarray:
    """``(K,)*draws`` float64 -- P(draws 1, 2, 3 land in classes a, b, c), exactly.

    The literal boundary draw of §7.5, over any partition of the cards into ``K``
    classes (15 numbers, or 6 effects).  ``deck`` and ``pool`` are per-class
    counts of the undrawn construction cards and of
    :func:`boundary_pool_composition`.  With ``D = deck.sum()``:

    * the first ``min(D, draws)`` draws come off the deck, without replacement;
    * ``_draw`` reforms only when it *finds* the deck empty, so the rest come off
      the pool, without replacement.  The reform shuffles the remaining deck in
      too, but it is empty by then, so the two sources are disjoint and
      independent.

    Falling factorials, never products of marginals: §9.3 bans ``1 - (1-p)^3``.
    Denominators follow §9.4 (``max(den, EPS)``);
    a pool too small to finish the draw gives an all-zero joint, a state the
    engine itself raises on.

    For a probability over a *set* of outcomes use :func:`ordered_draw_counts`
    and :func:`draw_probability`, not a sum over this array -- see there.
    """
    num, den = ordered_draw_counts(deck, pool, draws)
    return num / max(den, EPS)


def ordered_draw_counts(
    deck: np.ndarray, pool: np.ndarray, draws: int = 3
) -> tuple[np.ndarray, float]:
    """:func:`ordered_draw_distribution` as ``(numerators, denominator)``.

    Every numerator is a product of card counts -- an exact integer held in
    float64 (at most ``81 * 80 * 79``) -- so any masked sum of them is exact in
    ANY summation order, and a probability is one division at the end.

    ⚠ This is what makes §10.6 reachable.  Summing the float joint instead goes
    through numpy's pairwise summation, whose order Rust would have to copy
    bit for bit; summing integers makes the order irrelevant.
    """
    deck = np.asarray(deck, dtype=np.float64)
    pool = np.asarray(pool, dtype=np.float64)
    from_deck = min(int(round(deck.sum())), draws)
    num_deck, den_deck = _without_replacement(deck, from_deck)
    num_pool, den_pool = _without_replacement(pool, draws - from_deck)
    num = np.asarray(np.multiply.outer(num_deck, num_pool), dtype=np.float64)
    return num, den_deck * den_pool


def draw_probability(num: np.ndarray, den: float, masks) -> float:
    """P(draw ``i`` lands in ``masks[i]`` for every ``i``), from :func:`ordered_draw_counts`.

    ``masks`` are 0/1 vectors, one per draw.  Exact: the masked numerator sum is
    an integer, and the only rounding is the final division.
    """
    mask = np.multiply.outer(np.multiply.outer(masks[0], masks[1]), masks[2])
    hits = float((num * mask).sum())
    return min(1.0, max(0.0, hits / max(den, EPS)))


def _without_replacement(counts: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    """Ordered joint of ``n`` draws without replacement as ``(numerators, den)``.

    ``(K,)*n`` numerators (a scalar ``1.0`` at ``n = 0``) and the falling
    factorial ``total * (total - 1) * ...`` of ``n`` terms.
    """
    if n == 0:
        return np.float64(1.0), 1.0
    k = counts.shape[0]
    eye = np.eye(k)
    if n == 1:
        num = counts
    elif n == 2:
        num = counts[:, None] * (counts[None, :] - eye)
    elif n == 3:
        num = (
            counts[:, None, None]
            * (counts[None, :, None] - eye[:, :, None])
            * (counts[None, None, :] - eye[:, None, :] - eye[None, :, :])
        )
    else:
        raise ValueError(f"a boundary draws at most three cards, not {n}")
    total = float(counts.sum())
    den = 1.0
    for i in range(n):
        den *= total - i
    return np.clip(num, 0.0, None), den


# ──────────────────────────────────────────────────────────────────────────
# Masked draw probabilities -- integer inclusion-exclusion
#
# The production form of every "P(each draw lands in its own set)" question.
# ``ordered_draw_counts`` materialises the whole K^3 joint; this computes the
# masked numerator directly from per-mask sums (review 2026-09-25, throughput
# #1), in exact integers, so Python and Rust agree bit for bit and the order of
# summation is irrelevant.  ``ordered_draw_counts`` stays as the literal oracle
# the tests compare against.
# ──────────────────────────────────────────────────────────────────────────
def _as_counts(values) -> list[int]:
    return [int(round(float(v))) for v in values]


def falling(total: int, n: int) -> int:
    """``total * (total - 1) * ... `` over ``n`` terms (1 at ``n = 0``)."""
    out = 1
    for i in range(n):
        out *= total - i
    return out


def masked_draw_numerator(counts: list[int], masks) -> tuple[int, int]:
    """Ordered draws without replacement, draw ``i`` landing in ``masks[i]``.

    Returns ``(numerator, denominator)`` as exact integers, ``len(masks) <= 3``.
    With ``S_i = sum c m_i``, ``P_ij = sum c m_i m_j``, ``T = sum c m_0 m_1 m_2``::

        1 draw:  S_0
        2 draws: S_0 S_1 - P_01
        3 draws: S_0 S_1 S_2 - P_01 S_2 - P_02 S_1 - P_12 S_0 + 2 T

    inclusion-exclusion over "two draws took the same card".  Masks are 0/1.
    """
    n = len(masks)
    den = falling(sum(counts), n)
    if n == 0:
        return 1, den
    s = [sum(c * m for c, m in zip(counts, mask)) for mask in masks]
    if n == 1:
        return s[0], den

    def pair(a, b) -> int:
        return sum(c * x * y for c, x, y in zip(counts, masks[a], masks[b]))

    if n == 2:
        return s[0] * s[1] - pair(0, 1), den
    if n != 3:
        raise ValueError(f"a boundary draws at most three cards, not {n}")
    triple = sum(
        c * x * y * z for c, x, y, z in zip(counts, masks[0], masks[1], masks[2])
    )
    return (
        s[0] * s[1] * s[2]
        - pair(0, 1) * s[2]
        - pair(0, 2) * s[1]
        - pair(1, 2) * s[0]
        + 2 * triple
    ), den


def _probability(num: int, den: int) -> float:
    return min(1.0, max(0.0, float(num) / max(float(den), EPS)))


def next_draw_probability(deck, pool, masks) -> float:
    """P(each of the next three draws lands in its own mask), reform-aware.

    The literal boundary draw of §7.5: the first ``min(D, 3)`` come off the
    deck, the rest off ``pool`` (``boundary_pool_composition``).  Equal to
    ``draw_probability(*ordered_draw_counts(deck, pool), masks)`` exactly.
    """
    deck_c = _as_counts(deck)
    pool_c = _as_counts(pool)
    split = min(sum(deck_c), 3)
    a, da = masked_draw_numerator(deck_c, [_as_counts(m) for m in masks[:split]])
    b, db = masked_draw_numerator(pool_c, [_as_counts(m) for m in masks[split:]])
    return _probability(a * b, da * db)


#: Effect index of TEMP -- the only effect that changes which numbers a card can
#: be written as, so the only effect distinction a playability mask can see.
_TEMP = EFFECT_INDEX[Effect.TEMP]


def two_triple_probability(matrix, mask_non_temp, mask_temp) -> float:
    """P(all three stacks miss) when effects and numbers come from DIFFERENT cards.

    A stack offers the number of one card and the effect of another: after a
    queued reshuffle the engine draws three cards that become the asides
    (effects), then three more that become the tops (numbers), all six distinct
    and from the same pool.  ``matrix`` is that pool as ``(15, 6)`` counts; stack
    ``i``'s number misses ``mask_temp`` if its effect is TEMP, else
    ``mask_non_temp``.

    ⚠ **An approximation, not exact:** the effect triple and the number triple
    are each drawn without replacement, but *independently of each other* --
    the depletion of the number pool by the three effect cards is ignored.
    Exact conditioning costs ~216 x 15^3 terms per sheet, which the Python
    oracle cannot afford.  ``tests/test_deck_knowledge.py`` measures the error
    against brute-force six-card enumeration.

    Exact integers throughout: one shared denominator, so the sum over the eight
    effect-class sequences is order-free.
    """
    rows = _as_counts(np.asarray(matrix, dtype=np.float64).sum(axis=1))
    cols = _as_counts(np.asarray(matrix, dtype=np.float64).sum(axis=0))
    total = sum(rows)
    temp = cols[_TEMP]
    classes = (total - temp, temp)
    masks = (_as_counts(mask_non_temp), _as_counts(mask_temp))
    # With 0/1 masks every term of `masked_draw_numerator` for any sequence
    # drawn from these two masks is one of three sums: S_x = sum c m_x, and
    # sum c m_non m_temp for any product mixing the two (m * m = m).  So all
    # eight sequences come from three sums -- the same integers, far fewer
    # passes.  `masked_draw_numerator` stays the definition; the equivalence
    # is tested.
    single = [sum(c * m for c, m in zip(rows, mask)) for mask in masks]
    both = sum(c * x * y for c, x, y in zip(rows, masks[0], masks[1]))

    def overlap(*ts: int) -> int:
        return single[ts[0]] if len(set(ts)) == 1 else both

    num = 0
    for t0 in (0, 1):
        for t1 in (0, 1):
            for t2 in (0, 1):
                seq = (t0, t1, t2)
                effect_num = 1
                used = [0, 0]
                for t in seq:
                    effect_num *= classes[t] - used[t]
                    used[t] += 1
                if effect_num <= 0:
                    continue
                number_num = (
                    single[t0] * single[t1] * single[t2]
                    - overlap(t0, t1) * single[t2]
                    - overlap(t0, t2) * single[t1]
                    - overlap(t1, t2) * single[t0]
                    + 2 * overlap(t0, t1, t2)
                )
                num += effect_num * number_num
    return _probability(num, falling(total, 3) * falling(total, 3))


def effect_supply_rate(state: GameState, player: int) -> np.ndarray:
    """``(6,)`` float64 -- P(effect *e* is among the three effects offered on turn+2).

    Next turn's effects are printed and already in :func:`known_next_effects`;
    the cards the *coming* boundary draws are the ones flipped aside a turn
    later, so this is exact for turn+2 (§18 R5 #11), not a steady state.  Order
    is :data:`DECK_EFFECT_ORDER`, so ``temp_availability_rate`` and
    ``bis_availability_rate`` are two entries of this vector.

    ``1 - P(all three draws miss e)`` over the literal boundary draw
    (:func:`ordered_draw_distribution`), so ``D < 3`` reforms mid-draw rather
    than taking an approximation (§9.3 R5).

    ⚠ Branches on the viewer's **own** reshuffle vote, never on
    ``reshuffle_next_turn``, which leaks an earlier actor's hidden vote (§6.4 R5).
    If the viewer voted yes, the boundary reforms first and draws twice from the
    reformed pool; turn+2's effects are the *second* triple, which by
    exchangeability has the first triple's distribution over
    :func:`after_reshuffle_composition`.  A yes vote from someone else is hidden
    and left to search and the value head.
    """
    _require_pre_boundary(state)
    if state.reshuffle_vote_for(player):
        deck = after_reshuffle_composition(state, player).sum(axis=0)
        pool = np.zeros(NUM_EFFECTS)
    else:
        deck = deck_composition(state, player).sum(axis=0)
        pool = boundary_pool_composition(state, player).sum(axis=0)
    rate = np.empty(NUM_EFFECTS, dtype=np.float64)
    for e in range(NUM_EFFECTS):
        miss = [0 if k == e else 1 for k in range(NUM_EFFECTS)]
        rate[e] = 1.0 - next_draw_probability(deck, pool, (miss, miss, miss))
    return np.clip(rate, 0.0, 1.0)


def _normalise(vector: np.ndarray) -> np.ndarray:
    total = float(vector.sum())
    if total <= 0.0:
        return np.full_like(vector, 1.0 / max(vector.shape[0], 1))
    return (vector / total).astype(np.float32)


def summarise(state: GameState, player: int) -> str:
    """Readable dump of what the player can infer, for debugging and notebooks."""
    numbers = next_number_distribution(state, player)
    lines = [
        f"deck: {int(deck_composition(state, player).sum())} cards, "
        f"discard: {len(state.discard)}"
    ]
    likely = np.argsort(numbers)[::-1][:3]
    lines.append(
        "next number most likely: "
        + ", ".join(f"{CARD_NUMBERS[i]} p={numbers[i]:.3f}" for i in likely)
    )
    for i, (number, effect) in enumerate(state.visible_cards(player)):
        nxt = state.next_effects(player)[i]
        lines.append(
            f"  stack {i}: in play {number}/{effect.name if effect else '?'}"
            f"   next turn's effect: {nxt.name if nxt else 'n/a'}"
        )
    return "\n".join(lines)


def reveals_to_reform(state: GameState) -> int:
    """How many reveals until ``_reform_deck`` runs, as an UPPER BOUND.

    ``floor(D / 3) + 1``.  The ``+ 1`` is not cosmetic: ``_draw`` reforms only
    when it *finds* the deck empty, so with ``D = 3`` the next reveal consumes the
    last three cards and the reform fires on the reveal after that.

    It is a **bound**, not a prediction.  Exhaustion is deterministic at three
    cards a turn, but the first player to finish a City Plan may also *choose* a
    reshuffle, and one YES from anyone fires it.  That is a decision, not a state,
    and it can only make the refresh arrive sooner -- so this stays sound.
    ``reshuffle_race`` carries the opportunity; the policy learns the choice.
    """
    return state.deck_remaining // 3 + 1
