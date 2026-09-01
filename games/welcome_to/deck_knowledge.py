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
    NUMBER_INDEX,
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


# ──────────────────────────────────────────────────────────────────────────
# Encoder v3: prefix sums and exact supply rates
#
# ENCODER_V3_SPEC.md §7.1, §7.4 and §9.3.  Two rules run through all of it:
#
#   * divide by the ACTUAL sum of the matrix being summed, never by
#     `deck_remaining`.  In solo the undrawn deck also holds `SOLO_CARD_ID`,
#     which is not a printed construction card and so is absent from
#     `DECK_MATRIX`; `sum(c) == deck_remaining - 1` there.
#   * there is no `D < 3` approximation.  The next reveal still produces three
#     cards -- `_draw` reforms the discard mid-draw and carries on -- so the
#     boundary is enumerated exactly rather than waved away.
# ──────────────────────────────────────────────────────────────────────────
def number_prefix_sums(
    state: GameState, player: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative card counts over printed numbers 1..15.

    Returns ``(deck, reform, reshuffled)``, each a ``(16,)`` array where
    ``p[k]`` is the number of cards with printed number ``<= k``.  The count of
    cards in an exclusive range ``(low, high)`` is then a single subtraction::

        cards_in(low, high) = p[min(high - 1, 15)] - p[max(low, 0)]

    which is what makes the fit-probability planes affordable: the sums are
    built **once per state** and reused across all four sheets and every gap.

    * ``deck``       -- the undrawn deck as it stands;
    * ``reform``     -- what a mid-draw reform would put back (discard plus the
      three aside cards, which ``_discard_step`` sweeps in before ``_draw``
      finds the deck empty);
    * ``reshuffled`` -- the pool after a queued reshuffle, i.e. both together.
    """
    deck = deck_composition(state, player).sum(axis=1)
    reform = (
        discard_composition(state, player) + aside_composition(state, player)
    ).sum(axis=1)
    reshuffled = deck + reform

    def prefix(counts: np.ndarray) -> np.ndarray:
        out = np.zeros(NUM_NUMBERS + 1, dtype=np.float64)
        out[1:] = np.cumsum(counts)
        return out

    return prefix(deck), prefix(reform), prefix(reshuffled)


def _falling(n: float, k: int) -> float:
    """``n * (n-1) * ... * (n-k+1)``, and 1.0 for ``k == 0``."""
    out = 1.0
    for i in range(k):
        out *= max(0.0, n - i)
    return out


def _p_none_in_next_three(deck_hits: float, deck_size: float,
                          reform_hits: float, reform_size: float) -> float:
    """P(none of the three cards revealed next turn is a "hit").

    The reveal draws three cards **without replacement**, so this is a product of
    falling factorials, never ``(1 - p)**3``.  And when the deck holds fewer than
    three, ``_draw`` reforms the discard *mid-draw* and keeps going, so the three
    cards come from two pools in sequence -- the draw does not simply stop.

    Both phases are exact hypergeometrics; the deck phase takes as many cards as
    it has, the reform phase supplies the remainder.
    """
    from_deck = min(3, int(deck_size))
    from_reform = 3 - from_deck

    p = 1.0
    if from_deck:
        denominator = _falling(deck_size, from_deck)
        if denominator <= 0.0:
            return 0.0
        p *= _falling(deck_size - deck_hits, from_deck) / denominator
    if from_reform:
        denominator = _falling(reform_size, from_reform)
        if denominator <= 0.0:
            # Nothing left anywhere to draw: the question is vacuous, and
            # reporting "a hit is impossible" is the honest reading.
            return 1.0
        p *= _falling(reform_size - reform_hits, from_reform) / denominator
    return p


def effect_supply_rate(state: GameState, player: int) -> np.ndarray:
    """``(6,)`` -- P(effect ``e`` is among the three cards revealed next turn).

    ⚠ **Not** ``1 - (1 - p)**3``.  Three cards are drawn without replacement, so
    the independent-trials form is an approximation; at ``D = 40, k = 9`` it gives
    0.535 against 0.545 exact, and the gap widens as the deck drains.

    This is **exact for turn+2**, not a steady-state estimate: the three cards
    drawn at the next boundary are precisely the ones whose effects are offered
    two turns from now, because they become the asides.  Next turn's effects are
    already certainties in :func:`known_next_effects`.
    """
    deck = deck_composition(state, player)
    reform = discard_composition(state, player) + aside_composition(state, player)
    deck_size = float(deck.sum())
    reform_size = float(reform.sum())

    rates = np.zeros(NUM_EFFECTS, dtype=np.float32)
    for e in range(NUM_EFFECTS):
        rates[e] = 1.0 - _p_none_in_next_three(
            float(deck[:, e].sum()), deck_size, float(reform[:, e].sum()), reform_size
        )
    return rates


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
