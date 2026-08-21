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
number distribution, and :func:`after_reshuffle_composition` gives the exact deck
that choice would produce, so the decision can be computed instead of guessed.

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


def after_reshuffle_composition(state: GameState, player: int) -> np.ndarray:
    """``(15, 6)`` — the deck the player would face if they took the reshuffle.

    The counterfactual behind the only genuinely strategic use of card counting in
    the base game: whoever completes the first City Plan chooses whether the
    discard goes back in.  Comparing this against :func:`deck_composition` turns
    that into arithmetic — do I want the low numbers back, given the gaps left on
    my sheet and the plans still open?
    """
    return deck_composition(state, player) + discard_composition(state, player)


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
