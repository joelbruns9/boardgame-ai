"""Per-slot reveal risk: what a removal is likely to hand the opponent.

The gap this fills
------------------
The encoder already tells the network, for every *face-up* card, whether it
gives a seat the sixth science symbol or immediate military supremacy
(``gives_sixth`` and ``shields >= dist_win`` in ``_tableau_card_per_player``).
It says nothing about the cards a removal would *uncover*, and those decide
positions:

* table 904750590 row 24 -- five actions take the same card and uncover the same
  two slots. Four of the ten reveal-worlds hand the opponent a green card and
  the position is worth ~0.02%; the other six are worth 44-54%.
* W3's control channels cannot express this. ``control_key_word`` is keyed on
  ``(age, present-mask, who-moves, tempo)`` with **no card identities**, so it
  returns an identical answer in all ten worlds. Its chance-invariance is
  exactly why it is a cheap table lookup, and exactly why it is blind here.

What is computed, and why it is cheap
-------------------------------------
No search and no joint enumeration. For each present slot:

``reveal_n``          how many currently-hidden slots become accessible (and so
                      are revealed) when this slot is removed.
``reveal_*_sixth``    summed over the slots it uncovers: the fraction of THAT
                      SLOT'S back-specific pool that would give that seat a
                      sixth distinct science symbol.
``reveal_*_mil``      the same, for cards that would end it militarily at once.

The fractions are one pass over each back's unseen pool per seat; the counts are
local geometry. So the whole block is O(unseen pool + slots), which is the same
price class as the features already in the encoder -- unlike a proof, which has
to enumerate reveals jointly.

A card's back is public even while its face is not, and it says which pool the
card comes from: a Guild-backed slot cannot turn over an Age III card, and an
Age III slot cannot turn over a Guild. Pooling every relevant back and scaling
by the count -- which this did until 2026-09-07 -- therefore priced a Guild
reveal with cards it could never produce. Per-back is also the only reading that
is right in Ages I and II, where the pooled version mixed in whole future ages.

The ``*_sixth`` and ``*_mil`` values are **expected counts of decisive cards
revealed**, not probabilities: uncovering two slots whose pools are 40% decisive
reads 0.8. That keeps the two factors the network needs -- how much you uncover,
and how dangerous what you uncover is -- in one number, while ``reveal_n`` is
still there separately if it wants them apart.

What this is NOT
----------------
It is a *marginal* risk per revealed slot, not a joint probability over the
actual multi-card deal, and it ignores whether the opponent can afford the card
or reach it first. Affordability and turn order are what the rest of the encoder
and W3 already carry. A single number here that pretended to be "probability I
lose to this reveal" would be the same error W3 is constrained to avoid.
"""

from __future__ import annotations

import os

from .data import CARDS_BY_NAME, TABLEAU_LAYOUTS, BackType, covering_slots

#: Back order for the per-back sums below. Fixed, and shared with Rust, so the
#: two languages accumulate the same floats in the same order.
_BACKS = tuple(BackType)

#: Per-slot channels, appended to TABLEAU_FEATURES. Order is the schema.
REVEAL_FEATURES = (
    "reveal_n",
    "reveal_my_sixth",
    "reveal_opp_sixth",
    "reveal_my_mil",
    "reveal_opp_mil",
)

#: Default OFF, in both languages: `reveal.rs` reads the same variable with the
#: same parsing rather than assuming a default, so the two agree whether or not
#: anyone calls the setter. Rust computes the block now, so ON is no longer
#: Python-only -- but a self-play run still has to turn it on deliberately.
_ENABLED = os.environ.get(
    "SWD_REVEAL_FEATURES", "0"
).strip().lower() not in ("0", "false", "no", "off")


def reveal_features_enabled() -> bool:
    return _ENABLED


def set_reveal_features(enabled: bool) -> None:
    """Toggle the reveal channels, in Python and in Rust together.

    Off-mode emits zeros rather than dropping the columns, so the schema -- and
    therefore the encoder signature -- is identical in both arms and a single
    checkpoint can be evaluated either way.

    Setting one language only would be worse than either arm: the replay path
    and the self-play path would disagree about what the model is being shown.
    """

    global _ENABLED
    _ENABLED = bool(enabled)
    try:
        import seven_wonders_rust
    except ImportError:
        return
    setter = getattr(seven_wonders_rust, "set_reveal_features_enabled", None)
    if setter is not None:
        setter(bool(enabled))


def newly_revealed_backs(obs) -> dict:
    """``slot_id -> {back: how many hidden slots of that back it would reveal}``.

    A slot at row ``r`` is covered by slots at ``r + 1`` (``covering_slots``).
    Removing a coverer reveals a covered slot only when it was the **last**
    present coverer, which is the same condition ``TableauState.take_accessible``
    reports as newly accessible.

    The BACK of each uncovered slot is carried, not just the count, because a
    card's back is public even while its face is not, and it says which pool the
    card comes from. A Guild-backed slot cannot turn over an Age III card and an
    Age III slot cannot turn over a Guild -- so pooling them, as this did until
    2026-09-07, priced a Guild reveal with Age III cards it could never produce.
    """

    present = {card.slot_id: card for card in obs.tableau if card.present}
    if not present:
        return {}
    layout = TABLEAU_LAYOUTS[max(obs.age, 1)]
    by_id = {(slot.row, slot.x): slot for slot in layout}

    revealed = {slot_id: {} for slot_id in present}
    for slot_id, card in present.items():
        if card.revealed:
            continue
        coverers = [
            (coverer.row, coverer.x)
            for coverer in covering_slots(layout, by_id[slot_id])
            if (coverer.row, coverer.x) in present
        ]
        # Exactly one coverer left: removing it uncovers this hidden card.
        if len(coverers) == 1:
            # `card.back`, never `card.card_name`: this card is face DOWN, so
            # the observation carries no name for it. The back is public.
            back = card.back
            counts = revealed[coverers[0]]
            counts[back] = counts.get(back, 0) + 1
    return revealed


def newly_revealed_counts(obs) -> dict:
    """``slot_id -> how many hidden slots removing it would reveal``."""

    return {
        slot_id: sum(counts.values())
        for slot_id, counts in newly_revealed_backs(obs).items()
    }


def decisive_fractions(derived, seat: int, back=None) -> tuple[float, float]:
    """``(sixth-symbol fraction, immediate-military fraction)`` of a card pool.

    Mirrors the per-card tests the encoder already applies to face-up cards, so
    a revealed card is judged by the same rule whether it is visible or not.

    ``back`` restricts the pool to the cards that could actually be under a slot
    with that back. Omitted, every relevant back is pooled -- which is what a
    caller wants for "the pool at large", and is exactly wrong for one slot.
    """

    if back is None:
        names = []
        for relevant in derived.relevant_backs:
            names.extend(derived.pool.cards[relevant])
    else:
        names = list(derived.pool.cards.get(back, ()))
    if not names:
        return 0.0, 0.0

    have = derived.symbols[seat]
    one_away = len(have) + 1 >= 6
    dist_win = 9 - derived.rel_position(seat)

    sixth = 0
    military = 0
    for name in names:
        card = CARDS_BY_NAME[name]
        if one_away and card.science is not None and card.science not in have:
            sixth += 1
        if derived.effective_shields(seat, name) >= dist_win:
            military += 1
    total = float(len(names))
    return sixth / total, military / total


def reveal_values(derived) -> dict:
    """``slot_id -> [reveal_n, my_sixth, opp_sixth, my_mil, opp_mil]``.

    All zero when the channels are off, or when there is nothing hidden left to
    uncover -- which is the honest reading, not a missing value: a fully
    revealed tableau carries no reveal risk.
    """

    zeros = [0.0] * len(REVEAL_FEATURES)
    obs = derived.obs
    present = [card.slot_id for card in obs.tableau if card.present]
    if not _ENABLED or not present:
        return {slot_id: list(zeros) for slot_id in present}

    revealed = newly_revealed_backs(obs)
    if not any(counts for counts in revealed.values()):
        return {slot_id: list(zeros) for slot_id in present}

    # One pass per (seat, back) rather than per slot, and only for the backs
    # this position can actually turn over -- in Age III that is two of four.
    occurs = {back for counts in revealed.values() for back in counts}
    fractions = {}
    for seat in (derived.actor, 1 - derived.actor):
        for back in _BACKS:
            fractions[(seat, back)] = (
                decisive_fractions(derived, seat, back)
                if back in occurs else (0.0, 0.0)
            )

    opponent = 1 - derived.actor
    out = {}
    for slot_id in present:
        counts = revealed.get(slot_id, {})
        n = float(sum(counts.values()))
        # Summed over backs in a FIXED order, so the two languages accumulate
        # the same floats in the same order and stay bit-identical.
        totals = [0.0, 0.0, 0.0, 0.0]
        for back in _BACKS:
            count = counts.get(back)
            if not count:
                continue
            mine = fractions[(derived.actor, back)]
            theirs = fractions[(opponent, back)]
            totals[0] += count * mine[0]
            totals[1] += count * theirs[0]
            totals[2] += count * mine[1]
            totals[3] += count * theirs[1]
        out[slot_id] = [n / 2.0, totals[0], totals[1], totals[2], totals[3]]
    return out
