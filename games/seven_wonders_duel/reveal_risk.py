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
``reveal_*_sixth``    ``reveal_n`` x the fraction of the unseen pool that would
                      give that seat a sixth distinct science symbol.
``reveal_*_mil``      ``reveal_n`` x the fraction that would end it militarily
                      for that seat at once.

The fractions are one pass over the unseen pool per seat; the counts are local
geometry. So the whole block is O(unseen pool + slots), which is the same price
class as the features already in the encoder -- unlike a proof, which has to
enumerate reveals jointly.

The ``*_sixth`` and ``*_mil`` values are **expected counts of decisive cards
revealed**, not probabilities: with ``reveal_n = 2`` and a pool 40% decisive the
value is 0.8. That keeps the two factors the network needs -- how much you
uncover, and how dangerous the pool is -- in one number, while ``reveal_n`` is
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

from .data import CARDS_BY_NAME, TABLEAU_LAYOUTS, covering_slots

#: Per-slot channels, appended to TABLEAU_FEATURES. Order is the schema.
REVEAL_FEATURES = (
    "reveal_n",
    "reveal_my_sixth",
    "reveal_opp_sixth",
    "reveal_my_mil",
    "reveal_opp_mil",
)

#: Default OFF. The Rust encoder has no reveal block, so leaving these on would
#: make the two languages disagree on every tableau token -- the exact failure
#: `test_both_languages_agree_in_off_mode` exists to catch. Python-only
#: experiments (the offline A/B replays through Python) turn them on explicitly;
#: self-play cannot use them until Rust computes them too.
_ENABLED = os.environ.get(
    "SWD_REVEAL_FEATURES", "0"
).strip().lower() not in ("0", "false", "no", "off")


def reveal_features_enabled() -> bool:
    return _ENABLED


def set_reveal_features(enabled: bool) -> None:
    """Toggle the reveal channels. Zero-filled when off, exactly like control.

    Off-mode emits zeros rather than dropping the columns, so the schema -- and
    therefore the encoder signature -- is identical in both arms and a single
    checkpoint can be evaluated either way.
    """

    global _ENABLED
    _ENABLED = bool(enabled)


def newly_revealed_counts(obs) -> dict:
    """``slot_id -> how many hidden slots removing it would reveal``.

    A slot at row ``r`` is covered by slots at ``r + 1`` (``covering_slots``).
    Removing a coverer reveals a covered slot only when it was the **last**
    present coverer, which is the same condition ``TableauState.take_accessible``
    reports as newly accessible.
    """

    present = {card.slot_id: card for card in obs.tableau if card.present}
    if not present:
        return {}
    layout = TABLEAU_LAYOUTS[max(obs.age, 1)]
    by_id = {(slot.row, slot.x): slot for slot in layout}

    counts = {slot_id: 0 for slot_id in present}
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
            counts[coverers[0]] += 1
    return counts


def decisive_fractions(derived, seat: int) -> tuple[float, float]:
    """``(sixth-symbol fraction, immediate-military fraction)`` of the unseen pool.

    Mirrors the per-card tests the encoder already applies to face-up cards, so
    a revealed card is judged by the same rule whether it is visible or not.
    """

    names = []
    for back in derived.relevant_backs:
        names.extend(derived.pool.cards[back])
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

    counts = newly_revealed_counts(obs)
    if not any(counts.values()):
        return {slot_id: list(zeros) for slot_id in present}

    mine = decisive_fractions(derived, derived.actor)
    theirs = decisive_fractions(derived, 1 - derived.actor)

    out = {}
    for slot_id in present:
        n = float(counts.get(slot_id, 0))
        out[slot_id] = [
            n / 2.0,
            n * mine[0],
            n * theirs[0],
            n * mine[1],
            n * theirs[1],
        ]
    return out
