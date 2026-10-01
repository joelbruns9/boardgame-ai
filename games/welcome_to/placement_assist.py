"""
Placement assistance: replace a write by the same-card write that best
preserves sheet hygiene.

Used in two places:

* ``hygiene_rescue`` -- the diagnostic that measured it (2026-10-01: learner
  plans +0.14 from the learner's own sheet, +0.28 with every seat assisted).
* S2 generation (``self_play`` ``--assist-fraction``) -- a **phased-out
  scaffold**, chosen 2026-10-01.  The learner plays clean sheets in a share of
  games so the value head finally sees what clean sheets lead to; the
  on-policy data had no such contrast (the rule changed 95% of the learner's
  placements).  The policy target stays the search's visit distribution, so the
  network never imitates the rule.  The share decays to zero, and play after
  removal is what judges it.

The rule: among legal writes of the chosen stack slot -- same number and
effect, any temp delta and box -- take the one leaving the acting seat the most
placement capacity, then the most span.  Ties keep the seat's own choice.
Refusals, roundabouts and every effect decision are untouched.
"""
from __future__ import annotations

from typing import Optional

from games.welcome_to import macro_codec as mc
from games.welcome_to import snapshot
from games.welcome_to.game import GameState


def is_write(macro: int) -> bool:
    return mc.M_WRITE <= macro < mc.M_REFUSE


def hygiene_key(state: GameState, macro: int) -> tuple[int, int]:
    """(placement capacity, span) the acting seat is left with after ``macro``."""
    sheet = mc.step_macro(state, macro).sheets[state.actor]
    return sum(sheet.placement_capacity()), sheet.total_span()


def assisted_choice(state: GameState, choice: int) -> int:
    """The same-slot write that best preserves hygiene; ``choice`` on ties or
    when ``choice`` is not a write."""
    if not is_write(choice):
        return choice
    slot = mc.decode_macro_write(choice)[0]
    candidates = [
        macro
        for macro in mc.legal_macros(state)
        if is_write(macro) and mc.decode_macro_write(macro)[0] == slot
    ]
    keys = {macro: hygiene_key(state, macro) for macro in candidates}
    best = max(keys.values())
    if keys[choice] == best:
        return choice
    return min(macro for macro, key in keys.items() if key == best)


def assist_rust(rust_state, choice: int) -> int:
    """:func:`assisted_choice` for a ``welcome_to_rust.RustGameState``."""
    if not is_write(choice):
        return choice
    return assisted_choice(snapshot.from_snapshot(rust_state.snapshot()), choice)


class Assistant:
    """A ``move_override``: assists ``seats`` (``None`` = every seat) through
    turn ``through``, and counts what it saw and changed."""

    def __init__(self, seats: Optional[frozenset[int]], through: int) -> None:
        self.seats = seats
        self.through = through
        self.decisions = 0
        self.changed = 0

    def __call__(self, rust_state, choice: int) -> int:
        actor = int(rust_state.actor)
        if (
            rust_state.turn > self.through
            or (self.seats is not None and actor not in self.seats)
            or not is_write(choice)
        ):
            return choice
        picked = assist_rust(rust_state, choice)
        self.decisions += 1
        self.changed += picked != choice
        return picked
