"""Exact positional control on the public tableau (Workstream 3, topology only).

The encoder already answers the FEASIBILITY question -- `sci_win_feasible` and
`science_missing_obtainable` are a set union over reachable cards and a count of
missing symbols. That contains no turn order, no cover graph and no extra-turn
Wonders, so a sixth science symbol buried under three coverers scores identically
to the same symbol one removal away with a tempo resource in hand. This solver
answers the other question: **who gets there first.**

Deliberately narrow, per the contract in `WORLD_CLASS_MODEL_EVOLUTION_PLAN.md`:

* **Topology only.** Removal order, turn alternation, and extra-turn Wonders as a
  tempo budget. It does NOT model coins, production, discounts, chains, military
  or the next-Age starter choice, so it cannot and must not emit forcing claims
  about victories. Every output here is named as a topology fact -- `can_take_first`,
  `decisions_until_accessible`, `must_open` -- never `forced_science_win_in_k`.
  A topology fact wearing a game-theoretic label is exactly the error this
  workstream exists to eliminate.
* **A supplement, not a replacement.** It supplies exact structure to a neural
  evaluation that keeps doing everything else.
* **Public information only.** It reads the present/absent mask and slot
  geometry. It never reads a face-down card's identity, so it is safe to call on
  a determinized state -- which matters, because `advisor_scrape` hands the
  searcher exactly that.

The model of a turn follows the engine:

    ordinary turn      remove one accessible slot
    extra-turn Wonder  remove one accessible slot (the burial) and move again

so an extra-turn Wonder is worth exactly one extra removal in the same turn,
which is the tempo resource that decides the reference case: the opponent buries
the newly exposed slot under `The Temple of Artemis` and takes `School` before
the actor can respond.

Both players are assumed to play optimally FOR THE TARGET -- the attacker to take
it as early as possible, the defender to take it first or deny it. That is a
two-player game on the removal poset, solved exactly by memoized minimax. It is a
bound on what is positionally possible, not a prediction: a player with better
things to do will not spend its whole Age racing for one card.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .data import TABLEAU_LAYOUTS, covering_slots
from .search import state_actor

ATTACKER, DEFENDER = 0, 1


@dataclass(frozen=True, slots=True)
class Layout:
    """Slot geometry for one Age, indexed for bitmask work."""

    age: int
    slots: tuple                      # slot ids, canonical order
    index: dict                       # slot id -> bit position
    covered_by: tuple                 # bit position -> mask of its coverers

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def for_age(age: int) -> "Layout":
        layout = TABLEAU_LAYOUTS[age]
        slots = tuple(sorted((s.row, s.x) for s in layout))
        index = {slot: i for i, slot in enumerate(slots)}
        by_id = {(s.row, s.x): s for s in layout}
        covered = []
        for slot in slots:
            mask = 0
            for coverer in covering_slots(layout, by_id[slot]):
                mask |= 1 << index[(coverer.row, coverer.x)]
            covered.append(mask)
        return Layout(age, slots, index, tuple(covered))

    def accessible(self, present: int) -> int:
        """Bitmask of present slots with no present coverer."""

        out = 0
        remaining = present
        while remaining:
            bit = remaining & -remaining
            i = bit.bit_length() - 1
            if not (self.covered_by[i] & present):
                out |= bit
            remaining ^= bit
        return out


def present_mask(tableau, layout: Layout) -> int:
    """The public present/absent mask. Identities are never read."""

    mask = 0
    for slot, card in tableau.cards.items():
        if card.present and slot in layout.index:
            mask |= 1 << layout.index[slot]
    return mask


_INF = 99


class ControlSolver:
    """Exact answers to 'who can take this slot first', by memoized minimax.

    `turns_to_take` is measured in the ATTACKER's decisions, so it is directly
    comparable across positions and reads naturally as "the opponent can have it
    in k of its turns".
    """

    def __init__(self, age: int):
        self.layout = Layout.for_age(age)
        self._memo: dict = {}
        self.nodes = 0

    def solve(self, present: int, target, to_move: int, extras: tuple[int, int]):
        """Earliest attacker-decision on which ATTACKER can take `target`.

        Returns `_INF` if the defender can always take or deny it first. Both
        sides play optimally with respect to this target alone.
        """

        bit = 1 << self.layout.index[target]
        if not (present & bit):
            return _INF  # already gone: no longer a control question
        return self._search(present, bit, to_move, extras, 0)

    def _search(self, present, target_bit, to_move, extras, spent):
        if spent > _INF:
            return _INF
        key = (present, to_move, extras)
        memo = self._memo.get(key)
        if memo is not None:
            return memo
        self.nodes += 1

        accessible = self.layout.accessible(present)
        if not accessible:
            self._memo[key] = _INF          # Age exhausted, nobody takes it
            return _INF
        if not (present & target_bit):
            self._memo[key] = _INF
            return _INF

        best = _INF if to_move == ATTACKER else -1
        remaining = accessible
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            after = present ^ bit

            if bit == target_bit:
                # Whoever is on move takes it now.
                value = (spent + 1) if to_move == ATTACKER else _INF
                best = min(best, value) if to_move == ATTACKER else max(best, value)
                if to_move == DEFENDER and best >= _INF:
                    break
                continue

            # Ordinary continuation: the turn passes.
            cost = 1 if to_move == ATTACKER else 0
            value = self._search(after, target_bit, 1 - to_move, extras, spent + cost)
            if to_move == ATTACKER:
                best = min(best, value)
            else:
                best = max(best, value)

            # Extra-turn Wonder: this removal was the burial, and the same
            # player moves again without the turn passing.
            if extras[to_move] > 0:
                spent_extras = list(extras)
                spent_extras[to_move] -= 1
                again = self._search(
                    after, target_bit, to_move, tuple(spent_extras), spent
                )
                if to_move == ATTACKER:
                    best = min(best, again)
                else:
                    best = max(best, again)

        self._memo[key] = best
        return best


def decisions_until_accessible(present: int, target, layout: Layout) -> int:
    """Minimum removals before `target` becomes accessible. Topology only.

    Ignores whose turn it is: this is the CHAIN DISTANCE the threat corpus
    stratifies on, not a claim about who does the removing.
    """

    bit = 1 << layout.index[target]
    if not (present & bit):
        return _INF
    seen, frontier = {present}, [(present, 0)]
    while frontier:
        mask, depth = frontier.pop(0)
        if layout.accessible(mask) & bit:
            return depth
        accessible = layout.accessible(mask) & ~bit
        remaining = accessible
        while remaining:
            one = remaining & -remaining
            remaining ^= one
            nxt = mask ^ one
            if nxt not in seen:
                seen.add(nxt)
                frontier.append((nxt, depth + 1))
    return _INF


def must_open(present: int, target, layout: Layout, to_move: int) -> bool:
    """Is every legal move for `to_move` one that uncovers `target`?

    The reference case's shape: the actor's own move is what exposes the
    threatened card. When every accessible option does that, the exposure is
    forced rather than chosen -- which is a different, and much stronger,
    statement than 'a move exists that exposes it'.
    """

    bit = 1 << layout.index[target]
    accessible = layout.accessible(present)
    if not accessible or (accessible & bit):
        return False
    remaining, options, exposing = accessible, 0, 0
    while remaining:
        one = remaining & -remaining
        remaining ^= one
        options += 1
        if layout.accessible(present ^ one) & bit:
            exposing += 1
    return options > 0 and options == exposing


# --------------------------------------------------------------------------
# Counterfactual control -- the feature set
#
# The useful quantity is not one control answer but the DIFFERENCE between
# control under the tempo you have and control under the tempo you might have.
# "One unspent extra-turn Wonder takes Age III control from 10% to 100%" is a
# fact about the position that no count of obtainable symbols can express.
#
# Affordability is deliberately NOT modelled. The solver states the topological
# fact -- with k extra turns you control these positions -- and the network,
# which already knows its own coins, production, discounts and chains, learns
# whether k is attainable. That keeps the solver exact within a contract it can
# actually honour.
# --------------------------------------------------------------------------


def wonder_tempo_budget(game, player, *, assume_theology: bool = False) -> int:
    """Extra turns `player` could still take, as a count of Wonder builds.

    `Theology` makes EVERY Wonder grant an extra turn, so the budget is not the
    count of unbuilt PLAY_AGAIN Wonders but the count of unbuilt Wonders. That
    is the whole point of the token, and the reference case is precisely a game
    where conceding it turned three ordinary Wonders into three extra turns.

    The seventh-Wonder rule caps this hard: once seven Wonders are built across
    both cities there is no eighth, so the budget is zero whatever remains
    unbuilt. That rule is what forced `The Sphinx` in table 907773062, and a
    tempo feature blind to it would misread that whole game.
    """

    from .data import WONDERS_BY_NAME, EffectKind

    city = game.cities[player]
    retired = set(getattr(game, "retired_wonders", ()) or ())
    unbuilt = [
        w for w in city.wonders
        if w not in city.built_wonders and w not in retired
    ]
    built_total = sum(len(c.built_wonders) for c in game.cities)
    if built_total >= 7 or not unbuilt:
        return 0

    theology = assume_theology or "Theology" in city.progress_tokens
    if theology:
        usable = len(unbuilt)
    else:
        usable = sum(
            1 for name in unbuilt
            if any(
                e.kind is EffectKind.PLAY_AGAIN
                for e in WONDERS_BY_NAME[name].effects
            )
        )
    # No more than the number of Wonder builds the seventh-Wonder rule allows.
    return min(usable, 7 - built_total)


def control_map(present: int, age: int, to_move_is_attacker: bool,
                extras: tuple[int, int]) -> tuple:
    """Which present slots the attacker takes first. Public information only."""

    layout = Layout.for_age(age)
    first = ATTACKER if to_move_is_attacker else DEFENDER
    owned = []
    remaining = present
    while remaining:
        bit = remaining & -remaining
        remaining ^= bit
        slot = layout.slots[bit.bit_length() - 1]
        if ControlSolver(age).solve(present, slot, first, extras) < _INF:
            owned.append(slot)
    return tuple(owned)


def control_features(game, seat: int) -> dict:
    """Counterfactual positional control for `seat`, as encoder-ready facts.

    Every value is a COUNT or FRACTION of present slots this seat takes first
    under optimal play for that slot. None of it is a victory claim.
    """

    layout = Layout.for_age(game.tableau.age)
    present = present_mask(game.tableau, layout)
    total = bin(present).count("1")
    if not total:
        return {}
    opponent = 1 - seat
    on_move = state_actor(game) == seat

    mine = wonder_tempo_budget(game, seat)
    theirs = wonder_tempo_budget(game, opponent)
    theirs_theology = wonder_tempo_budget(game, opponent, assume_theology=True)
    mine_theology = wonder_tempo_budget(game, seat, assume_theology=True)

    def frac(my_tempo, their_tempo):
        return len(control_map(
            present, game.tableau.age, on_move, (my_tempo, their_tempo)
        )) / total

    now = frac(mine, theirs)
    return {
        # What I control as things stand.
        "control_now": now,
        # What one more unspent extra turn is worth -- the lever the reference
        # case turns on.
        "control_with_one_more_tempo": frac(mine + 1, theirs),
        # What losing my tempo would cost.
        "control_if_i_spend_tempo": frac(max(0, mine - 1), theirs),
        # What the opponent taking Theology would cost me. On the reference
        # case this is the difference between counting one PLAY_AGAIN Wonder
        # and counting all three unbuilt ones.
        "control_if_opponent_takes_theology": frac(mine, theirs_theology),
        "control_if_i_take_theology": frac(mine_theology, theirs),
        # The budgets themselves, since control is non-monotone in a DIFFERENCE
        # of tempo and the network needs both.
        "my_tempo": mine,
        "their_tempo": theirs,
        "their_tempo_with_theology": theirs_theology,
        "on_move": float(on_move),
        "slots_present": total,
    }
