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

# --------------------------------------------------------------------------
# Tempo state
#
# The first version of this solver gave each player an independent count of
# "extra turns". That is wrong in the one way that matters: Wonders are drawn
# from a SHARED pool of seven builds, and constructing the seventh retires every
# Wonder still unbuilt on both sides. An ordinary Wonder build is therefore not
# just a slower card take -- it can be the move that erases the opponent's
# extra-turn Wonder. That mechanism (`The Pyramids` retiring `The Sphinx`)
# decided table 907773062, and a tempo model blind to it reads that game as the
# opponent holding a guaranteed tempo coupon it did not actually hold.
#
# A tempo state is the 5-tuple
#
#     (builds_left, ordinary[0], extra[0], ordinary[1], extra[1])
#
# `builds_left` is 7 minus the Wonders already constructed across both cities:
# the shared capacity. `ordinary` counts a player's unbuilt Wonders that grant
# no extra turn, `extra` those that do (under `Theology`, every Wonder does).
# Indices follow ATTACKER / DEFENDER.
# --------------------------------------------------------------------------

_UNBOUNDED = 90          # a builds_left the search can never draw down

Tempo = tuple  # (builds_left, ord_att, ext_att, ord_def, ext_def)

NO_TEMPO: Tempo = (_UNBOUNDED, 0, 0, 0, 0)


def coupons(attacker: int, defender: int) -> Tempo:
    """An ABSTRACT tempo counterfactual: k free extra turns, no Wonder pool.

    Retirement cannot fire and no ordinary Wonder build exists, so this asks
    "what could this player do if it simply had k extra turns" -- a strictly
    optimistic bound on what its actual unbuilt Wonders can deliver. Use
    `tempo_state()` for a real position; use this only when the question really
    is the counterfactual.
    """

    return (_UNBOUNDED, 0, attacker, 0, defender)


def _spend(tempo: Tempo, player: int, extra: bool) -> Tempo:
    """Construct one Wonder for `player`; return the tempo state afterwards.

    When this build is the seventh, every Wonder still unbuilt on either side is
    retired and all four counters collapse to zero.
    """

    builds = tempo[0] - 1
    counts = list(tempo[1:])
    counts[2 * player + (1 if extra else 0)] -= 1
    if builds <= 0:
        return (0, 0, 0, 0, 0)
    return (builds, *counts)


def _can_build(tempo: Tempo, player: int, extra: bool) -> bool:
    return tempo[0] > 0 and tempo[1 + 2 * player + (1 if extra else 0)] > 0


class ControlSolver:
    """Exact answers to 'who can take this slot first', by memoized minimax.

    The returned distance is counted in the ATTACKER's turns, RELATIVE to the
    position asked about, so it reads as "the opponent can have it in k of its
    turns". Relative is not a stylistic choice: an absolute count makes the
    memoized value depend on the path taken to a node, and a memo key cannot see
    the path. That, plus `target_bit` in the key, is what makes one solver safe
    to reuse across targets and across root positions.

    A turn is one of:

        take a card        remove one accessible slot; the turn passes
        ordinary Wonder    remove one accessible slot (the burial), spend a
                           Wonder build; the turn passes
        extra-turn Wonder  remove one accessible slot, spend a Wonder build,
                           and move AGAIN in the same turn

    Both players are assumed to play optimally FOR THE TARGET -- the attacker to
    take it as early as possible, the defender to take it first or deny it. This
    is a bound on what is positionally possible, not a prediction.
    """

    def __init__(self, age: int):
        self.layout = Layout.for_age(age)
        self._memo: dict = {}
        self.nodes = 0

    def solve(self, present: int, target, to_move: int, tempo: Tempo):
        """Attacker turns until ATTACKER can take `target`, or `_INF`.

        `tempo` is a 5-tuple from `tempo_state()` or `coupons()`. A bare
        `(a, d)` pair is accepted and read as `coupons(a, d)`.
        """

        if len(tempo) == 2:
            tempo = coupons(*tempo)
        bit = 1 << self.layout.index[target]
        if not (present & bit):
            return _INF  # already gone: no longer a control question
        return self._search(present, bit, to_move, tempo)

    def _search(self, present, target_bit, to_move, tempo):
        key = (present, target_bit, to_move, tempo)
        memo = self._memo.get(key)
        if memo is not None:
            return memo
        self.nodes += 1

        accessible = self.layout.accessible(present)
        if not accessible:
            self._memo[key] = _INF          # Age exhausted, nobody takes it
            return _INF

        best = _INF if to_move == ATTACKER else -1
        remaining = accessible
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            after = present ^ bit

            if bit == target_bit:
                if to_move == ATTACKER:
                    best = min(best, 1)     # take it now: one of my turns
                else:
                    best = _INF             # gone for good
                    break
                continue

            # A plain card take and an ordinary Wonder build remove the same
            # slot and pass the turn; they differ only in what they do to the
            # shared Wonder pool -- which is the whole reason the ordinary build
            # is modelled at all.
            for nxt in _pass_turn_options(tempo, to_move):
                got = self._search(after, target_bit, 1 - to_move, nxt)
                if got < _INF and to_move == ATTACKER:
                    got += 1                # the attacker just used a turn
                best = min(best, got) if to_move == ATTACKER else max(best, got)

            # An extra-turn Wonder: this removal was the burial, and the same
            # player moves again without the turn passing.
            if _can_build(tempo, to_move, True):
                again = self._search(
                    after, target_bit, to_move, _spend(tempo, to_move, True)
                )
                best = min(best, again) if to_move == ATTACKER else max(best, again)

            if to_move == DEFENDER and best >= _INF:
                break

        best = min(best, _INF)
        self._memo[key] = best
        return best


def _pass_turn_options(tempo: Tempo, player: int):
    """Tempo states reachable by a removal that passes the turn."""

    yield tempo                                    # take the card
    if _can_build(tempo, player, False):
        yield _spend(tempo, player, False)         # ordinary Wonder build


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
# reachability under the tempo you have and under the tempo you might have.
#
# Affordability is deliberately NOT modelled: the solver states the topological
# fact -- with this Wonder pool these positions are reachable first -- and the
# network, which already knows its own coins, production, discounts and chains,
# learns whether the build is payable. Because of that the outputs are named as
# reachability under topology, never as control "as things stand".
# --------------------------------------------------------------------------


def _unbuilt(game, player):
    from .data import WONDERS_BY_NAME, EffectKind

    city = game.cities[player]
    retired = set(getattr(game, "retired_wonders", ()) or ())
    names = [
        w for w in city.wonders
        if w not in city.built_wonders and w not in retired
    ]
    grants = {
        name: any(
            e.kind is EffectKind.PLAY_AGAIN
            for e in WONDERS_BY_NAME[name].effects
        )
        for name in names
    }
    return names, grants


def tempo_state(game, attacker: int, *,
                theology: tuple[bool, bool] | None = None) -> Tempo:
    """The real shared-pool tempo state, from ATTACKER's point of view.

    `theology[p]` overrides whether player `p` is treated as holding the token,
    indexed ATTACKER / DEFENDER; `None` reads each city's actual progress
    tokens. Under `Theology` EVERY Wonder grants an extra turn, which is what
    turned three ordinary Wonders into three extra turns in the reference game.
    """

    built_total = sum(len(c.built_wonders) for c in game.cities)
    builds_left = max(0, 7 - built_total)
    counts = []
    for slot, player in enumerate((attacker, 1 - attacker)):
        names, grants = _unbuilt(game, player)
        has = (theology[slot] if theology is not None
               else "Theology" in game.cities[player].progress_tokens)
        if has:
            counts += [0, len(names)]
        else:
            extra = sum(1 for n in names if grants[n])
            counts += [len(names) - extra, extra]
    builds_left = min(builds_left, sum(counts))
    if builds_left <= 0:
        return (0, 0, 0, 0, 0)
    return (builds_left, *counts)


def control_map(present: int, age: int, to_move_is_attacker: bool,
                tempo: Tempo, solver: "ControlSolver | None" = None) -> tuple:
    """Which present slots the attacker reaches first. Public information only.

    A solver may be shared across targets and across positions: the memo key
    identifies the target and the cached value is relative to its node, so reuse
    is a pure saving.
    """

    layout = Layout.for_age(age)
    first = ATTACKER if to_move_is_attacker else DEFENDER
    solver = solver if solver is not None else ControlSolver(age)
    owned = []
    remaining = present
    while remaining:
        bit = remaining & -remaining
        remaining ^= bit
        slot = layout.slots[bit.bit_length() - 1]
        if solver.solve(present, slot, first, tempo) < _INF:
            owned.append(slot)
    return tuple(owned)


def _plus_extra(tempo: Tempo, player: int, n: int = 1) -> Tempo:
    """Hand `player` n more extra-turn Wonders AND the pool capacity to build
    them, so the counterfactual stays internally legal."""

    counts = list(tempo[1:])
    counts[2 * player + 1] += n
    return (tempo[0] + n, *counts)


def _minus_extra(tempo: Tempo, player: int) -> Tempo:
    counts = list(tempo[1:])
    i = 2 * player + 1
    if counts[i] <= 0:
        return tempo
    counts[i] -= 1
    return (tempo[0], *counts)


def _as_theology(tempo: Tempo, player: int) -> Tempo:
    """Fold a player's ordinary unbuilt Wonders into its extra-turn ones."""

    counts = list(tempo[1:])
    o, e = 2 * player, 2 * player + 1
    counts[e] += counts[o]
    counts[o] = 0
    return (tempo[0], *counts)


def control_features(game, seat: int) -> dict:
    """Counterfactual positional reachability for `seat`, as exact facts.

    Every value is a FRACTION of present slots this seat reaches first under
    play optimal for that slot, given the shared Wonder pool. None of it is a
    victory claim, and none of it asserts the builds are affordable.
    """

    layout = Layout.for_age(game.tableau.age)
    present = present_mask(game.tableau, layout)
    total = bin(present).count("1")
    if not total:
        return {}
    on_move = state_actor(game) == seat
    now = tempo_state(game, seat)
    solver = ControlSolver(game.tableau.age)

    def frac(tempo):
        return len(control_map(
            present, game.tableau.age, on_move, tempo, solver
        )) / total

    return {
        # Reachability under the Wonder pool exactly as it stands.
        "control_under_topology": frac(now),
        # What one more unspent extra-turn Wonder would be worth -- the lever
        # the reference case turns on.
        "control_with_one_more_tempo": frac(_plus_extra(now, ATTACKER)),
        # What losing one would cost.
        "control_if_i_spend_tempo": frac(_minus_extra(now, ATTACKER)),
        # What each side taking Theology would be worth.
        "control_if_opponent_takes_theology": frac(_as_theology(now, DEFENDER)),
        "control_if_i_take_theology": frac(_as_theology(now, ATTACKER)),
        # The pool itself, since reachability is non-monotone in a DIFFERENCE
        # of tempo and the network needs the raw counts too.
        "wonder_builds_remaining": now[0],
        "my_ordinary_wonders": now[1],
        "my_extra_turn_wonders": now[2],
        "their_ordinary_wonders": now[3],
        "their_extra_turn_wonders": now[4],
        "on_move": float(on_move),
        "slots_present": total,
    }
