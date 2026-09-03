"""Exact positional control: correctness, and the case it was built for.

Everything here runs on a clean checkout. The two reference positions live in
`fixtures/control_positions.json` as minimized PUBLIC information -- age, the
present/absent slot list, the revealed card names and the shared Wonder pool --
which is all the solver is allowed to read anyway. The BGA logs are used only by
the tests that check we agree with the live engine, and those stay conditional.

The load-bearing tests:

* `test_extra_turn_is_what_flips_control` -- on the reviewed position the
  opponent takes `School` in one turn WITH an extra-turn Wonder and cannot
  without it. That is the mechanic the encoder's feasibility features cannot
  express, stated as a topology fact.
* `test_seventh_wonder_retires_the_opponents_tempo` -- with six Wonders built
  the two players do NOT hold one tempo each. The pool holds one build, and
  spending it on an ordinary Wonder retires the opponent's extra-turn Wonder.
  Modelling those budgets independently is what made the first version misread
  table 907773062.
* `test_solver_reuse_matches_a_fresh_solver` -- the memo key must identify the
  target, and cached values must be relative to their node.
"""

from __future__ import annotations

import json
import random
import time

import pytest

from .game import TableauState
from .tableau_control import (
    ATTACKER,
    DEFENDER,
    ControlSolver,
    Layout,
    control_map,
    coupons,
    decisions_until_accessible,
    must_open,
    present_mask,
    tempo_state,
)
from .threat_corpus_scan import REPO_ROOT

LOG_DIR = REPO_ROOT / "runs/seven_wonders_duel/bga_game_log"
FIXTURES = REPO_ROOT / "games/seven_wonders_duel/fixtures/control_positions.json"
_INF = 99

needs_logs = pytest.mark.skipif(
    not LOG_DIR.exists(), reason="BGA game logs are not present"
)


# -- fixtures ---------------------------------------------------------------


def fixture(name: str) -> dict:
    """One minimized public position. No BGA log, no card identities needed."""

    data = json.loads(FIXTURES.read_text(encoding="utf-8"))[name]
    layout = Layout.for_age(data["age"])
    mask = 0
    for row, x in data["present"]:
        mask |= 1 << layout.index[(row, x)]
    return {
        "layout": layout,
        "age": data["age"],
        "actor": data["actor"],
        "present": mask,
        "slot": {n: tuple(s) for n, s in data["revealed_slots"].items()},
        "tempo": {int(k): tuple(v) for k, v in data["tempo_by_attacker"].items()},
        "wonders_built": data["wonders_built"],
    }


def load(table: str, row: int):
    """A real game state. Only the engine-parity tests need this."""

    from .advisor_scrape import determinize_observation, observation_from_wire
    from .bga_extract import wire_from_bga_payload

    path = LOG_DIR / f"table_{table}.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    decisions = [r for r in rows if r.get("kind") == "decision"]
    payload = wire_from_bga_payload(decisions[row]["state"])
    obs = observation_from_wire(payload["observation"])
    return determinize_observation(
        obs, random.Random(0),
        unknown_burial_ages=tuple(int(a) for a in payload.get("unknown_burial_ages", ())),
    )


# -- geometry ---------------------------------------------------------------


def reachable_masks(layout: Layout, full: int, depth: int) -> set:
    """Every mask reachable from `full` by up to `depth` legal removals."""

    frontier, seen = {full}, {full}
    for _ in range(depth):
        nxt = set()
        for mask in frontier:
            accessible = layout.accessible(mask)
            while accessible:
                bit = accessible & -accessible
                accessible ^= bit
                child = mask ^ bit
                if child not in seen:
                    seen.add(child)
                    nxt.add(child)
        frontier = nxt
    return seen


@pytest.mark.parametrize("age", (1, 2, 3))
def test_accessible_matches_the_engine_on_every_reachable_mask(age):
    """The bitmask accessibility must agree with `TableauState.is_accessible`;
    a private reimplementation of the cover rule is where drift hides.

    Exhaustive over every position reachable in four removals, plus random deep
    masks -- the full 2**20 is not enumerable, but four plies covers every
    cover-relation shape in the layout."""

    layout = Layout.for_age(age)
    n = len(layout.slots)
    tableau = TableauState.from_deck(age, tuple(f"c{i}" for i in range(n)))
    full = (1 << n) - 1

    masks = reachable_masks(layout, full, 4)
    rng = random.Random(11)
    masks |= {rng.getrandbits(n) for _ in range(400)}

    for mask in masks:
        for slot, card in tableau.cards.items():
            card.present = bool(mask >> layout.index[slot] & 1)
        expected = 0
        for slot in layout.slots:
            if tableau.is_accessible(slot):
                expected |= 1 << layout.index[slot]
        assert layout.accessible(mask) == expected, f"age {age} mask {mask:b}"


def test_must_open_is_true_when_every_move_uncovers_the_target():
    """A true positive, not just the absence of false ones: a position holding
    only the target and its coverers forces the exposure."""

    layout = Layout.for_age(2)
    target = (0, 1)
    bit = layout.index[target]
    mask = (1 << bit) | layout.covered_by[bit]
    assert layout.covered_by[bit], "target must actually be covered"
    assert must_open(mask, target, layout, ATTACKER)
    # ...and one unrelated accessible slot elsewhere destroys the claim.
    spare = next(
        i for i, s in enumerate(layout.slots)
        if not (mask >> i & 1) and not (layout.covered_by[i] & mask)
    )
    assert not must_open(mask | (1 << spare), target, layout, ATTACKER)


def test_absent_target_is_not_a_control_question():
    f = fixture("908370787#17")
    gone = next(
        s for s in f["layout"].slots
        if not (f["present"] >> f["layout"].index[s] & 1)
    )
    solver = ControlSolver(f["age"])
    assert solver.solve(f["present"], gone, ATTACKER, coupons(1, 1)) == _INF


# -- an independent oracle --------------------------------------------------


def brute_force(present: frozenset, target, to_move: int, tempo: tuple,
                cover: dict) -> int:
    """A deliberately naive reimplementation: sets, no memo, no bit tricks.

    Written from the rules rather than from `ControlSolver`, so that agreement
    between the two is evidence and not a tautology. Exponential -- only ever
    called on a handful of slots.
    """

    builds, counts = tempo[0], list(tempo[1:])

    def spend(player, extra):
        c = list(counts)
        c[2 * player + (1 if extra else 0)] -= 1
        return (0, 0, 0, 0, 0) if builds - 1 <= 0 else (builds - 1, *c)

    accessible = [s for s in present if not (cover[s] & present)]
    if not accessible:
        return _INF

    values = []
    for slot in accessible:
        rest = present - {slot}
        if slot == target:
            values.append(1 if to_move == ATTACKER else _INF)
            continue
        moves = [tempo]
        if builds > 0 and counts[2 * to_move] > 0:
            moves.append(spend(to_move, False))
        for nxt in moves:                       # the turn passes
            got = brute_force(rest, target, 1 - to_move, nxt, cover)
            values.append(got if got >= _INF else got + (to_move == ATTACKER))
        if builds > 0 and counts[2 * to_move + 1] > 0:   # move again
            values.append(
                brute_force(rest, target, to_move, spend(to_move, True), cover)
            )
    return min(values) if to_move == ATTACKER else max(values)


@pytest.mark.parametrize("seed", range(12))
def test_solver_agrees_with_the_brute_force_oracle(seed):
    """Exact means exact: check it against an independent implementation."""

    rng = random.Random(seed)
    age = rng.choice((1, 2, 3))
    layout = Layout.for_age(age)
    n = len(layout.slots)
    keep = sorted(rng.sample(range(n), 6))
    mask = sum(1 << i for i in keep)

    cover = {
        layout.slots[i]: frozenset(
            layout.slots[j] for j in range(n)
            if layout.covered_by[i] >> j & 1 and mask >> j & 1
        )
        for i in keep
    }
    cover = {k: frozenset(v) for k, v in cover.items()}
    present = frozenset(layout.slots[i] for i in keep)

    def covers(slot, live):
        return bool(cover[slot] & live)

    tempo = (rng.randint(0, 2), rng.randint(0, 1), rng.randint(0, 1),
             rng.randint(0, 1), rng.randint(0, 1))
    solver = ControlSolver(age)
    for target in present:
        for to_move in (ATTACKER, DEFENDER):
            want = brute_force(present, target, to_move, tempo, cover)
            got = solver.solve(mask, target, to_move, tempo)
            assert got == want, (age, keep, target, to_move, tempo, got, want)


# -- memoization ------------------------------------------------------------


@pytest.mark.parametrize("tempo", [coupons(4, 4), coupons(1, 0), (2, 1, 1, 1, 1)])
def test_solver_reuse_matches_a_fresh_solver(tempo):
    """The regression for the memo bug: a cached value must be identified by
    its target and be relative to its node, or one query answers another.

    Before the fix, asking a reused solver for a second Age III target returned
    the first target's distance -- reporting control of a slot the defender
    denies outright."""

    layout = Layout.for_age(3)
    full = (1 << len(layout.slots)) - 1
    shared = ControlSolver(3)
    positions = [full] + sorted(reachable_masks(layout, full, 2) - {full})[:4]
    for mask in positions:
        for to_move in (ATTACKER, DEFENDER):
            for target in layout.slots:
                if not (mask >> layout.index[target] & 1):
                    continue
                assert shared.solve(mask, target, to_move, tempo) == \
                    ControlSolver(3).solve(mask, target, to_move, tempo)


def test_reuse_saves_work():
    """If reuse were not a saving there would be no reason to make it safe.

    Be precise about WHERE. Putting the target in the key is what makes reuse
    correct, and it also means two different targets share nothing -- a map is
    not cheaper for being solved by one solver. What reuse buys is the same
    target across root positions of the same turn parity, which is exactly the
    access pattern a precomputed structural cache would have. (A root reached
    with the other side to move shares nothing either: `to_move` is in the key
    and flips all the way down.)"""

    layout = Layout.for_age(3)
    full = (1 << len(layout.slots)) - 1
    tempo = coupons(2, 2)
    target = layout.slots[3]

    # Two plies on from the full board: same side to move.
    accessible, bits = layout.accessible(full), []
    while accessible and len(bits) < 2:
        bit = accessible & -accessible
        accessible ^= bit
        bits.append(bit)
    later = full ^ bits[0] ^ bits[1]

    shared = ControlSolver(3)
    shared.solve(full, target, ATTACKER, tempo)
    shared.solve(later, target, ATTACKER, tempo)

    a, b = ControlSolver(3), ControlSolver(3)
    a.solve(full, target, ATTACKER, tempo)
    b.solve(later, target, ATTACKER, tempo)
    assert shared.nodes < a.nodes + b.nodes, (shared.nodes, a.nodes, b.nodes)


# -- the reference case -----------------------------------------------------


def test_extra_turn_is_what_flips_control():
    """The mechanic the feasibility features cannot express.

    After `Caravansery` goes, `r2c10` covers `School`. The opponent to move can
    bury `r2c10` under an extra-turn Wonder and take `School` in the SAME turn.
    Without that Wonder it needs two turns, and the actor moves in between.
    """

    f = fixture("908370787#17")
    after = f["present"] ^ (1 << f["layout"].index[f["slot"]["Caravansery"]])
    school = f["slot"]["School"]

    with_extra = ControlSolver(f["age"]).solve(after, school, ATTACKER, coupons(1, 0))
    without = ControlSolver(f["age"]).solve(after, school, ATTACKER, coupons(0, 0))

    assert with_extra == 1, "one extra-turn Wonder should buy School in one turn"
    assert without > with_extra, "without the Wonder it must cost more"


def test_school_is_one_removal_away_after_the_discard():
    f = fixture("908370787#17")
    after = f["present"] ^ (1 << f["layout"].index[f["slot"]["Caravansery"]])
    assert decisions_until_accessible(after, f["slot"]["School"], f["layout"]) == 1


def test_taking_the_target_first_denies_it():
    """A defender who can reach the slot first denies it outright."""

    f = fixture("908370787#17")
    school = f["slot"]["School"]
    solver = ControlSolver(f["age"])
    assert solver.solve(f["present"], school, DEFENDER, coupons(0, 0)) == _INF


def test_extra_turns_never_make_control_worse():
    """Monotonicity: more tempo cannot delay the attacker."""

    f = fixture("908370787#17")
    solver = ControlSolver(f["age"])
    for slot in f["layout"].slots:
        if not (f["present"] >> f["layout"].index[slot] & 1):
            continue
        none = solver.solve(f["present"], slot, ATTACKER, coupons(0, 0))
        one = solver.solve(f["present"], slot, ATTACKER, coupons(1, 0))
        two = solver.solve(f["present"], slot, ATTACKER, coupons(2, 0))
        assert one <= none and two <= one


# -- the shared Wonder pool -------------------------------------------------


def test_six_wonders_built_leaves_one_shared_build_not_one_each():
    """The independent-budget bug, stated as a fact about the fixture."""

    f = fixture("907773062#28")
    assert f["wonders_built"] == [3, 3] or sum(f["wonders_built"]) == 6
    for attacker in (0, 1):
        tempo = f["tempo"][attacker]
        assert tempo[0] == 1, "one build left in the shared pool"
        assert sum(tempo[1:]) >= 2, "but both players still hold unbuilt Wonders"


def test_seventh_wonder_retires_the_opponents_tempo():
    """Spending the last build on an ordinary Wonder erases the opponent's
    extra-turn Wonder. That is `The Pyramids` retiring `The Sphinx`, and it is
    why a per-player tempo budget misreads this position.

    The three cases below differ in one variable at a time on the same board,
    so the drop cannot be blamed on pool size alone:

        (1, 0,1, 1,0)  one build left, defender HOLDS an ordinary Wonder
        (2, 0,1, 1,0)  same holdings, pool big enough for both
        (1, 0,1, 0,0)  one build left, defender holds NOTHING to spend

    Only the first lets the defender close the pool, and only the first collapses
    the attacker's reach. A per-player tempo budget scores all three alike."""

    layout = Layout.for_age(3)
    full = (1 << len(layout.slots)) - 1
    solver = ControlSolver(3)
    # Defender on move: with the attacker to move first it reaches the whole
    # board under any of these, and the effect has no room to show.
    reach = lambda tempo: set(control_map(full, 3, False, tempo, solver))

    retired = reach((1, 0, 1, 1, 0))
    roomy = reach((2, 0, 1, 1, 0))
    nothing_to_spend = reach((1, 0, 1, 0, 0))

    assert len(retired) < len(nothing_to_spend), (
        "the defender must be able to retire the attacker's extra turn"
    )
    assert len(retired) < len(roomy), "a bigger pool must survive the same threat"
    assert retired <= nothing_to_spend and retired <= roomy


def test_an_empty_pool_grants_no_tempo_at_all():
    """With seven Wonders built there is no eighth, whatever remains unbuilt."""

    f = fixture("907773062#29")
    assert sum(f["wonders_built"]) == 7
    for attacker in (0, 1):
        assert f["tempo"][attacker] == (0, 0, 0, 0, 0)

    layout = Layout.for_age(3)
    full = (1 << len(layout.slots)) - 1
    solver = ControlSolver(3)
    assert control_map(full, 3, True, (0, 0, 0, 0, 0), solver) == \
        control_map(full, 3, True, coupons(0, 0), solver)


def test_coupons_are_an_optimistic_bound_on_a_real_pool():
    """`coupons(k, 0)` cannot be beaten by any real pool giving k extra turns:
    the coupon mode is exactly the same holding with retirement switched off."""

    layout = Layout.for_age(3)
    full = (1 << len(layout.slots)) - 1
    solver = ControlSolver(3)
    real = set(control_map(full, 3, True, (1, 0, 1, 1, 0), solver))
    abstract = set(control_map(full, 3, True, coupons(1, 0), solver))
    assert real <= abstract


# -- cost -------------------------------------------------------------------


def test_a_full_control_map_is_affordable():
    """The whole point is that this is small. Bound the FULL map -- one target
    is not the quantity anyone pays for."""

    layout = Layout.for_age(3)
    full = (1 << len(layout.slots)) - 1
    solver = ControlSolver(3)
    start = time.perf_counter()
    control_map(full, 3, True, (4, 1, 1, 1, 1), solver)
    elapsed = time.perf_counter() - start
    assert solver.nodes < 400_000, f"state space blew up: {solver.nodes} nodes"
    assert elapsed < 5.0, f"full map took {elapsed:.2f}s"


# -- agreement with the live engine (needs the BGA logs) --------------------


@needs_logs
def test_fixtures_match_the_live_positions():
    """The committed fixtures are derived data; check they still describe the
    games they were minimized from."""

    for name in ("908370787#17", "907773062#28", "907773062#29"):
        table, row = name.split("#")
        game = load(table, int(row))
        f = fixture(name)
        layout = Layout.for_age(game.tableau.age)
        assert game.tableau.age == f["age"]
        assert present_mask(game.tableau, layout) == f["present"]
        for attacker in (0, 1):
            assert tempo_state(game, attacker) == f["tempo"][attacker]


@needs_logs
def test_public_only_no_identity_is_read():
    """Two determinizations of the same public position must give identical
    answers -- the solver is called on determinized states and must not leak."""

    a = load("908370787", 17)
    b = load("908370787", 17)
    for slot, card in b.tableau.cards.items():
        if card.present and not card.revealed:
            card.card_name = "Brewery"  # scramble every hidden identity

    layout = Layout.for_age(a.tableau.age)
    ma, mb = present_mask(a.tableau, layout), present_mask(b.tableau, layout)
    assert ma == mb
    target = (1, 10)
    sa = ControlSolver(a.tableau.age).solve(ma, target, ATTACKER, coupons(1, 0))
    sb = ControlSolver(b.tableau.age).solve(mb, target, ATTACKER, coupons(1, 0))
    assert sa == sb


# -- independent evidence, and the shape of the tempo space ------------------


def sequence_oracle(present: frozenset, target, to_move: int, tempo: tuple,
                    cover: dict, cap: int = 8) -> int:
    """A THIRD implementation, deliberately not a minimax.

    `brute_force` above and `ControlSolver` share a shape -- recursive min/max
    over successors -- so a misconception about the game could live in both.
    This one iteratively deepens on the question "can the attacker take the
    target using at most k of its own turns", where the defender must survive
    EVERY reply. Same answer, different reasoning.
    """

    def legal(state):
        live, mover, holding = state
        builds, counts = holding[0], list(holding[1:])
        out = []
        for slot in (s for s in live if not (cover[s] & live)):
            rest = live - {slot}
            out.append((slot, "card", (rest, 1 - mover, holding)))
            for extra in (False, True):
                i = 2 * mover + (1 if extra else 0)
                if builds > 0 and counts[i] > 0:
                    c = list(counts)
                    c[i] -= 1
                    nxt = (0, 0, 0, 0, 0) if builds - 1 <= 0 else (builds - 1, *c)
                    out.append((slot, "extra" if extra else "ord",
                                (rest, mover if extra else 1 - mover, nxt)))
        return out

    def within(state, turns):
        live, mover, _ = state
        if target not in live:
            return False
        moves = legal(state)
        if not moves:
            return False
        if mover == ATTACKER:
            if turns <= 0:
                return False
            return any(
                slot == target
                or within(nxt, turns - (0 if kind == "extra" else 1))
                for slot, kind, nxt in moves
            )
        # The defender chooses, so the attacker needs EVERY reply to still
        # work. (Writing this as `not within(...)` inverts the quantifier and
        # silently turns the oracle into a different question.)
        return all(
            slot != target and within(nxt, turns)
            for slot, kind, nxt in moves
        )

    for k in range(1, cap + 1):
        if within((present, to_move, tempo), k):
            return k
    return _INF


@pytest.mark.parametrize("seed", range(6))
def test_solver_agrees_with_a_structurally_different_oracle(seed):
    """Two agreeing implementations of the same shape are weaker evidence than
    two of different shapes."""

    rng = random.Random(1000 + seed)
    age = rng.choice((1, 2, 3))
    layout = Layout.for_age(age)
    n = len(layout.slots)
    keep = sorted(rng.sample(range(n), 5))
    mask = sum(1 << i for i in keep)
    cover = {
        layout.slots[i]: frozenset(
            layout.slots[j] for j in range(n)
            if layout.covered_by[i] >> j & 1 and mask >> j & 1
        )
        for i in keep
    }
    present = frozenset(layout.slots[i] for i in keep)
    tempo = (rng.randint(0, 3), rng.randint(0, 1), rng.randint(0, 2),
             rng.randint(0, 1), rng.randint(0, 2))

    solver = ControlSolver(age)
    for target in present:
        for to_move in (ATTACKER, DEFENDER):
            assert solver.solve(mask, target, to_move, tempo) == \
                sequence_oracle(present, target, to_move, tempo, cover), \
                (age, keep, target, to_move, tempo)


@pytest.mark.parametrize("age", (1, 2, 3))
def test_tempo_monotonicity_with_the_pool_held_fixed(age):
    """Properties that hold for any correct solver, checked without an oracle.

    The pool must be held FIXED in all of them -- see the non-monotonicity test
    below for why."""

    rng = random.Random(77 + age)
    layout = Layout.for_age(age)
    mask = (1 << len(layout.slots)) - 1
    solver = ControlSolver(age)

    for _ in range(12):
        for _ in range(rng.randint(0, 6)):
            accessible = layout.accessible(mask)
            if not accessible:
                break
            bits = [b for b in range(len(layout.slots)) if accessible >> b & 1]
            mask ^= 1 << rng.choice(bits)
        live = [s for s in layout.slots if mask >> layout.index[s] & 1]
        if not live:
            break
        target = rng.choice(live)
        to_move = rng.choice((ATTACKER, DEFENDER))
        base = (rng.randint(1, 3), rng.randint(0, 1), rng.randint(0, 1),
                rng.randint(0, 1), rng.randint(0, 1))
        here = solver.solve(mask, target, to_move, base)

        def shift(index):
            counts = list(base[1:])
            counts[index - 1] += 1
            return solver.solve(mask, target, to_move, (base[0], *counts))

        # An extra OPTION for the attacker can never hurt it.
        assert shift(1) <= here, ("attacker ordinary Wonder", base, target)
        assert shift(2) <= here, ("attacker extra-turn Wonder", base, target)
        # An option for the defender can never help the attacker: a defender
        # ordinary Wonder costs it no turns and threatens retirement.
        assert shift(3) >= here, ("defender ordinary Wonder", base, target)
        assert shift(4) >= here, ("defender extra-turn Wonder", base, target)


def test_pool_size_is_deliberately_non_monotone():
    """Raising `builds_left` helps BOTH players, so it moves control in either
    direction. That is the retirement mechanic, not a defect -- pinned here so
    a future "monotonicity fix" has to argue with a test.

    Witnesses are searched for rather than hardcoded: the effect depends on the
    board as much as on the pool, and a tuple copied without its mask asserts
    something about a different position.
    """

    rng = random.Random(4242)
    layout = Layout.for_age(1)
    solver = ControlSolver(1)
    full = (1 << len(layout.slots)) - 1

    hurt = helped = None
    for _ in range(300):
        mask = full
        for _ in range(rng.randint(0, 8)):
            accessible = layout.accessible(mask)
            if not accessible:
                break
            bits = [b for b in range(len(layout.slots)) if accessible >> b & 1]
            mask ^= 1 << rng.choice(bits)
        live = [s for s in layout.slots if mask >> layout.index[s] & 1]
        if not live:
            continue
        target = rng.choice(live)
        small = (1, rng.randint(0, 1), rng.randint(0, 1),
                 rng.randint(0, 1), rng.randint(0, 1))
        for grow in (2, 4):                # which side also gains the extra
            big = list(small)
            big[0] += 1
            big[grow] += 1
            a = solver.solve(mask, target, ATTACKER, small)
            b = solver.solve(mask, target, ATTACKER, tuple(big))
            if b > a and hurt is None:
                hurt = (mask, target, small, tuple(big), a, b)
            if b < a and helped is None:
                helped = (mask, target, small, tuple(big), a, b)
        if hurt and helped:
            break

    assert hurt, "a bigger pool must be able to HURT the attacker (retirement)"
    assert helped, "a bigger pool must be able to HELP the attacker"
    # Re-check each witness so the assertion names a concrete position.
    for mask, target, small, big, a, b in (hurt, helped):
        assert solver.solve(mask, target, ATTACKER, small) == a
        assert solver.solve(mask, target, ATTACKER, big) == b


@needs_logs
def test_tempo_state_matches_the_engines_seventh_wonder_rule():
    """The engine retires exactly ONE Wonder on the seventh build
    (`engine.py`: "seventh Wonder must leave exactly one unbuilt Wonder"),
    while `_spend` collapses every counter to zero. Those agree only because
    eight Wonders are drafted and seven get built, so the unbuilt count is
    always `builds_left + 1`.

    That identity is the load-bearing assumption of the tempo model, and it is
    a fact about the draft rather than about this module -- so check it against
    real games instead of asserting it.
    """

    from .advisor_scrape import determinize_observation, observation_from_wire
    from .bga_extract import wire_from_bga_payload

    checked = 0
    for path in sorted(LOG_DIR.glob("table_*.jsonl"))[:12]:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in [r for r in rows if r.get("kind") == "decision"][::4]:
            try:
                payload = wire_from_bga_payload(row["state"])
                obs = observation_from_wire(payload["observation"])
                game = determinize_observation(
                    obs, random.Random(0),
                    unknown_burial_ages=tuple(
                        int(a) for a in payload.get("unknown_burial_ages", ())
                    ),
                )
            except Exception:
                continue
            if sum(len(c.wonders) for c in game.cities) < 8:
                continue          # the draft is not finished
            tempo = tempo_state(game, 0)
            if tempo[0] == 0:
                continue          # pool exhausted: the retirement already fired
            assert sum(tempo[1:]) == tempo[0] + 1, (path.name, tempo)
            checked += 1
    assert checked > 50, f"only {checked} positions checked"
