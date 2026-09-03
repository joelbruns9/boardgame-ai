"""Exact positional control: correctness, and the case it was built for.

The load-bearing test is `test_extra_turn_is_what_flips_control`: on the reviewed
position, the opponent can take `School` in one decision WITH an extra-turn
Wonder and cannot without it. That is the mechanic the encoder's feasibility
features cannot express, stated as a topology fact.
"""

from __future__ import annotations

import json
import random

import pytest

from .advisor_scrape import determinize_observation, observation_from_wire
from .bga_extract import wire_from_bga_payload
from .tableau_control import (
    ATTACKER,
    DEFENDER,
    ControlSolver,
    Layout,
    decisions_until_accessible,
    must_open,
    present_mask,
)
from .threat_corpus_scan import REPO_ROOT

LOG_DIR = REPO_ROOT / "runs/seven_wonders_duel/bga_game_log"
_INF = 99

pytestmark = pytest.mark.skipif(
    not LOG_DIR.exists(), reason="BGA game logs are not present"
)


def load(table: str, row: int):
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


def test_accessible_matches_the_engine():
    """The bitmask accessibility must agree with `TableauState.is_accessible`;
    a private reimplementation of the cover rule is where drift hides."""

    for table, row in (("908370787", 17), ("907773062", 29)):
        game = load(table, row)
        layout = Layout.for_age(game.tableau.age)
        mask = present_mask(game.tableau, layout)
        got = layout.accessible(mask)
        expected = 0
        for slot, card in game.tableau.cards.items():
            if card.present and game.tableau.is_accessible(slot):
                expected |= 1 << layout.index[slot]
        assert got == expected


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
    sa = ControlSolver(a.tableau.age).solve(ma, target, ATTACKER, (1, 0))
    sb = ControlSolver(b.tableau.age).solve(mb, target, ATTACKER, (1, 0))
    assert sa == sb


# -- the reference case -----------------------------------------------------


def test_extra_turn_is_what_flips_control():
    """The mechanic the feasibility features cannot express.

    After `Caravansery` goes, `r2c10` covers `School`. The opponent to move can
    bury `r2c10` under an extra-turn Wonder and take `School` in the SAME turn.
    Without that Wonder it needs two turns, and the actor moves in between.
    """

    game = load("908370787", 17)
    layout = Layout.for_age(game.tableau.age)
    mask = present_mask(game.tableau, layout)

    caravansery = next(
        slot for slot, c in game.tableau.cards.items()
        if c.present and c.revealed and c.card_name == "Caravansery"
    )
    school = next(
        slot for slot, c in game.tableau.cards.items()
        if c.present and c.revealed and c.card_name == "School"
    )
    after = mask ^ (1 << layout.index[caravansery])

    with_extra = ControlSolver(game.tableau.age).solve(after, school, ATTACKER, (1, 0))
    without = ControlSolver(game.tableau.age).solve(after, school, ATTACKER, (0, 0))

    assert with_extra == 1, "one extra-turn Wonder should buy School in one decision"
    assert without > with_extra, "without the Wonder it must cost more"


def test_school_is_one_removal_away_after_the_discard():
    game = load("908370787", 17)
    layout = Layout.for_age(game.tableau.age)
    mask = present_mask(game.tableau, layout)
    caravansery = next(
        slot for slot, c in game.tableau.cards.items()
        if c.present and c.revealed and c.card_name == "Caravansery"
    )
    school = next(
        slot for slot, c in game.tableau.cards.items()
        if c.present and c.revealed and c.card_name == "School"
    )
    after = mask ^ (1 << layout.index[caravansery])
    assert decisions_until_accessible(after, school, layout) == 1


# -- semantics --------------------------------------------------------------


def test_taking_the_target_first_denies_it():
    """A defender who can reach the slot first denies it outright."""

    game = load("908370787", 17)
    layout = Layout.for_age(game.tableau.age)
    mask = present_mask(game.tableau, layout)
    school = next(
        slot for slot, c in game.tableau.cards.items()
        if c.present and c.revealed and c.card_name == "School"
    )
    # Defender on move, no tempo for either side.
    assert ControlSolver(game.tableau.age).solve(mask, school, DEFENDER, (0, 0)) == _INF


def test_absent_target_is_not_a_control_question():
    game = load("908370787", 17)
    layout = Layout.for_age(game.tableau.age)
    mask = present_mask(game.tableau, layout)
    gone = next(slot for slot, c in game.tableau.cards.items() if not c.present)
    assert ControlSolver(game.tableau.age).solve(mask, gone, ATTACKER, (1, 1)) == _INF


def test_extra_turns_never_make_control_worse():
    """Monotonicity: more tempo cannot delay the attacker."""

    game = load("908370787", 17)
    layout = Layout.for_age(game.tableau.age)
    mask = present_mask(game.tableau, layout)
    for slot, card in list(game.tableau.cards.items())[:6]:
        if not card.present:
            continue
        none = ControlSolver(game.tableau.age).solve(mask, slot, ATTACKER, (0, 0))
        one = ControlSolver(game.tableau.age).solve(mask, slot, ATTACKER, (1, 0))
        two = ControlSolver(game.tableau.age).solve(mask, slot, ATTACKER, (2, 0))
        assert one <= none and two <= one


def test_must_open_detects_a_forced_exposure():
    """`must_open` is true only when EVERY legal move uncovers the target --
    a stronger statement than 'some move exposes it'."""

    game = load("908370787", 17)
    layout = Layout.for_age(game.tableau.age)
    mask = present_mask(game.tableau, layout)
    accessible = layout.accessible(mask)
    # An accessible slot is never 'about to be opened'.
    for slot in layout.slots:
        if accessible & (1 << layout.index[slot]):
            assert not must_open(mask, slot, layout, ATTACKER)


def test_solver_is_tractable_on_a_full_age():
    """The whole point is that this is small. Bound it, do not assume it."""

    game = load("908370787", 17)
    layout = Layout.for_age(game.tableau.age)
    mask = present_mask(game.tableau, layout)
    school = next(
        slot for slot, c in game.tableau.cards.items()
        if c.present and c.revealed and c.card_name == "School"
    )
    solver = ControlSolver(game.tableau.age)
    solver.solve(mask, school, ATTACKER, (1, 1))
    assert solver.nodes < 200_000, f"state space blew up: {solver.nodes} nodes"
