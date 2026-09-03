"""The precomputed control table must be indistinguishable from the solver.

A shipped table is a cache with no invalidation story of its own, so the tests
that matter are: does it agree with the thing it caches, does it cover the keys
real games produce, and does it fail loudly rather than defaulting when it does
not. A zero-filled control feature is not a missing feature -- it reads as "the
opponent gets everything", which is a confident lie.
"""

from __future__ import annotations

import json
import random

import pytest

from .control_table import (
    ABSENT,
    build,
    DEFAULT_DIR,
    MAX_TURNS,
    SCHEMA_VERSION,
    TEMPO_STATES,
    UNREACH,
    ControlTable,
    MissingControlEntry,
    reachable_masks,
    rule_identity,
    tempo_states,
)
from .game import Phase
from .loop_adapter import SevenWondersDuelLoopAdapter
from .tableau_control import (
    ATTACKER,
    DEFENDER,
    ControlSolver,
    Layout,
    _INF,
    _as_theology,
    present_mask,
    tempo_state,
)

#: Tests that need the COMPLETE shipped artifact (coverage of real games, the
#: full manifest contract). Everything else runs against `table` below.
needs_full_table = pytest.mark.skipif(
    not (DEFAULT_DIR / "manifest.json").exists(),
    reason="full control table not generated (python -m ...control_table build)",
)

#: Tempo states the subset fixture covers -- spread across the enumeration
#: rather than clustered, so the sample is not all low-tempo.
_SUBSET_TEMPO = [0, 37, 91, 145, 199, len(TEMPO_STATES) - 1]
_SUBSET_AGES = (3,)


@pytest.fixture(scope="session")
def table(tmp_path_factory):
    """The shipped table if it exists, otherwise a small one built on the spot.

    The 8.7 MB artifact is regenerable in minutes and deliberately not committed,
    but "not committed" must not mean "untested on a clean checkout" -- that was
    a finding against this workstream's first test suite. A subset table costs a
    couple of seconds and exercises every correctness path.
    """

    if (DEFAULT_DIR / "manifest.json").exists():
        return ControlTable.load()
    out = tmp_path_factory.mktemp("control_table")
    build(out, jobs=1, log=lambda *_: None,
          ages=_SUBSET_AGES, tempo_indices=_SUBSET_TEMPO)
    return ControlTable.load(out)


def _covered(tbl):
    """(ages, tempo states) this table actually holds."""

    return ([int(a) for a in tbl.manifest["ages"]],
            [tuple(t) for t in tbl.manifest["tempo_states"]])


# -- the key space, which needs no table -----------------------------------


def test_reachable_masks_are_the_down_set_not_the_power_set():
    """The whole cost argument rests on this: the cover relation is a partial
    order, so only its down-set is reachable."""

    counts = {age: len(reachable_masks(age)) for age in (1, 2, 3)}
    assert counts == {1: 428, 2: 428, 3: 132}, counts
    for age, masks in ((a, reachable_masks(a)) for a in (1, 2, 3)):
        layout = Layout.for_age(age)
        assert masks[-1] == (1 << len(layout.slots)) - 1  # the full pyramid
        assert 0 in masks                                  # and the empty one
        assert len(set(masks)) == len(masks)


def test_tempo_set_is_closed_under_theology():
    """The Theology counterfactual must not escape the generated set, or the
    encoder's second and third maps would be table misses in ordinary play."""

    states = set(tempo_states())
    for state in states:
        for player in (ATTACKER, DEFENDER):
            assert _as_theology(state, player) in states, (state, player)


def test_tempo_states_obey_the_draft_identity():
    """`builds_left == total_unbuilt - 1` -- eight Wonders drafted, seven built.
    This is the identity that makes collapsing every counter on the seventh
    build equivalent to the engine retiring the single survivor."""

    for builds, ord_a, ext_a, ord_d, ext_d in TEMPO_STATES:
        total = ord_a + ext_a + ord_d + ext_d
        assert builds == min(7, total - 1)
        assert ord_a + ext_a <= 4 and ord_d + ext_d <= 4


def test_key_count_matches_the_plan():
    keys = sum(len(reachable_masks(a)) for a in (1, 2, 3)) * 2 * len(TEMPO_STATES)
    assert keys == 434_720, keys


# -- agreement with the solver ---------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_table_agrees_with_the_live_solver(table, seed):
    """The one test that makes the table trustworthy."""

    ages, tempos = _covered(table)
    rng = random.Random(500 + seed)
    for _ in range(6):
        age = rng.choice(ages)
        masks = reachable_masks(age)
        mask = rng.choice(masks)
        tempo = rng.choice(tempos)
        first = rng.choice((True, False))
        layout = Layout.for_age(age)
        cells = table.lookup(age, mask, first, tempo)
        solver = ControlSolver(age)
        for i, slot in enumerate(layout.slots):
            if not (mask >> i) & 1:
                assert cells[i] == ABSENT
                continue
            turns = solver.solve(mask, slot, ATTACKER if first else DEFENDER, tempo)
            want = UNREACH if turns >= _INF else turns
            assert cells[i] == want, (age, hex(mask), slot, tempo, first,
                                      int(cells[i]), want)


def test_sentinels_never_collide_with_a_distance(table):
    """`_INF` is 99 and must never reach the encoder as a turn count."""

    for age in _covered(table)[0]:
        plane = table._planes[age]
        real = plane[(plane != ABSENT) & (plane != UNREACH)]
        assert real.size
        assert int(real.max()) <= MAX_TURNS
        assert int(real.min()) >= 0


def test_absent_cells_are_exactly_the_absent_slots(table):
    ages, tempos = _covered(table)
    rng = random.Random(7)
    for age in ages:
        layout = Layout.for_age(age)
        for mask in rng.sample(reachable_masks(age), 12):
            for first in (True, False):
                cells = table.lookup(age, mask, first, tempos[0])
                for i in range(len(layout.slots)):
                    present = bool((mask >> i) & 1)
                    assert (cells[i] != ABSENT) == present, (age, hex(mask), i)


# -- coverage of what real games produce -----------------------------------


@needs_full_table
def test_every_key_a_real_game_produces_is_covered():
    """A miss in production is a crash, so the table must cover ordinary play.

    Only clean `PLAY_AGE` states are checked, because those are the only ones
    the encoder will look up -- see the applicability contract in
    `W3_ENCODER_INTEGRATION_REVIEW_REQUEST.md`."""

    table = ControlTable.load()
    adapter = SevenWondersDuelLoopAdapter()
    rng = random.Random(3)
    checked = 0
    for game_i in range(12):
        game = adapter.new_game(seed=70_000 + game_i)
        guard = 0
        while not adapter.terminal(game) and guard < 500:
            guard += 1
            legal = adapter.legal_actions(game)
            if not legal:
                break
            if game.phase is Phase.PLAY_AGE and game.pending_choice is None:
                for seat in (0, 1):
                    layout = Layout.for_age(game.tableau.age)
                    mask = present_mask(game.tableau, layout)
                    tempo = tempo_state(game, seat)
                    if tempo[0] > 0:            # closed pools are not looked up
                        table.lookup(game.tableau.age, mask, True, tempo)
                        checked += 1
            game = adapter.step(game, rng.choice(legal))
    assert checked > 500, f"only {checked} keys exercised"


# -- failing loudly ---------------------------------------------------------


def test_a_miss_raises_and_never_defaults(table):
    """Zero-filling a control feature reads as 'the opponent gets everything'."""

    tempos = _covered(table)[1]
    layout = Layout.for_age(3)
    unreachable = next(
        m for m in range(1 << len(layout.slots))
        if m not in set(reachable_masks(3))
    )
    with pytest.raises(MissingControlEntry):
        table.lookup(3, unreachable, True, tempos[0])
    with pytest.raises(MissingControlEntry):
        table.lookup(3, reachable_masks(3)[0], True, (9, 9, 9, 9, 9))


def test_contract_check_rejects_a_table_from_different_rules(table):
    """A git commit is provenance; this is the compatibility check."""

    table.check_contract()                        # matching code: silent
    with pytest.raises(MissingControlEntry):
        table.check_contract(rule_identity_expected="0" * 64)


@needs_full_table
def test_manifest_pins_the_contract():
    manifest = json.loads((DEFAULT_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["rule_identity"] == rule_identity()
    assert manifest["keys"] == 434_720
    assert len(manifest["tempo_states"]) == len(TEMPO_STATES)
    assert manifest["cell_encoding"]["UNREACH"] == UNREACH
