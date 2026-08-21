"""Differential fuzz of the sheet geometry against a literal transcription of
the BGA PHP.

The reference implementations below are deliberately *transliterated* from
``BGA Files/welcometo`` — same loop structure, same sentinels, same off-by-one
opportunities — rather than rewritten idiomatically.  The point is that they can
diverge from ``sheet.py`` in exactly the ways a hand-port diverges, so fuzzing
them against each other over reachable sheets catches what reading cannot.

Sheets come from random play, so they are reachable states rather than synthetic
ones, and they carry real fences, bis runs and roundabouts.
"""
from __future__ import annotations

import random

import pytest

from games.welcome_to.constants import (
    FENCE_SIZES,
    NUM_STREETS,
    ROUNDABOUT,
    STREET_SIZES,
)
from games.welcome_to.game import GameConfig, GameState
from games.welcome_to.sheet import Sheet


# ──────────────────────────────────────────────────────────────────────────
# Reference implementations, transliterated from the BGA PHP
# ──────────────────────────────────────────────────────────────────────────
def php_available_locations(sheet: Sheet, number: int | None) -> list[tuple[int, int]]:
    """``Houses::getAvailableLocations`` (Houses.php)."""
    locations = [[False] * n for n in STREET_SIZES]
    for x in range(NUM_STREETS):
        max_number = -1
        for y in range(STREET_SIZES[x]):
            cell = sheet.numbers[x][y]
            if cell is None:
                locations[x][y] = number is None or number > max_number
            else:
                locations[x][y] = False
                max_number = -1 if cell == ROUNDABOUT else cell

    for x in range(NUM_STREETS):
        min_number = 18
        for y in range(STREET_SIZES[x] - 1, -1, -1):
            cell = sheet.numbers[x][y]
            if cell is None:
                locations[x][y] = number is None or (
                    locations[x][y] and number < min_number
                )
            else:
                min_number = 18 if cell == ROUNDABOUT else cell

    return [
        (x, y) for x in range(NUM_STREETS) for y in range(STREET_SIZES[x]) if locations[x][y]
    ]


def php_available_locations_for_bis(sheet: Sheet, number: int) -> list[tuple[int, int]]:
    """``Houses::getAvailableLocationsForBis`` (Houses.php).

    Note the ``$hole`` counter: the bis must sit *directly* beside the house it
    copies, and a fence resets the carried number, so a bis cannot cross a fence.
    """
    locations = [[False] * n for n in STREET_SIZES]

    for x in range(NUM_STREETS):
        max_number = -1
        hole = 0
        for y in range(STREET_SIZES[x]):
            cell = sheet.numbers[x][y]
            if cell is None:
                locations[x][y] = number == max_number and hole == 0
                hole += 1
            else:
                hole = 0
                max_number = -1 if cell == ROUNDABOUT else cell
            if y < FENCE_SIZES[x] and sheet.fences[x][y]:
                max_number = -1

    for x in range(NUM_STREETS):
        min_number = 18
        hole = 0
        for y in range(STREET_SIZES[x] - 1, -1, -1):
            cell = sheet.numbers[x][y]
            if cell is None:
                locations[x][y] = locations[x][y] or (number == min_number and hole == 0)
                hole += 1
            else:
                hole = 0
                min_number = 18 if cell == ROUNDABOUT else cell
            if y - 1 >= 0 and sheet.fences[x][y - 1]:
                min_number = 18

    return [
        (x, y) for x in range(NUM_STREETS) for y in range(STREET_SIZES[x]) if locations[x][y]
    ]


def php_surveyor_zones(sheet: Sheet) -> list[tuple[int, int]]:
    """``Actions/Surveyor::getAvailableZones``."""
    zones = []
    for i in range(NUM_STREETS):
        for j in range(FENCE_SIZES[i]):
            if sheet.fences[i][j]:
                continue
            if sheet.numbers[i][j] is not None:
                if sheet.top_fences[i][j] and sheet.top_fences[i][j + 1]:
                    continue
                nxt = sheet.numbers[i][j + 1]
                if nxt is not None and sheet.numbers[i][j] == nxt:
                    continue
            zones.append((i, j))
    return zones


def php_get_estates(sheet: Sheet) -> list[tuple[int, int, int]]:
    """``Actions/RealEstate::getEstates``."""
    estates = []
    for i in range(NUM_STREETS):
        start = 0
        full = True
        size = STREET_SIZES[i]
        for j in range(size):
            cell = sheet.numbers[i][j]
            if cell is None or cell == ROUNDABOUT:
                full = False
            if j == size - 1 or sheet.fences[i][j]:
                if full:
                    estates.append((i, start, j - start + 1))
                full = True
                start = j + 1
    return estates


# ──────────────────────────────────────────────────────────────────────────
# Corpus
# ──────────────────────────────────────────────────────────────────────────
def _sheet_corpus(games: int = 60) -> list[Sheet]:
    """Snapshot every sheet at every turn of random advanced games."""
    sheets: list[Sheet] = []
    for g in range(games):
        rng = random.Random(90210 + g)
        state = GameState.new(
            seed=4242 + g,
            config=GameConfig(players=2, advanced=True, solo_rules=False),
        )
        while not state.is_terminal:
            state = state.step(rng.choice(state.legal_actions()))
            for s in state.sheets:
                sheets.append(s)
    return sheets


CORPUS = _sheet_corpus()


def test_corpus_is_rich_enough_to_be_worth_fuzzing():
    assert len(CORPUS) > 5000
    assert any(s.roundabouts > 0 for s in CORPUS), "no roundabout ever built"
    assert any(s.bis_marks > 0 for s in CORPUS), "no bis ever written"
    assert any(any(f for f in s.fences[x]) for s in CORPUS for x in range(NUM_STREETS))


@pytest.mark.parametrize("number", [None, 0, 1, 7, 8, 15, 17])
def test_available_locations_matches_php(number):
    for sheet in CORPUS:
        assert sorted(sheet.available_locations(number)) == sorted(
            php_available_locations(sheet, number)
        ), sheet.pretty()


def test_bis_candidates_match_php():
    for sheet in CORPUS:
        ours = {(x, y, n) for x, y, n, _ in sheet.bis_candidates()}
        theirs = {
            (x, y, n)
            for n in range(0, 18)
            for x, y in php_available_locations_for_bis(sheet, n)
        }
        assert ours == theirs, sheet.pretty()


def test_surveyor_zones_match_php():
    for sheet in CORPUS:
        assert sorted(sheet.surveyor_zones()) == sorted(php_surveyor_zones(sheet))


def test_estates_match_php():
    for sheet in CORPUS:
        assert sorted(sheet.estates()) == sorted(php_get_estates(sheet))


# ──────────────────────────────────────────────────────────────────────────
# Named edge cases
# ──────────────────────────────────────────────────────────────────────────
def test_roundabout_may_go_in_any_empty_box_including_both_edges():
    """``argRoundabout`` passes ``null`` to ``getAvailableHousesForNumber``, so the
    ascending-order rule does not constrain a roundabout at all."""
    sheet = Sheet.new()
    sheet.write(9, (0, 4), turn=1)
    empty = [
        (x, y)
        for x in range(NUM_STREETS)
        for y in range(STREET_SIZES[x])
        if sheet.numbers[x][y] is None
    ]
    assert sorted(sheet.available_locations(None)) == sorted(empty)
    for edge in [(0, 0), (0, 9), (1, 0), (1, 10), (2, 0), (2, 11)]:
        assert edge in empty


def test_bis_may_duplicate_any_number_on_the_sheet_not_just_the_one_just_written():
    """``getAvailableNumbersForBis`` loops over every number 0..17, so the bis is
    a free copy of *any* house, decoupled from this turn's write."""
    sheet = Sheet.new()
    sheet.write(3, (0, 0), turn=1)
    sheet.write(12, (0, 5), turn=2)
    numbers = {n for _, _, n, _ in sheet.bis_candidates()}
    assert numbers == {3, 12}


def test_bis_cannot_cross_a_fence():
    sheet = Sheet.new()
    sheet.write(6, (0, 3), turn=1)
    assert (0, 4) in {(x, y) for x, y, _, _ in sheet.bis_candidates()}
    sheet.fences[0][3] = True
    assert (0, 4) not in {(x, y) for x, y, _, _ in sheet.bis_candidates()}


def test_a_roundabout_can_never_be_duplicated_by_a_bis():
    sheet = Sheet.new()
    sheet.build_roundabout((1, 5), turn=1)
    assert all(n != ROUNDABOUT for _, _, n, _ in sheet.bis_candidates())
    assert (1, 4) not in {(x, y) for x, y, _, _ in sheet.bis_candidates()}
    assert (1, 6) not in {(x, y) for x, y, _, _ in sheet.bis_candidates()}


def test_a_fence_may_not_split_a_bis_pair():
    sheet = Sheet.new()
    sheet.write(8, (0, 2), turn=1)
    sheet.write(8, (0, 3), turn=1, is_bis=True)
    assert (0, 2) not in sheet.surveyor_zones()


def test_estate_of_seven_or_more_scores_nothing():
    """``getAssocSizeNumber`` only counts ``$size < 7``.  A run of seven fenced
    houses is worth zero, which makes an unsplittable bis run a real hazard."""
    sheet = Sheet.new()
    for y in range(7):
        sheet.write(y + 1, (2, y), turn=1)
    sheet.fences[2][6] = True
    assert (2, 0, 7) in sheet.estates()
    assert sheet.estate_size_counts() == [0] * 6
    assert sheet.estate_score() == 0


def test_a_roundabout_is_a_free_double_fence():
    """``buildRoundabout`` scribbles ``estate-fence`` at ``pos-1`` *and* ``pos``.

    So a roundabout is never merely a hole: it auto-isolates, which closes the
    estates on both sides of it.  That makes it worth two surveyor actions on top
    of the capacity repair, and is why the ``$full = false`` branch of
    ``getEstates`` is effectively defensive — a roundabout is always alone in its
    own segment.
    """
    sheet = Sheet.new()
    sheet.write(2, (0, 0), turn=1)
    sheet.build_roundabout((0, 1), turn=1)
    sheet.write(5, (0, 2), turn=1)
    sheet.fences[0][2] = True

    assert sheet.fences[0][0] and sheet.fences[0][1], "roundabout must fence both sides"
    estates = sheet.estates()
    assert (0, 0, 1) in estates, "the house left of the roundabout is closed off"
    assert (0, 2, 1) in estates, "and so is the house right of it"
    assert not [e for e in estates if e[1] == 1], "the roundabout itself is no estate"


def test_a_roundabout_on_a_street_edge_drops_the_off_board_fence():
    """``$pos[1] - 1`` is -1 in box 0; the engine must skip it, not wrap."""
    left = Sheet.new()
    left.build_roundabout((0, 0), turn=1)
    assert left.fences[0][0]
    assert not any(left.fences[1]) and not any(left.fences[2])

    right = Sheet.new()
    right.build_roundabout((2, STREET_SIZES[2] - 1), turn=1)
    assert right.fences[2][FENCE_SIZES[2] - 1]
    assert sum(right.fences[2]) == 1, "only the left-hand fence exists at the edge"


# ──────────────────────────────────────────────────────────────────────────
# City Plan validation, transliterated from modules/php/Plans/*.php
#
# The subtlety that makes these agree is ``Zone::getAvailableZones``: for a 2-D
# zone it ``break``s after the first free box in each row, so it returns at most
# ONE entry per street.  ``count(Park::getAvailableZones(...)) <= 1`` therefore
# means "at most one street incomplete", not "at most one park box unbuilt".
# ──────────────────────────────────────────────────────────────────────────
from collections import Counter

from games.welcome_to import plans as P
from games.welcome_to.constants import EXTREMITY_POSITIONS, PARK_BOXES


def _php_park_zones(sheet: Sheet) -> list[list[int]]:
    """``Park::getAvailableZones($player, false)`` -- first free park per street."""
    out = []
    for i in range(NUM_STREETS):
        if sheet.parks[i] < PARK_BOXES[i]:
            out.append([i, sheet.parks[i]])
    return out


def _php_pool_completed(sheet: Sheet) -> list[bool]:
    """``Pool::getCompleted`` -- all three pools of a street built."""
    return [sheet.pools[i] == 3 for i in range(NUM_STREETS)]


def _php_free_estates(sheet: Sheet) -> list[tuple[int, int, int]]:
    """``EstatePlan::getAvailableEstates`` -- estates no plan has consumed."""
    return [
        (x, y, size)
        for x, y, size in php_get_estates(sheet)
        if not any(sheet.top_fences[x][y + i] for i in range(size))
    ]


def php_can_be_scored(plan, sheet: Sheet) -> bool:
    kind = plan.kind

    if kind is P.PlanKind.ESTATE:
        # subtract_array($this->conditions, $sizes) must come out empty
        sizes = Counter(size for _, _, size in _php_free_estates(sheet))
        need = Counter(plan.required_sizes)
        return all(sizes[s] >= n for s, n in need.items())

    if kind is P.PlanKind.FULL_STREET:
        x = plan.params[0]
        not_used = not any(sheet.top_fences[x])
        no_free_box = all(loc[0] != x for loc in php_available_locations(sheet, None))
        return not_used and no_free_box

    if kind is P.PlanKind.FIVE_BIS:
        bis = [0, 0, 0]
        for x in range(NUM_STREETS):
            for y in range(STREET_SIZES[x]):
                if sheet.is_bis[x][y]:
                    bis[x] += 1
        return bis[0] > 4 or bis[1] > 4 or bis[2] > 4

    if kind is P.PlanKind.SEVEN_TEMP:
        # dim-1 zone: no free box at all, or the first free temp index > 6
        return sheet.temps >= 11 or sheet.temps > 6

    if kind is P.PlanKind.EXTREMITIES:
        built = True
        for x, y in EXTREMITY_POSITIONS:
            built = (
                built and sheet.numbers[x][y] is not None and not sheet.top_fences[x][y]
            )
        return built

    if kind is P.PlanKind.COMPLETE_STREET:
        streets = [True, True, True]
        for zone in _php_park_zones(sheet):
            streets[zone[0]] = False
        pools = _php_pool_completed(sheet)
        for i in range(NUM_STREETS):
            streets[i] = streets[i] and pools[i]
        roundabouts = [False, False, False]
        for x in range(NUM_STREETS):
            for y in range(STREET_SIZES[x]):
                if sheet.numbers[x][y] == ROUNDABOUT:
                    roundabouts[x] = True
        for i in range(NUM_STREETS):
            streets[i] = streets[i] and roundabouts[i]
        return streets[0] or streets[1] or streets[2]

    if kind is P.PlanKind.DECORATIVE:
        what = plan.params[0]
        if what == "park":
            return len(_php_park_zones(sheet)) <= 1
        if what == "pool":
            return sum(1 for c in _php_pool_completed(sheet) if c) >= 2
        if what == "pool&park":
            x = plan.params[1]
            parks = [True, True, True]
            for zone in _php_park_zones(sheet):
                parks[zone[0]] = False
            return parks[x] and _php_pool_completed(sheet)[x]

    raise NotImplementedError(kind)


SUPPORTED_PLANS = [p for p in P.PLANS if p.kind is not P.PlanKind.UNSUPPORTED]


def test_every_supported_plan_kind_is_covered():
    kinds = {p.kind for p in SUPPORTED_PLANS}
    assert kinds == {
        P.PlanKind.ESTATE,
        P.PlanKind.FULL_STREET,
        P.PlanKind.FIVE_BIS,
        P.PlanKind.SEVEN_TEMP,
        P.PlanKind.EXTREMITIES,
        P.PlanKind.DECORATIVE,
        P.PlanKind.COMPLETE_STREET,
    }


def test_can_be_scored_matches_php_for_every_plan():
    sample = CORPUS[::7]
    assert len(sample) > 500
    seen_true = Counter()
    for sheet in sample:
        for plan in SUPPORTED_PLANS:
            ours = P.can_be_scored(plan, sheet)
            theirs = php_can_be_scored(plan, sheet)
            assert ours == theirs, f"plan {plan.id} ({plan.kind})\n{sheet.pretty()}"
            if ours:
                seen_true[plan.kind] += 1
    # A vacuous pass would be worthless: the corpus must actually satisfy plans.
    assert seen_true[P.PlanKind.ESTATE] > 0, "no estate plan ever satisfied"


def test_progress_reaches_one_exactly_when_the_plan_is_scorable():
    for sheet in CORPUS[::11]:
        for plan in SUPPORTED_PLANS:
            fraction, left = P.progress(plan, sheet)
            assert (fraction >= 1.0) == P.can_be_scored(plan, sheet), plan.id
            assert (left == 0) == P.can_be_scored(plan, sheet), plan.id


# ──────────────────────────────────────────────────────────────────────────
# Remaining paths: temp clamp, permit refusal, reshuffle counterfactual
# ──────────────────────────────────────────────────────────────────────────
def test_temp_can_only_push_a_number_into_0_to_17():
    from games.welcome_to.constants import MAX_NUMBER, MIN_NUMBER

    seen = set()
    for g in range(40):
        rng = random.Random(31337 + g)
        state = GameState.new(
            seed=808 + g, config=GameConfig(players=2, advanced=True, solo_rules=False)
        )
        while not state.is_terminal:
            state = state.step(rng.choice(state.legal_actions()))
        for s in state.sheets:
            for x in range(NUM_STREETS):
                for y in range(STREET_SIZES[x]):
                    n = s.numbers[x][y]
                    if n is not None and n != ROUNDABOUT:
                        seen.add(n)
                        assert MIN_NUMBER <= n <= MAX_NUMBER
    assert 0 in seen or 17 in seen, "temp never reached a bound; test is vacuous"


def test_a_player_with_no_legal_write_takes_a_permit_refusal():
    """The refusal track is the end condition, so it must never be skippable."""
    from games.welcome_to.constants import PERMIT_BOXES

    ended_on_permits = 0
    for g in range(30):
        rng = random.Random(55 + g)
        state = GameState.new(
            seed=222 + g, config=GameConfig(players=2, advanced=True, solo_rules=False)
        )
        while not state.is_terminal:
            state = state.step(rng.choice(state.legal_actions()))
        assert all(s.permits <= PERMIT_BOXES for s in state.sheets)
        if any(s.permits == PERMIT_BOXES for s in state.sheets):
            ended_on_permits += 1
    assert ended_on_permits > 0


def test_reshuffle_counterfactual_is_deck_plus_discard():
    from games.welcome_to import deck_knowledge as dk

    rng = random.Random(9)
    state = GameState.new(
        seed=17, config=GameConfig(players=2, advanced=True, solo_rules=False)
    )
    checked = 0
    while not state.is_terminal and checked < 40:
        state = state.step(rng.choice(state.legal_actions()))
        deck = dk.deck_composition(state, 0)
        after = dk.after_reshuffle_composition(state, 0)
        discard = dk.discard_composition(state, 0)
        assert (after == deck + discard).all()
        assert after.sum() >= deck.sum(), "a reshuffle can never shrink the deck"
        checked += 1
    assert checked == 40


# ──────────────────────────────────────────────────────────────────────────
# Constructed boundary cases for the plan kinds random play never reaches.
#
# Measured over the 13,382-sheet corpus, random play satisfies ESTATE (777),
# EXTREMITIES (176) and FULL_STREET (63) but NEVER satisfies FIVE_BIS,
# SEVEN_TEMP, DECORATIVE or COMPLETE_STREET.  The differential test above
# therefore passes vacuously for those four, so each one is pinned here on a
# hand-built sheet that satisfies it and one that is exactly one mark short.
# Both sides of the differential are asserted, not just ours.
# ──────────────────────────────────────────────────────────────────────────
def _plan_of_kind(kind, predicate=None):
    for p in SUPPORTED_PLANS:
        if p.kind is kind and (predicate is None or predicate(p)):
            return p
    raise AssertionError(f"no plan of kind {kind}")


def _both(plan, sheet) -> bool:
    """Assert ours and the PHP transliteration agree, and return the verdict."""
    ours = P.can_be_scored(plan, sheet)
    theirs = php_can_be_scored(plan, sheet)
    assert ours == theirs, f"plan {plan.id} ({plan.kind}) disagrees\n{sheet.pretty()}"
    return ours


def test_five_bis_needs_five_on_one_street_not_five_in_total():
    plan = _plan_of_kind(P.PlanKind.FIVE_BIS)

    spread = Sheet.new()
    for y in range(3):
        spread.write(4, (0, y), turn=1, is_bis=True)
    for y in range(2):
        spread.write(4, (1, y), turn=1, is_bis=True)
    assert spread.bis_count_per_street() == [3, 2, 0]
    assert not _both(plan, spread), "five bis spread over two streets must not score"

    four = Sheet.new()
    for y in range(4):
        four.write(4, (2, y), turn=1, is_bis=True)
    assert not _both(plan, four)

    five = Sheet.new()
    for y in range(5):
        five.write(4, (2, y), turn=1, is_bis=True)
    assert _both(plan, five)


def test_seven_temp_boundary_is_seven_not_six():
    plan = _plan_of_kind(P.PlanKind.SEVEN_TEMP)
    for temps, expected in [(0, False), (6, False), (7, True), (11, True)]:
        sheet = Sheet.new()
        sheet.temps = temps
        assert _both(plan, sheet) is expected, f"temps={temps}"


def _complete_parks(sheet, x):
    sheet.parks[x] = PARK_BOXES[x]


def _complete_pools(sheet, x):
    sheet.pools[x] = 3


def test_decorative_park_needs_two_complete_streets():
    plan = _plan_of_kind(P.PlanKind.DECORATIVE, lambda p: p.params[0] == "park")

    one = Sheet.new()
    _complete_parks(one, 0)
    assert not _both(plan, one)

    # one street complete and another one park short is still not two
    almost = Sheet.new()
    _complete_parks(almost, 0)
    almost.parks[1] = PARK_BOXES[1] - 1
    assert not _both(plan, almost)

    two = Sheet.new()
    _complete_parks(two, 0)
    _complete_parks(two, 1)
    assert _both(plan, two)


def test_decorative_pool_needs_two_complete_streets():
    plan = _plan_of_kind(P.PlanKind.DECORATIVE, lambda p: p.params[0] == "pool")

    one = Sheet.new()
    _complete_pools(one, 2)
    assert not _both(plan, one)

    two = Sheet.new()
    _complete_pools(two, 1)
    _complete_pools(two, 2)
    assert _both(plan, two)


def test_decorative_pool_and_park_is_street_specific():
    plan = _plan_of_kind(P.PlanKind.DECORATIVE, lambda p: p.params[0] == "pool&park")
    x = plan.params[1]
    other = (x + 1) % NUM_STREETS

    wrong_street = Sheet.new()
    _complete_parks(wrong_street, other)
    _complete_pools(wrong_street, other)
    assert not _both(plan, wrong_street), "the plan names one specific street"

    parks_only = Sheet.new()
    _complete_parks(parks_only, x)
    assert not _both(plan, parks_only)

    both = Sheet.new()
    _complete_parks(both, x)
    _complete_pools(both, x)
    assert _both(plan, both)


def test_complete_street_needs_parks_pools_and_a_roundabout_on_the_same_street():
    plan = _plan_of_kind(P.PlanKind.COMPLETE_STREET)

    # every ingredient present, but scattered across three different streets
    scattered = Sheet.new()
    _complete_parks(scattered, 0)
    _complete_pools(scattered, 1)
    scattered.build_roundabout((2, 5), turn=1)
    assert not _both(plan, scattered)

    # right street, missing only the roundabout
    no_roundabout = Sheet.new()
    _complete_parks(no_roundabout, 1)
    _complete_pools(no_roundabout, 1)
    assert not _both(plan, no_roundabout)

    done = Sheet.new()
    _complete_parks(done, 1)
    _complete_pools(done, 1)
    done.build_roundabout((1, 5), turn=1)
    assert _both(plan, done)


def test_extremities_is_blocked_by_a_house_already_spent_on_another_plan():
    plan = _plan_of_kind(P.PlanKind.EXTREMITIES)
    sheet = Sheet.new()
    for i, (x, y) in enumerate(EXTREMITY_POSITIONS):
        sheet.write(1 + i % 3, (x, y), turn=1)
    assert _both(plan, sheet)

    spent = sheet.copy()
    x, y = EXTREMITY_POSITIONS[0]
    spent.top_fences[x][y] = True
    assert not _both(plan, spent), "a house consumed by another plan cannot be reused"


# ──────────────────────────────────────────────────────────────────────────
# A roundabout is a *built house* for plan purposes
#
# ``writeNumber(ROUNDABOUT, $pos)`` goes through ``Houses::add`` like any other
# house, so every plan that asks "is this box built?" sees a roundabout as built.
# ``ExtremitiesPlan`` tests ``!is_null($streets[$x][$y])`` and ``FullStreetPlan``
# asks for no free box in the street (its own description says "roundabout also
# works").  Combined with the auto-fencing above, a roundabout dropped on a
# street end can *simultaneously* complete Extremities and close an estate.
# ──────────────────────────────────────────────────────────────────────────
def test_a_roundabout_counts_as_a_built_house_for_the_extremities_plan():
    plan = _plan_of_kind(P.PlanKind.EXTREMITIES)

    sheet = Sheet.new()
    # Build five of the six extremities with ordinary houses...
    for i, (x, y) in enumerate(EXTREMITY_POSITIONS[:-1]):
        sheet.write(1 + i % 3, (x, y), turn=1)
    assert not _both(plan, sheet), "five of six extremities is not enough"

    # ...and the last one with a roundabout.
    last_x, last_y = EXTREMITY_POSITIONS[-1]
    sheet.build_roundabout((last_x, last_y), turn=2)
    assert sheet.numbers[last_x][last_y] == ROUNDABOUT
    assert _both(plan, sheet), "a roundabout must count towards Extremities"


def test_a_roundabout_counts_as_a_built_house_for_the_full_street_plan():
    plan = _plan_of_kind(P.PlanKind.FULL_STREET)
    x = plan.params[0]

    sheet = Sheet.new()
    for y in range(STREET_SIZES[x] - 1):
        sheet.write(y + 1, (x, y), turn=1)
    assert not _both(plan, sheet)

    sheet.build_roundabout((x, STREET_SIZES[x] - 1), turn=2)
    assert _both(plan, sheet), "FullStreet's own description says roundabouts work"


def test_a_roundabout_on_a_street_end_both_completes_extremities_and_fences():
    """The two effects compose, which is what makes an end-of-street roundabout
    a genuinely strong move rather than just a capacity repair."""
    plan = _plan_of_kind(P.PlanKind.EXTREMITIES)
    sheet = Sheet.new()
    for i, (x, y) in enumerate(EXTREMITY_POSITIONS[:-1]):
        sheet.write(1 + i % 3, (x, y), turn=1)
    sheet.build_roundabout(EXTREMITY_POSITIONS[-1], turn=2)

    assert _both(plan, sheet)
    x, y = EXTREMITY_POSITIONS[-1]
    assert sheet.fences[x][y - 1], "and it fences the box to its left for free"


# ──────────────────────────────────────────────────────────────────────────
# Temp agency: pure rank, and no temps means no points
#
# ``Temp::computeCounters`` builds its counter map from *scribbles*, so a player
# who never hired a temp has NO ENTRY and never appears in the ordering at all.
# They score 0 -- they do not inherit second place by default.  Ties share the
# better value, because ``array_unique`` collapses equal counts into one rank.
# ──────────────────────────────────────────────────────────────────────────
def _state_with_temps(counts):
    state = GameState.new(
        seed=1, config=GameConfig(players=len(counts), advanced=True, solo_rules=False)
    )
    for p, c in enumerate(counts):
        state.sheets[p].temps = c
    return state


def test_a_player_who_never_hires_a_temp_scores_zero_not_second_place():
    assert _state_with_temps([5, 0]).temp_scores() == [7, 0]


def test_two_player_temp_is_seven_and_four_by_rank():
    assert _state_with_temps([5, 3]).temp_scores() == [7, 4]
    assert _state_with_temps([1, 11]).temp_scores() == [4, 7]


def test_tied_temp_counts_share_the_better_value():
    assert _state_with_temps([4, 4]).temp_scores() == [7, 7]
    assert _state_with_temps([6, 4, 4]).temp_scores() == [7, 4, 4]


def test_rank_is_computed_only_among_players_who_hired_at_least_one():
    # C has none, so B is second, not third -- the zero player is not a rank.
    assert _state_with_temps([5, 3, 0]).temp_scores() == [7, 4, 0]
    assert _state_with_temps([5, 3, 1, 0]).temp_scores() == [7, 4, 1, 0]
    assert _state_with_temps([0, 0, 0, 0]).temp_scores() == [0, 0, 0, 0]


def test_only_the_top_three_temp_ranks_pay_anything():
    assert _state_with_temps([9, 7, 5, 3]).temp_scores() == [7, 4, 1, 0]


# ──────────────────────────────────────────────────────────────────────────
# Houses spent on a City Plan cannot be reused by another City Plan
#
# The mechanism is the ``top-fence`` scribble.  Exactly three plan kinds override
# ``AbstractPlan::validate`` to lay them down -- EstatePlan (every box of each
# estate it used), FullStreetPlan (every box of the street) and ExtremitiesPlan
# (the six end boxes).  The other four score off *tracks* -- bis count, temp
# count, park/pool completion -- and consume no houses at all, so their plans
# stack freely with everything else.
#
# The asymmetry is worth stating plainly because it drives plan selection: a
# FullStreet or Extremities plan permanently spends board, an Estate plan spends
# exactly the estates it names, and the four track plans are free.
# ──────────────────────────────────────────────────────────────────────────
CONSUMING_KINDS = {
    P.PlanKind.ESTATE,
    P.PlanKind.FULL_STREET,
    P.PlanKind.EXTREMITIES,
}
TRACK_KINDS = {
    P.PlanKind.FIVE_BIS,
    P.PlanKind.SEVEN_TEMP,
    P.PlanKind.DECORATIVE,
    P.PlanKind.COMPLETE_STREET,
}


def test_exactly_three_plan_kinds_consume_houses():
    assert CONSUMING_KINDS | TRACK_KINDS == {p.kind for p in SUPPORTED_PLANS}

    sheet = Sheet.new()
    for x in range(NUM_STREETS):
        for y in range(STREET_SIZES[x]):
            sheet.write(y + 1, (x, y), turn=1)

    for plan in SUPPORTED_PLANS:
        # Estate plans consume only the estates handed over, so give them one.
        chosen = ((0, 0, 1),) if plan.kind is P.PlanKind.ESTATE else ()
        cells = P.validation_cells(plan, sheet, chosen)
        if plan.kind in TRACK_KINDS:
            assert cells == [], f"plan {plan.id} ({plan.kind}) must consume nothing"
        else:
            assert cells, f"plan {plan.id} ({plan.kind}) must consume houses"


def test_a_street_spent_on_full_street_cannot_supply_estates_to_another_plan():
    """The user's example: a row used for the all-houses plan is spent."""
    full_street = _plan_of_kind(P.PlanKind.FULL_STREET)
    x = full_street.params[0]

    sheet = Sheet.new()
    for y in range(STREET_SIZES[x]):
        sheet.write(y + 1, (x, y), turn=1)
    sheet.fences[x][2] = True  # splits the street into a 3 and a remainder

    sizes_before = sorted(size for sx, _, size in sheet.estates() if sx == x)
    assert sizes_before, "the street should hold estates before the plan is spent"
    assert _both(full_street, sheet)

    # Score the FullStreet plan: every box of the street gets a top fence.
    sheet.mark_top_fences(P.validation_cells(full_street, sheet))

    assert not any(sx == x for sx, _, _ in sheet.free_estates()), (
        "no estate in a spent street may be offered to another plan"
    )
    # The estates still physically exist -- they are just no longer available.
    assert sorted(size for sx, _, size in sheet.estates() if sx == x) == sizes_before
    # And the plan can no longer be re-scored off the same street.
    assert not _both(full_street, sheet)


def test_an_estate_plan_spends_only_the_estates_it_actually_used():
    # Two adjacent size-2 estates: boxes 0-1 and boxes 2-3, split by a fence.
    # A fence does not reset the ascending rule (only a roundabout does), so the
    # numbers still have to climb across it.
    sheet = Sheet.new()
    for y in range(4):
        sheet.write(y + 1, (0, y), turn=1)
    sheet.fences[0][1] = True
    sheet.fences[0][3] = True

    estates = [e for e in sheet.free_estates() if e[0] == 0]
    assert len(estates) >= 2, estates
    spent, kept = estates[0], estates[1]

    plan = _plan_of_kind(P.PlanKind.ESTATE)
    sheet.mark_top_fences(P.validation_cells(plan, sheet, (spent,)))

    free = sheet.free_estates()
    assert spent not in free, "the estate handed over is spent"
    assert kept in free, "an estate not handed over stays available"


def test_track_plans_are_unaffected_by_houses_spent_elsewhere():
    """7 temps, 5 bis and the park/pool plans keep scoring after the board is
    consumed, because none of them looks at top fences."""
    sheet = Sheet.new()
    for y in range(5):
        sheet.write(4, (2, y), turn=1, is_bis=True)
    sheet.temps = 7
    _complete_parks(sheet, 0)
    _complete_parks(sheet, 1)

    five_bis = _plan_of_kind(P.PlanKind.FIVE_BIS)
    seven_temp = _plan_of_kind(P.PlanKind.SEVEN_TEMP)
    park = _plan_of_kind(P.PlanKind.DECORATIVE, lambda p: p.params[0] == "park")
    for plan in (five_bis, seven_temp, park):
        assert _both(plan, sheet)

    # Spend every house on the sheet, as three other plans might.
    sheet.mark_top_fences(
        [(x, y) for x in range(NUM_STREETS) for y in range(STREET_SIZES[x])]
    )
    for plan in (five_bis, seven_temp, park):
        assert _both(plan, sheet), f"{plan.kind} must not care about spent houses"


def test_extremities_and_full_street_compete_for_the_same_boxes():
    """Both consume board, and they overlap at the ends of every street -- so
    ordering matters and the model has to see it."""
    extremities = _plan_of_kind(P.PlanKind.EXTREMITIES)
    full_street = _plan_of_kind(P.PlanKind.FULL_STREET)
    x = full_street.params[0]

    sheet = Sheet.new()
    for sx in range(NUM_STREETS):
        for y in range(STREET_SIZES[sx]):
            sheet.write(y + 1, (sx, y), turn=1)
    assert _both(extremities, sheet)
    assert _both(full_street, sheet)

    sheet.mark_top_fences(P.validation_cells(full_street, sheet))
    assert not _both(extremities, sheet), (
        "FullStreet consumed both ends of its street, blocking Extremities"
    )
