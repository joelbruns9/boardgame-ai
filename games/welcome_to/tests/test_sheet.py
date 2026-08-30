"""Sheet-local rules: where numbers may go, what an estate is, what it scores."""
from __future__ import annotations

import pytest

from games.welcome_to.constants import (
    DECK_COUNTS,
    CARD_TABLE,
    NUM_BASE_CARDS,
    POOL_POSITIONS,
    ROUNDABOUT,
    SOLO_CARD_ID,
    STREET_SIZES,
    box_coords,
    box_index,
    fence_coords,
    fence_index,
)
from games.welcome_to.sheet import Sheet


# ──────────────────────────────────────────────────────────────────────────
# Static data
# ──────────────────────────────────────────────────────────────────────────
def test_deck_has_81_construction_cards():
    assert sum(sum(counts) for counts in DECK_COUNTS.values()) == 81
    assert NUM_BASE_CARDS == 81
    assert len(CARD_TABLE) == 82  # plus the solo card
    assert CARD_TABLE[SOLO_CARD_ID][0] is None


def test_card_numbers_run_one_to_fifteen():
    numbers = {n for n, _ in CARD_TABLE[:NUM_BASE_CARDS]}
    assert numbers == set(range(1, 16))


def test_box_and_fence_index_round_trip():
    for x, size in enumerate(STREET_SIZES):
        for y in range(size):
            assert box_coords(box_index(x, y)) == (x, y)
    for x, size in enumerate((9, 10, 11)):
        for j in range(size):
            assert fence_coords(fence_index(x, j)) == (x, j)


def test_pool_positions_are_three_per_street():
    per_street = [sum(1 for x, _ in POOL_POSITIONS if x == s) for s in range(3)]
    assert per_street == [3, 3, 3]
    for x, y in POOL_POSITIONS:
        assert 0 <= y < STREET_SIZES[x]


# ──────────────────────────────────────────────────────────────────────────
# Writing numbers
# ──────────────────────────────────────────────────────────────────────────
def test_empty_sheet_accepts_a_number_anywhere():
    sheet = Sheet.new()
    assert len(sheet.available_locations(5)) == 33
    assert len(sheet.available_locations(None)) == 33


def test_numbers_must_ascend_within_a_street():
    sheet = Sheet.new()
    sheet.write(5, (0, 3), turn=1)

    # the same number can no longer go anywhere in that street
    street0 = [pos for pos in sheet.available_locations(5) if pos[0] == 0]
    assert street0 == []

    # a bigger number only fits to the right of it
    bigger = [y for x, y in sheet.available_locations(6) if x == 0]
    assert bigger == [4, 5, 6, 7, 8, 9]

    # a smaller number only to the left
    smaller = [y for x, y in sheet.available_locations(4) if x == 0]
    assert smaller == [0, 1, 2]

    # other streets are untouched
    assert len(sheet.available_locations(5)) == 11 + 12


def test_roundabout_breaks_the_ascending_chain_in_both_directions():
    sheet = Sheet.new()
    sheet.numbers[0][3] = ROUNDABOUT
    open_boxes = [y for x, y in sheet.available_locations(5) if x == 0]
    assert open_boxes == [0, 1, 2, 4, 5, 6, 7, 8, 9]


def test_has_free_box_tracks_a_full_sheet():
    sheet = Sheet.new()
    assert sheet.has_free_box()
    for x, size in enumerate(STREET_SIZES):
        for y in range(size):
            sheet.numbers[x][y] = y
    assert not sheet.has_free_box()
    assert sheet.available_locations(None) == []


# ──────────────────────────────────────────────────────────────────────────
# Bis
# ──────────────────────────────────────────────────────────────────────────
def test_bis_needs_a_neighbour():
    sheet = Sheet.new()
    assert sheet.bis_candidates() == []

    sheet.write(5, (0, 3), turn=1)
    assert set(sheet.bis_candidates()) == {(0, 2, 5, 1), (0, 4, 5, 0)}


def test_bis_cannot_cross_a_fence():
    sheet = Sheet.new()
    sheet.write(5, (0, 3), turn=1)
    sheet.fences[0][3] = True  # between boxes 3 and 4
    assert set(sheet.bis_candidates()) == {(0, 2, 5, 1)}
    assert sheet.bis_number_at(0, 4, 0) is None
    assert sheet.bis_number_at(0, 2, 1) == 5


def test_bis_cannot_duplicate_a_roundabout():
    sheet = Sheet.new()
    sheet.numbers[0][3] = ROUNDABOUT
    assert sheet.bis_candidates() == []


def test_bis_must_be_adjacent_not_merely_nearby():
    sheet = Sheet.new()
    sheet.write(5, (0, 3), turn=1)
    # box 5 is two away from the 5 and has no neighbour of its own
    assert sheet.bis_number_at(0, 5, 0) is None


# ──────────────────────────────────────────────────────────────────────────
# Estates
# ──────────────────────────────────────────────────────────────────────────
def _fill_street(sheet: Sheet, x: int) -> None:
    for y in range(STREET_SIZES[x]):
        sheet.numbers[x][y] = y


def test_a_full_street_with_no_fences_is_one_estate():
    sheet = Sheet.new()
    _fill_street(sheet, 0)
    assert sheet.estates() == [(0, 0, 10)]
    # sizes above six score nothing
    assert sheet.estate_size_counts() == [0] * 6


def test_fences_split_a_street_into_estates():
    sheet = Sheet.new()
    _fill_street(sheet, 0)
    sheet.fences[0][2] = True
    assert sheet.estates() == [(0, 0, 3), (0, 3, 7)]
    assert sheet.estate_size_counts()[2] == 1


def test_a_hole_or_a_roundabout_kills_an_estate():
    sheet = Sheet.new()
    _fill_street(sheet, 0)
    sheet.fences[0][2] = True
    sheet.numbers[0][1] = None
    assert sheet.estates() == [(0, 3, 7)]

    sheet.numbers[0][1] = 1
    sheet.numbers[0][0] = ROUNDABOUT
    assert sheet.estates() == [(0, 3, 7)]


def test_free_estates_exclude_ones_a_plan_consumed():
    sheet = Sheet.new()
    _fill_street(sheet, 0)
    sheet.fences[0][2] = True
    sheet.top_fences[0][0] = True
    assert sheet.free_estates() == [(0, 3, 7)]


# ──────────────────────────────────────────────────────────────────────────
# Surveyor
# ──────────────────────────────────────────────────────────────────────────
def test_every_fence_slot_is_open_on_an_empty_sheet():
    assert len(Sheet.new().surveyor_zones()) == 30


def test_a_fence_cannot_split_a_bis_pair():
    sheet = Sheet.new()
    sheet.write(5, (0, 3), turn=1)
    sheet.write(5, (0, 4), turn=1, is_bis=True)
    assert (0, 3) not in sheet.surveyor_zones()
    assert (0, 2) in sheet.surveyor_zones()


def test_a_fence_cannot_split_two_houses_used_by_one_plan():
    sheet = Sheet.new()
    sheet.write(4, (0, 3), turn=1)
    sheet.write(5, (0, 4), turn=1)
    sheet.top_fences[0][3] = True
    sheet.top_fences[0][4] = True
    assert (0, 3) not in sheet.surveyor_zones()


def test_an_existing_fence_is_not_offered_again():
    sheet = Sheet.new()
    sheet.fences[0][0] = True
    assert (0, 0) not in sheet.surveyor_zones()
    assert len(sheet.surveyor_zones()) == 29


# ──────────────────────────────────────────────────────────────────────────
# Roundabouts
# ──────────────────────────────────────────────────────────────────────────
def test_building_a_roundabout_fences_both_sides():
    sheet = Sheet.new()
    sheet.build_roundabout((1, 4), turn=1)
    assert sheet.numbers[1][4] == ROUNDABOUT
    assert sheet.fences[1][3] and sheet.fences[1][4]
    assert sheet.roundabouts == 1


def test_a_roundabout_at_the_edge_only_fences_the_inside():
    sheet = Sheet.new()
    sheet.build_roundabout((0, 0), turn=1)
    assert sheet.fences[0][0]
    assert sheet.roundabouts == 1


# ──────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────
def test_park_pool_and_penalty_tracks():
    sheet = Sheet.new()
    sheet.parks = [3, 4, 5]
    assert sheet.park_score() == 10 + 14 + 18

    sheet.pools = [3, 3, 3]
    assert sheet.pool_score() == 36

    sheet.bis_marks = 9
    assert sheet.bis_penalty() == 28
    sheet.permits = 3
    assert sheet.permit_penalty() == 5
    sheet.roundabouts = 2
    assert sheet.roundabout_penalty() == 8


def test_estate_score_multiplies_count_by_row_value():
    sheet = Sheet.new()
    _fill_street(sheet, 0)
    sheet.fences[0][2] = True  # estates of size 3 and 7
    assert sheet.estate_size_counts()[2] == 1
    assert sheet.estate_score() == 3  # row 3, no boxes crossed

    sheet.estate_marks[2] = 2
    assert sheet.estate_score() == 5


def test_penalties_are_subtracted_from_the_total():
    sheet = Sheet.new()
    sheet.parks = [3, 0, 0]  # +10
    sheet.bis_marks = 2  # -3
    score = sheet.local_score()
    assert score.parks == 10
    assert score.bis == 3
    assert score.total == 7


def test_tiebreak_prefers_more_estates_then_smaller_ones():
    a = Sheet.new()
    _fill_street(a, 0)
    a.fences[0][0] = True
    a.fences[0][1] = True  # sizes 1, 1, 8

    b = Sheet.new()
    _fill_street(b, 0)
    b.fences[0][0] = True  # sizes 1, 9

    assert a.tiebreak_key() > b.tiebreak_key()


def test_copy_is_deep():
    sheet = Sheet.new()
    clone = sheet.copy()
    clone.write(7, (2, 5), turn=1)
    clone.parks[0] = 2
    assert sheet.numbers[2][5] is None
    assert sheet.parks[0] == 0


@pytest.mark.parametrize("street,expected", [(0, 9), (1, 10), (2, 11)])
def test_fence_row_lengths(street, expected):
    assert len(Sheet.new().fences[street]) == expected


# ──────────────────────────────────────────────────────────────────────────
# Bis runs
# ──────────────────────────────────────────────────────────────────────────
def test_a_bis_can_itself_be_copied_into_a_run():
    """8, 8bis, 8bis, 8bis ... a bis is an ordinary neighbour to the next one."""
    sheet = Sheet.new()
    sheet.write(8, (0, 3), turn=1)
    for y in (4, 5, 6):
        assert sheet.bis_number_at(0, y, 0) == 8, f"box {y} should copy from the left"
        sheet.write(8, (0, y), turn=1, is_bis=True)
    assert sheet.numbers[0][3:7] == [8, 8, 8, 8]
    assert sheet.bis_number_at(0, 7, 0) == 8, "the run can keep growing"


def test_a_bis_run_can_grow_leftwards_too():
    sheet = Sheet.new()
    sheet.write(8, (0, 5), turn=1)
    assert sheet.bis_number_at(0, 4, 1) == 8
    sheet.write(8, (0, 4), turn=1, is_bis=True)
    assert sheet.bis_number_at(0, 3, 1) == 8


def test_no_fence_can_split_a_bis_run_anywhere_along_it():
    sheet = Sheet.new()
    sheet.write(8, (0, 3), turn=1)
    for y in (4, 5):
        sheet.write(8, (0, y), turn=1, is_bis=True)

    zones = sheet.surveyor_zones()
    for j in (3, 4):  # between 3|4 and 4|5, both equal numbers
        assert (0, j) not in zones, f"fence slot {j} splits the bis run"
    assert (0, 2) in zones, "the fence before the run is still legal"
    assert (0, 5) in zones, "and the one after it, since box 6 is empty"


def test_a_fence_beside_a_bis_run_still_blocks_further_bis():
    sheet = Sheet.new()
    sheet.write(8, (0, 3), turn=1)
    sheet.write(8, (0, 4), turn=1, is_bis=True)
    sheet.fences[0][4] = True  # between 4 and 5
    assert sheet.bis_number_at(0, 5, 0) is None


def test_a_bis_run_counts_once_per_house_for_the_five_bis_plan():
    sheet = Sheet.new()
    sheet.write(8, (0, 3), turn=1)
    for y in (4, 5, 6, 7):
        sheet.write(8, (0, y), turn=1, is_bis=True)
    assert sheet.bis_count_per_street() == [4, 0, 0], "the original 8 is not a bis"


def test_a_bis_run_is_still_one_estate():
    sheet = Sheet.new()
    for y in range(10):
        sheet.numbers[0][y] = y
    sheet.numbers[0][5] = 4
    sheet.is_bis[0][5] = True
    assert sheet.estates() == [(0, 0, 10)]


# ──────────────────────────────────────────────────────────────────────────
# Placement capacity
# ──────────────────────────────────────────────────────────────────────────
def test_an_empty_sheet_can_take_every_box():
    assert Sheet.new().placement_capacity() == [10, 11, 12]


def test_a_high_number_at_the_start_of_a_street_burns_it():
    sheet = Sheet.new()
    sheet.write(15, (0, 0), turn=1)
    # only 16 and 17 can still go to the right of a 15
    assert sheet.placement_capacity()[0] == 2


def test_a_low_number_at_the_start_of_a_street_costs_nothing_extra():
    sheet = Sheet.new()
    sheet.write(1, (0, 0), turn=1)
    assert sheet.placement_capacity()[0] == 9


def test_consecutive_numbers_leave_a_dead_gap():
    sheet = Sheet.new()
    sheet.write(5, (0, 3), turn=1)
    sheet.write(6, (0, 5), turn=1)
    assert sheet.box_spans()[0][4] == 0, "nothing fits strictly between 5 and 6"
    # boxes 0..2 take 0..4, box 4 is dead, boxes 6..9 take 7..17
    assert sheet.placement_capacity()[0] == 3 + 0 + 4


def test_a_roundabout_lifts_the_bound_on_both_sides():
    sheet = Sheet.new()
    sheet.write(15, (0, 0), turn=1)
    assert sheet.placement_capacity()[0] == 2
    sheet.numbers[0][1] = ROUNDABOUT
    # past the roundabout the street starts over
    assert sheet.placement_capacity()[0] == 8


def test_box_spans_are_zero_on_filled_boxes():
    sheet = Sheet.new()
    sheet.write(7, (2, 6), turn=1)
    assert sheet.box_spans()[2][6] == 0
    assert sheet.box_spans()[2][5] == 7, "boxes left of a 7 accept 0..6"


# ──────────────────────────────────────────────────────────────────────────
# Encoder v3 helpers: hypotheticals, and the actions that bypass numeric fit
#
# Every test below exists because a review round produced an unsound
# feasibility rule by reasoning about numbers and forgetting that a roundabout,
# a bis or a fence writes a house without one.  See ENCODER_V3_SPEC.md §6.1.
# ──────────────────────────────────────────────────────────────────────────
def _written(rows: list[list[object]]) -> Sheet:
    """A sheet with ``rows`` written directly, padded with ``None``."""
    sheet = Sheet.new()
    for x, row in enumerate(rows):
        for y, n in enumerate(row):
            if n is not None:
                sheet.numbers[x][y] = n
    return sheet


def test_free_estate_size_counts_excludes_plan_consumed_estates():
    # Street 0: 1 2 | 5  -> estates of size 2 and 1.
    sheet = _written([[1, 2, 5]])
    sheet.fences[0][1] = True
    sheet.fences[0][2] = True
    assert sheet.estate_size_counts()[:2] == [1, 1]
    assert sheet.free_estate_size_counts()[:2] == [1, 1]

    # A City Plan eats the size-1 estate.  It still SCORES -- estate_score
    # counts every estate -- but it is no longer available to another plan.
    sheet.mark_top_fences([(0, 2)])
    assert sheet.estate_size_counts()[:2] == [1, 1]
    assert sheet.free_estate_size_counts()[:2] == [0, 1]


def test_span_if_roundabout_revives_a_dead_box():
    # 5 _ 6 : the middle box is bounded by 5 and 6, so nothing fits.
    sheet = _written([[5, None, 6]])
    assert sheet.box_spans()[0][1] == 0
    # One roundabout cannot help: the box is its own whole gap, so there is no
    # other empty box in the run to place a roundabout in.
    assert sheet.span_if_roundabout()[0][1] == 0

    # 5 _ _ 6 : now a roundabout in the run removes one of the bounds.
    sheet = _written([[5, None, None, 6]])
    assert sheet.box_spans()[0][1] == 0
    revived = sheet.span_if_roundabout()[0][1]
    assert revived > 0


def test_span_if_roundabout_collapses_when_none_available():
    sheet = _written([[5, None, None, 6]])
    assert sheet.span_if_roundabout(available=False) == sheet.box_spans()

    sheet.roundabouts = 2  # both spent
    assert not sheet.can_build_roundabout()
    assert sheet.span_if_roundabout() == sheet.box_spans()


def test_capacity_if_roundabout_never_below_plain_capacity():
    sheet = _written([[9, None, None, None, 10]])
    plain = sheet.placement_capacity()
    repaired = sheet.capacity_if_roundabout()
    assert repaired[0] >= plain[0]
    assert sheet.capacity_if_roundabout(available=False) == plain


def test_bis_reaches_boxes_no_number_can():
    """The R5 counter-example: capacity 0, every span 0, every gap bis-writable."""
    # The whole street, so no slack gap is left over at the right-hand end.
    sheet = _written([[1, None, 2, None, 3, None, 4, None, 5, 6]])
    assert sheet.placement_capacity()[0] == 0
    assert all(sheet.box_spans()[0][y] == 0 for y in (1, 3, 5, 7))
    # ...yet a bis copies the neighbour, with no ascending-order check.
    assert all(sheet.bis_reachable(0, y) for y in (1, 3, 5, 7))
    assert sheet.bis_reach()[0] == 4


def test_bis_reachable_is_permissive_about_the_future():
    # An empty neighbour counts: it may be written later.
    sheet = _written([[None, None, None]])
    assert sheet.bis_reachable(0, 0)
    # A roundabout neighbour never counts, and a fence blocks the side.
    sheet = _written([[ROUNDABOUT, None, 4]])
    assert sheet.bis_reachable(0, 1)      # via the 4 on the right
    sheet.fences[0][1] = True
    assert not sheet.bis_reachable(0, 1)  # roundabout left, fenced right


def test_bis_stays_legal_after_its_track_saturates():
    """The track saturates, it does not gate -- `legal_actions` never reads it."""
    sheet = _written([[1, None, 2, None, 3, None, 4, None, 5, 6]])
    sheet.bis_marks = 9  # BIS_BOXES: every penalty box already crossed off
    assert sheet.bis_reachable(0, 1)
    assert sheet.bis_reach()[0] == 4


def test_an_extremity_is_bis_reachable_from_box_one():
    """Why the EXTREMITIES death test needs a bis term (R5 finding 1)."""
    sheet = _written([[None, 7]])
    assert sheet.box_spans()[0][0] == 7      # MIN_NUMBER is 0, so 0..6 fit
    # A 0 is reachable -- MIN_NUMBER is 0 and a temp shifts a 1 down to it --
    # and nothing at all fits to the left of one.
    sheet = _written([[None, 0]])
    assert sheet.box_spans()[0][0] == 0
    assert sheet.span_if_roundabout()[0][0] == 0   # no other empty box in the gap
    assert sheet.bis_reachable(0, 0)               # but a bis copies the 0
