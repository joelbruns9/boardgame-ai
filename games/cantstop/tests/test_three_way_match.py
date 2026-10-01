import math
from games.cantstop.three_way_match import ORDERS, summarize


def test_all_seats_and_opponent_orders():
    assert len(set(ORDERS)) == 6
    for player in range(3):
        for seat in range(3):
            assert sum(order[seat] == player for order in ORDERS) == 2
    # Winner is a seat; attribution must follow each seating permutation.
    assert [sum(order[0] == p for order in ORDERS) for p in range(3)] == [2, 2, 2]


def test_equal_players():
    r = summarize([[100, 100, 100] for _ in ORDERS])
    assert r['games'] == 1800
    assert all(p['p_bonferroni'] == 1 for p in r['pairwise'])
    assert all(not p['significant'] for p in r['pairwise'])
    assert math.isclose(r['pairwise'][0]['se']**2, (200/299/300)/6)


def test_difference_and_multiple_comparisons():
    r = summarize([[500, 300, 200] for _ in ORDERS])
    p = r['pairwise'][0]
    assert math.isclose(p['difference'], .2)
    assert p['significant']
    assert p['simultaneous_ci95'][0] > 0
    assert p['p_bonferroni'] == 3*p['p_two_sided']


def test_unbalanced_counts_rejected():
    import pytest
    with pytest.raises(ValueError):
        summarize([[10, 10, 10]]*5 + [[11, 10, 10]])
