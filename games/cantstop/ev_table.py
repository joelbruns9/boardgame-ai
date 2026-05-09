# ev_table.py
# Calculates exact EV statistics for every column combination in Can't Stop.
# Uses deterministic enumeration of all 6^4 = 1296 possible dice rolls.
# No simulation, no approximation — exact values every time.

from itertools import product


# ---- BOARD CONSTANTS ----
COLUMN_HEIGHTS = {
    2: 3,  3: 5,  4: 7,  5: 9,  6: 11,
    7: 13, 8: 11, 9: 9, 10: 7, 11: 5, 12: 3
}


def get_all_dice_outcomes():
    """
    Returns all 1296 possible outcomes of rolling 4 dice.
    Each outcome is a tuple of 4 values e.g. (1, 2, 3, 4).
    
    Think of this like listing every possible card draw
    before calculating poker odds — exhaustive, exact.
    """
    return list(product(range(1, 7), repeat=4))


def get_pair_sums(dice):
    """
    Given 4 dice, return all possible column pairs.
    The three ways to split 4 dice into 2 pairs.
    e.g. (1,2,3,4) → {(3,7), (4,6), (5,5)}
    """
    d1, d2, d3, d4 = dice
    return {
        (d1+d2, d3+d4),
        (d1+d3, d2+d4),
        (d1+d4, d2+d3),
    }


def calc_prob_advance(columns):
    """
    Exact probability of rolling at least one of the target columns.
    
    columns: a set or list of column numbers e.g. {3, 4, 5}
    
    Method: count every dice outcome where at least one pair sum
    matches a target column. Divide by total outcomes (1296).
    """
    columns = set(columns)
    outcomes = get_all_dice_outcomes()
    hits = 0

    for dice in outcomes:
        pairs = get_pair_sums(dice)
        # Flatten all sums from all pairs into one set
        all_sums = {s for pair in pairs for s in pair}
        if all_sums & columns:  # intersection — any match?
            hits += 1

    return hits / len(outcomes)


def calc_avg_progress(columns):
    """
    Average WEIGHTED progress per roll across target columns.
    
    Each step is normalized by column height so progress is
    measured in 'fraction of column completed' not raw steps.
    
    e.g. 1 step on column 2 (height 3) = 1/3 = 0.333
         1 step on column 7 (height 13) = 1/13 = 0.077
    
    This makes progress comparable across different column combinations.
    """
    columns = set(columns)
    outcomes = get_all_dice_outcomes()
    total_weighted_progress = 0

    for dice in outcomes:
        pairs = get_pair_sums(dice)
        best_weighted = 0

        for pair in pairs:
            # Sum the fractional advancement for each column hit
            weighted = sum(
                1 / COLUMN_HEIGHTS[col]
                for col in pair
                if col in columns
            )
            best_weighted = max(best_weighted, weighted)

        total_weighted_progress += best_weighted

    return total_weighted_progress / len(outcomes)


def calc_break_even(columns):
    """
    Break even point in weighted progress units.
    
    If current_weighted_progress >= break_even, stopping is correct.
    
    Both numerator and denominator are in the same unit:
    'fraction of column completed per roll'
    """
    prob_adv = calc_prob_advance(columns)
    avg_weighted = calc_avg_progress(columns)

    if prob_adv >= 1.0:
        return float('inf')

    return (prob_adv * avg_weighted) / (1 - prob_adv)


def build_ev_table():
    """
    Build the complete EV lookup table for all valid 3-column combinations.
    
    Returns a dictionary keyed by sorted column tuple:
    {
        (2, 4, 10): {
            "prob_adv":   0.7393,
            "avg_prog":   0.1438,
            "break_even": 0.5514,
        },
        ...
    }
    """
    from itertools import combinations

    columns = list(COLUMN_HEIGHTS.keys())  # [2, 3, 4, ... 12]
    ev_table = {}

    for combo in combinations(columns, 3):
        combo = tuple(sorted(combo))
        ev_table[combo] = {
            "prob_adv":   round(calc_prob_advance(combo),   6),
            "avg_prog":   round(calc_avg_progress(combo),   6),
            "break_even": round(calc_break_even(combo),     6),
        }

    return ev_table


# ---- TEST IT ----
if __name__ == "__main__":
    print("Building exact EV table for all column combinations...")
    print("Enumerating all 1,296 dice outcomes per combination\n")

    table = build_ev_table()

    print(f"Calculated {len(table)} combinations\n")

    # Compare a few against your Excel values
    test_cases = [
        ((3, 4, 5),  0.66786),   # your Excel value
        ((6, 7, 8),  0.91986),   # your Excel value
        ((2, 7, 12), 0.63166),   # your Excel value
    ]

    print(f"{'Combo':<12} {'Exact':>10} {'Your Excel':>12} {'Difference':>12}")
    print("-" * 50)
    for combo, excel_val in test_cases:
        exact = table[combo]["prob_adv"]
        diff = abs(exact - excel_val)
        print(f"{str(combo):<12} {exact:>10.6f} {excel_val:>12.5f} {diff:>12.6f}")

    print("\nFull entry for columns (6,7,8) — the strongest combo:")
    entry = table[(6,7,8)]
    print(f"  Prob advance:  {entry['prob_adv']}")
    print(f"  Avg progress:  {entry['avg_prog']}")
    print(f"  Break even:    {entry['break_even']}")