from games.kingdomino.random_play import play_random_game


def total_score(score):
    return (
        score.territory_score
        + score.harmony_bonus
        + score.middle_kingdom_bonus
    )


def main():
    games = 1000

    all_totals = []
    harmony_count = 0
    middle_count = 0

    for seed in range(games):
        state = play_random_game(seed=seed, verbose=False)

        scores = [board.score() for board in state.boards]
        totals = [total_score(score) for score in scores]

        all_totals.extend(totals)

        harmony_count += sum(1 for score in scores if score.harmony_bonus > 0)
        middle_count += sum(1 for score in scores if score.middle_kingdom_bonus > 0)

        print(
            f"seed={seed:03d} "
            f"totals={totals} "
            f"breakdowns={scores} "
            f"steps={len(state.history)}"
        )

    avg_score = sum(all_totals) / len(all_totals)

    print()
    print("Random game summary")
    print("-------------------")
    print(f"games: {games}")
    print(f"player-results: {len(all_totals)}")
    print(f"min score: {min(all_totals)}")
    print(f"max score: {max(all_totals)}")
    print(f"avg score: {avg_score:.2f}")
    print(f"harmony bonuses: {harmony_count}")
    print(f"middle kingdom bonuses: {middle_count}")


if __name__ == "__main__":
    main()