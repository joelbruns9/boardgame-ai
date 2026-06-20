from games.kingdomino.test_engine_rules import main as engine_main
from games.kingdomino.test_scoring_rules import main as scoring_main
from games.kingdomino.test_placement_rules import main as placement_main
from games.kingdomino.random_play import play_random_game


def random_smoke_test(games=100):
    for seed in range(games):
        state = play_random_game(seed=seed, verbose=False)

        if len(state.history) != 52:
            raise AssertionError(
                f"Unexpected game length at seed={seed}: {len(state.history)}"
            )

    print(f"Random smoke test passed ({games} games)")


def main():
    print("Running engine rule tests...")
    engine_main()

    print()
    print("Running scoring tests...")
    scoring_main()

    print()
    print("Running placement tests...")
    placement_main()

    print()
    print("Running random smoke test...")
    random_smoke_test()

    print()
    print("All Kingdomino tests passed")


if __name__ == "__main__":
    main()