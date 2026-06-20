"""
LEGACY — Match runner for the pre-AlphaZero MCTS bot (mcts.py).

Use round_robin_eval.py for evaluation of trained AlphaZero checkpoints.
Use bot_match.py for simple baseline bot matches.
"""
import argparse

from games.kingdomino.bots import GreedyBot
from games.kingdomino.mcts import MCTSBot
from games.kingdomino.bot_match import run_match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--depth", type=int, default=8)
    args = parser.parse_args()

    run_match(
        label=f"MCTS 25 sims depth-{args.depth} random rollout as P0 vs Greedy as P1",
        bot0=MCTSBot(
            simulations=25,
            rollout_policy="random",
            rollout_depth_limit=args.depth,
            seed=1,
        ),
        bot1=GreedyBot(),
        games=args.games,
        seed_offset=50_000,
        verbose=True,
    )

    run_match(
        label=f"Greedy as P0 vs MCTS 25 sims depth-{args.depth} random rollout as P1",
        bot0=GreedyBot(),
        bot1=MCTSBot(
            simulations=25,
            rollout_policy="random",
            rollout_depth_limit=args.depth,
            seed=2,
        ),
        games=args.games,
        seed_offset=60_000,
        verbose=True,
    )


if __name__ == "__main__":
    main()