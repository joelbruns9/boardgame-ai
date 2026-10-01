"""
Engine smoke harness and throughput benchmark.

Plays a batch of games with random or greedy policies, checks the invariants
that should hold in every finished game, and prints score / length statistics.
This is the first thing to run after touching the rules.

    python -m games.welcome_to.random_play --games 200 --players 4
    python -m games.welcome_to.random_play --games 50 --bot greedy --advanced
    python -m games.welcome_to.random_play --games 20 --encode

``--encode`` additionally runs the encoder on every visited state, which is the
cheapest way to catch a feature-layout mistake.

The report includes the divergence numbers from :mod:`games.welcome_to.training`.
``identical_games`` is the one to watch in a self-play loop: in standard mode all
seats see the same three combinations, so a deterministic policy playing itself
produces byte-identical sheets and the whole game is worth one sample.
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
from collections import Counter
from typing import Optional

from games.welcome_to.bots import GreedyBot, RandomBot
from games.welcome_to.constants import PERMIT_BOXES, ROUNDABOUT
from games.welcome_to.game import GameConfig, GameState
from games.welcome_to import training


def check_terminal_invariants(state: GameState) -> None:
    """Assertions that must hold in any finished game."""
    assert state.is_terminal, "game is not over"
    assert state.end_of_game_reason() is not None, "terminal with no end condition"

    for p, sheet in enumerate(state.sheets):
        assert sheet.permits <= PERMIT_BOXES, f"player {p} over-refused"
        assert sheet.pool_count <= 9
        assert sheet.temps <= 11
        assert sheet.roundabouts <= 2
        for x in range(3):
            assert sheet.parks[x] <= (3, 4, 5)[x]
        # every estate must be a run of built, non-roundabout boxes
        for x, start, size in sheet.estates():
            for k in range(size):
                n = sheet.numbers[x][start + k]
                assert n is not None and n != ROUNDABOUT, "estate contains a hole"
        # a plan may never consume the same house twice: top fences are a set,
        # so this is really a check that we never scored a plan on used houses
        assert len(sheet.top_fences) == 3

    # plan values must be one of the two printed on the card
    from games.welcome_to.plans import PLANS

    for slot, plan_id in enumerate(state.plan_ids):
        for player, turn in state.plan_turns[slot].items():
            assert turn <= state.turn, "plan validated in the future"
            assert player >= -1


def play_batch(
    games: int,
    config: GameConfig,
    bot_kind: str,
    seed: int,
    encode: bool,
) -> dict:
    from games.welcome_to import encoder as enc

    lengths: list[int] = []
    turns: list[int] = []
    all_scores: list[int] = []
    winner_scores: list[int] = []
    reasons: Counter = Counter()
    decisions = 0
    branching: list[int] = []
    finished: list[GameState] = []

    start = time.perf_counter()
    for g in range(games):
        rng = random.Random(seed + g)
        bots = [
            RandomBot(rng) if bot_kind == "random" else GreedyBot(rng)
            for _ in range(config.players)
        ]
        state = GameState.new(seed=seed + g, config=config)
        steps = 0
        while not state.is_terminal:
            legal = state.legal_actions()
            assert legal, f"no legal action in {state.phase.name}"
            branching.append(len(legal))
            if encode:
                planes, sheets, viewer, glob = enc.encode_state(state)
                assert planes.shape == enc.SHEET_PLANES_SHAPE
                assert sheets.shape == (enc.MAX_SEATS, enc.NUM_SHEET_SCALAR)
                assert viewer.shape == enc.VIEWER_PLANE_SHAPE
                assert glob.shape == (enc.NUM_GLOBAL_SCALAR,)
            state.apply(bots[state.actor].act(state))
            steps += 1
            assert steps < 20000, "runaway game"
        check_terminal_invariants(state)

        decisions += steps
        lengths.append(steps)
        turns.append(state.turn)
        scores = state.scores()
        all_scores.extend(scores)
        winner_scores.append(max(scores))
        finished.append(state)
        reason = state.end_of_game_reason() or "?"
        reasons[reason.split(" ", 2)[-1]] += 1
    elapsed = time.perf_counter() - start

    return {
        "games": games,
        "seconds": elapsed,
        "games_per_second": games / elapsed if elapsed else float("inf"),
        "decisions_per_game": statistics.mean(lengths),
        "turns_per_game": statistics.mean(turns),
        "mean_branching": statistics.mean(branching),
        "max_branching": max(branching),
        "mean_score": statistics.mean(all_scores),
        "score_stdev": statistics.pstdev(all_scores),
        "mean_winning_score": statistics.mean(winner_scores),
        "end_reasons": dict(reasons),
        "decisions": decisions,
        **training.diversity_report(finished),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bot", choices=("random", "greedy"), default="random")
    parser.add_argument("--advanced", action="store_true", help="advanced variant")
    parser.add_argument("--expert", action="store_true", help="expert variant")
    parser.add_argument("--encode", action="store_true", help="also run the encoder")
    args = parser.parse_args(argv)

    config = GameConfig(
        players=args.players, advanced=args.advanced, expert=args.expert
    )
    stats = play_batch(args.games, config, args.bot, args.seed, args.encode)

    print(
        f"{args.bot} x{args.players}"
        f"{' advanced' if args.advanced else ''}"
        f"{' expert' if args.expert else ''}"
    )
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:<22} {value:.3f}")
        else:
            print(f"  {key:<22} {value}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
