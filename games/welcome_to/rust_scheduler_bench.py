"""M6 real-path throughput and discrete-fingerprint gate.

The blocking arm is M5: Rust owns each game/search but calls Torch one row at a
time. The scheduler arm keeps several complete games in flight, coalesces both
LEAF and POLICY rows, and immediately replenishes a slot when its game ends.
They use the same network, game seeds, per-decision search seeds, and search
configuration; only evaluation geometry differs.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from games.welcome_to import mcts
from games.welcome_to import network as nw
from games.welcome_to import rust_search
from games.welcome_to.portable_rng import derive_search_seed

import welcome_to_rust as wr


@dataclass
class LiveGame:
    seed: int
    state: object
    decision: int = 0
    actions: list[int] | None = None

    def __post_init__(self) -> None:
        if self.actions is None:
            self.actions = []


def _new(seed: int, players: int, advanced: bool) -> LiveGame:
    return LiveGame(
        seed,
        wr.RustGameState(
            seed,
            players=players,
            advanced=advanced,
            solo_rules=False,
        ),
    )


def _fingerprints(games: list[LiveGame]) -> dict[int, str]:
    return {
        game.seed: mcts.trajectory_fingerprint(game.actions or []) for game in games
    }


def run_blocking(
    net: nw.WelcomeToNet,
    device: torch.device,
    config: mcts.SearchConfig,
    seeds: list[int],
    players: int,
    advanced: bool,
) -> tuple[dict[int, str], dict[str, float]]:
    evaluator = rust_search.PackedNetEvaluator(net, device, config)
    finished: list[LiveGame] = []
    started = time.perf_counter()
    for seed in seeds:
        game = _new(seed, players, advanced)
        searches = [rust_search.native_search(config) for _ in range(players)]
        while not game.state.is_terminal:
            actor = game.state.actor
            result = searches[actor].play(
                game.state,
                evaluator,
                derive_search_seed(seed, game.decision),
                actor,
            )
            choice = result["choice"]
            game.state.apply_macro(choice)
            game.actions.append(choice)
            game.decision += 1
        finished.append(game)
    wall = time.perf_counter() - started
    return _fingerprints(finished), {
        "wall": wall,
        "games_h": len(seeds) * 3600.0 / wall,
        "rows_s": evaluator.rows / wall,
        "calls": float(evaluator.calls),
        "rows": float(evaluator.rows),
    }


def run_scheduler(
    net: nw.WelcomeToNet,
    device: torch.device,
    config: mcts.SearchConfig,
    seeds: list[int],
    players: int,
    advanced: bool,
    inflight: int,
    max_batch: int,
) -> tuple[dict[int, str], dict[str, float]]:
    width = min(inflight, len(seeds))
    scheduler = rust_search.native_scheduler(config, capacity=width * players)
    evaluator = rust_search.PackedNetEvaluator(net, device, config)
    live: list[LiveGame | None] = [None] * width
    next_seed = 0
    for slot in range(width):
        live[slot] = _new(seeds[next_seed], players, advanced)
        next_seed += 1

    finished: list[LiveGame] = []
    started = time.perf_counter()
    while len(finished) < len(seeds):
        active_slots = [slot for slot, game in enumerate(live) if game is not None]
        games = [live[slot] for slot in active_slots]
        states = [game.state for game in games]
        actors = [game.state.actor for game in games]
        search_slots = [
            game_slot * players + actor
            for game_slot, actor in zip(active_slots, actors)
        ]
        search_seeds = [
            derive_search_seed(game.seed, game.decision) for game in games
        ]
        results = scheduler.play(
            states,
            evaluator,
            search_seeds,
            roots=actors,
            slots=search_slots,
            max_batch=max_batch,
        )
        for game_slot, game, result in zip(active_slots, games, results):
            choice = result["choice"]
            game.state.apply_macro(choice)
            game.actions.append(choice)
            game.decision += 1
            if not game.state.is_terminal:
                continue
            finished.append(game)
            for seat in range(players):
                scheduler.reset(game_slot * players + seat)
            if next_seed < len(seeds):
                live[game_slot] = _new(seeds[next_seed], players, advanced)
                next_seed += 1
            else:
                live[game_slot] = None
    wall = time.perf_counter() - started
    return _fingerprints(finished), {
        "wall": wall,
        "games_h": len(seeds) * 3600.0 / wall,
        "rows_s": evaluator.rows / wall,
        "calls": float(evaluator.calls),
        "rows": float(evaluator.rows),
        "mean_batch": evaluator.rows / max(evaluator.calls, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--inflight", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--players", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--advanced", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.games <= 0 or args.inflight <= 0:
        parser.error("games and inflight must be positive")

    torch.manual_seed(0xA17E)
    device = torch.device(args.device)
    net = nw.WelcomeToNet().to(device).eval()
    config = mcts.SearchConfig(simulations=args.simulations)
    seeds = list(range(606_000, 606_000 + args.games))
    blocking_fp, blocking = run_blocking(
        net, device, config, seeds, args.players, args.advanced
    )
    scheduled_fp, scheduled = run_scheduler(
        net,
        device,
        config,
        seeds,
        args.players,
        args.advanced,
        args.inflight,
        args.max_batch,
    )
    if scheduled_fp != blocking_fp:
        mismatches = [seed for seed in seeds if scheduled_fp[seed] != blocking_fp[seed]]
        raise AssertionError(f"M6 changed {len(mismatches)} game fingerprints: {mismatches[:8]}")
    print(
        f"M5 blocking: {blocking['games_h']:,.1f} games/h, "
        f"{blocking['rows_s']:,.1f} rows/s, {int(blocking['calls']):,} calls"
    )
    print(
        f"M6 scheduler: {scheduled['games_h']:,.1f} games/h, "
        f"{scheduled['rows_s']:,.1f} rows/s, {int(scheduled['calls']):,} calls, "
        f"mean batch {scheduled['mean_batch']:.1f}"
    )
    print(
        f"speedup {scheduled['games_h'] / blocking['games_h']:.2f}x; "
        f"{len(seeds)}/{len(seeds)} discrete fingerprints identical"
    )


if __name__ == "__main__":
    main()
