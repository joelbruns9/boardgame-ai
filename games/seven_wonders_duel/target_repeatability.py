"""Test A: is `policy_top1` pinned at the network's ceiling, or at the target's?

cloud6 trains toward search targets and sits at `policy_top1 = 0.78` -- it picks
search's move 78% of the time and has stopped improving.  Two readings:

  * the net has absorbed everything learnable and the residual 22% is noise in
    the targets themselves.  No learning rate, model size or step count fixes
    that, because there is nothing left to fit.
  * the targets are sharp and the 22% is real structure the net is failing to
    take on.  Then the stall is optimization or capacity, and is fixable.

The discriminator is whether the search agrees with *itself*.  Search the same
position twice at the same budget under two different seeds: chance sampling and
the search's own tie-breaking make it non-deterministic.  If two independent
1600-sim searches only pick the same move 78% of the time, then a network
matching them 78% of the time has hit the target's own self-consistency and is
finished.  If they agree 95% of the time, the net is leaving 17 points on the
table.

All three numbers come off one shared corpus so they are directly comparable:

  search_vs_search   how often two searches of the same position agree
  search_vs_prior    how often the search agrees with the bare network
  headroom           search_vs_search - search_vs_prior

The prior arm uses ``sims=1`` under ``puct_root``: with every child at Q=0 the
single visit lands on the argmax of the network prior, so the returned policy is
the bare net's move.

Runs against a saved checkpoint. No training, no promotion, nothing written back
into a run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

from games.az_loop import hardware_identity, wilson_interval

from .phase_e import load_evaluator
from .rust_bridge import (
    rust_flat_batch_adapter,
    rust_game_from_prefix,
    rust_games_for_self_play,
)


def _argmax_action(game, result) -> int:
    """The search's chosen move as an absolute action index.

    Deliberately the argmax of the improved policy rather than ``result["action"]``:
    the latter carries the search's own exploration perturbation, which would
    show up as disagreement that no network could ever have predicted.
    """

    legal = game.legal_action_indices()
    policy = result["policy"]
    return legal[max(range(len(legal)), key=lambda i: policy[i])]


def _collect_positions(
    swr,
    adapter,
    *,
    games,
    gen_sims,
    top_k,
    batch_cap,
    slots,
    seed,
    rng,
    force_root_chance,
    age_deal_samples,
):
    """Play games at a cheap budget, then keep one mid-game position from each.

    One position per game by default: positions inside a single game share a
    board and would understate the variance of the agreement estimate.
    """

    collected: list[tuple[int, int, list[int]]] = []
    for start in range(0, games, slots):
        block = list(range(start, min(start + slots, games)))
        seeds = [seed + pair for pair in block]
        first_players = [pair % 2 for pair in block]
        live_games = rust_games_for_self_play(seeds, first_players)
        prefixes: list[list[int]] = [[] for _ in live_games]
        live = list(range(len(live_games)))
        move_index = 0
        while live:
            by_seat: dict[int, list[int]] = {0: [], 1: []}
            for slot in live:
                by_seat[live_games[slot].actor].append(slot)
            for seat, slots_now in by_seat.items():
                if not slots_now:
                    continue
                results = swr.search_many_flat_net(
                    adapter,
                    [live_games[slot] for slot in slots_now],
                    [
                        seeds[slot] + move_index * 1_000_003 + seat * 7_919
                        for slot in slots_now
                    ],
                    batch_cap,
                    1,
                    gen_sims,
                    top_k,
                    force=force_root_chance,
                    age_deal_samples=age_deal_samples,
                    puct_root=True,
                )
                for slot, result in zip(slots_now, results):
                    action = _argmax_action(live_games[slot], result)
                    live_games[slot].apply_index(action)
                    prefixes[slot].append(action)
            move_index += 1
            live = [slot for slot in live if not live_games[slot].is_complete()]
        for slot, prefix in enumerate(prefixes):
            # Middle game only. Openings are near-deterministic and the endgame
            # is often forced; neither says anything about target sharpness.
            low, high = int(0.2 * len(prefix)), int(0.8 * len(prefix))
            if high <= low:
                continue
            collected.append((seeds[slot], first_players[slot], prefix[: rng.randrange(low, high)]))
        print(f"  collected {len(collected)}/{games} positions", flush=True)
    return collected


def run(
    checkpoint: Path,
    *,
    positions: int,
    sims: int,
    gen_sims: int,
    top_k: int,
    device: str,
    slots: int,
    global_batch_cap: int,
    seed: int,
    force_root_chance: bool = True,
    age_deal_samples: int = 32,
    z: float = 1.96,
) -> dict:
    import seven_wonders_rust as swr

    evaluator = load_evaluator(str(checkpoint), device)
    evaluator.max_batch = global_batch_cap
    adapter = rust_flat_batch_adapter(evaluator)
    rng = random.Random(seed)

    started = time.monotonic()
    print(f"generating {positions} positions at {gen_sims} sims...", flush=True)
    prepared = _collect_positions(
        swr,
        adapter,
        games=positions,
        gen_sims=gen_sims,
        top_k=top_k,
        batch_cap=global_batch_cap,
        slots=slots,
        seed=seed,
        rng=rng,
        force_root_chance=force_root_chance,
        age_deal_samples=age_deal_samples,
    )
    collected = time.monotonic()
    print(
        f"searching {len(prepared)} positions twice at {sims} sims...",
        flush=True,
    )

    ss_agree = 0
    sp_agree = 0
    total = 0
    for start in range(0, len(prepared), slots):
        batch = prepared[start : start + slots]
        rebuilt = [
            rust_game_from_prefix(game_seed, first_player, prefix)[1]
            for game_seed, first_player, prefix in batch
        ]
        seeds_a = [seed + 900_000 + start + i for i in range(len(batch))]
        seeds_b = [seed + 4_500_000 + start + i for i in range(len(batch))]

        def search(search_seeds, budget):
            return swr.search_many_flat_net(
                adapter,
                rebuilt,
                search_seeds,
                global_batch_cap,
                1,
                budget,
                top_k,
                force=force_root_chance,
                age_deal_samples=age_deal_samples,
                puct_root=True,
            )

        first = search(seeds_a, sims)
        second = search(seeds_b, sims)
        prior = search(seeds_a, 1)
        for game, one, two, bare in zip(rebuilt, first, second, prior):
            a = _argmax_action(game, one)
            b = _argmax_action(game, two)
            p = _argmax_action(game, bare)
            ss_agree += a == b
            sp_agree += a == p
            total += 1
        print(
            f"  {total}/{len(prepared)} | search-vs-search {ss_agree / total:.3f} "
            f"| search-vs-prior {sp_agree / total:.3f} "
            f"| {time.monotonic() - collected:.0f}s",
            flush=True,
        )

    ss_rate = ss_agree / total
    sp_rate = sp_agree / total
    ss_lo, ss_hi = wilson_interval(ss_agree, total, z=z)
    sp_lo, sp_hi = wilson_interval(sp_agree, total, z=z)
    # Non-overlapping intervals are sufficient for a real gap but not necessary;
    # the two rates are measured on the same positions and are therefore
    # positively correlated, which this interval ignores. Treat overlap as
    # "not established", never as "no difference".
    headroom = ss_rate - sp_rate
    separated = sp_hi < ss_lo
    # Same trap as the search-gain probe: on a small corpus every interval
    # overlaps every other one, and "overlapping" would otherwise be reported as
    # the ceiling finding. Distinguishing a real 10-point headroom needs roughly
    # 300 positions.
    underpowered = max(ss_hi - ss_lo, sp_hi - sp_lo) / 2 > 0.05

    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "positions": total,
        "sims": sims,
        "generation_sims": gen_sims,
        "force_root_chance": force_root_chance,
        "age_deal_samples": age_deal_samples,
        "underpowered": underpowered,
        "measures_dirichlet_noise": False,
        "search_vs_search": {"rate": ss_rate, "lower": ss_lo, "upper": ss_hi},
        "search_vs_prior": {"rate": sp_rate, "lower": sp_lo, "upper": sp_hi},
        "headroom": headroom,
        "intervals_separated": separated,
        "seconds": {
            "generation": collected - started,
            "search": time.monotonic() - collected,
        },
        "hardware": hardware_identity(),
        "verdict": (
            f"INCONCLUSIVE: {total} positions leaves intervals too wide to "
            "separate the ceiling hypothesis from a real headroom. Re-run with "
            "more positions; this is not a null result"
            if underpowered
            else f"search agrees with itself {ss_rate:.3f} and with the bare net "
            f"{sp_rate:.3f}: the net has essentially reached the targets' own "
            "self-consistency, so the residual disagreement is target noise and "
            "no amount of training, learning rate or model size recovers it"
            if not separated
            else f"search agrees with itself {ss_rate:.3f} but with the bare net "
            f"only {sp_rate:.3f}, leaving {headroom:.3f} of real structure "
            "unlearned: the targets are sharp and the stall is optimization or "
            "capacity, not target noise"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--positions", type=int, default=300)
    parser.add_argument("--sims", type=int, default=1600, help="production full-search budget")
    parser.add_argument(
        "--gen-sims",
        type=int,
        default=100,
        help="budget used only to walk games to a sampling ply; does not enter any measurement",
    )
    parser.add_argument("--top-k", type=int, default=16)
    # Production values (phase_d.py:488-489). These are the whole experiment:
    # with force off and zero age-deal samples the search is deterministic, two
    # seeds agree 100% of the time, and the measurement is vacuous.
    parser.add_argument("--age-deal-samples", type=int, default=32)
    parser.add_argument("--no-force-root-chance", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--global-batch-cap", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args(argv)

    if args.positions <= 0:
        parser.error("--positions must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"no checkpoint at {args.checkpoint}")

    report = run(
        args.checkpoint,
        positions=args.positions,
        sims=args.sims,
        gen_sims=args.gen_sims,
        top_k=args.top_k,
        device=args.device,
        slots=args.slots,
        global_batch_cap=args.global_batch_cap,
        seed=args.seed,
        force_root_chance=not args.no_force_root_chance,
        age_deal_samples=args.age_deal_samples,
    )
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
