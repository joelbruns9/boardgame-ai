"""Is the policy improvement operator still doing anything?

An AlphaZero loop only moves because search beats the raw net: search plays the
better move, the net is trained toward it, the net gets better, search gets
better still.  If search at the training sim count is no longer meaningfully
stronger than the bare network, there is no signal to learn from and the loop is
a fixed point -- more iterations cannot help, whatever the hyperparameters say.

cloud6 stalled with 0 promotions in 38,000 games while its weights moved 1.6% of
their available travel over 1,900 optimizer steps.  That is what a fixed point
looks like from the inside, but it does not say whether the cause is upstream
(search has nothing left to teach) or downstream (it does, and the teaching is
not reaching the weights).  This separates them.

The **same checkpoint** sits on both sides, so the null is known exactly: with
equal search budgets it must score 0.500, and any deviation is purely the search
budget.  Seat-paired and fixed-N, so no seed-variance confound.

Reading the result:

  strong ~0.65+  search is still well ahead of the net.  The improvement
                 operator is healthy and the stall is downstream -- look at the
                 learning rate, the replay window, and the PUCT/Dirichlet
                 target path, which has never been validated in a training run.

  strong ~0.52   search has nothing left to teach this net at this sim count.
                 The loop is mathematically a fixed point and "the model is
                 maxed out" is the finding, not a guess.

The weak arm defaults to ``--weak-sims 1``.  Under ``puct_root`` every child
starts at Q=0, so the single visit goes to the argmax of the network prior and
the returned policy is the bare network's move.  If that turns out degenerate on
your build, raise it to 8; the comparison is still 1600-vs-shallow and reads the
same way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from games.az_loop import hardware_identity, wilson_interval

from .inference import Evaluator
from .phase_d import PhaseDConfig, PhaseDLoop
from .rust_bridge import rust_flat_batch_adapter, rust_games_for_self_play
from .train import build_model, heads_from_config


def _play(swr, config, games, seeds, adapter, seat_sims: tuple[int, int]) -> list[dict]:
    """Step a batch of games; each seat searches at its own sim count.

    Mirrors ``PhaseDLoop._play_two_net_games`` except that the sim count comes
    from the seat rather than from one global ``gate_sims``.  Both seats share
    one adapter because both seats are the same network -- the search budget is
    the only difference, which is what makes the 0.500 null exact.
    """

    live = list(range(len(games)))
    actions_played = [0] * len(games)
    move_index = 0
    while live:
        by_seat: dict[int, list[int]] = {0: [], 1: []}
        for slot in live:
            by_seat[games[slot].actor].append(slot)
        for seat, slots in by_seat.items():
            if not slots:
                continue
            results = swr.search_many_flat_net(
                adapter,
                [games[slot] for slot in slots],
                [
                    seeds[slot] + move_index * 1_000_003 + seat * 7_919
                    for slot in slots
                ],
                config.gate_batch_cap(),
                1,
                seat_sims[seat],
                config.top_k,
                force=config.force_root_chance,
                age_deal_samples=config.age_deal_samples,
                puct_root=True,
            )
            for slot, result in zip(slots, results):
                legal = games[slot].legal_action_indices()
                policy = result["policy"]
                best = max(range(len(legal)), key=lambda i: policy[i])
                games[slot].apply_index(legal[best])
                actions_played[slot] += 1
        move_index += 1
        live = [slot for slot in live if not games[slot].is_complete()]
    return [
        {"winner": game.winner, "victory_type": game.victory_type, "actions": actions}
        for game, actions in zip(games, actions_played)
    ]


def run(
    checkpoint: Path,
    *,
    games: int,
    device: str,
    strong_sims: int,
    weak_sims: int,
    slots: int,
    global_batch_cap: int,
    work_dir: Path,
    z: float = 1.96,
) -> dict:
    import torch

    import seven_wonders_rust as swr

    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = stored.get("config", {})
    d_model = int(meta.get("d_model", 384))
    layers = int(meta.get("layers", 8))
    heads = heads_from_config(meta)

    config = PhaseDConfig(
        run_dir=str(work_dir),
        device=device,
        d_model=d_model,
        layers=layers,
        heads=heads,
        precision=str(meta.get("precision", "bf16")),
        gate_backend="rust",
        gate_max_games=games,
        gate_slots=slots,
        rust_slots=slots,
        rust_global_batch_cap=global_batch_cap,
        promotion_every=0,
        seed_games=0,
    )
    loop = PhaseDLoop(config)
    spec = loop._model_agent_spec(checkpoint, "search_gain_probe")

    model = build_model("transformer", spec.d_model, spec.layers, spec.heads)
    model.load_state_dict(spec.model_state)
    adapter = rust_flat_batch_adapter(
        Evaluator(model, device, config.gate_batch_cap(), precision=config.precision)
    )

    pairs = games // 2
    started = time.monotonic()
    strong_points = 0.0
    decisive = 0
    moves = 0
    for start in range(0, pairs, slots):
        block = list(range(start, min(start + slots, pairs)))
        seeds = [config.seed + 71_000_000 + pair for pair in block]
        first_players = [pair % 2 for pair in block]
        for strong_seat in (0, 1):
            seat_sims = (
                (strong_sims, weak_sims) if strong_seat == 0 else (weak_sims, strong_sims)
            )
            records = _play(
                swr,
                config,
                rust_games_for_self_play(seeds, first_players),
                seeds,
                adapter,
                seat_sims,
            )
            for record in records:
                moves += record["actions"]
                if record["winner"] is None:
                    strong_points += 0.5
                elif record["winner"] == strong_seat:
                    strong_points += 1.0
                    decisive += 1
                else:
                    decisive += 1
        print(
            f"  {min(start + slots, pairs)}/{pairs} pairs | "
            f"strong so far {strong_points / (min(start + slots, pairs) * 2):.3f} | "
            f"{time.monotonic() - started:.0f}s",
            flush=True,
        )

    played = pairs * 2
    rate = strong_points / played
    lower, upper = wilson_interval(strong_points, played, z=z)
    seconds = time.monotonic() - started

    # An interval wide enough to contain both hypotheses is not evidence for
    # either.  Without this guard a 4-game run reports "search has nothing left
    # to teach" with total confidence, because the null is inside [0.30, 0.95]
    # -- so is 0.65.  Distinguishing ~0.52 from ~0.65 needs a half-width under
    # about 0.065, which is roughly 200 games.
    half_width = (upper - lower) / 2
    underpowered = half_width > 0.065
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "architecture": {"d_model": d_model, "layers": layers, "heads": heads},
        "strong_sims": strong_sims,
        "weak_sims": weak_sims,
        "games": played,
        "pairs": pairs,
        "strong_score_rate": rate,
        "wilson": {"lower": lower, "upper": upper, "z": z},
        "null": 0.50,
        "null_inside_interval": lower <= 0.50 <= upper,
        "interval_half_width": half_width,
        "underpowered": underpowered,
        "decisive_games": decisive,
        "moves_per_game": moves / played if played else 0.0,
        "seconds": seconds,
        "games_per_second": played / seconds if seconds else 0.0,
        "hardware": hardware_identity(),
        "verdict": (
            f"INCONCLUSIVE: {played} games gives a half-width of {half_width:.3f}, "
            "which contains both the fixed-point hypothesis (~0.52) and the "
            "healthy-search one (~0.65). Re-run with more games; this is not a "
            "null result"
            if underpowered
            else f"{strong_sims} sims is indistinguishable from {weak_sims}: "
            f"{rate:.3f} [{lower:.3f},{upper:.3f}]. Search has nothing left to "
            "teach this net, and the training loop is a fixed point"
            if lower <= 0.50 <= upper
            else f"{strong_sims} sims beats {weak_sims} by "
            f"{rate:.3f} [{lower:.3f},{upper:.3f}]: the improvement operator is "
            "alive and the stall is downstream of search"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strong-sims", type=int, default=1600)
    parser.add_argument("--weak-sims", type=int, default=1)
    # Deliberately modest: this is meant to run beside a training job without
    # taking slots or VRAM away from it.
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--global-batch-cap", type=int, default=512)
    args = parser.parse_args(argv)

    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even number (seat pairs)")
    if args.strong_sims <= args.weak_sims:
        parser.error("--strong-sims must exceed --weak-sims")
    if not args.checkpoint.is_file():
        parser.error(f"no checkpoint at {args.checkpoint}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    report = run(
        args.checkpoint,
        games=args.games,
        device=args.device,
        strong_sims=args.strong_sims,
        weak_sims=args.weak_sims,
        slots=args.slots,
        global_batch_cap=args.global_batch_cap,
        work_dir=args.work_dir,
    )
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
