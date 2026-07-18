"""Phase B wiring gate (plan §4): overfit a 500-game bot buffer to ~zero loss,
and check the aux heads beat base-rate baselines on held-out bot games.

Run: python -m games.seven_wonders_duel.phase_b_gate [--games 500] [--epochs 80]

Steps:
1. Generate (or reuse) a 500-game bot buffer via GameRecorder — pairings cycle
   Greedy/Random for diversity; policy targets are the played moves.
2. Featurize once (cached .pt of Example objects).
3. Overfit gate: transformer on all states — policy CE and value CE must fall
   below thresholds that only full memorization reaches.
4. Generalization check: fresh transformer + MLP control on a game-honest
   split — value/joint7 accuracy must beat the majority-class base rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .bots import GreedyBot, RandomBot
from .buffer import GameRecorder, append_records, read_records
from .codec import encode_action
from .dataset import examples_from_records
from .game import Phase
from .train import baselines, build_model, evaluate, game_honest_split, train_loop

RUNS = Path(__file__).parent / "runs"


def generate_buffer(path: Path, games: int) -> None:
    print(f"generating {games} bot games -> {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in range(games):
        pairing = seed % 4
        bots = {
            0: (GreedyBot(), GreedyBot()),
            1: (GreedyBot(), RandomBot(seed=seed * 2 + 1)),
            2: (RandomBot(seed=seed * 2 + 1), GreedyBot()),
            3: (RandomBot(seed=seed * 2 + 1), RandomBot(seed=seed * 2 + 2)),
        }[pairing]
        names = {0: "greedy/greedy", 1: "greedy/random", 2: "random/greedy", 3: "random/random"}
        recorder = GameRecorder(
            seed,
            first_player=seed % 2,
            agents={"p0": names[pairing].split("/")[0], "p1": names[pairing].split("/")[1]},
        )
        while recorder.game.phase is not Phase.COMPLETE:
            actor = (
                recorder.game.pending_choice.player
                if recorder.game.pending_choice is not None
                else recorder.game.active_player
            )
            action = bots[actor].select_action(recorder.game)
            recorder.play(encode_action(recorder.game, action))
        records.append(recorder.finish())
        if (seed + 1) % 100 == 0:
            print(f"  {seed + 1}/{games} games")
    append_records(path, records)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    buffer_path = RUNS / "phase_b_bot_buffer.jsonl"
    cache_path = RUNS / "phase_b_examples.pt"
    if not buffer_path.exists():
        generate_buffer(buffer_path, args.games)
    records = read_records(buffer_path)
    print(f"buffer: {len(records)} games")

    if cache_path.exists():
        examples = torch.load(cache_path, weights_only=False)
        print(f"featurized cache: {len(examples)} states")
    else:
        print("featurizing (one-time)...")
        examples = examples_from_records(records)
        torch.save(examples, cache_path)
        print(f"featurized {len(examples)} states -> {cache_path}")

    base = baselines(examples)
    print(f"baselines: {json.dumps({k: round(v, 4) for k, v in base.items()})}")
    results: dict = {"baselines": base, "states": len(examples)}

    # --- Gate 1: overfit to ~zero loss (wiring check) -----------------------
    print("\n[gate 1] transformer overfit on all states")
    model = build_model("transformer", 128, 4)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
    train_loop(
        model,
        list(examples),
        None,
        device=args.device,
        epochs=args.epochs,
        lr=4e-4,
        log=lambda m: print(f"  {m}"),
    )
    overfit = evaluate(model, examples, args.device)
    results["overfit"] = overfit
    overfit_pass = overfit["policy"] < 0.15 and overfit["value"] < 0.10
    print(
        f"  overfit: policy {overfit['policy']:.4f} value {overfit['value']:.4f} "
        f"policy_top1 {overfit['policy_top1']:.3f} -> "
        f"{'PASS' if overfit_pass else 'FAIL'}"
    )

    # --- Gate 2: aux heads beat base rates on held-out games ----------------
    print("\n[gate 2] generalization vs base rates (game-honest split)")
    train_examples, val_examples = game_honest_split(list(examples), 0.15)
    gate2 = {}
    for name in ("transformer", "mlp"):
        fresh = build_model(name, 128, 4)
        train_loop(
            fresh,
            train_examples,
            val_examples,
            device=args.device,
            epochs=min(args.epochs, 40),
            log=lambda m: print(f"  [{name}] {m}"),
        )
        metrics = evaluate(fresh, val_examples, args.device)
        gate2[name] = metrics
        print(
            f"  {name}: value_acc {metrics['value_acc']:.3f} "
            f"(base {base['value_base_rate']:.3f}) "
            f"joint7_acc {metrics['joint7_acc']:.3f} "
            f"(base {base['joint7_base_rate']:.3f})"
        )
    results["generalization"] = gate2
    aux_pass = all(
        gate2[m]["value_acc"] > base["value_base_rate"]
        and gate2[m]["joint7_acc"] > base["joint7_base_rate"]
        for m in gate2
    )
    print(f"  aux-vs-baseline: {'PASS' if aux_pass else 'FAIL'}")

    (RUNS / "phase_b_gate.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nresults -> {RUNS / 'phase_b_gate.json'}")
    if overfit_pass and aux_pass:
        print("PHASE B GATE: PASS")
        return 0
    print("PHASE B GATE: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
