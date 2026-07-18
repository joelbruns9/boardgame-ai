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
   split — ALL heads must beat their trivial baselines: value and joint7
   accuracy above majority-class rate, margin/military/science MAE below the
   predict-the-mean baseline.

The buffer must match the requested game count, and the featurized cache is
validated against the encoder signature and buffer file hash — a stale cache
is re-featurized, never silently reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .bots import GreedyBot, RandomBot
from .buffer import GameRecorder, append_records, read_records
from .codec import encode_action
from .dataset import examples_from_records
from .encoder import ENCODER_SIGNATURE
from .game import Phase
from .train import baselines, build_model, evaluate, game_honest_split, train_loop

RUNS = Path(__file__).parent / "runs"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_examples(buffer_path: Path, cache_path: Path):
    """Featurize with a validated cache: reuse only when the encoder signature
    AND the buffer file hash both match; anything else re-featurizes."""

    buffer_sha = _file_sha256(buffer_path)
    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        if (
            isinstance(cached, dict)
            and cached.get("encoder_signature") == ENCODER_SIGNATURE
            and cached.get("buffer_sha256") == buffer_sha
        ):
            print(f"featurized cache valid: {len(cached['examples'])} states")
            return cached["examples"]
        print("featurized cache stale (encoder or buffer changed); rebuilding")
    print("featurizing (verified replay)...")
    examples = examples_from_records(read_records(buffer_path))
    torch.save(
        {
            "encoder_signature": ENCODER_SIGNATURE,
            "buffer_sha256": buffer_sha,
            "examples": examples,
        },
        cache_path,
    )
    print(f"featurized {len(examples)} states -> {cache_path}")
    return examples


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
    if len(records) != args.games:
        raise SystemExit(
            f"buffer {buffer_path} holds {len(records)} games but --games is "
            f"{args.games}; delete the buffer to regenerate at the new size"
        )
    print(f"buffer: {len(records)} games")
    examples = load_examples(buffer_path, cache_path)

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

    # --- Gate 2: every head beats its baseline on held-out games ------------
    print("\n[gate 2] generalization: all six heads vs baselines (game-honest split)")
    train_examples, val_examples = game_honest_split(list(examples), 0.15)
    val_base = baselines(val_examples)
    gate2 = {}
    head_checks = {}
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
        checks = {
            "value": metrics["value_acc"] > val_base["value_base_rate"],
            "joint7": metrics["joint7_acc"] > val_base["joint7_base_rate"],
            "policy": metrics["policy"] < val_base["policy_uniform_loss"],
            "margin": metrics["margin_mae"] < val_base["margin_mae"],
            "military": metrics["military_mae"] < val_base["military_mae"],
            "science": metrics["science_mae"] < val_base["science_mae"],
        }
        head_checks[name] = checks
        print(
            f"  {name}: value_acc {metrics['value_acc']:.3f}/{val_base['value_base_rate']:.3f} "
            f"joint7_acc {metrics['joint7_acc']:.3f}/{val_base['joint7_base_rate']:.3f} "
            f"policy {metrics['policy']:.3f}/{val_base['policy_uniform_loss']:.3f}"
        )
        print(
            f"    margin_mae {metrics['margin_mae']:.4f}/{val_base['margin_mae']:.4f} "
            f"military_mae {metrics['military_mae']:.4f}/{val_base['military_mae']:.4f} "
            f"science_mae {metrics['science_mae']:.4f}/{val_base['science_mae']:.4f} "
            f"-> {'all heads PASS' if all(checks.values()) else 'FAIL: ' + str([k for k, v in checks.items() if not v])}"
        )
    results["generalization"] = gate2
    results["val_baselines"] = val_base
    results["head_checks"] = head_checks
    aux_pass = all(all(checks.values()) for checks in head_checks.values())
    print(f"  all-heads-vs-baselines: {'PASS' if aux_pass else 'FAIL'}")

    (RUNS / "phase_b_gate.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nresults -> {RUNS / 'phase_b_gate.json'}")
    if overfit_pass and aux_pass:
        print("PHASE B GATE: PASS")
        return 0
    print("PHASE B GATE: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
