"""
S0 — the supervised bootstrap.

Clone GreedyBot into the network to prove ``encoder -> net -> loss ->
checkpoint`` against a known reference, before MCTS exists and can hide a bug in
any of them.  The point is not the resulting player.  If the net cannot
reproduce greedy's play without search, nothing downstream will work, and this is
the cheapest place to find that out.

THE SEAT MIXTURE IS NOT A DETAIL
────────────────────────────────
The corpus is captured at **2/3/4 seats, 60/30/10**.  A seat costs the same per
trajectory whatever the table size, and the mixture is what keeps every seat of
the encoder's seat axis non-zero from the first gradient step.  Training only at
two seats would leave half the shared sheet encoder's input permanently zero, and
first-layer weights trained against a constant have to be *unlearned* when it
comes alive.  It is also what makes the rank distribution mean anything: at two
seats it is a Bernoulli, and the whole reason it replaced a scalar rank is that
the same head has to serve three table sizes.

THE GATE
────────
Three numbers, from ``SELF_PLAY_PLAN.md`` S0:

1. policy top-1 agreement with greedy **>= 60%** on held-out games;
2. the net playing greedily off its own policy, no search, within **2 points**
   of greedy on a **paired** seed set;
3. the ``permits`` head beats predict-the-mean.

(3) is reported as R^2 against the held-out variance, which is exactly "beats
predict-the-mean > 0".  Every other head is reported the same way, for free --
but only ``permits`` gates, because it is the head this whole target set exists
for.  ``houses`` used to gate alongside it and no longer does: it is
extension-tier in ``AUX_TARGETS_SPEC.md`` §9.3 and could be dropped from the set
entirely later, and a gate should not depend on a target that might not exist.

(2) is **paired**: the same seeds, the same configs, greedy and the net measured
on each.  Welcome To's per-game score variance is large enough that an unpaired
comparison of a few hundred games says nothing about a 2-point difference.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import torch

from games.welcome_to import datagen
from games.welcome_to import encoder as enc
from games.welcome_to import network as nw
from games.welcome_to import training
from games.welcome_to.bots import GreedyBot
from games.welcome_to.datagen import Sample, Trajectory
from games.welcome_to.game import GameConfig, GameState

#: Seat counts and their share of the corpus, per SELF_PLAY_PLAN.md §2.
SEAT_MIX: tuple[tuple[int, float], ...] = ((2, 0.6), (3, 0.3), (4, 0.1))


# ──────────────────────────────────────────────────────────────────────────
# Corpus
# ──────────────────────────────────────────────────────────────────────────
def _greedy_factory(players: int):
    def factory(rng: random.Random):
        bots = [GreedyBot(random.Random(rng.randrange(1 << 30))) for _ in range(players)]
        return lambda state: bots[state.actor].act(state)

    return factory


def seat_counts(games: int, mix: Sequence[tuple[int, float]] = SEAT_MIX) -> list[int]:
    """How many games at each seat count, summing to ``games`` exactly.

    Largest-remainder rather than rounding, so a 10% share of a small corpus
    still gets its games instead of vanishing -- four-seat games are the rarest
    and the ones that keep the far end of the seat axis alive.
    """
    exact = [(count, games * share) for count, share in mix]
    out = {count: int(math.floor(value)) for count, value in exact}
    remainder = games - sum(out.values())
    for count, value in sorted(exact, key=lambda cv: -(cv[1] % 1.0))[:remainder]:
        out[count] += 1
    return [count for count, _ in mix for _ in range(out[count])]


def build_corpus(
    games: int, seed: int = 0, advanced: bool = True
) -> list[Trajectory]:
    """GreedyBot trajectories at the 60/30/10 seat mixture."""
    out: list[Trajectory] = []
    for players, _ in SEAT_MIX:
        n = sum(1 for c in seat_counts(games) if c == players)
        if not n:
            continue
        out.extend(
            datagen.generate(
                n,
                _greedy_factory(players),
                config=GameConfig(players=players, advanced=advanced),
                seed=seed + players * 1_000_003,
            )
        )
    return out


def iter_batches(
    trajectories: Sequence[Trajectory],
    batch_size: int,
    rng: random.Random,
    shuffle_buffer: int = 8192,
) -> Iterator[dict[str, np.ndarray]]:
    """Replay trajectories into shuffled batches.

    The corpus is stored as trajectories, not tensors -- 5000 games is about a
    megabyte on disk against many gigabytes of float32 -- so replay happens here,
    every epoch.  Consecutive samples from one game are near-identical states, so
    they go through a shuffle buffer first; a batch of one game's decisions is a
    batch of one gradient repeated.
    """
    order = list(trajectories)
    rng.shuffle(order)
    buffer: list[Sample] = []
    for trajectory in order:
        for sample in datagen.replay(trajectory):
            buffer.append(sample)
            if len(buffer) >= shuffle_buffer:
                rng.shuffle(buffer)
                while len(buffer) >= shuffle_buffer // 2 + batch_size:
                    yield datagen.batch(buffer[-batch_size:])
                    del buffer[-batch_size:]
    rng.shuffle(buffer)
    # The tail is yielded as a short batch rather than dropped.  Dropping it is
    # harmless for a training epoch and wrong for evaluation: a held-out set
    # smaller than one batch would silently evaluate on nothing at all and
    # report a metric of zero.
    for start in range(0, len(buffer), batch_size):
        yield datagen.batch(buffer[start : start + batch_size])


# ──────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    net: nw.WelcomeToNet,
    trajectories: Sequence[Trajectory],
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, float]:
    """Held-out policy agreement, and R^2 for every regression head.

    R^2 is against the held-out variance of the target itself, so ``> 0`` is
    exactly "beats predict-the-mean" and the number is comparable across heads
    whose scales differ.  Per-seat heads are scored **only over valid seats** --
    including padded ones would score the head on a constant and inflate every
    figure at two seats, which is 60% of the corpus.
    """
    net.eval()
    agree = total = 0
    sq_err: dict[str, float] = {}
    sums: dict[str, float] = {}
    sq_sums: dict[str, float] = {}
    weights: dict[str, float] = {}

    rng = random.Random(0)
    for raw in iter_batches(trajectories, batch_size, rng, shuffle_buffer=batch_size * 2):
        batch = nw.to_tensors(raw, device)
        out = net(
            batch["sheet_planes"],
            batch["sheet_scalars"],
            batch["viewer_plane"],
            batch["global_scalars"],
        )
        logits = out["policy_logits"].masked_fill(batch["legal"] <= 0, -1e9)
        agree += int((logits.argmax(-1) == batch["action"]).sum())
        total += int(batch["action"].shape[0])

        for name in nw.PER_SEAT_HEAD_TARGETS + nw.GLOBAL_HEAD_TARGETS:
            target, prediction = batch[name], out[name]
            if name in nw.PER_SEAT_HEAD_TARGETS:
                mask = batch["seat_valid"]
                mask_name = training.MASKED_TARGETS.get(name)
                if mask_name is not None:
                    mask = mask * batch[mask_name]
            else:
                mask = torch.ones_like(target)
            sq_err[name] = sq_err.get(name, 0.0) + float(
                (mask * (prediction - target) ** 2).sum()
            )
            sums[name] = sums.get(name, 0.0) + float((mask * target).sum())
            sq_sums[name] = sq_sums.get(name, 0.0) + float((mask * target**2).sum())
            weights[name] = weights.get(name, 0.0) + float(mask.sum())

    if total == 0:
        raise ValueError("nothing to evaluate: the held-out set produced no samples")
    metrics = {"policy_top1": agree / total, "eval_samples": float(total)}
    for name, count in weights.items():
        if count <= 0:
            metrics[f"r2_{name}"] = float("nan")
            continue
        variance = sq_sums[name] / count - (sums[name] / count) ** 2
        mse = sq_err[name] / count
        metrics[f"r2_{name}"] = 1.0 - mse / variance if variance > 0 else float("nan")
    return metrics


def greedy_policy(net: nw.WelcomeToNet, device: torch.device):
    """Play the policy head's best legal action.  No search, no sampling."""

    @torch.no_grad()
    def policy(state: GameState) -> int:
        net.eval()
        arrays = enc.encode_state(state)
        tensors = [torch.as_tensor(a).unsqueeze(0).float().to(device) for a in arrays]
        logits = net(*tensors)["policy_logits"][0]
        legal = torch.as_tensor(state.legal_mask()).to(device)
        return int(logits.masked_fill(legal <= 0, -1e9).argmax())

    return policy


def paired_score_gap(
    net: nw.WelcomeToNet,
    device: torch.device,
    games: int = 60,
    seed: int = 9_000,
    advanced: bool = True,
) -> dict[str, float]:
    """Replace **one seat** with the net and measure what that seat scores.

    A **controlled** substitution, and the control matters more than it looks.
    The obvious design -- one game with the net in every seat against one with
    GreedyBot in every seat -- changes the whole table at once, and the two arms
    then differ in something other than the policy under test: how *correlated*
    the seats are.  Every seat of an all-net table sees the same stacks and runs
    the same deterministic argmax, so those sheets converge on each other
    (measured: 0.34 mean divergence against greedy's 0.80).  Correlated sheets
    complete plans on the same turn, so they share first-place plan values
    instead of racing for them, and they tie on the temp-agency rank.  Those are
    scoring rules, worth 6-14 and 7/4/1 points, moving for a reason that has
    nothing to do with placement skill.

    So: the net plays seat ``k``, GreedyBots play the rest; the baseline arm is
    the same game with a GreedyBot in seat ``k`` and **the same RNG streams**
    everywhere else.  The evaluated seat rotates, and the statistic is the mean
    of per-game deltas at that seat.

    Pairing on the seed is what makes 60 games say anything: per-game score
    variance is tens of points, so an unpaired comparison cannot resolve the
    2-point difference the gate is about.  The counterfactual is not perfectly
    clean -- the substituted seat changes when the game ends and who wins which
    plan, so the opponents' games are not bit-identical -- but the deck, the
    plans and the opponents' policies are shared, which removes most of it.
    """
    counts = seat_counts(games)
    policy = greedy_policy(net, device)
    deltas: list[float] = []
    net_scores: list[float] = []
    bot_scores: list[float] = []

    def play(config: GameConfig, game_seed: int, net_seat: Optional[int]) -> list[int]:
        bots = {
            p: GreedyBot(random.Random(game_seed * 100 + p))
            for p in range(config.players)
        }
        state = GameState.new(seed=game_seed, config=config)
        while not state.is_terminal:
            actor = state.actor
            action = policy(state) if actor == net_seat else bots[actor].act(state)
            state.apply(action)
        return state.scores()

    for i, players in enumerate(counts):
        config = GameConfig(players=players, advanced=advanced)
        game_seed = seed + i
        evaluated = i % players  # rotate, so no seat's luck is the whole result

        with_net = play(config, game_seed, evaluated)[evaluated]
        without = play(config, game_seed, None)[evaluated]
        net_scores.append(float(with_net))
        bot_scores.append(float(without))
        deltas.append(float(with_net - without))

    n = len(deltas)
    mean = sum(deltas) / n
    variance = sum((d - mean) ** 2 for d in deltas) / max(n - 1, 1)
    return {
        "net_score": sum(net_scores) / n,
        "greedy_score": sum(bot_scores) / n,
        "score_gap": mean,
        "score_gap_stderr": math.sqrt(variance / n),
        "paired_games": float(n),
    }


# ──────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class TrainConfig:
    games: int = 2000
    val_fraction: float = 0.1
    epochs: int = 4
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    seed: int = 0
    paired_games: int = 60


def gate(metrics: dict[str, float]) -> dict[str, bool]:
    """The three S0 conditions, evaluated.  See the module docstring."""
    return {
        "policy_agreement": metrics.get("policy_top1", 0.0) >= 0.60,
        "score_within_2": abs(metrics.get("score_gap", 99.0)) <= 2.0,
        "permits_beats_mean": metrics.get("r2_permits", -1.0) > 0.0,
    }


def train(
    config: Optional[TrainConfig] = None,
    net_config: Optional[nw.NetConfig] = None,
    device: Optional[str] = None,
    out_dir: Optional[Path] = None,
    log: bool = True,
) -> tuple[nw.WelcomeToNet, dict[str, float]]:
    config = config or TrainConfig()
    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)

    started = time.time()
    corpus = build_corpus(config.games, seed=config.seed)
    rng.shuffle(corpus)
    split = max(1, int(len(corpus) * config.val_fraction))
    val, train_set = corpus[:split], corpus[split:]
    if log:
        print(
            f"corpus {len(corpus)} games "
            f"({len(train_set)} train / {len(val)} val) "
            f"in {time.time() - started:.1f}s"
        )

    net = nw.WelcomeToNet(net_config).to(torch_device)
    optimiser = torch.optim.AdamW(
        net.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    if log:
        print(f"{nw.parameter_count(net):,} parameters on {torch_device}")

    for epoch in range(config.epochs):
        net.train()
        running: dict[str, float] = {}
        steps = 0
        for raw in iter_batches(train_set, config.batch_size, rng):
            batch = nw.to_tensors(raw, torch_device)
            out = net(
                batch["sheet_planes"],
                batch["sheet_scalars"],
                batch["viewer_plane"],
                batch["global_scalars"],
            )
            total, parts = nw.losses(out, batch)
            optimiser.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
            optimiser.step()
            steps += 1
            running["total"] = running.get("total", 0.0) + float(total.detach())
            for name, part in parts.items():
                running[name] = running.get(name, 0.0) + float(part.detach())
        if log:
            mean = {k: v / max(steps, 1) for k, v in running.items()}
            print(
                f"epoch {epoch}  steps {steps}  "
                f"loss {mean['total']:.4f}  policy {mean['policy']:.4f}  "
                f"score {mean['score']:.4f}  permits {mean['permits']:.4f}"
            )

    metrics = evaluate(net, val, torch_device)
    metrics.update(paired_score_gap(net, torch_device, games=config.paired_games))
    metrics["parameters"] = float(nw.parameter_count(net))
    metrics["train_seconds"] = time.time() - started
    passed = gate(metrics)
    metrics.update({f"gate_{k}": float(v) for k, v in passed.items()})

    if log:
        print(
            f"policy top-1 {metrics['policy_top1']:.3f}  "
            f"score {metrics['net_score']:.2f} vs greedy "
            f"{metrics['greedy_score']:.2f} ({metrics['score_gap']:+.2f})  "
            f"R2 permits {metrics['r2_permits']:.3f}  "
            f"R2 score {metrics['r2_score']:.3f}"
        )
        print("gate: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in passed.items()))

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": net.state_dict(),
                "net_config": asdict(net.config),
                "train_config": asdict(config),
                "metrics": metrics,
            },
            out_dir / "s0.pt",
        )
        (out_dir / "s0_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
    return net, metrics


def load(path: str | Path, device: str = "cpu") -> nw.WelcomeToNet:
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    net = nw.WelcomeToNet(nw.NetConfig(**blob["net_config"]))
    net.load_state_dict(blob["state_dict"])
    return net.to(device)


def main(argv: Optional[Sequence[str]] = None) -> int:
    # A short ASCII description, not __doc__: argparse prints help to a cp1252
    # console on Windows, which cannot encode the box-drawing rules above.
    parser = argparse.ArgumentParser(
        description="S0 bootstrap: clone GreedyBot into the network."
    )
    # `slots=True` means the class attributes are slot descriptors, not the
    # defaults -- read them off an instance.
    default = TrainConfig()
    parser.add_argument("--games", type=int, default=default.games)
    parser.add_argument("--epochs", type=int, default=default.epochs)
    parser.add_argument("--batch-size", type=int, default=default.batch_size)
    parser.add_argument("--lr", type=float, default=default.lr)
    parser.add_argument("--seed", type=int, default=default.seed)
    parser.add_argument("--paired-games", type=int, default=default.paired_games)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="runs/welcome_to_s0")
    args = parser.parse_args(argv)

    _, metrics = train(
        TrainConfig(
            games=args.games,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            paired_games=args.paired_games,
        ),
        device=args.device,
        out_dir=Path(args.out),
    )
    return 0 if all(gate(metrics).values()) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
