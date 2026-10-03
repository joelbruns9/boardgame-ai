"""
Paired placement targets (review idea A, built 2026-10-02).

WHY
───
The sibling probe (``sibling_probe.py``, 2026-10-02) measured that telling two
placements of the same card apart takes ~50 shared futures: one game outcome
carries ~1/50 of the signal, so ordinary self-play learns it only slowly. With
reliable labels, checkpoint 35 picked the better box 67% of the time on clear
pairs and its value choice left 2.07 points of regret per decision against a
1.02-point ceiling. These targets supply that signal directly.

WHAT A PAIRED ROOT IS
─────────────────────
From an iteration's ordinary games (not curriculum or assisted ones): a learner
card+write decision, the played box plus ``alternatives`` other boxes for the
same stack slot and temp delta (one the policy's favourite, the rest uniform),
each played to the **real end of the game** under ``futures`` redeterminized
futures shared by every candidate, every seat on the generating checkpoint's
argmax policy. Labels are outcomes only -- final scores and the tiebreak-aware
rank distribution -- never a heuristic.

They are ``Q^π`` labels for that checkpoint's continuation policy, so they age
as the policy improves: the plan is a large share early, tapered on the fixed
benchmark (``benchmark``), with a small trickle kept.

HOW THEY TRAIN
──────────────
``paired_loss``: per root, the absolute anchors (per-seat score MSE, learner
rank cross-entropy, weighted by ``rank_weight``) plus ``pair_weight`` times the
paired per-seat score difference. It trains the heads search reads -- the
score and rank heads -- not an auxiliary head. Roots split by game family with
the same rule as the replay (``s2_train.split_family``).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from games.welcome_to import encoder as enc
from games.welcome_to import network as nw
from games.welcome_to import rust_search, self_play, sibling_probe, training

PAIRS_FILE = "pairs.pt"
_KEEP = ("game_seed", "players", "turn", "candidates", "afterstates", "scores", "ranks", "blend")


def build(
    iteration_dir: Path,
    checkpoint: Path,
    *,
    roots: int,
    alternatives: int = 2,
    futures: int = 48,
    seed: int = 0,
    simulations: int = 200,
    device: str = "cuda",
) -> Path:
    """Write ``iteration_dir/pairs.pt``; resumable through per-chunk saves."""
    from games.welcome_to import s2_promotion, s2_train

    iteration_dir = Path(iteration_dir)
    out = iteration_dir / PAIRS_FILE
    if out.exists():
        return out
    settings = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "roots": roots,
        "alternatives": alternatives,
        "futures": futures,
        "seed": seed,
    }
    work = iteration_dir / "pairs_work"
    work.mkdir(exist_ok=True)
    manifest = work / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != settings:
            raise ValueError(f"{work} was built with different settings; delete it to rebuild")
    else:
        manifest.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    net, _ = s2_train.load_training_checkpoint(checkpoint, device)
    net.eval()
    cfg = s2_promotion.gate_search_config(simulations)
    packed = rust_search.PackedNetEvaluator(net, torch.device(device), cfg)
    games = sorted(
        (
            game
            for game in self_play.read_trajectories(iteration_dir / "trajectories.jsonl")
            if game.restart is None and game.assist_through is None
        ),
        key=lambda game: game.seed,
    )
    # ~2.4 roots per game (one per turn bucket, when eligible): draw enough
    # games, in a seeded order, to reach the requested count.
    random.Random(seed).shuffle(games)
    wanted = games[: min(len(games), math.ceil(roots / 2.0))]
    wanted.sort(key=lambda game: game.seed)
    selected = sibling_probe.select_roots(
        wanted, packed, max_candidates=1 + alternatives, seed=seed
    )[:roots]
    started = time.perf_counter()
    sibling_probe.rollout_roots(
        selected, packed, cfg, futures, seed, checkpoint_dir=work
    )
    payload = {
        "settings": settings,
        "futures": futures,
        "seconds": time.perf_counter() - started,
        "roots": [{name: getattr(root, name) for name in _KEEP} for root in selected],
    }
    temporary = out.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(out)
    return out


def load_window(
    replay_root: Path, through_iteration: int, window: int
) -> list[dict]:
    """Paired roots from the last ``window`` iterations that have them."""
    roots: list[dict] = []
    for iteration in range(max(1, through_iteration - window + 1), through_iteration + 1):
        path = Path(replay_root) / f"iter_{iteration:04d}" / PAIRS_FILE
        if path.exists():
            roots.extend(torch.load(path, weights_only=False)["roots"])
    return roots


def split(roots: Sequence[dict], val_fraction: float, salt: str) -> tuple[list[dict], list[dict]]:
    """Same family rule as the replay split: a root's game seed is its family."""
    from games.welcome_to import s2_train

    train, val = [], []
    for root in roots:
        held, _ = s2_train.stable_family_is_validation(root["game_seed"], val_fraction, salt)
        (val if held else train).append(root)
    return train, val


def paired_loss(
    net: nw.WelcomeToNet,
    roots: Sequence[dict],
    device,
    *,
    pair_weight: float,
    rank_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One forward over every candidate afterstate of ``roots``.

    Per root: per-seat score MSE to the mean final score (÷80), learner rank
    cross-entropy to the mean rank distribution, and the paired per-seat
    score difference, each normalized within the root so a root with more
    candidates does not count more.
    """
    encodings = [a for root in roots for a in root["afterstates"]]
    out = net(*[torch.as_tensor(np.stack(col)).float().to(device) for col in zip(*encodings)])
    absolute, rank, pair = [], [], []
    offset = 0
    for root in roots:
        c, seats = len(root["candidates"]), root["players"]
        rows = slice(offset, offset + c)
        offset += c
        target = torch.as_tensor(
            root["scores"].mean(axis=1) / training.SCORE_SCALE, device=device
        )
        predicted = out["score"][rows, :seats]
        absolute.append(((predicted - target) ** 2).mean())
        mask = torch.zeros((c, training.MAX_RANKS), device=device)
        mask[:, :seats] = 1.0
        log_p = nw.masked_log_softmax(out["rank_logits"][rows], mask)
        target_rank = torch.as_tensor(root["ranks"].mean(axis=1), device=device)
        rank.append(-(target_rank * log_p * mask).sum(-1).mean())
        dp = predicted[:, None, :] - predicted[None, :, :]
        dt = target[:, None, :] - target[None, :, :]
        pair.append(((dp - dt) ** 2).sum() / max(1, c * (c - 1) * seats))
    absolute_t = torch.stack(absolute).mean()
    rank_t = torch.stack(rank).mean()
    pair_t = torch.stack(pair).mean()
    total = absolute_t + rank_weight * rank_t + pair_weight * pair_t
    return total, {
        "paired_score_abs": float(absolute_t.detach()),
        "paired_rank": float(rank_t.detach()),
        "paired_diff": float(pair_t.detach()),
    }


@torch.no_grad()
def decision_metrics(net: nw.WelcomeToNet, roots: Sequence[dict], device, simulations: int = 200, fit: int = 0) -> dict:
    """``sibling_probe.score_decisions`` over ``roots``: truth = futures ``fit:``."""
    from games.welcome_to import s2_promotion

    if not roots:
        return {}
    cfg = s2_promotion.gate_search_config(simulations)
    was_training = net.training
    net.eval()
    out: dict[str, float] = {"roots": float(len(roots))}
    # Both decision rules: the search's blended leaf value, and the predicted
    # score margin (which ranked placements better in the probe).
    for choose_by in ("blend", "margin"):
        values = [sibling_probe.model_values(net, root, cfg, device, choose_by) for root in roots]
        result = sibling_probe.score_decisions(values, list(roots), fit)
        prefix = "" if choose_by == "blend" else "margin_"
        out.update(
            {
                f"{prefix}regret": result["regret"]["mean"],
                f"{prefix}score_regret": result["score_regret"]["mean"],
                f"{prefix}pair_sign_accuracy": result["pair_sign_accuracy"]["mean"],
                f"{prefix}resolved_pair_sign_accuracy": result["resolved_pair_sign_accuracy"]["mean"],
            }
        )
        out["regret_played"] = result["regret_played"]["mean"]
        out["score_regret_played"] = result["score_regret_played"]["mean"]
    if was_training:
        net.train()
    return out


def benchmark(checkpoint: Path, dataset: Path, device: str = "cuda", split_seed: int = 0) -> dict:
    """Score a checkpoint on a fixed sibling-probe dataset's held-out roots,
    with that dataset's independent eval futures as truth."""
    from games.welcome_to import s2_train

    payload = torch.load(dataset, weights_only=False)
    test = sibling_probe._split(payload["roots"], split_seed)["test"]
    net, _ = s2_train.load_training_checkpoint(checkpoint, device)
    return decision_metrics(net, test, torch.device(device), payload["simulations"], fit=payload["fit_futures"])


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--iteration-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--roots", type=int, default=300)
    parser.add_argument("--alternatives", type=int, default=2)
    parser.add_argument("--futures", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    path = build(
        Path(args.iteration_dir), Path(args.checkpoint), roots=args.roots,
        alternatives=args.alternatives, futures=args.futures, seed=args.seed, device=args.device,
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
