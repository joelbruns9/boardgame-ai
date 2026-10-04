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
_KEEP = ("game_seed", "players", "turn", "candidates", "afterstates", "scores", "ranks", "blend", "steered")


#: Bumped when the meaning of a paired root changes (selection, labels, keys).
PAIRS_FORMAT = 2


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recipe_digest(settings: dict) -> str:
    """Everything a paired root's meaning depends on (review 2026-10-03 F5):
    checkpoint contents, source corpus, engine/encoder/target identity,
    continuation policy, seeds, candidate selection and steering."""
    import hashlib

    import welcome_to_rust as wr

    identity = {
        **settings,
        "pairs_format": PAIRS_FORMAT,
        "encoder_abi": enc.ENCODER_ABI_VERSION,
        "table_signature": int(wr.table_signature()),
        "shard_version": int(wr.TRAINING_SHARD_VERSION),
        "continuation": "argmax policy, every seat; pool rule for steered learner",
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _stratified_roots(candidates: list, roots: int, seed: int) -> list:
    """Cap ``candidates`` at ``roots`` in proportion to player count x turn
    bucket, each stratum drawn in seeded random order (review F1: truncating a
    seed-sorted list cut the four-player seed block entirely)."""
    rng = random.Random(seed ^ 0x53545241)
    strata: dict[tuple[int, int], list] = {}
    for root in candidates:
        bucket = next(i for i, (lo, hi) in enumerate(sibling_probe.TURN_BUCKETS) if lo <= root.turn <= hi)
        strata.setdefault((root.players, bucket), []).append(root)
    if len(candidates) <= roots:
        return sorted(candidates, key=lambda r: (r.game_seed, r.turn))
    for group in strata.values():
        rng.shuffle(group)
    # largest-remainder quota per stratum, proportional to its size
    exact = {key: roots * len(group) / len(candidates) for key, group in strata.items()}
    quota = {key: int(value) for key, value in exact.items()}
    for key in sorted(exact, key=lambda k: -(exact[k] % 1.0))[: roots - sum(quota.values())]:
        quota[key] += 1
    chosen = [root for key, group in strata.items() for root in group[: quota[key]]]
    return sorted(chosen, key=lambda r: (r.game_seed, r.turn))


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
    plan_aware_share: float = 0.0,
) -> Path:
    """Write ``iteration_dir/pairs.pt``; resumable through per-chunk saves.

    An existing ``pairs.pt`` is reused only if its recipe digest matches
    (review F5); otherwise it is refused rather than silently trained on.
    """
    from games.welcome_to import s2_promotion, s2_train

    iteration_dir = Path(iteration_dir)
    out = iteration_dir / PAIRS_FILE
    corpus = iteration_dir / "trajectories.jsonl"
    settings = {
        "checkpoint_sha256": _file_sha256(Path(checkpoint)),
        "corpus": str(corpus.resolve()),
        "roots": roots,
        "alternatives": alternatives,
        "futures": futures,
        "seed": seed,
        "simulations": simulations,
        "plan_aware_share": plan_aware_share,
    }
    recipe = recipe_digest(settings)
    if out.exists():
        existing = torch.load(out, weights_only=False)
        if existing.get("recipe") != recipe:
            raise ValueError(f"{out} was built with a different recipe; delete it to rebuild")
        return out
    work = iteration_dir / "pairs_work"
    work.mkdir(exist_ok=True)
    manifest = work / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")).get("recipe") != recipe:
            raise ValueError(f"{work} was built with a different recipe; delete it to rebuild")
    else:
        manifest.write_text(json.dumps({**settings, "recipe": recipe}, indent=2), encoding="utf-8")

    net, _ = s2_train.load_training_checkpoint(checkpoint, device)
    net.eval()
    cfg = s2_promotion.gate_search_config(simulations)
    packed = rust_search.PackedNetEvaluator(net, torch.device(device), cfg)
    # Ordinary games only: no curriculum restart, no assisted game, no forced
    # deal (review F4) -- the population A is meant to describe.
    games = sorted(
        (
            game
            for game in self_play.read_trajectories(corpus)
            if game.restart is None and game.assist_through is None and game.plan_ids is None
        ),
        key=lambda game: game.seed,
    )
    # Draw source games stratified by player count, in seeded random order,
    # with headroom so the root quota is met even when positions lack siblings.
    rng = random.Random(seed)
    by_players: dict[int, list] = {}
    for game in games:
        by_players.setdefault(game.players, []).append(game)
    wanted_games = min(len(games), math.ceil(roots / 2.0 * 1.5))
    wanted = []
    for players, group in sorted(by_players.items()):
        rng.shuffle(group)
        wanted += group[: math.ceil(wanted_games * len(group) / max(1, len(games)))]
    wanted.sort(key=lambda game: game.seed)
    candidates = sibling_probe.select_roots(wanted, packed, max_candidates=1 + alternatives, seed=seed)
    selected = _stratified_roots(candidates, roots, seed)
    if plan_aware_share > 0.0:
        # Pool-rule playouts (owner, 2026-10-03): for a share of the roots where
        # the learner holds a live pool plan, its continuation follows the pool
        # rule. A helper: its share follows the helper schedule.
        from games.welcome_to import pool_rescue
        from games.welcome_to import snapshot as snap

        steer_rng = random.Random(seed ^ 0x504F4F4C)
        for root in selected:
            state = snap.from_snapshot(root.snapshot)
            if pool_rescue.needed_streets(state, 0)[2] and steer_rng.random() < plan_aware_share:
                root.steered = True
    steer = sibling_probe._POOL_STEER
    steer.calls = steer.overrides = 0
    started = time.perf_counter()
    sibling_probe.rollout_roots(selected, packed, cfg, futures, seed, checkpoint_dir=work, recipe=recipe)
    players_count: dict[str, int] = {}
    for root in selected:
        players_count[str(root.players)] = players_count.get(str(root.players), 0) + 1
    payload = {
        "settings": settings,
        "recipe": recipe,
        "encoder_abi": enc.ENCODER_ABI_VERSION,
        "futures": futures,
        "seconds": time.perf_counter() - started,
        "roots": [{name: getattr(root, name) for name in _KEEP} for root in selected],
        "requested_roots": roots,
        "realized_roots": len(selected),
        "candidate_roots": len(candidates),
        "roots_by_players": players_count,
        "steered_roots": sum(root.steered for root in selected),
        "steer_calls": steer.calls,
        "steer_overrides": steer.overrides,
        "execution_audit": execution_audit(selected),
    }
    temporary = out.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(out)
    return out


def execution_audit(roots: Sequence) -> Optional[dict]:
    """Steered versus unsteered continuations on the same roots, candidates
    and futures (review 2026-10-03 Q1): how much the steering is worth, and
    whether it changes which candidate looks best."""
    audited = [r for r in roots if getattr(r, "steered", False) and r.unsteered_scores is not None]
    if not audited:
        return None
    gaps, pool_steered, pool_plain, changes, played_gaps = [], [], [], 0, []
    for root in audited:
        steered = root.scores[:, :, 0].mean(axis=1)
        plain = root.unsteered_scores[:, :, 0].mean(axis=1)
        gaps.append(float((steered - plain).mean()))
        played_gaps.append(float(steered[0] - plain[0]))
        changes += int(np.argmax(steered) != np.argmax(plain))
        pool_steered.append(float(root.pool_done.mean()))
        pool_plain.append(float(root.unsteered_pool_done.mean()))
    return {
        "roots": len(audited),
        "mean_score_gap": float(np.mean(gaps)),
        "played_score_gap": float(np.mean(played_gaps)),
        "preferred_candidate_changed": changes / len(audited),
        "pool_plan_done_steered": float(np.mean(pool_steered)),
        "pool_plan_done_unsteered": float(np.mean(pool_plain)),
    }


def load_window(
    replay_root: Path, through_iteration: int, window: int
) -> list[dict]:
    """Paired roots from the last ``window`` iterations that have them."""
    roots: list[dict] = []
    for iteration in range(max(1, through_iteration - window + 1), through_iteration + 1):
        path = Path(replay_root) / f"iter_{iteration:04d}" / PAIRS_FILE
        if path.exists():
            payload = torch.load(path, weights_only=False)
            if payload.get("encoder_abi") != enc.ENCODER_ABI_VERSION:
                raise ValueError(
                    f"{path} holds encoder ABI {payload.get('encoder_abi')} afterstates; "
                    f"this build is ABI {enc.ENCODER_ABI_VERSION}"
                )
            roots.extend(payload["roots"])
    return roots


def split(
    roots: Sequence[dict],
    val_fraction: float,
    salt: str,
    *,
    validation_families: Optional[set] = None,
    training_families: Optional[set] = None,
) -> tuple[list[dict], list[dict]]:
    """Same family rule as the replay split: a root's game seed is its family.

    When the ordinary split's family sides are given, a root follows its
    family's side -- including a family the small-corpus fallback moved
    (review F6); families absent from the replay window use the hash.
    """
    from games.welcome_to import s2_train

    train, val = [], []
    for root in roots:
        family = root["game_seed"]
        if validation_families is not None and family in validation_families:
            held = True
        elif training_families is not None and family in training_families:
            held = False
        else:
            held, _ = s2_train.stable_family_is_validation(family, val_fraction, salt)
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

    import welcome_to_rust as wr

    payload = torch.load(dataset, weights_only=False)
    test = sibling_probe._split(payload["roots"], split_seed)["test"]
    # Re-encode every afterstate with the CURRENT encoder: the dataset's stored
    # encodings belong to the encoder that collected it, and a benchmark must
    # survive encoder changes (v4 broke the v3 layout).
    for root in test:
        base = wr.RustGameState.from_snapshot(root["snapshot"])
        root["afterstates"] = [sibling_probe._encode(base.step_macro(c)) for c in root["candidates"]]
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
