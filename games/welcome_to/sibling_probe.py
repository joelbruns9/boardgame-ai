"""
Sibling probe (review 2026-10-02 §3): can the current network learn the
outcome difference between two placements of the same card?

The diagnosis so far: the value head is indifferent between a clean and a
messy placement of one card (HYGIENE_AND_PLANS_REVIEW_REQUEST.md §6.7), more
data barely moves it (§6.8), and assisted clean-state data did not help (§6.6).
That fits two different causes -- the training data never shows the contrast,
or this network cannot carry it -- and they call for different fixes (paired
outcome targets vs a new representation). This probe separates them on a fixed
dataset, before another long self-play run.

DATASET (``collect``)
─────────────────────
* **Fresh source games** played by the frozen checkpoint at ordinary
  generation settings, on a seed range no training run used.
* **Roots:** the learner's card+write decisions, up to one per turn bucket
  (≤8, 9–16, ≥17) per game, chosen at random -- **not** by the cleanup
  heuristic or by value disagreement.
* **Candidates:** the move played plus other boxes for the *same stack slot and
  the same temp delta* (box-only, the cleanest contrast): the policy's top
  alternatives and uniform ones, at most ``--candidates``.
* **Futures:** ``fit + eval`` redeterminizations of the root. Within one future
  every candidate starts from the same reshuffled deck, and every seat then
  plays the checkpoint's argmax policy **to the actual end of the game**. Fit
  futures label training; eval futures are independent and only score
  decisions.
* Per (root, candidate, future): every seat's final score, the tiebreak-aware
  rank distribution, and the learner's terminal blend (the search's own leaf
  scale).

ARMS (``probe``) -- all on the same dataset, split by source game
──────────────────────────────────────────────────────────────────
1. ``checkpoint``: the network as it is.
2. ``readout``: sheet encoder and trunk frozen, the value readouts (per-seat
   and global heads) retrained on absolute + paired targets.
3. ``finetune``: the whole network, absolute + paired targets, with ordinary
   replay rows mixed in so the policy and other heads do not drift.
4. ``shuffled``: as 3, with candidate labels permuted within each root -- the
   control a learner should not beat.

Scored on held-out roots with eval futures: paired sign accuracy, rank
correlation within a root, and **decision regret** (empirical value of the best
candidate minus that of the model's choice). The played move and a uniform pick
are reported as references.

Reading it: if ``readout`` closes the gap, the features exist and only the
targets were missing (build paired outcome targets, review idea A). If only
``finetune`` does, the representation had to adapt but no new architecture is
needed. If neither does while labels are stable, the network cannot carry the
contrast and encoder/architecture work comes first (idea E).

    python -m games.welcome_to.sibling_probe collect --checkpoint <ckpt> --out <dir>
    python -m games.welcome_to.sibling_probe probe --checkpoint <ckpt> --out <dir>
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from games.welcome_to import encoder as enc
from games.welcome_to import macro_codec as mc
from games.welcome_to import mcts
from games.welcome_to import network as nw
from games.welcome_to import rust_search, self_play, snapshot, training

TURN_BUCKETS: tuple[tuple[int, int], ...] = ((0, 8), (9, 16), (17, 99))


def _is_write(macro: int) -> bool:
    return mc.M_WRITE <= macro < mc.M_REFUSE


def _slot_delta(macro: int) -> tuple[int, int]:
    slot, delta, _x, _y = mc.decode_macro_write(macro)
    return slot, delta


def _encode(rust_state, viewer: int = 0) -> tuple[np.ndarray, ...]:
    raw = rust_state.encode_state(viewer)
    shapes = (
        enc.SHEET_PLANES_SHAPE,
        (enc.MAX_SEATS, enc.NUM_SHEET_SCALAR),
        enc.VIEWER_PLANE_SHAPE,
        (enc.NUM_GLOBAL_SCALAR,),
    )
    return tuple(
        np.frombuffer(bytes(buf), dtype="<f4").reshape(shape).copy()
        for buf, shape in zip(raw, shapes)
    )


@dataclass
class Root:
    game_seed: int
    players: int
    turn: int
    played: int
    candidates: list[int]
    snapshot: dict
    #: outcome arrays, shape (candidates, futures, ...)
    scores: Optional[np.ndarray] = None       # (C, F, players)
    ranks: Optional[np.ndarray] = None        # (C, F, MAX_RANKS) learner rank distribution
    seat_ranks: Optional[np.ndarray] = None   # (C, F, players, MAX_RANKS)
    blend: Optional[np.ndarray] = None        # (C, F) learner terminal blend
    afterstates: list = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────
# Collection
# ──────────────────────────────────────────────────────────────────────────
def select_roots(
    trajectories: Sequence[self_play.SelfPlayTrajectory],
    packed: rust_search.PackedNetEvaluator,
    *,
    max_candidates: int,
    seed: int,
) -> list[Root]:
    import welcome_to_rust as wr

    rng = random.Random(seed)
    roots: list[Root] = []
    for trajectory in trajectories:
        state = wr.RustGameState(
            trajectory.engine_seed,
            players=trajectory.players,
            advanced=True,
            expert=False,
            solo_rules=False,
        )
        eligible: dict[int, list[tuple[int, object, int]]] = {}
        for decision, action in enumerate(trajectory.actions):
            if state.actor == 0 and state.is_macro_root and _is_write(action):
                bucket = next(
                    i for i, (lo, hi) in enumerate(TURN_BUCKETS) if lo <= state.turn <= hi
                )
                eligible.setdefault(bucket, []).append((decision, state.snapshot(), action))
            state.apply_macro(action)
        for bucket in sorted(eligible):
            _decision, snap, action = rng.choice(eligible[bucket])
            root = wr.RustGameState.from_snapshot(snap)
            slot, delta = _slot_delta(action)
            siblings = [
                m
                for m in root.legal_macros()
                if _is_write(m) and m != action and _slot_delta(m) == (slot, delta)
            ]
            if not siblings:
                continue
            policies, _legals = packed.policy_states([root], [trajectory.players])
            prior = policies[0]
            by_prior = sorted(siblings, key=lambda m: (-float(prior[m]), m))
            budget = max_candidates - 1
            top = by_prior[: max(1, budget // 2)]
            rest = [m for m in siblings if m not in top]
            uniform = rng.sample(rest, min(len(rest), budget - len(top)))
            roots.append(
                Root(
                    game_seed=trajectory.seed,
                    players=trajectory.players,
                    turn=int(root.turn),
                    played=action,
                    candidates=[action, *top, *uniform],
                    snapshot=snap,
                )
            )
    return roots


def _finish_all(states: list, players: list[int], packed, chunk: int = 4096) -> None:
    """Advance every state to the end with every seat on argmax policy."""
    live = list(range(len(states)))
    while live:
        still = []
        for start in range(0, len(live), chunk):
            ids = [i for i in live[start : start + chunk] if not states[i].is_terminal]
            if not ids:
                continue
            policies, legals = packed.policy_states(
                [states[i] for i in ids], [players[i] for i in ids]
            )
            for i, policy, legal in zip(ids, policies, legals):
                choice = max(legal, key=lambda m: (float(policy[m]), -m))
                states[i].apply_macro(int(choice))
                if not states[i].is_terminal:
                    still.append(i)
        live = still


def rollout_roots(
    roots: list[Root], packed, cfg: mcts.SearchConfig, futures: int, seed: int
) -> None:
    """Fill each root's outcome arrays: candidates x futures, paired draws."""
    import welcome_to_rust as wr

    states, players, index = [], [], []
    for r_id, root in enumerate(roots):
        base = wr.RustGameState.from_snapshot(root.snapshot)
        root.afterstates = [_encode(base.step_macro(c)) for c in root.candidates]
        for f in range(futures):
            drawn, _ = base.redeterminize((seed * 1_000_003 + r_id * 1_009 + f) & ((1 << 64) - 1))
            for c_id, candidate in enumerate(root.candidates):
                states.append(drawn.step_macro(candidate))
                players.append(root.players)
                index.append((r_id, c_id, f))
    _finish_all(states, players, packed)
    for root in roots:
        n = len(root.candidates)
        root.scores = np.zeros((n, futures, root.players), dtype=np.float32)
        root.ranks = np.zeros((n, futures, training.MAX_RANKS), dtype=np.float32)
        root.seat_ranks = np.zeros((n, futures, root.players, training.MAX_RANKS), dtype=np.float32)
        root.blend = np.zeros((n, futures), dtype=np.float32)
    for state, (r_id, c_id, f) in zip(states, index):
        root = roots[r_id]
        final = snapshot.from_snapshot(state.snapshot())
        root.scores[c_id, f] = final.scores()
        dists = training.rank_distributions(final)
        root.seat_ranks[c_id, f] = dists
        root.ranks[c_id, f] = dists[0]
        root.blend[c_id, f] = mcts.terminal_value(final, 0, cfg)


def collect(args) -> None:
    from games.welcome_to import s2_promotion, s2_train

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net, _ = s2_train.load_training_checkpoint(args.checkpoint, args.device)
    net.eval()
    cfg = s2_promotion.gate_search_config(args.simulations)
    started = time.perf_counter()
    games, _ = self_play.generate(
        net,
        config=self_play.SelfPlayConfig(
            games=args.games,
            inflight=min(256, args.games),
            max_batch=min(256, args.games),
            seed=args.seed,
        ),
        search_config=self_play.default_search_config(args.simulations),
        device=args.device,
    )
    generation = time.perf_counter() - started
    packed = rust_search.PackedNetEvaluator(net, torch.device(args.device), cfg)
    roots = select_roots(games, packed, max_candidates=args.candidates, seed=args.seed)
    started = time.perf_counter()
    rollout_roots(roots, packed, cfg, args.fit_futures + args.eval_futures, args.seed)
    rollouts = time.perf_counter() - started
    terminals = sum(len(r.candidates) for r in roots) * (args.fit_futures + args.eval_futures)
    torch.save(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "fit_futures": args.fit_futures,
            "eval_futures": args.eval_futures,
            "simulations": args.simulations,
            "seed": args.seed,
            "roots": [r.__dict__ for r in roots],
        },
        out / "dataset.pt",
    )
    info = {
        "games": len(games),
        "roots": len(roots),
        "candidates_per_root": statistics.fmean(len(r.candidates) for r in roots),
        "terminals": terminals,
        "generation_seconds": generation,
        "rollout_seconds": rollouts,
        "ms_per_terminal": 1000 * rollouts / max(terminals, 1),
    }
    (out / "collect.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


# ──────────────────────────────────────────────────────────────────────────
# Probe
# ──────────────────────────────────────────────────────────────────────────
def _split(roots: list[dict], seed: int) -> dict[str, list[dict]]:
    """By source game: every candidate and future of a game on one side."""
    games = sorted({r["game_seed"] for r in roots})
    random.Random(seed).shuffle(games)
    n = len(games)
    side = {g: ("train" if i < 0.7 * n else "val" if i < 0.85 * n else "test") for i, g in enumerate(games)}
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in roots:
        out[side[r["game_seed"]]].append(r)
    return out


def _stack(encodings: list[tuple[np.ndarray, ...]], device) -> list[torch.Tensor]:
    return [torch.as_tensor(np.stack(col)).float().to(device) for col in zip(*encodings)]


def _rank_mask(players: Sequence[int], device) -> torch.Tensor:
    mask = torch.zeros((len(players), training.MAX_RANKS), device=device)
    for row, n in enumerate(players):
        mask[row, :n] = 1.0
    return mask


@torch.no_grad()
def model_values(net, root: dict, cfg, device) -> np.ndarray:
    """The search's leaf value for each candidate afterstate."""
    out = net(*_stack(root["afterstates"], device))
    players = [root["players"]] * len(root["candidates"])
    probs = nw.rank_probabilities(out["rank_logits"], _rank_mask(players, device)).cpu().numpy()
    scores = out["score"].cpu().numpy()
    return np.array(
        [mcts.blend_value(probs[i], scores[i], root["players"], cfg)[0] for i in range(len(players))]
    )


def score_decisions(values_by_root: list[np.ndarray], roots: list[dict], fit: int) -> dict:
    """Decision metrics on eval futures (independent of the fit labels)."""
    regrets, played_regrets, uniform_regrets, signs, spearmans = [], [], [], [], []
    resolved_signs = []
    for values, root in zip(values_by_root, roots):
        truth = root["blend"][:, fit:]                     # (C, F_eval)
        mean = truth.mean(axis=1)
        best = mean.max()
        regrets.append(best - mean[int(np.argmax(values))])
        played_regrets.append(best - mean[0])
        uniform_regrets.append(best - mean.mean())
        for a, b in itertools.combinations(range(len(values)), 2):
            diff = truth[a] - truth[b]
            empirical = diff.mean()
            if empirical == 0:
                continue
            agree = float(np.sign(values[a] - values[b]) == np.sign(empirical))
            signs.append(agree)
            se = diff.std(ddof=1) / math.sqrt(len(diff)) if len(diff) > 1 else float("inf")
            if abs(empirical) > 2 * se:
                resolved_signs.append(agree)
        if len(values) > 2 and np.std(values) > 0 and np.std(mean) > 0:
            rv = np.argsort(np.argsort(values)); rm = np.argsort(np.argsort(mean))
            spearmans.append(float(np.corrcoef(rv, rm)[0, 1]))

    def ci(xs):
        if len(xs) < 2:
            return {"mean": float("nan"), "se": float("nan"), "n": len(xs)}
        return {"mean": statistics.fmean(xs), "se": statistics.stdev(xs) / math.sqrt(len(xs)), "n": len(xs)}

    return {
        "regret": ci(regrets),
        "regret_played": ci(played_regrets),
        "regret_uniform": ci(uniform_regrets),
        "pair_sign_accuracy": ci(signs),
        "resolved_pair_sign_accuracy": ci(resolved_signs),
        "spearman_within_root": ci(spearmans),
    }


def _targets(root: dict, fit: int, shuffle: Optional[random.Random]) -> tuple[np.ndarray, np.ndarray]:
    """Per-candidate mean per-seat score (÷80, padded to 4) and learner rank
    distribution over the fit futures."""
    scores = root["scores"][:, :fit].mean(axis=1) / training.SCORE_SCALE      # (C, players)
    ranks = root["ranks"][:, :fit].mean(axis=1)                               # (C, 4)
    if shuffle is not None:
        order = list(range(len(scores)))
        shuffle.shuffle(order)
        scores, ranks = scores[order], ranks[order]
    padded = np.zeros((len(scores), enc.MAX_SEATS), dtype=np.float32)
    padded[:, : scores.shape[1]] = scores
    return padded, ranks


def sibling_loss(net, roots: list[dict], fit: int, pair_weight: float, device, shuffle=None):
    """Absolute score/rank anchors plus the paired score difference, per root."""
    total = None
    for root in roots:
        out = net(*_stack(root["afterstates"], device))
        n_seats = root["players"]
        target_scores, target_ranks = _targets(root, fit, shuffle)
        ts = torch.as_tensor(target_scores, device=device)[:, :n_seats]
        tr = torch.as_tensor(target_ranks, device=device)
        ps = out["score"][:, :n_seats]
        mask = _rank_mask([n_seats] * len(ts), device)
        log_p = nw.masked_log_softmax(out["rank_logits"], mask)
        absolute = ((ps - ts) ** 2).mean() + (-(tr * log_p * mask).sum(-1)).mean()
        # every ordered pair once: (s_a - s_b) - (t_a - t_b), all seats
        dp = ps[:, None, :] - ps[None, :, :]
        dt = ts[:, None, :] - ts[None, :, :]
        c = len(ts)
        pair = ((dp - dt) ** 2).sum() / max(1, c * (c - 1) * n_seats)
        loss = absolute + pair_weight * pair
        total = loss if total is None else total + loss
    return total / len(roots)


def train_arm(net, data, fit, *, mode, pair_weight, steps, lr, device, replay=None, shuffle_seed=None):
    if mode == "readout":
        for name, param in net.named_parameters():
            param.requires_grad = name.startswith(("per_seat_head", "global_head"))
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    rng = random.Random(0)
    shuffle = random.Random(shuffle_seed) if shuffle_seed is not None else None
    train = data["train"]
    best = (float("inf"), None)
    for step in range(steps):
        net.train()
        batch = rng.sample(train, min(32, len(train)))
        loss = sibling_loss(net, batch, fit, pair_weight, device, shuffle)
        if replay is not None:
            raw = next(replay)
            replay_loss, _ = nw.losses(net(*[nw.to_tensors(raw, device)[k] for k in ("sheet_planes", "sheet_scalars", "viewer_plane", "global_scalars")]), nw.to_tensors(raw, device))
            loss = loss + replay_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
        if (step + 1) % 50 == 0 or step + 1 == steps:
            net.eval()
            with torch.no_grad():
                val = float(sibling_loss(net, data["val"], fit, pair_weight, device))
            if val < best[0]:
                best = (val, {k: v.detach().clone() for k, v in net.state_dict().items()})
    if best[1] is not None:
        net.load_state_dict(best[1])
    net.eval()
    return net


def probe(args) -> None:
    from games.welcome_to import s2_promotion, s2_train

    out = Path(args.out)
    payload = torch.load(out / "dataset.pt", weights_only=False)
    roots = payload["roots"]
    fit = payload["fit_futures"]
    data = _split(roots, args.seed)
    cfg = s2_promotion.gate_search_config(payload["simulations"])
    device = torch.device(args.device)
    print({k: len(v) for k, v in data.items()}, "roots", flush=True)

    def evaluate(net, name):
        result = {
            split: score_decisions([model_values(net, r, cfg, device) for r in data[split]], data[split], fit)
            for split in ("train", "test")
        }
        reg = result["test"]["regret"]
        print(
            f"{name:12s} test regret {reg['mean']:.4f} ± {reg['se']:.4f}  "
            f"played {result['test']['regret_played']['mean']:.4f}  uniform {result['test']['regret_uniform']['mean']:.4f}  "
            f"sign acc {result['test']['pair_sign_accuracy']['mean']:.3f}  "
            f"resolved {result['test']['resolved_pair_sign_accuracy']['mean']:.3f} "
            f"(n={result['test']['resolved_pair_sign_accuracy']['n']})  "
            f"spearman {result['test']['spearman_within_root']['mean']:.3f}  "
            f"| train regret {result['train']['regret']['mean']:.4f}",
            flush=True,
        )
        return result

    def fresh():
        return s2_train.load_training_checkpoint(args.checkpoint, args.device)[0].eval()

    def replay_stream():
        if not args.replay:
            return None
        trajectories = [t for p in args.replay for t in self_play.read_trajectories(p)]
        loader = self_play.rust_training_loader(trajectories, 256, shuffle_seed=args.seed)

        def stream():
            while True:
                loader.reset_random(random.getrandbits(63), 1_000_000)
                yield from self_play.iter_rust_training_batches(loader)

        return stream()

    results = {"checkpoint": evaluate(fresh(), "checkpoint")}
    for weight in args.pair_weights:
        results[f"readout_w{weight:g}"] = evaluate(
            train_arm(fresh(), data, fit, mode="readout", pair_weight=weight, steps=args.steps, lr=args.lr, device=device),
            f"readout w{weight:g}",
        )
        results[f"finetune_w{weight:g}"] = evaluate(
            train_arm(fresh(), data, fit, mode="finetune", pair_weight=weight, steps=args.steps, lr=args.lr / 3, device=device, replay=replay_stream()),
            f"finetune w{weight:g}",
        )
    weight = max(args.pair_weights)
    results["shuffled"] = evaluate(
        train_arm(fresh(), data, fit, mode="finetune", pair_weight=weight, steps=args.steps, lr=args.lr / 3, device=device, replay=replay_stream(), shuffle_seed=args.seed),
        "shuffled",
    )
    (out / "probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--games", type=int, default=600)
    c.add_argument("--simulations", type=int, default=200)
    c.add_argument("--candidates", type=int, default=6)
    c.add_argument("--fit-futures", type=int, default=12)
    c.add_argument("--eval-futures", type=int, default=12)
    c.add_argument("--seed", type=int, default=7_100_000)
    c.add_argument("--device", default="cuda")
    p = sub.add_parser("probe")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--pair-weights", type=float, nargs="+", default=[0.0, 25.0, 100.0])
    p.add_argument("--replay", action="append", default=[])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    collect(args) if args.command == "collect" else probe(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
