"""Forced-move continuation duel for deck=8 split/control disagreements.

The two arms differ only in the first action, which is taken from the completed
6,400-simulation control and sampled-split artifacts. Every continuation after
that action uses the same high-budget aliased open-loop engine and checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from games.kingdomino.action_codec import decode_action, encode_action
from games.kingdomino.chance_leverage_probe import DEFAULT_POSITIONS
from games.kingdomino.denial_search import load_checkpoint_network, public_state_key
from games.kingdomino.denial_signal_sweep import file_sha256, load_frozen_positions
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.game import Phase
from games.kingdomino.nnue.sparse_encoder import swap_players
from games.kingdomino.promotion import DEFAULT_CURRENT_BEST, sha256_file
from games.kingdomino.self_play import make_rust_evaluator


DEFAULT_DIR = Path("runs/kingdomino/chance_correct_a1")
DEFAULT_BASE = DEFAULT_DIR / "deck8_causal_leverage_v1.json"
DEFAULT_SPLIT = DEFAULT_DIR / "deck8_sampled_split_ablation_v1.json"
DEFAULT_OUTPUT = DEFAULT_DIR / "deck8_forced_move_duel_v1.json"
DUEL_VERSION = "deck8-forced-move-disagreement-duel-v1"
ARM_ORDER = ("control_forced", "split_forced")
EXPECTED_DISAGREEMENTS = 33
EXPECTED_PICK_DISAGREEMENTS = 17


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _source_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _stable_seed(*parts: int) -> int:
    payload = ":".join(str(int(part)) for part in parts).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _mean_ci(values: Sequence[float], *, seed: int, resamples: int = 20_000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def extract_disagreements(
    base: dict[str, Any], sampled: dict[str, Any], *, budget: int
) -> list[dict[str, Any]]:
    """Freeze the exact positions/actions changed by sampled split."""
    base_rows = {
        (int(row["position_index"]), int(row["configured_budget"]), str(row["arm"])): row
        for row in base["results"]
    }
    sampled_rows = {
        (int(row["position_index"]), int(row["configured_budget"])): row
        for row in sampled["results"]
    }
    indices = sorted(
        index for index, row_budget in sampled_rows if row_budget == int(budget)
    )
    out = []
    for position_index in indices:
        control = base_rows[(position_index, int(budget), "control")]
        split = sampled_rows[(position_index, int(budget))]
        control_action = int(control["top_action_idx"])
        split_action = int(split["top_action_idx"])
        if control_action == split_action:
            continue
        out.append({
            "position_index": position_index,
            "control_action_idx": control_action,
            "split_action_idx": split_action,
            "control_pick_rank": int(control["top_pick_rank"]),
            "split_pick_rank": int(split["top_pick_rank"]),
            "pick_changed": int(control["top_pick_rank"]) != int(split["top_pick_rank"]),
            "paired_search_seed": int(control["seed"]),
        })
    return out


def _legal_action_by_index(state, action_idx: int):
    legal = {int(encode_action(action, state)): action for action in state.legal_actions()}
    if int(action_idx) not in legal:
        # Decode once for a more useful error if the artifact contains a stale or
        # representation-specific index.
        decode_action(int(action_idx), state)
        raise ValueError(f"forced action {action_idx} is not legal")
    return legal[int(action_idx)]


def prepare_forced_state(
    root,
    *,
    action_idx: int,
    deck_seed: int,
    mirrored: bool,
):
    """Shuffle hidden order, optionally relabel players, then force one action."""
    if root.phase != Phase.PLACE_AND_SELECT or root.actor_index != 0 or len(root.deck) != 8:
        raise ValueError("forced duel roots must be deck=8 first-selection positions")
    state = root.copy()
    random.Random(int(deck_seed)).shuffle(state.deck)
    deck_order = tuple(int(tile) for tile in state.deck)
    original_actor = int(state.current_actor)
    if mirrored:
        state = swap_players(state)
    chooser = int(state.current_actor)
    if chooser != (1 - original_actor if mirrored else original_actor):
        raise RuntimeError("player relabeling did not flip the chooser")
    action = _legal_action_by_index(state, int(action_idx))
    child = state.step(action)
    return child, chooser, deck_order


def build_tasks(
    positions,
    disagreements: Sequence[dict[str, Any]],
    *,
    continuation_seeds: int,
    deck_seed_base: int,
    search_seed_base: int,
) -> list[dict[str, Any]]:
    tasks = []
    for disagreement in disagreements:
        position_index = int(disagreement["position_index"])
        root, source = positions[position_index]
        for continuation_index in range(int(continuation_seeds)):
            deck_seed = _stable_seed(deck_seed_base, position_index, continuation_index)
            # Deliberately independent of arm and mirror. Cohorts are run
            # separately, so the same unique seed can be reused across them.
            game_seed = _stable_seed(search_seed_base, position_index, continuation_index)
            for mirror in (0, 1):
                tasks.append({
                    **disagreement,
                    "continuation_index": continuation_index,
                    "deck_seed": deck_seed,
                    "game_seed": game_seed,
                    "mirror": mirror,
                    "root_state_key": public_state_key(root),
                    "position_source": source,
                })
    return tasks


def _points(outcome: int) -> float:
    return 1.0 if outcome > 0 else 0.5 if outcome == 0 else 0.0


def _subgroup_summary(position_rows: Sequence[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    margin = [float(row["mean_margin_delta"]) for row in position_rows]
    points = [float(row["mean_points_delta"]) for row in position_rows]
    return {
        "positions": len(position_rows),
        "mean_chooser_margin_delta": statistics.fmean(margin) if margin else 0.0,
        "mean_chooser_margin_delta_bootstrap_95": _mean_ci(margin, seed=seed),
        "mean_chooser_points_delta": statistics.fmean(points) if points else 0.0,
        "mean_chooser_points_delta_bootstrap_95": _mean_ci(points, seed=seed + 1),
        "positions_favoring_split_margin": sum(value > 0.0 for value in margin),
        "positions_tied_margin": sum(value == 0.0 for value in margin),
        "positions_favoring_control_margin": sum(value < 0.0 for value in margin),
    }


def build_summary(
    results: Sequence[dict[str, Any]],
    disagreements: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_key = {
        (
            int(row["position_index"]),
            int(row["continuation_index"]),
            int(row["mirror"]),
            str(row["arm"]),
        ): row
        for row in results
    }
    pick_changed = {
        int(row["position_index"]): bool(row["pick_changed"])
        for row in disagreements
    }
    seed_pairs = []
    position_indices = sorted(pick_changed)
    for position_index in position_indices:
        continuation_indices = sorted({
            key[1] for key in by_key if key[0] == position_index
        })
        for continuation_index in continuation_indices:
            mirror_deltas = []
            complete = True
            for mirror in (0, 1):
                control = by_key.get((position_index, continuation_index, mirror, "control_forced"))
                split = by_key.get((position_index, continuation_index, mirror, "split_forced"))
                if control is None or split is None:
                    complete = False
                    break
                mirror_deltas.append({
                    "mirror": mirror,
                    "margin_delta": float(split["chooser_margin"] - control["chooser_margin"]),
                    "points_delta": float(split["chooser_points"] - control["chooser_points"]),
                })
            if not complete:
                continue
            seed_pairs.append({
                "position_index": position_index,
                "continuation_index": continuation_index,
                "pick_changed": pick_changed[position_index],
                "mean_margin_delta": statistics.fmean(row["margin_delta"] for row in mirror_deltas),
                "mean_points_delta": statistics.fmean(row["points_delta"] for row in mirror_deltas),
                "mirror_margin_abs_difference": abs(
                    mirror_deltas[0]["margin_delta"] - mirror_deltas[1]["margin_delta"]
                ),
                "mirrors": mirror_deltas,
            })

    position_rows = []
    for position_index in position_indices:
        rows = [row for row in seed_pairs if row["position_index"] == position_index]
        if not rows:
            continue
        position_rows.append({
            "position_index": position_index,
            "pick_changed": pick_changed[position_index],
            "continuation_pairs": len(rows),
            "mean_margin_delta": statistics.fmean(row["mean_margin_delta"] for row in rows),
            "mean_points_delta": statistics.fmean(row["mean_points_delta"] for row in rows),
            "mean_mirror_margin_abs_difference": statistics.fmean(
                row["mirror_margin_abs_difference"] for row in rows
            ),
        })

    all_positions = _subgroup_summary(position_rows, seed=20260851)
    pick_positions = _subgroup_summary(
        [row for row in position_rows if row["pick_changed"]], seed=20260853
    )
    low, high = all_positions["mean_chooser_margin_delta_bootstrap_95"]
    margin = all_positions["mean_chooser_margin_delta"]
    points = all_positions["mean_chooser_points_delta"]
    if position_rows and low > 0.0 and margin > 0.0 and points >= 0.0:
        screen = "positive"
    elif position_rows and high < 0.0 and margin < 0.0 and points <= 0.0:
        screen = "negative"
    else:
        screen = "inconclusive"
    return {
        "completed_cells": len(results),
        "completed_seed_pairs": len(seed_pairs),
        "completed_positions": len(position_rows),
        "uncertainty_unit": "disagreement position after averaging continuation seeds and label mirrors",
        "screen": screen,
        "all_disagreements": all_positions,
        "pick_changing_disagreements": pick_positions,
        "mean_mirror_margin_abs_difference": (
            statistics.fmean(row["mean_mirror_margin_abs_difference"] for row in position_rows)
            if position_rows else 0.0
        ),
        "position_rows": position_rows,
        "seed_pairs": seed_pairs,
    }


class CountingEvaluator:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.batch_sizes: list[int] = []

    def __call__(self, my, opp, flat, legal):
        self.batch_sizes.append(int(len(my)))
        return self.evaluator(my, opp, flat, legal)

    def summary(self) -> dict[str, Any]:
        counts = Counter(self.batch_sizes)
        return {
            "calls": len(self.batch_sizes),
            "rows": sum(self.batch_sizes),
            "max_batch": max(self.batch_sizes, default=0),
            "batch_histogram": {str(key): counts[key] for key in sorted(counts)},
        }


def run_state_batch(states, game_seeds, evaluator, settings: dict[str, Any]):
    import kingdomino_rust as kr

    counted = CountingEvaluator(evaluator)
    mcts = kr.BatchedMCTS.from_states(
        states,
        game_seeds,
        int(settings["sims"]),
        leaf_batch=int(settings["leaf_batch"]),
        virtual_loss=1,
        cpuct=float(settings["cpuct"]),
        fpu=float(settings["fpu"]),
        dirichlet_alpha=0.3,
        dirichlet_eps=0.0,
        temp_moves=0,
        score_scale=float(settings["score_scale"]),
        margin_gain=float(settings["margin_gain"]),
        alpha=float(settings["alpha"]),
        exact_endgame_max_secs=float(settings["exact_endgame_max_secs"]),
        async_solve=True,
        solver_cpus=int(settings["solver_cpus"]),
    )
    started = time.perf_counter()
    finished = []
    ticks = 0
    while not mcts.done():
        my, opp, flat, legal = mcts.step()
        values, gathered = counted(my, opp, flat, legal)
        finished.extend(mcts.update(values, gathered))
        ticks += 1
        if ticks > 2_000_000:
            raise RuntimeError("forced continuation batch exceeded tick guard")
    elapsed = time.perf_counter() - started
    rows = {}
    for seed, _examples, scores in finished:
        seed = int(seed)
        if seed in rows:
            raise RuntimeError(f"duplicate finished seed {seed}")
        rows[seed] = {
            "score0": int(scores[0]),
            "score1": int(scores[1]),
            "outcome0": int(scores[2]),
        }
    if set(rows) != set(int(seed) for seed in game_seeds):
        raise RuntimeError("forced continuation batch returned the wrong seed set")
    diagnostics = {
        "games": len(states),
        "ticks": ticks,
        "elapsed_seconds": elapsed,
        "inference": counted.summary(),
        "exact_solve_count": int(mcts.exact_solve_count),
        "exact_tree_solve_count": int(mcts.exact_tree_solve_count),
        "exact_fallback_count": int(mcts.exact_fallback_count),
        "chance_panels": int(mcts.deck8_chance_panel_count),
    }
    if diagnostics["chance_panels"] != 0:
        raise RuntimeError("continuation arbiter unexpectedly enabled chance panels")
    return rows, diagnostics


def _load_design(args: argparse.Namespace):
    base_path = Path(args.base_artifact)
    split_path = Path(args.split_artifact)
    positions_path = Path(args.positions_path)
    checkpoint = Path(args.checkpoint)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    sampled = json.loads(split_path.read_text(encoding="utf-8"))
    if not base.get("completed") or not sampled.get("completed"):
        raise ValueError("forced duel requires both completed Test 1 artifacts")
    if sampled["provenance"]["base_artifact_sha256"] != file_sha256(base_path):
        raise ValueError("sampled-split artifact is not paired to this base artifact")
    base_provenance = base["provenance"]
    if file_sha256(positions_path) != base_provenance["positions_sha256"]:
        raise ValueError("frozen position hash differs from Test 1")
    if sha256_file(checkpoint) != base_provenance["checkpoint_sha256"]:
        raise ValueError("checkpoint hash differs from Test 1")
    disagreements = extract_disagreements(base, sampled, budget=int(args.budget))
    if not args.allow_nonstandard_design:
        if len(disagreements) != EXPECTED_DISAGREEMENTS:
            raise ValueError(
                f"expected {EXPECTED_DISAGREEMENTS} disagreements, got {len(disagreements)}"
            )
        pick_count = sum(bool(row["pick_changed"]) for row in disagreements)
        if pick_count != EXPECTED_PICK_DISAGREEMENTS:
            raise ValueError(
                f"expected {EXPECTED_PICK_DISAGREEMENTS} pick disagreements, got {pick_count}"
            )
    positions = load_frozen_positions(positions_path)
    if args.limit_positions:
        allowed = {row["position_index"] for row in disagreements[: args.limit_positions]}
        disagreements = [row for row in disagreements if row["position_index"] in allowed]
    for row in disagreements:
        state = positions[int(row["position_index"])][0]
        _legal_action_by_index(state, int(row["control_action_idx"]))
        _legal_action_by_index(state, int(row["split_action_idx"]))
    return base, sampled, positions, disagreements


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.continuation_seeds <= 0 or args.batch_slots <= 0 or args.sims <= 0:
        raise ValueError("continuation seeds, batch slots, and sims must be positive")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    base, sampled, positions, disagreements = _load_design(args)
    repo = Path(__file__).resolve().parents[2]
    runner_path = Path(__file__).resolve()
    rust_path = repo / "games/kingdomino/kingdomino_rust/src/lib.rs"
    checkpoint = Path(args.checkpoint)
    checkpoint_hash = sha256_file(checkpoint)
    net, checkpoint_cfg = load_checkpoint_network(checkpoint, args.device)
    settings = {
        "sims": int(args.sims),
        "leaf_batch": int(args.leaf_batch),
        "batch_slots": int(args.batch_slots),
        "cpuct": float(checkpoint_cfg.get("c_puct", 1.5)),
        "fpu": float(checkpoint_cfg.get("fpu", -0.2)),
        "score_scale": float(checkpoint_cfg.get("score_scale", 160.0)),
        "margin_gain": float(checkpoint_cfg.get("margin_gain", 2.0)),
        "alpha": float(checkpoint_cfg.get("alpha", 0.5)),
        "exact_endgame_max_secs": float(
            checkpoint_cfg.get("exact_endgame_max_secs", 3.0)
        ),
        "solver_cpus": int(args.solver_cpus),
        "engine": "aliased_open_loop",
        "chance_enumeration": False,
        "dirichlet_epsilon": 0.0,
        "temperature": 0.0,
    }
    provenance = {
        "version": DUEL_VERSION,
        "source_commit": _source_commit(repo),
        "runner_sha256": file_sha256(runner_path),
        "rust_sha256": file_sha256(rust_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "positions_path": str(Path(args.positions_path)),
        "positions_sha256": file_sha256(Path(args.positions_path)),
        "base_artifact": str(Path(args.base_artifact)),
        "base_artifact_sha256": file_sha256(Path(args.base_artifact)),
        "split_artifact": str(Path(args.split_artifact)),
        "split_artifact_sha256": file_sha256(Path(args.split_artifact)),
        "budget": int(args.budget),
        "continuation_seeds_per_position": int(args.continuation_seeds),
        "deck_seed_base": int(args.deck_seed_base),
        "search_seed_base": int(args.search_seed_base),
        "seed_derivation": "sha256(base:position_index:continuation_index) first 64 bits",
        "arms": list(ARM_ORDER),
        "label_mirrors": [0, 1],
        "disagreements": disagreements,
        "settings": settings,
        "scope": "conditional action quality on frozen 6400-sim disagreement positions",
    }
    output = Path(args.output)
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("provenance") != provenance:
            raise ValueError(f"existing duel artifact provenance differs: {output}")
        if payload.get("completed"):
            return payload
    else:
        payload = {"provenance": provenance, "results": [], "chunks": []}

    tasks = build_tasks(
        positions,
        disagreements,
        continuation_seeds=int(args.continuation_seeds),
        deck_seed_base=int(args.deck_seed_base),
        search_seed_base=int(args.search_seed_base),
    )
    completed = {
        (
            int(row["position_index"]),
            int(row["continuation_index"]),
            int(row["mirror"]),
            str(row["arm"]),
        )
        for row in payload["results"]
    }
    evaluator = make_rust_evaluator(
        net,
        device=args.device,
        amp=bool(args.amp_inference),
        margin_gain=settings["margin_gain"],
        alpha=settings["alpha"],
    )

    # Mirrors are separate cohorts so the exact same game_seed can be reused;
    # arm order rotates by mirror to avoid a fixed warm/thermal ordering.
    for mirror in (0, 1):
        arms = ARM_ORDER if mirror == 0 else tuple(reversed(ARM_ORDER))
        mirror_tasks = [task for task in tasks if int(task["mirror"]) == mirror]
        for arm in arms:
            pending = [
                task for task in mirror_tasks
                if (
                    int(task["position_index"]),
                    int(task["continuation_index"]),
                    mirror,
                    arm,
                ) not in completed
            ]
            for start in range(0, len(pending), int(args.batch_slots)):
                chunk = pending[start : start + int(args.batch_slots)]
                states = []
                seeds = []
                prepared = []
                for task in chunk:
                    root = positions[int(task["position_index"])][0]
                    action_idx = int(
                        task["control_action_idx"]
                        if arm == "control_forced" else task["split_action_idx"]
                    )
                    child, chooser, deck_order = prepare_forced_state(
                        root,
                        action_idx=action_idx,
                        deck_seed=int(task["deck_seed"]),
                        mirrored=bool(mirror),
                    )
                    rust_state = _rust_state_from_python(child)
                    if rust_state is None:
                        raise RuntimeError("failed to convert forced continuation state")
                    states.append(rust_state)
                    seeds.append(int(task["game_seed"]))
                    prepared.append((task, action_idx, chooser, deck_order, public_state_key(child)))
                finished, diagnostics = run_state_batch(states, seeds, evaluator, settings)
                for task, action_idx, chooser, deck_order, child_key in prepared:
                    game = finished[int(task["game_seed"])]
                    outcome_chooser = int(game["outcome0"] if chooser == 0 else -game["outcome0"])
                    chooser_margin = int(
                        game["score0"] - game["score1"]
                        if chooser == 0 else game["score1"] - game["score0"]
                    )
                    row = {
                        **task,
                        "arm": arm,
                        "forced_action_idx": action_idx,
                        "chooser": chooser,
                        "forced_child_state_key": child_key,
                        "shuffled_deck_order": list(deck_order),
                        **game,
                        "outcome_chooser": outcome_chooser,
                        "chooser_margin": chooser_margin,
                        "chooser_points": _points(outcome_chooser),
                    }
                    payload["results"].append(row)
                    completed.add((
                        int(task["position_index"]),
                        int(task["continuation_index"]),
                        mirror,
                        arm,
                    ))
                payload["chunks"].append({
                    "mirror": mirror,
                    "arm": arm,
                    "first_position_index": int(chunk[0]["position_index"]),
                    "games": len(chunk),
                    **diagnostics,
                })
                _atomic_json(output, payload)
                print(
                    f"mirror={mirror} arm={arm} completed={len(payload['results'])}/"
                    f"{len(tasks) * len(ARM_ORDER)}",
                    flush=True,
                )

    expected_cells = len(tasks) * len(ARM_ORDER)
    if len(payload["results"]) != expected_cells or len(completed) != expected_cells:
        raise RuntimeError("forced duel finished with incomplete or duplicate cells")
    payload["summary"] = build_summary(payload["results"], disagreements)
    payload["completed"] = True
    _atomic_json(output, payload)
    return payload


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CURRENT_BEST))
    parser.add_argument("--positions-path", default=str(DEFAULT_POSITIONS))
    parser.add_argument("--base-artifact", default=str(DEFAULT_BASE))
    parser.add_argument("--split-artifact", default=str(DEFAULT_SPLIT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--budget", type=int, default=6400)
    parser.add_argument("--continuation-seeds", type=int, default=16)
    parser.add_argument("--deck-seed-base", type=int, default=2_026_090_100)
    parser.add_argument("--search-seed-base", type=int, default=2_026_090_200)
    parser.add_argument("--sims", type=int, default=1600)
    parser.add_argument("--batch-slots", type=int, default=32)
    parser.add_argument("--leaf-batch", type=int, default=6)
    parser.add_argument("--solver-cpus", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-inference", action="store_true")
    parser.add_argument("--limit-positions", type=int, default=0)
    parser.add_argument("--allow-nonstandard-design", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    payload = run(_parse_args(argv))
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
