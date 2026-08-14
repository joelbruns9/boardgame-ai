"""Stage-1 deep-target screen for the frozen BGA development positions.

Runs independent low-budget open-loop searches on each reconstructable public
state and records both joint-action and pick-group telemetry.  This is a
high-recall triage pass: it does not produce training labels and it never reads
the frozen confirmation split by default.

Example::

    python -m games.kingdomino.deep_target_screen --split development --sims 800
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import kingdomino_rust as kr

from games.kingdomino.action_codec import decode_action, encode_action
from games.kingdomino.bga_reanalysis_corpus import DEFAULT_CORPUS
from games.kingdomino.denial_search import AZBatchEvaluator, load_checkpoint_network
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.game import GameState, PickAction, TurnAction
from games.kingdomino.late_model_policy_audit import prepare_bga_state
from games.kingdomino.self_play import make_rust_evaluator


SCHEMA = "kingdomino-deep-target-screen/v1"
SUMMARY_SCHEMA = "kingdomino-deep-target-screen-summary/v1"
DEFAULT_CHECKPOINT = Path("runs/kingdomino/best_checkpoint/current_best.pt")
DEFAULT_OUTPUT = Path(
    "runs/kingdomino/placement_audit/deep_target_screen_development_s800_r2.jsonl"
)
DEFAULT_SUMMARY = Path(
    "runs/kingdomino/placement_audit/deep_target_screen_summary_development_s800_r2.json"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _seed(position_id: str, repeat: int, base_seed: int) -> int:
    digest = hashlib.blake2b(
        f"{position_id}:screen:{base_seed}:{repeat}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little")


def _control_selected(position_id: str, fraction: float, base_seed: int) -> bool:
    if fraction <= 0:
        return False
    digest = hashlib.blake2b(
        f"{position_id}:control:{base_seed}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") / 2**64 < min(1.0, fraction)


def _placement_json(placement: Any | None) -> dict[str, Any] | None:
    if placement is None:
        return None
    return {
        "x1": int(placement.x1),
        "y1": int(placement.y1),
        "x2": int(placement.x2),
        "y2": int(placement.y2),
        "flipped": bool(placement.flipped),
    }


def _action_json(state: GameState, action_idx: int) -> dict[str, Any]:
    action = decode_action(int(action_idx), state)
    if isinstance(action, PickAction):
        return {
            "action_idx": int(action_idx),
            "kind": "pick",
            "pick_domino_id": int(action.domino_id),
            "placement": None,
        }
    if not isinstance(action, TurnAction):
        raise TypeError(f"unexpected action type: {type(action).__name__}")
    return {
        "action_idx": int(action_idx),
        "kind": "turn",
        "pick_domino_id": (
            None if action.pick_domino_id is None else int(action.pick_domino_id)
        ),
        "placement": _placement_json(action.placement),
    }


def _pick_key(state: GameState, action_idx: int) -> int | None:
    action = decode_action(int(action_idx), state)
    if isinstance(action, PickAction):
        return int(action.domino_id)
    return None if action.pick_domino_id is None else int(action.pick_domino_id)


def aggregate_search(
    state: GameState,
    children: list[tuple[int, int, float, float]],
    *,
    root_value_p0: float,
    elapsed_seconds: float,
    seed: int,
) -> dict[str, Any]:
    """Convert Rust child tuples into actor-framed joint and pick telemetry."""
    actor_sign = 1.0 if int(state.current_actor) == 0 else -1.0
    legal = {int(encode_action(action, state)) for action in state.legal_actions()}
    returned = {int(child[0]) for child in children}
    if returned != legal:
        missing = sorted(legal - returned)
        extra = sorted(returned - legal)
        raise ValueError(f"search root action mismatch: missing={missing}, extra={extra}")

    joint: list[dict[str, Any]] = []
    groups: dict[int | None, dict[str, Any]] = {}
    total_visits = sum(int(child[1]) for child in children)
    for action_idx, visits, value_sum_p0, prior in children:
        visits = int(visits)
        value_sum_actor = actor_sign * float(value_sum_p0)
        q_actor = value_sum_actor / visits if visits else None
        item = {
            **_action_json(state, int(action_idx)),
            "visits": visits,
            "visit_share": visits / total_visits if total_visits else 0.0,
            "q_actor": q_actor,
            "prior": float(prior),
        }
        joint.append(item)
        key = _pick_key(state, int(action_idx))
        group = groups.setdefault(
            key,
            {
                "pick_domino_id": key,
                "visits": 0,
                "value_sum_actor": 0.0,
                "prior": 0.0,
                "actions": [],
            },
        )
        group["visits"] += visits
        group["value_sum_actor"] += value_sum_actor
        group["prior"] += float(prior)
        group["actions"].append(item)

    def joint_rank(item: dict[str, Any]) -> tuple[float, float, float, int]:
        return (
            float(item["visits"]),
            float("-inf") if item["q_actor"] is None else float(item["q_actor"]),
            float(item["prior"]),
            -int(item["action_idx"]),
        )

    joint.sort(key=joint_rank, reverse=True)
    pick_groups: list[dict[str, Any]] = []
    for group in groups.values():
        visits = int(group.pop("visits"))
        value_sum_actor = float(group.pop("value_sum_actor"))
        actions = group.pop("actions")
        best_action = max(actions, key=joint_rank)
        pick_groups.append(
            {
                "pick_domino_id": group["pick_domino_id"],
                "visits": visits,
                "visit_share": visits / total_visits if total_visits else 0.0,
                "q_actor": value_sum_actor / visits if visits else None,
                "prior": float(group["prior"]),
                "joint_action_count": len(actions),
                "best_joint_action": {
                    key: best_action[key]
                    for key in (
                        "action_idx",
                        "kind",
                        "pick_domino_id",
                        "placement",
                        "visits",
                        "visit_share",
                        "q_actor",
                        "prior",
                    )
                },
            }
        )

    def group_rank(item: dict[str, Any]) -> tuple[float, float, float, int]:
        key = item["pick_domino_id"]
        return (
            float(item["visits"]),
            float("-inf") if item["q_actor"] is None else float(item["q_actor"]),
            float(item["prior"]),
            0 if key is None else -int(key),
        )

    pick_groups.sort(key=group_rank, reverse=True)
    top2_q_gap = None
    if len(pick_groups) >= 2:
        q0, q1 = pick_groups[0]["q_actor"], pick_groups[1]["q_actor"]
        if q0 is not None and q1 is not None:
            top2_q_gap = abs(float(q0) - float(q1))
    return {
        "seed": int(seed),
        "elapsed_seconds": float(elapsed_seconds),
        "root_value_actor": actor_sign * float(root_value_p0),
        "root_total_visits": total_visits,
        "selected_action": joint[0],
        "selected_pick_domino_id": pick_groups[0]["pick_domino_id"],
        "top2_pick_q_gap": top2_q_gap,
        "top_pick_visit_share": float(pick_groups[0]["visit_share"]),
        "starved_pick_group_count": sum(group["visits"] == 0 for group in pick_groups),
        "top_joint_actions": joint[:3],
        "pick_groups": pick_groups,
    }


def aggregate_raw_policy(
    state: GameState,
    policy: dict[int, float],
) -> dict[str, Any]:
    legal = {int(encode_action(action, state)) for action in state.legal_actions()}
    if set(policy) != legal:
        raise ValueError("raw policy action set does not match legal actions")
    joint = [
        {**_action_json(state, action_idx), "probability": float(probability)}
        for action_idx, probability in policy.items()
    ]
    joint.sort(key=lambda item: (-item["probability"], item["action_idx"]))
    groups: dict[int | None, dict[str, Any]] = {}
    for item in joint:
        key = item["pick_domino_id"]
        group = groups.setdefault(
            key,
            {"pick_domino_id": key, "probability": 0.0, "best_joint_action": item},
        )
        group["probability"] += float(item["probability"])
        if item["probability"] > group["best_joint_action"]["probability"]:
            group["best_joint_action"] = item
    pick_groups = sorted(
        groups.values(),
        key=lambda item: (
            -item["probability"],
            0 if item["pick_domino_id"] is None else item["pick_domino_id"],
        ),
    )
    return {
        "selected_action": joint[0],
        "selected_pick_domino_id": pick_groups[0]["pick_domino_id"],
        "top_joint_actions": joint[:3],
        "pick_groups": pick_groups,
    }


def screen_flags(
    position: dict[str, Any],
    searches: list[dict[str, Any]],
    *,
    q_gap_threshold: float,
    visit_share_threshold: float,
    control_fraction: float,
    base_seed: int,
) -> dict[str, bool]:
    selected_actions = [int(search["selected_action"]["action_idx"]) for search in searches]
    selected_picks = [search["selected_pick_domino_id"] for search in searches]
    human = position.get("human_action")
    human_pick = None if not human else human.get("pick_domino_id")
    flags = {
        "seed_action_disagreement": len(set(selected_actions)) > 1,
        "seed_pick_disagreement": len(set(selected_picks)) > 1,
        "top2_q_close": any(
            search["top2_pick_q_gap"] is not None
            and float(search["top2_pick_q_gap"]) <= q_gap_threshold
            for search in searches
        ),
        "top_pick_visit_share_low": any(
            float(search["top_pick_visit_share"]) < visit_share_threshold
            for search in searches
        ),
        "pick_group_starved": any(
            int(search["starved_pick_group_count"]) > 0 for search in searches
        ),
        "human_pick_disagreement": bool(
            human
            and human_pick is not None
            and any(pick != human_pick for pick in selected_picks)
        ),
        "exact_candidate": bool(position.get("audit_tags", {}).get("exact_candidate")),
        "random_control": False,
    }
    if not any(flags.values()):
        flags["random_control"] = _control_selected(
            str(position["position_id"]), control_fraction, base_seed
        )
    return flags


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def run_screen(
    *,
    corpus_path: Path,
    checkpoint: Path,
    output_path: Path,
    summary_path: Path,
    split: str,
    device: str,
    sims: int,
    repeats: int,
    base_seed: int,
    q_gap_threshold: float,
    visit_share_threshold: float,
    control_fraction: float,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if repeats < 2:
        raise ValueError("Stage 1 requires at least two independent repeats")
    positions = [row for row in _read_jsonl(corpus_path) if row.get("split") == split]
    if limit is not None:
        positions = positions[: max(0, int(limit))]
    if not positions:
        raise ValueError(f"no {split!r} positions found in {corpus_path}")

    net, config = load_checkpoint_network(checkpoint, device)
    margin_gain = float(config.get("margin_gain", 2.0))
    alpha = float(config.get("alpha", 0.5))
    raw_evaluator = AZBatchEvaluator(
        net,
        device=device,
        batch_size=256,
        margin_gain=margin_gain,
        alpha=alpha,
    )
    rust_evaluator = make_rust_evaluator(
        net,
        device=device,
        amp=bool(config.get("inference_amp", False)),
        margin_gain=margin_gain,
        alpha=alpha,
    )

    prepared: list[GameState] = []
    for position in positions:
        rules = position["state"].get("rules", {})
        prepared.append(
            prepare_bga_state(
                position["state"],
                harmony=bool(rules.get("harmony", True)),
                middle=bool(rules.get("middle_kingdom", True)),
            )
        )
    raw_evaluator.policies(prepared)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, (position, state) in enumerate(zip(positions, prepared)):
            raw = aggregate_raw_policy(state, raw_evaluator.policy(state))
            rust_state = _rust_state_from_python(state)
            searches: list[dict[str, Any]] = []
            for repeat in range(repeats):
                search_seed = _seed(str(position["position_id"]), repeat, base_seed)
                search_started = time.perf_counter()
                children, root_value_p0 = kr.advisor_open_loop_search(
                    rust_state,
                    rust_evaluator,
                    int(sims),
                    dirichlet_eps=0.0,
                    fpu=float(config.get("fpu", -0.2)),
                    cpuct=float(config.get("c_puct", 1.5)),
                    seed=search_seed,
                    leaf_batch=max(1, int(config.get("leaf_batch", 6))),
                    virtual_loss=int(config.get("virtual_loss", 1)),
                    score_scale=float(config.get("score_scale", 160.0)),
                    margin_gain=margin_gain,
                    alpha=alpha,
                )
                searches.append(
                    aggregate_search(
                        state,
                        children,
                        root_value_p0=float(root_value_p0),
                        elapsed_seconds=time.perf_counter() - search_started,
                        seed=search_seed,
                    )
                )
            flags = screen_flags(
                position,
                searches,
                q_gap_threshold=q_gap_threshold,
                visit_share_threshold=visit_share_threshold,
                control_fraction=control_fraction,
                base_seed=base_seed,
            )
            row = {
                "schema": SCHEMA,
                "position_id": position["position_id"],
                "table_id": position["table_id"],
                "source_decision_index": position["source_decision_index"],
                "split": position["split"],
                "actor": position["actor"],
                "actor_role": position["actor_role"],
                "phase": position["phase"],
                "deck_count": position["deck_count"],
                "state_sha256": position["state_sha256"],
                "legal_action_count": position["legal"]["action_count"],
                "legal_pick_domino_ids": position["legal"]["pick_domino_ids"],
                "human_action": position.get("human_action"),
                "audit_tags": position.get("audit_tags", {}),
                "raw_policy": raw,
                "searches": searches,
                "screen_flags": flags,
                "escalate_to_4800": any(flags.values()),
            }
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            rows.append(row)
            if (index + 1) % 50 == 0 or index + 1 == len(positions):
                elapsed = time.perf_counter() - started
                print(
                    f"  {index + 1}/{len(positions)} positions "
                    f"({elapsed:.1f}s, {(index + 1) / elapsed:.2f} positions/s)",
                    flush=True,
                )

    trigger_counts: Counter[str] = Counter()
    by_deck: Counter[int] = Counter()
    by_phase: Counter[str] = Counter()
    escalated = 0
    action_agreement = 0
    pick_agreement = 0
    for row in rows:
        for key, enabled in row["screen_flags"].items():
            if enabled:
                trigger_counts[key] += 1
        if row["escalate_to_4800"]:
            escalated += 1
            by_deck[int(row["deck_count"])] += 1
            by_phase[str(row["phase"])] += 1
        searches = row["searches"]
        action_agreement += len({s["selected_action"]["action_idx"] for s in searches}) == 1
        pick_agreement += len({s["selected_pick_domino_id"] for s in searches}) == 1

    summary = {
        "schema": SUMMARY_SCHEMA,
        "screen_schema": SCHEMA,
        "split": split,
        "positions": len(rows),
        "sims": int(sims),
        "repeats": int(repeats),
        "total_searches": len(rows) * repeats,
        "thresholds": {
            "top2_pick_q_gap": float(q_gap_threshold),
            "top_pick_visit_share": float(visit_share_threshold),
            "random_control_fraction": float(control_fraction),
        },
        "two_seed_action_agreement": action_agreement / len(rows),
        "two_seed_pick_agreement": pick_agreement / len(rows),
        "escalate_to_4800": escalated,
        "escalation_fraction": escalated / len(rows),
        "trigger_counts": _counter_dict(trigger_counts),
        "escalated_by_deck_count": _counter_dict(by_deck),
        "escalated_by_phase": _counter_dict(by_phase),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_config": config,
        "corpus": str(corpus_path),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "output": str(output_path),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "confirmation_positions_searched": sum(row["split"] == "confirmation" for row in rows),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--split", choices=("development", "confirmation"), default="development")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--q-gap-threshold", type=float, default=0.05)
    parser.add_argument("--visit-share-threshold", type=float, default=0.60)
    parser.add_argument("--control-fraction", type=float, default=0.15)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    _rows, summary = run_screen(
        corpus_path=args.corpus,
        checkpoint=args.checkpoint,
        output_path=args.output,
        summary_path=args.summary,
        split=args.split,
        device=args.device,
        sims=args.sims,
        repeats=args.repeats,
        base_seed=args.seed,
        q_gap_threshold=args.q_gap_threshold,
        visit_share_threshold=args.visit_share_threshold,
        control_fraction=args.control_fraction,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
