"""Exact deck-8 chance-boundary oracle for Kingdomino.

The production exact solver is fast only after future chance has disappeared.
At the final action of a round with eight tiles left in the public bag, that
action reveals four tiles and leaves a deterministic four-tile tail.  This
harness enumerates all C(8, 4) = 70 public rows, solves every conditioned tail,
and averages those exact values uniformly.

This is intentionally *not* a general deck-8 root solver.  Earlier actors in the
round introduce a large minimax tree before the reveal.  Callers must provide an
explicit, frozen prefix that reaches the final pre-reveal actor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from games.kingdomino.action_codec import encode_action
from games.kingdomino.denial_search import chance_public_state_key_v1, public_state_key
from games.kingdomino.denial_signal_sweep import file_sha256, load_frozen_positions
from games.kingdomino.endgame_solver import _step_public_futures, exact_endgame_value
from games.kingdomino.game import GameState, Phase


SCHEMA_VERSION = "kd-deck8-boundary-oracle-v1"
EXPECTED_ROWS = math.comb(8, 4)


@dataclass(frozen=True)
class UtilitySpec:
    score_scale: float = 160.0
    margin_gain: float = 2.0
    alpha: float = 0.5


TailSolver = Callable[[GameState], tuple[float, bool]]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _legal_action_map(state: GameState) -> dict[int, object]:
    out: dict[int, object] = {}
    for action in state.legal_actions():
        action_idx = int(encode_action(action, state))
        if action_idx in out:
            raise ValueError(f"duplicate canonical action index {action_idx}")
        out[action_idx] = action
    return out


def apply_prefix_actions(state: GameState, action_indices: Iterable[int]) -> GameState:
    current = state.copy()
    for offset, requested in enumerate(action_indices):
        legal = _legal_action_map(current)
        action_idx = int(requested)
        if action_idx not in legal:
            raise ValueError(
                f"prefix action {offset} index {action_idx} is not legal; "
                f"legal indices are {sorted(legal)}"
            )
        current = current.step(legal[action_idx])
    return current


def validate_boundary_state(state: GameState) -> None:
    if state.phase != Phase.PLACE_AND_SELECT:
        raise ValueError(
            f"deck-8 oracle requires PLACE_AND_SELECT, got {state.phase.name}"
        )
    if len(state.deck) != 8:
        raise ValueError(f"deck-8 oracle requires exactly 8 bag tiles, got {len(state.deck)}")
    if not state.pending_claims:
        raise ValueError("deck-8 oracle requires pending claims")
    if state.actor_index != len(state.pending_claims) - 1:
        raise ValueError(
            "deck-8 oracle requires the final pre-reveal actor; "
            f"actor_index={state.actor_index}, pending={len(state.pending_claims)}"
        )
    if len(state.current_row) != 1:
        raise ValueError(
            f"final pre-reveal actor must have one public pick left, got {state.current_row}"
        )
    if not state.legal_actions():
        raise ValueError("deck-8 boundary state has no legal actions")


def conditioned_tails(state: GameState, action: object) -> list[tuple[GameState, float]]:
    """Return and validate the complete uniform row support after ``action``."""
    validate_boundary_state(state)
    futures = _step_public_futures(state, action)
    if len(futures) != EXPECTED_ROWS:
        raise ValueError(
            f"expected {EXPECTED_ROWS} conditioned rows, got {len(futures)}"
        )
    rows = [tuple(int(x) for x in child.current_row) for child, _ in futures]
    if len(set(rows)) != EXPECTED_ROWS:
        raise ValueError("conditioned row support contains duplicates")
    expected_p = 1.0 / EXPECTED_ROWS
    for child, probability in futures:
        if child.phase != Phase.PLACE_AND_SELECT or len(child.deck) != 4:
            raise ValueError(
                "conditioned tail must be a no-chance PLACE_AND_SELECT deck-4 state"
            )
        if not math.isclose(float(probability), expected_p, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"non-uniform deck-8 row probability {probability}")
    if not math.isclose(sum(float(p) for _, p in futures), 1.0, abs_tol=1e-12):
        raise ValueError("conditioned row probabilities do not sum to one")
    return futures


def evaluate_boundary_in_memory(
    state: GameState,
    solve_tail: TailSolver,
) -> list[dict[str, Any]]:
    """Small pure seam used by tests; production runs use resumable cells."""
    validate_boundary_state(state)
    actor = int(state.current_actor)
    results = []
    for action_idx, action in sorted(_legal_action_map(state).items()):
        values = []
        for child, probability in conditioned_tails(state, action):
            value0, solved = solve_tail(child)
            if not solved or not math.isfinite(float(value0)):
                raise RuntimeError(f"tail solve failed for action {action_idx}")
            values.append((float(value0), float(probability)))
        expected0 = sum(value * probability for value, probability in values)
        results.append({
            "action_idx": action_idx,
            "expected_value_player0": expected0,
            "expected_value_actor": expected0 if actor == 0 else -expected0,
            "rows_solved": len(values),
        })
    return sorted(results, key=lambda row: (-row["expected_value_actor"], row["action_idx"]))


def _cell_path(cells_dir: Path, action_idx: int, row: tuple[int, ...]) -> Path:
    row_label = "-".join(str(x) for x in row)
    return cells_dir / f"action_{action_idx:04d}__row_{row_label}.json"


def _read_cell(path: Path, expected_key: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"cell schema mismatch: {path}")
    if payload.get("tail_state_key") != expected_key or payload.get("solved") is not True:
        raise ValueError(f"cell identity/solve mismatch: {path}")
    value = float(payload["value_player0"])
    if not math.isfinite(value):
        raise ValueError(f"cell contains non-finite value: {path}")
    return payload


def run_oracle(args: argparse.Namespace) -> dict[str, Any]:
    positions_path = Path(args.positions_path)
    positions = load_frozen_positions(positions_path)
    if not 0 <= int(args.position_index) < len(positions):
        raise ValueError(
            f"position index {args.position_index} outside corpus of {len(positions)}"
        )
    root, source = positions[int(args.position_index)]
    prefix = [int(x) for x in str(args.prefix_actions).split(",") if x.strip()]
    boundary = apply_prefix_actions(root, prefix)
    validate_boundary_state(boundary)

    utility = UtilitySpec(
        score_scale=float(args.score_scale),
        margin_gain=float(args.margin_gain),
        alpha=float(args.alpha),
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "positions_sha256": file_sha256(positions_path),
        "position_index": int(args.position_index),
        "root_state_key": public_state_key(root),
        "prefix_actions": prefix,
        "boundary_state_key": public_state_key(boundary),
        "boundary_chance_key_hex": chance_public_state_key_v1(boundary).hex(),
        "utility": asdict(utility),
    }
    identity["oracle_id"] = _stable_digest(identity)

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    cells_dir = output_dir / "cells"
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing_summary.get("status") == "complete":
            raise FileExistsError(f"completed oracle output already exists: {summary_path}")
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise ValueError(f"oracle manifest identity mismatch: {manifest_path}")
    else:
        _atomic_json(manifest_path, {
            "identity": identity,
            "selection_reason": str(args.selection_reason),
            "source": source,
            "positions_path": str(positions_path.resolve()),
            "boundary": {
                "actor": int(boundary.current_actor),
                "actor_index": int(boundary.actor_index),
                "deck": sorted(int(x) for x in boundary.deck),
                "current_row": [int(x) for x in boundary.current_row],
                "legal_action_indices": sorted(_legal_action_map(boundary)),
            },
        })

    max_cells = int(args.max_cells)
    max_total_secs = float(args.max_total_secs)
    started = time.perf_counter()
    solved_this_run = 0
    timeouts: list[dict[str, Any]] = []
    actor = int(boundary.current_actor)
    action_rows: list[dict[str, Any]] = []
    stop = False

    for action_idx, action in sorted(_legal_action_map(boundary).items()):
        cells = []
        for ordinal, (tail, probability) in enumerate(conditioned_tails(boundary, action)):
            row = tuple(int(x) for x in tail.current_row)
            tail_key = public_state_key(tail)
            cell_path = _cell_path(cells_dir, action_idx, row)
            cell = _read_cell(cell_path, tail_key)
            if cell is None:
                elapsed_total = time.perf_counter() - started
                if (max_cells > 0 and solved_this_run >= max_cells) or (
                    max_total_secs > 0.0 and elapsed_total >= max_total_secs
                ):
                    stop = True
                    break
                call_started = time.perf_counter()
                value0, solved = exact_endgame_value(
                    tail,
                    max_secs=float(args.max_secs_per_tail),
                    rng=random.Random(int(args.seed) + 1009 * ordinal + action_idx),
                    score_scale=utility.score_scale,
                    margin_gain=utility.margin_gain,
                    alpha=utility.alpha,
                )
                elapsed = time.perf_counter() - call_started
                if not solved or not math.isfinite(float(value0)):
                    timeouts.append({
                        "action_idx": action_idx,
                        "row": list(row),
                        "elapsed_seconds": elapsed,
                    })
                    stop = bool(args.stop_on_timeout)
                    if stop:
                        break
                    continue
                cell = {
                    "schema_version": SCHEMA_VERSION,
                    "oracle_id": identity["oracle_id"],
                    "action_idx": action_idx,
                    "row": list(row),
                    "probability": float(probability),
                    "tail_state_key": tail_key,
                    "tail_chance_key_hex": chance_public_state_key_v1(tail).hex(),
                    "value_player0": float(value0),
                    "solved": True,
                    "elapsed_seconds": elapsed,
                    "max_secs_per_tail": float(args.max_secs_per_tail),
                }
                _atomic_json(cell_path, cell)
                solved_this_run += 1
            cells.append(cell)
        complete = len(cells) == EXPECTED_ROWS
        expected0 = None
        if complete:
            expected0 = sum(
                float(cell["value_player0"]) * float(cell["probability"])
                for cell in cells
            )
        action_rows.append({
            "action_idx": action_idx,
            "rows_solved": len(cells),
            "rows_expected": EXPECTED_ROWS,
            "complete": complete,
            "expected_value_player0": expected0,
            "expected_value_actor": (
                None if expected0 is None else (expected0 if actor == 0 else -expected0)
            ),
            "tail_seconds": sum(float(cell["elapsed_seconds"]) for cell in cells),
        })
        if stop:
            break

    complete_actions = [row for row in action_rows if row["complete"]]
    complete = len(complete_actions) == len(_legal_action_map(boundary))
    ranking = sorted(
        complete_actions,
        key=lambda row: (-float(row["expected_value_actor"]), int(row["action_idx"])),
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "oracle_id": identity["oracle_id"],
        "status": "complete" if complete else "incomplete",
        "selection_reason": str(args.selection_reason),
        "position_index": int(args.position_index),
        "prefix_actions": prefix,
        "boundary_actor": actor,
        "expected_rows_per_action": EXPECTED_ROWS,
        "legal_actions": len(_legal_action_map(boundary)),
        "complete_actions": len(complete_actions),
        "solved_cells_this_run": solved_this_run,
        "timeouts_this_run": timeouts,
        "run_elapsed_seconds": time.perf_counter() - started,
        "actions": action_rows,
        "ranking": ranking,
    }
    _atomic_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positions-path",
        default="runs/kingdomino/denial_search/signal_positions.jsonl",
    )
    parser.add_argument("--position-index", type=int, required=True)
    parser.add_argument(
        "--prefix-actions",
        required=True,
        help="Comma-separated canonical action indices reaching the final pre-reveal actor",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-reason", required=True)
    parser.add_argument("--score-scale", type=float, default=160.0)
    parser.add_argument("--margin-gain", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-secs-per-tail", type=float, default=60.0)
    parser.add_argument(
        "--max-total-secs",
        type=float,
        default=0.0,
        help="Stop before starting another cell after this wall time; 0 disables",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=0,
        help="Solve at most this many new cells; 0 disables",
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--continue-on-timeout",
        dest="stop_on_timeout",
        action="store_false",
        help="Continue to other cells after an exact-solver timeout",
    )
    parser.set_defaults(stop_on_timeout=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_oracle(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
