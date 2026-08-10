"""Analyze exact deck-8 boundary cells as a fixed calibration corpus."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from games.kingdomino.denial_signal_sweep import file_sha256


SCHEMA_VERSION = "kd-deck8-oracle-corpus-v2"


def balanced_panel(bag: list[int], exposure: int, rng: random.Random) -> list[tuple[int, ...]]:
    if len(bag) != 8 or exposure <= 0:
        raise ValueError("balanced deck-8 panel requires eight tiles and positive exposure")
    rows = []
    for _ in range(exposure):
        shuffled = list(bag)
        rng.shuffle(shuffled)
        rows.extend((tuple(sorted(shuffled[:4])), tuple(sorted(shuffled[4:]))))
    return rows


def iid_panel(bag: list[int], rows: int, rng: random.Random) -> list[tuple[int, ...]]:
    if len(bag) != 8 or rows <= 0:
        raise ValueError("IID deck-8 panel requires eight tiles and positive width")
    return [tuple(sorted(rng.sample(bag, 4))) for _ in range(rows)]


def exhaustive_panel(bag: list[int]) -> list[tuple[int, ...]]:
    if len(bag) != 8:
        raise ValueError("exhaustive deck-8 panel requires eight tiles")
    return [tuple(row) for row in itertools.combinations(sorted(bag), 4)]


def _choose(values: dict[int, float], rng: random.Random) -> int:
    best = max(values.values())
    tied = sorted(
        idx
        for idx, value in values.items()
        if math.isclose(value, best, rel_tol=0.0, abs_tol=1e-12)
    )
    return rng.choice(tied)


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _load_oracle(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError(f"incomplete oracle: {summary_path}")
    actor = int(summary["boundary_actor"])
    exact = {
        int(row["action_idx"]): float(row["expected_value_actor"])
        for row in summary["actions"]
    }
    cells: dict[int, dict[tuple[int, ...], float]] = {idx: {} for idx in exact}
    for path in (summary_path.parent / "cells").glob("*.json"):
        cell = json.loads(path.read_text(encoding="utf-8"))
        if cell.get("schema_version") != summary.get("schema_version"):
            raise ValueError(f"cell schema mismatch: {path}")
        if cell.get("oracle_id") != summary.get("oracle_id"):
            raise ValueError(f"cell oracle identity mismatch: {path}")
        if cell.get("solved") is not True:
            raise ValueError(f"unsolved cell present in completed oracle: {path}")
        action_idx = int(cell["action_idx"])
        if action_idx not in cells:
            raise ValueError(f"cell action {action_idx} absent from summary: {path}")
        row = tuple(int(x) for x in cell["row"])
        if row in cells[action_idx]:
            raise ValueError(f"duplicate cell row for action {action_idx}: {path}")
        value0 = float(cell["value_player0"])
        cells[action_idx][row] = value0 if actor == 0 else -value0
    support = None
    for action_idx, row_values in cells.items():
        if len(row_values) != 70:
            raise ValueError(f"action {action_idx} has {len(row_values)} rows in {summary_path}")
        action_support = set(row_values)
        if support is None:
            support = action_support
        elif action_support != support:
            raise ValueError(f"row support differs by action in {summary_path}")
        reconstructed = statistics.fmean(row_values.values())
        if not math.isclose(reconstructed, exact[action_idx], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"cell mean {reconstructed} does not reproduce action {action_idx} "
                f"oracle value {exact[action_idx]} in {summary_path}"
            )
    if support is None:
        raise ValueError(f"oracle has no cells: {summary_path}")
    bag = sorted(set().union(*support))
    if len(bag) != 8:
        raise ValueError(f"oracle support does not imply an eight-tile bag: {summary_path}")
    return {
        "summary": summary,
        "summary_path": summary_path,
        "exact": exact,
        "cells": cells,
        "bag": bag,
    }


def _panel_trial(
    oracle: dict[str, Any],
    rows: list[tuple[int, ...]],
    rng: random.Random,
) -> dict[str, Any]:
    estimates = {
        action_idx: statistics.fmean(row_values[row] for row in rows)
        for action_idx, row_values in oracle["cells"].items()
    }
    selected = _choose(estimates, rng)
    exact = oracle["exact"]
    exact_best = max(exact.values())
    exact_best_set = {
        idx
        for idx, value in exact.items()
        if math.isclose(value, exact_best, rel_tol=0.0, abs_tol=1e-12)
    }
    return {
        "selected": selected,
        "top1_exact": selected in exact_best_set,
        "regret": exact_best - exact[selected],
    }


def _search_scores(screen_boundary: dict[str, Any], exact: dict[int, float]) -> dict[str, Any]:
    exact_best = max(exact.values())
    exact_best_set = {
        idx
        for idx, value in exact.items()
        if math.isclose(value, exact_best, rel_tol=0.0, abs_tol=1e-12)
    }
    out = {}
    for arm_name, arm in screen_boundary["arms"].items():
        actions = [int(x) for x in arm["selection"]["actions"]]
        rows = []
        for selected in actions:
            if selected not in exact:
                raise ValueError(f"screen action {selected} absent from exact oracle")
            rows.append({
                "selected_action_idx": selected,
                "top1_exact": selected in exact_best_set,
                "exact_regret_actor": exact_best - exact[selected],
            })
        out[arm_name] = {
            "searches": len(rows),
            "top1_exact_rate": statistics.fmean(float(row["top1_exact"]) for row in rows),
            "mean_exact_regret_actor": statistics.fmean(row["exact_regret_actor"] for row in rows),
            "max_exact_regret_actor": max(row["exact_regret_actor"] for row in rows),
            "rows": rows,
        }
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"corpus output already exists: {output}")
    screen_path = Path(args.screen)
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    x0_boundaries = {}
    x0_boundary_keys: dict[int, str] = {}
    for boundary in screen["boundaries"]:
        for origin in boundary["origins"]:
            if origin["line_arm"] == "x0":
                position_index = int(origin["position_index"])
                boundary_key = str(boundary["boundary_key_hex"])
                if position_index in x0_boundaries:
                    raise ValueError(f"multiple x0 boundaries for position {position_index}")
                x0_boundaries[position_index] = boundary
                x0_boundary_keys[position_index] = boundary_key
    if len(set(x0_boundary_keys.values())) != len(x0_boundary_keys):
        collisions: dict[str, list[int]] = {}
        for position_index, boundary_key in x0_boundary_keys.items():
            collisions.setdefault(boundary_key, []).append(position_index)
        repeated = {key: rows for key, rows in collisions.items() if len(rows) > 1}
        raise ValueError(
            "x0 calibration positions are not independent public boundaries: "
            f"{repeated}"
        )

    summary_paths = list(Path(args.calibration_dir).glob("position_*_x0/summary.json"))
    summary_paths.append(Path(args.position11_summary))
    oracles = [_load_oracle(path) for path in summary_paths]
    by_position = {int(item["summary"]["position_index"]): item for item in oracles}
    if len(by_position) != len(oracles):
        raise ValueError("multiple oracle summaries claim the same position index")
    expected_positions = set(x0_boundaries)
    if set(by_position) != expected_positions:
        raise ValueError(
            f"oracle positions {sorted(by_position)} != screen positions {sorted(expected_positions)}"
        )

    exposures = [int(x) for x in str(args.exposures).split(",") if x.strip()]
    trials = int(args.trials)
    if trials <= 0:
        raise ValueError("trials must be positive")
    positions = []
    for position_index in sorted(by_position):
        oracle = by_position[position_index]
        exact = oracle["exact"]
        exact_order = sorted(exact, key=lambda idx: (-exact[idx], idx))
        panel_results = {}
        for exposure in exposures:
            schemes = {}
            for scheme_offset, scheme in enumerate(("balanced", "iid")):
                rows = []
                for trial in range(trials):
                    rng = random.Random(
                        int(args.seed) + 1_000_003 * position_index
                        + 10_007 * exposure + 101 * scheme_offset + trial
                    )
                    panel = (
                        balanced_panel(oracle["bag"], exposure, rng)
                        if scheme == "balanced"
                        else iid_panel(oracle["bag"], 2 * exposure, rng)
                    )
                    trial_row = _panel_trial(oracle, panel, rng)
                    trial_row["distinct_rows"] = len(set(panel))
                    rows.append(trial_row)
                schemes[scheme] = {
                    "rows_per_panel": 2 * exposure,
                    "full_support_rows": 70,
                    "trials": trials,
                    "mean_distinct_rows_covered": statistics.fmean(
                        row["distinct_rows"] for row in rows
                    ),
                    "mean_distinct_support_fraction": statistics.fmean(
                        row["distinct_rows"] / 70.0 for row in rows
                    ),
                    "min_distinct_rows_covered": min(
                        row["distinct_rows"] for row in rows
                    ),
                    "max_distinct_rows_covered": max(
                        row["distinct_rows"] for row in rows
                    ),
                    "top1_exact_rate": statistics.fmean(float(row["top1_exact"]) for row in rows),
                    "mean_exact_regret_actor": statistics.fmean(row["regret"] for row in rows),
                    "p90_exact_regret_actor": _percentile([row["regret"] for row in rows], 0.9),
                }
            panel_results[str(exposure)] = schemes
        exact_panel = exhaustive_panel(oracle["bag"])
        exact_panel_trial = _panel_trial(
            oracle,
            exact_panel,
            random.Random(int(args.seed) + 7_919 * position_index),
        )
        positions.append({
            "position_index": position_index,
            "oracle_summary": str(oracle["summary_path"].resolve()),
            "oracle_sha256": file_sha256(oracle["summary_path"]),
            "actor": int(oracle["summary"]["boundary_actor"]),
            "legal_actions": len(exact),
            "exact_best_action_idx": exact_order[0],
            "exact_runner_up_action_idx": exact_order[1],
            "exact_top_gap_actor": exact[exact_order[0]] - exact[exact_order[1]],
            "search": _search_scores(x0_boundaries[position_index], exact),
            "panels": panel_results,
            "exhaustive_panel": {
                "rows_per_panel": 70,
                "distinct_rows_covered": 70,
                "support_fraction": 1.0,
                "top1_exact": bool(exact_panel_trial["top1_exact"]),
                "exact_regret_actor": float(exact_panel_trial["regret"]),
            },
        })

    aggregate_panels = {}
    for exposure in exposures:
        aggregate_panels[str(exposure)] = {}
        for scheme in ("balanced", "iid"):
            rows = [position["panels"][str(exposure)][scheme] for position in positions]
            aggregate_panels[str(exposure)][scheme] = {
                "positions": len(rows),
                "mean_position_top1_exact_rate": statistics.fmean(
                    row["top1_exact_rate"] for row in rows
                ),
                "mean_position_exact_regret_actor": statistics.fmean(
                    row["mean_exact_regret_actor"] for row in rows
                ),
                "mean_position_p90_regret_actor": statistics.fmean(
                    row["p90_exact_regret_actor"] for row in rows
                ),
                "mean_position_distinct_support_fraction": statistics.fmean(
                    row["mean_distinct_support_fraction"] for row in rows
                ),
            }
    aggregate_search = {}
    arm_sets = [set(position["search"]) for position in positions]
    if not arm_sets or any(arms != arm_sets[0] for arms in arm_sets[1:]):
        raise ValueError("screen arm names differ across calibration positions")
    for arm_name in sorted(arm_sets[0]):
        rows = [position["search"][arm_name] for position in positions]
        aggregate_search[arm_name] = {
            "positions": len(rows),
            "searches": sum(row["searches"] for row in rows),
            "mean_position_top1_exact_rate": statistics.fmean(
                row["top1_exact_rate"] for row in rows
            ),
            "mean_position_exact_regret_actor": statistics.fmean(
                row["mean_exact_regret_actor"] for row in rows
            ),
            "max_exact_regret_actor": max(row["max_exact_regret_actor"] for row in rows),
            "aggregation_unit": "strategic position; search seeds nested within position",
        }

    aggregate_exhaustive = {
        "positions": len(positions),
        "rows_per_position": 70,
        "distinct_support_fraction": 1.0,
        "top1_exact_rate": statistics.fmean(
            float(position["exhaustive_panel"]["top1_exact"])
            for position in positions
        ),
        "mean_exact_regret_actor": statistics.fmean(
            position["exhaustive_panel"]["exact_regret_actor"]
            for position in positions
        ),
    }

    result = {
        "schema_version": SCHEMA_VERSION,
        "scope": "eight-position fixed boundary calibration; repeated panels/search seeds are nested within position",
        "selection_reason": str(args.selection_reason),
        "screen": str(screen_path.resolve()),
        "screen_sha256": file_sha256(screen_path),
        "configuration": {
            "positions": sorted(by_position),
            "exposures": exposures,
            "trials_per_position_scheme_exposure": trials,
            "seed": int(args.seed),
        },
        "positions": positions,
        "aggregate": {
            "search": aggregate_search,
            "panels": aggregate_panels,
            "exhaustive_panel": aggregate_exhaustive,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--position11-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-reason", required=True)
    parser.add_argument("--exposures", default="1,2,4,8,16,35")
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
