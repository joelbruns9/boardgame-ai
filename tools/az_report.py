"""Generate a portable AlphaZero run report from schema-v1 or schema-v2 logs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from games.az_loop.stats import stats_from_row, wilson_interval


SEVEN_WONDERS_REFERENCE = {
    "civilian": 0.709,
    "scientific": 0.130,
    "military": 0.161,
}


def load_rows(run_dir: Path) -> list[dict[str, Any]]:
    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        return list(json.loads(manifest.read_text(encoding="utf-8")).get("iterations", []))
    log = run_dir / "training_log.jsonl"
    if log.exists():
        return [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    raise FileNotFoundError(f"no run_manifest.json or training_log.jsonl in {run_dir}")


def normalized(row: dict[str, Any]) -> dict[str, Any]:
    typed = stats_from_row(row)
    if typed is not None:
        payload = typed.to_dict()
        return {
            "iteration": int(row.get("iteration", 0)),
            "generation": payload["generation"],
            "outcomes": payload["outcomes"],
            "training": payload["training"],
            "gates": payload["gates"],
            "resources": payload["resources"],
            "model": payload["model"],
        }
    generation = row.get("generation_performance", {})
    performance = generation.get("performance", generation)
    summary = generation.get("summary", {})
    gate = row.get("promotion_gate")
    gates = []
    if gate:
        games = int(gate.get("games", 0))
        rate = float(gate.get("score_rate", 0.0))
        # V1 retained only the aggregate game rate, not pair outcomes. Use the
        # smaller pair count so the reconstructed interval is conservative
        # rather than pretending seat-swapped games were independent.
        pairs = games // 2
        lcb, ucb = wilson_interval(rate * pairs, pairs)
        gates.append(
            {
                "opponent": gate.get("opponent", ""),
                "score_rate": rate,
                "wilson_lcb": lcb,
                "wilson_ucb": ucb,
                "games": games,
                "pairs": pairs,
                "decision": gate.get("decision", ""),
                "stop_reason": "legacy_sprt",
            }
        )
    training = row.get("training_performance", {})
    steps = training.get("steps") or []
    latest = steps[-1] if steps else {}
    return {
        "iteration": int(row.get("iteration", 0)),
        "generation": {
            "games": int(row.get("generated_games", summary.get("games", 0))),
            "moves": int(summary.get("moves", 0)),
            "seconds": float(performance.get("seconds", 0.0)),
            "games_per_second": float(performance.get("games_per_second", 0.0)),
            "nn_rows": 0,
            "forced_row_share": 0.0,
            "mean_batch_size": 0.0,
            "opponent_mix": summary.get("game_kinds", {}),
        },
        "outcomes": {
            "terminal_reason": summary.get("victory_types", {}),
        },
        "training": {
            "validation_losses": latest.get("val", {}),
            "accuracies": latest.get("val", {}),
            "learning_rate": latest.get("lr"),
            "seconds": sum(float(item.get("secs", 0.0)) for item in steps),
        },
        "gates": gates,
        "resources": {},
        "model": {},
    }


def _mean(rows: list[float]) -> float | None:
    return statistics.fmean(rows) if rows else None


def _opponent_summary(counts: Counter[str]) -> dict[str, Any]:
    total = sum(counts.values())
    ordered = dict(sorted(counts.items()))
    return {
        "games": total,
        "counts": ordered,
        "rates": {
            key: value / total if total else 0.0
            for key, value in ordered.items()
        },
        # Categories are exclusive, while these operational shares intentionally
        # include the real HOF-vs-bot overlap on both axes.
        "realized_hof_share": (
            (counts["hof"] + counts["hof_bot"]) / total if total else None
        ),
        "realized_bot_share": (
            (counts["bot"] + counts["hof_bot"]) / total if total else None
        ),
    }


def _legacy_opponent_mix(raw: dict[str, int]) -> dict[str, int]:
    mapped: Counter[str] = Counter()
    aliases = {
        "self_play": "current_best",
        "mixed": "bot",
        "curriculum_seed": "bot",
        "league": "hof",
        "league_mixed": "hof_bot",
    }
    for name, count in raw.items():
        mapped[aliases.get(name, name)] += int(count)
    return dict(mapped)


def build_report(rows: list[dict[str, Any]], block_size: int) -> dict[str, Any]:
    data = [normalized(row) for row in rows]
    blocks = []
    for start in range(0, len(data), block_size):
        block = data[start : start + block_size]
        terminal: Counter[str] = Counter()
        opponents: Counter[str] = Counter()
        for row in block:
            terminal.update(row["outcomes"].get("terminal_reason", {}))
            opponents.update(
                _legacy_opponent_mix(
                    row["generation"].get("opponent_mix", {})
                )
            )
        games = sum(terminal.values())
        mix = {}
        for reason, count in sorted(terminal.items()):
            rate = count / games if games else 0.0
            error = 1.96 * math.sqrt(rate * (1.0 - rate) / games) if games else 0.0
            mix[reason] = {"count": count, "rate": rate, "error_95": error}
        blocks.append(
            {
                "iterations": [block[0]["iteration"], block[-1]["iteration"]],
                "games": games,
                "outcome_mix": mix,
                "opponent_mix": _opponent_summary(opponents),
                "games_per_second": _mean(
                    [
                        float(row["generation"].get("games_per_second", 0.0))
                        for row in block
                    ]
                ),
                "mean_batch_size": _mean(
                    [
                        float(row["generation"].get("mean_batch_size", 0.0))
                        for row in block
                    ]
                ),
                "validation_total": _mean(
                    [
                        float(row["training"]["validation_losses"]["total"])
                        for row in block
                        if row["training"].get("validation_losses", {}).get("total")
                        is not None
                    ]
                ),
                "peak_rss_bytes": max(
                    (
                        int(row["resources"].get("peak_rss_bytes", 0))
                        for row in block
                    ),
                    default=0,
                ),
            }
        )
    gates = [
        {"iteration": row["iteration"], **gate}
        for row in data
        for gate in row["gates"]
    ]
    curve = [
        {
            "iteration": row["iteration"],
            "games_per_second": row["generation"].get("games_per_second"),
            "mean_batch_size": row["generation"].get("mean_batch_size"),
            "forced_row_share": row["generation"].get("forced_row_share"),
            "peak_rss_bytes": row["resources"].get("peak_rss_bytes"),
            "validation_losses": row["training"].get("validation_losses", {}),
            "accuracies": row["training"].get("accuracies", {}),
        }
        for row in data
    ]
    quartile = max(1, len(data) // 4)
    early = _mean(
        [float(row["generation"].get("games_per_second", 0.0)) for row in data[:quartile]]
    )
    late = _mean(
        [float(row["generation"].get("games_per_second", 0.0)) for row in data[-quartile:]]
    )
    decay = None if not early else (late - early) / early
    totals: Counter[str] = Counter()
    opponent_totals: Counter[str] = Counter()
    for row in data:
        totals.update(row["outcomes"].get("terminal_reason", {}))
        opponent_totals.update(
            _legacy_opponent_mix(
                row["generation"].get("opponent_mix", {})
            )
        )
    total_games = sum(totals.values())
    reference = {
        key: {
            "observed": totals.get(key, 0) / total_games if total_games else None,
            "reference": value,
        }
        for key, value in SEVEN_WONDERS_REFERENCE.items()
    }
    return {
        "iterations": len(data),
        "schema_versions": sorted(
            {int(row.get("log_schema_version", 1)) for row in rows}
        ),
        "blocks": blocks,
        "gates": gates,
        "learning_and_resources": curve,
        "opponent_mix": _opponent_summary(opponent_totals),
        "throughput_diagnostic": {
            "early_quartile_games_per_second": early,
            "late_quartile_games_per_second": late,
            "relative_change": decay,
            "diagnostic_fields_present": {
                "mean_batch_size": any(
                    row["generation"].get("mean_batch_size") for row in data
                ),
                "forced_row_share": any(
                    row["generation"].get("forced_row_share") for row in data
                ),
                "rss": any(row["resources"].get("peak_rss_bytes") for row in data),
            },
        },
        "seven_wonders_reference_mix": reference,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero run report",
        "",
        f"Iterations: {report['iterations']}; schema versions: "
        f"{', '.join(map(str, report['schema_versions']))}.",
        "",
        "## Throughput",
        "",
    ]
    diagnostic = report["throughput_diagnostic"]
    change = diagnostic["relative_change"]
    lines.append(
        "Early/late quartile games/s: "
        f"{diagnostic['early_quartile_games_per_second']!s} / "
        f"{diagnostic['late_quartile_games_per_second']!s}"
        + (f" ({change:+.1%})." if change is not None else ".")
    )
    present = diagnostic["diagnostic_fields_present"]
    available = [name for name, exists in present.items() if exists]
    lines.append(
        "Diagnostic fields present: "
        + (", ".join(available) if available else "none (legacy log)")
        + "."
    )
    lines.extend(["", "## Outcome, learning, and resource blocks", ""])
    for block in report["blocks"]:
        mix = ", ".join(
            f"{name} {item['rate']:.1%} +/- {item['error_95']:.1%}"
            for name, item in block["outcome_mix"].items()
        )
        validation = block.get("validation_total")
        learning = (
            f"; validation total {validation:.4f}" if validation is not None else ""
        )
        resources = (
            f"; mean batch {block['mean_batch_size']:.1f}; "
            f"peak RSS {block['peak_rss_bytes'] / 1024**2:.0f} MiB"
        )
        opponents = ", ".join(
            f"{name} {rate:.1%}"
            for name, rate in block["opponent_mix"]["rates"].items()
        )
        lines.append(
            f"- Iterations {block['iterations'][0]}-{block['iterations'][1]}: "
            f"{block['games']} games; {mix}; {block['games_per_second']:.3f} games/s"
            f"{learning}{resources}; opponents {opponents or 'n/a'}."
        )
    opponent_mix = report["opponent_mix"]
    hof_share = opponent_mix["realized_hof_share"]
    bot_share = opponent_mix["realized_bot_share"]
    lines.extend(
        [
            "",
            "## Opponent attribution",
            "",
            "- Exclusive counts: "
            + (
                ", ".join(
                    f"{name} {count}"
                    for name, count in opponent_mix["counts"].items()
                )
                or "none"
            )
            + ".",
            "- Realized HOF share: "
            + ("n/a" if hof_share is None else f"{hof_share:.1%}")
            + "; realized curriculum-bot share: "
            + ("n/a" if bot_share is None else f"{bot_share:.1%}")
            + ".",
        ]
    )
    lines.extend(["", "## Gate ladder", ""])
    if not report["gates"]:
        lines.append("No gates recorded.")
    for gate in report["gates"]:
        lines.append(
            f"- Iteration {gate['iteration']}: {gate.get('opponent', '')}, "
            f"{gate.get('score_rate', 0.0):.1%} "
            f"[{gate.get('wilson_lcb', 0.0):.1%}, "
            f"{gate.get('wilson_ucb', 1.0):.1%}], "
            f"{gate.get('games', 0)} games, {gate.get('decision', '')} "
            f"({gate.get('stop_reason', '')})."
        )
    lines.extend(["", "## ZeusAI/BGA reference mix", ""])
    for name, item in report["seven_wonders_reference_mix"].items():
        observed = item["observed"]
        observed_text = "n/a" if observed is None else f"{observed:.1%}"
        lines.append(
            f"- {name}: observed {observed_text}; "
            f"reference {item['reference']:.1%}."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args(argv)
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    report = build_report(load_rows(args.run_dir), args.block_size)
    prefix = args.output_prefix or args.run_dir / "az_report"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prefix.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(prefix.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
