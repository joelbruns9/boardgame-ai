"""Measure how often consecutive own decisions share a tile-reveal window.

This is an offline decision aid for STREAMING_ADVISOR_PLAN.md Item 3. It does
not retain or reuse search state.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


TABLE_RE = re.compile(r"(?:[?&]table=)(\d+)")


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _game_id(record: dict[str, Any], source: Path) -> str:
    table = record.get("table_id")
    if table not in (None, "", "unknown"):
        return str(table)
    match = TABLE_RE.search(str(record.get("url") or ""))
    return match.group(1) if match else source.stem


def _window_key(state: dict[str, Any]) -> tuple[Any, ...] | None:
    """Fingerprint information that changes when BGA reveals the next row."""
    debug = state.get("debug") if isinstance(state.get("debug"), dict) else {}
    deck = debug.get("deck")
    if isinstance(deck, list):
        return ("hidden-deck", *sorted(int(tile) for tile in deck))
    deck_count = state.get("deck_count")
    if isinstance(deck_count, (int, float)):
        return ("deck-count", int(deck_count))
    return None


def load_own_decisions(paths: Iterable[Path]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    games: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_lines = 0
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    invalid_lines += 1
                    continue
                if not isinstance(record, dict) or record.get("kind") != "decision":
                    continue
                if str(record.get("viewer_id")) != str(record.get("active_player")):
                    continue
                state = record.get("state")
                timestamp = _timestamp(record.get("captured_at"))
                if not isinstance(state, dict) or timestamp is None:
                    continue
                window = _window_key(state)
                if window is None:
                    continue
                game_id = _game_id(record, path)
                identity = (game_id, timestamp.isoformat(), json.dumps(state, sort_keys=True))
                if identity in seen:
                    continue
                seen.add(identity)
                games[game_id].append({"timestamp": timestamp, "window": window})
    for decisions in games.values():
        decisions.sort(key=lambda item: item["timestamp"])
    return dict(games), invalid_lines


def summarize_game(game_id: str, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    window_sizes: list[int] = []
    recurring_gaps: list[float] = []
    all_gaps: list[float] = []
    recurring_pairs = 0
    current_window: tuple[Any, ...] | None = None
    current_size = 0
    for index, decision in enumerate(decisions):
        if decision["window"] != current_window:
            if current_size:
                window_sizes.append(current_size)
            current_window = decision["window"]
            current_size = 1
        else:
            current_size += 1
        if index:
            gap = max(0.0, (decision["timestamp"] - decisions[index - 1]["timestamp"]).total_seconds())
            all_gaps.append(gap)
            if decision["window"] == decisions[index - 1]["window"]:
                recurring_pairs += 1
                recurring_gaps.append(gap)
    if current_size:
        window_sizes.append(current_size)
    pairs = max(0, len(decisions) - 1)
    return {
        "game_id": game_id,
        "decisions": len(decisions),
        "windows": len(window_sizes),
        "recurring_pairs": recurring_pairs,
        "pairs": pairs,
        "recurrence_rate": recurring_pairs / pairs if pairs else 0.0,
        "decisions_per_window": statistics.mean(window_sizes) if window_sizes else 0.0,
        "max_decisions_per_window": max(window_sizes, default=0),
        "all_gaps": all_gaps,
        "recurring_gaps": recurring_gaps,
    }


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}s"


def format_report(summaries: list[dict[str, Any]], invalid_lines: int = 0) -> str:
    lines = ["Kingdomino within-reveal-window recurrence"]
    for item in summaries:
        lines.append(
            f"game {item['game_id']}: decisions={item['decisions']}, windows={item['windows']}, "
            f"recurrence={item['recurring_pairs']}/{item['pairs']} ({item['recurrence_rate']:.1%}), "
            f"decisions/window={item['decisions_per_window']:.2f} (max {item['max_decisions_per_window']}), "
            f"gap median/mean={_seconds(_median(item['all_gaps']))}/{_seconds(_mean(item['all_gaps']))}"
        )

    decisions = sum(item["decisions"] for item in summaries)
    windows = sum(item["windows"] for item in summaries)
    recurring = sum(item["recurring_pairs"] for item in summaries)
    pairs = sum(item["pairs"] for item in summaries)
    all_gaps = [gap for item in summaries for gap in item["all_gaps"]]
    recurring_gaps = [gap for item in summaries for gap in item["recurring_gaps"]]
    rate = recurring / pairs if pairs else 0.0
    lines.append(
        f"aggregate: games={len(summaries)}, decisions={decisions}, windows={windows}, "
        f"recurrence={recurring}/{pairs} ({rate:.1%}), "
        f"decisions/window={(decisions / windows if windows else 0.0):.2f}, "
        f"gap median/mean={_seconds(_median(all_gaps))}/{_seconds(_mean(all_gaps))}, "
        f"same-window gap median/mean={_seconds(_median(recurring_gaps))}/{_seconds(_mean(recurring_gaps))}"
    )
    if invalid_lines:
        lines.append(f"ignored malformed JSON lines: {invalid_lines}")
    if rate >= 0.5:
        lines.append(
            "verdict: recurrence is high enough to clear the measurement gate, but subtree reuse remains "
            "deferred until live play shows decisions routinely occur before the first fresh-search refresh converges."
        )
    else:
        lines.append(
            "verdict: recurrence is below the 50% measurement gate; keep cross-move subtree reuse deferred."
        )
    return "\n".join(lines)


def _input_files(raw_paths: list[str]) -> list[Path]:
    inputs = [Path(value) for value in raw_paths] if raw_paths else [Path("runs/kingdomino/bga_game_log")]
    files: list[Path] = []
    for path in inputs:
        files.extend(sorted(path.glob("*.jsonl")) if path.is_dir() else [path])
    return [path for path in files if path.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="JSONL files or directories (default: current BGA game logs)")
    args = parser.parse_args()
    games, invalid_lines = load_own_decisions(_input_files(args.paths))
    summaries = [summarize_game(game_id, games[game_id]) for game_id in sorted(games)]
    print(format_report(summaries, invalid_lines))


if __name__ == "__main__":
    main()
