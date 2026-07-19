"""Persistent lightweight Elo ledger for fixed anchors and checkpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable

from .core import MatchOutcome


class EloLedger:
    def __init__(
        self,
        directory: str | Path,
        *,
        initial_rating: float = 1000.0,
        k_factor: float = 24.0,
        fixed_ratings: dict[str, float] | None = None,
    ):
        self.directory = Path(directory)
        self.db_path = self.directory / "elo.json"
        self.games_path = self.directory / "elo_games.jsonl"
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.fixed_ratings = dict(fixed_ratings or {})

    def ratings(self) -> dict[str, float]:
        if not self.db_path.exists():
            return {}
        return {
            name: float(value)
            for name, value in json.loads(
                self.db_path.read_text(encoding="utf-8")
            ).get("ratings", {}).items()
        }

    def record(self, outcomes: Iterable[MatchOutcome]) -> dict[str, float]:
        rows = list(outcomes)
        ratings = {**self.ratings(), **self.fixed_ratings}
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.games_path.open("a", encoding="utf-8", newline="\n") as handle:
            for game in rows:
                a, b = game.agents
                ra = ratings.get(a, self.initial_rating)
                rb = ratings.get(b, self.initial_rating)
                expected_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
                actual_a = game.score_for(0)
                delta = self.k_factor * (actual_a - expected_a)
                if a not in self.fixed_ratings:
                    ratings[a] = ra + delta
                if b not in self.fixed_ratings:
                    ratings[b] = rb - delta
                handle.write(
                    json.dumps(
                        {
                            "seed": game.seed,
                            "p0": a,
                            "p1": b,
                            "winner": game.winner,
                            "score_p0": actual_a,
                            "victory_type": game.victory_type,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        payload = {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "ratings": dict(sorted(ratings.items())),
            "fixed_ratings": dict(sorted(self.fixed_ratings.items())),
        }
        temporary = self.db_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.db_path)
        return ratings

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))
