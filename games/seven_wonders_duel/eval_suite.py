"""Reusable arena evaluation for 7WD checkpoints.

Promoted from the throwaway script that produced ``postrun_eval`` for run 02.
Two properties that script lacked and that made its output hard to trust:

* **Fingerprinted resume.**  The old script skipped any match whose *label*
  already appeared in ``summary.json``, so a summary could silently mix results
  produced under different sims, seeds or checkpoints.  Every match now records
  the settings it ran under and a resume only skips a match whose fingerprint
  still matches.

* **A symmetry self-check.**  Arena games are played with
  ``deterministic_actions=True`` and paired seeds, so running ``A vs B`` and
  ``B vs A`` at the same seed offset replays the *same* physical games with the
  roles relabelled.  The two score rates must therefore sum to exactly 1.0.
  Any deviation is a seat/scoring bug, and it is worth knowing that before
  reading anything into a match result.

The suite deliberately keeps model-vs-model and model-vs-bot matches side by
side.  Self-play strength and out-of-distribution strength can move in opposite
directions -- run 02's iteration 11 beat iteration 0 head to head while losing
ground against two of the five scripted bots -- so a promotion number alone
does not establish that a checkpoint got better.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Sequence

from .bots import GreedyBot
from .phase_d import (
    CURRICULUM_BOT_TYPES,
    BotAgentSpec,
    PhaseDConfig,
    PhaseDLoop,
)

BOT_TYPES = (GreedyBot, *CURRICULUM_BOT_TYPES)
BOT_BY_NAME = {bot_type.name: bot_type for bot_type in BOT_TYPES}


class _NeverStops:
    """Collector for the Rust gate helpers that plays every scheduled game."""

    def update(self, _score: float):
        return SimpleNamespace(decision="continue")


@dataclass(frozen=True, slots=True)
class ArenaSettings:
    """Everything that changes what a match measures."""

    work_dir: str = "runs/seven_wonders_duel/eval_suite"
    sims: int = 64
    seed: int = 20260724
    device: str = "cuda"
    d_model: int = 128
    layers: int = 4
    top_k: int = 16
    search_mode: str = "closed"
    age_deal_samples: int = 32
    force_root_chance: bool = True

    def loop(self, games: int) -> PhaseDLoop:
        return PhaseDLoop(
            PhaseDConfig(
                run_dir=self.work_dir,
                seed=self.seed,
                seed_games=0,
                device=self.device,
                gate_backend="rust",
                gate_sims=self.sims,
                gate_max_games=games,
                promotion_every=0,  # no gate here; suppresses the power warning
                rust_slots=16,
                rust_global_batch_cap=256,
                rust_max_inflight_batches=1,
                rust_scheduler_workers=1,
                leaf_batch=1,
                force_root_chance=self.force_root_chance,
                age_deal_samples=self.age_deal_samples,
                d_model=self.d_model,
                layers=self.layers,
                top_k=self.top_k,
                search_mode=self.search_mode,
            )
        )

def fingerprint(
    settings: ArenaSettings,
    *,
    games: int,
    offset: int,
    participants: Sequence[tuple[str, str]],
) -> str:
    """Stable digest of the settings and inputs a match result depends on.

    ``participants`` are ``(role, identity)`` pairs -- a checkpoint sha256 for a
    model, the bot name for a bot -- so replacing a checkpoint file under the
    same label invalidates the cached result instead of silently keeping it.
    """

    payload = {
        "settings": {
            "sims": settings.sims,
            "seed": settings.seed,
            "d_model": settings.d_model,
            "layers": settings.layers,
            "top_k": settings.top_k,
            "search_mode": settings.search_mode,
            "age_deal_samples": settings.age_deal_samples,
            "force_root_chance": settings.force_root_chance,
        },
        "games": games,
        "offset": offset,
        "participants": [list(pair) for pair in participants],
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def _checkpoint_id(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def summarize(
    label: str,
    candidate: str,
    opponent: str,
    outcomes,
    elapsed: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Paired summary; the CI is over seed-pairs, not over individual games.

    Both legs of a pair share a seed and swap seats, so the games within a pair
    are strongly correlated and treating them as independent would understate
    the interval by roughly sqrt(2).
    """

    scores: list[float] = []
    seat_scores: dict[int, list[float]] = {0: [], 1: []}
    rows: list[dict[str, Any]] = []
    for index, outcome in enumerate(outcomes):
        candidate_seat = index % 2
        score = outcome.score_for(candidate_seat)
        scores.append(score)
        seat_scores[candidate_seat].append(score)
        rows.append(
            {
                "match": label,
                "candidate": candidate,
                "opponent": opponent,
                "game_index": index,
                "pair_index": index // 2,
                "candidate_seat": candidate_seat,
                "seed": outcome.seed,
                "first_player": outcome.first_player,
                "winner": outcome.winner,
                "score": score,
                "victory_type": outcome.victory_type,
                "actions": outcome.actions,
            }
        )
    pair_scores = [
        (scores[i] + scores[i + 1]) / 2.0 for i in range(0, len(scores) - 1, 2)
    ]
    mean = sum(pair_scores) / len(pair_scores)
    if len(pair_scores) > 1:
        variance = sum((v - mean) ** 2 for v in pair_scores) / (len(pair_scores) - 1)
        half = 1.96 * math.sqrt(variance / len(pair_scores))
    else:
        half = 0.0
    summary = {
        "match": label,
        "candidate": candidate,
        "opponent": opponent,
        "games": len(scores),
        "pairs": len(pair_scores),
        "wins": sum(s == 1.0 for s in scores),
        "losses": sum(s == 0.0 for s in scores),
        "draws": sum(s == 0.5 for s in scores),
        "score_rate": sum(scores) / len(scores),
        "paired_95_ci": [max(0.0, mean - half), min(1.0, mean + half)],
        "candidate_as_seat0": sum(seat_scores[0]) / max(len(seat_scores[0]), 1),
        "candidate_as_seat1": sum(seat_scores[1]) / max(len(seat_scores[1]), 1),
        "victory_types": dict(Counter(r["victory_type"] for r in rows)),
        "elapsed_seconds": elapsed,
        "games_per_second": len(scores) / elapsed if elapsed else 0.0,
    }
    return summary, rows


def run_model_match(
    settings: ArenaSettings,
    label: str,
    candidate: tuple[str, Path],
    opponent: tuple[str, Path],
    games: int,
    offset: int,
):
    loop = settings.loop(games)
    candidate_name, candidate_path = candidate
    opponent_name, opponent_path = opponent
    candidate_spec = loop._model_agent_spec(candidate_path, candidate_name)
    opponent_spec = loop._model_agent_spec(opponent_path, opponent_name)
    started = time.time()
    outcomes = loop._rust_model_gate_waves(
        candidate_spec, opponent_spec, _NeverStops(), offset
    )
    return summarize(
        label, candidate_name, opponent_name, outcomes, time.time() - started
    )


def run_bot_match(
    settings: ArenaSettings,
    label: str,
    candidate: tuple[str, Path],
    bot_name: str,
    games: int,
    offset: int,
):
    loop = settings.loop(games)
    candidate_name, candidate_path = candidate
    candidate_spec = loop._model_agent_spec(candidate_path, candidate_name)
    bot = BOT_BY_NAME[bot_name]()
    started = time.time()
    outcomes = loop._rust_bot_gate_waves(
        candidate_spec, BotAgentSpec(bot), _NeverStops(), offset
    )
    return summarize(label, candidate_name, bot.name, outcomes, time.time() - started)


def symmetry_check(
    settings: ArenaSettings,
    a: tuple[str, Path],
    b: tuple[str, Path],
    *,
    games: int = 32,
    offset: int = 99_000_000,
) -> dict[str, Any]:
    """A_vs_B + B_vs_A must equal exactly 1.0 under deterministic actions."""

    forward, _ = run_model_match(settings, "symmetry_forward", a, b, games, offset)
    reverse, _ = run_model_match(settings, "symmetry_reverse", b, a, games, offset)
    total = forward["score_rate"] + reverse["score_rate"]
    return {
        "forward": forward["score_rate"],
        "reverse": reverse["score_rate"],
        "sum": total,
        "games": games,
        "passed": abs(total - 1.0) < 1e-9,
    }


class EvalSuite:
    """Runs a set of matches into ``summary.json`` / ``games.jsonl``."""

    def __init__(self, output_dir: Path, settings: ArenaSettings):
        self.output_dir = Path(output_dir)
        self.settings = settings
        self.summary_path = self.output_dir / "summary.json"
        self.games_path = self.output_dir / "games.jsonl"

    def _load(self) -> list[dict[str, Any]]:
        if not self.summary_path.exists():
            return []
        payload = json.loads(self.summary_path.read_text(encoding="utf-8"))
        return payload.get("matches", [])

    def _write(self, matches: list[dict[str, Any]], extra: dict[str, Any]) -> None:
        self.summary_path.write_text(
            json.dumps({"matches": matches, **extra}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def run(
        self,
        jobs: Sequence[dict[str, Any]],
        *,
        symmetry: dict[str, Any] | None = None,
        log=print,
    ) -> list[dict[str, Any]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        matches = self._load()
        cached = {
            row["match"]: row.get("fingerprint")
            for row in matches
            if row.get("fingerprint")
        }
        extra = {"symmetry_check": symmetry} if symmetry else {}
        with self.games_path.open("a", encoding="utf-8", newline="\n") as handle:
            for index, job in enumerate(jobs, start=1):
                label = job["label"]
                mark = job["fingerprint"]
                if cached.get(label) == mark:
                    log(f"[{index}/{len(jobs)}] {label}: cached (fingerprint match)")
                    continue
                if label in cached:
                    log(
                        f"[{index}/{len(jobs)}] {label}: settings changed since the "
                        "cached result; re-running"
                    )
                    matches = [row for row in matches if row["match"] != label]
                log(f"[{index}/{len(jobs)}] {label}: {job['games']} games")
                if job["kind"] == "model":
                    summary, rows = run_model_match(
                        self.settings,
                        label,
                        job["candidate"],
                        job["opponent"],
                        job["games"],
                        job["offset"],
                    )
                else:
                    summary, rows = run_bot_match(
                        self.settings,
                        label,
                        job["candidate"],
                        job["opponent"],
                        job["games"],
                        job["offset"],
                    )
                summary["fingerprint"] = mark
                summary["sims"] = self.settings.sims
                matches.append(summary)
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                self._write(matches, extra)
                low, high = summary["paired_95_ci"]
                log(
                    f"  {summary['wins']}-{summary['losses']}-{summary['draws']} "
                    f"score={summary['score_rate']:.1%} "
                    f"paired95=[{low:.1%}, {high:.1%}] "
                    f"speed={summary['games_per_second']:.3f}/s"
                )
        self._write(matches, extra)
        return matches


def build_jobs(
    settings: ArenaSettings,
    checkpoints: dict[str, Path],
    *,
    model_games: int,
    bot_games: int,
    bots: Sequence[str],
) -> list[dict[str, Any]]:
    """Round-robin among the checkpoints, plus every checkpoint against bots."""

    identity = {name: _checkpoint_id(path) for name, path in checkpoints.items()}
    names = list(checkpoints)
    jobs: list[dict[str, Any]] = []
    offset = 70_000_000
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            jobs.append(
                {
                    "kind": "model",
                    "label": f"{a}_vs_{b}",
                    "candidate": (a, checkpoints[a]),
                    "opponent": (b, checkpoints[b]),
                    "games": model_games,
                    "offset": offset,
                    "fingerprint": fingerprint(
                        settings,
                        games=model_games,
                        offset=offset,
                        participants=[("model", identity[a]), ("model", identity[b])],
                    ),
                }
            )
            offset += 1_000_000
    offset = 80_000_000
    for name in names:
        for bot_name in bots:
            jobs.append(
                {
                    "kind": "bot",
                    "label": f"{name}_vs_{bot_name}",
                    "candidate": (name, checkpoints[name]),
                    "opponent": bot_name,
                    "games": bot_games,
                    "offset": offset,
                    "fingerprint": fingerprint(
                        settings,
                        games=bot_games,
                        offset=offset,
                        participants=[("model", identity[name]), ("bot", bot_name)],
                    ),
                }
            )
            offset += 100_000
    return jobs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a labelled checkpoint to evaluate; repeat for a round robin",
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--model-games", type=int, default=400)
    parser.add_argument("--bot-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--bots",
        nargs="*",
        default=[bot_type.name for bot_type in BOT_TYPES],
        choices=[bot_type.name for bot_type in BOT_TYPES],
    )
    parser.add_argument(
        "--skip-symmetry-check",
        action="store_true",
        help="skip the A_vs_B + B_vs_A == 1 arena self-check",
    )
    args = parser.parse_args(argv)

    checkpoints: dict[str, Path] = {}
    for entry in args.checkpoint:
        if "=" not in entry:
            parser.error(f"--checkpoint expects NAME=PATH, got {entry!r}")
        name, _, raw = entry.partition("=")
        path = Path(raw)
        if not path.is_file():
            parser.error(f"checkpoint not found: {path}")
        checkpoints[name] = path
    if len(checkpoints) < 1:
        parser.error("need at least one checkpoint")

    output_dir = Path(args.out)
    settings = ArenaSettings(
        work_dir=str(output_dir),
        sims=args.sims,
        seed=args.seed,
        device=args.device,
        d_model=args.d_model,
        layers=args.layers,
    )

    symmetry = None
    if not args.skip_symmetry_check and len(checkpoints) >= 2:
        names = list(checkpoints)
        a = (names[0], checkpoints[names[0]])
        b = (names[1], checkpoints[names[1]])
        print("arena symmetry check (A_vs_B + B_vs_A must equal 1.0)...")
        symmetry = symmetry_check(settings, a, b)
        status = "PASS" if symmetry["passed"] else "FAIL"
        print(
            f"  {symmetry['forward']:.4f} + {symmetry['reverse']:.4f} = "
            f"{symmetry['sum']:.4f}  [{status}]"
        )
        if not symmetry["passed"]:
            print(
                "  arena seat/scoring bookkeeping is broken; match results below "
                "are not trustworthy"
            )

    jobs = build_jobs(
        settings,
        checkpoints,
        model_games=args.model_games,
        bot_games=args.bot_games,
        bots=args.bots,
    )
    suite = EvalSuite(output_dir, settings)
    suite.run(jobs, symmetry=symmetry)
    print(f"Complete: {suite.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
