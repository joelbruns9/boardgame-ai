"""Promotion-gate helpers for Kingdomino checkpoints.

Milestone 6 separates "latest learner" from "current best".  This module keeps
the statistical decision, audit metadata, and optional copy/update mechanics in
one place so standalone promotion and in-training gated self-play use the same
rules.
"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib

import torch

from games.kingdomino.network import KingdominoNet
from games.kingdomino.round_robin_eval import (
    checkpoint_config,
    checkpoint_state_dict,
)


DEFAULT_BEST_DIR = Path("runs/kingdomino/best_checkpoint")
DEFAULT_CURRENT_BEST = DEFAULT_BEST_DIR / "current_best.pt"
DEFAULT_FIXED_SUITE = Path("data/kingdomino/eval_suite_v1.jsonl")


@dataclass
class MatchStats:
    games: int
    wins: int
    losses: int
    draws: int
    points: float
    win_rate: float
    lower_confidence_bound: float
    mean_margin: float


@dataclass
class FixedSuiteComparison:
    checked: bool
    passed: bool
    tolerance: float
    candidate_mean_abs_exact_value_error: float | None = None
    baseline_mean_abs_exact_value_error: float | None = None
    delta_mean_abs_exact_value_error: float | None = None
    reason: str = ""


@dataclass
class PromotionDecision:
    passed: bool
    bootstrap: bool
    reasons: list[str]
    match: MatchStats | None
    fixed_suite: FixedSuiteComparison | None
    min_win_rate: float
    min_lcb: float


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_lower_bound(points: float, games: int, z: float = 1.96) -> float:
    """Wilson lower bound for a binomial score where draws count as 0.5."""
    games = int(games)
    if games <= 0:
        return 0.0
    p = max(0.0, min(1.0, float(points) / games))
    z2 = z * z
    denom = 1.0 + z2 / games
    centre = p + z2 / (2.0 * games)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * games)) / games)
    return max(0.0, (centre - margin) / denom)


def match_stats_from_pair(pair, *, z: float = 1.96) -> MatchStats:
    points = float(pair.a_wins) + 0.5 * float(pair.draws)
    games = int(pair.games)
    win_rate = points / games if games else 0.0
    return MatchStats(
        games=games,
        wins=int(pair.a_wins),
        losses=int(pair.b_wins),
        draws=int(pair.draws),
        points=points,
        win_rate=win_rate,
        lower_confidence_bound=wilson_lower_bound(points, games, z=z),
        mean_margin=float(pair.avg_margin_a),
    )


def evaluate_network_match(
    candidate_net: KingdominoNet,
    baseline_net: KingdominoNet,
    *,
    games: int,
    sims: int,
    device: str,
    batch_slots: int = 32,
    leaf_batch: int = 6,
    seed: int = 20260630,
    c_puct: float = 1.5,
    fpu: float = -0.2,
    margin_gain: float = 2.0,
    alpha: float = 0.8,
    z: float = 1.96,
) -> MatchStats:
    """Run a paired, seat-swapped open-loop match using the Elo evaluator."""
    from games.kingdomino.elo_rating import EloConfig, play_rating_games

    paired_seeds = max(1, int(math.ceil(int(games) / 2)))
    cfg = EloConfig(
        games_per_anchor=paired_seeds,
        sims=int(sims),
        device=str(device),
        n_slots=int(batch_slots),
        leaf_batch=int(leaf_batch),
        c_puct=float(c_puct),
        fpu=float(fpu),
        margin_gain=float(margin_gain),
        alpha=float(alpha),
        seed=int(seed),
        verbose=False,
    )
    candidate_net = candidate_net.to(device).eval()
    baseline_net = baseline_net.to(device).eval()
    pair, _games = play_rating_games(
        candidate_net, baseline_net,
        "candidate", "current_best",
        paired_seeds, int(seed), cfg,
    )
    return match_stats_from_pair(pair, z=z)


def _net_from_checkpoint(path: str | Path, device: str) -> KingdominoNet:
    ckpt = torch.load(path, map_location="cpu")
    cfg = checkpoint_config(ckpt)
    net = KingdominoNet(
        channels=int(cfg.get("channels", 96)),
        blocks=int(cfg.get("blocks", 8)),
        bilinear_dim=int(cfg.get("bilinear_dim", 64)),
        score_scale=float(cfg.get("score_scale", 160.0)),
    )
    net.load_state_dict(checkpoint_state_dict(ckpt))
    net.to(device)
    net.eval()
    return net


def evaluate_checkpoint_match(
    candidate: str | Path,
    baseline: str | Path,
    *,
    games: int,
    sims: int,
    device: str,
    batch_slots: int = 32,
    leaf_batch: int = 6,
    seed: int = 20260630,
    c_puct: float = 1.5,
    fpu: float = -0.2,
    margin_gain: float = 2.0,
    alpha: float = 0.8,
    z: float = 1.96,
) -> MatchStats:
    return evaluate_network_match(
        _net_from_checkpoint(candidate, device),
        _net_from_checkpoint(baseline, device),
        games=games,
        sims=sims,
        device=device,
        batch_slots=batch_slots,
        leaf_batch=leaf_batch,
        seed=seed,
        c_puct=c_puct,
        fpu=fpu,
        margin_gain=margin_gain,
        alpha=alpha,
        z=z,
    )


def fixed_suite_summary_for_net(
    net: KingdominoNet,
    *,
    suite: str | Path = DEFAULT_FIXED_SUITE,
    device: str = "cpu",
    checkpoint_label: str = "in_memory",
) -> dict[str, Any]:
    from scripts.run_eval_suite import load_suite, evaluate_record, summarize

    records = load_suite(Path(suite))
    net = net.to(device).eval()
    score_scale = float(getattr(net, "score_scale", 160.0) or 160.0)
    rows = [evaluate_record(rec, net, device, score_scale) for rec in records]
    return summarize(rows, Path(suite), checkpoint_label)


def fixed_suite_summary_for_checkpoint(
    checkpoint: str | Path,
    *,
    suite: str | Path = DEFAULT_FIXED_SUITE,
    device: str = "cpu",
) -> dict[str, Any]:
    return fixed_suite_summary_for_net(
        _net_from_checkpoint(checkpoint, device),
        suite=suite,
        device=device,
        checkpoint_label=str(checkpoint),
    )


def compare_fixed_suite(
    candidate_summary: dict[str, Any] | None,
    baseline_summary: dict[str, Any] | None,
    *,
    tolerance: float,
) -> FixedSuiteComparison:
    if candidate_summary is None or baseline_summary is None:
        return FixedSuiteComparison(
            checked=False,
            passed=True,
            tolerance=float(tolerance),
            reason="fixed-suite comparison skipped",
        )

    cand = candidate_summary.get("mean_abs_exact_value_error")
    base = baseline_summary.get("mean_abs_exact_value_error")
    if cand is None or base is None:
        return FixedSuiteComparison(
            checked=True,
            passed=True,
            tolerance=float(tolerance),
            candidate_mean_abs_exact_value_error=cand,
            baseline_mean_abs_exact_value_error=base,
            reason="no exact-valued fixed-suite positions to compare",
        )

    delta = float(cand) - float(base)
    passed = delta <= float(tolerance)
    return FixedSuiteComparison(
        checked=True,
        passed=passed,
        tolerance=float(tolerance),
        candidate_mean_abs_exact_value_error=float(cand),
        baseline_mean_abs_exact_value_error=float(base),
        delta_mean_abs_exact_value_error=delta,
        reason=("no fixed-suite regression"
                if passed else "fixed-suite exact-value error regressed"),
    )


def decide_promotion(
    match: MatchStats | None,
    fixed_suite: FixedSuiteComparison | None,
    *,
    min_win_rate: float = 0.55,
    min_lcb: float = 0.50,
    bootstrap: bool = False,
) -> PromotionDecision:
    reasons: list[str] = []
    passed = True

    if bootstrap:
        reasons.append("bootstrap promotion requested")
    else:
        if match is None:
            passed = False
            reasons.append("missing head-to-head match stats")
        else:
            if match.win_rate < min_win_rate:
                passed = False
                reasons.append(
                    f"win_rate {match.win_rate:.3f} < required {min_win_rate:.3f}")
            else:
                reasons.append(
                    f"win_rate {match.win_rate:.3f} >= required {min_win_rate:.3f}")
            if match.lower_confidence_bound <= min_lcb:
                passed = False
                reasons.append(
                    f"LCB {match.lower_confidence_bound:.3f} <= required {min_lcb:.3f}")
            else:
                reasons.append(
                    f"LCB {match.lower_confidence_bound:.3f} > required {min_lcb:.3f}")

    if fixed_suite is not None:
        if fixed_suite.passed:
            reasons.append(fixed_suite.reason)
        else:
            passed = False
            reasons.append(fixed_suite.reason)

    return PromotionDecision(
        passed=passed,
        bootstrap=bool(bootstrap),
        reasons=reasons,
        match=match,
        fixed_suite=fixed_suite,
        min_win_rate=float(min_win_rate),
        min_lcb=float(min_lcb),
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(out)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def atomic_copy(src: str | Path, dst: str | Path) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def promotion_payload(
    *,
    candidate: str | Path,
    current_best: str | Path,
    decision: PromotionDecision,
    candidate_fixed_summary: dict[str, Any] | None = None,
    baseline_fixed_summary: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate = Path(candidate)
    current_best = Path(current_best)
    payload = {
        "timestamp": now_iso(),
        "candidate": str(candidate),
        "candidate_sha256": sha256_file(candidate) if candidate.exists() else None,
        "current_best": str(current_best),
        "current_best_sha256": (
            sha256_file(current_best) if current_best.exists() else None
        ),
        "decision": asdict(decision),
        "candidate_fixed_suite": candidate_fixed_summary,
        "baseline_fixed_suite": baseline_fixed_summary,
    }
    if extra:
        payload.update(extra)
    return payload


def promote_current_best(
    candidate: str | Path,
    *,
    best_dir: str | Path = DEFAULT_BEST_DIR,
    current_best: str | Path | None = None,
    payload: dict[str, Any],
) -> Path:
    best_dir = Path(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    target = Path(current_best) if current_best is not None else best_dir / "current_best.pt"
    if target.exists():
        backup = best_dir / f"previous_current_best_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.pt"
        atomic_copy(target, backup)
        payload["previous_current_best_backup"] = str(backup)
        payload["previous_current_best_backup_sha256"] = sha256_file(backup)
    atomic_copy(candidate, target)
    payload["promoted_path"] = str(target)
    payload["promoted_sha256"] = sha256_file(target)
    write_json(best_dir / "current_best.json", payload)
    append_jsonl(best_dir / "promotion_log.jsonl", payload)
    return target
