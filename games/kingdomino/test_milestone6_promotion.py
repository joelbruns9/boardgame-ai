from __future__ import annotations

import json
from pathlib import Path

import pytest

from games.kingdomino.network import KingdominoNet
from games.kingdomino.promotion import (
    FixedSuiteComparison,
    MatchStats,
    compare_fixed_suite,
    decide_promotion,
    promote_current_best,
    promotion_payload,
    wilson_lower_bound,
)
from games.kingdomino.self_play import (
    SelfPlayConfig,
    run_self_play_training,
    save_checkpoint,
)


def test_wilson_lcb_is_conservative_for_noisy_small_samples() -> None:
    assert wilson_lower_bound(11.0, 20) < 0.50
    assert wilson_lower_bound(240.0, 400) > 0.50


def test_promotion_requires_win_rate_lcb_and_fixed_suite() -> None:
    match = MatchStats(
        games=20,
        wins=11,
        losses=9,
        draws=0,
        points=11.0,
        win_rate=0.55,
        lower_confidence_bound=wilson_lower_bound(11.0, 20),
        mean_margin=1.0,
    )
    fixed = FixedSuiteComparison(
        checked=True,
        passed=True,
        tolerance=0.05,
        reason="no fixed-suite regression",
    )
    decision = decide_promotion(match, fixed, min_win_rate=0.55, min_lcb=0.50)
    assert not decision.passed
    assert any("LCB" in reason for reason in decision.reasons)

    strong = MatchStats(
        games=400,
        wins=240,
        losses=160,
        draws=0,
        points=240.0,
        win_rate=0.60,
        lower_confidence_bound=wilson_lower_bound(240.0, 400),
        mean_margin=4.0,
    )
    decision = decide_promotion(strong, fixed, min_win_rate=0.55, min_lcb=0.50)
    assert decision.passed


def test_fixed_suite_regression_blocks_promotion() -> None:
    baseline = {"mean_abs_exact_value_error": 0.20}
    candidate = {"mean_abs_exact_value_error": 0.28}
    cmp = compare_fixed_suite(candidate, baseline, tolerance=0.05)
    assert cmp.checked
    assert not cmp.passed
    assert cmp.delta_mean_abs_exact_value_error == pytest.approx(0.08)


def test_promote_current_best_writes_audit_files_and_backup(tmp_path: Path) -> None:
    best_dir = tmp_path / "best_checkpoint"
    old_best = best_dir / "current_best.pt"
    candidate = tmp_path / "candidate.pt"
    old_best.parent.mkdir(parents=True)
    old_best.write_bytes(b"old")
    candidate.write_bytes(b"new")

    decision = decide_promotion(
        None,
        FixedSuiteComparison(False, True, 0.05, reason="skipped"),
        bootstrap=True,
    )
    payload = promotion_payload(
        candidate=candidate,
        current_best=old_best,
        decision=decision,
    )
    promoted = promote_current_best(
        candidate,
        best_dir=best_dir,
        current_best=old_best,
        payload=payload,
    )

    assert promoted == old_best
    assert old_best.read_bytes() == b"new"
    meta = json.loads((best_dir / "current_best.json").read_text(encoding="utf-8"))
    assert meta["promoted_sha256"]
    assert Path(meta["previous_current_best_backup"]).exists()
    assert (best_dir / "promotion_log.jsonl").exists()


def test_gated_selfplay_smoke_uses_current_best_generator(tmp_path: Path) -> None:
    current_best = tmp_path / "best_checkpoint" / "current_best.pt"
    checkpoint_dir = tmp_path / "run"
    base_cfg = SelfPlayConfig(
        channels=8,
        blocks=1,
        bilinear_dim=8,
        device="cpu",
        exact_endgame_max_secs=0.0,
        benchmark_every=0,
        elo_every=0,
    )
    current_best.parent.mkdir(parents=True)
    best_net = KingdominoNet(
        channels=base_cfg.channels,
        blocks=base_cfg.blocks,
        bilinear_dim=base_cfg.bilinear_dim,
    )
    save_checkpoint(str(current_best), best_net, base_cfg, 0, {"benchmark": []})

    cfg = SelfPlayConfig(
        channels=8,
        blocks=1,
        bilinear_dim=8,
        n_iterations=1,
        games_per_iteration=1,
        train_steps_per_iteration=0,
        min_buffer_to_train=999999,
        n_simulations=1,
        engine="open_loop",
        device="cpu",
        exact_endgame_max_secs=0.0,
        benchmark_every=0,
        elo_every=0,
        checkpoint_dir=str(checkpoint_dir),
        log_path=str(checkpoint_dir / "training_log.jsonl"),
        current_best_path=str(current_best),
        gated_selfplay=True,
        promotion_every=0,
    )

    result = run_self_play_training(cfg, verbose=False)
    assert len(result["buffer"]) > 0
    row = json.loads((checkpoint_dir / "training_log.jsonl").read_text().splitlines()[-1])
    assert row["gated_selfplay"] is True
    assert row["selfplay_source"] == str(current_best)
    assert row["promotion_checked"] is False
