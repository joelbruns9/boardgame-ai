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
    _generator_action_after_promotion_check,
    _run_smart_elo_rating,
    run_self_play_training,
    save_checkpoint,
)


def _match_stats(win_rate: float, lcb: float, games: int = 100) -> MatchStats:
    wins = int(round(win_rate * games))
    losses = games - wins
    return MatchStats(
        games=games,
        wins=wins,
        losses=losses,
        draws=0,
        points=float(wins),
        win_rate=float(win_rate),
        lower_confidence_bound=float(lcb),
        mean_margin=(win_rate - 0.5) * 10.0,
    )


def _write_current_best(tmp_path: Path) -> tuple[Path, SelfPlayConfig]:
    current_best = tmp_path / "best_checkpoint" / "current_best.pt"
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
    return current_best, base_cfg


def _soft_gate_smoke_config(tmp_path: Path, current_best: Path) -> SelfPlayConfig:
    checkpoint_dir = tmp_path / "run"
    return SelfPlayConfig(
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
        hof_dir=str(tmp_path / "best_checkpoint" / "hof"),
        selfplay_generator_mode="soft_gate",
        promotion_every=1,
        promotion_games=32,
        promotion_sims=50,
        promotion_skip_fixed_suite=True,
        smart_elo=True,
        smart_elo_on_promote=True,
        smart_elo_games_per_anchor=32,
        smart_elo_sims=100,
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
    assert row["generator_mode"] == "strict_gate"
    assert row["generator_source"] == str(current_best)
    assert row["generator_checkpoint_path"] == str(current_best)
    assert row["generator_sha256"]
    assert row["promotion_checked"] is False


def test_soft_gate_phase1_uses_latest_generator_and_logs_state(tmp_path: Path) -> None:
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
        selfplay_generator_mode="soft_gate",
        promotion_every=0,
    )

    result = run_self_play_training(cfg, verbose=False)
    assert len(result["buffer"]) > 0
    row = json.loads((checkpoint_dir / "training_log.jsonl").read_text().splitlines()[-1])
    assert row["gated_selfplay"] is False
    assert row["selfplay_source"] == "learner_latest"
    assert row["generator_mode"] == "soft_gate"
    assert row["generator_source"] == "learner_latest"
    assert row["generator_checkpoint_path"] is None
    assert row["generator_sha256"] is None
    assert row["generator_baseline_source"] == str(current_best)
    assert row["generator_baseline_sha256"]
    assert row["generator_action"] == "initial"


def test_soft_gate_action_thresholds() -> None:
    weak = MatchStats(
        games=100,
        wins=47,
        losses=53,
        draws=0,
        points=47.0,
        win_rate=0.47,
        lower_confidence_bound=0.40,
        mean_margin=-2.0,
    )
    equal = MatchStats(
        games=100,
        wins=52,
        losses=48,
        draws=0,
        points=52.0,
        win_rate=0.52,
        lower_confidence_bound=0.45,
        mean_margin=0.5,
    )
    strong = MatchStats(
        games=100,
        wins=58,
        losses=42,
        draws=0,
        points=58.0,
        win_rate=0.58,
        lower_confidence_bound=0.51,
        mean_margin=3.0,
    )

    assert _generator_action_after_promotion_check(
        mode="soft_gate",
        match=weak,
        promotion_passed=False,
        revert_win_rate=0.48,
    ) == "revert"
    assert _generator_action_after_promotion_check(
        mode="soft_gate",
        match=equal,
        promotion_passed=False,
        revert_win_rate=0.48,
    ) == "probation"
    assert _generator_action_after_promotion_check(
        mode="soft_gate",
        match=strong,
        promotion_passed=True,
        revert_win_rate=0.48,
    ) == "promote"
    assert _generator_action_after_promotion_check(
        mode="strict_gate",
        match=equal,
        promotion_passed=False,
        revert_win_rate=0.48,
    ) == "reject"


def test_promotion_decision_lcb_and_fixed_suite_failures_stay_probationary() -> None:
    fixed_pass = FixedSuiteComparison(
        checked=True,
        passed=True,
        tolerance=0.05,
        reason="no fixed-suite regression",
    )
    fixed_fail = FixedSuiteComparison(
        checked=True,
        passed=False,
        tolerance=0.05,
        reason="fixed-suite exact-value error regressed",
    )

    lcb_fail = _match_stats(0.56, 0.49)
    decision = decide_promotion(lcb_fail, fixed_pass)
    assert not decision.passed
    assert _generator_action_after_promotion_check(
        mode="soft_gate",
        match=lcb_fail,
        promotion_passed=decision.passed,
        revert_win_rate=0.48,
    ) == "probation"

    fixed_suite_fail = _match_stats(0.56, 0.51)
    decision = decide_promotion(fixed_suite_fail, fixed_fail)
    assert not decision.passed
    assert _generator_action_after_promotion_check(
        mode="soft_gate",
        match=fixed_suite_fail,
        promotion_passed=decision.passed,
        revert_win_rate=0.48,
    ) == "probation"


@pytest.mark.parametrize(
    ("match", "expected_action", "expected_smart_elo"),
    [
        (_match_stats(0.47, 0.40), "revert", False),
        (_match_stats(0.52, 0.45), "probation", False),
        (_match_stats(0.58, 0.51), "promote", True),
    ],
)
def test_soft_gate_training_loop_transitions_and_smart_elo_trigger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    match: MatchStats,
    expected_action: str,
    expected_smart_elo: bool,
) -> None:
    current_best, _base_cfg = _write_current_best(tmp_path)
    cfg = _soft_gate_smoke_config(tmp_path, current_best)
    eval_calls = []
    smart_elo_calls = []

    def fake_evaluate_network_match(*args, **kwargs):
        eval_calls.append(kwargs)
        return match

    def fake_run_smart_elo_rating(**kwargs):
        smart_elo_calls.append(kwargs)
        return {"elo_rating": 1111.0, "elo_stderr": 11.0, "elo_n_games": 64}

    monkeypatch.setattr(
        "games.kingdomino.self_play.evaluate_network_match",
        fake_evaluate_network_match,
    )
    monkeypatch.setattr(
        "games.kingdomino.self_play._run_smart_elo_rating",
        fake_run_smart_elo_rating,
    )

    result = run_self_play_training(cfg, verbose=False)
    assert len(result["buffer"]) > 0

    row = json.loads((Path(cfg.log_path)).read_text().splitlines()[-1])
    assert row["promotion_checked"] is True
    assert row["promotion_action"] == expected_action
    assert row["smart_elo_triggered"] is expected_smart_elo
    assert len(smart_elo_calls) == int(expected_smart_elo)
    assert eval_calls[0]["games"] == 32
    assert eval_calls[0]["sims"] == 50

    if expected_action == "revert":
        assert row["generator_source"] == str(current_best)
        assert row["generator_checkpoint_path"] == str(current_best)
        assert row["generator_action"] == "revert"
    elif expected_action == "probation":
        assert row["generator_source"] == "learner_iter_0001"
        assert row["generator_checkpoint_path"].endswith("iter_0001.pt")
        assert row["generator_action"] == "probation"
    else:
        assert row["generator_source"] == "learner_iter_0001"
        assert row["generator_checkpoint_path"].endswith("iter_0001.pt")
        assert row["generator_action"] == "promote_current_best"
        assert row["smart_elo_rating"] == 1111.0
        assert (current_best.parent / "promotion_log.jsonl").exists()
        assert any(Path(cfg.hof_dir).glob("*.pt"))


def test_smart_elo_uses_smart_defaults_and_skips_existing_rating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_run_elo_rating(**kwargs):
        calls.append(kwargs)
        return {"elo_rating": 1234.0, "elo_stderr": 12.0, "elo_n_games": 64}

    monkeypatch.setattr(
        "games.kingdomino.self_play._run_elo_rating",
        fake_run_elo_rating,
    )

    db_path = tmp_path / "elo_db.json"
    cfg = SelfPlayConfig(
        elo_db=str(db_path),
        elo_games_per_anchor=99,
        elo_sims=999,
        smart_elo_games_per_anchor=32,
        smart_elo_sims=100,
    )
    result = _run_smart_elo_rating(
        checkpoint_path=str(tmp_path / "iter_0001.pt"),
        checkpoint_name="run_iter_0001_promoted",
        cfg=cfg,
        reason="promote",
        verbose=False,
    )

    assert result["elo_rating"] == 1234.0
    assert result["smart_elo_reason"] == "promote"
    assert result["smart_elo_name"] == "run_iter_0001_promoted"
    smart_cfg = calls[0]["cfg"]
    assert smart_cfg.elo_games_per_anchor == 32
    assert smart_cfg.elo_sims == 100

    db_path.write_text(
        json.dumps({
            "checkpoints": {
                "run_iter_0001_promoted": {
                    "rating": 1200.0,
                    "rating_stderr": 10.0,
                    "n_games": 64,
                },
            },
        }),
        encoding="utf-8",
    )
    result = _run_smart_elo_rating(
        checkpoint_path=str(tmp_path / "iter_0001.pt"),
        checkpoint_name="run_iter_0001_promoted",
        cfg=cfg,
        reason="promote",
        verbose=False,
    )

    assert len(calls) == 1
    assert result["elo_rating"] == 1200.0
    assert result["smart_elo_skipped"] == "already_rated"
