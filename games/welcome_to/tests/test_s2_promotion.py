"""Paired S2 strength endpoint and atomic promotion gates."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from games.welcome_to import network as nw
from games.welcome_to import s2_promotion


pytest.importorskip("welcome_to_rust")

_SMALL = nw.NetConfig(
    sheet_hidden=16,
    sheet_out=8,
    trunk_hidden=24,
    trunk_blocks=1,
    head_hidden=16,
)


def _net() -> nw.WelcomeToNet:
    torch.manual_seed(451)
    return nw.WelcomeToNet(_SMALL).eval()


def test_normalized_rank_averages_ties():
    assert s2_promotion.normalized_rank([10, 5], 0) == 1.0
    assert s2_promotion.normalized_rank([5, 10], 0) == 0.0
    assert s2_promotion.normalized_rank([10, 10], 0) == 0.5
    assert s2_promotion.normalized_rank([20, 10, 10, 0], 1) == pytest.approx(0.5)


def test_identical_candidate_and_incumbent_are_paired_null():
    net = _net()
    config = s2_promotion.GateConfig(
        games=4,
        simulations=1,
        inflight=4,
        max_batch=4,
        scheduler_workers=2,
        seed=45_000,
    )
    report = s2_promotion.run_gate(net, net, config=config, device="cpu")
    assert report.decision == "reject"
    assert report.primary_margin_delta.mean == 0.0
    assert report.secondary_rank_delta.mean == 0.0
    assert not report.primary_significant
    assert report.secondary_not_regressed
    assert report.candidate == report.incumbent


def test_gate_search_is_noiseless_and_secondary_requires_evidence_to_reject():
    search = s2_promotion.gate_search_config(37)
    assert search.simulations == 37
    assert search.dirichlet_alpha is None
    assert search.dirichlet_concentration is None
    assert search.dirichlet_weight == 0.0
    assert search.noise_fresh_fraction == 0.0

    uncertain_negative = s2_promotion.Estimate(
        mean=-0.02, stderr=0.03, lower=-0.08, upper=0.04
    )
    established_regression = s2_promotion.Estimate(
        mean=-0.08, stderr=0.01, lower=-0.10, upper=-0.06
    )
    assert s2_promotion.secondary_not_regressed(uncertain_negative, 0.0)
    assert not s2_promotion.secondary_not_regressed(established_regression, 0.0)


def test_promotion_archives_then_atomically_replaces_current_best(tmp_path):
    candidate = tmp_path / "candidate.pt"
    current_best = tmp_path / "current_best.pt"
    candidate.write_bytes(b"candidate checkpoint")
    current_best.write_bytes(b"incumbent checkpoint")
    null = s2_promotion.Estimate(0.0, 0.0, 0.0, 0.0)
    arm = s2_promotion.ArmSummary(0.0, 0.5, 20.0, 0.5, 1.0, 0.1)
    report = s2_promotion.PromotionReport(
        format=s2_promotion.GATE_FORMAT,
        version=s2_promotion.GATE_VERSION,
        decision="promote",
        games=300,
        primary_margin_delta=replace(null, mean=1.0, lower=0.1, upper=1.9),
        secondary_rank_delta=null,
        primary_significant=True,
        secondary_not_regressed=True,
        candidate=arm,
        incumbent=arm,
        diagnostics={},
        config={},
    )
    record = tmp_path / "gate.json"
    assert s2_promotion.install_promotion(
        report,
        candidate,
        current_best,
        archive_dir=tmp_path / "promoted",
        record_path=record,
        league_manifest=tmp_path / "league.json",
        iteration=7,
    )
    assert current_best.read_bytes() == b"candidate checkpoint"
    archives = list((tmp_path / "promoted").glob("best_*.pt"))
    assert len(archives) == 1
    assert archives[0].read_bytes() == b"incumbent checkpoint"
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["promoted"] is True
    assert payload["league_entry"]["archived_at_iteration"] == 7
    league = json.loads((tmp_path / "league.json").read_text(encoding="utf-8"))
    assert len(league["entries"]) == 1


def test_rejected_candidate_never_changes_current_best(tmp_path):
    candidate = tmp_path / "candidate.pt"
    current_best = tmp_path / "current_best.pt"
    candidate.write_bytes(b"candidate")
    current_best.write_bytes(b"best")
    zero = s2_promotion.Estimate(0.0, 1.0, -1.96, 1.96)
    arm = s2_promotion.ArmSummary(0.0, 0.5, 20.0, 0.5, 1.0, 0.1)
    report = s2_promotion.PromotionReport(
        format=s2_promotion.GATE_FORMAT,
        version=s2_promotion.GATE_VERSION,
        decision="reject",
        games=300,
        primary_margin_delta=zero,
        secondary_rank_delta=zero,
        primary_significant=False,
        secondary_not_regressed=True,
        candidate=arm,
        incumbent=arm,
        diagnostics={},
        config={},
    )
    assert not s2_promotion.install_promotion(
        report,
        candidate,
        current_best,
        archive_dir=tmp_path / "promoted",
        record_path=tmp_path / "gate.json",
    )
    assert current_best.read_bytes() == b"best"
    assert not (tmp_path / "promoted").exists()
    payload = json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))
    assert payload["status"] == "complete"


def test_crash_after_install_recovers_from_durable_intent(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate.pt"
    current_best = tmp_path / "current_best.pt"
    candidate.write_bytes(b"candidate checkpoint")
    current_best.write_bytes(b"incumbent checkpoint")
    null = s2_promotion.Estimate(0.0, 0.0, 0.0, 0.0)
    arm = s2_promotion.ArmSummary(0.0, 0.5, 20.0, 0.5, 1.0, 0.1)
    report = s2_promotion.PromotionReport(
        format=s2_promotion.GATE_FORMAT,
        version=s2_promotion.GATE_VERSION,
        decision="promote",
        games=300,
        primary_margin_delta=replace(null, mean=1.0, lower=0.1, upper=1.9),
        secondary_rank_delta=null,
        primary_significant=True,
        secondary_not_regressed=True,
        candidate=arm,
        incumbent=arm,
        diagnostics={},
        config={},
    )
    record = tmp_path / "gate.json"
    real_copy = s2_promotion.atomic_copy
    calls = 0

    def crash_after_second_copy(source, destination):
        nonlocal calls
        calls += 1
        result = real_copy(source, destination)
        if calls == 2:
            raise RuntimeError("simulated crash after candidate install")
        return result

    monkeypatch.setattr(s2_promotion, "atomic_copy", crash_after_second_copy)
    with pytest.raises(RuntimeError, match="simulated crash"):
        s2_promotion.install_promotion(
            report,
            candidate,
            current_best,
            archive_dir=tmp_path / "promoted",
            record_path=record,
            league_manifest=tmp_path / "league.json",
            iteration=8,
        )
    assert current_best.read_bytes() == b"candidate checkpoint"
    assert json.loads(record.read_text(encoding="utf-8"))["status"] == "installing"

    monkeypatch.setattr(s2_promotion, "atomic_copy", real_copy)
    assert s2_promotion.recover_pending_promotion(
        record, candidate_path=candidate, current_best_path=current_best
    )
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["league_entry"]["archived_at_iteration"] == 8
    archives = list((tmp_path / "promoted").glob("best_*.pt"))
    assert len(archives) == 1
    assert archives[0].read_bytes() == b"incumbent checkpoint"
