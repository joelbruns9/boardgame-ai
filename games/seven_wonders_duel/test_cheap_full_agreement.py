"""Arithmetic of the cheap-vs-full agreement probe.

The probe exists to decide whether to spend compute on run-length full search or
periodic full-sims iterations, so a quiet error here would buy or refuse real
hardware time on a wrong number.
"""

from __future__ import annotations

import math

import pytest

from .cheap_full_agreement import _kl, _prior_rank, summarise


def _row(agree=True, rank=0.0, kl=0.0, **extra):
    row = {"seed": 0, "move": 0, "cards_left": 5, "agree": agree,
           "full_prior_rank": rank, "kl_cheap_full": kl}
    row.update(extra)
    return row


def test_kl_is_zero_for_identical_policies():
    p = {1: 0.5, 2: 0.3, 3: 0.2}
    assert _kl(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_counts_actions_the_full_search_never_visited():
    """The case that matters most, and the easy one to drop.

    An action with cheap mass and zero full mass is maximal disagreement. Summing
    only over the shared support would report the smallest divergence exactly
    where the two searches differ most.
    """

    diverged = _kl({1: 1.0}, {2: 1.0})
    assert diverged > 10, "unvisited actions must dominate, not vanish"


def test_prior_rank_is_scale_free():
    """"Low-prior" must mean the same thing at 4 legal moves and at 40."""

    few = {10: 0.7, 11: 0.2, 12: 0.1}
    many = {i: 1.0 / (i + 1) for i in range(40)}
    assert _prior_rank(few, 10) == 0.0
    assert _prior_rank(few, 12) == 1.0
    assert _prior_rank(many, 0) == 0.0
    assert _prior_rank(many, 39) == 1.0


def test_prior_rank_of_a_single_legal_move_is_the_top():
    """No division by zero when the position is forced."""

    assert _prior_rank({7: 1.0}, 7) == 0.0


def test_low_prior_disagreement_is_conditioned_not_aggregated():
    """The number that decides whether the mechanism is real.

    Two of the four rows disagree, but only one of the two low-prior rows does,
    so the overall rate and the conditioned rate must differ -- collapsing them
    would hide exactly the signal being looked for.
    """

    rows = [
        _row(agree=False, rank=0.9),
        _row(agree=True, rank=0.9),
        _row(agree=False, rank=0.0),
        _row(agree=True, rank=0.0),
    ]
    summary = summarise(rows, low_prior_rank=0.5)
    assert summary["disagreement_rate"] == 0.5
    assert summary["low_prior_moves"] == 2
    assert summary["disagreement_rate_on_low_prior"] == 0.5


def test_positions_without_a_proof_do_not_count_as_safe():
    """Only solved positions enter the provably-losing rate.

    Counting unsolved ones as non-losing would drive the rate toward zero with
    sample size and read as "the cheap path drops nothing".
    """

    rows = [
        _row(cheap_loses=True, full_loses=False),
        _row(),  # no proof at this position
        _row(),
    ]
    summary = summarise(rows, low_prior_rank=0.5)
    assert summary["solved_positions"] == 1
    assert summary["cheap_provably_losing"] == 1.0
    assert summary["full_provably_losing"] == 0.0


def test_an_empty_probe_reports_nothing_rather_than_dividing_by_zero():
    assert summarise([], low_prior_rank=0.5) == {"cheap_moves": 0}


def test_no_low_prior_moves_reports_nan_not_zero():
    """Zero would read as 'perfect agreement on the cases that matter'."""

    summary = summarise([_row(agree=False, rank=0.0)], low_prior_rank=0.5)
    assert math.isnan(summary["disagreement_rate_on_low_prior"])


def test_rows_survive_a_stopped_run(tmp_path, monkeypatch):
    """A partial run must still be a usable sample.

    The probe runs for hours and has been stopped part-way twice; the first time
    it lost 72 games of solving because rows lived only in memory until the end.
    """

    import json

    from . import cheap_full_agreement as module

    calls = {"n": 0}

    def fake_probe_game(seed, evaluator, **kwargs):
        calls["n"] += 1
        if calls["n"] > 2:
            raise KeyboardInterrupt("stopped part-way, as production runs were")
        return [_row(agree=(calls["n"] == 1), rank=0.9)]

    monkeypatch.setattr(module, "probe_game", fake_probe_game)
    monkeypatch.setattr(module, "load_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(module, "model_from_config", lambda *a, **k: None)
    monkeypatch.setattr(module, "Evaluator", lambda *a, **k: None)
    monkeypatch.setattr(module.torch, "load", lambda *a, **k: {"config": {}})

    out = tmp_path / "cfa.json"
    with pytest.raises(KeyboardInterrupt):
        module.main(
            ["--checkpoint", str(tmp_path / "none.pt"), "--games", "5", "--out", str(out)]
        )

    written = [
        json.loads(line)
        for line in out.with_suffix(".rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(written) == 2, "both completed games must be on disk"
