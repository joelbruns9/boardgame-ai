"""Gates for the encoder audit.

What has to hold for the audit to be worth running: it must find the bug we
already know about. The discard-pile blindness -- a card revivable with an
unbuilt Mausoleum being invisible to the science and military reachability
features -- is reintroduced here on purpose, and the audit is required to
falsify a claim because of it, with no hint about where to look.

That test is also what settled how the audit works. Waiting for a *game* to
contradict a feasibility flag was the obvious design and it is far too weak:
with the bug reintroduced, 60 games and 7,396 retrospective checks found
nothing, because a science victory needs a whole chain of luck on top of the
wrong claim. Hunting the underlying *bound* instead falsifies it from 20 games.
"""

from __future__ import annotations

import pytest

from games.seven_wonders_duel import encoder
from games.seven_wonders_duel import encoder_audit as audit


@pytest.fixture()
def blind_to_the_discard(monkeypatch):
    """The original bug, put back: the discard pile stops being reachable."""

    monkeypatch.setattr(
        encoder._Derived, "_revivable_cards", lambda self, seat: (), raising=True
    )


def test_audit_is_clean_on_the_current_encoder():
    report = audit.run(6, hunt_attempts=4, hunt_positions=3)
    assert report.violations == [], "\n".join(str(v) for v in report.violations[:5])
    # A run that checked nothing would also report no violations.
    assert report.checks["card_cost"] > 0
    assert report.checks["hunt_sci_missing_obtainable"] > 0


def test_audit_rediscovers_the_known_discard_bug(blind_to_the_discard):
    """The whole point: no human insight, just games and claims."""

    report = audit.run(20, hunt_attempts=6, hunt_positions=6)
    hits = [v for v in report.violations if v.check == "hunt_sci_missing_obtainable"]
    assert hits, (
        "the audit no longer finds the bug it was built to find; "
        f"checks run: {dict(report.checks)}"
    )
    # A finding is only useful if it can be gone back to: seed, ply and seat
    # are enough to replay the position, and the detail names both numbers.
    first = hits[0]
    assert first.seed >= 0 and first.ply >= 0 and first.seat in (0, 1)
    assert "bound said" in first.detail


def test_the_corpus_reaches_the_endings_the_claims_are_about():
    """Random play almost never wins by science, and then nothing is tested."""

    report = audit.run(8)
    assert report.endings["scientific"] > 0
    assert report.endings["military"] > 0


def test_bounds_are_checked_against_what_the_game_actually_did():
    """The retrospective half must be exercised, not merely present."""

    report = audit.run(4)
    for check in ("sci_missing_obtainable", "mil_shields_obtainable", "score_block"):
        assert report.checks[check] > 0


def test_stub_checks_are_run_on_both_states(blind_to_the_discard):
    """The stub comparisons must not quietly become no-ops.

    They are the checks with no second formula to disagree with, so if they
    ever stop running, the audit loses the only thing that catches a
    reconstruction losing data rather than a rule being wrong.
    """

    report = audit.run(2)
    for check in ("card_cost", "stub_economy", "global_block"):
        assert report.checks[check] > 0
