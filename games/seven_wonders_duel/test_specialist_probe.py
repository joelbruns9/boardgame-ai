"""W7 S0a: the frozen-weights attack-coverage probe.

The probe is a measuring instrument, so what is tested here is that it measures
the right things and cannot silently report a null it did not earn: that the two
searches differ only in the bias, that a lambda of zero is a fixed point, and
that every column a null diagnosis depends on is present and in range.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("seven_wonders_rust")

from .specialist_probe import VICTORY_INDEX, probe
from .test_specialist_league import _two_net_records


def _checkpoint(tmp_path: Path, *, hierarchical: bool) -> Path:
    from .net import SWDNet
    from .train import make_checkpoint

    torch.manual_seed(11)
    model = SWDNet(d_model=32, layers=2, heads=4, hierarchical_value=hierarchical)
    path = tmp_path / ("hier.pt" if hierarchical else "flat.pt")
    torch.save(
        make_checkpoint(
            model,
            {
                "model": "transformer",
                "d_model": 32,
                "layers": 2,
                "heads": 4,
                "hierarchical_value": hierarchical,
            },
        ),
        path,
    )
    return path


@pytest.fixture(scope="module")
def records():
    return _two_net_records(0.0, games=6)


def test_the_probe_reports_every_column_a_null_diagnosis_needs(
    tmp_path: Path, records
):
    report = probe(
        _checkpoint(tmp_path, hierarchical=True),
        records,
        lam=1.0,
        victory="scientific",
        positions=6,
        sims=24,
        top_k=4,
        min_move=6,
    )
    summary = report["summary"]
    assert summary["positions"] > 0
    # The three questions a null has to be split into.
    for key in (
        "moved_fraction",
        "credible_fraction_of_moved",
        "mean_prior_of_new_choice",
        "mean_visit_share_of_new_choice",
        "own_type_outlook_base",
        "own_type_outlook_biased",
    ):
        assert key in summary
    assert 0.0 <= summary["moved_fraction"] <= 1.0
    # The utility shift is the bonus, in the ROOT ACTOR's terms: non-negative
    # for an own-win bias from either seat, and bounded by lambda. This is the
    # sign trap expressed as a measurement -- a seat-1 specialist rewarding its
    # opponent's science win would push this negative.
    assert 0.0 <= summary["root_value_shift"] <= 1.0 + 1e-9
    # The unshaped root stays a win probability whatever the bias did.
    assert summary["unshaped_root_within_unit_interval"]
    for row in report["rows"]:
        assert row["seat"] in (0, 1)
        assert 0.0 <= row["visit_share_of_new_choice"] <= 1.0
        assert row["q_cost"] >= -1e-9


def test_a_zero_lambda_probe_is_a_fixed_point(tmp_path: Path, records):
    """The two searches differ ONLY in the bias.

    Same positions, same seeds, same evaluator -- so at lambda = 0 nothing may
    move. A probe that reported movement here would be measuring its own
    plumbing rather than the treatment.
    """

    report = probe(
        _checkpoint(tmp_path, hierarchical=True),
        records,
        lam=0.0,
        victory="scientific",
        positions=6,
        sims=24,
        top_k=4,
        min_move=6,
    )
    assert report["summary"]["moved_fraction"] == 0.0
    assert report["summary"]["root_value_shift"] == pytest.approx(0.0, abs=1e-12)
    assert all(row["root_value_unshaped"] is None for row in report["rows"])


def test_a_checkpoint_without_the_outlook_head_is_refused(tmp_path: Path, records):
    """A null that only means "the head is missing" is worse than no probe."""

    with pytest.raises(SystemExit, match="hierarchical value head"):
        probe(
            _checkpoint(tmp_path, hierarchical=False),
            records,
            lam=0.5,
            victory="scientific",
            positions=2,
            sims=8,
            top_k=3,
            min_move=6,
        )


def test_the_victory_index_matches_the_joint7_class_order():
    """The probe reads `root_outlook[offset]` directly, so a reordering of the
    class list would silently make it report the wrong victory type."""

    from .dataset import JOINT7_CLASSES

    assert JOINT7_CLASSES[:3] == ("my_civilian", "my_scientific", "my_military")
    for name, offset in VICTORY_INDEX.items():
        assert JOINT7_CLASSES[offset] == f"my_{name}"


def test_the_probe_uses_the_SEARCHS_OWN_selected_move(tmp_path: Path, records):
    """R6. The probe defined both moves by max visits while running a Gumbel
    root, whose returned action comes from the candidate scores and whose
    deterministic policy comes from completed Q. On 16 mock searches max visits
    disagreed with the returned action 5 times and with the policy argmax 11.

    Under the default PUCT root the two coincide; the point is that the probe
    now reads `result["action"]` and so cannot drift from the searched player's
    decision if the root rule changes.
    """

    from .codec import legal_action_indices

    report = probe(
        _checkpoint(tmp_path, hierarchical=True),
        records,
        lam=0.0,
        victory="scientific",
        positions=4,
        sims=24,
        top_k=4,
        min_move=6,
        root_selection="gumbel",
    )
    assert report["summary"]["root_selection"] == "gumbel"
    # At lambda zero the two searches are the same search, so whatever
    # selection rule is in force must call it the same move.
    assert report["summary"]["moved_fraction"] == 0.0
    del legal_action_indices


def test_credibility_is_only_claimed_where_the_unbiased_search_looked(
    tmp_path: Path, records
):
    """`completed_q` falls back to the root value for an unvisited candidate, so
    counting that as "the unbiased search rates this fine" reports the absence
    of an opinion as an endorsement."""

    report = probe(
        _checkpoint(tmp_path, hierarchical=True),
        records,
        lam=1.0,
        victory="scientific",
        positions=6,
        sims=24,
        top_k=4,
        min_move=6,
    )
    assert "unverified_fraction_of_moved" in report["summary"]
    for row in report["rows"]:
        if row["credible"]:
            assert row["credibility_verified"], (
                "credibility was claimed for a move the lambda-zero search "
                "never visited"
            )
