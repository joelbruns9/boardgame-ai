"""W4's soft victory-type target, the joint7 replacement arm, and what the advisor shows.

What these protect:

* a row's outlook is used only when it is a real, unbiased distribution -- a
  specialist's search over-weights its own victory type, and an imported buffer
  has no outlook at all;
* the loss is exactly ``(1 - b) * hard + b * outlook`` on usable rows and the
  hard label elsewhere, and ``b = 0`` is the historical NLL bit for bit;
* the weight ramps in on the training clock rather than arriving at full
  strength while the freshly added head is still untrained;
* under the replacement arm the advisor's headline victory-type read is W4's,
  because the flat head it used to show stopped training.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

torch = pytest.importorskip("torch")

from .buffer import GameRecorder
from .codec import legal_action_indices
from .dataset import collate, examples_from_record, usable_root_outlook
from .game import Phase
from .net import SWDNet
from .train import compute_losses, make_checkpoint

_OUTLOOK = [0.1, 0.4, 0.05, 0.2, 0.1, 0.1, 0.05]


@pytest.fixture(scope="module")
def examples():
    recorder = GameRecorder(23, agents={"p0": "random", "p1": "random"})
    rng = random.Random(23)
    while recorder.game.phase is not Phase.COMPLETE:
        recorder.play(rng.choice(legal_action_indices(recorder.game)))
    return examples_from_record(recorder.finish())


def _with_outlook(example, outlook=_OUTLOOK, search_lambda=0.0):
    return dataclasses.replace(example, root_outlook=list(outlook), search_lambda=search_lambda)


def test_a_usable_outlook_is_normalised():
    class Row:
        root_outlook = [2.0, 0, 0, 0, 0, 0, 2.0]
        search_lambda = 0.0

    assert usable_root_outlook(Row()) == pytest.approx([0.5, 0, 0, 0, 0, 0, 0.5])


@pytest.mark.parametrize(
    "outlook, search_lambda",
    [
        (None, 0.0),
        (_OUTLOOK, 3.0),                      # a biased specialist search
        (_OUTLOOK[:6], 0.0),                  # wrong length
        ([0.5, -0.1, 0.2, 0.2, 0.1, 0.05, 0.05], 0.0),
        ([float("nan")] + [0.1] * 6, 0.0),
        ([0.0] * 7, 0.0),
    ],
)
def test_unusable_outlooks_keep_the_hard_label(outlook, search_lambda):
    class Row:
        pass

    row = Row()
    row.root_outlook = outlook
    row.search_lambda = search_lambda
    assert usable_root_outlook(row) is None


def test_collate_carries_the_outlook_only_where_it_is_usable(examples):
    rows = [_with_outlook(examples[0]), _with_outlook(examples[1], search_lambda=3.0), examples[2]]
    batch = collate(rows)
    assert batch["outlook_soft_valid"].tolist() == [True, False, False]
    assert torch.allclose(batch["outlook_soft"][0], torch.tensor(_OUTLOOK))
    assert float(batch["outlook_soft"][1].abs().sum()) == 0.0


def _model():
    torch.manual_seed(1)
    return SWDNet(d_model=32, layers=1, heads=4, hierarchical_value=True)


def test_zero_weight_is_the_historical_loss(examples):
    batch = collate([_with_outlook(example) for example in examples[:8]])
    outputs = _model()(batch)
    _, off = compute_losses(outputs, batch, outlook_bootstrap=0.0)
    expected = torch.nn.functional.nll_loss(outputs["hier_joint7"], batch["joint7"])
    assert off["hier_value"] == pytest.approx(float(expected.detach()), rel=1e-6)


def test_the_blend_is_exact_on_usable_rows_and_hard_elsewhere(examples):
    rows = [_with_outlook(example) for example in examples[:4]] + list(examples[4:8])
    batch = collate(rows)
    outputs = _model()(batch)
    b = 0.5
    _, parts = compute_losses(outputs, batch, outlook_bootstrap=b)
    log_p = outputs["hier_joint7"].detach()
    hard = torch.nn.functional.one_hot(batch["joint7"], 7).float()
    expected_rows = []
    for row in range(8):
        target = hard[row]
        if row < 4:
            target = (1 - b) * hard[row] + b * torch.tensor(_OUTLOOK)
        expected_rows.append(-(target * log_p[row]).sum())
    assert parts["hier_value"] == pytest.approx(float(torch.stack(expected_rows).mean()), rel=1e-5)


def test_the_soft_target_pulls_the_head_toward_the_outlook(examples):
    """Train on one position with b = 1: the head should move toward the outlook,
    not toward the realised class."""

    model = _model()
    batch = collate([_with_outlook(examples[10])])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(150):
        optimizer.zero_grad()
        total, _ = compute_losses(model(batch), batch, hier_value_weight=1.0, outlook_bootstrap=1.0)
        total.backward()
        optimizer.step()
    with torch.no_grad():
        learned = model(batch)["hier_joint7"].exp()[0]
    assert torch.allclose(learned, torch.tensor(_OUTLOOK), atol=0.05)


def test_the_weight_ramps_in_on_the_training_clock():
    from .phase_d import PhaseDConfig

    config = PhaseDConfig(
        hierarchical_value=True, hier_value_weight=0.2,
        outlook_bootstrap=0.5, outlook_bootstrap_games=10_000,
    )
    schedule = config.outlook_bootstrap_schedule()
    assert schedule.value(0) == 0.0
    assert schedule.value(5_000) == pytest.approx(0.25)
    assert schedule.value(10_000) == pytest.approx(0.5)
    assert schedule.value(40_000) == pytest.approx(0.5)
    immediate = PhaseDConfig(hierarchical_value=True, hier_value_weight=0.2, outlook_bootstrap=0.5)
    assert immediate.outlook_bootstrap_schedule().value(0) == pytest.approx(0.5)


def test_the_bootstrap_needs_the_head_and_a_sane_weight():
    from .phase_d import PhaseDConfig

    with pytest.raises(ValueError, match="requires --hierarchical-value"):
        PhaseDConfig(outlook_bootstrap=0.5).validate()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PhaseDConfig(hierarchical_value=True, hier_value_weight=0.2, outlook_bootstrap=1.5).validate()
    with pytest.raises(ValueError, match="non-negative"):
        PhaseDConfig(
            hierarchical_value=True, hier_value_weight=0.2,
            outlook_bootstrap=0.5, outlook_bootstrap_games=-1,
        ).validate()


def test_validation_scores_the_hard_label(examples):
    """Held-out numbers must mean the same thing at any bootstrap weight."""

    from .train import evaluate

    model = _model()
    rows = [_with_outlook(example) for example in examples[:16]]
    metrics = evaluate(model, rows, "cpu", batch_size=16)
    batch = collate(rows)
    with torch.no_grad():
        expected = torch.nn.functional.nll_loss(model(batch)["hier_joint7"], batch["joint7"])
    assert metrics["hier_value"] == pytest.approx(float(expected), rel=1e-5)


class _Row:
    def __init__(self, flat, hier):
        self.joint7 = flat
        self.hier_joint7 = hier
        self.hier_wdl = [sum(hier[0:3]), hier[6], sum(hier[3:6])]
        self.wdl = [0.5, 0.0, 0.5]
        self.margin = 0.0
        self.military = 0.0
        self.science = [0.0, 0.0]


class _StubEvaluator:
    def __init__(self, replaced):
        self.model = type("M", (), {"joint7_replaced": replaced})()
        self.row = _Row([0.7, 0, 0, 0.3, 0, 0, 0], [0.1, 0.5, 0.0, 0.1, 0.0, 0.3, 0.0])

    def evaluate_states(self, games):
        return [self.row for _ in games]


def _advisor_outlook(replaced):
    from .advisor_adapter import SevenWondersAdvisor
    from .test_hierarchical_value import _draft_prefix

    adapter = SevenWondersAdvisor(evaluator=_StubEvaluator(replaced))
    state = adapter.state_from_wire({"seed": 7, "first_player": 0, "prefix": _draft_prefix(7)})
    return adapter.state_to_public(state)["victory_outlook"]


def test_the_advisor_headlines_w4_when_joint7_was_replaced():
    outlook = _advisor_outlook(replaced=True)
    assert outlook["victory_type_source"] == "hierarchical"
    assert outlook["flat_joint7_stale"] is True
    assert outlook["victory_type"]["you_scientific"] == pytest.approx(0.5)
    assert outlook["you_win"] == pytest.approx(0.6)


def test_the_advisor_keeps_the_flat_read_otherwise():
    outlook = _advisor_outlook(replaced=False)
    assert outlook["victory_type_source"] == "flat"
    assert "flat_joint7_stale" not in outlook
    assert outlook["you_win"] == pytest.approx(0.7)
    assert outlook["hierarchical"]["you_win"] == pytest.approx(0.6)


def test_loading_a_replacement_checkpoint_marks_the_flat_head_stale(tmp_path):
    from .phase_e import load_evaluator

    model = SWDNet(d_model=32, layers=1, heads=4, hierarchical_value=True, hierarchical_value_detach=False)
    for replaced, name in ((True, "replaced.pt"), (False, "kept.pt")):
        path = tmp_path / name
        torch.save(
            make_checkpoint(
                model,
                {"model": "transformer", "d_model": 32, "layers": 1, "heads": 4,
                 "hier_value_replaces_joint7": replaced},
            ),
            path,
        )
        assert load_evaluator(str(path), "cpu").model.joint7_replaced is replaced
