"""W3 auxiliary control head: labels, masking, inertness, and that it learns.

This is the cheapest of the four W3 arms -- no inference cost, no Rust parity
work -- so its job is to answer one question: can the trunk be pushed to
represent exact positional control per slot. The tests here protect the two ways
that question can be answered dishonestly: a label that is silently wrong, and a
head that quietly perturbs the incumbent policy.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from .buffer import GameRecorder
from .codec import legal_action_indices
from .control_table import DEFAULT_DIR, UNREACH, ControlTable, control_key
from .dataset import CONTROL_MAX_DIST, collate, examples_from_record
from .game import Phase
from .net import SWDNet
from .tableau_control import ATTACKER, DEFENDER, ControlSolver, Layout, _INF
from .train import compute_losses, model_from_config

needs_table = pytest.mark.skipif(
    not (DEFAULT_DIR / "manifest.json").exists(),
    reason="control table not generated",
)


def _examples(seed: int = 83):
    recorder = GameRecorder(seed, agents={"p0": "test", "p1": "test"})
    rng = random.Random(seed * 101)
    while recorder.game.phase is not Phase.COMPLETE:
        choice = rng.choice(legal_action_indices(recorder.game))
        recorder.play(choice, policy_target={choice: 1.0})
    return examples_from_record(recorder.finish())


@pytest.fixture(scope="module")
def labelled():
    examples = _examples()
    return examples, collate(examples[:32], control_table=ControlTable.load())


# -- labels -----------------------------------------------------------------


@needs_table
def test_labels_match_the_solver_through_the_whole_pipeline(labelled):
    """Key derivation, table lookup and token alignment, checked end to end
    against the live solver rather than against the table that produced them."""

    examples, batch = labelled
    checked = 0
    for row, example in enumerate(examples[:32]):
        if example.control_key is None:
            assert not bool(batch["control_row_valid"][row])
            continue
        assert bool(batch["control_row_valid"][row])
        age, mask, who, tempo = example.control_key
        layout = Layout.for_age(age)
        solver = ControlSolver(age)
        for slot_i, slot in enumerate(layout.slots):
            if not (mask >> slot_i) & 1:
                assert not bool(batch["control_slot_valid"][row, slot_i])
                continue
            turns = solver.solve(mask, slot, ATTACKER if who else DEFENDER, tempo)
            want_reach = 0.0 if turns >= _INF else 1.0
            assert float(batch["control_reachable"][row, slot_i]) == want_reach
            if want_reach:
                assert float(batch["control_distance"][row, slot_i]) == pytest.approx(
                    min(turns, CONTROL_MAX_DIST) / CONTROL_MAX_DIST
                )
            checked += 1
    assert checked > 50, f"only {checked} slots checked"


@needs_table
def test_labelled_tokens_are_the_tableau_tokens_for_those_slots(labelled):
    """The alignment that everything rests on: cell k must describe the slot
    whose token the head is told to read."""

    from .dataset import TYPE_IDS
    from .encoder import TokenType

    examples, batch = labelled
    tableau_type = TYPE_IDS[TokenType.TABLEAU]
    for row, example in enumerate(examples[:32]):
        if example.control_key is None:
            continue
        age, mask, _who, _tempo = example.control_key
        layout = Layout.for_age(age)
        for slot_i in range(len(layout.slots)):
            if not (mask >> slot_i) & 1:
                continue
            position = int(batch["control_token_index"][row, slot_i])
            assert int(example.type_ids[position]) == tableau_type
            # feature 0 and 2 of a tableau token are its row and column
            assert (float(example.features[position][0]),
                    float(example.features[position][2])) == \
                (float(layout.slots[slot_i][0]), float(layout.slots[slot_i][1]))


@needs_table
def test_distance_is_zero_wherever_the_slot_is_unreachable(labelled):
    """Distance is undefined without reachability; a sentinel would be learned
    as a number."""

    _examples_, batch = labelled
    unreachable = (batch["control_slot_valid"] & (batch["control_reachable"] == 0))
    assert unreachable.any()
    assert float(batch["control_distance"][unreachable].abs().max()) == 0.0


@needs_table
def test_rows_without_a_key_are_masked_not_defaulted(labelled):
    """A zero-filled control label reads as 'the opponent gets everything'."""

    examples, batch = labelled
    unlabelled = [i for i, e in enumerate(examples[:32]) if e.control_key is None]
    assert unlabelled, "expected some non-PLAY_AGE rows in a real game"
    for row in unlabelled:
        assert not bool(batch["control_row_valid"][row])
        assert not batch["control_slot_valid"][row].any()


def test_collate_without_a_table_emits_no_control_keys():
    """The arm must be entirely absent when it is not being run."""

    batch = collate(_examples()[:8])
    assert not [key for key in batch if key.startswith("control_")]


# -- the head ---------------------------------------------------------------


@needs_table
def test_head_does_not_perturb_the_incumbent(labelled):
    """Append-only: policy and value must be bit-identical to the model without
    the head, or the arm is not separable from the thing it is measured against.
    """

    _examples_, batch = labelled
    torch.manual_seed(0)
    with_head = SWDNet(d_model=48, layers=2, control_head=True).eval()
    without = SWDNet(d_model=48, layers=2).eval()
    without.load_state_dict(
        {k: v for k, v in with_head.state_dict().items()
         if not k.startswith("control_scorer.")},
        strict=False,
    )
    with torch.no_grad():
        a, b = with_head(batch), without(batch)
    for key in b:
        assert torch.equal(a[key], b[key]), key
    assert "control_reach_logit" in a


@needs_table
def test_head_is_inert_without_labels():
    """A batch collated with no table must not make the head fire."""

    batch = collate(_examples()[:8])
    model = SWDNet(d_model=32, layers=1, control_head=True).eval()
    with torch.no_grad():
        out = model(batch)
    assert "control_reach_logit" not in out
    _loss, parts = compute_losses(out, batch)
    assert parts["control"] == 0.0


@needs_table
def test_the_head_actually_learns_control(labelled):
    """The arm's premise: the trunk can be pushed to represent this."""

    _examples_, batch = labelled
    torch.manual_seed(3)
    model = SWDNet(d_model=48, layers=2, control_head=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = last = None
    for step in range(40):
        loss, parts = compute_losses(model(batch), batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0:
            first = parts["control"]
        last = parts["control"]
    assert last < first * 0.8, f"control loss {first:.3f} -> {last:.3f}"


@needs_table
def test_loss_ignores_masked_slots(labelled):
    """Perturbing predictions on masked slots must not move the loss, or the
    head is being trained on positions with no label."""

    _examples_, batch = labelled
    model = SWDNet(d_model=32, layers=1, control_head=True).eval()
    with torch.no_grad():
        out = model(batch)
    base = compute_losses(out, batch)[1]["control"]
    masked = ~batch["control_slot_valid"]
    poisoned = dict(out)
    poisoned["control_reach_logit"] = out["control_reach_logit"].masked_fill(masked, 50.0)
    poisoned["control_distance_pred"] = out["control_distance_pred"].masked_fill(
        masked, 50.0
    )
    assert compute_losses(poisoned, batch)[1]["control"] == pytest.approx(base)


# -- checkpoints ------------------------------------------------------------


def test_a_legacy_config_rebuilds_without_the_head():
    """Every checkpoint predating this arm must rebuild as exactly itself."""

    assert model_from_config({"d_model": 32, "layers": 1}).control_scorer is None
    assert model_from_config(
        {"d_model": 32, "layers": 1, "control_head": True}
    ).control_scorer is not None


def test_an_aux_arm_checkpoint_round_trips_through_load_evaluator(tmp_path):
    """The failure that cost the 2026-09-05 scoring run.

    `phase_e.load_evaluator` named every architecture switch by hand except
    `control_head`, so an aux-arm checkpoint rebuilt WITHOUT the head; its four
    `control_scorer` parameters then had no counterpart and the additive-only
    migration refused the file -- after ten arms had already been measured.
    """

    from .phase_e import load_evaluator
    from .train import build_model, make_checkpoint

    model = build_model("transformer", 32, 1, control_head=True)
    checkpoint = make_checkpoint(model, {"d_model": 32, "layers": 1})
    assert checkpoint["config"]["control_head"] is True
    assert any(key.startswith("control_scorer.") for key in checkpoint["model_state"])

    path = tmp_path / "aux.pt"
    torch.save(checkpoint, path)
    rebuilt = load_evaluator(str(path), "cpu", migrate=True).model
    assert rebuilt.control_scorer is not None
