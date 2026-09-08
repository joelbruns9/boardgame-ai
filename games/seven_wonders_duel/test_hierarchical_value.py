"""Workstream 4: one consistent distribution over winner and victory type.

Three claims, in order of how much they matter to a strong incumbent:

* it CANNOT harm strength in the default arm. The head is shadow-only, so it
  changes no served number; detached, its loss cannot reach a trunk weight
  either. That leaves throughput as the whole cost, which is the bar this
  project set for a speculative addition.
* it is CONSISTENT. The 7-way distribution and its win/draw/loss marginal are
  one object, not two heads that happen to agree -- which is the defect it
  exists to remove, since `value` and `joint7` can disagree today and nothing
  says which to believe.
* it LEARNS, and the attached arm really does reach the trunk, so the opt-in is
  a real arm rather than a differently-spelled version of the safe one.
"""

import random

import pytest

torch = pytest.importorskip("torch")

from games.seven_wonders_duel.buffer import GameRecorder
from games.seven_wonders_duel.codec import legal_action_indices
from games.seven_wonders_duel.dataset import JOINT7_CLASSES, collate, examples_from_record
from games.seven_wonders_duel.game import Phase
from games.seven_wonders_duel.net import SWDNet
from games.seven_wonders_duel.train import compute_losses, migrate_state_dict


@pytest.fixture(scope="module")
def examples():
    recorder = GameRecorder(11, agents={"p0": "random", "p1": "random"})
    rng = random.Random(7)
    while recorder.game.phase is not Phase.COMPLETE:
        recorder.play(rng.choice(legal_action_indices(recorder.game)))
    return examples_from_record(recorder.finish())


@pytest.fixture(scope="module")
def batch(examples):
    return collate(examples)


def _model(**kwargs):
    torch.manual_seed(0)
    return SWDNet(d_model=32, layers=2, heads=4, **kwargs)


# --- it cannot harm strength in the default arm -----------------------------


def test_the_served_heads_are_untouched(batch):
    """`value`, `joint7` and `policy` are bit-identical with the head present.

    Shadow means shadow: search reads the same numbers it always did, so no
    arena result can move because this head exists.
    """

    plain = _model().eval()
    with torch.no_grad():
        before = plain(batch)

    shadowed = _model(hierarchical_value=True)
    report = migrate_state_dict(plain.state_dict(), shadowed)
    assert not report["zeroed"], report["zeroed"]
    assert all(key.startswith("hier_value.") for key in report["initialized"])
    shadowed.eval()
    with torch.no_grad():
        after = shadowed(batch)
    for key, value in before.items():
        assert torch.equal(value, after[key]), key


def test_the_detached_head_cannot_move_a_trunk_weight(batch):
    """The whole safety argument, asserted rather than reasoned.

    A shadow head still shares a trunk, so its loss can shape representations
    for better or worse. Detached, that path does not exist: only the head's
    own three projections receive gradient, whatever the loss does.
    """

    model = _model(hierarchical_value=True)
    model.train()
    outputs = model(batch)
    # The head's loss ALONE, so anything that moves is the head's doing.
    torch.nn.functional.nll_loss(outputs["hier_joint7"], batch["joint7"]).backward()

    moved = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and parameter.grad.abs().sum() > 0
    }
    assert moved, "the head itself must train"
    assert all(name.startswith("hier_value.") for name in moved), sorted(moved)


def test_the_attached_arm_really_does_reach_the_trunk(batch):
    """Otherwise the opt-in would be the safe arm under another name."""

    model = _model(hierarchical_value=True, hierarchical_value_detach=False)
    model.train()
    outputs = model(batch)
    torch.nn.functional.nll_loss(outputs["hier_joint7"], batch["joint7"]).backward()

    trunk = model.embedder.type_embedding.weight.grad
    assert trunk is not None and trunk.abs().sum() > 0


# --- it is consistent -------------------------------------------------------


def test_the_marginal_and_the_joint_are_one_object(batch):
    """Summing the 7-way distribution over victory type reproduces W/D/L.

    Exactly, not approximately -- the head emits the factors, so this is
    arithmetic rather than something training has to discover. It is the whole
    point: `value` and `joint7` are separate linear heads today and nothing
    makes them agree.
    """

    model = _model(hierarchical_value=True).eval()
    with torch.no_grad():
        out = model(batch)
    joint = out["hier_joint7"].exp()
    win = joint[:, 0:3].sum(-1)
    loss = joint[:, 3:6].sum(-1)
    draw = joint[:, 6]
    marginal = out["hier_value"].exp()

    assert torch.allclose(joint.sum(-1), torch.ones_like(draw), atol=1e-6)
    assert torch.allclose(win, marginal[:, 0], atol=1e-6)
    assert torch.allclose(draw, marginal[:, 1], atol=1e-6)
    assert torch.allclose(loss, marginal[:, 2], atol=1e-6)


def test_the_flat_heads_are_free_to_disagree(batch):
    """The defect, shown rather than described.

    `value` and `joint7` are independent projections, so their implied win
    probabilities differ by whatever initialization gives them. This is what
    makes "which do I believe" unanswerable today.
    """

    model = _model().eval()
    with torch.no_grad():
        out = model(batch)
    flat_joint = out["joint7"].softmax(-1)
    implied_win = flat_joint[:, 0:3].sum(-1)
    stated_win = out["value"].softmax(-1)[:, 0]
    assert not torch.allclose(implied_win, stated_win, atol=1e-3)


def test_the_class_layout_is_pinned():
    """The factorisation hard-codes the order of `JOINT7_CLASSES`.

    Reordering that tuple would leave the arithmetic valid and the meaning
    wrong, so `net` refuses at import instead of mislabelling victory types.
    """

    from games.seven_wonders_duel import net

    assert JOINT7_CLASSES[:3] == ("my_civilian", "my_scientific", "my_military")
    assert JOINT7_CLASSES[6] == "draw"
    original = net.JOINT7_CLASSES
    net.JOINT7_CLASSES = ("draw", *original[:6])
    try:
        with pytest.raises(RuntimeError, match="changed order"):
            net._check_joint7_layout()
    finally:
        net.JOINT7_CLASSES = original


# --- it learns --------------------------------------------------------------


def test_the_loss_is_reported_and_weighted(batch):
    model = _model(hierarchical_value=True)
    outputs = model(batch)

    off, parts_off = compute_losses(outputs, batch, hier_value_weight=0.0)
    on, parts_on = compute_losses(outputs, batch, hier_value_weight=0.5)
    assert parts_on["hier_value"] > 0
    assert parts_off["hier_value"] == pytest.approx(parts_on["hier_value"])
    assert float(on.detach()) > float(off.detach()), "the weight must reach the total"

    plain_outputs = _model()(batch)
    _, parts_absent = compute_losses(plain_outputs, batch)
    assert parts_absent["hier_value"] == 0.0


def test_the_head_fits_a_position_it_is_shown(batch):
    """A gradient is not learning; this checks the loss actually descends."""

    model = _model(hierarchical_value=True)
    model.train()
    optimizer = torch.optim.Adam(model.hier_value.parameters(), lr=0.05)
    first = last = None
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.nll_loss(
            model(batch)["hier_joint7"], batch["joint7"]
        )
        loss.backward()
        optimizer.step()
        if step == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    assert last < first


# --- option 1: the head's output escapes the model ---------------------------


def test_the_evaluator_reports_the_hierarchical_read():
    """`Evaluation` carries it, and reports None for a model without the head.

    None rather than a copy of the flat numbers: a caller must be able to tell
    "this checkpoint has no W4 head" from "its head agrees with the flat one".
    """

    from games.seven_wonders_duel.game import GameState
    from games.seven_wonders_duel.inference import Evaluator

    states = [GameState.new(seed=11)]
    plain = Evaluator(_model(), "cpu", 8, fuse_embedder=False)
    assert plain.evaluate_states(states)[0].hier_joint7 is None

    withhead = Evaluator(
        _model(hierarchical_value=True), "cpu", 8, fuse_embedder=False
    )
    row = withhead.evaluate_states(states)[0]
    assert row.hier_joint7 is not None and row.hier_wdl is not None
    assert row.hier_joint7.shape == (7,)
    # Probabilities, not log-probabilities: `exp` at the boundary, and never a
    # second softmax over an already-normalised vector, which would silently
    # flatten it rather than fail.
    assert row.hier_joint7.sum() == pytest.approx(1.0, abs=1e-5)
    assert row.hier_joint7[0:3].sum() == pytest.approx(
        float(row.hier_wdl[0]), abs=1e-5
    )


def test_the_advisor_reports_both_reads_and_their_disagreement():
    """Alongside the flat numbers, never instead of them.

    Every existing checkpoint and every recorded measurement used the flat
    pair, and the plan promotes the new one only after calibration. The panel
    itself still renders the flat read, because the head is untrained: showing
    it today would render noise.
    """

    from games.seven_wonders_duel.advisor_adapter import SevenWondersAdvisor
    from games.seven_wonders_duel.inference import Evaluator

    adapter = SevenWondersAdvisor(
        evaluator=Evaluator(
            _model(hierarchical_value=True), "cpu", 8, fuse_embedder=False
        )
    )
    # Past the draft, where the outlook is deliberately withheld.
    state = adapter.state_from_wire(
        {"seed": 7, "first_player": 0, "prefix": _draft_prefix(7)}
    )
    outlook = adapter.state_to_public(state)["victory_outlook"]
    assert outlook is not None
    assert "victory_type" in outlook  # the flat read survives untouched
    hier = outlook["hierarchical"]
    assert hier["you_win"] + hier["opponent_wins"] + hier["draw"] == pytest.approx(
        1.0, abs=1e-5
    )
    assert hier["you_win"] == pytest.approx(hier["wdl"][0], abs=1e-5)
    assert hier["flat_disagreement"] >= 0.0


def _draft_prefix(seed: int) -> list[int]:
    """A legal action prefix that walks the Wonder draft to the first Age.

    Built by playing, not written down: the draft's legal set depends on the
    deal, so a hard-coded prefix would be a different position under any seed
    change and an illegal one under most.
    """

    from games.seven_wonders_duel.game import GameState, Phase

    game = GameState.new(seed=seed, first_player=0)
    prefix = []
    while game.phase is Phase.WONDER_DRAFT:
        action = legal_action_indices(game)[0]
        prefix.append(action)
        from games.seven_wonders_duel.codec import decode_action
        from games.seven_wonders_duel.engine import apply_action

        # `apply_action` mutates in place and returns only the chance events.
        apply_action(game, decode_action(game, action))
    return prefix
