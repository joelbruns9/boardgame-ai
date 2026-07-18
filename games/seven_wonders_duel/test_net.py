"""Phase B CI tests: dataset bridge, model forward/backward, checkpoint
discipline. Tiny dims, CPU-only, fast; the full wiring gate is
phase_b_gate.py."""

import random

import pytest

torch = pytest.importorskip("torch")

from games.seven_wonders_duel.buffer import GameRecorder
from games.seven_wonders_duel.codec import NUM_ACTIONS
from games.seven_wonders_duel.dataset import (
    MAX_FEATURES,
    collate,
    examples_from_record,
)
from games.seven_wonders_duel.game import Phase
from games.seven_wonders_duel.mlp import SWDMlp
from games.seven_wonders_duel.net import SWDNet, masked_policy_log_softmax
from games.seven_wonders_duel.train import (
    compute_losses,
    load_checkpoint,
    make_checkpoint,
)


@pytest.fixture(scope="module")
def bot_record():
    recorder = GameRecorder(11, agents={"p0": "random", "p1": "random"})
    rng = random.Random(42)
    from games.seven_wonders_duel.codec import legal_action_indices

    while recorder.game.phase is not Phase.COMPLETE:
        recorder.play(rng.choice(legal_action_indices(recorder.game)))
    return recorder.finish()


@pytest.fixture(scope="module")
def examples(bot_record):
    return examples_from_record(bot_record)


def test_examples_have_actor_relative_outcomes(bot_record, examples):
    assert len(examples) == len(bot_record.moves)
    winner = bot_record.winner
    for example, move in zip(examples, bot_record.moves):
        assert example.policy_target.sum() == pytest.approx(1.0)
        assert len(example.policy_target) == len(example.legal)
        if winner is None:
            assert example.value_class == 1
            assert example.joint7_class == 6
        elif move.actor == winner:
            assert example.value_class == 0
            assert example.joint7_class in (0, 1, 2)
        else:
            assert example.value_class == 2
            assert example.joint7_class in (3, 4, 5)
    # Actor-relative military finals must be exact mirrors between the seats.
    seats = {m.actor: e for e, m in zip(examples, bot_record.moves)}
    if len(seats) == 2:
        assert seats[0].military_final == pytest.approx(-seats[1].military_final)


def test_collate_shapes_and_masks(examples):
    batch = collate(examples[:16])
    size, tokens = batch["type_ids"].shape
    assert size == 16
    assert batch["features"].shape == (size, tokens, MAX_FEATURES)
    assert batch["legal_mask"].shape == (size, NUM_ACTIONS)
    for row in range(size):
        real = ~batch["pad_mask"][row]
        assert real.sum() == len(examples[row].type_ids)
        assert batch["policy"][row].sum() == pytest.approx(1.0)
        assert (batch["policy"][row] > 0).sum() <= batch["legal_mask"][row].sum()


@pytest.mark.parametrize("model_cls", [SWDNet, SWDMlp])
def test_forward_shapes_and_masked_policy(examples, model_cls):
    model = model_cls(d_model=32) if model_cls is SWDMlp else model_cls(32, 1, 2)
    batch = collate(examples[:8])
    outputs = model(batch)
    assert outputs["policy"].shape == (8, NUM_ACTIONS)
    assert outputs["value"].shape == (8, 3)
    assert outputs["joint7"].shape == (8, 7)
    assert outputs["margin"].shape == (8,)
    assert outputs["science"].shape == (8, 2)
    log_probs = masked_policy_log_softmax(outputs["policy"], batch["legal_mask"])
    probs = log_probs.exp()
    assert torch.allclose(probs.sum(-1), torch.ones(8), atol=1e-5)
    assert (probs[~batch["legal_mask"]] == 0).all()


@pytest.mark.parametrize("model_cls", [SWDNet, SWDMlp])
def test_one_training_step_reduces_loss(examples, model_cls):
    torch.manual_seed(0)
    model = model_cls(d_model=32) if model_cls is SWDMlp else model_cls(32, 1, 2)
    batch = collate(examples[:32])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first, _ = compute_losses(model(batch), batch)
    for _ in range(5):
        optimizer.zero_grad()
        loss, _ = compute_losses(model(batch), batch)
        loss.backward()
        optimizer.step()
    final, _ = compute_losses(model(batch), batch)
    assert float(final) < float(first)


def test_checkpoint_enforces_encoder_signature(tmp_path, examples):
    model = SWDNet(32, 1, 2)
    path = tmp_path / "model.pt"
    torch.save(make_checkpoint(model, {"model": "transformer"}), path)
    fresh = SWDNet(32, 1, 2)
    checkpoint = load_checkpoint(path, fresh)
    assert checkpoint["config"]["model"] == "transformer"

    tampered = torch.load(path, weights_only=False)
    tampered["encoder_signature"] = "stale"
    torch.save(tampered, path)
    with pytest.raises(ValueError, match="signature"):
        load_checkpoint(path, SWDNet(32, 1, 2))
