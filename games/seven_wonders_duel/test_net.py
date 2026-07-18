"""Phase B CI tests: dataset bridge, model forward/backward, checkpoint
discipline. Tiny dims, CPU-only, fast; the full wiring gate is
phase_b_gate.py."""

import random

import pytest

torch = pytest.importorskip("torch")

import dataclasses

import numpy as np

from games.seven_wonders_duel.buffer import GameRecorder, ReplayMismatchError
from games.seven_wonders_duel.codec import NUM_ACTIONS
from games.seven_wonders_duel.dataset import (
    MAX_FEATURES,
    _actor_value_class,
    _joint7_class,
    collate,
    examples_from_record,
)
from games.seven_wonders_duel.game import Phase, VictoryType
from games.seven_wonders_duel.mlp import SWDMlp
from games.seven_wonders_duel.net import SWDNet, masked_policy_log_softmax
from games.seven_wonders_duel.train import (
    baselines,
    compute_losses,
    evaluate,
    game_honest_split,
    load_checkpoint,
    make_checkpoint,
    migrate_state_dict,
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


def test_featurization_rejects_tampered_records(bot_record):
    moves = list(bot_record.moves)
    moves[3] = dataclasses.replace(moves[3], mask_hash="sha256:0000000000000000")
    with pytest.raises(ReplayMismatchError):
        examples_from_record(dataclasses.replace(bot_record, moves=tuple(moves)))


def test_visit_distribution_targets_and_illegal_visits(bot_record):
    moves = list(bot_record.moves)
    legal_action = moves[0].action
    moves[0] = dataclasses.replace(moves[0], visits={legal_action: 3, 999999: 0})
    with pytest.raises(ValueError, match="illegal"):
        examples_from_record(dataclasses.replace(bot_record, moves=tuple(moves)))

    moves[0] = dataclasses.replace(moves[0], visits={legal_action: 3})
    examples = examples_from_record(
        dataclasses.replace(bot_record, moves=tuple(moves))
    )
    assert examples[0].policy_target.sum() == pytest.approx(1.0)

    moves[0] = dataclasses.replace(moves[0], visits={legal_action: 0})
    with pytest.raises(ValueError, match="zero"):
        examples_from_record(dataclasses.replace(bot_record, moves=tuple(moves)))


def test_policy_excluded_moves_carry_no_policy_loss(bot_record):
    moves = tuple(
        dataclasses.replace(move, policy_excluded=True) for move in bot_record.moves
    )
    examples = examples_from_record(dataclasses.replace(bot_record, moves=moves))
    batch = collate(examples[:8])
    assert not batch["has_policy"].any()
    model = SWDNet(32, 1, 2)
    _, parts = compute_losses(model(batch), batch)
    assert parts["policy"] == 0.0


def test_outcome_class_mappings_are_exhaustive():
    for victory, offset in (
        (VictoryType.CIVILIAN, 0),
        (VictoryType.SCIENTIFIC, 1),
        (VictoryType.MILITARY, 2),
    ):
        assert _joint7_class(0, victory, actor=0) == offset
        assert _joint7_class(0, victory, actor=1) == 3 + offset
    assert _joint7_class(None, VictoryType.SHARED_CIVILIAN, actor=0) == 6
    assert _actor_value_class(0, 0) == 0
    assert _actor_value_class(0, 1) == 2
    assert _actor_value_class(None, 0) == 1


def test_each_head_receives_gradient(examples):
    model = SWDNet(32, 1, 2)
    batch = collate(examples[:8])
    outputs = model(batch)
    head_modules = {
        "policy": model.heads.policy,
        "value": model.heads.value,
        "joint7": model.heads.joint7,
        "margin": model.heads.margin,
        "military": model.heads.military,
        "science": model.heads.science,
    }
    for name, module in head_modules.items():
        model.zero_grad()
        outputs = model(batch)
        outputs[name].sum().backward()
        assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0
        for other_name, other in head_modules.items():
            if other_name != name:
                assert other.weight.grad is None or other.weight.grad.abs().sum() == 0


def test_regression_baselines_and_uniform_policy_formula(examples):
    base = baselines(examples)
    assert base["margin_mae"] >= 0 and base["military_mae"] > 0
    expected_uniform = sum(
        float(np.log(len(e.legal))) for e in examples if e.has_policy
    ) / sum(1 for e in examples if e.has_policy)
    assert base["policy_uniform_loss"] == pytest.approx(expected_uniform)
    metrics = evaluate(SWDNet(32, 1, 2), examples[:32], "cpu")
    assert "margin_mae" in metrics and "science_mae" in metrics


def test_evaluate_honors_aux_weight(examples):
    model = SWDNet(32, 1, 2)
    torch.manual_seed(0)
    low = evaluate(model, examples[:16], "cpu", aux_weight=0.0)
    high = evaluate(model, examples[:16], "cpu", aux_weight=1.0)
    assert low["total"] != high["total"]
    assert low["policy"] == pytest.approx(high["policy"])


def test_iteration_split_holds_out_whole_iterations(examples):
    labeled = []
    for index, example in enumerate(examples):
        clone = dataclasses.replace(example) if False else example
        labeled.append(
            dataclasses.replace(
                example, iteration=index % 4, game_key=index
            )
        )
    train, val = game_honest_split(labeled, 0.25)
    train_iters = {e.iteration for e in train}
    val_iters = {e.iteration for e in val}
    assert val_iters and not (train_iters & val_iters)
    assert max(val_iters) == 3  # most recent iterations held out


def test_aux_padding_row_stays_zero_after_training(examples):
    model = SWDNet(32, 1, 2)
    batch = collate(examples[:16])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad()
        loss, _ = compute_losses(model(batch), batch)
        loss.backward()
        optimizer.step()
    assert model.embedder.aux.weight[0].abs().sum() == 0


def test_migrate_state_dict_zero_inits_new_type_components(examples):
    donor = SWDNet(32, 1, 2)
    old_state = dict(donor.state_dict())
    # Simulate an older checkpoint from before the POOL type existed: drop its
    # per-type modules and shrink the type-embedding table by one row.
    removed = [k for k in old_state if ".entity.pool." in k or ".feature.pool." in k]
    assert removed, "expected pool-type parameters in the state dict"
    for key in removed:
        del old_state[key]
    type_key = "embedder.type_embedding.weight"
    old_state[type_key] = old_state[type_key][:-1].clone()

    target = SWDNet(32, 1, 2)
    report = migrate_state_dict(old_state, target)
    assert any(".entity.pool." in k for k in report["zeroed"])
    assert type_key in report["grown"]
    for key in removed:
        assert dict(target.state_dict())[key].abs().sum() == 0
    assert dict(target.state_dict())[type_key][-1].abs().sum() == 0
    # Old-type parameters restored exactly, and the model still runs.
    restored = dict(target.state_dict())
    for key, tensor in old_state.items():
        if key != type_key:
            assert torch.equal(restored[key], tensor)
    target(collate(examples[:4]))


def test_inference_api_batches_match_singles(examples, bot_record):
    from games.seven_wonders_duel.buffer import replay
    from games.seven_wonders_duel.codec import legal_action_indices
    from games.seven_wonders_duel.encoder import encode
    from games.seven_wonders_duel.inference import Evaluator

    states = []

    def keep(game, move):
        if move.i < 5:
            states.append(game.clone())

    replay(bot_record, on_state=keep)
    model = SWDNet(32, 1, 2)
    evaluator = Evaluator(model)
    batched = evaluator.evaluate_states(states)
    singles = [evaluator.evaluate_states([s])[0] for s in states]
    for state, together, alone in zip(states, batched, singles):
        assert len(together.policy) == len(legal_action_indices(state))
        assert together.policy.sum() == pytest.approx(1.0, abs=1e-4)
        assert together.wdl.sum() == pytest.approx(1.0, abs=1e-4)
        assert np.allclose(together.policy, alone.policy, atol=1e-4)
        assert np.allclose(together.wdl, alone.wdl, atol=1e-4)


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
