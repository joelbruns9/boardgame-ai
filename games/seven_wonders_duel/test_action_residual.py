"""W5a legal-action residual: structure, migration, and boundary contracts.

These are deliberately narrow prototype gates. They do not claim playing
strength; they protect the existing policy while the new shared scorer learns.
"""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from .buffer import GameRecorder
from .codec import (
    BUILD_BASE,
    CARD_TO_WONDER_BASE,
    DESTROY_BASE,
    DISCARD_BASE,
    MAUSOLEUM_BASE,
    NEXT_AGE_BASE,
    NUM_ACTIONS,
    NUM_ACTION_FAMILIES,
    PROGRESS_BOARD_BASE,
    PROGRESS_LIBRARY_BASE,
    ActionFamily,
    ActionSource,
    action_components,
    legal_action_indices,
)
from . import dataset, train
from .dataset import (
    MAX_FEATURES, TYPE_IDS, collate, collate_inputs, examples_from_record,
    legal_action_tensors, vectorize,
)
from .encoder import TokenType, encode
from .game import Phase, new_game
from .inference import Evaluator
from .rust_bridge import rust_flat_batch_adapter
from .train import (
    action_residual_from_config,
    compute_losses,
    load_checkpoint,
    make_checkpoint,
    migrate_state_dict,
    model_from_config,
)
from .net import SWDNet


def _playing_game(seed: int = 37):
    game = new_game(seed)
    while game.phase is Phase.WONDER_DRAFT:
        game.pick_wonder(game.legal_wonder_choices()[0])
    game.cities[game.active_player].coins = 100
    return game


def _input_batch(game):
    actor = game.active_player
    legal = legal_action_indices(game)
    vectorized = vectorize(encode(game.observation(actor)))
    return collate_inputs(
        [vectorized], [legal], contextual_actions=True
    ), vectorized, legal


def test_codec_decomposition_covers_the_frozen_action_space():
    expected = {
        ActionFamily.DRAFT_WONDER: 12,
        ActionFamily.BUILD: 73,
        ActionFamily.DISCARD: 73,
        ActionFamily.CONSTRUCT_WONDER: 73 * 12,
        ActionFamily.DESTROY: 73,
        ActionFamily.MAUSOLEUM_REVIVE: 73,
        ActionFamily.PROGRESS_BOARD: 10,
        ActionFamily.PROGRESS_LIBRARY: 10,
        ActionFamily.NEXT_AGE_SELF: 1,
        ActionFamily.NEXT_AGE_OPPONENT: 1,
    }
    counts = {family: 0 for family in ActionFamily}
    for index in range(NUM_ACTIONS):
        counts[action_components(index).family] += 1
    assert counts == expected
    assert len(ActionFamily) == NUM_ACTION_FAMILIES

    assert action_components(BUILD_BASE).source == ActionSource.TABLEAU
    assert action_components(DISCARD_BASE).family == ActionFamily.DISCARD
    construct = action_components(CARD_TO_WONDER_BASE + 4 * 12 + 7)
    assert (construct.source_entity, construct.wonder_entity) == (4, 7)
    assert action_components(DESTROY_BASE).source == ActionSource.CITY_CARD
    assert action_components(MAUSOLEUM_BASE).source == ActionSource.DISCARD
    assert action_components(PROGRESS_BOARD_BASE).source == ActionSource.PROGRESS
    assert action_components(PROGRESS_LIBRARY_BASE).source == ActionSource.PROGRESS
    assert action_components(NEXT_AGE_BASE).source == ActionSource.NONE
    with pytest.raises(ValueError, match="out of range"):
        action_components(NUM_ACTIONS)


def test_legal_candidates_gather_the_exact_contextual_entities():
    batch, _vectorized, legal = _input_batch(_playing_game())
    assert batch["legal_indices"][0, : len(legal)].tolist() == list(legal)
    assert not batch["legal_pad_mask"][0, : len(legal)].any()

    for column, index in enumerate(legal):
        component = action_components(index)
        assert int(batch["action_families"][0, column]) == int(component.family)
        if component.source != ActionSource.NONE:
            token = int(batch["action_source_indices"][0, column])
            assert batch["action_source_present"][0, column]
            assert int(batch["entity_ids"][0, token]) == component.source_entity
        if component.wonder_entity >= 0:
            token = int(batch["action_wonder_indices"][0, column])
            assert batch["action_wonder_present"][0, column]
            assert int(batch["entity_ids"][0, token]) == component.wonder_entity


def test_zero_gate_is_bit_exact_to_the_inherited_policy():
    torch.manual_seed(20260903)
    inherited = SWDNet(32, 1, 2).eval()
    upgraded = SWDNet(32, 1, 2, action_residual=True).eval()
    report = migrate_state_dict(inherited.state_dict(), upgraded)
    assert report["initialized"]
    assert not report["zeroed"]
    assert float(upgraded.action_scorer.gate) == 0.0
    assert not upgraded.action_scorer.gate.requires_grad

    batch, _vectorized, _legal = _input_batch(_playing_game())
    with torch.no_grad():
        before = inherited(batch)
        after = upgraded(batch)
    for key, value in before.items():
        assert torch.equal(value, after[key]), key
    assert after["action_policy"].shape == (1, NUM_ACTIONS)


def _labeled_examples():
    recorder = GameRecorder(83, agents={"p0": "test", "p1": "test"})
    rng = random.Random(8301)
    while recorder.game.phase is not Phase.COMPLETE:
        legal = legal_action_indices(recorder.game)
        choice = rng.choice(legal)
        recorder.play(choice, policy_target={choice: 1.0})
    return examples_from_record(recorder.finish())


def test_independent_policy_loss_trains_the_scorer_behind_a_zero_gate():
    model = SWDNet(32, 1, 2, action_residual=True)
    batch = collate(_labeled_examples()[:4], contextual_actions=True)
    total, _parts = compute_losses(model(batch), batch, action_policy_weight=1.0)
    total.backward()

    assert float(model.action_scorer.gate) == 0.0
    assert model.action_scorer.gate.grad is None
    scorer_grad = model.action_scorer.family.weight.grad
    assert scorer_grad is not None
    assert float(scorer_grad.abs().sum()) > 0.0


def test_training_collation_defaults_off_without_calling_w5(monkeypatch):
    examples = _labeled_examples()[:2]
    enabled = collate(examples, contextual_actions=True)

    def unexpected(*args, **kwargs):
        pytest.fail("legacy collation must not build W5 metadata")

    monkeypatch.setattr(dataset, "legal_action_tensors", unexpected)
    legacy = collate(examples)
    assert "legal_indices" not in legacy
    for key, value in legacy.items():
        assert torch.equal(value, enabled[key]), key


def test_missing_source_is_rejected_only_when_contextual_actions_are_enabled():
    example = _labeled_examples()[0]
    # Keep an otherwise collatable row, but remove every draft-offer identity.
    malformed = replace(example, type_ids=np.zeros_like(example.type_ids))
    assert "legal_indices" not in collate([malformed])
    with pytest.raises(ValueError, match="no contextual"):
        collate([malformed], contextual_actions=True)


def test_repeated_hidden_backs_are_allowed_but_ambiguous_legal_sources_are_not():
    tableau = TYPE_IDS[TokenType.TABLEAU]
    # Repeated hidden backs (entity 73) must not poison a distinct visible card.
    tensors = legal_action_tensors(
        [([tableau] * 3, [73, 73, 0])], [[BUILD_BASE]]
    )
    assert int(tensors["action_source_indices"][0, 0]) == 2
    # Even three duplicates must retain the ambiguity sentinel, not the last
    # token index. Test both primary sources and the secondary Wonder gather.
    with pytest.raises(ValueError, match="no contextual"):
        legal_action_tensors([([tableau] * 3, [0, 0, 0])], [[BUILD_BASE]])
    wonder = TYPE_IDS[TokenType.WONDER]
    with pytest.raises(ValueError, match="no contextual Wonder"):
        legal_action_tensors(
            [([tableau, wonder, wonder], [0, 0, 0])], [[CARD_TO_WONDER_BASE]]
        )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("entrypoint", ["evaluate", "train_steps"])
def test_training_entrypoints_gate_metadata_from_the_model(monkeypatch, enabled, entrypoint):
    examples = _labeled_examples()[:2]
    model = SWDNet(32, 1, 2, action_residual=enabled)
    build_metadata = dataset.legal_action_tensors
    calls = []

    def checked(*args, **kwargs):
        assert enabled, "legacy training/validation must not build W5 metadata"
        calls.append(1)
        return build_metadata(*args, **kwargs)

    monkeypatch.setattr(dataset, "legal_action_tensors", checked)
    if entrypoint == "evaluate":
        train.evaluate(model, examples, "cpu", batch_size=2)
    else:
        train.train_steps(
            model, examples, examples, device="cpu", steps=1,
            batch_size=2, validate_every=1, action_policy_weight=0.1,
            log=lambda _: None,
        )
    assert bool(calls) == enabled


def test_padded_candidates_have_no_output_or_gradient_even_without_score_masking(monkeypatch):
    _batch, vectorized, legal = _input_batch(_playing_game())
    batch = collate_inputs(
        [vectorized] * 3, [legal[:1], legal[:2], []], contextual_actions=True
    )
    padded = batch["legal_pad_mask"]
    assert torch.all(batch["legal_indices"][padded] == NUM_ACTIONS)
    # Exercise action 0 explicitly at the scatter boundary. Context gathering
    # is tested separately; these remapped action indices are synthetic.
    batch["legal_indices"][0, 0] = 0
    batch["legal_indices"][1, :2] = torch.tensor([0, 1])
    scores = torch.tensor([[2., 91.], [3., 5.], [99., 101.]], requires_grad=True)
    model = SWDNet(32, 1, 2, action_residual=True).eval()
    monkeypatch.setattr(model.action_scorer, "forward", lambda *args: scores)
    residual = model(batch)["action_policy"]
    expected = torch.zeros(3, NUM_ACTIONS)
    expected[0, 0] = 2.
    expected[1, :2] = torch.tensor([3., 5.])
    assert torch.equal(residual, expected)
    residual.sum().backward()
    assert torch.equal(scores.grad, (~padded).to(scores.dtype))


def test_checkpoint_migration_and_rebuild_keep_w5_optional(tmp_path):
    inherited = SWDNet(32, 1, 2)
    path = tmp_path / "inherited.pt"
    torch.save(
        make_checkpoint(
            inherited,
            {"model": "transformer", "d_model": 32, "layers": 1, "heads": 2},
        ),
        path,
    )
    upgraded = SWDNet(32, 1, 2, action_residual=True)
    checkpoint = load_checkpoint(path, upgraded, migrate=True)
    assert checkpoint["migration"]["initialized"]
    assert float(upgraded.action_scorer.gate) == 0.0

    upgraded_checkpoint = make_checkpoint(
        upgraded,
        {"model": "transformer", "d_model": 32, "layers": 1, "heads": 2},
    )
    assert action_residual_from_config(upgraded_checkpoint["config"])
    rebuilt = model_from_config(upgraded_checkpoint["config"])
    rebuilt.load_state_dict(upgraded_checkpoint["model_state"])
    assert rebuilt.action_residual

    with pytest.raises(ValueError, match="additive only"):
        migrate_state_dict(upgraded.state_dict(), inherited)


def test_phase_d_defaults_to_shadow_and_requires_explicit_integration():
    from .phase_d import PhaseDConfig, PhaseDLoop

    shadow = PhaseDConfig(action_residual=True, action_policy_weight=0.1)
    loop = PhaseDLoop.__new__(PhaseDLoop)
    loop.config = shadow
    assert not loop._new_model().action_scorer.gate.requires_grad

    integrated = PhaseDConfig(
        action_residual=True,
        action_policy_weight=0.1,
        train_action_gate=True,
    )
    loop.config = integrated
    assert loop._new_model().action_scorer.gate.requires_grad

    with pytest.raises(ValueError, match="positive --action-policy-weight"):
        PhaseDConfig(action_residual=True).validate()
    with pytest.raises(ValueError, match="requires --action-residual"):
        PhaseDConfig(train_action_gate=True).validate()


def _flat_payload(vectorized, legal):
    type_ids, entity_ids, aux_ids, features = vectorized
    tokens = len(type_ids)
    return {
        "rows": 1,
        "tokens": tokens,
        "max_tokens": tokens,
        "feature_width": MAX_FEATURES,
        "token_offsets": bytearray(np.asarray([0, tokens], dtype="<i4").tobytes()),
        "type_ids": bytearray(type_ids.astype(np.uint8).tobytes()),
        "entity_ids": bytearray(entity_ids.astype("<i2").tobytes()),
        "aux_ids": bytearray(aux_ids.astype("<i2").tobytes()),
        "features": bytearray(features.astype("<f4").tobytes()),
        "actors": bytearray(np.asarray([0], dtype=np.uint8).tobytes()),
        "legal_offsets": bytearray(
            np.asarray([0, len(legal)], dtype="<i4").tobytes()
        ),
        "legal_actions": bytearray(np.asarray(legal, dtype="<u2").tobytes()),
    }


def test_flat_rust_boundary_builds_w5_metadata_only_when_enabled():
    game = _playing_game()
    _batch, vectorized, legal = _input_batch(game)
    payload = _flat_payload(vectorized, legal)

    plain = rust_flat_batch_adapter(Evaluator(SWDNet(32, 1, 2), device="cpu"))
    plain_batch, *_rest = plain.build_device_batch(payload)
    assert "legal_indices" not in plain_batch

    model = SWDNet(32, 1, 2, action_residual=True)
    contextual = rust_flat_batch_adapter(Evaluator(model, device="cpu"))
    contextual_batch, *_rest = contextual.build_device_batch(payload)
    assert contextual_batch["legal_indices"][0, : len(legal)].tolist() == list(legal)
    with torch.no_grad():
        assert model(contextual_batch)["policy"].shape == (1, NUM_ACTIONS)
