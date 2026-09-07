"""Workstream 1: the learned Age/slot identity on tableau tokens.

Three things are worth pinning, and only the first is about the embedding:

* the identity is the RIGHT one -- every printed location gets its own name,
  the same name in every game of that Age, and a different one in a different
  Age even where the coordinates coincide;
* switching it on is EXACTLY neutral, so an inherited checkpoint keeps its
  strength while the table starts from nothing (the plan's migration gate);
* the zero start still TRAINS. A zeroed MLP is dead -- that is why the W3
  control head is exempt from zero-init -- but a lookup's gradient does not
  pass through its own value, so that reasoning does not carry over and the
  table needs no warm-up gate. This asserts it rather than trusting it.
"""

import random

import pytest

torch = pytest.importorskip("torch")

from games.seven_wonders_duel.buffer import GameRecorder
from games.seven_wonders_duel.codec import legal_action_indices
from games.seven_wonders_duel.data import TABLEAU_LAYOUTS
from games.seven_wonders_duel.slot_identity import AGE_SLOT_IDS, NUM_AGE_SLOTS
from games.seven_wonders_duel.dataset import TOKEN_TYPES, collate, examples_from_record
from games.seven_wonders_duel.encoder import GLOBAL_FEATURES, TokenType
from games.seven_wonders_duel.game import Phase
from games.seven_wonders_duel.net import SWDNet
from games.seven_wonders_duel.train import (
    load_checkpoint,
    make_checkpoint,
    migrate_state_dict,
    model_from_config,
)

_TABLEAU = TOKEN_TYPES.index(TokenType.TABLEAU)
_AGE_COLUMNS = [GLOBAL_FEATURES.index(f"age_{age}") for age in (1, 2, 3)]
_SLOT_COORDINATES = {value: key for key, value in AGE_SLOT_IDS.items()}


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


def _row_age(example) -> int | None:
    """The Age the row was encoded under, or None outside an Age."""

    onehot = [float(example.features[0][column]) for column in _AGE_COLUMNS]
    if max(onehot) < 0.5:
        return None
    return onehot.index(max(onehot)) + 1


# --- the identity itself ----------------------------------------------------


def test_every_printed_slot_has_its_own_identity():
    total = sum(len(layout) for layout in TABLEAU_LAYOUTS.values())
    assert len(AGE_SLOT_IDS) == total
    assert sorted(AGE_SLOT_IDS.values()) == list(range(NUM_AGE_SLOTS))


def test_the_same_coordinates_in_two_ages_are_two_identities():
    """Age I and Age III both open at (row 0, x 5) and diverge by row 3.

    A shared numbering would hand one name to two different structural roles,
    which is the confusion the per-Age key exists to remove.
    """

    assert AGE_SLOT_IDS[(1, 0, 5)] != AGE_SLOT_IDS[(3, 0, 5)]


def test_derived_ids_match_the_layout_on_a_real_game(examples, batch):
    """The index is reconstructed from `row`, `x` and the GLOBAL Age one-hot.

    Asserted against `data.TABLEAU_LAYOUTS` -- the geometry -- rather than
    against the encoder, so this checks the reconstruction and not that two
    copies of one expression agree.
    """

    ids = _model(slot_embedding=True).embedder.slot_ids(batch)
    seen = set()
    for row, example in enumerate(examples):
        age = _row_age(example)
        for index, type_id in enumerate(example.type_ids):
            slot = int(ids[row, index])
            if type_id != _TABLEAU:
                assert slot == 0, "only tableau tokens carry a slot identity"
                continue
            assert age is not None, "a tableau token implies an Age"
            slot_age, slot_row, slot_x = _SLOT_COORDINATES[slot - 1]
            assert slot_age == age
            assert (slot_row, slot_x) == (
                round(float(example.features[index][0])),
                round(float(example.features[index][2])),
            )
            seen.add(slot - 1)
    assert len(seen) > 20, "a whole game should touch most of the pyramid"


def test_a_draft_row_carries_no_slots(examples, batch):
    """The Wonder draft emits no tableau tokens, so no row of it names a slot.

    Its Age one-hot is still set -- the draft happens "in" Age I -- so what
    protects these rows is the token-type mask, not the Age.
    """

    ids = _model(slot_embedding=True).embedder.slot_ids(batch)
    draft_rows = [
        row
        for row, example in enumerate(examples)
        if not (example.type_ids == _TABLEAU).any()
    ]
    assert draft_rows, "a full game starts with the Wonder draft"
    for row in draft_rows:
        assert int(ids[row].sum()) == 0


def test_a_batch_with_no_age_resolves_to_the_empty_plane(batch):
    """Age 0 is the padding plane.

    Synthetic batches exist -- `w0_sizing` benches on zeroed tensors -- and a
    row whose token 0 is not a real GLOBAL token must land somewhere defined
    rather than indexing out of bounds or silently borrowing Age I.
    """

    ids = _model(slot_embedding=True).embedder.slot_ids(
        {
            "features": torch.zeros_like(batch["features"]),
            "type_ids": torch.full_like(batch["type_ids"], _TABLEAU),
        }
    )
    assert int(ids.sum()) == 0


# --- the migration gate -----------------------------------------------------


def test_switching_it_on_reproduces_the_inherited_model_exactly(batch):
    """Bit-identical, not merely close.

    An added token TYPE is only near-neutral, because a zero-valued token still
    participates in attention normalization. This adds no token, so the stronger
    claim holds and is what is asserted.
    """

    inherited = _model()
    inherited.eval()
    with torch.no_grad():
        before = inherited(batch)

    migrated = _model(slot_embedding=True)
    report = migrate_state_dict(inherited.state_dict(), migrated)
    assert report["neutral"] == ["embedder.slot.weight"]
    assert not report["zeroed"], report["zeroed"]
    assert not migrated.embedder.slot.weight.any()

    migrated.eval()
    with torch.no_grad():
        after = migrated(batch)
    for key, value in before.items():
        assert torch.equal(value, after[key]), key


def test_a_zero_table_still_trains(batch):
    """The reason it needs no warm-up gate."""

    model = _model(slot_embedding=True)
    model.train()
    model(batch)["policy"].square().mean().backward()
    grad = model.embedder.slot.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0
    # `padding_idx` pins "not a tableau token" at zero for good.
    assert not grad[0].any()


def test_the_fused_inference_path_agrees(batch):
    model = _model(slot_embedding=True)
    with torch.no_grad():
        torch.nn.init.normal_(model.embedder.slot.weight)
        model.embedder.slot.weight[0].zero_()
    model.eval()
    with torch.no_grad():
        loop = model(batch)
        model.embedder.fuse()
        fused = model(batch)
    for key, value in loop.items():
        assert torch.allclose(value, fused[key], atol=1e-5), key


def test_the_switch_survives_a_checkpoint_round_trip(tmp_path, batch):
    """A rebuild that forgot the switch would load everything but the table."""

    model = _model(slot_embedding=True)
    with torch.no_grad():
        torch.nn.init.normal_(model.embedder.slot.weight)
        model.embedder.slot.weight[0].zero_()
    checkpoint = make_checkpoint(model, {"d_model": 32, "layers": 2, "heads": 4})
    assert checkpoint["config"]["slot_embedding"] is True
    path = tmp_path / "slots.pt"
    torch.save(checkpoint, path)

    rebuilt = model_from_config(checkpoint["config"])
    load_checkpoint(path, rebuilt)
    rebuilt.eval()
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(batch)["policy"], rebuilt(batch)["policy"])
