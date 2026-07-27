"""Phase 3b gates: the fused token embedder.

Phase 0 measured `TokenEmbedder` at 73% of the forward's dispatch. The cause is
structural: the per-type loop does boolean-mask indexing, whose output shape is
data-dependent, so every one of the nine types forces host synchronisations on
top of its handful of small kernels.

`fuse()` replaces the loop with one gather and one matmul. It is *not* an exact
refactor — the same arithmetic in a different reduction order — so the gates here
are (a) numerical agreement within a stated bound, (b) that the fused path is
genuinely the same function and not merely close on average, and (c) that the
snapshot can never be trained through, which would silently learn nothing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from .dataset import ENTITY_SPACES, FEATURE_COUNTS, MAX_FEATURES, TOKEN_TYPES
from .net import SWDNet, fuse_for_inference


#: Float reduction order differs, so outputs move. Phase 3b measured ~2e-6 at
#: production widths; 1e-4 is a loose ceiling that still fails loudly if the
#: fused path is ever actually wrong.
TOLERANCE = 1e-4


def _batch(rows=6, tokens=24, seed=17):
    """A batch shaped like the packed boundary: every type present, tail zeroed."""

    generator = torch.Generator().manual_seed(seed)
    type_ids = torch.randint(0, len(TOKEN_TYPES), (rows, tokens), generator=generator)
    entity_ids = torch.zeros(rows, tokens, dtype=torch.long)
    features = torch.zeros(rows, tokens, MAX_FEATURES)
    for index in range(len(TOKEN_TYPES)):
        mask = type_ids == index
        count = int(mask.sum())
        if not count:
            continue
        entity_ids[mask] = torch.randint(
            0, ENTITY_SPACES[index], (count,), generator=generator
        )
        own = torch.zeros(count, MAX_FEATURES)
        own[:, : FEATURE_COUNTS[index]] = torch.rand(
            count, FEATURE_COUNTS[index], generator=generator
        )
        features[mask] = own
    return {
        "type_ids": type_ids,
        "entity_ids": entity_ids,
        "aux_ids": torch.randint(0, 74, (rows, tokens), generator=generator),
        "features": features,
        "pad_mask": torch.zeros(rows, tokens, dtype=torch.bool),
        "actors": torch.zeros(rows, dtype=torch.long),
    }


def _model(seed=3):
    torch.manual_seed(seed)
    return SWDNet(32, 1, 2).eval()


def test_fused_embedder_agrees_with_the_per_type_loop():
    model = _model()
    batch = _batch()
    with torch.no_grad():
        reference = model.embedder(batch)
    model.embedder.fuse()
    with torch.no_grad():
        fused = model.embedder(batch)
    deviation = float((fused - reference).abs().max())
    assert deviation < TOLERANCE, deviation
    # Not merely close on average: every element must agree.
    assert torch.allclose(fused, reference, rtol=0, atol=TOLERANCE)


def test_fused_whole_network_agrees_on_every_head():
    model = _model()
    batch = _batch(rows=9, tokens=31, seed=29)
    with torch.no_grad():
        reference = model(batch)
    fuse_for_inference(model)
    with torch.no_grad():
        fused = model(batch)
    assert set(fused) == set(reference)
    for key in reference:
        deviation = float((fused[key] - reference[key]).abs().max())
        assert deviation < TOLERANCE, f"{key}: {deviation}"


def test_padding_is_respected_by_the_fused_path():
    """Padded positions must stay exactly zero, not merely small."""

    model = _model()
    batch = _batch()
    batch["pad_mask"][:, 5:] = True
    model.embedder.fuse()
    with torch.no_grad():
        fused = model.embedder(batch)
    assert torch.count_nonzero(fused[:, 5:]) == 0


def test_fused_entity_offsets_select_the_right_table():
    """Each type's ids must land in that type's own block of the fused table.

    A wrong offset is the failure mode that agreement-on-random-input can hide,
    because two tables can happen to be close; this checks the mapping directly.
    """

    model = _model()
    model.embedder.fuse()
    fused = model.embedder._fused
    offsets = fused["entity_offsets"]
    table = fused["entity_weight"]
    for index, token_type in enumerate(TOKEN_TYPES):
        own = model.embedder.entity[token_type.value].weight
        block = table[offsets[index] : offsets[index] + ENTITY_SPACES[index]]
        assert torch.equal(block, own), token_type.value
    assert int(offsets[0]) == 0
    assert int(offsets[-1]) + ENTITY_SPACES[-1] == table.shape[0] == sum(ENTITY_SPACES)


def test_fused_feature_weights_are_zero_beyond_each_types_own_width():
    """The padded columns must contribute nothing, whatever the features hold.

    This is what makes the single wide matmul equivalent to nine narrow ones, so
    it is asserted on the weights rather than inferred from outputs.
    """

    model = _model()
    model.embedder.fuse()
    d_model = model.embedder.d_model
    weight = model.embedder._fused["feature_weight"].reshape(
        len(TOKEN_TYPES), d_model, MAX_FEATURES
    )
    for index, token_type in enumerate(TOKEN_TYPES):
        own = model.embedder.feature[token_type.value]
        assert torch.equal(weight[index, :, : own.in_features], own.weight)
        assert torch.count_nonzero(weight[index, :, own.in_features :]) == 0

    # And prove the consequence: garbage past a token's own width changes nothing.
    batch = _batch()
    with torch.no_grad():
        clean = model.embedder(batch)
    for index in range(len(TOKEN_TYPES)):
        mask = batch["type_ids"] == index
        batch["features"][mask, FEATURE_COUNTS[index] :] = 7.5
    with torch.no_grad():
        polluted = model.embedder(batch)
    assert torch.equal(clean, polluted)


def test_checkpoint_layout_is_untouched():
    """Every existing checkpoint must still load, and fusing must not alter keys."""

    model = _model()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model.embedder.fuse()
    after = model.state_dict()
    assert set(after) == set(before)
    for key in before:
        assert torch.equal(after[key], before[key]), key
    # A checkpoint saved before fusing loads into a fused model.
    fresh = _model(seed=99)
    fresh.embedder.fuse()
    fresh.load_state_dict(before)
    assert set(fresh.state_dict()) == set(before)


def test_loading_weights_invalidates_the_snapshot():
    """A stale cache must be impossible, not merely documented.

    The cache copies `entity`/`feature` but reads `type_embedding`/`aux` live, so
    an in-place parameter rewrite would leave it *partially* stale — a silently
    wrong forward. Loading a state dict therefore drops it.
    """

    model = _model()
    batch = _batch()
    model.embedder.fuse()
    with torch.no_grad():
        before = model.embedder(batch)

    other = _model(seed=99)
    model.load_state_dict(other.state_dict())
    assert model.embedder._fused is None, "load_state_dict must invalidate the cache"

    with torch.no_grad():
        after = model.embedder(batch)
    assert not torch.equal(before, after)
    other.eval()
    with torch.no_grad():
        expected = other.embedder(batch)
    assert torch.equal(after, expected), "must reflect the newly loaded weights"

    # Re-fusing then agrees with the loop on the new weights.
    model.embedder.fuse()
    with torch.no_grad():
        refused = model.embedder(batch)
    assert torch.allclose(refused, expected, rtol=0, atol=TOLERANCE)


def test_moving_the_model_invalidates_the_snapshot():
    """`.to()` / `.float()` move parameters but not the cache."""

    model = _model()
    model.embedder.fuse()
    model.float()
    assert model.embedder._fused is None
    model.embedder.fuse()
    model.to(torch.device("cpu"))
    assert model.embedder._fused is None


def test_training_cannot_run_through_a_stale_snapshot():
    """The trap this design has to close.

    A detached snapshot plus a training step would update the real parameters
    while the forward kept reading copies — the model would appear to train and
    learn nothing. `train()` drops the cache, and `forward` refuses the fused path
    while training regardless.
    """

    model = _model()
    model.embedder.fuse()
    assert model.embedder._fused is not None
    model.train()
    assert model.embedder._fused is None, "entering training mode must unfuse"

    # Gradients reach the per-type parameters, i.e. the real ones.
    batch = _batch()
    loss = model(batch)["value"].sum()
    loss.backward()
    for token_type in TOKEN_TYPES:
        entity = model.embedder.entity[token_type.value]
        feature = model.embedder.feature[token_type.value]
        assert feature.weight.grad is not None, token_type.value
        # Not every entity table is necessarily touched by this batch, but the
        # feature projection of a present type always is.
        del entity

    # Even with a cache forced in place, training mode never takes it.
    model.eval()
    model.embedder.fuse()
    model.embedder.train(False)  # stay in eval, keep the cache
    assert model.embedder._fused is not None
    model.embedder.training = True  # simulate a stray flag
    with torch.no_grad():
        forced = model.embedder(batch)
    model.embedder.training = False
    model.embedder.unfuse()
    with torch.no_grad():
        loop = model.embedder(batch)
    assert torch.equal(forced, loop), "training mode must use the per-type loop"


def test_fused_self_play_does_the_same_work():
    """End-to-end: the schedule must be unchanged, not merely the arithmetic.

    Identical batch counts and simulation totals through the real scheduler are
    the strongest available evidence that a ~2e-6 shift flipped no search
    decision.
    """

    import seven_wonders_rust as swr

    from .inference import Evaluator
    from .rust_bridge import rust_flat_batch_adapter, rust_games_for_self_play

    torch.manual_seed(5)
    model = SWDNet(32, 1, 2).eval()
    evaluator = Evaluator(model, device="cpu", max_batch=32)
    seeds = [2026072900 + index for index in range(3)]
    first_players = [0, 1, 0]
    kwargs = dict(
        global_batch_cap=16,
        leaf_batch=4,
        cheap_sims_min=2,
        cheap_sims_max=2,
        full_sims_min=2,
        full_sims_max=2,
        full_search_fraction=0.0,
        top_k=4,
        draft_prior=0.55,
        iteration=3,
        force=False,
        max_inflight_batches=2,
        conflict_free_waves=True,
    )

    def play():
        return swr.self_play_many_flat_net(
            adapter=rust_flat_batch_adapter(evaluator),
            games=rust_games_for_self_play(seeds, first_players),
            game_seeds=seeds,
            **kwargs,
        )

    model.embedder.unfuse()
    _plain_records, plain = play()
    model.embedder.fuse()
    _fused_records, fused = play()

    assert fused["simulations"] == plain["simulations"]
    assert fused["global_batches"] == plain["global_batches"]
    assert fused["moves"] == plain["moves"]
    assert list(fused["batch_rows"]) == list(plain["batch_rows"])
