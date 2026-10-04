"""G2 (`MODEL_GROWTH_PLAN.md`): the value target contract, per head and proof type.

Acceptance is "effective targets equal the proved quantity in the correct
player's frame", checked at each boundary a target crosses: derivation (both
backends), batching (`collate` and the packed W0 path), target construction,
the loss, and the head search serves.
"""

from __future__ import annotations

from dataclasses import replace
import random

import pytest

torch = pytest.importorskip("torch")

from .buffer import GameRecorder, ReplayMismatchError
from .codec import legal_action_indices
from .dataset import (
    certain_win_moves,
    collate,
    derive_records_rust,
    examples_from_record,
)
from .game import Phase
from .net import SWDNet
from .train import _utility_loss, compute_losses, value_targets


def _record(seed: int, *, iteration: int = 3):
    recorder = GameRecorder(seed, first_player=seed % 2, iteration=iteration)
    rng = random.Random(seed ^ 0x6A2)
    while recorder.game.phase is not Phase.COMPLETE:
        legal = legal_action_indices(recorder.game)
        visits = {candidate: offset + 1 for offset, candidate in enumerate(legal)}
        recorder.play(
            rng.choice(legal),
            visits=visits,
            root_value=rng.random() * 2.0 - 1.0,
            sims=sum(visits.values()),
        )
    return recorder.finish()


@pytest.fixture(scope="module")
def records():
    return [_record(seed) for seed in range(40)]


def _hier_model():
    torch.manual_seed(0)
    return SWDNet(d_model=32, layers=2, heads=4, hierarchical_value=True,
                  hierarchical_value_detach=False)


# --- derivation --------------------------------------------------------------


def test_certain_win_tail_stops_at_the_opponent_and_at_chance(records):
    record = next(r for r in records if r.winner is not None)
    moves = list(record.moves)
    winner = record.winner
    n = len(moves)
    # Last three moves the winner's, the one before the opponent's.
    actors = [1 - winner] * (n - 3) + [winner] * 3
    forged = replace(
        record, moves=tuple(replace(m, actor=a) for m, a in zip(moves, actors))
    )
    no_chance = [0] * n
    assert certain_win_moves(forged, no_chance) == {n - 3, n - 2, n - 1}
    # Chance on the LAST move is after the win and does not matter...
    assert certain_win_moves(forged, no_chance[:-1] + [2]) == {n - 3, n - 2, n - 1}
    # ...chance inside the tail makes everything before it a gamble.
    gamble = list(no_chance)
    gamble[n - 2] = 1
    assert certain_win_moves(forged, gamble) == {n - 1}
    assert certain_win_moves(replace(forged, winner=None), no_chance) == set()
    with pytest.raises(ReplayMismatchError):
        certain_win_moves(forged, no_chance[:-1])


def test_both_backends_flag_the_same_certain_wins(records):
    derived = derive_records_rust(records, batch_games=8)
    flagged = 0
    for record, (rust_examples, _stats) in zip(records, derived):
        python_examples = examples_from_record(record)
        assert [e.certain_win for e in python_examples] == [
            e.certain_win for e in rust_examples
        ]
        for example in python_examples:
            if example.certain_win:
                flagged += 1
                # Actor frame: a certain row is the actor's own exact win.
                assert example.value_class == 0
                assert example.joint7_class in (0, 1, 2)
    # Random games end on the winner's move often enough to exercise this.
    assert flagged > 0


# --- batching ----------------------------------------------------------------


def _labelled(records):
    """Examples with one exact proof, one expectimax proof and certain rows."""

    examples = []
    for record in records[:6]:
        examples.extend(examples_from_record(record))
    examples[0] = replace(examples[0], solver_value=-1.0, solver_exact=True)
    examples[1] = replace(examples[1], solver_value=0.4, solver_exact=False)
    assert any(e.certain_win for e in examples)
    return examples


def test_collate_and_packed_batches_carry_the_same_contract(records):
    from .w0_sizing import _pack_examples

    examples = _labelled(records)
    batch = collate(examples)
    packed = _pack_examples(examples, 0.0, "g2")["storage"]
    for key in ("value_solver_valid", "value_solver_exact", "value_solver_utility",
                "value_certain", "value_solver"):
        assert torch.equal(batch[key], packed[key][: len(examples)]), key
    assert batch["value_solver_exact"][0] and not batch["value_solver_exact"][1]
    assert batch["value_solver_utility"][1] == pytest.approx(0.4)


# --- targets -----------------------------------------------------------------


@pytest.fixture(scope="module")
def batch(records):
    return collate(_labelled(records))


def test_legacy_hierarchical_loss_is_the_old_joint_nll(batch):
    model = _hier_model()
    outputs = model(batch)
    _, parts = compute_losses(outputs, batch, value_target_contract="legacy",
                              outlook_bootstrap=0.0)
    old = torch.nn.functional.nll_loss(outputs["hier_joint7"].detach(), batch["joint7"])
    assert parts["hier_value"] == pytest.approx(float(old), rel=1e-5)


def test_g2_targets_equal_the_proved_quantity(batch):
    targets = value_targets(batch, value_bootstrap=0.5, short_term_value_weight=0.25,
                            outlook_bootstrap=0.5, hierarchical=True)
    loss = torch.tensor([0.0, 0.0, 1.0])
    legacy = value_targets(batch, outlook_bootstrap=0.5, hierarchical=True,
                           contract="legacy")
    # Exact proof: both heads' outcome target IS the proof.
    assert torch.equal(targets["flat"][0], loss)
    assert torch.equal(targets["hier_outcome"][0], loss)
    # Expectimax proof: utility only, on both heads.
    assert bool(targets["utility_rows"][1]) and bool(targets["hier_utility_rows"][1])
    # A proof neither supplies nor removes a victory type: the conditional keeps
    # the realised (outlook-blended) type mass it always had.
    for row in (0, 1):
        assert torch.equal(targets["hier_type"][row], legacy["hier_type"][row])
    # Certain win: exact outcome and exact victory type, nothing blended in.
    certain = batch["value_certain"].nonzero().flatten()
    for row in certain.tolist():
        assert torch.equal(targets["flat"][row], torch.tensor([1.0, 0.0, 0.0]))
        assert torch.equal(targets["hier_outcome"][row], torch.tensor([1.0, 0.0, 0.0]))
        expected = torch.zeros(7)
        expected[int(batch["joint7"][row])] = 1.0
        assert torch.equal(targets["hier_type"][row], expected)
        assert not bool(targets["utility_rows"][row])


def test_the_short_term_term_now_reaches_the_hierarchical_head(batch):
    plain = value_targets(batch, hierarchical=True)
    short = value_targets(batch, short_term_value_weight=0.25, hierarchical=True)
    open_rows = (
        batch["value_short_valid"]
        & ~batch["value_solver_valid"]
        & ~batch["value_certain"]
    )
    assert bool(open_rows.any())
    assert not torch.allclose(plain["hier_outcome"][open_rows],
                              short["hier_outcome"][open_rows])


def test_the_utility_loss_is_minimised_at_the_proved_expectation():
    utility = torch.tensor([0.4])
    # Two heads with the same P(win) - P(loss) = 0.4 but different draw mass.
    for probs in ([0.7, 0.0, 0.3], [0.6, 0.2, 0.2]):
        logits = torch.tensor([probs]).log().requires_grad_(True)
        loss = _utility_loss(torch.log_softmax(logits, dim=-1), utility)
        loss.sum().backward()
        assert float(logits.grad.abs().max()) < 1e-5
    off = _utility_loss(torch.tensor([[0.5, 0.0, 0.5]]).log(), utility)
    on = _utility_loss(torch.tensor([[0.7, 0.0, 0.3]]).log(), utility)
    assert float(off) > float(on)


def test_validation_is_unchanged_by_the_contract(batch):
    model = _hier_model()
    outputs = model(batch)
    kwargs = dict(solver_value_target=False, row_weights=False)
    _, legacy = compute_losses(outputs, batch, value_target_contract="legacy", **kwargs)
    _, g2 = compute_losses(outputs, batch, **kwargs)
    # Certain rows' exact target equals their realised one, so with proofs off
    # the held-out numbers mean the same thing under either contract.
    assert g2["value"] == pytest.approx(legacy["value"], rel=1e-6)
    assert g2["hier_value"] == pytest.approx(legacy["hier_value"], rel=1e-6)


# --- the served head ---------------------------------------------------------


def test_training_moves_the_served_hierarchical_value_toward_the_proof(batch):
    """End to end: proofs reach the head `value_source='hierarchical'` serves."""

    rows = 32
    small = {k: v[:rows] for k, v in batch.items()}
    small["value_solver_valid"] = torch.ones(rows, dtype=torch.bool)
    small["value_solver_exact"] = torch.ones(rows, dtype=torch.bool)
    small["value_certain"] = torch.zeros(rows, dtype=torch.bool)
    # Prove the OPPOSITE of the realised outcome on every row.
    flipped = torch.zeros(rows, 3)
    flipped[torch.arange(rows), 2 - small["value_class"].long()] = 1.0
    small["value_solver"] = flipped

    def served_on_proof(model):
        with torch.no_grad():
            outcome = model(small)["hier_value"].exp()
        return float((outcome * flipped).sum(dim=1).mean())

    for contract, should_move in (("g2", True), ("legacy", False)):
        model = _hier_model()
        before = served_on_proof(model)
        optimizer = torch.optim.Adam(model.hier_value.parameters(), lr=1e-2)
        for _ in range(60):
            optimizer.zero_grad()
            total, _ = compute_losses(model(small), small, hier_value_weight=1.0,
                                      value_target_contract=contract)
            total.backward()
            optimizer.step()
        after = served_on_proof(model)
        if should_move:
            assert after > before + 0.2, (before, after)
        else:
            assert after < before + 0.05, (before, after)
