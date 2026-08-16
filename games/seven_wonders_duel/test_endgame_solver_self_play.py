"""Gate: the exact endgame solver's contribution to self-play targets.

`test_endgame_solver_rust.py` gates whether the solver is RIGHT, against the
Python reference on 86 banked positions. This file gates what self-play then
does with a right answer, which is a separate and independently wrong-able
thing: a correct optimal set can still be aligned to the wrong action indices,
renormalised over the wrong support, or written into a record the trainer reads
as an estimate rather than a proof.

The masking assertion re-solves each masked position from scratch, through the
same state-injection boundary the advisor uses, rather than trusting the
solver's own report of what it did.
"""

from __future__ import annotations

import pytest

from .buffer import replay
from .rust_bridge import (
    phase_d_record_from_rust,
    rust_game_from_state,
    rust_games_for_self_play,
)
from .test_f4_scheduler import _common

seven_wonders_rust = pytest.importorskip("seven_wonders_rust")

#: Age III only, and small enough that even a wide human-shaped position solves
#: in milliseconds. The trigger is deliberately generous in cards (8) and tight
#: in nodes, matching how a run should be configured.
SOLVER_MAX_NODES = 5_000_000
SOLVER_MAX_CARDS = 8

SEEDS = [2026081610, 2026081611, 2026081612, 2026081613]


@pytest.fixture(autouse=True)
def _solver_disabled_after_each_test():
    """The solver setting is process-wide, so leaking it would silently change
    the targets every other self-play test in this session asserts on."""

    yield
    seven_wonders_rust.set_endgame_solver(0)


def _records(*, max_nodes: int, max_cards: int = SOLVER_MAX_CARDS, mask: bool = True):
    seven_wonders_rust.set_endgame_solver(max_nodes, 60.0, max_cards, mask)
    records, _ = seven_wonders_rust.self_play_many_mock(
        games=rust_games_for_self_play(SEEDS, [0, 1, 0, 1]),
        game_seeds=SEEDS,
        force=True,
        # Every move a full search: the solver only ever fires on those, and a
        # mixed schedule would make "no solved move in this sample" an ambiguous
        # result rather than a failure.
        **(_common(leaf_batch=1, global_batch_cap=8) | {"full_search_fraction": 1.0}),
    )
    return records


def _solved_moves(records):
    return [
        move
        for record in records
        for move in record["moves"]
        if move["solver_value"] is not None
    ]


def _attempted_moves(records):
    return [
        move for record in records for move in record["moves"] if move["solver_attempted"]
    ]


def test_the_disabled_solver_changes_nothing_at_all():
    """The default. Every gate in this package was written against a generator
    without a solver in it, so an off solver must not be a different generator."""

    baseline = _records(max_nodes=0)
    assert _attempted_moves(baseline) == []
    # Off by budget and off by trigger are the same generator, and both are the
    # generator that existed before this feature.
    assert _records(max_nodes=0, max_cards=99) == baseline
    assert _records(max_nodes=SOLVER_MAX_NODES, max_cards=0) == baseline


def test_the_solver_reaches_real_endgames_and_records_what_it_proved():
    records = _records(max_nodes=SOLVER_MAX_NODES)
    solved = _solved_moves(records)
    assert solved, "no position was solved: the trigger never fired"
    for move in solved:
        assert move["solver_regime"] in {"exact", "exact_expectimax"}
        assert -1.0 <= move["solver_value"] <= 1.0
        assert 0 < move["solver_nodes"] <= SOLVER_MAX_NODES
        assert move["solver_masked"] is True
        assert move["solver_attempted"] is True
        assert move["solver_stop"] is None
        if move["solver_regime"] == "exact":
            # Chance-free lines end at a terminal, whose value is the result --
            # which is what makes the W/D/L class sound for these rows and only
            # these rows.
            assert min(abs(move["solver_value"] - v) for v in (-1.0, 0.0, 1.0)) < 1e-9
    # The search's own estimate is kept alongside the proof, not overwritten by
    # it: comparing the two per position is how the solver's value is measured.
    assert all(move["root_value"] is not None for move in solved)


def test_a_masked_target_is_supported_exactly_on_the_proven_optimal_set():
    """The claim the whole design rests on, re-derived rather than trusted.

    Each masked position is replayed out of the record, injected into a fresh
    solver, and solved again. The recorded policy target must put mass on
    exactly the moves that solve proves optimal -- no losing move retained, and
    no optimal move dropped, since dropping one teaches an aversion the position
    does not justify.
    """

    records = _records(max_nodes=SOLVER_MAX_NODES)
    checked = 0
    for raw in records:
        record = phase_d_record_from_rust(raw, validate=False)
        masked = {move.i: move for move in record.moves if move.solver_masked}
        if not masked:
            continue
        positions: dict[int, object] = {}
        counter = [0]

        def capture(game, move, positions=positions, counter=counter):
            if move.i in masked:
                positions[move.i] = game.clone()
            counter[0] += 1

        replay(record, on_state=capture)
        for index, move in masked.items():
            answer = rust_game_from_state(positions[index]).solve_endgame(
                SOLVER_MAX_NODES, 60.0, "exact", "star1"
            )
            assert answer is not None, f"move {index} no longer solves"
            values = {int(k): float(v) for k, v in answer["per_action_value"].items()}
            best = max(values.values())
            optimal = {a for a, value in values.items() if value >= best - 1e-9}
            support = {a for a, p in move.policy_target.items() if p > 0.0}
            assert support == optimal
            assert sum(move.policy_target.values()) == pytest.approx(1.0, abs=1e-9)
            assert float(move.solver_value) == pytest.approx(best, abs=1e-9)
            checked += 1
    assert checked, "no masked move was available to re-verify"


def test_masking_preserves_the_search_ranking_among_the_survivors():
    """The mask must remove mass, never redistribute it.

    Uniform-over-ties was the tempting alternative and is wrong here: 77-88% of
    legal moves at these positions are proven optimal, so flattening them
    deletes the search's discrimination across most of the action space.

    Comparable because of a timing detail: up to and including a game's FIRST
    masked move, the masked and unmasked runs have played identical actions from
    identical states with identical search seeds, and both consume one RNG draw
    to choose the move. So the unmasked run's target at that index is exactly
    the distribution the mask was applied to.
    """

    masked_runs = _records(max_nodes=SOLVER_MAX_NODES)
    plain_runs = _records(max_nodes=SOLVER_MAX_NODES, mask=False)
    compared = 0
    for masked_record, plain_record in zip(masked_runs, plain_runs):
        first = next(
            (move for move in masked_record["moves"] if move["solver_masked"]), None
        )
        if first is None:
            continue
        before = plain_record["moves"][first["i"]]
        assert list(before["legal"]) == list(first["legal"])
        survivors = [
            (original, masked)
            for original, masked in zip(before["policy_target"], first["policy_target"])
            if masked > 0.0
        ]
        assert survivors
        scale = survivors[0][1] / survivors[0][0]
        for original, masked in survivors:
            assert masked == pytest.approx(original * scale, rel=1e-9)
        compared += 1
    assert compared, "no masked move to compare against its unmasked original"


def test_the_value_only_variant_keeps_the_proof_and_leaves_the_policy_alone():
    """The two halves ship behind independent switches on purpose: they have
    different risk profiles, and a run has to be able to attribute a result to
    one of them."""

    records = _records(max_nodes=SOLVER_MAX_NODES, mask=False)
    solved = _solved_moves(records)
    assert solved
    assert not any(move["solver_masked"] for move in solved)
    # No mask means no changed move, so the trajectory is the unsolved one.
    assert [
        [move["action"] for move in record["moves"]] for record in records
    ] == [
        [move["action"] for move in record["moves"]] for record in _records(max_nodes=0)
    ]


def test_a_masked_policy_actually_changes_the_games_that_get_played():
    """The converse of the disabled test: a hook that quietly did nothing would
    pass every assertion above about records it never touched."""

    assert _records(max_nodes=SOLVER_MAX_NODES) != _records(max_nodes=0)


def test_a_declined_solve_records_what_it_cost_and_why():
    """A refusal must not look like a move the trigger never selected.

    With a one-node budget every attempt fails, so the trigger's selections are
    exactly the declined set. Without this, the declined positions cannot be
    found in the buffer and the throughput cost of failed attempts -- which is
    paid synchronously -- cannot be measured at all.
    """

    records = _records(max_nodes=1)
    attempted = _attempted_moves(records)
    assert attempted, "the trigger never fired"
    # Not every attempt fails even at one node: the last card of a game is a
    # single move to a terminal, which the budget covers.
    declined = [move for move in attempted if move["solver_value"] is None]
    assert declined, "a one-node budget declined nothing"
    for move in declined:
        assert move["solver_regime"] is None
        assert move["solver_masked"] is False
        assert move["solver_stop"] in {"budget", "unsolvable"}
    # The cost is visible rather than reported as zero work, which is the whole
    # point: these nodes were spent synchronously inside a scheduler slot.
    assert any(move["solver_nodes"] > 0 for move in declined)


def test_the_value_only_run_does_not_pay_for_exact_per_action_pricing():
    """`PolicyMode::Exact` is needed only by the mask. With the mask off, the
    overlay reads nothing but `root_value`, and the narrower `ValueOnly` window
    -- which also lets star1 bite far harder -- must be what actually runs."""

    masked = _records(max_nodes=SOLVER_MAX_NODES)
    value_only = _records(max_nodes=SOLVER_MAX_NODES, mask=False)
    masked_nodes = sum(move["solver_nodes"] for move in _solved_moves(masked))
    value_only_nodes = sum(move["solver_nodes"] for move in _solved_moves(value_only))
    assert value_only_nodes < masked_nodes


# --- the value target ------------------------------------------------------


def _example(**overrides):
    import numpy as np

    from .dataset import MAX_FEATURES, Example

    base = dict(
        type_ids=np.zeros(1, dtype=np.int8),
        entity_ids=np.zeros(1, dtype=np.int16),
        aux_ids=np.zeros(1, dtype=np.int16),
        features=np.zeros((1, MAX_FEATURES), dtype=np.float16),
        legal=np.asarray([0, 1], dtype=np.int16),
        policy_target=np.asarray([0.5, 0.5], dtype=np.float32),
        has_policy=True,
        value_class=0,
        joint7_class=0,
        margin=0.0,
        margin_valid=False,
        military_final=0.0,
        sci_final_my=0.0,
        sci_final_opp=0.0,
        game_key=1,
        iteration=0,
    )
    return Example(**(base | overrides))


def test_a_chance_free_proof_becomes_a_one_hot_value_target():
    from .dataset import collate

    batch = collate(
        [
            _example(solver_value=1.0, solver_exact=True),
            _example(solver_value=0.0, solver_exact=True),
            _example(solver_value=-1.0, solver_exact=True),
            _example(),
        ]
    )
    assert batch["value_solver_valid"].tolist() == [True, True, True, False]
    assert batch["value_solver"].tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],  # a chance-free 0 is a proven DRAW
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ]


def test_an_expectimax_proof_supplies_no_value_target_at_all():
    """A scalar expected utility does not determine a three-class distribution.

    The solver returns ``P(win) - P(loss)``, and 7WD has real draws, so a 0.0
    could be a certain draw, balanced wins and losses, or any mixture. Turning
    it into ``(0.5, 0, 0.5)`` would be a fabricated proof, and most confidently
    wrong exactly where the truth is most certain -- a position all of whose
    chance outcomes draw. These rows keep the realised outcome instead.
    """

    from .dataset import collate

    batch = collate(
        [
            _example(solver_value=0.0, solver_exact=False),
            _example(solver_value=0.5, solver_exact=False),
        ]
    )
    assert batch["value_solver_valid"].tolist() == [False, False]
    assert batch["value_solver"].sum() == 0.0


def test_a_proven_value_replaces_the_outcome_in_training_but_not_in_validation():
    import torch

    from .dataset import collate
    from .train import compute_losses

    # Row 0 is a proven LOSS whose game was nevertheless won -- exactly the case
    # the substitution exists for, since the opponent then blundered a decided
    # position.
    batch = collate(
        [
            _example(value_class=0, solver_value=-1.0, solver_exact=True),
            _example(value_class=0),
        ]
    )
    outputs = {
        "policy": torch.zeros(2, 1202),
        # Confidently predicting the proven loss.
        "value": torch.tensor([[0.0, 0.0, 20.0], [0.0, 0.0, 20.0]]),
        "joint7": torch.zeros(2, 7),
        "margin": torch.zeros(2),
        "military": torch.zeros(2),
        "science": torch.zeros(2, 2),
    }
    _, trained = compute_losses(outputs, batch)
    _, validated = compute_losses(outputs, batch, solver_value_target=False)
    # Averaged over one row scored against its proof (~0) and one against the
    # realised win (~20), training halves what validation charges for both.
    assert trained["value"] == pytest.approx(10.0, abs=0.1)
    assert validated["value"] == pytest.approx(20.0, abs=0.1)
