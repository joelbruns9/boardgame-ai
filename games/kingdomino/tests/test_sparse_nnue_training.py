"""Correctness gates for the Step-3 sparse training scaffold."""
from __future__ import annotations

import json
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
kr = pytest.importorskip("kingdomino_rust")

from games.kingdomino.board import Placement
from games.kingdomino.game import GameState, Phase, PickAction, TurnAction
from games.kingdomino.nnue import datagen
from games.kingdomino.nnue import sparse_encoder as se
from games.kingdomino.nnue import summary_encoder as sm
from games.kingdomino.nnue.d4 import D4_ELEMENTS, sparse_perm, summary_perm
from games.kingdomino.nnue.sparse_data import (
    AUX_SCORE_SCALES,
    MARGIN_SCALE,
    derive_records,
    load_packed,
    save_packed,
)
from games.kingdomino.nnue.sparse_net import SparseNNUE


def _random_record(seed=913):
    gs = GameState.new(seed=seed)
    rs = kr.RustGameState(
        int(gs.start_player), list(gs.deck), list(gs.current_row),
        gs.config.harmony, gs.config.middle_kingdom,
    )
    rng = random.Random(seed)
    actions = []
    while rs.phase != datagen.GAME_OVER:
        legal = rs.legal_actions()
        action = legal[rng.randrange(len(legal))]
        actions.append(datagen._ser_action(action))
        rs = rs.step(action[0], action[1])
    scores = [int(x) for x in rs.scores()]
    return {
        "seed": seed,
        "start_player": int(gs.start_player),
        "deck": [int(x) for x in gs.deck],
        "current_row": [int(x) for x in gs.current_row],
        "harmony": bool(gs.config.harmony),
        "middle_kingdom": bool(gs.config.middle_kingdom),
        "actions": actions,
        "final_scores": scores,
        "outcome_p0": int(kr.SearchEngine(rs).official_outcome()),
        "n_positions": len(actions),
        "engine_version": datagen.ENGINE_VERSION,
        "format_version": datagen.FORMAT_VERSION,
        "catalog_hash": datagen.catalog_hash(),
    }, rs


def test_embedding_bag_equals_explicit_column_sum():
    torch.manual_seed(3)
    net = SparseNNUE(12, 3, acc_width=5, tail_hidden=4)
    indices = torch.tensor([1, 4, 7, 0, 11], dtype=torch.long)
    offsets = torch.tensor([0, 3, 5], dtype=torch.long)
    summary = torch.zeros((2, 3))
    got = net.accumulator(indices, offsets) + net.accumulator_bias
    want = torch.stack([
        net.accumulator.weight[[1, 4, 7]].sum(0),
        net.accumulator.weight[[0, 11]].sum(0),
    ]) + net.accumulator_bias
    assert torch.equal(got, want)
    outputs = net(indices, offsets, summary)
    assert [tuple(x.shape) for x in outputs] == [(2,), (2,), (2, 6), (2, 4)]


def test_fixed_weight_network_output_seat_swap():
    """Historical hand asymmetry stays absent through the complete network frame."""
    state = GameState.new(seed=41)
    rng = random.Random(41)
    for _ in range(17):
        state = state.step(rng.choice(state.legal_actions()))
    assert state.phase != Phase.GAME_OVER
    swapped = se.swap_players(state)

    torch.manual_seed(2026)  # fixed, untrained weights: this is a structural gate
    net = SparseNNUE(se.CORE_SIZE, sm.SUMMARY_SIZE, acc_width=32, tail_hidden=16).eval()

    def evaluate(st):
        actor = int(st.current_actor)
        idx = se.encode_core(st, actor).astype(np.int64)
        summary = sm.encode_summary(st, actor)[None]
        with torch.no_grad():
            prob, margin = net.evaluate(
                torch.from_numpy(idx),
                torch.tensor([0, len(idx)], dtype=torch.long),
                torch.from_numpy(summary),
            )
        actor_value = 2.0 * prob[0] - 1.0
        p0_value = actor_value if actor == 0 else -actor_value
        p0_margin = margin[0] if actor == 0 else -margin[0]
        return actor, prob[0], margin[0], p0_value, p0_margin

    a = evaluate(state)
    b = evaluate(swapped)
    assert b[0] == 1 - a[0]
    assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
    assert torch.equal(a[3], -b[3]) and torch.equal(a[4], -b[4])


def test_replay_derivation_targets_csr_d4_and_roundtrip(tmp_path):
    rec, terminal = _random_record()
    data = derive_records([rec])
    assert len(data) == rec["n_positions"]
    assert data.offsets[-1] == len(data.indices)
    assert set(data.actors.tolist()) == {0, 1}
    assert data.metadata["core_schema_hash"] == se.core_schema_hash()
    assert data.metadata["summary_schema_hash"] == sm.summary_schema_hash()

    outcome_p0 = rec["outcome_p0"]
    scores = rec["final_scores"]
    breakdowns = terminal.score_breakdowns()
    for row, actor in enumerate(data.actors):
        actor = int(actor)
        expected_outcome = ((outcome_p0 if actor == 0 else -outcome_p0) + 1) / 2
        assert data.outcome[row] == expected_outcome
        assert data.margin[row] == pytest.approx(
            (scores[actor] - scores[1 - actor]) / MARGIN_SCALE
        )
        ordered = (breakdowns[actor], breakdowns[1 - actor])
        raw_aux = np.asarray(
            [ordered[0][1], ordered[0][2], ordered[0][3],
             ordered[1][1], ordered[1][2], ordered[1][3]], np.float32,
        )
        assert np.array_equal(data.aux_scores[row], raw_aux / AUX_SCORE_SCALES)

    row = 9
    base_start, base_stop = data.offsets[row:row + 2]
    base_idx = data.indices[base_start:base_stop]
    for choice, (k, flip) in enumerate(D4_ELEMENTS):
        batch = data.batch([row], d4_choices=choice)
        expected_idx = sparse_perm(k, flip)[base_idx]
        assert np.array_equal(np.sort(batch["indices"].numpy()), np.sort(expected_idx))
        expected_summary = summary_perm(k, flip)(data.summaries[row])
        assert np.array_equal(batch["summary"].numpy()[0], expected_summary)

    path = save_packed(data, tmp_path / "one_game.npz")
    loaded = load_packed(path)
    for name in ("indices", "offsets", "summaries", "outcome", "margin",
                 "aux_scores", "aux_bonus", "actors", "game_index"):
        assert np.array_equal(getattr(data, name), getattr(loaded, name))

    # Semantic hashes are enforced on load, not merely recorded for inspection.
    with np.load(path, allow_pickle=False) as z:
        arrays = {name: z[name].copy() for name in z.files}
    meta = json.loads(str(arrays["metadata"].item()))
    meta["core_schema_hash"] = "stale00000000000"
    arrays["metadata"] = np.asarray(json.dumps(meta))
    stale = tmp_path / "stale.npz"
    np.savez(stale, **arrays)
    with pytest.raises(ValueError, match="core_schema_hash"):
        load_packed(stale)


def test_rust_final_breakdown_matches_python():
    rec, rust_terminal = _random_record(seed=111)
    py = GameState.new(seed=111)
    for sa in rec["actions"]:
        placement, pick = datagen._deser_action(sa)
        if py.phase == Phase.INITIAL_SELECTION:
            action = PickAction(pick)
        else:
            action = TurnAction(None if placement is None else Placement(*placement), pick)
        py = py.step(action)
    assert py.phase == Phase.GAME_OVER
    got = rust_terminal.score_breakdowns()
    for player in (0, 1):
        sb = py.boards[player].score(py.config.harmony, py.config.middle_kingdom)
        assert tuple(got[player]) == (
            sb.total,
            sb.territory_score,
            sb.largest_territory_size,
            sb.total_crowns,
            sb.harmony_bonus,
            sb.middle_kingdom_bonus,
        )
