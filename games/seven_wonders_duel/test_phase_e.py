"""Phase E trap-suite tests: detector semantics, harvest invariants, and the
exact ground-truth pipeline (structure only — the statistical Tier-1 verdict
comes from a real run, deliberately not asserted here)."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from . import phase_e as pe
from .bots import GreedyBot
from .codec import decode_action, legal_action_indices
from .data import CARDS_BY_NAME
from .engine import ActionUse, apply_action
from .game import Phase, new_game
from .inference import Evaluator
from .search import GumbelMCTS, SearchConfig, state_actor
from .train import build_model


def _state_with_affordable_shield_build():
    """Walk greedy games until the actor can construct a card with shields
    (early game, so no competing science threat muddies the assertions)."""

    bot = GreedyBot()
    for seed in range(200):
        game = new_game(seed)
        for _ in range(40):
            if game.phase is Phase.COMPLETE:
                break
            if game.phase is Phase.PLAY_AGE and game.pending_choice is None:
                for index in legal_action_indices(game):
                    action = decode_action(game, index)
                    if action.use is ActionUse.CONSTRUCT_BUILDING:
                        card = CARDS_BY_NAME[
                            game.tableau.cards[action.slot_id].card_name
                        ]
                        if card.shields >= 1:
                            return game
            apply_action(game, bot.select_action(game))
    raise AssertionError("no shield build found in 200 seeds")


def test_guaranteed_win_military_detection():
    game = _state_with_affordable_shield_build()
    actor = state_actor(game)
    game.conflict_position = 8 if actor == 0 else -8
    assert pe.threat_possible(game, actor)
    assert pe.guaranteed_win_now(game)
    game.conflict_position = 0
    assert not pe.guaranteed_win_now(game)


@pytest.fixture(scope="module")
def harvested_records():
    return pe.fresh_bot_records(12, seed=987)


@pytest.fixture(scope="module")
def harvested(harvested_records):
    found, seen = [], set()
    stats = {"games": 0, "positions": 0, "candidates": 0, "replay_mismatches": 0}
    pe.harvest_records(
        harvested_records,
        "test",
        quota=4,
        per_game_cap=2,
        found=found,
        seen_ids=seen,
        stats=stats,
    )
    assert stats["replay_mismatches"] == 0
    assert found, "no trap positions in 12 bot games — detector likely broken"
    return found


def test_harvest_invariants(harvested):
    for position in harvested:
        traps = set(position["traps"])
        safe = set(position["safe"])
        assert traps and safe
        assert not traps & safe
        by_action = {entry["action"]: entry for entry in position["actions"]}
        for action in traps:
            entry = by_action[action]
            assert entry["n_losing"] > 0
            assert entry["losing_mass"] > 0.0
            assert entry["has_reveal"]
        for action in safe:
            assert by_action[action]["n_losing"] == 0
        state = pe.reconstruct(position)
        assert state_actor(state) == position["actor"]
        assert len(legal_action_indices(state)) == position["n_legal"]
        assert json.loads(json.dumps(position)) == position  # serializable


def test_exact_ground_truth_contract(harvested):
    evaluator = Evaluator(build_model("transformer", 32, 1), device="cpu")
    position = harvested[0]
    state = pe.reconstruct(position)
    truth = pe.exact_ground_truth(state, evaluator, depth=1, frontier_cap=60000)
    assert truth is not None
    legal = set(legal_action_indices(state))
    assert {int(a) for a in truth["exact_q"]} == legal
    assert truth["exact_best"] in legal
    values = [q for q in truth["exact_q"].values()]
    assert all(-1.0 - 1e-9 <= q <= 1.0 + 1e-9 for q in values)
    assert truth["exact_root"] == pytest.approx(max(values))


def test_forced_root_q_is_probability_weighted(harvested):
    evaluator = Evaluator(build_model("transformer", 32, 1), device="cpu")
    position = next(
        p for p in harvested if any(a["n_chains"] > 1 for a in p["actions"])
    )
    state = pe.reconstruct(position)
    mcts = GumbelMCTS(
        evaluator,
        SearchConfig(sims=1, mode="closed", force_expand_root_chance=True),
    )
    root_state = state.clone()
    root_state.search_barrier = True
    root = mcts._make_closed_node(root_state)
    mcts._expand_closed(root)
    mcts._force_expand_root(root)
    edge = next(edge for edge in root.edges if len(edge.children) > 1)
    expected = sum(
        child.probability * child.node.value_p0
        for child in edge.children.values()
    )
    assert edge.visits == 0  # expectation exists before any sampled descent
    assert edge.probability_weighted
    assert edge.q_p0 == pytest.approx(expected)


def test_evaluate_compares_selected_action_q(
    tmp_path, harvested, monkeypatch
):
    position = harvested[0]
    chosen = position["safe"][0]
    exact_q = {str(entry["action"]): 0.9 for entry in position["actions"]}
    exact_q[str(chosen)] = 0.2
    truths = {
        position["id"]: {
            "exact_q": exact_q,
            "exact_root": 0.9,
            "skipped": False,
        }
    }

    class FakeMCTS:
        def __init__(self, _evaluator, _config):
            pass

        def search(self, _state):
            return SimpleNamespace(
                action_index=chosen,
                action_value=0.25,
                root_value=-0.75,
            )

    monkeypatch.setattr(pe, "GumbelMCTS", FakeMCTS)
    args = SimpleNamespace(variants=["closed"], sims=[1], seeds=1, top_k=1)
    rows = pe.run_evaluate(args, tmp_path, [position], truths, evaluator=None)
    assert rows[0]["action_q_error"] == pytest.approx(0.05)
    assert rows[0]["action_regret"] == pytest.approx(0.7)


def test_harvest_commits_only_integrity_checked_games(
    harvested, harvested_records
):
    game_index = int(harvested[0]["id"].split(":")[1])
    corrupt = replace(harvested_records[game_index], final_digest="sha256:corrupt")
    found, persisted = [], []
    stats = {"games": 0, "positions": 0, "candidates": 0, "replay_mismatches": 0}
    pe.harvest_records(
        [corrupt],
        "test",
        quota=2,
        per_game_cap=2,
        found=found,
        seen_ids=set(),
        stats=stats,
        on_found=persisted.append,
    )
    assert stats["replay_mismatches"] == 1
    assert found == persisted == []


def test_report_segments(tmp_path):
    positions = [
        {"id": "a", "traps": [1], "safe": [2], "unsafe_other": []},
        {"id": "b", "traps": [3], "safe": [4], "unsafe_other": []},
    ]
    truths = {
        "a": {"trap_gap": 0.5},
        "b": {"trap_gap": 0.01},
    }
    rows = []
    for pid, picks in (("a", (True, False)), ("b", (True, True))):
        for seed, trap_pick in enumerate(picks):
            rows.append(
                {
                    "id": pid,
                    "variant": "closed",
                    "sims": 32,
                    "seed": seed,
                    "action": 0,
                    "trap_pick": trap_pick,
                    "unsafe_pick": trap_pick,
                    "root_value": 0.0,
                    "action_q_error": 0.1,
                    "ms": 5.0,
                }
            )
    summary = pe.run_report(tmp_path, positions, rows, truths)
    all_cells = summary["segments"]["all"]
    assert all_cells[0]["rows"] == 4
    assert all_cells[0]["trap_pick_rate"] == pytest.approx(0.75)
    consequential_name = f"consequential (trap_gap >= {pe.CONSEQUENTIAL_GAP})"
    cons = summary["segments"][consequential_name]
    assert cons[0]["rows"] == 2
    assert cons[0]["trap_pick_rate"] == pytest.approx(0.5)
