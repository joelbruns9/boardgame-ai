from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import pytest

from games.kingdomino.action_codec import encode_action
from games.kingdomino.denial_search import (
    DenialSearch,
    EvalStats,
    SearchConfig,
    _as_pre_reveal_leaf,
    _public_state_key_uncached,
    chance_rows,
    denial_policy_target,
    public_state_key,
)
from games.kingdomino.game import Claim, GameState, Phase
from games.kingdomino.denial_signal_sweep import (
    CELL_SPECS,
    STABILITY_NAMES,
    _parse_args,
    load_frozen_positions,
    run_cell,
    write_frozen_positions,
)


def test_crn_determinism_across_sibling_bag_orders():
    bag = list(range(1, 17))
    rows_a, mode_a = chance_rows(bag, 8, seed=1234)
    rows_b, mode_b = chance_rows(list(reversed(bag)), 8, seed=1234)
    assert mode_a == mode_b == "sampled"
    assert rows_a == rows_b
    assert all(tuple(sorted(row)) == row for row in rows_a)


def test_exact_when_combination_count_is_at_most_k():
    bag = list(range(1, 9))
    k = math.comb(len(bag), 4)
    rows, mode = chance_rows(bag, k, seed=7)
    assert mode == "enumerated"
    assert rows == list(itertools.combinations(bag, 4))
    # Once exact, changing the seed cannot change the distribution.
    assert chance_rows(bag, k + 10, seed=999)[0] == rows


def test_policy_target_is_valid_and_uncertain_ties_share_mass():
    policy = denial_policy_target([0.3, 0.3, 0.1], [0.0, 0.0, 0.2], temperature=0.1)
    assert sum(policy) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in policy)
    assert policy[0] == pytest.approx(policy[1])
    assert policy[2] > 0.0


def test_forced_pick_policy_is_one_hot():
    assert denial_policy_target([-0.73], [0.4]) == [1.0]


class _ConstantEvaluator:
    def __init__(self):
        self.policy_cache = {}
        self.leaf_cache = {}
        self.stats = EvalStats()

    def policies(self, states):
        for state in states:
            actions = state.legal_actions()
            p = 1.0 / len(actions)
            self.policy_cache[public_state_key(state)] = {
                int(encode_action(action, state)): p for action in actions
            }

    def policy(self, state):
        self.policies([state])
        return self.policy_cache[public_state_key(state)]

    def values_p0(self, states):
        out = {}
        for state in states:
            key = public_state_key(state)
            self.leaf_cache[key] = 0.0
            out[key] = 0.0
        return out


def test_forced_pick_search_emits_single_legal_pick(tmp_path):
    state = GameState.new(seed=17)
    # Finish opening selection, then advance three claims in the first normal
    # round.  The fourth claim has exactly one tile left to pick.
    while state.phase == Phase.INITIAL_SELECTION:
        state = state.step(state.legal_actions()[0])
    for _ in range(3):
        state = state.step(state.legal_actions()[0])
    assert len(state.current_row) == 1
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    search = DenialSearch(
        _ConstantEvaluator(), checkpoint_path=str(checkpoint),
        config=SearchConfig(pick_plies=1, chance_k=2, root_search_sims=0,
                            placement_top_k=1),
    )
    label = search.search_position(state)
    assert label["status"] == "ok"
    assert label["legal_pick_ids"] == [state.current_row[0]]
    assert label["policy_target"] == [1.0]
    assert label["structure"]["completed"]


def test_extract_reply_labels_are_grouped_complete_and_opponent_only(tmp_path):
    state = next(
        candidate for seed in range(100, 200)
        if (
            (candidate := _round_start(seed)).pending_claims[0].player
            != candidate.pending_claims[1].player
        )
    )
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    search = DenialSearch(
        _ConstantEvaluator(), checkpoint_path=str(checkpoint),
        config=SearchConfig(pick_plies=2, chance_k=1, root_search_sims=0,
                            placement_top_k=2),
    )

    root_label = search.search_position(state)
    replies = search.extract_reply_labels(state, root_label=root_label)

    assert len(replies) == len(root_label["legal_pick_ids"])
    for reply in replies:
        assert reply["actor"] != reply["root_actor"]
        assert reply["schema_version"] == 1
        assert reply["state_key"] == public_state_key(reply["_state"])
        assert sum(reply["denial_policy_target"]) == pytest.approx(1.0)
        assert len(reply["per_pick"]) == len(reply["legal_pick_ids"])
        legal = reply["_state"].legal_actions()
        assert [row["action_idx"] for row in reply["legal_actions"]] == sorted(
            encode_action(action, reply["_state"]) for action in legal)
        for pick_row in reply["per_pick"]:
            conditional = pick_row["baseline_conditional_placements"]
            assert conditional
            assert sum(row["conditional_probability"] for row in conditional) == pytest.approx(1.0)


def test_extract_reply_labels_skips_same_player_continuation(tmp_path):
    state = _round_start(101)
    first = state.pending_claims[0]
    second = state.pending_claims[1]
    state.pending_claims[1] = Claim(first.player, second.domino_id)
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    search = DenialSearch(
        _ConstantEvaluator(), checkpoint_path=str(checkpoint),
        config=SearchConfig(pick_plies=2, chance_k=1, root_search_sims=0,
                            placement_top_k=1),
    )
    root_label = search.search_position(state)

    assert search.extract_reply_labels(state, root_label=root_label) == []
    assert search.extract_reply_labels(
        state, root_label=root_label, opponent_only=False)


def test_game_over_search_emits_outcome_without_policy(tmp_path):
    state = GameState.new(seed=23)
    while state.phase != Phase.GAME_OVER:
        state = state.step(state.legal_actions()[0])
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    search = DenialSearch(
        _ConstantEvaluator(), checkpoint_path=str(checkpoint),
        config=SearchConfig(pick_plies=8, chance_k=2, root_search_sims=0),
    )
    label = search.search_position(state)
    assert label["status"] == "game_over"
    assert label["policy_target"] == []
    assert -1.0 <= label["corrected_value_player0"] <= 1.0


def _normalised_label(label):
    import copy
    out = copy.deepcopy(label)
    out.get("provenance", {}).pop("elapsed_seconds", None)
    return out


def _round_start(seed=31):
    state = GameState.new(seed=seed)
    while state.phase == Phase.INITIAL_SELECTION:
        state = state.step(state.legal_actions()[0])
    return state


def test_root_search_cache_reuses_identical_public_root_and_preserves_label(tmp_path):
    state = _round_start()
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    evaluator = _ConstantEvaluator()
    search = DenialSearch(
        evaluator, checkpoint_path=str(checkpoint),
        config=SearchConfig(pick_plies=1, chance_k=1, root_search_sims=1,
                            placement_top_k=1),
    )
    policy = evaluator.policy(state)
    result = (
        {idx: float(probability) for idx, probability in policy.items()},
        0.0,
        {idx: (float(probability), 0.0) for idx, probability in policy.items()},
    )
    calls = []

    def fake_compute(_state, seed):
        calls.append(seed)
        evaluator.stats.root_search_calls += 1
        return result

    search._root_search_compute = fake_compute
    first = search._root_search(state)
    second = search._root_search(state.copy())
    assert first is second
    assert len(calls) == 1
    assert evaluator.stats.root_search_cache_hits == 1

    cached_label = search.search_position(state)
    search.clear_root_search_cache()
    search._node_tt.clear()
    uncached_label = search.search_position(state)
    assert _normalised_label(cached_label) == _normalised_label(uncached_label)


def test_four_ply_derived_from_eight_matches_independent_search(tmp_path):
    state = _round_start(seed=37)
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    full = DenialSearch(
        _ConstantEvaluator(), checkpoint_path=str(checkpoint),
        config=SearchConfig(pick_plies=8, chance_k=1, root_search_sims=0,
                            placement_top_k=1),
    )
    full.search_position(state)
    derived = full.derive_four_ply_position(state)

    independent = DenialSearch(
        _ConstantEvaluator(), checkpoint_path=str(checkpoint),
        config=SearchConfig(pick_plies=4, chance_k=1, root_search_sims=0,
                            placement_top_k=1),
    ).search_position(state)
    assert _normalised_label(derived) == _normalised_label(independent)
    assert full.evaluator.stats.node_tt_pass_hits > 0


def test_public_state_key_memo_matches_uncached_across_fresh_states():
    initial = GameState.new(seed=41)
    assert public_state_key(initial) == _public_state_key_uncached(initial)

    copied = initial.copy()
    assert not hasattr(copied, "_denial_public_state_key")
    assert public_state_key(copied) == _public_state_key_uncached(copied)

    stepped = initial.step(initial.legal_actions()[0])
    assert public_state_key(stepped) == _public_state_key_uncached(stepped)

    boundary = _round_start(seed=43)
    for _ in range(3):
        boundary = boundary.step(boundary.legal_actions()[0])
    dealt = boundary.step(boundary.legal_actions()[0])
    pre_reveal = _as_pre_reveal_leaf(dealt, boundary.deck)
    assert public_state_key(pre_reveal) == _public_state_key_uncached(pre_reveal)


def test_frozen_position_roundtrip_is_deterministic_and_config_independent(tmp_path):
    records = [(_round_start(seed=seed), {"seed": seed}) for seed in (51, 52, 53)]
    path = tmp_path / "positions.jsonl"
    written = write_frozen_positions(records, path)
    first = load_frozen_positions(path)
    second = load_frozen_positions(path)
    first_keys = [public_state_key(state) for state, _source in first]
    second_keys = [public_state_key(state) for state, _source in second]
    assert first_keys == second_keys == written["state_keys"]
    # Denial-search sims are not part of the frozen artifact and cannot change it.
    low = SearchConfig(root_search_sims=32)
    high = SearchConfig(root_search_sims=400)
    assert low.root_search_sims != high.root_search_sims
    assert [public_state_key(state) for state, _source in load_frozen_positions(path)] == first_keys


def test_k64_cells_require_explicit_permission_and_are_not_stability_cells():
    args = _parse_args([
        "--mode", "cell", "--cell", "chance_s128_k64_seed0",
    ])
    with pytest.raises(PermissionError, match="--allow-k64"):
        run_cell(args, CELL_SPECS[args.cell])
    assert all(CELL_SPECS[name].chance_k <= 16 for name in STABILITY_NAMES)
    assert CELL_SPECS["sims_s400_k32_seed0"].chance_k == 32


def test_persisted_reference_cell_reproduces_published_anchor_when_available():
    root = Path(__file__).resolve().parents[3]
    run_dir = root / "runs" / "kingdomino" / "denial_search"
    positions_path = run_dir / "signal_positions.jsonl"
    reference_path = run_dir / "signal_cells" / "reference_s32_k4_seed0.json"
    if not positions_path.exists() or not reference_path.exists():
        pytest.skip("local denial-signal artifacts are not present")

    first = load_frozen_positions(positions_path)
    second = load_frozen_positions(positions_path)
    assert len(first) == len(second) == 50
    first_keys = [public_state_key(state) for state, _source in first]
    assert first_keys == [public_state_key(state) for state, _source in second]

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    assert [row["state_key"] for row in reference["positions"]] == first_keys
    metrics = reference["metrics"]
    assert metrics["starved_picks_upweighted"] == 8
    assert metrics["high_fragility_starved_picks_upweighted"] == 0
    assert metrics["high_fragility_positions"] == 13
    assert metrics["fragility"]["median"] == pytest.approx(0.03666265920868941)


def test_chance_k_stderr_nonincreasing_and_exact_at_combination_cap(tmp_path):
    state = _round_start(seed=59)
    while len(state.deck) > 8:
        state = state.step(state.legal_actions()[0])
    while not (state.phase == Phase.PLACE_AND_SELECT
               and state.actor_index == 3 and len(state.deck) == 8):
        state = state.step(state.legal_actions()[0])
    assert len(state.current_row) == 1
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    stderrs = []
    modes = []
    for chance_k in (1, 4, 70):
        search = DenialSearch(
            _ConstantEvaluator(), checkpoint_path=str(checkpoint),
            config=SearchConfig(pick_plies=2, chance_k=chance_k,
                                root_search_sims=0, placement_top_k=1),
        )
        label = search.search_position(state)
        stderrs.append(label["per_pick"][0]["mc_standard_error"])
        modes.append(label["structure"]["chance_events"])
    assert stderrs[0] >= stderrs[1] >= stderrs[2]
    assert modes[0]["sampled"] > 0 and modes[1]["sampled"] > 0
    assert modes[2]["enumerated"] > 0 and modes[2]["sampled"] == 0


@pytest.mark.parametrize("values", [[], [0.0, float("nan")], [0.0, float("inf")]])
def test_policy_target_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        denial_policy_target(values)
