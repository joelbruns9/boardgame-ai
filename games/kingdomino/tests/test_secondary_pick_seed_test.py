from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from games.kingdomino.action_codec import encode_action
from games.kingdomino.denial_search import DenialSearch, SearchConfig, public_state_key
from games.kingdomino.denial_signal_sweep import load_frozen_positions, write_frozen_positions
from games.kingdomino.promotion import sha256_file
from games.kingdomino.secondary_pick_seed_test import (
    ROOT_SEEDS,
    SIMS,
    TREE_SEEDS,
    build_report,
    phase0_equivalence,
    root_q_by_pick,
    tie_guarded_flip,
)
from games.kingdomino.tests.test_denial_search import _ConstantEvaluator, _round_start


def _search(tmp_path, *, sims=11, chance_k=2):
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    evaluator = _ConstantEvaluator()
    search = DenialSearch(
        evaluator,
        checkpoint_path=str(checkpoint),
        config=SearchConfig(
            pick_plies=1,
            chance_k=chance_k,
            seed=7,
            placement_top_k=1,
            root_search_sims=sims,
        ),
    )
    return search


def _install_axis_sensitive_root(search, state):
    policy = search.evaluator.policy(state)

    def fake_compute(_state, seed):
        visits = {}
        info = {}
        for ordinal, action in enumerate(state.legal_actions(), 1):
            idx = int(encode_action(action, state))
            visits[idx] = float(ordinal)
            q = float(seed * 1e-6 + search.config.root_search_sims * 1e-4 + ordinal * 1e-3)
            info[idx] = (float(policy[idx]), q)
        return visits, 0.0, info

    search._root_search_compute = fake_compute


def test_phase0_rederived_root_q_is_byte_identical(tmp_path):
    state = _round_start(seed=101)
    search = _search(tmp_path)
    _install_axis_sensitive_root(search, state)

    gate = phase0_equivalence(search, state, seed=ROOT_SEEDS[0])

    assert gate["passed"]
    assert gate["comparisons"]
    assert all(row["byte_identical"] for row in gate["comparisons"])


def test_root_q_seed_and_sim_axes_change_without_changing_public_state(tmp_path):
    state = _round_start(seed=103)
    search = _search(tmp_path, sims=10)
    _install_axis_sensitive_root(search, state)
    key = public_state_key(state)

    seed0 = search._root_search(
        state, seed_override=ROOT_SEEDS[0], cache_namespace="seed0_s10")
    seed1 = search._root_search(
        state, seed_override=ROOT_SEEDS[1], cache_namespace="seed1_s10")
    search.config.root_search_sims = 20
    sims20 = search._root_search(
        state, seed_override=ROOT_SEEDS[0], cache_namespace="seed0_s20")

    q0 = root_q_by_pick(search, state, seed0)
    q1 = root_q_by_pick(search, state, seed1)
    q20 = root_q_by_pick(search, state, sims20)
    assert [row["root_q"] for row in q0.values()] != [row["root_q"] for row in q1.values()]
    assert [row["root_q"] for row in q0.values()] != [row["root_q"] for row in q20.values()]
    assert public_state_key(state) == key


def test_tie_guard_kills_equal_searched_value_flip():
    event = tie_guarded_flip(
        {10: 0.4, 20: 0.5},
        {10: 0.7, 20: 0.7 - 5e-7},
        tie_tolerance=1e-6,
    )
    assert event["root_top"] == 20
    assert event["search_best"] == 10
    assert event["would_be_flip"]
    assert event["tie_guard_killed"]
    assert not event["flip"]


def test_chance_k_does_not_change_root_search_output(tmp_path):
    state = _round_start(seed=107)
    search = _search(tmp_path, sims=13, chance_k=16)
    _install_axis_sensitive_root(search, state)
    first = search._root_search(
        state, seed_override=ROOT_SEEDS[0], cache_namespace="k16")
    search.config.chance_k = 32
    second = search._root_search(
        state, seed_override=ROOT_SEEDS[0], cache_namespace="k32")
    assert first == second


def test_frozen_fifty_is_independent_of_tree_seed_and_root_sims_when_available():
    root = Path(__file__).resolve().parents[3]
    path = root / "runs" / "kingdomino" / "denial_search" / "signal_positions.jsonl"
    if not path.exists():
        return
    baseline = [public_state_key(state) for state, _source in load_frozen_positions(path)]
    assert len(baseline) == 50
    for seed, sims in ((TREE_SEEDS[0], 800), (TREE_SEEDS[-1], 3200),
                       (ROOT_SEEDS[-1], 10_000)):
        config = SearchConfig(seed=seed, root_search_sims=sims)
        assert config.seed == seed and config.root_search_sims == sims
        observed = [public_state_key(state) for state, _source in load_frozen_positions(path)]
        assert observed == baseline


def test_report_builds_from_complete_resumable_artifacts(tmp_path):
    state = _round_start(seed=109)
    positions_path = tmp_path / "positions.jsonl"
    frozen = write_frozen_positions([(state, {"fixture": True})], positions_path)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = sha256_file(checkpoint)
    run_dir = tmp_path / "secondary_seed"
    run_dir.mkdir()
    picks = list(state.current_row)
    searched = {pick: 0.8 - 0.1 * ordinal for ordinal, pick in enumerate(picks)}

    for seed_index, seed in enumerate(TREE_SEEDS):
        row = {
            "position_index": 0,
            "state_key": public_state_key(state),
            "positions_sha256": frozen["sha256"],
            "checkpoint_sha256": checkpoint_sha,
            "elapsed_seconds": 1.0,
            "per_pick": [
                {"pick_domino_id": pick,
                 "searched_value_actor": value + (seed_index - 1) * 0.001}
                for pick, value in searched.items()
            ],
        }
        (run_dir / f"tree_seed{seed}.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8")

    ladder_lines = []
    fragility = {800: 0.12, 3200: 0.06, 10_000: 0.01}
    for sims in SIMS:
        for seed_index, seed in enumerate(ROOT_SEEDS):
            ladder_lines.append(json.dumps({
                "position_index": 0,
                "sims": sims,
                "seed": seed,
                "positions_sha256": frozen["sha256"],
                "checkpoint_sha256": checkpoint_sha,
                "elapsed_seconds": 0.1,
                "per_pick": [
                    {"pick_domino_id": pick,
                     "root_q": value + fragility[sims] + (seed_index - 2) * 0.001}
                    for pick, value in searched.items()
                ],
            }))
    (run_dir / "root_ladder.jsonl").write_text(
        "\n".join(ladder_lines) + "\n", encoding="utf-8")
    tie_row = {
        "position_index": 0,
        "positions_sha256": frozen["sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "elapsed_seconds": 0.2,
        "tie_pairs_k16": [],
        "tie_pairs_k32": [],
        "dissolved_ties": [],
        "new_ties": [],
    }
    (run_dir / "tie_probe.jsonl").write_text(
        json.dumps(tie_row) + "\n", encoding="utf-8")
    (run_dir / "phase0_gate.json").write_text(json.dumps({
        "passed": True,
        "positions_sha256": frozen["sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "elapsed_seconds": 0.3,
    }), encoding="utf-8")
    output = tmp_path / "report.json"
    args = Namespace(
        checkpoint=str(checkpoint), positions_path=str(positions_path),
        run_dir=str(run_dir), output=str(output), sims=SIMS,
    )

    report = build_report(args)

    assert output.exists()
    assert report["secondary_pick_count"] == len(picks) - 1
    assert report["rank1_pick_count"] == 1
    rank_conditioned = report["rank_conditioned_fragility"]
    for sims in SIMS:
        key = str(sims)
        assert rank_conditioned["rank1"][key]["median"] == pytest.approx(fragility[sims])
        assert rank_conditioned["secondary"][key]["median"] == pytest.approx(
            fragility[sims])
        assert (
            rank_conditioned["secondary_specificity"][key]
            ["secondary_minus_rank1_median_fragility"]
            == pytest.approx(0.0)
        )
        rank1_composition = report["root_q_observation_composition"]["rank1"][key]
        assert rank1_composition["expected_pick_seed_cells"] == len(ROOT_SEEDS)
        assert rank1_composition["observed_pick_seed_cells"] == len(ROOT_SEEDS)
        assert rank1_composition["missing_pick_seed_cells"] == 0
    assert report["routing"]["classification"] == "NOISE"
    assert len(report["searched_reference_stability"]["per_pick"]) == len(picks)
