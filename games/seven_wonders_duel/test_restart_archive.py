"""G12: restart archive -- harvest, restart, forced branch, merged records,
derivation without the realised result."""

from __future__ import annotations

import random

import numpy as np
import pytest

swr = pytest.importorskip("seven_wonders_rust")
torch = pytest.importorskip("torch")

from . import restart_archive as ra
from .buffer import from_json_line, replay, to_json_line
from .codec import legal_action_indices
from .dataset import collate, derive_records_rust, examples_from_record
from .inference import Evaluator
from .rust_bridge import (
    phase_d_records_from_rust,
    rust_flat_batch_adapter,
    rust_games_for_self_play,
)
from .train import build_model, compute_losses, value_targets

KWARGS = dict(
    global_batch_cap=32, leaf_batch=1, cheap_sims_min=2, cheap_sims_max=3,
    full_sims_min=4, full_sims_max=6, full_search_fraction=0.5, top_k=4,
    draft_prior=0.0, iteration=1, max_active_slots=8,
)


@pytest.fixture(scope="module")
def evaluator():
    torch.manual_seed(1212)
    return Evaluator(build_model("transformer", 32, 1), "cpu", 64)


def _play(evaluator, games, seeds, **extra):
    raw, _metrics = swr.self_play_many_flat_net(
        adapter=rust_flat_batch_adapter(evaluator), games=games, game_seeds=seeds,
        **KWARGS, **extra,
    )
    return phase_d_records_from_rust(raw, validate=False)


@pytest.fixture(scope="module")
def ancestors(evaluator):
    seeds = [7101 + k for k in range(8)]
    return _play(evaluator, rust_games_for_self_play(seeds, [k % 2 for k in range(8)]), seeds)


@pytest.fixture(scope="module")
def entries(ancestors):
    found = ra.harvest(ancestors, iteration=1, seed=3)
    assert found, "no decisive position in the fixture games"
    return found


def test_harvest_points_before_a_decisive_position(ancestors, entries):
    by_seed = {r.seed: r for r in ancestors}
    for entry in entries:
        record = by_seed[entry.seed]
        _iteration, decisive, back = entry.source
        assert 0 <= back <= ra.DEFAULT_WINDOW and entry.ply == decisive - back
        assert entry.tried == [record.moves[entry.ply].action]
        assert entry.legal_count >= 2
        assert [a for a, _, _ in entry.prefix] == [m.action for m in record.moves[: entry.ply]]
    per_game = {}
    for entry in entries:
        per_game[entry.seed] = per_game.get(entry.seed, 0) + 1
    assert max(per_game.values()) <= ra.DEFAULT_PER_GAME


@pytest.fixture(scope="module")
def restarted(evaluator, entries):
    chosen = entries[:4]
    games = [ra.restart_game(entry) for entry in chosen]
    for entry, game in zip(chosen, games):
        # The rebuilt position IS the ancestor's: same legal set, same mover.
        actor, _action, mask_hash = entry.prefix[-1] if entry.prefix else (None, None, None)
        assert len(game.legal_action_indices()) == entry.legal_count
    seeds = [90_000 + k for k in range(len(chosen))]
    continuations = _play(
        evaluator, games, seeds, first_move_excludes=[list(e.tried) for e in chosen]
    )
    return chosen, continuations


def test_a_restart_plays_a_branch_history_did_not(restarted):
    chosen, continuations = restarted
    for entry, record in zip(chosen, continuations):
        assert record.moves[0].action not in entry.tried
        # The training target is the search's own, untouched by the exclusion.
        assert sum(record.moves[0].policy_target.values()) == pytest.approx(1.0)


def test_a_merged_restart_record_replays_from_its_seed_and_round_trips(restarted):
    chosen, continuations = restarted
    for entry, continuation in zip(chosen, continuations):
        merged = ra.merge_record(entry, continuation, iteration=2)
        replay(merged)  # masks, actors, chance log and both digests
        assert merged.restart_from == entry.ply
        assert merged.moves[entry.ply].action == continuation.moves[0].action
        again = from_json_line(to_json_line(merged))
        assert again.restart_from == entry.ply and again.moves == merged.moves


def test_restart_rows_start_at_the_restart_and_carry_no_outcome(restarted):
    chosen, continuations = restarted
    merged = [ra.merge_record(e, c, iteration=2) for e, c in zip(chosen, continuations)]
    rust = derive_records_rust(merged, tactic_labels=True, batch_games=4)
    for record, (rust_rows, _stats) in zip(merged, rust):
        python_rows = examples_from_record(record, tactic_labels=True)
        assert [e.move_index for e in python_rows] == [e.move_index for e in rust_rows]
        assert rust_rows and min(e.move_index for e in rust_rows) >= record.restart_from
        assert all(e.outcome_free for e in rust_rows)
        for a, b in zip(python_rows, rust_rows):
            assert np.allclose(a.policy_target, b.policy_target)


def test_outcome_free_rows_train_on_the_search_value_and_skip_end_of_game_heads(
    restarted, ancestors
):
    chosen, continuations = restarted
    merged = ra.merge_record(chosen[0], continuations[0], iteration=2)
    rows = examples_from_record(merged) + examples_from_record(ancestors[0])[:8]
    batch = collate(rows)
    assert "outcome_free" in batch and bool(batch["outcome_free"].any())
    targets = value_targets(batch, value_bootstrap=0.5)
    free = batch["outcome_free"] & batch["value_soft_valid"]
    # An ordinary row's target blends the outcome with the search value; a
    # restart row's outcome IS the search value, so the blend is the search
    # value alone (proofs and certain wins still override).
    proven = batch["value_solver_valid"] | batch["value_certain"]
    rows_checked = free & ~proven
    assert bool(rows_checked.any())
    assert torch.allclose(targets["flat"][rows_checked], batch["value_soft"][rows_checked])
    ordinary = ~batch["outcome_free"] & batch["value_soft_valid"] & ~proven
    hard = torch.nn.functional.one_hot(batch["value_class"].long(), 3).float()
    assert torch.allclose(
        targets["flat"][ordinary],
        0.5 * hard[ordinary] + 0.5 * batch["value_soft"][ordinary],
    )
    torch.manual_seed(0)
    model = build_model("transformer", 32, 1)
    total, parts = compute_losses(model(batch), batch, value_bootstrap=0.5)
    assert torch.isfinite(total)
    # A plain batch keeps its historical key set.
    assert "outcome_free" not in collate(examples_from_record(ancestors[0])[:8])


def test_the_archive_charges_restarts_prunes_and_persists(entries, tmp_path):
    archive = ra.Archive(max_restarts=2, max_age=3)
    assert archive.add(entries) == len({e.key for e in entries})
    assert archive.add(entries) == 0
    rng = random.Random(0)
    first = archive.draw(3, rng)
    assert len({e.key for e in first}) == len(first)
    archive.note_played(first[0], 999)
    assert 999 in archive.entries[first[0].key].tried
    for _ in range(3):
        archive.draw(len(archive.entries), rng)
    removed = archive.prune(iteration=1)
    assert removed and all(e.restarts < 2 for e in archive.entries.values())
    path = tmp_path / "archive.json"
    archive.save(path)
    loaded = ra.Archive.load(path)
    assert loaded.entries.keys() == archive.entries.keys()
    assert ra.Archive.load(tmp_path / "missing.json").entries == {}
    archive.prune(iteration=10)
    assert archive.entries == {}


def test_phase_d_restarts_games_from_its_archive(tmp_path):
    """Iteration 0 fills the archive; iteration 1 restarts a share of its games
    from it, writes replayable merged records, and charges the archive."""

    from .buffer import read_records
    from .phase_d import PhaseDConfig, PhaseDLoop

    config = PhaseDConfig(
        run_dir=str(tmp_path / "run"), workers=1, games_per_iteration=8,
        seed_games=0, opponent_fraction=0.0, d_model=32, layers=1,
        cheap_sims_min=2, cheap_sims_max=2, full_sims_min=3, full_sims_max=3,
        full_search_fraction=1.0, top_k=2, device="cpu",
        restart_fraction=0.5, restart_harvest_games=8,
    )
    loop = PhaseDLoop(config)
    loop.initialize()
    model = loop.load_model(loop.current_best)
    first = loop.generate_iteration(model, 0)
    assert all(r.restart_from is None for r in first)
    archived = loop.last_generation_stats["restarts"]["archive"]
    assert archived > 0 and loop.restart_archive_path.exists()
    second = loop.generate_iteration(model, 1)
    restarted = [r for r in second if r.restart_from is not None]
    stats = loop.last_generation_stats["restarts"]
    assert len(restarted) == stats["games"] == min(4, archived)
    written = read_records(loop.buffer_dir / "iter_0001.jsonl")
    assert sum(r.restart_from is not None for r in written) == len(restarted)
    for record in written:
        replay(record)
    for record in restarted:
        assert record.iteration == 1 and "restart_of" in record.agents


# --- review of 50e02c7: one regression per finding ---------------------------


def _merged(restarted):
    chosen, continuations = restarted
    return [ra.merge_record(e, c, iteration=2) for e, c in zip(chosen, continuations)]


def _restart_batch(restarted, rust: bool):
    merged = _merged(restarted)
    if rust:
        rows = [row for rows, _ in derive_records_rust(merged, batch_games=4) for row in rows]
    else:
        rows = [row for record in merged for row in examples_from_record(record)]
    assert rows and all(row.outcome_free for row in rows)
    return collate(rows)


RECIPE = dict(value_bootstrap=0.5, short_term_value_weight=0.25, outlook_bootstrap=0.5,
              hierarchical=True)


@pytest.mark.parametrize("rust", [False, True])
def test_finding_1_the_terminal_result_cannot_reach_restart_targets(restarted, rust):
    """Changing only the realised result -- and the short-term return anchored
    on it -- leaves every restart target unchanged, flat and W4."""

    batch = _restart_batch(restarted, rust)
    before = value_targets(batch, **RECIPE)
    flipped = dict(batch)
    flipped["value_class"] = 2 - batch["value_class"]
    flipped["joint7"] = (batch["joint7"] + 3) % 6
    flipped["value_short"] = batch["value_short"].flip(dims=[1])
    after = value_targets(flipped, **RECIPE)
    for key in ("flat", "hier_outcome", "hier_type"):
        assert torch.allclose(before[key], after[key]), key


def _one_row(restarted, **overrides):
    batch = _restart_batch(restarted, rust=False)
    one = {k: (v[:1].clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    one.update(overrides)
    return one


def test_finding_2_a_certain_win_keeps_its_exact_labels(restarted):
    soft = torch.tensor([[0.6, 0.0, 0.4]])
    one = _one_row(
        restarted,
        value_class=torch.tensor([0]),
        joint7=torch.tensor([1]),  # my_military
        value_soft=soft, value_soft_valid=torch.tensor([True]),
        value_certain=torch.tensor([True]),
        value_solver=torch.tensor([[1.0, 0.0, 0.0]]),
        value_solver_valid=torch.tensor([True]),
        value_solver_exact=torch.tensor([True]),
    )
    targets = value_targets(one, **RECIPE)
    assert torch.allclose(targets["flat"], torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(targets["hier_outcome"], torch.tensor([[1.0, 0.0, 0.0]]))
    expected_type = torch.zeros(1, 7)
    expected_type[0, 1] = 1.0
    assert torch.allclose(targets["hier_type"], expected_type)


def test_finding_3_no_outlook_means_no_realised_type_and_soft_outcome(restarted):
    soft = torch.tensor([[0.6, 0.0, 0.4]])
    base = dict(
        value_soft=soft, value_soft_valid=torch.tensor([True]),
        outlook_soft_valid=torch.tensor([False]),
        value_certain=torch.tensor([False]),
        value_solver_valid=torch.tensor([False]),
    )
    targets = value_targets(_one_row(restarted, **base), **RECIPE)
    assert torch.allclose(targets["hier_outcome"], soft)
    assert float(targets["hier_type"].abs().sum()) == 0.0
    assert bool(targets["hier_rows"][0]) and bool(targets["value_rows"][0])
    # Nothing permitted at all: the row is not supervised, rather than invented.
    bare = dict(base, value_soft_valid=torch.tensor([False]))
    targets = value_targets(_one_row(restarted, **bare), **RECIPE)
    assert not bool(targets["value_rows"][0]) and not bool(targets["hier_rows"][0])


def test_finding_4_zero_bootstrap_still_builds_the_restart_target(restarted, ancestors):
    batch = _restart_batch(restarted, rust=False)
    targets = value_targets(batch, value_bootstrap=0.0, short_term_value_weight=0.0)
    assert targets["flat"] is not None
    free = batch["value_soft_valid"] & ~batch["value_solver_valid"] & ~batch["value_certain"]
    assert torch.allclose(targets["flat"][free], batch["value_soft"][free])
    # The loss reads that target, so changing only the realised label changes nothing.
    torch.manual_seed(0)
    model = build_model("transformer", 32, 1)
    outputs = model(batch)
    flipped = dict(batch, value_class=2 - batch["value_class"])
    _, a = compute_losses(outputs, batch)
    _, b = compute_losses(outputs, flipped)
    assert a["value"] == pytest.approx(b["value"])
    # Mixed with ordinary rows, the restart rows keep the same targets.
    mixed_rows = examples_from_record(ancestors[0])[:8] + [
        row for record in _merged(restarted) for row in examples_from_record(record)
    ]
    mixed = collate(mixed_rows)
    mixed_targets = value_targets(mixed, value_bootstrap=0.0, short_term_value_weight=0.0)
    rows = mixed["outcome_free"] & mixed["value_soft_valid"] & ~mixed["value_solver_valid"] & ~mixed["value_certain"]
    assert torch.allclose(mixed_targets["flat"][rows], mixed["value_soft"][rows])


def test_finding_5_restarts_of_one_ancestor_have_distinct_identities(evaluator, entries):
    by_seed: dict = {}
    for entry in entries:
        by_seed.setdefault(entry.seed, []).append(entry)
    pair = next((group[:2] for group in by_seed.values() if len(group) >= 2), None)
    if pair is None:
        pair = [entries[0], entries[0]]
    games = [ra.restart_game(entry) for entry in pair]
    seeds = [80_001, 80_002]
    continuations = _play(evaluator, games, seeds,
                          first_move_excludes=[list(e.tried) for e in pair])
    merged = [ra.merge_record(e, c, iteration=2) for e, c in zip(pair, continuations)]
    assert len({(r.iteration, r.seed) for r in merged}) == 2
    for record, entry in zip(merged, pair):
        assert record.deal_seed == entry.seed and record.replay_seed == entry.seed
        replay(record)
        again = from_json_line(to_json_line(record))
        assert (again.seed, again.deal_seed, again.family) == (
            record.seed, record.deal_seed, record.family
        )


def test_finding_6_a_family_shares_one_holdout_side(restarted, ancestors):
    from .train import stable_game_split

    merged = _merged(restarted)
    by_seed = {r.seed: r for r in ancestors}
    for salt in ("a", "b", "c", "swd-v1"):
        for record in merged:
            ancestor = by_seed[record.deal_seed]
            assert record.family == (ancestor.iteration, ancestor.seed)
            rows = examples_from_record(ancestor) + examples_from_record(record)
            train, val = stable_game_split(rows, 0.5, salt)
            assert not train or not val, "family split across the holdout"
    # A restart of a restart stays in the ROOT family.
    second = ra.harvest(merged, iteration=2, seed=5)
    for entry in second:
        assert tuple(entry.family) in {r.family for r in merged}
