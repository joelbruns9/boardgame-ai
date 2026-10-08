"""G12 seeding (owner 2026-10-08): run07's Mausoleum science setups preloaded
into the restart archive, spent at a capped pace, never aged out."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import random

import pytest

swr = pytest.importorskip("seven_wonders_rust")

from . import restart_archive as ra
from .codec import decode_action
from .engine import apply_action
from .game import new_game
from .mausoleum_seeds import is_setup

SEEDS = Path(__file__).parent / "seeds" / "mausoleum_run07.json"


def _entry(k: int, seeded: bool) -> ra.Entry:
    return ra.Entry(seed=k, first_player=0, prefix=[(k, 0, "h")], chance_prefix=[],
                    tried=[0], born=0, source=[0, 1, 0], legal_count=5, seeded=seeded)


def test_draw_spends_a_fixed_number_of_seeded_entries_per_iteration():
    archive = ra.Archive()
    archive.add(_entry(k, True) for k in range(50))
    archive.add(_entry(100 + k, False) for k in range(50))
    drawn = archive.draw(20, random.Random(0), seeded_cap=3)
    assert len(drawn) == 20 and sum(e.seeded for e in drawn) == 3
    # Early in a run the archive is nearly all seeds: still only the cap.
    early = ra.Archive()
    early.add(_entry(k, True) for k in range(50))
    assert len(early.draw(20, random.Random(0), seeded_cap=3)) == 3
    # No cap: the old uniform draw.
    assert len(early.draw(20, random.Random(0))) == 20


def test_seeded_entries_do_not_age_out_but_still_retire():
    archive = ra.Archive(max_restarts=2, max_age=3)
    archive.add([_entry(1, True), _entry(2, False)])
    archive.prune(iteration=50)
    assert [e.seeded for e in archive.entries.values()] == [True]
    archive.draw(1, random.Random(0), seeded_cap=1)
    archive.draw(1, random.Random(0), seeded_cap=1)
    archive.prune(iteration=51)
    assert archive.entries == {}


def test_the_committed_seeds_rebuild_on_the_engine_before_a_setup():
    archive = ra.Archive.load(SEEDS)
    assert archive.seed_source and len(archive.entries) >= 600
    entries = sorted(archive.entries.values(), key=lambda e: e.key)
    assert all(e.seeded and e.tried and e.legal_count >= 2 for e in entries)
    for entry in random.Random(4).sample(entries, 12):
        _iteration, setup, back = entry.source
        assert 0 <= back <= ra.DEFAULT_WINDOW and entry.ply == setup - back
        game = ra.restart_game(entry)
        assert len(game.legal_action_indices()) == entry.legal_count
        state = new_game(entry.seed, first_player=entry.first_player)
        for action, _actor, _hash in entry.prefix:
            apply_action(state, decode_action(state, action))
        assert is_setup(state) == (back == 0)


def test_phase_d_preloads_the_seeds_once(tmp_path):
    from .phase_d import PhaseDConfig, PhaseDLoop

    seeds = ra.Archive(seed_source="unit.json")
    seeds.add([_entry(k, False) for k in range(4)])  # flagged seeded on load
    path = tmp_path / "unit.json"
    seeds.save(path)
    config = PhaseDConfig(
        run_dir=str(tmp_path / "run"), workers=1, games_per_iteration=8,
        seed_games=0, opponent_fraction=0.0, d_model=32, layers=1, device="cpu",
        restart_fraction=0.5, restart_seed_archive=str(path),
    )
    loop = PhaseDLoop(config)
    archive = loop._restart_archive()
    assert len(archive.entries) == 4 and all(e.seeded for e in archive.entries.values())
    assert loop.restart_archive_path.exists()
    # A resume reads the run's archive and does not seed again.
    for entry in archive.entries.values():
        entry.restarts = 1
    archive.save(loop.restart_archive_path)
    again = PhaseDLoop(config)._restart_archive()
    assert all(e.restarts == 1 for e in again.entries.values())
    with pytest.raises(ValueError, match="restart_fraction"):
        dataclasses.replace(config, restart_fraction=0.0).validate()


def test_phase_d_restarts_seeded_mausoleum_positions(tmp_path):
    """Iteration 0, archive = the committed seeds only: exactly the cap is
    restarted, and every merged record replays from its run07 deal."""

    from .buffer import read_records, replay
    from .phase_d import PhaseDConfig, PhaseDLoop

    config = PhaseDConfig(
        run_dir=str(tmp_path / "run"), workers=1, games_per_iteration=8,
        seed_games=0, opponent_fraction=0.0, d_model=32, layers=1,
        cheap_sims_min=2, cheap_sims_max=2, full_sims_min=3, full_sims_max=3,
        full_search_fraction=1.0, top_k=2, device="cpu",
        restart_fraction=0.5, restart_harvest_games=8,
        restart_seed_archive=str(SEEDS), restart_seed_per_iteration=2,
    )
    loop = PhaseDLoop(config)
    loop.initialize()
    model = loop.load_model(loop.current_best)
    records = loop.generate_iteration(model, 0)
    stats = loop.last_generation_stats["restarts"]
    assert stats["games"] == stats["seeded"] == 2
    assert stats["seeded_left"] >= 600
    restarted = [r for r in read_records(loop.buffer_dir / "iter_0000.jsonl")
                 if r.restart_from is not None]
    assert len(restarted) == 2
    for record in restarted:
        replay(record)
        assert record.restart_from > 0
