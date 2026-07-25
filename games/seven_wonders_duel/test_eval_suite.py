"""Eval-suite bookkeeping: fingerprinted resume and the paired summary.

Arena games themselves need CUDA and the Rust engine, so these cover the
bookkeeping that made run 02's ``postrun_eval/summary.json`` hard to read: a
label-keyed resume that could silently mix settings, and paired statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from games.seven_wonders_duel.eval_suite import (
    ArenaSettings,
    EvalSuite,
    build_jobs,
    fingerprint,
    summarize,
)


def _outcome(seed: int, winner: int) -> SimpleNamespace:
    return SimpleNamespace(
        seed=seed,
        first_player=0,
        winner=winner,
        victory_type="civilian",
        actions=70,
        score_for=lambda seat, winner=winner: 1.0 if seat == winner else 0.0,
    )


def test_fingerprint_changes_with_sims_and_checkpoint_identity():
    base = ArenaSettings(sims=64)
    args = {"games": 100, "offset": 1, "participants": [("model", "aaaa")]}
    same = fingerprint(base, **args)
    assert fingerprint(base, **args) == same
    assert fingerprint(ArenaSettings(sims=128), **args) != same
    assert fingerprint(ArenaSettings(sims=64, seed=7), **args) != same
    assert (
        fingerprint(base, games=200, offset=1, participants=[("model", "aaaa")])
        != same
    )
    assert (
        fingerprint(base, games=100, offset=1, participants=[("model", "bbbb")])
        != same
    )


def test_resume_reruns_a_match_whose_settings_changed(tmp_path: Path):
    """Run 02's runner skipped on label alone.

    Two ``*_vs_random`` rows survived from an earlier invocation and sat in the
    same summary as freshly generated rows, with nothing recording that they
    had been produced under different settings.
    """

    settings = ArenaSettings(sims=64, work_dir=str(tmp_path))
    suite = EvalSuite(tmp_path, settings)
    suite.summary_path.write_text(
        json.dumps(
            {
                "matches": [
                    {
                        "match": "a_vs_b",
                        "score_rate": 0.5,
                        "fingerprint": "stale-fingerprint",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    ran: list[str] = []

    def fake_model_match(_settings, label, _cand, _opp, games, _offset):
        ran.append(label)
        return (
            {
                "match": label,
                "games": games,
                "wins": games,
                "losses": 0,
                "draws": 0,
                "score_rate": 1.0,
                "paired_95_ci": [1.0, 1.0],
                "games_per_second": 1.0,
            },
            [],
        )

    import games.seven_wonders_duel.eval_suite as module

    original = module.run_model_match
    module.run_model_match = fake_model_match
    try:
        jobs = [
            {
                "kind": "model",
                "label": "a_vs_b",
                "candidate": ("a", tmp_path / "a.pt"),
                "opponent": ("b", tmp_path / "b.pt"),
                "games": 10,
                "offset": 0,
                "fingerprint": "current-fingerprint",
            }
        ]
        matches = suite.run(jobs, log=lambda *_: None)
    finally:
        module.run_model_match = original

    assert ran == ["a_vs_b"], "changed settings must invalidate the cached row"
    rows = [row for row in matches if row["match"] == "a_vs_b"]
    assert len(rows) == 1, "the stale row must be replaced, not duplicated"
    assert rows[0]["fingerprint"] == "current-fingerprint"
    assert rows[0]["score_rate"] == 1.0


def test_resume_skips_a_match_whose_fingerprint_still_matches(tmp_path: Path):
    settings = ArenaSettings(work_dir=str(tmp_path))
    suite = EvalSuite(tmp_path, settings)
    suite.summary_path.write_text(
        json.dumps(
            {"matches": [{"match": "a_vs_b", "score_rate": 0.5, "fingerprint": "keep"}]}
        ),
        encoding="utf-8",
    )
    jobs = [
        {
            "kind": "model",
            "label": "a_vs_b",
            "candidate": ("a", tmp_path / "a.pt"),
            "opponent": ("b", tmp_path / "b.pt"),
            "games": 10,
            "offset": 0,
            "fingerprint": "keep",
        }
    ]
    matches = suite.run(jobs, log=lambda *_: None)
    assert [row["score_rate"] for row in matches] == [0.5]


def test_summarize_pairs_seats_and_reports_a_paired_interval():
    # Four games = two seed-pairs; the candidate alternates seats by index.
    # Seat 0 always wins, so the candidate wins its seat-0 leg and loses its
    # seat-1 leg in each pair -- a pure first-player effect, not strength.
    outcomes = [_outcome(1, 0), _outcome(1, 0), _outcome(2, 0), _outcome(2, 0)]
    summary, rows = summarize("m", "cand", "opp", outcomes, elapsed=2.0)
    assert summary["games"] == 4
    assert summary["pairs"] == 2
    # Candidate wins as seat 0 in both pairs and loses as seat 1 in both.
    assert summary["candidate_as_seat0"] == 1.0
    assert summary["candidate_as_seat1"] == 0.0
    assert summary["score_rate"] == 0.5
    # Every pair scored exactly 0.5, so the paired interval has no width -- the
    # per-game interval would have implied spurious spread.
    assert summary["paired_95_ci"] == [0.5, 0.5]
    assert [row["candidate_seat"] for row in rows] == [0, 1, 0, 1]


def test_build_jobs_covers_the_round_robin_and_the_bot_anchors(tmp_path: Path):
    paths = {}
    for name in ("m0", "m1", "m2"):
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode())
        paths[name] = path
    settings = ArenaSettings(work_dir=str(tmp_path))
    jobs = build_jobs(
        settings, paths, model_games=8, bot_games=4, bots=["greedy"]
    )
    labels = [job["label"] for job in jobs]
    assert labels[:3] == ["m0_vs_m1", "m0_vs_m2", "m1_vs_m2"]
    assert labels[3:] == ["m0_vs_greedy", "m1_vs_greedy", "m2_vs_greedy"]
    assert len({job["offset"] for job in jobs}) == len(jobs)
    assert len({job["fingerprint"] for job in jobs}) == len(jobs)


def test_identical_checkpoint_bytes_under_different_labels_share_identity(
    tmp_path: Path,
):
    """Run 02's ``current_best.pt`` and ``candidate_0000.pt`` were byte-identical.

    Fingerprints key on content, so relabelling the same weights cannot make a
    cached result look like a fresh measurement of a different model.
    """

    (tmp_path / "a.pt").write_bytes(b"same-weights")
    (tmp_path / "b.pt").write_bytes(b"same-weights")
    settings = ArenaSettings(work_dir=str(tmp_path))
    jobs = build_jobs(
        settings,
        {"a": tmp_path / "a.pt", "b": tmp_path / "b.pt"},
        model_games=8,
        bot_games=4,
        bots=["greedy"],
    )
    bot_jobs = [job for job in jobs if job["kind"] == "bot"]
    assert bot_jobs[0]["fingerprint"] != bot_jobs[1]["fingerprint"], (
        "different seed offsets keep the two rows distinct"
    )
