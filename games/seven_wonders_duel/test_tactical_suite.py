"""G0 tactical suite: exact labels, harvesting, splits and scoring."""

from __future__ import annotations

from collections import Counter
import json
import random

import pytest

swr = pytest.importorskip("seven_wonders_rust")
torch = pytest.importorskip("torch")

from . import phase_e as pe
from . import tactical_suite as ts
from . import tactics as tc
from .buffer import append_records, replay
from .codec import legal_action_indices
from .game import Phase
from .rust_bridge import rust_game_from_state
from .search import state_actor


@pytest.fixture(scope="module")
def records():
    # Rush bots drive both win conditions to the edge, so every tactical
    # class turns up in a handful of games.
    return pe.fresh_bot_records(30, seed=777)


@pytest.fixture(scope="module")
def cases(records):
    rng = random.Random(0)
    out = []
    for record in records:
        out.extend(ts.classify_record(record, "mem", rng, ordinary_rate=0.05, per_game_cap=3))
    return out


def test_rust_action_labels_agree_with_the_python_reference(records):
    compared = Counter()
    for record in records[:12]:
        states = []
        replay(record, on_state=lambda game, _move: states.append(game.clone()))
        for state in states:
            if state.phase is not Phase.PLAY_AGE:
                continue
            expected = tc.classify_actions(state)
            assert list(rust_game_from_state(state).classify_actions()) == expected
            compared["positions"] += 1
            compared["wins"] += expected.count(1)
            compared["losses"] += expected.count(-1)
    assert compared["wins"] >= 5 and compared["losses"] >= 5, compared


def test_rust_losing_mass_agrees_with_the_python_reference(records):
    compared = Counter()
    for record in records[:8]:
        states = []
        replay(record, on_state=lambda game, _move: states.append(game.clone()))
        for state in states:
            if state.phase is not Phase.PLAY_AGE or state.pending_choice is not None:
                continue
            expected = tc.losing_mass(state)
            got = rust_game_from_state(state).losing_mass()
            assert len(got) == len(expected)
            for g, e in zip(got, expected):
                assert (g is None) == (e is None)
                if e is not None:
                    assert g[1] == e[1] and g[0] == pytest.approx(e[0], abs=1e-12)
                    compared["partial"] += 0.0 < e[0] < 1.0
            compared["positions"] += 1
    assert compared["positions"] >= 100, compared


def test_labels_are_what_the_class_claims(cases, records):
    by_game = {(r.iteration, r.seed): r for r in records}
    found = Counter(case.cls for case in cases)
    for cls in ("own_win", "must_block", "quiet", "ordinary"):
        assert found[cls] >= 1, found
    for case in cases:
        if case.cls == "own_win":
            assert case.winning and case.value == 1.0 and case.value_exact
        elif case.cls == "forced_loss":
            assert case.value == -1.0 and case.value_exact
        elif case.cls == "must_block":
            assert case.losing and not case.winning
        elif case.cls == "predecessor":
            assert case.leads_to is not None and case.leads_to > case.move
        if not case.value_exact:
            record = by_game[(case.iteration, case.seed)]
            assert case.value in (-1.0, 0.0, 1.0)
            assert (record.winner is None) == (case.value == 0.0)
    assert len({case.id for case in cases}) == len(cases)


def test_sealing_is_by_whole_game_and_deterministic(cases):
    by_game = {}
    for case in cases:
        key = (case.iteration, case.seed)
        assert by_game.setdefault(key, case.split) == case.split
        assert case.split == ("sealed" if ts.sealed(*key) else "dev")
    splits = Counter(ts.sealed(i, s) for i in range(5) for s in range(400))
    assert 0.1 < splits[True] / sum(splits.values()) < 0.3


def test_harvest_round_trips_and_replays_the_same_states(tmp_path, records):
    buffer = tmp_path / "iter_0001.jsonl"
    append_records(buffer, records[:10])
    out = tmp_path / "cases.jsonl"
    summary = ts.harvest([buffer], out, ordinary_rate=0.05, log=lambda *_: None)
    assert summary["games"] == 10
    every = ts.read_cases(out, split=None)
    assert every and len(ts.read_cases(out, "dev")) + len(ts.read_cases(out, "sealed")) == len(every)
    states = ts.load_states(every)
    for case in every:
        state = states[case.id]
        labels = tc.classify_actions(state)
        legal = legal_action_indices(state)
        if case.cls == "own_win":
            assert set(case.winning) == {a for a, l in zip(legal, labels) if l == 1}


def test_scoring_rewards_the_right_reads(cases, records):
    picked = [c for c in cases if c.cls in ("own_win", "must_block")][:20]
    assert picked
    good, bad = [], []
    for case in picked:
        safe = next(
            (a for a in case.winning),
            None,
        )
        wrong = case.losing[0] if case.losing else None
        mass = {a: 0.0 for a in case.winning + case.losing}
        if safe is not None:
            good.append(ts.Reading(value=case.value, action=safe, mass={**mass, safe: 1.0}))
        else:
            good.append(ts.Reading(value=case.value, action=-1, mass=mass))
        bad.append(ts.Reading(
            value=-case.value if case.value else 0.9,
            action=wrong if wrong is not None else -1,
            mass={**mass, **({wrong: 1.0} if wrong is not None else {})},
        ))
    perfect = ts.score(picked, good)
    worst = ts.score(picked, bad)
    if "own_win" in perfect:
        assert perfect["own_win"]["found_win"] == 1.0
        assert perfect["own_win"]["value_mae"] == 0.0
        assert worst["own_win"]["found_win"] == 0.0
        assert worst["own_win"]["overconfident_wrong"] == 1.0
    if "must_block" in worst:
        assert worst["must_block"]["blunder"] == 1.0
        assert perfect["must_block"]["blunder"] == 0.0


def test_the_network_and_search_readers_run_end_to_end(cases, records):
    from .inference import Evaluator
    from .train import build_model

    picked = [c for c in cases if c.cls in ("own_win", "must_block", "ordinary")][:12]
    by_game = {(r.iteration, r.seed): r for r in records}
    states = []
    for case in picked:
        grabbed = {}
        replay(
            by_game[(case.iteration, case.seed)],
            on_state=lambda game, move, grabbed=grabbed, case=case: grabbed.setdefault(
                "s", game.clone()) if move.i == case.move else None,
        )
        states.append(grabbed["s"])
    evaluator = Evaluator(build_model("transformer", 32, 1), "cpu")
    for readings in (
        ts.read_network(evaluator, states),
        ts.read_search(evaluator, states, 32, exact_tactics=True),
    ):
        assert len(readings) == len(picked)
        report = ts.score(picked, readings)
        assert {name.split("/")[0] for name in report} <= set(ts.CLASSES)
        for state, reading in zip(states, readings):
            assert reading.action in legal_action_indices(state)
            assert -1.0 <= reading.value <= 1.0
    # The suite leaves the process-wide switch as it found it.
    assert swr.exact_tactics() is False


def test_paired_compare_counts_discordant_pairs_and_clusters_by_game(tmp_path):
    def write(path, flips):
        with path.open("w", encoding="utf-8") as handle:
            for i in range(40):
                handle.write(json.dumps({
                    "mode": "network", "id": f"c{i}", "cls": "must_block",
                    "near_end": i < 10, "game": [1, i // 2],
                    "value": 0.0, "abs_error": 0.5 if i in flips else 0.2,
                    "action": 0, "blunder": i in flips,
                }) + "\n")

    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write(a, flips=set(range(12)))  # 12 blunders
    write(b, flips={0, 1, 30})  # 3 blunders: 10 fixed, 1 new
    report = ts.compare(a, b, draws=500)
    entry = report["modes"]["network"]["must_block"]
    assert entry["cases"] == 40 and entry["games"] == 20
    stats = entry["blunder"]
    assert stats["a"] == pytest.approx(12 / 40) and stats["b"] == pytest.approx(3 / 40)
    assert (stats["only_a"], stats["only_b"]) == (10, 1)
    assert stats["mcnemar_p"] < 0.02
    low, high = stats["diff_ci95"]
    assert low < -0.225 < high < 0  # mean diff -9/40, and clearly below zero
    assert "must_block/near_end" in report["modes"]["network"]
    assert ts._mcnemar(0, 0) == 1.0 and ts._mcnemar(5, 5) == 1.0


def test_search_never_targets_a_proven_losing_move(cases, records):
    """G4 guard: at a must_block position a low-budget search's target and move
    carry no mass on a move PROVEN to lose while another exists."""

    from .inference import Evaluator
    from .train import build_model

    picked = [c for c in cases if c.cls == "must_block"][:8]
    assert picked
    by_game = {(r.iteration, r.seed): r for r in records}
    states = []
    for case in picked:
        grabbed = {}
        replay(
            by_game[(case.iteration, case.seed)],
            on_state=lambda game, move, grabbed=grabbed, case=case: grabbed.setdefault(
                "s", game.clone()) if move.i == case.move else None,
        )
        states.append(grabbed["s"])
    evaluator = Evaluator(build_model("transformer", 32, 1), "cpu")
    readings = ts.read_search(evaluator, states, 16, exact_tactics=True)
    for case, reading in zip(picked, readings):
        assert reading.action not in case.losing
        assert sum(reading.mass.get(a, 0.0) for a in case.losing) == pytest.approx(0.0, abs=1e-12)
