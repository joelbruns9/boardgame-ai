"""Replaying games from BGA's notification packets.

A real recon game (table 925113041, trimmed: every packet, a quarter of the
board captures) pins the wire format; simulated games written in the same
format pin the replay against known truth on other rule sets.
"""

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest

from games.cantstop.advisor_adapter import wins_on_stop
from games.cantstop.bga_packets import (PacketMismatch, packet_events, replay,
                                        starts_at_game_start)
from games.cantstop.engine import (COLUMN_HEIGHTS, GameState, Phase, RuleSet,
                                   apply_move, can_stop, random_dice, roll,
                                   stop)
from games.cantstop.luck import (board_key, build_ledger, load_game,
                                 read_turns)
from games.cantstop.rust_solver import RustTurnSolver
from games.cantstop.solver import ProgressHeuristic

FIXTURE = Path(__file__).parent / "fixtures" / "bga_packets_925113041.jsonl"
HEURISTIC = ProgressHeuristic()


def rows():
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


def real_replay(rs=None):
    game = load_game(FIXTURE)
    return replay(packet_events(rs or rows()), game.captures[0].state.rules, game.player_ids), game


def test_real_game_replays_and_matches_every_board_capture():
    (turns, final), game = real_replay()
    assert starts_at_game_start(packet_events(rows()))
    assert len(turns) == 29 and final.game_over and final.winner == 1
    assert [sum(t.end == e for t in turns) for e in ("stop", "bust", "win")] == [14, 14, 1]
    seen = set()
    for t in turns:
        for s in t.steps:
            if s.dice is not None:
                b = t.start.clone()
                seen.add((board_key(b), tuple(sorted(s.runners.items())), tuple(sorted(s.dice))))
    for c in game.captures:
        assert (board_key(c.state), tuple(sorted(c.state.runners.items())), c.state.dice) in seen


def test_real_game_ledger_has_nothing_unobserved():
    game = load_game(FIXTURE)
    turns = read_turns(FIXTURE, game)
    assert game.source == "packets"
    L = build_ledger(game, turns, HEURISTIC)
    assert not any(e["kind"] == "gap" or e.get("inferred") for e in L.entries)
    busts = [e for e in L.entries if e["kind"] == "luck" and e["busted"]]
    assert len(busts) == 14 and all(e["bust_dice"] for e in busts)
    np.testing.assert_allclose(L.outcome, [0.0, 1.0])
    np.testing.assert_allclose(L.start + sum(e["delta"] for e in L.entries), L.outcome, atol=1e-9)


def _mutate(kind, change, skip=0):
    rs = copy.deepcopy(rows())
    for r in rs:
        for p in (r.get("extra") or {}).get("packets") or []:
            for e in p["data"]:
                if e["type"] == kind:
                    if skip == 0:
                        change(e, p)
                        return rs
                    skip -= 1
    raise AssertionError("nothing to mutate")


@pytest.mark.parametrize("kind,change", [
    ("moveToken", lambda e, p: e["args"].update(height=int(e["args"]["height"]) - 1)),
    ("rollDice", lambda e, p: e["args"].update(dice=[1, 1, 1, 1])),
    ("rollDice", lambda e, p: e["args"].update(           # the other player rolls
        player_id=96364907 if str(e["args"]["player_id"]) == "89146710" else 89146710)),
    ("saveProgress", lambda e, p: e["args"].update(column_list="7")),
    ("rollDice", lambda e, p: p["data"].remove(e)),
])
def test_a_corrupted_stream_does_not_replay(kind, change):
    with pytest.raises(PacketMismatch):
        real_replay(_mutate(kind, change, skip=3))


def test_a_stream_with_a_hole_is_not_a_whole_game():
    events = [e for e in packet_events(rows()) if e[0] != 40]
    assert not starts_at_game_start(events)


# ---- simulated games, written as BGA packets ----

def simulate_packets(rules, seed, deviate=0.25):
    """Play a game and write it the way BGA notifies it."""
    rng = random.Random(seed)
    ids = [str(100 + i) for i in range(rules.num_players)]
    state = GameState(rules)
    packets, truth, move_id = [], [], 0

    def emit(*data):
        nonlocal move_id
        move_id += 1
        packets.append({"move_id": move_id, "packet_id": move_id, "data": list(data)})

    while not state.game_over:
        solver = RustTurnSolver(state, HEURISTIC)
        while True:
            seat = state.active_player
            dice = random_dice(rng)
            moves = roll(state, dice)
            emit({"type": "rollDice", "args": {"player_id": ids[seat], "dice": list(dice)}})
            if not moves:
                truth.append((seat, None, "bust"))
                break
            m = rng.choice(moves) if rng.random() < deviate else solver.choose_move(state)
            apply_move(state, m)
            emit(*[{"type": "moveToken", "args": {
                "column_id": c, "height": COLUMN_HEIGHTS[c] - state.runners[c]}}
                for c in sorted(set(m))],
                {"type": "gameStateChange", "args": {"id": 3}})
            if wins_on_stop(state) or (can_stop(state) and (
                    rng.random() < deviate / 2 or solver.should_stop(state))):
                claimed = [c for c, p in state.runners.items() if p >= COLUMN_HEIGHTS[c]]
                tokens = {str(c): COLUMN_HEIGHTS[c] - p for c, p in state.runners.items()}
                stop(state)
                truth.append((seat, m, "stop"))
                emit({"type": "saveProgress", "args": {
                    "column_list": ",".join(map(str, claimed)), "tokens": tokens}},
                    {"type": "gameStateChange", "args": {"id": 99 if state.game_over else 7}})
                break
            truth.append((seat, m, "roll"))
            emit({"type": "gameStateChange", "args": {"id": 10}})
    return [{"kind": "bga_packets", "extra": {"packets": packets}}], ids, truth, state.winner


@pytest.mark.parametrize("rules", [RuleSet.make(2, extended=True),
                                   RuleSet.make(3, extended=True, blocking=True),
                                   RuleSet.make(4, blocking=True)])
def test_simulated_games_replay_to_the_truth(rules):
    rs, ids, truth, winner = simulate_packets(rules, 5)
    turns, final = replay(packet_events(rs), rules, ids)
    got = []
    for t in turns:
        for s in t.steps:
            got.append((t.seat, s.move, "bust" if s.dice is None else s.then))
    # a bust step carries no move; the move before it is on the previous step
    assert [(s, m, then) for s, m, then in got] == [
        (s, None if then == "bust" else m, then) for s, m, then in truth]
    assert final.winner == winner


def test_live_luck_summary_from_a_game_log(tmp_path):
    from games.cantstop.advisor_adapter import CantStopAdvisor
    from games.cantstop.live_luck import LiveLuck
    live = LiveLuck(CantStopAdvisor(evaluator=HEURISTIC), tmp_path)
    assert not live.summary("925113041")["available"]          # nothing logged
    log = tmp_path / "table_925113041.jsonl"
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    # board captures only: no packet record, so no numbers
    log.write_text("\n".join(l for l in lines if '"bga_packets"' not in l) + "\n", encoding="utf-8")
    assert not live.summary("925113041")["available"]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    r = live.summary("925113041")
    assert r["available"] and r["game_over"] and r["turns"] == 29
    by = {p["player_id"]: p for p in r["players"]}
    assert sum(p["busts"] for p in by.values()) == 14
    assert by["89146710"]["dice_pts"] == pytest.approx(-by["96364907"]["dice_pts"], abs=0.11)
    again = live.summary("925113041")                            # cached solves
    assert again == r


def test_a_damaged_log_line_is_skipped(tmp_path):
    """One line holding only the tail of a record (the 2026-10-03 write
    race) must not stop the game from loading or replaying."""
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    i = next(k for k, l in enumerate(lines) if '"decision"' in l and k > 40)
    lines[i] = lines[i][len(lines[i]) // 2:]
    log = tmp_path / "table_925113041.jsonl"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    game = load_game(log)
    assert game.skipped >= 1
    turns = read_turns(log, game)
    assert game.source == "packets" and turns[-1].end == "win"


def test_a_wakeup_nudge_without_a_move_id_is_not_a_hole():
    """BGA nudges an idle table with a packet that has no move id; the
    record is still whole. A move-less packet that carried a roll is not."""
    rs = rows()
    rs.append({"kind": "bga_packets", "extra": {"packets": [
        {"move_id": None, "packet_id": 98, "data": [{"type": "wakeupPlayers", "args": []}]}]}})
    assert starts_at_game_start(packet_events(rs))
    (turns, final), _ = real_replay(rs)
    assert final.winner == 1
    rs.append({"kind": "bga_packets", "extra": {"packets": [
        {"move_id": None, "packet_id": 99, "data": [{"type": "rollDice", "args": {"dice": [1, 1, 1, 1]}}]}]}})
    assert not starts_at_game_start(packet_events(rs))


def test_a_game_ended_without_a_winner_still_replays():
    """A concession or timeout: BGA sends game over mid-game. The record up
    to then replays; the result is left unknown."""
    events = packet_events(rows())
    cut = next(i for i, (m, k, a) in enumerate(events) if m >= 120 and k == "rollDice")
    events = events[:cut] + [(events[cut][0], "gameStateChange", {"id": 99})]
    game = load_game(FIXTURE)
    turns, final = replay(events, game.captures[0].state.rules, game.player_ids)
    assert not final.game_over and turns[-1].end == "ended_early"
    L = build_ledger(game, turns, HEURISTIC)
    assert L.outcome is None
