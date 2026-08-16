"""Gates for the BGA differential harness.

The harness is how the engine is checked against an authority that is not us,
so what these tests protect is mostly *its ability to fail*: a replay that
silently drifts, or one that quietly compares nothing, would report a clean
game either way.

``testdata/bga_packets.json`` holds BGA's notification packets for two real
games -- table 899263451 (ended by concession, the only captured game with
Economy transfers) and 899383864 (played to full end-game scoring, and the only
one exercising the rebate token, an opponent coin loss and an immediate progress
token payout). Both are trimmed to the entries the harness reads, with the
packet sequence left intact; the trim was accepted only after checking it
reproduced the full logs' counters and verdict exactly.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from games.advisor.game_log import GameLogWriter
from games.seven_wonders_duel import bga_differential as D

FIXTURE = Path(__file__).parent / "testdata" / "bga_packets.json"

SCORED_TABLE = "899383864"  # played out; end-game scoring published
CONCEDED_TABLE = "899263451"  # conceded; no end-game scoring, but Economy used


@pytest.fixture(scope="module")
def corpus() -> dict[str, list[dict]]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_real_games_match_bga_exactly(corpus):
    """The headline gate: our arithmetic equals BGA's on every published number."""

    for table, packets in corpus.items():
        result = D.replay(table, packets)
        assert result.skipped is None, result.skipped
        assert result.problems == [], f"{table}: {result.problems}"


def test_every_check_category_is_exercised(corpus):
    """A silent harness passes too, so assert the corpus really drives each check."""

    totals = dict.fromkeys(D.COUNTER_NAMES, 0)
    for table, packets in corpus.items():
        for name, count in D.replay(table, packets).counters.items():
            totals[name] += count

    unexercised = [name for name, count in totals.items() if count == 0]
    assert unexercised == [], f"no captured game exercises: {unexercised}"


def test_final_scoring_is_compared_only_where_it_exists(corpus):
    """A conceded game never scores; that is silence, not a pass."""

    scored = D.replay(SCORED_TABLE, corpus[SCORED_TABLE])
    conceded = D.replay(CONCEDED_TABLE, corpus[CONCEDED_TABLE])
    assert scored.counters["final_score_checks"] == 2  # both seats
    assert conceded.counters["final_score_checks"] == 0
    assert conceded.problems == []


def test_a_corrupted_price_is_caught(corpus):
    """Perturb BGA's side and the harness must disagree -- otherwise it is decoration."""

    packets = copy.deepcopy(corpus[SCORED_TABLE])
    for packet in packets:
        for entry in packet["data"]:
            # BGA writes a free build's cost as "", not 0.
            cost = D._as_int(entry["args"].get("cost"))
            if entry["type"] == "constructBuilding" and cost > 0:
                entry["args"]["cost"] = cost + 1
                break
        else:
            continue
        break

    result = D.replay(SCORED_TABLE, packets)
    assert any("cost ours=" in p for p in result.problems)


def test_start_player_comes_from_the_draft_and_is_cross_checked(corpus):
    """Seat 0 is BGA's start player; getting it wrong flips the military sign."""

    packets = corpus[SCORED_TABLE]
    derived = D.start_player_from_packets(packets)
    assert derived is not None

    # Agreeing with the logged position is the ordinary case.
    assert D.replay(SCORED_TABLE, packets, start_player=derived).skipped is None

    # Disagreeing is refused rather than replayed under a guess.
    result = D.replay(SCORED_TABLE, packets, start_player="999999")
    assert result.skipped is not None and "start player disagrees" in result.skipped
    assert result.problems == []


def test_incomplete_capture_is_skipped_not_replayed(corpus):
    """One dropped construct silently rebases every later coin comparison."""

    packets = [p for p in corpus[SCORED_TABLE] if str(p["move_id"]) != "20"]
    result = D.replay(SCORED_TABLE, packets)
    assert result.skipped is not None and "missing move" in result.skipped
    assert result.problems == []


def test_missing_moves_reports_the_gaps(corpus):
    packets = [p for p in corpus[SCORED_TABLE] if str(p["move_id"]) not in {"5", "9"}]
    assert D.missing_moves(packets) == [5, 9]


def test_packets_are_read_deduped_and_ordered_from_the_game_log(tmp_path, corpus):
    """Each capture re-posts the whole history, so the reader must dedupe it."""

    packets = corpus[SCORED_TABLE]
    writer = GameLogWriter(tmp_path)
    # Three overlapping captures, as a real game produces: growing prefixes.
    for cut in (10, 40, len(packets)):
        writer.append(
            None,
            table_id=SCORED_TABLE,
            kind=D.PACKET_KIND,
            extra={"packets": packets[:cut]},
        )

    collected = D.packets_by_table(tmp_path)
    assert list(collected) == [SCORED_TABLE]
    assert len(collected[SCORED_TABLE]) == len(packets)
    # Numeric order, not the string order BGA's move ids would sort in.
    moves = [int(p["move_id"]) for p in collected[SCORED_TABLE]]
    assert moves == sorted(moves)
    assert D.replay(SCORED_TABLE, collected[SCORED_TABLE]).problems == []


def test_a_fuller_copy_of_a_packet_wins(tmp_path, corpus):
    """Preferring whichever copy was read first would drop real entries."""

    packets = corpus[SCORED_TABLE]
    thin = [dict(p, data=[]) for p in packets]
    writer = GameLogWriter(tmp_path)
    writer.append(None, table_id=SCORED_TABLE, kind=D.PACKET_KIND, extra={"packets": thin})
    writer.append(
        None, table_id=SCORED_TABLE, kind=D.PACKET_KIND, extra={"packets": packets}
    )

    collected = D.packets_by_table(tmp_path)[SCORED_TABLE]
    assert D.replay(SCORED_TABLE, collected).counters["priced"] > 0


def test_run_reports_nothing_rather_than_passing_on_an_empty_log(tmp_path):
    """An empty corpus must not read as a clean bill of health."""

    results = D.run(tmp_path)
    assert results == []
    assert "nothing was compared" in D.format_report(results)


def test_report_names_the_unexercised_checks(corpus):
    """The loose ends worth chasing are the checks no real game has reached yet."""

    report = D.format_report([D.replay(CONCEDED_TABLE, corpus[CONCEDED_TABLE])])
    assert "final_score_checks" in report
    assert "NOT EXERCISED" in report
