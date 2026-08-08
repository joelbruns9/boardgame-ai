"""Buffer-schema gates: bit-exact replay, chance-log cross-check, JSON
round-trip stability (CODEC_SPEC.md §6)."""

from dataclasses import replace
import hashlib
import json
import random

import pytest

from games.seven_wonders_duel.buffer import (
    LEGACY_DIGEST_VERSION,
    LOGIC_DIGEST_VERSION,
    GameRecorder,
    OPPONENT_TYPES,
    ReplayMismatchError,
    from_json_line,
    logic_state_digest,
    read_records,
    replay,
    resolve_opponent_type,
    state_digest,
    to_json_line,
)
from games.seven_wonders_duel.codec import legal_action_indices
from games.seven_wonders_duel.game import ChanceKind, Phase


def _record_random_game(seed, with_stats=False):
    recorder = GameRecorder(
        seed, first_player=seed % 2, agents={"p0": "random", "p1": "random"}
    )
    rng = random.Random(seed * 31 + 7)
    while recorder.game.phase is not Phase.COMPLETE:
        indices = legal_action_indices(recorder.game)
        index = rng.choice(indices)
        if with_stats:
            recorder.play(
                index,
                visits={i: 1 for i in indices[:3]},
                root_value=0.25,
                sims=64,
                mode="closed",
                gumbel_topk=tuple(indices[:4]),
            )
        else:
            recorder.play(index)
    return recorder.finish()


def test_replay_reproduces_games_bit_exactly():
    for seed in range(25):
        record = _record_random_game(seed)
        final = replay(record)  # raises on any mask/chance/digest divergence
        assert logic_state_digest(final) == record.final_digest
        assert record.digest_version == LOGIC_DIGEST_VERSION
        assert final.winner == record.winner
        assert (final.victory_type.value if final.victory_type else None) == (
            record.victory_type
        )
        assert record.chance_log, "full games always contain chance events"
        kinds = {kind for kind, _ in record.chance_log}
        assert ChanceKind.WONDER_GROUP_REVEAL.value in kinds
        assert ChanceKind.AGE_DEAL.value in kinds


def test_json_round_trip_is_byte_stable():
    record = _record_random_game(3, with_stats=True)
    line = to_json_line(record)
    recovered = from_json_line(line)
    assert recovered == record
    assert to_json_line(recovered) == line
    assert replay(recovered).phase is Phase.COMPLETE


def test_legacy_rng_digest_records_remain_replayable_and_missing_field_defaults():
    recorder = GameRecorder(
        17,
        first_player=1,
        digest_version=LEGACY_DIGEST_VERSION,
    )
    rng = random.Random(17)
    while recorder.game.phase is not Phase.COMPLETE:
        recorder.play(rng.choice(legal_action_indices(recorder.game)))
    record = recorder.finish()
    assert replay(record).phase is Phase.COMPLETE
    payload = json.loads(to_json_line(record))
    payload.pop("digest_version")
    legacy_line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    recovered = from_json_line(legacy_line)
    assert recovered.digest_version == LEGACY_DIGEST_VERSION
    assert replay(recovered).phase is Phase.COMPLETE


def test_parsing_memoizes_content_digest_and_mutation_invalidates_it():
    record = _record_random_game(3, with_stats=True)
    line = to_json_line(record)
    recovered = from_json_line(line)
    assert recovered.source_digest == hashlib.sha256(line.encode("utf-8")).hexdigest()
    assert replace(recovered, iteration=999).source_digest is None


@pytest.mark.parametrize(
    ("agents", "expected"),
    [
        ({"kind": "self_play"}, "current_best"),
        ({"kind": "mixed"}, "bot"),
        ({"kind": "curriculum_seed"}, "bot"),
        ({"kind": "league"}, "hof"),
        ({"kind": "league_mixed"}, "hof_bot"),
        (
            {"kind": "self_play", "opponent_type": "hof_bot"},
            "hof_bot",
        ),
    ],
)
def test_opponent_type_is_explicit_with_legacy_fallback(agents, expected):
    assert resolve_opponent_type(agents) == expected
    assert expected in OPPONENT_TYPES


def test_unknown_explicit_opponent_type_is_rejected():
    with pytest.raises(ValueError, match="unknown opponent_type"):
        resolve_opponent_type({"opponent_type": "mystery"})


def test_jsonl_file_round_trip(tmp_path):
    from games.seven_wonders_duel.buffer import append_records

    records = [_record_random_game(seed) for seed in (5, 6)]
    path = tmp_path / "buffer.jsonl"
    append_records(path, records)
    append_records(path, [_record_random_game(7, with_stats=True)])
    recovered = read_records(path)
    assert len(recovered) == 3
    assert recovered[:2] == records
    for record in recovered:
        replay(record)


def test_search_stats_survive_round_trip():
    record = _record_random_game(3, with_stats=True)
    move = record.moves[0]
    assert move.sims == 64 and move.mode == "closed"
    assert move.visits and all(isinstance(k, int) for k in move.visits)
    assert move.gumbel_topk is not None and len(move.gumbel_topk) <= 4
    recovered = from_json_line(to_json_line(record))
    assert recovered.moves[0] == move


def test_replay_detects_tampered_actions():
    record = _record_random_game(9)
    moves = list(record.moves)
    # Swap in a different (still in-range) action for move 5.
    tampered_action = (moves[5].action + 1) % 1202
    import dataclasses

    moves[5] = dataclasses.replace(moves[5], action=tampered_action)
    tampered = dataclasses.replace(record, moves=tuple(moves))
    with pytest.raises((ReplayMismatchError, ValueError)):
        replay(tampered)


def test_replay_detects_tampered_chance_log():
    record = _record_random_game(9)
    import dataclasses

    log = list(record.chance_log)
    kind, outcome = log[0]
    log[0] = (kind, "Lumber Yard" if outcome != "Lumber Yard" else "Clay Pool")
    tampered = dataclasses.replace(record, chance_log=tuple(log))
    with pytest.raises(ReplayMismatchError):
        replay(tampered)


@pytest.mark.parametrize("field", ["final_digest", "trajectory_digest"])
def test_python_reference_remains_the_rng_inclusive_digest_audit(field):
    record = _record_random_game(9)
    tampered = replace(record, **{field: "sha256:" + "0" * 64})
    with pytest.raises(ReplayMismatchError):
        replay(tampered)
