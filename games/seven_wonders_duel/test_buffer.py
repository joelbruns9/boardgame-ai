"""Buffer-schema gates: bit-exact replay, chance-log cross-check, JSON
round-trip stability (CODEC_SPEC.md §6)."""

import random

import pytest

from games.seven_wonders_duel.buffer import (
    GameRecorder,
    ReplayMismatchError,
    from_json_line,
    read_records,
    replay,
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
        assert state_digest(final) == record.final_digest
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
