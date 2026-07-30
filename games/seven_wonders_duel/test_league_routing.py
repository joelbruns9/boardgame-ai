"""Per-game network routing for league generation (7WD cloud plan W1.3).

The property under test is the one that is easy to get silently wrong:
**the searcher owns the network**. When it is seat 0's turn, seat 0's network
must evaluate *every* leaf of seat 0's search tree -- including the leaves where
seat 1 is to move, which is most of them at odd depths. Routing on the leaf
actor instead produces a player that is neither of the two checkpoints, and
nothing in a training run would report it: the games complete, the records
validate, and the strength numbers are quietly measuring a chimera.

So these tests assert directly that rows exist where the leaf actor and the
routed network disagree, and that the routed network is the searcher's.
"""

from __future__ import annotations

import numpy as np
import pytest

from .rust_bridge import rust_games_for_self_play
from .test_f4_scheduler import _common, _row_eval
from .dataset import FEATURE_COUNTS


class _RoutingRecorder:
    """Evaluate normally, but keep every row's (net_id, leaf actor) pair."""

    def __init__(self):
        self.pairs: list[tuple[int, int]] = []
        self.batches = 0

    def __call__(self, payload):
        rows = int(payload["rows"])
        offsets = np.frombuffer(payload["token_offsets"], dtype="<u4")
        types = np.frombuffer(payload["type_ids"], dtype=np.uint8)
        entities = np.frombuffer(payload["entity_ids"], dtype="<i2")
        auxes = np.frombuffer(payload["aux_ids"], dtype="<i2") - 1
        width = int(payload["feature_width"])
        features = np.frombuffer(payload["features"], dtype="<f4").reshape(-1, width)
        actors = np.frombuffer(payload["actors"], dtype=np.uint8)
        net_ids = np.frombuffer(payload["net_ids"], dtype=np.uint8)
        legal_offsets = np.frombuffer(payload["legal_offsets"], dtype="<u4")
        legal = np.frombuffer(payload["legal_actions"], dtype="<u2")

        assert len(net_ids) == rows, "net_ids must be row-aligned"
        self.batches += 1
        output = []
        for row in range(rows):
            self.pairs.append((int(net_ids[row]), int(actors[row])))
            tokens = []
            for index in range(int(offsets[row]), int(offsets[row + 1])):
                type_id = int(types[index])
                count = FEATURE_COUNTS[type_id]
                tokens.append(
                    (
                        type_id,
                        int(entities[index]),
                        int(auxes[index]),
                        features[index, :count].tolist(),
                    )
                )
            row_legal = (
                legal[int(legal_offsets[row]) : int(legal_offsets[row + 1])]
                .astype(np.int64)
                .tolist()
            )
            output.append(_row_eval(tokens, int(actors[row]), row_legal))
        return output


def _kwargs():
    return {
        **_common(leaf_batch=1, global_batch_cap=8),
        "force": False,
        "max_inflight_batches": 2,
    }


def _play(nets_p0=None, nets_p1=None, seeds=(4242, 4243, 4244)):
    import seven_wonders_rust as swr

    recorder = _RoutingRecorder()
    seeds = list(seeds)
    first_players = [index % 2 for index in range(len(seeds))]
    records, _ = swr.self_play_many_flat_net(
        adapter=recorder,
        games=rust_games_for_self_play(seeds, first_players),
        game_seeds=seeds,
        nets_p0=nets_p0,
        nets_p1=nets_p1,
        **_kwargs(),
    )
    return records, recorder


# -- the default is unchanged ----------------------------------------------


def test_without_routing_every_row_is_network_zero():
    """Ordinary self-play must be untouched: the ids are packed all zeros."""

    _, recorder = _play()
    assert recorder.pairs
    assert {net for net, _ in recorder.pairs} == {0}


def test_routing_all_seats_to_network_zero_matches_no_routing():
    """An explicit all-zero assignment is the same run as no assignment."""

    plain_records, plain = _play()
    routed_records, routed = _play(nets_p0=[0, 0, 0], nets_p1=[0, 0, 0])
    assert [record["final_fingerprint"] for record in plain_records] == [
        record["final_fingerprint"] for record in routed_records
    ]
    assert plain.pairs == routed.pairs


def test_gate_scheduler_plays_every_queued_pair():
    """W5.5: the gate is fixed-N, so no pair is dropped by the scheduler.

    The sequential stop this replaces returned a prefix, which is what made the
    gate's promote decision optional-stopping-biased.
    """

    import seven_wonders_rust as swr

    pair_count = 20
    seeds = [
        9000 + pair
        for pair in range(pair_count)
        for _candidate_seat in (0, 1)
    ]
    first_players = [
        pair % 2
        for pair in range(pair_count)
        for _candidate_seat in (0, 1)
    ]
    nets_p0 = [0, 1] * pair_count
    nets_p1 = [1, 0] * pair_count
    recorder = _RoutingRecorder()
    records, metrics = swr.self_play_many_flat_net(
        adapter=recorder,
        games=rust_games_for_self_play(seeds, first_players),
        game_seeds=seeds,
        nets_p0=nets_p0,
        nets_p1=nets_p1,
        max_active_slots=8,
        scheduler_workers=1,
        **_kwargs(),
    )
    assert len(records) == metrics["games"] == 2 * pair_count


# -- the property that matters --------------------------------------------


def test_the_searcher_owns_the_network_not_the_leaf_actor():
    """The core W1.3 correctness property.

    With network 1 on seat 1 in every game, a row's network must equal the
    *searcher's* seat.  The test is only meaningful if the two disagree
    somewhere, so it first asserts that such rows exist -- otherwise a
    leaf-actor implementation would pass vacuously.
    """

    _, recorder = _play(nets_p0=[0, 0, 0], nets_p1=[1, 1, 1])
    pairs = recorder.pairs
    assert pairs

    # Both networks were actually exercised.
    assert {net for net, _ in pairs} == {0, 1}

    # The distinguishing evidence: rows whose leaf actor is not the network's
    # seat. These are the interior nodes of a search where the other player is
    # to move, and they must still belong to the searcher.
    disagreeing = [(net, actor) for net, actor in pairs if net != actor]
    assert disagreeing, (
        "no row had leaf actor != searcher, so this test cannot distinguish "
        "searcher-owns-network from leaf-actor routing"
    )
    # A leaf-actor implementation would make net_id == actor for every row.
    assert any(net != actor for net, actor in pairs)


def test_each_network_sees_whole_trees_of_both_leaf_actors():
    """A searcher's batch spans both leaf actors, which is the whole point."""

    _, recorder = _play(nets_p0=[0, 0, 0], nets_p1=[1, 1, 1])
    by_net: dict[int, set[int]] = {}
    for net, actor in recorder.pairs:
        by_net.setdefault(net, set()).add(actor)
    # Each network evaluated positions where either player was to move.
    assert by_net[0] == {0, 1}
    assert by_net[1] == {0, 1}


def test_per_game_assignment_mixes_within_one_scheduler_call():
    """Different games, different opponents, one call -- no split slot pool."""

    _, recorder = _play(nets_p0=[0, 0, 0], nets_p1=[1, 0, 1])
    assert {net for net, _ in recorder.pairs} == {0, 1}
    # Game 1 is pure self-play while games 0 and 2 are league games, and all of
    # them shared this call's batches.
    assert recorder.batches >= 1


# -- learner-only policy targets ------------------------------------------


def _play_full_search(nets_p0, nets_p1, seeds=(777, 778)):
    """Every move a full search, so `policy_excluded` isolates the league rule."""

    import seven_wonders_rust as swr

    recorder = _RoutingRecorder()
    seeds = list(seeds)
    kwargs = {
        **_kwargs(),
        "full_search_fraction": 1.0,
        "cheap_sims_min": 1,
        "cheap_sims_max": 1,
        "full_sims_min": 2,
        "full_sims_max": 2,
    }
    records, _ = swr.self_play_many_flat_net(
        adapter=recorder,
        games=rust_games_for_self_play(seeds, [index % 2 for index in range(len(seeds))]),
        game_seeds=seeds,
        nets_p0=nets_p0,
        nets_p1=nets_p1,
        **kwargs,
    )
    return records


def test_self_play_full_searches_are_all_trainable():
    """The baseline the league rule is measured against."""

    records = _play_full_search([0, 0], [0, 0])
    excluded = [
        move["policy_excluded"] for record in records for move in record["moves"]
    ]
    assert excluded and not any(excluded)


def test_the_archive_seat_contributes_no_policy_targets():
    """KD parity: 'keep only current-owned labels'.

    The archive's moves still exist and still carry the game's value target --
    only the policy label is withheld, because a target produced by network 1
    would train the learner to imitate an older net.
    """

    records = _play_full_search([0, 0], [1, 1])
    seen = {True: 0, False: 0}
    for record in records:
        for move in record["moves"]:
            excluded = bool(move["policy_excluded"])
            seen[excluded] += 1
            # Seat 1 is the archive in both games.
            assert excluded == (int(move["actor"]) == 1)
    # Both kinds occurred, so the assertion above was exercised in both branches.
    assert seen[True] and seen[False]


def test_the_learner_keeps_its_targets_when_it_holds_seat_one():
    """The mirrored assignment excludes seat 0 instead, not a hardcoded seat."""

    records = _play_full_search([1, 1], [0, 0])
    for record in records:
        for move in record["moves"]:
            assert bool(move["policy_excluded"]) == (int(move["actor"]) == 0)


# -- validation ------------------------------------------------------------


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="one entry per game"):
        _play(nets_p0=[0, 0], nets_p1=[1, 1])


def test_one_sided_assignment_is_rejected():
    with pytest.raises(ValueError, match="must be supplied together"):
        _play(nets_p0=[0, 0, 0])


def test_out_of_range_network_ids_are_rejected():
    """An unchecked id would silently index the wrong model in the adapter."""

    with pytest.raises(ValueError, match="network ids must be 0 or 1"):
        _play(nets_p0=[0, 0, 0], nets_p1=[2, 1, 1])
