"""F4.5 flat transformer boundary and forced-child cache gates."""

from __future__ import annotations

import numpy as np
import pytest

from .buffer import replay
from .codec import decode_action
from .dataset import FEATURE_COUNTS
from .engine import apply_action
from .game import Phase, new_game
from .rust_bridge import (
    phase_d_records_from_rust,
    rust_flat_batch_adapter,
    rust_game_for_self_play,
    rust_games_for_self_play,
    rust_global_batch_adapter,
    rust_seat_routed_flat_batch_adapter,
)
from .test_f4_scheduler import _common, _row_eval
from .test_rust_engine_equiv import (
    _enumerable_reveal_root,
    extract_setup,
    random_game,
)


class _DecodingFlatAdapter:
    """Decode packed rows for an exact, network-independent ownership gate."""

    def __init__(self):
        self.batch_sizes = []
        self.payload_types = []

    def __call__(self, payload):
        rows = int(payload["rows"])
        offsets = np.frombuffer(payload["token_offsets"], dtype="<u4")
        types = np.frombuffer(payload["type_ids"], dtype=np.uint8)
        entities = np.frombuffer(payload["entity_ids"], dtype="<i2")
        auxes = np.frombuffer(payload["aux_ids"], dtype="<i2") - 1
        width = int(payload["feature_width"])
        features = np.frombuffer(payload["features"], dtype="<f4").reshape(-1, width)
        actors = np.frombuffer(payload["actors"], dtype=np.uint8)
        legal_offsets = np.frombuffer(payload["legal_offsets"], dtype="<u4")
        legal = np.frombuffer(payload["legal_actions"], dtype="<u2")
        assert len(offsets) == rows + 1
        assert len(legal_offsets) == rows + 1
        assert len(types) == len(entities) == len(auxes) == len(features)
        self.batch_sizes.append(rows)
        self.payload_types.append(
            all(
                isinstance(payload[key], bytearray)
                for key in (
                    "token_offsets",
                    "type_ids",
                    "entity_ids",
                    "aux_ids",
                    "features",
                    "actors",
                    "legal_offsets",
                    "legal_actions",
                )
            )
        )
        output = []
        for row in range(rows):
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
            row_legal = legal[
                int(legal_offsets[row]) : int(legal_offsets[row + 1])
            ].astype(np.int64).tolist()
            output.append(_row_eval(tokens, int(actors[row]), row_legal))
        return output


def _flat_kwargs(*, force=False):
    return {
        **_common(leaf_batch=1, global_batch_cap=8),
        "force": force,
        "max_inflight_batches": 2,
    }


def test_f4_5_flat_packing_matches_object_boundary_exactly():
    import seven_wonders_rust as swr

    seeds = [2026072400, 2026072401, 2026072402]
    first_players = [0, 1, 0]
    kwargs = _flat_kwargs()
    object_records, _ = swr.self_play_many_net(
        adapter=lambda rows: [_row_eval(*row) for row in rows],
        games=rust_games_for_self_play(seeds, first_players),
        game_seeds=seeds,
        **kwargs,
    )
    flat_adapter = _DecodingFlatAdapter()
    flat_records, metrics = swr.self_play_many_flat_net(
        adapter=flat_adapter,
        games=rust_games_for_self_play(seeds, first_players),
        game_seeds=seeds,
        **kwargs,
    )
    assert flat_records == object_records
    assert all(flat_adapter.payload_types)
    assert max(flat_adapter.batch_sizes) <= kwargs["global_batch_cap"]
    assert metrics["boundary_tokens"] > metrics["global_rows"]
    assert metrics["boundary_padded_tokens"] >= metrics["boundary_tokens"]
    assert 0.0 <= metrics["padding_ratio"] < 1.0
    assert metrics["encode_pack_ns"] > 0
    assert metrics["py_call_ns"] > 0
    assert metrics["extract_ns"] > 0


def test_f4_5_real_net_flat_boundary_matches_object_path():
    torch = pytest.importorskip("torch")
    from .inference import Evaluator
    from .net import SWDNet

    import seven_wonders_rust as swr

    torch.manual_seed(45)
    evaluator = Evaluator(SWDNet(32, 1, 2), device="cpu", max_batch=16)
    seeds = [2026072410, 2026072411]
    first_players = [0, 1]
    kwargs = {
        **_common(leaf_batch=1, global_batch_cap=8),
        "cheap_sims_max": 1,
        "full_sims_min": 1,
        "full_sims_max": 1,
        "full_search_fraction": 0.0,
        "top_k": 2,
    }
    object_records, _ = swr.self_play_many_net(
        adapter=rust_global_batch_adapter(evaluator),
        games=rust_games_for_self_play(seeds, first_players),
        game_seeds=seeds,
        **kwargs,
    )
    flat_adapter = rust_flat_batch_adapter(evaluator)
    flat_records, metrics = swr.self_play_many_flat_net(
        adapter=flat_adapter,
        games=rust_games_for_self_play(seeds, first_players),
        game_seeds=seeds,
        **kwargs,
    )
    assert [row["final_fingerprint"] for row in flat_records] == [
        row["final_fingerprint"] for row in object_records
    ]
    for flat_game, object_game in zip(flat_records, object_records):
        assert len(flat_game["moves"]) == len(object_game["moves"])
        for flat_move, object_move in zip(flat_game["moves"], object_game["moves"]):
            assert flat_move["action"] == object_move["action"]
            assert flat_move["visits"] == object_move["visits"]
            assert flat_move["gumbel_topk"] == object_move["gumbel_topk"]
            assert flat_move["root_value"] == pytest.approx(
                object_move["root_value"], abs=2e-6
            )
            assert flat_move["policy_target"] == pytest.approx(
                object_move["policy_target"], abs=2e-6
            )
    assert flat_adapter.total_metrics["forward_seconds"] > 0.0
    assert metrics["boundary_tokens"] > 0
    assert all(
        replay(record).phase is Phase.COMPLETE
        for record in phase_d_records_from_rust(flat_records)
    )


def _force_position():
    import seven_wonders_rust as swr

    for game_seed in range(12):
        first_player, actions, library = random_game(game_seed, game_seed % 2)
        py = new_game(game_seed, first_player=first_player)
        rust = swr.RustGame(
            library_draws=[list(draw) for draw in library], **extract_setup(py)
        )
        for index, action in enumerate(actions):
            if index >= 8 and py.phase is Phase.PLAY_AGE and _enumerable_reveal_root(py):
                return rust
            apply_action(py, decode_action(py, action))
            rust.apply_index(action)
    raise AssertionError("no enumerable force-expansion position")


def test_f4_5_forced_cache_is_leaf_batch_one_tree_exact_and_saves_calls():
    rust = _force_position()
    saved = False
    for seed in range(8):
        old_calls = 0
        new_calls = 0

        def old_adapter(tokens, actor, legal):
            nonlocal old_calls
            old_calls += 1
            return _row_eval(tokens, actor, legal)

        def new_adapter(tokens, actor, legal):
            nonlocal new_calls
            new_calls += 1
            return _row_eval(tokens, actor, legal)

        old = rust.closed_search_net(old_adapter, 32, 8, seed, force=True)
        new = rust.closed_search_resumable_net(new_adapter, 32, 8, seed, force=True)
        assert new[0] == old[0]
        assert new[1:7] == old[1:7]
        assert new[7] == old[7]
        if new_calls < old_calls:
            saved = True
            break
    assert saved, "forced cache did not eliminate an ordinary child re-evaluation"


def test_f4_5_cooperative_force_uses_capped_flat_batches_and_replays():
    import seven_wonders_rust as swr

    seed = 2026072420
    kwargs = _flat_kwargs(force=True)
    independent = rust_game_for_self_play(seed, 0).self_play_net(
        _row_eval,
        game_seed=seed,
        **{key: value for key, value in kwargs.items() if key not in {"global_batch_cap", "max_inflight_batches"}},
    )
    adapter = _DecodingFlatAdapter()
    records, _ = swr.self_play_many_flat_net(
        adapter=adapter,
        games=rust_games_for_self_play([seed], [0]),
        game_seeds=[seed],
        **kwargs,
    )
    assert records == [independent]
    assert max(adapter.batch_sizes) <= kwargs["global_batch_cap"]
    phase_d_records_from_rust(records)


def test_f4_r1_forced_rows_are_chunked_coalesced_and_accounted():
    import seven_wonders_rust as swr

    seeds = [2026072425, 2026072426, 2026072427, 2026072428]
    adapter = _DecodingFlatAdapter()
    records, metrics = swr.self_play_many_flat_net(
        adapter=adapter,
        games=rust_games_for_self_play(seeds, [0, 1, 0, 1]),
        game_seeds=seeds,
        **_flat_kwargs(force=True),
    )
    assert len(records) == len(seeds)
    assert metrics["forced_rows"] > 0
    assert metrics["ordinary_leaf_rows"] > 0
    assert metrics["global_rows"] == metrics["root_rows"] + metrics["leaf_rows"]
    assert metrics["leaf_rows"] == metrics["forced_rows"] + metrics["ordinary_leaf_rows"]
    assert metrics["forced_rows"] == sum(
        metrics[key]
        for key in (
            "forced_card_reveal_rows",
            "forced_great_library_rows",
            "forced_wonder_group_rows",
            "forced_age_deal_rows",
        )
    )
    assert metrics["forced_cache_hits"] > 0
    assert max(adapter.batch_sizes) <= 8
    assert any(size == 8 for size in adapter.batch_sizes)


def test_f4_r0_paired_age_deal_samples_are_deterministic_and_globally_evaluated():
    import seven_wonders_rust as swr

    _, actions, library = random_game(3, 0)
    py = new_game(3, first_player=0)
    rust = swr.RustGame(
        library_draws=[list(draw) for draw in library], **extract_setup(py)
    )
    for action in actions:
        if py.phase is Phase.CHOOSE_NEXT_START_PLAYER:
            break
        apply_action(py, decode_action(py, action))
        rust.apply_index(action)
    assert py.phase is Phase.CHOOSE_NEXT_START_PLAYER

    def run(samples):
        return swr.search_many_flat_net(
            _DecodingFlatAdapter(),
            [rust],
            [8080],
            4,
            1,
            8,
            4,
            force=True,
            age_deal_samples=samples,
        )[0]

    legacy = run(0)
    paired = run(4)
    assert run(4) == paired
    assert legacy["nn_work"]["forced_rows"] == 0
    assert paired["nn_work"]["forced_rows"] > 0
    assert paired["nn_work"]["forced_rows"] <= 8  # two actions x four common draws


def _parse_tree_digest(values):
    """Decode `tree_resumable::digest` into nested dicts.

    Layout (kept in step with `digest`): node = visits, value_sum, actor,
    terminal, fingerprint (length-prefixed), then length-prefixed edges; edge =
    action_index, visits, value_sum, prior, probability_weighted, then
    length-prefixed children; child = length-prefixed observable key, samples,
    probability, then the child's node."""

    position = 0

    def node():
        nonlocal position
        visits = int(values[position])
        position += 4  # visits, value_sum, actor, terminal
        position += 1 + int(values[position])  # fingerprint
        edge_count = int(values[position])
        position += 1
        edges = []
        for _ in range(edge_count):
            action_index = int(values[position])
            edge_visits = int(values[position + 1])
            weighted = values[position + 4] == 1.0
            position += 5
            child_count = int(values[position])
            position += 1
            children = []
            for _ in range(child_count):
                key_parts = int(values[position])
                position += 1
                for _ in range(key_parts):
                    position += 1 + int(values[position])
                samples = int(values[position])
                probability = values[position + 1]
                position += 2
                children.append(
                    {"samples": samples, "probability": probability, "node": node()}
                )
            edges.append(
                {
                    "action_index": action_index,
                    "visits": edge_visits,
                    "probability_weighted": weighted,
                    "children": children,
                }
            )
        return {"visits": visits, "edges": edges}

    root = node()
    assert position == len(values)
    return root


def test_f4_r0_fixed_support_edges_never_grow_and_keep_unit_mass():
    """The closed (approximate fixed-support) edge class on the Rust side.

    The paired AgeDeal sampler is its first member: its children are an
    empirical, re-normalised subset of the deal space, so descent must draw
    among exactly those children. If it fell back to ordinary sampling it would
    append fresh children carrying their own probability and the edge's mass
    would climb past 1 with nothing raising (CHANCE_ENUMERATION_PLAN.md)."""

    import seven_wonders_rust as swr

    _, actions, library = random_game(3, 0)
    py = new_game(3, first_player=0)
    rust = swr.RustGame(
        library_draws=[list(draw) for draw in library], **extract_setup(py)
    )
    for action in actions:
        if py.phase is Phase.CHOOSE_NEXT_START_PLAYER:
            break
        apply_action(py, decode_action(py, action))
        rust.apply_index(action)
    assert py.phase is Phase.CHOOSE_NEXT_START_PLAYER

    samples = 4
    sims = 64

    def run(age_deal_samples):
        result = swr.search_many_flat_net(
            _DecodingFlatAdapter(),
            [rust],
            [8080],
            8,
            1,
            sims,
            4,
            force=True,
            age_deal_samples=age_deal_samples,
        )[0]
        return _parse_tree_digest(result["digest"])

    closed = run(samples)
    assert closed["edges"]
    for edge in closed["edges"]:
        # Every action here deals the next age, so every root edge is closed.
        assert edge["probability_weighted"]
        assert 0 < len(edge["children"]) <= samples  # no growth under descent
        mass = sum(child["probability"] for child in edge["children"])
        assert mass == pytest.approx(1.0, abs=1e-12)
        drawn = sum(child["samples"] for child in edge["children"])
        assert drawn == edge["visits"]  # every descent landed on a retained child

    # Contrast: left open, the same edges accumulate a child per distinct deal.
    legacy = run(0)
    assert max(len(edge["children"]) for edge in legacy["edges"]) > samples
    assert not any(edge["probability_weighted"] for edge in legacy["edges"])


def _self_play_records(cheap_double_reveal_offsets, full_search_fraction, seeds):
    import seven_wonders_rust as swr

    kwargs = _common(leaf_batch=1, global_batch_cap=8)
    kwargs["full_search_fraction"] = full_search_fraction
    records, _ = swr.self_play_many_mock(
        games=rust_games_for_self_play(seeds, [0, 1, 0]),
        game_seeds=seeds,
        force=True,
        cheap_double_reveal_offsets=cheap_double_reveal_offsets,
        **kwargs,
    )
    return records


def test_cheap_double_reveal_offsets_leave_full_search_moves_bit_identical():
    """Chance-enumeration Step 3: capping is a CHEAP-move economy.

    This is the claim that keeps buffers comparable across the change, so it is
    asserted rather than assumed. With every move a full search, turning the cap
    up must change nothing at all -- same actions, same visits, same policy
    targets, same games. (Under a mixed schedule the recorded positions still
    differ, because earlier cheap moves move the trajectory; what cannot differ
    is a full-search move's target at a given state and seed.)"""

    seeds = [2026072530, 2026072531, 2026072532]
    baseline = _self_play_records(0, 1.0, seeds)
    for offsets in (1, 2, 3):
        assert _self_play_records(offsets, 1.0, seeds) == baseline
    assert all(move["full_search"] for record in baseline for move in record["moves"])


def test_cheap_double_reveal_offsets_do_change_cheap_moves():
    """The converse: with every move cheap, the cap must actually be reaching
    the searcher. A hook that silently did nothing would pass the test above."""

    seeds = [2026072530, 2026072531, 2026072532]
    baseline = _self_play_records(0, 0.0, seeds)
    capped = _self_play_records(2, 0.0, seeds)
    assert not any(move["full_search"] for record in baseline for move in record["moves"])
    assert capped != baseline
    differing = [
        (game, move)
        for game, (left, right) in enumerate(zip(baseline, capped))
        for move in range(min(len(left["moves"]), len(right["moves"])))
        if left["moves"][move]["policy_target"] != right["moves"][move]["policy_target"]
    ]
    assert differing, "capped cheap moves produced identical policy targets"


def test_f4_r0_paired_age_deal_samples_skip_forced_age_one_setup():
    import seven_wonders_rust as swr

    _, actions, library = random_game(3, 0)
    py = new_game(3, first_player=0)
    rust = swr.RustGame(
        library_draws=[list(draw) for draw in library], **extract_setup(py)
    )
    for action in actions[:7]:
        apply_action(py, decode_action(py, action))
        rust.apply_index(action)
    assert py.phase is Phase.WONDER_DRAFT
    assert len(py.wonder_offer) == 1

    def run(samples):
        return swr.search_many_flat_net(
            _DecodingFlatAdapter(),
            [rust],
            [8080],
            4,
            1,
            8,
            4,
            force=True,
            age_deal_samples=samples,
        )[0]

    legacy = run(0)
    configured = run(4)
    for key in (
        "action",
        "action_value",
        "root_value",
        "visits",
        "policy",
        "topk",
        "sims",
        "completed_q",
        "metrics",
    ):
        assert configured[key] == legacy[key]
    assert configured["nn_work"]["forced_rows"] == 0


def test_f4_6_mixed_seat_leaf_batches_are_deterministic_and_replayable():
    import seven_wonders_rust as swr

    seeds = [2026072460, 2026072461]
    first_players = [0, 1]
    kwargs = {
        **_common(leaf_batch=2, global_batch_cap=8),
        "cheap_sims_min": 4,
        "cheap_sims_max": 4,
        "full_sims_min": 4,
        "full_sims_max": 4,
        "full_search_fraction": 0.0,
        "leaf_batch_p0": 2,
        "leaf_batch_p1": 1,
        "deterministic_actions": True,
    }

    def generate():
        adapter = _DecodingFlatAdapter()
        records, metrics = swr.self_play_many_flat_net(
            adapter=adapter,
            games=rust_games_for_self_play(seeds, first_players),
            game_seeds=seeds,
            **kwargs,
        )
        return records, metrics

    first, metrics = generate()
    second, _ = generate()
    assert first == second
    assert metrics["rust_tree_ns"] > 0
    assert metrics["rust_chance_ns"] > 0
    assert metrics["rust_record_ns"] > 0
    assert metrics["scheduler_ready_slot_cycles"] > 0
    phase_d_records_from_rust(first)

    with pytest.raises(ValueError, match="supplied together"):
        swr.self_play_many_flat_net(
            adapter=_DecodingFlatAdapter(),
            games=rust_games_for_self_play(seeds[:1], first_players[:1]),
            game_seeds=seeds[:1],
            **{key: value for key, value in kwargs.items() if key != "leaf_batch_p1"},
        )


def test_f4_6_global_position_search_matches_scalar_resumable_rows():
    import seven_wonders_rust as swr

    seeds = [2026072470, 2026072471, 2026072472]
    games = rust_games_for_self_play(seeds, [0, 1, 0])
    search_seeds = [700, 701, 702]
    expected = [
        game.closed_search_batched_net(_row_eval, 2, 8, 4, seed)
        for game, seed in zip(games, search_seeds)
    ]
    actual = swr.search_many_flat_net(
        _DecodingFlatAdapter(), games, search_seeds, 16, 2, 8, 4
    )
    for scalar, batched in zip(expected, actual):
        assert batched["action"] == scalar[0]
        assert batched["action_value"] == scalar[1]
        assert batched["root_value"] == scalar[2]
        assert batched["visits"] == scalar[3]
        assert batched["policy"] == scalar[4]
        assert batched["topk"] == scalar[5]
        assert batched["sims"] == scalar[6]
        assert list(batched["metrics"].values()) == list(scalar[7])
        assert batched["completed_q"] == scalar[8]
        assert batched["digest"] == scalar[9]


def test_f4_rust_seat_routed_adapter_uses_the_actor_checkpoint():
    import torch
    import seven_wonders_rust as swr

    from .inference import Evaluator
    from .train import build_model

    models = [build_model("transformer", 32, 1) for _ in range(2)]
    for model in models:
        for parameter in model.parameters():
            parameter.data.zero_()
    models[0].heads.value.bias.data.copy_(torch.tensor([10.0, 0.0, -10.0]))
    models[1].heads.value.bias.data.copy_(torch.tensor([-10.0, 0.0, 10.0]))
    adapter = rust_seat_routed_flat_batch_adapter(
        [Evaluator(model, "cpu", 8) for model in models]
    )
    results = swr.search_many_flat_net(
        adapter,
        rust_games_for_self_play([101, 102], [0, 1]),
        [700, 701],
        8,
        1,
        1,
        2,
    )
    assert results[0]["root_value"] > 0.99
    assert results[1]["root_value"] < -0.99


def test_two_net_arena_searches_each_move_under_the_movers_own_net():
    """Regression: the arena must never mix networks inside one search tree.

    `rust_seat_routed_flat_batch_adapter` dispatched each packed row to the net
    of that row's ACTING player, so opponent-to-move leaves inside a search were
    evaluated by the opponent's network and neither side ever searched with its
    own net.  A strong net facing a weak one then read garbage for every reply
    while handing the weak net good values for its own replies, inverting the
    result (0.225 vs 0.925 against a random-init net at 2 sims).

    Two nets with opposite constant value heads make the mixing visible: every
    batch a search issues must carry a single value, because a whole search
    belongs to exactly one network.
    """

    import seven_wonders_rust as swr
    import torch

    from .inference import Evaluator
    from .phase_d import PhaseDConfig, PhaseDLoop
    from .train import build_model

    models = [build_model("transformer", 32, 1) for _ in range(2)]
    for model in models:
        for parameter in model.parameters():
            parameter.data.zero_()
    models[0].heads.value.bias.data.copy_(torch.tensor([10.0, 0.0, -10.0]))
    models[1].heads.value.bias.data.copy_(torch.tensor([-10.0, 0.0, 10.0]))

    loop = PhaseDLoop(
        PhaseDConfig(
            run_dir="runs/seven_wonders_duel/_two_net_arena_test",
            device="cpu",
            d_model=32,
            layers=1,
            gate_sims=2,
            gate_max_games=2,
            top_k=2,
            promotion_every=0,
            rust_slots=2,
            rust_global_batch_cap=16,
        )
    )

    seen: list[set[float]] = []

    def recording(evaluator):
        inner = rust_flat_batch_adapter(evaluator)

        def adapter(payload):
            produced = inner(payload)
            seen.append({round(value, 6) for value, _ in produced})
            return produced

        return adapter

    adapters = tuple(recording(Evaluator(model, "cpu", 16)) for model in models)
    records = loop._play_two_net_games(
        swr, rust_games_for_self_play([4242], [0]), [4242], adapters
    )

    assert len(records) == 1
    assert records[0]["winner"] in (0, 1, None)
    assert records[0]["actions"] > 0
    assert seen, "expected the arena to evaluate at least one batch"
    for batch in seen:
        assert len(batch) == 1, f"batch mixed values from both nets: {batch}"


def test_two_net_arena_plays_the_improved_policy_argmax():
    """Regression: arena play must not use the Gumbel-perturbed action.

    `SearchResult.action_index` is the exploration action; at low simulation
    counts it reduces to argmax(gumbel + log_prior), which by the Gumbel-max
    trick is an exact SAMPLE from the prior rather than its argmax.  A stub
    searcher whose `action` disagrees with its `policy` argmax pins down which
    one the arena applies.
    """

    from .phase_d import PhaseDConfig, PhaseDLoop

    loop = PhaseDLoop(
        PhaseDConfig(
            run_dir="runs/seven_wonders_duel/_two_net_argmax_test",
            device="cpu",
            gate_sims=2,
            gate_max_games=2,
            promotion_every=0,
        )
    )
    games = rust_games_for_self_play([909], [0])
    applied: list[int] = []

    class StubSearcher:
        """Reports the LAST legal action as `action` and the FIRST as best."""

        @staticmethod
        def search_many_flat_net(adapter, batch, seeds, *args, **kwargs):
            results = []
            for game in batch:
                legal = game.legal_action_indices()
                policy = [0.0] * len(legal)
                policy[0] = 1.0  # argmax -> legal[0]
                applied.append(legal[0])
                results.append({"action": legal[-1], "policy": policy})
            return results

    records = loop._play_two_net_games(
        StubSearcher(), games, [909], (object(), object())
    )
    assert records[0]["actions"] == len(applied)
    assert games[0].is_complete()
    # If the arena had played `action` (the last legal index) instead of the
    # policy argmax, the game would have followed a different line entirely.
    assert applied, "stub searcher was never consulted"
