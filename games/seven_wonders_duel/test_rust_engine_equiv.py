"""Phase F1 equivalence gates for the Rust 7WD engine (`seven_wonders_rust`).

Method is the Kingdomino `test_rust_*_equiv` discipline (AZ_PROJECT_PLAN §2b,
PHASE_F.md): drive the Rust engine and the Python reference from the same action
sequence and assert bit-for-bit agreement of a language-neutral fingerprint at
*every* decision, not just the end.

- **F1a** — replay agreement: at each move the Rust fingerprint and legal-action
  mask equal Python's; the final state matches. Corpus = real buffer games (all
  phases, all effects) plus random-policy games for breadth.
- **F1b** — make/unmake: `RustGame.roundtrip_ok(index)` (apply then undo)
  restores the fingerprint at every decision.

`logic_fingerprint` mirrors `state.rs::GameState::fingerprint` exactly: the same
field order, numeric-id sorts, and length prefixes, and it excludes Python's RNG
internal state (Rust models no RNG — see PHASE_F.md). Both sides map component
names to ids through the identical `data.py` tables, so id agreement is implied.
"""

from __future__ import annotations

import math
import os
import random

from .buffer import from_json_line
from .codec import decode_action, legal_action_indices
from .data import BackType, CARD_IDS, PROGRESS_IDS, WONDER_IDS, ScienceSymbol
from .encoder import Encoding, Token, encode as py_encode, TokenType
from .engine import apply_action
from .game import ChanceKind
from .pool import unseen_pool as py_unseen_pool
from .portable_rng import PortableRng
from .search import (
    GumbelMCTS,
    SearchConfig,
    chance_signature as py_chance_signature,
    enumerate_chains as py_enumerate_chains,
    sample_outcomes as py_sample_outcomes,
)

_CHANCE_KIND_ID = {
    ChanceKind.CARD_REVEAL: 0,
    ChanceKind.GREAT_LIBRARY_DRAW: 1,
    ChanceKind.WONDER_GROUP_REVEAL: 2,
    ChanceKind.AGE_DEAL: 3,
}
_BACK_ID = {
    BackType.AGE_I: 0,
    BackType.AGE_II: 1,
    BackType.AGE_III: 2,
    BackType.GUILD: 3,
}
_CHANCE_ID_MAP = {
    ChanceKind.CARD_REVEAL: CARD_IDS,
    ChanceKind.GREAT_LIBRARY_DRAW: PROGRESS_IDS,
    ChanceKind.WONDER_GROUP_REVEAL: WONDER_IDS,
}
# Sampling additionally maps AGE_DEAL outcomes (card names) to card ids.
_CHANCE_SAMPLE_ID_MAP = {**_CHANCE_ID_MAP, ChanceKind.AGE_DEAL: CARD_IDS}
_NUM_CARDS = len(CARD_IDS)


def _map_key(specs, py_key):
    """Python observable key -> Rust's Vec<Vec<i32>> encoding. AGE_DEAL keys mix
    face-up card ids with `NUM_CARDS + back_id` markers for face-down slots."""

    out = []
    for spec, part in zip(specs, py_key):
        if spec.kind is ChanceKind.AGE_DEAL:
            # BackType is a str subclass, so test it before the card-name case.
            out.append(
                [
                    _NUM_CARDS + _BACK_ID[e] if isinstance(e, BackType) else CARD_IDS[e]
                    for e in part
                ]
            )
        elif spec.kind is ChanceKind.CARD_REVEAL:
            out.append([CARD_IDS[part]])  # part is a single card name
        else:  # GreatLibrary / wonder flip: part is a tuple of names
            out.append([_CHANCE_ID_MAP[spec.kind][name] for name in part])
    return out


_MASK64 = (1 << 64) - 1


def _mock_mix(h):
    h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & _MASK64
    return h ^ (h >> 31)


def _mock_fold(fp):
    h = 0x9E3779B97F4A7C15
    for x in fp:
        h = _mock_mix((h ^ (x & _MASK64)) & _MASK64)
    return h


def _mock_unit(h):
    return (h >> 11) / (1 << 53)


def _mock_terminal_value(game):
    if game.winner is None:
        return 0.0
    return 1.0 if game.winner == 0 else -1.0


def mock_eval(game):
    """Python reference for the deterministic fingerprint-based leaf oracle
    (mirrors eval.rs::MockEval): (value_p0, priors aligned to legal indices)."""

    h = _mock_fold(logic_fingerprint(game))
    value = _mock_unit(h) * 2 - 1
    if game.phase is Phase.COMPLETE:
        return _mock_terminal_value(game), []
    legal = legal_action_indices(game)
    # Raw weights (unnormalized) — see eval.rs::MockEval for why.
    return value, [
        _mock_unit(_mock_mix(h ^ ((a * 0x9E3779B97F4A7C15) & _MASK64))) for a in legal
    ]


def test_mock_eval_matches_python():
    """F3.2 foundation: the Rust MockEval oracle equals the Python reference
    (value + aligned priors) at every state incl. terminal, bit-for-bit."""

    for seed in range(20):
        first_player, actions, library = random_game(seed, seed % 2)
        py = new_game(seed, first_player=first_player)
        rg = swr.RustGame(library_draws=[list(d) for d in library], **extract_setup(py))
        for idx in actions + [None]:
            exp_v, exp_priors = mock_eval(py)
            rust_v, rust_priors = rg.mock_eval()
            assert rust_v == exp_v, f"seed {seed}: mock value {rust_v} != {exp_v}"
            assert list(rust_priors) == exp_priors, f"seed {seed}: mock priors differ"
            if idx is None:
                break
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)


def _mock_evaluate(state):
    """(value_p0, priors dict) mock for a GumbelMCTS subclass."""

    value, weights = mock_eval(state)
    if state.phase is Phase.COMPLETE:
        return value, {}
    legal = legal_action_indices(state)
    return value, {a: w for a, w in zip(legal, weights)}


def _closed_tree_ref(state, sims, seed):
    """Python reference closed tree with the mock oracle + fixed round-robin root
    schedule (mirrors tree.rs::closed_tree_fixed), reusing the real searcher."""

    mcts = GumbelMCTS(None, SearchConfig(mode="closed", seed=seed, c_puct=1.5))
    mcts._evaluate = _mock_evaluate  # type: ignore[method-assign]
    root_state = state.clone()
    root_state.search_barrier = True
    root = mcts._make_closed_node(root_state)
    root.visits += 1
    root.value_sum_p0 += mcts._expand_closed(root)
    n = max(len(root.edges), 1)
    for i in range(sims):
        mcts._descend_closed(root, forced_edge=root.edges[i % n])
    return root


def _digest_ref(node, out):
    out.append(float(node.visits))
    out.append(node.value_sum_p0)
    out.append(float(node.actor))
    out.append(1.0 if node.terminal else 0.0)
    fp = logic_fingerprint(node.state)
    out.append(float(len(fp)))
    out.extend(float(x) for x in fp)
    out.append(float(len(node.edges)))
    for edge in node.edges:
        out.append(float(edge.action_index))
        out.append(float(edge.visits))
        out.append(edge.value_sum_p0)
        out.append(edge.prior)
        out.append(1.0 if edge.probability_weighted else 0.0)
        out.append(float(len(edge.children)))
        for key, child in edge.children.items():
            mapped = _map_key(edge.specs, key)
            out.append(float(len(mapped)))  # number of parts
            for part in mapped:
                out.append(float(len(part)))  # length of this part
                out.extend(float(k) for k in part)
            out.append(float(child.samples))
            out.append(float("nan") if child.probability is None else child.probability)
            _digest_ref(child.node, out)


def _assert_digest_equal(expected, got, ctx):
    assert len(got) == len(expected), f"{ctx}: digest length {len(got)} != {len(expected)}"
    for i, (e, g) in enumerate(zip(expected, got)):
        if e != e and g != g:  # both NaN
            continue
        assert g == pytest.approx(e, rel=0, abs=1e-9), f"{ctx}: digest[{i}] {g} != {e}"


def test_closed_tree_matches_python():
    """F3.2: the Rust closed tree (nodes/edges/children, PUCT descent,
    outcome-keyed materialization) is bit-identical to the Python reference under
    the mock oracle — full DFS digest, across sims and RNG seeds."""

    checked = 0
    for game_seed in range(8):
        first_player, actions, library = random_game(game_seed, game_seed % 2)
        py = new_game(game_seed, first_player=first_player)
        rg = swr.RustGame(library_draws=[list(d) for d in library], **extract_setup(py))
        tested_here = False
        for i, idx in enumerate(actions):
            if (
                not tested_here
                and i >= 6
                and py.phase is Phase.PLAY_AGE
                and py.pending_choice is None
            ):
                for sims in (16, 48):
                    for seed in (1, 7):
                        expected = []
                        _digest_ref(_closed_tree_ref(py, sims, seed), expected)
                        got = list(rg.closed_tree_digest(sims, seed))
                        _assert_digest_equal(
                            expected, got, f"game {game_seed} move {i} sims {sims} seed {seed}"
                        )
                        checked += 1
                tested_here = True
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)
    assert checked > 20


def _mock_search(state, sims, top_k, seed, force):
    mcts = GumbelMCTS(
        None,
        SearchConfig(
            sims=sims,
            top_k=top_k,
            mode="closed",
            seed=seed,
            force_expand_root_chance=force,
        ),
    )
    mcts._evaluate = _mock_evaluate  # type: ignore[method-assign]
    result = mcts.search(state)
    return result, mcts._closed_root


def test_closed_search_matches_python():
    """F3.3: the full closed search (Gumbel root + sequential halving + policy
    target, with force-expansion off AND on) matches the Python reference —
    chosen action, visits, top-k, values, policy target, and the whole tree."""

    checked = 0
    for game_seed in range(6):
        first_player, actions, library = random_game(game_seed, game_seed % 2)
        py = new_game(game_seed, first_player=first_player)
        rg = swr.RustGame(library_draws=[list(d) for d in library], **extract_setup(py))
        tested_here = False
        for i, idx in enumerate(actions):
            if (
                not tested_here
                and i >= 8
                and py.phase is Phase.PLAY_AGE
                and py.pending_choice is None
            ):
                legal = legal_action_indices(py)
                for sims in (16, 64):
                    for seed in (1, 5):
                        for force in (False, True):
                            result, root = _mock_search(py, sims, 8, seed, force)
                            (act, av, rv, visits, policy, topk, rsims, dig) = rg.closed_search(
                                sims, 8, seed, force=force
                            )
                            ctx = f"game {game_seed} sims {sims} seed {seed} force {force}"
                            assert act == result.action_index, f"{ctx}: action"
                            assert rsims == result.sims, f"{ctx}: sims"
                            assert list(topk) == list(result.gumbel_topk), f"{ctx}: topk"
                            assert list(visits) == [result.visits[a] for a in legal], (
                                f"{ctx}: visits"
                            )
                            assert av == pytest.approx(result.action_value, abs=1e-9), f"{ctx}: av"
                            assert rv == pytest.approx(result.root_value, abs=1e-9), f"{ctx}: rv"
                            for j, a in enumerate(legal):
                                assert policy[j] == pytest.approx(
                                    result.policy_target[a], abs=1e-9
                                ), f"{ctx}: policy[{a}]"
                            exp_dig = []
                            _digest_ref(root, exp_dig)
                            _assert_digest_equal(exp_dig, list(dig), f"{ctx}: tree")
                            checked += 1
                tested_here = True
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)
    assert checked >= 20


def test_resumable_leaf_batch_one_matches_f3_oracle():
    """F4.1 exact refactor gate: the arena-backed selection/eval/backprop state
    machine at leaf_batch=1 reproduces the permanent F3.3 sequential Rust oracle,
    including canonical topology and all discrete/floating outputs."""

    checked = 0
    for game_seed in range(6):
        first_player, actions, library = random_game(game_seed, game_seed % 2)
        py = new_game(game_seed, first_player=first_player)
        rg = swr.RustGame(library_draws=[list(d) for d in library], **extract_setup(py))
        tested_here = False
        for i, idx in enumerate(actions):
            if (
                not tested_here
                and i >= 8
                and py.phase is Phase.PLAY_AGE
                and py.pending_choice is None
            ):
                for sims in (16, 64):
                    for seed in (1, 5):
                        for force in (False, True):
                            expected = rg.closed_search(sims, 8, seed, force=force)
                            got = rg.closed_search_resumable(sims, 8, seed, force=force)
                            ctx = (
                                f"game {game_seed} move {i} sims {sims} "
                                f"seed {seed} force {force}"
                            )
                            assert got[0] == expected[0], f"{ctx}: action"
                            assert got[3] == expected[3], f"{ctx}: visits"
                            assert got[5] == expected[5], f"{ctx}: top-k"
                            assert got[6] == expected[6], f"{ctx}: sims"
                            assert got[1] == pytest.approx(expected[1], abs=1e-9), f"{ctx}: av"
                            assert got[2] == pytest.approx(expected[2], abs=1e-9), f"{ctx}: rv"
                            assert list(got[4]) == pytest.approx(
                                list(expected[4]), rel=0, abs=1e-9
                            ), f"{ctx}: policy"
                            _assert_digest_equal(list(expected[7]), list(got[7]), f"{ctx}: tree")
                            checked += 1
                tested_here = True
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)

    # Dedicated chance roots cover the two root-specific structures that a
    # random play-age sample may miss: enumerable forced children and AGE_DEAL.
    for seed, picks, force in ((0, 3, True), (1, 7, False)):
        _py, rg = _drive_to_draft(seed, picks)
        expected = rg.closed_search(32, 8, 3, force=force)
        got = rg.closed_search_resumable(32, 8, 3, force=force)
        ctx = f"draft seed {seed} picks {picks} force {force}"
        assert got[0] == expected[0], f"{ctx}: action"
        assert got[3] == expected[3], f"{ctx}: visits"
        assert got[5] == expected[5], f"{ctx}: top-k"
        assert got[6] == expected[6], f"{ctx}: sims"
        assert got[1] == pytest.approx(expected[1], rel=0, abs=1e-9), f"{ctx}: av"
        assert got[2] == pytest.approx(expected[2], rel=0, abs=1e-9), f"{ctx}: rv"
        assert list(got[4]) == pytest.approx(list(expected[4]), rel=0, abs=1e-9)
        _assert_digest_equal(list(expected[7]), list(got[7]), f"{ctx}: tree")
        checked += 1
    assert checked >= 20


def test_resumable_leaf_batch_one_phase_strata():
    """F4.1 coverage hardening across every searchable game phase, pending
    choices, and a late Age-III root whose recorded action reaches a terminal
    leaf. Configs deliberately vary non-power-of-two budgets/top-k/seeds."""

    wanted = {
        "draft": 2,
        "age_1": 2,
        "age_2": 2,
        "age_3": 2,
        "between_ages": 2,
        "pending_choice": 2,
        "terminal_leaf_root": 2,
    }
    covered = {name: 0 for name in wanted}

    def compare(rg, stratum, game_seed, move):
        ordinal = sum(covered.values())
        sims = (7, 13, 23)[ordinal % 3]
        top_k = (3, 5, 8)[ordinal % 3]
        seed = 1009 + game_seed * 101 + move
        force = ordinal % 4 == 0
        expected = rg.closed_search(sims, top_k, seed, force=force)
        got = rg.closed_search_resumable(sims, top_k, seed, force=force)
        ctx = (
            f"{stratum} game {game_seed} move {move} sims {sims} "
            f"top_k {top_k} seed {seed} force {force}"
        )
        assert got[0] == expected[0], f"{ctx}: action"
        assert got[3] == expected[3], f"{ctx}: visits"
        assert got[5] == expected[5], f"{ctx}: top-k"
        assert got[6] == expected[6], f"{ctx}: sims"
        assert got[1] == pytest.approx(expected[1], rel=0, abs=1e-9), f"{ctx}: av"
        assert got[2] == pytest.approx(expected[2], rel=0, abs=1e-9), f"{ctx}: rv"
        assert list(got[4]) == pytest.approx(list(expected[4]), rel=0, abs=1e-9)
        _assert_digest_equal(list(expected[7]), list(got[7]), f"{ctx}: tree")
        covered[stratum] += 1

    for game_seed in range(60):
        if all(covered[name] >= count for name, count in wanted.items()):
            break
        first_player, actions, library = random_game(game_seed, game_seed % 2)
        py = new_game(game_seed, first_player=first_player)
        rg = swr.RustGame(library_draws=[list(d) for d in library], **extract_setup(py))
        for move, idx in enumerate(actions):
            strata = []
            if py.pending_choice is not None:
                strata.append("pending_choice")
            elif py.phase is Phase.WONDER_DRAFT:
                strata.append("draft")
            elif py.phase is Phase.CHOOSE_NEXT_START_PLAYER:
                strata.append("between_ages")
            elif py.phase is Phase.PLAY_AGE:
                strata.append(f"age_{py.age}")

            # This recorded legal action is known to finish the game. With a
            # top-k covering all late-root actions, at least one scheduled
            # simulation therefore exercises the no-evaluator terminal path.
            probe = py.clone()
            apply_action(probe, decode_action(probe, idx))
            if probe.phase is Phase.COMPLETE:
                strata.append("terminal_leaf_root")

            for stratum in strata:
                if stratum in wanted and covered[stratum] < wanted[stratum]:
                    compare(rg, stratum, game_seed, move)
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)

    assert covered == wanted


def test_wu_leaf_waves_accounting_and_cleanup():
    """F4.2 WU waves schedule every simulation exactly once, deduplicate only
    NN rows (never backups), drain before reductions, and leave no incomplete
    counts at successful return. The Rust session enforces the latter two as
    runtime invariants; these assertions gate the externally visible accounting."""

    collision_seen = False
    shortened_wave_seen = False
    for game_seed in range(4):
        _py, rg = _play_age_position(game_seed)
        for leaf_batch in (2, 4, 8):
            result = rg.closed_search_batched(
                leaf_batch, 67, 8, 700 + game_seed, force=game_seed % 2 == 0
            )
            _act, _av, _rv, visits, policy, topk, sims, metrics, root_q, _digest = result
            (
                scheduled,
                requested,
                unique,
                terminal,
                collisions,
                waves,
                max_paths,
                max_unique,
            ) = metrics
            ctx = f"game {game_seed} leaf_batch {leaf_batch}"
            assert sims == scheduled == 67, ctx
            assert sum(visits) == 67, ctx
            assert requested + terminal == 67, ctx
            assert unique + collisions == requested, ctx
            assert 1 <= waves <= requested, ctx
            assert 1 <= max_paths <= leaf_batch, ctx
            assert 1 <= max_unique <= max_paths, ctx
            assert len(policy) == len(visits), ctx
            assert len(root_q) == len(visits), ctx
            assert len(topk) <= 8, ctx
            collision_seen |= collisions > 0
            shortened_wave_seen |= max_paths < leaf_batch or requested % leaf_batch != 0
    assert collision_seen, "WU corpus produced no duplicate pending leaf"
    assert shortened_wave_seen, "no wave was shortened at a round/budget boundary"


def test_wu_leaf_batch_one_surface_is_exact_and_rejects_zero():
    """The general F4.2 surface must retain the F4.1 exact bypass at batch one
    and reject a zero-sized wave rather than spin or silently degrade."""

    _py, rg = _play_age_position(2)
    expected = rg.closed_search(31, 5, 91, force=True)
    got = rg.closed_search_batched(1, 31, 5, 91, force=True)
    assert got[:7] == expected[:7]
    scheduled, requested, unique, terminal, collisions, waves, max_paths, max_unique = got[7]
    assert scheduled == 31
    assert requested + terminal == 31
    assert requested == unique
    assert collisions == 0
    assert waves == requested
    assert max_paths == max_unique == 1
    assert len(got[8]) == len(got[3])
    _assert_digest_equal(list(expected[7]), list(got[9]), "F4.2 leaf_batch=1 tree")
    with pytest.raises(ValueError):
        rg.closed_search_batched(0, 31, 5, 91)


def test_wu_terminal_leaves_need_no_nn_rows():
    """A late root with terminal descendants accounts those simulations but
    does not send them through the evaluator batch."""

    found = False
    for game_seed in range(20):
        first_player, actions, library = random_game(game_seed, game_seed % 2)
        py = new_game(game_seed, first_player=first_player)
        rg = swr.RustGame(library_draws=[list(d) for d in library], **extract_setup(py))
        for idx in actions:
            probe = py.clone()
            apply_action(probe, decode_action(probe, idx))
            if probe.phase is Phase.COMPLETE:
                result = rg.closed_search_batched(8, 16, 16, 31337 + game_seed)
                metrics = result[7]
                assert metrics[0] == 16
                assert metrics[3] > 0, "known terminal root produced no terminal leaves"
                assert metrics[1] + metrics[3] == 16
                found = True
                break
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)
        if found:
            break
    assert found, "failed to harvest a late terminal-reaching root"


def test_wu_evaluator_error_clears_pending_and_preserves_original_error():
    """An evaluator error during a multi-leaf wave surfaces unchanged. A fresh
    search on the same RustGame then succeeds, guarding against leaked session
    state or stranded WU counters at the PyO3 boundary."""

    _py, rg = _play_age_position(0)
    calls = 0

    def flaky(tokens, actor, legal):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("F4.2 evaluator sentinel")
        return 0.0, [1.0] * len(legal)

    with pytest.raises(RuntimeError, match="F4.2 evaluator sentinel"):
        rg.closed_search_batched_net(flaky, 8, 32, 8, 17)

    shape_calls = 0

    def bad_shape(tokens, actor, legal):
        nonlocal shape_calls
        shape_calls += 1
        size = len(legal) + (1 if shape_calls == 3 else 0)
        return 0.0, [1.0] * size

    with pytest.raises(ValueError, match="priors"):
        rg.closed_search_batched_net(bad_shape, 8, 32, 8, 17)
    good = lambda tokens, actor, legal: (0.0, [1.0] * len(legal))
    result = rg.closed_search_batched_net(good, 8, 32, 8, 17)
    assert result[6] == result[7][0] == 32


def _tree_stats(node, stats=None):
    if stats is None:
        stats = {"weighted": 0, "weighted_multi": 0, "none_prob": 0, "age_deal": 0}
    for edge in node.edges:
        if edge.probability_weighted:
            stats["weighted"] += 1
            if len(edge.children) > 1:
                stats["weighted_multi"] += 1
        if any(s.kind is ChanceKind.AGE_DEAL for s in edge.specs):
            stats["age_deal"] += 1
        for child in edge.children.values():
            if child.probability is None:
                stats["none_prob"] += 1
            _tree_stats(child.node, stats)
    return stats


def _drive_to_draft(seed, n_picks):
    """Both engines driven `n_picks` draft picks (legal[0] each). At n_picks==3
    the next pick fires WONDER_GROUP_REVEAL (enumerable); at 7 it fires
    AGE_DEAL(1) (sample-only). Draft picks 0-2 fire no chance, and the wonder
    flip at pick 3 resolves from locked state on both sides."""

    py = new_game(seed, first_player=seed % 2)
    rg = swr.RustGame(library_draws=[], **extract_setup(py))
    for _ in range(n_picks):
        idx = legal_action_indices(py)[0]
        apply_action(py, decode_action(py, idx))
        rg.apply_index(idx)
    return py, rg


def test_closed_search_force_expansion_coverage():
    """F3.3: force-expansion actually engages — a WONDER_GROUP_REVEAL root edge
    is materialized as a probability-weighted edge with many children, and the
    tree stays bit-identical to Python."""

    py, rg = _drive_to_draft(0, 3)
    specs = py_chance_signature(py, decode_action(py, legal_action_indices(py)[0]))
    assert any(s.kind is ChanceKind.WONDER_GROUP_REVEAL for s in specs)
    result, root = _mock_search(py, 32, 8, 3, True)
    stats = _tree_stats(root)
    assert stats["weighted"] >= 1, "force-expansion produced no weighted edge"
    assert stats["weighted_multi"] >= 1, "no weighted edge with multiple children"
    act, _av, _rv, _v, _p, _tk, _s, dig = rg.closed_search(32, 8, 3, force=True)
    expected = []
    _digest_ref(root, expected)
    _assert_digest_equal(expected, list(dig), "force-expansion tree")
    assert act == result.action_index


def test_closed_search_age_deal_coverage():
    """F3.3: AGE_DEAL edges materialize sample-only (probability=None) children,
    coalesced by observable deal key, bit-identical to Python."""

    py, rg = _drive_to_draft(1, 7)
    specs = py_chance_signature(py, decode_action(py, legal_action_indices(py)[0]))
    assert any(s.kind is ChanceKind.AGE_DEAL for s in specs)
    result, root = _mock_search(py, 32, 8, 2, False)
    stats = _tree_stats(root)
    assert stats["age_deal"] >= 1, "no AGE_DEAL edge exercised"
    assert stats["none_prob"] >= 1, "no None-probability (sample-only) child"
    act, _av, _rv, _v, _p, _tk, _s, dig = rg.closed_search(32, 8, 2, force=False)
    expected = []
    _digest_ref(root, expected)
    _assert_digest_equal(expected, list(dig), "age-deal tree")
    assert act == result.action_index


def test_closed_search_rejects_bad_config():
    """F3.3: the search enforces the Python config contract (sims/top_k > 0, a
    searchable root) rather than silently degrading."""

    py = new_game(0, first_player=0)
    rg = swr.RustGame(library_draws=[], **extract_setup(py))
    # Advance out of the draft so there is a normal searchable root.
    for _ in range(9):
        idx = legal_action_indices(py)[0]
        apply_action(py, decode_action(py, idx))
        rg.apply_index(idx)
    with pytest.raises(ValueError):
        rg.closed_search(0, 8, 1)
    with pytest.raises(ValueError):
        rg.closed_search(16, 0, 1)
    # Terminal root: drive a full random game and search at the end.
    first_player, actions, library = random_game(0, 0)
    end = swr.RustGame(library_draws=[list(d) for d in library], **extract_setup(new_game(0, first_player=first_player)))
    for idx in actions:
        end.apply_index(idx)
    assert end.is_complete()
    with pytest.raises(ValueError):
        end.closed_search(16, 8, 1)


def _make_net_adapter(evaluator):
    """Python `(tokens, actor, legal) -> (value_actor, priors)` adapter that the
    Rust PyEval calls: rebuild an Encoding from Rust tokens, run the net, and
    return the same value/priors Python's `_evaluate` computes."""

    token_types = list(TokenType)

    def adapter(tokens, actor, legal):
        toks = tuple(
            Token(token_types[ti], eid, aid, tuple(feats))
            for ti, eid, aid, feats in tokens
        )
        ev = evaluator.evaluate([Encoding(actor=actor, tokens=toks)], [list(legal)])[0]
        value_actor = float(ev.wdl[0] - ev.wdl[2])
        return value_actor, [float(p) for p in ev.policy]

    return adapter


def test_closed_search_net_matches_python():
    """F3.4: the Rust searcher driven by the REAL net (via PyEval + the F2 Rust
    encoder) is bit-identical to Python's searcher on the same net — the
    end-to-end validation of the whole F3 port with real evaluations."""

    pytest.importorskip("torch")
    import torch

    from .inference import Evaluator
    from .net import SWDNet

    torch.manual_seed(3)
    evaluator = Evaluator(SWDNet(32, 1, 2))
    adapter = _make_net_adapter(evaluator)

    checked = 0
    for game_seed in range(3):
        first_player, actions, library = random_game(game_seed, game_seed % 2)
        py = new_game(game_seed, first_player=first_player)
        rg = swr.RustGame(library_draws=[list(d) for d in library], **extract_setup(py))
        tested_here = False
        for i, idx in enumerate(actions):
            if (
                not tested_here
                and i >= 8
                and py.phase is Phase.PLAY_AGE
                and py.pending_choice is None
            ):
                legal = legal_action_indices(py)
                for sims, seed, force in [(16, 1, False), (32, 5, False), (24, 3, True)]:
                    mcts = GumbelMCTS(
                        evaluator,
                        SearchConfig(
                            sims=sims,
                            top_k=8,
                            mode="closed",
                            seed=seed,
                            force_expand_root_chance=force,
                        ),
                    )
                    result = mcts.search(py)
                    root = mcts._closed_root
                    act, av, rv, visits, policy, topk, rsims, dig = rg.closed_search_net(
                        adapter, sims, 8, seed, force=force
                    )
                    ctx = f"game {game_seed} sims {sims} seed {seed} force {force}"
                    assert act == result.action_index, f"{ctx}: action"
                    assert rsims == result.sims, f"{ctx}: sims"
                    assert list(topk) == list(result.gumbel_topk), f"{ctx}: topk"
                    assert list(visits) == [result.visits[a] for a in legal], f"{ctx}: visits"
                    assert av == pytest.approx(result.action_value, abs=1e-9), f"{ctx}: av"
                    assert rv == pytest.approx(result.root_value, abs=1e-9), f"{ctx}: rv"
                    for j, a in enumerate(legal):
                        assert policy[j] == pytest.approx(
                            result.policy_target[a], abs=1e-9
                        ), f"{ctx}: policy"
                    expected = []
                    _digest_ref(root, expected)
                    _assert_digest_equal(expected, list(dig), f"{ctx}: tree")

                    # The independently implemented F4.1 arena/state-machine
                    # path must also reproduce the F3.4 scalar-net oracle.
                    resumed = rg.closed_search_resumable_net(
                        adapter, sims, 8, seed, force=force
                    )
                    assert resumed[0] == act, f"{ctx}: resumable action"
                    assert resumed[3] == visits, f"{ctx}: resumable visits"
                    assert resumed[5] == topk, f"{ctx}: resumable top-k"
                    assert resumed[6] == rsims, f"{ctx}: resumable sims"
                    assert resumed[1] == pytest.approx(av, abs=1e-9), f"{ctx}: resumable av"
                    assert resumed[2] == pytest.approx(rv, abs=1e-9), f"{ctx}: resumable rv"
                    assert list(resumed[4]) == pytest.approx(
                        list(policy), rel=0, abs=1e-9
                    ), f"{ctx}: resumable policy"
                    _assert_digest_equal(list(dig), list(resumed[7]), f"{ctx}: resumable tree")
                    checked += 1
                tested_here = True
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)
    assert checked >= 6


def _play_age_position(game_seed, first_i=8):
    first_player, actions, library = random_game(game_seed, game_seed % 2)
    py = new_game(game_seed, first_player=first_player)
    rg = swr.RustGame(library_draws=[list(d) for d in library], **extract_setup(py))
    for i, idx in enumerate(actions):
        if i >= first_i and py.phase is Phase.PLAY_AGE and py.pending_choice is None:
            return py, rg
        apply_action(py, decode_action(py, idx))
        rg.apply_index(idx)
    raise AssertionError("no play-age position found")


def test_closed_search_net_propagates_adapter_error():
    """F3.4: an adapter raising a Python exception surfaces as that exception
    through search, not a PanicException (operational errors are not Rust
    invariants)."""

    _py, rg = _play_age_position(0)

    def bad(tokens, actor, legal):
        raise RuntimeError("adapter sentinel")

    with pytest.raises(RuntimeError, match="adapter sentinel"):
        rg.closed_search_net(bad, 8, 8, 1)


def test_closed_search_net_validates_contract():
    """F3.4: PyEval enforces the evaluator contract (finite value, priors aligned
    to legal, finite/nonnegative priors, positive mass)."""

    _py, rg = _play_age_position(0)
    # A valid uniform adapter runs to completion.
    rg.closed_search_net(lambda t, a, legal: (0.0, [1.0] * len(legal)), 8, 8, 1)
    bad_adapters = [
        lambda t, a, l: (0.0, [1.0] * (len(l) + 1)),      # too many priors
        lambda t, a, l: (0.0, [1.0] * (len(l) - 1)),      # too few priors
        lambda t, a, l: (0.0, [float("nan")] * len(l)),   # non-finite prior
        lambda t, a, l: (0.0, [-1.0] * len(l)),           # negative prior
        lambda t, a, l: (0.0, [0.0] * len(l)),            # zero-mass policy
        lambda t, a, l: (float("inf"), [1.0] * len(l)),   # non-finite value
    ]
    for adapter in bad_adapters:
        with pytest.raises(ValueError):
            rg.closed_search_net(adapter, 8, 8, 1)


def _enumerable_reveal_root(game):
    for a in legal_action_indices(game):
        kinds = {s.kind for s in py_chance_signature(game, decode_action(game, a))}
        if ChanceKind.CARD_REVEAL in kinds and ChanceKind.AGE_DEAL not in kinds:
            return True
    return False


def test_closed_search_net_force_expansion():
    """F3.4: force-expansion composed with the REAL net — a CARD_REVEAL root is
    materialized as a probability-weighted, multi-outcome edge, bit-for-bit (to
    1e-9) with Python on the same net."""

    pytest.importorskip("torch")
    import torch

    from .inference import Evaluator
    from .net import SWDNet

    torch.manual_seed(3)
    evaluator = Evaluator(SWDNet(32, 1, 2))
    adapter = _make_net_adapter(evaluator)

    found = None
    for game_seed in range(8):
        first_player, actions, library = random_game(game_seed, game_seed % 2)
        py = new_game(game_seed, first_player=first_player)
        rg = swr.RustGame(library_draws=[list(d) for d in library], **extract_setup(py))
        for i, idx in enumerate(actions):
            if i >= 8 and py.phase is Phase.PLAY_AGE and py.pending_choice is None:
                if _enumerable_reveal_root(py):
                    found = (py, rg)
                    break
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)
        if found:
            break
    assert found is not None, "no enumerable CARD_REVEAL root found"
    py, rg = found

    mcts = GumbelMCTS(
        evaluator,
        SearchConfig(sims=32, top_k=8, mode="closed", seed=1, force_expand_root_chance=True),
    )
    result = mcts.search(py)
    root = mcts._closed_root
    stats = _tree_stats(root)
    assert stats["weighted"] >= 1, "real-net force-expansion produced no weighted edge"
    assert stats["weighted_multi"] >= 1, "no weighted edge with multiple outcomes"
    act, *_rest, dig = rg.closed_search_net(adapter, 32, 8, 1, force=True)
    expected = []
    _digest_ref(root, expected)
    _assert_digest_equal(expected, list(dig), "net force-expansion tree")
    assert act == result.action_index


def test_ln_parity_matches_python():
    """F3.3: cross-runtime ln() parity over the range log_prior covers, so the
    Gumbel selection is bit-identical even on positions the search gate misses."""

    xs = [(i + 1) / 100000 for i in range(100000)]  # (0, 1]
    xs += [1e-12, 0.5, 1.0, 2.0, 1e-9, 0.9999999]
    assert swr.ln_values(xs) == [math.log(x) for x in xs]


def test_gumbel_stream_matches_python():
    """F3.3 prerequisite: Rust gumbel() equals Python's in bulk across seeds
    (cross-runtime ln parity, not just a 3-value golden)."""

    for seed in (0, 1, 7, 99, 2**40 + 5):
        rust = swr.gumbel_stream(seed, 500)
        rng = PortableRng(seed)
        expected = [rng.gumbel() for _ in range(500)]
        assert rust == expected, f"gumbel divergence at seed {seed}"


def _expected_sample(game, index, seed):
    """Python sample_outcomes for a fresh PortableRng(seed): (outcomes, prob, key)
    with outcomes and key as id lists."""

    specs = py_chance_signature(game, decode_action(game, index))
    outcomes, prob, key = py_sample_outcomes(game, specs, PortableRng(seed))
    mapped = []
    for spec, outcome in zip(specs, outcomes):
        id_map = _CHANCE_SAMPLE_ID_MAP[spec.kind]
        if isinstance(outcome, str):
            mapped.append([id_map[outcome]])
        else:
            mapped.append([id_map[name] for name in outcome])
    return mapped, prob, _map_key(specs, key)


def _expected_signature(game, index):
    """Python chance_signature as (kind_id, context) — the shape Rust returns."""

    out = []
    for spec in py_chance_signature(game, decode_action(game, index)):
        if spec.kind is ChanceKind.CARD_REVEAL:
            (row, x), back = spec.context
            ctx = [row, x, _BACK_ID[back]]
        elif spec.kind is ChanceKind.AGE_DEAL:
            ctx = [spec.context[0]]
        else:
            ctx = []
        out.append((_CHANCE_KIND_ID[spec.kind], ctx))
    return out


def _expected_chains_dict(game, index):
    """Python enumerate_chains as {outcome-id-tuple: probability}."""

    specs = py_chance_signature(game, decode_action(game, index))
    result = {}
    for outcomes, prob, _key in py_enumerate_chains(game, specs):
        key = []
        for spec, outcome in zip(specs, outcomes):
            id_map = _CHANCE_ID_MAP[spec.kind]
            if isinstance(outcome, str):
                key.append((id_map[outcome],))
            else:
                key.append(tuple(id_map[name] for name in outcome))
        result[tuple(key)] = prob
    return result

_TOKEN_TYPE_INDEX = {token_type: index for index, token_type in enumerate(TokenType)}


def _expected_encoding(game):
    """Python encoding as `(type_id, entity_id, aux_id, features)` tuples, with
    `type_id` in TokenType declaration order (the encoder ignores the viewer, so
    seat 0 is arbitrary)."""

    encoding = py_encode(game.observation(0))
    return [
        (_TOKEN_TYPE_INDEX[tok.type], tok.entity_id, tok.aux_id, tuple(tok.features))
        for tok in encoding.tokens
    ]


def _assert_encoding_equal(seed, move, expected, actual):
    if actual == expected:
        return
    assert len(actual) == len(expected), (
        f"seed {seed} move {move}: token count {len(actual)} != {len(expected)}"
    )
    for i, (exp, act) in enumerate(zip(expected, actual)):
        if exp == act:
            continue
        if exp[:3] != act[:3]:
            raise AssertionError(
                f"seed {seed} move {move} token {i}: header {act[:3]} != {exp[:3]} "
                f"(type_id, entity_id, aux_id)"
            )
        for f, (ef, af) in enumerate(zip(exp[3], act[3])):
            if ef != af:
                raise AssertionError(
                    f"seed {seed} move {move} token {i} (type {exp[0]}) feature "
                    f"{f}: rust {af!r} != python {ef!r}"
                )
        raise AssertionError(
            f"seed {seed} move {move} token {i}: feature count "
            f"{len(act[3])} != {len(exp[3])}"
        )
from .game import (
    ChanceKind,
    Phase,
    PendingChoiceKind,
    VictoryType,
    new_game,
)

import seven_wonders_rust as swr

_SCIENCE_ORDER = {s: i for i, s in enumerate(ScienceSymbol)}
_PHASE_ORD = {
    Phase.WONDER_DRAFT: 0,
    Phase.PLAY_AGE: 1,
    Phase.CHOOSE_NEXT_START_PLAYER: 2,
    Phase.COMPLETE: 3,
}
_VICTORY_ORD = {
    VictoryType.MILITARY: 0,
    VictoryType.SCIENTIFIC: 1,
    VictoryType.CIVILIAN: 2,
    VictoryType.SHARED_CIVILIAN: 3,
}
_PENDING_ORD = {
    PendingChoiceKind.DESTROY_OPPONENT_BROWN: 0,
    PendingChoiceKind.DESTROY_OPPONENT_GREY: 1,
    PendingChoiceKind.BUILD_FROM_DISCARD_FREE: 2,
    PendingChoiceKind.CHOOSE_UNUSED_PROGRESS: 3,
    PendingChoiceKind.CHOOSE_AVAILABLE_PROGRESS: 4,
}
_PROGRESS_PENDING = {
    PendingChoiceKind.CHOOSE_UNUSED_PROGRESS,
    PendingChoiceKind.CHOOSE_AVAILABLE_PROGRESS,
}


def logic_fingerprint(game) -> list[int]:
    """Language-neutral integer fingerprint of all game-logic state.

    Byte-for-byte identical to `state.rs::GameState::fingerprint`.
    """

    out: list[int] = []

    def push_list(names, id_map):
        ids = [id_map[n] for n in names]
        out.append(len(ids))
        out.extend(ids)

    out.append(_PHASE_ORD[game.phase])
    out.append(game.first_player)
    out.append(game.active_player)
    out.append(game.age)
    out.append(game.wonder_round)
    out.append(game.wonder_pick_index)

    for city in game.cities:
        out.append(city.coins)
        push_list(city.wonders, WONDER_IDS)
        push_list(city.built_wonders, WONDER_IDS)
        push_list(city.buildings, CARD_IDS)
        push_list(city.progress_tokens, PROGRESS_IDS)
        pairs = sorted(_SCIENCE_ORDER[s] for s in city.claimed_science_pairs)
        out.append(len(pairs))
        out.extend(pairs)

    push_list(game.available_progress_tokens, PROGRESS_IDS)
    push_list(game.unused_progress_tokens, PROGRESS_IDS)
    push_list(game.wonder_groups[0], WONDER_IDS)
    push_list(game.wonder_groups[1], WONDER_IDS)
    push_list(game.unused_wonders, WONDER_IDS)
    push_list(game.wonder_offer, WONDER_IDS)
    for age in (1, 2, 3):
        push_list(game.age_decks[age], CARD_IDS)
    for age in (1, 2, 3):
        push_list(game.removed_age_cards[age], CARD_IDS)
    push_list(game.selected_guilds, CARD_IDS)
    push_list(game.unused_guilds, CARD_IDS)

    slots = sorted(game.tableau.cards.items())  # by (row, x)
    out.append(len(slots))
    for (row, x), card in slots:
        out.append(row)
        out.append(x)
        out.append(CARD_IDS[card.card_name])
        out.append(int(card.present))
        out.append(int(card.present and card.revealed))

    push_list(game.discard_pile, CARD_IDS)
    push_list(game.buried_cards, CARD_IDS)

    burials = sorted(
        (WONDER_IDS[w], CARD_IDS[c]) for w, c in game.wonder_burials.items()
    )
    out.append(len(burials))
    for w, c in burials:
        out.append(w)
        out.append(c)

    retired = sorted(WONDER_IDS[w] for w in game.retired_wonders)
    out.append(len(retired))
    out.extend(retired)

    pending = game.pending_choice
    if pending is None:
        out.append(-1)
    else:
        out.append(_PENDING_ORD[pending.kind])
        out.append(pending.player)
        out.append(int(pending.consume_all_options))
        id_map = PROGRESS_IDS if pending.kind in _PROGRESS_PENDING else CARD_IDS
        ids = [id_map[o] for o in pending.options]
        out.append(len(ids))
        out.extend(ids)
    out.append(int(game.pending_extra_turn))
    out.append(game.pending_shields)

    out.append(game.conflict_position)
    mil = sorted(game.military_tokens_remaining.items())
    out.append(len(mil))
    for pos, pen in mil:
        out.append(pos)
        out.append(pen)

    out.append(-1 if game.winner is None else game.winner)
    out.append(-1 if game.victory_type is None else _VICTORY_ORD[game.victory_type])
    if game.final_scores is None:
        out.append(-1)
    else:
        out.append(1)
        out.append(game.final_scores[0])
        out.append(game.final_scores[1])
    return out


def extract_setup(game) -> dict:
    """Constructor kwargs for `RustGame` from a fresh Python `GameState`."""

    return {
        "first_player": game.first_player,
        "available_progress": list(game.available_progress_tokens),
        "unused_progress": list(game.unused_progress_tokens),
        "wonder_group0": list(game.wonder_groups[0]),
        "wonder_group1": list(game.wonder_groups[1]),
        "unused_wonders": list(game.unused_wonders),
        "age1": list(game.age_decks[1]),
        "age2": list(game.age_decks[2]),
        "age3": list(game.age_decks[3]),
        "removed1": list(game.removed_age_cards[1]),
        "removed2": list(game.removed_age_cards[2]),
        "removed3": list(game.removed_age_cards[3]),
        "selected_guilds": list(game.selected_guilds),
        "unused_guilds": list(game.unused_guilds),
    }


_BACK_ORDER = (BackType.AGE_I, BackType.AGE_II, BackType.AGE_III, BackType.GUILD)


def _expected_pool(game):
    """Python `unseen_pool` mapped to the sorted id lists the Rust `unseen_pool`
    returns: (age1, age2, age3, guild, wonders, offboard_progress)."""

    up = py_unseen_pool(game.observation(game.active_player))
    cards = tuple(
        tuple(sorted(CARD_IDS[n] for n in up.cards[back])) for back in _BACK_ORDER
    )
    return cards + (
        tuple(sorted(WONDER_IDS[n] for n in up.wonders)),
        tuple(sorted(PROGRESS_IDS[n] for n in up.offboard_progress)),
    )


def compare_game(
    seed,
    first_player,
    action_indices,
    library_draws,
    *,
    check_roundtrip=True,
    deep_every=None,
    check_pool=False,
    check_encode=False,
):
    """Drive Python and Rust from the same action sequence; assert agreement.

    Returns the number of decisions compared. Raises AssertionError with a
    localized message on the first divergence. When ``deep_every`` is set, every
    ``deep_every``-th decision also runs the exhaustive F1b audit
    (`roundtrip_all_ok`): full-state undo + apply determinism over *every* legal
    action to depth 2, not just the trajectory action.
    """

    py = new_game(seed, first_player=first_player)
    setup = extract_setup(py)

    py_fps: list[list[int]] = []
    py_masks: list[list[int]] = []
    py_pools: list[tuple] = []
    py_encodings: list[list] = []
    for idx in action_indices:
        py_fps.append(logic_fingerprint(py))
        py_masks.append(list(legal_action_indices(py)))
        if check_pool:
            py_pools.append(_expected_pool(py))
        if check_encode:
            py_encodings.append(_expected_encoding(py))
        apply_action(py, decode_action(py, idx))
    assert py.phase is Phase.COMPLETE, f"seed {seed}: python game did not complete"
    py_final = logic_fingerprint(py)

    rg = swr.RustGame(library_draws=[list(d) for d in library_draws], **setup)
    for i, idx in enumerate(action_indices):
        rust_fp = rg.fingerprint()
        if rust_fp != py_fps[i]:
            _diff_and_raise(seed, i, py_fps[i], rust_fp)
        rust_mask = rg.legal_action_indices()
        assert rust_mask == py_masks[i], (
            f"seed {seed} move {i}: legal mask mismatch\n"
            f"  python: {py_masks[i]}\n  rust:   {rust_mask}"
        )
        if check_roundtrip:
            assert rg.roundtrip_ok(idx), (
                f"seed {seed} move {i}: make/unmake did not restore state (F1b)"
            )
        if deep_every and i % deep_every == 0:
            assert rg.roundtrip_all_ok(2), (
                f"seed {seed} move {i}: exhaustive make/unmake audit failed (F1b)"
            )
        if check_pool:
            rust_pool = tuple(tuple(lst) for lst in rg.unseen_pool())
            assert rust_pool == py_pools[i], (
                f"seed {seed} move {i}: unseen-pool mismatch (F2.1)\n"
                f"  python: {py_pools[i]}\n  rust:   {rust_pool}"
            )
        if check_encode:
            rust_enc = [
                (ti, eid, aid, tuple(feats)) for ti, eid, aid, feats in rg.encode()
            ]
            _assert_encoding_equal(seed, i, py_encodings[i], rust_enc)
        rg.apply_index(idx)

    rust_final = rg.fingerprint()
    if rust_final != py_final:
        _diff_and_raise(seed, len(action_indices), py_final, rust_final)
    assert rg.is_complete(), f"seed {seed}: rust game did not complete"
    return len(action_indices)


def _diff_and_raise(seed, move, py_fp, rust_fp):
    where = "final" if move == "final" else f"move {move}"
    n = min(len(py_fp), len(rust_fp))
    first = next((k for k in range(n) if py_fp[k] != rust_fp[k]), n)
    raise AssertionError(
        f"seed {seed} {where}: fingerprint mismatch at index {first} "
        f"(py len {len(py_fp)}, rust len {len(rust_fp)})\n"
        f"  py[{first}:{first + 8}]   = {py_fp[first:first + 8]}\n"
        f"  rust[{first}:{first + 8}] = {rust_fp[first:first + 8]}"
    )


def random_game(seed, first_player):
    """Play a full game under a seeded random-legal policy; collect the action
    sequence and the ordered Great Library draws."""

    game = new_game(seed, first_player=first_player)
    rng = random.Random((seed << 1) ^ 0x9E3779B9 ^ first_player)
    actions: list[int] = []
    library: list[list[str]] = []
    while game.phase is not Phase.COMPLETE:
        legal = list(legal_action_indices(game))
        idx = rng.choice(legal)
        actions.append(idx)
        result = apply_action(game, decode_action(game, idx))
        for ev in result.events:
            if ev.kind is ChanceKind.GREAT_LIBRARY_DRAW:
                library.append(list(ev.outcome))
    return first_player, actions, library


import glob

import pytest

BUFFER_DIR = os.path.join(os.path.dirname(__file__), "runs", "phase_d_toy", "buffers")
# F1a corpus size. Default is a fast-but-real multi-file subset for routine CI;
# the documented ≥10k acceptance gate is `SWR_F1A_GAMES=0 pytest -k buffer`
# (0 = every game in every buffer file). See PHASE_F.md.
F1A_GAMES = int(os.environ.get("SWR_F1A_GAMES", "400"))
F1A_DEEP_GAMES = 4  # games that additionally get the exhaustive depth-2 audit


def iter_buffer_records(limit):
    """Yield `(index, record)` round-robin across the buffer files (one game per
    file per pass), up to `limit` games total (`limit <= 0` = all). Round-robin
    gives a stratified multi-file sample even for small limits, so a subset is
    never just a lexicographic prefix of the first file."""

    paths = sorted(glob.glob(os.path.join(BUFFER_DIR, "*.jsonl")))
    handles = [open(p, "r", encoding="utf-8") for p in paths]
    try:
        n = 0
        while True:
            progressed = False
            for fh in handles:
                line = fh.readline()
                while line and not line.strip():
                    line = fh.readline()
                if not line:
                    continue
                progressed = True
                if limit > 0 and n >= limit:
                    return
                yield n, from_json_line(line)
                n += 1
            if not progressed:
                return
    finally:
        for fh in handles:
            fh.close()


def _library_draws(record):
    return [
        outcome
        for kind, outcome in record.chance_log
        if kind == ChanceKind.GREAT_LIBRARY_DRAW.value
    ]


# --- pytest entry points ------------------------------------------------------


def test_random_games_equivalent():
    total = 0
    for seed in range(60):
        fp = seed % 2
        first_player, actions, library = random_game(seed, fp)
        # Exhaustively audit make/unmake on a few of these games.
        deep_every = 4 if seed < F1A_DEEP_GAMES else None
        total += compare_game(
            seed, first_player, actions, library, deep_every=deep_every
        )
    assert total > 0


def test_buffer_games_equivalent():
    """F1a: byte-exact replay over the buffer corpus.

    Skips (does not silently pass) when buffers are absent — they live under the
    gitignored ``runs/`` and are present on the gate box, not a fresh checkout.
    Runs ``SWR_F1A_GAMES`` games across all buffer files (default 400; set 0 for
    the full ≥10k acceptance gate) and asserts the corpus was non-empty.
    """

    if not os.path.isdir(BUFFER_DIR) or not glob.glob(
        os.path.join(BUFFER_DIR, "*.jsonl")
    ):
        pytest.skip(f"no buffer corpus under {BUFFER_DIR} (F1a needs replay buffers)")

    n = 0
    for index, record in iter_buffer_records(F1A_GAMES):
        deep_every = 25 if index < F1A_DEEP_GAMES else None
        compare_game(
            record.seed,
            record.first_player,
            [m.action for m in record.moves],
            _library_draws(record),
            deep_every=deep_every,
        )
        n += 1
    assert n > 0, "buffer corpus present but yielded no games"
    if F1A_GAMES > 0:
        assert n == F1A_GAMES, f"compared {n} games, requested {F1A_GAMES}"


F2_GAMES = int(os.environ.get("SWR_F2_GAMES", "60"))
# Default is a fast multi-file subset; the ≥100k-state acceptance gate is
# `SWR_F2_GAMES=2000 pytest -k encode_corpus` (or 0 = every buffer game). At
# acceptance scale the gate enforces the criteria below. See PHASE_F.md.
F2_ACCEPT_GAMES = 2000
F2_ACCEPT_MIN_STATES = 100_000
_ALL_TOKEN_TYPES = frozenset(range(9))
_ALL_DECISIONS = frozenset(range(9))


def _compare_encodings(seed, first_player, action_indices, library_draws, coverage):
    """Lean encode-only equivalence driver (no fingerprint/mask/roundtrip): drive
    both engines and assert bit-identical encodings at every decision *including
    the terminal COMPLETE state*. Records decision- and token-type coverage into
    `coverage`. Returns the number of states compared."""

    py = new_game(seed, first_player=first_player)
    setup = extract_setup(py)
    rg = swr.RustGame(library_draws=[list(d) for d in library_draws], **setup)

    def check(move):
        expected = _expected_encoding(py)
        rust_enc = [(ti, eid, aid, tuple(feats)) for ti, eid, aid, feats in rg.encode()]
        _assert_encoding_equal(seed, move, expected, rust_enc)
        coverage["token_types"].update(tok[0] for tok in expected)
        # The GLOBAL token (always tokens[0]) carries the decision one-hot first.
        coverage["decisions"].add(expected[0][3][:9].index(1.0))

    for i, idx in enumerate(action_indices):
        check(i)
        apply_action(py, decode_action(py, idx))
        rg.apply_index(idx)
    check(len(action_indices))  # terminal COMPLETE state (else decision 8 is untested)
    return len(action_indices) + 1


def test_encode_corpus_equivalent():
    """F2.3: bit-exact encoder over the buffer corpus. Skips when buffers are
    absent; runs ``SWR_F2_GAMES`` games round-robin across all files (default 60;
    2000 or 0 = acceptance). At acceptance scale it enforces ≥100k states and
    full decision/token-type coverage."""

    if not os.path.isdir(BUFFER_DIR) or not glob.glob(
        os.path.join(BUFFER_DIR, "*.jsonl")
    ):
        pytest.skip(f"no buffer corpus under {BUFFER_DIR} (F2.3 needs replay buffers)")

    coverage = {"token_types": set(), "decisions": set()}
    games = 0
    states = 0
    for _index, record in iter_buffer_records(F2_GAMES):
        states += _compare_encodings(
            record.seed,
            record.first_player,
            [m.action for m in record.moves],
            _library_draws(record),
            coverage,
        )
        games += 1
    assert games > 0, "buffer corpus present but yielded no games"
    if F2_GAMES > 0:
        assert games == F2_GAMES, f"compared {games} games, requested {F2_GAMES}"
    print(
        f"F2.3 encode corpus: {states} states over {games} games; "
        f"decisions={sorted(coverage['decisions'])} "
        f"token_types={sorted(coverage['token_types'])}"
    )
    if F2_GAMES == 0 or F2_GAMES >= F2_ACCEPT_GAMES:
        assert states >= F2_ACCEPT_MIN_STATES, (
            f"acceptance run compared only {states} states (< {F2_ACCEPT_MIN_STATES})"
        )
        assert coverage["decisions"] == _ALL_DECISIONS, (
            f"missing decision branches: {sorted(_ALL_DECISIONS - coverage['decisions'])}"
        )
        assert coverage["token_types"] == _ALL_TOKEN_TYPES, (
            f"missing token types: {sorted(_ALL_TOKEN_TYPES - coverage['token_types'])}"
        )


def test_unseen_pool_equivalent():
    """F2.1: Rust `unseen_pool` (encoder foundation) matches Python's public
    projection at every decision, across all phases (random games span draft →
    all three ages → endgame)."""

    total = 0
    for seed in range(40):
        first_player, actions, library = random_game(seed, seed % 2)
        total += compare_game(
            seed, first_player, actions, library, check_pool=True
        )
    assert total > 0


def test_chance_signature_and_chains_equivalent():
    """F3.1a: Rust chance_signature and enumerate_chains match Python at every
    legal action across all phases (random games span draft/reveals/Great
    Library/age boundaries). AGE_DEAL specs are sample-only — both refuse to
    enumerate them."""

    checked_sig = 0
    checked_chains = 0
    seen_kinds = set()
    seen_deal_ages = set()
    multi_reveal_seen = False
    for seed in range(25):
        first_player, actions, library = random_game(seed, seed % 2)
        py = new_game(seed, first_player=first_player)
        rg = swr.RustGame(library_draws=[list(d) for d in library], **extract_setup(py))
        for idx in actions:
            for index in legal_action_indices(py):
                expected_sig = _expected_signature(py, index)
                rust_sig = [(k, list(ctx)) for k, ctx in rg.chance_signature(index)]
                assert rust_sig == expected_sig, (
                    f"seed {seed} action {index}: chance_signature mismatch\n"
                    f"  python: {expected_sig}\n  rust:   {rust_sig}"
                )
                checked_sig += 1
                seen_kinds.update(k for k, _ in expected_sig)
                seen_deal_ages.update(
                    ctx[0]
                    for k, ctx in expected_sig
                    if k == _CHANCE_KIND_ID[ChanceKind.AGE_DEAL]
                )
                if sum(1 for k, _ in expected_sig if k == 0) >= 2:
                    multi_reveal_seen = True
                if any(k == _CHANCE_KIND_ID[ChanceKind.AGE_DEAL] for k, _ in expected_sig):
                    with pytest.raises(ValueError):
                        rg.enumerate_chains(index)
                    continue
                expected = _expected_chains_dict(py, index)
                rust = {}
                for outcomes, prob, key in rg.enumerate_chains(index):
                    outcome_tup = tuple(tuple(o) for o in outcomes)
                    # Off AGE_DEAL the observable key equals the outcomes.
                    assert [list(k) for k in key] == [list(o) for o in outcomes], (
                        f"seed {seed} action {index}: enumerate key != outcomes"
                    )
                    rust[outcome_tup] = prob
                assert rust.keys() == expected.keys(), (
                    f"seed {seed} action {index}: chain outcome set mismatch"
                )
                for key, prob in expected.items():
                    assert rust[key] == pytest.approx(prob, abs=1e-12)
                checked_chains += 1
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)
    assert checked_sig > 1000 and checked_chains > 500
    # Coverage guard (F1a/F2.3 lesson): the corpus must exercise every chance
    # kind, all three age-deal ages (age-3 = the guild path), and a sequential
    # multi-reveal — else a branch could be broken yet untested.
    assert seen_kinds == {0, 1, 2, 3}, f"chance kinds not all covered: {seen_kinds}"
    assert seen_deal_ages == {1, 2, 3}, f"age-deal ages not all covered: {seen_deal_ages}"
    assert multi_reveal_seen, "no sequential multi-reveal action exercised"


def test_sample_outcomes_equivalent():
    """F3.1b: Rust sample_outcomes reproduces Python's sampled chain under a
    shared seed (portable RNG parity), including AGE_DEAL's shuffle path."""

    checked = 0
    sampled_kinds = set()
    sampled_deal_ages = set()
    for seed in range(25):
        first_player, actions, library = random_game(seed, seed % 2)
        py = new_game(seed, first_player=first_player)
        rg = swr.RustGame(library_draws=[list(d) for d in library], **extract_setup(py))
        for idx in actions:
            for index in legal_action_indices(py):
                specs = py_chance_signature(py, decode_action(py, index))
                if not specs:
                    continue
                sampled_kinds.update(_CHANCE_KIND_ID[s.kind] for s in specs)
                sampled_deal_ages.update(
                    s.context[0] for s in specs if s.kind is ChanceKind.AGE_DEAL
                )
                for rng_seed in (0, 1, 12345):
                    exp_outcomes, exp_prob, exp_key = _expected_sample(py, index, rng_seed)
                    rust_outcomes, rust_prob, rust_key = rg.sample_outcomes(index, rng_seed)
                    rust_outcomes = [list(o) for o in rust_outcomes]
                    rust_key = [list(k) for k in rust_key]
                    assert rust_outcomes == exp_outcomes, (
                        f"seed {seed} action {index} rng {rng_seed}: sample mismatch\n"
                        f"  python: {exp_outcomes}\n  rust:   {rust_outcomes}"
                    )
                    assert rust_key == exp_key, (
                        f"seed {seed} action {index} rng {rng_seed}: key mismatch\n"
                        f"  python: {exp_key}\n  rust:   {rust_key}"
                    )
                    if exp_prob is None:
                        assert rust_prob is None
                    else:
                        assert rust_prob == pytest.approx(exp_prob, abs=1e-12)
                    checked += 1
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)
    assert checked > 500
    # AGE_DEAL age-3 exercises the guild-split triple-shuffle path specifically.
    assert sampled_kinds == {0, 1, 2, 3}, f"sample kinds not all covered: {sampled_kinds}"
    assert sampled_deal_ages == {1, 2, 3}, f"sample age-deal ages: {sampled_deal_ages}"


def _outcome_to_ids(specs, outcomes):
    """A single chain's per-spec outcomes -> id lists (Rust's format)."""

    rust = []
    for spec, outcome in zip(specs, outcomes):
        id_map = _CHANCE_SAMPLE_ID_MAP[spec.kind]
        if isinstance(outcome, str):
            rust.append([id_map[outcome]])
        else:
            rust.append([id_map[name] for name in outcome])
    return rust


def _py_fingerprint_after_chance(game, index, py_outcomes):
    clone = game.clone()
    clone.search_barrier = True
    apply_action(clone, decode_action(clone, index), chance_outcomes=py_outcomes)
    return logic_fingerprint(clone)


def _reveal_source(game, spec, name):
    """Which SWAP branch a reveal outcome exercises: the slot's own card,
    a sibling face-down slot, the removed pile, or the unused guilds."""

    slot_id, _back = spec.context
    if name == game.tableau.cards[slot_id].card_name:
        return "self"
    for sid, card in game.tableau.cards.items():
        if sid != slot_id and card.present and not card.revealed and card.card_name == name:
            return "sibling"
    if name in game.removed_age_cards.get(game.age, ()):
        return "removed"
    if name in game.unused_guilds:
        return "guild"
    return "unknown"


def test_make_with_chance_equivalent():
    """F3.1b: applying an action with a supplied chance outcome (the SWAP path)
    yields the same complete state in Rust as Python's
    apply_action(chance_outcomes=...), across every chance kind."""

    checked = 0
    covered = set()
    swap_sources = set()
    deal_ages = set()
    seq_same_back = False
    for seed in range(20):
        first_player, actions, library = random_game(seed, seed % 2)
        py = new_game(seed, first_player=first_player)
        rg = swr.RustGame(library_draws=[list(d) for d in library], **extract_setup(py))
        for idx in actions:
            for index in legal_action_indices(py):
                specs = py_chance_signature(py, decode_action(py, index))
                if not specs:
                    continue
                covered.update(_CHANCE_KIND_ID[s.kind] for s in specs)
                deal_ages.update(
                    s.context[0] for s in specs if s.kind is ChanceKind.AGE_DEAL
                )
                reveal_backs = [s.context[1] for s in specs if s.kind is ChanceKind.CARD_REVEAL]
                if len(reveal_backs) >= 2 and len(set(reveal_backs)) == 1:
                    seq_same_back = True
                if any(s.kind is ChanceKind.AGE_DEAL for s in specs):
                    outcome_lists = [
                        py_sample_outcomes(py, specs, PortableRng(s))[0] for s in (0, 7)
                    ]
                else:
                    chains = py_enumerate_chains(py, specs)
                    picks = sorted({0, len(chains) // 2, len(chains) - 1})
                    outcome_lists = [chains[i][0] for i in picks]
                for outcomes in outcome_lists:
                    for spec, outcome in zip(specs, outcomes):
                        if spec.kind is ChanceKind.CARD_REVEAL:
                            swap_sources.add(_reveal_source(py, spec, outcome))
                    py_fp = _py_fingerprint_after_chance(py, index, outcomes)
                    rust_fp = rg.fingerprint_after_chance(
                        index, _outcome_to_ids(specs, outcomes)
                    )
                    assert rust_fp == py_fp, (
                        f"seed {seed} action {index}: make_with_chance state mismatch\n"
                        f"  outcomes: {outcomes}"
                    )
                    checked += 1
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)
    assert checked > 500
    assert covered == {0, 1, 2, 3}, f"make_with_chance kinds not all covered: {covered}"
    # SWAP branch coverage (reviewer): the gate must exercise sibling, removed,
    # and unused-guild reveal sources, sequential same-back reveals, and every
    # age-deal age (age-3 = the guild-partition rebuild).
    assert {"sibling", "removed", "guild"} <= swap_sources, f"swap sources: {swap_sources}"
    assert seq_same_back, "no sequential same-back reveal exercised"
    assert deal_ages == {1, 2, 3}, f"age-deal ages: {deal_ages}"


def test_make_with_chance_rejects_malformed():
    """F3.1b: apply_with_chance validates before mutating and rejects malformed
    supplied outcomes rather than panicking on a partial state."""

    fp, actions, library = random_game(3, 1)
    py = new_game(3, first_player=1)
    rg = swr.RustGame(library_draws=[list(d) for d in library], **extract_setup(py))
    for idx in actions:
        for index in legal_action_indices(py):
            specs = py_chance_signature(py, decode_action(py, index))
            reveal = next(
                (i for i, s in enumerate(specs) if s.kind is ChanceKind.CARD_REVEAL),
                None,
            )
            if reveal is None:
                continue
            chains = py_enumerate_chains(py, specs)
            good = _outcome_to_ids(specs, chains[0][0])
            # Corrupt the reveal outcome to an illegal shape (two cards).
            bad = [list(o) for o in good]
            bad[reveal] = bad[reveal] * 2
            fp_before = rg.fingerprint()
            with pytest.raises(ValueError):
                rg.fingerprint_after_chance(index, bad)
            assert rg.fingerprint() == fp_before  # state untouched
            return
        apply_action(py, decode_action(py, idx))
        rg.apply_index(idx)
    raise AssertionError("no reveal-bearing action found to test rejection")


def test_encoder_signature_matches():
    """F2.3: the Rust build is bound to the Python encoder schema signature.
    Diverging feature order/count/naming changes Python's ENCODER_SIGNATURE and
    fails here until the Rust constant is updated in lockstep."""

    from .encoder import ENCODER_SIGNATURE

    assert swr.encoder_signature() == ENCODER_SIGNATURE


def test_encode_equivalent():
    """F2.2: Rust `encode` is bit-identical (token type/entity/aux/features) to
    Python `encode(observation)` at every decision, across all phases."""

    total = 0
    for seed in range(40):
        first_player, actions, library = random_game(seed, seed % 2)
        total += compare_game(
            seed, first_player, actions, library, check_encode=True
        )
    assert total > 0


def test_apply_index_rejects_illegal_action():
    """The public apply boundary must reject a non-legal index rather than
    mutating state through an unchecked decode (e.g. an unowned wonder)."""

    py = new_game(0, first_player=0)
    rg = swr.RustGame(library_draws=[], **extract_setup(py))
    legal = set(rg.legal_action_indices())
    illegal = next(i for i in range(swr.num_actions()) if i not in legal)
    with pytest.raises(ValueError):
        rg.apply_index(illegal)
    # State is unchanged and a legal action still applies.
    assert rg.legal_action_indices() == sorted(legal)
    rg.apply_index(next(iter(legal)))


def test_generated_rust_data_matches_python():
    """`src/data_gen.rs` must equal a fresh `export_rust_data.generate()` — makes
    "cannot drift" enforced, not just documented."""

    import importlib.util

    gen_py = os.path.join(
        os.path.dirname(__file__), "seven_wonders_rust", "export_rust_data.py"
    )
    spec = importlib.util.spec_from_file_location("swr_export_rust_data", gen_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    gen_rs = os.path.join(
        os.path.dirname(__file__), "seven_wonders_rust", "src", "data_gen.rs"
    )
    with open(gen_rs, "r", encoding="utf-8") as fh:
        on_disk = fh.read()
    fresh = module.generate()
    assert fresh.replace("\r\n", "\n") == on_disk.replace("\r\n", "\n"), (
        "src/data_gen.rs is stale — regenerate with "
        "`python -m games.seven_wonders_duel.seven_wonders_rust.export_rust_data`"
    )
