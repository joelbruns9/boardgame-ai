from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pytest

import games.kingdomino.endgame_solver as endgame_solver
from games.kingdomino.board import Board
from games.kingdomino.encoder import encode_state
from games.kingdomino.endgame_solver import count_endgame_nodes, exact_endgame_value
from games.kingdomino.dominoes import Terrain
from games.kingdomino.game import Claim, GameConfig, GameState, Phase
from games.kingdomino.mcts_az import OpenLoopMCTS, terminal_search_value


def _reach_state(seed: int = 42, *, max_deck: int = 2, min_deck: int = 0) -> GameState:
    rng = random.Random(0)
    state = GameState.new(seed=seed)
    while state.phase != Phase.GAME_OVER:
        if (
            state.phase in (Phase.PLACE_AND_SELECT, Phase.FINAL_PLACEMENT)
            and min_deck <= len(state.deck) <= max_deck
        ):
            return state
        state = state.step(rng.choice(state.legal_actions()))
    raise AssertionError("Random playout reached GAME_OVER before target state.")


def _finish_game(seed: int = 123) -> GameState:
    rng = random.Random(seed)
    state = GameState.new(seed=seed)
    while state.phase != Phase.GAME_OVER:
        state = state.step(rng.choice(state.legal_actions()))
    return state


def _reach_final_placement(seed: int = 11) -> GameState:
    rng = random.Random(0)
    state = GameState.new(seed=seed)
    while state.phase not in (Phase.FINAL_PLACEMENT, Phase.GAME_OVER):
        state = state.step(rng.choice(state.legal_actions()))
    assert state.phase == Phase.FINAL_PLACEMENT
    return state


def _full_board_with_no_legal_placements() -> Board:
    board = Board()
    cx, cy = board.castle_pos
    board.terrain.fill(int(Terrain.EMPTY))
    board.crowns.fill(0)
    board.domino_id.fill(0)
    board._occupied = set()
    board._cell = {}

    for y in range(cy - 3, cy + 4):
        for x in range(cx - 3, cx + 4):
            terrain = int(Terrain.CASTLE if (x, y) == (cx, cy) else Terrain.WHEAT)
            board.terrain[y, x] = terrain
            board.domino_id[y, x] = -1 if terrain == int(Terrain.CASTLE) else 99
            board._occupied.add((x, y))
            board._cell[(x, y)] = terrain

    board._min_x = cx - 3
    board._max_x = cx + 3
    board._min_y = cy - 3
    board._max_y = cy + 3
    return board


def _forced_discard_deck_four_state() -> GameState:
    return GameState(
        config=GameConfig(),
        boards=[_full_board_with_no_legal_placements(), _full_board_with_no_legal_placements()],
        deck=[21, 22, 23, 24],
        current_row=[1, 2, 3, 4],
        pending_claims=[Claim(0, 5), Claim(1, 6), Claim(1, 7), Claim(0, 8)],
        next_claims=[],
        phase=Phase.PLACE_AND_SELECT,
    )


def _uniform_evaluator(_mb, _ob, _flat, idxs):
    return 0.0, np.zeros(len(idxs), dtype=np.float32)


def test_exact_vs_sampled_convergence():
    state = _reach_state(seed=0, max_deck=0)
    exact, solved = exact_endgame_value(
        state,
        max_nodes=200_000,
        rng=random.Random(1),
        score_scale=100.0,
        margin_gain=2.0,
        alpha=0.8,
    )
    assert solved

    mcts = OpenLoopMCTS(
        _uniform_evaluator,
        n_simulations=80,
        c_puct=1.0,
        dirichlet_epsilon=0.0,
        exact_endgame_enabled=True,
        exact_endgame_max_nodes=200_000,
        exact_endgame_max_hidden_tiles=3,
    )
    _visit_counts, root = mcts.search(state, rng=np.random.default_rng(0))
    assert mcts._exact_solve_count > 0
    assert abs(root.value - exact) < 0.05


def test_mcts_exact_solves_deck_four_by_default():
    state = _forced_discard_deck_four_state()
    mcts = OpenLoopMCTS(
        _uniform_evaluator,
        n_simulations=8,
        c_puct=1.0,
        dirichlet_epsilon=0.0,
        exact_endgame_enabled=True,
        exact_endgame_max_nodes=50_000,
    )

    _visit_counts, _root = mcts.search(state, rng=np.random.default_rng(1))
    assert mcts._exact_solve_count > 0


# ── OPT-1/2/3/4 optimization tests ──────────────────────────────────────────


def _reach_deck4_rust(seed_start: int = 0, *, want: int = 1):
    """Yield (GameState, RustGameState) pairs at deck==4 PLACE_AND_SELECT from
    real random playouts. Used by the optimization tests."""
    kr = pytest.importorskip("kingdomino_rust")
    from games.kingdomino.endgame_solver import _rust_state_from_python

    found = 0
    seed = seed_start
    while found < want:
        rng = random.Random(10_000 + seed)
        state = GameState.new(seed=seed)
        seed += 1
        while state.phase != Phase.GAME_OVER:
            if len(state.deck) == 4 and state.phase == Phase.PLACE_AND_SELECT:
                rs = _rust_state_from_python(state)
                if rs is not None:
                    found += 1
                    yield state, rs
                break
            state = state.step(rng.choice(state.legal_actions()))


def test_alpha_beta_prunes_full_tree():
    """OPT-3/4: alpha-beta + move ordering visit strictly fewer nodes than the
    full minimax tree. Restricted to positions whose TRUE full count fits under
    the count cap (so the comparison uses an exact full count, not a saturated
    sentinel)."""
    kr = pytest.importorskip("kingdomino_rust")

    COUNT_CAP = 1_200_000
    checked = 0
    for _state, rs in _reach_deck4_rust(seed_start=0, want=25):
        full = kr.count_endgame_nodes_no_chance(rs, COUNT_CAP)
        # Need an exact full count (under cap) and a non-trivial tree to prune.
        if not (100_000 <= full < COUNT_CAP):
            continue
        value, solved, nodes = kr.exact_endgame_value_no_chance(
            rs, 2_000_000, 100.0, 2.0, 0.0
        )
        assert solved
        assert nodes < full  # pruning visited strictly fewer nodes than the full tree
        assert -1.0 <= value <= 1.0
        checked += 1
        if checked >= 3:
            break
    assert checked > 0, "no deck=4 position with an exact, non-trivial full count found"


def test_alpha_beta_value_exact_and_public_consistent():
    """OPT-3: alpha-beta returns the EXACT value — independent of the node budget
    (pruning never changes the answer) and independent of hidden deck order (the
    deck=4 tiles become a sorted row, so order is public-irrelevant).

    Equivalence to plain minimax on a small deck=4 tree is covered separately by
    test_rust_deck_four_matches_python; a full Python solve of a real deck=4 tree
    (~10M nodes) is too slow to use as a reference here."""
    kr = pytest.importorskip("kingdomino_rust")
    from games.kingdomino.endgame_solver import _rust_state_from_python

    checked = 0
    for state, rs in _reach_deck4_rust(seed_start=0, want=12):
        v_500k, ok_500k, _ = kr.exact_endgame_value_no_chance(rs, 500_000, 100.0, 2.0, 0.0)
        v_2m, ok_2m, _ = kr.exact_endgame_value_no_chance(rs, 2_000_000, 100.0, 2.0, 0.0)
        if not (ok_500k and ok_2m):
            continue
        # Budget-independence: a larger budget cannot change an exact result.
        assert v_500k == pytest.approx(v_2m, abs=1e-12)
        # Hidden-order invariance: reverse the deck, rebuild, re-solve.
        permuted = state.copy()
        permuted.deck = list(reversed(state.deck))
        rs2 = _rust_state_from_python(permuted)
        v_perm, ok_perm, _ = kr.exact_endgame_value_no_chance(rs2, 2_000_000, 100.0, 2.0, 0.0)
        assert ok_perm
        assert v_perm == pytest.approx(v_2m, abs=1e-12)
        checked += 1
        if checked >= 3:
            break
    assert checked > 0, "no solvable deck=4 position found for exactness check"


def test_solve_rate_deck4_500k():
    """OPT-2/3/4: solve rate on real deck=4 positions is high at 500k nodes."""
    kr = pytest.importorskip("kingdomino_rust")

    solved = 0
    total = 0
    for _state, rs in _reach_deck4_rust(seed_start=0, want=10):
        _v, ok, _n = kr.exact_endgame_value_no_chance(rs, 500_000, 100.0, 2.0, 0.0)
        total += 1
        solved += int(ok)
    assert total == 10
    assert solved / total >= 0.5  # measured ~97%; floor well below to avoid flakiness


def test_solve_once_cache():
    """OPT-1: a leaf solved exactly is cached on the node; later simulations
    reaching it are served from the cache rather than re-solving."""
    state = _forced_discard_deck_four_state()
    mcts = OpenLoopMCTS(
        _uniform_evaluator,
        n_simulations=64,
        c_puct=1.0,
        dirichlet_epsilon=0.0,
        exact_endgame_enabled=True,
        exact_endgame_max_nodes=500_000,
    )
    _visit_counts, _root = mcts.search(state, rng=np.random.default_rng(7))
    assert mcts._exact_solve_count > 0
    # With 64 simulations over a tiny tree, most leaf visits must hit the cache;
    # each distinct leaf is solved exactly once.
    assert mcts._exact_cache_hits > 0
    assert mcts._exact_cache_hits > mcts._exact_solve_count


def test_exact_solving_gated_off_at_non_terminal_adjacent_root():
    """Correctness gate: from a root that is NOT terminal-adjacent (deck >
    max_hidden_tiles), exact solving never fires. A deck=4 leaf reached from such
    a root has a determinization-dependent board, so caching one value would be
    wrong; the gate prevents both the incorrect cache and the throughput hit of
    re-solving deep leaves every simulation."""
    state = _reach_state(seed=5, max_deck=8, min_deck=8)
    assert len(state.deck) == 8
    mcts = OpenLoopMCTS(
        _uniform_evaluator,
        n_simulations=32,
        c_puct=1.0,
        dirichlet_epsilon=0.0,
        exact_endgame_enabled=True,
        exact_endgame_max_nodes=500_000,
    )
    _visit_counts, _root = mcts.search(state, rng=np.random.default_rng(3))
    assert mcts._exact_endgame_active is False
    assert mcts._exact_solve_count == 0
    assert mcts._exact_cache_hits == 0


# ── OPT-4b (better ordering) + OPT-6 (parallel YBW) tests ────────────────────


def test_parallel_matches_serial():
    """OPT-6: the YBW parallel solver returns the same exact value as the serial
    solver on real deck=4 positions (minimax value is order-independent)."""
    pytest.importorskip("kingdomino_rust")

    checked = 0
    for _state, rs in _reach_deck4_rust(seed_start=0, want=16):
        v_ser, ok_ser, _n_ser, _t = rs.measure_endgame_tree(2_000_000, 100.0, 2.0, 0.8, False)
        v_par, ok_par, _n_par, _t = rs.measure_endgame_tree(2_000_000, 100.0, 2.0, 0.8, True)
        if ok_ser and ok_par:
            assert v_ser == pytest.approx(v_par, abs=1e-9)
            checked += 1
    assert checked >= 8, "too few positions solved within 2M to validate equivalence"


def test_parallel_no_budget_regression():
    """OPT-6: the parallel solver reliably solves a deck=0 FINAL_PLACEMENT
    position (tiny tree) and agrees with the serial solver."""
    pytest.importorskip("kingdomino_rust")
    from games.kingdomino.endgame_solver import _rust_state_from_python

    state = _reach_final_placement(seed=11)
    assert len(state.deck) == 0
    rs = _rust_state_from_python(state)
    v_par, ok_par, _n, _t = rs.measure_endgame_tree(50_000, 100.0, 2.0, 0.8, True)
    v_ser, ok_ser, _n2, _t2 = rs.measure_endgame_tree(50_000, 100.0, 2.0, 0.8, False)
    assert ok_par and ok_ser
    assert v_par == pytest.approx(v_ser, abs=1e-9)
    assert -1.0 <= v_par <= 1.0


def test_ordering_shrinks_nodes():
    """OPT-4b: the score-delta move ordering keeps the median serial node count
    below the documented pre-OPT-4b baseline (p50=111,856 over the same seeds)."""
    pytest.importorskip("kingdomino_rust")

    nodes = []
    for _state, rs in _reach_deck4_rust(seed_start=0, want=10):
        _v, ok, n, _t = rs.measure_endgame_tree(10_000_000, 100.0, 2.0, 0.8, False)
        if ok:
            nodes.append(n)
    assert len(nodes) >= 8
    nodes.sort()
    median = nodes[len(nodes) // 2]
    assert median < 111_856, f"median node count {median:,} did not improve on baseline"


def test_encoder_hidden_order_independence():
    state = _reach_state(seed=7, max_deck=6)
    clone = state.copy()
    clone.deck = list(reversed(clone.deck))
    assert clone.deck != state.deck

    enc_a = encode_state(state, 0)
    enc_b = encode_state(clone, 0)
    assert all(np.array_equal(a, b) for a, b in zip(enc_a, enc_b))


def test_exact_solver_public_consistent():
    # FINAL_PLACEMENT is a cheap real-engine state where deck order should be
    # irrelevant to exact values. We install a small unordered hidden bag to
    # exercise the public-consistency contract without exploding the tree.
    state = _reach_final_placement(seed=11)
    state = state.copy()
    state.deck = [1, 2, 3]
    clone = state.copy()
    clone.deck = list(reversed(clone.deck))
    assert clone.deck != state.deck

    v_a, solved_a = exact_endgame_value(
        state,
        max_nodes=200_000,
        rng=random.Random(2),
        score_scale=100.0,
        margin_gain=2.0,
        alpha=0.8,
    )
    v_b, solved_b = exact_endgame_value(
        clone,
        max_nodes=200_000,
        rng=random.Random(3),
        score_scale=100.0,
        margin_gain=2.0,
        alpha=0.8,
    )
    assert solved_a and solved_b
    assert v_a == pytest.approx(v_b, abs=1e-6)


def test_no_exact_solve_above_budget():
    state = _reach_state(seed=13, max_deck=2)
    value, solved = exact_endgame_value(
        state,
        max_nodes=1,
        rng=random.Random(4),
        score_scale=100.0,
        margin_gain=2.0,
        alpha=0.8,
    )
    assert value == 0.0
    assert not solved
    assert count_endgame_nodes(state) > 1


def test_game_over_state_returns_terminal_value():
    state = _finish_game()
    value, solved = exact_endgame_value(
        state,
        max_nodes=1,
        rng=random.Random(5),
        score_scale=100.0,
        margin_gain=2.0,
        alpha=0.8,
    )
    expected = terminal_search_value(
        state,
        player=0,
        score_scale=100.0,
        margin_gain=2.0,
        alpha=0.8,
    )
    assert solved
    assert value == pytest.approx(expected)


def test_rust_deck_empty_matches_python(monkeypatch):
    state = _reach_state(seed=12, max_deck=0)

    rust_nodes = count_endgame_nodes(state, max_nodes=50_000)
    rust_value, rust_solved = exact_endgame_value(
        state,
        max_nodes=50_000,
        rng=random.Random(9),
        score_scale=100.0,
        margin_gain=2.0,
        alpha=0.8,
    )

    monkeypatch.setattr(endgame_solver, "_rust_count_no_chance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(endgame_solver, "_rust_exact_no_chance", lambda *_args, **_kwargs: None)

    py_nodes = count_endgame_nodes(state, max_nodes=50_000)
    py_value, py_solved = exact_endgame_value(
        state,
        max_nodes=50_000,
        rng=random.Random(10),
        score_scale=100.0,
        margin_gain=2.0,
        alpha=0.8,
    )

    assert rust_nodes == py_nodes
    assert rust_solved and py_solved
    assert rust_value == pytest.approx(py_value, abs=1e-12)


def test_rust_deck_four_matches_python(monkeypatch):
    state = _forced_discard_deck_four_state()
    assert len(state.deck) == 4
    assert all(action.placement is None for action in state.legal_actions())

    rust_nodes = count_endgame_nodes(state, max_nodes=50_000)
    rust_value, rust_solved = exact_endgame_value(
        state,
        max_nodes=50_000,
        rng=random.Random(11),
        score_scale=100.0,
        margin_gain=2.0,
        alpha=0.8,
    )

    monkeypatch.setattr(endgame_solver, "_rust_count_no_chance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(endgame_solver, "_rust_exact_no_chance", lambda *_args, **_kwargs: None)

    py_nodes = count_endgame_nodes(state, max_nodes=50_000)
    py_value, py_solved = exact_endgame_value(
        state,
        max_nodes=50_000,
        rng=random.Random(12),
        score_scale=100.0,
        margin_gain=2.0,
        alpha=0.8,
    )

    assert rust_nodes == py_nodes
    assert rust_solved and py_solved
    assert rust_value == pytest.approx(py_value, abs=1e-12)


@dataclass
class _FakeState:
    name: str
    phase: Phase
    current_actor: int = 0
    deck: list[int] | None = None
    current_row: list[int] | None = None
    actor_index: int = 0

    def __post_init__(self):
        if self.deck is None:
            self.deck = []
        if self.current_row is None:
            self.current_row = []

    def legal_actions(self):
        return {
            "root": ["greedy_now", "patient"],
            "greedy_reply": ["punish", "miss"],
            "patient_reply": ["hold", "overreach"],
        }.get(self.name, [])

    def step(self, action):
        transitions = {
            ("root", "greedy_now"): _FakeState("greedy_reply", Phase.PLACE_AND_SELECT, 1),
            ("root", "patient"): _FakeState("patient_reply", Phase.PLACE_AND_SELECT, 1),
            ("greedy_reply", "punish"): _TerminalState(5, 8),
            ("greedy_reply", "miss"): _TerminalState(8, 5),
            ("patient_reply", "hold"): _TerminalState(7, 6),
            ("patient_reply", "overreach"): _TerminalState(10, 4),
        }
        return transitions[(self.name, action)]


class _TerminalState(_FakeState):
    def __init__(self, p0: int, p1: int):
        super().__init__("terminal", Phase.GAME_OVER, 0)
        self._scores = [p0, p1]

    def scores(self):
        return self._scores


def test_minimax_beats_greedy():
    root = _FakeState("root", Phase.PLACE_AND_SELECT, current_actor=0)
    value, solved = exact_endgame_value(
        root,  # type: ignore[arg-type]
        max_nodes=100,
        rng=random.Random(6),
        score_scale=100.0,
        margin_gain=2.0,
        alpha=0.8,
    )
    greedy_after_opponent, _ = exact_endgame_value(
        root.step("greedy_now"),  # type: ignore[arg-type]
        max_nodes=100,
        rng=random.Random(7),
        score_scale=100.0,
        margin_gain=2.0,
        alpha=0.8,
    )
    patient_after_opponent, _ = exact_endgame_value(
        root.step("patient"),  # type: ignore[arg-type]
        max_nodes=100,
        rng=random.Random(8),
        score_scale=100.0,
        margin_gain=2.0,
        alpha=0.8,
    )

    assert solved
    assert greedy_after_opponent < patient_after_opponent
    assert value == pytest.approx(patient_after_opponent)


# ── BatchedMCTS exact endgame integration tests ─────────────────────────────


def _batched_uniform_eval(mb, ob, flat, idxs):
    """Uniform evaluator for BatchedMCTS: zero value, zero logits."""
    b = mb.shape[0]
    return (
        np.zeros(b, dtype=np.float32),
        [np.zeros(len(ix), dtype=np.float32) for ix in idxs],
    )


def _drive_batched(mcts, eval_fn=_batched_uniform_eval):
    """Run a BatchedMCTS to completion, returning the finished games list."""
    finished = []
    ticks = 0
    while not mcts.done():
        mb, ob, flat, idxs = mcts.step()
        values, gathered = eval_fn(mb, ob, flat, idxs)
        finished.extend(mcts.update(values, gathered))
        ticks += 1
        if ticks > 200_000:
            raise AssertionError("BatchedMCTS tick guard exceeded")
    return finished


def test_batched_exact_solve_fires():
    """The endgame solver fires during BatchedMCTS self-play (deck∈{0,4} roots)."""
    kr = pytest.importorskip("kingdomino_rust")
    mcts = kr.BatchedMCTS(
        n_slots=1, n_games=1, base_seed=0, n_sims=400, leaf_batch=6,
        open_loop=True, exact_endgame_max_nodes=15_000_000,
    )
    _drive_batched(mcts)
    assert mcts.exact_solve_count > 0
    assert mcts.exact_tree_solve_count > 0
    assert mcts.exact_cache_hit_count > 0
    assert mcts.exact_tree_solve_count < mcts.exact_solve_count


def test_batched_exact_policy_is_valid():
    """Every training record (exact-solved and MCTS) carries a valid policy
    target: non-empty, non-negative, sums to 1, supported on legal actions.
    (The exact policy peaking on the minimax-optimal move is unit-tested in Rust
    `exact_policy_tests`.)"""
    kr = pytest.importorskip("kingdomino_rust")
    mcts = kr.BatchedMCTS(
        n_slots=2, n_games=2, base_seed=0, n_sims=64, leaf_batch=6,
        open_loop=True, exact_endgame_max_nodes=15_000_000,
    )
    finished = _drive_batched(mcts)
    assert mcts.exact_solve_count > 0
    assert finished
    for _seed, examples, _scores in finished:
        for ex in examples:
            # ex = (my, opp, flat, policy_idx, policy_val, legal_idx, z, own, opp, win)
            policy_idx = np.asarray(ex[3])
            policy_val = np.asarray(ex[4])
            legal_idx = np.asarray(ex[5])
            assert policy_val.size >= 1
            assert abs(float(policy_val.sum()) - 1.0) < 1e-5
            assert np.all(policy_val >= 0.0)
            assert set(policy_idx.tolist()).issubset(set(legal_idx.tolist()))


def test_batched_exact_vs_mcts_value():
    """Exact-optimal endgame play yields different value targets than MCTS play.

    With the solver on, the endgame is played minimax-optimally; with it off, the
    same games' endgames are played by MCTS. They share the pre-endgame trajectory
    (same seed + evaluator) and diverge at the first deck=4 root, so at least one
    game reaches a different final score — hence a different value target z. The
    exact value is the game-theoretically correct one (we assert difference, not
    direction)."""
    kr = pytest.importorskip("kingdomino_rust")

    def run(max_nodes):
        m = kr.BatchedMCTS(
            n_slots=4, n_games=8, base_seed=0, n_sims=64, leaf_batch=6,
            open_loop=True, exact_endgame_max_nodes=max_nodes,
        )
        fin = _drive_batched(m)
        scores = {int(seed): (int(s[0]), int(s[1])) for seed, _ex, s in fin}
        return scores, m

    exact_scores, m_on = run(15_000_000)
    mcts_scores, m_off = run(0)
    assert m_on.exact_solve_count > 0
    assert m_off.exact_solve_count == 0
    assert exact_scores.keys() == mcts_scores.keys()
    differing = [k for k in exact_scores if exact_scores[k] != mcts_scores[k]]
    assert len(differing) >= 1, "exact endgame play never changed an outcome"


def test_batched_exact_fallback_count_zero():
    """Over 10 games, the 15M node budget solves every endgame root — no silent
    fallback to MCTS during the endgame."""
    kr = pytest.importorskip("kingdomino_rust")
    mcts = kr.BatchedMCTS(
        n_slots=4, n_games=10, base_seed=0, n_sims=64, leaf_batch=6,
        open_loop=True, exact_endgame_max_nodes=15_000_000,
    )
    _drive_batched(mcts)
    assert mcts.exact_solve_count > 0
    assert mcts.exact_tree_solve_count < mcts.exact_solve_count
    assert mcts.exact_fallback_count == 0
