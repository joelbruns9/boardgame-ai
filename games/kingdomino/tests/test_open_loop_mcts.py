"""
test_open_loop_mcts.py — Foundational invariant tests for open-loop MCTS.

These three tests are gates that must pass before any open-loop search
code is implemented (Phase 3.1). They prove the properties the open-loop
design depends on:

  TEST 1: ENCODE INVARIANCE
    encode_state(s, p) is byte-identical to encode_state(redeterminize(s), p)
    across 100 determinizations, for both players, mid-game state.
    Proves: deck order is not in the encoder output. Open-loop can resample
    freely without changing what the network sees.

  TEST 2: BAG INVARIANCE
    The bag feature slice of the flat vector is byte-identical across 100
    determinizations of the same state.
    Proves: the bag is computed from public observations (current_row,
    claims, placed tiles), not from deck order. Confirms one specific
    feature of TEST 1 in isolation.

  TEST 3: POLICY-TARGET PUBLIC-STATE INVARIANCE
    visit_counts_to_policy produces an identical policy vector whether
    called on the public state or any of its determinizations.
    Proves: the pick axis (current_row slot indices) is public at the root.
    Training targets extracted at the root are determinization-independent.

  TESTS 4-12 (Phase 3c): full correctness suite for OpenLoopMCTS.

IMPLEMENTATION FINDINGS (read from mcts_az.py before writing these tests):
  (a) search() returns visit_counts keyed by ENGINE ACTION OBJECTS
      (PickAction / TurnAction), built via decode_action(idx, root_state).
      Internally children are keyed by slot-relative joint-index ints; the
      conversion to action objects happens only in the return value. TEST 4
      asserts the returned keys are action objects.
  (b) decode_action is called with root_state (public) for the return value,
      and with each simulation's determinized state inside _simulate (stepping).
  (c) _select_child returns None when no stored child is legal in the current
      determinization; _simulate treats None as a dead-end (stops descending,
      evaluates the current state, does not re-expand).
  (d) The fallback counter is self._fallback_count — initialized in __init__
      and RESET to 0 at the start of every search(). TEST 11 therefore reads
      it after each search and accumulates.

  Two deliberate deviations from the spec pseudocode, for correctness:
   - TEST 8 checks legality by joint index (encode_action(a, s) in the legal
     index set), NOT `action in state.legal_actions()`. The latter
     false-positives on symmetric dominoes: decode_action canonicalises to
     flipped=False, which is != by object to the legal-list representative
     even though it is physically the same (and accepted by board.place).
   - TEST 10 enumerates the continuations of a near-terminal snapshot to pick
     one whose every line is a decisive win for the same player, so the
     0.0-evaluator average has an unambiguous sign and magnitude > 0.05.
"""
import numpy as np
import random
import time
import unittest.mock as mock

from games.kingdomino.game import (
    GameState, Phase, PickAction, TurnAction, determine_winner,
)
from games.kingdomino.encoder import compute_target_z, redeterminize, encode_state
from games.kingdomino.action_codec import encode_action, decode_action
from games.kingdomino.mcts_az import (
    OpenLoopMCTS, AlphaZeroMCTS, make_serial_evaluator,
    visit_counts_to_policy, select_move,
)


def test_encode_invariance():
    """encode_state byte-identical across 100 determinizations."""
    import random as py_random
    from games.kingdomino.game import GameState, Phase
    from games.kingdomino.encoder import encode_state, redeterminize

    # Advance to a mid-game PLACE_AND_SELECT state with claims and placements
    state = GameState.new(seed=42)
    rng = py_random.Random(42)
    steps = 0
    while state.phase == Phase.INITIAL_SELECTION or steps < 6:
        state = state.step(rng.choice(state.legal_actions()))
        steps += 1
    # state is now mid-game

    ref_mb0, ref_ob0, ref_flat0 = encode_state(state, player=0)
    ref_mb1, ref_ob1, ref_flat1 = encode_state(state, player=1)

    det_rng = py_random.Random(999)
    for i in range(100):
        det = redeterminize(state, det_rng)
        mb0, ob0, flat0 = encode_state(det, player=0)
        mb1, ob1, flat1 = encode_state(det, player=1)

        assert np.array_equal(mb0, ref_mb0),   f"my_board p0 differs at det {i}"
        assert np.array_equal(ob0, ref_ob0),   f"opp_board p0 differs at det {i}"
        assert np.array_equal(flat0, ref_flat0), f"flat p0 differs at det {i}"
        assert np.array_equal(mb1, ref_mb1),   f"my_board p1 differs at det {i}"
        assert np.array_equal(ob1, ref_ob1),   f"opp_board p1 differs at det {i}"
        assert np.array_equal(flat1, ref_flat1), f"flat p1 differs at det {i}"

    print("TEST 1 PASS: encode_state byte-identical across 100 determinizations")


def test_bag_invariance():
    """Bag feature slice byte-identical across 100 determinizations."""
    import random as py_random
    from games.kingdomino.game import GameState, Phase
    from games.kingdomino.encoder import encode_state, redeterminize, FLAT_LAYOUT

    state = GameState.new(seed=7)
    rng = py_random.Random(7)
    steps = 0
    while state.phase == Phase.INITIAL_SELECTION or steps < 8:
        state = state.step(rng.choice(state.legal_actions()))
        steps += 1

    _, _, ref_flat = encode_state(state, player=0)
    bag_slice = FLAT_LAYOUT['bag']
    ref_bag = ref_flat[bag_slice].copy()

    det_rng = py_random.Random(12345)
    for i in range(100):
        det = redeterminize(state, det_rng)
        _, _, flat = encode_state(det, player=0)
        assert np.array_equal(flat[bag_slice], ref_bag), \
            f"bag feature differs at det {i}: got {flat[bag_slice]}, expected {ref_bag}"

    # Also confirm the bag is non-trivial (not all zeros/ones — mid-game)
    n_in_bag = int(ref_bag.sum())
    assert 0 < n_in_bag < 48, \
        f"Expected a mid-game bag (0 < tiles < 48), got {n_in_bag}"

    print(f"TEST 2 PASS: bag feature ({n_in_bag} tiles) byte-identical "
          f"across 100 determinizations")


def test_policy_target_invariance():
    """visit_counts_to_policy identical on public state vs any determinization.

    This proves that the pick axis encodes SLOT POSITION in current_row
    (public at the root), not concrete domino_id (which varies across
    determinizations). A training target extracted at the root is
    therefore determinization-independent.
    """
    import random as py_random
    from games.kingdomino.game import GameState, Phase
    from games.kingdomino.encoder import encode_state, redeterminize
    from games.kingdomino.action_codec import encode_action
    from games.kingdomino.mcts_az import visit_counts_to_policy

    # Use INITIAL_SELECTION phase: pure pick actions, maximum sensitivity
    # to any domino-id vs slot-id confusion in the pick axis.
    state = GameState.new(seed=13)
    assert state.phase == Phase.INITIAL_SELECTION, \
        "Expected INITIAL_SELECTION at game start"

    # Build a synthetic visit_counts dict using legal actions
    legal = state.legal_actions()
    visit_counts = {a: (i + 1) * 10 for i, a in enumerate(legal)}

    ref_policy = visit_counts_to_policy(visit_counts, state, temperature=1.0)
    assert abs(ref_policy.sum() - 1.0) < 1e-5, "reference policy doesn't sum to 1"

    det_rng = py_random.Random(77)
    for i in range(100):
        det = redeterminize(state, det_rng)
        # visit_counts_to_policy calls encode_action(a, state) internally;
        # confirm it uses `state` (the public state), not `det`.
        # Re-encode using the determinized state to test invariance:
        det_visit_counts = {a: (j + 1) * 10 for j, a in enumerate(det.legal_actions())}
        det_policy = visit_counts_to_policy(det_visit_counts, det, temperature=1.0)

        assert np.array_equal(ref_policy, det_policy), \
            (f"Policy target differs at det {i}.\n"
             f"ref nonzero at: {np.nonzero(ref_policy)[0].tolist()}\n"
             f"det nonzero at: {np.nonzero(det_policy)[0].tolist()}\n"
             "This indicates pick encoding is NOT slot-relative — "
             "a domino_id leak would make open-loop incorrect.")

    # PLACE_AND_SELECT phase: confirm invariance holds with placement + pick
    state2 = GameState.new(seed=5)
    rng2 = py_random.Random(5)
    while state2.phase == Phase.INITIAL_SELECTION:
        state2 = state2.step(rng2.choice(state2.legal_actions()))
    assert state2.phase == Phase.PLACE_AND_SELECT

    legal2 = state2.legal_actions()
    vc2 = {a: (i + 1) for i, a in enumerate(legal2)}
    ref_policy2 = visit_counts_to_policy(vc2, state2, temperature=1.0)

    for i in range(100):
        det2 = redeterminize(state2, det_rng)
        det_legal2 = det2.legal_actions()
        det_vc2 = {a: (j + 1) for j, a in enumerate(det_legal2)}
        det_policy2 = visit_counts_to_policy(det_vc2, det2, temperature=1.0)
        assert np.array_equal(ref_policy2, det_policy2), \
            f"PLACE_AND_SELECT policy differs at det {i}"

    print("TEST 3 PASS: visit_counts_to_policy identical across "
          "100 determinizations (INITIAL_SELECTION + PLACE_AND_SELECT)")


# ─── shared helpers for the Phase 3c suite ──────────────────────────────────
def _tiny_net():
    """A small, deterministic network shared across the search tests (fast)."""
    import torch
    from games.kingdomino.network import KingdominoNet
    if not hasattr(_tiny_net, "_net"):
        torch.manual_seed(0)
        _tiny_net._net = KingdominoNet(channels=16, blocks=2, bilinear_dim=16).eval()
    return _tiny_net._net


def _advance(seed, extra_steps=0):
    """Fresh game → past INITIAL_SELECTION → `extra_steps` more random plies."""
    state = GameState.new(seed=seed)
    rng = random.Random(seed)
    while state.phase == Phase.INITIAL_SELECTION:
        state = state.step(rng.choice(state.legal_actions()))
    for _ in range(extra_steps):
        if state.phase == Phase.GAME_OVER:
            break
        state = state.step(rng.choice(state.legal_actions()))
    return state


def _advance_to_phase(seed, target_phase):
    """Play random moves until `target_phase` (or GAME_OVER) is reached."""
    state = GameState.new(seed=seed)
    rng = random.Random(seed)
    while state.phase != target_phase and state.phase != Phase.GAME_OVER:
        state = state.step(rng.choice(state.legal_actions()))
    return state


def test_visit_counts_key_type():
    """visit_counts keys are engine action objects, compatible with callers.

    Verifies: (a) keys are PickAction or TurnAction (not int), (b)
    visit_counts_to_policy accepts them without error, (c) select_move accepts
    them without error, (d) the policy vector sums to 1.0, (e) the chosen move
    is legal in the original state.
    """
    ev = make_serial_evaluator(_tiny_net())

    s_init = GameState.new(seed=3)
    assert s_init.phase == Phase.INITIAL_SELECTION, "expected INITIAL_SELECTION"
    s_place = _advance(5, extra_steps=2)
    assert s_place.phase == Phase.PLACE_AND_SELECT, "expected PLACE_AND_SELECT"

    for label, state in (("INITIAL_SELECTION", s_init),
                         ("PLACE_AND_SELECT", s_place)):
        mcts = OpenLoopMCTS(ev, n_simulations=30)
        vc, root = mcts.search(state, rng=np.random.default_rng(0))

        assert vc, f"{label}: empty visit_counts"
        assert all(isinstance(a, (PickAction, TurnAction)) for a in vc), \
            (f"{label}: visit_counts keys must be action objects, got "
             f"{[type(a).__name__ for a in vc][:3]}")

        pol = visit_counts_to_policy(vc, state, temperature=1.0)
        assert abs(pol.sum() - 1.0) < 1e-5, f"{label}: policy must sum to 1"

        chosen = select_move(vc, 0.0, np.random.default_rng(0))
        assert isinstance(chosen, (PickAction, TurnAction)), \
            f"{label}: select_move must return an action object"

        legal_idxs = {encode_action(a, state) for a in state.legal_actions()}
        assert encode_action(chosen, state) in legal_idxs, \
            f"{label}: chosen action is not legal in the original state"

    print("TEST 4 PASS: visit_counts keys are action objects; "
          "visit_counts_to_policy / select_move accept them (sum=1, move legal)")


def test_determinism():
    """Fixed seed → identical visit_counts across two independent searches."""
    ev = make_serial_evaluator(_tiny_net())
    state = _advance(42, extra_steps=4)
    assert state.phase == Phase.PLACE_AND_SELECT

    mcts = OpenLoopMCTS(ev, n_simulations=50)
    vc_a, root_a = mcts.search(state, rng=np.random.default_rng(7))
    vc_b, root_b = mcts.search(state, rng=np.random.default_rng(7))

    assert vc_a == vc_b, f"non-deterministic visit_counts:\n{vc_a}\n!=\n{vc_b}"
    assert root_a.visit_count == root_b.visit_count == 51, \
        f"root visits {root_a.visit_count}/{root_b.visit_count}, expected 51"
    print("TEST 5 PASS: determinism (n_sims=50) — identical visit_counts, "
          "root.visit_count=51")


def test_root_visit_count():
    """root.visit_count == n_simulations + 1 after search."""
    ev = make_serial_evaluator(_tiny_net())
    for n in (10, 25, 50):
        state = _advance(200 + n, extra_steps=2)  # fresh state each time
        mcts = OpenLoopMCTS(ev, n_simulations=n)
        vc, root = mcts.search(state, rng=np.random.default_rng(1))
        assert root.visit_count == n + 1, \
            f"n_sims={n}: root.visit_count={root.visit_count}, expected {n + 1}"
    print("TEST 6 PASS: root.visit_count == n_simulations + 1 for n in {10,25,50}")


def test_value_range():
    """All backed-up Q values are in [-1, 1] after 100 simulations."""
    ev = make_serial_evaluator(_tiny_net())
    state = _advance(11, extra_steps=2)
    mcts = OpenLoopMCTS(ev, n_simulations=100)
    vc, root = mcts.search(state, rng=np.random.default_rng(2))

    all_nodes = []
    queue = [root]
    while queue:                       # BFS/DFS over the whole tree
        node = queue.pop()
        all_nodes.append(node)
        queue.extend(node.children.values())

    visited = [n for n in all_nodes if n.visit_count > 0]
    for n in visited:
        assert -1.0 <= n.value <= 1.0, f"node.value {n.value} out of [-1, 1]"
    assert len(all_nodes) >= 10, \
        f"expected >= 10 nodes (tree should expand), got {len(all_nodes)}"
    print(f"TEST 7 PASS: all {len(visited)} visited node values in [-1, 1] "
          f"({len(all_nodes)} nodes total)")


# TEST 8 and TEST 11 share one 5-seed x 200-sim pass (1000 sims), computed once.
_SAFETY_CACHE = {}


def _safety_fallback_pass():
    """Run 5 seeds x 200 sims once; return (violations, fallbacks, selections).

    Monkey-patches GameState.step (only inside each search) to verify every
    stepped action is legal in its concrete state — by JOINT INDEX, which is
    robust to symmetric-domino canonicalisation. Also accumulates the per-
    search _fallback_count and counts total descent steps (= successful
    selections) so the fallback rate can be computed.
    """
    if "result" in _SAFETY_CACHE:
        return _SAFETY_CACHE["result"]

    net = _tiny_net()
    seeds = [0, 7, 13, 42, 99]
    sims = 200
    original_step = GameState.step
    step_count = [0]
    violations = []

    def checked_step(self, action):
        step_count[0] += 1
        try:
            legal_idxs = {encode_action(a, self) for a in self.legal_actions()}
            if encode_action(action, self) not in legal_idxs:
                violations.append((self.phase.name, str(action)))
        except Exception as e:   # encode raised → action references a missing tile
            violations.append((self.phase.name, str(action), repr(e)))
        return original_step(self, action)

    total_fallbacks = 0
    for seed in seeds:
        state = GameState.new(seed=seed)
        rng_setup = random.Random(seed)
        while state.phase == Phase.INITIAL_SELECTION:   # setup OUTSIDE the patch
            state = state.step(rng_setup.choice(state.legal_actions()))
        mcts = OpenLoopMCTS(make_serial_evaluator(net), n_simulations=sims)
        with mock.patch.object(GameState, "step", checked_step):
            mcts.search(state, rng=np.random.default_rng(seed))
        total_fallbacks += mcts._fallback_count

    total_selections = step_count[0] + total_fallbacks
    _SAFETY_CACHE["result"] = (violations, total_fallbacks, total_selections)
    return _SAFETY_CACHE["result"]


def test_legal_action_safety():
    """No simulation steps into an action illegal in its concrete state."""
    t0 = time.time()
    violations, _, total_selections = _safety_fallback_pass()
    assert len(violations) == 0, (
        f"{len(violations)} illegal steps across 1000 sims: "
        f"{violations[:3]}{'...' if len(violations) > 3 else ''}"
    )
    print(f"TEST 8 PASS: 0 illegal steps across 1000 simulations "
          f"(5 seeds x 200 sims, {total_selections} selections, "
          f"{time.time() - t0:.1f}s)")


def test_degenerate_bag():
    """With an empty (degenerate) bag, open-loop == closed-loop top action.

    At FINAL_PLACEMENT the deck is empty, so there is exactly one deck
    "permutation": redeterminize is a no-op and open-loop searches the same
    deterministic tree as closed-loop. The greedy top action must agree.
    """
    ev = make_serial_evaluator(_tiny_net())

    fp = None
    for seed in range(40):
        s = _advance_to_phase(seed, Phase.FINAL_PLACEMENT)
        if s.phase == Phase.FINAL_PLACEMENT and len(s.legal_actions()) >= 2:
            fp = s
            break
    assert fp is not None, "could not reach a FINAL_PLACEMENT state with >=2 actions"
    assert len(fp.deck) == 0, "FINAL_PLACEMENT bag should be empty (degenerate)"

    az = AlphaZeroMCTS(ev, n_simulations=30)
    vc_az, _ = az.search(fp, rng=np.random.default_rng(0))
    top_az = select_move(vc_az, 0.0, np.random.default_rng(0))

    # Exact endgame solving is disabled here so the comparison is apples-to-apples:
    # this test checks the open-loop SEARCH MACHINERY matches closed-loop at a
    # degenerate (empty-bag) tree. With exact solving on, open-loop replaces the
    # network leaf value with the exact minimax score at deck=0 leaves, while
    # AlphaZeroMCTS has no exact solver — so they would legitimately diverge.
    # Exact-solver equivalence is covered separately in test_endgame_exact.py.
    ol = OpenLoopMCTS(ev, n_simulations=30, exact_endgame_enabled=False)
    vc_ol, _ = ol.search(fp, rng=np.random.default_rng(0))
    top_ol = select_move(vc_ol, 0.0, np.random.default_rng(0))

    assert encode_action(top_az, fp) == encode_action(top_ol, fp), (
        f"degenerate bag: top action mismatch — closed-loop "
        f"{encode_action(top_az, fp)} vs open-loop {encode_action(top_ol, fp)}"
    )
    print("TEST 9 PASS: degenerate (empty) bag — open-loop top action == "
          "closed-loop top action")


def test_win_frame_backup():
    """Terminal win backs up with the correct sign in the player-0 frame.

    Uses a 0.0 mock evaluator (the only value signal is terminal backups) and
    a last-decision snapshot whose EVERY continuation is a decisive win for the
    same player. Asserts root.value sign matches that known winner.
    """
    snap = None
    sign = 0
    for seed in range(120):
        state = GameState.new(seed=seed)
        rng = random.Random(seed)
        path = [state]
        while state.phase != Phase.GAME_OVER:
            state = state.step(rng.choice(state.legal_actions()))
            path.append(state)
        if len(path) < 2:
            continue
        s = path[-2]                          # the final decision (1 ply to end)
        if s.phase == Phase.GAME_OVER:
            continue
        children = [s.step(a) for a in s.legal_actions()]
        # require an all-terminal, decisively one-sided final decision
        if not children or any(c.phase != Phase.GAME_OVER for c in children):
            continue
        zs = [compute_target_z(c, 0) for c in children]
        if all(z > 0.10 for z in zs):
            snap, sign = s, +1
            break
        if all(z < -0.10 for z in zs):
            snap, sign = s, -1
            break
    assert snap is not None, "no decisive last-decision snapshot found in 120 seeds"

    known = 0 if sign > 0 else 1
    winner = determine_winner(snap.step(snap.legal_actions()[0]))
    assert winner == known, f"winner {winner} != expected player {known}"

    def zero_eval(mb, ob, flat, idxs):
        return 0.0, np.zeros(len(idxs), dtype=np.float32)

    mcts = OpenLoopMCTS(zero_eval, n_simulations=20)
    vc, root = mcts.search(snap, rng=np.random.default_rng(0))

    assert root.visit_count == 21, f"root.visit_count={root.visit_count}, expected 21"
    assert -1.0 < root.value < 1.0, f"root.value {root.value} out of (-1, 1)"
    if sign > 0:
        assert root.value > 0.05, \
            f"player 0 won but root.value={root.value:+.3f} (expected > 0.05)"
    else:
        assert root.value < -0.05, \
            f"player 1 won but root.value={root.value:+.3f} (expected < -0.05)"
    print(f"TEST 10 PASS: win-frame backup — winner=P{known}, "
          f"root.value={root.value:+.3f} (sign matches), root.visit_count=21")


def test_fallback_rate():
    """Legal-action fallback rate < 1% across 1000 simulations."""
    violations, total_fallbacks, total_selections = _safety_fallback_pass()
    rate = total_fallbacks / max(1, total_selections)
    assert rate < 0.01, (
        f"fallback rate {rate:.4%} >= 1% "
        f"({total_fallbacks} fallbacks / {total_selections} selections)"
    )
    print(f"TEST 11 PASS: fallback rate {rate:.4%} "
          f"({total_fallbacks}/{total_selections}) < 1%")


def test_dirichlet_noise():
    """Root child priors sum to 1.0 after Dirichlet noise is applied."""
    ev = make_serial_evaluator(_tiny_net())
    state = _advance(8, extra_steps=2)
    mcts = OpenLoopMCTS(ev, n_simulations=30)
    vc, root = mcts.search(state, add_noise=True, rng=np.random.default_rng(0))

    priors = [c.prior for c in root.children.values()]
    prior_sum = sum(priors)
    assert abs(prior_sum - 1.0) < 1e-5, \
        f"root priors sum to {prior_sum}, expected 1.0"
    assert all(p > 0 for p in priors), "all root priors must be > 0 after noise"
    print(f"TEST 12 PASS: root priors sum to {prior_sum:.6f} after Dirichlet "
          f"noise; all {len(priors)} priors > 0")


def test_d4_symmetry_consistency():
    """Open-loop search consistency gate (promoted from a Phase 3c diagnostic).

    APPROACH — documented tradeoff (8-seed search-consistency, not strict D4
    equivariance):
      The rigorous D4 test would search the same position in all 8 orientations
      and require inverse-transformed visit distributions to agree.  But that
      only holds if the EVALUATOR is D4-equivariant, and architectural
      equivariance is *learned* via augmentation training — it is NOT a property
      of an untrained net or of a uniform mock.  The prescribed uniform mock is
      trivially rotation-INVARIANT, so transforming its inputs changes nothing
      and the "8 orientations" collapse to the SAME search under different RNG
      seeds.  We therefore implement that directly: run open-loop search on one
      fixed mid-game state with 8 fixed seeds and require the visit
      distributions to agree (pairwise total variation < 0.10).  This is the
      conservative open-loop noise floor (redeterminization adds stochasticity)
      and catches gross search asymmetries / nondeterminism.  Strict network
      D4-equivariance is covered by tests/test_augmentation.py and can be
      promoted to a full equivariance gate once a D4-trained net is available.
    """
    state = _advance(42, extra_steps=2)
    assert state.phase == Phase.PLACE_AND_SELECT, "expected PLACE_AND_SELECT"
    legal = state.legal_actions()
    assert len(legal) >= 10, f"need >= 10 legal actions for a non-trivial test, got {len(legal)}"

    # Uniform mock evaluator: removes network signal so only search (PUCT +
    # per-simulation redeterminization) variance remains.
    def mock_eval(mb, ob, flat, idxs):
        return 0.0, np.zeros(len(idxs), dtype=np.float32)

    policies = []
    for seed in range(8):
        mcts = OpenLoopMCTS(mock_eval, n_simulations=200)
        vc, _ = mcts.search(state, add_noise=False,
                            rng=np.random.default_rng(seed))
        policies.append(visit_counts_to_policy(vc, state, temperature=1.0))

    def tv(p, q):
        return 0.5 * float(np.abs(p - q).sum())

    max_tv, worst = 0.0, None
    for i in range(8):
        for j in range(i + 1, 8):
            d = tv(policies[i], policies[j])
            if d > max_tv:
                max_tv, worst = d, (i, j)

    # Diagnostics (not asserted): top-action agreement + entropy spread.
    tops = [int(np.argmax(p)) for p in policies]
    top_agree = max(tops.count(t) for t in set(tops)) / len(tops)
    ents = [float(-(p[p > 0] * np.log(p[p > 0])).sum()) for p in policies]

    assert max_tv < 0.10, (
        f"max pairwise TV {max_tv:.3f} >= 0.10 (pair {worst}) — open-loop search "
        f"is inconsistent across seeds beyond the redeterminization floor.")
    print(f"TEST 13 PASS: open-loop search consistency — max pairwise TV="
          f"{max_tv:.3f} < 0.10 (8 seeds x 200 sims), top-action agreement="
          f"{top_agree:.0%}, entropy {min(ents):.2f}-{max(ents):.2f}  "
          f"[8-seed approx; strict D4 equivariance: test_augmentation.py]")


def test_duplicate_node_separate_values():
    """Issue 1: two simulations reaching the same OLNode under different
    determinizations get INDEPENDENT eval values (not the first's, deduped).

    Drives the real Rust BatchedMCTS open-loop path with leaf_batch=2 at the
    INITIAL_SELECTION root.  Both sims descend the same (unchanged) tree this
    tick and tie on child 0 (uniform priors), so they collide on one OLNode but
    with DIFFERENT decks → different concrete states.  With the de-dup bug the
    tick would emit ONE eval row and back up one value twice; with the fix it
    emits TWO rows and the shared child's value_sum is the sum of BOTH distinct
    values.
    """
    import kingdomino_rust as kr

    # sims=4 (> leaf_batch) so the first Searching tick does NOT finalize the
    # move (which would clear the arena before we can inspect it).
    # virtual_loss=0: VL between descents would deliberately SPREAD the two sims
    # to different children (its whole purpose), so we disable it to force the
    # collision this test is about — two sims on the SAME node, different decks.
    b = kr.BatchedMCTS(1, 1, 0, 4, leaf_batch=2, virtual_loss=0, cpuct=1.5,
                       fpu=0.0, dirichlet_alpha=0.3, dirichlet_eps=0.0,
                       temp_moves=0, open_loop=True)

    def run_mock(mb, ob, flat, idxs):
        k = int(np.asarray(mb).shape[0])
        # Distinct per-row values so colliding sims get different evaluations.
        vals = np.array([0.2 + 0.4 * i for i in range(k)], dtype=np.float32)
        gathered = [np.zeros(len(np.asarray(idxs[i])), dtype=np.float32)
                    for i in range(k)]
        return vals, gathered

    # Tick 1 — root eval (exactly one row).
    mb, ob, flat, idxs = b.step()
    assert int(mb.shape[0]) == 1, f"root eval should be 1 row, got {mb.shape[0]}"
    vals, gathered = run_mock(mb, ob, flat, idxs)
    b.update(vals, gathered)

    # Tick 2 — Searching with leaf_batch=2: both sims collide on child 0.
    mb, ob, flat, idxs = b.step()
    n_rows = int(mb.shape[0])
    vals, gathered = run_mock(mb, ob, flat, idxs)
    b.update(vals, gathered)

    # Core de-dup assertion: 2 simulations → 2 eval rows (bug would give 1).
    assert n_rows == 2, (
        f"de-dup not removed: expected 2 eval rows for 2 colliding sims, "
        f"got {n_rows}")

    # Child node 1 (root's first child) was reached by BOTH sims; its value_sum
    # is the sum of two DISTINCT backed-up values.  The two raw values 0.2 and
    # 0.6 share the child's actor sign, so |value_sum| == 0.8 (fix), not 0.4
    # (= 2*0.2, which the de-dup bug would have produced).
    vc, vsum = b.debug_ol_node(0, 1)
    assert vc == 2, f"child reached by both sims should have visit_count 2, got {vc}"
    assert abs(abs(vsum) - 0.8) < 1e-5, (
        f"child value_sum {vsum:+.4f}: expected |0.8| (two distinct values "
        f"0.2+0.6), not |0.4| (de-dup would reuse 0.2 twice)")

    print(f"TEST 14 PASS: duplicate-node separate values — 2 colliding sims "
          f"emitted {n_rows} rows, child value_sum |{abs(vsum):.3f}| = 0.2+0.6 "
          f"(distinct, not deduped)")


def test_missing_child_expansion():
    """Issue 2: the missing-child diagnostic is exposed and the root stays
    complete (every root-legal action appears in the tree's children).

    Root current_row is public, so all determinizations share the root's legal
    set and it is fully expanded — this checks that invariant and that the
    fallback/missing-child diagnostics are Python-readable.  The deterministic
    missing-child INSERT logic (a deep node gaining children from a later
    determinization) is covered rigorously by the Rust unit test
    `ol_tests::issue2_missing_child_insert`.
    """
    import kingdomino_rust as kr
    from games.kingdomino.game import GameState

    state = GameState.new(seed=0)
    root_legal = {kr_idx for kr_idx in
                  {encode_action(a, state) for a in state.legal_actions()}}

    b = kr.BatchedMCTS(1, 1, 0, 30, leaf_batch=6, virtual_loss=1, cpuct=1.5,
                       fpu=0.0, dirichlet_alpha=0.3, dirichlet_eps=0.0,
                       temp_moves=0, open_loop=True)

    # Diagnostics exposed as Python-readable attributes (gate requirement).
    assert isinstance(b.fallback_count, int) and b.fallback_count >= 0
    assert isinstance(b.missing_child_count, int) and b.missing_child_count >= 0

    first_example = None
    while not b.done():
        mb, ob, flat, idxs = b.step()
        k = int(mb.shape[0])
        vals = np.zeros(k, dtype=np.float32)
        gathered = [np.zeros(len(np.asarray(idxs[i])), dtype=np.float32)
                    for i in range(k)]
        finished = b.update(vals, gathered)
        for seed, examples, scores in finished:
            if first_example is None and examples:
                first_example = examples[0]

    assert first_example is not None, "no examples emitted"
    # legal_idx is field 5 of the 10-tuple; at move 0 it must cover every
    # root-legal joint index (root fully expanded — no missing children).
    legal_idx = set(int(x) for x in np.asarray(first_example[5]))
    assert root_legal <= legal_idx, (
        f"root children incomplete: missing {sorted(root_legal - legal_idx)[:5]}")
    assert isinstance(b.missing_child_count, int) and b.missing_child_count >= 0

    print(f"TEST 15 PASS: missing-child diagnostic exposed "
          f"(fallback={b.fallback_count}, missing={b.missing_child_count}); "
          f"root complete ({len(root_legal)} legal actions all present)")


def test_fallback_no_illegal_step():
    """Issue 2 safety: deep open-loop search never steps an action illegal in
    its concrete state, even when descent dead-ends / stops to add children."""
    net = _tiny_net()
    original_step = GameState.step
    violations = []

    def checked_step(self, action):
        try:
            legal_idxs = {encode_action(a, self) for a in self.legal_actions()}
            if encode_action(action, self) not in legal_idxs:
                violations.append((self.phase.name, str(action)))
        except Exception as e:
            violations.append((self.phase.name, str(action), repr(e)))
        return original_step(self, action)

    fallbacks = 0
    for seed in (1, 23, 77):
        state = _advance(seed, extra_steps=4)
        if state.phase == Phase.GAME_OVER:
            continue
        mcts = OpenLoopMCTS(make_serial_evaluator(net), n_simulations=200)
        with mock.patch.object(GameState, "step", checked_step):
            mcts.search(state, rng=np.random.default_rng(seed))
        fallbacks += mcts._fallback_count

    assert len(violations) == 0, (
        f"{len(violations)} illegal steps in deep open-loop search: "
        f"{violations[:3]}")
    print(f"TEST 16 PASS: fallback no-illegal-step — 0 illegal steps across "
          f"3 deep searches (200 sims each), {fallbacks} fallbacks all handled "
          f"as leaves")


def test_win_target_perspective():
    """Issue/TEST 17: finalize_move sets own_score/opp_score/win_target with the
    correct per-actor perspective flip.

    For a finished game with scores (s0, s1): every example has own+opp == s0+s1,
    win is consistent with own vs opp (>:1.0, <:0.0, ==:0.5), and on a decisive
    game BOTH perspectives appear with complementary win targets.
    """
    import kingdomino_rust as kr

    b = kr.BatchedMCTS(4, 6, 0, 30, leaf_batch=6, virtual_loss=1, cpuct=1.5,
                       fpu=0.0, dirichlet_alpha=0.3, dirichlet_eps=0.0,
                       temp_moves=0, open_loop=True)
    games = []
    while not b.done():
        mb, ob, flat, idxs = b.step()
        k = int(mb.shape[0])
        vals = np.zeros(k, dtype=np.float32)
        gathered = [np.zeros(len(np.asarray(idxs[i])), dtype=np.float32)
                    for i in range(k)]
        games.extend(b.update(vals, gathered))

    assert games, "no finished games"
    decisive_checked = 0
    for seed, examples, scores in games:
        s0, s1 = int(scores[0]), int(scores[1])
        owns = set()
        for ex in examples:
            own, opp, win = float(ex[8]), float(ex[9]), float(ex[10])
            assert abs((own + opp) - (s0 + s1)) < 1e-4, (
                f"own+opp={own + opp} != s0+s1={s0 + s1}")
            assert own in (float(s0), float(s1)), f"own {own} not in scores {scores}"
            if own > opp:
                assert win == 1.0, f"own>opp but win={win}"
            elif own < opp:
                assert win == 0.0, f"own<opp but win={win}"
            else:
                assert win == 0.5, f"own==opp but win={win}"
            owns.add(own)
        if s0 != s1:
            # Decisive game: both perspectives must be present with complementary
            # win targets (one actor's examples win=1.0, the other's win=0.0).
            wins = {float(ex[8]): float(ex[10]) for ex in examples}
            assert wins.get(float(max(s0, s1))) == 1.0, "winner perspective win!=1.0"
            assert wins.get(float(min(s0, s1))) == 0.0, "loser perspective win!=0.0"
            decisive_checked += 1

    assert decisive_checked >= 1, "expected at least one decisive game to check"
    print(f"TEST 17 PASS: win-target perspective correct on {len(games)} games "
          f"({decisive_checked} decisive); own/opp/win consistent + complementary")


def test_final_placement_ol_descent():
    """TEST 18: open-loop search in FINAL_PLACEMENT (empty pick axis) completes
    cleanly — all legal actions have pick=None and need no current_row entry."""
    ev = make_serial_evaluator(_tiny_net())
    fp = None
    for seed in range(60):
        s = _advance_to_phase(seed, Phase.FINAL_PLACEMENT)
        if s.phase == Phase.FINAL_PLACEMENT and len(s.legal_actions()) >= 1:
            fp = s
            break
    assert fp is not None, "could not reach a FINAL_PLACEMENT state"
    assert len(fp.deck) == 0, "FINAL_PLACEMENT deck should be empty"

    mcts = OpenLoopMCTS(ev, n_simulations=20)
    vc, root = mcts.search(fp, rng=np.random.default_rng(0))
    assert root.visit_count == 21, \
        f"root.visit_count={root.visit_count}, expected 21"
    assert vc, "FINAL_PLACEMENT search produced empty visit_counts"
    chosen = select_move(vc, 0.0, np.random.default_rng(0))
    assert isinstance(chosen, (PickAction, TurnAction))
    print(f"TEST 18 PASS: FINAL_PLACEMENT ol-descent — root.visit_count=21, "
          f"{len(vc)} legal actions, no empty-row exception")


def test_codec_phase_shapes():
    """TEST 19: encode_action rejects phase-mismatched actions.

    PickAction is only legal in INITIAL_SELECTION; TurnAction only in
    PLACE_AND_SELECT / FINAL_PLACEMENT.  encode_action must raise ValueError on
    the wrong phase rather than silently producing a bogus index.
    """
    s_init = GameState.new(seed=3)
    assert s_init.phase == Phase.INITIAL_SELECTION
    s_place = _advance(5, extra_steps=2)
    assert s_place.phase == Phase.PLACE_AND_SELECT

    # TurnAction in INITIAL_SELECTION → ValueError.
    try:
        encode_action(TurnAction(placement=None, pick_domino_id=None), s_init)
        raised_turn = False
    except ValueError:
        raised_turn = True
    assert raised_turn, "TurnAction in INITIAL_SELECTION should raise ValueError"

    # PickAction in PLACE_AND_SELECT → ValueError.
    some_domino = s_place.current_row[0]
    try:
        encode_action(PickAction(domino_id=some_domino), s_place)
        raised_pick = False
    except ValueError:
        raised_pick = True
    assert raised_pick, "PickAction in PLACE_AND_SELECT should raise ValueError"

    # A correctly-phased action round-trips.
    legal0 = s_init.legal_actions()[0]
    idx = encode_action(legal0, s_init)
    assert isinstance(idx, int) and 0 <= idx < 3390
    back = decode_action(idx, s_init)
    assert isinstance(back, (PickAction, TurnAction))

    print("TEST 19 PASS: codec phase-shape invariants — wrong-phase actions "
          "raise ValueError; correctly-phased action round-trips")


if __name__ == "__main__":
    import sys

    tests = [
        ("TEST 1: Encode invariance", test_encode_invariance),
        ("TEST 2: Bag invariance", test_bag_invariance),
        ("TEST 3: Policy-target public-state invariance", test_policy_target_invariance),
        ("TEST 4: visit_counts key type and caller compatibility", test_visit_counts_key_type),
        ("TEST 5: Determinism", test_determinism),
        ("TEST 6: root.visit_count == n_simulations + 1", test_root_visit_count),
        ("TEST 7: Value range", test_value_range),
        ("TEST 8: Legal-action safety (1000 sims)", test_legal_action_safety),
        ("TEST 9: Degenerate bag", test_degenerate_bag),
        ("TEST 10: Win frame through backup", test_win_frame_backup),
        ("TEST 11: Fallback rate < 1%", test_fallback_rate),
        ("TEST 12: Dirichlet noise", test_dirichlet_noise),
        ("TEST 13: D4 / search consistency", test_d4_symmetry_consistency),
        ("TEST 14: Duplicate node, separate values (Issue 1)", test_duplicate_node_separate_values),
        ("TEST 15: Missing-child diagnostic + root completeness (Issue 2)", test_missing_child_expansion),
        ("TEST 16: Fallback no illegal step", test_fallback_no_illegal_step),
        ("TEST 17: Win-target perspective", test_win_target_perspective),
        ("TEST 18: FINAL_PLACEMENT ol-descent", test_final_placement_ol_descent),
        ("TEST 19: Codec phase-shape invariants", test_codec_phase_shapes),
    ]

    t_start = time.time()
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"FAIL {name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} PASS  {failed} FAIL  ({time.time() - t_start:.1f}s)")
    sys.exit(0 if failed == 0 else 1)
