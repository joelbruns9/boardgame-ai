"""Workstream 9: shared action statistics across chance siblings.

Two independent searcher mechanisms, both off by default:

* ``wonder_group_selection`` -- select the Wonder, then the burial target.
* ``chance_sibling_bias`` -- bias selection inside a chance world by how the
  same canonical action scored in that world's siblings.

The load-bearing test is the first one: with both flags off the searcher must
reproduce its previous output exactly, because everything already gated against
this searcher (Python/Rust parity, every recorded gate) assumes it. The rest
pin the invariants each mechanism claims -- that full-action statistics survive
grouping, and that sharing stays out of ``q_p0``.
"""

from __future__ import annotations

import json
import random

import pytest

from .codec import (
    CARD_TO_WONDER_BASE,
    DESTROY_BASE,
    NUM_WONDERS,
    decode_action,
    legal_action_indices,
)
from .engine import ActionUse, apply_action
from .game import Phase, new_game
from .inference import Evaluator
from .search import (
    GumbelMCTS,
    SearchConfig,
    canonical_action_group,
    structural_action_key,
)
from .train import build_model


@pytest.fixture(scope="module")
def model():
    return build_model("transformer", 32, 1)


def _positions(count: int = 6, *, min_plies: int = 8, max_plies: int = 30):
    """Mid-game PLAY_AGE positions from random playouts."""

    out = []
    for seed in range(40):
        if len(out) >= count:
            break
        game = new_game(seed, first_player=seed % 2)
        rng = random.Random(seed)
        for _ in range(rng.randrange(min_plies, max_plies)):
            if game.phase is Phase.COMPLETE:
                break
            apply_action(game, decode_action(game, rng.choice(legal_action_indices(game))))
        if game.phase is not Phase.COMPLETE:
            out.append(game)
    return out


def _search(model, state, *, sims=200, **flags):
    config = SearchConfig(sims=sims, top_k=4, mode="closed", seed=3, **flags)
    mcts = GumbelMCTS(Evaluator(model, "cpu"), config)
    return mcts.search(state), mcts


def _fingerprint(result) -> str:
    """Everything a caller can observe about a search."""

    return json.dumps(
        {
            "action": result.action_index,
            "visits": sorted(result.visits.items()),
            "policy": sorted((k, round(v, 12)) for k, v in result.policy_target.items()),
            "completed_q": sorted(
                (k, round(v, 12)) for k, v in result.completed_q.items()
            ),
            "sims": result.sims,
        },
        sort_keys=True,
    )


# -- the migration gate -----------------------------------------------------


def test_both_flags_off_reproduces_the_current_search_exactly(model):
    """The whole strength-preservation claim. Off is not "close enough": every
    recorded gate and the Python/Rust parity tests are anchored to this output,
    so an off-path difference of any size is a regression, not a new sample.
    """

    for state in _positions():
        baseline = _fingerprint(_search(model, state)[0])
        explicit = _fingerprint(
            _search(model, state, wonder_group_selection=False, chance_sibling_bias=0.0)[0]
        )
        assert baseline == explicit


def test_each_flag_alone_actually_changes_the_search(model):
    """Guards the test above from passing vacuously: if a flag were silently
    inert, off-equivalence would be trivially true and prove nothing."""

    changed_wonder = changed_sibling = 0
    states = _positions()
    for state in states:
        baseline = _fingerprint(_search(model, state)[0])
        changed_wonder += (
            _fingerprint(_search(model, state, wonder_group_selection=True)[0]) != baseline
        )
        changed_sibling += (
            _fingerprint(_search(model, state, chance_sibling_bias=1.0)[0]) != baseline
        )
    assert changed_wonder, "wonder_group_selection changed nothing anywhere"
    assert changed_sibling, "chance_sibling_bias changed nothing anywhere"


# -- the shared key ---------------------------------------------------------


def test_wonder_actions_group_by_wonder_not_by_buried_card():
    """The key mechanism 1 shares on and mechanism 2 groups by. Burial targets
    of one Wonder must collapse together; different Wonders must not."""

    a = CARD_TO_WONDER_BASE + 3 * NUM_WONDERS + 5  # card 3 -> wonder 5
    b = CARD_TO_WONDER_BASE + 9 * NUM_WONDERS + 5  # card 9 -> wonder 5
    c = CARD_TO_WONDER_BASE + 3 * NUM_WONDERS + 6  # card 3 -> wonder 6
    assert canonical_action_group(a) == canonical_action_group(b) == ("wonder", 5)
    assert canonical_action_group(c) == ("wonder", 6)


def test_non_wonder_actions_are_their_own_group():
    assert canonical_action_group(0) == ("action", 0)
    assert canonical_action_group(CARD_TO_WONDER_BASE - 1) == (
        "action", CARD_TO_WONDER_BASE - 1
    )
    assert canonical_action_group(DESTROY_BASE) == ("action", DESTROY_BASE)


def test_group_key_agrees_with_the_decoded_action(model):
    """The arithmetic shortcut must match what the codec actually decodes --
    it is used in selection precisely to avoid decoding."""

    for state in _positions(3):
        for index in legal_action_indices(state):
            action = decode_action(state, index)
            kind, value = canonical_action_group(index)
            if action.use is ActionUse.CONSTRUCT_WONDER:
                assert kind == "wonder"
            else:
                assert (kind, value) == ("action", index)
        # Two burial variants of one Wonder share a key; that is the case the
        # mechanisms exist for, so assert it on a real position when present.
        by_wonder: dict = {}
        for index in legal_action_indices(state):
            action = decode_action(state, index)
            if action.use is ActionUse.CONSTRUCT_WONDER:
                by_wonder.setdefault(action.wonder_name, []).append(index)
        for indices in by_wonder.values():
            keys = {canonical_action_group(i) for i in indices}
            assert len(keys) == 1


# -- mechanism 2 ------------------------------------------------------------


def test_grouping_preserves_every_full_action(model):
    """Grouping factors SELECTION only. Each full action keeps its own edge,
    its own visits and its original index, so the recorded policy target has
    the same shape it always had."""

    for state in _positions(3):
        flat, _ = _search(model, state)
        grouped, mcts = _search(model, state, wonder_group_selection=True)
        assert set(grouped.visits) == set(flat.visits) == set(legal_action_indices(state))
        assert set(grouped.policy_target) == set(flat.policy_target)
        assert sum(grouped.visits.values()) == sum(flat.visits.values())
        root = mcts._closed_root
        assert [edge.action_index for edge in root.edges] == list(root.legal)


def _burial_group_violations(root, *, root_only: bool) -> tuple[int, int]:
    """(violations, multi-variant Wonder groups) over one tree.

    A violation is a Wonder group holding an unexplored burial target while
    another has been visited more than once -- the "funding pooled on whichever
    variant went first" failure the within-group guarantee exists to prevent.
    """

    violations = groups = 0
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        by_key: dict = {}
        for edge in node.edges:
            by_key.setdefault(canonical_action_group(edge.action_index), []).append(edge)
            for child in edge.children.values():
                stack.append((child.node, depth + 1))
        if root_only != (depth == 0):
            continue
        for key, members in by_key.items():
            if key[0] != "wonder" or len(members) < 2:
                continue
            groups += 1
            unexplored = any(edge.visits == 0 for edge in members)
            if unexplored and max(edge.visits for edge in members) > 1:
                violations += 1
    return violations, groups


def test_grouping_gives_every_burial_target_a_look_before_any_gets_two(model):
    """The within-group guarantee, at the interior nodes ``_select_closed``
    governs. Without it the low-prior failure recurs one level down: the group
    gets funded and the funding pools on whichever variant happened to go
    first. Measured across the test positions, that happens on roughly 7-9% of
    multi-variant groups; with grouping on it must never happen.
    """

    for root_selection in ("gumbel", "puct"):
        off = on = None
        for flag in (False, True):
            totals = [0, 0]
            for state in _positions(6):
                _, mcts = _search(
                    model, state, sims=400,
                    root_selection=root_selection, wonder_group_selection=flag,
                )
                bad, groups = _burial_group_violations(
                    mcts._closed_root, root_only=False
                )
                totals[0] += bad
                totals[1] += groups
            if flag:
                on = totals
            else:
                off = totals
        assert off[1] and on[1], "no multi-variant Wonder group in any position"
        assert on[0] == 0, f"{root_selection}: {on[0]}/{on[1]} groups violated with grouping on"
        assert off[0] > 0, (
            f"{root_selection}: baseline violated nothing, so this test proves "
            "nothing about the guarantee"
        )


def test_grouping_does_not_reach_the_gumbel_root(model):
    """A scope limit worth pinning rather than rediscovering.

    The Gumbel root runs top-k + sequential halving and calls ``simulate`` on a
    chosen action directly, so root edges never pass through ``_select_closed``
    and grouping cannot reach them. This matters because visit-count policy
    targets are built from ROOT visits: under the training root, mechanism 2
    reshapes the interior search but not the target's root allocation.

    Under the PUCT root grouping does govern, but the guarantee already held
    there without it -- an unvisited edge's prior term is large enough on its
    own -- so the observable root behaviour is unchanged either way.
    """

    for root_selection, expect_equal in (("gumbel", True), ("puct", True)):
        seen = {}
        for flag in (False, True):
            totals = [0, 0]
            for state in _positions(6):
                _, mcts = _search(
                    model, state, sims=400,
                    root_selection=root_selection, wonder_group_selection=flag,
                )
                bad, groups = _burial_group_violations(
                    mcts._closed_root, root_only=True
                )
                totals[0] += bad
                totals[1] += groups
            seen[flag] = tuple(totals)
        assert (seen[False] == seen[True]) is expect_equal, (
            f"{root_selection}: root behaviour changed {seen[False]} -> {seen[True]}"
        )


# -- mechanism 1 ------------------------------------------------------------


def test_sibling_stats_never_reach_edge_values(model):
    """Selection-only, stated as bookkeeping: an edge's running mean must be
    exactly its own backups. Folding sibling values into ``q_p0`` would mix
    distinguishable chance worlds into one expectation -- the unsound merge
    ``chance.rs`` deliberately refuses."""

    for state in _positions(3):
        _, mcts = _search(model, state, sims=300, chance_sibling_bias=1.0)
        stack = [mcts._closed_root]
        seen = 0
        while stack:
            node = stack.pop()
            for edge in node.edges:
                if edge.visits and not edge.probability_weighted:
                    assert edge.q_p0 == pytest.approx(
                        edge.value_sum_p0 / edge.visits, abs=1e-12
                    )
                    seen += 1
                for child in edge.children.values():
                    stack.append(child.node)
        assert seen


def test_sibling_stats_equal_the_sum_over_that_edges_children(model):
    """The local-exclusion subtraction in ``_sibling_bonus`` is only correct if
    the shared table is exactly the sum of its children's own edge statistics.
    If these ever drift, a world's own visits leak back as its own "sibling"
    evidence and the bias self-reinforces."""

    checked = 0
    for state in _positions(4):
        _, mcts = _search(model, state, sims=400, chance_sibling_bias=1.0)
        stack = [mcts._closed_root]
        while stack:
            node = stack.pop()
            for edge in node.edges:
                for child in edge.children.values():
                    stack.append(child.node)
                if not edge.sibling_stats:
                    continue
                checked += 1
                expected: dict = {}
                for child in edge.children.values():
                    for inner in child.node.edges:
                        key = structural_action_key(
                            child.node.state, inner.action_index
                        )
                        entry = expected.setdefault(key, [0, 0.0])
                        entry[0] += inner.visits
                        entry[1] += inner.value_sum_p0
                for key, (visits, value_sum) in edge.sibling_stats.items():
                    assert visits == expected[key][0]
                    assert value_sum == pytest.approx(expected[key][1], abs=1e-9)
    assert checked, "no edge accumulated sibling statistics in any test position"


def test_bias_is_zero_without_sibling_evidence(model):
    """A single world has no siblings to learn from, so the term must vanish --
    not merely be small. Otherwise the mechanism perturbs positions it has no
    evidence about."""

    _, mcts = _search(model, _positions(1)[0], sims=50, chance_sibling_bias=1.0)
    node = mcts._closed_root
    sign = 1.0 if node.actor == 0 else -1.0

    assert mcts._sibling_bonuses(node, {}, sign) == {}
    # Shared totals that are entirely this node's own contribution: nothing to
    # import, so no bonus.
    only_local: dict = {}
    for edge in node.edges:
        key = structural_action_key(node.state, edge.action_index)
        entry = only_local.setdefault(key, [0, 0.0])
        entry[0] += edge.visits
        entry[1] += edge.value_sum_p0
    assert mcts._sibling_bonuses(node, only_local, sign) == {}


def test_bias_decays_with_local_visits_and_respects_its_cap(model):
    """Local evidence must override sibling evidence, and the shared signal is
    a nudge toward LOOKING at an action, never a verdict on it."""

    _, mcts = _search(model, _positions(1)[0], sims=120, chance_sibling_bias=1.0)
    node = mcts._closed_root
    sign = 1.0 if node.actor == 0 else -1.0
    # Siblings loved every action, and none of it is this node's own doing.
    shared = {
        structural_action_key(node.state, edge.action_index): [1000, 1000.0 * sign]
        for edge in node.edges
    }
    bonuses = mcts._sibling_bonuses(node, shared, sign)
    assert bonuses, "no edge received a bonus from unanimous sibling evidence"

    cap = mcts.config.chance_sibling_bias_cap
    for edge in node.edges:
        bonus = bonuses.get(id(edge))
        if bonus is None:
            continue
        # Capped, and decaying in the edge's OWN visit count.
        assert bonus <= mcts.config.chance_sibling_bias * cap + 1e-12
        assert bonus == pytest.approx(
            bonuses_at(mcts, node, shared, sign, edge), rel=1e-9
        )
    # More local visits must mean a smaller share of the same evidence.
    ranked = sorted(
        (e for e in node.edges if id(e) in bonuses), key=lambda e: e.visits
    )
    if len(ranked) >= 2 and ranked[0].visits < ranked[-1].visits:
        assert bonuses[id(ranked[0])] > bonuses[id(ranked[-1])]


def bonuses_at(mcts, node, shared, sign, edge):
    """Recompute one edge's bonus from the documented formula."""

    key = structural_action_key(node.state, edge.action_index)
    local_v = sum(
        e.visits for e in node.edges
        if structural_action_key(node.state, e.action_index) == key
    )
    local_s = sum(
        e.value_sum_p0 for e in node.edges
        if structural_action_key(node.state, e.action_index) == key
    )
    sib_v = shared[key][0] - local_v
    sib_q = sign * ((shared[key][1] - local_s) / sib_v)
    adv = sib_q - sign * node.value_p0
    cap = mcts.config.chance_sibling_bias_cap
    adv = max(-cap, min(cap, adv))
    return mcts.config.chance_sibling_bias * adv / (1 + edge.visits)


def test_positive_only_clamp_refuses_to_argue_against_looking(model):
    """The review finding of 2026-09-01, pinned.

    A symmetric clamp lets a stale sibling estimate suppress an action. Since
    those estimates begin as the raw network values that caused the neglect, a
    large gain then buries exactly the move the mechanism exists to surface --
    measured: tracked visits collapsed 172 -> 5 at gain 30. Positive-only is the
    default; the signed form stays reachable for attribution.
    """

    _, mcts = _search(model, _positions(1)[0], sims=120, chance_sibling_bias=1.0)
    node = mcts._closed_root
    sign = 1.0 if node.actor == 0 else -1.0
    # Siblings hated everything: a purely negative advantage.
    hostile = {
        structural_action_key(node.state, edge.action_index): [1000, -1000.0 * sign]
        for edge in node.edges
    }
    assert mcts.config.chance_sibling_bias_positive_only is True
    assert mcts._sibling_bonuses(node, hostile, sign) == {}

    _, signed = _search(
        model, _positions(1)[0], sims=120,
        chance_sibling_bias=1.0, chance_sibling_bias_positive_only=False,
    )
    node2 = signed._closed_root
    sign2 = 1.0 if node2.actor == 0 else -1.0
    hostile2 = {
        structural_action_key(node2.state, edge.action_index): [1000, -1000.0 * sign2]
        for edge in node2.edges
    }
    negative = signed._sibling_bonuses(node2, hostile2, sign2)
    assert negative and all(v < 0 for v in negative.values()), (
        "the signed formulation must still be able to suppress, or this test "
        "no longer distinguishes the two"
    )


def test_flags_are_separable(model):
    """A regression must be attributable to one mechanism, so the two must not
    be entangled -- each has to change the search on its own."""

    state = _positions(1)[0]
    baseline = _fingerprint(_search(model, state)[0])
    m1 = _fingerprint(_search(model, state, chance_sibling_bias=1.0)[0])
    m2 = _fingerprint(_search(model, state, wonder_group_selection=True)[0])
    both = _fingerprint(
        _search(model, state, chance_sibling_bias=1.0, wonder_group_selection=True)[0]
    )
    assert len({baseline, m1, m2, both}) > 1
