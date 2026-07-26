"""Phase 2 gates: conflict-free leaf waves.

The rule is that no two in-flight simulations may sit in the same root
candidate's subtree. Distinct root edges own disjoint arena nodes, so members of
a conflict-free wave cannot see each other's virtual loss — which should make
`leaf_batch > 1` an *exact* batching of `leaf_batch = 1` rather than the
WU approximation of it that today's batching is.

The plan states that as a strong, falsifiable claim: with the rule on, **every**
budget and **every** legal-count stratum should be bit-identical to
`leaf_batch = 1`. If it holds, leaf batching needs no quality gate at all; if it
fails anywhere, the failing stratum names the mechanism. These tests test exactly
that, and they assert the invariant itself rather than only its consequences.
"""

from __future__ import annotations

import pytest

from .codec import decode_action, legal_action_indices
from .engine import apply_action
from .game import Phase, new_game
from .rust_bridge import rust_setup


# Legal-action-count strata. Phase 0's corpus measurement found 52% of decisions
# have <= 4 legal actions, which is exactly the half the earlier bit-identity
# result never covered.
STRATA = {
    "legal_le_2": lambda n: n <= 2,
    "legal_3_4": lambda n: 3 <= n <= 4,
    "legal_5_8": lambda n: 5 <= n <= 8,
    "legal_9_16": lambda n: 9 <= n <= 16,
    "legal_gt_16": lambda n: n > 16,
}

# Cheap budgets are the production `sims in 16..24` band; 67 stands in for a full
# search, where more halving rounds mean more chances to repeat a candidate.
BUDGETS = (16, 17, 20, 24, 67)


def _random_walk(game_seed: int, limit: int = 400):
    """Walk one game, yielding (rust_game, legal_count) at every decision."""

    import random

    import seven_wonders_rust as swr

    rng = random.Random(game_seed ^ 0x5EED)
    python_game = new_game(game_seed, first_player=game_seed % 2)
    library: list[list[str]] = []
    setup = rust_setup(python_game)
    replay: list[int] = []
    for _ in range(limit):
        if python_game.phase is Phase.COMPLETE:
            break
        legal = list(legal_action_indices(python_game))
        if not legal:
            break
        rust_game = swr.RustGame(library_draws=[list(d) for d in library], **setup)
        for index in replay:
            rust_game.apply_index(index)
        yield rust_game, len(legal)
        choice = rng.choice(legal)
        action = decode_action(python_game, choice)
        apply_action(python_game, action)
        if action.wonder_name == "The Great Library":
            pending = python_game.pending_choice
            if pending is not None:
                library.append(list(pending.options))
        replay.append(choice)


def _search(game, leaf_batch, sims, seed, *, force, conflict_free):
    result = game.closed_search_batched(
        leaf_batch,
        sims,
        8,
        seed,
        force=force,
        conflict_free_waves=conflict_free,
    )
    action, action_value, root_value, visits, policy, topk, done, metrics, root_q, digest = (
        result
    )
    return {
        "action": action,
        "action_value": action_value,
        "root_value": root_value,
        "visits": list(visits),
        "policy": list(policy),
        "topk": list(topk),
        "sims": done,
        "waves": metrics[5],
        "max_paths": metrics[6],
        "collisions": metrics[4],
        "requested": metrics[1],
        "digest": list(digest),
    }


def _assert_identical(baseline, candidate, context):
    assert candidate["action"] == baseline["action"], f"{context}: action"
    assert candidate["visits"] == baseline["visits"], f"{context}: visits"
    assert candidate["topk"] == baseline["topk"], f"{context}: top-k"
    assert candidate["sims"] == baseline["sims"], f"{context}: sims"
    assert candidate["action_value"] == pytest.approx(
        baseline["action_value"], rel=0, abs=0
    ), f"{context}: action value"
    assert candidate["root_value"] == pytest.approx(
        baseline["root_value"], rel=0, abs=0
    ), f"{context}: root value"
    assert candidate["policy"] == pytest.approx(
        baseline["policy"], rel=0, abs=0
    ), f"{context}: policy"
    assert candidate["digest"] == pytest.approx(
        baseline["digest"], rel=0, abs=0
    ), f"{context}: tree digest"


def _stratified_corpus(per_stratum=2, games=40):
    """Collect positions covering every legal-count stratum."""

    covered = {name: 0 for name in STRATA}
    corpus = []
    for game_seed in range(games):
        if all(count >= per_stratum for count in covered.values()):
            break
        for rust_game, legal_count in _random_walk(game_seed):
            for name, predicate in STRATA.items():
                if predicate(legal_count) and covered[name] < per_stratum:
                    covered[name] += 1
                    corpus.append((name, legal_count, rust_game, game_seed))
                    break
            if all(count >= per_stratum for count in covered.values()):
                break
    return corpus, covered


@pytest.fixture(scope="module")
def corpus():
    collected, covered = _stratified_corpus()
    # The gate is only meaningful if the strata the earlier result missed are
    # actually present.
    assert covered["legal_le_2"] > 0, covered
    assert covered["legal_3_4"] > 0, covered
    return collected


def test_conflict_free_waves_are_bit_identical_across_every_stratum(corpus):
    """The plan's strong claim, tested where it was previously untested."""

    checked = 0
    for stratum, legal_count, game, game_seed in corpus:
        for sims in BUDGETS:
            for leaf_batch in (2, 4, 8, 16):
                seed = 4001 + game_seed * 17 + sims
                force = (sims + leaf_batch) % 3 == 0
                baseline = _search(
                    game, 1, sims, seed, force=force, conflict_free=True
                )
                candidate = _search(
                    game, leaf_batch, sims, seed, force=force, conflict_free=True
                )
                context = (
                    f"{stratum} (legal={legal_count}) game {game_seed} "
                    f"sims={sims} leaf_batch={leaf_batch} force={force}"
                )
                _assert_identical(baseline, candidate, context)
                checked += 1
    assert checked >= len(BUDGETS) * 4 * len(STRATA)


def test_the_invariant_holds_and_collisions_vanish(corpus):
    """Assert the invariant, not only its consequence.

    Rust raises if a wave ever holds a root edge twice, so reaching the end is
    the invariant check. Collisions are the visible consequence: two paths can
    only reach the same leaf if they share a root subtree, so a conflict-free run
    must report none.
    """

    for stratum, legal_count, game, game_seed in corpus:
        for leaf_batch in (4, 16):
            result = _search(
                game, leaf_batch, 24, 5000 + game_seed, force=False, conflict_free=True
            )
            context = f"{stratum} (legal={legal_count}) leaf_batch={leaf_batch}"
            assert result["collisions"] == 0, context
            assert 1 <= result["max_paths"] <= leaf_batch, context


def test_unconstrained_batching_is_the_approximation_it_replaces(corpus):
    """The rule must be doing real work, or identity would prove nothing.

    Without it, wide `leaf_batch` diverges from `leaf_batch = 1` somewhere in the
    corpus. If this ever stops finding a divergence the identity test above has
    lost its meaning, so the negative control is gated too.
    """

    divergences = 0
    for _stratum, _legal, game, game_seed in corpus:
        for sims in BUDGETS:
            baseline = _search(game, 1, sims, 6000 + game_seed, force=False, conflict_free=True)
            loose = _search(game, 16, sims, 6000 + game_seed, force=False, conflict_free=False)
            if (
                loose["visits"] != baseline["visits"]
                or loose["action"] != baseline["action"]
                or list(loose["digest"]) != list(baseline["digest"])
            ):
                divergences += 1
    assert divergences > 0, (
        "unconstrained leaf batching matched leaf_batch=1 everywhere, so the "
        "conflict-free identity result is not evidence of anything"
    )


def test_conflict_rule_tapers_waves_instead_of_being_configured(corpus):
    """Wave width should settle well below `leaf_batch` on its own.

    Phase 0's corpus replay predicted a mean width near 2.6 at cheap budgets.
    The assertion here is only the qualitative one the rule guarantees: raising
    `leaf_batch` far above the taper cannot raise the realized width without
    bound, because the candidate set runs out first.
    """

    widths = []
    for _stratum, _legal, game, game_seed in corpus:
        wide = _search(game, 16, 20, 7000 + game_seed, force=False, conflict_free=True)
        widths.append(wide["max_paths"])
        # Waves cannot be wider than the number of distinct root candidates.
        assert wide["max_paths"] <= 16
    assert widths
    assert sum(widths) / len(widths) < 16.0


def test_wave_batching_reduces_batch_count_versus_leaf_batch_one(corpus):
    """The point of the exercise: fewer, wider evaluation batches."""

    single_total = 0
    batched_total = 0
    for _stratum, _legal, game, game_seed in corpus:
        single = _search(game, 1, 24, 8000 + game_seed, force=False, conflict_free=True)
        batched = _search(game, 16, 24, 8000 + game_seed, force=False, conflict_free=True)
        single_total += single["waves"]
        batched_total += batched["waves"]
    assert batched_total < single_total
    print(
        f"waves per search: leaf_batch=1 -> {single_total}, "
        f"conflict-free leaf_batch=16 -> {batched_total} "
        f"({single_total / batched_total:.2f}x fewer)"
    )
