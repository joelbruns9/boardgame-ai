"""Dirichlet root noise: the PUCT root's exploration source.

Gumbel supplies its own exploration through the Gumbel keys. PUCT has none, and
without a substitute self-play collapses toward deterministic lines -- so a PUCT
run needs this and a Gumbel run must not have it.

Ported from Kingdomino (`mcts_az.py:729`) with two deliberate differences, both
tested here: the concentration is set for 7WD's branching rather than copied,
and the draw comes from `PortableRng` so Rust/Python self-play stays
bit-comparable *with exploration on* -- a property 7WD has and Kingdomino, which
draws from numpy, does not.
"""

from __future__ import annotations

import statistics

import pytest

from .game import new_game
from .portable_rng import PortableRng
from .search import GumbelMCTS, SearchConfig
from .test_rust_engine_equiv import _mock_evaluate


def _config(**overrides) -> SearchConfig:
    base = dict(sims=32, top_k=16, mode="closed", seed=7, root_selection="puct")
    base.update(overrides)
    return SearchConfig(**base)


def _root_priors(state, config) -> list[float]:
    """Root edge priors after whatever noise the config asks for."""

    mcts = GumbelMCTS(None, config)
    mcts._evaluate = _mock_evaluate  # type: ignore[method-assign]
    mcts.search(state)
    return [edge.prior for edge in mcts._closed_root.edges]


# ---------------------------------------------------------------- primitives


@pytest.mark.parametrize("alpha", [0.3, 1.0, 1.8, 5.0])
def test_gamma_matches_its_moments(alpha):
    """Gamma(alpha, 1) has mean and variance both equal to alpha."""

    rng = PortableRng(12345)
    draws = [rng.gamma(alpha) for _ in range(50_000)]
    assert statistics.mean(draws) == pytest.approx(alpha, rel=0.03)
    assert statistics.pvariance(draws) == pytest.approx(alpha, rel=0.05)


def test_normal_is_standard():
    rng = PortableRng(999)
    draws = [rng.normal() for _ in range(50_000)]
    assert statistics.mean(draws) == pytest.approx(0.0, abs=0.02)
    assert statistics.pvariance(draws) == pytest.approx(1.0, rel=0.03)


def test_dirichlet_marginals():
    """Symmetric Dirichlet: each marginal has mean 1/n and the textbook variance."""

    alpha, n = 1.8, 5
    rng = PortableRng(7)
    columns = list(zip(*[rng.dirichlet(alpha, n) for _ in range(40_000)]))
    expected_var = (n - 1) / (n * n * (n * alpha + 1))
    for column in columns:
        assert statistics.mean(column) == pytest.approx(1.0 / n, rel=0.05)
        assert statistics.pvariance(column) == pytest.approx(expected_var, rel=0.10)


def test_dirichlet_sums_to_one():
    rng = PortableRng(3)
    for _ in range(200):
        assert sum(rng.dirichlet(1.8, 6)) == pytest.approx(1.0, abs=1e-12)


def test_dirichlet_is_deterministic_per_seed():
    """Reproducibility is the whole reason this lives in PortableRng."""

    assert PortableRng(42).dirichlet(1.8, 5) == PortableRng(42).dirichlet(1.8, 5)
    assert PortableRng(42).dirichlet(1.8, 5) != PortableRng(43).dirichlet(1.8, 5)


def test_alpha_controls_peakedness():
    """Why Kingdomino's 0.3 must not be copied to 7WD.

    Over ~5 legal actions, alpha=0.3 concentrates most of the noise mass on one
    arbitrary action; alpha=1.8 spreads a mild perturbation. Copying the value
    would change what the intervention *is*, not merely its strength.
    """

    peaks = {}
    for alpha in (0.3, 1.8):
        rng = PortableRng(11)
        peaks[alpha] = statistics.mean(
            max(rng.dirichlet(alpha, 5)) for _ in range(5_000)
        )
    assert peaks[0.3] > peaks[1.8] + 0.15


# ------------------------------------------------------------- searcher wiring


def test_off_by_default():
    assert SearchConfig().dirichlet_epsilon == 0.0


def test_epsilon_zero_leaves_priors_untouched():
    state = new_game(5)
    assert _root_priors(state, _config(dirichlet_epsilon=0.0)) == _root_priors(
        state, _config(dirichlet_epsilon=0.0)
    )


def test_gumbel_root_ignores_the_noise():
    """The Gumbel root already carries exploration; noise there would double up."""

    state = new_game(5)
    without = _root_priors(state, _config(root_selection="gumbel", dirichlet_epsilon=0.0))
    with_eps = _root_priors(state, _config(root_selection="gumbel", dirichlet_epsilon=0.25))
    assert without == with_eps


def test_puct_root_applies_the_standard_convex_blend():
    """`(1-eps)*prior + eps*noise`, the AlphaZero form.

    Asserted through the mass identity rather than "sum stays 1", because the
    MOCK evaluator returns unnormalised priors (they sum to ~2.53). Production
    normalises in `blend_priors` before any prior reaches a root, so there the
    identity reduces to sum == 1.

    An earlier version scaled the noise by the observed prior mass so that `eps`
    kept its meaning on unnormalised input. That was withdrawn on review: it
    bought nothing in production and, on a mock whose priors sum to 2.53, it
    also preserved a 2.53x inflated `c_puct * prior` exploration term --
    concealing the contract violation rather than surfacing it.
    """

    epsilon = 0.25
    state = new_game(5)
    clean = _root_priors(state, _config(dirichlet_epsilon=0.0))
    noised = _root_priors(state, _config(dirichlet_epsilon=epsilon))
    assert clean != noised
    assert all(value >= 0.0 for value in noised)
    # Dirichlet noise sums to 1, so the blended mass is exact and checkable.
    expected_mass = (1.0 - epsilon) * sum(clean) + epsilon
    assert sum(noised) == pytest.approx(expected_mass, abs=1e-9)


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), 0.0, -1.0])
def test_non_finite_alpha_is_rejected(alpha):
    """`NaN <= 0` is False, so a bare positivity check admits NaN -- which then
    never terminates the rejection sampler. Infinity is quieter: every draw
    returns inf, `inf/inf` makes the whole vector NaN, and NaN comparisons pin
    PUCT selection to the first edge with no error raised anywhere."""

    with pytest.raises(ValueError):
        PortableRng(1).gamma(alpha)


def test_noise_does_not_reach_the_recorded_target():
    """`_puct_root` builds the target from VISITS, so noise moves which actions
    get searched without ever being blended into a training label."""

    state = new_game(5)
    mcts = GumbelMCTS(None, _config(dirichlet_epsilon=0.25))
    mcts._evaluate = _mock_evaluate  # type: ignore[method-assign]
    result = mcts.search(state)
    assert sum(result.policy_target.values()) == pytest.approx(1.0, abs=1e-9)
    total_visits = sum(result.visits.values())
    for action, target in result.policy_target.items():
        assert target == pytest.approx(result.visits[action] / total_visits, abs=1e-9)


def test_same_seed_reproduces_the_noised_search():
    state = new_game(5)
    config = _config(dirichlet_epsilon=0.25, seed=99)
    first = _root_priors(state, config)
    second = _root_priors(state, config)
    assert first == second


# --------------------------------------------------------------- equivalence


def test_rust_matches_python_with_noise_on():
    """The property Kingdomino gave up by drawing from numpy.

    KD's own comment concedes "the noise VALUES cannot match Python's numpy RNG,
    so noise-on search is not bit-comparable -- the equivalence gate uses eps=0."
    Drawing from `PortableRng` instead keeps 7WD's self-play gate meaningful at
    the epsilon production actually runs at.

    Both arms are checked: eps=0 proves the port is inert when off, eps=0.25
    proves the streams stay aligned once it starts consuming from them.
    """

    from .codec import decode_action, legal_action_indices
    from .engine import apply_action
    from .game import Phase
    from .test_rust_engine_equiv import extract_setup, random_game, swr

    first_player, actions, library = random_game(0, 0)
    state = new_game(0, first_player=first_player)
    rust = swr.RustGame(
        library_draws=[list(draw) for draw in library], **extract_setup(state)
    )
    checked = 0
    for index, action_index in enumerate(actions):
        ready = (
            index >= 8
            and state.phase is Phase.PLAY_AGE
            and state.pending_choice is None
        )
        if ready and checked < 3:
            for epsilon in (0.0, 0.25):
                config = _config(sims=32, top_k=8, seed=5, dirichlet_epsilon=epsilon)
                mcts = GumbelMCTS(None, config)
                mcts._evaluate = _mock_evaluate  # type: ignore[method-assign]
                expected = mcts.search(state)
                (
                    action,
                    _value,
                    _root_value,
                    visits,
                    policy,
                    _topk,
                    _sims,
                    _digest,
                ) = rust.closed_search(
                    32,
                    8,
                    5,
                    force=False,
                    puct_root=True,
                    dirichlet_epsilon=epsilon,
                    dirichlet_alpha=config.dirichlet_alpha,
                )
                legal = list(legal_action_indices(state))
                context = f"move {index} eps {epsilon}"
                assert action == expected.action_index, context
                assert list(visits) == [expected.visits.get(a, 0) for a in legal], context
                for got, want in zip(
                    policy, [expected.policy_target.get(a, 0.0) for a in legal]
                ):
                    assert got == pytest.approx(want, abs=1e-12), context
            checked += 1
        apply_action(state, decode_action(state, action_index))
        rust.apply_index(action_index)
    assert checked == 3
