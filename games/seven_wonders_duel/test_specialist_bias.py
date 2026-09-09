"""W7 S0 acceptance gates for the specialist leaf-utility bias.

The plan's acceptance is three claims, and each has a test here:

1. ``lambda = 0`` is bit-identical to a build without the feature, verified on
   the RESUMABLE path specifically -- that is the searcher production self-play
   runs, and a bias that existed only in the scalar reference would be inert
   where it matters.
2. ``lambda > 0`` has its own equivalence gate: Python and Rust agree
   move-for-move on a biased search. Without it the bias exists in the reference
   implementation and not in the one that generates the data.
3. Cached and forced evaluations carry the same bias as fresh ones -- replaying
   a cached leaf must not change its utility.

Plus the two things that would make a null result meaningless: the sign
convention (a specialist on seat 1 must not reward its opponent's win), and the
refusal to bias silently when an evaluator supplies no outlook.
"""

from __future__ import annotations

import pytest

from .buffer import logic_fingerprint
from .codec import decode_action, legal_action_indices
from .engine import apply_action
from .game import Phase, new_game
from .search import (
    GumbelMCTS,
    LeafBias,
    SearchConfig,
    _terminal_outlook_p0,
)
from .test_rust_engine_equiv import (
    _MASK64,
    _mock_fold,
    _mock_mix,
    _mock_unit,
    extract_setup,
    mock_eval,
    random_game,
)

swr = pytest.importorskip("seven_wonders_rust")

#: Mirrors `eval.rs::OUTLOOK_SALT`.
_OUTLOOK_SALT = 0x2545F4914F6CDD1D


def mock_outlook(game):
    """Python mirror of `eval.rs::MockOutlookEval::outlook_of`.

    Normalised by an explicit left fold, which is what Python's ``sum`` and
    Rust's ``fold(0.0, +)`` both do, so the two agree to the last bit.
    """

    if game.phase is Phase.COMPLETE:
        return _terminal_outlook_p0(game)
    h = _mock_fold(logic_fingerprint(game))
    raw = [
        _mock_unit(_mock_mix(h ^ ((_OUTLOOK_SALT * (k + 1)) & _MASK64)))
        for k in range(7)
    ]
    mass = 0.0
    for value in raw:
        mass += value
    return [value / mass for value in raw]


def mock_eval_dict(state):
    """The historical two-tuple mock: an evaluator with no outlook at all."""

    value, weights = mock_eval(state)
    if state.phase is Phase.COMPLETE:
        return value, {}
    legal = legal_action_indices(state)
    return value, {a: w for a, w in zip(legal, weights)}


def _mock_evaluate_with_outlook(state):
    """`GumbelMCTS._evaluate` stand-in carrying the mock outlook."""

    value, priors = mock_eval_dict(state)
    return value, priors, mock_outlook(state)


def _biased_search(state, sims, top_k, seed, *, force, bias):
    mcts = GumbelMCTS(
        None,
        SearchConfig(
            sims=sims,
            top_k=top_k,
            mode="closed",
            seed=seed,
            force_expand_root_chance=force,
            specialist_lambda=bias[0],
            specialist_victory=bias[1],
            specialist_seat=bias[2],
            specialist_symmetric=bias[3] if len(bias) > 3 else False,
        ),
    )
    mcts._evaluate = _mock_evaluate_with_outlook  # type: ignore[method-assign]
    return mcts.search(state), mcts._closed_root


def _positions(games=4, first=8):
    """One mid-Age-I play position per seed, in both engines."""

    out = []
    for game_seed in range(games):
        first_player, actions, library = random_game(game_seed, game_seed % 2)
        py = new_game(game_seed, first_player=first_player)
        rg = swr.RustGame(
            library_draws=[list(d) for d in library], **extract_setup(py)
        )
        for i, idx in enumerate(actions):
            if i >= first and py.phase is Phase.PLAY_AGE and py.pending_choice is None:
                out.append((game_seed, py, rg))
                break
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)
    assert out, "no searchable positions found"
    return out


# --------------------------------------------------------------------------
# The utility formula itself
# --------------------------------------------------------------------------

#: `dataset.JOINT7_CLASSES` order.
ONE_HOT = {
    "p0_civilian": 0,
    "p0_scientific": 1,
    "p0_military": 2,
    "p1_civilian": 3,
    "p1_scientific": 4,
    "p1_military": 5,
    "draw": 6,
}


def _one_hot(index):
    out = [0.0] * 7
    out[index] = 1.0
    return out


@pytest.mark.parametrize("name,index", sorted(ONE_HOT.items(), key=lambda kv: kv[1]))
@pytest.mark.parametrize("seat", [0, 1])
def test_all_seven_terminal_utilities_are_pinned_and_agree_across_languages(
    name, index, seat
):
    """The whole sign convention, one class at a time.

    A science specialist on seat 1 must be rewarded for `p1_scientific` and NOT
    for `p0_scientific`: the outlook is player-0 relative while "my victory
    type" is specialist-relative, and conflating them reverses the bonus on
    exactly half the seats.
    """

    lam = 0.5
    value_p0 = 0.25
    outlook = _one_hot(index)
    got = LeafBias(lambda_=lam, victory="scientific", seat=seat).shape(
        value_p0, outlook
    )
    own = "p0_scientific" if seat == 0 else "p1_scientific"
    if name == own:
        expected = value_p0 + lam if seat == 0 else value_p0 - lam
    else:
        expected = value_p0
    assert got == pytest.approx(expected, abs=1e-12), name
    assert got == pytest.approx(
        swr.specialist_utility(value_p0, outlook, lam, "scientific", seat, False),
        abs=1e-12,
    )


@pytest.mark.parametrize("seat", [0, 1])
def test_symmetric_is_seat_independent_and_own_win_is_not(seat):
    """The property that makes symmetric a mirror-player rather than an attacker.

    In p0 terms the symmetric utility is the SAME scalar for both seats, so both
    specialists optimise one thing -- which is why it is behind a flag and why
    S0 defaults to own-win.
    """

    outlook = [0.05, 0.30, 0.10, 0.05, 0.20, 0.10, 0.20]
    symmetric = LeafBias(
        lambda_=0.4, victory="scientific", seat=seat, symmetric=True
    ).shape(0.1, outlook)
    reference = LeafBias(
        lambda_=0.4, victory="scientific", seat=0, symmetric=True
    ).shape(0.1, outlook)
    assert symmetric == pytest.approx(reference, abs=1e-12)
    assert symmetric == pytest.approx(
        swr.specialist_utility(0.1, outlook, 0.4, "scientific", seat, True), abs=1e-12
    )

    own = LeafBias(lambda_=0.4, victory="scientific", seat=seat).shape(0.1, outlook)
    other = LeafBias(lambda_=0.4, victory="scientific", seat=1 - seat).shape(
        0.1, outlook
    )
    assert own != pytest.approx(other, abs=1e-9)


@pytest.mark.parametrize("victory", ["civilian", "scientific", "military"])
def test_each_victory_class_reads_its_own_outlook_slot(victory):
    outlook = [0.10, 0.20, 0.30, 0.05, 0.15, 0.10, 0.10]
    offset = {"civilian": 0, "scientific": 1, "military": 2}[victory]
    assert LeafBias(lambda_=1.0, victory=victory, seat=0).shape(
        0.0, outlook
    ) == pytest.approx(outlook[offset], abs=1e-12)
    assert LeafBias(lambda_=1.0, victory=victory, seat=1).shape(
        0.0, outlook
    ) == pytest.approx(-outlook[3 + offset], abs=1e-12)


def test_terminal_outlook_matches_the_rust_exact_form():
    """A finished game knows how it ended; both languages must say the same."""

    for game_seed in range(4):
        first_player, actions, library = random_game(game_seed, game_seed % 2)
        py = new_game(game_seed, first_player=first_player)
        rg = swr.RustGame(
            library_draws=[list(d) for d in library], **extract_setup(py)
        )
        for idx in actions:
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)
        assert py.phase is Phase.COMPLETE
        assert _terminal_outlook_p0(py) == pytest.approx(rg.mock_outlook(), abs=0.0)
        assert sum(_terminal_outlook_p0(py)) == pytest.approx(1.0, abs=1e-12)


def test_a_missing_outlook_under_a_live_lambda_is_a_hard_error():
    """Never a silent zero bias.

    A treatment that reaches some leaves and not others measures nothing, and
    several `LeafOut` branches legitimately construct `None`: mock evaluators,
    checkpoints without W4's head, the solver boundary.
    """

    with pytest.raises(ValueError, match="no outlook"):
        LeafBias(lambda_=0.3, victory="scientific", seat=0).shape(0.0, None)
    with pytest.raises(ValueError):
        swr.specialist_utility(0.0, None, 0.3, "scientific", 0, False)
    assert LeafBias().shape(0.4, None) == 0.4
    assert swr.specialist_utility(0.4, None, 0.0, "scientific", 0, False) == 0.4


def test_a_search_whose_evaluator_has_no_outlook_refuses_to_run_biased():
    """The whole search, not just the shaping helper."""

    _seed, py, _rg = _positions(games=1)[0]
    mcts = GumbelMCTS(
        None,
        SearchConfig(sims=16, top_k=8, mode="closed", seed=3, specialist_lambda=0.4),
    )
    mcts._evaluate = mock_eval_dict  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="no outlook"):
        mcts.search(py)


def test_invalid_bias_configuration_is_rejected():
    with pytest.raises(ValueError):
        LeafBias(lambda_=-0.1)
    with pytest.raises(ValueError):
        LeafBias(lambda_=float("inf"))
    with pytest.raises(ValueError):
        LeafBias(seat=2)
    with pytest.raises(ValueError):
        LeafBias(victory="economic")
    with pytest.raises(ValueError):
        swr.specialist_utility(0.0, _one_hot(1), 0.3, "economic", 0, False)


# --------------------------------------------------------------------------
# The mock outlook oracle -- the instrument the lambda > 0 gate runs on
# --------------------------------------------------------------------------


def test_mock_outlook_matches_python_bit_for_bit():
    for game_seed in range(6):
        first_player, actions, library = random_game(game_seed, game_seed % 2)
        py = new_game(game_seed, first_player=first_player)
        rg = swr.RustGame(
            library_draws=[list(d) for d in library], **extract_setup(py)
        )
        for idx in actions + [None]:
            expected = mock_outlook(py)
            got = list(rg.mock_outlook())
            assert got == expected, f"seed {game_seed}: mock outlook differs"
            assert sum(expected) == pytest.approx(1.0, abs=1e-12)
            if idx is None:
                break
            apply_action(py, decode_action(py, idx))
            rg.apply_index(idx)


# --------------------------------------------------------------------------
# Acceptance 1: lambda = 0 is bit-identical
# --------------------------------------------------------------------------


@pytest.mark.parametrize("resumable", [False, True])
def test_lambda_zero_is_bit_identical_to_the_unbiased_searcher(resumable):
    """The biased plumbing is inert until it is asked for.

    Run through the entry point that carries the bias, at lambda = 0, against
    the one that has no such field -- on the RESUMABLE path as well as the
    scalar oracle, because the resumable searcher is what production runs.
    """

    checked = 0
    for _seed, _py, rg in _positions():
        for sims in (16, 64):
            for seed in (1, 5):
                for force in (False, True):
                    if resumable:
                        base = rg.closed_search_resumable(sims, 8, seed, force=force)
                    else:
                        base = rg.closed_search(sims, 8, seed, force=force)
                    biased = rg.closed_search_biased(
                        sims,
                        8,
                        seed,
                        0.0,
                        "scientific",
                        0,
                        force=force,
                        resumable=resumable,
                    )
                    ctx = f"sims {sims} seed {seed} force {force}"
                    assert tuple(biased[:8]) == tuple(base), ctx
                    assert biased[8] is None, ctx
                    checked += 1
    assert checked >= 16


# --------------------------------------------------------------------------
# Acceptance 2: lambda > 0 Python/Rust equivalence
# --------------------------------------------------------------------------


BIASES = [
    (0.5, "scientific", 0),
    (0.5, "scientific", 1),
    (0.25, "military", 0),
    (1.0, "civilian", 1),
    (0.5, "scientific", 0, True),
]


@pytest.mark.parametrize("bias", BIASES, ids=lambda b: "_".join(str(x) for x in b))
@pytest.mark.parametrize("resumable", [False, True])
def test_biased_search_matches_python_move_for_move(bias, resumable):
    """The lambda > 0 equivalence gate.

    Chosen action, visits, top-k, values, policy target and the unshaped root --
    the same assertions the unbiased gate makes, on a biased search.
    """

    checked = 0
    for _seed, py, rg in _positions():
        legal = legal_action_indices(py)
        for sims in (16, 64):
            for seed in (1, 5):
                for force in (False, True):
                    result, _root = _biased_search(
                        py, sims, 8, seed, force=force, bias=bias
                    )
                    got = rg.closed_search_biased(
                        sims,
                        8,
                        seed,
                        bias[0],
                        bias[1],
                        bias[2],
                        specialist_symmetric=bias[3] if len(bias) > 3 else False,
                        force=force,
                        resumable=resumable,
                    )
                    ctx = f"bias {bias} sims {sims} seed {seed} force {force}"
                    assert got[0] == result.action_index, f"{ctx}: action"
                    assert got[6] == result.sims, f"{ctx}: sims"
                    assert list(got[5]) == list(result.gumbel_topk), f"{ctx}: topk"
                    assert list(got[3]) == [
                        result.visits[a] for a in legal
                    ], f"{ctx}: visits"
                    assert got[1] == pytest.approx(
                        result.action_value, abs=1e-9
                    ), f"{ctx}: action_value"
                    assert got[2] == pytest.approx(
                        result.root_value, abs=1e-9
                    ), f"{ctx}: root_value"
                    for j, a in enumerate(legal):
                        assert got[4][j] == pytest.approx(
                            result.policy_target[a], abs=1e-9
                        ), f"{ctx}: policy[{a}]"
                    assert got[8] == pytest.approx(
                        result.root_value_unshaped, abs=1e-9
                    ), f"{ctx}: root_value_unshaped"
                    checked += 1
    assert checked >= 16


# --------------------------------------------------------------------------
# Acceptance 3: the bias reaches every leaf path, including cached ones
# --------------------------------------------------------------------------


def test_forced_and_cached_leaves_carry_the_same_bias_as_fresh_ones():
    """Force-expansion materialises children, evaluates them in one batch and
    replays the cached evaluation on the first ordinary visit.

    If the bias were applied only where a fresh evaluation arrives, the seeded
    probability-weighted Q and the replayed backup would disagree, and the
    scalar oracle (which has no cache) would diverge from the resumable searcher
    (which does). Asserting the two agree under force IS the cache test.
    """

    checked = 0
    for _seed, _py, rg in _positions():
        for sims in (16, 64):
            for seed in (2, 7):
                scalar = rg.closed_search_biased(
                    sims, 8, seed, 0.6, "scientific", 0, force=True, resumable=False
                )
                cached = rg.closed_search_biased(
                    sims, 8, seed, 0.6, "scientific", 0, force=True, resumable=True
                )
                assert cached[0] == scalar[0]
                assert list(cached[3]) == list(scalar[3])
                assert cached[2] == pytest.approx(scalar[2], abs=1e-9)
                assert list(cached[7]) == pytest.approx(list(scalar[7]), abs=1e-9)
                checked += 1
    assert checked >= 8


# --------------------------------------------------------------------------
# The bias must actually do something
# --------------------------------------------------------------------------


def test_a_live_lambda_changes_the_search_and_widens_the_utility_scale():
    """A null result must not be explicable by "the flag did nothing"."""

    moved = 0
    for _seed, _py, rg in _positions():
        for seed in (1, 5, 9):
            unbiased = rg.closed_search_biased(
                64, 8, seed, 0.0, "scientific", 0, force=False
            )
            biased = rg.closed_search_biased(
                64, 8, seed, 1.0, "scientific", 0, force=False
            )
            if list(biased[3]) != list(unbiased[3]):
                moved += 1
            # The utility scale widens to [-1-lambda, 1+lambda]; the unshaped
            # root stays a win probability in [-1, 1].
            assert -2.0 <= biased[2] <= 2.0
            assert -1.0 <= biased[8] <= 1.0
    assert moved > 0, "a lambda of 1.0 moved no visit distribution at all"


def test_the_unshaped_root_stays_within_one_lambda_of_the_shaped_root():
    """A cheap invariant on the two separately accumulated sums.

    They run over the same leaves and differ per leaf by a bonus bounded by
    lambda, so their means differ by at most lambda. A reconstruction bug --
    subtracting the wrong outlook sum -- breaks this immediately.
    """

    for _seed, py, _rg in _positions(games=2):
        for lam in (0.3, 0.8):
            result, _ = _biased_search(
                py, 64, 8, 4, force=False, bias=(lam, "scientific", 0)
            )
            assert result.root_value_unshaped is not None
            assert abs(result.root_value - result.root_value_unshaped) <= lam + 1e-9
            assert -1.0 <= result.root_value_unshaped <= 1.0
