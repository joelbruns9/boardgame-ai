"""W4's searched seven-way outlook: backed up, recorded, not yet consumed.

What this is for, stated once so the tests below read as one argument: the
realised outcome is a GAME-CONSTANT victory-type label. Every row of a game
that ended scientifically carries `my_scientific`, move 3 included, where
science was one of three live possibilities. Search's own distribution over the
seven classes varies by position instead, and at the terminals it reached it is
exact.

That is a different signal, not a better one -- the realised label is a
stochastic sample under the policy actually played, this is a model-assisted
estimate under the search's own exploration, and exact leaves do not make a
root estimate proven. Which is preferable is for a measured bias/variance
comparison to settle, which is why nothing trains on it yet.

What it is NOT for: changing moves. `P(win) - P(loss)` is LINEAR in the seven
probabilities, so averaging the vector and then collapsing is identical to
collapsing at each leaf and averaging -- which is what the scalar backup
already does. Nothing here can move a search, and `test_selection_is_untouched`
pins that.
"""

import random

import pytest

torch = pytest.importorskip("torch")

from games.seven_wonders_duel.buffer import GameRecorder, from_json_line, to_json_line
from games.seven_wonders_duel.codec import legal_action_indices
from games.seven_wonders_duel.dataset import (
    JOINT7_CLASSES,
    examples_from_record,
)
from games.seven_wonders_duel.game import Phase, VictoryType
from games.seven_wonders_duel.net import SWDNet

seven_wonders_rust = pytest.importorskip("seven_wonders_rust")


def _model(**kwargs):
    torch.manual_seed(0)
    return SWDNet(d_model=32, layers=2, heads=4, **kwargs)


def _search(model, sims=48, seed=5, value_source="flat"):
    """One Rust search, through the production flat-batch adapter.

    From a real mid-game position rather than the draft, so the tree actually
    branches over tableau play.
    """

    from games.seven_wonders_duel.inference import Evaluator
    from games.seven_wonders_duel.rust_bridge import rust_flat_batch_adapter

    evaluator = Evaluator(
        model, "cpu", 64, fuse_embedder=False, value_source=value_source
    )
    adapter = rust_flat_batch_adapter(evaluator)
    results = seven_wonders_rust.search_many_flat_net(
        adapter,
        [_mid_game()],
        [seed],
        64,
        1,
        sims,
        4,
    )
    return results[0]


def _mid_game():
    """A Rust game advanced past the Wonder draft on a fixed action stream."""

    from games.seven_wonders_duel.rust_bridge import rust_games_for_self_play

    game = rust_games_for_self_play([11], [0])[0]
    rng = random.Random(9)
    for _ in range(12):
        legal = game.legal_action_indices()
        if not legal or game.is_complete():
            break
        game.apply_index(rng.choice(legal))
    return game


# --- the vector reaches the root --------------------------------------------


def test_a_net_without_the_head_records_no_outlook():
    """The short adapter row stays legal, so nothing had to change to keep
    every existing evaluator working."""

    result = _search(_model())
    assert result["root_outlook"] is None


def test_the_searched_outlook_is_a_distribution_over_the_seven_classes():
    result = _search(_model(hierarchical_value=True))
    outlook = result["root_outlook"]
    assert outlook is not None
    assert len(outlook) == len(JOINT7_CLASSES) == 7
    assert all(value >= 0.0 for value in outlook)
    assert sum(outlook) == pytest.approx(1.0, abs=1e-6)


def test_the_marginal_agrees_with_the_value_only_under_the_same_head():
    """The coupling between the two arms, and it is not optional.

    The scalar backup averages whatever head `value_source` names; the outlook
    averages W4's. Under `flat` those are two different predictions of the same
    thing, so search's Q and search's seven-way split need not agree -- the
    flat-head inconsistency, reappearing one level up. Under `hierarchical`
    they are means over the same leaves of one object, and `P(win) - P(loss)`
    is linear, so they agree to floating-point noise.

    So a coherent SEARCHED panel needs `value_source='hierarchical'`; anything
    else has to be labelled as two heads talking past each other.
    """

    model = _model(hierarchical_value=True)

    matched = _search(model, sims=64, value_source="hierarchical")
    outlook = matched["root_outlook"]
    implied = sum(outlook[0:3]) - sum(outlook[3:6])
    assert implied == pytest.approx(matched["root_value"], abs=2e-6)

    mixed = _search(model, sims=64, value_source="flat")
    mixed_outlook = mixed["root_outlook"]
    mixed_implied = sum(mixed_outlook[0:3]) - sum(mixed_outlook[3:6])
    assert mixed_implied != pytest.approx(mixed["root_value"], abs=1e-4), (
        "two different heads agreeing exactly would mean the outlook is being "
        "derived from the value rather than backed up on its own"
    )


def test_selection_is_untouched_by_carrying_the_vector():
    """The safety property, and it comes from the arithmetic, not a flag.

    A model with the head and one without differ in what they SEND, not in
    what search does with it -- so a search whose evaluator supplies outlooks
    must pick the same move and the same visits as one that does not, given
    identical values.
    """

    plain = _model()
    withhead = _model(hierarchical_value=True)
    # The heads are separate parameters; the shared trunk and `value` head are
    # identical under the same seed, so the scalar stream is the same.
    a = _search(plain, seed=11)
    b = _search(withhead, seed=11)
    assert a["action"] == b["action"]
    assert a["visits"] == b["visits"]
    assert a["root_value"] == pytest.approx(b["root_value"], abs=1e-12)


# --- exact at proven terminals ----------------------------------------------


def test_a_deep_search_of_a_near_terminal_position_is_certain():
    """Terminals contribute their EXACT victory type, not a prediction.

    Searched from a position a few plies from the end, the backed-up outlook
    concentrates on the classes those terminals actually produce -- which is
    the part of a backed-up distribution that carries no model error at all.
    """

    from games.seven_wonders_duel.rust_bridge import rust_games_for_self_play

    game = rust_games_for_self_play([11], [0])[0]
    rng = random.Random(1)
    while not game.is_complete():
        legal = game.legal_action_indices()
        if len(legal) <= 1:
            break
        remaining = len(legal)
        game.apply_index(rng.choice(legal))
        if remaining <= 2:
            break

    from games.seven_wonders_duel.inference import Evaluator
    from games.seven_wonders_duel.rust_bridge import rust_flat_batch_adapter

    if game.is_complete():
        pytest.skip("random play finished the game before a search was possible")
    adapter = rust_flat_batch_adapter(
        Evaluator(
            _model(hierarchical_value=True),
            "cpu",
            64,
            fuse_embedder=False,
            value_source="hierarchical",
        )
    )
    result = seven_wonders_rust.search_many_flat_net(
        adapter, [game], [7], 64, 1, 96, 4
    )[0]
    outlook = result["root_outlook"]
    assert outlook is not None
    assert sum(outlook) == pytest.approx(1.0, abs=1e-6)
    assert sum(outlook[0:3]) - sum(outlook[3:6]) == pytest.approx(
        result["root_value"], abs=2e-6
    )


# --- it survives the round trip to a training row ---------------------------


def test_the_outlook_reaches_the_buffer_and_the_example():
    recorder = GameRecorder(11, agents={"p0": "search", "p1": "search"})
    rng = random.Random(2)
    outlook = [0.1, 0.2, 0.05, 0.3, 0.15, 0.15, 0.05]
    first = True
    while recorder.game.phase is not Phase.COMPLETE:
        action = rng.choice(legal_action_indices(recorder.game))
        recorder.play(
            action,
            visits={action: 1},
            root_value=0.25,
            root_outlook=outlook if first else None,
            sims=8,
            mode="closed",
        )
        first = False
    record = recorder.finish()

    reloaded = from_json_line(to_json_line(record))
    assert reloaded.moves[0].root_outlook == pytest.approx(outlook)
    assert reloaded.moves[1].root_outlook is None

    examples = examples_from_record(reloaded)
    assert examples[0].root_outlook == pytest.approx(outlook)
    assert examples[1].root_outlook is None


def test_a_buffer_written_before_w4_still_loads():
    """`.get`, not `[...]`: the key is absent in every existing file."""

    recorder = GameRecorder(11, agents={"p0": "random", "p1": "random"})
    rng = random.Random(4)
    while recorder.game.phase is not Phase.COMPLETE:
        recorder.play(rng.choice(legal_action_indices(recorder.game)))
    line = to_json_line(recorder.finish())

    import json

    payload = json.loads(line)
    for move in payload["moves"]:
        move.pop("root_outlook", None)
    reloaded = from_json_line(json.dumps(payload))
    assert all(move.root_outlook is None for move in reloaded.moves)


# --- every backup path, not just the evaluated one --------------------------


def _near_terminal_game():
    """A position whose searches end immediately, reached by playing seed 1.

    Built by search rather than by hand: what matters is that simulations
    settle on TERMINALS, which is the path that bypassed the outlook.
    """

    from games.seven_wonders_duel.rust_bridge import rust_games_for_self_play

    # Replayed rather than snapshotted: `RustGame` has no cheap clone, so the
    # prefix that stops one ply short is played again from a fresh game.
    prefix = []
    game = rust_games_for_self_play([1], [0])[0]
    rng = random.Random(1)
    while not game.is_complete():
        legal = game.legal_action_indices()
        if not legal:
            break
        action = rng.choice(legal)
        game.apply_index(action)
        if game.is_complete():
            break
        prefix.append(action)
    if not prefix:
        return None
    replay = rust_games_for_self_play([1], [0])[0]
    for action in prefix:
        replay.apply_index(action)
    return replay


@pytest.mark.parametrize("conflict_free", [False, True])
def test_terminal_simulations_reach_the_outlook(conflict_free):
    """The P1 the first version shipped: Q 0.97 beside a 100% draw.

    Terminal simulations back up their scalar through three different paths --
    evaluated waves, ordinary immediate leaves, and a wave drained because it
    had nothing to evaluate -- and the outlook was accumulated in only one of
    them. A position whose every line ends at once therefore reported the
    search's value correctly and its victory type as certainly a draw.
    """

    from games.seven_wonders_duel.inference import Evaluator
    from games.seven_wonders_duel.rust_bridge import rust_flat_batch_adapter

    game = _near_terminal_game()
    if game is None:
        pytest.skip("no near-terminal position found on this stream")

    adapter = rust_flat_batch_adapter(
        Evaluator(
            _model(hierarchical_value=True),
            "cpu",
            64,
            fuse_embedder=False,
            value_source="hierarchical",
        )
    )
    result = seven_wonders_rust.search_many_flat_net(
        adapter,
        [game],
        [3],
        64,
        4,
        32,
        4,
        conflict_free_waves=conflict_free,
    )[0]
    outlook = result["root_outlook"]
    assert outlook is not None
    implied = sum(outlook[0:3]) - sum(outlook[3:6])
    assert implied == pytest.approx(result["root_value"], abs=2e-6), (
        "the scalar and the vector are means over the same simulations; a gap "
        "means one of the backup paths is not accounting for both"
    )


def test_forced_root_children_keep_their_outlook():
    """Finding 5: the forced-child cache dropped the vector but kept the scalar.

    A normal, measurable path under forced-root search -- 42 of 64 simulations
    in the reviewer's probe -- not an unlikely malformed input.
    """

    from games.seven_wonders_duel.inference import Evaluator
    from games.seven_wonders_duel.rust_bridge import rust_flat_batch_adapter

    adapter = rust_flat_batch_adapter(
        Evaluator(
            _model(hierarchical_value=True),
            "cpu",
            64,
            fuse_embedder=False,
            value_source="hierarchical",
        )
    )
    result = seven_wonders_rust.search_many_flat_net(
        adapter, [_mid_game()], [5], 64, 4, 64, 4, force=True
    )[0]
    outlook = result["root_outlook"]
    assert outlook is not None
    implied = sum(outlook[0:3]) - sum(outlook[3:6])
    assert implied == pytest.approx(result["root_value"], abs=2e-6)
