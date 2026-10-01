"""M0-B's second half: a trajectory only replays against the generator that
dealt it (``RUST_PORT_PLAN.md`` M0-B).

A seed is not a game.  ``datagen.replay`` rebuilds the deal from
``GameState.new(seed=...)``, so a trajectory captured under ``random.Random``
and replayed under the portable RNG meets a **different deck** — the recorded
actions become illegal, or the scores diverge.  The port therefore records
*which* generator dealt each trajectory, and the default is the one that keeps
the pre-port corpus valid.

⚠ These are cheap tests guarding an expensive failure: a corpus that silently
stops replaying is a data-loss bug, and the loudest symptom it would otherwise
produce is a training run quietly learning from nothing.
"""

from __future__ import annotations

import json
import random

import pytest

from games.welcome_to import datagen
from games.welcome_to.bots import GreedyBot
from games.welcome_to.game import DEFAULT_RNG_KIND, GameConfig, GameState

CONFIG = GameConfig(players=2, advanced=True, expert=False, solo_rules=False)


def _play(seed: int, rng_kind: str) -> tuple[list[int], list[int]]:
    """One greedy game, as recorded actions and final scores."""
    state = GameState.new(seed=seed, config=CONFIG, rng_kind=rng_kind)
    bot = GreedyBot(random.Random(0))
    actions: list[int] = []
    while not state.is_terminal:
        action = bot.act(state)
        actions.append(action)
        state.apply(action)
    return actions, list(state.scores())


def test_a_trajectory_line_without_an_rng_field_is_a_cpython_deal():
    """Every trajectory captured before the port has no ``rng`` key, and the
    default is what keeps that corpus replayable."""
    actions, scores = _play(seed=42, rng_kind="cpython")
    line = json.dumps(
        {
            "seed": 42,
            "players": 2,
            "advanced": True,
            "expert": False,
            "solo_rules": False,
            "actions": actions,
            "scores": scores,
        }
    )
    trajectory = datagen.Trajectory.from_json(line)
    assert trajectory.rng == "cpython"
    assert list(datagen.replay(trajectory))


def test_a_freshly_captured_trajectory_records_the_portable_deal():
    trajectory = datagen.play_trajectory(
        GreedyBot(random.Random(1)).act, seed=7, config=CONFIG
    )
    assert trajectory.rng == DEFAULT_RNG_KIND == "portable"
    assert '"rng"' in trajectory.to_json()
    assert datagen.Trajectory.from_json(trajectory.to_json()) == trajectory
    assert list(datagen.replay(trajectory))


def test_the_two_generators_deal_different_games_from_the_same_seed():
    """The reason the field exists.  If this ever stopped being true the field
    would be harmless — and so would a corpus replayed under the wrong one."""
    cpython_actions, cpython_scores = _play(seed=42, rng_kind="cpython")
    portable_actions, portable_scores = _play(seed=42, rng_kind="portable")
    assert (cpython_actions, cpython_scores) != (portable_actions, portable_scores)


def test_replaying_a_trajectory_under_the_wrong_generator_fails_loudly():
    """Not silently: a mislabelled corpus must raise, not train on a different
    game than the one that was played."""
    actions, scores = _play(seed=42, rng_kind="cpython")
    mislabelled = datagen.Trajectory(
        seed=42,
        players=2,
        advanced=True,
        expert=False,
        solo_rules=False,
        actions=tuple(actions),
        scores=tuple(scores),
        rng="portable",  # wrong: this game was dealt by random.Random
    )
    with pytest.raises(ValueError):
        list(datagen.replay(mislabelled))


def test_an_unknown_generator_name_is_refused_at_construction():
    with pytest.raises(ValueError, match="rng kind"):
        GameState.new(seed=1, config=CONFIG, rng_kind="mersenne")
