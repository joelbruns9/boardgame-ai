"""
Trajectory capture and replay for the supervised bootstrap.

WHY TRAJECTORIES AND NOT TENSORS
────────────────────────────────
A game is ~100 decisions per seat. Storing encoded samples costs a few thousand
floats each, so five thousand games is well over a **gigabyte** of float32.
Storing the *trajectory* — a seed, a config and 75 small integers — is about
**1 MB** for the same games, and the engine is deterministic given the seed, so
replaying reproduces the encoded samples exactly.

So: capture trajectories, replay them into tensors at training time. The replay
also re-runs the rules, which means a rules change invalidates old data loudly
(the action indices stop being legal) instead of silently training on stale
encodings.

THE BOOTSTRAP THIS IS FOR
─────────────────────────
Before any search, train the network on :class:`~games.welcome_to.bots.GreedyBot`
games: policy target = the action greedy chose, value target = the final
score, plus the auxiliary heads from :mod:`games.welcome_to.training`. It is
behaviour cloning, and its point is not the resulting player — it is that the
whole pipeline (encoder → network → loss → checkpoint) gets proved out in an hour
against a known reference, before MCTS is in the picture. If the net cannot
reproduce greedy's 33.2 mean score without search, nothing downstream will work.

It also gives a warm start, and a clean ablation target: the trained net is
supposed to *beat* greedy once search is added, and beating your own teacher is a
visible milestone.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import numpy as np

from games.welcome_to import encoder as enc
from games.welcome_to import macro_codec as mc
from games.welcome_to import training
from games.welcome_to.game import GameConfig, GameState

#: A policy is anything that maps a state to a legal action index.
Policy = Callable[[GameState], int]


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One complete game, small enough to keep by the million."""

    seed: int
    players: int
    advanced: bool
    expert: bool
    solo_rules: bool
    actions: tuple[int, ...]
    scores: tuple[int, ...]

    @property
    def config(self) -> GameConfig:
        return GameConfig(
            players=self.players,
            advanced=self.advanced,
            expert=self.expert,
            solo_rules=self.solo_rules,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "Trajectory":
        raw = json.loads(line)
        return cls(
            seed=raw["seed"],
            players=raw["players"],
            advanced=raw["advanced"],
            expert=raw["expert"],
            solo_rules=raw["solo_rules"],
            actions=tuple(raw["actions"]),
            scores=tuple(raw["scores"]),
        )


@dataclass(frozen=True, slots=True)
class Sample:
    """One decision, ready to batch.

    The four encoder arrays are kept apart rather than concatenated: the seat
    axis of ``sheet_planes`` and ``sheet_scalars`` is what the shared per-sheet
    encoder runs over, and flattening it here would throw away the structure the
    network is built around.  ``targets`` is indexed along the same seat axis.
    """

    sheet_planes: np.ndarray
    sheet_scalars: np.ndarray
    viewer_plane: np.ndarray
    global_scalars: np.ndarray
    legal: np.ndarray
    action: int
    actor: int
    turn: int
    targets: dict[str, float | tuple[float, ...]]


def play_trajectory(
    policy: Policy,
    seed: int,
    config: Optional[GameConfig] = None,
    max_steps: int = 20000,
) -> Trajectory:
    """Run one game under ``policy`` and record what it did."""
    # Multiplayer by default -- a one-seat game switches scoring rules, so a
    # single-seat corpus trains on a different game.  See SELF_PLAY_PLAN.md.
    config = config or GameConfig(players=2, advanced=True, solo_rules=False)
    state = GameState.new(seed=seed, config=config)
    actions: list[int] = []
    for _ in range(max_steps):
        if state.is_terminal:
            break
        action = policy(state)
        actions.append(action)
        state.apply(action)
    else:
        raise RuntimeError("game did not terminate")

    return Trajectory(
        seed=seed,
        players=config.players,
        advanced=config.advanced,
        expert=config.expert,
        solo_rules=config.solo_rules,
        actions=tuple(actions),
        scores=tuple(state.scores()),
    )


def replay(trajectory: Trajectory) -> Iterator[Sample]:
    """Re-run a trajectory, yielding an encoded sample per **macro** decision.

    The stored trajectory is primitive; the labels are not.  ``macro_codec``
    collapses each ``CHOOSE_STACK -> (WRITE | PERMIT_REFUSAL)`` pair into one
    action, so no sample is emitted at ``WRITE_NUMBER`` -- under the frozen
    vocabulary the network is never asked there.  Keeping the corpus primitive
    is what lets the vocabulary change without recapturing a single game.

    Raises if a recorded action is no longer legal, which is what makes a rules
    change invalidate stale data loudly instead of quietly.
    """
    state = GameState.new(seed=trajectory.seed, config=trajectory.config)
    visits: list[tuple] = []

    for visited, macro in mc.collapse(state, trajectory.actions):
        actor = visited.actor
        visits.append(
            (
                enc.encode_state(visited),
                mc.legal_mask(visited),
                macro,
                actor,
                visited.turn,
                # the seat axis is captured with the encoding, because the
                # targets have to be indexed by the same one
                enc.seat_order(visited, actor),
            )
        )

    if not state.is_terminal:
        raise ValueError("trajectory does not reach a terminal state")
    if tuple(state.scores()) != trajectory.scores:
        raise ValueError(
            f"replay diverged: scores {state.scores()} != recorded {trajectory.scores}"
        )

    outcomes = training.final_outcomes(state)
    for encoded, legal, action, actor, turn, order in visits:
        sheet_planes, sheet_scalars, viewer_plane, global_scalars = encoded
        yield Sample(
            sheet_planes=sheet_planes,
            sheet_scalars=sheet_scalars,
            viewer_plane=viewer_plane,
            global_scalars=global_scalars,
            legal=legal,
            action=action,
            actor=actor,
            turn=turn,
            targets=training.sample_targets(outcomes, order, turn),
        )


def batch(samples: Sequence[Sample]) -> dict[str, np.ndarray]:
    """Stack samples into arrays ready for a training step."""
    n = len(samples)
    out = {
        "sheet_planes": np.stack([s.sheet_planes for s in samples]),
        "sheet_scalars": np.stack([s.sheet_scalars for s in samples]),
        "viewer_plane": np.stack([s.viewer_plane for s in samples]),
        "global_scalars": np.stack([s.global_scalars for s in samples]),
        "legal": np.stack([s.legal for s in samples]),
        "action": np.asarray([s.action for s in samples], dtype=np.int64),
        "turn": np.asarray([s.turn for s in samples], dtype=np.float32),
    }
    for name in training.TARGET_NAMES:
        out[name] = np.asarray([s.targets[name] for s in samples], dtype=np.float32)
    assert out["sheet_planes"].shape == (n, *enc.SHEET_PLANES_SHAPE)
    assert out["sheet_scalars"].shape == (n, enc.MAX_SEATS, enc.NUM_SHEET_SCALAR)
    assert out["viewer_plane"].shape == (n, *enc.VIEWER_PLANE_SHAPE)
    assert out["global_scalars"].shape == (n, enc.NUM_GLOBAL_SCALAR)
    assert out["legal"].shape == (n, mc.NUM_MACRO_ACTIONS)
    for name in training.PER_SEAT_TARGETS:
        assert out[name].shape == (n, training.MAX_SEATS), name
    for name in training.GLOBAL_TARGETS:
        assert out[name].shape == (n,), name
    return out


def write_trajectories(path: str | Path, trajectories: Sequence[Trajectory]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(trajectory.to_json() + "\n")
    return path


def read_trajectories(path: str | Path) -> list[Trajectory]:
    with Path(path).open(encoding="utf-8") as handle:
        return [Trajectory.from_json(line) for line in handle if line.strip()]


def generate(
    games: int,
    policy_factory: Callable[[random.Random], Policy],
    config: Optional[GameConfig] = None,
    seed: int = 0,
) -> list[Trajectory]:
    """Play ``games`` games, each with its own card stream and policy RNG."""
    return [
        play_trajectory(policy_factory(random.Random(seed * 1_000_003 + g)), seed + g, config)
        for g in range(games)
    ]


def main(argv: Optional[list[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse

    from games.welcome_to.bots import GreedyBot

    parser = argparse.ArgumentParser(description="capture bootstrap trajectories")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--players", type=int, default=1)
    parser.add_argument(
        "--base-game",
        action="store_true",
        help="drop the advanced variant (roundabouts + extra plans); "
        "advanced is the primary training target",
    )
    parser.add_argument("--out", default="welcome_to_bootstrap.jsonl")
    args = parser.parse_args(argv)

    config = GameConfig(
        players=args.players,
        advanced=not args.base_game,
        solo_rules=args.players > 1,
    )
    trajectories = generate(
        args.games, lambda rng: GreedyBot(rng).act, config, seed=args.seed
    )
    path = write_trajectories(args.out, trajectories)
    decisions = sum(len(t.actions) for t in trajectories)
    scores = [s for t in trajectories for s in t.scores]
    print(f"wrote {len(trajectories)} games / {decisions} decisions to {path}")
    print(f"  bytes on disk   {path.stat().st_size:,}")
    print(f"  mean score      {sum(scores) / len(scores):.2f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
