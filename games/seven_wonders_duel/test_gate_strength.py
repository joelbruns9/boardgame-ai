"""Outcome-level regression for the two-net gate path.

The seat-routing bug survived 448 tests because every test that touched search
used a SINGLE network, so nothing ever exercised two different nets inside one
arena.  The tests added with the fix close that gap at the mechanism level --
`test_two_net_arena_searches_each_move_under_the_movers_own_net` asserts a
batch never mixes values from both nets, and
`test_two_net_arena_plays_the_improved_policy_argmax` pins the played action.

Neither would fail if the gate were broken some *other* way.  This module tests
the property the gate actually has to have: **a stronger network must beat a
weaker one, and the margin must not shrink as search deepens.**  That is the
shape the routing bug destroyed -- iter0 scored 0.125 against a random-init net
at 2 sims and 0.344 at 32, dropping toward the wrong side as simulations grew,
because every extra simulation added another leaf scored by the wrong net.

Real strength cannot be synthesised: a randomly-built net has no meaningful
prior or value, so "strong vs weak" needs a trained checkpoint.  These tests are
therefore integration tests over run artifacts and skip when none are present.
Point them at any run with `SEVEN_WD_STRENGTH_CKPTS=strong.pt:weak.pt`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from .eval_suite import ArenaSettings, run_model_match

# Iteration 11 of the 10h laptop run against the random-init bootstrap net.  A
# trained net versus an untrained one is the largest strength gap available
# without training anything, which keeps the thresholds below far from the
# noise floor.
DEFAULT_RUN = (
    Path(__file__).parent / "runs" / "laptop_training_10h_02" / "checkpoints"
)
DEFAULT_STRONG = DEFAULT_RUN / "latest.pt"
DEFAULT_WEAK = DEFAULT_RUN / "_bootstrap_init.pt"

# Below this the pairing is indistinguishable from a coin flip at 32 games
# (+/-17%), and any value under 0.5 means the arena has inverted the result.
DOMINANCE = 0.75

# Search may legitimately compress a gap -- a weak net's value head can recover
# some of what its prior gives away -- so this does not demand monotonicity.
# It demands the gap not COLLAPSE with depth, which is what the bug did.
MAX_EROSION = 0.10

SIMS = (2, 8, 32)
GAMES = 32


def _checkpoints() -> tuple[Path, Path]:
    override = os.environ.get("SEVEN_WD_STRENGTH_CKPTS")
    if override:
        strong, _, weak = override.partition(":")
        return Path(strong), Path(weak)
    return DEFAULT_STRONG, DEFAULT_WEAK


def _settings(sims: int, tmp_path: Path) -> ArenaSettings:
    device = "cuda"
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass
    # d_model/layers must match the checkpoint being loaded, not the defaults.
    return ArenaSettings(
        work_dir=str(tmp_path), sims=sims, device=device, d_model=128, layers=4
    )


def _score(sims: int, tmp_path: Path) -> float:
    strong, weak = _checkpoints()
    summary, _ = run_model_match(
        _settings(sims, tmp_path),
        f"strength_sims_{sims}",
        ("strong", strong),
        ("weak", weak),
        GAMES,
        # Offset well clear of the eval suite's ranges so a failure here can be
        # reproduced without colliding with recorded match seeds.
        95_000_000 + sims * 10_000,
    )
    return summary["score_rate"]


@pytest.fixture(scope="module")
def scores(tmp_path_factory) -> dict[int, float]:
    strong, weak = _checkpoints()
    for path in (strong, weak):
        if not path.is_file():
            pytest.skip(f"no checkpoint at {path}; set SEVEN_WD_STRENGTH_CKPTS")
    root = tmp_path_factory.mktemp("gate_strength")
    measured = {sims: _score(sims, root / str(sims)) for sims in SIMS}
    # A pass/fail alone cannot tell you whether the margin is comfortable or a
    # hair above threshold, and the depth gradient is the diagnostic signal --
    # print it so `-s` (or any failure) shows the shape, not just the verdict.
    print(f"\nstrength vs depth ({GAMES} games/point, dominance >= {DOMINANCE}):")
    for sims, score in measured.items():
        print(f"  sims={sims:<4d} score={score:.4f}")
    return measured


@pytest.mark.slow
@pytest.mark.parametrize("sims", SIMS)
def test_stronger_net_beats_weaker_net(scores, sims: int):
    """The gate's core promise: the better network wins the match."""

    assert scores[sims] >= DOMINANCE, (
        f"trained net scored {scores[sims]:.3f} vs an untrained net at "
        f"{sims} sims; the gate cannot rank checkpoints it cannot separate"
    )


@pytest.mark.slow
def test_advantage_does_not_erode_with_search_depth(scores):
    """Deeper search must not hand the advantage back.

    This is the assertion the routing bug failed.  Search is a policy
    improvement operator applied to the mover's own network, so more
    simulations cannot systematically favour the weaker side.  A negative
    gradient here means simulations are feeding something other than the
    mover's own evaluation into the tree.
    """

    shallow, deep = scores[SIMS[0]], scores[SIMS[-1]]
    assert deep >= shallow - MAX_EROSION, (
        f"advantage eroded from {shallow:.3f} at {SIMS[0]} sims to "
        f"{deep:.3f} at {SIMS[-1]} sims; deeper search is favouring the "
        f"weaker net, the signature of mixed-network evaluation"
    )
