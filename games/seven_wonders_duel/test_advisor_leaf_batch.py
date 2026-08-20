"""The advisor and evaluation must search at the same leaf batch.

The promotion gate is what certifies the advisor. If the two batch differently
they run different algorithms at the root -- one selects under virtual loss and
the other does not -- and the advisor's reported numbers stop meaning what the
gate's mean. The two were 16 and 1 respectively before this was wired up.
"""

from __future__ import annotations

import pytest

from .advisor_adapter import ADVISOR_LEAF_BATCH
from .phase_d import PhaseDConfig


def _launcher_eval_leaf_batch() -> int:
    """The EVALUATION value the launcher ships, which is what a gate gets."""

    import re
    from pathlib import Path

    launcher = (
        Path(__file__).resolve().parents[2] / "setup_cloud_7wd.sh"
    ).read_text(encoding="utf-8")
    match = re.search(r'^EVAL_LEAF_BATCH="\$\{EVAL_LEAF_BATCH:-(\d+)\}"$', launcher, re.M)
    assert match, "the launcher no longer ships an --eval-leaf-batch default"
    return int(match.group(1))


def test_the_advisor_matches_the_leaf_batch_the_launcher_ships():
    """The invariant that is NOT tautological.

    The first version of this test built a config FROM ADVISOR_LEAF_BATCH and
    asserted equality with it -- always true, and it survived a mutation that
    moved the advisor to 16. The real coupling is between the advisor's default
    and the value the launcher passes to a run, because those are the two
    numbers that meet in production.
    """

    shipped = _launcher_eval_leaf_batch()
    assert ADVISOR_LEAF_BATCH == shipped, (
        f"the advisor searches at {ADVISOR_LEAF_BATCH} and the launcher gives "
        f"evaluation {shipped}. The gate certifies the advisor; if they batch "
        "differently the gate certifies a search the advisor does not run."
    )
    # Training is deliberately DIFFERENT: it batches across hundreds of games
    # already, and it is the only one of the three producing policy targets.
    config = PhaseDConfig(run_dir="x", leaf_batch=6, eval_leaf_batch=shipped,
                          virtual_loss_root=True)
    config.validate()
    assert config.evaluation_leaf_batch() == shipped
    assert config.leaf_batch == 6


def test_the_launcher_passes_the_leaf_batching_it_configures():
    """A flag the launcher does not pass is a default nobody chose -- the same
    failure as the missing --weight-decay. All of this work reaches a run only
    through TRAIN_CMD."""

    from pathlib import Path

    launcher = (
        Path(__file__).resolve().parents[2] / "setup_cloud_7wd.sh"
    ).read_text(encoding="utf-8")
    command = launcher[launcher.index("TRAIN_CMD=(") :]
    command = command[: command.index("\n)")]
    for flag in ("--leaf-batch", "--cheap-leaf-batch", "--eval-leaf-batch",
                 "LEAF_BATCH_FLAGS"):
        assert flag in command, f"TRAIN_CMD does not pass {flag}"
    # The array must be BUILT before TRAIN_CMD: an array referenced in the
    # literal expands at construction, so defining it afterwards drops the
    # flags silently -- which is how this was first written.
    assert launcher.index("LEAF_BATCH_FLAGS=()") < launcher.index("TRAIN_CMD=(")


def test_the_advisor_default_is_a_named_constant_not_a_literal():
    """It was `req.options.get("leaf_batch", 16)` -- a literal in a call, with
    nothing tying it to the value evaluation uses."""

    from pathlib import Path

    source = (Path(__file__).with_name("advisor_adapter.py")).read_text(
        encoding="utf-8"
    )
    assert 'req.options.get("leaf_batch", 16)' not in source
    assert 'req.options.get("leaf_batch", ADVISOR_LEAF_BATCH)' in source


def test_a_caller_may_still_override_it_deliberately():
    """Overriding per request is how the difference gets MEASURED; what must not
    happen is the two drifting apart silently."""

    from pathlib import Path

    source = (Path(__file__).with_name("advisor_adapter.py")).read_text(
        encoding="utf-8"
    )
    assert 'req.options.get("leaf_batch"' in source, "the override must survive"


@pytest.mark.parametrize("value", [1, 4, 6, 8])
def test_evaluation_follows_training_for_any_shipped_value(value):
    """The coupling is structural, not a coincidence of the current number."""

    config = PhaseDConfig(run_dir="x", leaf_batch=value, virtual_loss_root=True)
    config.validate()
    assert config.evaluation_leaf_batch() == value
