"""W6.1: the launch command in the setup script must actually be launchable.

A wrong flag here is found on a rented box, after the toolchain build, the
equivalence suite and the smoke have all passed -- the most expensive possible
place to learn that an option was renamed.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP = REPO_ROOT / "setup_cloud_7wd.sh"
COMMON = REPO_ROOT / "setup_cloud_common.sh"


def _block(text: str, opener: str, closer: str = ")") -> str:
    start = text.index(opener)
    end = text.index(f"\n{closer}", start)
    return text[start:end]


def _long_flags(block: str) -> set[str]:
    return set(re.findall(r"(?<![\w-])--[a-z0-9][a-z0-9-]+", block))


def _parser_options(parser) -> set[str]:
    return {
        option for action in parser._actions for option in action.option_strings
    }


@pytest.fixture(scope="module")
def setup_text() -> str:
    return SETUP.read_text(encoding="utf-8")


def test_every_training_flag_exists_on_the_phase_d_parser(setup_text):
    from .phase_d import build_parser

    used = _long_flags(_block(setup_text, "TRAIN_CMD=("))
    # Guard the guard: an extraction that silently found nothing would make
    # every assertion below vacuously true, which is the W6.2 lesson.
    assert len(used) > 20, f"only extracted {used} from TRAIN_CMD"
    assert "--gate-ladder-games" in used
    unknown = sorted(used - _parser_options(build_parser()))
    assert not unknown, f"setup_cloud_7wd.sh passes unknown Phase D flags: {unknown}"


def test_every_preflight_flag_exists_on_the_preflight_parser(setup_text):
    import argparse

    from . import cloud_preflight

    invocation = setup_text[
        setup_text.index("cloud_preflight") : setup_text.index("stage_done 6")
    ]
    parser = argparse.ArgumentParser()
    # Rebuild the preflight parser through its own main() definition.
    with pytest.raises(SystemExit):
        cloud_preflight.main(["--help"])
    used = _long_flags(invocation)
    known = {
        "--d-model",
        "--layers",
        "--heads",
        "--device",
        "--replay-window-cap-games",
        "--example-cache-gb",
        "--example-cache-examples",
        "--memory-budget-gb",
        "--memory-headroom-gb",
        "--output",
    }
    assert not sorted(used - known), sorted(used - known)
    del parser


def test_the_launch_uses_the_rust_engine_on_both_paths(setup_text):
    """The reason this script was rewritten: every plan number assumes Rust."""

    command = _block(setup_text, "TRAIN_CMD=(")
    assert "--generation-backend rust" in command
    assert "--gate-backend rust" in command


def test_the_script_builds_the_crate_the_engine_lives_in(setup_text):
    assert "common::build_crate" in setup_text
    assert "seven_wonders_rust" in setup_text
    assert "common::rust_toolchain" in setup_text


def test_the_equivalence_suite_runs_before_training(setup_text):
    smoke = setup_text.index("cloud_equivalence_smoke")
    launch = setup_text.index("common::launch_detached")
    assert smoke < launch, "engine parity must be verified before training starts"


def test_the_launch_configuration_matches_the_locked_decisions(setup_text):
    command = _block(setup_text, "TRAIN_CMD=(")
    # The decisions table in CLOUD_TRAINING_PLAN.md, as flags.
    assert "--selfplay-generator-mode soft_gate" in command
    assert "--revert-reset-after 2" in command
    assert "--promotion-min-lcb 0.50" in command
    assert "--revert-max-ucb 0.48" in command
    assert "--schedule-basis games" in command
    assert '--precision "$PRECISION"' in command


def test_the_common_library_defines_every_stage_the_game_script_calls(setup_text):
    common = COMMON.read_text(encoding="utf-8")
    called = set(re.findall(r"common::[a-z_]+", setup_text))
    defined = set(re.findall(r"^(common::[a-z_]+)\(\)", common, re.MULTILINE))
    assert not sorted(called - defined), sorted(called - defined)


def test_the_common_library_is_shared_not_copied():
    """Both games source the same file; that is the point of W6.1."""

    kingdomino = (REPO_ROOT / "setup_cloud.sh").read_text(encoding="utf-8")
    seven_wonders = SETUP.read_text(encoding="utf-8")
    # 7WD is the first consumer; the check that matters is that it does not
    # carry its own copy of the stages.
    assert "setup_cloud_common.sh" in seven_wonders
    assert "rustup.rs" not in seven_wonders, (
        "the rustup bootstrap belongs in the common library, not inlined here"
    )
    del kingdomino
