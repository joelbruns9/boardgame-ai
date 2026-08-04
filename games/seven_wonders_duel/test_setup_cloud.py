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
    from .cloud_preflight import build_parser

    invocation = setup_text[
        setup_text.index("cloud_preflight") : setup_text.index("stage_done 6")
    ]
    used = _long_flags(invocation)
    assert len(used) > 5, f"only extracted {used} from the preflight invocation"
    unknown = sorted(used - _parser_options(build_parser()))
    assert not unknown, f"setup_cloud_7wd.sh passes unknown preflight flags: {unknown}"


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
    assert '--bootstrap-policy "$BOOTSTRAP_POLICY"' in command
    assert '--promotion-every "$PROMOTION_EVERY"' in command
    assert '--revert-reset-after "$REVERT_RESET_AFTER"' in command
    assert '--probation-reset-after "$PROBATION_RESET_AFTER"' in command
    assert "--promotion-min-lcb 0.50" in command
    assert "--revert-max-ucb 0.48" in command
    assert "--schedule-basis games" in command
    assert '--precision "$PRECISION"' in command


def test_the_run03_lifecycle_defaults_match_the_documented_command(setup_text):
    expected = {
        "PROMOTION_EVERY": "5",
        "BOOTSTRAP_POLICY": "auto_first_trained",
        "PROBATION_RESET_AFTER": "4",
        "REVERT_RESET_AFTER": "3",
        "GATE_LADDER": "200 600 1000 1500",
    }
    for name, value in expected.items():
        assert f'{name}="${{{name}:-{value}}}"' in setup_text

    parameters = (REPO_ROOT / "games/seven_wonders_duel/training_parameters.md").read_text(
        encoding="utf-8"
    )
    for flag, value in (
        ("--promotion-every", "5"),
        ("--bootstrap-policy", "auto_first_trained"),
        ("--probation-reset-after", "4"),
        ("--revert-reset-after", "3"),
    ):
        assert f"{flag} {value}" in parameters
    assert "--gate-ladder-games 200 600 1000 1500" in parameters


def test_train_steps_are_derived_from_games_per_iteration(setup_text):
    """`--train-steps` must be passed, and must track games per iteration.

    The parser default is 300 regardless of how many games an iteration
    produces. At the shipped 1,000 games that is ~8x sample reuse and at 500 it
    is ~16x, against the ~5x this loop is tuned for -- and, like the lifecycle
    flags before the run-03 remediation, a flag the launcher does not pass is a
    default nobody chose.
    """

    command = _block(setup_text, "TRAIN_CMD=(")
    for flag in ("--train-steps", "--train-warmup-steps", "--train-batch-size"):
        assert flag in command, f"the launch command does not pass {flag}"

    # Derived in the script, not hard-coded: changing GAMES_PER_ITERATION must
    # carry the step budget with it.
    assert 'TRAIN_STEPS="${TRAIN_STEPS:-$(( (GAMES_PER_ITERATION * 19 + 99) / 100 ))}"' in setup_text
    assert 'TRAIN_WARMUP_STEPS="${TRAIN_WARMUP_STEPS:-$(( TRAIN_STEPS / 3 ))}"' in setup_text

    games = int(re.search(r'GAMES_PER_ITERATION="\$\{GAMES_PER_ITERATION:-(\d+)\}"', setup_text).group(1))
    steps = (games * 19 + 99) // 100
    # ~19.4 recorded positions a game at batch 512: between 4x and 6x reuse.
    reuse = steps * 512 / (games * 19.4)
    assert 4.0 <= reuse <= 6.0, f"{steps} steps at {games} games is {reuse:.1f}x reuse"
    # The parser's warmup default would otherwise exceed the whole budget.
    assert steps // 3 < steps


def test_the_launch_is_sized_for_the_two_hundred_thousand_game_run(setup_text):
    defaults = {
        "ITERATIONS": "200",
        "GAMES_PER_ITERATION": "1000",
        "SELF_ANCHOR_GAMES": "400",
    }
    for name, value in defaults.items():
        assert f'{name}="${{{name}:-{value}}}"' in setup_text


def test_the_preflight_is_told_the_length_of_the_run(setup_text):
    """The disk budget is only meaningful if the preflight knows the plan.

    Checkpoints are written per iteration and never pruned, so disk scales with
    the run's length -- and disk is fixed when the instance is rented.
    """

    invocation = setup_text[
        setup_text.index("cloud_preflight") : setup_text.index("stage_done 6")
    ]
    for flag in (
        "--iterations",
        "--games-per-iteration",
        "--seed-games",
        "--promotion-every",
        "--run-dir",
        "--disk-budget-gb",
    ):
        assert flag in invocation, f"the preflight is not told {flag}"


def test_the_smoke_can_run_the_launch_geometry(setup_text):
    """`--plumbing-smoke` must survive the width the launch actually uses.

    The smoke shrinks the model to 32x1, so an explicit `--heads 6` -- the
    shipped value, which does not divide 32 -- used to abort in `build_model`.
    The box's own smoke stage passes no width and so never saw it; anyone
    smoking the real launch flag set did.
    """

    from dataclasses import replace

    from .phase_d import PhaseDConfig, smoke_config
    from .train import build_model

    def default(name: str) -> int:
        return int(re.search(rf'{name}="\$\{{{name}:-(\d+)\}}"', setup_text).group(1))

    launch = replace(
        PhaseDConfig(),
        d_model=default("D_MODEL"),
        layers=default("LAYERS"),
        heads=default("HEADS"),
    )
    assert (launch.d_model, launch.layers, launch.heads) == (384, 8, 6)

    smoked = smoke_config(launch)
    assert smoked.heads is None, "the smoke must drop the launch head count"
    build_model("transformer", smoked.d_model, smoked.layers, smoked.heads)


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


# -- stage 8b: the sweep invocations must actually parse ---------------------
#
# Both sweep calls in stage 8b were wrong and would have killed the setup
# script on any box where SWEEP_CHECKPOINT was set: --work-dir does not exist
# on f4_phase_d_sweep, its --output is a directory not a file, and its axes are
# comma-separated strings while w5_gate_slots_sweep takes space-separated
# lists. Two harnesses, two conventions, and nothing checked either.


def _invocation(text: str, module: str) -> str:
    """The raw shell text of stage 8b's call to one sweep harness."""

    start = text.index(f'"$PY" -m games.seven_wonders_duel.{module}')
    end = text.index("|| die", start)
    return text[start:end]


def _module_options(module: str) -> set[str]:
    """Flags the module's real parser accepts, from its own --help.

    A subprocess rather than importing and introspecting: both parsers are
    built inside main(), and scraping --help is what an operator's shell would
    hit anyway.
    """

    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", f"games.seven_wonders_duel.{module}", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"{module} --help failed: {result.stderr[-400:]}"
    return set(re.findall(r"(?<![\w-])--[a-z0-9][a-z0-9-]+", result.stdout))


@pytest.mark.parametrize(
    "module", ["f4_phase_d_sweep", "w5_gate_slots_sweep"]
)
def test_the_sweep_invocations_only_use_flags_that_exist(setup_text, module):
    used = _long_flags(_invocation(setup_text, module))
    assert len(used) >= 6, f"only extracted {used} from the {module} call"
    unknown = sorted(used - _module_options(module))
    assert not unknown, f"stage 8b passes flags {module} rejects: {unknown}"


def test_the_generation_sweep_passes_comma_separated_axes(setup_text):
    """f4_phase_d_sweep takes one string per axis, not a list.

    `--slots 48 96 144` parses as --slots=48 plus three stray positionals, and
    argparse rejects the whole command. That is the original bug.
    """

    block = _invocation(setup_text, "f4_phase_d_sweep")
    for axis in ("--slots", "--caps", "--inflight"):
        match = re.search(rf'{axis}\s+"([^"]*)"', block)
        assert match, f"{axis} must be quoted as a single argument: {block}"
        assert " " not in match.group(1), (
            f"{axis} got {match.group(1)!r}; this harness splits on commas"
        )
    assert "--work-dir" not in block, (
        "f4_phase_d_sweep has no --work-dir; its --output is a directory"
    )


def test_the_gate_sweep_passes_a_work_dir_and_a_file_output(setup_text):
    block = _invocation(setup_text, "w5_gate_slots_sweep")
    assert "--work-dir" in block, "this harness does take a --work-dir"
    assert ".json" in block, "its --output is a file, not a directory"


def test_the_sweep_env_handoff_matches_the_launcher(setup_text):
    """measured_env.sh must set variables the launcher actually reads."""

    from .sweep_launch_env import render

    rendered = render(
        {
            "RUST_SLOTS": 96,
            "RUST_GLOBAL_BATCH_CAP": 1024,
            "RUST_MAX_INFLIGHT_BATCHES": 1,
            "GATE_SLOTS": 144,
            "GATE_GLOBAL_BATCH_CAP": 1024,
        }
    )
    exported = [
        line.removeprefix("export ").split("=", 1)[0]
        for line in rendered.splitlines()
        if line.startswith("export ")
    ]
    assert "SKIP_SWEEPS" in exported, "pass 2 must not re-measure"
    for name in exported:
        assert f'{name}="${{{name}:-' in setup_text, (
            f"measured_env.sh exports {name}, but the launcher never reads it"
        )
