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


def test_every_budget_the_preflight_sizes_is_also_given_to_the_run(setup_text):
    """A budget checked at setup but not passed to training checks nothing.

    `EXAMPLE_CACHE_GB` was in the preflight invocation and absent from
    TRAIN_CMD, so the preflight sized host memory against a cache ceiling the
    run never received -- it used the parser default instead. Same shape as the
    lifecycle flags and `--train-steps` before them.
    """

    preflight = setup_text[
        setup_text.index("cloud_preflight") : setup_text.index("stage_done 6")
    ]
    command = _block(setup_text, "TRAIN_CMD=(")
    for flag in ("--example-cache-gb", "--memory-budget-gb"):
        assert flag in preflight
        assert flag in command, f"{flag} is sized by the preflight but never passed"


def test_process_workers_does_not_default_to_a_many_core_box_count(setup_text):
    """192 processes, each importing torch, for a stage the Rust path skips."""

    assert 'PROCESS_WORKERS="${PROCESS_WORKERS:-$(nproc)}"' not in setup_text
    assert "PROCESS_WORKERS" in setup_text
    assert "-gt 16" in setup_text


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


def test_a_crashing_preflight_is_not_reported_as_a_refusal(setup_text):
    """Exit 1 means "this box is too small"; anything else means the check broke.

    They shared one `die` message, so a FileNotFoundError writing the report
    told the operator to destroy the instance and rent a bigger one.
    """

    stage = setup_text[
        setup_text.index("STAGE 6") : setup_text.index("stage_done 6")
    ]
    assert '_preflight_status" -eq 1' in stage, "refusal is not distinguished by exit code"
    assert "CRASHED" in stage
    assert "not a verdict on this box" in stage
    # The advice that only makes sense for a real refusal must stay on that branch.
    refusal, crash = stage.split("elif", 1)
    assert "rent a bigger one" in refusal
    assert "rent a bigger one" not in crash


def test_operator_supplied_paths_are_checked_before_anything_is_built(setup_text):
    """A missing scp costs seconds, not an hour of toolchain build.

    Every one of these paths arrives from another machine, so "not uploaded
    yet" is the ordinary failure -- and it used to surface at the stage that
    consumed it, after rustup, torch and the crate build had all completed.
    """

    call = setup_text.index("require_operator_files \\")
    assert call < setup_text.index('stage 1 "Rust toolchain'), (
        "operator files are checked after the build has already started"
    )
    for name in (
        "PRECISION_ARENA_CHECKPOINT",
        "SWEEP_CHECKPOINT",
        "LAUNCH_FLAGS_JSON",
    ):
        assert f'"{name}=${{{name}:-}}"' in setup_text


def test_nothing_before_the_clone_depends_on_the_shared_library(setup_text):
    """The common library comes from the checkout stage 2 has not updated yet.

    This script is curl'd fresh every run, but on a box with an existing clone
    the library beside it is whatever the last run left there. So a new
    `common::` function called before stage 2 is "command not found" on exactly
    the boxes that have run this before -- which is where it was found.

    The allowlist is pinned rather than derived: a test cannot know what is
    deployed on some box, but it can make *adding* to this set a deliberate act
    rather than an accident. Both entries predate every copy in the field.
    """

    deployed_everywhere = {"common::require_python", "common::rust_toolchain"}
    prologue = setup_text[: setup_text.index('stage 2 "Clone repo')]
    called = set(re.findall(r"^\s*(common::[a-z_]+)", prologue, re.MULTILINE))
    assert called <= deployed_everywhere, (
        "called before the clone is updated, so an old checkout's library will "
        f"not have it: {sorted(called - deployed_everywhere)}. Define it in "
        "setup_cloud_7wd.sh instead, which is curl'd fresh."
    )


def test_an_arena_that_could_not_run_is_not_reported_as_a_verdict(setup_text):
    """Exit 1 means bf16 really differs; anything else means the check broke.

    `precision_arena` exits 1 when the precisions disagree and 2 (argparse) when
    it cannot run at all, so the two conclusions are distinguishable -- they were
    not when a missing checkpoint printed "bf16 differs from fp32".
    """

    stage = setup_text[
        setup_text.index("STAGE 8:") : setup_text.index("stage_done 8")
    ]
    from .precision_arena import DISAGREEMENT_EXIT_CODE

    assert f'_arena_status" -eq {DISAGREEMENT_EXIT_CODE}' in stage, (
        "the launcher does not use the arena's own disagreement exit code"
    )
    verdict, broke = stage.split("elif", 1)
    assert "bf16 differs from fp32" in verdict
    assert "PRECISION=fp32" not in broke
    assert "NOT a verdict on bf16" in broke


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


def test_both_sweeps_measure_at_the_precision_the_run_will_use(setup_text):
    """A geometry chosen at fp32 is chosen against the wrong cost curve.

    W0 measured bf16 at 1.69x on L, and the cap/slot optimum is a throughput
    optimum -- so sweeping at a precision the run does not use picks settings
    for a machine that is not the one being configured. The gate sweep already
    passed $PRECISION; the generation sweep did not.
    """

    for module in ("f4_phase_d_sweep", "w5_gate_slots_sweep"):
        block = _invocation(setup_text, module)
        assert '--precision "$PRECISION"' in block, f"{module} sweeps at a fixed precision"


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


# --- omission, not typos ----------------------------------------------------
#
# Every test above checks that what the script DOES pass is valid. None of them
# noticed that until 2026-08-19 the script passed no sims, no search mode, no
# Dirichlet and no solver at all -- so a launch from it took the parser's
# laptop defaults: 16-24 cheap sims against the shipped 100, 64-128 full against
# 1600, Gumbel where the plan ships PUCT, and the endgame solver off. cloud6's
# command carried them by hand, so the gap never showed.
#
# A flag left to its default is the failure mode these cover.


def test_the_launch_sets_the_search_budget_rather_than_inheriting_it(setup_text):
    """Parser defaults are laptop-scale: 16-24 cheap and 64-128 full."""

    used = _long_flags(_block(setup_text, "TRAIN_CMD=("))
    for flag in (
        "--cheap-sims-min",
        "--cheap-sims-max",
        "--full-sims-min",
        "--full-sims-max",
        "--full-search-fraction",
        "--top-k",
    ):
        assert flag in used, f"{flag} left to the parser default"


def test_the_launch_sets_every_search_mode(setup_text):
    """The eval mode is the one that matters most: the advisor deploys under
    PUCT, so a gate run under Gumbel promotes on a number nobody will see
    again."""

    used = _long_flags(_block(setup_text, "TRAIN_CMD=("))
    for flag in (
        "--selfplay-search-mode",
        "--cheap-search-mode",
        "--eval-search-mode",
        "--dirichlet-epsilon",
        "--dirichlet-alpha",
        "--forced-playout-k",
    ):
        assert flag in used, f"{flag} left to the parser default"


def test_the_launch_carries_the_architecture_switches(setup_text):
    """--pooled-readout and --reply-head change which parameters exist, so a
    run that omits them trains a different model than the plan describes."""

    assert "ARCH_FLAGS+=(--pooled-readout)" in setup_text
    assert "ARCH_FLAGS+=(--reply-head)" in setup_text
    assert '"${ARCH_FLAGS[@]}"' in _block(setup_text, "TRAIN_CMD=(")


def test_the_launch_configures_the_endgame_solver(setup_text):
    """The solver is the differentiator this run is built around, and it is OFF
    by default -- `--endgame-solver-max-nodes` defaults to 0."""

    assert "--endgame-solver-max-nodes" in setup_text
    assert "--endgame-cost-model" in setup_text
    assert "--solver-threads" in setup_text
    assert '"${SOLVER_FLAGS[@]}"' in _block(setup_text, "TRAIN_CMD=(")


def test_the_solver_clock_is_derived_from_the_node_budget(setup_text):
    """A constant generous at one node budget binds at another. A 3-second
    clock censored 11.3% of solves on the 2026-08-18 shakedown and made which
    positions got a proof depend on machine load."""

    assert "ENDGAME_SOLVER_MAX_SECS=\"$(( (ENDGAME_SOLVER_MAX_NODES / NODE_RATE" in setup_text
    assert "measure_node_rate" in setup_text, "the rate must be measured on the box"


def test_a_missing_cost_model_stops_the_launch(setup_text):
    """The cost model and the card cap select different positions, so falling
    back silently would produce a solver configuration nobody chose."""

    assert "ENDGAME_COST_MODEL=$ENDGAME_COST_MODEL not found" in setup_text


def test_the_scheduler_worker_count_is_flagged_as_unmeasured(setup_text):
    """It is the generation half of the core split, and its right value is a
    measurement. Shipping a placeholder silently would let a guess look like a
    decision."""

    assert "RUST_SCHEDULER_WORKERS" in _block(setup_text, "TRAIN_CMD=(")
    assert "PLACEHOLDER" in setup_text


def test_the_documented_cloud_command_matches_the_launcher(setup_text):
    """`training_parameters.md` reproduces the launch command for review.

    A copy that nothing compares drifts, and then the page people configure runs
    from describes a run nobody launches -- which is how the launcher came to be
    missing sims, search modes and the solver while the plan described all
    three.

    Two flags are exempt: the launcher derives them from the box (the solver's
    clock from the node budget and measured rate, the solver thread count from
    the core split), so the document names the formulas instead of values.
    """

    doc = (REPO_ROOT / "games" / "seven_wonders_duel" / "training_parameters.md").read_text(
        encoding="utf-8"
    )
    block = doc[doc.index("## Recommended Cloud Command") : doc.index("## Overnight")]
    documented = set(re.findall(r"(?<![\w-])--[a-z0-9][a-z0-9-]+", block))

    launched = _long_flags(_block(setup_text, "TRAIN_CMD=("))
    # Flags the launcher builds into an array rather than writing inline.
    launched |= {"--pooled-readout", "--reply-head", "--endgame-solver-max-nodes",
                 "--endgame-cost-model", "--solver-fallback-research"}
    derived_on_the_box = {"--endgame-solver-max-secs", "--solver-threads"}

    missing = sorted(launched - documented - derived_on_the_box)
    assert not missing, (
        f"the launcher passes flags the documented cloud command omits: {missing}. "
        "Update the Recommended Cloud Command section, or the page describes a "
        "run nobody launches."
    )
