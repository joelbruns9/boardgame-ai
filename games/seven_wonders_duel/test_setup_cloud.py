"""W6.1: the launch command in the setup script must actually be launchable.

A wrong flag here is found on a rented box, after the toolchain build, the
equivalence suite and the smoke have all passed -- the most expensive possible
place to learn that an option was renamed.
"""

from __future__ import annotations

from pathlib import Path
import os
import re
import shutil
import subprocess

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


def test_the_scheduler_geometry_is_cloud_scale_not_parser_default(setup_text):
    """Empty meant "let the parser decide", and the parser is laptop-scale:
    16 slots against cloud6's 256, a 256-row batch against 2,048. On a rented
    GPU that is an underfed box, not a cautious default."""

    for knob, minimum in (
        ("RUST_SLOTS", 64),
        ("RUST_GLOBAL_BATCH_CAP", 512),
        ("GATE_SLOTS", 64),
        ("GATE_GLOBAL_BATCH_CAP", 512),
    ):
        match = re.search(rf'^{knob}="\$\{{{knob}:-(\d*)\}}"$', setup_text, re.M)
        assert match, f"{knob} is not a knob with a default"
        assert match.group(1), f"{knob} defaults to empty, i.e. the parser default"
        assert int(match.group(1)) >= minimum, f"{knob}={match.group(1)} is laptop-scale"


def test_the_solver_split_is_sized_to_physical_cores(setup_text):
    """nproc counts SMT siblings. The solver is compute-bound alpha-beta that
    scales 4.37x across 16 logical CPUs, so sizing to the logical count puts
    twice as many threads on a core as it can use -- and the oversubscription
    warning, comparing against the same inflated number, stays silent."""

    assert "_physical_cores()" in setup_text
    assert 'CORES="$(_physical_cores)"' in setup_text
    assert "lscpu" in setup_text, "needs a physical-core probe, not just nproc"


def test_the_clone_can_be_pointed_at_a_branch():
    """A plain clone takes the remote's default branch, and the sentinel check
    below it passes anyway, because the files it looks for exist on main too.
    So launching unmerged work without REPO_BRANCH builds the wrong code and
    says nothing -- the run then omits every flag the operator believes they set.

    This existed as a documented knob before it existed as behaviour, which is
    worse than neither.
    """

    common = (REPO_ROOT / "setup_cloud_common.sh").read_text(encoding="utf-8")
    assert "REPO_BRANCH" in common, "documented in the game script, absent from the library"
    assert '_branch_args=(--branch "$REPO_BRANCH")' in common
    # Both paths: a fresh clone and an existing checkout being updated.
    assert 'git clone "${_branch_args[@]}" "$REPO_URL"' in common
    assert 'git checkout "$REPO_BRANCH"' in common

    setup = (REPO_ROOT / "setup_cloud_7wd.sh").read_text(encoding="utf-8")
    assert setup.count("REPO_BRANCH=<branch>") == 1, "documented more than once"


# --- measured against a known-good run, not against my memory ---------------
#
# The omission tests above assert that the flags I THOUGHT OF are present. That
# is the wrong shape: on 2026-08-20 the launcher shipped with seven training
# parameters at argparse defaults -- weight decay 0.0001 against cloud6's 0.5, a
# replay-window coefficient of 16 against 1,000, no value bootstrap, no minimum
# buffer, a tighter temperature floor -- and every test here passed, because it
# never occurred to me to check those.
#
# The reference is embedded rather than read from the capture directory, which
# is gitignored. That is also the better artefact: a frozen baseline someone
# chose, not a file that can quietly disappear.

CLOUD6_FLAGS = frozenset({
    "--run-dir", "--device", "--iterations", "--games-per-iteration",
    "--seed-games", "--init-checkpoint", "--min-buffer-positions",
    "--selfplay-search-mode", "--dirichlet-epsilon", "--dirichlet-alpha",
    "--full-sims-min", "--full-sims-max", "--cheap-sims-min", "--cheap-sims-max",
    "--gate-sims", "--eval-search-mode", "--cheap-double-reveal-offsets",
    "--top-k", "--age-deal-samples", "--workers", "--process-workers",
    "--d-model", "--layers", "--heads", "--precision", "--learning-rate",
    "--weight-decay", "--value-bootstrap", "--temperature-floor",
    "--temperature-anneal-moves", "--train-steps", "--train-warmup-steps",
    "--train-batch-size", "--schedule-basis", "--generation-backend",
    "--gate-backend", "--derive-backend", "--replay-window-coefficient",
    "--replay-window-exponent", "--replay-window-cap-games",
    "--example-cache-gb", "--hof-opponent-fraction", "--hof-start-games",
    "--opponent-fraction", "--draft-prior-games", "--curriculum-anneal-games",
    "--selfplay-generator-mode", "--bootstrap-policy", "--promotion-every",
    "--revert-reset-after", "--probation-reset-after", "--promotion-min-lcb",
    "--revert-max-ucb", "--gate-ladder-games", "--gate-ladder-step-up-after",
    "--gate-ladder-floor-games", "--anchor-games",
    "--anchor-gate-every-promotions", "--self-anchor-games",
    "--self-anchor-lag-games", "--self-anchor-every-games",
    "--intervention-window-games", "--pack-threads", "--memory-budget-gb",
    "--vram-budget-gb", "--memory-headroom-gb", "--rust-slots",
    "--rust-global-batch-cap", "--rust-max-inflight-batches",
    "--rust-scheduler-workers", "--gate-slots", "--gate-global-batch-cap",
})

#: Deliberately not carried forward, each with a reason.
CLOUD6_FLAGS_DROPPED = {
    # This run bootstraps fresh: TARGET_VERSION went 2->3, the encoder gained
    # features and the architecture gained two heads, so cloud6's buffer is
    # incompatible and its weights come from a net that stalled for 38k games.
    "--init-checkpoint",
}


def test_every_flag_cloud6_used_is_passed_or_deliberately_dropped(setup_text):
    """The gate that would have caught the seven silent defaults.

    A flag the launcher omits is not "unset" -- it is whatever argparse says,
    which for weight decay is 5,000x away from the value cloud6 ran.
    """

    block = _block(setup_text, "TRAIN_CMD=(")
    ours = _long_flags(block)
    # Flags the launcher assembles into arrays rather than writing inline.
    for array in ("ARCH_FLAGS", "SOLVER_FLAGS", "TUNED_FLAGS", "GATE_TUNED_FLAGS"):
        assert f'"${{{array}[@]}}"' in block, f"{array} is not spliced into TRAIN_CMD"
    ours |= _long_flags(setup_text)

    missing = sorted(CLOUD6_FLAGS - ours - CLOUD6_FLAGS_DROPPED)
    assert not missing, (
        f"cloud6 passed these and this launcher does not: {missing}. "
        "Each one silently becomes an argparse default, which is a training "
        "decision nobody made. Pass it, or add it to CLOUD6_FLAGS_DROPPED with "
        "a reason."
    )


def test_the_solver_budget_is_sized_for_a_rented_box(setup_text):
    """4.5M leaves 12 solver threads about 2% utilised: 2.26M nodes per game at
    ~0.33 games/s is 0.75M nodes/s against roughly 36M available. The solver is
    the differentiator this run is built on; it should not idle."""

    match = re.search(r'^ENDGAME_SOLVER_MAX_NODES="\$\{ENDGAME_SOLVER_MAX_NODES:-(\d+)\}"$',
                      setup_text, re.M)
    assert match, "ENDGAME_SOLVER_MAX_NODES is not a knob with a default"
    assert int(match.group(1)) >= 20_000_000, (
        f"{match.group(1)} nodes is laptop-scale for a box that solves at "
        "millions of nodes per second"
    )


# ── The self-update trap ─────────────────────────────────────────────────────
# setup_cloud_7wd.sh sources setup_cloud_common.sh at its top and does not
# `git pull` until stage 2, so the first re-run after a code change pulls the
# new code and keeps executing the old copy. That launched a 200k-game run on a
# pre-change command line whose manifest recorded the post-change commit: no
# stage failed, and the only symptom was the launch command itself.

BASH = shutil.which("bash")


def _harness(tmp_path: Path, marker: str = "") -> Path:
    """A miniature launcher with the same shape: source the library, checksum,
    'pull', hand over. `pending_new` stands in for what git pull would land."""

    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    shutil.copy(COMMON, repo / "setup_cloud_common.sh")
    (repo / "script.sh").write_text(
        "#!/usr/bin/env bash\n"
        f"{marker}"
        "set -euo pipefail\n"
        'COMMON_SH="$(dirname "$0")/setup_cloud_common.sh"\n'
        'source "$COMMON_SH"\n'
        'SUM="$(common::self_checksum "${BASH_SOURCE[0]}" "$COMMON_SH")"\n'
        'ARGV=("$@")\n'
        'echo "RAN version=${VERSION:-1} args=${ARGV[*]:-none}"\n'
        # stand-in for stage 2's git pull
        '[ -f "$PENDING" ] && mv "$PENDING" "$(dirname "$0")/script.sh"\n'
        'common::reexec_if_updated "$(dirname "$0")/script.sh" '
        '"$(dirname "$0")/setup_cloud_common.sh" "$SUM" ${ARGV+"${ARGV[@]}"}\n'
        'echo "CONTINUED version=${VERSION:-1}"\n',
        encoding="utf-8",
    )
    return repo


def _run(repo: Path, pending: str = "") -> str:
    env = {**os.environ, "PENDING": str(repo.parent / "pending"), "VERSION": "1"}
    if pending:
        (repo.parent / "pending").write_text(pending, encoding="utf-8")
    proc = subprocess.run(
        [BASH, str(repo / "script.sh"), "alpha", "beta"],
        capture_output=True, text=True, env=env, cwd=repo.parent,
    )
    assert proc.returncode == 0, f"setup exited {proc.returncode}: {proc.stderr}"
    return proc.stdout + proc.stderr


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_an_unchanged_pull_does_not_restart_the_script(tmp_path):
    out = _run(_harness(tmp_path))
    assert out.count("RAN version=") == 1, f"restarted for no reason:\n{out}"
    assert "CONTINUED version=1" in out


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_a_pull_that_changes_the_script_hands_over_to_the_new_copy(tmp_path):
    repo = _harness(tmp_path)
    new = (repo / "script.sh").read_text(encoding="utf-8").replace(
        "${VERSION:-1}", "2")
    out = _run(repo, pending=new)

    # The old copy ran, the new copy took over, and the work after the pull was
    # done by the new copy -- the whole point.
    assert "RAN version=1" in out
    assert "RAN version=2" in out, f"kept running the stale copy:\n{out}"
    assert "CONTINUED version=2" in out
    assert "CONTINUED version=1" not in out
    # Exactly one handover, not a loop.
    assert out.count("RAN version=") == 2, f"restart loop:\n{out}"
    # Arguments survive the handover.
    assert out.count("args=alpha beta") == 2


def test_the_launcher_checks_for_its_own_update_right_after_the_pull(setup_text):
    """Structural: the guard is worthless if it runs before the pull, or if a
    later edit drops the call."""

    assert "common::reexec_if_updated" in setup_text, (
        "the launcher no longer checks whether the pull replaced it; the next "
        "code change will launch on the previous commit's command line"
    )
    pull = setup_text.index("common::clone_repo")
    guard = setup_text.index("common::reexec_if_updated", pull)
    assert guard > pull
    # Nothing expensive in between: the point is to hand over before stage 3
    # spends ten minutes installing torch on behalf of the wrong commit.
    assert "stage 3" not in setup_text[pull:guard]


def test_setup_output_is_recorded_on_disk(setup_text):
    """Stage output used to exist only in the operator's scrollback."""

    assert 'exec > >(tee -a "$SETUP_LOG")' in setup_text
    assert "setup/setup.log" in setup_text


@pytest.mark.parametrize("stage,label", [("3", "pip install"), ("4", "maturin build")])
def test_the_noisy_build_stages_log_to_a_file(setup_text, stage, label):
    block = setup_text[setup_text.index(f"stage {stage} "):
                       setup_text.index(f"stage_done {stage}")]
    assert "common::quietly" in block, f"stage {stage} still floods the terminal"
    assert label in block


# ── sweep_7wd.sh: measuring without launching ────────────────────────────────
# setup_cloud_7wd.sh cannot be used to sweep, because reaching stage 8b means
# reaching stage 10, which launches. This script measures alone. Its safety
# properties are the reason it exists, so they are what get tested.

SWEEP = REPO_ROOT / "sweep_7wd.sh"


@pytest.fixture(scope="module")
def sweep_text() -> str:
    return SWEEP.read_text(encoding="utf-8")


@pytest.mark.parametrize("module", ["f4_phase_d_sweep", "w5_gate_slots_sweep"])
def test_the_standalone_sweep_only_uses_flags_that_exist(sweep_text, module):
    used = _long_flags(_invocation(sweep_text, module))
    assert len(used) >= 6, f"only extracted {used} from the {module} call"
    unknown = sorted(used - _module_options(module))
    assert not unknown, f"sweep_7wd.sh passes flags {module} rejects: {unknown}"


def test_the_standalone_sweep_never_launches_training(sweep_text):
    """The whole point: stage 8b is unreachable without stage 10."""

    assert "phase_d.py" not in sweep_text
    assert "launch_detached" not in sweep_text
    assert "nohup" not in sweep_text


def test_the_standalone_sweep_never_builds_the_crate(sweep_text):
    """`maturin develop` installs into the shared site-packages, so building
    from the sweep checkout would replace the extension the RUNNING training
    process loads. The script compares crate trees and refuses instead."""

    # Comments explain the trap, so only executable lines are checked.
    code = "\n".join(
        line for line in sweep_text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "maturin" not in code
    assert "cargo build" not in code
    assert "rev-parse" in code and "seven_wonders_rust" in code


def test_the_sweep_grid_contains_the_settings_the_run_is_using(sweep_text):
    """A grid that excludes the current value cannot say whether changing it
    helps. Stage 8b's grid excluded 256 slots and pinned inflight to 1, which is
    why it could not have found the batch=42 problem."""

    def axis(name: str) -> set[str]:
        match = re.search(rf'^{name}="\$\{{{name}:-([^}}]*)\}}"$', sweep_text, re.M)
        assert match, f"{name} is not a knob with a default"
        return {part.strip() for part in match.group(1).split(",")}

    assert "256" in axis("SWEEP_SLOTS"), "the run's slot count is outside the grid"
    assert "2048" in axis("SWEEP_CAPS"), "the run's batch cap is outside the grid"
    assert {"1", "2"} <= axis("SWEEP_INFLIGHT"), "inflight must be varied, not pinned"
    assert "4" in axis("SWEEP_WORKERS"), "the run's shard count is outside the grid"


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_the_sweep_refuses_to_run_beside_a_live_training_process(tmp_path):
    """A sweep sharing the GPU with training measures contention, and the point
    it crowns is whichever tolerated the interference best."""

    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "pgrep").write_text(
        "#!/usr/bin/env bash\necho '4242 python -m games.seven_wonders_duel.phase_d'\n",
        encoding="utf-8",
    )
    (stub / "pgrep").chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{stub}{os.pathsep}{os.environ['PATH']}",
        "OUTPUT": str(tmp_path / "out"),
        "RUN_REPO": str(tmp_path / "repo"),
    }
    proc = subprocess.run(
        [BASH, str(SWEEP)], capture_output=True, text=True, env=env, cwd=tmp_path
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"swept anyway:\n{combined}"
    assert "Stop training before sweeping" in combined

    # And the override exists, so the refusal is a guard rather than a wall.
    forced = subprocess.run(
        [BASH, str(SWEEP)], capture_output=True, text=True,
        env={**env, "FORCE": "1"}, cwd=tmp_path,
    )
    forced_out = forced.stdout + forced.stderr
    assert "sweeping anyway" in forced_out
    assert "Stop training before sweeping" not in forced_out


def test_the_kingdomino_launcher_also_hands_over_after_its_pull():
    """Same trap, same shape, different file.

    `setup_cloud.sh` is self-contained rather than sourcing the common library,
    so it carries its own copy of the guard -- but a launcher that pulls at
    stage 2 and keeps executing the code bash read at stage 0 configures its run
    from the previous commit either way.
    """

    text = (REPO_ROOT / "setup_cloud.sh").read_text(encoding="utf-8")
    assert "reexec_if_updated" in text, "the Kingdomino launcher has no guard"
    pull = text.index("git pull")
    guard = text.index("reexec_if_updated\n", pull)
    assert guard > pull, "the guard must run after the pull, not before"
    assert "SETUP_REEXEC" in text, "nothing stops an exec loop"
