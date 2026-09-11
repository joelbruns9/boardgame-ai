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
    """The raw shell text of the call to one sweep harness.

    Skips occurrences that are not the real invocation -- sweep_7wd.sh also runs
    `--help` as a capability probe, and matching that instead swept up the next
    command's flags and reported them as rejected by this module.
    """

    needle = f'"$PY" -m games.seven_wonders_duel.{module}'
    start = -1
    while True:
        start = text.find(needle, start + 1)
        if start == -1:
            raise AssertionError(f"no invocation of {module} found")
        block = text[start : text.index("|| die", start)]
        if "--checkpoint" in block:
            return block


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
    "module", ["f4_staged_sweep", "w5_gate_slots_sweep"]
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

    for module in ("f4_staged_sweep", "w5_gate_slots_sweep"):
        block = _invocation(setup_text, module)
        assert '--precision "$PRECISION"' in block, f"{module} sweeps at a fixed precision"


def test_the_generation_sweep_passes_comma_separated_axes(setup_text):
    """f4_phase_d_sweep takes one string per axis, not a list.

    `--slots 48 96 144` parses as --slots=48 plus three stray positionals, and
    argparse rejects the whole command. That is the original bug.
    """

    block = _invocation(setup_text, "f4_staged_sweep")
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
    for array in ("ARCH_FLAGS", "SOLVER_FLAGS"):
        assert f'"${{{array}[@]}}"' in block, f"{array} is not spliced into TRAIN_CMD"
    # The two MEASURED arrays are appended at stage 10 instead of interpolated
    # here, because the command is now assembled before the sweep that fills
    # them -- that is what lets `--emit-config` describe the run to the sweep.
    # They still have to reach the command line, and "spliced somewhere" is the
    # property that matters; "spliced in this literal" was only ever a proxy.
    assert (
        'TRAIN_CMD+=("${TUNED_FLAGS[@]}" "${GATE_TUNED_FLAGS[@]}")' in setup_text
    ), "the measured scheduler flags never reach TRAIN_CMD"
    assembly = setup_text.index("TRAIN_CMD=(")
    append = setup_text.index('TRAIN_CMD+=("${TUNED_FLAGS[@]}"')
    sweep = setup_text.index('stage 8b "Scheduler sweeps')
    launch = setup_text.index("common::launch_detached")
    assert assembly < sweep < append < launch, (
        "the command must be assembled BEFORE the sweep that measures it, and "
        "the measured flags appended AFTER it and before the launch"
    )
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


def test_the_standalone_sweep_never_installs_into_the_shared_environment(sweep_text):
    """`maturin develop` installs into the shared site-packages, replacing the
    .so a training process has mapped.

    This originally REFUSED when the crate differed between checkouts. That
    blocked the case it was never meant to -- sweeping a configuration whose
    entire point is new Rust -- so it now builds a wheel into its own directory,
    which removes the conflict instead of arbitrating it.
    """

    # Comments explain the trap, so only executable lines are checked.
    code = "\n".join(
        line for line in sweep_text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "maturin develop" not in code, "must build a wheel, never install in place"
    assert "maturin build" in code
    assert "--target" in code, "the wheel must go to an isolated directory"
    assert "site-packages" in sweep_text, "EXT_DIR inside site-packages must be refused"
    # Per invocation, not per file: a bare `in code` check passed while the
    # generation sweep had lost its PYTHONPATH, because the gate sweep and the
    # isolation probe still carried one. Both harnesses have to use it or half
    # the measurement runs the shared engine.
    for module in ("f4_phase_d_sweep", "w5_gate_slots_sweep"):
        invocations = [
            line
            for line in code.splitlines()
            # The `--help` capability probe reads the harness's flags, not the
            # engine, so it needs no extension. Every line that MEASURES does.
            if f"-m games.seven_wonders_duel.{module}" in line and "--help" not in line
        ]
        assert invocations, f"no invocation of {module} found"
        for line in invocations:
            assert 'PYTHONPATH="$EXT_DIR"' in line, (
                f"{module} is not run against the isolated extension: {line.strip()}"
            )


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


def _fake_run_repo(tmp_path, harness: bool = True):
    """A real git checkout: stage 1 requires one before the stop check runs.

    Without it the refusal test would pass for the WRONG reason -- dying at
    "not a git checkout" rather than at the live-training refusal.
    """

    run_repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(run_repo)], check=True)
    (run_repo / "seed.txt").write_text("x", encoding="utf-8")
    if harness:
        # A stub that advertises the flags stage 1 probes for, so tests of LATER
        # stages are not stopped by the capability check.
        pkg = run_repo / "games" / "seven_wonders_duel"
        pkg.mkdir(parents=True, exist_ok=True)
        for parent in (run_repo / "games", pkg):
            (parent / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "f4_phase_d_sweep.py").write_text(
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            "for flag in ('--workers', '--solver-threads-total',\n"
            "             '--config-from-manifest', '--sims-divisor',\n"
            "             '--checkpoint'):\n"
            "    p.add_argument(flag)\n"
            "p.parse_args()\n",
            encoding="utf-8",
        )
    subprocess.run(["git", "-C", str(run_repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(run_repo), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "seed"], check=True,
    )
    return run_repo


def _pgrep_stub(tmp_path):
    stub = tmp_path / "bin"
    stub.mkdir(exist_ok=True)
    (stub / "pgrep").write_text(
        "#!/usr/bin/env bash\necho '4242 python -m games.seven_wonders_duel.phase_d'\n",
        encoding="utf-8",
    )
    (stub / "pgrep").chmod(0o755)
    return stub


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_the_sweep_refuses_to_run_beside_a_live_training_process(tmp_path):
    """A sweep sharing the GPU with training measures contention, and the point
    it crowns is whichever tolerated the interference best."""

    stub = _pgrep_stub(tmp_path)
    run_repo = _fake_run_repo(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{stub}{os.pathsep}{os.environ['PATH']}",
        "OUTPUT": str(tmp_path / "out"),
        "RUN_REPO": str(run_repo),
        "SWEEP_REPO": str(tmp_path / "sweep"),
        "REPO_URL": str(run_repo),
        # SWEEP_REF defaults to main; `git init` here makes master.
        "SWEEP_REF": _default_branch(run_repo),
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


def test_the_live_profile_runs_before_the_stop_check(sweep_text):
    """The read-only diagnosis must be reachable WITHOUT stopping training.

    Ordered the other way, the one step that costs nothing and answers the
    solver question would be gated behind killing the run it is meant to
    diagnose.
    """

    profile = sweep_text.index("generation_profile")
    stop_check = sweep_text.index('stage 3 "Training must not be running"')
    assert profile < stop_check
    assert "PROFILE_ONLY" in sweep_text


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_profile_only_does_not_stop_at_the_live_training_check(tmp_path):
    """PROFILE_ONLY must exit cleanly even with Phase D running -- that is the
    entire point of it."""

    stub = _pgrep_stub(tmp_path)
    # A real checkout for stage 1, so the run reaches stage 2 rather than dying
    # earlier for an unrelated reason and passing vacuously.
    run_repo = _fake_run_repo(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{stub}{os.pathsep}{os.environ['PATH']}",
        "OUTPUT": str(tmp_path / "out"),
        "RUN_REPO": str(run_repo),
        "SWEEP_REPO": str(tmp_path / "sweep"),
        "REPO_URL": str(run_repo),  # no network in tests
        "SWEEP_REF": _default_branch(run_repo),
        "PROFILE_ONLY": "1",
    }
    proc = subprocess.run(
        [BASH, str(SWEEP)], capture_output=True, text=True, env=env, cwd=tmp_path
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"PROFILE_ONLY refused to run:\n{combined}"
    assert "Stop training before sweeping" not in combined
    assert "PROFILE_ONLY=1" in combined
    # It stopped before anything that touches the GPU.
    assert "Generation sweep" not in combined


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_a_second_run_picks_up_code_pushed_since_the_first(tmp_path):
    """`git fetch` + `git checkout main` is NOT `git pull`.

    Fetch advances origin/main; the local `main` a clone left behind stays where
    it was. Checking out the local branch therefore re-runs the code the clone
    was made at -- which on the box meant the second invocation silently
    re-measured the first invocation's code and printed the stale commit as if
    it were current.
    """

    origin = _fake_run_repo(tmp_path)
    sweep_repo = tmp_path / "sweep"
    env = {
        **os.environ,
        "PATH": f"{_pgrep_stub(tmp_path)}{os.pathsep}{os.environ['PATH']}",
        "OUTPUT": str(tmp_path / "out"),
        "RUN_REPO": str(origin),
        "SWEEP_REPO": str(sweep_repo),
        "REPO_URL": str(origin),
        "SWEEP_REF": "master" if _default_branch(origin) == "master" else "main",
        "PROFILE_ONLY": "1",
    }
    first = subprocess.run(
        [BASH, str(SWEEP)], capture_output=True, text=True, env=env, cwd=tmp_path
    )
    assert first.returncode == 0, first.stdout + first.stderr

    # Something lands on the remote after that clone exists.
    (origin / "new.txt").write_text("after", encoding="utf-8")
    subprocess.run(["git", "-C", str(origin), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(origin), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "later"], check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(origin), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    second = subprocess.run(
        [BASH, str(SWEEP)], capture_output=True, text=True, env=env, cwd=tmp_path
    )
    combined = second.stdout + second.stderr
    assert second.returncode == 0, combined
    assert head[:12] in combined, (
        f"second run did not reach {head[:12]}; it re-used the stale clone:\n"
        f"{combined}"
    )
    # And the file from that commit is really on disk for the sweep to run.
    assert (sweep_repo / "new.txt").is_file()


def _default_branch(repo) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def test_the_sweep_holds_solver_threads_constant_across_the_worker_axis(sweep_text):
    """`--solver-threads` is PER SHARD, so a fixed per-shard count across a
    worker sweep varies total solver load with the axis being measured: 1 shard
    would solve on 3 threads and 4 shards on 12, and fewer shards would lose
    partly because they were under-solving."""

    block = _invocation(sweep_text, "f4_phase_d_sweep")
    assert "--solver-threads-total" in block, (
        "the worker sweep confounds solver load with the worker count"
    )


def test_the_sweep_measures_the_search_the_run_actually_runs(sweep_text):
    """PhaseDConfig defaults are gumbel at 24/128 sims. This run is PUCT at
    100/1600 -- roughly 50 simulations a move versus a measured 522, under a
    different algorithm. Simulations per move set the leaf arrival rate, which
    is exactly what the slot and worker axes act on, so a sweep on defaults
    optimises a machine nobody is running."""

    block = _invocation(sweep_text, "f4_phase_d_sweep")
    assert "--config-from-manifest" in block
    assert "run_manifest.json" in block


def test_the_sweep_checkout_defaults_to_main_not_the_runs_commit(sweep_text):
    """This checkout exists to run tooling NEWER than a run that must not be
    updated. Defaulting to the run's commit pinned the tooling to be permanently
    as old as the run, and silently reverted a checkout the operator had just
    moved to main."""

    assert 'SWEEP_REF="${SWEEP_REF:-main}"' in sweep_text


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_a_stale_harness_fails_at_stage_1_with_the_fix(tmp_path):
    """The old harness has no --workers, and argparse's 'unrecognized arguments'
    names the flag without saying the checkout is the cause. Discovering it at
    stage 6 also wastes a GPU gate and a 61MB checkpoint copy first."""

    run_repo = _fake_run_repo(tmp_path, harness=False)
    # A checkout whose harness predates the axes: the module exists but its
    # --help advertises none of the required flags.
    pkg = run_repo / "games" / "seven_wonders_duel"
    pkg.mkdir(parents=True, exist_ok=True)
    for parent in (run_repo / "games", pkg):
        (parent / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "f4_phase_d_sweep.py").write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--slots')\n"
        "p.parse_args()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(run_repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(run_repo), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "old harness"], check=True,
    )

    proc = subprocess.run(
        [BASH, str(SWEEP)],
        capture_output=True, text=True, cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{_pgrep_stub(tmp_path)}{os.pathsep}{os.environ['PATH']}",
            "OUTPUT": str(tmp_path / "out"),
            "RUN_REPO": str(run_repo),
            "SWEEP_REPO": str(tmp_path / "sweep"),
            "REPO_URL": str(run_repo),
            "SWEEP_REF": _default_branch(run_repo),
        },
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"stale harness was accepted:\n{combined}"
    assert "does not support" in combined
    assert "--workers" in combined
    assert "SWEEP_REF=main" in combined, "the error must name the fix"
    # And it stopped before spending time on setup.
    assert "Checkpoint" not in combined
    assert "Generation sweep" not in combined


# ── Cheap-path leaf batching ─────────────────────────────────────────────────
# Cheap and full moves differ in what batching costs: cheap moves emit no policy
# target, a full move's visit distribution IS the target. The knobs are separate
# for that reason, and each refusal below exists to stop a config that would
# either corrupt targets or silently buy nothing.


def _config(**overrides):
    from .phase_d import PhaseDConfig

    base = dict(
        run_dir="x",
        selfplay_search_mode="puct",
        cheap_search_mode="gumbel",
        conflict_free_waves=True,
        round_robin_candidates=True,
    )
    base.update(overrides)
    config = PhaseDConfig(**base)
    # `validate()` is a separate method, not __post_init__: constructing the
    # dataclass checks nothing, so a test that only constructs asserts nothing.
    config.validate()
    return config


def test_cheap_leaf_batching_is_refused_under_a_puct_cheap_root():
    """Same reason --leaf-batch is: the root would select under virtual loss,
    and a PUCT root's visit distribution is the policy target."""

    with pytest.raises(ValueError, match="Gumbel cheap root"):
        _config(cheap_search_mode="puct", cheap_leaf_batch=8)
    # `same` inherits selfplay's PUCT, so it must be refused too -- this is the
    # case a check on the literal string would miss.
    with pytest.raises(ValueError, match="Gumbel cheap root"):
        _config(cheap_search_mode="same", cheap_leaf_batch=8)


def test_cheap_leaf_batching_is_refused_when_it_would_be_inert():
    """Without round-robin, the conflict-free rule cuts every wave to width 1.
    Accepting the flag would report a configuration change that buys nothing."""

    with pytest.raises(ValueError, match="inert"):
        _config(cheap_leaf_batch=8, round_robin_candidates=False)
    with pytest.raises(ValueError, match="inert"):
        _config(cheap_leaf_batch=8, conflict_free_waves=False)


def test_the_defaults_change_nothing():
    """Everything ships off, so the equivalence gate stays comparable."""

    config = _config(cheap_leaf_batch=0, conflict_free_waves=False,
                     round_robin_candidates=False)
    assert config.cheap_leaf_batch == 0
    assert config.conflict_free_waves is False
    assert config.round_robin_candidates is False
    # And a full-path PUCT run still refuses --leaf-batch, as before.
    with pytest.raises(ValueError, match="requires --leaf-batch 1"):
        _config(leaf_batch=8)


def test_the_new_flags_exist_on_the_parser():
    from .phase_d import build_parser

    options = _parser_options(build_parser())
    for flag in ("--cheap-leaf-batch", "--conflict-free-waves",
                 "--round-robin-candidates"):
        assert flag in options


def test_virtual_loss_root_is_what_unlocks_a_batched_puct_root():
    """Opt-in, never implied by --leaf-batch: batching a PUCT root is a
    different algorithm there, and on full moves the root's visit distribution
    is the policy target."""

    with pytest.raises(ValueError, match="requires --leaf-batch 1"):
        _config(leaf_batch=8, virtual_loss_root=False)
    # With the opt-in it is permitted -- that is the whole point of the flag.
    assert _config(leaf_batch=8, virtual_loss_root=True).leaf_batch == 8
    # And it unlocks the cheap override under a PUCT cheap root too.
    assert _config(
        cheap_search_mode="puct", cheap_leaf_batch=8, virtual_loss_root=True
    ).cheap_leaf_batch == 8


# ── leaf_batch_test.sh ───────────────────────────────────────────────────────
# The A/B may run beside training because a win rate is not a timing: both arms
# play the same game on the same GPU, so contention slows without biasing. What
# must NOT happen is replacing the extension the live training process loaded.

LEAF_BATCH_SH = REPO_ROOT / "leaf_batch_test.sh"


@pytest.fixture(scope="module")
def leaf_batch_text() -> str:
    return LEAF_BATCH_SH.read_text(encoding="utf-8")


def test_the_leaf_batch_test_never_installs_into_the_shared_environment(leaf_batch_text):
    """`maturin develop` installs into site-packages, replacing the .so the
    training process has mapped -- and Phase D's resume guard hashes the REPO,
    not the extension, so the resulting engine drift would be silent."""

    code = "\n".join(
        line for line in leaf_batch_text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "maturin develop" not in code, "must build a wheel, never install in place"
    assert "maturin build" in code
    assert "--target" in code, "the wheel must go to an isolated directory"


def test_it_refuses_an_extension_directory_inside_site_packages(leaf_batch_text):
    assert "site-packages" in leaf_batch_text
    assert "dist-packages" in leaf_batch_text


def test_it_proves_the_isolation_rather_than_assuming_it(leaf_batch_text):
    """Both halves are checked at runtime: the A/B resolves the isolated copy,
    and the shared copy is still where it was."""

    assert "isolated extension resolves first" in leaf_batch_text
    assert "shared extension MOVED" in leaf_batch_text


def test_it_only_uses_flags_the_ab_harness_accepts(leaf_batch_text):
    used = _long_flags(_invocation(leaf_batch_text, "leaf_batch_ab"))
    assert len(used) >= 5, f"only extracted {used}"
    unknown = sorted(used - _module_options("leaf_batch_ab"))
    assert not unknown, f"leaf_batch_test.sh passes flags the harness rejects: {unknown}"


def test_the_leaf_batch_test_puts_cargo_on_path_itself(leaf_batch_text):
    """rustup installs cargo to ~/.cargo/bin and adds it to the shell PROFILE,
    which a non-login shell never reads -- so a box that BUILT this crate at
    setup still has no cargo in a fresh ssh session. maturin reports that as
    "Cargo metadata failed", naming the symptom rather than the cause. The box
    hit exactly this.

    Structural rather than behavioural: simulating a PATH without cargo needs
    symlinks this test suite cannot create on Windows.
    """

    assert '[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"' in leaf_batch_text
    build = leaf_batch_text.index("maturin build")
    assert leaf_batch_text.index(".cargo/env") < build, "must be sourced BEFORE the build"
    # And it must not paper over a still-missing cargo.
    assert "cargo is not on PATH" in leaf_batch_text


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_a_stale_copy_hands_over_to_the_checkouts_copy(tmp_path):
    """raw.githubusercontent is CDN-cached: a curl seconds after a push returns
    the previous file. That happened on the box -- the checkout advanced to the
    new commit while the script driving it was the old one, and only the version
    line made it visible. Third appearance of one trap."""

    origin = _fake_run_repo(tmp_path, harness=False)
    fresh = (REPO_ROOT / "leaf_batch_test.sh").read_text(encoding="utf-8")
    (origin / "leaf_batch_test.sh").write_text(fresh, encoding="utf-8")
    (origin / "games" / "seven_wonders_duel").mkdir(parents=True, exist_ok=True)
    (origin / "games" / "seven_wonders_duel" / "leaf_batch_ab.py").write_text(
        "", encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(origin), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(origin), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "fresh"], check=True,
    )

    # A "CDN-cached" copy: same script, one marker line older.
    stale = tmp_path / "stale.sh"
    stale.write_text(
        fresh.replace("LEAF_BATCH_SCRIPT_VERSION=3", "LEAF_BATCH_SCRIPT_VERSION=1"),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [BASH, str(stale)],
        capture_output=True, text=True, cwd=tmp_path,
        env={
            **os.environ,
            "OUTPUT": str(tmp_path / "out"),
            "RUN_REPO": str(origin),
            "SWEEP_REPO": str(tmp_path / "sweep"),
            "REPO_URL": str(origin),
            "SWEEP_REF": _default_branch(origin),
            "EXT_DIR": str(tmp_path / "ext"),
        },
    )
    combined = proc.stdout + proc.stderr
    assert "version 1" in combined, f"the stale copy did not start:\n{combined}"
    assert "version 3" in combined, f"no handover to the fresh copy:\n{combined}"
    assert "Handing over" in combined
    # Exactly one handover, not a loop.
    assert combined.count("version 3") == 1, f"restart loop:\n{combined}"


def test_the_sweep_can_measure_a_configuration_no_run_has_used(sweep_text):
    """The sweep reads its search settings from a manifest, which is correct
    once a run exists and useless before one does. Choosing geometry for
    leaf batching means sweeping settings no manifest yet contains -- otherwise
    it optimises for leaf_batch=1, a value nobody intends to run."""

    assert "CONFIG_OVERRIDES" in sweep_text
    block = _invocation(sweep_text, "f4_phase_d_sweep")
    assert "CONFIG_OVERRIDES" in block, "the overrides must reach the harness"


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_a_quieted_stage_that_fails_stops_the_caller():
    """`$?` after a completed `if...fi` is the status of the IF STATEMENT, which
    is 0 when the condition failed and there is no else. common::quietly read it
    there, so it announced "FAILED (exit 0)" and returned success -- and the
    `|| die` at every call site never fired.

    The equivalence suite and the plumbing smoke were both unable to stop a
    launch. On the box, the smoke failed and training was launched anyway.
    """

    script = (
        f'source "{COMMON.as_posix()}"\n'
        'common::quietly /tmp/qt_fail.log "boom" -- bash -c "exit 3" '
        '&& echo CALLER_CONTINUED || echo "CALLER_STOPPED $?"\n'
        'common::quietly /tmp/qt_ok.log "fine" -- bash -c "exit 0" '
        '&& echo OK_CONTINUED || echo OK_STOPPED\n'
    )
    proc = subprocess.run([BASH, "-c", script], capture_output=True, text=True)
    combined = proc.stdout + proc.stderr
    assert "CALLER_STOPPED 3" in combined, f"a failing stage returned success:\n{combined}"
    assert "CALLER_CONTINUED" not in combined
    assert "OK_CONTINUED" in combined, "a succeeding stage must not stop the caller"


def test_resuming_across_a_code_change_is_opt_in_and_warned(setup_text):
    """The guard exists because a pull landing mid-run splits the run across two
    engines with nothing recording it. An override is legitimate -- the operator
    may know the change is inert -- but it must be stated, not defaulted."""

    assert 'ALLOW_RESUME_CODE_DRIFT="${ALLOW_RESUME_CODE_DRIFT:-0}"' in setup_text, (
        "the override must default to OFF"
    )
    assert "--allow-resume-code-drift" in setup_text
    # And it must say what it costs, at launch, not only in a comment.
    launch = setup_text[setup_text.index('if [ "$ALLOW_RESUME_CODE_DRIFT" = "1" ]; then') :]
    launch = launch[: launch.index("common::launch_detached")]
    assert "different engines" in launch
    assert "warn " in launch


@pytest.mark.parametrize(
    "wave_flags,leaf_batch,virtual_loss", [(1, 6, 1), (0, 1, 0), (1, 1, 0), (0, 6, 1)]
)
def test_every_combination_the_launcher_can_produce_validates(
    wave_flags, leaf_batch, virtual_loss
):
    """A launcher knob combination that Phase D refuses is a defect in the
    launcher, not in the operator.

    CHEAP_LEAF_BATCH was pinned at 16 while the wave flags were switchable, so
    WAVE_FLAGS=0 produced `cheap_leaf_batch=16` with no waves -- which Phase D
    correctly refuses as inert. The run died at stage 10, five stages after the
    decision, twice.
    """

    from .phase_d import PhaseDConfig

    cheap = 16 if wave_flags else 0
    config = PhaseDConfig(
        run_dir="x", selfplay_search_mode="puct", cheap_search_mode="gumbel",
        leaf_batch=leaf_batch, virtual_loss_root=bool(virtual_loss),
        cheap_leaf_batch=cheap,
        cheap_conflict_free_waves=bool(wave_flags),
        cheap_round_robin_candidates=bool(wave_flags),
        eval_leaf_batch=16 if virtual_loss else 1,
    )
    config.validate()


def test_the_cheap_leaf_batch_default_follows_the_wave_setting(setup_text):
    assert 'CHEAP_LEAF_BATCH="${CHEAP_LEAF_BATCH:-16}"' in setup_text
    assert 'CHEAP_LEAF_BATCH="${CHEAP_LEAF_BATCH:-0}"' in setup_text
    waves = setup_text.index('WAVE_FLAGS="${WAVE_FLAGS:-1}"')
    cheap = setup_text.index('CHEAP_LEAF_BATCH="${CHEAP_LEAF_BATCH:-16}"')
    assert waves < cheap, "the wave setting must be known before the default"


def test_the_launcher_validates_its_command_before_detaching(setup_text):
    """Three relaunches died at stage 10 on knob pairings decided in the
    launcher and rejected by Phase D -- each after the toolchain, preflight,
    equivalence suite and smoke had run. Checking costs two seconds."""

    assert "--validate-config" in setup_text
    validate = setup_text.index("--validate-config")
    launch = setup_text.index("common::launch_detached")
    assert validate < launch, "validation must run BEFORE the process is detached"
    # It must reconstruct the arguments, not the interpreter and module: index
    # 0/1/2 are $PY, -m and the module name, so the slice starts at 3.
    assert '"${TRAIN_CMD[@]:3}"' in setup_text
    # And it must stop the launch, not warn and continue.
    block = setup_text[validate : validate + 700]
    assert "die " in block, "a rejected config must not proceed to launch"


def test_validate_config_exists_and_does_not_train():
    from .phase_d import build_parser

    assert "--validate-config" in _parser_options(build_parser())
    source = (REPO_ROOT / "games/seven_wonders_duel/phase_d.py").read_text(
        encoding="utf-8"
    )
    block = source[source.index("if args.validate_config:") :][:600]
    assert "return 0" in block
    assert "loop.run()" not in block, "validation must not start training"


# ---------------------------------------------------------------------------
# The two-pass contract: the run carries what you chose, on numbers this box
# measured
# ---------------------------------------------------------------------------

RUN_FILE = REPO_ROOT / "launch_7wd_run.sh"


def test_every_workstream_is_reachable_from_the_launcher(setup_text):
    """Until W7 these all defaulted off in `phase_d` and NONE was reachable
    here, so a run meant to carry the new architecture would have carried none
    of it while the manifest recorded a commit containing all of it.
    """

    for flag in (
        "--slot-embedding",        # W1
        "--graph-module",          # W2
        "--hierarchical-value",    # W4
        "--hier-value-weight",
        "--action-residual",       # W5
        "--action-exposes",
        "--action-policy-weight",
        "--specialists",           # W7
        "--specialist-floor-every",
        "--specialist-reanalysis",
    ):
        assert flag in setup_text, f"{flag} is not reachable from the launcher"


def test_w3_control_is_pinned_rather_than_inherited(setup_text):
    """W3 rides on an encoder env var that happens to default on. A run should
    record a decision, not inherit a default that could change."""

    assert "SWD_CONTROL_FEATURES" in setup_text
    assert "export SWD_CONTROL_FEATURES" in setup_text


def test_specialists_without_the_outlook_head_is_refused_by_the_launcher(
    setup_text,
):
    """The bias reads W4's outlook and a leaf without one is a hard error.
    Phase D refuses the combination too, but failing here names the launcher
    knob rather than the flag."""

    assert 'if [ "$HIERARCHICAL_VALUE" != "1" ]; then' in setup_text
    index = setup_text.index('if [ "$HIERARCHICAL_VALUE" != "1" ]; then')
    assert "die" in setup_text[index : index + 400]


def test_the_reanalysis_settings_are_pinned_rather_than_inherited(setup_text):
    """A default nobody chose is the `--train-steps` mistake again.

    Both numbers were measured on a LAPTOP 3070, and `--reanalysis-slots 256`
    is a memory choice as much as a speed one. Leaving either to the parser
    means a rented box silently runs whatever the default was on the day, and
    the launch log records nothing about it.

    They ride inside the `SPECIALIST_REANALYSIS` branch on purpose: Phase D
    reads them only when reanalysis is on, so passing them otherwise would put
    a knob on the line that changes nothing and invite someone to conclude
    reanalysis was running when it was not.
    """

    block = _block(
        setup_text, 'if [ "$SPECIALIST_REANALYSIS" = "1" ]; then', "  fi"
    )
    for flag in (
        "--specialist-reanalysis",
        "--reanalysis-backend",
        "--reanalysis-slots",
    ):
        assert flag in block, f"{flag} left to the parser default"


# --------------------------------------------------------------------------
# setup_dryrun.sh -- the harness that executes the launcher against stubs
# --------------------------------------------------------------------------

DRYRUN = REPO_ROOT / "setup_dryrun.sh"


@pytest.fixture(scope="module")
def dryrun_text() -> str:
    return DRYRUN.read_text(encoding="utf-8")


def test_the_dry_run_releases_the_guard_it_deliberately_trips(dryrun_text):
    """The harness sets SKIP_SWEEPS=1, so it can never source a measured sweep
    -- and pass 2 then refuses to launch on any machine holding an old
    `measured_env.sh`, which a real run leaves behind. Found red: the dry run
    stopped at stage 10 and checked no part of the launch command."""

    assert "SKIP_SWEEPS=1" in dryrun_text
    assert "export ALLOW_UNMEASURED_LAUNCH=1" in dryrun_text, (
        "the dry run skips the sweeps, so without this it stops at the "
        "pass-2 guard and never reaches the launch command it exists to check"
    )


def test_the_dry_run_cannot_die_on_its_own_empty_grep(dryrun_text):
    """`set -e` plus `pipefail` made a grep that matched nothing kill the
    harness ONE LINE above the check that reports exactly that -- so its most
    important failure was the one it could not name."""

    line = next(
        line for line in dryrun_text.splitlines() if line.startswith("LAUNCH=$(")
    )
    assert line.rstrip().endswith("|| true"), (
        f"{line!r} exits non-zero when no launch line was assembled, and the "
        "harness dies before reporting it"
    )


def test_pass_two_refuses_to_launch_on_defaults_when_a_sweep_exists(setup_text):
    """The whole point of two passes.

    `RUST_SLOTS` and friends carry cloud6 defaults, so pass 2 without sourcing
    `measured_env.sh` launches on those while printing "Measured generation
    flags" -- and the two cases produce identical command lines on a run that
    lasts a day.
    """

    assert "SWEEP_MEASURED" in setup_text
    assert "measured_env.sh" in setup_text
    assert "ALLOW_UNMEASURED_LAUNCH" in setup_text
    guard = setup_text[setup_text.index("The pass-2 guard") :][:1500]
    assert "die" in guard, "the guard must refuse, not warn"


def test_the_measured_wording_is_only_used_when_it_was_measured(setup_text):
    """Reporting defaults as "measured" is worse than reporting nothing."""

    # The emitting line, not the comment above the guard that quotes it.
    index = setup_text.index('ok "Measured generation flags')
    window = setup_text[max(0, index - 200) : index]
    assert 'SWEEP_MEASURED" = "1"' in window
    # ... and the unmeasured branch must say so rather than staying silent.
    assert "NOT measured on this box" in setup_text


def test_the_sweep_installs_the_solver_node_budget(setup_text):
    """Threads alone measure nothing.

    `f4_phase_d_sweep` gates solving on `solver_threads > 0 AND
    solver_max_nodes > 0`, and this stage passed only the first -- so
    `solver_wants` refused every position and the core-split axis measured a
    solver that never ran. That is the defect THROUGHPUT_LEVERS.md section 3.1
    records, on a run where the solver took 22-37% of generation wall, and
    `rehearse_sweep_laptop.sh` already asserts against it.

    The RUN's budget, not a sweep-specific one: a split measured against a
    cheaper solver is a split for a run nobody is launching.
    """

    block = _block(setup_text, "SWEEP_SOLVER_ARGS=()", "  else")
    assert "--solver-max-nodes" in block, (
        "the sweep configures solver threads without a node budget, so every "
        "point runs with solving disabled"
    )
    assert "$ENDGAME_SOLVER_MAX_NODES" in block, (
        "the sweep must use the run's own budget"
    )


def test_the_sweep_asserts_the_solver_actually_solved(setup_text):
    """Liveness, not configuration.

    Asserting that threads were CONFIGURED is exactly what let the missing node
    budget go unnoticed -- every point reported a split and none of them solved
    anything.
    """

    assert "solves_attempted" in setup_text, (
        "nothing checks that the swept solver did any work"
    )
    index = setup_text.index("solves_attempted")
    window = setup_text[max(0, index - 2000) : index + 2000]
    assert "measured a solver that never ran" in window, (
        "the solves_attempted check does not stop the run"
    )


def test_the_worker_count_is_swept_rather_than_pinned(setup_text):
    """Defaulting the axis to the shipped value sweeps ONE point.

    That is the same shape as measuring one solver split and reporting it as
    the answer: the grid returns the setting it was given and nothing about it
    was measured. Shard count decides whether the CPU can keep the coalesced
    batch full, and post-coalescer it no longer fragments batches, so there is
    no longer a reason to hold it fixed.

    Centred on the shipped value so the current setting is always IN the grid:
    a sweep that cannot return today's configuration cannot say it was right.
    """

    assert "SWEEP_WORKERS_DEFAULT" in setup_text
    assert '--workers "${SWEEP_WORKERS_CSV:-$SWEEP_WORKERS_DEFAULT}"' in setup_text, (
        "the workers axis still defaults to the single shipped value"
    )
    block = _block(setup_text, 'SWEEP_WORKERS_DEFAULT="${SWEEP_WORKERS_DEFAULT:-', "  )}")
    assert "RUST_SCHEDULER_WORKERS / 2" in block
    assert "RUST_SCHEDULER_WORKERS * 2" in block


def test_the_sweep_measures_the_generation_solver_core_split(setup_text):
    """The split is contended -- the solver runs synchronously inside a shard,
    so a thread given to it is a thread taken from leaf production. A single
    fixed value measures one split and reports it as the answer."""

    assert "SWEEP_SOLVER_THREADS_CSV" in setup_text
    assert "--solver-threads-total" in setup_text


def test_the_preflight_is_told_which_league_the_run_will_play(setup_text):
    """A specialist archives a checkpoint per TRAIN STEP and prunes none, and
    disk cannot be raised after the instance is rented."""

    preflight = _block(setup_text, '"$PY" -m games.seven_wonders_duel.cloud_preflight', "  &&")
    assert "--specialists" in preflight


@pytest.mark.skipif(not RUN_FILE.is_file(), reason="run file not present")
def test_the_run_file_sets_no_scheduler_geometry():
    """The decision file is portable; the geometry belongs to one rented box.

    Baking a slot count into the run file would silently carry a dead box's
    measurement onto a live one.
    """

    text = RUN_FILE.read_text(encoding="utf-8")
    for measured in (
        "export RUST_SLOTS=",
        "export RUST_GLOBAL_BATCH_CAP=",
        "export RUST_MAX_INFLIGHT_BATCHES=",
        "export GATE_SLOTS=",
        "export SOLVER_THREADS=",
    ):
        assert measured not in text, f"{measured} pins a per-box measurement"


@pytest.mark.skipif(not RUN_FILE.is_file(), reason="run file not present")
def test_the_run_file_only_sets_knobs_the_launcher_reads():
    """A typo here is silent: an unread export is indistinguishable from a
    setting that had no effect."""

    text = RUN_FILE.read_text(encoding="utf-8")
    setup = SETUP.read_text(encoding="utf-8")
    exported = set(re.findall(r"^export ([A-Z0-9_]+)=", text, flags=re.M))
    unread = sorted(name for name in exported if name not in setup)
    assert not unread, f"the launcher never reads: {unread}"


# ---------------------------------------------------------------------------
# measured_env.sh is SOURCED, so it has to be sourceable
# ---------------------------------------------------------------------------


def _sweep_dir(tmp_path, *, vary_solver: bool):
    import json

    generation = tmp_path / "generation"
    generation.mkdir(parents=True)
    rows = [
        {
            "slots": 256,
            "global_batch_cap": 2048,
            "max_inflight_batches": 1,
            "scheduler_workers": 4,
            "solver_threads_per_shard": 2,
            "median_seconds": 10.0,
        },
        {
            "slots": 128,
            "global_batch_cap": 1024,
            "max_inflight_batches": 1,
            "scheduler_workers": 4,
            "solver_threads_per_shard": 2 if not vary_solver else 0,
            "median_seconds": 12.0,
        },
    ]
    (generation / "phase_d_sweep.json").write_text(
        json.dumps({"summary": rows}), encoding="utf-8"
    )
    (tmp_path / "gate_200.json").write_text(
        json.dumps({"best": {"slots": 144, "global_batch_cap": 256}}),
        encoding="utf-8",
    )
    return tmp_path


def _render(tmp_path, *, vary_solver: bool) -> str:
    from games.seven_wonders_duel.sweep_launch_env import build_env, render

    directory = _sweep_dir(tmp_path, vary_solver=vary_solver)
    return render(build_env(directory, "200"))


def test_the_measured_env_carries_the_provenance_the_guard_reads(tmp_path):
    """`SKIP_SWEEPS` cannot serve as the marker -- an operator sets that by hand
    to skip measuring, which is exactly the case the guard must catch."""

    text = _render(tmp_path, vary_solver=True)
    assert "export SWEEP_MEASURED=1" in text
    assert "SWEEP_MEASURED_FROM=" in text
    assert "export SKIP_SWEEPS=1" in text


def test_the_solver_split_is_pinned_only_when_it_was_varied(tmp_path):
    """A grid that held the split fixed measured one split and says nothing
    about the others; emitting its value would dress a constant as a result."""

    varied = _render(tmp_path / "a", vary_solver=True)
    assert "export SOLVER_THREADS=2" in varied
    fixed = _render(tmp_path / "b", vary_solver=False)
    assert "export SOLVER_THREADS=" not in fixed
    assert "deliberately ABSENT" in fixed


def test_the_measured_env_survives_a_path_with_a_space(tmp_path):
    """The file is `source`d, so an unquoted value containing a space would
    split into two words and export something that is not the measurement."""

    directory = tmp_path / "run dir with spaces"
    directory.mkdir()
    text = _render(directory, vary_solver=True)
    written = directory / "measured_env.sh"
    written.write_text(text, encoding="utf-8")

    bash = shutil.which("bash")
    if bash is None:  # pragma: no cover - environment dependent
        pytest.skip("bash is not available")
    probe = subprocess.run(
        [
            bash,
            "-c",
            f'source "{written.as_posix()}"; '
            'echo "$RUST_SLOTS|$SWEEP_MEASURED|$SWEEP_MEASURED_FROM"',
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    slots, measured, source = probe.stdout.strip().split("|", 2)
    assert slots == "256"
    assert measured == "1"
    assert source.endswith("run dir with spaces")


# ---------------------------------------------------------------------------
# The sweep measures the RUN: --emit-config, --config-from-manifest, and what
# buying the box hours back is allowed to cost
# ---------------------------------------------------------------------------


def test_the_sweep_is_told_the_configuration_the_run_will_launch(setup_text):
    """Without this the sweep measures `PhaseDConfig`'s defaults for everything
    the launcher does not name -- Gumbel at 128 full simulations, to configure a
    PUCT run at 1600. Simulations per move set the leaf arrival rate, which is
    exactly what the slot and worker axes act on, so the optimum found that way
    belongs to a machine nobody is running.

    The run's own manifest cannot serve: it does not exist until the run starts,
    and the sweep runs first.
    """

    assert "--emit-config" in setup_text, (
        "nothing describes this run to the sweep"
    )
    block = _invocation(setup_text, "f4_staged_sweep")
    assert '--config-from-manifest "$RUN_CONFIG_JSON"' in block, (
        "the sweep is not pointed at the emitted config, so it measures "
        "dataclass defaults"
    )
    # And the emit has to happen BEFORE the sweep reads it.
    assert setup_text.index("--emit-config") < setup_text.index(
        "games.seven_wonders_duel.f4_staged_sweep"
    )


def test_emit_config_exists_and_does_not_train():
    from .phase_d import build_parser

    assert "--emit-config" in _parser_options(build_parser())
    source = (REPO_ROOT / "games/seven_wonders_duel/phase_d.py").read_text(
        encoding="utf-8"
    )
    block = source[source.index("if args.emit_config:") :][:800]
    assert "return 0" in block
    assert "loop.run()" not in block, "emitting a config must not start training"


def test_the_generation_sweep_is_staged_rather_than_a_full_product(setup_text):
    """Six axes at the run's own search is a sweep that costs more than the run
    it configures. The staged driver ranks geometry first and sweeps the
    batching axes at the winner: 18 + 6 points against 108."""

    block = _invocation(setup_text, "f4_staged_sweep")
    for flag in ("--stage-a-inflight", "--stage-a-wait-ms"):
        assert flag in block, f"{flag} is not pinned, so the stages are not tied"


def test_the_stage_a_pins_are_inside_the_axes_they_pin(setup_text):
    """The driver refuses a pin outside its axis, because stage B would then
    never re-measure the point stage A chose. The launcher's own defaults must
    not be the case that trips it."""

    from .f4_staged_sweep import _values

    for pin, axis in (
        ("SWEEP_STAGE_A_INFLIGHT", "SWEEP_INFLIGHT_CSV"),
        ("SWEEP_STAGE_A_WAIT_MS", "SWEEP_INFERENCE_WAIT_CSV"),
    ):
        pinned = re.search(rf'^{pin}="\$\{{{pin}:-([^}}]*)\}}"$', setup_text, re.M)
        assert pinned, f"{pin} has no default"
        swept = re.search(rf'"\$\{{{axis}:-([^}}]*)\}}"', setup_text)
        assert swept, f"{axis} has no default"
        assert pinned.group(1) in _values(swept.group(1)), (
            f"{pin}={pinned.group(1)} is not one of {axis}={swept.group(1)}, so "
            "the staged sweep would refuse its own launcher's defaults"
        )


def test_the_sweep_budget_knob_scales_the_solver_by_the_same_factor(setup_text):
    """Dividing the simulations without dividing the solver's node budget would
    hand the solver the concurrency cheaper generation vacates, and the sweep
    would rank geometry against a solver share no run has."""

    assert 'SWEEP_SIMS_DIVISOR="${SWEEP_SIMS_DIVISOR:-4}"' in setup_text
    block = _invocation(setup_text, "f4_staged_sweep")
    assert '--sims-divisor "$SWEEP_SIMS_DIVISOR"' in block

    source = (REPO_ROOT / "games/seven_wonders_duel/f4_phase_d_sweep.py").read_text(
        encoding="utf-8"
    )
    divided = source[source.index("if args.sims_divisor > 1 and solver_max_nodes > 0:") :][
        :700
    ]
    assert "solver_max_nodes / args.sims_divisor" in divided, (
        "the node budget is not divided by the same factor as the simulations"
    )
    assert "args.solver_max_secs / args.sims_divisor" in divided, (
        "an explicit deadline is not divided, so node declines become deadline "
        "declines -- which makes a proof depend on how busy the box was"
    )


def test_the_divisor_keeps_the_search_shape_it_claims_to_keep():
    """The whole claim of `--sims-divisor` is that it runs the SAME search
    shallower. A divisor that also moved the algorithm, the mix or top_k would
    be the defect it was written to avoid, one layer down."""

    from .f4_phase_d_sweep import apply_sims_divisor
    from .phase_d import PhaseDConfig

    config = PhaseDConfig(
        run_dir="x",
        selfplay_search_mode="puct",
        cheap_search_mode="gumbel",
        cheap_sims_min=100,
        cheap_sims_max=100,
        full_sims_min=1600,
        full_sims_max=1600,
        top_k=16,
        full_search_fraction=0.25,
    )
    apply_sims_divisor(config, 4)
    assert (config.cheap_sims_min, config.cheap_sims_max) == (25, 25)
    assert (config.full_sims_min, config.full_sims_max) == (400, 400)
    assert config.selfplay_search_mode == "puct"
    assert config.cheap_search_mode == "gumbel"
    assert config.top_k == 16
    assert config.full_search_fraction == 0.25


def test_the_divisor_never_rounds_a_budget_to_zero():
    from .f4_phase_d_sweep import apply_sims_divisor
    from .phase_d import PhaseDConfig

    config = PhaseDConfig(
        run_dir="x", cheap_sims_min=1, cheap_sims_max=2, full_sims_min=1,
        full_sims_max=2,
    )
    apply_sims_divisor(config, 64)
    assert config.cheap_sims_min >= 1 and config.full_sims_min >= 1
    config.validate()


def test_a_divisor_of_one_changes_nothing():
    """The default must be inert, or every undivided sweep carries a rounding
    this was never meant to apply."""

    from .f4_phase_d_sweep import apply_sims_divisor
    from .phase_d import PhaseDConfig

    config = PhaseDConfig(
        run_dir="x", cheap_sims_min=100, cheap_sims_max=100,
        full_sims_min=1600, full_sims_max=1600,
    )
    apply_sims_divisor(config, 1)
    assert (config.cheap_sims_max, config.full_sims_max) == (100, 1600)


# ---------------------------------------------------------------------------
# The staged sweep, and what `sweep_launch_env` has to learn from it
# ---------------------------------------------------------------------------


def _stage_row(**overrides):
    row = {
        "slots": 256,
        "global_batch_cap": 2048,
        "max_inflight_batches": 1,
        "scheduler_workers": 4,
        "inference_wait_ms": 0.0,
        "solver_threads_per_shard": 2,
        "solver_threads_total": 8,
        "median_seconds": 10.0,
        "median_games_per_hour": 360.0,
        "median_parked_slot_fraction": 0.18,
        "median_requests_per_forward": 2.4,
    }
    row.update(overrides)
    return row


def _staged_sweep_dir(tmp_path, *, sims_divisor=1):
    """A staged output: stage A varied the split, stage B held it constant.

    This is the exact shape that defeats the unstaged rule. `summary` is stage
    B's, whose solver column is constant BECAUSE STAGE A PINNED IT -- not
    because nobody measured it.
    """

    import json

    generation = tmp_path / "generation"
    generation.mkdir(parents=True)
    stage_b = [
        _stage_row(inference_wait_ms=2.0, median_seconds=9.0),
        _stage_row(inference_wait_ms=0.0, median_seconds=10.0),
    ]
    stage_a = [
        _stage_row(),
        _stage_row(solver_threads_per_shard=0, solver_threads_total=0,
                   median_seconds=11.0),
    ]
    (generation / "phase_d_sweep.json").write_text(
        json.dumps(
            {
                "config": {"sims_divisor": sims_divisor, "staged": True},
                "summary": stage_b,
                "staged": {
                    "winner": stage_b[0],
                    "swept_axes": [
                        "inference_wait_ms",
                        "solver_threads_total",
                    ],
                    "carryover_drift": 0.01,
                    "stages": [
                        {"name": "geometry", "summary": stage_a},
                        {"name": "batching", "summary": stage_b},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "gate_200.json").write_text(
        json.dumps({"best": {"slots": 144, "global_batch_cap": 256}}),
        encoding="utf-8",
    )
    return tmp_path


def test_a_staged_sweep_pins_the_split_its_first_stage_measured(tmp_path):
    """Stage B's summary has ONE solver value, so the unstaged rule
    (`len(splits) > 1`) would conclude the split was never swept and decline to
    pin it -- throwing away the measurement stage A paid for."""

    from games.seven_wonders_duel.sweep_launch_env import build_env, render

    env = build_env(_staged_sweep_dir(tmp_path), "200")
    assert env["SOLVER_THREADS"] == 2, (
        "the staged sweep measured the split at stage A and it was dropped"
    )
    assert env["RUST_INFERENCE_WAIT_MS"] == 2.0
    assert env["_WAIT_WAS_SWEPT"] is True
    text = render(env)
    assert "MEASURED here, so it is pinned" in text
    assert "STAGED sweep" in text


def test_the_staged_winner_is_the_row_the_driver_named(tmp_path):
    """Not `summary[0]` by luck. They agree here on purpose -- stage B's summary
    is sorted fastest-first -- and the point is that the winner is READ rather
    than re-derived, so a driver that ever sorts differently cannot silently
    hand over a different point."""

    from games.seven_wonders_duel.sweep_launch_env import build_env

    env = build_env(_staged_sweep_dir(tmp_path), "200")
    assert env["RUST_SLOTS"] == 256
    assert env["RUST_GLOBAL_BATCH_CAP"] == 2048
    assert env["RUST_SCHEDULER_WORKERS"] == 4


def test_a_divided_sweep_says_so_where_the_operator_reads_it(tmp_path):
    """The geometry is still the geometry to launch on. The games/hour is not
    the run's rate, and the file carrying the numbers is the only place anyone
    would find that out."""

    from games.seven_wonders_duel.sweep_launch_env import build_env, render

    divided = render(build_env(_staged_sweep_dir(tmp_path / "a", sims_divisor=4), "200"))
    assert "1/4 of the run's SIMULATION budget" in divided
    assert "not a prediction" in divided
    # ...and it must not be exported: it is not a launcher knob.
    assert "export _SIMS_DIVISOR" not in divided
    assert "SIMS_DIVISOR=" not in [
        line.removeprefix("export ").split("=", 1)[0] + "="
        for line in divided.splitlines()
        if line.startswith("export ")
    ]

    plain = render(build_env(_staged_sweep_dir(tmp_path / "b", sims_divisor=1), "200"))
    assert "SIMULATION budget" not in plain


def test_an_unstaged_sweep_still_works_unchanged(tmp_path):
    """The staged block is additive. A `phase_d_sweep.json` from the plain
    harness -- which is what `sweep_7wd.sh` still writes -- must read exactly as
    it did before."""

    from games.seven_wonders_duel.sweep_launch_env import build_env

    env = build_env(_sweep_dir(tmp_path, vary_solver=True), "200")
    assert env["RUST_SLOTS"] == 256
    assert env["SOLVER_THREADS"] == 2
    assert "_STAGED_AXES" not in env


def test_the_driver_refuses_a_pin_its_second_stage_never_measures():
    """A pin outside the stage-B axis means the two stages share no row, so they
    can only be compared across a configuration change -- and the carryover
    check that catches a drifting box has nothing to compare."""

    from games.seven_wonders_duel import f4_staged_sweep

    with pytest.raises(SystemExit) as raised:
        f4_staged_sweep.main(
            [
                "--checkpoint", "x", "--output", "y",
                "--inflight", "1,2", "--stage-a-inflight", "4",
            ]
        )
    assert "never re-measure" in str(raised.value)


def test_the_driver_reports_the_split_by_the_column_that_means_it():
    """`solver_threads_total` is split x workers, so it varies whenever the
    WORKER axis does -- a grid that held the total fixed across workers has a
    moving total and a split nobody swept. `sweep_launch_env` has always read
    the per-shard column for this, and the staged output has to agree or a
    constant gets dressed up as a measurement."""

    from games.seven_wonders_duel.f4_staged_sweep import AXES

    _flag, reported, varied_key = AXES["solver"]
    assert reported == "solver_threads_total"
    assert varied_key == "solver_threads_per_shard"


def test_the_driver_pins_stage_a_s_winner_into_stage_b(tmp_path, monkeypatch):
    """The plumbing, end to end, without a GPU: stage A's winning geometry must
    arrive at stage B as PINNED AXES, and the solver must arrive as the winning
    row's own per-shard count rather than as the flag's.

    That last one is the trap. `f4_phase_d_sweep` falls back to
    `--solver-threads` whenever the total axis is exactly `[0]`, so forwarding
    the launcher's per-shard value would revive the solver at a stage whose
    pinned total is 0 -- and "solver off" is a measured point with its own cost
    curve, not an absence.
    """

    import json
    from games.seven_wonders_duel import f4_staged_sweep

    seen: list[dict] = []

    def fake_main(argv):
        flags = {}
        for index, token in enumerate(argv):
            if token.startswith("--") and index + 1 < len(argv):
                flags[token] = argv[index + 1]
        seen.append(flags)
        # Stage A: the 512-slot / 1-shard / solver-off point wins.
        if len(seen) == 1:
            rows = [
                _stage_row(slots=512, scheduler_workers=1,
                           solver_threads_per_shard=0, solver_threads_total=0,
                           median_seconds=8.0),
                _stage_row(median_seconds=9.0),
            ]
        else:
            rows = [
                _stage_row(slots=512, scheduler_workers=1,
                           solver_threads_per_shard=0, solver_threads_total=0,
                           max_inflight_batches=2, median_seconds=7.0),
                _stage_row(slots=512, scheduler_workers=1,
                           solver_threads_per_shard=0, solver_threads_total=0,
                           median_seconds=8.0),
            ]
        return {"summary": rows, "config": {"sims_divisor": 4}}

    monkeypatch.setattr(f4_staged_sweep.sweep, "main", fake_main)
    assert (
        f4_staged_sweep.main(
            [
                "--checkpoint", "ckpt.pt",
                "--output", str(tmp_path),
                "--slots", "256,512",
                "--workers", "1,4",
                "--solver-threads-total", "0,8",
                "--solver-threads", "3",
                "--inflight", "1,2",
                "--inference-wait-ms", "0,2",
                "--sims-divisor", "4",
            ]
        )
        == 0
    )

    stage_a, stage_b = seen
    # Stage A sweeps geometry and sits still on the batching axes.
    assert stage_a["--slots"] == "256,512" and stage_a["--workers"] == "1,4"
    assert stage_a["--inflight"] == "1" and stage_a["--inference-wait-ms"] == "0"
    # Stage B is the mirror image, at the winner.
    assert stage_b["--slots"] == "512" and stage_b["--workers"] == "1"
    assert stage_b["--inflight"] == "1,2"
    assert stage_b["--inference-wait-ms"] == "0,2"
    assert stage_b["--solver-threads-total"] == "0"
    assert stage_b["--solver-threads"] == "0", (
        "stage B revived the solver the winning point had OFF"
    )
    # And both stages measured the same search.
    assert stage_a["--sims-divisor"] == stage_b["--sims-divisor"] == "4"

    payload = json.loads(
        (tmp_path / "phase_d_sweep.json").read_text(encoding="utf-8")
    )
    assert payload["summary"][0]["max_inflight_batches"] == 2
    assert payload["staged"]["winner"]["max_inflight_batches"] == 2
    # Varied at stage A only, at stage B only, and nowhere.
    swept = payload["staged"]["swept_axes"]
    assert "slots" in swept and "scheduler_workers" in swept
    assert "solver_threads_total" in swept
    assert "max_inflight_batches" in swept
    assert "global_batch_cap" not in swept
    # Stage B re-measured stage A's point, so the stages can be compared.
    assert payload["staged"]["carryover_drift"] == pytest.approx(0.0)


def test_the_driver_refuses_a_stage_b_that_lost_the_pin(tmp_path, monkeypatch):
    """A flag forwarded to the wrong stage produces a winner describing a
    geometry nobody measured, and it looks exactly like a legitimate result."""

    from games.seven_wonders_duel import f4_staged_sweep

    calls = []

    def fake_main(argv):
        calls.append(argv)
        slots = 256 if len(calls) == 1 else 128
        return {"summary": [_stage_row(slots=slots)], "config": {}}

    monkeypatch.setattr(f4_staged_sweep.sweep, "main", fake_main)
    with pytest.raises(SystemExit) as raised:
        f4_staged_sweep.main(
            ["--checkpoint", "c", "--output", str(tmp_path)]
        )
    assert "pin did not reach the harness" in str(raised.value)


# ---------------------------------------------------------------------------
# rehearse_sweep_laptop.sh -- it only rehearses what the box actually runs
# ---------------------------------------------------------------------------

REHEARSAL = REPO_ROOT / "rehearse_sweep_laptop.sh"


@pytest.fixture(scope="module")
def rehearsal_text() -> str:
    return REHEARSAL.read_text(encoding="utf-8")


def test_the_rehearsal_drives_the_harness_the_box_drives(rehearsal_text, setup_text):
    """This script exists to prove the box's stage-8b plumbing on a laptop
    before anything is rented. A rehearsal of a DIFFERENT harness than the one
    stage 8b calls is worse than none: it reports OK for a pipeline nobody is
    going to run."""

    launcher = re.search(
        r"games\.seven_wonders_duel\.(f4_\w*sweep)", setup_text
    )
    assert launcher, "the launcher calls no generation sweep"
    assert f"games.seven_wonders_duel.{launcher.group(1)}" in rehearsal_text, (
        f"the launcher drives {launcher.group(1)} and the rehearsal does not"
    )


def test_the_rehearsal_checks_the_staged_handoff(rehearsal_text):
    """The one thing the staged output adds over the plain one is `swept_axes`,
    and nothing else in the rehearsal would notice if it stopped arriving --
    the geometry would still be right and SOLVER_THREADS would just vanish."""

    assert "swept_axes" in rehearsal_text
    assert "carryover_drift" in rehearsal_text
    assert "export SOLVER_THREADS=" in rehearsal_text, (
        "nothing checks the split reaches measured_env.sh"
    )


def test_the_rehearsal_reads_both_stages_not_just_the_last(rehearsal_text):
    """Stage B pins everything stage A won, so its summary shows one solver
    split, one slot count and one shard count. Every `did this axis vary` check
    would fail on a sweep that measured all of them."""

    assert "stage_summaries" in rehearsal_text
    assert 'staged["stages"]' in rehearsal_text


def test_the_solver_liveness_check_reads_both_stages(setup_text):
    """The batching stage pins whatever the geometry stage won, including the
    split. If the solver-OFF point wins, every row in `summary` reads zero and
    the check reports "no point ran with the solver on" about a sweep that
    measured the split thoroughly -- passing, silently, for the wrong reason."""

    block = setup_text[setup_text.index("<<'PYSOLVES'") :]
    block = block[: block.index("PYSOLVES\n", 20)]
    assert 'staged["stages"]' in block, (
        "the liveness check reads only the last stage's summary"
    )


def test_the_emitted_config_is_the_search_the_sweep_then_measures(tmp_path):
    """The whole chain, on a laptop: launcher-style flags -> `--emit-config` ->
    `config_from_manifest`. Each half has its own test; neither notices if the
    JSON one writes is not the JSON the other reads, and that is the failure
    this ordering change exists to make impossible.
    """

    import json
    import sys

    from .f4_phase_d_sweep import (
        apply_sims_divisor,
        config_from_manifest,
        _find_manifest_value,
    )

    emitted = tmp_path / "run_config.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "games.seven_wonders_duel.phase_d",
            "--emit-config", str(emitted),
            "--run-dir", str(tmp_path / "run"), "--device", "cpu",
            "--selfplay-search-mode", "puct", "--cheap-search-mode", "gumbel",
            "--cheap-sims-min", "100", "--cheap-sims-max", "100",
            "--full-sims-min", "1600", "--full-sims-max", "1600",
            "--top-k", "16", "--full-search-fraction", "0.25",
            "--endgame-solver-max-nodes", "40000000",
            "--endgame-solver-max-secs", "170", "--solver-threads", "1",
            "--pooled-readout", "--reply-head",
            "--rust-slots", "256", "--rust-global-batch-cap", "2048",
            "--rust-scheduler-workers", "4",
            "--train-steps", "120", "--weight-decay", "0.5",
            "--leaf-batch", "1", "--virtual-loss-root",
            "--cheap-leaf-batch", "16", "--eval-leaf-batch", "16",
            "--cheap-conflict-free-waves", "--cheap-round-robin-candidates",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stdout[-800:] + proc.stderr[-800:]

    config = config_from_manifest(
        emitted,
        output=tmp_path / "sweep",
        device="cpu",
        games=8,
        precision="fp32",
        geometry={"d_model": 128, "layers": 4, "heads": 4},
    )
    # The search, which is what the whole reorder was for.
    assert config.selfplay_search_mode == "puct"
    assert (config.cheap_sims_max, config.full_sims_max) == (100, 1600)
    assert config.top_k == 16 and config.full_search_fraction == 0.25
    # The architecture, which decides what a leaf costs.
    assert config.pooled_readout and config.reply_head
    # The solver budget, which the sweep reads out of the manifest separately.
    assert (
        _find_manifest_value(
            json.loads(emitted.read_text(encoding="utf-8")),
            "endgame_solver_max_nodes",
        )
        == 40000000
    )
    # And the UNMEASURED geometry, which is the baseline the grid is ranked
    # against -- "what would changing this buy me" is only answered against the
    # status quo. It is absent from the emitted config only if the launcher
    # appended its measured flags too early.
    assert config.rust_slots == 256
    assert config.rust_global_batch_cap == 2048
    assert config.rust_scheduler_workers == 4

    apply_sims_divisor(config, 4)
    assert (config.cheap_sims_max, config.full_sims_max) == (25, 400)


# ---------------------------------------------------------------------------
# Two solver caps: the attempt bar and the timeout
# ---------------------------------------------------------------------------


def test_the_attempt_bar_is_reachable_from_the_launcher(setup_text):
    """A knob the launcher cannot set is a decision nobody can make. The bar
    defaults to the timeout, so leaving it unreachable would be invisible --
    the run would work, and would simply never narrow admission."""

    assert 'ENDGAME_SOLVER_ATTEMPT_NODES="${ENDGAME_SOLVER_ATTEMPT_NODES:-0}"' in setup_text
    block = _block(setup_text, "SOLVER_FLAGS=()", "fi")
    assert "--endgame-solver-attempt-nodes" in block


def test_the_attempt_bar_defaults_to_the_timeout():
    """0 must reproduce the single shared number exactly, or shipping the split
    changes every run that does not set it."""

    from .phase_d import configure_endgame_solver

    applied = configure_endgame_solver(40_000_000, 75.0, 0, True)
    try:
        if applied is None:
            pytest.skip("no Rust generator to configure")
        assert applied[0] == 40_000_000
        assert applied[4] == 40_000_000, "the 0 sentinel did not resolve"
    finally:
        configure_endgame_solver(0, 60.0, 0, False)


def test_a_bar_above_the_timeout_is_refused_by_the_launcher_path():
    """Every position admitted above the timeout spends it in full and answers
    nothing -- the exact waste the split removes. Refused where the flag names
    are known, so the message names knobs rather than Rust arguments."""

    from .phase_d import configure_endgame_solver

    with pytest.raises(ValueError, match="exceeds"):
        configure_endgame_solver(40_000_000, 75.0, 0, True, 80_000_000)


def test_the_sweep_carries_the_runs_attempt_bar(sweep_text):
    """A sweep that left the bar at its default would measure the run's TIMEOUT
    as its admission threshold. On a run that narrowed the bar that admits a far
    larger set of positions, so the solver load being measured is not the run's
    -- the same defect class as measuring a solver that never ran, inverted."""

    source = (REPO_ROOT / "games/seven_wonders_duel/f4_phase_d_sweep.py").read_text(
        encoding="utf-8"
    )
    assert "--solver-attempt-nodes" in source
    assert 'endgame_solver_attempt_nodes' in source, (
        "the sweep never reads the run's bar out of the manifest"
    )
    # And it must reach run_point, not merely be parsed.
    assert "solver_attempt_nodes=solver_attempt_nodes" in source
    del sweep_text


def test_the_divisor_scales_the_bar_with_the_timeout():
    """Dividing the timeout alone would leave the same positions admitted
    against a quarter of the budget, turning proofs into declines and measuring
    a solver that fails far more often than the run's does."""

    source = (REPO_ROOT / "games/seven_wonders_duel/f4_phase_d_sweep.py").read_text(
        encoding="utf-8"
    )
    block = source[source.index("if args.sims_divisor > 1 and solver_max_nodes > 0:") :][
        :1400
    ]
    assert "solver_attempt_nodes / args.sims_divisor" in block


def test_the_prediction_is_recorded_on_the_move(setup_text):
    """The attempt bar filters on the prediction. Without it in the buffer the
    bar cannot be tuned from a run's own output: the only costs observable are
    those of positions the bar already admitted."""

    del setup_text
    from .buffer import MoveRecord
    import dataclasses

    names = {f.name for f in dataclasses.fields(MoveRecord)}
    assert "solver_predicted_nodes" in names
    source = (REPO_ROOT / "games/seven_wonders_duel/buffer.py").read_text(
        encoding="utf-8"
    )
    # Both directions, or a buffer loses it on the way back in.
    assert '"solver_predicted_nodes": move.solver_predicted_nodes' in source
    assert 'solver_predicted_nodes=move.get("solver_predicted_nodes")' in source


def test_an_old_buffer_without_the_prediction_still_loads():
    """Every buffer written before this field exists, which is all of them.

    Exercised through `from_json_line`, not the constructor: the constructor
    would only prove the dataclass has a default, while the thing that has to
    hold is that a line with no such key still parses.
    """

    import json

    from .buffer import from_json_line, to_json_line, GameRecord, MoveRecord

    record = GameRecord(
        seed=1, first_player=0, agents={}, iteration=0, winner=0,
        victory_type="civilian", scores=(1, 0), chance_log=(),
        moves=(
            MoveRecord(
                i=0, actor=0, action=0, mask_hash="sha256:0",
                solver_attempted=True, solver_nodes=7,
                solver_predicted_nodes=1234.0,
            ),
        ),
        final_digest="sha256:0", trajectory_digest="sha256:0",
    )
    line = to_json_line(record)
    payload = json.loads(line)
    # Strip the key the way a pre-split buffer would not have had it at all.
    for move in payload["moves"]:
        move.pop("solver_predicted_nodes", None)
    reloaded = from_json_line(json.dumps(payload))
    assert reloaded.moves[0].solver_predicted_nodes is None
    assert reloaded.moves[0].solver_nodes == 7


def test_parked_slots_are_excluded_from_the_slot_budget(setup_text):
    """`phase_d` defaults this OFF -- it shipped behind a flag and nobody
    flipped it -- so until now a slot parked on an endgame solve held its
    --rust-slots token AND counted toward `active_count`, which divides the
    batch cap and narrows the row allowance of every slot that IS working.

    Set here rather than left to the parser default, which is the same class of
    omission as --train-steps: a default nobody chose."""

    assert 'EXCLUDE_PARKED_FROM_BUDGET="${EXCLUDE_PARKED_FROM_BUDGET:-1}"' in setup_text
    block = _block(setup_text, "SOLVER_FLAGS=()", "fi")
    assert "--exclude-parked-from-budget" in block, (
        "the flag never reaches the launch line"
    )


def test_the_parked_slot_regime_is_decided_before_the_slot_sweep(setup_text):
    """It redefines --rust-slots -- concurrent games becomes concurrent
    SEARCHING games -- so a slot optimum measured under one regime does not
    transfer to the other. The sweep has to run under the regime the run will
    use, which means the decision must precede it."""

    decided = setup_text.index('EXCLUDE_PARKED_FROM_BUDGET="${EXCLUDE_PARKED_FROM_BUDGET:-1}"')
    swept = setup_text.index('stage 8b "Scheduler sweeps')
    assert decided < swept


def test_the_soak_runs_the_parked_slot_regime_the_box_runs(setup_text):
    """The soak exists to prove the box's mechanism set survives running
    together. A soak under the other slot regime rehearses something else."""

    soak = (REPO_ROOT / "run_laptop_soak.ps1").read_text(encoding="utf-8")
    assert "--exclude-parked-from-budget" in soak
    assert "ExcludeParkedFromBudget" in soak
    del setup_text


# ---------------------------------------------------------------------------
# Solver sizing: priced once off the box, measured on it
# ---------------------------------------------------------------------------


def test_the_node_rate_is_measured_under_contention(setup_text):
    """`measure_node_rate` returns the SINGLE-THREAD rate and says so. A run
    solves with --solver-threads x --rust-scheduler-workers of them beside the
    generation shards, and the per-thread rate falls under that contention --
    the re-solve study measured 857,015 nodes/s/thread across 8. Sizing a node
    budget off the uncontended figure overstates the box."""

    assert "measure_node_rate_contended" in setup_text
    assert "SOLVER_RATE_THREADS" in setup_text


def test_the_thread_split_is_decided_before_the_rate_is_measured(setup_text):
    """The rate is now measured AT a thread count, so that count has to exist
    first. It did not: the split used to be derived after the measurement,
    which was harmless only while the measurement was single-threaded."""

    split = setup_text.index("_total_solver=$(( SOLVER_THREADS * GENERATION_THREADS ))")
    rate = setup_text.index("measure_node_rate_contended")
    assert split < rate


def test_the_solver_caps_are_sized_after_the_generation_sweep(setup_text):
    """The budget is `threads x GENERATION WALL x rate x share`, and the wall is
    what the sweep measures. Sizing before it would need a wall nobody had."""

    swept = setup_text.index('stage 8b "Scheduler sweeps')
    sized = setup_text.index("games.seven_wonders_duel.solver_sizing")
    launch = setup_text.index("common::launch_detached")
    assert swept < sized < launch


def test_the_sizing_charges_the_solver_for_generation_not_the_iteration(setup_text):
    """Solving happens during generation. Charging it for training time would
    inflate its budget by however long the learner runs."""

    block = _block(setup_text, "_GEN_WALL=\"$(\"$PY\" - ", "PYWALL")
    assert "median_games_per_hour" in block
    assert "GAMES_PER_ITERATION" in setup_text


def test_a_missing_corpus_is_reported_rather_than_guessed_around(setup_text):
    """Sizing the caps off nothing would produce numbers indistinguishable from
    measured ones -- the same failure the pass-2 guard exists for."""

    index = setup_text.index("SOLVER_CORPUS=")
    block = setup_text[index : index + 700]
    assert "No solver corpus" in block
    assert "warn " in block


def test_the_sized_caps_reach_the_file_pass_two_sources(setup_text):
    """Two files to source is one file to forget. The geometry and the caps were
    measured beside each other and have to travel together."""

    assert 'cat "$SWEEP_DIR/solver_env.sh" >> "$SWEEP_DIR/measured_env.sh"' in setup_text


def test_the_target_share_is_a_knob_with_headroom(setup_text):
    """Not 100%: the corpus prices one net's endgames, and a run reaches
    different ones as it strengthens."""

    match = re.search(
        r'^SOLVER_TARGET_SHARE="\$\{SOLVER_TARGET_SHARE:-([0-9.]+)\}"$',
        setup_text,
        re.M,
    )
    assert match, "the target share is not a knob"
    assert 0.0 < float(match.group(1)) <= 1.0
