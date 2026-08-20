"""Do the checks this project relies on have the power to fail?

Every gate here was believed to be protecting something. Two were not:

* `test_async_solver.py` -- the file the notes call THE gate for the async
  solver -- drove `self_play_many_mock`, which routes to a scheduler that never
  builds a `SolverPool`. Every `set_solver_threads` value took the identical
  synchronous path, so its identity assertions compared a run against itself. It
  could not fail, and had not, for weeks.
* `test_cost_model_parity.py` checked Rust against Python on bot games, which
  retire no wonders at all (0 of 146 positions). The one feature divergence that
  actually existed lived in retired-wonder handling, in 24% of the positions the
  model was fit on. It could not fail on the bug it was written to prevent.

Neither was caught by reading the code. Both were caught by something unrelated
happening to exercise the real path. That is not a reliable discovery process,
so this makes it deliberate: break the thing a gate guards, and confirm the gate
goes red.

Each mutation is a REAL bug class, not a syntax error -- an off-by-one in a
military band, a feature computed differently on one side of a language
boundary, a promotion rule that ignores its own confidence bound. A gate that
stays green under one of these is not testing what its name says.

Usage::

    python -m games.seven_wonders_duel.gate_power_audit            # all
    python -m games.seven_wonders_duel.gate_power_audit --only engine_equiv
    python -m games.seven_wonders_duel.gate_power_audit --list

The working tree must be clean for the files under mutation; the audit refuses
to run otherwise, because its restore step would otherwise discard real work.
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUST = REPO / "games" / "seven_wonders_duel" / "seven_wonders_rust"


@dataclasses.dataclass(frozen=True)
class Mutation:
    """One deliberate defect, and the gate that should notice it."""

    name: str
    path: str
    old: str
    new: str
    gate: tuple[str, ...]
    why: str
    needs_build: bool = False


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="engine_equiv",
        path="games/seven_wonders_duel/seven_wonders_rust/src/engine.rs",
        old="    pub fn legal_actions(&self) -> Vec<Action> {",
        new=(
            "    pub fn legal_actions(&self) -> Vec<Action> {\n"
            "        // MUTATION: drop the last legal action.\n"
            "        #[cfg(not(test))]\n"
            "        {\n"
            "            let mut out = self.legal_actions_unmutated();\n"
            "            out.pop();\n"
            "            return out;\n"
            "        }\n"
            "        #[allow(unreachable_code)]\n"
            "        self.legal_actions_unmutated()\n"
            "    }\n"
            "\n"
            "    pub fn legal_actions_unmutated(&self) -> Vec<Action> {"
        ),
        gate=("games/seven_wonders_duel/test_rust_engine_equiv.py",),
        why="Rust offering a different move set than Python is the divergence "
        "the equivalence corpus exists to catch.",
        needs_build=True,
    ),
    Mutation(
        name="cost_model_parity",
        path="games/seven_wonders_duel/seven_wonders_rust/src/cost_model.rs",
        old="        state.tableau.accessible_indices().len() as f64,",
        new="        (state.tableau.accessible_indices().len() + 1) as f64,  // MUTATION",
        gate=("games/seven_wonders_duel/test_cost_model_parity.py",),
        why="A feature computed differently in Rust than in the Python the "
        "coefficients were fit against. Never raises; silently prices every "
        "position with the wrong weights.",
        needs_build=True,
    ),
    Mutation(
        name="solver_mask",
        path="games/seven_wonders_duel/seven_wonders_rust/src/self_play.rs",
        old="fn mask_and_renormalise(policy: &mut [f64], keep: &[bool]) {",
        new=(
            "fn mask_and_renormalise(policy: &mut [f64], keep: &[bool]) {\n"
            "    // MUTATION: keep one provably-losing move alive.\n"
            "    let keep: Vec<bool> = keep\n"
            "        .iter()\n"
            "        .enumerate()\n"
            "        .map(|(i, &k)| k || i == 0)\n"
            "        .collect();\n"
            "    let keep = &keep[..];"
        ),
        gate=("games/seven_wonders_duel/test_endgame_solver_self_play.py",),
        why="The mask is the solver's whole contribution to the policy target. "
        "Leaving a losing move in it trains toward a proven loss.",
        needs_build=True,
    ),
    Mutation(
        name="promotion_rule",
        path="games/az_loop/stats.py",
        old="def wilson_interval(",
        new=(
            "def wilson_interval(*args, **kwargs):\n"
            "    # MUTATION: a bound that always clears any threshold.\n"
            "    return (1.0, 1.0)\n"
            "\n"
            "\n"
            "def _wilson_interval_unmutated("
        ),
        gate=("games/az_loop/",),
        why="The promotion gate decides whether a run learns. A bound that "
        "always clears means every candidate is promoted, including a "
        "regression.",
    ),
    Mutation(
        name="checkpoint_contract",
        path="games/seven_wonders_duel/train.py",
        old='        config[switch] = actual',
        new='        config[switch] = False  # MUTATION: record the wrong architecture',
        gate=(
            "games/seven_wonders_duel/test_pooled_readout.py",
            "games/seven_wonders_duel/test_checkpoint_rebuild_sites.py",
        ),
        why="A checkpoint whose config misdescribes its own weights. This bug "
        "shipped five times; the guard was added after the fifth.",
    ),
    Mutation(
        name="reply_target_leak",
        path="games/seven_wonders_duel/dataset.py",
        old='        if getattr(following, "policy_excluded", False):',
        new='        if False:  # MUTATION: let excluded policies supervise the reply head',
        gate=("games/seven_wonders_duel/test_reply_head.py",),
        why="Cheap-search and archived-net policies supervising the reply head "
        "is the exact leak the external review found.",
    ),
    Mutation(
        name="cost_model_feature_order",
        path="games/seven_wonders_duel/validate_cost_trigger.py",
        old="RUST_FEATURES = tuple(\n    name for name in TRIGGER_FEATURES if name not in NEEDS_REACHABILITY\n)",
        new=(
            "RUST_FEATURES = tuple(\n"
            "    reversed(  # MUTATION: weights applied to the wrong features\n"
            "        [name for name in TRIGGER_FEATURES if name not in NEEDS_REACHABILITY]\n"
            "    )\n"
            ")"
        ),
        gate=("games/seven_wonders_duel/test_cost_model_parity.py",),
        why="Weights are applied positionally in Rust, so a reordering never "
        "raises -- it prices every position with the wrong coefficients.",
    ),
)


def _run(cmd: list[str], cwd: Path = REPO) -> subprocess.CompletedProcess:
    # utf-8 with replacement, not the Windows console default: maturin prints
    # emoji, and cp1252 raised UnicodeDecodeError inside subprocess's reader
    # threads. That did not corrupt a verdict -- exit codes still came back --
    # but it discarded the output a verdict is explained by.
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def _dirty(paths: list[str]) -> list[str]:
    out = _run(["git", "status", "--porcelain", "--"] + paths).stdout
    return [line for line in out.splitlines() if line.strip()]


def _restore(path: str) -> None:
    _run(["git", "checkout", "--", path])


def _build() -> bool:
    built = _run(["cargo", "build", "--release"], cwd=RUST)
    if built.returncode != 0:
        return False
    installed = _run(
        [sys.executable, "-m", "maturin", "develop", "--release", "-m", str(RUST / "Cargo.toml")]
    )
    return installed.returncode == 0


def audit(mutation: Mutation, *, rebuild: bool) -> dict:
    """Apply, run the gate, restore. Returns what happened."""

    target = REPO / mutation.path
    source = target.read_text(encoding="utf-8")
    if mutation.old not in source:
        return {"name": mutation.name, "verdict": "SKIP", "detail": "anchor not found"}

    target.write_text(source.replace(mutation.old, mutation.new, 1), encoding="utf-8")
    try:
        if mutation.needs_build:
            if not rebuild:
                return {"name": mutation.name, "verdict": "SKIP", "detail": "needs --rebuild"}
            if not _build():
                return {
                    "name": mutation.name,
                    "verdict": "SKIP",
                    "detail": "mutation did not compile",
                }
        gate = _run(
            [sys.executable, "-m", "pytest", *mutation.gate, "-x", "-q", "-p", "no:randomly"]
        )
        detail = gate.stdout.strip().splitlines()[-1] if gate.stdout else ""
        # pytest: 0 all passed, 1 tests FAILED, 2 interrupted, 3 internal,
        # 4 usage, 5 nothing collected. Only 1 is a gate doing its job.
        #
        # The first version of this treated any non-zero exit as "caught", and
        # duly reported CAUGHT for a mutation whose gate pointed at a test file
        # that does not exist -- pytest exited 5, "no tests ran". An audit of
        # whether checks can fail is worth nothing if it credits itself for a
        # missing check.
        if gate.returncode == 1:
            verdict = "CAUGHT"
        elif gate.returncode == 0:
            verdict = "MISSED"
        else:
            verdict = "BROKEN"
            detail = f"pytest exit {gate.returncode}: {detail}"
        return {"name": mutation.name, "verdict": verdict, "detail": detail}
    finally:
        target.write_text(source, encoding="utf-8")
        if mutation.needs_build and rebuild:
            _build()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="include mutations that need a Rust rebuild (slow: ~2 min each)",
    )
    args = parser.parse_args(argv)

    chosen = [m for m in MUTATIONS if not args.only or m.name in args.only]
    if args.list:
        for mutation in chosen:
            build = " [rust]" if mutation.needs_build else ""
            print(f"{mutation.name}{build}: {mutation.why}")
        return 0

    paths = sorted({m.path for m in chosen})
    if dirty := _dirty(paths):
        print("refusing to run: these files have uncommitted changes, and the")
        print("audit restores by checkout, which would discard them:")
        for line in dirty:
            print(" ", line)
        return 2

    results = []
    for mutation in chosen:
        print(f"-- {mutation.name} ...", flush=True)
        result = audit(mutation, rebuild=args.rebuild)
        results.append(result)
        print(f"   {result['verdict']}: {result['detail']}", flush=True)

    print()
    missed = [r for r in results if r["verdict"] == "MISSED"]
    broken = [r for r in results if r["verdict"] == "BROKEN"]
    for result in results:
        print(f"{result['verdict']:>7}  {result['name']}")
    print()
    if missed:
        print(f"{len(missed)} gate(s) did not notice their mutation:")
        for result in missed:
            mutation = next(m for m in MUTATIONS if m.name == result["name"])
            print(f"  {result['name']}: {mutation.why}")
    if broken:
        print()
        print(
            f"{len(broken)} mutation(s) could not be judged -- the gate did not "
            "run at all. Fix these before reading the result:"
        )
        for result in broken:
            print(f"  {result['name']}: {result['detail']}")
    if leftover := _dirty(paths):
        print("\nWARNING: files still modified after restore:")
        for line in leftover:
            print(" ", line)
        return 3
    return 1 if (missed or broken) else 0


if __name__ == "__main__":
    raise SystemExit(main())
