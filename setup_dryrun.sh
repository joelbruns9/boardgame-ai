#!/usr/bin/env bash
# =============================================================================
# Dry-run harness for setup_cloud_7wd.sh — runs on a laptop, no Linux required.
#
# The first rental burned hours on setup-script bugs that were all of one class:
# ordering, quoting, unset variables, and flags that did not exist. None of them
# needed a GPU, CUDA, apt or a real network to surface — only for the script's
# *control flow* to actually execute.
#
# So execute it, with every external command replaced by a logging stub on PATH.
# `command -v` finds the stubs, `set -u` still fires on unset variables, every
# conditional and every flag-assembly line runs for real, and the recorded call
# log shows exactly what would have been invoked, in order, with what arguments.
#
# WHAT THIS CATCHES: unset/misspelled variables, stage ordering, quoting bugs,
# flags assembled wrongly or dropped, `set -e` aborts, missing files the script
# expects, and the launch command line the run would actually get.
#
# WHAT IT CANNOT CATCH: anything about the real environment — apt package names,
# CUDA/driver behaviour, cgroup contents, and whether mimalloc and rayon compile
# on Linux. Those need WSL, Docker or the box itself.
#
#   bash setup_dryrun.sh              # run it
#   bash setup_dryrun.sh --show-log   # and print every stubbed invocation
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHOW_LOG=0
[ "${1:-}" = "--show-log" ] && SHOW_LOG=1

STUB_DIR="$(mktemp -d)"
CALL_LOG="$STUB_DIR/calls.log"
: > "$CALL_LOG"
trap 'rm -rf "$STUB_DIR"' EXIT

# Every stub logs its full argv and succeeds. Where the script parses output
# rather than just checking status, the stub emits something plausible --
# a version string the comparison can sort, not a placeholder.
stub() {
  local name="$1" body="${2:-}"
  cat > "$STUB_DIR/$name" <<EOF
#!/usr/bin/env bash
printf '%s' "$name" >> "$CALL_LOG"
printf ' %s' "\$@" >> "$CALL_LOG"
printf '\n' >> "$CALL_LOG"
$body
exit 0
EOF
  chmod +x "$STUB_DIR/$name"
}

stub rustup
stub cargo   'case "${1:-}" in --version) echo "cargo 1.85.0 (stub)";; esac'
stub rustc   'case "${1:-}" in --version) echo "rustc 1.85.0 (stub)";; esac'
stub curl
stub git
stub pip
stub maturin
stub apt-get
stub nvidia-smi 'echo "stub nvidia-smi"'
stub cc      'case "${1:-}" in --version) echo "cc (stub) 13.2.0";; esac'
stub gcc     'case "${1:-}" in --version) echo "gcc (stub) 13.2.0";; esac'
stub nproc   'echo 16'
# `python` must succeed for import checks, `-m` module runs and heredocs. It is
# deliberately inert: this harness tests the *shell*, and running the real
# preflight would need torch and a GPU.
stub python
stub python3

export PATH="$STUB_DIR:$PATH"

# Point the script at this checkout so its file-existence guards see real files.
export REPO_DIR="$SCRIPT_DIR"
export PY=python
# Skip the sweeps: they are a separate two-pass flow with their own harness.
export SKIP_SWEEPS=1

echo "== dry run: setup_cloud_7wd.sh with stubbed externals =="
set +e
bash "$SCRIPT_DIR/setup_cloud_7wd.sh" > "$STUB_DIR/stdout.log" 2>&1
STATUS=$?
set -e

FAILED=0
fail() { echo "FAIL: $*"; FAILED=1; }

# Reaching STAGE 10 is the bar, not exit 0: the stubbed `python` exits at once,
# so the launch liveness check correctly reports a dead process. Everything
# before that is the shell logic this harness exists to exercise.
grep -q "STAGE 10" "$STUB_DIR/stdout.log" || fail "never reached the launch stage"
for n in 1 2 3 4 5 6 7 9; do
  grep -q "STAGE $n COMPLETE" "$STUB_DIR/stdout.log" || fail "stage $n did not complete"
done

# `set -u` turns an unset variable into an error rather than an empty expansion,
# which is the bug class that cost hours on the first rental.
grep -qE "unbound variable|parameter (not set|null)" "$STUB_DIR/stdout.log"   && fail "unset variable: $(grep -oE '[A-Za-z_]+: unbound variable' "$STUB_DIR/stdout.log" | head -1)"

# A literal ${...} on a command line means a quoting or escaping bug.
grep -qE '\$\{[A-Za-z_]' "$CALL_LOG" && fail "unexpanded variable reached a command line"

grep -q "C toolchain" "$STUB_DIR/stdout.log"   || fail "C toolchain never checked (mimalloc needs cc)"

# The launch line is the payload. Every flag below has a recorded defect behind
# it, so absence is a regression, not a style question.
LAUNCH=$(grep -E "phase_d --run-dir runs/seven_wonders_duel/cloud" "$CALL_LOG" | tail -1)
if [ -z "$LAUNCH" ]; then
  fail "no phase_d launch command was assembled"
else
  # A flag the launcher does not pass is a default nobody chose: --train-steps
  # was omitted once and silently took the parser's 300, ~8x the intended reuse.
  case "$LAUNCH" in
    *--train-steps*) ;;
    *) fail "--train-steps absent; phase_d would take the parser default" ;;
  esac
  # Default is strict_gate, under which a never-deciding gate freezes self-play
  # at iteration-0 weights for the whole run.
  case "$LAUNCH" in
    *"--selfplay-generator-mode soft_gate"*) ;;
    *) fail "--selfplay-generator-mode soft_gate absent" ;;
  esac
  case "$LAUNCH" in
    *"--generation-backend rust"*) ;;
    *) fail "--generation-backend rust absent" ;;
  esac
  case "$LAUNCH" in
    *"--gate-backend rust"*) ;;
    *) fail "--gate-backend rust absent" ;;
  esac
  # --heads must divide --d-model, or the model cannot be built at all.
  d_model=$(printf '%s' "$LAUNCH" | grep -oE -- "--d-model [0-9]+" | awk '{print $2}')
  heads=$(printf '%s' "$LAUNCH" | grep -oE -- "--heads [0-9]+" | awk '{print $2}')
  if [ -n "$d_model" ] && [ -n "$heads" ] && [ "$((d_model % heads))" -ne 0 ]; then
    fail "--heads $heads does not divide --d-model $d_model"
  fi
  echo "OK: launch command assembled with every guarded flag"
fi

echo
echo "stubbed invocations: $(wc -l < "$CALL_LOG")"
if [ "$SHOW_LOG" -eq 1 ]; then
  echo "--- call log ---"
  cat "$CALL_LOG"
  echo "--- stdout ---"
  cat "$STUB_DIR/stdout.log"
fi
echo
echo "--- launch command that would run ---"
printf '%s
' "${LAUNCH:-<none assembled>}" | tr ' ' '
' | paste -sd' ' - | fold -s -w 100

[ "$FAILED" -eq 0 ] && echo "DRY RUN OK" || echo "DRY RUN FOUND PROBLEMS"
exit "$FAILED"
