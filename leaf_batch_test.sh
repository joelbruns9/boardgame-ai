#!/usr/bin/env bash
# =============================================================================
# leaf_batch_test.sh — run the leaf-batch A/B on a box that is still training.
#
# The A/B measures a WIN RATE, not wall time. Both arms play inside the same
# game, on the same GPU, at the same moment, so contention slows the run without
# biasing it. That is why this may run beside training, where the throughput
# sweep may not.
#
# The obstacle is the extension, not the GPU. The harness needs Rust built from
# newer code, and `maturin develop` installs into the SHARED site-packages --
# replacing the .so the live training process loads, and leaving the run's
# engine silently out of step with the commit its manifest records. Phase D's
# resume guard hashes the REPO, not the extension, so it would not catch that.
#
# So this never installs into the shared environment. It builds a wheel and
# unpacks it into its own directory, and runs the A/B with PYTHONPATH pointed
# there. The training venv is untouched; verified by printing which copy each
# side resolves.
#
# Usage (training may be running):
#   curl -fsSL https://raw.githubusercontent.com/joelbruns9/boardgame-ai/main/leaf_batch_test.sh -o leaf_batch_test.sh
#   LEAF_BATCHES=8 GAMES=400 bash leaf_batch_test.sh
#
# Knobs:
#   RUN_DIR=~/boardgame-ai/runs/seven_wonders_duel/cloud   run to read config from
#   SWEEP_REPO=~/sweep-checkout    checkout holding the newer code
#   SWEEP_REF=main
#   EXT_DIR=~/swr_leafbatch        isolated extension (never the shared venv)
#   CHECKPOINT=<path>              default: the run's current_best.pt
#   LEAF_BATCHES=8                 comma separated, each tested against 1
#   GAMES=400                      paired games per value; half seat-swapped
#   OUTPUT=~/leaf_batch_ab         results directory
# =============================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/joelbruns9/boardgame-ai.git}"
RUN_REPO="${RUN_REPO:-$HOME/boardgame-ai}"
RUN_DIR="${RUN_DIR:-$RUN_REPO/runs/seven_wonders_duel/cloud}"
SWEEP_REPO="${SWEEP_REPO:-$HOME/sweep-checkout}"
SWEEP_REF="${SWEEP_REF:-main}"
EXT_DIR="${EXT_DIR:-$HOME/swr_leafbatch}"
LEAF_BATCHES="${LEAF_BATCHES:-8}"
GAMES="${GAMES:-400}"
OUTPUT="${OUTPUT:-$HOME/leaf_batch_ab}"
PRECISION="${PRECISION:-bf16}"

if [ -f "$RUN_REPO/setup_cloud_common.sh" ]; then
  # shellcheck source=setup_cloud_common.sh
  source "$RUN_REPO/setup_cloud_common.sh"
else
  log()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
  ok()   { printf '\033[1;32m[OK]\033[0m %s\n' "$*"; }
  warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
  die()  { printf '\033[1;31m[FATAL]\033[0m %s\n' "$*" >&2; exit 1; }
  stage()      { printf '\n\033[1;36m=== STAGE %s: %s ===\033[0m\n' "$1" "$2"; }
  stage_done() { printf '\033[1;32m=== STAGE %s COMPLETE ===\033[0m\n' "$1"; }
fi

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python

mkdir -p "$OUTPUT"
LOG="$OUTPUT/leaf_batch_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
LEAF_BATCH_SCRIPT_VERSION=1
log "leaf_batch_test.sh version $LEAF_BATCH_SCRIPT_VERSION (log: $LOG)"

# ── STAGE 1: code ────────────────────────────────────────────────────────────
stage 1 "Checkout at $SWEEP_REPO"
if [ ! -d "$SWEEP_REPO/.git" ]; then
  git clone "$REPO_URL" "$SWEEP_REPO" || die "could not clone $REPO_URL"
fi
git -C "$SWEEP_REPO" fetch --all --quiet --tags || warn "fetch failed; using disk"
# Remote ref, then detach: `git checkout main` in an existing clone lands on the
# commit the clone was made at, however old. Fetch and checkout are not a pull.
if git -C "$SWEEP_REPO" rev-parse --verify --quiet "origin/$SWEEP_REF^{commit}" >/dev/null; then
  git -C "$SWEEP_REPO" checkout --quiet --detach "origin/$SWEEP_REF" || die "checkout failed"
else
  git -C "$SWEEP_REPO" checkout --quiet --detach "$SWEEP_REF" || die "checkout failed"
fi
ok "at $(git -C "$SWEEP_REPO" rev-parse --short=12 HEAD)"
[ -f "$SWEEP_REPO/games/seven_wonders_duel/leaf_batch_ab.py" ] \
  || die "this checkout has no leaf_batch_ab; SWEEP_REF=$SWEEP_REF is too old"
stage_done 1

# ── STAGE 2: an extension that is NOT the training process's ─────────────────
stage 2 "Isolated extension at $EXT_DIR"
SHARED_EXT="$("$PY" -c 'import seven_wonders_rust as s; print(s.__file__)' 2>/dev/null || true)"
log "training environment uses: ${SHARED_EXT:-<none importable>}"
case "$EXT_DIR" in
  *site-packages*|*dist-packages*)
    die "EXT_DIR points inside a site-packages tree. This must never install
  over the extension the training process has loaded." ;;
esac
rm -rf "$EXT_DIR"
( cd "$SWEEP_REPO/games/seven_wonders_duel/seven_wonders_rust" \
  && maturin build --release ) || die "wheel build failed"
WHEEL="$(ls -t "$SWEEP_REPO/games/seven_wonders_duel/seven_wonders_rust/target/wheels/"*.whl | head -1)"
[ -n "$WHEEL" ] || die "no wheel produced"
"$PY" -m pip install --quiet --target "$EXT_DIR" "$WHEEL" || die "wheel install failed"

ISOLATED="$(PYTHONPATH="$EXT_DIR" "$PY" -c 'import seven_wonders_rust as s; print(s.__file__)')"
log "the A/B will use:          $ISOLATED"
case "$ISOLATED" in
  "$EXT_DIR"*) ok "isolated extension resolves first" ;;
  *) die "PYTHONPATH did not take effect; refusing to run against the shared extension" ;;
esac
if [ -n "$SHARED_EXT" ]; then
  STILL="$("$PY" -c 'import seven_wonders_rust as s; print(s.__file__)')"
  [ "$STILL" = "$SHARED_EXT" ] || die "the shared extension MOVED; training may be affected"
  ok "shared extension unchanged: training is unaffected"
fi
stage_done 2

# ── STAGE 3: the A/B ─────────────────────────────────────────────────────────
stage 3 "Leaf-batch A/B"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/checkpoints/current_best.pt}"
[ -f "$CHECKPOINT" ] || die "no checkpoint at $CHECKPOINT"
# Copied, because the live run rewrites current_best.pt on every promotion and a
# mid-run reload would compare arms measured on two different networks.
cp "$CHECKPOINT" "$OUTPUT/ab_checkpoint.pt"
log "leaf batches: $LEAF_BATCHES   games per value: $GAMES"
log "(this shares the GPU with training: slower, but a win rate is not a timing)"
cd "$SWEEP_REPO"
PYTHONPATH="$EXT_DIR" "$PY" -m games.seven_wonders_duel.leaf_batch_ab \
  --checkpoint "$OUTPUT/ab_checkpoint.pt" \
  --config-from-manifest "$RUN_DIR/run_manifest.json" \
  --leaf-batches "$LEAF_BATCHES" \
  --games "$GAMES" \
  --precision "$PRECISION" \
  --out "$OUTPUT/leaf_batch_ab.json" \
  || die "the A/B did not complete; nothing was measured"
stage_done 3

cat <<EOF

Nothing was launched, nothing was stopped, and the training environment's
extension was not replaced. Results: $OUTPUT/leaf_batch_ab.json

Read the interval, not the point estimate. A result that contains 0.500 means
this many games could not tell -- which is not the same as no effect.
EOF
