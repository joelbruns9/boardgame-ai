#!/usr/bin/env bash
# Prepare (but never start) the frozen Kingdomino chance-progressive cloud run.
#
# Usage on a fresh paid box:
#   REPO_REF=codex/kingdomino-chance-correct EXPECTED_COMMIT=<40-char-sha> \
#     bash setup_chance_progressive_cloud.sh
#
# The base checkpoint is intentionally Git-ignored. Copy it to the documented
# repository-relative path before running this script; its SHA is verified.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/joelbruns9/boardgame-ai.git}"
REPO_DIR="${REPO_DIR:-$HOME/boardgame-ai}"
REPO_REF="${REPO_REF:-codex/kingdomino-chance-correct}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
PYTHON="${PYTHON:-python3}"

# shellcheck source=setup_cloud_common.sh
source "$(dirname "$0")/setup_cloud_common.sh"

[ -n "$EXPECTED_COMMIT" ] \
  || die "EXPECTED_COMMIT is required: paid runs must use an exact reviewed commit."

stage 1 "Toolchains and source checkout"
common::require_python
common::rust_toolchain
common::clone_repo \
  games/kingdomino/chance_progressive_cloud.py \
  games/kingdomino/configs/chance_progressive_cloud_v1.json
git fetch origin "$REPO_REF"
git checkout --detach "$EXPECTED_COMMIT"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
[ "$ACTUAL_COMMIT" = "$EXPECTED_COMMIT" ] \
  || die "source commit $ACTUAL_COMMIT != EXPECTED_COMMIT $EXPECTED_COMMIT"
git merge-base --is-ancestor a16fed8 "$ACTUAL_COMMIT" \
  || die "source does not contain the reviewed G3 implementation commit a16fed8"
[ -z "$(git status --porcelain --untracked-files=no)" ] \
  || die "tracked source is dirty; refusing ambiguous cloud provenance"
ok "Exact source commit: $ACTUAL_COMMIT"
stage_done 1

stage 2 "Pinned dependencies, release build, and GPU gate"
common::python_deps
common::build_crate games/kingdomino/kingdomino_rust kingdomino_rust
common::gpu_gate
stage_done 2

stage 3 "Frozen inputs and focused verification"
BASE_CKPT="runs/kingdomino/best_checkpoint/current_best.pt"
[ -f "$BASE_CKPT" ] \
  || die "Missing Git-ignored $BASE_CKPT; transfer it separately, then rerun setup."
EXPECTED_CKPT_SHA="4bf07b0ca14e5452e6533a9232967e89bb0ab0df88c99e9928a65f402b1f04b3"
ACTUAL_CKPT_SHA="$(sha256sum "$BASE_CKPT" | awk '{print $1}')"
[ "$ACTUAL_CKPT_SHA" = "$EXPECTED_CKPT_SHA" ] \
  || die "base checkpoint SHA mismatch: $ACTUAL_CKPT_SHA"
"$PY" -m games.kingdomino.chance_progressive_cloud validate-config
"$PY" -m pytest -q \
  games/kingdomino/test_chance_progressive_cloud.py \
  games/kingdomino/test_milestone6_promotion.py \
  games/kingdomino/tests/test_chance_aware_training.py \
  games/kingdomino/tests/test_chance_progressive_g3.py
ok "Pinned checkpoint and focused suites verified."
stage_done 3

stage 4 "Calibration command and launch dry runs"
mkdir -p runs/kingdomino/chance_progressive_cloud_v1/calibration
"$PY" -m games.kingdomino.cloud_calibration \
  --preset bootstrap \
  --out runs/kingdomino/chance_progressive_cloud_v1/calibration \
  --channels 80 --primary_channels 80 --blocks 6 \
  --sims 200 --selfplay_games 12

# These values are placeholders for syntax/provenance validation only. Replace
# them with the calibration recommendation when launching G4.
for phase in g4 phase_a phase_b; do
  "$PY" -m games.kingdomino.chance_progressive_cloud run \
    --phase "$phase" \
    --run-root runs/kingdomino/chance_progressive_cloud_v1 \
    --batch-slots 96 --game-cpus 12
done
stage_done 4

cat <<EOF

Cloud preparation passed. No training was started.

Source commit:     $ACTUAL_COMMIT
Base checkpoint:  $ACTUAL_CKPT_SHA
Calibration:      runs/kingdomino/chance_progressive_cloud_v1/calibration
Next action:      review calibration, then launch G4 exactly as documented in
                  games/kingdomino/CHANCE_PROGRESSIVE_CLOUD_RUN.md
EOF
