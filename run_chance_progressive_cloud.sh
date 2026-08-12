#!/usr/bin/env bash
# Launch one frozen phase detached. The Python runner performs the final source,
# checkpoint, prerequisite, and no-overwrite checks before self-play starts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PHASE="${1:-}"
case "$PHASE" in
  g4|phase_a|phase_b) ;;
  *) echo "usage: $0 {g4|phase_a|phase_b}" >&2; exit 2 ;;
esac

EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
BATCH_SLOTS="${BATCH_SLOTS:-}"
GAME_CPUS="${GAME_CPUS:-}"
PYTHON="${PYTHON:-python3}"
RUN_ROOT="${RUN_ROOT:-runs/kingdomino/chance_progressive_cloud_v1}"

[ -n "$EXPECTED_COMMIT" ] || { echo "EXPECTED_COMMIT is required" >&2; exit 2; }
[ -n "$BATCH_SLOTS" ] || { echo "BATCH_SLOTS is required (use calibration result)" >&2; exit 2; }
[ -n "$GAME_CPUS" ] || { echo "GAME_CPUS is required (use calibration result)" >&2; exit 2; }

# shellcheck source=setup_cloud_common.sh
source "$SCRIPT_DIR/setup_cloud_common.sh"
LOG_FILE="$RUN_ROOT/$PHASE/cloud_process.log"
PID_FILE="$RUN_ROOT/$PHASE/cloud_process.pid"

common::launch_detached "$LOG_FILE" \
  "$PYTHON" -m games.kingdomino.chance_progressive_cloud run \
  --phase "$PHASE" --run-root "$RUN_ROOT" \
  --batch-slots "$BATCH_SLOTS" --game-cpus "$GAME_CPUS" \
  --expected-commit "$EXPECTED_COMMIT" --execute
printf '%s\n' "$LAUNCHED_PID" > "$PID_FILE"
ok "PID recorded in $PID_FILE"
