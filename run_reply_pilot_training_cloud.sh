#!/usr/bin/env bash
# Wait for the production-root freeze, generate/validate reply labels, and train
# the equal-step control/treatment pilot. This never updates current_best.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-/venv/main/bin/python}"
PILOT_DIR="${PILOT_DIR:-runs/kingdomino/reply_pilot/cloud}"
BASE_CKPT="${BASE_CKPT:-runs/kingdomino/best_checkpoint/current_best.pt}"
REPLAY_BUFFER="${REPLAY_BUFFER:-runs/kingdomino/cloud_80x6_run10/buffer_final.pkl}"
NUM_SHARDS="${PILOT_SHARDS:-4}"
RAYON_THREADS="${PILOT_THREADS:-4}"
EXPECTED_ROOTS="${EXPECTED_ROOTS:-500}"
MIN_ACCEPTED="${MIN_ACCEPTED:-800}"
MAX_ACCEPTED="${MAX_ACCEPTED:-2000}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"

cd "$REPO_DIR"

die() { printf '[FATAL] %s\n' "$*" >&2; exit 1; }
log() { printf '\n==> %s\n' "$*"; }
need_file() { [ -s "$1" ] || die "missing or empty artifact: $1"; }

command -v "$PYTHON" >/dev/null 2>&1 || die "Python is unavailable: $PYTHON"
need_file "$BASE_CKPT"
need_file "$REPLAY_BUFFER"
need_file "$PILOT_DIR/calibration_summary.json"
need_file "$PILOT_DIR/calibration_roots.jsonl"

LOCK_DIR="$PILOT_DIR/.production_to_training.lock"
mkdir "$LOCK_DIR" 2>/dev/null \
  || die "another production-to-training runner is active (or left $LOCK_DIR)"
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

ROOTS="$PILOT_DIR/training_roots.jsonl"
ROOTS_MANIFEST="$PILOT_DIR/training_roots.manifest.json"
FREEZE_PID_FILE="$PILOT_DIR/training_freeze.pid"

if [ ! -s "$ROOTS_MANIFEST" ]; then
  need_file "$FREEZE_PID_FILE"
  FREEZE_PID="$(tr -d '[:space:]' < "$FREEZE_PID_FILE")"
  [ -n "$FREEZE_PID" ] || die "empty freeze PID file: $FREEZE_PID_FILE"
  log "Waiting for root freeze PID $FREEZE_PID"
  while [ ! -s "$ROOTS_MANIFEST" ]; do
    kill -0 "$FREEZE_PID" 2>/dev/null \
      || die "root freeze exited without producing $ROOTS_MANIFEST"
    sleep "$WAIT_SECONDS"
  done
fi

need_file "$ROOTS"
"$PYTHON" - "$ROOTS_MANIFEST" "$EXPECTED_ROOTS" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
expected = int(sys.argv[2])
if manifest.get("positions") != expected:
    raise SystemExit(f"expected {expected} roots, found {manifest.get('positions')}")
keys = manifest.get("state_keys", [])
if len(keys) != expected or len(set(keys)) != expected:
    raise SystemExit("training-root manifest is missing unique state keys")
if manifest.get("eligible_opponent_reply_roots") != expected:
    raise SystemExit("training-root eligibility count failed")
if manifest.get("held_out_state_keys", 0) < 1:
    raise SystemExit("training roots were not generated with a held-out set")
print(json.dumps({
    "positions": expected,
    "held_out_state_keys": manifest["held_out_state_keys"],
    "held_out_candidates_skipped": manifest.get("held_out_candidates_skipped", 0),
    "duplicate_candidates_skipped": manifest.get("duplicate_candidates_skipped", 0),
}, sort_keys=True))
PY

readarray -t FILTER < <("$PYTHON" - "$PILOT_DIR/calibration_summary.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))["proposed_filter"]
for key in (
    "min_top_two_margin",
    "max_mc_standard_error",
    "max_target_entropy",
    "max_searched_seed_sd",
    "min_top_pick_agreement",
):
    print(p[key])
PY
)
[ "${#FILTER[@]}" -eq 5 ] || die "calibration filter is incomplete"
MIN_MARGIN="${FILTER[0]}"
MAX_STDERR="${FILTER[1]}"
MAX_ENTROPY="${FILTER[2]}"
MAX_SEED_SD="${FILTER[3]}"
MIN_TOP_AGREE="${FILTER[4]}"

log "Generating or resuming $NUM_SHARDS production shards"
for ((i = 0; i < NUM_SHARDS; i++)); do
  "$PYTHON" -m games.kingdomino.reply_pilot \
    --mode generate --engine rust --rayon-threads "$RAYON_THREADS" \
    --checkpoint "$BASE_CKPT" \
    --positions-path "$ROOTS" \
    --shards-dir "$PILOT_DIR/production_shards" \
    --num-shards "$NUM_SHARDS" --shard-index "$i" \
    --pick-plies 8 --chance-k 16 --search-sims 3200 \
    --min-top-two-margin "$MIN_MARGIN" \
    --max-mc-standard-error "$MAX_STDERR" \
    --max-target-entropy "$MAX_ENTROPY" \
    --max-searched-seed-sd "$MAX_SEED_SD" \
    --min-top-pick-agreement "$MIN_TOP_AGREE" --reject-ties
done

log "Merging and structurally validating accepted reply labels"
"$PYTHON" -m games.kingdomino.reply_pilot \
  --mode merge --shards-dir "$PILOT_DIR/production_shards" \
  --accepted-only --output "$PILOT_DIR/reply_labels.jsonl"
"$PYTHON" -m games.kingdomino.reply_pilot \
  --mode validate --output "$PILOT_DIR/reply_labels.jsonl"

ACCEPTED="$(awk 'NF { count += 1 } END { print count + 0 }' \
  "$PILOT_DIR/reply_labels.jsonl")"
if [ "$ACCEPTED" -lt "$MIN_ACCEPTED" ] || [ "$ACCEPTED" -gt "$MAX_ACCEPTED" ]; then
  die "accepted-label gate failed: $ACCEPTED is outside [$MIN_ACCEPTED, $MAX_ACCEPTED]"
fi
printf '[OK] accepted-label gate: %s examples\n' "$ACCEPTED"

log "Creating a root-disjoint 80/20 reply split"
"$PYTHON" -m games.kingdomino.reply_pilot \
  --mode split --input "$PILOT_DIR/reply_labels.jsonl" \
  --validation-fraction 0.20 --split-seed 20260719 \
  --train-output "$PILOT_DIR/reply_train.jsonl" \
  --validation-output "$PILOT_DIR/reply_validation.jsonl"

TRAINING_REPORT="$PILOT_DIR/training/pilot_training_report.json"
if [ -e "$TRAINING_REPORT" ] && [ "${FORCE_TRAIN:-0}" != "1" ]; then
  die "training report already exists; set FORCE_TRAIN=1 only to intentionally replace it"
fi

log "Training equal-step control and treatment arms"
"$PYTHON" -m games.kingdomino.reply_training \
  --checkpoint "$BASE_CKPT" \
  --reply-train "$PILOT_DIR/reply_train.jsonl" \
  --reply-validation "$PILOT_DIR/reply_validation.jsonl" \
  --replay-buffer "$REPLAY_BUFFER" \
  --output-dir "$PILOT_DIR/training" \
  --device cuda --steps 1000 --batch-size 256 \
  --reply-fraction 0.15 --lambda-reply 0.15 \
  --validation-batch-size 256 --buffer-capacity 200000 \
  --sample-workers 4 --lr 1e-4 --weight-decay 1e-4

"$PYTHON" - "$TRAINING_REPORT" <<'PY'
import json, pathlib, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "trained_not_promoted":
    raise SystemExit(f"unexpected training status: {report.get('status')}")
if report.get("current_best_updated") is not False:
    raise SystemExit("current_best safety assertion failed")
for key in ("control_checkpoint", "treatment_checkpoint"):
    if not pathlib.Path(report[key]).is_file():
        raise SystemExit(f"missing trained checkpoint: {report[key]}")
print(json.dumps({
    "status": report["status"],
    "reply_train_examples": report["reply_train_examples"],
    "reply_validation_examples": report["reply_validation_examples"],
    "elapsed_seconds": report["elapsed_seconds"],
    "current_best_updated": report["current_best_updated"],
}, indent=2, sort_keys=True))
PY

printf '\n[OK] Dataset gates passed and NN pilot training completed.\n'
printf 'Review before search evaluation: %s\n' "$TRAINING_REPORT"
