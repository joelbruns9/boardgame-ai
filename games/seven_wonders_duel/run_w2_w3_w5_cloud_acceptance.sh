#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 CHECKPOINT RUN_DIR ACCEPTANCE_DIR [ITERATION_SECONDS]" >&2
  exit 2
fi

checkpoint="$1"
run_dir="$2"
acceptance_dir="$3"
iteration_seconds="${4:-0}"

mkdir -p "$acceptance_dir"

python -m games.seven_wonders_duel.w5_gate_bench \
  --checkpoint "$checkpoint" \
  --work-dir "$acceptance_dir/gate_work" \
  --output "$acceptance_dir/gate_cost_L_bf16.json" \
  --sizes 200 400 800 \
  --precision bf16 \
  --sims 64 \
  --slots 48 \
  --global-batch-cap 256 \
  --max-inflight-batches 2 \
  --scheduler-workers 1 \
  --iteration-seconds "$iteration_seconds"

if [[ -f "$run_dir/run_manifest.json" ]]; then
  python tools/az_report.py "$run_dir" \
    --output-prefix "$acceptance_dir/az_report"
  completed="$(
    python -c 'import sys; from pathlib import Path; from tools.az_report import load_rows; print(len(load_rows(Path(sys.argv[1]))))' \
      "$run_dir"
  )"
  if (( completed >= 60 )); then
    python tools/validate_az_memory_stability.py "$run_dir" \
      --tail 30 \
      --minimum-iterations 60 \
      --max-drift 0.05 \
      --output "$acceptance_dir/memory_stability_L_bf16.json"
  else
    echo "memory stability deferred: $completed/60 iterations complete"
  fi
fi
