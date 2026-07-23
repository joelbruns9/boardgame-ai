#!/usr/bin/env bash
set -euo pipefail

# Rust-only F4.6 production calibration. Quality-sensitive search fields come
# exclusively from f4_quality_lock_v2.json; this script sweeps scheduler geometry.
if [[ $# -ne 3 ]]; then
  echo "usage: $0 CHECKPOINT QUALITY_LOCK OUTPUT_DIR" >&2
  exit 2
fi

checkpoint=$1
quality_lock=$2
output=$3
python_bin=${PYTHON_BIN:-python}
device=${F4_DEVICE:-cuda}
sweep_games=${F4_SWEEP_GAMES:-32}
confirmation_games=${F4_CONFIRMATION_GAMES:-100}
confirmation_repetitions=${F4_CONFIRMATION_REPETITIONS:-5}
isolated_forward_rows=${F4_ISOLATED_FORWARD_ROWS_PER_SECOND:-0}

mkdir -p "$output"

run_rust() {
  local name=$1
  local slots=$2
  local cap=$3
  local inflight=$4
  local pinned=${5:-0}
  local compile_mode=${6:-none}
  local workers=${7:-1}
  local extra=()
  if [[ "$pinned" == "1" ]]; then extra+=(--pinned-memory); fi
  if [[ "$compile_mode" != "none" ]]; then extra+=(--torch-compile "$compile_mode"); fi
  "$python_bin" -m games.seven_wonders_duel.f4_throughput_bench \
    --mode rust \
    --checkpoint "$checkpoint" \
    --quality-lock "$quality_lock" \
    --output "$output/$name" \
    --device "$device" \
    --warmup-games 4 \
    --games "$sweep_games" \
    --repetitions 2 \
    --slots "$slots" \
    --global-batch-cap "$cap" \
    --max-inflight-batches "$inflight" \
    --scheduler-workers "$workers" \
    --isolated-forward-rows-per-second "$isolated_forward_rows" \
    --record-failures \
    "${extra[@]}"
}

# 1. Correctness/environment smoke.
run_rust smoke 4 32 1

# 2. Transformer/global-batch geometry.
for cap in 32 64 128 256 512; do
  slots=32
  if [[ "$cap" -ge 512 ]]; then slots=64; fi
  run_rust "geometry_cap_${cap}" "$slots" "$cap" 2
done

# 3. Concurrent slot x persistent coarse scheduler-worker geometry.
physical_cores=$(nproc)
half_cores=$(( physical_cores / 2 ))
minus_one=$(( physical_cores - 1 ))
if [[ "$half_cores" -lt 1 ]]; then half_cores=1; fi
if [[ "$minus_one" -lt 1 ]]; then minus_one=1; fi
for workers in 1 "$half_cores" "$minus_one"; do
  for slots in 32 64 128 256; do
    cap=256
    if [[ "$slots" -ge 64 ]]; then cap=512; fi
    if [[ "$slots" -ge 128 ]]; then cap=1024; fi
    run_rust "slots_${slots}_workers_${workers}" "$slots" "$cap" 2 0 none "$workers"
  done
done

# 4. Global cap x buffering depth.
for cap in 128 256 512 1024; do
  for inflight in 1 2 3; do
    slots=32
    if [[ "$cap" -ge 512 ]]; then slots=64; fi
    if [[ "$cap" -ge 1024 ]]; then slots=128; fi
    run_rust "buffer_cap_${cap}_flight_${inflight}" "$slots" "$cap" "$inflight"
  done
done

# 5. Transfer/compile options at the strongest baseline geometry. Token
# bucketing remains conditional on an observed material padding ratio.
run_rust feature_pinned 64 512 2 1 none
run_rust feature_compile_reduce 64 512 2 0 reduce-overhead
run_rust feature_pinned_compile_reduce 64 512 2 1 reduce-overhead

"$python_bin" -m games.seven_wonders_duel.f4_cloud_select \
  --root "$output" --output "$output/selected.json"

readarray -t winner < <("$python_bin" -c '
import json,sys
x=json.load(open(sys.argv[1], encoding="utf-8"))["winner"]["manifest"]
print(x["slots"]); print(x["global_batch_cap"]); print(x["max_inflight_batches"])
print(x["scheduler_workers"]); print(1 if x["pinned_memory"] else 0); print(x["torch_compile"])
' "$output/selected.json")

confirmation_extra=()
if [[ "${winner[4]}" == "1" ]]; then confirmation_extra+=(--pinned-memory); fi
if [[ "${winner[5]}" != "none" ]]; then
  confirmation_extra+=(--torch-compile "${winner[5]}")
fi

# 6. Long repeated confirmation of the selected geometry.
"$python_bin" -m games.seven_wonders_duel.f4_throughput_bench \
  --mode rust \
  --checkpoint "$checkpoint" \
  --quality-lock "$quality_lock" \
  --output "$output/confirmation" \
  --device "$device" \
  --warmup-games 16 \
  --games "$confirmation_games" \
  --repetitions "$confirmation_repetitions" \
  --slots "${winner[0]}" \
  --global-batch-cap "${winner[1]}" \
  --max-inflight-batches "${winner[2]}" \
  --scheduler-workers "${winner[3]}" \
  --isolated-forward-rows-per-second "$isolated_forward_rows" \
  "${confirmation_extra[@]}"

# Synchronized diagnostic is deliberately separate from the throughput result.
"$python_bin" -m games.seven_wonders_duel.f4_throughput_bench \
  --mode rust \
  --checkpoint "$checkpoint" \
  --quality-lock "$quality_lock" \
  --output "$output/diagnostic" \
  --device "$device" \
  --warmup-games 4 \
  --games 8 \
  --repetitions 1 \
  --slots "${winner[0]}" \
  --global-batch-cap "${winner[1]}" \
  --max-inflight-batches "${winner[2]}" \
  --scheduler-workers "${winner[3]}" \
  --isolated-forward-rows-per-second "$isolated_forward_rows" \
  --diagnostic-sync \
  "${confirmation_extra[@]}"

"$python_bin" -m games.seven_wonders_duel.f4_cloud_finalize \
  --selected "$output/selected.json" \
  --confirmation "$output/confirmation/summary.json" \
  --diagnostic "$output/diagnostic/summary.json" \
  --output "$output/production_config.json"
