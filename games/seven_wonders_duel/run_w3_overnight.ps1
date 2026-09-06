# Two-stage W3 overnight run.
#
#   Stage 1  offline A/B    -- can the network USE control? (~2.5 h, GPU)
#   Stage 2  reference pass -- the COMMON yardstick regret is measured against
#                              (~4 h, CPU, 4 shards)
#
# The two are independent: stage 2 measures the INCUMBENT, not the arms, so it
# does not wait on stage 1's checkpoints. It is the expensive, arm-independent
# half, and it is reusable -- every future arm and every future change is scored
# against the same reference without paying for it again.
#
# Scoring the arms is then minutes, tomorrow:
#
#   python -m games.seven_wonders_duel.w3_corpus_regret `
#     --reference-dir runs/seven_wonders_duel/threat_corpus/w3_reference `
#     --arm baseline=runs/seven_wonders_duel/w3_offline_ab/baseline_seed20260904.pt `
#     --arm inputs=runs/seven_wonders_duel/w3_offline_ab/inputs_seed20260904.pt `
#     --arm aux=runs/seven_wonders_duel/w3_offline_ab/aux_seed20260904.pt
#
# Stage 1 runs on the GPU and stage 2 on CPU shards, so they are launched
# together deliberately. If that contention worries you, run stage 1 first and
# start stage 2 when it finishes -- stage 2's result does not depend on it.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\..

$py = ".\.venv\Scripts\python.exe"
$buffers = "runs/seven_wonders_duel/cloud2/7wd_cloud_20260825T005745Z/buffers"
$incumbent = "extension_7wd/candidate_0085.pt"
$refDir = "runs/seven_wonders_duel/threat_corpus/w3_reference"
$live = "runs/seven_wonders_duel/threat_corpus/measured/triage/summary_merged.json"

if (-not (Test-Path $incumbent)) { throw "missing incumbent checkpoint: $incumbent" }
if (-not (Test-Path $buffers))   { throw "missing buffers: $buffers" }
if (-not (Test-Path $live))      { throw "missing live-position list: $live" }

New-Item -ItemType Directory -Force $refDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logs = "runs/seven_wonders_duel/w3_overnight_$stamp"
New-Item -ItemType Directory -Force $logs | Out-Null
"started $(Get-Date -Format o)" | Out-File -Encoding utf8 "$logs/START.txt"

# --- Stage 2 first (CPU shards, longest) -----------------------------------
# --all-episodes matters: without it the corpus falls back to the 17-position
# pre-triage sample, and --live-from then filters that to a handful. That is
# how an earlier run measured 7 positions instead of 60.
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
0..3 | ForEach-Object {
  Start-Process -NoNewWindow -FilePath $py -ArgumentList @(
    "-m", "games.seven_wonders_duel.threat_corpus_measure",
    "--all-episodes",
    "--live-from", $live,
    "--checkpoint", $incumbent,
    "--allow-migration",
    "--shard", "$_/4", "--limit", "12",
    "--out-dir", $refDir,
    "--summary-out", "$refDir/summary_shard$_.json"
  ) -RedirectStandardError "$logs/reference_shard$_.log" `
    -RedirectStandardOutput "$logs/reference_shard$_.out" | Out-Null
  "launched reference shard $_"
}

# --- Stage 1 (GPU) ----------------------------------------------------------
# Five seeds because one cannot resolve the 3-5 point effects this project's
# gates actually see. Deltas are paired per seed against the baseline that saw
# the same seed and split.
"launching stage 1 (offline A/B)"
& $py -m games.seven_wonders_duel.w3_offline_ab `
  --checkpoint $incumbent `
  --buffers $buffers `
  --arms baseline,inputs,aux,shuffled `
  --games 4000 --steps 400 --seeds 5 `
  --device cuda --precision bf16 `
  --save-dir runs/seven_wonders_duel/w3_offline_ab `
  --out runs/seven_wonders_duel/w3_offline_ab.json `
  *> "$logs/offline_ab.log"

"stage 1 finished $(Get-Date -Format o)"

# Wait for the reference shards before reporting.
while (Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
       Where-Object { $_.CommandLine -like '*threat_corpus_measure*' }) {
  Start-Sleep -Seconds 60
}
"reference pass finished $(Get-Date -Format o)"

$referenceSummaries = 0..3 | ForEach-Object {
  $path = "$refDir/summary_shard$_.json"
  if (-not (Test-Path $path)) { throw "missing reference summary: $path" }
  Get-Content $path -Raw | ConvertFrom-Json
}
if ($referenceSummaries.totals.failed | Where-Object { $_ -ne 0 }) {
  throw "one or more reference shards reported failed positions"
}
@(
  "reference measurement complete"
  $referenceSummaries | ForEach-Object {
    "shard $($_.params.shard): positions=$($_.totals.positions) ok=$($_.totals.ok) " +
      "failed=$($_.totals.failed) rechecked=$($_.totals.rechecked) " +
      "rank_changed=$($_.totals.rank_changed_on_recheck) " +
      "minutes=$($_.totals.wall_clock_minutes)"
  }
  "total positions=$(($referenceSummaries.totals.positions | Measure-Object -Sum).Sum)"
) | Set-Content -Encoding utf8 "$logs/reference_summary.txt"

"done $(Get-Date -Format o)" | Out-File -Append -Encoding utf8 "$logs/START.txt"
"logs in $logs"
