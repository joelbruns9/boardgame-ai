param(
    [int]$Iterations = 60,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$Python = "C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe"
$Repo = "C:\Users\joeld\projects\boardgame-ai"
$LogDir = Join-Path $Repo "logs"
$CheckpointDir = "checkpoints_m9_warm_currentbest_32x4_s1600_b100k_t300_i$Iterations"
$WarmStart = "checkpoints_best/kingdomino_current_best.pt"
$LogName = "m9_warm_currentbest_32x4_s1600_b100k_t300_i$Iterations.log"
$LogPath = Join-Path $LogDir $LogName

Set-Location $Repo
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$checkpointPath = Join-Path $Repo $CheckpointDir
if ((Test-Path $checkpointPath) -and -not $Force) {
    $existing = Get-ChildItem -Path $checkpointPath -Filter "iter_*.pt" -ErrorAction SilentlyContinue
    if ($existing.Count -gt 0) {
        throw "Checkpoint directory already contains $($existing.Count) checkpoint(s): $CheckpointDir. Use a new directory or pass -Force."
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Starting M9 warm-start higher-sims run"
Write-Host "Warm start: $WarmStart"
Write-Host "Model: 32x4, bilinear_dim=64"
Write-Host "Sims: 1600"
Write-Host "Buffer: 100000"
Write-Host "Train steps: 300"
Write-Host "Iterations: $Iterations"
Write-Host "Checkpoint dir: $CheckpointDir"
Write-Host "Log: $LogPath"
Write-Host "============================================================"

& $Python -m games.kingdomino.self_play `
    --device cuda `
    --engine batched `
    --channels 32 `
    --blocks 4 `
    --bilinear_dim 64 `
    --sims 1600 `
    --batch_slots 86 `
    --leaf_batch 6 `
    --amp_inference `
    --games_per_iter 192 `
    --iterations $Iterations `
    --train_steps 300 `
    --batch_size 256 `
    --lr 0.001 `
    --buffer 100000 `
    --warm_start $WarmStart `
    --checkpoint_dir $CheckpointDir 2>&1 | Tee-Object -FilePath $LogPath

if ($LASTEXITCODE -ne 0) {
    throw "Warm-start higher-sims run failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Completed M9 warm-start higher-sims run."
Write-Host "Checkpoint dir: $CheckpointDir"
Write-Host "Log: $LogPath"
