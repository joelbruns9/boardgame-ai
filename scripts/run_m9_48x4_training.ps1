param(
    [int]$Iterations = 80,
    [string]$CheckpointDir = "checkpoints_m9_48x4_s800",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$Python = "C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe"
$Repo = "C:\Users\joeld\projects\boardgame-ai"
$LogDir = Join-Path $Repo "logs"
$LogPath = Join-Path $LogDir "m9_48x4_s800.log"
$CheckpointPath = Join-Path $Repo $CheckpointDir

Set-Location $Repo
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if ((Test-Path $CheckpointPath) -and -not $Force) {
    $existing = Get-ChildItem -Path $CheckpointPath -Filter "iter_*.pt" -ErrorAction SilentlyContinue
    if ($existing.Count -gt 0) {
        Write-Host "Checkpoint directory already contains $($existing.Count) checkpoint(s): $CheckpointDir"
        Write-Host "Use a new -CheckpointDir, or rerun with -Force to allow overwriting iter_*.pt files."
        exit 1
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Starting M9 48x4 training"
Write-Host "Iterations: $Iterations"
Write-Host "Checkpoint dir: $CheckpointDir"
Write-Host "Log: $LogPath"
Write-Host "============================================================"

& $Python -m games.kingdomino.self_play `
    --device cuda `
    --engine batched `
    --channels 48 `
    --blocks 4 `
    --bilinear_dim 64 `
    --sims 800 `
    --batch_slots 86 `
    --leaf_batch 6 `
    --amp_inference `
    --games_per_iter 192 `
    --iterations $Iterations `
    --train_steps 200 `
    --batch_size 256 `
    --lr 0.001 `
    --buffer 50000 `
    --checkpoint_dir $CheckpointDir 2>&1 | Tee-Object -FilePath $LogPath

if ($LASTEXITCODE -ne 0) {
    throw "M9 48x4 training failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "M9 48x4 training completed."
