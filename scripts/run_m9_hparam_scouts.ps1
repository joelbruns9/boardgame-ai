param(
    [int]$Iterations = 40,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$Python = "C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe"
$Repo = "C:\Users\joeld\projects\boardgame-ai"
$LogDir = Join-Path $Repo "logs"

Set-Location $Repo
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Assert-CheckpointDirAvailable {
    param(
        [Parameter(Mandatory=$true)][string]$CheckpointDir
    )

    $checkpointPath = Join-Path $Repo $CheckpointDir
    if ((Test-Path $checkpointPath) -and -not $Force) {
        $existing = Get-ChildItem -Path $checkpointPath -Filter "iter_*.pt" -ErrorAction SilentlyContinue
        if ($existing.Count -gt 0) {
            throw "Checkpoint directory already contains $($existing.Count) checkpoint(s): $CheckpointDir. Use a new directory or pass -Force."
        }
    }
}

function Invoke-ScoutRun {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][int]$Buffer,
        [Parameter(Mandatory=$true)][int]$TrainSteps,
        [Parameter(Mandatory=$true)][string]$CheckpointDir
    )

    Assert-CheckpointDirAvailable -CheckpointDir $CheckpointDir

    $logPath = Join-Path $LogDir "$Name.log"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Starting $Name"
    Write-Host "Iterations: $Iterations"
    Write-Host "Buffer: $Buffer"
    Write-Host "Train steps: $TrainSteps"
    Write-Host "Checkpoint dir: $CheckpointDir"
    Write-Host "Log: $logPath"
    Write-Host "============================================================"

    & $Python -m games.kingdomino.self_play `
        --device cuda `
        --engine batched `
        --channels 32 `
        --blocks 4 `
        --bilinear_dim 64 `
        --sims 800 `
        --batch_slots 86 `
        --leaf_batch 6 `
        --amp_inference `
        --games_per_iter 192 `
        --iterations $Iterations `
        --train_steps $TrainSteps `
        --batch_size 256 `
        --lr 0.001 `
        --buffer $Buffer `
        --checkpoint_dir $CheckpointDir 2>&1 | Tee-Object -FilePath $logPath

    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "Completed $Name"
}

Invoke-ScoutRun `
    -Name "m9_hparam_buffer100k_i40" `
    -Buffer 100000 `
    -TrainSteps 200 `
    -CheckpointDir "checkpoints_m9_hparam_buffer100k_i40"

Invoke-ScoutRun `
    -Name "m9_hparam_trainsteps400_i40" `
    -Buffer 50000 `
    -TrainSteps 400 `
    -CheckpointDir "checkpoints_m9_hparam_trainsteps400_i40"

Write-Host ""
Write-Host "All queued M9 hyperparameter scout runs completed."
