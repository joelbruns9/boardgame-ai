$ErrorActionPreference = "Stop"

$Python = "C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe"
$Repo = "C:\Users\joeld\projects\boardgame-ai"
$LogDir = Join-Path $Repo "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Set-Location $Repo

function Invoke-TrainingRun {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][int]$Channels,
        [Parameter(Mandatory=$true)][int]$Blocks,
        [Parameter(Mandatory=$true)][string]$CheckpointDir
    )

    $LogPath = Join-Path $LogDir "$Name.log"
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Starting $Name"
    Write-Host "Log: $LogPath"
    Write-Host "============================================================"

    & $Python -m games.kingdomino.self_play `
        --device cuda `
        --engine batched `
        --channels $Channels `
        --blocks $Blocks `
        --bilinear_dim 64 `
        --sims 800 `
        --batch_slots 86 `
        --leaf_batch 6 `
        --amp_inference `
        --games_per_iter 192 `
        --iterations 80 `
        --train_steps 200 `
        --batch_size 256 `
        --lr 0.001 `
        --buffer 50000 `
        --checkpoint_dir $CheckpointDir 2>&1 | Tee-Object -FilePath $LogPath

    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "Completed $Name"
}

Invoke-TrainingRun `
    -Name "m9_32x6_s800" `
    -Channels 32 `
    -Blocks 6 `
    -CheckpointDir "checkpoints_m9_32x6_s800"

Invoke-TrainingRun `
    -Name "m9_48x4_s800" `
    -Channels 48 `
    -Blocks 4 `
    -CheckpointDir "checkpoints_m9_48x4_s800"

Write-Host ""
Write-Host "All queued M9 training runs completed."
