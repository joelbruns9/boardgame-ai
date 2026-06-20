param(
    [int]$EvalSims = 100,
    [int]$EvalGamesPerPair = 300,
    [int]$EvalBatchSlots = 86,
    [int]$EvalLeafBatch = 6,
    [double]$PromotionThreshold = 0.55,
    [int]$NextSims = 1600,
    [int]$NextIterations = 40,
    [switch]$SkipNextTraining
)

$ErrorActionPreference = "Stop"

$Python = "C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe"
$Repo = "C:\Users\joeld\projects\boardgame-ai"
$Manifest = "eval_manifests/m9_size_search_candidates.csv"
$LogDir = Join-Path $Repo "logs"
$EvalPairs = "eval_results/m9_size_search_s${EvalSims}_g${EvalGamesPerPair}_pairs.csv"
$EvalLeaderboard = "eval_results/m9_size_search_s${EvalSims}_g${EvalGamesPerPair}_leaderboard.csv"
$EvalGames = "eval_results/m9_size_search_s${EvalSims}_g${EvalGamesPerPair}_games.csv"
$EvalLog = Join-Path $LogDir "m9_size_search_s${EvalSims}_g${EvalGamesPerPair}.log"

Set-Location $Repo
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host ""
Write-Host "============================================================"
Write-Host "M9 size-search eval"
Write-Host "Manifest: $Manifest"
Write-Host "Pairs: $EvalPairs"
Write-Host "Log: $EvalLog"
Write-Host "============================================================"

& $Python -m games.kingdomino.round_robin_eval `
    --manifest $Manifest `
    --engine batched `
    --device cuda `
    --sims $EvalSims `
    --games_per_pair $EvalGamesPerPair `
    --batch_slots $EvalBatchSlots `
    --leaf_batch $EvalLeafBatch `
    --amp_inference `
    --seed 20260610 `
    --output $EvalPairs `
    --leaderboard_output $EvalLeaderboard `
    --game_log_output $EvalGames 2>&1 | Tee-Object -FilePath $EvalLog

if ($LASTEXITCODE -ne 0) {
    throw "M9 size-search eval failed with exit code $LASTEXITCODE"
}

if ($SkipNextTraining) {
    Write-Host "Eval complete; skipping next training because -SkipNextTraining was set."
    exit 0
}

if (-not (Test-Path $EvalPairs)) {
    throw "Eval pairs CSV not found: $EvalPairs"
}

function Get-CandidateScore {
    param(
        [Parameter(Mandatory=$true)][object[]]$Rows,
        [Parameter(Mandatory=$true)][string]$Candidate,
        [Parameter(Mandatory=$true)][string]$Current
    )

    foreach ($row in $Rows) {
        if ($row.a -eq $Candidate -and $row.b -eq $Current) {
            return [double]$row.a_win_rate
        }
        if ($row.a -eq $Current -and $row.b -eq $Candidate) {
            return 1.0 - [double]$row.a_win_rate
        }
    }
    throw "No pair row found for $Candidate vs $Current"
}

$rows = Import-Csv $EvalPairs
$current = "current_best_32x4_i080"
$candidates = @(
    [pscustomobject]@{ Name = "candidate32x6_i060"; Model = "32x6"; Channels = 32; Blocks = 6 },
    [pscustomobject]@{ Name = "candidate48x4_i060"; Model = "48x4"; Channels = 48; Blocks = 4 }
)

$scored = foreach ($candidate in $candidates) {
    $score = Get-CandidateScore -Rows $rows -Candidate $candidate.Name -Current $current
    [pscustomobject]@{
        Name = $candidate.Name
        Model = $candidate.Model
        Channels = $candidate.Channels
        Blocks = $candidate.Blocks
        Score = $score
    }
}

Write-Host ""
Write-Host "Equal-wall-clock scores vs current best:"
foreach ($item in $scored) {
    Write-Host ("  {0}: {1:P1}" -f $item.Name, $item.Score)
}
Write-Host ("Promotion threshold for auto-next architecture: {0:P1}" -f $PromotionThreshold)

$winner = $scored |
    Where-Object { $_.Score -ge $PromotionThreshold } |
    Sort-Object Score -Descending |
    Select-Object -First 1

if ($null -eq $winner) {
    $winner = [pscustomobject]@{
        Name = $current
        Model = "32x4"
        Channels = 32
        Blocks = 4
        Score = 0.5
    }
    Write-Host "No fair-checkpoint candidate cleared the threshold; using current-best architecture for the sims sweep."
} else {
    Write-Host "Auto-selected next architecture: $($winner.Model) from $($winner.Name)"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$nextName = "m9_autonext_$($winner.Model)_s${NextSims}_$stamp"
$checkpointDir = "checkpoints_$nextName"
$trainLog = Join-Path $LogDir "$nextName.log"

Write-Host ""
Write-Host "============================================================"
Write-Host "Starting automatic next training"
Write-Host "Model: $($winner.Model)"
Write-Host "Channels: $($winner.Channels)"
Write-Host "Blocks: $($winner.Blocks)"
Write-Host "Sims: $NextSims"
Write-Host "Iterations: $NextIterations"
Write-Host "Checkpoint dir: $checkpointDir"
Write-Host "Log: $trainLog"
Write-Host "============================================================"

& $Python -m games.kingdomino.self_play `
    --device cuda `
    --engine batched `
    --channels $winner.Channels `
    --blocks $winner.Blocks `
    --bilinear_dim 64 `
    --sims $NextSims `
    --batch_slots 86 `
    --leaf_batch 6 `
    --amp_inference `
    --games_per_iter 192 `
    --iterations $NextIterations `
    --train_steps 200 `
    --batch_size 256 `
    --lr 0.001 `
    --buffer 50000 `
    --checkpoint_dir $checkpointDir 2>&1 | Tee-Object -FilePath $trainLog

if ($LASTEXITCODE -ne 0) {
    throw "Automatic next training failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Automatic eval + training sequence completed."
