# Overnight laptop soak for the 7WD stack (PowerShell).
#
# A FUNCTIONAL soak, not a strength run: the net is 5.2M params on a 3070, so
# nothing here measures playing strength. What it tests is whether specialists,
# reanalysis, the endgame solver, the evaluator coalescer and the hierarchical
# value head all survive ten hours together -- and whether the specialist
# accept/REVERT cycle works when given enough iterations to revert.
#
#   .\run_laptop_soak.ps1            # start
#   .\run_laptop_soak.ps1 -Resume    # continue from the run dir
#
# Kill it deliberately around hour 2 and resume: resume with an active
# specialist league has never been tested, and finding it broken at hour 8
# wastes the night.
#
# Sized from a measured 0.656 games/s of generation; with --promotion-every 5
# an iteration averages ~236s, so ~150 iterations and ~15,000 games in 10h.
#
# NOTE: hof-start-games is 300 here so specialists actually play within the
# run. The box uses 10,000. Do not carry this number over.

param([switch]$Resume)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$RunDir = "runs/seven_wonders_duel/laptop_soak"
$Seed = "$RunDir/seed/seed_256x6.pt"

if (-not $Resume) {
    if (-not (Test-Path $Seed)) {
        Write-Error "seed checkpoint missing: $Seed"
        exit 1
    }
}

# An array, splatted -- PowerShell parses `--flag` in an expression as a unary
# operator, so a bash-style multi-line invocation is a parse error here.
$a = @(
    "-m", "games.seven_wonders_duel.phase_d",
    "--run-dir", $RunDir,
    "--iterations", "150",
    "--games-per-iteration", "100",
    "--d-model", "256",
    "--layers", "6",
    "--hierarchical-value",
    "--hier-value-weight", "0.3",
    # W7. Lambda 3 is the measured peak of the pursuit curve (S0a).
    "--specialists", "science:0.15:3,military:0.10:3",
    "--specialist-bootstrap-games", "300",
    "--specialist-floor-every", "5",
    # S2b, off by default and never exercised. The likeliest thing to fail
    # early; drop this one flag if protecting the soak matters more than
    # testing it.
    "--specialist-reanalysis",
    # Must be <= specialist-bootstrap-games, or the league seeds and never
    # plays. PhaseDConfig.validate refuses the inversion.
    "--hof-start-games", "300",
    "--hof-opponent-fraction", "0.15",
    "--cheap-sims-min", "32",
    "--cheap-sims-max", "48",
    "--full-sims-min", "128",
    "--full-sims-max", "192",
    "--rust-slots", "24",
    "--rust-scheduler-workers", "2",
    "--rust-global-batch-cap", "512",
    "--solver-threads", "2",
    "--endgame-solver-max-nodes", "200000",
    "--train-steps", "100",
    "--weight-decay", "0.0",
    "--leaf-batch", "4",
    "--promotion-every", "5",
    "--gate-ladder-games", "60",
    "--gate-sims", "48",
    "--replay-window", "6",
    "--replay-window-cap-games", "2000",
    # Conservative: four processes were OOM-killed on this machine today.
    "--example-cache-gb", "1.5",
    "--memory-budget-gb", "9",
    "--memory-headroom-gb", "1",
    "--min-games-to-train", "20",
    "--min-buffer-positions", "200",
    "--buffer-autosave-every", "5",
    "--device", "cuda",
    "--precision", "bf16"
)

if (-not $Resume) {
    $a += @("--init-checkpoint", $Seed)
}

# No tee and no 2>&1: --run-log already writes <run-dir>/run.log, and in
# Windows PowerShell 5.1 redirecting a native executable's stderr wraps every
# line in an ErrorRecord and sets $? to false even on a clean exit.
Write-Host "==> $(if ($Resume) {'resuming'} else {'starting'}) soak; transcript at $RunDir/run.log"
$started = Get-Date
python @a
$code = $LASTEXITCODE
$elapsed = (Get-Date) - $started
Write-Host ("==> exited {0} after {1:hh\:mm\:ss}" -f $code, $elapsed)
exit $code
