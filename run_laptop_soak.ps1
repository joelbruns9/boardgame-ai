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
# Resume with an active specialist league was exercised for real on 2026-09-10:
# a Windows update rebooted the box at 01:31 after 8 iterations, and the run
# resumed cleanly from the training log.
#
# MEASURED sizing, replacing the estimate that shipped with this file. That
# estimate said ~236s/iteration on the assumption that --promotion-every 5
# gated every fifth iteration; under the strict_gate lifecycle it did not (it
# only fed revert suppression), so all 400 gate games ran EVERY iteration and
# reanalysis added ~1000s more. Real cost was ~25 min/iteration -- 150
# iterations would have taken 60 hours, not 10.
#
# Under the soft gate --promotion-every 5 IS the gate cadence, and the
# coalesced reanalysis backend took S2b from ~1000s to ~30s (34x, measured
# 2539 -> 74.8 ms/position). Budget per iteration, measured on this laptop:
#
#   generation  ~160s   gate ~320s   train ~24s   replay ~15s   reanalysis ~30s
#
# so roughly 9-10 min/iteration and ~65 iterations in a 10-hour night. Note
# --iterations is how many MORE to run on a resume, not a target total.
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
    # cloud2's search split: PUCT for the full search and the gates, Gumbel for
    # the cheap search. A PUCT root cannot run under leaf batching -- it would
    # select against virtual loss, which is a different algorithm, not a slower
    # one -- so --leaf-batch drops to 1 below. cloud2 ran leaf_batch 1 too.
    "--selfplay-search-mode", "puct",
    "--cheap-search-mode", "gumbel",
    "--eval-search-mode", "puct",
    "--rust-slots", "24",
    "--rust-scheduler-workers", "2",
    "--rust-global-batch-cap", "512",
    "--solver-threads", "2",
    "--endgame-solver-max-nodes", "200000",
    "--train-steps", "100",
    "--weight-decay", "0.0",
    # 1, not 4: PhaseDConfig.validate refuses a PUCT root above 1.
    "--leaf-batch", "1",
    # ...but the CHEAP root is Gumbel, so it can still batch. Cheap moves are
    # `policy_excluded` -- their visit distributions are never policy targets --
    # so widening them costs no target fidelity, and they are ~76% of moves.
    # The waves/round-robin pair is mandatory: without it every wave is cut back
    # to width 1 and the cheap batch is inert.
    "--cheap-leaf-batch", "4",
    "--cheap-conflict-free-waves",
    "--cheap-round-robin-candidates",
    "--promotion-every", "5",
    # Soft gate (now the default, named here so the log records the choice).
    # Under this lifecycle --promotion-every really is the gate cadence; under
    # strict_gate it only fed revert suppression and every iteration gated.
    "--selfplay-generator-mode", "soft_gate",
    "--bootstrap-policy", "auto_first_trained",
    # The probation/revert counters cloud2 ran. They default to 0 (disabled),
    # so a soft-gate run without them never resets the learner.
    "--revert-reset-after", "3",
    "--probation-reset-after", "4",
    "--promotion-min-lcb", "0.50",
    "--revert-max-ucb", "0.48",
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
