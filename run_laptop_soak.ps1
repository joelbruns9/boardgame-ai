# Overnight laptop soak for the 7WD stack (PowerShell).
#
# A FUNCTIONAL soak, not a strength run: the net is ~5M params on a 3070, so
# nothing here measures playing strength. What it tests is whether every
# mechanism the RENTED BOX will run survives running together.
#
#   .\run_laptop_soak.ps1                      # start
#   .\run_laptop_soak.ps1 -Resume              # continue from the run dir
#   .\run_laptop_soak.ps1 -RunDir runs/...     # somewhere else
#
# WHY 20 ITERATIONS AND NOT 150. Under the soft gate a gate runs every
# `--promotion-every` iterations, and the lifecycle counters are counted in GATE
# CHECKS, not iterations (`training_control.py`). So the schedule is:
#
#   iter 3   specialists seeded, league opens (--specialist-bootstrap-games)
#   iter 5   first soft gate
#   iter 10  the military specialist is first drawn (ONE class per iteration)
#   iter 15  third gate -- --revert-reset-after 3 can fire
#   iter 20  fourth gate -- --probation-reset-after 4 can fire
#
# Everything after iteration 20 repeats mechanisms already exercised, and both
# counters have direct unit coverage in `test_az_loop_controller.py`. ~20
# iterations is roughly 3 hours here, short enough to read the result and run it
# again the same day.
#
# WHAT CHANGED, AND WHY IT IS THE POINT. The first soak (2026-09-09, killed by a
# Windows reboot at iteration 8) passed ~50 fewer flags than the cloud launcher
# assembles, and several were whole MECHANISMS rather than values: the
# pooled-readout/reply-head architecture, Dirichlet noise,
# forced playouts, the value bootstrap, the self-anchor, curriculum annealing
# and the gate-side scheduler geometry. A long run of a configuration the box
# will not use is worth less than a short run of the one it will. Diff this
# file's flags against `bash setup_dryrun.sh` before changing either.
#
# The VALUES are laptop-scale; the MECHANISM SET is meant to match the box.
#
# Deliberately NOT matched to the box, with reasons:
#   --hof-start-games 300 / --specialist-bootstrap-games 300  (box: 50,000 /
#     10,000) so the league actually plays inside a 2,000-game run.
#   sims, d-model, slots, solver budget -- laptop hardware.
#
# A resume is REFUSED across a commit change (W6.5) and across a generator-mode
# change, so a code change means a new run directory, not -Resume.
#
# --iterations is how many MORE iterations to run on a resume, not a target
# total.

param(
    [switch]$Resume,
    # Build and validate the config, then exit without training. The cloud
    # launcher does this before it detaches; a flag combination that Phase D
    # refuses should cost a second here, not the first iteration of a run.
    [switch]$ValidateOnly,
    [string]$RunDir = "runs/seven_wonders_duel/laptop_soak2",

    # ---- Phase 1: measure what the endgame solver's DECLINES cost ----------
    # Runs BEFORE the soak and never beside it. Both are throughput-sensitive
    # and they contend for the same cores, so overlapping them would invalidate
    # each other's numbers -- the reason RENTING_A_BOX.md insists a sweep gets
    # the box to itself.
    [switch]$SkipResolve,
    [switch]$ResolveOnly,
    # cloud2's most recent iteration: the strongest net that run produced, so
    # its endgames are the closest available match to what the next loop will
    # reach. Older iterations are weaker and less representative.
    [string]$ResolveBuffer =
        "runs/seven_wonders_duel/cloud2/7wd_cloud_20260825T005745Z/buffers/iter_0096.jsonl",
    [int]$ResolveThreads = 8,
    # 0 = every censored position in the buffer. A small value is for smoke
    # tests; the frontier needs the whole tail to be meaningful.
    [int]$ResolveLimit = 0,
    [string]$ResolveOut = "runs/seven_wonders_duel/endgame_study/censored_iter_0096.json"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# No --init-checkpoint, because the box does not use one either: it starts from
# a random initialisation plus --seed-games of curriculum play, and
# --bootstrap-policy auto_first_trained promotes the first trained net. The old
# seed checkpoint also predates --pooled-readout/--reply-head, so loading it
# here would fail the model contract check rather than seed anything.

# An array, splatted -- PowerShell parses `--flag` in an expression as a unary
# operator, so a bash-style multi-line invocation is a parse error here.
$a = @(
    "-m", "games.seven_wonders_duel.phase_d",
    "--run-dir", $RunDir,
    "--iterations", "20",
    "--games-per-iteration", "100",

    # ---- Architecture. The box trains THIS model; a soak on a different one
    # ---- exercises different parameters and a different checkpoint contract.
    "--d-model", "256",
    "--layers", "6",
    # 256/4 = 64 per head, the same head width as the box's 384/6.
    "--heads", "4",
    "--pooled-readout",
    "--reply-head",
    "--hierarchical-value",
    "--hier-value-weight", "0.3",

    # ---- Search. cloud2's split: PUCT on recorded moves, Gumbel on cheap ones.
    "--selfplay-search-mode", "puct",
    "--cheap-search-mode", "gumbel",
    "--eval-search-mode", "puct",
    # Root exploration. Both default to off, and the box turns both on.
    "--dirichlet-epsilon", "0.25",
    "--dirichlet-alpha", "1.8",
    # KataGo forced playouts; needs a PUCT root, and is inert without one.
    "--forced-playout-k", "1.0",
    # 1, not 6. Leaf batching and the coalescer fill the same forward pass, and
    # the coalescer does it from INDEPENDENT slots, exactly. Within-tree
    # batching is the approximate version: it needs virtual loss at a PUCT root,
    # and on full moves the root's visit distribution IS the policy target.
    # Measured here: --leaf-batch 6 delivered a wave width of 3.33 with 75,643
    # conflict cuts, while batch size came overwhelmingly from slot count.
    "--leaf-batch", "1",
    # INERT for generation at leaf-batch 1 (Rust gates it on leaf_batch > 1),
    # but --eval-leaf-batch below is refused without it, because evaluation runs
    # a PUCT root.
    "--virtual-loss-root",
    # The cheap root is Gumbel and batches by conflict-free waves instead. The
    # waves/round-robin pair is mandatory: without it every wave is cut back to
    # width 1 and the cheap batch is inert.
    "--cheap-leaf-batch", "16",
    "--cheap-conflict-free-waves",
    "--cheap-round-robin-candidates",
    "--eval-leaf-batch", "16",
    "--cheap-sims-min", "32",
    "--cheap-sims-max", "48",
    "--full-sims-min", "128",
    "--full-sims-max", "192",
    "--full-search-fraction", "0.25",
    "--full-search-every-games", "25",
    "--top-k", "16",
    "--age-deal-samples", "32",
    "--cheap-double-reveal-offsets", "3",
    # The same cap on FULL searches, which carry the training targets. Double
    # reveals are 54.5% of all forced chance children; forced rows were 35.5% of
    # every network row in the first soak. Stratified, not truncated.
    "--double-reveal-offsets", "3",

    # ---- Training.
    "--train-steps", "100",
    "--train-warmup-steps", "33",
    "--train-batch-size", "512",
    "--learning-rate", "5e-5",
    "--weight-decay", "0.5",
    # The setting that makes a shaped root dangerous, and therefore the one that
    # lets W7's `assert_no_shaped_bootstrap` fail at all.
    "--value-bootstrap", "0.5",
    "--action-policy-weight", "0",
    # Action sampling. Defaults anneal to a hard argmax; the box holds a floor,
    # so its games carry exploration this one would not have had.
    "--temperature-floor", "0.35",
    "--temperature-anneal-moves", "30",

    # ---- Schedules. The games basis, with the knots pulled in so each one is
    # ---- actually CROSSED inside ~2,000 games instead of never firing.
    "--schedule-basis", "games",
    "--seed-games", "5000",
    "--curriculum-anneal-games", "1000",
    "--draft-prior-games", "800",
    "--opponent-fraction", "0",
    "--intervention-window-games", "2000",
    "--replay-window-coefficient", "1000",
    "--replay-window-exponent", "0.6",
    "--replay-window-cap-games", "2000",

    # ---- Specialist league (W7).
    # Lambda 3 is the measured peak of the pursuit curve (S0a).
    "--specialists", "science:0.15:3,military:0.10:3",
    # Must be <= specialist-bootstrap-games, or the league seeds and never
    # plays. PhaseDConfig.validate refuses the inversion.
    "--specialist-bootstrap-games", "300",
    "--specialist-floor-every", "5",
    "--specialist-reanalysis",
    "--reanalysis-backend", "rust_coalesced",
    "--reanalysis-slots", "256",
    "--hof-start-games", "300",
    "--hof-opponent-fraction", "0.15",

    # ---- Anchors. The box leaves the bot-suite gate off and relies on the SELF
    # ---- anchor, so this does too -- scaled to fire ~5 times in 2,000 games
    # ---- rather than never.
    "--anchor-gate-every-promotions", "0",
    # Inert while the gate above is 0, exactly as on the box. Named so the flag
    # diff against `setup_dryrun.sh` stays empty and stays worth reading.
    "--anchor-games", "40",
    "--self-anchor-games", "40",
    "--self-anchor-lag-games", "400",
    "--self-anchor-every-games", "400",

    # ---- Lifecycle. cloud2's methodology.
    "--selfplay-generator-mode", "soft_gate",
    "--bootstrap-policy", "auto_first_trained",
    "--promotion-every", "5",
    # These default to 0, i.e. disabled -- a soft-gate run without them never
    # resets the learner however badly it does.
    "--revert-reset-after", "3",
    "--probation-reset-after", "4",
    "--promotion-min-lcb", "0.50",
    "--revert-max-ucb", "0.48",
    # Two rungs so the ladder can actually step up, which one rung cannot test.
    "--gate-ladder-games", "60", "200",
    "--gate-ladder-step-up-after", "2",
    "--gate-ladder-floor-games", "1000",
    "--gate-sims", "48",
    # Gate-side scheduler geometry, deliberately separate from generation's.
    "--gate-slots", "48",
    "--gate-global-batch-cap", "512",

    # ---- Endgame solver.
    "--endgame-solver-max-nodes", "200000",
    "--endgame-solver-max-secs", "10",
    "--solver-threads", "2",
    "--endgame-cost-model", "games/seven_wonders_duel/endgame_cost_model.json",
    "--solver-fallback-research",

    # ---- Backends and scheduler geometry.
    "--generation-backend", "rust",
    "--gate-backend", "rust",
    "--derive-backend", "rust",
    "--rust-slots", "24",
    "--rust-scheduler-workers", "2",
    "--rust-global-batch-cap", "512",
    "--rust-max-inflight-batches", "1",
    "--pack-threads", "0",
    # CPU parallelism for seed generation and the Python fallback paths. Lower
    # than the box's 8/16 because this is a 16-thread laptop that is also the
    # machine you are using.
    "--workers", "4",
    "--process-workers", "8",

    # ---- Memory. Conservative: four processes were OOM-killed on this machine.
    "--example-cache-gb", "1.5",
    "--memory-budget-gb", "9",
    "--memory-headroom-gb", "1",
    "--vram-budget-gb", "0",
    "--min-games-to-train", "20",
    "--min-buffer-positions", "200",
    "--buffer-autosave-every", "5",

    "--device", "cuda",
    "--precision", "bf16"
)

if ($ValidateOnly) {
    $a += "--validate-config"
    Write-Host "==> validating the launch configuration only"
    python @a
    exit $LASTEXITCODE
}

# ---- Phase 1 -------------------------------------------------------------
#
# cloud2 spent 46% of all solver nodes on the 3.2% of attempts that DECLINED,
# and a declined position is right-censored: we know only that it cost at least
# the budget. Refitting the cost model on buffers cannot fix that -- every
# censored row sits at the same cap -- so the tail has to be measured.
#
# Sized from a measured sample: ~96s of CPU per position under 8-way contention,
# 259 positions in iter_0096, so roughly 50-60 minutes. True costs in the sample
# were 46M-366M nodes against the 40M cap that censored them, median 2.4x.
if (-not $SkipResolve -and -not $ValidateOnly) {
    Write-Host "==> phase 1: re-solving censored endgames from $ResolveBuffer"
    if (-not (Test-Path $ResolveBuffer)) {
        Write-Error "resolve buffer not found: $ResolveBuffer"
        exit 1
    }
    $r = @(
        "-m", "games.seven_wonders_duel.resolve_censored",
        $ResolveBuffer,
        "--threads", "$ResolveThreads",
        # 20x the 40M cap. Nothing in the sample came close, so a decline at
        # this budget is a genuinely expensive position rather than a cap.
        "--max-nodes", "800000000",
        # Slack on purpose: a deadline stop censors a position a SECOND time,
        # and for a machine-load-dependent reason.
        "--max-secs", "1200",
        "--censored-at", "40000000",
        "--limit", "$ResolveLimit",
        "--out", $ResolveOut
    )
    $rs = Get-Date
    python @r
    $rc = $LASTEXITCODE
    Write-Host ("==> phase 1 exited {0} after {1:hh\:mm\:ss}" -f $rc, ((Get-Date) - $rs))
    if ($rc -ne 0) {
        Write-Error "phase 1 failed; not starting the soak on a contended machine"
        exit $rc
    }
    if ($ResolveOnly) { exit 0 }
}
elseif ($ResolveOnly) {
    Write-Error "-ResolveOnly and -SkipResolve are contradictory"
    exit 1
}

# ---- Phase 2 -------------------------------------------------------------
# No tee and no 2>&1: --run-log already writes <run-dir>/run.log, and in
# Windows PowerShell 5.1 redirecting a native executable's stderr wraps every
# line in an ErrorRecord and sets $? to false even on a clean exit.
Write-Host "==> $(if ($Resume) {'resuming'} else {'starting'}) soak in $RunDir; transcript at $RunDir/run.log"
$started = Get-Date
python @a
$code = $LASTEXITCODE
$elapsed = (Get-Date) - $started
Write-Host ("==> exited {0} after {1:hh\:mm\:ss}" -f $code, $elapsed)
exit $code
