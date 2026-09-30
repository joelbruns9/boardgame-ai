# Runs the queued Can't Stop experiments back to back, stopping at the
# first failure. Run from the boardgame-ai-cantstop folder with the venv
# active:
#
#     powershell -ExecutionPolicy Bypass -File games\cantstop\run_queue.ps1
#
# Each step logs to runs\<name>.log. A step whose output folder already
# exists is refused rather than overwritten (train.py would append to it);
# delete or rename it to re-run. Rough total: ~12-13 hours.
#
# -Skip leaves out finished steps by name, e.g.
#     ... -File games\cantstop\run_queue.ps1 -Skip refl_on,confirm_reflect

param([string[]]$Skip = @())

$ErrorActionPreference = "Stop"
# Under "powershell -File", "a,b" arrives as ONE string, not an array.
$Skip = @($Skip | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() })
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))

function Step($name, $out, [string[]]$pyArgs) {
    if ($Skip -contains $name) {
        Write-Host "SKIP  $name"
        return
    }
    if ($out -and (Test-Path $out)) {
        throw "$out already exists - refusing to overwrite (step $name)"
    }
    $log = "runs\$name.log"
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm') START $name -> $log"
    # Through cmd, not PowerShell redirection: Windows PowerShell 5.1 turns
    # every stderr line (torch warnings included) into a terminating error
    # under "Stop", and *> writes UTF-16 logs. cmd keeps python's exit code.
    cmd /c ("python " + ($pyArgs -join " ") + " > `"$log`" 2>&1")
    if ($LASTEXITCODE -ne 0) {
        throw "step $name failed (exit $LASTEXITCODE), see $log"
    }
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm') DONE  $name"
}

$refl = @("--iterations", "100", "--games", "400",
          "--init-checkpoint", "runs/td0_personas/iter_0120.pt",
          "--td-target", "exact", "--td-lambda", "0", "--replay-window", "10",
          "--passes", "5", "--conservative", "0.2", "--aggressive", "0.1",
          "--lr-schedule", "1:5e-5", "--arena-games", "200", "--seed", "7")

# 1. Reflection augmentation: 60 more iterations from td0_personas/iter_0120 (~3.5 h).
#    No matched no-augmentation control (skipped to save time): the
#    confirmation below compares against the STARTING checkpoint, so any
#    difference mixes the augmentation with 60 iterations of extra training.
Step "refl_on" "runs/refl_on" (@("-m", "games.cantstop.train", "--out", "runs/refl_on") + $refl + @("--reflect-augment"))

# 2. Confirmation: augmented continuation vs its start, 10,000 fresh seat-balanced games, +/-1 pt (~25 min)
Step "confirm_reflect" $null @("-m", "games.cantstop.confirm",
    "--a", "runs/refl_on/iter_0060.pt", "--b", "runs/td0_personas/iter_0120.pt",
    "--games", "10000", "--margin", "0.01", "--out", "runs/confirm_reflect.json")

$p4 = @("--iterations", "40", "--td-lambda", "0", "--conservative", "0.2",
        "--aggressive", "0.1", "--personas-from", "6", "--reflect-augment",
        "--lr-schedule", "1:1e-3", "20:3e-4",
        "--eval-every", "10", "--reference-iter", "20", "--seed", "11")

# 3. Phase 4 generalist pilot, all ten variants, fresh net (~4-5 h)
Step "p4_pilot" "runs/p4_pilot" (@("-m", "games.cantstop.phase4", "--out", "runs/p4_pilot") + $p4)

# 4. 3-player, 4-column, blocking specialist at matched rows per iteration (~4-5 h)
Step "p4_spec_3p4b" "runs/p4_spec_3p4b" (@("-m", "games.cantstop.phase4", "--out", "runs/p4_spec_3p4b",
    "--variants", "3:4:b", "--rows-per-variant", "40000") + $p4)

# 5. Open-ended filler: resume the generalist from its state.pt and carry it
#    on to 80 iterations (LR 1e-4 from 60), for however long the GPU is
#    free. Safe to Ctrl+C at any time: state.pt is saved every iteration,
#    and "--resume" with a higher --iterations continues it later.
Step "p4_pilot_ext" $null (@("-m", "games.cantstop.phase4", "--out", "runs/p4_pilot") + $p4 +
    @("--iterations", "80", "--lr-schedule", "1:1e-3", "20:3e-4", "60:1e-4", "--resume"))

Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm') ALL DONE"
