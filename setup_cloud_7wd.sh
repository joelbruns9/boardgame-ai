#!/usr/bin/env bash
# =============================================================================
# setup_cloud_7wd.sh — first-login setup + launch for the 7 Wonders Duel
# Phase D toy-scale loop on a Vast.ai GPU box (RTX 5060/5070 or similar).
#
# Brings a fresh Linux/CUDA instance to "toy run launched":
#   1. clone (or update) the repo at ~/boardgame-ai
#   2. Python deps — cu128 torch FIRST (Blackwell sm_120 needs it), then numpy
#   3. HARD-FAIL GPU verification gate (wrong wheel/driver dies in ~30s)
#   4. plumbing smoke on CUDA (~2 min: full loop cycle with tiny budgets)
#   5. launch the toy run detached with nohup so it survives SSH disconnect
#
# Design notes:
#   * Pure Python — no Rust toolchain or crate build (that is Kingdomino-only;
#     see setup_cloud.sh).
#   * Idempotent: re-running updates the repo and RESUMES the toy run — the
#     Phase D manifest detects completed iterations and --iterations N always
#     means "N more".
#   * The GPU gate keys off the actual device capability, so it also passes on
#     non-Blackwell rentals; the R570+ driver floor is enforced only when the
#     device really is sm_120.
#
# Usage (fresh box):
#   curl -fsSL https://raw.githubusercontent.com/joelbruns9/boardgame-ai/main/setup_cloud_7wd.sh -o setup_cloud_7wd.sh
#   bash setup_cloud_7wd.sh
# or from an existing clone:
#   bash ~/boardgame-ai/setup_cloud_7wd.sh
#
# Knobs (env vars):
#   ITERATIONS=20 GAMES_PER_ITERATION=500 SEED_GAMES=5000 WORKERS=8
#   PROCESS_WORKERS=$(nproc)  self-play processes; the throughput knob on
#                             many-core boxes (set 0 for threaded generation)
#   RUN_DIR_REL=runs/seven_wonders_duel/phase_d_toy
#   LAUNCH_TOY=1     set 0 to stop after the smoke (setup/verify only)
#   SKIP_SMOKE=0     set 1 to skip the CUDA plumbing smoke
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/joelbruns9/boardgame-ai.git"
REPO_DIR="${REPO_DIR:-$HOME/boardgame-ai}"
CU128_INDEX="https://download.pytorch.org/whl/cu128"
RUN_DIR_REL="${RUN_DIR_REL:-runs/seven_wonders_duel/phase_d_toy}"
ITERATIONS="${ITERATIONS:-20}"
GAMES_PER_ITERATION="${GAMES_PER_ITERATION:-500}"
SEED_GAMES="${SEED_GAMES:-5000}"
WORKERS="${WORKERS:-8}"
PROCESS_WORKERS="${PROCESS_WORKERS:-$(nproc)}"
LAUNCH_TOY="${LAUNCH_TOY:-1}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

log()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[OK]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[FATAL]\033[0m %s\n' "$*" >&2; exit 1; }

stage()      { printf '\n\033[1;36m=== STAGE %s: %s ===\033[0m\n' "$1" "$2"; }
stage_done() { printf '\033[1;32m=== STAGE %s COMPLETE ===\033[0m\n' "$1"; }

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
command -v "$PY" >/dev/null 2>&1 || die "No python3/python on PATH."

# ── STAGE 1: Clone (or update) the repo ──────────────────────────────────────
stage 1 "Clone repo into $REPO_DIR"
if [ -d "$REPO_DIR/.git" ]; then
  ok "Repo already present; updating with git pull."
  cd "$REPO_DIR"
  git pull --ff-only || warn "git pull failed; continuing with the existing checkout."
else
  if git clone "$REPO_URL" "$REPO_DIR"; then
    ok "Cloned $REPO_URL"
  else
    warn "Public clone failed — the repo may be private right now."
    if [ -t 0 ]; then
      read -r -p "GitHub username: " GH_USER
      read -r -s -p "GitHub personal access token (input hidden): " GH_TOKEN; echo
      git clone "https://${GH_USER}:${GH_TOKEN}@github.com/joelbruns9/boardgame-ai.git" \
        "$REPO_DIR" || die "Authenticated clone failed."
      ok "Cloned with token."
    else
      die "No TTY to prompt for a token. Clone manually then re-run this script."
    fi
  fi
  cd "$REPO_DIR"
fi
[ -f "games/seven_wonders_duel/phase_d.py" ] \
  || die "games/seven_wonders_duel/phase_d.py missing — push the Phase D code to main first."
stage_done 1

# ── STAGE 2: Python dependencies — cu128 torch FIRST, then numpy ─────────────
stage 2 "Python dependencies (cu128 torch first)"
"$PY" -m pip install --upgrade pip >/dev/null
log "Installing torch from cu128 index (sm_120 Blackwell kernels)"
"$PY" -m pip install torch --index-url "$CU128_INDEX"
log "Installing numpy"
"$PY" -m pip install "numpy>=1.24"
stage_done 2

# ── STAGE 3: GPU verification gate (HARD FAIL before any run) ────────────────
stage 3 "GPU verification gate"
GATE_PY="$(mktemp --suffix=.py)"
cat > "$GATE_PY" <<'PYGATE'
import subprocess, sys

BLACKWELL_MIN_DRIVER = 570
FAILED = []

def check(name, why):
    def deco(fn):
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            print(f"         why it matters: {why}")
            FAILED.append(name)
    return deco

import torch  # noqa: E402
print(f"  (torch {torch.__version__})")

@check("torch.cuda.is_available() is True",
       "no usable CUDA device visible to torch — wrong base image, no --gpus, "
       "or a CPU-only wheel.")
def _avail():
    assert torch.cuda.is_available()

CAP = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None

@check("driver new enough for this GPU",
       "sm_120 (Blackwell, RTX 50-series) needs driver R570+/CUDA 12.8+; an "
       "older host driver cannot be upgraded on a rental.")
def _driver():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True).strip().splitlines()[0].strip()
    major = int(out.split(".")[0])
    print(f"         driver_version={out}  device={torch.cuda.get_device_name(0)}  cap={CAP}")
    if CAP is not None and CAP[0] >= 12:
        assert major >= BLACKWELL_MIN_DRIVER, f"driver {out} < {BLACKWELL_MIN_DRIVER}"

@check("installed torch wheel has kernels for this device",
       "the cu126-wheel trap: torch imports fine but ships no sm_120 kernels — "
       "every forward fails with 'no kernel image available'.")
def _arch():
    arch = torch.cuda.get_arch_list()
    want = f"sm_{CAP[0]}{CAP[1]}"
    print(f"         arch_list={arch}")
    assert want in arch, f"{want} not in arch list"

@check("CUDA forward pass (TransformerEncoderLayer, the Phase D model family)",
       "a real attention + matmul kernel launch — catches wheels that import "
       "fine but cannot execute on this device.")
def _forward():
    layer = torch.nn.TransformerEncoderLayer(
        d_model=128, nhead=4, dim_feedforward=256, batch_first=True,
    ).to("cuda").eval()
    x = torch.randn(64, 40, 128, device="cuda")
    with torch.inference_mode():
        y = layer(x)
    torch.cuda.synchronize()
    assert y.shape == (64, 40, 128), y.shape

if FAILED:
    print("\n  Failed checks: " + ", ".join(FAILED))
    print("\n=== INSTANCE FAILED VERIFICATION — destroy this instance and re-rent ===")
    sys.exit(1)
print("\n=== ALL CHECKS PASSED — ready for Phase D ===")
PYGATE

if "$PY" "$GATE_PY"; then
  rm -f "$GATE_PY"
  stage_done 3
else
  rm -f "$GATE_PY"
  exit 1
fi

# ── STAGE 4: Plumbing smoke on CUDA (~2 min, tiny budgets) ───────────────────
stage 4 "Phase D plumbing smoke on CUDA"
cd "$REPO_DIR"
if [ "$SKIP_SMOKE" = "1" ]; then
  warn "SKIP_SMOKE=1; skipping the CUDA plumbing smoke."
else
  # Fresh dir each invocation: smoke runs are throwaway and must not resume.
  SMOKE_DIR="runs/seven_wonders_duel/phase_d_smoke_$(date +%Y%m%dT%H%M%S)"
  # --process-workers 2 exercises the spawn/process generation path the toy
  # run will use, on top of the CUDA training/gate path.
  "$PY" -m games.seven_wonders_duel.phase_d \
    --run-dir "$SMOKE_DIR" --device cuda --plumbing-smoke --process-workers 2 \
    || die "CUDA plumbing smoke failed — do not launch the toy run."
  ok "Smoke completed: $SMOKE_DIR"
fi
stage_done 4

# ── STAGE 5: Launch the toy run detached (survives SSH disconnect) ───────────
stage 5 "Launch Phase D toy run"
RUN_DIR="$REPO_DIR/$RUN_DIR_REL"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/launch_$(date +%Y%m%dT%H%M%S).log"

if [ "$LAUNCH_TOY" != "1" ]; then
  warn "LAUNCH_TOY=$LAUNCH_TOY; setup verified but not launching. Launch manually with:"
  warn "  cd $REPO_DIR && nohup $PY -m games.seven_wonders_duel.phase_d \\"
  warn "    --run-dir $RUN_DIR_REL --device cuda --iterations $ITERATIONS \\"
  warn "    --games-per-iteration $GAMES_PER_ITERATION --seed-games $SEED_GAMES \\"
  warn "    --workers $WORKERS --process-workers $PROCESS_WORKERS >> $LOG_FILE 2>&1 &"
  stage_done 5
  ok "Setup complete."
  exit 0
fi

nohup "$PY" -m games.seven_wonders_duel.phase_d \
  --run-dir "$RUN_DIR_REL" \
  --device cuda \
  --iterations "$ITERATIONS" \
  --games-per-iteration "$GAMES_PER_ITERATION" \
  --seed-games "$SEED_GAMES" \
  --workers "$WORKERS" \
  --process-workers "$PROCESS_WORKERS" \
  >> "$LOG_FILE" 2>&1 &
TOY_PID=$!
disown "$TOY_PID" 2>/dev/null || true
sleep 5
kill -0 "$TOY_PID" 2>/dev/null || die "Toy run died within 5s — check $LOG_FILE"
ok "Toy run launched: pid=$TOY_PID iterations=$ITERATIONS games/iter=$GAMES_PER_ITERATION process_workers=$PROCESS_WORKERS"
stage_done 5

cat <<EOF

Monitor progress:
  tail -f "$LOG_FILE"
  python3 - <<'PYEOF'
import json
rows = json.load(open("$RUN_DIR/run_manifest.json"))["iterations"]
for r in rows:
    print(r["iteration"], "promoted" if r["promoted"] else "rejected",
          f'{r["promotion_gate"]["score_rate"]:.3f}',
          f'{r["generation_performance"].get("games_per_second", 0):.2f} games/s')
PYEOF

Resume after interruption (manifest skips completed iterations; --iterations
always means "N more"):
  bash $REPO_DIR/setup_cloud_7wd.sh        # or rerun just the nohup line above

Pull results back to the laptop when done (run FROM the laptop):
  scp -r -P <ssh-port> root@<instance-ip>:$RUN_DIR runs/seven_wonders_duel/

EOF
ok "Setup complete."
