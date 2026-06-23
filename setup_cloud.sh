#!/usr/bin/env bash
# =============================================================================
# setup_cloud.sh — first-login setup for a Vast.ai RTX 5090 (Blackwell) box.
#
# Brings a fresh Linux/CUDA-12.8 instance to "ready to benchmark" for the
# Kingdomino AlphaZero training run:
#   1. Rust toolchain via rustup (>= 1.85, for Cargo edition 2024)
#   2. clone (or reuse) the repo at ~/boardgame-ai
#   3. Python deps — cu128 torch FIRST, then the rest of requirements.txt
#   4. build the kingdomino_rust crate with maturin
#   5. HARD-FAIL GPU verification gate (wrong wheel/driver dies in ~30s)
#   6. print the exact benchmark + training commands to run next
#
# Design notes:
#   * Idempotent: safe to re-run after a partial failure. Rust install, clone,
#     and the crate build all detect prior state.
#   * No credentials baked in. Clones public over HTTPS; if the repo is private
#     and a TTY is present, prompts for a GitHub token.
#   * The GPU gate runs BEFORE any benchmark; benchmarks run BEFORE any training
#     (this script stops at the gate and only PRINTS the next commands).
#
# Usage (fresh box):
#   curl -fsSL https://raw.githubusercontent.com/joelbruns9/boardgame-ai/main/setup_cloud.sh -o setup_cloud.sh
#   bash setup_cloud.sh
# or, if you already cloned the repo, just run it from anywhere:
#   bash ~/boardgame-ai/setup_cloud.sh
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/joelbruns9/boardgame-ai.git"
REPO_DIR="${REPO_DIR:-$HOME/boardgame-ai}"
CRATE_DIR_REL="games/kingdomino/kingdomino_rust"
CU128_INDEX="https://download.pytorch.org/whl/cu128"
MIN_DRIVER=570

log()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[OK]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[FATAL]\033[0m %s\n' "$*" >&2; exit 1; }

# Stage banners (spec format: easy to follow progress in the terminal).
stage()      { printf '\n\033[1;36m=== STAGE %s: %s ===\033[0m\n' "$1" "$2"; }
stage_done() { printf '\033[1;32m=== STAGE %s COMPLETE ===\033[0m\n' "$1"; }

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
command -v "$PY" >/dev/null 2>&1 || die "No python3/python on PATH."

# ── STAGE 1: Rust via rustup (NOT apt; we need >= 1.85 for edition 2024) ─────
stage 1 "Rust toolchain (rustup)"
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
else
  ok "cargo already present: $(cargo --version)"
fi
# Always source the env so cargo is on PATH for the rest of this script.
# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
command -v cargo >/dev/null 2>&1 || die "cargo still not on PATH after rustup."
ok "$(rustc --version)"
stage_done 1

# ── STAGE 2: Clone (or update) the repo ──────────────────────────────────────
stage 2 "Clone repo into $REPO_DIR"
if [ -d "$REPO_DIR/.git" ]; then
  # Idempotent: already cloned → cd in and fast-forward to latest.
  ok "Repo already present; updating with git pull."
  cd "$REPO_DIR"
  git pull --ff-only || warn "git pull failed; continuing with the existing checkout."
else
  # Public repo → plain HTTPS, no credentials. The token fallback only fires if
  # the public clone fails AND a TTY is present; nothing is hardcoded.
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
[ -d "$CRATE_DIR_REL" ] || die "Expected crate dir '$CRATE_DIR_REL' missing — wrong repo?"
stage_done 2

# ── STAGE 3: Python dependencies — cu128 torch FIRST, then the rest ──────────
stage 3 "Python dependencies (cu128 torch first)"
"$PY" -m pip install --upgrade pip >/dev/null
log "Installing torch + torchvision from cu128 index (sm_120 Blackwell kernels)"
"$PY" -m pip install torch torchvision --index-url "$CU128_INDEX"

if [ -f requirements.txt ]; then
  # Strip torch/torchvision lines so requirements.txt can NEVER pull a wrong
  # (cu126 / PyPI) wheel over the cu128 build we just installed. Comment lines
  # (starting with #) are preserved by the anchored alternation below.
  REQ_NOGPU="$(mktemp)"
  grep -viE '^[[:space:]]*(torch|torchvision|torchaudio)([[:space:]]|$|[=<>~!])' \
    requirements.txt > "$REQ_NOGPU" || true
  log "Installing remaining deps (torch/torchvision/torchaudio lines stripped)"
  "$PY" -m pip install -r "$REQ_NOGPU"
  rm -f "$REQ_NOGPU"
  ok "requirements.txt installed (GPU wheels left untouched)."
else
  warn "No requirements.txt found; installing minimal build deps directly."
  "$PY" -m pip install numpy "maturin>=1.5"
fi
stage_done 3

# ── STAGE 4: Build the Rust crate (maturin develop --release) ────────────────
stage 4 "Build kingdomino_rust ($CRATE_DIR_REL)"
command -v maturin >/dev/null 2>&1 || "$PY" -m pip install "maturin>=1.5"
(
  cd "$CRATE_DIR_REL"
  maturin develop --release
)
"$PY" -c "import kingdomino_rust; print('kingdomino_rust import OK')" \
  || die "kingdomino_rust failed to import after build."
ok "Rust crate built and importable."
stage_done 4

# ── STAGE 5: GPU verification gate (HARD FAIL before any benchmark/training) ─
stage 5 "GPU verification gate"
GATE_PY="$(mktemp --suffix=.py)"
cat > "$GATE_PY" <<'PYGATE'
import subprocess, sys

MIN_DRIVER = 570
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

@check("nvidia-smi driver >= %d (Blackwell minimum)" % MIN_DRIVER,
       "sm_120 kernels need driver R570+/CUDA 12.8+; an older host driver "
       "cannot run the 5090 and CANNOT be upgraded on a rental.")
def _driver():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True).strip().splitlines()[0].strip()
    major = int(out.split(".")[0])
    assert major >= MIN_DRIVER, f"driver {out} < {MIN_DRIVER}"
    print(f"         driver_version={out}")

# torch is imported lazily so a driver failure reports first/clearly.
import torch  # noqa: E402
print(f"  (torch {torch.__version__})")

@check("torch.cuda.is_available() is True",
       "no usable CUDA device visible to torch — wrong base image, no --gpus, "
       "or a CPU-only wheel.")
def _avail():
    assert torch.cuda.is_available()

@check("device capability == (12, 0)  [sm_120 / Blackwell]",
       "the 5090 is sm_120; a non-(12,0) reading means the GPU isn't the "
       "expected Blackwell part.")
def _cap():
    cap = torch.cuda.get_device_capability(0)
    assert cap == (12, 0), f"got {cap}"
    print(f"         capability={cap}  ({torch.cuda.get_device_name(0)})")

@check("'sm_120' in torch.cuda.get_arch_list()",
       "the installed torch wheel was NOT built with sm_120 kernels (this is "
       "the cu126-wheel trap) — forwards fail with 'no kernel image available'.")
def _arch():
    arch = torch.cuda.get_arch_list()
    assert 'sm_120' in arch, f"arch list = {arch}"
    print(f"         arch_list={arch}")

@check("CUDA forward pass (conv + matmul, batch=45)",
       "a real kernel launch — catches wheels that import fine but have no "
       "runnable kernels for this device.")
def _forward():
    dev = "cuda"
    x = torch.randn(45, 9, 13, 13, device=dev)
    conv = torch.nn.Conv2d(9, 32, 3, padding=1).to(dev)
    y = conv(x)                                  # conv kernel
    flat = y.flatten(1)
    w = torch.randn(flat.shape[1], 128, device=dev)
    z = flat @ w                                  # matmul kernel
    torch.cuda.synchronize()
    assert z.shape == (45, 128), z.shape

@check("import triton",
       "torch.compile's inductor backend needs Triton to codegen GPU kernels; "
       "without it --compile silently falls back to eager (no speedup). The "
       "Linux cu128 wheel bundles Triton — its absence signals a wrong wheel.")
def _triton():
    import triton  # noqa: F401
    print(f"         triton {getattr(triton, '__version__', '?')}")

@check("torch.compile(net, dynamic=True) forward actually EXECUTES (not just imports)",
       "compile can import yet fail at first real call (Triton/inductor codegen "
       "for sm_120); dynamic=True is the mode training uses for the variable "
       "leaf-eval batch, so we verify exactly that path runs a compiled kernel.")
def _compiled():
    dev = "cuda"
    m = torch.nn.Sequential(
        torch.nn.Conv2d(9, 32, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Flatten(), torch.nn.Linear(32 * 13 * 13, 64),
    ).to(dev).eval()
    cm = torch.compile(m, dynamic=True)
    x = torch.randn(45, 9, 13, 13, device=dev)
    with torch.inference_mode():
        out = cm(x)                               # forces compile + execution
    torch.cuda.synchronize()
    assert out.shape == (45, 64), out.shape

if FAILED:
    print("\n  Failed checks: " + ", ".join(FAILED))
    print("  (each FAILED line above states which check and why it matters.)")
    print("\n=== INSTANCE FAILED VERIFICATION — destroy this instance and re-rent ===")
    sys.exit(1)
print("\n=== ALL CHECKS PASSED — ready for benchmarks ===")
PYGATE

if "$PY" "$GATE_PY"; then
  rm -f "$GATE_PY"
  stage_done 5
else
  rm -f "$GATE_PY"
  # The gate already printed which check failed, why, and the destroy message.
  exit 1
fi

# ── STAGE 6: Next steps (benchmarks BEFORE training) ─────────────────────────
stage 6 "Next steps — run these IN ORDER (benchmarks before training)"
cat <<EOF

  Repo:   $REPO_DIR
  Python: $($PY --version 2>&1)

  1) torch.compile A/B  (decides whether to pass --compile)
       cd "$REPO_DIR"
       python -m games.kingdomino.bench_compile --device cuda --sims 200 --games 20
     -> read the final 'USE --compile' / 'DO NOT USE --compile' verdict.

  2) double-buffer A/B  (decides whether to pass --double_buffer)
       python -m games.kingdomino.bench_doublebuffer --device cuda --sims 200 --games 50 --channels 80 --blocks 6
     -> read the final 'USE --double_buffer' / 'DO NOT USE --double_buffer' verdict.

  3) batch_slots sweep  (pick the best N for the 5090; keep leaf_batch=6 fixed)
       for N in 32 48 64 96; do
         echo "=== batch_slots=\$N ==="
         python -m games.kingdomino.self_play --engine batched_open_loop --device cuda \\
           --channels 80 --blocks 6 --leaf_batch 6 --batch_slots \$N \\
           --iterations 1 --games_per_iter 30 --benchmark_every 0 --elo_every 0
       done
     -> pick the N with the highest games/s.

  4) CHECK the three outputs above, THEN launch the 80ch/6b cold-start training
     run (see games/kingdomino/CLOUD_RUN.md Phase 4). Add --compile (and
     --compile_dynamic on) / --double_buffer ONLY if steps 1-2 said USE, and
     --batch_slots <N> from step 3. Do not skip the benchmarks.

EOF
stage_done 6
ok "Setup complete."
