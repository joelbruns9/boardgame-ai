#!/usr/bin/env bash
# =============================================================================
# setup_cloud_common.sh — shared first-login setup for a rented GPU box.
#
# W6.1. Sourced, never executed. Every stage that is the same for Kingdomino and
# 7 Wonders Duel lives here exactly once; the per-game scripts supply what is
# genuinely different — the crate to build, the smoke to run, and the command to
# launch — and call these in order.
#
# The three traps this file exists to hold, all of which cost a rental before:
#   1. cu128 torch must be installed BEFORE requirements.txt, and the torch
#      lines must be stripped from it, or pip quietly downgrades to a cu126
#      wheel with no sm_120 kernels: imports fine, every forward dies.
#   2. Rust must come from rustup, not apt — Cargo edition 2024 needs >= 1.85.
#   3. The GPU gate must HARD FAIL. A box that cannot run a forward pass should
#      cost thirty seconds, not a night of training that dies at the first gate.
#
# Contract for a per-game script:
#   REPO_URL, REPO_DIR, CU128_INDEX   — set by this file's defaults, override
#                                       before sourcing if needed
#   common::require_python            — resolve $PY
#   common::rust_toolchain            — rustup >= 1.85 on PATH
#   common::clone_repo <sentinel...>  — clone/update, verify the checkout
#   common::python_deps               — cu128 torch first, then requirements
#   common::build_crate <dir> <module>— maturin develop --release + import check
#   common::gpu_gate                  — hard-fail device verification
#   common::launch_detached <log> ... — nohup + survives-SSH launch
# =============================================================================

REPO_URL="${REPO_URL:-https://github.com/joelbruns9/boardgame-ai.git}"
REPO_DIR="${REPO_DIR:-$HOME/boardgame-ai}"
CU128_INDEX="${CU128_INDEX:-https://download.pytorch.org/whl/cu128}"
RUST_MIN_VERSION="${RUST_MIN_VERSION:-1.85.0}"

log()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[OK]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[FATAL]\033[0m %s\n' "$*" >&2; exit 1; }

stage()      { printf '\n\033[1;36m=== STAGE %s: %s ===\033[0m\n' "$1" "$2"; }
stage_done() { printf '\033[1;32m=== STAGE %s COMPLETE ===\033[0m\n' "$1"; }

# common::quietly <logfile> <label> -- <command...>
# Run a verbose stage with its output in a file rather than on the terminal.
#
# The stage banners are the navigation: they say which check passed and which
# refused, and a stage that prints thousands of lines scrolls the ones before it
# out of reach. On failure the tail is printed, because a failure nobody can see
# is worse than noise -- and the full log is always on disk.
common::quietly() {
  local logfile="$1" label="$2"; shift 2
  [ "$1" = "--" ] && shift
  mkdir -p "$(dirname "$logfile")"
  if [ "${VERBOSE_STAGES:-0}" = "1" ]; then
    "$@" | tee "$logfile"
    return "${PIPESTATUS[0]}"
  fi
  echo "  $label (output -> $logfile)"
  if "$@" >"$logfile" 2>&1; then
    echo "  $label: ok, $(wc -l < "$logfile" | tr -d ' ') lines"
    return 0
  fi
  local status=$?
  warn "$label FAILED (exit $status). Last 40 lines of $logfile:"
  tail -n 40 "$logfile" >&2
  return "$status"
}

common::require_python() {
  PY="${PYTHON:-python3}"
  command -v "$PY" >/dev/null 2>&1 || PY=python
  command -v "$PY" >/dev/null 2>&1 || die "No python3/python on PATH."
  ok "python: $("$PY" --version 2>&1)"
}

# mimalloc compiles a bundled C library, so the crate no longer builds with the
# Rust toolchain alone. On a minimal image the absence of `cc` surfaces as an
# opaque linker error inside `cargo build` at the crate stage -- after rustup and
# a multi-gigabyte torch install have already run, on a box billed by the hour.
# Check it before any of that.
common::native_build_deps() {
  if command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1; then
    ok "C toolchain present: $( (cc --version 2>/dev/null || gcc --version) | head -n1 )"
    return 0
  fi
  warn "No C compiler on PATH; mimalloc cannot build. Installing build-essential."
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq >/dev/null 2>&1 || true
    apt-get install -y -qq build-essential >/dev/null 2>&1 || true
  fi
  if command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1; then
    ok "C toolchain installed: $( (cc --version 2>/dev/null || gcc --version) | head -n1 )"
    return 0
  fi
  die "mimalloc requires a C compiler (cc or gcc) and none could be installed."
}

common::rust_toolchain() {
  if ! command -v cargo >/dev/null 2>&1; then
    log "Installing Rust via rustup (apt's rustc is too old for edition 2024)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  else
    ok "cargo already present: $(cargo --version)"
  fi
  # shellcheck disable=SC1091
  [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
  command -v cargo >/dev/null 2>&1 || die "cargo still not on PATH after rustup."
  common::native_build_deps
  if ! command -v rustup >/dev/null 2>&1; then
    warn "cargo exists but rustup is missing; installing rustup so Rust can be updated."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
  fi
  local version
  version="$(rustc --version | awk '{print $2}')"
  if [ "$(printf '%s\n' "$RUST_MIN_VERSION" "$version" | sort -V | head -n1)" != "$RUST_MIN_VERSION" ]; then
    warn "rustc $version is older than $RUST_MIN_VERSION; updating stable."
    rustup default stable
    rustup update stable
  fi
  version="$(rustc --version | awk '{print $2}')"
  [ "$(printf '%s\n' "$RUST_MIN_VERSION" "$version" | sort -V | head -n1)" = "$RUST_MIN_VERSION" ] \
    || die "rustc $version is still older than the required $RUST_MIN_VERSION."
  ok "rustc $version"
}

# common::clone_repo <path-that-must-exist> [more paths...]
# Idempotent: an existing checkout is fast-forwarded, so re-running the script
# is the documented way to resume a run.
common::clone_repo() {
  # REPO_BRANCH is empty by default, which takes the remote's default branch.
  # Set it when the code being launched has NOT been merged. A plain clone lands
  # on main; the sentinel check below still passes, because the files it looks
  # for exist there too; and the box then builds and launches code missing every
  # flag the operator believes they are running.
  local _branch_args=()
  [ -n "${REPO_BRANCH:-}" ] && _branch_args=(--branch "$REPO_BRANCH")
  if [ -d "$REPO_DIR/.git" ]; then
    ok "Repo already present; updating with git pull."
    cd "$REPO_DIR"
    if [ -n "${REPO_BRANCH:-}" ]; then
      git fetch origin "$REPO_BRANCH" &&
        git checkout "$REPO_BRANCH" ||
        die "Could not check out '$REPO_BRANCH' — is it pushed?"
    fi
    git pull --ff-only || warn "git pull failed; continuing with the existing checkout."
  else
    if git clone "${_branch_args[@]}" "$REPO_URL" "$REPO_DIR"; then
      ok "Cloned $REPO_URL${REPO_BRANCH:+ (branch $REPO_BRANCH)}"
    else
      warn "Public clone failed — the repo may be private right now."
      if [ -t 0 ]; then
        read -r -p "GitHub username: " GH_USER
        read -r -s -p "GitHub personal access token (input hidden): " GH_TOKEN; echo
        git clone "${_branch_args[@]}" \
          "https://${GH_USER}:${GH_TOKEN}@github.com/joelbruns9/boardgame-ai.git" \
          "$REPO_DIR" || die "Authenticated clone failed."
        ok "Cloned with token."
      else
        die "No TTY to prompt for a token. Clone manually then re-run this script."
      fi
    fi
    cd "$REPO_DIR"
  fi
  local required
  for required in "$@"; do
    [ -e "$required" ] || die "Expected '$required' in the checkout — wrong repo or unpushed code?"
  done
}

common::python_deps() {
  "$PY" -m pip install --upgrade pip >/dev/null
  log "Installing torch + torchvision from the cu128 index (sm_120 kernels)"
  "$PY" -m pip install torch torchvision --index-url "$CU128_INDEX"
  if [ -f requirements.txt ]; then
    # Strip torch lines so requirements.txt can never pull a cu126 wheel over
    # the cu128 build just installed. Comments are preserved.
    local stripped
    stripped="$(mktemp)"
    grep -viE '^[[:space:]]*(torch|torchvision|torchaudio)([[:space:]]|$|[=<>~!])' \
      requirements.txt > "$stripped" || true
    log "Installing the rest of requirements.txt (torch lines stripped)"
    "$PY" -m pip install -r "$stripped"
    rm -f "$stripped"
    ok "requirements.txt installed; GPU wheels untouched."
  else
    warn "No requirements.txt; installing minimal build deps."
    "$PY" -m pip install numpy "maturin>=1.5" "pytest>=7.0"
  fi
}

# common::build_crate <crate-dir> <python-module-name>
common::build_crate() {
  local crate_dir="$1" module="$2"
  [ -d "$crate_dir" ] || die "Crate dir '$crate_dir' missing."
  command -v maturin >/dev/null 2>&1 || "$PY" -m pip install "maturin>=1.5"
  (
    cd "$crate_dir"
    maturin develop --release
  ) || die "maturin develop failed for $crate_dir."
  "$PY" -c "import $module; print('$module import OK')" \
    || die "$module failed to import after build."
  ok "Rust crate $module built and importable."
}

common::gpu_gate() {
  local gate_py
  gate_py="$(mktemp --suffix=.py)"
  cat > "$gate_py" <<'PYGATE'
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

@check("CUDA forward pass (TransformerEncoderLayer, the shipped model family)",
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
print("\n=== ALL CHECKS PASSED ===")
PYGATE
  if "$PY" "$gate_py"; then
    rm -f "$gate_py"
  else
    rm -f "$gate_py"
    exit 1
  fi
}

# common::launch_detached <log-file> <command...>
# nohup + disown so the run survives the SSH session, then a liveness check so a
# process that dies on its first line is reported here instead of looking like a
# successful launch.
common::launch_detached() {
  local log_file="$1"; shift
  mkdir -p "$(dirname "$log_file")"
  nohup "$@" >> "$log_file" 2>&1 &
  local pid=$!
  disown "$pid" 2>/dev/null || true
  sleep 5
  kill -0 "$pid" 2>/dev/null || die "Process died within 5s — check $log_file"
  ok "Launched pid=$pid; log: $log_file"
  LAUNCHED_PID="$pid"
}
