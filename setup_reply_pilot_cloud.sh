#!/usr/bin/env bash
# setup_reply_pilot_cloud.sh
#
# Pilot-specific copy of setup_cloud.sh. The original known-good setup script
# is intentionally unchanged. This script prepares and validates a Blackwell
# RTX 5080/5090 Linux instance for the Kingdomino opponent-reply pilot.
#
# It installs/builds dependencies, verifies immutable inputs, runs tests, and
# runs one production-shape equivalence benchmark. It never generates labels,
# starts training, promotes a checkpoint, or modifies current_best.
#
# Fresh tracked checkout:
#   REPO_REF=codex/reply-pilot EXPECTED_COMMIT=<40-char-sha> \
#     bash setup_reply_pilot_cloud.sh
#
# Already-synchronized checkout (for example, rsync/scp before setup):
#   SKIP_GIT_UPDATE=1 bash setup_reply_pilot_cloud.sh
#
# If artifacts will be copied after environment setup:
#   REQUIRE_PILOT_ARTIFACTS=0 RUN_PILOT_BENCHMARK=0 \
#     SKIP_GIT_UPDATE=1 bash setup_reply_pilot_cloud.sh
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/joelbruns9/boardgame-ai.git}"
REPO_DIR="${REPO_DIR:-$HOME/boardgame-ai}"
REPO_REF="${REPO_REF:-main}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
SKIP_GIT_UPDATE="${SKIP_GIT_UPDATE:-0}"

CRATE_DIR_REL="games/kingdomino/kingdomino_rust"
CU128_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
MIN_DRIVER="${MIN_DRIVER:-570}"
REQUIRE_TORCH_COMPILE="${REQUIRE_TORCH_COMPILE:-0}"

REQUIRE_PILOT_ARTIFACTS="${REQUIRE_PILOT_ARTIFACTS:-1}"
RUN_PILOT_TESTS="${RUN_PILOT_TESTS:-1}"
RUN_PILOT_BENCHMARK="${RUN_PILOT_BENCHMARK:-1}"
PILOT_THREADS="${PILOT_THREADS:-1,2,4,8}"
PILOT_DIR_REL="${PILOT_DIR_REL:-runs/kingdomino/reply_pilot/cloud}"

BASE_CKPT_REL="runs/kingdomino/best_checkpoint/current_best.pt"
BASE_CKPT_SHA256="4bf07b0ca14e5452e6533a9232967e89bb0ab0df88c99e9928a65f402b1f04b3"
REPLAY_BUFFER_REL="runs/kingdomino/cloud_80x6_run10/buffer_final.pkl"
REPLAY_BUFFER_SHA256="1ea7a6dd0c48caf2aeca5604d1e2cded762639c7dabf8dc6ea69acd832c7051b"
FROZEN_POSITIONS_REL="runs/kingdomino/denial_search/signal_positions.jsonl"
FROZEN_POSITIONS_SHA256="d38ebb5f1e0430f2f78dfaa7998a1cb48ece0584b27ebb42ba8d144737768f9a"
REFERENCE_DIR_REL="runs/kingdomino/denial_search/secondary_seed"
REFERENCE_SEED_20260717_SHA256="a4d21498bf4d645fc68e6f76fc25a7fb78948a10647aadca1e9d75d36b6bf2da"
REFERENCE_SEED_21260720_SHA256="8735e094a0155302ebe4353a9d55fa26cc21ac24ec28f5fb1571da8dc98cc02b"
REFERENCE_SEED_22260723_SHA256="ae4d5e67ab56b2fd8ed3c9d77bd498a0557cb37b5c70bb52f6b5090dc1c9dbe9"

log()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[OK]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[FATAL]\033[0m %s\n' "$*" >&2; exit 1; }
stage()      { printf '\n\033[1;36m=== STAGE %s: %s ===\033[0m\n' "$1" "$2"; }
stage_done() { printf '\033[1;32m=== STAGE %s COMPLETE ===\033[0m\n' "$1"; }
verify_sha() {
  local relative="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$relative" | awk '{print $1}')"
  [ "$actual" = "$expected" ] || die "SHA-256 mismatch for $relative: $actual"
}

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
command -v "$PY" >/dev/null 2>&1 || die "No python3/python on PATH."
PY="$(command -v "$PY")"

stage 1 "Rust toolchain (rustup, Rust >= 1.85)"
if ! command -v cargo >/dev/null 2>&1; then
  command -v curl >/dev/null 2>&1 || die "curl is required to install Rust."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
command -v cargo >/dev/null 2>&1 || die "cargo is not on PATH after Rust setup."
if ! command -v rustup >/dev/null 2>&1; then
  command -v curl >/dev/null 2>&1 || die "curl is required to install rustup."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1091
  [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
fi
RUST_VERSION="$(rustc --version | awk '{print $2}')"
if [ "$(printf '%s\n' "1.85.0" "$RUST_VERSION" | sort -V | head -n1)" != "1.85.0" ]; then
  rustup default stable
  rustup update stable
fi
RUST_VERSION="$(rustc --version | awk '{print $2}')"
[ "$(printf '%s\n' "1.85.0" "$RUST_VERSION" | sort -V | head -n1)" = "1.85.0" ] \
  || die "rustc $RUST_VERSION is older than required 1.85.0."
ok "$(rustc --version)"
stage_done 1

stage 2 "Pilot source checkout and provenance"
command -v git >/dev/null 2>&1 || die "git is required."
if [ -d "$REPO_DIR/.git" ]; then
  cd "$REPO_DIR"
  if [ "$SKIP_GIT_UPDATE" = "1" ]; then
    warn "SKIP_GIT_UPDATE=1; using the pre-synchronized checkout."
  else
    [ -n "$EXPECTED_COMMIT" ] \
      || die "Set EXPECTED_COMMIT for a paid run, or explicitly use SKIP_GIT_UPDATE=1."
    git fetch --tags origin || die "git fetch failed; refusing stale source."
    git checkout "$REPO_REF" || die "could not check out REPO_REF=$REPO_REF"
    git pull --ff-only origin "$REPO_REF" \
      || die "fast-forward pull failed; refusing stale source."
  fi
else
  [ "$SKIP_GIT_UPDATE" != "1" ] \
    || die "SKIP_GIT_UPDATE=1 but $REPO_DIR is not a Git checkout."
  [ -n "$EXPECTED_COMMIT" ] \
    || die "Set EXPECTED_COMMIT before cloning source for a paid run."
  git clone --branch "$REPO_REF" "$REPO_URL" "$REPO_DIR" \
    || die "clone failed; clone a private repo securely before rerunning."
  cd "$REPO_DIR"
fi
[ -d "$CRATE_DIR_REL" ] || die "Expected crate directory is missing: $CRATE_DIR_REL"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [ -n "$EXPECTED_COMMIT" ] && [ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]; then
  die "source commit $ACTUAL_COMMIT != EXPECTED_COMMIT $EXPECTED_COMMIT"
fi
if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
  if [ "$SKIP_GIT_UPDATE" = "1" ]; then
    warn "Tracked source has local modifications; tests and implementation hashes remain mandatory."
  else
    die "tracked source is dirty after checkout; refusing ambiguous provenance."
  fi
fi
ok "Source commit: $ACTUAL_COMMIT"
stage_done 2

stage 3 "Pinned Python/CUDA dependencies"
"$PY" -m pip install --upgrade pip >/dev/null
log "Installing torch $TORCH_VERSION + torchvision $TORCHVISION_VERSION from cu128"
"$PY" -m pip install \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
  --index-url "$CU128_INDEX"
if [ -f requirements.txt ]; then
  REQ_NOGPU="$(mktemp)"
  trap 'rm -f "${REQ_NOGPU:-}" "${GATE_PY:-}"' EXIT
  grep -viE '^[[:space:]]*(torch|torchvision|torchaudio)([[:space:]]|$|[=<>~!])' \
    requirements.txt > "$REQ_NOGPU" || true
  "$PY" -m pip install -r "$REQ_NOGPU"
else
  "$PY" -m pip install "numpy>=1.24" "maturin>=1.5"
fi
"$PY" -m pip install "pytest>=8,<10"
"$PY" -c "import pytest; print('pytest', pytest.__version__)"
stage_done 3

stage 4 "Release build of kingdomino_rust"
"$PY" -m pip show maturin >/dev/null 2>&1 || "$PY" -m pip install "maturin>=1.5"
(
  cd "$CRATE_DIR_REL"
  "$PY" -m maturin develop --release
)
"$PY" -c "import kingdomino_rust; assert hasattr(kingdomino_rust, 'denial_forced_tree'); print('kingdomino_rust denial tree OK')" \
  || die "release Rust extension is missing denial_forced_tree."
stage_done 4

stage 5 "Blackwell CUDA verification gate"
GATE_PY="$(mktemp --suffix=.py)"
cat > "$GATE_PY" <<'PYGATE'
import os
import subprocess
import sys

FAILED = []

def check(name, why):
    def deco(fn):
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            print(f"         why it matters: {why}")
            FAILED.append(name)
    return deco

@check("NVIDIA driver minimum", "Blackwell/CUDA 12.8 needs a sufficiently new host driver.")
def _driver():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()[0].strip()
    assert int(out.split(".")[0]) >= int(os.environ["MIN_DRIVER"]), out
    print(f"         driver_version={out}")

import torch  # noqa: E402
print(f"  torch={torch.__version__}")

@check("pinned torch version", "the environment must match the validated dependency pair.")
def _version():
    actual = torch.__version__.split("+")[0]
    assert actual == os.environ["EXPECTED_TORCH_VERSION"], actual

@check("CUDA is available", "a CPU-only or hidden-GPU environment cannot run the pilot.")
def _available():
    assert torch.cuda.is_available()

@check("Blackwell capability sm_120", "the rental must be an RTX 5080/5090-class Blackwell GPU.")
def _capability():
    cap = torch.cuda.get_device_capability(0)
    assert cap == (12, 0), cap
    print(f"         gpu={torch.cuda.get_device_name(0)} capability={cap}")

@check("torch wheel contains sm_120", "a wheel without sm_120 kernels will fail at execution.")
def _arch():
    arches = torch.cuda.get_arch_list()
    assert "sm_120" in arches, arches

@check("real CUDA convolution and matrix multiply", "imports alone do not prove kernels execute.")
def _forward():
    x = torch.randn(45, 9, 13, 13, device="cuda")
    conv = torch.nn.Conv2d(9, 32, 3, padding=1).cuda()
    y = conv(x)
    z = y.flatten(1) @ torch.randn(y[0].numel(), 128, device="cuda")
    torch.cuda.synchronize()
    assert z.shape == (45, 128)

@check("optional torch.compile execution", "only required when REQUIRE_TORCH_COMPILE=1.")
def _compiled():
    if os.environ.get("REQUIRE_TORCH_COMPILE") != "1":
        print("         skipped: reply pilot uses eager inference")
        return
    import triton  # noqa: F401
    model = torch.nn.Sequential(
        torch.nn.Conv2d(9, 32, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Flatten(), torch.nn.Linear(32 * 13 * 13, 64),
    ).cuda().eval()
    compiled = torch.compile(model, dynamic=True)
    with torch.inference_mode():
        out = compiled(torch.randn(45, 9, 13, 13, device="cuda"))
    torch.cuda.synchronize()
    assert out.shape == (45, 64)

if FAILED:
    print("\nFailed checks: " + ", ".join(FAILED))
    sys.exit(1)
print("\nAll required CUDA checks passed.")
PYGATE
MIN_DRIVER="$MIN_DRIVER" EXPECTED_TORCH_VERSION="$TORCH_VERSION" \
REQUIRE_TORCH_COMPILE="$REQUIRE_TORCH_COMPILE" "$PY" "$GATE_PY" \
  || die "CUDA verification failed; do not start paid pilot work."
stage_done 5

stage 6 "Reply-pilot source and immutable artifacts"
cd "$REPO_DIR"
PILOT_DIR="$REPO_DIR/$PILOT_DIR_REL"
mkdir -p "$PILOT_DIR"
required_source=(
  "games/kingdomino/reply_pilot.py"
  "games/kingdomino/reply_training.py"
  "games/kingdomino/reply_pilot_evaluation.py"
  "games/kingdomino/kingdomino_rust/src/denial_tree.rs"
  "games/kingdomino/REPLY_PILOT_CLOUD_RUN.md"
  "games/kingdomino/tests/test_reply_pilot.py"
  "games/kingdomino/tests/test_reply_training.py"
  "games/kingdomino/tests/test_reply_pilot_evaluation.py"
  "games/kingdomino/tests/test_rust_denial_tree_equiv.py"
)
for relative in "${required_source[@]}"; do
  [ -s "$relative" ] || die "required pilot source is missing: $relative"
done
PILOT_SOURCE_SHA256="$(sha256sum "${required_source[@]}" | sha256sum | awk '{print $1}')"
ok "Pilot source-bundle SHA-256: $PILOT_SOURCE_SHA256"
"$PY" -c "import games.kingdomino.reply_pilot, games.kingdomino.reply_training, games.kingdomino.reply_pilot_evaluation; print('pilot imports OK')"

required_artifacts=(
  "$BASE_CKPT_REL"
  "$REPLAY_BUFFER_REL"
  "$FROZEN_POSITIONS_REL"
  "$REFERENCE_DIR_REL/tree_seed20260717.jsonl"
  "$REFERENCE_DIR_REL/tree_seed21260720.jsonl"
  "$REFERENCE_DIR_REL/tree_seed22260723.jsonl"
)
ARTIFACTS_READY=1
for relative in "${required_artifacts[@]}"; do
  if [ ! -s "$relative" ]; then
    warn "required pilot artifact is missing: $relative"
    ARTIFACTS_READY=0
  fi
done
if [ "$ARTIFACTS_READY" != "1" ] && [ "$REQUIRE_PILOT_ARTIFACTS" = "1" ]; then
  die "copy all registered pilot artifacts, then rerun this script"
fi
if [ "$ARTIFACTS_READY" = "1" ]; then
  verify_sha "$BASE_CKPT_REL" "$BASE_CKPT_SHA256"
  verify_sha "$REPLAY_BUFFER_REL" "$REPLAY_BUFFER_SHA256"
  verify_sha "$FROZEN_POSITIONS_REL" "$FROZEN_POSITIONS_SHA256"
  verify_sha "$REFERENCE_DIR_REL/tree_seed20260717.jsonl" "$REFERENCE_SEED_20260717_SHA256"
  verify_sha "$REFERENCE_DIR_REL/tree_seed21260720.jsonl" "$REFERENCE_SEED_21260720_SHA256"
  verify_sha "$REFERENCE_DIR_REL/tree_seed22260723.jsonl" "$REFERENCE_SEED_22260723_SHA256"
  "$PY" -c "from games.kingdomino.denial_signal_sweep import load_frozen_positions; p='$FROZEN_POSITIONS_REL'; rows=load_frozen_positions(p); assert len(rows)==50, len(rows); print('frozen positions:', len(rows))"
  ok "Required artifact hashes passed."
else
  warn "Artifact gate explicitly bypassed; production benchmark will be skipped."
fi
if [ ! -d "runs/kingdomino/bga_game_log" ]; then
  warn "BGA logs are absent; copy them before the post-behavior BGA gate."
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
printf 'logical_cpus=%s\n' "$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '?')"
df -h "$REPO_DIR" | tail -n 1
stage_done 6

stage 7 "Focused pilot and shared-Rust tests"
if [ "$RUN_PILOT_TESTS" = "1" ]; then
  "$PY" -m pytest \
    games/kingdomino/tests/test_denial_search.py \
    games/kingdomino/tests/test_secondary_pick_seed_test.py \
    games/kingdomino/tests/test_reply_pilot.py \
    games/kingdomino/tests/test_reply_training.py \
    games/kingdomino/tests/test_reply_pilot_evaluation.py \
    games/kingdomino/tests/test_rust_denial_tree_equiv.py \
    games/kingdomino/tests/test_rust_nnue_features.py \
    games/kingdomino/tests/test_rust_augment.py \
    games/kingdomino/tests/test_parallel_self_play.py -q
else
  warn "RUN_PILOT_TESTS=$RUN_PILOT_TESTS; tests skipped by explicit override."
fi
stage_done 7

stage 8 "Production-shape Python/Rust/Rayon benchmark"
BENCHMARK_OUT="$PILOT_DIR/cloud_engine_benchmark.json"
if [ "$RUN_PILOT_BENCHMARK" = "1" ] && [ "$ARTIFACTS_READY" = "1" ]; then
  "$PY" -m games.kingdomino.reply_pilot \
    --mode benchmark \
    --positions-path "$FROZEN_POSITIONS_REL" \
    --checkpoint "$BASE_CKPT_REL" \
    --pick-plies 8 --chance-k 16 --search-sims 3200 \
    --benchmark-limit 1 --benchmark-threads "$PILOT_THREADS" \
    --output "$BENCHMARK_OUT"
  [ -s "$BENCHMARK_OUT" ] || die "benchmark did not produce $BENCHMARK_OUT"
else
  warn "Pilot benchmark skipped by override or because artifacts are missing."
fi
stage_done 8

SETUP_REPORT="$PILOT_DIR/setup_report.json"
SETUP_REPORT="$SETUP_REPORT" ACTUAL_COMMIT="$ACTUAL_COMMIT" \
ARTIFACTS_READY="$ARTIFACTS_READY" BENCHMARK_OUT="$BENCHMARK_OUT" \
PILOT_SOURCE_SHA256="$PILOT_SOURCE_SHA256" \
"$PY" - <<'PYREPORT'
import json
import os
import platform
import subprocess
from pathlib import Path
import torch

def output(*args):
    return subprocess.check_output(args, text=True).strip()

report = {
    "status": "ready_for_reply_pilot" if os.environ["ARTIFACTS_READY"] == "1" else "setup_only_artifacts_missing",
    "source_commit": os.environ["ACTUAL_COMMIT"],
    "pilot_source_bundle_sha256": os.environ["PILOT_SOURCE_SHA256"],
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "gpu_capability": list(torch.cuda.get_device_capability(0)),
    "rustc": output("rustc", "--version"),
    "cargo": output("cargo", "--version"),
    "artifacts_ready": os.environ["ARTIFACTS_READY"] == "1",
    "benchmark": os.environ["BENCHMARK_OUT"] if Path(os.environ["BENCHMARK_OUT"]).exists() else None,
    "current_best_updated": False,
}
path = Path(os.environ["SETUP_REPORT"])
path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
PYREPORT

cat <<EOF

Reply-pilot setup is complete.

Setup report:
  $SETUP_REPORT

Benchmark:
  $BENCHMARK_OUT

Continue with:
  games/kingdomino/REPLY_PILOT_CLOUD_RUN.md

No labels were generated, no training was started, and current_best was not modified.
EOF
ok "Reply-pilot cloud setup complete."
