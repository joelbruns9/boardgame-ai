# Cloud Run Runbook — Kingdomino AlphaZero (80ch/6b, RTX 5090)

Goal: stand up a Vast.ai RTX 5090 instance and run a **cold-start** training
run of an **80-channel / 6-block** network, while preserving the shared Elo
ladder.

This runbook consolidates the original cloud plan plus the decisions and
caveats worked out in discussion. Work top to bottom: everything in
**Phase 0** happens locally *before* renting an instance.

> Architecture decision: **cold start at 80ch/6b.** We do NOT `--warm_start`
> from `best_32x4.pt` — the tensor shapes don't match (32→80 channels, 4→6
> blocks), so weight transfer is impossible. We start from random weights.
> The prior buffer can still *optionally* seed self-play (samples are
> architecture-independent), but they come from a weaker policy — see Phase 4.

---

## Phase 0 — Local preparation (before renting anything)

### 0.1 Decide and pin the Blackwell-capable PyTorch wheel
The RTX 5090 is **Blackwell**, compute capability **sm_120 / `(12, 0)`**.
Our local build is `torch==2.12.0+cu126`, which **does not ship sm_120
kernels** — on the 5090 it fails with `no kernel image available for
execution on the device` (or silently falls back). torch.compile/Triton
breaks first, but plain forwards can fail too.

- [ ] Confirm the exact torch version available at
      `https://download.pytorch.org/whl/cu128` (or newer cuXXX). Record the
      version to pin.
- [ ] Plan to install it on cloud via:
      `pip install torch --index-url https://download.pytorch.org/whl/cu128`
- [ ] Do **NOT** reuse a frozen `requirements.txt` that pins `+cu126`.
      Triton ships *inside* the Linux cu128 wheel — no separate install.

### 0.2 Instrument `bench_compile.py` for the real workload
`torch.compile` is **not testable on Windows** (Triton unavailable), so both
the speedup *and* the recompilation behavior can only be checked on cloud.
The filled inference batch **varies tick-to-tick** (mean ~45 at leaf_batch=6,
not a fixed `batch_slots × leaf_batch`), which can trigger recompilation
storms that silently eat the expected 1.2–1.5× gain.

- [ ] Edit `bench_compile.py` to feed **realistic variable batch sizes**.
- [ ] Run it on cloud with `TORCH_LOGS=recompiles` set, and try `dynamic=True`.
      This must be the *first* thing run on the GPU.

### 0.3 Push latest code
- [ ] `git push origin main` — the cloud box clones from GitHub.

### 0.4 Confirm what transfers via git vs. scp
- `best_checkpoint/`, `elo_db.json`, `elo_games.jsonl` are **committed** →
  arrive automatically with `git clone`. No scp needed for the Elo ladder.
- For a cold start we need **no weights**.
- **Optional:** `runs/kingdomino/<prev_run>/buffer_final.pkl` (gitignored,
  `*.pkl`) — scp it over only if seeding the buffer (Phase 4).

---

## Phase 1 — Instance selection (Vast.ai)

- [ ] GPU = **RTX 5090**.
- [ ] Host **driver ≥ R570 / CUDA ≥ 12.8** (Blackwell requirement). The host
      driver is fixed by the provider — you cannot upgrade it on a rental, so
      filter for it up front.
- [ ] Prefer a base image with CUDA 12.8+, e.g.
      `pytorch/pytorch:*-cuda12.8-cudnn9-*` or `nvidia/cuda:12.8.*`.
- [ ] **Cost guardrail:** set a max duration / max spend on the instance
      *before* starting. Estimate hours from games/s after the benchmark
      sweep (Phase 3).

---

## Phase 2 — First-login setup (`setup_cloud.sh`)

Create `setup_cloud.sh` to automate first-login setup. It must:

1. **Install Rust via `rustup`** (NOT `apt install rustc`). `Cargo.toml` is
   `edition = "2024"`, which requires **Rust ≥ 1.85**; distro packages are
   far too old.
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
   source "$HOME/.cargo/env"
   ```
2. **Clone the repo** from GitHub.
3. **Install Python deps**, installing torch from the **cu128** index (0.1),
   plus the rest (numpy, maturin, etc.).
4. **Build the Rust crate**: `maturin develop --release` (in
   `games/kingdomino/kingdomino_rust`).
5. **Run the GPU verification gate** (Phase 2.1) — fail fast if wrong.
6. **Run the throughput benchmark sequence** (Phase 3).

### 2.1 GPU verification gate (hard fail before training)
Bake this into `setup_cloud.sh` so a wrong wheel/driver fails in ~30s instead
of three iterations into a paid run:

- [ ] `torch.cuda.is_available()` → `True`
- [ ] `torch.cuda.get_device_capability(0)` → `(12, 0)`
- [ ] `'sm_120' in torch.cuda.get_arch_list()`
- [ ] a tiny CUDA conv + matmul runs with **no** kernel-image error
- [ ] `import triton` succeeds AND one `torch.compile`'d forward actually
      executes (not just imports)

---

## Phase 3 — Throughput benchmarks (run before training)

Run immediately after the verification gate. Results decide `--batch_slots`,
`--compile`, and `--double_buffer` for the training command.

- [ ] **`bench_compile.py`** — confirm torch.compile gain on the 5090, with
      variable batch sizes and `TORCH_LOGS=recompiles` (Phase 0.2). Only pass
      `--compile` to training if this shows a real, recompile-free win.
      `python -m games.kingdomino.bench_compile --device cuda --sims 200 --games 20`
- [ ] **`bench_doublebuffer.py`** — was −8.5% on the RTX 3070 (GPU-bound).
      The faster GPU may flip the bottleneck to CPU; re-test. Pass
      `--double_buffer` to training only if it now helps.
- [ ] **Profile the eval breakdown** — `eval_h2d_sec / eval_forward_sec /
      eval_readback_sec` to see whether CPU has become the bottleneck.
- [ ] **`batch_slots` re-sweep** — 32 was optimal on the RTX 3070 at 32ch/4b;
      a faster GPU wants more concurrent slots to stay fed. Time short runs at
      `batch_slots ∈ {32, 48, 64, 96}`, compare games/s, pick the best:
      `python -m games.kingdomino.self_play --engine batched_open_loop --device cuda --channels 80 --blocks 6 --leaf_batch 6 --batch_slots <N> --iterations 1 --games_per_iter 30 --benchmark_every 0 --elo_every 0`

> **Keep `--leaf_batch 6` fixed.** It is an *accuracy* ceiling (highest lb
> before search quality degrades), not a throughput knob — do NOT sweep it up
> on faster hardware. Keep `--virtual_loss 1` pinned to it as well.
> `batch_slots` is the only pure-throughput knob (it batches independent
> games, zero search-quality effect).

---

## Phase 4 — Training command (80ch/6b cold start)

Linux shell, single line. Substitute `<batch_slots>` from the Phase 3 sweep
and add `--compile` / `--double_buffer` only if Phase 3 confirmed them.

```bash
python -m games.kingdomino.self_play \
  --engine batched_open_loop \
  --device cuda \
  --iterations 60 \
  --games_per_iter 300 \
  --train_steps 1200 \
  --sims 1600 \
  --channels 80 --blocks 6 \
  --batch_slots <batch_slots> --leaf_batch 6 --virtual_loss 1 \
  --batch_size 256 \
  --lr 1e-3 --weight_decay 1e-4 \
  --buffer 500000 \
  --lambda_score 0.5 --lambda_w 0.25 --score_scale 160.0 \
  --policy_weight 1.0 --grad_clip 1.0 --margin_gain 2.0 \
  --alpha 0.8 --c_puct 1.5 --temp_moves 20 --fpu -0.2 \
  --benchmark_every 10 --benchmark_sims 50 --benchmark_seeds 20 \
  --checkpoint_dir runs/kingdomino/cloud_80x6_run1 \
  --save_buffer runs/kingdomino/cloud_80x6_run1/buffer_final.pkl \
  --elo_every 10 --elo_sims 400 --elo_games_per_anchor 40 \
  --elo_db elo_db.json --elo_games_log elo_games.jsonl \
  --seed 0
```

Notes that differ from a warm-start laptop run:

- **No `--warm_start`** (cold start — random weights at 80ch/6b).
- **`--lr 1e-3`, not 3e-4.** 3e-4 was the *fine-tuning* rate for a plateaued
  32x4 net. A fresh net should start at the standard AlphaZero 1e-3, then drop
  to 3e-4 (and later 1e-4) once policy loss plateaus.
- **`--alpha 0.8`** early (margin is the reliable signal while the win head is
  uncalibrated) → switch to **`--alpha 0.0`** for mature play. Switch on the
  **signal, not a fixed iteration**: watch `win_brier` flatten (the doc's
  ~iter 50 is a warm-run estimate; a cold 80x6 net may calibrate later).
  To switch: warm-start from the chosen checkpoint with `--alpha 0.0`.
- **Add `--compile` only if Phase 3 confirmed it** (and add
  `--double_buffer` likewise).
- **Optional buffer seed:** add `--warm_buffer runs/kingdomino/<prev_run>/buffer_final.pkl`
  to skip the empty-buffer cold period. Samples are architecture-independent,
  but come from a weaker 32x4 policy — acceptable as early filler, not as
  long-term signal.

### 4.1 Replay-ratio sanity (two independent ratios)
With `batch_size=256`, `games_per_iter=300`, ~80 positions/game:

- **Staleness window** = `buffer / (gpi × pos_per_game)` = `500k / 24k` ≈
  **21 iterations** of history. (Matches the 20–30 target.)
- **Sample reuse** = `train_steps × batch_size / (gpi × pos_per_game)` =
  `1200 × 256 / 24k` ≈ **12.8×** gradient updates per fresh position.

These are **independent knobs**: buffer size sets staleness; `train_steps`
sets reuse. The 12.8× reuse is the same regime the laptop runs use (it's what
the "train_steps ≈ 4× games_per_iter" rule encodes), so it's validated — but
it's high in absolute terms (classic AZ is ~1–4×) and bites *harder* on a cold
start, where a random net emits noisy MCTS labels.

- [ ] **Consider ramping `train_steps`** over the first ~10–15 iters (e.g.
      start ~400, climb to 1200) rather than 1200 from iter 1, to avoid
      entrenching early noise. Phased: run a short low-`train_steps` segment,
      then warm-start the next segment with higher `train_steps`.
- [ ] **Verify `samples_per_iter`** from the iter-0 training log — confirm
      pos/game really is ~80 at 80x6 (game length is rules-bound, so it should
      hold).

---

## Phase 5 — Mid-run checkpoint sync (don't lose the run)

Vast.ai instances are interruptible/ephemeral. The cost guardrail protects
your wallet, not your run — if the box dies at iter 40 with no off-box copy,
the run is gone.

- [ ] Periodically pull checkpoints + `training_log.jsonl` **off** the instance
      (every ~10 iters), e.g. from your local machine:
      ```bash
      rsync -avz user@cloud:~/boardgame-ai/runs/kingdomino/cloud_80x6_run1/ \
        runs/kingdomino/cloud_80x6_run1/
      ```
      (or scp to a durable bucket). `--save_buffer` only writes at the end, so
      the buffer is *not* protected mid-run — checkpoints are what matter here.

---

## Phase 6 — After the run

Follows `WORKFLOW.md` "After a Cloud Run":

1. **Merge Elo data.** Download the cloud `elo_games.jsonl`, append to local,
   **deduplicate** (key on the per-line fields: checkpoint, opponent, seed,
   orientation, timestamp — `elo_games.jsonl` is already ~1.5 MB and growing,
   so a concrete dedup key matters), then re-solve:
   ```bash
   python -m games.kingdomino.elo_rating --resolve --verbose
   ```
2. **Download the best checkpoint** if improved. Note it's 80x6 now, so it is
   *not* a drop-in replacement for `best_checkpoint/best_32x4.pt` — store it
   alongside (e.g. `best_80x6.pt`) and update `best_checkpoint/README.md`.
3. **Update the Elo anchor pool** if the new model rates >150 Elo above the
   current top anchor (`--reanchor`).
4. **Commit:**
   ```bash
   git add elo_db.json elo_games.jsonl games/kingdomino/best_checkpoint/
   git commit -m "Cloud run cloud_80x6_run1: <summary>"
   git push origin main
   ```

---

## Quick checklist

- [ ] 0.1 cu128 torch wheel chosen & pinned
- [ ] 0.2 `bench_compile.py` instrumented for variable shapes
- [ ] 0.3 code pushed
- [ ] 1 instance: RTX 5090, driver ≥ 570, cost cap set
- [ ] 2 `setup_cloud.sh` written (rustup, clone, cu128 deps, maturin, gate, benches)
- [ ] 2.1 GPU verification gate passes
- [ ] 3 benchmarks run; `batch_slots` chosen; `--compile`/`--double_buffer` decided
- [ ] 4 training launched (cold start, lr 1e-3, alpha 0.8, leaf_batch 6)
- [ ] 4.1 replay ratios sanity-checked; train_steps ramp considered
- [ ] 5 mid-run checkpoint sync running
- [ ] 6 Elo merged, checkpoint saved, committed
```
