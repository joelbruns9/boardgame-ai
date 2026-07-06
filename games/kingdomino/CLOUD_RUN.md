# Cloud Run Runbook — Kingdomino AlphaZero (80x6/96x6, cloud GPU)

Goal: stand up a cloud GPU instance, calibrate the hardware, and run a
**cold-start** training run of an **80-channel / 6-block** or
**96-channel / 6-block** network, while preserving the shared Elo ladder.

This runbook consolidates the original cloud plan plus the decisions and
caveats worked out in discussion. Work top to bottom: everything in
**Phase 0** happens locally *before* renting an instance.

> Architecture decision: **cold start at 80x6 or 96x6.** We do NOT
> `--warm_start` from `best_32x4.pt` — the tensor shapes don't match
> (32→80/96 channels, 4→6 blocks), so weight transfer is impossible. We
> start from random weights.
>
> **This is also the first run on the new 333-flat encoder + win-gated value
> (2026-07).** Every prior checkpoint AND replay buffer is **261-flat** and is
> therefore INCOMPATIBLE — there is no warm-start weight *and* no buffer seed
> available. This run is genuinely from scratch: random weights, empty buffer.
> Two config consequences (details in Phase 4): use a **fixed `--alpha 0.5`**
> (the win-gated margin band B; the old `--alpha_schedule` no longer applies),
> and do **not** pass `--warm_buffer`.
>
> Depth decision: **6 blocks is the default.** Six blocks gives full-board
> receptive-field coverage on the encoded Kingdomino board. Test 8 blocks only
> if the 6-block model later shows clear underfitting; it taxes every MCTS
> inference call.

---

## First cloud run? Plain-English walkthrough (never run a `.sh` before)

This is the whole run in eight concrete steps. It reuses the Phases below — those
have the detail and the reasoning; this is "what do I literally type." You rent a
remote computer with a GPU by the hour, set it up with **one script**, start
training, check on it, copy the results back, and shut it down.

> A `.sh` file is just a plain text file containing shell commands. You don't
> double-click it — you hand it to the `bash` program, which runs the lines in
> order. `bash setup_cloud.sh` always works; you never need to "install" it.

**Step 1 — Rent a GPU box (Vast.ai website, in your browser).**
- Make a Vast.ai account and add a little credit.
- Search/filter for: GPU **RTX 5090**, host **driver ≥ 570** (CUDA ≥ 12.8), and a
  template image with **CUDA 12.8+** (e.g. a `pytorch/pytorch ...cuda12.8...`
  image). See Phase 1.
- **Set a max spend / max duration before you click rent** — this is your money
  guardrail.
- Rent it. Vast shows you an **SSH command** to connect, like
  `ssh -p 12345 root@203.0.113.7`. Copy it.

**Step 2 — Connect to the box (from your Windows laptop).**
- Open **Git Bash** (you already have it) — or PowerShell.
- Paste the SSH command from Vast and press Enter. The first time it asks
  "are you sure you want to continue connecting?" → type `yes`.
- Your prompt is now *on the remote box*. Every command from here runs there, not
  on your laptop, until you type `exit`.

**Step 3 — Set the box up with one script (this is "running a `.sh`").**
```bash
curl -fsSL https://raw.githubusercontent.com/joelbruns9/boardgame-ai/main/setup_cloud.sh -o setup_cloud.sh
bash setup_cloud.sh
```
This installs Rust, clones the repo, installs the correct PyTorch (cu128 for the
5090), builds the Rust engine, runs a **GPU self-check**, then runs the
calibration benchmark. It takes ~15–40 min. Watch for the green `=== STAGE N
COMPLETE ===` banners.
- If it prints **`INSTANCE FAILED VERIFICATION`**, the box's GPU/driver is wrong.
  **Destroy it (Step 8) and rent a different one** — don't try to fix it.
- If it prints **`ALL CHECKS PASSED`** and finishes calibration, you're good.

**Step 4 — Read the calibration result and note your settings.**
```bash
cat ~/boardgame-ai/runs/kingdomino/cloud_calibration/summary.md
```
Write down the recommended `channels` (80 or 96), `batch_slots`, `game_cpus`,
`exact_endgame_max_secs`, and `full_search_fraction` (Phase 3).

**Step 5 — Start training inside `tmux` (so it survives a dropped connection).**
If your SSH drops, anything running normally dies with it. `tmux` keeps it alive.
```bash
tmux new -s train                 # opens a persistent session
cd ~/boardgame-ai
# paste the Phase 4 training command, with your Step-4 numbers filled in.
# (fixed --alpha 0.5, NO --warm_start, NO --warm_buffer — see Phase 4.)
```
Then **detach** (leave it running): press `Ctrl+b`, release, then press `d`.
Reattach anytime with `tmux attach -t train`.

**Step 6 — Watch it (optional).**
```bash
tail -f ~/boardgame-ai/runs/kingdomino/cloud_<channels>x6_run1/training_log.jsonl
nvidia-smi        # GPU utilization
```
`Ctrl+c` stops *watching*, not the training.

**Step 7 — Copy results back to your laptop every ~10 iters (run on your LAPTOP,
not the box).** Open a *second* Git Bash window:
```bash
rsync -avz -e "ssh -p <port>" \
  root@<ip>:~/boardgame-ai/runs/kingdomino/cloud_<channels>x6_run1/ \
  runs/kingdomino/cloud_<channels>x6_run1/
```
(If `rsync` isn't available, use
`scp -P <port> -r root@<ip>:~/boardgame-ai/runs/kingdomino/cloud_<channels>x6_run1 .`.)
The box is ephemeral — if it dies with no off-box copy, the run is gone (Phase 5).

**Step 8 — Stop paying: DESTROY the instance.**
When training is done and you've copied results back, go to the Vast.ai web UI and
**Destroy** the instance. *Stopping* is not enough — billing continues until it's
destroyed. Then finish with Phase 6 (merge Elo, save the checkpoint, commit).

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

### 0.2 Prepare the calibration benchmark runner
The first paid hour should produce enough evidence to choose run settings, not
a pile of one-off shell history. Prepare an automated benchmark runner that
emits one CSV row per test and a short summary of selected settings.

Required CSV fields:

- hardware: GPU name, CPU model if available, logical CPU count, CUDA, driver,
  torch version, torch CUDA arch list.
- model: channels, blocks, batch size, AMP inference, compile, channels-last.
- MCTS geometry: batch_slots, leaf_batch, sims, full_search_fraction.
- CPU split: game_cpus, solver_cpus, async_solve, exact_endgame_max_secs.
- throughput: games/sec, useful training positions/sec, exact solved/sec,
  buffer rows/game, total evals/sec, requests/sec, mean batch fill.
- timing: eval_h2d_sec, eval_forward_sec, eval_readback_sec, solve_wall_sec,
  train_step_sec if applicable.
- quality/proxy metrics when available: policy target entropy, search-vs-prior
  KL, exact fallback attempts/fallbacks by retry point.

The runner should execute Phase 3 in order, write CSV continuously, and write a
Markdown or JSON summary with the recommended settings.

### 0.3 Push latest code
- [ ] `git push origin main` — the cloud box clones from GitHub.

### 0.4 Confirm what transfers via git vs. scp
- `best_checkpoint/`, `elo_db.json`, `elo_games.jsonl` are **committed** →
  arrive automatically with `git clone`. No scp needed for the Elo ladder.
- For a cold start we need **no weights**.
- **Buffer seeding is NOT available for this run.** Prior `buffer_final.pkl` files
  hold **261-flat** encoded examples; the current encoder is **333-flat**, so they
  cannot train the new net. Start with an empty buffer (no `--warm_buffer`).

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
6. **Run the calibration benchmark sequence** (Phase 3).

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

## Phase 3 — Calibration benchmarks (run before training)

Run immediately after the verification gate. Results decide the model size,
`--batch_slots`, CPU split, exact cap, and optional feature flags for the
training command.

### 3.1 Forward ceiling
Goal: measure the raw neural-network throughput ceiling before the MCTS loop
adds CPU and queueing effects.

- [ ] Test model sizes: **80x6**, **96x6**, and optional **80x8**.
- [ ] Test batch sizes: `64,96,128,160,192,224,256,320,384,512`.
- [ ] Test AMP inference off/on.
- [ ] Test `torch.compile` off/on with `TORCH_LOGS=recompiles`; try dynamic
      shapes if variable live batches trigger recompiles.
- [ ] Test channels-last only if the forward profile suggests it may help.

Decision rule: choose the largest model whose live MCTS loop can still keep the
GPU reasonably full. A pure forward win only matters if it survives end-to-end
self-play.

### 3.2 Self-play batch geometry
Goal: maximize end-to-end NN eval throughput with solver disabled so the CPU
solver does not hide the batching result.

- [ ] Use `--exact_endgame_max_secs 0`.
- [ ] Sweep `--batch_slots`: `32,48,64,80,96,128`.
- [ ] Keep `--leaf_batch 6` as the production default.
- [ ] Include one `--leaf_batch 8` row only as a throughput ceiling. Do not use
      it for the real run unless a later quality/Elo test clears it.

`batch_slots` controls how many independent games feed the shared GPU queue.
It is mostly a machine/utilization knob. `batch_size` controls training-update
minibatches and forward-benchmark tensor sizes. `leaf_batch` controls how many
new MCTS leaves can be expanded at a node before search feedback is observed;
it is primarily a game/search-quality knob, and prior runs showed degradation
around 8.

For fast/no-noise moves, keep leaf batching conservative. If all leaf slots are
spent down the same high-prior path, throughput can rise while search diversity
falls.

### 3.3 Async solver CPU split
Goal: find the best generation/solver balance on the target CPU.

- [ ] Use the selected model and batch geometry.
- [ ] Enable `--async_solve --exact_endgame_max_secs 3.0`.
- [ ] Sweep `--game_cpus`; solver CPUs default to `logical_cpus - game_cpus`.
- [ ] Suggested sweeps:
      - 16 logical CPUs: `2,4,6,8`
      - 32 logical CPUs: `4,8,12,16`
      - 64 logical CPUs: `4,8,12,16,24`

Prefer the setting with the best useful positions/hour, not just games/sec.
On high-core CPUs, more solver workers may improve exact coverage; on
low-frequency many-core hosts, a smaller count can win if individual solve
latency gets worse.

### 3.4 Exact endgame cap
Goal: decide how much wall time to spend buying exact labels.

- [ ] Sweep `--exact_endgame_max_secs`: `3,5,7,10`.
- [ ] Record attempts/fallbacks separately for `deck4_initial`,
      `deck4_retry`, and `deck0`.
- [ ] Compare exact solved positions/hour, added wall time, and useful
      examples/hour.

The shared cap is fine: all three retry points use the same timeout, while the
metrics reveal where the fallbacks occur.

### 3.5 Full-search fraction
Goal: choose the learning-throughput balance, not the fastest games/sec.

- [ ] Sweep `--full_search_fraction`: `0.25,0.35,0.45`; optionally `0.55`.
- [ ] Keep fast moves unrecorded, no-noise, and greedy.
- [ ] Compare useful examples/hour, games/hour, buffer growth/hour, policy
      target entropy, and search-vs-prior KL.
- [ ] If two settings are close, run a short fixed-budget training/Elo pilot.

Very low values improve games/sec but produce fewer trained positions per game.
Very high values produce more positions per game but slow generation and may
reduce game diversity. The right value is the best learning signal per hour,
not the maximum throughput row.

### 3.6 Training-step throughput
Goal: make sure the learner is not the new bottleneck after generation is
calibrated.

- [ ] Test `--batch_size`: `256,384,512`.
- [ ] Test prefetching if available.
- [ ] Keep sample workers conservative unless profiling shows data loading is
      limiting.

This is mostly a throughput/stability check. Larger batches can improve GPU
efficiency but change optimizer dynamics, so prefer the largest batch that does
not require changing the learning-rate plan.

### 3.7 One-time feature tests
Run once on the selected setup:

- [ ] `torch.compile` in the live loop; require a real speedup and no
      recompilation storm.
- [ ] AMP inference; use it only if it improves live evals/sec and does not
      introduce numerical weirdness.
- [ ] `--double_buffer`; it was negative locally, but may flip on a faster GPU.
- [ ] `--profile_eval_timing`; keep if the overhead is negligible during short
      calibration, remove for long training unless needed.
- [ ] channels-last only if Phase 3.1 showed a 5-8%+ end-to-end win at live
      batch sizes.

---

## Phase 4 — Training command (80x6/96x6 cold start)

Linux shell, single line. Substitute `<channels>`, `<batch_slots>`,
`<game_cpus>`, `<exact_secs>`, and `<full_search_fraction>` from Phase 3.
Add `--compile`, AMP inference, or `--double_buffer` only if Phase 3 confirmed
them.

```bash
python -m games.kingdomino.self_play \
  --engine batched_open_loop \
  --device cuda \
  --async_solve \
  --game_cpus <game_cpus> \
  --iterations 60 \
  --games_per_iter 300 \
  --train_steps 1200 \
  --sims_schedule "0:1000,10:1300,20:1600" \
  --channels <channels> --blocks 6 \
  --batch_slots <batch_slots> --leaf_batch 6 --virtual_loss 1 \
  --exact_endgame_max_secs <exact_secs> \
  --batch_size 256 \
  --lr_schedule "0:1e-3,20:3e-4,45:1e-4" --weight_decay 1e-4 \
  --buffer 500000 \
  --lambda_score 0.5 --lambda_w 0.25 --score_scale 160.0 \
  --policy_weight 1.0 --grad_clip 1.0 --margin_gain 2.0 \
  --alpha 0.5 \
  --c_puct 1.5 --temp_moves 20 --fpu -0.2 \
  --playout_cap_randomization \
  --full_search_fraction <full_search_fraction> \
  --fast_move_sims 100 --fast_move_dirichlet_epsilon 0.0 --fast_move_temp_moves 0 \
  --policy_target_pruning \
  --benchmark_every 10 --benchmark_sims 50 --benchmark_seeds 20 \
  --checkpoint_dir runs/kingdomino/cloud_<channels>x6_run1 \
  --save_buffer runs/kingdomino/cloud_<channels>x6_run1/buffer_final.pkl \
  --exact_fallback_positions runs/kingdomino/cloud_<channels>x6_run1/exact_fallback_positions.jsonl \
  --elo_every 20 --elo_sims 400 --elo_games_per_anchor 32 \
  --elo_anchors games/kingdomino/elo_anchors_80x6.csv \
  --elo_db elo_db_80x6.json --elo_games_log elo_games_80x6.jsonl \
  --seed 0
```

Notes that differ from a warm-start laptop run:

- **No `--warm_start`** (cold start — random weights at 80x6 or 96x6).
- **`alpha` is fixed at `0.5`, not scheduled (2026-07 win-gated value).** `alpha`
  is now the reserved margin band **B** in `(1-B)*win + B*win^4*margin`, not the
  old convex-blend weight. The old `--alpha_schedule "0:0.8,20:0.5,40:0.2"` ramps B
  *down*, which shrinks margin-fighting exactly when the net is strong enough to
  use it — the opposite of what we want. Use `--alpha 0.5`. If you ever schedule
  B, ramp it **up** (score head is miscalibrated early), never down.
- **Use schedules for sims and LR.** The sample command ramps sims and decays LR
  in stages so training is not jolted by one abrupt change.
- **Tune `--game_cpus` on the target box.** The exact-solver pool gets all
  remaining logical CPUs by default. Use `--solver_cpus` only as an explicit
  override.
- **Keep `--leaf_batch 6`.** It was the best known quality setting; `8` is
  only a calibration ceiling unless Elo later says otherwise.
- **Add `--compile`, AMP inference, or `--double_buffer` only if Phase 3
  confirmed them.**
- **No buffer seed (this run).** Old buffers are **261-flat** encoded examples,
  incompatible with the **333-flat** encoder — do **not** pass `--warm_buffer`.
  The buffer fills from scratch over the first few iterations.
- **Elo: use the 80x6 pool, never the legacy one (2026-07-05).** The legacy
  `games/kingdomino/elo_anchors.csv` / `elo_db.json` anchors are OLD-ENCODER
  (261-flat) checkpoints — they cannot evaluate against current nets. The
  command above points at `elo_anchors_80x6.csv` (5 bootstrapped anchors from
  run1/run2, ratings 950–1762) with its own db/games log; `--elo_every 20`
  keeps the rating cost to ~3 sessions per 60-iteration run. The anchor
  checkpoint files (`runs/kingdomino/cloud_80x6_run1/iter_00{10,20,40,66}.pt`,
  `runs/kingdomino/cloud_80x6_run2/iter_0050.pt`) must exist on the cloud box
  at those relative paths — sync them before launch.
- **Exact-solver defaults (2026-07-05 restructure).** `--exact_policy_mode`
  defaults to `argmax_ties` (won the label-shape ablation 231-162-7 and is the
  cheapest mode; `soft_clamp` / `exact` remain available for ablation). With
  the within-solve transposition table, the local ablation arms saw 2-7%
  first-attempt deck4 fallback at `--exact_endgame_max_secs 3.0` (old solver:
  12% at 3.0s, 43% at 1.0s) — prefer 3.0s over run2's 1.0s squeeze.

For same-architecture continuation runs where
`runs/kingdomino/best_checkpoint/current_best.pt` matches `<channels>x6`, prefer
soft-gated training and promotion-triggered Elo instead of scheduled Elo:

```bash
  --warm_start_current_best \
  --selfplay_generator_mode soft_gate \
  --promotion_every 5 --promotion_games 384 --promotion_sims 100 \
  --soft_gate_revert_win_rate 0.48 \
  --smart_elo --smart_elo_on_promote --smart_elo_games_per_anchor 32 \
  --elo_every 0
```

### 4.1 Replay-ratio sanity (two independent ratios)
Use the Phase 3 measured `recorded_positions_per_game`, not raw game length.
With playout-cap randomization, only full-search moves enter the buffer, so a
25% full-search fraction may record roughly a quarter of the game's positions.

- **Staleness window** = `buffer / (gpi × recorded_positions_per_game)`.
  Target roughly **20-30 iterations** of history.
- **Sample reuse** =
  `train_steps × batch_size / (gpi × recorded_positions_per_game)`. Keep this
  in the same regime as the laptop runs unless there is a deliberate reason to
  change optimizer pressure.

These are **independent knobs**: buffer size sets staleness; `train_steps`
sets reuse. Because only recorded full-search moves enter the buffer, lowering
`full_search_fraction` can quietly raise sample reuse and stretch the buffer
history window. Recompute both ratios from the first training log before
letting the run continue overnight.

- [ ] **Consider ramping `train_steps`** over the first ~10–15 iters (e.g.
      start ~400, climb to 1200) rather than 1200 from iter 1, to avoid
      entrenching early noise. Phased: run a short low-`train_steps` segment,
      then warm-start the next segment with higher `train_steps`.
- [ ] **Verify `samples_per_iter`** from the iter-0 training log — confirm
      recorded positions/game matches the Phase 3 estimate. If the buffer grows
      too slowly, increase `games_per_iter`, `full_search_fraction`, or reduce
      `buffer` so the history window remains in range.

---

## Phase 5 — Mid-run checkpoint sync (don't lose the run)

Vast.ai instances are interruptible/ephemeral. The cost guardrail protects
your wallet, not your run — if the box dies at iter 40 with no off-box copy,
the run is gone.

- [ ] Periodically pull checkpoints + `training_log.jsonl` **off** the instance
      (every ~10 iters), e.g. from your local machine:
      ```bash
      rsync -avz user@cloud:~/boardgame-ai/runs/kingdomino/cloud_<channels>x6_run1/ \
        runs/kingdomino/cloud_<channels>x6_run1/
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
2. **Download the best checkpoint** if improved. Note it's 80x6 or 96x6 now,
   so it is *not* a drop-in replacement for `best_checkpoint/best_32x4.pt` —
   store it alongside (e.g. `best_80x6.pt` or `best_96x6.pt`) and update
   `best_checkpoint/README.md`.
3. **Update the Elo anchor pool** if the new model rates >150 Elo above the
   current top anchor (`--reanchor`).
4. **Commit:**
   ```bash
   git add elo_db.json elo_games.jsonl games/kingdomino/best_checkpoint/
   git commit -m "Cloud run cloud_<channels>x6_run1: <summary>"
   git push origin main
   ```

---

## Quick checklist

- [ ] 0.1 cu128 torch wheel chosen & pinned
- [ ] 0.2 calibration runner prepared with CSV + summary output
- [ ] 0.3 code pushed
- [ ] 1 instance: RTX 5090, driver ≥ 570, cost cap set
- [ ] 2 `setup_cloud.sh` written (rustup, clone, cu128 deps, maturin, gate, benches)
- [ ] 2.1 GPU verification gate passes
- [ ] 3 calibration run; model size, `batch_slots`, CPU split, exact cap,
      full-search fraction, and feature flags chosen
- [ ] 4 training launched (cold start, schedules enabled, leaf_batch 6)
- [ ] 4.1 replay ratios sanity-checked; train_steps ramp considered
- [ ] 5 mid-run checkpoint sync running
- [ ] 6 Elo merged, checkpoint saved, committed
