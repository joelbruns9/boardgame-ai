# 7WD cloud training: readiness plan

**Status:** proposed, nothing built. Revision 3 -- rewritten against an external
review (10 findings, 9 confirmed against code, 1 partially accepted).
**Date:** 2026-07-28.
**Trigger:** `runs/laptop_training_03` died at iteration 70 of 90 with a host
`MemoryError` after 11 h. The loop itself is healthy -- 9 promotions in 13 gates
-- so the blocker is operational, not algorithmic.

Every number here was measured on `runs/laptop_training_03` or on this laptop
(RTX 3070, 8 GB) during planning; the appendix records how.

**Scoreboard for the cloud run:**

| source | civilian | scientific | military |
|---|---|---|---|
| ZeusAI (converged) | 61.7% | 21.4% | 16.9% |
| BGA humans | 58.0% | 25.6% | 16.4% |
| **run 03, iters 60-69** | **70.9%** | **13.0%** | **16.1%** |

Military is at reference level; the whole gap is science. ZeusAI needed ~100k
games to discover science organically and ran 1,000 sims/move; run 03 played 28k
games at 39 average sims. The gap may be volume, not architecture. This plan
does not close it -- it makes it **measured every iteration**.

---

## Decisions locked

| question | decision | implemented by |
|---|---|---|
| model size | from the W0 experiment; **5-15M expected**, not 92M. No mid-run growth -- size is **manual** | W0 |
| precision | **bf16 on every path**, explicit and persisted (see the corrected baseline in W0.3) | W0 |
| replay window | **growing window**, sublinear power law in *total games* | W1 |
| curriculum bots | early accelerant only, annealed **in games** (~10k) | W1 |
| opponent diversity | **HOF league sampling** during generation | W1 |
| schedules | every schedule expressed in **games, not iterations** | W1 |
| gate statistic | **port Kingdomino's Wilson-LCB three-way rule** (promote / probation / revert) | W5 |
| gate budget | cap **800**, pair-level observations, futility via Wilson UCB | W5 |
| gate cost | fix throughput **before** raising the cap | W5 |
| stagnation | **games-indexed** anchor + metric-triggered intervention ladder | W7 |
| cloud target | vast.ai, single box | W6 |

---

## Workstreams

| # | Workstream | Size | Blocks launch? |
|---|---|---|---|
| W0 | Model size + precision, decided by measurement | M | **Yes** |
| W1 | Training schedules, growing window, HOF league | M | **Yes** |
| W2 | Memory: fix the crash, bound the footprint, measure it | M | **Yes** |
| W3 | Shared run-stats contract + reporting | M | **Yes** |
| W4 | BGA advisor end-to-end with the iter-60 model | M | No -- parallel |
| W5 | Gate efficiency + Wilson-LCB decision rule | M | **Yes** |
| W6 | Cloud setup script | M | **Yes** |
| W7 | Stagnation detection + intervention ladder | M | **Yes** (both parts) |

**W1 precedes W2 and W5 acceptance**: HOF league sampling changes both the
memory profile and batch composition those acceptances measure, so measuring
them first measures the wrong system.

---

## W0 -- Size and precision, decided by measurement

**W0.1 -- Size, on the banked buffer.** *(M)*
`runs/laptop_training_03/buffers/` holds 28,000 games. Compare 128x4 (baseline),
256x6, and 384x8.

Design, strengthened after review -- a single supervised run would measure
imitation fit, not capacity or strength:

- **equal data and equal optimizer steps** per arm, identical splits;
- **at least 3 seeds per arm**, reporting spread, not a single number;
- a **per-width LR/warmup schedule** (a fixed 2e-4 favours the narrow arm);
- a **paired arena** between the trained arms at fixed sims and paired seeds,
  scored with the same Wilson bound as W5 -- validation loss alone cannot rank
  playing strength;
- a **search-improvement check**: does the wider net's policy still gain from
  search at the production sims budget, or has it absorbed the gap?
- report as a **validation / strength / throughput Pareto**, not a winner.

Primary value is a cheap **negative gate**: if a wider net cannot fit data you
already own, do not scale. The arena carries any positive decision.

Sizing context (measured):

| config | params | fp32 rows/s @b256 | bf16 rows/s | advisor 800-sim move |
|---|---|---|---|---|
| **128x4 (current)** | 1.03 M | 14,118 | 32,182 | 0.3 s |
| 256x6 | 5.2 M | 3,645 | 9,835 | 0.3 s |
| 384x8 | 14.9 M | 1,430 | 4,356 | 0.55 s |
| 512x8 | 26.2 M | 824 | 2,781 | 0.93 s |
| 768x12 (ZeusAI) | 86.5 M | 239 | -- | 3.1 s |

Generation demands **~4,960 NN rows/s** (A3), so the 1M net runs at ~35% NN
utilisation. **Laptop advisor latency is not the binding constraint at any size
in this table**; training throughput is.

**W0.2 -- Expose and scale attention heads.** *(S)*
`SWDNet.__init__` hard-codes `heads=4` and `build_model` does not expose it
(`net.py:300`, `train.py:313`). At d_model 384/512 that is 96/128-dim heads
against ZeusAI's 64-dim. Land `heads = d_model // 64` **before** W0.1 measures
anything, or the wide arms are handicapped by construction.

**W0.3 -- Precision as an explicit, persisted, per-path config.** *(M)*

The starting point is not what revision 2 assumed. Audited:

| path | share of run-03 wall clock | precision today |
|---|---|---|
| self-play generation | 77% | **fp32** (`rust_bridge.py:368` has no autocast; line 318 casts features float16 → **float32**) |
| gates | 10% | **fp32**, same adapter |
| validation | small | **fp32**, outside autocast (`train.py:96`) |
| advisor / `Evaluator` | n/a | **fp32** (`inference.py:73`) |
| training | ~2% | **already AMP fp16** with a `GradScaler` (`train.py:488`, no dtype argument → fp16 on CUDA) |

So the work is: **add bf16 autocast to the four inference paths** (where 87% of
the wall clock and all of the measured 2.3-3.0x live), and **change training's
implicit fp16 to explicit bf16** (no meaningful speed change; drops the
GradScaler and the overflow risk as the model widens).

Requirements:
- a single `--precision {fp32,bf16}` config value, **persisted in the manifest**
  and validated on resume alongside `lifecycle_config`;
- propagated to training, validation, self-play, gates, and the advisor;
- a test per path asserting the dtype actually in effect -- an autocast context
  that silently does nothing is the failure mode here;
- A/B on `_generate_iteration_rust` via `f4_phase_d_ab.py`, **not** the
  microbenchmark (history: 1.99x → 1.89x, +48% → +21%).

Fidelity is already measured (A4): bf16 changes the argmax on 2 of 512 real
positions, both where the net's own top-1/top-2 priors differ by 0.0016, worst
value delta 5.3e-3 against a 39-sim noise floor of ~0.16. Context worth
recording: **run 03 was already trained under fp16 AMP** -- reduced precision in
training is the status quo that produced value_acc 0.732, not a new risk.

**W0.4 -- Re-baseline the determinism gates.** *(S)*
Bit-exact record comparison stops being the right assertion. Use the established
shape: mock evaluator → byte-identical; real net → batch-invariance first, then
*measured* divergence on the discrete trajectory (actions, digests, sims).

**Acceptance:** a chosen `d_model`/`layers`/`heads` with a Pareto justification
including arena results; `--precision` persisted, propagated, and per-path
tested; generation A/B measured; determinism gates re-baselined.

---

## W1 -- Training schedules, growing window, HOF league

Revision 2 declared these mandatory and gave them no implementing task; W3 only
*recorded* their values. Current code contradicts all three: `replay_window:
int = 20` is a fixed scalar (`phase_d.py:128`), `curriculum_anneal_iterations`
and `draft_prior_iterations` are iteration-counted (`phase_d.py:137`, `:143`),
and HOF is write-only -- `self.hof.add(...)` at `phase_d.py:1968` is its only
use; nothing ever samples it.

**W1.1 -- Growing replay window.** *(M)*
`window_games(total_games) = min(cap, c * total_games**alpha)` with alpha in
0.5-0.8, fitted to your own staleness-vs-value-accuracy data rather than
imported. Small early (on-policy, fast adaptation), growing late (variance
reduction and the diversity that mattered in Kingdomino). Requirements: the
schedule value and the realised window in every stats row; **resume recomputes
from total games, never from a stored window**; the cap is derived from the
memory budget (W2.3), not chosen independently.

Motivating evidence: run 03's window grew 1,400 → 8,000 games over iterations
0-20 and then froze; value accuracy peaked at iteration 35 and decayed after.
Suggestive, not proof.

**W1.2 -- Every schedule in games.** *(M)*
Curriculum bot mix, draft prior, and window all keyed on cumulative games.
Bots anneal out over the first ~10k games -- measured exhaustion point, where
the net's win rate against them passes ~95% (A6). Run 03 used 400
games/iteration; a cloud run at 500-800 would silently rescale every
iteration-keyed schedule.

**W1.3 -- HOF league sampling.** *(M)*
Sample a configurable fraction of generation opponents from archived HOF
checkpoints instead of always `current_best`. Requirements: sampling policy
(uniform over last N, or recency-weighted) in config; the opponent identity
recorded per game so W3 can split outcomes by opponent; a deterministic,
resume-stable sampler keyed on the run seed.

**W1.4 -- Tests and resume semantics.** *(S)*
Schedule functions unit-tested at boundaries; a resume mid-schedule
reproduces identical values; changing a schedule across resume is refused by
the same mechanism as `lifecycle_config`.

**Acceptance:** a short run shows window growth tracking the formula, bot mix
reaching zero at the configured game count, HOF opponents appearing in
generation at the configured rate, and a resume reproducing all three exactly.

---

## W2 -- Memory

### What happened

1. A Python-level allocation failed inside the gate's inference boundary at
   iteration 70 (`run.log:151`; the bare `MemoryError: ` with its trailing space
   is pyo3's `Display for PyErr`, so the failure was on the Python side of the
   adapter call).
2. The worker is fail-fast: `if request.reply.send(result).is_err() || failed {
   break; }` (`eval.rs:670`). The thread exited.
3. The next `submit_prepared(...).wait()` found the channel disconnected and
   raised `ValueError: global inference worker dropped its response`
   (`eval.rs:549`) -- the exception that ended the run, and a red herring.

### Tasks

- **W2.1** *(S)* Latch the worker's terminal `PyErr` and re-raise it from the
  next `wait()`. A run must never again die reporting a symptom.
- **W2.2** *(S)* Replace `PyByteArray::new` with `PyByteArray::new_with`
  (`eval.rs:426-443`). The former is pyo3's **unchecked** constructor -- it
  wraps a possibly-NULL pointer instead of returning `Err`, which is exactly the
  call that cannot fail cleanly under memory pressure.
- **W2.3** *(M)* **RSS-calibrated** cache bound, not an `nbytes` sum.
  `Example` holds **six** numpy arrays; summing their `nbytes` gives ~13.1 KB
  per example, but the measured cost is **17.8 KB** (A1) -- `nbytes` understates
  by ~26% because it excludes ndarray objects, the dataclass, cache keys, list
  overhead, and allocator fragmentation. Bound on `nbytes x calibration_factor`
  with the factor measured at startup, and hold explicit headroom. Accept the
  old `--example-cache-examples` flag, converting at 17.8 KB; fix the 12.5 KB
  constant at `phase_d.py:1412`.
- **W2.4** *(S)* Per-iteration RSS telemetry (psutil, already a dependency) at
  post-generation, post-training, post-gate, plus peak and cache bytes, into the
  W3 stats block.
- **W2.5** *(S)* **Pre-gate admission check**, not only post-hoc pressure
  detection. The iteration-70 failure happened *inside* the gate, so a
  post-gate sample would have observed it after the crash. Before a gate runs:
  estimate its peak (two models + evaluators + `global_batch_cap` rows of packed
  buffers), and if projected RSS exceeds the budget, evict the cache **first**.
- **W2.6** *(S)* `--memory-budget-gb`; on breach evict to a floor and log a
  `memory_pressure` event. Slower re-replay beats losing a run.
- **W2.7** *(S)* Free the gate's models explicitly at gate exit -- `del` +
  `gc.collect()` + `torch.cuda.empty_cache()`.

**Acceptance:** a memory-starved run reproduces a **clean** `MemoryError` with
the real traceback; RSS flat within 5% across the last 30 iterations of a
60+-iteration run **with W1 active**; `laptop_training_03` resumes 70 → 90.

---

## W3 -- Shared run-stats contract

`games/az_loop/contract.py` types every result's metrics as `dict[str, Any]` and
`run_controller._build_row` copies them verbatim. Three consequences:

- answering "what were the victory types" took four throwaway scripts;
- the Rust generation metrics -- NN rows, batch sizes, forced-row share, wave
  stats -- are **collected and discarded**: `phase_d.py:1251` appends them and
  logs only `len(rust_metrics)` as `rust_chunks`;
- generation decayed **27%** over run 03 (1.03 → 0.81 games/s) and the metric
  that would explain it is the discarded one. On a 24-hour cloud run that is
  ~6 hours of generation, unexplained.

New `games/az_loop/stats.py`: a versioned, typed `IterationStats` filled by the
adapter and validated by the controller. `LOG_SCHEMA_VERSION` 1 → 2, readers
tolerant of v1 so run 03 stays analyzable.

**Core (every game fills):**

| group | fields |
|---|---|
| `generation` | games, moves, moves/game (mean, median, p10, p90), decisions searched, mean sims, seconds, games/s, **NN rows, rows/s, forced-row share, mean batch size**, **opponent mix (HOF vs current_best vs bot)** |
| `outcomes` | **`terminal_reason` histogram**, winner distribution, first-player win rate, mean margin, **split by opponent type** |
| `replay` | window games, examples, new examples, reuse factor, staleness (max, mean, p90), **schedule value and realised window** |
| `training` | per-head train/val losses, accuracies, lr, grad norm, steps, seconds, **replay-derivation seconds**, **precision in effect** |
| `gates` | opponent, **win rate, Wilson LCB/UCB, games**, decision, stop reason |
| `resources` | RSS post-gen/post-train/post-gate, peak RSS, cache bytes, GPU util, VRAM peak, seconds per phase |

`terminal_reason` is the game-agnostic form of victory type: Kingdomino emits
`{"score": n}`, 7WD the civilian/military/scientific histogram.

**Game extension** (`game_specific`, adapter-owned) for 7WD:
- **victory type x game length**, and the **age and move index at which the game
  ended**. Measured: military/science wins land at move ~64-65 vs ~73 civilian
  -- deep Age III conversions, not rushes (A6). Logging the age turns "did a
  rush plan ever emerge" into a query.
- science: sixth-symbol races, tokens taken, pairs completed.
- military: max track position, tokens triggered, gold pillaged.
- draft/wonder: wonders built vs discarded, Age III completion rate.

**W3.4 -- `tools/az_report.py`.** *(M)* One command, any run directory: the
block-aggregated outcome mix with binomial error bars, the gate ladder with
Wilson bounds, the learning curve, throughput, RSS, and the victory mix against
the ZeusAI/BGA reference. Without it the schema is just tidier JSON.

**Acceptance:** a 7WD run and a synthetic second-game fixture both emit valid
schema-v2 rows; `az_report.py` reproduces the planning tables; the 27%
generation decay is diagnosable from the log alone.

---

## W4 -- BGA advisor with the iter-60 model (parallel)

Already built: the shared host (`games/advisor/`), the 7WD adapter with a scrape
branch (`advisor_adapter.py:147`), the determinizer, the endgame solver, the
`gamedatas` → wire extractor (`bga_extract.py`, live-verified, freshness guard),
the raw-dump snippet (`bga_snippet.js`), and a local UI (`web_app.py` +
`web_static/index.html`).

- **W4.1** *(S)* Add the `{"bga": <raw gamedatas>}` branch to `state_from_wire`;
  `wire_from_bga` already emits exactly what the observation branch accepts.
  ~15 lines plus tests against `testdata/bga_*.json`.
- **W4.2** *(M)* Package `bga_snippet.js` as an MV3 extension: grab
  `window.gameui.gamedatas`, POST to the local host, render ranked moves. All
  game knowledge stays server-side.
- **W4.3** *(M)* Play 10+ games with iter-60. Measure advisor latency at
  production sims, `UnsupportedBgaState` rate and triggers, stale-payload
  catches, advisor-vs-played-move agreement. **The model is the weak part and
  that is fine** -- this gate tests plumbing.

**Acceptance:** 10 games with zero silently-wrong positions; every unsupported
state raises rather than guesses; documented latency budget.

---

## W5 -- Gate efficiency, then the decision rule

### What the gate costs

Measured (A5): **1.08 h of the 11.06 h run, 9.8%**, for 1,240 games. Per-game
cost decomposes as **~193 s fixed per gate + ~1.74 s per game**. The marginal
1.74/1.09 = **1.60x** over a self-play game matches the 64/39.1 = **1.64x** sims
ratio -- the marginal gate game costs exactly what the extra search costs.
**All the waste is the fixed 193 s.** A 20-game gate spends 227 s, the
equivalent of 85 self-play games.

| | self-play | gate |
|---|---|---|
| scheduler calls | **1 per 400 games** | **~840 per gate** (per ply, per seat, per wave, per leg) |
| inference worker | one, for the iteration | spawned and joined **every call** |
| games in flight | 48-slot rolling pool, refills | <=48 pairs per wave, split by seat (~24 rows), drains to empty |
| models built | 1 per iteration | 2 per gate, plus ~4 redundant `torch.load` |
| seat legs | n/a | **sequential** |

This is trap #3 from the throughput programme ("fixed chunk submission decays
concurrency to zero as each chunk drains") still live in the gate path.

### Throughput first

- **W5.1** *(M)* One persistent inference worker across all plies of a gate.
- **W5.2** *(S)* Build models/evaluators **once per gate**; remove the redundant
  `torch.load` in `checkpoint_agent_name`.
- **W5.3** *(M)* Run the two seat-legs **concurrently** -- independent, and
  interleaving restores batch width without reintroducing the seat-routing bug
  that once inverted gate results.
- **W5.4** *(S)* Rolling refill across pairs and a gate-specific slot count.

Target 193 s → ~50 s, which is what pays for the larger cap.

### Decision rule -- port Kingdomino's

- **W5.5** *(M)* Port `games/kingdomino/promotion.py`'s Wilson rule and
  `_generator_action_after_promotion_check` (`self_play.py:2908`):
  - `wilson_lower_bound(points, games, z=1.96)`, **draws = 0.5 points**;
  - **promote** if LCB > `promotion_min_lcb` (0.50);
  - **revert** if win rate < `revert_win_rate` (0.48);
  - **probation** otherwise;
  - **observation unit = the seat pair**, outcome in {0, 0.5, 1}. Wilson assumes
    independence, and paired games share a seed; pairing the unit makes the
    bound exact rather than mildly anti-conservative;
  - cap **800**; futility stop when the Wilson **upper** bound falls below 0.50
    (promotion arithmetically unreachable).

Why this and not the Bayesian spec from revision 2: it is already implemented
and tested in this repo, Wilson handles draws natively (the Beta-Binomial
fudged them), and the three-way output maps directly onto `az_loop`'s existing
ACCEPT / CONTINUE / REJECT consumed by `gate_transition`.

Replaying run 03 under this rule agrees on **9 of 13** gates (A7). The three
promote → probation flips (gates 15, 20, 60) are gates SPRT truncated at 74-154
games, where the interval is simply too wide to clear 0.50.

**This settles the cap.** Win rate required for LCB > 0.50: **0.598** at 100
games, **0.569** at 200, 0.549 at 400, **0.535** at 800. At today's 200-game cap
a candidate needs a **+7%** edge to promote; at 800 it needs **+3.5%**.

Negative result worth recording: a statistically valid futility stop **never
fires at n <= 200** -- even a 0.422 win rate at 96 games still has UCB ~0.52. It
begins paying at n ~ 800, cutting clear losers around 300-400 games. Futility is
worth having *because* of the larger cap, not instead of it.

- **W5.6** *(S)* **Gates are decisions; anchors are measurements.** Early
  stopping biases the score rate toward whichever boundary stopped it, so W7's
  anchor check runs **fixed-N with no early stopping**. Note run 03 feeds
  truncated matches straight into `self.elo.record()`, biasing the ladder the
  same way.

### Known residual risk: no promotion rollback

Verified: `ACCEPT` sets `replace_best=True` (`training_control.py:195`);
`REJECT` sets `replace_best=False` (`:225`), discarding the *candidate* and
keeping the possibly-bad best; `reset_learner` copies `current_best → latest`
(`run_controller.py:450-458`), propagating it; anchor gates return metrics with
**no lifecycle effect** (`run_controller.py:460`). **A bad promotion is never
undone anywhere in the system.**

The mitigation is the strict promote bound: LCB > 0.50 at z=1.96 is ~97.5%
one-sided confidence, and the asymmetry against the 0.48 point-estimate revert
threshold is deliberate -- promotion is irreversible and must be confident;
reverting only discards a candidate. **Periodic HOF revalidation of
`current_best` against archived checkpoints, with demotion on loss, is the real
fix and is deferred**; it is recorded here so the gap is known rather than
assumed away.

**Rejected during planning, recorded so they are not re-proposed:**

- *Tree reuse across plies* -- deferred pending measurement. The usual 1.5-2x is
  deterministic-game folklore; 7WD resolves a reveal after each move, so only
  the subtree under the *realized* outcome survives and the reuse fraction may
  be 10-20%. Measure visits-inherited / visits-total for one arena game first;
  below ~15% it is not worth the plumbing. If built, inherited visits are
  **deducted from the budget** so "64 sims" keeps its meaning.
- *Control variates* -- paired seeds and seats already capture the variance;
  victory type is an **outcome**, not an exogenous covariate, and adjusting for
  it would bias rather than sharpen.
- *Accumulating evidence across gates* -- the candidate is a different net every
  gate, so the pair never repeats.
- *Harvesting gate games as training data* -- 1,240 games is **4.4%** of the
  28,000 self-play games, argmax-only, no search distributions, off-policy for
  both nets, and `_play_two_net_games` builds no `GameRecord` at all.

---

## W6 -- Cloud setup

`setup_cloud_7wd.sh` (250 lines) **predates the Rust engine**: pure Python, no
`rustup`, no `maturin`, header explicitly says the Rust build is
Kingdomino-only. Every number in this plan assumes `--generation-backend rust
--gate-backend rust`. `setup_cloud.sh` (Kingdomino, 301 lines) already has the
missing stages.

- **W6.1** *(M)* Factor `setup_cloud_common.sh`: clone/update, Python + cu128
  torch ordering, GPU hard-fail gate, rustup >= 1.85, crate build, smoke,
  detached `nohup` launch, idempotent resume. Per-game files supply crate path,
  smoke command, launch command. Third instance of the standardization pattern
  after az_loop and the advisor.
- **W6.2** *(M)* **A smoke that cannot pass vacuously.** As specified in
  revision 2 it would have: `pytest` is **absent from `requirements.txt`**;
  `test_buffer_games_equivalent` calls `pytest.skip` when the corpus is missing
  (`test_rust_engine_equiv.py:1424` -- its docstring says the buffers live "on
  the gate box, not a fresh checkout"); and `runs/` is **gitignored**
  (`.gitignore:37`). Required: add `pytest` to requirements; **commit a small
  fixed corpus** (~50 games) outside `runs/`; run the equivalence suite with
  `-p no:randomly -W error` and **fail on any unexpected skip** (assert the
  collected/skipped counts against an expected manifest).
- **W6.3** *(M)* Sweep → launch-flag pipeline (`run_f4_cloud_sweep.sh` →
  `f4_cloud_select.py` → `f4_cloud_finalize.py`) with an **explicit flag
  translation layer**: the bench takes `--slots`, `--global-batch-cap`,
  `--max-inflight-batches`, `--scheduler-workers`
  (`f4_throughput_bench.py:1055-1068`); Phase D takes the `--rust-` prefixed
  forms (`phase_d.py:2208-2211`). Unit-test the mapping. It fails loudly today
  (argparse rejects unknown flags) but silently produces a hand-copy step.
- **W6.4** *(S)* Preflight memory sizing against the **maximum scheduled
  window** from W1.1, not the nominal current one, using W2.3's calibrated
  factor plus headroom. Fail at setup, not at 3 a.m.
- **W6.5** *(S)* **Pin the commit and refuse incompatible resume.** The manifest
  records `git.commit` (`manifest.py:73`) and resume never compares it;
  `_validate_lifecycle_config` (`run_controller.py:238`) checks only lifecycle
  fields and checkpoint hashes. Record commit + `--precision` + schedule
  identifiers, and refuse a resume whose code or config differs unless
  explicitly overridden.
- **W6.6** *(S)* Run-health heartbeat: one line per iteration (iteration,
  games/s, RSS, best_iter, last gate LCB, anchor score).
- **W6.7** *(S)* `snapshot` helper for manual downloads. Copying mid-iteration
  can produce an inconsistent set -- `pending_iteration.json`, `_recovery/`,
  `run_manifest.json`, and the newest buffer file are written at different
  moments. Wait for an iteration boundary, write the manifest last, report the
  minimal resumable set. Per-iteration buffers are immutable, so incremental
  pulls are ~12 MB/iteration.

**Acceptance:** fresh box → training launched unattended in one command; the
equivalence suite **runs** (zero unexpected skips) on the rented GPU before
training starts; launch flags come from that box's measured sweep; re-running
resumes; a resume on a different commit is refused.

---

## W7 -- Stagnation

Both halves ship **before** launch. Revision 2 had the intervention ladder
landing mid-run; that does not work -- a running process will not pick it up,
and restarting after a pull changes controller semantics mid-run, which W6.5
will (correctly) then refuse. **W7b ships present-but-disabled**, enabled by
config at launch or at a deliberate restart.

### W7a -- Detection

Run 03's Elo cannot detect stagnation by construction: every candidate played
only its own `current_best`, so the ladder has no fixed reference.

A **promotion-lagged** anchor fails too, and fails exactly when it matters. With
anchor = `best[-K promotions]`: while promotions land, both pointers advance and
the score is meaningful. When promotions stop, **both pointers freeze** -- you
are comparing two frozen networks, the score is a constant (say 0.60), and a
`< 0.55` threshold never fires. It answers "were the last K promotions real?",
and when promotions stop, the numerator and denominator stop with them.

**The anchor is indexed by games.** Anchor = whatever `current_best` was N games
ago (e.g. 20k). While learning, the anchor trails a genuinely weaker net. When
learning stops, the anchor **advances until it catches up to the frozen
`current_best`**, and the score converges to exactly **0.50** -- a net against
itself. Unambiguous, self-calibrating, cannot go stale.

Specification: fixed N games, **no early stopping** (measurement, not decision --
W5.6); report the score with its Wilson interval; trigger on the **interval or a
slope across several measurements**, never a single point; before N games have
elapsed, use the bootstrap net or skip rather than comparing against nothing.

Also in the heartbeat: promotions in the last M games; val `value_acc` trend
(run 03's turned at iteration 35, ~25 iterations before the run died); victory
mix drift against the ZeusAI/BGA reference. The one automatic stop stays
`--revert-reset-after`.

### W7b -- Intervention ladder (present, disabled by default)

States NORMAL → STAGNANT → INTERVENTION(k) → RE-MEASURE, escalating one rung at
a time with a fixed measurement window between rungs so each effect is
attributable:

1. **Raise the search budget** (`full_sims_max`, `full_search_fraction`) -- the
   most principled first response: stagnation usually means the search-improved
   policy is no longer better than the raw policy at that budget.
2. **Jump the replay window** up its schedule.
3. **Raise the HOF opponent fraction.**
4. **LR warm restart or decay.**

**Model growth is not on this ladder.** Size is manual.

**Schedule vs metric:**

| lever | today | plan |
|---|---|---|
| curriculum bot mix | schedule (iterations) | **schedule, in games** (~10k) |
| draft prior anneal | schedule (iterations) | **schedule, in games** |
| replay window | fixed | **schedule (power law in games)**, metric-triggered jump |
| sims budget | fixed ranges | schedule default, **metric-triggered raise** |
| HOF opponent fraction | absent | config, **metric-triggered raise** |
| LR | schedule (warmup) | schedule, **metric-triggered restart** |
| model size | fixed | **manual** |

Principle: schedules for what should change regardless; metrics for what should
change only on evidence. All schedules in **games**.

---

## Sequencing

```
W0 (size + precision)
   └─► W1 (schedules, window, HOF) ─► W2 (memory) ─► W3 (stats) ─► W5 (gates) ─► W6 (cloud) ─► W7 ─► LAUNCH

W4 (BGA advisor, iter-60) ── parallel throughout
```

W0 first: size and precision change every downstream sizing number.
W1 before W2/W5: HOF league sampling changes the memory profile and batch
composition their acceptances measure.
W2 before W3: memory telemetry is a stats-schema field.
W3 before W5/W6: an unattended multi-day run is only worth launching if it can
be read afterwards -- the 27% generation decay is already unexplained.
W5 throughput before the cap increase: the fixes pay for it.
W7 in full before launch: W6.5 will refuse a mid-run pull.

---

## Out of scope

- **The science gap.** W3 makes it visible per iteration; closing it is a
  decision for after the cloud run produces data.
- **Model growth mid-run.** Manual.
- **Promotion rollback / HOF revalidation.** Deferred, recorded as a known
  residual risk in W5.
- **Multi-box sharded generation.** `phase_e/launch_shards.sh` is the precedent
  if throughput becomes the limit.
- **Kingdomino migration onto the lifecycle controller.** W3 makes it possible
  later, not required now.
- **A games budget.** Replaced by strength-per-GPU-hour against the games-indexed
  anchor.

---

## Appendix -- measurements (2026-07-28)

**A1. Memory footprint.** 100 games from `buffer_final.jsonl` through the real
derivation path, RSS-delta: **17.8 KB per `Example`**, **122 KB per
`GameRecord`**. `Example` holds **six** numpy arrays summing to ~13.1 KB
(features dominate: 49 tokens x 130 float16 ≈ 12.7 KB), so `nbytes`
**understates the real cost by ~26%**. At run 03's settings: 250k-example cache
= 4.45 GB, 8,000-game window = 0.98 GB, ~5.4 GB live since iteration 25 on a
16 GB laptop. `phase_d.py:1412` assumes 12.5 KB -- 1.4x optimistic.

**A2. Crash chain.** `run.log:151` bare `MemoryError: ` (pyo3 `Display for
PyErr`, verified against pyo3 0.28.3 `src/err/mod.rs:663`) → worker breaks at
`eval.rs:670` → `ValueError` at `eval.rs:549`. Unchecked allocation site:
`eval.rs:426-443`.

**A3. Generation NN demand.** 28,233 searched moves x 39.08 avg sims = 1.10 M
search evals/iteration; forced root-chance rows ~55% of all NN rows (throughput
programme), so ~2.45 M rows over 494 s ≈ **4,960 rows/s**, against 14,118 rows/s
capacity at 1M params fp32 → **~35% NN utilisation**.

**A4. Precision fidelity and speed.** 512 real positions through
`current_best.pt` (iter 60):

| | mean policy KL | max dP | argmax agreement | mean dP(win) | max dP(win) |
|---|---|---|---|---|---|
| TF32 | 7.9e-08 | 6.5e-04 | **100.00%** | 9.0e-05 | 4.2e-04 |
| bf16 | 8.3e-06 | 5.3e-03 | 99.61% | 1.1e-03 | 5.3e-03 |

The 2 bf16 flips had top1-top2 prior gaps of 0.0016 (overall mean 0.426).
Speedups @b256: TF32 **1.6-2.0x**, bf16 **2.3-3.0x**. Path audit in W0.3:
generation/gates/validation/advisor are fp32 today; **training is already AMP
fp16** (`train.py:488`, no dtype → fp16 on CUDA).

**A5. Gate cost.** Per-iteration wall clock from buffer mtimes minus generation
seconds. Non-gate baseline **57 s** (training ~10 s + replay derivation ~45 s).
Gates: **1.08 h / 11.06 h = 9.8%** for 1,240 games = **3.14 s/game** vs
**1.09 s/game** self-play (**2.9x**). Two-point fit: **193 s fixed + 1.74 s per
game**; 1.74/1.09 = 1.60x matches the 64/39.1 = 1.64x sims ratio.

**A6. Victory mix, game length, bots.** Self-play only:

| iterations | n | civilian | military | scientific |
|---|---|---|---|---|
| 00-09 | 3,475 | 75.0% | 17.1% | 7.9% |
| 30-39 | 3,857 | 69.4% | 16.9% | 13.7% |
| 60-69 | 4,000 | 70.9% | 16.1% | 13.0% |

2 sd at n=4,000 is ±1.1%. Gate games (64 sims, argmax, n=1,240): 69.0/15.7/15.2%
-- non-civilian endings are as common in argmax play as in self-play. Length,
iters 60-69: military 64.0 mean moves, scientific 65.1, civilian 73.0 (p10
55/56/71) -- Age III conversions, not rushes. Net vs curriculum bots: 58.7% (mil)
/ 78.0% (sci) at iters 00-09 → 98.6% / 91.4% by 30-39; **exhausted as opponents
by ~iteration 20** (~10k games), which sets W1.2's anneal.

**A7. Gate rule replay (Kingdomino Wilson LCB, z=1.96, min_lcb 0.50, revert
0.48).** Agreement with run 03's SPRT decisions: **9/13**.

| gate | n | win rate | LCB | KD rule | run 03 |
|---|---|---|---|---|---|
| 5 | 20 | 0.900 | 0.699 | promote | promote |
| 15 | 110 | 0.573 | 0.479 | probation | promote |
| 20 | 74 | 0.608 | 0.494 | probation | promote |
| 25 | 200 | 0.530 | 0.461 | probation | probation |
| 35 | 56 | 0.643 | 0.512 | promote | promote |
| 50 | 200 | 0.465 | 0.397 | revert | probation |
| 60 | 154 | 0.552 | 0.473 | probation | promote |
| 65 | 96 | 0.422 | 0.328 | revert | revert |

Win rate needed for LCB > 0.50: **0.598** @100, **0.569** @200, 0.549 @400,
**0.535** @800. A Wilson-UCB futility stop **never fires at n <= 200** (a 0.422
win rate at 96 games still has UCB ~0.52); it begins paying near n = 800.

**Caveat on A7 and the superseded revision-2 replay:** both treated seat-paired
games as independent. W5.5 makes the *pair* the observation unit, which is
exact; the absolute bounds above are therefore mildly anti-conservative and
should be recomputed at implementation. Relative comparisons between rules are
unaffected, since the same approximation applies on both sides.
