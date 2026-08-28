# Review request — S2 generation, inference, and playout-cap runtime

This request covers the production path from an in-flight game through Rust
search and inference to an atomically written training shard. It combines the
previously unreviewed committed S2 runtime with the current performance and
KataGo-style fast/full-search changes because they now execute as one path.

## 1. Scope

### Committed foundation

| Change | Commit | Main files |
|---|---|---|
| searched S2 trajectory loop | `9e900ba` | `self_play.py`, `rust_search.py`, scheduler/search glue and tests |
| generation hardening and CUDA sweep harness | `40e4532` | `self_play.py`, `s2_throughput.py`, tests and plans |
| resumable fixed-worker cloud scheduler | `df05885` | `scheduler.rs`, `search.rs`, evaluator and tests |
| direct Rust row capture and durable WTS shards | `d8509b3` | `samples.rs`, `self_play.py`, writer/oracle tests |

### Current working-tree slices

* Evaluator ABI v3 and raw heads: `network.py::forward_inference`,
  `rust_search.py::PackedNetEvaluator`, `scheduler.rs::evaluate_chunk`, the ABI
  constants in `lib.rs`, and their tests.
* Complete-turn playout-cap randomization: `SelfPlayConfig`,
  `full_search_game_seeds`, `full_search_for_turn`, and the full/fast scheduler
  routing in `self_play.generate`.
* GPU observability: CUDA-event sampling in `PackedNetEvaluator`, the in-process
  NVML sampler in `s2_throughput.py`, and `--cuda-events` plumbing.

The replay-window selection, league implementation, promotion policy, and dense
plan-target schema are owned by the other two requests even where they share a
file.

## 2. Runtime contract

* Only learner decisions with more than one legal macro are searched as targets.
* A full-search decision records the root's sparse visit distribution; fast
  decisions advance the real trajectory but emit no training row.
* The fast/full choice is made once per complete learner turn. The initial card
  choice and all later effect, plan, validation, and reshuffle decisions in that
  turn use the same budget class.
* Outside an exact 5% quota of wholly-full games, ordinary turns are 25% full
  and 75% fast. Full uses 800 simulations; fast uses 64.
* Fast turns receive no Dirichlet noise. The existing action-temperature
  schedule remains `tau=1` for the first ten turns and `tau=0` afterwards.
* Rust owns captured encoder rows, legal masks, sparse visits, metadata, and
  terminal targets. The raw portable trajectory remains in every WTS game so
  Python can independently replay and rederive it.

## 3. What is already gated

* Rust-captured samples are compared field-for-field with Python replay:
  encodings, legal masks, policy, action/actor/turn, and every float32 target.
* Writer restart, truncated shard rejection, atomic cleanup, seed resume, and
  mixed-run manifest rejection have focused tests.
* Scheduler width comparisons assert complete discrete trajectory equality.
* Fast-only test makes root-noise calls fatal and verifies no search rows are
  captured while games remain replayable.
* Playout-cap assignment is deterministic by game/turn, produces the exact
  whole-game quota, stays near the requested ordinary-turn fraction, and gives
  identical games and counters at scheduler widths 1 and 4.
* Selective inference hooks assert policy runs for all rows while rank and score
  heads run only for LEAF rows, with selected outputs matching the full path.
* ABI v3 validates row counts, ids, CSR offsets, legal support, finite logits,
  leaf ids, and head buffer lengths before routing responses.

## 4. Review focus

### 4.1 Must be correct — ABI v3 changes search arithmetic ownership

Python now returns sparse legal logits plus raw rank/score heads. Rust performs
segmented policy softmax, rank masking/softmax, and value blending. Review the
response-order contract carefully: policy rows are keyed by `request_id`, value
rows by `leaf_request_id`, and each CSR segment is in response order while legal
action identities remain in the original request.

There is not yet a dedicated v2-versus-v3 paired trajectory gate. The current
batch-width test compares v3 width 1 to v3 width 4, while the Python/Rust one-row
oracle exercises the blocking evaluator path. Please decide whether the existing
coverage proves strength neutrality or require a direct old-Python-postprocess
versus new-Rust-postprocess comparison over real trajectories before sign-off.

### 4.2 Float parity at softmax and value blend

Rust intentionally calculates the exponentials and sums in f32, normalizes rank
probabilities in f32, then widens to f64 for `blend_value`. Confirm this matches
Torch/NumPy closely enough for the stated discrete-equivalence contract. Pay
special attention to legal lists of very different widths, near-equal PUCT
scores, ties, and two/three/four-seat rank masks.

### 4.3 POLICY rows skip value heads without losing training effect

Roughly two-thirds of requests are simulated-opponent POLICY rows. They need
trunk and policy only; rank and score cannot affect the already-computed trunk.
Verify row-index creation/device placement, empty-LEAF batches, response shape,
and the assumption that no hook or future head mutates shared features.

### 4.4 Complete-turn fast/full state

Two persistent schedulers retain separate subtrees. Check the boundary where a
new real turn receives its class, especially after opponent moves, reshuffles,
and resumed generation. No later decision in that learner turn may switch caps.
No fast-root visit target or noise may leak into the durable shard, and a full
turn must not inherit a fast scheduler's retained subtree.

### 4.5 Direct capture and interruption durability

Review the ownership transfer from completed Rust game to the bounded writer
queue, close/flush behavior on clean exit, Ctrl-C, and Python exceptions, temp
file synchronization and atomic rename, restart indexing, and duplicate seeds.
The raw trajectory is the recovery/oracle record; verify it cannot disagree with
the captured row count or terminal state without a gate failing.

### 4.6 CUDA measurements should measure, not perturb, the sweep

CUDA events are sampled every 32 evaluator calls and drained at profile time or
after 2,048 pending pairs. Confirm event placement encloses the forward only,
that the eventual synchronization is not charged as forward time, and that the
2,048-pair safety drain cannot materially distort a long throughput arm.

The NVML sampler maps the Torch device by UUID when possible and never reports
unavailable telemetry as zero. Confirm multi-GPU index/UUID behavior, mid-run
driver loss, thread shutdown, and the semantics of `--require-gpu-telemetry`.

## 5. Known limitations and documentation debt

* `RUST_PORT_PLAN.md` still describes ABI v2/full priors and says postprocessing
  remains in Python. `PackedNetEvaluator.forward` also has a stale v2/f64-values
  docstring. Treat code and this request as the v3 description; update the plan
  after the review settles the contract.
* No post-v3 5090 inflight/worker sweep has been run.
* CUDA/NVML paths are hardware-specific and are not exercised by the CPU suite.
* Search-internal opponent POLICY requests still use the learner checkpoint;
  frozen opponent models govern actual opponent moves only. This is deliberate
  and is not changed in this request.
* Fast moves improve throughput by withholding targets. Whether 25/75 and
  800/64 improve strength is an empirical decision after correctness review.

## 6. Running the gates

Current focused result on 2026-08-27, as part of the combined review suite:
**138 Python tests passed with one test-only warning; 24 Rust tests passed.**

```powershell
.\.venv\Scripts\python.exe -m pytest `
  games/welcome_to/tests/test_network.py `
  games/welcome_to/tests/test_rust_search.py `
  games/welcome_to/tests/test_self_play.py -q

Push-Location games/welcome_to/welcome_to_rust
cargo test
Pop-Location

# CUDA-only smoke/sweep after correctness sign-off
.\.venv\Scripts\python.exe -m games.welcome_to.s2_throughput `
  --checkpoint MODEL.pt --games 512 --inflight 128,256,512 `
  --scheduler-workers 4,8,16 --simulations 800 `
  --monitor-hz 10 --require-gpu-telemetry --out SWEEP.json
```

## 7. Sign-offs requested

1. Does ABI v3 route every sparse policy segment and LEAF value to the right
   request under arbitrary response order?
2. Is a dedicated v2/v3 paired parity gate required before calling this a
   strength-neutral throughput change?
3. Can any POLICY row execute or consume rank/score outputs?
4. Is one fast/full choice preserved through every decision in a complete turn,
   with no fast noise or recorded row?
5. Are writer interruption and resume semantics bounded-loss and corruption
   detecting?
6. Are CUDA-event and NVML metrics trustworthy on a multi-GPU 5090 host?

---

# Review response — 2026-08-27

Three findings were applied and one missing gate was built; four more are
reported with measurements and left for you to decide. The six sign-offs follow,
and §8.4 answers the question this review was actually run to answer: why City
Plan completion is not being learned.

Suite after the changes: `games/welcome_to/tests` **580 passed, 1 skipped**;
Rust **25 passed**; the new ABI v3 postprocess gate green at 2, 3 and 4 seats.

## 8.1 Findings and disposition

### G1 — frozen-seat policy streams collide across games *(fixed)*

`self_play.py::_new_live`, new `self_play.derive_seat_stream`.

`PortableRng(seed ^ _POLICY_DOMAIN ^ seat)`. Seeds arrive as a contiguous block
and seats are 1..3, so the seat index sat in the same low bits as the seed and
game `s` seat 1 shared a stream with game `s ^ 3` seat 2. Measured over one
64-seed block: **128 of 192 frozen-opponent streams were duplicates.**

Two seats on one stream take correlated tie-breaks and correlated
temperature-`1` samples at the same decision index. That bites hardest in the
opening, from identical empty sheets against a shared policy — which is exactly
where `training.sheet_divergence` says the entire self-play divergence problem
lives. This is the same defect class as `derive_search_seed` (see
`RUST_M3_M6_REVIEW_REQUEST.md` F1) in a second place, and it takes the same
repair: one draw of the stream itself, `PortableRng(seed ^ DOMAIN)` advanced
`seat` states.

### G2 — the NVML sampler never resolved by UUID *(fixed, verified on this host)*

`s2_throughput.py::_GpuSampler`.

The lookup was `nvmlDeviceGetHandleByUUID(str(properties.uuid))` with a bare
index fallback. `str(torch_uuid)` has no `GPU-` prefix and NVML requires one.
Measured here:

```
str(uuid)  = '4f80abf9-12d5-0336-8e72-bca4796ed263'   -> NVMLError_NotFound
'GPU-' + str(uuid)                                    -> resolves
```

So the UUID path raised on **every** call and the index fallback ran every time
— the opposite of the intent. NVML enumerates by PCI bus id, CUDA does not, and
`CUDA_VISIBLE_DEVICES` remaps CUDA's indices while leaving NVML's alone, so on
the multi-GPU host this sweep is written for the sampler would attribute another
card's power, clock and utilisation to the run **while still reporting
`nvml_available: true`**. `--require-gpu-telemetry` cannot catch that: the
telemetry is present, it is just the wrong device's.

Now: prefix and encode the UUID, read the handle's UUID back and refuse it if it
disagrees, and fall back to an index only when exactly one GPU is visible, where
there is nothing to confuse it with. Two tests drive it with a fake NVML,
including the case where CUDA index 0 is NVML index 1.

Also in `__exit__`: `nvmlShutdown()` ran even when `join(timeout=5.0)` had not
actually stopped the sampler thread, which tears the library down underneath a
live caller. It now skips shutdown and records an error instead — leaking one
handle for the rest of a sweep process is the cheaper failure.

### G3 — the v2/v3 parity gate the request asks for *(built, green)*

New `rust_postprocess_equiv.py`, plus a fast case in `tests/test_rust_search.py`.

§4.1 asks whether existing coverage proves strength neutrality or whether a
direct old-postprocess-versus-new-postprocess comparison is required. It is
required, and it is buildable without resurrecting dead code, because **both
arithmetics are still in the tree**: `evaluate_request` keeps the NumPy dense
masked softmax and `mcts.blend_value`, while `forward` returns raw heads and
Rust does the segmented softmax, the rank softmax and the blend. Driving one
real trajectory through `RustMcts` and `RustCloudScheduler` on a shared tape is
exactly that comparison, and it is strictly stronger than comparing v3 to itself
at two batch widths.

Result — 18 trajectories, 1,785 decisions, 1,524 searched roots:

| seats | decisions | searched roots | trees | largest backup-total drift |
|---:|---:|---:|---|---:|
| 2 | 372 | 311 | identical | 3.03e-07 |
| 3 | 602 | 525 | identical | 2.72e-07 |
| 4 | 811 | 688 | identical | 5.52e-07 |

Root actions, visit counts and the chosen macro match at **every** decision. The
drift is `f32` rounding on the backup totals and is reported rather than
asserted away, because a large enough prior difference could in principle flip
PUCT's first-max tie-break; this gate is what would say so.

## 8.2 Reported, not changed

### G4 — "complete discrete trajectory equality" is stronger than the evidence

§3 records that scheduler width comparisons assert complete discrete trajectory
equality. On CPU, at widths 1 and 4, they do. **Your own CUDA sweeps record
otherwise**, in `runs/welcome_to_s2/inflight_sweep*.json`:

| file | arm | `trajectory_agreement` | mismatched games |
|---|---|---:|---:|
| `inflight_sweep_20260824` | inflight 16 | 0.906 | 3 |
| `inflight_sweep_20260824` | inflight 32 | 0.875 | 4 |
| `inflight_sweep_32_64` | inflight 64 | 0.906 | 6 |
| `inflight_sweep_64_128` | inflight 128 | 0.945 | 7 |

So 4.5–12.5% of games differ between width arms on CUDA. This is almost
certainly not a scheduler defect — the CPU width test passes, and G3 shows the
tree is identical whenever the logits are — but batched cuBLAS picks different
kernels and reduction orders by batch shape, so the logits themselves are
width-dependent, and a near-tie in PUCT occasionally resolves the other way.
**Inherent, not fixable, and not a training problem.** The claim in §3 should be
narrowed to "CPU, at the widths the unit test runs", and any statement that a
run reproduces bit-for-bit across widths should be withdrawn.

### G5 — batch width is the throughput problem, and it is structural (measured)

Measured on CPU, 64 games, 200 simulations, the real S0 checkpoint, inflight 64:

| arm | mean width | calls at width 1 |
|---|---:|---:|
| single arm (`playout_cap_randomization` off) | **9.0** | 27% |
| full arm (pcr on) | **3.9** | — |
| fast arm (pcr on) | **6.7** | — |

`max_batch` is 256 and never binds. Your inflight-256 CUDA sweep agrees:
`mean_batch` **20.7**, `batch_p50` **6**, and **47,098 of 258,315 calls (18%) at
width 1**, with `full_batch_fraction` 0.0013. Three causes compound:

1. **The learner group is `inflight / players`.** Only games with `actor == 0`
   are searched in an iteration, so at a 60/30/10 seat mix the ceiling is around
   40% of inflight before anything else happens.
2. **Playout-cap randomization halves what is left.** The full and fast classes
   go to two separate schedulers and two sequential `play` calls, so the arm
   carrying ~81% of the request volume (800 sims × 25% of turns against 64 × 75%)
   runs at the *smaller* width. Measured: 9.0 → 3.9.
3. **`RustCloudScheduler.play` blocks until the whole group finishes**, so its
   width decays toward 1 as sessions complete. That is what puts 18–27% of calls
   at a single row.

(1) is the cheap lever and `games_per_hour` was still climbing at inflight 256
(180 → 232 → 398 → 506 → 735 → 961 across 8 → 256). (2) would need a per-row
simulation budget on `PlaySession` so one scheduler can serve both classes;
`noise_required` would have to become per-row too, so it is a small API change
rather than a one-liner. (3) needs a scheduler that accepts new sessions while
others are in flight. **None of these were changed here** — they are design
changes that deserve their own gate, not a review-pass edit.

### G6 — two thirds of all requests are opponent policies that never change

Measured: 507,848 requests for 863 searched roots at 200 simulations = 2.9
requests per simulation, of which **63% are simulated-opponent POLICY rows**,
matching §4.3's estimate.

Those rows are re-evaluated on every simulation, and they need not be. The
opponent's encoder input is a function of their information set, and
determinization permutes only the *undrawn* deck — which changes neither
`deck_composition` (base minus table minus discard) nor any sheet. So the same
tree position yields a bit-identical opponent row on every simulation, and a
per-search memo keyed by the information key would collapse most of them without
touching a single RNG draw.

⚠ **That memo is sound precisely because of sign-off 1 of
`RUST_M3_M6_REVIEW_REQUEST.md`**: it is only safe if the information key
contains everything the encoder consumes, which is the containment property that
review established (and which finding F2 had to repair). Worth building; worth
building deliberately.

### G7 — the manifest pins throughput knobs, so re-tuning forbids a resume

`run_manifest` embeds `asdict(config)`, which includes `inflight`, `max_batch`
and `scheduler_workers`, and `ensure_run_manifest` requires exact equality. So a
partially complete overnight corpus cannot be resumed at a better width — which
is the whole point of `s2_throughput`. Given G4, pinning is defensible (width
does change which games you get); but the seeds still partition cleanly and each
game is an independent draw, so what changes is *which* sample, not what the
sample means. Splitting the manifest into corpus keys (seed, games, search
config, checkpoints, table signature) and runtime keys (inflight, max_batch,
workers, cuda_events) would let you re-tune mid-run. Not changed: it would
invalidate every manifest already on disk, which is a migration and your call.

### G8 — the shard writer does not fsync the parent directory

`samples.rs::write_shard` syncs the file and renames it atomically, but never
fsyncs the containing directory, so a host-level crash immediately after the
rename can lose the directory entry on POSIX. The loss stays bounded to one
shard and the reader detects truncation, so this is a note rather than a defect.

## 8.3 Sign-offs

1. **Does ABI v3 route every sparse policy segment and LEAF value correctly under
   arbitrary response order?** Policy segments are routed **positionally**, not by
   id: `response_offsets[response_row]` is read at the response's own row while
   `request_id` supplies the row→request mapping. That is the documented
   contract ("each CSR segment is in response order"), and `forward` satisfies it
   trivially because it never reorders. The length check
   (`compact_end - compact_start == legal.len()`) catches a reorder that changes
   segment widths but not one that swaps two equal-width rows, so "arbitrary
   response order" is safe for ids and for LEAF heads (keyed by
   `leaf_request_id`) but **not** for the policy segments. No caller reorders, and
   nothing needs changing; the contract simply must not be described as
   order-independent.
2. **Is a dedicated v2/v3 paired parity gate required?** Yes, and it now exists —
   G3. The prior coverage could not have caught a postprocessing bug: the
   batch-width test compares v3 to v3, and the one-row oracle exercises the
   blocking seam alone. With the gate green over 1,524 searched roots at 2, 3 and
   4 seats, ABI v3 is strength-neutral on the CPU tape.
3. **Can any POLICY row execute or consume rank/score outputs?** No.
   `forward_inference` gathers `h`/`h_seat` with `index_select(0, leaf_rows)`
   before the per-seat and global heads, so those modules never see a POLICY row;
   `leaf_rows` is moved to the feature device and cast to `long`; an all-POLICY
   wave produces `(0, …)` heads, empty buffers, and Rust's `leaf_count == 0`
   accepts them. `_features` is pure and the heads only read it, so no head can
   perturb the shared trunk. `validate_session_response` requires a value only
   for LEAF.
4. **Is one fast/full choice preserved through a complete turn?** Yes, and by two
   independent mechanisms. `full_search_for_turn` is keyed only on
   `(game_seed, global turn)`, so it cannot depend on scheduler width, dispatch
   order, or how many decisions the turn contained; and the class is re-drawn
   only when `game.state.turn` changes, which cannot happen mid-turn because the
   turn increments in `prepare_turn_boundary`. No fast row can leak: capture is
   inside `if full`, and fast rows pass `noises=None`. No subtree can leak either:
   `Search::retain` gives up at a turn boundary (`within_turn_successor` requires
   `next.turn == turn`) *and* `take_retained` compares a `position_key` that
   contains the turn, so the idle scheduler's stale retention is always rejected.
   Both schedulers are reset when a slot is recycled.
5. **Are writer interruption and resume semantics bounded-loss and
   corruption-detecting?** Yes. Loss is bounded to the in-memory shard; a failed
   flush records the error, stops admitting, does not advance `next_shard`, and
   removes its temp file; `close()` runs from a `finally` on clean exit, Ctrl-C
   and exceptions alike; shards are written to a pid-suffixed temp, `sync_all`ed
   and renamed. On the read side the shard header pins the version, both target
   counts, `MAX_SEATS`, the table signature **and the exact target names**, and a
   game record must consume its declared length, carry `sample_count ==
   len(trajectory.searches)`, and leave no trailing bytes. Duplicate seeds are
   refused on both the Python and Rust sides. See G8 for the one gap.
6. **Are CUDA-event and NVML metrics trustworthy on a multi-GPU 5090 host?** The
   CUDA events are: `record()` brackets only the `forward_inference` launch, the
   H2D copies are enqueued before the start event and so are excluded by stream
   ordering, and the synchronisation is charged to `postprocess_sync` rather than
   to forward time. The 2,048-pair safety drain cannot distort a long arm — at
   one sample per 32 calls it fires roughly four times in a 250k-call sweep, on
   events long since complete, and the final drain happens after
   `generation_seconds` is taken. NVML was **not** trustworthy before G2 and is
   now; `--require-gpu-telemetry` still only proves telemetry is *present*, so
   the UUID verification is what makes it prove the telemetry is *this device's*.

## 8.4 The City Plan question, measured

This review was run because plan completion is not being learned. It is not, on
this evidence, a plan-specific bug in the generation path. The measurements:

**The bootstrap failed its own gate.** `runs/welcome_to_s0/s0_metrics.json`
records `net_score` **21.83** against `greedy_score` **51.48** — a gap of
**−29.65 ± 2.03** over 60 paired games — with `gate_policy_agreement: 0.0` and
`gate_score_within_2: 0.0`. `datagen`'s module docstring states the bar plainly:
"If the net cannot reproduce greedy's mean score without search, nothing
downstream will work." It did not, and S2 has been running on top of it.

**S2 does not close the gap.** Independently reproduced here on CPU — 24 games,
200 simulations, the S0 checkpoint, the standard 60/30/10 seat mix:

| | GreedyBot | S2 learner (S0-seeded, 200 sims) |
|---|---:|---:|
| mean score | **50.84** | **32.13** |
| plans per seat-game | **0.406** | **0.180** |
| permits per seat-game | 2.21 | 1.97 |
| games ending on all three plans | 0.005 | 0.0 |

The learner score matches your recorded inflight-256 sweep (`learner_score`
32.14) to two decimals, so this is the same phenomenon, not a CPU artefact.
Search lifts 21.8 to 32.1 and stops there, 19 points short of a one-ply
heuristic.

**Two hypotheses tested and rejected**, so they need not be re-tried:

* *The confidence gate hides the score channel.* `blend_value` routes score —
  and therefore a plan's 6–13 points — only through
  `alpha · confidence · margin`, and `confidence` is zero exactly when the rank
  head is uniform. Measured over 4,317 learner decision nodes under the S0
  checkpoint: mean confidence **0.346**, median 0.255, and it is zero in
  **0.1%** of nodes. The channel is open.
* *The search cannot see that validating beats passing.* Measured on the
  `CHOOSE_PLAN` nodes that have two same-turn leaf children: the leaf-value gap
  between validating and passing is mean **0.093**, max **0.216** on a [−1, 1]
  scale. That is a large signal, and the search still puts **35%** of its visits
  on `PASS_PLAN`.

**What the corpus actually looks like.** Of 863 captured roots, 9 are
`CHOOSE_PLAN` and 7 are `VALIDATE_PLAN` — **1.9%** of the training rows carry
any plan decision at all. That is not a capture defect; the learner simply
reaches plan-scorable positions rarely, because completing a plan requires a
well-built sheet and the sheet play is the thing that is weak. Note the S0
head-quality profile agrees: `r2_turns_left` 0.92 and `r2_turns_to_plan_*`
0.75–0.78 (clock-like targets, easy), against `r2_score_plans` **0.085** and
`r2_plans_completed` **0.198**.

**So the plan symptom is downstream of a much larger problem: the policy is far
weaker than the trivial baseline it was cloned from.** Before tuning anything
plan-specific, S0 should be brought up to its own gate — `policy_top1` is 0.475
and `r2_score` is 0.20, both of which say the behaviour cloning did not converge,
not that the target set is wrong. Adding plan-specific supervision
(`PLAN_SIGNAL_REVIEW_REQUEST.md`) to a policy that scores 32 against greedy's 51
is unlikely to be the binding constraint.
