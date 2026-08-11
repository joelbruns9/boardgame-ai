# 7WD plateau: what was tested, what was found, what is left

**Written 2026-08-10, after cloud6 was stopped at iteration 43.**

Read this before restarting any 7WD strength work. Its purpose is to stop the
same hypotheses being re-tested. Several of the obvious ones are dead, and two
of the deaths are counter-intuitive enough that they will otherwise look like
fresh ideas in six months.

Every number below is measured on this project's own data unless explicitly
flagged as an inference. Retracted claims are kept, not deleted — knowing what
looked true and wasn't is most of the value here.

---

## 1. The situation

`cloud6` was seeded from cloud5's `current_best` and ran 43 iterations / 43,000
games with **zero promotions**.

| gate (iteration) | score | n |
|---|---|---|
| 15 | 0.495 [0.439, 0.551] | 600 |
| 20 | 0.470 [0.414, 0.527] | 600 |
| 25 | 0.516 [0.472, 0.560] | 1000 |
| 30 | 0.510 [0.466, 0.554] | 1000 |
| 35 | 0.499 [0.464, 0.535] | 1500 |
| 40 | 0.503 [0.468, 0.539] | 1500 |

The decisive stop signal was the **first self-anchor at iteration 40: 0.488
[0.444, 0.532]** against its own ancestor 40,000 games earlier. That is
independent of the gate machinery and immune to "maybe the gate is insensitive".
40,000 games of training produced no measurable strength change.

A probation-reset fired at iteration 30 (`--probation-reset-after 4`), which
logs as `action=revert_reset`, not `probation_reset`.

### The two runs compared throughout

Identical: `lr 5e-5`, `train-steps 190`, `batch 512`, 1000 games/iteration,
`replay-window-cap-games 40000`.

| | cloud5 | cloud6 |
|---|---|---|
| full sims | ~512 (`--full-sims-min 256`, recorded 507–512) | 1600 fixed |
| cheap sims | 8 | 100 |
| search | Gumbel | PUCT + Dirichlet (ε 0.25, α 1.8) |
| games/s | 1.38 | 0.33 |
| outcome | **promoted at iters 20 and 70** | **never promoted** |

cloud6 spent ~6× the search compute per full move and learned nothing. cloud6
was also the first run ever to use PUCT self-play and the first with Dirichlet
actually switched on.

---

## 2. Hypotheses tested and killed

### 2.1 Cheap-search targets polluting the policy — RULED OUT

Playout-cap randomization is correctly implemented; cheap moves emit no policy
targets (`phase_d.py:1459`, `dataset.py:167`). Confirmed end-to-end three ways:
`full_search_fraction` defaults to 0.25, predicting mean sims
`0.25×1600 + 0.75×100 = 475` against **473 logged**; ~16,100 new positions per
1,000 games ≈ 16/game ≈ 25% of a ~64-move game; and directly from the buffer,
**52,090 full rows against 157,116 cheap = 24.9%**.

Note `is_fast_search_move` is `policy_excluded and sims > 0` — bot/archive moves
(`sims == 0`) are *not* cheap searches and still contribute value labels.

### 2.2 `--weight-decay 0.5` — RULED OUT (independently, twice)

Already nulled by `weight_decay_probe.py` on 2026-08-08; re-derived here from
checkpoints without knowing that, and agreed.

- Whole-net RMS moved **0.065488 → 0.064881 (ratio 0.9907)** across 20 iterations.
- A synthetic AdamW probe (`lr 5e-5`, 200k steps) found the *documented default*
  of `1e-4` is **numerically inert** — identical to `wd=0` to 7+ digits
  (both 1.094308). So `0.5` is roughly the smallest value that acts at all at
  this learning rate. **Do not "fix" it back to 1e-4.**
- Mechanism confirmed but consequence nil: equilibrium `|w*| ≈ (signal/noise)/λ`
  (predicted 0.200, measured 0.2026). Pure decay over 4,940 steps takes 0.100 to
  0.088393.
- There is **no threshold** below which a gradient fails to move a weight;
  signals of 1e-8 and of exactly 0 both move it. Decay sets the *equilibrium
  scale*, it does not gate motion.

LayerNorm gains are all below their 1.0 init (mean 0.68, min 0.60, max 0.847),
drifting ~−0.0026/iteration along a lineage and restored by a reset. Real, and
too small to matter.

**Retracted:** the prediction that no weight can exceed `1/λ = 2.0`. Falsified —
11 tensors exceed it, to 2.84 (embeddings). The bound assumed
`|m̂/(√v̂+ε)| ≤ 1`, which fails for coordinates whose gradient is one-signed and
growing.

### 2.3 "The learner isn't moving" — RETRACTED

cloud6's learner covered only **1.56% of its per-coordinate travel budget** over
1,900 steps (relative movement 5.08%), which looked damning. It isn't. cloud5,
with identical hyperparameters and two promotions:

| run | span | relative movement | travel / budget |
|---|---|---|---|
| cloud5 | 20→30 | 4.96% | 1.54% |
| cloud5 | 25→35 | 4.95% | 1.51% |
| **cloud5** | **65→75** (contains the iter-70 promotion) | **5.01%** | **1.57%** |
| cloud5 | 75→85 | 4.98% | 1.48% |
| cloud6 | 15→25 | 5.05% | 1.60% |
| cloud6 | 30→40 | 5.05% | 1.60% |

The span containing a genuine promotion is indistinguishable from cloud6's flat
ones. **Weight travel is not diagnostic at these settings.** ~98% of AdamW's
motion cancelling is simply what this optimizer does here, in working and broken
runs alike. Do not resurrect this.

### 2.4 Search exhausted / "the model is maxed out" — RULED OUT

`search_gain_probe.py`: same checkpoint both sides, 1600-sim search versus the
bare network (`--weak-sims 1`), seat-paired, n=200, all decisive.

**0.850 [0.794, 0.893].**

**Interpretation caveat:** a large net-vs-search gap is *normal and permanent* in
AlphaZero — a single forward pass cannot replicate 1600 simulations. This rules
out "search has degenerated / the targets are worthless". It does **not** prove
learnable headroom exists. The 0.65/0.52 decision thresholds originally written
into that tool were too generous to it.

### 2.5 Target noise / label-noise ceiling — RULED OUT

`target_repeatability.py`: same position searched twice at 1600 sims under
different seeds, n=300.

| | rate | 95% interval |
|---|---|---|
| search vs **itself** | **0.973** | [0.948, 0.986] |
| search vs **bare net** | **0.777** | [0.726, 0.820] |

Targets are sharp, not noisy. Independently corroborated: `search_vs_prior`
0.777 lands on the training log's `policy_top1` ≈ 0.78 from a different corpus
and code path.

**Retire cloud4's "~20% label noise" figure for PUCT.** That was measured on
*Gumbel* targets; PUCT at 1600 sims carries **~2.7%**.

Caveats: this cannot see Dirichlet root noise (that lives in `self_play.rs`, not
the search entry point), so real targets are somewhat noisier — conservative for
a ceiling finding, optimistic for a headroom one. And reproducible ≠ learnable
by a feedforward net; a deterministic 1600-sim result can still be uncomputable
in one pass.

### 2.6 Target entropy collapse / `c_scale` — RULED OUT, and it inverts

Hypothesis was that cloud6's 6× sims increase collapsed target entropy to
one-hot without the `c_scale` compensation (which is not CLI-exposed;
`lib.rs:285`). **The opposite is true**, measured on ~50,000 full-search targets
per run:

| | cloud5 — Gumbel (**promoted**) | cloud6 — PUCT (**never**) |
|---|---|---|
| median entropy | **0.0118 nats** | **0.6504 nats** |
| mean entropy | 0.2005 | 0.7099 |
| median peak mass | **0.9985** | **0.8044** |
| peak > 0.95 | 70.8% | 22.8% |
| peak > 0.99 | 60.0% | 3.7% |
| exactly one-hot | 11.6% | 0.4% |
| mean legal actions | 7.1 | 7.1 |

cloud6's targets are **55× higher entropy** — textbook distributional targets.
The run that *learned* had near-degenerate one-hot labels. Target shape is not
the differentiator, and `c_scale` is a dead end for this problem.

Why they differ: PUCT's target is normalized visit counts, so softness is
intrinsic to the exploration bonus. Gumbel's is `softmax(log_prior + σ·q̃)` with
σ scaling as `max_visits` — sharpness is a **transform artifact**, not
conviction. cloud4 already showed target entropy is *flat* across a 24×
difference in real Q spread.

### 2.7 The policy head has room to improve — RULED OUT

`train.py:66` computes `-(target · log_pred).sum()` — full cross-entropy, so the
floor is the target's own entropy. Anything above it is real model error.

| | policy loss | floor (target entropy) | **excess = real error** |
|---|---|---|---|
| cloud5 (promoted twice) | 0.748 | 0.201 | **0.548 nats** |
| cloud6 (never promoted) | 0.897 | 0.710 | **0.187 nats** |

**cloud6's policy head is within 0.19 nats of its information-theoretic
minimum**; cloud5 sat 0.55 nats above and was still promoting. Conservative, too
— if one position can draw different targets, the true floor is *higher* than
mean entropy, pushing cloud6 closer still.

**Trap:** raw policy loss is **not comparable across Gumbel and PUCT runs.**
cloud6's 0.90 vs cloud5's 0.75 looks worse and is almost entirely the floor
moving.

### 2.8 The value head is the bottleneck — RULED OUT

`value_ceiling_probe.py`, 6,439 positions from 400 late-run games:

| game progress | n | net value acc | search acc | share \|rv\|>0.9 |
|---|---|---|---|---|
| 0–10% | 657 | 0.677 | 0.680 | 0.000 |
| 10–20% | 628 | 0.697 | 0.685 | 0.000 |
| 20–30% | 648 | 0.708 | 0.701 | 0.000 |
| 30–40% | 654 | 0.745 | 0.735 | 0.002 |
| 40–50% | 654 | 0.769 | 0.769 | 0.031 |
| 50–60% | 686 | 0.761 | 0.770 | 0.079 |
| 60–70% | 650 | 0.752 | 0.777 | 0.151 |
| 70–80% | 643 | 0.843 | 0.860 | 0.292 |
| 80–90% | 654 | 0.859 | 0.887 | 0.445 |
| 90–100% | 565 | 0.920 | 0.966 | 0.825 |

Accuracy tracks how much of the game is actually decided. `|root_value| > 0.9`
is **exactly 0.000 for the first 30%** — a 1600-sim search also declines to
commit there. A head that says "I don't know" where a full search doesn't know
is not a bottleneck. The 0.715–0.74 headline that has held across five runs is
dominated by positions where the game is genuinely undecided.

The net matches search mid-game (0.769 vs 0.769) and falls behind only as
positions become calculable: +1.7, +2.8, +4.6pp over the last three deciles —
where 1600 sims is effectively solving, so most of that is the permanent
net-vs-search gap.

Also kills the naive "the net can't do the scoring arithmetic" story: the
encoder already computes and feeds `score_player` (`encoder.py:515`).

In-sample caveat: these positions are inside the training window, so accuracy is
optimistic (0.771 here vs the log's held-out 0.738). The *shape* and the
net-vs-search comparison carry the argument; both are affected equally.

---

## 3. Synthesis

There is no broken component left to find:

- policy head within 0.19 nats of its floor
- value head at search's level except where search is solving
- targets sharp (97.3% reproducible) and 55× softer than the run that learned
- weight movement identical to a run that promoted twice
- playout cap, weight decay, target noise, entropy collapse all eliminated
- five consecutive runs (cloud2–cloud6) stalling the same way

**This is a plateau, not a defect.** The model has extracted very nearly
everything available from this training signal. Encoder work and a larger model
are both expensive bets with small *measured* headroom behind them.

The encoder has been "the top suspect by elimination" since cloud4 and has
**still never been measured**. That status has not improved — it is what is left
after eliminating everything else, which is not the same as evidence.

---

## 4. Tools built (reusable, with guards)

| tool | committed | what it answers |
|---|---|---|
| `search_gain_probe.py` | `5964889` | Does search still beat the bare net? Same ckpt both sides, sims differ. |
| `target_repeatability.py` | `3501a5e` | Do two searches of one position agree? Separates target noise from unlearned structure. |
| `value_ceiling_probe.py` | *uncommitted* | Value accuracy by game progress, vs search, from buffers. No GPU search. |

Scratchpad-only: `adamw_decay_probe.py`, `weight_delta.py`, `travel_control.py`,
`target_entropy_compare.py`.

Both committed probes report **INCONCLUSIVE** rather than a null when the
interval is too wide. This is not decoration — the first smoke run of each
confidently announced the ceiling finding on n=4 and n=16 samples. The denial
curriculum's false null was exactly this failure.

### Gotchas that cost time

- **`target_repeatability.py` must pass production `force_root_chance=True` and
  `age_deal_samples=32`.** Without them the search is deterministic, two seeds
  agree 100%, and the measurement is silently vacuous.
- **The intervention ladder has never been enabled** — cloud4 *and* cloud6 passed
  `--intervention-window-games` but not `--intervention-ladder`, leaving
  `stagnation.json` at `rung -1 / state normal / history []`. Check on any
  future launch.
- `learner_NNNN.pt` on a reset holds the **restored best**, not the rejected
  candidate. Deltas across a reset boundary are not trajectories.
- cloud6's `learner_0000/0005/0010` are **bit-identical** — training was skipped
  through iteration 11 during buffer warmup — so `learner_0000.pt` *is* cloud5's
  `current_best.pt`.
- `--min-buffer-positions 200000` cost cloud6 **12 iterations (~10 h) of pure
  generation** before the first gradient step.
- Buffer `policy_target` is a `{action: prob}` **dict** in cloud5 and needs
  filtering by `sims` to isolate full searches.

---

## 5. Data captured (the instance is destroyed; this is all that survives)

`games/seven_wonders_duel/runs/cloud runs/`

- `cloud6_capture/cloud6/` — 2.0 GB, commit `3501a5e`: buffers `iter_0000`–`0043`,
  every 5th learner + `current_best` + `latest`, `training_log.jsonl` (218 MB),
  `hof/`, `search_gain.json`, `target_repeat.json`, `stagnation.json`
- `cloud5_capture/cloud5/` — 1.1 GB: `training_log.jsonl` (88 MB), 18 learner
  snapshots (iters 0–85), `run.log`, `heartbeat.log`, `stagnation.json`
- `cloud5_buffers.gz` — cloud5 buffers, iters 66–85 only
- `legacy_runs_keep.gz` — cloud2/3/4 training logs, cloud3 `current_best.pt`

cloud5's *early* buffers were not captured and are gone.

---

## 6. Open questions, in the order I would attack them

1. **Absolute strength — the actual next step.** Every measurement here is
   *relative*: can the learner beat its own ancestor. That number stopped moving
   five runs ago and says nothing about whether the current net+search is strong.
   The BGA advisor pipeline and CDP setup already exist. This decides whether
   you are sitting on a strong engine that needs shipping or a mediocre one that
   needs a different approach — and it changes which expensive bet is worth
   making. **This is what the project moved to on 2026-08-10.**

2. **Regret-weighted disagreement.** The 0.777-vs-0.973 gap is *unweighted*
   top-1: it scores a near-tie the same as a blunder. With a median peak mass of
   0.80, a soft target's argmax flips cheaply. Weighting by how much target mass
   the search assigns to the net's move versus its own would say whether that
   20-point gap is worth anything in Elo. Buffers + one forward pass per
   position; no search, no rental. **Cheapest remaining test with real
   decision value.**

3. **The intrinsic value ceiling.** Whether ~0.68 early-game accuracy is the
   game's own uncertainty is currently an *inference*. Measuring it needs
   repeated playouts from identical positions — the value-side analogue of
   `target_repeatability.py`.

4. **Encoder.** Still unmeasured, still by elimination only. Do not start here.

5. **Data scaling** (train on 5k/10k/20k/40k games, watch holdout `policy_top1`).
   Rationale is weak: the window was full at 40k and all of it came from a policy
   that barely changed, so it measures quantity while the plausible problem is
   diversity.

### Deliberately not recommended

- Lowering the learning rate. Standard for a saturated model, but cloud5
  promoted twice on identical settings, so the optimizer is not the difference.
- Raising sims. cloud6 already went ~512 → 1600 and got nothing.
- `c_scale` / `c_visit` tuning. Dead for this problem (§2.6).
- A bigger model. ≤0.19 nats of policy headroom, and the 2026-08-08 probe found
  the model overfits a narrow slice happily — not capacity-starved on fitting.
