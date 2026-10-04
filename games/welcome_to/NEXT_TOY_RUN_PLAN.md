# Welcome To… — next toy run: plan for review

**Date:** 2026-10-03. **Status: BUILT 2026-10-03** (commits 94c0a08, ac3223d and the
encoder-v4 commit) -- ready for the toy run, command in §4. Encoder v4: ABI 4, global
inputs 439, training shard version 7, Python/Rust equivalence 60,127 encodings with zero
divergences; the fixed paired benchmark re-encodes its positions with the current
encoder.
**Framing (owner, 2026-10-03):** the laptop model is a toy whose job is to show the
system *can learn* the pieces a strong player needs. The real model trains from
scratch on a rented cloud box afterwards. Breaking checkpoints and the encoder is
fine; evidence about what helps learning is what matters.

## 1. What the last runs established

| finding | evidence |
|---|---|
| Paired outcome labels (option A) break the plateau | v3_curriculum_01 iters 36–55: plans/seat 0.34 → 0.51, score 37 → 47, promotions at 40/50/55; the paired benchmark improved (clear-pair accuracy 69% → ~80%) |
| A teaches placement, not the capacity proxy | turn-16 capacity per empty box unchanged at ~0.57 |
| The model learns mid-size estate plans and nothing else hard | iter 55: estates 3+4 70%, 2+5 60%; **0%** for 6+6, 1×6, 1+1+1+6, 5 bis, all three pool plans, complete street |
| Plan endings never happen | `plan_ending_fraction` = 0.0 at every iteration (high-level human play: ~99%, owner's observation) |
| The network does not protect pool plans | it plays plan-killing placements 19% vs 14% of legal; after A its value is indifferent (48% keep) |
| Pool play is reachable and worth ~5 points per pool deal | pool rescue: pool plan done 0% → 10.7%, margin +5.2 [+2.7, +7.7], pool points +7.3 |
| Why A could not fix pools | A's playouts use the current policy, which never builds pool plans, so "keep the pool alive" is honestly worth ~0 under that policy (review §7.3) |

## 2. Goal and success criteria for the toy run

From scratch, compared with **v3_curriculum_01 at equal iterations** (same seed 60000,
so the deals match iteration by iteration). "Everything on" -- this run asks whether
the combined package learns the missing pieces; per-piece ablations come after, if
the package works.

| yardstick | v3_curriculum_01 reference | success looks like |
|---|---|---|
| `plan_ending_fraction` (games ending on 3 plans) | 0.0 everywhere | clearly > 0 and rising |
| per-plan completion, pool plans | 0% | > 0 and rising |
| per-plan completion, all plans | §1 table | broader, not only mid-size estates |
| pool share of points | ~7% | rising |
| learner score / plans per seat | 46.9 / 0.51 at iter 55 | reached in fewer iterations |
| paired benchmark (margin, clear pairs) | 69% → ~80% | at least as good |

These metrics move into generation itself (§3.5) so every iteration reports them
without throwaway scripts.

## 3. Components

### 3.1 Encoder v4 (one ABI break, Python + Rust, equivalence-gated)

1. **Remove the absolute viewer-seat one-hot** (review §2.4: learner rows are always
   seat 0, opponent inference runs as seats 1–3; 3% argmax change from the bit alone).
2. **Pool boxes in the plan planes.** Planes 19–21 ("plan k still needs a house
   here") are all zero for pool plans today. Mark, for pool plans, the empty pool
   boxes still usable in the streets the plan needs. A fact about the sheet, the
   direct analogue of the full-street plane -- not a valuation.
3. **Plan characteristics.** A fixed per-slot description shared across plans: kind,
   streets involved, needs pools / parks / roundabout / bis / temp / fences, the
   required estate-size multiset, first and later point values. Alongside the
   28-way identity, so "pool plans" form one family that learns together.
4. **A shared plan encoder** (network change). Each slot's features -- identity,
   characteristics, and each seat's requirement/progress block for that slot -- go
   through one small MLP shared by the three slots, the way seats share one sheet
   encoder. A plan is learned once, whichever slot it sits in; this is also what
   would make off-stack deals (3.3) transfer.

### 3.2 Option A with plan-aware playouts

Keep A as built (box choice, 48 shared futures, score-only paired loss). Add:

* **Plan-aware continuations for a share of paired roots.** When the learner holds a
  live pool plan, play out that root's candidates with the pool rule steering the
  learner's continuation (the rule from `pool_rescue.py`). The labels then say what
  keeping the pool alive is worth *when followed up*, which the network's own
  continuation can never show.
* **Card-choice comparisons** (later, if needed): compare the played card against
  other offered cards, where the "take the pool card" decision is made.

⚠ **Owner decision needed (D1).** The pool rule is hand-written. Here it steers
*exploration* only -- every label is still a real game outcome, nothing imitates
the rule and no target is a heuristic score -- but the outcomes are conditional on
a heuristic continuation. That is closer to the "no heuristic training signal"
line than anything so far. The alternative without it is card-choice A plus the
deal curriculum, which may never discover pool play.

### 3.3 Plan-deal curriculum

A minority of games (proposed 25%) deal plans weighted by how often the learner
completed each plan last iteration: easy plans first, harder plans entering as they
start completing. Gives the first experience of three-plan endings, first-finisher
races and ending the game while ahead. Reported separately, excluded from strength
metrics.

* Engine work: both engines accept fixed plan ids at construction; trajectories
  record them; every replay path honours them (Python replay, Rust capture,
  curriculum, paired targets, diagnostics). Equivalence-gated.
* **D2 (decided):** legal deals only; the off-stack switch is not built for now.

### 3.4 Curriculum restarts (existing) -- keep, small fix

Keep the 20% near-completion restarts. Fix from the review (§7.1): restart games
should be able to seed later restarts' archive so rare successes are not lost after
one iteration -- a bounded archive across iterations, stratified by plan type.
Lower priority; can wait.

### 3.5 Metrics in generation

Per iteration, in `trajectories.jsonl.metrics.json` and `progress.jsonl`:
plan-ending fraction, completion per plan id, the point mix (parks / pools /
estates / plans / temp and the penalties), "two plans done, third unfinished"
rate, turn of each plan completion. Ordinary games only, as today.

## 4. Run design

* From scratch, seed 60000, 500 games/iteration, 200 simulations, gates every 5.
* A from iteration 3 (it needs a policy that plays real games before its labels
  mean anything), 300 paired roots/iteration, of which the plan-aware share is
  proposed at 30% of roots that have a live pool plan (D1).
* Deal curriculum 25% from iteration 2; restarts 20% from iteration 2.
* **Helper schedule:** one switch, `--helpers-end-iteration` (proposed 12 of ~20).
  Each helper's share falls linearly from its start value to zero at that
  iteration; iterations after it are helper-free and are where success is judged
  (§2). Same pattern as the assist schedule, which ended at iteration 6.
* Length: ~27 min/iteration with A → about 20 iterations in a 10-hour night.
  v3_curriculum_01 only had A from iteration 36, so iterations 1–20 compare
  "package" against "curriculum only" directly.
* Launch as a separate Windows process (the memory-watchdog workaround) or by the
  owner in a terminal:

  ```
  .\.venv\Scripts\python.exe -m games.welcome_to.s2_run --run-dir runs/welcome_to_s2/v4_package_01 --iterations 20 --restart-fraction 0.2 --deal-fraction 0.25 --pairs-roots 300 --pairs-start-iteration 3 --pairs-plan-aware 0.3 --helpers-end-iteration 12 --gate-games 200 --inflight 512 --pairs-benchmark runs/welcome_to_s2/v3_curriculum_01/_sibling_probe_48/dataset.pt
  ```

## 5. Build order and size

| step | size | depends on |
|---|---|---|
| 1. Generation metrics (3.5) | small | — |
| 2. Encoder v4 + shared plan encoder (3.1) | large: Python + Rust + equivalence gate + network | — |
| 3. Fixed-plan deals in both engines + curriculum (3.3) | medium | — |
| 4. Plan-aware A continuations (3.2) | small–medium | D1 |
| 5. Smoke run, then the toy run | — | 1–4 |

Steps 2 and 3 are independent and could be built in either order. Estimated
total: a couple of days of build and test before the overnight run.

## 6. Decisions for the owner

**Owner answers, 2026-10-03:** D2 — legal deals only; do not build the off-stack
switch for now. D3 — all changes in one go, shared plan encoder included. D4 —
the proposed shares are reasonable starts. D1 — **yes**, include the pool-rule
playouts, **time-limited**.

**Helper principle (owner, 2026-10-03):** the pool-rule playouts and the other
early training helpers exist to teach key concepts. Every helper ends after a set
number of iterations, after which the model learns on its own from that foundation
and can find novel strategies. So every helper gets an explicit schedule ending in
zero, and the run's last iterations are helper-free -- they are the real test.
Helpers covered: pool-rule playouts, plan-deal curriculum, near-completion
restarts. (A itself is a training signal, not a scaffold; it tapers on the
benchmark rather than switching off by date -- see §4.)


* **D1** — may the pool rule steer A's continuations (exploration only, outcomes as
  labels)? If not, the pool-discovery route is card-choice A + deal curriculum.
* **D2** — deal curriculum legal-only by default, off-stack switch built but off?
* **D3** — include the shared plan encoder now (bigger network change), or start
  with characteristics + pool planes only?
* **D4** — share sizes: deal curriculum 25%, restarts 20%, plan-aware roots 30% of
  live-pool roots. Fine as starting points?

## 7. Risks

* **Too many changes in one run.** If it fails, attribution is hard. Mitigation: the
  metrics in 3.5 are per-component readable (pool share, plan endings, per-plan
  completion), and the package can be ablated afterwards.
* **Heuristic continuation bias (D1).** Labels from rule-steered playouts can
  overstate pool value if the network cannot actually execute the follow-up. The
  per-plan completion in ordinary games is the check.
* **From-scratch runs are slow to reach the interesting regime.** v3_curriculum_01
  needed ~35 iterations to plateau; 20 iterations may show learning speed but not
  the final ceiling.
