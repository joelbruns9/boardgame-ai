# Welcome To… — review request: paired targets, plan learning, and the toy-run package

**Date:** 2026-10-03. **Branch:** `main` at `5da865f` (pushed). **Previous review:**
`HYGIENE_AND_PLANS_REVIEW.md` (2026-10-02), answering
`HYGIENE_AND_PLANS_REVIEW_REQUEST.md` -- read that pair first; this request
assumes their game description and system summary (§1–§5 of the request).

**What is asked:** a review of the code *and* the reasoning behind everything built
since that review, before a ~10-hour from-scratch toy run. The run is the evidence
the owner will use to decide what goes into a rented-cloud model, so a conceptual
flaw found now is worth much more than one found after the run.

**Framing (owner, 2026-10-03).** The laptop model is a toy whose job is to show the
system *can learn* the pieces a strong player needs. Breaking checkpoints and the
encoder is fine. Hand-written guidance is acceptable **only** as a temporary
helper: every helper has a schedule ending in zero, and success is judged on the
iterations after the helpers switch off.

---

## 1. Scope

| commit | what | main files |
|---|---|---|
| `72fea46` | Review fixes: family-grouped train/validation split, gate rank uses the estate tiebreak, gate rank check renamed, Brier/log-loss skill metrics | `s2_train.py`, `s2_promotion.py` |
| `5469dab`, `71dc4db` | Sibling probe: fixed-dataset test of whether the network can learn same-card placement contrast; resumable collection | `sibling_probe.py` |
| `56ee36d` | **Option A**: paired placement targets in training | `paired_targets.py`, `s2_train.py`, `s2_run.py` |
| `a5b0813` | Pool-rescue diagnostic | `pool_rescue.py` |
| `94c0a08` | Forced City Plan deals in both engines; plan/point metrics in generation | `game.py`, `welcome_to_rust/src/{game,lib,samples}.rs`, `self_play.py`, all replay paths |
| `ac3223d` | Plan-deal curriculum; pool-rule-steered playouts for A; the helper schedule | `deal_curriculum.py`, `sibling_probe.py`, `paired_targets.py`, `s2_run.py` |
| `5da865f` | **Encoder v4**: seat one-hot removed, plan characteristics, pool targets, shared plan encoder | `plans.py`, `encoder.py`, `network.py`, `welcome_to_rust/src/{plans,encoder}.rs` |

32 files, +3,649 / −96 lines. Plan of record for the next run:
`NEXT_TOY_RUN_PLAN.md`.

## 2. What is already gated (no need to re-verify by hand)

| gate | result |
|---|---|
| Full Welcome To suite | 760 passed, 3 skipped |
| Rust unit tests | 28 passed |
| Encoder v4 Python ↔ Rust equivalence | **60,127 encodings / 18,757 states / 102 games, zero divergences** (every seat count, base and advanced, random / no-refusal / greedy drivers) |
| Forced deals | Python and Rust snapshots equal through whole games; a forced deal keeps the natural game's deck and stacks; off-stack deals refused by both engines |
| Rust capture vs Python replay | exact row and target equality for ordinary, curriculum-restart, assisted and forced-deal games |
| Planted bugs caught | Rust refusal window, Rust death dating, skipped restart reshuffle (Python), assist one turn short, focal arm assisting every seat |
| End-to-end | from-scratch 3-iteration `s2_run` smoke with every component on, helpers switching off at the set iteration |

## 3. Measurements that motivated the work

All single-seed; ± is one standard error unless an interval is shown.

### 3.1 Sibling probe (`sibling_probe.py`)

Frozen iteration-35 checkpoint, fresh games, roots chosen at random (not by any
heuristic), candidates = the played box plus same-slot, same-delta alternatives,
every candidate played **to the end** under shared redeterminized futures with
argmax policy for every seat.

| | run 1: 12+12 futures, 1,445 roots | run 2: 48+48 futures, 920 roots |
|---|---|---|
| label reliability (fit vs eval halves, score) | corr 0.42, sign agree 63% | corr 0.75, sign agree 75% |
| checkpoint, clear-pair accuracy | 53% | **67%** (blend) / 69% (margin) |
| best trained arm vs shuffled control | no difference | score-only fine-tune: **76%** vs 68% |

Lessons drawn: one game outcome carries ~1/50 of the signal needed to separate two
boxes (explains "data-limited"); run 1's "network is blind" was label noise; rank
labels gave nothing, score labels generalised; ranking candidates by predicted score
margin beat ranking by the search's blended value even untrained.

### 3.2 Option A in training (v3_curriculum_01, iterations 36–55)

Paired roots per iteration: 300, played box + 2 alternatives, 48 futures, generating
checkpoint's argmax continuation, score-only paired loss (pair weight 25), 16 roots
per training step. **No control run** (see §6, Q2).

| | iters 30–35 (no A) | iter 55 |
|---|---|---|
| plans per seat | 0.34–0.36 | 0.51 |
| learner score | 36–38 | 46.9 |
| decisions per game | 112 | 124 |
| benchmark clear-pair accuracy (margin) | 69% | ~77–82% |
| turn-16 capacity per empty box | 0.57 | 0.57 (unchanged) |
| promotions | at 35 | at 40, 50, 55 |

### 3.3 What the model can and cannot do (iterations 5–55, learner, ordinary games)

* Plan endings (`plan_ending_fraction`): **0.0 at every iteration** (owner:
  ~99% of high-level human games end on three plans).
* Learned: mid-size estate plans (iter 55: estates 3+4 70%, 2+5 60%, 1+2+2+3 54%).
* Never or almost never: estates 6+6 and 1×6 (0%), 5 bis (0%), all three pool plans
  (0%), complete street (0%); parks in two streets reached 18% only at iter 55.
* Point mix stable: estates ~41%, parks ~30%, pools ~7% (flat ~3.5 points/game).

### 3.4 Pool check and pool rescue

* Pool check (768 positions, iter-55 games): the learner plays plan-killing
  placements 19% of the time against 14% of legal placements; after A its value is
  indifferent (keep vs kill preferred 48%; 62% before A). Interpretation: A's playouts
  use a policy that never completes pool plans, so keeping one alive is honestly
  worth ~0 under that policy.
* Pool rescue (600 deals, 196 with a pool plan; rule changed 11% of learner
  decisions): pool plan completed 0% → 10.7% [+6.4, +15.1]; score +5.3 [+3.2, +7.4];
  margin +5.2 [+2.7, +7.7]; pool points +7.3; refusals −0.23; other plans unchanged.

## 4. What was built, and the choices to review

### 4.1 Option A — paired placement targets (`paired_targets.py`)

* Roots: learner WRITE decisions from the iteration's ordinary games, one per game
  per turn bucket, chosen at random; candidates: played box + policy-top + uniform
  alternative, **same stack slot and temp delta**.
* Labels: mean final per-seat score over 48 shared futures; paired loss
  `MSE(score) + 25 × MSE(Δscore between candidates)`, normalised per root; rank loss
  off by default; mixed 16 roots into every normal training step; split by game
  family.
* Continuation policy: the generating checkpoint's argmax for every seat.

### 4.2 Plan-deal curriculum (`deal_curriculum.py`)

A share of games deal one **legal** plan per stack, drawn with probability ∝ last
iteration's learner completion rate + 0.03. Plans are drawn as usual and then
replaced, so the game's cards match its natural twin.

### 4.3 Pool-rule-steered playouts (`sibling_probe._finish_all`, `pool_rescue.pool_choice`)

For a share of paired roots where the learner holds a live pool plan, the
**learner's** moves in the playouts follow the pool rule (take safe pool/park
progress, swap plan-killing writes, never pass a pool prompt). Opponents and the
labelled decision are untouched; the labels are still real final outcomes.

### 4.4 Helper schedule (`s2_run.helper_share`)

Restarts, the deal curriculum and the steered playouts fall linearly to zero at
`--helpers-end-iteration` and stay zero. A itself is *not* scheduled -- it is a
training signal, intended to taper on the benchmark.

### 4.5 Encoder v4

* Absolute viewer-seat one-hot removed.
* 26 plan characteristics per slot (kind, named street, needs pools / parks /
  roundabout / bis / temp / estates, estate-size multiset, estate count, stack).
* Pool targets in planes 19–21 for live pool plans (empty, still usable pool boxes
  in streets that can still finish their pools); dead plans mark nothing.
* Shared plan encoder: per slot, identity + characteristics + every seat's 34-float
  progress block → one MLP (→128→64) shared by the three slots → joins the trunk
  input. 4.40M parameters total.

### 4.6 The planned run (`NEXT_TOY_RUN_PLAN.md` §4)

From scratch, 20 iterations, 500 games, 200 simulations; restarts 20%, deal
curriculum 25%, A from iteration 3 (300 roots, 30% steered where eligible), helpers
end at iteration 12. Compared against v3_curriculum_01 at equal iterations (same
seed, matching deals).

## 5. Known limitations

* **Single seed** everywhere; the A result has no control (§6 Q2).
* **Label/policy staleness:** A's labels are `Q^π` for the generating checkpoint,
  re-made every iteration but trained over a 4-iteration window.
* **A covers box choice only**; card choice, fences, plan claims and roundabouts are
  still learned only from search targets and single outcomes.
* **The benchmark** is 143 held-out roots from iteration-35 games; it drifts out of
  the current policy's distribution as the model improves, and it measures box
  choice only.
* **Steered playouts are slow** (a Python state conversion per learner step);
  cost not yet measured at scale.
* **Plan characteristics** are hand-chosen features of the plan cards. They describe
  the card, not a valuation, but their choice is a judgement.
* **The pool rule is crude**: protects pool boxes and builds pools; no roundabout
  planning or range management -- which the owner says is what pool plans really
  need.
* **Everything-on run:** if it fails, attribution among encoder v4, A, steering, the
  deal curriculum and restarts is hard; the per-component metrics (§3.5 of the plan)
  are the mitigation.

## 6. Questions for the reviewer

**Conceptual**

1. **Steered playouts (§4.3).** Labels conditional on a heuristic continuation teach
   the value head that keeping pool plans alive is worth points *when followed up*.
   Is this a sound bridge to self-discovered pool play, or will it teach an
   overestimate the network cannot execute? What would you measure to tell?
2. **Attribution of the A result (§3.2).** Strength had been flat for ten iterations
   before A started. Is that enough to attribute the gain to A, or is a no-A
   continuation from iteration 35 needed before the cloud decision? The owner has
   chosen to fold attribution into from-scratch comparisons instead.
3. **The deal curriculum (§4.2).** Does skewing deals toward easy plans risk the
   network learning "three-plan endings are easy" in a way that does not transfer to
   natural deals, despite the plans being inputs? Is 25% with completion-rate
   weights a reasonable shape, and is the 0.03 floor enough for hard plans to start?
4. **The helper schedule (§4.4).** Is a linear decay to zero at iteration 12 of 20 a
   sensible test of "foundation then independence"? Should A taper too, and on what
   signal?
5. **Encoder v4 (§4.5).** Do the plan characteristics cross the "no hand-written
   valuation" line? Is concatenating the three plan embeddings (slot order kept) the
   right aggregation, versus a permutation-invariant pool or attention?
6. **Plan endings.** The model has never ended a game on plans. Beyond the deal
   curriculum, what would make "finish the third plan and end the game while ahead"
   learnable? Is anything in the value design (rank/margin blend, `end_trigger_*`
   heads) working against it?
7. **Blend vs margin (§3.1).** Ranking placements by predicted margin beat the
   blended leaf value. Should the search's leaf value change, and how would you test
   it without confounding it with the rest of the package?

**Code**

8. `paired_targets.paired_loss` and its use in `s2_train.fit`: normalisation per
   root, the interaction of its weight with the group-weighted ordinary loss, and
   the family split of paired roots.
9. Forced deals across every replay path (`SelfPlayTrajectory.new_python_state` /
   `new_rust_state`, Rust `replay_history`, curriculum restarts inheriting a source's
   deal, resume validation).
10. `pool_rescue.pool_choice` and `plans.pool_target_boxes`: correctness of "needed
    streets", the resolved-state check (`_resolved`) used to avoid misreading a
    progress move as a kill, and agreement between the Python and Rust
    `pool_target_boxes`.
11. `network.py`'s shared plan encoder: slice indices into `global_scalars` and
    `sheet_scalars`, and whether padded seats' zero progress blocks are handled
    correctly (they enter the plan encoder as zeros).
12. Anything in the review fixes (`72fea46`) that does not do what its commit says.

## 7. How to run the gates

```
.\.venv\Scripts\python.exe -m pytest games/welcome_to/tests -n 8
cd games/welcome_to/welcome_to_rust; cargo test --release
.\.venv\Scripts\python.exe -m games.welcome_to.rust_encode_equiv --encodings 60000
```

Diagnostics: `sibling_probe.py` (collect / probe), `pool_rescue.py`,
`hygiene_rescue.py`. Run artefacts: `runs/welcome_to_s2/v3_curriculum_01/`
(`progress.jsonl`, `_sibling_probe*`, `_pool_rescue`, `_plans_and_points.json`).

## 8. Requested sign-offs

* Steered playouts as a scheduled helper (Q1) — proceed, change, or drop.
* The everything-on run design (§4.6) — proceed as planned, or split into
  attributable arms first.
* Encoder v4 (Q5) — accept as the representation for the cloud model, or change
  before the toy run.
