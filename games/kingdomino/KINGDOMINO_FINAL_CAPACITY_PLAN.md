# Kingdomino final improvement attempt: capacity + data plan

- **Status:** **Closed negative on 2026-08-14.** P1 and P2 both failed their
  frozen hard gates. Per Section 6, there is no build and no 5090 rental;
  `current_best` ships and this model line is closed.
- **Date:** 2026-08-14
- **Incumbent:** `runs/kingdomino/best_checkpoint/current_best.pt`
  (sha `4bf07b0c…`, 80x6)
- **Owner decisions recorded here:** open-loop search is the frozen champion —
  no further search work. Contextual open loop / CORAL is **closed by decision
  without running Phase 0** (low expected value given Gate-0 parity; recorded
  so no future session reopens it). The final attempt is a larger network plus
  better data, gated against the incumbent.

## 1. What changed to make this worth one more attempt

Three facts, two of them new today:

1. **Stage-3 correction (new).** The original deep-target null was an artifact:
   the Rust `root_allowed_actions` mask leaked via open-loop missing-child
   recovery, so no matched tile search ever received its full budget. The
   corrected development audit **passes** its predeclared gate — deeper tile
   choices beat 4,800-sim choices with a positive cross-seed, game-clustered
   lower bound (+0.00038 to +0.01847 Q), concentrated in close-Q "contested"
   positions. The frozen confirmation later failed; see Section 8 and
   `DEEP_TARGET_STAGE3_FINDINGS.md`.
2. **The capacity loophole was untested at meaningful scale.** Every
   null ran at 80x6; the recorded 96x6/80x10 bake-off arms were <1.7x params
   on an exhausted buffer. A 3-4x jump with global pooling has never been
   tried before this plan. Test A below subsequently closed that loophole.
3. **The BGA corpus exists.** 1,400 reconstructable strong-human roots supply
   restart states the self-play loop provably does not generate (run11a).

Everything else stays closed: search topology, chance conditioning, KL
anchors, human one-hot labels, broad 30k self-play, placement supervision as a
weakness fix.

## 2. Preconditions — all local, all cheap, run before any build

P1 and P2 are logically independent — neither gates the other — but their GPU
workloads are **serialized** on the local machine: concurrency would distort
P1's search timings and possibly Test A's training behavior. Recommended
order: P1 first (shorter; its verdict scopes whether the reanalysis worker
gets built), then Test A. Both are kill-capable.

### P1 — Deep-target confirmation pass (hard gate for the reanalysis component)

Run the **frozen** staged method on the untouched 460-root confirmation split:
same selection rules (800-sim repeats → close-Q cohort → paired 4,800 →
suspicious cohort), same corrected-mask 10k matched-tile searches, same
cross-seed statistic and game-level bootstrap. No adaptive changes.

- **Pass (LCB > 0):** selective reanalysis is authorized as a training
  component (Section 4.4) and sidecar capture gets its primary purpose.
- **Fail:** the reanalysis component is dropped; sidecars demote to a
  diagnostic corpus; the rest of the plan proceeds unchanged.

Dev-scale restricted work took ~17 minutes on the laptop 3070; confirmation is
larger but still laptop/local scale.

### P2 — Capacity Test A: fit ladder on the frozen buffer (hard gate for everything)

**Prerequisite build:** the global-pooling trunk must be implemented before
Test A runs — pooling is part of the candidate's identity, so the test must
compare the actual candidate architecture, not a plain wide net. This is a
small self-contained change (pooling blocks only; no new heads, no sidecars,
no restart path).

Train on the identical frozen replay buffer with matched optimization effort.
The stored `Example` format has no game provenance, so a true game-level
split is impossible on the old buffer; `capacity_bakeoff.py` approximates it
with contiguous 400-example blocks (one game's examples are adjacent). Use
that block split and compute all Test A uncertainty with a **paired
contiguous-block bootstrap**. Do not describe any Test A interval as
game-clustered. Every newly generated dataset in this plan must carry game
IDs so future splits are exact.

| Arm | Purpose |
|---|---|
| 80x6 re-converged | Control — the incumbent's ceiling on this data |
| 80x6 + global pooling | Attribution control and cheap deployment candidate — if pooling alone closes most of the gap, an 80x6-sized deployment keeps full laptop sim throughput |
| 128x8 + global pooling | Primary candidate |
| 128x10 + global pooling | Depth probe — included only to earn its place |

Aux heads are **off** in Test A (the stored buffer has no ownership targets);
this is a pure policy-CE / value-Brier capacity probe.

**What Test A can and cannot show.** The buffer's targets are strictly
stronger than the network that generated the games: policy targets are
4,800-sim visit distributions (the policy-improvement operator's output, not
the 80x6 prior) and value targets are realized outcomes/margins (ground
truth, not the 80x6 value head). So the ceiling of this test is not "80x6
strength" — a better validation fit demonstrates real unabsorbed signal that
80x6 lacked capacity to extract. The test's power is asymmetric, and that is
its design: a **fail is decisive** (nothing left in the current data for any
size; cancel before renting), while a **pass proves potential, not
strength** — this project has already observed losses improving while
head-to-head strength did not (chance-cycle Phase B). Strength is only ever
proven by the Section 5 ladder.

- **Gate (decided 2026-08-13):** policy CE is primary. A large arm passes if
  its *relative* validation policy-CE improvement over the control seed-mean
  is at least 1.5%, the paired block-bootstrap interval excludes zero, and
  the gap exceeds the seed-to-seed spread. Value Brier must be non-inferior
  (no material regression); it is not required to improve, because outcome
  targets carry game luck and the distillation signal of interest is the
  policy fit. Significance alone is not a pass: with ~500 blocks the
  bootstrap can certify differences too small to matter (the tile-Q lesson).
- **Optimization matching (decided):** equal example presentations across
  arms with identical optimizer, batch size, and early-stopping rule — not
  equal wall-clock, which is P3's concern. A predeclared two-point LR grid
  ({1x, 0.5x} of base), applied identically to every arm including the
  control, best-by-validation per arm — this closes the "capacity null was
  really an LR mismatch" hole without opening a tuning garden.
- **Seeds (decided):** two optimization seeds for **every** arm from the
  start — identifying a leading arm from single seeds would select on seed
  luck. Additional seeds only if the top large arms are inside each other's
  spread and the winner matters for P3. The gate compares seed means; a gap
  smaller than the seed-to-seed spread does not pass.
- **Size selection:** take 128x10 over 128x8 only if it beats 128x8 on
  validation by a paired block-bootstrap interval excluding zero **and** its
  measured throughput cost (P3) is acceptable. Otherwise 128x8. Depth is not included
  for its own sake: receptive field saturated at 6 blocks; depth pays a
  sequential-latency tax on both the 5090 (self-play) and the laptop
  (advisor).
- **Fail (no arm beats control):** cancel the attempt entirely. Capacity is
  closed at 3-4x and the project ships. This is the cheapest possible way to
  find that out.

### P3 — Throughput and deployment feasibility (measurement, informs P2 choice)

For each surviving size, measure and record:

- 5090 self-play games/s with recalibrated `batch_slots` / `leaf_batch` / AMP;
- **laptop advisor simulations/s** on the 3070, same harness the live advisor
  uses, versus the 80x6 baseline.

The laptop number matters because the advisor is the deployment target: 80x6
currently reaches thousands of sims on tricky positions. A 3-4x net will not.
The bet — stronger priors compensating for fewer sims — is testable and is
gated in Section 5.3; nothing here assumes it.

P3 costs about an hour with random-weight candidates through the real
harnesses and feeds the 128x8-vs-128x10 choice, rental sizing, and salvage
likelihood. It is also a **pre-rental kill gate on deployment throughput**:
predeclare a floor — a minimum simulation count the candidate must reach
within the advisor's normal per-decision time budget on the laptop
(equivalently, a maximum slowdown versus 80x6; value in Section 7). A
candidate below the floor does not proceed to the rental as-is: either the
smaller P2-passing arm is selected, or the plan explicitly commits to the
Section 5.4 distillation salvage **before** the rental — discovering
catastrophic advisor slowdown after the cloud run would waste the rental.
Mitigating context: the measured sims curve showed no knee from 800 to
10,000, so the advisor's thousands of sims likely sit past the useful region
and a 2-3x sims cut may cost far less strength than the raw ratio suggests.

### Test B — absorption gap (diagnostic only, not a gate)

On the existing Stage-1/Stage-2 artifacts: net top-1 versus its own 4,800-sim
top-1, benchmarked against 4,800 cross-seed self-agreement on the same roots.
Costs minutes (forward passes; searches already exist). Sets expectations: a
small gap means gains must come from the new data, not better fitting, and
informs training length/LR.

## 3. Infrastructure build (only after P1/P2 verdicts)

1. **Architecture.** Winning size (pooling trunk already built as the P2
   prerequisite). **The official win head stays and remains the backup
   source** — P(margin > 0) is not official win probability because a zero
   margin resolves through the tiebreak cascade, so search continues to back
   up the existing official-outcome win head, unchanged. The
   **distributional score-margin head** (2-point bins spanning the realistic
   margin range, cross-entropy on the realized bin) is strictly auxiliary,
   which makes it genuinely droppable at inference. The one **new** auxiliary
   target is **per-cell final own-board state** (coverage/terrain/crowns via
   a 1x1 conv head) — the network already predicts own and opponent final
   scores, so scalar score heads are not new work. Aux heads are justified as
   sample-efficiency scaffolding for the capacity jump, not as a placement
   fix — the placement closure stands.
2. **Extended self-play targets.** Record final own-board state and final
   margin for every stored example (required by the new heads). Loss =
   `policy + value + margin-CE + small-lambda * ownership`, lambdas set by a
   smoke run with per-head loss/gradient telemetry.
3. **Sidecar capture.** A recorded full-search move is captured when **all**
   of the following hold, with Q defined at the aggregated tile-group level:
   - `abs(root Q) <= 0.4` (excludes saturated wins/losses where close tile
     Qs are meaningless);
   - at least two adequately visited tile groups (excludes noisy
     under-visited groups masquerading as contested);
   - top-two tile-group Q gap `<= 0.03`.
   Additionally capture a **small random control sample outside the filter**
   to estimate the correction rate the filter misses — without it, the
   filter's recall is unmeasurable (methodological rule 2). Each sidecar is a
   reconstructable `GameState` with game ID, Q values, visit distribution,
   deck count, and seed, versioned alongside buffer shards, volume-capped.
   Purposes: (a) in-loop selective reanalysis targets if P1 passed;
   (b) frozen hard-position eval suites; (c) restart-seeding reserve.
4. **Selective reanalysis worker (only if P1 passed).** Two-step label
   production, because equal-budget matched-tile searches cannot have their
   raw visits combined — the allocation gives every tile its full budget by
   construction, so pooled visits are meaningless as a policy:
   - **matched-tile Q** (corrected mask, frozen budgets) validates/selects
     the preferred tile;
   - an **ordinary deep search** at that root supplies the final joint-action
     visit distribution, which becomes the replacement policy target.
   Value targets stay game-outcome/margin — Stage-3 explicitly does not
   authorize raw deep Q values in the buffer. Cap the reanalyzed fraction per
   iteration; log it.
5. **BGA restart path.** Rust start-from-public-state; redeterminize only the
   unordered hidden bag; reconstruction/legality tests. Training restarts draw
   **only from the 940 roots belonging to the development-split games**; the
   confirmation-split games (460 roots) stay out of training entirely and
   remain the external anchor. No one-hot
   human claim labels anywhere (claim-order trap, documented).
6. **Gates plumbing.** Fixed-iteration nomination; sequential paired gate
   (512 → 1,024 → 2,048, pair-aware, alpha-spending, LCB > 50%); fresh seed
   blocks (all previously consumed blocks listed and excluded); fixed-suite
   evaluator running on the new architecture; cloud dependency check (the
   FastAPI lesson).

## 4. The training run (rented 5090)

- **Warm start:** initialize from the Test A winning checkpoint (it exists
  free of charge). Do **not** assume it is near incumbent strength — better
  frozen-buffer validation fit does not establish gameplay strength. Before
  renting, run a local 256-512-game paired sanity match against
  `current_best`. It does not need to pass any LCB gate; it only needs to
  exclude a badly broken initialization (predeclare a floor, e.g. >= 40%
  paired score). Note the alternative (from-scratch loop) was declined for
  compute reasons.
- **Self-play:** the large net generates its own games at the existing
  4,800/200 playout-cap schedule, open-loop search untouched. Restart mix is
  specified by **training-example mass, not episode count** — midgame
  restarts produce far fewer examples per episode, so an 85/10/5 episode mix
  would not yield an 85/10/5 buffer. Target ~85% ordinary / ~10% BGA-dev
  restart / ~5% sidecar-contested restart **of buffer examples**, controlled
  by measured examples-per-episode from the smoke run. Restart root sampling
  is two-stage: sample a source game first, then a position within it, so
  adjacent roots from one BGA game cannot dominate the restart stream.
  Selective reanalysis on flagged positions if authorized.
- **Optimization:** conservative LR (start 3e-5, smoke-tuned), modest reuse
  (~2.8x, the v1 lesson), buffer sized to the measured games/s so reuse stays
  honest.
- **Budget:** size for roughly 40-60k games — about 2-3x the v1 cycle,
  because the larger net is data-hungrier and games/s drops 2-3x. Day 0 is a
  G4-style feasibility probe (throughput floor, finite balanced losses, sane
  ownership predictions, legal restart games); a failed probe destroys the box
  before it burns money. Checkpoint every iteration; the run may be extended
  or stopped at nomination points without invalidating gates.

## 5. Promotion ladder

1. **Nomination:** fixed iterations, 512-game first looks, shared diagnostic
   seed block; freeze the candidate before any confirmation seeds open.
2. **Training verdict:** candidate vs `current_best`, both at 4,800-sim open
   loop, sequential paired gate, fresh seeds, LCB > 50% plus fixed-suite
   non-regression.
3. **Deployment verdict (the one that matters):** candidate vs `current_best`
   on the **laptop advisor at equal wall-clock** — each net at whatever sims
   it actually achieves in the same time budget, paired decks/seats,
   LCB > 50%. This is the direct test of "stronger priors compensate for
   fewer sims." Record the equal-sims result too, for information only.
4. **Salvage path (predeclared):** if the training verdict passes but the
   deployment verdict fails on throughput, one distillation experiment is
   authorized — big teacher into 80x6/96x6 student on the big net's data and
   targets — gated identically. Not a license to iterate; one shot.
   Expectation setting: distilling from a stronger teacher reliably beats
   training the same small net without one (better data, better targets), but
   the capacity-dependent share of the teacher's gain cannot compress back
   into an 80x6 student by definition — the student recovers the data-quality
   share and loses the representation share. If the P2 winner was the
   80x6+pooling arm, this whole path is moot: that candidate already deploys
   at full laptop throughput.

## 6. Stop conditions and closure semantics

- P2 fails → no build, no rental; capacity closed at 3-4x; ship.
- P1 fails → reanalysis dropped; everything else proceeds.
- Feasibility probe fails → fix or abandon before the rental continues.
- Final gates fail → ship `current_best`. The bundle closes **as a unit**; no
  post-hoc rescue of individual components (existing combination rule).
- Any pass → promote per the ladder and update the levers doc.

This plan supersedes the "recommended next measurement" section of
`KINGDOMINO_NEXT_LEVERS.md` (contextual open loop closed by decision) and
amends its deep-target closure (Stage-3 correction). Update both records when
verdicts land.

## 7. Resolved defaults and unneeded downstream choices

- Test A used the signed-off 1.5% relative policy-CE floor, block size 400,
  two seeds per selected LR, and the {1x, 0.5x} LR grid.
- Margin-bin width and range for the distributional head; aux lambdas.
- Sidecar filter values: `abs(root Q) <= 0.4`, adequate-visit threshold per
  tile group, Q-gap 0.03, and the outside-filter control-sample rate.
- Per-iteration reanalysis cap.
- Restart mixture (85/10/5 by example mass) and whether sidecar restarts
  anneal.
- P3 deployment floor: at most 3x slowdown versus 80x6, reported at the normal
  15-second budget and the difficult-position 60-second budget.
- Warm-start sanity-match floor (40% paired score default, 256-512 games).
- Rental duration versus games target (40-60k default).

The remaining architecture/data/restart defaults above were never activated:
P2 failed before the infrastructure build or rental.

## 8. Execution results (2026-08-14)

### P1 — failed; selective reanalysis dropped

The untouched 460-root confirmation split was run through the frozen staged
procedure. Stage 2 selected 84 roots; Stage 3 selected 11 roots from 7 games.
The primary matched-teacher cross-seed uplift was +0.00041 Q
decision-weighted and -0.00006 Q game-weighted, with a game-clustered 95%
interval of **[-0.00165, +0.00127]**. Its lower bound is not above zero, so P1
fails mechanically. See `DEEP_TARGET_STAGE3_FINDINGS.md`.

### Test B — meaningful joint-action absorption gap

On 172 existing Stage-1/Stage-2 positions from 24 games:

| Diagnostic | Prior vs 4,800 | 4,800 cross-seed | Gap | Game-clustered 95% gap interval |
|---|---:|---:|---:|---:|
| Joint action | 45.35% | 77.91% | 32.56 pp | [22.12, 40.14] pp |
| Tile group | 84.59% | 92.44% | 7.85 pp | [3.63, 14.89] pp |

The incumbent has largely absorbed tile selection but not the full
placement-plus-claim distribution. This is diagnostic only and does not
override either hard gate.

### P2 / Test A — failed; no large arm passed

The frozen run10 buffer contained 200,000 examples (SHA-256
`1ea7a6dd0c48caf2aeca5604d1e2cded762639c7dabf8dc6ea69acd832c7051b`).
The fixed split used 180,000 train examples and 50 held-out contiguous
400-example blocks (20,000 examples). Each completed seed received 20,400
steps × 256 examples = 5,222,400 presentations.

| Arm | Selected LR | Seed-mean policy CE | Relative CE improvement | Paired-block 95% interval, control minus arm | Seed-mean Brier | Gate |
|---|---:|---:|---:|---:|---:|---|
| 80x6 | 1.0e-3 | 1.50631 | control | — | 0.16195 | control |
| 80x6 + pooling | 5.0e-4 | 1.50725 | -0.06% | [-0.00561, +0.00368] | 0.17158 | fail |
| 128x8 + pooling | 5.0e-4 | 1.63847 | -8.77% | [-0.13959, -0.12502] | 0.19247 | fail |
| 128x10 + pooling | 5.0e-4 | 1.55593 | -3.29% | [-0.05518, -0.04451] | 0.18540 | fail |

Negative relative improvement means worse than the control. Every candidate
failed the policy gate and also had a Brier regression whose paired interval
was entirely below zero. At the owner's request, after the first 128x10 1x-LR
seed completed at 1.71158 CE, the second 1x seed was skipped; the 128x10 gate
uses the complete two-seed 0.5x pair, matching 128x8's selected LR. The lone
1x result is retained but mechanically excluded from selection and inference.

### P3 — laptop floor passed; 5090 measurement cancelled by P2

The real live-advisor Rust harness measured 12 frozen hard roots, two repeats,
1,600 simulations per search, and `leaf_batch=8` on the RTX 3070 laptop:

| Arm | Simulations/s | Slowdown vs 80x6 | Projected sims at 15 s | Projected sims at 60 s | 3x floor |
|---|---:|---:|---:|---:|---|
| 80x6 | 2,294.8 | 1.00x | 34,422 | 137,688 | pass |
| 80x6 + pooling | 1,858.5 | 1.23x | 27,878 | 111,510 | pass |
| 128x8 + pooling | 1,769.4 | 1.30x | 26,541 | 106,166 | pass |
| 128x10 + pooling | 1,531.0 | 1.50x | 22,965 | 91,859 | pass |

Advisor throughput was not the blocker. The 5090 self-play measurement was
pre-rental and conditional; P2's hard failure cancels the rental, so it is not
run.

### Final decision and artifacts

Per the predeclared closure rule: **do not build the downstream data system,
do not rent the 5090, and ship `current_best`.** Capacity at this 3–4x ladder
is closed on the current data.

- P1 summary: `runs/kingdomino/placement_audit/deep_target_stage3_summary_confirmation_s30000_r1.json`
  (`37d41a9ea14d39da17f4d24ff8169d733923176fd2c06facdaec72740fe6df59`)
- Test A summary: `runs/kingdomino/final_capacity_test_a/test_a_summary.json`
  (`6240b1c49b792a5fc9032af92c37419020732d6ff1a057c46a4c48a8a4deb6fa`)
- Test B: `runs/kingdomino/placement_audit/absorption_gap_v1.json`
  (`05554359c76bf6718f3432edc3f628aa96a691c66a81407e7f6756109b71b14d`)
- P3 laptop: `runs/kingdomino/final_capacity_p3_laptop.json`
  (`19e16dd529cc426c92dcd107a7d483bea22f9cb756009c19fd55017f460be70e`)
