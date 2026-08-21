# Welcome To... — training project plan

**Goal:** the strongest *Welcome To...* player there is.

> **Search and the self-play loop live in `SELF_PLAY_PLAN.md`** (stages S0–S3),
> which replaces the Phase 2–3 milestones that used to sit here. Phase 1, the
> network design, the metric set and the appendix below all still stand.
>
> **Training runs 2–4 seats throughout.** One-seat play is out of scope: it costs
> no less per learner decision searched, and it silently switches scoring rules.

**Primary configuration: the advanced variant.** Roundabouts and the ten extra
City Plans are in from the start — `GameConfig(advanced=True)`. Everything below
assumes it; the base game is a fallback for ablations, not the target.

Why that matters more than it looks: a roundabout is the game's only *capacity
repair* tool. It breaks the ascending chain in both directions, so it can turn a
dead street back into a live one at a cost of 3 or 8 points. On a measured
example, a street holding `15` in box 0 has a placement capacity of 2; adding a
roundabout in box 1 takes it back to 8. That is a genuinely hard tradeoff — six
future placements against three points — and it is exactly the kind of judgement
that separates a strong player from a greedy one. The advanced plan cards
(`FullStreet`, `FiveBis`, `SevenTemp`, `Extremities`, `Decorative`,
`CompleteStreet`) also widen the plan race well beyond "collect estates of size
N", which is what makes the race features worth having.

---

## 0. Where things stand

**Done.** Engine (rules-exact against the BGA PHP), 357-slot action codec,
information-set-safe encoder (18×3×12 planes + 473 flat features), exact deck
knowledge, City Plans with distance-to-completion, baseline bots, auxiliary
training targets, trajectory capture/replay, and the `games.az_loop` adapter seam.
The pytest suite has not been run yet — **that is step 1**.

**Reusable, do not rebuild.** `games/az_loop` is game-agnostic: run controller,
soft gate, checkpoint lifecycle, games ledger, HOF, Elo, SPRT, stagnation
detection, run log. `games/kingdomino` is the worked example of both seams.

**Measured baselines** (GreedyBot; see the appendix for the full table):

| configuration | games/s | turns | score | plans | roundabouts |
|---|---|---|---|---|---|
| advanced, 2 seats | 4.7 | 30.8 | 51.4 | 0.42 | 1.22 |
| advanced, 4 seats | 2.2 | 30.5 | 50.9 | 0.41 | 1.00 |

---

## Phase 1 — Foundations

### M0. Verify the engine
**Do:** `python -m pytest games/welcome_to/tests -q`, then
`python -m games.welcome_to.random_play --games 200 --players 4 --encode`.

**Gate:** suite green; 200 random games finish with no illegal action, no
runaway, and cards conserved.

**Why first:** nothing below is worth starting on an unverified engine, and the
suite encodes several rules that were got wrong at least once (the reveal
structure, the bis-fence interaction, the roundabout livelock).

---

### M1. Supervised bootstrap — no search
**Do:** capture GreedyBot advanced trajectories at 2–4 seats
(`python -m games.welcome_to.datagen --games 20000`), train the network on
policy = the action greedy chose, value = final score, plus the auxiliary heads.

**Deliverables:** `network.py`, `train.py`, a checkpoint.

**Gate:**
- policy top-1 agreement with greedy ≥ 60% on held-out games;
- the net *playing greedily off its own policy, with no search*, scores within
  2 points of greedy's 47.8 on a paired seed set;
- `permits` and `houses` heads beat predict-the-mean on held-out data.

**Why:** it proves the whole pipeline — encoder → network → loss → checkpoint —
against a known reference in an hour, before MCTS can hide a bug. If the net
cannot clone greedy, nothing downstream will work.

**Known limits of this teacher, to be accepted rather than fixed here:** greedy
completes ~0.5 plans per game, so plan-completion data is thin and the
`turns_to_plan_*` heads will be mostly masked; and greedy has no notion of the
race. The bootstrap produces a placement-competent, race-blind policy. That is
fine — plans are worth 6–14 points, so a score-trained value head will push search
toward them in M3. If M3 shows it does not, add a plan-seeking heuristic bot to
the bootstrap mix rather than trying to fix greedy.

---

### M2. Network
**Deliverable:** `network.py`.

Design decisions, none of them inherited from Kingdomino:

* **Trunk.** 18×3×12 = 648 spatial floats next to 473 flat features. This is an
  MLP or a small 1-D convolution along the street axis. Kingdomino's 13×13 ResNet
  is the wrong shape and mostly wasted parameters. Start at ~1–2M params and use
  the capacity ladder from the Kingdomino work rather than assuming depth.
* **Heads.** Policy (357 logits, masked by `legal_mask()`); value as three parts —
  expected final **score**, **win probability**, score **margin** vs the best
  opponent; plus the ~20 auxiliary heads in `training.TARGET_NAMES`, several of
  them masked.
* **Loss weights.** Score dominant early, policy next, auxiliaries small.
  Normalise score by ~60.
* **One policy head or twelve?** All phases share one head with disjoint legal
  sets. One masked head is simpler and standard — measure before splitting it.

**Gate:** batched inference throughput measured and recorded; ablation harness
working via `encoder.block_slice(name)` so a feature block can be zeroed and the
effect measured.

---

## Phases 2-3 — moved

Search, the self-play loop and the plan races now live in **`SELF_PLAY_PLAN.md`**
(stages S0-S3), which replaces the milestones that used to sit here.

The old M3-M6 assumed a single-seat curriculum on the grounds that one seat is
cheaper. It is not: training cost is paid per *learner decision searched*, and
that is flat across seat counts (100.6 decisions at two seats against 98.4 at
four). A one-seat game also switches scoring rules -- `TEMP_SOLO_SCORE` replaces
the 7/4/1 ranking and every City Plan pays its first-place value -- so it trains
on a different game. Training runs 2-4 seats throughout.

---

### M7. Endgame
Optional, and worth it only if measurement says so. The branching that bites is
chance, not moves. Two tractable forms: exact expectimax over the last two or
three turns, enumerating over *offers* rather than card identities (many cards
produce the same (number, effect) pair, which collapses most of the branching);
or perfect-information Monte Carlo over sampled futures. PIMC assumes you will
know the future, so it overvalues lines needing one specific card and never pays
to keep options open — use it as an evaluator, not a policy.

---

## Cross-cutting

### Metrics to log every iteration
| metric | why |
|---|---|
| mean score, paired-seed delta vs best | the promotion signal |
| mean permits | the direct signature of capacity mismanagement |
| mean houses placed, capacity at turn 10 | the same thing, earlier |
| plans completed per game | the race, and the clearest strength proxy |
| **fraction of games ending on the plans** | greedy is at 0; a strong player should not be |
| roundabouts built per game | is the model using its capacity-repair tool? |
| `identical_games`, `mean_first_divergence_turn` | the self-play collapse canary |
| mean branching, decisions/game | search cost |

### Risks
| risk | mitigation |
|---|---|
| Bootstrap teaches a plan-blind policy (greedy completes ~0.1 plans/game) | accept in M1; if M3 does not fix it, add a plan-seeking bot to the bootstrap mix |
| Phase 2 misprices temps and plans | do not let those heads calibrate in Phase 2; ramp into M5 rather than switching |
| Symmetric self-play collapses | independent per-seat sampling; log `identical_games` |
| Roundabout adds a 34-way branch for a rarely-right action | per-phase search budget; the decline is now sticky, so the branch cannot cycle |
| Encoding throughput gates the GPU (~3.1k samples/s single-threaded) | parallelise trajectory replay; it is embarrassingly parallel |
| Rules drift silently invalidating replay data | already handled: replay re-runs the rules and fails loudly on an illegal action |

### Open decisions
- MLP trunk versus 1-D convolution along the street axis.
- One masked policy head versus per-phase heads.
- Whether depth-1 explicit chance expansion beats pure open loop — settle by
  measurement on positions where a specific number matters.
- Seat-count mixture weights across 2/3/4. Uniform to start.

---

## Appendix — measured baselines

GreedyBot, advanced variant unless stated. Games all end on a third permit
refusal; greedy essentially never completes a plan.

| configuration | games/s | turns | score | permits | plans | roundabouts |
|---|---|---|---|---|---|---|
| advanced 2 seats | 4.7 | 30.8 | 51.4 | 2.40 | 0.42 | 1.22 |
| advanced 3 seats | 3.2 | 30.2 | 50.6 | 2.13 | 0.41 | 1.10 |
| advanced 4 seats | 2.2 | 30.5 | 50.9 | 2.00 | 0.41 | 1.00 |

GreedyBot's three shaping terms, advanced, 150 paired seeds:

| terms | score | gain |
|---|---|---|
| capacity only | 33.9 | — |
| + total span | 42.6 | +8.2 (t = 4.1) |
| + positional fit | 45.8 | +3.2 (t = 2.2) |

A pure rule-based bot using the positional rule *instead of* lookahead runs 14×
faster (181 games/s) but scores 12.5, because it cannot judge fences — and fences
are where the estate points are. The positional rule is worth having as an
evaluation term, not as a replacement for search.

Branching by phase (base game, one seat — advanced adds the roundabout phase):

| phase | share | mean options | max |
|---|---|---|---|
| CHOOSE_CARDS | 37% | 2.3 | 3 |
| WRITE_NUMBER | 34% | 13.1 | 165 |
| ACTION_ESTATE | 8% | 6.2 | 7 |
| ACTION_PARK | 7% | 2.0 | 2 |
| ACTION_SURVEYOR | 7% | 28.5 | 31 |
| ACTION_BIS | 4% | 9.7 | 19 |
| plan / pool / reshuffle | 3% | ~2.1 | 6 |

Other constants worth having to hand: 357 actions; 18×3×12 spatial planes and
473 flat features; ~3.1k samples/s replay-and-encode single-threaded; 5000
trajectories are ~1.6 MB on disk against ~1.6 GB as float32 tensors.

Random play is a **rules fuzzer, not a baseline** — it blocks its own streets out
and ends games four turns early with a mean score of 7.5.
