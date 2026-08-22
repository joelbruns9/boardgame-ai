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
The pytest suite is **green: 238 passed, 0 failed** (2026-08-21). Two defects found by external review are fixed with regressions — see §M0.

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

**Status: DONE, 238 passed.** Two defects were found afterwards by external review
and fixed on 2026-08-21, both of which the suite had missed:

- **`redeterminize()` returned the same shuffle every call.** `copy()` clones the
  RNG state exactly, so the optional `rng` argument defaulted to an identical
  generator. `rng` is now required, matching Kingdomino's signature, and the copy
  also gets a fresh forward generator so mid-rollout reshuffles do not correlate.
- **`after_reshuffle_composition` was three cards short.** `_begin_turn` runs
  `_discard_step()` *before* `_reshuffle_decks()`, so the turn's aside cards are in
  the pool and the number cards beside them are not. Added `aside_composition`.
  **The old test passed because it called `_reform_deck()` directly, reproducing the
  same wrong ordering** — a reminder that a test written against the implementation's
  model confirms the model rather than the behaviour.

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
  2 points of greedy's **51.4** (advanced, 2 seats — 47.8 was the one-seat figure,
  and one-seat play is out of scope) on a paired seed set;
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

* **Trunk: shared per-sheet encoder, then one MLP** (revised 2026-08-21; the
  earlier "MLP over a flat concat" is superseded by the symmetric encoder in
  `ENCODER_V2_SPEC.md`).

  ```
  for each seat s:  [planes_s (17,3,12), scalars_s (127)] --> SHARED MLP --> h_s (128)
  trunk_input = [h_me, h_opp0, h_opp1, h_opp2, viewer_plane, global_scalars (380)]
                                    --> main MLP trunk
  ```

  Every seat runs through the **same weights**, which is the Kingdomino precedent
  (`my_board` and `opp_board` share one ResNet trunk). Concatenate the opponents
  rather than pooling them, so identity survives — *which* opponent finishes plan
  2 first is exactly the question. MLP rather than convolution for the sheet
  encoder: the 3×12 grid has no symmetry group, so the sharing that pays here is
  across seats, not across positions.

  Kingdomino's 13×13 ResNet remains the wrong shape. Start around 4M params and
  use the capacity ladder from the Kingdomino work rather than assuming depth.

  **Cost.** The sheet encoder is ~5% of the network, so running it 2–4× instead of
  once costs roughly **4% of network compute** — not the 2.5–4× an earlier estimate
  claimed by false analogy to KD, where the per-board trunk *is* the dominant cost.
  CPU-side encoding is the larger relative cost and is well under 4×, because the
  deck prefix sums are computed once per state and reused across all four sheets.
  Measure it against the current 10.1 games/s.
* **Heads.** Policy (**684 actions**, masked; vocabulary frozen in
  `ENCODER_V2_SPEC.md` §10.6). Then **one per-seat head evaluated four times** (11
  units) and one global head (5 units) — 16 output units producing 49 predictions.
  `margin` is **derived**, not a head, exactly as in Kingdomino.
* **The per-seat head is contextual**, reading `concat(h_s, h_main)` rather than an
  isolated `h_s`. Several of its targets — `plan_k_first`, `final_score`,
  `end_trigger` — are definitionally cross-seat, so a head that sees only one
  sheet's encoding cannot predict them. Weight sharing across seats is preserved.
* **The value is a rank *distribution*, not a rank scalar** (decided 2026-08-21).
  The global head emits a 4-wide softmax over finishing positions, masked to the
  seat count, and the search applies a utility `u_r = (n−1−r)/(n−1)` to it. KD's
  `win_value ** 4` certainty gate is **only valid at two players** — with a binary
  outcome the mean is a sufficient statistic for the distribution, and with three
  or more it is not, so a certain second place and a coin flip between first and
  last produce the identical gate. The correct generalisation is the variance of
  the utility, floor-corrected for seat count:
  `confidence = max(0, (1 − 4·Var[u] − floor_n)/(1 − floor_n)) ** k` with
  `floor_n = 1 − (n+1)/(3(n−1))`. At two seats `floor = 0` and it reduces *exactly*
  to `win_value ** 2`, so `k = 2` reproduces KD's curve rather than departing from
  it. **`k = 1` is the starting default** — measured 2p confidence runs 0.10 → 0.54
  over a game, and `k = 2` would leave margin contributing ~2% of leaf value at
  turn 12. Derivation, the gradient-dead failure case, the seat-count floor and the
  measurement are in `AUX_TARGETS_SPEC.md` §5.1–5.2.
* Full head specification, including the six defects in the current
  `training.TARGET_NAMES` set and the masking rules, is in
  **`AUX_TARGETS_SPEC.md`**.
* **Loss weights.** Score dominant early, policy next, auxiliaries small.
  **Normalise score by 80** (revised 2026-08-21, was ~60). The divisor is set
  from real BGA games at the target strength, not from GreedyBot: observed
  losing scores run 46–90 and winning scores 65–115, centring near 75–80.
  Dividing by 80 puts typical play at ~0.94 and the observed range at 0.58–1.44,
  with headroom above 1.0 for a net that outplays the reference. The head has no
  final activation, so unbounded is intended. 120 would compress everything into
  0.38–0.96 and weaken the score loss against policy and auxiliaries; 60 would
  push strong games to 1.9 and saturate the `tanh` in the margin blend.
  **Set `MARGIN_GAIN` only after the divisor**: at 80, margins of 5–40 points
  normalise to 0.06–0.5, which `MARGIN_GAIN = 2.0` maps to `tanh(0.12 … 1.0)` —
  responsive across the range without saturating. Expect early self-play targets
  near 0.6 rising through training; the divisor is a unit, not a normalisation of
  the current policy, so do not retune it as the net improves.
* **One policy head or twelve?** All phases share one head with disjoint legal
  sets. One masked head is simpler and standard — measure before splitting it.
* **Plan one-hot is 28 wide, not 37** (decided 2026-08-21). `PLANS` holds 37
  entries for BGA id fidelity, but only 28 are ever dealt — stack 1 gets 11
  (basic 0–5 + advanced 18–22), stack 2 gets 11 (basic 6–11 + advanced 23–27),
  stack 3 gets 6 (basic 12–17). Ids 28–36 are the seasonal boards and
  `available_plan_ids` never deals them, so a 37-wide one-hot carries **9
  permanently dead input slots** — the same dead-input problem that settled
  `MAX_OPPONENTS`. Use a dense `plan_id → index` map over the dealt set.
  **Size it at the advanced superset (28) regardless of variant**, so a
  base-rules game is a strict subset of the advanced input space — ten slots stay
  dark, roundabout actions never enter the legal mask — and one weight set serves
  both. That is what lets the advisor read base-rules tables and give sensible
  (if untuned) advice without a second model. Full block layout in
  `ENCODER_V2_SPEC.md`.

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
| Bootstrap teaches a plan-blind policy (greedy completes **0.42** plans/game) | accept in S0; if S1 does not fix it, add a plan-seeking bot to the bootstrap mix |
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
