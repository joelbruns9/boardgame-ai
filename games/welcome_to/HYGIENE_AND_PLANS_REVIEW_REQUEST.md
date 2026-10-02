# Welcome To… — review request and ideation: deferred value (sheet hygiene and City Plans)

**Date:** 2026-10-02. **Branch:** `main` at `65632c3` (Welcome To merged from
`welcome-to-engine`). **Asks for:** a review of three things and a brainstorm.

1. **Encoder inputs** (§8.1): are the right signals present, in a form a 4M-parameter
   MLP can use, for sheet hygiene and City Plans?
2. **Temperature and noise settings** (§8.2): is self-play exploring the decisions that
   matter, and do the settings starve the value head of contrast?
3. **Deferred credit assignment** (§8.3): how can the network learn the value of board
   hygiene and City Plans when their payoff arrives 10–20 turns after the decision that
   causes it?
4. **Ideation** (§9): which of the candidate fixes are worth building, in what order,
   and what are we missing?

Everything in §6 is **measured**, with the script or artefact named. Please treat §6 as
settled unless you think a measurement is wrong — in which case say which and why.
Single seed per training run throughout; §7 lists that and the other caveats.

---

## 1. The game, in the parts that matter here

**Welcome To…** is a roll-and-write for 2–4 players (we train 2/3/4 seats at
60/30/10; base board, *advanced* rules, non-expert). Each player has a private sheet of
three streets with **10, 11 and 12 houses (33 boxes)**.

* **Every turn** three *combinations* are revealed — a house **number** (1–15) paired
  with an **effect** — and they are the **same for every player**. Each player picks
  one combination and writes the number into an empty box. Turns are simultaneous.
* **The ascending rule.** Within a street, numbers must strictly increase left to right
  (gaps allowed). A number can only go where it fits between its neighbours. Writing 15
  in a street's first box kills every box to its right; writing 8 in the middle of an
  empty street splits it into two narrow windows. This is the root of "hygiene".
* **Effects:** *surveyor* (place a fence → creates housing estates), *real estate*
  (raise an estate size's value), *park* (mark the street's park track), *pool* (only
  on the three pool boxes of a street — mark a pool), *temp agency* (write the number
  ±1/±2), *bis* (duplicate an adjacent number, filling a box no draw could fill).
  Advanced rules add **roundabouts** (built **in addition to** the turn's number and
  effect, even on a refusal turn; it resets the ascending constraint and costs points —
  ⚠ corrected after review, the first draft said "instead of a number"; the engine was
  always right).
* **Permit refusal.** If none of the three numbers can be written anywhere, the player
  must take a **refusal**. **The third refusal by any player ends the game for
  everyone.** (A voluntary refusal also exists, only when the printed number has nowhere
  to go but the temp agency could place it; it is ~2% of refusals in our games.)
* **City Plans.** Three shared objectives (one from each of three stacks), e.g. "two
  estates of size 4", "all parks and pools of street 2", "fill both ends of every
  street", "five bis in one street", "seven temp marks". The **first** player(s) to
  complete a plan score its high value; later finishers score less. Completing a plan
  **consumes** the houses it used (they cannot count for another plan).
* **Game end:** a player has completed all three plans, or filled the sheet, or taken a
  third refusal. In our games it is **always** the third refusal.
* **Scoring** sums parks, pools, estates (by size and real-estate level), plans, temp
  agency rank, bis (negative), refusals (negative), roundabouts (negative).

**Why this is hard to learn.** A placement on turn 4 changes the score on turn 4 by
**zero**. Its cost appears ~turn 16–24 as refusals (and the game ending early) and as
plans never completed; its benefit appears as plans completed many turns later. Card
draws intervene every turn. Plans are worth a lot (plan values 6–13 first / 3–7 later,
≈13 points of a ~75-point strong game) but only when completed — partial progress is
worth nothing. Finishing all 3 city plans also ends the game which denies opponents the
opportunity to score more. In high level human play, 99% of games end via 3 completed
plans.

## 2. The goal

A strong player for BGA-style tables (2–3 players, advanced), trained by self-play from
**random weights** (behaviour cloning of GreedyBot was tried and dropped). Two standing
project rules shape what we will accept:

* **No GreedyBot / hand-written heuristics as training signal.** Diagnostics may use a
  heuristic; training targets may not encode "goodness".
* **Targets are outcomes, not judgements.** Auxiliary heads predict things that happen
  (refusals, plan completion, final components), never "this move is good". Aux heads
  never enter the leaf value (`AUX_TARGETS_SPEC.md` §1).

The strength yardsticks are plans completed per seat, learner score and margin, and the
paired promotion gate.

## 3. The system (as of `65632c3`)

| part | what it is | where |
|---|---|---|
| engine | Python + Rust (bit-equivalent, gated), BGA-PHP-derived rules | `game.py`, `welcome_to_rust/` |
| actions | 684-wide **macro** vocabulary: `WRITE(slot, temp delta, box)` collapses card choice + placement into one decision; refusals, roundabout, effect decisions separate | `macro_codec.py` |
| encoder | v3, ABI 3: **22 planes** per seat (4 seats × 3 streets × 12), **194 per-seat scalars**, **367 global scalars**; viewer-relative, symmetric shared sheet encoder | `ENCODER_V3_SPEC.md`, `encoder.py` |
| network | shared sheet MLP per seat → concat → 768-wide trunk (2 residual blocks) → policy (684), contextual per-seat heads (29 outputs), global head (rank 4 + turns left). **4.17M parameters** | `network.py` |
| search | Rust open-loop MCTS with determinization (undrawn deck reshuffled per simulation), chance progressive widening (C=1, α=0.5, ≤4 particles), c_puct 1.5, **200 simulations** | `rust_search.py`, `SEARCH_SPEC.md` |
| leaf value | `(1−α)·rank_value + α·confidence·tanh(margin·2)`, α=0.5; rank from a 4-way masked softmax over finishing positions; margin from per-seat score heads (÷80) | `mcts.blend_value` |
| self-play | learner = seat 0 only, searches every turn; **opponents never search** — they sample the policy of league checkpoints | `self_play.generate` |
| training | 500 games / iteration, 400 steps × 256, AdamW lr 3e-4, replay window grows to ~12 iterations (≈6k games, ≈200k positions); policy target = visit distribution | `s2_train.py`, `s2_run.py` |
| gate | 300 paired deals, T=0, no root noise, every 5 iterations; promote on margin CI > 0 | `s2_promotion.py` |
| aux targets | final score + 8 components, permits, houses, capacity left, turns left, plans completed, turns-to-plan, will-complete-plan ×3, plan-first ×3, end-trigger ×3, **forced refusals in t..t+3**, **plan k dies in t..t+3** (masked) | `AUX_TARGETS_SPEC.md`, `training.py` |
| curriculum | 20% of games restart 1/2/4/8 turns before a plan the learner completed last iteration (reshuffled deck) | `curriculum.py` |

### 3.1 Temperature and noise (current defaults)

| setting | value | applies to |
|---|---|---|
| move selection, turns ≤ 10 | sample visits at **T = 1** | learner |
| move selection, turns > 10 | **argmax visits (T = 0)** | learner |
| root Dirichlet | concentration 10 (α = 10 / #legal), weight 0.25 | learner, every searched root |
| noise-fresh fraction | ≥ 25% of a root's budget is fresh simulations under noise (tree reuse keeps the rest) | learner |
| opponents | sample raw policy, **T = 1 for turns ≤ 10, argmax after** | all non-learner seats |
| playout-cap randomization | available, **off** in `s2_run` | — |
| gate / diagnostics | T = 0, no noise | — |

## 4. The problem in one paragraph

The learner keeps a messy sheet (many boxes that no future number can fill), so it
starts taking forced refusals around turn 18, and some seat's third refusal ends every
game around turn 23–25 — GreedyBot's games last 30.6 turns. Short games and messy
sheets mean City Plans are rarely completed: park/pool plans essentially never, estate
plans ~10–15% of the time. So the training data almost never shows the payoff of either
a clean sheet or a completed plan. **The network can see hygiene** (§5) **but does not
value it**: its value head is indifferent between a clean and a messy placement of the
same card, and more search, more training steps and 4× more games have not changed that.

## 5. What the encoder already gives the network about hygiene and plans

Per seat (all seats, viewer-relative):

* **Planes:** `box_spans` (how many numbers could still go in each empty box, ÷18),
  `span_if_roundabout`, `writable_no_temp`, `writable_temp_only`, delta-0 `positional_fit`,
  P(fit | deck) with and without temp, after a reshuffle, after an optimal roundabout,
  `p_fit_next_turn`, plus three "plan slot k still needs a house here" planes.
* **Scalars:** per-street `capacity` (4), `total_span`, `roundabout_repair` (3), a
  5-float **refusal block** (P(no slot playable), the same after the best roundabout,
  P(printed number unplaceable), roundabout rescue available, steady-state forced-refusal
  probability), `houses_this_turn`, `free_boxes`; per plan slot 34 floats (progress,
  steps left, turns lower bound, expected turns, demand by street/effect, feasibility),
  `plan_conflict_seat` (overlap and directed kills between the seat's plans).
* **Global:** stacks, deck composition, effect supply rates, turns to reform, plan
  identity (28-way), next effects.

GreedyBot's whole placement policy is a function of three of these (capacity, span, an
estate term); its ablation shows capacity alone is worth 33.9 points and +span 42.6
(`ENCODER_V3_SPEC.md` §11). The 2026-08-30 encoder audit and the v3 review validated the
features against the engine (20,876 encodings Python↔Rust, zero divergences).

## 6. Measurements so far

### 6.1 The plateau (v3_random_01, 8 iterations, 2026-09-26)

Learns from random weights: plans/seat 0.025 → 0.22, score 12.7 → 27.1. Then flat.

| | learner (iter 7–8) | GreedyBot |
|---|---|---|
| capacity per empty box, turn 16 | 0.51 | 1.00 |
| dead boxes, turn 16 | 5.3 | 0.1 |
| game length (turns) | ~24 | 30.6 |
| how games end | 3rd forced refusal, every game | 3rd refusal |
| park/pool/bis plans completed | 0 / 113 | — |
| estate plans completed | 57 / 418 | — |

Plan heads sat exactly at base rate (accuracy = 1 − positive rate). Refused: "the
learner ends games deliberately while ahead" (refusals ~97% forced; voluntary refusers
usually behind) and "weak opponents end games" (learner vs itself: 23 → 24 turns).

### 6.2 Reviewer round 1 tests (frozen iter-8 checkpoint, 300 paired deals, T = 0)

| test | result | reading |
|---|---|---|
| ungated margin (confidence power 0) vs production blend | +0.4 [−1.3, +2.2] | the confidence gate is not hiding the score channel |
| 800 vs 200 simulations | +5.4 [+3.5, +7.2] margin, plans 0.68 vs 0.56 | search helps… |
| policy only vs 200 | −11.6 | …a lot |
| hygiene at policy / 200 / 800 | **identical** (t16 cap/empty 0.50–0.53, ~5.7 dead) | **search optimises around a blind spot; the evaluator does not see placement cost** |
| 1,600 vs 400 updates on the same replay | held-out rank CE 0.75 → **1.00** (worse) | not under-trained; **data-limited** |
| exploration temperature | T=0 hygiene = training hygiene | temperature is not the cause of the mess |

### 6.3 Placement rollouts (reviewer test 2, 2026-09-26)

240 positions (turns 6–16) where the net's placement and the same-card
capacity-maximising placement disagree; 16 shared redeterminized continuations each,
argmax policy for all seats.

| clean minus net's choice | value |
|---|---|
| margin | **+1.02 ± 0.29** |
| refusals | **−0.145 ± 0.021** (7 s.e.) |
| plans | +0.005 (none) |
| value head's preference vs actual outcome difference | **correlation +0.03** |
| value head preferred the better move | 95 / 240 |

A real, learnable signal the evaluator misses. One better placement does not create
plans by itself.

### 6.4 Short-horizon targets + curriculum (v3_curriculum_01, iterations 1–8, 2026-09-30)

New: `forced_refusals_soon` (forced refusals in t..t+3, ÷3), `plan_k_dies_soon` (plan
becomes provably infeasible in t..t+3, masked once settled), and the near-completion
curriculum. Same seed and deals as v3_random_01.

| iter 7–8, ordinary games | v3_random_01 | v3_curriculum_01 |
|---|---|---|
| t16 capacity per empty box | 0.516 | **0.542** (+3.4 s.e.) |
| t16 dead boxes | 5.28 | 4.97 |
| forced refusals | 1.32 | 1.23 |
| plans, game length, plan deaths | — | unchanged |

Heads: `forced_refusals_soon` held-out R² **0.51**; `capacity_left` R² 0.09 → 0.21;
score R² 0.42 → 0.49; rank CE 0.75 → 0.55. ⚠ **Corrected after review (2026-10-02):** I wrote
that `plan_k_dies_soon` "stayed at base rate", judging by accuracy ≈ 1 − positive rate. That
was the wrong yardstick. Brier skill against the constant-rate predictor is **+1%, +24%, +2%**
for slots 0/1/2 at iteration 8 and **+18%, +28%, −13%** at iteration 35: slots 0–1 carry real
signal, slot 2 does not. Whether that signal helps sibling choices is untested.
Curriculum restarts re-finished the source plan ~50% of the time; ordinary-game plans
did not rise.

Plan deaths (v3_random_01 iter 7–8): park/pool plans die in **70%** of games at median
turn **8** (a pool box written without a pool); complete-street plans 95%, median turn
13; estate plans stay feasible to the end in 80% of games and simply are not built.

### 6.5 Hygiene rescue (reviewer test 1, 2026-10-01; v3_curriculum_01 iter-8 checkpoint)

300 paired deals, T = 0. Assistance: replace a write by the same-card write that leaves
the most placement capacity (then span), through turn 16, then normal play.
`hygiene_rescue.py`, results `runs/welcome_to_s2/v3_curriculum_01/_hygiene_rescue/`.

| | normal | focal (learner assisted) | all seats assisted |
|---|---|---|---|
| learner plans / game | 0.50 | 0.64 (**+0.14** [+0.06, +0.22]) | 0.78 (**+0.28** [+0.20, +0.37]) |
| plans / seat, all seats | 0.33 | 0.38 | 0.59 |
| game end turn | 25.8 | 25.8 (0) | 28.8 (**+3.0** [+2.6, +3.3]) |
| learner score | 34.0 | 39.5 | 44.0 |
| learner margin | 13.7 | 18.1 (+4.3) | 10.8 (opponents improved too) |
| learner forced refusals | 1.47 | 1.04 | 2.02 (longer games) |
| learner plan deaths | 0.64 | 0.64 | 0.87 |
| placements changed by the rule | — | 95% (4,552 / 4,791) | 95% |

**Hygiene is causal for plans**: half via the learner's own sheet, half via game length
(opponents' third refusals end games). Even perfect hygiene to turn 16 leaves games
ending on refusals (~turn 29) and 0.7% plan endings — plan pursuit is a separate limit.

### 6.6 Placement-assist scaffold (v3_assist_01, 2026-10-01) — failed

In 50% → 0% (by iteration 6) of games the learner's placements through turn 16 were
replaced by the rule; **policy targets stayed the search's visits**, only the outcome
changed. Iterations 6–8 (after removal) vs v3_curriculum_01:

| | curriculum | assist |
|---|---|---|
| t16 capacity per empty box | 0.537 | **0.502** (−0.035, ~5 s.e.) |
| t16 dead boxes | 4.99 | 5.58 |
| learner score | 26.8 | 25.3 |
| plans | 0.35 | 0.33 (n.s.) |

### 6.7 Value-preference check (sibling afterstates)

Positions where the cleanup rule disagrees with the move played (turns 4–16, capacity
gain ≥ 1). Value of clean afterstate minus played afterstate, gate blend, [−1, 1] scale.

| checkpoint | positions from curriculum iter-8 games | from assist iter-8 games |
|---|---|---|
| curriculum 8 | +0.000, prefers clean 51% | +0.009, 52% |
| assist 5 (end of assistance) | +0.004, 54% | −0.002, 47% |
| assist 8 | +0.009, 52% | −0.002, 48% |

All checkpoints within ±0.011, 43–60%. **The sign flips with the position source** — a
net looks anti-clean on its own games (positions qualify only where it already chose
messy) and mildly pro-clean on the other run's. Scored fairly: indifferent.

### 6.8 Scale: v3_curriculum_01 extended to 35 iterations (2026-10-02, ~15k games)

| | iter 8 | iter 20 | iter 35 | rescue rule |
|---|---|---|---|---|
| plans / seat (all games) | 0.21 | 0.33 | 0.34 | |
| learner score | 28.3 | 36.0 | 37.4 | |
| game end turn | 23.1 | 25.2 | 25.4 | |
| learner plans / game | 0.38 | 0.54 | 0.51 | |
| **t16 capacity per empty box** | **0.539** | **0.561** | **0.575** | 1.00 |
| t16 dead boxes | 4.9 | 4.9 | 4.7 | 0 |
| learner plan deaths | 0.55 | 0.67 | 0.71 | |
| value prefers clean (own / other games) | 51% / 49% | 54% / 54% | 56% / 52% | |

Strength scaled (promotions at 15, 25, 35; flattening after ~25). Hygiene and the
value head's sibling preference barely moved.

## 7. Caveats

* **One seed per training run.** Differences under ~2 s.e. between runs are noise.
* The cleanup rule is a **diagnostic instrument**, never a training signal; it is also
  not optimal (it ignores plans, pools and estates — it protects capacity only).
* `plans.feasible` is **sound, not complete**: it never reports a death that is not
  real, and misses some. Plan-death numbers are lower bounds.
* Value-preference positions are selected where *some* net chose the messy move; see
  §6.7 for the bias and why both sources are reported.
* Gate games are T = 0 with seat 0 searching and other seats on argmax policy; training
  games are not. Strength metrics in §6.4–§6.8 exclude curriculum and assisted games.

## 8. What we would like reviewed

### 8.1 Encoder inputs

* §5 lists what the network sees. **Is anything missing** that a strong human uses to
  judge a placement — e.g. the *expected number of future refusals* of a sheet given
  the deck, the *number of boxes made unfillable by this placement* (a delta, not a
  level), per-street "window" structure, or a direct cost-of-placement feature?
  We have avoided hand-computed valuations as inputs (project rule), but a *consequence*
  feature (e.g. "boxes that become dead if you write here") is borderline — where would
  you draw that line?
* **Representation vs valuation.** The levels are present (capacity, spans, refusal
  probabilities), yet sibling afterstates — which differ by a few boxes' spans — get
  near-identical values. Is a flat MLP over ~1k inputs per seat (792 plane floats + 194 scalars) a poor fit for picking up
  *small differences* between siblings? Would an **afterstate/delta encoding** (encode
  the change a candidate placement makes) or a per-box convolution over the 3×12 grid
  help, despite the ascending rule breaking translation symmetry?
* Is the input scaling (all spans ÷18, probabilities in [0,1], ~1k floats per seat with
  many near-zero) likely to bury the hygiene signal?

### 8.2 Temperature and noise

* After turn 10 the learner plays **argmax visits** and opponents play **argmax
  policy**. Placement mistakes are made mid-game. Does T = 0 after turn 10 remove exactly
  the variation (alternative placements of the same card) the value head would need to
  learn hygiene? Would a placement-only temperature or epsilon (perturb the box, keep the
  card) be a cleaner exploration mechanism?
* Root Dirichlet over a 684-wide vocabulary with concentration 10 → α ≈ 0.1–0.5 per
  legal macro; weight 0.25. Placement alternatives of one card can number 10–30
  macros. Does this noise spread mass usefully across placements, or mostly across
  cards?
* **Opponents never search** in generation. They are the reason games end early (§6.5,
  focal vs all). Is policy-only opponent play a mistake for this game, or is the
  cheaper fix to give opponents a little search?
* 200 simulations over a high-branching, chance-heavy tree (determinization +
  progressive widening): is the visit distribution at placement decisions informative
  enough to be a policy target, or close to the prior? Would Gumbel root selection
  (sequential halving) give a better policy-improvement signal at this budget?

### 8.3 Deferred value: hygiene and City Plans

The central question. The facts: hygiene is causal (§6.5); the value head is blind to it
between siblings (§6.3, §6.7); search does not compensate (§6.2); more updates hurt and
more games barely help (§6.2, §6.8); clean-state data via a scaffold did not help (§6.6);
short-horizon refusal targets helped the trunk a little (§6.4).

* **Why is the sibling contrast not learned?** Our working hypothesis: on-policy data
  contains almost no clean placements (the rule changed 95% of them), so the value head
  never sees two near-identical states with different outcomes; and the final-outcome
  label is ~17 turns of chance away from the decision. Do you agree? What else could
  explain it?
* **City Plans.** Park/pool plans die by turn 8 (a pool box written without a pool);
  estates are left unbuilt even when feasible. The plan value is entirely terminal and
  conditional on finishing. What target or training structure would make the trunk
  value *progress toward* a plan without us hand-writing a progress bonus? Is a
  short-horizon "plan completed within k turns" target worth it given positives are rare
  (§9)? Is the curriculum (§3) the right tool, and why did it not move ordinary-game
  plans?
* **Value target design.** Final-outcome targets (rank distribution + score heads) with
  a margin/rank blend. Would n-step / TD(λ) value targets from search values, or a
  KataGo-style short-horizon *value* (not aux) target, be appropriate here? The search
  is itself blind (§6.2), so bootstrapping from it may launder the blindness.

## 9. Ideation — candidate fixes

Ranked by our current belief; please re-rank, kill, or add.

| # | idea | targets | status / cost | concerns |
|---|---|---|---|---|
| A | **Rollout-judged placement targets**: at sampled learner placement decisions, branch on every legal box of the chosen card, play each a few turns with shared draws, train value (and optionally policy) toward the outcome differences | sibling contrast, directly | not built; cost = rollouts per sampled decision | rollout policy quality; horizon length; bias toward short-term |
| B | **Placement-level exploration** in generation: temperature or epsilon over boxes of the chosen card, mid-game | contrast in on-policy data | cheap | dilutes play quality; noisy labels |
| C | **Opponents search** (or assisted opponents) in generation | game length (half the rescue gain) | moderate (eval cost) | learner sees a different opponent distribution than gates |
| D | Gumbel root selection / sequential halving | policy improvement at low budget | moderate | does not fix a blind value |
| E | Afterstate / delta inputs, or grid convolution | representation of small differences | encoder ABI break | §8.1 |
| F | More short-horizon outcome targets: refusals in t..t+6/10, houses written in t..t+k, boxes dead at t+k | trunk features | cheap | §6.4 says these move the trunk, not the value |
| G | n-step / TD value targets; reanalysis of replay with the current net | label variance | moderate | search blindness launders in |
| H | Plan-specific: "plan completed within k turns" target; plan-weighted curriculum toward park/pool plans | plan value | cheap / exists | positives rare (3.8% of live rows) |
| I | Scale: bigger replay, more games, larger net | everything | compute | §6.8: strength scales, hygiene does not |
| J | Phased-out cleanup scaffold (built, §6.6) | clean-state data | done, **failed** | kept for record |

Our proposed next build is **A**, starting from the iteration-35 checkpoint.
We would particularly value a view on **A vs B** (counterfactual labels vs on-policy
exploration) and on whether **C** should run alongside.

## 10. Reproduce / inspect

| what | where |
|---|---|
| training runs | `runs/welcome_to_s2/v3_random_01`, `v3_curriculum_01` (35 iters), `v3_assist_01` |
| per-iteration metrics | `<run>/progress.jsonl`, `<run>/candidate_iter_NNNN.pt.metrics.json` |
| hygiene rescue | `python -m games.welcome_to.hygiene_rescue --checkpoint <ckpt> --out <dir>` |
| value preference | `runs/welcome_to_s2/v3_assist_01/_value_preference*.json`, `runs/welcome_to_s2/v3_curriculum_01/_value_preference_src*.json` |
| specs | `AUX_TARGETS_SPEC.md`, `ENCODER_V3_SPEC.md`, `SEARCH_SPEC.md`, `SELF_PLAY_PLAN.md` |
| new code since the last review | `training.py` (`ReplayLog`), `curriculum.py`, `hygiene_rescue.py`, `placement_assist.py`, `welcome_to_rust/src/samples.rs` (`replay_history`) |
| tests | `python -m pytest games/welcome_to/tests -n 8` (733 passed, 3 skipped) |

## 11. Explicit questions

1. Do you agree that the binding problem is **sibling-level value contrast**, not
   representation and not search budget? If not, what measurement would separate them?
2. Encoder: what, if anything, should be added or re-expressed (§8.1)?
3. Temperature/noise: should mid-game play be stochastic, and at which level — card,
   placement, or both (§8.2)?
4. Should opponents search in generation (§8.2, idea C)?
5. For deferred plan value, which training structure would you try first (§8.3, ideas
   A/F/G/H)?
6. Re-rank §9, and name anything missing.
