# Welcome To… — search specification

**Status:** the within-turn tree is **implemented** (`mcts.py`, `macro_codec.py`)
and described here as built. The chance boundary is **proposed, not built** — §6
onward is for review before implementation.

**Revision history.** Two review rounds, both of which blocked the chance
design. Round 1 found: future deck order retained in a scenario (strategy
fusion), an incorrect sampled-support weighting, and a boundary modelled as
"draw three cards". Round 2 found the remaining **P0**: §7 described the retained
transition inconsistently — as a *pre-reveal afterstate* in one place and as
*including opponent results* in another — which cannot both be true and breaks
outright for any root that is not seat 0. All are corrected below, and the wrong
versions are recorded so they are not reintroduced.

**Decisions taken 2026-08-22** (§7.6). The transition semantics of §7.1/§7.3
are agreed; `K` is a schedule rather than a constant; the in-search opponent
model is the network, batched, and never a bot; merging and particles are
measured inert and will not be built in v1; retained children are discarded at
every real turn boundary.

**Reads with:** `SELF_PLAY_PLAN.md` S1 (root-player contract, gate),
`ENCODER_V2_SPEC.md` §10.6 (frozen 684 vocabulary), `AUX_TARGETS_SPEC.md` §5
(leaf value). Research references are collected in §13.

---

## 1. What this document decides

Two things, and they are independent:

1. **Within a turn** — how the root player's own sequence of decisions is
   represented. Settled; §3–§5.
2. **At the turn boundary** — how the card reveal is represented. Open; §6–§10.

They are independent because of the single most useful structural fact about this
game:

> **No chance occurs inside a turn.** Every transition from the start of a
> player's turn to the end of it is deterministic given their actions.

So the within-turn subtree is exactly reproducible across simulations, its nodes
take repeat visits, and its values back up through exact deterministic
transitions. Everything difficult is at the boundary.

---

## 2. The engine's phase graph — ground truth

Transcribed from `game.py`. Twelve phases (`Phase`), mirroring `states.inc.php`.

```
                    ┌───────────────────────────────────────┐
                    │                                       │
                    ▼                                       │
             CHOOSE_CARDS ──ROUNDABOUT_OPEN──▶ ROUNDABOUT_PLACE
                    │                                       │
                    │ choose_stack(s)              (place or pass)
                    ▼                                       │
             WRITE_NUMBER ◀────────────────────────────────-┘
                    │
      ┌─────────────┼──────────────┬─────────────┬──────────────┐
      │ SURVEYOR    │ ESTATE       │ PARK        │ POOL     BIS │  TEMP
      ▼             ▼              ▼             ▼          ▼   │
 ACTION_SURVEYOR  ACTION_ESTATE  ACTION_PARK  ACTION_POOL  ACTION_BIS
      │             │              │             │          │   │
      └─────────────┴──────────────┴─────────────┴──────────┴───┘
                                   │
                                   ▼
                            CHOOSE_PLAN ◀──────────────┐
                                   │                   │
                    ┌──────────────┼───────────┐       │
                    │ pass         │ automatic │ estate│
                    ▼              ▼           ▼       │
            _finish_player_turn  ASK_RESHUFFLE  VALIDATE_PLAN
                    │              │                   │
                    │              └───────────────────┘
                    ▼
            next seat, or the TURN BOUNDARY (§6.3)
```

Facts that are easy to get wrong and are load-bearing:

- **`ASK_RESHUFFLE` returns to `CHOOSE_PLAN`, not to the turn end.** A player may
  validate more than one City Plan in a turn, so plan/validate/reshuffle is a
  **loop**, not a chain.
- **A permit refusal reaches `CHOOSE_PLAN` from two different phases** — from
  `CHOOSE_CARDS` when nothing is playable at all, and from `WRITE_NUMBER` when
  the taken combination's *printed* number has nowhere to go. Both skip every
  effect phase.
- **`ROUNDABOUT_PLACE` returns to `CHOOSE_CARDS`**, so a roundabout turn visits
  `CHOOSE_CARDS` twice. It terminates because building sets `ctx.last_house` and
  passing sets `ctx.roundabout_declined`, and both guard the offer.
- **`_settle()` resolves four transitions without asking**: `ACTION_PARK` with no
  available street, `ACTION_POOL` when unavailable, `CHOOSE_PLAN` with nothing
  scorable, `ASK_RESHUFFLE` when the reshuffle is not on offer. **These are not
  decisions and must never become nodes.**

---

## 3. The macro boundary — what is collapsed, and why

`macro_codec.py`, implementing `ENCODER_V2_SPEC.md` §10.6. **One** split is
collapsed: `CHOOSE_CARDS → WRITE_NUMBER`.

```
macro write     495   (stack, temp delta, box) = 3 × 5 × 33   at CHOOSE_CARDS
macro refuse      3   (stack, PERMIT_REFUSAL)                 at CHOOSE_CARDS
direct refuse     1   nothing playable at all                 at CHOOSE_CARDS
roundabout open   1                                           at CHOOSE_CARDS
primitives      184   the 357-slot codec minus the four above at their phases
                ---
                684
```

⚠ **Corrected justification.** An earlier draft — and `ENCODER_V2_SPEC.md` §10.6,
which has the same wording — said a sequential decomposition "scores *take stack
2* by the **mean** quality of its placements rather than its best". **That is not
true of a correctly constructed deterministic tree.** The stack child is another
decision node controlled by the same player, and UCT is consistent under its
standard assumptions: with enough visits, selection concentrates on the best
continuation and the backup approaches the max, not the mean (§13, Kocsis &
Szepesvári).

The macro is still right, for **finite-budget and action-semantics** reasons:

- it matches the semantic unit of choice — you pick a combination *for* a
  placement;
- it removes an intermediate action that has no meaning on its own;
- it shortens the horizon and improves credit assignment at realistic budgets;
- **measured: it removes 28% of network evaluations** — 8,808 primitive
  decisions become 6,360 macro ones over 30 three-seat GreedyBot games, and there
  are no `WRITE_NUMBER` evaluations at all, which is where the branching lived
  (13.1 mean, 165 max).

The temp delta must stay inside the macro for a hard reason: **it changes
placement legality**, so deferring it would make the write node's legal set
depend on a decision not yet taken.

**Legality is enumerated end to end, never intersected.** `legal_macros` steps
into each playable stack and reads that child's own `legal_actions()`, because
`WRITE` legality depends on which stack was taken. Verified: 193,400 legal macros
applied across 6,360 macro roots without one illegal step.

---

## 4. The within-turn node table — as implemented

One phase is active at a time; the network emits 684 logits and the legal mask
exposes only that phase's actions.

| node | vocabulary | width | after §5.1 pruning |
|---|---|---|---|
| roundabout declaration | one entry in the `CHOOSE_CARDS` set | — | unchanged |
| roundabout placement | box, or pass | 34 | **33** (pass pruned) |
| **combination + write** | (stack, temp delta, box) | **495** | unchanged |
| refusal | stack-specific, or direct | 4 | unchanged |
| surveyor | fence, or pass | 31 | unchanged — a real decision |
| estate | row, or pass | 7 | **6** (pass pruned) |
| park | street, or pass | 4 | **node disappears** |
| pool | build, or pass | 2 | **node disappears** |
| bis | (box, side), or pass | 67 | unchanged — a real decision |
| city plan | slot, or pass | 4 | unchanged |
| plan estate validation | matching estate | ≤ 33 | unchanged |
| reshuffle | yes / no | 2 | unchanged |

**Measured cost:** a turn costs **1.91 network decisions on average** —
distribution 2 in 169 turns, 1 in 62, 7–8 in 8, over 239 sampled turns of
two-seat advanced play. (Measured under a fixed "first legal macro" policy; the
mean will shift with a trained policy, the shape will not.)

---

## 5. Why the turn is not bundled further

Rejected. Two independent reasons, both measured.

**It is too large to index.** A complete-turn action — roundabout, write, effect,
plan, validation, reshuffle as one choice — over 239 sampled turns:

| | sequences per turn |
|---|---|
| mean | 6,367 |
| median | 1,470 |
| max | **71,334** |
| turns over 10,000 | 48 / 239 |

A flat head cannot index that, and a fixed vocabulary would have to cover the
union rather than the realised set. It would need a factored or autoregressive
decoder.

**And it buys nothing.** Because no chance occurs inside a turn, the within-turn
chain is **already fully revisitable** — the same keys recur every simulation and
the chain develops to its natural length. Bundling would save **0.91 network
calls per turn** and cost the conditional sharing that lets a fence-placement
policy be learned across every write that reaches the same surveyor state.

### 5.1 Dominance pruning — search-only, not an engine change

Park, pool and estate passes are **dominated**, and the search should not spend
budget on them. This is a *provable* property, not a valuation:

- `PARK_SCORES` rows are strictly increasing — `(0,2,4,10)`, `(0,2,4,6,14)`,
  `(0,2,4,6,8,18)`; `POOL_SCORES` is strictly increasing —
  `(0,3,6,9,13,17,21,26,31,36)` (`constants.py`);
- all three consume **no** box, fence, number, turn or personal resource — they
  only advance a track: `estate_marks[row] += 1`, `parks[x] += 1`,
  `pools[street] += 1`;
- every plan predicate that reads parks or pools (`DECORATIVE`,
  `COMPLETE_STREET`) is **monotone** — more can only help;
- **plans are not auto-validated** (`CHOOSE_PLAN` has `PASS_PLAN`), so taking a
  park can never force an unwanted three-plan game end.

Bis and the surveyor fence are **not** dominated and keep their pass: bis calls
`sheet.write(..., is_bis=True)`, filling a box and taking the penalty, and the
fence partitions a street into estates and can destroy an `EstatePlan`'s required
sizes.

⚠ **Estate is not a yes/no.** `estate_rows()` is a choice of *which* value row to
advance and the rows score differently, so pruning removes the pass and leaves a
≤6-way node. Park is normally binary (one street — the one just written in) and
pool is binary, so those two nodes vanish entirely once forced nodes collapse.

**`PASS_ROUNDABOUT` is also pruned.** Opening and then passing reaches the same
`CHOOSE_CARDS` state as never opening, minus the roundabout option — so not
opening weakly dominates it. Pruning makes `ROUNDABOUT_OPEN` mean "I will place
one", which is the real decision.

⚠ **This is a search mask, not a rules change.** An earlier draft rejected it on
BGA-fidelity grounds; **that argument applies to the engine, not to the search**,
and conflating them was the error. `GameState.legal_actions()` keeps every BGA
action, the codec keeps all 684 logits, and replay compatibility is untouched.
The search already differs from BGA's primitive tree through the choose/write
macro; fidelity was never a claim about which moves the search explores.

---

## 6. The chance boundary — PROPOSED

Everything from here is for review.

### 6.1 What is built today, and why it is not enough

`MCTS._advance` samples opponents forward to the root player's next decision and
returns `tuple(state.table_cards(root))`; a child is keyed on
`(action, observation)`.

This is correct — it never merges distinct observable states — and it **has no
depth**. Measured mean leaf depth **1.59**, unresponsive to budget or to prior
sharpness:

| prior | mean leaf depth | range |
|---|---|---|
| uniform | 1.59 | 1.06 – 2.08 |
| one-hot | 1.59 | 1.00 – 2.01 |

(4 network seeds × 6 positions, 256 simulations, 2/24 cells differing by >0.5.
One-hot is the *upper bound* on sharpness, so no trained policy deepens it.)

Every boundary crossing draws a key never seen before and expands a fresh leaf.
Budget past roughly one simulation per root action goes into **root averaging**,
not depth.

⚠ **The key is also under-specified, latently.** Raw card **IDs**, no opponent
sheets, no race state. Measured, it costs nothing today — 0 spurious splits in 60
samples — because near-unique reveals mask it. **15 of the 66 printed card types
have two physical copies**, so identical-looking reveals key to different children
the moment children are reused. Fixing the key is a prerequisite for §7.

### 6.2 What a reveal actually is

Not an ordered triple of numbers. A construction card prints its own effect on its
number face, so a card drawn at a boundary supplies a **number for this turn** and
an **effect for the following turn**. Measured, one turn ahead over 60
determinizations of one action sequence:

| | distinct |
|---|---|
| effects in play this turn | **1** — certain, equal to `next_effects` |
| effects for the following turn | **46** |
| numbers in play this turn | 60 |

So a reveal is an ordered triple of printed **card types**: **66 distinct types**
in an 81-card deck (multiplicity 1–2), on the order of 277,000 valid ordered
triples. **Exhaustive expansion is not available.**

### 6.3 ⚠ The boundary is not always "draw three from the histogram"

**Verified against the engine.** A boundary can do four different things, and a
search that samples "three cards from `deck_composition`" is wrong on three of
them:

**First, what a boundary does to the table** (`ConstructionCards::discardAux`,
standard branch — verified against the BGA PHP). It is not "replace three cards":

1. the three **aside** cards (this turn's effects) go to the discard;
2. the three **number** cards are *promoted* into the aside slots — so **this
   turn's numbers become next turn's effects**, which is exactly why
   `next_effects` is a certainty and not a posterior;
3. three **new** number cards are drawn.

The cards on the table are therefore never part of the material being reshuffled;
only the discard is.

| case | behaviour |
|---|---|
| ordinary | discard aside, promote numbers, draw 3 via `_draw_playable()` |
| **exact-empty reform** | the deck is empty **at the start** of the draw: reform from the discard, then draw 3 |
| **queued reshuffle** | `_reshuffle_decks()` reforms, draws **3**, runs `_discard_step()`; then `_begin_turn` calls `_draw_step()` for **3 more** — **six ordered cards, in two batches, with a discard cycle between** |
| terminal | `_end_turn` may reach `GAME_OVER` before any reveal |

⚠ **"Mid-draw exhaustion" is NOT a standard-mode case, and an earlier draft
wrongly listed it.** `_draw()` *can* reform partway through a sequence, but in
2–4 seat standard play deck consumption stays a **multiple of three** — six at
setup, three per ordinary turn, six per requested reshuffle — so the deck is
either empty before the first of three draws or holds at least three.

**Measured:** `deck_remaining % 3 == 0` at the start of **every** `_draw_step` —
56,205 observations over 120 games at 2, 3 and 4 seats, no exceptions. Of 1,058
reforms, none occurred between two draws of a triple.

Keep a generic mid-draw unit test if expert mode is to stay supported; it is not
a search-critical case. Solo (`_draw_playable` resolving `SOLO_CARD_ID`) is
likewise out of the training scope.

**The search must not reimplement this.** Refactor an engine-owned triple:

```
prepare_turn_boundary()          -> afterstate, or terminal
sample_boundary_outcome(rng)     -> one immediate outcome, all four cases
apply_boundary_outcome(outcome)  -> the next observed state
```

with a test per case. Reimplementing the draw in the search is a correctness bug
waiting on a reshuffle turn.

---

## 7. Proposed: fixed-support chance nodes

### 7.1 Shape

⚠ **Not a "pre-reveal afterstate".** An earlier draft placed the chance node
after every seat had acted and sampled only the reveal. That is **wrong for any
root that is not seat 0**, and it contradicted §7.3. From seat 1's decision to
seat 1's next decision, the environment does three things: later seats act, the
boundary reveals, **and then seat 0 acts** — after the reveal, on information
seat 1 does not have, and possibly in response to seat 1's own now-public
previous action. There is no single point in that sequence where "the player
actions are done and only chance remains".

**The retained object is the whole root-to-root environment transition**, sampled
causally: from "the root player finishes acting" to "the root player acts again".
One sample contains exactly the randomness consumed during that one transition —
the opponents' decisions, wherever they fall relative to the reveal, and the
boundary outcome — and **nothing else**. At it:

- draw `K` transitions, each sampled causally end to end;
- give every sample mass **`1/K`**; merge samples with **identical public
  observations** and give the merged child mass **`count/K`**;
- **retain** those children for the life of the search and **close the edge
  against growth**;
- where several samples share a viewer information state but differ in hidden
  detail — most often the opponents' *live* current-turn sheets — the child holds
  a **particle collection**, not one canonical concrete state. Collapsing them to
  a single particle silently narrows the belief (§13, POMCP / POMCPOW);
- the node's value is the **weight-weighted mean** of its children's backed-up
  values — a chance node averages, it does **not** PUCT;
- later descents **sample among the retained children by their weights**.

Retained children are revisited **by construction**, which is the one thing
neither exact observation keying nor prior sharpening can produce. Start with
`K ∈ {4, 8, 16}`.

### 7.2 ⚠ The weighting rule, corrected

The first draft said: sample from the true distribution, keep each outcome's
**original probability**, and renormalise those across the retained subset. **That
is wrong.** It estimates the value *conditional on landing in the retained
subset*, and it double-counts: a common outcome is both sampled more often *and*
carries a larger original probability.

For IID sparse sampling the correct rule is the one in §7.1 — **`count/K`,
empirical mass**. For a *stratified* construction, a representative carries the
mass of its **stratum**, which is again not its individual outcome probability
renormalised.

⚠ **`sum(weights) == 1` proves only mass conservation.** It does **not** prove the
weights approximate the intended distribution. The first draft treated the 7WD
closure check as if it did.

**Consequence for the port:** `seven_wonders_duel/search.py`'s *edge invariants*
are reusable — the three edge classes, closure against growth, the mass check.
Its **support construction is not**: `balanced_double_reveal_chains` assigns
`1/(n·X)` per representative for a specific double-reveal geometry that Welcome
To does not have. Port the invariants; construct the support here.

### 7.3 ⚠ A scenario must NOT contain the future deck order

**The blocker in the first draft.** It defined a retained child by *deck order and
opponent RNG together*. That violates **non-anticipativity**: two scenarios that
reveal the same cards now but differ in their hidden tails would become different
nodes, so the tree's later decisions could differ between worlds the player cannot
distinguish. The tree then knows the future deck through its own node identity —
textbook determinization **strategy fusion**, which is precisely what information
set search exists to prevent (§13, Cowling et al.).

The correct rule:

- a fixed-support entry contains **only the randomness consumed by that one
  root-to-root transition** (§7.1) — the opponents' sampled decisions and the
  boundary outcome;
- decision children are keyed by the resulting **public observation**, never by an
  RNG seed and never by a hidden future;
- **equivalent public outcomes merge** (which is what makes `count/K` the right
  weight);
- the remaining deck stays an **unordered histogram**;
- the **next** boundary draws a fresh conditional support.

Opponent randomness obeys the same rule: retain the opponents' *realised public
actions*, not their future random stream.

This also resolves — correctly — the concern the first draft raised, that a
retained child with resampled opponents is undefined. The answer is not to retain
the RNG; it is that the transition is sampled **once, causally**, and the child is
keyed by what the viewer can then see, with hidden residue carried as particles
rather than as a canonical state.

### 7.4 Common random outcomes across candidate actions

The reveal distribution does not depend on which house the root player wrote, so
candidate root actions should be compared against the same randomness. This
cancels most of the variance in their *differences*, which is what selection
needs.

⚠ **Share the underlying uniforms, not realised opponent actions.** An earlier
draft said opponents "may be shared… they cannot see the root player's concurrent
choice". That is true only of opponents acting in the **same** turn. Opponents
acting **after** the boundary *can* see the root's now-public previous result, so
their policy and even their legal continuations may legitimately differ between
candidate actions; reusing their realised moves would force a counterfactual that
the game does not permit.

The safe contract, which also covers the **reshuffle** decision (which changes
both the distribution and, per §6.3, how many cards are drawn):

> Share the underlying random **uniforms** across candidate actions and map them
> through each action's own causal transition.

### 7.5 Re-rooting on the real reveal

⚠ **The tree does not survive a real boundary, and that is accepted.** With ~277k
possible reveals and `K` samples, the real reveal will essentially never be among
the retained children — measured, 16 sampled reveals were 16 distinct outcomes at
every deck size tested, down to 18 cards remaining. So every real turn starts a
**fresh root** and everything searched below the chance node is discarded.

This is not a regression — it is what happens today — but it means §7.5's
"reuse the matching child" is aspirational and must not be counted on. **Tree
reuse pays inside a turn, not across one** (§12 step 3), because there the
transitions are deterministic and re-rooting is exact.



Construct the new public state, reuse the matching sampled child if there is one,
otherwise start a fresh root. The actual public observation is the anchor; no
persistent ordinal identity is needed.

Within a turn the deterministic subtree **can** be re-rooted after each real
action, because no chance intervenes. **That does not extend across a boundary.**

### 7.6 Decisions taken

**`K` is a schedule, not a constant.** Holding a target depth `H` needs
`K^H ≲ N`, so `K ≈ N^(1/H)`; for depth 2 that is `K ≈ √N` — 8 at 64 simulations,
16 at 256, 32 at 1024. Low `K` early, when simulation counts are low and depth is
what is missing; higher `K` later, when counts rise and the chance expectation
becomes the limiting error.

⚠ **Note what that rule is.** `C · visits^α` with `α = 0.5` **is** progressive
widening. So the staged-`K` intuition and the deferred PW alternative (§10) are
the same idea reached from opposite directions, and PW obtains it *adaptively*
instead of requiring a schedule pinned to training stage. Fixed `K` is still
built first — easier to reason about, easier to batch — but the deferral now
reads "build the simpler thing first", not "probably unnecessary".

**Depth scope stays fluid.** Whether the chance node applies at every boundary
(cost `K^H`) or only near the root is deliberately not fixed here; decide it
against measured training impact once the infrastructure exists.

**The in-search opponent model is the network, batched. Never a bot.** The only
real choice was network versus heuristic — "batched" is a throughput property of
the same model, not a different one. A bot is excluded on principle: GreedyBot
completes **0.42 plans per game** and is structurally race-blind, so using it as
the opponent model would bake race-blindness into every value estimate the search
produces, in a game whose entire competitive surface is the plan race.

**Merging and particles are not built in v1.** Measured inert: at `K = 16`,
sampled reveals were 16-of-16 distinct at every deck size tested, so every
retained child holds exactly one particle and `count/K` collapses to `1/K`.
Implement a **counter** that records any collision; build the machinery only if it
ever fires. (Definitions, for the record: *merging* keeps one child at weight
`2/K` when two samples reach the same viewer information state; *particles* handle
the case where samples share a viewer information state but differ in hidden
detail — most plausibly opponents having written different numbers this turn,
invisible until the next boundary — where the child is a belief, not a state, and
collapsing it to one canonical state narrows that belief silently.)

### 7.7 Root allocation: PUCT now, Gumbel as its own arm

**These are separable from everything above**, because both searches consume the
same priors and the same value head — so the network cannot tell which produced
its targets, and it is legitimate to **train with one and serve with the other**.

The case for Gumbel top-`k` + sequential halving is not mainly low-simulation
strength. It is that **AlphaZero's visit-count policy carries no
policy-improvement guarantee at small `n`** — the target can be worse than the
prior it came from — whereas the Gumbel construction guarantees improvement at
any budget (§13, Danihelka et al.). That is a *training-target* property, and it
binds hardest exactly where throughput wants us: few simulations per move, many
games. Our masked 495-wide macro root is its intended shape.

Serving is not the constraint: a human waiting on the BGA advisor will tolerate a
far larger budget than self-play can afford per move.

⚠ **Sequencing.** Sequential halving allocates a fixed budget across root actions
in rounds, which interacts with a stochastic transition beneath each action.
Compatible, but another moving part — land the chance node first, then add Gumbel
as its own bakeoff arm.

---

## 8. ⚠ Fixed `K` multiplies inference cost by `K`

`NetEvaluator._forward` evaluates **one** state per call. Materialising `K`
children at every new chance node turns 128 simulations into up to **128 × K**
leaf evaluations — ~2,048 at `K = 16` — which would make any bakeoff at "equal
simulations" meaningless.

⚠ **And a leaf is not the only cost.** Because the retained object is the whole
root-to-root transition (§7.1), each sample also pays for the **opponents' policy
evaluations** — which are network calls, per §7.6, not cheap bot rollouts. At four seats and the measured 1.91 decisions per turn, one
`K = 16` expansion approaches

```
16 × (1 leaf + 3 opponents × 1.91 decisions) ≈ 108 evaluated positions
```

before any deeper search happens.

**Required before the bakeoff**, at least one of:

- batch all `K` children of a new afterstate in one tensor call;
- better, batch leaves **across games and searches**;
- or introduce children lazily, so the effective chance-sample count is part of
  the simulation budget rather than a multiplier on it.

**"Equal network-evaluation budget" must count evaluated positions**, not Python
forward invocations. Report positions/second and accelerator utilisation
alongside strength.

---

## 9. ⚠ What fixed `K` does to the estimator, corrected

The first draft said fixed `K` is "lower variance, but biased by that particular
sample". That is loose. For IID `1/K` samples:

- the estimator is **unbiased across independently drawn supports**;
- **a particular retained support carries sampling error**, and everything in that
  search is conditioned on it;
- its variance **across support seeds** is *higher* than an `N`-sample estimator
  when `K < N`;
- repeated traversal reduces internal tree noise but supplies **no additional
  chance information**;
- taking the max over noisy action estimates introduces **optimiser bias**.

⚠ **Therefore visit-based stderr inside one retained tree is falsely tiny** — it
counts repeat visits to the same `K` outcomes as new evidence. Root uncertainty
must be measured **across independent support seeds**.

Add to the bakeoff:

- between-support-seed value variance;
- action agreement across support seeds;
- effective unique chance samples;
- optimiser bias against the exact late-game oracle.

**The oracle:** late-game states with a small remaining deck, where every legal
ordered reveal can be enumerated exactly. Compare each approximation's root
ranking *and* its root value against the exact expectation.

---

## 10. Rejected and deferred alternatives

| design | verdict |
|---|---|
| **Bundle the whole turn** | Rejected — §5. Up to 71,334 sequences per turn to save 0.91 network calls. |
| **Sort the stacks** (KD's `sorted(deck[:4])`) | Rejected. KD's sort is meaningful because domino number *is* next-round pick order, so the slot encodes tempo whatever tile it carries. Sorting Welcome To's stacks gives "lowest of three random draws" — 1 in one determinization, 13 in another. No stable quantity. |
| **Merge on `(number, effect, box)` + availability counts** | Experimental arm. Invariant, and availability counts are well founded (§13, Cowling et al.), but **context-abstracted**: one `Q` averaged over what the other two offers are, what next-turn effects were exposed, and the race state. Also lossy — a 7 printed and a 7 made with TEMP consume different cards. |
| **Progressive widening on chance** | Deferred, not rejected. Withdrawn once for a bad reason: "sharper priors do not deepen the tree" is measured and true and does **not** imply nothing structural does. Fixed `K` first because it is easier to reason about and to batch (§13, Couëtoux & Doghmen). |
| **Gumbel top-`k` + sequential halving at the root** | Later arm. The masked 495-wide macro root is a good fit — it targets simple regret where simulations are scarce relative to root width (§13, Danihelka et al.). Keep it out of the first chance-node implementation. |

---

## 11. Bakeoff

At equal wall-clock **and** equal evaluated-position budget:

| arm | chance handling |
|---|---|
| current | fresh exact observation every simulation |
| semantic ISMCTS | merged `(number, effect, box)` + availability counts |
| sparse chance | retained `K = 4 / 8 / 16`, `count/K` weights |
| sparse chance + afterstate head | learned chance expectation |

Report for each: paired score gap and stderr over 300+ games; strength at
64/128/256 evaluated positions; mean turn boundaries crossed; action agreement
across search RNGs **and across support seeds**; root value variance between
support seeds; effective unique chance samples; optimiser bias against the oracle;
positions/second, wall time and accelerator utilisation; plans, permits, end
reasons.

⚠ **The gate needs ~300 games, not 60.** Measured: the per-game paired delta has a
standard deviation of about **18 points**, so stderr is 4.1 at 60 games and 1.0 at
330. A 4-point difference read off 60 games sits inside its own noise.

⚠ **All of this waits on S0.** Every arm needs a trained network. Two claims in
`SELF_PLAY_PLAN.md` were wrong because they were measured on an untrained one.

---

## 12. Implementation order

1. **Search-only dominance pruning** — park/pool/estate passes, `PASS_ROUNDABOUT`
   (§5.1). No engine change.
2. **Collapse forced internal nodes** inside simulations, not only at the external
   root. `MCTS.play` already does the root; `_simulate` does not.
3. **Deterministic within-turn re-rooting.** `MCTS.play` discards the tree after
   choosing; re-rooting the selected child within a turn is *exact* and preserves
   the work.
4. **Engine-owned boundary transition** (§6.3), with a test for each of ordinary
   draw, **exact-empty reform**, queued reshuffle, and terminal-before-reveal;
   plus a generic mid-draw test only if expert mode stays supported.
5. **Define and test the viewer information-state key** (§6.1, §7.3, §12.1).
6. **Batched evaluator** — successor batches *and* opponent-policy batches (§8).
   **Before** step 7, because a retained transition costs both.
7. **Fixed-support environment edge** (§7) with `count/K` weights and particle
   children, behind a flag, the current version kept as the control arm. Build it
   **lazily** if step 6 slips, so no unbatched eager `K` materialisation exists
   even temporarily.
8. **Run S0, then the bakeoff** (§11).
9. **Only then:** progressive widening, Gumbel root allocation, afterstate head.

Steps 1–3 are small, independent of the review's outcome, and can start now.
Steps 4–7 are what this document exists to have reviewed; step 7 should not begin
until the transition semantics of §7.1 and §7.3 are agreed.

### 12.1 The key is an information state, not a public observation

"Public observation" is too weak a name: the key is **viewer-relative**, and must
contain

- the viewer's **live** sheet;
- the opponents' **turn-start public snapshots** (`public_sheets`);
- the **printed card types** on the table, not card IDs;
- viewer-visible plan validations and race state (`plan_turns_for`);
- the exact visible deck and discard composition;
- the viewer's **own** reshuffle vote;
- turn, phase, and the relevant within-turn context.

and must **exclude**

- any future deck order;
- the opponents' hidden current-turn writes;
- the table-wide `reshuffle_next_turn` aggregate (which is private mid-turn — see
  `ENCODER_V2_SPEC.md` §9.3a).

**The test that matters** is seat 1 after seat 0 has acted: mutating seat 0's
*live* current-turn sheet must leave seat 1's key **unchanged**, while mutating
`public_sheets[0]` must **change** it.

---

## 13. Research references

Collected so the reasoning above can be re-examined rather than re-derived.

**Information set search and strategy fusion** — Cowling, Powley & Whitehouse,
*Information Set Monte Carlo Tree Search* (2012).
<https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf>
Why §7.3's non-anticipativity rule is not optional, and the source of the
availability-count correction used by the semantic-ISMCTS arm.

**Sparse sampling** — Kearns, Mansour & Ng, *A Sparse Sampling Algorithm for
Near-Optimal Planning in Large MDPs* (1999).
<https://ai.stanford.edu/~ang/papers/ijcai99-largemdp.pdf>
The foundational result behind fixed-`K`: planning cost independent of state-space
size, exponential in horizon. The source of the `1/K` weighting in §7.1.

**Densely stochastic games** — Lanctot, Saffidine, Veness, Archibald & Winands,
*Monte Carlo \*-Minimax Search* (2013). <https://arxiv.org/abs/1304.6057>
Directly on point: sparse chance sampling where chance successors almost never
repeat. Evaluated on Can't Stop.

**UCT consistency** — Kocsis & Szepesvári, *Bandit-Based Monte Carlo Planning*
(2006).
<https://aima.cs.berkeley.edu/~russell/classes/cs294/s11/readings/Kocsis%2BSzepesvari%3A2006.pdf>
Why §3's original "mean not best" justification for the macro was wrong, and what
the macro's real (finite-budget) benefit is.

**Belief-state search and particle collapse** — Silver & Veness, *Monte-Carlo
Planning in Large POMDPs* (POMCP, 2010).
<https://papers.nips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf>
Sunberg & Kochenderfer, *Online Algorithms for POMDPs with Continuous State,
Action, and Observation Spaces* (POMCPOW, 2018).
<https://arxiv.org/abs/1709.06196>
Why §7.1's children need particle collections: widening with a single state
particle collapses the belief incorrectly.

**Double progressive widening** — Couëtoux & Doghmen, *Adding Double Progressive
Widening to Upper Confidence Trees to Cope with Uncertainty in Planning Problems*
(2011).
<https://ewrl.wordpress.com/wp-content/uploads/2011/08/ewrl2011_submission_29.pdf>
The deferred alternative in §10, designed for exactly the case where vanilla MCTS
keeps meeting new successors.

**Gumbel planning** — Danihelka, Guez, Schrittwieser & Silver, *Policy Improvement
by Planning with Gumbel* (2022).
<https://openreview.net/pdf/4f2c0c813d0fbe127329c69b1ba216fbcd95d52c.pdf>
Simple-regret root allocation for wide roots at small budgets — the later arm in
§10.

**Afterstate factorisation** — Antonoglou, Schrittwieser, Ozair, Hubert & Silver,
*Planning in Stochastic Environments with a Learned Model* (Stochastic MuZero,
2022). <https://mlanthology.org/iclr/2022/antonoglou2022iclr-planning/>
The afterstate value head of §10/§12.9. Only the factorisation is wanted; the
chance distribution here is known exactly, so the learned chance model is not.

**Value networks with expectimax** — Matsuzaki, *Developing Value Networks for
Game 2048* (2021).
<https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_article>
Empirical precedent for a learned value plus shallow expectimax in a densely
stochastic game.

**In-repo precedent** — `games/seven_wonders_duel/search.py`. The `_Edge` class,
its three edge classes, `close_fixed_support()` and `fixed_support_index()`.
Reusable for the **invariants**; its support construction is specific to that
game's reveal geometry (§7.2).
