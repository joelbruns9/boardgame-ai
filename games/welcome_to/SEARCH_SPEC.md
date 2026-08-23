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

**Decisions taken 2026-08-22** (§7.6). `K` is a schedule rather than a
constant; the in-search opponent model is the network, batched, and never a bot;
retained children are discarded at every real turn boundary.

⚠ **Two of those decisions were withdrawn on review.** "The transition
semantics of §7.1/§7.3 are agreed" was premature — §7.1a shows the enumerated
edge counted *reveals* where §7.1 defines the outcome as a whole transition. And
"merging and particles are measured inert and will not be built in v1" was
measured only down to 18 cards.

**Decisions taken 2026-08-23.** §7.1a resolution **C**: the chance edge is
**progressive widening** with empirical `count/samples` weights and children
merged by viewer information state — so merging is the mechanism rather than an
optional extra, particles are built, and the enumerated second edge class is
dropped. §7.8: Dirichlet noise applies at **every** decision root, with
`α = 10 / legal actions` and a fresh-simulation floor as a fraction of the
budget.

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

**Measured cost — re-measured after §12 steps 1–2, and the old figure retired.**

⚠ **The previous number, "1.91 network decisions per turn over 239 sampled
turns", did not reproduce and has been withdrawn.** Under a literal
"first legal macro" policy — `legal_macros(state)[0]` — the mean over 740
player-turns is **1.64**, and *no* turn costs more than two decisions. The old distribution's
tail (7–8 decisions in 8 turns) can only come from roundabout turns, and this
policy never opens a roundabout, because a `MACRO_WRITE` always precedes
`ROUNDABOUT_OPEN` in `legal_macros`. Whatever policy produced 1.91, it was not
this one. The figures below are therefore measured on **GreedyBot** play, which
does open roundabouts at the reference rate (1.22 built per seat-game against
the 1.28 quoted elsewhere, so the harness agrees with the earlier corpus).

| driver | player-turns | macro decisions / turn | **network decisions / turn** |
|---|---|---|---|
| GreedyBot, 2 seats | 1,488 | 2.45 | **2.14** |
| GreedyBot, 3 seats | 2,199 | 2.41 | **2.10** |
| GreedyBot, 4 seats | 2,932 | 2.42 | **2.11** |
| first legal macro, 2 seats | 740 | 1.64 | 1.47 |

⚠ **Everything in this section is measured on GreedyBot trajectories, and that
limits what it may be used for.** The *structural* facts are policy-independent
and are rules, not statistics — park is offered only for the street just written
in, pool only for the box just written, `estate_rows()` masks full rows,
`ROUNDABOUT_PLACE` offers the whole free sheet. The *frequencies* are not: which
effect phases a turn reaches depends on which cards you take, and a competent
player takes different cards. Read the shape here, and re-read the rates off a
real checkpoint before costing anything against them.

⚠ **The widths in the table above are the *vocabulary*, not the legal set**, and
the difference is a factor of ten where it matters most. Measured over 7,807
real search roots at 2 and 3 seats — decisions with more than one legal macro,
so exactly what the search sees:

| phase | roots | mean legal | median | p90 | max |
|---|---|---|---|---|---|
| `CHOOSE_CARDS` | 4,580 | **49.7** | 30 | 124 | 331 |
| `ROUNDABOUT_PLACE` | 1,163 | 16.7 | 16 | 30 | 33 |
| `ACTION_ESTATE` | 796 | 5.2 | 5 | 6 | 6 |
| `ACTION_SURVEYOR` | 734 | 27.0 | 28 | 31 | 31 |
| `ACTION_BIS` | 361 | 6.7 | 7 | 10 | 16 |
| `CHOOSE_PLAN` | 87 | 2.0 | 2 | 2 | 3 |
| `VALIDATE_PLAN` | 51 | 2.8 | 2 | 4 | 7 |
| `ASK_RESHUFFLE` | 35 | 2.0 | 2 | 2 | 2 |
| **all** | **7,807** | **35.1** | 19 | | |

The combination-and-write root is 495 *indices* wide and **49.7 actions** wide in
practice. Anything sized against the root — Dirichlet `alpha` (§7.8), Gumbel
`top_k` (§7.7), a widening constant — must be sized against this table, not
against 495.

*Macro decisions* counts every state the macro layer decides at. *Network
decisions* counts only the ones the search evaluates — after §5.1 pruning and
forced-node collapse. **Use 2.14 wherever the old 1.91 was used, and 2.45 for
anything that costs an opponent's turn** (§8): `_advance` samples opponents
through *every* decision, forced ones included. Seat count barely moves either
number, which is what a private per-sheet turn should do.

Where the saving comes from — 25 GreedyBot games at two seats, shipped config:

| phase | states visited | nodes before | nodes after |
|---|---|---|---|
| `CHOOSE_CARDS` | 1,973 | 1,857 | 1,857 |
| `ROUNDABOUT_PLACE` | 485 | 485 | **484** |
| `ACTION_ESTATE` | 320 | 320 | 320 |
| `ACTION_SURVEYOR` | 305 | 305 | 305 |
| **`ACTION_PARK`** | 280 | 280 | **0** |
| `ACTION_BIS` | 145 | 145 | 145 |
| `VALIDATE_PLAN` | 53 | 22 | 22 |
| `CHOOSE_PLAN` | 38 | 38 | 38 |
| **`ACTION_POOL`** | 37 | 37 | **0** |
| `ASK_RESHUFFLE` | 16 | 16 | 16 |

**The pruning table above is accurate; what it does not say is how often each
row fires.** Two of the four narrow a node that stays a node, and it is worth
knowing *why*, because "it would collapse if the track ran out" is true and
never happens:

- **`ACTION_ESTATE` narrows 7 → ≤ 6 and never collapses.** Full rows are
  already masked by the engine — `estate_rows()` is
  `estate_marks[i] < ESTATE_ROW_BOXES[i]` over `(1, 2, 3, 4, 4, 4)`, so the
  size-1 row leaves the legal set after its single mark and no search-side
  masking is needed. Emptying the node needs **five of six rows full, 14 of the
  18 boxes**, and the phase is only reached ~6.4 times per seat-game. Measured
  over 320 visits: never below **3** rows, 85% of visits at 5 or 6, and a mean
  of 4.74 from turn 15 on. It is a genuine 3-to-6-way choice essentially always,
  which is the point of §5.1's "estate is not a yes/no".
- **`ROUNDABOUT_PLACE` narrows 34 → 33 and never collapses.** A roundabout may
  go in *any* empty box, so the engine offers `available_locations(None)` — the
  whole free sheet. Measured over 485 visits the placement count spans 1…33 and
  is a singleton exactly **once**; 8% of visits have ≤ 4 boxes free (late game),
  10% have 30+.
- **Park and pool do vanish, all 317 of them**, and that is the whole of the
  measured 2.45 → 2.14. Park is offered only for the street the number just went
  into and pool only for the box just written, so both are binary; with the pass
  dominated, the build is all that is left.

**Re-rooting (§12 step 3) preserves 9% to 48% of the simulation budget**, and
*which* depends entirely on how concentrated the policy is. Measured under the
shipped config, four games, two seats:

| evaluator | budget | decisions re-rooted | simulations preserved |
|---|---|---|---|
| untrained net (flat priors) | 32 | 34% (45 / 132) | 8.7% |
| untrained net (flat priors) | 128 | 38% (57 / 152) | 11.9% |
| GreedyBot-cloned prior (S0's shape) | 32 | 51% (124 / 243) | **47.7%** |
| GreedyBot-cloned prior (S0's shape) | 128 | 53% (133 / 253) | **45.6%** |

A flat prior spreads 32 visits over a ~50-way root, so the child re-rooted onto
carries almost nothing; a prior that concentrates — which is what behaviour
cloning produces, and what the whole design assumes — carries most of the tree.
**Budget for the S0-shaped figure, not the untrained one.** The budget is
almost irrelevant to the fraction — 8.7% → 11.9% flat, 47.7% → 45.6% sharp,
from 32 to 128 simulations — because it scales both the retained and the fresh
side. The **prior** is the variable that matters, and it is worth a factor of
four. Note also that only about half of all decisions re-root at all: the rest
are the first decision of a turn, where the tree is correctly discarded.

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
the chain develops to its natural length. Bundling would save **1.14 network
calls per turn** — one call replacing the 2.14 of §4 — and cost the conditional
sharing that lets a fence-placement policy be learned across every write that
reaches the same surveyor state. (This read 0.91 while §4 read 1.91; both were
re-measured together.)

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
`CHOOSE_CARDS` state as never opening, minus the roundabout option — it is BGA's
confirm dialog, opened and cancelled — so not opening weakly dominates it.
Pruning makes `ROUNDABOUT_OPEN` mean "I will place one", which is the real
decision. ⚠ It is the one pruning behind a switch, because it is the one that
interacts with a *bootstrap* prior: §5.1a, and read roundabouts per game off the
S0 checkpoint.

⚠ **This is a search mask, not a rules change.** An earlier draft rejected it on
BGA-fidelity grounds; **that argument applies to the engine, not to the search**,
and conflating them was the error. `GameState.legal_actions()` keeps every BGA
action, the codec keeps all 684 logits, and replay compatibility is untouched.
The search already differs from BGA's primitive tree through the choose/write
macro; fidelity was never a claim about which moves the search explores.

### 5.1a ⚠ `PASS_ROUNDABOUT` is pruned — and the S0 bootstrap needs watching

**What `PASS_ROUNDABOUT` actually is.** In advanced play, `CHOOSE_CARDS` offers
`ROUNDABOUT_OPEN` *alongside* the normal "take a stack" actions. Taking it moves
to `ROUNDABOUT_PLACE` and **does not touch the sheet**; there you either place a
roundabout in an empty box or play `PASS_ROUNDABOUT`, which sets
`ctx.roundabout_declined` and returns to `CHOOSE_CARDS` with the offer now
suppressed and your actual turn still ahead of you. So open-then-pass is
**BGA's confirm dialog, opened and cancelled** — a pure no-op that exists
because the UI has a button. Pruning the pass makes `ROUNDABOUT_OPEN` mean "I
will place one", which is the only decision there ever was. §5.1's dominance
argument is right and this ships on.

⚠ **The one thing to watch is the bootstrap prior, and the reason is measured.**
GreedyBot plays `ROUNDABOUT_OPEN` ~30% of the times it is offered, and that is
**not a preference** — it is a tie-break artifact of a no-op action. Because
opening changes nothing on the sheet, GreedyBot's one-ply
`_evaluate(state.step(OPEN))` is *identical* to doing nothing, so it ties with
every other score-neutral move:

| over 1,418 offers at `CHOOSE_CARDS`, 25 games, 2 seats | |
|---|---|
| `ROUNDABOUT_OPEN` in the tied-best set | **100%** |
| `ROUNDABOUT_OPEN` strictly best | 4% |
| mean size of that tie set | 3.5 actions |

A uniform pick over 3.5 tied actions is where the ~30% comes from, and then at
`ROUNDABOUT_PLACE` the bot passes, because *placing* does cost points and does
break the tie. S0 is behaviour-cloned from this, so **S0 will faithfully
reproduce a coin flip on a no-op** — and with the pass pruned, that coin flip
now commits to a −3 or −8 build.

How much it costs, against a synthetic 0.8-mass clone of that prior (six games,
one search seat, two seats, 32 simulations, identical seeds):

| pruning | roundabouts / game | seat score |
|---|---|---|
| none | 1.00 | 48.5 |
| park + pool + estate | 1.33 | 63.3 |
| all four | **2.00** — `ROUNDABOUT_BOXES`, i.e. the cap | 39.3 |

and 128 simulations does not recover it (still 2.00; acceptance 35% → 29%).

⚠ **Do not read those points as a verdict on the pruning.** They measure the
artifact. The prior under test is a hard clone of a bot's coin flip, the value
head is a stand-in, and six bot games are six bot games — this says what happens
to *that* prior, not what a competent policy does. A good player never opens the
dialog without meaning to build, so for any policy worth having, "open" and
"build" are the same decision and the pruning is free.

**What is actionable, and it is not a spec change:**

- **Read roundabouts per game off the S0 checkpoint before trusting it**, next
  to plans per game in the S1 arena report. GreedyBot builds 1.22 per seat-game.
  Near `2.00` means the bootstrap inherited the coin flip.
- If it has, the fix is in the **bootstrap data or the prior**, not the search:
  measured, the search will not out-visit 0.8 of prior mass at any budget tried.
  `SearchConfig.prune_roundabout_pass = False` exists as a bootstrap-only
  escape hatch, not as a recommendation.
- The clean fix, if one is wanted later, is at the data level: an
  open-then-pass pair in a recorded trajectory is a no-op that teaches the
  network an action with no meaning. That is a `datagen` question and is out of
  scope here — the labels and replay masks are frozen (§12 step 1).

### 5.1a-bis ✅ RESOLVED on the real S0 checkpoint — 2026-08-23

§5.1a's caveat said the synthetic 0.8 clone was not a trained network, and that
the conclusion would change if a trained S0 put *less* mass on
`ROUNDABOUT_OPEN` than its teacher. **S0 has now been run (5,000 games, 4
epochs), and it puts far more.**

| | roundabouts / game | seat score | `ROUNDABOUT_OPEN` taken |
|---|---|---|---|
| GreedyBot | 1.12 | 50.60 | **36%** of offers |
| **S0 checkpoint** | **0.00** | 23.10 | **90%** of offers |

⚠ **The mechanism is not the one §5.1a described, and the correction cuts both
ways.**

- **Predicted:** a cloned prior opens at roughly the teacher's rate (~31% in the
  synthetic arm) and, with the pass pruned, converts those openings into builds.
- **Measured:** the real S0 opens **90%** of offers and then **passes every
  time** — 0.00 roundabouts built across 40 games. Open-then-pass is a no-op, so
  this costs no points. What it costs is *decisions*: following the net,
  `ROUNDABOUT_PLACE` is **27%** of everything it does, against 12.8% following
  greedy. A compulsive habit, faithfully cloned from a coin flip.

**The conclusion is unchanged and its margin is larger.** Pruning
`PASS_ROUNDABOUT` converts openings into builds, and the real checkpoint offers
~837 of them per 40 games where the synthetic arm implied ~290 — so the cost of
that flag is roughly **three times** what §5.1a measured.
`SearchConfig.prune_roundabout_pass = False` stands, on stronger evidence.

⚠ **And this is not where S0's points went.** Opening and passing is free; the
score gap is a write-quality problem. See `SELF_PLAY_PLAN.md` S0.

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

✅ **The key was under-specified, and is now fixed** (§12.1, step 5).
`MCTS._advance` returns `mcts.information_key(state, root)`.

It was wrong in *both* directions at once, which is why "it costs nothing today"
was true and misleading:

- **too fine** — raw card **ids**, and 15 of the 66 printed types have two
  physical copies, so two reveals identical to everyone at the table key apart;
- **too coarse** — the table and nothing else: no opponent sheets, no race
  state, no deck composition, so two transitions differing only in what an
  opponent *published* would merge.

⚠ **Measured neutral to change, and that is the point.** Over **3,840**
`(node, action, observation)` edges at 128–256 simulations, the id key and the
information key induce **exactly the same partition** — 0 edges differ. Reveals
are near-unique, so the table already separated every crossing. The fix is
therefore free to take now and the search is unchanged by it; what it buys is a
key that is still correct once §7.1a retains and merges children, which is when
both failure modes start costing.

(An earlier reading of this measurement said the information key *split* 13 edges
the id key merged. That was a grouping error — the id key is only ever compared
among one node's own children, and grouped that way the two agree exactly.)

**Cost:** ~31 µs per simulation step, about **+7%** of search wall clock today,
and a larger share once step 6 makes the network cheaper. Two `(15, 6)`
composition matrices are 42% of it; they encode as `int16` bytes rather than
tuples of Python ints, which took the key from 48 µs to 31 µs.

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
triples.

⚠ **Exhaustive expansion is unavailable *early* and available *late*, and an
earlier draft claimed it was simply unavailable.** The outcome space shrinks with
the deck, and it shrinks fast. Measured, on real states, with the exact ordered
type-triple space computed from the remaining multiset:

| deck left | possible reveals | distinct of 16 sampled | collisions |
|---|---|---|---|
| **3** | **6** | 5 | **11** |
| **6** | 120 | 15 | 1 |
| 9 | 504 | 16 | 0 |
| 12 | 1,020 | 16 | 0 |
| 18 | 2,856 | 16 | 0 |
| 30 | 14,160 | 16 | 0 |

Because consumption is a multiple of three and reform happens at zero (§6.3), the
deck passes through 6 and 3 **exactly once per cycle** — roughly 2 turns in 25.
That regime is guaranteed to occur, and it is precisely where enumeration is
trivial.

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
reforms, none occurred between two draws of a triple. ✅ Now asserted per seat
count by `test_standard_play_never_exhausts_the_deck_mid_triple`.

The generic mid-draw case is kept as a unit test because `_draw` supports it and
expert mode is still in the engine; it is driven directly, since standard mode
cannot reach it. Solo (`_draw_playable` resolving `SOLO_CARD_ID`) is likewise out
of the training scope — but it is an *extra* raw draw, so it is recorded and
replayed like any other card rather than being a case the outcome type has to
know about.

**The search must not reimplement this.** ✅ **Built** (`game.py`), as an
engine-owned triple:

```
prepare_turn_boundary()      -> bool   True if a reveal follows, False if the
                                       game ended here (the fourth case)
sample_boundary_outcome(rng) -> BoundaryOutcome    does not modify self
apply_boundary_outcome(out)  -> None   in place, deterministic
```

⚠ **The refactor established something the design above did not state, and §7
depends on it: **in standard 2–4-seat play**, *which* case fires is not
chance.** A queued reshuffle is `reshuffle_next_turn` and an exact-empty reform
is `deck_remaining == 0`, and both are settled by the afterstate before a card is
seen. Only *which cards* is chance. So a chance node's support is over card
sequences of a **known length** — 3 ordinarily, **6** on a queued reshuffle — and
never over "which of four things happens".

⚠ **Scoped deliberately, and solo is the reason.** `_draw_playable` consumes an
*extra* raw card when the solo marker turns up, so a solo deck can reform
mid-sequence and the case stops being a function of the afterstate alone.

⚠ **The justification is the rules argument, not the sample count.**
Consumption is a multiple of three — six at setup, three per ordinary turn, six
per requested reshuffle — so the deck is either empty before the first of three
draws or holds at least three. The measurement (**84,808 boundary crossings** at
2/3/4 seats: 83,251 ordinary, 960 exact-empty, 597 queued reshuffle; 3 draws on
every ordinary and exact-empty boundary, 6 on every queued reshuffle, no
exceptions) is **regression coverage of that argument**. It is not independent
evidence: most of those crossings are inside GreedyBot's one-ply lookahead and
are not independent samples.

`sample` and `apply` are the same code path with the draw redirected — recording
on one side, replaying on the other — so all four cases are correct *by
construction* rather than by a second implementation agreeing with the first.
`test_the_three_part_boundary_is_the_engine_s_own_boundary` pins that on all
three reveal cases, and there is a test per case besides.

**`BoundaryOutcome` is the ordered sequence of raw draws**, not the resulting
table, because the four cases put cards in different places. Two properties it
holds structurally rather than by convention:

- **every card in it is one the boundary makes public** — tested; so an outcome
  is a legitimate child key;
- **it does not record the deck order it leaves behind.** `apply` swaps each
  named card to the top of the undrawn region, so composition stays exact while
  the residual order stays arbitrary. That is §7.3's non-anticipativity rule
  made unbreakable: a scenario cannot leak the future because it never contains
  it.

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
boundary outcome — and **nothing else**.

### 7.1a ✅ DECIDED: progressive widening with empirical weights

**A reveal is not a transition.** This invalidated the two-edge-class design
below as written, and blocked step 7 until **2026-08-23, when resolution C was
taken**. Step 7 is unblocked and builds C.

The paragraph above defines the retained object as the **whole root-to-root
transition** — the opponents' sampled decisions *and* the boundary outcome. The
"small outcome space" counts (`6` at `D=3`, `120` at `D=6`) are from §6.2 and
count **reveals**. They are not counts of transitions.

With three cards left there are six ordered reveals, but each of them still has
many opponent trajectories underneath it. So "enumerate every outcome, weight it
by its true probability" cannot be done as stated:

- weighting six children by `P(reveal)` while each holds **one** opponent
  trajectory prices the reveal exactly and the opponent expectation not at all —
  it collapses the belief to a single particle, which is precisely the failure
  §13's POMCP reference is cited for;
- weighting them by the true probability of the *transition* would require
  enumerating opponent action sequences too, and that is not small — the
  opponent policy is a 684-wide distribution per decision and §4 measures 2.45
  decisions per opponent turn.

Two consequences elsewhere in this document, both of which follow and neither of
which was noticed:

- **§7.6's "particles are unnecessary for the enumerated edge" is false.** An
  enumerated reveal child aggregates opponent trajectories that share a viewer
  information state and differ in hidden detail — exactly the definition of the
  particle case given in that same section.
- **§9's "exact late-game oracle" is not exact.** Enumerating ordered reveals
  fixes the chance layer only; it is an oracle for the root value only if
  opponent randomness is also fixed (making it a different game) or integrated
  (making it not enumerable).

**Three resolutions were on the table. ✅ C is decided.**

| | design | weights | cost |
|---|---|---|---|
| **A** | **Factor the transition**: sampled-opponent → *exact-reveal* → sampled-opponent, so the enumerated layer is only the chance layer | reveal children carry `P(reveal)` exactly; opponent layers stay Monte Carlo | three layers to key and batch; reveal children still hold particle collections |
| **B** | **Retain `K` complete trajectories at every deck size**, late-game reveal enumeration demoted to **stratification** | `P(reveal) × empirical conditional opponent mass` | one edge class; still needs particles where a stratum draws more than one sample |
| **C** ✅ | **Progressive widening on the chance edge**, children merged by viewer information state | **`count / samples`** — empirical, never enumerated | merging becomes mandatory rather than optional; two constants, both pinned below |

**Why C was taken, and why it makes the blocker disappear rather than
answering it.**

The problem above exists *only because the design tried to assign exact
probabilities to enumerated outcomes.* Empirical weights never need to know
`P(transition)`; they estimate it. So the reveal-versus-transition confusion
cannot arise — whatever the outcome turns out to be, `count / samples` converges
to its true mass, opponent randomness included, which is exactly the quantity A
has to build three layers to price and B has to factor into two.

Concretely, at a chance node with `n` visits:

- allow `⌈C · n^α⌉` distinct children; when the criterion fires, sample **one**
  fresh transition causally (opponents and `sample_boundary_outcome`);
- if it lands on an existing child's viewer information state, **merge** —
  `count += 1` — instead of adding a child;
- weight `= count / samples`; the node's value is the weight-weighted mean; a
  descent samples among children by weight. A chance node averages, it never
  PUCTs;
- samples sharing an information state but differing in hidden detail are held
  as **particles** — a list of concrete states, uniform on descent.

⚠ **RETRACTED 2026-08-23 — the late-game case does *not* fix itself.** This
said that at `D = 3` widening would keep merging onto the six possible reveals,
saturate at the true support and stop. Two things are wrong with it.

- **A cap cannot close an edge whose support is smaller than the cap.**
  `len(outcomes)` stops at the support while `ceil(C·n^α)` keeps growing, so the
  closure predicate goes false for ever and the edge re-samples on nearly every
  traversal. Ordinary progressive widening **cannot detect exhaustion from
  collisions alone**, and §7.9 records what that cost before it was fixed.
- **Six is the number of *reveals*, not of transitions** — the same conflation
  §7.1a exists to correct. The retained outcome is the whole root-to-root
  transition and carries the opponents' sampled decisions, so a three-card deck
  gives a chance edge **20** outcomes, not six.

What survives is narrower and is the actual reason C was taken: `count/draws` is
an **empirical** weight, so it never needs `P(transition)` — the quantity that
could not be computed. Support exhaustion is a separate question and this design
does not answer it. Determinism *is* detectable, and is handled exactly:
a transition that changes neither the turn nor the game's end consumed no
randomness, so its support is provably one and its edge is closed permanently.

**α is not a free parameter, and this document already derived it.** §7.6 fixes
`K ≈ N^(1/H)` from a target depth `H`, and notes in the same breath that
`C · visits^α` with `α = 0.5` *is* progressive widening. So `α = 1/H`, and the
depth-2 target gives `α = 0.5`. The staged-`K` schedule and PW are the same rule;
PW just reads `n` instead of being pinned to a training stage, which is what
"scale with the simulation count" requires and what a schedule keyed on training
stage cannot deliver — the same checkpoint searches at different budgets in
self-play, arena and analysis.

**What C costs, stated honestly:**

- **Merging stops being optional.** It is the mechanism, not a bookkeeping
  nicety, so §7.6's "not built in v1" is withdrawn for merging as well as for
  particles.
- **A young node's weights are non-stationary** — a child's mass moves as
  siblings arrive — so early chance estimates are noisier than a closed
  `K`-sample average. Fixed `K` stays the control arm in the bakeoff (§11)
  rather than being assumed worse.
- **Children arrive one at a time instead of `K` at once.** ⚠ This is a *gain*
  against §12 step 7's own requirement to "build it lazily … so no unbatched
  eager `K` materialisation exists even temporarily": there is no eager
  materialisation to avoid. Batching then has to come from step 6's
  across-games leaf batching, which §8 already says is the better form.

⚠ **Under *any* of the three, particles stop being optional at the small end**,
so §7.6's "not built in v1" has to be revisited rather than carried forward. The
minimum viable form is small: a child holds a **list** of concrete states instead
of one, and a descent picks among them uniformly. That is a particle collection,
and it is nearer ten lines than a subsystem.

**Two edge classes, chosen by the size of the outcome space** (§6.2) — ⚠ **as
written this is the design 7.1a invalidates; kept for the review, not to build
from.** This is why `seven_wonders_duel/search.py`'s three-class structure is
worth porting faithfully — Welcome To genuinely uses both, rather than one plus
defensive bookkeeping:

| outcome space | edge | weights |
|---|---|---|
| small (late deck: **6 reveals** at `D=3`, **120 reveals** at `D=6`) | **`probability_weighted`** — enumerate every outcome | each outcome's **true** probability |
| large (everything else) | **`fixed_support`** — sample `K` | `count/K` |

Enumeration also **subsumes the merging question** in the only regime where
merging fires: an enumerated edge lists each outcome once, so there is nothing to
merge. Pick the threshold on outcome count, not on `deck_remaining` — a cap
around a few hundred covers `D ≤ 6` cheaply and can be raised if profiling allows.

At a sampled edge:

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
- decision children are keyed by the resulting **viewer information state**
  (§12.1 — "public observation" is too weak a name, and this line said it),
  never by an RNG seed and never by a hidden future;
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
transitions are deterministic and re-rooting is exact — ✅ built, and measured at
47.7% of the budget against an S0-shaped prior (§4).

So the rule at a real boundary is: construct the new state, reuse the matching
sampled child **if** there happens to be one, and otherwise start a fresh root.
The viewer information state is the anchor; no persistent ordinal identity is
needed. Expect the fresh-root path essentially always.

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

**Merging: not built for the sampled edge. ⚠ Particles: needed after all —
see §7.1a.**

⚠ **An earlier version of this decision said merging is "measured inert", full
stop. That was measured only down to 18 cards and is wrong at the small end** —
at `D = 6` sampling 16 reveals collides once, and at `D = 3` there are only 6
possible reveals, so 16 samples give 5 distinct and 11 collisions. The deck
reaches both once per cycle by construction.

The resolution is to give the small end systematic coverage rather than to
implement merging: **enumerate (or stratify over) exactly where the outcome space
is small**, which is exactly where collisions occur. On the sampled edge
collisions are genuinely absent — 16-of-16 distinct at every deck size from 9
upward — so `count/K` collapses to `1/K` and each child holds one particle.

⚠ **The "and there is nothing to merge at the small end" half of this was wrong,
and §7.1a is why.** Those counts are of *reveals*. A child that prices a reveal
exactly still aggregates the opponent trajectories underneath it, which share a
viewer information state and differ in hidden detail — the particle case, by the
definition given two paragraphs below. Whichever resolution §7.1a takes, the
small end needs particle children.

Keep a **counter** on the sampled edge anyway. If it fires, the enumeration
threshold is set too low, which is a cheaper fix than building particles.

(Definitions, for the record: *merging* keeps one child at weight `2/K` when two
samples reach the same viewer information state; *particles* handle samples that
share a viewer information state but differ in **hidden** detail — most plausibly
opponents having written different numbers this turn, invisible until the next
boundary — where the child is a belief, not a state, and collapsing it to one
canonical state narrows that belief silently.)

### 7.9 ✅ BUILT: what widening did, and the estimator bug on the way

**§6.1's null is reversed.** Open loop measured mean leaf depth 1.59, *unmoved*
by budget or by prior sharpness, because every boundary crossing drew a key never
seen before and expanded a fresh leaf — budget went into root averaging, not
depth. A finite, revisitable chance edge is the thing that changes it. Measured
over 12 positions at two seats:

| arm | 64 sims | 256 sims | gain |
|---|---|---|---|
| open loop (control) | 1.32 | 1.42 | +0.10 |
| **widening `C = 1`** | 1.77 | **2.14** | **+0.37** |
| widening `C = 2` | 1.40 | 1.70 | +0.30 |

So depth now *answers to the budget*, which is the property §6.1 said was
missing. `C = 2` is worse than `C = 1`, as `K**H ≲ N` predicts: a wider edge is a
shallower tree at fixed budget.

**Two counters, and confusing them is a real bug.** `edge_visits` counts
**traversals** and drives the widening cap; `Outcome.count` counts **fresh
draws** from the real transition and drives the weights.

⚠ **The first implementation used one counter for both, and it was a Pólya
urn.** Sampling a child in proportion to a count that the sampling itself
increments reinforces whichever outcome arrived first, so the weights converge to
a random limit rather than to the truth. It was visible in the output: an edge
whose outcomes were near-unique reveals — which must be uniform — came out at
`[0.045, 0.045, 0.091, 0.091, 0.727]`. With the counters separated the same edge
reads `[0.2, 0.2, 0.2, 0.2, 0.2]`, which is §7.1's `count/K` collapsing to `1/K`,
off 5 fresh draws and 23 traversals.

⚠ **RETRACTED — "at a three-card deck it finds the support and stops".** The
measurement behind it (6 outcomes from 6 fresh draws over 36 traversals) was the
**cap binding at six**, not the support being exhausted, and six is the count of
*reveals* while a chance edge holds whole transitions. With the closure defect
fixed, the same edge shows **20** outcomes.

The six-reveal claim is true and is now tested where it holds: on the boundary
sampler, where 400 samples at `D = 3` give exactly **6** distinct
`BoundaryOutcome.draws`, near-uniformly
(`test_a_three_card_deck_has_exactly_six_reveals`).

**What the shape of the tree actually looks like**, 512 simulations:

- 89% of *all* edges carry one outcome — but among edges traversed 8+ times only
  21% do. PUCT concentrates its traversals on a few root actions, and those deep
  repeated paths are exactly the ones that reach a turn boundary.
- 558 of 976 traversals were **reuses**: resumed from a particle, with no
  opponent evaluated and no card drawn.

⚠ **A closure defect, found in review and fixed.** A within-turn edge has support
one, and the first implementation left it to the cap to notice — which a cap
cannot do, since `len(outcomes)` stops at 1 while `ceil(C·n^α)` grows. Measured:
support-one edges reused **6 of 77** traversals, 7.8%, while this document
claimed every later descent resumed from a particle. Determinism is now
**proven** rather than inferred — a transition that changes neither the turn nor
the game's end consumed no randomness — and the same edges reuse **97.4%**
(111 of 114). Multi-outcome edges were unaffected (67% before, 64% after).

⚠ **Particles are reservoir-sampled, not first-come.** Keeping the first
`max_particles` and dropping the rest is unbiased in expectation but freezes the
conditional belief on whichever determinizations arrived first; the collection
never improves however long the edge is searched. `max_particles = 4` is
**unmeasured** and belongs in the bakeoff alongside `C`.

⚠ **What reuse costs, and why the control arm stays.** Resuming from a particle
re-uses that transition's randomness, including the determinization behind it.
That is the intent (§7.3), but it means a widened search explores fewer distinct
futures per simulation than the open-loop one. Whether the depth is worth the
diversity is a **strength** question, not a throughput one — it belongs in §11's
bakeoff as equal-wall-clock arms, and `chance_widening=None` is the control.

### 7.8 ✅ DECIDED: when and how much Dirichlet noise

**What it is for.** The search's move comes from visit counts, and visits follow
the prior. A confidently-wrong prior is never searched against, so the action
never enters the training data, so the prior stays wrong. Dirichlet noise breaks
that loop by perturbing the **root** prior:
`prior ← (1−w)·prior + w·Dirichlet(α)`, `w = 0.25`.

**Root only, self-play only.** At an internal node it would corrupt the search's
own estimates rather than diversify the data; in the arena it would measure a
handicapped player.

#### When: every decision, not once per turn

⚠ **Measured, 53.7% of all search roots are *later* in a turn, not the first**
(4,193 of 7,807), and they average 22.6 legal actions against 49.6 for the
turn's first decision. Noising once per turn would leave over half the emitted
policy targets with no exploration at all, and they are not trivial nodes — the
surveyor averages 27 actions and bis 6.7. So noise applies at **every** decision
root that has two or more legal macros.

The cost of that choice is real and is the second half of this decision:
re-rooting hands a noised root visits gathered under the *un-noised* prior, so
those visits are sound as value estimates and biased as a policy target.

#### How much, part 1: `α` scales with the width of the root

⚠ **A single α is wrong for this game**, and §4's width table is why: a root
ranges from **2** actions (`ASK_RESHUFFLE`, `CHOOSE_PLAN`) to **331**
(`CHOOSE_CARDS`), mean 35.1. One constant either drowns the narrow nodes or does
nothing to the wide ones.

So `α = concentration / len(legal actions)`, with **`concentration = 10.0`**.
That is not a guess — it reproduces AlphaZero's own published constants, which
are all ≈ 10 / branching factor: Go 0.03 at ~250 moves, chess 0.3 at ~35, shogi
0.15 at ~70. Here it gives:

| root | mean legal | `α` |
|---|---|---|
| `CHOOSE_CARDS` | 49.7 | 0.20 |
| `ACTION_SURVEYOR` | 27.0 | 0.37 |
| `ROUNDABOUT_PLACE` | 16.7 | 0.60 |
| `ACTION_BIS` | 6.7 | 1.5 |
| `ACTION_ESTATE` | 5.2 | 1.9 |
| `CHOOSE_PLAN` / `ASK_RESHUFFLE` | 2.0 | 5.0 |

`SearchConfig.dirichlet_alpha` keeps the absolute form as an escape hatch, since
it is what KD (0.3) and 7WD (1.8) use; `dirichlet_concentration` overrides it.

#### How much, part 2: how many simulations must *observe* it

⚠ **At zero fresh simulations the noise is provably inert** — prior perturbed,
nothing selecting against it, policy target bit-identical to the un-noised
search. That was a real defect, found in review and fixed.

`noise_fresh_fraction` is the floor, as a **fraction of the budget** rather than
a count, for the same reason `K` is a schedule (§7.6): one checkpoint searches at
different budgets in self-play, arena and analysis.

- **`1.0` is the shipped default** — a noised root pays for a fresh search, which
  is what plain AlphaZero does, and it is the setting that cannot be wrong.
- **`0.25` is the recommendation for S2**, because 1.0 forfeits re-rooting's
  saving whenever noise is on and that is **47.7% of the budget** against an
  S0-shaped prior (§4).

⚠ **0.25 is a starting point, not a derived constant**, and it is the one number
in this section with no argument behind it. The quantity that matters is whether
the boosted actions actually get visited, which depends on `c_puct`, on the
Q-gaps at the root and on how much of the budget re-rooting supplied — none of
which are known before S2 runs. **Measure it**: the readout is the share of
root-visit mass landing on actions the noise boosted, and it belongs next to the
S1 arena numbers.

### 7.7 Root allocation: Gumbel first, PUCT behind the same interface

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

**Decided: start with Gumbel, build PUCT behind the same interface, and switch
when the model plateaus and simulation counts rise.** That matches where each
one's advantage lies — Gumbel's improvement guarantee binds hardest at small `n`,
and PUCT is what the BGA advisor will most likely serve.

⚠ **The switch changes the training target, not just the search.** Gumbel trains
the policy head toward its completed-`Q` improved policy; PUCT trains it toward
raw visit counts. Both are valid improvement operators, so the switch should be
safe — but it is a discontinuity in the target distribution and should be
measured across it (paired arena either side, plus policy-target entropy) rather
than assumed harmless.

⚠ **Sequencing.** Sequential halving allocates a fixed budget across root actions
in rounds, which interacts with a stochastic transition beneath each action.
Compatible, but another moving part — land the chance node first, then Gumbel.

---

## 8. ⚠ Fixed `K` multiplies inference cost by `K`

`NetEvaluator._forward` evaluates **one** state per call. Materialising `K`
children at every new chance node turns 128 simulations into up to **128 × K**
leaf evaluations — ~2,048 at `K = 16` — which would make any bakeoff at "equal
simulations" meaningless.

⚠ **And a leaf is not the only cost.** Because the retained object is the whole
root-to-root transition (§7.1), each sample also pays for the **opponents' policy
evaluations** — which are network calls, per §7.6, not cheap bot rollouts. The
right per-turn figure here is the **unpruned** 2.42 of §4, not the 2.14: `_advance`
walks an opponent through every decision it has, forced ones included, and neither
§5.1 pruning nor the forced-node collapse touches that path. So at four seats one
`K = 16` expansion approaches

```
16 × (1 leaf + 3 opponents × 2.42 decisions) ≈ 132 evaluated positions
```

before any deeper search happens.

⚠ **A saving is still sitting in `_advance`.** An opponent decision with one
legal action is a network call that cannot change anything, and measured, 13% of
opponent decisions are exactly that (2.42 → 2.14 in §4). Applying the forced
collapse there is worth roughly 13% of the opponent term above. Not taken yet;
unlike the batching below it is a *semantic* change to the opponent model, so it
wants its own measurement.

### 8a. What step 6 actually built, and what it is worth

**The seam, not the tuning.** `MCTS.search_gen` suspends at every network
request and `run_searches` decides when and with what else they are computed. So
cross-game batching is a driver; nothing in the descent knows whether it is
running alone or in a wave. That is the whole deliverable — the tuning comes
after a working self-play loop, and it will not need this rewritten.

⚠ **Both kinds of request must suspend, and this is the finding.** An earlier
version yielded only at leaves. Measured on a 32-search wave: leaves batched
*perfectly* at 32.0 rows per call — and opponent policy sampling was still
**53% of all rows and 97% of all calls**, every one of them a batch of one,
because `_advance` called the network directly. Batching the leaves alone bought
1.24×.

⚠ **And both kinds must share a *call*.** A leaf wants `(priors, value)` and an
opponent wants priors alone, but both read the same heads of the same forward.
Splitting the wave by kind measured a mean batch of **12.1** where 32 searches
were live; answering them together gives **22.4**.

Measured, 32 concurrent searches at 64 simulations, two seats, CPU:

| driver | calls | mean batch | leaves/s | speedup |
|---|---|---|---|---|
| one at a time | 4,430 | 1.0 | 360 | — |
| wave, `max_batch=8` | 608 | 7.3 | 541 | 1.50× |
| wave, `max_batch=16` | 340 | 13.0 | 562 | 1.56× |
| wave, `max_batch=32` | 198 | 22.4 | 588 | **1.63×** |
| wave, `max_batch=64` | 198 | 22.4 | 591 | 1.64× |

⚠ **Read those two columns together, because they disagree.** Pooling the kinds
took mean batch width from 12.1 to 22.4 — **+85% on the diagnostic** — and
throughput from 1.58× to 1.64×, **+4%**. `THROUGHPUT_LEVERS.md` §4.7 names this
exactly: batch width is not throughput, and a win banked on the first number
mostly is not there. `max_batch` is a ceiling, not a target; 32 is the shipped
default because the fixed per-call cost measured worth ~19 rows.

**Why it stops at 1.64×.** Fixed cost per forward is 0.677 ms and marginal cost
per row 0.037 ms, so batching removes almost all of the forward — but the
forward is only ~30% of search wall clock. `encode_state` is another ~30% and is
per-row Python that **does not batch away**; it becomes the larger share exactly
when batching starts working (§4.2). The next throughput lever is the encoder,
not a wider batch.

⚠ **Class A, verified rather than assumed.** A batched forward is not
bit-identical to a single one — float reductions run in a different order,
measured at ~1e-7 on priors and values. Too small to matter as a value, and able
in principle to flip a comparison tied to seven digits, so the check is on
discrete outputs: `trajectory_fingerprint` over actions and visit counts, never
float targets (§A). 32 of 32 searches came out identical.

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

⚠ **Enumerating reveals alone is not an exact root-value oracle** (§7.1a). It
fixes the chance layer and leaves the opponents stochastic. To be an oracle it
must additionally either **fix** opponent randomness — which makes it exact for a
different game, still useful as a *ranking* check — or **integrate** it, which
means many opponent samples per enumerated reveal and a stated Monte Carlo error
bar on the "exact" value. Say which, in the bakeoff, or the comparison measures
the opponent sampler rather than the chance approximation.

---

## 10. Rejected and deferred alternatives

| design | verdict |
|---|---|
| **Bundle the whole turn** | Rejected — §5. Up to 71,334 sequences per turn to save 1.14 network calls. |
| **Sort the stacks** (KD's `sorted(deck[:4])`) | Rejected. KD's sort is meaningful because domino number *is* next-round pick order, so the slot encodes tempo whatever tile it carries. Sorting Welcome To's stacks gives "lowest of three random draws" — 1 in one determinization, 13 in another. No stable quantity. |
| **Merge on `(number, effect, box)` + availability counts** | Experimental arm. Invariant, and availability counts are well founded (§13, Cowling et al.), but **context-abstracted**: one `Q` averaged over what the other two offers are, what next-turn effects were exposed, and the race state. Also lossy — a 7 printed and a 7 made with TEMP consume different cards. |
| **Progressive widening on chance** | ⚠ **Promoted — it is now the recommended shape of the chance edge itself (§7.1a C), not an alternative to it.** Deferred twice before: once for a bad reason ("sharper priors do not deepen the tree" is measured and true and does **not** imply nothing structural does), then behind fixed `K` as "easier to reason about and to batch". The second reason weakened on inspection: PW materialises children *lazily*, which is what §12 step 7 wanted anyway, and it prices the transition empirically, which is what §7.1a needs (§13, Couëtoux & Doghmen). |
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

1. ✅ **Search-only dominance pruning** — `macro_codec.search_legal_macros` /
   `search_legal_mask`, called only from `mcts.py`. `legal_macros`, `legal_mask`,
   `GameState.legal_actions()` and every `datagen` path are unchanged, and there
   are tests asserting so rather than trusting it. Measured: the 317 park and
   pool nodes disappear completely; the estate node narrows to ≤ 6 but **never**
   collapses (0 of 320) (§4). All four dominated passes ship pruned. ⚠
   `PASS_ROUNDABOUT` sits behind `SearchConfig.prune_roundabout_pass` (on)
   because it is the one that interacts with the **bootstrap** prior — **read
   roundabouts per game off the S0 checkpoint before trusting it** (§5.1a).
2. ✅ **Collapse forced internal nodes** inside simulations, not only at the
   external root — `MCTS._collapse_forced`, guarded to stop at the turn
   boundary so that two reveals can never end up behind one key. With step 1 it
   takes a turn from **2.45 to 2.14 network decisions** (§4). The same saving is
   still available in `_advance` for opponent sampling and is deliberately not
   taken yet (§8).
3. ✅ **Deterministic within-turn re-rooting** — `MCTS._retain` /
   `_take_retained`, verified against a full position identity (`_position_key`)
   so a subtree from another line or another game cannot be reused by accident.
   Preserves **8.7% of the budget under flat priors and 47.7% under an
   S0-shaped one** (§4). Discarded at the turn boundary, with a test that it is.
4. ✅ **Engine-owned boundary transition** (§6.3) —
   `prepare_turn_boundary` / `sample_boundary_outcome` / `apply_boundary_outcome`
   in `game.py`, a test for each of the four cases plus the generic mid-draw one,
   and an equivalence test that the triple *is* the engine's own path on all
   three reveal cases. Established on the way, and load-bearing for step 7:
   **which case fires is deterministic given the afterstate; only which cards is
   chance**, so a chance node's support has a known length (3, or 6 on a queued
   reshuffle). Nothing is wired into `mcts.py` — that is step 7.
5. ✅ **Viewer information-state key** — `mcts.information_key(state, viewer)`,
   and `_advance` now returns it. §12.1's checklist is implemented item by item
   and tested item by item, including its "test that matters". Measured neutral
   to adopt (0 of 3,840 edges change) and ~31 µs per simulation step.
6. ✅ **Batched evaluator** — the *seam*, not the tuning (§8a). The search
   suspends at every network request (`MCTS.search_gen`, `Ask`), so batching is
   a **driver** (`run_searches`) rather than a rewrite of the descent. Measured
   **1.64×** with 32 concurrent searches, 4,430 calls → 198, and 32/32 identical
   trajectories. The "successor batches" half of this item is moot: §7.1a's
   progressive widening materialises children one at a time, so there is no `K`
   to batch.
7. ✅ **The chance edge**, as §7.1a resolution C: progressive widening with
   empirical `count/draws` weights, children merged by viewer information state,
   and particle collections. Behind `SearchConfig.chance_widening` with the
   open-loop version kept as the control arm, which allocates nothing. Lazy by
   construction — children arrive one at a time, so the eager `K`
   materialisation this item warned about never exists. **Reverses §6.1's
   depth null: see §7.9.**
8. **Run S0, then the bakeoff** (§11).
9. **Gumbel root allocation** (§7.7) — the intended default for self-play, with
   PUCT retained behind the same interface for the later switch.
10. **Only then:** the afterstate head. (Progressive widening has moved *into*
    step 7 — see §7.1a C.)

**Steps 1–7 are done**, and with them everything this document was written to
decide. The chance boundary was the blocker and §7.1a settled it; §7.9 records
what the build measured, including the estimator bug it found.

⚠ **Nothing after this point is settled by argument.** Step 8 is the run and the
bakeoff, and the two open questions are both *strength* questions that only
equal-wall-clock arms can answer: whether widening's depth is worth the
determinization diversity it spends (`chance_widening=None` is the control), and
what `noise_fresh_fraction` should be (§7.8). Neither is a throughput knob and
neither should be read off games per second.

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

✅ **Built:** `mcts.information_key`, with that test as
`test_the_key_hides_an_opponents_live_sheet_and_shows_its_public_snapshot` and
one per exclusion besides — future deck order, the table-wide
`reshuffle_next_turn` against the viewer's own vote, printed types against
physical ids, and deck/discard composition. Plus the other direction, which the
checklist does not state and which matters just as much under §7.1a: **distinct
positions must not share a key**, or the empirical weights are biased. Verified
over 300+ positions with no collisions.

⚠ **One exclusion the checklist misses:** `ctx` — the acting seat's slot, number
and last house — is *this turn's hidden write*, so it belongs in the key only
when the viewer **is** the actor. In the search that always holds, because a
child is keyed only where the root player is to act; it is excluded anyway,
because a key that is only safe when its caller behaves is not safe.

⚠ **Do not confuse it with `mcts._position_key`.** They look alike and are
opposites: `_position_key` compares two of the caller's own *real* states to
guard re-rooting, so it reads hidden fields deliberately and totality is its
correctness condition; `information_key` labels *determinizations*, so reading a
hidden field is a clairvoyance bug.

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
