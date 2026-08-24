# Welcome To... — the self-play training loop

**Status:** plan of record for stages S0–S3. Supersedes Phases 2–3 of
`PROJECT_PLAN.md`. `PROJECT_PLAN.md` remains correct on the engine, the network
design, and the metric set.

**Scope, fixed:**

- **Advanced variant throughout** — `GameConfig(advanced=True)`.
- **Two to four seats throughout.** Every stage of this plan, including data
  capture, search validation and evaluation, runs multiplayer.
- **One-seat play is out of scope and is not to be reintroduced.** A one-seat
  game silently switches scoring rules — `TEMP_SOLO_THRESHOLD` / `TEMP_SOLO_SCORE`
  replace the 7/4/1 ranking, and every City Plan pays its first-place value — so
  any single-seat corpus is training on a different game. It also costs no less
  than a two-seat game (§1). There is no stage of this plan it belongs in.

---

## 1. Why multiplayer throughout, including the cheap early stages

The tempting shortcut is to bootstrap on one seat because it looks cheaper and
because most of the game is played on your own sheet. Both halves of that are
wrong.

### It is not cheaper

Training cost is not paid per game. It is paid per **learner decision searched** —
MCTS simulations × decisions where the learner is to move — and that quantity is
flat across seat counts:

| seats | games/s (greedy) | turns | decisions, seat 0 | cost per learner trajectory |
|---|---|---|---|---|
| 2 | 5.00 | 30.6 | 100.6 | 1.000× |
| 3 | 3.19 | 30.2 | 98.7 | 0.980× |
| 4 | 2.06 | 29.9 | 98.4 | 0.977× |

(40 paired seeds, advanced, GreedyBot in every seat.)

Put cheap GreedyBots in the non-learner seats and the learner searches the same
~100 decisions either way; the engine cost of the extra seats is noise next to a
batch of network evaluations. Seat count is therefore free to choose on merit.

⚠ **Corrected 2026-08-21.** An earlier version of this paragraph said a four-seat
game "yields four trajectories". That is true **only if all four seats are
learners running full search** — which is not what S2 specifies. S2 puts one
learner in seat 0 and draws the rest from the pool, so a game yields **one
searched trajectory**; the other seats produce bot moves whose policy targets
would be bot policies, not search visit distributions. The table above is the
one-learner regime and its numbers are correct; the "four trajectories" claim
described a different one.

**Multi-seat learning is a legitimate lever, with two caveats**, and is recorded
here as a named option rather than left implicit:

1. **The trajectories are correlated.** All seats see the *same three stacks* —
   the shared card stream that causes symmetric-self-play collapse in the first
   place. Four trajectories from one game share their entire chance history, so
   the effective sample size is meaningfully below 4×.
2. **It costs the opponent pool.** S2 is learner-vs-ramped-pool specifically
   because a pool structurally prevents collapse and supplies curriculum.
   All-seats-learner is symmetric self-play again, with `identical_games` as the
   canary to watch.

Revisit if data volume becomes the binding constraint. Note that per-seat encoding
is *already* symmetric — `encode_state(state, p)` gives seat `p` the full own-sheet
treatment — so nothing in the encoder blocks it.

### Four mechanics do not exist without opponents

These are not mispriced without an opponent — they are absent or inverted, and no
value-head rescaling recovers them:

| mechanic | what an opponent supplies | size |
|---|---|---|
| **City Plans** | the first finisher is paid roughly double | mean premium **4.27 points per plan** (min 2, max 7) across the 37 plans; three plans on the table, so **~13 points** of a ~50-point game |
| **Temp agency** | pure rank, 7/4/1, and **zero temps scores 0** | the marginal value of the *first* temp is 4–7 and of the rest ≈ 0 — see below |
| **Game-end tempo** | an opponent's third refusal can end your game early | `permits` is the single best low-noise signature of capacity destruction |
| **Reshuffle race** | the first plan of the game uniquely grants the deck-reshuffle choice | the one decision a counting model can *compute* where humans reflex — a plausible edge over strong humans |

**Temp is the sharpest of the four.** `Temp::computeCounters` builds its counter
map from scribbles, so a player who never hired a temp has *no entry* and never
enters the ordering: they score 0 rather than inheriting second place. Rank is
computed only among players who hired at least one, and ties share the better
value. So the first temp box is worth 4 points (7 if no opponent has one) and
every temp after it is worth nothing unless it flips a rank. That shape only
exists with opponents on the table.

### Bis and roundabouts are priced through the plans

`BIS_SCORES` and `ROUNDABOUT_SCORES` are penalties, and it is tempting to read
them as self-contained trades. They are not: the reward they buy is largely
plan-mediated. `FiveBisPlan` scores *only* bis houses; every `EstatePlan` needs
exact estate sizes, which is what a bis is for; `CompleteStreetPlan` and
`FullStreetPlan` need streets finished, which is what capacity repair is for.
Flatten the plan race and you flatten the reward while the penalty stays fixed.

**Neither penalty is forced.** `passAction()` is generic, so a player takes a BIS
card for its *number* and declines the bis write. The penalty is entirely opt-in,
which means the failure mode is not a visible avoidance of bis cards — it is a
silent under-use of the bis write, which looks fine on score. **Log
bis-writes-per-game from the first iteration.**

### The encoder wants opponents too

`encoder.py` reserves a seat axis of `MAX_SEATS = 4`; at two seats, half of
`sheet_planes` and `sheet_scalars` is padding, flagged by `seat_valid = 0`.

The table below was measured on the **pre-restructure** encoder, where the
opponent blocks were `plan_race` (24), `reshuffle_race` (4) and `opponents` (27)
inside a single flat vector. The shape has changed and these have not been
re-measured; the argument they support has not changed either, and under full
symmetry the padding is a larger fraction of the input, not a smaller one:

| seats | all-zero planes (of 18) | non-zero flat features (of 446) |
|---|---|---|
| 2 | 8 | 123 |
| 4 | **5** | 129 |

Four seats is the configuration the representation was built for. Training on
inputs where a block is constant zero teaches first-layer weights that have to be
unlearned when it comes alive, so the mixture should keep every block live from
the first gradient step.

---

## 2. Seat counts: 2–4, mixed, weighted 60/30/10

**DECIDED: target 2–4 seats, sampled per game at 60% / 30% / 10%.** BGA's
`gameinfos.inc.php` declares `'players' => [1..12]` with
`suggest_player_number => 4`, and our engine runs correctly at every count tested
— but `MAX_OPPONENTS = 3` means the encoder individually models three opponents,
which is *exactly* 2–4 seats. No encoder work is needed and nothing is dropped or
dead.

**The weights follow the deployment target, not throughput** (revised
2026-08-21). BGA arena is **two players, base board, advanced rules**, and has
historically been 2–3 players. That is the distribution the advisor will actually
face, so 2p carries the mixture.

The tempting throughput argument for 2p does *not* hold and should not be cited:
per **learner trajectory** the seat counts cost 1.000 / 0.980 / 0.977, i.e. flat
within 2%. A 4p game costs 2.4× per game, and under S2 it still yields **one
searched trajectory** — see §1, which corrects the "four trajectories" reading that
also stood in this paragraph. Seat count is free to choose on merit; merit says
match the target.

What the two minority slices buy:

- **30% at three seats** is not optional. `TEMP_RANK_SCORES = (7,4,1)` only ever
  reaches ranks 1 and 2 at two seats — the third-place value of 1 never fires, so
  a 2p-only corpus cannot train that case at all.
- **10% at four seats** exists solely to keep the third opponent's block trained
  (2 spatial planes, its `plan_race` entries, 9 flat features). Train at 2–3 only
  and those weights stay at initialisation while receiving live signal whenever
  the advisor is pointed at a 4p table — it would run, but nothing about the
  output would be principled.

**Sample the seat count per game** rather than training at two and generalising
later:

1. **Temp has no structure below three seats.** `TEMP_RANK_SCORES = (7,4,1)`
   needs three rankable players. A two-seat-only curriculum leaves the temp head
   learning a degenerate two-way split.
2. **It is nearly free** — 1.000 / 0.980 / 0.977 per learner trajectory.
3. **Most of the game is seat-count-invariant** placement skill feeding one
   shared trunk. What varies is race intensity and temp depth, and the encoder
   already carries seat count as a feature so the net can condition on it.

Mixing also avoids a distribution shift that would otherwise land exactly when
the policy has sharpened — the worst moment for one.

If 5+ tables ever matter, the move is a fixed-size *aggregate* opponent block
(best/median opponent progress per plan, temp-count distribution) populated at any
seat count, keeping individual modelling for three. Raising `MAX_OPPONENTS` is the
option to avoid: it costs 2 planes per extra opponent and leaves them dead at
small tables.

---

## 3. The ladder

### S0 — Bootstrap, no search *(supersedes M1)*

Clone GreedyBot into the network to prove `encoder → net → loss → checkpoint`
against a known reference before MCTS can hide a bug.

Capture the corpus with **mixed 2/3/4 seats at 60/30/10** (§2), which costs the
same per trajectory as any other configuration and makes every opponent and race
block non-zero from the first gradient step.

**Gate:** policy top-1 agreement with greedy ≥ 60% on held-out games; the net
playing greedily off its own policy, with no search, scores within 2 points of
greedy on a paired seed set; the `permits` head beats predict-the-mean.

`houses` used to gate alongside `permits` and no longer does. `AUX_TARGETS_SPEC.md`
§9.3 makes it extension-tier, so it may not survive to S1 — and a gate must not
depend on a target that might be deleted. It is still *reported*: `train.evaluate`
returns R² for every head, so dropping it from the gate costs no information.

Implemented in `network.py` (shared sheet encoder → trunk → per-seat + global
heads, 3.94M parameters) and `train.py` (`python -m games.welcome_to.train`),
which reports all three numbers and exits non-zero if any fails.

**S0 is built and now speaks the frozen 684-macro vocabulary** — `macro_codec`
landed first, as planned, so the top-1 number this gate produces is measured in
the representation S1 actually uses. The corpus did not need recapturing:
trajectories are stored as primitives and `datagen.replay` does the collapse.

One consequence for reading the gate: a decision is now a *macro*, so the top-1
denominator is 28% smaller than it was and each label is harder — one choice
among up to ~500 combination-and-placement pairs rather than one among three
slots followed by one among thirteen boxes. The ≥ 60% threshold was written
against primitives and has not been re-derived for macros; treat the first
measurement as calibration.

### ⚠ S0 RUN 2026-08-23 — the ≥ 60% gate is UNREACHABLE, and is withdrawn

5,000 games, 4 epochs, 4.03M parameters, 78 minutes.

| gate | result | |
|---|---|---|
| policy top-1 | **0.475** | ≥ 0.60 — FAIL |
| paired score vs greedy | **−29.65** (21.83 vs 51.48), stderr 2.03 | ±2 — FAIL |
| permits R² | **0.089** | > 0 — **PASS** |

**The pipeline works.** `encoder → net → loss → checkpoint` ran clean, score R²
reached 0.200, and the race-timing heads learned (`turns_to_plan_0` R² +0.75).
That was S0's stated purpose and it succeeded.

⚠ **But ≥ 60% top-1 cannot be reached by any network, ever.** GreedyBot picks
**uniformly among its tied-best actions**, so a deterministic cloner matches its
particular draw with probability `1/|ties|`. Measured over 3,758 non-forced
decisions, `E[1/|ties|]` — the hard ceiling for *any* cloner — is **0.562**:

| phase | ceiling | share of decisions with ties |
|---|---|---|
| `CHOOSE_CARDS` | **0.354** | **94%** |
| `ACTION_ESTATE` | 0.480 | 68% |
| `ACTION_SURVEYOR` | 0.568 | 56% |
| `ROUNDABOUT_PLACE`, `ACTION_PARK`, `ACTION_POOL` | 1.000 | 0% |

So 0.475 is **85% of the achievable maximum**, not 79% of a reachable target.
The threshold was written for the primitive vocabulary, where the tie structure
was different, and it did not survive the move to macros.

**Replacement metric, proposed:** *the fraction of decisions where the net's pick
lies **inside** the teacher's tied-best set.* That is the quantity the old
threshold was reaching for, it has a ceiling of 1.0, and it does not punish a
cloner for the teacher's coin flips. Measured on this checkpoint: **68.5%**
overall — `CHOOSE_CARDS` 76%, `ACTION_ESTATE` 92%, `ACTION_SURVEYOR` 41%.

**The score gate is legitimate and the failure is real.** A perfect cloner
sampling from the tie set would match greedy; ours is outside the tie set ~31% of
the time. Verified it is not an argmax-versus-sampling artefact — sampling from
the policy gives −32.07 and `T = 0.5` gives −31.55, both within noise of argmax's
−30.75, so the deterministic tie-break is **not** the mechanism.

**Where the points go:** ~24% of `CHOOSE_CARDS` picks fall outside greedy's
tied-best set, and in this game a bad write is catastrophic rather than marginal
— `bots.py`: *"a single careless write destroys a whole street: numbers must
ascend strictly, so putting a 15 in the first box takes its remaining capacity
from ten houses to two."*

**The lever is corpus size, not epochs.** Training loss was still falling at
epoch 3 (2.1219 → 2.0408 → 1.9960 → 1.9575; policy 1.7078 → 1.6025) and the
corpus is replayed fresh every epoch, so more epochs re-fit the same 5,000 games
rather than seeing new ones.

⚠ **Trap, for whoever runs this next:** `--out` defaults to
`runs/welcome_to_s0`, so a 60-game smoke run and a 5,000-game real run write the
same two files and are indistinguishable afterwards. Use a run-id'd directory.

**The paired comparison replaces one seat, not the whole table.** The first
implementation played an all-net game against an all-greedy game at the same
seed, and that is not a controlled substitution: every seat of an all-net table
sees the same stacks and runs the same deterministic argmax, so the sheets
converge on each other (measured 0.34 mean divergence against greedy's 0.80).
Correlated sheets complete plans on the same turn and so *share* first-place plan
values instead of racing for them, and they tie on the temp-agency rank — two
scoring rules worth 6–14 and 7/4/1 points moving for a reason unrelated to
placement skill. The gate now puts the net in seat *k* against GreedyBots on the
same RNG streams, pairs against the same game with a GreedyBot in seat *k*,
rotates *k*, and averages the per-game delta. It also reports
`score_gap_stderr`, because a ±2 point threshold means nothing without knowing
what 2 points is worth at the sample size used.

Accept that greedy is race-blind — it completes ~0.42 plans per game. The
bootstrap produces a placement-competent, race-blind policy, which is fine.

### S1 — Search *(supersedes M3)*

Build `macro_codec` **first**, then `mcts.py`. Validate at **two seats against
GreedyBot**, the cheapest configuration that still has all four race mechanics
live.

**`macro_codec` (684 indices, frozen in `ENCODER_V2_SPEC.md` §10.6).** It leads
because everything downstream is measured in its vocabulary: the S0 policy head
is 684 wide, `datagen.replay` collapses each `CHOOSE_STACK → (WRITE |
PERMIT_REFUSAL)` pair into one macro label, and there are no `WRITE_NUMBER`
network calls. It sits with search rather than with S0 because the legal-mask
contract — a macro index is legal iff its **full primitive sequence** is legal
end to end, never a per-step mask intersection — is the same enumeration the
search needs to build edges. Building it twice would be building it two ways.

`network.py` currently imports `NUM_ACTIONS` from `action_codec`; it becomes the
macro width, and the S0 checkpoint from before the change is not loadable
across it. There is no migration to write — S0 is cheap to re-run.

#### The root-player contract — binding

The old wording here ("no minimax negation; back up the learner's own score") was
**underspecified in a way that produces a real bug**, and it has to be replaced by
an explicit contract. `encode_state` defaults to `state.actor`, so a leaf reached
while seat 1 is to move silently yields *seat 1's* value — which cannot be backed
up as seat 0's. The search is a **learner-only semi-MDP**, and all four clauses are
load-bearing:

1. **Tree nodes belong only to the root player `r`.** No node in the tree is ever
   an opponent's decision.
2. **Every other seat is sampled forward** — one policy sample per determinization,
   no nested search — until `r` is to act again. An opponent's turn is a
   *transition*, not a node.
3. **Leaves are evaluated as `encode_state(state, r)`, never `state.actor`.**
4. **The backed-up scalar stays in `r`'s frame** for the whole path. No negation,
   no reframing.

This is legitimate here because Welcome To is near-solitaire: an opponent's
concurrent choice cannot change your legal moves or your immediate scoring, since
the stacks are shared and never consumed. Sampling them is a good approximation of
an expectation, and no equilibrium reasoning is required.

**S3 needs more than this.** Once several seats are learners, a single scalar tree
that changes viewpoint is not a fixable variant — it is the wrong object. Run a
**separate root-player search per acting seat**, or move to a vector-valued Maxⁿ
design. Budget it as real work in S3, not as a flag.

#### Chance must be keyed, not merged

Open loop with `GameState.redeterminize(search_rng)` per simulation. Two things
that a naive action-sequence tree gets wrong:

**Observations belong in the node key.** If nodes are keyed on the action sequence
alone — Kingdomino's `OpenLoopMCTS` shape — then one node aggregates simulations in
which *different next-turn offers were revealed*. Its priors, children and Q then
mix distinct observable states. The sharpest consequence is a **selection bias**: an
action that is only *legal* under favourable reveals accumulates Q solely from
those favourable simulations, so it is systematically overvalued. Key nodes on
`(action sequence, observed chance outcomes)`, or give chance explicit
observation nodes. Chance enters only at turn boundaries, so the branching this
adds is confined and shallow.

**The RNG must advance.** `redeterminize` now *requires* a search RNG for exactly
this reason — see the fix note in §5. Passing the state's own generator, or calling
it without one as the old signature allowed, returns the identical shuffle every
time.

#### ⚠ Measured: keying chance exactly means the tree does not deepen

Built and measured 2026-08-21. "The branching this adds is confined and shallow"
is true of *where* chance enters and **misleading about the effect**. A turn
boundary reveals three fresh cards, so a crossing almost always produces a key
never seen before, and therefore a fresh leaf rather than a revisit.

**Prior sharpness does not change this, and that is worth knowing because the
obvious objection is that it should.** If most simulations are wasted on moves a
trained policy would know are bad, concentrating them ought to buy depth.
Measured over 4 network seeds x 6 positions at 256 simulations, comparing a
uniform prior against a **one-hot** one (the limit of sharpening):

| prior | mean leaf depth | range |
|---|---|---|
| uniform | 1.59 | 1.06 - 2.08 |
| one-hot | 1.59 | 1.00 - 2.01 |

Two of 24 cells differ by more than 0.5. **Because one-hot is the upper bound on
sharpness, this closes the question rather than leaving it open** - no trained
policy can do better than the extreme.

The reason is structural. With a concentrated prior every simulation takes the
same root action, reaches the same child - no chance has been crossed yet, so the
key recurs - and walks the same within-turn chain. Then it hits the turn
boundary, where **every simulation draws a unique observation and terminates as a
fresh leaf**. So depth is about "the root player's own decisions remaining this
turn, plus one", which is 1-2 in most turns. A uniform prior reaches the same
depth because those within-turn chains are deterministic given the action and get
developed after a couple of visits regardless.

So the budget beyond roughly one simulation per root action goes into **root
averaging over fresh leaves**, not into depth. That is what this search is, and
it is genuinely chance-limited.

(Two earlier versions of this note were wrong in opposite directions. The first
claimed depth was capped at 2 regardless of budget, measured at one position. The
second claimed depth was prior-limited and roughly doubled under sharp priors -
measured once, with an **unseeded** network, and not reproducible. The table above
is 24 cells with seeded weights. Measure the extreme case: it bounds the claim.)

**Do not reach for progressive widening.** An earlier version of this note called
it "the fix"; that was written before the following measurement and is withdrawn.

Ask first why keying is needed at all, given that open loop already averages over
chance. Two distinct problems live below a chance node, and only the second
justifies what was built. Measured one turn ahead, same action sequence, 40
determinizations:

| | |
|---|---|
| moves in the union that are only *conditionally* legal | **47%** |
| distinct legal macro sets | 25 of 40 |
| numbers a fixed `macro_write(slot, 0, box)` can mean | **13–15, spanning 1–15** |

The first row is the selection bias §above describes. It is real — and it is the
classic ISMCTS problem, which has a fix that costs no depth: **availability
counts**, i.e. use "how often was this action available at this node" in the UCB
denominator rather than the node's total visits. So row one alone would *not*
justify observation keying.

The third row is what does. A macro index is `(slot, delta, box)`; the move is
`(number, box)`. At the root those coincide because the table is known. One turn
out, "take slot 0, box 3" means writing anything from 1 to 15 — so merging those
simulations averages opposite moves in a game whose entire constraint is
ascending order. No UCB correction repairs that, because the node's identity is
genuinely a lottery.

**Kingdomino is the counterexample, and it is worth being precise about.** Its
`OpenLoopMCTS` keys children on a slot-relative joint index and **no observation
at all**, and its docstring names this very problem — *"at a deep node the
concrete domino in a given pick slot differs across determinizations"*. It merges
them and works. So merging tolerates semantic drift, and "no UCB correction
repairs that" was too strong.

What differs is one line of Kingdomino's engine: `current_row = sorted(deck[:4])`.
**Its draft is sorted**, so slot 0 always means the lowest-numbered domino — the
concrete tile varies, the slot's *ordinal* meaning does not, and merging averages
over things that are alike. Welcome To's three stacks carry no ordering: slot 0
is whichever pile is leftmost, and its number is near-uniform over 1-15.

Which opens a third option, arguably the cleanest: **canonicalise the stack order
by number**, exactly as Kingdomino sorts its draft. Slot index would then mean
"lowest / middle / highest number on offer", merging would be as sound as
Kingdomino's, and depth would come without keying at all. The cost is that a
stack's identity carries its **known next effect** across turns — `next_effects`
is a certainty and a real feature — and sorting scrambles that correspondence.
A genuine trade-off, not a free win, and it would change the frozen vocabulary.

#### The edge identity, and why sorting is not the answer

Kingdomino keys children on a **slot-relative joint index with no observation in
the key** and re-decodes it per simulation (`decode_action(idx, concrete_state)`);
its docstring names this very problem - *"at a deep node the concrete domino in a
given pick slot differs across determinizations"*. So open loop does not need a
post-chance hierarchy. It needs an **invariant action encoding**.

⚠ **Sorting Welcome To's stacks does not provide one, and an earlier version of
this note wrongly proposed it as the analogue of `sorted(deck[:4])`.** Kingdomino's
sort is meaningful because domino number *is* next-round pick order, so the slot
encodes tempo whatever terrain it carries. Sorting three Welcome To stacks gives
"lowest of three random draws" - which could be 1 or 13 in different
determinizations, with different effects attached, merged into one edge. The sort
carries no stable quantity.

**What is invariant here is the action's own semantics.** Measured, one turn
ahead, over 60 determinizations of the same action sequence:

| | distinct values |
|---|---|
| effect triple on offer | **1** - certain, and equal to `next_effects` predicted in advance |
| number triple on offer | 60 |

**The chance is entirely in the numbers; the effects are deterministic one turn
out.** So an edge identified as *write number N, with effect E, into box B* is
invariant, and only its **availability** varies.

That gives the design, and it costs no vocabulary change:

- the policy head keeps emitting the frozen `(slot, delta, box)` 684 at states
  where the table is visible, where the two encodings are in bijection;
- the **tree** keys edges by `(number, effect, box)` and translates per
  simulation - find the slot offering `(number, effect)` in *this*
  determinization, read that macro's prior, apply that macro;
- an edge not offered this simulation is **unavailable**, and availability counts
  go in the UCB denominator instead of the node's total visits (ISMCTS).

Structurally Kingdomino's `decode_action`, with invariance coming from semantics
rather than from a sorted draft. Search-side only: no head resize, no retraining,
and `datagen` already records the number written.

What each keying makes `Q` mean is the whole argument:

| keying | `Q(edge)` means |
|---|---|
| `(slot, delta, box)`, merged | "take pile 1 and box 3, whatever number pile 1 holds" - not a plan |
| `(number, effect, box)`, merged | "if a 7 comes up on the PARK stack, put it in box 3" - a contingent plan |
| observation-keyed (what is built) | correct, but every child is unique, so no depth |

⚠ **The risk to measure before adopting it.** A specific `(number, effect)` is on
offer roughly 1 turn in 15, so individual edges are visited rarely and the whole
thing rests on the availability counts being right. Compare depth *and*
`arena.paired` strength, merged-with-availability against the current
observation-keyed version, at equal budget. Do not assume it wins.

#### ⚠ SUPERSEDED by `SEARCH_SPEC.md` — do not implement from this section

The chance design below was corrected twice after it was written here. **Four of
its statements are now known wrong**: "sample three from the remaining
histogram" (the boundary has four cases and a reshuffle draws six — `SEARCH_SPEC`
§6.3); "reuse opponent samples" (share uniforms, not realised actions — §7.4);
retaining deck order and opponent RNG in a scenario (non-anticipativity /
strategy fusion — §7.3); and "biased by that particular sample" (IID `1/K` is
unbiased across supports; the real hazard is that visit-based stderr inside one
retained tree is falsely tiny — §9).

`SEARCH_SPEC.md` is the spec of record for search. What follows is kept only as
the record of how the design got there.

#### External review, 2026-08-22: sparse chance sampling is the reference design

Reviewed independently, verified against the engine, and it corrects two things
above.

**Correction 1 - the reveal is not "entirely in the numbers".** That is true only
of the *current* offer. A construction card prints its own effect on its number
face, so the card drawn at a boundary supplies a number for **this** turn and an
effect for the **following** one (`next_effects`). Measured, one turn ahead over
60 determinizations of one action sequence:

| | distinct |
|---|---|
| effects in play this turn | **1** (certain) |
| effects for the following turn | **46** |

So a reveal is an ordered triple of printed **card types**, not of numbers. There
are 66 distinct printed types in an 81-card deck (multiplicity 1-2), giving on the
order of 277,000 valid ordered type triples - exhaustive chance expansion is not
available, which the design above already assumed.

This also weakens the `(number, effect, box)` edge proposed above: it is lossy
(a 7 printed vs a 7 made with TEMP consume different cards) and, more seriously,
**context-abstracted** - one `Q` averaged over what the other two offers are, what
next-turn effects were just exposed, and the opponents' race state. Keep it as an
experimental arm, not the reference.

**Correction 2 - the observation key is under-specified.** `MCTS._advance`
returns `tuple(state.table_cards(root))`: raw card **IDs**, with no opponent
sheets and no race state. Measured, the ID granularity costs nothing today - 0
spurious splits in 60 samples, because near-unique triples mask it - which is
precisely why it would bite as soon as chance children are deliberately reused.
15 of the 66 printed types have two physical copies.

**Correction 3 - progressive widening was withdrawn for a bad reason.** "Sharper
priors do not deepen the tree" is measured and true; it does **not** imply that
nothing structural does. Prior sharpening still draws a fresh reveal every
simulation. Progressive widening changes that behaviour by capping stochastic
children and revisiting them. Two different claims; only the first was tested.

**The recommended design.** Search the player's deterministic decisions normally
(the macro vocabulary already removes the artificial choose/write split), then
insert an **explicit sampled chance node at the turn boundary**, on a canonical
pre-reveal afterstate before `_draw_step`. Sample `K` reveals from the exact
remaining histogram, **retain those children for the whole search**, and back up
their arithmetic mean - chance nodes average, they do not PUCT. Start with
`K = 4, 8, 16`. Retained children are revisited *by construction*, which is the
one thing neither exact keying nor prior sharpening can produce. This is sparse
sampling (Kearns/Mansour/Ng) and the setting of Monte Carlo *-Minimax, which was
built for densely stochastic games and evaluated on Can't Stop.

**Use common random scenarios across candidate root actions.** The reveal
distribution does not depend on which house the root player wrote (reshuffle
aside), so candidates should be compared against the *same* sampled scenarios,
which cancels most of the variance in their differences. Opponent samples can be
reused for the same reason: opponents cannot see the root player's concurrent
choice.

⚠ **Two cautions before building it.**

*The bias/variance trade runs the other way at the root.* Drawing a fresh reveal
per simulation - what is built today - makes the root's `Q` an average over ~N
distinct futures: unbiased, high N. Fixed-`K` sparse sampling replaces that with
an average over `K` fixed futures: lower variance, but biased by that sample, and
every simulation past the `K`th reuses it. Sparse sampling is **not** strictly
more accurate; it trades a worse depth-1 expectation for having a depth-2 at all.
Measure root value bias directly, or a depth-1 regression will hide inside a
depth-2 gain. Late-game states with a small remaining deck are the oracle:
enumerate every legal ordered reveal exactly and compare each approximation's
root ranking and value bias against it.

*The scenario must include the opponents, or the child is not well defined.* The
review asks the child key to carry now-public opponent sheets, but opponent
actions are **sampled**, not chance, and the current design deliberately keeps
them out of the key so a node averages over the opponent model. Retaining a
chance child while resampling opponents leaves the child's state undefined. The
coherent reading is the review's own common-random-streams: a scenario `w` is
**deck order and opponent randomness together**, and retained children are
indexed by `w`. Half of this - retained deck, resampled opponents - produces a
plausible-looking tree with incoherent statistics.

**Then, only if sparse chance wins:** distil the expectation into an *afterstate
value head*, `V_after(x) = E_c[V_post(x, c)]`, trained against an average over
several independently sampled immediate reveals, so a turn's search can stop at
the pre-reveal afterstate for one network call. This is Stochastic MuZero's
afterstate factorisation, which is worth borrowing without the rest of it since
the chance distribution here is known exactly. It is a real build - new head, new
target, several reveals sampled per training position - so gate it on the arm
winning first.

**Bakeoff**, at equal wall-clock *and* equal network-evaluation budget: current
(fresh reveal per simulation) | semantic-edge ISMCTS with availability counts |
sparse chance at `K = 4/8/16` | sparse chance plus afterstate head. Report paired
score gap and stderr over 300+ games, strength at 64/128/256 evaluations, mean
turn boundaries crossed, action agreement across search RNGs, root value stderr,
evaluations and wall time per move, and plans / permits / end reasons.

**Sequencing: this waits on S0.** Every arm needs a trained network to compare;
on an untrained one both arms produce noise, which is how two claims in this
document came to be wrong.

**Measure before building any of it.** Once S0 exists, run `arena.paired` at 64,
256 and 1024 simulations. If strength is flat past ~128 the question is closed,
and the compute belongs in games and network quality instead. If it is not flat,
the sorted-stack option is the first one to price - it is the only one of the
three that buys depth rather than working around the lack of it.

(A first version of this note claimed depth was capped at 2 regardless of budget.
That was measured at one position and was wrong; a rare line does reach 4. The
mean is the number that holds.)

Budget per phase, not flat: half of all decisions have two options, and the width
is concentrated in `WRITE_NUMBER` (13.1 mean, 165 max) and `ACTION_SURVEYOR`
(28.5).

**Retired gate (2026-08-24):** do not block S2 on search + S0 beating GreedyBot.
That comparison proves neither plan-race strength nor preserved placement
flexibility, and optimising it would select for the one-ply score objective that
S2 is intended to escape. The completed S0 run remains a pipeline shakedown.

**The paired-seed harness remains useful, but its opponent changed.** Search and
network changes are compared candidate-versus-frozen-incumbent under the S2
margin-versus-best contract. `arena.py`'s GreedyBot arm is retained only as a
mechanical regression diagnostic; it is not a strength or promotion gate.

⚠ **Historical variance measurement.** GreedyBot against GreedyBot at two
seats has a per-game paired-delta standard deviation of about **18 points**, so
the standard error is `18/√n` — **4.1 at 60 games, 1.0 at 330**. This no longer
defines a gate, but it remains a warning that every paired arena conclusion needs
its own reported standard error and adequate sample size.

### S2 — The first self-play loop *(the main stage; trajectory, optimiser and durable generation built 2026-08-24)*

Learner in seat 0 with full search; the other seats drawn from an opponent pool;
seat count sampled per game across 2/3/4 at 60/30/10 (§2).

**Pool composition, ramped rather than switched (revised 2026-08-24):**

1. start with the frozen incumbent checkpoint, sampled independently in every
   opponent seat;
2. add each promoted checkpoint to the pool as it promotes;
3. weight toward recent strong checkpoints while retaining older promoted ones
   for policy diversity.

**Why GreedyBot was removed from S2.** S0 used it to prove the supervised
pipeline, not to define strength. Welcome To defers score; early play is chiefly
about racing to City Plans and preserving future placement flexibility. A
one-ply score bot systematically underprices both, so comparison against it and
continued use as the default S2 curriculum select for the wrong objective. The
incumbent checkpoint is imperfect, but it at least represents the policy being
improved and makes the paired promotion comparison coherent.

Do not replace that mistake with a hand-built plan-seeking bot. If plan learning
stalls, diagnose the plan targets, root exploration, and opponent-pool diversity
directly rather than installing a second handcrafted objective.

**Promotion gate: paired-seed MARGIN VERSUS THE BEST OPPONENT.** Same seeds, same
opponents, candidate versus incumbent, compared with a paired significance test.

⚠ **Pinned 2026-08-21.** This previously read "paired-seed score margin", which is
ambiguous between *candidate score minus incumbent score* and *margin against the
field* — and the first reading gates on a different objective than the search
optimises. The search values denial and game-ending tempo (rank plus a
variance-gated margin term, `AUX_TARGETS_SPEC` §5); raw score does not. A candidate
can raise its own score while raising the **best opponent's** score by more, and a
raw-score gate would promote it.

```
primary   : paired mean of ( own_final_score − best_opponent_final_score )
secondary : paired mean normalised rank  (n−1−rank)/(n−1), ties averaged
diagnostic: raw own score, win rate, plans-completed-first
```

Gate on the primary, require the secondary not to regress, and **never gate on raw
score or win rate**. Score is still the densest signal in a game this
low-interaction, which is why it stays as a diagnostic — but density is not the same
as being the objective.

**Gate to leave S2:** three consecutive significant promotions against the
incumbent under the primary/secondary contract above; `turns_to_plan_*` beats
predict-the-mean on its masked support, with the support count reported; and
`plans_completed` improves. `first_plan` was deliberately dropped in
`AUX_TARGETS_SPEC.md` and must not reappear here as a phantom gate. These are the
direct evidence that race learning is happening rather than being assumed.

#### S2 trajectory implementation

`self_play.py` is the built generation/replay leg:

* learner fixed in seat 0, with one persistent M6 search slot per live game;
* advanced 2/3/4-seat games at 60/30/10, with the small-corpus allocator fixed
  so any run of at least three games exercises every supported seat count;
* complete macro-action games plus sparse `(root actions, integer visits)` only
  at searched learner decisions — forced learner moves and opponent moves do
  not masquerade as MCTS targets;
* M0-F per-search seeds, Python-generated scaled Dirichlet noise, `tau=1` for
  the first ten turns and deterministic choice afterwards, with the temperature
  supplied per root so retained trees survive the schedule change;
* real opponent moves batched by frozen checkpoint and sampled from independent
  per-seat portable streams using that **same opening/late temperature schedule**;
  no GreedyBot fallback;
* replay through the Python rules oracle before a game is admitted to disk,
  checking search legality, final scores, seat-axis targets and visit mass;
* replay-validated games written as immutable atomic shards (25 games by
  default), so a hard process loss discards at most the open shard; restart
  validates the deterministic seed/seat schedule and generates only missing
  seeds;
* a sidecar manifest pins the model hashes, rules/table signature, search
  settings, temperatures and scheduler width. A restart refuses mixed or
  unverifiable evidence instead of silently appending it;
* soft policy cross-entropy in `network.losses`; S0 teacher samples are promoted
  to one-hot distributions, so old bootstrap trajectories remain compatible;
* streaming shuffle-buffer batches via `self_play.iter_batches`, and the metric
  set in §4 emitted on every generation run.

`s2_train.py` is the built optimisation leg:

* random AlphaZero bootstrap, S0 warm-start and S2 resume are all explicit CLI
  paths — S0 is useful prior information, not a prerequisite;
* validation splits on complete trajectories, never adjacent states from one
  game;
* evaluation scores the visit distribution with cross-entropy, KL, entropy and
  tie-aware visit-best agreement; sampled-action top-1 is diagnostic only;
* rank cross-entropy/KL and every auxiliary R² are reported, together with the
  actual mask support for sparse plan-timing targets;
* AdamW moments, model/config provenance, completed epochs and metrics are saved
  in an atomic, resume-capable checkpoint that remains readable by the generic
  network loader.

The search's internal opponent POLICY requests still use the learner checkpoint,
which is the existing single-model opponent proxy; the frozen pool governs the
actual game. Routing simulated opponents through their assigned frozen networks
would be a strength change and needs a separate bakeoff, not a silent scheduler
change.

Replay-window retention and the paired promotion controller remain to be built
on this trajectory boundary. They are deliberately not hidden inside generation
or optimisation: a trajectory is immutable evidence, while promotion changes
which policy produces the next evidence.

### S3 — Raise the symmetric share; HOF, Elo, SPRT

Not a new configuration — a *mixture shift*: raise the fraction of games where
every seat is a searched network rather than a cheap bot, and switch on HOF, Elo
and SPRT gating.

**The collapse hazard is specific to this game and is measured.** Every seat sees
the same three combinations, so one policy playing itself from identical empty
sheets writes identical sheets forever: 100% identical games, zero score spread.
Required guards:

- independent Dirichlet noise **and** an independent sampling stream per seat —
  easy to get wrong in a batched worker where seats share an RNG;
- hold τ = 1 through the first ~8–12 turns, not the first few moves. Divergence is
  self-amplifying, so the entire risk sits in the opening, and it *grows* as the
  policy sharpens — precisely when a conventional schedule has dropped to argmax;
- `training.diversity_report()` logging `identical_games` every iteration as the
  canary. Non-zero is a stop-the-run condition.

S2 already gives the searched learner and every real opponent seat independent
sampling streams, even when the initial candidate and incumbent are the same
checkpoint. S3 raises the risk because every seat becomes a sharpening searched
policy, so its explicit collapse guards remain mandatory.

### S4 — Endgame

Optional, and only if measurement says so. The branching that bites is chance, not
moves. Enumerate over *offers* rather than card identities — many cards produce the
same (number, effect) pair, which collapses most of the branching. If PIMC is
used, use it as an evaluator and not as a policy: it assumes the future is known,
so it overvalues lines needing one specific card and never pays to keep options
open.

---

## 4. Metrics, every iteration

| metric | why |
|---|---|
| paired-seed margin-vs-best vs incumbent | the promotion signal |
| mean `permits` | the direct signature of capacity mismanagement |
| mean houses placed, capacity at turn 10 | the same thing, earlier |
| plans completed per game | the race, and the clearest strength proxy (greedy: 0.42) |
| fraction of games ending on the plans | greedy is at 0; a strong player should not be |
| roundabouts built per game | is the model using its capacity-repair tool? (greedy: ~1.2) |
| **bis writes per game** | the opt-in penalty (§1) — a model quietly declining every bis looks fine on score |
| estates of size 7+ per game | the unsplittable-bis-run trap (§5 #9); should be ~0 |
| temp boxes crossed, temp rank distribution | the mechanic with the steepest first-unit value |
| `identical_games`, `mean_first_divergence_turn` | the collapse canary — **mandatory from S3** |
| mean branching, decisions/game | search cost, and the denominator of every cost estimate here |

---

## 5. Rule edge cases, verified against the BGA PHP

Verified by `tests/test_bga_differential.py`, which transliterates
`Houses::getAvailableLocations`, `Houses::getAvailableLocationsForBis`,
`Surveyor::getAvailableZones` and `RealEstate::getEstates` straight from the PHP
and fuzzes them against `sheet.py` over 13,382 reachable sheets drawn from random
advanced two-seat games. **All four agree everywhere in the corpus.** The named
cases below are pinned as individual tests.

| # | rule | source | our engine |
|---|---|---|---|
| 1 | **A roundabout may go in *any* empty box**, including both street edges — `argRoundabout` passes `null` to `getAvailableHousesForNumber`, so the ascending-order rule does not constrain it at all | `WriteNumberTrait.php:157` | ✅ `available_locations(None)` |
| 2 | **A roundabout is a free double fence** — `buildRoundabout` scribbles `estate-fence` at both `pos-1` and `pos`. Worth two surveyor actions on top of the capacity repair, and it always isolates itself, so it can *close* the estates on both sides | `WriteNumberTrait.php:174-175` | ✅ `build_roundabout` |
| 2b | **A roundabout is a built house for plan purposes** — it goes through `Houses::add` like any other house, so `ExtremitiesPlan` (`!is_null($streets[$x][$y])`) and `FullStreetPlan` both count it. A roundabout on a street end therefore completes Extremities *and* fences, in one move | `ExtremitiesPlan.php:33`, `FullStreetPlan.php` | ✅ `can_be_scored` |
| 3 | At a street edge the off-board fence is silently dropped (`$pos[1]-1 == -1`), not wrapped | same | ✅ range-guarded |
| 4 | **A bis may duplicate *any* number already on the sheet**, not the number just written — `getAvailableNumbersForBis` loops `0..17`. The bis is a second house per turn, decoupled from this turn's write | `Player.php:417-427` | ✅ `bis_candidates` |
| 5 | **The bis penalty is opt-in** — `passAction()` is generic, so a BIS card can be taken purely for its number | `ActionsTrait.php:22-27` | ✅ `A_PASS_BIS` |
| 6 | A bis must be *directly* adjacent (the `$hole` counter) and **cannot cross a fence** | `Houses.php:154-198` | ✅ |
| 7 | A roundabout can never be duplicated by a bis (sentinel resets the carried number) | same | ✅ |
| 8 | **No fence may split a bis pair** — two adjacent equal numbers are unsplittable, so a long bis run permanently welds an estate size | `Surveyor.php:31-32` | ✅ `surveyor_zones` |
| 9 | **An estate of 7+ scores nothing** — `getAssocSizeNumber` counts only `$size < 7`. Combined with #8, an unsplittable bis run is a genuine trap | `RealEstate.php:53` | ✅ `estate_size_counts` |
| 10 | A roundabout inside a fenced segment voids the whole segment (`$full = false`) — defensive only, since #2 means it is always alone | `RealEstate.php:25-27` | ✅ `estates` |
| 11 | Both street ends act as estate boundaries; the leftmost segment counts only if completely full | `RealEstate.php:22-42` | ✅ |

Rules #2, #2b, #4, #5 and #9 carry real strategic weight. #9 in particular is a
trap the model can walk into while its score still looks healthy.

### Houses spent on a City Plan cannot be reused

The mechanism is the `top-fence` scribble, and the split is clean:

| plan kind | consumes houses? | what it scores off |
|---|---|---|
| `EstatePlan` | **yes** — every box of each estate handed over | the estates it names |
| `FullStreetPlan` | **yes** — every box of the street | the street |
| `ExtremitiesPlan` | **yes** — the six end boxes | those six boxes |
| `FiveBisPlan` | no | the bis count |
| `SevenTempPlan` | no | the temp track |
| `DecorativePlan` | no | park / pool completion |
| `CompleteStreetPlan` | no | parks + pools + roundabout |

Exactly those three override `AbstractPlan::validate` to lay top fences; the other
four inherit the base `validate`, which records the scoring and scribbles nothing.
Their `canBeScored` never consults top fences either, so track plans stack freely
with everything else.

Consequences worth surfacing to the model: a `FullStreetPlan` permanently spends a
whole street, so no estate in it can feed an `EstatePlan` afterwards — the estates
still physically exist, they are just no longer *available*. `FullStreetPlan` and
`ExtremitiesPlan` also compete for the same boxes at every street end, so the
order they are taken in matters.

### City Plan validation

All seven supported plan kinds are fuzzed against a transliteration of
`Plans/*.php`, and agree. The detail that makes them agree is easy to get wrong:
**`Zone::getAvailableZones` `break`s after the first free box in each row**, so a
2-D zone returns at most *one* entry per street. `DecorativePlan`'s
`count(Park::getAvailableZones(...)) <= 1` therefore means "at most one street
incomplete" — i.e. two streets finished — and not "at most one park box unbuilt".
Read literally without the `break`, the two readings diverge on any sheet with two
finished streets and a third barely started.

**The random corpus satisfies only three of the seven kinds**, so the fuzz alone
would have been vacuous for the rest:

| kind | satisfied in corpus (13,382 sheets) | covered by |
|---|---|---|
| ESTATE | 777 | fuzz |
| EXTREMITIES | 176 | fuzz + constructed reuse case |
| FULL_STREET | 63 | fuzz |
| FIVE_BIS | **0** | constructed: 5-on-one-street vs 3+2 spread |
| SEVEN_TEMP | **0** | constructed: 6 / 7 boundary |
| DECORATIVE | **0** | constructed: park, pool, and pool&park variants |
| COMPLETE_STREET | **0** | constructed: scattered vs same-street |

Both sides of the differential are asserted on the constructed sheets, not just
ours. Also verified: the temp clamp never leaves `0..17`, the refusal track never
exceeds three and games do end on it.

⚠ **The reshuffle counterfactual was NOT exactly `deck + discard`** — that claim
stood here until 2026-08-21 and was wrong by three cards. The reshuffle resolves at
the *next* turn boundary, and `_begin_turn` runs `_discard_step()` **before**
`_reshuffle_decks()`, so the turn's three aside cards are already in the discard
when `_reform_deck()` fires, while the number cards beside them are discarded
*after* the reform and stay out of it. The pool is
`deck + discard + aside`. Fixed in `deck_knowledge.after_reshuffle_composition`,
with `aside_composition` added and regressions in `tests/test_deck_knowledge.py`.
The old test passed because it called `_reform_deck()` directly, reproducing the
same wrong ordering.

### Temp ranking

Pinned separately, since it is pure rank and the zero case is easy to get wrong: a
player with no temps scores 0 rather than inheriting second place; rank is computed
only among players who hired at least one; ties share the better value
(`array_unique` collapses equal counts into one rank); only the top three ranks pay
anything. Our `temp_scores` already matched — `{c for c in counts if c > 0}` is the
equivalent of `computeCounters` having no entry for a zero-temp player.

The `where('turn', '<', getCurrentTurn())` filter in `computeCounters` is the
simultaneous-play leak filter (`Scribbles.php:36`), not a final-turn exclusion —
temps hired on the last turn do count.

`tests/test_bga_differential.py` is 45 tests and found **no engine defect**.

---

## 6. Throughput: the batched-inference design

**This section exists because the plan had no inference architecture at all**, only
GreedyBot `games/s` figures — which measure the engine, not the thing that will
actually be slow.

### The arithmetic

Roughly **100 searched decisions per learner trajectory**. At a modest 200
simulations that is **~20,000 leaf evaluations per game**, and S3 multiplies it by
the number of searched seats. Batch-1 inference at that volume is not viable at any
network size, so batching is not an optimisation to add later — it is the design.

### Required components

1. ✅ **Cross-game batching and fixed CPU pool.** `RustCloudScheduler` keeps
   resumable searches for every inflight game on a persistent, independently
   sized worker pool. One deterministic global barrier collects both LEAF and
   POLICY rows; batch width is bounded by ready games and `max_batch`, not by
   worker count. `--scheduler-workers` is pinned in the run manifest.
2. **No within-search leaf batching.** The earlier requirement here was wrong.
   One search exposes one suspended request; adding more requires virtual loss
   or collision handling and changes the visit distribution that is itself the
   policy target. That is a class C strength lever, not S2 infrastructure.
3. **Evaluation cache** keyed on the encoded-state hash, shared across games in the
   same worker. Welcome To transposes weakly (sheets diverge fast), so expect this
   to pay less than in Kingdomino — measure the hit rate before sizing it.
4. **Feature cache — the Welcome To-specific one.** Several encoder blocks are
   expensive and *do not change within a turn*: the deck prefix sums, the supply
   rates, `expected_turns_to_plan`, and above all the §6.3 completion enumeration.
   Cache them on `(seat, sheet hash, offer hash)` and reuse across every simulation
   in the turn. Under the symmetric encoder these are computed for four seats, so
   this is the difference between symmetry costing ~18% and costing 4×.
5. ✅ **Backpressure.** Each resumable search has exactly one outstanding request
   and every worker blocks at the global inference barrier. There is no
   unbounded leaf queue: inflight search slots are the hard memory bound.

The production evaluator uses `forward_inference` (policy/rank/score only),
reused pinned CUDA staging buffers, and nonblocking H2D copies. Its stage profile
is written beside generation metrics together with the Rust profile (search,
encode, coordinator wait, pack, Python forward, decode, wall, waves, and batch
histogram). These counters are the cloud calibration surface; do not infer the
next bottleneck from mean batch width alone.

### The metric

**Measured end-to-end simulations per second, and mean realised batch size**, at
the chosen network size and seat mixture. Not GreedyBot `games/s`, which says
nothing about the configuration that will actually run.

Report both before committing to a network size: the capacity ladder in
`PROJECT_PLAN.md` M2 is only meaningful once the cost of a simulation is known.

### Production CUDA sweep — 2026-08-24

`s2_throughput.py` swept only CUDA, as requested, using the S0 checkpoint, the
production 200 simulations per learner root, the 60/30/10 seat mixture and
identical seed sets within each comparison. `inflight == max_batch` in every
arm. No CPU result is mixed into this selection.

| in-flight | games in arm | games/hour | evaluator rows/s | mean batch | p90 batch |
|---:|---:|---:|---:|---:|---:|
| 8 | 32 | 180.8 | 1,022 | 2.06 | 4 |
| 16 | 32 | 232.0 | 1,309 | 3.05 | 6 |
| 32 | 64 | 392.8 | 2,383 | 4.90 | 11 |
| 64 | 128 | 546.5 | 3,147 | 8.47 | 21 |
| 128 | 128 | 735.3 | 4,230 | 12.68 | 40 |
| **256** | **256** | **961.0** | **5,582** | **20.72** | **71** |

The 256 arm used only 36 MiB of Torch-allocated CUDA memory (about 1.86 GiB
device use including the process/runtime), about 2.4 GiB process working set,
and showed no thermal problem. It is now the default. The 961 games/hour result
is conservative for a long run: a fixed 256-game arm stops replenishing slots
as it drains, creating a long low-utilisation tail that an overnight corpus does
not have.

Concurrency is **not bitwise strength-neutral**. Identical games compared across
batch widths agreed between 90.6% and 94.5% in the larger paired sweeps; changed
floating-point batch geometry can flip close choices even though aggregate game
metrics stayed stable. The run manifest therefore pins both `inflight` and
`max_batch`. Do not resume a width-256 corpus at another width. For the first
overnight run, 7,500 games is about 7.8 hours of generation at the conservative
measured rate and leaves time for optimisation and evaluation.

---

## 7. Multiplayer orchestration: `games/az_loop` needs extending

`games.az_loop` is described in `PROJECT_PLAN.md` as "reusable, do not rebuild".
**That is true of the run controller, soft gate, checkpoint lifecycle, games
ledger, HOF, stagnation detection and run log. It is not true of match play.**

```python
def play_match(
    adapter: GameAdapter,
    agents: tuple[Agent, Agent],      # <- exactly two
    ...
    rngs = (random.Random(...), random.Random(...))   # <- exactly two
```

S2 and S3 need **2–4 independently selected agents** per game, so this is a real
build, budgeted here rather than discovered mid-stage. It stays game-agnostic —
Kingdomino is 2p and must keep working.

| piece | what it needs |
|---|---|
| `play_match` | N agents, N RNG streams, seat assignment; 2-agent calls keep working |
| `MatchOutcome` | per-seat scores, ranks and margins, not a single win/draw/loss |
| statistics | paired margin-vs-best and normalised rank (§S2 gate), aggregated per seat count |
| rating | **DECIDED: rate the 2p slice only** — see below |
| SPRT | pairwise; keep it on the 2p slice where it is valid |
| pool selection | independent draws per seat, with the 60/30/10 seat-count sample |

### Rating — DECIDED 2026-08-21: 2p slice only

Elo assumes **pairwise** outcomes and a four-player finishing order is not one.
Three options were considered:

| option | verdict |
|---|---|
| Rate the 2p slice only; 3–4p games train and gate but are unrated | **CHOSEN** |
| Decompose each finishing order into pairwise results (4p → 6 pairs) and feed Elo | rejected — the 6 pairs from one game are **not independent**, so Elo over-updates and its confidence intervals lie |
| An order-based rating (Plackett-Luce / TrueSkill-shaped) over finishing positions | rejected for now — a real build in a module that must stay game-agnostic for Kingdomino |

**Why 2p-only is sufficient: Elo is not the gate.** Promotion runs on paired
margin-vs-best-opponent (§S2). Elo's remaining jobs are opponent-pool selection and
the human-readable progress curve, and the 2p slice — 60% of games and 100% of the
deployment target — serves both.

**What is given up**, stated so it is not rediscovered as a surprise: a checkpoint
that is specifically strong at 3–4p (better temp-rank play, say) earns no rating
credit and may be under-weighted in the pool. Accepted. The decision is reversible
and forecloses nothing: an order-based rating can be added later without changing
anything else here.

---

## 8. Open decisions

- ~~**Value target composition.**~~ CLOSED 2026-08-21: a 4-wide masked rank
  softmax, utility applied at search time, blended with a variance-gated margin —
  `AUX_TARGETS_SPEC.md` §5. Margin is **vs best opponent**; margin-vs-each is not
  pursued. Loss weights remain a tuning question, not a design one.
- ~~**Trunk shape.**~~ CLOSED 2026-08-21: shared per-sheet MLP encoder over 17
  planes + 127 scalars, opponents concatenated, then a main MLP —
  `ENCODER_V2_SPEC.md` §2–3.
- **`k` in the confidence gate.** `k = 1` is the measured starting default,
  `k = 2` the Kingdomino-parity reference — `AUX_TARGETS_SPEC.md` §5.2. Re-dump the
  floor-corrected confidence distribution on the S0 corpus and re-choose. A tuning
  question with a starting point, not a design question.
- ~~**Multiplayer rating**~~ CLOSED 2026-08-21: rate the 2p slice only; 3–4p games
  train and gate but are unrated (§7).
- **One masked policy head versus per-phase heads.** Measure before splitting.
- ~~**Seat-count mixture weights.**~~ CLOSED 2026-08-21: 60/30/10 over 2/3/4,
  set by the arena target rather than by cost (§2). Revisit only if the advisor's
  deployment target changes.

---

## Appendix A — engine verification, 2026-08-20

First execution of the suite since the engine was written:
**187 passed, 1 failed**. The failure was a test assertion that was too tight, not
an engine defect: `test_independent_sampling_diverges_immediately` asserted
`first_divergence_turn(state) == 1` and observed 2. The claim it protects held
(`sheet_divergence` cleared its threshold in the same test); the assertion was
seed-sensitive and now asserts `<= 3`, with the magnitude assertion left strict.

**Current state: 228 passed, 0 failed**, including 45 differential tests. No
engine defect was found by any of this work.

## Appendix B — measured baselines

GreedyBot, advanced variant, all seats greedy, 40 paired seeds:

| seats | games/s | turns | dec/seat | dec/game | score (seat 0) |
|---|---|---|---|---|---|
| 2 | 5.00 | 30.6 | 100.6 | 198.6 | 51.5 |
| 3 | 3.19 | 30.2 | 98.7 | 294.0 | 50.6 |
| 4 | 2.06 | 29.9 | 98.4 | 388.5 | 51.8 |

City Plans (all 37): mean first-place value 8.84, mean second-place value 4.57,
**mean race premium 4.27** (min 2, max 7).

Score tracks, from `constants.py`: `BIS_SCORES` and `ROUNDABOUT_SCORES` are
penalties; `PERMIT_SCORES = (0,0,3,5)` with the third refusal ending the game;
`TEMP_RANK_SCORES = (7,4,1)`.

Branching by phase, one learner seat: `CHOOSE_CARDS` 37% (2.3 options),
`WRITE_NUMBER` 34% (13.1, max 165), `ACTION_ESTATE` 8% (6.2), `ACTION_PARK` 7%
(2.0), `ACTION_SURVEYOR` 7% (28.5), `ACTION_BIS` 4% (9.7).

Random play is a **rules fuzzer, not a baseline** — it blocks its own streets out
and ends games early with a mean score of 7.5.
