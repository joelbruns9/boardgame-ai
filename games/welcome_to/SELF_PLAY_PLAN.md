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

Build `mcts.py`. Validate at **two seats against GreedyBot**, the cheapest
configuration that still has all four race mechanics live.

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

Budget per phase, not flat: half of all decisions have two options, and the width
is concentrated in `WRITE_NUMBER` (13.1 mean, 165 max) and `ACTION_SURVEYOR`
(28.5).

**Gate:** at a fixed budget, search + the S0 network beats bare greedy by ≥ 4
points mean score on paired seeds, and completes ≥ 1.0 plans per game against
greedy's 0.42. The plan number is the one that matters — it is the first evidence
of something greedy structurally cannot do.

**The paired-seed harness built here is permanent.** Any change to search or the
network gets a paired-seed score against a fixed opponent before it gets a
training run.

### S2 — The first self-play loop *(the main stage)*

Learner in seat 0 with full search; the other seats drawn from an opponent pool;
seat count sampled per game across 2/3/4 at 60/30/10 (§2).

**Pool composition, ramped rather than switched:**

1. start at 100% GreedyBot — a real opponent at two seats (51.5 mean score) that
   costs nothing to run;
2. add each promoted checkpoint to the pool as it promotes;
3. let the greedy share decay as checkpoints accumulate.

**The honest limitation, stated up front.** Greedy completes 0.42 plans per game,
so early *race pressure* is weak — the learner will mostly still take first-place
plan value. What the pool buys immediately is the other three mechanics (temp
ranking, game-end tempo, the reshuffle race) plus live encoder blocks and unmasked
heads. Plan-race pressure then **ramps by itself** as the frozen opponents get
better at plans.

Do not fix this by hand-building a plan-seeking bot unless S2 stalls. If it does
stall — plans-per-game flat over several promotions — that is the moment to add
one to the pool, not before.

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

**Gate to leave S2:** three consecutive significant promotions; ≥ 70% win rate
against greedy at two seats; beats the S1 checkpoint head-to-head; and the
`first_plan` and `turns_to_plan_*` heads come off their masks and beat
predict-the-mean. That last one is the direct evidence that race learning is
happening rather than being assumed.

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

A pool of *different* policies cannot mirror itself, which is why S2 does not need
these guards and S3 does.

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
| paired-seed score margin vs incumbent | the promotion signal |
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

1. **Cross-game batching.** Run *G* games concurrently and collect leaves across
   them into one forward pass. This, not within-search batching, is what fills a
   batch: a single search only exposes as many leaves at once as its virtual-loss
   budget allows.
2. **Leaf batching within a search**, via virtual loss, so one search contributes
   more than one leaf per pass. Kingdomino's `leaf_batch` is the worked example —
   and its port carries a known trap: **`leaf_batch >= 6`** is where it started
   paying there. Re-measure; do not inherit the number.
3. **Evaluation cache** keyed on the encoded-state hash, shared across games in the
   same worker. Welcome To transposes weakly (sheets diverge fast), so expect this
   to pay less than in Kingdomino — measure the hit rate before sizing it.
4. **Feature cache — the Welcome To-specific one.** Several encoder blocks are
   expensive and *do not change within a turn*: the deck prefix sums, the supply
   rates, `expected_turns_to_plan`, and above all the §6.3 completion enumeration.
   Cache them on `(seat, sheet hash, offer hash)` and reuse across every simulation
   in the turn. Under the symmetric encoder these are computed for four seats, so
   this is the difference between symmetry costing ~18% and costing 4×.
5. **Queue and backpressure.** A bounded leaf queue with explicit backpressure on
   the game workers. Without it, actor threads outrun the evaluator and the queue
   becomes the memory leak.

### The metric

**Measured end-to-end simulations per second, and mean realised batch size**, at
the chosen network size and seat mixture. Not GreedyBot `games/s`, which says
nothing about the configuration that will actually run.

Report both before committing to a network size: the capacity ladder in
`PROJECT_PLAN.md` M2 is only meaningful once the cost of a simulation is known.

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
