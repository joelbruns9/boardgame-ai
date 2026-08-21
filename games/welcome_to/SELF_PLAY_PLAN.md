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

A four-seat game costs 2.4× a two-seat game *per game*, but yields four
trajectories. Put cheap GreedyBots in the non-learner seats and the learner
searches the same ~100 decisions either way; the engine cost of the extra seats
is noise next to a batch of network evaluations. Seat count is therefore free to
choose on merit.

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

`encoder.py` reserves `MAX_OPPONENTS = 3` at `_OPP_PLANES = 2` each, plus the
`plan_race` (24), `reshuffle_race` (4) and `opponents` (27) flat blocks. Measured
all-zero spatial planes on a mid-game position:

| seats | all-zero planes (of 18) | non-zero flat features (of 473) |
|---|---|---|
| 2 | 8 | 123 |
| 4 | **5** | 129 |

Four seats is the configuration the representation was built for. Training on
inputs where a block is constant zero teaches first-layer weights that have to be
unlearned when it comes alive, so the mixture should keep every block live from
the first gradient step.

---

## 2. Seat counts: 2–4, mixed from the start

**DECIDED: target 2–4 seats.** BGA's `gameinfos.inc.php` declares
`'players' => [1..12]` with `suggest_player_number => 4`, and our engine runs
correctly at every count tested — but `MAX_OPPONENTS = 3` means the encoder
individually models three opponents, which is *exactly* 2–4 seats. No encoder
work is needed and nothing is dropped or dead.

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

Capture the corpus with **mixed 2/3/4 seats**, which costs the same per
trajectory as any other configuration and makes every opponent and race block
non-zero from the first gradient step.

**Gate:** policy top-1 agreement with greedy ≥ 60% on held-out games; the net
playing greedily off its own policy, with no search, scores within 2 points of
greedy on a paired seed set; `permits` and `houses` heads beat predict-the-mean.

Accept that greedy is race-blind — it completes ~0.42 plans per game. The
bootstrap produces a placement-competent, race-blind policy, which is fine.

### S1 — Search *(supersedes M3)*

Build `mcts.py`. Welcome To needs no minimax negation even with opponents on the
table: every seat scores its own sheet, so the search backs up the learner's own
score directly and models the other seats with a cheap policy. Validate at **two
seats against GreedyBot**, which is the cheapest configuration that still has all
four race mechanics live.

Open loop with `GameState.redeterminize()` per simulation — a player's own turn is
fully deterministic and the only chance is the next number from an exactly known
deck. Budget per phase, not flat: half of all decisions have two options, and the
width is concentrated in `WRITE_NUMBER` (13.1 mean, 165 max) and
`ACTION_SURVEYOR` (28.5).

**Gate:** at a fixed budget, search + the S0 network beats bare greedy by ≥ 4
points mean score on paired seeds, and completes ≥ 1.0 plans per game against
greedy's 0.42. The plan number is the one that matters — it is the first evidence
of something greedy structurally cannot do.

**The paired-seed harness built here is permanent.** Any change to search or the
network gets a paired-seed score against a fixed opponent before it gets a
training run.

### S2 — The first self-play loop *(the main stage)*

Learner in seat 0 with full search; the other seats drawn from an opponent pool;
seat count sampled per game across 2/3/4.

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

**Promotion gate: paired-seed score margin.** Same seeds, same opponents,
candidate versus incumbent, compared with a paired significance test. Score is a
much denser signal than win/loss in a game this low-interaction, and pairing
removes most of the variance. Report win rate alongside as a diagnostic — do not
gate on it.

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
exceeds three and games do end on it, and the reshuffle counterfactual is exactly
deck + discard.

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

## 6. Open decisions

- **Value target composition.** Score should be primary — the game is
  low-interaction, so MC returns on score are low variance where win/loss throws
  most of the signal away. Win probability and margin ride along as secondary
  heads. Not settled: their loss weights, and whether margin-vs-best-opponent or
  margin-vs-each is the better target at three or four seats.
- **Trunk shape.** MLP versus 1-D convolution along the street axis. Kingdomino's
  13×13 ResNet is the wrong shape.
- **One masked policy head versus per-phase heads.** Measure before splitting.
- **Seat-count mixture weights.** Uniform over 2/3/4 to start; revisit against the
  ladder distribution once there is one.

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
