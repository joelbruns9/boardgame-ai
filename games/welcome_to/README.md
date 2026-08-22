# Welcome To... — game engine

A rules-exact Python engine for *Welcome To...* (Blue Cocker Games), transcribed
from the Board Game Arena implementation in `BGA Files/welcometo`, built to be
the substrate for an AlphaZero-style trained player.

Structure follows `games/kingdomino`: a pure engine, a flat action codec, an
information-set-safe encoder, baseline bots, and a pytest suite.

```
constants.py       card deck, sheet geometry, every score track
sheet.py           one player's score sheet, its rules, its scoring, its capacity
plans.py           the 37 City Plans, completion predicates, distance-to-completion
game.py            GameState: phases, legal actions, stepping, scoring, determinization
deck_knowledge.py  exact deck composition and next-reveal facts
action_codec.py    the 357-slot flat policy index space
encoder.py         (4, 12, 3, 12) + (4, 45) per seat, (1, 3, 12) + (358,) global
network.py         shared sheet encoder, trunk, per-seat + global heads (3.94M)
train.py           the S0 bootstrap: corpus, loop, gate
bots.py            RandomBot (fuzzer), GreedyBot (the actual floor), play_match
training.py        auxiliary self-play targets + the divergence meter
datagen.py         trajectory capture/replay for the supervised bootstrap
loop_adapter.py    GameAdapter seam for games.az_loop
random_play.py     smoke harness / throughput benchmark
tests/             pytest suite
```

## Status

The pytest suite has **not** been run. What has been executed: a byte-compile of
every file, an import of every module, one random game per configuration, and
targeted numeric checks of the pieces where reasoning was not enough. Those
checks earned their keep twice — they caught an inverted face index in
`redeterminize`, and a `GreedyBot` tie-break term that was measuring nothing.

```bash
cd C:/Users/joeld/projects/boardgame-ai
python -m pytest games/welcome_to/tests -q
python -m games.welcome_to.random_play --games 200 --players 4 --encode
```

Measured baselines, 2 players, 20 seeds:

| matchup | mean score | mean turns | how it ended |
|---|---|---|---|
| greedy vs greedy (advanced, 2 seats) | 51.4 | 30.8 | 24/25 third permit refusal |
| greedy vs greedy (advanced, 1 seat) | 47.8 | 31.6 | 57/60 third permit refusal |
| random vs random (base, 2 seats) | 7.5 | 19.6 | 20/20 third permit refusal |

Two things to read off that. Random play is not a weak opponent, it is a
*different game* — it ends ten turns early because it blocks itself out. And even
greedy ends almost every game on a third permit refusal rather than by filling its
sheet or completing the plans, so **"fraction of games ending on the plans" is a
free strength metric** to track through training.

## Training plan

**[SELF_PLAY_PLAN.md](SELF_PLAY_PLAN.md)** is the plan of record for training
(stages S0–S3) and for the rule edge cases verified against the BGA PHP.
**[PROJECT_PLAN.md](PROJECT_PLAN.md)** still holds Phase 1, the network design,
the metric set and the measured baselines.

The primary target is the **advanced variant** (`GameConfig(advanced=True)`) —
roundabouts and the ten extra City Plans — at **2–4 seats**, and the training
entry points (`loop_adapter.py`, `datagen.py`) default to that. One-seat play is
supported by the engine because BGA offers it, but it is not a training
configuration: it silently switches two scoring rules.

## Starting training: what is left to build

Reusable as-is, do not rebuild: **`games/az_loop`** (run controller, soft gate,
checkpoint lifecycle, games ledger, HOF, Elo, SPRT, stagnation, run log — all
game-agnostic), and `games/kingdomino` as the worked example of both seams
(`loop_adapter.py` for `core.GameAdapter`, and its lifecycle plumbing for
`contract.LifecycleAdapter`).

Done here: the engine, codec, encoder, aux targets, baselines, harness,
`loop_adapter.py`, and `datagen.py`.

Still to build, in the order that de-risks fastest:

**M1 — supervised bootstrap, no search.** Clone `GreedyBot` at 2–4 seats:
policy target = the action it chose, value target = final score, plus the aux
heads. The point is not the resulting player, it is proving the whole pipeline
(encoder → network → loss → checkpoint) against a known reference in an hour. If
the net cannot reproduce greedy's 33.2 mean score without search, nothing
downstream will work. `datagen.py` is the data side and already runs: 200 games in
11 s, 64 KB on disk, replayed and encoded at ~3.1 k samples/s.

*Storage note:* trajectories, not tensors. Five thousand games is ~1.6 MB as
trajectories against ~1.6 GB as float32 tensors, and replay re-runs the rules, so
a rules change invalidates stale data loudly instead of silently.

**M2 — `network.py`, `train.py`. BUILT.** `python -m games.welcome_to.train`
captures a GreedyBot corpus at the 60/30/10 seat mixture, trains, and reports the
S0 gate. Decisions taken, none of them inherited from Kingdomino:

* *Trunk.* A **shared per-sheet encoder** runs over each seat's 12 × 3 × 12
  planes and 45 scalars, and the main trunk reads the concatenated per-seat
  representations next to the 358 global features. This is an MLP or a small 1-D
  convolution along the street axis — the 13 × 13 ResNet from Kingdomino is the
  wrong shape and mostly wasted parameters. Concatenate the seats; do not pool
  them, because identity is what the plan race is about.
* *Heads.* Policy (357 logits, masked by `legal_mask()`), score, the 4-way rank
  distribution, plus the auxiliary heads from `training.TARGET_NAMES`, several of
  which are masked. `AUX_TARGETS_SPEC.md` is the spec of record for the set.
* *Weights.* By **group**, not per target — `policy` 1.0, `objective` 1.0,
  `capacity` 0.3, `plan_race` 0.3, `components` 0.2. Twenty-odd individual knobs
  is not a tunable object; five is, and a group is what an ablation switches off.
  Scores are normalised in `training.py`, not here: ÷80.
* *One policy head or several?* Twelve phases share one head with disjoint legal
  sets. One masked head is simpler and standard; measure before splitting it.

**M3 — `mcts.py`.** Single-player first, which is markedly simpler: no minimax
negation, back the score up directly. Chance lives only at the turn boundary, so
open loop with `redeterminize()` per simulation. The branching table above is the
sizing guide — half of all decisions have two options and need almost no search,
while `WRITE_NUMBER` and `ACTION_SURVEYOR` carry the real width.

**M4 — `self_play.py` + `train.py`.** Same trajectory format as M1, with visit
counts added as the policy target. `training.sample_targets` already provides
everything else.

**M5 — `lifecycle_adapter.py`.** Promotion is **paired-seed score margin** —
same seeds, same opponents, candidate against incumbent, with a paired
significance test — rather than a raw win rate. Score is a much denser signal
than win/loss in a game this low-interaction, and pairing removes most of the
variance. Report win rate alongside as a diagnostic. Everything else in `az_loop`
applies unchanged.

Defer until phase 2: HOF, Elo and SPRT (no opponents yet), inference batching
(until throughput actually binds), the endgame solver, and expert mode.

## What is implemented

| | |
|---|---|
| Base game | yes, 1–N players |
| Advanced variant (extra City Plans + roundabouts) | yes, `GameConfig(advanced=True)` |
| Expert variant (draft number/effect, pass the third card) | yes, `GameConfig(expert=True)` |
| Solo (solo card, deck-out end condition, flat temp bonus) | yes, `players=1` |
| One seat, standard rules | yes, `GameConfig(players=1, solo_rules=False)` — **engine support only, not a training configuration** |
| Ice Cream / Christmas / Easter boards | **no** — plan ids reserved, never dealt |

## Rules mapping

Every rule is traceable to a BGA source file; the method docstrings name the PHP
function each was transcribed from. The load-bearing ones:

| Rule | BGA source | Here |
|---|---|---|
| 81-card deck, number × effect multiplicities | `ConstructionCards::$deck` | `constants.DECK_COUNTS` |
| Numbers ascend strictly left→right; roundabouts reset the chain | `Houses::getAvailableLocations` | `Sheet.available_locations` |
| Bis copies an adjacent number, no fence between, never a roundabout | `Houses::getAvailableLocationsForBis` | `Sheet.bis_candidates` |
| Estate = fenced run with no hole and no roundabout | `RealEstate::getEstates` | `Sheet.estates` |
| Fence may not split equal neighbours or two plan-spent houses | `Surveyor::getAvailableZones` | `Sheet.surveyor_zones` |
| Park only on the street just built on | `Park::getAvailableZones` | `GameState._park_streets` |
| Pool only under a house on a printed pool | `Pool::canBuild` | `Sheet.can_build_pool_at` |
| Temp agency modifies ±1/±2, clamped to 0..17, and always crosses a box | `Player::getAvailableNumbersOfCombination`, `stActionTemp` | `GameState.numbers_for` |
| Roundabout fences both sides and costs a penalty box | `WriteNumberTrait::buildRoundabout` | `Sheet.build_roundabout` |
| City Plans ranked by turn, ties share first place | `AbstractPlan::getValidations` | `GameState.plan_scores` |
| Temp scored 7/4/1 by rank across players | `Actions/Temp::getScore` | `GameState.temp_scores` |
| Total = plans + parks + pools + temp + estates − bis − permits − roundabouts | `Player::computeScore` | `SheetScore.total` |
| Game ends on a full sheet, all three plans, three refusals (or empty deck in solo) | `EndOfGameTrait::isEndOfGame` | `GameState._is_end_of_game` |
| Ties broken by estate count, then size 1, size 2, … | `stComputeScores` | `Sheet.tiebreak_key` |

### Bis, in detail

Bis is the fiddliest rule in the game and it interacts with fences in both
directions, so it is worth stating exactly what the engine does.

* A bis writes the number of a **directly adjacent** house into an empty box.
  One gap is already too far (`hole == 0` in the BGA loops).
* It may copy from either side, and the two sides can offer **different**
  numbers for the same empty box. That is why the action codec stores
  `(box, side)` rather than `(box, number)`.
* An estate fence between the two boxes blocks it. A roundabout can never be
  copied.
* **A bis is an ordinary house to the next bis.** `8, 8b, 8c, 8d, …` is legal and
  the engine builds it: `bis_candidates` reads the neighbour's number and does
  not care how it got there. Runs grow in either direction, and each copy after
  the original counts towards the FiveBis plan.
* **No fence can be built anywhere inside such a run.** `Sheet.surveyor_zones`
  rejects any fence slot whose two neighbours show the same number, which is
  exactly a bis join — the ascending rule makes equal adjacent numbers impossible
  otherwise. A run of four 8s is four houses welded into one estate that no
  surveyor can ever split.

That last point is a real strategic constraint, not bookkeeping: a long bis run
permanently fixes the size of the estate containing it.

## How the reveal works, and what is actually hidden

Three stacks sit number-face-up. Each turn the top card of each stack is flipped
over and set beside it, so the table reads *effect* on the flipped card and
*number* on the new top of the stack; that is the pair players use.

The part that matters for modelling: **a card's number face also prints the
effect from its own back**, in two corners. So the card now showing a number
already tells you which effect it will contribute when it is flipped next turn.

```
turn T      [ effect of card T-1 ]  [ number of card T  (+ its own effect, printed) ]
turn T+1    [ effect of card T   ]  [ number of card T+1 (+ its own effect, printed) ]
```

Confirmed on three independent sources: `getPossibleCombinations` reads `action`
from the earlier-drawn card and `number` from the later one; `wtoCards.scss` gives
`.construction-card-front` — the *number* face — `.top-right-corner` and
`.bottom-left-corner` elements backed by `actions.png` and keyed on
`data-action`; and `getForPlayer` sends the client full rows, both faces, for both
cards. BGA and the cardboard game agree.

**So nothing on the table is hidden.** Both cards in a stack are fully
identified. This corrects an earlier version of this engine, which modelled the
top card's effect as face-down and carried a `GameConfig.chain` switch for a
"physical" reveal order that does not exist. Both are gone. The consequences:

* the effect each stack offers next turn is a **certainty**, not a posterior —
  `GameState.next_effects()`, fed to the network as a one-hot per stack;
* the only unknown is the next *number*, which lives on the card underneath;
* `redeterminize()` is therefore a pure shuffle of the undrawn deck. It never has
  to guess at a face, which makes it trivially correct.

## Card counting

`deck_knowledge.py` does the bookkeeping, and because the table is fully public
it is **exact, not estimated**: the deck is the printed 81 cards minus the discard
pile minus the six cards on the table, every one of which has been seen in full.
`deck_composition()` returns that as a `(15, 6)` histogram and its sum always
equals `state.deck_remaining` (tested).

The next number's distribution therefore sharpens all game long — near-uniform
over the printed multiplicities early (8 and 9 at nine copies each, 1 and 2 at
three), and by the back half of the deck a counting player knows which numbers are
spent. `next_number_distribution()` is that distribution, exactly.

The **joint** histogram is kept, not just the marginals, for two reasons. It
preserves the number/effect correlation — numbers 1, 2, 5, 11, 14 and 15 carry no
POOL, TEMP or BIS card; 3 and 13 carry no PARK or ESTATE — and it is what makes
the reshuffle decision computable rather than a matter of taste (below).

**Expert mode is deliberately not counted.** `getAllDatas` sends each client only
`getForPlayer($pId)`, so an expert player never sees the opponents' cards and
cannot attribute the shared discard. Counting there would leak, so expert
subtracts only the player's own three cards.

## Placement capacity

The strictly-ascending rule means a badly placed number destroys future
placements wholesale. `Sheet.placement_capacity()` measures it: each run of empty
boxes bounded by `low` and `high` can hold at most `min(run, high − low − 1)`
houses.

| sheet | street-0 capacity |
|---|---|
| empty | 10 |
| `1` in box 0 | 9 |
| `15` in box 0 | **2** |
| `5` in box 3, `6` in box 5 | 7 (box 4 is dead) |

This is the resource the game is really about, and it is why uniform random play
is not a baseline. It is exposed as a flat feature block, and per-box as
`Sheet.box_spans()` on spatial plane 10, so the network does not have to
rediscover the arithmetic of the ascending rule from scratch.

## Design decisions

### Simultaneous turns, serialised — and why that is safe here

Welcome To is simultaneous. The engine serialises a turn into `players`
consecutive private turns, which is exact rather than approximate because
nothing a player does during a turn changes what another player *may* do during
the same turn: **the stacks are shared but not consumed, so every player can take
the same combination**; City Plans are ranked by turn number, so two players
finishing the same plan on the same turn both take first place; and the temp
ranking and end-of-game check are evaluated once, at the end of the turn.

What serialising would leak is *information*. BGA hides exactly this
(`Houses::getOfPlayer` filters out the current turn for everyone but the current
player), and so does the engine: `GameState.public_sheets` is a snapshot taken at
the start of each turn, and `sheet_for(viewer, target)` / `plan_turns_for` are
what the encoder reads. The raw `sheets` list is ground truth and is *not*
information-set safe.

### The engine speaks integers

`legal_actions()` returns indices into a single fixed 357-slot space and
`step(i)` consumes one. There is no second action representation that can drift
away from the rules, and the policy head is exactly `NUM_ACTIONS` wide. Two
choices are worth noting: a **bis** is `(box, side)` for the reason given above,
and an **estate plan** is validated one estate at a time, the engine naming the
required size and the policy naming the estate by its leftmost box. Because each
requirement asks for an *exact* size, any estate of that size keeps the rest
satisfiable, so the sequential form never traps the player — and the action index
never depends on an enumeration order, which a policy head could not learn.

## Known deviations from BGA

1. **Voluntary permit refusal.** `permitRefusal()` in BGA only checks that a
   refusal box is free, so its server would technically accept a refusal while a
   pair is playable; its client only offers one when nothing is playable. The
   engine follows the client (and the printed rules).
2. **Expert-mode reshuffle.** BGA re-drafts a card per player on top of the one
   already passed, leaving two cards in one location with the same sort key. The
   engine tops up only a player whose pending slot is empty. Standard-mode
   reshuffle is reproduced exactly, including BGA's extra burned draw/discard
   cycle.
3. **Expert determinization is partial.** `redeterminize()` shuffles the undrawn
   deck but does not resample opponents' private stacks or the card in transit.
4. **Turn confirmation and restart** (`ST_CONFIRM_TURN`, `cancelTurn`) are UI
   affordances, not game rules, and are not modelled.
5. **Seasonal boards** are out of scope; their plans raise `NotImplementedError`
   if a caller reaches them.

## Toward a trained player

### Three races, not one

Welcome To is mostly played on your own sheet, but the interaction is where games are decided, and it is
not a single race — it is three, at different time scales.

**1. Finish all three plans first.** This ends the game on your schedule and is
the usual winning shape. Encoded as plans-completed per seat, plus
`plans.progress()` distance-to-completion for every plan and every seat off their
public sheets (`plan_race` block). Completion flags cannot express "two parks
away"; a `(fraction, steps_left)` pair can, and `steps_left` is kept separate
because a fraction hides whether the remaining work is one mark or six.

**2. Finish each individual plan first.** Each plan pays its first finisher
roughly double (12/7, 13/7, 10/5 …), so this is where the points actually swing.
Encoded per plan slot: both printed values, whether the first-place slot is still
open, and who has already banked it.

**3. Finish the *first* plan of the game — and then choose the reshuffle.** This
is the interesting one. Whoever completes the first City Plan in the game may
shuffle the discard back into the deck. Human players mostly reshuffle by reflex.
For a model that knows the exact deck composition, the gaps left on its own
sheet, the fences it still needs, and how many parks are outstanding, this is a
computable decision with a large one-off swing in the number distribution — and
it is a decision *nobody else at the table gets to make*.

The engine exposes it as a real policy action (`RESHUFFLE_YES` / `RESHUFFLE_NO`,
offered exactly when BGA offers it), and the encoder gives the model what it needs
to answer it: the current deck composition, the discard composition, and
`after_reshuffle_composition()` — the exact deck the choice would produce. The
`reshuffle_race` block additionally says whether the option is still open at all,
which is what makes racing for it worth anything.

Getting this one right is a plausible source of edge over strong humans, because
it is exactly the kind of decision people make on feel.

### Targets

Train a multi-part value: expected **final score** (dense, low variance, teaches
own-sheet play), **win probability** (the thing to optimise), and score margin
against the best opponent. At four players raw win/loss is a noisy label — much
of the variance is other people's luck — so the score head carries the gradient
early and the win head is what you gate on.

Add a **turns-to-finish head**: per plan slot and per seat, how many turns until
that seat completes it, with a "never" bucket, read straight off the finished
game. It is free supervision from self-play, it is exactly the quantity all three
races turn on, and KataGo-style auxiliary heads of this kind reliably sharpen the
trunk even when the output is never used at play time. A **plan-order head**
(which plan this player completes first) is the natural companion.

### Network

The spatial part is small (12 × 3 × 12 = 432 floats per seat) next to 45
per-seat and 358 global flat features, so this is not a convnet problem. A flatten-and-MLP trunk with residual blocks, or
1-D convolutions along the street axis, will do — start small (~1–2M params) and
use the capacity ladder from the Kingdomino work rather than assuming depth.
Policy head: 357 logits masked by `legal_mask()`. `encoder.block_slice(name)`
exists so a feature block can be ablated and the model checked for actually using
it — worth doing for `next_effects`, `capacity` and the reshuffle features.

### Search: agreed, decide it with data

Open loop is the right thing to build first, and the branching factor is the
reason. For the record, the two options and what the structure of this game says
about them:

* **Open loop** (Kingdomino's approach). The tree is indexed by action sequences
  only; chance is resampled every simulation and a node's statistics average over
  the worlds sampled beneath it. Cheap, small tree, no outcome explosion — but a
  node value is a blend across worlds, so it cannot represent "play A if the 7
  comes, B otherwise".
* **Explicit chance nodes** (7WD's approach). A reveal becomes a real node whose
  children are the outcomes, each with its own statistics. Sharper, and it needs
  the outcome set small enough that every child is visited often.

Two facts about Welcome To that make open loop the sane default:

* **A player's own turn is fully deterministic** — choose stack, write, resolve
  effect, maybe validate a plan. Three to five decisions, no chance anywhere, so
  the within-turn tree needs no chance handling at all.
* **Chance appears once, at the turn boundary, and it is only the numbers.** The
  effects are already known (see above), so the reveal is three fresh numbers
  from a known deck: roughly 15³ outcomes before deduplication. Too many for
  explicit children, and — crucially — the *effect* half of next turn's offer,
  which is what a chance node would most want to branch on, is not random at all.

So: open loop with per-simulation `redeterminize()`, plus **search only the
acting player's own subtree**, folding opponents into the value head and the race
features. A low-interaction game means that costs little and buys most of the depth back.
Whether a layer of explicit expansion at depth 1 helps is exactly the kind of
question to settle by measurement, on positions where a specific number matters.

### The long-horizon problem

Writing a 15 into the first box of a street on turn 3 costs nine future
placements, and your score on turn 3 is unchanged. The bill arrives around turn 20
as permit refusals and as parks, pools and estates you never got to build. No
search reaches that far — 25 turns with chance at every boundary is out of reach at
any realistic simulation count — so the value function has to carry it, and a
single scalar "final score" is a thin, high-variance way to teach it.

Three levers, in order of how much they buy:

**Decompose the target.** `training.final_outcomes()` reads off a finished game
the quantities whose *causes are local* even though their *values are late*.
Chief among them is **`permits`** — the number of refusals this player ends up
taking is the direct, low-noise signature of capacity mismanagement, it is a small
integer, and it is predictable from turn 3. Alongside it: final house count,
capacity left, and the eight score components separately rather than as one total.
That is eight or ten gradients instead of one, each attributable to a different
part of the sheet, and it is the single biggest thing you can do about credit
assignment here. `sample_targets(outcome, turn)` turns an outcome into the
per-visited-state target dict; `TARGET_NAMES` enumerates the heads.

**Give the arithmetic away.** `placement_capacity` and `box_spans` are already
encoder features. Capacity is a deterministic function of the sheet, so a big
enough net could learn it — but then it spends capacity learning *what capacity
is* instead of *what capacity is worth*. Hand it over.

**Consider variance, not just horizon.** Because the game is low-interaction, your
final score depends mostly on your own play plus card luck, not on opponents, so
Monte Carlo returns on **score** are much less noisy than the usual multiplayer
win/loss label. Train the score head on MC returns first; if the win head is too
noisy, bootstrap it with TD(λ) off the search value (λ ≈ 0.8) rather than reaching
for reward shaping. Potential-based shaping on capacity is available and
policy-invariant in principle, but a wrong potential quietly changes the optimum,
and the auxiliary heads get you the same signal without that risk.

**Watch it directly.** Mean permits, mean final house count, and capacity at turn
10 are cheap per-iteration metrics that move long before Elo does. So does
"fraction of games ending on the plans" — greedy self-play ends 20/20 on the third
permit refusal, so any movement there is real progress.

### Self-play divergence

In standard mode every seat sees the same three combinations. A deterministic
policy playing itself from identical empty sheets writes identical sheets forever,
and the game is worth one sample between all the seats. This is not hypothetical —
measured, with `GreedyBot` mirrored across both seats:

| self-play setup | sheet divergence | identical games | score spread |
|---|---|---|---|
| same policy, same tie-break stream | 0.00 | **100%** | 0.0 |
| independent sampling per seat | 0.84 | 0% | 12.0 |
| independent sampling, 4 players | 0.83 | 0% | 21.3 |

What rescues it is that **divergence is self-amplifying**: one differing choice
makes the sheets differ, every later decision is then taken in a different state,
and the trajectories never re-converge. First divergence lands on turn 1 above, and
84% of the sheet differs by the end. So the entire problem is concentrated in the
opening, and the fixes are cheap:

1. **Independent Dirichlet noise and an independent sampling stream per seat** —
   not shared across seats, which is easy to get wrong in a batched self-play
   worker.
2. **Hold τ = 1 longer than usual**, through the first ~8–12 turns rather than the
   first few moves. Divergence only has to happen once, and it has to happen
   early; the risk grows precisely as the policy sharpens, which is when a
   conventional temperature schedule has already dropped to argmax.
3. **Log it every iteration.** `training.diversity_report()` gives
   `identical_games`, `mean_sheet_divergence` and `mean_first_divergence_turn`,
   and `random_play.py` prints them. `identical_games` climbing off zero is the
   canary.

There is also a structural answer: play **asymmetric** games. Divergence is only a
problem when the policies are identical, so running the current net against a
frozen checkpoint or a pool of past ones guarantees divergence by construction and
needs no temperature discipline at all. Given the HOF machinery already exists from
the Kingdomino work, this is probably the cheaper route to reliability than tuning
τ schedules.

Worth noting what non-divergence would and would not cost, if it happened. Policy
targets are unaffected — MCTS visit counts do not care. Score-head targets are
merely duplicated. It is the win/loss head and the race heads that go dead, because
identical seats always draw. That is another argument for making score the primary
value target and win probability the secondary one.

### Curriculum

**Training runs 2-4 seats throughout.** The full argument is in
`SELF_PLAY_PLAN.md`; the short version is that a single-seat curriculum buys
nothing and costs correctness.

**It is not cheaper.** Training cost is paid per *learner decision searched*, not
per game, and that is flat across seat counts -- 100.6 decisions for seat 0 at two
seats against 98.4 at four (40 paired seeds, greedy). A four-seat game costs 2.4x
per game but yields four trajectories, and putting cheap bots in the non-learner
seats leaves the learner searching the same ~100 decisions it would have searched
alone.

**And one seat is a different game.** Two scoring rules change:

| component | one seat | 2+ seats |
|---|---|---|
| temp | flat 7 at six-plus boxes (`TEMP_SOLO_SCORE`) | pure rank 7/4/1; **zero temps scores 0** |
| City Plans | every plan pays its first-place value | first finisher paid roughly double |

The temp case is the sharp one, because the *shape* inverts rather than the level.
At one seat the first five temps are worth exactly nothing and the sixth is worth
7. At two seats the first temp is worth 4-7 and every one after it is worth
nothing unless it flips a rank. A policy trained at one seat learns "ignore temps
unless committing to six", which is close to the worst available multiplayer
policy, and no value-head rescaling recovers it.

`permits` is a third casualty: it saturates at 3 at one seat because that *is* the
end condition, so the single best low-noise signature of capacity destruction
carries no signal at all.

**Do not put a turn penalty in the objective.** "End fast" is a multiplayer idea,
and even there the right target is `turns_to_plan_*`, which is already a
first-class head -- ending early via permit refusals is a penalty *and* stops you
scoring. Optimise score and let turn pressure arrive with the opponents. Game
length is near horizon-neutral anyway (30.6 turns at two seats, 29.9 at four);
re-measure once the model can end games on the plans.

The resulting shape is three stages, all multiplayer:

1. **Bootstrap -- behaviour cloning from GreedyBot**, captured at 2-4 seats. Same
   cost per trajectory as anything else, and every opponent and race feature is
   live from the first gradient step.
2. **Main -- against an opponent pool**, starting at 100% GreedyBot and adding
   each promoted checkpoint. This is where most of the training happens. A pool of
   *different* policies also solves the divergence problem structurally, since two
   different policies cannot mirror each other.
3. **Polish -- raise the symmetric share**, with HOF, Elo and SPRT on. Apply the
   temperature and noise discipline above, and watch `identical_games`.

Sample the seat count per game rather than training at two and generalising later:
the temp majority has no structure below three seats, so 3p and 4p from the start
are what make that head learnable at all.

### Endgame

An exact solver is attractive and awkward for the same reason: the branching that
bites is chance, not moves. Two tractable forms:

* **Exact expectimax over the last two or three turns.** Free boxes bound the
  turns remaining, and by then the deck is short and its composition exactly
  known. Enumerate over *offers* rather than card identities — many distinct
  cards produce the same (number, effect) pair — which collapses most of the
  branching.
* **Perfect-information Monte Carlo.** Sample N complete futures from the known
  deck composition, solve each deterministically, average. Cheap and strong, with
  the standard caveat: PIMC assumes you will know the future, so it overvalues
  lines needing one specific card and never pays to keep options open. Evaluator,
  not policy.

### Baselines

`GreedyBot` is the floor, not `RandomBot` — see the table at the top. `RandomBot`
stays as a rules fuzzer, which is a job it does well and the only one it should
have. After the trained net, the real yardstick is BGA game logs.

**First measurements worth taking**, before any training: the score distribution
and end-reason mix from `random_play.py`; mean branching factor (it drives search
cost); how often a game ends on the plans versus the sheet filling; and, to check
the premise behind all three races, how often the player who completes all three
plans first goes on to win.
