# Welcome To... — encoder v2 specification

**Status:** spec of record for the encoder. Nothing here is implemented yet.

**Revision note.** An earlier draft of this document (same filename, 2026-08-21)
specified an *asymmetric* encoder — 17 planes for the viewer, 2 per opponent —
plus a 45-float `opponent_sheet_summary` compromise. That draft is **superseded**
and its compromise block is deleted. The decision below is full symmetry. The
reasoning that changed it is in §2; read it before proposing a return to
asymmetry.

**Scope:** `encoder.py`, plus the small helpers it needs in `plans.py` and
`deck_knowledge.py`. Auxiliary training targets are specified separately in
`AUX_TARGETS_SPEC.md`; the two documents are designed against each other and
should be read together.

**Target configuration:** base board, **advanced rules**, **2–4 players weighted
60/30/10**, non-expert. BGA arena is 2p base+advanced, historically 2–3p. Expert
rules are out of scope and are not a config flag away (§9.3).

---

## 1. The rule that decides what goes in

Two categories, and only one of them is safe:

**Category 1 — a pure function of observable state.** Placement capacity, box
spans, plan distance, deck composition, resource requirements, supply rates.
These carry no preference. They cannot teach a wrong idea because they contain no
idea. Adding them changes what is *cheap to learn*, never what is learnable.
**Add freely when cheap.**

**Category 2 — a feature that embeds a valuation.** A weighted move score, a
heuristic's ranking, "this placement is good". These install a human policy prior
in the input, create a shortcut the trunk leans on, and cap the ceiling.
**Excluded by contract** — `encoder.py` docstring point 6, "training-isolated,
nothing here imports a heuristic evaluator". That contract stands.

The line is not "raw vs derived". Every quantity below is derived. The test is
whether a reasonable player could disagree with it. Nobody can disagree with
"this gap admits 4 distinct numbers". Anyone can disagree with "a 15 belongs on
the right".

Two features sit near the line and are flagged as first-class ablations:
`positional_fit` (§8.1) and `expected_turns_to_plan` (§6.3).

---

## 2. The decision that shapes everything: full symmetry

**Every seat is encoded with the same planes and the same scalars, by the same
function, and runs through the same shared weights.**

### Why

The goal is the strongest Welcome To player there is. **At the top of the field,
everyone is a good solitaire player** — the placement half of the game is table
stakes. The differentiator is the other half: the plan races, reading what an
opponent will finish and when, and knowing how soon the game ends and at what
score. A model that represents its own sheet richly and its opponents' sheets as
two occupancy planes cannot compete on the axis that actually decides games at
that level.

The counter-argument — that the non-solitaire content is only ~5% of the game —
cuts both ways. It means the opponent model rarely matters, *and* it means that
when it matters it decides the game. Against a field of strong solitaire players,
the 5% is the entire competitive surface.

Kingdomino is the precedent: `my_board (B,9,13,13)` and `opp_board (B,9,13,13)`
run through the **same** ResNet trunk with shared weights. Its `opp_score` head
worked because it read a representation identical in richness to the one
`own_score` read.

### The cost is small, and an earlier estimate in this project was wrong

An earlier estimate put full symmetry at 2.5–4× compute. **That was wrong**, and
the error is worth recording so it is not repeated: it came from the KD analogy,
where the per-board trunk *is* a full 13×13 ResNet and therefore the dominant
cost. Welcome To's sheet is **36 cells**. A per-sheet stem over 36 cells is tiny,
and the bulk of the network is the MLP over the pooled sheet representations plus
~380 global features.

⚠ **The first version of this table was also wrong, in the other direction.** It
priced the sheet encoder against a ~17M trunk while `PROJECT_PLAN.md` M2 says to
start near **4M**. At the size we actually intend to build:

| component | approx params | runs per state |
|---|---|---|
| shared per-sheet encoder | ~250k | **2–4×** |
| main trunk | **~4M** | 1× |

```
asymmetric baseline : 250k × 1 + 4M = 4.25M
symmetric (4 seats) : 250k × 4 + 4M = 5.00M      →  +17.6%
```

So the honest figure is **roughly 18%, not 4%** — and if the capacity ladder later
lands on a larger trunk it falls back toward 4%. That is still cheap enough that
the decision stands, but the number in a spec has to be the number.

CPU-side encoding is the larger relative cost and is well under 4×, because the
expensive shared work — the deck prefix sums behind planes 13–16 and the supply
rates of §7.3 — is computed **once per state and reused across all four sheets**.
**Benchmark the chosen architecture end to end before accepting any of this**;
measured simulations/s is the number that matters, not a parameter ratio (see
`SELF_PLAY_PLAN.md` §6).

### What symmetry is not

Symmetry is **not** needed to make opponents play well, and it never was. The
encoder is viewer-relative and *the viewer rotates*: `encode_state(state)`
defaults to `state.actor`, so whichever seat is acting gets the full own-sheet
treatment, and `datagen.replay` records `outcomes[actor]`. In search, a node where
seat *p* moves is evaluated as `encode_state(state, p)` — p's own viewpoint. No
seat has ever been handicapped when acting. The asymmetry was a limit on the
viewer's *model of* opponents, never on their play.

Symmetry is also **not** what yields four trajectories per game. That is a
self-play regime choice (see `SELF_PLAY_PLAN.md` §1) and is available at any
encoder shape.

---

## 3. Shape

```
sheet_planes    (4, 17, 3, 12)      one block per seat, identical function
sheet_scalars   (4, 127)            one block per seat, identical function
viewer_plane    (1, 3, 12)          phase scratch, viewer only
global_scalars  (380,)              game-wide and viewer-relative
```

Seats are padded to 4 and carry a validity flag; `MAX_OPPONENTS = 3` still covers
exactly 2–4 seats. Seat order is the viewer first, then turn order, as
`_seat_order()` already does.

**Architecture this implies** (network design, recorded here because the encoder
shape assumes it):

```
for each seat s:  [planes_s, scalars_s] ──► SHARED sheet encoder (MLP) ──► h_s (128)

trunk_input = [h_me, h_opp0, h_opp1, h_opp2, viewer_plane, global_scalars]
                                    ↓
                              main MLP trunk
```

**Concatenate the opponents; do not pool them.** Pooling is seat-count-invariant
but destroys identity, and identity is exactly what is wanted — *which* opponent
finishes plan 2 first, *what each* will score. Zero-pad and mask instead.

**MLP, not convolution, for the sheet encoder.** The 3×12 grid has no symmetry
group: streets are not interchangeable and the strictly-ascending rule breaks
reflection. Weight sharing across seats is the sharing that pays here; weight
sharing across positions is not.

**No stored-tensor migration.** The corpus stores *trajectories*, not tensors
(5000 games = 1.6 MB vs 1.6 GB), and replay re-runs the rules. Stale data fails
loudly. The `encode_state` return signature changes, so `datagen`, `loop_adapter`
and the encoder tests all need updating — that is the real migration cost.

---

## 4. Per-sheet planes (17)

Identical function for every seat. Opponents' sheets come from
`state.sheet_for(viewer, target)`, i.e. the turn-start public snapshot; the
viewer's own is live.

| plane | contents |
|---|---|
| 0 | validity mask for the 3×12 right-padding |
| 1 | box is written |
| 2 | number / 17 |
| 3 | is bis |
| 4 | is roundabout |
| 5 | top fence (house consumed by a plan) |
| 6 | fence to the right |
| 7 | is a pool position |
| 8 | **writable with any currently-offered combination** (redefined, §4.1) |
| 9 | estate size / 6 |
| 10 | `box_spans` / 18 |
| 11 | `positional_fit` of the numbers on offer — **ablation target, §8.1** |
| 12 | `span_if_roundabout` / 18 |
| 13 | P(fit \| deck, no temp) |
| 14 | P(fit \| deck, temp ±2) |
| 15 | P(fit \| post-reshuffle deck, no temp) |
| 16 | P(fit \| deck, no temp, after an optimal roundabout) |

Everything here is computable for an opponent because their sheet is fully public
and **the three stacks are shared** (`_group_of()` returns 0 for every player in
non-expert mode), so the numbers on offer are the same for everyone.

### 4.1 Plane 8 is redefined, and it improves for the viewer too

The v1 definition read `state.ctx.number` — the combination *this player has
already locked in this turn*. That is viewer-only scratch state, and under
symmetry it would be **all-zero for every opponent by construction**: Welcome To
is concurrent, and BGA reveals nothing about anyone's choice until the turn
resolves. Our engine already enforces this (`public_sheets` freezes at turn start,
`plan_turns_for` hides same-turn completions); the serialised `actor` loop is an
implementation detail, not information anyone holds.

So plane 8 becomes **"writable with any currently-offered combination"** —
well-defined for every seat, and the natural "what are this seat's options"
signal.

This is an improvement for the viewer as well, not just an accommodation. The v1
plane populated only during `WRITE_NUMBER` and `ROUNDABOUT_PLACE`, so at
`CHOOSE_CARDS` — where you are deciding *which combination to take* — it was
blank. The new definition is live exactly when it is needed.

### 4.2 Planes 12–16 are per-gap, broadcast

Exact, not an approximation: `box_spans` bounds each empty box by the nearest
written number left and right, and for every box inside a maximal empty run those
bounds are identical. Compute once per gap (≤ ~10 per sheet), broadcast to its
boxes.

With exclusive bounds `(low, high)` — sentinels `low = MIN_NUMBER - 1 = -1`,
`high = MAX_NUMBER + 1 = 18`, a roundabout or street end removing a bound — and
`c[n]` the deck count of card number `n`, `D = deck_remaining`:

```
P(fit | deck, no temp)  = ( Σ n in (low,high) ∩ [1,15] of c[n] ) / D
P(fit | deck, temp ±2)  = ( Σ n in (low-2, high+2) ∩ [1,15] of c[n] ) / D
P(fit | reshuffled)     = as the first, over c + discard + aside, / (D + |discard| + 3)
```

Each is a prefix-sum lookup over a 15-vector. **Build the prefix sums once per
state and reuse them across all four sheets** — this is what keeps symmetric
encoding cheap.

Writable values are `0..17` (`MIN_NUMBER = 0`, `MAX_NUMBER = 17`, so
`NUM_NUMBER_VALUES = 18`); printed card numbers are `1..15`. `TEMP_DELTAS =
(0,-2,-1,1,2)` and `numbers_for()` clamps, so `numbers_for(1, TEMP)` is
`[1, 0, 2, 3]` — no −1.

**`span_if_roundabout` (plane 12).** For each empty box `b`, the largest span `b`
could have if one roundabout were placed at another empty box in the same street.
A roundabout removes the bound on the side it sits, so only the nearest empty box
on each side of `b` can matter; the naive max over ≤33 sites is also fine.

*This is the plane that makes the deferred-roundabout line representable.* Putting
a high number mid-street kills the boxes to its right — `box_spans` correctly
reports them dead — but a banked roundabout resurrects them entirely. Without it
the encoder says "dead" and the option is invisible. With it, the net sees "dead
now, alive for the price of a roundabout", the −3 / −8 penalty is already in the
score, and the trade becomes learnable. It says nothing about whether the play is
good. `plane 16 − plane 13` is that option value in probability units.

**Why there is no per-stack fit plane.** One deck feeds all three stacks in
standard mode, so the next-number distribution is identical across them. The only
per-stack difference is the known next effect, which decides whether the temp
widening applies — already a one-hot in `next_effects`. Planes 13 + 14 +
`next_effects` determine per-stack fit exactly.

---

## 5. Viewer-only plane (1)

**Locked-in writable mask.** Boxes legal for the combination the viewer has
already chosen this turn (`WRITE_NUMBER`), or every empty box
(`ROUNDABOUT_PLACE`), else zero.

This is the v1 plane-8 semantics, kept because it is effectively the legal-action
mask in spatial form and the policy head benefits from the trunk seeing it. It
lives **outside** the shared sheet encoder and feeds the main trunk directly,
because it is phase scratch state, not a property of a sheet.

---

## 6. Per-sheet scalars (127)

Identical function for every seat, fed to the shared encoder alongside that
seat's planes.

| block | size | contents |
|---|---|---|
| tracks | 20 | 3 parks, pool count, temps, bis marks, permits, roundabouts, 6 estate marks, 6 estate size counts |
| score breakdown | 9 | 8 components + total, from `score_breakdown(seat, viewer=viewer)` |
| capacity | 4 | `placement_capacity()` per street + total |
| roundabout repair | 3 | `capacity_if_roundabout` per street |
| total span | 1 | `total_span()` |
| plans | 48 | 3 slots × 16 — see §6.3 |
| demand | 24 | 18 number-demand + 6 effect-demand — see §6.1 |
| reshuffle contraction | 8 | see §6.2 — notemp/temp × number/effect × deck/reshuffled |
| refusal and blocking | 5 | see §6.4 |
| placement capacity this turn | 2 | see §6.5 |
| free boxes | 1 | ÷33 |
| is_viewer | 1 | 1.0 for the encoding seat |
| seat_valid | 1 | 0.0 for padded seats |
| | **127** | |

### 6.1 Demand

```
number_demand[v] = # of (empty box, v) pairs that are legal,   v in 0..17
effect_demand[e] = marks of effect e still wanted,             e in the 6 effects
```

`number_demand` is an **unweighted count of legal (box, value) pairs** — pure
combinatorics, no judgement. Weighting it by "how much I want that box" would move
it into category 2. Computed per gap: a gap of length `L` contributes `L` to every
value in its interval.

`effect_demand` comes from the plan requirement vectors (§6.3) plus remaining
track boxes, with one special case:

**ESTATE.** No plan kind requires the ESTATE *effect* — estate plans require
estate *sizes*, which come from fences. But the effect is critical for scoring and
its value is exactly computable:

```
estate_demand = Σ i of  count[i] × ( ROW_SCORES[i][marks[i]+1] − ROW_SCORES[i][marks[i]] )

ESTATE_ROW_SCORES = ( (1,3), (2,3,4), (3,4,5,6),
                      (4,5,6,7,8), (5,6,7,8,10), (6,7,8,10,12) )
```

A difference of two table lookups times a count. It captures why the estate effect
swings so hard: a mark multiplies across *every* estate of that size, so a size-6
row at 3 marks is worth `2 × (10−8) = 4` with two size-6 estates and nothing with
none.

### 6.2 Reshuffle contraction

⚠ **The demand and supply vectors live in different spaces and must be projected
before they can be contracted.** `number_demand` is indexed by **writable value**
`0..17`; deck supply is indexed by **printed card number** `1..15`. An earlier
draft wrote `Σ number_demand[n] × supply_number[n]`, which is undefined. Project
demand into card-number space first, in two variants:

```
card_demand_notemp[n] = # empty boxes b where n is legal in b                 n in 1..15
card_demand_temp[n]   = # empty boxes b where SOME d in TEMP_DELTAS makes
                        (n+d) legal in b, clipped to 0..17                    n in 1..15
```

Both are **counts of boxes**, not of (box, value) pairs — a box takes one value, so
a card either can or cannot serve it. Counting pairs would double-count a card that
reaches the same box through two different deltas.

```
fit_rate(supply)  = Σ n of card_demand[n] × supply_number[n] / (33 × Σ supply)
eff_rate(supply)  = Σ e of effect_demand[e] × supply_effect[e] / norm
```

evaluated with `card_demand_notemp` and `card_demand_temp` separately, and each
against the deck and against the reshuffled pool. `number_demand` over `0..17`
stays in the per-sheet block: it is a *sheet* fact and is useful on its own, but it
is never the operand of a contraction.

evaluated against the deck and against deck+discard — four scalars.

The reshuffle is not "do I like low numbers" in the abstract; it is "does the
post-reshuffle deck fit the holes I have left, better than the current deck does"
— and, for every opponent, "does it help them more than me". Under symmetry that
second half is answered for every seat automatically. Feeding both contractions
makes the decision a single subtraction, which is the one decision a counting model
can *compute* where strong humans reshuffle by reflex.

**Live every turn, not only at `ASK_RESHUFFLE`.** Anticipating the reshuffle is
what makes racing to the first plan worth more than its face value. Expect the net
to price the race higher as a result — intended; §7.4 says how to detect
over-racing.

**Tie rule, confirmed against the engine.** `A_RESHUFFLE_YES` sets
`reshuffle_next_turn = True` and nothing clears it until it is applied, so **one
YES from anyone shuffles**, regardless of how many NOs. A NO vote is therefore only
effective if every tied player votes NO — and their vote is concurrent and hidden.
This is the single genuinely game-theoretic micro-decision in the game, and the
threat features in §6.3 are what let the net condition on who else could be voting.

### 6.3 Plans — 3 slots × 16

Per slot, for this seat:

| field(s) | n | notes |
|---|---|---|
| `fraction`, `steps_left` | 2 | existing `plans.progress()` |
| requirement vector | 7 | parks, pools, temps, bis, houses, fences, roundabouts needed |
| `feasible` | 1 | ⚠ §7.1 |
| `turns_lower_bound` | 1 | hard bound |
| `effect_rate_turns` | 1 | expected turns from effect demand ÷ supply |
| `number_rate_supply` | 1 | fraction of the deck supplying this plan's number needs |
| `expected_turns_to_plan` | 1 | combined estimate — **ablation target** |
| `can_complete_this_turn` | 1 | exact-ish predicate, §7.2 |
| `p_complete_next_turn` | 1 | |
| | **16** | |

**Requirement vectors are the most important idea in this spec.** A `plan_id`
one-hot forces the trunk to memorise 28 separate plan semantics from scratch, and
each plan appears in roughly 3/28 of games — the rarest never get enough data. A
requirement vector describes a plan by *what it demands*, in a vocabulary the net
has seen hundreds of thousands of times. The one-hot stays (in global scalars) so
specifics can be memorised on top, but the requirement vector is what transfers —
and it is the only reason a shared model across board expansions is ever possible
(§9.2).

**Feasibility is not distance.** `progress()` answers "how far?"; it cannot answer
"is this still possible?". An estate plan wanting a size-3 estate is **dead** if
every remaining run is welded to length 5 by a bis pair, because `surveyor_zones`
refuses to split equal neighbours. An estate of 7+ scores zero, so a long bis run
is a trap. A net must learn a fence-legality rule *and* a combinatorial
reachability argument to see that. It will not, reliably.

**Rate, not just distance — the supply features.** This is what turns plan
*distance* into plan *speed*, and it is why the demand vectors are per-seat.

The effect column sums are strongly non-uniform: SURVEYOR 18, PARK 18, ESTATE 18,
but POOL 9, TEMP 9, BIS 9, out of 81. So "three steps away" means very different
things depending on which effect those steps need, and it shifts further as the
deck depletes. Two worked cases:

- *Six-single-estates plan.* The opponent needs fences (SURVEYOR). If the deck is
  low on surveyor, the plan is slow no matter how close `steps_left` says they are.
- *`ExtremitiesPlan`.* It needs the six end boxes; `(x, 0)` wants a **low** number
  and `(x, last)` a **high** one, and 1, 2, 14 and 15 are the **scarcest cards at
  3 copies each** against 9 copies of 8. "Two boxes away" can mean *next turn* or
  *effectively dead*, and `(fraction, steps_left)` cannot distinguish them.

```
expected_turns_to_plan ≈ Σ e of (marks of effect e still needed) / effect_supply_rate[e]
```

Summed rather than maxed, because you take **one combination per turn** — needing
both parks and fences means competing with yourself.

⚠ **`expected_turns_to_plan` is the one feature here that embeds a behavioural
assumption** ("takes the needed effect whenever offered"), which puts it closest to
the category-2 line. It goes in because nets are demonstrably bad at exactly this
arithmetic composition — the same argument that justifies planes 13–16 — but it is
a **first-class ablation**. `effect_rate_turns` and `number_rate_supply` are clean
facts and stay regardless.

**Threat — `can_complete_this_turn`.** `plan_race` says "two parks away"; it does
not say "and the cards on the table right now supply exactly that". The second is
computable, because all seats see the same three stacks. Combined with public
sheets, "could seat *p* complete plan *k* this turn?" is a deterministic predicate
over information the viewer is entitled to have.

This matters because of the tie rule: `plan_scores` pays `scores[0]` to *every*
player with `turn == first`, so a same-turn tie gives both the full first-place
value. The live question is never "will they beat me" but **"is this my last turn
to finish at all, or my last turn to finish alone"**.

⚠ **"Supplies the needed resource" is not a sound predicate, and this spec
originally used it.** It is resource-blind where the plans are position-bound:

- **park / pool** progress must happen in the *relevant street* or an actual pool
  box, not just anywhere — `Actions/Park` and `can_build_pool_at` both bind to
  position;
- **`CompleteStreet`** needs parks, pools **and** a roundabout **in the same
  street**;
- **`FiveBis`** needs five bis houses **in one street** (`bis_count_per_street`),
  not five on the sheet;
- **estate** plans need an exact **multiset** of sizes, which can conflict with
  houses already spent by a previous plan's top fences;
- **two remaining steps** are not generically reachable — whether a bis or a
  roundabout can supply the second one is position-dependent.

**Use bounded exact enumeration instead.**

```
if steps_left >= 3:               return 0        # unreachable in one turn
if steps_left <= 2:               enumerate this turn's legal action sequences,
                                  filtered to those that touch this plan,
                                  and test plans.can_be_scored on the result
```

Affordable because it only fires for a plan within two steps, which is rare — and
it is *exact*, which the abstraction never was. Cap the enumeration and cache the
result on `(seat, sheet hash, offer hash)`; the cache is also what makes the
per-seat symmetric version affordable (§9.4).

The same enumeration gives `p_complete_next_turn` by integrating over the next
number distribution with the **known** next effects.
`p_complete_next_turn` integrates the next-number distribution over the numbers
that would supply the step, using the **known** next effects.

**Honest limitation, to be in the docstring:** this answers "could they", not "did
they". Whether a seat has already acted this turn is genuinely hidden and correctly
so. Search and the value head absorb the rest.

### 6.4 Refusal and blocking (5)

⚠ **Corrected 2026-08-21 after external review.** An earlier draft specified a
single `P(refusal)` and said "temp applies, bis does not". Both halves were wrong
against the engine, and the shape of the error matters: **"refusal probability" is
not a state function at all.** A refusal is something a player *chooses*, and the
engine offers that choice under two different rules at two different phases.

**What the engine actually does.**

```python
# CHOOSE_CARDS  (game.py)
actions = [choose_stack(s) for s in self.playable_slots()]      # temp INCLUDED
if not actions and sheet.can_take_permit():
    actions.append(A_PERMIT_REFUSAL)                            # FORCED refusal
if advanced and ctx.last_house is None and not ctx.roundabout_declined         and sheet.can_build_roundabout() and sheet.has_free_box():
    actions.append(A_ROUNDABOUT_OPEN)                           # can RESCUE the turn

# WRITE_NUMBER
if not sheet.available_locations(ctx.number) and sheet.can_take_permit():
    actions.append(A_PERMIT_REFUSAL)         # printed number only -- VOLUNTARY
```

Three consequences the old feature missed:

1. **The two refusals have different legality rules.** At `CHOOSE_CARDS` a refusal
   is legal only when *no* combination is playable, and `playable_slots()` goes
   through `numbers_for`, so **temp is included**. At `WRITE_NUMBER` it becomes
   available whenever the combination's **printed** number has nowhere to go — even
   if temp could rescue it. The engine comment is explicit: *"never force a player
   to spend the temp agency just to have somewhere to write."* So a player may
   refuse voluntarily while a legal temp write exists.
2. **A roundabout can rescue an otherwise-forced refusal.** `ROUNDABOUT_OPEN` is
   legal at `CHOOSE_CARDS` whenever no house has been written this turn, and
   placing one returns to `CHOOSE_CARDS` (`game.py`) with the ascending chain reset,
   so `playable_slots()` is recomputed and may become non-empty.
3. **Therefore `P(refusal)` is behavioural, not factual.** It depends on whether the
   player elects to spend a temp, and on whether they spend a roundabout.

**The factual decomposition** — five scalars, each a pure state function:

| feature | definition |
|---|---|
| `p_no_slot_playable` | P(no offered combination has any legal write, **temp included**) |
| `p_no_slot_playable_after_roundabout` | the same, given the best legal roundabout placement; equals the above when no roundabout is available |
| `p_printed_unplaceable` | P(a given stack's **printed** number has nowhere to go) — the condition that *opens* the voluntary refusal |
| `roundabout_rescue_available` | 1.0 when `ROUNDABOUT_OPEN` is legal and would change `playable_slots()` |
| `p_forced_refusal_steady` | `p_no_slot_playable_after_roundabout` at the steady-state deck |

The first three are **exact for next turn**, because next turn's effects are
printed. Beyond that they use the steady-state composition of §7.3.

**Still not derivable from planes 13/14.** Those are per-gap; these are a *union
across gaps with overlapping number ranges*, needing inclusion–exclusion that a net
will not do reliably.

Under symmetry these are per-seat, which makes them the direct input to *"how soon
will they end the game"*: three refusals is an end condition and `permits` is the
aux head that predicts it. Note the model is being given the *opportunity*, and
learning the *choice* — which is the correct division of labour.

### 6.5 Placement capacity this turn (2)

⚠ **Corrected in the same review.** The earlier draft said "1 house normally, 2
with a bis". **It can be three.** `ROUNDABOUT_OPEN` is legal *before* the write
(`ctx.last_house is None`), placing a roundabout returns to `CHOOSE_CARDS`, and a
roundabout **counts as a built house**. So a turn can be:

```
roundabout (house 1)  →  choose combination + write (house 2)  →  bis (house 3)
```

capped at one roundabout per turn by the same `ctx.last_house` guard.

| feature | definition |
|---|---|
| `max_houses_this_turn` | 0–3, the largest number of houses the **currently legal** action sequences could place, ÷3 |
| `bis_usable` | 1.0 when a written house has an empty neighbour, so a bis write would be legal |

**How many houses actually get placed is behavioural** — a player may decline the
roundabout (it costs 3 or 8 points) and may decline the bis (`passAction()` is
generic, so the bis penalty is opt-in). The feature states the ceiling; the policy
learns the choice.

`bis_availability_rate` and `roundabout` supply live in §7.3 and the `tracks` block
respectively.

**Temp and bis still never appear in the same formula.** Temp governs §6.1–6.4
(which values are reachable); bis governs §6.5 (how many houses fit). The
roundabout is the one mechanic that appears in both, because it changes what is
placeable *and* adds a house — which is exactly why the original single-quantity
framing failed.

---

## 7. Global scalars (380)

| block | size |
|---|---|
| `phase` | 12 |
| `turn` | 1 |
| `stacks` | 78 |
| `chosen_combination` | 25 |
| `last_house` | 34 |
| `pending_estate` | 7 |
| `plans` (identity: one-hot 28 + scores + banked flags, ×3 slots) | 99 |
| `plan_conflict` | 6 |
| `reshuffle_race` | 4 |
| `next_effects` | 18 |
| `deck` | 74 |
| `effect_supply_rate` | 6 |
| `temp_availability_rate` | 1 |
| `bis_availability_rate` | 1 |
| `config` | 4 |
| `seat` | 6 |
| `seat_validity` | 4 |
| | **380** |

### 7.1 The plan one-hot is 28 wide, not 37

`PLANS` holds 37 entries for BGA id fidelity, but only 28 are ever dealt:

| stack | basic | advanced | dealt |
|---|---|---|---|
| 1 | ids 0–5 | ids 18–22 | **11** |
| 2 | ids 6–11 | ids 23–27 | **11** |
| 3 | ids 12–17 | — | **6** |

Ids 28–36 are the seasonal boards and `available_plan_ids` never deals them, so a
37-wide one-hot carries **9 permanently dead input slots** — the same dead-input
problem that settled `MAX_OPPONENTS`.

**New helpers:** `plans.DEALT_PLAN_IDS` and `plans.dense_index(plan_id)`.

**Size the one-hot at the advanced superset (28) regardless of variant**, so a
base-rules game is a strict subset of the advanced input space — ten slots stay
dark, roundabout actions never enter the legal mask — and one weight set serves
both. That is what lets the advisor read a base-rules table and give sensible if
untuned advice. Train on advanced only; base is supported, not trained.

### 7.2 `plan_conflict` (6)

`3 pairs × 2`: box-overlap fraction, mutually-exclusive flag.

Exactly three plan kinds consume houses — `EstatePlan`, `FullStreetPlan`,
`ExtremitiesPlan` — and they compete. `FullStreet` permanently *spends* a street;
`FullStreet` and `Extremities` fight over the same end boxes. "Completing slot 0
costs me slot 2" is not derivable from independent distances.

### 7.3 `effect_supply_rate` (6)

Per-turn probability that effect `e` appears among the three on offer.

⚠ **Not `1 − (1 − p)³`.** Three cards are drawn **without replacement**, so the
independent-trials form is an approximation, and this spec previously called it
exact. Use the hypergeometric complement:

```
k    = deck_effect_count[e]          D = deck_remaining
rate = 1 − (D−k)(D−k−1)(D−k−2) / ( D(D−1)(D−2) )
```

The magnitude is small — at `D = 40, k = 9` the approximation gives 0.535 against
0.545 exact — but it costs nothing to be right, the gap widens as the deck drains,
and a spec that says "exact" must be. Fall back to the approximation only if
`D < 3`, where the exact form is undefined.

The same correction applies to `temp_availability_rate`, `bis_availability_rate`
and the steady-state term in §6.4.

Next turn's effects are already *certainties* (`next_effects`, printed on the card
corners), so this is the steady-state rate for turn+2 onward. The two together give
near-exact short-horizon supply. `temp_availability_rate` and
`bis_availability_rate` are the same quantity surfaced separately because §6.4 and
§6.5 each need exactly one of them.

**There is deliberately no per-gap number supply *rate*.** The `1 − (1−p)³`
transform of plane 13 is monotone in a value already in the input, and a couple of
layers learn it trivially. `effect_supply_rate` earns its place because the effect
marginal is not otherwise in a usable shape.

---

## 8. What the encoder must NOT do

- **No heuristic evaluator import.** Contract point 6 stands.
- **No scalar "priority" or "goodness".** Requirement vectors stay vectors;
  contractions stay `<fact, fact>`.
- **No reading the undrawn deck.** Everything comes from `deck_knowledge`.
- **No live opponent sheets.** Everything opponent-side routes through
  `sheet_for()` / `plan_turns_for()`. Symmetry means *same features*, never *more
  information*.
- **No same-turn leakage.** Welcome To is concurrent; nothing another seat did
  this turn may appear anywhere, including in a symmetric block. §4.1 is the
  worked example.
- **Search must still `redeterminize(rng)` at the root.** A clean encoder does not
  make a cheating rollout honest. The `rng` argument is **required and must be a
  search RNG the caller advances between simulations** — see `game.py`. It was
  optional once, defaulting to the copy's own generator, and since `copy()` clones
  RNG state exactly, every determinization returned the *same* shuffle. Fixed
  2026-08-21 with regressions in `tests/test_game.py`.

---

## 9. Correctness risks and required tests

### 9.1 `feasible` fuzz test — blocking

`tests/test_plan_feasibility.py`: generate reachable sheets (the ~9k corpus from
`test_bga_differential.py` **plus constructed sheets**), brute-force whether each
plan is still satisfiable, assert the flag agrees.

Constructed sheets are mandatory: random play satisfies only ESTATE / EXTREMITIES
/ FULL_STREET, so a random corpus is vacuous for FIVE_BIS, SEVEN_TEMP, DECORATIVE
and COMPLETE_STREET.

⚠ **`feasible` is the only feature in v2 that can be silently wrong.** Everything
else is a count or a probability that is either right or obviously broken. A
feasibility flag can be confidently, quietly false — exactly the shape of the 7WD
`sci_win_feasible` blind spot, where a feasibility feature misread for a long time
because the encoder omitted the discard pile. It ships with this test or it does
not ship.

### 9.2 `can_complete_this_turn` bidirectional test — blocking

⚠ **The first version of this test was itself unsound.** It checked only false
negatives, and it checked them *against the same "supplies the step" abstraction
the predicate used* — so it would have validated a faulty predicate by agreeing
with it. A test written in the language of the thing under test proves nothing.

Test **both directions against an independent oracle**: for every position with a
plan within two steps, brute-force the legal action sequences for the turn and
record whether any reaches `can_be_scored`. Assert exact agreement.

- A **false negative** makes the net believe it has a free turn — the dangerous
  direction for the race.
- A **false positive** makes it panic-race for something unreachable, which is how
  a plan-blind policy turns into a plan-obsessed one.

Include constructed positions for `COMPLETE_STREET` and `FIVE_BIS`, whose
same-street requirements are exactly what the abstraction got wrong and which
random play never reaches (§9.1).

### 9.3 Symmetry test — blocking

**The single most valuable test in this spec.** Build a state, encode it from seat
0 and from seat 1, and assert that seat 1's per-sheet block in the first encoding
is *identical* to seat 1's own-sheet block in the second, except for `is_viewer`
and any quantity legitimately dependent on the turn-start snapshot.

Symmetry is easy to break by accident — one helper reading `state.sheets[p]`
instead of `sheet_for(viewer, p)` both breaks symmetry and leaks information. This
test catches both at once.

### 9.4 Fit-probability exactness test

Assert planes 13–15 equal a brute-force count over the deck histogram, that the
per-gap broadcast equals a naive per-box computation, and that plane 14 ≥ plane 13
everywhere (temp can only widen).

### 9.5 Metrics to log from iteration 1

- **roundabouts built per game *and the turn built*.** GreedyBot builds 1.28 in
  advanced. Early = treating them as scoring moves; late = as repair. This
  distribution says whether plane 12 did its job.
- **bis writes per game.** The penalty is opt-in (`passAction()` is generic), so
  the failure mode is silent under-use that looks fine on score.
- **plans completed, split by eventual winner and loser**, plus plans completed
  *first*.
- **reshuffle-choice frequency**, and score conditional on winning the first plan.
  Over-racing shows up as "wins the first-plan race, loses the game".
- **fraction of games ending on the plans** rather than on a third permit refusal.

### 9.6 Advisor train/serve skew — blocking for deployment

The model is trained assuming **full identification of all six table cards**,
including immediately after a reshuffle, where three cards go draw → flipped-aside
inside one transition and never spend a turn showing their number. That is
legitimate on BGA: `getForPlayer()` calls `getTopOf($stack, 2, false)`, which
returns full rows with `$customFields = ['number', 'action']`, and the DOM carries
both `data-number` and `data-action` on every card.

**The `games/advisor` extractor must read both faces of both cards in every
stack**, not just the visible face. Reading only the showing face feeds a degraded
state the model never saw, and the degradation is worst right after a reshuffle,
exactly where the counting edge is largest. Needs a test against a captured
post-reshuffle position.

This is an information-set *choice* and belongs in the `deck_knowledge` docstring:
the model reads from the wire what a human at a physical table would have to have
memorised. If physical-table fidelity is ever wanted, the fix is localised to
`known_cards()` — mark those three cards, subtract them from the effect marginal
only, leave their numbers in the deck.

---

## 10. Considered and rejected

### 10.1 `positional_fit` modification — rejected; it becomes an ablation target

`positional_fit` scores a number by its distance from the proportional ideal box in
its gap. It earned its place on GreedyBot (+3.2 points, t=2.2) and is a clean
geometric quantity — but it encodes a *theory*, and there is a concrete line where
the theory is wrong: a high number placed mid-street scores badly by construction,
which is exactly the deferred-roundabout play.

Making it roundabout-aware was considered and rejected — it would make one feature
mean two things. The right fix is one layer down: planes 12 and 16 make the option
representable without touching the definition.

It stays, because it measurably helps and removing it pre-emptively would presume a
strategic conclusion. But: it is a first-class `block_slice` ablation; the ablation
must be measured on **roundabout play quality, not mean score** (mean score will
hide it); and the S0 bootstrap clones GreedyBot, which *uses* it, so the initial
policy carries the proportional-placement prior.

### 10.2 The (15,6) deck joint — demoted to optional

Originally flagged as a confirmed defect. **It is not.** The offered combination
pairs **two different cards** — `combination()` reads the number off `stack_new`
and the effect off `stack_old` — so the deck matrix's structural zeros constrain
what is printed on *one card*, never what appears as an offer. `(1, POOL)` and
`(4, SURVEYOR)` are ordinary; at game start `P((1,POOL))` ≈ (3/81)(9/81) ≈ 0.004,
about one appearance every three games.

Since number and effect come from different cards, the next number is a plain
marginal draw, and the next *effect* is printed and readable, **the marginals
capture essentially everything knowable.** The joint's residual value is two-turn
planning, which is second-order.

Optional block if an ablation wants it: deck joint 90 + discard joint 90. Not in
the 380.

### 10.3 `opponent_sheet_summary` (45 floats) — deleted

The compromise from the superseded draft: derived per-opponent summaries instead of
full planes. Subsumed entirely by §4 and §6.

### 10.4 Per-stack fit planes, per-gap number supply rate — rejected

§4.2 and §7.3 respectively. Both are redundant transforms of values already
present.

### 10.5 Raising `MAX_OPPONENTS` — rejected

Costs a full sheet block per extra seat and leaves it dead at small tables. If 5+
seats ever matter, add a fixed-size *aggregate* opponent block. With a 2–4 seat
target, 3 is exactly right.

### 10.6 The policy vocabulary — FROZEN

An earlier draft said "flat, masked, 495" with `CHOOSE_STACK + WRITE` as an MCTS
macro-action, and left six things undefined. Frozen here.

**The macro covers the whole `CHOOSE_CARDS → WRITE_NUMBER` segment**, because that
is where the coupling is: you pick a combination *for* a placement, and a sequential
decomposition scores "take slot 2" by the *mean* quality of its placements rather
than its best — backwards, since you choose the placement.

| index range | n | contents | legal when |
|---|---|---|---|
| macro write | 495 | `(slot, temp delta, box)`, `3 × 5 × 33` | `CHOOSE_CARDS` |
| macro refuse | 3 | `(slot, PERMIT_REFUSAL)` — take a slot whose **printed** number is unplaceable, then refuse (§6.4) | `CHOOSE_CARDS` |
| direct refuse | 1 | `A_PERMIT_REFUSAL` with no slot playable | `CHOOSE_CARDS` |
| roundabout open | 1 | `A_ROUNDABOUT_OPEN` | `CHOOSE_CARDS` |
| primitives | 184 | the 357-slot codec minus `CHOOSE_STACK` (6), `WRITE` (165), `PERMIT_REFUSAL` (1), `ROUNDABOUT_OPEN` (1) — all subsumed above | their own phases |
| | **684** | | |

Answering each open question directly:

- **Where non-write actions live:** the 184 primitives, unchanged, at their own
  phases — `ROUNDABOUT_POS` (33), `SURVEYOR_FENCE` (30), `ESTATE_ROW` (6),
  `PARK_STREET` (3), `POOL_BUILD` (1), `BIS` (66), `CHOOSE_PLAN` (3),
  `VALIDATE_ESTATE` (33), the seven `PASS_*`, and `RESHUFFLE_*` (2).
- **How roundabout and refusal compete at a macro root:** they are ordinary entries
  in the same head, legal simultaneously with the macro writes. A roundabout is not
  a nested decision — taking it applies `ROUNDABOUT_OPEN`, and the placement is a
  separate primitive decision that returns to `CHOOSE_CARDS` for a second macro
  choice. That is also why the roundabout can rescue an otherwise-forced refusal
  (§6.4).
- **How S0 primitive trajectories become macro labels:** replay the recorded
  primitive trajectory and collapse each `CHOOSE_STACK → (WRITE | PERMIT_REFUSAL)`
  pair into its macro index; `ROUNDABOUT_OPEN` and a direct `PERMIT_REFUSAL` map
  1:1. Deterministic and total — every primitive trajectory has exactly one macro
  reading. `datagen.replay` does the collapse, so the stored corpus stays primitive.
- **Whether WRITE-phase network calls still exist:** **no.** The macro edge applies
  both engine steps, so there is no evaluation at `WRITE_NUMBER`. That is also where
  the branching went (13.1 mean / 165 max), so it is the main saving.
- **Total width:** **684**, fixed, independent of seat count.
- **Legal-mask contract:** a macro index is legal **iff its full primitive sequence
  is legal end to end** in the engine. The mask is built by enumerating engine-legal
  sequences, never by intersecting per-step masks — `WRITE` legality depends on the
  slot chosen, so a per-step intersection would admit illegal pairs.

**Engine and codec are unchanged.** The macro layer is a `macro_codec` module plus
the search's edge construction, so it can be reverted to primitives without
touching the rules.

**Flat, not bilinear.** At 684 a flat masked head is ~350k params at D=512, while
bilinear needs a D×D interaction (~262k) *plus* two encoders — the larger model,
saving nothing. Bilinear's generalisation benefit applies to large, sparsely
co-visited action spaces (Kingdomino's 3390); Welcome To's factors are tiny and
densely visited, and its cost is a rank constraint on the (combination × box)
logit matrix. Revisit only if `SURVEYOR_FENCE` (28.5 mean branching) is ever folded
into the macro, which would multiply the space into five figures.

---

## 11. Forward compatibility

### 11.1 Player counts

Fixed shape, seats padded to 4 with a validity mask, seat count as a feature, seats
sampled 60/30/10 across 2/3/4. The 10% at four seats keeps the third opponent's
sheet block trained; without it those weights sit at initialisation while receiving
live signal whenever the advisor meets a 4p table.

### 11.2 Board expansions

Boards (base / ice cream / christmas / easter) and rules (base / advanced / expert)
are **orthogonal and freely combinable** — `OPTION_BOARD` is separate from the
rules options, and each board brings extra City Plans. The seasonal boards keep the
3-street geometry (`Christmas.php` indexes `$streets[$h['x']][$h['y']]`) and are
mutually exclusive with each other. So an expansion is: one extra per-street track,
one extra action, three replacement stack-3 plans.

Four accommodations now, at near-zero cost, keep a shared model possible later:

1. **Requirement vectors carry the load; the plan one-hot is decoration** (§6.3).
   Decisive — a one-hot makes a shared model impossible, because a new expansion
   plan is a slot with no learned meaning.
2. **A board-variant one-hot**, not the current `(advanced, expert, solo)` booleans.
3. **Reserve the effect axis** at the union width; extra columns sit at zero.
4. **One spare per-street track slot** in the per-sheet `tracks` block.

Decide single-vs-specialist empirically once there is a base-board baseline to
regress against.

### 11.3 Expert rules — out of scope, and not a flag

Expert breaks the foundations: the shared discard is not attributable so
`deck_composition()` degrades to counting your own three cards, and
`redeterminize()` does not resample opponents' private stacks. It also breaks §4's
premise that all seats see the same three stacks. Real hidden-information
machinery, not a config toggle.

---

## 12. Implementation order

1. **`plans.DEALT_PLAN_IDS` + `dense_index`**, plan one-hot 37 → 28.
   Self-contained, no new maths. Do it first.
2. **Restructure to the seat axis** — `encode_state` returns
   `(sheet_planes, sheet_scalars, viewer_plane, global_scalars)`; move
   `own_tracks` / `own_score` / `opponents` into the per-sheet block; add the
   symmetry test (§9.3) **before** adding any new feature.
3. **Fit-probability planes 13–15** + the shared deck prefix-sum helper, with §9.4.
4. **Supply rates** (§7.3) and **refusal / houses-per-turn** (§6.4, §6.5).
5. **Demand + reshuffle contraction** (§6.1, §6.2) — reuses the prefix sums.
6. **Threat predicates** (§6.3), with §9.2.
7. **Roundabout planes 12, 16 + repair scalars.**
8. **Plan requirement vectors, feasibility, `expected_turns_to_plan`,
   `plan_conflict`** — with §9.1 first. Last because it is the only block that can
   be silently wrong.

Step 2 is the API break and everything else rides on it; do it in one commit with
the symmetry test green before touching features.

Each step is independently ablatable via `encoder.block_slice(name)`. Keep the
plane-index constants public so tests cannot drift.

**Every step ships with its ablation.** The vector roughly doubles; the burden of
proof is on each block to earn it.
