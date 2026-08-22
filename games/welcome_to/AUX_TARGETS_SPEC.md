# Welcome To... — auxiliary target specification

**Status:** spec of record for `training.py`'s target set and the network's head
structure. **Steps 1 and 2 of §10 are implemented** — the rank distribution, the
dropped `first_plan`, the target scales, `masked_mean`, the sentinel test and the
seat-indexed target set are all in `training.py`. Steps 3–5 are not.

**Reads with:** `ENCODER_V2_SPEC.md`. The two were designed against each other —
the head structure below depends on the symmetric per-seat encoder, and several
targets are the output-side twin of an input-side feature. Read that document
first.

**Supersedes:** the 24-target set currently in `training.py`. Six defects in that
set are itemised in §3; three of them became live when the seat mixture was fixed
at 60/30/10.

---

## 1. What an auxiliary head is, and what it is not

The network is a trunk plus heads. The trunk turns the encoded state into a
representation `h`; each head is a small map from `h` to an output.

**During training** every head gets a loss against what actually happened in the
finished game, and all of those gradients flow back through the heads **into the
shared trunk**.

**During play** MCTS uses the policy (as priors) and the value (as the leaf
value). The auxiliary heads are computed and discarded — or not computed at all.

So an auxiliary target is not information handed to the search. **It is a
constraint on what the trunk is allowed to forget.**

### Why that matters here more than in Kingdomino

What the trunk keeps in `h` is decided entirely by what the losses ask for. Train
against final score alone and the trunk only needs to retain whatever predicts one
number — and a representation adequate for one scalar can be impoverished in ways
that look fine on the training distribution and fail off it.

Welcome To has an unusually long causal path. **Writing a 15 into box 0 on turn 3
costs nine future placements and changes your turn-3 score by zero.** The bill
arrives around turn 20 as permit refusals and buildings never made. Seventeen
turns of noise between cause and signal, with chance at every turn boundary, and
no search reaches that far.

`permits` — "this sheet will end up taking two refusals" — is predictable from turn
3, low variance, and *directly caused* by that turn-3 placement. Short, clean
gradient path. The trunk learns capacity-consequence reasoning from the permits
head; the value head inherits it.

Kingdomino needed none of this: place a domino and your board score changes
immediately, so final score is a good target with a direct causal path from every
decision. That difference — not a general belief that more heads are better — is
the entire justification for deviating.

### Aux heads never enter the leaf value

Writing `leaf_value = score + 0.3 × permits` would mean *we* decided what a permit
is worth. That is hand-tuning a valuation — the category-2 mistake from the
encoder spec, moved to the output side. It is also redundant: if predicted permits
are informative about final score, the score head already uses them, since both
read the same `h`.

**The one legitimate blend is `margin` against the rank value** (§5), because those
are two framings of *the objective itself*, not features being assigned a price.
Choosing between "maximise margin" and "maximise rank" is choosing an objective;
deciding a permit is worth 0.3 is not.

Aux heads *may* be read at play time for **explanation** — an advisor saying "I
expect you to take 2 permits and finish plan 2 around turn 19" is free, since the
forward pass already happened. That is a UI feature, not a search input.

---

## 2. How many, and why this many

| project | non-policy heads | scalars |
|---|---|---|
| Kingdomino | 3 (`own_score`, `opp_score`, `win_prob`) | **3** |
| 7 Wonders Duel | 5 (`value`, `joint7`, `margin`, `military`, `science`) | **14** |
| **Welcome To** | **2 head groups** | **16 output units → 49 predictions** |

KD has **zero** auxiliary heads: every output *is* the objective, and `margin` is
not even a head — it is derived in `mcts_az.py` as
`tanh((own_norm − opp_norm) × MARGIN_GAIN)`.

7WD has a few because its outcome is **multi-modal**: three win conditions, and a
W/D/L target destroys *which one*. `joint7` recovers the mode; `military` and
`science` are the track positions determining two of the three paths. Note that
all of 7WD's aux heads are about **outcome structure**, not intermediate
mechanics — there is no "how many cards will I buy" head.

Welcome To sits between them, and the count is smaller than it looks because of
the structure in §4: **16 output units of parameters, evaluated to produce 49
predictions.**

**The cost of a target is not parameters.** It is gradient competition (every
target pulls the trunk toward serving it), knob count (each needs a scale and a
weight), and diminishing returns from correlation. So the selection criterion is
**not** "is this true and computable" — nearly everything about a finished game
is. It is:

> **Does this target force a representation the others don't?**

Everything that failed that test is in §8, including several things an earlier
draft had in the core.

---

## 3. Defects in the current 24-target set

**D1 — `rank` is not seat-count-invariant.** `rank_of[player]` is a 0-indexed
position: `{0,1}` at two seats, `{0,1,2,3}` at four. With the 60/30/10 mixture the
head sees `1.0` meaning "last of two" in 60% of games and "second of four" in 10%
— different things, same number. It will learn the average of two incompatible
meanings. Fixed by the rank distribution and its utility `u_r` (§5).

Second half: `ranking()` uses `sorted()`, so two seats with identical
`(score, tiebreak_key)` get distinct ranks assigned **by seat order**, making the
target depend on seat index. Ties must share the averaged rank.

**D2 — `first_plan` regresses a categorical.**
`float(outcome.plan_order[0])` emits a slot index as a float. Slot 1 is not
"between" slot 0 and slot 2. Dropped; `will_complete_plan_k` subsumes its value.

**D3 — the masked `turns_to_plan_k` has no complement.** The masking itself is
correct, but it means the head **only ever trains on completions**. GreedyBot
completes 0.42 plans per game across three slots, so the mask is 1 roughly **14%
of the time**, and the model is structurally unable to represent "I will not
finish this plan" — the modal outcome. Fixed by the unmasked
`will_complete_plan_k` (§6).

**D4 — `won` disagrees with `returns()` on ties.** `won = player in winners` gives
1.0 to *every* tied winner, while `returns()` gives `2 × share − 1` = 0.0 for a
2-way tie at 2p. Resolved by dropping `won` in favour of the rank distribution,
which spreads mass across tied ranks and is consistent by construction (§5).

**D5 — no opponent-outcome targets at all.** No margin, no per-opponent score, no
end-condition. Under the symmetric encoder these are the whole point (§5, §6).

**D6 — no normalisation is specified.** Targets are emitted raw: `score` ≈ 75,
`permits` ∈ 0–3, `capacity_left` ∈ 0–33. Under a shared MSE with equal weights,
`score` dominates by construction and `permits` — the target the module docstring
calls the most important — contributes almost nothing. Scaling is part of this
spec (§5, §6), not left to the training loop.

---

## 4. Head structure

Because every seat has its own `h_s` from the shared sheet encoder, the per-seat
heads are **one head evaluated four times**, not four heads. Same trick as the
encoder, same benefit: the parameters see 4× the data, and a rare situation in
seat 3 trains the same weights as a common one in seat 0.

### ⚠ The per-seat head must be CONTEXTUAL

An earlier draft applied the head directly to an isolated `h_s`. **That cannot
work**, and the reason is structural rather than a matter of accuracy: `h_s` comes
from the shared sheet encoder, which sees only *that seat's* planes and scalars.
Several of its targets are definitionally about other seats.

| target | why isolated `h_s` cannot predict it |
|---|---|
| `plan_k_first` | "did I get there **before everyone else**" — a comparison, not a property |
| `final_score` | includes plan-race premiums and temp **rank**, both cross-seat |
| `end_trigger` | an opponent ending the game truncates this seat's plans |
| `permits` | how many turns remain depends on when *anyone* triggers the end |

So the head reads the seat **and** the context:

```
z_s  = concat(h_s, h_main)          h_main = the main trunk output
out  = per_seat_head(z_s)           weights still shared across seats
```

`h_main` already mixes every seat's `h` with the global scalars, so `z_s` carries
"this seat, in this game". Weight sharing across seats is preserved — which is the
whole reason the per-seat structure is cheap — while the inputs stop being blind to
the interaction the targets are about.

The alternative (seat-indexed outputs emitted straight from the main trunk) works
too, but loses the shared-weights benefit: seat 3's rare positions would no longer
train the same parameters as seat 0's common ones.

### Per-seat head — 11 output units, applied to each `z_s = concat(h_s, h_main)`

| output | width | activation | scale |
|---|---|---|---|
| `final_score` | 1 | none | ÷80 |
| `permits` | 1 | none | ÷3 |
| `end_trigger` | 3 | sigmoid (independent) | — |
| `will_complete_plan_k` | 3 | sigmoid | — |
| `plan_k_first` | 3 | sigmoid | — (masked) |

### Global head — 5 output units, applied to the main `h`

| output | width | activation | scale |
|---|---|---|---|
| `rank_logits` | 4 | softmax, masked to seat count | — |
| `turns_left` | 1 | none | ÷25 |

`rank_logits` is a **distribution over finishing positions**, not a scalar. §5
explains why a scalar cannot work at three or four seats, and §5.1 derives the
confidence term that replaces KD's `win_value ** 4`.

### Derived, not a head

```
margin = final_score[me] − max over valid opponents of final_score[opp]
```

Exactly KD's `own_score` / `opp_score` structure, generalised to four seats. It is
derived rather than predicted for the same reason KD derives it: predicting both
sides and subtracting is better conditioned than predicting a difference, and it
gives the search per-opponent detail for free.

**Masks:** 3 per seat, for `plan_k_first`. Totals: **49 predictions, 12 masks.**

---

## 5. The objective heads

### `final_score` — per seat, ÷80

*Computes:* `state.scores()[seat]` at the terminal state.

*Why:* the primary value signal. Welcome To is near-solitaire, so MC returns on
score are low-variance — far better behaved than win/loss, especially early in
training when both seats play badly and win/loss is nearly pure noise.

**Predicted for every seat, which is what full symmetry bought.** Under the old
asymmetric encoder, asking a head to predict an opponent's absolute score from two
occupancy planes was asking for a prediction the input could not support. With the
opponent's sheet running through the same weights as ours, it is the same class of
problem as predicting our own.

*Divisor 80* from real BGA games at target strength: losing scores 46–90, winning
65–115, centring near 75–80. That puts typical play at ~0.94 and the observed
range at 0.58–1.44, with headroom above 1.0. **120 would compress everything into
0.38–0.96** and weaken the score loss against policy and auxiliaries; **60 would
push strong games to 1.9** and saturate the `tanh` in the margin blend. The
divisor is a *unit*, not a normalisation of the current policy — early self-play
will sit near 0.6 and rise. **Do not retune it as the net improves.**

### `rank_logits` — global, 4-way masked softmax

*Computes:* a distribution `p_r = P(this seat finishes rank r)`, `r = 0 .. n-1`,
softmax over 4 logits masked to the actual seat count.

An **earlier draft of this spec used a scalar `norm_rank`** and that was wrong;
the reasoning is in §5, and it is the single most important correction in this
document. The scalar version remains the definition of the *utility*, but it is
applied to the distribution at search time rather than being the head's output:

```
u_r = (n − 1 − r) / (n − 1)          1.0 for first, 0.0 for last, any table size
```

*Why a distribution:* seat-count-invariant, **collapses exactly to a Bernoulli
over win/lose at two seats**, and — the part a scalar cannot do — it carries its
own uncertainty, which §5.1 needs. It is also the fix for D1 and D4.

*Tie convention:* spread mass evenly across the tied ranks. Two seats tied for
first at 3p gives `(0.5, 0.5, 0)`. This matches `winners()` returning every tied
seat and `plan_scores` paying first-place value to all of them. **Needs a test** —
it is the kind of convention that gets silently reimplemented differently in the
loss than in the target builder.

*The utility becomes a search-time choice, free.* Because the head predicts the
distribution and the utility is applied afterwards, `u_r` can be changed **without
retraining**:

```
u = [1, 2/3, 1/3, 0]     linear rank          (arena-rating-like; the default)
u = [1, 0, 0, 0]         pure win probability (only first place matters)
```

The head answers *what will happen*; the utility encodes *what we want*.
Decoupling them means the objective can be A/B'd on a frozen checkpoint instead of
in two training runs. A scalar head bakes the utility into the weights.

A separate `win` head is **not** included: `P(1st)` is simply `p_0`.

⚠ **Masked softmax over variable support is M4 territory** (§7). Padded seats must
not receive probability mass, and the mask must be applied to the *logits* before
the softmax, not to the probabilities after — normalising over dead classes and
then zeroing them leaves the live classes summing to less than one.

### The blend

```
u_r          = (n − 1 − r) / (n − 1)
rank_value   = 2 × Σ_r p_r·u_r  −  1
margin_value = tanh((score_me − score_opp_best) / 80 × MARGIN_GAIN)
spread       = 1 − 4·Var[u]                                 Var[u] = Σ p_r·u_r² − (Σ p_r·u_r)²
floor_n      = 1 − (n+1) / (3·(n−1))                        0, 1/3, 4/9 for n = 2, 3, 4
confidence   = max(0, (spread − floor_n) / (1 − floor_n)) ** k
leaf_value   = (1 − α) × rank_value + α × confidence × margin_value
```

`floor_n` is the seat-count correction derived in §5.1; at two seats it is zero and
the whole expression collapses to `(1 − 4·Var[u]) ** k`, which is KD's gate.

`margin_value` is unchanged from KD and comes from the per-seat `final_score`
heads. **The rank representation changes only the gate**, never the margin term
and never how much weight `α` allows it.

**Absolute score alone would be denial-blind.** Ending the game a turn early when
it costs you 2 points and costs the leader 8 is invisible to it. Note that
bis-for-the-race and roundabouts-for-flexibility are *already* priced by final
score — those cases do not discriminate. Tempo denial is the one that does, and it
is exactly what the three-plan end condition creates.

### 5.1 Why the gate is variance, not `win_value ** 4`

**KD's gate is only valid at two players, and the reason is easy to miss.** With a
binary outcome the mean is a *sufficient statistic* for the whole distribution:
`E[win] = p` fully determines `Bernoulli(p)`, so `win_prob → 0.5` genuinely means
uncertain and `→ 1` genuinely means certain. The gate can read confidence off the
mean because the mean contains it.

That equivalence dies with three or more outcomes. An expectation over ranks
cannot carry its own variance:

| position | `E[u]` | scalar `win_value` | gate `w⁴` |
|---|---|---|---|
| 3p, **certain** 2nd | 0.50 | 0.00 | **0** |
| 3p, coin-flip 1st-or-3rd | 0.50 | 0.00 | **0** |
| 4p, **certain** 2nd | 0.67 | 0.33 | **0.012** |

Two states with opposite certainty get an identical gate. This is a category
error; no exponent fixes it.

**The failure is worse than imprecision — it goes gradient-dead.** Take 4p locked
into second place. The ordinal outcome is settled, so `rank_value` has **zero
gradient** and margin is the only remaining signal. The scalar gate suppresses
margin by 98% precisely there. Rank flat *and* margin gated off: the value function
is blind in a position common in the last third of every 3–4p game.

**Variance of the utility is the correct statistic.** The gate is asking "is the
ordinal outcome settled?", and that is a question about spread:

```
confidence = 1 − 4·Var[u]
```

The 4 normalises it — the maximum variance of any random variable on `[0,1]` is
0.25 — so `confidence ∈ [0, 1]`.

| position | `Var[u]` | confidence | correct? |
|---|---|---|---|
| 4p, certain 2nd | 0 | **1.0** | yes, margin fully open |
| 3p, certain 2nd | 0 | **1.0** | yes |
| 3p, coin-flip 1st/3rd | 0.25 | **0.0** | yes, chase rank |
| 4p, 50/50 first-or-last | 0.25 | **0.0** | yes |
| 4p, uniform over ranks | 0.139 | 0.44 | **no — should read 0**, see §5.1a |

**And it is not a departure from KD — it is KD, generalised.** At two seats
`u ∈ {1, 0}` with `P(u=1) = p`, so `Var[u] = p(1−p)` and

```
confidence = 1 − 4p(1−p) = (2p − 1)² = win_value²
```

exactly. KD's `w⁴` is `confidence²`. So `k = 2` reproduces KD's curve **precisely**
at two seats and generalises upward with a principled meaning. Verified numerically
across the range:

| `p` | `(1 − 4·Var[u])²` | KD `w⁴` |
|---|---|---|
| 0.50 | 0.0000 | 0.0000 |
| 0.60 | 0.0016 | 0.0016 |
| 0.75 | 0.0625 | 0.0625 |
| 0.90 | 0.4096 | 0.4096 |
| 0.99 | 0.9224 | 0.9224 |

Not approximately — identically. It is the same function, written in a form that
still means something when there are more than two outcomes.

**Entropy was considered and rejected.** It measures how many outcomes are
plausible, not how much value is at stake. A 4p distribution of
`(0.5, 0, 0, 0.5)` — a coin flip between first and last — has only two live
outcomes, so normalised entropy is 0.5 and entropy-confidence would call it half
settled. In value terms it is maximally unsettled. Variance gets it right (0.0);
entropy does not.

### 5.1a The seat-count floor — a correction

⚠ **`1 − 4·Var[u]` does not reach zero above two seats, and an earlier version of
this spec shipped the bug in its own example table** (the 4p-uniform row above,
labelled "reasonable" at 0.44 when it should read 0).

The 4 normalises against 0.25, the variance of a fair coin on `{0, 1}`. At three or
more seats the intermediate ranks pull variance down, so a **maximally uncertain**
distribution scores well above zero. For a uniform distribution over `n` ranks,
`Var[u] = (n+1) / (12(n−1))`, giving

```
floor_n = 1 − (n+1) / (3(n−1))          n = 2 → 0     3 → 1/3     4 → 4/9
```

So a completely unsettled three-seat position reports **33% confidence** and a
four-seat one **44%**, opening margin on positions where nothing whatever is
decided.

The deeper reading: the statistic measures **value spread**, not **ordinal
settledness**, and those coincide only at two seats. Rescale so it means ordinal
settledness at every seat count:

```
confidence = max(0, (spread − floor_n) / (1 − floor_n)) ** k
```

At two seats `floor = 0` and this is unchanged, so **the KD identity survives
intact** and `k = 2` still reproduces `w⁴` exactly.

**This also makes confidence comparable across seat counts**, which the raw form is
not — a fact that matters when reading §5.1b, where the 3-seat numbers look higher
than the 2-seat ones almost entirely because of this floor.

### 5.1b Measured: how decided is a Welcome To position?

Measured 2026-08-21, **empirically rather than by proxy**: fork a position, resample
the unseen deck 16 ways with `redeterminize`, play each out with GreedyBot, and take
the spread of seat 0's final rank. 14 positions × 7 checkpoint turns.

| turn | 2p `Var[u]` | 2p conf | implied `p(win)` | 3p `Var[u]` | 3p conf |
|---|---|---|---|---|---|
| 4 | 0.224 | 0.10 | 0.66 | 0.157 | 0.37 |
| 12 | 0.200 | 0.20 | 0.72 | 0.136 | 0.46 |
| 20 | 0.176 | 0.30 | 0.77 | 0.098 | 0.61 |
| 27 | 0.116 | 0.54 | 0.87 | 0.061 | 0.76 |

(`conf` here is the raw `1 − 4·Var`; apply §5.1a before comparing the columns.
The 2p `p(win)` column is exact, since `1 − 4·Var[u] = w²` at two seats.)

**The gate does real work.** By turn 12 a two-seat position is already equivalent to
a 72% win probability, and by turn 27 to 87%. This is not a game that stays a coin
flip until the end.

⚠ **A superseded measurement said the opposite, and the way it failed is worth
keeping.** The first attempt used *current score standings* as the predictor and
concluded confidence was ≈0 for ~90% of every game. In Welcome To score is a
**lagging and early-inverting** indicator: a player banking fast points is often the
one burning their sheet, while a capacity-rich player looks behind and is not. So it
measured predictability-from-score and was presented as a bound on
predictability-in-principle. The gap between them is precisely what the encoder
exists to represent.

Two caveats on the numbers above: they are **GreedyBot** playouts, and a stronger
policy converting flexibility more reliably should push confidence *higher*; and
**self-play is balanced by construction**, so games stay closer than they will
against a weaker human opponent on BGA. Re-measure with the S0 net.

### 5.2 Tuning

⚠ **`MARGIN_GAIN = 2.0` and `α = 0.5` are KD's and were tuned for a score scale of
160.** At divisor 80, margins of 5–40 points normalise to 0.06–0.5, which
`MARGIN_GAIN = 2.0` maps to `tanh(0.12 … 1.0)` — responsive without saturating, so
2.0 is a defensible starting point.

**`k = 1` is the starting default; `k = 2` is the KD-parity reference.** Against the
measured two-seat confidences of §5.1b:

| turn | `k = 1` | `k = 2` (KD parity) |
|---|---|---|
| 12 | 0.20 | 0.04 |
| 20 | 0.30 | 0.09 |
| 27 | 0.54 | 0.29 |

At `k = 2` with `α = 0.5`, margin contributes about **2% of leaf value at turn 12**.
In a game where score is the dense low-variance signal and rank is nearly flat
through the midgame, that discards the better signal for most of the game. `k = 1`
keeps margin contributing 10–27% across the same span.

This is a starting point from one GreedyBot measurement, not a tuned value. **Dump
the confidence distribution again on the S0 corpus and re-choose `k`** — that net
does not need to be good, only to exist. Plot floor-corrected confidence (§5.1a)
against turn number, per seat count.

Note this is now a question about **`k`**, not about whether the gate is coherent or
whether it earns its place. Two earlier positions in this spec are withdrawn: a
"start with no gate" hedge (withdrawn on the derivation), and a later claim that the
gate would be ≈0 throughout and therefore decoration (withdrawn on §5.1b, which
measured the opposite). The gate is well-defined at every seat count, does
measurable work from the midgame on, and is what keeps the value function alive in
locked-rank positions.

---

## 6. The auxiliary heads

### `permits` — per seat, ÷3

*Computes:* `sheet.permits` at the end.

*Why:* **the single most important auxiliary target, and the reason `training.py`
exists.** It is the low-noise signature of capacity destruction and it is
predictable from turn 3 — a short clean gradient path to the thing the score head
learns only through seventeen turns of noise. If only one auxiliary head survives,
this is it.

Per seat, it is also *"how soon will they end the game"*: three refusals is an end
condition. Its input-side twin is the refusal probability in `ENCODER_V2_SPEC`
§6.4.

### `turns_left` — global, ÷25

*Computes:* `final_turn − visit_turn`, relative to the visit turn so it is
comparable across a game. Already correct in `sample_targets`.

*Why:* the horizon variable everything else conditions on. Every "is this penalty
worth it" question is really "how many turns do I have to amortise it".

*Divisor 25, not 30 and not the maximum.* The theoretical maximum is **35** — 33
boxes at one house per turn, plus two permit refusals before the third ends it —
but a divisor's job is to put the *typical* value in range, not to bound it; there
is no activation, so occasional values above 1.0 are fine.

30 would be anchoring to GreedyBot (30.6 / 30.2 / 29.9 turns), and a strong player's
games are **shorter**: bis writes two houses a turn, roundabouts keep streets
alive, and three completed plans end it early. Unlike score, improvement moves this
target *down*, away from 1.0, so a greedy-anchored divisor leaves trained targets
small. This is the same trap as anchoring the score divisor to greedy's ~50.

Game length is largely set by the **City Plan draw**, so the realistic range is
roughly 10–35 turns. Divisor 25 puts that at 0.4–1.4 — good dynamic range. And
because the cause is *observable* (the three plans are right there in the input),
this is a genuinely learnable head whose prediction forces the trunk to represent
plan-completion pace.

### `end_trigger` — per seat, 3 independent sigmoids

*Computes:* the three clauses of `_is_end_of_game()`, evaluated **independently**
for this seat at the terminal state:

```python
full_sheet = not sheet.has_free_box()
all_plans  = all(p in self.plan_turns[slot] for slot in range(3))
max_permit = sheet.permits >= PERMIT_BOXES
```

"Did not end it" is all three at zero; it needs no class of its own.

⚠ **These are NOT mutually exclusive, and an earlier draft specified a 4-way
softmax on the grounds that they were.** `_is_end_of_game()` returns on the *first*
matching clause, which hides the overlap — but a final placement can fill the sheet
*and* enable the third plan's validation on the same turn, and more than one seat
can trigger on the same turn. A softmax would force the model to pick one, and
would be trained on whichever clause the engine's loop happened to reach first,
which is an artefact of clause order rather than a fact about the game.

Three independent bits, three independent BCE losses. If a single categorical is
ever wanted for reporting, derive it from the bits under a **documented and tested**
priority rule — do not bake the rule into the target.

*Why:* the analogue of 7WD's `joint7` — not just what happened but *how*. The modes
are **opposite in value**: ending by finishing your plans is the best outcome in the
game; ending by seizing up on a third refusal is the worst. A single
`did_I_end_it` binary averages them into noise.

It is also the free strength metric made predictable: greedy ends 20/20 on the
third permit refusal and never on plans or a full sheet, so the shift in this
distribution *is* the learning curve.

### `will_complete_plan_k` — per seat, 3 sigmoids, **unmasked**

*Computes:* did this seat ever complete plan slot k.

*Why:* the fix for D3, and the target that answers the row-planning question. The
binary is the training **label**; the head's **output is a probability** — on turn
5 with the plan half-built it might say 0.62. Same structure as KD's `win_prob`,
whose label is also binary.

Being unmasked it trains on **every** sample, including the ~86% where the answer
is no. To output a sensible number on turn 5, the trunk must encode "am I on track
for this plan" — a plan-commitment representation a score head never has to build.

Per seat, this is *"which City Plans will they finish"*. Its input-side twins are
`plans.progress()`, the requirement vectors, and `expected_turns_to_plan`.

### `plan_k_first` — per seat, 3 sigmoids, masked on completion

*Computes:* did this seat's completion turn equal `min(turns.values())` for that
slot — i.e. did it collect `scores[0]` rather than `scores[1]`.

*Why:* the first-finisher premium, worth a mean **4.27 points per plan** across the
37 plans — roughly 13 points of a ~75-point game, and the entire competitive
interaction in an otherwise solitaire game. The score head sees it only as an
unexplained lump.

`plan_scores` pays `scores[0]` to **every** player with `turn == first`, so a
same-turn tie gives both the full first-place value. The live question is therefore
never "will they beat me" but **"is this my last turn to finish at all, or my last
turn to finish alone"** — and this head is the only thing that expresses it.

Masked because it is undefined when the plan was never completed, which is exactly
why the unmasked `will_complete_plan_k` sits beside it.

---

## 7. Masking discipline

This is where a silent failure is most likely, regardless of how many targets
survive.

### M1 — normalise a masked loss by the mask sum, not the batch size

```
loss = Σ(mask · err) / max(Σ mask, 1)        NOT  Σ(mask · err) / batch_size
```

Dividing by batch size silently down-weights rare events **in proportion to their
rarity**. `plan_k_first` has mask=1 roughly 14% of the time at bootstrap, so the
batch-size version applies about a **7× hidden discount** to precisely the targets
the whole race argument says matter. The symptom is "the plan heads just don't
learn," and nothing looks broken.

### M2 — the sentinel must never reach a loss

`NEVER = -1` may appear only where `mask == 0`. Assert it in a test, for every
masked target.

### M3 — every masked target has an unmasked complement

`plan_k_first` (masked) is paired with `will_complete_plan_k` (unmasked). Any
future masked target needs the same pairing, or the model cannot represent the case
the mask removes.

### M4 — undefined at some seat count means masked, not zero-filled

Padded seats carry `seat_valid = 0` and contribute **nothing** to any per-seat
loss. Zero is a value; absent is not. Getting this wrong teaches the shared
per-seat head that a nonexistent player scores zero and never completes a plan —
and because the head is shared, that error contaminates the real seats too.

**M4 is the new one.** The shared per-seat head is what makes the encoder cheap and
what makes rare situations train well, and it is also what makes padded-seat leakage
contaminate everything. Assert that per-seat losses over a 2p game are numerically
identical whether the padded seats are filled with zeros or with garbage.

⚠ **`rank_logits` needs M4 applied to the *logits*, not the probabilities.** Mask
dead ranks to `-inf` **before** the softmax. Softmaxing over four classes and then
zeroing the dead two leaves the live classes summing to less than one, which
silently corrupts `E[u]` and `Var[u]` — and therefore the value *and* the gate,
in the same direction, so the error will not look like noise. A 2p game must
produce a two-class distribution summing to exactly 1.0. Test it.

---

## 8. Loss weighting

Weight by **group**, not per target — 44 independent knobs is not tunable.

```
policy        1.0
objective     1.0     final_score, rank_logits    (score dominant early)
capacity      0.3     permits, turns_left
outcome mode  0.2     end_trigger
plan race     0.3     will_complete_plan_k, plan_k_first
```

Consistent with `PROJECT_PLAN.md` M2: "score dominant early, policy next,
auxiliaries small." Ablate whole groups, not individual targets.

⚠ **The coefficient multiplies the group's mean, once — not each member.** The
first implementation applied it per target, which makes a group's real influence
its coefficient times its size: `components` (8 targets × 0.2 = 1.6) outweighed
`capacity` (4 × 0.3 = 1.2), and the table above described nothing. It also drifts
silently as the target set changes — step 4 below adds nine targets to
`plan race`, which would have tripled that group's pull without anyone editing a
weight. Fixed 2026-08-21; `network.losses` returns a `group_*` entry per group
holding the mean the weight was applied to, and a test asserts the
reconstruction.

---

## 9. Considered and rejected

Everything here failed the "does it force a representation the others don't" test,
or is better served elsewhere. All are **extension candidates**, gated on the S1
paired-seed harness — which exists anyway, and whose gate is plans-per-game against
greedy's 0.42.

### 9.1 The 8 score components — extension, not core

`score_parks … score_roundabouts`, each ÷80.

They sum exactly to `final_score`, so they carry **no new information about the
total**; their entire value is *attribution*, forcing the trunk to represent where
points come from. That is a real mechanism and it is what the current module
docstring argues for.

**But the evidence is thin, and there is a counter-example in-house**: Kingdomino's
score is also decomposable (territory × crowns) and KD shipped a strong player
without decomposing it. The argument that eight genuinely different subsystems is a
different situation from KD's two is reasoning, not measurement.

There is also a known weakness: a component total hides **efficiency**. Being one
box short on every park row yields a respectable `score_parks` while representing
bad board management.

⚠ **The fix for that is not an efficiency ratio.** `score / marks` gives 1.0 both
to a player with one park mark and one point and to a player with twelve of each —
division destroys magnitude and small denominators explode. If efficiency turns out
to matter, predict **numerator and denominator separately** (the ~20 final track
counts) and let the trunk form whatever combination is useful. That is strictly
more information than the ratio.

**First extension experiment: components on vs off.** If they help, the natural
follow-up is the track denominators.

### 9.2 `turns_to_plan_k` — extension

Currently implemented. Masked to ~14% coverage, and both its uses are better served
directly: the race by `plan_k_first`, the commitment by `will_complete_plan_k`. It
teaches tempo, which is real, but it overlaps heavily with two heads that train far
more often.

### 9.3 `houses`, `capacity_left` — extension

Both measure the same underlying quantity as `permits`, which the module docstring
itself identifies as the sharpest of the three. Correlated targets give diminishing
returns.

### 9.4 Per-street structure — extension

Per-street final capacity, `roundabout_in_street_x`, the final estate-size
histogram. These might force genuine row-structure representation, or they might
restate `capacity_left`. A hypothesis, not a decision. `roundabout_in_street_x` is
the most interesting of the three: it is the output-side twin of encoder planes 12
and 16.

### 9.5 Behaviour canaries — logs, not heads

Bis writes per game, roundabouts built and **the turn built**, plans completed split
by eventual winner and loser, reshuffle-choice frequency, fraction of games ending
on the plans.

These are **log lines off the game record**, not heads. Promoting a metric to a head
spends trunk capacity; you get the diagnostic for free without it. Only promote one
if you actually want the trunk to represent it.

The roundabout timing log is the important one: GreedyBot builds 1.28 per game in
advanced, and **early = treating them as scoring moves, late = treating them as
repair**. That single distribution says whether encoder plane 12 did its job.

### 9.6 A turn-a-roundabout-was-built *target* — rejected

It is not in the final state, so it would need replay instrumentation, and
`roundabout_in_street_x` captures the representation. Keep it as a log (§9.5).

---

## 10. Implementation order

1. ~~**Fix the six defects in the existing set** (§3) without adding anything —
   the rank distribution replacing `rank` and `won`, drop `first_plan`, add the mask-sum
   normalisation (M1) and the sentinel test (M2). This is the highest
   correctness-per-line change in the document and it is independent of the encoder
   restructure.~~ **DONE.** D1, D2, D4, D6 are fixed; D3 and D5 are structural and
   land with steps 2–4. `sample_targets` also emits `rank_mask_0..3` alongside
   `rank_p_0..3` — four more masks than §4's inventory, carried so that M4's
   "mask the logits, not the probabilities" rule has its data in the batch
   instead of being re-derived from a seat count at loss time.
2. ~~**Restructure `PlayerOutcome` and `sample_targets` to be seat-indexed.** Every
   target in §5–§6 is a function of the final state and the visit turn, and
   `final_outcomes` already builds `PlayerOutcome` with the whole terminal state in
   scope — **so no interface widening is needed**, only extra fields. `datagen.replay`
   currently passes `outcomes[actor]`; it will pass all seats.~~ **DONE.**
   `sample_targets(outcomes, order, turn)`: per-seat targets come back as tuples
   along `encoder.seat_order`, padded to 4 with `seat_valid = 0`, and the caller
   passes the same order it encoded with. The alignment between target seat *k*
   and encoded seat *k* is the one thing here that fails silently — every shape
   still matches when it is wrong — so it is asserted directly, along with
   `training.MAX_SEATS == encoder.MAX_SEATS`.
3. **Add the per-seat head and the global head** (§4), with the M4 padded-seat test.
4. **Add `end_trigger`, `will_complete_plan_k`, `plan_k_first`.**
5. **Port the win-gated blend** (§5), and dump the `win_gate` distribution before
   choosing the power.

Steps 1 and 2 are worth doing even if the encoder restructure slips; nothing in
them depends on symmetry except step 2's seat indexing, which is harmless on its
own.
