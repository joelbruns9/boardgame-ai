# Welcome To… — encoder v3 specification

**Status: draft, review round 1 applied 2026-08-30. Nothing here is implemented.**

Round 1 raised nine findings, all valid. Four were outright errors in the draft
(§6.1, §6.2, §6.4, §7.2) and five were underspecification (§6.4 scope, §9.3,
§9.4, §10.2, §10.3). What changed is summarised in §14; the corrected text is
inline and marked ⚠ **R1**.

v3 supersedes `ENCODER_V2_SPEC.md` §6.3 and §12 and completes §4–§7. It is a
single ABI break (`ENCODER_ABI_VERSION` 1 → 2) that lands every remaining v2
feature block plus three corrections the v2 design does not contain.

Read `ENCODER_V2_SPEC.md` first. Everything it argues about symmetry (§2), shape
(§3), what the encoder must not do (§8), rejected ideas (§10) and forward
compatibility (§11) **carries over unchanged and is not restated here**.

---

## 0. Why v3 exists

### 0.1 v2 was never built past step 2

Verified in code, both languages, 2026-08-30:

```
Python  SHEET_PLANES 12   NUM_SHEET_SCALAR 45   NUM_GLOBAL_SCALAR 358   ABI 1
Rust    SHEET_PLANES 12   NUM_SHEET_SCALAR 45   NUM_GLOBAL_SCALAR 358   ABI 1
        (encoder.rs:20-26; all five cross-checked at import, lib.rs:809-835)
absent  demand · reshuffle_contraction · refusal · plan_conflict
        effect_supply_rate · roundabout_repair · total_span
        plans block is 9 wide (v2 §6.3 specifies 48)
```

`git log -- encoder.py` has four commits: the engine, the specs, step 2
(`b72b545`, "make the encoder symmetric"), its review fixes, and the Rust port.
**No commit ever added a v2 step 3–8 feature.** The Rust port faithfully ported
the incomplete encoder, so the two languages agree — on 12/45.

### 0.2 Four observed failure modes

From live advisor play against the served S2 checkpoint, plus
`welcome-to-bga-advisor` and `welcome-to-plan-symptom-diagnosis`:

| symptom | encoder verdict |
|---|---|
| cannot finish City Plans consistently | **gap** — §2, §3 |
| opens with both roundabouts by turn 3, never holds one | **gap** — §4 |
| does not price the temp agency ±1/±2 | **gap + active defect** — §5 |
| does not maximise open span when placing | **NOT the encoder** — see below |

`box_spans` (plane 10), `placement_capacity` and `total_span` are already
computed by `sheet.py`, and `GreedyBot` beats the net comfortably using exactly
those three terms. ⚠ **Three different GreedyBot means are quoted in this spec and
they are different corpora — do not mix them.** `bots.py:49-72` **45.8**, advanced,
*one seat*, 150 paired seeds, its own term ablation. `welcome-to-plan-symptom-
diagnosis` **50.8**, advanced, the 60/30/10 seat mix, 2–4p. `welcome-to-bga-advisor`
**54.4** advanced / **51.4** base, 25 paired seeds, 2 seats. §11.1's base-game
comparison is the last of these. The net has that signal and does not use it. That is the S0
bootstrap failure recorded in `runs/welcome_to_s0/s0_metrics.json` — `net_score`
21.83 against `greedy_score` 51.48, `policy_top1` 0.475 — and **no encoder
change addresses it.** It is tracked separately and must be re-gated after v3
lands (§11.2).

### 0.3 Three corrections v2 does not contain

These are why v3 is a new document rather than "finish v2".

1. **v2's requirement vector is estate-blind** (§2). 18 of the 28 dealt plans
   are estate plans across 18 distinct size multisets; v2's 7-wide vector has no
   slot for estate sizes and collapses all eighteen to "N fences needed".
2. **v2's requirement vector is locus-blind** (§3). 9 of the 10 non-estate plans
   are street- or box-bound; a scalar count of "parks needed" cannot say *which
   street*, and for `CompleteStreet` cannot say *the same street*.
3. **Plane 8 conflates temp-reachable with free** (§5). A live defect in shipped
   code, not an omission.

### 0.4 The checkpoint is discarded

Decided by the user, 2026-08-30. The existing net has served its purpose
(demonstrating that the loop learns); its weights and dimensions are not
preserved. So v3 needs **no** checkpoint migration, **no** WTS
back-compatibility and **no** legacy head zero-fill. Existing shards are
regenerated from scratch. This is why A–E migrate at once rather than in five
ABI steps.

---

## 1. Shape

| | v2 shipped | v2 spec | **v3** |
|---|---:|---:|---:|
| planes per seat | 12 | 17 | **22** |
| scalars per seat | 45 | 127 | **202** |
| global scalars | 358 | 380 ⚠ | **367** |
| viewer plane | 1 | 1 | 1 |

⚠ v2's global total of 380 is stale: it predates the step-2 restructure that
moved the per-seat banked flags out of `plan_identity` (99 → 93) and shrank
`reshuffle_race` (4 → 2). The correct v2-complete figure is 372, and v3 adds one
global block beyond v2's list (`turns_to_reform`, §7.4).

```
sheet_planes    (4, 22, 3, 12)   one block per seat, identical function
sheet_scalars   (4, 202)         one block per seat, identical function
viewer_plane    (1, 3, 12)       phase scratch, viewer only
global_scalars  (367,)           game-wide and viewer-relative
```

**Cost.** The shared sheet encoder's input goes 477 → 994 floats, so its first
layer goes 122k → 254k parameters; `trunk_in` gains 12k. Total **+144k on ~4M,
i.e. +3.6%** — consistent with the measured finding that the sheet encoder is
~5% of this network. Throughput is not a consideration here.

---

## 2. Correction 1 — the requirement vector must carry estate sizes

### 2.1 The defect

v2 §6.3 specifies, per plan slot:

> requirement vector | 7 | parks, pools, temps, bis, houses, fences, roundabouts needed

There is no slot for estate *sizes*. But:

```
dealt plans by kind: ESTATE 18, DECORATIVE 4, FULL_STREET 2,
                     FIVE_BIS 1, SEVEN_TEMP 1, EXTREMITIES 1, COMPLETE_STREET 1
18 distinct required-size multisets:
  (1,1,1,1,1,1) (1,1,1,6) (1,2,2,3) (1,2,6) (1,4,5) (2,2,2,2) (2,3,5) (2,5)
  (3,3,3) (3,3,4) (3,4) (3,6) (4,1,1,1) (4,4) (4,5) (5,2,2) (5,5) (6,6)
```

Under v2 as written, `(1,1,1,6)`, `(3,3,4)` and `(2,5)` all encode as "fences
needed: N". Building v2 §6.3 exactly as specified would not fix the loudest
observed symptom.

### 2.2 What is and is not already there

`plans.progress()` performs the correct multiset match
(`min(need[size], supply[size])` over `free_estates()`), so `steps_left` is
**right**: it says "two estates short". Nothing says *which two*.

The encoder has the magnitude of the gap and none of its direction.

⚠ **And the one size-resolved feature that exists is the wrong one.** The
`tracks` block feeds `sheet.estate_size_counts()`, which counts **all** estates
including those a completed plan has already consumed — it is the scoring helper
(`sheet.py:493`, `RealEstate::getAssocSizeNumber`). Plan eligibility runs off
`free_estates()`. So even the indirect route the net could in principle learn is
fed a mismatched vector.

### 2.3 The fix

**Per plan slot, 6 scalars: `estate_shortfall[s]` for s in 1..6.**

```
need         = Counter(plan.required_sizes)          # () for non-estate plans
supply       = Counter(size for _, _, size in sheet.free_estates())
shortfall[s] = max(0, need[s] - supply[s]) / 6.0
```

Zero for every non-estate plan, which is correct and costs 6 dead floats on 10
of 28 plans — cheap against making the other 18 legible.

**And in `tracks`, add 6: free-estate size counts.** Keep `estate_size_counts()`
(it is the scoring quantity and the score head needs it); add
`free_estate_size_counts()` alongside. New helper in `sheet.py`:

```python
def free_estate_size_counts(self) -> list[int]:
    """Estates of size 1..6 that no City Plan has consumed."""
```

`tracks` 20 → 26.

---

## 3. Correction 2 — the requirement vector must carry the locus

### 3.1 The defect

Checked against `plans.can_be_scored` and `plans.progress` for all 10 non-estate
dealt plans:

| plan | what it actually requires | v2's 7-vector says | lost |
|---|---|---|---|
| 21 `SEVEN_TEMP` | 7 temps, sheet-wide | `temps: N` | **nothing — exact** |
| 18/19 `FULL_STREET(x)` | every box of street *x*; **dead** if any `top_fences[x]` | `houses: N` | which street; deadness |
| 22 `EXTREMITIES` | the 6 boxes in `EXTREMITY_POSITIONS`; ends want the scarce 1/2/14/15 | `houses: N` | which boxes; scarcity |
| 20 `FIVE_BIS` | 5 bis **in one street** (`bis_count_per_street`) | `bis: N` | the concentration |
| 23 / 25 `DECORATIVE park` / `pool` | all parks (pools) in the **best 2 of 3** streets | `parks: N` | which two |
| 26 / 27 `pool&park(x)` | parks **and** pools in street *x* | `parks: N, pools: M` | which street |
| 24 `COMPLETE_STREET` | parks + pools + roundabout **in the same street** | three counts | the co-location |

**Exactly one of 28 dealt plans is fully captured by v2's requirement vector.**

The mechanics this rests on, verified in `constants.py` / `sheet.py`: parks are
street-bound (`self.parks[x]`, `PARK_BOXES = (3,4,5)`), pools are both box-bound
(`POOL_POSITIONS`, nine printed locations) and street-tracked (`self.pools[x]`),
and `has_roundabout_in_street(x)` is street-bound.

As with estates, `progress()` already does the street-aware arithmetic correctly
— best-2-of-3 for decorative, min-over-streets for `CompleteStreet`,
max-over-streets for `FiveBis`. The loss is again **direction, not magnitude**.

> **The unifying statement of both corrections: v2's requirement vector answers
> "how much" and never "where".**

### 3.2 The fix — 18 per-street scalars per plan slot

Per plan slot, for each of the 3 streets, 6 scalars:

| field | definition | plans it serves |
|---|---|---|
| `parks_needed[x]` | `(PARK_BOXES[x] - parks[x]) / PARK_BOXES[x]`, 0 if the plan wants no parks | 23, 24, 26, 27 |
| `pools_needed[x]` | `(3 - pools[x]) / 3`, 0 if the plan wants no pools | 24, 25, 26, 27 |
| `houses_needed[x]` | empty boxes in street *x* this plan needs written, ÷ `STREET_SIZES[x]` | 18, 19, 22 |
| `bis_needed[x]` | `max(0, 5 - bis_count_per_street()[x]) / 5` | 20 |
| `roundabout_needed[x]` | 1.0 if the plan needs a roundabout in *x* and there is none | 24 |
| `street_serves[x]` | ⚠ **R5** 1.0 if street *x* is **alive** for this plan — i.e. not provably unable to contribute. A **completed** street is alive (1.0); remaining work is carried by `parks_needed[x]` etc. | all |

`street_serves` is the field that carries what a summed count destroys:

* `FULL_STREET(2)` → `(0, 0, 1)`, and `(0,0,0)` once `any(top_fences[2])` — the
  plan is **dead**, which no distance feature can express;
* `DECORATIVE park` → 1.0 on every street that can still serve the plan,
  **including one whose parks are already complete** — a completed street is the
  one contributing *most*, and the draft's gloss ("streets whose parks are not yet
  complete") read it as 0, contradicting the aliveness definition three lines
  above. The best-2-of-3 selection stays visible because `parks_needed[x]` carries
  the remaining work; that split is the whole §3 argument;
* `pool&park(1)` → `(0, 1, 0)`;
* `SEVEN_TEMP` → `(0, 0, 0)` — correctly, it is sheet-wide.

### 3.3 The fix — 3 plan-target planes

Two plans are **box**-bound, not merely street-bound, and no scalar reaches
them. Add one plane per plan slot:

> **plane `19 + k`: "plan *k* still needs a house written in this box."**

* `FULL_STREET(x)`: every empty box of street *x*
* `EXTREMITIES`: the unwritten boxes of `EXTREMITY_POSITIONS`
* every other kind: all zero

This is what lets `EXTREMITIES` interact with planes 14/15 (fit probability): the
two together say "I need `(2,11)` and the deck is nearly out of 14s and 15s",
which is the worked case v2 §6.3 raises and then does not encode.

⚠ Do **not** extend these planes to estate plans by marking boxes whose filling
"would create a needed size". That is a search, not a feature; it is expensive
per box, and it embeds a behavioural assumption — the exact category-2 line v2 §8
draws. Estate direction is carried by `estate_shortfall`, and the exact question
is answered by `can_complete_this_turn` (§6.4).

### 3.4 The resulting plan block — 3 slots × 36

| field(s) | n | source |
|---|---|---|
| `fraction`, `steps_left` | 2 | `plans.progress()`, unchanged |
| `banked` | 1 | `state.plan_turns_for(viewer, slot)`, unchanged |
| sheet-wide requirement: `temps_needed`, `fences_needed` | 2 | §3.2 |
| `estate_shortfall[1..6]` | 6 | §2.3 |
| per-street requirement, 3 × 6 | 18 | §3.2 |
| `feasible` | 1 | §6.1 |
| `turns_lower_bound` | 1 | §6.2 |
| `effect_rate_turns` | 1 | §6.3 |
| `number_rate_supply` | 1 | §6.3 |
| `expected_turns_to_plan` | 1 | §6.3 — **ablation target** |
| `can_complete_this_turn` | 1 | §6.4 |
| `p_complete_next_turn` | 1 | §6.4 |
| | **36** | |

`banked` is carried over from the shipped implementation; v2 §6.3's table
omitted it, which was an oversight — whether this seat has already scored the
slot changes the meaning of every other field in it.

---

## 4. Roundabouts — planes 13 and 17, and the repair scalars

The measured verdict (`welcome-to-bga-advisor`, 25 paired seeds) is that
roundabouts are worth **+18 points** and that masking them collapses the net from
43.6 to 25.2. The symptom is not that the net builds them; it is that it **cannot
represent holding one**, so the only value it can find is spending them
immediately.

`box_spans` correctly reports the boxes to the right of a badly placed high
number as dead. A banked roundabout resurrects them entirely. Without a feature
for that, the option is invisible and the deferred line is unlearnable.

**Plane 13 — `span_if_roundabout / 18`.** For each empty box *b*, the largest
span *b* could have if one roundabout were placed at another empty box in the
same street. A roundabout removes the bound on the side it sits, so only the
nearest empty box on each side of *b* can matter; the naive max over ≤33 sites is
also affordable.

**Plane 17 — `P(fit | deck, no temp, after an optimal roundabout)`.** Plane 14
evaluated against plane 13's bounds instead of `box_spans`'.
`plane 17 − plane 14` is the option value of the banked roundabout, in
probability units.

**Scalars — `roundabout_repair`, 3.** `capacity_if_roundabout()[x]` per street:
`placement_capacity` recomputed with the best legal roundabout placement in that
street.

⚠ **R5 — both planes must collapse when no roundabout remains.** §6.1 already
guards its post-roundabout death tests on `roundabouts < ROUNDABOUT_BOXES and
config.advanced`; §4 said only that they are zero in a base game. In an advanced
game with **both roundabouts spent**, plane 13 must equal `box_spans` and plane 17
must equal plane 14 — otherwise `plane 17 − plane 14` reports option value for an
option that no longer exists, which is the same lie §5.2 is deleting from plane 8.

These say nothing about whether the play is good; the −3 / −8 penalty is already
in the score block, and the policy learns the trade. Both planes collapse in a
base-rules game, where roundabouts do not exist — which is why the base game is
the right place to measure §2 and §3 (see §11.1).

New `sheet.py` helpers:

```python
def span_if_roundabout(self) -> list[list[int]]: ...
def capacity_if_roundabout(self) -> list[int]: ...
```

---

## 5. Correction 3 — the temp agency, and a live defect

### 5.1 The defect

`encoder._offered_numbers` extends its result with `state.numbers_for(number,
effect)`, which applies `TEMP_DELTAS = (0, -2, -1, 1, 2)` and clamps. Planes 8
(`P_WRITABLE`) and 11 (`P_FIT`) are computed over that union.

**So a box reachable only by spending a temp is marked identically to a box that
is free.** The net sees temp-reachable capacity as free capacity. That is a lie
in the input, not an omission — and the temp is scarce (`TEMP_BOXES = 11`),
scoring, and contested (`TEMP_RANK_SCORES = (7, 4, 1)`).

### 5.2 The fix

Split the plane. The two are disjoint and their union is today's plane 8, so no
information is lost and one bit is gained:

* **plane 8 — `writable_no_temp`**: writable with some offered combination using
  delta 0.
* **plane 12 — `writable_temp_only`**: writable with some offered combination,
  but only for a non-zero delta.

Plane 11 (`positional_fit`) is computed over the **delta-0** numbers only; the
temp-widened fit is recoverable from planes 14/15 and does not need a second fit
plane.

### 5.3 The price of a temp becomes a subtraction

With v2's planes 14 and 15 in place:

```
plane 14 = P(fit | deck, no temp)      over (low,   high  ) ∩ [1,15]
plane 15 = P(fit | deck, temp ±2)      over (low-2, high+2) ∩ [1,15]
```

`plane 15 − plane 14` is the per-box value of holding a temp, in probability
units, and `card_demand_temp − card_demand_notemp` (§7.3) is its sheet-wide form.
Neither exists today.

⚠ **Do not mix temp and bis in one formula.** Temp governs *which values are
reachable* (§5, §7); bis governs *how many houses fit* (§8). The roundabout is
the one mechanic in both, which is why v2's original single "placement" quantity
failed review.

---

## 6. Plan speed, feasibility and threat

This is v2 §6.3's second half, unchanged in intent. Restated here only where v3
sharpens it.

### 6.1 `feasible` — sound, not complete

**Feasibility is not distance.** `progress()` answers "how far?"; it cannot
answer "is this still possible?". An estate plan wanting a size-3 estate is dead
if every remaining run is welded to length 5 by a bis pair, because
`surveyor_zones` refuses to split equal neighbours. An estate of 7+ scores zero,
so a long bis run is a trap. A net must learn a fence-legality rule *and* a
combinatorial reachability argument to see that. It will not, reliably.

⚠ **This is the one block that can be silently wrong, so it is specified as a
one-sided test.** `feasible` returns `0.0` **only when the plan is provably
unreachable**, and `1.0` otherwise. It is *sound* (no false deaths), not
*complete* (it will miss some real deaths). A false death is a feature that lies;
a missed death is a feature that is merely weak. §10.1 is the blocking test for
soundness.

Provable-death checks, by kind:

⚠ **R1 — every death check must be evaluated after roundabout repair.**
`build_roundabout` calls `write(ROUNDABOUT, pos)` (`sheet.py:409-421`), so a
roundabout **occupies the box as a written house** and **fences both its sides**,
and `available_locations(None)` ignores numeric fit entirely. Three consequences
the first draft missed:

* `placement_capacity()[x] <` empty boxes is **not** death — a roundabout removes
  a bound and `capacity_if_roundabout()[x]` (§4) can exceed it;
* a `box_spans == 0` extremity box is **not** dead — a roundabout can be written
  there, and `can_be_scored(EXTREMITIES)` tests `numbers[x][y] is not None`, which
  the sentinel satisfies;
* a dead pool position is **not** dead — a roundabout placed *elsewhere in the
  street* repairs its span.

So every span- or capacity-derived death test below reads the post-roundabout
helper whenever `roundabouts < ROUNDABOUT_BOXES` **and** `config.advanced`. In a
base-rules game roundabouts do not exist and the plain helpers are used.

⚠ **R5 — and the same is true of BIS, which R1 through R4 all missed.**
`Sheet.bis_candidates` (`sheet.py:311-338`) requires only that the box be empty,
that an unfenced neighbour hold a non-roundabout number, and nothing else —
**there is no ascending-order check**. A bis therefore fills a box no drawn
number could legally reach, exactly like a roundabout. A street written
`1 _ 2 _ 3 _` has `placement_capacity` 0 and every gap `box_spans` 0, and every
one of those gaps is bis-writable from either side.

Loose sound bound, in the R4 style:

```
bis_reach(x) = |{ empty b in street x : b is bis_reachable }|
```

⚠ **R6 — the bis track SATURATES, it does not GATE, so `BIS_BOXES` must not
appear in any bound.** `legal_actions` offers `bis_candidates()` at `ACTION_BIS`
**without consulting `bis_marks`** (`game.py:1009-1014`), and the apply path
writes the house and then does `bis_marks = min(bis_marks + 1, BIS_BOXES)`. So a
sheet at nine marks keeps placing bis houses — now *free of further penalty*,
which makes them strictly good — and any death test subtracting `bis_marks` from
`BIS_BOXES` under-counts reachability. R4 introduced exactly that term as a
"fix"; it was unsound.

This is the **temp agency's saturating track a second time** (§5.3: past eleven
marks a temp is free but worthless). Two of the six tracks saturate rather than
gate. ⚠ **Before writing any bound over a track, check whether `legal_actions`
actually reads its counter** — `PERMIT_BOXES` and `ROUNDABOUT_BOXES` do gate
(`can_take_permit`, `can_build_roundabout`); `BIS_BOXES` and `TEMP_BOXES` do not.

`bis_reachable(b)` is deliberately permissive: an *empty* neighbour counts (it
may be written later and a bis copies whatever lands there) and an unfenced slot
counts (fences are only added, never removed). Only a roundabout neighbour and an
already-fenced side are excluded.

This was the third consecutive round in which a house-writing action that
bypasses numeric fit broke a death test. **Any future span- or capacity-derived
claim must be checked against all three: writes, roundabouts and bis.**

⚠ **R2 — `FULL_STREET` is a house-reuse plan, and that is a cross-plan death.**
Confirmed against `Plans/FullStreetPlan.php` and our `plans.py:197`:

* **every box of the street must be filled, and a roundabout counts** — the PHP
  says so in as many words ("roundabout also works"), and `can_be_scored` tests
  `n is not None`, which the `ROUNDABOUT` sentinel satisfies;
* **the street may hold estates of any size** — no size condition appears;
* **but no house in the street may already be spent on another City Plan.**
  `canBeScored` requires every `TopFence` in that street to be null, i.e.
  `not any(top_fences[x])`. So completing an *estate* plan that spends an estate
  lying in street *x* permanently kills `FULL_STREET(x)`;
* **and once it completes, no fence can ever be added to that street again** —
  `validate()` top-fences all of street *x*, and `Surveyor::getAvailableZones`
  refuses any slot between two top-fenced houses (`sheet.py:366-385` matches).
  Emergent from the top-fence rule, not a separate one, and already correct in
  our engine.

⚠ **R3 — `EXTREMITIES` is the second house-reuse plan, and it is the more
destructive of the two.** `Plans/ExtremitiesPlan.php` carries the identical
warning ("you cannot re-use a house already used for another city plan"),
`canBeScored` requires all six `EXTREMITY_POSITIONS` un-top-fenced, and
`validate()` top-fences all six. Our `plans.py` matches.

The consequence is larger than "six houses spent", and it is not obvious from
the rule text: **an extremity box is the first or last box of a street, so
top-fencing it removes the entire estate containing it from `free_estates()`.**
Completing `EXTREMITIES` can therefore delete up to **six** estates — both end
estates of all three streets — from every estate plan's supply at once. On the
base board every plan is an estate plan (§11.3), so in an advanced game where
`EXTREMITIES` is dealt alongside two estate plans this is frequently the single
biggest decision on the sheet.

These two points are what the encoder must carry: a **conflict between two slots
on one seat's sheet**, invisible to any per-plan distance and invisible to a
global plan-pair table. They are the concrete cases `plan_conflict_seat` (§9.2a)
exists for, and `EXTREMITIES` is why its `kills[a,b]` field is evaluated by
actually re-running `feasible` rather than by counting shared boxes.

| kind | dead when |
|---|---|
| `ESTATE` | for some size *s*, `need[s] > free_estates[s] + reachable_new[s]` — where ⚠ **R5** `reachable_new[s]` must admit **re-partitioning of already-built runs**, not only new houses (see below) |
| `FULL_STREET(x)` | ⚠ **R5** `any(top_fences[x])` — including a top fence placed by *another* plan. **The capacity clause is deleted**: with roundabouts and bis both able to fill numerically-dead boxes, no cheap capacity bound is sound, and `any(top_fences[x])` is *exact* on its own |
| `EXTREMITIES` | some extremity box is `top_fenced` — including by *another* plan; or ⚠ **R5** it is empty with `span_if_roundabout` 0 **and** no roundabout remains **and** it is not bis-reachable (an extremity is adjacent to box 1 of its street, so a bis copying that neighbour writes it) |
| `FIVE_BIS` | ⚠ **R6** `max_x ( bis_count_per_street[x] + bis_reach()[x] ) < 5`. **No `BIS_BOXES` term** — see below. R4's `min(BIS_BOXES − bis_marks, …)` was unsound, and R5's undefined "bis-able free boxes" would have read `bis_candidates()`, unsound the other way because ordinary future writes create new candidates |
| `SEVEN_TEMP` | `TEMP_BOXES < 7` — i.e. never; kept for uniformity |
| `DECORATIVE` / `COMPLETE_STREET` | the required street(s) cannot complete: pools need free `POOL_POSITIONS` in that street with non-zero `span_if_roundabout`; a roundabout requirement needs `roundabouts < ROUNDABOUT_BOXES` |

⚠ **R5 — `reachable_new[s]` counting only free boxes is unsound.**
`Sheet.estates` (`sheet.py:426-446`) bounds an estate by **fences**, not by
writes, so one SURVEYOR fence creates estates of new sizes while consuming **no
free box at all**. Reachable counter-example:

> Plan 5 requires `(6, 6)`. Street 2 is fully written and unfenced: one estate of
> size 12, so `free_estates[6] = 0`. The rest of the sheet is full, so free boxes
> are 0. §13.2's resolved bound declares the plan **dead** — and a single fence at
> slot (2, 5) splits the run into 6 + 6 and completes it.

This is the identical fact §6.4 uses to justify "no early exit" on the estate
ceiling. It was applied to the ceiling and not to the death test.

**Corrected loose bound, still obviously sound:** for every maximal built run of
length `L` that is not fully top-fenced and is not welded solid by bis pairs,
every `s ≤ L` is reachable; plus the free-box term for runs that can still grow.
`reachable_new[s]` is the count of runs admitting `s`. Cheap — a single pass over
`estates()` and `surveyor_zones()`.

### 6.2 `turns_lower_bound`

⚠ **R1 — "one required box = one turn" is wrong and was in the first draft.** A
single turn can place **three** houses (§8: `roundabout → choose + write → bis`),
so a `FULL_STREET` needing three boxes can finish this turn. The house-derived
term must be divided by the house ceiling.

A hard bound, sound by construction:

```
effect_term = Σ_e ( marks of effect e still needed )      # ⚠ R5: SUM, not max
house_term  = ceil( houses this plan still needs / 3 )     # 3 = absolute per-turn ceiling
turns_lower_bound = max(effect_term, house_term)
```

⚠ **R5 — `effect_term` sums; the draft maxed.** One combination per turn applies
exactly **one** effect mark (`_EFFECT_PHASE`, `game.py`), so the sum is a valid
lower bound and is strictly tighter than `max(max_e marks, #distinct effects)`.
§6.3 already makes this argument three paragraphs later — "summed rather than
maxed, because needing both parks and fences is competing with yourself" — and
§6.2 contradicted it. Both are sound; the draft was needlessly weak on the one
feature sold as a *hard bound*.

⚠ **R4 — `fences_needed` must NOT enter `effect_term` as `steps_left`.** A single
SURVEYOR fence can raise the estate match by two (§6.4), so "one missing estate =
one fence" is not a lower bound and would make `turns_lower_bound` unsound in the
one direction that matters. For `ESTATE` plans the fence contribution is

```
estate_fence_term = 0 if steps_left == 0 else 1
```

— weak, but sound, which is the correct trade for a hard bound. `fences_needed`
survives in the §3.4 requirement vector as a **descriptor** of remaining work; it
is never an operand of a bound. Tightening it needs the same reachability
argument as §13.2 and is deferred with it.

`3` is the constant absolute ceiling, not `max_houses_this_turn` — the latter is
a *current-state* quantity and using it would make a "lower bound on turns"
fluctuate with this turn's offer. Pure arithmetic, no behaviour.

### 6.3 Rate, not just distance

The effect column sums are strongly non-uniform — SURVEYOR 18, PARK 18, ESTATE
18, but POOL 9, TEMP 9, BIS 9 out of 81 — so "three steps away" means very
different things depending on which effect those steps need, and it shifts as the
deck depletes.

```
effect_rate_turns      = Σ_e (marks of e still needed) / effect_supply_rate[e]
number_rate_supply     = fraction of the deck supplying this plan's number needs
expected_turns_to_plan ≈ effect_rate_turns, combined with number_rate_supply
```

Summed rather than maxed, because one combination per turn means needing both
parks and fences is competing with yourself.

⚠ `expected_turns_to_plan` is the one feature here embedding a behavioural
assumption ("takes the needed effect whenever offered"), which puts it closest to
the v2 §8 category-2 line. It goes in because nets are demonstrably bad at
exactly this arithmetic composition, and it is a **first-class ablation**.
`effect_rate_turns` and `number_rate_supply` are clean facts and stay regardless.

### 6.4 Threat — `can_complete_this_turn`, `p_complete_next_turn`

⚠ **R1 — scope: `config.standard` only.** The premise "all seats see the same
three stacks" is `Globals::isStandard` (`game.py:174-177`), i.e. **not expert and
not solo**. In expert mode stacks are per-player and private, so computing an
opponent's predicate from *their* stacks leaks, while computing it from the
viewer's offer answers a different question; and `known_next_effects` is all-zero
in both expert and solo (`deck_knowledge.py:186-196`), so `p_complete_next_turn`
loses its certainty term too.

Resolution: expert stays out of scope entirely (v2 §11.3) and the encoder raises
on it. In solo there are no opponents to threaten, so both scalars emit `0.0` for
every seat. **The predicate is computed only when `state.config.standard`**, and
§10.2 asserts that.

Under `standard`, all seats see the same three stacks and opponents' sheets are
public, so "could seat *p* complete plan *k* this turn?" is a deterministic
predicate over information the viewer is entitled to have.

It matters because of the tie rule: `plan_scores` pays `scores[0]` to *every*
player whose completion turn equals the first, so a same-turn tie gives both the
full first-place value. The live question is never "will they beat me" but **"is
this my last turn to finish at all, or my last turn to finish alone"**.

⚠ **"Supplies the needed resource" is not a sound predicate** — it is
resource-blind where the plans are position-bound, which §3.1 now tabulates in
full. **Use bounded exact enumeration:**

⚠ **R1 — `steps_left >= 3` is an unsound early exit and was in the first draft.**
§8 establishes that one turn places up to three houses, so a `FULL_STREET` three
empty boxes from done completes *this turn*. The cutoff must be a **sound
per-turn progress ceiling**, kind-specific:

```
def one_turn_ceiling(plan, state, seat) -> int:
    ESTATE          -> len(plan.required_sizes)   # i.e. NEVER early-exit; see below
    FULL_STREET     -> 3      # roundabout + write + bis
    EXTREMITIES     -> 3      # same, and a roundabout satisfies an extremity box
    FIVE_BIS        -> 1      # one BIS mark per combination
    SEVEN_TEMP      -> 1      # one TEMP mark per combination
    DECORATIVE      -> 1      # one PARK or POOL mark per combination
    COMPLETE_STREET -> 2      # one mark, plus a roundabout in the same turn
```

```
if steps_left > one_turn_ceiling(plan, state, seat):   return 0.0
else:                enumerate this turn's legal action sequences,
                     filter to those touching this plan,
                     test plans.can_be_scored on the result
```

⚠ **R4 — the ESTATE ceiling of 3 was unsound; it is now "no early exit".** The
draft reasoned from houses, but estate progress is measured in *matched estates*,
and a **single fence can change the match by two**: a fenced run of 6 with a plan
needing `(3,3)` scores 0 matched, and one SURVEYOR fence in its middle scores 2.
A turn supplies up to **three** fences — `build_roundabout` fences *both* sides
(`sheet.py:409-421`) and SURVEYOR adds one — so the reachable jump is far above 3.
Since `steps_left = len(required) − matched ≤ 6` for every estate plan, "no early
exit" costs at most a ceiling of 6 and is the only obviously sound choice. The
cache carries the cost.

The ceiling is a *bound*, not a prediction: it may admit an enumeration that
finds nothing, which is correct and cheap. §10.2 tests the ceiling itself over
**all** `steps_left`, not only `≤ 2` — a ceiling that is too low is exactly the
silent false negative this finding caught.

Cap the enumeration and cache on `(viewer, seat, sheet hash, offer hash)`; the
cache is what makes the per-seat symmetric version affordable. ⚠ The key
**must** include the viewer, or one seat's answer is served to another under a
different information set.

⚠ **R1 — `p_complete_next_turn` needs the joint draw, not the marginal.**
`next_number_distribution` (`deck_knowledge.py:181`) is a marginal over one card.
The three stacks reveal **three different cards drawn without replacement**, so
their numbers are correlated and "P(any stack supplies the step)" is not a
function of the marginal.

⚠ **R5 — "certain" is conditional on no reshuffle firing, and the encoder cannot
see whether one will.** `_reveal_step` runs `_reshuffle_decks` *before*
`_draw_step`; `_reshuffle_decks` (`game.py:576-592`) reforms the deck, draws a
fresh `stack_new`, and runs a **second** `_discard_step` — so the cards currently
on show never become asides and `next_effects` (which reads `stack_new`,
`game.py:875-891`) describes effects that will not be offered.

⚠ And the trigger is **not readable**: `reshuffle_next_turn` is the table-wide OR
of concurrent hidden votes, and reading it would leak that an earlier actor
completed a plan — the exact thing `plan_turns_for` hides. The encoder may read
only `reshuffle_vote_for(viewer)`.

So both `S_i` (§6.4) and `F_i` (§7.5) branch on the viewer's own vote:

* `reshuffle_vote_for(viewer)` **false** — proceed as specified; the effects are
  certain *unless another seat votes yes*, which is genuinely hidden and which
  search and the value head absorb;
* `reshuffle_vote_for(viewer)` **true** — the viewer knows a reshuffle fires.
  Effects are then unknown, so fall back to the **effect-marginal** form: build
  `F_i` from the no-temp interval and evaluate against
  `after_reshuffle_composition`. State the fallback in the docstring rather than
  pretending to certainty.

⚠ **R2 — the certain-effects case is far cheaper than R1's framing implied.** `known_next_effects` reads them off the card
corners (v2 §7.3), so only the three **numbers** are unknown. That collapses the
enumeration from "ordered draws over the 15 × 6 number-effect composition" to an
exact sum over **15³ = 3,375 ordered number-class triples**:

```
S_i = { printed numbers n : with stack i's KNOWN effect e_i, some legal
        one-turn sequence starting from combination (n, e_i) completes the plan }
      -- at most 15 one-turn enumerations per stack, cached on (sheet, n, e)

P(none supplies) = Σ over (n1,n2,n3), n_i ∉ S_i, of
                     c[n1] · (c[n2] − [n2=n1]) · (c[n3] − [n3=n1] − [n3=n2])
                   ÷ ( D · (D−1) · (D−2) )

p_complete_next_turn = 1 − P(none supplies)
```

Falling factorials, not products of marginals — that is what makes it a joint
draw. Exact, deterministic, and identical in both languages.

The real cost is the ≤45 one-turn enumerations per `(seat, slot)` that build the
`S_i`, and they cache hard: most numbers give the same answer, and the cache key
`(viewer, seat, sheet hash, number, effect)` is shared across slots.

Three further requirements, all of which the marginal form silently got wrong:

* **skip the solo marker.** `SOLO_CARD_ID` sits in the deck in solo mode and is
  not a construction card; it resolves and forces another draw (`game.py:471`).
  Moot under the §6.4 scope rule (solo emits `0.0`), but the helper must not
  assume it away.
* **model the reform.** When fewer than three construction cards remain, the
  discard is reformed mid-reveal (`game.py:436, 455`), so the composition for the
  later draws is the post-reform pool, not the depleted deck. See §7.4 — this is
  not a rare corner, it is most of a game.
* **effects are per stack.** The temp widening applies only to a stack whose
  known next effect is TEMP, so `S_i` is built per stack, not per number.

⚠ **R4 — both threat scalars are 0.0 for a seat that has already banked the
slot.** `plans.can_be_scored` deliberately tests only the sheet; the "has not
already scored" half lives in the game, not the plan (`plans.py` docstring). So a
non-consuming plan stays true forever — `SEVEN_TEMP` is satisfied for the rest of
the game once `temps >= 7` — and a naive enumeration would report a permanent
completion threat from a player who cannot score it again. Short-circuit both
scalars on `seat in state.plan_turns_for(viewer, slot)` **before** enumerating.
The `banked` flag in §3.4 does not save us here: it would be the net's job to
multiply two features, which is what §7.5 establishes we do not rely on.

**Honest limitation, to be in the docstring:** this answers "could they", not
"did they". Whether a seat has already acted this turn is genuinely hidden and
correctly so. Search and the value head absorb the rest.

---

## 7. Fit probability, demand and the reshuffle

Carried from v2 §4.2, §6.1, §6.2 with the §9 renumbering and one correction.

### 7.1 Planes 14–16, per gap, broadcast

Exact, not an approximation: `gap_bounds` bounds each empty box by the nearest
written number left and right, and for every box inside a maximal empty run those
bounds are identical. Compute once per gap (≤ ~10 per sheet), broadcast.

With exclusive bounds `(low, high)` — sentinels `low = MIN_NUMBER - 1 = -1`,
`high = MAX_NUMBER + 1 = 18`, a roundabout or street end removing a bound — and
`c[n]` the deck count of printed card number `n`, `D = deck_remaining`:

```
plane 14  P(fit | deck, no temp) = ( Σ_{n ∈ (low,high) ∩ [1,15]} c[n] ) / D
plane 15  P(fit | deck, temp ±2) = ( Σ_{n ∈ (low-2,high+2) ∩ [1,15]} c[n] ) / D
plane 16  P(fit | reshuffled)    = as plane 14, over c + discard + aside,
                                   ÷ Σ(supply)        # ⚠ R5, not D + |discard| + 3
```

⚠ **R5 — two denominator defects, both live because planes 14–16 carry no scope
restriction.**

* The fixed `+ 3` over-counts outside standard: `aside_composition` returns zeros
  when `not config.standard` and `discard_composition` returns zeros in expert.
  **Always divide by the actual `Σ supply`** of the matrix being summed.
* In solo the undrawn deck holds `SOLO_CARD_ID`, which is not in `DECK_MATRIX`, so
  `Σ_n c[n] = deck_remaining − 1`. Planes 14 and 15 divide by `D =
  deck_remaining` and therefore never sum to the deck. §6.4 remembers the solo
  marker; §7.1 did not. **Divide by `Σ_n c[n]`, not `deck_remaining`.**

Each is a prefix-sum lookup over a 15-vector. **Build the prefix sums once per
state and reuse them across all four sheets** — this is what keeps symmetric
encoding cheap. New helper in `deck_knowledge`:

```python
def number_prefix_sums(state, player) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative counts over card numbers 1..15 for (deck, discard+aside, reshuffled)."""
```

Writable values are `0..17`; printed card numbers are `1..15`. `numbers_for`
clamps, so `numbers_for(1, TEMP)` is `[1, 0, 2, 3]` — no −1.

**No per-stack fit plane.** One deck feeds all three stacks in standard mode, so
the next-number distribution is identical across them. The only per-stack
difference is the known next effect, already a one-hot in `next_effects`. Planes
14 + 15 + `next_effects` determine per-stack fit exactly.

### 7.2 Demand (24)

```
number_demand[v] = # of (empty box, v) pairs that are legal,   v in 0..17   (18)
effect_demand[e] = marks of effect e still wanted,             e in 6       (6)
```

`number_demand` is an **unweighted count of legal (box, value) pairs** — pure
combinatorics, no judgement. Computed per gap: a gap of length *L* contributes
*L* to every value in its interval.

`effect_demand` comes from the §3.4 requirement vectors plus remaining track
boxes, with one special case. **No plan kind requires the ESTATE *effect*** —
estate plans require estate *sizes*, which come from fences — but the effect is
critical for scoring and its value is exactly computable:

```
estate_demand = Σ_i count[i] × delta(i)

delta(i) = 0                                              if marks[i] >= ESTATE_ROW_BOXES[i]
           ESTATE_ROW_SCORES[i][marks[i]+1]
             − ESTATE_ROW_SCORES[i][marks[i]]             otherwise
```

⚠ **R5 — the guard is required, not defensive.** `ESTATE_ROW_SCORES[i]` has
`ESTATE_ROW_BOXES[i] + 1` entries and `marks[i]` reaches `ESTATE_ROW_BOXES[i]`, so
the unguarded index raises **on every row once it saturates** — for size 1 that is
a single ESTATE mark, i.e. routine. Verified for all six sizes. §9.4 promises a
defined degenerate case for every scalar; this was the one that raised.

A difference of two table lookups times a count. It captures why the estate
effect swings so hard: a mark multiplies across *every* estate of that size.

⚠ **R1 — use `estate_size_counts()` here, NOT the free counts.** The first draft
said the opposite and it is flatly wrong: `Sheet.estate_score()` (`sheet.py:493`)
multiplies `estate_size_counts()`, which counts **all** estates. Top fences stop a
City Plan from *reusing* an estate; they do not remove its real-estate points. So:

* `estate_demand` (a **scoring** quantity) → `estate_size_counts()`
* `estate_shortfall` and `feasible` (**plan eligibility**) → `free_estate_size_counts()`

Both vectors are in `tracks` for exactly this reason.

### 7.3 Reshuffle contraction (8)

⚠ **Demand and supply live in different spaces and must be projected before they
can be contracted.** `number_demand` is indexed by writable value `0..17`; deck
supply is indexed by printed card number `1..15`. Project demand into card-number
space first, in two variants:

```
card_demand_notemp[n] = # empty boxes b where n is legal in b               n in 1..15
card_demand_temp[n]   = # empty boxes b where SOME d in TEMP_DELTAS makes
                        (n+d) legal in b, clipped to 0..17                  n in 1..15
```

Both are **counts of boxes**, not of (box, value) pairs — a box takes one value,
so a card either can or cannot serve it. Counting pairs would double-count a card
reaching the same box through two deltas.

```
fit_rate(supply) = Σ_n card_demand[n] × supply_number[n] / (33 × Σ supply)
eff_rate(supply) = Σ_e effect_demand[e] × supply_effect[e] / norm
```

evaluated with `card_demand_notemp` and `card_demand_temp` separately, each
against the deck and against the reshuffled pool: **notemp/temp × number/effect ×
deck/reshuffled = 8**.

The reshuffle question is not "do I like low numbers"; it is "does the
post-reshuffle deck fit the holes I have left better than the current deck does" —
and, for every opponent, "does it help them more than me". Under symmetry the
second half is answered for every seat automatically, and the decision becomes a
single subtraction.

**Live every turn, not only at `ASK_RESHUFFLE`.** Anticipating the reshuffle is
what makes racing to the first plan worth more than its face value.

**Tie rule, confirmed against the engine.** `A_RESHUFFLE_YES` sets
`reshuffle_next_turn = True` and nothing clears it until applied, so **one YES
from anyone shuffles**. A NO is effective only if every tied player votes NO, and
their votes are concurrent and hidden. The §6.4 threat features are what let the
net condition on who else could be voting.

### 7.4 The reshuffle horizon — why a 0% fit is usually not a 0% fit

⚠ **R2 — new section, and the most important thing in §7.** A fit probability
computed on the *current* deck goes stale, and in this game it goes stale
constantly. The numbers:

* the construction deck is **81 cards** (`NUM_BASE_CARDS`);
* standard mode discards the old row and draws a new one every turn, so on an
  ordinary turn it depletes at **exactly 3 cards** (`_discard_step` /
  `_draw_step`) — deterministic, not stochastic. ⚠ **R5:** a *reshuffle* boundary
  takes **6** off the reformed deck (`_reshuffle_decks` draws 3, then `_draw_step`
  draws 3), and `_draw_playable` consumes an extra card whenever the solo marker
  turns up. Both make the refresh arrive **sooner**, so the `floor(D/3)+1` bound
  below survives — but it is a bound, and the depletion rate is not universally 3;
* a game runs ~25 turns, so **the deck is consumed almost exactly once per
  game**, and `_reform_deck` puts every discarded card back;
* on top of that, the first player to complete a City Plan is offered a
  **queued reshuffle**, and one YES from anyone fires it (§7.3).

So "no card left that fits this gap" is, for most of a game, a statement with a
short and *computable* expiry. A model given only plane 14 would read a dead gap
as permanently dead and mis-price every hold-and-wait line — the same class of
error as the roundabout blindness in §4, and for the same reason: the option is
real but unrepresentable.

**The three quantities are read together, and v3 already has two of them.**

| | meaning |
|---|---|
| plane 14 | P(fit) against the deck **as it stands** |
| plane 16 | P(fit) against the **immediate-reshuffle pool** (deck + discard + aside) ⚠ |
| `turns_to_reform` | **new** — how long the first number holds |

```
reveals_to_reform = floor(deck_remaining / 3) + 1
turns_to_reform   = min(reveals_to_reform, TURNS_CAP) / TURNS_CAP
```

⚠ **R4 — the `+ 1` is not cosmetic.** `_draw` reforms only when it *finds* the
deck empty (`game.py:434-441`), so with `D = 3` the next reveal consumes the last
three cards and the reform fires on the reveal **after** that: the true horizon is
2, and `D / 3` said 1. Checked at `D = 0, 3, 4`: `floor(D/3)+1` gives 1, 2, 2,
which matches the engine's sequencing in each case.

⚠ **R3 — this is an upper BOUND, not a prediction, and the spec must say so.**
There are two paths to a refreshed pool and only one of them is deterministic:

* **exhaustion** — the deck empties and `_reform_deck` fires. Depletion is fixed
  at exactly 3 cards per turn, so this arrival time is *known*;
* **the queued reshuffle** — the first player to finish a City Plan is *offered*
  the choice, and one YES from anyone fires it (§7.3). That is a **decision**, not
  a state, and predicting it is exactly the category-2 line v2 §8 forbids.

A queued reshuffle can only make the refresh arrive **sooner**, never later. So
`deck_remaining / 3` is a sound upper bound on turns-until-refresh, and that is
the honest thing to feed. The *opportunity* is fed alongside it — `reshuffle_race`
already carries `_may_ask_reshuffle` and the viewer's own vote — and the policy
learns the choice. Same division of labour as §8's refusal block: **the encoder
states what is available, the net learns what is taken.**

One global scalar. (It added one to the global block in R2; R4 then deleted the
dead `plan_conflict` block, so the final global width is **367** — §9.3.)

**How the temp agency enters the fit calculation.** Three separate things, and
keeping them separate is what makes the block correct:

1. **Reachability** — plane 15 widens the *interval* to `(low−2, high+2)`, because
   one card takes one delta from `TEMP_DELTAS`. Exact: the reachable set of a
   drawn number `n` is `{n−2 … n+2}` clipped to `0..17`.
2. **Applicability** — the widening applies only to a stack whose effect is TEMP.
   That is not guesswork: next turn's effects are printed and sit in
   `next_effects`, so planes 14 + 15 + `next_effects` give per-stack fit exactly
   (§7.1).
3. **Price** — `plane 15 − plane 14` is what a temp *buys*; what it *costs* comes
   from the `tracks` block (`temps / TEMP_BOXES`, per seat under symmetry) and the
   `TEMP_RANK_SCORES = (7, 4, 1)` majority those tracks feed.

⚠ **The widening is NOT gated on remaining temp boxes, and that is faithful.**
Verified against the oracle: `Player::getAvailableNumbersOfCombination` applies
`[-2,-1,1,2]` with no check on the temp track, and our `numbers_for`
(`game.py:907`) matches; `sheet.temps` then saturates via
`min(temps + 1, TEMP_BOXES)` (`game.py:1140`). So past 11 marks the shift stays
legal and simply stops scoring — the temp becomes **free but worthless**. Plane 15
is therefore correct without a gate, and the non-monotonic price is visible to
the net through the saturating `temps` track. Do not "fix" this by gating the
widening; it would diverge from BGA.

⚠ **R4 — plane 16 is the *immediate* reshuffle pool, not the natural-reform
pool.** They differ: a reshuffle taken now folds in today's discard and aside,
whereas a natural reform some turns later excludes whatever row is then on the
table. Plane 16 is exact for the queued-reshuffle path (§7.3) and an
approximation of the exhaustion path — which is the right way round, because the
queued path is the one a player can act on. The plane is **named for what it is**
so nothing downstream reads it as the exhaustion pool.

With all three present, "dead now, alive in four turns" is a subtraction and a
lookup, exactly as `plane 17 − plane 14` made the roundabout option legible.

**This does not make the net *use* it — see §11.3.** Supplying an exact quantity
and having the trunk exploit it are different claims, and this project already
has a counter-example sitting in the input (`box_spans`).

### 7.5 `p_fit_next_turn` — the plane the temp question exposed

⚠ **R3 — new plane, prompted by a worked example.** Take a box whose gap admits
**only an 8**, with no 8s left in the deck but 6s, 7s, 9s and 10s remaining.

Plane 15's interval handles it, and exactly — verified numerically at the
sentinels too:

```
gap bounds (low, high) = (7, 9)     # exclusive; only 8 fits
fits without temp                    -> [8]
fits with a temp (reachable ±2)      -> [6, 7, 8, 9, 10]
spec interval (low-2, high+2)        -> [6, 7, 8, 9, 10]     ✓ identical
```

So `plane 14 = c[8]/D = 0` while `plane 15 = (c6+c7+c9+c10)/D > 0`, and the
difference is precisely "this box is dead unless I spend a temp". **The answer to
the question as asked is yes.**

**But the example exposes a gap.** Plane 15 is *unconditional on a TEMP card
actually turning up*. If none of next turn's three stacks carries the TEMP
effect, plane 15 describes an option that does not exist next turn. The
information to resolve that is present — next turn's effects are printed and sit
in `next_effects` — but composing "plane 15 **and** a TEMP stack **and** the
without-replacement joint draw over three stacks" is exactly the arithmetic
composition §4.2 and §6.3 argue nets do not do reliably. Handing over the two
factors and hoping for the product contradicts the reason the fit planes exist.

**So compute it.** Per gap, using each stack's **known** next effect:

```
F_i = printed numbers that fit this gap on stack i
    = (low-1 .. high+1) ∩ [1,15]     if stack i's known next effect is TEMP
      (low+1 .. high-1) ∩ [1,15]     otherwise
```

and, with `x_i[n] = 1` when `n ∉ F_i`, `M_i = Σ_n x_i[n]·c[n]`,
`P_ij = Σ_n x_i[n]x_j[n]c[n]`, `R = Σ_n x_1x_2x_3 c[n]`:

```
none = M1·M2·M3 − P12·M3 − P13·M2 − P23·M1 + 2R
p_fit_next_turn = 1 − none / ( D·(D−1)·(D−2) )
```

**Verified against brute-force enumeration of all ordered card triples on 300
random compositions, zero mismatches.** Falling factorials, not a product of
marginals — the same without-replacement correctness §6.4 requires, and the same
reason `1 − (1−p)³` is banned in §9.3.

Cost is **O(1) per gap**: the `F_i` are intervals, and because the no-temp set is
a strict subset of the temp set, every `P_ij` and `R` is one more prefix-sum
lookup over the same 15-vector §7.1 already builds. Per gap, not per box — ≤10
gaps a sheet, broadcast.

⚠ **R4 — `D < 3` must NOT emit 0.0; the draft was wrong.** The next reveal still
produces three cards: `_draw` reforms the discard mid-draw and carries on
(`game.py:434-441`), so the true probability is generally **non-zero**, and
emitting 0 states a falsehood at exactly the moment a dead-looking gap comes back
to life. Plane 16 cannot stand in — it is a marginal, no-temp, single-card fit and
composes neither the three known effects nor the joint draw.

Use the **literal boundary-draw enumeration**: draw from the current deck until it
is exhausted, reform, continue, and evaluate the same `F_i` test on the resulting
three cards.

⚠ **R5 — the reform pool is `remaining + discard + ASIDES`, three cards more than
the draft said.** `_discard_step` sweeps this turn's three asides into
`self.discard` *before* `_draw_step` runs, so by the time `_draw` finds the deck
empty and calls `_reform_deck` (`game.py:426-441`) they are already in it. Read
literally off `state.discard` at encode time they are missed.
`deck_knowledge.aside_composition` / `after_reshuffle_composition` exist for
exactly this, and their docstring records that undercounting the pool by three
was **already a shipped bug once**. Use those helpers; never hand-roll the pool. This is the identical machinery §6.4
already requires for `p_complete_next_turn`'s reform case, so it is one helper
serving both, not new work.

⚠ **Scope: `config.standard` only**, like the §6.4 threat predicates and for the
same reason — `known_next_effects` is all-zero in solo and expert
(`deck_knowledge.py:186-196`), so `F_i` cannot be built. Expert raises; solo emits
plane 14's value, which is the honest no-effect-information fallback.

**Why this is worth a plane rather than trusting the trunk.** It is the one
quantity in the encoder that directly answers the question a player actually asks
at `CHOOSE_CARDS` — *can I still fill this hole, with what is on the table and
what is coming* — and it is the composition of four things the net would
otherwise have to multiply itself: the gap, the temp widening, the printed
effects, and the joint draw.

---

## 8. Refusal, blocking and houses per turn

Carried from v2 §6.4 and §6.5 unchanged. Restated in brief; v2 has the full
argument and the engine excerpt.

**Refusal and blocking (5)** — `P(refusal)` is not a state function; a refusal is
a *choice*, offered under two different rules at two phases. The factual
decomposition:

| feature | definition |
|---|---|
| `p_no_slot_playable` | P(no offered combination has any legal write, **temp included**) |
| `p_no_slot_playable_after_roundabout` | the same, given the best legal roundabout placement |
| `p_printed_unplaceable` | P(a stack's **printed** number has nowhere to go) — what *opens* the voluntary refusal |
| `roundabout_rescue_available` | 1.0 when `ROUNDABOUT_OPEN` is legal and would change `playable_slots()` |
| `p_forced_refusal_steady` | `p_no_slot_playable_after_roundabout` at the steady-state deck |

The first three are exact for next turn, because next turn's effects are printed.
Beyond that they use §9.3's steady-state rate. Not derivable from planes 14/15:
those are per-gap, these are a *union across gaps with overlapping number ranges*,
needing inclusion–exclusion.

Under symmetry these are per-seat, which makes them the direct input to "how soon
will they end the game" — three refusals is an end condition.

**Houses this turn (2)** — it can be **three**: `ROUNDABOUT_OPEN` is legal before
the write, a roundabout counts as a built house and returns to `CHOOSE_CARDS`, so
`roundabout → choose + write → bis`.

| feature | definition |
|---|---|
| `max_houses_this_turn` | 0–3, the most houses the **currently legal** sequences could place, ÷3 |
| `bis_usable` | 1.0 when a written house has an empty neighbour, so a bis write would be legal |

How many are actually placed is behavioural. The feature states the ceiling.

---

## 9. Full layout

### 9.1 Planes (22)

| plane | contents | status |
|---|---|---|
| 0 | validity mask for the 3×12 right-padding | shipped |
| 1 | box is written | shipped |
| 2 | number / 17 | shipped |
| 3 | is bis | shipped |
| 4 | is roundabout | shipped |
| 5 | top fence (house consumed by a plan) | shipped |
| 6 | fence to the right | shipped |
| 7 | is a pool position | shipped |
| 8 | **`writable_no_temp`** | **§5.2 — redefined** |
| 9 | estate size / 6 | shipped |
| 10 | `box_spans` / 18 | shipped |
| 11 | `positional_fit`, **delta-0 numbers only** — ablation target | **§5.2 — narrowed** |
| 12 | **`writable_temp_only`** | **§5.2 — new** |
| 13 | `span_if_roundabout` / 18 | §4 |
| 14 | P(fit \| deck, no temp) | §7.1 |
| 15 | P(fit \| deck, temp ±2) | §7.1 |
| 16 | P(fit \| post-reshuffle deck, no temp) | §7.1 |
| 17 | P(fit \| deck, no temp, after an optimal roundabout) | §4 |
| 18 | **`p_fit_next_turn`** — P(some stack next turn supplies a fitting number) | **§7.5 — R3, new** |
| 19 | plan slot 0 still needs a house here | **§3.3 — new** |
| 20 | plan slot 1 still needs a house here | **§3.3 — new** |
| 21 | plan slot 2 still needs a house here | **§3.3 — new** |

⚠ **Planes are renumbered relative to v2 §4**, which put `span_if_roundabout` at
12. Keep the `P_*` constants public in both languages so tests cannot drift, and
never refer to a plane by literal index outside those constants.

### 9.2 Per-sheet scalars (202)

| block | size | § |
|---|---|---|
| `tracks` | 26 | 20 shipped + 6 free-estate size counts (§2.3) |
| `score` | 9 | shipped |
| `capacity` | 4 | shipped |
| `roundabout_repair` | 3 | §4 |
| `total_span` | 1 | v2 §6 |
| `plans` | 108 | 3 slots × 36 (§3.4) |
| `demand` | 24 | §7.2 |
| `reshuffle_contraction` | 8 | §7.3 |
| `refusal` | 5 | §8 |
| `houses_this_turn` | 2 | §8 |
| `plan_conflict_seat` | 9 | §9.2a — R2, widened R4 |
| `free_boxes` | 1 | shipped |
| `is_viewer` | 1 | shipped |
| `seat_valid` | 1 | shipped |
| | **202** | |

### 9.2a `plan_conflict_seat` (9) — where the real conflict lives

⚠ **R4 — widened to 9 and given an exact selection rule.** R2 specified 3 pairs ×
2 with one `kills` bit per unordered pair, and referred the estate selection to
`progress()`. Both were wrong: **`kills` is directed** (completing *a* may kill
*b* while completing *b* leaves *a* alive), and `progress()` does a pure
size-*count* match (`min(need[size], supply[size])`, `plans.py`) — it never
chooses actual estates, so it cannot supply the "cheapest satisfying selection"
the draft leaned on. `_estates_available_for` has the same shape: it returns a
supply `Counter`, not a set of estates.

Per seat: **3 unordered pairs × 1 overlap + 6 ordered pairs × 1 kill = 9.**

| field | n | definition |
|---|---|---|
| `overlap[a,b]` | 3 | `|T(a) ∩ T(b)| / max(1, min(|T(a)|, |T(b)|))` over the **selected** target boxes on this sheet |
| `kills[a→b]` | 6 | 1.0 if completing *a* would make *b* `feasible == 0.0`, evaluated by top-fencing `T(a)` on a copy and re-running §6.1 |

**The selection rule `T(·)`, stated so two implementations agree.**

* `FULL_STREET(x)` → every box of street *x*. `EXTREMITIES` → the six
  `EXTREMITY_POSITIONS`. Both plan-fixed; no choice involved.
* `ESTATE` → the estates a **canonical** satisfying selection would spend, chosen
  by this exact rule: process required sizes **descending**; for each, take the
  eligible free estate with the **lowest `(street, start)`**; the boxes of the
  taken estates are `T`. Deterministic, no optimisation, identical in Rust.
* **When no satisfying selection exists** (the plan is not yet completable —
  which is the common case), `T = ∅`, `overlap = 0`, and `kills[a→·] = 0`. A plan
  that cannot be completed cannot consume anything, so this is correct rather
  than merely convenient.
* A **banked** plan (`seat in plan_turns_for(viewer, slot)`) has already spent its
  houses; its top fences are on the sheet and `feasible` sees them, so `T = ∅` and
  its outgoing kills are 0.

⚠ Descending order is not an optimisation and is not claimed to be optimal — it
is a **tie-break**, chosen so the rule is total and reproducible. `feasible`
(§6.1) remains the authority on whether the plan is reachable at all; `T` only
says which boxes a completion would consume *if it happened*.

Cost is 6 hypothetical `feasible` evaluations per seat per state, against a §6.1
that is a handful of table lookups.

⚠ Symmetric and information-safe like everything else in the per-seat block: it
reads `sheet_for(viewer, seat)`, never a live opponent sheet.

### 9.3 Global scalars (367)

| block | size | change |
|---|---|---|
| `phase` | 12 | — |
| `turn` | 1 | — |
| `stacks` | 78 | — |
| `chosen_combination` | 25 | — |
| `last_house` | 34 | — |
| `pending_estate` | 7 | — |
| `plan_identity` | 93 | — |
| `reshuffle_race` | 2 | — |
| `next_effects` | 18 | — |
| `deck` | 74 | — |
| `effect_supply_rate` | 6 | **new** |
| `temp_availability_rate` | 1 | **new** |
| `bis_availability_rate` | 1 | **new** |
| `turns_to_reform` | 1 | **R2, new** — §7.4 |
| `config` | 4 | — |
| `seat` | 6 | — |
| `seat_validity` | 4 | — |
| | **367** | |

⚠ **`plan_conflict` — DELETED in R4, and it was dead on arrival.** The block was
specified over the plans with plan-fixed target boxes: `FULL_STREET` (ids 18, 19)
and `EXTREMITIES` (id 22). **All three are stack-1 plans**, and the game deals
exactly one plan from each of the three stacks — so two target-bearing plans can
never be on the table at once, and all six floats are identically zero in every
reachable game. Verified by enumeration over `DEALT_PLAN_IDS`.

This is the same dead-input problem that settled `MAX_OPPONENTS` (v2 §10.5) and
the 37→28 plan one-hot (§7.1 of v2), reached by the same mistake: reasoning about
which plans *conflict* without checking which plans can *co-occur*. Global scalars
**373 → 367**.

The conflict that does occur is `ESTATE` against `FULL_STREET`/`EXTREMITIES`, and
those *can* co-occur because estate plans are dealt from all three stacks. It is
per-seat and sheet-dependent, so it lives entirely in §9.2a.

**`effect_supply_rate` (6)** — per-turn probability that effect *e* appears among
the three on offer. ⚠ **Not `1 − (1−p)³`.** Three cards are drawn without
replacement:

```
k = deck_effect_count[e],  D = deck_remaining
rate = 1 − (D−k)(D−k−1)(D−k−2) / ( D(D−1)(D−2) )
```

⚠ **R5 — there is no `D < 3` approximation; use the boundary-draw enumeration.**
The draft kept `1 − (1 − k/max(D,EPS))³` there "because the exact form is
undefined". It is not undefined: R4 established that at `D < 3` the reveal reforms
mid-draw and still yields three cards, and §7.5 already mandates the literal
boundary-draw helper for exactly this. One helper serves both. (The draft's
fallback also divides by `EPS` at `D = 0`.) Wrong only in the last turn or two of a
deck cycle — but a spec that says "exact" must be. The same correction applies to `temp_availability_rate`,
`bis_availability_rate` and §8's steady-state term.

`temp_availability_rate` and `bis_availability_rate` are the same quantity
surfaced separately because §8's two blocks each need exactly one of them. Next
turn's effects are already certainties (`next_effects`), so these are the
steady-state rate for turn+2 onward.

**There is deliberately no per-gap number supply *rate*.** The `1 − (1−p)³`
transform of plane 14 is monotone in a value already in the input.

### 9.4 Numeric contract — every emitted float, exactly

⚠ **R1 — new section.** The layout tables above fix *where* each float goes; they
do not fix its *value*, and several §6–§8 quantities were left as prose that two
implementations could satisfy differently — or that could emit `inf`. Since the
whole point of §10.6 is bit-exact Python/Rust equality, every scalar needs an
equation, a range and a defined degenerate case.

**Global rules.**

| rule | value |
|---|---|
| `EPS` | `1e-6`, the only guard constant; declared once in both languages |
| division | every quotient is `num / max(den, EPS)`, then clamped to its stated range |
| `TURNS_CAP` | `12.0` — every turn-valued feature is `min(t, TURNS_CAP) / TURNS_CAP`, so all land in `[0, 1]` |
| unreachable | a turn estimate whose supply is zero emits **exactly 1.0** (= `TURNS_CAP`, "never"), never `inf` |
| dtype | all arithmetic in `f64`, cast to `f32` once at write time — the pyo3 f64-not-f32 rule from the Kingdomino port |
| counts | every count is divided by its structural maximum, listed below, and clamped to `[0, 1]` |

**Per-scalar table.** Anything not listed is a `[0,1]` flag or an existing
shipped block.

| scalar | equation | range |
|---|---|---|
| `estate_shortfall[s]` | `max(0, need[s] − free[s]) / 6` | [0,1] |
| `parks_needed[x]` | `(PARK_BOXES[x] − parks[x]) / PARK_BOXES[x]`, `0` if the plan wants no parks | [0,1] |
| `pools_needed[x]` | `(3 − pools[x]) / 3`, `0` if none wanted | [0,1] |
| `houses_needed[x]` | plan-needed empty boxes in *x* `/ STREET_SIZES[x]` | [0,1] |
| `bis_needed[x]` | `max(0, 5 − bis_count_per_street[x]) / 5` | [0,1] |
| `temps_needed` | `max(0, 7 − temps) / 7` | [0,1] |
| `fences_needed` | `steps_left` for `ESTATE` kinds `/ 6`, else 0 | [0,1] |
| `steps_left` | `min(steps, TURNS_CAP) / TURNS_CAP` (unchanged from shipped) | [0,1] |
| `turns_lower_bound` | §6.2, then `min(t, TURNS_CAP) / TURNS_CAP` | [0,1] |
| `effect_rate_turns` | `Σ_e need[e] / max(effect_supply_rate[e], EPS)`, then capped | [0,1] |
| `number_rate_supply` | `Σ_{n ∈ needed numbers} c[n] / max(D, EPS)` | [0,1] |
| `expected_turns_to_plan` | `max(effect_rate_turns_raw, number_rate_turns_raw)` where `number_rate_turns_raw = houses_needed_total / max(number_rate_supply, EPS)`, then capped | [0,1] |
| `eff_rate(supply)` | `Σ_e effect_demand[e] · supply_effect[e] / max(NORM_EFF, EPS)`, ⚠ **R4** `NORM_EFF = (Σ_e supply_effect[e]) · (Σ_e effect_demand[e])` | [0,1] |
| `fit_rate(supply)` | `Σ_n card_demand[n] · supply_number[n] / max(33 · Σ supply, EPS)` | [0,1] |
| `number_demand[v]` | `count / NUM_BOXES` | [0,1] |
| `effect_demand[e]` | `marks / track size of e`; for ESTATE, `estate_demand / 66` ⚠ **R4** | [0,1] |
| `effect_supply_rate[e]` | §9.3 hypergeometric; `D < 3` → `1 − (1 − k/max(D,EPS))³` | [0,1] |
| `max_houses_this_turn` | `n / 3` | [0,1] |
| `free_estate_size_counts[s]` | `count / 4`, ⚠ **R5 UNCLAMPED** — matching the shipped `estate_size_counts` exactly | [0, 8.25] |
| `roundabout_repair[x]` | `capacity_if_roundabout()[x] / STREET_SIZES[x]` | [0,1] |
| `total_span` | `total_span() / 594` — `NUM_BOXES × 18`, the empty-sheet maximum | [0,1] |
| `overlap[a,b]`, `kills[a→b]` | §9.2a; already in [0,1] | [0,1] |
| `turns_to_reform` | `min(floor(D/3)+1, TURNS_CAP) / TURNS_CAP` | [0,1] |
| `p_fit_next_turn` | §7.5 | [0,1] |

⚠ **R5 — `free_estate_size_counts` must NOT be clamped.** The shipped
`estate_size_counts` block is `count / 4.0` with no clamp (`encoder.py:376-377`).
Clamping only the new vector breaks the differencing that justifies the shared
scale, and it breaks it at five size-1 estates — routine, and exactly where estate
plans live. **Clamp both or clamp neither**; v3 clamps neither, so the pair
differences correctly and the shipped block is untouched.

⚠ **R4 corrections to two scales the draft got wrong.**

* `estate_demand / 24` was invented, not derived. The structural maximum is
  `NUM_BOXES × max one-mark delta = 33 × 2 = 66` (every box a size-1 estate;
  `ESTATE_ROW_SCORES[0] = (1,3)` so the first mark is worth 2 each). Verified by
  enumeration over `ESTATE_ROW_SCORES`.
* `NORM_EFF = 81 · Σ demand` divides by the **full** deck size, so the same sheet
  against the same *proportions* scores lower simply because the deck has drained.
  That is a deck-size signal leaking into a rate. `Σ supply · Σ demand` makes it a
  genuine proportion, matching what `fit_rate` already does with `Σ supply`.

⚠ **`expected_turns_to_plan` is `max`, not a sum, of the two rate terms.** The
first draft said "combined" and left it open. Effects and numbers are consumed by
the *same* combination — you take one card and it carries both — so the binding
constraint is whichever is slower, not their sum. (Within `effect_rate_turns` the
sum over *e* stays, because those compete with each other across turns; v2 §6.3's
argument for summing is about distinct effects, not about effect-vs-number.)

⚠ **Zero supply is a real state, not an edge case.** Late in a deck an effect can
genuinely have zero copies left, and `need / 0` is where an `inf` reaches a
`LayerNorm` and poisons a whole batch. The clamp is not defensive coding, it is
the definition: "no supply" means "not in `TURNS_CAP` turns" means `1.0`.

---

## 10. Blocking tests

Everything in v2 §9 carries over. These are the ones v3 adds or changes.

### 10.1 `feasible` soundness fuzz — blocking

Over ≥20k random reachable sheets × dealt plans: **if `feasible` is 0.0, no
continuation completes the plan.** Verified by bounded exhaustive rollout on the
sheet alone. A single false death fails the test. Completeness is explicitly
*not* asserted (§6.1).

⚠ **R1 — the oracle must include roundabout actions.** The first draft listed
"writes, bis, fences", which omits the one action that repairs spans and can
occupy a numerically-dead box. An oracle missing roundabouts would have
*confirmed* the three unsound death checks §6.1 now corrects. The rollout
enumerates writes, bis, **roundabouts**, fences and permit refusals, to the
free-box horizon, with `roundabouts < ROUNDABOUT_BOXES` respected.

### 10.2 `can_complete_this_turn` bidirectional — blocking

⚠ **R1 — over ALL `steps_left`, not `≤ 2`.** Restricting the test to the range
the predicate chooses to enumerate makes it structurally incapable of catching a
too-low cutoff, which is precisely the defect found in review.

Over random states, **for every value of `steps_left`**: the predicate is true
**iff** some legal action sequence this turn ends with `plans.can_be_scored`
true. Both directions. Two further assertions:

* **the ceiling is sound**: for every kind, no reachable one-turn sequence
  advances the plan by more than `one_turn_ceiling` steps. This is the assertion
  that would have failed on the first draft's constant 3;
* **the cap never binds**: the enumeration cap is not reached on any reachable
  state, or the helper raises rather than returning a quiet `0.0`.

The generator must reach `steps_left ∈ {1, 2, 3}` for `FULL_STREET` and
`EXTREMITIES` specifically — the three-house window is the case at issue and
random play reaches it rarely.

### 10.2a `p_complete_next_turn` joint-draw exactness — blocking, new

Against a brute-force reference that enumerates all ordered 3-card draws from the
literal remaining deck list: exact agreement to `1e-9` over ≥2k states, including
≥100 states with fewer than three construction cards left (the reform path) and
≥100 with a solo marker in the deck.

### 10.3 Requirement-vector fidelity — blocking, new

⚠ **R1 — the first draft's two `iff`s excluded reachable states.** A
`FULL_STREET` or `EXTREMITIES` plan can have *every target box written* and still
be incomplete, because another plan top-fenced one of them: the unwritten-target
mask is then legitimately zero on an incomplete plan. And an all-zero
`street_serves` means sheet-wide **or dead or already complete**, not the first
two only. Both are fixed by testing the **exact expected value** against an
independent reference rather than a non-zero-ness proxy, and by treating
*complete* as its own status.

For every dealt plan and ≥5k random sheets, classify each `(plan, sheet)` as
`COMPLETE` / `DEAD` / `LIVE` from `can_be_scored` and `feasible`, then assert:

* `estate_shortfall` equals `max(0, Counter(required) − Counter(free_estates))`
  elementwise, recomputed independently — for every status;
* `street_serves` equals the reference mask elementwise, where the **reference is
  aliveness**: `1.0` iff street *x* is not provably unable to contribute, with a
  completed street scoring `1.0` (⚠ R5 — the draft asserted "equals the reference
  mask" without defining the reference, so the test could not arbitrate between
  the two readings in §3.2). All-zero is expected for `SEVEN_TEMP` and for `DEAD`,
  and is a **failure** for `COMPLETE` or for a `LIVE` street-bound plan;
* planes 19–21 equal the exact expected mask — the unwritten, un-top-fenced
  target boxes — which is all-zero for a `COMPLETE` plan **and** for a `DEAD` one
  whose targets are all written or fenced;
* mutation check: perturbing any one requirement field changes at least one
  encoded float.

### 10.4 Temp-split exactness — blocking, new

`plane 8 ∪ plane 12` equals the shipped v2 plane 8 exactly, and
`plane 8 ∩ plane 12 = ∅`, over ≥5k random states. This is what proves the split
loses nothing.

### 10.5 Symmetry and leak — carried, still blocking

v2 §9.3 is **two** tests, not one. At a turn boundary the live sheet and the
public snapshot are equal, so a helper reaching for `state.sheets[p]` passes a
boundary symmetry test unnoticed; the leak is only visible mid-turn. Assert
symmetry at a boundary, and assert the leak mid-turn against the same block
before and after an opponent writes. Both must be mutation-checked.

⚠ Every new per-seat feature in v3 must route through `sheet_for(viewer, seat)`,
`score_breakdown(..., viewer=)` and `plan_turns_for(viewer, slot)`. §6.4's
`can_complete_this_turn` is the highest-risk block here: it reads three sheets and
the offer, and **the cache key must include the viewer** or it will serve a
viewer-unsafe answer across seats.

### 10.6 Rust/Python equivalence — carried

`rust_encode_equiv.py` over ≥8k states, exact float equality. The import-time ABI
cross-check (`lib.rs:809-835`) compares `(ABI, SHEET_PLANES, NUM_SHEET_SCALAR,
NUM_GLOBAL_SCALAR, MAX_SEATS)` and raises `PyImportError` on drift — it will fire
the moment Python is bumped and Rust is not, which is the intended failure.

---

## 11. Measurement

### 11.1 Ablate in the base game, not the advanced game

`welcome-to-bga-advisor` measured the served net at **22.6 vs GreedyBot 51.4** in
the **base** game, with estates 8.8 vs 30.2 and plans 0.4 vs 3.6 — against 43.6 vs
54.4 in the advanced game. Roundabouts do not exist in the base game, so the
+18-point roundabout line cannot mask an estate-building failure there.

**Primary metric for §2 and §3: base-game estates/seat-game and plans/seat-game.**
Mean score is secondary, and advanced-game mean score is the *worst* of the three
for this purpose.

`positional_fit` (plane 11) and `expected_turns_to_plan` remain first-class
ablation targets, measured on **roundabout play quality** and **plan completion
rate** respectively — not mean score.

⚠ Drive every measurement macro with `mc.apply_macro`. `mc.primitives_for(macro)[0]`
applies only the card choice and lets the placement fall to `legal_actions()[0]`;
that truncated form already produced one wrong verdict in this project.

### 11.2 GreedyBot is a cloning target, NOT a strength target

⚠ **R2 — the first draft blurred these and the user is right to separate them.**

**GreedyBot is a bad model of strong play.** It maximises immediate score. High-
level Welcome To is *maximise final score while minimising time to the three City
Plans* — a two-term objective in which banking points early is frequently the
losing line, and one this project has already measured: score is a **lagging,
early-inverting** indicator (v2 §9 measurement trap). Nothing in v3 should aim at
GreedyBot's 50.8, and "we beat greedy" is not evidence of strength.

**But the S0 gate is not a strength claim.** It asks one narrow question: *given a
fixed teacher, can this trunk and this optimiser fit it?* `net_score` 21.83
against `greedy_score` 51.48 with `gate_policy_agreement` **0.0** and
`policy_top1` 0.475 says the clone did not converge on a teacher it was handed
outright. That is a statement about the *learner*, and it holds no matter how
mediocre the teacher is — a fit failure against a weak target is worse news than
a fit failure against a strong one, not better.

⚠ **R3 — one correction to the user's framing, with the measurement.** "Placement
means almost nothing to a greedy bot" is not true of *this* GreedyBot.
`bots.py:49-122` records a paired-seed ablation of its own terms:

| terms | score |
|---|---|
| capacity only | 33.9 |
| + total span | 42.6 (+8.2, t = 4.1) |
| + positional fit | 45.8 (+3.2, t = 2.2) |

Placement is ~12 of its 45.8 points, and `_fit` scores the write itself, not the
resulting score. So a cloned placement is not arbitrary.

**But the sharper version of the objection is right and matters more.**
GreedyBot's three placement terms are *sheet-hygiene* heuristics — capacity, span,
proportional fit — and it is **completely plan-blind**. `_evaluate` reads
`breakdown.total`, which includes `plans`, so it collects a plan the turn it
happens to complete one; it never places a house *in order to* build an estate of
a needed size, and it never races an opponent. Cloning it therefore teaches
hygiene and nothing about the objective in §11.2's first paragraph.

Two consequences:

* **S0 is a hygiene clone, and should be described as one.** It is still the
  right fit test — but "the clone converged" must never be read as "the model
  understands plans", and the §11.3 base-game curriculum will clone a plan-blind
  policy on an all-estate board unless the teacher changes.
* **The teacher is the cheapest lever, and it is outside this spec.** Adding a
  fourth GreedyBot term for plan progress — `estate_shortfall` reduction, which
  §2.3 now computes anyway — would make the base-game clone teach the skill the
  base game is being used to isolate. Teachers may be heuristic; only *features*
  are bound by the v2 §8 no-goodness rule. **Recommended, and to be decided
  before step 8, not during it.**

So the gate stays, with its meaning stated correctly:

| | measures | target |
|---|---|---|
| S0 clone gate | can the trunk fit its teacher (**sheet hygiene only**, unless the teacher gains a plan term) | agreement and score-within-2 vs **the teacher**, not an absolute score |
| strength | the actual objective | plans/seat-game, **turns to third plan**, and paired margin vs the previous checkpoint |

⚠ **Strength is never measured against GreedyBot.** It is measured as paired
margin against the previous promoted checkpoint, with `plans/seat-game` and
`turns_to_third_plan` reported alongside, because those two are the objective
GreedyBot does not optimise.

### 11.3 Base game first — and for a better reason than "simpler"

Measured, not assumed:

```
base pool     18 plans:  ESTATE 18
advanced pool 28 plans:  ESTATE 18, DECORATIVE 4, FULL_STREET 2,
                         FIVE_BIS 1, SEVEN_TEMP 1, EXTREMITIES 1, COMPLETE_STREET 1
```

**Every City Plan in the base game is an estate plan.** So the base game is not
merely a smaller advanced game — it is precisely the sub-game that §2's estate
shortfall vector addresses, with roundabouts absent so the +18-point roundabout
line cannot mask an estate-building failure, and with the 9 locus-bound plans of
§3 removed so the estate signal is not competing for capacity.

That makes base-first a **curriculum with a diagnostic payoff**, not just a ramp:

1. clone S0 on base games and gate it. If the clone converges here and not on
   advanced, the problem is representational load, and §3 is doing its job;
2. if it fails *here* — on 18 plans of one kind, no roundabouts — then the
   binding constraint is optimisation or capacity and no feature block should be
   built until that is resolved (the §11.2 rule);
3. then extend to advanced. The plan one-hot is already sized at the advanced
   superset (v2 §7.1), so base is a strict subset of the same input space and the
   weights transfer with no surgery.

v2's "base: supported, not trained" was a statement about the *deployment* target
and is unchanged. This is a statement about training order.

### 11.4 Exact features do not become used features

`box_spans` has been in the encoder since v1, `GreedyBot` reaches 50.8 with it,
and the net does not use it. That is the standing counter-example to "if we
compute it exactly, the model will learn it", and it applies directly to the
§7.1/§7.4 fit probabilities — the most carefully specified block in v3 and the
one with the least guarantee of being exploited.

The encoder supplies facts; **what forces the trunk to represent a fact is a
target that cannot be predicted without it.** So the fit-probability block ships
with one new auxiliary target:

> **`empty_boxes_at_end`** — per seat, the number of boxes still unwritten at the
> terminal state, ÷ 33.

It is exactly what the fit probabilities integrate to over the rest of the game,
it is dense (defined for every seat on every row — no masking, so it avoids the
M1 mask-normalisation trap in `AUX_TARGETS_SPEC.md`), and it is terminal truth
rather than a label of goodness. If planes 14–17 and `turns_to_reform` carry real
information, this head's `r2` is where it shows up first; if that `r2` stays low,
the block is not being used and the ablation says so cheaply.

One head unit, in `PER_SEAT_HEAD_TARGETS`. Not an encoder change.

---

## 12. Implementation order

One ABI break. Nothing is preserved: no checkpoint migration, no WTS
back-compatibility, no legacy head zero-fill (§0.4).

| # | step | files |
|---|---|---|
| 1 | `sheet.py` helpers: `free_estate_size_counts`, `span_if_roundabout`, `capacity_if_roundabout` | `sheet.py`, `tests/test_sheet.py` |
| 2 | `plans.py`: `requirements()`, `feasible()`, `turns_lower_bound()` — **§10.1 and §10.3 first** | `plans.py`, `tests/test_plans.py` |
| 3 | `deck_knowledge.py`: `number_prefix_sums`, `effect_supply_rate` (hypergeometric) | `deck_knowledge.py`, tests |
| 4 | `game.py`: `one_turn_ceiling`, `can_complete_this_turn`, `p_complete_next_turn`, `max_houses_this_turn` — **§10.2 and §10.2a** | `game.py`, tests |
| 5 | `encoder.py`: all planes and blocks, ABI → 2, block tables updated — §10.4, §10.5 | `encoder.py`, `tests/test_encoder.py` |
| 6 | `encoder.rs`: port, then **§10.6 equivalence gate green before anything else runs** | `welcome_to_rust/src/encoder.rs`, `rust_encode_equiv.py` |
| 7 | network shapes (auto-derived from `encoder` constants), regenerate WTS shards | `network.py`, `self_play.py`, `samples.rs` |
| 8 | re-run S0 from scratch; gate per §11.2 | `train.py`, `datagen.py` |

Steps 1–4 are independently testable and land before the encoder touches them.
Step 6 is the gate that must be green before any data is generated — a silent
Python/Rust divergence in a new block would poison every shard.

Each block stays independently ablatable via `encoder.block_slice(name)`.

---

## 13. Open questions for review

1. ~~**`street_serves` semantics.**~~ **RESOLVED (R2): permissive form, all three
   streets.** Marking the best two would pre-sum a choice, which is what §3
   exists to stop.
2. ~~**`reachable_new[s]`.**~~ **RESOLVED (R2): ship the loose bound**
   (`reachable_new[s] =` free boxes in the street), so `feasible` fires only on
   unambiguous deaths. ⚠ **Open work item, deliberately deferred to after the
   build:** develop a test plan to tighten it safely. §10.1 proves soundness but
   says nothing about how much reachability the loose bound gives away, and the
   tightened version is the highest-risk code in v3. Do not tighten it without a
   measured false-negative rate first.
3. ~~**Plane count and shard size.**~~ **RESOLVED (R2): keep all 22 planes.**
   Numbers, so the trade is explicit rather than vague:

   | | floats/row | KB/row | 1M rows | 20M rows |
   |---|---:|---:|---:|---:|
   | v2 shipped | 2,302 | 9.0 | 9.2 GB | 184 GB |
   | **v3** | 4,379 | 17.1 | 17.5 GB | 350 GB |

   WTS stores the **encoded floats**, not the state (`samples.rs:663` sizes a
   record from `SHEET_PLANES_LEN + SHEET_SCALARS_LEN + VIEWER_PLANE_LEN +
   NUM_GLOBAL_SCALAR`), so shard size tracks encoder width 1:1 — this is disk and
   shard-read bandwidth, **not** model or search cost, which is the +3.4% in §1.

   Since strength is the decision driver, nothing is dropped. If disk ever binds,
   the lever is **not** to cut planes but to store the `snapshot` and re-encode at
   train time: snapshots are ~4× smaller and the encoder is already in Rust. That
   trades CPU for disk without touching what the model sees.
4. ~~**`p_complete_next_turn` cost.**~~ **RESOLVED (R2) — it stays, and the cost
   concern was mine to retract.** Next turn's effects are certain, so the joint
   draw is an exact 3,375-term sum over number-class triples, not an enumeration
   over the number-effect composition. §6.4 carries the corrected form. Beating
   or **tying** an opponent to a plan is worth `scores[0]` rather than
   `scores[1]` — up to 6 points on a single slot — so this is core, not optional.
5. ~~**`plan_conflict` thinness.**~~ **RESOLVED (R2) — widened per-seat.** The
   global fixed-box block stays; `plan_conflict_seat` (§9.2a) is added because
   the `FULL_STREET` house-reuse rule makes the decisive conflict a per-sheet,
   per-choice fact. Per-sheet 193 → 199.

---

## 14. Review round 1 — what changed

Nine findings, all verified against the code and all accepted. Four were errors
in the draft; five were underspecification. No finding was disputed.

| # | finding | verdict | resolution |
|---|---|---|---|
| 1 | `steps_left >= 3` early exit rejects legal one-turn completions | **error** — §8's own three-house sequence contradicts it | §6.4 `one_turn_ceiling`, kind-specific; §10.2 extended to all `steps_left` |
| 2 | free-box count is not a turns lower bound | **error** — same root cause | §6.2 `house_term = ceil(houses / 3)` |
| 3 | death checks ignore roundabout repair | **error** — `build_roundabout` writes the sentinel and ignores numeric fit (`sheet.py:409`) | §6.1 reads `capacity_if_roundabout` / `span_if_roundabout`; §10.1 oracle gains roundabout actions |
| 4 | threat premise leaks or misstates expert/solo | **valid** — the premise is `Globals::isStandard` (`game.py:174`); `known_next_effects` is all-zero in both | §6.4 scoped to `config.standard`; expert raises, solo emits 0.0 |
| 5 | estate demand uses the wrong population | **error** — `estate_score()` uses all `estates()` (`sheet.py:493`); top fences block plan *reuse*, not points | §7.2 corrected: scoring → `estate_size_counts`, eligibility → `free_estate_size_counts` |
| 6 | next-turn probability needs a joint draw | **valid** — three cards without replacement are correlated | §6.4 ordered-draw enumeration + solo marker + mid-reveal reform; new §10.2a |
| 7 | numeric ABI not fully specified | **valid** — prose admitted divergent implementations and `inf` | new §9.4: equation, range and degenerate case for every scalar |
| 8 | requirement blocking assertions exclude valid states | **valid** — a written-but-fenced target, and `COMPLETE` vs `DEAD` | §10.3 tests exact masks against a reference, with a three-way status |
| 9 | `plan_conflict` has no implementable semantics | **valid** — estate targets are sheet- and choice-dependent | §9.3 narrowed to plan-fixed target boxes; estate conflict carried per-seat by `feasible` |

Findings 1, 2 and 3 share one root cause worth stating plainly: **the draft
treated "one step" as "one turn" and "one box" as "one house"**, when Welcome To
lets a single turn place three houses and lets a roundabout occupy a box no
number could reach. §8 contained the counter-example to §6.1, §6.2 and §6.4 and
the draft did not apply it. Any future feature that reasons about per-turn
progress must be checked against §8 first.

Shape was unchanged by round 1: 21 planes / 193 per-sheet / 372 global.
(Round 2 later took it to **199 / 373** — see §15.) Finding 9 removed no floats (the block
keeps its 6 slots with a narrower, exactly-computable definition), and finding 4
changes values, not widths.

---

## 15. Review round 2 — what changed

Round 2 was a rules check plus four scoping decisions from the user. Five changes.

| topic | outcome |
|---|---|
| `FULL_STREET` semantics | **Confirmed on all four points** against `Plans/FullStreetPlan.php`: all boxes filled, roundabout counts, estates of any size allowed, but **no house already spent on another plan**, and no fence can be added after completion. Our engine matches exactly (`plans.py:197`, `sheet.py:366`). The house-reuse rule is a *cross-plan death* and drove §9.2a. |
| fit-probability staleness | **Valid and severe.** Deck is 81 cards depleting at exactly 3/turn, so it is consumed roughly once per ~25-turn game, plus the queued post-plan reshuffle. New §7.4 adds `turns_to_reform`; planes 14 / 16 / horizon are read together. Global 372 → 373. |
| "will the model learn it?" | **No guarantee, and `box_spans` is the standing counter-example.** New §11.4: the fit block ships with an `empty_boxes_at_end` aux target, which is what those probabilities integrate to and is where their `r2` becomes visible. |
| GreedyBot as a yardstick | **Conceded.** Greedy maximises immediate score; the real objective is final score *and* time to three plans. §11.2 rewritten: the S0 gate is a **cloning-fidelity** test against a fixed teacher, never a strength claim, and strength is paired margin vs the previous checkpoint plus `plans/seat-game` and `turns_to_third_plan`. |
| base-first curriculum | **Adopted, with a stronger reason than simplicity.** Measured: the base pool is **18 plans, all ESTATE**; advanced adds the 10 locus-bound ones. Base is exactly the sub-game §2 targets, with roundabouts absent. §11.3 makes it a diagnostic ramp: a clone that fails *there* proves the constraint is optimisation, not representation. |

Open questions 1–5 are all resolved in §13; two cost concerns from earlier rounds
were **retracted**, both mine:

* `p_complete_next_turn` is not expensive — next turn's effects are certain, so
  the joint draw is an exact 3,375-term sum, not an enumeration over the full
  number-effect composition;
* `plan_conflict` was narrowed too far in R1 — the per-seat version (§9.2a) is
  where the `FULL_STREET` house-reuse conflict actually lives.

Shape after round 2: 21 planes / 199 per-sheet / 373 global.
(Round 3 later took the planes to **22** — see §16.)

---

## 16. Review round 3 — what changed

| topic | outcome |
|---|---|
| `EXTREMITIES` house reuse | **Confirmed and escalated.** `Plans/ExtremitiesPlan.php` top-fences all six end boxes on validate and refuses any already fenced — identical to `FULL_STREET`. Non-obvious consequence now in §6.1: an extremity box is a street *end*, so fencing it removes the whole estate containing it, and completing the plan can delete **up to six estates** from `free_estates()` at once. |
| `turns_to_reform` is not deterministic | **Correct — demoted to an upper bound.** Exhaustion at 3 cards/turn is known; the queued reshuffle is a *vote*, and predicting a choice is the category-2 line. A queued reshuffle only makes the refresh sooner, so `deck_remaining / 3` is sound as a bound. The opportunity is fed via `reshuffle_race`; the net learns the choice. |
| how temp enters the fit calculation | **Answered in §7.4**, in three separate parts: reachability (plane 15 widens the interval), applicability (`next_effects` says which stack is TEMP), price (`tracks` + the 7/4/1 majority). Also verified the widening is **not** gated on remaining temp boxes in either BGA or our engine — past 11 marks a temp is free but worthless — so plane 15 needs no gate. |
| GreedyBot and placement | **Half conceded.** The measured ablation (`bots.py:49-122`) says placement is ~12 of GreedyBot's 45.8 points, so cloned placements are not arbitrary. But it is **plan-blind** — it collects a plan it happens to finish and never places to set one up. §11.2 now calls S0 a *hygiene* clone and flags the teacher's missing plan term as a decision to take before step 8. |

| temp fit probability | **Answered, and it exposed a gap.** Plane 15's interval is exactly right — verified numerically that a gap admitting only an 8 counts 6/7/9/10 with a temp, at the sentinels too. But plane 15 is *unconditional on a TEMP card appearing*, and composing it with `next_effects` and the joint draw is the arithmetic the fit planes exist to avoid. New plane 18 `p_fit_next_turn` (§7.5) computes it exactly, O(1) per gap, closed form verified against brute force on 300 compositions. |

Shape after this round: **22 planes / 199 per-sheet scalars / 373 global
scalars** — 4,373 floats per row, +3.6% parameters, 17.1 KB per training row.

---

## 17. Review round 4 — what changed

Nine findings, all verified and all accepted. Two were structural.

| # | finding | verdict | resolution |
|---|---|---|---|
| 1 | estate one-turn ceiling of 3 too low; `fences_needed` unsound as a bound | **error** — one fence can change the estate match by **two** (a fenced run of 6 against `(3,3)`), and a turn supplies up to 3 fences | §6.4 ESTATE ceiling → `len(required_sizes)`, i.e. **no early exit**; §6.2 fence term → `0 if steps_left == 0 else 1` |
| 2 | roundabout capacity comparison off by one | **error** — the roundabout consumes one of the empties | §6.1 compares against the **hypothetical sheet's** empty count (`E − 1`) |
| 3 | `FIVE_BIS` death test undefined and unsound | **error** — "bis-able free boxes" would read `bis_candidates()`, which future writes grow | §6.1 → `min(BIS_BOXES − bis_marks, free boxes in x)`, both loose upper bounds |
| 4 | banked plans give a permanent phantom threat | **valid** — `can_be_scored` checks only the sheet; `SEVEN_TEMP` stays true forever | §6.4 short-circuits both threat scalars on `plan_turns_for` |
| 5 | `p_fit_next_turn` returns a false 0.0 at `D < 3` | **error** — the reveal reforms mid-draw and still yields three cards | §7.5 uses the literal boundary-draw enumeration, shared with §6.4; scoped to `config.standard` |
| 6 | reform horizon off by one | **error** — `_draw` reforms only when it *finds* the deck empty | §7.4 → `floor(D/3) + 1`, checked at `D = 0, 3, 4`; plane 16 relabelled the **immediate-reshuffle** pool |
| 7 | per-seat conflict asymmetric and underdefined | **valid** — `kills` is directed, and `progress()` counts sizes rather than choosing estates | §9.2a → 9 floats (3 overlap + **6 ordered** kills) with a canonical, total selection rule |
| 8 | global conflict block permanently zero | **valid, and fatal to the block** — ids 18, 19, 22 are **all stack-1**, and one plan is dealt per stack, so two target-bearing plans can never co-occur | **block deleted**; global 373 → 367 |
| 9 | numeric contract still incomplete | **valid** | §9.4 gains 7 scales; `estate_demand` bound corrected 24 → **66** (derived, not invented); `NORM_EFF` → `Σ supply · Σ demand` |

Finding 8 is the one worth remembering: the block was specified by asking *which
plans conflict* without asking *which plans can be on the table together*. That is
the same dead-input mistake v2 already made twice — `MAX_OPPONENTS` and the 37-wide
plan one-hot — and it survived three review rounds here because every check was a
check of the definition, never of reachability. **Any future block keyed on a pair
of plans must first be tested for co-occurrence.**

Findings 1, 2, 3 and 5 share the R1 root cause, still not fully burned out: the
draft kept reasoning about *quantities* (houses, boxes, cards) where the engine
counts *events* (fences, reveals, matched estates).

Final shape: **22 planes / 202 per-sheet scalars / 367 global scalars** —
4,379 floats per row, +3.6% parameters, 17.1 KB per training row.

---

## 18. Review round 5 — what changed

Fifteen findings (6 errors, 5 underspecifications, 4 staleness), all verified
against the engine and all accepted. Full text in `ENCODER_V3_REVIEW_R5.md`.

| # | finding | verdict | resolution |
|---|---|---|---|
| 1 | death tests ignore BIS | **error** — `bis_candidates` (`sheet.py:311-338`) has **no ascending-order check**, so a bis fills a numerically-dead box exactly like a roundabout. `1 _ 2 _ 3 _` has capacity 0 and every gap bis-writable | §6.1 gains `bis_reach(x)`; `FULL_STREET`'s capacity clause **deleted** (`any(top_fences[x])` is exact alone); `EXTREMITIES` gains a bis term |
| 2 | ESTATE death test ignores fence-splitting | **error** — `estates()` is bounded by fences, not writes, so a fence makes new sizes consuming **zero** free boxes. Plan 5 `(6,6)` vs a full unfenced street 2: declared dead, one fence completes it | `reachable_new[s]` admits re-partitioning of built runs |
| 3 | a queued reshuffle destroys `next_effects` | **error** — `_reshuffle_decks` redraws `stack_new`, so the cards on show never become asides | §6.4/§7.5 branch on `reshuffle_vote_for(viewer)`; the "certainty" claim weakened. ⚠ `reshuffle_next_turn` stays unreadable — it leaks an earlier actor's vote |
| 4 | reform pool 3 cards short | **error** — `_discard_step` sweeps the asides in before `_reform_deck` fires | use `aside_composition` / `after_reshuffle_composition`; never hand-roll the pool (this was a shipped bug once) |
| 5 | plane 16 denominator wrong outside standard; `D` wrong in solo | **error** | divide by `Σ supply`; planes 14/15 divide by `Σ_n c[n]`, not `deck_remaining` (the solo marker is not in `DECK_MATRIX`) |
| 6 | `estate_demand` raises at a saturated row | **error** — `ESTATE_ROW_SCORES[i][marks+1]` overflows on **all six** sizes; size 1 after a single mark | `delta(i) = 0` when the row is full |
| 7 | §6.2 maxes where §6.3 sums | underspec — sound but needlessly weak on the one *hard bound* | `effect_term = Σ_e marks needed` |
| 8 | `street_serves` had two contradictory definitions | underspec — and §10.3 never defined its reference, so the blocking test could not arbitrate | **aliveness**: complete ⇒ 1.0; `parks_needed[x]` carries the work; reference defined in §10.3 |
| 9 | `free_estate_size_counts` clamp breaks its own justification | underspec — shipped `estate_size_counts` is `/4.0` unclamped | clamp neither |
| 10 | planes 13/17 need a "no roundabout left" case | underspec | collapse to `box_spans` / plane 14 when `roundabouts == ROUNDABOUT_BOXES` |
| 11 | §9.3's `D<3` fallback restates the falsehood R4 removed | underspec | boundary-draw enumeration; also noted `effect_supply_rate` is **exact** for turn+2, not steady-state |
| 12–15 | staleness | — | 21→22 planes and +3.4%→+3.6% in §13; `NUM_GLOBAL_SCALAR` **is** in Rust (`encoder.rs:26`); the three GreedyBot means (45.8 / 50.8 / 54.4) labelled with their corpora; "exactly 3 cards per turn" qualified for reshuffle and solo-marker turns |

**Findings 1 and 2 are the R1 root cause for the third and fourth time.** The
draft keeps reasoning about *numbers* where the engine writes *houses*. Both a
roundabout and a bis fill a box no drawn number can reach, and a fence creates
estates with no box at all. The standing rule, now in §6.1: **every span-,
capacity- or estate-derived claim must be checked against writes, roundabouts,
bis and fences — all four.**

Shape is unchanged by this round: **22 planes / 202 per-sheet / 367 global**,
4,379 floats per row. No finding added or removed a float; every one changed a
value, a guard or a denominator.
