# Encoder v3 spec — review round 5

Reviewed `ENCODER_V3_SPEC.md` against `sheet.py`, `plans.py`, `game.py`,
`deck_knowledge.py`, `encoder.py`, `constants.py`, `bots.py` and
`welcome_to_rust/src/{encoder,lib}.rs` on 2026-08-30.

Six errors, five underspecifications, four cosmetics. The two structural ones
(#1, #2) are both the R1 root cause again: **the draft still reasons about
numbers where the engine writes houses.**

---

## Errors

### 1. §6.1 — the death tests ignore BIS. Same shape as R1 finding 3.

`Sheet.bis_candidates` (`sheet.py:305-330`) requires only that the box be empty,
that its neighbour hold a non-roundabout number, and that no estate fence sit
between them. **There is no ascending-order check**, and `ACTION_BIS`
(`game.py:1172-1181`) then calls `write(..., is_bis=True)`. So a bis, exactly
like a roundabout, fills a box that no drawn number could legally reach.

Two of the death tests are therefore unsound:

* **`FULL_STREET(x)`** — `capacity_if_roundabout()[x] < E − 1` is not death.
  A street written `1 _ 2 _ 3 _ …` has `placement_capacity` 0 and every gap
  `box_spans` 0, yet every gap is bis-writable from either side, and
  `can_be_scored` tests only `n is not None` (`plans.py:200-202`). Up to
  `BIS_BOXES − bis_marks` boxes can still be filled.
* **`EXTREMITIES`** — "`span_if_roundabout` is 0 **and** no roundabout remains"
  is not death: an extremity box is adjacent to box 1 of its street, so a bis
  copying that neighbour writes it.

The inconsistency is visible inside the §6.1 table itself: the `FIVE_BIS` row
(corrected in R4) bounds bis reach properly, and the two rows above it do not.

§10.1's oracle does enumerate bis, so the fuzz *would* fail — but the stated
rule is what gets built first, and this is the third round in a row where a
house-writing action that bypasses numeric fit was missed.

**Fix.** Every span- or capacity-derived death test needs a bis term, as a loose
upper bound in the R4 style:
`bis_reach(x) = min(BIS_BOXES − bis_marks, |{empty b in x : some unfenced
neighbour holds a non-roundabout number}|)`. For `FULL_STREET` the honest cheap
option is to drop the capacity clause entirely and keep only `any(top_fences[x])`,
which is exact.

### 2. §6.1 / §13.2 — the ESTATE death test ignores fence-splitting, and is unsound.

`Sheet.estates` (`sheet.py:427-446`) bounds an estate by *fences*, not by
writes. One SURVEYOR fence splits an already-built run into two estates of new
sizes, **consuming no free box at all**.

`need[s] > free_estates[s] + reachable_new[s]`, with §13.2's resolved
`reachable_new[s] =` free boxes in the street, counts only new houses. Worked
counter-example, both sound and reachable:

> Plan 5 requires `(6, 6)`. Street 2 is fully written with no fences: one estate
> of size 12, so `free_estates[6] = 0`. The rest of the sheet is full, so
> `reachable_new[6] = 0`. The test fires — and a single fence at slot (2,5)
> splits the run into 6 + 6 and completes the plan.

This is the identical fact §6.4-R4 uses to raise the estate ceiling ("a fenced
run of 6 with a plan needing (3,3) scores 0 matched, and one SURVEYOR fence in
its middle scores 2"). It was applied to the ceiling and not to the death test.

**Fix.** `reachable_new[s]` must admit re-partitioning: for every maximal built
run, the sizes obtainable by adding fences at slots in `surveyor_zones()`.
A loose sound version is cheap — for each built run of length `L` that is not
fully top-fenced and not welded by a bis pair, every `s ≤ L` is reachable.

### 3. §6.4 / §7.5 — a queued reshuffle destroys `next_effects`, and neither block accounts for it.

`_reveal_step` (`game.py:522-534`) runs `_reshuffle_decks` before `_draw_step`.
`_reshuffle_decks` (`game.py:576-592`) reforms the deck, draws a fresh
`stack_new`, and then runs a **second** `_discard_step`, which discards the
current top cards. Its own docstring: *"the pair on show after a reshuffle is
made of two freshly drawn cards."*

So on a reshuffle boundary the current top cards never become asides, and next
turn's effects are **not** `next_effects[i]`. Both `S_i` (§6.4) and `F_i` (§7.5)
are then built on effects that will not be offered, from a pool that is wrong,
and §7.5 states flatly that the effects are "printed and certain".

Note the information-set constraint: `reshuffle_next_turn` is explicitly **not
public** (`game.py:1264-1274` — reading the table-wide OR leaks an earlier
actor's vote). The encoder may read only `reshuffle_vote_for(viewer)`.

**Fix.** Specify the `reshuffle_vote_for(viewer) == True` branch: either
evaluate against `after_reshuffle_composition` with the effects unknown (falling
back to the plane-14 marginal), or emit the fallback and say so. And weaken the
"certainty" claim — another seat's YES is genuinely hidden, so certainty is
conditional on nobody having voted.

### 4. §7.5-R4 / §6.4 — the boundary-draw reform pool is three cards short.

`_reform_deck` (`game.py:426-432`) uses `self.discard` *at the moment `_draw`
finds the deck empty*, and `_discard_step` has by then already swept this turn's
three aside cards into it. §7.5 says "reform (`remaining + discard`)", which, read
literally off `state.discard` at encode time, misses them.

`deck_knowledge.aside_composition` / `after_reshuffle_composition`
(`deck_knowledge.py:134-176`) exist precisely for this, and their docstring
records that undercounting the pool by three cards was already a shipped bug
once. The reform helper must add `aside_composition`.

### 5. §7.1 — plane 16's denominator is wrong outside standard, and `D` is the wrong denominator in solo.

* `aside_composition` returns zeros when `not config.standard`, and
  `discard_composition` returns zeros in expert. The fixed `D + |discard| + 3`
  over-counts by 3 in solo. Use `Σ supply`.
* In solo the undrawn deck holds `SOLO_CARD_ID`, which is not in `DECK_MATRIX`,
  so `Σ_n c[n] = deck_remaining − 1`. Planes 14 and 15 divide by
  `D = deck_remaining` and so never sum to the deck. §6.4 remembers the solo
  marker; §7.1 does not.

Planes 14–16 carry no `config.standard` scope restriction, so this is live.

### 6. §7.2 — `estate_demand` is undefined at a saturated estate row.

`ESTATE_ROW_SCORES[i]` has `ESTATE_ROW_BOXES[i] + 1` entries — `(2,3,4,5,5,5)` —
and `estate_marks[i]` reaches `ESTATE_ROW_BOXES[i]` (`estate_rows()` then stops
offering the row). So `ESTATE_ROW_SCORES[i][marks[i] + 1]` indexes past the end.
For size 1 that is a single ESTATE mark, i.e. common. §9.4 promises an equation,
a range and a defined degenerate case for every scalar; this one raises.
Define the delta as `0` when the row is full.

---

## Underspecification

### 7. §6.2 — `effect_term` maxes where §6.3 sums, and the sum is equally sound.

One combination per turn means at most **one** effect mark per turn
(`_EFFECT_PHASE`, `game.py:1455-1465` — each effect phase applies one mark).
So `Σ_e (marks of e still needed)` is a valid lower bound and strictly tighter
than `max(max_e marks, #distinct effects)`. §6.3 states this argument three
paragraphs later ("summed rather than maxed, because one combination per turn
means needing both parks and fences is competing with yourself") and §6.2
contradicts it. Sound either way — but weak for no reason, on the one feature
sold as a hard bound.

### 8. §3.2 — `street_serves` has two incompatible definitions.

The field is defined as "1.0 if street *x* can still contribute to this plan at
all", and the `FULL_STREET` example uses it as an **aliveness** flag: `(0,0,0)`
means dead. The `DECORATIVE` gloss then says "1.0 on each street whose parks are
not yet complete" — which marks a *completed* street 0, i.e. the street that
contributes most to plan 23 reads as "cannot contribute".

§10.3 asserts `street_serves` "equals the reference mask elementwise" but never
defines the reference, so the blocking test cannot arbitrate between the two
readings. Pick aliveness (complete ⇒ 1.0) and let `parks_needed[x]` carry the
remaining work — that split is the whole §3 argument.

### 9. §9.4 — the `free_estate_size_counts` clamp breaks the differencing that justifies it.

The shipped `estate_size_counts` block is `count / 4.0` with **no clamp**
(`encoder.py:376-377`). §9.4 clamps the new free vector to `[0,1]` and justifies
the `/4` scale by "the two vectors must be read on the same scale to be
differenced". With five size-1 estates — routine — they are not on the same
scale, and the difference is wrong exactly where estate plans live. Clamp both
or clamp neither.

### 10. §4 — planes 13/17 need a "no roundabout left" case.

§6.1 correctly guards its post-roundabout death tests on
`roundabouts < ROUNDABOUT_BOXES and config.advanced`; §4 says only that the
planes are zero in a base game. In an advanced game with both roundabouts spent,
plane 13 must collapse to `box_spans` and plane 17 to plane 14, or
`plane 17 − plane 14` reports option value for an option that no longer exists —
the same lie §5.1 is deleting from plane 8.

### 11. §9.3 — the `D < 3` fallback restates the falsehood R4 removed from plane 18.

R4 finding 5 established that at `D < 3` the reveal reforms mid-draw and still
yields three cards. §9.3 keeps `1 − (1 − k/max(D,EPS))³` for `effect_supply_rate`
(and for `temp_availability_rate`, `bis_availability_rate` and §8's steady-state
term) "because the exact form is undefined". It is not undefined — it is the
boundary-draw enumeration §7.5 now mandates, and one helper already serves both.
Lower severity: wrong only in the last turn or two of a deck cycle. At `D = 0` it
also divides by `EPS`.

---

## Cosmetic / staleness

* §13 open question 3 says "keep all **21** planes"; §13.3 refers to "the
  **+3.4%** in §1". R3/R4 took these to 22 planes and +3.6%.
* §0.1's table marks `NUM_GLOBAL_SCALAR` absent from Rust. It is present —
  `encoder.rs:26`, value 358 — and is one of the five constants cross-checked at
  `lib.rs:809-835`. The "absent" list below it (demand, refusal, …) is right.
* §0.2 says GreedyBot "reaches 50.8 mean score using exactly those three terms";
  `bots.py:49-72` records 45.8 for the same three terms, and §11.1/§11.2 quote
  51.4 / 51.48 / 50.8. Different corpora (seats, variant, seed count) — say
  which, or one of them will be quoted at the wrong thing later.
* §7.4's "exactly 3 cards per turn" holds only on non-reshuffle standard turns.
  A reshuffle boundary takes **6** off the reformed deck (`_reshuffle_decks`
  draws 3, then `_draw_step` draws 3), and `_draw_playable` consumes an extra
  card whenever the solo marker turns up. The `floor(D/3)+1` bound is unaffected
  — both make the refresh arrive sooner — but the prose overstates.

---

## Checked and correct — do not re-litigate

* **R4 finding 8.** Plans 18, 19 and 22 are all `stack=1` (`plans.py:102-106`)
  and one plan is dealt per stack, so two target-bearing plans can never
  co-occur. Deleting the global `plan_conflict` block is right.
* **§7.5's inclusion–exclusion.** `M1M2M3 − (P12M3 + P13M2 + P23M1) + 2R` is the
  exact count of ordered distinct-card triples: all three pairwise intersections
  and the triple intersection equal `R`, so the union is `ΣP − 3R + R`. The
  `P_ij`/`R` prefix-sum shortcut also holds — the no-temp interval nests inside
  the temp interval, so every complement is two tails of one 15-vector.
* **Plane 15's interval.** `n` serves a box in `(low, high)` iff
  `{n−2 … n+2} ∩ [low+1, high−1] ≠ ∅` iff `n ∈ [low−1, high+1]`, i.e.
  `(low−2, high+2)`. `numbers_for` (`game.py:907-915`) *drops* out-of-range
  deltas rather than clamping, but the reachable set is identical either way
  (`numbers_for(1, TEMP) = [1,0,2,3]`), so no divergence.
* **`floor(D/3) + 1`.** Matches `_draw`'s reform-on-finding-empty sequencing at
  `D = 0..6`. `_reform_deck` is only reached when `deck_pos >= len(deck)`.
* **`effect_supply_rate` is exact for turn+2, not an approximation.** The three
  cards drawn at the *next* boundary are exactly the ones whose effects are
  offered at turn+2 (they become the asides), and they come off the current deck
  of `D`. The hypergeometric over the current composition is therefore exact,
  not steady-state — §9.3 undersells it.
* **The three-house ceiling.** `A_ROUNDABOUT_OPEN` requires
  `ctx.last_house is None` (`game.py:958-966`) and `build_roundabout` sets it, so
  at most one roundabout per turn: roundabout + write + bis = 3, and no more.
* **`estate_demand / 66`.** Derivable: size-1 yields 2 points per box (`(1,3)`)
  and dominates every other size per box consumed, so `33 × 2` is the structural
  maximum.
* **Shape arithmetic.** 26+9+4+3+1+108+24+8+5+2+9+1+1+1 = **202**; globals
  358 + 6 + 1 + 1 + 1 = **367**; plan slot 2+1+2+6+18+6 = **36**;
  22×36 + 202 = **994** per seat; 4×994 + 36 + 367 = **4,379** per row. All
  consistent, including the 477 → 994 / 122k → 254k first-layer figure.
* **The tie rule.** `plan_scores` (`game.py:1336-1354`) pays `scores[0]` to every
  player whose turn equals `min(turns)` — §6.4's premise is right.
* **§8's refusal decomposition.** `p_printed_unplaceable` matches
  `game.py:984-988` exactly: the voluntary refusal opens when
  `available_locations(ctx.number)` is empty, i.e. on the *printed* number, not
  the temp-widened set.
