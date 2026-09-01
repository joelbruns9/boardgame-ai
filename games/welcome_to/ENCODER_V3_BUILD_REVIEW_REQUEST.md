# Review request — encoder v3 build, steps 1–4

`ENCODER_V3_SPEC.md` §12 steps 1–4: the `sheet.py`, `plans.py`,
`deck_knowledge.py` and `game.py` support code the encoder will consume, plus
the tests that gate it. **Step 5 (`encoder.py`, the ABI break to 2) is NOT in
scope and is not written** — this is the last point at which a signature or a
field's meaning can change cheaply, which is why the review is here.

Full Python suite: **562 passed, 3 skipped**. The four Rust equivalence modules
are excluded; they need a rebuilt extension and belong to step 6.

## 1. Scope

| area | files | state |
|---|---|---|
| sheet helpers | `sheet.py`, `tests/test_sheet.py` | committed `4e65beb` |
| plan requirements + feasibility | `plans.py` | committed `4e65beb` |
| the reachability oracle | `tests/plan_reachability.py` | committed `4e65beb` |
| feasibility gate | `tests/test_plan_feasibility.py` | committed `4e65beb` |
| prefix sums + supply rates | `deck_knowledge.py`, `tests/test_deck_knowledge.py` | uncommitted |
| threat predicates | `game.py`, `tests/test_plan_threat.py` | uncommitted |

`game.py` also carries an unrelated advisor change from an earlier session; only
the block below `# Encoder v3:` is in scope.

## 2. What to check hardest, in order

### 2.1 `plans.feasible` — the block with the worst track record

Five spec rounds produced **four unsound death tests**, every one by reasoning
about *numbers* while forgetting the three actions that put a house on a sheet
without one. The rule now stated at the function: check every span-, capacity-
or estate-derived claim against **writes, roundabouts, bis and fences**.

Per kind, the current death conditions:

| kind | dead when |
|---|---|
| `ESTATE` | some size `s` has `need[s] > free_estates[s] + reachable_estate_counts[s]` |
| `FULL_STREET(x)` | `any(top_fences[x])` — **the only clause**, and exact |
| `EXTREMITIES` | an extremity is top-fenced; or empty with `span_if_roundabout == 0`, no roundabout left, and not `bis_reachable` |
| `FIVE_BIS` | `max_x(bis_count_per_street[x] + bis_reach()[x]) < 5` |
| `SEVEN_TEMP` | never |
| `COMPLETE_STREET` | no street can reach 3 pools **and** a roundabout |
| `DECORATIVE` | park: never. pool: fewer than 2 streets can reach 3 pools. pool&park(x): street *x* cannot |

**Please attack these directly.** The claim is one-sided — `False` only when
provably unreachable — so the failure that matters is a death the game can
escape from.

### 2.2 `_one_turn_hopeless` and `_estate_houses_short` — newest, least reviewed

Added for throughput after measuring the one-turn enumeration at **31,873
sheets** for a single offer on a sparse sheet (33 roundabout sites × 33 write
boxes × 30 fence resolutions) against ~300 on a realistic late-game sheet.

The added claim: an estate of size `s` needs `s` contiguous *written* boxes, so a
shortfall needs `Σ shortfall[s] × s` boxes, minus every non-top-fenced written box
counted as reusable supply. If that exceeds 3 — the per-turn house ceiling — no
turn can complete the plan.

Both clauses are lower bounds against a ceiling, so a `True` should never be a
missed threat. **Is the "every written box is reusable" term actually generous
enough?** Fences only accumulate, so an existing run can be split but never
merged — the concern is whether some configuration makes a written box
unavailable in a way this ignores.

### 2.3 `one_turn_sheets` — completeness

A turn is modelled as: optional opening roundabout → chosen combination written
somewhere → that combination's effect resolved (or passed). Refusal branches are
omitted on the grounds that a refusal writes no house and marks no track, so it
cannot make a plan scoreable.

**Does this miss a legal turn shape?** The predicate is asserted *bidirectionally*
against this same enumeration, so an omission here would not fail its own test —
it would make both sides of the test agree on a wrong answer. This is the most
dangerous kind of gap in the build and the test cannot catch it.

### 2.4 `one_turn_ceiling`

| kind | ceiling | reasoning |
|---|---|---|
| `ESTATE` | `len(required_sizes)` — never early-exit | one fence can close **two** steps |
| `FULL_STREET`, `EXTREMITIES` | 3 | roundabout → write → bis |
| `COMPLETE_STREET` | 2 | one effect mark plus a roundabout |
| `FIVE_BIS`, `SEVEN_TEMP`, `DECORATIVE` | 1 | one combination is one effect mark |

Tested over **every** `steps_left`, not the range the predicate enumerates — that
restriction is what let the original ESTATE ceiling of 3 survive.

### 2.5 The oracle, and the one argument that is not mechanical

`tests/plan_reachability.py` over-approximates (unlimited deck) so "no completion
found" is trustworthy, and its cap **raises** rather than returning `False`.
Moves are pruned by what `can_be_scored` mechanically reads.

⚠ **One pruning step is stronger than field-disjointness and was accepted
deliberately (2026-08-30):** fences are pruned for the six kinds that do not read
the fence grid, justified by *monotonicity* — `bis_candidates` and
`surveyor_zones` each test the fence only as a conjunct requiring `False`, so
setting one can only remove entries, never add. If this is wrong, three plan
kinds lose their coverage silently.

### 2.6 `requirements()` field semantics

The invariant, which deviates from what §10.3 of the spec first said:
**aliveness has exactly one home.** `feasible` (mirrored by `street_serves`)
carries it; every other field states what the plan still *wants* and is gated on
`done` alone. A dead estate plan is still short a 3 and a 6.

`street_serves` is **aliveness, not remaining work** — a street whose parks are
already complete reads `1`, and `parks_needed[x]` reads `0`.

### 2.7 `_p_any_stack_supplies` and `effect_supply_rate`

Both are without-replacement and neither may be a product of marginals.
`effect_supply_rate` is checked against brute-force enumeration of the literal
remaining deck; `_p_any_stack_supplies` against permutations of literal cards.
Both model the `D < 3` boundary as two sequential phases because `_draw` reforms
the discard mid-draw and still yields three cards.

## 3. Known holes, disclosed

* **Oracle coverage is not uniform.** Of the declared deaths across the fixture:
  verified `DECORATIVE` 346, `COMPLETE_STREET` 56, `FIVE_BIS` 28, `ESTATE` 4,
  `EXTREMITIES` 1; **undecidable** `COMPLETE_STREET` 47, `ESTATE` 12,
  `FIVE_BIS` 2. Zero false deaths among the verified. `ESTATE` has the thinnest
  coverage and the loosest bound — an uncomfortable pairing.
* **`reachable_estate_counts` is deliberately the loose bound** of §13.2: it
  ignores fence legality and contiguity. Tightening it is an outstanding, separate
  reviewed change and needs its own test plan.
* **The reshuffle branch of `p_complete_next_turn` over-estimates.** When the
  viewer has voted for a reshuffle, `next_effects` is destroyed, so the completing
  numbers are taken as the union over all six effects. Over-warning is the chosen
  direction for a threat feature; it is not exact.
* **`p_complete_next_turn` returns `1.0` when the plan is already scoreable.**
  Deliberate, but worth a second opinion on whether the feature should instead
  read 0 there, with `banked`/`can_complete_this_turn` carrying that case.
* **Scope is `config.standard`.** Expert is out of scope entirely; solo emits
  `0.0` for both threat predicates. Neither is trained on.

## 4. Already verified — please do not re-litigate

* the exact interval for the temp widening, `(low−2, high+2)`, including at the
  sentinels;
* `floor(D/3) + 1` for the reform horizon, checked against `_draw`'s
  reform-on-finding-empty sequencing;
* the hypergeometric against the spec's worked example (0.545 exact against 0.535
  for the banned `1 − (1−p)³`);
* the closed form `M₁M₂M₃ − ΣP·M + 2R` for ordered distinct-card triples;
* the bis and temp tracks **saturate rather than gate** — `legal_actions` never
  reads their counters — so no bound may subtract from `BIS_BOXES` or `TEMP_BOXES`;
* all shape arithmetic: 22 planes / 202 per-sheet / 367 global / 4,379 per row.

## 5. Defects this build has already produced

Recorded because they show where the errors cluster, not for credit:

| defect | found by |
|---|---|
| `bis_marks` cap in the oracle, in `sheet.bis_reachable`, and in the spec's own §6.1 | external review |
| `street_serves` ignored the whole-plan verdict | §10.3 test |
| demand vectors zeroed on death, contradicting §10.3 | §10.3 test |
| `any(playable_slots(...))` — truthiness on slot **indices**, false when only slot 0 is playable | re-reading |
| `max_houses_this_turn` read `self.sheets[player]`, a live mid-turn sheet, for the wrong seat | re-reading |

The last two were found by reading rather than by any test, and neither would
have been caught by the suite as written. `max_houses_this_turn` has no direct
test of its information-set safety, which is a gap worth closing.
