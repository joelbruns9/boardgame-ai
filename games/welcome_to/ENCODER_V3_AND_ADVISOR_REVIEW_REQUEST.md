# Review request — encoder v3 (steps 1–7) and the BGA advisor

Everything on `welcome-to-engine` after the last reviewed commit, `26696f6`
(S2 training/promotion hardening, see `REVIEW_BACKLOG.md`). Eight commits, none
of them externally reviewed.

**This supersedes `ENCODER_V3_BUILD_REVIEW_REQUEST.md`** (steps 1–4), which was
written at `5a73fe6` and never answered. Its §2 detail is still accurate for the
code it describes and is cited below rather than copied, **but its priorities
are out of date**: the §6.4 threat predicates it spends most of its attention on
are no longer encoded (§3.4 below).

The design document is `ENCODER_V3_SPEC.md` — 1,850 lines, five design-review
rounds (§14–§18), plus §19 (demotion + Rust port) and §20 (step 7). **Do not
review the spec as a whole.** It is the reference for what the code is meant to
do; the questions below point at the sections that matter.

---

## 1. Scope

| commit | what | files in scope |
|---|---|---|
| `4e65beb` | spec; steps 1–2: sheet helpers, plan `requirements` / `feasible` / `turns_lower_bound`, reachability oracle | `sheet.py`, `plans.py`, `tests/plan_reachability.py`, `tests/test_plan_feasibility.py`, `tests/test_sheet.py` |
| `5a73fe6` | step 3 (first version) + step 4: one-turn enumeration, threat predicates, `turns_lower_bound` fix | `game.py` (below `# Encoder v3:`), `plans.py`, `deck_knowledge.py`, `tests/test_plan_threat.py`, `tests/engine_turn_oracle.py` |
| `7744273` | the live BGA advisor | `advisor_adapter.py`, `bga_extract.py`, `bga_snippet.js`, `web_app.py`, three test modules |
| `d7bed84` | `turn_reach.py`, an **unwired** step-4 rewrite, preserved | `turn_reach.py` — keep/delete decision only (§3.7) |
| `084db58` | step 3 replaced by the exact boundary-draw version from another branch | `deck_knowledge.py`, `constants.py` (`EPS`), `tests/test_deck_knowledge.py` |
| `ecac2fa` | step 5: `encoder.py` at the full layout, ABI 1 → 2 | `encoder.py`, `tests/test_encoder.py`, `tests/test_encoder_v3.py` |
| `b154363` | step 6: Rust port; §6.4 threat pair demoted; integer-numerator draw sums | `welcome_to_rust/src/{encoder,plans,sheet}.rs`, `deck_knowledge.py`, `encoder.py` |
| `13ea612` | step 7: ABI guard on shards and checkpoints; two exact encoder speed-ups | `network.py`, `train.py`, `s2_train.py`, `advisor_adapter.py`, `self_play.py`, `welcome_to_rust/src/samples.rs`, `game.py`, both encoders |

Current layout: **22 planes / 196 per-sheet / 367 global**,
`ENCODER_ABI_VERSION = 2`, `TRAINING_SHARD_VERSION = 3`.

Nothing has been trained on v3 yet. **This is the last point at which a field's
meaning can change for free** — once step 8 generates data, every change is a
re-generation.

---

## 2. Already gated — please do not re-verify by hand

| gate | result at `13ea612` |
|---|---|
| full WT suite `pytest games/welcome_to/tests` | 691 passed, 3 skipped |
| Rust crate `cargo test --release` | 28 passed |
| §10.6 Python ↔ Rust encoder equivalence, exact `np.array_equal` | **20,876 encodings / 6,564 states / 35 games, zero divergences** — every seat count × base/advanced × random/no-refusal/greedy driver |
| gate power | a coverage replay of the same workload reaches every rare branch (§7.5 `D < 3`: 3,326 gaps; viewer-voted §9.3 rate: 12; roundabout rescue: 2,778; steady refusal falling back to the reshuffle pool: 703; plan-conflict kills: 4,515). A planted bug in the thinnest one was caught at seed 13 |
| §10.4 temp split partitions the old writable plane | ≥5,000 seat-states |
| §10.5 symmetry at a boundary + mid-turn leak, Python and Rust | green |
| old shards refused | versions 1 and 2, both readers, by test |
| old checkpoints refused | every `runs/welcome_to_*` artifact, all three loaders |
| v3 end to end | 500 generated games → 200 training steps via the Rust loader → reload |

⚠ **What the equivalence gate does NOT prove:** that either implementation
matches the spec. Rust was written from the Python, so a Python misreading of
the spec passes the gate on both sides. §3.1 is an example found while writing
this document.

---

## 3. What to check hardest, in order

### 3.1 ⚠ Known spec deviation: the viewer's reshuffle vote (§7.5, §8) — found, NOT fixed

Spec R5 finding #3 (§7.5 text, spec lines ~672–692): a queued reshuffle
**destroys** `next_effects`, so everything that treats next turn's effects as
certain must branch on `reshuffle_vote_for(viewer)`. When it is true, the spec
requires falling back to the **effect-marginal form over
`after_reshuffle_composition`**.

| feature | branches on the vote? |
|---|---|
| §9.3 `effect_supply_rate` (+ temp/bis rates) | **yes** |
| §7.5 plane 18 `p_fit_next_turn` | **no** — uses `next_is_temp` from `next_effects` unconditionally |
| §8 first three refusal floats | **no** — `_playable_sets` reads `next_effects` unconditionally |

Plane 18 is a plain spec deviation. §8 is not covered by the R5 text, but
"exact for next turn, because next turn's effects are printed" is the same
premise R5 withdrew, so I believe it needs the same branch. The case is rare (12
encodings in the 35-game gate) but it is exactly the reshuffle decision point.

**Asked:** confirm both, and whether §8's fallback should be the same
effect-marginal form. I propose fixing both before step 8.

### 3.2 §8 `roundabout_rescue_available` — which turn?

Spec: "1.0 when `ROUNDABOUT_OPEN` is legal and would change `playable_slots()`".
`playable_slots()` is **this** turn's offer. The implementation compares **next
turn's** per-stack miss sets before and after the best roundabout — consistent
with the block's other three floats, but not what the sentence says. Which is
intended?

Also here: the "best" roundabout is SPEC GAP 3 (§3.3).

### 3.3 The four SPEC GAP rules — sign-off wanted

Each was an open choice in the spec, resolved in code and marked `SPEC GAP` at
the point of use (`encoder.py` module docstring lists them). Rust mirrors each.

1. **SURVEYOR `effect_demand` denominator** = `3 slots × 6 = 18`. SURVEYOR has no
   track to normalise by.
2. **`reshuffle_contraction` is 8 floats carrying 6 floats of information.** The
   temp half's two effect entries equal the no-temp half's by construction
   (`eff_rate` never reads card demand). Kept at the literal width.
3. **"The best legal roundabout" (§8)** = the placement maximising total
   `placement_capacity`, tie-broken by lowest `(street, box)`. It is the same
   quantity `capacity_if_roundabout` maximises, so §4 and §8 agree about which
   roundabout they mean.
4. **Per-street → per-effect reduction in `_effect_needs`.** §3.4's vectors are
   per-street **alternatives**, and summing them is the mistake that once made
   `turns_lower_bound` too high. Rule: use `steps_left` where `progress()`
   already aggregated; otherwise the cheapest alive street, tie-broken by index.

### 3.4 The §6.4 demotion (`b154363`) — is it complete, and what should happen to the code?

User's call: `can_complete_this_turn` / `p_complete_next_turn` are **not
encoded** (they were 96.5% of encode time; batching and dedup were both measured
worthless). The plan slot went 36 → 34 floats. ABI stayed 2, because nothing was
ever generated at the 202-wide layout.

* **Is anything left that assumed them?** In particular `ENCODER_V3_SPEC.md`
  §3.3 still says the exact estate question "is answered by
  `can_complete_this_turn`". The encoder no longer answers it at all.
* **~470 lines of `game.py`** (`one_turn_sheets`, `_resolve_effect`, `_one_turn_hopeless`, the threat
  predicates, `plan_threats`), plus `test_plan_threat.py` and
  `engine_turn_oracle.py`, are now reachable from tests only. They are kept as an
  ablation to re-add. Keep, or delete and rely on git? The old request's §2.2 and
  §2.3 questions about their soundness still apply **if** they are kept.

### 3.5 Integer-numerator draw probabilities (`b154363`)

Bit-exact floats need one defined summation order, and NumPy's pairwise
`.sum()` and Python 3.12's compensated builtin `sum()` are neither a plain loop.
Every boundary-draw probability is now
`draw_probability(num, den, masks) = Σ masked integer numerators / den`
(`deck_knowledge.ordered_draw_counts`), so the order is irrelevant and the only
rounding is the final division.

**Asked:** check the exactness claim. Numerators are products of card counts. At
`D ≥ 3` the largest is `81·80·79`; in the mixed `D < 3` case a deck numerator
(≤ 2) multiplies a pool numerator (≤ 81³). Every masked sum is therefore well
under 2⁵³. Is there a case where a numerator is not an integer, or a negative
factor survives the clip?

The one non-integer reduction, `eff_rate`, is an explicit left-to-right loop in
both languages.

### 3.6 Information-set safety of the new per-seat blocks

Every per-seat read is meant to go through `sheet_for(viewer, seat)` and
`plan_turns_for(viewer, ·)`. §10.5's leak test mutates the live sheet mid-turn
and checks that the opponent's block is unchanged, in both languages. Two things
it does not cover:

* `max_houses_this_turn` and the §8 block evaluate **every** seat against the
  **viewer's** offer. That is correct only because standard mode shares the
  three stacks. Expert raises at the encoder, but these helpers themselves do
  not guard.
* The old request's §5 recorded that `max_houses_this_turn` once read
  `self.sheets[player]` (a live mid-turn sheet) and was found by reading, not by
  a test. It still has no direct information-set test.

### 3.7 Steps 1–4 support code — re-prioritised

Detail is in `ENCODER_V3_BUILD_REVIEW_REQUEST.md` §2. What changed:
`feasible`, `requirements` and `turns_lower_bound` are now **encoded for every
seat on every row**, so they move to the top. The threat enumeration (old
§2.2–§2.4) moves to "only if kept" (§3.4).

* **`plans.feasible`** (old §2.1) — the block with the worst record: five spec
  rounds, four unsound death tests. The claim is one-sided; the failure that
  matters is a declared death the game can escape. Check every kind against
  writes, roundabouts, bis and fences.
* **The oracle's monotonicity pruning** (old §2.5) — accepted deliberately and
  stronger than field-disjointness. If it is wrong, three plan kinds lose
  coverage silently.
* **`requirements()` semantics** (old §2.6) — aliveness lives only in `feasible`
  and `street_serves`; every other field is gated on `done` alone.
* ⚠ `plans._pool_boxes_alive` calls `span_if_roundabout()` with its default
  `available=True`, **ignoring whether the variant has roundabouts**. So in a
  base-rules game a pool box counts as alive via a roundabout that can never be
  built. This is sound, since it only over-counts aliveness, but it is weaker
  than it needs to be. Intended?

`d7bed84` `turn_reach.py`: an unwired step-4 rewrite with three fixes `game.py`
lacks and one measured regression (see its docstring). If §3.4 decides to
delete the threat code, this goes with it.

### 3.8 The ABI guard (`13ea612`)

* A checkpoint **without** `encoder_abi` is taken to be ABI 1. That is true of
  every artifact that exists. Is refusing on absence the right default, or should
  absence be its own error?
* Two legacy paths are now unreachable from real data but kept: the shard
  *target* upgrade (`self_play._decode_wts_targets`, Rust `append_training_targets`)
  and the checkpoint *head-row* expansion (`nw.load_state_dict_compatible`,
  advisor `legacy_heads`). Both have their own unit tests. Spec §0.4 says "no
  legacy head zero-fill". Delete both?

### 3.9 The BGA advisor (`7744273`) — lower priority, separate concern

A diagnostic instrument, not a player. It is the third consumer of the shared
`games.advisor` host, and the adapter supplies only position reading, move
naming, search and the forecast panel.

* **Train/serve skew.** The v3 encoder assumes all six table cards are fully
  identified: number and effect of both the top and the aside card.
  `bga_extract.py` reconstructs `stack_old` from the card pool
  (`pool.take(*stack["old"])`). Please confirm that holds right after a
  reshuffle, where three cards go draw → aside in one transition and never show
  their number face.
* **Mid-turn reconstruction.** `scoreSheet` is start-of-turn for every player;
  the extractor rebuilds the viewer's mid-turn sheet from the DOM patch plus a
  card ledger. Is there a mid-turn state the ledger cannot recover?
* Seat identity uses `gameui.player_id`. A trap recorded elsewhere in this
  project read the wrong seat from a similar field.

---

## 4. Known limitations, disclosed

* **Throughput.** v3 costs **−37% evaluator rows/s** at the production laptop
  config (11,059 → 7,005; inflight 256, 8 workers, 200 simulations), after two
  exact encoder savings (Rust standalone 323 → 108 µs). **12 workers is worse**
  (5,932): about 8 physical cores. Inside the scheduler, encode runs ~3× its
  standalone cost for the same reason. No single hot spot is left. Accepting
  this or cutting further is an open decision for step 8, not a review finding.
* **`reachable_estate_counts` is the deliberately loose bound** (spec §13.2).
  `ESTATE` has the thinnest oracle coverage and the loosest bound.
* **Oracle coverage** (from the old request §3): verified deaths `DECORATIVE`
  346, `COMPLETE_STREET` 56, `FIVE_BIS` 28, `ESTATE` 4, `EXTREMITIES` 1; zero
  false deaths among them. Undecidable: `COMPLETE_STREET` 47, `ESTATE` 12,
  `FIVE_BIS` 2.
* **Scope is the 2+ player standard game** (§0.5). Expert, one-seat play and
  prepared-boundary afterstates raise.
* Python encode is 8.5 ms median. Only the advisor and tests use it.

---

## 5. Already verified — please do not re-litigate

From the old request §4, still true: the temp interval `(low−2, high+2)`
including at the sentinels; `floor(D/3) + 1` for the reform horizon; the
hypergeometric against the worked example (0.545 vs 0.535 for the banned
`1 − (1−p)³`); the closed form `M₁M₂M₃ − ΣP·M + 2R`; bis and temp tracks
saturate rather than gate.

New: both `max_houses_this_turn` shortcuts in `13ea612` (a non-BIS write scores
`placed + 1` wherever it lands; skip a start whose ceiling cannot beat the
running best) are result-preserving, and the gate was re-run after them.

---

## 6. How to run the gates

```powershell
# rebuild the extension first -- the ABI cross-check fires at import otherwise
cd games\welcome_to\welcome_to_rust
..\..\..\.venv\Scripts\maturin.exe develop --release
cargo test --release
cd ..\..\..
.\.venv\Scripts\python.exe -m pytest games/welcome_to/tests -q          # ~10 min
.\.venv\Scripts\python.exe -m games.welcome_to.rust_encode_equiv --encodings 20000   # ~2.5 min
```

Full suite at `13ea612`: **691 passed, 3 skipped** (10 min; the one warning is the long-standing test-only tensor-to-float conversion).

---

## 7. Sign-offs requested

1. §3.1 — plane 18 and §8 must branch on the viewer's vote (and what §8's
   fallback is).
2. §3.2 — which turn `roundabout_rescue_available` refers to.
3. §3.3 — each of the four SPEC GAP rules.
4. §3.4 — keep or delete the demoted threat code (and `turn_reach.py`).
5. §3.5 — the integer-numerator exactness argument.
6. §3.8 — delete the two now-unreachable legacy paths, or keep them.
7. The whole — anything that should change **before** step 8 generates data.

---

## 8. Response

*(for the reviewer's findings and their disposition)*
