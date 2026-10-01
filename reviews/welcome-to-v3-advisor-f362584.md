# Welcome To encoder v3 and advisor review

Reviewed 2026-09-25, HEAD `f362584`, implementation through `13ea612`, against the request's `26696f6` baseline. Production files were not changed. The findings below remain open.

**Recommendation: fix the encoder correctness findings before step 8.** Python/Rust equivalence is valuable, but both implementations share the false-death and impossible-temp-fit errors below. The advisor also has several reconstruction failures at ordinary, strategically important positions.

This review used the implementation, the requested portions of the spec, the earlier review request, and the bundled BGA PHP/JavaScript source. It did not repeat the already-recorded full suite or 20,000-encoding gate. BGA findings are verified against that bundled source and synthetic captures, not a new live-table session.

## Findings

### F1 — P1: EXTREMITIES can be declared dead when a roundabout completes it

Locations: [plans.py:544](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/plans.py:544), [Rust plans.rs:471](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/welcome_to_rust/src/plans.rs:471).

The death check requires `span_if_roundabout > 0` or bis reachability, but omits the spec's separate “no roundabout remains” condition. `span_if_roundabout` considers placing a roundabout **elsewhere**, whereas an extremity can itself be filled by a roundabout.

Reproduction: street 0 begins `_ 0`, with a fence between those boxes; the other five extremities are written and unconsumed. The left extremity has span zero and no bis access. With a roundabout still available, `feasible(PLANS[22], sheet)` is false, but placing a roundabout at `(0,0)` makes `can_be_scored` true. The independent reachability oracle also returns true.

This is a false death, not the disclosed conservative over-counting. It corrupts feasibility, street-serving flags, and any plan-conflict calculation that calls feasibility. Guard the numeric/bis death clause with roundabout unavailability, in both languages. Add this explicit regression; the random oracle workload's one verified extremities death did not cover it.

### F2 — P1: TEMP fit reports positive probability for a gap containing no integer

Locations: [encoder.py:617](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/encoder.py:617), [encoder.py:425](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/encoder.py:425), [Rust encoder.rs:714](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/welcome_to_rust/src/encoder.rs:714).

The widened interval is valid only when the original gap contains at least one legal integer. For `7 _ 8`, no house number can satisfy the ascending rule, even with TEMP. Widening `(7,8)` to `(5,10)` incorrectly makes printed numbers 6–9 look useful.

A focused state produced `P_FIT_TEMP = 0.40000000596` and `P_FIT_NEXT_TURN = 0.40000000596` for that box; enumerating all legal values 0–17 found none. Rust matched Python exactly. This is a missing precondition on the already-reviewed interval formula, not a dispute about its endpoints or its valid nonempty-gap cases.

Before widening, require a nonempty integer interval inside 0–17. Otherwise emit zero for both temp fit and next-turn fit. Test consecutive/equal bounding numbers and the empty sentinel-edge cases in addition to ordinary gaps.

### F3 — P1, already disclosed: queued reshuffles use the wrong draw pool and effects

Locations: [encoder.py:359](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/encoder.py:359), [encoder.py:369](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/encoder.py:369), [encoder.py:1079](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/encoder.py:1079); Rust mirrors these paths.

Confirmed: plane 18 and the first three refusal probabilities ignore the viewer's yes vote. This is both an effect error and a population error: `ordered_joint()` continues to use the old remaining deck and natural-reform pool. Even `p_printed_unplaceable`, which needs no effects, must switch to the reshuffled number population.

Use only `reshuffle_vote_for(viewer)`, never the hidden aggregate vote. Marginalize over the new effects for the first two refusal fields and plane 18, and use the reshuffled three-number marginal for the printed-only field.

**An important qualification to the proposed fallback:** the engine draws six distinct cards after a queued reshuffle. The first triple supplies effects; the second supplies numbers. A stack's number and effect come from different cards. Exact marginalization must preserve this two-triple sampling, including depletion between the triples. Do not implement it by treating three cards as each supplying their own number/effect pair, or by multiplying independent marginals. F6 demonstrates the difference.

The global `next_effects` block also continues to emit the old one-hots after a yes vote. Either make its meaning explicitly “effects if no reshuffle,” or replace/invalidate that certainty consistently. The existing vote flag supplies context, but the current “known next turn” documentation is false in this branch.

### F4 — P2: requirement vectors still erase demand on dead streets

Locations: [plans.py:631](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/plans.py:631), [plans.py:640](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/plans.py:640), [plans.py:655](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/plans.py:655); [Rust plans.rs:561](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/welcome_to_rust/src/plans.rs:561).

The `FIVE_BIS`, `COMPLETE_STREET`, and pool-dependent decorative branches populate their demand vectors only after a street passes an aliveness test. The later comment saying demand is gated on `done` alone does not undo those early skips.

Reproduction: plan 27 requires pools and parks in street 1. Fill that street with ordinary numbers, give it two pools and zero parks. The unfinished, dead plan returns all-zero `pools_needed` and `parks_needed`, despite still wanting one pool and four parks. Its effect-rate estimate consequently also collapses to zero.

Populate each relevant street's remaining work independently of aliveness; gate `street_serves` separately. Keep the whole-plan `done` clearing rule. Extend fidelity tests to the non-estate kinds and to dead individual alternatives of otherwise alive plans.

### F5 — P2: the live houses-this-turn feature ignores the viewer's current phase

Locations: [game.py:1718](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/game.py:1718), [game.py:1745](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/game.py:1745), [Rust encoder.rs:1190](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/welcome_to_rust/src/encoder.rs:1190).

The helper always starts a fresh roundabout/choose/write/effect sequence from the current sheet. The encoder calls it at mid-turn decision nodes too. In a base-game `ACTION_BIS` state, the probe returns two houses, while only the optional single bis remains legal. In advanced play it can also offer another hypothetical roundabout despite `last_house` or `roundabout_declined` prohibiting it.

The information-set accessor is now correct: directly mutating an opponent's hidden live sheet left this helper unchanged in the probe. The issue is temporal semantics, not an opponent leak.

Choose a contract before freezing data: either calculate the viewer's legal remaining-turn ceiling using their context, retaining a full-turn potential for opponents' public snapshots, or rename this a hypothetical full-turn potential and stop describing it as “currently legal.” Under the requested spec, the former is the fix. Boundary symmetry still holds; mid-turn private context need not be symmetric.

### F6 — P2: steady refusal assigns a card its own effect instead of its stack partner's

Locations: [encoder.py:1106](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/encoder.py:1106), [encoder.py:1124](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/encoder.py:1124), [Rust encoder.rs:1162](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/welcome_to_rust/src/encoder.rs:1162).

The steady term counts unplayable `(number, effect)` cells as though both faces of the same card formed a playable combination. Standard mode pairs the current number with the previous card's effect. The hypergeometric arithmetic is correct for the coded surrogate population, but that population describes a different offer mechanism.

An exhaustive six-card illustration uses `(3,TEMP), (4,TEMP), (7,PARK), (8,SURVEYOR), (9,BIS), (10,ESTATE)` and a sheet writable only at value 8. Draw three effects, then three numbers without replacement: some stack fits in 540/720 outcomes, or 0.75. The current same-card model gives 0.50. These are all valid construction-card faces.

Use a declared future horizon and the actual pairing process, with reform handling, or explicitly rename this field as an approximation. Do not reuse its same-card calculation as the exact reshuffle fallback. Its current assertion of exactness is unjustified.

### F7 — P1 for serving: ordinary plan completion is absent from the advisor's capture

Location: [bga_extract.py:356](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/bga_extract.py:356).

`validated_plan_slots` comes only from `gamedatas.planValidations`. In the bundled BGA client, `notif_scorePlan` draws a plan stamp through `validateCurrentPlayerPlan`; it does not update that gamedatas array. The array updates at the turn boundary. The capture does not read the plan stamps or interpret `args.selectedPlans` as completed-plan evidence.

Thus, immediately after a real validation at `askReshuffle`, replay has no record of the plan just scored. It passes the plan stage and runs past the viewer's turn. A synthetic capture with the actual stale-array behavior reproduces that error. The round-trip test misses it because its renderer inserts live engine `plan_turns` into the purported BGA payload.

Capture current-turn plan stamps, reconcile them with private state arguments and past validations, then replay them. Relevant BGA source: `modules/js/States/PlanValidationTrait.js:26` and `modules/js/wtoPlanCards.js:105`. Add a fixture preserving the stale array rather than regenerating it from engine truth.

### F8 — P2: a mid-turn page reload replays an already-applied house

Location: [bga_extract.py:448](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/bga_extract.py:448).

The assertion that every `scoreSheet` is turn-start data is true for the usual post-boundary cached payload, but false after a page load during the viewer's turn. BGA `getAllDatas()` calls `Houses::getOfPlayer` and `Scribbles::getOfPlayer`; both filter current-turn marks only for **other** players. The viewer's fresh payload already contains their live marks.

The extractor builds the starting sheet without its available `up_to_turn` filter and then replays the same DOM house into an occupied box. The probe fails with an illegal `WRITE_NUMBER` action. Filter the viewer's starting sheet at the current turn, while retaining the marks for replay. Test captures from both an uninterrupted page and a reload after each kind of action. Stale marks after undo/restart deserve the same reconciliation check.

### F9 — P2: adjacent estates cannot be reconstructed from contiguous top fences

Location: [bga_extract.py:814](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/bga_extract.py:814).

`_contiguous_runs` merges adjacent consumed estates even when an estate fence separates them. A legal plan-2 selection of three adjacent size-3 estates at street-0 starts 0, 3 and 6 produces one nine-house top-fence run. `_validate_action` searches for a run of length 3 and raises.

The probe constructs the three legal estates and confirms the plan is scoreable before reproducing the replay failure. Reconstruct using the sheet's actual free-estate boundaries and membership in the observed consumed-house set; contiguous marks alone are insufficient. This remains necessary after F7 is fixed.

### F10 — P2: natural deck reform reintroduces old ledger entries

Locations: [bga_snippet.js:319](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/bga_snippet.js:319), [bga_extract.py:556](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/bga_extract.py:556).

When `cardsLeft` increases, `mergeLedger` clears storage and immediately rescans every card holder in the DOM. An explicit reshuffle notification clears those holders, but natural empty-deck reform does not: BGA's automatic `Pieces::reformDeckFromDiscard` has no corresponding client-clear notification. Normal `newTurn` removes only an existing holder of a card being redrawn.

After the first full pass through the deck, the DOM still contains all 81 identities. The next capture therefore marks the entire new deck as already seen again. A corresponding mapper probe reconstructs **zero** undrawn cards when BGA has **75**. `_deck_from_ledger` warns but leaves the contradictory deck size in the state, so search and forecasts use the wrong process.

Track a reveal epoch and newly revealed cards across reforms; old DOM membership cannot mean “drawn in this epoch.” On detection, keep only proven post-reform observations, with explicit uncertainty for missed events. Do not return an engine state whose deck size contradicts `cardsLeft` merely because a warning was attached.

## Requested sign-offs

| Request | Disposition |
|---|---|
| §3.1 vote branch | **Required before data generation.** All three refusal probabilities need the pool change; the first two and plane 18 also need effect marginalization over the actual two-triple draw. See F3/F6. |
| §3.2 rescue turn | Use **this turn**, as `ROUNDABOUT_OPEN` and `playable_slots()` specify. For the viewer, honor phase and roundabout guards; for opponents use a hypothetical start-of-turn interpretation on their public sheets. If a next-turn indicator is wanted, name it separately. The present comparison can fire on a number absent from the deck, so it is not even an event with positive next-turn probability. |
| SPEC GAP 1, SURVEYOR scale 18 | Accept as a stable descriptor scale, not a fence-action count. Eighteen is a conservative bound, not the maximum actually attainable under the three dealt stacks: estate requirement counts max out at 6+4+4=14. No correctness problem from the larger denominator. |
| SPEC GAP 2, eight values carrying six | Correctly disclosed, numerically harmless. Prefer removing the two exact duplicates before the final ABI freeze; keeping them is also acceptable if avoiding layout churn. Do not expect a meaningful throughput recovery from deleting eight floats across four seats. |
| SPEC GAP 3, best roundabout | Do not sign off on calling maximum capacity “best” for refusal risk. Accept only under an explicit name such as “refusal after capacity-maximizing roundabout.” `capacity_if_roundabout` returns independent per-street optional maxima, not one global selected action, so the claimed agreement is overstated. Prefer minimizing refusal probability, and compute rescue existentially over legal roundabouts. The cheaper draw calculation below makes this more practical. |
| SPEC GAP 4, alternative-street reduction | Accept the rule for avoiding sums across alternatives, **after F4**. For COMPLETE_STREET, cheapest mark count is a deterministic heuristic, not the fastest rate; a rate-weighted choice is a modeling improvement. For two-street pool plans, `progress()` can favor an impossible street; alive alternatives should inform a feature called expected completion time. |
| §3.4 threat demotion | No active encoder fields still emit the pair. Delete the unused threat implementation and `turn_reach.py`, relying on Git for retrieval. Preserve `one_turn_ceiling`, `max_houses_this_turn`, `bis_usable`, and supporting code still called by them; retain/move their tests rather than deleting the entire old test file blindly. Update the spec's estate-answer claim and other stale threat comments. No full soundness sign-off is given to code recommended for removal. |
| §3.5 integer exactness | Accept for production integer count vectors and binary masks. All products and masked sums fit exactly in float64; clipped invalid repeated-card terms cannot leave negative counts. For a mixed draw the total numerator sum equals `(D)_d × (P)_(3-d)`, far below 2^53. The final division rounds; complements such as `1-p` are separate floating operations. Arbitrary fractional inputs accepted by the helper would not satisfy the proof. |
| §3.6 information-set safety | New per-seat reads inspected use viewer-safe accessors. Direct hidden-sheet mutation passed for `max_houses_this_turn`. Add a permanent regression and standard-mode guards to exported offer-dependent helpers, or keep them explicitly internal. F5 remains a phase issue. |
| §3.7 feasibility/oracle | F1 blocks sign-off. Other death tests inspected remain conservative: the loose estate count over-counts existing supply, and pool/bis reachability over-counts possible completions. The oracle's standalone-fence pruning is sound for the current rules: removing a fence cannot inhibit a retained write, roundabout, park, temp, or bis move; only estate predicates observe the removed fence directly. Its bounds and over-approximations should stay explicit. |
| §3.7 base-game pool reach | Allowing roundabouts is an overestimate, not a false death. More specifically, `_pool_boxes_alive` is reached by pool/decorative/complete-street plans, which are advanced-only in the actual dealt pool. This disclosed base-game concern does not affect ordinary valid base-game deals. A future generalized API should still receive variant availability explicitly. |
| §3.8 unstamped checkpoint | Refusing it is correct. “Unstamped; treated as legacy ABI 1” is a useful diagnostic; absence need not be a separate exception class. Never infer compatibility from tensor widths alone. |
| §3.8 legacy migrations | Delete both unreachable shard-target upgrade paths and legacy head-row expansion/neutral-head UI support. Keep rejection tests. Matching current ABI metadata should lead to strict current shapes, not a second migration mechanism. This simplifies the contract; existing old-artifact rejection is already effective. |

The ABI guards inspected cover S0 training load, S2 load, advisor load, and both shard readers. No additional bypass was found in those changed paths. After semantic corrections, explicitly inventory the v3 smoke-test artifacts too: they used ABI 2 even though no production run has begun. Either discard those artifacts as part of the documented pretraining freeze or advance the ABI/shard versions; do not silently mix them with corrected features.

## Throughput opportunities, in priority order

### 1. Replace the dense three-draw contractions with integer inclusion-exclusion

The expensive representation is unnecessary even for the union of many non-contiguous sheet gaps. For arbitrary binary per-stack masks, let `Si = Σ c[n] mi[n]`, `Pij = Σ c[n] mi[n] mj[n]`, and `T = Σ c[n] m0[n] m1[n] m2[n]`. Then the ordered masked numerator is:

```
S0*S1*S2 - P01*S2 - P02*S1 - P12*S0 + 2*T
```

The same identity already used for plane 18 applies to refusal masks; it does not require intervals or nested sets. For D=1, multiply the one-draw deck numerator by the two-draw pool numerator `S1*S2-P12`. For D=2, use the two-draw deck numerator times the one-draw pool numerator. D=0 uses the pool triple. Multiply integer numerators and denominators before the single final division, preserving current bit equality.

The review prototype matched the current implementation exactly on **5,000** count/mask cases covering 6 and 15 classes and D=0,1,2. An optimized standalone Rust benchmark compared a faithful copy of the current precomputed-joint contraction with this formula: **720.0 ns versus 24.6 ns per call**, 256 bit-identical comparisons, synthetic masks. That is about **29× for this kernel only**. It excludes joint construction, sheet work, scheduler contention and GPU work; it is not a predicted end-to-end gain.

This removes the 3,375-entry number joint, its allocation, and repeated scans in each seat's refusal block. Effect availability can likewise use falling factorials over the effect complement. Preserve a literal enumeration as a test oracle, not a production prerequisite. Implement and measure this before cutting model signal to recover speed.

### 2. Build reusable sheet geometry, rather than repeatedly scanning for it

Rust already shares per-plan `SlotFacts`; that optimization should not be proposed again. The remaining opportunity is below it: one gap list, ordinary-writable value bitset, estate/free-estate histograms, spans, and per-street capacity per sheet. Reuse them in planes, demand, refusal, plan requirements and scores.

In particular, `writable_values` calls a location enumerator 18 times and allocates locations just to ask whether any exist. A union of gap value intervals supplies the same 18-bit answer directly. The TEMP-playable printed mask follows from that set. The special hypothetical plan-conflict calls change top fences, not number geometry, so number/gap results remain reusable there.

### 3. Share roundabout work and update only the affected street

`capacity_if_roundabout`, `best_roundabout_sheet` and `max_houses_this_turn` independently visit candidate roundabouts. The best-sheet helper clones an entire sheet and rescans all three streets per candidate even though only one street's geometry changes. Reuse per-street results; materialize only the selected sheet when necessary. Distinguish “optional capacity repair” from the refusal objective rather than forcing both into a misleading common result.

### 4. Cache stable facts at information-set boundaries

During a simultaneous turn, opponents' public sheets and the public deck composition recur across many evaluated decisions and determinizations. Cache small immutable geometry/deck facts with content-based keys, scoped to the search/turn. Include the viewer's own vote in vote-dependent data and plan completions in banked/conflict data. Do not key by live opponent sheets or determinized deck order, and do not cache a viewer's phase-sensitive answer as a public fact.

This is separate from the already-measured failed batching/deduplication of the demoted threat enumeration. Measure hit rate and lookup cost before adding a large full-encoding cache.

### 5. Measure total work before changing worker count or feature width

Keep the production geometry from the request for the comparison. Report encode/search/packing/evaluator time, allocation pressure, batch width, games/hour and playing strength per fixed wall time. The reported regression from 8 to 12 workers argues against adding concurrency as the first remedy. Four output vectors currently contain 4,355 float32 values, about 17 KiB per row; an `encode_into` interface with reusable worker buffers may reduce allocations/copies, subject to scheduler ownership.

Removing two redundant scalar entries or converting tiny constants will not plausibly explain the observed 37% loss. The analytic contraction and repeated geometry work are concrete operations to measure first.

## Modeling improvements

1. **Represent estate construction jointly.** Most dealt plan identities are estate multisets, yet target planes are empty for them and the exact completion features were removed. Add a compact interval/partition calculation over each length-10/11/12 street: candidate estates, missing houses, required boundary fences, existing uncuttable bis joins, and consumed houses. A bounded dynamic program over the plan's required size counts can expose joint achievable multisets or minimum remaining writes/fences. Unlike six independent size upper bounds, it cannot reuse the same region simultaneously for two incompatible estates. Start as descriptive features; prove any new negative feasibility decision against the oracle before using it to declare death.

2. **Measure useful effects at the required locations.** A POOL card matters only when its paired number can be written at an unbuilt pool position; a PARK must serve the needed street. The current global effect rate says neither. Cheap per-plan/per-street useful-effect availability, with the proper pairing process, would make the surviving rate fields better substitutes for removed threats. Distinguish exact next-turn opportunity from a longer-run rate assumption. For alternative streets, compare rate-weighted cost or retain a small vector of alternatives rather than choosing the least number of marks.

3. **Model the competition between completion time and game end.** Existing `will_complete_plan` and conditional `turns_to_plan` heads help, but expected completion time alone can mislead when some games end before completion. Consider discrete “complete within 1/2/4/8 turns” probabilities, jointly assessed against an end-of-game survival curve or remaining-turn distribution. Treat never-completed plans as censored/failure outcomes explicitly, and evaluate calibration near the three-permit/all-plans/full-sheet end conditions. Keep feasibility as a structural bound, not a predicted outcome.

4. **Preserve estate-selection option value.** The canonical lowest-index estate selection makes conflicts deterministic but can hide an equally valid completion that preserves another plan. A cheap alternative is minimum/maximum overlap or a “some satisfying selection preserves plan B” feature. Another is to attach affected-estate and consumed-box descriptors to validation actions so the policy can compare the actual choices. Do not interpret a canonical-selection kill as an unavoidable kill.

5. **Use local structure without pretending streets are interchangeable.** The lack of a whole-board reflection/translation symmetry does not rule out shared local computations. Gap/estate tokens with street, coordinate, pool and park-cap embeddings could share number-spacing and fence reasoning while retaining location. A small relational encoder over sheet, gap and plan tokens is an experiment, not a prerequisite for step 8. Compare it with the existing shared-seat MLP at equal wall-clock search cost.

6. **Treat incomplete advisor knowledge as a belief, not one permanent guessed deck.** The current seeded missing-discard guess gives a precise forecast conditional on one arbitrary composition. For diagnostic use, average a modest number of consistent deck reconstructions or display uncertainty across them, and separate it from search variance. Correct the exact-ledger lifecycle first; uncertainty modeling should not conceal a capture bug.

Before a large run, use small, paired-seed ablations on base rules, then advanced rules and each seat count. Compare both equal simulations and equal wall time. Include plan completion by slot, joint multi-plan completion, refusal/full-sheet outcomes, score components, and calibration at rare decisions. Whole-game averages alone can conceal the estate-planning weakness the v3 features are meant to address.

## Verification and limits

The retained [focused probes](C:/Users/joeld/projects/boardgame-ai/reviews/welcome_to_v3_review_probes.py) reproduce F1, F2, F4, F5 and F7–F10, check the direct hidden-sheet mutation, verify 5,000 analytic draw comparisons, and exhaustively demonstrate F6's pairing issue. The [Rust microbenchmark](C:/Users/joeld/projects/boardgame-ai/reviews/welcome_to_draw_kernel_bench.rs) checks 256 bit-identical outputs and measures the isolated probability kernel. All probes completed successfully; their assertions intentionally confirm the reviewed defects, not corrected behavior.

Run the Python probes from the repository root with the project venv and the escalated permissions required by AGENTS.md. No production implementation or model weights were changed, no new shards were generated, and no full-suite pass is claimed for a corrected implementation. After fixes, add permanent regressions for these independent counterexamples, then repeat the established Python/Rust and information-set gates once on the final semantics.
