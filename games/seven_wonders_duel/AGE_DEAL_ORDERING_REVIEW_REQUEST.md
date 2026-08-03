# Review request: dealing the next Age before the starter choice

**Commit `38b144b` on `bga-advisor-live`, 17 files, +457/-87.** Plan and
outcome: `ENGINE_AGE_DEAL_ORDERING.md`. Normative spec touched: `CODEC_SPEC.md`
§4.2.

Please review for **logic**, **assumptions**, and **throughput**. Findings that
change the code matter; style does not.

---

## What changed, in one paragraph

The engine dealt the next Age as a *consequence* of the start-player choice, so
the chooser decided blind to the pyramid. The physical game and BGA deal first.
The deal now fires when an Age is exhausted, inside `_finish_turn` /
`finish_turn`, and `start_next_age` reduces to setting the active player and the
phase. Both engines moved together and `test_rust_engine_equiv` is the gate.
`SPEC_VERSION` → `codec-2`; `replay` refuses older records up front.

Everything else in this document is a claim I am asking you to attack.

---

## 1. The dry-run prediction — the change I am least sure of

`chance_signature` must name the chance events an action fires, **before**
applying it, from public information. That was trivial when the deal rode on the
starter choice. It is not now: whether the take that empties the pyramid ends
the Age depends on whether the same action wins the game (military or
scientific) or defers into a pending choice — in which case the deal rides on
the later `RESOLVE_PENDING_CHOICE` instead.

Rather than re-derive those rules in the searcher in two languages, both
implementations apply the action to a throwaway clone and read the resulting
phase:

* `search.py:_exhausts_the_age` (~line 102)
* `seven_wonders_rust/src/chance.rs:exhausts_the_age` (~line 56)

Gated on a cheap public precondition: exactly one present slot for a take, zero
for a pending resolution, and `age < 3`.

**Claims to attack:**

1. **The precondition is exhaustive.** I claim the Age ends *only* when the take
   removes the last present slot, because the pyramid always leaves a bottom row
   uncovered, so `accessible_slot_ids()` is empty iff no slot is present. If
   there is a reachable state with present slots but none accessible, the deal
   fires with no spec predicted and search dies on `HiddenInformationError`.
2. **The output is leak-free.** The clone reads hidden state to run; I claim the
   *result* is a function of public information only — the last card is face up,
   and so is everything deciding whether taking it ends the game (shields,
   tokens, conflict position, science symbols). If that is wrong, the spec list
   itself leaks, which the barrier design exists to prevent.
3. **Clearing the barrier on the clone is safe.** `search.py:129` sets
   `clone.search_barrier = False` so the clone's own AGE_DEAL resolves from the
   locked deck. This assumes every state reaching `chance_signature` is a
   consistent world with `age_decks[next_age]` populated (true for engine-built
   and determinized states). Is there a path — advisor, injection, a partially
   reconstructed scrape — where it is not?
4. **Both languages agree at the boundary.** Rust pre-installs supplied outcomes
   *before* `apply_action` while Python overrides mid-apply, so `validated_age_deal`
   computes its `visible` set at different moments — differing by exactly the
   card this action takes. I argued that is inert because the taken card belongs
   to the *previous* Age's universe and the deal is for the next, so it can
   affect neither `removed_age_cards[next_age]` nor the "already visible" check.
   Please check that argument rather than trust it; the Age II→III boundary is
   the case to think hardest about.
5. **Failure modes stay loud.** Over-prediction raises "unconsumed chance
   outcome(s) supplied"; under-prediction raises `HiddenInformationError`.
   Neither degrades silently — is that actually true on the Rust path, where
   `apply_with_chance` compares `specs.len() != outcomes.len()` up front?

**What backs this today:**
`test_search.py::test_chance_signature_matches_engine_events_for_every_legal_action`
checks every legal action at every state of two full games (>500 actions)
against real `StepResult` events, and
`test_rust_engine_equiv::test_chance_signature_and_chains_equivalent` checks
Rust against Python. Both green. **Neither is a proof of (1) or (2)** — they
sample games rather than enumerate the awkward states, and a state that never
arises in a random playout is exactly the one that would break this.

---

## 2. Throughput — the numbers I have and the ones I do not

**Measured (static counts, 60 random games, `/tmp` probe, not committed):**

| | before | after |
|---|---|---|
| paired root | starter choice, exactly **2** actions | last take, mean **2.35**, max **5** |
| forced NN rows per boundary at `age_deal_samples=32` | 64 | ~75, worst case 160 |
| boundaries per game | 2 | 2 |
| roots where pairing applies | 120/120 | 106/120 |

`age_deal_samples` defaults to **32** in production self-play
(`phase_d.py:465`), so this is a training-path change, not a diagnostic one. I
had initially mis-read the Rust comment as implying the knob was off by default;
it is not.

**Claims to attack:**

6. **The dry run is not in the hot path.** It runs once per legal action at
   expansion (`search.py:_expand_closed`), and only when the precondition holds
   — one present slot, so a handful of actions, twice per game. I claim the cost
   is lost in the noise. **Not benchmarked.** If you disagree, the cheap fix is
   to hoist the clone: all actions at that root share the same answer for the
   "does the Age end" question except where a pending choice or a victory
   intervenes.
7. **Rust clones a whole `GameState` where the crate otherwise uses make/unmake.**
   `chance.rs:exhausts_the_age` does `g.clone()` — deliberately against the
   grain of the F1 make/unmake work. Same precondition argument applies, but
   this is the line most likely to be a real regression, and F4's throughput
   programme is the context it lands in.
8. **14 of 120 boundaries are not pairable** (single legal action at the last
   take). Those roots now get ordinary sampled chance where they previously got
   32 paired deals at the starter choice. I claim that is fine because with one
   action there is nothing to pair. Does the variance argument survive that?

---

## 3. The paired sampler retarget

`materialize_paired_age_deals` (`tree_resumable.rs:~642`) keyed off
`phase == ChooseNextStartPlayer` — a root that now carries no chance at all, so
`age_deal_samples` would have become a **silent no-op** in training. It now
selects edges whose *only* chance event is the deal, requiring ≥2.

**Claims to attack:**

9. **Dropping the hard error is right.** The old code raised "root AgeDeal
   actions do not share one chance signature" if signatures differed. That case
   can now arise legitimately: the Great Library built with the last card of an
   Age draws *and* deals. I left it to ordinary sampling instead of erroring.
   The alternative — pair the deal component and sample the draw separately —
   is more faithful and more code. Is the simple version acceptable?
10. **Mixing closed and open edges at one root is sound.** After this, a root can
    hold fixed-support (closed) edges beside ordinary sampled ones. The
    fixed-support invariant is per-edge, so I claim yes, but
    `CHANCE_ENUMERATION_PLAN.md` is the authority and I did not re-read it in
    full.
11. **One definition, two consumers.** `f4_corpus.is_paired_age_deal_root`
    mirrors the Rust filter and is used by `f4_age_deal_positions.py` and the
    tests. Nothing enforces that the mirror stays true — should it?

---

## 4. Assumptions about the trained model

12. **`current_best.pt` still plays, with one decision off-distribution.**
    Measured over 40 games: the `next_age_starter` encoding goes from 43.5
    tokens mean (32–54) to 63.5 (52–74), and the global `present` / `face_down`
    features move from always-`0/0` to the real layout. The net never saw those
    inputs at that phase. I claim this is bounded — 2 of ~78 decisions, and
    everything downstream of the choice is in-distribution, so only the prior at
    that node is affected, not the values search reads. **I have not measured
    the strength cost.** If you think a head-to-head against the pre-change
    engine is warranted before the cloud run, say so.
13. **The chooser is unchanged and correct.** `_finish_turn` picks the player
    behind on the military track, ties to the player who took the last card.
    Unmodified by this work; I assert it matches the rules.

---

## 5. Version and buffer handling

14. **`SPEC_VERSION` → `codec-2` with refusal in `replay`, not in the reader.**
    Reading stays permissive so stale buffers can still be opened; replay raises
    `StaleSpecVersionError` immediately rather than failing on a digest hundreds
    of moves later. `test_target_version.py` pins both halves. Is `replay` the
    right chokepoint, or should `check_target_versions`-style enforcement sit at
    the training boundary too?
15. **`test_bf16_real_position_fidelity` now skips.** It read run-03's buffers
    for 512 real positions; those are unreplayable. I made it skip on the
    refusal because its subject is bf16 numerics, not engine semantics. The
    alternative is regenerating a small corpus under `codec-2` and pinning it.
    Skipping loses real coverage of a CUDA-only path — is that trade acceptable?

---

## 6. Re-baselined tests — please check I regenerated rather than loosened

§4.1 of the plan required regeneration. Four tests moved:

* `test_search.py::test_closed_search_samples_hidden_boundaries_...` — searches
  the last take; **added** an assertion that the chooser root now has no specs.
* `test_search.py::test_age_three_deal_samples_...` — moved to the last take of
  Age II. **This one had been passing for the wrong reason**: it selected on
  `age == 2` at the boundary phase, which used to mean the II→III transition and
  now means I→II. It stayed green through the change while silently testing a
  different position. Worth asking whether other `age`-at-that-phase reads exist
  that I did not find; I grepped `CHOOSE_NEXT_START_PLAYER` across `*.py`/`*.rs`
  and found none, but the failure mode here was a test that *passed*.
* `test_chance.py::test_age_transition_emits_age_deal_and_respects_barrier` —
  moved to the exhausting take.
* `test_f4_boundary.py` paired-sampler tests — moved to a pairable root, **and
  the seed changed 3 → 1** because seed 3 has a single legal action at both of
  its Age-ending takes. A seed change to make a test pass deserves suspicion;
  the justification is that seed 3 no longer contains the position under test at
  all, not that it failed.

New: `test_engine.py::test_the_next_age_is_dealt_before_the_chooser_is_asked` —
the point of the whole change. Asserts the AGE_DEAL rides on the exhausting
take, the chooser's observation carries all 20 slots of the new Age with 12
revealed, the choice itself fires no events, and the layout the chooser saw is
the one that gets played.

---

## What I did not do

* No benchmark of the dry run or of the paired-sampler row-count change (claims
  6, 7).
* No strength measurement of `current_best.pt` under the new ordering (claim 12).
* No re-read of `CHANCE_ENUMERATION_PLAN.md` in full (claim 10).
* Advisor item F, which this was sequenced ahead of, is untouched — the scrape
  codec still refuses `CHOOSE_NEXT_START_PLAYER`.

## Verification that did run

`games/seven_wonders_duel` + `games/advisor`: **816 passed, 1 skipped**
(the run-03 bf16 test). `cargo test --release`: 16 passed.
`test_rust_engine_equiv`: 32 passed, including the chance-signature parity gate
and the Python/Rust chance-log comparison that caught the mid-port divergence at
move 28.
