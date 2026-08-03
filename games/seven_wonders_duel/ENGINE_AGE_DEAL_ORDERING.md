# Deal the next Age before asking who starts it

**Goal: make the start-of-Age decision match the real game, before the cloud
run. The laptop model keeps the current behaviour; this is a pre-run engine
correction, not a fix to an existing checkpoint.**

Written to be picked up cold. Decided 2026-08-02. **SHIPPED 2026-08-03** — see
"What shipped" at the end, including the two things this plan did not
anticipate.

---

## 1. The deviation

In the physical game and on BGA, the new Age's structure is **laid out first**,
and then the player with the weaker military chooses who begins. The chooser is
looking at the pyramid.

The engine does the opposite. `engine.start_next_age` (`engine.py:873`) is
reached only by *applying* the choice:

```
game.age += 1
supplied = ctx.draw(ChanceKind.AGE_DEAL)
game.tableau = TableauState.from_deck(game.age, deal)
```

Its own docstring says it: *"Prepare the next Age after the military chooser is
resolved."*

So at the moment the net picks a starter, `game.tableau` is the **exhausted
previous age** — 20 slots, all `present=False`. Confirmed live: the observation
at that phase carries 20 tableau entries with 0 present, and the encoder emits
36 tokens with `decision="next_age_starter"`.

**The net has therefore learned this decision blind to the layout it is
choosing about**, across ~168k such decisions (2.0 per game, measured).

This is a rules-fidelity gap, not merely an advisor inconvenience: it is the
only place the engine gives a player *less* information than the real game.

## 2. Why it is worth fixing now and not before

It is 2 decisions out of ~78 a game, so it does not justify invalidating a
trained net. It *does* justify being correct in a fresh run. Fixing it changes
the chance ordering, which changes every game a seed produces — cheap before a
cloud run, expensive at any other time.

**Explicitly accepted:** `laptop_training_03_w7/current_best.pt` keeps the
deficiency. The advisor will keep serving it, and its start-player policy prior
stays layout-blind. Search partly compensates — it plays forward into positions
that *do* contain the real layout and evaluates those normally — so only the
prior is affected, not the value estimates.

## 3. The change

Move the deal from "consequence of the choice" to "entry into the choice":

| | now | after |
|---|---|---|
| age exhausted | `phase = CHOOSE_NEXT_START_PLAYER`, old tableau | `age += 1`, `AGE_DEAL`, new tableau, then `phase = CHOOSE_NEXT_START_PLAYER` |
| applying the choice | `age += 1`, `AGE_DEAL`, deal, set active | set active, `phase = PLAY_AGE` |

The action space is untouched: the two actions stay at indices **1200/1201**
(`codec.py:44-45`), so no retraining is forced by the action encoding and no
codec migration is needed.

**The encoder needs no change either, and this is the elegant part.** It already
computes `present` and `face_down` over `obs.tableau` for every decision
including `next_age_starter` (`encoder.py:557,563-564`); today those are 0 and 0
at that phase. After the change they carry the real structure. The net simply
starts receiving information that was always plumbed.

### Both engines, in lockstep

Python `engine.start_next_age` and Rust `engine.rs::start_next_age` (called from
`ActionUse::ChooseNextStartPlayer`, `engine.rs:371-374`, phase set at `:599`)
must move together or `test_rust_engine_equiv` fails immediately — which is the
desired safety net, not a problem.

## 4. Blast radius — read before starting

### 4.1 Every seed produces a different game

`AGE_DEAL` moving earlier shifts the RNG stream. Same seed, different game. This
is the consequence that reaches furthest.

* Golden/seed-anchored expectations need re-baselining. Grep found candidates in
  `test_codec.py`, `test_encoder.py`, `test_portable_rng.py`, `test_phase_d.py`.
* `test_rust_engine_equiv` compares Python against Rust and both move together,
  so it stays valid — but any **stored** digest inside it must be regenerated.
* Re-baseline by regenerating, never by loosening an assertion. A digest test
  that gets weakened to pass has stopped being a digest test.

### 4.2 The existing buffer becomes unreplayable

`buffer.replay` verifies the `chance_log`, a per-move `mask_hash`, and
`final_digest` / `trajectory_digest`. A different chance ordering fails all of
them, so **every existing `GameRecord` is dead for the new engine** — including
the 84k-game run's buffer, and reanalyze over it.

There is already a seam for this: `SCHEMA_VERSION = 1`, `SPEC_VERSION =
"codec-1"`, `TARGET_VERSION = 2` (`buffer.py:22-25`), with the docstring noting
that *"schema/spec_version cover replay: what a state and an action mean."*
**Bump `SPEC_VERSION`** — the meaning of a state at `CHOOSE_NEXT_START_PLAYER`
genuinely changes — so old records are refused loudly instead of replaying into
a subtly different game. `test_target_version.py` already covers stale-record
handling.

### 4.3 What does not change

* Action space and codec indices.
* Encoder token layout (only the values it reads).
* The advisor's scrape codec — item F in `ADVISOR_IMPLEMENTATION_PLAN.md` gets
  **easier**, because the engine will then agree with BGA about ordering and the
  three seeding mechanisms in that item mostly evaporate.

## 5. Order of work

1. **Python engine**, plus a test asserting the observation at
   `CHOOSE_NEXT_START_PLAYER` now shows the *new* age with a full structure.
   That single assertion is the whole point of the change.
2. **Rust engine** to match; `test_rust_engine_equiv` is the gate.
3. **Re-baseline** the seed-anchored tests by regeneration.
4. **Bump `SPEC_VERSION`** and confirm old records are refused, not misread.
5. **Then** advisor item F, which becomes mostly plumbing.

## 6. Open questions — all three answered

* **Who chooses, and is that already right?** Yes, and untouched.
  `_finish_turn` picks the player behind on the military track (`conflict_position
  > 0` means player 0 leads, so player 1 chooses), and on a tie the player who
  took the last card of the Age. That is the rule.
* **Does anything else read `tableau` during `CHOOSE_NEXT_START_PLAYER`** and
  assume it is empty? No. The encoder already read `obs.tableau` generically;
  the bots score the *child* state; `advisor_scrape` refuses the phase outright.
  What did move is the meaning of `game.age` **during** that phase — it is now
  the new Age, not the exhausted one. One test read it the old way and silently
  changed subject (see below).
* **`age_deal_samples`** — the knob's *root* moved, and it had to be retargeted;
  see below.

## 7. What shipped (2026-08-03)

Both engines moved together, as designed: the AGE_DEAL now fires inside
`_finish_turn` / `finish_turn` via `_deal_next_age` / `deal_next_age`, and
`start_next_age` is reduced to setting the active player and the phase.
`test_rust_engine_equiv` was the gate and is green.

Two consequences this plan did not anticipate, both structural rather than
incidental:

**1. Predicting the event got harder, and the fix is a dry run.**
`chance_signature` has to say — from public information, before applying —
which events an action fires. The starter choice was trivially predictable; the
take that empties the pyramid is not, because whether it *ends the Age* depends
on whether the same action wins the game (military or scientific) or defers into
a pending choice, in which case the deal rides on the later
RESOLVE_PENDING_CHOICE instead. Re-deriving those rules in the searcher would be
a second copy of the engine to keep in sync, in two languages.

So `_exhausts_the_age` (and Rust `exhausts_the_age`) applies the action to a
throwaway clone and reads the resulting phase. A cheap public precondition —
one present slot for a take, none for a pending resolution — gates the clone, so
it runs on the one take per Age that can empty the pyramid and never in the hot
path. The result is still a function of public information, which is what the
leak-free contract requires: the last card is face up, and so is everything that
decides whether taking it ends the game.

**2. The paired AgeDeal sampler had to be retargeted or it became a no-op.**
`materialize_paired_age_deals` keyed off `phase == ChooseNextStartPlayer` — a
root that now carries no chance at all, so `age_deal_samples` would have
silently stopped doing anything. It now selects edges whose *only* chance event
is the deal, requiring at least two of them. That keeps the intent (compare
sibling actions under one common set of deals), moves it to the root where the
deal actually happens, and drops the "root AgeDeal actions do not share one
chance signature" hard error for the case that can now arise legitimately: the
Great Library built with the last card of an Age draws as well as deals, so it
cannot share the sampled outcomes and is left to ordinary sampling.
`is_paired_age_deal_root` in `f4_corpus.py` is the one definition of that
filter, used by the diagnostic generator and the tests.

**Re-baselined by regeneration, never by loosening** — as §4.1 required:

* `test_search.py::test_closed_search_samples_hidden_boundaries_...` searches
  the last take, and additionally asserts the chooser root now has no specs.
* `test_search.py::test_age_three_deal_samples_...` moved to the last take of
  Age II. It had been passing for the wrong reason: it selected on
  `age == 2` at the boundary phase, which used to mean the Age II→III
  transition and now means Age I→II.
* `test_chance.py::test_age_transition_emits_age_deal_and_respects_barrier`
  moved to the exhausting take.
* `test_f4_boundary.py`'s two paired-sampler tests moved to a pairable root, on
  seed 1 — seed 3 has a single legal action at both of its Age-ending takes.

`SPEC_VERSION` is `codec-2`, and `replay` refuses an older record up front with
`StaleSpecVersionError` rather than letting it fail confusingly on a digest.
The only casualty in the tree is `test_bf16_real_position_fidelity`, which reads
run-03's buffers for 512 real positions; it now skips on that refusal, since its
subject is bf16 numerics and not engine semantics.

**Still to do:** advisor item F (`ADVISOR_IMPLEMENTATION_PLAN.md`), which this
change was sequenced ahead of.
