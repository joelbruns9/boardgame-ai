# Review request — pre-retrain work, 2026-08-17/18

**Scope:** `831336c~2..HEAD`, 18 commits, 39 files, +5,937 / −147.
**Suite:** 1,019 passed / 5 skipped / 0 deselected (the two engine-equivalence
tests are green again for the first time in weeks — see §5). 37 Rust tests.

Everything below is committed and green. What I want from review is **not** "does
it run" but "is the reasoning sound and are the assumptions defensible", because
several of these decisions steer a paid cloud run.

Read §1 and §2 first if time is short. Those are the two places where a wrong
assumption produces a plausible number rather than a failure.

---

## 1. The endgame cost model replaces the card cap ⚠ HIGHEST VALUE TO REVIEW

`endgame_cost_model.json`, `validate_cost_trigger.py`,
`seven_wonders_rust/src/cost_model.rs`, wired at `self_play::solver_wants`.

The solver's trigger was `cards_left <= max_cards`. It is now a fitted linear
model of `log10(nodes)` over 20 cheap position features; a solve is attempted iff
`predict + margin <= log10(budget)`. On held-out strong-play positions this
attempts 20% of 11-card positions and skips 4% of 8-card ones, buying the same
805 proofs as `cap <= 10` for 44% of the nodes.

**Assumptions I would most like challenged:**

1. **The margin and the node budget are the same knob at the trigger.**
   `affordable()` tests `predict + margin <= log10(budget)`, so only the
   difference matters. I concluded the sweep should vary the margin at a fixed
   budget, and that the budget's *separate* role is deciding when an in-flight
   solve is abandoned. If that reasoning is wrong the §C sweep grid is wrong.

2. **The shipped margin (0.4) is anchored to a legacy cap.** I chose it because
   it matches `max_cards 10`'s proof count at 44% of its nodes. That is a
   reference, not an objective. I flagged this in the plan rather than hiding it,
   but a reviewer may think a default chosen this way should not ship at all.

3. **The fit underpredicts ~half the expensive positions** (51% of censored
   floors, best model). I argue the margin should therefore come from the
   residual p90 (~0.8 decades, 6.3×) rather than the median, and shipped 0.4
   anyway because it was measured to work. Those two statements are in tension
   and a reviewer should decide whether 0.4 is too aggressive.

4. **Two features were dropped because they measured equal**, not because they
   looked unimportant: `mil_win_feasible` / `sci_win_feasible` need Python's
   `_Derived` reachability sets, and dropping them gives R² 0.939 either way and
   the same 805 solves. The measurement was on one corpus (cloud6 iterations
   37–43). If those features matter on a differently-trained net, the Rust port
   cannot express them at all.

5. **Rust/Python feature parity is tested on bot-game positions** (100+ Age III
   positions) while the model is **fit on cloud positions**. Feature definitions
   are position-local so I believe parity transfers, but the test corpus and the
   fit corpus are different distributions and I did not verify parity on the fit
   corpus itself.

**Tricky bits worth a second pair of eyes:**

- Weights are applied **positionally** in Rust. A feature reordering never
  raises — it silently prices every position with the wrong weights. Guarded by
  passing the names alongside and refusing a mismatch
  (`set_endgame_cost_model`), plus `test_cost_model_parity.py`. Is that enough?
- `9 - military` in `cost_model.rs` assumes `|conflict_position| <= 9`. At
  exactly 9 the game is over, so in `PlayAge` it should be ≤ 8 and the term
  non-negative. I did not add an assertion.
- `science_threat` is `f64::from(science_max >= 5)`, mirroring Python's
  `int(... >= 5)`. Threshold is arbitrary (one symbol from the six-symbol win).

---

## 2. The frozen proven-endgame benchmark ⚠ SECOND HIGHEST

`proven_benchmark.py`, `testdata/proven_endgames.jsonl` (1,000 positions,
500 KB), `test_proven_benchmark.py`.

Ground truth for the value head: exact solver proofs, scored at one forward pass
per position, paired across arms. SE 0.0158 on mean |error| against 0.026 at the
old 133 positions.

**Assumptions to challenge:**

1. **`unrevealed == 0` ⟺ `regime == exact`.** Measured 26/26 and 0/51 over 77
   solves, and used as a pre-filter to make the build affordable. I believe a
   Great Library draw is a chance edge with no face-down card, which would break
   the "⟸" direction — but correctness does not depend on it, because the regime
   is still checked before banking and a non-exact answer is discarded. So this
   is an efficiency assumption only. **Please confirm I have that right**; if the
   filter can also skip bankable positions, the benchmark is biased toward a
   subset I did not intend.

2. **Sign convention.** The solver's `root_value` is actor-relative; the value
   head's W/D/L is actor-relative; I compare `wdl[0] - wdl[2]` against the proof
   directly. If either convention is seat-relative rather than actor-relative,
   every number in the benchmark is wrong in a way that looks plausible.

3. **The benchmark is narrower than §E implied**, and this is forced rather than
   chosen: proofs require full revelation, full revelation only happens late, so
   every position has **1–5 cards left and nothing face down**. It measures
   "can the prior see a forced result three or four plies out on a deterministic
   board". A reviewer may reasonably say that is too narrow to carry the weight
   §E puts on it.

4. **`_instant_win_threat`** uses `|conflict_position| >= 8` or `>= 5` distinct
   science symbols. Both thresholds are mine. The bucket exists so mean/max
   pooling can be tested against its existential claim; if the thresholds are
   wrong the bucket tests nothing.

---

## 3. Terminal forced children (Rust) — a real crash, fixed

`tree_resumable.rs::settle_terminal_forced`. This killed the overnight run 600
games in with `ValueError: both buffer length (0) and count (-1) must not be 0`
— `torch.frombuffer` on an empty `legal_actions`, a batch of 5 rows every one of
which was terminal.

Cause: a card play that completes a military or scientific victory returns
before `finish_turn`, which is what deals the next age. So an action whose chance
spec says "this deals Age N+1" can end the game instead, and every sampled deal
outcome collapses onto the same finished position. Those nodes have no legal
actions and no priors, and the forced path sent them to the network. The wave
path has always routed such leaves to `drain_immediate_wave`.

**Please check the fix's arithmetic**: a settled terminal child is given
`visits = 1`, `value_sum_p0 = terminal_value_p0(state)`, and
`cached_evaluation = Some((value, vec![]))`. My claim is that this is *exactly*
what an evaluated child would have contributed to the edge's
probability-weighted Q, and that the empty prior vector is safe because
`finalize_forced` only checks presence. If a settled child should contribute
something else, the forced-root Q is now subtly wrong rather than crashing —
which is worse.

I also note I had **seen this error once before** (in a `-k` subset run), decided
it was test-ordering because it passed in isolation, and moved on. It was not.

---

## 4. Architecture switches, six sites

`--pooled-readout` / `--reply-head` change which parameters exist. Adding them
left six rebuild sites constructing a model the weights no longer fit. Fixed in
two halves:

- **Writing:** `make_checkpoint` now derives the switches from the model and
  **raises** if a caller's config contradicts it. This is a behavioural change to
  a function every checkpoint goes through — worth confirming the raise cannot
  fire spuriously on a legitimate caller.
- **Reading:** `train.model_from_config` is the one place that knows which config
  keys are architecture. `phase_d` and `build_equiv_corpus` use it.

`test_checkpoint_rebuild_sites.py` enumerates the remaining seven offline probes
rather than converting them, and fails if a converted module is left on the list.
**Judgement call for review:** is enumerating that debt acceptable, or should all
seven have been converted? All are offline tools, none on the training or gate
path.

The detector is a coarse heuristic plus an allow-list. It also names five further
modules that pass a head count *onward* into a config dict, which a regex cannot
follow and which the test does not cover.

---

## 5. Engine equivalence was unverified for weeks

`test_buffer_games_equivalent` and `test_encode_corpus_equivalent` were red and
deselected from every suite run. Two causes:

1. The committed corpus predated the military off-by-one fix. Regenerated.
2. **The gate prefers the large gitignored buffers under `runs/phase_d_toy`**,
   which also predate the fix. So the fresh corpus was ignored and the gate
   stayed red with `illegal action: CONSTRUCT_WONDER`, which reads as an engine
   bug rather than a stale fixture.

The selector now **replays one record** and falls back to the committed corpus if
it does not reconstruct. **Assumption to challenge:** one record is treated as
representative of the directory. A directory of mixed provenance could pass the
probe and still contain stale games. I judged the extra cost of checking more not
worth it; a reviewer may disagree, and the failure mode is silent.

---

## 6. The solver's wall clock was binding (measurement, not code)

The 12-iteration shakedown declined 3,146 of 27,787 solves. **None were
node-capped** — `nodes_max` peaked at 3.71M against a 4.50M cap. Every decline
was `--endgame-solver-max-secs 3`, and the decline rate tracks generation
throughput at **r = −0.817**. Which positions got a proof depended on machine
load.

This is exactly what §7 of the plan warns about, and the plan's own launch
command was violating it. Raised to 30. Second-order consequence recorded: the
shakedown corpus cannot answer "what fraction is solvable within 4.5M" at all
(bounds 44%–100% at cards 10), because the clock censored below the budget.

---

## 7. Censored regression (`fit_censored`)

Tobit EM: censored rows are imputed at `max(prediction, floor)` and refit to a
fixed point. **It under-corrects by construction** — that imputation understates
a right-censored normal's conditional mean. On a synthetic truth with slope 0.5
it recovers 0.44 where survivors-only gives 0.42. Documented in the docstring as
"do not quote these as unbiased estimates".

**For review:** is a partial correction acceptable here, given the trigger
applies a safety margin on top? The alternative is a full Tobit likelihood with a
variance parameter.

A test of this passed **for the wrong reason** at first: it passed the noise's
index in as a feature, so the model explained the noise exactly and neither fit
was biased. Fixed; worth checking the replacement actually bites.

---

## 8. Smaller items

- **`study_rows` floored censored rows at 0** — turning the most expensive
  positions in the corpus into the cheapest, so every affordability test counted
  them as trivially affordable. Now floored at the study budget. Caught by
  arithmetic that did not add up, not by a test.
- **`best_cap`'s objective was degenerate** (solves per node picks cap 0, since
  attempting only free positions maximises it). Rules are now compared at matched
  solve counts.
- **`--from-buffer`** prices endgame positions from existing buffers. Tolerating
  the cloud buffers' final-digest drift is **opt-in**, because for a freshly
  generated record the same mismatch means the engine disagrees with itself. Two
  tests pin that a tampered final digest is accepted only under the flag and a
  tampered mid-game mask is refused even with it.
- **Cloud buffer salvage:** 57–68% of cloud6 games replay with every mask and
  actor matching; the rest diverge at a median 65% through the game and are
  dropped whole. Positions from them are strong-play positions **re-derived under
  corrected rules**, not bit-copies of what the cloud net saw.
- **`cheap_full_agreement.py`** (§B2 probe) — running now, no result yet. Note
  two methodological choices: cheap and full searches use **adjacent seeds**
  rather than the same seed, and the full/cheap schedule is an independent
  Bernoulli per move. An earlier version played the cheap answer at every move,
  making the whole game all-cheap; fixed before the run that matters.

---

## What I have NOT verified

- The cost model on a **strong** net's endgames after the retrain. It is fit on
  cloud6, which stalled. Refit recipe is documented.
- The §B2 probe result — in flight, and it runs on the weak 128×4 shakedown net,
  so its disagreement rates are probably an upper bound.
- `exact_fallback_positions` sidecar (§B) — not built.
- The §C sweep — needs the box.
- Any claim about **strength**. Everything here measures proxies.
