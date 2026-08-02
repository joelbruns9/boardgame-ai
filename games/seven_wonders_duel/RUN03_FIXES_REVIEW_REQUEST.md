# Review request — the run 03 fixes (gate resolution, learner arrest, anchor, HOF)

**Please review for logical and assumption errors, not style or bugs.** The
tests pass and the code does what it says; what needs a second opinion is
whether the *reasoning* holds — whether each fix addresses the cause it claims
to, whether the evidence supports the strength of the claim, and whether any
change quietly removes a safeguard that existed for a reason I did not find.

Commits: `7a2d769` (seven fixes), `98494c1` (HOF-on-resume + docs), plus an
uncommitted correction to `games/az_loop/stagnation.py`.

## 0. Where I am least confident

Ordered by how much a wrong call would cost. If review time is limited, spend it
here.

| # | Claim | Why it might be wrong |
|---|---|---|
| 1 | The learner had **materially regressed**, and the gate was too coarse to resolve most smaller changes | The regression is measured; its cause is not. A wrong causal story would still misdirect the interventions below. |
| 2 | `--probation-reset-after 4` is a sound trigger | 4 is a guess. It discards all unpromoted progress on inconclusive evidence. |
| 3 | A recorded HOF regime boundary may be accepted explicitly | Metrics across it remain incomparable and must be segmented. |
| 4 | Retaining mixed-policy league value labels may help | This is an experiment, not a correctness claim. |
| 5 | Anchor slope is telemetry only | At fixed lag it measures acceleration, not continued learning. |
| 6 | Removing manifest rewrites empirically stopped the RSS creep | The mechanism remains unresolved. |

## 1. The founding diagnosis (§0 #1)

Run `laptop_training_03_w7`, iterations 90–135:

| signal | 90 | 135 |
|---|---|---|
| train loss | 1.834 | **1.771** (falls) |
| val loss | 1.929 | **2.170** (rises) |
| train/val gap | +0.095 | +0.399 |
| gate vs `current_best` | 0.485 | **0.335** |
| gradient norm | 2.82 | 3.29 |

The gate is the load-bearing number: at iteration 135 the candidate lost to the
net it had started from, 0.335. `revert_reset` restored the iteration-60 weights
and it promoted 10 iterations later at 0.610.

**Corrected conclusion after review:** the candidate had materially regressed
relative to the protected best. Overfitting to a low-diversity replay regime and
the `soft_gate` feedback loop are hypotheses about the cause, not results; no
ablation separated them.

**What to challenge.** The val set is a salted game-level holdout drawn from the
*same* buffer, so both sides move with the policy. Rising val loss could mean
"the data got harder/noisier" rather than "the model got worse" — I leaned on
the gate as the independent check, but the gate is 200 games. Is 0.335 at n=200
enough to carry this? Second: I claim diversity collapse is the cause, but the
timing argument (curriculum → 0 at iteration 45, trouble from ~60) is
circumstantial and I never tested it. Third: I assumed the feedback loop
(degrading net generates its own data) *compounded* the problem, but the run
recovered while training on exactly that data (iteration 136 onward trained on
the buffer the degraded net had filled), which is evidence against it mattering
much. I did not reconcile that.

## 2. Gate resolution and the ladder

At n=200 with z=1.96, promotion needs score > 0.598 and revert needs ≤ ~0.402.
Everything between is unresolvable. Every real signal in run 03 lived there:
0.565 and 0.580 refused; 0.415/0.435/0.465 waved through as probation.

Iterations 150–209 with the ladder on: **two promotions, at 0.577 (n=1000) and
0.536 (n=1500) — both below the n=200 bar of 0.598**, so both were only possible
because the gate got wider.

**What to challenge.** This is the result I am most confident in, so the useful
question is the opposite one: is a wider gate *buying* anything real, or is it
promoting noise? A 0.536 [0.500, 0.571] promotion at n=1500 clears the LCB by
0.0004. If the true strength difference is ~0.52, a large gate will eventually
promote it — is that desirable? I assumed yes (any real improvement should
promote). An alternative view is that the bar should encode a minimum
*meaningful* improvement, not merely a non-zero one, in which case the ladder is
laundering marginal candidates into `current_best`.

## 3. The probation reset (§0 #2)

Two changes: a `probation` no longer clears `consecutive_reverts`, and a new
`--probation-reset-after N` resets the learner after N consecutive probations.

Evidence: in iterations 150–209 `consecutive_reverts` **never left zero** — not
one gate was ever significantly worse — so the revert-only mechanism would not
have fired at all. The probation counter fired twice, at iterations 165 and 190,
and the run promoted five iterations later **both times**.

**What to challenge.** Three things.

*The threshold.* 4 is a guess. The cost of firing wrongly is discarding every
iteration of progress since the last promotion, and `current_best` only advances
on promotion, so the amount discarded grows the longer the drought — the counter
is most destructive exactly when it is most likely to fire.

*The direction of causation.* "Reset, then promote 5 iterations later" happened
three times (135, 165, 190). I read that as the reset causing the recovery. But
a reset is followed by ~5 iterations of generation from `current_best`, and the
gate 5 iterations later compares a freshly-reset learner against
`current_best` — a comparison that may be favourable for reasons unrelated to
learning, since the candidate has just been trained *from* the opponent. Is
some of this a measurement artifact rather than a real recovery?

*The interaction with §2.* Both changes make promotions more likely. If the
ladder alone explains the two promotions, the probation reset may be doing
nothing except periodically destroying progress. I have no arm that separates
them.

## 4. HOF on a resume (§0 #3)

**Correction after review:** all three allowed fields create a forward regime
boundary, and `hof_start_games` is positional. Recording the boundary does not
make metrics across it comparable; it gives consumers the provenance needed to
segment them. The flag is a narrow explicit override, not a claim that the two
regimes are comparable.

`--allow-hof-change` lets a resume move `hof_opponent_fraction` /
`hof_sampling_mode` / `hof_start_games`, which `schedule_identity` otherwise
refuses.

**Original argument (retracted).** The guard's own docstring says a mid-run change makes the
iterations either side incomparable *"and the run has no way to record that it
happened."* I claim (a) the second clause is fixable — the change is now written
to `schedule_changes` in the manifest with its games clock — and (b) the first
clause does not apply to this field, because the HOF share is a **level** applied
per-iteration rather than a **position** on an annealing curve. Changing
`--curriculum-anneal-games` retroactively moves where the run is on a curve;
changing the HOF share only alters the opponent mix going forward.

**What to challenge.** Is the level/position distinction real, or is it a
rationalisation for the change I wanted? Specific doubts:

- `hof_start_games` **is** a position (it is *when* league play begins), and I
  put it in the allowed set anyway, on the grounds that a start point already in
  the past is harmless. That is a different argument from the level one, and I
  did not separate them.
- Recording a change is not the same as making it comparable. A reader can now
  see the regime boundary, but any metric aggregated across it is still mixing
  two populations, and nothing forces a consumer to respect the boundary.
- The intervention ladder's rung 3 (`raise_hof_fraction=0.30`) already changed
  the HOF share mid-run, *bypassing* this guard entirely, because interventions
  are applied after the guard runs. So the guard was already not airtight, and I
  did not notice this until writing this document. Does that undermine the
  guard's premise, or is it a second bug?

## 5. The HOF training-target filter (§0 #4)

**Correction after following the path into Rust:** the production scheduler
already excluded the archive seat at `self_play.rs::finish_move` whenever the
actor used network 1. `archive_policy_seats` is defense in depth for imported,
legacy, Python-written, or retagged records; it did not change the live Rust
generation path. Its retained value labels remain an experiment because league
trajectories follow a mixed policy.

**Original claim (retracted):** `examples_from_record` emitted a policy target for **both** seats of a league
game, so `--hof-opponent-fraction 0.15` would have trained the learner to
imitate the archive on ~7.5% of positions. `archive_policy_seats` now drops the
policy target on the archive's seat and **keeps its value labels**.

**What to challenge.** I kept value labels by analogy to curriculum-bot moves,
which are `policy_excluded` but whose value labels the code explicitly retains
("bot moves are a curriculum device whose value labels stay useful"). Two
possible errors:

- The analogy may not hold. A *weaker* opponent systematically inflates the
  value of positions where it is to move: the learner would win more often from
  those positions than the recorded outcome suggests. That is a known league
  bias, and I asserted the outcome "stays valid for every position" without
  addressing it.
- The predicate is deliberately narrow — it fires only on `kind == "league"` with
  `league_assignment_used == "true"`. I chose narrow after finding that the
  obvious wide predicate ("any seat not named `network`") would have silently
  deleted the curriculum's policy signal, since curriculum-seed games name a bot
  on both seats and are policy-eligible on purpose. But narrow means any future
  writer that produces archive games with different metadata gets no filtering
  and no warning. Is a fail-open predicate right here?

## 6. The self-anchor, W7c (§0 #5)

W7a indexed the promotion lineage; once `current_best` had been frozen longer
than the lag, the reference resolved to `current_best` itself and the loop
returned a **synthetic 0.500 without playing**. That read 0.500 for 35
consecutive iterations spanning a collapse *and* a recovery. It now indexes
immutable post-transition `learner_NNNN.pt` snapshots, which record the weights
actually left in `latest.pt` after promotion or reset effects. This avoids
treating a rejected pre-reset candidate as learner history.

**What to challenge.**

- W7a's null was 0.500 *by construction* (same file). W7c's is 0.500
  *statistically*. Every threshold in `StagnationDetector` was chosen under the
  first regime and none has been re-derived. I have documented this but not
  fixed it.
- The observed series ended `0.790 → 0.725 → 0.570 → 0.695 → 0.560 → 0.510`,
  but reset-adjacent points used rejected candidates as references. That series
  must be re-derived from post-transition learner snapshots before it supports a
  strength-trajectory or headroom conclusion.
- Reusing the promotion gate's result when the anchor resolves to the same
  matchup: I claim this is exact, not an approximation. Same subject, same
  opponent *weights* (compared by digest, not path), same fixed-N rule. Is there
  a seed or pairing difference that makes the two matches non-identical?

## 7. `slope_epsilon` and the intervention ladder

**Correction after review:** the slope trigger was conceptually invalid and has
been removed from lifecycle decisions. For fixed lag `L`, the anchor measures
`S(t) - S(t-L)`: steady learning produces a constant advantage and therefore a
near-zero slope, while a decelerating but healthy learning curve produces a
negative slope. It measures acceleration, not continued learning. The OLS value
remains telemetry; the Wilson-interval rule is now the only lifecycle trigger.

**Original analysis (retracted):** `StagnationDetector` had two independent triggers, OR'd: **interval** (the last 3
anchor intervals all include 0.500) and **slope** (OLS slope of score vs games ≤
`slope_epsilon = 0.005` per 10k games).

The first live series produced slopes from **+0.31 to −0.55 per 10k games**, and
in run w7 **every** STAGNANT verdict came from the slope trigger; the interval
trigger — which uses Wilson bounds and so accounts for sample size — never fired.

**I initially concluded the threshold was ~2 orders of magnitude too small. That
was wrong, and the correction is the interesting part.** The slope is fitted to
three scores, so its standard error is `SE(score) / (spacing × √2)` — a
*denser* cadence produces a *noisier* slope, because the x-spread shrinks faster
than the count grows. With SE(score) ≈ 0.049 for a 200-game anchor:

| `--self-anchor-every-games` | SE(slope)/10k | fires when truly +0.10/10k | when truly flat |
|---|---|---|---|
| 10,000 (documented default) | 0.035 | 0.3% | 56% |
| 4,000 | 0.087 | 14% | 52% |
| **2,000 (what w7 ran)** | 0.174 | **29%** | 51% |

So the threshold is defensible at the default cadence and near-useless at five
times that density. The defect is an undocumented coupling between two flags,
not a bad constant.

**What to challenge.** Three things. (a) The SE(score) ≈ 0.049 figure is read
off one observed Wilson interval, not derived from the pair-outcome
distribution, and paired seats make the pair variance smaller than a naive
binomial — so the table may be pessimistic. (b) I assume three evenly spaced
measurements; `min_measurements` is 3 but the cadence can change mid-run under
an intervention, which breaks the even spacing the arithmetic assumes. (c) Given
the interval trigger is self-calibrating and the slope trigger needs a
cadence-dependent constant, is the slope trigger worth keeping at all?

**The ladder itself** (four rungs, one at a time, 20k-game window between,
disabled by default): raise search budget → widen replay window → HOF 0.30 → LR
×3. Two observations I could not resolve:

- The last rung is now honestly named `lr_jump` (LR ×3). It retains AdamW
  moments and skips warmup; it is not an optimizer or cosine restart. The old
  `lr_warm_restart` name incorrectly implied otherwise. On the theory that "the optimiser,
  not the data, is stuck". My run-03 analysis found the opposite: after the
  reset, *the same* LR trained cleanly for 14 iterations, so LR was not the
  problem — accumulated weights were. Is the last rung aimed at the wrong thing?
- **Reset-to-`current_best` is not a rung**, yet it is the only intervention with
  direct evidence of working in this codebase (three times: 135, 165, 190). It
  sits in the lifecycle instead. Should the ladder include it, and if so where?

## 8. Manifest and memory (§0 #6)

**Corrected attribution after review:** the rewrite was empirically effective,
but the mechanism is unresolved. `store.iterations()` still creates a measured
~261 MB transient versus the old manifest path's ~320 MB, weakening the
allocator-high-water explanation. The removed path also performed roughly
380 MB of file I/O per iteration; no arm separated allocation, I/O, write
ordering, and row-content effects.

`append_iteration` re-read and re-serialised the whole manifest each iteration,
making it a byte-for-byte duplicate of `training_log.jsonl` at quadratic cost —
188.5 MB, 3.7 s and a ~320 MB transient per iteration by iteration 149. Rows now
live only in the log; the manifest keeps a count. Measured after: 4.4 KB,
12.9 ms, and **137 bytes of growth across 60 iterations**.

Separately, run 03 grew ~8.7 MB/iteration *after* the example cache saturated. I
built an isolated probe that drove the cache's LRU churn with no generation, no
training and no manifest; it showed +0.31 MB/step and a *shrinking* rss−live gap,
clearing the cache. Post-fix, iterations 170–209 of the real run show **+0.70
MiB/iter, non-monotone, ±365 MiB scatter** against the old +8.7 monotone.

**Remaining uncertainty.** The isolated manifest test showed its transient
memory returning cleanly, so no mechanism links that allocation to a monotone
RSS floor. "The creep vanished when we removed the manifest" remains a useful
empirical result, not a causal attribution; the change also altered file I/O,
write ordering, and row content.

## 9. Things a reviewer should also check

- **The reset counters interact with the ladder.** `_ladder_after` zeroes
  `consecutive_probations` on every step-up, which is why
  `probations_since_decisive` is a separate field. Are there other consumers of
  either counter that now see different values?
- **`revert_suppressed` amnesty.** A recorded HOF change becomes a knot and the
  next gate cannot revert. The controller now threads the stop reason into the
  lifecycle: the row still shows probation and `revert_suppressed_knot`, while
  revert, ladder-probation, and reset-probation counters remain unchanged.
- **`_sanitize_non_finite` returns the input object when clean.** Callers must
  not assume a fresh copy. Only `_append_training_log` calls it today.
- **Test-file hazard.** `games.seven_wonders_duel` resolves under two module
  names, so a *relative* import inside a pytest-collected test yields duplicate
  classes and enum identity silently fails. I converted my imports to absolute;
  ~9 pre-existing relative imports remain in `test_phase_d.py`.

## 10. What is not claimed

- That S has headroom at 36k games. Run 03 had regressed, so it says nothing
  about model width, and the question is still open.
- That HOF diversity fixes the overfitting. It is the leading hypothesis and has
  never been run.
- That the gate scheduler settings transfer off the laptop 3070. The sweep is
  hardware-specific; `setup_cloud_7wd.sh` re-runs it on the box.
- That `--probation-reset-after 4` is the right number.
