# Review request: W7 specialist league (S0-S4 + S2b)

**What this is.** `SPECIALIST_LEAGUE_PLAN.md` implemented. The plan is unchanged
and is still the specification; this document says what was built, where the
implementation departs from it and why, what is measured versus assumed, and
where I think a reviewer will find something.

**Reviewed 2026-09-08** (`SPECIALIST_LEAGUE_IMPLEMENTATION_REVIEW.md`): eight
findings, all reproduced, all fixed, plus two additional S2b concerns and two
overclaims in this document. See §9 for the audit trail. Sections 2-8 describe
the code as it now stands.

**Status: built and tested, UNRUN.** S0a (frozen-weights attack coverage), S0b
(the small general-training A/B) and the S5 pilot are *runs*. None has been run,
so nothing here claims the workstream works — only that the mechanism exists, is
correct against its own gates, and is inert until switched on.

**Default behaviour is unchanged.** With `--specialists` empty (the default) no
search carries a bias, no record carries new bytes, every target routes to the
general, and the committed engine/search equivalence gates pass untouched.

---

## 1. What to read, in order

| File | What it holds |
|---|---|
| `seven_wonders_rust/src/eval.rs` | `VictoryClass`, `LeafBias` (the utility formula and the sign convention), `MockOutlookEval` |
| `seven_wonders_rust/src/tree.rs` | Scalar reference searcher: `Backup`, shaping in `descend` / `force_expand_root` / root init, `root_value_unshaped` |
| `seven_wonders_rust/src/tree_resumable.rs` | **The production searcher.** Shaping at `settle_simulation`, `settle_terminal_forced`, the forced branch of `apply_evaluations`, and both root initialisers |
| `seven_wonders_rust/src/self_play.rs` | `SpecialistSpec`, `TargetRoute`, `SelfPlayConfig::{leaf_bias_for, net_is_training, target_route_for}`, the widened `policy_excluded`, the solver policy |
| `search.py` | Python reference: `LeafBias`, `_terminal_outlook_p0`, `_leaf`, shaping at the same four kinds of site |
| `buffer.py` / `dataset.py` | `root_value_unshaped` + provenance on `MoveRecord`; `bootstrap_root_value`, route-aware `has_policy`, `project_examples` |
| `specialist.py` | S1-S4 + S2b selection: shares arithmetic, lineage, archive, floor, step sizing, reanalysis |
| `phase_d.py` | `LeagueAssignment` opponent classes, seeding, `train_specialist`, `specialist_floor_check`, `reanalysis_for_general` |
| `specialist_probe.py` | The S0a instrument |

Tests: `test_specialist_bias.py` (S0, 39 cases), `test_specialist_league.py`
(S1-S4 + S2b + end-to-end, 52), `test_specialist_probe.py` (4).

Size: ~2,100 lines changed across 12 files, ~2,800 new.

---

## 2. S0: the bias

### The formula and the sign trap

`eval.rs::LeafBias::shape`, mirrored in `search.py::LeafBias.shape`:

```
seat 0:  utility_p0 = value_p0 + lambda * outlook_p0[p0_<type>_win]
seat 1:  utility_p0 = value_p0 - lambda * outlook_p0[p1_<type>_win]
```

Own-win by default; `symmetric` behind a flag. The seat is the **searcher's**,
resolved in `make_search_meta` from `net_by_player[actor]`, never from the leaf
actor.

All seven terminal utilities are pinned for both seats and both languages
(`test_all_seven_terminal_utilities_are_pinned_and_agree_across_languages`),
through a new `swr.specialist_utility` binding that exposes the Rust shaping
directly so the convention is testable without inferring it from a whole
search's outputs. `symmetric` is additionally pinned as **seat-independent** in
p0 terms, which is precisely why it is a mirror-player and not an attacker.

### Every path that produces a leaf value

The plan named the sites and warned the first draft would have silently
no-opped. All of them carry the bias:

* **Python search** — `_descend_closed` terminal + expand, `_expand_closed`,
  `_force_expand_root`, root init in `_search_closed` and `make_root`.
* **Rust scalar tree** — `descend` (terminal + expand), `force_expand_root`
  child seeding, root init in `search_closed`.
* **Rust resumable tree (production)** — `settle_simulation` (the single place
  any backup is accounted, so the cached-replay and fresh-evaluation paths
  cannot diverge), `settle_terminal_forced`, the forced branch of
  `apply_evaluations`, and both `begin_search_from_root*` root initialisers.
* **Root accumulation** — `outlook_sum` / `outlook_visits` are untouched, since
  the outlook is never shaped.
* **Solver** — see §2.4.

The invariant that makes this reviewable: **`value_sum_p0` always holds shaped
utility; `cached_evaluation` always holds the raw value plus its outlook.** Every
site that moves a raw value into a sum shapes it exactly once.

### `root_value_unshaped` is accumulated, not reconstructed

A second `(sum, visits)` pair runs over the same simulations, seeded with the
root's own unshaped expansion. The plan is explicit that subtracting
`lambda * root_outlook` is exact only if the value sum and the outlook sum share
a visit set, and nothing guarantees that — a leaf with no outlook contributes to
one and not the other.

`net_root_value` was also split out: it is documented as "the network's RAW
evaluation of the root", and before this change `SearchSession` used one field
for both that and the q_hat fallback. Under a bias those are different numbers.
`net_root_value` stays raw; the q_hat fallback is the shaped utility, because it
must live on the same scale as every visited action's Q.

### A missing outlook is a hard error

`LeafBias::shape` refuses a `None` outlook whenever `lambda > 0`, in both
languages. A treatment that reaches some leaves and not others measures nothing.

Two consequences, both implemented:

* `PhaseDConfig.validate` refuses `--specialists` without `--hierarchical-value`
  — the right error at launch rather than minutes into generation.
* `specialist_probe` refuses a checkpoint with no `hier_value` head, because a
  null that only means "the head is missing" is worse than no probe.

### The solver policy (a decision the plan left open)

**A biased searcher does not take the endgame-solver shortcut.** Gated on the
bias in both `run` and the scheduler's `finish_move`, so it applies to the async
pool as well as the synchronous path.

Reason: the endgame mask is a proof of *unbiased* optimality. Applied to a shaped
search it would delete exactly the attacking continuations the bias funded
whenever they are provably a shade worse — silently turning the specialist back
into the general at every solved endgame. Specialists are sparring equipment;
losing their endgame proofs costs nothing the experiment needs. Keyed on the
bias, not the seat or the net id, so an archive on the same seat still solves
(`test_a_biased_searcher_does_not_take_the_solver_shortcut` asserts both halves).

I checked the plan's other worry first: the solver's value never enters the tree.
`endgame_overlay` returns a value and a keep-set; the value is recorded on the
move and the keep-set masks the policy target. So "solver values entering the
tree" is not a live path, and the mask is the whole interaction.

### Acceptance

| Plan requirement | Where |
|---|---|
| `lambda = 0` bit-identical, verified on the resumable path | `test_lambda_zero_is_bit_identical_to_the_unbiased_searcher` (both engines, force on and off), plus the committed corpus gates in `test_rust_engine_equiv.py`, unchanged and passing |
| `lambda > 0` Python/Rust move-for-move on a fresh biased corpus | `test_biased_search_matches_python_move_for_move` — 5 bias settings × 2 engines × 4 positions × 2 budgets × 2 seeds × force on/off; action, visits, top-k, action value, root value, policy target and `root_value_unshaped` |
| Cached and forced evaluations carry the same bias as fresh ones | `test_forced_and_cached_leaves_carry_the_same_bias_as_fresh_ones` — the scalar oracle has no cache and the resumable searcher does, so their agreement under `force=True` *is* the cache test |
| The bias must do something | `test_a_live_lambda_changes_the_search_and_widens_the_utility_scale`, and end-to-end `test_a_live_lambda_changes_the_games_that_are_generated` |

The λ>0 gate needed an oracle that supplies an outlook, since `MockEval`
correctly supplies none. `MockOutlookEval` folds a seven-way distribution from
the same fingerprint hash, normalised by an explicit left fold so Python and Rust
agree to the last bit (`test_mock_outlook_matches_python_bit_for_bit`), and keeps
the exact outlook at terminals.

### The widened utility scale is **not** compensated

Utility becomes `[-1-lambda, 1+lambda]`. The Gumbel root is unaffected —
`sigma_vector` min-max rescales completed Q — but PUCT's
`Q + c_puct * P * sqrt(N)/(1+n)` is not scale invariant, so a nonzero lambda
makes exploration relatively cheaper, and self-play runs PUCT on recorded moves.
Documented in three places and deliberately left alone: auto-scaling `c_puct`
would fold two changes into one flag and make the S0b A/B uninterpretable.
**This is a real behavioural side effect a reviewer should weigh.**

---

## 3. S1/S2: routing

### `policy_excluded` was widened, not replaced

`!full || net_by_player[actor] != 0` became `!full || !net_is_training(actor)`.
An archive has no `SpecialistSpec`, so it is unchanged; a learning specialist
keeps its targets.

The plan demanded an audit of every consumer before touching this field. What I
found and what it means:

| Consumer | Effect |
|---|---|
| `is_fast_search_move` (`policy_excluded and sims > 0`) | Previously classified an *archive's full move* as a fast search and dropped it from examples entirely — so archives contributed no value labels either, contradicting `archive_policy_seats`' docstring. Unchanged for archives. A specialist's full moves now correctly become examples. **This asymmetry between the code and the comment predates W7 and is worth a look on its own.** |
| `reply_targets` (skips when the next move is excluded) | A specialist's full move now supplies a reply label. Desirable — the reply head learns the opponent's actual reply — but it is a change, live only when a specialist is configured. |
| `archive_policy_seats` | Now returns empty when the opponent carries a specialist route. Without this the specialist's own labels would be deleted at the example boundary — the same failure this predicate's deliberate narrowness protects the curriculum bots from. |
| `target_version_for_moves` | Specialist rows now count. Same target definition, no version change. |
| Old buffers | Carry no route and default to `"general"`, which is what they were. |

### Routes are additive, not a replacement

`MoveRecord.target_route` is a fourth fact (`general` / `specialist:<id>` /
`none`) beside search quality, which network searched, and eligibility.
`Example.has_policy` is `eligibility AND route == derived_for`.

**Curriculum-bot moves route to `general`, not `none`.** The plan's parenthetical
groups them with archived HOF, but `buffer.archive_policy_seats` documents at
length that imitating those bots is the entire point of the curriculum and that
widening the exclusion "would delete the curriculum's policy signal while every
test still passed". I followed the code's documented invariant. Flagging it
because it is a deliberate departure.

### The value-leak contract

`dataset.bootstrap_root_value(move, derived_for) -> (value, shaped)`:

* unbiased search → `root_value`, anyone may use it;
* biased search, own model → `root_value`, flagged shaped;
* biased search, any other model → `root_value_unshaped`, or **nothing** when the
  record does not carry one.

Per move, not per buffer — the general is *meant* to learn value from the
specialist's positions, so those rows reach its buffer by design.

The isolation assertion the plan asks for runs **inside `train_candidate`**, on
every derived buffer, before anything trains
(`specialist.assert_no_shaped_bootstrap`). It reads a recorded flag
(`Example.root_value_shaped`) rather than re-deriving the rule, so a refactor of
`bootstrap_root_value` cannot make it pass silently. End-to-end coverage over
real generated records is `test_the_general_never_bootstraps_from_a_shaped_root_end_to_end`.

### Exploration: "any net that is training"

Root noise and forced playouts were gated on `net_by_player == 0` because network
1 was always frozen. The predicate is now `net_is_training`, which coincides with
the old one whenever no specialist is configured. Tested by making seat 0 a
scripted bot so seat 1's exploration is observable in isolation: an archive
produces byte-identical games at `eps = 0` and `eps = 0.25`, a specialist does
not.

### One derivation per record, projected per model

The obvious implementation keys the example cache on the route. That triples a
real run's replay derivation and its cache footprint at two specialists, for rows
that differ in four scalars. Instead the cache stores one entry per record —
always derived for the general — and `dataset.project_examples` re-labels it.
`test_projection_equals_a_full_derivation_for_the_same_model` asserts the cheap
path and the expensive path agree field by field.

---

## 4. S3/S4: training, archive, floor

* **Step sizing** (`specialist.steps_for_inflow`) scales each model's steps by
  its own **measured** policy inflow, reported per route in
  `last_training_stats["policy_inflow"]`. `train_every` banks iterations and
  scales the count back up; both levers exist because which is right depends on a
  number only a run can measure.
* **The frozen general anchor** is pinned once, when the first specialist is
  seeded, and never rebuilt (`freeze_general_anchor`, idempotent and tested).
* **Per-type archive** from the first accepted specialist. Its first entry is the
  frozen attacker, and `frozen_reference()` is asserted stable across later
  accepts. Generation draws the live specialist ~2/3 of the time and its archive
  otherwise.
* **No promotion gate.** A specialist advances on every train step. The collapse
  floor — score rate against the frozen anchor, checked every
  `--specialist-floor-every` iterations — is the only thing that rolls it back,
  and it reverts to `last_good.pt` and logs loudly.
* **Lifecycles are separate.** Specialists live under `run_dir/specialists/<class>/`
  with their own optimizer state, update counts and journal;
  `run_specialist_iteration` runs after the general's train step and **before**
  its gate, so a rejected general candidate cannot touch a specialist and a
  collapsed specialist cannot touch the general.

---

## 5. S2b: reanalysis

Off by default (`--specialist-reanalysis`), because it is an arm of the S5 pilot.

* **Selection** (`reanalysis_candidates`): positions where the bias moved the
  search's own valuation by at least `reanalysis_gap`, plus every full-budget
  specialist move in a game the specialist won by its own type.
* **Cap** (`cap_reanalysis`): a fraction of the general's policy inflow, trimmed
  round-robin so one long game cannot take the whole allowance. Share, positions
  and seconds are reported.
* **Evaluator** is the general's own current net, at `lambda = 0`.
* **Clairvoyance**: re-search runs from the recorded pre-move state, which holds
  no hidden card identities — reveals are chance events the search samples for
  itself — so it sees exactly what the player saw.

### ⚠ The one place I could not implement the plan as written

The plan says: *"the biased search already knows where lambda mattered: it holds
both the shaped and the unshaped root utility, so 'lambda changed the chosen
move' is a cheap local flag."*

It does not follow, and I could not make it follow. Holding both root utilities
says how much the bias moved the *valuation*; it does not say whether the
*argmax* changed. The played action and the visit distribution are both the
biased search's, and a lambda-zero argmax over them cannot be reconstructed
without re-searching — which is the expensive thing the selection exists to
avoid.

So `DEFAULT_REANALYSIS_GAP` uses the **size of the gap** as a proxy, and says so
in its own docstring rather than presenting itself as the flag the plan asked
for. It is strictly weaker: it will select positions where the bias moved the
value but not the move, and it will miss positions where a small value shift
flipped a near-tie.

**There may be no cheap real flag.** My first suggestion was to have the biased
search record a lambda-zero argmax over its own completed Q. The 2026-09-08
review is right that this would not be a counterfactual either: the sampled tree
and the completed-Q vector are both products of the biased search, so it would
measure a re-ranking of shaped statistics, not what an unbiased search would
have chosen. With separate per-action RAW statistics it could honestly be called
a ranking change on the same sampled tree, which is a weaker but well-defined
claim. A true counterfactual needs a second search, which is the cost this
selection exists to avoid.

---

## 6. S0a: the probe

`specialist_probe.py` searches the same positions with and without the bias,
same seeds, same evaluator, and reports the columns a null has to be split into:
`moved_fraction`, `credible_fraction_of_moved` (attacks a lambda-zero search
still rates within `--credible-gap` of its own best), `mean_q_cost_of_moved`,
`mean_prior_of_new_choice` and `mean_visit_share_of_new_choice` (the search never
funded it, versus the prior never offered it), and `own_type_outlook` biased
against unbiased.

`search_many_flat_net` gained the bias parameters so the probe can run through
the production position-search boundary, and `SearchResult`'s root prior is now
serialised (it was computed and dropped).

**One observation from a smoke run, on an untrained net** — reported because it
is what the instrument is for, not as a result: at `lambda = 1.0`,
`root_value_shift` came out at +0.0376 against a mean own-type outlook of 0.0367,
i.e. exactly `lambda x outlook` as the arithmetic requires, while
`moved_fraction` was 0.0. An untrained net's outlook barely varies across moves,
so a nearly constant additive bonus does not move an argmax. That is the expected
shape of an S0a null on an untrained net and it validates the instrument; it says
nothing about a trained one.

---

## 7. What is measured, what is assumed, what is not done

**Measured:** per-route policy inflow; reanalysis share, positions and seconds;
shaped share of a buffer (`specialist.shaped_share`); specialist score against
the frozen anchor; games, learner score rate and victory-type mix **split by
opponent class** in `summarize_records["opponent_classes"]` (never one
aggregate, and pure self-play games contribute to no class rather than pulling
every class toward 0.5); `moved` / `credible` / `q_cost` / prior / visit-share /
own-type outlook in the probe.

**Assumed, and worth attacking:**

* The default `lambda` values. There is no evidence for 0.5; the plan says start
  with one per type and add a population after S0 shows the effect is real.
* `DEFAULT_REANALYSIS_GAP = 0.05` and `collapse_floor = 0.15` are guesses.
* The 2/3 live / 1/3 archive split for which specialist checkpoint plays.
* That a specialist fine-tuned from a promoted general at `hof_start_games`
  is strong enough to teach anything.

**Not done, deliberately:**

* S0a, S0b and the S5 pilot — runs, and the user launches long runs.
* The lambda population on a Pareto frontier (plan: after S0).
* Anything that lifts the `{0, 1}` net-id cap. Rotation is the first choice, as
  the plan says.
* The measurement suite over the threat corpus (frozen attackers, frozen general
  anchor, fixed position suite). The anchor and the frozen attackers are built;
  the *reporting* against the threat corpus is not, and it is what makes the
  general's improvement — the actual success criterion — legible.

---

## 8. Where I would attack this first

1. **The S2b selection proxy** (§5). Still the one place the plan asked for
   something I did not deliver, and there may be no cheap substitute that is
   more than a labelled heuristic.
2. **The uncompensated PUCT scale change** (§2.5). It is a second, unlogged
   treatment riding along with every nonzero lambda, and S0b is a training A/B.
3. **The `is_fast_search_move` / `archive_policy_seats` contradiction** (§3.1).
   Pre-existing, but W7 walks straight through it; the 2026-09-08 review agreed
   it is pre-existing rather than new.
4. **`policy_excluded` widening.** I audited five consumers; if there is a sixth
   I did not find, it will fail quietly rather than loudly.
5. **The curriculum-bot route decision** (§3.2) — a deliberate departure from the
   plan's parenthetical, which the review found consistent with preserving the
   curriculum signal.
6. **Specialist checkpoint sizing.** A specialist is a full copy of the general's
   architecture, per class, plus its optimizer state, a `last_good` copy, two
   optimizer snapshots and a growing archive. At two classes that is a
   meaningful addition to a run's disk and its host-memory preflight, and I have
   still not sized it.
7. **The lifecycle paths generally.** Five of the 2026-09-08 review's eight
   findings were there rather than in the search, and they are the paths a green
   suite reaches least.

---

## 9. Response to the 2026-09-08 implementation review

`SPECIALIST_LEAGUE_IMPLEMENTATION_REVIEW.md` raised eight findings. **All eight
reproduced, all eight are fixed**, along with the two additional S2b concerns
and two overclaims in this document. Nothing was disputed. What follows is what
changed and, where the fix differs from the one suggested, why.

Sections 2-8 above describe the code as it now stands; this section is the
audit trail.

### R1 — rollback left the rejected specialist in play *(fixed)*

Two real defects behind one symptom. `_specialist_opponent` read the newest
ARCHIVE entry for its "live" branch, and `accept` archives every candidate, so
after a revert the rejected weights were still what generation played — and were
still reachable by archive sampling. The rejected optimizer state also survived,
so the next step would have re-applied the rejected update's momentum.

* `SpecialistLineage.live_entry()` resolves the live branch from `latest.pt`,
  not from the archive.
* `LineageState.accepted_since_good` tracks entries admitted since the last
  clear measurement; `revert` moves them to `quarantined`, and
  `sample_archive()` skips those.
* `mark_good` snapshots `optimizer.pt` to `optimizer_last_good.pt`; `revert`
  restores it, or deletes the optimizer state if there is no snapshot (absence
  is a recoverable cold start, the rejected moments are not).

Three tests, asserting the next opponent and the next optimizer state rather
than the bytes of `latest.pt` — which is what the review asked for and what the
original test got wrong.

### R2 — specialists trained their WDL head on shaped utility *(fixed)*

**The strongest finding, and the plan itself is wrong here.** The plan says
`root_value` "is consumed only by the model whose lambda produced it", which is
right for the search's arithmetic and wrong for `value_soft`: that target trains
a W/D/L PROBABILITY head, and that head is what `_evaluate` reads back as
`wdl[0] - wdl[2]`. The specialist would learn the bonus as win probability and
its search would then add the bonus again. Worse, `root_value_unshaped` is
computed *from those same outputs*, so the distortion would have reached the
general through the one channel the quarantine exists to close.

`bootstrap_root_value(move)` now returns the **unshaped** root for every model
including the owner, and lost its `derived_for` parameter — there is nothing
left for it to decide. `assert_no_shaped_bootstrap` was strengthened from "no
shaped root in the wrong buffer" to "no shaped root in ANY value target". The
shaped root stays recorded, for auditing and for S2b selection, and trains
nothing.

The review's alternative — a separate utility head — is the right shape if
learned utility is ever wanted. It is not built, and this document no longer
implies the current arrangement is one.

### R3 — idle iterations invented training inflow *(fixed)*

`steps_for_inflow` multiplied the newest iteration's row count by
`iterations_since_train + 1`, and that counter incremented whether or not the
iteration supplied any rows. With one opponent class drawn per iteration, an
iteration supplying a given specialist nothing is the *ordinary* case, not an
edge one. Inflows of 0, 0, 100 earned 75 steps where 100 rows warrant 25.

`LineageState.banked_inflow` now persists unconsumed rows; `bank_inflow` adds to
it each iteration and `accept` zeroes it. `steps_for_inflow` lost the
`accumulated_iterations` parameter entirely: banking N iterations of rows scales
the step count on its own, because the rule is samples-per-new-position matching
and the row count is the only input it needs. `iterations_since_train` survives
as what it always should have been — the `train_every` cadence, and nothing else.

### R4 — a retried iteration trained specialists twice *(fixed)*

Specialist updates commit inside `adapter.train`, before the controller commits
the iteration, and the controller's rollback restores general artifacts only.

`LineageState.last_trained_iteration` is the completion identity;
`train_specialist` returns `skipped: "already trained this iteration"` when it
matches. This is the review's "persist a completion identity" option rather than
its "journal and restore" one, because a retry re-runs generation from the same
seeds and therefore produces the same records — the rows have genuinely already
been spent, and the correct action is to not spend them again. A normal general
promotion rejection still does not touch a committed specialist update.

### R5 — the collapse floor tested an unbiased player *(fixed)*

The floor built a plain `ModelAgentSpec` and called the generic model match,
which carries no lambda. It measured the specialist's *weights* with the bonus
switched off — a different player from the one that generates the data, and one
whose competence says nothing about whether the biased decisions are sound.

* `self_play_many_flat_net` gained `specialist_net` (Rust), because in a gate
  the subject is network 0 while in generation the specialist is network 1.
* `_wilson_model_match` and `_rust_model_gate_rolling` take an optional
  `specialist` and put the bias on network 0. Exploration noise stays off.
* A biased floor match on the **Python** gate backend is now refused outright
  rather than run: that path has no leaf bias and would silently measure the
  wrong player.
* The `specialist` keyword is passed to `_rust_model_gate_rolling` **only when
  there is one**, so an ordinary gate's call is byte-identical to its pre-W7
  call. The first version passed it always and broke the precision arena's spy
  — a fair catch by the full suite, and the same additive-and-inert discipline
  the rest of this build follows.
* The result row records `biased: True` plus the lambda and victory class, so a
  report cannot be misread as a measurement of the deployed opponent.

Seeding also now refuses a `current_best` whose *weights* lack the outlook head,
not just a config missing the flag — a resumed run predating
`--hierarchical-value` would otherwise have seeded a specialist that dies at its
first biased leaf.

### R6 — S0a used the wrong selection rule *(fixed)*

The probe ran a Gumbel root (the default) but defined both moves by max visits,
which is neither Gumbel's returned action nor its deterministic policy choice.

* `--root-selection` is explicit and **defaults to `puct`**, which is what
  self-play runs on the moves it records — so the probe measures the decision
  the training data would carry.
* Both moves now come from the search's own `result["action"]`, so the metric
  cannot drift from the searched player's decision if the root rule changes.
* A PUCT root forces `leaf_batch = 1`, because a wide wave selects the root
  under virtual loss and the root's distribution is the thing being measured.
* Credibility is claimed only where the lambda-zero search actually **visited**
  the biased choice. `completed_q` falls back to the root value for an unvisited
  candidate, and counting that as an endorsement reports the absence of an
  opinion as verification. `unverified_fraction_of_moved` is reported beside it,
  and a large value is itself an S0a finding.

### R7 — disabling specialists changed HOF sampling *(fixed)*

`draw_opponent_class` consumed a random value even when HOF was the only
possible class, shifting the stream `hof.sample` then reads. Existing
HOF-enabled runs would have selected different archived opponents at the same
seed and iteration — a silent change to a running experiment, in exactly the
configuration W7 claims to leave alone. **This was the most embarrassing
finding**: this document asserted disabled-mode neutrality, and the tests only
covered pure self-play.

The draw is skipped when one class is possible. A regression test replays the
pre-W7 RNG path and asserts the same archive is chosen at twenty iterations.

### R8 — reanalysis caps changed on resume *(fixed)*

`reanalysis_for_general` read `last_training_stats` before the current training
call updates it — the previous iteration's number — and on a fresh process fell
back to counting every move in the replay window. `train_candidate` now measures
this iteration's general policy inflow from the examples already in hand and
passes it explicitly. `cap_reanalysis` also enforces a finite count allowance
across the whole cap range; `cap >= 1.0` and a zero inflow both used to
short-circuit to "everything", so the two configurations that most needed a
bound had none.

### Additional S2b concerns *(both addressed)*

* **Stale teacher.** `reanalysis_for_general` loaded `current_best` even when
  `train_candidate` was continuing a newer `source_checkpoint`. It now takes the
  teacher explicitly, defaulting to the checkpoint being continued, and records
  which one it used.
* **Coverage for the specialist's candidates.** Not a mechanism, a
  **measurement**: the reanalysis reports
  `specialist_move_visited_fraction` — how often the unbiased re-search funded
  the move the specialist actually played. At 7WD's median branching of 4
  against a top-k of 16 the candidate set should already contain it, and this
  says whether that holds before a mechanism is built for a problem that may not
  exist.

### Corrections to this document

* **The Gumbel claim was too broad**, as the review says. Sigma min-max rescales
  completed Q at the *root*, so the root's halving is scale free, but every
  interior node still selects by PUCT on the raw utility. Corrected in
  `eval.rs`, `search.py` and `training_parameters.md`; §2.5 above stands
  otherwise.
* **My proposed fix for the S2b selection proxy was not the improvement I
  claimed.** I suggested recording a lambda-zero argmax over the biased search's
  own completed Q. The review is right that this is not a counterfactual either:
  the sampled tree and the completed-Q vector are both products of the biased
  search. With separate per-action *raw* statistics it could honestly be called
  a ranking change on the same sampled tree; a true counterfactual still needs a
  second search. The proxy stays, labelled, and §5 no longer offers a fix that
  would not have been one.

### What the review changes about readiness

Five P1 defects sat in lifecycle and training-objective code that the green
suite did not reach — rollback, budgets, retry, the floor's subject, and what
the value head is actually taught. That is a fair characterisation of where the
coverage was thin: the search had gates and the *loop around it* had assertions
about bytes and counters rather than about the next opponent, the next update,
and the next target. The new regression tests are written against those.

## 10. Two regressions the full suite caught, and what they say

Both were found by running the whole suite rather than the new tests, which is
why they are recorded here rather than quietly fixed.

1. **`test_rust_engine_equiv._closed_tree_ref`** unpacked `_expand_closed`,
   whose return became `(utility, raw_value)`. A test *helper*, not a gate --
   but it is the helper the F3.2 tree-equivalence gate is built on, so the gate
   went from passing to erroring. Fixed by unpacking; the two values are equal
   on that unbiased path.
2. **`PhaseDLoop.specialist_configs` did not exist on a bypassed `__init__`.**
   Several harnesses build a loop with `object.__new__` plus a few hand-set
   attributes to exercise one method without a run directory, and
   `league_assignment` now reads the league from an instance attribute. That
   raised `AttributeError` deep inside generation. Fixed with **class-level
   defaults** meaning "no league", which is what such a harness intends.

The second is the more interesting one: any future field `league_assignment` or
`generate_iteration` reads has the same failure mode, and the class-level
default is the pattern that closes it.

## 11. Test evidence

```
test_specialist_bias.py       39 passed   (S0: formula, sign, parity, cache, inertness)
test_specialist_league.py     65 passed   (S1-S4, S2b, end-to-end, review regressions)
test_specialist_probe.py       6 passed   (S0a instrument)
games/seven_wonders_duel    1621 passed, 5 skipped   (full suite, 28 min)
```

The league file grew by 13 tests and the probe by 2 in response to the
2026-09-08 review. They assert the things the original tests did not: the next
OPPONENT after a rollback rather than the bytes of `latest.pt`, the next
OPTIMIZER state, the step count a zero-inflow iteration earns, what a retried
iteration does, whether the floor's subject carries its bias, what `collate`
hands the value head, and that a disabled league reproduces the pre-W7 HOF
draw.

Existing gates re-run unchanged: `test_rust_engine_equiv.py` (33, including the
committed corpus and the resumable-vs-oracle gate), `test_search.py` (46),
`test_phase_d_example_cache.py` (23), `test_league_generation.py` /
`test_league_routing.py` (29), `test_training_adapter.py`, `test_train_loop.py`,
`test_buffer.py`, `test_data.py`, `test_rust_derivation.py`,
`test_target_version.py`, `test_training_parameters_doc.py`.

---

## 10. S0a measured (2026-09-09) — lambda is ~6x larger than shipped, and the
##     search-time steer is small

**Run at last.** S0a had never been executed against a real net. It has now,
and it moves the shipped default and bounds what the mechanism can do.

### The net

No existing checkpoint could serve, on two independent counts:

* **Encoder** — iter0085's `tableau` block is 26 features against today's 37
  (W3's control channels) and `global` is 132 against 133. The embedder's
  `Linear` shapes do not fit; this is not a fussy signature check.
* **No `hier_joint7`** — the bias reads W4's hierarchical outlook via
  `Evaluator.outlook_tensor`, NOT the flat `joint7` head. iter0085 has
  `heads.joint7` and no hierarchical head, so it would have hard-errored at the
  first biased leaf even with a matching encoder.

So a net was trained from cloud2's own buffers, which store **game records, not
encoded tensors** and therefore re-derive through the current encoder: 8k games,
3 epochs, 256x6 (5.2M params), `--hierarchical-value`. `joint7_acc` 0.512
against a 0.334 base rate (iter0085 reached 0.582 — this net is weaker, which
matters below).

### The ladder, scientific, 200 positions at 256 sims

| lambda | moved | credible | q_cost | pursuit (rel) |
|---|---|---|---|---|
| 0 | 0.000 | — | — | +0.0% (clean control) |
| 0.5 | 0.060 | 0.583 | 0.079 | +2.2% |
| 1 | 0.075 | 0.600 | 0.071 | +3.6% |
| 2 | 0.105 | 0.571 | 0.066 | +5.8% |
| **3** | 0.155 | 0.516 | 0.076 | **+6.2%** |
| 5 | 0.170 | 0.471 | 0.085 | +6.0% |
| 8 | 0.225 | 0.511 | 0.094 | +5.1% |
| 12 | 0.230 | 0.435 | 0.105 | +3.6% |

**Pursuit peaks at lambda 3 and DECLINES above it** while credibility falls
monotonically: past the peak the specialist pays more Q, breaks more moves, and
pursues its type *less*. The shipped `science:0.15:0.5` carries lambda **0.5**
(the 0.15 is the share), which sits at 6% of decisions and +2.2% pursuit. Now
`science:0.15:3,military:0.10:3`.

### Why the ceiling is where it is

Measured directly across sibling moves, median spread:

| quantity | spread |
|---|---|
| **utility** | **0.567** |
| science outlook | 0.031 |
| military outlook | 0.058 |

**The outlook separates moves ~18x less than value does.** For the science bias
to rival the value difference between moves, lambda would have to be ~18 — at
which point move choice is nearly independent of strength.

### The symmetric arm: a hypothesis raised and killed

`own - other` separates siblings 2.6x better than `own` alone (0.081 vs 0.031),
which predicted more steering authority per unit lambda. It delivered more
CHURN and no more pursuit:

| lambda 3 | moved | credible | q_cost | pursuit |
|---|---|---|---|---|
| asymmetric | 0.155 | **0.516** | 0.076 | **+6.2%** |
| symmetric | 0.200 | 0.425 | 0.107 | +6.0% |

The ceiling did not move when the discriminative power changed by 2.6x, so the
ceiling is **not** set by the bias's steering power. It is set by what is
reachable one ply out: the bias selects among available moves and cannot
manufacture science potential that is not on the board. `--symmetric` is kept
because it is the arm that rules the hypothesis out; asymmetric is what to run.

Military is the same story at lower amplitude: up to 26% of moves changed at
**zero or negative** pursuit, in both forms, at every lambda.

### What this does and does not say

It does **not** reject the workstream, and the plan already says why: S0a is
frozen-weights and single-ply. A specialist plays ~40 decisions, and whether a
+6% steer at each compounds into a materially different game is **exactly what
this probe cannot see.** Pursuit over a game is S0b and the pilot.

It does say: **lambda alone will not double the science rate.** The target of
20% -> 40% of wins corresponds to root outlook ~0.11 -> ~0.21; the best any
lambda achieved was 0.119. If the workstream delivers, the training loop
delivers it and the bias only bends data collection.

### Caveats on the number 3

Lambda's bite scales with how sharply the outlook head separates sibling moves,
and this was measured on a deliberately small net. **A stronger net should want
LESS than 3, not more** — bracket downward at bootstrap. 200 positions, one
buffer file, PUCT root.

