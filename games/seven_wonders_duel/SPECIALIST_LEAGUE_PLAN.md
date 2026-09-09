# Specialist opponents inside the training loop (W7 build plan)

**Status: BUILT (2026-09-08), UNRUN.** S0-S4 and S2b are implemented and
tested; S0a, S0b and the S5 pilot are runs, and none of them has been run. See
`SPECIALIST_LEAGUE_REVIEW_REQUEST.md` for what was built, the three places the
implementation departs from this document, and what a reviewer should attack
first. The build sheet below is unchanged and remains the specification.

This is the build sheet for Workstream 7 of
`WORLD_CLASS_MODEL_EVOLUTION_PLAN.md`. That workstream says *what* specialists
are for and *when* to build them; this says *what code has to change* and in
what order, and records the one design decision that departs from it.

## The design, in one paragraph

A specialist is the same architecture as the general net, fine-tuned from a
promoted general checkpoint, whose **search leaf value carries a bonus for its
intended victory type**. It does not get its own self-play campaign. It plays as
the opponent seat inside the ordinary loop, exactly where an HOF archive sits
today; its own seat's search targets go to its own buffer, the general seat's
targets go to the general's buffer, and **every value target the general trains
on is unshaped** -- which, as the value-leak section below shows, takes more than
leaving the terminal label alone. One loop, three buffers, three train steps,
three sets of weights.

**The goal has two halves, and the integrated league only serves one of them
directly.** Facing specialists teaches the general to *defend*; nothing in the
seat arrangement teaches it to *execute* those attacks, because the attacking
policy targets belong to the specialist. S2b exists for the second half, and the
pilot in S5 is designed to measure whether it is needed.

### Why folded in rather than a separate run

A specialist trained in its own loop plays only lambda-biased copies of itself.
Nothing there punishes over-commitment to the rush, so it converges on always
rushing and becomes a smarter `ScienceAggressiveBot` -- which is the object W7
exists to replace. The general model is what keeps a specialist honest, by
beating it whenever the rush is unsound, and that pressure exists only if they
share a loop. Folding in also makes the plan's "constantly refreshed" property
structural rather than a scheduled chore, and costs almost nothing in
generation: specialist games *replace* games the general was going to play.

### Why the bias goes in search, not in the labels

The parent plan writes the shaping as
`reward = game result + lambda * is_science_win`, and then has to quarantine
shaped rewards in specialist-only buffers. Two reasons to put lambda at the leaf
instead:

1. **There is no scalar to add to.** The value target is a 3-class label
   (`dataset._actor_value_class`) and the outcome target is the 7-class
   `joint7` (`dataset._joint7_class`). `result + lambda * indicator` is not
   expressible as a class.
2. **Behaviour reaches the model through the search targets.** The policy learns
   the visit distribution. Bias the leaf value and the visits move; reweight a
   stored label and nothing moves, because that target already points where the
   unbiased search pointed.

Consequence: every value head stays a calibrated win probability, and the
general buffer can take the whole game for value.

**But "keep the terminal label unshaped" is not sufficient, and the first
version of this plan was wrong to stop there.** `value_bootstrap` blends the
recorded per-move `root_value` into the value target as `value_soft`
(`dataset.py`, `train.py`), and cloud2 ran it at **0.5** -- half the value target
is the search's own root estimate. A specialist's `root_value` contains the
lambda bonus, so any example drawn from a specialist's move would teach the
general a distorted win probability at full strength. The search utility and the
value target must therefore be **separate recorded quantities**:

* `root_value` keeps its current meaning -- the search's utility, whatever that
  search was optimising -- and is consumed only by the model whose lambda
  produced it;
* a second field records the same root under **lambda = 0**, and that is what
  feeds any other model's `value_soft`. Accumulate it directly as a second value
  sum over the same visits -- do **not** reconstruct it by subtracting
  `lambda * root_outlook` afterwards. The bias is additive per leaf, but the root
  outlook is accumulated over its own visit set, so the subtraction is only exact
  if those two sets coincide, and nothing currently guarantees that;
* an example whose route is not this model's carries no `value_soft` at all if
  the unshaped root is unavailable.

This is a per-move contract, not a per-buffer one: quarantining by buffer does
not help, because the general is meant to learn value from the specialist's
positions.

## What already exists

More than expected. The seat-routing work from W1.3 and the outlook work from W4
between them supply most of the mechanism.

| Capability | Where | Notes |
|---|---|---|
| Per-seat network routing in generation | `self_play.rs` `net_by_player`; `LeagueAssignment.nets_p0/nets_p1` (`phase_d.py:2201`) | Network 0 = learner, 1 = archive; routed on the **searcher**, not the leaf actor |
| Two-net batched generation | `rust_bridge.rust_searcher_routed_flat_batch_adapter` | One `self_play_many_flat_net` call mixes routed and pure games |
| Per-move policy exclusion | `buffer.MoveRecord.policy_excluded`, set in `self_play.rs` `finish_move` | Already drops an opponent's policy while keeping its value |
| Seven-way outlook at every leaf | `eval.rs` `Outlook = [f64; 7]`, `LeafOut.outlook_p0`, `terminal_outlook_p0` | **This is the lambda hook.** P(my science win) is already computed, p0-relative, for network leaves and terminals alike |
| Root outlook recorded per move | `root_outlook` on each move in `GameRecord` | Lets a specialist's behaviour be audited after the fact |
| Opponent archive and sampling | `az_loop/hof.py`, `hof_opponent_fraction`, `hof_sampling_mode` | Only promoted checkpoints are admitted (`phase_d.py:5308`) |
| Generator / promotion state machine | `az_loop/training_control.py` | `soft_gate` generates from `latest`, falls back to `current_best` after a reject; `probation_reset_after` resets the frontier |
| Per-opponent outcome grouping | `LeagueAssignment.name` (`hof_iter_NNNN_hash`) | Stats can already be grouped by opponent identity |

## What has to be built

Staged so each stage answers a question before the next one spends anything.
**Stage 0 is the point of the ordering** -- it costs a config flag and tells you
whether the workstream is worth its cloud time.

### S0. Biased opponents with no new weights (cheap diagnostic)

Add a leaf-utility bias, keyed by the **searching** net (not the leaf actor), in
both engines.

**The utility formula, decided.** Use the parent plan's own-win form, written in
p0 terms because that is what the utility already is:

```
specialist on seat 0:   utility_p0 = value_p0 + lambda * outlook_p0[p0_science_win]
specialist on seat 1:   utility_p0 = value_p0 - lambda * outlook_p0[p1_science_win]
```

Not the symmetric `outlook[p0_science] - outlook[p1_science]`, which the first
draft of this plan wrote. Symmetric is a different agent: it rewards pursuing
science *and* penalises conceding it, making a science mirror-player rather than
a science attacker, and it moves the specialist's defensive behaviour in exactly
the dimension S0 is trying to measure. Keep symmetric behind a flag; default to
own-win.

**The sign trap, stated exactly.** `Outlook` is p0-relative
(`eval::outlook_to_p0`) and the utility is p0-relative, but "my science win" is
specialist-relative. Writing `+ lambda * outlook_p0[my_win]` with a
specialist-relative index **reverses the bonus when the specialist is p1** -- it
would reward the opponent's science win. Hence the two explicit forms above.
Pin all seven terminal utilities in tests, keep the convention fixed across extra
turns, and document the widened utility range: the bias enlarges the value scale,
which changes how PUCT's exploration constant is scaled against it.

**Integration contract -- every path that produces a leaf value.** The first
draft named only two sites and would have silently no-opped in production:

| Path | Sites |
|---|---|
| Python search | `search.py` `value_actor = wdl[0] - wdl[2]`; `_terminal_value_p0` |
| Rust batched tree | `tree.rs` leaf handling; `eval.rs::terminal_value_p0` |
| **Rust resumable search -- the production generator** | `tree_resumable.rs`: `ImmediateLeaf` construction, its `terminal_value_p0` call, the **cached-evaluation** replay path, and the forced/wave `EvalBatchRequest` results |
| Root initialisation | the root's own expansion value, which seeds `value_sum_p0` |
| Solver shortcuts | `solver.rs` exact values entering the tree (`solver_value` on the move record). These carry no outlook, so the missing-outlook policy below decides them |
| Root accumulation | `outlook_sum` / `outlook_visits` must keep the *unshaped* outlook, so `root_outlook` stays an honest record |

**Specialist identity travels with the search session**, not on a mutable global
and not derived from the leaf actor. `SearchSession` already exists per searcher
(`self_play.rs`); the lambda and type belong there. Deriving from the leaf actor
would reproduce the bug that made `rust_seat_routed_flat_batch_adapter` the wrong
routing for league play -- a third player belonging to neither side.

**Outlook must become mandatory on any biased path.** `LeafOut.outlook_p0` and
`ImmediateLeaf.immediate_outlook` are `Option` today, and several branches
construct `None` -- including models without a W4 outlook head and solver
boundaries. A `None` outlook under lambda > 0 must be a hard error, not a silent
zero bias, or the treatment reaches some leaves and not others and the experiment
measures nothing. Two prerequisites follow: nonzero lambda **requires** a net
with a usable outlook source, and the solver path needs a stated policy (derive
the outlook from the exact terminal, or refuse the shortcut under lambda).

**Acceptance.**

* `lambda = 0` is bit-identical to today, verified on the resumable path
  specifically: the committed equivalence corpus is the instrument.
* `lambda > 0` gets its own equivalence gate -- Python and Rust must agree
  move-for-move on a fresh biased corpus. Without it the bias exists in the
  reference implementation and not in the one that generates the data.
* Cached and forced evaluations carry the same bias as fresh ones: replaying a
  cached leaf must not change its utility.

**S0 is two experiments; run them as two.**

* **S0a, frozen weights.** Utility correctness and attack *coverage*: does the
  biased search actually visit and fund the attacking continuations, or does the
  incumbent's prior keep excluding them? This needs no training and answers
  whether the mechanism does anything at all.
* **S0b, a small general-training A/B.** Defence can only improve through
  training and held-out evaluation; a generation-only test can show a behaviour
  change and nothing more.

**Acceptance is not "attempted attacks went up".** An agent can attempt more
*unsound* rushes and teach the general nothing. Require credible attacks --
attacks that a lambda-zero search still rates as reasonable -- and ordinary
playing strength that has not collapsed.

**This is a diagnostic, not a rejection gate.** A null is ambiguous by
construction: biasing an unchanged net can fail because that net undervalues
attacking lines, because it assigns attack moves tiny priors, or because search
cannot find their continuations inside the budget -- all of which fine-tuning is
meant to change. A null triggers diagnosis of attack credibility, outlook
quality, candidate coverage and statistical power, and permits a bounded
fine-tuning pilot before the workstream is rejected.

### S1. Specialist identity in the league assignment

Generalise `LeagueAssignment` from "one archive" to "one **opponent class** per
iteration", drawn from `{hof, science, military}` by configured shares.

* Carry the opponent's class, its lambda, and its buffer route on the assignment.
* Extend `name` so stats group by class as well as by checkpoint.
* Schedule the shares the way `hof_opponent_fraction` is scheduled today.

**Constraint, stated precisely.** `_SearcherRoutedModel.forward` hard-rejects any
net id outside `{0, 1}` (`rust_bridge.py`, "packed net ids must be 0 or 1"), so a
**single generation call** can carry at most two networks. That is narrower than
"one class per iteration": several two-net batches within one iteration can each
use a different opponent class, so iteration-level mixing needs no fan-out
change. Only one *mixed call* would. Rotation is still the reasonable first
choice, for the batching reason the class already documents -- one opponent model
cached on the device per call.

### S2. Per-seat target routing and buffers

Today `policy_excluded = !full || net_by_player[actor] != 0` -- *any* non-learner
policy target is dropped. A specialist needs the opposite: its targets kept, and
routed to its own buffer.

* Replace the boolean with a per-move target **route**: `general`,
  `specialist:<id>`, or `none` (cheap searches, curriculum bots, archived HOF).
  **A route is not a substitute for eligibility.** `policy_excluded` currently
  conflates three separate facts -- search quality (full vs cheap), which model
  searched, and who may learn the label. Keep all three, and audit every consumer
  before changing the field: `is_fast_search_move`, the archive-name exclusion,
  the reply-policy label, the action-policy loss, the cached-example path, and
  old-buffer compatibility (records written before the change carry no route).
* `examples_from_record` gains a route filter, so one record yields the general's
  examples and the specialist's examples separately.
* One replay buffer and example cache per model. Terminal value and outcome
  labels are shared across models; the policy label and the bootstrapped
  `value_soft` are routed (see the value-leak section -- `value_soft` came from a
  search that was optimising something else).
* Dirichlet noise is currently learner-only, for a good reason -- handicapping an
  archive inflates the learner's league win rate. A *learning* specialist must
  explore, so the predicate becomes "any net that is training", not "net 0".

### S2b. Transfer of attacking knowledge (lambda-zero reanalysis)

Without this, the general learns only defence. The specialist's attacking policy
targets train the specialist; the general's own seat was defending.

Retain the positions where the specialist's attack was live, re-search them at
**lambda = 0**, and route the resulting targets to the general. That teaches the
general to play sound attacks -- what the *unbiased* search makes of the position
the biased agent steered into -- without teaching it to trade wins for a
preferred victory type.

* **Selection, so the cost stays bounded.** Reanalysis is search compute, so do
  not re-search everything. The biased search already knows where lambda
  mattered: it holds both the shaped and the unshaped root utility, so "lambda
  changed the chosen move" is a cheap local flag. Re-search that subset, plus
  positions from games the specialist won by its type.
* **Route the results as general examples**, marked as reanalysis in the record,
  so their share of the general's buffer is measurable and capped.
* The reanalysis evaluator is the general's own current net. These are targets
  for the general, not a second opinion from the specialist.
* **Re-search from the actor's observation, not the realised deal.** The stored
  record knows the hidden cards; a reanalysis that sees them produces targets no
  player could have computed, and would train the general on clairvoyant play.
* Give the specialist's candidate moves enough search coverage that the general's
  weak prior cannot simply exclude them again -- the failure S0a is designed to
  detect applies here too.

### S3. Three train steps per iteration, sized by inflow

`run_iteration` (`phase_d.py`) performs one `train_candidate`; it becomes one per
model. **Do not give them equal step counts.**

At an opponent share of *f*, a specialist sees roughly *f*/2 of the positions the
general sees -- one seat of the games it appears in. At *f* = 0.15 that is ~7.5%.
cloud2 ran at `samples_per_new_position` around 5.2; the same step count on 1/13
of the inflow is roughly 65 samples per new position, which memorises the buffer
within a couple of iterations. Scale each model's steps to its own measured
inflow, or train specialists every N iterations.

### S4. Specialist archive and collapse floor

**Archive from the first accepted specialist, not later.** One active net plus
one rollback checkpoint per type is not enough: attacking styles are forgotten
the same way general strategies are, and a specialist that drifts takes its
earlier style with it. Keep a small per-type archive beside the general HOF,
admitted on the same "was accepted" rule, and sample it into the opponent share.
The first entry of each type doubles as the frozen reference the measurement
section needs.

A specialist is sparring equipment, not the product, so it does not need the
promotion gate. It needs a floor that catches divergence before a broken
specialist poisons the general's buffer:

* its score rate against the current generator must stay above a configured floor;
* if it falls through, revert that specialist to its last good checkpoint and log
  it loudly.

Deliberately **not** the general soft-gate controller: that gate produced 0
promotions over 38k games in cloud6, and a specialist population that silently
never advances is a full run wasted before anyone notices.

### S5. Pilot one specialist, and test the transfer question directly

Do not build all three classes at once. Take one type -- science, since the threat
corpus and the documented value-head weakness are both science-side -- fine-tune
one specialist inside the loop, and run the comparison that decides the shape of
everything after it:

**defensive exposure alone** vs. **defensive exposure plus lambda-zero
reanalysis (S2b)**.

Both arms face the same specialist; only the general's target stream differs.
That isolates whether attack transfer earns its search compute, which is the one
question the seat arrangement cannot answer by itself.

## Contracts to settle before coding

**Shares arithmetic.** Shares are fractions of *all* games. With 40% league games
and one class per iteration, drawing HOF/science/military at 37.5%/37.5%/25%
gives the intended 15%/15%/10% expected game shares.

**Generation is not free, and "specialist games replace general games" is not
cost-neutral for the general.** At that mixture roughly 20% of move opportunities
belong to an opponent seat, and those moves yield the general no policy target.
Replacing games preserves the general's *game* count, not its policy-target
inflow. Track policy-eligible inflow separately from shared value rows, and
measure end-to-end generation and training cost rather than calling S0 or the
league free. The S3 step-sizing rule depends on this number being measured, not
assumed.

**Provenance on every example.** Record the utility type and lambda that produced
each search, and which model the target is for. Without it, a buffer cannot be
audited after the fact and the S2b reanalysis share cannot be capped. Add an
isolation test that runs with `value_bootstrap > 0` and asserts no shaped root
reaches the general's `value_soft` -- the configuration that makes this dangerous
is the one cloud2 actually ran.

**Resume and rollback.** Each specialist needs its own persisted lineage:
checkpoint chain, optimizer and scheduler state, replay/cache identity, update
count, lambda, and the deterministic assignment state that decides which games it
plays. Define what happens to its replay window after a rollback or reseed. And
keep the lifecycles separate: **a specialist must not be reset because the
general's soft gate rejected the general's own candidate.**

## Measurement

Report by opponent class, never as one aggregate.

**Everything below is measured against frozen references.** Scoring an improving
general against an improving specialist cannot separate "defence got stronger"
from "the attacks got weaker", and that ambiguity would make the workstream
unfalsifiable. Fix these at the start of the run and never rebuild them:

* a frozen attacker of each type -- the first accepted specialist of that type;
* a frozen general anchor;
* a fixed position suite. The threat corpus is the natural source: it already
  carries labelled science and military threat positions.

The S4 collapse floor uses the frozen anchor for the same reason -- a floor
defined against the current generator moves under the thing it is meant to
protect.

**Is the specialist doing its job**

* intended-victory-type rate when it is the attacker;
* score rate against the frozen panel, plus *how* it loses. Win rate alone
  cannot separate "loses 45-55, useful pressure" from "loses 5-95, already
  solved". Two corrections to the first draft of this line: **margin is the wrong
  instrument** -- `margin_valid` is false for every non-civilian ending
  (`dataset.py`), which is exactly the endings a specialist produces -- and the
  claim that only near-balanced opponents teach anything is **too strong**: a
  specialist with a low overall score can still be worth keeping if it holds a
  reproducible exploit on the fixed panel;
* exploitability: how the general's score against it trends over iterations;
* **pursuit and defence, separately**: how often the specialist creates a live
  threat of its type, and how often the general converts a live threat when *it*
  is the attacker. The second is the S2b metric, and it has no reason to move
  without S2b.

**Is the general actually improving** -- the real success criterion, and the one
easiest to omit because it is not a property of the specialist at all:

* the general's loss rate to science and to military attacks, over iterations --
  read only against the frozen attackers, since a falling loss rate against a
  *live* specialist may just mean the specialist stopped attacking well;
* sound attacking opportunities converted, and defensive opportunities handled,
  on the fixed position suite;
* threat-value error on that suite. **Do not optimise the general's raw share of
  science/military wins** -- choosing a safer civilian win is often correct.
* the science-threat value error recorded in the project notes (the value head
  under-predicts committed opponent science, worst early) should shrink if this
  is working.

Specialists are expected to score *worse* overall than the general, by design.
That is not a failure; falling through the collapse floor is.

## Open decisions

1. **Opponent shares.** A 15% HOF / 15% science / 10% military split leaves 60%
   pure self-play. For reference: AlphaZero is 100% self-play, OpenAI Five 80%,
   AlphaStar main agents 35%. No published number transfers directly -- in
   AlphaStar every league agent has its own generation budget, whereas here
   specialists are seats inside the general's games, which is exactly what makes
   the S3 inflow problem real.
2. **One class per iteration vs. mixing within one.** Rotation is cheaper and
   needs no net-id change; only lift the `{0, 1}` cap if rotation proves too
   coarse.
3. **Lambda values.** The parent plan wants a small population on a Pareto
   frontier. Start with one lambda per type; add the population after S0 shows
   the effect is real.
4. **HOF sampling.** `recency` biases towards the newest promoted net, which is
   the *least* diverse choice; PFSP instead samples where the win rate sits in a
   target band, deliberately retaining awkward older opponents. At cloud2's HOF
   size -- **7 entries**, iterations 0, 5, 10, 15, 25, 35, 50 -- recency gives the
   newest 25% against uniform's 14%, so this is a low-priority knob until the
   pool grows.

## Not in scope

* Dense proxy rewards (green-card count, shield count, track position). The
  parent plan is right that these produce agents which accumulate the resource
  and lose the game.
* Replacing the scripted rush bots. They stay for curriculum and smoke testing.
* Any of this before W1/W2/W3/W5 have been through a real run. Specialists
  trained against an architecture that is about to change are compute spent
  twice, which is why W7 is last.
