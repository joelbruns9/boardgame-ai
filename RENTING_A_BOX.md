# Renting a box

A game-agnostic playbook for launching a multi-day training run on rented
hardware, written from the defects that actually happened rather than from
general principles.

The recurring failure is not "something broke." It is **something looked fine
and was wrong**, discovered hours later on a machine billed by the hour. Every
rule below exists because a specific plausible-looking output was false.

---

## 1. Before you rent

Everything here is free on a laptop and expensive on a box. None of it needs a
GPU.

### 1.1 Run the real launch command, not a smoke test

The launcher assembles a command line. Assemble it and read it.

- **A flag the launcher does not pass is not "unset" — it is whatever argparse
  says**, which is a training decision nobody made. Instances: `--train-steps`
  defaulting to 300 (8× sample reuse instead of the ~5× the loop is tuned for);
  `--weight-decay` missing entirely; `--opponent-fraction` missing.
- Diff the assembled command against the **previous run's manifest**. Anything
  present there and absent now must be a deliberate decision with a reason
  recorded, not an omission.
- Keep a test that every flag the launcher passes exists on the parser, and that
  every flag the previous run used is either passed or listed as deliberately
  dropped. Both have caught real omissions.

### 1.2 Put the values in the command, and make the rest visible

The most expensive recurring defect in this project is not a wrong value. It is
a value *configured in one place and decided in another*: the launcher
"configures" a run, argparse quietly supplies what the launcher omitted, and the
manifest records a number nobody chose. It happened to `--train-steps` (8x the
intended sample reuse), `--weight-decay` (5,000x off), `--opponent-fraction`,
and to `--leaf-batch` — a value settled by a paired A/B and then never passed at
all, so the run would have discarded the conclusion silently.

Banning defaults is not the answer; there are 130+ options and most should have
one. Three mechanisms, in increasing strength:

1. **Pass the training values explicitly in the launch command**, even where the
   default is currently right. A default that happens to match is not the same
   as a decision, and it stops matching the day someone changes it.
2. **Print what was not passed.** Diff `sys.argv` against the parser at startup
   and log every setting taking its default, into the run log and the manifest.
   Buried defaults stop being buried: they become a list read in the first ten
   minutes rather than discovered on iteration forty. Count each *setting* once
   — `--x`/`--no-x` is one decision and `--help` is none — or the list is twice
   as long as it should be and gets dismissed.
3. **Refuse on a short critical list.** A handful of flags whose wrong value
   fails *quietly* rather than loudly: the run exits rather than starting on an
   unstated value. The selection test is not "is this important" but "would a
   wrong value here be noticed" — a flag that crashes when wrong does not need
   to be listed, and a long list gets bypassed.

**Coupled knobs must not default independently.** Where one setting is only
meaningful given another, deriving the dependent one is not a nicety — two
independent defaults produce combinations nobody chose. Two instances in one
launcher, one session:

- a cheap-path batch size pinned at 16 while the wave flags it requires were
  switchable, so turning waves off produced a config the run rejects;
- an evaluation batch pinned at 16 while the opt-in it requires was switchable,
  which killed two consecutive launches.

Both were rejected correctly and far too late. Derive the dependent value from
the one it depends on, and let a test walk every combination the launcher can
emit through the run's own validator.

### 1.3 Make the launcher validate the command it assembled

Before detaching. Not after.

Give the training entry point a `--validate-config` mode that builds the
configuration, validates it, prints `config OK` and exits. Have the launcher run
it on the assembled command and refuse to launch if it fails.

Three consecutive relaunches died on knob pairings decided in the launcher and
rejected by the run — each **after** the toolchain, Python deps, crate build,
GPU gate, preflight, equivalence suite and plumbing smoke had all run, and each
inside a detached process whose only trace was a log file. The check costs about
two seconds.

One detail worth getting right: when reconstructing the arguments, drop the
interpreter, `-m` and the module name. Passing the module as an argument makes
the check pass on a command it never really parsed.

### 1.4 Make the launcher survive its own update

**Trap:** the launcher sources shared code at the top and `git pull`s several
stages later. The first re-run after any change pulls the new code and then
keeps executing the copy bash already read. The manifest records the new commit
while the command line comes from the old one. Nothing fails.

This appeared **three times** in one project: in each of the two game launchers,
and again in a helper script fetched by `curl`.

The fix is a checksum before the update and an `exec` of the fresh copy after
it, guarded against looping (`common::self_checksum` / `common::reexec_if_updated`).

### 1.5 Give every script a version marker

Print a version and a checksum of itself at startup:

```
==> sweep_7wd.sh version 6 (checksum 2904670517)
```

Without it, "the fix did not work" and "you ran a stale copy" are the same
observation. `raw.githubusercontent.com` is CDN-cached and a `curl` seconds
after a push can legitimately return the previous file — this happened, and the
version line is the only reason it was diagnosed in one round trip instead of
several. **Prefer running from a git checkout over `curl`;** git does not cache.

### 1.6 Check that your gates can fail

A gate that cannot fail is worse than no gate: it reports safety it never
checked. Mutate the code the gate protects and confirm it goes red. Real
instances of gates that could not fail:

- an async-solver test that drove a scheduler with no solver pool;
- a cost-model parity check run on games that retire no wonders;
- tests asserting a config could *compute* a value, while the call site
  ignored it;
- an isolation test asserting `PYTHONPATH in file`, which passed while one of
  two invocations had lost it.

**A stage that reports failure must actually fail.** A helper that ran each
verification stage announced `FAILED (exit 0)` and then returned success, so the
`|| die` at every call site was inert: the equivalence suite and the plumbing
smoke had never been able to stop a launch, which is the entire reason they run
before one. The cause was reading `$?` after a completed `if ... fi`, which is
the status of the **if statement** — zero when the condition failed and there is
no else branch. Capture the status at the point of failure:

```bash
local status=0
( "$@" ) >"$logfile" 2>&1 || status=$?
```

Test it by making a quieted stage fail and asserting the caller stops.

**Write the fixture from the producer, not from your reader.** A log-shape
fixture invented by the consumer will confirm the consumer's own error — that is
exactly how a profiler shipped reading one nesting level too shallow, and every
test passed.

### 1.7 Know which settings are frozen once the run starts

Every loop has some notion of run identity. Find it and write down both lists:

- **Refused on resume** — schedule constants, precision, code identity.
- **Free to change** — usually the scheduler geometry, thread counts, batch
  sizes, learning rate.

Getting this wrong in either direction is costly: you either abandon a healthy
run you could have kept, or you silently invalidate one you should have
restarted.

---

## 2. The first ten minutes after launch

### 2.1 Read the manifest back

Not the console output — the manifest. Confirm every flag you intended, by name
and value. This is the step that catches §1.4.

### 2.2 Confirm the run's own arithmetic

Launchers derive values. Print the derivation and check it:

```
Solver threads: 7 per shard x 2 shards = 14 concurrent solves, on 16 cores.
```

**Per-unit values multiplied across a boundary are the single most common
defect class.** Solver threads were per *shard*, so a worker count silently
multiplied them. Leaf batch was per *seat*. Wave width is per *game* while batch
width is per *shard*. Whenever a number is "per" something, print the product.

### 2.3 Surface what the run already records

Before building any new instrumentation, check what is already being written and
never read. In this project the scheduler had always recorded per-shard time
breakdowns, batch widths, wave widths and solver blocking — none of it surfaced.
Questions were being answered by inference for weeks while the answers sat in
`training_log.jsonl`.

### 2.4 When it fails on the box, read the log before theorising

The most expensive habit in this whole exercise was diagnosing a remote failure
by inference. Three launches failed; two diagnoses were confident, plausible and
wrong, and each shipped a fix for a defect that was not the cause. The log
contained the exact `ValueError` both times.

- **Read the log first.** A process that dies in five seconds leaves a
  traceback or a one-line refusal. It is one command.
- **A fix shipped on a hypothesis is a second change to reason about**, not a
  step forward. The two "fixes" were real defects, which made it worse — they
  were plausible enough to look confirmed.
- **`exit 0` does not mean the work happened.** A backgrounded pipeline reported
  success having produced no output; a test run rejected an unknown argument and
  still exited 0. Check for the *result* — a pass count, a written file — not
  the status.
- **State which copy ran.** Version markers exist so "the fix did not work" and
  "you ran a cached copy" stop being the same observation.

---

## 3. Defect patterns to check for by name

| pattern | what it looks like | how it was caught |
|---|---|---|
| **Silent default** | a value crossing a boundary and being defaulted | reading the launch command back |
| **Stale code** | update fetched, old copy still executing | version markers |
| **Wrong configuration measured** | the benchmark configures X and measures Y | comparing benchmark config to the run's manifest |
| **Confounded axis** | a second variable moves with the one being swept | printing the per-point split |
| **Gate that cannot fail** | green test, absent check | mutation |
| **Unreadable failure** | "no data found", cause unstated | make the tool print what it *did* find |
| **Advice outliving its evidence** | a tool recommending what the project later measured as wrong | re-read hints when a measurement lands |
| **Coupled defaults** | two knobs defaulting independently when one requires the other | walk every combination through the validator |
| **Inert failure** | a stage reports FAILED and returns success | make a stage fail; assert the caller stops |
| **Mutually exclusive mechanisms** | one global flag silently disables another | measure the mechanism you think you enabled |

### 3.1 `git fetch` + `git checkout` is not `git pull`

Fetch advances `origin/main`; it does **not** move the local `main` a clone left
behind. `git checkout main` in an existing clone lands on the commit that clone
was made at. Check out the **remote** ref and detach:

```bash
git fetch --all --quiet --tags
git checkout --quiet --detach origin/main
```

### 3.2 Toolchains are installed but not on `PATH`

`rustup` writes to `~/.cargo/bin` and adds it to the shell *profile*, which a
non-login `ssh` command never reads. A box that built your crate at setup will
still report "Cargo metadata failed" in a fresh session. Source
`$HOME/.cargo/env` explicitly and fail with the cause, not the symptom.

### 3.3 Dataclass slots hide defaults

`@dataclass(slots=True)` makes `Config.field` a slot **descriptor**, not the
default value. Code comparing `Config.field` to a measured value silently never
matches. Read defaults via `dataclasses.fields`.

---

## 4. Rules for any measurement tool

A tool that measures the wrong thing is worse than no tool, because its output
is used.

1. **Measure the configuration you are configuring.** Reconstruct the run's
   settings from its manifest; do not let unnamed fields fall back to library
   defaults. One sweep measured Gumbel at 24/128 simulations to configure a PUCT
   run at 100/1600 — a different algorithm at a twelfth of the search.
2. **Put the current setting inside the grid.** A sweep whose axis excludes the
   value you are running cannot tell you whether changing it helps.
3. **Hold everything else constant across an axis.** If a per-unit value
   multiplies with the swept axis, express it as a total and divide.
4. **Report the realized value, not the requested one.** Asking for a batch of 8
   and receiving 1.97 is the interesting number.
5. **Separate contention-sensitive from contention-insensitive measurements.**
   A throughput sweep needs the box to itself. A *win rate* does not — both arms
   play the same game on the same GPU, so contention slows without biasing. That
   distinction decides whether a measurement can run beside training.
6. **Check the mechanism engaged before believing a null.** A sweep reported
   that leaf batching bought no throughput. It had been run with a global flag
   that caps the batching it was measuring, so the mechanism was half disabled
   and the null was an artefact. When a change measures as "no effect", first
   confirm the change took effect — a counter that moves is the cheapest
   possible check.
7. **Know which regime you are in before tuning a knob.** Batching only recovers
   fixed per-call cost, so its value depends entirely on how large that cost is
   relative to per-row work. Two points fit the model: at one batch size the
   overhead was ~11% of the call, and batching returned almost exactly that. The
   same knob is transformative on a small model or a single-position search and
   nearly worthless on a large model with high concurrency. Fit the curve before
   assuming a lever exists.
8. **State uncertainty as an interval.** "48.2%" invites a conclusion; "[43.4%,
   53.1%] — cannot separate from even at this sample size" states what was
   learned. Convert to the unit of the decision (Elo, hours, dollars).

---

## 5. Isolation: never write to the shared environment

A build tool that installs in place (`maturin develop`, `pip install -e`)
replaces the artifact a **running** process has loaded, and leaves the run's
engine out of step with the commit its manifest records. Resume guards typically
hash the *repository*, not the installed extension, so that drift is invisible.

Build an artifact and point at it instead:

```bash
maturin build --release
pip install --quiet --target "$EXT_DIR" path/to/wheel.whl
PYTHONPATH="$EXT_DIR" python -m your.module
```

Then **verify both halves**: the isolated copy must resolve first, and the
shared copy must still be where it was. Refuse an `EXT_DIR` inside any
`site-packages`.

This removes a conflict rather than arbitrating it. An earlier version refused
to run when the code differed, which blocked the exact case it was meant to
serve — measuring a configuration whose whole point is new code.

---

## 6. What actually needs the box

Rent for the GPU, not for the convenience.

| work | needs the box? |
|---|---|
| Training | yes |
| Throughput sweeps | yes — and exclusively; contention invalidates them |
| Strength A/Bs (win rates) | no bias from contention, so *may* share |
| Reading logs a run already wrote | no |
| Code, tests, engine changes, parity work | no |

**A stopped rented box costs exactly what a running one costs.** If a decision
will take hours, either let the run continue meanwhile or destroy the instance —
"stopped while I think" is the one clearly wasteful state.

Before destroying an instance, snapshot what a warm start needs: the checkpoint,
the manifest, the training log, and the replay games (concatenating per-iteration
buffer files produces a valid warm-buffer import). Verify the archive locally
before you tear the box down.

---

## 7. Resuming versus restarting

Prefer **resume** over warm start when the loop allows it: a warm start restarts
the games ledger, so every games-keyed schedule re-anneals from zero. Scaffolding
you already annealed out comes back.

Resume usually requires the checkout to stay at the commit the manifest records.
That has a practical consequence worth planning around: **do not `git pull` in
the run's checkout.** Put new tooling in a *separate* checkout, and have it fetch
its own code.

If you must warm start, zero the scaffolding schedules explicitly and pass the
staleness window, or an import filter will silently discard most of the history
you carefully preserved.

---

## 8. Schedules: decide what ramps before you rent

A rented run is a one-shot commitment: the settings that matter most are the
ones you cannot change once it starts. Section 1.7 says to write down which
those are. This section is about the consequence — **anything you would want to
change at hour 40 has to be a schedule at hour 0, or a controller inside the
process. There is no third option**, because noticing a plateau and restarting
with better flags is exactly what resume identity refuses.

### 8.1 Pick one clock and put everything on it

`--schedule-basis games` indexes every schedule on cumulative games rather than
iteration count. That is the right default because it decouples the schedule
from `--games-per-iteration`: you can resize an iteration on a resume without
moving any schedule position. Under an iterations basis that same change
silently re-times every anneal you configured.

`schedule_identity()` in `phase_d.py` is the enforcement. It pins the
games-basis constants — window coefficient, window exponent, window cap,
curriculum anneal, draft prior, opponent fraction, seed retain, sims schedule —
and refuses a resume that changes any of them, because a schedule changed
mid-run moves the meaning of every later iteration rather than only the current
one. `games_per_iteration` is deliberately absent from that list; that freedom
is the whole point of the games basis.

### 8.2 Ramp the expensive things up, not down

Early iterations train on a weak net whose targets are noisy anyway. Spending a
full search budget there buys precision the labels cannot carry. The general
shape:

| quantity | direction | why |
|---|---|---|
| search sims | **up** | cheap while the net is weak; expensive once targets are worth sharpening |
| replay window | **up** | a small window early keeps data fresh; a large one later resists overfitting |
| learning rate | **down** | large steps while the loss surface is coarse, small ones to converge |
| scaffolding (curriculum, draft prior, seed retain) | **down to zero** | it exists to bootstrap, and outstays its welcome |

### 8.3 What this project actually has

Three of those four are implemented; one is not.

- **Replay window — done, and it is a growth curve, not a constant.**
  `window = clamp(coefficient * total_games ** exponent, floor, cap)`
  (`games/az_loop/schedule.py`). At `--replay-window-coefficient 1000
  --replay-window-exponent 0.6 --replay-window-cap-games 20000` it grows
  sublinearly and then flattens at the cap.
- **Search sims — implemented, unused.** `--full-sims-schedule` takes
  piecewise-constant knots as `games:sims`, e.g. `0:400,10000:900,25000:1600`,
  and must define a value at 0 games. Runs that pass a flat
  `--full-sims-min/--full-sims-max` are opting out of it.
- **Scaffolding anneals — done.** `--curriculum-anneal-games`,
  `--draft-prior-games`, `--seed-retain-fraction`, plus `--gate-ladder-games`
  and `--hof-start-games` for the evaluation side.
- **Learning rate — missing.** There is no cross-iteration LR schedule.
  `train.py` accepts a `cosine_decay` argument, but `phase_d` never passes it,
  so `--learning-rate` is flat for the life of the run and
  `--train-warmup-steps` only warms within each iteration. This is the gap.

**Piecewise-constant, not continuous.** A continuously drifting budget makes
every iteration incomparable with every other, instead of only those across a
step. Steps give you segments you can average within.

### 8.4 Metric-based is the upgrade, and it must live in-process

A clock schedule guesses *when* the plateau will arrive. A metric-driven one
waits for it. The three levels:

0. **Constants.** Every value chosen once, at launch.
1. **Clock schedules.** Values move on the games counter. Where this project is.
2. **Metric triggers.** Values move when a measurement says the current setting
   has stopped paying — validation loss flat across N iterations, gate margin
   decaying, promotions per 10k games falling.

Level 2 is already built here and switched off. `--intervention-ladder`
(`intervention_ladder: bool = False`) multiplies exactly the right three axes —
`sims_multiplier`, `window_multiplier` (applied to both the window coefficient
and its cap) and `learning_rate_multiplier` — and holds a rung for
`--intervention-window-games` before judging its effect.

Its docstring names the blocker precisely: *"a running process cannot pick up a
mid-run change, and restarting to get one is what W6.5 refuses."* That is the
whole argument for level 2. The manual alternative — watch the metrics, stop the
run, relaunch with new flags — is refused by resume identity for a good reason,
and doing it anyway costs an in-flight iteration each time.

The existence proof that in-process control works is already running: the
soft-gate controller promotes, puts on probation and reverts entirely on
measured gate outcomes, with no operator in the loop.

### 8.5 Rules

- **Every schedule on the same clock.** Mixed clocks drift relative to each
  other the moment iteration size changes.
- **Print the resolved value every iteration**, not the configuration that
  produced it. A `schedule:` line showing `curriculum=0.000 draft_prior=0.000
  seed_retain=0.000` is how you know scaffolding actually annealed out.
- **A metric-triggered change is still a regime boundary.** Record it and its
  games clock so later analysis can segment at it — the pattern
  `--allow-hof-change` already uses, where the change is persisted and its clock
  becomes a revert-suppress knot. Recording a change does not make the
  iterations across it comparable; it makes the incomparability findable.
- **Decide the ramps on a laptop.** A schedule is cheap to reason about and
  impossible to change later, which is the worst combination to discover under
  billing.

---

## 9. Cost discipline

Throughput tuning has its own playbook: **`THROUGHPUT_LEVERS.md`** covers which
levers can be swept on wall clock, which need a strength measurement, and what a
sweep harness has to prove before its numbers mean anything.


- Decide the acceptable share of wall clock spent on evaluation *before* the
  run, not when the gate ladder surprises you.
- Prefer one long measurement to several short ones: a paired design resolves an
  effect with far fewer games than an unpaired one.
- Convert every proposed optimisation into days saved before doing it. A 35%
  throughput gain on a 7.6-day run is 2.6 days; an unmeasured strength risk on
  the same run may be worth more or less than that, and only saying both numbers
  out loud makes the trade visible.

---

## 10. The one-line version

Everything above reduces to: **verify the thing you are about to pay for, using
the thing itself, before you pay for it** — the real command, the real manifest,
the real log shape, the real extension, on a laptop, where a mistake costs
minutes.
