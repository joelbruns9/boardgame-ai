# W3 experiment design — review request

**Status: harnesses BUILT and smoke-tested, no result yet.** This asks whether
the measurement can be trusted *before* it runs, because the failure mode we
care about is not a crash — it is a number that looks like an answer and is not.

Two harnesses, to be run tonight:

| stage | question | cost |
|---|---|---|
| `w3_offline_ab` | can the network USE exact positional control? | ~2.5 h, GPU |
| `threat_corpus_measure` (reference pass) | the common yardstick arm regret is scored against | ~4 h, 4 CPU shards |
| `w3_corpus_regret` | per-arm regret against that yardstick | minutes, tomorrow |

Prior context: `W3_ENCODER_INTEGRATION_REVIEW_REQUEST.md` (design, rev 4) and
`W3_ENCODER_BUILD_REVIEW_REQUEST.md` (the build, seven findings worked).

---

## What each harness measures

**Stage 1.** Five arms from one frozen checkpoint (`extension_7wd/candidate_0085.pt`,
the model actually served by the advisor), same steps, same data, same split,
differing only in what the model is shown:

    baseline   control inputs OFF, no auxiliary head
    inputs     control inputs ON
    aux        inputs OFF, auxiliary control head ON
    both       inputs ON, auxiliary head ON        (omitted tonight, for time)
    shuffled   inputs ON, control CHANNELS PERMUTED across positions

Scored on held-out **games** — never positions, because every position in a game
shares its outcome label, so a position-level split lets a model score by
recognising a game it trained on.

**Reference pass.** The incumbent evaluates *every* legal action at each of ~48
live corpus positions. Expensive, arm-independent, reusable.

**Stage 3.** Each arm contributes only the action it would *play* (one search),
and regret is the reference value of the reference's best action minus the
reference value of the arm's choice. Every number from one yardstick.

---

## Design decisions taken deliberately

**Arms are compared to `baseline`, never to the incumbent.** The incumbent has
not had this training run, so comparing to it conflates the effect of control
with the effect of more training.

**Input-off is a runtime switch, not a schema variant.** Removing the channels
would move `MAX_FEATURES` and the signature, so arms would differ in width and
architecture. Pinned to zero, arms are bit-identical in shape and the control
columns receive exactly zero gradient (`x` is zero, so `dL/dW` is zero) — the
model *cannot* use control rather than being discouraged from it.

**Deltas are paired per seed.** Each arm is compared against the baseline that
saw the same seed and the same split, so training randomness cancels rather than
becoming the effect.

**`shuffled` is the causal control.** It permutes control channels across
positions within groups of equal tableau-token count, preserving every marginal
— same values, same distribution, same count of nonzeros — and destroying only
the correspondence between a position and its map. It operates on the encoded
features, not the auxiliary labels, because the inputs arm bakes control into
tokens at encode time; shuffling labels would leave the treatment untouched.

**Five seeds.** The cloud2 log shows iterations 75–90 scoring
0.530/0.545/0.533/0.537 against `best_iter_70` — individually short of the
promotion threshold, collectively about z = +2. Effects here are 3–5 points, and
one seed cannot resolve them.

---

## Known limits, stated before the run

**Stage 1 does not measure playing strength.** The cloud2 buffers were generated
by a policy that never saw control features, so it asks whether control helps
predict *existing* search targets. A positive result argues for an arena; a flat
one does not rule out a benefit in self-play, where a policy could actively
exploit the feature.

**Regret is agreement with the incumbent's search, not ground truth.** The
reference is the strongest evaluator available, not an oracle. Two arms
disagreeing with it is not proof either plays better.

**The corpus is a development set, not a holdout.** It has been inspected
extensively across this workstream.

**The reference pass covers ~48 of 164 live positions** (`--limit 12` × 4
shards), and those come from `--all-episodes` filtered by a triage that itself
covers 367 of 385 snapshots.

**One checkpoint, one architecture.** Nothing here says anything about a
different model size or a self-play run.

---

## What I want reviewed

1. **Is stage 1's metric the right one?** It reports held-out `policy_top1`,
   `value_acc` and total loss. Policy top-1 asks whether the model reproduces
   the search's chosen move. If control helps the model *evaluate* without
   changing its top choice, top-1 will miss it. Is a distributional metric
   (policy KL against the search target) the better primary, with top-1
   secondary?

2. **Is the reference model the right yardstick?** It is the incumbent at 600
   sims. An arm that agrees with the incumbent scores well *by construction*,
   which biases toward arms that changed nothing — precisely the null we would
   most like to detect. Would a deeper reference (higher sims, or the exact
   endgame solver where it applies) be worth the extra cost, and does the
   current design systematically favour `baseline`?

3. **The `shuffled` arm's permutation is grouped by tableau-token count.**
   That preserves shape but correlates with game stage, so a shuffled position
   receives a control map from a position at a similar point in the game. Does
   that leave enough real signal to make the control weak?

4. **Five seeds, four arms, one split.** The split seed is fixed across arms so
   they share a split. That makes the comparison paired but means a single
   unlucky split biases every arm the same way. Is one split with five training
   seeds the right allocation, or should splits vary too?

5. **The auxiliary head trains against labels from the same table the inputs arm
   reads.** If the table is wrong in some regime, `aux` and `inputs` inherit the
   same error and would agree with each other. Is there a check that
   distinguishes "both helped" from "both wrong in the same direction"?

6. **Nothing measures the harm case.** The plan warns that a science-heavy,
   preserve-tempo evaluation would reward a hoard-Wonders bias and call it
   improvement. Stage 1's holdout is ordinary self-play positions, which should
   dilute that, but nothing explicitly tests positions where **spending** tempo
   is correct. Should that be a required slice before any promotion?

---

## Not done

- No result of any kind; no arm has run.
- The `both` arm is omitted tonight for time.
- Corpus regret is scripted but has only been smoke-tested on 3 positions.
- The full test suite has not been re-run since these harnesses landed; the
  targeted tests pass.
- Stage 3 has no test coverage — it is a measurement script, and a bug in it
  would produce a plausible wrong number rather than an error.
