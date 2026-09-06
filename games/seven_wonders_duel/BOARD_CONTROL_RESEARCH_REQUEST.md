# Research request: exact board-structure understanding in 7 Wonders Duel

**Status:** REVIEWED TWICE 2026-09-05. §8 holds round 1; **§9 holds round 2, which SUPERSEDES §8 on Q5 and on the next build.** Read §9 first.
**Audience:** an external reviewer with game-AI / search background, reading cold.
**What is wanted:** a judgement on *mechanism*, not another incremental test. See §6.

---

## 1. The goal chain

**Main goal.** The strongest 7 Wonders Duel player in the world.

**Sub-goal.** The engine must have an *exact* understanding of board structure —
the card pyramid's cover graph, whose turn it is, and who can therefore reach a
given card first.

**Why the sub-goal supports the main goal.** 7WD has two sudden-death win
conditions — 6 distinct science symbols, and the shared conflict pawn reaching
the opponent's capital (`abs(conflict_position) == 9`, `engine.py:477`; the pawn
is moved by the *net* shield difference, so this is a tug-of-war, not an
accumulation of 9 shields) — and Age III opens
with a choice of who moves first. A position like "Age III is starting, I choose
the order, I hold 5 science symbols" is decided by a question with an exact
answer: *given the visible pyramid and the removal order it forces, can my
opponent be prevented from taking the 6th symbol before me?* A human expert
answers this by counting. An engine that approximates it will, at some rate,
either walk into a loss it could have seen or decline a win it had.

Two sub-questions, and they are not the same problem:

* **(a) Determinate control.** Over the *face-up* pyramid, who reaches a target
  slot first under optimal play for that target, given turn alternation and
  extra-turn Wonders as a tempo budget?
* **(b) Information control.** Every removal from the pyramid *uncovers* new
  cards. Which removals minimise the chance of handing the opponent the card
  that beats me — and, symmetrically, maximise what I force them to reveal?

Sub-question (b) is not a static feature of the position at all. It is a
property of the action, and it is where our measured errors are largest (§3.2).

---

## 2. Why this is hard here, and what is already ruled out

**The exact endgame solver does not reach.** `seven_wonders_rust/src/solver.rs`
solves to terminal with full rules, is gated by an 86-position equivalence
corpus, and is admitted by a Tobit-fit cost model (held-out R² 0.94,
`endgame_cost_model.json`). It was used in the most recent training run. In
practice it solves in reasonable time only with **8–11 cards left**. Age III
opens with 20. So the positions in §1 are exactly the positions the solver
cannot reach. This is the gap that motivated everything below.

**The trunk already half-knows control.** A group-aware probe over 2200 examples
and 8 seeds (`control_probe_v3.json`) recovers the control solver's own outputs
from the incumbent's token representations at:

| quantity | R² from tokens | R² from pooled |
|---|---|---|
| `control_now` | 0.746 | 0.661 |
| `control_with_one_more_tempo` | 0.667 | 0.578 |
| `control_if_opponent_takes_theology` | 0.722 | 0.621 |
| `my_tempo` | 0.924 | 0.856 |
| `their_tempo` | 0.918 | 0.848 |

So the net is not blind to control; it is *fuzzy* about it. That matters for
interpreting the null in §4: an added input channel is competing with something
the trunk already reconstructs at R² ≈ 0.75, and the residual is the part that
decides sudden-death races.

---

## 3. Exact examples from recorded BGA games

All positions below are real, from human games captured by the advisor
(`runs/seven_wonders_duel/bga_game_log/`), replayed through our engine. All
"reference" numbers are deep-search estimates from the incumbent
(`candidate_0085.pt`) evaluating **every legal action** across sampled
face-down worlds — 600 sims per world — not solved values. They are sufficient
to expose a large error, not to prove a small one.

### 3.1 Example A — tempo decides a 30-point swing (table 904750590, row 24)

Age III, actor to move, tracked target **Observatory**. The actor's options
include taking `Tacticians Guild`, at the single slot coordinate `[6, 7]`
(`threat_corpus_scan.py:307` writes `"removes": list(slot_id)` — one
coordinate, not two slots). Removing it **uncovers two cards**, giving 10
equally likely worlds. `University` and `Observatory` are the two green
(science) cards in the unseen pool; both carry the ARMILLARY_SPHERE symbol.

The five actions that all take `Tacticians Guild` — same card, same slot, same
ten worlds:

| action | weighted win % | range across worlds |
|---|---|---|
| Wonder: Piraeus (using Tacticians Guild) | **57.9** | 53.8 – 68.4 |
| Wonder: The Sphinx (using Tacticians Guild) | 57.1 | 49.1 – 72.8 |
| Discard for coins: Tacticians Guild | 30.2 | **0.0** – 52.9 |
| Build: Tacticians Guild | 29.6 | **0.0** – 54.3 |
| Wonder: Circus Maximus (using Tacticians Guild) | 27.0 | **0.0** – 50.3 |

The bottom three collapse to ~0.02% in exactly the four worlds that reveal
`University` or `Observatory`, and sit at 44–54% in the other six. The top two
never collapse: their worst worlds are 53.8% (Piraeus) and 49.1% (Sphinx)
respectively — so the floor differs between them and 53.8% is not a common
bound.

The mechanism is exact and structural: `Piraeus` and `The Sphinx` carry
`PLAY_AGAIN` (`data.py:303,305`); `Circus Maximus` does not (`data.py:297`). The
extra turn lets the actor *take the science card it just uncovered*. Without it,
uncovering a science card hands the opponent the game.

The extra turn is not good in itself — it is good *because of what this removal
uncovers*. The same position's `Chamber of Commerce` options remove a
**different** card, one that uncovers nothing (every such action was priced in a
single deterministic world with an empty reveal), and there the ordering
inverts: `Build: Chamber of Commerce` 55.0%, but `Wonder: Piraeus (using Chamber
of Commerce)` only 43.3%. Tempo is worth ~28 points on the reveal-bearing card
and negative on the card that reveals nothing.

**This is a ~28-point ranking gap between two moves that take the same card,
decided entirely by tempo and cover topology.** It is precisely what an exact
control model is for, and precisely what a fuzzy R² = 0.75 reconstruction can be
expected to get wrong at the margin.

### 3.2 Example B — search misprices its own choice by 20 points (table 908370787, row 17)

Age II. The advisor's search (PUCT, closed mode) at 1000/2000/3000 sims ranks:

| rank | action | visits @1000 | prior | search's win % |
|---|---|---|---|---|
| 1 | Discard for coins: Caravansery | 936 | 0.734 | 69.9 |
| 2 | Wonder: Circus Maximus (using Caravansery) | 35 | 0.135 | 62.7 |
| 3 | Discard for coins: Aqueduct | 15 | 0.059 | 62.9 |

The deep reference on the same position says the ranking is inverted:

| action | reference win % |
|---|---|
| Wonder: Circus Maximus (using Caravansery) | **68.9** |
| Wonder: Circus Maximus (using Aqueduct) | 60.0 |
| Build: Aqueduct | 57.1 |
| Discard for coins: Aqueduct | 56.8 |
| Discard for coins: Caravansery | **49.3** |

Search plays the move worth 49.3% while reading it as 69.9% — **~20 points
optimistic on its own choice, ~19.6 points of regret** — and this is stable
across a 3× sim increase, so it is not a budget problem at this scale.

The bookkeeping cause is measured (`w9_reference/baseline_closed_loop.json`):
the refuting reply is **chance-independent** — it works whatever is revealed —
but the search splits it across 10 face-down worlds at 164.9 visits per world
on average, so the refutation is a prior-0.03 child of a ~165-visit node and
receives 172 visits *in total across all ten worlds*. It is top in 1 of 10
worlds when partitioned; in isolated single-world probes at 6000 sims it is top
in **5 of 6**. Promoting it in *any* world takes 2000 sims; in half of worlds,
never within the 3000 measured.

This is sub-question (b): the error is about what gets uncovered, and no static
encoder channel addresses it.

### 3.3 The corpus these came from

`runs/seven_wonders_duel/threat_corpus/episodes.json` — 267 episodes over 81
tables, one episode per physical (table, card, slot) run, filtered to positions
where an extra-turn Wonder is available:

| threat class | episodes |
|---|---|
| military_band | 146 |
| science_pair | 107 |
| science_win | 12 |
| military_win | 2 |

219 of 267 are at minimum distance 0 (the target is one removal away). This is
the natural evaluation set for any proposal here.

Across the 48-position reference pass, the largest per-action spread across
revealed worlds reaches **89.5 points** (table 905755009 row 34: the same action
is worth 3.0% in one world and 92.5% in another). Reveal choice is not a
second-order effect in Age III.

---

## 4. What we have tried, and the results

### 4.1 Built: an exact topology solver (Workstream 3)

`tableau_control.py` solves a two-player minimax on the removal poset by
memoized recursion: turn alternation, the cover graph, and extra-turn Wonders as
a one-extra-removal tempo budget. Public information only — it never reads a
face-down identity, so it is safe on a determinized state.

Deliberately narrow, and this is the crux: it models **no coins, production,
discounts, chains, military, or next-Age starter choice**. Its own docstring
forbids victory claims — outputs are named `can_take_first`,
`decisions_until_accessible`, never `forced_science_win_in_k` — and
`tableau_control.py:448` states plainly that "none of it asserts the builds are
affordable."

It emits 6 channels per tableau slot: `can_force` and `turns_s` under three maps
(`now`, `theology_mine`, `theology_theirs`).

Validated by a 10,000-game equivalence soak (`w3_equivalence_soak.json`):
716,212 rows, zero mismatched columns.

### 4.2 Tested: offline A/B, five seeds (2026-09-04 overnight)

`w3_offline_ab.py`. All arms warm-start from the same checkpoint
(`candidate_0085.pt`), train 400 steps at lr 5e-5 on the same replayed buffer
with the same game-honest split, and differ only in what the model is shown or
asked to predict. New input columns are **zero-initialised** by
`migrate_state_dict` (`train.py:616-638`), so at step 0 every arm computes what
the incumbent computed.

Paired per-seed deltas vs `baseline`, n = 5:

| arm | what it is | Δ policy top-1 | Δ value acc |
|---|---|---|---|
| `inputs` | solver outputs as encoder channels | +0.00034 ± 0.00028 | +0.00032 ± 0.00059 |
| `shuffled` | same channels, **permuted between positions** | +0.00037 ± 0.00027 | +0.00047 ± 0.00023 |
| `aux` | inputs OFF; auxiliary head predicts solver outputs | **−0.00202 ± 0.00174** | −0.00037 |

Baseline seed-to-seed sd on policy top-1 is 0.0015, so `inputs` and `shuffled`
are indistinguishable from each other and from zero. **The causal control moved
as much as the real information.** The `aux` arm is negative on 5/5 seeds and
raised policy CE by +0.0023 while adding 39% to training wall-clock.

**Three reasons this null is weaker than it looks**, all structural:

1. **The targets are control-blind.** The buffer came from a policy whose search
   never saw control features, so the arms were scored on predicting decisions
   made *without* the information. The harness says so in its own `note` field.
2. **The budget starts at exactly zero.** 400 steps at lr 5e-5 is the entire
   allowance for zero-initialised columns to become useful.
3. **The metric is an aggregate over average positions.** The §3 effects live in
   a small minority of Age III positions. A fix worth 30 points on 5% of
   positions is invisible in a 4000-game top-1 mean.

The `both` arm (inputs + head) is defined in the harness but was not run.

### 4.3 Tested: four search-side mechanisms for the reveal problem

Separately, four search mechanisms were tried against the §3.2 reference case
(chance-sibling bias with and without a positive-only clamp, wonder-group
selection, and combinations). None solved it. The conclusion recorded at the
time was that the correction belongs in *training*, not in more search
bookkeeping.

### 4.4 Not tried

* Any measurement with search *in the loop* on control features. The 20 arm
  checkpoints and the 48-position reference pass both exist on disk;
  `w3_corpus_regret.py` is written and would score each arm's *played* move
  against a common reference. It has not been run.
* Any throughput measurement of control features on vs off during self-play.
  The A/B trains from pre-encoded examples, so it cannot see the encoder cost.

---

## 5. The honest summary of where this leaves us

* Exact control information **exists** and is validated (§4.1).
* The net **already reconstructs it fuzzily**, R² ≈ 0.75 (§2).
* Handing it to the net as input channels is **indistinguishable from handing it
  scrambled** (§4.2). That test *could* have shown better prediction of its
  existing targets and did not; what it did not measure is better *decisions*
  during search.
* Asking the net to predict it as an auxiliary task is **mildly harmful** (§4.2).
* The errors we can actually demonstrate (§3) are large — 20 to 30 points — and
  at least one of them (§3.2) is not an information problem at all but a visit-
  allocation problem.
* The exact solver that *would* settle these positions doesn't reach them (§2).

There is a real risk of a testing spiral here: build, test, get "A but not B",
build another test, get inconclusive, repeat. That pattern is the reason this
document exists. What is wanted is a judgement about which *mechanism* is
right, before another instrument gets built.

---

## 6. Questions for the reviewer

**Q1 — Is "feature" the wrong delivery mechanism for an exact fact?** An
internal hypothesis: a proof handed to a network as a float channel gets
smeared. When our endgame solver proves something we do not feed it in; we clamp
the node value and mask losing moves. Should exact control be a *search
terminal / move mask* rather than an encoder input — and does that hypothesis
also explain why §4.2 read flat regardless of how good the information is?

**Q2 — Is proof-number search the right instrument for the Age III gap?** Full
expectimax prices every action to terminal, which is far more than "do I have a
forced win?" asks. That question is boolean: one winning child settles an OR
node, one refutation settles an AND node. Three things look favourable in 7WD:
the science/military win conditions truncate the tree the moment they fire (no
scoring, no playout to the end of Age III); most moves are irrelevant to the
proposition and may be collapsible; and a *guarantee* under hidden information
fails fast, since one bad determinization kills it, where expectimax must weight
all of them. There is no proof-number or AND/OR search anywhere in this repo.
Is this the standard answer, and what is the realistic reach — does it plausibly
cover 20 cards, or only extend 11 to 14?

**Q3 — Or should the abstraction be widened instead?** For the 5-symbols case,
`can_force` on a slot bearing the missing symbol is *nearly* a forced win
already, because the 6th symbol ends the game immediately. The gaps are
affordability and the opponent's own clock (their fastest science or military
win). Adding those two would turn a topology fact into a sufficient condition
for a win — cheap relative to Q2, and it converts something already built. Is
this the better first move, or does the affordability closure drag the whole
economy back in and cost the exactness?

**Q4 — What actually fixes sub-question (b)?** §3.2 is a chance-independent
refutation starved by partitioning across 10 face-down worlds. Four search
mechanisms failed on it. Is the right answer at the *node* level (some form of
sibling sharing or afterstate merging across chance outcomes that we got wrong
four times), at the *target* level (train the value head on the reference's
answer for these positions), or is there a formulation from POMDP /
information-set search that makes "which removal reveals least to my opponent" a
first-class quantity rather than something search has to rediscover per world?

**Q5 — Given the null, include the W3 encoder channels or not?** The measured
facts: no benefit, no measured harm, unmeasured self-play encoder cost, and it
bumps the encoder signature so every existing checkpoint needs a migrating warm
start. The counter-argument for including is that it is the best mechanism we
have thought of for the net to learn this *and use it during search*, and the
offline test could not falsify that. Is "include on a no-harm basis" defensible,
or is it the kind of accumulated speculative complexity that makes later
attribution impossible?

**Q6 — Sequencing.** If a reviewer had one build to spend before the next
multi-hour training run, which of Q2 / Q3 / Q4 is it, and what is the cheapest
measurement that would tell us it worked — ideally one that reads out on the
267-episode threat corpus rather than on aggregate top-1?

---

## 7. Pointers

| thing | where |
|---|---|
| control solver | `games/seven_wonders_duel/tableau_control.py` |
| control encoder channels | `encoder.py:246-268` (`_CONTROL_MAPS`, `CONTROL_FEATURES`) |
| auxiliary control head | `net.py:335-362`; loss at `train.py:116-133` |
| offline A/B harness | `w3_offline_ab.py`; results `runs/seven_wonders_duel/w3_offline_ab.json` |
| zero-init warm start | `train.py:575-656` (`migrate_state_dict`) |
| redundancy probe | `runs/seven_wonders_duel/threat_corpus/control_probe_v3.json` |
| threat corpus | `runs/seven_wonders_duel/threat_corpus/episodes.json` (267 episodes) |
| reference pass (48 positions) | `runs/seven_wonders_duel/threat_corpus/w3_reference/` |
| Example A | `w3_reference/cbf1861ab036_r24.json` |
| Example B | `runs/seven_wonders_duel/w9_reference/baseline_closed_loop.json`, `sufficiency_common_ply.json` |
| unrun regret harness | `w3_corpus_regret.py` |
| exact endgame solver | `seven_wonders_rust/src/solver.rs`; `advisor_endgame.py`; plan in `SOLVER_SELF_PLAY_PLAN.md` |
| equivalence soak | `runs/seven_wonders_duel/w3_equivalence_soak.json` |

---

## 8. Review outcome (2026-09-05)

External reviewer read the request, inspected the implementation and saved
results, ran no tests and changed no files. Three factual corrections were
raised and all three verified against the code; §1 and §3.1 above are corrected
accordingly (slot coordinate semantics, Sphinx floor, military win condition).

**Verdict: Q4.** Improve the training targets at reveal decisions and the
opponent replies search misses. Keep the W3 encoder channels **off** for the
next run. Use verified tactical wins in search, but do not treat topology
outputs as victory proofs.

### Decisions taken

| question | decision |
|---|---|
| Q1 | A *proven game outcome* bypasses the net (terminate node / mask proven-losing action, with sane behaviour when all actions lose). Topology control, failed proofs and unresolved continuations get **no** clamp and **no** mask. |
| Q2 | Proof-number search stays on the shelf. It finds guarantees; it does not price a move whose one bad reveal is survivable. Reach is a function of proof size, not card count — "20 vs 14" was an unfounded framing on our side. |
| Q3 | Widening the abstraction is worth doing only as a *sufficient-condition certifier that may answer "unknown"*, reusing the real rules engine for certified continuations with topology as move ordering. Not as a small patch that completes the game. |
| Q4 | **Selected.** Targeted reanalysis producing corrected policy *and* value targets. |
| Q5 | Channels stay disabled by default. Keep the implementation as an option. Auxiliary head: no. |
| Q6 | One build: reanalysis for reveal decisions and their refuting replies. |

### Corrections to our reasoning, accepted

* **The smearing hypothesis is not established.** §4.2 shows that *this* setup
  extracted no advantage from these channels. Limited adaptation, redundant
  representation, unsuitable targets and an insensitive metric all remain live
  explanations. Do not restate "nets smear exact facts" as a finding.
* **Ignoring affordability helps *both* players**, so the topology result is not
  automatically a conservative bound on the real game. This kills the intuition
  that W3's answer is "safe because it is pessimistic".
* **"Cannot force this science card" ≠ "cannot win".** Any certifier must keep
  *no forced win within horizon*, *no forced win*, and *losing* distinct.
* **Four failed sharing mechanisms do not prove training is the only fix** — but
  they do weaken the case for spending the next build on a fifth selection bonus.
* **Example B's 49.3% is the value of the discarded-Caravansery *action*, not the
  root's best-play value.** Training the root value head on 49.3% would teach a
  falsehood when another action is worth 68.9%. This is the specific trap in the
  build below.

### What "reanalysis" means here

Revisit recorded positions with **deliberately allocated** search effort, so a
low prior cannot stop a candidate reply from getting enough visits to expose its
consequence. Re-running the same PUCT allocation for longer reproduces the
failure — §3.2 is stable across a 3× sim increase.

Training examples must include all three of:

1. the original reveal decision, with corrected action preferences;
2. the resulting opponent decision, with the overlooked reply promoted;
3. successor positions whose values correct the optimistic continuation.

Policy correction makes the reply discoverable; successor-value correction makes
its consequences visible earlier. Root value is *not* set to the refuted
action's value.

### Evaluation plan for that build

1. Split **by table before generating examples**, keeping related snapshots and
   reveal variants together.
2. Small paired training comparison, then measure the move actually chosen on
   held-out referenced positions.
3. Report **reference-relative regret at equal search time**, weighted toward
   large mistakes — this is what `w3_corpus_regret.py` already computes.
4. Separately record whether overlooked opponent replies now receive useful
   attention, and whether optimism on the chosen action falls.

Scope limit to respect: only the 48 positions with references support numerical
regret claims. The 267 episodes give coverage by threat class and distance, not
ground truth. Note also that just **14 of 267** episodes are immediate
science/military win threats — the other 253 are ordinary strength, which is
why Q4 was preferred over the sudden-death-only instruments.

Showing the new targets expose the missed tactics is cheap and comes first.
Showing the model *learned* them needs the training-and-search comparison.
Neither alone demonstrates overall playing strength.

### Open sources cited by the reviewer

* Proof-number search survey — https://dke.maastrichtuniversity.nl/m.winands/documents/ICGA2012PNS.pdf
* Cowling, Powley, Whitehouse, information-set MCTS (strategy fusion) — https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf
* Silver & Veness, POMCP — https://papers.nips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf
* Anthony, Tian, Barber, Expert Iteration — https://arxiv.org/abs/1705.08439

---

## 9. Second review round (2026-09-05) — supersedes §8 on Q5 and Q6

The reviewer revised, after the goal was restated as *maximise winning chances*
and the Age II→III strategic chain was made explicit.

**Revised verdict: continue W3 and extend it into a targeted tactical evaluator
used during ordinary self-play. Keep the encoder channels ON for the next run.
Do not build a general Age III solver.**

### What changed and why

Round 1 rejected W3 as an input partly because its output is not a proof. The
reviewer withdraws that weighting: **omitting costs does not make W3 unsuitable
as an input, it makes its output *conditional*, and a network combining
conditional information with coins, affordability, Wonders and science threats
is exactly the intended use.** The proof standard belongs to §9's requirement 2,
not to requirement 1.

The intended learning, stated by the reviewer, is the sentence W3 exists to make
learnable: *"spending this Wonder improves my position now, but preserving it —
and the money to build it — protects an otherwise favourable game against
science."*

### The three requirements of the next build

1. **Expose control through the existing W3 channels** during ordinary self-play
   training, so the net can combine it with economic context. Explicitly *a
   hypothesis carried forward*, not a claim the channels have shown benefit.
2. **Certify narrow, decisive control situations during search.** When a visible
   card would complete science or deliver military supremacy, use W3 to propose
   a candidate forcing strategy, then verify constructions, costs, Wonder
   retirement and opposing winning replies against the **real rules engine**. A
   completed certificate supplies an exact value; an incomplete check leaves the
   net's estimate alone. Checking one principal line is *not* a certificate.
   Trigger on threat presence, **including inside the tree** — "start of Age III"
   is too narrow, the same consequence must be caught several moves later or
   inside a considered continuation.
3. **Feed the corrected consequences into ordinary training** as proven values
   and supported action preferences, keeping a bounded number of certified
   positions including ones the improved policy learns to avoid.

Requirement 2 is the small solver component, and is best understood as
**extending W3 from structural control to verified tactical consequence.** Its
cost and coverage are unestablished.

### The milestone

**Make the Observatory sequence (§9.1) affect search immediately; then get the
corrected result into ordinary self-play training alongside the W3 features.**

### 9.1 The reference case for that milestone — MEASURED 2026-09-05

BGA table `907773062` (RollwJoel vs Alexmp1), target `Observatory` — the card
giving the opponent a sixth distinct science symbol. Corpus episode
`e65eef1222ac` (rows 85–87).

**The published ladder does not reproduce as written, and the row index in the
older note is off.** Measured with `candidate_0085.pt` on CPU, one persistent
tree per row, via
`w9_reference_case --table 907773062 --decision-row R --tracked Observatory
--stages ladder --ladder 800,2000,5000 --no-verify-position --allow-migration`:

| decision row | 800 sims | 2000 | 5000 | extra-turn Wonder available |
|---|---|---|---|---|
| 84 | +0.454 / 72.7% | +0.404 / 70.2% | +0.332 / 66.6% | yes — Sphinx, two ways |
| 85 | **+0.1575 / 57.87%** | +0.105 / 55.3% | +0.136 / 56.8% | yes — Sphinx, one way |
| 86 | +0.070 / 53.5% | −0.183 / 40.8% | **−0.435 / 28.2%** | **none** |

Published (`SCIENCE_BLOCKING_AND_WONDER_TEMPO_REVIEW.md:120-127`): 800 → +0.158
/ 57.9%, 2,006 → −0.107 / 44.6%, 5,053 → −0.382 / 30.9%, then 25,365 → 8.4%,
100,738 → 3.2%, 250,000 → 1.8%.

* Row **85**'s 800-sim rung matches the published 800 rung to three decimals
  (+0.1575 vs +0.158; 57.87% vs 57.9%). That pins the note's "captured record
  84" to **decision row 85**.
* But row 85 stays *positive* deeper, while row **86** reproduces the published
  trajectory's shape — positive at 800, negative by 2000, ≈ −0.43 at 5000 —
  without matching its values.
* No single row reproduces the whole published ladder. Most plausible cause,
  **unverified**: the searcher has changed since that note (`force_expand_root_chance`,
  chance-sibling and wonder-group work), so the prior-dominated 800-sim read
  survives while deeper behaviour diverges.

**The structural claim survives, on row 86.** Row 86 is the first row with **no
Wonder action legal at all** — the Sphinx is gone — which is exactly W3's
"control flips when no unbuilt extra-turn Wonder remains". The model reads
53.5% there at 800 sims in a position worth 28.2% by 5000.

**Sharpens the build:** at row 86 the model already prices `Build: Senate` at
1.25% win, so it is *not* blind to the visible sixth-symbol threat. The error is
in the continuation it actually plays — `Build: Gardens`, 54.8% at 800 sims and
28.5% at 5000. The certifier must correct the surviving line's value, which is a
harder target than "spot the winning card".

**Pass/fail bar for step 2, revised:**

* **Positive case — decision row 86:** 800 sims must stop reading 53.5%.
* **Negative control — decision row 85:** same threat, extra-turn Wonder still
  in hand. The certifier must return `unknown` (or a materially milder verdict)
  rather than firing. This asymmetry is the hypothesis.

Raw reports: `<scratchpad>/rec84_ladder.json`, `rec_r85.json`, `rec_r86.json`.

### 9.2 Corrections to costs previously listed as open

The self-play encode cost of the W3 channels is **not** a live solve. Rust
`control.rs` packs (age, present-slot mask, who-moves, five tempo values) into a
u64 key (`control_key_word`) and returns three precomputed `&'static [u8]` maps
from an installed table (`control_maps`, `control.rs:312`). The per-position cost
is a mask build plus a lookup; the ~180 ms figure in
`W3_CONTROL_REVIEW_REQUEST.md` is the *offline table build* for a fresh Age III,
not a per-encode cost. §4.4's "unmeasured throughput cost" is therefore mostly
closed. A missing table panics at first encode by design rather than emitting
zeros.

### 9.3 What still binds from round 1

* The auxiliary control head stays dead (−0.0020 policy top-1, 5/5 seeds).
* The §4.2 null is still a null. Requirement 1 rides on judgment, and the next
  run therefore **cannot attribute** any gain to W3 if channels, certifier and
  corrected targets all ship together. If attribution matters, sequence them.
* Ignoring affordability helps *both* players, so topology is still not a
  conservative bound — which is why requirement 2 verifies against the real
  engine rather than trusting W3's answer.
* "No forced win in horizon" ≠ "no forced win" ≠ "losing".
* Example B's 49.3% is an *action* value, not a root value.
* One realized Age III layout cannot establish that the earlier (Age II)
  Wonder expenditure was wrong; earlier decisions need weighting across deals.

### 9.4 The open risk the reviewer names

Better tactical evaluation only teaches the earlier strategy if these
consequences enter searches or training examples **often enough**. That is a
coverage question, not evidence that another feature or a different solver is
needed.

---

## 10. Step 2 built and probed (2026-09-05)

`control_certify.py` — a three-valued AND/OR proof that a seat cannot escape a
sudden-death defeat, executed entirely by `engine.apply_action` (real costs,
chains, Wonder retirement), chance enumerated not sampled, clones barred.
`control_certify_probe.py` runs it on recorded BGA decision rows.

`PROVEN` / `REFUTED` / `UNKNOWN`, and `REFUTED` means only *no sudden death
within the horizon* — never "not losing". `UNKNOWN` can never be upgraded to
`PROVEN` by an unexamined branch, which is what makes an incomplete search safe.

### Reference case, table 907773062, max_plies 8

| row | extra-turn Wonder | gate | verdict | nodes | seconds |
|---|---|---|---|---|---|
| 84 | Sphinx, two ways | none (opp on 4 symbols) | UNKNOWN (plies) | 5,229 | 21.0 |
| 85 | Sphinx, one way | science_missing 1 | UNKNOWN (plies) | 4,045 | 20.2 |
| **86** | **none** | science_missing 1 | **PROVEN** | 3,373 | 10.0 |
| 87 | none | science_missing 1 | **PROVEN** | 1,827 | 6.8 |

The asymmetry the hypothesis predicted: the proof succeeds exactly where the
extra-turn Wonder is gone. Row 86 is the row whose 800-sim search reads 53.5%.

### Falsification checks, all passed

* Same rows, `--loser 1` (the *other* seat): REFUTED in 59 and 27 nodes. The
  certifier is directional, not a machine that prints PROVEN.
* Age 1/2 rows 70/74/78/80: no spurious PROVEN. Two returned UNKNOWN on
  `age_deal` (correct — a deal is not enumerable), two on the ply cap.

### Cost, measured — this is the problem

* **~340 nodes/second.** The PROVEN at row 86 took 10.0 s; ply-capped UNKNOWNs
  took 15–21 s.
* **The cheap gate fires on 22.6% of play-age rows** (45 of 199, five tables:
  907773062, 904750590, 905755009, 906802058, 905634974).

22.6% of nodes at ~10 s each is three orders of magnitude too slow to call
inside search. Step 3 therefore has a head start and a clear target: the options
are a tighter gate, memoisation across the tree, a Rust port, or accepting that
this runs only at the root / in reanalysis rather than at every node.

### Profile (step 3): the cost is `clone`, not the rules

cProfile on row 86 (41.5 s under the profiler vs 10.0 s without, so read the
shares, not the absolutes):

| | cumulative | share |
|---|---|---|
| `game.clone()` -> `copy.deepcopy` | 31.5 s | **77%** |
| `engine.apply_action` | 6.2 s | 15% |
| `legal_actions` incl. `minimum_payment` | 5.5 s | 13% |
| `chance_signature` | 1.2 s | 3% |

12,763 clones cost **17.2 M** `deepcopy` calls. The rules engine is a sixth of
the bill and chance enumeration is negligible -- an earlier guess that chance
was the blow-up was wrong.

So the first optimisation is **not** a Rust port. It is make/unmake with a
journal -- the pattern the Rust solver already uses and audits
(`engine::journal_undo_audit`) -- or a hand-written `GameState.clone` that
shares the immutable parts instead of deep-copying card-name strings and frozen
dataclasses. Either plausibly buys 3-5x in Python, which reaches "root and
reanalysis" but still not "every search node".

### Row 85 deep run: UNRESOLVED

`--max-plies 14 --max-secs 1800`: **UNKNOWN**, 282,798 nodes, the full 1800 s,
~157 nodes/s (slower than row 86's ~340 because deeper nodes carry more chance
children). Seat 0 holds `The Sphinx` unbuilt, seat 1 `The Pyramids`.

So the negative control is **not established**. Whether the extra-turn Wonder
actually saves the game, or the loss was already forced a ply before W3 says,
needs the make/unmake rewrite first. This is the single open question that
decides whether the trigger fires early enough.

**A reporting defect found and fixed while reading this result.** `Certificate`
recorded only the *first* budget reason, so the run reported `stopped_by:
"plies"` when it had also run out the clock -- and "the horizon was too short"
and "the machine is too slow" call for opposite fixes. `Certificate` now carries
`limits_hit` (every limit that fired anywhere in the tree) alongside
`stopped_by` (the one that ended the search). Same class as the reporting
artifact that produced the earlier denial-rescore null.

---

## 11. Reveal-risk feature, built and tested (2026-09-05)

`reveal_risk.py`: five per-slot channels appended after `CONTROL_FEATURES` --
`reveal_n` (hidden slots this removal uncovers, by the last-coverer test) and
`reveal_{my,opp}_{sixth,mil}` (`reveal_n` x the fraction of the unseen pool that
would give that seat a sixth science symbol / immediate military supremacy).
O(unseen pool + slots): no search, no joint enumeration. Reuses the encoder's
own face-up card tests so a card is judged the same whether visible or not.

Built because W3's control channels are **chance-invariant by construction** --
`control_key_word` carries no card identities -- so they return an identical
answer in all ten reveal-worlds of a position whose value ranges 0.02%-54%.

**It computes the right thing.** On Example A (table 904750590 row 24), the only
nonzero slot is (6, 7) `Tacticians Guild`: `reveal_n = 2`,
`reveal_opp_sixth = 0.429` (opponent on five symbols, 3 of 14 unseen cards carry
the missing one). Every other slot is zero.

### Result: null, paired against its own shuffle, 5 seeds

| metric | reveal - baseline | shuffled - baseline | **reveal - shuffled** |
|---|---|---|---|
| policy_top1 | +0.00003 | +0.00016 | **-0.00013** +/- 0.00067 |
| value_acc | -0.00014 | +0.00002 | **-0.00016** +/- 0.00057 |
| policy CE | -0.00001 | +0.00006 | -0.00007 |

3 of 5 seeds negative on top-1; baseline seed-sd is 0.0013.

### Why the harness could not have detected it

The decisive channels are nonzero on **0.80% of tableau tokens** (27 of 3,381)
and **6.4% of positions** (20 of 312), measured over eight tables already biased
toward threats. Scaling the paired difference by the affected fraction, this
experiment resolves only per-position effects larger than **~2 points of top-1
on affected positions**.

So the null is a **power** statement, not a verdict on the feature. And it now
applies to three arms from the same instrument -- `inputs`, `shuffled`,
`reveal` -- including two features with opposite structure (chance-invariant vs
chance-aware). **Stop using aggregate top-1 on buffer-average positions to test
narrow features.** The right instrument is per-position regret restricted to
positions where the feature is nonzero: `w3_corpus_regret.py`, which exists and
has still never been run.

### Shipping state

* `SWD_REVEAL_FEATURES` defaults **off**. With it on, `test_both_languages_
  agree_in_off_mode` fails: the Rust encoder has no reveal block, and self-play
  encodes in Rust. This cannot ship to self-play until Rust computes it too.
  **Superseded by §13 (2026-09-06): Rust computes it, and the two languages
  agree bit-for-bit with the channels live.**
* The digest test now strips control **and** reveal, since the claim under test
  is "removing everything W3 added reproduces the pre-W3 digest".
* All 11 `test_control_encoder.py` tests pass.

### A harness bug that produced a convincing fake null

`w3_offline_ab` cached encoded examples keyed on the **control** flag alone, so
the `reveal` arm reused baseline-encoded examples and returned a result
identical to baseline to four decimals. Nothing about it looked wrong. Now keyed
on `(control_on, reveal_on)`. Any future arm that changes the encoding must
extend that key -- this failure is silent and looks exactly like a measurement.

---

## 12. Certifier cost, step 3 (2026-09-06)

§10 predicted make/unmake or a sharing clone would buy 3-5x and named `clone`
as 77% of the bill. `fast_clone.py` delivered the sharing clone, and the
prediction was **wrong about the size**: on row 86 it moved 10.0 s to 8.3 s
wall, about 1.2x. Deep-copying was the largest single line in the profile, but
removing it exposed that the rules engine underneath was most of the rest.

### What the profile said once clone was gone

| | share |
|---|---|
| `minimum_payment` (42,026 calls) | 46% |
| `is_accessible` (**1.18 M** calls) | 26% |
| `covering_slots` (635 k calls) | 14% |
| `chance_signature` + `observation` | 11% |

Three fixes, none of which changes a single result:

1. **`game.COVERING_SLOT_IDS`** — which slots cover which is printed on the
   board, so `is_accessible` reads a table built at import instead of scanning
   the layout and rebuilding a tuple on every one of 1.18 M calls.
2. **`engine.PricingContext`** — the four city scans a price needs
   (`_fixed_production`, `_choice_producers`, `_opponent_trade_production`,
   `_trade_discounts`) were rebuilt inside every `minimum_payment`. They are now
   built once per `legal_actions` call, in one pass over each city rather than
   four. `minimum_payment` still builds its own when a caller passes none, so
   every existing call site is unchanged.
3. **`legal_actions` prices each Wonder once**, not once per accessible slot: a
   Wonder's price does not depend on which card is spent to build it, and six
   slots by three Wonders bought the same three prices eighteen times. Plus two
   shortcuts inside `minimum_payment` — a coin-only cost has exactly one
   payment, and a candidate that buys nothing is minimal on both comparison
   keys, so the assignment loop can stop at it.

### Measured, on CPU time

Wall clock on this laptop varies 2.9-6.9 s for identical work (same node count,
same verdict), so single wall-clock readings below ~10 s cannot be compared.
These are best-of-three `time.process_time`.

| | before | after | |
|---|---|---|---|
| row 86 | 4.11 s, 821 nodes/s | **2.72 s, 1241 nodes/s** | 1.51x |
| row 87 | 4.53 s, 403 nodes/s | **1.94 s, 943 nodes/s** | 2.34x |

Node counts are identical before and after (3,373 and 1,827) — the proofs are
the same proofs, reached faster.

### Verified, not assumed

* 1,829,370 prices — every card and every Wonder, for both seats, at every
  state of 150 random games — identical to the original `minimum_payment` body.
* `legal_actions` output identical, element for element, across 8,607 states:
  order is part of the codec, so a reordering would silently rewrite action
  indices.
* Engine, game, rules, codec, full-game, bots, buffer and Rust-equivalence
  suites pass unchanged.

### The trap this hit, worth knowing before touching the engine again

`control_table.rule_identity` hashes the **bytes** of `data.py` and
`tableau_control.py`. The topology table therefore could not live in `data.py`
where it belongs by subject: adding a comment there invalidates the W3 control
table and every checkpoint that records its digest. Worse, `core.autocrlf` is
`true` in this checkout while the index holds LF, so a plain `git checkout --
data.py` rewrites it with CRLF and changes the digest **with no source change at
all**. That happened during this work; restoring LF restored identity
`4265ccd51617`. A digest over normalised content would end the hazard, but
changing `rule_identity` itself invalidates the table, so it needs a
regeneration to go with it.

### The next lever, not taken

`apply_action` validates with `if action not in legal_actions(game)`, so every
application rebuilds and linearly scans the whole legal-action list: 12,762 of
the 16,135 `legal_actions` calls in this proof are that check, roughly 40% of
what remains. The certifier decodes its actions **from** `legal_action_indices`
on the same state, so it already knows they are legal. Skipping the re-check
needs an explicit opt-in keyword on `apply_action`, which trades a guard the
whole engine leans on for speed — a judgment call, deliberately left to a human.

---

## 13. Reveal risk in Rust (2026-09-06)

§11's shipping blocker is gone: `seven_wonders_rust/src/reveal.rs` computes the
same five channels, so self-play (which encodes in Rust) can be shown them.

Both languages now agree **bit-for-bit with the channels live**, over a whole
random game, in `test_both_languages_agree_with_reveal_on`. Off-mode agreement
was never enough on its own -- it proves only that both sides emit zeros -- so
the test also asserts the channels were nonzero somewhere, or the comparison is
agreement about zeros with extra steps.

Ported faithfully rather than reimplemented: `rel_position` and
`effective_shields` are the encoder's own helpers, made `pub(crate)` and passed
in, so a revealed card is judged by the same rule as a face-up one in both
languages. The unseen pool is narrower than `obtainable_cards` -- cards already
face up on the board cannot be revealed by anything.

### A latent two-language disagreement, found on the way

`encoder.py` reads `SWD_CONTROL_FEATURES` for its default; `control.rs` assumed
`true`. Exporting `SWD_CONTROL_FEATURES=0` therefore turned the control channels
off in Python and left them on in Rust -- the replay path and the self-play path
shown different inputs, with nothing reporting it. Both flags now read their
variable, with Python's parsing, so the two agree without anyone remembering to
call the setter. `SWD_REVEAL_FEATURES` was built that way from the start.

### Cost

+8% on the Rust derive path with the channels on (0.359 s -> 0.391 s cpu for
2,868 rows, best of 15). That is the encode-only path; in self-play the network
forward dominates, so the end-to-end share is smaller.

### Test state

Eight red tests are now none. `ENCODER_VERSION` is `7wd-encoder-7` and the
signature and goldens are re-pinned, deliberately and with the evidence
recorded in `test_encoder_signature_is_pinned` -- including one check that only
a correct widening passes: the DRAFT golden is byte-identical across the bump,
because a draft observation carries no tableau tokens. A change there would
have meant the new channels leaked into token types they have no business in.

The earlier state, for the record: eight red tests were three once Rust
computed the block. `test_rust_engine_equiv` (33) and
`test_f4_boundary` (20) are green. Unrelated and pre-existing: `cargo test` fails
`tests::encoder_feature_counts_match_schema`, which encodes a PlayAge state
without installing the control table and hits the deliberate panic. It fails
identically with every change here stashed.

---

## 14. Row 85 is not a gate (2026-09-06)

§10 left "the row-85 negative control" as the single open question. It is now
closed, by measurement rather than by a longer run, and the answer is that the
question as §10 posed it cannot be answered by compute at all.

### The branching, measured

| | legal actions | chance children **per action** | face-down slots |
|---|---|---|---|
| row 86 | 4 | **1** | 5 |
| row 85 | 3 | **90** | 7 |

Each of row 85's actions uncovers two hidden slots against a 10-card Age III +
Guild unseen pool, so every action forks into 10x9 = 90 ordered worlds, and
every later uncovering ply multiplies again: 90, 8,100, 729,000. This -- not
the interpreter, not `deepcopy` -- is why row 86 proves in 3,373 nodes and row
85 does not prove in 400,000. **A Rust port would not settle row 85**: 20-50x
against a 90x per-ply fan-out buys less than one extra ply.

It also explains the cost asymmetry that had been read as being about the
Wonder: row 86 is cheap because nothing is revealed there at all.

### The two directions have opposite costs

* **PROVEN** needs ONE forcing move that survives every reply and every
  outcome. Short, and cheap when the forcing line uncovers nothing.
* **REFUTED** needs one defence after which the winner has NO forced win, and
  establishing "no forced win" is exhaustive over every winner option in every
  chance world. Combinatorially out of reach in Age III.

Only PROVEN reaches the mechanism -- §10 already says failed proofs and
unresolved continuations get no clamp and no mask -- so the intractable
direction is one the certifier never needs in search. What it blocks is the
negative control as posed.

### And the control had already passed

`control_certify_probe`'s own criterion is "does it decline (REFUTED or
UNKNOWN) rather than fire?". Row 85 returned UNKNOWN on the first run, which is
a decline. Two questions had been merged: the control (wants a decline; passed)
and earliness (wants a PROVEN at row 85, meaning the loss was already forced a
ply earlier). They want opposite verdicts, and only the second is open. It is
also the cheap direction, so it was not unreasonable to chase -- it simply has
not succeeded against a 90x first-ply fan-out.

### Transposition: measured and rejected

The remaining §10 option. It does not pay:

| | visits | distinct positions | a sound TT would skip |
|---|---|---|---|
| row 86, plies 8 | 12,763 | 12,315 | **3.5%** |
| row 85, plies 14 | 1,012,713 | 949,525 | **6.2%** |

1.04 and 1.07 visits per position: each chance outcome carries its own deck, so
positions essentially never coincide. (Sound means keyed on (state, remaining
plies): PROVEN at a shorter horizon carries to a longer one, REFUTED does not.)

### The reporting trap, again, one layer out

The row-85 runs were read as "died on the node cap, so it needs more nodes".
That was wrong. At plies 14, **87.9% of the recursion hits the ply horizon**
(439,798 of 500,253 calls) and `limits_hit` is `nodes` AND `plies`. The console
line printed `stopped_by` alone -- the binding limit -- and `limits_hit` goes
only to the JSON, which is not written without `--out`.

`_Budget` records every limit precisely because "the horizon was too short" and
"the machine was too slow" call for opposite fixes; the display layer then threw
one away and reproduced the trap the class was written to prevent. The probe now
prints both. Row 86's PROVEN also truncated branches on the horizon -- its proof
just did not need them.

Consequence: a `--max-plies 4` probe cannot help. A shorter horizon truncates
MORE branches, so it can reach UNKNOWN faster but can never produce the REFUTED
that would settle the Wonder question.

### What to do instead

Stop treating row 85 as a gate; fan-out and horizon each explain its UNKNOWN on
their own, so it has never been evidence about the extra-turn Wonder. The open
question worth answering is §9.4's coverage one, and it is cheap: across the
267-episode corpus, how often does a *certifiable* position occur, and how many
plies before the blunder? That decides whether the certifier changes training at
all.

If the Wonder question itself matters, the tractable route is symmetry
reduction, not compute: of those 10 unseen cards exactly ONE carries a science
symbol, and the proof reads only a revealed card's symbol, shields, colour and
cost. Cards identical on what the rules read are interchangeable within the
horizon, collapsing 90 worlds to a handful of classes -- soundly, if the
equivalence is proved against what is actually read. That is a real build, and
it must buy depth as well as width.
