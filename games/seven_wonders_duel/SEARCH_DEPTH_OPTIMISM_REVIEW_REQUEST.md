# Review request: shallow search overrates the side NOT to move

**For:** a third-party reviewer, to brainstorm how the network could learn this
rather than have search compensate for it.
**Written:** 2026-09-27. **Branch:** `sevenwd-w9-prototype`.
**Cross-reference:** `WORLD_CLASS_MODEL_EVOLUTION_PLAN.md` (below: "the plan") --
its reference case, Workstreams 9-11, and *Correct confidently wrong priors
deliberately*.

---

## 1. The problem in one paragraph

When our search spends a few hundred simulations on a position where the
**opponent is to move**, it rates that position **9-19 win-percentage points too
good for us** compared with a 3,200-simulation search of the same position. The
error is largest after moves that turn cards face up, because the search splits a
move's budget across every possible revealed card (5-11 branches), so each branch
is searched shallowly. In two real losses to BGA's ZeusAI the advisor's top move
was overrated by up to **25 points**, and ZeusAI's decisive replies were moves our
search had barely explored. The player followed the advisor at almost every
decision; the move-choice cost of the human's own play was ~0 in both games.

The plan already found this failure in one position (table `908370787`) and
tried four search mechanisms against it, all null (§5). The new evidence here
says the defect is **not specific to killer reveals and not a funding accident in
one branch**: it is a uniform, depth-dependent optimism about positions where the
opponent has the move. That points at what the network learns, which is what we
want ideas on.

---

## 2. Setup the numbers come from

| item | value |
|---|---|
| network | `run07_iter35.pt` -- run07 iteration 35 (W1 slots, W2 graph, W3 control, W4 hierarchical value, W5 action residual at alpha ~0.67), warm-started from `candidate_0085.pt` |
| search | the advisor's Rust PUCT search, `leaf_batch` 16, laptop RTX 3070 GPU, fp32 evaluator (the advisor's default precision) |
| games | BGA tables `922514551` (loss) and `922535304` (loss), human vs ZeusAI, human = first player |
| "truth" | a 3,200-simulation search. **Not solved values** -- a deeper reference, and where it matters we say so |
| logs | `runs/seven_wonders_duel/bga_game_log/table_<id>.jsonl` (gitignored; local only) |

Tools (all committed, `019d38c` and earlier):

* `bga_replay.py` -- rebuilds **both** players' decision positions from the
  advisor's per-turn snapshots plus BGA's own move packets (anchored replay,
  hidden cards corrected to what was later seen face up). 67 and 72 decisions;
  every consecutive pair of rebuilt positions agrees exactly. Only the opponent's
  opening move of Ages II and III is missing (the new deal isn't in the packets).
* `bga_review.py` -- searches every decision (3,200 sims) and splits each swing in
  our win probability into three parts:
  * **move choice** -- Q(best) minus Q(played), from the mover's side;
  * **search overrating** -- the search's Q for the played move minus a direct
    3,200-sim evaluation of the positions it actually leads to (averaged over the
    possible reveals);
  * **reveal luck** -- the actual reveal's value minus the mean over 8
    alternative reveals.
  Age-closing moves (whose "after" contains a guessed next deal) are excluded
  from all totals; moves with under 100 visits are excluded from the move-choice
  and overrating totals (the two that read the move's tree value). The first
  version applied the visit filter to move choice only -- corrected after review;
  the §3.1 totals predate that correction.

---

## 3. Evidence

### 3.1 Whole-game decomposition

| game | human move-choice cost | reveal luck on human moves | search overrating of human moves |
|---|---|---|---|
| `922514551` | **-0.02** | +0.05 | **+0.31** (summed over the game) |
| `922535304` | **+0.07** | **-0.30** | **+0.36** |

The human's play cost nothing by the advisor's own standard. Game 2 was also
unlucky (-30 points of reveal luck, mostly one card). In both games the search
systematically thought the human's moves were better than the positions they led
to.

### 3.2 Example A -- Port, game `922514551`, first move of Age III (decision #48)

| step | human win probability |
|---|---|
| search at the decision (3,200 sims): Q of `Build: Port` | 0.72 |
| positions Port leads to, direct 3,200-sim evaluation, mean over 9 reveals | **0.47** -> overrated by **25 points** |
| the reveal that happened (University) | 0.43 -> luck only **-4** |
| after ZeusAI's actual reply `Build: Obelisk` | 0.42 -> the reply was what search expected |

A 12,800-sim search of the same decision gave Port **0.59** and tied it with
`The Appian Way (using Port)`. More search removes about half of the error.

History: an earlier pass of this review attributed -12 points to luck here. That
came from a replay bug (a card uncovered and taken between two snapshots was
corrected only from the moment it was taken), now fixed. Do not quote -12.

### 3.3 Example B -- Brickyard, game `922535304`, Age II (decision #43)

`Build: Brickyard` uncovered the `Laboratory`; ZeusAI took it immediately for a
science pair and the `Agriculture` token. Human win probability 0.58 -> 0.31, of
which **-23 points is reveal luck** (mean over the 8 alternative reveals 0.53).
Included as the counterexample: this swing is genuinely luck, and the method
separates it cleanly from the overrating cases.

### 3.4 Example C -- ZeusAI's decisive moves were ones our search had dismissed

| game | ZeusAI move | our search's view at ZeusAI's decision | what happened |
|---|---|---|---|
| `922514551` #60 | `Circus Maximus (using Armory)` | 57 visits; Q said it was worse for ZeusAI than its best alternative | human 0.27 -> **0.12** |
| `922514551` #52 | `Build: Academy` | 12 visits; Q said it handed the human **0.83** | human 0.43 -> 0.51 (a small gift, not 0.83) |

Unexplored opponent moves carry badly wrong Q values. This is the plan's
low-prior loop (*Correct confidently wrong priors deliberately*) observed in play
against an external engine.

### 3.5 The decisive test -- is the bias concentrated in a few reveal branches?

The natural remedy for "splitting across reveals dilutes the reply search" is to
screen every reveal and then deepen selectively (or, ZeusAI-style, sample a few
reveals and widen with visits). That only helps if the error sits in a few
branches. So, for each human move with a **single** reveal and overrating >= 8
points, every possible revealed card was enumerated and its branch valued twice:

* **shallow** -- a search at the budget the tree actually gave that branch
  (the move's visits divided by the number of branches); this reproduces the tree
  (Horse Breeders: shallow mean 0.88 vs tree Q 0.86; Port G2: 0.27 vs 0.27);
* **deep** -- 3,200 simulations.

| move | branches | sims/branch | mean gap (shallow - deep) | sd across branches | share of gap in worst 2 (even = 2/n) |
|---|---|---|---|---|---|
| G1 #61 `Build: Arena` | 5 | ~340 | **+0.19** | 0.03 | 47% (even 40%) |
| G2 #36 `Discard: Horse Breeders` | 8 | ~182 | **+0.13** | 0.03 | 29% (even 25%) |
| G2 #51 `Build: Port` | 10 | ~308 | **+0.09** | 0.02 | 24% (even 20%) |

Per-branch detail, G2 #36 (gap per revealed card): School +0.16, Brewery +0.16
(actual), Aqueduct +0.15, Forum +0.14, Glass-Blower +0.14, Shelf Quarry +0.15,
Courthouse +0.11, **Laboratory +0.07**. The killer card is, if anything, the
*least* overrated branch.

**Every branch is overrated by about the same amount.** No one or two reveals
carry the error. Selective deepening would not have fixed these moves.

Not covered: game 1's Port (#48) and game 2's Customs House (#39) were **double**
reveals and were excluded from this enumeration.

### 3.6 What the evidence says, taken together

1. The error is a function of **search depth in positions where the opponent is
   to move**: ~180-340 simulations leave them 9-19 points too good for us, and
   3,200 largely corrects it. Reveals matter only because they cut the depth per
   branch by a factor of 5-11.
2. It is **not** a killer-reveal problem, and **not** one under-funded branch.
3. It is consistent with the plan's reference case, where the raw value of the
   post-burial position read **78.4%** and 200 simulations brought it to
   **41.6%** (Workstream 10), and with Workstream 10's measured reply-node
   discovery threshold of **165-400 visits**.
4. The plan's Workstream 9 trace concluded "a prior failure, not a value
   failure" for its one position, because eight visits of search corrected the
   refutation's Q. The new evidence broadens the picture: the gap persists at
   **~300 simulations per branch** across ordinary positions, so whatever the
   mechanism at the refutation edge, the value that search backs up from a
   shallow opponent-to-move subtree is itself optimistic. The plan's own amended
   text already says value targets are "second and still wanted"; this makes the
   value side at least co-equal.

---

## 4. What we have NOT established (please weigh these)

* **Positions with no reveal.** The bias should appear there too, smaller
  because the branch gets the move's whole budget. Not yet measured. If it does
  not appear, the story is reveal-specific after all.
* **Side-to-move or side-of-the-logger?** Every branch measured is
  "human just moved, ZeusAI to move". We have not measured the mirror case
  ("ZeusAI just moved, human to move"). If shallow search is optimistic for
  *whoever just moved*, it is a tempo/side-to-move bias in the value head; if it
  is always optimistic for the same seat, it is something else (e.g. the
  first-player seat -- the human was first player in both games).
* **Reference quality.** 3,200 simulations is a deeper search, not the solved
  value. The direction is robust (12,800 sims moved Port further down, 0.70 ->
  0.59), the exact magnitudes are not.
* **One network.** All numbers are `run07_iter35`. The plan's reference case was
  `candidate_0085`. Same direction, different nets, but not a controlled
  comparison.
* **Two games, one opponent.** ZeusAI may exploit this more than typical
  opponents do.
* **The run itself.** run07 is promoting (iteration 20 beat `candidate_0085`
  57.8% over 600 games) with a positive self-anchor (0.585 at iteration 30). The
  bias does not stop self-play improvement; it may cap it against a stronger
  outside opponent, which self-play never supplies.

---

## 5. What has already been tried (from the plan)

All four search-side mechanisms were built and measured on the reference case
(table `908370787`, frozen `candidate_0085`):

| mechanism | plan section | result |
|---|---|---|
| chance-sibling action bias | W9 mechanism 1 | refutation funded 37-58% more, never promoted in half the worlds, recommendation unchanged; the shared statistic is seeded from the same wrong values, so a strong bonus freezes the error |
| Wonder-action factorization | W9 mechanism 2 | a correctness property at interior nodes only; cannot reach Gumbel root policy targets |
| approximate afterstate clustering | W10 | sound only where the game discards the revealed identity (Wonder burial); runtime licensing unsolved; **NULL** |
| public tactical leaf extension | W11 | recovers 7 of 37 points (19%) of the leaf error; the value does not live in a short forced line; **NULL** |

Also relevant and NOT a fix for this: **chance capping** (`CHANCE_ENUMERATION_PLAN.md`)
keeps `n x 3` stratified pairs for *double* reveals, as a throughput measure; it
is on in run07 and does not touch single reveals.

**Proposed in the plan but never executed:** the targeted training correction
("*the training correction is the lever these four mechanisms are not*"). A
corpus of actor-created-threat episodes exists (267 episodes, plan §*The
actor-created-threat corpus*); no network has been trained on corrections from it.

**What an outside engine does** (ZeusAI paper): afterstates capped at 11 children
(the maximum for one revealed card -- so single reveals are fully expanded, as
ours are), widened with visits at play time; 1k simulations in self-play with
visit-proportional move choice, **5k deterministic** at play time. ZeusAI has the
same dilution in principle; its visible advantage is play-time depth.

---

## 6. Candidate directions (for the brainstorm)

Grouped by where the fix lives. Ordered within each group by our current guess at
promise; we would value the reviewer re-ranking them.

### 6.1 Teach the value head what depth finds

1. **Deep-search value targets on opponent-to-move positions.** Extend the S2b
   reanalysis (already coalesced: 45 ms/position) to re-search a sample of
   post-move positions at high simulation counts and train the value head on
   those values, not only on game outcomes. The measured gap (9-19 points between
   ~300 and 3,200 sims) is exactly the signal to distil. Open questions: which
   positions (all, post-reveal, high-disagreement?), how deep, how to mix with
   outcome targets without over-trusting a search that is itself biased.
2. **Target the disagreement directly.** Select positions where shallow and deep
   search disagree most (the gap is cheap to measure) and weight them in the value
   loss -- hard-example mining for the value head.
3. **n-step / TD(lambda) value targets bootstrapped from deeper search**, so that
   a position's target reflects the opponent's best reply rather than an average
   over self-play continuations.
4. **Side-to-move calibration.** If §4's mirror measurement shows the bias is
   "whoever just moved looks too good", test whether it is a calibration issue a
   small correction (or data balancing) removes, before any architecture change.

### 6.2 Teach the policy the opponent's resources

5. **The plan's targeted prior correction, finally executed.** Forced examination
   of every legal reply on correction positions; counterfactual ranking targets;
   the 267-episode corpus as seed data. §3.4's Circus Maximus (57 visits) and
   Academy (12 visits) are fresh examples of the low-prior loop.
6. **Opponent-reply auxiliary head.** Predict, for the side to move, the size of
   its best available swing (max over actions of Q minus the position's value).
   A head trained to notice "the opponent has a big move here" gives the value
   and the search an explicit signal the flat value head averages away.
7. **Train against stronger, different opponents.** Self-play never faces a
   player that punishes our blind spot. Options: BGA-seeded self-play starts
   from logged positions, games against a ZeusAI-like engine if one is
   available, or specialists (W7) whose objective is to find replies our search
   under-rates.

### 6.3 Search, only where it complements learning

8. **Play-time depth.** The cheapest mitigation, and ZeusAI's actual edge: more
   simulations before committing on reveal moves (the advisor already streams; it
   could refuse to show a confident ranking until the per-branch depth passes
   ~500).
9. **Screen-then-deepen at chance nodes** (one evaluation per reveal, then
   visits proportional to probability x value uncertainty -- Neyman allocation).
   Not blind to killer reveals. Our §3.5 data says it would **not** fix the
   uniform bias, so it is listed for completeness and for double reveals, where
   branching is 90-way rather than 10.
10. **Pessimistic or variance-aware backup** at shallow opponent-to-move nodes
    (e.g. shrink low-visit child Q toward a learned prior of the gap). A
    search-side patch for a learned bias; useful only if 6.1 is slow.

### Things we would especially like the reviewer to consider

* Is "shallow search is optimistic for the side not to move" a known phenomenon
  in AlphaZero-style value learning, and what fixed it elsewhere?
* Given self-play only ever meets its own blind spots, is any purely self-play
  target (6.1) enough, or does this need an outside or adversarial opponent (6.2
  item 7)?
* How to keep a deep-search value target from inheriting the same bias at a
  smaller scale.

---

## 7. Reproducing the numbers

From `C:\Users\joeld\projects\boardgame-ai-7wd` with the shared `.venv`:

```powershell
# full review of one game (~20 min on the laptop GPU)
.venv\Scripts\python.exe -m games.seven_wonders_duel.bga_review `
    runs\seven_wonders_duel\bga_game_log\table_922514551.jsonl `
    --checkpoint extension_7wd\run07_iter35.pt --sims 3200 --worlds 8 --output review.json
```

The per-branch enumeration of §3.5 was a one-off script over `bga_replay` +
`bga_review.Searcher`: for each single-reveal human move with overrating >= 0.08,
enumerate the hidden pool for the revealed slot, force each card there
(`bga_replay._force_card`), apply the move, and search the result at
`visits_move // n_branches` and at 3,200 simulations. Worth promoting to a tool if
the reviewer wants the mirror and no-reveal measurements from §4.
