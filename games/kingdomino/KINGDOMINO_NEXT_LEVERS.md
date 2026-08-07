# Kingdomino: what is left to try

- **Status:** Forward-looking notes. Nothing here is committed work.
- **Date:** 2026-08-07
- **Current model:** `runs/kingdomino/best_checkpoint/current_best.pt`
  (sha `4bf07b0c…`, 80x6), placed **3rd in a three-month BGA arena**.
- **Purpose:** capture credible remaining levers and, more importantly, the
  reasons the obvious ones are already closed — so a future session does not
  re-run a null.

## 1. What is already closed, and how firmly

Five independent nulls. This matters more than any idea below: the cheap and
obvious directions have been tried.

| Attempt | Result |
|---|---|
| run9 — diversity package | null |
| run10 — pick-group visit floors (depth 1-2) | null |
| run10 — spite personalities in the HOF pool | null |
| run11a — exploiter loop (PSRO-lite) | null: **locally unexploitable at equal capacity** |
| 2026-08 — tile action-value head (M0-M2.5) | closed, see `AZ_TILE_Q_HEAD_PLAN.md` |

The run11a result is the strongest: an exploiter warm-started as a clone of the
banked net, trained specifically to beat it, plateaued at ~48.5% over ~15,000
games with no trend across ~8,700 training steps. It never learned to want the
squeeze position it was pointed at.

**Read the qualifier carefully: "at equal capacity."** Every null above was run
at 80x6. That is the loophole none of them tested.

Independent external evidence that the policy is already strong: the BGA anchor
(606 clean opponent decisions, 36 games, top-30 opponents) found the net's top
pick matched the human's **76.4%** of the time, top-2 **95.5%**, with only
**3.6%** of human picks being moves the net rated below 5%.

## 2. Recommended first: two measurements that can close directions

Both are cheap, neither trains anything, and each can end a line of work
outright. Given that the last arc spent months on a lever an early measurement
would have closed, measurements that can say "stop" come first.

### 2.1 Paired-seat flip rate — how much does luck decide?

**Question.** Is the variance in this game large enough that no achievable skill
edge produces a dominant win rate? If so, "superhuman" needs a different
yardstick than win rate.

**Design.** The opponent choice is the whole experiment. Mirror play (net vs
itself) cannot answer it — with identical players skill decides nothing, so you
would only measure the seat advantage. Instead use **the same net at different
simulation budgets**: `current_best` at 10,000 sims versus the same net at 800.
Same weights, same style, a real and tunable skill gap with no confound.

Play each deck twice with seats swapped, then classify:

- stronger side wins both orientations -> **skill** decided that deck
- the same seat wins both -> the **deck/seat** decided it
- split -> noise

Sweep the ratio (10k vs 6400 / 3200 / 1600 / 800) to get a curve: how large must
a skill edge be before it survives the variance?

**What it buys.** Answers the ceiling question, and calibrates the sample size
for every future gate — something this programme has repeatedly underestimated.

**Caveat.** Humans differ in style from the net, so this bounds the variance
floor rather than predicting arena results exactly.

**Cost.** A few hundred paired games. `round_robin_eval.py` already supports
paired deck seeds.

### 2.2 Placement headroom audit — is there score left on the table?

**Why placement is special in 2p.** The boards are disjoint: placements never
interact with the opponent. Placement is therefore a **pure single-agent
optimisation**, not a game-theoretic problem — the one part of this game where
something close to ground truth is computable.

It is also the least-searched dimension in the stack. Even the 8-ply reference
search delegates placement to the policy's top-1 (top-2 for the opponent),
`denial_search.py:560`. And in this variant placement drives most of the raw
score.

**Design.** This is a measurement decomposition, NOT a proposal to unbundle the
joint `(placement, pick)` action — that action space is correct for deciding.

For one player in a completed game:

1. Take their **ordered sequence of claimed dominoes** as fixed.
2. Replay it changing only the placements: each domino placed legally, in the
   order it arrived, on an incrementally growing board.
3. Beam-search the best achievable final score (a wide beam gives a strong lower
   bound on the optimum, which is sufficient).
4. **Gap = optimum - actual** = points lost purely to placement.

Respecting the arrival order is what makes this a real measurement rather than a
fantasy upper bound — you cannot rearrange retroactively.

**The pick/placement coupling makes this conservative, not invalid.** The player
chose tiles that suited their own placement plan; a better placer might have
chosen different tiles and scored higher still. So the measured gap is a **lower
bound** on the total value of better placement. Forced discards (a domino with
no legal placement) are counted automatically.

**The number that matters is relative.** Run the identical audit on the top-30
humans in the BGA logs (moves are reconstructable from consecutive-state
deltas):

- model gap **~=** human gap -> equally hard for everyone; no relative edge,
  though absolute gains still help against the field
- model gap **>** human gap -> the model places worse than its opponents;
  direct headroom, quantified in points
- model gap **<** human gap -> the model already out-places them; look elsewhere

**Cost.** Pure CPU combinatorial search — no GPU, no training, no network
anywhere in the arbiter. Roughly 24 dominoes per player and a few dozen legal
placements each; seconds per game with a wide beam.

**Do this one first.** Unlike the flip-rate study it can find a concrete number
of points, not just calibrate future experiments.

## 3. Other credible levers

Ranked by my estimate of expected value, with the case against each.

**Capacity.** Every null was at 80x6, and run11a's unexploitability was
explicitly *at equal capacity*. The recorded pivot order already says "rerun
capacity bake-off on a squeeze-containing buffer before any 80x10 talk."
*Against:* run5's verdict was **data exhaustion**, so more capacity on the same
data may only overfit. This is why the bake-off is conditioned on better data
rather than run standalone.

**Endgame exactness.** The solver is exact at deck <= 4
(`endgame_solver.py`). Pushing the frontier to deck <= 8 would make the final
rounds provably optimal, and endgame errors translate directly into final score.
Its great virtue is that it is **verifiable against ground truth** — no proxy
metric, no gate to mis-specify. *Against:* the exact region is small, so the
win may be small too; measure the frontier cost before committing.

**BGA-seeded self-play.** Number one in the recorded pivot order and still
undone. Rationale from run11a: mirror self-play never *generates* the
interesting positions, so no search fix can find them; starting episodes from
real strong-human positions broadens the state distribution. *Against:* needs
Rust start-from-state, and the seed set is thin (~360 reliable states logged,
possibly more in the newer tables).

**Distributional score head.** Third in the recorded pivot order. Never
attempted; no evidence either way.

**Ship.** Fourth in the recorded pivot order, and the record calls an
"exploiter-certified plateau a defensible stopping point." Third place in a
three-month arena against elite opponents is consistent with a genuinely strong
player in a game with real variance. This is not a failure state.

## 4. Levers I would NOT revisit without new evidence

- **Pick-side denial / secondary-pick sharpening.** Closed after M0-M2.5 (see
  `AZ_TILE_Q_HEAD_PLAN.md`). The premise held — searched values really do
  disagree with the raw value head, monotonically in policy rank — but every
  attempt to convert that into a better *decision* failed.
- **Pick-group visit floors as a shipped setting.** Measured a wash at 10,000
  sims (fixes 3 positions, breaks 3). They remain useful as a **label-generation
  device** at low sim counts, not as an inference setting.
- **More offline analysis against the 8-ply forced reference.** Its own
  reliability is unverifiable on the frozen 50: the exact solver needs deck <= 4
  and that set's minimum deck is 8, so it cannot check its own judge.

## 5. Methodological lessons that should govern future work

These cost real time to learn and are the most transferable thing here.

1. **Measure the objective, not a proxy for it.** The last arc repeatedly
   measured *calibration* when the objective was *selection*. Raw child value
   cut MAE 3.2x versus a zero constant while scoring identically on top-1 tile
   choice, because the decision gaps (median 0.033 between best and second-best
   tile) sit well inside the estimator's error.

2. **Every gate needs a control cohort.** An absolute magnitude on a treated
   group cannot show the effect is specific to it. This was gotten wrong twice.

3. **Compare paired quantities with paired statistics.** Subtracting one arm's
   marginal confidence bound from another's point estimate bounds nothing. This
   error was identified, fixed, and then rebuilt two milestones later — it
   produced an invalid "proceed" verdict.

4. **Cluster by the unit that generated the data.** Positions from one game are
   correlated; treating them as independent understates every interval.

5. **Check the baseline is not trivially achievable.** A "25% better than raw"
   gate turned out to be three-quarters clearable by a single constant per rank.

6. **Prefer arbiters with no network in them.** Everything scored against the
   8-ply reference inherits that reference's value head. The exact solver, real
   game outcomes, and the placement optimiser do not.

## 6. Reference numbers

Measured 2026-08-06/07 unless noted, on `current_best` (sha `4bf07b0c…`).

| Quantity | Value |
|---|---|
| Policy top-1 agreement with 8-ply reference (held-out games) | 69.0% |
| Policy pairwise ordering accuracy | 82.7% |
| Policy mean searched-Q regret | 0.0076 |
| Median searched-Q gap, best vs second-best tile | 0.033 |
| Secondary tile visit share @3,200 / @10,000 sims | 2.19% / 1.16% |
| Best-prior placement == searched-best placement | 42.0% |
| BGA anchor: net top pick == top-30 human's pick | 76.4% (top-2: 95.5%) |
| run11a exploiter plateau vs banked net | ~48.5% over 15,000 games |

Related: `AZ_TILE_Q_HEAD_PLAN.md` (closed experiment),
`SECONDARY_PICK_FRAGILITY_FINDINGS.md`, `RUN10_PLAN.md`.
