# Kingdomino: next levers for world-best play

- **Status:** Forward plan; experiments require their own frozen configs and
  preregistered gates before expensive training.
- **Date:** 2026-08-07
- **Current model:** `runs/kingdomino/best_checkpoint/current_best.pt`
  (sha `4bf07b0c…`, 80x6), placed **3rd in a three-month BGA arena**.
- **Goal:** make the strongest Kingdomino player in the world, not merely find
  another proxy metric that moves.
- **Primary workstreams:**
  1. chance-correct, observation-split enhancement of open-loop search;
  2. high-value BGA positions as restart states in the training loop.

## 1. Executive decision

Do not give up on Kingdomino enhancement. The evidence closes the specific
tile-Q/child-seeding lever, not the game. The most credible remaining weakness
is now the interaction between the state distribution and stochastic search:

1. Ordinary self-play rarely reaches the elite-human and failure positions in
   which the remaining mistakes live.
2. The current open-loop tree correctly hides future deck order, but aliases
   different public states after a row reveal. This can suppress the value of
   being able to adapt to the row once it becomes visible.

The two workstreams reinforce one another. BGA restarts supply strategically
interesting roots; chance-correct search supplies better decisions and training
targets at those roots.

The design must preserve the lesson that motivated open-loop search in the
first place: **a decision made before a reveal must not depend on that reveal.**
The intended change is not a return to a determinized, known-deck tree.

## 2. What is already closed, and how firmly

| Attempt | Result |
|---|---|
| run9 — diversity package | null |
| run10 — pick-group visit floors (depth 1-2) | null |
| run10 — spite personalities in the HOF pool | null |
| run11a — exploiter loop (PSRO-lite) | null: **locally unexploitable at equal capacity** |
| 2026-08 — tile action-value head (M0-M2.5) | closed; see `AZ_TILE_Q_HEAD_PLAN.md` on `kingdomino-tile-q` |

The tile-Q programme found real rank-conditioned disagreement with its searched
reference, but did not convert it into better selection:

- the small head overfit 2,868 labels;
- its best checkpoint was no better than rank constants;
- deployable child-value selection lost to the incumbent policy;
- best-prior placement agreed with searched-best placement only 42.0%; and
- the root-probe's nominal proceed result was caused by a near-copy arm escaping
  a weak futility criterion, not by a positive point estimate.

Do not revive that programme without qualitatively new data or a new arbiter.

Capacity is closed on the current training distribution. The recorded bake-off
in `RUN10_PLAN.md` found 80x6 matched 96x6 and 80x10, so another larger-net arm
on the same buffer is not useful background work. Reopen capacity only after a
search or restart treatment has materially changed the data distribution, and
then require a new bake-off rather than assuming the larger net will help.

Independent evidence that the policy is already strong: the BGA anchor (606
clean opponent decisions, 36 games, top-30 opponents) found the net's top pick
matched the human's 76.4% of the time and its top two contained the human pick
95.5% of the time. The remaining gain is therefore likely concentrated, not a
broad failure of basic play.

## 3. Objective and correctness prerequisites

### 3.1 Authoritative promotion objective

BGA is long-term external validation, not the experiment-calibration metric.
The advisor runs continuously while a BGA move is pending and the top move is
played after it stabilizes. A different placement—still within the advisor's
top three—is chosen less than 1% of the time. Execution fidelity is therefore
already high enough that UI/human overrides are not a primary model-strength
lever.

The three-month Arena finish (3rd) and current overall rank (31st) are useful
real-world outcomes, but there is no defensible conversion between BGA rating
and the development loop. The authoritative development objective is the same
promotion rule used by 7 Wonders Duel:

> Promote a candidate only when the lower confidence bound on paired match score
> against `current_best` exceeds 50%.

- Use identical deck seeds with seats swapped and treat each two-game deck/seat
  pair as the statistical cluster.
- Score a win as 1, an official draw as 0.5 and a loss as 0.
- Hold every setting except the isolated treatment fixed and match
  simulations/NN evaluations. For checkpoint promotion, both players use the
  same deployed search; a frozen-net search experiment may differ only in the
  search treatment being tested.
- Reuse the 7WD confidence level and valid fixed-sample or sequential stopping
  procedure; do not repeatedly inspect an ordinary interval or extend an
  inconclusive match after seeing its result.
- If candidates or settings are repeatedly selected on the normal gate, reserve
  an unseen final seed set to limit evaluation overfitting.

An inconclusive result means no promotion, not proof of inferiority. The large
game cost of `LCB > 50%` is accepted because false promotions are more expensive
at this stage than evaluation games. A paired-seat variance curve is still
useful for forecasting how many games/GPU-hours a 50.5%, 51% or 52% candidate
will require; it does not set a separate practical-effect promotion threshold.
The existing mean searched-Q regret (0.0076) and median tile gap (0.033) remain
diagnostics, not ceilings on possible playing-strength gain.

### 3.2 Repair terminal outcome consistency

**Implemented 2026-08-07.** Terminal search now optimizes the official outcome
consistently in Python MCTS, Rust closed/open-loop MCTS and exact endgame search.

- `game.py:199` remains the Python source of truth: score, largest connected
  territory, total crowns, then a true draw.
- Python and Rust terminal backups use that cascade for the decisive outcome
  component while retaining the actual raw score difference for the margin
  component. A tiebreak win therefore does not invent a fractional score margin.
- Exact alpha-beta uses a solver-only scalar: ordinary integer raw margins, with
  `-0.25 / 0 / +0.25` ordering a score-tied tiebreak loss, true draw and
  tiebreak win. Conversion back to the training value restores zero raw margin.
- The separable deck-0 solver maximizes each board's full cascade key rather than
  total score alone.
- Batched results now export `(score0, score1, official_outcome0)`, so Elo,
  promotion and batched round-robin matches no longer record every raw-score tie
  as a draw. The serial Rust self-play label path was corrected at the same time.

A deterministic 600-game random-legal audit found 17 equal-raw-score games
(2.83%): 16 were decided by the official tiebreak (2.67% of all games) and only
one was a true draw (0.17%). This is a reach diagnostic, not an estimate for
elite model play. A model-play terminal-leaf counter would still refine the
search-backup exposure rate, but is not required for correctness.

Verification covers raw-score wins, territory and crown tiebreaks, true draws,
Rust/Python cascade parity, exact-solver parity and the exported batched outcome.

## 4. Workstream A — chance-correct enhancement of open-loop search

### 4.1 What open loop fixed

The earlier deterministic-deck tree let search and its value targets exploit
future tile order. The network learned early-game values conditional on future
reveals and consequently undervalued flexibility. Open-loop redeterminization
fixed the information leak by making pre-reveal actions average across futures.

That property is non-negotiable:

- the encoder sees remaining bag membership, never hidden order;
- a pre-reveal decision cannot condition on the future row; and
- changing hidden deck order while holding the public state fixed cannot change
  direct network inference.

### 4.2 The remaining open-loop blind spot

The current stateless tree is keyed by action sequence. At depth, slot-relative
actions and their statistics can be shared by materially different public
states reached after different row reveals. Consequences include:

- post-reveal policies are partly forced to share action-rank statistics across
  rows that require different choices;
- cached priors can come from the first determinization reaching a node;
- legal children can differ across determinizations; and
- stale nodes can terminate descent early and fall back to leaf evaluation.

This is directly relevant to Kingdomino tempo. Let:

- `x` be a decision made before the next row is known;
- `R` be the revealed row; and
- `y` be the next decision made after observing `R`.

The correct public-information value is:

```text
max_x E_R [ max_y Q(x, R, y) ]
```

Two incorrect extremes are:

```text
E_R [ max_x max_y Q(x, R, y) ]   # clairvoyant determinization
max_x max_y E_R [ Q(x, R, y) ]   # post-reveal action aliasing
```

The first lets the early commitment depend on hidden information. The second
can prevent the later action from adapting to newly public information. The
target architecture must implement the middle expression exactly in meaning,
even when chance outcomes are sampled approximately.

### 4.3 Target tree structure

Use alternating decision, afterstate, chance, and observation-conditioned
decision nodes:

```text
public decision state
    -> action x
    -> afterstate (row still unknown)
    -> chance node draws row R
    -> distinct public decision node for R
    -> action y chosen with R visible
```

- PUCT or Gumbel selects only at player decision nodes.
- Chance outcomes are sampled from the true distribution and backed up by
  expectation, never maximized.
- Decision/transposition keys contain only public state: boards, visible row,
  claims, phase, discards and remaining bag membership.
- Hidden permutation order is never in a node key, network input, cache key or
  training record.

This follows the action-observation history principle in POMCP and the
afterstate/chance split in Stochastic MuZero. Kingdomino has a perfect simulator,
so only the search topology is relevant; no learned dynamics model is needed.

### 4.4 Controlled random reveals

At a chance node with `n` remaining tiles, generate `X` independent random
permutations of the public bag and partition each permutation into four-tile
rows. This produces:

```text
K = n * X / 4 sampled rows
```

Keep the notation explicit: `X` is the per-tile exposure count and `K` is the
number of row children. Every tile appears `X` times in one search panel, not
`2X`; the revealed row is shared by both players, and a seat-swapped evaluation
game is a separate search rather than extra support for the first one.

Properties when `n` is divisible by four:

- every tile appears exactly `X` times;
- every individual four-tile block is marginally a uniform draw from all
  `C(n, 4)` rows;
- the equal-weight panel mean is therefore an unbiased estimator of the chance
  expectation; and
- rows within a permutation cycle are dependent, so uncertainty must cluster
  by cycle rather than pretend all `K` rows are IID.

In two-player Kingdomino, a pre-reveal bag must have `n % 4 == 0`. Assert this
at every chance node and fail loudly if it is violated; do not silently truncate
a permutation or invent weights for an invalid state.

Examples:

| Bag | X=1 | X=2 | X=4 |
|---:|---:|---:|---:|
| 44 | 11 rows | 22 rows | 44 rows |
| 24 | 6 rows | 12 rows | 24 rows |
| 8 | 2 rows | 4 rows | 8 rows |

Tile balance controls first-order inclusion, not pair/crown/terrain
interactions. Compare it directly with the same number of IID uniform rows.
Do not add a pair-balanced design unless deck-8 ground truth shows first-order
balance is insufficient; a greedy constrained sampler can silently bias the
row distribution.

Use common random numbers: when candidate root actions leave the same hidden
bag, evaluate all of them against the same sampled row panel. This reduces the
variance of action differences. Keep chance probability separate from child
search visits: spending more downstream simulations on a row must not make that
row more probable.

The actual self-play game still receives one ordinary uniform reveal. Balanced
panels are counterfactual search outcomes, not a modified game rule.

### 4.5 Staged experiment

#### A-1 — contingency-headroom audit

**Initial measurement completed 2026-08-08; A-1 shows actionable headroom.** The existing
Python denial-search graph now has an opt-in `contingency_audit` pass. At its
single interior reveal it computes both `E_R[max_y Q]` and `max_y E_R[Q]` for:

- a fixed slot-relative pick rank with placement still optimized; and
- a fixed full joint action index (placement plus pick rank).

Both counterfactuals are propagated back through the pre-reveal tree, so the
artifact records event-level gaps as well as backed root top-1 changes and
adaptive regret. Actions must be legal in every sampled row to enter the
aliased arm; rows retain their distinct public nodes in the adaptive arm.

The same probe now supports `chance_sampling=balanced`: `X` independently
shuffled public bags are partitioned into four-tile rows, making every tile
appear exactly `X` times. Each row stores its permutation-cycle ID and reported
gap uncertainty clusters by that cycle. `X=1` therefore reports unavailable
sampling uncertainty rather than false zero uncertainty. IID/enumerated
sampling remains the default, and the new audit is default-off, so existing
denial-search label generation is unchanged.

The companion `iid_exposure` arm derives `K=nX/4` independently sampled rows
from each position's public bag. It therefore matches the balanced arm's support
width at every deck size without hand-maintained per-deck `K` settings.

Focused tests cover exact tile exposure, hidden-bag-order invariance, marginal
row support over many seeds, invalid bag sizes, actor-frame min/max behavior,
common-action legality, the adaptive-versus-aliased calculation and an
integrated balanced one-reveal graph. The focused denial-search suite has 25
passing tests; the combined related suite has 42 passing tests.

Probe flags are `--contingency-audit --chance-sampling
{iid_exposure,balanced} --chance-exposure X`. Use the same checkpoint, position
manifest and base seed for paired arms; write each arm to a distinct artifact.

The initial screen used checkpoint `4bf07b0c...`, 128 search simulations and a
frozen 50-position corpus stratified across public bag sizes 8, 12, 16, 20, 24
and 28. This is smaller than the intended 100-300-position confirmation corpus
and does not yet include the BGA failure suite. It is sufficient for deciding
whether to build the one-reveal hybrid, not for selecting a shipped exposure
schedule or claiming playing-strength improvement.

| Arm | positions/hour | pick-rank root changes | full-joint root changes | mean pick gap | mean joint gap |
|---|---:|---:|---:|---:|---:|
| balanced `X=1` | 86.2 | 5/50 | 6/50 | 0.01363 | 0.01793 |
| matched IID `X=1` | 94.8 | 5/50 | 6/50 | 0.01382 | 0.01732 |
| balanced `X=2` | 49.6 | 2/50 | 5/50 | 0.01785 | 0.02218 |
| matched IID `X=2` | 50.0 | 4/50 | 4/50 | 0.01915 | 0.02386 |
| balanced `X=4` | 25.0 | 4/50 | 6/50 | 0.02017 | 0.02546 |
| matched IID `X=4` | 25.2 | 5/50 | 7/50 | 0.02106 | 0.02731 |

At `X=4`, balanced and matched IID agree on 49/50 adaptive root choices and
48/50 pick-aliased choices. Position 48 (bag 28) changes the backed root pick
from 4 to 17 in all six arms, with `X=4` adaptive regret 0.0168-0.0187. Position
43 (bag 24) changes 8 to 9 in five of six pick-rank arms and all six full-joint
arms, with `X=4` regret 0.0208-0.0368. Position 25 (bag 24) also agrees in both
`X=4` arms, although its lower-exposure results are less stable. These repeated
root changes reject the hypothesis that observation aliasing is operationally
irrelevant.

The verdict is therefore **A-1 positive: build A1**. This is evidence of
decision headroom under the frozen network, not evidence that the hybrid wins
games. A2 still requires equal-compute selection/regret and paired game tests.
Do not run a blind full-corpus `X=8` doubling: the balanced adaptive root is
49/50 stable from `X=2` to `X=4`, while the remaining disagreement is
concentrated in bag-8 cases that can be adjudicated by exhaustive chance
enumeration. Use exact bag-8 oracles and repeated independently seeded panels to
set deck-specific exposure.

Treat full-joint results as secondary in this scaffold. Placement delegation
does not expand every joint action in every revealed row, so the common-action
intersection shrinks as support grows (4,624, 4,530 and 4,392 observations for
balanced `X=1,2,4`). Pick rank remains common across rows and is the cleaner
primary A-1 measure. The production A1 selection test must evaluate its actual
placement-and-pick topology rather than inherit this intersection artifact.

Measure whether observation splitting has enough decision value before building
new production search topology. Reuse `denial_search.py` as the broad probe
scaffold: it already has public-state keys, common sampled chance rows, batched
evaluation, weighted backup and one interior reveal. Its current rows are IID,
so add tile-balanced panels as a probe arm rather than treating the module as
the final implementation.

For each pre-reveal action `x` and common future-row panel, estimate both:

```text
adaptive(x) = E_R [ max_y Q(x, R, y) ]
aliased(x)  = max_y E_R [ Q(x, R, y) ]
contingency gap(x) = adaptive(x) - aliased(x)
```

Report the gap and its clustered interval for:

- pick-only choices;
- full joint placement-and-pick actions;
- the opponent's first response after the reveal; and
- backed root rankings, including top-1 changes and regret.

Track concrete tile, placement and opponent-response changes. The frequency
with which the best *rank* changes is useful diagnosis, but it is not a kill
gate: the same rank can refer to a different domino, placement or downstream
reply, and reveal-dependent value magnitudes can change the earlier root choice
without changing rank.

Use 100-300 frozen early/midgame positions spanning deck sizes and the BGA
failure suite. This is an approximate headroom audit, not exact solving: the
existing exact counter refuses bags above eight, and its fast path does not
already solve early/midgame chance states. Stop Workstream A only if the
confidence bound rules out a preregistered practically meaningful contingency
gap across the relevant joint-action and response measurements—not merely if
rank flips are rare.

#### A0 — information and probability invariants

**Completed 2026-08-08.** Before measuring strength:

1. Permuting hidden deck order leaves encoded tensors and raw inference exactly
   unchanged.
2. A chance panel contains every tile exactly `X` times.
3. Over many seeds, every four-tile row has the correct inclusion probability.
4. Chance backup equals an explicit weighted mean on synthetic trees.
5. Two revealed rows requiring different pick ranks produce different
   post-reveal policies.
6. Python and Rust agree on state keys, row probabilities and backup frame.
7. Every reachable pre-reveal bag satisfies `n % 4 == 0`.

All seven contracts now have executable gates:

| Invariant | Executable evidence |
|---|---|
| hidden-order safety | Python and Rust encoders remain byte-identical across redeterminizations; a deterministic raw network forward now also returns exactly identical value and legal logits for both player perspectives |
| exact tile exposure | balanced-panel tests count every public-bag tile exactly `X` times and verify each permutation cycle contains the bag once |
| marginal row law | repeated-seed support covers every deck-8 four-tile combination near its uniform expectation |
| probability backup | the generic Rust chance test pins an unequal `0.25/0.75` weighted expectation, and Python/Rust forced-tree tests match backed values across a sampled reveal |
| observation separation | two concrete revealed public rows have distinct versioned keys and retain deliberately different rank-conditioned policy maps |
| Python/Rust parity | versioned `KD-PUBLIC-v1` key material is byte-identical across 18 stepped states; all 70 deck-8 rows and their `1/70` probabilities match exactly; forced-tree actor-frame values match across the chance boundary |
| reachable bag shape | 64 complete random-legal games crossed 704 deals with every live and reconstructed pre-reveal bag divisible by four |

The legacy denial-search digest remains unchanged so existing frozen artifacts
stay readable. It includes Python's display/history-only board domino-ID array,
which the Rust rules state does not retain. New chance-correct topology uses the
versioned cross-language key material instead; it includes terrain, crowns,
sorted public bag membership, visible row, ordered claims, phase/actor fields,
rules flags and discard history, but never hidden bag order.

Focused verification: 4 new A0 tests, 32 existing panel/Rust-parity tests and
the unequal-weight Rust unit test all pass. The full 51-test Rust crate run
compiled successfully and its first 47 tests passed, including the new gate,
before the command ceiling caught four pre-existing long exact-solver stress
tests; those unrelated tests were not treated as an A0 failure.

#### A1 — one-reveal offline probe

Do not rewrite the entire production tree first.

1. Freeze the A-1 early/midgame public positions and a small selected deck=8
   oracle set.
2. Keep current open loop until the next reveal.
3. At that reveal, create a distinct subtree for each sampled row.
4. Search one observation-conditioned decision segment, then optionally fall
   back to existing open loop.
5. At every reachable pre-reveal deck size, sweep tile exposure
   `X={1, 2, 4, 8, 16}` and extend to 32 only where the curve has not
   saturated. Compare against IID panels with the same row count
   `K=nX/4`. Eliminate dominated widths before the game test.
6. On the selected overnight-solvable deck=8 set, use exhaustive chance search
   as ground truth: enumerate all `C(8,4)=70` unordered first-row outcomes and
   solve after conditioning on each revealed row. Do not label the broader
   early/midgame probe “exact.”

**Implementation checkpoint, 2026-08-08 — topology slice complete; strength
verdict still open.** The advisor-equivalent Rust open-loop search now has an
opt-in first-reveal split (`chance_exposure > 0`; default zero is exactly the
incumbent). It keeps the stateless action tree before the reveal, inserts an
afterstate/chance node when the next row becomes public, routes each distinct
public row to its own observation-conditioned subtree, and falls back to the
existing open-loop topology after that one observation segment. The fixed
support is exhaustive when `C(n,4) <= chance_enum_max_rows` (70 by default, so
deck=8 is exhaustive over rows) and otherwise uses `X` tile-balanced permutation
cycles. Duplicate rows are coalesced with their multiplicity retained as fixed
probability mass.

Chance Q is the explicit probability-weighted mean of conditional observation
subtree values. It never infers row probability from child visits. The sampled
next row is chosen independently of player action, all candidate actions at the
same public bag share the support, and remaining hidden order is re-randomized.
The treatment derives its chance draw from the incumbent simulation seed rather
than consuming an extra search RNG value, so enabling A1 does not shift every
later determinization. The actual game reveal remains unchanged. The new path is
currently probe/advisor-facing only; `BatchedMCTS` self-play and all defaults are
unchanged until A1/A2 pass.

Focused executable gates cover exact tile exposure, exhaustive 70-row deck=8
support, closed public observation identities, probability-weighted Q under
deliberately unequal child visits, exact disabled-path equivalence and a live
equal-simulation search. Six Rust open-loop tests and the combined 27-test
A1/denial Python suite pass. The companion
`chance_correct_search_probe.py` runs equal-simulation incumbent/X arms on the
frozen corpus and reports action/pick agreement, visit- and Q-pairwise ordering,
regret and latency against separate stronger incumbent and hybrid references.

The first 12-position training-budget screen used 4,800 simulations, `FPU=-0.2`,
`X={1,2,4}` and two independent 10,000-simulation topology references. It is
informative but not a verdict:

- the two strong references agreed on the exact joint action in only 7/12
  positions (58.3%), but on the picked tile rank in 11/12 (91.7%);
- four of the five joint disagreements were placement-only; the sole pick
  disagreement was a deck=24 position;
- the 4,800-sim incumbent matched the strong incumbent on 9/12 roots and the
  strong hybrid on 8/12; the 4,800-sim `X=4` arm matched the strong hybrid on
  12/12 but the strong incumbent on only 7/12; and
- equal simulation counts produced measured mean latency of approximately
  2.5-2.6 seconds/root for every arm in this small run. The first artifact did
  not separately count terminal-free NN rows, so the probe now records exact NN
  evaluations per arm; require that field for the equal-compute gate. Support
  width showed no wall-time difference at fixed simulations in this screen; its
  expected cost is fragmented visits per observation. A counted two-position
  smoke confirmed 129 NN rows for both 128-simulation arms (one root evaluation
  plus 128 leaves) and 513 for both 512-simulation references.

Matching a same-topology reference is not evidence of superiority. These data
show that the implementation has enough influence to change converged search,
but the two search families have not supplied a common arbiter. Do not proceed
to self-play integration from this screen. Next, rerun the five reference-
disagreement roots with multiple independent chance/search seeds, adjudicate the
selected deck=8 reveal-boundary cases with the exhaustive chance oracle, and
expand only after the reference consensus rate is adequate. The earlier
50-position 128-sim `FPU=0` run remains a topology/fragmentation smoke, not a
training-target result.

The implementation should reuse the public-state/chance machinery in
`denial_search.py` and port the fixed-support semantics already reviewed in the
7 Wonders Duel search. The latter is a design reference, not a drop-in tree:
Kingdomino still needs its own action topology, backup frame and Rust path.

##### Deck-specific support saturation

Do not ship one global `X`. Freeze representative positions for every reachable
pre-reveal bag size (`n=44, 40, 36, …, 8, 4`) and measure a separate convergence
curve. Each `(n, X)` cell must use multiple independently seeded panels, while
all candidate root actions within a panel share the same rows.

Examples of the balanced support size:

| Bag `n` | `X=1` | `X=2` | `X=4` | `X=8` | `X=16` |
|---:|---:|---:|---:|---:|---:|
| 44 | 11 | 22 | 44 | 88 | 176 |
| 32 | 8 | 16 | 32 | 64 | 128 |
| 24 | 6 | 12 | 24 | 48 | 96 |
| 16 | 4 | 8 | 16 | 32 | 64 |
| 8 | 2 | 4 | 8 | 16 | 32 |

Measure two curves rather than conflating chance coverage with search budget:

1. **Chance-estimation curve:** hold downstream work per row approximately
   fixed, so total compute grows with `X`. This asks whether additional support
   improves the evaluation at all.
2. **End-to-end search curve:** hold total NN evaluations or deployed wall time
   fixed, so larger `X` receives fewer downstream visits per row. This asks
   whether the extra coverage is worth fragmenting the tree.

For each cell report root top-1/pairwise agreement, mean and p90 regret, Q and
ranking variance across panel seeds, visits per row, NN evaluations, latency and
self-play games/second. Use exact enumeration at deck=4 and on the selected
deck=8 oracle set; benchmark whether deck=12 (`C(12,4)=495`) is practical before
calling it an oracle. Larger decks use a frozen high-support, multi-seed
consensus reference and must not be described as exact.

Failure to find a significant `X -> 2X` improvement is not proof of saturation.
Use an equivalence-style plateau rule: choose the smallest `X` only when the
upper confidence bound rules out a preregistered meaningful improvement from
doubling it across selection, regret and seed-stability metrics, then confirm
with one further doubling where affordable. When uncertainty remains, retain
the larger support because the objective is strength rather than maximum
throughput.

The likely result is two versioned lookup tables rather than one compromise:

```text
chance_exposure_training[deck_size]
chance_exposure_advisor[deck_size]
```

The advisor can justify a larger support because it runs continuously until its
top move stabilizes. Training search should pay that throughput cost only if the
larger support improves targets enough to produce a stronger checkpoint.

Separate bias reduction from budget fragmentation. At both the self-play
training budget and the 10,000-simulation advisor budget, compare:

- incumbent open loop at its normal budget;
- incumbent open loop at the hybrid's realized NN-evaluation/wall-time cost;
- IID observation splitting across the surviving `K` values; and
- tile-balanced splitting at matched `K` and total NN evaluations.

Keep total evaluations fixed when comparing search algorithms, and also report
per-observation visits. Otherwise a weak result cannot distinguish an
unimportant alias from spreading too little search across too many subtrees.

Primary measurements:

- on the solved deck=8 subset, root top-1 and pairwise action agreement with the
  exhaustive oracle; elsewhere, agreement/regret against a frozen
  stronger-search reference;
- mean and p90 root regret;
- action-ranking variance across chance seeds;
- policy changes on hand-checked robust-versus-brittle positions;
- the full contingency measurements from A-1, including rank-change frequency;
  and
- NN evaluations, wall time and self-play games/second.

This probe succeeds only if observation splitting improves selection or reduces
regret at equal compute. A value-MAE improvement alone is insufficient.

#### A2 — frozen-net game-strength test

Compare current open loop with the one-reveal hybrid using identical weights,
equal NN evaluations, paired decks and swapped seats. Include BGA anchor roots
as a separate stratum. Use the Section 3 `LCB > 50%` procedure for the frozen-net
search matchup; the paired-seat variance curve forecasts cost but does not
replace the promotion-strength gate.

Keep two verdicts separate:

- **advisor verdict:** the treatment must improve frozen-net selection or game
  strength at the deployed 10,000-simulation budget; and
- **training-search verdict:** a low-simulation improvement may proceed to a
  small training treatment if its targets are closer to the selected deck=8 or
  stronger-search arbiters, even when the frozen 10,000-sim advisor is unchanged.

A low-simulation-only win is therefore not an advisor enhancement, but it can
still create a stronger learned network. A regression at advisor budget is not
hidden by averaging the two verdicts.

#### A3 — training-loop treatment

Only after the corresponding A2 advisor or training-search verdict is positive:

- use chance-correct root visits/completed-Q targets for learner moves;
- retain standard terminal win, own-score and opponent-score targets;
- keep the actual environment reveal uniformly random;
- stamp the search/target version in replay metadata; and
- refuse to mix incompatible old and new policy targets silently.

Any resulting checkpoint still promotes only by clearing the Section 3 paired
`LCB > 50%` gate against `current_best` with equal search settings.

#### A4 — expand the observation split only if justified

If one reveal helps, extend chance nodes deeper using outcome progressive
widening or balanced panels. Measure realized child counts by depth. Do not
materialize all `C(44,4)` rows, and do not assume a full-tree rewrite is better
than the successful shallow hybrid.

### 4.6 Flexibility-specific audit

Maintain a small frozen suite containing:

- a robust placement/pick that retains several terrain plans;
- a brittle action that is excellent under a narrow set of rows;
- actions that trade current tile quality for earlier next-round draft order;
- two future rows whose correct post-reveal picks have different ranks; and
- defensive choices whose value depends on the opponent's opportunity after the
  reveal.

For each action, estimate its expected value over the same row panels. This is a
direct measure of flexibility under uncertainty, not legal-move count or a
hand-authored flexibility bonus.

## 5. Workstream B — BGA restart positions in the training loop

### 5.1 Why the archive matters—and its current limit

BGA roots are not intended to rival one iteration's raw position count. They
change the starting-state distribution. Each root can generate multiple
independent continuations and many downstream examples, while normal self-play
still supplies most of the buffer.

The human action is metadata, not the training label. From a BGA public state,
both sides resume self-play using the current search, with a newly randomized
hidden deck order consistent with the remaining bag.

This is search control: revisit states with unusually high learning value
rather than always starting from the opening. Regret-Guided Search Control is
the closest AlphaZero precedent; it reports gains on already-trained agents by
prioritizing high-regret restart states.

There are currently 43 raw `table_*.jsonl` logs and 36 clean games in the BGA
anchor. That is enough for schema/reconstruction tests, legal restart smoke tests
and a small curriculum pilot. It is not enough to populate four strategic
strata and estimate a precise held-out BGA effect. Multiple continuations
increase training trajectories, but they do not create new independent source
games for confidence intervals.

### 5.2 B0 — permission-safe corpus growth

The primary source is games already logged while the advisor is legitimately
used for play. Continue passive local logging of future Arena, tournament and
strong-opponent games; this is the most relevant distribution because it
contains the deployed model's own difficult states and failures.

Do not build a replay scraper or try to choose a request rate that appears less
suspicious. BGA's current terms prohibit automated use/content collection unless
expressly permitted and restrict substantial database extraction:
<https://en.boardgamearena.com/legal?section=legal>. The observed 200-replay
daily limit is a ceiling, not authorization to automate up to it.

Corpus expansion order:

1. Preserve the existing 36-game anchor definition and keep its held-out games
   out of restart training.
2. Train on newly passively logged advisor games, prioritizing legitimate play
   against strong opponents. More ordinary high-level play grows the archive
   without replay harvesting.
3. Ask BGA for written permission before any bulk replay collection. Request an
   official export/API or an approved narrow collector, specifying Kingdomino
   only, public moves/states only, a maximum rate/total, private noncommercial
   use, pseudonymization and retention/deletion policy.
4. Strong players may voluntarily contribute game identifiers or permitted
   exports, but player consent does not override BGA's platform terms; use only
   a retrieval mechanism BGA permits.
5. If permission is declined or absent, use advisor-logged games only. Do not
   replace the denied collector with low-rate automation, multiple accounts or
   another evasion mechanism.

There is no arbitrary 150-game prerequisite. Set collection aspirations from
strategic coverage and realized learning value, while remembering that the
authoritative evidence is the paired `LCB > 50%` checkpoint match—not a BGA
corpus metric. Report opponent strength, result, seat, reconstruction quality
and strategic coverage as the archive grows; raw game count alone is
insufficient.

For every proposed restart fraction, write down the reuse arithmetic. If an
iteration generates `I` games, restart share is `f`, the training archive has
`G` independent source games and each sampled root gets `c` redeterminizations,
report at least:

```text
restart games per iteration = f * I
mean restart trajectories per source game = (f * I) / G
nominal continuations per selected root = c
```

Also report unique source games and unique public roots actually touched. Use a
reuse cap based on those realized counts; thousands of continuations from a few
games are useful training data but not broad evidence.

### 5.3 Which games and positions to keep

Do not select only losses. A loss can be correct play under bad reveals, while
a win can contain an unpunished large error.

Initial archive strata:

| Stratum | Starting share | Purpose |
|---|---:|---|
| high-Elo losses and close games | 40% | likely punished weaknesses |
| high search/policy disagreement from any result | 30% | direct decision headroom |
| rare/high-leverage structures | 20% | blocking, tempo, bonus and geometry coverage |
| random clean high-Elo control | 10% | prevent a pathology-only archive |

Treat these as starting quotas, not permanent truths. Report realized counts and
adjust only between preregistered runs.

Eligibility:

- strong opponent or competitive arena/table;
- reliable viewer-turn state with no reconstruction warning;
- no forfeit, abandoned game or known capture corruption;
- deduplicate identical public states; and
- split by whole BGA game before any position-level filtering.

Use opponent Elo as a quality filter, not proof that the recorded move is
optimal.

### 5.4 Handling luck correctly

Do not filter an entire game merely because its final result looks unlucky.
Separate reveal luck from decision regret:

- use `runs/kingdomino/bga_luck_analysis.py` to compare actual reveals with
  counterfactual rows from the public bag;
- use `runs/kingdomino/bga_postmortem.py` and deeper paired search to identify
  adverse decision swings;
- downweight a lost game with poor reveal percentiles but no meaningful own
  action regret; and
- retain losses containing a large chance-controlled action regret even when
  the game was also unlucky.

Every training continuation discards the historical future deck order and
redeterminizes from the public remaining bag. The actual BGA future is one
sample for analysis, never privileged ground truth.

### 5.5 Chance-controlled action regret

For candidate roots, evaluate the played/recommended action and alternatives
against the same future-row panels. Record:

- expected win-value regret;
- pairwise action ordering;
- own-score delta;
- negative opponent-score delta (denial contribution);
- whether the best action changes draft order; and
- uncertainty across permutation cycles.

This also tests the blocking hypothesis. Identify actions where own score falls
but opponent score falls more, then measure whether policy/search systematically
underranks these defensive-dominant actions. Do not add a new blocking head
unless this audit demonstrates a specific deficit.

### 5.6 Training integration

The minimal capture schema needs source game/table, timestamp, opponent-rating
bucket, seat, raw event/state payload, capture version and reconstruction
status. Exclude chat and unrelated user data; replace usernames/player IDs with
stable pseudonymous IDs. Build the versioned public-state archive on top of it
with:

- source table/game and opponent metadata;
- public-state hash and serialized state;
- actor/seat, phase, deck size and game result;
- reconstruction/luck quality flags;
- played human action as metadata;
- search disagreement/regret and uncertainty;
- tags for tempo, blocking, bonuses and placement geometry; and
- a permanent train/validation/test assignment made at the game level.

Add a Rust start-from-public-state path that samples a fresh hidden permutation.
With the current thin archive, begin only with a pipeline smoke/pilot:

- 15% of self-play games starting from the game-disjoint BGA training split;
- 85% beginning from the normal opening;
- 5-10 independently redeterminized continuations per selected root over time;
- a per-iteration reuse cap so a thin archive cannot dominate one buffer; and
- sampling within the archive weighted by regret/rarity, with a nonzero uniform
  floor.

Sweep restart fraction only after a frozen pilot. Reasonable comparison arms are
0%, 10%, 20% and 30%; do not jump directly to a BGA-majority buffer.

The pilot may establish legality, stability, learning movement and gross
regressions. Treat its held-out BGA effect estimate as exploratory until passive
collection supplies enough independent source games for a useful clustered
diagnostic; checkpoint promotion does not wait on that metric and still uses the
paired head-to-head gate.

### 5.7 Gates

Cheap guards before full game evaluation:

1. No reconstruction failures or information leaks.
2. Restarted games complete legally under Python and Rust.
3. Train and held-out roots are game-disjoint.
4. Held-out BGA action regret improves without degrading the random high-Elo
   control stratum.
5. Policy change is concentrated enough to be explainable; global collapse or
   imitation of human moves is a failure.
6. Normal-opening validation and the frozen flexibility suite do not regress.

Promotion still requires the equal-search paired `LCB > 50%` gate against
`current_best`, plus no regression on the BGA anchor. Internal regret reduction
alone is not a promotion result.

## 6. Supporting levers

### 6.1 Gumbel sequential halving

Porting the 7WD implementation is attractive because the machinery already
exists. Its best-supported use is better policy improvement at low simulation
budgets, not a claim that it fixes the 10,000-sim advisor.

Kingdomino must not apply Gumbel top-k naively over flat joint actions: several
candidate placements can belong to one tile and still starve another tile.
Test a pick-stratified or hierarchical candidate set that represents all four
tile groups. Frozen-net arms should compare current PUCT/root floors, flat
Gumbel and pick-stratified Gumbel at equal NN evaluations. Use completed-Q
policy targets; a deterministic advisor should not execute a random
Gumbel-perturbed winner.

This is priority three, after the two primary workstreams have executable
pilots.

### 6.2 Deck=8 exact solving

Do not target the self-play loop or interactive advisor: prior testing indicates
deck=8 full solving is operationally untenable there. Its value is as an offline
oracle for Workstream A.

At deck=8, enumerate all 70 possible first rows and condition later decisions on
the row actually revealed. The solver must optimize the official outcome
cascade and average chance outcomes; solving a single fixed permutation would
reintroduce clairvoyance and is not a valid label.

Do not implement this as a Python loop around the existing deck-4 Rust solver
for arbitrary roots. On three frozen bag-8 positions, the current legal tree
reaches 26,112-41,400 concrete prefixes before the first reveal; multiplying by
70 rows would require roughly 1.8-2.9 million conditioned tail solves per root.
The existing fast solver handles only no-chance bag-4/bag-0 states, so that
wrapper would reproduce the already observed operational failure.

Start with a modest, strategically selected set at or immediately before the
reveal boundary, where 70 conditioned tails are tractable. Those cases can
validate balanced-panel convergence, flexibility rankings and backup semantics.
Exact adjudication of the original earlier roots requires the chance node and
its transposition/caching support inside the Rust solver rather than above it in
Python. This remains an offline oracle; it does not need advisor latency or
self-play throughput.

### 6.3 Paired-seat variance curve

This is now a prerequisite for expensive strength gates, not an optional
afterthought. Compare the same checkpoint at relevant simulation budgets on
identical decks with seats swapped. Use the paired distribution to forecast how
many games/GPU-hours candidates at 50.5%, 51% and 52% are likely to need before
the Section 3 lower confidence bound clears 50%. It does not create a separate
practical-effect gate. Do not reuse the 0.0076 searched-Q regret or 0.033 median
action gap as if either were a playing-strength ceiling.

### 6.4 Placement headroom audit

A beam search over a fixed ordered sequence of claimed dominoes can measure
placement score headroom, but it does not by itself produce a training target
that controls future draft uncertainty. Keep it as a diagnostic and compare
model versus high-Elo human gaps. Do not place it ahead of the two primary
workstreams.

### 6.5 Capacity and distributional value

Do not run a larger-capacity background arm on the current buffer: the recorded
80x6/96x6/80x10 bake-off already closed it. Revisit capacity only after the new
search or BGA curriculum demonstrably changes the buffer, and make that a fresh
controlled bake-off.

A distributional score/outcome head remains credible for representing
stochastic returns, but the mean is sufficient for risk-neutral optimal play;
it is not a substitute for correct chance topology or better state coverage.

## 7. Levers not to revisit without new evidence

- Tile-Q/child-value seeding from M0-M2.5.
- Pick-group visit floors as a shipped 10,000-sim inference setting. They remain
  possible low-sim label-generation machinery.
- More calibration work against the same 8-ply forced reference without a
  stronger arbiter.
- A deterministic known-deck state tree.
- Training directly on historical BGA actions as if they were ground truth.
- A hand-authored flexibility reward or legal-move-count proxy.
- Another blocking/opponent-reply head before a chance-controlled defensive
  regret audit demonstrates selection headroom.
- A larger 96x6/80x10 net on the unchanged training buffer.
- Combining two treatments that each failed its own selection, strength or
  non-regression gate.

## 8. Methodological rules

1. **Measure selection and game strength, not only calibration.** The failed
   tile-Q work showed that large MAE improvements can leave decisions unchanged.
2. **Every treatment needs a control cohort.** Include random BGA roots,
   rank-1 actions and normal-opening games as appropriate.
3. **Use paired statistics for paired designs.** Reuse chance panels and deck
   seeds across arms, then bootstrap the paired difference.
4. **Cluster by the data-generating unit.** BGA positions cluster by game;
   balanced chance rows cluster by permutation cycle; paired games cluster by
   deck/seat pair.
5. **Freeze gates before opening expensive results.** A confidence interval
   touching zero is not evidence of positive signal.
6. **Prefer network-independent arbiters.** Official outcomes, exhaustive
   deck=8 chance search and placement optimization are stronger arbiters than a
   forced tree using the same value head.
7. **Preserve information-set safety structurally.** Tests and serialization
   contracts should make future-order leakage impossible, not merely unlikely.
8. **Separate independent evidence from generated volume.** More
   redeterminizations or restart continuations improve estimation/training, but
   do not increase the number of independent BGA games or chance cycles.
9. **Measure the full contingency value.** Reveal-dependent rank changes alone
   cannot rule out concrete-tile, placement, opponent-response or backed-root
   effects.
10. **Promote only on strength.** Offline selection, regret, exact solving and
    BGA anchors screen and explain candidates; only the paired `LCB > 50%`
    match against `current_best` promotes one.

## 9. Execution order

1. Lock the 7WD-style paired `LCB > 50%` promotion procedure and run the
   paired-seat variance curve early enough to forecast later gate costs. Treat
   BGA rank as external validation, not a development-loop Elo target.
2. **Completed 2026-08-07:** measured terminal tie frequency and fixed official
   tiebreak consistency in Python, Rust, exact solving and batched evaluation.
3. Start B0 passive collection of newly advisor-logged games with the minimal
   versioned capture/archive schema. Seek written BGA permission for any replay
   corpus; do not build an automated collector without it.
4. **Completed initial screen 2026-08-08:** A-1 found stable backed-root
   contingency changes in matched balanced/IID arms through `X=4`. Expand the
   confirmation corpus with BGA failures during A1, but do not delay the hybrid
   build for a larger version of the same headroom probe.
5. **A0 and the opt-in A1 Rust topology slice completed 2026-08-08.** The first
   dual-reference training-budget screen is unresolved because the 10,000-sim
   incumbent and hybrid agree on only 7/12 joint actions (11/12 pick ranks).
   Next rerun those disagreement roots across independent seeds and adjudicate
   the selected deck=8 reveal-boundary cases exactly. Only then expand the full
   corpus and produce separate per-deck training/advisor exposure schedules from
   `X` saturation curves and width/compute controls. Do not integrate with
   `BatchedMCTS` self-play or spend on global `X=8` arms before that gate.
6. Add Rust start-from-public-state and run legal/reconstruction smoke tests.
   Run a small BGA restart training pilot with the current search once there is
   a game-disjoint training subset; label BGA-specific effect estimates
   exploratory while the corpus remains thin.
7. Run the A2 frozen-net search matchup and BGA restart checkpoint treatment as
   independent experiments. Preserve separate advisor and training-search
   verdicts for Workstream A; promote only when the relevant paired match clears
   the Section 3 lower-confidence-bound gate.
8. Combine BGA restarts with chance-correct targets only if each component has
   independent positive evidence.
9. Port pick-stratified Gumbel as the next low-cost training-efficiency arm.
10. Reconsider capacity only after a promoted treatment has materially changed
    the search-generated buffer.

The combined treatment is not allowed to rescue two individually negative
components. Each primary workstream must first pass its own selection and
non-regression gates.

## 10. Reference numbers

Measured 2026-08-06/07 unless noted, on `current_best` (sha `4bf07b0c…`).

| Quantity | Value |
|---|---|
| Policy top-1 agreement with 8-ply reference (held-out games) | 69.0% |
| Policy pairwise ordering accuracy | 82.7% |
| Policy mean searched-Q regret | 0.0076 |
| Median searched-Q gap, best vs second-best tile | 0.033 |
| Secondary tile visit share @3,200 / @10,000 sims | 2.19% / 1.16% |
| Best-prior placement == searched-best placement | 42.0% |
| BGA raw logs / clean anchor games | 43 / 36 |
| BGA anchor: net top pick == top-30 human's pick | 76.4% (top-2: 95.5%) |
| BGA deployment: non-top advisor placement selected | <1%, still within top 3 |
| BGA external result | 3rd in three-month Arena; currently 31st overall |
| run11a exploiter plateau vs banked net | ~48.5% over 15,000 games |

## 11. Research references

- Gumbel AlphaZero: <https://openreview.net/forum?id=bERaNdoegnO>
- POMCP/action-observation histories:
  <https://papers.nips.cc/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html>
- Stochastic MuZero afterstates/chance outcomes:
  <https://openreview.net/forum?id=X6D9bAHhBQ1>
- Sparse sampling for stochastic planning:
  <https://www.ijcai.org/Proceedings/99-2/Papers/093.pdf>
- Monte Carlo *-Minimax for stochastic zero-sum games:
  <https://arxiv.org/abs/1304.6057>
- Variance reduction and common random numbers in MCTS:
  <https://papers.neurips.cc/paper_files/paper/2011/hash/d736bb10d83a904aefc1d6ce93dc54b8-Abstract.html>
- Regret-Guided Search Control:
  <https://arxiv.org/abs/2602.20809>
- KataGo auxiliary targets and self-play improvements:
  <https://arxiv.org/abs/1902.10565>

Related local documents: `SECONDARY_PICK_FRAGILITY_FINDINGS.md`, `RUN10_PLAN.md`,
and `AZ_TILE_Q_HEAD_PLAN.md` on the `kingdomino-tile-q` branch.
