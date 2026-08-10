# Kingdomino: next levers for world-best play

- **Status:** Forward plan; experiments require their own frozen configs and
  preregistered gates before expensive training.
- **Date:** 2026-08-09
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

The completed A1b probe gives a narrower implementation direction. Explicit
observation splitting has useful signal, and locally balanced reveal routing
improves coverage, but a fixed global exposure `X` fragments conditional search
depth and the lazy sampled/Hájek arms did not identify one reference-stable
winner. The next candidate is therefore **fully initialized balanced chance
panels with visit-controlled progressive widening**:

1. initialize one complete tile-balanced cycle at a visited chance node;
2. batch-evaluate every row in that active panel;
3. back up its direct probability-weighted mean; and
4. add whole independent balanced cycles only as node visits justify more
   chance width.

This keeps the information-set safety of open loop while giving every revealed
public row its own adaptive continuation. The lazy sampled and Hájek variants
remain valuable ablations, not the intended production design.

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

At a chance node with `n` remaining tiles, one tile-balanced cycle is a random
permutation of the public bag partitioned into four-tile rows. `X` independent
cycles produce:

```text
K = n * X / 4 sampled rows
```

Keep the notation explicit: `X` is the current per-tile exposure count and `K`
is the number of row children. In the proposed progressive design `X` is
node-local and increases with chance-node visits; it is not one global search
setting. Every tile appears `X` times in one search panel, not `2X`; the
revealed row is shared by both players, and a seat-swapped evaluation game is a
separate search rather than extra support for the first one.

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

The first active cycle must be fully initialized before selective refinement:
evaluate all `n/4` rows, preferably in one NN batch, and form the direct
probability-weighted panel mean. That removes missing active probability mass
from the primary estimator. It does **not** make the estimate exact for the full
game: panel sampling error, network error and downstream search error remain.
Initialization work must be charged as `n/4` NN evaluations even when GPU
batching makes its wall time much smaller.

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
5. Use static `X={1,2,4}` only to diagnose coverage, estimator and width/depth
   behavior. The completed A1b result supersedes the proposed global sweep;
   A1c tests visit-controlled widening instead.
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
equal-simulation search. Ten Rust open-loop tests and the expanded 54-test
A1/denial Python suite pass (with one unrelated skip). The companion
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
but the two search families have not supplied a common arbiter. This triggered
the five-root paired-seed follow-up below; it did not authorize self-play
integration. The earlier
50-position 128-sim `FPU=0` run remains a topology/fragmentation smoke, not a
training-target result.

**Paired-seed follow-up, 2026-08-08.** The five disagreement roots were rerun
with eight paired seeds per root, four equal-budget 4,800-simulation arms and
separate 10,000-simulation open-loop and one-reveal references. Positions 3 and
6 were joint-placement search noise: every 4,800-simulation arm had the same
unanimous modal action. Position 2 retained a placement disagreement but both
reference families picked rank 2 on 8/8 seeds. Position 11 (deck=8) also retained
only a placement disagreement; both references picked rank 2 on 8/8 seeds. The
only persistent tile-pick disagreement was position 7 (deck=24): the incumbent
reference picked rank 3 on 8/8 seeds while the one-reveal reference picked rank
2 on 5/8. This narrows the substantive early/midgame discriminator to position
7, but 5/8 is not a sufficiently stable hybrid reference to declare it correct.

The follow-up added realized-support diagnostics and exposed an important
qualification. `X` fixes the composition of the chance support; it does not by
itself force every support row to be evaluated below every materialized chance
node. On representative 4,800-simulation reruns of positions 2, 7 and 11, the
visit-weighted evaluated probability mass was 71.1%, 65.6% and 63.3% for
`X={1,2,4}` respectively. The corresponding unweighted means, which give fringe
and principal actions equal weight, were 40.2%, 28.6% and 20.5%. Position 7 was
better covered (96.6%, 93.1%, 90.0% visit-weighted), so its pick disagreement is
not explained by broad support starvation. Position 2 fell to 64.3% at `X=4`.

Most importantly, the deck=8 root materialized all 70 unordered rows per chance
node but evaluated only 35.6% of visit-weighted probability mass; no materialized
chance node visited all 70 outcomes. The current closed-mean backup assigns
unvisited outcomes the actor-framed FPU value (`-0.2` in this probe). Therefore
"exhaustive support" must not be described as exhaustive evaluation or an exact
chance estimate. The exact deck=8 oracle remains a separate operation that must
actually evaluate/solve every conditioned row.

This finding defined the initial A1b search-semantics review:

1. At each visited chance node, traverse the frozen support in probability-
   balanced randomized cycles before repeating outcomes. For sampled panels this
   realizes the promised per-tile exposure locally, rather than only in the
   global panel definition.
2. Compare the current FPU-filled closed expectation with a sampled-chance backup
   that propagates the realized conditional value. The latter is an unbiased
   Monte Carlo estimate under probability-balanced traversal while preserving a
   distinct public subtree for every observed row.
3. Where affordable, add a fully initialized closed-mean arm that evaluates every
   support row once before revisiting it. Deck=8 exhaustive-oracle positions are
   the primary correctness anchor; wider early-game panels may make this arm
   intentionally impractical.
4. Hold total NN evaluations fixed, report visit-weighted unobserved probability
   mass, and reject any claimed strength gain that disappears when the backup's
   missing-mass treatment changes.

This A1b comparison precedes exact adjudication of the deck=8 placement and any
`BatchedMCTS` integration. It tests whether the observed treatment effect comes
from preserving public observations or from the arbitrary value assigned to
unvisited chance outcomes.

**Review correction, 2026-08-08.** The FPU-filled estimator was removed before
any further strength claim. Chance Q now omits outcomes with no real evaluation
and renormalizes the registered probabilities over visited mass; once every row
has been visited it becomes the fixed-panel expectation. Observation-child
virtual loss is excluded from that base estimate, while a separate chance-node
in-flight overlay supplies the full-strength virtual-loss penalty to ancestor
PUCT. Returned root-child Q is reconstructed from the current chance estimate
rather than accumulated stale expectation snapshots. Consequently, all action
comparisons reported above describe the superseded FPU-filled implementation
and are useful only as evidence that the old experiment was confounded.

The probe now uses a paired reference seed stream disjoint from the candidate
arms, records `disabled|sampled_balanced|exhaustive` plus panel row count, and
stratifies metrics by deck size. The disabled path has a nontrivial frozen
golden-vector test. Fixed-support registration is lazy: a deck=8 4,800-simulation
probe registered 24,710 chance outcomes but allocated only 1,897 observation
subtrees, avoiding 22,813 unused arena nodes. This reduced memory/allocation
work but did not produce a clear wall-time win in the GPU-dominated single-root
measurement; treatment arms remained within roughly 0-8% of the incumbent in
that run at identical 4,801 NN evaluations.

The corrected one-seed smoke changed selected actions materially (including the
deck=8 pick), confirming that the old dual-reference result is invalid as a
strength gate. Do not reuse it. The reviewed A1b comparison was defined as:

1. IID support traversal versus probability-balanced randomized cycles at each
   chance node, so local outcome coverage—not only global panel composition—is
   controlled.
2. Realized sampled-value backup as the null versus the current visited-mass-
   renormalized estimator as the treatment, at fixed NN evaluations. The
   observation split is the structural intervention; renormalization is an
   additional Hájek-estimator assumption that may add bias/noise at low
   coverage. If they tie, prefer sampled-value backup.
3. Fully initialized/exact conditioned-row evaluation on the selected deck=8
   oracle roots.

Entry and interpretation must use realized evaluated probability mass rather
than nominal `X`. Report exhaustive deck<=8 panels separately from truncated
deck>=12 panels, and measure whether chance-action visit count correlates with Q
rank before treating the renormalized arm as a strength result. The five-root
paired-seed probe was then rerun; its completed result is recorded below.

**Second review hardening, 2026-08-08.** The selection hot path no longer scans
every registered chance outcome for every candidate action. Each chance node
maintains visited probability mass and probability-weighted observation value
incrementally, making chance-Q lookup O(1); end-of-search diagnostics recompute
the values in debug builds to guard the cache. Virtual-only bookkeeping is now
restricted to chance nodes, so the disabled path and ordinary decision nodes do
not pay those extra writes.

The Rust path also asserts the three assumptions on which lazy row routing
depends: strictly sorted support, a stable post-reveal actor, and a one-to-one
mapping from `(chance node, sorted row)` to the full public information state.
The probe reports running-root and current-child root estimates under separate
names, emits null rather than fake consensus for one-seed runs, includes
position-clustered ordering metrics, and requires a recorded reason for targeted
position subsets. Its artifact now labels exhaustive references separately from
independently truncated panel estimates. Disabled-path regression coverage now
includes a deterministic deck=12, 512-simulation search in addition to the
opening smoke.

**A1b development checkpoint, 2026-08-08 — ready for logic review, not yet a
strength result.** The one-reveal engine now exposes the preregistered
`sampled|hajek` backup and `iid|balanced` traversal axes while leaving all
production defaults unchanged. In both traversal modes PUCT commits to the
pre-reveal action before a row is selected, so local balancing cannot reveal a
row through legality, priors or Q. Balanced routing maintains a separately
shuffled, multiplicity-expanded cycle at each materialized chance node; every
raw panel slot is consumed locally before that node repeats a cycle.

Sampled backup propagates the realized leaf value through the chance afterstate
and ancestors, while retaining the row-specific observation subtree. Hájek
backup uses the same topology but substitutes the incrementally cached
registered-probability mean over evaluated rows. The probe records both mode
labels, per-arm realized-mass eligibility at a configurable threshold, local
balanced-route/cycle counts, and the mean within-parent Spearman correlation
between chance-action visits and actor-framed Q rank. Schema-v5 artifacts carry
the complete arm matrix; `chance_correct_a1b_probe.py` preconfigures one x0
incumbent plus the `sampled,hajek × iid,balanced` cross-product for every
positive `X`.

At this checkpoint the longer frozen-position run was held for review of this
boundary and the sampled-backup arithmetic. The subsequent run started with the
selected roots and equal NN budgets; an arm failing the realized-mass gate
remains a coverage diagnostic, not strength evidence.

**A1b review correction.** The matrix uses `chance_enum_max_rows=12` as the
exhaustive-enumeration threshold, rather than materializing the 70-row deck=8
panel. This deliberately makes the 50% realized-mass threshold attainable and
asks whether estimator/traversal choice helps once coverage is adequate. It is
not evidence that the small panel approximates the
true reveal distribution; panel fidelity remains a separate A1 question using
exhaustive deck<=8 adjudication.

A CPU smoke on the real deck=8 position showed that the lower cap alone is not
enough at 128 simulations: balanced x4 reached only 0.36-0.39 visit-weighted
mass. The A1b preset therefore uses the established 4,800-simulation candidate
budget and 10,000-simulation references. The gate, rather than that expectation,
remains authoritative; if 4,800 still misses 0.5 on any search, its headline is
withheld.

The strong hybrid reference is evaluated twice under the same balanced panel:
once with sampled backup and once with Hájek backup. A result is actionable only
if the arm winner is stable across both reference estimators. Headline metrics
are now withheld for any arm with a search below the realized-mass threshold;
raw metrics remain diagnostic. The visit/Q-rank check is read only as the
matched `Hájek - sampled` contrast at equal `X` and traversal, using Fisher-z
aggregation and parent groups with at least three chance actions. The
theoretically important result is the `hajek_balanced` interaction cell, not an
isolated main effect.

The implementation should reuse the public-state/chance machinery in
`denial_search.py` and port the fixed-support semantics already reviewed in the
7 Wonders Duel search. The latter is a design reference, not a drop-in tree:
Kingdomino still needs its own action topology, backup frame and Rust path.

**A1b GPU result, 2026-08-09 — signal found; static lazy `X` not promoted.** The
reviewed 12-position screen completed in 10m58s, followed by five targeted roots
times eight paired seeds in 33m56s. Every treatment used approximately 4,801 NN
evaluations, and treatment runtime stayed within ±0.7% of `X=0`, so the O(1)
chance-Q cache removed hot-path throughput as the immediate concern.

The five-root result favored balanced routing, but not one estimator/exposure
combination across both references:

- against the sampled hybrid reference, `X=1` Hájek/balanced reduced mean regret
  from 0.013678 to 0.007390, with top-1 0.800 versus 0.625;
- against the Hájek hybrid reference, `X=2` Hájek/balanced reduced mean regret
  from 0.014081 to 0.009434, while `X=1` reached 0.009889;
- `X=1` balanced covered all 40 searches and more visit-weighted probability
  mass than matched IID; and
- all `X=4` arms failed the realized-mass gate in 8/40 searches, concentrated in
  the widest early-game root. `X=2` sampled/IID failed there as well.

The strict cross-reference winner gate therefore did not pass. Eight seeds show
search stability, not 40 independent strategic positions: the effective
generalization sample remains five roots. The two 10,000-simulation references
agreed on only 3/5 exact actions and 4/5 tile picks. In particular, the deck=8
position 11 disagreement remains an oracle question, not evidence that either
search family is correct. These results support observation splitting and local
balance, but do not yet establish better training targets.

#### A1c — fully initialized, progressively widened chance panels

Do not ship one global `X`, and do not make the lazy Hájek estimator the primary
path. Implement the following candidate behind an opt-in flag:

1. Below a deck-dependent chance-node threshold `N_init(n)`, traverse one
   correctly sampled row at a time and use sampled-value backup. Do not pay for
   a full cycle at a node that may never be revisited.
2. When `N_init(n)` is crossed, atomically create the rest of one independent
   tile-balanced cycle, giving `K=n/4` active rows and exposing every remaining
   tile once. Batch-evaluate the missing rows and only then switch from sampled
   backup to the complete-panel mean; never treat a partial cycle as complete.
3. Store each active row's network value, legal policy and public observation
   identity. Form the direct probability-weighted mean over the complete active
   panel; no active probability mass is missing.
4. Keep initialization estimates distinct from MCTS visit counts. A row's
   bootstrap evaluation contributes to chance value, but must not masquerade as
   a search visit or inflate its policy-target count.
5. After initialization, route refinement in randomized balanced cycles and
   deepen one row-specific public subtree at a time. Chance routing is never
   selected by row value.
6. Add another whole independent balanced cycle only when chance-node visits
   cross a widening boundary such as:

   ```text
   X(N, n) = min(X_max(n), ceil(c * N^alpha))
   K(N, n) = n * X(N, n) / 4
   ```

   The exact schedule is an experiment parameter, not a correctness claim.
7. Share active panels across competing actions that leave the same public bag
   (common random support), while keeping the randomized traversal cursor local
   to each chance node.
8. Re-randomize all still-hidden order after the sampled row. Hidden order never
   enters a node key, NN input or training record.

For equal-NN-budget experiments, preregister a hard initialization guardrail:

```text
initialization_nn_evals / total_nn_evals <= 0.25
```

If completing another first cycle would exceed that cap, leave that node in
sampled mode. The 25% value is an engineering safety limit, not a theoretical
optimum; it may change only between preregistered experiments.

This is analogous to the useful part of the 7WD chance implementation—evaluate
all active chance expansions and average them—but adds selective MCTS refinement
because Kingdomino cannot exhaust its early-game chance tree. Progressive
widening addresses the width/depth tradeoff: at fixed work, larger `X` splits
visits across more observation subtrees and can make the search shallower. Full
initialization guarantees coverage of the active panel, not unlimited width.

**A1c implementation checkpoint, 2026-08-09 — wave-safe advisor prototype.**
The opt-in advisor diagnostic path now preserves whole panel cycles, separates
balanced panel construction from the matched-width IID ablation, delays the
first cycle until `N_init`, admits at most one complete cycle per chance node and
leaf-parallel wave, batch-evaluates every missing row before committing the
cycle, and uses bootstrap values without incrementing MCTS visits. The cumulative
initialization guard charges the proposed batch before admission and reports
initialized/blocked cycles, NN rows and initialization fraction. It also reports
the real sampled-backup visits that predate first-panel admission and their
fraction of visits to initialized chance nodes; those earlier contributions
remain in ancestor running means and can dilute a finite-budget A1c effect. The
incumbent and lazy modes retain their prior defaults and regression vectors.

Panel admission is now wave-safe and `leaf_batch=8` is restored. Descent only
records an admission request. Every path in the wave then completes evaluation,
removes virtual loss and backs up under the estimator it traversed. Requests are
deduplicated by chance node only after those backups, and at most one atomic
cycle per node is committed before the next wave. Diagnostics report requested
paths, unique requested nodes, committed cycles and admission waves; functional
tests pin visit conservation through the transition.

The advisor diagnostic search now also accepts an optional hard total-NN-work
budget. It charges the root, ordinary leaves and every A1c bootstrap row; shortens
ordinary waves before they can overshoot; and refuses a whole initialization
cycle when it cannot fit rather than partially committing it. Diagnostics report
ordinary/initialization work, evaluator calls and batch sizes, completed
simulations/waves, budget exhaustion, unused budget and separately blocked A1c
cycles. When several nodes request admission together, constrained work goes to
the highest real-visit nodes first; equal-visit ties use a seeded permutation
rather than action/node insertion order. The Python probe independently counts
actual evaluator rows and fails on any Rust/Python accounting mismatch.

The deck-8 oracle harness now enables A1c `X=4,8` balanced/IID specifications at
the same `leaf_batch=8` as their controls only with a positive hard NN budget. It
rejects the artifact if any arm hits its simulation ceiling before exhausting
that budget. This makes equal-NN work a diagnostic of search efficiency, not the
final strength objective. Retain separate later comparisons at equal wall time
and at each search family's strongest feasible configuration; a slower A1c arm
can still advance if its oracle decisions and paired game strength justify the
cost. No GPU comparison has yet been run, so this checkpoint is not evidence of
stronger play.

Retain three ablations: incumbent open loop, lazy sampled/balanced and lazy
Hájek/balanced. Hájek self-normalization is a finite-sample biased estimator and
the available theory does not cover its adaptive use inside this PUCT tree; it
should win empirically before displacing the simpler complete-panel mean.

##### Support and budget validation

Before tuning, freeze 240 independent positions stratified across every
reachable pre-reveal bag size (`n=44, 40, 36, …, 8, 4`), ordinary self-play,
BGA failures, flexibility/draft-order cases and defensive-blocking cases. Assign
120 positions permanently to tuning and 120 to untouched confirmation. Freeze
the schedule before opening confirmation results; repeated search seeds estimate
noise but do not increase either position count.

Initially fix `alpha=0.5` and tune only `N_init`, the widening constant and a
small monotone set of early/mid/late deck-band caps. Do not fit an independent
`X_max` for every deck size. Target a 95% position-clustered CI half-width no
larger than 0.0025 for mean regret and five percentage points for
epsilon-optimal tile-selection rate. If 120 confirmation positions do not reach
the precision target, add a pre-reserved confirmation tranche without changing
the schedule.

One fully initialized cycle is 11 rows at deck=44, 7 at deck=28,
6 at deck=24, 4 at deck=16 and 2 at deck=8. `X={1,2,4,8}` therefore means
`K={7,14,28,56}` at deck=28. This makes explicit why a larger exposure can lose
at fixed simulations despite estimating chance more broadly.

Measure two curves rather than conflating chance coverage with search budget:

1. **Chance-estimation curve:** hold downstream work per row approximately
   fixed, so total compute grows with support. This asks whether wider support
   improves the evaluation at all.
2. **End-to-end search curve:** hold total NN evaluations or deployed wall time
   fixed, so wider support receives fewer conditional visits per row. This asks
   whether coverage is worth the lost depth.

Count every initialized row as an NN evaluation even when evaluated in one GPU
batch. Also record batch size/occupancy, peak arena memory, initialized rows,
initialization NN evaluations and their fraction of total work, searched visits
per row, conditional depth by reveal, latency and self-play games/second. The
laptop establishes logic and scaling curves; an RTX 5090 cloud box is used only
after the architecture and batching path pass review.

Use epsilon-aware ordering metrics because several placements can be
strategically equivalent:

- joint-action regret and membership in a preregistered epsilon-optimal set;
- tile-pick regret after optimizing placement, reported separately from
  conditional placement regret;
- pairwise ordering, mean and p90 regret and ranking variance across panel
  seeds; and
- clustered intervals by strategic position, not by repeated search seed.

Use exact enumeration at deck=4 and the selected deck=8 oracle set. Larger decks
use a frozen high-support, multi-seed consensus reference and must not be called
exact. For deck>=12, same-family or competing-family reference regret is a
screen, not an arbiter or pass gate. Drop the A1b 50% realized-mass eligibility
filter from cross-arm strength comparisons: it structurally favors fully
initialized A1c. Report active-panel completion, unique full-support fraction,
tile balance, effective panel size, realized mass and conditional depth as
covariates without withholding regret. A1c passes only on deck=8 oracle evidence
and/or paired game outcomes at matched work, while retaining feasible depth and
batched cost. Value MAE or same-topology agreement alone is insufficient.

#### A2 — frozen-net game-strength test

The existing promotion harness is not sufficient: it supplies one shared
simulation/search configuration to both seats, and the one-reveal path is not
yet in `BatchedMCTS`. Add these explicit deliverables:

1. an asymmetric paired search-match API with independent immutable `SearchSpec`
   values for player A and player B, identical deck seeds with seats swapped,
   official outcome scoring and the Section 3 clustered LCB; and
2. a serial backend that calls the existing advisor open-loop/one-reveal search
   on every move, followed later by a `BatchedMCTS` backend using the same specs
   and cross-checked against fixed serial games.

**A2 harness checkpoint, 2026-08-09 — serial backend complete.**
`chance_correct_match.py` now provides independent immutable
`SearchSpec` values, the existing Rust advisor search on every non-forced move,
paired identical deck seeds with seats swapped, deterministic visit/prior action
selection, official outcome scoring and the promotion module's pair-clustered
LCB. It records per-arm search calls, NN evaluations and search time, requires a
selection reason and refuses to overwrite a completed artifact. Focused unit,
real Rust-boundary and promotion-statistics tests pass; a tiny random-network
two-game CPU smoke completed legally with equal 94-evaluation arm counts. This
validates plumbing only. A2a's simulation budget, fixed pair count, confidence
level, futility boundary and output path were frozen before the real checkpoint
match.

**A2a — pre-A1c pulse match.** Before building A1c, use the serial harness to
compare the already implemented `X=1`, Hájek/balanced treatment against `X=0`
at one preregistered low-simulation training budget. This is a non-circular game-
outcome test of whether observation splitting has a playing-strength pulse. Use
the normal paired `LCB > 50%` success rule plus a preregistered futility boundary.
A negative result lowers A1c's priority but does not automatically close it,
because A1c specifically changes lazy missing-mass and initialization behavior.

**A2a result, 2026-08-09 — inconclusive with a negative point estimate.** The
sealed 256-pair/512-game match used `current_best`, 800 simulations per move and
seed range beginning at 2026082000. `X=1`, Hájek/balanced scored 119 paired
points (46.48%) against `X=0`; its pair-clustered 95% Wilson interval was
40.47%-52.60%. Pair outcomes were 32 wins, 174 draws and 50 losses; game
outcomes were 238-0-274 with a -1.49 mean treatment margin. It therefore clears
neither the `LCB > 50%` success gate nor the `UCB < 50%` confidently-harmful
gate. Do not extend this sample adaptively or reinterpret the point estimate as
a pass. The result supplies no playing-strength evidence for the existing lazy
one-reveal treatment and lowers A1c's priority, but it does not adjudicate A1c's
delayed full-cycle initialization or missing-mass correction.

The run completed in 9,499.8 seconds (2 h 38 m). Treatment and control used
12,598 and 12,588 search calls, 711.54 and 710.69 NN evaluations per call, and
0.3754 and 0.3780 search seconds per call. The disjoint four-pair pilot used
717.11/726.96 evaluations and 0.4437/0.4513 seconds per call. The full run thus
shows no treatment-side budget or throughput regression; its lower per-call
times are consistent with amortization relative to the small pilot.

**A2b — candidate frozen-net match.** After A1c clears deck=8 oracle screening
and is integrated into `BatchedMCTS`, compare current open loop with the fully
initialized/progressively widened search using identical weights, equal NN
evaluations, paired decks and swapped seats. Include BGA anchor roots as a
separate stratum. Include an incumbent arm with enough extra simulations to
match treatment wall time so the result cannot be explained by budget accounting
alone.

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

Only after the A1c target-quality gate and corresponding A2 advisor or
training-search verdict are positive:

- use chance-correct root visits/completed-Q targets for learner moves;
- retain standard terminal win, own-score and opponent-score targets;
- keep the actual environment reveal uniformly random;
- stamp the search/target version in replay metadata; and
- refuse to mix incompatible old and new policy targets silently.

Before renting the cloud box, require code review, information/probability
invariants, deck=8 oracle support, a laptop end-to-end `BatchedMCTS` smoke and a
measured batch/memory scaling curve. Then resume the current best 80x6 network;
do not start with another capacity change. Use a lower visit-driven support cap
for self-play and a larger cap for the continuously running advisor only when
their separate curves justify it.

Evaluate two gates. First compare challenger and incumbent checkpoints under the
same new search to isolate learning. Then compare the complete new
model-plus-search system with the currently deployed model-plus-open-loop system.
Both use paired decks/seats and the Section 3 `LCB > 50%` rule; the first gate is
the authoritative network-promotion decision.

#### A4 — expand the observation split only if justified

If one reveal helps, extend the same fully initialized/progressively widened
semantics to later reveals one boundary at a time. Measure active support,
conditional depth and memory by reveal depth. Do not materialize all
`C(44,4)` rows, and do not assume a full-tree rewrite is better than the
successful shallow hybrid.

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

- 5% of self-play games starting from the game-disjoint BGA training split;
- 95% beginning from the normal opening;
- 5-10 independently redeterminized continuations per selected root over time;
- a per-iteration reuse cap so a thin archive cannot dominate one buffer; and
- sampling within the archive weighted by regret/rarity, with a nonzero uniform
  floor.

The first controlled comparison is 0% versus 5%. Expand to 10% only if the 5%
arm moves the intended held-out strata without degrading normal-opening play;
larger fractions require new independent source games and a fresh preregistered
experiment. The point is rare elite-failure coverage, not making a 36-game
archive numerically comparable with one self-play iteration.

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

Full row enumeration exposes each of the eight tiles in `C(7,3)=35` rows, so it
is equivalent to `X=35` first-order exposure, not `X=8`. Average the 70
conditioned results with exact `1/70` probability. Report an epsilon-optimal
joint-action set, epsilon-optimal tile set after placement optimization, tile
regret and conditional placement regret rather than forcing one arbitrary top-1
when several placements are tied or practically indistinguishable.

Do not implement this as a Python loop around the existing deck-4 Rust solver
for arbitrary roots. Earlier measurements found 26,112-41,400 concrete prefixes
before the first reveal on three bag-8 roots. The explicit position-11 count is
larger: 105,576 pre-reveal action sequences and 7,390,320 conditioned deck-4
tail solves. The existing fast solver handles only no-chance bag-4/bag-0 states,
so that wrapper would reproduce the already observed operational failure.

Start with a modest, strategically selected set at or immediately before the
reveal boundary, where 70 conditioned tails per legal placement are tractable.
Those cases can validate balanced-panel convergence, flexibility rankings and
backup semantics. Freeze candidate boundaries using explicit action prefixes;
prefer boundaries where frozen search families disagree, then add cases that
distinguish draft-order flexibility and blocking. This does not exactly
adjudicate the original position-11 root: it adjudicates the chance-boundary
values reached by a stated continuation. A few overnight boundaries are useful
even though the solver is too slow for self-play or the live advisor: their value
is independent ground truth for architecture and target-quality decisions.

Deck-4 solve time is heavy-tailed and varies with board complexity and play.
Never estimate a corpus from a small easy prefix alone. Persist every solved
`(placement, row)` cell, run tiered time caps, and retry only missing cells at a
higher cap. An action receives an oracle expectation and rank only after all 70
rows solve; do not average the solved subset or silently replace timed-out rows.
Exact adjudication of the original earlier roots requires the chance node and
its transposition/caching support inside the Rust solver rather than above it in
Python. This remains an offline oracle; it does not need advisor latency or
self-play throughput.

Drive conditioned tails serially. Each no-chance call already uses the Rust
solver's Lookahead2Clustered ordering, root-level YBW parallelism and a shared
bound-flagged transposition table; concurrently launching many internally
parallel tails risks CPU oversubscription and deadline-induced fallbacks. The
remaining possible optimization is cross-tail reuse of exact solve results.
Measure the existing transposition-rate diagnostic first; do not build a shared
cross-tail cache unless it shows meaningful overlap.

**Boundary-oracle checkpoint, 2026-08-09.** `deck8_oracle.py` implements the
guarded final-pre-reveal boundary described above. It requires exactly eight bag
tiles, the last current-round actor and one remaining public pick; enumerates 70
unique uniform rows; invokes the no-chance solver on every resulting deck-4
state; persists identity-checked cells; resumes missing work; and refuses to
rank incomplete actions. Focused boundary, actor-frame, support and resume tests
pass.

The first completed position-11 incumbent-aligned boundary froze prefix actions
`1997,3152,3165`. All 490 tails for seven legal placements solved exactly with
no timeout in 222.5 seconds. This particular boundary was easier than the known
tail: median 0.363 s, p90 0.823 s, p99 1.185 s and max 1.654 s per tail. Exact
action 275 led action 2880 by 0.01737 actor utility. In eight repeated 4,800-sim
searches, both `X=0` and lazy `X=1` selected action 275 on 8/8 seeds with zero
oracle regret. `X=0` had 89.3% exact-Q pairwise ordering versus 86.3% for lazy
`X=1`; both had 81.0% visit ordering. This is one boundary and validates the
oracle/comparison seam, not A1c. The alternative treatment-aligned position-11
continuation also produced the same `X=0`/`X=1` boundary choice and was not
exact-solved. Screen for a genuinely discriminating frozen boundary before the
next exact run.

**Fixed calibration corpus, 2026-08-09.** The preregistered screen derived both
`X=0`- and lazy-`X=1`-directed continuations from all eight frozen deck-8 roots,
deduplicating 16 lines to 15 public boundaries. Across three paired 4,800-sim
seeds, it found no stable disagreement: 12/15 boundaries had both arms
unanimous, while three were noisy. Two position-5 boundaries had different
modal choices but failed the unanimous gate and were not cherry-picked for
exact solving.

The follow-up therefore froze a separate calibration corpus independently of
the screen outcome: the `X=0`-derived boundary from every frozen root. All eight
positions, 68 legal placements and 4,760 `(placement, row)` tails solved exactly
with no timeout; total exact wall time was about 37.5 minutes. The original
position-11 result was reused. Cell means are now checked to reproduce every
stored oracle action expectation before analysis. A review subsequently found
that the generic solver API could silently fall back to Python if Rust state
conversion failed. New oracle cells now require and record the Rust no-chance
backend, while these already-completed cells are labeled `legacy_unverified`;
their zero-timeout completion is strong operational evidence but not explicit
per-cell backend provenance.

Against these oracles, three-seed 4,800-sim `X=0` search selected an exact-best
placement on 66.7% of nested searches with mean position regret 0.00576. Lazy
`X=1` scored 62.5% and 0.00697. These are eight independent positions, not 24;
the repeats estimate search stability. They reinforce A2a's lack of positive
evidence for the lazy treatment but remain a boundary-placement diagnostic, not
a game-strength result.

Exact-tail panel resampling used common rows across actions and 2,000 trials per
position, scheme and exposure. Top-1 recovery and mean exact regret were:

| Exposure | Rows | Balanced top-1 / regret / unique support | Matched IID top-1 / regret / unique support |
|---:|---:|---:|---:|
| `X=1` | 2 | 81.1% / 0.00116 / 2.9% | 82.6% / 0.00094 / 2.8% |
| `X=2` | 4 | 87.8% / 0.00060 / 5.6% | 88.5% / 0.00053 / 5.6% |
| `X=4` | 8 | 92.6% / 0.00029 / 10.9% | 93.4% / 0.00023 / 10.9% |
| `X=8` | 16 | 96.1% / 0.00011 / 20.7% | 96.3% / 0.00009 / 20.6% |
| `X=16` | 32 | 98.4% / 0.00003 / 37.1% | 98.2% / 0.00003 / 36.9% |
| `X=35` | 70 sampled | 99.4% / 0.00001 / 63.8% | 99.4% / 0.00001 / 63.4% |
| exhaustive | 70 unique | 100% / 0 / 100% | 100% / 0 / 100% |

`X=35` above is still a 70-row random panel, not enumeration of every unique
combination; the full oracle is 100% by definition. This isolates chance-panel
sampling with exact conditional tails and therefore cannot be compared as if it
were an equal-compute MCTS arm. It says `X=1` is light, most sampling benefit is
present by `X=4`-`X=8`, and first-order tile balance has not outperformed
matched-width IID on this small corpus. Preserve IID as an A1c ablation and do
not claim a balanced advantage. A1c must still demonstrate that its conditional
NN/search values retain this oracle ordering under its charged NN budget.

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
11. **Never optimize chance.** Row generation and traversal may be balanced or
    widened by visit count, but never selected from value, policy or legality
    information that was unavailable before the reveal.
12. **Charge batched work honestly.** One forward batch over `K` initialized
    rows is `K` NN evaluations even if it costs much less than `K` separate
    launches. Report both evaluation count and wall time.
13. **Do not confuse repeated seeds with new positions.** Seeds estimate search
    noise for one root. Strategic generalization intervals cluster by frozen
    position, and BGA intervals cluster by source game.

## 9. Execution order

1. Lock the 7WD-style paired `LCB > 50%` promotion procedure and run the
   paired-seat variance curve early enough to forecast later gate costs. Treat
   BGA rank as external validation, not a development-loop Elo target.
2. **Completed 2026-08-07:** measured terminal tie frequency and fixed official
   tiebreak consistency in Python, Rust, exact solving and batched evaluation.
3. **Completed through A1b, 2026-08-09:** A-1 found contingency headroom; A0
   locked the information/probability invariants; A1 built the one-reveal Rust
   topology; and the reviewed GPU probe found balanced-routing signal without a
   reference-stable static-`X` winner. Do not rerun the same five-root matrix as
   if more seeds created more strategic evidence.
4. **A2a completed 2026-08-09: inconclusive, negative point estimate.** The
   frozen `X=1` Hájek/balanced treatment scored 46.48% paired points versus
   `X=0`, with a 95% interval of 40.47%-52.60%. Do not add games adaptively.
   This does not support the existing lazy one-reveal treatment, while leaving
   A1c's distinct initialization hypothesis unresolved.
5. **Deck-8 boundary oracle and fixed eight-position calibration corpus
   completed 2026-08-09.** Exact position-11 root solving is millions of tails
   and is not the laptop oracle. Boundary oracles show `X=1` is light,
   `X=4`-`X=8` captures most exact-tail sampling benefit, matched IID is at least
   competitive with balance, and the existing lazy treatment does not improve
   oracle decisions. Review this slice, then specify and implement A1c. It starts
   in sampled mode, atomically
   initializes its first balanced cycle only after `N_init`, caps initialization
   at 25% of NN work, then widens in whole balanced cycles. Wave-safe advisor
   admission, `leaf_batch=8` and matched total-NN-work stopping/accounting are
   complete. The deck-8 A1c oracle comparison is now mechanically enabled but
   has not been run. Preserve
   incumbent and lazy sampled/Hájek paths as ablations. Neither the deck>=12 stronger-
   search screen nor A2a alone is an A1c target-quality gate.
6. Freeze the 120-position tuning and 120-position confirmation split before
   schedule tuning. On the laptop, run A1c correctness smokes and per-deck
   width/depth curves on tuning positions only, then exercise the real
   `BatchedMCTS` self-play path. Measure batched initialization throughput,
   initialization-budget fraction and peak memory. Do not launch a large global
   `X=8` sweep.
7. Start B0 passive collection from advisor-logged games and add the versioned
   Rust start-from-public-state path. Run reconstruction, redeterminization and
   legal-completion tests; the first training comparison is 0% versus 5% BGA
   restarts. Seek written permission before any replay corpus automation.
8. Freeze A1c schedules, open the untouched confirmation set, and run A2b through
   the asymmetric `BatchedMCTS` harness at matched NN work, plus the wall-time-
   matched incumbent control. Preserve separate training-budget and advisor-
   budget verdicts; only deck=8 oracle evidence and paired game outcomes are
   gates, while deck>=12 reference regret remains diagnostic.
9. Request final logic/throughput review. Only after the architecture, oracle,
   laptop smokes, `BatchedMCTS` integration and batch/memory curve pass should an
   RTX 5090 cloud box be rented.
10. Resume `current_best` 80x6 for controlled challenger training. First isolate
    chance-correct targets and the small BGA restart curriculum; do not use a
    larger net or revive the Q head. Gate the challenger against the incumbent
    under the same new search, then gate the complete new system against the
    currently deployed system, both with paired decks/seats and `LCB > 50%`.
11. Combine BGA restarts with chance-correct targets only if each component has
    independent positive evidence. Port pick-stratified Gumbel next; reconsider
    capacity only after a promoted treatment materially changes the buffer.

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
| A1b reviewed GPU screen | 12 roots × 1 seed in 10m58s; targeted 5 roots × 8 seeds in 33m56s |
| A1b treatment NN work / runtime overhead | ~4,801 evaluations; within ±0.7% of `X=0` |
| A1b best sampled-reference arm | `X=1` Hájek/balanced: regret 0.007390 vs 0.013678 |
| A1b best Hájek-reference arm | `X=2` Hájek/balanced: regret 0.009434 vs 0.014081 |
| A1b strong-reference agreement | 3/5 exact actions; 4/5 tile picks |

## 11. Research references

The literature backs the architecture class, not the exact Kingdomino recipe.
In particular, there is no theorem here that directly validates a
tile-balanced, without-replacement panel combined with neural PUCT, full panel
initialization and visit-controlled widening. Those details remain empirical
and are kept behind oracle, equal-compute and playing-strength gates.

- **Information-safe observation branching:** Silver and Veness, *Monte-Carlo
  Planning in Large POMDPs* (POMCP), indexes search by action-observation
  histories and uses root sampling. This supports preserving the pre-reveal
  information set while separating post-reveal public decisions:
  <https://mlanthology.org/neurips/2010/silver2010neurips-montecarlo/>.
- **Neural afterstate/chance topology:** Antonoglou et al., *Planning in
  Stochastic Environments with a Learned Model* (Stochastic MuZero), separates
  deterministic afterstates from stochastic outcomes and demonstrates neural
  MCTS in 2048 and backgammon. Kingdomino has an exact simulator, so only the
  topology—not the learned dynamics—is borrowed:
  <https://mlanthology.org/iclr/2022/antonoglou2022iclr-planning/>.
- **Sparse stochastic planning:** Kearns, Mansour and Ng prove that a finite
  IID sampled successor tree can support near-optimal planning with complexity
  independent of total state count. Their theorem uses independent generative-
  model samples; it does not directly justify the balanced panel:
  <https://www.ijcai.org/Proceedings/99-2/Papers/093.pdf>.
- **Adversarial stochastic games:** Lanctot et al., *Monte Carlo Star-Minimax
  Search*, applies with-replacement chance sampling in two-player stochastic
  games and reports practical gains. The paper explicitly says its proof does
  not cover sampling without replacement, which is why deck=8 and paired
  empirical gates remain necessary:
  <https://www.ijcai.org/Proceedings/13/Papers/093.pdf>.
- **Progressive widening:** Couëtoux and Doghmen study adding double progressive
  widening to UCT so stochastic outcome width grows with node visits, directly
  motivating the proposed width/depth controller while also showing the schedule
  parameters require tuning:
  <https://ewrl.wordpress.com/wp-content/uploads/2011/08/ewrl2011_submission_29.pdf>.
- **Progressive-widening caveat:** Sunberg and Kochenderfer show that naive DPW
  can converge to a suboptimal policy in continuous-observation POMDPs because
  its belief particles collapse. Kingdomino's reveal becomes fully public and
  its outcome set is finite, so the failure is not directly transferable, but it
  warns against treating widening alone as a correctness proof:
  <https://arxiv.org/abs/1709.06196>.
- **Hájek/self-normalized estimation caveat:** Cardoso et al. give modern
  finite-sample bias, variance and concentration results for self-normalized
  importance sampling. Self-normalization introduces bias, and their setting is
  not adaptive PUCT; this supports retaining Hájek as an ablation rather than a
  presumed-correct primary backup:
  <https://proceedings.neurips.cc/paper_files/paper/2022/hash/04bd683d5428d91c5fbb5a7d2c27064d-Abstract-Conference.html>.
- **Locally balanced chance streams—related, not direct:** Li, Chen and Huang's
  2026 preprint uses persistent correlated chance streams in MCCFR, proves local
  frequency control and reports lower exploitability. It is recent and studies
  CFR rather than MCTS, so it is supporting motivation for randomized balanced
  cycles, not validation of this implementation:
  <https://arxiv.org/abs/2607.27035>.
- **Common random numbers in MCTS:** Veness, Lanctot and Bowling:
  <https://papers.neurips.cc/paper_files/paper/2011/hash/d736bb10d83a904aefc1d6ce93dc54b8-Abstract.html>.
- **Low-simulation policy improvement:** Gumbel AlphaZero:
  <https://openreview.net/forum?id=bERaNdoegnO>.
- **Restart-state search control:** Regret-Guided Search Control:
  <https://arxiv.org/abs/2602.20809>.
- **Auxiliary targets and self-play engineering:** KataGo:
  <https://arxiv.org/abs/1902.10565>.

Related local documents: `SECONDARY_PICK_FRAGILITY_FINDINGS.md`, `RUN10_PLAN.md`,
and `AZ_TILE_Q_HEAD_PLAN.md` on the `kingdomino-tile-q` branch.
