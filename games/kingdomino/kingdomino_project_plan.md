# Kingdomino AlphaZero — Project Plan (Final)
## Goal: Train the world's best Kingdomino player through pure self-play

---

## Strategic Core

> **HOF broadens the state distribution. Reanalysis upgrades the labels. Exact endgame search fixes the highest-confidence value targets. Calibration diagnostics tell us whether the model is actually learning the right thing.**

Model size and game count are necessary but not sufficient. The infrastructure around the training loop determines whether scale translates to strength. The smoke run (Milestone 10) is the proof-of-concept gate — the first moment the full stack proves itself on real training data before committing cloud spend.

---

## Guiding Principles

- **No human game data** — per AlphaZero paper, removing human data enabled stronger play. Pure self-play discovers strategies humans never considered.
- **Measure before building** — empirical benchmarking before every optimization. Rerun throughput benchmarks whenever sims or model size changes.
- **Equivalence testing catches real bugs** — multiple real bugs found exclusively through equivalence testing in this project. Never skip correctness gates.
- **Correctness before throughput** — fixes applied in correctness order; optimization comes after gates pass.
- **Promotion gating with confidence** — only carry forward checkpoints that genuinely beat the prior best, with statistical confidence, not raw win rate alone.
- **Provenance everywhere** — once HOF, schedules, gating, reanalysis, and cloud hardware are in play, every checkpoint must be traceable to exact code, rules, and config.
- **The smoke run is the gate** — prove the full infrastructure stack works before committing cloud budget.

---

## Current State

**32ch/4b model (~1029–1046 Elo):** Confirmed saturated. The self-play distribution collapsed into a Nash equilibrium of mutual mistakes. Not a capacity issue — a training distribution issue.

**48ch/6b model:** Run 6 completed (cold start, alpha=0.8, lr=3e-4, 50 iters, 140 games/iter, 200k buffer). Elo 955 at iter 25, 977 at iter 50. Still below 32ch/4b ceiling but trajectory is normal — fast early gains followed by slow steady climb is expected AlphaZero behavior. Run 7 planned as warm continuation.

**BGA advisor:** End-to-end verified correct. Engine scores match BGA exactly (opponent board: engine=131=BGA; your board: engine=31=BGA). Public-information-safe search active, evaluator at alpha=0.0. All opponent board reconstruction bugs resolved (KINGDOM string, container ID rotation, CSS transform scale, cell pitch, vertical half-assignment rot1/rot3).

**Known remaining advisor limitation:** Endgame win probability overconfident because the network's leaf value estimate is wrong at terminal-adjacent positions. Standard search reaches too few terminal leaves to correct it. More simulations does not fix this — it averages more wrong estimates. Exact endgame search (Milestone 1) is the fix.

---

## Understanding the Training Loop

### Sequential vs Concurrent Training

The current training loop is **strictly sequential per iteration**:
1. Generate N self-play games (~86% of iteration time — GPU doing MCTS inference)
2. Add games to buffer
3. Run M training steps on random buffer samples (~2% of iteration time)
4. Log metrics, optionally evaluate Elo

The network used to generate games in iteration N was trained at the end of iteration N-1. There is always a one-iteration lag.

**Concurrent training** (overlapping generation and training) has value for GPU utilization and more gradient steps per game, but at current throughput the bottleneck is game generation (86% of time), not training (2%). Eliminating training time entirely would give only ~2% wall-clock speedup. The real throughput lever is faster game generation via Rust acceleration (Milestone 12), not concurrent training architecture. Option 1 (periodic sync — overlap training with next iteration's generation) is the right minimal improvement when the time comes; full async (Leela-style) is appropriate at cloud scale with multiple CPU cores.

### What the Iteration Output Means

```
batched timing: step=92.6s (12%), eval=669.6s (86%), update=16.6s (2%)
```

- **eval (86%)** — GPU forward passes for MCTS leaf evaluation. Dominant cost. `mean_batch=155.5/192 (81% fill)` shows GPU batch utilization.
- **step (12%)** — CPU/Rust tree work: PUCT selection, node expansion, backpropagation.
- **update (2%)** — training: sample buffer, compute losses, backpropagate gradients, update weights.

```
diff=-2.2+25.2
```
Mean score differential across self-play games (mean, std). Near-zero mean with high std = competitive, close games = healthy.

```
buf=200000(age=13.2)
```
Buffer at capacity, average position age 13.2 iterations. With ~7,300 positions/iter and 200k buffer, expected average age is ~13.7 — correct and healthy.

```
grads: pol=0.469 win=0.109 own=0.017 opp=0.017
```
Gradient norms per head. Reflects lambda weighting: policy_weight=1.0 dominates, lambda_w=0.25 for win, lambda_score=0.5 split between own/opp. Policy gradient still substantial = still learning.

### Reading Training Metrics

**Policy loss** is NOT the primary signal for whether the model is getting stronger. It measures how well the network predicts its own MCTS visit distributions. It can plateau for two very different reasons:

1. **Network has learned the policy well** (good) — KL between network prior and MCTS visits is small because the network accurately predicts what search would choose
2. **MCTS targets have stopped improving** (bad/neutral) — teacher collapse, search rubber-stamping a weak network

You cannot distinguish these from policy loss alone. This is why KL diagnostics matter (Milestone 2).

**Policy loss rising then flattening is normal** in the middle training phase — the model has learned easy patterns and is now refining harder ones where targets are genuinely difficult to predict. The 32ch/4b cloud run showed the same shape.

**Win brier is a cleaner signal** — ground truth is game outcomes (not moving targets), so continued improvement here reflects genuine win head calibration.

**Elo is the ground truth metric.** A model can gain 200 Elo with policy loss moving less than 0.1. Evaluate against the anchor pool every 25-50 iterations. Slow steady Elo climb with flat policy loss is normal and expected at scale.

**What would actually be alarming:**
- Elo declining for 3+ consecutive evaluations
- Win brier rising back toward baseline
- Policy entropy collapsing (policy collapse)
- Elo completely flat for 50+ iterations

### Policy Loss vs Model Strength at Scale

At 500k–1M games the expected pattern is:
- Policy loss oscillates or stays roughly flat — targets keep evolving as the model improves
- Win brier continues improving slowly — cleaner ground truth
- Elo is the ground truth — don't use policy loss to judge whether a long run is working

### KL Divergence

KL divergence measures how different two probability distributions are. Here it measures how much the MCTS visit distribution (what search concludes) differs from the network policy prior (what the network predicts before search).

**Never interpret KL alone — always pair with Elo:**

| KL | Elo | Interpretation |
|---|---|---|
| High | MCTS improves Elo | Search teaching useful corrections — good |
| High | MCTS doesn't improve | Search noisy or miscalibrated |
| Low | High Elo | Genuine convergence |
| Low | Mediocre Elo | Teacher collapse / rubber-stamping |
| Low (endgame only) | Any | Exact search needed — MCTS not correcting value |
| High (openings) | Any | Policy uncertain — more HOF/exploration needed |

### Why Endgame Is Hard Despite "Simple" Positions

With ≤3 tiles remaining, the tree has at most ~1,296 leaf nodes (6^4). At 800 simulations you could theoretically cover every path. But AlphaZero MCTS **never plays to terminal nodes** — it stops at leaves and uses the network's value estimate. Even with exhaustive coverage of the branching, every leaf is evaluated by the miscalibrated network. More simulations just averages more wrong estimates. The fix is exact endgame search (Milestone 1) — replace network estimates with `board.score()` completions.

This is distinct from pure MCTS (Can't Stop style) which uses random rollouts to actual game completion. AlphaZero's network-based leaf evaluation is both its strength (strong mid-game evaluation) and its weakness (imprecise terminal-adjacent values that don't improve with more sims).

### Alpha Schedule Rationale

- **Alpha=0.8 early** — win head needs dense gradient to calibrate; margin signal provides this
- **Alpha=0.2 mid (~iter 30-50)** — transition once win_brier confirms win head is calibrated. Mostly win probability with margin as a tiebreaker for won/lost positions
- **Alpha=0.0 late** — aligns training with evaluation (advisor already uses 0.0); pure win probability
- **Alpha=0.0 always for evaluation** — confirmed best Elo in sweep

The abrupt alpha switch (e.g. 0.8 to 0.2 cold) can destabilize training because the model's internal representations were built around a specific leaf value distribution. Transitions should happen after buffer has flushed old data at the prior alpha (roughly one staleness window).

### Why the Model Saturates

The 32ch/4b model saturated at ~1029–1046 Elo. **It means** the model learned everything it could from the distribution of positions two copies of itself create — a Nash equilibrium of mutual mistakes. **It does not mean** the game is solved or the model plays optimally.

The self-play loop has a structural weakness: training data quality is bounded by the quality of the games being played, which is bounded by the model. Circular dependency. Even at 50k+ games, endgame diversity is thin — each game has 4-6 endgame positions, but the space of endgame configurations is enormous, and both players make the same kinds of mistakes symmetrically.

**The fix:** HOF broadens the distribution, reanalysis upgrades the labels, exact endgame search provides perfect ground truth at terminal-adjacent positions, and diagnostics confirm whether any of it is working — together far stronger than increasing model size or game count alone.

### KataGo Paper — Applicable Techniques

KataGo (2020) achieved ~50x training efficiency vs the original AlphaZero on Go.
Key techniques and their applicability:

| Technique | KataGo finding | Applicability |
|-----------|---------------|---------------|
| Score/value aux targets (NoVAux) | Second-largest ablation loss | ✅ Already implemented (own_score, opp_score, win_prob heads) |
| Game-specific features (NoGoFeat) | Largest ablation loss | ✅ Already implemented (flat encoder: terrain counts, crowns, claims, progress) |
| Playout cap randomization | Outperforms all fixed-N alternatives | ✅ Added to Milestone 5 (`--playout_cap_randomization`; legacy `--fast_game_fraction` retained) |
| Policy target pruning | Decouples MCTS noise from policy learning | ✅ Added to Milestone 5 (`--policy_target_pruning`) |
| Global pooling bias | Third-largest ablation loss | ⏳ Future — investigate if training plateaus; score head may benefit from nonlocal board context |
| Auxiliary policy target | Modest benefit | ⏳ Future — architecture change, defer until capacity limit is reached |

**Global pooling** (KataGo's biggest remaining opportunity for this project):
computes board-wide statistics from the spatial trunk and injects them as biases
into later convolutional layers. For Kingdomino, the `flat` input partially achieves
this (pre-computed global features), but dynamic global pooling computed from the
board representation itself may improve score head accuracy. Investigate if the
diagnostics show persistent margin_mae plateau after sufficient training.

---

## A Note on Search Terminology

The current implementation uses **root redeterminization / PIMC**: the search treats the root as a known world, and `run_pimc` samples determinizations via `redeterminize()` and aggregates root visit counts. This is PIMC over closed-loop searches, not necessarily true open-loop MCTS where tree nodes are action sequences rather than concrete states.

This distinction matters for the leak tests and exact endgame implementation. The binding requirement: **hidden tile order must be handled by public-information-safe search — all advisor and training paths must sample only from the public-consistent remaining bag.**

---

## Milestone 0 — Model Contract Verification
**Gate: active branch contract printed and asserted before anything else runs**

```
python -m games.kingdomino.print_model_contract --checkpoint <checkpoint.pt>
```

Output must include and assert: `checkpoint_version`, `FLAT_SIZE`, `NUM_JOINT_ACTIONS`, board shape, channels, blocks, value head structure, policy head size, `ruleset_hash`.

Runs as a precondition before training, advisor startup, reanalysis, and promotion. Architecture constants are treated as *unverified* until this gate passes. Do not hard-code them into downstream code until confirmed from the active branch.

---

## Milestone 1 — Exact Endgame Search + Hidden-Order Leak Tests
**Gate: engine score = BGA score in all tested positions; all leak tests pass**

### Scope: Both advisor AND training self-play

**Advisor path:** when budget allows, replace network leaf value with exact board scoring for terminal-adjacent positions. Immediate benefit to recommendation quality.

**Training path (higher leverage):** when a self-play game reaches a terminal-adjacent position, the MCTS leaf evaluation switches from the network value head to exact scoring using `board.score()`. The simulation plays out remaining moves exhaustively using the public-consistent bag and backs up the true terminal value. Policy and value targets recorded for those positions are grounded in exact outcomes rather than network estimates. This compounds across every future self-play game — the online fix is more valuable than the offline reanalysis fix.

Additionally, stored terminal-adjacent positions in the buffer can be retroactively relabeled via reanalysis (Milestone 9 — offline fix).

### Budgeted public-information-safe solver

**Current implementation note:** in 2-player Mighty Duel, deck size is always a
multiple of 4. The implemented online exact-solve states are `deck == 0` and
`deck == 4`. With four tiles left, hidden order is irrelevant because those four
tiles deterministically become the sorted next row, so the tree is no-chance
minimax rather than public expectimax. BatchedMCTS solves these roots exactly
once in Rust, skips GPU/MCTS for that move, emits exact child-value policy
targets, and falls back to normal MCTS only if the wall-clock budget is exceeded.
The current BatchedMCTS routine training default is
`exact_endgame_max_secs=3.0`, exposed as a CLI/run setting. Use `0.0` for
ablation and larger budgets for quality-first or reanalysis runs. On a timeout the
slot marks the game `exact_unsolvable` and uses MCTS for the rest of that game
(reset per new game), so a hard endgame costs at most ~one fallback per game.
Python OpenLoop/advisor paths use the same wall-clock budget plus a three-state
per-node cache (Unsolved / Solved / Unsolvable) to avoid per-simulation retries.

The 1600-sim smoke figures below were measured under the old node budget and are
historical: `off=0.0742 games/s`, `500k=0.0816`, `2M=0.0769`, `5M=0.0760`.

The original budget sketch below is retained as historical context; it is
superseded by the current deck `{0,4}` no-chance implementation described above.

The trigger is not simply "≤3 tiles" — actual cost depends on: number of hidden unrevealed tiles (chance branching), legal placements per tile (can exceed 6), pick/turn-order branches, discard/no-placement cases, and current phase.

```
--exact_endgame_max_nodes        50000
--exact_endgame_max_hidden_tiles 3
--exact_endgame_mode             public_expectation
```

Estimate public-consistent tree size first. If below budget, solve exactly. If not, fall back to sampled PIMC with **exact terminal scoring at leaves** — capturing the value wherever exhaustive solving is feasible.

### Hidden-order leak tests (bundled here)

- Same public state, different hidden deck order → encoder output identical
- Same public state, shuffled hidden deck order → recommendation distribution statistically similar
- Exact small-bag enumeration vs sampled search → sampled converges toward exact as sims increase
- No true-deck access in advisor or training mode → fails loudly if hidden order is used
- Endgame solver → enumerates only public-consistent futures

### Where
`web_app.py` (advisor), `mcts_az.py` (training), `self_play.py` (game generation), `test_endgame_exact.py`

---

## Milestone 2 — Calibration and Phase-Sliced Diagnostics ✅
**Gate: diagnostics produce interpretable values on existing checkpoints**
**Status: Complete**

### Diagnostic metrics

| Metric | Slices | Purpose |
|---|---|---|
| `win_brier_by_phase` | opening / midgame / endgame | Catches endgame overconfidence early |
| `margin_mae_by_phase` | tiles-remaining buckets | Score head accuracy |
| `value_calibration_bins` | predicted win% vs actual | Calibration curve |
| `root_value_vs_final_margin` | all positions | Systematic over/underestimation |
| `mcts_lift_rate` | per phase | Does search choose better than raw network? |
| `network_policy_vs_mcts_policy_kl` | per phase | Network prior vs MCTS visits |
| `raw_net_top1_vs_high_sim_top1` | per phase | Directly measures policy correction by search |
| `exact_endgame_value_error` | endgame only | Value error where ground truth exists |

### `mcts_lift_rate` — precise definition

For positions where raw-network top-1 differs from MCTS top-1, compare both candidate moves from the same public state via exact search, high-sim search, or paired rollout. This adjudicates the disagreement directly rather than relying on noisy game outcome.

### Alpha transition trigger (empirical, not fixed schedule)

Monitor `win_brier_by_phase` every N iterations. Trigger alpha transition when `win_brier / baseline_brier` crosses a threshold (e.g. win head 50%+ better than baseline in the endgame slice). The fixed iteration schedule is a pragmatic fallback; the empirical trigger is the goal. Build empirical trigger as an enhancement to Milestone 5 dynamic schedules.

### Where
`games/kingdomino/diagnostics.py`, called from `self_play.py`, output to training log JSONL

### Results
- `games/kingdomino/diagnostics.py`: 7 metrics, `compute_all_diagnostics`
  entry point, `check_alpha_transition` stub
- Phase slicing via `game_progress` from existing flat encoder (no schema change)
- Gate passed: all metrics JSON-serialisable, internally consistent
- Key finding: endgame already well-calibrated due to Milestone 1 exact labels;
  opening is now the hardest phase (win_brier=0.146 vs endgame=0.013)
- alpha_trigger would fire on current checkpoint (ratio=0.05 << 0.5 threshold)
- 8 tests passing in `test_diagnostics.py`

---

## Milestone 3 — Fixed Evaluation Suite
**Gate: suite produces stable, reproducible scores across checkpoints**
**Status: Complete**

### Storage strategy
- Small, committed, reviewable `eval_suite_v1.jsonl`
- Larger generated suite referenced by seed/config, not committed wholesale

### Position categories
Endgame (validated against exact search values), model-disagreement, phase-representative, bonus-tension (harmony almost complete, middle kingdom live/dead).

### Position schema
```json
{
  "position_id": "...",
  "ruleset_hash": "...",
  "public_state": {},
  "source": "selfplay|advisor|generated|human_review",
  "phase": "...",
  "tiles_remaining": 0,
  "expected_exact_value": null,
  "notes": null
}
```

### Where
`scripts/build_eval_suite.py`, `scripts/run_eval_suite.py`, `data/kingdomino/eval_suite_v1.jsonl`

### Results
- `data/kingdomino/eval_suite_v1.jsonl`: 17 committed public-state roots
  - opening: 4
  - midgame: 4
  - endgame: 9
  - exact-valued endgames: 9
- `scripts/build_eval_suite.py`: deterministic suite builder; stores sorted hidden-bag membership, not true deck order
- `scripts/run_eval_suite.py`: deterministic raw-network fixed-suite runner for any checkpoint, with summary JSON and per-position JSONL details
- Self-test command:
```powershell
.\.venv\Scripts\python.exe .\scripts\run_eval_suite.py --suite .\data\kingdomino\eval_suite_v1.jsonl --out .\eval_results\eval_suite_v1_selftest_summary.json --details_out .\eval_results\eval_suite_v1_selftest_details.jsonl --device cpu --channels 8 --blocks 1 --bilinear_dim 16 --selftest
```
- Reproducibility gate passed: two self-test runs produced byte-identical summary and detail outputs

---

## Milestone 4 — Expanded Open-Loop / PIMC Leak Tests
**Gate: all tests pass before any cloud spend**
**Status: Complete once the Milestone 4 public-info gate command passes**

Beyond terminal-boundary tests (Milestone 1): training game generation (no deck leak into encoded positions), buffer storage (public state only), encoder symmetry (player-0 vs player-1 relative encoding), advisor state reconstruction (`normalizeBgaState` sends only public info), determinization independence (each simulation draws independently from the public-consistent bag).

### Where
Extended `test_open_loop.py`, run in CI before cloud-run approval.

### Gate command
```powershell
.\.venv\Scripts\python.exe .\scripts\run_milestone4_public_info_gate.py
```

The named gate covers:
- encoder/public bag invariance under hidden-deck reorder and redeterminization
- root policy/legality invariance for public states
- replay-buffer schema guard: stored examples cannot carry true deck order
- advisor import guard: `debug.deck` order is encoding-inert, and the BGA extension sends a sorted hidden-bag membership list
- Python and Rust redeterminization independence from the same public bag
- Python/Rust open-loop equivalence and exact-endgame public-consistency tests

---

## Milestone 5 — Dynamic Training Schedules
**Gate: single run executes full curriculum without manual intervention**
**Status: Complete for schedule plumbing + KataGo-inspired playout-cap/pruning smoke; strength ablation pending**

### Schedule flags
```
--lr_schedule                   "0:1e-3,50:3e-4,150:1e-4"
--alpha_schedule                "0:0.8,30:0.2,60:0.0"
--sims_schedule                 "0:800,20:1600,400:3200"
--games_per_iter_schedule       "0:140,20:180"
--c_puct_schedule
--dirichlet_epsilon_schedule
--temp_moves_schedule
--train_steps_schedule
--buffer_capacity_schedule
--hof_fraction_schedule         "0:0.0,50:0.1,100:0.2,200:0.3"
--reanalyze_fraction_schedule
--fast_game_fraction            0.15   # KataGo: fraction of games per iter using fast sims for exploration
--fast_game_sims                100    # sim count for fast exploration games (pairs with --sims_schedule full cap)
--playout_cap_randomization     True   # preferred KataGo-style move-level fast/full search mix
--full_search_fraction          0.25   # probability a non-exact move uses the full sim cap
--fast_move_sims                100    # sim count for fast moves
--record_fast_moves             False  # fast moves advance play but are not policy examples by default
--fast_move_dirichlet_epsilon   0.0    # fast moves are noiseless by default
--fast_move_temp_moves          0      # fast moves are greedy by default
--policy_target_pruning         True   # KataGo-inspired: prune <=1-visit policy noise before storing pi
```

`hof_fraction` ramps gradually — a cold jump to 30% HOF games would destabilize a still-forming policy.

Alpha transitions should happen after the buffer has flushed data at the prior alpha (roughly one staleness window) to avoid destabilizing the model.

### KataGo-derived additions

**Legacy game-level playout-cap mix** (`--fast_game_fraction`): within each iteration,
a fraction of games use a small fixed sim count (e.g. 100) for exploration while
the remainder use the full scheduled cap. KataGo ablations show this outperforms
any fixed-N alternative — it decouples the tension between policy targets (which
want many simulations to be accurate) and value targets (which want diverse board
states). `--fast_game_fraction_schedule` is available but a fixed value is the
recommended starting point.

**Preferred move-level playout cap randomization**
(`--playout_cap_randomization`): each non-exact self-play move independently
chooses a full or fast search. Full moves use the scheduled sim cap, normal root
noise, normal temperature, and are recorded. Fast moves default to strong cheap
play: `--fast_move_sims 100`, `--fast_move_dirichlet_epsilon 0.0`,
`--fast_move_temp_moves 0`, and `--record_fast_moves` off. This is closer to
KataGo than the older game-level mix because it mixes fast/full searches within
games instead of making whole games fast or whole games full. Exact-solved
endgame moves are still recorded. When this flag is enabled, the loop bypasses
the legacy `--fast_game_fraction` split so the two modes do not stack.

**Policy target pruning** (`--policy_target_pruning`): current implementation
performs the replay-record-safe part of KataGo's cleanup: prune MCTS policy
children whose mass is consistent with <=1 visit, renormalize the target, and
leave exact endgame targets untouched. Full forced-playout subtraction
(`n_forced(c) = sqrt(k * P(c) * sum_N(c'))` with k=2) requires root priors in
the Rust/Python self-play record contract and is deferred as a follow-up.

Gate addition for these two: after 20 iterations with/without, check that
`policy_kl_opening` decreases faster with pruning enabled and that `sp_score_diff_std`
increases with fast games mixed in. Both should show signal within 20 iterations.

Enhancement: add empirical alpha trigger based on `win_brier_by_phase` threshold rather than fixed iteration.

### Where
`games/kingdomino/self_play.py`

### Implemented
- CLI schedule flags for LR, alpha, simulations, games/iteration, `c_puct`,
  root noise epsilon, temperature moves, train steps, buffer capacity, and
  fast-game fraction.
- Per-iteration active config is applied to self-play, training, benchmark
  construction, checkpoints, diagnostics, and JSONL logging.
- KataGo-inspired move-level playout-cap randomization via
  `--playout_cap_randomization`, `--full_search_fraction`,
  `--fast_move_sims`, `--record_fast_moves`,
  `--fast_move_dirichlet_epsilon`, and `--fast_move_temp_moves`.
- Legacy game-level playout-cap mix remains available via
  `--fast_game_fraction`, `--fast_game_fraction_schedule`, and
  `--fast_game_sims`.
- Conservative policy target pruning via `--policy_target_pruning`, with
  exact-endgame examples skipped.

### Gate commands run
```
.\.venv\Scripts\python.exe -m py_compile .\games\kingdomino\self_play.py .\games\kingdomino\test_milestone5_schedules.py
.\.venv\Scripts\python.exe -m pytest -q .\games\kingdomino\test_milestone5_schedules.py
.\.venv\Scripts\python.exe -m games.kingdomino.self_play --engine python --device cpu --iterations 2 --games_per_iter 2 --train_steps 0 --sims 2 --sims_schedule 0:2,1:3 --games_per_iter_schedule 0:2,1:2 --lr_schedule 0:0.001,1:0.0005 --alpha_schedule 0:0.8,1:0.2 --fast_game_fraction 0.5 --fast_game_sims 1 --policy_target_pruning --channels 8 --blocks 1 --bilinear_dim 16 --benchmark_every 0 --elo_every 0 --exact_endgame_max_secs 0 --log_path .\tmp_m5_smoke.jsonl
cargo check
.\.venv\Scripts\python.exe -m maturin develop --release --manifest-path .\games\kingdomino\kingdomino_rust\Cargo.toml
.\.venv\Scripts\python.exe -m games.kingdomino.self_play --engine batched_open_loop --device cpu --iterations 1 --games_per_iter 2 --train_steps 0 --sims 4 --playout_cap_randomization --full_search_fraction 0.5 --fast_move_sims 1 --channels 8 --blocks 1 --bilinear_dim 16 --batch_slots 2 --leaf_batch 2 --benchmark_every 0 --elo_every 0 --exact_endgame_max_secs 0 --log_path .\tmp_m5_playout_cap_smoke.jsonl
```

Results: compile passed, 7/7 M5 tests passed, Rust `cargo check` passed, the
extension rebuilt via maturin, and the playout-cap smoke completed with fast
moves played but not recorded (`recorded_fast=0`). Temporary smoke logs removed
after verification.

---

## Milestone 5.5 — Run Manifests and Model Contracts
**Gate: every checkpoint traceable to exact code, rules, schedules, architecture, and provenance**

Every run saves:
```
run_manifest.json
git_commit.txt
dirty_diff.patch
model_contract.json
ruleset_hash.json
schedule_config.json
hardware_benchmark.json
```

Placed before promotion gating and HOF deliberately — once those systems are active, ambiguity about what produced a checkpoint is unrecoverable.

### Where
`games/kingdomino/run_manifest.py`, invoked at run start and checkpoint save

---

## Milestone 6 — Promotion Gating + Checkpoint Role Separation
**Gate: regression never propagates to advisor or HOF pool**

### Checkpoint roles

| Role | Path | Updated when |
|---|---|---|
| `latest` | `runs/.../iter_NNNN.pt` | Every iteration |
| `current_best` | `best_checkpoint/current_best.pt` | Passes statistical promotion gate |
| `hof/` | `best_checkpoint/hof/iter_NNNN.pt` | Every 50 iters, only from `current_best` |
| `advisor_default` | `best_checkpoint/advisor_default.pt` | Manual promotion + human review |

### Statistical promotion gate

Raw 55% over 200 games is insufficient — falsely promotes near-equal models often. Promote only if:

```
win_rate >= 55%
AND lower_confidence_bound(win_rate) > 50%
AND no fixed-suite regression beyond tolerance
```

| Gate | Config |
|---|---|
| Fast smoke | 200 games, 55%, informational only |
| Real promotion | 400–800 games, seat-swapped fixed seeds, LCB > 50% |
| Cloud/HOF | LCB > 50–52% |
| Advisor | Gate + human review |

HOF is fed only from `current_best` — one bad promotion contaminates the HOF pool and all future training distribution. SPRT or Elo-confidence is the acceptable implementation.

### Where
`games/kingdomino/self_play.py`, `scripts/promote_checkpoint.py` (with confirmation prompt)

---

## Milestone 7 — Hall-of-Fame Opponent Sampling + Sample Ownership
**Gate: HOF mixing does not destabilize losses; Elo continues improving**

### Mechanism
HOF pool seeded/grown from `current_best`. Each iteration: ~70% self-play, HOF fraction per schedule (ramping to 30%). HOF sampling weighted toward recent entries with diversity across the full pool.

### Sample ownership rules (written before implementation)

| Game type | Train on |
|---|---|
| Current vs current | Both players |
| Current vs HOF | Current player only |
| HOF vs current | Current player only |
| HOF vs HOF | Evaluation / distribution mining only — not training |

The current model learns from positions HOF created (diverse states) with labels from the current model's search (strong targets). Training on HOF policy targets would teach imitation of an older, weaker policy.

### Explicit buffer tracking
Store per position:
```
owner     = "current" | "hof" | "self"
trainable = True | False
```

### New flags
```
--hof_dir             best_checkpoint/hof/
--hof_fraction        0.3
--hof_start_iter      50
--hof_sample_weights  "recency"
```

### Where
`games/kingdomino/self_play.py`, `parallel_self_play.py`

---

## Milestone 8 — Endgame Targeted Position Generation
**Gate: generated positions are legal, reachable, and diverse**

### Goal
Produce novel endgame configurations that fall **outside the self-play distribution** — board shapes, terrain distributions, and crown placements that human-style or adversarial play would create but pure self-play rarely generates. These feed the reanalysis pipeline with diverse states + perfect labels (exact scoring at near-terminal positions).

### Generation approaches (in order of priority)

1. **Mine existing replay buffer** — filter stored positions by `tiles_remaining`, `phase`, or scenario criteria. Free and immediately available. Best starting point.
2. **Adversarial early play** — run self-play games with a randomized/adversarial policy for the first N rounds to diversify board shapes, then save the resulting endgame states. Guarantees legal, reachable positions while producing configurations normal self-play policy would never create.
3. **Programmatic construction** — use `game.py` engine API to place tiles directly. Hard to guarantee diversity but gives precise scenario control.

All positions must be reachable from legal game sequences. An illegal position generator is worse than none.

### Priority scenarios
Final 2–4 tiles, high-crown tile still unseen, harmony almost complete, middle kingdom tension, cramped 7×7 with holes. Endgame suite first; others added after reanalysis infrastructure is stable.

### Where
`scripts/generate_position_suite.py`, `data/kingdomino/position_suites/endgame_v1.jsonl`

---

## Milestone 9 — Replay Reanalysis Infrastructure
**Gate: data format supports reanalysis before first reanalysis run**

Re-label stored positions with stronger search targets from the current model. Allows improving label quality without generating new games — critical given ~16 min/iter generation cost at sims=1600.

### Data format (design now, use later)
```python
{
  "position_id": str,
  "ruleset_hash": str,
  "public_state": dict,        # FULL engine state — not just encoded tensors.
                               # Reanalysis must reconstruct legal actions,
                               # chance futures, and scoring.
  "legal_mask": list,
  "policy_target": list,
  "value_target": float,
  "owner": str,                # current | hof | self
  "trainable": bool,
  "search_metadata": {
    "sims": int,
    "checkpoint_iter": int,
    "alpha": float,
    "exact": bool,
    "search_type": str         # selfplay|pimc|open_loop|exact_endgame|reanalyze
  },
  "checkpoint_provenance": str,
  "reanalysis_count": int
}
```

### Staged reanalysis plan

| Stage | Type | Sims | When |
|---|---|---|---|
| Early | Endgame/terminal-adjacent | Exact | After Milestone 1 |
| Mid | Hard-position suite | 1600–3200 | After Milestone 8 |
| Later | HOF/replay refresh | 3200 | After HOF pool matures |
| Polish | Top-loss / high-disagreement | 3200+ | Final stages |

### New flags
```
--reanalyze_every      25
--reanalyze_positions  50000
--reanalyze_sims       3200
--reanalyze_source     "endgame|hard|hof|mixed"
```

### Where
`games/kingdomino/self_play.py`, `games/kingdomino/replay_buffer.py`

---

## Milestone 10 — Infrastructure Smoke Run (48ch/6b local)
**Gate: diagnostics sane; HOF stable; proceed to cloud**

Run on **48ch/6b locally**, not 80ch/6b. The smoke run validates the infrastructure stack (HOF, dynamic schedules, promotion gating, diagnostics, manifests, reanalysis format) — not the model's ability to reach superhuman play. Running locally on a known architecture means:

- Faster games = more iterations per dollar for infrastructure validation
- Existing strong baseline checkpoints available for HOF seeding (run 4 iter 100, run 6 iter 50)
- Familiar territory — anomalies are easier to detect
- Cheaper to discover infrastructure bugs at 48ch/6b than 80ch/6b
- The gate is "infrastructure works and diagnostics look sane" — not "model achieves new Elo"

### HOF seeding
Seed with best 32ch/4b and best 48ch/6b checkpoints, labeled explicitly as external HOF seeds. Validates HOF mechanics end-to-end before the pool grows organically.

### Success criteria
- Policy loss trending or stable (not climbing uncontrollably)
- win_brier improving vs baseline
- `mcts_lift_rate` positive
- HOF mixing causes no loss spikes
- No promotion-gate regressions
- Diagnostics produce interpretable values
- Elo improving vs 48ch/6b baseline at iter 25

### Config
```
Architecture: 48ch/6b (local)
Sims:         800 → 1600 (schedule)
Alpha:        0.8 → 0.2 (schedule, empirical trigger if ready)
LR:           1e-3 → 3e-4 (schedule)
Games/iter:   140
Iterations:   ~25–30
Buffer:       200k
HOF:          seeded with external 32x4 + 48x6, ramped from iter 15
```

If the smoke run fails, diagnose against calibration diagnostics before any further spend. The specific failure mode will point to which infrastructure component needs fixing.

---

## Milestone 11 — 80ch/6b Staged Cloud Run
**Gate: smoke run passed all criteria**

### Architecture rationale
80ch/6b — width over depth for Kingdomino's complexity. GPU parallelizes width efficiently; depth requires more games AND slower games/s (negative feedback loop).

### Full curriculum

| Phase | Iters | Sims | Alpha | LR | Opponents | Buffer |
|---|---|---|---|---|---|---|
| Warmup | 0–50 | 800 | 0.8 | 1e-3 | Self | 200k |
| Main | 50–200 | 1600 | 0.8→0.2 | 1e-3→2e-4 | Self | 400k |
| Diversify | 200–400 | 1600 | 0.2→0.0 | 2e-4→5e-5 | Self + HOF (ramped to 30%) | 500k |
| Polish | 400+ | 3200 | 0.0 | 5e-5 | Self + HOF 30% | 500k |
| Reanalysis | Ongoing | 3200 | — | — | HOF games relabeled | — |

Alpha transitions triggered empirically by win_brier_by_phase when available; iteration-based schedule as fallback.

### Scale table (at 140 games/iter, measured 1.35 games/s on RTX 5090)

| Games | Iters | Time | Cost |
|---|---|---|---|
| 50k | ~357 | ~10 hrs | ~$4 |
| 200k | ~1,429 | ~41 hrs | ~$17 |
| 500k | ~3,571 | ~103 hrs | ~$43 |
| 1M | ~7,143 | ~206 hrs | ~$87 |

**Throughput caveat:** 1.35 games/s will not hold uniformly as sims rise 800→1600→3200. Rerun `bench_compile.py` and `bench_doublebuffer.py` on the actual RTX 5090 / Ryzen 9600X before committing to long runs. Use phase-specific measured rates for cost estimates, not flat extrapolation.

### Checkpoint strategy
- Promote to `current_best` every 10 iters if gate passes (LCB > 50%)
- Add to HOF every 50 iters from `current_best`
- Commit `elo_db.json` and `elo_games.jsonl` to repo after run
- Update `advisor_default` only after human review of advisor quality

### Hardware
RTX 5090, Ryzen 9600X, ~$0.42/hr on-demand.

---

## Milestone 12 — Rust Acceleration
**Priority: adjudication and advisor; training is already Rust**

Training self-play already uses the Rust `BatchedMCTS` engine via PyO3. The throughput of 0.18–0.19 games/s is already the Rust path.

### Remaining gaps

**Advisor (primary):** Single-position open-loop Rust search (`RustOpenLoopMCTS`). Advisor currently uses Python `OpenLoopMCTS` — ~4-5s at 800 sims. Rust path: <1s. Enables higher sim counts for better recommendation quality.

**High-sim adjudication (secondary):** Reanalysis and promotion gating at sims=3200 on many positions. Python path at 3200 sims is prohibitively slow at scale. Rust single-position high-sim search makes adjudication and reanalysis feasible.

**Concurrent training (tertiary, cloud scale):** At cloud scale with multiple CPU cores (Ryzen 9600X has 6), Option 1 concurrent training (overlap training with next iteration's generation at iteration boundaries) reduces wall-clock time with minimal code change. Full async (Leela-style, separate training and self-play networks synced periodically) is worth considering for the Polish phase. True concurrent (shared weights, continuous updates) introduces staleness risks not worth taking on.

---

## Appendix A — Key Decisions

| Decision | Rationale |
|---|---|
| No human game data | AlphaZero paper: removing it enabled stronger play |
| Alpha=0.0 for evaluation | Confirmed best Elo in sweep; advisor uses this |
| Alpha=0.8 early training | Win head needs dense gradient to calibrate |
| Alpha transitions post-buffer-flush | Abrupt switch destabilizes training; wait one staleness window |
| score_scale=160 | Confirmed from checkpoint config |
| Public-information-safe search required | All paths sample only from public-consistent remaining bag |
| terminal_search_value() single source | All terminal backups go through this |
| Promotion: LCB > 50% (not raw 55%) | Sampling noise at 200 games falsely promotes near-equal models; HOF contamination is unrecoverable |
| HOF fed only from current_best | Prevents regressed checkpoints entering training distribution |
| HOF sample ownership | Current-player-only for mixed games; HOF-vs-HOF for eval only |
| Architecture invariants unverified until Milestone 0 | Print and assert from active branch before hard-coding |
| Smoke run on 48ch/6b not 80ch/6b | Infrastructure validation; cheaper to find bugs locally |
| Training is already Rust | BatchedMCTS via PyO3 — Milestone 12 is advisor + adjudication only |

---

## Appendix B — Run History

| Run | Arch | Iters | Peak Elo | Notes |
|---|---|---|---|---|
| cloud_iter100 | 32ch/4b | 100 | ~454 | First cloud run |
| local_cont_iter100 | 32ch/4b | 100 | ~854 | Local continuation |
| local_cont5_lr2e4_iter055 | 32ch/4b | 55 | ~1009 | Best 32x4 |
| 32ch/4b ceiling | — | — | ~1029–1046 | Confirmed saturated |
| local_48x6_run2 | 48ch/6b | 100 | 886 | |
| local_48x6_run3 | 48ch/6b | 100 | 974 | lr=1e-3, buffer=400k |
| local_48x6_run4 | 48ch/6b | 100 | 1005 | lr=5e-4, warm start run3 |
| local_48x6_run6 | 48ch/6b | 50 | 977 | Cold start, lr=3e-4, 200k buffer |

## Appendix C — Buffer Sizing Guide

| Phase | Buffer | Rationale |
|---|---|---|
| Cold start short run | 200k | Fills ~iter 27; keeps data fresh while model changes fast |
| Warm continuation | 300–400k | Model more stable; older data still useful |
| Cloud main phase | 400–500k | Longer run; larger staleness window acceptable |
| HOF phase | 500k | HOF games add diversity; need capacity for both self-play and HOF data |

Buffer size should track model stability. Fast-changing model = small buffer for freshness. Stable/converging model = large buffer for diversity.
