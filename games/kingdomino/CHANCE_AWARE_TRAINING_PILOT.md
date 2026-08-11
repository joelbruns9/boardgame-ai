# Chance-aware training pilot: deck=8 and deck=12

Status: the all-slot deck-8/deck-12 pilot, exploratory analysis, confirmatory
match, reproducibility audits, and training-aligned search diagnostic are
complete. No checkpoint has been promoted.

## Objective

Warm-start the current best Kingdomino network and fine-tune it on policy
targets produced by explicit sampled chance splitting. This is the first test of
whether the model can absorb the search improvement; it is not another
current-network search ablation.

## Frozen treatment

The treatment applies only when a batched open-loop self-play search starts at:

- `PLACE_AND_SELECT`;
- a public hidden-deck count of either 8 or 12.

All four real selection roots (`actor_index` 0 through 3) are treated. Every
self-play move is a new MCTS root and a normal training example; restricting the
treatment to slot 0 would leave slots 1-3 learning the historical aliased
open-loop policy even though the same reveal uncertainty is relevant there.

All other nonterminal searches remain aliased open loop. Deck=4 and deck=0
remain routed to the existing exact endgame solver.

At an eligible root, each simulation keeps the hidden order unread until PUCT
selects the action that triggers the next reveal. That action becomes an
afterstate/chance node. Public reveal rows route to separate observation
subtrees, with:

- complete fixed row identity: `C(8,4)=70` or `C(12,4)=495` support entries;
- lazy allocation of observation subtrees;
- balanced sampled traversal of that support;
- ordinary sampled backup;
- zero panel bootstrap evaluations and zero initialization network rows; and
- the unchanged per-move simulation budget.

The complete support is routing metadata, not enumeration work for the neural
network. This carries forward the sampled-split mechanism whose deck=8 policy
matched the exhaustive panel while avoiding its 70-row inference cost.

Deck=12 is a deliberate extrapolation. It has not received the deck=8 forced-
move quality validation, but it is the closest earlier chance boundary and
provides four additional treated roots per ordinary game. Deck=8 and deck=12
accounting must remain separate in the training log, including counts for root
selection slots 0-3.

## Replay treatment

Every recorded example carries:

- whether at least one simulation actually crossed an explicit chance split;
- the hidden-deck count at its root; and
- the root turn slot used for per-slot observability.

Eligible roots that never reach the chance boundary are not labeled treated.
The default replay behavior remains unchanged. This pilot opts into a 2x
sampling weight for treated examples. Expanding from two to eight eligible
roots per game supplies roughly four times as many natural targets, so the
original provisional 4x weight would overconcentrate optimizer batches. Under
the incumbent 25% full-search recording regime, the expected natural treated
share is about 9% and 2x weighting raises the draw share to roughly 17%.
When an example also qualifies for the
existing exact-endgame weight, the sampler uses the larger weight rather than
multiplying them.

The learner warm-starts from the current best checkpoint but begins with a fresh
replay buffer. Loading an old open-loop buffer would dilute the sparse changed
targets and make the treatment harder to interpret.

## Required observability

The iteration log and compact console summary must report:

- configured sampled-split deck counts;
- treated recorded examples at deck=8 and deck=12;
- treated recorded examples by root selection slot 0-3 at both deck counts;
- simulation paths that crossed an explicit split at each deck count;
- zero exhaustive bootstrap rows for this treatment;
- treated fraction in the replay buffer; and
- treated fraction actually drawn into training batches.

The existing exhaustive `deck8_chance_enumeration` mode remains available for
historical evaluation, but construction must reject enabling it together with
sampled training splits.

## Run boundary

Implementation verification may use short smoke games and must confirm both
deck counts are reached and tagged. The full warm-start training command will
be recorded here after the checkpoint configuration and measured smoke
throughput are known. Do not replace `current_best` or launch promotion from a
smoke run.

## Implementation record

The implementation is opt-in and leaves historical defaults unchanged.

- Rust `BatchedMCTS` accepts a `sampled_chance_split_deck_mask`. Version 1
  rejects bits other than 8 and 12, requires open-loop search, and rejects
  simultaneous exhaustive deck-8 panels. Every `PLACE_AND_SELECT` root at an
  enabled deck count is eligible, independent of selection slot.
- The production open-loop descent reuses the tested full-support chance-node
  machinery with balanced traversal and sampled backup, but sets
  `bootstrap_full_panel=false`. The evaluator therefore receives only normal
  search leaves; deck-12's 495 rows are identity/routing records, not 495
  network requests.
- A move is marked treated only when `chance_step` was returned for at least
  one simulation. Root eligibility without a crossing remains untreated.
- The Python example tuple stays at 12 outer elements so the HOF actor filter
  remains `tuple[-1]`. The three provenance values are appended inside
  `root_stats`; conversion accepts both the new six-field and legacy
  three-field forms.
- HOF batched games use the sampled treatment when configured. Their search
  and path counters are included in iteration totals, while treated-example
  counts still include only learner-owned labels.
- Replay weights are `max(endgame_weight, chance_weight)`. The buffer records
  actual treated draws around the optimizer block rather than estimating the
  rate from a separate sample.

Primary files:

- `games/kingdomino/kingdomino_rust/src/lib.rs`
- `games/kingdomino/self_play.py`
- `games/kingdomino/tests/test_chance_aware_training.py`

## Verification record

Completed:

- `cargo fmt`
- `cargo check`
- Python syntax compilation of the core `self_play.py` wiring before the final
  logging/test additions (the focused pytest command below remains the final
  Python verification)
- release rebuild/install of `kingdomino_rust` from this feature worktree
- `git diff --check`
- focused chance/production suites: **11 passed**
- existing replay-oversampling regressions: **2 passed**

Commands:

```powershell
C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m pytest games\kingdomino\tests\test_chance_aware_training.py games\kingdomino\tests\test_deck8_chance_production.py -q
C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m pytest games\kingdomino\test_endgame_exact.py -k oversample -q
```

The routing smoke initially used a zero-logit evaluator. At 800 simulations its
maximally diffuse policy did not reach deck 8, reproducing the depth-fragmentation
risk rather than finding a routing defect. The mechanical smoke now uses a
deterministic concentrated mock policy, because its purpose is to exercise
routing and tagging. Realistic coverage was then checked separately with the
actual current-best network.

The original four-game, zero-training current-best preflight at 1,600
simulations passed for the then-current slot-0-only treatment:

- deck-8 treated searches/examples: 4/4;
- deck-12 treated searches/examples: 4/4;
- crossed paths: 5,506 at deck 8 and 5,098 at deck 12;
- exhaustive/bootstrap rows: 0;
- treated replay examples: 8/208 (3.85%); and
- wall time: 54.5 seconds with only four slots, at 0.079 games/second.

The structured result is
`runs/kingdomino/chance_correct_a1/chance_training_preflight_s1600_4games.jsonl`.
It contains all required deck-specific fields. Therefore the planned network
does reach both split boundaries comfortably; this treatment is not merely
eligible on paper.

The refreshed all-slot preflight used the same four games and current-best
network at 1,600 simulations:

- deck-8 treated examples: 16, exactly `[4, 4, 4, 4]` across slots 0-3;
- deck-12 treated examples: 16, exactly `[4, 4, 4, 4]` across slots 0-3;
- crossed paths: 24,361 at deck 8 and 23,804 at deck 12;
- exhaustive/bootstrap rows: 0;
- maximum evaluator batch: 24, equal to four slots times leaf batch 6;
- treated replay examples under full recording: 32/208 (15.38%); and
- wall time: 61.0 seconds, versus 54.5 seconds for slot 0 only.

The added cost is tree routing/separation, not network inference rows. The
structured result is
`runs/kingdomino/chance_correct_a1/chance_training_preflight_all_slots_s1600_4games.jsonl`.
The launch uses the incumbent's 25% full-search recording, so only roughly two
of eight eligible labels per game enter replay; combined with always-recorded
exact endgame labels, the projected natural treated share is about 9% and its
2x weighted draw share is about 17%.

### End-to-end training smoke

A dedicated four-game, one-optimizer-step CUDA smoke completed successfully in
`runs/kingdomino/chance_aware_deck8_12_train_smoke`:

- wall time: 65.1 seconds;
- treated labels: 16 at deck 8 and 16 at deck 12;
- per-slot labels: `[4, 4, 4, 4]` at both deck counts;
- crossed paths: 23,394 at deck 8 and 22,162 at deck 12;
- bootstrap rows: 0;
- buffer treatment share: 32/208 (15.38%) under full recording;
- actual 2x-weighted optimizer draw: 71/256 (27.73%), consistent with the
  expected `2 * 15.38% / (1 + 15.38%) = 26.67%` up to batch noise;
- one optimizer step completed with finite losses and gradient norms;
- checkpoint, training log, run manifest, exact-fallback sidecar, and replay
  buffer were written successfully; and
- checkpoint and buffer readback preserved `sampled_chance_split_decks="8,12"`,
  weight `2.0`, and balanced per-slot provenance.

The global current-best checkpoint remained unchanged at SHA-256
`4bf07b0ca14e5452e6533a9232967e89bb0ab0df88c99e9928a65f402b1f04b3`.

Acceptance checks satisfied:

- all focused tests pass;
- the one-game sampled smoke reports nonzero searches and crossed paths for
  both deck 8 and deck 12;
- both deck counts produce treated examples in all root slots 0-3;
- `deck8_chance_bootstrap_rows == 0`; and
- the largest evaluator batch does not grow beyond `leaf_batch` in the sampled
  smoke.

## Proposed first training run

Loading the checkpoint configuration showed that current-best is an 80-channel,
6-block, bilinear-dimension-64 network trained with 4,800 full simulations,
86 slots, leaf batch 6, move-level playout caps (25% full; 200 fast), 400 games,
300 optimizer steps, a 200k buffer, fixed 1e-4 learning rate, and no endgame
oversampling. Those settings supersede the earlier provisional 1,600-simulation
command. Although 1,600 already reaches both chance nodes, reducing label search
from the incumbent's 4,800 would introduce an unnecessary quality confound.

The pilot preserves the incumbent search/replay regime and changes the intended
treatment: sampled splits at all deck-8/12 selection roots plus 2x
treated-example replay weight. With eight eligible roots per game and 25%
recorded full searches, 400 games should produce roughly 800 treated labels per
iteration before HOF/trajectory variation. The expected natural treated share
is about 9%; 2x weighting should put roughly 17% of optimizer draws on treated
examples while retaining broad whole-game training.

Ten iterations make this a bounded first fine-tune. The learner is the latest
generator so the mechanism actually operates in the self-play feedback loop.
HOF games, promotion, and Elo are omitted from this first run: they would add
opponent and evaluation changes before we know whether the network absorbs the
new targets. The global current-best file remains read-only.

```powershell
C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m games.kingdomino.self_play --engine batched_open_loop --device cuda --async_solve --game_cpus 4 --warm_start_current_best --current_best_path runs\kingdomino\best_checkpoint\current_best.pt --selfplay_generator_mode latest --sampled_chance_split_decks 8,12 --chance_split_oversample 2.0 --iterations 10 --games_per_iter 400 --train_steps 300 --sims 4800 --playout_cap_randomization --full_search_fraction 0.25 --fast_move_sims 200 --fast_move_dirichlet_epsilon 0.0 --fast_move_temp_moves 0 --channels 80 --blocks 6 --bilinear_dim 64 --batch_slots 86 --leaf_batch 6 --virtual_loss 1 --batch_size 256 --lr 1e-4 --weight_decay 1e-4 --buffer 200000 --min_buffer 5000 --lambda_score 0.5 --lambda_w 0.25 --score_scale 160.0 --policy_weight 1.0 --grad_clip 1.0 --margin_gain 2.0 --alpha 0.5 --c_puct 1.5 --temp_moves 30 --fpu -0.2 --exact_endgame_max_secs 3.0 --endgame_oversample 1.0 --policy_target_pruning --benchmark_every 0 --elo_every 0 --checkpoint_dir runs\kingdomino\chance_aware_deck8_12_pilot --save_buffer runs\kingdomino\chance_aware_deck8_12_pilot\buffer_final.pkl --buffer_autosave_every 1 --exact_fallback_positions runs\kingdomino\chance_aware_deck8_12_pilot\exact_fallback_positions.jsonl --seed 20260810
```

This command intentionally omits `--warm_buffer`, HOF games, soft-gate
generation, promotion, and Elo. It writes only to a new pilot directory and
cannot replace `current_best`. After iteration 1, verify that treated buffer
share remains near 9%, treated optimizer draws are near the predicted 17%,
both deck path counts are nonzero, and bootstrap rows remain zero. Strength
evaluation should compare saved pilot checkpoints against the unchanged
current-best checkpoint after training.

## Safe overnight stopping and recovery

The launch writes `iter_NNNN.pt` after every completed iteration and uses
`--buffer_autosave_every 1`. Buffer autosaves use a temporary file plus atomic
replacement, so a hard kill during a save leaves the previous buffer intact.
The JSONL log is appended once per completed iteration.

The preferred stop mechanism is a `STOP` file, not Task Manager or repeated
Ctrl+C:

```powershell
New-Item -ItemType File runs\kingdomino\chance_aware_deck8_12_pilot\STOP
```

The loop consumes this file at its next safe check, finishes the current
self-play/training/checkpoint/log/autosave phase, and exits through the final
buffer save. Because Rust search releases the Python GIL, Ctrl+C delivery can
lag while a batched search is running; the STOP file is deterministic but may
also wait for the current iteration to finish.

If the process is force-killed, every fully completed iteration remains on
disk. At worst, work and replay examples from the in-progress iteration are
lost. The prior atomic buffer autosave, earlier checkpoints, manifest, and
earlier JSONL rows remain usable. Resume into a new continuation directory
using the last intact checkpoint plus the last autosaved buffer; do not reuse
the original directory because iteration numbering restarts at 1.

## Completed pilot run

The proposed command above ran from 2026-08-11 00:15 to 08:01 America/Chicago
and completed all ten iterations without stderr output. Artifacts are under
`runs/kingdomino/chance_aware_deck8_12_pilot/`.

- 4,000 self-play games and 3,000 optimizer steps completed.
- `iter_0001.pt` through `iter_0010.pt` were written successfully.
- The final replay buffer contains 87,903 examples, including 8,071 treated
  examples (9.18%).
- Treated optimizer draws stayed between 16.6% and 17.2%, matching the
  preregistered approximately 17% expectation for 2x weighting.
- Deck-8 and deck-12 labels remained approximately balanced, as did slots
  0-3 within each deck count.
- Every iteration recorded millions of paths crossing each enabled chance
  boundary and zero bootstrap rows.
- Search throughput remained stable at approximately 0.14-0.15 games/second.
- The global `current_best.pt` was not changed.

The training losses are not a strength test. Policy loss rose from 1.344 in
iteration 1 to 1.414 in iteration 10, and win Brier loss rose from 0.0769 to
0.0869. The learner generated progressively changing targets and the replay
buffer aged throughout the run, so those changes cannot be interpreted as a
playing-strength regression without an external comparison.

## Initial exploratory analysis

All analysis below is exploratory. The two game screens use only 32 independent
same-deck, seat-swapped pairs per checkpoint. They are useful for detecting a
large collapse and for estimating direction, but they are not sufficiently
powered to rank nearby checkpoints or establish promotion.

### Frozen 17-position regression suite

The frozen suite contains nine exact-valued endgame positions. Lower exact
value MAE is better.

| Checkpoint | Exact-value MAE | Mean policy entropy | Mean top-action probability |
| --- | ---: | ---: | ---: |
| starting `current_best` | 0.16572 | 1.34264 | 0.53042 |
| `iter_0010` | 0.16333 | 1.38416 | 0.51293 |

Iteration 10 therefore showed no exact-value regression on this small suite.
The 0.00238 MAE improvement is too small and the suite too small to claim a
value improvement. The final policy is less peaked, particularly on the four
opening positions, so policy entropy alone is direction-free.

Review artifacts:

- `runs/kingdomino/chance_aware_deck8_12_pilot/analysis/fixed_suite_current_best.json`
- `runs/kingdomino/chance_aware_deck8_12_pilot/analysis/fixed_suite_iter10.json`

### Held-out deck-8 policy absorption

This check reused the 64 deck-8 positions frozen before training from seed
20260830. Training used seed 20260810. The references were the already-computed
6,400-simulation control and sampled-split visit policies produced by the
unchanged starting network. No new search labels were generated after seeing
the trained checkpoint.

Across all 64 positions, iteration 10 was essentially unchanged in absolute
KL to the old sampled-split target: 0.73010 to 0.73040. Its total variation to
that target changed from 0.45532 to 0.46645. Neither is evidence that it learned
the full old target distribution.

The more specific disagreement test did move in the intended direction. On the
33 positions where control and sampled-split search selected different top
actions:

- the raw-network probability margin `P(split top) - P(control top)` moved from
  -0.0577 to +0.0138;
- the paired mean change was +0.0716, with a position-bootstrap 95% interval of
  [+0.0241, +0.1219];
- 25 positions moved toward the split preference and 8 moved away; and
- raw top actions changed from 9 split / 16 control / 8 other to 11 split /
  11 control / 11 other.

The split-specific relative-fit measure, `KL(control target) - KL(split
target)`, moved from -0.1260 to +0.0562. Its paired change was +0.1822 with a
position-bootstrap 95% interval of [+0.0845, +0.2960]. This is evidence that
the network absorbed a chance-aware preference signal relative to the aliased
control. It is not evidence that the final network fully imitated the frozen
split policy: absolute split KL and top-action agreement did not improve.
That distinction matters because the self-play targets used 4,800 simulations
and evolved with the learner, while this frozen diagnostic used the starting
network at 6,400 simulations.

### Exploratory paired game screens

Both screens used the unchanged batched open-loop evaluation engine, 400
simulations per move, no search noise, 32 common deck seeds, and seat swapping
for 64 games. The strict promotion calculation treats each seat-pair, not each
game, as one independent observation.

| Candidate | W-L-D | Game score | Mean score margin | Pair W-L-D | Pair-score LCB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `iter_0010` | 34-30-0 | 53.1% | +0.30 | 7-5-20 | 36.4% |
| `iter_0002` | 39-25-0 | 60.9% | +4.41 | 10-3-19 | 43.7% |

Review artifacts:

- `runs/kingdomino/chance_aware_deck8_12_pilot/analysis/iter10_vs_current_best_64g_s400.json`
- `runs/kingdomino/chance_aware_deck8_12_pilot/analysis/iter02_vs_current_best_64g_s400.json`

Neither checkpoint clears the 50% pair-level lower-confidence promotion gate.
Iteration 2's larger observed result does not establish that it is stronger
than iteration 10: both estimates are based on only 32 pairs, and iteration 2
was selected for a second screen after inspecting checkpoint diagnostics.
Treating the two 64-game point estimates as a checkpoint ranking would be a
multiple-comparison/selection error.

### Current conclusion and boundary

The pilot succeeded mechanically and produced an observable network-level
shift toward sampled-split preferences on held-out deck-8 disagreement
positions. Neither the fixed suite nor the two small paired screens show an
obvious regression; both game point estimates are positive. However, the
current evidence does not establish a playing-strength gain, identify the
strongest iteration, or justify promotion.

At this stage, evaluation was deliberately stopped after the iteration-2
screen. The subsequent confirmatory strength test below chose one checkpoint
and sample size before looking at a fresh paired-seed block; it did not reuse
the 32 exploratory pairs to select and validate the same candidate.

## Iteration-2 confirmatory match

Iteration 2 was frozen as the sole candidate before a new match. The complete
design and decision rule were written to
`runs/kingdomino/chance_aware_deck8_12_pilot/analysis/ITER02_CONFIRMATION_PREREGISTRATION.md`
before launch. The match used 256 previously unused deck seeds, two seat-swapped
games per seed, the unchanged open-loop evaluator at 400 simulations, and no
optional stopping. No other checkpoint was evaluated on this seed block.

The full 512-game result was:

| Candidate | W-L-D | Game score | Mean score margin | Pair W-L-D | Pair score | Pair-score 95% Wilson interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `iter_0002` | 219-293-0 | 42.77% | -2.81 | 29-66-161 | 42.77% | [36.86%, 48.90%] |

This meets the preregistered confirmed-regression rule: the pair-score upper
confidence bound is below 50% and the mean score margin is negative. The prior
39-25 result on 32 exploratory pairs did not replicate. It was a small-sample
positive fluctuation and cannot support promotion or a claim that iteration 2
was the strongest checkpoint.

This result does not erase the held-out policy evidence that the network moved
toward chance-aware preferences relative to aliased control. It establishes
that, for iteration 2, that policy shift did not produce stronger whole-game
play under the incumbent 400-simulation open-loop evaluator. No checkpoint was
promoted and `current_best.pt` remained unchanged.

Result artifact:

- `runs/kingdomino/chance_aware_deck8_12_pilot/analysis/iter02_confirm_vs_current_best_512g_s400_seed20310000.json`

### Original-block reproducibility audit

The original 32-pair iteration-2 screen was rerun with the same seed block and
all settings unchanged. The rerun reproduced the original match object exactly:

- games: 39-25-0;
- mean margin: +4.40625;
- pairs: 10 wins, 3 losses, 19 ties;
- pair score: 19.5 / 32; and
- pair-score lower confidence bound: 43.7493%.

Candidate hash, baseline hash, gate configuration, and every stored match field
were identical. This clears ordinary run-to-run nondeterminism as the cause of
the reversal. It does not yet test whether embedding the same seeds in a batch
that replenishes its 32 slots changes results; that separate audit was not run.

Audit artifact:

- `runs/kingdomino/chance_aware_deck8_12_pilot/analysis/iter02_original64_exact_reproduction_audit.json`

### Batch-embedding reproducibility audit

The original 32-seed block was then evaluated twice with per-game score
capture in both seat orientations:

1. as a standalone 32-game orientation, exactly filling the 32 search slots;
2. as the first 32 seeds of a 256-game orientation whose slots were repeatedly
   replenished.

All 64 shared seed/orientation records had identical player scores and official
winners. Mismatch count was zero. The embedded prefix also reproduced the
39-25 game result, +4.40625 mean margin, and 10-3-19 pair result exactly.

This clears game-count-dependent slot replenishment and inference batching as
the cause of the exploratory/confirmation reversal. Together with the exact
standalone rerun, the audit establishes that the evaluator is deterministic for
the original block and invariant to embedding those games in the larger
schedule. The materially larger, preregistered 256-pair result therefore remains
the appropriate strength estimate; the original 32-pair block was an unusually
favorable seed sample.

Audit implementation and artifact:

- `runs/kingdomino/chance_aware_deck8_12_pilot/analysis/audit_batch_embedding.py`
- `runs/kingdomino/chance_aware_deck8_12_pilot/analysis/iter02_batch_embedding_audit_seed20260811.json`

## Training-aligned search diagnostic

One remaining explanation for the confirmed regression was an inference
mismatch: iteration 2 learned partly from deck-8/deck-12 sampled-split targets
but the strength matches graded it with aliased open-loop search. This was
tested without further training. Both iteration 2 and `current_best` received
the same sampled chance splitting at every deck-8 and deck-12 selection root;
all other match settings and the 256 confirmation seeds were unchanged.

The design and interpretation rule were frozen in
`runs/kingdomino/chance_aware_deck8_12_pilot/analysis/ALIGNED_SEARCH_MATCH_PREREGISTRATION.md`
before a disjoint mechanical smoke and the full run. The smoke confirmed both
deck boundaries were exercised in both seat orientations, with zero bootstrap
rows and zero exhaustive panels.

The complete 512-game aligned-search result was:

| Evaluation search | W-L-D | Game score | Mean score margin | Pair W-L-D | Pair score | Pair-score 95% Wilson interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shared open loop | 219-293-0 | 42.77% | -2.81 | 29-66-161 | 42.77% | [36.86%, 48.90%] |
| shared deck-8/deck-12 split | 218-294-0 | 42.58% | -3.24 | 32-70-154 | 42.58% | [36.67%, 48.70%] |

The aligned upper bound is below 50%, satisfying the preregistered
`not_rescued` rule. The full artifact contains exactly 512 games, two
orientations for every seed from 20310000 through 20310255, balanced candidate
seats, the frozen checkpoint hashes, nonzero deck-8/deck-12 split activity in
both orientations, and zero bootstrap rows.

The open-loop evaluator was therefore not the cause of iteration 2's apparent
weakness. Under search semantics matched to its treated labels, its result is
essentially the same confirmed regression. No checkpoint was promoted and the
global `current_best.pt` remains unchanged.

This conclusion is intentionally checkpoint-specific. It does not establish
that sampled splitting is a bad search mechanism; earlier forced-move evidence
still says it can improve selected decisions. It establishes that this pilot's
iteration-2 weights did not convert that mechanism into whole-game strength,
and aligned inference cannot recover the loss. Distinguishing replay weighting,
self-play distribution shift, target nonstationarity, and general fine-tuning
dynamics would require a different controlled training experiment, which was
not run here.

Result artifact:

- `runs/kingdomino/chance_aware_deck8_12_pilot/analysis/iter02_vs_current_best_aligned_deck8_12_512g_s400_seed20310000.json`

## Matched open-loop training control

The next experiment tested whether the iteration-2 regression required the
chance-aware treatment at all. A control was warm-started from the identical
`current_best.pt`, used training seed 20260810, and matched the pilot's first
two iterations: 400 games and 300 optimizer steps per iteration, the same
4,800/200 playout-cap schedule, fresh buffer, architecture, augmentation,
optimizer, exact-endgame behavior, and latest-generator feedback loop. It
removed deck-8/deck-12 sampled splitting and used the inactive 1x chance replay
weight.

The design and a fresh evaluation block were frozen before training in
`runs/kingdomino/open_loop_matched_control_2iter/analysis/MATCHED_CONTROL_PREREGISTRATION.md`.
Field-by-field manifest comparison found no material differences beyond the
two treatment settings, two-versus-ten iteration boundary, and isolated output
paths.

Training completed in 100.2 minutes:

- 800 games, 600 optimizer steps, and checkpoints for both iterations;
- final fresh-buffer size 17,584;
- zero sampled-split searches, paths, examples, or replay draws in both
  iterations; and
- zero exhaustive panels and bootstrap rows.

Control iteration 2 was then evaluated against unchanged `current_best` on 256
new seat-swapped seed pairs (20320000-20320255) under shared 400-simulation
open-loop search:

| Candidate | W-L-D | Game score | Mean score margin | Pair W-L-D | Pair score | Pair-score 95% Wilson interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| open-loop control `iter_0002` | 232-280-0 | 45.31% | -2.24 | 33-57-166 | 45.31% | [39.33%, 51.43%] |

The negative point estimate and margin did not meet the preregistered confirmed
regression rule because the upper bound remains above 50%. The result is
`inconclusive`, not evidence of parity.

For comparison, chance-aware iteration 2 scored 42.58%, -3.24 margin, with a
[36.67%, 48.70%] interval on a different frozen seed block. The intervals
overlap substantially. Their 2.73-point difference is not paired and cannot
establish treatment harm. The control's own negative direction makes generic
short warm-start fine-tuning from a fresh buffer a credible explanation for a
substantial part of the pilot regression; chance-specific additional harm
remains possible but unproven.

No checkpoint was promoted. Control iteration 2 SHA-256 is
`d4e35bfc7ed8e1f591054b23b24137e6ff0d3f0de4bcd95669d29dc535700a34`,
and `current_best.pt` remains
`4bf07b0ca14e5452e6533a9232967e89bb0ab0df88c99e9928a65f402b1f04b3`.

Artifacts:

- `runs/kingdomino/open_loop_matched_control_2iter/iter_0002.pt`
- `runs/kingdomino/open_loop_matched_control_2iter/training_log.jsonl`
- `runs/kingdomino/open_loop_matched_control_2iter/analysis/iter02_vs_current_best_512g_s400_seed20320000.json`
