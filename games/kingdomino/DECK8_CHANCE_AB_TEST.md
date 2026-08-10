# Deck=8 chance-panel whole-game A/B

## Question

Does the production deck=8 exhaustive reveal panel improve the playing strength
of the frozen current-best Kingdomino network when total move-search work is
held to 400 units?

This is a search-mechanism screen, not a training run or model-promotion gate.
No weights are changed.

## Frozen design

- Checkpoint: `runs/kingdomino/best_checkpoint/current_best.pt`.
- Engine: Rust `BatchedMCTS`, open-loop, greedy evaluation play.
- Move budget: 400 units for both agents.
- Treatment accounting: at most 70 units for one complete deck=8 panel; the
  remaining units are real search paths.
- Control accounting: all 400 units remain ordinary search paths.
- Shared settings come from the checkpoint where applicable: `c_puct=1.5`,
  `fpu=-0.2`, `score_scale=160`, `margin_gain=2`, `alpha=0.5`, and a three-second
  exact endgame budget.
- No Dirichlet noise and zero move temperature.
- CUDA AMP inference, leaf batch 6, 32 game slots, four solver CPUs.

The strength match uses 64 paired deck seeds, or 128 games. For each seed the
same checkpoint plays twice:

1. treatment in seat 0, control in seat 1;
2. control in seat 0, treatment in seat 1.

The sealed strength seed block starts at `2026082200`. The smoke used the
disjoint block starting at `2026081200`; because smoke outcomes are visible,
none of those seeds may enter the full comparison.

The new engine seat selector affects only evaluation matches. Normal self-play
retains the existing default in which enabling the flag applies it to both
seats. The match artifact must show nonzero panel counts and exactly 70
bootstrap rows per panel.

## Metrics and interpretation

Primary strength metrics are paired treatment points and paired raw-score
margin. Each seed's two seat-swapped games are averaged before a deterministic
20,000-resample paired-seed bootstrap. Also report overall decisive win rate,
draws, and seat-specific results.

- Positive screen: both point estimates favor treatment, at least one 95%
  interval excludes its null in the favorable direction, neither seat shows a
  material reversal, and all accounting checks pass.
- Negative screen: both point estimates favor control, with stronger evidence
  if either upper interval excludes its null.
- Otherwise: inconclusive. Do not tune the mechanism after viewing this run;
  increase independent paired seeds in a separately preregistered follow-up.

Because 64 paired seeds detect only a fairly large effect, an interval crossing
the null is expected to remain inconclusive rather than being described as
evidence of equivalence.

## Throughput measurement

A separate 16-game all-off cohort and 16-game all-on cohort use identical deck
seeds. These symmetric games are cost measurements only. Report wall time,
games/second, inference rows and calls, batch-size distribution, peak allocated
GPU memory, panel counts, and budget-blocked attempts. They are not strength
evidence.

## Execution stages

1. Compile and test the seat selector and pure paired analysis.
2. Run a two-paired-seed/two-throughput-game smoke to validate provenance,
   accounting, output atomicity, and estimated runtime.
3. Commit the frozen harness before the full run.
4. Run the 64-paired-seed sealed comparison once without inspecting partial
   game outcomes.
5. Validate the final JSON and record the result in this document.

Planned full command:

```powershell
$env:PYTHONIOENCODING='utf-8'
C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe `
  -m games.kingdomino.deck8_chance_ab `
  --output runs/kingdomino/chance_correct_a1/deck8_production_ab_v1.json `
  --seed 2026082200 `
  --paired-seeds 64 --throughput-games 16 --sims 400 `
  --batch-slots 32 --leaf-batch 6 --device cuda --amp-inference `
  --solver-cpus 4
```

## Result

**Valid weak-negative screen; do not enable the treatment in self-play.** Both
preregistered point estimates favored control, so the run meets the weak form
of the negative-screen rule. Neither interval excluded its null, however, so
this is not evidence that the treatment is confidently harmful or equivalent
to control.

The sealed run completed 64 paired seeds / 128 games in the strength match:

| Metric | Result | Paired-seed bootstrap 95% interval |
|---|---:|---:|
| Treatment wins - control wins - draws | 63 - 65 - 0 | — |
| Treatment points rate | 49.22% | 46.88% to 51.56% |
| Treatment decisive win rate | 49.22% | descriptive |
| Treatment raw-score margin | -0.766 | -1.625 to +0.055 |

The seat-swapped pairing was load-bearing. Treatment went 24-40 with mean
margin -5.95 in seat 0 and 39-25 with mean margin +4.42 in seat 1. At the paired
seed level, 61/64 seeds split one official win apiece, one seed was a treatment
sweep, and two were control sweeps. Paired raw-score margin was positive on 14
seeds, zero on 25, and negative on 25. The paired aggregate therefore removes a
large seat effect; the unpaired seat records must not be read as a mechanism
interaction.

Each 64-game strength orientation committed 128 panels (8,960 bootstrap rows),
with zero budget-blocked attempts. Both orientations evaluated exactly
1,029,768 network rows, and their exact-solver counts and fallbacks also matched
(760 and 4). This rules out panel admission, total evaluator work, or endgame
fallback imbalance as explanations for the result.

The identical-seed throughput cohorts produced:

| Metric | All off | All on | Treatment/control |
|---|---:|---:|---:|
| Games | 16 | 16 | — |
| Wall time | 63.18 s | 70.91 s | 1.122x |
| Games/second | 0.2533 | 0.2256 | 0.891x |
| Inference calls | 2,857 | 2,945 | 1.031x |
| Inference rows | 257,442 | 258,244 | 1.003x |
| Maximum batch | 96 | 1,216 | 12.67x |
| Peak allocated GPU memory | 69.8 MB | 712.7 MB | 10.2x |
| Panels / bootstrap rows | 0 / 0 | 64 / 4,480 | — |
| Budget-blocked attempts | 0 | 0 | — |

The small inference-row difference is caused by different game trajectories;
within every searched move the arms shared the same 400-unit budget. The
throughput result is a cost measurement, not strength evidence.

The complete atomic artifact is
`runs/kingdomino/chance_correct_a1/deck8_production_ab_v1.json`, SHA-256
`9bb989d9f29cc72591de16f238ca2f27cfea83b06e6383c6433e0f457945df5c`.
It records checkpoint SHA-256
`4bf07b0ca14e5452e6533a9232967e89bb0ab0df88c99e9928a65f402b1f04b3`,
source commit `c068e2161dff22426f5bff3d6ddb2fb97d904565`, source hashes, every raw
game, batch histograms, and all diagnostics above. Hash and accounting
validation passed after the run.

Decision: keep `deck8_chance_enumeration` opt-in and off for training. Do not
extend this sample after seeing the result. Any redesigned panel/blending or
separate-budget treatment needs its own preregistered seed block; the current
mechanism has supplied no positive whole-game evidence at the 400-unit training
screen.

### Smoke gate

The two-paired-seed/two-throughput-game smoke passed. Each strength orientation
committed four panels, the all-on throughput cohort committed eight, and the
all-off cohort committed zero. Every cohort satisfied `bootstrap_rows = 70 ×
panel_count`; all-off and all-on each evaluated 32,882 rows. Two-slot cohort
times ranged from 18.6 to 22.0 seconds. These smoke outcomes are plumbing and
runtime checks only and use no sealed-run seeds.
