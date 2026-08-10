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

Pending the sealed run.

### Smoke gate

The two-paired-seed/two-throughput-game smoke passed. Each strength orientation
committed four panels, the all-on throughput cohort committed eight, and the
all-off cohort committed zero. Every cohort satisfied `bootstrap_rows = 70 ×
panel_count`; all-off and all-on each evaluated 32,882 rows. Two-slot cohort
times ranged from 18.6 to 22.0 seconds. These smoke outcomes are plumbing and
runtime checks only and use no sealed-run seeds.
