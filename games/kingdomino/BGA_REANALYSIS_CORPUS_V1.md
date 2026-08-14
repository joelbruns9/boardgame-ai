# Kingdomino BGA reanalysis corpus v1

## Outcome

The 36-game placement-audit corpus now has a searchable, reconstructable
position artifact suitable for qualifying deep-target reanalysis.

| Quantity | Value |
|---|---:|
| Games | 36 |
| Development positions | 940 |
| Confirmation positions | 460 |
| Total positions | 1,400 |
| Legally reconstructed human actions | 1,393 |
| Exact-candidate late roots | 311 |
| Viewer / opponent roots | 829 / 571 |

The source whole-game split is unchanged from
`placement_audit_corpus_v1.json`. No deep-search result was inspected while
building the corpus.

## Artifacts

- Builder: `games/kingdomino/bga_reanalysis_corpus.py`
- Corpus: `runs/kingdomino/placement_audit/bga_reanalysis_positions_v1.jsonl`
- Summary and hashes:
  `runs/kingdomino/placement_audit/bga_reanalysis_positions_summary_v1.json`
- Tests: `games/kingdomino/tests/test_bga_reanalysis_corpus.py`

Rebuild from the repository root with:

```powershell
.\.venv\Scripts\python.exe -m games.kingdomino.bga_reanalysis_corpus build --split all
```

## Information contract

Every row contains a compact public `GameState`, legal action indices, source
provenance, reconstruction status, and the human action when it was uniquely
recoverable. The hidden domino identities are stored as a sorted **unordered
bag** solely for deterministic serialization and hashing. Their serialized order
is not the future reveal order.

Any search consumer must redeterminize from that bag. It must not use
`diagnostics_only`, source final scores, the human action, or any future logged
state as search input. Those fields are offline evaluation metadata only.

The seven roots without a unique human action remain useful as search states,
but they cannot support human-action comparisons. All 119 dropped reconstruction
rows remain excluded.

## What this does and does not establish

This corpus solved the practical prerequisite that blocked replay reanalysis:
it supplies reconstructable states without modifying the compact AlphaZero
buffer. It did **not** establish that deeper labels were better, and it is not a
training dataset by default.

The downstream development/confirmation deep-target audit used it to:

1. Compare ordinary 4,800-sim search with repeated 30,000-sim searches under
   common random numbers.
2. Probe every pick group fairly and record forced-probe evidence separately
   from ordinary MCTS visits.
3. Use official-outcome exact search where eligible.
4. Measure the deep teacher's value loss after forcing the 4,800-sim action;
   action disagreement alone is not sufficient.
5. Cluster uncertainty by source game, freeze the method on development games,
   and inspect confirmation once.

The corrected development audit found independently positive cross-seed uplift
from the 30,000-simulation teacher on a concentrated 24-root cohort. Its
game-clustered 95% lower bound was positive, so the development gate passed.
The frozen confirmation split is still untouched and must pass before selective
deep relabeling earns a training pilot. See `DEEP_TARGET_STAGE3_FINDINGS.md`.
The corpus also remains useful as a source of verified public roots for a
separately specified BGA-seeded self-play or production-search distillation
experiment.
