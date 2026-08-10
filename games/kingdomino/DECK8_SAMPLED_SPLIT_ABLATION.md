# Deck=8 sampled explicit-split ablation

## Purpose

The completed causal-leverage experiment showed that production-like chance
search changes the root policy substantially, but it did not isolate the source
of that change.  Production treatment has two coupled components:

1. reveal outcomes receive separate public-observation subtrees instead of
   being aliased in one open-loop node;
2. the first admitted reveal action receives a complete 70-row network
   bootstrap and switches to probability-mean backup.

This ablation adds the missing middle arm to distinguish those components.

## Preregistered arm

`sampled_split` uses:

- the same exhaustive `C(8,4) = 70` fixed support as the production arm;
- the same split point, after PUCT selects the reveal-triggering action;
- the same explicit public-observation subtree routing;
- sampled backup and balanced row traversal;
- **zero** bootstrap rows and zero initialization NN work;
- B ordinary simulation paths at B = 800, 1,600, 3,200, and 6,400.

It is paired to the completed artifact by using the exact checkpoint, 64 frozen
positions, budgets, search configuration, and per-cell seeds recorded in that
artifact.  Only the 256 new sampled-split cells are run; control and panel cells
are not repeated.

## Comparisons

- `control` → `sampled_split` measures the effect of explicit chance topology,
  sampled backup, and balanced traversal relative to aliased open loop.
- `sampled_split` → `panel_extra` measures the incremental effect of the full
  70-row bootstrap and probability-mean backup without compute displacement.
- `sampled_split` → `panel_charged` shows that same incremental treatment under
  the production work budget.

The primary descriptive metrics are paired root-visit total variation, top
joint-action change rate, top-pick change rate, and the fraction of positions
with visit TV at least 0.10.  Wilson 95% intervals are used for top-action
changes and deterministic position-bootstrap intervals for mean TV.

## Interpretation

- If control→sampled is large and sampled→panel is small, most of the original
  mechanism effect came from correcting open-loop aliasing/strategy fusion;
  enumeration added little with the current network.
- If control→sampled is small and sampled→panel is large, the 70-row network
  expectation was the main lever.
- If both are large, topology and enumeration both matter.
- If sampled and panel move in different, unstable directions, the current
  network's counterfactual value surface or panel backup is a leading concern.

This remains a search-behavior diagnosis.  It does not establish playing
strength and does not enable the feature in self-play.

## Artifacts and command

Base artifact:
`runs/kingdomino/chance_correct_a1/deck8_causal_leverage_v1.json`

Output:
`runs/kingdomino/chance_correct_a1/deck8_sampled_split_ablation_v1.json`

```powershell
.\.venv\Scripts\python.exe -m games.kingdomino.chance_split_ablation
```

The output is atomically rewritten after every completed position/budget cell
and refuses to mix a changed base artifact, checkpoint, position set, or search
configuration.
