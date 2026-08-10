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

## Results

The run completed all 256 preregistered cells: 64 frozen positions at each of
the four budgets. Every sampled-split cell satisfied the implementation
invariants: the search reached explicit chance nodes, performed exactly B
ordinary simulation paths, and reported zero bootstrap and initialization
network rows.

| Sims | Control to sampled mean TV | Top action changes | Top pick changes | Sampled to panel-extra mean TV | Panel-extra top action changes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 800 | 0.166 | 15/64 (23.4%) | 7/64 (10.9%) | 0.015 | 3/64 (4.7%) |
| 1,600 | 0.235 | 21/64 (32.8%) | 9/64 (14.1%) | 0.014 | 1/64 (1.6%) |
| 3,200 | 0.313 | 32/64 (50.0%) | 14/64 (21.9%) | 0.016 | 1/64 (1.6%) |
| 6,400 | 0.383 | 33/64 (51.6%) | 17/64 (26.6%) | 0.015 | 1/64 (1.6%) |

The paired bootstrap 95% intervals for control-to-sampled mean TV were
[0.133, 0.202], [0.187, 0.284], [0.255, 0.371], and [0.319, 0.448] in ascending
budget order. The corresponding sampled-to-panel-extra intervals were only
[0.011, 0.019], [0.010, 0.020], [0.010, 0.024], and [0.009, 0.023].

The production-budget comparison led to the same conclusion. Mean TV between
sampled split and `panel_charged` was 0.024, 0.018, 0.017, and 0.016; at 6,400
sims it changed the top action and top tile pick in only one of 64 positions.

### Conclusion

The result matches the first preregistered interpretation: almost all of the
previous panel effect is reproduced without evaluating all 70 reveal rows.
The operative change is therefore the explicit public-observation split (and
the removal of open-loop aliasing/strategy fusion), not the exhaustive
network-value bootstrap. Increasing the simulation budget makes the explicit
split diverge further from control, while its policy remains very close to the
fully enumerated panel.

This does not prove that the network is unbiased. It shows that enumerating 70
counterfactual network evaluations does not overcome any such bias or add a
material root-policy effect in this setup. A sampled explicit split is the more
efficient search primitive to carry forward. Because its root decisions nearly
duplicate the panel whose gameplay A/B was weakly negative/null, another
current-checkpoint gameplay A/B is unlikely to answer the remaining question.
The higher-value next experiment is chance-aware training or fine-tuning with
sampled split, followed by a crossed evaluation of old versus new network and
open-loop versus sampled-split search. That directly tests whether the model
and search need to be trained together.

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

Completed artifact facts:

- 256/256 unique sampled-split cells; no errors;
- 64 positions at each of 800, 1,600, 3,200, and 6,400 simulations;
- frozen-position SHA-256:
  `dad379d2a350cf74a73b2f804269cba2199ee46c8a5526cf0fa5c5a35f6276bd`;
- checkpoint SHA-256:
  `4bf07b0ca14e5452e6533a9232967e89bb0ab0df88c99e9928a65f402b1f04b3`;
- paired base-artifact SHA-256:
  `c4b77147a2b09e65cd53ee6d6f067808535f63f2a9e2cfbf3b2e03fb287c8d60`.
