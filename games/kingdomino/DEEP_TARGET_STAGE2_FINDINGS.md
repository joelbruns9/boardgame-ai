# Kingdomino deep-target audit: Stage 2 findings

## Outcome

The revised filter worked: additional search changed tile picks in the
deliberately contested cohort but in none of the easy or starvation controls.
This is evidence that 800 simulations are insufficient for screening close BGA
roots. It is **not yet evidence that 4,800-sim training labels have material
regret**; Stage 2 measures action stability, not the value loss from forcing the
4,800 action.

The frozen 30,000-sim qualification pass therefore remained selective. Stage 3
escalated 32 of the 940 development roots (3.4%):

- 11 roots where both 800 searches agreed, both 4,800 searches agreed, and the
  agreed tile changed;
- 13 roots whose two 4,800 searches still disagree on the tile; and
- 8 additional roots where a formerly unvisited tile was within 0.03 actor Q
  of the best tile in a matched forced-pick probe.

The three sets do not overlap. The other 908 development roots were not
escalated, and the 460 confirmation roots remained frozen.

## Frozen Stage-2 cohort

Selection was frozen before any 4,800 result was examined. It used the two
saved 800-sim repeats:

| Stratum | Rule | Roots |
|---|---|---:|
| Live and close | `abs(mean root Q) <= 0.4` and top-two pick Q gap `<= 0.03` in both repeats | 122 |
| Pick unstable | Two 800 repeats selected different tiles | 9 |
| Easy control | Preselected Stage-1 random easy control | 33 |
| Starvation control | One live-band zero-visit root per deck-count bucket, outside the core union | 11 |
| Union | Overlapping flags counted once | 172 |

The root-Q band avoids treating saturated wins and losses as informative merely
because several moves have similar values. The Q values are the actor-framed
composite search utility, not literal win probabilities.

Every selected root received two 4,800-sim searches using the exact two Stage-1
seeds. No confirmation root was searched. Runtime was 26.7 minutes on the RTX
3070 Laptop GPU.

## Results

| Result | All 172 | Live/close 122 | Easy controls 33 | Starvation controls 11 |
|---|---:|---:|---:|---:|
| Tile changed in at least one paired 800->4,800 comparison | 31 | 25 | 0 | 0 |
| Stable consensus tile changed | 11 | 11 | 0 | 0 |
| Two 4,800 searches disagree on tile | 13 | 13 | 0 | 0 |
| Q gap `<= 0.03` in both 4,800 searches | 68 | 64 | 1 | 1 |

The nine Stage-1 tile-unstable roots all changed in at least one paired
comparison. Seven resolved to one tile at 4,800; two remained unstable. These
flags overlap the live/close stratum, so column totals should not be added.

Across every paired repeat, tile choice changed 44/344 times (12.8%), while the
exact joint placement/pick action changed 118/344 times (34.3%). The larger
joint-action rate again shows that placement identity is more sensitive than
tile identity. The selected cohort's two-seed tile agreement was 94.8% at 800
and 92.4% at 4,800. This decrease is not a population-level regression: the
cohort was selected for close alternatives, and deeper search exposed more of
those alternatives without fully converging.

## Matched zero-visit check

Eighteen Stage-2 roots still had at least one tile group with zero ordinary
4,800 visits. For those roots, every available tile—not only the unvisited
one—received two restricted 800-sim searches with common seeds.

| Matched forced-pick result | Count |
|---|---:|
| Affected roots | 18 |
| All tile groups searched | 72 |
| Formerly zero-visit groups | 20 |
| Formerly zero-visit group ranked best | 0 |
| Formerly zero-visit group within 0.03 Q of best | 8 |
| Formerly zero-visit group within 0.05 Q of best | 15 |

Therefore, zero ordinary visits did not hide the preferred tile in this sample,
but it also did not prove that the tile was bad. Eight near alternatives remain
worthy of the selective deep pass.

## What Stage 3 answered

Stage 3 ran the frozen 32 roots with paired ordinary 30,000-sim searches and
matched 10,000-sim searches for every tile. Cross-seed teacher uplift was
effectively zero and its game-clustered interval included zero. Extra search was
predominantly tie-breaking; replay reanalysis did not earn an engineering or
training run. See `DEEP_TARGET_STAGE3_FINDINGS.md`.

## Artifacts

- Stage-2 runner: `games/kingdomino/deep_target_stage2.py`
- Frozen cohort: `runs/kingdomino/placement_audit/deep_target_stage2_cohort_v1.json`
- Stage-2 rows: `runs/kingdomino/placement_audit/deep_target_stage2_development_s4800_r2.jsonl`
- Stage-2 summary: `runs/kingdomino/placement_audit/deep_target_stage2_summary_development_s4800_r2.json`
- Matched forced-pick runner: `games/kingdomino/deep_target_forced_pick_probe.py`
- Matched rows: `runs/kingdomino/placement_audit/deep_target_stage2_matched_forced_pick_v1.jsonl`
- Matched summary: `runs/kingdomino/placement_audit/deep_target_stage2_matched_forced_pick_summary_v1.json`
- Frozen Stage-3 cohort: `runs/kingdomino/placement_audit/deep_target_stage3_cohort_v1.json`

The Stage-2 result SHA-256 is
`06443e5020c64b1dc1d52124cc5d20b1861234cf97a5b87b509b190bfe7f9cfc`.
The matched-probe result SHA-256 is
`7623c734eccd9def7b6e0f3cee205cb8ff80f7fe311569685bbe3bbcfbfc206c`.
