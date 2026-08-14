# Kingdomino deep-target audit: Stage 2 findings

## Outcome

The ordinary Stage-2 screen remains valid. Two 4,800-simulation searches on a
frozen 172-root development cohort changed the selected tile in at least one
paired 800-to-4,800 comparison on 31 roots. Eleven roots changed a stable
two-seed consensus, and 13 roots still disagreed by seed at 4,800 simulations.
None of the easy or starvation controls changed tile.

The first matched forced-pick probe was invalid: a root-action mask in the Rust
open-loop search was bypassed by missing-child recovery, so excluded actions
were silently reintroduced. The output looked plausible, but no requested tile
group received its full nominal 800 simulations. That probe and the 32-root
Stage-3 cohort derived from it are superseded.

After fixing the mask, all 144 restricted searches received exactly 800 visits
inside the requested pick group. The corrected probe reduced the frozen
Stage-3 cohort from 32 to 24 roots:

- 11 roots where the stable 800-simulation consensus changed at 4,800; and
- 13 roots whose two 4,800-simulation searches still disagreed on the tile.

The 460 confirmation roots remain untouched.

## Frozen Stage-2 cohort

Selection was frozen before any 4,800 result was examined. It used two saved
800-simulation repeats:

| Stratum | Rule | Roots |
|---|---|---:|
| Live and close | `abs(mean root Q) <= 0.4` and top-two pick Q gap `<= 0.03` in both repeats | 122 |
| Pick unstable | Two 800 repeats selected different tiles | 9 |
| Easy control | Preselected Stage-1 random easy control | 33 |
| Starvation control | One live-band zero-visit root per deck-count bucket, outside the core union | 11 |
| Union | Overlapping flags counted once | 172 |

Every selected root received two 4,800-simulation searches using the exact two
Stage-1 seeds. Runtime was 26.7 minutes on the RTX 3070 Laptop GPU.

| Result | All 172 | Live/close 122 | Easy controls 33 | Starvation controls 11 |
|---|---:|---:|---:|---:|
| Tile changed in at least one paired comparison | 31 | 25 | 0 | 0 |
| Stable consensus tile changed | 11 | 11 | 0 | 0 |
| Two 4,800 searches disagree on tile | 13 | 13 | 0 | 0 |
| Q gap `<= 0.03` in both 4,800 searches | 68 | 64 | 1 | 1 |

Across all paired repeats, tile choice changed 44/344 times (12.8%), while the
exact joint placement/pick action changed 118/344 times (34.3%). Stage 2
measures action stability, not the value loss from forcing the 4,800 action.

## Corrected matched zero-visit check

Eighteen Stage-2 roots had at least one tile group with zero ordinary 4,800
visits. Every available tile at those roots received two properly restricted
800-simulation searches with common seeds.

| Corrected matched result | Count |
|---|---:|
| Affected roots | 18 |
| Tile groups searched | 72 |
| Restricted searches | 144 |
| Searches with exactly 800 group visits | 144 |
| Formerly zero-visit groups | 20 |
| Formerly zero-visit group ranked best | 0 |
| Within 0.03 Q of best | 0 |
| Within 0.05 Q of best | 3 |

The eight starvation-derived roots previously escalated at the 0.03 threshold
were therefore removed. No new roots were added.

## Stage-3 implication

The corrected 24-root development qualification produced a positive
game-clustered lower confidence bound. That passes the development gate and
requires the frozen confirmation split before a training decision. See
`DEEP_TARGET_STAGE3_FINDINGS.md`.

## Artifacts

- Stage-2 rows: `runs/kingdomino/placement_audit/deep_target_stage2_development_s4800_r2.jsonl`
- Corrected matched rows: `runs/kingdomino/placement_audit/deep_target_stage2_matched_forced_pick_v2.jsonl`
- Corrected matched summary: `runs/kingdomino/placement_audit/deep_target_stage2_matched_forced_pick_summary_v2.json`
- Corrected Stage-3 cohort: `runs/kingdomino/placement_audit/deep_target_stage3_cohort_v2.json`
- Tests: `games/kingdomino/tests/test_deep_target_stage2.py`

SHA-256:

- ordinary Stage-2 rows: `06443e5020c64b1dc1d52124cc5d20b1861234cf97a5b87b509b190bfe7f9cfc`
- corrected matched rows: `7fc819e95913e0fe1182b90cc10eea99a889c244e1775fe16be52b57e2b8e492`
