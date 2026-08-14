# Kingdomino placement-audit feasibility findings

- **Date:** 2026-08-13
- **Corpus:** `kingdomino-bga-clean-v1`
- **Scope:** frozen five-game development probe, opponent/player 1 only

## Reconstruction gate

- All 36 frozen clean games have complete ordered 24-domino sequences for both players.
- 31 games have a unique score-rule interpretation and valid final scores, making them eligible for whole-game score analysis.
- The reconstruction audit keeps 1,400 of 1,519 captured placement decisions. It verifies 1,215 decisions in whole-game-eligible games, including 495 opponent decisions.
- Terminal decisions without a later snapshot are retained only when exhaustive legal suffix enumeration reproduces the authoritative BGA final score. No final-placement optimality assumption is used.

## Solver feasibility probe

The fixed-sequence solver preserves tile order, uses the production placement legality and scoring rules, handles forced discards, and deduplicates identical behavioral board states between layers. Beam expansion uses the Rust `RustBoard` implementation; the capped exact-DP check is independent Python code.

Exact dynamic programming exceeded 50,000 unique states by placement layer 4 or 5 in every probe game. Exact whole-game dynamic programming is therefore not practical with the current state representation.

| Table | BGA score | Beam 256 | Beam 1,024 | Beam 4,096 | Beam 16,384 | Beam 65,536 |
|---|---:|---:|---:|---:|---:|---:|
| 881199380 | 113 | 117 | 120 | 129 | 129 | 141 |
| 881159658 | 133 | 115 | 117 | 110 | 123 | 130 |
| 881651578 | 119 | 126 | 123 | 139 | 142 | 143 |
| 881170142 | 136 | 111 | 126 | 116 | 134 | 140 |
| 883077657 | 144 | 128 | 128 | 138 | 153 | 153 |

Only one of five games is score-stable from width 16,384 to 65,536. One width-65,536 result remains three points below the human's recorded score. Narrower beam scores are also non-monotonic because a larger beam can retain different high-ranked prefixes and later lose a trajectory kept by a narrower beam.

## Gate decision

The current beam heuristic has **not converged** and is not yet a valid hindsight reference. Scores below the observed human outcome are solver misses, not negative placement headroom. They must not enter the human-versus-model regret analysis.

Do not advance to confirmation games or interpret placement headroom from these whole-game beam scores. The next solver iteration should improve prefix ranking or use an adaptive/hybrid suffix method, then repeat this same frozen development probe. A usable reference must at minimum:

1. match or exceed the observed human final score in every feasibility case;
2. be stable across the two widest practical settings, or carry an explicit unresolved approximation interval; and
3. continue to label beam results as approximate rather than exact.

## Exact late-game suffix boundary

An additional probe froze each player's actual reconstructed board prefix and
exhaustively enumerated only the remaining claimed-tile suffix. With a cap of
one million distinct boards per layer:

- 12 or 10 tiles remaining was not reliably feasible;
- 9 tiles remaining completed in three observed cases, exceeded the cap in one,
  and lacked a reliable snapshot in one;
- 8 tiles remaining completed in every observed case, peaking between 22,523
  and 94,955 distinct states and taking 0.08–0.35 seconds per suffix;
- 6 tiles remaining peaked at 7,113 states or fewer.

The robust exact cutoff in this probe is therefore after 16 of the player's 24
placement opportunities: start with placement 17, at the beginning of the last
four two-domino rounds. This can support an exact late-placement audit, but it
conditions on the actual first 16 placements and therefore cannot measure
whether early placements created or destroyed later flexibility.

Machine-readable reports:

- `runs/kingdomino/placement_audit/solver_feasibility_probe_v1.json`
- `runs/kingdomino/placement_audit/solver_wide_probe_65536_v1.json`
- `runs/kingdomino/placement_audit/exact_suffix_feasibility_v1.json`
- `runs/kingdomino/placement_audit/exact_suffix_boundary_v1.json`

## Development late-placement result

Using the exact placement-17+ reference on 112 reconstructable opponent
decisions from 21 eligible development games:

| Actor | Game-weighted mean regret | Zero-regret decisions |
|---|---:|---:|
| Strong-human opponent | 0.581 | 80.4% |
| Current-best raw policy, logged pick fixed | 0.817 | 78.6% |
| Current-best 4,800-sim search, logged pick fixed at root | 0.662 | 80.4% |

The paired raw-policy-minus-human difference was +0.236 points per decision,
with a game-clustered 95% bootstrap interval of [-0.182, +0.725]. The searched
difference was +0.082, interval [-0.181, +0.371]. Development therefore does
not establish a positive model-minus-human placement gap; search repairs most
of the raw point estimate.

Settings were frozen in `placement_late_audit_protocol_v1.json` before opening
the confirmation split.
