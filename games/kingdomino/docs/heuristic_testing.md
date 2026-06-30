# Kingdomino Endgame Move-Ordering Heuristic Testing

## Summary

This note records the alpha-beta move-ordering experiments for contested
Kingdomino deck-4 endgames. The goal was to improve exact endgame solve
throughput for self-play, especially on network-generated positions where close
margins make alpha-beta cutoffs weaker.

The primary heuristic going forward is:

`lookahead2_clustered`

Behavior:

- Root node: always order by exact 1-ply post-move raw margin.
- Depth 1 and 2: use recursive 1-ply ordering only when the cheap heuristic is
  clustered:
  - `legal.len() >= 8`
  - at least 4 moves are within 4 cheap-heuristic points of the best cheap score
- Deeper nodes: use the cheap heuristic.

This was selected because it had the best total time, p50, p75, and p90 among
tested variants on the shared 50-position network benchmark set. It does trade
off some p95 stability versus root-only and depth-1-clustered variants, but the
training throughput target was p90/average performance.

## Code Locations

- Rust solver and heuristic variants:
  - `games/kingdomino/kingdomino_rust/src/lib.rs`
- Benchmark script:
  - `games/kingdomino/bench_endgame_tail.py`
- Correctness tests:
  - `games/kingdomino/test_endgame_exact.py`
- Saved benchmark positions:
  - `runs/kingdomino/benchmarks/network_positions_50.pkl`

## Important Environment Note

In Codex, project Python/Torch/CUDA commands must be run with escalated
permissions. Use the project venv explicitly:

```powershell
C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe ...
```

Examples:

```powershell
C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m maturin develop --release
C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m pytest games\kingdomino\test_endgame_exact.py -v
```

## Benchmark Setup

All timing comparisons below used the same 50 network-generated deck-4
positions:

```powershell
C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m games.kingdomino.bench_endgame_tail `
  --from-network runs\kingdomino\local_48x6_run8\iter_0042.pt `
  --channels 48 --blocks 6 --bilinear-dim 64 `
  --sims 200 --n 50 --no-time-limit `
  --save-positions runs\kingdomino\benchmarks\network_positions_50.pkl `
  --alpha 0.8 --label baseline --ordering baseline
```

Subsequent variants loaded the saved positions:

```powershell
C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m games.kingdomino.bench_endgame_tail `
  --load-positions runs\kingdomino\benchmarks\network_positions_50.pkl `
  --no-time-limit --alpha 0.8 `
  --label <label> --ordering <ordering>
```

## Results

All rows are parallel YBW solver timings on the same 50 saved network positions.

| Variant | Ordering | CSV | Total | Avg | p50 | p75 | p90 | p95 | p99 | Max |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | `baseline` | `endgame_tail_20260628_204842_baseline.csv` | 83.1s | 1663ms | 1049ms | 2170ms | 4519ms | 5350ms | 7866ms | 7866ms |
| Opponent denial | `denial` | `endgame_tail_20260628_205011_option_a_denial.csv` | n/a | n/a | 1016ms | 1913ms | 4218ms | 5346ms | 7594ms | 7594ms |
| Root 1-ply | `lookahead` | `endgame_tail_20260628_205134_option_b_lookahead.csv` | 73.2s | 1463ms | 724ms | 1721ms | 3822ms | 5079ms | 6761ms | 6761ms |
| Combined denial + root 1-ply | `combined` | `endgame_tail_20260628_205305_option_c_combined.csv` | n/a | n/a | 1024ms | 2138ms | 4473ms | 5411ms | 8145ms | 8145ms |
| Recursive 1-ply depth <= 2 | `lookahead2` | `endgame_tail_20260628_212555_option_b_lookahead2.csv` | n/a | n/a | 718ms | 1419ms | 3496ms | 5565ms | 7419ms | 7419ms |
| Adaptive depth <= 2, legal >= 12 | `lookahead2_adaptive` | `endgame_tail_20260628_213824_option_b_lookahead2_adaptive.csv` | 68.4s | 1368ms | 668ms | 1651ms | 3398ms | 5510ms | 6406ms | 6406ms |
| Adaptive depth <= 2, legal >= 16 | `lookahead2_adaptive16` | `endgame_tail_20260628_215355_option_b_lookahead2_adaptive16.csv` | 69.9s | 1398ms | 666ms | 1804ms | 3574ms | 5494ms | 6885ms | 6885ms |
| Adaptive depth <= 2, legal >= 8 | `lookahead2_adaptive8` | `endgame_tail_20260628_220323_option_b_lookahead2_adaptive8.csv` | 69.7s | 1394ms | 726ms | 1347ms | 3535ms | 5787ms | 6855ms | 6855ms |
| Clustered depth 1-2 | `lookahead2_clustered` | `endgame_tail_20260628_221923_option_b_lookahead2_clustered.csv` | 67.5s | 1350ms | 635ms | 1309ms | 3351ms | 5746ms | 6762ms | 6762ms |
| Clustered depth 1 only | `lookahead1_clustered` | `endgame_tail_20260628_223021_option_b_lookahead1_clustered.csv` | 67.8s | 1356ms | 751ms | 1713ms | 3566ms | 5169ms | 6557ms | 6557ms |

Notes:

- `lookahead2_clustered` is the best tested option for total time and p90.
- `lookahead1_clustered` is a strong tail-protection alternative, with much
  better p95 than clustered depth 1-2 and nearly the same total time.
- `lookahead` root-only still has the best p95 among the faster variants, but
  loses clear average/p90 throughput.

## Correctness Tests

Run:

```powershell
C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m pytest games\kingdomino\test_endgame_exact.py -v
```

Latest result after adding clustered depth-1:

- 38 passed

The relevant invariance tests assert that move ordering changes speed only, not
the solved minimax value:

- `test_option_a_does_not_change_optimal_value`
- `test_option_b_does_not_change_optimal_value`
- `test_recursive_lookahead2_does_not_change_optimal_value`
- `test_adaptive_lookahead2_does_not_change_optimal_value`
- `test_clustered_lookahead2_does_not_change_optimal_value`
- `test_clustered_lookahead1_does_not_change_optimal_value`

## How To Test More Variants

1. Add a new `SolverOrderMode` in
   `games/kingdomino/kingdomino_rust/src/lib.rs`.
2. Wire the string name in `SolverOrderMode::from_str`.
3. Add it to `--ordering` choices in
   `games/kingdomino/bench_endgame_tail.py`.
4. Add a value-invariance test in
   `games/kingdomino/test_endgame_exact.py`.
5. Rebuild:

```powershell
cd games\kingdomino\kingdomino_rust
C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m maturin develop --release
cd ..\..\..
```

6. Run tests:

```powershell
C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m pytest games\kingdomino\test_endgame_exact.py -v
```

7. Run benchmark on the saved positions:

```powershell
C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m games.kingdomino.bench_endgame_tail `
  --load-positions runs\kingdomino\benchmarks\network_positions_50.pkl `
  --no-time-limit --alpha 0.8 `
  --label <label> --ordering <ordering>
```

## Remaining Heuristics Worth Testing

1. Clustered gate tuning:
   - `legal.len()` threshold: 6, 10, 12
   - top-band size: 3, 5, 6
   - delta: 2, 3, 6
2. Clustered with legal-count upper cap:
   - Example: fire only when `8 <= legal.len() <= 32`
   - Goal: protect the p95/p99 tail from very high-branching nodes.
3. Root-only 2-ply shallow minimax:
   - At root, order each child by opponent's best immediate reply.
   - More expensive, but root-local and may avoid 1-ply traps.
4. Mobility-aware tie-breaker:
   - For near-ties in margin/cheap score, prefer moves that leave the opponent
     fewer legal actions.
5. End-state bonus-aware placement ordering:
   - Add harmony and middle-kingdom completion/preservation estimates near final
     placement.
6. Killer/history heuristic:
   - Track action features that cause cutoffs and try similar actions earlier at
     sibling nodes.
7. Best-child cache ordering:
   - Store best child index with exact result cache entries and try it first on
     repeated public states.

