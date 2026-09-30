# Full-game audit pool

`PoolRolloutBackend` runs baseline full-game continuation through the existing
Rust game pool. It is useful for decision audits, NN action-ranking diagnostics,
future rollout labels or reanalysis, and baseline-continuation search experiments.
The existing NN training/self-play path already uses this pool; this adapter does
not claim a speedup for that training loop.

The forced root turn stays in Python against one shared immutable turn table.
Only nonterminal turn-start boards go to Rust. Each carries the completed root
turn count, root dice-roll count, root correction, and original shared-dice seed.
Rust implements the same v1 SHA-256 dice stream, rejection sampling, and
turn/roll coordinates as the serial backend. It accumulates each within-turn
dice-luck correction from the same table before and after the roll, including
the original bust leaf. Future opening rolls remain uncorrected. Audit games
retain terminal boards, counters and corrections without training histories.

All outcomes go through the existing `RolloutBackend` aggregation. Samples are
not clipped, renormalized, or excluded. A failure in any game fails the batch;
the audit's transactional checkpoints retain only previously completed batches.
Turn and roll limits include the root turn. Independent sample offsets are
preserved across chunking, thread counts and queue sizes. Pool resource limits
bound games in flight and inference rows; one oversized turn table may exceed
the row limit, as in the existing pool. Baseline continuation, a full-game
horizon, and shared dice are required. Selective continuation is not supported.

The decision-audit CLI keeps `serial` as its default. Add `--backend pool` to
use the adapter; optional settings are `--threads 0 --in-flight 64
--max-rows 1000000`. The report includes source and native-extension hashes.
Changing backend, resource settings, checkpoint, code or binary requires a new
output; previous completed reports remain valid records of their own runs.

When auditing a pilot produced with the previous native binary, also supply
`--allow-native-rebuild`. This explicit option permits a native-only hash change;
all Python source hashes must still match. Before any audit samples are drawn,
every saved pilot baseline is recomputed. Legal actions, choices, and values
must match (value tolerance 2e-6). Both binary hashes and the recheck are recorded.
This does not permit resuming an existing audit under a different binary.

## Measured 2026-09-30 result

Checkpoint `p4_pilot/iter_0080.pt`, three actual audit positions, 128 samples per
action, two independent repeats with alternating execution order. Each backend
completed **1,536 full-game rollouts**. Times include root preparation, native
continuation and common estimator aggregation; measured throughput varies by
position and laptop load.

| Position | Serial total | Pool total | Speedup |
| --- | ---: | ---: | ---: |
| Early 2p blocking | 105.91 s | 31.34 s | 3.38x |
| Middle 4p nonblocking | 40.35 s | 11.79 s | 3.42x |
| Late 4p blocking | 43.08 s | 12.80 s | 3.37x |
| Total | **189.35 s** | **55.94 s** | **3.39x** |

There were **zero raw winner disagreements**. All paired numerical diagnostic
intervals were inside the declared margin; the largest absolute difference in
an action's mean adjusted payoff was 0.0000011 percentage points. The pool may
still take different intermediate paths because of NN batch rounding. This
benchmark does not establish playing strength or a universal speedup.

The proposed 100-position headroom experiment's prior 1.7–2.6 hour estimate
becomes approximately **30–50 minutes** at this throughput. Different position
mixes, three-action prevalence and GPU load can change that forecast.

Validation: **212 distinct tests passed**, including the existing training,
self-play, native-pool and rollout regressions plus the new adapter checks.
The actual GPU audit CLI also completed a 36-rollout smoke run with atomic
checkpoints; all **60 original baseline choices and values matched exactly**
after the native rebuild. The small smoke run is plumbing validation only.

Artifacts: `runs/audit_pool_benchmark_20260930.json` and
`runs/audit_pool_cli_smoke_20260930.json`. The adapter and release extension are
installed locally; the headroom experiment has not been launched.

## Matched benchmark

```powershell
Set-Location C:\Users\joeld\projects\boardgame-ai-cantstop
& C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m games.cantstop.benchmark_audit_pool --checkpoint runs/p4_pilot/iter_0080.pt --suite runs/decision_audit_20260930.positions.json --samples 128 --repeats 2 --positions 3 --out runs/audit_pool_benchmark.json
```

This alternates execution order on the same positions and dice seeds. It records
wall time, NN rows/calls, Rust/inference costs, raw outcome disagreements, and
paired differences between the two backends' adjusted samples. A declared
0.5-percentage-point margin is checked using per-action approximate 95%
diagnostic intervals; these are numerical diagnostics, not arena evidence.
Real NN batch composition can change floating-point predictions and near-tied
choices, so bit identity is required on deterministic mocks instead. The mock
gate covers board-hashed, feature-hashed and mover-wins evaluators, all three
phases, 2/3/4 players, corrections on/off, sample offsets, thread/refill/row
limits, and failure handling. Existing training/self-play regressions are also
required after changes to the shared pool.

## Rebuild the extension

The native wheel was rebuilt and installed in the project's shared venv. For a
later checkout/build, use the same interpreter and install the resulting wheel:

```powershell
Set-Location C:\Users\joeld\projects\boardgame-ai-cantstop\games\cantstop\cantstop_rust
& C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m maturin build --release --interpreter C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe --out ..\..\..\runs\audit_pool_wheels
& C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps ..\..\..\runs\audit_pool_wheels\cantstop_rust-0.1.0-cp312-cp312-win_amd64.whl
```

In Codex, project Python commands require escalated permissions per `AGENTS.md`.
