# Run7 prep report (2026-07-08)

Staying at 80x6. Run7 = run6's diversity package + three fixes addressing the
run6 diagnosis (promotion never passed → learner chased a frozen target and
drifted; ~21% of buffer examples were 400-sim HOF-seat labels blurring the
policy).

## Item 1 — Bank run6's best (revised protocol: 3-way round-robin)

Candidates: run6 iter_0020 / iter_0025 / iter_0040 (the three strongest
paired-gate signals in run6).

**Protocol revision (user decision, 2026-07-08):** the originally pre-registered
vs-incumbent gate (each candidate vs current_best, promote iff LCB > 0.51) was
dropped mid-session. Rationale: run6's own in-run gate checks already had
iter_0020 and iter_0025 above current_best (WR 0.52/0.52; they failed only the
over-strict 0.53 wall run7 removes), so the incumbent match is redundant.
Revised rule: round-robin the three candidates — 2500 paired seat-swapped games
@ sims=300 per pairing via `promotion.evaluate_checkpoint_match` — and promote
the highest aggregate win% **unconditionally**. Run on the run6 cloud box
(`run7_item1_rr_cloud.py`; ~28 min/pairing on the 5090).

Pairings (a-frame):

| pairing | games | a WR | W-L-D | a mean margin |
|---|---|---|---|---|
| iter_0020 vs iter_0025 | 2500 | 0.4990¹ | — | — |
| iter_0020 vs iter_0040 | 2500 | 0.5242 | 1264-1143-93 | +0.87 |
| iter_0025 vs iter_0040 | 2500 | 0.5292 | 1288-1142-70 | +1.29 |

¹ detail rows clipped in the log paste-back; points (1247.5/2500) reconstructed
exactly from the aggregate table.

Aggregate standings (5000 games each):

| candidate | win% | points |
|---|---|---|
| **iter_0025** | **0.5151** | 2575.5 |
| iter_0020 | 0.5116 | 2558.0 |
| iter_0040 | 0.4733 | 2366.5 |

**Outcome: run6 `iter_0025` promoted to `best_checkpoint/current_best.pt`**
(sha `df6c14cd…`; old best backed up as `previous_current_best_20260708T183552Z.pt`
and added to the run7 HOF pool as `pre_run7`). Applied by
`runs/kingdomino/run7_item1_bank.py`; audit trail in
`best_checkpoint/current_best.json` + `promotion_log.jsonl`. iter_0025 is
run7's warm start.

Caveat kept honest: the round-robin establishes iter_0025 as best-of-run6, not
as a confident gain over the run5 net (the 0.52 in-run signals were low-power).
The separation from iter_0040 is real (0.5292, LCB 0.5096); 0025 vs 0020 is a
coin flip (0.4990 in 0020's frame). Ranking by aggregate win% per the revised
rule.

## Item 2 — HOF asymmetric deep targets

run6's HOF games searched BOTH seats at `hof_sims=400`, so ~21% of buffer
examples carried shallow 400-sim policy labels. The HOF opponent's job (per
`hof.py`) is steering the learner into novel positions — not labelling them.

Engine change (`kingdomino_rust/src/lib.rs`): `BatchedMCTS` gained a per-seat
search override — constructor kwargs `hof_opponent_seat` / `hof_opponent_sims`
/ `hof_opponent_dirichlet_eps` / `hof_opponent_temp_moves`. The override seat
(the frozen HOF net) is pinned to a fixed shallow no-record profile at every
move-profile decision point; the other seat keeps the normal profile. Default
`hof_opponent_seat=-1` = off, so normal self-play, elo, and promotion paths are
untouched (regression tests pass).

`self_play.play_hof_games_batched` now builds the engine with the MAIN
self-play search config for the learner seat — sims=`--sims`,
`playout_cap_randomization` + `full_search_fraction` ON, fast moves at
`--fast_move_sims` never recorded — and pins the HOF seat (per orientation) to
`--hof_sims` / `--hof_temp_moves` / `--hof_dirichlet_epsilon`. Only learner
full-search (and exact-endgame) moves become training examples, exactly like
normal self-play.

Pruning bug fixed: the HOF prune call in the training loop passed
`total_visits=hof_sims`; recorded HOF moves are now learner searches at
`n_simulations`, so it prunes against that (`self_play.py` ~L3250).

Pool curation (near-peer preference, simple option): run7 pool
`runs/kingdomino/hof_run7/` drops run6's weakest opponent run1_iter66
(Elo ~1648 — blowout fodder for a full-strength learner). Final pool:
run3_iter80 (~1809) + run5_avg_0006_0090 (~run5 level) + pre_run7 (the
outgoing run5 current_best, banked by Item 1). Sampling stays uniform.
Seeded by `seed_hof_run7.py`.

### Verification (Item 5), tiny job `run7_verify_tiny` (2 iters, 8 games/iter, sims 600/fast 100/hof 100)

From `run7_verify_inspect.py` over the run's saved buffer + checkpoint:

- (a) **HOF learner moves at full sims**: all 22 recorded HOF MCTS moves have
  root visit sum exactly 600 (= `--sims`), not 100 (= `--hof_sims`).
- (b) **Only learner-seat moves recorded**: all 46 HOF examples have
  owner=current (21 `current_vs_hof` + 25 `hof_vs_current`; 24 are
  exact-endgame learner moves). Engine-level: the frozen seat never records.
- (c) **Prune vs frontier budget**: min policy-target mass among HOF MCTS
  moves is 2/600; no 1/600 entries survive (one-visit pruning at
  `total_visits=n_simulations`).
- (d) **Gate flags in effect**: checkpoint config reads back
  `promotion_games=2500, promotion_sims=300, promotion_min_win_rate=0.51,
  promotion_min_lcb=0.51, soft_gate mode, revert 0.48`.
- Search asymmetry cross-check (CPU mini-run, counting leaf rows per net):
  learner net evaluated 30,215 leaves vs HOF net 5,894 (5.1×; predicted ≈5.4×
  from (600×0.5+50×0.5)/60 at full_search_fraction 0.5).
- Recorded HOF examples/game dropped from ~25 (run6: every learner move) to
  ~11.5 (full-search fraction of learner moves + exact endgame) — the shallow
  labels are gone, not merely diluted.

## Item 3 — Promotion ratchet (config only)

`decide_promotion` already ratchets on strict LCB; run7 changes only the gate
power/thresholds in the launch flags:

| flag | run6 | run7 |
|---|---|---|
| `--promotion_min_win_rate` | 0.53 | 0.51 |
| `--promotion_min_lcb` | 0.50 | 0.51 |
| `--promotion_games` | 1032 | 2500 |
| `--promotion_sims` | 100 | 300 |

Fixed-suite guard and `--soft_gate_revert_win_rate 0.48` unchanged. **Honest
caveat**: this unlocks the ratchet so real micro-gains get banked — it does
not by itself create strength. If the learner never truly exceeds
current_best by >51%, run7 will (correctly) bank nothing.

## Item 4 — Launch

`runs/kingdomino/run7_launch.sh` (run6 base + deltas): warm start =
post-Item-1 `best_checkpoint/current_best.pt` copied to the run dir, **fresh
buffer** (no `--warm_buffer` — run6's buffer is exactly the 400-sim-HOF /
drifted data being removed), `--hof_dir runs/kingdomino/hof_run7`, ratchet
gate flags above, `--seed 7`. Everything else identical to run6 (sims 4800,
playout-cap 0.25/200, lr 1e-4, hof_fraction 0.2, soft_gate).

Launch (on the box, after syncing repo + `best_checkpoint/current_best.pt` +
`hof_run7/`):

```bash
cd /root/boardgame-ai && bash runs/kingdomino/run7_launch.sh
```

Note: `maturin develop --release` (or wheel build) must run on the box first —
run7 requires the new `kingdomino_rust` with the per-seat override.

## Status

- Item 1 done: `iter_0025` banked as current_best (round-robin winner).
- Item 2 done + verified: asymmetric HOF deep targets in `kingdomino_rust`
  (`hof_opponent_seat` override) + `play_hof_games_batched`; prune bug fixed;
  `hof_run7` pool curated (run3_iter80, run5_avg, pre_run7).
- Items 3+4 done: `run7_launch.sh` ready (ratchet gate 0.51/0.51/2500/300,
  fresh buffer, seed 7).
- Item 5 done: tiny run verified (a)-(d); see verification section.
- Remaining (manual, on the box): sync repo + `best_checkpoint/current_best.pt`
  + `runs/kingdomino/hof_run7/`, rebuild the engine
  (`cd games/kingdomino/kingdomino_rust && maturin develop --release` — run7
  REQUIRES the new per-seat override), then
  `bash runs/kingdomino/run7_launch.sh`.
