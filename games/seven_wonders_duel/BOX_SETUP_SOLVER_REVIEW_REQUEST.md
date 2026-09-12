# Review request: box setup — sweep fidelity, solver caps, solver sizing

**What this is.** Eight commits (`192aee7`..`cc2a5ab`) on `sevenwd-w9-prototype`
that change how `setup_cloud_7wd.sh` measures a rented box and what it launches
on. Three threads, which became entangled and are easier to review together than
apart:

1. the throughput sweep now measures the RUN rather than `PhaseDConfig`'s
   defaults, and had to be made affordable afterwards;
2. the endgame solver's one node cap became two, and every attempted solve now
   records what the cost model predicted;
3. the solver's caps are now SIZED from the box rather than pinned, using a
   corpus priced once off-box.

**Reviewed 2026-09-11: seven findings, all reproduced, all fixed.** Five were
defects that would have produced wrong numbers on a rented box, including one
that made the sweep's entire solver axis measure ZERO solves. Two claims in this
document were disproved and are corrected in place (§6, §7). See §11 for the
audit trail. Sections 1-10 describe the code as it now stands.

**Status: built, tested, and all of it has now RUN at least once** — the laptop
soak (20/20 iterations, clean) exercised the mechanisms end to end, and
`rehearse_sweep_laptop.sh` runs the box's stage 6b/8b/8c pipeline green. What has
NOT happened is a rented box. Every absolute number here belongs to a laptop or
to cloud2's archived run.

**Default behaviour is NOT unchanged**, and that is the thing to scrutinise
hardest. §6 lists every behaviour change and what it costs.

Size: 23 files, ~3,900 insertions. `setup_cloud_7wd.sh` alone is +677 lines, most
of which is a reordering (§1) rather than new logic.

---

## 1. What to read, in order

| File | What it holds |
|---|---|
| `setup_cloud_7wd.sh` | The reorder (flags before stage 8b), the new solver knobs, stage 6b's contended rate, stage 8c's sizing |
| `f4_staged_sweep.py` | **New.** Two-stage driver: geometry, then batching |
| `f4_phase_d_sweep.py` | `--sims-divisor`, `parked_slot_fraction`, `main(argv)` returning its payload |
| `sweep_launch_env.py` | Reads the staged block; `swept_axes` provenance |
| `solver_corpus.py` | **New.** Prices positions once, anywhere; `admission_ceiling` |
| `solver_sizing.py` | **New.** budget = threads x wall x rate x share; the stall column |
| `seven_wonders_rust/src/self_play.rs` | `SOLVER_ATTEMPT_NODES`, `cost_prediction_nodes`, `solver_predicted_nodes` |
| `seven_wonders_rust/src/cost_model.rs` | `FEATURE_COUNT` (was five hardcoded widths) |
| `endgame_trigger_study.py` | `measure_node_rate_contended`, `contention_curve` |
| `rehearse_sweep_laptop.sh` | Rehearses stages 6b, 8b, 8c on a laptop |
| `endgame_cost_model.json` | Refitted (§4) |
| `solver_corpus.json` | **New artifact**, 8,013 priced positions, 1.5 MB |

Tests: `test_solver_sizing.py` (20, new), `test_setup_cloud.py` (140),
`test_solver_scheduler.py` (26), plus the cost-model and parity suites.

---

## 2. The sweep now measures the run — and what that cost

`f4_phase_d_sweep --config-from-manifest` existed and the launcher never passed
it, so the box sweep measured Gumbel at 128 full simulations to configure a PUCT
run at 1600. `phase_d --emit-config` closes that: the launcher builds the run's
config from its own flags and points the sweep at it.

**This forced the reorder.** The flag assembly moved from stage 10 to before
stage 8b, because the sweep needs the run's description and the run's manifest
does not exist until the run starts. `TUNED_FLAGS` / `GATE_TUNED_FLAGS` are
appended at stage 10 instead of interpolated — they come OUT of the sweep. The
assembled launch line is byte-identical to before the move, verified by diffing
the dry run against `HEAD~`, and a line-level audit found exactly four changed
non-comment lines.

Each grid point then cost what an iteration costs, against 6 axes. Two knobs buy
it back:

* **`f4_staged_sweep`** — rank geometry (slots x caps x workers x solver split),
  then sweep inflight x wait at the winner. 18 + 6 points against 108.
* **`--sims-divisor`** (launcher default 4) — every point at 1/N of the run's
  simulations, with the solver's node budget divided by the same N.

**Where I expect a reviewer to push.** Staging assumes stage B cannot reorder
stage A, and there is one known place that is weak: the coalescing wait rewards
high shard counts, and stage A ranks shards at a pinned wait of 0.
`--stage-a-wait-ms` exists for it and the driver records which way it ran. I
think this is a real residual risk, not a solved problem.

---

## 3. Two solver caps, not one

`CostModel::affordable` is `predict + margin <= log10(budget)`, and one number
served as both the admission bar and the abandonment budget — so raising the
timeout widened admission by the same factor unless `margin_decades` was moved
by hand, in log space, to compensate.

`--endgame-solver-attempt-nodes` splits them. `0` resolves to the timeout, which
reproduces every earlier run; the getter reports the RESOLVED value so a manifest
never records a `0` that reads as "attempts nothing"; a bar above the timeout is
refused at both the pyo3 boundary and the Python knob.

**Measured, cloud2 iteration 96, one number at 40M:** 8,013 solves attempted,
7,754 answered (96.8%), 259 declined on nodes and none on the deadline. Median
answered solve 3,731 nodes — four orders of magnitude under the cap — and
**45.9% of all solver nodes went to the 3.2% that answered nothing**, because a
decline costs exactly the timeout.

**Measured, laptop soak 4, split at 100k/400k:** 11,501 attempted, **99.89%**
answered, 13 declines, **6.9%** wasted — against 35.4% on the previous soak's
single number.

`solver_predicted_nodes` is now on every attempted move, declines included,
because the declines are the rows a different bar would move.

---

## 4. The cost model, refitted

The model under-predicted its own tail monotonically: median residual +0.01
decades at 10^0-10^3 rising to +0.37 at 10^7-10^8, on 8,006 rows. The shipped fit
saw 2,235 positions with 165 (7.4%) censored, so the expensive tail was what it
could see least clearly. `resolve_censored` has since re-solved 252 of the 259
censored positions to true costs.

New fit: 8,013 positions, 823 games, 7 censored (0.09%). Held-out R² **0.9360 ->
0.9476**; on the same priced corpus, median residual +0.06 -> +0.02, censored
tail +1.08 -> +0.90, worst-1% +1.58 -> +1.45. The flattening is gone — the decile
ladder is within ±0.04 across six deciles, only the top at +0.14.

**Tried and dropped, recorded because the reasoning may be worth re-examining.**
Of the 24 positions predicted under 40M that cost over 300M, an unbuilt Great
Library is **4.08x over-represented**, present in 6 of the 8 worst cases at only
8-9 cards left. The mechanism is real — `chance_fanout` walks unrevealed CARD
slots and never visits the progress-token pool, so the Library's 3-of-5 draw
(log10 C(5,3) = 1.00 decade alone) was carried by a binary flag worth 0.658, and
a chance node multiplies the subtree rather than adding to it. Adding
`library_fanout` and `library_x_cards` bought **+0.0008 R²**. Reverted. My
reading is that the information was already reachable from
`chance_wonders` x `cards_left` and the residual is a limit of the linear-in-log
form — but I am not confident, and a reviewer who disagrees has a concrete lead.

---

## 5. Sizing the caps from the box

The split of labour, which `measure_node_rate`'s own docstring states: node
counts are a property of positions and identical on every box; only the RATE is
machine-specific.

* **Priced once, off-box:** `solver_corpus.json` — 8,013 positions with the 20
  cost features and the TRUE node cost, merging `resolve_censored` so the
  censored tail is a cost rather than a floor. Features, not predictions, so a
  refit reprices it without rebuilding.
* **Measured on the box:** stage 6b's node rate, the thread split, and stage 8b's
  generation wall.
* **Crossed at stage 8c:** `budget = threads x generation_wall x rate x share`
  (share 0.80), `demand = games x nodes_per_game(bar, timeout)`, choose the pair
  maximising proofs that fits. Appended to `measured_env.sh` so pass 2 sources
  one file.

**Stage 6b's rate is now measured CONTENDED.** `measure_node_rate` returns the
single-thread figure and says so; a run solves with `--solver-threads x
--rust-scheduler-workers` of them. Measured here:

| threads | nodes/s/thread | efficiency |
|---|---|---|
| 1 | 1,407,226 | 1.00 |
| 4 | 1,003,035 | 0.83 |
| 8 | 795,281 | 0.68 |

**1.77x apart**, so the old measurement would have sized the budget 1.77x
optimistic. 795k at 8 threads brackets the re-solve study's 857k on different
hardware.

**`admission_ceiling` is the subtle part.** A corpus holds what a run ATTEMPTED,
so anything its bar refused is ABSENT rather than expensive, and a wider
candidate bar prices identically to the collecting one — reading as "widening
buys nothing" when the truth is "this corpus cannot see". The ceiling is
RECORDED, not inferred: inference gives `10**(max(prediction) + margin)` =
39,975,202 against a true 40,000,000, which excluded the collecting run's own
settings — the one candidate that must always be priceable.

---

## 6. Behaviour changes, and what each costs

| Change | Default | Cost / risk |
|---|---|---|
| Sweep measures the run | on | Each grid point costs an iteration; mitigated by staging + divisor |
| `--sims-divisor 4` | on | games/hour from the sweep is ~4x the run's rate and is **not** a prediction |
| Staged sweep | on | Stage B cannot reorder stage A (§2) |
| `--exclude-parked-from-budget` | **on** (phase_d defaults off) | Redefines `--rust-slots` from concurrent games to concurrent SEARCHING games, so slot counts are not comparable across the two. Set before stage 8b for that reason. Records are byte-identical either way (`test_excluding_parked_slots_changes_no_record`) |
| Attempt bar 40M / timeout 320M | **changed from 40M/40M** | §7 |
| Cost model refitted | on | **Target-changing**: at a 40M bar the refit refuses 214 of the corpus's 8,013. "None newly admitted" was **wrong** and is withdrawn — it was computed ON the corpus, which by construction holds only what the old model admitted, so newly-admitted positions cannot appear in it. Counterexample: seed 116260767 move 64, required bar 48,162,395 under the collecting model and 34,617,215 under the refit. Buffers either side do not share a target definition |

---

## 7. The 320M timeout, and why the stall is accepted

Priced on `solver_corpus.json` at cloud2's rate and 12 solver threads:

```
  bar   timeout  proofs  nodes/iter  wasted   stall
  40M       40M   7,636      16.60B   39.3%     47s   <- one number, before
  40M      320M   7,786      28.49B   14.6%    373s   <- chosen
  40M     1280M   7,798      32.05B    4.0%   1494s
```

40M is the widest bar this corpus can price. Narrowing it costs hard proofs — it
filters on predicted cost, so it drops the expensive positions first, which are
the ones worth proving (at a 5M bar, hard proofs fall 1,572 -> 592).

**The stall is real and accepted with a weaker justification than I first
gave.** The scheduler cannot end an iteration while a game is parked on a solve
(`work_remaining()` is `active_count > 0`).

What is measured: across 97 cloud2 iterations the drain tail below 25% of peak
occupancy was a median **16.5%** of generation wall (~730s) at a 40M cap, and
idle after the last NN batch was **3s median, 3s worst**.

What that does NOT establish, and I previously implied it did:

* A 730s drain does not show that a 373s solve overlaps it. A solve that STARTS
  late extends the iteration by however much of it remains when everything else
  has finished, and the drain's existence says nothing about when solves begin.
* "Idle after the last NN batch" only catches a stall at the very end. A stall
  in the middle of the drain, followed by another batch, is invisible to it.

So the honest claim is narrower: at a 40M cap the solver contributes ~3s of
end-of-iteration idle, and the run already has a large low-occupancy tail that a
longer solve *may* overlap. Whether a 373s solve is absorbed is **not measured**.
`solver_sizing` flags it (it exceeds half the drain), and the box's own
`parked_slot_fraction` and batch timings are what would settle it.

I revised this twice: first claiming 747s was "16.9% of an iteration" as though
additive, then over-correcting to "absorbed". Neither is supported.

---

## 8. What is measured vs assumed

**Measured.** Everything in §3, §4, §5, §7. The contention curve. The soak's
solver behaviour. The drain tail across 97 iterations. The sweep's own
`parked_slot_fraction`, now recorded per point.

**Assumed, and I would like these challenged:**

1. **A proof is worth having.** Never A/B'd. There is no Elo, no win-rate, no
   arena comparison for the solver — cost is measured everywhere, value nowhere,
   and the mask docstring records that 77-88% of legal moves at these positions
   are already proven equally optimal. The operator's position is that an exact
   answer is categorically better than an estimate and BGA play shows it saving
   contested endgames. I record the gap without disputing the decision.
2. **The corpus transfers.** It is cloud2 iteration 96 — one net's endgames. A
   run reaches different positions as it strengthens.
3. **Thread-seconds is the binding constraint.** Only true because
   `--exclude-parked-from-budget` is now on. Without it a solve also withholds
   slot concurrency and every number in §5 understates the price.
4. **The drain fraction transfers.** 16.5% is cloud2's geometry (1,000 games over
   256 slots). The box sweep picks a different one; `--drain-fraction` takes a
   measured value.

---

## 9. Where I think a reviewer will find something

* **§2's staging assumption.** The one place I know it is weak is documented; I
  do not know whether there are others.
* **The sizer's objective** maximises proof COUNT. Hard proofs and total proofs
  happened to agree at the chosen point; they need not on a box with a different
  budget. There is no weighting by difficulty and no waste term.
* **`solver_sizing` reports the stall but never filters on it.** Deliberate, but
  it means the tool will reach for a 1,280M timeout if the budget allows.
* **`admission_ceiling`'s inference fallback** is off by exactly the amount that
  matters, and warns. A corpus built without `--manifest` is quietly less useful
  than one built with it.
* **The rehearsal's solver liveness is thin** — the toy grid attempted ONE solve,
  which passes a check written to catch zero. Liveness, not coverage.
* **`price()` is a counterfactual, not a simulation.** It holds the position set
  fixed; a real run would not, because a solve masks the policy target and
  changes the move sampled. Measured on a toy grid: 57 attempts against 64, with
  one position unique to the narrow run.

---

## 10. How to check it yourself

```bash
bash setup_dryrun.sh                    # executes the launcher against stubs
python -m pytest games/seven_wonders_duel/test_setup_cloud.py \
                 games/seven_wonders_duel/test_solver_sizing.py \
                 games/seven_wonders_duel/test_solver_scheduler.py -q
bash rehearse_sweep_laptop.sh           # stages 6b/8b/8c on a laptop GPU
                                        # REHEARSE_WORKERS_CSV=2,4 to exercise
                                        # the coalescing wait
python -m games.seven_wonders_duel.solver_sizing \
  --corpus games/seven_wonders_duel/solver_corpus.json \
  --rate 857015 --threads 12 --generation-wall-seconds 4422 \
  --games 1000 --target-share 0.80     # reproduces the §7 table
```


---

## 11. Review audit trail (2026-09-11)

Seven findings, every one reproduced against the code before it was fixed.

| # | Finding | Reproduced | Fix |
|---|---|---|---|
| P1 | The sweep never installed the manifest's cost model, so `solver_wants` fell back to `cards_left <= max_cards` with `max_cards = 0` — a test no Age III position passes. **Every point attempted zero solves.** | Structural: `cards_left >= 1` in mid-play, and `grep` shows `set_endgame_cost_model` was never called | `f4_phase_d_sweep` installs the model from the manifest into THIS process; the liveness check now requires `solves_with_prediction > 0`, which only the model can produce |
| P1 | `SWEEP_SOLVER_ARGS` passed the timeout but not the bar, and the harness's manifest fallback for the bar was nested inside the timeout's `if` — so an explicit timeout suppressed it and the split vanished for the sweep | Read: the `if solver_max_nodes <= 0` block encloses the bar lookup | Bar passed explicitly; each fallback resolves on its own absence |
| P1 | A recorded bar does not establish coverage under a **refitted** model | seed 116260767 move 64: unattempted, required 48,162,395 under the collecting model, 34,617,215 under the refit | The corpus records the collecting MODEL; `admission_ceiling` shrinks by `min(new_required / old_required)` when they differ, and says it is an extrapolation |
| P1 | `price` divided by the REQUESTED games and `size` multiplied by the same number, so they cancelled — demand was identical at 100 / 1,000 / 10,000 games while capacity grew with the wall | 28.494B at all three | The corpus records `collecting_games`; demand normalises by that and scales to the target |
| P1 | Stage 8c sized against stage 6b's PRELIMINARY thread split and rate, before the worker and solver axes were swept | Read: `_total_solver` and `NODE_RATE` are set at 6b and never replaced | Reads the winner's `scheduler_workers` / `solver_threads_total`, and re-measures the contended rate at that thread count |
| P2 | The generation wall came from a DIVIDED sweep, so at divisor 4 the budget was ~¼ of the truth — this file says three times that a divided sweep's games/hour is not the run's rate, then used it as one | Read: `median_games_per_hour` used directly | One confirmation point at the winner's geometry with `--sims-divisor 1`; the fallback scales and labels itself `EXTRAPOLATED` |
| P2 | Sizing was conditional only on the corpus existing, so a run with `ENDGAME_SOLVER_MAX_NODES=0` still got positive caps written into `measured_env.sh` | Read: the branch tests only `-f "$SOLVER_CORPUS"` | Skips sizing and preserves zero when the solver is off |

**Two claims withdrawn.** "None newly admitted" (§6) was circular — computed on
a corpus that by construction holds only what the old model admitted. And the
drain argument (§7) is weakened: a 730s drain does not establish that a 373s
solve overlaps it, and "idle after the last NN batch" misses a stall followed by
another batch.

**One consequence the reviewer did not raise but the P1 coverage fix forces.**
Under the refitted model this corpus can only price bars up to **8,092,786**
nodes, not 40M — so the 40M bar in §7 is *not* supported by it. The options are
to reprice at a bar the corpus covers, rebuild the corpus from a run that used
the refit model, or keep the collecting model for the box run. **This is an open
decision, not a fixed defect.**
