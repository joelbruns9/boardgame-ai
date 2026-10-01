# Review request: Can't Stop training targets, search options and personas

A review of everything built since the Phase 3 Rust port review
(`PHASE3_RUST_PORT_REVIEW_REQUEST.md`, closed by `3d72ef3`):
- how the value net is trained: the replay window, TD targets and a
  learning-rate schedule;
- three new search options: exact-root values, a selective 2-turn
  lookahead, and stop-bias personas;
- two diagnostic tools whose conclusions are steering the next runs.

The design history and every measurement are in `VARIANT_SOLVER_PLAN.md`
(Phase 3 "Search options", "Value-net diagnosis" and "Self-play
personas"). This brief is the part a reviewer needs.

Branch `cantstop-variant-solver`, reviewed range `3d72ef3..403ad7d` (21
files, ~2k lines added).

**Why now:** the next run (λ=0, exact targets, personas, 10-iteration
window, resumed from the current best) is the first to rely on all of this
at once. §4.1–4.5 are the places where a silent error would waste that run
rather than crash it.

## 1. Scope

| Commit | What | Primary files |
|---|---|---|
| `01d1fc5` | Replay window (iterations) and lifetime passes | `train.py` (`ReplayBuffer`, `steps_for_passes`) |
| `594f648` | TD(λ) value targets | `self_play.py` (`td_targets`, `stack_training`, `turn_values`), `model.py` (`masked_soft_cross_entropy`), `src/selfplay.rs` (per-turn root values) |
| `cdf32ef` | Exact-root values; selective 2-turn lookahead; per-seat search; per-round row budget | `self_play.py` (`Search`, `play_turn`), `lookahead.py`, `src/lookahead.rs`, `src/solver.rs` (`leaf_reach`, `rebackup`), `src/selfplay.rs` (Game stages), `rust_pool.py`, `arena.py`, `lookahead_arena.py` |
| `403ad7d` | Stop-bias personas; unbiased persona targets; LR schedule; diagnostics | `solver.py` / `src/solver.rs` (`stop_bias`), `train.py` (`persona_seating`, `lr_at`), `value_accuracy.py`, `capacity_probe.py`, `run_pool` (`starts`, `allow_unfinished`) |

Tests: `test_td_targets.py`, `test_lookahead.py`, `test_personas.py`, plus
the additions to `test_train.py` and `test_rust_pool.py`.

## 2. What it does

- **Training targets.** A row is the board after turn *i*. Its target is
  the λ-return
  `G_i = (1 − λ) · v_{i+1} + λ · G_{i+1}`, where:
  - `G` at the end of the game is the one-hot winner;
  - `v_{i+1}` is the solver's value at the next turn's root.

  An opening-roll bust with no solve passes `G` through unchanged. λ is
  applied when rows enter the buffer, using values recorded by the net
  that played the game. Targets are soft distributions over seat slots,
  trained with `masked_soft_cross_entropy`.
- **Where the next turn's value comes from:**
  - **sampled** (default): the turn is solved after its opening roll, so
    the value is for that roll only;
  - **exact** (`exact_root`): the turn is solved before the roll, so the
    root roll value is the exact expectation over every opening roll,
    busts included. Moves are identical either way; only the recorded
    value changes.
- **Replay window and passes.** The buffer keeps the last `--replay-window`
  iterations. Each iteration trains on `passes × buffer_rows ÷ window`
  samples, so every row is sampled `passes` times over its life, however
  thin the buffer was when it arrived.
- **Selective 2-turn lookahead** (`lookahead_k`):
  1. Rank the end-of-turn boards by `leaf_reach`, the probability that
     best play reaches each board's stop-or-roll decision. The bust board
     scores by bust probability.
  2. Refine the bust board and the top k−1 stop boards: solve the next
     player's whole turn from each, from before their roll.
  3. Chosen boards take the refined value. With `offset`, every other
     board shifts by the reach-weighted mean refinement.
  4. Back up again.

  Every sum runs in a fixed order so Rust and Python match bit for bit.
- **Personas** (`stop_bias`, per seat). A player stops when stop value +
  bias ≥ roll value: +0.03 is conservative, −0.03 aggressive. A persona
  **plays** biased but **records the best-play value** as its target,
  from the same table backed up once more without the bias. The mix is
  set with `--conservative` / `--aggressive` fractions: one persona seat
  per chosen game, rotating seat.
- **LR schedule.** `--lr-schedule ITER:LR …` is a step schedule. Before
  each iteration the rate is written into Adam's parameter groups.

## 3. Already gated (please don't re-verify by hand)

Python is the reference. Every comparison is exact.

- **TD targets:**
  - λ=1 reproduces the old winner labels exactly;
  - λ=0 equals the rotated next-turn values, checked by an independent
    walk;
  - a hand-worked 3-row game that includes a bust;
  - targets are distributions over live seats;
  - the soft loss equals the hard loss on one-hot targets;
  - the pool records `turn_values` bit-identical to Python's.
- **Exact root:**
  - the pool plays exactly the plain games (same moves and rows, all 10
    rule sets);
  - the root value equals the probability-weighted average of the
    after-roll values over all 109 roll classes.
- **Lookahead:**
  - per turn from mid-game positions, bit-exact against Python in 3 rule
    sets × 2 mocks × {exact, k2, k3 without offset, k3 exact};
  - mixed per-seat searches;
  - reach splits into bust, taken stops and wins, summing to 1;
  - `refine` replaces the chosen boards and applies one common offset;
  - 3 of 3 planted bugs caught: offset never applied, least-reached boards
    chosen, reach ignoring stops.
  - Manual whole-game check: 5 whole games bit-exact with Python (3 with
    lookahead, 2 with exact root).
- **Personas:**
  - bias direction moves bust risk the right way;
  - ±1 bias always or never stops;
  - bias 0 is identical to best play under `mover_wins` (every decision
    on a rounding error);
  - Rust matches Python per turn, including mixed persona seats;
  - a persona's recorded value equals best play's exactly from the same
    position and roll, while its moves differ;
  - both backends play identical persona games;
  - no random numbers are drawn when both fractions are 0;
  - 2 of 2 planted bugs caught: flipped bias sign, and recording the
    biased value.
- **Window and passes:** a simulated run with uneven iteration sizes
  gives every full-lifetime row exactly `passes`.

## 4. Focus areas (what I could not settle myself)

### Targets: highest risk for the next run

1. **Row/turn alignment in `td_targets`.** Row *i* pairs with
   `turn_values[i + 1]`, and each row's seat to move is recovered as
   `(winner − winner_slot) mod n`. The tests are consistent with this,
   but they share its assumptions. An off-by-one here would train every
   board on the wrong turn's value without failing anything. Please read
   it cold.
2. **λ=0 with a 10-iteration window.** Targets are frozen at generation,
   so the oldest rows were bootstrapped from a net 10 iterations old.
   PureTD trains on fresh data only.
   - Is 10 a reasonable window, or should targets be recomputed (reanalysis)?
   - Is there any fixed-point failure mode specific to this setup that
     we should watch for in the log? Exact wins anchor only the final
     turn.
3. **Exact-root value semantics.** At a turn start the root value
   includes the bust branch, scored by the net on the bust board. For an
   opening-roll bust in exact mode, the recorded value and the row both
   exist; in sampled mode that turn's value is `None` and G passes
   through. Are the two modes' targets estimating the same quantity?
4. **Unbiased persona targets.**
   - Recording best-play values keeps the targets' meaning uniform, and
     makes the net value positions as if both sides play their best.
   - With λ>0, some persona outcome still leaks into targets; at λ=0,
     none does.
   - The lookahead's refinements of the next turn always run unbiased.

   Is this the right semantics for a net meant to play humans who stop
   early? Or should persona turns simply be excluded from the targets?
5. **Loss on soft targets.** Absent seats have zero target mass and are
   zeroed out of the log-probabilities. Is there any gradient pathway
   through masked logits that the `masked_fill` misses?

### Lookahead design

6. **Is the offset correction sound?** It shifts every unrefined board
   by the reach-weighted mean refinement, so an option isn't favored just
   for having been refined. With k small, most of a turn's reach goes
   unrefined: measured, the top 3 boards hold ~19% of it and the top 15
   ~31%. Is there a better-founded way to mix refined and unrefined
   values, or a better way to choose which boards to refine?
7. **Reach as the ranking.** A stop board scores by the probability of
   *arriving at* its decision, whether or not best play stops there, so
   rejected stops still score. Reasonable, or should it be
   margin-weighted, favoring decisions close to a tie?
8. **Memory.** The per-round row budget (`max_rows`) bounds the net's
   batch, but not the live solver tables. k=16 with exact root exhausted
   host memory at 64 games in flight. Should the pool budget tables
   too, for example by admitting games until estimated table memory
   reaches a limit?

### Training loop

9. **Resume semantics.** `--init-checkpoint` loads weights only: Adam's
   moments reset and the buffer starts empty. With the lifetime-passes
   rule, the first `window` iterations train lightly. Acceptable for a
   resume, or should warm restarts keep a minimum step count?
10. **LR schedule applied by overwriting `param_groups`** each iteration.
    Any interaction with Adam's bias correction worth noting?

### Diagnostics whose conclusions steer decisions

11. **`value_accuracy.py`.**
    - The truth is 400 rollouts per board under plain one-turn play by
      the same net.
    - Each estimator's squared error has the binomial noise
      `p̂(1−p̂)/(R−1)` subtracted.
    - Estimators are compared by paired differences per board.

    Is the noise correction right? And is "value under current play" the
    right truth for judging a *two-turn* estimate, which assumes better
    play for one turn than the rollouts use?
12. **`capacity_probe.py`.** The data is split by game; four sizes get
    the same epochs and a cosine-decayed learning rate; the best held-out
    loss is compared. All sizes tied at 0.520. Does that support "not
    capacity-bound at this data volume," given the targets were sampled
    λ=0.7 (noisy)?

## 5. Known limitations (not asking you to find these)

- **Experiment results are single runs:**
  - overnight1's plateau;
  - the LR step-down at +~1 point;
  - exact targets at λ=0.7 giving no gain;
  - the capacity tie;
  - the accuracy check at 60 boards (the gains are directional; the
    one-turn vs two-turn difference is within noise).
- **Throughput figures** are from one laptop with an untrained net:
  - plain 30.2k games/h;
  - exact 6.0k;
  - k=4 1,260;
  - k=16 189.
- **The first per-turn lookahead gate was vacuous.** Its size filter
  matched no positions. It now takes the smallest positions available
  and asserts it played.
- **No reanalysis:** stale bootstrapped targets are accepted within the
  window.
- **Opponent modelling isn't built.** The net values positions assuming
  best play by both sides.
- **The test suite takes ~20+ minutes**, dominated by the Python
  reference games.

## 6. How to run

From `boardgame-ai-cantstop`, with the main folder's venv and the
extension built (`maturin develop --release` in
`games/cantstop/cantstop_rust`):

```
python -m pytest games/cantstop/tests/test_td_targets.py games/cantstop/tests/test_lookahead.py games/cantstop/tests/test_personas.py -q
python -m pytest games/cantstop/tests -q                       # full, ~20+ min
python -m games.cantstop.value_accuracy --net runs/overnight1/iter_0400.pt --boards 60 --rollouts 400
python -m games.cantstop.capacity_probe --net runs/lr1e4/iter_0120.pt --games 20000
python -m games.cantstop.lookahead_arena --net runs/mvp2/iter_0010.pt --k 4 --games 400
```

## 7. Sign-offs requested

1. **TD target construction** (§4.1, 4.3, 4.5): correct as built.
2. **The next run's recipe** (§4.2): λ=0 with exact targets and a
   10-iteration window, without reanalysis. Proceed, or change first?
3. **Persona target semantics** (§4.4): keep "biased play, best-play
   targets", or exclude persona turns.
4. **Lookahead** (§4.6–4.8): whether the offset design is worth further
   investment, or whether its measured costs and small accuracy gain say
   to park it.
5. **Diagnostic methodology** (§4.11–4.12): sound enough to act on.
