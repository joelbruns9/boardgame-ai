# The `*_vs_random` rows are real, and they point at an open bug

`summary.json` contains an ordering that cannot all be true at once:

| match | score |
|---|---:|
| `trained_iter_0_vs_greedy` | 97.0% |
| `latest_iter_11_vs_greedy` | 98.0% |
| `random_iter_-1_vs_greedy` | 16.0% |
| `trained_iter_0_vs_random` | **37.8%** |
| `latest_iter_11_vs_random` | **51.0%** |

`random_iter_-1` is `_bootstrap_init.pt` -- a *randomly initialised network*
played through 64-sim MCTS, not a random-move bot. The trained networks crush
the scripted bots, the random-init network loses badly to those same bots, and
yet the trained networks do not beat the random-init network head to head.

These two rows were initially suspected of being stale artifacts carried over
by the old runner's label-keyed resume. **They are not.** Two follow-up
experiments (2026-07-24) settled it.

## 1. Arena bookkeeping is correct

Arena games use `deterministic_actions=True` with paired seeds, so running
`A_vs_B` and `B_vs_A` at the same seed offset replays the *same* physical games
with the roles relabelled. The score rates must therefore sum to exactly 1.

```
iter0_vs_random  = 0.4688   random_vs_iter0  = 0.5312   sum = 1.0000
iter11_vs_iter0  = 0.4688   iter0_vs_iter11  = 0.5312   sum = 1.0000
```

Exact. There is no seat, scoring, or pairing bug in `_rust_model_gate_waves`,
and `iter0_vs_random = 0.4688` reproduces the anomaly on fresh seeds.

`eval_suite.py` now runs this check automatically before any match.

## 2. The effect is monotone in search depth, and runs the wrong way

Both checkpoints were confirmed to hold different weights first (91 comparable
tensors, 0 identical, mean |delta| = 0.0078), so this is not a loading problem.

```
sims=  2   iter0_vs_random = 0.1250
sims=  8   iter0_vs_random = 0.2188
sims= 32   iter0_vs_random = 0.3438
sims=128   iter0_vs_random = 0.6562
```

(32 games each, so roughly +/-17% per point; the trend across four points is
the signal, not any single value.)

The trained network is **worse** than a random-init network at low simulation
counts and only overtakes it somewhere between 32 and 128 sims. Search is
rescuing it, not amplifying it.

At 2 sims, play is essentially the raw policy prior. A network trained largely
on 1,000 greedy-bot curriculum games should have a serviceable prior; scoring
12.5% against a near-uniform prior means its prior is actively harmful. The
value head appears to be fine -- that is what more simulations are recovering.

## Why this matters more than the numbers themselves

* `gate_sims` is 64, right in the middle of the crossover region. The promotion
  gate and the Elo ledger both run at 64 sims, so both were measuring a regime
  where net quality and search interact strongly.
* Self-play uses 16-24 cheap sims for ~75% of moves. If the prior is harmful at
  low sims, most generated positions are being played out at close to the worst
  point of this curve.
* The bot anchors at the same 64 sims look healthy, so the network is not
  globally broken. Something specific to the policy prior is.

## RESOLVED (2026-07-24), FIXED (2026-07-25): actor-routed evaluation

All three defects below are fixed. The sims sweep that exposed this, iter0
against a random-init net, 32 games per point:

| sims | before | after |
|-----:|-------:|------:|
|    2 |  0.125 | 1.0000 |
|    8 |  0.219 | 0.9688 |
|   32 |  0.344 | 1.0000 |
|  128 |  0.656 | 1.0000 |

Search now improves on the raw prior (0.850 by argmax with no search) instead
of destroying it, and matches the clean-Python control of 0.925 / 1.000. Rust
and Python remain bit-identical (30/30 equivalence tests); 448 tests pass.

### The defect

`rust_seat_routed_flat_batch_adapter` (`rust_bridge.py:356-366`) routes every
packed row to the net of **that row's acting player**.  Inside a single MCTS
tree, leaves where the opponent is to move are therefore evaluated by the
OPPONENT's network.  Neither player ever runs a clean search with its own net.

A strong net facing a random net gets random values for every opponent reply --
its search is poisoned.  The random net simultaneously gets the strong net's
values for *its* opponent's replies -- its search is helped.  The advantage
inverts.

Reproduced in pure Python by routing evaluations the same way, changing nothing
else:

```
sims=2   iter0_vs_random   clean=0.9250   actor-routed=0.2250
sims=8   iter0_vs_random   clean=1.0000   actor-routed=0.0750
```

Clean search improves with simulations (prior argmax 0.850 -> 0.925 -> 1.000).
Actor-routed search *degrades* with simulations, because each extra simulation
adds more opponent-to-move leaves evaluated by the wrong network.

### Why it stayed hidden

* Bot anchors use `rust_flat_batch_adapter` -- a single net, no routing.  The
  97% vs greedy figure is sound.
* Self-play uses a single net.  Training data and targets are unaffected.
* F3.4 (`test_closed_search_net_matches_python`) uses a single net through a
  per-row adapter.  F4.6 compares flat vs scalar with a mock evaluator.
* `test_f4_rust_seat_routed_adapter_uses_the_actor_checkpoint` zeroes every
  parameter and asserts only the sign of the root value.  It proves routing
  happens; it cannot show routing is wrong for search.
* The flat adapter's arithmetic is correct -- checked against `Evaluator`,
  max |dpolicy| = 6.0e-08, zero argmax disagreements.

### The fix

Each player's search must see only its own network.  Either:

1. route by the **root actor** (whose turn it is in the game the row belongs
   to) rather than the row's own actor -- needs a Rust-side change to pack a
   root-seat byte per row; or
2. run arena games through the single-net path, alternating the net by whose
   turn it is (what the Python `SearchAgent` gate path already does).  Correct
   but slower: two nets can no longer share one scheduler pass.

Actor-relative values are fine under either -- a net evaluating an
opponent-to-move node returns that node's actor-relative value and the tree
negates as usual.

### Secondary issues found in the same investigation

* `SearchResult.action_index` is the Gumbel-perturbed action and every
  evaluation path plays it (`phase_d.py:425`, `self_play.rs:365`, `:783`).
  `deterministic_actions=true` only suppresses the extra temperature sampling.
  Evaluation should play `argmax(policy_target)`, which is built from the same
  logits without the Gumbel keys.  Costs real strength: iter11 vs iter0 at 8
  sims scored 0.35 with Gumbel vs 0.575 with the target argmax.
* `_sigma(q, max_visits) = (c_visit + max_visits) * c_scale * q` with
  `c_visit=50`, `c_scale=1.0` and raw `q` in [-1, 1].  Those are the Gumbel
  AlphaZero constants, but the paper applies them to a min-max **normalised**
  Q.  Unnormalised, sigma spans +/-50 while log-prior differences are ~1-3, so
  the prior contributes almost nothing to `policy_target`.
* Phase E's `trap_pick` / `unsafe_pick` / `action_regret` were computed from
  `result.action_index`, so the E-Tier-1 verdict rested on Gumbel-perturbed
  action choices. Fixed; see below.

### Open question -- ANSWERED 2026-07-25, see "Depth compression" below

The Rust sims sweep recovered with depth (0.125 at 2 sims -> 0.656 at 128)
while the Python reproduction degrades with depth.  The Python run used default
`force_root_chance=False` / `age_deal_samples=0` against the arena's `True` /
`32`, so the two are not exactly comparable.  The routing bug is confirmed
either way, but the depth behaviour is not fully explained.

### Not affected

The prior itself is good and improved steadily across the run -- by argmax with
no search at all: iter0 beats random 0.850, iter11 beats random 0.967, and
**iter11 beats iter0 0.950**.  The run learned far more than its own metrics
showed.  The advisor is also clean: it uses plain PUCT `descend()`, never
`_gumbel_root`, and runs a single net.

> **Caveat added 2026-07-25.**  That 0.950 is condition-dependent and should
> not be quoted as "iter11's true edge".  It was measured in the Python control
> at `force_root_chance=False` / `age_deal_samples=0`; re-measured through the
> fixed arena under the settings the gate actually uses, the same no-search
> comparison gives **0.805**, not 0.950.  The qualitative claim survives -- the
> run learned much more than its metrics showed -- but the number does not.


## Fixes applied (2026-07-25)

1. **Routing.** `_rust_model_gate_waves` no longer uses
   `rust_seat_routed_flat_batch_adapter`. Games are stepped from Python
   (`PhaseDLoop._play_two_net_games`); each ply partitions live games by whose
   turn it is and issues one batched `search_many_flat_net` per seat, so a
   whole search always runs under the mover's own network. Games sharing a
   mover still batch together, so roughly half the rows per batch versus the
   old shared-tree call. Needed three new `RustGame` getters (`actor`,
   `winner`, `victory_type`, `final_scores`).

2. **Played action.** Evaluation now plays `argmax(policy_target)` instead of
   `SearchResult.action_index`. Three sites: `phase_d.py` `SearchAgent`, and
   both `deterministic_actions` branches in `self_play.rs` (via the new
   `best_policy_action`, first-max tie-break over legal order to match
   Python's `max`). Self-play is untouched -- it still samples `policy_target`
   with temperature, which is correct.

3. **Sigma normalisation.** `sigma` now min-max rescales completed Q to [0, 1]
   across the root's legal actions before applying `(c_visit + max_visits) *
   c_scale`, and `c_scale` drops 1.0 -> 0.1 (mctx's `value_scale`). Applied in
   `search.py`, `tree.rs` (`sigma_vector`, now shared by `tree_resumable.rs` so
   the two cannot drift), and all 12 pyo3 defaults in `lib.rs`.

### Consequence for existing data

Fix 3 changes `policy_target`, which is a **training target**. Every
`policy_target` already recorded in `buffer_final.jsonl` and `buffers/*.jsonl`
was computed under the old unnormalised sigma. Mixing those with newly
generated games trains on two inconsistent target definitions; start the next
run from a fresh buffer, or accept the inconsistency knowingly.

4. **Phase E metrics.** `phase_e.py` now scores traps, regret and Q error
   against the action a competitive game would PLAY (`argmax(policy_target)`)
   rather than `result.action_index`, and records `gumbel_action` /
   `gumbel_disagreed` alongside so the size of the old distortion is visible in
   new runs. `RESULT_SCHEMA` bumps 2 -> 3, which makes the loader skip every
   schema-2 row automatically: the recorded E-Tier-1 numbers are superseded and
   the evaluate stage must be re-run before the report stage will accept them.

   `SearchResult` gained a `completed_q` map (action -> completed Q) for this.
   `action_value` only ever held the Gumbel-selected action's Q, so a caller
   that plays a different action had no way to ask for its value.


## Post-fix retest (2026-07-25)

The Rust extension had to be rebuilt first: the installed `.pyd` was built at
00:05 while `lib.rs` / `self_play.rs` / `tree.rs` / `tree_resumable.rs` were
last edited at 00:42, so the compiled module predated fixes 2 and 3.  Anything
measured before that rebuild was still running the old search.

iter11 vs iter0, 400 games, 64 sims, seed 20260724 at offset 70,000,000 -- the
same seeds as the original run, so this is a paired comparison:

| | before | after |
|---|---:|---:|
| iter11 score | 0.595 | **0.650** |
| paired 95% CI | [0.548, 0.642] | [0.604, 0.696] |
| record | 238-162-0 | 260-140-0 |

Seat-balanced (0.34 / 0.36 for iter0 as seat 0 / seat 1).  Victory mix barely
moved (civilian 308->303, military 72->62, scientific 20->35): the fixes
changed who wins, not how games end.  Throughput fell 0.194 -> 0.145 games/s,
the expected cost of one batched search per seat per ply instead of one shared
tree.

Arena symmetry still exact on the new Python-stepped path: `0.3125 + 0.6875 =
1.0000`.


## Depth compression (2026-07-25) -- answers the open question above

The gap between iter0 and iter11 narrows as search deepens, under matched
conditions.  Same seeds, same offset, same `force_root_chance=True` /
`age_deal_samples=32`, 400 games each, only `sims` varying:

| sims | iter0 | **iter11** | 95% CI (iter0) |
|-----:|------:|-----------:|---|
|    2 | 0.195 | **0.805**  | [0.154, 0.236] |
|   64 | 0.350 | **0.650**  | [0.304, 0.396] |

Non-overlapping intervals, so the effect is real.  In Elo, iter11's edge falls
from **+246 to +107** -- deeper search erases about **57% of the gap**.

**This is not a bug.**  Search is a policy-improvement operator, and it
extracts more from a weak prior than from a strong one, so a head-to-head
compresses as both sides get more of it.  iter0's profile is exactly the one
search rescues most: the routing investigation already concluded its value head
was fine and its prior was the problem.  Supporting evidence that nothing is
still mis-routed -- iter11 vs the random-init net *improves* with depth
(0.969 at 2 sims, 0.969 at 8, 1.000 at 32), and the equivalent check is now a
standing regression test (`test_gate_strength.py`).

Resolution of the original discrepancy, which was roughly half artifact and
half real: the 0.950 no-search figure came from an unmatched Python control;
matched, it is 0.805.  The remaining 0.805 -> 0.650 is genuine compression.

### Consequence for `gate_sims`

`gate_sims=64` while self-play generates at `cheap_sims` 16-24 means the
promotion gate judges a regime the training data never comes from, and it sees
only ~43% of the improvement that generation-regime play would show.  This
systematically under-credits policy-head gains -- which is precisely what this
run was making.

There is a real tension here rather than one right answer.  Gating at
generation sims matches the regime the data comes from and makes the ratchet
responsive to the improvements actually being produced.  Gating at 64 is the
more honest measure of deployed strength if the end product is a
deep-searching advisor.  Suggested split: gate the promotion ratchet at
generation sims (it is a data-quality decision) and keep the existing periodic
high-sims anchor (`anchor_gate_every_promotions=3`) for true-strength
tracking.  That is a parameter change; the two-tier structure already exists.
