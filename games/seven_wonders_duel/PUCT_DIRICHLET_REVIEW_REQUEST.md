# Review request: PUCT self-play root + Dirichlet exploration noise

**Status:** implemented, tested, **not yet run in training**. Every flag defaults
to the existing behaviour, so this is inert until explicitly enabled.

**Reviewer:** please weight §4 and §5 heaviest. §4 is a defect I shipped and then
found; I want to know whether the same class of mistake is still present
elsewhere. §5 is where I am guessing rather than measuring.

---

## 1. Why

7WD self-play has always used the Gumbel root. Two measured findings motivate
switching to PUCT for a final run:

1. **Gumbel targets carry no information about whether the move choice matters.**
   `_sigma` min-max normalises completed Q across root actions, so the target's
   sharpness is independent of how much the actions actually differ. Measured
   over 150 positions: target entropy is flat across a **24x** range of raw Q
   spread (Q1 spread 0.025 -> entropy 0.315; Q4 spread 0.613 -> entropy 0.325).
   Lowering `c_scale` cannot fix this — it reverts the target toward the prior
   (`argmax(log_prior + scale*q) -> argmax(prior)` as `scale -> 0`), and accuracy
   falls with confidence (0.763 -> 0.674).
2. **The defect is specifically costly at deployment.** The advisor runs PUCT,
   which consumes prior *magnitude* as an exploration budget
   (`Q + c_puct*P(a)*sqrt(N)/(1+n(a))`). A prior whose confidence is uncorrelated
   with value differences misallocates search.

Visit-count targets have neither problem. Measured on a real checkpoint:

| target | median entropy | fraction >0.95 on one action |
|---|---|---|
| Gumbel sigma @ 96 sims | 0.093 nats | 55.3% |
| PUCT visits @ 200 sims | 0.626 nats | 31.3% |

Dirichlet noise is required because Gumbel's exploration lives in its keys; PUCT
has no other source and collapses toward deterministic lines without it.

## 2. What changed

~440 lines across 9 files. Ported from Kingdomino (`mcts_az.py:729`,
`kingdomino_rust/src/lib.rs:4783`) with three deliberate departures (§3).

| file | change |
|---|---|
| `portable_rng.py` | `normal()`, `gamma()`, `dirichlet()` |
| `rng.rs` | same three, + 4 golden tests at **exact f64 equality** |
| `search.py` | `dirichlet_epsilon`/`dirichlet_alpha`; `_add_dirichlet_noise`; clean-prior snapshot before blend |
| `tree.rs` | config fields; `blend_dirichlet` (shared math); two clean-prior snapshots |
| `tree_resumable.rs` | applies the blend; `clean_priors` on the session; both result paths use it |
| `self_play.rs` | config fields; **net-id gating**; validation |
| `lib.rs` | plumbing through ~14 sites; pyo3 params defaulting to inert |
| `phase_d.py` | `--selfplay-search-mode`, `--dirichlet-epsilon`, `--dirichlet-alpha` |
| `test_dirichlet_noise.py` | 16 tests (new) |

Suites green: 108 (searcher/PUCT/equivalence/Dirichlet), 103 (Phase D), 20 (Rust).

## 3. Departures from Kingdomino — please sanity-check these

**(a) `alpha = 1.8`, not KD's 0.3.** Convention is `alpha ~ 10 / branching`:
0.3 chess (~35 moves), 0.15 shogi (~92), 0.03 Go (~250). 7WD measures **median 4,
mean 5.6** legal actions over 28,533 decisions. Measured consequence: at
alpha=0.3 over 5 actions the mean max noise component is **0.638** vs **0.392**
at 1.8 — 0.3 dumps most of the mass on one arbitrary action.
*Caveat: the branching histogram came from RANDOM playouts, not trained play.*

**(b) Draws come from `PortableRng`, not numpy.** KD concedes noise-on search is
not bit-comparable and gates equivalence at eps=0. 7WD's Gumbel keys already come
from the portable stream, so self-play is bit-comparable *with exploration on*;
using numpy would have given that up. Required adding portable `gamma`, hence
`normal`. Verified: at eps=0.25 the noise changed the chosen move (21 -> 27) and
**both implementations changed to the same move**.

**(c) ~~The noise is scaled to the existing prior mass~~ — WITHDRAWN on review.**
I had used `(1-eps)*p + eps*noise*total` so `eps` kept its nominal meaning on
unnormalised priors (a mock evaluator returns priors summing to 2.53). The
reviewer's decisive objection: production already normalises in
`self_play.rs::blend_priors`, so the scaling bought nothing there — and on the
mock it *also* preserved a 2.53x inflated `c_puct * prior` exploration term,
concealing the contract violation instead of surfacing it. I had reasoned only
about epsilon's meaning and missed what else rides on prior magnitude.
Reverted to the standard AlphaZero blend in both languages.

## 4. A defect I shipped and then found — is this class of error still present?

**`tree::puct_root` is not the searcher production self-play uses.** Self-play
runs through `tree_resumable` (`self_play.rs:560`, `:1543`, `:1550`). I put
`add_dirichlet_noise` in `tree.rs` only, so noise was **completely inert in a
real run** while the flag read 0.25.

My cross-language parity test passed anyway, because `rg.closed_search` routes to
`tree.rs` — it exercised the reference path, not the production one.

Caught by a behavioural check, not by reading:

```
self-play eps=0.00: mean target entropy = 0.699
self-play eps=0.25: mean target entropy = 0.699   <- bit-identical
```

Also found in the same pass: `make_search_meta` (the cooperative path production
actually uses) had `dirichlet_epsilon: 0.0` hardcoded, because a bulk edit
matched only one indentation level.

**Both are fixed**, and the fix is confirmed behaviourally rather than by reading:

```
before:  eps=0.00 -> entropy 0.6988   eps=0.25 -> entropy 0.6988   (inert)
after:   eps=0.00 -> entropy 0.6988   eps=0.25 -> entropy 0.8236
```

**A third defect, found by that same verification run.** `blend_dirichlet` gated
on `puct_root && epsilon > 0` only, so **cheap moves received noise too** —
contradicting the Kingdomino design this port is based on
(`fast_move_dirichlet_epsilon = 0.0`). Cheap moves emit no training targets, so
noise there buys no exploration of the label space and only degrades the
trajectory, and they are 75% of all moves. Now gated on `full && net == 0`.

The underlying hazard is that this crate has *two* searchers with separate
`Node`/`Edge` types and overlapping responsibilities, and the tests
preferentially cover the scalar one. Reviewer: please check whether any other
config field is honoured in `tree.rs` but silently dropped in
`tree_resumable.rs`.

**A note on how these were found.** All three were invisible to the test suite —
108 tests passed with the feature completely inert. What caught them was a
behavioural probe comparing a summary statistic across epsilon arms. Reviewer:
consider whether a permanent test of that shape belongs in the suite, since the
unit tests demonstrably could not distinguish "wired up" from "not wired up".

Mitigation applied: the blend math now lives once (`tree::blend_dirichlet`) over
a bare prior slice, with a thin adapter per searcher.

## 4b. External review round 1 — findings and disposition

An external reviewer raised four issues. All are resolved; 215 tests pass.

| finding | disposition |
|---|---|
| **[P1]** Python generation backend dropped `root_selection` and both Dirichlet fields — `--generation-backend python --selfplay-search-mode puct` would record PUCT and generate Gumbel | **Confirmed, fixed.** Wired in `_search_move`, with `full` gating to match the Rust generator |
| **[P1]** Cheap learner moves received noise | **Already fixed** ~1h before the review; independent identical diagnosis |
| **[P2]** Non-finite alpha accepted (`NaN <= 0` is false) | **Confirmed, fixed** at three layers: `Rng::gamma`, `PortableRng.gamma`, `PhaseDConfig.validate`. Python **hung forever** on NaN (every comparison in the rejection loop false); infinity silently produced an all-NaN Dirichlet, pinning selection to edge 0 |
| **[P2]** PUCT inference width capped at active-game count | **Partially.** Mechanism real; the quoted 16 slots / 256 cap are pyo3 defaults, not production (`--rust-slots 256`, `--rust-global-batch-cap 2048`, observed mean batch 320). Crucially `leaf_batch` defaults to 1 and production never overrides it, so **Gumbel runs under the same ceiling today** — not a regression. But it removes an escape hatch PUCT cannot use without the virtual-loss entry point, and the ceiling is untested at 1000 sims. Moved to the pre-launch measurement list |

**The pattern worth noting:** three defects found internally and two by the
reviewer were *all* invisible to a passing suite. At one point 108 tests passed
with the feature completely inert. Unit tests here verify that code computes what
it computes; none of them could distinguish "wired into the production path" from
"not wired in at all". Every one of these was caught by a behavioural probe
comparing a summary statistic across epsilon arms, or by reading call graphs.

## 5. Assumptions I am not confident in

1. **`epsilon = 0.25` is copied from KD/AlphaZero and never tuned for 7WD.**
2. **`alpha = 1.8` rests on a random-playout branching histogram.** Trained play
   may have a different distribution.
3. **Net-id gating assumes network 0 is always the learner.** It reuses the
   predicate `policy_excluded` already relies on
   (`self_play.rs:1637`), so it is consistent — but nothing enforces the
   invariant. This is what keeps ARCHIVED opponents noise-free; league games are
   ~15% of training data, and a handicapped opponent would bias their value
   labels optimistic.
4. **Per-seat routing would have been wrong here** and I built it first before
   noticing: `LeagueAssignment` alternates the archive across seats within an
   iteration, so a fixed p0/p1 pair cannot express "the archive plays clean".
5. **Sub-1 alpha uses `powf`**, which has no correct-rounding guarantee. Golden-
   tested today; could drift across toolchains. Production alpha >= 1 avoids it.
6. **Cheap moves at 100 sims with eps=0 under PUCT is untested.** KD's precedent
   (`fast_move_sims=100` against `n_simulations=100`) does not cover the intended
   100-against-1000 ratio; at 100 sims PUCT is close to greedy prior play, and
   cheap moves are 75% of all moves.
7. **No evidence any of this learns better.** Everything established is
   mechanical: parity, target shape, plumbing. The 7x entropy result came from
   6 games / 115 moves on a *Gumbel-trained* net.
8. **`check_puct_root` rejects `leaf_batch > 1`** unless the virtual-loss variant
   is used. Production runs `leaf_batch=1`, so this is currently moot.

## 6. Invariants a reviewer can check quickly

- Noise applies **only** when `puct_root && epsilon > 0`. Gumbel is untouched.
- Noise **never** reaches a training label: `policy_target` comes from visits;
  the recorded `prior` and the terminal policy fallback both use a snapshot
  taken before the blend. Two snapshots per searcher, four in total.
- Evaluation is noise-free **by construction**: the gate/anchor calls
  (`phase_d.py:3769`, `:3926`) do not pass Dirichlet, so they take the pyo3
  default of 0.0. There is no separate eval flag to forget.
- Every new flag defaults to current behaviour.
