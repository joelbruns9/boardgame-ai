# Everything that should land before the next 7WD training run

**Written to be picked up cold.** Every number here was measured on this machine
(2026-08-16/17) and is reproducible from the committed tools. Where a number
contradicts an earlier claim elsewhere in the repo, this file is the later
measurement and the earlier one has been corrected at its source.

The retrain is the gate on everything: **no checkpoint loads on the current
encoder** (all 580 carry `7d68ff20…`), and **no recorded self-play replays**
(0 of 100 cloud6 `iter_0010` games). So there is no strength number, no error
tail, and no on-distribution cost profile until a net exists. That makes the
retrain a prerequisite rather than housekeeping — and makes it worth landing
the changes below first, since each one costs a retrain to add later.

---

## 1. Where things stand

Already committed on `sevenwd-engine-correctness`:

| | |
|---|---|
| Both BGA-found engine defects | fixed (`ec08bc1`) — discard reachability, military band off-by-one |
| Exact endgame solver in self-play | committed, **off by default** (`2a509ab`, `a820979`) |
| Oracle probe (value head vs proven values) | `solver_oracle_probe.py` (`cb5cd2e`) |
| Trigger study + per-machine calibrator | `endgame_trigger_study.py` (`373ab15`) |
| Encoder tempo primitives | `7wd-encoder-4` (`28f1413`) |

Nothing in this file is blocked on anything except the ordering in §3.

---

## 2. Decisions already taken — do not relitigate

**Search: PUCT for recorded moves, Gumbel for cheap ones.** Three reasons, in
order of weight. The advisor is fixed PUCT (`RustPuctSearch`, `puct_root=true`),
so training and gating on Gumbel means the only surface a human judges is the
only one off-distribution. Depth is the lever left for strength, and PUCT gives
its best move a larger share of the budget. And the hybrid is free of the
target-mixing hazard it appears to have: **cheap moves emit no policy target at
all** (`policy_excluded = !full`, and `dataset.is_fast_search_move` drops them
from the example set), so every recorded target is a PUCT visit distribution and
the buffer stays homogeneous.

**Cheap moves are the place to spend diversity.** They carry no targets, so
varying sims, width, search type and chance capping there costs nothing and adds
state-distribution variety, which is a known limiting factor.

**Solve at every triggered ply; no solver cache.** A cache ceilings at 40% (the
deepest solve is 60% of per-game cost), but the 40% buys the thing that matters:
per-ply masking makes both sides play the endgame provably optimally, so the
game's realised result becomes the proven value of the first solved position —
which de-noises the outcome label for *every* row of the game, not just endgame
rows. With only ~1 independent outcome label per game, that is plausibly worth
more than the endgame labels themselves.

**Value targets from chance-free proofs only.** A scalar expected utility is
`P(win) − P(loss)` and does not determine a three-class distribution when draws
exist. `exact_expectimax` rows keep the realised outcome. Recovering them needs
the solver to propagate a real (win, draw, loss) distribution under a stated tie
policy; the scalar cannot be post-processed into one.

**Keep the encoder feasibility flags.** `military_bound` /
`science_missing_obtainable` are thresholds on *sums over reachable sets*;
max-pooling computes *existentials*. Different operators, not duplicates.
Removing them would force the net to learn a sum over a variable-size masked set
under a capacity constraint we already suspect is binding. The bug risk is real
and is answered by the probe suite in §6, not by removal.

---

## 3. Work items, in dependency order

### A. The PUCT switch — do this first, everything composes against it

`selfplay_search_mode` and `eval_search_mode` already exist and already accept
`"puct"`; they default to `"gumbel"`. So the arm itself is a flag. What is not
a flag:

1. **`TARGET_VERSION` bump.** PUCT's target is visit counts (`_puct_root`);
   Gumbel's is completed-Q. This is exactly what the version exists for. Batch
   every other target change (§D) into the same bump.
2. **Expose Dirichlet.** Ported and alpha-tuned (**1.8**, not KD's 0.3 — the
   branching differs), currently inert because no CLI flag reaches it. It
   applies to the PUCT root only. Under Gumbel that was harmless; under PUCT it
   is the only root exploration there is.
3. **Forced playouts + policy target pruning.** Not optional under PUCT:
   Dirichlet noise pollutes a visit-count target directly, which is the problem
   KataGo invented PTP for. Under Gumbel there was nothing to subtract.
4. **Composition order** — load-bearing, and easy to get silently wrong:
   **prune forced visits (search artifact) → zero provably-losing moves (proof)
   → renormalise.** Renormalising before the mask normalises over the wrong
   support.
5. **Per-move mode switch.** `puct_root` is currently per-config; the hybrid
   needs it resolved per move (`puct_root && full`).
6. **Raise `full_sims`.** 64–128 is low for PUCT by AlphaZero standards. This is
   the sweep dimension that actually costs throughput.
7. **Update the stale comments** asserting "self-play must stay Gumbel"
   (`self_play.rs`, `SelfPlayConfig::puct_root`). They become wrong; leaving them
   misleads the next reader.

### B. Wire the solver in properly

* **`endgame_oversample`** (KD uses 2.0) — solved rows carry exact labels, so
  concentrate gradient there.
* **Cost-predicted trigger**, replacing the card cap. Model: `cards_left`,
  `unrevealed`, `log10(legal)`, and the interaction `cards_left × log10(legal)`
  — held-out R² **0.904**. Attempt iff predicted cost fits the budget, with
  margin for the model's 0.51-decade (~3.3×) residual. Worth ~1.5×: it matches
  `cards ≤ 11`'s masked-move value at **61%** of the cost, because a card cap
  must discard a whole card of depth at once while the predictor takes the cheap
  11s and drops the dear 9s.
* **Async solving.** A `SolvePending` slot stage plus a solver thread pool and a
  results channel, mirroring how a slot already parks on an NN batch and yields
  its thread. The mask must be applied *before* the move is chosen, so the
  **slot** waits while the **thread** does not. This is what lets the budget rise
  without stalling generation.
* **Statistics into the iteration log**: attempts, declines by reason
  (`unsolvable` / `budget`), nodes spent including on failures, solver seconds as
  a share of generation, solved-position count, and the
  `net_root_value` / `root_value` / `solver_value` triple per solved row — so the
  oracle diagnostic runs continuously instead of as a separate job.
* **`exact_fallback_positions` sidecar** — declined roots, kept out of the replay
  buffer. Now straightforward: `solver_attempted` and `solver_stop` identify them.

### C. The sweep

**Objective: maximise proven labels per hour, subject to games/hour unchanged.**
One objective and one invariant, rather than a trade-off to referee. The
all-core-clock effect (solver threads depress generation clocks — the plan
measured solver scaling at 2.89×/4, 3.77×/8, 4.37×/16 threads and attributed the
ceiling to clock and SMT, *not* bandwidth) then shows up as a measured drop in
games/hour that the sweep simply avoids.

Dimensions: solver cores vs generation cores; node budget; predictor threshold;
`full_sims` and `full_search_fraction` (they interact — more sims per full move
means fewer full moves per hour); `endgame_oversample`; Gumbel `top_k` on cheap
moves.

`endgame_trigger_study.py --calibrate` already sizes node budget and depth for a
given box from a stored cost profile plus a one-minute rate measurement, because
**node counts are machine-independent**. The sweep is the throughput half that
the calibrator cannot answer.

### D. Network and targets

* **Mean/max readout.** The readout is `encoded[:, 0]`, a single CLS-style
  GLOBAL token. Concatenate masked mean- and max-pools over real tokens and
  project `3d → d` so `Heads` is unchanged:

  ```python
  h = self.final_norm(encoded)
  mask = ~batch["pad_mask"]
  counts = mask.sum(1, keepdim=True).clamp(min=1)
  mean = (h * mask.unsqueeze(-1)).sum(1) / counts
  maxed = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
  readout = self.readout_proj(torch.cat([h[:, 0], mean, maxed], -1))
  ```

  Masking is load-bearing in both. The point is **max**: attention is an
  averaging operator, and "is there *any* token with property X" is an
  existential it approximates poorly. The encoder hand-codes two such
  existentials (`sci_win_feasible`, `mil_win_feasible`); this generalises the
  pattern instead of adding a third bespoke flag.

* **Auxiliary reply head.** A second policy head predicting the opponent's
  improved policy at the next decision. Target for row *t* is row *t+1*'s
  `policy_target` over *t+1*'s legal set, when *t+1* is full-search and its actor
  differs. No Rust: the buffer already holds it. Weight ~0.15.

  **It adds no information** — Q already integrates the opponent's reply, which
  is why "don't take X, it uncovers Y" is already implicit in the recorded
  target. What it adds is supervision density and explicit pressure on the trunk
  to encode opponent intent, i.e. a better **prior** — which is where the oracle
  probe located the error (raw net mean |err| 0.221 vs search's 0.191; search
  improves on the net in only 71% of proven positions).

  Land it **after** the PUCT switch: under PUCT the row-*t+1* target is a visit
  distribution, so building it first supervises against a definition about to be
  replaced.

### E. Measurement

* **Gumbel vs PUCT against proven values**, matched sims, on the solver-proven
  set — `|value − truth|` and top-1 agreement with the proven-optimal set. Cheap,
  needs no training run, and settles §A empirically rather than by argument.
* **Gate under `--eval-search-mode puct`** so the number promoted on is the
  number the advisor delivers.
* **Error-tail analysis** (§6).
* **Regenerate the equivalence corpus** once a net loads — closes
  `test_buffer_games_equivalent` and `test_encode_corpus_equivalent`, red only
  because they replay games the corrected engine no longer produces.

---

## 4. Measured facts — do not re-derive

**Solver cost.** ~0.03 s/game at `max_cards 8` (55k–103k nodes/game, median
475–1,089 nodes/position). Cheap enough that async is about *raising* the budget,
not about affording the current one.

**Reach at a 3s budget** (= 4.5M nodes at this machine's measured 1.51M nodes/s):
97% of 8-card positions solve, 91% at 9, 73% at 10, 56% at 11, **9% at 12**.
Doubling machine speed buys ~2% more solves at cap 10 — the cost curve in card
count is far steeper than any plausible hardware difference.

**Declines at a 2M budget:** 3% at 8 cards, 18% at 9, 33% at 10, 67% at 11, 97%
at 12. **Marginal cost per extra solve:** 0.71M nodes for 8→9, 1.79M for 9→10,
4.91M for 10→11, **66.71M for 11→12**.

**Cost predictors** (Spearman vs log₁₀ nodes, 358 positions): `cards_left`
+0.938, `unrevealed` +0.925, `moves_from_end` +0.793, `accessible` +0.496,
`legal` +0.478. Within a fixed card count `legal` still carries +0.47…+0.73.
Variability within a card count is extreme: p90/median 6.7×–35.8×, and a 7-card
position ranges 1,148 to 6,020,300 nodes.

**Expensive positions are the valuable ones.** log₁₀(nodes) vs optimal-move
fraction **−0.346**; vs |root_value| **−0.499**. Cheapest decile: |value| 1.00
and 94% of moves already optimal (the mask does nothing). Dearest deciles:
|value| ~0.87 with 51–61% optimal. Most of this runs *through depth* — within a
fixed card count the effect weakens to −0.13…−0.27.

**Ties:** 78–82% of legal moves proven optimal, reproducing the plan's 77–88% on
self-play rather than the corpus.

**Gumbel's allocation** at 128 sims — the eventual winner's share is nearly flat
in `top_k`: `m=12` → 2, 5, 12, 38 per round, winner gets **57**; `m=4` → 48;
`m=2` → 64. Sequential halving is near-optimal in allocation, so **narrowing the
candidate set does not concentrate budget on the best line.** Low `top_k` is a
*decision-quality* lever (it raises the round-zero floor from 2 sims per
candidate to 16, and 2 is a noisy basis for eliminating half the field), not a
depth lever. Depth comes from `sims`.

**Oracle probe, migrated net, 133 proven positions:** net |err| mean 0.221,
median 0.056, **p90 0.919**; search 0.191 / 0.021 / 0.662; sign agreement 92%
both; search moves toward truth 71% of the time. Read one-directionally — the
net is off-distribution, so a *good* score is trustworthy and a bad one is not,
while the net-vs-search *delta* is a within-subject comparison and largely
cancels the mismatch.

**Civilian VP available:** Age I 11 total, Age II 28, Age III 61, wonders 42.

---

## 5. Rejected — do not retry without new information

* **Civilian VP reachability bound.** Age I holds 11 VP in total, and the Age III
  ceiling is ~31 VP at 10 cards left against typical gaps of 5–15, so it does not
  bind until 2–4 cards remain — where the solver already answers exactly in a
  median of 10–46 nodes. Its whole informative region is where it is redundant,
  and it carries the largest edge-case surface available (Mausoleum reachability,
  Great Library, Mathematics' `VP_PER_PROGRESS` nonlinearity, top-K truncation,
  accessibility). Not a modelling weakness: **nothing is civilian-decided in
  Age I**, so there is no fact to encode.
* **`chance_fanout`** (Σ log pool size over face-down slots). Correlates +0.916
  alone but *adds nothing* over the base model and slightly hurts (held-out R²
  0.8843 vs 0.8901). The unseen pool is nearly drained by late Age III, so the
  5–20 fan-out spread it was designed for does not exist there.
* **Decidedness features as cost predictors**: `vp_gap` −0.002, `science_threat`
  +0.006, `science_max` −0.018, `military_to_win` +0.076, `pending_options`
  −0.031. Closeness *does* drive cost (proven value −0.499), but civilian VP is
  not a proxy for it — a position level on points can be decided by tempo,
  military or science.
* **`i_take_last_card` / `i_choose_next_start` as encoder features.** They are
  projections assuming nobody repeats a turn, so they are false exactly when the
  age-transition fight is live — the same shape as `sci_win_feasible` being wrong
  exactly in the Mausoleum case. Replaced by unconditional primitives
  (`cards_remaining_odd`, `military_tied`) with the conjunction left to the net.
* **`opp_can_extra_turn` as a scalar.** Every wonder token already carries
  `mine`, `built`, `affordable` and `grants_extra_turn` (Theology-aware), so it
  is a conjunction of present flags that attention aggregates.
* **Per-action conditional reply target.** Q already integrates the opponent's
  reply, so a 1202×1202-shaped output re-expresses the same tree.
* **KataGo global pooling as such.** It exists because convolutions are local.
  Attention is all-to-all and the readout is already a dedicated GLOBAL token.
  Only the mean/max *readout* transfers (§D).
* Earlier solver rejections stand: star2 probing, lexicographic margin
  refinement, transposition tables, allocation tricks — see
  `SOLVER_SELF_PLAY_PLAN.md` §9.

---

## 6. Verification the retrain unblocks

**Science-threat probe suite** — the gate `sevenwd_science_blindspot` says the
Mausoleum fix should have had: families for a symbol face-up, under a face-down
card, in the discard with an unbuilt Mausoleum, and via the Law token. Cheap now
and it protects the feature we chose to keep in §2.

**Error-tail analysis, pre-registered.** Sort proven positions by `|net − truth|`,
take the top decile, and compare bucket membership against the body:

* unbuilt Mausoleum present **and** discard non-empty (the fixed bug's exact
  trigger);
* science threat live (either side ≥5 symbols);
* military within 2 of an instant win;
* parity and `military_tied` both live (tempo-decided);
* `exact` vs `exact_expectimax` regime.

The Mausoleum bucket doubles as end-to-end verification of `ec08bc1`: if the tail
is no longer enriched there, the fix worked. If it still is, `_revivable_cards`
is incomplete. This must run **after** the retrain — the migrated net was trained
with the buggy encoder and served with the fixed one, so its errors conflate the
value head with a train/serve mismatch concentrated in exactly those positions.

Any bucket that survives should be re-checked the way `ec08bc1` was: compute what
the encoder says, compute the truth, feed corrected features to the same weights,
and see whether the read moves. That is what turns "the net is bad here" into
"this feature is wrong."

---

## 7. Standing risks

**Capacity.** `cloud6` ruled out cheap targets, weight decay, search exhaustion
and target noise, leaving optimization-or-capacity. §D adds a head and widens the
readout; §A raises sims. If capacity is binding, some of this hurts. Keep each
piece independently ablatable, and prefer measuring the stall before assuming
more features fix it.

**Feature selection by reasoning has a poor record here.** Three features argued
from mechanism did not survive: `chance_fanout`, the decidedness set, and the VP
bound — two refuted by measurement, one by argument. The one proposal that held
up (tempo) came from an engine rule plus an external observation, not a story.
Prefer the error tail (§6) as the selection method.

**Wall-clock budgets break reproducibility.** A solve bounded by seconds makes
generation irreproducible from `(seed, net)`. Bound by nodes; keep `max_secs` as
a non-binding safety net. The corpus gate has the same flaw — `BUILD_MAX_SECS` is
a deadline, so gate *coverage* varies with machine load (83, 85 and 86 of 86
positions across three runs with no code change). Worth pinning to the node
budget alone.

**Duplicated constants.** The padded feature width was a literal in four places
and adding two features left two behind, surfacing as a matmul shape error naming
nothing. All four now derive from the schema. The class of bug is worth watching
for elsewhere.

---

## 8. Suggested order

1. §E first bullet — Gumbel vs PUCT against proven values. Cheap, and it either
   confirms or overturns §A before anything is built on it.
2. §A — the PUCT switch, with Dirichlet, forced playouts + PTP, the per-move
   mode split, and one `TARGET_VERSION` bump covering §D as well.
3. §B — solver wiring: oversample, cost-predicted trigger, async, statistics.
4. §D — mean/max readout, then the reply head.
5. §C — the sweep, on the finished loop.
6. Retrain. Then §6, and regenerate the equivalence corpus.

Steps 2–4 all bump the same target version, so they should ship together or the
buffer fragments across definitions.
