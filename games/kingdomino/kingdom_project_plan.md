# Project Plan: Four-Head Network + Open-Loop MCTS

**Target:** World-class 2-player Mighty Duel Kingdomino agent via AlphaZero self-play, with a Board Game Arena advisor as the deployment surface.

**Scope of this milestone:** Two coupled architectural changes implemented together before the cloud training run, plus an encoder audit and a post-training strategic probe suite.

---

## Design Decisions (Settled)

These were decided through analysis and are fixed inputs to implementation.

### Network: four heads, margin derived

Replace the single margin value head with **own_score** and **opponent_score** heads. Final head set:

1. **policy** — visit-count targets (unchanged)
2. **own_score** — own final score, regression
3. **opponent_score** — opponent final score, regression
4. **win** — win probability, BCE

**Rationale.** Kingdomino is genuinely adversarial in the 2-player Mighty Duel format. The own/opponent decomposition gives two strategic signals that a single margin head entangles:

- **Blocking** (taking a domino that is suboptimal for own score but reduces opponent score) is supervised directly through `opponent_score`.
- **Tempo / first-pick** (taking a worse domino now to claim a better one later) is primarily an own-trajectory decision and is captured cleanly by `own_score` — the head predicting exactly the quantity that the tempo tradeoff optimizes.

Margin is recovered as a derived quantity for MCTS leaf evaluation; it is not a separate head.

### Search: open-loop MCTS

Resample deck order **per simulation** rather than per search. Tree nodes keyed on action sequences, not concrete states.

**Rationale.** det=1 (one fixed deck order per search) trains the network on a game that doesn't exist — one where the future draw is revealed before deciding. This systematically under-values moves that hedge across futures. Open-loop averages over many sampled futures, which is the only mechanism that makes future-conditional value (flexibility, tempo) visible. This is the central change for reaching world-class play.

### Why these two changes ship together — and how they are isolated for debugging

Open-loop changes the training distribution fundamentally. A network trained on closed-loop targets and fine-tuned on open-loop would suffer a confused transition. Train from scratch under the correct distribution from iteration zero of the cloud run.

A review raised the concern that shipping two large changes together makes failures hard to attribute. The right response is **not** a closed-loop *training* run — that validates whether the heads *learn*, which is uninformative because they will train under a different (open-loop) distribution anyway. The valuable isolation is **deterministic value-correctness**: prove that every value (targets, normalization, margin recovery, centered win value, per-head frame conversion) is computed bit-exactly on hand-checkable inputs *before* open-loop adds stochastic search on top.

The discipline:

```text
Phase 1 proves the VALUE LAYER is correct, deterministically, with no training and no search.
Phase 3 adds open-loop SEARCH on top of a value layer already trusted.
```

This means open-loop debugging never has to ask "is the leaf value wrong or is the search wrong?" — the leaf value is already proven. This is the isolation the review wanted, achieved through assertions on known inputs rather than through a misleading closed-loop training run.

### Strategic concepts: capture via value signal + search, verify by probe

Three high-level strategic concepts were analyzed. Decision: **do not pre-engineer input features or dedicated heads for them.** Each is well-matched to the own/opponent decomposition plus open-loop search, and each will be verified by a post-training probe rather than assumed.

- **First-pick / tempo value** — captured by `own_score` (own-trajectory tradeoff). Action required: encoder audit to confirm turn order and current claims are explicit inputs.
- **Flexibility / strategic non-commitment** — this is *not* legal-move count. It is keeping multiple terrain strategies alive in the early game. Its value literally *is* higher expected `own_score` over the draft distribution, which `own_score` + open-loop is built to capture. No explicit feature (the frontier-openness proxy measures the wrong thing and was dropped). Verify by the domino-20-style probe.
- **Opponent blocking** — captured by `opponent_score`. Verify by a blocking probe.

If a probe fails after training, that failure becomes the concrete hypothesis justifying a targeted feature or head — built then, not speculatively now.

---

## Pre-Implementation Decisions (Record in Checkpoint Config)

Resolve and record before writing code. No silent defaults.

- **Win target with full tiebreaker cascade — implemented as an ENGINE fix, single source of truth:**

  The engine currently has **no winner-determination function** — `game.scores()` returns point totals and `board.score()` computes territory score, but nothing decides who won, and score ties are resolved arbitrarily. The official Kingdomino tiebreaker cascade must be implemented authoritatively in the engine, and `compute_target_win` must **call that same function** rather than reimplementing the cascade (two implementations would drift — exactly the silent-divergence class your equivalence testing exists to catch).

  Cascade (official rules):
  1. Highest total score wins.
  2. If score-tied: **most extended property** — largest single connected same-terrain territory by tile count (crowns irrelevant for this step).
  3. If still tied: **most total crowns** on the board.
  4. If still tied: shared victory → **0.5**.

  **Engine changes required.** The current `board.score()` accumulates `area * crowns` into one total (board.py ~line 227) and discards the per-territory `area` and the crown count — exactly the data the cascade needs. Surface two additional quantities from the same connected-components pass:
  - `largest_territory_size` (max `area` over all territories, crowns ignored)
  - `total_crowns` (sum of all crowns on the board)

  Add an authoritative `determine_winner(state) -> {0, 1, draw}` (or per-player win value) on the engine that applies the cascade using score, then `largest_territory_size`, then `total_crowns`. `compute_target_win` and the self-play winner logic both call it.

  **Correctness gate.** This is a genuine engine bug fix (score ties were previously mis-resolved), so it gets its own equivalence/unit coverage: construct boards hitting each cascade level and assert `determine_winner` returns the rule-correct result. Only an all-levels tie returns draw/0.5.
- **Score-head output convention (explicit):** the score heads output **normalized score**, not raw score. Targets are `score / SCORE_SCALE`. There is no second division in MCTS — the head output is already normalized. (`SCORE_SCALE ≈ 100`.)
- **Centered win value:** the win head outputs `win_prob ∈ (0,1)`, but MCTS uses `win_value = 2*win_prob - 1 ∈ (-1,1)`. Centering makes the win term antisymmetric so it combines cleanly with margin AND frame-converts by **negation** (see frame note below — centering changes win from "complement" to "negate").
- **Margin recovery gain:** `MARGIN_GAIN` in `tanh((own_norm_pred - opp_norm_pred) * MARGIN_GAIN)`. With `SCORE_SCALE=100`, a +25 margin is `0.25` pre-gain; `MARGIN_GAIN=4` maps +25→`tanh(1)≈0.76`, `MARGIN_GAIN=2` maps +50→`tanh(1)`. **Start `MARGIN_GAIN = 2.0–3.0`.** Calibrate so derived margin matches the old `compute_target_z` *target* scale (see target-equivalence test).
- **Leaf combination weight:** `alpha = 0.8` (margin-dominant; keep win auxiliary early), for `leaf_value = alpha * margin_value + (1 - alpha) * win_value`.
- **Loss weights (start low until scales are known):** `lambda_score = 0.5` (applied equally to own and opponent score losses), `lambda_w = 0.25`. Policy is the core AlphaZero signal; do not let auxiliary losses dominate. Tune up only after observing component magnitudes.
- **checkpoint_version:** bump to detect the architecture migration.

**Loss-balance gate:** log unweighted and weighted components separately (`policy_loss`, `own_loss`, `opp_loss`, `win_loss`, `weighted_total`). No single non-policy weighted loss should dominate the total by more than ~3–5× for sustained steps; if it does, adjust weights before proceeding.

---

## Phase 0: Engine & Encoder Prerequisites

Cheap, high-value, do first. Two independent items, both prerequisites for later phases.

### 0.1 Engine Winner Determination (tiebreaker cascade)

The engine has no authoritative winner function and currently mis-resolves score ties (see the tiebreaker-cascade decision above). This is a standalone engine bug fix and a prerequisite for `compute_target_win`.

**Task.**
- Extend `board.score()` (or add a sibling) to surface `largest_territory_size` and `total_crowns` from the existing connected-components pass — both are currently discarded.
- Add `determine_winner(state)` on the engine applying the cascade: score → largest territory → total crowns → draw.
- Route both `compute_target_win` and self-play winner logic through `determine_winner` (single source of truth).

**Test.** Construct boards hitting each cascade level (score-decided, territory-decided, crowns-decided, full-draw) and assert rule-correct results.

**Gate.** `determine_winner` correct on all four cascade levels; self-play and the win target both call it.

### 0.2 Encoder Audit (Turn Order)

**Task.** Confirm `encode_state` represents, as explicit input channels/features:
- Which domino each player has currently claimed
- The implied turn order for the next round

If turn order is implicit or must be reconstructed by the network, make it an explicit feature. A clean turn-order input lets the network learn tempo/first-pick value efficiently rather than inferring it through several layers.

**Test.** Encoder test asserting turn-order features are present and correct for reference mid-game states with known claim/turn-order configurations.

**Gate.** Turn order and current claims are explicit, tested input features.

---

## Phase 1: Network Changes

### 1.1 Architecture

```
KingdominoNet
├── trunk (residual blocks) — unchanged
├── policy_head             — unchanged
├── own_score_head          — new, regression (normalized by SCORE_SCALE)
├── opponent_score_head     — new, regression (normalized by SCORE_SCALE)
└── win_head                — new, sigmoid ∈ (0,1)
```

The previous `value_head` (margin) is removed; margin becomes derived. Each new head mirrors the old value-head structure (pool → linear → activation → linear). Final layers initialized with `std=0.01` to avoid saturation at init.

**Checkpoint compatibility.** The architecture change breaks old checkpoint loading. The loader should detect `checkpoint_version` and either remap or refuse cleanly. Old single-margin checkpoints remain usable for evaluation/comparison only (not for resuming training under the new architecture). Document this.

### 1.2 Encoder Targets

Add to `encoder.py`:
- `compute_target_own_score(state, player)` — own final score (raw, normalized at train time)
- `compute_target_opponent_score(state, player)` — opponent final score
- `compute_target_win(state, player)` — 1.0 / 0.5 / 0.0 via the full tiebreaker cascade (score → largest territory → total crowns → 0.5)

All guarded to terminal states, 2-player Mighty Duel only, consistent with existing target functions.

### 1.3 Training

Self-play stores `own_score`, `opponent_score`, and `win` targets per position, broadcast from game end, from the storing player's perspective. Score targets are normalized at train time: `own_target_norm = own_score / SCORE_SCALE` (Option 2 — heads output normalized score).

```python
policy_loss = cross_entropy(policy_logits, visit_targets, mask=legal_mask)
own_loss    = mse(own_score_norm_pred, own_target_norm)
opp_loss    = mse(opp_score_norm_pred, opp_target_norm)
win_loss    = bce(win_prob, win_target)
loss = policy_loss + lambda_score * (own_loss + opp_loss) + lambda_w * win_loss
```

Log all components (weighted and unweighted) separately for the loss-balance gate.

### 1.4 Inference / MCTS Leaf Evaluation

```python
own_norm_pred = own_score_head_output    # already normalized (Option 2)
opp_norm_pred = opp_score_head_output    # already normalized

margin_value = tanh((own_norm_pred - opp_norm_pred) * MARGIN_GAIN)   # bounded, antisymmetric
win_value    = 2.0 * win_prob - 1.0                                  # centered, antisymmetric
leaf_value   = alpha * margin_value + (1 - alpha) * win_value
```

Then apply the player-frame conversion to `leaf_value`. Both terms are now in the same `[-1,1]` frame (−1 bad, 0 neutral, +1 good), so the combination is clean — no centered/uncentered mixing.

**Frame conversion — handle each head's symmetry correctly:**
- **margin** (derived): antisymmetric — **negate** when flipping perspective.
- **win_value** (centered `2p-1`): antisymmetric — **negate** when flipping perspective. *(Note: raw `win_prob` would complement (`1-p`); once centered it negates. The plan previously said "complement" — that was correct only for the uncentered probability. Use negate for `win_value`.)*
- **own/opponent score**: these **swap** when flipping perspective (player 1's "own" is player 0's "opponent"). Handle explicitly.

Because `leaf_value` is built entirely from antisymmetric terms, the combined scalar negates under perspective flip. Combine into the scalar first, then convert frame — document the order in `_evaluate`.

### 1.5 Testing — Deterministic Value-Correctness (the isolation gate)

This phase proves the **value layer** is correct with no training and no search. Every check is an assertion on a known input. This is the isolation the review asked for, done right.

`tests/test_network.py`:
- Forward returns four head outputs with correct shapes/ranges (scores in normalized region, win ∈ (0,1))
- Checkpoint round-trip identical
- `checkpoint_version` migration path behaves (loads old for eval, refuses/remaps for training)
- Score heads near expected init range; win head outputs in (0.4, 0.6) at init (not saturated)
- Gradient isolation: zeroing any one loss leaves trunk gradients from the others non-zero

**Frame-conversion tests, per head (deterministic, hand-checked):**
- Derived **margin negates** under perspective flip
- **win_value = 2*win_prob - 1 negates** under perspective flip (NOT complement — centering changed this)
- **own/opponent scores swap** under perspective flip
- The combined `leaf_value` negates under perspective flip (follows from all terms being antisymmetric) — assert on a position with known head outputs

**Leaf-value computation test (deterministic):**
- Feed known `own_norm_pred`, `opp_norm_pred`, `win_prob`; assert `leaf_value` equals the hand-computed value of `alpha*tanh((own-opp)*MARGIN_GAIN) + (1-alpha)*(2*win_prob-1)`. This pins the exact formula before search is ever involved.

`tests/test_encoder.py`:
- own/opponent/win targets correct for: clear win (30,20), clear loss (20,30), score-tie broken by **larger territory**, score-and-territory-tie broken by **more crowns**, and a true all-levels tie → 0.5
- Non-terminal raises
- win sign agrees with `(own - opponent)` sign **when scores differ**; when scores are equal, win is set by the cascade (territory → crowns → 0.5), so the test checks the cascade logic, not the score sign

**Target/frame equivalence (replaces the ill-defined "predictions match old head" test):**
- **Target equivalence:** the *derived target* margin from final own/opp scores (`tanh((own-opp)/SCORE_SCALE * MARGIN_GAIN)`) reproduces the old `compute_target_z` formula's intent on reference final scores. This calibrates `MARGIN_GAIN`. (This is about *targets*, not model predictions — do not compare new model outputs to the old model.)
- **Frame equivalence:** swapping player perspective swaps own/opp targets and negates the derived margin target.

**Gate.** All deterministic value-correctness checks pass: head ranges, per-head frame conversions (with win_value negating), the pinned leaf-value formula, target/frame equivalence, tie handling. **No open-loop work begins until this gate is green** — the value layer must be trusted before search is layered on.

### 1.6 Documentation

- `network.py`: four-head docstring; per-head output ranges and losses; frame-conversion asymmetries (margin negate, **win_value negate after centering**, scores swap)
- `encoder.py`: three target functions; tie rule; turn-order features
- `mcts_az.py` `_evaluate`: centered win value (`2p-1`); margin recovery formula; combine-then-convert order
- Checkpoint config: all recorded hyperparameters

---

## Phase 2: Augmentation Changes

### 2.1 Signature

Add `own_target`, `opponent_target`, `win_target` to `augment()` / `augment_all()`. All three are scalars and pass through **byte-identical** across all 8 transforms (rotation/reflection is a coordinate change; final scores and outcome are invariant). Update `TrainingTuple` and all callers.

> Note: if Phase 0 added **spatial** turn-order channels, those DO transform under D4 and must be tested for correct permutation — unlike the scalar targets. Distinguish the two clearly.

### 2.2 Correctness Contracts

Extend the `augmentation.py` invariant list:
```
7. own_target, opponent_target, win_target are scalars,
   byte-identical across all 8 transforms.
```
Plus, if applicable, a spatial contract for any new turn-order channels.

### 2.3 Testing

- Scalar targets identical across all 8 augmented copies
- Existing spatial contracts still pass (regression)
- If new spatial channels exist: they permute correctly under each of the 8 transforms (explicit per-transform assertions)

**Gate.** All correctness contracts pass, including new scalar targets and any new spatial channels.

---

## Phase 3: Open-Loop MCTS

### 3.0 BLOCKING Design Gate — Pick-Encoding Semantics

**This must resolve before any open-loop code is written.** It is the highest-risk item in the milestone and is in the same family as prior codec bugs (PyO3 `Vec<u8>` boundary, symmetric-domino joint-index ambiguity) that were caught only by equivalence testing.

The earlier claim that "legal actions never diverge" is **wrong at depth**. It holds only at the root and within the current public row. At deeper open-loop nodes — after replaying an action sequence into future rounds — the future draft row is drawn from the bag and **differs across determinizations**. So the available picks at a deep node legitimately differ per simulation.

Whether this is benign or catastrophic depends entirely on how the action codec encodes picks:

```text
SLOT-RELATIVE  (pick_idx = "slot k of whatever row is current here"):
    Replaying an action sequence is well-defined across determinizations.
    Slot k exists in every determinization (it points to a different concrete
    domino, which is fine — open-loop averages over exactly that).
    Open-loop works cleanly.

DOMINO-ID-RELATIVE  (pick encodes a concrete domino_id):
    Replaying "pick domino 37" fails in determinizations where 37 isn't in
    that future row. Open-loop breaks.
```

**Required action:** audit `action_codec.py` and the engine action representation to determine pick-encoding semantics. Then:
- If slot-relative: document it explicitly as the property that makes open-loop valid. Confirm the engine action does not secretly carry a concrete `domino_id` alongside the slot.
- If domino-id-relative: either (a) change pick encoding to slot-relative, or (b) implement per-determinization availability — select only among children legal in the current concrete state, merge by action key, maintain availability counts.

**Correctness note even in the slot-relative case:** deep-node slot indices point to different concrete dominoes across determinizations, so the *value* of a deep action varies across sims — this is correct and is what open-loop averages. The training target is only extracted at the **root**, where slots map to known public dominoes, so the target is well-defined. Invariant 3.1 (policy-target public-state invariance) tests exactly this, but only because the root is public; make explicit in docs that deep-node slot ambiguity is acceptable precisely because deep nodes produce no training targets.

**Gate.** Pick-encoding semantics audited and documented; if domino-id-relative, remediation chosen and specified before implementation.

### 3.1 Foundational Invariant Tests (write BEFORE implementing)

`tests/test_encoder.py` / `tests/test_open_loop_mcts.py`:

- **Encode invariance:** `encode_state(s)` byte-identical to `encode_state(redeterminize(s))` across 100 determinizations (board + flat).
- **Bag invariance:** `flat[bag]` byte-identical across 100 determinizations.
- **Policy-target public-state invariance:** `visit_counts_to_policy` identical whether computed from the public state or any determinization (confirms pick axis / current_row is public AT THE ROOT).

**Gate.** All three pass before any open-loop implementation.

### 3.2 Node Structure

`Node` stores `action`, `depth`, `prior`, `visit_count`, `value_sum`, `children` (keyed by action), `is_expanded`. **No stored state** — concrete state is reconstructed per simulation by replaying actions from root on a fresh determinization. Root receives the public state at search time.

### 3.3 Simulation Loop

```
search(public_state):
    root = Node(...)
    det = redeterminize(public_state, rng)
    expand root from det; seed root stats
    for _ in range(n_sims):
        det = redeterminize(public_state, rng)   # fresh per simulation
        simulate(root, det)
```

Within a simulation, step `det` forward by applying selected actions as you descend; the concrete state at any node is the replay of that node's path on this simulation's `det`.

### 3.4 Legal Action Availability Under Future-Row Uncertainty

**Corrected from an earlier overstatement.** Legal actions are identical across determinizations **only at the root and within the current public row.** After hidden future rows are revealed (depth ≥ next round), the sampled future row differs across determinizations, so the available pick actions at deep nodes diverge.

The resolution is set by the Phase 3.0 gate:
- **If pick encoding is slot-relative:** deep-node action *indices* (slot k) are valid in every determinization; they point to different concrete dominoes, which open-loop correctly averages over. No special handling needed beyond confirming slot-relative semantics.
- **If not, or to be safe:** at each node, select only among children legal in the current concrete state; merge children by action key; maintain per-action availability counts so PUCT normalizes correctly when an action is absent in some determinizations.

Document the chosen approach explicitly. This is the part of open-loop that is genuinely trickier than a single public-information game, and the earlier "never diverges" framing was wrong.

### 3.5 Dirichlet Noise / Virtual Loss

Both unchanged in semantics (root-only noise on action-keyed children; path-based virtual loss). Add a noise test confirming root child priors sum to 1.0 after noise.

### 3.6 Testing

**Hard gates (must pass before integration):**
- **Determinism:** fixed seed → identical visit counts
- **Bag-marginal value convergence (primary gate):** small remaining bag (e.g. 6 dominoes, 720 perms), **deterministic mock evaluator, Dirichlet disabled, fixed high sim count.** Assert (a) root *value* converges to the brute-force average over all permutations, and (b) the top action agrees. Test *value* convergence, not visit TV — PUCT visit distributions carry their own noise from priors and tie-breaking even when values agree.
- **Legal-action safety:** 1000 sims, zero steps into an action illegal in that sim's concrete state
- **Value range:** all backed-up Q ∈ [-1, 1]
- **Degenerate bag:** single permutation → identical to closed-loop with same seed
- **Win-frame through backup:** terminal win backs up correctly in player-0 frame regardless of terminal actor

**Staged diagnostics → promote to gates once implementation is stable:**
- **Real-network visit TV:** with the trained network, root visit distribution vs brute-force average. Diagnostic first (TV threshold meaningful only in the controlled mock setting); promote to a soft gate (e.g. TV < 0.1) once stable.
- **Symmetry consistency:** search the same position in all 8 D4 orientations (same seed/sims); inverse-transform each; check visit distributions consistent (pairwise TV below the `policy_compare.py` variance floor). Open-loop redeterminization adds variance that may make this noisy at first, so run it as a **diagnostic** during initial implementation and **promote to a required gate before the cloud run** — D4 correctness of open-loop must be proven before spending money, just not before first implementation.

**Gate (before integration).** Determinism, mock-evaluator value convergence, legal-action safety, value range, degenerate-bag, win-frame backup all pass. Symmetry consistency proven as a gate before the cloud run.

### 3.7 Rust Port (Phase 3R — later)

Python open-loop is the correctness reference. Port `BatchedMCTS` to open-loop only after Python passes all gates. The Rust port must pass the same bag-marginal convergence and symmetry consistency tests. Start during early cloud training, not before.

### 3.8 Documentation

`mcts_az.py` module docstring: open-loop formulation; why `redeterminize` moved to per-simulation; why legal-action divergence does not occur in Kingdomino. `redeterminize()` docstring: the encode-invariance foundational property. `Node` docstring: stateless design and per-simulation replay.

---

## Phase 4: Integration & Self-Play Validation

### 4.1 Smoke Test (10 games, random init)
- No crashes; all losses finite and non-zero
- Score heads in expected init range; win head ≈ 0.5
- Records contain own/opponent/win targets per position
- Augmented batches carry the new scalar targets

### 4.2 Short Training Run (5 iterations)
- All losses decreasing
- **Win head calibration (corrected):** the trivial constant baseline (predict the batch base rate) has Brier ≈ 0.25 for binary labels — so "< 0.30" is *worse* than trivial and is the wrong gate. Instead compute the actual baseline on the validation batch (`baseline_brier = mean((base_rate - target)**2)`) and require the win head's Brier to **improve over that baseline** by iter 5. Also require: predictions not saturated, calibration not inverted (a U-shape implies a frame-conversion bug — likely the win_value centering/negation).
- Policy entropy decreasing
- Score heads tracking actual scores (own/opp MSE decreasing)
- Loss-balance gate holding (no non-policy weighted loss dominating by >3–5×)

**Gate.** All losses decreasing by iter 5; win head beats its computed base-rate Brier baseline; win unsaturated; calibration not inverted; loss balance sane.

### 4.3 Open-Loop Distribution Sanity (post-5-iter)
Re-run the bag-marginal convergence test with the partially trained network. Convergence is a property of the search, not the network — it must still pass. Failure here (when it passed at random init) indicates the network is somehow influencing determinization sampling — a bug.

### 4.4 Equivalence vs Prior Checkpoint
200 games: new open-loop agent (10 iters) vs best prior closed-loop checkpoint. Need legal, non-degenerate play — not a win. Any crash, illegal move, or degenerate pattern (always-discard, always-first-tile, win head stuck at 0/1) is a failure.

**Gate.** 200 games complete cleanly; no degenerate behavior.

### 4.5 Throughput Profile (feeds network sizing)
Profile open-loop Python self-play: games/s, mean inference batch, GPU utilization, and the CPU cost of per-simulation `redeterminize` + state replay. Open-loop may be more CPU-bound than closed-loop because of per-simulation determinization. Output: (a) whether Rust open-loop is required before the cloud run, and (b) how much network headroom actually exists — this sets the Phase 5 network size. Do not assume prior closed-loop throughput carries over.

### 4.6 Documentation
Update self-play modules (new targets), checkpoint saving (config fields), training loop (loss-weight rationale).

---

## Phase 5: Cloud Run Preparation

### 5.1 Config (confirm before launch)
```python
config = {
    'lambda_score': 0.5, 'lambda_w': 0.25,   # start low; tune up after observing scales
    'alpha': 0.8, 'SCORE_SCALE': 100.0, 'MARGIN_GAIN': 2.0,  # MARGIN_GAIN calibrated in 1.5
    'sims': 1600, 'open_loop': True,
    'channels': 64, 'blocks': 6,             # FLOOR = proven size; see sizing note
    'checkpoint_version': 2,
}
```
All fields in every checkpoint. No silent defaults.

**Network sizing (corrected).** Do **not** default to 32×4 — that would shrink the model just as more prediction tasks and subtler strategy are added. Use the proven size as a **floor** (64×6). Whether to go larger (96×8, 128×10) is an **output of the Phase 4 throughput profile**, not an assumption — open-loop's per-simulation redeterminization may be meaningfully more CPU-expensive than closed-loop, so do not assume GPU headroom carries over from prior runs. Profile open-loop first, then size: proven size as floor, scale up only if profiling shows real headroom.

### 5.2 Evaluation Schedule
Per checkpoint, evaluate vs: best prior closed-loop checkpoint, random baseline, self round-robin (plateau detection). Record: win rate, mean score margin, win-head Brier, root policy entropy.

### 5.3 Early Stopping (define before launch)
- Investigate if win Brier not improving after 20 iters (check `lambda_w`, frame conversion)
- Investigate if policy entropy not decreasing after 20 iters (check open-loop correctness, sims)
- Plateau: win rate vs prior best within ±5% for 10 consecutive iters
- Continue if all losses still decreasing at iter 20

---

## Phase 6: Post-Training Strategic Probes

Run after the cloud run converges. These verify the high-level strategy emerged rather than assuming it. Each is a constructed position with a known "expert" answer.

1. **Tempo / first-pick probe.** Positions where claiming a low domino (worse current tile) for better next-round pick is correct. Check the policy favors it. Tests whether `own_score` + open-loop learned tempo.

2. **Blocking probe.** Positions where a claim suboptimal for own score but damaging to opponent score is correct. Check the policy favors it. Tests whether `opponent_score` drives adversarial play.

3. **Flexibility / non-commitment probe (domino-20 case).** Positions where a flexibility-preserving placement and a flexibility-destroying placement have equal immediate value but the flexible one preserves future high-value options. Check the policy prefers the flexible one. Tests whether `own_score` + open-loop captured strategic non-commitment.

**Interpretation.** If a probe passes, the concept emerged for free — cost nothing. If a probe fails, that concrete failure is the hypothesis justifying a targeted intervention (a defined feature or head), built then with a real failure to study — not speculatively now.

---

## Sequencing

```
Phase 0  0.1 Engine winner determination (tiebreaker cascade) ── gate
         0.2 Encoder audit (turn order) ───────────────────────── gate
Phase 1  Network (four heads)
         VALUE-CORRECTNESS GATE (deterministic, no training/search):
         frame conversions, pinned leaf formula, target/frame equivalence
         → value layer trusted before any search work
Phase 2  Augmentation ───────────────────────────── gate
Phase 3  Open-loop MCTS
         3.0 PICK-ENCODING DESIGN GATE (slot- vs domino-id-relative) ── BLOCKING, before impl
         3.1 foundational invariants ── gate (before impl)
         3.2–3.5 implementation
         3.6 correctness (mock-evaluator value convergence) ─── gate
             symmetry: diagnostic now → gate before cloud run
Phase 4  Integration
         4.1 smoke ─ 4.2 short train ─ 4.3 sanity ─ 4.4 equivalence ─ 4.5 throughput profile ── gates
Phase 5  Cloud run (network size from 4.5 profile; floor 64×6)
Phase 6  Strategic probes
Phase 3R Rust open-loop port (parallel with early cloud training)
```

---

## Definition of Done

**Engine & Encoder (Phase 0)**
- [ ] `determine_winner` implements the full cascade (score → territory → crowns → draw), correct on all four levels
- [ ] `board.score()` surfaces `largest_territory_size` and `total_crowns`
- [ ] Self-play winner logic and `compute_target_win` both route through `determine_winner` (single source of truth)
- [ ] Turn order + current claims are explicit, tested input features

**Network (value-correctness gate)**
- [ ] Four heads pass unit tests; checkpoint_version migration behaves
- [ ] Win head unsaturated at init
- [ ] Frame conversions correct per head (margin negate, **win_value negate after centering**, scores swap)
- [ ] Pinned leaf-value formula test passes (centered win value)
- [ ] Tiebreaker cascade correct (score → territory → crowns → 0.5); win sign agrees with score difference when scores differ
- [ ] **Target** equivalence (derived margin target = old `compute_target_z`) and **frame** equivalence pass — NOT new-vs-old model prediction matching
- [ ] All hyperparameters recorded in config

**Augmentation**
- [ ] New scalar targets byte-identical across 8 transforms
- [ ] Any new spatial turn-order channels permute correctly under D4
- [ ] All correctness contracts pass

**Open-Loop MCTS**
- [ ] Pick-encoding semantics audited (slot- vs domino-id-relative); remediation specified if needed
- [ ] Foundational invariants pass (encode/bag/policy-target invariance)
- [ ] Mock-evaluator bag-marginal **value** convergence + top-action agreement
- [ ] Legal-action safety: 0 violations / 1000 sims
- [ ] Degenerate-bag matches closed-loop
- [ ] Win frame correct through backup
- [ ] Symmetry consistency proven as a gate before cloud run

**Integration**
- [ ] Smoke test clean (10 games)
- [ ] All losses decreasing by iter 5; win head beats computed base-rate Brier; calibration not inverted; loss balance sane
- [ ] Open-loop distribution sanity passes with trained net
- [ ] 200-game equivalence clean, no degenerate behavior
- [ ] Throughput profiled; network size decided from profile

**Cloud prep**
- [ ] Config confirmed and recorded (network floor 64×6)
- [ ] Evaluation schedule + early-stopping criteria defined

**Probes**
- [ ] Tempo, blocking, flexibility probes constructed and run post-training

**Documentation**
- [ ] network.py, encoder.py, mcts_az.py, augmentation.py, self-play modules, checkpoint config all updated