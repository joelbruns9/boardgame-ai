# 7 Wonders Duel — AlphaZero Training Loop Project Plan

**Goal: the best 7 Wonders Duel player in the world.** Not a ZeusAI reproduction — a
stronger agent built with the post-2020 training playbook (Gumbel search, KataGo-style
training science, reanalyze) plus an exact endgame layer ZeusAI does not have.

Status of prerequisites: engine milestones 1–6 complete (`README.md`) — rules engine,
data tables, legal actions, outcomes, rules-oracle tests, seeded/greedy bots, match
runner. This plan is milestone 7 onward.

---

## 1. Reference points

### ZeusAI (arXiv 2406.00741) — the benchmark recipe
- Vanilla 2017 AlphaZero + transformer: 12 layers × 12 heads × 768 dim, ~92M params.
- PUCT with up to 1k sims (training) / 5k sims (eval). No Gumbel, no aux targets,
  no reanalyze, no exact search.
- Closed-loop stochastic MCTS: afterstate (chance) nodes capped at ≤11 children in
  training, relaxed at eval. 11 ≈ full enumeration of a single reveal (see §4).
- Science victory discovered organically after ~100k games (no seeding — a ceiling
  we should beat by a wide margin).
- Converged victory mix: 61.7% civilian / 21.4% science / 16.9% military
  (sanity/calibration target, not ground truth).
- Wonder tier list (converged pick preference): extra-turn five on top (Temple of
  Artemis, Piraeus, Hanging Gardens, Appian Way, Sphinx) > Statue of Zeus, Great
  Library > Mausoleum / Circus Maximus / Colossus > Great Lighthouse > Pyramids last.
- Beat 2 of 3 top BGA players in bo5 (thin margin ~7–5 in games).

### Structural facts about 7WD this plan exploits
- **Hidden information is symmetric** — no private hands. Face-down cards, the 3
  removed cards per age, and unused guilds are unknown to *both* players. The game is
  a perfect-information game with chance nodes; no belief states or ISMCTS needed.
- **Single reveals are exactly enumerable.** By exchangeability, a face-down slot's
  identity is uniform over the current unseen pool of its back type
  (face-down-elsewhere vs removed doesn't matter). Pool ≤ ~8 face-down + 3 removed
  = 11, shrinking through the age. Age III backs distinguish guild vs age cards
  (public info). Multi-card uncovers = two sequential chance nodes, never a cap hack.
- **No duplicate cards** — card identity is unambiguous, enabling an
  identity-indexed action space.
- The discard pile is public and action-relevant (Mausoleum).

---

## 2. Locked design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Network | Encoder-only set transformer over entity tokens | Card-interaction structure (chains, guilds, science pairs); ZeusAI-validated; NOT "LLM-style" — no autoregression/tokenizer |
| Starting size | ~4 layers × 4 heads × 128–192 dim (~1–3M params) | Grow mid-run by retraining bigger net on existing buffer |
| Policy indexing | Card-identity (73 ids × verbs), not slot-indexed | Transfers across layouts/ages; pointer head = later refinement |
| Root selection | Gumbel top-k + sequential halving, n = 32–64 sims | Provable policy improvement at low sims; the single biggest compute+quality lever vs ZeusAI's 1k-sim PUCT |
| Wonder draft | In-policy from day 1, ZeusAI tier-list prior blended into draft-node priors, annealed to 0 | 8 high-impact decisions/game; prior is context-free so it must decay |
| Chance handling | **Dual-mode interface, decided by experiment** (§4): closed-loop exact chance nodes vs open-loop per-descent determinization | User-validated A/B; Rust implements only the winner |
| Exploration seeding | ScienceRushBot + MilitaryRushBot buffer seeding (annealed) + early opponent mixing | Deletes ZeusAI's 100k-game discovery tax; also teaches the denial side |
| Value/aux heads | Win/draw/loss + aux: joint winner × victory type (7-way: self/opp × civ/sci/mil, + draw — an explicit trained estimate of each moonshot threat), final VP margin, final military-track position, final science-symbol counts | KataGo lesson: dense long-horizon signal; W/D/L (never margin-blended) is what makes the agent risk-seeking when behind and blocking when ahead |
| Control model | Flat MLP on same features | Fast baseline + architecturally independent sparring partner (run5/run10 monoculture lesson) |
| Buffer schema | Replayable from day 1: seed + action sequence + per-move search stats | Reanalyze (Phase G) is a bolt-on, not a migration |
| Gating | SPRT (fishtest-style), not fixed-N matches | ~3× cheaper gating on clear results |
| Exact layer | Expectiminimax + Star1/Star2 tail solver, Age III exact / Ages I–II exact-to-boundary | §7; reuses the generic Rust `search::Game` trait (game #2) |

---

## 2b. Kingdomino reuse map

7WD is game #2 for a stack deliberately built to be extracted at game #2. Explicit
inventory so no phase silently rebuilds something that exists — and so nothing
KD-shaped gets forced onto 7WD where it doesn't fit.

### Reuse via extraction (the "build one, extract at two" items — extraction happens now)
| Asset | Where it lives | Action |
|---|---|---|
| Generic search trait (`Game`/`Eval`/`SearchConfig`, `splitmix64`, expectiminimax `value`/`choose_action`, Star-ready chance children) | `kingdomino_rust/src/search.rs` | Physical crate split into a shared search crate at Phase F; 7WD implements `Game` (needed by the Phase H solver regardless) |
| Python loop harness: worker orchestration, promotion/gating, HOF, run manifests | `threaded_self_play.py`, `hof.py`, `evaluation.py`, `test_run_manifest.py` pattern | **Audit coupling in early Phase D, extract a game-agnostic loop package behind a small GameAdapter interface** (setup/step/encode/codec/match-runner). Same rule that justified `search.rs` |
| Elo ledger + anchors tooling | `elo_rating.py`, `elo_db*.json`, `elo_anchors*.csv`, `round_robin_eval.py` | Near-verbatim with agent adapters; add SPRT termination (new — KD used fixed-N round robins) |
| Batched inference service (leaf coalescing, GIL release / `allow_threads`, double-buffering) | `inference_service.py`, M6 Rust coalescing design | Port the design; the transformer swaps in behind the same batching boundary |

### Reuse as pattern / template (copy-adapt, don't share code)
- **Trainer skeleton**: `nnue/train.py` — game-honest `iteration_split`, metrics vs
  trivial baselines, early-stop, no-scipy Spearman. Phase B copies the shape, swaps
  the model and heads.
- **Versioned model-export discipline**: `nnue/export.py` + Rust loader — magic/
  version/dims header + **encoder-signature hash enforced at load**. Applies the day
  Phase F4 moves inference in-process (tch/ONNX); until then it's the template for
  checkpoint manifests.
- **Differential-gate test style**: `test_rust_*_equiv.py` family — byte-exact replay
  gates, golden files, make/unmake fingerprint round-trips. This is the Phase F
  acceptance methodology, not optional flavor.
- **Cloud/box launch scripts**: `CLOUD_RUN.md`, `setup_cloud.sh`, `runs/` layout,
  `training_log_*.jsonl` conventions — Phase G runs reuse the operational playbook.
- **BGA anchoring**: the scraping/Elo-anchor approach (e.g. `bga_denial_anchor.py`)
  — Phase I rebuilds it against 7WD's BGA tables.

### Explicitly NOT reused (decided, don't revisit by accident)
- KD `action_codec.py` / `encoder.py` — 7WD's are new by design (identity-indexed,
  token schema). Only the *testing* style carries over.
- KD buffer `Example` pickle format — superseded by the replayable A4 schema.
- Sparse NNUE accumulator + dense-eval path (`sparse_nnue.rs`, `nnue_features.rs`) —
  a transformer under MCTS batches on GPU; the accumulator solves a per-leaf-CPU-eval
  problem 7WD doesn't have. (It returns only if a 7WD alpha-beta+NNUE line is ever
  spun up.)
- Any outcome·BIG+margin blended scalar (KD precedent, see §13).

### Open decision recorded: the Rust batched MCTS
KD's AlphaZero MCTS (M4–M6: arena tree, batched leaves, in-process coalescing) is
KD-coupled in `lib.rs` and is **PUCT + KD's chance model — not Gumbel, no explicit
enumerable chance nodes**. Decision for Phase F3: **write the generic searcher fresh
in the shared crate** (Gumbel root + sequential-halving + first-class chance nodes
differ structurally from the KD tree), while **porting the proven scaffolding
around it** — arena allocation, leaf-batch coalescing, `allow_threads` boundary,
double-buffer scheduling. Bending the KD tree into Gumbel-with-chance-nodes would
cost more than the rewrite and forfeit the clean equivalence gate against the
Python reference searcher.

---

## 3. Phase A — Contracts: action codec, encoder, chance interface (Python)

The M3-style spec everything downstream hangs off. **Write the spec doc first, then
implement against it.** Deliverable: `CODEC_SPEC.md` + `codec.py` + `encoder.py` +
`tests`.

### A1. Action codec (fixed integer space, identity-indexed)
Approximate layout (~1,200 actions):

| Block | Actions | Notes |
|---|---|---|
| Wonder draft pick | 12 (wonder id) | masked to visible draft row |
| Build card | 73 (card id) | masked to uncovered + affordable |
| Discard card | 73 | masked to uncovered |
| Card → wonder | 73 × 12 | masked to uncovered × own unbuilt drafted + affordable |
| Zeus destroy target | 73 | masked to opponent brown/grey |
| Mausoleum revive | 73 | masked to discard pile |
| Progress token pick (board) | 10 | masked to the 5 on board |
| Great Library token pick | 10 | masked to the 3 drawn |
| Next-age starter choice | 2 | self / opponent |

Gates: round-trip encode/decode over full games; every legal action from the engine
maps to exactly one index and back; mask generation exact vs `engine.legal_actions`
on ≥10k sampled states.

### A2. Token schema (encoder)
Sequence ≈ 60–110 tokens. Every token = learned embedding(entity id) + status features.

- **Global token**: coins (both), military pawn position, science symbol counts
  (both), age, turn player, pending-decision type, cards remaining, extra-turn flag.
- **Tableau slots** (≤20): face-up → card id embedding; face-down → back-type
  embedding (Age I / II / III / Guild) + covered/uncovered + row/position features.
- **My city / opponent city**: one token per built card (seat-relative frame, as in
  Kingdomino actor-relative encoding).
- **Wonders** (8 drafted): id + owner + built/unbuilt (+ what was buried under it).
- **Progress tokens**: board (5), mine, opponent's; Great Library candidates when
  pending.
- **Discard pile**: one token per discarded card (Mausoleum-relevant).
- **Unseen-pool summary**: per back type, pooled embedding (mean of unseen card
  embeddings) + count. Fed from the same structure the chance layer uses (A3).

Gates: encoder is a pure function of the *observation* (never of hidden identities —
verify by encoding the same observation under different hidden assignments and
asserting identical output); deterministic; golden-file tests on scripted games.

### A3. Unseen-pool structure + chance interface (single source of truth)
`UnseenPool`: per back type, the set of card ids not visible anywhere (tableau
face-up, cities, discard, buried-under-wonders). Consumers:
1. Encoder features (A2).
2. Chance-node children + probabilities (closed-loop mode): reveal of a slot with
   back type b → children = pool(b), each p = 1/|pool(b)|.
3. Determinizer sampling (open-loop mode): consistent assignment respecting back
   types.

Engine contract addition: `step()` results flag reveal events (slot, back type) as
first-class; multi-card uncovers surface as an ordered list (searcher treats them as
sequential chance nodes).

Gate: statistical test — sampled determinizations reproduce the exact marginals the
closed-loop enumeration claims (chi-squared over ≥100k samples on fixed positions).

### A4. Buffer schema
Per game: setup seed, agent versions, full action-index sequence, and per decision:
legal mask hash, root visit distribution, root value estimate, sims used, mode
(open/closed), Gumbel top-k set. Replay(seed, actions) must reproduce every state
bit-exactly (gate). This is what makes reanalyze, exact relabeling, and trap-suite
harvesting free later.

---

## 4. Phase B — Net + trainer (PyTorch)

- `net.py`: set transformer (pre-LN, no positional encoding beyond structural
  features; masked mean-pool or global-token readout). Heads: policy (~1.2k logits,
  legality-masked softmax), value (W/D/L 3-way), aux (winner × victory type 7-way, VP margin
  regression, military-track final position, science-count finals).
- `mlp.py`: control model on flattened features, same heads, same trainer.
- `train.py`: reuse the Kingdomino two-head trainer skeleton (BCE/CE + MSE, val
  metrics vs trivial baselines, early stop). Game-honest splits by iteration, as in
  Kingdomino `iteration_split`.
- Mixed precision + `torch.compile` from the start; batched-inference server API
  (even if Phase C calls it synchronously at first).

Gate: overfit a 500-game bot buffer to ~zero loss (wiring check); aux heads beat
predict-the-base-rate baselines on held-out bot games.

## 5. Phase C — Search (Python, dual-mode)

One MCTS core, Gumbel root (top-k=16 of legal actions, sequential halving, n=64
default), PUCT interior; chance handled by mode toggle:

- **Closed**: materialize chance children from `UnseenPool` with exact probs;
  sample-on-descent proportional to p; visit-weighted backup. Optional
  force-expansion of all chance children within 2 plies of root (one batched NN eval
  each — catastrophe coverage).
- **Open**: per-descent determinization from `UnseenPool`; nodes keyed by action
  path; per-world legality masking; priors cached at first expansion (the known
  weakness — that's the point of the A/B).

Gates: deterministic given seed; closed-mode root value on small positions matches a
brute-force expectimax with net leaves to 1e-6; open-mode converges to the same value
at high sims on the same positions.

## 6. Phase D — Loop bring-up at toy scale (Python)

Assemble: self-play workers (batched inference) → buffer → trainer → SPRT gate →
promote. Built on the loop harness extracted from Kingdomino per §2b (worker
orchestration, ELO ledger, HOF, run manifests) behind the GameAdapter interface,
with the 7WD engine as the first adapter client alongside a KD regression adapter
to prove the extraction didn't break the original.

Seeding & shaping, from iteration 0:
- Rush-bot buffer seeding: ~5k ScienceRushBot / MilitaryRushBot / greedy games mixed
  in, annealed out over ~10 iterations.
- Opponent mixing: 10–20% of self-play games vs rush bots / greedy early, annealed.
- Draft-prior blend at draft nodes: `p = (1−λ)p_net + λp_tier`, λ: 1 → 0 over ~20
  iterations.
- Playout cap randomization (KataGo): most moves cheap (n=16–24), random ~25% full
  (n=64–128); only full-search moves contribute policy targets.

Toy-scale defaults (tune freely): 500–1,000 games/iter, buffer window ~20 iters,
batch 512, lr 2e-4 cosine, temperature 1 → 0.25 by move ~20.

**Phase gate: gated net beats Greedy ≥65% and both rush bots ≥60% (SPRT-confirmed),
and the victory-type aux head predicts bot-game outcomes well above base rate.**
This proves codec, encoder, masking, search, and loop end-to-end before any Rust.

## 7. Phase E — Open vs closed loop A/B (the decision experiment)

Run after Phase D so priors "have opinions" (a flat-prior net understates the
stale-prior failure mode; re-check mid-training later).

- **Tier 1 — trap suite (hours):** harvest ≥100 positions from bot/self-play games
  where an uncovering action has ≥1 consistent reveal that hands the opponent an
  immediate win AND a safe alternative exists (engine detects mechanically). Ground
  truth = full-enumeration shallow expectimax with net leaves. Measure trap-pick
  rate + root-Q error: both modes × sims ∈ {32, 64, 128, 256} × ~20 seeds.
- **Tier 2 — head-to-head (weekend, if Tier 1 is close):** same net, both modes,
  alternating seats, SPRT stop. Report at BOTH equal-sims and equal-NN-evals /
  equal-wall-clock (closed does more evals at reveal layers; the compute-normalized
  number is what matters for training throughput).
- **Tier 3 — dual training runs: skipped** unless Tiers 1–2 are genuinely ambiguous.

**Decision rule: Rust implements only the winning mode; Python keeps dual-mode as
the slow reference implementation for equivalence gates.**
Prediction on record: closed wins trap coverage at low sims; open may win
equal-wall-clock strength. If results split along that line, prefer closed +
force-expansion near root, with sims budget adjusted.

## 8. Phase F — Rust port (the Kingdomino discipline)

Reuse `kingdomino_rust`'s generic `search::Game` trait — 7WD is impl #2, the
"extract at two" moment. Physical crate split into a shared search/NN crate happens
here, per the standing plan.

Order, each step behind a bit-exact/1e-6 equivalence gate vs the Python reference
(the Kingdomino M1–M6 pattern):
1. **Engine with make/unmake from day one** (solver and MCTS share one state core —
   do not retrofit like Kingdomino had to). Differential gate: replay ≥10k Python
   games byte-exactly; make/unmake round-trip fingerprint test. This is the cost
   center — 7WD effect resolution (chains, pendings, supremacies, extra turns) is
   meaningfully more intricate than Kingdomino.
2. **Encoder + codec in Rust**, bit-exact vs Python on ≥100k sampled states.
3. **Searcher (winning mode only)** + Gumbel root — written fresh in the shared
   crate per the §2b decision, porting KD's arena/coalescing/`allow_threads`
   scaffolding; tree values match the Python reference searcher on fixed
   seeds/positions.
4. **Batched inference bridge**: leaf coalescing + GIL release (port the Kingdomino
   `allow_threads` + in-process coalescing design); or ONNX/tch in-process if the
   Python hop dominates profiles.

Phase gate: ≥20× self-play throughput vs Python loop at equal settings (Kingdomino
achieved ~28×).

## 9. Phase G — Scaled training + modern training science

Now spend compute. In rough order of expected leverage:
1. **Reanalyze** (MuZero-style): background job re-searches buffer positions with the
   current net and refreshes policy/value targets; buffer becomes a compounding
   asset. (Schema already supports it — A4.)
2. **Net growth**: when curves flatten, train 6L×8H×256 (~10–20M) on the existing
   buffer, gate, continue. Repeat as justified. 92M is not a target; it's ZeusAI's
   unexamined default.
3. **Forced playouts + policy target pruning** at the root (KataGo).
4. **League play**: HOF checkpoints + rush-bot exploiters + the MLP control net as
   opponents in a fraction of games (monoculture insurance).
5. **Draws**: W/D/L head already models them; keep the official tiebreak cascade in
   terminal values (Kingdomino lesson — never blend outcome and margin into one
   scalar).

Instrumentation from the first scaled run: victory-type mix per iteration (expect
drift toward ~60/20/17 civ/sci/mil), science-win discovery iteration, draft pick-rate
table vs ZeusAI tiers, trap-suite blunder rate per checkpoint, Elo ledger vs frozen
anchors (greedy, rush bots, phase-D net, each HOF gate).

Card/wonder analytics module (`buffer_stats.py`): offline over the replayable
buffer — per-card build/discard/bury rates and conditional win rates split by
victory type, wonder pick + win tables, pairing lift (P(win | both built) vs
baseline), resource-cornering frequency. Analytics and health monitoring ONLY:
these statistics are policy-generated and never become encoder features or
training targets (circularity); the sole stats-into-training channels remain the
annealed draft prior and buffer seeding (§2, §6).

## 10. Phase H — Exact layer (expectiminimax tail solver)

Not pure alpha-beta: expectiminimax with alpha-beta on decision layers + Star1/Star2
on reveal chance nodes (bounded, 3-valued terminal outcomes prune hard). Exactness is
game-theoretically sound despite hidden cards because information is symmetric —
expectimax over the exchangeability marginals IS the true game.

- **Age III tail solve**: exact win/tie/loss from the official cascade; catches
  mid-age military/science instant wins. Feasibility: decision branching ~4–8
  (pyramid tapers), reveal m ≤ 6 late, TT + NN-policy move ordering → last 6–8 cards
  in ~seconds at Rust speeds; trigger on cards-remaining threshold or tree-size
  estimate; iterative deepening under the move clock; completed solve overrides MCTS.
- **Ages I/II**: exact-to-boundary with NN leaf at the pre-deal age boundary (solving
  past the deal is combinatorial). Ensure boundary states appear in training data or
  average the NN over sampled deals.
- **Root tactical verifier** (all phases of the game): shallow exact check that no
  candidate move allows an immediate loss via reveal or reply; cheap veto layer over
  the MCTS choice.
- **Training feedback**: relabel all buffer positions within solver range with proven
  values — the highest-quality value targets available, manufactured offline.

## 11. Phase I — Evaluation & the world-best campaign

- **Eval-time scaling**: tree reuse across moves, pondering, 10–50k sims for matches,
  force-expanded chance nodes at the root, tail solver active. Strength here is
  nearly free and ZeusAI's eval config (5k sims) is the bar to clear on search alone.
- **Anchors**: fixed ladder = random → greedy → rush bots → Phase-D net → HOF
  checkpoints; Elo ledger maintained per run (Kingdomino tooling).
- **Human benchmark**: BGA — the Kingdomino scraping/anchoring playbook applies.
  Target: sustained top-of-ladder Elo, then arranged matches vs top players
  (ZeusAI's bar: 2 of 3 top players beaten in bo5 — beat 3 of 3, decisively).
- **Robustness audit before any "world best" claim**: trap-suite blunder rate ≈ 0 at
  match settings; no exploitable opening-book hole vs league exploiters; victory-mix
  flexibility (agent can win all three ways when the position demands it).

---

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| 7WD engine port complexity (pendings, chains, supremacies) blows Phase F budget | Make/unmake differential gates vs Python on day 1; port effect-by-effect against the rules-oracle matrix |
| Gumbel low-sim search misses 1/11 catastrophic reveals | Closed-mode force-expansion near root; root tactical verifier; trap-suite regression per checkpoint |
| Stale-prior pathology if open mode wins the A/B | Re-run trap suite mid-training; per-descent prior re-masking; revisit decision if blunder rate drifts |
| Rush-bot seeding biases the policy | Anneal to zero; monitor victory mix; seeded games excluded from policy targets after iteration ~10 (value targets only) |
| Draft prior locks in ZeusAI's (possibly wrong) tiers | λ→0 anneal; log pick-rate divergence from tiers as the net takes over |
| Aux heads distract capacity at small scale | Aux loss weights ~0.1–0.3; ablate once in Phase D |
| Value head miscalibrated at age boundaries (pre-deal states) | Include boundary states in training encodes; or average NN over sampled deals in solver leaves |
| Buffer schema missing a field reanalyze needs later | A4 gate: full bit-exact replay from (seed, actions) — anything derivable is recoverable |

## 13. What is deliberately NOT in scope (yet)

- Pointer-style policy head (refinement after the flat identity codec is proven).
- Pantheon/Agora expansions, solo mode.
- Tier-3 dual training runs for the chance-mode question.
- Any margin/outcome blended scalar objective (banned by Kingdomino precedent).

## 14. Immediate next actions

1. Write `CODEC_SPEC.md` (action table with exact index ranges, token schema, mask
   rules, reveal-event contract, buffer record format).
2. Implement `codec.py` + tests (round-trip + mask-exactness gates).
3. Implement `UnseenPool` + engine reveal-event flagging + tests.
4. Implement `encoder.py` + observation-purity and golden-file tests.
5. `net.py` / `train.py` skeleton; overfit-500-games wiring gate.
6. Dual-mode searcher; brute-force-expectimax equivalence gate.
7. Rush bots (`bots.py` additions) + trap-position harvester script.
8. Coupling audit of `threaded_self_play.py` / `hof.py` / `evaluation.py` /
   `inference_service.py`; extract the game-agnostic loop package (§2b).
9. Toy loop on the extracted harness; Phase D gate; then the Phase E A/B.
