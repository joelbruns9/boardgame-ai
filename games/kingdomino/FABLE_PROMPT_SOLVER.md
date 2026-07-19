Analyze and propose improvements to the Kingdomino exact-endgame solver in this
repo (games/kingdomino/), covering three goals below. This is an ANALYSIS AND
PROPOSAL task, not an implementation task — read the code, run the existing
tooling, produce concrete findings and prioritized recommendations. Only
prototype/benchmark small things if it directly helps decide between options
(mirroring how endgame_solver_harness.py itself was built: measure before
proposing). Do not make large code changes without checking in first.

BACKGROUND — read these files first, in this order, before proposing anything:
1. games/kingdomino/exact_endgame_solver.md — the full design doc, including a
   "Status Update" section at the top with the current roadmap and open
   questions. This is the primary source of truth; don't rely on secondhand
   summaries.
2. games/kingdomino/kingdomino_rust/src/lib.rs — search for: solve_endgame_ab,
   solve_endgame_ab_parallel, order_legal_for_solver / order_legal_for_solver_at_depth
   / order_legal_for_solver_lookahead (the existing move-ordering heuristics —
   there are ~8 hand-crafted strategies: baseline, denial, lookahead, lookahead2,
   lookahead2_adaptive8/12/16/20, lookahead2_clustered (current production
   default), lookahead1_clustered, combined), solve_endgame_ab_value_cached
   (dead code — an abandoned leaf-level memoization attempt, NOT on the training
   path), terminal_search_value / margin_to_training_value (the win-gated leaf
   value, described below), and the no-retry-after-timeout sentinel logic
   (exact_unsolvable flag; a failed deck=4 solve gets exactly one retry once
   ≤2 claims remain, plus a guaranteed-cheap deck=0 attempt).
3. games/kingdomino/kingdomino_rust/src/deck0_draft_dp.rs — an EXPERIMENTAL,
   not-on-training-path deck=0 separable/DP benchmark. Full deck=4 separability
   (splitting placement from draft/selection) was explicitly considered and
   REJECTED: at deck=4 a selection round remains, and selection is the
   irreducible adversarial coupling (e.g. handing the opponent a 15-point tile
   to win the margin) — splitting the trees would discard exactly that. Do not
   re-propose full deck=4 separability. A narrower idea — using the fast deck=0
   separable solver as the LEAF evaluator beneath a deck=4 draft-only minimax —
   was parked as a smaller possible follow-up, not reopened.
4. games/kingdomino/endgame_solver_harness.py — the diagnostic tool. `generate`
   plays games and measures solve time/coverage/unpruned-tree-size on snapshotted
   endgame roots, persisting them (from_parts-compatible JSONL). `analyze` reports
   the tail distribution, timeout rate at various budgets, and a hardness
   signature splitting placement-axis vs draft/selection-axis branching (via
   decoded joint indices), plus a two-ordering value-equivalence guard (bit-
   identical solved values — use this to validate ANY ordering/pruning change).

NEW REAL DATA — this is the key opportunity:
games/kingdomino/runs/kingdomino/cloud_80x6_run1/exact_fallback_positions.jsonl
(2819 positions) contains REAL exact-solver timeout fallbacks from an actual
80-channel/6-block model's self-play, at a point where the model was already
beating a GreedyBot baseline 100% of the time (genuinely competent play, not
random/greedy proxy data). This is the first time faithful, competent-play hard
positions have been available — all prior tuning used random/greedy self-play as
a proxy, which is known to differ structurally from competent-play endgame
hardness. Load this file directly into `endgame_solver_harness.py analyze` (it
reads sidecar-format JSONL) as your primary evidence source for Goal 1.

Context on WHY positions get hard: as the model improves, self-play games become
genuinely competitive (net vs. itself), producing much harder-to-resolve endgame
positions than random/greedy play — this was observed live during training (the
exact-solver fallback rate rose from ~1-3% to ~12% and stayed there once the
model got strong, coinciding with self-play throughput collapsing ~3x as the
solver's wall-clock share grew). The retry+deck0 safety net recovered ~100% of
the initial-attempt fallbacks in the observed data (i.e., fallback at the FIRST
attempt doesn't mean a position goes unsolved — it usually just gets solved later,
cheaper, at the retry stage). Keep this in mind when reasoning about what a
"fallback" position actually costs.

=== GOAL 1: Improve the pruning heuristic using the real fallback corpus ===

Run `endgame_solver_harness.py analyze --in <path to exact_fallback_positions.jsonl>`
(and re-measure at a few different orderings if useful) to get the REAL
draft-axis vs placement-axis hardness signature on this faithful data — this
supersedes any conclusion drawn from proxy (random/greedy) data.

Then propose and evaluate concrete levers, in order of how directly they address
what the diagnosis shows is actually driving the hardness:
- Policy-prior move ordering: the one ordering signal NOT yet tried (all 8
  existing orderings are hand-crafted heuristics on board/domino features, not
  informed by the trained network's policy head). Only worth pursuing if the
  diagnosis shows ordering quality (not raw branching factor) is the bottleneck,
  since it couples the solver to a net-forward per node.
- A transposition table for a SINGLE root solve (not the abandoned leaf-level
  cache across the whole MCTS tree — that's the dead code above). Assess actual
  transposition-hit potential on the real corpus before committing to this.
- Aspiration windows / iterative deepening on the raw margin search.
Validate any proposed change with the harness's value-equivalence guard
(solved values must stay bit-identical to the current solver) before considering
it viable, and report node-count / solve-time deltas on the real corpus.

=== GOAL 2: Increase training signal from exact-solved endgames ===

Exact-solved positions get a 2x oversample weight during training
(ReplayBuffer.sample_batch's endgame_oversample_weight, gated on
game_progress >= 0.75 — see self_play.py). There is currently NO equivalent
midgame- or opening-specific oversampling mechanism.

Open question worth addressing: is there a point of diminishing returns on
endgame oversampling, where the network has extracted most of the learnable
signal from that game-progress band and gradient would be better spent on
midgame/opening (which have less-reliable labels and, per training logs, much
higher win-prediction error — endgame win_brier was ~0.05 vs midgame ~0.11 and
opening ~0.21 in one observed run at iteration ~65)? Note: midgame positions
ALREADY receive perfectly-accurate final-outcome (z) labels for free, since every
self-play game's actual ending is determined by the exact solver whenever it
reaches deck<=4 — oversampling endgame does NOT change midgame's z accuracy, it
only concentrates gradient on the endgame STATE-TO-VALUE mapping itself. Any
proposal here should be honest about that distinction rather than assuming
oversampling endgame is what "teaches" midgame.

Propose concrete, ideally data-driven mechanisms — e.g., a phase-brier-triggered
schedule (the codebase already computes per-phase win_brier/margin_mae
diagnostics; consider using that signal to trigger a gradual shift of oversample
weight rather than picking a fixed iteration number), or an entirely different
lever for extracting more value from the exact-solved subset (e.g., auxiliary
losses, distillation of exact values into earlier-in-game evaluations, or
using the solved endgame values to detect/calibrate value-head bias by phase).
Look at self_play.py's existing SelfPlayConfig schedule mechanism
(_compiled_schedules / _active_config_for_iteration — schedules are piecewise,
keyed by iteration) as the natural implementation pattern if you propose a
schedule-based mechanism.

=== GOAL 3: Applicability of an Azul NNUE paper ===

Read this paper: https://jakubkowalski.tech/Supervising/Rzepecki2025ImplementingSuperhuman.pdf
(implementing a superhuman NNUE-style agent for the board game Azul). Assess
what, if anything, is applicable to Kingdomino's setup — the current approach is
an AlphaZero-style CNN + MCTS with a Rust exact endgame solver, NOT NNUE-style
efficient incremental evaluation. Be clear-eyed about differences between Azul
and Kingdomino (game mechanics, branching factor, information structure) rather
than assuming blind transferability. In particular consider: whether NNUE's
efficiently-updatable evaluation concept has any relevance for the exact-solver
performance work in Goal 1 (incremental scoring during search, mentioned as
OPT-7/"incremental score tracking" in exact_endgame_solver.md, status "measure
before building" — is this paper informative there?), whether any of the paper's
training/self-play curriculum ideas are relevant to Goal 2, and whether the
paper's endgame-handling approach (if any) offers a different angle on Goal 1.

DELIVERABLE: a written analysis with concrete findings (cite actual numbers from
running the harness on the real data) and a prioritized list of recommendations
for each goal. Flag clearly which recommendations are "ready to implement" vs.
"needs a decision from the user first."


RESULTS:
Exact-Endgame Solver: Analysis & Recommendations

Method. Read exact_endgame_solver.md, the Rust solver internals, deck0_draft_dp.rs, the harness, and the Goal-2 training code; then ran three measurement passes against the real fallback corpus (runs/kingdomino/cloud_80x6_run1/exact_fallback_positions.jsonl, 2,819 positions): (1) a 96-position re-measure with the production-default YBW value solver at 10s, (2) a 48-position × 5-ordering sweep with the value-equivalence guard, (3) a 16-position emulation of the production per-child solve, plus a path-enumeration transposition check and a full pass over the run's training_log.jsonl. One note: the sidecar records lack the harness's measured fields, so analyze can't consume them directly yet — I re-measured via measure_endgame_tree with the same parameters the harness uses (scripts + results JSONL are in my scratchpad; happy to fold a --remeasure flag into the harness).

---
Goal 1 — Shrink the fallback tail

Finding 1 (the big one): the training path doesn't run the solver you benchmarked

Every tail number in the design doc measures solve_endgame_ab/solve_endgame_ab_parallel — a single-value root solve with alpha-beta bound sharing. But the training path (solve_root_exact_cached, lib.rs:3982) solves every root child with a full [-200,200] window to get exact per-child policy targets. No bound ever passes between siblings. Measured on 16 real fallback positions:

┌──────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
│                        metric                        │                               value                                │
├──────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ total work, per-child full-window vs YBW value-solve │ median 11×, range 4×–51×                                           │
├──────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ cost concentration                                   │ top-3 of ~24–52 children hold only ~23% of cost — every child pays │
├──────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ max single-child serial time                         │ 3–6s → the parallel critical path alone busts the 3s budget        │
├──────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ children individually timing out at 6s serial        │ 31/452                                │
└──────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────┘

Example: position i=101 — YBW value solve 0.75s; production-stylechild. The "12% fallback rate" is largely this 11× tax, notintrinsic position hardness.

Why it matters economically: the training log shows the solver is_solver_secs ≈ 500s vs ~515s total generation wall per iterationat steady state, games/s fell 2.4 → 0.77 as solver share grew, and the run had to starve game generation (2 game CPUs vs 22 solver CPUs). Cutting mean solve cost ~directly buys back self-play throughput; the tail is only part of the bill.

Finding 2: the real tail is genuinely harder than the proxy — and the retry net already catches it

- Local YBW value-solve on 96 real fallbacks: 58% solve at 10s; 4ved 4.4s). The proxy corpus's p99 was ~3–4s. Absolute-tailconclusions from greedy/random data are indeed invalid, confirming the doc's suspicion.
- But: 2,812/2,819 fallbacks are first-attempt deck=4 full-row roots; only 7 retry-stage failures in 66 iterations (0.25%). exact_tree_solve_count = 400/400
games every iteration — every game's ending was eventually exact-s of wasted solver time plus MCTS-grade (not exact) targets for ~2 moves until the retry lands. No endgame goes unsolved.

Finding 3: hardness signature — neither axis explains the tail; oitions do

- n_picks is structurally constant (4) at fallback roots; n_placements is only 1.44× higher in the hardest quartile vs easiest (8.2 vs 5.7). Root branching      barely discriminates.
- Ordering sweep (48 positions, 10s): all five orderings statistically tied — lookahead2 31/48 solved, production lookahead2_clustered 28/48, plain baseline 27/48. The hand-crafted family is saturated. Yet per-position spread is real (median 1.5×, p90 4.3× best-vs-worst) and 11 of 20 production-ordering timeouts were solved by some other ordering. Value guard: 0 mismatches.
- Transpositions are systematic, not rare: advance_round (lib.rs:938) sorts next_claims, so pick order is discarded at round boundaries. Enumerating the current round on 6 real positions: 4.00–4.38× path collapse each round, ~solve's two selection rounds — concentrated in the deepest,largest tree segment. The original TT skip rationale ("states nearly always unique") considered coincidental board collisions and missed this structural collapse.
Goal 1 recommendations, in priority order
1. Restructure the root solve: value-first + aspiration-windowed stly ready to implement.)
  - 1a — Per-child aspiration windows (ready): solve the best-ordered child full-window, then siblings with a window centered on it; re-search wider on fail-high/low. Values and policy targets stay bit-identical (validate with the harness guard), and most of the 11× tax disappears because far-below-best children fail low cheaply. The Azul thesis's "fullWindow beat aspiration" result doesn't transfer: theirs was a noisy-heuristic root-value search; ours has exact integer margins and a principled center.
  - 1b — Value-only rescue on timeout (ready; small design decision on emitted targets): if per-child solving can't finish in budget, fall back to the YBW value solve (median ~11× cheaper) → keep the exact value target + minimolicy. Recovers the more important half of the training signalfor most of the current 12% fallbacks.                                                                                                                             - 1c — ε-approximate policy for dominated children (needs your l-low bound is far enough below best get softmax weight <ε anyway; accepting the bound as their "value" skips the re-search. Changes exact-policy labels by <ε — bigger savings than 1a alone, but breaks strict bit-identity of the policy target (value/move unchanged).                                                                                                                        2. Within-single-solve transposition table — measured 4×/round cos a two-step.) Step 1: 30-line instrumented counter (HashSet ofstate keys inside count_endgame_nodes_no_chance or a solve variant) to get the realized hit rate under alpha-beta on the corpus. Step 2: if confirmed, a Zobrist-hashed TT (the current EndgameKey is ~1KB/entry — fine for the plan cache, too heavy per-node; the Azul thesis's split-the-state-into-16-bit-pieces XOR  table scheme is directly reusable). Crucially, share the TT acros_exact_cached — sibling subtrees overlap heavily, attacking thesame 11× inflation from another side. Store exact/lower/upper flags; validate with the guard.
3. Policy-prior move ordering: defer. The diagnosis says ordering headroom exists (11/20 rescues) but the bottleneck is the full-window structure and missing    transposition folding, both cheaper and net-decoupled. A cheaper wins: race two complementary orderings (e.g., lookahead2 +baseline) on split cores — the phase-2 complementarity is measured, and it needs no net forward. Revisit policy-prior only if 1+2 leave a material tail.
4. OPT-7 / incremental scoring: unchanged verdict — profile first. Nothing in the Azul paper changes this (see Goal 3); their speedup was SIMD quantization of a tiny dense MLP, not incremental evaluation.

---
Goal 2 — Training signal from exact endgames

Findings

- Half the gradient already goes to the endgame band. ~54% of recorded examples per game are exact-solver moves (11.8 of 21.8), and at endgame_oversample=2.0 the realized batch share is ~55% (283/512), implying a ~38%-endgame buffer — far above the doc's "≈33% on a 20% buffer" design point. Competent self-play made
endgames a much larger fraction of recorded moves.
- The endgame value mapping is learned; the trigger you built for this already fires. Phase briers: endgame 0.014–0.06 vs midgame 0.06–0.15 vs opening ~0.21 throughout. alpha_trigger (check_alpha_transition, endgame brier < 50% of baseline sustained) has been True since iteration 25 and is logged but unused. The
endgame brier's rise (0.014→~0.055) after iter 10 tracks self-plaive (score_diff std ~21, mean ~0) — irreducible outcomeuncertainty, not forgetting.
- Honest framing, per your note: dialing oversampling down doesn't touch midgame z-label quality (already exact-ending-grounded); it reallocates gradient from a
nearly-saturated state-value band to the bands with 2–4× worse ca

Recommendations

1. Ready: add endgame_oversample to _compiled_schedules (it's the one knob of its kind not schedulable today) and schedule it down, e.g. 0:2.0,25:1.25 for the next run — one-line-ish, uses existing infra. That shifts ~20% of batch gradient to mid/opening.
2. Needs your decision (recommended): act on the trigger instead heck_alpha_transition-style sustained ratio fires, step oversample 2.0→1.25 (mirror the stub's window/threshold; log the transition). This is the "empirical trigger over fixed schedule" goal the project plan already states for alpha.
3. Ready, cheap: add an is_exact flag to Example so exact_endgamect-labeled subset precisely (its own docstring asks for this).That's the instrument for detecting diminishing returns directly rather than via the progress≥0.75 proxy.
4. Same change as Goal 1's value-rescue (1b): it's also the biggest signal add — exact value labels for the ~12% of games that currently get MCTS-grade endgame
values at the hardest, most training-valuable roots.
5. Skip for now: distillation/auxiliary losses on exact values. With endgame brier already ~4–5% of baseline and >half the batch endgame-weighted, the marginal
unit of endgame signal is cheap gradient reallocated from where eive evidence from the Azul thesis: their split-by-phase two-netexperiment hurt (Table 7.6).

---
Goal 3 — Azul NNUE paper (Rzepecki 2025, MSc thesis)

What it actually is: depth-limited alpha-beta + TT + iterative deh a tiny quantized MLP (best: 128×16×16×1, dense hand-extractedfeatures, AVX2 int SIMD, ~410k evals/s), trained value-only on game results (−1/0/1, BCE) from best-agent self-play, model selection by duel gauntlets. Chance handled by sampling one determinization (branchingFactor=1 won). Reached #1 on BGA. Notable negatives: MCTS-with-random-rollouts failed badly; aspiration/MTD(f)
didn't beat full-window; killer moves didn't help; per-phase nets

Clear-eyed applicability:

- Not transferable — the NNUE core. Two reasons. (a) It isn't really exercised in the paper: the winning encoding was dense features, where the incremental-accumulator trick (NNUE's defining idea) doesn't apply; the speed came from int quantization + AVX2. (b) Kingdomino's architecture needs a policy
head for MCTS priors and trains on GPU; a CNN over 2D boards is t Azul's state is a small fixed vector. Swapping paradigms is notindicated by anything in this paper.
- Not informative for OPT-7. "Efficiently updatable evaluation" iore lookup tables for a game whose scoring is table-like.Kingdomino's terminal board.score() BFS is a different animal; the existing plan (profile before building) stands. If profiling does flag it, the analogous move is incremental region tracking (union-find), not anything from this paper.
- Directly reusable — the hashing scheme (feeds Goal 1 rec #2). T state into 16-bit pieces → per-piece random tables → XOR) isexactly the shape of Zobrist keying a Kingdomino TT needs, and their TT+hash-move machinery is the standard playbook if the TT lands.
- Endgame handling — nothing to borrow. Their "soft endgame computation" just replaces the net with exact scoring at terminal states (10·points_diff +
rows_diff). Kingdomino's exact solver is already far past this. Ifrom an independent effort that (i) exact values must override the net near terminals and (ii) margin, not just win/loss, belongs in the terminal value — both already core decisions here (win-gated value). Their conclusions even list "a separate endgame search" as future work — this project is ahead on that axis.
- Curriculum — weak overlap. Their loop (generate with best, traies the existing Milestone-6 gated self-play design; their "BCEbeat MSE because sign matters more than magnitude" rhymes with the win-gated value; their two-phase-net failure is mild evidence for Goal 2 rec #5.

---
Suggested sequence

1. Goal 1 rec 1a + 1b (aspiration windows + value rescue) — biggealue-guard-validated.
2. Goal 2 recs 1+3 (schedulable oversample + is_exact flag) — small, next-run-ready.
3. Goal 1 rec 2 step 1 (TT hit-rate instrumentation) → decide TT build on data.
4. Decisions needed from you: ε-approximate policy targets (1c)? ample schedule? and whether to also persist re-measured harnessfields into the sidecar so analyze runs on it directly.

Scratchpad artifacts if you want them kept: remeasure_out.jsonl (ring sweep + per-child breakdown), corpus_features.json, and thethree scripts — say the word and I'll tidy them into games/kingdomino/ or fold the re-measure mode into the harness.