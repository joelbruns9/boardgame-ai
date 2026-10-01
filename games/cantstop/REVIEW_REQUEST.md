# Review request: Can't Stop advisor and XG-inspired search

Please review commit `69323eaa1d4dce3e7459b6372b1aef2be95eaa61` (`Add Cant Stop BGA advisor and rollout search framework`) on `cantstop-variant-solver`.

The commit contains 58 files, including the BGA advisor extension, training adjustments, the original selective turn search, and the newer rollout framework. Please prioritize correctness of the search estimates and decisions, then concurrency/recovery, then advisor and training regressions. This is a request for independent review, not a claim that the implementation is ready for training or deployment.

Paths below are relative to the repository root. The detailed design and implementation history are in [XG_SEARCH_PLAN.md](XG_SEARCH_PLAN.md); the earlier search is described in [SEARCH_EXPERIMENTS.md](SEARCH_EXPERIMENTS.md). The plan has historical progress entries: its opening status and earlier 'not started' statements are superseded by later implementation entries.

## Why we built this

The baseline already solves the remainder of the current turn exactly, conditional on neural-network values at turn boundaries. Its future-board values are estimates, so an exact turn calculation does not make the whole-game decision exact.

The first selective whole-turn search did not establish a strength gain: the four-expansion, depth-one confirmation won 989/2,000 games (49.45%; approximately 47.26%-51.64% at 95% confidence). This motivated a more systematic way to compare actual root actions, simulate future play, and reduce sampling noise. That result concerns the older search, not the newer rollout framework.

## Extreme Gammon foundation: scrutinize the transfer

The design foundation is the public Extreme Gammon documentation cited in the plan:

- [Search intervals](https://www.extremegammon.com/Searchinterval.aspx)
- [Support and technical explanations](https://www.extremegammon.com/support.aspx)
- [Extreme Gammon 2 manual](https://www.extremegammon.com/extremegammon2.pdf)

The three adopted ideas are progressive candidate evaluation/filtering, short or full rollouts with stronger early continuation play, and variance reduction. These are mechanics we intend to transfer; XG's numerical settings, speedup factors, and playing strength are not evidence for our implementation.

This is an XG-inspired design based on public descriptions, not a reproduction of XG's proprietary implementation. In particular, the current stronger continuation uses selective whole-turn expansion. It is not a fully expanded N-ply expectiminimax tree or a general MCTS implementation. Please flag documentation or results that imply otherwise.

Please assess whether the analogy is sound for Can't Stop: multiple decisions within a push-your-luck turn, variable turn lengths, blocking variants, and three/four-player self-interested opponents differ materially from two-player backgammon. In multiplayer, the actor maximizes its own win probability; opponents are not modeled as a coalition minimizing the challenger's value.

## Assumptions that need explicit review

1. **Value contract.** The NN consumes supported turn-boundary boards. Search results use absolute-seat probability vectors; network outputs may initially use a relative-seat representation. Validate every conversion, including mixed player counts, inactive-seat masking, terminal one-hot values, and next-player orientation.
2. **Horizon contract.** H means finishing the current turn plus H additional completed player turns. H=0 finishes this turn; H=1 also finishes the next player's turn. Stops and busts count equally as completed turns. Nonterminal endpoints are evaluated before the next opening dice.
3. **Policy-dependent values.** Full rollouts estimate outcomes under the specified continuation policy, not optimal play. Finite rollouts additionally inherit NN endpoint error. More samples reduce sampling noise; they do not remove either source of model error.
4. **Actual action comparison.** Force each legal dice or stop/roll action once, then permit subsequent decisions under the declared policy. Equivalent dice choices are deduplicated. Confirm the winning-bank, blocking, and fifth-column cases, and that a candidate does not accidentally inherit a forced baseline follow-up decision.
5. **Chance and random streams.** Dice must retain their correct marginal distribution. Search must never consume the arena's game-dice stream. Shared dice are indexed by simulation, completed turn, and within-turn roll; differing turn lengths must not shift later comparisons unintentionally.
6. **Uncertainty contract.** Adaptive confidence bounds cover sampling uncertainty for a fixed rollout policy and horizon only. They do not certify NN accuracy, perfect play, or improved arena strength. Progressive filtering is heuristic and carries no equivalent statistical guarantee.
7. **Baseline fallback.** Unresolved adaptive searches choose the baseline-preferred surviving action, even if another survivor has the highest corrected mean. Review whether this is consistently surfaced in reports and suitable for evaluating the proposed teacher.
8. **Cache identity.** Reuse assumes a fixed evaluator and continuation configuration within a cache lifetime. Review full board identity, rules, actor, phase, and stronger-versus-baseline distinctions. Existing advisor turn caching does not automatically apply to decision-specific future rollouts.
9. **Fixtures and pilots.** The 60 saved decisions are correctness fixtures (50 development, 10 reserved), not a representative strength dataset. Small timing and variance pilots cannot justify general performance claims.

## Highest-risk code and review questions

### 1. Rollout transitions and dice-luck correction

Files: `rollout_search.py`, `decision_search.py`, `rust_solver.py`, `turn_search.py`.

The correction is `expected(g) - realized(g)` added to the sampled payoff. The expectation must be computed before the dice draw using the same fixed table as the realized value; the bust branch must use the corresponding bust value. An inaccurate g is acceptable only if the correction still has zero conditional mean.

Please trace stop, bust, claim, terminal win, opening roll, and strengthened future-turn cases. Future opening rolls deliberately receive zero correction until a turn table exists. Corrected samples can leave [0,1] and must not be clipped or renormalized. Check paired covariance bookkeeping and whether random-seed reuse or control-variate construction introduces outcome-dependent bias.

### 2. Adaptive elimination and fallback semantics

File: `adaptive_search.py`.

Elimination uses bounded RAW paired payoffs, not the potentially unbounded corrected samples. For n paired samples, the radius is `sqrt(2 * log(2 * P * J / alpha) / n)`, where P is the original number of action pairs and J the predeclared number of stages. Review the union-bound argument, independence across simulation indices, aligned prefixes, and adaptive removal of candidates.

These bounds can be very conservative. Review the unresolved fraction and resulting tendency to retain baseline behavior before judging search strength. Confirm action selection, displayed option ranking, and reported value cannot be mistaken for the same rule when fallback is active.

### 3. Progressive filtering and nested continuation search

Files: `progressive_search.py`, `turn_search.py`.

Surviving contenders must be compared at the same final horizon and policy. Pruned values remain stage diagnostics, not comparable final estimates. Margin/candidate-cap filtering may remove the truly best action; the independent audit should expose that risk rather than validate against its own selection samples.

Inspect depth and completed-turn counters, bounded early-turn search, reach-weight priorities, ancestor backups, bust propagation, deterministic ties, and cache keys. Ensure stronger continuation never recursively launches unbounded rollouts. Cache position limits bound retained entries, not peak transient solve memory.

### 4. Native solver and shared inference

Files: `cantstop_rust/src/lib.rs`, `rust_solver.py`, `batched_evaluator.py`, `model.py`.

The native changes include search-support operations and solver sharing. Audit ownership, mutability, GIL/thread behavior, and the assumption that the shared root solver remains immutable during parallel candidate evaluation.

The batching broker has one inference worker, futures, a bounded pending-row queue, chunked requests, and per-request conversion back to absolute seats. Review deadlocks and lost futures around close, cancellation, queue backpressure, oversized requests, and inference exceptions. Check mixed two/three/four-player requests and output slicing. The model's seat-mask optimization must preserve behavior for noncontiguous inputs and all supported tensor shapes.

### 5. Resume, cancellation, and resource limits

Files: `adaptive_search.py`, `adaptive_compare.py`, `batched_evaluator.py`.

Stages commit only complete, aligned candidate batches. Interrupted partial work is discarded and regenerated from the same sample indices. Check transaction boundaries, checkpoint validation, atomic replacement, source/native/model identity, and resume equivalence. A checkpoint assumes a single writer; atomic replacement alone is not a multi-writer protocol or a guarantee against every storage failure.

Time limits are soft: baseline construction, a native solve, or inference can overrun them. The numeric-sample budget is not a total-process memory cap; Python lists, JSON serialization, transient tables, and GPU allocations add overhead. A baseline failure cannot produce a valid baseline fallback. Please check whether I/O failures and unexpected exception classes are handled and reported adequately.

### 6. Arena and experimental validity

Files: `adaptive_arena.py`, `adaptive_compare.py`, `decision_compare.py`, `benchmark_rollout_variance.py`, `search_arena.py`.

Verify seat rotation, rule selection, game/search RNG separation, challenger counts, and multiplayer null rate 1/player_count. Fixture-start smoke games must not yield strength verdicts. Confirm incomplete games/runs are not silently discarded in a way that biases results. Review seed reuse across decisions, code/checkpoint provenance, and whether each reported metric measures what its label claims.

### 7. Advisor and training changes bundled in this commit

Files: `advisor_adapter.py`, `advisor_cache.py`, `bga_extract.py`, `web_app.py`, `run_advisor.ps1`, repository-root `extension_cantstop/`, `phase4.py`, `train.py`, `run_queue.ps1`.

Review stale BGA captures, score/claim reconciliation, automatic blocking detection, opponent evaluation, dice-option ordering, fifth-column terminal banking, and retained per-turn results. Refresh/retry and request identity must prevent stale or failed responses from permanently blocking the panel or overwriting newer state.

Training changes delay personas for initially flat networks and allow a bounded share of unfinished games to be dropped. Review the five-percent threshold, empty-result edge cases, resume defaults, and selection bias from dropping stalled games. This is distinct from the new search-training integration, which has not been implemented.

## Evidence available and its limits

Prior implementation validation reported 354 targeted checks across search, advisor, and model behavior. Tests cover action transitions, variance correction, progressive reversals/filtering, adaptive allocation, resume, failures, serial/batched agreement, and model masks. These checks were not rerun merely to write this request; please independently reproduce the relevant suites against the commit and installed native extension.

Local artifacts under `runs/` are excluded from the commit and may need to be supplied separately:

- `decision_step1_nn_final_20260929.json`: 50/50 development cases matched baseline at zero extra budget.
- `rollout_variance_step3_final_cuda_20261001.json`: small exploratory variance pilot, not a strength result. The filename date is retained as saved.
- `progressive_step4_audit_cuda_20260930.json`: small independent rollout audit.
- `adaptive_step5_final_serial_r{1,2,3}_20260930.json` and matching `batched` files: three-position throughput repetitions. Median total elapsed time was about 0.803 seconds serial versus 1.087 seconds batched; all three positions remained unresolved. Serial remains the default.
- `adaptive_step5_resume_20260930.json`: soft-budget interruption/resume exercise.
- `adaptive_step5_arena_smoke_20260930.json`: fixture integration smoke, explicitly no strength verdict.

The installed native binary must match the source being reviewed; Python tests against an older binary are insufficient. Follow `AGENTS.md` for the project Python environment and elevated execution requirements. Extension checks are in `extension_cantstop/test_*.cjs`.

The new search has not yet established a playing-strength improvement. Planned S0-S6 quality/arena experiments remain outstanding; training integration and advisor integration of the new search remain gated on that evidence. Do not treat completed implementation steps as completed experimental gates.

## Requested review output

Please return findings in severity order with file/line references, a concrete failure scenario, and a suggested correction or targeted test. Distinguish verified defects from assumptions needing measurement. Explicitly assess:

- Whether the XG-inspired transfer and value/horizon contracts are valid.
- Whether rollout estimates and adaptive decisions have the claimed statistical meaning.
- Whether concurrency and recovery are safe enough for long offline runs.
- Whether the current tests justify starting the planned search-quality experiments.
- What must be fixed before search can generate training targets or enter the live advisor.

Please do not launch expensive arenas/training or modify the running advisor as part of the review. Identify any unavailable artifacts or validation gaps rather than assuming they passed.
