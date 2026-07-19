# Kingdomino Advisor — Throughput Fix (implementation prompt)

## Context

A prior diagnosis-only review (see memory `kingdomino_advisor_throughput_review`)
identified three throughput problems in the live BGA advisor. This prompt turns
that diagnosis into an implementation task. **Do not re-run the investigation** —
the causes are established. Read the referenced code, confirm the line anchors are
still accurate (files may have drifted), then implement.

All work is in `games/kingdomino/` unless noted. This is a **live, working tool** —
preserve existing behavior for the fast/common cases and add regression coverage.

---

## Task 1 (PRIMARY) — Exact solver: 30 s budget, then graceful fallback to NN MCTS

**Problem.** Competent-play `deck=4` endgames are genuinely intractable for the
exact solver (≥50M-node trees, unsolved even at 60 s). Today `_recommend_exact`
raises **HTTP 504** on any unsolved root or child, and the interactive default
budget is huge (`exact_max_secs` defaults to `3600.0`, extension default 300 s),
so the user gets a multi-minute freeze that ends in a 504 error overlay.

**Required behavior.**
1. The exact solver runs for **at most 30 seconds**. If the root — or any legal
   child — is not solved within that budget, **do not raise 504**; instead
   **fall back to the NN/MCTS engine** and return a normal recommendation.
2. The response must clearly indicate the fallback happened (e.g.
   `engine: "nn-mcts"` plus a flag like `exact_fallback: true` and a short
   `reason`), so the UI/logs can distinguish an exact result from a degraded one.
3. The fast path is unchanged: positions the exact solver *can* crack in ≤30 s
   still return the exact result as they do now.

**Where.**
- `web_app.py`
  - `RecommendRequest.exact_max_secs` (line ~177): the field currently defaults
    to `3600.0`. Set the **interactive default to `30.0`** (keep the `le=3600.0`
    ceiling so power users can still override upward). Confirm the extension's
    own `exactMaxSecs` (`extension_kingdomino/content.js` / `popup.js`) is aligned
    to 30 as well.
  - `recommend()` dispatch (lines ~1868–1874). The `auto` branch calls
    `_recommend_exact` when `_exact_supported_detail(...) is None`. Wrap that call
    so a solver-timeout does **not** propagate as 504 but instead falls through to
    the existing NN/MCTS path (block at ~1916).
  - `_recommend_exact` (lines ~1048–1286) raises 504 in **three** spots: root
    unsolved (~1095) and each unsolved child (~1147). Convert "unsolved within
    budget" into a **catchable signal** (a dedicated `ExactTimeout` exception, or
    a sentinel return) rather than an `HTTPException(504)`, so the `auto` branch
    can catch it and degrade. Decide the policy for **explicit `engine=exact`**:
    recommend it *also* degrades to NN with the `exact_fallback` flag (a 504 is
    poor UX even when explicitly requested) — but if you keep 504 for the explicit
    engine, document why.

**Acceptance.**
- A known-hard real `deck=4` root (pull one from the BGA endgame logs; 169 of 419
  logged states qualify) returns an NN recommendation in ≈30 s, `exact_fallback:
  true`, **no 504**.
- A known-easy `deck=4` root (random-play, ~25 ms) still returns the exact result.
- Extend `test_web_app_exact_advisor.py` with a fallback case (mock/force the
  solver to report unsolved) asserting the NN result + flag, no exception.

---

## Task 2 (AMPLIFIER) — Remove the 3× redundant exact solve per node

**Problem.** `_recommend_exact` solves the root and **every child three times** —
value (`alpha=0`), rank (`alpha=0.5`), margin (`alpha=1`) — via
`_cached_exact_value`, `_exact_cache_stats_for`, and `_exact_margin_pts`. All three
are exact monotone transforms of one raw integer margin (measured identical at
25.6 ms each), so ~2/3 of the exact work is thrown away.

**Fix.** Solve each node **once** for the raw margin, then derive value / rank /
margin-points from that single result in Python (round the recovered margin to an
integer before re-applying `alpha`/`gain`/`scale`). Keep the existing output
fields bit-for-bit equivalent.

**Where.** `web_app.py` `_recommend_exact` — the three helper calls per child at
lines ~1076–1085 (`_exact_margin_pts`), ~1121–1128 and ~1153–1162
(`_exact_cache_stats_for` rank), ~1086–1093 and ~1138–1145 (`_cached_exact_value`).

**Acceptance.** For a batch of solvable roots, the new single-solve path returns
identical `value`/`rank_value`/`margin_pts` (to integer-margin precision) as the
current three-solve path, at ~3× fewer solver calls. Guard with a regression test.

---

## Task 3 (AMPLIFIER, larger) — Route the advisor through the transposition-table solver

**Problem.** The advisor calls `solve_endgame_ab_parallel` (no TT); the training
path calls `solve_root_exact_cached`, whose within-solve transposition table the
docs measured at 62–86% duplicate visits / ~3.4× speedup. The advisor leaves the
single biggest per-position lever unused.

**Fix.** Route the advisor's exact solves through the TT-backed solver.

**Where.** Rust: `games/kingdomino/kingdomino_rust/src/lib.rs` (~9996,
`solve_endgame_ab_parallel`) vs. the `solve_root_exact_cached` path. **This is a
Rust change and requires a rebuild** (`maturin develop` / project build step) and
re-running the Rust solver tests. Verify bit-identical solver *values* before and
after — the TT must not change results, only speed.

**Note.** Larger blast radius than Tasks 1–2. Land Tasks 1–2 first; this one makes
the 30 s budget in Task 1 succeed on more positions but is not required for correct
fallback behavior.

---

## Task 4 (OPTIONAL) — Client settle detection

The prior review flagged the static `AUTO_POLL_MS = 2500` + two-poll stability rule
in `extension_kingdomino/content.js` (`pollTick`, lines ~2422–2427) as a fixed
2.5–5.0 s pre-send delay. The dynamic fix keeps the "two equal reads" safety
guarantee but confirms in a fast inner loop (re-read ~150–200 ms, fire on two
consecutive matches, cap ~2.5 s), dropping typical latency to ~0.2–0.4 s.

---

## Ordering & guardrails

1. Task 1 first — it's the user-visible fix (no more multi-minute freeze / 504).
2. Task 2 next — Python-only, provably equivalent, low risk.
3. Task 3 if pursuing the hard tail — Rust rebuild + solver-value parity check.
4. Task 4.

Add/extend tests alongside each task. Keep the fast common-case paths behaviorally
identical. Reference anchors above are from the diagnosis; re-verify them against
the current files before editing.
