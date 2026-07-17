# Implementation prompt — Streaming Advisor Phase 3: exact-as-job unification + Item-3 measurement gate

Context: `games/kingdomino/STREAMING_ADVISOR_PLAN.md` (Option B design). Read it first.
Phases 0–2 are **done**: job registry + `/api/recommend/start|poll|stop` in `web_app.py`, the
persistent Rust `AdvisorSearchHandle` (cumulative tree + public-state TT) in
`kingdomino_rust/src/lib.rs`, the extension streaming controller in `content.js`, the
milestone fragility model (static-once + live headline + rep-drift guard), and the exact-cache
audit (Item 2 — all exact solves route through `_EXACT_ADVISOR_MARGIN_CACHE`). Item 1 (handle)
is done; Item 3 (cross-move tree reuse) remains **deferred** — this phase produces the
measurement that decides whether it ever reopens, not the feature.

## Goal

Make the exact engine a first-class job so the extension has **one** controller for every
engine, then close out the plan with the cheap Item-3 recurrence measurement.

Today the client still has two divergent paths that must be kept in sync: the streaming
controller (NN, deck > 4) and the legacy one-shot POST (exact-eligible states, streaming-off,
and the explicit-`nn`-endgame fallback). Exact solves also occupy a Starlette threadpool
thread for up to `exact_max_secs` (30 s default) while the client waits synchronously — the
exact experience never benefited from the job model's cancel/poll/progress plumbing.

## Why this shape (do not re-scope)

- **Exact is single-shot by nature** — it is not sim-streamable, so the job wraps one solve:
  `running → (one snapshot) → done`. Do NOT invent a progressive exact protocol as the core
  deliverable; incremental per-child publishing is a bounded stretch item only (see Build 3).
- **The timeout fallback belongs inside the job.** The sync path already falls back
  `ExactTimeout → NN MCTS` (with the leaf exact-hook disabled). In the job model this becomes:
  the exact attempt times out, and the SAME job continues as a normal NN streaming search.
  That converts today's "wait 30 s, then get one NN shot" into "wait bounded exact attempt,
  then watch NN refine live" — strictly better, and it deletes the last client-side special
  case (the explicit-`nn` endgame fallback added in the Phase-2 review fixes).
- **`/api/recommend` stays untouched** — backward compat and the streaming-off toggle. This
  phase only changes what the streaming controller sends and what `/start` accepts.
- **Do NOT build Item 3.** The deliverable is the recurrence-rate number from game logs; the
  plan's reopen criteria (`STREAMING_ADVISOR_PLAN.md`, "Reopen only if") make the decision.

## Build

### 1. Server — accept exact-eligible states in `/api/recommend/start`
- Remove the deck ≤ 4 rejection in `recommend_start`. Route on the resolved engine the same
  way the sync endpoint does (`_exact_supported_detail`): exact-eligible → exact job;
  otherwise → existing NN streaming job. Keep the single-active-job policy and the
  `(state_key, params_key)` dedupe unchanged.
- Exact job worker: run `_recommend_exact` once with the request budget; publish its full
  response dict as the job snapshot (`status="done"`, one version bump). The response shape is
  already what `renderRecommendations` consumes — no client render changes needed for the
  happy path.
- **Timeout continuation:** on `ExactTimeout`, tag the job (`exact_fallback=True`, `reason`)
  and continue the SAME job as the standard NN streaming loop (handle path, milestones,
  fragility — everything `_run_search_job` already does). Reuse the sync path's guards:
  `exact_endgame_enabled=False` / `exact_endgame_max_secs=0.0` so MCTS leaves don't re-enter
  the solver that just timed out. Carry `exact_fallback`/`reason` on every subsequent
  snapshot so the overlay can show why it is streaming.
- **Cancellation:** thread `job._stop` into the exact path at its natural seams — between the
  root solve, each per-child solve, and each swindle/draft enumeration step (the deadline
  plumbing from the Phase-2 work already bounds each individual solve). Stop latency ≈ one
  child solve; that is acceptable — document it.
- Status vocabulary: reuse the existing set. `running` covers the solve; if you want the
  overlay to distinguish, add at most one status (`solving_exact`) mirroring how
  `testing_fragility` was added — poll body shape must not otherwise change.

### 2. Extension — one controller
- `streamingEligible` becomes: `options.streaming` and the payload engine is anything the
  server can job-ify (nn AND exact AND auto). Delete the engine === "nn" restriction, the
  explicit-`nn`-endgame legacy fallback, and the "stop streaming on non-streaming state"
  branch — a position change from deck 5 → deck 4 is now just a normal stop/start into an
  exact job (the auto-settle test at `tests/test_auto_settle.mjs:82` shows the pattern).
- Overlay: while the exact solve runs, show the solve status + elapsed instead of the sims
  progress line (there is no `sims_done` yet); on `exact_fallback` snapshots, surface the
  reason line the sync path already renders. The finished exact snapshot renders exactly as
  today (margins, swindle, rank chips).
- Legacy one-shot path remains only behind `streaming: false`.

### 3. Stretch (only if 1–2 land cleanly): partial exact snapshots
Because every child solve is individually cached and the response assembly is cheap, the
exact worker MAY publish an intermediate snapshot after the root + each child solve completes
(rows appear as they solve, `partial: true` until done). Keep it strictly additive to the
response shape and behind the same version/poll mechanics. Skip without guilt — the median
cached endgame solve is fast enough that this mostly benefits cold caches.

### 4. Item-3 measurement (offline script, no advisor changes)
Write a small standalone script (e.g. `games/kingdomino/measure_reveal_recurrence.py`) that
reads the accumulating BGA game log (`kingdomino-bga-gamelog/v1` records appended by the
extension via `/api/game-log/append`) and reports, per game and aggregate:
- **Within-reveal-window recurrence rate:** how often two consecutive own-decisions fall
  inside the same deck window (no tile reveal between them) — the quantity the plan's Item-3
  reopen criterion names.
- Decision counts per window, and (if timestamps allow) the gap between consecutive own
  decisions, as a proxy for whether a warm start could ever matter at the 1 s refresh cadence.
Print a one-paragraph verdict against the plan's reopen criteria. Do not build any reuse
machinery regardless of the number — record the result in the plan doc under Item 3.

## Validate

- Extend `test_web_app_streaming.py`: exact-eligible `/start` returns a job that finishes with
  the full exact response shape; `ExactTimeout` (monkeypatched, as
  `test_web_app_exact_advisor.py` does) continues the same job as an NN stream with
  `exact_fallback` tagged on snapshots and the leaf exact-hook disabled; `/stop` during the
  child-solve loop cancels within one solve; dedupe still keys on the complete start payload.
- Keep the sync-path exact tests green untouched — `/api/recommend` behavior must not change.
- Extension tests: unified eligibility (nn/exact/auto all stream when streaming is on), the
  deck 5 → 4 transition performs stop → start (no legacy fallback, no gap), overlay handles a
  snapshot with no sims progress.
- Live check: play a BGA endgame with streaming on — exact advice appears via the job path,
  a deliberately tiny `exact_max_secs` shows the in-job NN fallback streaming afterward.
- Run the Item-3 script on the current game log and paste its output into the summary.

## Guardrails (carry forward, do not relitigate)

- Single active job; single uvicorn worker; TTL reaper — unchanged.
- Draft-matrix mini-searches stay freshly rooted and never touch the `AdvisorSearchHandle`
  TT (the prior-starvation firewall).
- `/api/recommend` request/response contract frozen.
- The global single-job policy is intentional and documented at `web_app.py:2326` — the
  multi-table case is out of scope.
