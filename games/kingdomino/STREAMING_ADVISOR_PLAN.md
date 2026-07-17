# Streaming Advisor (Option B) — Implementation Plan

## Goal
Turn the NN/MCTS advisor from one-shot blocking into a background search that
runs up to a sim cap while the extension polls and re-renders every ~1 s, so the
user can act early on easy positions or let it keep thinking on hard ones.

**Scope:** NN engine only (deck > 4). The exact engine stays on the existing
bounded-solve path (it isn't sim-streamable). Optionally unify exact as a
single-shot job in a later phase.

---

## How it works today (baseline)
- `/api/recommend` is a synchronous, atomic call: `recommend()`
  (`web_app.py:1860`) is a blocking `def` that runs the entire search and
  returns a single JSON body. No streaming primitives (no WebSocket/SSE, no
  long-running search job).
- NN engine → `_rust_open_loop_search` (`web_app.py:1414`) →
  `kr.advisor_open_loop_search(rs, ev, sims, ...)` (`web_app.py:1441`) runs *all*
  `sims` simulations then returns `(children, value0)`. **No partial-result
  callback, no "run N more sims on the existing tree" interface** — the tree is
  built and discarded per call.
- Python fallback (`OpenLoopMCTS` / `run_pimc_open_loop`, `web_app.py:1549`) has
  the same fixed-budget, run-to-completion shape.
- Extension `triggerRecommend` (`content.js:2196`) POSTs once, awaits the full
  response, renders once. Guards that must be reworked: `inFlightRecommend`
  (`content.js:2197`), 1000 ms throttle (`content.js:2199`), and the
  `key === lastPayloadKey` "same decision state" skip (`content.js:2233`).
- `renderRecommendations` is already idempotent (re-renders the overlay on every
  auto-turn), so refreshing the display every ~1 s is trivial — the work is all
  in *producing* progressively-refined results.

---

## Server changes (`games/kingdomino/web_app.py`)

### 1. Refactor: extract the recommendation-assembly helper
The NN branch of `recommend()` (starts `web_app.py:1941`) inlines: root
trajectory pass, Rust/Python search, top-k sort, and draft-matrix. Extract the
**post-search** portion into:

```
_nn_recommendations_from_search(state, req, *, visit_counts, value0, action_info,
                                root_traj, checkpoint_path, net) -> dict
```

Both the existing sync endpoint and the new job snapshotter call it, so the
streamed payload is byte-identical in shape to today's response (this is what
lets `renderRecommendations` work unchanged). *Confirm the exact span to extract
when implementing — the branch continues past `web_app.py:1977`.*

**Exclude the draft matrix from this helper.** In streaming it is *not* part of
the per-refresh snapshot — it runs on its own milestone schedule (see "Fragile-
position testing under streaming"). The helper produces the fast headline body;
the stored fragility structure is merged into the snapshot when present. The sync
`/api/recommend` endpoint keeps running the draft matrix inline exactly as today.

### 2. Job registry (module-level, in-process)
```
@dataclass SearchJob:
    job_id: str
    state_key: str          # reuse _exact_state_key(state) for dedupe
    params_key: str         # engine/sims-cap/checkpoint/seed hash
    status: "running" | "done" | "error" | "cancelled"
    version: int            # bumped each snapshot; client skips unchanged
    sims_done: int
    sims_target: int
    snapshot: dict | None   # last _nn_recommendations_from_search() output
    error: str | None
    started_at / updated_at: float
    _stop: threading.Event
    _thread: threading.Thread
```
- `_SEARCH_JOBS: dict[str, SearchJob]` plus
  `_STATE_TO_JOB: dict[(state_key, params_key), job_id]` for dedupe.
- **Single active job policy:** starting a job cancels any other running job (GPU
  eval contends; one position analyzed at a time). Enforce with a lock.
- TTL/LRU reaper: drop `done`/`cancelled` jobs older than ~60 s on each `/start`
  to bound memory.

### 3. Background worker loop
- Reuses the cached evaluator (`_RUST_ADVISOR_EVAL_CACHE`) and
  `_load_nn_evaluator`.
- **Phase 1 (no Rust changes) — growing-budget chunks:** run
  `advisor_open_loop_search` at budgets `C, 2C, 3C … up to sims_target`,
  checking `_stop` between chunks. Each result *supersedes* the prior (a larger
  search is strictly better; visit counts are **not** additive across chunks —
  each is an independent re-determinized tree). After each chunk: build snapshot
  via the helper, bump `version`, set `sims_done`.
  - Cadence ≈ chunk duration; start small (e.g. C=150–200) for a fast first
    refresh, grow so late refreshes are the expensive ones.
  - Limitation: a large chunk can't be interrupted, so stop latency ≈ one chunk.
    Accepted for Phase 1.
- **Python fallback path (GIL-bound):** degrade to the same chunking; it can't
  run concurrently with polls as cleanly, acceptable because the live mid-game
  case is the Rust path.

### 4. New endpoints (keep `/api/recommend` as-is)
- `POST /api/recommend/start` — body = `RecommendRequest` + `max_sims` (cap) +
  optional `chunk` config. Resolves state, rejects non-NN-eligible (or, later,
  wraps exact as single-shot). Dedupes to a live job for the same
  `(state_key, params_key)`. Returns `{ job_id, status, state_key, version }`.
- `GET /api/recommend/poll?job_id=…&since_version=N` — returns
  `{ status, sims_done, sims_target, version, updated_at, ...snapshot }` where
  `...snapshot` is the standard recommendation body. Returns a thin "no change"
  body when `version == since_version`.
- `POST /api/recommend/stop` — `{ job_id }` sets `_stop`, marks `cancelled`.

### 5. Concurrency notes
- FastAPI sync `def` endpoints run in the Starlette threadpool → `/poll`
  responds while the Rust search runs (Rust releases the GIL via `py.detach`,
  per the milestone6 work). Keep endpoints sync `def`.
- **Single uvicorn worker required** — the registry is in-process. The dev
  command already runs one worker; multi-worker would split the registry.

---

## Rust change — Phase 2 payoff (`kingdomino_rust`)

The clean version that eliminates wasted tree rebuilds. Add to
`advisor_open_loop_search`:
- `progress_callback: Option<PyObject>` and `progress_every: usize` (sims).
- Every `progress_every` simulations, re-acquire the GIL (`Python::with_gil`)
  and call the callback with `(children_snapshot, value0, sims_done)`; callback
  returns a `bool` → cooperative stop. Keep the callback cheap (copy child stats
  only).
- Python side: the worker passes a callback that writes the snapshot + bumps
  `version`, and returns `_stop.is_set()`.

Result: **one** long search streams native progress at a clean ~1 s cadence with
correct cumulative visit counts, and stop latency drops to `progress_every`.
(Alternative design: a resumable search-handle object with `.advance(n)`
preserving the tree — bigger API surface; the callback is less invasive. See
"Search-state reuse" below — the handle design also unlocks cross-move reuse.)

---

## Extension changes (`extension_kingdomino/content.js`, `popup.js`)

### 1. Streaming controller (replaces one-shot POST for NN)
- On a new decision state (or manual capture): `POST /start` → `job_id`; begin a
  `refreshMs` (default 1000) poll loop hitting `/poll?since_version=`.
- Each changed snapshot → `renderRecommendations(...)` (already idempotent).
- Stop polling on `status=done|error`, on position change (new
  `autoCaptureKey`), or on user "Stop."
- On position change: `POST /stop` for the old job, then `/start` the new one.

### 2. Guards to rework
- **`key === lastPayloadKey` "same decision state" skip (`content.js:2233`)** —
  for streaming, "same state" means *attach to / keep polling the existing job*,
  not "do nothing." This is the central behavior flip.
- **`inFlightRecommend` (`content.js:2197`)** and the **1000 ms throttle
  (`content.js:2199`)** — move the single-flight concept from "HTTP request in
  flight" to "job active for this state." The 1 s poll cadence is the throttle.
- `pollTick` (`content.js:2449`) / auto-refresh: change from "fire once on state
  change" to "start/stop jobs on state change."

### 3. UI additions
- Progress line in the overlay: `sims_done / sims_target`, elapsed, and a
  "searching… / converged" badge.
- **"Stop / Good enough" button** → `/stop`; lets the user commit early.
- Optional convergence hint (serves the "easy position" case): if top-1
  `action_id` + `visit_frac` are stable across the last K snapshots, show
  "stable — safe to move."

### 4. Options (`DEFAULT_OPTIONS` `content.js:14`, `popup.js`)
- `maxSims` cap (augments/replaces fixed `sims`; the `SIM_OPTIONS` array
  `content.js:26` can seed the cap choices).
- `refreshMs` (default 1000).
- `streaming` on/off toggle — when off, fall back to the legacy one-shot
  `/api/recommend` path (keep it intact for backward compat and for exact).
- **`fragilityAtSims`** (new; default 1000, `0` = fragility off) — the main-search
  sim count at which the one-time static fragility pass runs. Maps to request
  field `fragility_at_sims`.
- **`fragilitySims`** (default 800) — sims each fragility mini-search runs. Maps to
  the existing request field `draft_search_sims` (`web_app.py:197`), which is
  server-only today; surface it in the options UI.

---

## Fragile-position testing under streaming

The draft matrix (`_draft_matrix`, `web_app.py:911`) — "fragile-position testing"
— splits into a **static** half and a **dynamic** half that must be scheduled
differently once the search is open-ended.

- **Dynamic (headline):** `headline_edge` for each pick is the main search's live
  Q — `value_sum/visits` (`web_app.py:1459`), surfaced per action in
  `action_info`. It refines every refresh and is free to read.
- **Static (robust / responses):** the per-opponent-response values come from
  nested mini-searches at a fixed budget (`draft_search_sims`, `web_app.py:1011`,
  `1062`). They are **not** a function of the main sim count — re-running them as
  main sims grow only reproduces the same numbers ± seed noise.

**Model — compute the static half once, track the dynamic half live:**
1. The main streaming search runs. When it crosses **`fragility_at_sims`**
   (default 1000), **pause** it (GPU contention → single active job; the pass also
   wants a frozen input snapshot) and run `_draft_matrix` **once** on that
   snapshot at **`draft_search_sims`** per mini-search.
2. Store, per pick: `{representative_action, responses, robust_edge,
   realistic_edge}`. Resume the main search (Phase 2 handle → exact resume from the
   preserved tree; Phase 1 → the next chunk is a larger independent search).
3. On **every** subsequent refresh, recompute each row cheaply:
   `fragility = live_Q(rep) − stored_robust`. No nested searches re-run; the ⚠
   panel tracks the sharpening headline for ~zero cost.

**Rep-drift guard (the only staleness case).** The stored robust was computed by
stepping the pick's *representative* action (most-visited, `web_app.py:1002`) and
scoring the resulting board, so it is valid only while that action stays the rep.
On each refresh, if a pick's live most-visited action ≠ its stored rep, flag that
row "robust stale" (optionally re-run only that pick). Rare in practice: in 2p the
opponent's response *set* depends on which tile you picked, not your placement
(disjoint boards), so rep drift changes only your board score in each line, not the
branch structure — and placements (a private optimization the net models well)
usually lock in early. A single targeted re-run of any flagged rows at Stop
suffices; no periodic full re-run.

**Two advisor inputs** (both exposed in the extension options **and** as
`RecommendRequest` fields):
- **`fragility_at_sims`** — new field; default 1000, `0` disables fragility. The
  main-search sim count at which the one-time static pass fires.
- **`draft_search_sims`** — existing field (`web_app.py:197`, default 800). The
  sims each fragility mini-search runs. Server-only today → also expose in the UI.

**Firewall (unchanged).** The fragility pass's nested mini-searches stay
independent and freshly-rooted — they never borrow the Phase-2 reusable handle/TT,
which would reintroduce the prior-starvation the analysis exists to catch.

---

## Phasing

| Phase | Deliverable | Rust? |
|-------|-------------|-------|
| **0** | Extract `_nn_recommendations_from_search`; job registry + start/poll/stop running the existing fixed budget once. Client gains start/poll/stop; behavior ≈ today. Validates plumbing. | No |
| **1** | Growing-budget chunk loop → real progressive refinement (cadence = chunk time). Ships the feature. | No |
| **2** | Rust progress callback + cooperative stop → single native streaming search, clean ~1 s cadence, low stop latency. The payoff. | Yes |
| **3 (opt)** | Convergence "safe to move" UX; wrap exact as a single-shot job for a uniform client; search-state reuse (see below). | Maybe |

---

## Key risks / decisions
- **GPU/eval contention** → enforce single active job (new start cancels old).
- **GIL:** Phase 2 callback must use `Python::with_gil` and stay cheap;
  Python-fallback searches can't stream mid-search → they degrade to Phase-1
  chunking.
- **Visit counts aren't additive across Phase-1 chunks** (independent
  re-determinized trees) — snapshot = latest full result, not an accumulation.
  Phase 2's single search accumulates correctly.
- **Single uvicorn worker** assumption (in-process registry).
- **Memory:** TTL/LRU reaping of finished jobs.
- **Backward compat:** `/api/recommend` untouched; exact stays there until
  (optionally) unified in Phase 3.

---

## Open questions before Phase 0
1. Exact span of the NN branch to extract into `_nn_recommendations_from_search`.
2. Whether to wrap exact into the job model now or leave it on the sync endpoint.

---

## Search-state reuse

### Item 1 — Persistent tree/TT *within* a streaming search (COMMITTED, folds into Phase 2)
The highest-ROI reuse, and it overlaps Phase 2 directly. Instead of the bare
progress-callback, implement Phase 2 as a **resumable search handle**:

- **Rust:** `advisor_open_loop_search` becomes a handle object —
  `SearchHandle::new(rs, ev, params)` + `.advance(n_sims) -> snapshot` +
  `.snapshot()`. The MCTS tree **and** a transposition table (keyed by an
  open-loop public-state signature) live inside the handle across `.advance`
  calls.
- **Python job:** holds the handle, calls `.advance(chunk)` in a loop, checks
  `_stop` between calls, writes each returned snapshot + bumps `version`.
- **Payoff:** eliminates Phase-1's chunk-restart waste — the tree persists
  across chunks so visit counts accumulate correctly, and the TT removes
  duplicate node expansion within a search. This supersedes Phase 1's
  growing-budget loop once landed; keep Phase 1 as the no-Rust fallback.

Safe and behavior-preserving: same search semantics, just not thrown away
between refreshes.

### Item 2 — Exact TT hit-rate audit (COMMITTED, no new machinery)
The exact caches `_EXACT_ADVISOR_VALUE_CACHE` / `_EXACT_ADVISOR_MARGIN_CACHE`
(`web_app.py:106-107`) already persist across requests keyed by
`_exact_state_key`. Because deck ≤ 4 is chance-free, solved margins/values are
**immutable** → reuse can never be wrong. The work is verifying the advisor
actually hits them — the throughput review flagged "advisor bypasses TT solver"
and ~3× redundant solves.

- Confirm `_recommend_exact` child solves consult the cache via
  `_cached_exact_margin` (check the key is stable across the root solve and the
  per-child solves).
- Confirm the swindle (`_swindle_for_move`) and draft-matrix (`_draft_matrix`)
  enumerations share the same cache — they solve many children that reach
  overlapping deeper states (the solver-restructure work measured 62–86%
  transposition duplication).
- Instrument the hit/miss ratio (already surfaced as `exact.cache_hits` /
  `cache_misses` in the response) and raise `_EXACT_ADVISOR_CACHE_MAX`
  (`web_app.py:108`) if the clear-on-full eviction is thrashing a high hit rate.

Low effort, safe, measurable; independent of streaming.

### Item 3 — MCTS subtree reuse across moves (DEFERRED — declined pending a recurrence-rate measurement)
Idea: reuse the previous decision's tree as a warm start for the next decision by
re-rooting at the actually-played path, **only within a "no-new-information
window" — reset to a fresh tree at every tile reveal.** Resetting at reveals
removes the correctness hazard (invalidation by newly drawn tiles) and makes the
re-root an *exact* operation, so the design is sound. It is deferred anyway
because the effort/payoff ratio is poor and the benefit lands in the wrong place.

**Why deferred (not cancelled):**
- **Small, self-anticorrelated payoff.** You inherit only the fraction of prior
  sims that went down the actually-played path (order ~10%, concentration-
  dependent), and only within a reveal window. It is largest when the game stays
  on the expected line (you were already confident) and thinnest exactly when the
  opponent surprises you (under-explored subtree → fall back to fresh). It helps
  least when you most need help.
- **Redundant with streaming.** The warm start only sharpens the *first* refreshes
  of the next decision. On easy positions you already decide fast without it; on
  hard positions streaming accumulates thousands of fresh sims that wash the
  inherited few hundred out in seconds. Streaming already delivers the "decide
  fast when easy, keep thinking when hard" goal this whole plan targets.
- **High, bug-prone complexity.** Needs cross-request session state (retain the
  Item-1 handle keyed by `table_id` + state fingerprint), path reconstruction
  from scraped captures, a deck-match gate, a net-new Rust re-root API, and every
  discard/fallback edge case (undo, re-capture, new table, unexplored opponent
  move). Failures are **silent** — a stale-tree bug yields worse recommendations,
  not a crash.
- **Firewall hazard vs. fragile-move testing.** The draft matrix
  (`_draft_matrix`, `web_app.py:911`) exists *because* prior-concentrated search
  starves rare-but-lethal opponent replies; its fix is deliberately independent,
  freshly-rooted mini-searches. Item 3's reuse premise is the opposite of that,
  so it must be firewalled to the main root search only and kept out of the draft
  matrix — one more constraint that is easy to break silently.

**Reopen only if** a cheap measurement justifies it: the **within-reveal-window
recurrence rate** (from game logs — how often two consecutive own-decisions fall
inside the same deck window) comes back high, **and** in practice you routinely
act on the first refresh before the fresh search converges. Short of that
evidence, do not build it. Items 1–2 attack the real bottleneck (draft-matrix
budget exhaustion) with bigger, safer wins.
