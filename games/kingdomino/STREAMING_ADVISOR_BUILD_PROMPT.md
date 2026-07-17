# Kingdomino Streaming Advisor — Build Prompt (Phases 0–2)

## Context

The full design is in **`games/kingdomino/STREAMING_ADVISOR_PLAN.md`** — read it
first; it is the source of truth. This prompt turns that plan into an
implementation task. **Do not redo the design review** — the architecture,
phasing, and reuse decisions are settled. Read the referenced code, confirm the
line anchors are still accurate (files drift), then implement in phase order.

**What we're building.** Turn the NN/MCTS advisor from a one-shot blocking call
into a background search that runs up to a sim cap while the extension polls and
re-renders every ~1 s. Goal: act early on easy positions, let it keep thinking on
hard ones.

**Scope guardrails (do not exceed without asking):**
- **NN engine only** (deck > 4). The exact engine stays on the existing bounded
  `/api/recommend` path — it is not sim-streamable.
- **`/api/recommend` stays byte-compatible.** Add new endpoints; do not change or
  remove the existing one. The `streaming` toggle (off) must fall back to it.
- **Item 3 (cross-move subtree reuse) is DEFERRED — do not build it.** See the
  plan's "Item 3" section for why. If you find yourself adding cross-request tree
  retention, stop.
- This is a **live, working tool.** Preserve the fast common-case behavior and add
  regression coverage at every phase.

Key files: `games/kingdomino/web_app.py`, `kingdomino_rust/src/lib.rs`,
`extension_kingdomino/content.js`, `extension_kingdomino/popup.js`. Tests:
`games/kingdomino/test_web_app_exact_advisor.py`,
`extension_kingdomino/tests/test_auto_settle.mjs`.

---

## Phase 0 (PRIMARY) — Plumbing: refactor + job registry + start/poll/stop

Ship the request/response machinery with **no behavior change** — one job runs the
existing fixed budget once and the client gets the same result via the new
endpoints. This de-risks everything downstream.

**Server (`web_app.py`):**
1. **Extract `_nn_recommendations_from_search(...)`** from the NN branch of
   `recommend()` (starts ~`web_app.py:1941`, continues past ~1977 — confirm the
   exact span). It takes the post-search inputs (`visit_counts, value0,
   action_info, root_traj, checkpoint_path, net`, plus `state, req`) and returns
   the recommendation dict. **Both** the existing sync endpoint and the new job
   snapshotter must call it, so the streamed payload is byte-identical in shape to
   today's response. Verify `/api/recommend` output is unchanged after the
   extraction (diff a few responses before/after).
2. **Job registry** (module-level, in-process): the `SearchJob` dataclass and
   `_SEARCH_JOBS` / `_STATE_TO_JOB` maps per the plan's §2. Single-active-job
   policy (new start cancels any running job; GPU eval contends). TTL/LRU reaper
   for finished jobs on each `/start`.
3. **Endpoints** (plan §4): `POST /api/recommend/start` (body = `RecommendRequest`
   + `max_sims` cap), `GET /api/recommend/poll?job_id=&since_version=`,
   `POST /api/recommend/stop`. Keep them **sync `def`** (Starlette threadpool → a
   poll can respond while a Rust search runs; Rust releases the GIL). Reject
   non-NN-eligible states from `/start` for now (exact stays on the sync path).
4. **Single uvicorn worker** — the registry is in-process. Confirm the dev launch
   command runs one worker; note it if not.

**Acceptance (Phase 0):**
- `POST /start` → `{job_id, status, version}`; `GET /poll` returns the standard
  recommendation body plus `{status, sims_done, sims_target, version}`; a poll
  with `since_version == version` returns a thin "no change" body; `POST /stop`
  cancels.
- The snapshot body is shape-identical to `/api/recommend` (same keys the
  extension's `renderRecommendations` consumes, including `draft_matrix`).
- New pytest coverage for the start→poll→done lifecycle and dedupe (two `/start`
  for the same `(state_key, params_key)` attach to one job).

---

## Phase 1 (SHIPS THE FEATURE) — Growing-budget chunk loop

Make the background worker actually refine progressively **without Rust changes.**

- Worker runs `advisor_open_loop_search` at budgets `C, 2C, 3C … up to
  max_sims`, checking `_stop` between chunks. Each result **supersedes** the prior
  (a larger independent re-determinized search is strictly better; visit counts
  are **not** additive across chunks). After each chunk: build snapshot via the
  helper, bump `version`, set `sims_done`.
- Start small (C ≈ 150–200) for a fast first refresh; grow so late refreshes are
  the expensive ones. Reuse the cached evaluator (`_RUST_ADVISOR_EVAL_CACHE` /
  `_load_nn_evaluator`).
- Python-fallback path degrades to the same chunking (can't stream concurrently as
  cleanly; acceptable — the live mid-game case is the Rust path).

**Extension (`content.js`, `popup.js`) — streaming controller:**
- On a new **settled** decision state (or manual capture): `POST /start` → poll
  loop at `refreshMs` (default 1000) hitting `/poll?since_version=`. Each changed
  snapshot → `renderRecommendations(...)` (already idempotent). Stop on
  `status=done|error`, position change, or user "Stop."
- **Rework the guards** (plan §Extension.2): `key === lastPayloadKey` now means
  *keep polling the existing job*, not "do nothing" (central behavior flip);
  `inFlightRecommend` + 1000 ms throttle become "job active for this state";
  `pollTick` starts/stops jobs on state change instead of firing once.
- On position change: `POST /stop` the old job, then `/start` the new one. This
  reuses the existing **settle** signal (`settleAutoCapture`) — extend
  `test_auto_settle.mjs` with a "settle triggers stop+restart, not a duplicate
  start" case.
- UI: progress line (`sims_done / sims_target`, elapsed, "searching…/converged"),
  a **"Stop / Good enough"** button → `/stop`, and an optional "stable — safe to
  move" hint when top-1 `action_id` + `visit_frac` are steady across the last K
  snapshots. Options: `maxSims` cap, `refreshMs`, `streaming` on/off (off → legacy
  one-shot path), plus the two fragility inputs below.

**Fragile-position testing — milestone pause/resume (plan §"Fragile-position
testing under streaming").** Fragility is split: the **static** half (opponent
robust/responses) is computed once; the **dynamic** half (headline = live search
Q) is tracked per refresh. Implement:
- **Two inputs**, wired through `popup.js`/`content.js` options **and**
  `RecommendRequest`:
  - **`fragility_at_sims`** — NEW field; default 1000, `0` disables fragility. The
    main-search sim count at which the one-time static pass fires.
  - **`draft_search_sims`** — EXISTING field (`web_app.py:197`, default 800); sims
    each fragility mini-search runs. Server-only today → surface it in the options
    UI (`fragilitySims`).
- **Worker:** when `sims_done` first crosses `fragility_at_sims`, set
  `status:"testing_fragility"`, **pause** the main search, run `_draft_matrix`
  **once** on that snapshot's `visit_counts`/`priors` at `draft_search_sims`, and
  store per pick `{representative_action, responses, robust_edge, realistic_edge}`.
  Resume the main search (Phase 1: next chunk is a larger independent search;
  Phase 2: exact resume from the preserved handle). Honor `_stop` during the pass.
- **Per-refresh merge (cheap):** each snapshot recomputes every stored row's
  `fragility = live_Q(rep) − stored_robust` from `action_info` — no nested searches
  re-run. **Rep-drift guard:** if a pick's live most-visited action ≠ its stored
  `representative_action`, flag that row "robust stale" (optionally re-run only
  that pick; a targeted re-run of flagged rows at Stop suffices). Label the panel
  with the sim count the static half was computed at.
- **Firewall:** the draft matrix's nested mini-searches stay independent and
  freshly-rooted — never routed through the Phase-2 handle/TT.

**Acceptance (Phase 1):**
- On a real deck>4 mid-game position the overlay refreshes ~1×/s with improving
  visit counts; the Stop button commits the current best immediately.
- With `fragility_at_sims=1000`, the ⚠ panel appears once the main search crosses
  1000 sims (brief "testing fragility" pause), then its headline/fragility values
  update live as sims climb **without** re-running the nested searches; `0`
  disables it. `draft_search_sims` changes the mini-search budget.
- `streaming` off → identical to today's one-shot behavior.
- Extension tests green, including the new settle stop/restart case.

---

## Phase 2 (THE PAYOFF) — Rust resumable handle (Item 1)

Replace the chunk-restart waste with a persistent tree. Implement Phase 2 as the
**resumable search handle** (plan Item 1), which is the preferred form of the
progress-callback:

- **Rust (`kingdomino_rust/src/lib.rs`):** turn the search into a handle —
  `SearchHandle::new(rs, ev, params)` + `.advance(n_sims) -> snapshot` +
  `.snapshot()`. The MCTS tree **and** a transposition table (keyed by an
  open-loop public-state signature) live inside the handle across `.advance`
  calls. The current entry is `advisor_open_loop_search_impl` (`lib.rs:5287`); the
  arena is `Vec<OLNode>` built and discarded per call today — lift it into the
  handle struct. **Requires a rebuild** (`maturin develop`) and re-running the
  Rust search tests.
- **Python job:** hold the handle, `.advance(chunk)` in a loop, check `_stop`
  between calls, write each snapshot + bump `version`.
- **Payoff:** the tree persists across chunks so visit counts **accumulate
  correctly** (no more independent-tree caveat), the TT removes duplicate
  expansion, and stop latency drops to one `.advance`. Supersedes Phase 1's loop
  once landed; keep Phase 1 as the no-Rust fallback.

**Firewall (carry into Phase 2 and beyond):** the handle/TT reuse is for the
**main root search only.** The draft matrix (`_draft_matrix`, `web_app.py:911`)
must keep using **independent, freshly-rooted** mini-searches — its whole purpose
is to defeat prior-starvation of rare opponent replies, which shared/warm trees
would silently reintroduce. Do not route the draft matrix through the reusable
handle.

**Acceptance (Phase 2):** one long streaming search shows monotonically
accumulating visit counts at a clean ~1 s cadence; stop latency ≈ one advance;
Rust solver/search value tests unchanged; draft-matrix fragility numbers unchanged
(it stays on the independent path).

---

## Also in scope — Item 2 (exact TT hit-rate audit), independent of streaming

Low-effort, safe, measurable; can be done any time (plan Item 2). Confirm
`_recommend_exact`, `_swindle_for_move`, and `_draft_matrix` all consult
`_cached_exact_margin` / the exact caches (`web_app.py:106-108`); instrument the
hit/miss ratio (`exact.cache_hits` / `cache_misses` in the response); raise
`_EXACT_ADVISOR_CACHE_MAX` if the clear-on-full eviction is thrashing a high hit
rate. Deck ≤ 4 is chance-free so cached margins are immutable — reuse can never be
wrong.

---

## Ordering & guardrails

1. **Phase 0** first — plumbing, no behavior change, validates the shape.
2. **Phase 1** — ships the streaming feature (no Rust).
3. **Phase 2** — Rust handle, the throughput payoff (rebuild + parity check).
4. **Item 2** — anytime; independent.
5. **Item 3** — **do not build.** Deferred per the plan.

Add/extend tests alongside each phase. Keep `/api/recommend` and every fast
common-case path behaviorally identical. Re-verify the line anchors above against
the current files before editing. If a phase's design detail is ambiguous, the
plan `.md` is authoritative; if the plan is also silent, ask rather than guess.
