// content.js — Can't Stop advisor extension
//
// Phase 4, step 4: scrape BGA board state, POST to the local advisor server,
// auto-refresh at decision points, and render a richer recommendation overlay.
//
// Two conventions differ between BGA's DOM and the engine:
//   * DOM data-height is distance from the TOP of a column (0 = claimed).
//   * The engine stores progress as steps advanced from the BOTTOM, reaching
//     COLUMN_HEIGHTS[col] when the column is claimed.
//     => engine_steps = COLUMN_HEIGHTS[col] - data_height
//   * The engine stores a runner as the INCREMENT above the active player's
//     saved progress (get_current_progress = saved + runner), NOT its absolute
//     position. => increment = (HEIGHT - runner_top) - saved_steps
//
// Isolated-world note: a content script can't read `gameui`, so we inject a
// <script> into the page and get the result back as a JSON *string* (strings
// cross the Firefox content-script <-> page Xray boundary cleanly).

const browserAPI = typeof browser !== "undefined" ? browser : chrome;

const ADVISOR_URL = "http://127.0.0.1:8765/recommend"; // used in step 2
const RESULT_EVENT = "bga-advisor-state";

// Auto-refresh settings. Polling is intentionally simple/reliable for BGA.
const AUTO_REFRESH_ENABLED = true;
const AUTO_REFRESH_INTERVAL_MS = 700;
const MIN_RECOMMEND_INTERVAL_MS = 900;
const DEFAULT_SIMULATIONS = 500;
const DEEP_SIMULATION_STEPS = [500, 1000, 2000, 5000];

let currentSimulations = Number(window.localStorage.getItem("cantstop_advisor_sims")) || DEFAULT_SIMULATIONS;
if (!DEEP_SIMULATION_STEPS.includes(currentSimulations)) currentSimulations = DEFAULT_SIMULATIONS;

let autoRefreshTimer = null;
let inFlightRecommend = false;
let lastDecisionKey = null;
let lastRecommendStartedAt = 0;
let lastRenderedPayloadKey = null;

// Mirror of engine.COLUMN_HEIGHTS, for content-script-side validation.
const COLUMN_HEIGHTS = {
  2: 3, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13, 8: 11, 9: 9, 10: 7, 11: 5, 12: 3,
};

console.log("[advisor] content.js loaded");

// ---------------------------------------------------------------------------
// Page-context scraper. Its SOURCE is stringified and injected into the page,
// so it must be fully self-contained (no references to this file's scope).
// ---------------------------------------------------------------------------
function pageReadBoardState(resultEvent) {
  // Self-contained copy — runs in page world.
  const HEIGHTS = { 2: 3, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13, 8: 11, 9: 9, 10: 7, 11: 5, 12: 3 };

  function emit(obj) {
    window.dispatchEvent(new CustomEvent(resultEvent, { detail: JSON.stringify(obj) }));
  }

  try {
    if (typeof gameui === "undefined" || !gameui.gamedatas) {
      emit({ error: "gameui not available (not in a game yet?)" });
      return;
    }

    const gd = gameui.gamedatas;
    const playerorder = gd.playerorder; // [bgaId0, bgaId1] -> advisor index 0/1
    const players = gd.players;

    const colorToIndex = {};
    playerorder.forEach((bgaId, idx) => {
      colorToIndex[players[bgaId].color] = idx;
    });

    // --- Pass 1: raw DOM heights (distance from TOP; 0 = claimed). ---
    const runnersTop = {};          // col -> top-distance of the black runner
    const progressTop = [{}, {}];   // per player: col -> top-distance of saved marker
    const claimed = [[], []];
    const skipped = [];             // tokens we couldn't interpret

    document.querySelectorAll(".tokenspace.token").forEach((el) => {
      const col = +el.dataset.column;
      const height = +el.dataset.height;
      const colorClass = [...el.classList].find((c) => c.startsWith("color_"));
      const color = colorClass && colorClass.slice("color_".length);

      // Skip template/placeholder tokens (BGA keeps invisible ones in the DOM).
      if (!Number.isFinite(col) || !Number.isFinite(height)) {
        skipped.push({ column: el.dataset.column, height: el.dataset.height, color });
        return;
      }

      if (color === "000000") {
        runnersTop[col] = height;
      } else if (color in colorToIndex) {
        const p = colorToIndex[color];
        if (height === 0) claimed[p].push(col);
        else progressTop[p][col] = height;
      }
    });

    // --- Pass 2: convert TOP-distance -> engine BOTTOM-distance. ---
    const progress = [{}, {}];
    [0, 1].forEach((p) => {
      Object.keys(progressTop[p]).forEach((colKey) => {
        const col = +colKey;
        const H = HEIGHTS[col];
        if (H === undefined) { skipped.push({ column: col, why: "unknown progress column" }); return; }
        progress[p][col] = H - progressTop[p][colKey];
      });
    });

    const gs = gd.gamestate;
    const activeIdx = playerorder.findIndex((id) => String(id) === String(gs.active_player));

    // Runners belong to the ACTIVE player; engine stores the increment above
    // that player's saved progress.
    const runners = {};
    Object.keys(runnersTop).forEach((colKey) => {
      const col = +colKey;
      const H = HEIGHTS[col];
      if (H === undefined) { skipped.push({ column: col, why: "unknown runner column" }); return; }
      const absoluteSteps = H - runnersTop[colKey];
      const saved = (activeIdx >= 0 && progress[activeIdx][col]) || 0;
      runners[col] = absoluteSteps - saved;
    });

    emit({
      active_player: activeIdx,
      scores: playerorder.map((id) => +players[id].score),
      dice: (gs.args && gs.args.dice) || [],
      phase: gs.name,
      runners: runners,
      progress: progress,
      claimed: claimed,
      _debug: { runnersTop, progressTop, skipped },
    });
  } catch (e) {
    emit({ error: String((e && e.message) || e) });
  }
}

// ---------------------------------------------------------------------------
// Content-script side: inject the scraper, await the JSON snapshot.
// ---------------------------------------------------------------------------
function readGameState() {
  return new Promise((resolve) => {
    const script = document.createElement("script");

    function handler(event) {
      window.removeEventListener(RESULT_EVENT, handler);
      script.remove();
      try {
        const data = JSON.parse(event.detail);
        if (data.error) {
          console.warn("[advisor] capture error:", data.error);
          resolve(null);
        } else {
          resolve(data);
        }
      } catch (e) {
        console.warn("[advisor] failed to parse captured state:", e);
        resolve(null);
      }
    }
    window.addEventListener(RESULT_EVENT, handler);

    script.textContent =
      "(" + pageReadBoardState.toString() + ")(" + JSON.stringify(RESULT_EVENT) + ");";
    (document.head || document.documentElement).appendChild(script);
  });
}

// ---------------------------------------------------------------------------
// Build the /recommend request body. Server reads active_player, claimed,
// progress, runners, dice; it ignores extra keys (so _debug/scores/phase are
// dropped here and never sent).
// ---------------------------------------------------------------------------
function toRecommendPayload(state) {
  return {
    active_player: state.active_player,
    claimed: state.claimed,
    progress: state.progress,
    runners: state.runners,
    dice: state.dice,
    num_simulations: currentSimulations,
  };
}

function isAdvisorMoment(state) {
  // Deliberately no "my turn" filter here. The advisor can evaluate either
  // player's diceChoice; active_player in the payload controls perspective.
  return state.phase === "diceChoice" && Array.isArray(state.dice) && state.dice.length > 0;
}

function timeoutSignal(ms) {
  if (typeof AbortSignal !== "undefined" && AbortSignal.timeout) {
    return AbortSignal.timeout(ms);
  }
  const controller = new AbortController();
  setTimeout(() => controller.abort(), ms);
  return controller.signal;
}

async function postRecommendDirect(payload) {
  const response = await fetch(ADVISOR_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: timeoutSignal(30000),
  });

  const text = await response.text();
  let json = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch (e) {
    throw new Error(`Advisor returned non-JSON HTTP ${response.status}: ${text.slice(0, 200)}`);
  }

  if (!response.ok) {
    throw new Error(`Advisor HTTP ${response.status}: ${JSON.stringify(json)}`);
  }

  return json;
}

function sendRuntimeMessage(message) {
  // Supports both Firefox's promise API and Chrome's callback API.
  try {
    const maybePromise = browserAPI.runtime.sendMessage(message);
    if (maybePromise && typeof maybePromise.then === "function") {
      return maybePromise;
    }
  } catch (e) {
    // Some Chrome contexts require callback style; fall through.
  }

  return new Promise((resolve, reject) => {
    browserAPI.runtime.sendMessage(message, (response) => {
      const err = browserAPI.runtime.lastError;
      if (err) reject(new Error(err.message));
      else resolve(response);
    });
  });
}

async function postRecommendViaBackground(payload) {
  const response = await sendRuntimeMessage({
    action: "recommend",
    url: ADVISOR_URL,
    payload,
  });

  if (!response || !response.ok) {
    throw new Error((response && response.error) || "background recommend failed");
  }

  return response.data;
}

async function postRecommend(payload) {
  try {
    const data = await postRecommendDirect(payload);
    return { data, transport: "content-fetch" };
  } catch (directError) {
    console.warn("[advisor] direct fetch failed; trying background fallback:", directError);
    const data = await postRecommendViaBackground(payload);
    return { data, transport: "background-fetch" };
  }
}

// Sanity checks against the engine's invariants — surfaces scrape bugs early.
function validate(state) {
  const warns = [];
  const act = state.active_player;

  [0, 1].forEach((p) => {
    for (const [colS, steps] of Object.entries(state.progress[p])) {
      const col = +colS;
      const H = COLUMN_HEIGHTS[col];
      if (H === undefined) warns.push(`progress[${p}] unknown column ${colS}`);
      else if (steps < 0 || steps > H) warns.push(`progress[${p}] col ${col} = ${steps} (range 0..${H})`);
    }
  });

  for (const [colS, inc] of Object.entries(state.runners)) {
    const col = +colS;
    const H = COLUMN_HEIGHTS[col];
    if (H === undefined) { warns.push(`runner on unknown column ${colS}`); continue; }
    if (inc <= 0) warns.push(`runner col ${col} increment = ${inc} (expected >= 1)`);
    const saved = (act >= 0 && state.progress[act] && state.progress[act][col]) || 0;
    if (saved + inc > H) warns.push(`col ${col}: saved ${saved} + runner ${inc} = ${saved + inc} > height ${H}`);
  }

  const nRunners = Object.keys(state.runners).length;
  if (nRunners > 3) warns.push(`${nRunners} runners (max 3)`);
  if (act < 0) warns.push("active player not found in playerorder");

  // BGA keeps pools of unplaced "template" tokens (undefined column AND height);
  // those are expected and harmless. Only warn about skips that look like they
  // should have been placed (a real column with a bad height, or unknown column).
  const skipped = (state._debug && state._debug.skipped) || [];
  const isTemplate = (s) =>
    (s.column === undefined || s.column === null || s.column === "") &&
    (s.height === undefined || s.height === null || s.height === "");
  const suspicious = skipped.filter((s) => !isTemplate(s));
  if (suspicious.length) warns.push(`${suspicious.length} unexpected skipped token(s) — see _debug.skipped`);

  return warns;
}


// ---------------------------------------------------------------------------
// Step 3: simple on-page advisor overlay.
// ---------------------------------------------------------------------------
function formatPct(x, digits = 1) {
  if (typeof x !== "number" || !Number.isFinite(x)) return "—";
  return (100 * x).toFixed(digits) + "%";
}

function normalizeMoveText(rec) {
  // Server response shapes may evolve; tolerate several common field names.
  const move = rec.move || rec.action || rec.columns || rec.move_key || rec.move_tuple;
  const decision = rec.decision || rec.choice || rec.stop_continue || "";

  let moveText = "move ?";
  if (Array.isArray(move)) {
    if (move.length === 1) moveText = `advance ${move[0]}`;
    else if (move.length === 2 && move[0] === move[1]) moveText = `advance ${move[0]} twice`;
    else if (move.length === 2) moveText = `advance ${move[0]} & ${move[1]}`;
    else moveText = JSON.stringify(move);
  } else if (typeof move === "string") {
    moveText = move;
  } else if (move !== undefined && move !== null) {
    moveText = JSON.stringify(move);
  }

  return decision ? `${moveText} — ${decision}` : moveText;
}

function pickScore(rec) {
  // Prefer visit share / probability for row ranking display. Fall back to Q/value.
  const fields = ["visit_share", "prob", "policy", "p", "q", "Q", "value", "win_prob"];
  for (const f of fields) {
    const v = rec[f];
    if (typeof v === "number" && Number.isFinite(v)) return { label: f, value: v };
  }
  return null;
}

function firstFiniteNumber(obj, fields) {
  for (const f of fields) {
    const v = obj && obj[f];
    if (typeof v === "number" && Number.isFinite(v)) return { label: f, value: v };
  }
  return null;
}

function formatNum(x, digits = 3) {
  if (typeof x !== "number" || !Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

function formatPctMaybe(x, digits = 1) {
  if (typeof x !== "number" || !Number.isFinite(x)) return "—";
  if (Math.abs(x) > 1.5) return x.toFixed(digits) + "%";
  return formatPct(x, digits);
}

function setCurrentSimulations(n) {
  const parsed = Number(n);
  if (!Number.isFinite(parsed) || parsed <= 0) return;
  currentSimulations = Math.round(parsed);
  window.localStorage.setItem("cantstop_advisor_sims", String(currentSimulations));
}

function nextDeeperSimulationCount() {
  for (const n of DEEP_SIMULATION_STEPS) {
    if (n > currentSimulations) return n;
  }
  return currentSimulations * 2;
}

function metricChip(label, value, title) {
  const chip = document.createElement("span");
  chip.textContent = `${label}: ${value}`;
  chip.title = title || "";
  chip.style.cssText = [
    "display:inline-block",
    "padding:2px 6px",
    "border-radius:999px",
    "background:rgba(15,23,42,0.9)",
    "border:1px solid rgba(71,85,105,0.9)",
    "color:#cbd5e1",
    "font-size:11px",
    "margin:4px 4px 0 0",
    "white-space:nowrap",
  ].join("; ");
  return chip;
}

function bar(widthPct) {
  const outer = document.createElement("div");
  outer.style.cssText = "height:5px;background:rgba(15,23,42,0.95);border-radius:999px;overflow:hidden;margin-top:6px;";
  const inner = document.createElement("div");
  const clamped = Math.max(0, Math.min(100, widthPct));
  inner.style.cssText = `height:100%;width:${clamped}%;background:#38bdf8;border-radius:999px;`;
  outer.appendChild(inner);
  return outer;
}

function getRecommendationMetrics(rec) {
  // Field names are intentionally permissive so the UI survives server schema tweaks.
  const visit = firstFiniteNumber(rec, ["visit_frac", "visit_share", "visits_frac", "visit_prob", "search_prob", "mcts_prob", "prob", "p"]);
  const prior = firstFiniteNumber(rec, ["prior", "policy_prior", "nn_prior", "raw_prior"]);
  const win = firstFiniteNumber(rec, ["win_prob", "win_probability", "q", "Q", "value", "child_q"]);
  const visits = firstFiniteNumber(rec, ["visits", "n", "N", "visit_count"]);
  return { visit, prior, win, visits };
}

function renderAdvisorOverlay(response, payload, transport, options = {}) {
  const existing = document.getElementById("cantstop-advisor-overlay");
  if (existing) existing.remove();

  const box = document.createElement("div");
  box.id = "cantstop-advisor-overlay";
  box.style.cssText = [
    "position: fixed",
    "top: 80px",
    "right: 16px",
    "z-index: 999999",
    "width: 390px",
    "max-height: 74vh",
    "overflow: auto",
    "background: rgba(18, 24, 38, 0.97)",
    "color: #f8fafc",
    "border: 1px solid rgba(148, 163, 184, 0.5)",
    "border-radius: 12px",
    "box-shadow: 0 12px 30px rgba(0,0,0,0.35)",
    "font: 13px/1.35 system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
    "padding: 12px",
  ].join("; ");

  const header = document.createElement("div");
  header.style.cssText = "display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:8px;";

  const title = document.createElement("div");
  title.textContent = "Can't Stop Advisor";
  title.style.cssText = "font-weight:700; font-size:15px;";

  const controls = document.createElement("div");
  controls.style.cssText = "display:flex;align-items:center;gap:6px;";

  const simSelect = document.createElement("select");
  simSelect.title = "MCTS simulations for the next search";
  simSelect.style.cssText = "background:rgba(51,65,85,0.9); color:#e2e8f0; border:1px solid rgba(100,116,139,0.9); border-radius:6px; font-size:12px; cursor:pointer; padding:2px 5px;";
  const simOptions = Array.from(new Set([...DEEP_SIMULATION_STEPS, currentSimulations])).sort((a, b) => a - b);
  simOptions.forEach((n) => {
    const opt = document.createElement("option");
    opt.value = String(n);
    opt.textContent = `${n} sims`;
    if (n === currentSimulations) opt.selected = true;
    simSelect.appendChild(opt);
  });
  simSelect.addEventListener("change", () => {
    setCurrentSimulations(simSelect.value);
    triggerRecommend({ force: true, reason: "sim-count-change" });
  });

  const deep = document.createElement("button");
  deep.textContent = "Think deeper";
  deep.title = "Increase simulations and recompute this position";
  deep.style.cssText = "background:rgba(22,101,52,0.9); color:#dcfce7; border:1px solid rgba(74,222,128,0.75); border-radius:6px; font-size:12px; cursor:pointer; padding:2px 7px;";
  deep.addEventListener("click", () => {
    setCurrentSimulations(nextDeeperSimulationCount());
    triggerRecommend({ force: true, reason: "think-deeper" });
  });

  const refresh = document.createElement("button");
  refresh.textContent = "↻";
  refresh.title = "Refresh now";
  refresh.style.cssText = "background:rgba(51,65,85,0.9); color:#e2e8f0; border:1px solid rgba(100,116,139,0.9); border-radius:6px; font-size:13px; cursor:pointer; padding:2px 7px;";
  refresh.addEventListener("click", () => triggerRecommend({ force: true, reason: "manual-overlay-refresh" }));

  const close = document.createElement("button");
  close.textContent = "×";
  close.title = "Close";
  close.style.cssText = "background:transparent; color:#cbd5e1; border:0; font-size:20px; cursor:pointer; line-height:1;";
  close.addEventListener("click", () => box.remove());

  controls.appendChild(simSelect);
  controls.appendChild(deep);
  controls.appendChild(refresh);
  controls.appendChild(close);
  header.appendChild(title);
  header.appendChild(controls);
  box.appendChild(header);

  const meta = document.createElement("div");
  meta.style.cssText = "color:#cbd5e1; margin-bottom:10px;";
  const valueText = typeof response.value === "number" ? formatPct(response.value) : "—";
  const searchMs = response.search_ms !== undefined ? `${response.search_ms}ms` : "—";
  const sims = response.num_simulations !== undefined ? response.num_simulations : "—";
  const when = new Date().toLocaleTimeString();
  const auto = AUTO_REFRESH_ENABLED ? "auto on" : "manual";
  meta.innerHTML = `Win chance: <b style="color:#f8fafc">${valueText}</b><br>Search: <b style="color:#f8fafc">${searchMs}</b> / <b style="color:#f8fafc">${sims}</b> sims<br>Active player: <b style="color:#f8fafc">${payload.active_player}</b> · ${transport} · ${auto} · ${when}`;
  box.appendChild(meta);

  const recs = Array.isArray(response.recommendations) ? response.recommendations : [];
  const list = document.createElement("div");
  list.style.cssText = "display:flex; flex-direction:column; gap:7px;";

  if (!recs.length) {
    const empty = document.createElement("div");
    empty.textContent = "No recommendations returned.";
    empty.style.cssText = "color:#fca5a5;";
    list.appendChild(empty);
  } else {
    const maxVisit = Math.max(...recs.map((r) => {
      const m = getRecommendationMetrics(r).visit;
      return m ? m.value : 0;
    }), 0.000001);

    recs.slice(0, 8).forEach((rec, idx) => {
      const row = document.createElement("div");
      row.style.cssText = "background:rgba(30,41,59,0.95); border:1px solid rgba(71,85,105,0.8); border-radius:9px; padding:8px;";

      const metrics = getRecommendationMetrics(rec);
      const primary = metrics.visit || metrics.win || metrics.prior;
      const primaryText = primary ? (metrics.visit ? formatPct(metrics.visit.value, 1) : formatPctMaybe(primary.value, 1)) : "";

      const line1 = document.createElement("div");
      line1.style.cssText = "display:flex; justify-content:space-between; gap:8px; font-weight:650;";
      const left = document.createElement("span");
      left.textContent = `${idx + 1}. ${normalizeMoveText(rec)}`;
      const right = document.createElement("span");
      right.textContent = primaryText;
      right.title = primary ? primary.label : "";
      line1.appendChild(left);
      line1.appendChild(right);
      row.appendChild(line1);

      if (metrics.visit) {
        const pct = maxVisit <= 1.000001 ? metrics.visit.value * 100 : (metrics.visit.value / maxVisit) * 100;
        row.appendChild(bar(pct));
      }

      const chips = document.createElement("div");
      chips.style.cssText = "margin-top:2px;";

      if (metrics.visit) chips.appendChild(metricChip("visit", formatPctMaybe(metrics.visit.value), "MCTS visit share / recommendation probability"));
      if (metrics.prior) chips.appendChild(metricChip("prior", formatPctMaybe(metrics.prior.value), "Neural network policy prior before search"));
      if (metrics.win) chips.appendChild(metricChip("win", formatPctMaybe(metrics.win.value), "Estimated win probability / Q from active player's perspective"));
      if (metrics.visits) chips.appendChild(metricChip("N", formatNum(metrics.visits.value, 0), "Raw visit count"));
      const shown = new Set(["rank", "visit_frac", "visit_share", "visits_frac", "visit_prob", "search_prob", "mcts_prob", "prob", "p", "prior", "policy_prior", "nn_prior", "raw_prior", "win_prob", "win_probability", "q", "Q", "value", "child_q", "visits", "n", "N", "visit_count", "action_idx", "action_index"]);
      const extras = Object.entries(rec)
        .filter(([k, v]) => !shown.has(k) && typeof v === "number" && Number.isFinite(v))
        .slice(0, 3);
      extras.forEach(([k, v]) => chips.appendChild(metricChip(k, Math.abs(v) <= 1.5 ? formatPctMaybe(v) : formatNum(v), "extra server diagnostic")));

      if (chips.children.length) row.appendChild(chips);

      list.appendChild(row);
    });
  }

  box.appendChild(list);

  const footer = document.createElement("div");
  footer.textContent = "Auto-refresh runs on new diceChoice states. Change sims or use Think deeper to recompute this position.";
  footer.style.cssText = "margin-top:10px; color:#64748b; font-size:11px;";
  box.appendChild(footer);

  document.body.appendChild(box);
}

function renderAdvisorError(message) {
  const existing = document.getElementById("cantstop-advisor-overlay");
  if (existing) existing.remove();

  const box = document.createElement("div");
  box.id = "cantstop-advisor-overlay";
  box.style.cssText = "position:fixed;top:80px;right:16px;z-index:999999;width:320px;background:rgba(127,29,29,0.96);color:white;border-radius:12px;padding:12px;font:13px system-ui, sans-serif;box-shadow:0 12px 30px rgba(0,0,0,0.35);";
  box.innerHTML = `<b>Can't Stop Advisor</b><br><br>${message}`;
  document.body.appendChild(box);
}

// ---------------------------------------------------------------------------
// Auto-refresh helpers.
// ---------------------------------------------------------------------------
function stableStringify(obj) {
  if (obj === null || typeof obj !== "object") return JSON.stringify(obj);
  if (Array.isArray(obj)) return "[" + obj.map(stableStringify).join(",") + "]";
  return "{" + Object.keys(obj).sort().map((k) => JSON.stringify(k) + ":" + stableStringify(obj[k])).join(",") + "}";
}

function decisionKeyFromPayload(payload) {
  return stableStringify(payload);
}

async function triggerRecommend({ force = false, reason = "auto" } = {}) {
  if (inFlightRecommend) return { ok: false, skipped: true, error: "recommendation already in flight" };

  const now = Date.now();
  if (!force && now - lastRecommendStartedAt < MIN_RECOMMEND_INTERVAL_MS) {
    return { ok: false, skipped: true, error: "throttled" };
  }

  const state = await readGameState();
  if (!state) {
    if (force) renderAdvisorError("no state captured");
    return { ok: false, error: "no state captured" };
  }

  if (!isAdvisorMoment(state)) {
    if (force) {
      const msg = `not a decision moment (phase="${state.phase}", dice=${JSON.stringify(state.dice)}). Advisor runs during "diceChoice" with dice present.`;
      renderAdvisorError(msg);
      return { ok: false, skipped: true, error: msg, state };
    }
    return { ok: false, skipped: true, error: "not a decision moment", state };
  }

  const payload = toRecommendPayload(state);
  const key = decisionKeyFromPayload(payload);
  if (!force && key === lastDecisionKey) {
    return { ok: false, skipped: true, error: "same decision state", state, payload };
  }

  lastDecisionKey = key;
  lastRecommendStartedAt = now;
  return captureAndRecommend({ preCapturedState: state, reason, payloadKey: key });
}

function startAutoRefresh() {
  if (!AUTO_REFRESH_ENABLED || autoRefreshTimer) return;
  autoRefreshTimer = setInterval(() => {
    triggerRecommend({ force: false, reason: "auto-poll" }).catch((e) => {
      console.warn("[advisor] auto-refresh failed:", e);
    });
  }, AUTO_REFRESH_INTERVAL_MS);
  console.log(`[advisor] auto-refresh enabled (${AUTO_REFRESH_INTERVAL_MS}ms poll)`);
}

// ---------------------------------------------------------------------------
// Step 2 entry point: capture + POST /recommend. Wired to popup's button.
// ---------------------------------------------------------------------------
async function captureAndRecommend(options = {}) {
  if (inFlightRecommend) return { ok: false, skipped: true, error: "recommendation already in flight" };
  inFlightRecommend = true;

  try {
    const state = options.preCapturedState || await readGameState();
    if (!state) {
      console.log("[advisor] no state captured");
      return { ok: false, error: "no state captured" };
    }

    console.log("[advisor] raw snapshot:", state);

    const payload = toRecommendPayload(state);
    console.log("[advisor] /recommend payload:\n" + JSON.stringify(payload, null, 2));

    const warns = validate(state);
    if (warns.length) {
      console.warn("[advisor] validation warnings:");
      warns.forEach((w) => console.warn("  - " + w));
    } else {
      console.log("[advisor] validation: clean ✓");
    }

    if (!isAdvisorMoment(state)) {
      const msg =
        `not a decision moment (phase="${state.phase}", dice=${JSON.stringify(state.dice)}). ` +
        `Advisor runs during "diceChoice" with dice present.`;
      console.log("[advisor] " + msg);
      renderAdvisorError(msg);
      return { ok: false, skipped: true, error: msg, state };
    }

    console.log(`[advisor] valid decision moment ✓ active_player=${state.active_player}`);

    try {
      const { data, transport } = await postRecommend(payload);
      console.log(`[advisor] /recommend response via ${transport}:`, data);

      // Save the full result for easy inspection from the console/popup.
      try {
        await browserAPI.storage.local.set({
          last_recommendation: data,
          last_payload: payload,
          last_transport: transport,
          last_recommendation_at: new Date().toISOString(),
        });
      } catch (storageError) {
        console.warn("[advisor] unable to save last recommendation:", storageError);
      }

      lastRenderedPayloadKey = options.payloadKey || decisionKeyFromPayload(payload);
      renderAdvisorOverlay(data, payload, transport, { reason: options.reason || "manual" });
      return { ok: true, transport, state, payload, response: data };
    } catch (e) {
      const error = String((e && e.message) || e);
      console.error("[advisor] /recommend failed:", e);
      renderAdvisorError(error);
      return { ok: false, error, state, payload };
    }
  } finally {
    inFlightRecommend = false;
  }
}

browserAPI.runtime.onMessage.addListener(function (message, sender, sendResponse) {
  if (message && message.action === "capture") {
    console.log("[advisor] capture/recommend triggered");
    triggerRecommend({ force: true, reason: "popup" }).then(sendResponse);
    return true; // keep the message channel open for async response
  }
});

startAutoRefresh();
