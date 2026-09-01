// Isolated-world half of the extension: networking and UI.
//
// Split of responsibility (see page_bridge.js for the other half):
//   page_bridge.js  MAIN world  -- sees window.gameui, captures, never fetches
//   content.js      isolated    -- fetches the advisor, renders the panel
//
// The fetch lives here because the advisor is http://127.0.0.1 while the page is
// https: a page-context request would be blocked as mixed content, whereas a
// content script with host_permissions is not. See background.js, which is where
// the request actually goes out from.
//
// WHAT THIS PANEL IS FOR. Not to play the game -- to show what the model
// believes, so its weaknesses are visible on a real board. That is why it shows
// three things a "best move" overlay would not:
//
//   * the PRIOR beside the visit count, because "the net wanted this and search
//     talked it out of it" and "the net never looked at it" are different
//     faults with different fixes;
//   * the net's own FORECAST -- predicted final score per seat, broken into
//     components, plus what it thinks each plan will do -- because a sane move
//     list on top of a plainly wrong forecast localises the problem instantly;
//   * every warning the host raises, including which auxiliary heads this
//     checkpoint never trained. A head that was never trained returns exactly
//     0.5, which is indistinguishable from a real coin flip unless it is said.
//
// Cross-browser: MV3, uses `chrome.*` where present and falls back to `browser.*`.

(() => {
  "use strict";

  const TAG = "wto-advisor";
  const api = typeof chrome !== "undefined" && chrome.runtime ? chrome : browser;
  // 8001, not 8000: the 7WD advisor owns 8000 and both hosts get left running.
  const HOST = "http://127.0.0.1:8001";
  const POLL_MS = 800;
  const TOP_N = 6;
  // A row with few visits has a Q that is essentially unsearched. Welcome To
  // roots are wide (mean 35 legal macros, up to 331), so this matters more here
  // than in a game with a dozen moves.
  const LOW_VISIT_FRAC = 0.02;
  // Welcome To search is slow: a simulation is a leaf evaluation plus a rollout
  // to the next turn boundary, measured at 29/s on CPU for the 3.94M-parameter
  // S2 net. So the budget is "keep thinking until the board changes" -- 20k is
  // about eleven minutes, well past any turn, and bounded so a forgotten tab
  // does not grow a tree without limit.
  const MAX_SIMS = 20000;
  const CHUNK_SIMS = 8;
  const MAX_RETRIES = 5;
  const RETRY_MS = 1500;

  let jobId = null;
  let pollTimer = null;
  let retryTimer = null;
  let retries = 0;
  let panel = null;
  let lastSignature = null;
  let lastState = null;

  // -- injection ------------------------------------------------------------

  function injectPageScripts() {
    for (const file of ["bga_snippet.js", "page_bridge.js"]) {
      const el = document.createElement("script");
      el.src = api.runtime.getURL(file);
      el.async = false; // preserve order: bga_snippet defines what the bridge calls
      (document.head || document.documentElement).appendChild(el);
      el.addEventListener("load", () => el.remove());
    }
  }

  // -- panel ----------------------------------------------------------------

  function ensurePanel() {
    if (panel && panel.isConnected) return panel;
    panel = document.createElement("div");
    panel.id = "wto_advisor_panel";
    panel.innerHTML =
      '<div class="wto-adv-head">' +
      '<span class="wto-adv-title">Welcome To Advisor</span>' +
      '<span class="wto-adv-status" data-role="status">idle</span>' +
      '<span class="wto-adv-drag" data-role="drag" title="drag">::</span>' +
      "</div>" +
      '<div class="wto-adv-sub" data-role="sub"></div>' +
      '<div class="wto-adv-warn" data-role="warn"></div>' +
      '<div class="wto-adv-rows" data-role="rows"></div>' +
      '<div class="wto-adv-forecast" data-role="forecast"></div>';
    document.body.appendChild(panel);
    makeDraggable(panel, panel.querySelector('[data-role="drag"]'));
    return panel;
  }

  function makeDraggable(node, handle) {
    let dx = 0;
    let dy = 0;
    const move = (e) => {
      node.style.left = e.clientX - dx + "px";
      node.style.top = e.clientY - dy + "px";
      node.style.right = "auto";
    };
    const up = () => {
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
    };
    handle.addEventListener("mousedown", (e) => {
      const r = node.getBoundingClientRect();
      dx = e.clientX - r.left;
      dy = e.clientY - r.top;
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
      e.preventDefault();
    });
  }

  function slot(role) {
    return ensurePanel().querySelector('[data-role="' + role + '"]');
  }

  function setStatus(text) {
    slot("status").textContent = text;
  }

  function setSub(text) {
    slot("sub").textContent = text;
  }

  function renderWarnings(list) {
    const box = slot("warn");
    box.textContent = "";
    for (const text of list || []) {
      const line = document.createElement("div");
      line.className = "wto-adv-warn-line";
      line.textContent = text;
      box.appendChild(line);
    }
  }

  // -- moves ----------------------------------------------------------------

  function render(job) {
    const rows = slot("rows");
    const snap = job.snapshot;
    const recs = ((snap && snap.recommendations) || []).slice(0, TOP_N);
    const sims = job.sims_done || (snap && snap.sims_done) || 0;

    setSub(
      sims.toLocaleString() +
        " sims" +
        (snap ? "  ·  root " + fmtSigned(snap.root_value) : "")
    );

    rows.textContent = "";
    for (const r of recs) {
      const row = document.createElement("div");
      row.className = "wto-adv-row";
      if ((r.visit_frac || 0) < LOW_VISIT_FRAC) row.classList.add("wto-adv-thin");

      const q = document.createElement("div");
      q.className = "wto-adv-q";
      q.textContent = fmtSigned(r.q_value);

      const text = document.createElement("div");
      text.className = "wto-adv-text";
      const label = document.createElement("div");
      label.className = "wto-adv-label";
      label.textContent = r.label;
      const meta = document.createElement("div");
      meta.className = "wto-adv-meta";
      // Visits are what ranks the move; the prior is what the network wanted
      // before any search. Showing both is the point of this panel.
      meta.textContent =
        (r.visits || 0).toLocaleString() +
        " visits  ·  prior " +
        (100 * (r.prior || 0)).toFixed(1) +
        "%";
      text.append(label, meta);

      row.append(q, text);
      rows.appendChild(row);
    }
    if (!recs.length) rows.textContent = snap ? "no legal moves" : "starting search…";
  }

  function fmtSigned(value) {
    const n = Number(value || 0);
    return (n >= 0 ? "+" : "") + n.toFixed(3);
  }

  // -- the net's own read of the position ------------------------------------
  //
  // One raw root evaluation, so it appears immediately and never changes as
  // search deepens. Fetched once per position from /api/state.

  const COMPONENT_ORDER = [
    "plans",
    "estates",
    "parks",
    "pools",
    "temp",
    "bis",
    "permits",
    "roundabouts",
  ];

  function renderForecast(pub) {
    const box = slot("forecast");
    box.textContent = "";
    if (!pub) return;

    const f = pub.forecast;
    const head = document.createElement("div");
    head.className = "wto-adv-fc-head";
    head.textContent =
      "turn " +
      pub.turn +
      "  ·  " +
      pub.phase.toLowerCase().replace(/_/g, " ") +
      "  ·  deck " +
      pub.deck_remaining;
    box.appendChild(head);

    box.appendChild(stacksStrip(pub));

    if (!f) {
      const none = document.createElement("div");
      none.className = "wto-adv-fc-none";
      none.textContent = "no forecast (host has no checkpoint loaded)";
      box.appendChild(none);
      return;
    }

    // Predicted FINAL scores, not current ones. The gap between a seat's
    // forecast and what it actually ends on is the single most legible
    // diagnostic this panel has.
    const best = Math.max(...f.seats.map((s) => s.final_score), 1);
    for (const seat of f.seats) {
      const row = document.createElement("div");
      row.className = "wto-adv-fc-seat" + (seat.you ? " wto-adv-fc-you" : "");

      const bar = document.createElement("span");
      bar.className = "wto-adv-fc-bar";
      bar.style.width = Math.max(2, Math.round((seat.final_score / best) * 100)) + "%";

      const label = document.createElement("span");
      label.className = "wto-adv-fc-label";
      const here = pub.seats[seat.seat] || {};
      label.textContent =
        (seat.you ? "you" : here.name || "seat " + (seat.seat + 1)) +
        "  " +
        seat.final_score.toFixed(0) +
        " pts  (now " +
        (here.score === undefined ? "?" : here.score) +
        ")";

      row.append(bar, label);
      box.appendChild(row);

      if (seat.you) box.appendChild(componentStrip(seat));
    }

    const foot = document.createElement("div");
    foot.className = "wto-adv-fc-foot";
    foot.textContent =
      "finish " +
      f.rank_probs.map((p) => (p * 100).toFixed(0) + "%").join(" / ") +
      "  ·  " +
      f.turns_left.toFixed(1) +
      " turns left";
    box.appendChild(foot);

    // Heads this checkpoint never trained come back as an exact 0.5. Saying so
    // is the difference between a diagnostic and a mirage.
    if (f.untrained_heads && f.untrained_heads.length) {
      const stale = document.createElement("div");
      stale.className = "wto-adv-fc-stale";
      stale.textContent =
        "untrained in this checkpoint: " + f.untrained_heads.join(", ");
      box.appendChild(stale);
    } else {
      box.appendChild(planStrip(pub, f));
    }
  }

  function stacksStrip(pub) {
    const strip = document.createElement("div");
    strip.className = "wto-adv-stacks";
    for (const stack of pub.stacks) {
      const chip = document.createElement("span");
      chip.className = "wto-adv-chip" + (stack.playable ? "" : " wto-adv-chip-dead");
      chip.textContent =
        stack.number + " / " + shortEffect(stack.effect) + " → " + shortEffect(stack.next_effect);
      strip.appendChild(chip);
    }
    return strip;
  }

  // The arrow above is not decoration: a card's number face prints its own
  // effect in the corners, so the effect each stack will offer NEXT turn is
  // already known with certainty. The model is fed it, and a human reading this
  // panel should be too.
  function shortEffect(name) {
    if (!name) return "?";
    return (
      {
        SURVEYOR: "fence",
        ESTATE: "estate",
        PARK: "park",
        POOL: "pool",
        TEMP: "temp",
        BIS: "bis",
      }[name] || name.toLowerCase()
    );
  }

  function componentStrip(seat) {
    const strip = document.createElement("div");
    strip.className = "wto-adv-fc-components";
    for (const key of COMPONENT_ORDER) {
      const value = seat.components[key];
      if (value === undefined) continue;
      const chip = document.createElement("span");
      chip.className = "wto-adv-chip";
      const penalty = key === "bis" || key === "permits" || key === "roundabouts";
      chip.textContent = key + " " + (penalty ? "-" : "") + Math.abs(value).toFixed(0);
      if (penalty && Math.abs(value) >= 1) chip.classList.add("wto-adv-chip-bad");
      strip.appendChild(chip);
    }
    return strip;
  }

  function planStrip(pub, forecast) {
    const strip = document.createElement("div");
    strip.className = "wto-adv-plans";
    const you = forecast.seats.find((s) => s.you) || forecast.seats[0];
    pub.plans.forEach((plan, slotIndex) => {
      const row = document.createElement("div");
      row.className = "wto-adv-plan";
      const done = Object.keys(plan.completed_by || {}).length > 0;
      row.textContent =
        "plan " +
        (slotIndex + 1) +
        " (" +
        plan.name +
        ", " +
        plan.scores.join("/") +
        "): " +
        (100 * you.will_complete_plan[slotIndex]).toFixed(0) +
        "% in ~" +
        you.turns_to_plan[slotIndex].toFixed(1) +
        " turns" +
        (done ? "  · already taken" : "");
      strip.appendChild(row);
    });
    return strip;
  }

  // -- advisor --------------------------------------------------------------

  // All network goes through the background script: the page is https and the
  // advisor is http://127.0.0.1, so a fetch from here is blocked as mixed
  // content. See background.js.
  async function call(path, init) {
    const reply = await api.runtime.sendMessage({
      kind: "wto-fetch",
      url: HOST + path,
      init,
    });
    if (!reply) throw Object.assign(new Error("no reply from background"), { status: 0 });
    if (!reply.ok) {
      // The host puts the real reason in the body -- FastAPI raises
      // HTTPException(400, "bad state: <exception>") when the adapter refuses a
      // position. Surfacing only the status code hides the one useful fact.
      let detail = reply.error || "";
      if (reply.body) {
        try {
          detail = JSON.parse(reply.body).detail || reply.body;
        } catch (e) {
          detail = reply.body;
        }
      }
      throw Object.assign(new Error(detail || path + " -> HTTP " + reply.status), {
        status: reply.status,
      });
    }
    return reply.body ? JSON.parse(reply.body) : null;
  }

  async function post(path, body) {
    return call(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  // "offline" and "the host rejected it" are different problems; say which.
  function reportFailure(err) {
    if (err && err.status === 0) return "host offline";
    if (err && err.status) return "host error " + err.status;
    return "request failed";
  }

  async function stopCurrent() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    if (retryTimer) {
      clearTimeout(retryTimer);
      retryTimer = null;
    }
    if (!jobId) return;
    const dying = jobId;
    jobId = null;
    // Explicit: otherwise the old search keeps burning CPU on a stale board.
    try {
      await post("/api/recommend/stop", { job_id: dying });
    } catch (e) {
      /* host may be gone; nothing to do */
    }
  }

  async function fetchForecast(state) {
    try {
      const pub = await post("/api/state", { state });
      renderForecast(pub);
      // Position-level warnings (an incomplete card ledger, say) ride on the
      // public state rather than the search response, because they are facts
      // about the capture and not about the search.
      if (pub && pub.warnings && pub.warnings.length) renderWarnings(pub.warnings);
    } catch (err) {
      renderForecast(null); // cosmetic; never blocks the search
    }
  }

  function tableId() {
    const match = /[?&]table=(\d+)/.exec(location.href);
    return match ? match[1] : null;
  }

  // Passive capture. The host already receives a faithful position every turn;
  // keeping it costs one request and turns ordinary play into a corpus of real
  // human positions -- both to restart self-play from and to validate the
  // engine against. Never blocks the search.
  async function logPosition(state) {
    try {
      await post("/api/game_log", { table_id: tableId(), state });
    } catch (err) {
      /* logging is best-effort; advice must not depend on it */
    }
  }

  async function startSearch(state) {
    await stopCurrent();
    setStatus("searching…");
    renderWarnings([]);
    fetchForecast(state);
    logPosition(state);
    let started;
    try {
      started = await post("/api/recommend/start", {
        state,
        max_sims: MAX_SIMS,
        chunk_sims: CHUNK_SIMS,
        top_k: TOP_N,
      });
    } catch (err) {
      const detail = (err && err.message) || "";
      // A rejected position is often transient: BGA animates a mark into place,
      // and a capture taken mid-flight disagrees with the replay. Re-ask a few
      // times before giving up.
      if (err && err.status === 400 && retries < MAX_RETRIES) {
        retries += 1;
        setStatus("retrying " + retries + "/" + MAX_RETRIES);
        setSub(detail);
        retryTimer = setTimeout(() => {
          window.postMessage({ __wto: TAG, type: "recapture" }, window.location.origin);
        }, RETRY_MS);
        return;
      }
      setStatus(reportFailure(err));
      setSub(detail);
      return;
    }
    retries = 0;
    jobId = started.job_id;
    pollTimer = setInterval(poll, POLL_MS);
  }

  async function poll() {
    if (!jobId) return;
    let resp;
    try {
      resp = await call("/api/recommend/poll?job_id=" + encodeURIComponent(jobId), {
        method: "GET",
      });
    } catch (err) {
      setStatus(reportFailure(err));
      clearInterval(pollTimer);
      pollTimer = null;
      return;
    }
    if (resp.error) {
      setStatus("error");
      setSub(resp.error);
      clearInterval(pollTimer);
      pollTimer = null;
      return;
    }
    const warnings = (resp.snapshot && resp.snapshot.warnings) || [];
    setStatus(resp.status === "done" ? "converged" : "thinking");
    render(resp);
    if (warnings.length) renderWarnings(warnings);
  }

  // -- bridge ---------------------------------------------------------------

  window.addEventListener("message", (event) => {
    if (event.source !== window) return;
    const msg = event.data;
    if (!msg || msg.__wto !== TAG) return;

    if (msg.type === "position") {
      ensurePanel();
      if (msg.payload.signature !== lastSignature) {
        lastSignature = msg.payload.signature;
        retries = 0; // genuinely new position
      }
      startSearch({
        bga: msg.payload.bga,
        dom: msg.payload.dom,
        seen: msg.payload.seen,
      });
    } else if (msg.type === "idle") {
      lastSignature = null;
      lastState = msg.payload && msg.payload.state;
      stopCurrent();
      ensurePanel();
      setStatus(lastState === "waitOthers" ? "turn done" : "waiting");
      setSub(lastState ? "BGA state: " + lastState : "");
    } else if (msg.type === "capture_error") {
      stopCurrent();
      ensurePanel();
      setStatus("capture failed");
      setSub(msg.payload.message || "see console");
    }
  });

  injectPageScripts();
})();
