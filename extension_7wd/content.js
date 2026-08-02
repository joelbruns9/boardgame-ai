// Isolated-world half of the extension: networking and UI.
//
// Split of responsibility (see page_bridge.js for the other half):
//   page_bridge.js  MAIN world  -- sees window.gameui, captures, never fetches
//   content.js      isolated    -- fetches the advisor, renders the panel
//
// The fetch lives here because the advisor is http://127.0.0.1 while the page is
// https: a page-context request would be blocked as mixed content, whereas a
// content script with host_permissions is not. The panel is built here too --
// the DOM is shared between worlds, so BGA's own stylesheet applies and card art
// comes free from `.building` + a background-position (no bundled sprites).
//
// Cross-browser: MV3, no background script, no browser-specific APIs. Uses
// `chrome.*` where present and falls back to `browser.*` for older Firefox.

(() => {
  "use strict";

  const TAG = "swd-advisor";
  const api = typeof chrome !== "undefined" && chrome.runtime ? chrome : browser;
  const HOST = "http://127.0.0.1:8000";
  const POLL_MS = 700;
  // Below this the win probability is noise: measured 89% at 600 sims against
  // 98.9% converged on the same position. Moves rank correctly long before the
  // number is trustworthy, so show the list and withhold the percentage.
  const MIN_SIMS_FOR_WIN_PCT = 5000;
  const TOP_N = 5;
  // A row with few visits has a Q that is essentially unsearched.
  const LOW_VISIT_FRAC = 0.02;

  let jobId = null;
  let pollTimer = null;
  let art = null;
  let panel = null;

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
    panel.id = "swd_advisor_panel";
    panel.innerHTML =
      '<div class="swd-adv-head">' +
      '<span class="swd-adv-title">7WD Advisor</span>' +
      '<span class="swd-adv-status" data-role="status">idle</span>' +
      '<span class="swd-adv-drag" data-role="drag" title="drag">::</span>' +
      "</div>" +
      '<div class="swd-adv-sub" data-role="sub"></div>' +
      '<div class="swd-adv-rows" data-role="rows"></div>';
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

  function setStatus(text) {
    ensurePanel().querySelector('[data-role="status"]').textContent = text;
  }

  // Case-insensitive: BGA title-cases some names ("Chamber Of Commerce") where
  // the engine follows the printed card ("Chamber of Commerce").
  function spriteFor(table, name) {
    if (!table || !name) return null;
    const want = String(name).toLowerCase();
    for (const entry of Object.values(table)) {
      if (String(entry.name).toLowerCase() === want) return entry.spriteXY;
    }
    return null;
  }

  function cardArt(cardName) {
    const xy = spriteFor(art && art.buildings, cardName);
    const box = document.createElement("div");
    if (!xy) {
      box.className = "swd-adv-noart";
      return box;
    }
    box.className = "building building_small swd-adv-art";
    box.style.backgroundPosition = `-${xy[0]}00% -${xy[1]}00%`;
    return box;
  }

  // `job` is the /poll envelope: {job_id, status, sims_done, error, snapshot},
  // where `snapshot` is the RecommendResponse and is null until the first chunk
  // has been published.
  function render(job) {
    const el = ensurePanel();
    const rows = el.querySelector('[data-role="rows"]');
    const sub = el.querySelector('[data-role="sub"]');
    const snap = job.snapshot;
    const recs = ((snap && snap.recommendations) || []).slice(0, TOP_N);

    const sims = job.sims_done || (snap && snap.sims_done) || 0;
    const pct = snap ? ((snap.root_value + 1) / 2) * 100 : 0;
    sub.textContent =
      sims.toLocaleString() +
      " sims" +
      (sims >= MIN_SIMS_FOR_WIN_PCT
        ? "  ·  win " + pct.toFixed(1) + "%"
        : "  ·  win % settling…");

    rows.textContent = "";
    for (const r of recs) {
      const row = document.createElement("div");
      row.className = "swd-adv-row";
      if ((r.visit_frac || 0) < LOW_VISIT_FRAC) row.classList.add("swd-adv-thin");
      row.appendChild(cardArt(r.fields && r.fields.card_name));
      const text = document.createElement("div");
      text.className = "swd-adv-text";
      const q = document.createElement("div");
      q.className = "swd-adv-q";
      q.textContent = (r.q_value >= 0 ? "+" : "") + r.q_value.toFixed(3);
      const lab = document.createElement("div");
      lab.className = "swd-adv-label";
      lab.textContent = r.label;
      const meta = document.createElement("div");
      meta.className = "swd-adv-meta";
      meta.textContent = (r.visits || 0).toLocaleString() + " visits";
      text.append(lab, meta);
      row.append(text, q);
      rows.appendChild(row);
    }
    if (!recs.length) {
      rows.textContent = snap ? "no legal moves" : "starting search…";
    }
  }

  // -- advisor --------------------------------------------------------------

  // All network goes through the background script: the page is https and the
  // advisor is http://127.0.0.1, so a fetch from here is blocked as mixed
  // content. See background.js.
  async function call(path, init) {
    const reply = await api.runtime.sendMessage({
      kind: "swd-fetch",
      url: HOST + path,
      init,
    });
    if (!reply) throw Object.assign(new Error("no reply from background"), { status: 0 });
    if (!reply.ok) {
      throw Object.assign(
        new Error(reply.error || path + " -> HTTP " + reply.status),
        { status: reply.status }
      );
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

  async function startSearch(state) {
    await stopCurrent();
    setStatus("searching…");
    let started;
    try {
      started = await post("/api/recommend/start", {
        state,
        // Effectively unbounded: the tree is cumulative and we stop on the next
        // position, so this is "keep thinking until the board changes".
        max_sims: 1000000,
        chunk_sims: 50,
        top_k: TOP_N,
      });
    } catch (err) {
      setStatus(reportFailure(err));
      ensurePanel().querySelector('[data-role="sub"]').textContent =
        (err && err.message) || "";
      return;
    }
    jobId = started.job_id;
    pollTimer = setInterval(poll, POLL_MS);
  }

  async function poll() {
    if (!jobId) return;
    let resp;
    try {
      resp = await call(
        "/api/recommend/poll?job_id=" + encodeURIComponent(jobId),
        { method: "GET" }
      );
    } catch (err) {
      setStatus(reportFailure(err));
      clearInterval(pollTimer);
      pollTimer = null;
      return;
    }
    if (resp.error) {
      setStatus("error");
      render({ snapshot: null, sims_done: 0 });
      ensurePanel().querySelector('[data-role="sub"]').textContent = resp.error;
      clearInterval(pollTimer);
      pollTimer = null;
      return;
    }
    setStatus(resp.status === "done" ? "converged" : "thinking");
    render(resp);
  }

  // -- bridge ---------------------------------------------------------------

  window.addEventListener("message", (event) => {
    if (event.source !== window) return;
    const msg = event.data;
    if (!msg || msg.__swd !== TAG) return;
    if (msg.type === "position") {
      art = msg.payload.art;
      ensurePanel();
      startSearch(msg.payload.state);
    } else if (msg.type === "idle") {
      stopCurrent();
      setStatus("waiting for your turn");
    } else if (msg.type === "capture_error") {
      stopCurrent();
      setStatus("capture failed");
      ensurePanel().querySelector('[data-role="sub"]').textContent =
        msg.payload.message || "see console";
    }
  });

  injectPageScripts();
})();
