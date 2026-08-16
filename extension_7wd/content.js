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
  let retryTimer = null;
  let retries = 0;
  // A rejected position is often transient rather than permanent: BGA animates a
  // card into a player's area, and a capture taken mid-flight can show fewer
  // buildings than playersSituation already reports, which _assert_fresh
  // correctly calls stale. Re-ask a few times before giving up.
  const MAX_RETRIES = 6;
  const RETRY_MS = 1500;
  let art = null;
  let panel = null;
  let lastSignature = null;

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
      '<div class="swd-adv-outlook" data-role="outlook"></div>' +
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

  // Which identity a row's art comes from, in priority order: [field on
  // ActionView.fields, captured sprite table, BGA's own class pair]. All three
  // spritesheets are already loaded by the page and all three templates
  // position them the same way (`-X00% -Y00%`, see jstpl_wonder /
  // jstpl_progress_token / jstpl_wonder_age_card), so nothing is bundled and no
  // geometry is hard-coded.
  //
  // `choice` is tried against both tables because a pending choice names a
  // token (science pair, Great Library) OR a card (Mausoleum revival, a
  // Zeus/Circus destroy target). A miss just falls through to the next entry.
  const ART_SOURCES = [
    ["card_name", "buildings", "building building_small swd-adv-card"],
    ["wonder_name", "wonders", "wonder wonder_small swd-adv-wonder"],
    ["choice", "progressTokens", "progress_token progress_token_small swd-adv-token"],
    ["choice", "buildings", "building building_small swd-adv-card"],
  ];

  function rowArt(fields) {
    const box = document.createElement("div");
    for (const [field, table, className] of ART_SOURCES) {
      const xy = spriteFor(art && art[table], fields && fields[field]);
      if (!xy) continue;
      box.className = className + " swd-adv-art";
      box.style.backgroundPosition = `-${xy[0]}00% -${xy[1]}00%`;
      return box;
    }
    // Genuinely artless: the start-of-Age player choice, which names no
    // component at all.
    box.className = "swd-adv-noart";
    return box;
  }

  // `job` is the /poll envelope: {job_id, status, sims_done, error, snapshot},
  // where `snapshot` is the RecommendResponse and is null until the first chunk
  // has been published.
  function render(job, warnings) {
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
    // Anything the host warns about belongs on screen. Today that is the arena
    // ceiling; a silent cap looks like a counter that stalled.
    for (const warning of warnings || []) {
      sub.textContent += "  ·  " + warning;
    }

    rows.textContent = "";
    for (const r of recs) {
      const row = document.createElement("div");
      row.className = "swd-adv-row";
      if ((r.visit_frac || 0) < LOW_VISIT_FRAC) row.classList.add("swd-adv-thin");
      row.appendChild(rowArt(r.fields));
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
      text.append(lab);
      // The rest of the move, when making it forces a second decision: which
      // card the Mausoleum revives, which building Zeus destroys, which token a
      // science pair takes. Usually most of the move's value, and the label
      // alone cannot say it.
      if (r.follow_up) {
        const followUp = document.createElement("div");
        followUp.className = "swd-adv-follow";
        followUp.textContent = r.follow_up;
        text.append(followUp);
      }
      text.append(meta);
      row.append(text, q);
      rows.appendChild(row);
    }
    if (!recs.length) {
      rows.textContent = snap ? "no legal moves" : "starting search…";
    }
  }

  // How the net expects the game to END, not just who wins. For 7WD that is
  // usually the more actionable fact: a position can be losing on points and
  // still be a 93% scientific win, because the game finishes before scoring.
  // One root evaluation, so it is available immediately and never changes as
  // search deepens -- rendered once per position, not per poll.
  const OUTLOOK_LABELS = {
    you_civilian: "you · civilian",
    you_scientific: "you · science",
    you_military: "you · military",
    opponent_civilian: "opp · civilian",
    opponent_scientific: "opp · science",
    opponent_military: "opp · military",
    draw: "draw",
  };

  function renderOutlook(outlook) {
    const box = ensurePanel().querySelector('[data-role="outlook"]');
    box.textContent = "";
    if (!outlook) return;
    const rows = Object.entries(outlook.victory_type || {})
      .sort((a, b) => b[1] - a[1])
      .filter(([, p]) => p >= 0.01)
      .slice(0, 3);
    for (const [key, p] of rows) {
      const row = document.createElement("div");
      row.className = "swd-adv-vt";
      const bar = document.createElement("span");
      bar.className = "swd-adv-bar";
      bar.style.width = Math.max(2, Math.round(p * 100)) + "%";
      if (key.startsWith("opponent")) bar.classList.add("swd-adv-bar-opp");
      const txt = document.createElement("span");
      txt.className = "swd-adv-vt-label";
      txt.textContent = (OUTLOOK_LABELS[key] || key) + "  " + (p * 100).toFixed(0) + "%";
      row.append(bar, txt);
      box.appendChild(row);
    }
    const foot = document.createElement("div");
    foot.className = "swd-adv-vt-foot";
    const margin = outlook.vp_margin;
    foot.textContent =
      "VP margin " +
      (margin >= 0 ? "+" : "") +
      margin.toFixed(1) +
      "  ·  science " +
      (outlook.final_science || [0, 0]).map((x) => (x * 6).toFixed(1)).join(" / ") +
      " symbols";
    box.appendChild(foot);
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
      // The host puts the real reason in the body -- FastAPI raises
      // HTTPException(400, "bad state: <exception>") when the adapter refuses a
      // position. Surfacing only the status code hides the one useful fact.
      let detail = reply.error || "";
      if (reply.body) {
        try {
          const parsed = JSON.parse(reply.body);
          detail = parsed.detail || reply.body;
        } catch (e) {
          detail = reply.body;
        }
      }
      throw Object.assign(
        new Error(detail || path + " -> HTTP " + reply.status),
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

  async function fetchOutlook(state) {
    try {
      const pub = await post("/api/state", { state });
      renderOutlook(pub && pub.victory_outlook);
    } catch (err) {
      renderOutlook(null); // cosmetic; never blocks the search
    }
  }

  // BGA puts the table id in the URL of the game frame; it is the only handle
  // that ties logged positions back to a replayable game.
  function tableId() {
    const match = /[?&]table=(\d+)/.exec(location.href);
    return match ? match[1] : null;
  }

  // Passive capture. The host already receives a faithful position every turn;
  // keeping it costs one request and turns ordinary play into both a self-play
  // restart corpus and an engine-validation corpus. Never blocks the search,
  // and the host dedupes re-posts of the same position.
  async function logPosition(state) {
    try {
      await post("/api/game_log", { table_id: tableId(), state });
    } catch (err) {
      /* logging is best-effort; advice must not depend on it */
    }
  }

  // BGA's own notification packets: the only record of what BGA *charged* for
  // each move, and so the oracle the differential harness compares our engine
  // against (games/seven_wonders_duel/bga_differential.py).
  //
  // Undelivered packets are kept and retried with the next batch rather than
  // dropped: the harness skips any game whose move sequence has holes, so
  // losing one batch to a host that was briefly down would cost the whole game
  // rather than one move.
  let pendingPackets = [];
  let postingPackets = false;
  const PACKET_RETRY_MS = 5000;

  async function logPackets(packets) {
    pendingPackets = pendingPackets.concat(packets);
    if (postingPackets || !pendingPackets.length) return;
    postingPackets = true;
    const batch = pendingPackets;
    try {
      await post("/api/game_log", {
        table_id: tableId(),
        kind: "bga_packets",
        extra: { packets: batch },
      });
      // Only what was actually sent is cleared; anything that arrived while the
      // request was in flight stays queued.
      pendingPackets = pendingPackets.slice(batch.length);
    } catch (err) {
      // Retry on a timer rather than waiting for the next drain: the batch most
      // likely to fail is the last one of the game (the host is shut down, the
      // game is over), and nothing would ever arrive to carry it.
      setTimeout(() => logPackets([]), PACKET_RETRY_MS);
    } finally {
      postingPackets = false;
    }
  }

  async function startSearch(state) {
    await stopCurrent();
    setStatus("searching…");
    fetchOutlook(state);
    logPosition(state);
    let started;
    try {
      started = await post("/api/recommend/start", {
        state,
        // "Keep thinking until the board changes" -- but bounded, because the
        // tree is cumulative in MEMORY too: on a wide root it allocates a node
        // per simulation, each holding a cloned game state (~4.4 KB). At a
        // million this reached 4+ GB and froze the machine. The advisor also
        // applies its own 512 MB arena budget and will report stopping early;
        // this is the second bound, in the unit the panel controls.
        //
        // Nothing is lost: on that position the root value had converged by
        // ~15k sims and the ranking was stable at every depth.
        max_sims: 200000,
        chunk_sims: 50,
        top_k: TOP_N,
      });
    } catch (err) {
      const detail = (err && err.message) || "";
      if (err && err.status === 400 && retries < MAX_RETRIES) {
        retries += 1;
        setStatus("retrying " + retries + "/" + MAX_RETRIES);
        ensurePanel().querySelector('[data-role="sub"]').textContent = detail;
        retryTimer = setTimeout(() => {
          // Ask the page world for a fresh capture of the same position.
          window.postMessage({ __swd: TAG, type: "recapture" }, window.location.origin);
        }, RETRY_MS);
        return;
      }
      setStatus(reportFailure(err));
      ensurePanel().querySelector('[data-role="sub"]').textContent = detail;
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
    // "done" covers two different endings and they must not read alike. A
    // search that hit the arena ceiling stopped because it ran out of memory
    // budget, not because it converged -- calling that "converged" is exactly
    // the misleading status the ceiling was added to avoid.
    const warnings = (resp.snapshot && resp.snapshot.warnings) || [];
    if (resp.status === "done") {
      setStatus(warnings.length ? "capped" : "converged");
    } else {
      setStatus("thinking");
    }
    render(resp, warnings);
  }

  // -- bridge ---------------------------------------------------------------

  window.addEventListener("message", (event) => {
    if (event.source !== window) return;
    const msg = event.data;
    if (!msg || msg.__swd !== TAG) return;
    if (msg.type === "packets") {
      logPackets(msg.payload.packets || []);
    } else if (msg.type === "position") {
      art = msg.payload.art;
      ensurePanel();
      if (msg.payload.signature !== lastSignature) {
        lastSignature = msg.payload.signature;
        retries = 0; // genuinely new position
      }
      startSearch(msg.payload.state);
    } else if (msg.type === "idle") {
      lastSignature = null;
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
