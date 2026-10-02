// Shared advisor host client: blocking recommendation + background proxy.
// The transport and lifecycle know only the games.advisor response contract.
(() => {
  const api = typeof browser !== "undefined" ? browser : chrome;
  const TAG = "cantstop-advisor", HOST = "http://127.0.0.1:8765";
  let lastResult = null, lastResultTable = null, lastResultTurn = null;
  // Display option only: opponents' decisions are always evaluated (the win
  // probabilities need their turn solve anyway) and logged; this decides
  // whether their options are shown. Default on.
  let showOpponents = true, serverHealth = null;
  const diagnostics = [];
  let captureRetryTimer = null, captureKey = null, captureRetries = 0;
  let lastCapture = null, bridgeRequest = 0, bridgeTimer = null;
  let panel, current, epoch = 0, status, rows, facts, players, luck;
  // Player win probabilities: their own sequence so a late answer for an old
  // roll never overwrites a newer one, and independent of the advice epoch.
  let winSeq = 0, winShownTurn = null;
  async function requestOnce(path, body) {
    let reply;
    let timer;
    try {
      reply = await Promise.race([api.runtime.sendMessage({kind:"advisor-fetch", url:HOST+path,
      init:body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)}}),
        new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("Advisor response timed out")), 65000); })
      ]);
    } catch (error) {
      throw Object.assign(new Error(String(error.message || error)), {status:0});
    } finally { clearTimeout(timer); }
    if (!reply) throw Object.assign(new Error("Advisor background did not respond"), {status:0});
    let data;
    try { data = JSON.parse(reply.body || "{}"); } catch { throw Object.assign(new Error("Invalid advisor response"), {status:reply.status || 0}); }
    if (!reply.ok) throw Object.assign(new Error(data.detail || reply.error || "Advisor unavailable"), {status:reply.status || 0});
    return data;
  }
  async function call(path, body) {
    const token = epoch;
    return advisorWithRetry(() => requestOnce(path, body), {
      isCurrent: () => token === epoch,
      onRetry: attempt => {
        if (status && path !== "/api/game_log") status.textContent = "Connection interrupted · retrying " + attempt + "/2…";
      },
      onError: (error, attempt) => {
        diagnostics.push({time:new Date().toISOString(), path, attempt,
          status:error.status ?? null, message:String(error.message || error)});
        if (diagnostics.length > 30) diagnostics.shift();
        console.warn("[Can't Stop advisor]", path, error.status, error.message);
      }
    });
  }
  function stopOld() {
    clearTimeout(captureRetryTimer); captureRetryTimer = null;
    epoch++;
    return epoch;
  }
  function recapture(recovered = false) {
    const request_id = ++bridgeRequest;
    clearTimeout(bridgeTimer);
    if (status) status.textContent = "Requesting a fresh board capture…";
    bridgeTimer = setTimeout(async () => {
      if (request_id !== bridgeRequest) return;
      diagnostics.push({time:new Date().toISOString(), path:"page_bridge",
        message:"No acknowledgement from page bridge"});
      if (recovered) {
        status.textContent = "Page bridge did not respond. Export diagnostics below; reload BGA to reconnect.";
        return;
      }
      try {
        status.textContent = "Reconnecting to the BGA page…";
        await loadBridge();
        if (request_id === bridgeRequest) recapture(true);
      } catch (e) { status.textContent = "Bridge recovery failed: " + e.message; }
    }, 2500);
    window.postMessage({advisor:TAG, type:"recapture", request_id}, location.origin);
  }
  function ensurePanel() {
    if (panel) return;
    panel = document.createElement("section"); panel.id = "cantstop-advisor-panel";
    panel.innerHTML = '<header>Can’t Stop Advisor</header><div data-role="players"></div><div data-role="luck"></div><div data-role="facts"></div>' +
      '<label><input type="checkbox" data-role="opponents"> Show opponents’ decisions</label>' +
      '<p data-role="status"></p><div data-role="rows"></div><footer><button data-action="retry">Refresh</button> <button data-action="export">Export capture</button></footer>';
    document.body.appendChild(panel);
    const opponents = panel.querySelector('[data-role="opponents"]');
    opponents.checked = showOpponents;
    opponents.onchange = () => {
      showOpponents = opponents.checked;
      api.storage.local.set({cantstopShowOpponentDecisions:showOpponents}).catch(()=>{});
      stopOld(); recapture();
    };
    status = panel.querySelector('[data-role="status"]');
    rows = panel.querySelector('[data-role="rows"]'); facts = panel.querySelector('[data-role="facts"]');
    players = panel.querySelector('[data-role="players"]');
    luck = panel.querySelector('[data-role="luck"]');
    panel.querySelector('[data-action="retry"]').onclick = () => { serverHealth = null; stopOld(); recapture(); };
    panel.querySelector('[data-action="export"]').onclick = async () => {
      // Page timing (timing_probe.js) lives in the page; ask the bridge for it.
      const timing = await new Promise(resolve => {
        const done = t => { window.removeEventListener("message", onDump); clearTimeout(timer); resolve(t); };
        const onDump = e => {
          if (e.source === window && e.data?.advisor === TAG && e.data.type === "timing_dump") done(e.data.payload);
        };
        const timer = setTimeout(() => done(null), 1000);
        window.addEventListener("message", onDump);
        window.postMessage({advisor:TAG, type:"timing_request"}, location.origin);
      });
      const report = JSON.stringify({...(current || lastCapture || {}),
        advisor_diagnostics:diagnostics, advisor_status:status.textContent,
        capture_is_current:!!current, page_timing:timing}, null, 2);
      let text = panel.querySelector("textarea");
      if (!text) {
        text = document.createElement("textarea");
        text.readOnly = true; text.rows = 8; text.style.width = "100%";
        text.setAttribute("aria-label", "Diagnostic report: copy if download does not appear");
        panel.appendChild(text);
      }
      text.value = report;
      const blob = new Blob([report], {type:"application/json"});
      const url = URL.createObjectURL(blob), a = document.createElement("a");
      a.href = url; a.download = "cantstop-bga-capture.json";
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
    };
  }
  function render(snap, selectedState = null) {
    rows.textContent = "";
    const groups = new Map();
    for (const r of snap.recommendations || []) {
      const key = r.kind === "move" ? r.fields.columns.join("+") : "decision";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(r);
    }
    for (const group of groups.values()) {
      const first = group[0], row = document.createElement("div");
      row.className = "advisor-move";
      if (first.kind === "move") {
        const title = document.createElement("strong");
        const selected = selectedState && sameAdvisorState(first.fields.after_move, selectedState);
        title.textContent = first.label + (selected ? " · selected" : "");
        if (selected) row.classList.add("advisor-selected");
        row.appendChild(title);
      }
      for (const decision of ["stop", "roll"].sort((a, b) => {
        const value = d => group.find(r => (r.fields.decision || r.kind) === d)?.q_value ?? -Infinity;
        return value(b) - value(a);
      })) {
        const option = group.find(r => (r.fields.decision || r.kind) === decision);
        if (decision === "roll" && first.fields.wins_game) continue;
        const line = document.createElement("div");
        if (option) {
          line.textContent = (decision === "stop" ? (option.fields.wins_game ? "Stop rolling and win" : "Stop and bank") : "Roll again") +
            " — " + ((option.q_value+1)*50).toFixed(1) + "%";
          if (option.rank === 1) line.textContent += " · best";
        } else {
          line.textContent = decision === "stop" ? "Stop — blocked" : "Roll — unavailable";
        }
        row.appendChild(line);
      }
      rows.appendChild(row);
    }
    status.textContent = "Estimated chance for the active player to win · " + snap.search_ms + " ms";
  }
  // Starting player first: BGA seat number when every player has one,
  // otherwise the captured order.
  function turnOrder(raw) {
    const order = [...raw.playerorder];
    if (order.every(p => Number(raw.players[p]?.no) > 0))
      order.sort((a, b) => raw.players[a].no - raw.players[b].no);
    return order;
  }
  function renderPlayers(raw, result) {
    players.textContent = "";
    for (const p of turnOrder(raw)) {
      const seat = result.player_ids.indexOf(String(p));
      const row = document.createElement("div");
      row.className = "advisor-player" + (p === raw.active_player ? " advisor-player-active" : "");
      const swatch = document.createElement("span");
      swatch.className = "advisor-swatch";
      swatch.style.background = "#" + String(raw.players[p]?.color || "888").replace(/^#/, "");
      const name = document.createElement("span");
      name.className = "advisor-player-name";
      name.textContent = (raw.players[p]?.name || p) + (p === raw.viewer_player ? " (you)" : "");
      const pct = document.createElement("span");
      pct.className = "advisor-player-pct";
      pct.textContent = seat < 0 ? "—" : (100 * result.seats[seat]).toFixed(1) + "%";
      row.append(swatch); row.append(name); row.append(pct);
      players.appendChild(row);
    }
    players.title = "Win chances " + (result.basis || "with best play assumed") +
      " for everyone. Updates on each roll.";
  }
  // Every player's chances: refreshed on each roll (dice showing). A
  // stop/roll decision keeps the roll's numbers, which already assumed the
  // best choice -- unless nothing is shown yet for this turn.
  async function updateWinProbs(raw) {
    const turn = raw.table_id + "|" + raw.turn_id;
    if (raw.phase !== "diceChoice" && winShownTurn === turn) return;
    const seq = ++winSeq;
    try {
      const result = await requestOnce("/api/cantstop/win_probabilities", {state:raw,
        options:{table_id:raw.table_id, turn_id:raw.turn_id}, device:"cuda"});
      if (seq !== winSeq) return;
      winShownTurn = turn;
      renderPlayers(raw, result);
    } catch (e) {
      if (seq !== winSeq) return;
      const why = e.status === 404
        ? "the advisor server is an older version without this feature; restart it from the updated folder"
        : String(e.message || e);
      players.title = "Win chances not updated: " + why;
      // Nothing shown yet: say so in the panel, not only in a tooltip.
      if (!players.children.length) players.textContent = "Win chances unavailable — " + why;
    }
  }
  function opponentHidden(raw) {
    return !showOpponents && raw.viewer_player && raw.active_player !== raw.viewer_player;
  }
  // Render a result unless the display option hides this opponent decision.
  function show(raw, result, selectedState = null) {
    if (opponentHidden(raw)) {
      rows.textContent = "";
      status.textContent = "Opponent’s decision evaluated and logged · hidden (Show opponents’ decisions is off)";
      return false;
    }
    render(result, selectedState);
    return true;
  }
  function confirmPosition(raw, result) {
    window.postMessage({advisor:TAG, type:"validated_position", payload:{state:raw,
      runners:(result.recommendations || []).filter(r => r.fields?.after_move)
        .map(r => r.fields.after_move.runners)}}, location.origin);
  }
  async function recommend(raw) {
    ensurePanel();
    current = raw; lastCapture = raw;
    const actor = raw.players[raw.active_player]?.name || raw.active_player;
    const key = JSON.stringify(raw);
    if (key !== captureKey) { captureKey = key; captureRetries = 0; }
    const token = stopOld();
    facts.textContent = actor + (raw.active_player === raw.viewer_player ? " (you)" : " (opponent)") + " · " + raw.playerorder.length + " players · " + raw.required_column_count +
      " columns · " + (raw.blocking ? "Blocking" : "No blocking") + " (from BGA)";
    const state = raw;
    status.textContent = "Checking position…";
    try {
      const health = serverHealth || await call("/health");
      if (token !== epoch) return;
      if (health.game_id !== "cantstop") throw new Error("This port is serving a different game");
      serverHealth = health;
      // /api/recommend already validates raw BGA captures. Only normalize
      // separately when matching a previously displayed post-selection result.
      const canReuse = raw.phase === "continueChoice" && lastResult &&
        lastResultTable === raw.table_id && lastResultTurn === raw.turn_id;
      const normalized = canReuse ? await call("/api/state", {state}) : state;
      if (token !== epoch) return;
      const cached = lastResultTurn === raw.turn_id && findAdvisorContinuation(lastResult, lastResultTable, raw.table_id, normalized);
      if (cached) {
        confirmPosition(raw, lastResult);
        if (show(raw, lastResult, normalized))
          status.textContent = "Dice selection marked above · all options retained from the same solve";
        return;
      }
      status.textContent = "Solving current turn…";
      const result = await call("/api/recommend", {state:normalized, engine:"auto",
        max_sims:1, chunk_sims:1, top_k:200, device:"cuda",
        options:{table_id:raw.table_id, turn_id:raw.turn_id}});
      if (token !== epoch) return;
      if (!result.ok) throw new Error(result.error || "Advisor search failed");
      lastResult = result; lastResultTable = raw.table_id; lastResultTurn = raw.turn_id;
      confirmPosition(raw, result);
      show(raw, result);
      call("/api/game_log", {table_id:raw.table_id, state:normalized,
        extra:{capture:state, recommendation:result, contract:health.contract}}).catch(()=>{});

    } catch (e) {
      if (token !== epoch || e.cancelled) return;
      serverHealth = null;
      window.postMessage({advisor:TAG, type:"capture_unsettled"}, location.origin);
      rows.textContent = "";
      const unsettled = e.status === 400 && /settle|animation|stale|score\/claims|duplicate marker/i.test(e.message);
      if (unsettled && captureRetries < 2) {
        captureRetries++;
        status.textContent = "Board is still updating · recapturing…";
        captureRetryTimer = setTimeout(() => { if (token === epoch) recapture(); }, 1500);
      } else {
        status.textContent = String(e.message || e) + " · Use Refresh in this panel to retry";
      }
    }
  }
  // Dice luck so far (games/cantstop/live_luck.py), refreshed after each
  // logged packet batch: one request in flight, one more queued if packets
  // arrived meanwhile, so a burst of rolls costs at most two requests.
  let luckBusy = false, luckAgain = false;
  async function updateLuck() {
    if (!packetTable) return;
    if (luckBusy) { luckAgain = true; return; }
    luckBusy = true;
    try {
      const result = await requestOnce("/api/cantstop/luck", {table_id: packetTable, device: "cuda"});
      ensurePanel();
      renderLuck(result);
    } catch (e) {
      ensurePanel();
      luck.textContent = "";
      luck.title = "Dice luck not updated: " + (e.status === 404
        ? "the advisor server is an older version; restart it from the updated folder"
        : String(e.message || e));
    } finally {
      luckBusy = false;
      if (luckAgain) { luckAgain = false; updateLuck(); }
    }
  }
  function renderLuck(result) {
    luck.textContent = "";
    if (!result.available) {
      luck.title = "";
      const note = document.createElement("div");
      note.className = "advisor-luck-note";
      note.textContent = "Dice luck: " + result.reason;
      luck.appendChild(note);
      return;
    }
    const head = document.createElement("div");
    head.className = "advisor-luck-head";
    head.textContent = "Dice luck so far (own rolls)";
    luck.appendChild(head);
    luck.title = "Each player's OWN rolls only. Points: win chance their dice gained (+) or " +
      "lost (−), each roll vs the average over every possible roll. Busts: actual vs expected " +
      "from the exact bust odds of each roll (no model). Progress: the most progress their dice " +
      "offered, in columns (a step = 1/height of its column), vs average dice, less bust losses " +
      "beyond their expected size (no model).";
    const raw = lastCapture;
    const byId = new Map(result.players.map(p => [String(p.player_id), p]));
    const order = raw?.playerorder && raw?.players ? turnOrder(raw).map(String) : [...byId.keys()];
    for (const pid of order) {
      const p = byId.get(pid);
      if (!p) continue;
      const row = document.createElement("div");
      row.className = "advisor-player advisor-luck-row";
      const swatch = document.createElement("span");
      swatch.className = "advisor-swatch";
      const color = raw?.players?.[pid]?.color;
      if (color) swatch.style.background = "#" + String(color).replace(/^#/, "");
      const name = document.createElement("span");
      name.className = "advisor-player-name";
      name.textContent = (raw?.players?.[pid]?.name || p.name) + (raw && pid === raw.viewer_player ? " (you)" : "");
      const signed = (x, digits) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(digits);
      const pts = document.createElement("span");
      pts.className = "advisor-player-pct";
      pts.textContent = signed(p.own_rolls_pts, 1) + " pts";
      const busts = document.createElement("span");
      busts.className = "advisor-luck-busts";
      busts.textContent = p.busts + " busts / " + p.busts_expected.toFixed(1) + " exp";
      row.append(swatch); row.append(name); row.append(pts); row.append(busts);
      luck.appendChild(row);
      const progress = document.createElement("div");
      progress.className = "advisor-luck-progress";
      progress.textContent = "progress " + signed(p.progress_cols, 1) + " columns vs average dice";
      luck.appendChild(progress);
    }
  }
  // BGA's packets go to the game log as "bga_packets" rows. Undelivered ones
  // stay queued and are retried on a timer: the batch most likely to fail is
  // the game's last (host closed), and no later drain would carry it.
  let pendingPackets = [], postingPackets = false, packetTable = null;
  async function logPackets(table, packets) {
    if (table) packetTable = table;
    pendingPackets = pendingPackets.concat(packets);
    if (postingPackets || !pendingPackets.length) return;
    postingPackets = true;
    const batch = pendingPackets;
    try {
      await requestOnce("/api/game_log", {table_id: packetTable || batch[0]?.table_id,
        kind: "bga_packets", extra: {packets: batch}});
      pendingPackets = pendingPackets.slice(batch.length);
      updateLuck();
    } catch {
      setTimeout(() => logPackets(null, []), 5000);
    } finally {
      postingPackets = false;
    }
  }
  window.addEventListener("message", e => {
    if (e.source !== window || e.origin !== location.origin || e.data?.advisor !== TAG) return;
    const {type, payload} = e.data;
    if (type === "capture_ack" && payload?.request_id === bridgeRequest) {
      clearTimeout(bridgeTimer); bridgeTimer = null;
    }
    if (type === "packets") logPackets(payload?.table_id, payload?.packets || []);
    if (type === "position") {
      ensurePanel();
      updateWinProbs(payload.state);
      recommend(payload.state);
    }
    if (type === "idle") { stopOld(); current = null; if (status) status.textContent = "Previous options shown · waiting for a settled decision…"; }
    if (type === "capture_error") { stopOld(); ensurePanel(); status.textContent = payload.message; }
  });
  // External extension assets avoid the legacy inline-script injection.
  async function loadBridge() {
    for (const file of ["bga_snippet.js", "turn_identity.js", "timing_probe.js", "packet_recorder.js", "page_bridge.js"]) {
      await new Promise((resolve, reject) => {
        const el = document.createElement("script"); el.src = api.runtime.getURL(file);
        const timeout = setTimeout(() => { el.remove(); reject(new Error("Page bridge load timed out")); }, 5000);
        el.onload = () => { clearTimeout(timeout); el.remove(); resolve(); };
        el.onerror = () => { clearTimeout(timeout); el.remove(); reject(new Error("Could not load page bridge")); };
        (document.head || document.documentElement).appendChild(el);
      });
    }
  }
  (async () => {
    try {
      const stored = await api.storage.local.get("cantstopShowOpponentDecisions");
      showOpponents = stored.cantstopShowOpponentDecisions !== false;
    } catch {}
    await loadBridge();
  })().catch(e => { ensurePanel(); status.textContent = e.message; });
})();
