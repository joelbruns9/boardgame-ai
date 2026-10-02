// MAIN-world bridge, following extension_7wd's capture/postMessage boundary.
(() => {
  if (window.__cantstopBridge?.dispose) window.__cantstopBridge.dispose();
  const tag = "cantstop-advisor";
  // One probe per page, surviving bridge reloads (see timing_probe.js).
  // A probe that failed to load (or cannot run) must never stop capture.
  const timing = window.__cantstopTiming ||= (typeof createCantStopTiming === "function"
    ? createCantStopTiming(window, 20000) : {install() {}, record() {}, dump: () => null});
  const post = (type, payload) => window.postMessage({advisor: tag, type, payload}, location.origin);
  let pending = null, sent = null, since = 0, validated = null;
  const color = m => String(m.color).toLowerCase().replace(/^#/, "");
  const markers = (state, black) => (state.markers || [])
    .filter(m => (color(m) === "000000") === black)
    .map(m => [m.column, m.height, color(m)]).sort((a,b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
  function base(state) {
    if (!state?.turn_id || !state.markers || !state.playerorder || !state.players) return null;
    return JSON.stringify([state.table_id, state.turn_id, state.active_player,
      state.playerorder, state.playerorder.map(p => state.players[p]?.color),
      state.required_column_count, state.movement_variant_raw, state.blocking, markers(state, false)]);
  }
  const runnerKey = state => JSON.stringify(markers(state, true));
  function fastPosition(state) {
    if (!validated || base(state) !== validated.base) return false;
    if (state.phase === "diceChoice" && (!Array.isArray(state.dice) || state.dice.length !== 4 ||
        state.dice.some(d => !Number.isInteger(d) || d < 1 || d > 6))) return false;
    // Only complete runner configurations predicted by the validated solve
    // qualify. Partial animations, rule/board changes and new turns wait longer.
    return validated.runners.has(runnerKey(state));
  }
  const turnIdentity = createCantStopTurnTracker(Date.now() + "-" + Math.random().toString(36).slice(2));
  function tick() {
    try {
      if (new URL(location.href).searchParams.has("testuser")) return;
      try { timing.install(); } catch {}
      const gd = window.gameui?.gamedatas;
      const turn = turnIdentity({
        table_id: new URL(location.href).searchParams.get("table") || String(gd?.table_id || "unknown"),
        active_player: String(gd?.gamestate?.active_player || ""),
        phase: String(gd?.gamestate?.name || "")
      });
      const state = captureCantStop(window, true);
      if (state) state.turn_id = turn;
      const signature = state ? JSON.stringify(state) : null;
      if (signature !== pending) {
        timing.record("sig", {phase: state?.phase || null, active: state?.active_player || null});
        pending = signature; since = Date.now(); sent = null;
        post("idle", null); // invalidate old advice immediately, before settling
      }
      const fast = !!signature && fastPosition(state);
      if (signature && signature !== sent && Date.now()-since >= (fast ? 200 : 1200)) {
        sent = signature;
        timing.record("sent", {phase: state.phase, active: state.active_player, fast,
          waited: Date.now() - since});
        post("position", {state, signature});
      }
    } catch (e) {
      pending = sent = null;
      post("capture_error", {message: String(e.message || e)});
    }
  }
  function onMessage(e) {
    // Firefox extension-originated postMessage can have source=null.
    // Still require this document's origin and the narrow recapture message.
    if ((e.source !== window && e.source !== null) || e.origin !== location.origin) return;
    if (e.data?.advisor !== tag) return;
    if (e.data.type === "capture_unsettled") validated = null;
    if (e.data.type === "timing_request") post("timing_dump", timing.dump());
    if (e.data.type === "validated_position") {
      const state = e.data.payload?.state;
      // Ignore late responses for a position the page has already left.
      if (JSON.stringify(state) === pending && base(state)) {
        const runners = new Set();
        if (state.phase === "continueChoice") runners.add(runnerKey(state));
        for (const positions of e.data.payload.runners || []) {
          runners.add(runnerKey({markers:Object.entries(positions).map(([c,p]) =>
            ({column:Number(c), height:13-2*Math.abs(7-Number(c))-p, color:"000000"}))}));
        }
        validated = {base:base(state), runners};
      }
    }
    if (e.data.type === "recapture") {
      post("capture_ack", {request_id:e.data.request_id});
      sent = null;
      tick();
      if (pending === null) post("idle", null);
    }
  }
  window.addEventListener("message", onMessage);
  const interval = setInterval(tick, 100);
  window.__cantstopBridge = {dispose() {
    clearInterval(interval);
    window.removeEventListener("message", onMessage);
  }};
  tick();
})();
