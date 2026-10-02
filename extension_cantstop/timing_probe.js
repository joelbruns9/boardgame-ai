// Timing probe (MAIN world): WHEN things happen on the BGA page, so capture
// delays can be sized from measurement instead of guessed.
//
// Records, in one ring buffer with performance.now() times:
//   notif  each packet through gameui.notifqueue.onNotification (the hook
//          extension_7wd verified live): its notification TYPES only
//   state  each gameui.onEnteringState(stateName) call
//   board  each DOM mutation batch touching a board marker or the dice
//   sig    the bridge saw a new capture signature (it polls every 100 ms)
//   sent   the bridge posted a settled position (with fast/slow path)
// Never packet contents or chat text. Hooks call the original first-class and
// swallow their own errors: recording must never break the table.
function createCantStopTiming(w, limit = 3000) {
  const events = [];
  const now = () => (w.performance?.now ? w.performance.now() : Date.now());
  const t0 = now(), wall = Date.now();
  let notifHooked = false, stateHooked = false, observer = null;
  const restore = [];
  function record(kind, detail) {
    events.push({t: Math.round((now() - t0) * 10) / 10, kind, ...detail});
    if (events.length > limit) events.splice(0, events.length - limit);
  }
  function notifTypes(packet) {
    try {
      if (typeof packet === "string") packet = JSON.parse(packet);
      return (packet?.data || []).map(e => String(e?.type || "")).filter(t =>
        !/chat|writing/i.test(t)).slice(0, 12);
    } catch { return []; }
  }
  function install() {
    const ui = w.gameui;
    if (!ui) return false;
    const queue = ui.notifqueue;
    if (!notifHooked && queue && typeof queue.onNotification === "function") {
      const original = queue.onNotification;
      queue.onNotification = function (packet) {
        try {
          const types = notifTypes(packet);
          if (types.length) record("notif", {types});
        } catch {}
        return original.apply(this, arguments);
      };
      restore.push(() => { queue.onNotification = original; });
      notifHooked = true;
    }
    if (!stateHooked && typeof ui.onEnteringState === "function") {
      const original = ui.onEnteringState;
      ui.onEnteringState = function (stateName) {
        try { record("state", {name: String(stateName), active: String(ui.gamedatas?.gamestate?.active_player || "")}); } catch {}
        return original.apply(this, arguments);
      };
      restore.push(() => { ui.onEnteringState = original; });
      stateHooked = true;
    }
    if (!observer && typeof w.MutationObserver === "function" && w.document?.body) {
      observer = new w.MutationObserver(list => {
        try {
          let tokens = 0, dice = 0;
          for (const m of list) {
            const el = m.target;
            const cls = typeof el?.className === "string" ? el.className : "";
            if (/tokenspace|token/.test(cls)) tokens++;
            else if (/dice|die/i.test(cls) || /dice|die/i.test(String(el?.id || ""))) dice++;
          }
          if (tokens || dice) record("board", {tokens, dice});
        } catch {}
      });
      observer.observe(w.document.body, {subtree: true, childList: true, attributes: true,
        attributeFilter: ["class", "data-column", "data-height", "style"]});
      restore.push(() => observer.disconnect());
    }
    return notifHooked && stateHooked;
  }
  return {
    install, record,
    dump: () => ({wall_start: wall, hooks: {notif: notifHooked, state: stateHooked, board: !!observer},
                  events: events.slice()}),
    dispose() { while (restore.length) { try { restore.pop()(); } catch {} } observer = null; },
  };
}
if (typeof module !== "undefined") module.exports = {createCantStopTiming};
