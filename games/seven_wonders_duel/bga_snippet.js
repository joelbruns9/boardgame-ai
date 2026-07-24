// BGA -> advisor: browser-side capture for the 7 Wonders Duel table page.
//
// This is the ONLY code that runs in the page, and it deliberately holds no
// game knowledge: it grabs `window.gameui.gamedatas` verbatim and hands it off.
// All mapping to the advisor wire lives in Python (bga_extract.wire_from_bga),
// so a BGA UI change can never silently corrupt a position -- worst case the
// Python mapper raises.
//
// FRESHNESS -- READ THIS. `gameui.gamedatas` is the *page-load* payload. BGA
// patches some fields from its notification stream (scores, wonders) but leaves
// others stale mid-game: playerBuildings, discardedBuildings, militaryTrack, and
// progressTokensSituation keep their load-time values until the next full load.
// So capture on a FRESHLY (RE)LOADED table. `wire_from_bga` cross-checks science
// counts and raises StaleGamedata if it sees a stale snapshot, but the browser
// side is responsible for triggering the reload. `captureAfterReload` does that;
// prefer it for any table that has been open a while.
//
// Usage from a content script / userscript / devtools:
//   const raw = captureBgaGamedatas();           // -> plain object, or throws
//   fetch("http://localhost:8000/api/recommend", {   // host wraps wire_from_bga
//     method: "POST",
//     headers: { "Content-Type": "application/json" },
//     body: JSON.stringify({ state: { bga: raw } }),
//   });
//
// The BGA game lives in a nested iframe; from the top window, hop to the frame
// that actually owns `gameui`.

function findGameWindow() {
  // Same-origin frames only; BGA's game iframe shares the origin.
  const stack = [window.top];
  while (stack.length) {
    const w = stack.pop();
    try {
      if (w.gameui && w.gameui.gamedatas && w.gameui.gamedatas.draftpool) return w;
      for (let i = 0; i < w.frames.length; i++) stack.push(w.frames[i]);
    } catch (e) {
      /* cross-origin frame: skip */
    }
  }
  throw new Error("7WD gamedatas not found; open the game table first");
}

function captureBgaGamedatas() {
  const g = findGameWindow().gameui.gamedatas;
  // Deep clone so we detach from the live object before serializing.
  return JSON.parse(JSON.stringify(g));
}

// Optional convenience: only the fields wire_from_bga reads, to keep payloads
// small. wire_from_bga also accepts the full gamedatas, so this is just a diet.
function captureBgaSlim() {
  const g = findGameWindow().gameui.gamedatas;
  const pick = (obj, keys) =>
    Object.fromEntries(keys.filter((k) => k in obj).map((k) => [k, obj[k]]));
  const idName = (tbl) =>
    Object.fromEntries(Object.entries(tbl).map(([k, v]) => [k, { name: v.name }]));
  return {
    startPlayerId: g.startPlayerId,
    players: Object.fromEntries(
      Object.entries(g.players).map(([k, v]) => [k, { name: v.name }])
    ),
    gamestate: pick(g.gamestate, ["name", "active_player"]),
    playersSituation: g.playersSituation,
    militaryTrack: g.militaryTrack,
    draftpool: g.draftpool,
    playerBuildings: g.playerBuildings,
    discardedBuildings: g.discardedBuildings,
    wondersSituation: g.wondersSituation,
    progressTokensSituation: g.progressTokensSituation,
    agora: g.agora,
    pantheon: g.pantheon,
    buildings: idName(g.buildings),
    wonders: idName(g.wonders),
    progressTokens: g.progressTokens ? idName(g.progressTokens) : {},
  };
}

// Reload the game frame, wait for gamedatas to come back fresh, then capture.
// Use this when the table may have been open across several moves.
async function captureAfterReload() {
  const w = findGameWindow();
  const before = w.gameui && w.gameui.gamedatas;
  w.location.reload();
  // Poll for a rebuilt gameui.gamedatas (new object identity) on the frame.
  for (let i = 0; i < 100; i++) {
    await new Promise((r) => setTimeout(r, 200));
    try {
      const g = findGameWindow();
      if (g.gameui && g.gameui.gamedatas && g.gameui.gamedatas !== before) {
        return JSON.parse(JSON.stringify(g.gameui.gamedatas));
      }
    } catch (e) {
      /* frame mid-reload */
    }
  }
  throw new Error("gamedatas did not refresh after reload");
}

if (typeof module !== "undefined") {
  module.exports = {
    findGameWindow,
    captureBgaGamedatas,
    captureBgaSlim,
    captureAfterReload,
  };
}
