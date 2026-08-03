// MAIN-world half of the extension. Runs *in the page*, so it can see
// `window.gameui`; a content script cannot (isolated world, and
// `wrappedJSObject` is Firefox-only).
//
// It does two things and nothing else:
//   1. watches for "it is my turn, in a state the advisor understands"
//   2. on demand, captures the board and postMessage()s it to the content script
//
// It deliberately does NOT fetch. The advisor runs on http://127.0.0.1 while the
// page is https, and a page-context fetch to a plain-http origin is mixed
// content. The isolated content script has host_permissions and is not subject
// to that, so all network access lives there.
//
// Capture functions come from bga_snippet.js, loaded into the page immediately
// before this file. That file is the single source of truth for how a 7WD board
// is read; see test_extension_assets.py, which fails if the copy here drifts.

(() => {
  "use strict";

  const TAG = "swd-advisor";

  // States where the advisor has something to say. Mirrors _DRAFT_STATE,
  // _MAIN_TURN_STATE, _START_PLAYER_STATE and _PENDING_STATES in
  // bga_extract.py; anything else is skipped rather than sent, so the Python
  // side never has to reject a routine transition. BGA auto-picks the last
  // wonder of each round (SelectWonderTrait.php:31-41), so only 6 of the 8
  // picks ever reach us. selectStartPlayer reaches us twice a game.
  const ADVISABLE = new Set([
    "selectWonder",
    "playerTurn",
    "selectStartPlayer",
    "chooseProgressToken",
    "chooseOpponentBuilding",
    "chooseDiscardedBuilding",
    "chooseProgressTokenFromBox",
  ]);

  const post = (type, payload) =>
    window.postMessage({ __swd: TAG, type, payload }, window.location.origin);

  function gameWindow() {
    // findGameWindow() walks from window.top and skips `testuser=` frames, so
    // every copy of this bridge (it runs in each frame) resolves to the same
    // one. Only the frame that IS that window proceeds, so we act once.
    try {
      return typeof findGameWindow === "function" ? findGameWindow() : null;
    } catch (e) {
      return null;
    }
  }

  function turnSignature(w) {
    const g = w.gameui && w.gameui.gamedatas;
    if (!g || !g.gamestate) return null;
    const state = String(g.gamestate.name || "");
    const active = String(g.gamestate.active_player || "");
    const me = String(g.me_id || "");
    if (!ADVISABLE.has(state)) return null;
    if (active !== me) return null; // not our decision
    // Distinguish successive turns in the same state: BGA bumps the move
    // counter, and the tableau shrinks as cards are taken.
    const moveNo =
      (w.gameui.gamedatas.gamestate && w.gameui.gamedatas.gamestate.id) || "";
    const cards = (g.draftpool && g.draftpool.cards && g.draftpool.cards.length) || 0;
    const built = Object.values(g.playerBuildings || {}).reduce(
      (n, list) => n + (list ? list.length : 0),
      0
    );
    // The draft needs its own progress signal, read from the DOM. Every
    // gamedatas field above is constant through it: the state name never
    // changes, draftpool is [] until an age is dealt, playerBuildings is empty,
    // and wondersSituation is stale for the whole draft. `active` is not enough
    // either -- the order is A-B-B-A, so picks 2 and 3 are the same player and
    // would collide on one signature, leaving the panel showing the previous
    // pick's advice.
    const offered = w.document.querySelectorAll(
      "#wonder_selection_container [data-wonder-id]"
    ).length;
    return [state, active, moveNo, cards, built, offered].join("|");
  }

  function capture() {
    // captureForAdvisor bundles gamedatas + gamestate.args + the DOM freshness
    // patch + the game log (for wonder-burial identities).
    const state = captureForAdvisor();
    const w = gameWindow();
    const g = w.gameui.gamedatas;
    // Sprite coordinates for the panel. Presentation only -- a miss costs an
    // image, never a wrong position, which is why this mapping may live in the
    // browser at all.
    //
    // Only BUILDINGS carry spriteXY: it is a server field (Building.php:24,73)
    // and no other class has one. BGA derives a wonder's and a token's cell
    // from its id instead (getWonderDivHtml :838-839,
    // getProgressTokenDivHtml :1307-1308), so we do the same rather than read a
    // field that does not exist. The column counts are the spritesheet's own,
    // matching --wonder-spritesheet-columns and
    // --progress-token-spritesheet-columns in sevenwondersduel.css.
    const cell = (id, columns) => [(id - 1) % columns, Math.floor((id - 1) / columns)];
    const art = { buildings: {}, wonders: {}, progressTokens: {} };
    for (const [id, b] of Object.entries(g.buildings || {})) {
      art.buildings[String(id)] = { name: b.name, spriteXY: b.spriteXY };
    }
    for (const [id, x] of Object.entries(g.wonders || {})) {
      art.wonders[String(id)] = { name: x.name, spriteXY: cell(Number(id), 5) };
    }
    for (const [id, t] of Object.entries(g.progressTokens || {})) {
      art.progressTokens[String(id)] = { name: t.name, spriteXY: cell(Number(id), 4) };
    }
    return { state, art, quality: document.getElementById("swd")?.dataset?.quality || "1x" };
  }

  let lastSignature = null;

  function tick() {
    const w = gameWindow();
    if (!w || w !== window) return; // another frame owns the board
    let signature = null;
    try {
      signature = turnSignature(w);
    } catch (e) {
      return;
    }
    if (signature === null) {
      if (lastSignature !== null) {
        lastSignature = null;
        post("idle", null);
      }
      return;
    }
    if (signature === lastSignature) return;
    lastSignature = signature;
    try {
      post("position", { signature, ...capture() });
    } catch (err) {
      post("capture_error", { signature, message: String(err && err.message) });
    }
  }

  // BGA writes the state name onto #swd on every transition
  // (sevenwondersduel.js:721), which makes a cheap, game-agnostic trigger. The
  // interval is a backstop: animations can land after the attribute changes, so
  // a taken card may not be out of the DOM at the instant we observe it.
  // 400ms was too eager: BGA animates a constructed card into the player area,
  // and a capture taken mid-flight shows fewer buildings than playersSituation
  // already reports -- which _assert_fresh correctly rejects as stale. The
  // content script also retries a rejected position, so this only has to be
  // right most of the time.
  const SETTLE_MS = 1200;
  const swd = document.getElementById("swd");
  if (swd) {
    new MutationObserver(() => setTimeout(tick, SETTLE_MS)).observe(swd, {
      attributes: true,
      attributeFilter: ["data-state"],
    });
  }
  setInterval(tick, 1500);
  setTimeout(tick, 800);

  window.addEventListener("message", (event) => {
    const data = event.data;
    if (!data || data.__swd !== TAG || data.type !== "recapture") return;
    lastSignature = null; // force the next tick to re-send
    tick();
  });
})();
