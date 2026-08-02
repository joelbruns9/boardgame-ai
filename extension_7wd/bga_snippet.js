// BGA -> advisor: browser-side capture for the 7 Wonders Duel table page.
//
// This is the ONLY code that runs in the page, and it deliberately holds no
// game knowledge: it grabs `window.gameui.gamedatas` verbatim and hands it off.
// All mapping to the advisor wire lives in Python (bga_extract.wire_from_bga),
// so a BGA UI change can never silently corrupt a position -- worst case the
// Python mapper raises.
//
// FRESHNESS -- READ THIS. `gameui.gamedatas` is the *page-load* payload. BGA
// patches some fields from its notification stream (scores, draftpool) but
// leaves five stale mid-game: playerBuildings, discardedBuildings,
// militaryTrack, progressTokensSituation, and -- during the wonder draft only --
// wondersSituation. They keep their load-time values until the next full load.
//
// `captureDomPatch` re-reads those five from the DOM so no reload is needed;
// `captureForAdvisor` bundles gamedatas + state args + that patch into the
// payload the host expects. `captureAfterReload` (F5 first) remains as the
// reference capture to diff a patched capture against.
//
// `wire_from_bga` still cross-checks science counts and raises StaleGamedata,
// which now catches a *broken patch* rather than a missing reload -- but note
// that check is blind during the draft, when both counts are 0.
//
// Usage from a content script / userscript / devtools:
//   const state = captureForAdvisor();           // -> plain object, or throws
//   fetch("http://localhost:8000/api/recommend", {   // host wraps wire_from_bga
//     method: "POST",
//     headers: { "Content-Type": "application/json" },
//     body: JSON.stringify({ state }),
//   });
//
// The BGA game lives in a nested iframe; from the top window, hop to the frame
// that actually owns `gameui`.

// All same-origin frames that own a 7WD `gameui`, shallowest first.
//
// Marker is `wondersSituation`, present in every 7WD state. Do NOT use
// `draftpool`: getAllDatas leaves it `[]` during the wonder draft
// (sevenwondersduel.game.php:685-688), which is only truthy in JS by accident.
function findGameWindows() {
  const found = [];
  const walk = (w, depth) => {
    try {
      if (w.gameui && w.gameui.gamedatas && w.gameui.gamedatas.wondersSituation) {
        found.push({ win: w, depth, url: String(w.location.href) });
      }
      for (let i = 0; i < w.frames.length; i++) walk(w.frames[i], depth + 1);
    } catch (e) {
      /* cross-origin frame: skip */
    }
  };
  walk(window.top, 0);
  found.sort((a, b) => a.depth - b.depth);
  return found;
}

// The frame for the seat we are advising.
//
// A table can expose MORE THAN ONE `gameui`: a nested `...&testuser=<id>` frame
// renders the *opponent's* seat, with its own `me_id`. Observed live on table
// 892846644. Picking the wrong one is not cosmetic -- per-seat private args
// vanish, so a Great Library position (`_private.progressTokensFromBox`) turns
// into an UnsupportedBgaState refusal, and any future per-seat field would be
// read from the wrong player.
//
// So: drop `testuser=` frames, then prefer the shallowest. In an ordinary table
// there is exactly one frame and this is a no-op.
function findGameWindow() {
  const all = findGameWindows();
  if (!all.length) {
    throw new Error("7WD gamedatas not found; open the game table first");
  }
  const real = all.filter((f) => !/[?&]testuser=/.test(f.url));
  const chosen = (real.length ? real : all)[0];
  return chosen.win;
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

// ---------------------------------------------------------------------------
// FRESHNESS PATCH (item B of ADVISOR_IMPLEMENTATION_PLAN.md)
//
// Five gamedatas fields are never refreshed after page load, so we re-read them
// from the DOM. This is the ONLY game knowledge in this file, and it is kept to
// a list of selectors that return **numeric ids only** -- the id -> card-name
// mapping stays in Python (bga_extract), so a BGA change breaks a selector
// loudly here rather than producing a plausible-but-wrong name.
//
// Selector provenance (BGA Files/sevenwondersduel/):
//   .player_buildings.player{PID}          tpl:222, written by notif_constructBuilding (js:3138)
//   #discarded_cards_container             tpl:391, js:1213 (add) / js:3437 (remove)
//   #board_progress_tokens                 tpl:307, js:1273
//   .player_info.player{PID} .player_area_progress_tokens   tpl:346,372, js:1312
//   .military_token_container[data-military-token-number]   tpl:316, js:5094
//   #wonder_selection_container            tpl:187, js:2625
//   .player_wonders.player{PID}            tpl:47,  js:2707
//
// TRAPS, both load-bearing:
//   * --conflict-pawn-position is already in SERVER frame (-9..9), same as
//     gamedatas.militaryTrack.conflictPawn. The per-seat mirroring is done by
//     CSS via --invert-military-positions (css:954). Do not flip it here.
//   * Military token slots are read by their data-military-token-number
//     attribute, NOT by #military_tokens>div:nth-of-type(i) -- that index is
//     mirrored by invertMilitaryTrack() (js:1746) for one of the two seats.

// Throw rather than return a partial patch: a missing selector must fail loudly.
function _required(root, selector) {
  const node = root.querySelector(selector);
  if (!node) throw new Error(`7WD DOM patch: selector not found: ${selector}`);
  return node;
}

function _idsUnder(root, selector, attr) {
  return [..._required(root, selector).querySelectorAll(`[${attr}]`)].map((n) =>
    parseInt(n.getAttribute(attr), 10)
  );
}

function captureDomPatch() {
  const w = findGameWindow();
  const doc = w.document;
  const g = w.gameui.gamedatas;
  const pids = Object.keys(g.players).map(String);

  const playerBuildings = {};
  const playerWonders = {};
  const playerProgressTokens = {};
  for (const pid of pids) {
    playerBuildings[pid] = _idsUnder(doc, `.player_buildings.player${pid}`, "data-building-id");
    playerWonders[pid] = _idsUnder(doc, `.player_wonders.player${pid}`, "data-wonder-id");
    playerProgressTokens[pid] = _idsUnder(
      doc,
      `.player_info.player${pid} .player_area_progress_tokens`,
      "data-progress-token-id"
    );
  }

  // Board progress tokens keep their slot: BGA renders slot N into the
  // (N+1)'th child of #board_progress_tokens (js:1283). Emit [slot, id] so
  // Python can rebuild location_arg rather than guess at ordering.
  const boardTokens = [];
  const tokenSlots = _required(doc, "#board_progress_tokens").children;
  for (let i = 0; i < tokenSlots.length; i++) {
    const tok = tokenSlots[i].querySelector("[data-progress-token-id]");
    if (tok) boardTokens.push([i, parseInt(tok.getAttribute("data-progress-token-id"), 10)]);
  }

  // Military tokens: value comes from the military_token_{value} class on the
  // child node; an empty container means the token has been captured.
  const militaryTokens = {};
  for (let n = 1; n <= 4; n++) {
    const container = _required(doc, `.military_token_container[data-military-token-number="${n}"]`);
    const token = container.querySelector(".military_token");
    let value = 0;
    if (token) {
      const m = /(?:^|\s)military_token_(\d+)(?:\s|$)/.exec(token.className);
      if (!m) throw new Error(`7WD DOM patch: unreadable military token value in slot ${n}`);
      value = parseInt(m[1], 10);
    }
    militaryTokens[n] = value;
  }

  const conflictPawnRaw = w
    .getComputedStyle(doc.documentElement)
    .getPropertyValue("--conflict-pawn-position")
    .trim();
  if (conflictPawnRaw === "") {
    throw new Error("7WD DOM patch: --conflict-pawn-position is unset");
  }

  return {
    playerBuildings,
    playerWonders,
    playerProgressTokens,
    boardProgressTokens: boardTokens,
    discardedBuildings: _idsUnder(doc, "#discarded_cards_container", "data-building-id"),
    wonderSelection: _idsUnder(doc, "#wonder_selection_container", "data-wonder-id"),
    militaryTokens,
    conflictPawn: parseInt(conflictPawnRaw, 10),
  };
}

// The payload the host's {"bga": ...} branch expects. `args` is the current
// state's server args (gameui.gamedatas.gamestate.args) -- fresh every turn,
// captured verbatim, used by Python only for cross-checks.
// Rendered game-log lines. Constructing a wonder buries an age card under it
// permanently, and `gamedatas` records only that card's *age* -- the identity
// lives solely in the constructWonder log line ("...using building X"). Python
// only trusts a parsed line when it agrees with the structural age, so a stale,
// trimmed or differently-localized log degrades to "counted but unnamed" rather
// than to a wrong position. Deduped: BGA renders each entry more than once.
function captureGameLog() {
  const doc = findGameWindow().document;
  const seen = new Set();
  for (const node of doc.querySelectorAll("#logs .log, .log")) {
    const text = node.textContent.trim();
    if (text) seen.add(text);
  }
  return [...seen];
}

function captureForAdvisor() {
  const g = captureBgaGamedatas();
  return {
    bga: g,
    args: g.gamestate && g.gamestate.args ? g.gamestate.args : null,
    dom: captureDomPatch(),
    log: captureGameLog(),
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
    findGameWindows,
    captureBgaGamedatas,
    captureBgaSlim,
    captureDomPatch,
    captureForAdvisor,
    captureAfterReload,
  };
}
