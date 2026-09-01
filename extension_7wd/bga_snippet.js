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
// `captureForAdvisor` bundles gamedatas + authoritative state-change args +
// that patch into the payload the host expects. `captureAfterReload` (F5 first) remains as the
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
//
// The `testuser=` URL is the only discriminator that has actually been
// observed, and it is a BGA convention rather than something we can derive:
// inside a frame, `me_id` and `gameui.player_id` both describe the seat that
// frame renders, so neither can tell an impersonated seat from a real one. The
// two checks below therefore do not try to be cleverer -- they make the cases
// this cannot resolve LOUD instead of silently picking the shallowest frame.
function findGameWindow() {
  const all = findGameWindows();
  if (!all.length) {
    throw new Error("7WD gamedatas not found; open the game table first");
  }
  const real = all.filter((f) => !/[?&]testuser=/.test(f.url));
  if (!real.length) {
    // EVERY board frame is impersonated. Falling back to them used to recreate
    // the exact wrong-seat failure this function exists to prevent: a lone
    // testuser frame passes the seat checks below, because its me_id really is
    // a seated player -- just not the human's. The URL is the only
    // discriminator we have, and here it says every candidate is wrong.
    //
    // BGA Studio genuinely works this way, so there is an opt-in: set
    // window.SWD_ALLOW_TESTUSER = true in the page to accept it anyway.
    if (!window.SWD_ALLOW_TESTUSER) {
      const err = new Error(
        "every 7WD frame on this page is a testuser= impersonation (" +
          all.length +
          " found); refusing rather than advise a seat that is not yours. Set " +
          "window.SWD_ALLOW_TESTUSER = true to override."
      );
      err.swdAmbiguous = true;
      throw err;
    }
  }
  const candidates = real.length ? real : all;

  const seatOf = (f) => String(((f.win.gameui || {}).gamedatas || {}).me_id || "");
  const seats = new Set(candidates.map(seatOf));
  if (seats.size > 1) {
    // Several frames, none marked testuser=, rendering different seats. Which
    // one is the human? Unknowable here, and guessing wrong costs the per-seat
    // private args -- which surfaces much later, and misleadingly, as an
    // UnsupportedBgaState on a Great Library position.
    const err = new Error(
      "several 7WD frames render different seats (" +
        [...seats].join(", ") +
        ") and none is marked testuser=; cannot tell which seat to advise"
    );
    err.swdAmbiguous = true;
    throw err;
  }

  const chosen = candidates[0];
  const gamedatas = chosen.win.gameui.gamedatas;
  const me = seatOf(chosen);
  if (!me || !Object.prototype.hasOwnProperty.call(gamedatas.players || {}, me)) {
    // A spectator frame reads the whole public board and none of the private
    // args, so it would advise happily right up until a Great Library position
    // refused. Without this it also fails silently in the extension, because
    // the panel only wakes when the active player is `me_id`.
    const err = new Error(
      "this frame is not seated at the table (me_id " +
        (me || "missing") +
        "); the advisor needs the playing seat -- per-seat private data, such " +
        "as the Great Library's box tokens, is sent only to it"
    );
    err.swdAmbiguous = true;
    throw err;
  }
  return chosen.win;
}

function captureBgaGamedatas(w) {
  w = w || findGameWindow();
  const g = w.gameui.gamedatas;
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

function captureCurrentStateArgs(w, gamedatas) {
  w = w || findGameWindow();
  const g = gamedatas || (w.gameui && w.gameui.gamedatas);
  const fallback = g && g.gamestate ? g.gamestate.args || null : null;
  const currentId = g && g.gamestate && g.gamestate.id;
  const store = w[SWD_PACKET_STORE];
  if (!store || currentId == null) return fallback;

  // `gameui.gamedatas.gamestate` updates the state id/name but can leave its
  // args at the page-load value. The raw gameStateChange notification is the
  // authoritative source. This is observable on chooseOpponentBuilding: the
  // packet carries args.buildingType = Brown/Grey while gamedatas carries no
  // colour, which made the advisor refuse the forced Wonder follow-up.
  //
  // Only inspect the MOST RECENT state change. Searching farther back for a
  // matching id could reuse an earlier Zeus/Circus choice if the current
  // transition packet were missing, silently turning a capture gap into a
  // plausible but wrong position.
  const packets = [...store.packets.values()];
  for (let i = packets.length - 1; i >= 0; i--) {
    const data = packets[i] && packets[i].data;
    if (!Array.isArray(data)) continue;
    for (let j = data.length - 1; j >= 0; j--) {
      const entry = data[j];
      if (!entry || entry.type !== "gameStateChange") continue;
      const change = entry.args || {};
      if (String(change.id) !== String(currentId)) return fallback;
      const args = change.args;
      return args && typeof args === "object"
        ? JSON.parse(JSON.stringify(args))
        : fallback;
    }
  }
  return fallback;
}

function captureForAdvisor() {
  const w = findGameWindow();
  const g = captureBgaGamedatas(w);
  return {
    bga: g,
    args: captureCurrentStateArgs(w, g),
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

// ---------------------------------------------------------------------------
// NOTIFICATION PACKET CAPTURE (the differential harness's oracle)
//
// `gamedatas` says what the board looks like. It never says what BGA *charged*
// -- the price it made you pay, the coins it handed back, where it moved the
// conflict pawn, what each end-game category scored. Those numbers exist only
// in BGA's notification packets, and they are the entire basis of the
// differential harness (games/seven_wonders_duel/bga_differential.py), which
// replays a real game and compares our engine's arithmetic to BGA's. That
// harness is how the military off-by-one was found.
//
// Provenance -- all four points verified live against a BGA replay page on
// 2026-08-15, not inferred:
//   * `gameui.notifqueue.onNotification(packet)` is the single entry point every
//     incoming packet passes through, raw and whole:
//       {channel, table_id, packet_id, packet_type, move_id, time, data: [...]}
//     which is exactly the shape the harness consumes.
//   * It also accepts a JSON *string* (it calls fromJson on it itself), so the
//     recorder has to as well or it would silently record nothing.
//   * A (re)loaded page is re-sent the history through that same call with
//     packet_type "resend". So opening the tab mid-game, or reloading it,
//     backfills instead of losing everything before it -- which matters,
//     because the harness refuses to replay a game with gaps in its moves.
//   * Replay/archive pages additionally preload `window.g_gamelogs` with the
//     same packet shape; the recorder seeds from it when it is there.
//
// Why hook the framework rather than subscribe per notification type: the
// per-type dojo topics carry only the game's own notifications, and the coin
// and score snapshots the harness compares against ride on framework
// `gameStateChange` packets. Subscribing would also leave holes in the move
// sequence, which the harness (correctly) treats as an untrustworthy capture.
//
// Kept: packets for THIS table, on either the /table channel or the /player
// channel this browser is subscribed to (our own seat's private stream -- a
// packet for a seat that is not ours is not delivered here in the first place).
// Both are kept because the harness refuses to replay a game whose move
// sequence has holes, and a move whose only packet came privately would
// otherwise read as a hole.
//
// Kept out: any other channel, and chat. The log is an engine oracle, not a
// transcript of the table talk.

const SWD_PACKET_STORE = "__swdPacketStore";

function _swdIsChat(packet) {
  const chatty = new Set([
    "chat",
    "groupchat",
    "chatmessage",
    "tablechat",
    "privatechat",
    "startWriting",
    "stopWriting",
  ]);
  if (chatty.has(String(packet.type || ""))) return true;
  const data = packet.data || [];
  return data.length > 0 && data.every((e) => chatty.has(String(e && e.type)));
}

function _swdRecord(store, packet) {
  if (typeof packet === "string") {
    // onNotification is documented above as accepting a raw JSON string.
    try {
      packet = JSON.parse(packet);
    } catch (e) {
      return;
    }
  }
  if (!packet || !packet.data || !Array.isArray(packet.data)) return;
  const channel = String(packet.channel || "");
  if (channel.substr(0, 6) !== "/table" && channel.substr(0, 7) !== "/player") return;
  // A /player packet with no table is a site notification, not this game.
  if (store.tableId && String(packet.table_id) !== store.tableId) return;
  if (_swdIsChat(packet)) return;
  const key = String(packet.move_id) + "/" + String(packet.packet_id);
  if (store.packets.has(key)) return;
  store.packets.set(key, packet);
  store.fresh.push(key);
}

// Idempotent: the bridge calls this on every tick, and the page keeps one store.
//
// It re-tries the hook while the store says it is unhooked, rather than
// returning early on the store's existence alone. The bridge can run before
// BGA has built `gameui`, and treating that first attempt as "installed" would
// leave the recorder permanently deaf on exactly the tables it is meant to
// watch -- silently, since a store would exist and simply stay empty.
function installPacketRecorder(w) {
  w = w || findGameWindow();
  let store = w[SWD_PACKET_STORE];
  if (!store) {
    store = { packets: new Map(), fresh: [], tableId: null, hooked: false, seeded: 0 };
    const match = /[?&]table=(\d+)/.exec(String(w.location.href));
    store.tableId = match ? match[1] : null;
    w[SWD_PACKET_STORE] = store;

    // Seed from the preloaded history where there is one (replay/archive pages).
    const preloaded = w.g_gamelogs;
    if (Array.isArray(preloaded)) {
      for (const packet of preloaded) _swdRecord(store, packet);
      store.seeded = store.packets.size;
    }
  }
  if (store.hooked) return store;

  const queue = w.gameui && w.gameui.notifqueue;
  if (queue && typeof queue.onNotification === "function") {
    const original = queue.onNotification;
    queue.onNotification = function (packet) {
      // Record first, then hand on untouched. Recording must never be able to
      // break the game: a throw here would take BGA's own dispatch with it.
      try {
        _swdRecord(store, packet);
      } catch (e) {
        /* never let capture break the table */
      }
      return original.apply(this, arguments);
    };
    store.hooked = true;
  }
  return store;
}

// Packets recorded since the last drain, oldest first. Draining rather than
// re-sending everything keeps each POST small; anything the caller fails to
// deliver is its problem to retry, and a reload re-seeds the history anyway.
function drainPackets(w) {
  w = w || findGameWindow();
  const store = w[SWD_PACKET_STORE];
  if (!store) return [];
  const fresh = store.fresh;
  store.fresh = [];
  return fresh.map((key) => store.packets.get(key)).filter(Boolean);
}

if (typeof module !== "undefined") {
  module.exports = {
    findGameWindow,
    findGameWindows,
    captureBgaGamedatas,
    captureBgaSlim,
    captureDomPatch,
    captureCurrentStateArgs,
    captureForAdvisor,
    captureAfterReload,
    installPacketRecorder,
    drainPackets,
  };
}
