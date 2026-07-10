const browserAPI = typeof browser !== "undefined" ? browser : chrome;
const usesPromiseAPI = typeof browser !== "undefined" && browserAPI === browser;

const ADVISOR_URL = "http://127.0.0.1:8000/api/recommend";
const PROBE_SAVE_URL = "http://127.0.0.1:8000/api/advisor-probe/save";
const GAME_LOG_URL = "http://127.0.0.1:8000/api/game-log/append";
const RESULT_EVENT = "kingdomino-advisor-state";
const OVERLAY_ID = "kingdomino-advisor-overlay";

// Architecture (channels/blocks/bilinear_dim) is intentionally NOT here. The
// server is the single source of truth for model architecture: it reads those
// values from the checkpoint's own config. The extension only sends a checkpoint
// path (or nothing, to use the server's autodiscovered current_best.pt).
const DEFAULT_OPTIONS = {
  engine: "auto",
  sims: 800,
  topK: 8,
  checkpoint: "",
  device: "cuda",
  exactMaxSecs: 300,
  exactThreads: 0,
  autoRefresh: true,
  gameLog: true,
};

const SIM_OPTIONS = [100, 200, 400, 800, 1600, 3200, 6400, 12800];

let inFlightRecommend = false;
let lastPayloadKey = null;
let lastStartedAt = 0;

console.log("[kingdomino-advisor] content.js loaded");

function getStorage(keys) {
  return new Promise((resolve, reject) => {
    try {
      if (usesPromiseAPI) {
        browserAPI.storage.local.get(keys).then((value) => resolve(value || {}), reject);
        return;
      }
      const result = browserAPI.storage.local.get(keys, (value) => resolve(value || {}));
      if (result && typeof result.then === "function") result.then((value) => resolve(value || {}), reject);
    } catch (e) {
      reject(e);
    }
  });
}

function setStorage(values) {
  return new Promise((resolve, reject) => {
    try {
      if (usesPromiseAPI) {
        browserAPI.storage.local.set(values).then(resolve, reject);
        return;
      }
      const result = browserAPI.storage.local.set(values, () => resolve());
      if (result && typeof result.then === "function") result.then(resolve, reject);
    } catch (e) {
      reject(e);
    }
  });
}

async function loadOptions() {
  const stored = await getStorage([
    "kingdomino_engine",
    "kingdomino_sims",
    "kingdomino_checkpoint",
    "kingdomino_top_k",
    "kingdomino_device",
    "kingdomino_exact_max_secs",
    "kingdomino_exact_threads",
    "kingdomino_auto_refresh",
    "kingdomino_game_log",
  ]);
  const sims = Number(stored.kingdomino_sims);
  const topK = Number(stored.kingdomino_top_k);
  const exactMaxSecs = Number(stored.kingdomino_exact_max_secs);
  const exactThreads = Number(stored.kingdomino_exact_threads);
  return {
    engine: stored.kingdomino_engine || DEFAULT_OPTIONS.engine,
    sims: Number.isFinite(sims) && sims > 0 ? Math.round(sims) : DEFAULT_OPTIONS.sims,
    topK: Number.isFinite(topK) && topK > 0 ? Math.round(topK) : DEFAULT_OPTIONS.topK,
    checkpoint: stored.kingdomino_checkpoint || DEFAULT_OPTIONS.checkpoint,
    device: stored.kingdomino_device || DEFAULT_OPTIONS.device,
    exactMaxSecs: Number.isFinite(exactMaxSecs) && exactMaxSecs >= 0 ? exactMaxSecs : DEFAULT_OPTIONS.exactMaxSecs,
    exactThreads: Number.isFinite(exactThreads) && exactThreads >= 0 ? Math.round(exactThreads) : DEFAULT_OPTIONS.exactThreads,
    autoRefresh: stored.kingdomino_auto_refresh === undefined
      ? DEFAULT_OPTIONS.autoRefresh
      : Boolean(stored.kingdomino_auto_refresh),
    gameLog: stored.kingdomino_game_log === undefined
      ? DEFAULT_OPTIONS.gameLog
      : Boolean(stored.kingdomino_game_log),
  };
}

async function saveOptionPatch(patch) {
  const out = {};
  if (patch.engine !== undefined) out.kingdomino_engine = patch.engine;
  if (patch.sims !== undefined) out.kingdomino_sims = patch.sims;
  if (patch.checkpoint !== undefined) out.kingdomino_checkpoint = patch.checkpoint;
  if (patch.topK !== undefined) out.kingdomino_top_k = patch.topK;
  if (patch.device !== undefined) out.kingdomino_device = patch.device;
  if (patch.exactMaxSecs !== undefined) out.kingdomino_exact_max_secs = patch.exactMaxSecs;
  if (patch.exactThreads !== undefined) out.kingdomino_exact_threads = patch.exactThreads;
  if (patch.autoRefresh !== undefined) out.kingdomino_auto_refresh = Boolean(patch.autoRefresh);
  if (patch.gameLog !== undefined) out.kingdomino_game_log = Boolean(patch.gameLog);
  await setStorage(out);
}

function timestampSlug(value) {
  const d = value ? new Date(value) : new Date();
  const iso = Number.isFinite(d.getTime()) ? d.toISOString() : new Date().toISOString();
  return iso.replace(/[:.]/g, "-");
}

function buildAdvisorProbe({ capture, payload = null, response = null, options = null, transport = null, reason = null, error = null }) {
  const capturedAt = (capture && capture.capturedAt) || new Date().toISOString();
  return {
    schema: "kingdomino-advisor-probe/v1",
    created_at: new Date().toISOString(),
    captured_at: capturedAt,
    url: (capture && capture.url) || location.href,
    table_id: capture && capture.tableId !== undefined ? capture.tableId : null,
    reason,
    options,
    transport,
    error,
    capture,
    advisor_payload: payload,
    advisor_response: response,
  };
}

function downloadProbe(probe) {
  const table = probe && probe.table_id ? `table-${probe.table_id}` : "table-unknown";
  const filename = `kingdomino-advisor-probe-${table}-${timestampSlug(probe && probe.captured_at)}.json`;
  return new Promise((resolve) => {
    try {
      const message = { action: "saveProbe", url: PROBE_SAVE_URL, filename, probe };
      if (usesPromiseAPI) {
        browserAPI.runtime.sendMessage(message).then(
          (response) => {
            if (response && response.ok) {
              resolve({ ...response, mode: "server-save", filename });
              return;
            }
            browserAPI.runtime.sendMessage({ action: "downloadProbe", filename, probe }).then(
              (fallback) => resolve(fallback && fallback.ok
                ? { ...fallback, mode: "browser-download", filename }
                : { ok: false, error: (response && response.error) || (fallback && fallback.error) || "probe save/download failed" }),
              (err) => resolve({ ok: false, error: String((err && err.message) || err) })
            );
          },
          (err) => resolve({ ok: false, error: String((err && err.message) || err) })
        );
        return;
      }
      const result = browserAPI.runtime.sendMessage(message, (response) => {
        if (browserAPI.runtime.lastError) {
          resolve({ ok: false, error: browserAPI.runtime.lastError.message });
        } else if (response && response.ok) {
          resolve({ ...response, mode: "server-save", filename });
        } else {
          browserAPI.runtime.sendMessage({ action: "downloadProbe", filename, probe }, (fallback) => {
            if (browserAPI.runtime.lastError) {
              resolve({ ok: false, error: (response && response.error) || browserAPI.runtime.lastError.message });
            } else {
              resolve(fallback && fallback.ok
                ? { ...fallback, mode: "browser-download", filename }
                : { ok: false, error: (response && response.error) || (fallback && fallback.error) || "probe save/download failed" });
            }
          });
        }
      });
      if (result && typeof result.then === "function") {
        result.then(
          (response) => resolve(response && response.ok
            ? { ...response, mode: "server-save", filename }
            : { ok: false, error: (response && response.error) || "probe save failed" }),
          (err) => resolve({ ok: false, error: String((err && err.message) || err) })
        );
      }
    } catch (e) {
      resolve({ ok: false, error: String((e && e.message) || e) });
    }
  });
}

function exactEligibleState(state) {
  if (!state || typeof state !== "object") return false;
  const phase = state.phase;
  const debugDeck = state.debug && Array.isArray(state.debug.deck) ? state.debug.deck : null;
  const deckCount = Number.isFinite(state.deck_count)
    ? state.deck_count
    : debugDeck
      ? debugDeck.length
      : null;
  if (phase === "GAME_OVER") return true;
  if (phase === "FINAL_PLACEMENT") return deckCount === 0;
  if (phase === "PLACE_AND_SELECT") {
    if (deckCount === 0) return true;
    return deckCount === 4 && debugDeck && debugDeck.length === 4;
  }
  return false;
}

function advisorEngineForState(state, requestedEngine) {
  const requested = String(requestedEngine || DEFAULT_OPTIONS.engine).toLowerCase();
  if (requested === "auto") {
    return exactEligibleState(state) ? "exact" : "nn";
  }
  return requestedEngine;
}

function pageReadKingdominoState() {
  let __result = { ok: false, error: "no result produced" };
  function emit(obj) {
    __result = obj;
  }

  function cloneSafe(value, depth, seen) {
    if (depth <= 0) return "[max-depth]";
    if (value === null) return null;
    const t = typeof value;
    if (t === "string" || t === "number" || t === "boolean") return value;
    if (t === "undefined") return null;
    if (t === "function") return "[function]";
    if (value instanceof Element) {
      return {
        tag: value.tagName,
        id: value.id || "",
        className: typeof value.className === "string" ? value.className : "",
        text: (value.textContent || "").trim().slice(0, 120),
      };
    }
    if (seen.indexOf(value) >= 0) return "[circular]";
    seen.push(value);
    if (Array.isArray(value)) {
      return value.slice(0, 250).map((x) => cloneSafe(x, depth - 1, seen));
    }
    const out = {};
    Object.keys(value).slice(0, 250).forEach((k) => {
      try {
        if (k === "notifqueue" || k === "socket" || k === "soundManager") return;
        out[k] = cloneSafe(value[k], depth - 1, seen);
      } catch (e) {
        out[k] = `[error: ${String((e && e.message) || e)}]`;
      }
    });
    seen.pop();
    return out;
  }

  function findCandidateState(gd) {
    const keys = [
      "kingdomino_advisor_state",
      "advisor_state",
      "debug_state",
      "public_state",
      "state",
    ];
    for (const k of keys) {
      const v = gd && gd[k];
      if (v && typeof v === "object" && (v.game === "kingdomino" || v.boards || v.current_row)) {
        return { state: cloneSafe(v, 8, []), source: `gamedatas.${k}` };
      }
    }
    return { state: null, source: null };
  }

  function asInt(value, fallback) {
    if (value === null || value === undefined || value === "") return fallback;
    const n = Number(value);
    return Number.isFinite(n) ? Math.trunc(n) : fallback;
  }

  const pageGlobal = typeof window !== "undefined" ? window : null;
  // The authoritative-board cache must survive a page reload (F5 during the
  // opponent's turn was the classic tiles-missing failure: BGA only exposes
  // the kingdom grid for the ACTIVE player, and a reload wiped the in-memory
  // copy of YOUR board). Mirror it in sessionStorage: per-tab, survives
  // reloads, and the tableId-scoped keys keep tables separate.
  const CACHE_SS_KEY = "__kingdominoAdvisorBoardCacheV1";
  let authoritativeBoardCache = {};
  if (pageGlobal) {
    if (!pageGlobal.__kingdominoAdvisorBoardCache) {
      try {
        const stored = pageGlobal.sessionStorage &&
          pageGlobal.sessionStorage.getItem(CACHE_SS_KEY);
        pageGlobal.__kingdominoAdvisorBoardCache = stored ? JSON.parse(stored) : {};
      } catch (e) {
        pageGlobal.__kingdominoAdvisorBoardCache = {};
      }
    }
    authoritativeBoardCache = pageGlobal.__kingdominoAdvisorBoardCache;
  }

  function persistBoardCache() {
    if (!pageGlobal || !pageGlobal.sessionStorage) return;
    try {
      pageGlobal.sessionStorage.setItem(
        CACHE_SS_KEY, JSON.stringify(authoritativeBoardCache));
    } catch (e) {
      /* quota/serialization failures: cache stays in-memory only */
    }
  }

  function cloneBoard(board) {
    if (!board || !Array.isArray(board.cells)) return null;
    return {
      canvas_size: board.canvas_size,
      castle_pos: Array.isArray(board.castle_pos) ? board.castle_pos.slice() : board.castle_pos,
      cells: board.cells.map((c) => ({ ...c })),
    };
  }

  function readDisplayedDeckCount() {
    try {
      const el = document.getElementById("dominoes_remaining");
      const text = el && el.textContent ? el.textContent.trim() : "";
      const match = text.match(/(\d+)\s+domino(?:es)?\s+remaining/i);
      return match ? asInt(match[1], null) : null;
    } catch (e) {
      return null;
    }
  }

  function sortedDominoIds(dominoes, predicate) {
    return Object.keys(dominoes || {})
      .map((k) => dominoes[k])
      .filter(predicate)
      .map((d) => asInt(d.number, asInt(d.id, null)))
      .filter((n) => Number.isFinite(n))
      .sort((a, b) => a - b);
  }

  function claimFromDomino(d, playerToIndex) {
    const player = playerToIndex[String(d.owner_player)];
    const domino = asInt(d.number, asInt(d.id, null));
    if (player === undefined || !Number.isFinite(domino)) return null;
    return { player, domino_id: domino };
  }

  function emptyBoards(playerCount, canvasSize) {
    const castle = Math.floor(canvasSize / 2);
    return Array.from({ length: playerCount }, () => ({
      canvas_size: canvasSize,
      castle_pos: [castle, castle],
      cells: [],
    }));
  }

  function terrainId(name) {
    // Case-insensitive by design: the live BGA capture (2026-06-23) confirmed
    // the kingdom-grid terrain strings are lowercase except the castle, which
    // BGA capitalizes as "Castle". The toLowerCase() below maps "Castle"→castle
    // →1, so do NOT remove it. Verified BGA names: field, forest, lake,
    // grassland, swamp, mountain, Castle. (wheat/water/grass/mine are kept as
    // engine-name aliases for the dominoesDescription path.)
    const key = String(name || "").toLowerCase();
    const ids = {
      castle: 1,
      field: 2,
      wheat: 2,
      forest: 3,
      lake: 4,
      water: 4,
      grassland: 5,
      grass: 5,
      swamp: 6,
      mountain: 7,
      mine: 7,
    };
    return ids[key] || 0;
  }

  function rotationOffset(rotation) {
    const r = asInt(rotation, 0);
    if (r === 1) return [0, -1];
    if (r === 2) return [-1, 0];
    if (r === 3) return [0, 1];
    return [1, 0];
  }

  function buildBoardFromKingdomArg(kingdom, canvasSize) {
    // kingdom[x][y] = {terrain, crowns} | null, castle at BGA (0,0).
    // engine coordinate = BGA coordinate + (castle, castle).
    const castle = Math.floor(canvasSize / 2);
    const cells = [];
    let hasCastle = false;
    if (kingdom && typeof kingdom === "object") {
      Object.keys(kingdom).forEach((xk) => {
        const col = kingdom[xk];
        if (!col || typeof col !== "object") return;
        Object.keys(col).forEach((yk) => {
          const c = col[yk];
          if (!c || !c.terrain) return;
          const bx = asInt(xk, null), by = asInt(yk, null);
          if (!Number.isFinite(bx) || !Number.isFinite(by)) return;
          const terr = String(c.terrain);
          // Reject cells outside the ±6 engine window around the castle; one
          // out-of-range cell makes the server reject the whole state (400).
          // bx/by are castle-relative (engine x = castle + bx = x - castlePos[0]).
          if (Math.abs(bx) > 6 || Math.abs(by) > 6) {
            console.warn('[advisor] buildBoardFromKingdomArg: skipping out-of-bounds cell',
              { x: castle + bx, y: castle + by, terrain: terr });
            return;
          }
          const isCastle = terr.toLowerCase() === "castle";
          if (isCastle) hasCastle = true;
          cells.push({
            x: castle + bx,
            y: castle + by,
            terrain: isCastle ? "CASTLE" : terr.toUpperCase(),
            terrain_id: terrainId(terr),
            crowns: asInt(c.crowns, 0),
            domino_id: isCastle ? -1 : -2, // -2 = placed (id unknown from this source)
          });
        });
      });
    }
    if (!hasCastle) {
      cells.push({ x: castle, y: castle, terrain: "CASTLE", terrain_id: 1, crowns: 0, domino_id: -1 });
    }
    return { canvas_size: canvasSize, castle_pos: [castle, castle], cells };
  }

  function buildBoardFromDom(bgaPlayerId, descriptions, castlePos) {
    // DOM-based reconstruction for a player whose authoritative per-cell grid is
    // NOT exposed in gamedatas — i.e. the OPPONENT (BGA only surfaces
    // gamestate.args.kingdom for the active player, and sends x/y/rotation=null
    // for every placed domino in gamedatas.dominoes). The placed tiles ARE in the
    // DOM though:
    //   <div id="kingdom_<bgaId>">
    //     <div id="domino_<N>[_placed]" class="domino shadow-rotation-<R>"
    //          style="left:<L>px; top:<T>px"> …
    // The castle sits at pixel (0,0) of the kingdom container and each cell is
    // 50px, so grid = (left/50, top/50) and engine = castlePos + grid.
    //
    // Tile selection: we match by the `shadow-rotation-` class, NOT by an
    // `_placed` id suffix. The 2026-06-23 live capture showed the OPPONENT's
    // placed tiles use a bare `domino_<N>` id (e.g. domino_19, domino_32) while
    // the active player's use `domino_<N>_placed` (domino_21_placed,
    // domino_25_placed). An `[id*='_placed']` selector would therefore miss the
    // opponent entirely — the one player this path exists to serve. Row/hand
    // tiles carry class "domino" with no shadow-rotation, so the rotation class
    // is the reliable "placed on the board" signal.
    //
    // Runs in the MAIN world (DOM available via chrome.scripting). Returns a
    // board in the same shape as buildBoardFromKingdomArg, or null when the
    // kingdom container is absent (e.g. spectator mode) so the caller can fall
    // back gracefully to a castle-only board.
    if (typeof document === "undefined" || !document.getElementById) return null;
    // Resolve the opponent kingdom container. The id is kingdom_<bgaPlayerId>,
    // but bgaPlayerId rotates per game/opponent and the direct id lookup can
    // miss (timing, id mismatch). When it does, the old code returned null and
    // the model got an EMPTY opponent board (inflated win prob). So fall back to
    // a STRUCTURAL lookup: BGA nests the opponent kingdom(s) inside the stable
    // container #other_players_kingdoms (the active player's own kingdom is NOT
    // in there). In a 2-player game there is exactly one kingdom div inside it.
    let container = null;
    let source = "none";
    const others = document.getElementById("other_players_kingdoms");
    if (others) {
      const kings = others.querySelectorAll("[id^='kingdom_']");
      // For hidden/opponent boards, prefer the stable opponent-board container.
      // In BGA hot-seat, a direct #kingdom_<playerId> can exist while actually
      // containing the active player's rendered tiles after the view flips.
      let match = null;
      for (const k of kings) {
        if (k.id === "kingdom_" + bgaPlayerId) { match = k; break; }
      }
      if (!match && kings.length === 1) match = kings[0];
      if (match) {
        container = match;
        source = "other_players_kingdoms";
      }
    }
    if (!container) {
      container = document.getElementById("kingdom_" + bgaPlayerId);
      if (container) source = "direct id";
    }
    console.log('[advisor] buildBoardFromDom: resolved opponent kingdom via',
      source,
      '-> container id:', container ? container.id : 'NONE');
    if (!container) return null;

    // The castle is not necessarily at pixel (0,0) of the kingdom container, so
    // read its actual rendered position and make every tile offset relative to
    // it. This keeps gx/gy castle-relative regardless of where BGA lays out the
    // castle within the container.
    const castleEl = container.querySelector('.castle');
    if (!castleEl) {
      console.warn('[advisor] buildBoardFromDom: no castle element found');
      return null;
    }
    // Coordinate system: derive every cell from RENDERED geometry
    // (getBoundingClientRect), using the castle's CENTER as the origin and its
    // rendered width as the cell pitch. Both castle and tiles are post-transform
    // screen pixels, so dividing their center difference by the rendered pitch
    // cancels any uniform scale — transform- and zoom-invariant, no hardcoded
    // pixel pitch. We use the element CENTER (not its corner): BGA renders each
    // domino as a 200x100 (2x1 cell) element and rotates it via CSS transform,
    // so getBoundingClientRect().left/top (the rotated bounding-box corner) is
    // offset by half a cell for vertical (rot1/rot3) tiles, which produced
    // half-cell positions that tripped the alignment guard. The center is
    // rotation-invariant and lands cleanly on the SEAM between the two cells.
    const castleRect = castleEl.getBoundingClientRect();
    const pitch = castleRect.width; // rendered cell size (handles scale/zoom)
    if (!pitch || !isFinite(pitch)) {
      console.warn('[advisor] buildBoardFromDom: castle has no rendered width; cannot reconstruct');
      return null; // caller falls back; warning banner will fire
    }
    const castleCx = castleRect.left + castleRect.width / 2;
    const castleCy = castleRect.top + castleRect.height / 2;
    console.log('[advisor] buildBoardFromDom: pitch=', pitch,
      'transform=',
      (typeof window !== "undefined" && window.getComputedStyle)
        ? window.getComputedStyle(container).transform : 'n/a');

    const castle = Array.isArray(castlePos) && castlePos.length === 2
      ? [asInt(castlePos[0], 7), asInt(castlePos[1], 7)]
      : [7, 7];
    const cells = [
      { x: castle[0], y: castle[1], terrain: "CASTLE", terrain_id: 1, crowns: 0, domino_id: -1 },
    ];
    // IDs of dominoes dropped from the reconstruction because a half fell outside
    // the engine bounds. Surfaced to the caller so an incomplete opponent board
    // becomes a visible failure rather than silently wrong recommendations.
    const skippedDominoes = [];

    let placed;
    try {
      placed = container.querySelectorAll("[class*='shadow-rotation-']");
    } catch (e) {
      return { canvas_size: 15, castle_pos: [castle[0], castle[1]], cells, skipped_dominoes: skippedDominoes };
    }

    placed.forEach((el) => {
      const idm = String(el.id || "").match(/domino_(\d+)/);
      if (!idm) return;
      const dominoId = asInt(idm[1], null);
      if (!Number.isFinite(dominoId)) return;

      const cls = typeof el.className === "string" ? el.className : "";
      const rm = cls.match(/shadow-rotation-([0-3])/);
      if (!rm) return;
      const rotation = asInt(rm[1], null);

      const desc = descriptions && descriptions[String(dominoId)];
      if (!desc || !desc.left || !desc.right) return;

      // Center-based grid math. The element center is rotation-invariant and
      // lands on the SEAM between the domino's two cells (half-integer on the
      // seam axis, integer on the other). cgx/cgy = element center in grid units
      // relative to the castle center.
      const r = el.getBoundingClientRect();
      const ex = r.left + r.width / 2;
      const ey = r.top + r.height / 2;
      const cgx = (ex - castleCx) / pitch;
      const cgy = (ey - castleCy) / pitch;

      // Two physical cells from the center, by orientation:
      //   rot 0/2 (horizontal): seam along x → center.x ~ half-int, center.y ~ int
      //   rot 1/3 (vertical):   seam along y → center.x ~ int,      center.y ~ half-int
      const horizontal = (rotation === 0 || rotation === 2);
      let seamFracOk;
      if (horizontal) {
        seamFracOk = Math.abs((cgx - Math.floor(cgx)) - 0.5) < 0.2
                  && Math.abs(cgy - Math.round(cgy)) < 0.2;
      } else {
        seamFracOk = Math.abs((cgy - Math.floor(cgy)) - 0.5) < 0.2
                  && Math.abs(cgx - Math.round(cgx)) < 0.2;
      }
      // Alignment sanity: the tile must sit cleanly on the seam (clean half/int
      // split). A poor fit means it is mid-animation; skip and record so the
      // caller's warning fires rather than committing a wrong cell.
      if (!seamFracOk) {
        skippedDominoes.push(dominoId);
        return;
      }

      // Map the two physical cells to the terrain halves with EXPLICIT
      // per-rotation anchors. rot1 and rot3 are vertical in OPPOSITE directions
      // and need OPPOSITE anchor-cell selection, so a single uniform y rule does
      // not work for both (a -oy negation fixes rot1 but collides on rot3).
      // cgx/cgy are grid units from the castle CENTER; smaller y is higher on
      // screen. Confirmed by live collision detection on opponent boards:
      //   rot0 (offset [1,0]):  desc.left=LEFT cell,  desc.right=RIGHT
      //   rot2 (offset [-1,0]): desc.left=RIGHT cell, desc.right=LEFT
      //   rot1 (offset [0,-1]): desc.left=UPPER cell, desc.right=LOWER
      //   rot3 (offset [0,+1]): desc.left=LOWER cell, desc.right=UPPER
      let leftX, leftY, rightX, rightY;
      if (rotation === 0) {
        leftX  = Math.floor(cgx); leftY  = Math.round(cgy);
        rightX = leftX + 1;       rightY = leftY;
      } else if (rotation === 2) {
        leftX  = Math.ceil(cgx);  leftY  = Math.round(cgy);
        rightX = leftX - 1;       rightY = leftY;
      } else if (rotation === 1) {
        leftX  = Math.round(cgx); leftY  = Math.floor(cgy);
        rightX = leftX;           rightY = leftY + 1;
      } else { // rotation === 3
        leftX  = Math.round(cgx); leftY  = Math.ceil(cgy);
        rightX = leftX;           rightY = leftY - 1;
      }
      // castle[] re-centers castle-relative grid coords into engine board coords;
      // the downstream bounds check and cell push both expect engine x/y.
      const halves = [
        { x: castle[0] + leftX,  y: castle[1] + leftY,  side: desc.left },
        { x: castle[0] + rightX, y: castle[1] + rightY, side: desc.right },
      ];
      // Bounds-check BOTH halves before adding either cell: a cell outside the
      // 0..14 canvas would index out of the server's board array. With the
      // rendered-geometry math this should never trigger for a real tile — it is
      // a safety net against a wildly mis-read tile. Skip the ENTIRE domino if
      // either half is out of range.
      const halfOutOfBounds = halves.some((h) =>
        h.x < 0 || h.x >= 15 || h.y < 0 || h.y >= 15
      );
      if (halfOutOfBounds) {
        console.warn('[advisor] buildBoardFromDom: skipping domino', dominoId,
          'out-of-bounds half (center grid', cgx.toFixed(2), cgy.toFixed(2),
          'rotation', rotation + ')');
        skippedDominoes.push(dominoId);
        return;
      }
      halves.forEach((h) => {
        cells.push({
          x: h.x,
          y: h.y,
          terrain: String(h.side.terrain || "").toUpperCase(),
          terrain_id: terrainId(h.side.terrain),
          crowns: asInt(h.side.crowns, 0),
          domino_id: dominoId,
        });
      });
    });

    return { canvas_size: 15, castle_pos: [castle[0], castle[1]], cells, skipped_dominoes: skippedDominoes };
  }

  function looksLikeKingdomGrid(obj) {
    // A BGA kingdom grid is kingdom[x][y] = {terrain, crowns} | null. Validate
    // the nested shape (at least one cell carrying a terrain) before trusting an
    // arbitrary object, so a same-named-but-unrelated field can't corrupt a board.
    if (!obj || typeof obj !== "object") return false;
    for (const xk of Object.keys(obj)) {
      const col = obj[xk];
      if (!col || typeof col !== "object") continue;
      for (const yk of Object.keys(col)) {
        const c = col[yk];
        if (c && typeof c === "object" && c.terrain) return true;
      }
    }
    return false;
  }

  function resolveKingdomForPlayer(gd, playerIndex, playerorder, activeIndex) {
    // Return BGA's authoritative per-cell kingdom grid for `playerIndex`, or null.
    //
    // AVAILABILITY: the only source BGA is CONFIRMED to expose is
    // gamestate.args.kingdom, and that is the board for the *current decision*,
    // i.e. the ACTIVE player only. The per-player lookups below are best-effort
    // for installs/states where BGA also surfaces opponent grids; each is
    // shape-validated, and when none match the caller falls back to the
    // (approximate) dominoes reconstruction. This keeps the active and opponent
    // code paths symmetric: same resolver, same builder, same fallback.
    const args = (gd && gd.gamestate && gd.gamestate.args) || {};
    if (playerIndex === activeIndex && looksLikeKingdomGrid(args.kingdom)) {
      return args.kingdom;
    }
    const bgaId = playerorder ? playerorder[playerIndex] : undefined;
    const candidates = [
      args.kingdoms && bgaId != null ? args.kingdoms[bgaId] : null,
      args.kingdoms ? args.kingdoms[playerIndex] : null,
      gd && gd.kingdoms && bgaId != null ? gd.kingdoms[bgaId] : null,
      gd && gd.players && bgaId != null && gd.players[bgaId] ? gd.players[bgaId].kingdom : null,
    ];
    for (const cand of candidates) {
      if (looksLikeKingdomGrid(cand)) return cand;
    }
    return null;
  }

  // APPROXIMATE fallback board reconstruction. This infers each placed tile's two
  // cells from a domino anchor (d.x, d.y) + rotation + desc.left/right half order.
  // That orientation arithmetic is the same class of logic that has been
  // bug-prone (see the flipped-half fix), so prefer buildBoardFromKingdomArg with
  // an authoritative per-cell grid whenever resolveKingdomForPlayer provides one.
  function buildBoardsFromDominoes(dominoes, descriptions, playerToIndex, playerCount, canvasSize) {
    const boards = emptyBoards(playerCount, canvasSize);
    const castle = Math.floor(canvasSize / 2);
    const placements = [];
    boards.forEach((board) => {
      board.cells.push({
        x: castle,
        y: castle,
        terrain: "CASTLE",
        terrain_id: 1,
        crowns: 0,
        domino_id: -1,
      });
    });

    Object.keys(dominoes || {}).forEach((k) => {
      const d = dominoes[k];
      if (String(d.location || "").toUpperCase() !== "KINGDOM") return;
      const player = playerToIndex[String(d.owner_player)];
      if (player === undefined || !boards[player]) return;
      const dominoId = asInt(d.number, asInt(d.id, asInt(k, null)));
      const desc = descriptions && descriptions[String(dominoId)];
      if (!desc) return;
      const bx = asInt(d.x, null);
      const by = asInt(d.y, null);
      // LIMITATION (confirmed by the 2026-06-23 live capture): for KINGDOM
      // dominoes BGA sends x:null, y:null, rotation:null in gamedatas.dominoes.
      // Without an anchor (bx,by) this approximate reconstruction cannot place
      // the tile, so every placed domino is skipped here and the owner's board
      // falls back to castle-only. This is the root cause of the empty opponent
      // board: the ACTIVE player's board is rebuilt authoritatively from
      // gamestate.args.kingdom (see buildBoardFromKingdomArg), but BGA does not
      // expose a per-cell grid for the opponent, so the opponent has only this
      // path available.
      //
      // The DOM does expose rotation via the placed tile's class
      // (e.g. id="domino_25_placed" class="domino shadow-rotation-2"), but NOT a
      // usable board anchor: position is pixel-based CSS (style.left/top) that
      // would require calibrating the board origin and cell size, and the
      // current scrape does not capture it. Rotation alone is insufficient
      // without (bx,by), so we intentionally do not partially reconstruct here.
      // Skipping (rather than guessing) keeps the fallback graceful: no crash,
      // and the owner board is castle-only instead of wrong.
      if (!Number.isFinite(bx) || !Number.isFinite(by)) return;
      const [dx, dy] = rotationOffset(d.rotation);
      const halves = [
        { x: bx, y: by, side_name: "left", side: desc.left },
        { x: bx + dx, y: by + dy, side_name: "right", side: desc.right },
      ];
      const placementDebug = {
        domino_id: dominoId,
        owner_bga: d.owner_player == null ? null : String(d.owner_player),
        player,
        location: d.location || null,
        rotation: d.rotation == null ? null : String(d.rotation),
        offset: [dx, dy],
        anchor_bga: [bx, by],
        cells: [],
      };
      halves.forEach((h) => {
        const cell = {
          x: castle + h.x,
          y: castle + h.y,
          terrain: String(h.side.terrain || "").toUpperCase(),
          terrain_id: terrainId(h.side.terrain),
          crowns: asInt(h.side.crowns, 0),
          domino_id: dominoId,
        };
        boards[player].cells.push(cell);
        placementDebug.cells.push({
          side: h.side_name,
          bga: [h.x, h.y],
          engine: [cell.x, cell.y],
          terrain: cell.terrain,
          crowns: cell.crowns,
        });
      });
      placements.push(placementDebug);
    });
    return { boards, placements };
  }

  function summarizeKingdom(kingdom) {
    if (!kingdom || typeof kingdom !== "object") {
      return { available: false, cells_seen: 0, non_empty_count: 0, samples: [] };
    }
    const samples = [];
    let cellsSeen = 0;
    let nonEmptyCount = 0;
    // Bounds were previously read as kingdom.xMin/xMax/yMin/yMax, but those
    // fields do not live on the grid object. The live capture (2026-06-23)
    // shows BGA stores them as siblings of `kingdom` on gamestate.args
    // (args.xMin, args.xMax, ...), so kingdom.xMin was always undefined → null.
    // Instead, derive the bounding box from the non-empty cells we actually
    // see. These are BGA coordinates (castle at 0,0) and match
    // gd.players[id].kingdom.{minX,minY,maxX,maxY} as a cross-check.
    let xMin = null, xMax = null, yMin = null, yMax = null;
    Object.keys(kingdom).forEach((xKey) => {
      const col = kingdom[xKey];
      if (!col || typeof col !== "object") return;
      Object.keys(col).forEach((yKey) => {
        cellsSeen += 1;
        const cell = col[yKey];
        const cloned = cloneSafe(cell, 5, []);
        const text = JSON.stringify(cloned);
        const isEmpty =
          cell === null ||
          cell === false ||
          cell === "" ||
          text === "{}" ||
          text === "[]" ||
          /empty/i.test(text);
        if (!isEmpty) {
          nonEmptyCount += 1;
          const bx = asInt(xKey, null);
          const by = asInt(yKey, null);
          if (Number.isFinite(bx)) {
            xMin = xMin === null ? bx : Math.min(xMin, bx);
            xMax = xMax === null ? bx : Math.max(xMax, bx);
          }
          if (Number.isFinite(by)) {
            yMin = yMin === null ? by : Math.min(yMin, by);
            yMax = yMax === null ? by : Math.max(yMax, by);
          }
          if (samples.length < 80) {
            samples.push({
              x: asInt(xKey, xKey),
              y: asInt(yKey, yKey),
              raw: cloned,
            });
          }
        }
      });
    });
    return {
      available: true,
      cells_seen: cellsSeen,
      non_empty_count: nonEmptyCount,
      samples,
      bounds: { xMin, xMax, yMin, yMax },
    };
  }

  function normalizeBgaState(gd) {
    const playerorder = Array.isArray(gd.playerorder) ? gd.playerorder.map((id) => String(id)) : [];
    const playerToIndex = {};
    playerorder.forEach((id, idx) => { playerToIndex[String(id)] = idx; });
    const players = Math.max(2, playerorder.length || Object.keys(gd.players || {}).length || 2);
    const activePlayer = gd.gamestate && gd.gamestate.active_player;
    const activeIndex = playerToIndex[String(activePlayer)];
    const gameStateName = gd.gamestate && gd.gamestate.name;
    const dominoes = gd.dominoes || {};
    const dominoesDescription = gd.dominoesDescription || {};
    const boardSize = asInt(gd.gridSize, 7);
    const canvasSize = 15;
    const tableId = gd.table_id || gd.tableId || "unknown";
    const cacheKeyForPlayer = (playerIndex) => `${tableId}:${playerorder[playerIndex] || playerIndex}`;
    const cachedBoardForPlayer = (playerIndex) => {
      const entry = authoritativeBoardCache[cacheKeyForPlayer(playerIndex)];
      return entry && entry.board ? cloneBoard(entry.board) : null;
    };
    const cacheAuthoritativeBoard = (playerIndex, board, expectedCells) => {
      if (!board || !Array.isArray(board.cells) || board.cells.length < expectedCells) return;
      authoritativeBoardCache[cacheKeyForPlayer(playerIndex)] = {
        board: cloneBoard(board),
        cell_count: board.cells.length,
        updated_at: Date.now(),
      };
      persistBoardCache();
    };
    const cachedBoardHasExpectedCells = (board, expectedCells) => {
      return !!(board && Array.isArray(board.cells) && board.cells.length >= expectedCells);
    };

    // Count dominoes on each player's board, by player index. Used to validate
    // the DOM-reconstructed opponent board: each placed domino occupies 2 cells,
    // so a complete board has 1 (castle) + placedCount*2 cells.
    // NOTE: BGA's location string for a tile placed on a board is "KINGDOM"
    // (confirmed via live gamedatas.dominoes capture), NOT "PLACED". Using
    // "PLACED" here counted 0 for every player, so expectedCells was always 1
    // (castle only) and the board_reconstruction_warning never fired.
    const placedCountByPlayer = {};
    const kingdomIdsByPlayer = {};
    Object.keys(dominoes).forEach((k) => {
      const d = dominoes[k];
      if (String(d.location || "").toUpperCase() !== "KINGDOM") return;
      const pi = playerToIndex[String(d.owner_player)];
      if (Number.isFinite(pi)) {
        placedCountByPlayer[pi] = (placedCountByPlayer[pi] || 0) + 1;
        const dominoId = asInt(d.number, asInt(d.id, asInt(k, null)));
        if (Number.isFinite(dominoId)) {
          if (!kingdomIdsByPlayer[pi]) kingdomIdsByPlayer[pi] = [];
          kingdomIdsByPlayer[pi].push(dominoId);
        }
      }
    });
    Object.keys(kingdomIdsByPlayer).forEach((pi) => {
      kingdomIdsByPlayer[pi].sort((a, b) => a - b);
    });

    const allClaims = Object.keys(dominoes)
      .map((k) => dominoes[k])
      .map((d) => claimFromDomino(d, playerToIndex))
      .filter(Boolean)
      .sort((a, b) => a.domino_id - b.domino_id);

    const futureOwned = Object.keys(dominoes)
      .map((k) => dominoes[k])
      .filter((d) => String(d.location || "").toUpperCase() === "FUTURE" && d.owner_player !== null && d.owner_player !== undefined)
      .map((d) => claimFromDomino(d, playerToIndex))
      .filter(Boolean)
      .sort((a, b) => a.domino_id - b.domino_id);

    const currentOwned = Object.keys(dominoes)
      .map((k) => dominoes[k])
      .filter((d) => String(d.location || "").toUpperCase() === "CURRENT" && d.owner_player !== null && d.owner_player !== undefined)
      .map((d) => claimFromDomino(d, playerToIndex))
      .filter(Boolean)
      .sort((a, b) => a.domino_id - b.domino_id);

    const activeDomino = gd.gamestate && gd.gamestate.args
      ? asInt(gd.gamestate.args.domino, null)
      : null;
    const activeClaim = activeIndex === undefined || !Number.isFinite(activeDomino)
      ? null
      : { player: activeIndex, domino_id: activeDomino };

    const unownedFuture = sortedDominoIds(dominoes, (d) => (
      String(d.location || "").toUpperCase() === "FUTURE" &&
      (d.owner_player === null || d.owner_player === undefined || d.owner_player === "")
    ));
    const visibleDominoIds = Object.keys(dominoes)
      .map((k) => {
        const d = dominoes[k];
        return asInt(d.number, asInt(d.id, asInt(k, null)));
      })
      .filter((n) => Number.isFinite(n));
    const allDominoIds = Object.keys(gd.dominoesDescription || {}).length
      ? Object.keys(dominoesDescription).map((k) => asInt(k, null)).filter((n) => Number.isFinite(n))
      : Array.from({ length: 48 }, (_, i) => i + 1);
    const visibleSet = {};
    visibleDominoIds.forEach((n) => { visibleSet[n] = true; });
    const displayedDeckCount = readDisplayedDeckCount();
    const inferredHiddenDeck = allDominoIds
      .filter((n) => !visibleSet[n])
      .sort((a, b) => a - b);
    let hiddenDeck = inferredHiddenDeck;
    if (Number.isFinite(displayedDeckCount) && displayedDeckCount >= 0 && displayedDeckCount < inferredHiddenDeck.length) {
      hiddenDeck = inferredHiddenDeck.slice(0, displayedDeckCount);
    }

    const debug = {
      source: "bga.gameui.gamedatas",
      bga_state_name: gameStateName,
      bga_active_player: activePlayer == null ? null : String(activePlayer),
      bga_playerorder: playerorder,
      // Histogram of raw BGA location strings. Confirmed live values:
      //   "KINGDOM" = tile placed on a player's board
      //   "CURRENT" = tile in hand / being placed this turn
      //   "FUTURE"  = tile in the future row (claimed if owner_player set)
      // (Do not assume "PLACED" — BGA never uses it; see placedCountByPlayer.)
      bga_locations: Object.keys(dominoes).reduce((acc, k) => {
        const loc = String(dominoes[k].location || "UNKNOWN").toUpperCase();
        acc[loc] = (acc[loc] || 0) + 1;
        return acc;
      }, {}),
      bga_current_domino: activeDomino,
      bga_current_position: gd.gamestate && gd.gamestate.args ? gd.gamestate.args.currentPosition : null,
      bga_turns_left: asInt(gd.turnsLeft, null),
      displayed_deck_count: displayedDeckCount,
      inferred_hidden_deck: inferredHiddenDeck,
      kingdom_summary: summarizeKingdom(gd.gamestate && gd.gamestate.args && gd.gamestate.args.kingdom),
      deck: hiddenDeck,
      notes: [],
    };

    if (activeIndex === undefined) {
      debug.notes.push("active BGA player is not in playerorder; defaulting start/current player to 0");
    }
    if (Number.isFinite(displayedDeckCount) && displayedDeckCount >= 0 && displayedDeckCount < inferredHiddenDeck.length) {
      debug.notes.push(`Clamped hidden deck from inferred ${inferredHiddenDeck.length} tile(s) to displayed ${displayedDeckCount}; omitted IDs: ${inferredHiddenDeck.slice(displayedDeckCount).join(", ") || "none"}.`);
    }

    let phase = null;
    let currentRow = [];
    let pendingClaims = [];
    let nextClaims = [];
    let actorIndex = 0;
    let initialPickCount = 0;
    let startPlayer = activeIndex === undefined ? 0 : activeIndex;
    const boardBuild = buildBoardsFromDominoes(dominoes, dominoesDescription, playerToIndex, players, canvasSize);
    const castleCenter = Math.floor(canvasSize / 2);

    // Per-player board resolution, in priority order:
    //   1. Authoritative BGA kingdom grid (gamestate.args.kingdom) — active
    //      player only; exact terrain/crowns per cell.
    //   2. DOM reconstruction (buildBoardFromDom) — opponents, whose grid BGA
    //      does not expose and whose gamedatas x/y/rotation are null. Reads the
    //      rendered tiles' pixel position + shadow-rotation class.
    //   3. Approximate dominoes reconstruction (buildBoardsFromDominoes, built
    //      above) — legacy fallback; inert when BGA sends null coords.
    //   4. Castle-only board — last resort (already present in every board).
    const domCastle = [castleCenter, castleCenter];
    // Set when a DOM-reconstructed opponent board has fewer cells than BGA says
    // it should — surfaced to the UI as a loud failure banner (see
    // renderRecommendations). Last incomplete board wins if several are bad.
    let boardReconstructionWarning = null;
    for (let p = 0; p < boardBuild.boards.length; p++) {
      const grid = resolveKingdomForPlayer(gd, p, playerorder, activeIndex);
      if (grid) {
        boardBuild.boards[p] = buildBoardFromKingdomArg(grid, canvasSize);
        cacheAuthoritativeBoard(p, boardBuild.boards[p], 1 + ((placedCountByPlayer[p] || 0) * 2));
        debug.notes.push(`Player ${p} board built from authoritative BGA kingdom grid.`);
        console.log('[advisor] board', p, 'built via: authoritative-grid',
          JSON.stringify(boardBuild.boards[p].cells.map(c => [c.x, c.y])));
        continue;
      }
      const bgaId = playerorder ? playerorder[p] : undefined;
      const domBoard = bgaId != null ? buildBoardFromDom(bgaId, dominoesDescription, domCastle) : null;
      console.log('[advisor] board', p, 'buildBoardFromDom ->',
        domBoard === null ? 'null' : (domBoard.cells.length + ' cell(s)'),
        domBoard ? JSON.stringify(domBoard.cells.map(c => [c.x, c.y])) : '');
      if (domBoard && domBoard.cells.length > 1) {
        const expectedIds = kingdomIdsByPlayer[p] || [];
        const reconstructedIds = Array.from(new Set(
          domBoard.cells
            .map((c) => asInt(c.domino_id, null))
            .filter((id) => Number.isFinite(id) && id > 0)
        )).sort((a, b) => a - b);
        const foreignIds = reconstructedIds.filter((id) => expectedIds.indexOf(id) < 0);
        if (expectedIds.length && foreignIds.length) {
          debug.notes.push(`Player ${p} DOM reconstruction rejected: expected KINGDOM domino IDs ${expectedIds.join(", ")}, but DOM board contained foreign IDs ${foreignIds.join(", ")}.`);
          const expectedCells = 1 + ((placedCountByPlayer[p] || 0) * 2);
          const cachedBoard = cachedBoardForPlayer(p);
          if (cachedBoardHasExpectedCells(cachedBoard, expectedCells)) {
            boardBuild.boards[p] = cachedBoard;
            debug.notes.push(`Player ${p} board restored from hot-seat authoritative cache after rejecting foreign DOM tiles.`);
            continue;
          }
        } else {
        boardBuild.boards[p] = domBoard;
        debug.notes.push(`Player ${p} board built from DOM reconstruction (${domBoard.cells.length - 1} placed cell(s) from rendered tiles).`);
        console.log('[advisor] board', p, 'built via: DOM');

        // Validate completeness against BGA's KINGDOM (on-board) tile count. A short board means
        // tiles were dropped (out-of-bounds skips) and the model would be fed a
        // partial opponent board — keep what we have but flag it loudly.
        const expectedCells = 1 + ((placedCountByPlayer[p] || 0) * 2);
        const actualCells = domBoard.cells.length;
        const boardIncomplete = actualCells < expectedCells;
        const expectedIds = kingdomIdsByPlayer[p] || [];
        const reconstructedIds = Array.from(new Set(
          domBoard.cells
            .map((c) => asInt(c.domino_id, null))
            .filter((id) => Number.isFinite(id) && id > 0)
        )).sort((a, b) => a - b);
        const missingIds = expectedIds.filter((id) => reconstructedIds.indexOf(id) < 0);
        const skippedIds = domBoard.skipped_dominoes && domBoard.skipped_dominoes.length
          ? domBoard.skipped_dominoes
          : missingIds;
        if (boardIncomplete) {
          // Try the authoritative cache BEFORE settling for a partial board —
          // this fallback existed on the foreign-ID and approximate paths but
          // was missing here, so a short DOM reconstruction shadowed a
          // complete cached board.
          const cachedBoard = cachedBoardForPlayer(p);
          if (cachedBoardHasExpectedCells(cachedBoard, expectedCells)) {
            boardBuild.boards[p] = cachedBoard;
            debug.notes.push(`Player ${p} board restored from authoritative cache (${cachedBoard.cells.length} cell(s)) after incomplete DOM reconstruction (${actualCells}/${expectedCells}).`);
            continue;
          }
          debug.notes.push(`Player ${p} board INCOMPLETE: expected ${expectedCells} cells (${placedCountByPlayer[p] || 0} placed dominoes × 2 + castle), got ${actualCells}. Skipped domino IDs: ${skippedIds.join(", ") || "unknown (out-of-bounds)"}.`);
          boardReconstructionWarning = {
            player: p,
            expected_cells: expectedCells,
            actual_cells: actualCells,
            expected_domino_ids: expectedIds,
            reconstructed_domino_ids: reconstructedIds,
            missing_domino_ids: missingIds,
            skipped_domino_ids: skippedIds,
            message: `Opponent board missing ${expectedCells - actualCells} cell(s) — advisor recommendations may be unreliable.`,
          };
        }
        continue;
        }
      }
      debug.notes.push(`Player ${p} board uses APPROXIMATE dominoes reconstruction / castle-only (no authoritative kingdom grid and no DOM-placed tiles found${p === activeIndex ? "" : "; BGA exposes the kingdom grid for the active player only"}).`);
      console.log('[advisor] board', p, 'built via:',
        boardBuild.boards[p].cells.length > 1 ? 'approx' : 'castle-only',
        JSON.stringify(boardBuild.boards[p].cells.map(c => [c.x, c.y])));
      const expectedCells = 1 + ((placedCountByPlayer[p] || 0) * 2);
      const actualCells = boardBuild.boards[p].cells.length;
      if (actualCells < expectedCells) {
        const cachedBoard = cachedBoardForPlayer(p);
        if (cachedBoardHasExpectedCells(cachedBoard, expectedCells)) {
          boardBuild.boards[p] = cachedBoard;
          debug.notes.push(`Player ${p} board restored from hot-seat authoritative cache (${cachedBoard.cells.length} cached cell(s), expected ${expectedCells}).`);
          continue;
        }
        const expectedIds = kingdomIdsByPlayer[p] || [];
        const reconstructedIds = Array.from(new Set(
          boardBuild.boards[p].cells
            .map((c) => asInt(c.domino_id, null))
            .filter((id) => Number.isFinite(id) && id > 0)
        )).sort((a, b) => a - b);
        const missingIds = expectedIds.filter((id) => reconstructedIds.indexOf(id) < 0);
        boardReconstructionWarning = {
          player: p,
          expected_cells: expectedCells,
          actual_cells: actualCells,
          expected_domino_ids: expectedIds,
          reconstructed_domino_ids: reconstructedIds,
          missing_domino_ids: missingIds,
          skipped_domino_ids: missingIds,
          message: `Opponent board missing ${expectedCells - actualCells} cell(s) - advisor recommendations may be unreliable.`,
        };
        debug.notes.push(`Player ${p} approximate board incomplete: expected IDs ${expectedIds.join(", ") || "none"}, reconstructed IDs ${reconstructedIds.join(", ") || "none"}, missing IDs ${missingIds.join(", ") || "none"}.`);
      }
    }

    if (gameStateName === "chooseDomino") {
      phase = "INITIAL_SELECTION";
      currentRow = unownedFuture;
      nextClaims = futureOwned;
      initialPickCount = nextClaims.length;
      if (activeIndex !== undefined) {
        if (initialPickCount === 0 || initialPickCount === 3) {
          startPlayer = activeIndex;
        } else if (initialPickCount === 1 || initialPickCount === 2) {
          startPlayer = 1 - activeIndex;
        }
      }
      debug.notes.push("Mapped BGA chooseDomino to engine INITIAL_SELECTION.");
    } else if (gameStateName === "placeDomino") {
      // Final round: the deck is exhausted so there are no future dominoes to
      // pick. The engine models this as FINAL_PLACEMENT (place only, no pick);
      // PLACE_AND_SELECT would require a pick from current_row and, with an
      // empty row, yields ZERO legal actions (place+pick over an empty row) —
      // which is why the advisor returned "No recommendations" on the last turn.
      const isFinalRound = unownedFuture.length === 0;
      phase = isFinalRound ? "FINAL_PLACEMENT" : "PLACE_AND_SELECT";
      currentRow = unownedFuture;
      pendingClaims = currentOwned.length ? currentOwned : (activeClaim ? [activeClaim] : allClaims);
      if (activeClaim && !pendingClaims.some((c) => c.player === activeClaim.player && c.domino_id === activeClaim.domino_id)) {
        pendingClaims.push(activeClaim);
        pendingClaims.sort((a, b) => a.domino_id - b.domino_id);
      }
      nextClaims = futureOwned;
      actorIndex = activeClaim
        ? pendingClaims.findIndex((c) => c.player === activeClaim.player && c.domino_id === activeClaim.domino_id)
        : -1;
      if (actorIndex < 0) {
        actorIndex = pendingClaims.findIndex((c) => c.player === activeIndex);
      }
      if (actorIndex < 0) actorIndex = 0;
      debug.active_claim = activeClaim;
      debug.current_owned_claims = currentOwned;
      debug.next_owned_claims = futureOwned;
      debug.actor_claim = pendingClaims[actorIndex] || null;
      debug.notes.push(isFinalRound
        ? "Mapped BGA placeDomino (no future dominoes left) to engine FINAL_PLACEMENT — place only, no pick."
        : "Mapped BGA placeDomino to engine PLACE_AND_SELECT using CURRENT owned dominoes as remaining pending claims.");
    } else {
      return {
        ok: false,
        error: `unsupported BGA gamestate ${JSON.stringify(gameStateName)}`,
        debug,
      };
    }

    if (!currentRow.length && phase !== "FINAL_PLACEMENT") {
      debug.notes.push("No unowned FUTURE dominoes found for current_row.");
    }

    return {
      ok: true,
      state: {
        game: "kingdomino",
        rules: {
          players,
          board_size: boardSize,
          canvas_size: canvasSize,
          harmony: true,
          middle_kingdom: true,
          mighty_duel: true,
        },
        phase,
        current_actor: activeIndex === undefined ? startPlayer : activeIndex,
        actor_index: actorIndex,
        initial_pick_count: initialPickCount,
        start_player: startPlayer,
        current_row: currentRow,
        pending_claims: pendingClaims,
        next_claims: nextClaims,
        deck_count: hiddenDeck.length,
        boards: boardBuild.boards,
        visible_history: [],
        // Present only when an opponent board failed completeness validation;
        // renderRecommendations reads this to show a loud warning banner.
        board_reconstruction_warning: boardReconstructionWarning,
        debug: {
          ...debug,
          reconstructed_placements: boardBuild.placements,
        },
      },
      source: "bga-normalized",
      debug: {
        ...debug,
        reconstructed_placements: boardBuild.placements,
      },
    };
  }

  function sampleDom() {
    const selectors = [
      "[id*='kingdomino' i]",
      "[class*='kingdomino' i]",
      "[id*='domino' i]",
      "[class*='domino' i]",
      "[data-tile-id]",
      "[data-domino-id]",
      "[data-x][data-y]",
    ];
    const samples = [];
    selectors.forEach((sel) => {
      try {
        document.querySelectorAll(sel).forEach((el) => {
          if (samples.length >= 80) return;
          if (el.id === "kingdomino-advisor-overlay" || (el.closest && el.closest("#kingdomino-advisor-overlay"))) return;
          samples.push({
            selector: sel,
            tag: el.tagName,
            id: el.id || "",
            className: typeof el.className === "string" ? el.className : "",
            dataset: Object.assign({}, el.dataset || {}),
            text: (el.textContent || "").trim().replace(/\s+/g, " ").slice(0, 160),
          });
        });
      } catch (e) {
        samples.push({ selector: sel, error: String((e && e.message) || e) });
      }
    });
    return samples;
  }

  function harvestSpriteMap() {
    const map = {};
    let spriteUrl = null;
    document.querySelectorAll('.domino-background').forEach(el => {
      let p = el;
      for (let i = 0; i < 10 && p; i++) {
        const m = (p.id || '').match(/domino_(\d+)$/);
        if (m) {
          const id = parseInt(m[1]);
          const style = window.getComputedStyle(el);
          // Grab sprite URL once
          if (!spriteUrl) {
            const bg = style.backgroundImage.match(/url\("([^"]+)"\)/);
            if (bg) spriteUrl = bg[1];
          }
          const pos = style.backgroundPosition.match(/([-\d]+)px\s+([-\d]+)px/);
          if (pos) map[id] = [parseInt(pos[1]), parseInt(pos[2])];
          break;
        }
        p = p.parentElement;
      }
    });
    return { spriteUrl, map };
  }

  try {
    if (typeof gameui === "undefined" || !gameui.gamedatas) {
      emit({ ok: false, error: "gameui.gamedatas is not available. Open an active BGA Kingdomino table first." });
      return __result;
    }

    const gd = gameui.gamedatas;
    const candidate = findCandidateState(gd);
    const normalized = candidate.state ? { ok: true, state: candidate.state, source: candidate.source } : normalizeBgaState(gd);
    const sprite = harvestSpriteMap();
    emit({
      ok: true,
      spriteUrl: sprite.spriteUrl,
      dominoSpriteMap: sprite.map,
      dominoesDescription: (typeof gameui !== "undefined" && gameui.gamedatas)
        ? gameui.gamedatas.dominoesDescription
        : null,
      game: gd.game_name || gd.gamename || gd.game || "unknown",
      // gamedatas table id is unreliable on some tables (null) — fall back to
      // the URL (?table=NNN appears on live tables and replays alike), so
      // per-table log files actually separate games.
      tableId: gd.table_id || gd.tableId
        || (location.href.match(/[?&]table=(\d+)/) || [null, null])[1] || null,
      activePlayer: gd.gamestate && gd.gamestate.active_player,
      // The VIEWING player's BGA id — lets the overlay distinguish "my turn"
      // (auto-refresh trigger) from "analyzing the opponent's move".
      viewerId: (typeof gameui.player_id !== "undefined" && gameui.player_id !== null)
        ? String(gameui.player_id)
        : null,
      gamestateName: gd.gamestate && (gd.gamestate.name || gd.gamestate.descriptionmyturn),
      candidateSource: normalized.source || candidate.source,
      state: normalized.state || null,
      normalization: {
        ok: Boolean(normalized.ok),
        error: normalized.error || null,
        debug: normalized.debug || null,
      },
      gamedatasKeys: Object.keys(gd).sort(),
      gamedatas: cloneSafe(gd, 5, []),
      domSamples: sampleDom(),
      capturedAt: new Date().toISOString(),
      url: location.href,
    });
  } catch (e) {
    emit({ ok: false, error: String((e && e.message) || e) });
  }
  return __result;
}

function readPageState() {
  // BGA's Content Security Policy blocks inline <script> injection, so we ask
  // the background service worker to run the page-context reader via
  // chrome.scripting.executeScript (MAIN world), which is not subject to the
  // page CSP. The function body itself is unchanged; it now returns its result
  // instead of emitting a CustomEvent.
  return new Promise((resolve) => {
    const fnSource = pageReadKingdominoState.toString();
    const finish = (resp) => resolve(resp || { ok: false, error: "no response from background worker" });
    try {
      if (usesPromiseAPI) {
        browserAPI.runtime
          .sendMessage({ action: "readPageState", fnSource })
          .then(finish, (err) => resolve({ ok: false, error: String((err && err.message) || err) }));
        return;
      }
      const result = browserAPI.runtime.sendMessage({ action: "readPageState", fnSource }, (resp) => {
        if (browserAPI.runtime.lastError) {
          resolve({ ok: false, error: browserAPI.runtime.lastError.message });
        } else {
          finish(resp);
        }
      });
      if (result && typeof result.then === "function") {
        result.then(finish, (err) => resolve({ ok: false, error: String((err && err.message) || err) }));
      }
    } catch (e) {
      resolve({ ok: false, error: String((e && e.message) || e) });
    }
  });
}

function stableStringify(obj) {
  if (obj === null || typeof obj !== "object") return JSON.stringify(obj);
  if (Array.isArray(obj)) return "[" + obj.map(stableStringify).join(",") + "]";
  return "{" + Object.keys(obj).sort().map((k) => JSON.stringify(k) + ":" + stableStringify(obj[k])).join(",") + "}";
}

function formatPct(x, digits = 1) {
  if (typeof x !== "number" || !Number.isFinite(x)) return "-";
  return (100 * x).toFixed(digits) + "%";
}

function formatValue(x) {
  if (typeof x !== "number" || !Number.isFinite(x)) return "-";
  return x >= -1 && x <= 1 ? x.toFixed(3) : x.toFixed(1);
}

function firstNumber(obj, keys) {
  for (const k of keys) {
    const v = obj && obj[k];
    if (typeof v === "number" && Number.isFinite(v)) return v;
  }
  return null;
}

function isExactResponse(response, payload) {
  const engine = String((response && response.engine) || (payload && payload.engine) || "").toLowerCase();
  return engine.startsWith("exact") || Boolean(response && response.exact && response.exact.solved);
}

function actionText(rec) {
  if (rec.label) return rec.label;
  if (rec.action_id) return rec.action_id;
  if (rec.kind === "pick" && rec.domino_id !== undefined) return `Pick domino ${rec.domino_id}`;
  if (rec.kind === "turn" && rec.domino_id !== undefined) return `Play domino ${rec.domino_id}`;
  return JSON.stringify(rec).slice(0, 140);
}

function metricChip(label, value, title) {
  const chip = document.createElement("span");
  chip.textContent = `${label}: ${value}`;
  chip.title = title || "";
  chip.style.cssText = [
    "display:inline-block",
    "padding:2px 6px",
    "border-radius:999px",
    "background:rgba(15,23,42,0.88)",
    "border:1px solid rgba(71,85,105,0.9)",
    "color:#cbd5e1",
    "font-size:11px",
    "margin:4px 4px 0 0",
    "white-space:nowrap",
  ].join("; ");
  return chip;
}

function bar(widthPct) {
  const outer = document.createElement("div");
  outer.style.cssText = "height:5px;background:rgba(15,23,42,0.95);border-radius:999px;overflow:hidden;margin-top:6px;";
  const inner = document.createElement("div");
  const clamped = Math.max(0, Math.min(100, widthPct));
  inner.style.cssText = `height:100%;width:${clamped}%;background:#38bdf8;border-radius:999px;`;
  outer.appendChild(inner);
  return outer;
}

function dominoTileEl(dominoId, spriteUrl, spriteMap) {
  const wrapper = document.createElement('div');
  wrapper.style.cssText = [
    'display:inline-block',
    'width:80px',
    'height:40px',
    'border-radius:4px',
    'overflow:hidden',
    'flex-shrink:0',
    'background:#1e293b',
    'border:1px solid rgba(71,85,105,0.6)',
    'vertical-align:middle',
  ].join(';');

  const pos = spriteMap && spriteMap[dominoId];
  if (spriteUrl && pos) {
    // Sprite cells are 200x100px displayed; we show at 80x40px (scale 0.4)
    const scale = 0.4;
    const inner = document.createElement('div');
    inner.style.cssText = [
      `width:200px`,
      `height:100px`,
      `background-image:url("${spriteUrl}")`,
      `background-position:${pos[0]}px ${pos[1]}px`,
      `background-repeat:no-repeat`,
      `transform:scale(${scale})`,
      `transform-origin:top left`,
    ].join(';');
    wrapper.appendChild(inner);
  } else {
    // Fallback: show domino number
    wrapper.style.cssText += ';display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:11px;';
    wrapper.textContent = `#${dominoId}`;
  }
  return wrapper;
}

const TERRAIN_COLORS = {
  field: "#e6c34d", wheat: "#e6c34d",
  forest: "#2f6b3a",
  lake: "#3a8ed6", water: "#3a8ed6",
  grassland: "#7cc04a", grass: "#7cc04a",
  swamp: "#6b5535",
  mountain: "#3a3a42", mine: "#3a3a42",
  castle: "#b08d57",
};

function terrainColor(name) {
  return TERRAIN_COLORS[String(name || "").toLowerCase()] || "rgba(148,163,184,0.10)";
}

function miniBoardEl(rec, dominoDesc, placedCells) {
  const placement = parsePlacement(rec);     // {ax,ay,bx,by,flipped} engine coords (castle 7,7)
  const ids = parseDominoIds(rec);
  const CASTLE = 7;
  const cellPx = 16, gap = 2;

  const wrapper = document.createElement("div");
  wrapper.style.cssText = "flex-shrink:0;";
  if (!placement) return wrapper;

  // Terrain for the two halves of the placed domino, mapped to the right cells.
  // desc.left is the engine A-half, desc.right is the B-half. The engine maps
  // halves to cells (see board.py / evaluation.py) as:
  //   h1, h2 = (b, a) if flipped else (a, b)
  // with h1 at (x1,y1)=(ax,ay) and h2 at (x2,y2)=(bx,by). When flipped=True the
  // A and B halves swap cells, so we must branch on placement.flipped here
  // instead of always putting A at (ax,ay).
  let halfA = null, halfB = null;   // halfA -> (ax,ay), halfB -> (bx,by)
  const desc = dominoDesc && ids.placeId != null ? dominoDesc[ids.placeId] : null;
  if (desc) {
    const sideA = desc.left;   // engine domino.a
    const sideB = desc.right;  // engine domino.b
    halfA = placement.flipped ? sideB : sideA;
    halfB = placement.flipped ? sideA : sideB;
  }

  // gather all engine cells we must show: castle, placed tiles, the new domino
  const placed = Array.isArray(placedCells) ? placedCells : [];
  const allX = [CASTLE, placement.ax, placement.bx];
  const allY = [CASTLE, placement.ay, placement.by];
  placed.forEach(c => { allX.push(c.x); allY.push(c.y); });

  let minX = Math.min(...allX), maxX = Math.max(...allX);
  let minY = Math.min(...allY), maxY = Math.max(...allY);

  // 1-cell margin
  minX -= 1; maxX += 1; minY -= 1; maxY += 1;

  // clamp span to at most 7x7, keeping the content roughly centered
  function clampSpan(lo, hi, maxSpan) {
    let span = hi - lo + 1;
    if (span > maxSpan) {
      const center = Math.round((lo + hi) / 2);
      lo = center - Math.floor(maxSpan / 2);
      hi = lo + maxSpan - 1;
    }
    return [lo, hi];
  }
  [minX, maxX] = clampSpan(minX, maxX, 7);
  [minY, maxY] = clampSpan(minY, maxY, 7);

  const cols = maxX - minX + 1;
  const rows = maxY - minY + 1;

  // occupancy maps keyed by engine "x,y"
  const placedMap = {};
  placed.forEach(c => { placedMap[c.x + "," + c.y] = c; });

  const newMap = {};
  newMap[placement.ax + "," + placement.ay] = { terrain: halfA ? halfA.terrain : "field", crowns: halfA ? halfA.crowns : 0 };
  newMap[placement.bx + "," + placement.by] = { terrain: halfB ? halfB.terrain : "field", crowns: halfB ? halfB.crowns : 0 };

  const W = cols * (cellPx + gap), H = rows * (cellPx + gap);
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("width", W);
  svg.setAttribute("height", H);
  svg.style.display = "block";

  for (let gy = 0; gy < rows; gy++) {
    for (let gx = 0; gx < cols; gx++) {
      const ex = minX + gx, ey = maxY - gy;     // engine coords; invert Y: higher engine-y renders toward the top
      const key = ex + "," + ey;
      const px = gx * (cellPx + gap), py = gy * (cellPx + gap);

      let fill = "rgba(148,163,184,0.10)";
      let stroke = "rgba(148,163,184,0.18)";
      let sw = 0.5;
      let crowns = 0;
      let isCastle = (ex === CASTLE && ey === CASTLE);
      let isNew = false;

      if (isCastle) {
        fill = terrainColor("castle"); stroke = "#e9d8a6"; sw = 1;
      } else if (newMap[key]) {
        fill = terrainColor(newMap[key].terrain); stroke = "#f8fafc"; sw = 1.5;
        crowns = newMap[key].crowns || 0; isNew = true;
      } else if (placedMap[key]) {
        const t = (placedMap[key].terrain || "").toString().toLowerCase();
        fill = terrainColor(t); stroke = "rgba(248,250,252,0.35)"; sw = 0.5;
        crowns = placedMap[key].crowns || 0;
      }

      const rect = document.createElementNS(svgNS, "rect");
      rect.setAttribute("x", px); rect.setAttribute("y", py);
      rect.setAttribute("width", cellPx); rect.setAttribute("height", cellPx);
      rect.setAttribute("rx", 2);
      rect.setAttribute("fill", fill);
      rect.setAttribute("stroke", stroke);
      rect.setAttribute("stroke-width", sw);
      // dim already-placed (non-new, non-castle) tiles slightly
      if (!isNew && !isCastle && placedMap[key]) rect.setAttribute("opacity", "0.55");
      svg.appendChild(rect);

      if (isCastle) {
        const t = document.createElementNS(svgNS, "text");
        t.setAttribute("x", px + cellPx/2); t.setAttribute("y", py + cellPx/2 + 4);
        t.setAttribute("font-size", "10"); t.setAttribute("text-anchor", "middle");
        t.setAttribute("fill", "#3d2f12"); t.textContent = "♛";
        svg.appendChild(t);
      } else if (crowns > 0) {
        const t = document.createElementNS(svgNS, "text");
        t.setAttribute("x", px + cellPx/2); t.setAttribute("y", py + cellPx/2 + 4);
        t.setAttribute("font-size", "10"); t.setAttribute("text-anchor", "middle");
        t.setAttribute("fill", "#fff8e1"); t.textContent = crowns > 1 ? String(crowns) : "♔";
        svg.appendChild(t);
      }
    }
  }

  wrapper.appendChild(svg);
  return wrapper;
}

function parseDominoIds(rec) {
  // Extract placed domino id and picked domino id from action fields or label
  let placeId = null;
  let pickId = null;

  // Try structured fields first
  if (rec.domino_id !== undefined) placeId = rec.domino_id;
  if (rec.pick_id !== undefined) pickId = rec.pick_id;

  // Fall back to parsing the label
  if (placeId === null || pickId === null) {
    const label = rec.label || rec.action_id || '';
    const placeMatch = label.match(/[Pp]lace domino (\d+)/);
    const pickMatch = label.match(/[Pp]ick(?:ing)? (\d+)/);
    if (placeMatch) placeId = parseInt(placeMatch[1]);
    if (pickMatch) pickId = parseInt(pickMatch[1]);
  }

  return { placeId, pickId };
}

// parsePlacement and mapHalvesToCells live in placement_mapping.js (loaded
// before this script in manifest.json) so the Node test can exercise the exact
// same logic. They are exposed on globalThis.KingdominoPlacement.
const { parsePlacement, mapHalvesToCells } = globalThis.KingdominoPlacement;

function placementToText(placement) {
  // Convert engine coords (castle at 7,7) to castle-relative description.
  // Returns a short human-readable string like "2 left, 1 up of castle".
  if (!placement) return '';
  const CASTLE = 7;

  function describe(x, y) {
    // engine x increases rightward, y increases downward (BGA convention from notes)
    const dx = x - CASTLE; // + = right
    const dy = y - CASTLE; // + = down
    const parts = [];
    if (dx !== 0) parts.push(`${Math.abs(dx)} ${dx > 0 ? 'right' : 'left'}`);
    if (dy !== 0) parts.push(`${Math.abs(dy)} ${dy > 0 ? 'down' : 'up'}`);
    if (parts.length === 0) return 'castle';
    return parts.join(', ');
  }

  const a = describe(placement.ax, placement.ay);
  const b = describe(placement.bx, placement.by);
  return `${a}  →  ${b}`;
}

function makeOverlayBase(titleText) {
  const existing = document.getElementById(OVERLAY_ID);
  if (existing) existing.remove();

  const box = document.createElement("div");
  box.id = OVERLAY_ID;
  box.style.cssText = [
    "position: fixed",
    "top: 80px",
    "right: 16px",
    "z-index: 999999",
    "width: 410px",
    "max-height: 76vh",
    "overflow: auto",
    "background: rgba(18, 24, 38, 0.97)",
    "color: #f8fafc",
    "border: 1px solid rgba(148, 163, 184, 0.5)",
    "border-radius: 10px",
    "box-shadow: 0 12px 30px rgba(0,0,0,0.35)",
    "font: 13px/1.35 system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
    "padding: 12px",
  ].join("; ");

  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;";

  const title = document.createElement("div");
  title.textContent = titleText;
  title.style.cssText = "font-weight:700;font-size:15px;";

  const close = document.createElement("button");
  close.textContent = "x";
  close.title = "Close";
  close.style.cssText = "background:transparent;color:#cbd5e1;border:0;font-size:18px;cursor:pointer;line-height:1;";
  close.addEventListener("click", () => box.remove());

  header.appendChild(title);
  header.appendChild(close);
  box.appendChild(header);
  return box;
}

async function renderDebugOverlay(capture, message) {
  const box = makeOverlayBase("Kingdomino Advisor Debug");
  const text = document.createElement("div");
  text.style.cssText = "color:#cbd5e1;margin-bottom:10px;";
  text.textContent = message || capture.error || "Captured BGA state, but no normalized Kingdomino engine state was found yet.";
  box.appendChild(text);

  const meta = document.createElement("pre");
  meta.style.cssText = "white-space:pre-wrap;max-height:280px;overflow:auto;background:rgba(15,23,42,0.95);border:1px solid rgba(71,85,105,0.9);border-radius:8px;padding:8px;color:#e2e8f0;font-size:11px;";
  meta.textContent = JSON.stringify({
    ok: capture.ok,
    game: capture.game,
    tableId: capture.tableId,
    activePlayer: capture.activePlayer,
    gamestateName: capture.gamestateName,
    candidateSource: capture.candidateSource,
    gamedatasKeys: capture.gamedatasKeys,
    domSamples: capture.domSamples,
    frame_debug: capture.frame_debug,
    error: capture.error,
  }, null, 2);
  box.appendChild(meta);

  const controls = document.createElement("div");
  controls.style.cssText = "display:flex;gap:8px;margin-top:10px;";

  const copy = document.createElement("button");
  copy.textContent = "Copy capture";
  copy.style.cssText = "flex:1;background:#2563eb;color:white;border:0;border-radius:6px;padding:7px;cursor:pointer;";
  copy.addEventListener("click", async () => {
    await navigator.clipboard.writeText(JSON.stringify(capture, null, 2));
    copy.textContent = "Copied";
  });

  const download = document.createElement("button");
  download.textContent = "Download probe";
  download.style.cssText = "flex:1;background:#166534;color:white;border:0;border-radius:6px;padding:7px;cursor:pointer;";
  download.addEventListener("click", async () => {
    const result = await downloadProbe(buildAdvisorProbe({ capture, reason: "debug" }));
    download.textContent = result.ok ? "Saved" : "Save failed";
    if (!result.ok) download.title = result.error || "download failed";
  });

  const refresh = document.createElement("button");
  refresh.textContent = "Refresh";
  refresh.style.cssText = "flex:1;background:#475569;color:white;border:0;border-radius:6px;padding:7px;cursor:pointer;";
  refresh.addEventListener("click", () => captureDebugOnly());

  controls.appendChild(copy);
  controls.appendChild(download);
  controls.appendChild(refresh);
  box.appendChild(controls);
  document.body.appendChild(box);
}

function renderErrorOverlay(message) {
  const box = makeOverlayBase("Kingdomino Advisor");
  const body = document.createElement("div");
  body.style.cssText = "color:#fecaca;background:rgba(127,29,29,0.55);border:1px solid rgba(248,113,113,0.5);border-radius:8px;padding:8px;";
  body.textContent = message;
  box.appendChild(body);
  document.body.appendChild(box);
}

function renderRecommendations(response, payload, options, transport, spriteUrl, spriteMap, dominoDesc, activeBoardCells, capture) {
  const box = makeOverlayBase("Kingdomino Advisor");

  // Loud failure: an incomplete opponent board reconstruction means the model
  // saw wrong information. Warn the player prominently — first child of box,
  // above controls and recommendations — but still show the recommendations.
  const reconWarn = payload && payload.state && payload.state.board_reconstruction_warning;
  if (reconWarn) {
    const banner = document.createElement("div");
    banner.style.cssText = "background:rgba(127,29,29,0.95);border:1px solid #f87171;border-radius:6px;padding:8px 10px;margin-bottom:8px;color:#fecaca;font-size:12px;line-height:1.4;";
    // Safe: every interpolated value is a number or string produced by our own
    // code (normalizeBgaState), never raw BGA DOM text.
    banner.innerHTML = `⚠️ <b>Opponent board incomplete</b> — missing ${reconWarn.expected_cells - reconWarn.actual_cells} cell(s). Recommendations may be unreliable.<br><span style="color:#fca5a5;font-size:11px;">${reconWarn.message}${reconWarn.skipped_domino_ids.length ? " Skipped tile IDs: " + reconWarn.skipped_domino_ids.join(", ") + "." : ""}</span>`;
    box.insertBefore(banner, box.firstChild);
  }

  const controls = document.createElement("div");
  controls.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:10px;";

  const simSelect = document.createElement("select");
  simSelect.title = "MCTS simulations";
  simSelect.style.cssText = "background:rgba(51,65,85,0.9);color:#e2e8f0;border:1px solid rgba(100,116,139,0.9);border-radius:6px;font-size:12px;cursor:pointer;padding:3px 5px;";
  Array.from(new Set([...SIM_OPTIONS, options.sims])).sort((a, b) => a - b).forEach((n) => {
    const opt = document.createElement("option");
    opt.value = String(n);
    opt.textContent = `${n} sims`;
    opt.selected = n === options.sims;
    simSelect.appendChild(opt);
  });
  simSelect.addEventListener("change", async () => {
    await saveOptionPatch({ sims: Number(simSelect.value) });
    triggerRecommend({ force: true, reason: "sim-change" });
  });

  const deeper = document.createElement("button");
  deeper.textContent = "Think deeper";
  deeper.title = "Increase simulations and recompute";
  deeper.style.cssText = "background:rgba(22,101,52,0.9);color:#dcfce7;border:1px solid rgba(74,222,128,0.75);border-radius:6px;font-size:12px;cursor:pointer;padding:3px 7px;";
  deeper.addEventListener("click", async () => {
    const next = SIM_OPTIONS.find((n) => n > options.sims) || options.sims * 2;
    await saveOptionPatch({ sims: next });
    triggerRecommend({ force: true, reason: "think-deeper" });
  });

  const refresh = document.createElement("button");
  refresh.textContent = "Refresh";
  refresh.title = "Refresh now";
  refresh.style.cssText = "background:rgba(51,65,85,0.9);color:#e2e8f0;border:1px solid rgba(100,116,139,0.9);border-radius:6px;font-size:12px;cursor:pointer;padding:3px 7px;";
  refresh.addEventListener("click", () => triggerRecommend({ force: true, reason: "manual-refresh" }));

  const probe = document.createElement("button");
  probe.textContent = "Download probe";
  probe.title = "Download a JSON debug bundle for local advisor replay";
  probe.style.cssText = "background:rgba(37,99,235,0.9);color:#dbeafe;border:1px solid rgba(96,165,250,0.8);border-radius:6px;font-size:12px;cursor:pointer;padding:3px 7px;";
  probe.addEventListener("click", async () => {
    const result = await downloadProbe(buildAdvisorProbe({
      capture,
      payload,
      response,
      options,
      transport,
      reason: "overlay-download",
    }));
    probe.textContent = result.ok ? "Saved" : "Save failed";
    if (!result.ok) probe.title = result.error || "download failed";
  });

  const autoWrap = document.createElement("label");
  autoWrap.title = "Automatically run the advisor when it becomes your turn";
  autoWrap.style.cssText = "display:flex;align-items:center;gap:4px;color:#cbd5e1;font-size:11px;cursor:pointer;user-select:none;";
  const autoBox = document.createElement("input");
  autoBox.type = "checkbox";
  autoBox.checked = Boolean(options.autoRefresh);
  autoBox.addEventListener("change", async () => {
    await saveOptionPatch({ autoRefresh: autoBox.checked });
  });
  autoWrap.appendChild(autoBox);
  autoWrap.appendChild(document.createTextNode("auto"));

  const logWrap = document.createElement("label");
  logWrap.title = "Passively log every decision state + final result of this game to the local server (runs/kingdomino/bga_game_log/)";
  logWrap.style.cssText = "display:flex;align-items:center;gap:4px;color:#cbd5e1;font-size:11px;cursor:pointer;user-select:none;";
  const logBox = document.createElement("input");
  logBox.type = "checkbox";
  logBox.checked = Boolean(options.gameLog);
  logBox.addEventListener("change", async () => {
    await saveOptionPatch({ gameLog: logBox.checked });
  });
  logWrap.appendChild(logBox);
  logWrap.appendChild(document.createTextNode("log"));

  controls.appendChild(simSelect);
  controls.appendChild(deeper);
  controls.appendChild(refresh);
  controls.appendChild(probe);
  controls.appendChild(autoWrap);
  controls.appendChild(logWrap);
  box.appendChild(controls);

  const meta = document.createElement("div");
  meta.style.cssText = "color:#cbd5e1;margin-bottom:10px;";
  const valueText = typeof response.value === "number" ? formatValue(response.value) : "-";
  const searchMs = response.search_ms !== undefined ? `${response.search_ms}ms` : "-";
  const sims = response.num_simulations !== undefined ? response.num_simulations : options.sims;
  const ri = response.root_inference;
  const exactResponse = isExactResponse(response, payload);
  const rootWinProb = exactResponse && typeof response.root_win_prob === "number"
    ? response.root_win_prob
    : null;
  const rootMarginPts = exactResponse && typeof response.root_margin_pts === "number"
    ? response.root_margin_pts
    : null;
  const marginText = rootMarginPts === null
    ? ""
    : ` · margin <b style="color:${rootMarginPts >= 0 ? "#4ade80" : "#f87171"}">${(rootMarginPts >= 0 ? "+" : "") + rootMarginPts.toFixed(1)}</b>`;
  const valueLine = rootWinProb !== null
    ? `Exact win: <b style="color:#f8fafc">${formatPct(rootWinProb)}</b>${marginText}`
    : `NN edge: <b style="color:#f8fafc">${valueText}</b>`;
  // Flag analyses of the opponent's decision loudly — same engine, same
  // output, just a different actor (state.current_actor is the ACTIVE player).
  const viewerId = capture && capture.viewerId != null ? String(capture.viewerId) : null;
  const capturedActive = capture && capture.activePlayer != null ? String(capture.activePlayer) : null;
  const opponentTurn = viewerId !== null && capturedActive !== null && viewerId !== capturedActive;
  const turnLine = opponentTurn
    ? `<div style="color:#fbbf24;font-weight:600;margin-bottom:3px;">Analyzing OPPONENT's move</div>`
    : "";
  const swindleLine = response.swindle_mode
    ? `<div style="color:#f472b6;font-weight:600;margin-bottom:3px;" title="You are not winning with perfect play. Ranking favors moves that maximize the chance your opponent errs (traps), not minimal losing margin.">SWINDLE MODE${response.swindle_truncated ? " (partial — budget hit)" : ""}</div>`
    : "";
  meta.innerHTML = `${turnLine}${swindleLine}${valueLine}<br>Search: <b style="color:#f8fafc">${searchMs}</b> / <b style="color:#f8fafc">${sims}</b> sims<br>Engine: <b style="color:#f8fafc">${response.engine || payload.engine}</b> · ${transport} · ${new Date().toLocaleTimeString()}`;
  if (response.exact && response.exact.solved) {
    const exact = document.createElement("div");
    exact.style.cssText = "font-size:11px;color:#86efac;margin-top:3px;";
    const hits = Number.isFinite(response.exact.cache_hits) ? response.exact.cache_hits : 0;
    const misses = Number.isFinite(response.exact.cache_misses) ? response.exact.cache_misses : 0;
    exact.textContent = `Exact solved - deck ${response.exact.deck_count} - cache ${hits}/${hits + misses} hits`;
    meta.appendChild(exact);
  }
  if (response.checkpoint_path) {
    const ck = document.createElement("div");
    ck.style.cssText = "font-size:11px;color:#94a3b8;margin-top:3px;word-break:break-all;";
    ck.textContent = response.checkpoint_path;
    meta.appendChild(ck);
  }
  box.appendChild(meta);

  // Root inference summary: the network's pre-search prediction (single forward
  // pass, NOT MCTS-updated). Only NN engines return root_inference.
  const engineName = String(response.engine || payload.engine || "");
  if (ri && engineName.startsWith("nn")) {
    const panel = document.createElement("div");
    panel.style.cssText = "background:rgba(15,23,42,0.9);border:1px solid rgba(71,85,105,0.6);border-radius:6px;padding:6px 10px;margin-bottom:8px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;";
    panel.title = "Network's prediction before MCTS search. Own/Opp are projected final scores.";

    const bold = (text, color) => {
      const b = document.createElement("b");
      b.textContent = text;
      if (color) b.style.color = color;
      return b;
    };
    const stat = (prefix, boldEl) => {
      const span = document.createElement("span");
      span.style.cssText = "color:#94a3b8;font-size:12px;";
      span.appendChild(document.createTextNode(prefix));
      span.appendChild(boldEl);
      return span;
    };

    panel.appendChild(stat("You: ", bold(String(Math.round(ri.own_score_est)))));
    panel.appendChild(stat("Opp: ", bold(String(Math.round(ri.opp_score_est)))));

    const margin = ri.score_margin_est;
    const marginText = (margin >= 0 ? "+" : "") + Math.round(margin);
    panel.appendChild(stat("Margin: ", bold(marginText, margin >= 0 ? "#4ade80" : "#f87171")));

    panel.appendChild(stat("Raw win head: ", bold(formatPct(ri.win_prob), ri.win_prob >= 0.5 ? "#4ade80" : "#f87171")));

    const note = document.createElement("span");
    note.textContent = "(pre-search estimates)";
    note.style.cssText = "color:#475569;font-size:10px;font-style:italic;";
    panel.appendChild(note);

    box.appendChild(panel);
  }

  const recs = Array.isArray(response.recommendations) ? response.recommendations : [];
  const list = document.createElement("div");
  list.style.cssText = "display:flex;flex-direction:column;gap:7px;";

  if (!recs.length) {
    const empty = document.createElement("div");
    empty.textContent = "No recommendations returned.";
    empty.style.cssText = "color:#fca5a5;";
    list.appendChild(empty);
  }

  // Phase + current row come from the captured BGA state (carried on the
  // payload). current_row lets us resolve an INITIAL_SELECTION rec's
  // legal_index (0-based into the row) to a concrete domino id; spriteUrl and
  // spriteMap (already params) render the tile thumbnail.
  const statePhase = (payload && payload.state && payload.state.phase) || null;
  const currentRow = (payload && payload.state && Array.isArray(payload.state.current_row))
    ? payload.state.current_row
    : [];

  const maxVisit = Math.max(...recs.map((r) => firstNumber(r, ["visit_frac", "visit_share", "prob"]) || 0), 0.000001);
  recs.slice(0, options.topK).forEach((rec, idx) => {
    const row = document.createElement("div");
    row.style.cssText = "background:rgba(30,41,59,0.95);border:1px solid rgba(71,85,105,0.8);border-radius:8px;padding:8px;";

    const visit = firstNumber(rec, ["visit_frac", "visit_share", "prob"]);
    const prior = firstNumber(rec, ["prior", "policy_prior", "nn_prior"]);
    const qWinProb = firstNumber(rec, ["q_win_prob"]);
    const qRankValue = firstNumber(rec, ["q_rank_value"]);
    const marginPts = firstNumber(rec, ["exact_margin_pts"]);
    const qEdge = qWinProb !== null ? (qWinProb * 2.0) - 1.0 : null;

    const rowFlex = document.createElement("div");
    rowFlex.style.cssText = "display:flex;align-items:center;gap:10px;";

    const rankEl = document.createElement("div");
    rankEl.textContent = `${idx + 1}.`;
    rankEl.style.cssText = "font-weight:500;color:#94a3b8;min-width:16px;flex-shrink:0;";
    rowFlex.appendChild(rankEl);

    // mini board
    rowFlex.appendChild(miniBoardEl(rec, dominoDesc, activeBoardCells));

    // right column: pick + bar
    const rightCol = document.createElement("div");
    rightCol.style.cssText = "display:flex;flex-direction:column;gap:5px;flex:1;";

    // Action identity row: shows the domino(s) involved with a sprite + label.
    // The previous version always printed a bare "then pick" because
    // parseDominoIds looked for rec.pick_id (the server sends pick_domino_id)
    // and the "Pick domino N" label did not match its pick regex.
    const actionRow = document.createElement("div");
    actionRow.style.cssText = "display:flex;align-items:center;gap:6px;flex-wrap:wrap;";

    if (statePhase === "INITIAL_SELECTION") {
      // Opening selection: rec.legal_index indexes current_row. Fall back to the
      // server-provided rec.domino_id if the row/index is unavailable.
      let dominoId = null;
      if (Number.isFinite(rec.legal_index) && rec.legal_index >= 0 && rec.legal_index < currentRow.length) {
        dominoId = currentRow[rec.legal_index];
      }
      if (dominoId == null && Number.isFinite(rec.domino_id)) dominoId = rec.domino_id;

      const label = document.createElement("span");
      label.style.cssText = "color:#cbd5e1;font-size:12px;";
      label.textContent = dominoId != null ? `pick domino ${dominoId}` : "pick";
      actionRow.appendChild(label);
      if (dominoId != null) actionRow.appendChild(dominoTileEl(dominoId, spriteUrl, spriteMap));
    } else {
      // PLACE_AND_SELECT: place rec.domino_id (sprite + coords), then pick the
      // next domino (rec.pick_domino_id).
      const fallback = parseDominoIds(rec);
      const placedId = Number.isFinite(rec.domino_id) ? rec.domino_id : fallback.placeId;
      const nextPick = Number.isFinite(rec.pick_domino_id) ? rec.pick_domino_id : fallback.pickId;

      const p = rec.placement;
      const coords = p && Number.isFinite(p.x1)
        ? `(${p.x1},${p.y1})→(${p.x2},${p.y2})${p.flipped ? " flipped" : ""}`
        : "";

      const placeLabel = document.createElement("span");
      placeLabel.style.cssText = "color:#cbd5e1;font-size:12px;";
      placeLabel.textContent = placedId != null
        ? `place ${placedId}${coords ? " " + coords : ""}`
        : (coords || "place");
      actionRow.appendChild(placeLabel);
      if (placedId != null) actionRow.appendChild(dominoTileEl(placedId, spriteUrl, spriteMap));

      if (nextPick != null) {
        const thenLabel = document.createElement("span");
        thenLabel.textContent = "then pick";
        thenLabel.style.cssText = "color:#64748b;font-size:11px;";
        actionRow.appendChild(thenLabel);
        actionRow.appendChild(dominoTileEl(nextPick, spriteUrl, spriteMap));
      }
    }
    rightCol.appendChild(actionRow);

    const barRow = document.createElement("div");
    barRow.style.cssText = "display:flex;align-items:center;gap:6px;";
    if (visit !== null) {
      const barWrap = bar((visit / maxVisit) * 100);
      barWrap.style.flex = "1";
      barRow.appendChild(barWrap);
    }
    const pct = document.createElement("span");
    pct.textContent = visit === null ? "" : formatPct(visit, 1);
    pct.style.cssText = "font-weight:500;font-size:13px;min-width:42px;text-align:right;";
    barRow.appendChild(pct);
    rightCol.appendChild(barRow);

    rowFlex.appendChild(rightCol);
    row.appendChild(rowFlex);

    const chips = document.createElement("div");
    chips.style.cssText = "margin-top:2px;";
    // Order: visit (what the search concluded) → prior (initial network belief)
    // → win% (win probability if this move is played).
    if (visit !== null) chips.appendChild(metricChip("visit", formatPct(visit), "MCTS visit share"));
    if (prior !== null) chips.appendChild(metricChip("prior", formatPct(prior), "Network prior (pre-search)"));
    if (qWinProb !== null && exactResponse) chips.appendChild(metricChip("win%", formatPct(qWinProb), "Exact win probability after this move"));
    if (marginPts !== null) chips.appendChild(metricChip("margin", (marginPts >= 0 ? "+" : "") + marginPts.toFixed(1), "Exact expected final-score margin (points) after this move"));
    const sw = rec.swindle;
    if (sw && sw.replies > 0) {
      const flips = (sw.flips_win || 0) + (sw.flips_draw || 0);
      chips.appendChild(metricChip("traps", `${flips}/${sw.replies}`,
        `Opponent replies that improve YOUR outcome: ${sw.flips_win || 0} flip to a win, ${sw.flips_draw || 0} to a draw (of ${sw.replies} legal replies)`));
      if (typeof sw.weighted_rate === "number") {
        chips.appendChild(metricChip("w", formatPct(sw.weighted_rate),
          "Trap probability weighted by the network's policy for the opponent — how likely a natural-looking reply walks into a trap"));
      }
      if (typeof sw.trap_payoff_pts === "number") {
        chips.appendChild(metricChip("if err", (sw.trap_payoff_pts >= 0 ? "+" : "") + sw.trap_payoff_pts.toFixed(1),
          "Your best final-score margin among the opponent's mistaken replies"));
      }
    }
    if (qEdge !== null && !exactResponse) chips.appendChild(metricChip("edge", formatValue(qEdge), "Uncalibrated NN/MCTS value edge after this move"));
    if (qRankValue !== null && marginPts === null) chips.appendChild(metricChip("rank", formatValue(qRankValue), "Exact margin-aware tie-break value"));
    if (chips.children.length) row.appendChild(chips);

    list.appendChild(row);
  });

  box.appendChild(list);
  document.body.appendChild(box);
}

function buildRecommendPayload(state, options) {
  // Note: no channels/blocks/bilinear_dim here. The server reads the model
  // architecture from the checkpoint config, so the extension never sends it.
  const resolvedEngine = advisorEngineForState(state, options.engine);
  const payload = {
    state,
    engine: resolvedEngine,
    requested_engine: options.engine,
    num_simulations: options.sims,
    nn_sims: options.sims,
    top_k: options.topK,
    determinizations: 1,
    temperature: 0.0,
    device: options.device,
    exact_max_secs: options.exactMaxSecs,
    exact_threads: options.exactThreads,
  };
  if (options.checkpoint && options.checkpoint.trim()) {
    payload.checkpoint_path = options.checkpoint.trim();
  }
  return payload;
}

async function postRecommend(payload) {
  try {
    const response = await fetch(ADVISOR_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${JSON.stringify(data)}`);
    return { data, transport: "direct" };
  } catch (directError) {
    console.warn("[kingdomino-advisor] direct fetch failed; trying background fallback:", directError);
    const wrapped = await new Promise((resolve) => {
      try {
        if (usesPromiseAPI) {
          browserAPI.runtime.sendMessage({ action: "recommend", url: ADVISOR_URL, payload }).then(
            resolve,
            (err) => resolve({ ok: false, error: String((err && err.message) || err) })
          );
          return;
        }
        const result = browserAPI.runtime.sendMessage({ action: "recommend", url: ADVISOR_URL, payload }, resolve);
        if (result && typeof result.then === "function") {
          result.then(resolve, (err) => resolve({ ok: false, error: String((err && err.message) || err) }));
        }
      } catch (e) {
        resolve({ ok: false, error: String((e && e.message) || e) });
      }
    });
    console.log("[advisor] background response:", JSON.stringify(wrapped));
    if (!wrapped || !wrapped.ok) {
      console.error("[advisor] background fetch failed:", wrapped && wrapped.error);
      throw new Error((wrapped && wrapped.error) || String((directError && directError.message) || directError));
    }
    return { data: wrapped.data, transport: "background" };
  }
}

async function captureDebugOnly() {
  const capture = await readPageState();
  const probe = buildAdvisorProbe({ capture, reason: "debug" });
  await setStorage({
    kingdomino_last_capture: capture,
    kingdomino_last_probe: probe,
  });
  await renderDebugOverlay(capture);
  return { ok: Boolean(capture.ok), capture };
}

async function triggerRecommend({ force = false, reason = "manual" } = {}) {
  if (inFlightRecommend) return { ok: false, skipped: true, error: "recommendation already in flight" };
  const now = Date.now();
  if (!force && now - lastStartedAt < 1000) return { ok: false, skipped: true, error: "throttled" };

  inFlightRecommend = true;
  lastStartedAt = now;
  try {
    const options = await loadOptions();
    const capture = await readPageState();
    await setStorage({ kingdomino_last_capture: capture });

    const spriteUrl = capture.spriteUrl || null;
    const spriteMap = capture.dominoSpriteMap || null;
    const dominoDesc = capture.dominoesDescription || null;
    const activeBoardCells =
      (capture.state && Array.isArray(capture.state.boards) &&
       capture.state.boards[capture.state.current_actor] &&
       Array.isArray(capture.state.boards[capture.state.current_actor].cells))
        ? capture.state.boards[capture.state.current_actor].cells
        : [];

    if (!capture.ok) {
      await renderDebugOverlay(capture, capture.error || "Could not capture BGA state.");
      return { ok: false, error: capture.error || "capture failed", capture };
    }

    if (!capture.state) {
      await renderDebugOverlay(
        capture,
        "Captured BGA data, but no normalized engine state was found yet. Send the copied capture or relevant BGA HTML so the scraper can be mapped."
      );
      return { ok: false, skipped: true, error: "no normalized Kingdomino state yet", capture };
    }

    const payload = buildRecommendPayload(capture.state, options);
    const key = stableStringify(payload);
    if (!force && key === lastPayloadKey) {
      return { ok: false, skipped: true, error: "same decision state", capture, payload };
    }
    lastPayloadKey = key;

    const { data, transport } = await postRecommend(payload);
    const probe = buildAdvisorProbe({
      capture,
      payload,
      response: data,
      options,
      transport,
      reason,
    });
    await setStorage({
      kingdomino_last_payload: payload,
      kingdomino_last_recommendation: data,
      kingdomino_last_transport: transport,
      kingdomino_last_recommendation_at: new Date().toISOString(),
      kingdomino_last_probe: probe,
    });
    // Log a compact summary of what the user was SHOWN, so post-mortems can
    // compare in-game advice against later recomputation (different nets,
    // sims, seeds otherwise make that impossible to reconstruct).
    if (options.gameLog) {
      const recs = Array.isArray(data.recommendations) ? data.recommendations : [];
      postGameLog(capture.tableId, {
        schema: "kingdomino-bga-gamelog/v1",
        kind: "advisor",
        captured_at: capture.capturedAt || new Date().toISOString(),
        table_id: capture.tableId != null ? String(capture.tableId) : null,
        gamestate_name: capture.gamestateName != null ? String(capture.gamestateName) : null,
        active_player: capture.activePlayer != null ? String(capture.activePlayer) : null,
        viewer_id: capture.viewerId != null ? String(capture.viewerId) : null,
        advisor: {
          engine: data.engine || payload.engine,
          sims: data.num_simulations,
          checkpoint: data.checkpoint_path || null,
          value: typeof data.value === "number" ? data.value : null,
          root_win_prob: typeof data.root_win_prob === "number" ? data.root_win_prob : null,
          root_margin_pts: typeof data.root_margin_pts === "number" ? data.root_margin_pts : null,
          swindle_mode: Boolean(data.swindle_mode),
          top: recs.slice(0, 3).map((r) => ({
            domino_id: r.domino_id, placement: r.placement || null,
            pick_domino_id: r.pick_domino_id,
            q_win_prob: r.q_win_prob, visit_frac: r.visit_frac,
            exact_margin_pts: r.exact_margin_pts,
          })),
        },
      });
    }
    renderRecommendations(data, payload, options, transport, spriteUrl, spriteMap, dominoDesc, activeBoardCells, capture);
    return { ok: true, transport, capture, payload, response: data, reason };
  } catch (e) {
    const error = String((e && e.message) || e);
    renderErrorOverlay(error);
    return { ok: false, error };
  } finally {
    inFlightRecommend = false;
  }
}

browserAPI.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message && message.action === "capture") {
    triggerRecommend({ force: true, reason: "popup" }).then(sendResponse);
    return true;
  }
  if (message && message.action === "debugCapture") {
    captureDebugOnly().then(sendResponse);
    return true;
  }
  if (message && message.action === "downloadLastProbe") {
    getStorage(["kingdomino_last_probe"]).then((stored) => {
      const probe = stored.kingdomino_last_probe;
      if (!probe) {
        sendResponse({ ok: false, error: "no advisor probe has been captured yet" });
        return;
      }
      downloadProbe(probe).then(sendResponse);
    }, (err) => sendResponse({ ok: false, error: String((err && err.message) || err) }));
    return true;
  }
  return false;
});

// ── Poller: auto-refresh on your turn + passive game logging ────────────────
// One capture per tick serves both features. Both use a two-poll stability
// rule before acting — BGA's tile animations produce transitional captures,
// so acting on the first sighting risks a half-updated board.
const AUTO_POLL_MS = 2500;
let lastAutoStateKey = null;
let lastLoggedKey = null;
let pendingLogKey = null;

function postGameLog(tableId, record) {
  // Fire-and-forget through the background worker's JSON-POST proxy; logging
  // must never break the page or the advisor.
  const message = {
    action: "recommend",
    url: GAME_LOG_URL,
    payload: { table_id: tableId == null ? null : String(tableId), record },
  };
  try {
    if (usesPromiseAPI) {
      browserAPI.runtime.sendMessage(message).catch(() => {});
      return;
    }
    browserAPI.runtime.sendMessage(message, () => {
      void browserAPI.runtime.lastError;
    });
  } catch (e) {
    /* ignore */
  }
}

function extractFinalResult(capture) {
  const gd = capture && capture.gamedatas;
  if (!gd || !gd.players) return null;
  const players = {};
  Object.keys(gd.players).forEach((id) => {
    const p = gd.players[id] || {};
    players[String(id)] = {
      name: p.name != null ? String(p.name) : null,
      score: p.score != null ? Number(p.score) : null,
    };
  });
  if (!Object.keys(players).length) return null;
  return {
    players,
    playerorder: Array.isArray(gd.playerorder) ? gd.playerorder.map(String) : null,
  };
}

function gameLogTick(capture) {
  // Decision states: every NEW normalized state (either player's turn).
  // Final results: once, when BGA reaches its end-of-game state (the scraper
  // reports those as unsupported gamestates, but the raw scores are present).
  const meta = {
    schema: "kingdomino-bga-gamelog/v1",
    captured_at: capture.capturedAt || new Date().toISOString(),
    url: capture.url || location.href,
    table_id: capture.tableId != null ? String(capture.tableId) : null,
    gamestate_name: capture.gamestateName != null ? String(capture.gamestateName) : null,
    active_player: capture.activePlayer != null ? String(capture.activePlayer) : null,
    viewer_id: capture.viewerId != null ? String(capture.viewerId) : null,
  };
  let key = null;
  let record = null;
  if (capture.state) {
    key = "state:" + stableStringify(capture.state);
    record = { ...meta, kind: "decision", state: capture.state };
  } else if (/gameEnd|endGame|gameResult|finalScoring|end$/i.test(String(capture.gamestateName || ""))) {
    const final = extractFinalResult(capture);
    if (final) {
      key = "final:" + stableStringify(final);
      record = { ...meta, kind: "final", final };
    }
  }
  if (!record || key === lastLoggedKey) return;
  if (key !== pendingLogKey) {
    pendingLogKey = key; // first sighting — wait one poll for stability
    return;
  }
  lastLoggedKey = key;
  postGameLog(meta.table_id, record);
}

async function pollTick() {
  try {
    if (inFlightRecommend) return;
    const options = await loadOptions();
    if (!options.autoRefresh && !options.gameLog) return;
    const capture = await readPageState();
    if (!capture || !capture.ok) return;
    if (options.gameLog) gameLogTick(capture);
    if (!options.autoRefresh || !capture.state) return;
    const viewerId = capture.viewerId != null ? String(capture.viewerId) : null;
    const active = capture.activePlayer != null ? String(capture.activePlayer) : null;
    if (!viewerId || !active || viewerId !== active) return;
    const key = stableStringify(capture.state);
    if (key !== lastAutoStateKey) {
      lastAutoStateKey = key; // first sighting — wait one poll for stability
      return;
    }
    await triggerRecommend({ reason: "auto-turn" });
  } catch (e) {
    // The poller must never take down the content script.
    console.warn("[kingdomino-advisor] poll tick failed:", e);
  }
}
setInterval(pollTick, AUTO_POLL_MS);
