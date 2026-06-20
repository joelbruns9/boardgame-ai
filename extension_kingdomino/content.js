const browserAPI = typeof browser !== "undefined" ? browser : chrome;
const usesPromiseAPI = typeof browser !== "undefined" && browserAPI === browser;

const ADVISOR_URL = "http://127.0.0.1:8000/api/recommend";
const RESULT_EVENT = "kingdomino-advisor-state";
const OVERLAY_ID = "kingdomino-advisor-overlay";

const DEFAULT_OPTIONS = {
  engine: "nn",
  sims: 800,
  topK: 8,
  checkpoint: "",
  device: "cuda",
  channels: 32,
  blocks: 4,
  bilinearDim: 64,
};

const SIM_OPTIONS = [100, 200, 400, 800, 1600, 3200];

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
  ]);
  const sims = Number(stored.kingdomino_sims);
  const topK = Number(stored.kingdomino_top_k);
  return {
    engine: stored.kingdomino_engine || DEFAULT_OPTIONS.engine,
    sims: Number.isFinite(sims) && sims > 0 ? Math.round(sims) : DEFAULT_OPTIONS.sims,
    topK: Number.isFinite(topK) && topK > 0 ? Math.round(topK) : DEFAULT_OPTIONS.topK,
    checkpoint: stored.kingdomino_checkpoint || DEFAULT_OPTIONS.checkpoint,
    device: stored.kingdomino_device || DEFAULT_OPTIONS.device,
    channels: DEFAULT_OPTIONS.channels,
    blocks: DEFAULT_OPTIONS.blocks,
    bilinearDim: DEFAULT_OPTIONS.bilinearDim,
  };
}

async function saveOptionPatch(patch) {
  const out = {};
  if (patch.engine !== undefined) out.kingdomino_engine = patch.engine;
  if (patch.sims !== undefined) out.kingdomino_sims = patch.sims;
  if (patch.checkpoint !== undefined) out.kingdomino_checkpoint = patch.checkpoint;
  if (patch.topK !== undefined) out.kingdomino_top_k = patch.topK;
  if (patch.device !== undefined) out.kingdomino_device = patch.device;
  await setStorage(out);
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
      bounds: {
        xMin: asInt(kingdom.xMin, null),
        xMax: asInt(kingdom.xMax, null),
        yMin: asInt(kingdom.yMin, null),
        yMax: asInt(kingdom.yMax, null),
      },
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
    const hiddenDeck = allDominoIds
      .filter((n) => !visibleSet[n])
      .sort((a, b) => a - b);

    const debug = {
      source: "bga.gameui.gamedatas",
      bga_state_name: gameStateName,
      bga_active_player: activePlayer == null ? null : String(activePlayer),
      bga_playerorder: playerorder,
      bga_locations: Object.keys(dominoes).reduce((acc, k) => {
        const loc = String(dominoes[k].location || "UNKNOWN").toUpperCase();
        acc[loc] = (acc[loc] || 0) + 1;
        return acc;
      }, {}),
      bga_current_domino: activeDomino,
      bga_current_position: gd.gamestate && gd.gamestate.args ? gd.gamestate.args.currentPosition : null,
      kingdom_summary: summarizeKingdom(gd.gamestate && gd.gamestate.args && gd.gamestate.args.kingdom),
      deck: hiddenDeck,
      notes: [],
    };

    if (activeIndex === undefined) {
      debug.notes.push("active BGA player is not in playerorder; defaulting start/current player to 0");
    }

    let phase = null;
    let currentRow = [];
    let pendingClaims = [];
    let nextClaims = [];
    let actorIndex = 0;
    let initialPickCount = 0;
    let startPlayer = activeIndex === undefined ? 0 : activeIndex;
    const boardBuild = buildBoardsFromDominoes(dominoes, dominoesDescription, playerToIndex, players, canvasSize);

    // Authoritative per-cell board reconstruction, applied symmetrically to
    // EVERY player. For each player we prefer BGA's kingdom grid (exact terrain
    // and crowns per cell); only where no authoritative grid is available do we
    // keep the approximate dominoes reconstruction built above. Historically only
    // the active player's board was upgraded this way, leaving the opponent on
    // the fragile dominoes path — this loop closes that gap.
    for (let p = 0; p < boardBuild.boards.length; p++) {
      const grid = resolveKingdomForPlayer(gd, p, playerorder, activeIndex);
      if (grid) {
        boardBuild.boards[p] = buildBoardFromKingdomArg(grid, canvasSize);
        debug.notes.push(`Player ${p} board built from authoritative BGA kingdom grid.`);
      } else {
        debug.notes.push(`Player ${p} board uses APPROXIMATE dominoes reconstruction (no authoritative kingdom grid available${p === activeIndex ? "" : "; BGA typically exposes the kingdom grid for the active player only"}).`);
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
      phase = "PLACE_AND_SELECT";
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
      debug.notes.push("Mapped BGA placeDomino to engine PLACE_AND_SELECT using CURRENT owned dominoes as remaining pending claims.");
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
      return;
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
      tableId: gd.table_id || gd.tableId || null,
      activePlayer: gd.gamestate && gd.gamestate.active_player,
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

  const refresh = document.createElement("button");
  refresh.textContent = "Refresh";
  refresh.style.cssText = "flex:1;background:#475569;color:white;border:0;border-radius:6px;padding:7px;cursor:pointer;";
  refresh.addEventListener("click", () => captureDebugOnly());

  controls.appendChild(copy);
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

function renderRecommendations(response, payload, options, transport, spriteUrl, spriteMap, dominoDesc, activeBoardCells) {
  const box = makeOverlayBase("Kingdomino Advisor");

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

  controls.appendChild(simSelect);
  controls.appendChild(deeper);
  controls.appendChild(refresh);
  box.appendChild(controls);

  const meta = document.createElement("div");
  meta.style.cssText = "color:#cbd5e1;margin-bottom:10px;";
  const valueText = typeof response.value === "number" ? formatValue(response.value) : "-";
  const searchMs = response.search_ms !== undefined ? `${response.search_ms}ms` : "-";
  const sims = response.num_simulations !== undefined ? response.num_simulations : options.sims;
  meta.innerHTML = `Value: <b style="color:#f8fafc">${valueText}</b><br>Search: <b style="color:#f8fafc">${searchMs}</b> / <b style="color:#f8fafc">${sims}</b> sims<br>Engine: <b style="color:#f8fafc">${response.engine || payload.engine}</b> · ${transport} · ${new Date().toLocaleTimeString()}`;
  if (response.checkpoint_path) {
    const ck = document.createElement("div");
    ck.style.cssText = "font-size:11px;color:#94a3b8;margin-top:3px;word-break:break-all;";
    ck.textContent = response.checkpoint_path;
    meta.appendChild(ck);
  }
  box.appendChild(meta);

  const recs = Array.isArray(response.recommendations) ? response.recommendations : [];
  const list = document.createElement("div");
  list.style.cssText = "display:flex;flex-direction:column;gap:7px;";

  if (!recs.length) {
    const empty = document.createElement("div");
    empty.textContent = "No recommendations returned.";
    empty.style.cssText = "color:#fca5a5;";
    list.appendChild(empty);
  }

  const maxVisit = Math.max(...recs.map((r) => firstNumber(r, ["visit_frac", "visit_share", "prob"]) || 0), 0.000001);
  recs.slice(0, options.topK).forEach((rec, idx) => {
    const row = document.createElement("div");
    row.style.cssText = "background:rgba(30,41,59,0.95);border:1px solid rgba(71,85,105,0.8);border-radius:8px;padding:8px;";

    const visit = firstNumber(rec, ["visit_frac", "visit_share", "prob"]);
    const prior = firstNumber(rec, ["prior", "policy_prior", "nn_prior"]);
    const value = firstNumber(rec, ["q_value", "q", "Q", "value", "win_prob"]);
    const visits = firstNumber(rec, ["visit_count", "visits", "N"]);

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

    const { pickId } = parseDominoIds(rec);
    const pickRow = document.createElement("div");
    pickRow.style.cssText = "display:flex;align-items:center;gap:6px;";
    const pickLabel = document.createElement("span");
    pickLabel.textContent = "then pick";
    pickLabel.style.cssText = "color:#64748b;font-size:11px;";
    pickRow.appendChild(pickLabel);
    if (pickId) pickRow.appendChild(dominoTileEl(pickId, spriteUrl, spriteMap));
    rightCol.appendChild(pickRow);

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
    if (visit !== null) chips.appendChild(metricChip("visit", formatPct(visit), "MCTS visit share"));
    if (prior !== null) chips.appendChild(metricChip("prior", formatPct(prior), "NN policy prior"));
    if (value !== null) chips.appendChild(metricChip("value", formatValue(value), "Estimated action/root value"));
    if (visits !== null) chips.appendChild(metricChip("N", String(Math.round(visits)), "Raw visit count"));
    if (rec.legal_index !== undefined) chips.appendChild(metricChip("legal", String(rec.legal_index), "Legal action index"));
    if (chips.children.length) row.appendChild(chips);

    list.appendChild(row);
  });

  box.appendChild(list);
  document.body.appendChild(box);
}

function buildRecommendPayload(state, options) {
  const payload = {
    state,
    engine: options.engine,
    num_simulations: options.sims,
    nn_sims: options.sims,
    top_k: options.topK,
    determinizations: 1,
    temperature: 0.0,
    device: options.device,
    channels: options.channels,
    blocks: options.blocks,
    bilinear_dim: options.bilinearDim,
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
    if (!wrapped || !wrapped.ok) {
      throw new Error((wrapped && wrapped.error) || String((directError && directError.message) || directError));
    }
    return { data: wrapped.data, transport: "background" };
  }
}

async function captureDebugOnly() {
  const capture = await readPageState();
  await setStorage({ kingdomino_last_capture: capture });
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
    await setStorage({
      kingdomino_last_payload: payload,
      kingdomino_last_recommendation: data,
      kingdomino_last_transport: transport,
      kingdomino_last_recommendation_at: new Date().toISOString(),
    });
    renderRecommendations(data, payload, options, transport, spriteUrl, spriteMap, dominoDesc, activeBoardCells);
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
  return false;
});