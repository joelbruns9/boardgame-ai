// BGA -> advisor: browser-side capture for the Welcome To table page.
//
// This is the only code that runs in the page. It holds no *rules* knowledge:
// it grabs `window.gameui.gamedatas`, re-reads the three fields BGA never
// refreshes, and hands the lot to Python, where `bga_extract.py` does every
// piece of interpretation. A BGA UI change then breaks a selector loudly here
// instead of producing a plausible-but-wrong board.
//
// FRESHNESS -- READ THIS.
// `gameui.gamedatas` is the page-load payload, patched by whichever
// notification handlers bother to write back into it. In this game:
//
//   FRESH   players[].scoreSheet, planValidations  (notif_updatePlayersData,
//           States/TurnTrait.js:55-56, at stApplyTurn)
//   FRESH   gamestate.name / .args  (BGA's own setupPrivateState,
//           modules/js/Game/game.js:114-119 -- these states are `type=>private`,
//           so gamestate.name really does hold the private state name)
//   STATIC  options, planCards
//   ABSENT  me_id -- 7WD has one, this game does not; the seat comes from
//           `gameui.player_id` instead (see `seatOf`)
//   STALE   constructionCards, turn, cardsLeft  (notif_newCards touches only
//           the DOM: TurnTrait.js:20-25)
//
// The three stale ones are re-read from the DOM by `captureTablePatch`.
//
// TWO THINGS GAMEDATAS NEVER HAD
// ------------------------------
// 1. YOUR CURRENT TURN. Everything you scribble mid-turn goes to the DOM alone
//    (`notif_addScribble`, `addHouseNumber`). That is a feature, not a bug --
//    the un-updated `scoreSheet` is exactly the turn-start snapshot the engine
//    calls `public_sheets` -- but your own marks have to be read back off the
//    sheet. `captureTurnMarks` does that.
//
// 2. THE DISCARD PILE. `cardsLeft` says how many cards have gone, never which.
//    A human counts them by watching; so do we. In standard mode BGA never
//    removes a construction card from its stack div (wtoConstructionCards.js
//    `newTurn` appends and re-stacks by z-index), so the whole history of the
//    table is still in the DOM -- and each card carries its own database id, so
//    a ledger keyed by that id counts every card exactly once with no
//    reasoning about turns at all. It is mirrored into localStorage under the
//    table id so an F5 does not throw the history away.
//
// Usage from a content script / userscript / devtools:
//   const state = captureForAdvisor();
//   fetch("http://127.0.0.1:8001/api/recommend", { ... body: {state} ... })

// All same-origin frames that own a Welcome To `gameui`, shallowest first.
// Marker is `constructionCards`, which `getAllDatas` sends in every state.
// ONE GLOBAL, ON PURPOSE.
//
// Every BGA advisor extension injects its capture script into the SAME page, and
// a `function foo()` at the top level of a classic script becomes `window.foo`.
// extension_7wd/bga_snippet.js declares `findGameWindow`, `findGameWindows`,
// `seatOf`, `_required` and `captureForAdvisor` too, so whichever content script
// ran second used to silently overwrite the other's -- and since this file's
// `findGameWindow` looks for `constructionCards`, it threw on a 7 Wonders Duel
// table and that panel simply went quiet. Both content scripts match
// `*://*.boardgamearena.com/*`, so this happened on every table of either game
// whenever both extensions were loaded.
//
// So nothing here is global except `window.__WTO_ADVISOR__`. page_bridge.js
// reaches the capture functions through that, and test_extension_assets.py
// fails if any two extensions in this repo ever share a page-world name again.

(() => {
  "use strict";

  function findGameWindows() {
    const found = [];
    const walk = (w, depth) => {
      try {
        if (w.gameui && w.gameui.gamedatas && w.gameui.gamedatas.constructionCards) {
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

  // Which seat a frame is playing.
  //
  // `gameui.player_id` is the BGA FRAMEWORK's field (bga-framework.d.ts:890) and
  // is what welcometo.js itself uses everywhere (welcometo.js:92, 103, 142).
  //
  // It is NOT `gamedatas.me_id`: that is a field a game's own `getAllDatas`
  // chooses to send, 7 Wonders Duel sends one and Welcome To does not
  // (welcometo.game.php:104-118). Reading it here produced exactly one symptom --
  // "this frame is not seated at the table (me_id missing)" on a perfectly normal
  // table -- so the two fallbacks below are kept for other games' conventions
  // rather than because this one needs them.
  //
  // A spectator has no player_id, which is the same "not seated" answer by a
  // different route; `isSpectator` is checked first so the message can say so.
  function seatOf(frame) {
    const w = frame.win || frame;
    const ui = w.gameui || {};
    if (ui.isSpectator) return "";
    const g = ui.gamedatas || {};
    const id = ui.player_id || g.me_id || g.current_player_id;
    return id === undefined || id === null ? "" : String(id);
  }

  // The frame for the seat we are advising.
  //
  // A table can expose more than one `gameui`: a nested `...&testuser=<id>` frame
  // renders another seat with its own `player_id`. Advising the wrong seat is not
  // cosmetic here -- the position IS one player's sheet -- so ambiguity is
  // reported rather than resolved by guessing.
  function findGameWindow() {
    const all = findGameWindows();
    if (!all.length) {
      throw new Error("Welcome To gamedatas not found; open the game table first");
    }
    const real = all.filter((f) => !/[?&]testuser=/.test(f.url));
    if (!real.length && !window.WTO_ALLOW_TESTUSER) {
      const err = new Error(
        "every Welcome To frame on this page is a testuser= impersonation (" +
          all.length +
          " found); refusing rather than advise a seat that is not yours. Set " +
          "window.WTO_ALLOW_TESTUSER = true to override."
      );
      err.wtoAmbiguous = true;
      throw err;
    }
    const candidates = real.length ? real : all;

    const seats = new Set(candidates.map(seatOf));
    if (seats.size > 1) {
      const err = new Error(
        "several Welcome To frames render different seats (" +
          [...seats].join(", ") +
          ") and none is marked testuser=; cannot tell which seat to advise"
      );
      err.wtoAmbiguous = true;
      throw err;
    }

    const chosen = candidates[0];
    const g = chosen.win.gameui.gamedatas;
    const me = seatOf(chosen);
    if (!me || !Object.prototype.hasOwnProperty.call(g.players || {}, me)) {
      const err = new Error(
        "this frame is not seated at the table (player_id " +
          (me || "missing") +
          "); a spectator sees every sheet but has no turn to advise"
      );
      err.wtoAmbiguous = true;
      throw err;
    }
    return chosen.win;
  }

  // Throw rather than return a partial capture: a missing selector must fail
  // loudly, because the alternative is a board that is quietly one turn old.
  function _required(doc, selector) {
    const node = doc.querySelector(selector);
    if (!node) throw new Error("Welcome To DOM patch: selector not found: " + selector);
    return node;
  }

  // ---------------------------------------------------------------------------
  // gamedatas, slimmed
  // ---------------------------------------------------------------------------
  // Only the fields bga_extract reads. A diet, not a filter: sending the whole
  // object also works, and the Python side accepts either.
  function captureGamedatas(w) {
    w = w || findGameWindow();
    const g = w.gameui.gamedatas;
    const players = {};
    for (const [pid, p] of Object.entries(g.players || {})) {
      players[pid] = {
        id: p.id,
        no: p.no,
        name: p.name,
        score: p.score,
        scoreSheet: {
          houses: (p.scoreSheet && p.scoreSheet.houses) || [],
          scribbles: (p.scoreSheet && p.scoreSheet.scribbles) || [],
        },
      };
    }
    return JSON.parse(
      JSON.stringify({
        // The wire calls it me_id because that is what the Python mapper reads for
        // every game; where it comes from is this file's business (see seatOf).
        me_id: seatOf(w),
        turn: g.turn,
        cardsLeft: g.cardsLeft,
        options: g.options,
        players,
        constructionCards: g.constructionCards,
        planCards: g.planCards,
        planValidations: g.planValidations,
        gamestate: {
          name: (g.gamestate || {}).name,
          args: (g.gamestate || {}).args || {},
        },
      })
    );
  }

  // ---------------------------------------------------------------------------
  // The table, re-read from the DOM
  // ---------------------------------------------------------------------------
  // Card holders are appended to their stack in chronological order and are never
  // removed in standard mode, so the last two children of a stack are the pair in
  // play: `[aside, top]`. That is the same order `getTopOf(..., 2)` returns
  // (state DESC), which is the order ConstructionCards::getCombination reads --
  // the aside card supplies the EFFECT, the top card supplies the NUMBER.
  function _cardsIn(doc, stack) {
    return [
      ..._required(doc, "#construction-cards-stack-" + stack).querySelectorAll(
        ".construction-card-holder"
      ),
    ];
  }

  function _cardRow(node) {
    const number = parseInt(node.getAttribute("data-number"), 10);
    const action = parseInt(node.getAttribute("data-action"), 10);
    if (!isFinite(number) || !isFinite(action)) {
      throw new Error(
        "Welcome To DOM patch: unreadable construction card " + node.id
      );
    }
    return { id: node.id.replace("construction-card-", ""), number, action };
  }

  function captureTablePatch(w) {
    const doc = (w || findGameWindow()).document;

    const constructionCards = [];
    for (let stack = 0; stack < 3; stack++) {
      const holders = _cardsIn(doc, stack);
      if (holders.length < 2) {
        throw new Error(
          "Welcome To DOM patch: stack " +
            stack +
            " shows " +
            holders.length +
            " card(s); a standard stack shows two"
        );
      }
      constructionCards.push(holders.slice(-2).map(_cardRow));
    }

    // `data-turn` on #game_play_area is written at setup and by notif_newCards
    // (welcometo.js:84, TurnTrait.js:22), which is the whole reason it is trusted
    // over gamedatas.turn.
    const turn = parseInt(
      _required(doc, "#game_play_area").getAttribute("data-turn"),
      10
    );
    if (!isFinite(turn)) {
      throw new Error("Welcome To DOM patch: #game_play_area has no data-turn");
    }

    // The counter shows cards-per-stack (`parseInt(cardsLeft / 3)`). Standard
    // mode only ever draws three at a time from a deck that starts at 81, so the
    // division is exact and multiplying back loses nothing.
    const perStack = parseInt(
      _required(doc, "#cards-count-status").textContent.trim(),
      10
    );
    if (!isFinite(perStack)) {
      throw new Error("Welcome To DOM patch: #cards-count-status is unreadable");
    }

    return { constructionCards, turn, cardsLeft: perStack * 3 };
  }

  // ---------------------------------------------------------------------------
  // The card ledger
  // ---------------------------------------------------------------------------
  // What a counting player knows: every card face that has passed under their
  // nose. Keyed by BGA's own card id, so a card seen on ten consecutive captures
  // is still one card and no turn arithmetic is needed to avoid double-counting.
  //
  // Mirrored into localStorage per table because the DOM history only goes back
  // to the last page load, and joining or reloading mid-game is exactly when the
  // ledger would otherwise be shortest. Python reports the shortfall against
  // `cardsLeft` as a warning rather than pretending the count is exact.
  const WTO_LEDGER_PREFIX = "wto_advisor_ledger_";

  function _ledgerKey(w) {
    const match = /[?&]table=(\d+)/.exec(String(w.location.href));
    return WTO_LEDGER_PREFIX + (match ? match[1] : "unknown");
  }

  function _readLedger(w) {
    try {
      const blob = JSON.parse(w.localStorage.getItem(_ledgerKey(w)) || "null");
      if (blob && typeof blob === "object" && blob.cards) return blob;
    } catch (e) {
      /* private window, cleared storage, quota: an empty ledger is safe */
    }
    return { cards: {}, cardsLeft: null };
  }

  // `cardsLeft` is the reshuffle detector, and it has to be one: the first player
  // to finish a City Plan may shuffle the discard back into the deck, after which
  // a card in the ledger is no longer "gone" -- it is back in the draw pile, and
  // may be dealt a second time under the same id. Nothing else in the capture says
  // that happened, but the deck count going UP says it unambiguously.
  //
  // The reset keeps only what is on the table, which is what the ledger would hold
  // if the game had just started. That undercounts the discard by the pair the
  // reshuffle itself pushed there (ConstructionCards::reshuffle draws, then
  // discards), and Python turns that into a stated warning rather than a silent
  // three-card lie.
  function mergeLedger(w, cardsLeft) {
    const doc = w.document;
    const blob = _readLedger(w);
    if (blob.cardsLeft !== null && cardsLeft > blob.cardsLeft) {
      blob.cards = {};
    }
    for (let stack = 0; stack < 3; stack++) {
      for (const node of _cardsIn(doc, stack)) {
        const card = _cardRow(node);
        blob.cards[card.id] = [card.number, card.action];
      }
    }
    blob.cardsLeft = cardsLeft;
    try {
      w.localStorage.setItem(_ledgerKey(w), JSON.stringify(blob));
    } catch (e) {
      /* storage is a convenience; losing it only costs ledger accuracy */
    }
    return Object.values(blob.cards);
  }

  // ---------------------------------------------------------------------------
  // The payload the host's {"bga": ...} branch expects
  // ---------------------------------------------------------------------------
  function captureForAdvisor() {
    const w = findGameWindow();
    const gamedatas = captureGamedatas(w);
    const patch = captureTablePatch(w);
    Object.assign(gamedatas, patch);
    return {
      bga: gamedatas,
      dom: captureTurnMarks(w, String(gamedatas.me_id), gamedatas.turn),
      seen: mergeLedger(w, patch.cardsLeft),
    };
  }

  // ---------------------------------------------------------------------------
  // Your current turn, off the sheet
  // ---------------------------------------------------------------------------
  // Every mark BGA draws carries `data-turn`, and every one of them is placed
  // INSIDE the zone div whose id encodes what and where it is
  // (welcometo_welcometo.tpl:264-304). So the id of the parent is the address and
  // no coordinate has to be inferred from geometry.
  //
  // Marks are keyed by that address (scribbles by their own database id), which
  // also de-duplicates the second copy of the sheet the overview modal renders.
  function _zoneAddress(node) {
    // "<pid>_<type>_<x>[_<y>]"; a zone type never contains an underscore.
    const parts = String((node && node.id) || "").split("_");
    if (parts.length < 3) return null;
    return {
      pid: parts[0],
      type: parts[1],
      x: parseInt(parts[2], 10),
      y: parts.length > 3 ? parseInt(parts[3], 10) : null,
    };
  }

  function captureTurnMarks(w, pid, turn) {
    const doc = (w || findGameWindow()).document;
    const houses = new Map();
    const scribbles = new Map();

    // Numbers written this turn. textContent is "7", or "7b" for a bis, because
    // addHouseNumber appends a <span>b</span> (wtoScoreSheet.js:413).
    for (const node of doc.querySelectorAll('.house-number[data-turn="' + turn + '"]')) {
      const at = _zoneAddress(node.parentNode);
      if (!at || at.pid !== pid || at.type !== "house") continue;
      const digits = /-?\d+/.exec(node.textContent || "");
      if (!digits) continue;
      houses.set(at.x + "," + at.y, {
        x: at.x,
        y: at.y,
        number: parseInt(digits[0], 10),
        isBis: !!node.querySelector("span"),
        turn: turn,
      });
    }

    // A roundabout is a house mark with no number of its own; 100 is the ROUNDABOUT
    // sentinel shared by constants.inc.php and constants.py.
    for (const node of doc.querySelectorAll(
      '.scribble-roundabout[data-turn="' + turn + '"]'
    )) {
      const at = _zoneAddress(node.parentNode);
      if (!at || at.pid !== pid || at.type !== "house") continue;
      houses.set(at.x + "," + at.y, {
        x: at.x,
        y: at.y,
        number: 100,
        isBis: false,
        turn: turn,
      });
    }

    for (const node of doc.querySelectorAll('[id^="scribble-"][data-turn]')) {
      if (node.getAttribute("data-turn") !== String(turn)) continue;
      const at = _zoneAddress(node.parentNode);
      if (!at || at.pid !== pid) continue;
      scribbles.set(node.id, { type: at.type, x: at.x, y: at.y, turn: turn });
    }

    return { houses: [...houses.values()], scribbles: [...scribbles.values()] };
  }

  // A signature that changes exactly when the position does, so the panel can tell
  // "the board moved" from "the same board, polled again". Cheap on purpose: it
  // runs on a timer and must not walk the whole sheet.
  function turnSignature(w) {
    try {
      const g = (w || findGameWindow()).gameui.gamedatas;
      const state = String((g.gamestate || {}).name || "");
      const doc = (w || findGameWindow()).document;
      const turn = _required(doc, "#game_play_area").getAttribute("data-turn");
      const marks = doc.querySelectorAll('[data-turn="' + turn + '"]').length;
      const selected = JSON.stringify(((g.gamestate || {}).args || {}).selectedCards);
      return [state, turn, marks, selected].join("|");
    } catch (e) {
      return null;
    }
  }


  // The only thing this file puts on `window`.
  window.__WTO_ADVISOR__ = {
    findGameWindows,
    findGameWindow,
    seatOf,
    captureGamedatas,
    captureTablePatch,
    captureTurnMarks,
    mergeLedger,
    captureForAdvisor,
    turnSignature,
  };
})();
