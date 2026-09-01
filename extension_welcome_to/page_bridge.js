// MAIN-world half of the extension. Runs *in the page*, so it can see
// `window.gameui`; a content script cannot (isolated world, and
// `wrappedJSObject` is Firefox-only).
//
// It does two things and nothing else:
//   1. watches for "there is a decision in front of me the advisor understands"
//   2. on demand, captures the board and postMessage()s it to the content script
//
// It deliberately does NOT fetch. The advisor runs on http://127.0.0.1 while the
// page is https, and a page-context fetch to a plain-http origin is mixed
// content. The isolated content script has host_permissions and is not subject
// to that, so all network access lives there.
//
// Capture functions come from bga_snippet.js, loaded into the page immediately
// before this file. That file is the single source of truth for how a Welcome To
// board is read; see test_extension_assets.py, which fails if the copy here
// drifts from games/welcome_to/bga_snippet.js.
//
// They are reached through `window.__WTO_ADVISOR__` rather than as bare globals,
// because every BGA advisor extension injects into the same page and the plain
// names collide -- see the note at the top of bga_snippet.js.

(() => {
  "use strict";

  const TAG = "wto-advisor";

  // The private states the advisor has something to say at. Mirrors
  // PHASE_OF_STATE in bga_extract.py; anything else is skipped rather than sent,
  // so the Python side never has to reject a routine transition.
  //
  // Welcome To's turn states are `type => private`, so BGA's own
  // setupPrivateState writes the private state name into
  // `gamedatas.gamestate.name` (modules/js/Game/game.js:114). Reading that field
  // is therefore correct and needs no special case -- but it also means the name
  // is *ours*: `waitOthers` means we are done, not that the table is.
  const ADVISABLE = new Set([
    "chooseCards",
    "buildRoundabout",
    "writeNumber",
    "actionSurveyor",
    "actionEstate",
    "actionPark",
    "actionPool",
    "actionBis",
    "choosePlan",
    "validatePlan",
    "askReshuffle",
  ]);

  const post = (type, payload) =>
    window.postMessage({ __wto: TAG, type, payload }, window.location.origin);

  // bga_snippet.js is injected immediately before this file, but "immediately
  // before" is an ordering guarantee about the tags, not about execution on a
  // slow load -- so every use is guarded rather than assumed.
  const capture = () => window.__WTO_ADVISOR__ || null;

  function gameWindow() {
    // findGameWindow() walks from window.top and skips `testuser=` frames, so
    // every copy of this bridge (it runs in each frame) resolves to the same
    // one. Only the frame that IS that window proceeds, so we act once.
    const api = capture();
    if (!api) return null;
    try {
      return api.findGameWindow();
    } catch (e) {
      // "gamedatas not found" is the ordinary answer in every frame that is not
      // the board, and this bridge runs in all of them -- staying quiet there is
      // correct. An ambiguous or unseated board is different: swallowing it
      // leaves the panel dead with no reason given.
      if (e && e.wtoAmbiguous && window === window.top) {
        post("capture_error", { signature: null, message: String(e.message) });
      }
      return null;
    }
  }

  function currentState(w) {
    const g = w.gameui && w.gameui.gamedatas;
    return String(((g || {}).gamestate || {}).name || "");
  }

  // Settling, and why it is two ticks rather than a fixed delay.
  //
  // BGA animates a written number into its box and a scribble onto its zone, so
  // a capture taken mid-flight sees fewer marks than the state machine has
  // already moved past -- which the Python replay correctly refuses as
  // inconsistent. `turnSignature` counts this turn's marks, so it changes while
  // the animation lands and stops changing when it is done. Waiting for two
  // equal readings is therefore a settle test rather than a guess at a duration,
  // and it costs one tick on a board that was never animating.
  let lastSignature = null; // last signature we OBSERVED
  let stableTicks = 0;
  let capturedSignature = null; // last signature we actually sent

  function tick() {
    const w = gameWindow();
    if (!w || w !== window) return; // another frame owns the board

    if (!ADVISABLE.has(currentState(w))) {
      lastSignature = null;
      stableTicks = 0;
      if (capturedSignature !== null) {
        capturedSignature = null;
        post("idle", { state: currentState(w) });
      }
      return;
    }

    const signature = capture().turnSignature(w);
    if (signature === null) return;
    if (signature !== lastSignature) {
      lastSignature = signature;
      stableTicks = 0;
      return;
    }
    stableTicks += 1;
    if (stableTicks < 1) return;
    if (signature === capturedSignature) return;

    capturedSignature = signature;
    try {
      post("position", { signature, ...capture().captureForAdvisor() });
    } catch (err) {
      capturedSignature = null; // let the next tick retry
      post("capture_error", { signature, message: String(err && err.message) });
    }
  }

  const TICK_MS = 700;
  setInterval(tick, TICK_MS);
  setTimeout(tick, 900);

  window.addEventListener("message", (event) => {
    const data = event.data;
    if (!data || data.__wto !== TAG || data.type !== "recapture") return;
    capturedSignature = null; // force the next tick to re-send
    lastSignature = null;
    tick();
  });
})();
