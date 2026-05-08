// content.js
const browserAPI = typeof browser !== "undefined" ? browser : chrome;

// ---- THE KEY FIX ----
// Instead of reading gameui directly (we can't, different sandbox)
// we inject a script into the PAGE itself, which CAN read gameui
function readGameState() {
  return new Promise((resolve) => {

    // Create a script element that runs IN the page's environment
    const script = document.createElement("script");
    script.id = "bga-capture-script";

    script.textContent = `
      (function() {
        if (typeof gameui === "undefined" || !gameui.gamedatas) {
          window.dispatchEvent(new CustomEvent("bga-capture-result", {
            detail: { error: "gameui not found" }
          }));
          return;
        }

        const raw = gameui.gamedatas;

        const state = {
          game: "cantstop",
          timestamp: new Date().toISOString(),
          phase: raw.gamestate.name,
          active_player: String(raw.gamestate.active_player),
          dice: raw.gamestate.args ? raw.gamestate.args.dice : [],
          possible_moves: raw.gamestate.args ? raw.gamestate.args.possibleMoves : [],
          columns_claimed: raw.columns || [],
          scores: Object.fromEntries(
            Object.values(raw.players).map(p => [p.player_id, p.score])
          ),
          action_taken: null
        };

        window.dispatchEvent(new CustomEvent("bga-capture-result", {
          detail: { state: state }
        }));
      })();
    `;

    // Listen for the result coming back from the page
    window.addEventListener("bga-capture-result", function handler(event) {
      window.removeEventListener("bga-capture-result", handler);
      script.remove();

      if (event.detail.error) {
        console.log("BGA Capture: error -", event.detail.error);
        resolve(null);
      } else {
        console.log("BGA Capture: got state!", event.detail.state);
        resolve(event.detail.state);
      }
    });

    // Inject the script into the page
    document.head.appendChild(script);
  });
}

// ---- SAVE THE GAME STATE ----
async function saveGameState() {
  const state = await readGameState();
  if (!state) {
    console.log("BGA Capture: nothing to save");
    return;
  }

  browserAPI.storage.local.get("captured_states", function(result) {
    const existing = result.captured_states || [];
    existing.push(state);
    browserAPI.storage.local.set({ captured_states: existing }, function() {
      console.log("BGA Capture: saved! Total states:", existing.length);
    });
  });
}

// ---- LISTEN FOR CAPTURE BUTTON ----
console.log("BGA Capture: content.js loaded and ready");

browserAPI.runtime.onMessage.addListener(function(message) {
  if (message.action === "capture") {
    console.log("BGA Capture: capture triggered");
    saveGameState();
  }
});