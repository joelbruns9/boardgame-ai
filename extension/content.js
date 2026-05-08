// content.js
// This file runs automatically on every BGA page.
// Think of it as our note-taker sitting at the table.

// ---- COMPATIBILITY NOTE ----
// Chrome uses "chrome.storage" — Firefox uses "browser.storage"
// This line makes "browser" work in both:
const browserAPI = typeof browser !== "undefined" ? browser : chrome;

// ---- STEP 1: READ THE GAME STATE ----
function captureGameState() {

  // Check we're actually in a game (gameui won't exist on non-game pages)
  if (typeof gameui === "undefined") {
    console.log("BGA Capture: not on a game page, skipping.");
    return null;
  }

  // Grab the raw data from BGA
  const raw = gameui.gamedatas;

  // ---- STEP 2: CLEAN IT UP ----
  // We only keep what's useful — remember our schema design
  const state = {

    // Which game is this? (cantstop, kingdomino etc in future)
    game: document.title,

    // When did we capture this?
    timestamp: new Date().toISOString(),

    // What decision does the active player face right now?
    phase: raw.gamestate.name,

    // Whose turn is it?
    active_player: String(raw.gamestate.active_player),

    // What did the dice show?
    dice: raw.gamestate.args ? raw.gamestate.args.dice : [],

    // What moves can they legally make?
    possible_moves: raw.gamestate.args ? raw.gamestate.args.possibleMoves : [],

    // Which columns has each player claimed?
    columns_claimed: raw.columns || [],

    // What are the current scores?
    scores: Object.fromEntries(
      Object.values(raw.players).map(p => [p.player_id, p.score])
    ),

    // What move did the player actually take? (filled in later)
    action_taken: null

  };

  return state;
}

// ---- STEP 3: SAVE IT ----
function saveGameState() {
  const state = captureGameState();

  if (!state) return; // nothing to save

  // Load existing saved states first
  browserAPI.storage.local.get("captured_states", function(result) {

    // If no saves yet, start with empty list
    const existing = result.captured_states || [];

    // Add new state to the list
    existing.push(state);

    // Save back to storage
    browserAPI.storage.local.set({ captured_states: existing }, function() {
      console.log("BGA Capture: state saved!", state);
    });

  });
}

// ---- STEP 4: LISTEN FOR INSTRUCTIONS ----
// Our popup button will send a "capture" message when clicked
// This code listens for that message and triggers the save
browserAPI.runtime.onMessage.addListener(function(message) {
  if (message.action === "capture") {
    saveGameState();
  }
});