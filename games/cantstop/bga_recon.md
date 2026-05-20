# BGA Can't Stop — State Extraction Recon

Notes from inspecting `gameui` and the rendered board on Board Game Arena's
Can't Stop, to support the local advisor extension. Recorded 2026-05-20.

## TL;DR

State for the advisor comes from two places:

- **Decision state** (whose turn, current phase, dice, legal moves) — read
  from `gameui.gamedatas`.
- **Board state** (runners, saved progress, claimed columns) — scraped from
  the DOM. **Not present in `gamedatas` at all.**

Player index mapping (BGA player ID → 0/1) is via `gamedatas.playerorder`.
Marker ownership is encoded as a `color_<hex>` CSS class on each marker
element. Disambiguation between runners / saved progress / claims uses
the color class plus `data-height`.

## Architecture Notes

- Content scripts on BGA pages can't see `gameui` directly (isolated
  world). The script-injection trick in the current `content.js` is
  required: inject a `<script>` element into the page, dispatch a
  `CustomEvent` back to the content script with results. Keep that
  pattern.
- Decision-state polling: the cheapest way to know "is it my turn, is
  it a diceChoice?" is to read `gamedatas.gamestate.name` and
  `.active_player` either via polling (~300 ms) or a MutationObserver
  on the dice area. Polling is simpler; revisit if it feels laggy.

## Game State Machine

`gamedatas.gamestate.id` and `.name` reveal where in the turn we are.
State IDs from `gamedatas.gamestates`:

| ID  | Name              | Type         | Meaning                                                  |
|-----|-------------------|--------------|----------------------------------------------------------|
| 1   | `gameSetup`       | manager      | Initial setup                                            |
| 2   | `nextPlayer`      | game         | Server transition between players                        |
| 3   | `continueChoice`  | activeplayer | "Stop or keep rolling?"                                  |
| 4   | `saveProgress`    | game         | Server commits runners → progress                        |
| 5   | `diceChoice`      | activeplayer | **"Pick a pair." Advisor should recommend in this state.** |
| 7   | `endTurn`         | game         | Server transition                                        |
| 8   | `failConfirm`     | activeplayer | Bust — confirm turn end                                  |
| 10  | `diceRoll`        | game         | Server rolling dice                                      |
| 99  | `gameEnd`         | manager      | Game over                                                |

Only `activeplayer` states are decision points; the others are
server-only and transient. We query the advisor during `diceChoice`
(state 5). At `continueChoice` (state 3) the dice from the previous
roll are already consumed and `gamestate.args.dice` is empty there.

## `gameui.gamedatas` — What's There

Top-level keys observed:

| Key                      | Use                                                                                                     |
|--------------------------|---------------------------------------------------------------------------------------------------------|
| `players`                | Per-player metadata: `player_id`, `score`, `color` (hex without `#`), `name`. Keyed by BGA player ID.    |
| `playerorder`            | Array of BGA player IDs in seating order. **Index in this array IS the advisor's 0/1 player index.**     |
| `gamestate`              | Live state: `id`, `name`, `active_player`, `args` (phase-dependent, see below).                          |
| `gamestates`             | Static state machine definition. Not needed at runtime.                                                  |
| `required_column_count`  | Win condition (3 in standard rules).                                                                     |
| `movement_variant`       | Rule variant flag.                                                                                       |
| `notifications`          | Packet sequencing — not useful directly.                                                                 |
| `columns`                | **Empty even mid-game. Do NOT rely on this.** Misleadingly named.                                        |

### `gamestate.args` during `diceChoice`

| Field           | Type     | Use                                                                                                                   |
|-----------------|----------|-----------------------------------------------------------------------------------------------------------------------|
| `dice`          | `int[4]` | The 4 rolled dice values.                                                                                             |
| `possibleMoves` | `array`  | BGA's enumeration of legal pair partitions. Each entry has `0`, `1`, `both` describing which side(s) are movable. Useful as a cross-check against `engine.py`'s legal-move generation. |

Verified empirically: even after several moves played, `gamedatas`
schema does not gain new top-level fields. The only parts that update
dynamically are `gamestate.*` (current phase, args, etc.) and
`players[id].score`.

## Board State — DOM Scraping

`gamedatas` does NOT contain runners, saved progress, or claimed
columns. These are read from the DOM.

### Marker Pattern

Every placed marker on the board is rendered as:

```html
<div id="token_<id>_<col>"
     class="tokenspace token color_<hex>"
     data-column="<col>"
     data-height="<h>"
     style="...">
</div>
```

Empty positions on the board are `<div class="tokenspace" ...>`
without the `token` class. They can be ignored for state extraction.

### Disambiguation Table

| `data-height` | Color class                 | Interpretation                                  |
|---------------|-----------------------------|-------------------------------------------------|
| `>= 1`        | `color_000000` (black)      | Active runner this turn (shared, no owner)      |
| `>= 1`        | `color_<player_color>`      | Saved progress for that player                  |
| `== 0`        | `color_<player_color>`      | Claimed column by that player                   |

The color hex matches `gamedatas.players[id].color` exactly, so the
mapping is direct.

### `data-height` Semantics

1-indexed from the base. `data-height=1` = first space above the
column's base; `data-height=N` = top of an N-space column.
`data-height=0` is overloaded as the "claimed" sentinel.

Standard Can't Stop column lengths (top space = max `data-height`):

| Cols   | Length |
|--------|--------|
| 2, 12  | 3      |
| 3, 11  | 5      |
| 4, 10  | 7      |
| 5, 9   | 9      |
| 6, 8   | 11     |
| 7      | 13     |

### ID Format Variants (Reference Only)

Element IDs follow `token_<X>_<col>` where `X` varies by marker type.
We don't need to parse IDs (column comes from `data-column`, owner
from color class), but for reference:

| Marker type     | `X` value                                            |
|-----------------|------------------------------------------------------|
| Active runner   | `0`, `1`, or `2` — which of the 3 runner pieces       |
| Saved progress  | An opaque marker UUID (e.g., `84634030`)             |
| Claim           | The owning player's BGA ID                           |

## Player Mapping Rule

```js
const playerorder = gameui.gamedatas.playerorder; // [bgaId0, bgaId1]
const players = gameui.gamedatas.players;

// Hex color (no '#') → advisor index (0 or 1)
const colorToIndex = {};
playerorder.forEach((bgaId, idx) => {
  colorToIndex[players[bgaId].color] = idx;
});
```

`playerorder` entries are numbers; `players` is keyed by strings. The
example above relies on JavaScript's loose property access; if it
breaks under stricter rules use `players[String(bgaId)]`.

## Full Extraction Logic

```js
function readBoardState() {
  const playerorder = gameui.gamedatas.playerorder;
  const players = gameui.gamedatas.players;

  const colorToIndex = {};
  playerorder.forEach((bgaId, idx) => {
    colorToIndex[players[bgaId].color] = idx;
  });

  const runners = {};            // {col: height}
  const progress = [{}, {}];     // per player: {col: height}
  const claimed = [[], []];      // per player: [col, col, ...]

  document.querySelectorAll('.tokenspace.token').forEach(el => {
    const col = +el.dataset.column;
    const height = +el.dataset.height;
    const colorClass = [...el.classList].find(c => c.startsWith('color_'));
    const color = colorClass && colorClass.slice('color_'.length);

    if (color === '000000') {
      runners[col] = height;
    } else if (color in colorToIndex) {
      const p = colorToIndex[color];
      if (height === 0) claimed[p].push(col);
      else              progress[p][col] = height;
    }
  });

  const gs = gameui.gamedatas.gamestate;
  const activeIdx = playerorder.indexOf(+gs.active_player);

  return {
    active_player: activeIdx,
    scores: playerorder.map(id => +players[id].score),
    dice: gs.args?.dice || [],
    phase: gs.name,
    runners,
    progress,
    claimed
  };
}
```

This produces the JSON shape the advisor server's `/recommend`
endpoint consumes. Field names should be reconciled with the
server contract before wiring up (the server may expect `claimed`
as a list-of-lists or list-of-arrays; verify when integrating).

## Inspected Examples

Reference samples from the recon, in case any pattern needs
re-verification later.

**Active runner on column 2, position 1:**

```html
<div id="token_0_2" class="tokenspace token color_000000"
     data-column="2" data-height="1"
     style="top: 289px; left: 70px; z-index: 1;"></div>
```

**Saved progress, Big-Tuna (`ff0000`) on column 8, position 6:**

```html
<div id="token_84634030_8" class="tokenspace token color_ff0000"
     data-column="8" data-height="6"
     style="top: 332px; left: 333px; z-index: 1;"></div>
```

**Claimed column 8 by RollwJoel (`0000ff`):**

```html
<div id="token_89146710_8" class="tokenspace token color_0000ff"
     data-column="8" data-height="0"
     style="top: 76px; left: 333px; z-index: 1;"></div>
```

## Open Questions / TODOs

- **Confirm `data-height` semantics empirically.** During recon,
  "5 rungs up" was reported but `data-height=6` was found. Probably a
  counting-convention mismatch on my side, but worth verifying with a
  known-position marker before we trust the extraction in production.
- **Bust handling.** When the active player busts and enters
  `failConfirm` (state 8), do the runners on the board disappear
  immediately or persist until `failConfirm` is acknowledged?
  Determines whether we query in state 8 or just skip it.
- **Turn-change detection.** Cheapest reliable trigger for "re-query
  advisor": MutationObserver on the dice area, polling
  `gamestate.name` every 250–500 ms, or hooking into BGA's
  `notifqueue`. Polling is simplest.
- **CORS for `localhost:8765`.** Advisor server already sets
  permissive CORS headers; verify in the integration round.
- **2+ player games.** All recon was done on a 2-player bot game.
  In a 3- or 4-player game, `playerorder` will have more entries
  and the advisor's `[player0, player1]` data structures need to
  generalize or we restrict the extension to 2-player tables only.