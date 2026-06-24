// placement_mapping.js — pure placement→display logic shared by the content
// script and the Node test. No DOM and no browser APIs, so it can run headless.
//
// Loaded as a classic content script (listed BEFORE content.js in manifest.json,
// so it shares the same scope) where it exposes `globalThis.KingdominoPlacement`.
// In Node it is a CommonJS module: `const { parsePlacement } = require(...)` or,
// from ESM, `import KingdominoPlacement from "../placement_mapping.js"`.
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;             // Node (CommonJS; ESM default-imports this)
  } else {
    root.KingdominoPlacement = api;   // browser content-script scope
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  // Parse a recommendation row into engine placement cells + the flipped flag.
  //
  //   (ax,ay) is the engine's first cell (x1,y1); (bx,by) is the second (x2,y2).
  //   `flipped` decides which domino half lands on which cell — see
  //   mapHalvesToCells. We prefer the structured placement object (it carries
  //   flipped reliably) and fall back to parsing the label text, where flipped
  //   shows up as a trailing word:
  //     "Place domino 1 at (6,7) -> (6,6) flipped; pick 20"  (arrow -> or →)
  function parsePlacement(rec) {
    const p = rec && rec.placement;
    if (p && p.x1 !== undefined && p.x1 !== null) {
      return {
        ax: parseInt(p.x1), ay: parseInt(p.y1),
        bx: parseInt(p.x2), by: parseInt(p.y2),
        flipped: !!p.flipped,
      };
    }
    const label = (rec && (rec.label || rec.action_id) ||
      (typeof actionText === "function" ? actionText(rec) : "")) || "";
    const m = label.match(/\((\d+)\s*,\s*(\d+)\)\s*(?:->|→)\s*\((\d+)\s*,\s*(\d+)\)/);
    if (!m) return null;
    return {
      ax: parseInt(m[1]), ay: parseInt(m[2]),
      bx: parseInt(m[3]), by: parseInt(m[4]),
      flipped: /\bflipped\b/i.test(label),
    };
  }

  // Map the two domino halves onto their engine cells, honoring `flipped`.
  //
  //   sideA is the engine A-half (desc.left / domino.a); sideB is the B-half
  //   (desc.right / domino.b). The engine maps halves to cells (see board.py /
  //   evaluation.py) as:
  //       h1, h2 = (b, a) if flipped else (a, b)
  //   with h1 at (x1,y1)=(ax,ay) and h2 at (x2,y2)=(bx,by). So when flipped is
  //   true the A and B halves swap cells. Returns an object keyed by the engine
  //   "x,y" string, each value being the side object that belongs on that cell.
  function mapHalvesToCells(placement, sideA, sideB) {
    const halfAtCell1 = placement.flipped ? sideB : sideA; // -> (ax,ay)=(x1,y1)
    const halfAtCell2 = placement.flipped ? sideA : sideB; // -> (bx,by)=(x2,y2)
    return {
      [placement.ax + "," + placement.ay]: halfAtCell1,
      [placement.bx + "," + placement.by]: halfAtCell2,
    };
  }

  // ── DOM opponent-board reconstruction (pure core) ──────────────────────────
  // BGA renders each placed tile in the DOM with a `shadow-rotation-N` class and
  // a pixel `style.left/top` offset from the castle (50px per cell). These pure
  // helpers turn that anchor + rotation into the two engine cells; the content
  // script's MAIN-world buildBoardFromDom inlines the identical logic (it cannot
  // import this module across the world boundary), and the Node test exercises
  // THIS copy. Keep the two in lockstep.

  // Quarter-turn offset from the anchor half to the second half, as engine
  // (dx,dy). The sequence is right→up→left→down for rotation 0→1→2→3, matching
  // BGA's shadow-rotation-N. Returns null for an unknown rotation so callers can
  // skip the tile instead of guessing.
  function domRotationOffset(rotation) {
    switch (rotation) {
      case 0: return [1, 0];   // → desc.right to the right
      case 1: return [0, -1];  // ↑ desc.right above
      case 2: return [-1, 0];  // ← desc.right to the left
      case 3: return [0, 1];   // ↓ desc.right below
      default: return null;
    }
  }

  // Reconstruct a placed domino's two engine cells from its castle-relative grid
  // anchor (anchorGx,anchorGy = style.left/50, style.top/50) and BGA rotation.
  //
  // VERIFIED against the 2026-06-23 live authoritative board: the style anchor
  // is the desc.LEFT half for ALL rotations, and desc.right extends by
  // domRotationOffset(rotation). (An earlier inferred table that used desc.right
  // as the anchor for rotations 2–3 placed crowns on the wrong cell — domino 25,
  // rotation 2, must put forest+1crown on the FAR cell, not the castle-adjacent
  // one.) Returns [{x,y,side}, {x,y,side}] in engine coords, or null if the
  // rotation is unknown.
  function domDominoCells(anchorGx, anchorGy, rotation, descLeft, descRight, castlePos) {
    const off = domRotationOffset(rotation);
    if (!off) return null;
    const cx = castlePos[0], cy = castlePos[1];
    return [
      { x: cx + anchorGx, y: cy + anchorGy, side: descLeft },                 // anchor = desc.left
      { x: cx + anchorGx + off[0], y: cy + anchorGy + off[1], side: descRight }, // second = desc.right
    ];
  }

  return { parsePlacement, mapHalvesToCells, domRotationOffset, domDominoCells };
});
