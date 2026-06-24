// Node test for the extension's placement→display logic. No DOM, no test
// framework — plain `node:assert`. Run with:
//     node extension_kingdomino/tests/test_placement_mapping.mjs
// Exits 0 on success, nonzero (assert throws) on failure.
//
// This imports the SAME module the content script uses (placement_mapping.js),
// so it exercises the real shipped logic rather than a copy.
//
// NOTE on side labels: in dominoes.py, domino 37 is a=WATER(lake), b=GRASS
// (grassland+crown). The extension feeds desc.left -> sideA (engine domino.a)
// and desc.right -> sideB (engine domino.b). So for domino 37:
//     sideA = WATER   (A-half / desc.left)
//     sideB = GRASS   (B-half / desc.right)
// The engine maps halves to cells as  h1,h2 = (b,a) if flipped else (a,b)
// with h1 at (x1,y1)=(ax,ay), h2 at (x2,y2)=(bx,by). The two orientations of
// the audited (10,5)/(10,4) placement are written out explicitly below.

import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { createRequire } from "node:module";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { parsePlacement, mapHalvesToCells, domRotationOffset, domDominoCells } = require(
  resolve(__dirname, "..", "placement_mapping.js")
);

// Domino 37 halves as the extension sees them (engine A/B order).
const WATER = { terrain: "WATER", crowns: 0 }; // domino.a / desc.left
const GRASS = { terrain: "GRASS", crowns: 1 }; // domino.b / desc.right

function terrainAt(map, x, y) {
  const half = map[x + "," + y];
  return half ? half.terrain : null;
}

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log(`  ok - ${name}`);
}

console.log("placement_mapping: mapHalvesToCells");

// ── The audited domino-37 case: flipped=true. This is the engine's actual legal
//    placement (GRASS half adjacent to grassland at (10,6)). The flipped-swap
//    bug used to render these two terrains on the wrong cells. ───────────────
check("domino 37, flipped=true -> GRASS@(10,5), WATER@(10,4)", () => {
  const placement = { ax: 10, ay: 5, bx: 10, by: 4, flipped: true };
  const map = mapHalvesToCells(placement, WATER, GRASS);
  assert.equal(terrainAt(map, 10, 5), "GRASS", "expected GRASS at (10,5)");
  assert.equal(terrainAt(map, 10, 4), "WATER", "expected WATER at (10,4)");
});

// ── Same cells/sides, flipped=false: the opposite orientation. Documents that
//    the mapping is NOT a constant — it must depend on flipped. ──────────────
check("domino 37, flipped=false -> WATER@(10,5), GRASS@(10,4)", () => {
  const placement = { ax: 10, ay: 5, bx: 10, by: 4, flipped: false };
  const map = mapHalvesToCells(placement, WATER, GRASS);
  assert.equal(terrainAt(map, 10, 5), "WATER", "expected WATER at (10,5)");
  assert.equal(terrainAt(map, 10, 4), "GRASS", "expected GRASS at (10,4)");
});

// ── Guard against the regression in the literal terms of the previous bug: a
//    flipped placement must differ from the unflipped one on both cells. ─────
check("flipped vs unflipped differ on both cells", () => {
  const base = { ax: 10, ay: 5, bx: 10, by: 4 };
  const flipped = mapHalvesToCells({ ...base, flipped: true }, WATER, GRASS);
  const unflipped = mapHalvesToCells({ ...base, flipped: false }, WATER, GRASS);
  assert.notEqual(terrainAt(flipped, 10, 5), terrainAt(unflipped, 10, 5));
  assert.notEqual(terrainAt(flipped, 10, 4), terrainAt(unflipped, 10, 4));
});

console.log("placement_mapping: parsePlacement");

// ── parsePlacement reads flipped from the structured placement object. ───────
check("parsePlacement reads flipped from structured placement", () => {
  const rec = { placement: { x1: 10, y1: 5, x2: 10, y2: 4, flipped: true } };
  const p = parsePlacement(rec);
  assert.deepEqual(p, { ax: 10, ay: 5, bx: 10, by: 4, flipped: true });
});

// ── parsePlacement falls back to the label, picking up the "flipped" word. ───
check("parsePlacement reads flipped from label text", () => {
  const rec = { label: "Place domino 37 at (10,5) → (10,4) flipped; pick 20" };
  const p = parsePlacement(rec);
  assert.deepEqual(p, { ax: 10, ay: 5, bx: 10, by: 4, flipped: true });
});

check("parsePlacement label without 'flipped' -> flipped=false", () => {
  const rec = { label: "Place domino 1 at (6,7) -> (6,6); pick 20" };
  const p = parsePlacement(rec);
  assert.deepEqual(p, { ax: 6, ay: 7, bx: 6, by: 6, flipped: false });
});

// ── End-to-end: parse a structured rec, then map halves, for the bug case. ───
check("parse + map end-to-end (domino 37 flipped) -> GRASS@(10,5)", () => {
  const rec = { placement: { x1: 10, y1: 5, x2: 10, y2: 4, flipped: true } };
  const p = parsePlacement(rec);
  const map = mapHalvesToCells(p, WATER, GRASS);
  assert.equal(terrainAt(map, 10, 5), "GRASS");
  assert.equal(terrainAt(map, 10, 4), "WATER");
});

// ── DOM opponent-board reconstruction: rotation → engine cells ───────────────
//
// Ground truth is the 2026-06-23 live capture's AUTHORITATIVE board (the active
// player's board, rebuilt cell-by-cell from gamestate.args.kingdom). Both test
// dominoes are on it, so their true engine cells are known:
//   domino 25 (desc.left=forest+1crown, desc.right=field): forest+1 @ (9,7),
//             field @ (8,7); DOM rotation 2, style left=100px top=0 → anchor (2,0)
//   domino 21 (desc.left=field+1crown, desc.right=grassland): field+1 @ (7,8),
//             grassland @ (7,9); DOM rotation 3, anchor grid (0,1)
//
// The CRITICAL property (the one an inverted table gets wrong): the style anchor
// is the desc.LEFT half for every rotation, so the crowned half lands on the
// correct cell. domDominoCells returns [{x,y,side}, {x,y,side}] in engine coords
// with side[0] = desc.left (anchor), side[1] = desc.right (second).
console.log("\nplacement_mapping: domDominoCells (DOM opponent reconstruction)");

const CASTLE = [7, 7];
const D25_LEFT = { terrain: "forest", crowns: 1 };   // desc.left
const D25_RIGHT = { terrain: "field", crowns: 0 };   // desc.right
const D21_LEFT = { terrain: "field", crowns: 1 };    // desc.left
const D21_RIGHT = { terrain: "grassland", crowns: 0 }; // desc.right

function cellAt(cells, x, y) {
  return cells.find((c) => c.x === x && c.y === y) || null;
}

// ── domino 25 @ rotation 2 (style left=100,top=0 → anchor grid (2,0)). The
//    forest+1crown half MUST land on the far cell (9,7), NOT the castle-adjacent
//    (8,7); the inverted table this fix corrects put it on (8,7). ─────────────
check("domino 25 rot2 -> forest+1 @ (9,7), field @ (8,7) [verified vs authoritative]", () => {
  const cells = domDominoCells(2, 0, 2, D25_LEFT, D25_RIGHT, CASTLE);
  const anchor = cellAt(cells, 9, 7);
  const second = cellAt(cells, 8, 7);
  assert.ok(anchor && second, "both cells present");
  assert.equal(anchor.side.terrain, "forest");
  assert.equal(anchor.side.crowns, 1, "crown must be on (9,7), the far cell");
  assert.equal(second.side.terrain, "field");
  assert.equal(second.side.crowns, 0);
});

// ── domino 21 @ rotation 3 (anchor grid (0,1)). desc.right extends DOWN (+y). ─
check("domino 21 rot3 -> field+1 @ (7,8), grassland @ (7,9) [verified vs authoritative]", () => {
  const cells = domDominoCells(0, 1, 3, D21_LEFT, D21_RIGHT, CASTLE);
  const anchor = cellAt(cells, 7, 8);
  const second = cellAt(cells, 7, 9);
  assert.ok(anchor && second, "both cells present");
  assert.equal(anchor.side.terrain, "field");
  assert.equal(anchor.side.crowns, 1);
  assert.equal(second.side.terrain, "grassland");
  assert.equal(second.side.crowns, 0);
});

// ── Offset directions for all four rotations (anchor at castle for clarity). ──
check("domRotationOffset is right/up/left/down for 0/1/2/3", () => {
  assert.deepEqual(domRotationOffset(0), [1, 0]);
  assert.deepEqual(domRotationOffset(1), [0, -1]);
  assert.deepEqual(domRotationOffset(2), [-1, 0]);
  assert.deepEqual(domRotationOffset(3), [0, 1]);
});

check("rotation 0 places desc.right to the right of the anchor", () => {
  const cells = domDominoCells(0, 0, 0, D25_LEFT, D25_RIGHT, CASTLE);
  assert.deepEqual([cells[0].x, cells[0].y], [7, 7]); // anchor = desc.left
  assert.deepEqual([cells[1].x, cells[1].y], [8, 7]); // second = desc.right (→)
  assert.equal(cells[0].side.terrain, "forest");
  assert.equal(cells[1].side.terrain, "field");
});

// ── Unknown rotation is rejected (caller skips the tile, no guessing). ────────
check("domDominoCells returns null for an unknown rotation", () => {
  assert.equal(domDominoCells(0, 0, 7, D25_LEFT, D25_RIGHT, CASTLE), null);
  assert.equal(domRotationOffset(9), null);
});

console.log(`\nAll ${passed} placement-mapping checks passed.`);
