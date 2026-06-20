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
const { parsePlacement, mapHalvesToCells } = require(
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

console.log(`\nAll ${passed} placement-mapping checks passed.`);
