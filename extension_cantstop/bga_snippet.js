// BGA cantstop.js 260212-1713 getVariantText: 1=Jump, 2=Forced movement.
// 0 is the standard game (also observed in table 923012474).
function cantStopBlocking(flag) {
  if (flag === 0 || flag === "0") return false;
  if (flag === 2 || flag === "2") return true;
  if (flag === 1 || flag === "1") throw new Error("BGA Jump variant is not supported by this model");
  throw new Error("Unknown BGA movement variant: " + String(flag));
}
// Game-specific capture. Coordinate conversion lives in the Python adapter.
function captureCantStop(w = window, includeOpponents = false) {
  const ui = w.gameui, gd = ui && ui.gamedatas;
  if (!gd || !gd.gamestate || !("required_column_count" in gd) || !("movement_variant" in gd)) return null;
  const phase = gd.gamestate.name;
  const me = String(gd.me_id || ui.player_id || "");
  const active = String(gd.gamestate.active_player || "");
  const order = (gd.playerorder || []).map(String);
  if (!order.includes(me) || (!includeOpponents && active !== me) || !["diceChoice", "continueChoice"].includes(phase)) return null;
  const players = {};
  for (const p of order) {
    const x = gd.players[p];
    // ``no`` is BGA's seat number (1 = starting player); playerorder is
    // rotated to start at the viewer, so it cannot give the turn order's start.
    players[p] = {color: x.color, score: Number(x.score), name: String(x.name || p),
      no: Number(x.no ?? x.player_no ?? 0)};
  }
  const markers = [];
  for (const el of w.document.querySelectorAll(".tokenspace.token")) {
    // Hidden templates often omit their coordinates entirely.
    if (el.dataset.column == null || el.dataset.height == null || el.dataset.column === "" || el.dataset.height === "") continue;
    const colorClass = [...el.classList].find(c => c.startsWith("color_"));
    if (!colorClass) throw new Error("Board marker has no color");
    markers.push({column: Number(el.dataset.column), height: Number(el.dataset.height), color: colorClass.slice(6)});
  }
  markers.sort((a,b) => a.column-b.column || a.color.localeCompare(b.color) || a.height-b.height);
  return {format: "bga-cantstop-v1", playerorder: order, players,
    active_player: active, viewer_player: me, phase, markers,
    dice: phase === "diceChoice" ? (gd.gamestate.args?.dice || []).map(Number) : [],
    required_column_count: Number(gd.required_column_count),
    movement_variant_raw: gd.movement_variant,
    blocking: cantStopBlocking(gd.movement_variant),
    possible_moves_raw: gd.gamestate.args?.possibleMoves || null,
    table_id: new URL(w.location.href).searchParams.get("table") || String(gd.table_id || "unknown")};
}
