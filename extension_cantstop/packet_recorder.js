// BGA notification packets (MAIN world): the server's own ordered record of
// every roll, pairing, stop, bust and the game end. Board captures only see
// decision screens, so they miss busting rolls, the pairing chosen before a
// bust, first-roll busts, rolls skipped by a throttled background tab, and
// who won (games/cantstop/luck.py). The packet stream has all of them.
//
// Ported from extension_7wd/bga_snippet.js, whose four facts were verified
// live 2026-08-15: gameui.notifqueue.onNotification(packet) sees every packet
// raw and whole ({channel, table_id, packet_id, packet_type, move_id, time,
// data: [...]}); it also accepts a JSON string; a (re)loaded page is re-sent
// the history through the same call (packet_type "resend"), so opening the
// tab mid-game backfills; replay pages preload window.g_gamelogs.
//
// Kept: packets for THIS table on /table or /player channels. Kept out: other
// channels and chat. Recording never throws into BGA's dispatch.
const CANTSTOP_PACKET_STORE = "__cantstopPacketStore";
const CANTSTOP_CHAT = new Set(["chat", "groupchat", "chatmessage", "tablechat",
  "privatechat", "startWriting", "stopWriting"]);

function cantstopRecordPacket(store, packet) {
  if (typeof packet === "string") {
    try { packet = JSON.parse(packet); } catch { return; }
  }
  if (!packet || !Array.isArray(packet.data)) return;
  const channel = String(packet.channel || "");
  if (!channel.startsWith("/table") && !channel.startsWith("/player")) return;
  // A /player packet with no table is a site notification, not this game.
  if (store.tableId && String(packet.table_id) !== store.tableId) return;
  if (CANTSTOP_CHAT.has(String(packet.type || ""))) return;
  if (packet.data.length && packet.data.every(e => CANTSTOP_CHAT.has(String(e?.type)))) return;
  const key = String(packet.move_id) + "/" + String(packet.packet_id);
  if (store.packets.has(key)) return;
  store.packets.set(key, packet);
  store.fresh.push(key);
}

// Idempotent, and re-tries the hook until gameui exists: an early call must
// not leave the recorder permanently deaf.
function installCantStopPacketRecorder(w) {
  let store = w[CANTSTOP_PACKET_STORE];
  if (!store) {
    const match = /[?&]table=(\d+)/.exec(String(w.location?.href || ""));
    store = {packets: new Map(), fresh: [], tableId: match ? match[1] : null, hooked: false};
    w[CANTSTOP_PACKET_STORE] = store;
    if (Array.isArray(w.g_gamelogs)) for (const p of w.g_gamelogs) cantstopRecordPacket(store, p);
  }
  if (store.hooked) return store;
  const queue = w.gameui?.notifqueue;
  if (queue && typeof queue.onNotification === "function") {
    const original = queue.onNotification;
    queue.onNotification = function (packet) {
      try { cantstopRecordPacket(store, packet); } catch {}
      return original.apply(this, arguments);
    };
    store.hooked = true;
  }
  return store;
}

// Packets recorded since the last drain, oldest first.
function drainCantStopPackets(w) {
  const store = w[CANTSTOP_PACKET_STORE];
  if (!store) return [];
  const fresh = store.fresh;
  store.fresh = [];
  return fresh.map(k => store.packets.get(k)).filter(Boolean);
}

if (typeof module !== "undefined")
  module.exports = {installCantStopPacketRecorder, drainCantStopPackets};
