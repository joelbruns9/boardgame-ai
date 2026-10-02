// Packet recorder: this table only, no chat, dedup, string packets, history
// seed, retry-until-hooked, originals always called.
const fs=require("node:fs"),vm=require("node:vm"),assert=require("node:assert/strict");
const ctx=vm.createContext({module:undefined,JSON,String,Array,Set,Map});
vm.runInContext(fs.readFileSync("extension_cantstop/packet_recorder.js","utf8"),ctx);
const seen=[];
const w={location:{href:"https://boardgamearena.com/1/cantstop?table=42"},
  g_gamelogs:[{channel:"/table/t42",table_id:"42",move_id:1,packet_id:1,data:[{type:"dice"}]}]};
let s=ctx.installCantStopPacketRecorder(w);
assert.equal(s.hooked,false,"no gameui yet");
assert.equal(s.packets.size,1,"seeded from g_gamelogs");
w.gameui={notifqueue:{onNotification(p){seen.push(p);return "orig";}}};
s=ctx.installCantStopPacketRecorder(w);
assert.equal(s.hooked,true,"hooks once gameui exists");
assert.equal(ctx.installCantStopPacketRecorder(w),s,"idempotent");
const q=w.gameui.notifqueue;
const pk=(m,extra={})=>({channel:"/table/t42",table_id:"42",move_id:m,packet_id:m,data:[{type:"moveTokens"}],...extra});
assert.equal(q.onNotification(pk(2)),"orig");
q.onNotification(pk(2));                                         // duplicate
q.onNotification(JSON.stringify(pk(3)));                        // string packet
q.onNotification(pk(4,{table_id:"99"}));                        // other table
q.onNotification(pk(5,{channel:"/general/emergency"}));         // other channel
q.onNotification(pk(6,{data:[{type:"chatmessage"}]}));          // chat
q.onNotification("not json");                                   // garbage
assert.equal(seen.length,7,"BGA's own handler sees every packet");
const drain=()=>JSON.parse(JSON.stringify(ctx.drainCantStopPackets(w)));
const out=drain();
assert.deepEqual(out.map(p=>p.move_id),[1,2,3]);
assert.deepEqual(drain(),[],"drain empties");
q.onNotification(pk(7));
assert.deepEqual(drain().map(p=>p.move_id),[7]);
console.log("packet recorder ok");
