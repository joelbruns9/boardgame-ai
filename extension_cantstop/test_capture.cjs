const fs = require("node:fs"), vm = require("node:vm"), assert = require("node:assert/strict");
const sandbox = {URL}; vm.createContext(sandbox);
vm.runInContext(fs.readFileSync("extension_cantstop/bga_snippet.js", "utf8"), sandbox);
const gd = {required_column_count:4,movement_variant:0,me_id:30,playerorder:[10,20,30],
  players:{10:{color:"ff0000",score:0},20:{color:"0000ff",score:0},30:{color:"00ff00",score:0}},
  gamestate:{name:"diceChoice",active_player:30,args:{dice:[1,2,5,6]}}};
const w = {gameui:{gamedatas:gd},location:{href:"https://boardgamearena.com/1/cantstop?table=123"},
 document:{querySelectorAll:()=>[{dataset:{column:"6",height:"1"},classList:["token","color_000000"]}]}};
const a=sandbox.captureCantStop(w);
assert.equal(a.playerorder.length,3); assert.equal(a.markers[0].height,1);
assert.equal(a.table_id,"123"); assert.equal(a.movement_variant_raw,0);
gd.gamestate.name="continueChoice";
assert.equal(sandbox.captureCantStop(w).dice.length,0);
gd.gamestate.active_player=10;
assert.equal(sandbox.captureCantStop(w),null);
gd.gamestate.active_player=30; gd.me_id=999;
assert.equal(sandbox.captureCantStop(w),null);
console.log("BGA capture tests passed (synthetic DOM)");


// Verify same-dice board changes invalidate advice immediately and settle again.
let now=0, tick, messages=[], current={phase:"diceChoice",runners:{6:1}};
const listeners={};
const bridgeWindow={postMessage:m=>messages.push(m),addEventListener:(type,fn)=>{listeners[type]=fn;},removeEventListener:()=>{}};
const bridge={window:bridgeWindow,location:{origin:"https://boardgamearena.com",href:w.location.href},
  URL,Date:{now:()=>now},setInterval:f=>{tick=f;return 1;},clearInterval:()=>{},captureCantStop:()=>current};
vm.createContext(bridge);
vm.runInContext(fs.readFileSync("extension_cantstop/turn_identity.js","utf8"),bridge);
vm.runInContext(fs.readFileSync("extension_cantstop/page_bridge.js","utf8"),bridge);
assert.equal(messages.at(-1).type,"idle");
now=1500; tick(); assert.equal(messages.at(-1).type,"position");
current={phase:"diceChoice",runners:{6:2}};
now=1600; tick(); assert.equal(messages.at(-1).type,"idle");
now=2900; tick(); assert.equal(messages.at(-1).type,"position");
current=null; now=3000; tick(); assert.equal(messages.at(-1).type,"idle");
console.log("Position settling and stale-advice invalidation tests passed");


assert.equal(a.blocking,false);
for (const flag of [0,"0"]) assert.equal(sandbox.cantStopBlocking(flag),false);
for (const flag of [2,"2"]) assert.equal(sandbox.cantStopBlocking(flag),true);
for (const flag of [1,"1",null,undefined,true,false,3,""])
  assert.throws(()=>sandbox.cantStopBlocking(flag));
gd.me_id=30; gd.gamestate.active_player=30; gd.movement_variant="2";
assert.equal(sandbox.captureCantStop(w).blocking,true);
gd.movement_variant=1;
assert.throws(()=>sandbox.captureCantStop(w),/Jump variant/);
console.log("Automatic BGA variant detection tests passed");

vm.runInContext(fs.readFileSync("extension_cantstop/decision_cache.js","utf8"),sandbox);
const expected={rules:{num_players:3,blocking:true},active_player:2,runners:{"6":11,"8":11},dice:[],phase:"continueChoice"};
const reordered={phase:"continueChoice",dice:[],runners:{"8":11,"6":11},active_player:2,rules:{blocking:true,num_players:3}};
assert.equal(sandbox.sameAdvisorState(expected,reordered),true);
assert.equal(sandbox.sameAdvisorState(expected,{...reordered,active_player:1}),false);
assert.equal(sandbox.sameAdvisorState(expected,{...reordered,runners:{"6":10,"8":11}}),false);
assert.equal(sandbox.sameAdvisorState(expected,{...reordered,rules:{num_players:3,blocking:false}}),false);
console.log("Combined decision cache matching tests passed");


// Firefox content-script requests carry source=null. They must be acknowledged
// and resend the same settled position; foreign origins/windows must not.
current={phase:"diceChoice",runners:{6:2}};
now=4000;tick();now=5500;tick();
messages.length=0;
listeners.message({source:null,origin:bridge.location.origin,
  data:{advisor:"cantstop-advisor",type:"recapture",request_id:42}});
assert.equal(messages[0].type,"capture_ack");
assert.equal(messages[0].payload.request_id,42);
assert.equal(messages[1].type,"position");
messages.length=0;
listeners.message({source:null,origin:"https://foreign.example",
  data:{advisor:"cantstop-advisor",type:"recapture",request_id:43}});
assert.equal(messages.length,0);
listeners.message({source:{},origin:bridge.location.origin,
  data:{advisor:"cantstop-advisor",type:"recapture",request_id:44}});
assert.equal(messages.length,0);
console.log("Firefox null-source Refresh regression passed; foreign messages rejected");

gd.movement_variant=0;gd.me_id=30;gd.gamestate.active_player=10;
assert.equal(sandbox.captureCantStop(w),null);
const opponent=sandbox.captureCantStop(w,true);
assert.equal(opponent.active_player,"10");
assert.equal(opponent.viewer_player,"30");
assert.equal(opponent.playerorder[0],"10");
const complete={recommendations:[
 {fields:{after_move:expected,decision:"stop"}},{fields:{after_move:expected,decision:"roll"}}
]};
assert.ok(sandbox.findAdvisorContinuation(complete,"a","a",expected));
assert.equal(sandbox.findAdvisorContinuation(complete,"a","b",expected),null);
assert.equal(sandbox.findAdvisorContinuation(complete,"a","a",{...expected,active_player:1}),null);
assert.equal(sandbox.findAdvisorContinuation(complete,"a","a",{...expected,phase:"diceChoice"}),null);
assert.equal(complete.recommendations.length,2);
console.log("Opponent capture and full-result continuation reuse tests passed");
