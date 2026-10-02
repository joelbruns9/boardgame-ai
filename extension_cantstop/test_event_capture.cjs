// Event-driven capture: read at BGA's state entry, send on two matching
// frame reads, cancel on newer entries, fall back to polling otherwise.
const fs=require("node:fs"),vm=require("node:vm"),assert=require("node:assert/strict");
let now=0,tick,messages=[],handler,timers=[];
const gd={gamestate:{name:"diceChoice",active_player:"a"}};
const ui={gamedatas:gd,onEnteringState(){}};
const window={gameui:ui,postMessage:m=>messages.push(m),addEventListener:(t,f)=>handler=f,
  removeEventListener:()=>{},performance:{now:()=>now}};
const board=(runners,dice=[1,2,3,4])=>({table_id:"t",active_player:"a",playerorder:["a","b"],
  players:{a:{color:"0000ff"},b:{color:"ff0000"}},required_column_count:5,movement_variant_raw:"0",
  blocking:false,phase:gd.gamestate.name,dice:gd.gamestate.name==="diceChoice"?dice:[],
  markers:[{column:7,height:5,color:"0000ff"},...runners.map(([c,h])=>({column:c,height:h,color:"000000"}))]});
let current=board([]);
const ctx=vm.createContext({window,URL,Date:{now:()=>now},
  location:{origin:"https://boardgamearena.com",href:"https://boardgamearena.com/?table=t"},
  setInterval:f=>{tick=f;return 1},clearInterval:()=>{},
  setTimeout:(fn,ms)=>{timers.push({at:now+ms,fn});return timers.length},
  captureCantStop:()=>current && {...current, phase:gd.gamestate.name}});
for(const f of ["turn_identity.js","timing_probe.js","page_bridge.js"])
  vm.runInContext(fs.readFileSync("extension_cantstop/"+f,"utf8"),ctx);
// Run due timers in time order (a fake event loop); the 100 ms poll stays off
// so only the event path can send.
const advance=ms=>{const end=now+ms;for(;;){timers.sort((x,y)=>x.at-y.at);
  const t=timers[0];if(!t||t.at>end)break;timers.shift();now=t.at;t.fn();}now=end;};
const positions=()=>messages.filter(m=>m.type==="position");
const enter=name=>{gd.gamestate.name=name;ui.onEnteringState(name,{});};

// 1. Board already final at entry: sent one frame later, not after 200/1200 ms.
now=1000; current=board([]); messages=[];
enter("diceChoice"); advance(40);
assert.equal(positions().length,1,"sent by the event path");
assert.ok(now-1000<=40);
const events=JSON.parse(JSON.stringify(window.__cantstopTiming.dump())).events;
assert.equal(events.filter(e=>e.kind==="sent").at(-1).trigger,"event");
const read=events.filter(e=>e.kind==="event_read").at(-1);
assert.deepEqual([read.reads,read.ms],[2,16],"second read, one frame after entry");

// 2. Board still changing at entry: waits for two matching reads.
messages=[]; current=board([[8,6]]);
enter("continueChoice");
advance(0); current=board([[8,5]]);                 // first read saw a mid-move board
advance(16); assert.equal(positions().length,0,"no send while reads differ");
advance(16); assert.equal(positions().length,1);
assert.deepEqual(positions()[0].payload.state.markers.at(-1),{column:8,height:5,color:"000000"});

// 3. A newer state entry cancels the older run; the board for the newer state is sent.
messages=[]; current=board([[8,5]],[6,6,6,6]);
enter("diceChoice"); advance(0);
current=board([[8,5],[6,2]]); enter("continueChoice"); advance(40);
assert.equal(positions().length,1,"only the newest state's board");
assert.equal(positions()[0].payload.state.phase,"continueChoice");

// 4. Other states never trigger an event send.
messages=[]; enter("endTurn"); advance(200);
assert.equal(positions().length,0);

// 5. A capture that never settles gives up after 12 reads; polling remains.
messages=[]; let flip=0;
ctx.captureCantStop=()=>({...board([[8,(flip++%2)+1]]),phase:gd.gamestate.name});
vm.runInContext("captureCantStop=captureCantStop",ctx);
enter("diceChoice"); advance(400);
assert.equal(positions().length,0,"never sends an unsettled board");
const giveup=JSON.parse(JSON.stringify(window.__cantstopTiming.dump())).events.filter(e=>e.kind==="event_giveup");
assert.equal(giveup.length,1);
console.log("Event capture passed: one-frame send, waits for matching reads, newest state wins, gives up to polling");
