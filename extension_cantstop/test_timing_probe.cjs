// Timing probe: hooks pass through untouched, record only types, never throw.
const fs=require("node:fs"),vm=require("node:vm"),assert=require("node:assert/strict");
let clock=0;
const observers=[];
class MO { constructor(cb){this.cb=cb;observers.push(this);} observe(){} disconnect(){this.off=true;} }
const calls=[];
const queue={onNotification(p){calls.push(["notif",p]);return "orig-notif";}};
const ui={notifqueue:queue,gamedatas:{gamestate:{active_player:"7"}},
  onEnteringState(name){calls.push(["state",name]);return "orig-state";}};
const w={performance:{now:()=>clock},document:{body:{}},MutationObserver:MO};
const ctx=vm.createContext({module:undefined,Date,JSON,String,Math});
vm.runInContext(fs.readFileSync("extension_cantstop/timing_probe.js","utf8"),ctx);
const probe=ctx.createCantStopTiming(w,5);
assert.equal(probe.install(),false,"no gameui yet: retry later");
w.gameui=ui;
assert.equal(probe.install(),true);
assert.equal(probe.install(),true,"idempotent");
clock=10;
assert.equal(queue.onNotification({data:[{type:"diceRolled",args:{secret:1}},{type:"chatmessage"}]}),"orig-notif");
assert.equal(queue.onNotification('{"data":[{"type":"moveMarker"}]}'),"orig-notif","string packets too");
assert.equal(queue.onNotification("not json"),"orig-notif","garbage passes through untouched");
clock=25;
assert.equal(ui.onEnteringState("diceChoice",{}),"orig-state");
clock=40;
observers[0].cb([{target:{className:"tokenspace token color_000000"}},{target:{className:"other"}}]);
observers[0].cb([{target:{className:"other"}}]);
assert.deepEqual(calls.map(c=>c[0]),["notif","notif","notif","state"],"originals always called");
const d=JSON.parse(JSON.stringify(probe.dump()));
assert.deepEqual(d.hooks,{notif:true,state:true,board:true});
assert.deepEqual(d.events.map(e=>e.kind),["notif","notif","state","board"]);
assert.deepEqual(d.events[0].types,["diceRolled"],"types only, chat dropped, no args");
assert.equal(JSON.stringify(d).includes("secret"),false,"no packet contents recorded");
assert.deepEqual([d.events[2].name,d.events[2].active,d.events[2].t],["diceChoice","7",25]);
for(let i=0;i<10;i++)probe.record("sig",{i});
assert.equal(probe.dump().events.length,5,"ring buffer bounded");
probe.dispose();
assert.equal(queue.onNotification({data:[{type:"x"}]}),"orig-notif");
assert.equal(probe.dump().events.at(-1).i,9,"disposed hooks no longer record");
assert.ok(observers[0].off);
console.log("Timing probe passed: pass-through hooks, types only, bounded, disposable");

// Bridge integration: signature changes and settled sends land in the probe,
// and a timing_request is answered with the dump (what Export capture uses).
{
let now=0,tick,messages=[],handler;
const gd={gamestate:{name:'diceChoice',active_player:'a'}};
const window={gameui:{gamedatas:gd},postMessage:m=>messages.push(m),
 addEventListener:(t,f)=>handler=f,removeEventListener:()=>{},performance:{now:()=>now}};
let current={table_id:'t',active_player:'a',playerorder:['a','b'],players:{a:{color:'0000ff'},b:{color:'ff0000'}},
 required_column_count:5,movement_variant_raw:'0',blocking:false,phase:'diceChoice',dice:[1,1,3,5],
 markers:[{column:7,height:5,color:'0000ff'},{column:8,height:4,color:'000000'}]};
const ctx=vm.createContext({window,URL,Date:{now:()=>now},location:{origin:'https://boardgamearena.com',href:'https://boardgamearena.com/?table=t'},
 setInterval:f=>{tick=f;return 1},clearInterval:()=>{},captureCantStop:()=>current});
for(const f of ['turn_identity.js','timing_probe.js','page_bridge.js'])vm.runInContext(fs.readFileSync('extension_cantstop/'+f,'utf8'),ctx);
for(const t of [100,600,1300]){now=t;tick();}
handler({source:null,origin:ctx.location.origin,data:{advisor:'cantstop-advisor',type:'timing_request'}});
const dump=JSON.parse(JSON.stringify(messages.find(m=>m.type==='timing_dump').payload));
const kinds=dump.events.map(e=>e.kind);
assert.ok(kinds.includes('sig')&&kinds.includes('sent'),"bridge records reads");
const sent=dump.events.find(e=>e.kind==='sent');
assert.equal(sent.fast,false);assert.ok(sent.waited>=1200,"slow path waited 1.2 s");
console.log("Timing probe bridge integration passed: reads recorded, dump on request");
}
