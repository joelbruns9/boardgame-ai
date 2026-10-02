// Player win-probability rows: seat order, roll-only updates, stale answers.
const fs=require("node:fs"),vm=require("node:vm"),assert=require("node:assert/strict");
class El {
 constructor(){this.children=[];this.parts={};this.style={};this.classList={add:()=>{}};this._text="";this.className="";}
 set textContent(v){this._text=v;this.children=[];} get textContent(){return this._text;}
 appendChild(el){this.children.push(el);if(el.src)queueMicrotask(()=>el.onload());}
 append(el){this.appendChild(el);} remove(){} setAttribute(){}
 querySelector(key){return this.parts[key] ||= new El();}
}
(async()=>{
 const handlers={},body=new El(),window={addEventListener:(t,f)=>handlers[t]=f,postMessage:()=>{}};
 const winCalls=[];   // {body, resolve, reject} -- answered by the test, in any order
 const api={storage:{local:{get:async()=>({}),set:async()=>{}}},
  runtime:{getURL:x=>x,sendMessage:msg=>{
   const path=new URL(msg.url).pathname;
   const body=msg.init.body ? JSON.parse(msg.init.body) : null;
   if(path==="/api/cantstop/win_probabilities")
    return new Promise((resolve,reject)=>winCalls.push({body,resolve,reject}));
   const result=path==="/health"?{game_id:"cantstop",contract:{}}:{ok:true,recommendations:[],search_ms:1};
   return Promise.resolve({ok:true,status:200,body:JSON.stringify(result)});
  }}};
 const ctx={window,document:{body,head:body,createElement:()=>new El()},browser:api,
  location:{origin:"https://boardgamearena.com"},setTimeout,clearTimeout,console,Date,URL,Blob};
 vm.createContext(ctx);
 for(const file of ["decision_cache.js","request_retry.js","content.js"])
  vm.runInContext(fs.readFileSync("extension_cantstop/"+file,"utf8"),ctx);
 const drain=async()=>{for(let i=0;i<8;i++)await new Promise(setImmediate);};
 await drain();
 const send=(type,payload)=>handlers.message({source:window,origin:ctx.location.origin,
  data:{advisor:"cantstop-advisor",type,payload}});
 const answer=(call,seats)=>call.resolve({ok:true,status:200,body:JSON.stringify(
  {seats,player_ids:["me","c","a"],active_seat:1,basis:"after roll, best play assumed"})});
 // BGA rotates playerorder to start at the viewer; seat numbers give the
 // real turn order: a (1) -> me (2) -> c (3).
 const raw={playerorder:["me","c","a"],viewer_player:"me",active_player:"c",
  players:{me:{name:"Me",color:"ff0000",no:2},c:{name:"Cat",color:"00ff00",no:3},a:{name:"Ann",color:"0000ff",no:1}},
  required_column_count:4,blocking:true,table_id:"t",turn_id:"s:1",phase:"diceChoice"};

 // 0. An old server without the route says so in the panel, not only a tooltip.
 send("position",{state:{...raw,turn_id:"s:0"}});await drain();
 winCalls[0].resolve({ok:false,status:404,body:JSON.stringify({detail:"Not Found"})});await drain();
 {const p=body.children.find(e=>e.id==="cantstop-advisor-panel").querySelector('[data-role="players"]');
  assert.match(p.textContent,/Win chances unavailable.*older version.*restart/);}
 winCalls.shift();

 // 1. An opponent's roll updates the list even with opponent advice off.
 send("position",{state:raw});await drain();
 assert.equal(winCalls.length,1,"win request on an opponent's roll");
 assert.deepEqual(winCalls[0].body.options,{table_id:"t",turn_id:"s:1"});
 answer(winCalls[0],[0.25,0.60,0.15]);await drain();
 const panel=body.children.find(e=>e.id==="cantstop-advisor-panel");
 const list=panel.querySelector('[data-role="players"]');
 const shown=()=>list.children.map(r=>r.children[1].textContent+" "+r.children[2].textContent);
 assert.deepEqual(shown(),["Ann 15.0%","Me (you) 25.0%","Cat 60.0%"],"starting player first, seat-mapped");
 assert.match(list.children[2].className,/advisor-player-active/);
 assert.equal(list.children[0].children[0].style.background,"#0000ff");

 // 2. The stop/roll decision of the same turn keeps the roll's numbers.
 send("position",{state:{...raw,phase:"continueChoice"}});await drain();
 assert.equal(winCalls.length,1,"no update on a decision");

 // 3. A decision with nothing shown for its turn does fetch.
 send("position",{state:{...raw,turn_id:"s:2",phase:"continueChoice"}});await drain();
 assert.equal(winCalls.length,2);
 answer(winCalls[1],[0.3,0.5,0.2]);await drain();
 assert.deepEqual(shown(),["Ann 20.0%","Me (you) 30.0%","Cat 50.0%"]);

 // 4. A late answer for an older roll never overwrites a newer one.
 send("position",{state:{...raw,turn_id:"s:2"}});await drain();
 send("position",{state:{...raw,turn_id:"s:2",phase:"diceChoice",dice:[1,1,1,2]}});await drain();
 assert.equal(winCalls.length,4);
 answer(winCalls[3],[0.1,0.8,0.1]);await drain();
 answer(winCalls[2],[0.9,0.05,0.05]);await drain();
 assert.deepEqual(shown(),["Ann 10.0%","Me (you) 10.0%","Cat 80.0%"],"newest roll wins");

 // 5. A server error keeps the last numbers and says why in the tooltip.
 send("position",{state:{...raw,turn_id:"s:3"}});await drain();
 winCalls[4].resolve({ok:false,status:500,body:JSON.stringify({detail:"boom"})});await drain();
 assert.deepEqual(shown(),["Ann 10.0%","Me (you) 10.0%","Cat 80.0%"]);
 assert.match(list.title,/not updated: boom/);
 console.log("Win-probability rows passed: seat order, roll-only updates, stale answers ignored, errors keep last values");
})().catch(e=>{console.error(e);process.exitCode=1;});
