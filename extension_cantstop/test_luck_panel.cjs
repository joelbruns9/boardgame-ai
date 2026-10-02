// Dice-luck block: refreshed after logged packet batches, turn order,
// unavailable reason shown, bursts coalesced to one queued request.
const fs=require("node:fs"),vm=require("node:vm"),assert=require("node:assert/strict");
class El {
 constructor(){this.children=[];this.parts={};this.style={};this.classList={add:()=>{}};this._text="";this.className="";}
 set textContent(v){this._text=v;this.children=[];}
 get textContent(){return this._text+this.children.map(c=>c.textContent).join("|");}
 appendChild(el){this.children.push(el);if(el.src)queueMicrotask(()=>el.onload());}
 append(el){this.appendChild(el);} remove(){} setAttribute(){}
 querySelector(key){return this.parts[key] ||= new El();}
}
(async()=>{
 const handlers={},body=new El(),window={addEventListener:(t,f)=>handlers[t]=f,postMessage:()=>{}};
 const calls=[];      // luck requests, answered by the test
 const logs=[];
 const api={storage:{local:{get:async()=>({}),set:async()=>{}}},
  runtime:{getURL:x=>x,sendMessage:msg=>{
   const path=new URL(msg.url).pathname, b=msg.init.body?JSON.parse(msg.init.body):null;
   if(path==="/api/cantstop/luck") return new Promise(resolve=>calls.push({b,resolve}));
   if(path==="/api/game_log") logs.push(b);
   const result=path==="/health"?{game_id:"cantstop",contract:{}}:{ok:true,recommendations:[],search_ms:1,seats:[.5,.5],player_ids:["me","op"]};
   return Promise.resolve({ok:true,status:200,body:JSON.stringify(result)});
  }}};
 const ctx={window,document:{body,head:body,createElement:()=>new El()},browser:api,
  location:{origin:"https://boardgamearena.com"},setTimeout,clearTimeout,console,Date,URL,Blob};
 vm.createContext(ctx);
 for(const f of ["decision_cache.js","request_retry.js","content.js"])
  vm.runInContext(fs.readFileSync("extension_cantstop/"+f,"utf8"),ctx);
 const drain=async()=>{for(let i=0;i<10;i++)await new Promise(setImmediate);};
 await drain();
 const send=(type,payload)=>handlers.message({source:window,origin:ctx.location.origin,data:{advisor:"cantstop-advisor",type,payload}});
 const raw={playerorder:["me","op"],viewer_player:"me",active_player:"op",
  players:{me:{name:"Me",color:"0000ff",no:2},op:{name:"Opp",color:"ff0000",no:1}},
  required_column_count:3,blocking:false,table_id:"t9",turn_id:"s:1",phase:"diceChoice"};
 send("position",{state:raw});await drain();
 const answer=(c,r)=>c.resolve({ok:true,status:200,body:JSON.stringify(r)});
 const ok={available:true,players:[
  {player_id:"me",name:"Me",busts:5,busts_expected:1.75,dice_pts:-23.7,own_rolls_pts:-9.6,progress_cols:-0.44},
  {player_id:"op",name:"Opp",busts:1,busts_expected:0.35,dice_pts:23.7,own_rolls_pts:54.6,progress_cols:1.56}]};

 send("packets",{table_id:"t9",packets:[{move_id:1}]});await drain();
 assert.equal(logs.at(-1).kind,"bga_packets");assert.equal(logs.at(-1).table_id,"t9");
 assert.equal(calls.length,1,"luck requested after the batch is logged");
 assert.deepEqual(calls[0].b,{table_id:"t9",device:"cuda"});
 // a burst while the request is in flight queues exactly one more
 send("packets",{table_id:"t9",packets:[{move_id:2}]});await drain();
 send("packets",{table_id:"t9",packets:[{move_id:3}]});await drain();
 assert.equal(calls.length,1,"still one in flight");
 answer(calls[0],ok);await drain();
 assert.equal(calls.length,2,"one queued follow-up");
 const panel=body.children.find(e=>e.id==="cantstop-advisor-panel");
 const luck=panel.querySelector('[data-role="luck"]');
 const text=luck.textContent;
 assert.match(text,/Dice luck so far \(own rolls\)/);
 assert.ok(text.indexOf("Opp")<text.indexOf("Me (you)"),"turn order: seat no 1 first");
 assert.match(text,/Me \(you\)\|−9\.6 pts\|5 busts \/ 1\.8 exp\|progress −0\.4 columns vs average dice/,"own rolls, not all dice");
 assert.match(text,/Opp\|\+54\.6 pts\|1 busts \/ 0\.[34] exp\|progress \+1\.6 columns/);
 answer(calls[1],{available:false,reason:"the game record does not reach back to the first roll"});await drain();
 assert.match(luck.textContent,/Dice luck: the game record does not reach back/);
 assert.equal(calls.length,2,"no extra request without new packets");
 console.log("luck panel ok");
})().catch(e=>{console.error(e);process.exit(1);});
