const fs=require("node:fs"),vm=require("node:vm"),assert=require("node:assert/strict");
class El {
 constructor(){this.children=[];this.parts={};this.style={};this.classList={add:()=>{}};this._text="";}
 set textContent(v){this._text=v;this.children=[];} get textContent(){return this._text;}
 appendChild(el){this.children.push(el);if(el.src)queueMicrotask(()=>el.onload());}
 append(el){this.appendChild(el);} remove(){} setAttribute(){}
 querySelector(key){return this.parts[key] ||= new El();}
}
(async()=>{
 const handlers={},body=new El(),window={addEventListener:(t,f)=>handlers[t]=f,postMessage:()=>{}};
 const after={phase:"continueChoice",active_player:0,runners:{"8":8}};
 let nn=0,normalized=after; const requests=[];
 const recs=["stop","roll"].map((d,i)=>({label:"Advance 8",kind:"move",rank:i+1,q_value:.2-i*.1,
  fields:{columns:[8],after_move:after,decision:d},follow_up:"then "+d}));
 const api={storage:{local:{get:async()=>({}),set:async()=>{}}},
  runtime:{getURL:x=>x,sendMessage:async msg=>{
   const path=new URL(msg.url).pathname;
   requests.push({path,body:msg.init.body ? JSON.parse(msg.init.body) : null});
   const result=path==="/health"?{game_id:"cantstop",contract:{}}:
    path==="/api/state"?normalized:
    path==="/api/recommend"?(nn++,{ok:true,recommendations:recs,search_ms:3}):{ok:true};
   return {ok:true,status:200,body:JSON.stringify(result)};
  }}};
 const ctx={window,document:{body,head:body,createElement:()=>new El()},browser:api,
  location:{origin:"https://boardgamearena.com"},setTimeout,clearTimeout,console,Date,URL,Blob};
 vm.createContext(ctx);
 for(const file of ["decision_cache.js","request_retry.js","content.js"])
  vm.runInContext(fs.readFileSync("extension_cantstop/"+file,"utf8"),ctx);
 const drain=async()=>{for(let i=0;i<8;i++)await new Promise(setImmediate);};
 await drain();
 const raw={playerorder:["me","other"],players:{me:{name:"Me"},other:{name:"Other"}},
  active_player:"me",viewer_player:"me",required_column_count:5,blocking:false,table_id:"a",phase:"diceChoice"};
 const send=(type,payload)=>handlers.message({source:window,origin:ctx.location.origin,
  data:{advisor:"cantstop-advisor",type,payload}});
 normalized={phase:"diceChoice"};
 send("position",{state:raw});await drain();
 assert.equal(nn,1);
 assert.equal(requests.filter(r=>r.path==="/api/state").length,0,"first roll validates in recommend");
 assert.equal(requests.find(r=>r.path==="/api/recommend").body.state.table_id,"a");
 const panel=body.children.find(e=>e.id==="cantstop-advisor-panel");
 const rows=panel.querySelector('[data-role="rows"]');
 const before=rows.children.length;assert.equal(before,1);
 send("idle",null);assert.equal(rows.children.length,before);
 normalized=after;
 send("position",{state:{...raw,phase:"continueChoice"}});await drain();
 assert.equal(nn,1,"matching continuation must not rerun NN");
 assert.equal(rows.children.length,before,"full dice comparison remains visible");
 const group=rows.children[0];assert.equal(group.children.length,3);
 assert.match(group.children[0].textContent,/selected/);
 // Opponents are always evaluated and logged; the toggle (default on) only
 // decides whether their options are displayed.
 const logs=()=>requests.filter(r=>r.path==="/api/game_log").length;
 const logged=logs();
 send("position",{state:{...raw,active_player:"other"}});await drain();
 assert.equal(nn,2,"opponent decision evaluated");
 assert.equal(rows.children.length,1,"shown by default");
 assert.equal(logs(),logged+1,"and logged");
 const toggle=panel.querySelector('[data-role="opponents"]');
 assert.equal(toggle.checked,true,"display option defaults on");
 toggle.checked=false;toggle.onchange();
 normalized={phase:"diceChoice",active_player:1};
 send("position",{state:{...raw,active_player:"other",turn_id:"opp-hidden"}});await drain();
 assert.equal(nn,3,"still evaluated when hidden");
 assert.equal(logs(),logged+2,"still logged when hidden");
 assert.equal(rows.children.length,0,"hidden by the display option");
 assert.match(panel.querySelector('[data-role="status"]').textContent,/hidden/);
 // A winning selection displays the guaranteed bank without a roll option.
 recs.splice(0, recs.length, {label:"Advance 8",kind:"move",rank:1,q_value:1,
  fields:{columns:[8],after_move:after,decision:"stop",wins_game:true}});
 normalized={phase:"diceChoice",active_player:0};
 send("position",{state:{...raw,turn_id:"winning"}});await drain();
 assert.equal(rows.children[0].children.length,2);
 assert.match(rows.children[0].children[1].textContent,/Stop rolling and win.*100.0%/);
 normalized=after;
 const calls=nn;
 send("position",{state:{...raw,turn_id:"winning",phase:"continueChoice"}});await drain();
 assert.equal(nn,calls);
 assert.match(rows.children[0].children[1].textContent,/Stop rolling and win.*100.0%/);
 recs.splice(0, recs.length, {label:"Stop and win",kind:"stop",rank:1,q_value:1,fields:{wins_game:true}});
 send("position",{state:{...raw,turn_id:"winning-fresh",phase:"continueChoice"}});await drain();
 assert.equal(rows.children[0].children.length,1);
 assert.match(rows.children[0].children[0].textContent,/Stop rolling and win.*100.0%/);
 assert.equal(requests.filter(r=>r.path==="/health").length,1,"reuse server identity across decisions");
 assert.equal(requests.filter(r=>r.path==="/api/state").length,2,"only matching continuations normalize separately");
 // Clear the refresh acknowledgement watchdog triggered by toggling.
 send("capture_ack",{request_id:1});
 panel.querySelector('[data-action="retry"]').onclick();
 send("position",{state:{...raw,turn_id:"refreshed",phase:"continueChoice"}});await drain();
 assert.equal(requests.filter(r=>r.path==="/health").length,2,"manual Refresh rechecks the server");
 send("capture_ack",{request_id:2});
 console.log("Panel integration passed: all branches retained, zero post-pick NN calls, opponents always evaluated, display toggle");
})().catch(e=>{console.error(e);process.exitCode=1;});
