const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
let now=0,tick,interval,messages=[],handler;
const gd={gamestate:{name:'diceChoice',active_player:'a'}};
const window={gameui:{gamedatas:gd},postMessage:m=>messages.push(m),
 addEventListener:(t,f)=>handler=f,removeEventListener:()=>{}};
let current={table_id:'t',active_player:'a',playerorder:['a','b'],players:{a:{color:'0000ff'},b:{color:'ff0000'}},
 required_column_count:5,movement_variant_raw:'0',blocking:false,phase:'diceChoice',dice:[1,1,3,5],
 markers:[{column:7,height:5,color:'0000ff'},{column:8,height:4,color:'000000'}]};
const ctx=vm.createContext({window,URL,Date:{now:()=>now},location:{origin:'https://boardgamearena.com',href:'https://boardgamearena.com/?table=t'},
 setInterval:(f,ms)=>{tick=f;interval=ms;return 1},clearInterval:()=>{},captureCantStop:()=>current});
for(const file of ['turn_identity.js','page_bridge.js'])vm.runInContext(fs.readFileSync('extension_cantstop/'+file,'utf8'),ctx);
const send=(type,payload,origin=ctx.location.origin)=>handler({source:null,origin,data:{advisor:'cantstop-advisor',type,payload}});
const at=t=>{now=t;tick();};
const positions=()=>messages.filter(m=>m.type==='position').length;
assert.equal(interval,100);
at(200);assert.equal(positions(),0);
at(1200);assert.equal(positions(),1);
send('validated_position',{state:structuredClone(current),runners:[{'8':8}]});
// Partial animation must not qualify for the fast path.
current={...current,phase:'continueChoice',dice:[],markers:[current.markers[0]]};
gd.gamestate.name='continueChoice';at(1300);at(1500);assert.equal(positions(),1);
// Predicted completed move becomes eligible after 200 ms of stability.
current={...current,markers:[current.markers[0],{column:8,height:3,color:'000000'}]};
at(1600);at(1799);assert.equal(positions(),1);at(1800);assert.equal(positions(),2);
// Next roll at the same runners is also fast, with no extra confirmation needed.
current={...current,phase:'diceChoice',dice:[2,2,3,3]};gd.gamestate.name='diceChoice';
at(1900);at(2100);assert.equal(positions(),3);
// Changed saved board must wait the full interval.
current={...current,markers:[{column:7,height:4,color:'0000ff'},current.markers[1]]};
at(2200);at(2400);assert.equal(positions(),3);at(3400);assert.equal(positions(),4);
// A stale or foreign acknowledgement cannot authorize a different board.
send('validated_position',{state:{...current,active_player:'b'},runners:[{'8':9}]});
send('validated_position',{state:structuredClone(current),runners:[{'8':9}]},'https://other.example');
current={...current,markers:[current.markers[0],{column:8,height:2,color:'000000'}]};
at(3500);at(3700);assert.equal(positions(),4);at(4700);assert.equal(positions(),5);
// A backend validation error revokes the fast path.
send('validated_position',{state:structuredClone(current),runners:[{'8':10}]});
send('capture_unsettled');
current={...current,markers:[current.markers[0],{column:8,height:1,color:'000000'}]};
at(4800);at(5000);assert.equal(positions(),5);at(6000);assert.equal(positions(),6);
// Even identical markers cannot reuse a prior player's/turn's fast path.
send('validated_position',{state:structuredClone(current),runners:[{'8':10}]});
gd.gamestate.active_player='b'; current={...current,active_player:'b'};
at(6100);at(6300);assert.equal(positions(),6);at(7300);assert.equal(positions(),7);
console.log('Fast capture: 200 ms predicted moves/rolls; 1200 ms fallback; stale/error/turn isolation passed');
