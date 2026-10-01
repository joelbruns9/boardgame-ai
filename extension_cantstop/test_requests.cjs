const fs=require("node:fs"), vm=require("node:vm"), assert=require("node:assert/strict");
const ctx={setTimeout};vm.createContext(ctx);
vm.runInContext(fs.readFileSync("extension_cantstop/request_retry.js","utf8"),ctx);
(async()=>{
  for (const status of [0,408,429,500,502,503,504]) {
    let calls=0;const waits=[];
    const result=await ctx.advisorWithRetry(async()=>{
      if (++calls<3) throw Object.assign(new Error("temporary"),{status});
      return "recovered";
    },{wait:async ms=>waits.push(ms)});
    assert.equal(result,"recovered");assert.equal(calls,3);
    assert.deepEqual(waits,[500,1500]);
  }
  for (const status of [400,401,403,404]) {
    let calls=0;
    await assert.rejects(ctx.advisorWithRetry(async()=>{
      calls++;throw Object.assign(new Error("permanent"),{status});
    },{wait:async()=>{}}),/permanent/);
    assert.equal(calls,1);
  }
  let calls=0;
  await assert.rejects(ctx.advisorWithRetry(async()=>{
    calls++;throw Object.assign(new Error("offline"),{status:0});
  },{wait:async()=>{}}),/offline/);
  assert.equal(calls,3);
  let current=true;calls=0;
  await assert.rejects(ctx.advisorWithRetry(async()=>{
    calls++;throw Object.assign(new Error("temporary"),{status:0});
  },{isCurrent:()=>current,wait:async()=>{current=false;}}),/Position changed/);
  assert.equal(calls,1);
  console.log("Request recovery tests passed: transient recovery, bounded retries, permanent errors, position cancellation");
})().catch(e=>{console.error(e);process.exitCode=1;});
