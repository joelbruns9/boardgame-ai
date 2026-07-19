import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = fs.readFileSync(new URL("../content.js", import.meta.url), "utf8")
  .replace("setInterval(pollTick, AUTO_POLL_MS);\npollTick();", "");
const placementSource = fs.readFileSync(
  new URL("../placement_mapping.js", import.meta.url),
  "utf8",
);

function loadContentContext() {
  const context = vm.createContext({
    chrome: {
      runtime: {
        onMessage: { addListener() {} },
        sendMessage() {},
      },
      storage: { local: {} },
    },
    console: { log() {}, warn() {}, error() {} },
    document: { documentElement: {} },
    location: { href: "https://boardgamearena.com/kingdomino" },
    MutationObserver: class { observe() {} },
    Promise,
    setTimeout,
    clearTimeout,
  });
  vm.runInContext(placementSource, context);
  vm.runInContext(source, context);
  vm.runInContext("delayMs = async () => {}; gameLogTick = () => {};", context);
  return context;
}

function capture(state, active = "p1", viewer = "p1") {
  return { ok: true, state, activePlayer: active, viewerId: viewer };
}

test("settleAutoCapture accepts two consecutive identical reads", async () => {
  const context = loadContentContext();
  context.__reads = [capture({ turn: 1 })];
  vm.runInContext("readPageState = async () => __reads.shift();", context);

  const result = await vm.runInContext(
    "settleAutoCapture({ok:true,state:{turn:1},activePlayer:'p1',viewerId:'p1'}, {gameLog:false})",
    context,
  );

  assert.equal(result.key, '{"turn":1}');
  assert.equal(context.__reads.length, 0);
});

test("settleAutoCapture resets stability when the state changes", async () => {
  const context = loadContentContext();
  context.__reads = [capture({ turn: 2 }), capture({ turn: 2 })];
  vm.runInContext("readPageState = async () => __reads.shift();", context);

  const result = await vm.runInContext(
    "settleAutoCapture({ok:true,state:{turn:1},activePlayer:'p1',viewerId:'p1'}, {gameLog:false})",
    context,
  );

  assert.equal(result.key, '{"turn":2}');
  assert.equal(context.__reads.length, 0);
});

test("settleAutoCapture ignores positions where it is not the viewer's turn", async () => {
  const context = loadContentContext();
  context.__reads = [capture({ turn: 1 })];
  vm.runInContext("readPageState = async () => __reads.shift();", context);

  const result = await vm.runInContext(
    "settleAutoCapture({ok:true,state:{turn:1},activePlayer:'p2',viewerId:'p1'}, {gameLog:false})",
    context,
  );

  assert.equal(result, null);
  assert.equal(context.__reads.length, 1);
});

test("auto-turn trigger is rejected on the opponent's turn", async () => {
  const context = loadContentContext();
  context.__starts = 0;
  vm.runInContext(`
    loadOptions = async () => ({streaming:true});
    startStreamingRecommend = async () => { __starts += 1; };
    __opponentCapture = {ok:true,state:{turn:2},activePlayer:"p2",viewerId:"p1"};
  `, context);

  const result = await vm.runInContext(
    `triggerRecommend({reason:"auto-turn",captureOverride:__opponentCapture})`,
    context,
  );

  assert.equal(result.skipped, true);
  assert.equal(result.error, "automatic advice only runs on your turn");
  assert.equal(context.__starts, 0);
});

test("opponent turn stops auto-started jobs but preserves manual refresh jobs", async () => {
  const context = loadContentContext();
  context.__stops = [];
  vm.runInContext(`
    loadOptions = async () => ({autoRefresh:true,gameLog:false});
    readPageState = async () => ({ok:true,state:{turn:2},activePlayer:"p2",viewerId:"p1"});
    stopActiveStreamingJob = async (reason) => { __stops.push(reason); activeStreamingJob = null; };
    activeStreamingJob = {startedAutomatically:true};
  `, context);
  await vm.runInContext("pollTick()", context);
  assert.deepEqual(context.__stops, ["opponent-turn"]);

  vm.runInContext("activeStreamingJob = {startedAutomatically:false};", context);
  await vm.runInContext("pollTick()", context);
  assert.deepEqual(context.__stops, ["opponent-turn"]);
});

test("streaming simulation choices extend to 100,000", () => {
  const context = loadContentContext();
  assert.equal(vm.runInContext("SIM_OPTIONS.includes(100000)", context), true);
  assert.equal(vm.runInContext("Math.max(...SIM_OPTIONS)", context), 100000);
});

test("overlay replacement preserves its scroll position", () => {
  const context = loadContentContext();
  vm.runInContext(`
    __existingOverlay = {scrollTop:173,remove() {}};
    __mountedOverlay = null;
    __element = () => ({
      style:{},children:[],scrollTop:0,
      appendChild(child) { this.children.push(child); },
      addEventListener() {},remove() {},
    });
    document = {
      getElementById: () => __existingOverlay,
      createElement: () => __element(),
      body: {appendChild(box) { __mountedOverlay = box; }},
    };
    __replacementOverlay = makeOverlayBase("Advisor");
    mountOverlay(__replacementOverlay);
  `, context);

  assert.equal(context.__mountedOverlay.scrollTop, 173);
});

test("draft matrix uses color assessment without status text", () => {
  const context = loadContentContext();
  assert.equal(vm.runInContext("draftMatrixAssessment({fragility:0.15})", context), "fragile");
  assert.equal(vm.runInContext("draftMatrixAssessment({fragility:0.149})", context), "robust");
  assert.equal(vm.runInContext("draftMatrixAssessment({fragility:null})", context), "unknown");
});

test("state or parameter change restarts streaming without duplicate start", async () => {
  const context = loadContentContext();
  context.__calls = [];
  vm.runInContext(`
    advisorRequest = async (url, init = {}) => {
      __calls.push({url, payload: init.payload});
      if (url === ADVISOR_START_URL) return {data:{job_id:"job-" + __calls.length},transport:"test"};
      return {data:{status:"cancelled"},transport:"test"};
    };
    pollStreamingJob = () => {};
    renderRecommendations = () => {};
  `, context);
  const options = {
    streaming: true, maxSims: 1000, refreshMs: 1000,
    fragilityAtSims: 0, fragilitySims: 100, topK: 3,
  };
  context.__options = options;

  await vm.runInContext(
    `startStreamingRecommend({ok:true,state:{turn:1}}, {state:{turn:1},engine:"nn"}, __options, "auto")`,
    context,
  );
  await vm.runInContext(
    `startStreamingRecommend({ok:true,state:{turn:1}}, {state:{turn:1},engine:"nn"}, __options, "auto")`,
    context,
  );
  context.__options2 = { ...options, maxSims: 2000 };
  await vm.runInContext(
    `startStreamingRecommend({ok:true,state:{turn:1}}, {state:{turn:1},engine:"nn"}, __options2, "cap-change")`,
    context,
  );
  await vm.runInContext(
    `startStreamingRecommend({ok:true,state:{turn:2}}, {state:{turn:2},engine:"nn"}, __options2, "auto")`,
    context,
  );

  assert.deepEqual(
    context.__calls.map((call) => String(call.url).split("/").at(-1)),
    ["start", "stop", "start", "stop", "start"],
  );
});

test("streaming controller accepts NN, exact, and auto endgames", () => {
  const context = loadContentContext();
  context.__payload = {
    engine: "nn",
    state: {
      phase: "PLACE_AND_SELECT",
      deck_count: 4,
      debug: { deck: [41, 42, 43, 44] },
    },
  };
  assert.equal(
    vm.runInContext("streamingEligible(__payload, {streaming:true})", context),
    true,
  );
  assert.equal(vm.runInContext("streamingEligible({...__payload, engine:'exact'}, {streaming:true})", context), true);
  assert.equal(vm.runInContext("streamingEligible({...__payload, engine:'auto'}, {streaming:true})", context), true);
  assert.equal(vm.runInContext("streamingEligible(__payload, {streaming:false})", context), false);
});

test("auto refresh keeps prior exact advice during an endgame pick-only split", async () => {
  const context = loadContentContext();
  context.__starts = 0;
  vm.runInContext(`
    loadOptions = async () => ({streaming:true});
    startStreamingRecommend = async () => { __starts += 1; };
    __pickCapture = {
      ok:true,activePlayer:"p1",viewerId:"p1",
      state:{phase:"INITIAL_SELECTION",deck_count:4,debug:{deck:[41,42,43,44]}},
    };
  `, context);

  const result = await vm.runInContext(
    `triggerRecommend({reason:"auto-turn",captureOverride:__pickCapture})`, context,
  );
  assert.equal(result.skipped, true);
  assert.equal(result.error, "keeping prior exact advice during endgame pick");
  assert.equal(context.__starts, 0);
});

test("opponent-turn cancellation removes a pending overlay", async () => {
  const context = loadContentContext();
  context.__removed = 0;
  vm.runInContext(`
    document = {getElementById: () => ({remove() { __removed += 1; }})};
    advisorRequest = async () => ({data:{status:"cancelled"},transport:"test"});
    activeStreamingJob = {
      jobId:"auto-job",timer:null,lastResponse:null,startedAutomatically:true,
    };
  `, context);

  await vm.runInContext(`stopActiveStreamingJob("opponent-turn")`, context);
  assert.equal(context.__removed, 1);
});

test("streaming jobs append one terminal lifecycle record", () => {
  const context = loadContentContext();
  context.__logs = [];
  vm.runInContext(`
    postGameLog = (tableId, record) => { __logs.push({tableId,record}); };
    __logJob = {
      jobId:"job-42",terminalLogged:false,options:{gameLog:true},
      payload:{engine:"exact"},capture:{tableId:"42",activePlayer:"p1",viewerId:"p1"},
    };
    __logData = {status:"done",engine:"exact",exact:{solved:true},recommendations:[]};
    logStreamingAdvisorTerminal(__logJob,__logData);
    logStreamingAdvisorTerminal(__logJob,__logData);
  `, context);

  assert.equal(context.__logs.length, 1);
  assert.equal(context.__logs[0].record.advisor.status, "done");
  assert.equal(context.__logs[0].record.advisor.exact_solved, true);
});

test("deck 5 to deck 4 remains on the streaming lifecycle", async () => {
  const context = loadContentContext();
  context.__calls = [];
  vm.runInContext(`
    advisorRequest = async (url, init = {}) => {
      __calls.push(url);
      if (url === ADVISOR_START_URL) return {data:{job_id:"job-" + __calls.length,status:"solving_exact"},transport:"test"};
      return {data:{status:"cancelled"},transport:"test"};
    };
    pollStreamingJob = () => {};
    renderPendingStreamingJob = () => {};
  `, context);
  context.__options = {streaming:true,maxSims:1000,refreshMs:1000};

  await vm.runInContext(
    `startStreamingRecommend({ok:true,state:{deck_count:5}}, {state:{deck_count:5},engine:"nn"}, __options, "auto")`,
    context,
  );
  await vm.runInContext(
    `startStreamingRecommend({ok:true,state:{deck_count:4}}, {state:{deck_count:4},engine:"exact"}, __options, "auto")`,
    context,
  );

  assert.deepEqual(
    context.__calls.map((url) => String(url).split("/").at(-1)),
    ["start", "stop", "start"],
  );
});

test("exact pending polls render without simulation fields", async () => {
  const context = loadContentContext();
  context.__pending = 0;
  vm.runInContext(`
    setTimeout = () => null;
    renderPendingStreamingJob = () => { __pending += 1; };
    advisorRequest = async () => ({
      data:{changed:true,status:"solving_exact",version:0}, transport:"test"
    });
    __job = {
      jobId:"exact-job",version:-1,timer:null,options:{refreshMs:1000},
      payload:{engine:"exact"},startedAt:Date.now()
    };
    activeStreamingJob = __job;
  `, context);

  await vm.runInContext("pollStreamingJob(__job)", context);
  assert.equal(context.__pending, 1);
});

test("status-only snapshots do not advance the stability window", () => {
  const context = loadContentContext();
  context.__job = {
    topHistory: [], lastStabilitySims: -1, stableHint: false,
  };
  context.__snapshot = {
    sims_done: 200,
    recommendations: [{ action_id: "a", visit_frac: 0.7 }],
  };
  assert.equal(vm.runInContext("streamingStableHint(__job, __snapshot)", context), false);
  assert.equal(vm.runInContext("streamingStableHint(__job, __snapshot)", context), false);
  assert.equal(context.__job.topHistory.length, 1);
  context.__snapshot.sims_done = 400;
  assert.equal(vm.runInContext("streamingStableHint(__job, __snapshot)", context), false);
  context.__snapshot.sims_done = 600;
  assert.equal(vm.runInContext("streamingStableHint(__job, __snapshot)", context), true);
});
