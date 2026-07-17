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
