const browserAPI = typeof browser !== "undefined" ? browser : chrome;
const usesPromiseAPI = typeof browser !== "undefined" && browserAPI === browser;

const DEFAULTS = {
  engine: "nn",
  sims: 800,
  checkpoint: "",
};

const statusEl = document.getElementById("status");
const captureBtn = document.getElementById("captureBtn");
const debugBtn = document.getElementById("debugBtn");
const engineEl = document.getElementById("engine");
const simsEl = document.getElementById("sims");
const checkpointEl = document.getElementById("checkpoint");

function getStorage(keys) {
  return new Promise((resolve, reject) => {
    try {
      if (usesPromiseAPI) {
        browserAPI.storage.local.get(keys).then((value) => resolve(value || {}), reject);
        return;
      }
      const result = browserAPI.storage.local.get(keys, (value) => resolve(value || {}));
      if (result && typeof result.then === "function") result.then((value) => resolve(value || {}), reject);
    } catch (e) {
      reject(e);
    }
  });
}

function setStorage(values) {
  return new Promise((resolve, reject) => {
    try {
      if (usesPromiseAPI) {
        browserAPI.storage.local.set(values).then(resolve, reject);
        return;
      }
      const result = browserAPI.storage.local.set(values, () => resolve());
      if (result && typeof result.then === "function") result.then(resolve, reject);
    } catch (e) {
      reject(e);
    }
  });
}

async function loadOptions() {
  const stored = await getStorage(["kingdomino_engine", "kingdomino_sims", "kingdomino_checkpoint"]);
  engineEl.value = stored.kingdomino_engine || DEFAULTS.engine;
  simsEl.value = String(stored.kingdomino_sims || DEFAULTS.sims);
  checkpointEl.value = stored.kingdomino_checkpoint || DEFAULTS.checkpoint;
}

async function saveOptions() {
  const sims = Number(simsEl.value);
  await setStorage({
    kingdomino_engine: engineEl.value || DEFAULTS.engine,
    kingdomino_sims: Number.isFinite(sims) && sims > 0 ? Math.round(sims) : DEFAULTS.sims,
    kingdomino_checkpoint: checkpointEl.value.trim(),
  });
}

function sendToActiveTab(message) {
  function queryTabs() {
    return new Promise((resolve, reject) => {
      try {
        if (usesPromiseAPI) {
          browserAPI.tabs.query({ active: true, currentWindow: true }).then((tabs) => resolve(tabs || []), reject);
          return;
        }
        const result = browserAPI.tabs.query({ active: true, currentWindow: true }, (tabs) => resolve(tabs || []));
        if (result && typeof result.then === "function") result.then((tabs) => resolve(tabs || []), reject);
      } catch (e) {
        reject(e);
      }
    });
  }

  function sendMessage(tabId) {
    return new Promise((resolve) => {
      try {
        if (usesPromiseAPI) {
          browserAPI.tabs.sendMessage(tabId, message).then(
            (response) => resolve(response || { ok: false, error: "no response from content script" }),
            (err) => resolve({ ok: false, error: String((err && err.message) || err) })
          );
          return;
        }
        const result = browserAPI.tabs.sendMessage(tabId, message, (response) => {
          const err = browserAPI.runtime.lastError;
          if (err) resolve({ ok: false, error: err.message });
          else resolve(response || { ok: false, error: "no response from content script" });
        });
        if (result && typeof result.then === "function") {
          result.then(
            (response) => resolve(response || { ok: false, error: "no response from content script" }),
            (err) => resolve({ ok: false, error: String((err && err.message) || err) })
          );
        }
      } catch (e) {
        resolve({ ok: false, error: String((e && e.message) || e) });
      }
    });
  }

  return new Promise((resolve) => {
    queryTabs().then((tabs) => {
      if (!tabs || tabs.length === 0) {
        resolve({ ok: false, error: "no active tab found" });
        return;
      }
      sendMessage(tabs[0].id).then(resolve);
    }, (err) => resolve({ ok: false, error: String((err && err.message) || err) }));
  });
}

async function capture(mode) {
  await saveOptions();
  statusEl.textContent = mode === "debug" ? "Capturing BGA state..." : "Querying advisor...";
  const response = await sendToActiveTab({ action: mode === "debug" ? "debugCapture" : "capture" });

  if (response.ok) {
    if (mode === "debug") {
      statusEl.textContent = "Debug capture saved. Open the page overlay or extension storage for details.";
      return;
    }
    const value = response.response && typeof response.response.value === "number"
      ? ` · value ${response.response.value.toFixed(3)}`
      : "";
    statusEl.textContent = `Advisor OK via ${response.transport || "unknown"}${value}`;
  } else if (response.skipped) {
    statusEl.textContent = `Skipped: ${response.error || "not ready"}`;
  } else {
    statusEl.textContent = `Error: ${response.error || "unknown error"}`;
  }
}

captureBtn.addEventListener("click", () => capture("recommend"));
debugBtn.addEventListener("click", () => capture("debug"));
loadOptions().catch((e) => {
  statusEl.textContent = String((e && e.message) || e);
});
