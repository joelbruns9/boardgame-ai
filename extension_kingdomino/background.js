const browserAPI = typeof browser !== "undefined" ? browser : chrome;

function timeoutSignal(ms) {
  if (typeof AbortSignal !== "undefined" && AbortSignal.timeout) {
    return AbortSignal.timeout(ms);
  }
  const controller = new AbortController();
  setTimeout(() => controller.abort(), ms);
  return controller.signal;
}

// Startup diagnostics — visible in about:debugging → Inspect. Confirms the
// background script initialized and whether fetch exists in its context.
console.log("[bg] background.js loaded, fetch available:",
  typeof fetch !== "undefined");

// One-shot connectivity probe: does the background context reach the local
// advisor server at all? Fires immediately on load (independent of any
// content-script message), so a FAILED here isolates network reachability
// from message-passing problems.
fetch("http://127.0.0.1:8000/api/recommend", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ engine: "greedy", num_simulations: 0 }),
  signal: timeoutSignal(5000),
})
  .then((r) => r.json())
  .then((d) => console.log("[bg] startup connectivity test: OK, engine:", d.engine))
  .catch((e) => console.error("[bg] startup connectivity test FAILED:",
    e.name, e.message));

async function postJson(url, payload) {
  return requestJson(url, "POST", payload);
}

async function requestJson(url, method, payload) {
  const init = {
    method: method || "GET",
    headers: { "Content-Type": "application/json" },
    signal: timeoutSignal(120000),
  };
  if (payload !== undefined && payload !== null) init.body = JSON.stringify(payload);
  const response = await fetch(url, init);
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; }
  catch (e) { throw new Error(`Advisor returned non-JSON HTTP ${response.status}: ${text.slice(0, 300)}`); }
  if (!response.ok) throw new Error(`Advisor HTTP ${response.status}: ${JSON.stringify(data)}`);
  return data;
}

async function downloadJson(filename, data) {
  if (!browserAPI.downloads || !browserAPI.downloads.download) {
    throw new Error("downloads API is not available; reload the extension after adding the downloads permission");
  }
  const text = JSON.stringify(data, null, 2);
  const url = "data:application/json;charset=utf-8," + encodeURIComponent(text);
  const downloadId = await browserAPI.downloads.download({
    url,
    filename,
    saveAs: false,
    conflictAction: "uniquify",
  });
  return downloadId;
}

// Runs the content script's page-reader function in the page's MAIN world via
// chrome.scripting. This is NOT subject to the page's Content Security Policy
// (which blocks inline <script> injection), so it works on BGA where the old
// inline-injection approach was blocked.
//
// We reconstruct the reader function from its source string (passed by the
// content script) and invoke it. The reader returns its result object directly.
async function readPageStateInTab(tabId, fnSource) {
  if (!tabId) {
    return { ok: false, error: "no tab id available for readPageState" };
  }
  if (!fnSource) {
    return { ok: false, error: "no reader function source provided" };
  }

  // Wrapper executed in the page MAIN world. It rebuilds the reader from source
  // and returns whatever the reader returns.
  function runner(source) {
    try {
      // eslint-disable-next-line no-new-func
      const fn = new Function("return (" + source + ")")();
      return fn();
    } catch (e) {
      return { ok: false, error: "reader execution failed: " + String((e && e.message) || e) };
    }
  }

  try {
    const results = await chrome.scripting.executeScript({
      target: { tabId, allFrames: true },
      world: "MAIN",
      func: runner,
      args: [fnSource],
    });
    if (!results || !results.length) {
      return { ok: false, error: "executeScript returned no result rows" };
    }

    const rows = results.map((row) => ({
      frameId: row.frameId,
      result: Object.prototype.hasOwnProperty.call(row, "result") ? row.result : undefined,
    }));
    const okRow = rows.find((row) => row.result && row.result.ok);
    if (okRow) {
      return okRow.result;
    }

    const usefulError = rows.find((row) => row.result && row.result.error);
    if (usefulError) {
      return {
        ...usefulError.result,
        frame_debug: rows.map((row) => ({
          frameId: row.frameId,
          ok: Boolean(row.result && row.result.ok),
          error: row.result && row.result.error ? row.result.error : row.result === undefined ? "undefined result" : null,
        })),
      };
    }

    return {
      ok: false,
      error: "reader returned no usable result from executeScript frames",
      frame_debug: rows.map((row) => ({
        frameId: row.frameId,
        resultType: row.result === undefined ? "undefined" : typeof row.result,
      })),
    };
  } catch (e) {
    return { ok: false, error: "executeScript failed: " + String((e && e.message) || e) };
  }
}

browserAPI.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message) return false;

  // Proxy recommend POST to localhost (bypasses BGA page CSP / CORS).
  if (message.action === "recommend") {
    postJson(message.url, message.payload)
      .then((data) => {
        console.log("[bg] recommend success, keys:", Object.keys(data || {}));
        sendResponse({ ok: true, data });
      })
      .catch((e) => {
        console.error("[bg] recommend FAILED:", e.name, e.message, String(e));
        sendResponse({ ok: false, error: String((e && e.message) || e) });
      });
    return true;
  }

  // Read BGA page state via chrome.scripting (bypasses inline-script CSP).
  if (message.action === "readPageState") {
    const tabId = sender.tab && sender.tab.id;
    readPageStateInTab(tabId, message.fnSource)
      .then((result) => sendResponse(result))
      .catch((e) => sendResponse({ ok: false, error: String((e && e.message) || e) }));
    return true;
  }


  if (message.action === "advisorRequest") {
    requestJson(message.url, message.method, message.payload)
      .then((data) => sendResponse({ ok: true, data }))
      .catch((e) => sendResponse({ ok: false, error: String((e && e.message) || e) }));
    return true;
  }

  if (message.action === "downloadProbe") {
    downloadJson(message.filename, message.probe)
      .then((downloadId) => sendResponse({ ok: true, downloadId }))
      .catch((e) => sendResponse({ ok: false, error: String((e && e.message) || e) }));
    return true;
  }

  if (message.action === "saveProbe") {
    postJson(message.url, { filename: message.filename, probe: message.probe })
      .then((data) => sendResponse({ ok: true, data }))
      .catch((e) => sendResponse({ ok: false, error: String((e && e.message) || e) }));
    return true;
  }

  return false;
});
