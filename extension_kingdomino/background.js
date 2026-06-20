const browserAPI = typeof browser !== "undefined" ? browser : chrome;

function timeoutSignal(ms) {
  if (typeof AbortSignal !== "undefined" && AbortSignal.timeout) {
    return AbortSignal.timeout(ms);
  }
  const controller = new AbortController();
  setTimeout(() => controller.abort(), ms);
  return controller.signal;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: timeoutSignal(120000),
  });

  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (e) {
    throw new Error(`Advisor returned non-JSON HTTP ${response.status}: ${text.slice(0, 300)}`);
  }

  if (!response.ok) {
    throw new Error(`Advisor HTTP ${response.status}: ${JSON.stringify(data)}`);
  }

  return data;
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
      target: { tabId },
      world: "MAIN",
      func: runner,
      args: [fnSource],
    });
    const result = results && results[0] && results[0].result;
    return result || { ok: false, error: "no result returned from executeScript" };
  } catch (e) {
    return { ok: false, error: "executeScript failed: " + String((e && e.message) || e) };
  }
}

browserAPI.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message) return false;

  // Proxy recommend POST to localhost (bypasses BGA page CSP / CORS).
  if (message.action === "recommend") {
    postJson(message.url, message.payload)
      .then((data) => sendResponse({ ok: true, data }))
      .catch((e) => sendResponse({ ok: false, error: String((e && e.message) || e) }));
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

  return false;
});