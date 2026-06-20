// background.js — localhost advisor fallback for BGA Can't Stop extension
//
// Content.js tries direct fetch first. If BGA/CSP/CORS blocks it, the request
// is retried here from the extension background context.

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
    signal: timeoutSignal(30000),
  });

  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (e) {
    throw new Error(`Advisor returned non-JSON HTTP ${response.status}: ${text.slice(0, 200)}`);
  }

  if (!response.ok) {
    throw new Error(`Advisor HTTP ${response.status}: ${JSON.stringify(data)}`);
  }

  return data;
}

browserAPI.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || message.action !== "recommend") return false;

  postJson(message.url, message.payload)
    .then((data) => sendResponse({ ok: true, data }))
    .catch((e) => sendResponse({ ok: false, error: String((e && e.message) || e) }));

  return true; // keep channel open for async response
});
