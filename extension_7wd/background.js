// Network proxy for the content script.
//
// WHY THIS EXISTS. An earlier version had content.js fetch the advisor directly
// and it failed with a bare network error on a live table: the BGA page is
// https and the advisor is http://127.0.0.1, so the request is blocked as mixed
// content before it leaves the browser. host_permissions does not exempt it. The
// background context is not a page, so it has no page CSP and no mixed-content
// rule -- which is why extension_kingdomino/background.js does the same thing.
//
// It is a dumb pipe on purpose: it takes a url and fetch init, returns status
// and body text, and holds no game or advisor knowledge.

const api = typeof chrome !== "undefined" && chrome.runtime ? chrome : browser;

api.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || msg.kind !== "swd-fetch") return false;
  (async () => {
    try {
      const res = await fetch(msg.url, msg.init || {});
      const body = await res.text();
      sendResponse({ ok: res.ok, status: res.status, body });
    } catch (err) {
      // status 0 distinguishes "never reached the host" from an HTTP error, so
      // the panel can say which.
      sendResponse({ ok: false, status: 0, error: String((err && err.message) || err) });
    }
  })();
  return true; // keep the channel open for the async sendResponse
});
