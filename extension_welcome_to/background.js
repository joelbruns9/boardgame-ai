// Network proxy for the content script.
//
// WHY THIS EXISTS. The BGA page is https and the advisor is http://127.0.0.1, so
// a fetch from any page context is blocked as mixed content before it leaves the
// browser, and `host_permissions` does not exempt a content script either. The
// background context is not a page, so it has no page CSP and no mixed-content
// rule. extension_7wd/background.js and extension_kingdomino/background.js do
// the same thing for the same reason; this was a real live failure, not a
// theoretical one (a bare "host offline" while the host was healthy).
//
// It is a dumb pipe on purpose: it takes a url and fetch init, returns status
// and body text, and holds no game or advisor knowledge.

const api = typeof chrome !== "undefined" && chrome.runtime ? chrome : browser;

api.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || msg.kind !== "wto-fetch") return false;
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
