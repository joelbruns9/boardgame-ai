// Same background-network boundary as extension_7wd; restricted to advisor routes.
const api = typeof chrome !== "undefined" && chrome.runtime ? chrome : browser;
const routes = new Set(["/health", "/api/state", "/api/recommend", "/api/recommend/start",
  "/api/recommend/poll", "/api/recommend/stop", "/api/game_log",
  "/api/cantstop/win_probabilities"]);
api.runtime.onMessage.addListener((msg, sender, reply) => {
  if (msg?.kind !== "advisor-fetch") return false;
  (async () => {
    try {
      const url = new URL(msg.url);
      if (url.origin !== "http://127.0.0.1:8765" || !routes.has(url.pathname))
        throw new Error("Unsupported advisor URL");
      const res = await fetch(url, {...msg.init, signal: AbortSignal.timeout(60000)});
      reply({ok: res.ok, status: res.status, body: await res.text()});
    } catch (e) { reply({ok:false, status:0, error:String(e.message || e)}); }
  })();
  return true;
});
