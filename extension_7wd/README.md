# BGA 7 Wonders Duel Advisor (browser extension)

Watches a live 7 Wonders Duel table on Board Game Arena. When it becomes your
turn it captures the board, starts a search on a local advisor host, and shows
the top moves in a floating panel that keeps refining while you think.

Firefox is the primary target; the same unmodified directory loads in Chrome.

## 1. Start the advisor host

```bash
pip install fastapi uvicorn
SWD_ADVISOR_CHECKPOINT=games/seven_wonders_duel/runs/laptop_training_03_w7/checkpoints/current_best.pt \
SWD_ADVISOR_DEVICE=cpu \
  uvicorn games.seven_wonders_duel.web_app:app --port 8000
```

The extension talks to `http://127.0.0.1:8000` and sends no checkpoint, so the
host's env default is what gets used. `http://127.0.0.1:8000/` also serves the
lab UI, which is a useful way to confirm the host is alive.

**`cpu` is deliberate, not an oversight.** Measured on an RTX 3070 Laptop,
3,000 sims on the same position:

| device | time | throughput |
|---|---|---|
| cpu | 0.67s | 4,485 sims/s |
| cuda | 0.65s | 4,593 sims/s |

A ~2% difference, inside the noise. Search evaluates **one leaf at a time**
against a **1.03M-parameter** net, which is far too little work per call to
amortize kernel-launch overhead — the bottleneck is the Python tree walk, not
the matmuls. CPU additionally skips CUDA context init at startup and leaves the
GPU free for training runs, which matters when the advisor sits running for a
whole game. Use `cuda` only if the CPU is otherwise busy.

## 2. Load the extension

**Firefox** — `about:debugging#/runtime/this-firefox` → *Load Temporary Add-on*
→ pick `manifest.json`. Temporary add-ons are dropped when Firefox restarts.

**Chrome** — `chrome://extensions` → enable *Developer mode* → *Load unpacked*
→ pick this directory.

## 3. Play

Open a table. On your turn the panel appears top-right (drag it by the `::`
handle) and fills in as search deepens. It stops the old search and starts a new
one whenever the position changes.

## How it works

Two halves, because neither can do the whole job:

| file | world | responsibility |
|---|---|---|
| `page_bridge.js` | MAIN (page) | sees `window.gameui`, detects your turn, captures the board |
| `content.js` | isolated | fetches the advisor, renders the panel |

A content script cannot read `window.gameui` — isolated world, and Firefox's
`wrappedJSObject` escape hatch does not exist in Chrome. So the bridge runs in
the page and hands data over by `postMessage`.

The split is not only about access. The advisor is `http://127.0.0.1` while BGA
is `https`, so a page-context fetch would be blocked as mixed content. A content
script with `host_permissions` is not subject to that, which is why all network
access lives on the isolated side.

`bga_snippet.js` is a **copy** of `games/seven_wonders_duel/bga_snippet.js` — an
extension has to be self-contained. `test_extension_assets.py` fails if the two
drift.

### Card art costs nothing

The panel is injected into the BGA page, so BGA's own stylesheet applies. A card
is a `div.building.building_small` with a `background-position` from
`spriteXY`, which resolves against the spritesheet the page has already loaded.
No images are bundled, art tracks BGA automatically, and nothing of BGA's is
redistributed.

Name lookups here are **case-insensitive** on purpose: BGA title-cases some
names (`Chamber Of Commerce`) where the engine follows the printed card
(`Chamber of Commerce`). Unlike the state mapping, a miss here is cosmetic — you
get a placeholder instead of a card, never a wrong position.

### Two display rules that come from measurement

* **The win percentage is withheld below 5,000 sims.** The same position read
  89% at 600 sims and 98.9% converged. Move *ranking* stabilises far earlier
  than the number does, so the list shows immediately and the percentage waits.
* **Rows under 2% of visits are dimmed.** Their Q values are essentially
  unsearched; showing `+0.985` on a 45-visit move next to a 53,000-visit one
  invites a bad decision.

## Limitations

* **Base game only.** Agora and Pantheon are rejected by the mapper — the
  trained net's action space does not include them.
* **No wonder-draft advice yet.** The draft is item A in
  `games/seven_wonders_duel/ADVISOR_IMPLEMENTATION_PLAN.md`; the bridge only
  wakes on `playerTurn` and the four mid-move choice states.
* **Not yet run against a live table.** Every piece is tested — capture,
  mapping, search, streaming, sprites — but this assembly has not been through a
  real game. Expect the first session to find something.

## Troubleshooting

| panel says | meaning |
|---|---|
| `host offline` | the uvicorn host is not running, or not on port 8000 |
| `capture failed` | a DOM selector missed; the message names it. Reloading the table is the quick workaround and confirms it is a freshness problem |
| `waiting for your turn` | working as intended — it is the opponent's move |
| a dashed placeholder instead of a card | the name lookup missed; cosmetic only |
