# BGA 7 Wonders Duel Advisor (browser extension)

Watches a live 7 Wonders Duel table on Board Game Arena. When it becomes your
turn it captures the board, starts a search on a local advisor host, and shows
the top moves in a floating panel that keeps refining while you think.

Firefox is the primary target; the same unmodified directory loads in Chrome.

## 1. Start the advisor host

`pip install fastapi uvicorn` once, then from the repo root:

PowerShell (Windows):

```powershell
$env:SWD_ADVISOR_CHECKPOINT = "extension_7wd/candidate_0085.pt"
$env:SWD_ADVISOR_DEVICE = "cpu"
$env:SWD_ADVISOR_EXACT_ENDGAME = "1"
.venv\Scripts\python.exe -m uvicorn games.seven_wonders_duel.web_app:app --port 8000
```

bash:

```bash
SWD_ADVISOR_CHECKPOINT=extension_7wd/candidate_0085.pt \
SWD_ADVISOR_DEVICE=cpu \
SWD_ADVISOR_EXACT_ENDGAME=1 \
  uvicorn games.seven_wonders_duel.web_app:app --port 8000
```

Confirm it before touching the browser: `http://127.0.0.1:8000/health` should
return `{"ok":true,...}`.

The extension talks to `http://127.0.0.1:8000` and sends no checkpoint, so the
host's env default is what gets used. `http://127.0.0.1:8000/` also serves the
lab UI, which is a useful way to confirm the host is alive.

The live advisor checkpoint is `extension_7wd/candidate_0085.pt` (~15.8M
parameters, encoder `7wd-encoder-5`). The old
`runs/laptop_training_03_w7/checkpoints/current_best.pt` no longer loads at all:
the encoder signature changed.

**`cpu` was deliberate for the *1.03M* model.** At that size CPU and CUDA
measured within ~2% of each other, and CPU additionally skips CUDA context init
and leaves the GPU free for training runs — which matters for a host that sits
running for a whole game. That rationale no longer describes the 15.8M
checkpoint above; the device has not been re-measured at this size.

**Revisit this for a larger net.** At 11.4M parameters CUDA was 1.6× unbatched
and 10.2× at batch 32. The advisor now batches leaf evaluation by default
(`leaf_batch=16`), so that second figure should finally transfer — it did not
when every leaf was evaluated on its own. See
`games/seven_wonders_duel/ADVISOR_RUST_UNIFICATION.md` §2.4.

## 2. Load the extension

**Firefox** — `about:debugging#/runtime/this-firefox` → *Load Temporary Add-on*
→ pick `manifest.json`. Temporary add-ons are dropped when Firefox restarts.

**Chrome** — `chrome://extensions` → enable *Developer mode* → *Load unpacked*
→ pick this directory.

## 3. Play

Open a table. On your turn the panel appears top-right (drag it by the `::`
handle) and fills in as search deepens. It stops the old search and starts a new
one whenever the position changes.

While you play, two things are also written to `runs/seven_wonders_duel/bga_game_log/`
in the background: every position the advisor was asked about, and BGA's own
notification packets. The packets are the input to the differential harness,
which replays the game and checks our engine's arithmetic against BGA's:

```bash
python -m games.seven_wonders_duel.bga_differential
```

**Open the table before the first move**, or at least reload it if you join
late: capture starts when the tab does, and the harness refuses to replay a
game whose move sequence has holes rather than report mismatches that are
really just a missing move.

## How it works

Two halves, because neither can do the whole job:

| file | world | responsibility |
|---|---|---|
| `page_bridge.js` | MAIN (page) | sees `window.gameui`, detects your turn, captures the board |
| `content.js` | isolated | renders the panel, proxies network via the background |
| `background.js` | extension | the only place that reaches the advisor |

A content script cannot read `window.gameui` — isolated world, and Firefox's
`wrappedJSObject` escape hatch does not exist in Chrome. So the bridge runs in
the page and hands data over by `postMessage`.

**Network has to go through the background script.** The advisor is
`http://127.0.0.1` while BGA is `https`, so the request is blocked as mixed
content from *any* page context -- and `host_permissions` does not exempt a
content script either. This was not theoretical: the first live run failed with
a bare `host offline` while the host was healthy. The background context is not
a page, so no page CSP or mixed-content rule applies. `extension_kingdomino`
does the same thing for the same reason.

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

### The victory-type outlook

Above the move list, the panel shows **how** the net expects the game to end,
not just who wins — fetched once per position from `/api/state`.

For 7WD that is often the more actionable fact. The committed Age III capture
reads a **−1.0 VP margin** — losing on points — alongside a **94% scientific
win**, because the game finishes before scoring ever happens. A single win
percentage cannot express that.

It comes from the net's `joint7` head (winner × victory type) plus the VP-margin
and science forecasts, all of which were computed on every evaluation and thrown
away until now. One root evaluation, so it appears immediately and does not
change as search deepens.

### Exact endgames

With `SWD_ADVISOR_EXACT_ENDGAME=1`, the host runs the Rust solver concurrently
with neural MCTS. A fitted solve-cost model decides which Age III positions are
predicted to need at most 10 million nodes; the old fixed card-count cutoff is
not used. An admitted solve has no node cutoff and runs until it completes or
reaches its 30-second wall-clock cap. If it times out, the panel simply keeps
the neural-search answer. If it succeeds, its annotation is folded into the
live snapshot and preserved across later MCTS updates.

The solver status strip is always explicit: it reports disabled, ready for Age
III, solving concurrently, skipped with the predicted-node comparison, timed
out, errored, or solved with elapsed time and actual nodes. Solver silence is
never used as a status.

When a proof succeeds, the panel replaces Q values with exact outcomes (or an
exact win percentage when chance genuinely matters) and marks every tied best
move. A +1/-1 expectation is displayed as guaranteed even if the traversal saw
chance: a boundary expectation cannot contain a different positive-probability
outcome. An expectimax zero is not called a guaranteed draw.

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
* **Run against a live table 2026-08-02**, and the advice looked sound. That
  session found three bugs, all since fixed: mixed-content blocking on the
  content-script fetch, a rejected position latching instead of retrying, and
  the host's rejection reason being swallowed. **Not re-tested since the Rust
  searcher, the batched evaluation boundary and the victory-type outlook
  landed** — the searcher is now ~19× faster than during that session.

## Troubleshooting

| panel says | meaning |
|---|---|
| `host offline` | the request never reached the host: uvicorn not running, wrong port, or blocked in the browser |
| `host error 4xx/5xx` | the host answered and rejected it -- look at the uvicorn terminal for the traceback |
| `capture failed` | a DOM selector missed; the message names it. Reloading the table is the quick workaround and confirms it is a freshness problem |
| `waiting for your turn` | working as intended — it is the opponent's move |
| a dashed placeholder instead of a card | the name lookup missed; cosmetic only |
