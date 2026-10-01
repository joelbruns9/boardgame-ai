# BGA Welcome To Advisor (browser extension)

Watches a live *Welcome To...* table on Board Game Arena. When a decision is in
front of you it captures the board, starts a search on a local advisor host, and
shows a floating panel that keeps refining while you think.

Firefox is the primary target; the same unmodified directory loads in Chrome.

**This is a diagnostic instrument first and a helper second.** The model is not
strong yet — S2 reaches 32.1 points where GreedyBot reaches 50.8 — so the panel
is built to show *why*, not just to name a move. It puts three things on screen
that a "best move" overlay would drop:

* the **prior** beside the visit count. "The net wanted this and search talked
  it out of it" and "the net never looked at it" are different faults.
* the net's **forecast**: predicted final score for every seat, split into
  components, plus what it thinks each City Plan will do and how many turns are
  left. A sane move list sitting on top of a plainly wrong forecast localises
  the problem in one glance.
* every **warning**, including which auxiliary heads the served checkpoint never
  trained. An untrained head returns exactly 0.5, which is indistinguishable
  from a real coin flip unless it is said out loud.

## 1. Start the advisor host

`pip install fastapi uvicorn` once, then from the repo root:

PowerShell (Windows):

```powershell
$env:WTO_ADVISOR_CHECKPOINT = "runs/welcome_to_s2/continuation_20x500_01/candidate_iter_0046.pt"
$env:WTO_ADVISOR_DEVICE = "cpu"
.venv\Scripts\python.exe -m uvicorn games.welcome_to.web_app:app --port 8001
```

bash:

```bash
WTO_ADVISOR_CHECKPOINT=runs/welcome_to_s2/continuation_20x500_01/candidate_iter_0046.pt \
WTO_ADVISOR_DEVICE=cpu \
  uvicorn games.welcome_to.web_app:app --port 8001
```

Confirm it before touching the browser: `http://127.0.0.1:8001/health` should
return `{"ok":true,...}` and name the checkpoint it loaded.

**Port 8001, not 8000.** The 7WD advisor owns 8000 and both hosts get left
running; a shared port would silently serve one game's extension from the
other's model. `test_extension_assets.py` pins it.

**`cpu` is right for this net.** Measured 29 simulations/second on CPU for the
3.94M-parameter S2 net, and a simulation here is a leaf evaluation plus a
rollout to the next turn boundary — the batch width an advisor can offer (one
position) is too small for CUDA to pay for its own context.

### Other environment

| variable | effect |
|---|---|
| `WTO_ADVISOR_CHECKPOINT` | the checkpoint served by default; S0 and S2 formats both load |
| `WTO_ADVISOR_DEVICE` | `cpu` (default) or `cuda` |
| `WTO_ADVISOR_ROUNDABOUT_PASS` | `1` gives the search back the roundabout pass — see *Known model behaviour* below |

## 2. Load the extension

**Firefox** — `about:debugging#/runtime/this-firefox` → *Load Temporary Add-on*
→ pick `manifest.json`. Temporary add-ons are dropped when Firefox restarts.

**Chrome** — `chrome://extensions` → enable *Developer mode* → *Load unpacked*
→ pick this directory.

## 3. Play

Open a table. The panel appears top-right (drag it by the `::` handle) and fills
in as the search deepens. It stops the old search and starts a new one whenever
the position changes.

Every position the advisor is asked about is also written to
`runs/welcome_to/bga_game_log/`. That costs one request and turns ordinary play
into a corpus of real human positions — both to restart self-play from and to
check the engine against.

**Open the table before your first turn**, or reload it if you join late. The
card ledger (below) only knows what it has seen, and a late start means the
discard pile is partly guessed — the panel says so when that happens.

## What it advises on, and what it does not

| BGA state | panel |
|---|---|
| `chooseCards` | ranks whole moves: which combination *and* where the number goes |
| `writeNumber` | the same list — see below |
| `buildRoundabout`, `actionSurveyor`, `actionEstate`, `actionPark`, `actionPool`, `actionBis` | ranks that decision |
| `choosePlan`, `validatePlan`, `askReshuffle` | ranks that decision |
| everything else | "waiting" |

**At `writeNumber` the panel answers the question one step earlier.** The
model's action vocabulary folds "take a combination" and "write the number" into
a single macro — you pick a combination *for* a placement — so `WRITE_NUMBER`
is inside a move rather than a decision of its own. And nothing has happened
yet: BGA has logged which stack you took and written nothing, and *restart* puts
you back at `chooseCards`. So the advisor backs up to the card choice, which
costs nothing and answers the more useful question: was the combination you just
took the right one?

**Base game and the advanced variant, 2–4 seats.** Expert mode (six ordered card
pairs) has no macro representation and the trained net has never seen it; the
seasonal boards (Ice Cream, Christmas, Easter) are not implemented at all. Both
are refused by name, not silently mis-read.

## How it works

Three halves, because none of them can do the other's job:

| file | world | responsibility |
|---|---|---|
| `page_bridge.js` | MAIN (page) | sees `window.gameui`, detects a live decision, captures the board |
| `content.js` | isolated | renders the panel, proxies network via the background |
| `background.js` | extension | the only place that reaches the advisor |

A content script cannot read `window.gameui` — isolated world, and Firefox's
`wrappedJSObject` escape hatch does not exist in Chrome. So the bridge runs in
the page and hands data over by `postMessage`.

**Network has to go through the background script.** The advisor is
`http://127.0.0.1` while BGA is `https`, so the request is blocked as mixed
content from *any* page context — and `host_permissions` does not exempt a
content script either. This was not theoretical: the first live 7WD run failed
with a bare `host offline` while the host was healthy. `extension_7wd` and
`extension_kingdomino` do the same thing for the same reason.

**The page world is shared with every other BGA advisor you have loaded.** A
top-level `function foo()` in an injected classic script becomes `window.foo`,
and `extension_7wd/bga_snippet.js` declares `findGameWindow`, `findGameWindows`,
`seatOf`, `_required` and `captureForAdvisor` too. With both extensions loaded,
whichever content script ran second silently replaced the other's — and since
this game's `findGameWindow` looks for `constructionCards`, it threw on a 7
Wonders Duel table and that panel just went quiet. So these page scripts publish
exactly one name, `window.__WTO_ADVISOR__`, and reach everything else through
it. `test_extension_assets.py` fails if any two extensions in this repo ever
share a page-world name again. (7WD's own globals are still unguarded; nothing
collides with them today, but namespacing that one too would make the rule hold
by construction rather than by test.)

`bga_snippet.js` is a **copy** of `games/welcome_to/bga_snippet.js` — an
extension has to be self-contained. `test_extension_assets.py` fails if the two
drift.

### What gamedatas will not tell you

`gameui.gamedatas` is the page-load payload, patched by whichever notification
handlers bother to write back into it. For this game:

* **fresh** — `players[].scoreSheet` and `planValidations`, rewritten by
  `notif_updatePlayersData` at `stApplyTurn`; `gamestate.name` and `.args`,
  because Welcome To's turn states are `type => private` and BGA's own
  `setupPrivateState` writes the private state name into that field.
* **stale** — `constructionCards`, `turn`, `cardsLeft`. `notif_newCards` touches
  only the DOM. All three are re-read from the DOM by `captureTablePatch`.

**And the seat is not in `gamedatas` either.** `gamedatas.me_id` is a field a
game's own `getAllDatas` chooses to send — 7 Wonders Duel sends one, Welcome To
does not. The seat comes from `gameui.player_id`, the BGA *framework* field
(`bga-framework.d.ts:890`) that `welcometo.js` itself uses everywhere; a
spectator is detected by `gameui.isSpectator`. Reading `me_id` here produced
exactly one symptom on the first live table: *"this frame is not seated at the
table (me_id missing)"* on a perfectly ordinary game.

Two things were never in `gamedatas` at all:

**Your current turn.** Everything you scribble mid-turn goes to the DOM alone.
That is a feature, not a bug: the un-updated `scoreSheet` is exactly the
turn-start snapshot the engine calls `public_sheets`, so the scrape lands on the
engine's own information-set boundary with nothing to hide by hand. Your own
marks are read back off the sheet by `captureTurnMarks` and **replayed as engine
actions** rather than poked into the engine's turn context — which means the
engine refuses anything the rules would not have offered, so a capture taken
mid-animation fails loudly instead of being advised on. The panel retries.

**The discard pile.** `cardsLeft` says how many cards have gone, never which. A
counting player knows, because every card that went to the discard was face-up
on the table first — and the deck's composition is what the model's number
forecasts and the whole reshuffle decision rest on. In standard mode BGA never
removes a construction card from its stack div, so the table's whole history is
still in the DOM, and each card carries its own database id: a ledger keyed by
that id counts every card exactly once with no reasoning about turns. It is
mirrored into `localStorage` under the table id so an F5 does not throw it away,
and it resets when `cardsLeft` goes *up*, which is the only signal that somebody
shuffled the discard back in.

### Seat 0 is you

Welcome To is played simultaneously and the engine serialises a turn into
consecutive private turns — safe here because nothing a player does during a
turn changes what another player may do during it. So seat *order* carries no
information, and the mapper reindexes it: you become seat 0 and the others
follow in BGA turn order. That makes the reconstruction self-consistent (every
other seat is at its turn-start sheet because nobody has acted yet this turn)
where keeping your real seat index would leave the players before you looking as
if they had skipped a turn.

## Known model behaviour worth knowing before you read the panel

**Roundabouts, in the advanced variant — a strategy, not a bug.** The served S2
checkpoint puts ~0.94 of its prior on *Build a roundabout* whenever one is
offered, opens on turn 1, and burns both of its two before turn 3 in every game.
That looks exactly like the artifact `SEARCH_SPEC.md` §5.1a warned about, and it
is not. Measured over 25 paired seeds, 2 seats, raw-policy argmax over whole
macros:

| advanced variant | total | estates | plans | roundabout penalty |
|---|---|---|---|---|
| net as trained | 43.6 | 30.6 | 5.0 | −8.0 |
| net, roundabout masked out | 25.2 | 11.3 | 0.4 | 0 |
| GreedyBot | 54.4 | 36.0 | 3.9 | −4.1 |

The habit is worth about **+18 points**, and it is the net's main source of
estate scoring: it completes 5.96 housing estates a game with roundabouts
against 3.04 without. A roundabout breaks the ascending-number chain, which is
what closes a street segment into a finished estate — and greedy uses the same
move, 1.16 times a game. The prior did sharpen over training (0.42 at S0, 0.28
at S2 iteration 1, 0.94 at iteration 46), but it sharpened onto something that
pays.

Two caveats on that table. Masking the top action forces the argmax onto a move
the policy never optimised, so the middle row overstates the loss. And none of
it uses search. The **base game** is the clean comparison, because roundabouts
do not exist there at all:

| base game | total | estates | plans |
|---|---|---|---|
| net as trained | 22.6 | 8.8 | 0.4 |
| GreedyBot | 51.4 | 30.2 | 3.6 |

That is the real weakness, and it is what to read the panel for: **without the
roundabout the net cannot build housing estates or finish City Plans.** Its
estate score is a quarter of greedy's and it completes 0.4 plans a game.

`WTO_ADVISOR_ROUNDABOUT_PASS=1` gives the search the roundabout pass back, which
is how you ask "should I decline this one?" — the pruned search cannot answer
that because the pass is not in its action set.

## Troubleshooting

| panel says | meaning |
|---|---|
| `host offline` | the request never reached the host: uvicorn not running, wrong port, or blocked in the browser |
| `host error 4xx/5xx` | the host answered and rejected it — the sub-line carries its reason, and the uvicorn terminal has the traceback |
| `retrying n/5` | the capture disagreed with the rules, almost always because BGA was still animating a mark into place; it re-captures |
| `capture failed` | a DOM selector missed; the message names it. Reloading the table is the quick workaround |
| `waiting` / `turn done` | working as intended — there is no decision in front of you |
| a warning about the card ledger | you joined or reloaded mid-game, so part of the discard pile is guessed; number forecasts and the reshuffle read are approximate |
| a warning about untrained heads | the checkpoint predates the plan-outcome and end-trigger heads, which are served as a meaningless 0.5 |

## Status

Built 2026-08-29. The Python half is covered end to end: `test_bga_extract.py`
round-trips every reachable capture back through the mapper and demands the
rebuilt position match the one it was taken from, and `test_advisor_adapter.py`
drives the adapter through the real host. **Not yet run against a live BGA
table** — the DOM selectors are read off `BGA Files/welcometo` rather than
observed, so the first live session should be treated as a shakedown.
