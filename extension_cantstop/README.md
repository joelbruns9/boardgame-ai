# Can't Stop BGA advisor — first integration

Uses the existing `games.advisor` FastAPI host and its recommendation,
state-validation, and game-log contracts. The browser uses the newer
`extension_7wd` architecture: a MAIN-world capture bridge, isolated panel,
and background network proxy. It does not use the legacy MCTS endpoint.
The game-specific capture is isolated in `bga_snippet.js`.

## Start

From the boardgame-ai-cantstop checkout:

```powershell
powershell -ExecutionPolicy Bypass -File games\cantstop\run_advisor.ps1
```

The launcher selects this checkout's venv or the sibling boardgame-ai venv.
Default model: `extension_cantstop/models/cantstop_generalist_iter0154.pt`,
committed with the extension so it does not depend on the untracked `runs/`
folder. It is a copy of `runs/p4_night/iter_0154.pt` (sha256 `c615b682e4a3...`),
the all-variant generalist that scored best on the Rule of 28 yardstick
(2026-10-01). CUDA, port 8765.
Use `-Device cpu` if needed. Or, with an activated project environment:

```powershell
python -m uvicorn games.cantstop.web_app:app --host 127.0.0.1 --port 8765
```

Health: http://127.0.0.1:8765/health.
Override the model with `CANTSTOP_ADVISOR_CHECKPOINT`; override device with
`CANTSTOP_ADVISOR_DEVICE`. The model is loaded once on the first decision.

## Load

Disable the old Can't Stop extension to avoid duplicate overlays.
Firefox: about:debugging#/runtime/this-firefox → Load Temporary Add-on →
select this directory's manifest.json.
Chrome: copy this directory to a separate local folder, replace that copy's
manifest.json with manifest_chrome.json, then load that folder unpacked.

Player count, columns-to-win, and blocking are read automatically from BGA.
No rule toggle is needed. The panel displays the detected setting.

The panel waits for a stable capture, checks the board on the server, and
ranks moves by estimated win probability. It automatically clears stale advice.
**Refresh** retries capture/server failures. **Export capture** downloads
the raw board for diagnosis. No game actions are clicked automatically.

## What is verified

- Solver advice matches the existing Rust solver for 2–4 players and both
  decision phases, including blocked stopping.
- Shared HTTP blocking and start/poll APIs, plus validated game logging.
- Real iteration-80 checkpoint inference on CUDA.
- Browser capture/settling on synthetic data, JavaScript syntax.

## Live BGA verification still required

This version has NOT yet been checked against a current live BGA table.

1. Confirm all three player colors/order and each saved marker's height.
2. Confirm black runner absolute heights, including a runner above saved progress.
3. Compare listed column moves with BGA, including doubles, partial moves,
   claimed columns, runner limits, and blocking.
4. Check continueChoice, forced rolling, bust, claim, and turn transitions.
5. Export captures at these points so they become real regression fixtures.

Coordinates follow the OLD EXTENSION'S implemented conversion:
absolute climbed steps = column height - BGA data-height.
The older recon document states a contradictory interpretation.
The runner is absolute, never the old engine's increment over saved progress.
Claimed-column inert saved markers are removed; duplicate active markers are
rejected as unsettled captures. DOM markers determine claimed columns. BGA
player metadata scores can remain stale after a claim and are diagnostic only.

BGA movement_variant is read automatically: 0 = standard, 2 = forced movement
(our blocking rule). Flag 1 is the unsupported Jump variant and is rejected.
Unknown flags are rejected. Verified against BGA cantstop.js version 260212-1713
getVariantText on 2026-09-28; standard flag 0 also appears in live table 923012474.
Source: https://x.boardgamearena.net/data/themereleases/260923-0920/games/cantstop/260212-1713/cantstop.js The BGA possibleMoves structure is captured for comparison,
but is not yet mechanically cross-checked against engine legal moves.
A stable DOM snapshot reduces animation races; live fixtures are still needed
to establish extraction correctness.

Validated positions, raw captures, and recommendations are logged under
runs/cantstop/bga_game_log/. Logs use the shared host format.

## Search

Complete current-turn Rust dynamic-programming solve with the trained value net
for later turns. Values are model estimates, not calibrated confidence bounds.
No MCTS visits or simulation controls: the host receives zero action visits and
ranks by value. Its probability convention is preserved using q = 2*p - 1,
including games with more than two players.

A solve is atomic: cancelling a job invalidates its display immediately but
does not interrupt a Rust solve already in progress. Model inference is
serialized per adapter to bound simultaneous memory use.

## Checks

```powershell
python -m pytest games/cantstop/tests/test_advisor.py games/advisor/test_jobs.py games/advisor/test_game_log.py -q
node extension_cantstop/test_capture.cjs
```


## Combined dice-choice and stop/roll advice

Each dice-choice recommendation includes a follow_up, such as "then stop and
bank" or "then roll again (stopping is blocked)". Both decisions come from the
same Rust turn solve. The browser uses one /api/recommend request for that solve.
When BGA advances to continueChoice, the already-computed follow-up is reused
only if the entire server-validated position matches that move's predicted
position. A mismatch, page reload, or direct entry at continueChoice triggers
fresh advice. Health/state-validation requests are separate from inference.
Reload the extension, refresh BGA, and restart the advisor after updating.

## Connection recovery

Transient transport failures and HTTP 408/429/500/502/503/504 responses retry
up to twice, after 0.5 and 1.5 seconds. Retries stop when the position changes.
Unsettled-board validation failures trigger up to two fresh captures instead
of retrying the same stale data. Other validation errors remain visible.
The panel's Refresh button retries without reloading BGA. Export capture now
includes the last 30 request errors (endpoint, status, attempt, time, message).
Run node extension_cantstop/test_requests.cjs for recovery checks.

Board reads are event-driven: when BGA enters a decision state
(gameui.onEnteringState, hooked by the timing probe), the bridge reads the
board at once and every 16 ms after, and sends it as soon as two reads in a
row match and show that state -- typically one frame after the state change.
Measured on a live game (2026-10-01): BGA applies every marker move before it
enters the next decision state, and the board then stays put. The 100 ms
polling with its 200 ms / 1.2 s settle waits remains as the fallback (for
example if the hook is unavailable or a capture never settles within 12
reads). Background tabs are throttled by the browser, which delays both paths.
Regression: node extension_cantstop/test_event_capture.cjs.

Export capture also includes `page_timing` from the timing probe
(timing_probe.js): when BGA enters each state, when board markers or dice
change, and when the bridge first saw and finally read each board. It records
only event names and times, never packet contents or chat. Summarise one or
more exports with `python -m games.cantstop.timing_report <export.json>...` to
size capture delays from measurement. Probe regression:
node extension_cantstop/test_timing_probe.cjs.

## Game record (BGA packets)

`packet_recorder.js` hooks `gameui.notifqueue.onNotification` and forwards
this table's notification packets (never chat, never other tables) to
`/api/game_log` as `bga_packets` rows in the same per-table JSONL. Board
captures only see decision screens; the packets are BGA's own ordered record
of every roll -- busts included -- every pairing, stop, and the game end, so
`games/cantstop/luck.py` can read a game instead of inferring it. A page
reload re-sends the history (packet type "resend"), so opening the tab
mid-game backfills. Undelivered batches are retried every 5 s.
Regression: node extension_cantstop/test_packet_recorder.cjs.

## Dice luck so far

Under the win chances, each player's luck on their OWN rolls this game, so
"my dice were average, theirs were hot" reads directly (in a 2-player game
the effect on you is your number minus theirs):

- **pts**: win chance their dice gained (+) or lost (-), each roll measured
  against the exact average over all 1296 rolls (uses the net to weigh how
  much each roll mattered).
- **busts / exp**: actual busts vs the exact bust odds of each roll taken.
  No model.
- **progress**: the most progress their dice offered (best pairing, in
  columns: a step is 1/height of its column) minus the average, less bust
  losses beyond their expected size. No model; it does not know which
  columns matter, which the pts do.

Served by `/api/cantstop/luck` (games/cantstop/live_luck.py), which replays
the table's logged packet record, so it needs that record from the first
roll: if the tab was opened mid-game the block says so (a reload makes BGA
re-send the history). Refreshed after each logged packet batch, at most one
request in flight plus one queued. Regression:
node extension_cantstop/test_luck_panel.cjs.

## Player win chances

The top of the panel lists every player, starting player first (by BGA seat
number), with their colour and chance to win; the player to move is
highlighted. The numbers update on every roll by any player: the turn
solver's value after the roll, assuming best play for the dice choice and the
rest of the turn. A stop/roll decision keeps the roll's numbers. They assume
everyone plays like the model from here, so against human tables they are
"chances with best play", not a forecast of this table. Served by
`/api/cantstop/win_probabilities`, sharing the turn solve the move advice uses.
If the server lacks that route (an older advisor still running), the panel
says so. Regression: node extension_cantstop/test_win_probs.cjs.

## Opponents and full choice comparisons

Every player's decisions are always evaluated and logged: the player list at
the top needs each opponent's turn solve anyway, and the BGA game log then
holds every decision for later review. **Show opponents’ decisions** (on by
default, remembered by the extension) only controls whether an opponent's
options are displayed. Names identify whose decision is shown; percentages in
the options always mean that active player's chance to win.
Every distinct legal column selection is listed with separate Stop and Roll
values. Equivalent dice pairings leading to the same column move are combined.
An illegal stop is labeled blocked, never offered as a legal recommendation.
After a selection is made, the full comparison stays visible and the selected
move is highlighted. A matching validated position reuses the original result
without another NN call. During transitions previous options remain visible
with a waiting label; new decisions replace them when ready.
Panel regression: node extension_cantstop/test_panel.cjs.

## Turn lookup cache

The first recommendation solves the reachable remainder of the active turn.
Later dice rolls query that same server-side solution without evaluating the
model again. HTTP requests still validate each new board and retrieve its choices;
selecting dice continues to reuse the already displayed result in the browser.
The bridge tags turns while observing every player's phases. Table, turn, saved board, rules, active player, and model
identity must match; runner regression or an unreachable position forces a new
solve. Legacy requests without table_id and turn_id options remain uncached.
The cache retains at most four turns and 100,000 positions, with a fixed 15-minute
expiry. Expired, evicted, or oversized turns are solved again when needed.
The health response includes turn_cache counts; a cache hit reports
"Cached turn lookup complete". After updating, rebuild/install cantstop_rust,
restart the advisor server, reload the extension, and refresh the BGA page.
The Rust TurnSolver binding is thread-safe (checked by Rust/PyO3 at build time),
so request workers and cache expiry may safely share/release it.
Regression checks: games/cantstop/tests/test_advisor_cache.py and
node extension_cantstop/test_turn_identity.cjs.

## Faster same-turn advice

The bridge checks every 100 ms. Positions whose saved board, rules, player and
turn match the last validated recommendation, with a complete predicted runner
configuration, need only 200 ms of stability. New turns and unexpected/partial
board changes retain the 1,200 ms wait. Backend errors revoke the fast path.
Every recommendation still validates the captured board on the server.
Health information is reused until an error or panel Refresh. Dice rolls send
raw captures directly to /api/recommend, which already validates them; a separate
/api/state request is reserved for reusing a displayed post-selection result.
These are extension-only changes: reload the extension and refresh BGA.
Timing and request regression checks: node extension_cantstop/test_speed.cjs and
node extension_cantstop/test_panel.cjs. Timing checks use a simulated clock;
actual display latency also depends on BGA animations and browser scheduling.
