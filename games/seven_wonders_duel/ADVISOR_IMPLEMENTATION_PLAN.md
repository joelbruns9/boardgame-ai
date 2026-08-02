# 7WD BGA advisor — implementation plan

**Goal: play real games on Board Game Arena with live advice from a trained
net, including the wonder draft, without reloading the page between moves.**

This document is written to be picked up cold. It states what already exists,
what is missing, why each gap exists, and how to close it. Line numbers are as
of `main` at commit `0b09708`; find them by the quoted search strings if they
have drifted.

## Status summary

| # | Item | Size | Blocks live play? |
|---|---|---|---|
| A | Wonder-draft support in the scrape codec + BGA mapper | 1–2 days | yes — draft is 8 of ~78 decisions and shapes the whole game |
| B | Fresh board state without a page reload | 1–2 days | yes — otherwise every move needs an F5 |
| C | Host `{"bga": …}` branch | ~1 hour | yes, but trivial |
| D | Browser extension | 1–2 days | yes |
| E | Draft-preference extraction | ~half day | no — analysis, not play |

Do them in order. C is deliberately last-but-one: it is small, and its payload
shape is decided by A and B.

## What already works — do not rebuild any of this

| piece | file | notes |
|---|---|---|
| Game-agnostic advisor host | `games/advisor/` | FastAPI, resumable jobs (`start`/`poll`/`stop`), ranking |
| Adapter protocol | `games/advisor/contract.py:269` | `AdvisorAdapter`: `state_from_wire`, `action_views`, `open_search`, … |
| 7WD serve entry point | `games/seven_wonders_duel/web_app.py` | `uvicorn games.seven_wonders_duel.web_app:app --port 8000` |
| Lab UI | `games/seven_wonders_duel/web_static/index.html` | served at `/` |
| 7WD adapter | `advisor_adapter.py` | `state_from_wire` at line 147 |
| Scrape codec + determinizer | `advisor_scrape.py` | `determinize_observation` (59), `observation_to_wire` (176), `observation_from_wire` (230) |
| Exact endgame solver | `advisor_endgame.py` | |
| **BGA `gamedatas` → advisor wire** | `bga_extract.py:359` | `wire_from_bga(gamedatas, resample_seed=0)`, tested against two captured real positions in `testdata/bga_887892216_*.json` |
| Browser capture | `bga_snippet.js` | `captureBgaGamedatas()`, `captureAfterReload()` |
| Checkpoint wiring | `web_app.py` | `SWD_ADVISOR_CHECKPOINT` env, or per-request `checkpoint_path` |

**The net needs no work.** Use
`games/seven_wonders_duel/runs/laptop_training_03_w7/checkpoints/current_best.pt`
(S = 128×4, promoted at iteration 195, 84k games). Wonder-draft actions are
already in its action space and it is already trained on them — see item E.

## Design principle worth preserving

`bga_snippet.js` holds **no game knowledge**: it grabs
`window.gameui.gamedatas` verbatim and every mapping lives in Python, so a BGA
UI change makes `wire_from_bga` *raise* rather than silently produce a wrong
position. Item B necessarily puts some knowledge back in the browser. Do that
deliberately and keep it as thin as possible — see B's "keep the blast radius
small".

---

# A. Wonder-draft support

## Why it is rejected today

Two independent places reject it:

* `advisor_scrape.py:56` — `_SUPPORTED = (Phase.PLAY_AGE, Phase.COMPLETE)`, and
  `determinize_observation` raises for anything else (line 63). The stated
  reason is that the draft's *"hidden structure is not reconstructable from a
  single public observation"*.
* `bga_extract.py:24-26` / `UnsupportedBgaState` (line 91) — the mapper rejects
  the draft, the between-age `CHOOSE_NEXT_START_PLAYER` transition, and both
  expansions.

**The stated reason is too pessimistic.** The draft's hidden structure is a
uniform draw from a known multiset, which is exactly what the existing
determinizer already does for age decks. It is real work, not impossible work.

## The hidden information, precisely

7WD base has **12 wonders**. `game.py:296` deals 8 as
`wonder_groups = (wonders[:4], wonders[4:8])`; the remaining 4 are
`unused_wonders`. Play alternates over group 1, then group 2.

**Confirmed 2026-08-01: BGA does not reveal group 2 during the first round.**
So from a first-round public observation the unknown is a partition of the 8
not-yet-seen wonders into (group 2 | unused) — 4 and 4, C(8,4) = **70 equally
likely partitions**. In the second round group 2 is visible and only the 4
unused remain hidden, which never affects play.

Note the engine's `wonder_round` is **0-indexed**: 0 is the first group, 1 the
second. `advisor_scrape.py:100-101` setting `wonder_round = 1,
wonder_pick_index = 4` therefore means "second group, all four taken", i.e. a
finished draft — which is why it is correct for PLAY_AGE and must be branched
for a draft position.

Also hidden at draft time: **all three age decks**, which have not been dealt
yet. These are a uniform deal from the full card pool.

## What to change

**1. `advisor_scrape.py`**

* Add `Phase.WONDER_DRAFT` to `_SUPPORTED`.
* `observation_to_wire` / `observation_from_wire` must carry the draft fields.
  `PlayerObservation` (`game.py:222`) exposes `wonder_offer` but **not**
  `wonder_round`, `wonder_pick_index` or `first_player`. All three are exactly
  derivable, so do not widen the observation:

  ```
  picked      = len(cities[0].wonders) + len(cities[1].wonders)
  wonder_round      = picked // 4      # 0-indexed: 0 = first group, 1 = second
  wonder_pick_index = picked % 4
  ```

  `first_player` follows from the draft order. `_draft_order` (`game.py:351`)
  returns `(f, 1-f, 1-f, f)` for round 0 and `(1-f, f, f, 1-f)` for round 1, so
  given `active_player` from the observation:

  ```
  round 0:  first_player = active_player      if pick_index in (0, 3) else 1 - active_player
  round 1:  first_player = 1 - active_player  if pick_index in (0, 3) else active_player
  ```

  This matters because `pick_wonder` (`game.py:371`) asserts
  `active_player == _draft_order(wonder_round)[wonder_pick_index]`, and
  `determinize_observation` builds its state with `new_game(0, 0)` — i.e.
  `first_player = 0` — so a draft determinization **must** set it explicitly or
  that assertion fires.

* `wonder_groups` must also be reconstructed, because `pick_wonder` reads
  `wonder_groups[1]` when it flips the second group face-up (`game.py:381-385`):

  ```
  round 0:  group 0 = picked-so-far + current offer;  group 1 = 4 sampled from the unseen 8
  round 1:  group 0 = the first 4 picks;              group 1 = picks 5..n + current offer
  ```
* `determinize_observation` currently **fakes a finished draft** — lines 100-101
  hardcode `state.wonder_round = 1` and `state.wonder_pick_index = 4`, which is
  correct for a PLAY_AGE position and wrong for a draft one. Branch here.
* For a draft position: set the real `wonder_round` / `wonder_pick_index` /
  `wonder_offer`, partition `pool.wonders` (from `unseen_pool(obs)`, line 103)
  randomly into group 2 and `unused_wonders`, and deal all three age decks
  fresh. **Skip the current-age tableau block entirely** (lines 108-150): at
  draft time no age has been dealt, and that code assumes a live tableau with
  face-down slots.

**2. `bga_extract.py`**

* `_phase` (line 145) must map BGA's draft state name to `Phase.WONDER_DRAFT`
  instead of raising.
* Emit the draft wire: the current offer, whose turn it is, and each player's
  already-picked wonders. Wonders are keyed by **name strings** BGA already
  exposes (`wonders[id].name`), consistent with the rest of the mapper — no
  numeric-id alignment.
* Leave `CHOOSE_NEXT_START_PLAYER` and expansions rejected.

**3. Determinization variance is real here.** A PLAY_AGE query has one plausible
determinization of the current age; a round-1 draft query has 70 partitions ×
however many age deals. **Run several determinizations per draft query and
aggregate**, the way the existing scrape path does. Expect draft advice to be
noisier than play advice and say so in the UI.

## Verification

* Round-trip: a synthetic draft position through
  `observation_to_wire` → `observation_from_wire` → `determinize_observation`
  must produce a state whose `observation(0)` equals the input (the
  "public-exact" property the docstring at line 60 promises).
* The determinized state must be legal: `legal_action_indices` returns exactly
  the offered wonders, and every sampled partition uses each of the 12 wonders
  exactly once across (picked | offer | group 2 | unused).
* Capture a **real BGA draft position** to `testdata/bga_<table>_draft.json` and
  add it to `test_bga_extract.py`, matching the two existing play captures.
* Sampling uniformity: over many seeds, group-2 membership should be roughly
  uniform over the 70 partitions.

---

# B. Fresh state without a reload

## The problem

`gameui.gamedatas` is the **page-load** payload. BGA patches scores and wonders
from its notification stream but leaves four fields at their load-time values
(`bga_snippet.js:9-16`):

```
playerBuildings   discardedBuildings   militaryTrack   progressTokensSituation
```

`bga_extract._assert_fresh` (line 126) catches this by an internal-consistency
check — each player's count of science-bearing buildings in `playerBuildings`
must equal BGA's own `scienceSymbolCount` — and raises `StaleGamedata`. So a
stale read fails loudly rather than advising on a wrong position. Good, but it
means every query currently needs an F5.

## What is NOT available

**Re-fetching `gamedatas` without a full page load is not possible.** The BGA
framework's own type definitions (`BGA Files/sevenwonders/bga-framework.d.ts`,
line 883) document `gamedatas` as *"the initial set of data to init the game,
created at game start or by game refresh (F5)"*. There is no `getGameData`,
no refresh call, no notification-history fetch; the only `ajaxcall` is
deprecated and is for sending actions. Do not spend time looking for one.

(Those files are for **7 Wonders**, BGG 68448, 3–7 players — the wrong game.
Only the two framework-level files, `bga-framework.d.ts` and `_ide_helper.php`,
are game-agnostic and useful. If a `sevenwondersduel` dump ever becomes
available it would give exact state names, notification types and argument
shapes, which is most of the hard part of both A and B.)

## The approach: read the DOM for the four stale fields

This is the **in-repo precedent**. `extension_kingdomino/content.js` does exactly
this hybrid — `gamedatas` for what is fresh, DOM for what BGA does not expose or
leaves stale:

```js
const kings = others.querySelectorAll("[id^='kingdom_']");   // ~line 512
const castleEl = container.querySelector('.castle');          // ~line 539
placed = container.querySelectorAll("[class*='shadow-rotation-']");  // ~line 581
```

with an explicit comment at line 477 that the opponent's board is *"NOT exposed
in gamedatas"*. That extension works against live BGA today.

All four stale 7WD fields are plainly visible on screen: built buildings per
player, the discard pile, the military track position, and which progress tokens
remain available.

**Keep the blast radius small.** Do not rewrite the mapper to take DOM input.
Instead, have the browser produce a *patch*: capture `gamedatas` verbatim as
today, plus a small object holding only the four fields re-read from the DOM,
and overwrite them in Python before `wire_from_bga` runs. Then:

* the mapper keeps its single input shape and all its existing tests;
* `_assert_fresh` still runs and still catches a bad patch;
* the browser's game knowledge is confined to four selectors, not the whole
  position.

## Alternative if the DOM proves unreadable

Subscribe to BGA's notification stream (`dojo.subscribe`; see
`bga-framework.d.ts:435`, `:1123`) and maintain deltas. This is the most
"correct" option and the worst to own: it means reimplementing 7WD's
notification semantics in JS, and it fails **silently** when it drifts, unlike a
DOM selector that returns nothing. Only if the DOM route fails.

## Verification

* Play a real game. After each move, capture without reloading and confirm
  `wire_from_bga` does **not** raise `StaleGamedata` and the resulting position
  matches the screen.
* Deliberately break one selector and confirm the failure is loud — either an
  empty patch that `_assert_fresh` catches, or an explicit raise. A silently
  wrong position is the failure mode to design against.

---

# C. Host `{"bga": …}` branch

`wire_from_bga` is written and tested but **nothing calls it**:
`grep -rn "bga" games/advisor/*.py` returns nothing.

`SevenWondersAdvisor.state_from_wire` (`advisor_adapter.py:147`) accepts two
payload shapes today:

```python
{"observation": <scrape wire>, "resample_seed": <int>}   # line 148
{"seed": <int>, "first_player": <int>, "prefix": [...]}  # line 158
```

Add a third, ahead of both:

```python
{"bga": <raw gamedatas>, "dom": {...}, "resample_seed": <int>}
```

which applies the DOM patch (item B), calls `wire_from_bga`, and falls into the
existing `"observation"` path. Roughly ten lines plus tests. Keep the mapping in
Python — that is the design principle above.

Note `RecommendBody` (`games/advisor/app.py:44`) already passes `state: dict`
through untouched, so no host change is needed, only the adapter.

---

# D. Browser extension

There is **no 7WD extension**. `extension/` is the **Can't Stop** advisor
(`manifest.json` → `"name": "BGA Can't Stop Advisor"`). Do not edit it.

`extension_kingdomino/` is the working reference:

```
manifest.json  manifest_chrome.json  background.js  content.js
popup.html     popup.js              placement_mapping.js  README.md
```

It matches `*://*.boardgamearena.com/*` and POSTs to
`http://127.0.0.1:8000/api/recommend` (`content.js:4`).

For 7WD, create `extension_7wd/` with:

* **manifest** matching `*://*.boardgamearena.com/*`;
* **content script** = `bga_snippet.js`'s capture + the four DOM selectors from
  item B, POSTing `{"bga": …, "dom": …}` to `/api/recommend`;
* **a panel** showing the ranked moves. Kingdomino's `content.js` is ~2,300
  lines but most of that is drawing tile-placement overlays; 7WD needs a text
  list, so this is much smaller.

Use the resumable endpoints (`/api/recommend/start`, `/poll`, `/stop`) rather
than the blocking one so the panel can stream results and the user can stop a
search — the host already supports this.

---

# E. Draft-preference extraction (analysis, not play)

Possible today with no training and no engine work; needs A only so that a draft
position can be handed to the net.

The action space has draft picks as first-class actions (`codec.py:8`):

```
WONDER_DRAFT      0–11     wonder_id
```

occupying indices 0–11 ahead of `BUILD_BASE = 12`. Measured on the last
iteration's buffer, drafts are searched and labelled at the normal rate:

```
draft decisions per game:  8.0 total
                           6.1 policy-excluded (playout-cap randomisation)
                           1.9 trainable
→ ~156k trainable draft targets across the 84k-game run
```

So: build a `WONDER_DRAFT` position, call the net, read the policy head over
indices 0–11. That gives a distribution over wonders directly. The more
meaningful query is conditional — *given these four on offer, which does it
take* — since 7WD drafting is as much about denial as selection.

Worth comparing against `WONDER_DRAFT_TIERS` in `phase_d.py`, the hardcoded
ZeusAI-seeded prior: it was blended in early and **annealed to zero**
(`draft_prior: 0.0` from iteration ~45), so today's preferences are the net's
own. Where the learned ranking disagrees with the published tiers is the
interesting part.

---

# Running it

```bash
pip install fastapi uvicorn
SWD_ADVISOR_CHECKPOINT=games/seven_wonders_duel/runs/laptop_training_03_w7/checkpoints/current_best.pt \
SWD_ADVISOR_DEVICE=cpu \
  uvicorn games.seven_wonders_duel.web_app:app --reload --port 8000
```

Then `http://127.0.0.1:8000/` for the lab UI.

**Shortcut worth knowing:** after item C, you can paste
`captureBgaGamedatas()` output from the browser console into the lab UI and get
advice on real positions with no extension at all. Ugly and manual, but enough
to answer "is this net any good against a human" before committing to D.

# What this test can and cannot tell you

The net has only ever played itself and scripted bots. It promoted at iteration
195 and beats its own 20k-games-ago self 66.5%, but has **no measurement against
any external reference**. A handful of games against humans is the first such
data: treat it as directional, not a rating. It is also an S model (128×4)
trained on a laptop — the cloud run exists to produce something better, and the
advisor should be able to swap checkpoints without changes.
