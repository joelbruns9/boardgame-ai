# 7WD BGA advisor — implementation plan

**Goal: play real games on Board Game Arena with live advice from a trained
net, including the wonder draft, without reloading the page between moves.**

This document is written to be picked up cold. It states what already exists,
what is missing, why each gap exists, and how to close it. Line numbers are as
of `main` at commit `0b09708`; find them by the quoted search strings if they
have drifted. Line numbers into `BGA Files/sevenwondersduel/` are as of the
dump committed alongside this revision — that is a third-party snapshot, so if
BGA ships an update, re-check rather than assume.

## Status summary

| # | Item | Size | Blocks live play? |
|---|---|---|---|
| ~~B~~ | ~~Fresh board state without a page reload~~ | **DONE 2026-08-02** | verified live against table 892846644 |
| ~~C~~ | ~~Host `{"bga": …}` branch~~ | **DONE 2026-08-02** | adapter accepts `{"bga","args","dom","log"}` |
| ~~D~~ | ~~Browser extension~~ | **BUILT 2026-08-02** | `extension_7wd/`, ran against a live game |
| ~~A~~ | ~~Wonder-draft support in the scrape codec + BGA mapper~~ | **DONE 2026-08-02** | draft advice live; see below |
| ~~stats~~ | ~~Panel victory-type outlook~~ | **DONE 2026-08-02** | `state_to_public.victory_outlook` |
| E | Draft-preference extraction | ~half day | no — analysis, not play |

**What is actually left**

| # | Item | Size | Notes |
|---|---|---|---|
| F | Start-player choice for the next age | ~1 day | a decision you are asked and get nothing for |
| G | Wonder art in the panel | ~15 min | draft rows show a placeholder |
| H | Follow-up move in the panel (Mausoleum et al.) | ~half day | advice is correct but incomplete |
| E | Draft-preference extraction | ~half day | analysis, not play |

Each is written up under "Remaining work" below. Also outstanding: **re-test the
extension**, which last ran before the Rust searcher, the batched evaluation
boundary, the victory-type outlook and the draft landed.

**Item A shipped 2026-08-02.** The draft was long assumed unreconstructable from
a public observation; it is not. BGA reveals only the current group, so the
hidden part is a uniform 4-of-8 partition into (group 2 | never-dealt box) — 70
equally likely splits, the same kind of object the age-deck determinizer already
samples. It needs its own branch only because no age is dealt yet, so there is no
tableau to reconstruct.

Two things that were not obvious:

* **`first_player` must be derived, and it is load-bearing.** `pick_wonder`
  asserts `active_player == _draft_order(round)[pick_index]`, and the skeleton
  state starts at `first_player = 0`, so a wrong derivation fires that assertion
  rather than quietly misplaying.
* **Pick order is not the concatenation of the two players' lists.** That
  interleaves rounds the moment round 1 starts, and `taken[:4]` stops being
  group 0. It *is* recoverable: the draft sequence is fixed once `first_player`
  is known, so replay it and pop from each player's list.

On the real captured draft (`testdata/bga_892846644_draft.json`), aggregated over
6 determinizations, the net wants **The Sphinx** (87.8% of visits, Q +0.15) and
ranks **The Great Library last** (0.4%) — the wonder actually taken first in that
game. That disagreement is item E's whole subject.

Play advice works end to end today: capture without reload, streaming search,
ranked moves with card art. The searcher moved to Rust and is ~19× faster than
when the extension was first tried — see `ADVISOR_RUST_UNIFICATION.md`, an
independent track that neither blocks nor is blocked by this one.

## Remaining work

### F. Start-player choice for the next age (`selectStartPlayer`)

Two decisions a game (Age I→II, II→III) and the advisor is silent for both:
`Phase.CHOOSE_NEXT_START_PLAYER` is still refused by the scrape codec, with a
test asserting it stays refused so that supporting the draft did not widen the
scope by accident.

**The hard part is a genuine modelling mismatch, not the plumbing.** BGA and the
engine disagree about *when* the next age is dealt:

* **BGA deals first, then asks.** Confirmed on the committed capture
  `testdata/bga_887892216_ageiii.json`, which is taken at `selectStartPlayer`
  and already carries **age 3 with all 20 cards** in `draftpool`. A human chooses
  while looking at the pyramid.
* **The engine deals as a consequence of the choice** (`engine.py:887-899`):
  `game.age += 1`, then an `AGE_DEAL` chance event, then
  `TableauState.from_deck`. At choice time the next age does not exist.

That difference is not cosmetic — the layout is exactly what makes the choice
worth making. Reconstructing the engine's model faithfully would average over
deals that never happen and answer a different question from the one on screen.

So the observed deal has to be seeded, which fights three mechanisms:

1. `age_decks[next_age]` must carry the observed cards **in slot order**, since
   `from_deck` places them positionally.
2. The `AGE_DEAL` chance node must resolve to that deal rather than sampling.
   `age_deal_samples` defaults to 0, so this may already hold — verify, do not
   assume.
3. `resample_hidden` re-deals every age `> state.age` and would throw the seeded
   deal away.

**The test is the fixture, and one assertion pins all three:** after applying
either choice to the reconstructed state, the resulting tableau must equal what
BGA shows in `draftpool`. `_tableau(gamedatas, next_age)` already reads that
structure, and face-down slots there are sampled exactly as in PLAY_AGE.

Note the engine keeps `age` at the age that just *ended* during this phase, with
an exhausted tableau (all 20 slots present=False). Absent slots retain their
`card_name` on both sides -- see the injection work in
`ADVISOR_RUST_UNIFICATION.md` -- but at an age boundary every card has already
gone to a city, the discard or under a wonder, so the pool arithmetic works out
even though the old structure cannot be read back from BGA.

**Do not ship the cheap version.** Following the engine's model is easy and
quietly worse: advice that looks authoritative while ignoring information the
player can see is worse than no advice.

**Decided 2026-08-02: the engine is being corrected instead**, before the cloud
run — see `ENGINE_AGE_DEAL_ORDERING.md`. The engine's ordering is a genuine
rules deviation, not just an advisor inconvenience: it is the only place a
player is given *less* information than the real game, and the net has learned
this decision layout-blind across ~168k of them.

That makes F mostly plumbing rather than a modelling problem — once the engine
deals before asking, it agrees with BGA and the three seeding mechanisms above
largely evaporate. **Sequence F after the engine change.** No retraining is
forced by F itself: the two actions already occupy indices 1200/1201 of the
trained action space and appear 2.0 times per self-play game.

### G. Wonder art in the panel (small)

Draft rows render the dashed placeholder instead of the wonder. `content.js:150`
passes only `r.fields.card_name` to `cardArt`, and a draft action has
`wonder_name` with `card_name = None`.

Everything needed is already in place: the bridge captures `art.wonders`
(`page_bridge.js:95`) and `ActionView.fields` carries `wonder_name`. BGA styles a
wonder as `div.wonder.wonder_small` against `img/wonders_v3.jpg`, exactly
parallel to `div.building.building_small` -- so the same trick works, still with
nothing bundled.

Fall back to the wonder sprite when `card_name` is absent. Progress tokens
(`img/progress_tokens_v3.jpg`, also already captured) would give pending-choice
rows their art too.

### H. Show the follow-up move (Mausoleum and friends)

The panel says `Wonder: The Mausoleum (using X)` and stops, but building the
Mausoleum immediately forces a second decision -- *which discarded card to take
for free* -- and that is most of the move's value. Same for any wonder that
triggers a choice.

The search already knows: it is the principal variation. Nothing downstream can
see it because `RustPuctSearch.snapshot` returns **root edges only**.

Needs a PV readout -- from the root's best edge, walk to the most-visited child
and report its best action -- surfaced as an extra field per recommendation.
Worth doing generically rather than special-casing the Mausoleum: it also covers
Zeus/Circus destroy targets and the science-pair token pick.

## The BGA source dump

`BGA Files/sevenwondersduel/` holds the **actual `sevenwondersduel` server and
client source** — `sevenwondersduel.game.php`, `sevenwondersduel.js`,
`states.inc.php`, `modules/php/**`, and the page template
`sevenwondersduel_sevenwondersduel.tpl`. Every empirical claim below that was
once "verified by experiment" is now verified against this code, and file:line
references throughout point into it.

(The neighbouring `BGA Files/sevenwonders/` is a **different game** — 7 Wonders,
BGG 68448, 3–7 players. Only its two framework-level files,
`bga-framework.d.ts` and `_ide_helper.php`, are game-agnostic and useful.)

What the dump settles, so nobody re-derives it:

| Claim | Evidence |
|---|---|
| BGA does not reveal wonder group 2 during round 1 | `Wonders::getSituation()` returns only `selection{round}` — `modules/php/Wonders.php:24-30` |
| 12 wonders → 4+4 offered, 4 unused (so C(8,4)=70) | `GameSetupTrait.php:33-41`: shuffle 12 → 8 to `selection` → 4 each to `selection1`/`selection2`, rest to `box` |
| Draft order is the engine's `(f,1-f,1-f,f)` / `(1-f,f,f,1-f)` | `WonderSelectedTrait.php:10` — *"Wonders are selected A-B-B-A, then B-A-A-B"*, matching `game.py:351` |
| Exactly four fields go stale mid-game (five during the draft) | `updatePlayerBuildings` / `updateDiscardedBuildings` / `updateProgressTokensSituation` are called **only** from setup (`sevenwondersduel.js:223-225,265`); `updateMilitaryTrack` only from a Pantheon-only notif (`:5031`). `draftpool` (`:1021`), `wondersSituation` (`:844`) and `playersSituation` (`:2539`) *are* written back |
| 3 cards per age deck are removed unseen | `GameSetupTrait.php:53` — already modelled by the engine (`game.py:299-308`) |
| There is no way to re-fetch `gamedatas` | Nothing in the dump provides one. Settled. |

It also explains **why `_assert_fresh` works**, which is worth knowing before
anyone "simplifies" it: `argPlayerTurn` (`PlayerTurnTrait.php:31-49`) returns
`playersSituation` on every single turn and `onEnteringState`
(`sevenwondersduel.js:724`) writes it into `gamedatas`, while `playerBuildings`
is never rewritten after setup. The check therefore compares a
guaranteed-fresh number against a guaranteed-stale list. That is structural, not
luck — but note it keys on science counts, so it is **blind during the draft**,
when both counts are zero.

## What the live captures settled (2026-08-02, table 892846644)

Three captures were taken from a real game: a **wonder-draft** position, and an
**Age III** position in two forms at the same moment — one taken with the page
open across the whole game (`bga_892846644_age3_patched.json`, stale `gamedatas`
+ DOM patch) and one right after an F5 (`..._reference.json`, fresh `gamedatas`).
All three are in `testdata/`.

**The staleness is total, not partial.** At Age III, with 21 cards built and 15
discarded, the un-reloaded `gamedatas` still held the draft-time board:

| field | `gamedatas` | DOM |
|---|---|---|
| playerBuildings (each) | `[]` | 11 / 10 cards |
| discardedBuildings | `[]` | 15 cards |
| board progress tokens | 5 | 4 |

**The patch is exact.** `wire_from_bga_payload(patched) == wire_from_bga(reference)`
— byte-identical wires. That is the reload-equivalence test this document asked
for, and it is now a committed test.

**Findings that changed the code:**

1. **`'Chamber Of Commerce'`.** BGA title-cases card names where the engine
   follows the printed card. This raised a bare `KeyError` and crashed the
   mapper on an ordinary Age III position. Fixed by `_card_name`, a case-folding
   canonicalizer applied wherever a BGA name **enters the wire** (not merely at
   lookup sites — the names flow on to the engine). A full audit against BGA's
   material tables found this is the *only* base-game divergence: 12/12 wonders
   and 10/10 progress tokens match exactly, and the other 18 mismatches are
   expansion-only content that `_require_base_game` rejects first. Engine card
   names are unique under case folding (73/73), so the fold is unambiguous.
2. **Building order is not canonical in BGA.** `getAllDatas` sorts a city by
   `card_location_arg` (build order); the DOM groups by colour column. A 7WD city
   is a *set*, so `_city` now sorts by card id and both sources agree.
3. **`draftpool` is `[]` during the draft** — confirmed live. Guarded so it
   raises `UnsupportedBgaState` rather than `TypeError` on list indices.
4. **BGA sends numbers as strings** (`draftpool.age == '3'`,
   `conflictPawn == '0'`, `wonderSelectionRound == '1'`). The mapper's `int()`
   casts are load-bearing; any new comparison needs one.
5. **Two frames expose `gameui`.** The table page nests a second game frame
   (`...&testuser=<opponent id>`) whose `me_id` is the *opponent*.
   `findGameWindow` walks a frame stack and can return either. `wire_from_bga`
   survives (it keys on `startPlayerId`, identical in both), but per-seat private
   data does not — notably Great Library's `_private.progressTokensFromBox`,
   which would be absent and turn a workable position into a refusal. **Open: make
   `findGameWindow` pick the frame whose `me_id` is the logged-in user.**

**Confirmed, no change needed:** the guild card backs. The Age III capture has
10 revealed age-III, 2 revealed guild, 7 face-down age-III and 1 face-down guild
— 2 + 1 = the 3 guilds BGA deals (`GameSetupTrait.php:66-72`), matching the
single deep-purple back visible on screen. Determinization round-trips
public-exact, holds `selected_guilds == 3` / `unused_guilds == 4`, and samples
the hidden guild uniformly over the 5 unseen ones. The engine also matches BGA's
*ordering* — 3 Age III cards go to the box **before** guilds are added, so
`removed_age_cards[3]` can never be a guild.

### Second sitting, same table: Great Library + off-centre military

A later capture (`testdata/bga_892846644_greatlibrary.json`, state
`chooseProgressTokenFromBox`) closed the two remaining gaps and opened a new one.

**Great Library — validated live.** `gamestate.args._private.progressTokensFromBox`
arrives **flattened** (no player-id level) and **id-keyed**
(`{"2": …, "5": …, "7": …}`), exactly as `_pending_choice` assumed. `.values()`
is correct; the `_ide_helper.php:2570-2574` contract predicted this and the live
payload confirms it. The wire came out as
`choose_unused_progress / [Architecture, Masonry, Philosophy] / consume_all=True`.

**Military — validated, sign and all.** DOM read `conflictPawn = 5` and slot 3
empty (token captured), against a stale `gamedatas` still claiming pawn 0 with
all four tokens. The wire produced `conflict_position = 5` and
`military_tokens_remaining = [[-7,5], [-4,2], [7,5]]` — slot 3, engine position
`+4`, correctly gone. That is self-consistent in a way a flipped sign could not
be: the pawn at `+5` has passed `+4` and taken exactly that token. With
`startPlayerId` == engine player 0, `_CONFLICT_SIGN = 1` is confirmed from a
second table.

**The two-frame hazard is now demonstrated, not theoretical.** The opponent-seat
frame (`...&testuser=<id>`) reported `hasPrivate: False` at the same moment the
real seat had the Great Library tokens. Capturing from the wrong frame turns a
workable position into an `UnsupportedBgaState` refusal. Fix `findGameWindow`
before item A.

### FIXED 2026-08-02: base-game wonder burials (two tiers)

Shipped as described below. **Tier 1** reads each burial's *age* from
`wondersSituation[...]["constructed"]` (structural, no parsing) and passes
`unknown_burial_ages` to `determinize_observation`, which counts those cards as
out of play alongside the 3 box-removed — age-distinct, so an age-2 burial can
never touch the age-3 pool. **Tier 2** identifies the cards from the game log
and puts exact `(wonder, card)` pairs in `observation.wonder_burials`, which
removes them from the unseen pool outright.

Measured on the Great Library capture: tier 2 recovered all six burials
(Great Library ← Chamber of Commerce, Piraeus ← Study, Sphinx ← Statue,
Circus Maximus ← Forum, Statue of Zeus ← Sawmill, Temple of Artemis ← Brickyard)
and no buried card can be dealt into a face-down slot. With the log withheld,
tier 1 yields `unknown_burial_ages = [2,2,2,2,3,3]` and determinizes cleanly
instead of raising — buried cards may still be *hypothesised* into face-down
slots, which is the correct posterior under ignorance, not an error.

A log line is trusted only when its card's age equals the wonder's structural
`constructed` age; otherwise that burial degrades to tier 1. So a trimmed,
mistranslated or mis-parsed log costs sharpness, never correctness.

**Self-play was never affected — measured, not assumed.** An earlier draft of
this section flagged `pool.visible_card_names` as a possible training bug. That
was wrong. In an engine-built state a taken slot keeps its `card_name` in the
tableau, so `visible_card_names` already sees a buried card that way: across 40
random games and 2243 PLAY_AGE observations, a buried card appeared in the unseen
pool **zero** times.

The gap existed only on the *scrape* path, because `_tableau` sets
`card_name: None` for an emptied slot — from a snapshot we cannot see what left
it. That is exactly what the log recovers.

**Second code path, also fixed.** The determinizer and the searcher read the pool
independently: `search.py:149` rebuilds `unseen_pool(...)` and calls
`enumerate_card_reveal` for chance nodes. Fixing only the determinizer left
buried cards enumerable as CARD_REVEAL outcomes (measured: 10 AGE_III candidates
where 8 are possible, with Chamber of Commerce and Study among them). So
`visible_card_names` now unions in `observation.wonder_burials` — a **no-op for
engine states** (re-verified: same 0 over the same 2243 observations) that closes
the scrape path. Candidates drop to 8 and still cover every truly face-down card.

**Not carried into Rust.** `seven_wonders_rust/src/pool.rs` handles
`buried_cards` only. That is harmless for self-play for the same reason the
Python change is a no-op there, and the current advisor searches with the Python
`GumbelMCTS` (`advisor_adapter.py:260`). It would matter if the advisor is ever
pointed at the Rust searcher on a *scraped* position — port it then.

### FIRST END-TO-END RUN 2026-08-02: the net advises on a real BGA position

The whole stack, real checkpoint (`laptop_training_03_w7/current_best.pt`), real
capture, CPU::

    sims 600/600 in 4.4s
    root_value +0.7829  -> 89.1% for the player to move

    action                          visits     Q(you)    prior
    Resolve choice: Architecture       352    +0.8016    0.428
    Resolve choice: Masonry            194    +0.7505    0.478
    Resolve choice: Philosophy          54    +0.7739    0.095

The search is doing real work rather than echoing the net: the policy prior
preferred Masonry (.478 vs .428) and the tree re-ranked to Architecture nearly
2:1 on visits. Whether that is *good* 7WD is a human judgement; what this
establishes is that the pipeline is coherent from `gamedatas` to ranked advice.

**It also immediately caught a blocking bug that every wire-level test missed** —
see the next item. Worth remembering: the wire tests all passed on this position
while search could not run on it for a single ply.

### Continuous search: already built, measured 2026-08-02

**No host code was needed.** `JobManager._advance_to_target` (`games/advisor/jobs.py`)
already loops in chunks and publishes a snapshot after each one; `_ClosedHandle`
holds its root across `advance` calls, so the tree is **cumulative** rather than
restarted. `/api/recommend/start` + `/poll` + `/stop` exist, `max_sims` validates
to 1,000,000 and `chunk_sims` to 100,000. Streaming is a matter of parameters
(`max_sims` high, `chunk_sims` 20-50) plus a panel that polls.

Measured on the real Great Library capture, `chunk_sims=50`::

    t= 2.0s sims=   350 root=+0.8505 win= 92.5%  Architecture .886 Masonry .757
    t= 6.0s sims=  3350 root=+0.9236 win= 96.2%  Architecture .937 Masonry .840
    t=12.0s sims= 15250 root=+0.9647 win= 98.2%  Architecture .971 Masonry .795
    t=26.0s sims= 50100 root=+0.9787 win= 98.9%  Architecture .983 Masonry .770

Three things this settles:

* **Throughput is ~1,900-2,500 sims/s on CPU**, not the ~136/s quoted earlier in
  this document — that figure came from a 600-sim run dominated by checkpoint
  load. A million sims is minutes, not hours.
* **600 sims is badly under-converged.** Root value climbs 0.85 -> 0.98 and the
  best action's Q climbs 0.886 -> 0.983. The *ranking* was stable at every depth
  (Architecture led throughout), but any win-probability number read off a short
  search is not trustworthy. Do not quote sub-5k-sim confidences.
* **Annotator cost is a non-issue.** ~1,000 publishes in 26s while sustaining
  50k sims; `ExactEndgameAnnotator` does not bottleneck small chunks. An earlier
  revision of this document flagged this as a risk to measure — measured, it is
  not one, and no "annotate every Nth publish" tweak is needed.

Regression: `test_streaming_search_deepens_a_cumulative_tree` asserts monotonic
streaming progress and a cumulative tree using a stub evaluator (no checkpoint).

What remains is panel-side, in item D: poll `/poll`, re-render, and restart the
job when a fresh capture shows the position changed.

### Mid-move captures: a pending CARD_REVEAL must be resolved on rebuild In the Great Library position, slot (3,4) was uncovered by the move in progress.
BGA defers its flip until the whole move resolves, so the capture shows it
present, face-down and unavailable.

An earlier revision of this section called that "legal and self-consistent". It
is not. The engine reveals an uncovered slot via a CARD_REVEAL chance event fired
with the same action (`TableauState.take_accessible`, `game.py:176`), and
`codec._card_at` treats accessible-but-face-down as a hard error. Reconstructing
the state without firing that reveal produced a position where the *root* was
fine — the three Great Library choices encoded correctly — but search died one
ply later with `ValueError: slot (3, 4) holds no revealed card`. Every wire-level
test passed on this capture; only running an actual search found it.

`determinize_observation` now resolves any pending reveal: a slot that is present,
face-down and accessible is flipped, its identity sampled from the unseen pool
like any other face-down slot. That is exactly what the engine would have done.
Tests assert round-trip *modulo* that resolved slot
(`_assert_roundtrip_modulo_midmove_flip`) and a stub-evaluator search now runs on
the real capture as a regression. Older pending-choice tests never hit this
because they graft a pending state onto a clean `playerTurn` fixture, which has
no half-resolved move.

### Original diagnosis: base-game wonder burials are never mapped

Constructing a wonder buries an age card under it. The engine models this
(`GameState.wonder_burials` / `buried_cards`, `game.py:232-233,268,279`), but
`wire_from_bga` hardcodes `"wonder_burials": []` with an "Agora-only" comment.
That comment is wrong: burial happens in the **base game**, every time.

Consequence: buried cards stay in the unseen pool, so the determinizer can deal
an already-buried card into a face-down slot.

* **Silent** when the buried cards are Age I/II — nothing checks those pools.
  The committed `bga_887892216_ageiii.json` fixture has two such burials (ages 1
  and 2) and has always passed.
* **Loud** once an Age III card is buried:
  `ValueError: age3 AGE_III 10 != facedown 5 + 3` out of
  `advisor_scrape.determinize_observation`. This is what the Great Library
  capture hit — 6 wonders built, 2 of them consuming Age III cards.

**Scope of the error.** Everything *visible* stays exact — cities, face-up
tableau, discard, military, tokens. What is wrong is the belief over *hidden*
cards. In the captured position the model spreads each face-down Age III slot
over 10 candidates when only 8 are possible: ~20% of the mass sits on cards that
cannot appear, and the bias is not neutral, since a buried card was one somebody
chose to take off the structure.

**`gamedatas` does not carry the identity** — `getWondersData`
(`Player.php:343-359`) gives `constructed` = the buried card's **age** (2, 3, 0)
and `ageCardSpriteXY` = that age's card *back*, nothing more.

**But the game log does, and it is in the snapshot.** The `constructWonder`
notification (`Wonder.php:58-73`) carries `buildingId` / `buildingName`, and its
log line renders as::

    <player> constructed Wonder "<wonder>" for N coin(s) using building "<card>"

Read live off the table: Great Library ← "Chamber Of Commerce", Piraeus ←
"Study", The Sphinx ← "Statue". The first two are exactly the two cards the
Age III arithmetic said were unaccounted — so the log closes the gap **exactly**,
with no sampling and no stateful notification tracking.

**Recommended fix: parse burials out of the log, and validate them structurally.**
Prose parsing is fragile on its own (the string is `clienttranslate`d, so it is
localized), so gate it on cross-checks that come from structured data:

* the number of parsed burials must equal the number of constructed wonders in
  `wondersSituation`;
* each parsed burial's card age must equal that wonder's `constructed` value;
* each parsed card must be in the unaccounted set (not visible anywhere else).

If any check fails, raise — an un-parsed log is then a loud failure, not a silent
wrong belief, which keeps the module's guarantee intact. Note the log repeats
each line (two rendered copies), and its names carry BGA's title-casing
("Chamber Of Commerce"), already handled by `_card_name`.

**Open question before implementing:** whether BGA retains the *full* log in the
DOM for a long game, or trims it. If it trims, early burials would go missing —
which the count cross-check above would catch, turning it into a refusal rather
than a wrong answer. Worth measuring on a completed game.

Until this is fixed, any position with an Age III card buried under a wonder
raises rather than advising, and positions with only Age I/II burials advise from
a subtly over-broad hidden-card pool.

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

Capturing `gamedatas.gamestate.args` (item B) does **not** violate this: like
`gamedatas` itself it is grabbed verbatim, with no interpretation in the page.
Only the eight DOM selectors are game knowledge.

---

# A. Wonder-draft support

**Depends on B.** See the next subsection.

## The fifth stale field — read this before starting

`gamedatas.wondersSituation` is the *only* field a draft position can be read
from, and it is stale for the whole draft:

* `updateWondersSituation` (`sevenwondersduel.js:844`) is the only thing that
  writes `gamedatas.wondersSituation`, and it is called from
  `notif_constructWonder` (`:3434`) and `:7351` — **not** from
  `notif_wonderSelected` (`:2693`), which only animates DOM nodes.
* `argSelectWonder` (`SelectWonderTrait.php:14-28`) does not return
  `wondersSituation` at all. It returns `wonderSelection`, and only when
  `count($cards) == 4` — i.e. at a round boundary.

So mid-draft, both `wondersSituation.selection` and each player's picked list
sit at page-load values, and `_assert_fresh` cannot catch it (it keys on science
counts, which are 0 during the draft). A draft mapper built without item B
would emit a confidently wrong position with no loud failure — the exact thing
this design is built to prevent.

**Therefore: do B first, and add `wondersSituation` to its DOM patch** (selectors
are listed under B).

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
`unused_wonders`. Play alternates over group 1, then group 2. BGA does the same
deal: `GameSetupTrait.php:33-41` shuffles all 12, moves 8 to `selection`, splits
those 4/4 into `selection1`/`selection2`, and sends the remaining 4 to `box`.

**BGA does not reveal group 2 during the first round** — confirmed 2026-08-01 by
experiment and since confirmed in source: `Wonders::getSituation()`
(`modules/php/Wonders.php:24-30`) returns only `selection{round}`, never both.
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
  returns `(f, 1-f, 1-f, f)` for round 0 and `(1-f, f, f, 1-f)` for round 1 —
  which is exactly BGA's order, per the comment in
  `WonderSelectedTrait.php:10`: *"Wonders are selected A-B-B-A, then B-A-A-B."*
  Given `active_player` from the observation:

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
  instead of raising. The state name is **`"selectWonder"`**
  (`sevenwondersduel.game.php:117`, `SevenWondersDuel::STATE_SELECT_WONDER_NAME`).
* Emit the draft wire: the current offer, whose turn it is, and each player's
  already-picked wonders. Wonders are keyed by **name strings** BGA already
  exposes (`wonders[id].name`), consistent with the rest of the mapper — no
  numeric-id alignment.
* **BGA's selection round is 1-indexed, the engine's is 0-indexed.**
  `VALUE_CURRENT_WONDER_SELECTION_ROUND` is 1 then 2
  (`SelectWonderTrait.php:15`, `:69`). The `picked // 4` derivation above is
  unaffected and remains the source of truth, but a fresh cross-check is
  available for free: `gamestate.args.wonderSelectionRound` (see B's note on
  `gamestate.args`). Assert `wonderSelectionRound - 1 == wonder_round`.
* **`gamedatas.draftpool` is `[]` during the draft**, not a dict.
  `getAllDatas` (`sevenwondersduel.game.php:685-688`) only populates it once
  `wondersSituation.selection` is empty. Any code on the draft path that assumes
  `draftpool` is a mapping will blow up; guard it.
* **Cross-check the seat frame on the first real capture.** `_seat_order` maps
  engine player 0 to `startPlayerId`, which BGA defines as the first player in
  table order (`sevenwondersduel.game.php:464-467`) and which is fixed for the
  whole game — it is what drives military-track direction, so it is well-defined
  during the draft. Whether BGA's *initially active* player at `selectWonder` is
  that same player is not pinned down by the dump. Keep the
  derive-from-`active_player` logic above (it is robust either way) and assert
  that it agrees with `startPlayerId`. That single assertion is the cheapest
  possible test of the whole seat-framing assumption.
* Leave `CHOOSE_NEXT_START_PLAYER` and expansions rejected.

**3. Determinization variance is real here.** A PLAY_AGE query has one plausible
determinization of the current age; a round-1 draft query has 70 partitions ×
however many age deals. **Run several determinizations per draft query and
aggregate**, the way the existing scrape path does. Expect draft advice to be
noisier than play advice and say so in the UI.

**4. Only 6 of the 8 picks are real decisions.** `enterStateSelectWonder`
(`SelectWonderTrait.php:31-41`) auto-picks when one wonder remains in a round, so
`wonder_pick_index == 3` never reaches a player in either round. The engine work
above is unchanged (those positions still exist in the state machine), but the
extension will only ever be asked 6 draft questions per game. See also item E.

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
  Capture it **on a fresh page load** even after B is done, and separately
  capture the same position mid-draft without reloading, to prove the B patch
  reproduces the reloaded offer exactly. Nothing else catches a stale draft:
  `_assert_fresh` is blind here (science counts are 0 for both players).
* Sampling uniformity: over many seeds, group-2 membership should be roughly
  uniform over the 70 partitions.

---

# B. Fresh state without a reload — DONE 2026-08-02

Shipped: `captureDomPatch` / `captureForAdvisor` in `bga_snippet.js`,
`apply_dom_patch` in `bga_extract.py`, and the reload-equivalence test in
`test_bga_extract.py`. See "What the live captures settled" above for the
verification and the four code changes it forced. The analysis below is kept
because the selector table and the two traps are still the reference for anyone
touching this.

## The problem

`gameui.gamedatas` is the **page-load** payload. BGA patches some fields from
its notification stream but leaves others at their load-time values
(`bga_snippet.js:9-16`). The full list to patch is five:

```
playerBuildings   discardedBuildings   militaryTrack   progressTokensSituation
wondersSituation
```

The first four are stale all game; `wondersSituation` is stale during the draft
only (it *is* refreshed once wonders start being constructed) — see A's "the
fifth stale field". Source: the corresponding `update*` functions are called
only from setup (`sevenwondersduel.js:223-225,265`), except
`updateMilitaryTrack`, which is called only from a Pantheon-only notif
(`:5031`), and `updateWondersSituation`, which is called only from
`notif_constructWonder` (`:3434`) and `:7351`.

`bga_extract._assert_fresh` (line 126) catches the common case by an
internal-consistency check — each player's count of science-bearing buildings in
`playerBuildings` must equal BGA's own `scienceSymbolCount` — and raises
`StaleGamedata`. So a stale read fails loudly rather than advising on a wrong
position. Good, but it means every query currently needs an F5, and it is
**blind during the draft**, when both science counts are 0.

## What is NOT available

**Re-fetching `gamedatas` without a full page load is not possible.** The BGA
framework's own type definitions (`BGA Files/sevenwonders/bga-framework.d.ts`,
line 883) document `gamedatas` as *"the initial set of data to init the game,
created at game start or by game refresh (F5)"*. There is no `getGameData`,
no refresh call, no notification-history fetch; the only `ajaxcall` is
deprecated and is for sending actions. The `sevenwondersduel` dump has now been
read end to end and contains nothing of the sort either. Settled — do not spend
time looking for one.

## What IS available for free: `gamestate.args`

BGA parks the current state's server-side args in
`gameui.gamedatas.gamestate.args`, and it refreshes them on every state entry
(`onEnteringState`, `sevenwondersduel.js:713-755`). For 7WD that means, **every
turn, fresh**:

* `argPlayerTurn` (`PlayerTurnTrait.php:31-49`) → `draftpool`,
  `wondersSituation`, `playersSituation`
* `argSelectWonder` (`SelectWonderTrait.php:14-28`) → `wonderSelectionRound`,
  and `wonderSelection` at round boundaries
* plus `gamestate.name` and `gamestate.active_player`

This does **not** remove the need for the DOM patch — none of the five
genuinely-stale fields appear there in a state you can rely on (`argPlayerTurn`
carries `wondersSituation`, but only outside the draft, which is precisely when
it is not needed). But it is a fresh, cross-checkable source
that costs nothing and, crucially, **carries no game knowledge into the
browser**: you capture `gamedatas.gamestate.args` verbatim, exactly as
`captureBgaGamedatas` already captures `gamedatas`. Do that unconditionally and
use it to sanity-check the DOM patch (e.g. `draftpool` from `args` vs the
tableau you already read, `wonderSelectionRound` vs the derived
`wonder_round`).

## The approach: read the DOM for the five stale fields

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

All five stale 7WD fields are plainly visible on screen: built buildings per
player, the discard pile, the military track position, which progress tokens
remain available, and the wonder offer.

## The selectors

Read out of the dump; all are stable data attributes written by the game's own
notification handlers, not positional guesses.

| field | selector |
|---|---|
| `playerBuildings` | `.player_buildings.player{PID} [data-building-id]` |
| `discardedBuildings` | `#discarded_cards_container [data-building-id]` |
| `progressTokensSituation.board` | `#board_progress_tokens > div:nth-of-type(n)` → `[data-progress-token-id]` |
| `progressTokensSituation[pid]` | `.player_info.{me\|opponent} .player_area_progress_tokens [data-progress-token-id]` |
| `militaryTrack.conflictPawn` | `getComputedStyle(document.documentElement).getPropertyValue('--conflict-pawn-position')` |
| `militaryTrack.tokens[N]` | `.military_token_container[data-military-token-number="N"] .military_token` → value from its `military_token_{value}` class |
| `wondersSituation.selection` | `#wonder_selection_container > div:nth-of-type(n)` → `[data-wonder-id]` |
| `wondersSituation[pid]` | `.player_wonders.player{PID} [data-wonder-id]` |

Provenance, so these can be re-derived if BGA changes:
`jstpl_player_building` and `jstpl_wonder`
(`sevenwondersduel_sevenwondersduel.tpl:626-648`) define the `data-building-id` /
`data-wonder-id` attributes; `notif_constructBuilding`
(`sevenwondersduel.js:3138-3140`) writes player buildings **for both players**;
`createDiscardedBuildingNode` (`:1213-1223`) adds to the discard pile and
`notif_constructBuilding` (`:3437`) removes from it on Mausoleum;
`updateProgressTokensSituation` / `updatePlayerProgressTokens` (`:1273-1323`)
own the token DOM; `notif_wonderSelected` (`:2707`) moves a picked wonder into
`.player_wonders.player{PID}`.

**Two traps.**

1. `--conflict-pawn-position` is already in **server frame** (-9..9), the same
   convention as `gamedatas.militaryTrack.conflictPawn`. The per-seat mirroring
   is done separately by CSS via `--invert-military-positions`
   (`sevenwondersduel.css:954`, `sevenwondersduel.view.php:383`). Read it raw;
   do **not** flip it.
2. Do **not** read military tokens as `#military_tokens > div:nth-of-type(i)`.
   That index is mirrored by `invertMilitaryTrack()`
   (`sevenwondersduel.js:1737,1746`) depending on whether *you* are the start
   player, so it silently transposes the track for one of the two seats. Use the
   `data-military-token-number` attribute (`:5094`), which carries the server
   slot number, and feed it straight into the existing
   `_MILITARY_SLOT_TO_POS` map in `bga_extract.py`.

That is roughly fifteen lines of JS, which is why this item is now ~half a day
rather than 1–2 days.

**Keep the blast radius small.** Do not rewrite the mapper to take DOM input.
Instead, have the browser produce a *patch*: capture `gamedatas` verbatim as
today, plus `gamedatas.gamestate.args` verbatim, plus a small object holding only
the five fields re-read from the DOM, and overwrite them in Python before
`wire_from_bga` runs. Then:

* the mapper keeps its single input shape and all its existing tests;
* `_assert_fresh` still runs and still catches a bad patch;
* the browser's game knowledge is confined to eight selectors, not the whole
  position.

While you are in `bga_snippet.js`: `findGameWindow` (line 35) identifies the
7WD frame by `gamedatas.draftpool` being truthy. During the draft `draftpool` is
`[]` (`sevenwondersduel.game.php:685-688`), which is truthy in JS, so this
happens to survive — but only by luck, and it would break the moment BGA emitted
`null` or omitted the key. **Switch the marker to `gamedatas.wondersSituation`**,
which is present in every 7WD state.

## Alternative if the DOM proves unreadable

Subscribe to BGA's notification stream (`dojo.subscribe`; see
`bga-framework.d.ts:435`, `:1123`) and maintain deltas. The dump makes this
*possible* to do correctly — `setupNotifications` (`sevenwondersduel.js:575-705`)
lists every notification, and the PHP side shows each one's exact args — but it
is still the worst option to own: it means reimplementing 7WD's notification
semantics in JS, and it fails **silently** when it drifts, unlike a DOM selector
that returns nothing. Only if the DOM route fails.

## Verification

* Play a real game. After each move, capture without reloading and confirm
  `wire_from_bga` does **not** raise `StaleGamedata` and the resulting position
  matches the screen.
* **Reload-equivalence test.** At several points in a game, capture patched
  (no reload) *and* capture after F5, and assert the two produce byte-identical
  wires. This is the only check that covers the draft, where `_assert_fresh` is
  blind, and the military track, where the mirroring traps live. Do it from both
  seats — trap 2 only manifests for one of them.
* Deliberately break one selector and confirm the failure is loud — either an
  empty patch that `_assert_fresh` catches, or an explicit raise. A silently
  wrong position is the failure mode to design against.

---

# C. Host `{"bga": …}` branch — DONE 2026-08-02

`SevenWondersAdvisor.state_from_wire` now takes the envelope ahead of both older
shapes and falls through to the `"observation"` path;
`bga_extract.wire_from_bga_payload` applies the DOM patch and calls
`wire_from_bga`. `dom` and `args` are both optional, so a plain freshly-loaded
`{"bga": …}` still works. The original notes follow.



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
{"bga": <raw gamedatas>, "args": <raw gamestate.args>, "dom": {...},
 "resample_seed": <int>}
```

which applies the DOM patch (item B), calls `wire_from_bga`, and falls into the
existing `"observation"` path. Roughly ten lines plus tests. Keep the mapping in
Python — that is the design principle above.

`args` is `gamedatas.gamestate.args`, captured verbatim; it is the free
fresh-data source described under B and is used for cross-checks, not as a
primary input. Make it optional so the existing captures in
`testdata/bga_887892216_*.json` still load.

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
* **content script** = `bga_snippet.js`'s capture + `gamestate.args` + the DOM
  selectors from item B, POSTing `{"bga": …, "args": …, "dom": …}` to
  `/api/recommend`;
* **a panel** showing the ranked moves. Kingdomino's `content.js` is ~2,300
  lines but most of that is drawing tile-placement overlays; 7WD needs a text
  list, so this is much smaller.

Use the resumable endpoints (`/api/recommend/start`, `/poll`, `/stop`) rather
than the blocking one so the panel can stream results and the user can stop a
search — the host already supports this.

Trigger the panel off `gamestate.name` changing to one of `playerTurn`,
`selectWonder`, or the four `_PENDING_STATES` already listed in
`bga_extract.py:52-57` — those are the only states where the advisor has
anything to say. Note the panel will be asked for draft advice **6 times per
game, not 8**: BGA auto-picks the last wonder of each round
(`SelectWonderTrait.php:31-41`).

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

Read those 8 knowing that **2 of them are forced**: the last pick of each round
has a single legal action, because BGA auto-selects it
(`SelectWonderTrait.php:31-41`) and the engine's offer is likewise down to one
card. So the net is spending a fraction of its draft budget labelling positions
with no choice in them, and only ~6 per game carry information. Worth measuring
whether those forced positions are being counted in the 1.9 trainable figure
before drawing conclusions from it.

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
