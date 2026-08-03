# Review request: advisor items F, G, H and frame selection

**Four commits on `bga-advisor-live`, none yet reviewed:**

| commit | what |
|---|---|
| `7c4e5fd` | **F** — advise on the between-Age start-player choice |
| `34719ac` | **G** — wonder and progress-token art in the panel |
| `7431cfc` | **H** — show the forced remainder of a move, plus two test corrections |
| `7bb2c9c` | frame-selection hardening + three stale doc claims corrected |

The engine age-deal reordering underneath all of this (`38b144b`) **was** reviewed
— see `AGE_DEAL_ORDERING_REVIEW_REQUEST.md`; its three findings are fixed or
conceded there. Please do not re-review it except where these four commits lean
on it.

**Please focus on logic errors and on the assumptions listed below.** They are
written as claims to attack, most load-bearing first. Where I have measured
something I say so; where I have not, I say that too. After this review and its
fixes, this branch goes to `main`.

**§0 and §0b are defects found after the four commits landed** — one while
writing this document, one from the user's live testing. Both are fixed here;
read them first, since they are the only places where advice could have been
wrong or the host unusable.

**State:** 826 passed / 1 skipped (7WD + advisor), `cargo test` 17 passed. The
skip is the run-03 bf16 test, refused by the `codec-2` spec bump.

---

## 0. One defect found while writing this, already fixed

Worth reading first, because it is the failure mode this whole module is built
to prevent and I introduced it in item F.

**A capture at `selectStartPlayer` taken before BGA's next-Age notification
lands produced a plausible-but-wrong position, silently.** BGA deals the Age and
*then* asks who starts it, but `gamedatas.draftpool` is only written when
`notif_nextAgeDraftpoolReveal` is processed (`sevenwondersduel.js:4227` →
`:1021`). Capture in that window and `draftpool` still describes the Age that
just **ended** — exhausted.

Nothing downstream caught it. `_assert_fresh` keys on science counts, which are
unaffected. And an exhausted structure determinizes *cleanly*, because it is
exactly the shape the pre-2026-08-03 engine produced at this phase. Measured:
the wire came out as age 2 with **zero** cards present and built a state with no
error. The advisor would then have given confident advice about a pyramid that
is not on screen.

Fixed by requiring a full structure (`len(TABLEAU_LAYOUTS[age])`) at that phase,
raising `UnsupportedBgaState` otherwise. Transient, and the extension retries a
rejected position. Test: `test_a_start_player_capture_before_the_deal_is_refused`.

**Question for the reviewer:** are there other transients of this shape? My
guard is "the structure must be complete at this one phase". The general worry
is that item F made a previously-impossible observation (`CHOOSE_NEXT_START_PLAYER`
with an exhausted tableau) into a *representable* one, and it is precisely the
old engine's shape, so it looks legitimate to every check we have.

---

## 0b. Post-review finding: the search tree had no memory ceiling

Raised by the user's live testing, after this document was written. Fixed;
please review the fix.

**The panel asked for `max_sims: 1_000_000`** — "keep thinking until the board
changes". That bounded CPU and nothing else. On a wide root the closed searcher
allocates a node per simulation, each owning a cloned `GameState`, so the tree
grows without limit for as long as a human sits on a position. Measured on a
12-action Age II turn with the real 128x4 checkpoint:

| sims | arena nodes | RSS over baseline |
|---|---|---|
| 10k | 9,982 | 67 MB |
| 100k | 99,931 | 441 MB |
| 400k | 399,855 | **1,695 MB** |

Node count tracks simulations **1:1** at ~4.4 KB apiece. Narrow roots never come
close — a three-action pending choice reached 2,589 nodes in 41k sims — which is
why a sim cap alone is the wrong instrument and the budget is in bytes.

**Fix.** A 512 MB arena budget (`arena_budget_mb`; 0 disables, so training and
analysis paths are untouched), plus `max_sims` 1,000,000 → 200,000 in the panel
as a second bound. Rust reports its arena exactly through a newly exposed
`arena_deep_bytes()`; the Python reference path estimates from a node counter at
the measured 4.4 KB, documented as an estimate. Measured after: peak RSS
1,087 MB (505 torch + 582 arena), stopping at ~132k sims.

**The stop is visible, not silent.** `SearchSnapshot.stop_reason` (new on the
shared contract) ends the host loop and surfaces as a warning, so a capped
search reads as "stopped growing the tree at 617 MB (budget 512 MB); the numbers
shown are final" rather than a stalled counter.

**Claims to attack:**

25. **Nothing is lost in advice quality.** The budget trips around 130k sims on
    that root; the same position's root value had converged by ~15k and the
    ranking was stable at every depth. An order of magnitude of headroom.
26. **The byte figure is capacity-based, so it overshoots** — it reported
    617 MB against a 512 MB budget, because `arena_deep_bytes` sums `capacity()`
    and Vec capacity doubles. Erring high is the safe direction for a budget,
    but the effective ceiling is ~1.2x the number configured.
27. **The Python path's estimate is a single measured constant** (4.4 KB/node,
    an RSS delta so allocator overhead is included). It will drift if the state
    grows. Rust, the default and the one that runs live, is exact.
28. **`stop_reason` is the right shape for the shared contract** — a
    human-readable string the host transports and shows, rather than a typed
    reason it branches on.

**What this was NOT.** The user's 3 GB reading and the freeze that prompted this
were almost certainly my own concurrent test runs, not the advisor: the pytest
process peaks at **5,039 MB on its own**, and a one-minute think costs ~730 MB
all-in. The arena defect is real and measured, but it was not the thing that
froze the machine, and this document should not be read as saying it was.

## 1. Item F — start-player choice

Four lines of mapping plus tests: `_START_PLAYER_STATE` recognised in `_phase`,
the phase added to `advisor_scrape._SUPPORTED`, `selectStartPlayer` added to
`page_bridge.js`'s `ADVISABLE`, and an actor-framed label. **No branch in the
determinizer** — the observation now carries the new Age's pyramid, so the
ordinary `PLAY_AGE` path reconstructs it.

**Claims to attack:**

1. **The `PLAY_AGE` path is genuinely correct here, not just non-crashing.**
   `obs.age` is the new Age; its face-down slots are sampled from the unseen
   pool the usual way; `resample_hidden` re-deals only ages *after* it, so the
   dealt structure survives. Verified public-exact on the real capture and on a
   synthetic position, and a search runs 200 sims on it. Is there a state
   variable that should differ at this phase and does not?
2. **The mid-move CARD_REVEAL resolver is inert here.** `determinize_observation`
   flips any slot that is present, face-down and accessible. At a fresh deal no
   accessible slot is face-down (the engine asserts accessible ⇒ revealed), so
   the loop should be a no-op. I reasoned this rather than measured it.
3. **`state.age_decks[obs.age]` is rebuilt from the present cards** (all 20).
   Correct for a freshly dealt Age, but it is the same line that serves a
   half-played one — worth a second look that it means the right thing in both.
4. **The chooser is whoever BGA says is active.** I take `gamestate.active_player`
   rather than deriving it from the military track, which is right only if BGA
   and the engine agree on who chooses. **Verified against the source while
   writing this** (`modules/php/States/NextAgeTrait.php:54-65`), and they agree
   case for case:

   | conflict pawn | BGA | engine (`_finish_turn`) |
   |---|---|---|
   | `== 0` | last active player ("no need to change the active player") | `player`, the one who just moved |
   | `< 0` | `gameStartPlayerId` | `active_player = 0` |
   | `> 0` | `opponent(gameStartPlayerId)` | `active_player = 1` |

   Engine seat 0 **is** BGA's `startPlayerId` (`_seat_order`), so the two lower
   rows match exactly. This also corroborates `_CONFLICT_SIGN = 1` a third time,
   from the server source rather than from a live pawn: BGA hands the choice to
   the start player when the pawn is negative, which is the seat the engine
   drives negative.
5. **`_assert_fresh` is meaningful at this phase** (both players have science
   counts). Unlike the draft, where it is blind.

---

## 2. Item G — art

`cardArt` → `rowArt`, an ordered table: `card_name`→buildings,
`wonder_name`→wonders, `choice`→progress tokens, `choice`→buildings.

**The plan's premise was wrong and this is the interesting part.** It said "the
bridge captures `art.wonders`, so the same trick works". It captured the table
and `spriteXY` was `undefined` in every entry: `spriteXY` is a **server** field
on `Building` and nothing else (`modules/php/Building.php:24,73`, the only
occurrence in the PHP tree). BGA *derives* a wonder's and a token's cell from
its id instead. Wonder art could never have rendered.

**Claims to attack:**

6. **The id→cell arithmetic matches BGA's.** `(id-1) % cols`,
   `floor((id-1)/cols)`, 5 columns for wonders (`getWonderDivHtml`,
   `sevenwondersduel.js:838-839`), 4 for progress tokens
   (`getProgressTokenDivHtml`, `:1307-1308`). Pinned by a test, together with the
   assumption that base-game ids are exactly 1..12 and 1..10.
7. **Name matching is safe.** The panel matches recommendation → sprite by name,
   case-insensitively (BGA title-cases some). Measured on the real capture: 12/12
   wonders and 10/10 tokens line up. I also checked there is **no** name shared
   between a progress token and a card (`choice` is tried against tokens first,
   so a collision would show the wrong art) — none exists. A miss is silent: the
   row falls back to the placeholder with no error anywhere.
8. **CSS custom properties resolve at the element, so setting `--scale` on the
   art box works.** `--building-small-width` is defined at `:root` in terms of
   `--building-small-scale`, itself `0.5 * var(--scale)`; substitution is lazy,
   so both resolve against the element's own `--scale`. This is standard CSS
   behaviour and the existing card art already relied on it, but the new
   per-type scales and the resized placeholder lean on it harder.
9. **The three scales make the boxes one age-card tall.** 0.30 / 0.346 / 0.758,
   derived by hand from the `:root` block (48.1px target). Arithmetic worth
   re-checking; the derivation is written into `panel.css`.
10. **Card-before-wonder is the right priority** for a wonder-construction row.
    The label already names both, and the consumed card is the part a human
    misreads. This was the plan's stated intent, but it is a judgement call.

**Not verified: that any of it looks right.** Nothing here has been rendered.

---

## 3. Item H — the forced remainder of a move

One-ply PV per root edge — most-sampled chance child, then that node's
most-visited edge — surfaced as `Recommendation.follow_up` ("then Glassworks")
and rendered under the label.

**Claims to attack:**

11. **Reporting only while a pending choice is open is the right scope.** It is
    narrower than "the child node has the same actor": a Theology extra turn
    also keeps the actor, but it is a *fresh decision*, not a forced remainder,
    and "then ..." would read as part of the move. Opponent replies excluded for
    the same reason. Is the pending-choice test too narrow — is there a forced
    continuation that is not modelled as a pending choice?
12. **A pending choice always belongs to the player who acted.** The label says
    "then X" as part of *your* move, so a pending choice raised for the opponent
    would be mislabelled. **Measured:** across 120 random games, 221 pending
    choices, **0** belonged to anyone but the acting player. That is empirical,
    not structural — the engine does not enforce it.
13. **A follow-up index is readable with no state.** The four pending-choice
    blocks are identity-indexed, so `codec.pending_choice_name` names the card or
    token from the index alone. This is what lets the adapter label a child node
    it never reconstructs. Note `DESTROY_BASE` serves brown and grey from one
    block, so the name is right but the *colour* is not recoverable — irrelevant
    for the label, worth knowing.
14. **Both engines implement the same walk.** They are separate structures;
    both are asserted against the same written-down string on a constructed
    Mausoleum position. Rust reads `node.state.pending_choice`; Python reads
    `child.node.state.pending_choice` via the edge's children dict.
15. **The host contract addition is game-agnostic.** `ActionStats.follow_up` /
    `Recommendation.follow_up` carry a **pre-rendered string**, because naming a
    move is game knowledge. The alternative — carrying an action id and having
    the host label it — cannot work: the follow-up belongs to a different
    position than the root's `ActionView`s describe. Is a rendered string the
    right thing to put on a frozen shared seam?
16. **`follow_ups()` is a separate Rust call, not a wider `snapshot` tuple**, to
    leave that tuple's shape and tests alone. It walks the root's edges on every
    `advance`, so it is O(root edges) per published chunk — not benchmarked, but
    it is the same order as the readout beside it.

**A test correction worth reviewing as a lesson, not just a fix.** The first
version of the H test drove a randomly initialised net and asserted only that
the two searchers *agreed*. That tests noise: under a random net every revival
is worth about the same, so the argmax is a coin flip between near-ties which
the two descents break differently. Measured over six torch seeds: **three
disagreements, and a fourth where neither engine expanded the child at all**, so
`follow_up` was `None` on both — which the other assertion would also have
failed. It passed alone and failed in the suite, because the weights depend on
global torch RNG state and therefore on what ran before. Replaced by a peaked
stub evaluator that forces the line, so both engines are checked against a
literal — a stronger gate than mutual agreement, since agreeing on the wrong
answer now fails. The repo already carried this lesson on
`test_rust_and_python_searchers_produce_the_same_snapshot_shape`; I wrote the
test it warns against.

---

## 4. Frame selection

`findGameWindow` drops `testuser=` frames and prefers the shallowest — that part
predates these commits. Added: two loud refusals for what the URL convention
cannot resolve, and reporting them to the panel.

**Claims to attack:**

17. **The URL is the only available discriminator.** Inside a frame, both
    `me_id` and `gameui.player_id` describe the seat *that frame renders*, so an
    impersonated seat is indistinguishable from a real one, and no logged-in-user
    id appears anywhere in the client dump. This is why the plan's proposal
    ("pick the frame whose `me_id` is the logged-in user") is not merely undone
    but **unbuildable** — I have recorded that rather than leave it as an open
    task someone retries.
18. **Ambiguity and spectator frames should raise rather than guess.** Both
    carry `err.swdAmbiguous`. The spectator case matters because such a frame
    reads the whole public board and none of the private args, so it advises
    happily until a Great Library position refuses.
19. **Making them loud required a `page_bridge.js` change**, and this is the
    subtle bit: it swallowed *every* exception from `findGameWindow` and
    returned null, so a throw would have left the panel dead with no reason —
    exactly how the original wrong-seat bug presented. Reported from the top
    frame only, since every frame walks the same tree. Plain "no gamedatas here"
    stays silent, because the bridge runs in every frame and most are not the
    board.
20. **Throwing affects the console workflow too.** `captureBgaGamedatas`,
    `captureBgaSlim` and `captureForAdvisor` all call `findGameWindow`, so a
    spectator pasting into the console now gets an error instead of a capture.
    Intended, but it is a behaviour change beyond the extension.
21. **`gamedatas.players` is keyed by player id and contains the seated
    players**, which is what the spectator check tests `me_id` against. Used
    elsewhere in the mapper the same way, so this is consistent rather than new.

---

## 5. Cross-cutting

22. **Two tests were red or flaky before this branch was clean**, both mine, both
    found by the full suite rather than by me:
    `test_page_bridge_states_match_the_mapper` had been red since item F (its
    expected set is derived from the mapper's constants and did not include the
    new one), and the H test above. Both were validated with *targeted* runs. I
    have moved to running the full suite before each commit. Worth checking
    whether anything else in these four commits was only ever exercised
    targeted.
23. **The advisor still serves `laptop_training_03_w7/current_best.pt`**, trained
    under the old age-deal ordering. The strength gate specified in the previous
    review was deliberately deferred by the user, so its last-card-of-Age play is
    **unverified, not verified-fine**. Nothing in these four commits depends on
    it, but the live session will.
24. **Nothing visual has been rendered.** G's art, H's follow-up line and F's new
    decision row have never been seen in a browser. The extension last ran
    before the Rust searcher, batched evaluation, the victory outlook and the
    draft landed.

## What I did not do

* No live session (the extension re-test is the next task after this review).
* No benchmark of `follow_ups()` per publish (claim 16).
* Item E (draft-preference extraction) is untouched.

Claim 4 was on this list and came off it: writing the document prompted the
check, and the source settled it. Claims 2, 8 and 9 remain reasoned rather than
measured.
