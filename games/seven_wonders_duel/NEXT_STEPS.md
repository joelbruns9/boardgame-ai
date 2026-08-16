# 7 Wonders Duel — Next Steps

_Status as of 2026-08-15. Goal: the strongest 7WD player in the world, which
means no known correctness bugs and an encoder that actually helps learning._

---

## Where we are

The engine is in good shape and, for the first time, that claim rests on
something external. Four real BGA games were replayed move by move against
BGA's own published arithmetic — every price, coin reward, pawn position and
end-game score category. All of it now matches.

Two real defects were found and fixed along the way:

- **The encoder could not see the discard pile.** A card revivable with an
  unbuilt Mausoleum was invisible to the "can this player still win by
  science / military" features. This is the bug behind the lost game that
  started the whole investigation: one move before losing to a science rush,
  the encoder was telling the net the opponent *could not* win that way.
- **The military track was off by one.** Coin tokens were modelled as sitting
  on single spaces instead of covering bands, and the victory-point bands were
  shifted with them. This changed who won games, not just the bookkeeping.

Both fixes are committed on the `sevenwd-engine-correctness` branch.

**The cost of those fixes:** the encoder's meaning changed, so every existing
checkpoint is stale. The advisor currently runs on the old weights fed the
corrected features, which measurement says is better than what we had before,
but it is a stopgap and it is labelled as one in every response.

We also now capture every position the advisor sees while you play, which is
both a validation corpus and a source of real positions for training.

---

## The open thread that started all this

Science-threat blindness has been **root-caused and fixed in the encoder**, but
**not yet confirmed on a retrained net**. That confirmation is the real finish
line for the original problem, and it can't happen until we retrain.

A related observation worth chasing later: across recent games the opponent has
repeatedly reached five of six science symbols. That may be normal play, or it
may say something about how the model values denial. It needs a retrained net
before it's worth asking.

---

## Next steps, in the order they unblock each other

### 1. Make the differential harness durable — DONE

    python -m games.seven_wonders_duel.bga_differential

replays every captured game and compares every number BGA published. Two things
changed from the plan. It does **not** fetch from BGA: the archive endpoint now
refuses to serve the notification log of a finished table
(`Cannot find gamenotifs log file of an archived table`), so a fetch-it-back
design would have rotted within days. Instead the extension keeps BGA's raw
notification packets while you play, exactly as it already keeps positions, and
the harness runs entirely off local files. And the seat framing is no longer
hand-entered per table: it is derived from who picked the first wonder, and
cross-checked against the `startPlayerId` in the logged position, because
getting it wrong flips the military sign silently.

The capture hooks `gameui.notifqueue.onNotification`, the one call every
incoming packet passes through — verified live against a replay page, along
with the two things that make it the right hook: a reloaded page is re-sent its
history through it, and the coin/score snapshots the harness compares ride on
framework packets that per-notification subscriptions never see. The whole path
was then run end to end on a real game — page hook, capture, game log, replay —
for 0 mismatches over 74 packets.

Two real games are checked into `testdata/bga_packets.json` so the harness is
tested against BGA's arithmetic on every run, not only when someone has a fresh
capture to hand. A replay whose packet capture has gaps is skipped rather than
replayed: one missed construct rebases every later coin comparison, which would
report as a wall of mismatches nowhere near the real problem.

### 2. Validate the encoder's existing features — DONE, and it is clean

    python -m games.seven_wonders_duel.encoder_audit --games 200 --hunt 6

`encoder_audit.py` plays games and tries to catch the encoder saying something
untrue, three ways: against what the game went on to do, against what the
engine did on the very next move, and against a directed search that tries to
beat the bounds the encoder claims. 200 games, 12,325 decision points, 378,627
individual claims, 4,800 of them searched against — **nothing contradicted**.
The corpus reached all three endings (70 civilian, 73 military, 57 scientific),
which matters: a run that never produced a science win would have tested the
science claims not at all and still reported clean.

**The finding worth keeping** is about the method, not the result. The design
NEXT_STEPS assumed — watch for "this player can no longer win by science" being
contradicted by the game — is far too weak. Put the original discard-pile bug
back and 60 games with 7,396 such checks find *nothing*, because a science
victory needs a chain of luck on top of the wrong claim. What does find it is
attacking the *bound underneath the flag* (`sci_missing_obtainable`: "at most N
more symbols are gettable") with a cooperative directed search: 20 games, and
it produces a concrete line. The exact solver was not needed and its five-card
depth limit would have made it useless here anyway.

`test_encoder_audit.py` reintroduces that bug and requires the audit to
rediscover it, so the audit cannot rot into decoration.

**Where the coverage stops**, and what step 3 should keep in mind: the checks
that run the engine's own helpers on the stub and on the real state
(`card_cost`, `stub_economy`, `score_block`, `global_block`) are formula-free
and catch reconstruction losing data — the class the discard bug belonged to.
They cannot catch a feature whose *definition* is wrong in the same way on both
sides. The tableau geometry features (`row`, `x`, `accessible`, `coverers`),
the wonder/progress/discard/pool token features, and the trade-price block are
checked only indirectly, through pricing. Nothing is checked against real BGA
positions, because a logged position gives an observation with no ground-truth
`GameState` behind it.

### 3. Decide the final encoder change — one shot

Only after step 2. The intent is that this is the **last** encoder change, so
anything we want to add gets batched into it: new features, better ways of
expressing threats, whatever step 2 reveals is missing. Doing this before
validating what's already there risks a second "last" change.

### 4. Retrain

Needed regardless, and it unblocks several things that are currently stuck: the
two equivalence tests that replay a recorded corpus, the strength gate, and the
confirmation that the science fix actually works. Should happen once, after
step 3, not before.

### 5. Port the exact endgame solver to Rust, then put it in self-play

The solver is pure Python today and runs at roughly 1,500 nodes per second,
which is why it can only answer positions with about five cards left. A Rust
port with the existing make/unmake engine should be worth tens of times that,
which is the difference between "answers the last few moves" and "answers real
endgames".

Then it belongs in the self-play loop, where exact endgame values are ideal
training targets — precisely in the region where we measured the value head
being badly wrong. Kingdomino already has this whole pattern (a dedicated
solver worker pool, a time budget that changes over the run, and a way to price
non-best moves) and 7WD should copy it rather than reinvent it.

Two things learned the hard way that should carry over: Kingdomino's
transposition-table speedup **does not transfer** — 7WD endgames barely
transpose, measured at 1.13× — and solving every move exactly for training
labels costs several times more than solving just the position, which is what
Kingdomino's policy-mode setting exists to manage.

### 6. Use the captured games as self-play starting points

Real human positions are a source of variety that self-play cannot generate on
its own. The capture already produces positions that load straight back into
the engine, so this is mostly plumbing into the training loop.

### 7. Show the exact endgame answer in the advisor

The host already computes it correctly and returns it; the browser panel simply
never displays it. Small, self-contained, and independent of everything above.
Worth doing whenever the solver is fast enough to be useful in a real turn.

### Ongoing: keep playing

Every game adds roughly forty priced moves and a hundred-plus checks for free,
and it costs nothing but the game you were going to play anyway. Afterwards:

    python -m games.seven_wonders_duel.bga_differential

Three checks no captured game has ever reached are worth a game aimed at them:
the **Masonry rebate** (`with_rebate_token`), an **Economy transfer**
(`economy_transfer`), and a game that ends *on points* with the conflict pawn at
3–5 or 6–8, which is the untested half of the military scoring bands. The report
names whichever checks went unexercised at the bottom of every run.

---

## Known loose ends

- **The military victory-point bands are still unverified against BGA.** The
  coin-token half is proven; the scoring half needs a game that ends *on
  points* with the conflict pawn at distance 3–5 or 6–8. Recent games each had
  one half of that condition and not the other.
- **Six progress tokens have never been taken** in any captured game: Masonry,
  Strategy, Theology, Philosophy, Mathematics, Agriculture. Masonry is the last
  untested pricing rebate.
- **Two tests fail and are expected to** until the retrain: both replay a
  recorded corpus of games generated by the previous engine. A third,
  `test_training_parameters_doc`, was already failing before any of this work
  and is unrelated.
- **The advisor is running on migrated weights.** It is honest about it — every
  response says so — but the numbers are off-distribution until the retrain.
- **No game has yet been captured start-to-finish by the shipped extension.**
  The path is verified end to end against a replay, but the first real game is
  the test of whether the tab is open early enough to catch move 1; if it is
  not, the harness says so (it skips a game with gaps rather than replaying it)
  and the fix is to reload the table. Until then the corpus is the four games
  captured by hand, kept as
  `runs/seven_wonders_duel/bga_archive/packets_4games_20260814.json`, two of
  them trimmed into `testdata/`.
- **The captured game logs are not in version control** (`runs/` is ignored),
  and they can no longer be re-fetched: BGA refuses to serve a finished table's
  notification log. The local copies are now the only ones.

---

## Rough sequencing

Steps 1 and 2 are **done**, so the retrain is no longer gated by validation —
it is gated by step 3, deciding the final encoder change, which is now the
current work. Step 2 found no bugs to fix, so what it feeds into step 3 is a
map of where the checks are thin (see the end of step 2), not a list of
corrections. Step 5 can proceed in parallel at any time — it touches nothing
the encoder work touches. Step 7 is a small win available whenever someone
wants it. Steps 4 and 6 follow once step 3 lands.
