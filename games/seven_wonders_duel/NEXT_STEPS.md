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

### 1. Make the differential harness durable — small, do first

The tool that found the military bug currently exists only as a scratch script.
It should live in the repo, read table ids out of the game log, fetch each
game's record from BGA itself, and run the comparison from one command. Until
that exists, "we'll validate more as I play more games" is not actually true.

### 2. Validate the encoder's existing features — the big one

We have checked that the *engine* is right. We have **not** checked that the
encoder's ~65 derived features say true things. We found the discard bug by
accident, from a game that happened to be lost in a memorable way. That is not
a method.

The sharpest available check: features like "this player can still win by
science" are *claims the exact solver can falsify*. Any position where the
encoder says impossible and the solver finds a winning line is a bug, found
automatically over thousands of positions with nobody watching. That single
check would have caught the original bug with no human insight involved.

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
and it costs nothing but the game you were going to play anyway.

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
- **The captured game logs are not in version control** (`runs/` is ignored).
  They can be re-fetched from BGA, but the local copies are the only ones.

---

## Rough sequencing

Steps 1 and 2 are the current work and gate the retrain. Step 5 can proceed in
parallel at any time — it touches nothing the encoder work touches. Step 7 is a
small win available whenever someone wants it. Steps 3, 4 and 6 follow in order
once step 2 is done.
