# W3: wiring exact positional control into the encoder — plan of record

**Status:** solver built and verified; nothing wired. **Revision 3**, after a
second review and a design discussion that cut the feature set roughly in half.

Prior review of the solver is in `W3_CONTROL_REVIEW_REQUEST.md` (six findings,
accepted, items 1–3 committed in `e017ab0`). Revision 2 answered a review of the
wiring plan. Revision 3 is the result of pressure-testing what each feature
actually buys, and most of the changes are **deletions**.

## What changed, and why

| revision 2 | revision 3 | why |
|---|---|---|
| counterfactuals: `+1`, `−1`, `−2` tempo | **all three dropped** | `+1` is unreachable; `−1`/`−2` duplicate a one-ply successor search already expands |
| per-action successor control lookup | **dropped from arm 1** | MCTS already encodes and evaluates each child; `top_k=16` vs mean legal width 5.6 |
| Theology maps "held for arm 2" | **promoted into arm 1** | the only reachable increase in tempo, and swings up to ±18 slots |
| — | **start-decision feature added** | swings up to 20 of 20 slots and sits an age beyond search's horizon |
| — | **auxiliary control head added** | direct supervision beats hoping outcome credit propagates |
| Age III scoping considered | **all ages; Age III is the evaluation focus** | tempo carries across ages, so the strategic cost is only visible game-wide |

Net effect: **six channels per tableau token plus one validity flag**, down from
ten channels, five maps and two per-action gathers.

---

## The two things the model must learn

Everything below is in service of these, and they are not equally served.

**Tactical — keep a critical card.** *The opponent needs one more science symbol;
I currently reach that slot first; does this move cede it?* Search can already
see this once the encoder makes it visible: MCTS encodes and evaluates each child
position, so a child where the card has changed hands carries its own (worse)
control features and the value head marks it down. The feature's job is to make
search's evaluation honest, so the **target** is right, so the distilled prior is
right. This is hardest to learn in Ages I–II, where the outcome signal is 20–60
moves away; it is most learnable, and most decisive, in Age III.

**Strategic — price tempo and the start decision.** *An extra-turn Wonder spent
in Age I is not available in Age III.* This is where features do something search
structurally **cannot**: the consequence sits an entire age — roughly 20 moves —
beyond any simulation budget. Fresh Age III, full pyramid:

| my unbuilt extra turns | I start | they start |
|---|---|---|
| 0 | 14/20 | **6/20** |
| 1 | 20/20 | 18/20 |
| 2 | 20/20 | 16/20 |

Spending your extra-turn Wonder early and then losing the start decision is the
difference between 18 slots and 6. Tempo counts carry across age boundaries, so
positions *entering* Ages II and III with depleted tempo evaluate visibly worse,
and that propagates back to the Age I build decision. Without control features
the same information exists only as raw tempo counts, and reconstructing the
topological consequence is exactly the generalisation the plateau says the net
cannot do.

A second strategic fact, live from the first Wonder and already modelled: every
Wonder built advances the shared pool toward its seventh, and closing the pool
retires whatever is still unbuilt — sometimes the opponent's tempo, sometimes
your own.

---

## What is already true

`tableau_control.py` answers, exactly, *who reaches this tableau slot first*,
given the removal poset, turn alternation, and the shared seven-Wonder pool
including retirement. Public information only, so it is safe on the determinized
states `advisor_scrape` hands the searcher.

| evidence | checks | result |
|---|---|---|
| brute-force oracle, 4–7 slot posets, all ages, pools 0–4 | 2,452 | agree |
| structurally different oracle (iterative deepening, not minimax) | 538 | agree |
| monotonicity properties, no oracle involved | 840 | hold |
| `tempo_state` invariant `unbuilt == builds_left + 1` on real positions | 289 | 0 violations |

**Not verified:** full 20-slot boards against an oracle (exponential); and the
abstraction itself — one Age, no coins, production, chains, military or early
victory. Documented limits, not measured ones.

**Known test gap:** the invariant test skips exhausted pools
(`if tempo[0] == 0: continue`), so it never exercises the sixth-to-seventh build,
the one moment retirement fires. Add explicit transition cases stepping the
engine through that build.

---

## Cost

Solving at encode time is dead: `control_features()` costs **816 ms mean, 5.8 s
worst** against a ~1 ms leaf. The key space is structural and small.

| quantity | value |
|---|---|
| reachable masks (age 1 / 2 / 3) | 428 / 428 / 132 = **988** |
| tempo states (Theology closure adds none — it maps natural states to natural states) | **220** |
| total keys, `(age, mask, who_moves, tempo)` | **434,720** |
| `control_map` mean / p90 / max | 23 / 64 / 534 ms |
| **full precompute** | **~2.8 core-hours — ~15 min on 12 cores** |
| **table size** at 20 bytes + header | **~10 MB** |

Precompute offline, ship the table, look up at encode time. The solver never runs
in the training loop and needs no Rust port; Rust reads the table and derives the
same key. **These are claims the generator must reproduce, not inputs to it.**

Age III alone would be 58,080 keys (~22 min single-core). Not worth scoping to:
the cost is negligible either way, and the strategic signal requires all ages.
Age III's distinctions are that it is the evaluation focus and that below seven
present cards it hands off to the existing exact solver
(`endgame_corpus.py`, `MAX_PRESENT = 6`), giving Age III unbroken coverage.

---

## Applicability: the P1 from revision 2

`state_actor` returns `pending_choice.player` when a choice is pending
(`search.py:352`), and `_finish_turn` stashes `pending_extra_turn` to apply after
the choice resolves (`engine.py:494`). The current decision-maker is therefore
not the next tableau mover. Who *actually* makes the next removal, measured over
complete games:

| state | share of all states | actor is the mover | actor is not |
|---|---|---|---|
| `PLAY_AGE`, no pending choice | 83.3% | — | — |
| `WONDER_DRAFT` | 11.2% | 160 | 160 |
| `PLAY_AGE` + pending choice | 2.7% | 4 | **61** |
| `CHOOSE_NEXT_START_PLAYER` | 2.8% | 36 | 44 |

`active_player` was identical to `state_actor` in every one of these, so there is
no rescue by picking a better field. **16.7% of training rows would carry a
`who_moves` that does not mean what the key says**, and because
`pending_choice.player` always equalled `active_player`, perfect Python/Rust
parity would hide it entirely. Taking Theology during a pending choice also
restates the tempo half of the key.

**Decision.** Emit `control_valid` as an explicit channel. Compute features only
in `PLAY_AGE` with no pending choice; elsewhere the flag is 0 and every control
channel is 0. `WONDER_DRAFT` stays masked indefinitely — a 50/50 mover coin flip
whose outcome also determines the tempo inventory the key depends on. Never
silently read "current decision maker" as "next tableau mover".

---

## Feature design

**Key.** `(age, present_mask, who_moves, tempo_state)` — public structure only.

**Entry.** Per slot: attacker turns to force the take, or unreachable.
`control_map` discards the distance `solve()` already computes; the table keeps
it at no extra solve cost.

**Per-slot on the existing tableau tokens.** `_tableau_tokens()`
(`encoder.py:704`) already emits one token per present slot, so control attaches
as extra channels on tokens that exist — no new token type, no sequence-length
change, and control lands on the same token as the card's identity. That
co-location is the point: *this is Observatory* and *they reach it first* must be
one object for the net to weigh them together.

**Three maps, six channels.** For live, my-Theology and their-Theology:

- `can_force_take_under_topology` — binary.
- `forced_take_turns_scaled` — bounded, normalized, **zero when the flag is
  zero**. `_INF` is 99 and must never be fed as a distance.

Plus the single global `control_valid` flag. Global fractions are dropped: prior
findings showed aggregate control obscuring the strategically important card, and
per-slot is strictly better placed.

### Which counterfactuals earn a channel

> A counterfactual map earns a channel only when the contingency lies **beyond
> what search will actually expand**.

- **`+1` extra turn — cut.** Unreachable. The count of unbuilt extra-turn Wonders
  never increases: 0 increases against 37 decreases across 1,473 consecutive
  state pairs. Your four Wonders are fixed after the draft.
- **`−1` and `−2` — cut.** Spending a Wonder is one legal action away, so search
  expands it, encodes the successor and evaluates it. A counterfactual that
  duplicates a one-ply successor is dead weight. (`−1` measured a state that
  cannot occur anyway: mask held fixed, option removed.) The live map already
  computes reachability *using* every extra turn you hold, chaining included.
- **Theology, both sides — keep.** The only way tempo increases, and it converts
  *all* your unbuilt Wonders at once rather than adding one. It may be many plies
  away and may never be expanded, and the swing is large: taking it is worth up
  to **+18 slots**, conceding it up to **−16**. That is the reference game's
  decisive concession.

**Semantics, stated so the names cannot mislead.** A zero flag means *the
attacker cannot force the take against a defender playing optimally for that
target in this abstraction*. It does not mean the card is unobtainable. Because
affordability is unmodelled for **both** players, the result is an exact outcome
in the abstract game and **not a bound in either direction** — the error can go
either way.

### The start decision

`CHOOSE_NEXT_START_PLAYER` fires twice per game, at the starts of Ages II and
III, and the next age's pyramid is already dealt at 20/20 when the choice is
made. The decision *is* the `who_moves` key parameter, so the feature is two
lookups on keys the table already holds: control under *I start* and under *they
start*. Swings measured on the fresh pyramid:

| age | my extra / opp extra | I start | they start | swing |
|---|---|---|---|---|
| 2 | 0 / 0 | 11 | 9 | 2 |
| 2 | 2 / 2 | 14 | 6 | **8** |
| 3 | 0 / 0 | 14 | 6 | **8** |
| 3 | 1 / 1 | 9 | 11 | **−2** |
| 3 | 2 / 2 | 20 | 0 | **20** |

Note the non-monotonicity: at Age III with one extra turn each, starting is
*worse*. No "prefer the extra-turn Wonder" heuristic gets that right, and it is
precisely where a science defender must decide whether to hand over the start.

### Auxiliary control head

Add a head that **predicts** the control map, trained against the table as
labels. Input tells the net the answer; an auxiliary target forces the trunk to
*represent* it rather than relying on outcome credit propagating back across 20–60
moves. Labels are free, and the head can be dropped at inference — so it costs
nothing in self-play and sidesteps Rust parity entirely. It is the cheaper half
to try first, and it matters most in Ages I–II where the outcome signal is
weakest.

### Data emphasis

Oversample contested-control positions. Measured: of Age II/III states where the
actor controls a *revealed* science card, **46%** have at least one legal move
that cedes it — commonly 2 or 3 such moves, sometimes 8. `dataset.py` already
carries the oversampling machinery.

---

## Failure, parity and staleness

**On a table miss: fail explicitly.** Validate coverage and contract
compatibility at startup. An unexpected miss raises a descriptive typed exception
naming the key and artifact identity — **not a bare `assert`**, which `-O`
disables. Self-play and advisor inference must never enter a multi-second solver.
Offline tooling gets an explicit "solve missing entries" mode. For advisor
availability, fall back to the whole known-good model/encoder pair rather than
filling W3 features with zeros.

**Parity, in three gates:**

1. **Key derivation** — generated states covering Wonder identities,
   built/retired combinations, Theology, both perspectives, phase boundaries.
2. **Table readers** — exhaustively compare Python and Rust lookups across the
   complete generated key set.
3. **End-to-end encoding** — compare final feature vectors including slot
   association, normalization and invalid-state masking, on real games and
   generated transitions.

Routing all Python encoding through Rust just to derive this small key is not
warranted.

**Staleness by contract, not by commit.** A git hash is provenance, not a
compatibility check; a no-op refactor producing identical contents must not
invalidate checkpoints. Pin: feature schema and normalization version;
layout/slot-order and rule-data identity; counterfactual definitions and
supported key set; table-content digest; and checkpoint metadata naming the
required contract.

---

## Sequencing

`PLATEAU_FINDINGS.md` ranks data scaling **fifth** and calls its rationale weak —
"it measures quantity while the plausible problem is diversity" — so it is not a
prerequisite. The same list says of the encoder: "Still unmeasured, still by
elimination only. **Do not start here.**" That judgement priced an expensive
encoder bet; at ~15 minutes of generation and ~10 MB it is no longer one, and the
question becomes tractable: *does this cheap structural input improve important
decisions while preserving existing strength?* We proceed on that basis
deliberately, not by overlooking the warning.

1. Fix feature semantics and applicability; add the sixth-to-seventh transition
   test.
2. Build the generator; validate the key-space and cost numbers above.
3. **Auxiliary head first** — cheapest signal on whether the trunk can represent
   control at all, with no inference cost and no parity work.
4. Encoder inputs; offline fine-tune against the same checkpoint, data and budget
   versus an identical run without W3.
5. Evaluate on held-out games **and** ordinary-strength positions, with Age III
   tactical positions as the focus and the strategic claim (start decision, tempo
   pricing) as the primary gate — it is where features do what search cannot.
6. Production Rust integration only if early evidence is useful.
7. Equal-wall-clock strength testing before any promotion.

**Migration is append-only and zero-initialized**, preserving the incumbent
exactly — the same discipline W5a uses for its gate.

**The tactical corpus is a development and regression set, not a holdout.** It
has been inspected extensively; wiring W3 to win exactly those positions would
end its value as evidence. Reserve fresh games and keep all positions from one
game together. A shuffled-feature arm is worth running but is an input-corruption
test, not clean causal attribution.

---

## Explicitly rejected

- **Per-action successor control in arm 1.** MCTS encodes and evaluates each
  child, and `top_k = 16` against a mean legal width of 5.6 means essentially
  every root child is expanded. The value head therefore already sees a ceded
  card's consequence. Revisit only as *prior shaping* if the corpus shows the
  prior is the binding constraint — which W9 found it to be at depth, but not at
  the root.
- **Building control into W5a's action scorer instead of the encoder.** The
  scorer feeds policy logits only, while the documented failure includes a value
  head 36 points wrong at post-burial nodes. Encoder placement reaches both
  heads, keeps the two workstreams separately evaluable, and puts the fact on the
  slot it belongs to. W5a remains the ideal *consumer*: it can gather the token
  and, with one added "exposes" index, reach the slot a burial uncovers rather
  than only the card it spends.
- **A control-aware advisor bot generating training data.** It would make the
  human stronger tomorrow but teaches the net to imitate the solver's preferences
  rather than to evaluate positions — the circularity trap the Kingdomino denial
  curriculum ran into.
- **Scoping to Age III.** Tempo carries across ages, so the strategic cost of
  early spending is only visible game-wide.

---

## Still open

- Does the auxiliary head suffice on its own, making encoder inputs (and all Rust
  parity work) unnecessary?
- Whether `forced_take_turns_scaled` should saturate or use a separate ceiling
  channel.
- Whether choice-conditioned features are ever worth building for the masked
  16.7%, or `control_valid` is the permanent answer.
- The "exposes" gather index for W5a: cheap, and the only way the action scorer
  reaches the reference case. Arm 2 or later.
