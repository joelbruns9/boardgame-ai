# W3: wiring exact positional control into the encoder — plan of record

**Status:** solver built and verified; nothing wired. **Revision 4**, after a
third review. Revision 3 cut the feature set roughly in half; revision 4 keeps
those cuts but **replaces their justifications**, because several were stated as
structural facts on partial evidence. Three code claims were checked and all
three were wrong (see *Corrections* below).

Prior review of the solver is in `W3_CONTROL_REVIEW_REQUEST.md` (six findings,
accepted, items 1–3 committed in `e017ab0`). Revision 2 answered a review of the
wiring plan. Revision 3 is the result of pressure-testing what each feature
actually buys, and most of the changes are **deletions**.

## What changed, and why

| revision 2 | revision 4 | why |
|---|---|---|
| counterfactuals: `+1`, `−1`, `−2` tempo | **all three dropped** | `+1` is unreachable (measured); `−1`/`−2` are one legal action away, so search routinely expands the successor |
| per-action successor control lookup | **deferred — a scope decision, not proven redundancy** | see *Corrections* |
| Theology maps "held for arm 2" | **promoted into arm 1** | the only reachable increase in tempo, and swings up to ±18 slots |
| — | **start-decision channels added, with their own layout and validity** | swings the whole age; resolves the applicability conflict |
| — | **auxiliary control head added, as an arm and not a gate** | direct supervision, but its failure vetoes nothing |
| Age III scoping considered | **all ages; Age III is the evaluation focus** | tempo carries across ages |

Net effect: **six per-slot channels plus two start-decision channels plus two
validity flags**, down from ten channels, five maps and two per-action gathers.

## Corrections carried into revision 4

Three claims in revision 3 were checked against the code and did not hold. They
are recorded rather than quietly edited, because each was stated categorically on
partial evidence.

1. **"Beyond any simulation budget."** `chance_signature` (`search.py:150`)
   handles `AGE_DEAL` explicitly and descent continues through it. There is no
   age-boundary cutoff; the difficulty is funding representative continuations
   through a large chance tree. Restated as a budget-and-data hypothesis.
2. **"Unbroken Age III coverage via the exact solver."** `MAX_PRESENT = 6`
   selects corpus candidates and exists because the Python reference times out
   above six cards. Not a runtime handoff. W3 now covers every reachable mask.
3. **"Essentially every root child is expanded."** Mean legal width does not
   bound the tail; the advisor runs `RustPuctSearch` (`advisor_adapter.py:673`),
   not Gumbel top-k; creating a legal edge is not evaluating its successor; and
   root coverage says nothing about low-prior decisions deeper in the tree.
   Deferring per-action successor features is now a scope decision.

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
in Age I is not available in Age III.* The hypothesis is that **W3 reduces the
search budget and training data needed to learn cross-age tempo value** — not
that search cannot reach it. Search models age-deal chance explicitly
(`chance_signature`, `search.py:150`) and descends through it; the difficulty is
funding representative continuations through a large chance tree. An earlier
draft called this a structural impossibility, which the code contradicts.

Note also that the inputs describe the **current** age's tableau. Better Age III
features improving Age I decisions is a claim about learning and propagation, not
something the feature construction guarantees. Fresh Age III, full pyramid, at
tempo `(builds_left = mine+theirs+2, ord_me = 1, ext_me = mine, ord_them = 1,
ext_them = theirs)`:

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
Age III's distinction is that it is the evaluation focus.

**W3 coverage does not stop below seven cards.** An earlier draft claimed a
handoff to the exact solver at `MAX_PRESENT = 6`. That constant selects *corpus
candidate positions* and its own comment gives the reason — "above ~6 the Python
reference itself times out (7 ~37k/29s)". It is a build-time gate, not a runtime
contract, and the advisor's exact solver is optional, cost-gated, and may decline
or time out. Generate W3 for every reachable mask.

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

**Decision.** Two independent channel groups, each with its own validity flag,
so no channel ever changes meaning by phase. Revision 3 zeroed all control
channels outside ordinary `PLAY_AGE` while also adding start-decision maps at
`CHOOSE_NEXT_START_PLAYER` — an implementer could not satisfy both. Separate
groups are chosen over a phase-dependent layout because they are auditable.

| group | channels | populated when | else |
|---|---|---|---|
| **live control** | 6 per tableau token (3 maps × flag + distance) | `PLAY_AGE`, no pending choice | all 0 |
| `control_valid` | 1 global | always | 1 or 0 |
| **start decision** | 2 per tableau token (I-start, they-start; flag + distance each → 4) | `CHOOSE_NEXT_START_PLAYER` | all 0 |
| `start_choice_valid` | 1 global | always | 1 or 0 |

The two groups are never both populated. `WONDER_DRAFT` and pending-choice states
populate neither: the draft is a 50/50 mover coin flip whose outcome also
determines the tempo inventory the key depends on. Never silently read "current
decision maker" as "next tableau mover".

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

These are **topology counts, not move recommendations.** Nine forceable slots
versus eleven does not mean a worse position: starting could secure the one
science card that decides the game while conceding several irrelevant ones. That
is exactly why global fractions were dropped, and the same caution applies here.
The non-monotone row is interesting because a "prefer the extra-turn Wonder"
heuristic cannot produce it — not because it tells you to decline the start.

The tempo tuples are published above because extra-turn counts alone do not
identify the solver state: ordinary Wonders and `builds_left` change the answer
through retirement.

### Auxiliary control head

Add a head that **predicts** the control map, trained against the table as
labels. Input tells the net the answer; an auxiliary target pushes the trunk to
*represent* it rather than relying on outcome credit propagating back across
20–60 moves. Labels are free, and the head can be dropped at inference — so it
costs nothing in self-play and sidesteps Rust parity entirely. That makes it the
cheapest arm to run first.

It is an **arm, not a gate**: see *The arms* under Sequencing. Its success can
make the input arm unnecessary; its failure cannot disqualify it.

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
3. **Auxiliary head first** — cheapest arm: no inference cost, no parity work.
4. Encoder inputs, offline fine-tune against the same checkpoint, data and
   budget.
5. Evaluate (below). Age III tactical positions are the focus; the strategic
   claim is an important gate but **not a veto on a real tactical gain**.
6. Production Rust integration only if early evidence is useful.
7. Equal-wall-clock strength testing before any promotion.

### The arms, and what each one answers

| arm | question |
|---|---|
| baseline | same checkpoint, data, training budget |
| auxiliary only | can extra supervision improve **decisions** without new inference inputs? |
| inputs only | does supplying exact abstract control beat learning to reconstruct it? |
| both, if warranted | does supervision add anything beyond supplied features? |

**An auxiliary failure does not veto the input arm.** They are different
interventions: the auxiliary head teaches the net to *compute* control, the
inputs give it control so it can *use* it. Failure to learn the calculation is if
anything an argument for supplying the answer, and conversely a head that
predicts control accurately does not show that policy or value uses it. The
existing probe already recovered some control from the trunk, so "can it
represent control at all" is the wrong question — **does auxiliary training
improve decisions** is the right one. Auxiliary-only may make the input arm
unnecessary by succeeding; it cannot disqualify it by failing.

**Judge on decisions, never on control-prediction accuracy alone:** tactical
regret, starter choices, earlier tempo-spending decisions, and general strength.
Include **military-critical** positions alongside science ones, and keep ordinary
-game data beside the contested-science oversampling — otherwise the experiment
can reward a "hoard Wonders" bias rather than stronger play. The strategic
evaluation must contain both positions where preserving tempo wins **and**
positions where spending it now is correct.

**Migration is append-only and zero-initialized**, preserving the incumbent
exactly — the same discipline W5a uses for its gate.

**The tactical corpus is a development and regression set, not a holdout.** It
has been inspected extensively; wiring W3 to win exactly those positions would
end its value as evidence. Reserve fresh games and keep all positions from one
game together. A shuffled-feature arm is worth running but is an input-corruption
test, not clean causal attribution.

---

## Explicitly rejected

- **Per-action successor control in arm 1 — DEFERRED, not disproven.** The
  argument that "search already evaluates every child" does not hold generally:
  mean legal width does not bound the tail, the advisor runs `RustPuctSearch`
  rather than Gumbel top-k, creating a legal edge is not evaluating its
  successor, forced root chance expansion is a separate mechanism with
  exclusions, and root coverage says nothing about low-prior decisions deeper in
  the tree. This is deferred to keep the first experiment small. **Before
  declaring it unnecessary, measure actual successor-evaluation coverage by
  search mode, budget, legal width and decision family.**
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
  parity work) unnecessary? (Only a *success* can settle this; a failure leaves
  the input arm untouched.)
- What is the real successor-evaluation coverage by search mode, budget, legal
  width and decision family? Until measured, deferring per-action features is a
  scope call rather than a finding.
- Whether `forced_take_turns_scaled` should saturate or use a separate ceiling
  channel.
- Whether choice-conditioned features are ever worth building for the masked
  16.7%, or `control_valid` is the permanent answer.
- The "exposes" gather index for W5a: cheap, and the only way the action scorer
  reaches the reference case. Arm 2 or later.
