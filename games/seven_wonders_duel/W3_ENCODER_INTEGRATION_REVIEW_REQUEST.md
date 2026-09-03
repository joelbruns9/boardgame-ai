# W3: wiring exact positional control into the encoder — plan of record

**Status:** solver built and verified; nothing wired. Revision 2, after review.

Prior review of the solver is in `W3_CONTROL_REVIEW_REQUEST.md`; its six findings
were accepted and items 1–3 are committed (`e017ab0`). Revision 1 of this
document proposed the wiring; the review of it found a P1 and two P2s, all
confirmed and all folded in below. The changes from revision 1 are large enough
that the diff is worth stating up front:

| revision 1 said | revision 2 says | why |
|---|---|---|
| key includes `who_moves`, from `state_actor` | plus a `control_valid` flag; features disabled outside clean `PLAY_AGE` | `state_actor` is not the next tableau mover in **16.7%** of encoder-visible states |
| per-slot for the live map, global fractions for the four counterfactuals | per-slot for **all five**, global fractions dropped from the first arm | the fractions discard the interaction we want learned |
| "a capability bound" | an exact outcome in an abstraction, **not a bound in either direction** | ignoring affordability grants options to *both* players |
| staleness enforced by recorded commit | enforced by a feature/table **contract** | a no-op refactor must not invalidate checkpoints |
| run the data-scaling curve first | no such prerequisite | `PLATEAU_FINDINGS.md` ranks it fifth and calls its rationale weak |

---

## What is already true

`tableau_control.py` answers, exactly, *who reaches this tableau slot first*,
given the removal poset, turn alternation, and the shared seven-Wonder build pool
including retirement. It reads only public information, so it is safe on the
determinized states `advisor_scrape` hands the searcher.

| evidence | checks | result |
|---|---|---|
| brute-force oracle, 4–7 slot posets, all ages, pools 0–4 | 2,452 | agree |
| a structurally different oracle (iterative deepening, not minimax) | 538 | agree |
| monotonicity properties, no oracle involved | 840 | hold |
| `tempo_state` invariant `unbuilt == builds_left + 1` on real positions | 289 | 0 violations |

**Not verified:** full 20-slot boards against an oracle (exponential); and the
abstraction itself — one Age, no coins, production, chains, military or early
victory. Those are documented limits, not measured ones.

**Gap the review found in the invariant test:** it skips exhausted pools
(`if tempo[0] == 0: continue`), so it never exercises the sixth-to-seventh
build — the one moment retirement fires. Add explicit transition cases that step
the engine through that build and check the derived tempo state against the
successor.

---

## The cost question, measured

Solving at encode time is dead: `control_features()` costs **816 ms mean, 5.8 s
worst** against a ~1 ms leaf. But the key space is structural and small.

| quantity | value |
|---|---|
| reachable tableau masks (age 1 / 2 / 3) | 428 / 428 / 132 = **988** |
| tempo states, closed under the counterfactuals | **444** |
| total keys | **877,344** |
| `control_map` mean / p90 / max | 23 / 64 / 534 ms |
| **full precompute** | **5.5 core-hours — ~0.5 h on 12 cores** |
| **table size** | **~21 MB** |

Precompute offline, ship the table, look up at encode time. The solver never runs
in the training loop and needs no Rust port; Rust reads the table and derives the
same key. **These numbers are claims to be reproduced by the generator, not
inputs to it** — the generator validates them.

---

## Applicability: the P1

`state_actor` returns `pending_choice.player` when a choice is pending
(`search.py:352`), and `_finish_turn` stashes `pending_extra_turn` to apply after
the choice resolves (`engine.py:494`). So the current decision-maker is not the
next tableau mover. Measured over 4,294 encoder-visible states from 60 complete
games:

| state | share |
|---|---|
| `PLAY_AGE`, no pending choice | 83.3% |
| `WONDER_DRAFT` | 11.2% |
| `CHOOSE_NEXT_START_PLAYER` | 2.8% |
| `PLAY_AGE` with pending choice | 2.7% |

**16.7% of training rows would carry a `who_moves` that does not mean what the
key says.** Note that `pending_choice.player` equalled `active_player` in every
one of these games: the corruption comes from `pending_extra_turn` and from the
phase, not from the actor field. Perfect Python/Rust parity would hide it
completely, which is exactly why parity is not the whole risk.

Choosing a progress token can also change the tempo inventory before the next
removal — taking Theology converts every unbuilt Wonder into an extra-turn
Wonder — so the tempo half of the key is stale in those states too, not just
`who_moves`.

**Decision.** Emit `control_valid` as an explicit channel. Features are computed
only in `PLAY_AGE` with no pending choice; everywhere else the flag is 0 and
every control channel is 0. Choice-conditioned features (what the key becomes
*after* this progress token is taken) are a later, separately specified arm. Do
not silently read "current decision maker" as "next tableau mover".

---

## Feature design

**Key.** `(age, present_mask, who_moves, tempo_state)` — public structure only.

**Entry.** Per slot, the attacker turns to force the take, or unreachable.
`control_map` currently discards the distance `solve()` already computes; the
table keeps it, at no extra solve cost.

**Per-slot for all five maps.** Revision 1 attached only the live map per slot
and reduced the four counterfactuals to global fractions. That preserves *one
more extra-turn Wonder improves my aggregate reach* and discards *one more
extra-turn Wonder is what lets me deny that sixth science symbol* — the second
being the interaction this workstream exists to represent. The lookups already
return those maps, so this costs feature channels, not solves.

Global fractions are **dropped from the first arm**. Prior findings already
showed aggregate control obscuring the strategically important card, and
`_tableau_tokens()` (`encoder.py:704`) gives us a per-slot home for free. They
remain a cheap later addition.

**Encoding per slot, per map:**

- `can_force_take_under_topology` — binary.
- `forced_take_turns_scaled` — bounded, normalized; **zero when the flag is
  zero**. `_INF` is 99 and must never be fed as a distance.
- gated by the single `control_valid` flag above.

**Semantics, stated so the names cannot mislead.** A zero flag means *the
attacker cannot force the take against a defender playing optimally for that
target in this abstraction*. It does not mean the card is inaccessible or
unobtainable. And because affordability is not modelled, the abstraction grants
extra Wonder options to **both** players — so the result is an exact outcome in
the abstract game and **not an upper or lower bound on the real one**. The error
can go either way. Revision 1 called it a capability bound; that was too strong.

**Counterfactual naming.** `_minus_extra` means *remove one extra-turn Wonder
option*, not *after I build one*: an actual build also removes a tableau card and
consumes shared build capacity. The docstring currently reads "what losing my
tempo would cost", which conflates the two.

---

## Failure, parity and staleness

**On a table miss: fail explicitly.** Validate table coverage and contract
compatibility at startup. An unexpected miss raises a descriptive typed
exception naming the key and the artifact identity — **not a bare `assert`**,
which `-O` disables. Self-play and advisor inference must never enter a
multi-second solver unexpectedly. Offline tooling gets an explicit
"solve missing entries" mode. For advisor availability, fall back to the whole
known-good model/encoder pair rather than filling W3 features with zeros.

**Parity, in three gates** rather than one corpus:

1. **Key derivation** — generated states covering Wonder identities,
   built/retired combinations, Theology, both perspectives, and phase
   boundaries.
2. **Table readers** — exhaustively compare Python and Rust lookups across the
   complete generated key set.
3. **End-to-end encoding** — compare final feature vectors, including slot
   association, normalization and invalid-state masking, on real games and
   generated transitions.

Requiring all Python encoding to route through Rust just to derive this small key
is not warranted.

**Staleness by contract, not by commit.** A git hash is provenance, not a
compatibility check; a harmless refactor producing identical table contents must
not invalidate checkpoints. Pin instead: feature schema and normalization
version; layout/slot-order and rule-data identity; counterfactual definitions and
supported key set; table-content digest; and checkpoint metadata naming the
required contract. A sixth counterfactual does not automatically invalidate the
table either — it may reuse existing keys. The gate is whether every requested
key and feature meaning is still covered.

---

## Sequencing

Revision 1 made the data-scaling curve a prerequisite. That was wrong on the
document of record: `PLATEAU_FINDINGS.md` ranks data scaling **fifth** and says
its "rationale is weak: the window was full at 40k and all of it came from a
policy that barely changed, so it measures quantity while the plausible problem
is diversity". More games from the same policy cannot cleanly separate a data
limit from a representation limit, and both may hold at once.

The same list also says of the encoder: "Still unmeasured, still by elimination
only. **Do not start here.**" That judgement was made when an encoder change was
an expensive bet. At 0.5 core-hours and ~21 MB it no longer is, and the question
becomes the tractable one: *does this cheap structural input improve important
decisions while preserving existing strength?* This plan proceeds on that basis
deliberately, not by overlooking the warning.

1. Fix feature semantics, applicability and per-slot counterfactual design.
2. Build a reproducible generator; validate the key-space and cost numbers above.
3. Offline W3 fine-tune against the same checkpoint, data and training budget,
   versus an identical run without W3.
4. Evaluate on held-out games **and** on ordinary-strength positions.
5. Production Rust integration only if the early evidence is useful.
6. Equal-wall-clock strength testing before any promotion.

**Migration is append-only and zero-initialized**, so the incumbent is preserved
exactly — the same discipline W5a uses for its gate.

**The tactical corpus is a development and regression set, not a holdout.** It
has been inspected extensively, and wiring W3 to win exactly those positions
would end its value as evidence. Reserve fresh games, and keep all positions from
one game together. A shuffled-feature arm is worth running but is an
input-corruption test, not clean causal attribution.

---

## Still open

- Choice-conditioned features for the 16.7%: worth building, or is the validity
  flag the permanent answer?
- Whether per-slot counterfactuals should be all four, or only the
  `+1 extra turn` map, given the channel cost on every tableau token.
- Whether W5a's shared action scorer is the right consumer — it can associate a
  per-slot control fact with the action that takes that card, which is better
  aligned than anything a pooled fraction supports.
