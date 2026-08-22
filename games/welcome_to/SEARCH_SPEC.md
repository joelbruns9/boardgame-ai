# Welcome To… — search specification

**Status:** the within-turn tree is **implemented** (`mcts.py`, `macro_codec.py`)
and described here as built. The chance boundary is **proposed, not built** — §6
onward is for review before implementation.

**Reads with:** `SELF_PLAY_PLAN.md` S1 (the root-player contract and the gate),
`ENCODER_V2_SPEC.md` §10.6 (the frozen 684 action vocabulary),
`AUX_TARGETS_SPEC.md` §5 (the leaf value).

**Reference implementation for §6:** `games/seven_wonders_duel/search.py` already
does fixed-support chance edges, has been reviewed twice, and documents the
invariant that a naive version violates. Port it; do not redesign it.

---

## 1. What this document decides

Two things, and they are independent:

1. **Within a turn** — how the root player's own sequence of decisions is
   represented in the tree. Settled; §3–§5.
2. **At the turn boundary** — how the card reveal is represented. Open; §6–§9.

The reason they are independent is the single most useful structural fact about
this game:

> **No chance occurs inside a turn.** Every transition from the start of a
> player's turn to the end of it is deterministic given their actions.

That makes the within-turn subtree exactly reproducible across simulations, so
its nodes take repeat visits and its values back up through exact deterministic
transitions. Everything difficult is at the boundary.

---

## 2. The engine's phase graph — ground truth

Transcribed from `game.py`; this is what the search must respect, not a model of
it. Twelve phases (`Phase`), mirroring `states.inc.php`.

```
                    ┌───────────────────────────────────────┐
                    │                                       │
                    ▼                                       │
             CHOOSE_CARDS ──ROUNDABOUT_OPEN──▶ ROUNDABOUT_PLACE
                    │                                       │
                    │ choose_stack(s)              (place or pass)
                    ▼                                       │
             WRITE_NUMBER ◀────────────────────────────────-┘
                    │
      ┌─────────────┼──────────────┬─────────────┬──────────────┐
      │ SURVEYOR    │ ESTATE       │ PARK        │ POOL     BIS │  TEMP
      ▼             ▼              ▼             ▼          ▼   │
 ACTION_SURVEYOR  ACTION_ESTATE  ACTION_PARK  ACTION_POOL  ACTION_BIS
      │             │              │             │          │   │
      └─────────────┴──────────────┴─────────────┴──────────┴───┘
                                   │
                                   ▼
                            CHOOSE_PLAN ◀──────────────┐
                                   │                   │
                    ┌──────────────┼───────────┐       │
                    │ pass         │ automatic │ estate│
                    ▼              ▼           ▼       │
            _finish_player_turn  ASK_RESHUFFLE  VALIDATE_PLAN
                    │              │                   │
                    │              └───────────────────┘
                    ▼
            next seat, or the TURN BOUNDARY
```

Facts that are easy to get wrong and are load-bearing:

- **`ASK_RESHUFFLE` returns to `CHOOSE_PLAN`, not to the turn end.** A player may
  validate more than one City Plan in a turn, so the plan/validate/reshuffle
  section is a **loop**, not a chain.
- **A permit refusal reaches `CHOOSE_PLAN` from two different places** — from
  `CHOOSE_CARDS` when nothing is playable at all, and from `WRITE_NUMBER` when
  the taken combination's *printed* number has nowhere to go. Both skip every
  effect phase.
- **`ROUNDABOUT_PLACE` returns to `CHOOSE_CARDS`**, so a roundabout turn visits
  `CHOOSE_CARDS` twice. It terminates because building sets `ctx.last_house` and
  passing sets `ctx.roundabout_declined`, and both guard the offer.
- **`_settle()` resolves four transitions without asking**: `ACTION_PARK` with no
  available street, `ACTION_POOL` when unavailable, `CHOOSE_PLAN` with nothing
  scorable, and `ASK_RESHUFFLE` when the reshuffle is not on offer. **These are
  not decisions and must never become nodes.** The engine already collapses them;
  the search inherits that for free.

---

## 3. The macro boundary — what is collapsed, and why

`macro_codec.py`, implementing `ENCODER_V2_SPEC.md` §10.6. **One** split is
collapsed: `CHOOSE_CARDS → WRITE_NUMBER`.

The reason is not throughput. **You pick a combination *for* a placement**, so
evaluating the two steps separately prices "take stack 2" by the *mean* quality
of the placements it leads to, when the player gets to choose the placement and
it is therefore worth what its *best* placement is worth. A sequential
decomposition answers a question nobody asked.

The temp-agency delta belongs inside the macro for a concrete reason: it changes
**placement legality**, so it cannot be deferred to a later node without making
the write node's legal set depend on a decision that has not been taken.

```
macro write     495   (stack, temp delta, box) = 3 × 5 × 33   at CHOOSE_CARDS
macro refuse      3   (stack, PERMIT_REFUSAL)                 at CHOOSE_CARDS
direct refuse     1   nothing playable at all                 at CHOOSE_CARDS
roundabout open   1                                           at CHOOSE_CARDS
primitives      184   the 357-slot codec minus the four above at their phases
                ---
                684
```

**Measured:** the collapse removes **28% of network calls** — 8,808 primitive
decisions become 6,360 macro ones over 30 three-seat GreedyBot games. There are
**no `WRITE_NUMBER` evaluations at all**, which is where the branching lived
(13.1 mean, 165 max).

**Legality is enumerated end to end, never intersected.** `legal_macros` steps
into each playable stack and reads that child's own `legal_actions()`, because
`WRITE` legality depends on which stack was taken — intersecting per-step masks
would admit jointly-illegal pairs. This also keeps the engine the sole authority
on the rules. Verified: 193,400 legal macros applied across 6,360 macro roots
without one illegal step.

---

## 4. The within-turn node table — as implemented

One phase is active at a time. The network emits 684 logits and the legal mask
exposes only that phase's actions.

| node | vocabulary | width | notes |
|---|---|---|---|
| roundabout declaration | `ROUNDABOUT_OPEN` vs a card action | — | not a separate node; it is one entry in the `CHOOSE_CARDS` set |
| roundabout placement | box, or pass | 34 | `ROUNDABOUT_PLACE` |
| **combination + write** | (stack, temp delta, box) | **495** | the macro; heavily masked |
| refusal | stack-specific, or direct | 4 | 3 macro + 1 direct |
| surveyor | fence, or pass | 31 | 30 fence slots |
| estate | row, or pass | 7 | 6 value rows |
| park | street, or pass | 4 | normally 1 street + pass |
| pool | build, or pass | 2 | |
| bis | (box, side), or pass | 67 | 33 boxes × 2 sides |
| city plan | slot, or pass | 4 | loops back after reshuffle |
| plan estate validation | matching estate | ≤ 33 | sequential, only for `EstatePlan` |
| reshuffle | yes / no | 2 | |

**Measured cost:** a turn costs **1.91 network decisions on average** —
distribution 2 in 169 turns, 1 in 62, 7–8 in 8, over 239 sampled turns of
two-seat advanced play. Most turns are the macro plus at most one effect node.
(Measured under a fixed "first legal macro" policy; the exact mean will shift
with a trained policy, the shape will not.)

---

## 5. Why the turn is not bundled further

Rejected. Two independent reasons, both measured.

**It is too large to index.** A complete-turn action — roundabout, write,
effect, plan, validation, reshuffle as one choice — has, over 239 sampled turns:

| | sequences per turn |
|---|---|
| mean | 6,367 |
| median | 1,470 |
| max | **71,334** |
| turns over 10,000 | 48 / 239 |

A flat head cannot index that, and a fixed vocabulary would have to cover the
union rather than the realised set. It would need a factored or autoregressive
decoder — a large build.

**And it buys nothing.** Because no chance occurs inside a turn, the within-turn
chain is **already fully revisitable**: the same keys recur in every simulation
and the chain is developed to its natural length. To the search it already
behaves like one optimised contingent plan, without materialising every plan as
a root edge. Bundling would save **0.91 network calls per turn** and cost the
conditional sharing that lets a fence-placement policy be learned across every
write that leads to the same surveyor state.

### 5.1 Sub-decisions that look forced but are kept

An argument was made for auto-applying park, pool and estate on the grounds that
declining is never right. **The game reasoning appears sound** — all three only
advance a scoring track:

```
ACTION_ESTATE : sheet.estate_marks[row] += 1
ACTION_PARK   : sheet.parks[x] += 1
ACTION_POOL   : sheet.pools[street] += 1
```

No box is consumed, no number written, no fence placed, and the City Plans that
touch parks and pools (`DECORATIVE`, `COMPLETE_STREET`) all want *more* of them.
Whereas **bis** calls `sheet.write(..., is_bis=True)` — it fills a box and takes
the penalty — and the **surveyor fence** partitions a street into estates and can
destroy an `EstatePlan`'s required sizes. So bis and fence are genuine decisions
and park/pool/estate look forced.

**They are kept anyway**, for three reasons:

1. **BGA fidelity.** The engine is transcribed from BGA's PHP, and both the
   differential harness and the advisor depend on speaking BGA's action space.
   Removing a legal option makes replays and advice diverge from the real site.
2. **It is a hand-coded valuation** — "always take the park" is a human
   judgement, the category-2 line `ENCODER_V2_SPEC.md` §1 draws. Very probably
   correct; if it is wrong anywhere, the model can never discover it.
3. **The saving is small.** These are 2-option nodes that PUCT resolves in about
   two simulations.

⚠ **Correction to the framing:** estate is **not** a yes/no. `estate_rows()`
offers a choice of *which* value row to advance, and the rows score differently.
Auto-applying would remove the pass and leave a node behind. Park is normally
binary (one street — the one just written in) and pool is binary; only those two
would disappear.

**The safe version of the same win** is a search-level change with no rules
change: skip forced one-action nodes **inside simulations**, not only at the
external root. `MCTS.play` already skips them at the real root; `_simulate` does
not. Before doing even that, measure what share of the 1.91 calls per turn are
park/pool nodes — it sizes the prize.

---

## 6. The chance boundary — PROPOSED

Everything from here is for review.

### 6.1 What is built today, and why it is wrong

`MCTS._advance` samples opponents forward to the root player's next decision and
returns `tuple(state.table_cards(root))`. A child is keyed on
`(action, observation)`.

This is correct — it never merges distinct observable states — and it **has no
depth**. Measured mean leaf depth **1.59**, and it does not respond to the
budget or to prior sharpness:

| prior | mean leaf depth | range |
|---|---|---|
| uniform | 1.59 | 1.06 – 2.08 |
| one-hot | 1.59 | 1.00 – 2.01 |

(4 network seeds × 6 positions, 256 simulations, 2/24 cells differing by >0.5.
One-hot is the *upper bound* on sharpness, so this closes the question: no
trained policy deepens it.)

The cause is that every boundary crossing draws a key never seen before, so it
always expands a fresh leaf. The budget past roughly one simulation per root
action goes into **root averaging over fresh leaves**, not depth.

⚠ **The key is also under-specified, latently.** It is raw card **IDs**, and it
carries neither the opponents' now-public sheets nor the race state. Measured, it
costs nothing today — 0 spurious splits in 60 samples — because near-unique
reveals mask it. **15 of the 66 printed card types have two physical copies**, so
identical-looking reveals key to different children the moment children are
reused. Fixing the key is a prerequisite for §7, not an optional cleanup.

### 6.2 What a reveal actually is

Not an ordered triple of numbers. A construction card prints its own effect on
its number face, so a card drawn at a boundary supplies a **number for this turn**
and an **effect for the following turn**. Measured, one turn ahead over 60
determinizations of one action sequence:

| | distinct |
|---|---|
| effects in play this turn | **1** — certain, and equal to `next_effects` |
| effects for the following turn | **46** |
| numbers in play this turn | 60 |

So a reveal is an ordered triple of printed **card types**. There are **66
distinct printed types** in an 81-card deck (multiplicity 1–2), giving on the
order of 277,000 valid ordered type triples. **Exhaustive expansion is not
available** and no design should assume it.

---

## 7. Proposed: a fixed-support chance node at the boundary

### 7.1 Shape

When every action of the turn has resolved, but **before** `_draw_step()`,
create a **pre-reveal afterstate** node. At it:

- sample `K` reveals from the **exact remaining card histogram** (public
  bookkeeping — `deck_knowledge.deck_composition`);
- materialise those `K` children and **retain them for the life of the search**;
- **re-normalise their probabilities to sum to 1** and **close the edge against
  growth**;
- the node's value is the **probability-weighted** mean of its children's
  backed-up values — a chance node averages, it does **not** PUCT;
- later simulations descending the edge **sample among the retained children by
  their re-normalised weight**.

Retained children are revisited **by construction**. That is the one thing
neither exact observation keying nor prior sharpening can produce, and it is the
entire point.

Start with `K ∈ {4, 8, 16}` and choose empirically.

### 7.2 The invariant that a naive implementation violates

Taken directly from `seven_wonders_duel/search.py`, which documents it because it
was got wrong once:

> An approximate edge **MUST** be closed. Ordinary descent samples from the
> COMPLETE chance distribution and appends any outcome it cannot find, carrying
> that outcome's original probability. On a truncated edge that would push the
> mass above 1, and `q` would then return a weighted sum over more than unit mass
> **with nothing raising**.

So three edge classes must stay distinct, and the code must not let them blur:

| class | support | later descent |
|---|---|---|
| `probability_weighted` | exhaustive, exact | always finds a child |
| `fixed_support` | retained subset, re-normalised | samples **only** among them |
| ordinary sampled | grows lazily | may materialise a child |

`close_fixed_support()` refuses to close unless the retained weights sum to 1.
Port that check; it is the difference between a correct expectation and a
plausible-looking wrong number.

### 7.3 A scenario is deck order **and** opponent randomness

⚠ **The ambiguity most likely to be implemented wrongly.**

Opponent actions are **sampled**, not chance, and the root-player contract
deliberately keeps them out of the node key so a node averages over the opponent
model. But if a chance child is *retained* while opponents are *resampled*, the
child's state is not well defined — its statistics accumulate over a mixture it
does not represent.

**Therefore a retained scenario `ω` is the deck order and the opponent
randomness together**, and children are indexed by `ω`. Retained deck with
resampled opponents is the failure mode: it produces a tree that looks healthy
and whose statistics are incoherent.

### 7.4 Common random scenarios across candidate actions

The reveal distribution does not depend on which house the root player wrote —
the draw is from a shared deck — so candidate root actions should be evaluated
against the **same** sampled scenarios `ω₁…ω_K`. This cancels most of the
variance in their *differences*, which is what selection actually needs.

The exception is the **reshuffle** decision, which does change the distribution.
There, reuse the same underlying uniforms mapped through the state-specific deck
distribution rather than sharing the outcomes directly.

Opponent samples can be shared for the same reason: opponents cannot see the root
player's concurrent choice, so their play is independent of it.

### 7.5 Re-rooting on the real reveal

When the real game reveals its cards: construct the new public state, reuse the
matching sampled child if there is one, otherwise start a fresh root. Nothing
here needs a persistent ordinal identity — the actual public observation is the
anchor.

Within a turn, the deterministic subtree **can** be re-rooted after each real
action, because no chance intervenes. **That reasoning does not extend across a
boundary** and must not be generalised.

---

## 8. Rejected alternatives, and why

| design | verdict |
|---|---|
| **Bundle the whole turn** | Rejected — §5. Up to 71,334 sequences per turn, and it saves 0.91 network calls. |
| **Sort the stacks** (KD's `sorted(deck[:4])`) | Rejected. KD's sort is meaningful because domino number *is* next-round pick order, so the slot encodes tempo whatever tile it carries. Sorting three Welcome To stacks gives "lowest of three random draws" — 1 in one determinization, 13 in another. No stable quantity. |
| **Merge on `(number, effect, box)` + ISMCTS availability counts** | Demoted to an experimental arm. It is invariant and availability counts are well founded, but it is **context-abstracted**: one `Q` averaged over what the other two offers are, what next-turn effects were exposed, and the opponents' race state. It is also lossy — a 7 printed and a 7 made with TEMP consume different cards. |
| **Progressive widening on the chance branch** | Not rejected, deferred. It was withdrawn once for a bad reason: "sharper priors do not deepen the tree" is measured and true, and does **not** imply nothing structural does. Fixed-`K` first because it is easier to reason about and to batch; PW only after that baseline exists. |

---

## 9. The trade this makes, and how to catch it going wrong

⚠ **Fixed-`K` sparse sampling is a step backwards at depth 1.**

The version built today draws a fresh reveal every simulation, so the root's `Q`
is an average over ~`N` distinct futures — **unbiased, high `N`**. Fixed-`K`
replaces that with an average over `K` fixed futures: lower variance, but
**biased by that particular sample**, and every simulation past the `K`th reuses
it.

So sparse sampling is not strictly more accurate. **It trades a worse depth-1
expectation for having a depth-2 at all.** That is very likely the right trade,
and it will not show up in a strength number alone: a depth-1 regression can hide
inside a depth-2 gain.

**Measure root value bias directly.** Late-game states with a small remaining
deck are the oracle — enumerate every legal ordered reveal exactly, and compare
each approximation's root action ranking *and* its root value against the exact
expectation.

---

## 10. Bakeoff

At equal wall-clock **and** equal network-evaluation budget:

| arm | chance handling |
|---|---|
| current | fresh exact observation every simulation |
| semantic ISMCTS | merged `(number, effect, box)` + availability counts |
| sparse chance | retained `K = 4 / 8 / 16` |
| sparse chance + afterstate head | learned chance expectation |

Report, for each: paired score gap and stderr over 300+ games; strength at
64/128/256 evaluations; mean turn boundaries crossed; selected-action agreement
across independent search RNGs; root value stderr and bias against the late-game
oracle; evaluations and wall time per move; plans, permits and end reasons.

⚠ **The gate needs ~300 games, not 60.** Measured: the per-game paired delta has
a standard deviation of about **18 points**, so stderr is 4.1 at 60 games and 1.0
at 330. A 4-point difference read off 60 games sits inside its own noise.

⚠ **All of this waits on S0.** Every arm needs a trained network; on an untrained
one both arms produce noise. Two claims in `SELF_PLAY_PLAN.md` were wrong because
they were measured on an untrained net.

---

## 11. Later: an afterstate value head

Only if sparse chance wins. Train `V_after(x) ≈ E_c[V_post(x, c)]` against an
average over several independently sampled immediate reveals, so a turn's search
can stop at the pre-reveal afterstate for one network call.

This preserves **recourse** — `V_post` assumes the future policy observes the
reveal before acting, so it is not perfect-information Monte Carlo over a fixed
deck. It is Stochastic MuZero's afterstate factorisation, worth borrowing without
the rest of it because the chance distribution here is known exactly.

It is a real build: new head, new target, several reveals sampled per training
position. Gate it on the bakeoff.

---

## 12. Implementation order

1. **Fix the observation key** (§6.1) — printed faces not card IDs, and include
   the opponents' public sheets and race state. Prerequisite for anything that
   reuses children.
2. **Skip forced one-action nodes inside simulations** (§5.1). No rules change,
   no structural change.
3. **Fixed-support chance edge** (§7), behind a flag, with the current version
   kept as the control arm. Port the invariants from
   `seven_wonders_duel/search.py`; do not re-derive them.
4. **Run S0.** Nothing below this line means anything without a trained net.
5. **Bakeoff** (§10), including the late-game exact oracle.
6. **Afterstate head** (§11), only if 3 wins.

Steps 1 and 2 are small and independent of the outcome of the review. Step 3 is
the substantive one and is what this document exists to have reviewed.
