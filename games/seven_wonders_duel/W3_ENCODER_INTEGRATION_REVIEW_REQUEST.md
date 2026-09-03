# W3: wiring exact positional control into the encoder — review request

**Status:** solver built and verified; nothing wired. This document is the plan
for wiring it, and the questions I want attacked before any of it is built.

Prior review of the solver itself is in `W3_CONTROL_REVIEW_REQUEST.md`; its six
findings were accepted, and items 1–3 of the agreed remediation are committed
(`e017ab0`). This is the sequel: the reviewer's item 5, "training and inference
must use identical calculations if these become inputs."

---

## What is already true

`tableau_control.py` answers, exactly, *who reaches this tableau slot first*,
given the removal poset, turn alternation, and the shared seven-Wonder build
pool including retirement. It reads only public information — the present/absent
mask and slot geometry — so it is safe on a determinized state.

Correctness evidence, all currently passing:

| line of evidence | checks | result |
|---|---|---|
| brute-force oracle, 4–7 slot posets, all ages, pools 0–4 | 2,452 | agree |
| a structurally different oracle (iterative deepening over move sequences, not minimax) | 538 | agree |
| monotonicity properties, no oracle involved | 840 | hold |
| `tempo_state` invariant `unbuilt == builds_left + 1` on real BGA positions | 289 | 0 violations |

That last row matters more than it looks. `_spend` collapses every Wonder
counter to zero on the seventh build, while `engine.py:695` retires exactly one
Wonder. Those agree only because eight Wonders are drafted and seven get built.
That identity is a fact about the draft, not about this module, so it is checked
against real games rather than asserted.

**Not verified:** full 20-slot boards against an oracle (exponential — covered
only by properties and the reuse test), and the abstraction itself. The solver
models one Age, no coins, no production, no chains, no military, no early
victory. Those are documented limits, not measured ones.

---

## The cost question, measured

Solving at encode time is dead. `control_features()` costs **816 ms mean and
5.8 s worst** on real positions, against a leaf evaluation of order 1 ms.

But the key space is structural and small. The removal poset is far more
constrained than `2**20` suggests:

| quantity | value |
|---|---|
| reachable tableau masks (age 1 / 2 / 3) | 428 / 428 / 132 = **988** |
| tempo states, closed under the five feature calls | **444** |
| total keys, `(age, mask, who_moves, tempo)` | **877,344** |
| `control_map` mean / p90 / max over that space | 23 / 64 / 534 ms |
| **full precompute** | **5.5 core-hours — ~0.5 h on 12 cores** |
| **table size** at 20 bytes + header per entry | **~21 MB** |

So: precompute offline, ship the table, look up at encode time. The solver never
runs in the training loop, and does not need a Rust port. The Rust side needs to
*read* the table and *derive the same key* — a data-loading job, not a port.

The 444 tempo states are the closure of the 220 naturally occurring states under
the five counterfactuals `control_features()` queries. `_plus_extra` raises a
count above the four-per-player draft bound and `_minus_extra` breaks the
`builds_left + 1` invariant, so the closure is strictly larger than the set of
states a real game can reach. **The closure is exact for the current five
features and only for those.** A sixth counterfactual invalidates the table.

---

## The design

**Key.** `(age, present_mask, who_moves, tempo_state)`. Public structure only —
no card identities — which is what makes 988 masks cover every game ever played,
and what makes the table safe to use on the determinized states `advisor_scrape`
hands the searcher.

**Entry.** One byte per slot: attacker turns to reach it first, or an
unreachable marker. `control_map` currently discards the distance `solve()`
already computed; the table keeps it, at no extra solve cost.

**Per-slot features are primary.** `_tableau_tokens()` (`encoder.py:704`)
already emits one token per present slot, so control attaches as extra channels
on tokens that already exist — no new token type, no sequence-length change. The
whole point is to let the net combine *who reaches this slot first* with *what
card is sitting in it*. Global fractions are kept only as summaries on the
GLOBAL token.

**Five lookups per encode.** One on the live tempo state for the per-slot
channels; four on counterfactual keys for the global "what one more extra turn
would be worth" / "what Theology would be worth" fractions.

**Naming.** Following the prior review: these are named as reachability under
topology, never as control "as things stand", because affordability is
deliberately not modelled. The net already knows its own coins, production,
discounts and chains, and learns whether the build is payable.

---

## Sequencing

1. **Generator.** Enumerate keys, solve, write a packed table plus a manifest
   recording code commit, the closure definition, and a digest. ~30 min on 12
   cores.
2. **Python encoder lookup.** Per-slot channels primary, fractions on GLOBAL.
   Assert on a table miss; never silently fall back to a default.
3. **Cost and shape delta.** Measure the encoder's time and the net's input
   change before anything trains.
4. **Rust reader plus a Python/Rust key-parity gate.**
5. **Gate on the tactical corpus**, not on a self-play arena first.

---

## What I want reviewed

1. **Key parity is the whole risk.** `tempo_state` reads built Wonders, retired
   Wonders, Theology, and the `PLAY_AGAIN` effect list. Any divergence between
   the Python and Rust derivations silently desynchronises training from
   inference — the failure is invisible, not loud. Is a Welcome-To-style
   equivalence gate over a large corpus of real positions sufficient, or does
   the key need to be computed once in Rust and passed to Python?

2. **Unreachable needs its own channel.** `_INF` is 99; feeding that as a
   distance would swamp every other input. Proposal is a binary reachable flag
   plus a distance that is only meaningful when the flag is set. Is there a
   better encoding — for instance a saturating distance where unreachable is
   just the ceiling?

3. **The aggregation is not a plan, and the net may read it as one.** Each slot
   is solved under play optimal *for that slot*. The opponent racing you for
   slot A is not simultaneously racing you for slot B, so the 20 numbers are 20
   separate capability bounds, not a coherent line. Does attaching them per-slot
   make this better (the net sees each bound against its own card) or worse (the
   net learns to sum them)? Should the global fractions be dropped entirely?

4. **Table miss policy.** Assert-on-miss makes a closure bug a crash in
   training. Is that right, or should a miss fall back to solving live and log
   loudly? Live-solving a miss reintroduces the 5.8 s worst case into a
   self-play loop.

5. **Staleness.** The table is a function of the solver's code. If the solver
   changes, every table and every checkpoint trained on it is stale. The
   manifest records a commit, but nothing enforces it. What is the right
   mechanism — a digest checked at load, or a version baked into the encoder's
   feature-name list?

6. **Is the gate right?** The plan is to judge this on the tactical corpus:
   ~210 live actor-created-threat positions, measuring prior/rank on the exact
   refutation, fixed-budget regret, and calibration, with shuffled control
   features as a causal negative control. The objection is that the corpus is
   optimisable — wire W3 to win exactly these positions and the corpus stops
   being evidence. What is the honest holdout?

7. **The bigger question, which I do not want lost in the plumbing.** The
   plateau findings (`PLATEAU_FINDINGS.md`) leave two live hypotheses,
   generalisation-limited data and encoder/architecture, and specify a
   data-scaling curve to distinguish them. That curve has never been run. W3 is
   a bet on the encoder branch. Is building the table before running the curve
   the right order, or is it 5.5 core-hours and an encoder change spent on the
   branch we have not yet shown we are in?
