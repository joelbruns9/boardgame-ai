# Review request: engine pricing, the reveal port, and the certifier findings

Six commits from 2026-09-06, none reviewed by anyone but their author. Branch
`sevenwd-w9-prototype`, `fab62f9..b2928b8`. The suite is green (1,428 passed, 5
skipped) — which is exactly why this document exists: the tests passing is the
weakest of the claims below, and the interesting risk is in what no test looks
at.

Read §2 first. It is the one item with a live operational consequence.

---

## 1. What is in scope

| commit | subject | strongest evidence | weakest link |
|---|---|---|---|
| `fab62f9` | `phase_e.load_evaluator` rebuilds via `model_from_config`; `control_head` joins `ARCHITECTURE_SWITCHES`; `w3_corpus_regret` writes per arm | regression test loads a real aux checkpoint | no test covers the other six rebuild sites still listed as debt |
| `d36ef17` | `COVERING_SLOT_IDS`, `PricingContext`, Wonder pricing hoisted, two `minimum_payment` shortcuts | 1.83M prices and 8,607 action lists compared against the original body | **that comparison is not in the repo** — §4.1 |
| `c18f1e9` | `control_certify` + `fast_clone` (built earlier, committed here) | reference-case positives, `--loser 1` negatives, Age 1/2 non-firing | **zero unit tests for 413 lines of proof logic** |
| `4daa1c1` | reveal channels in Python | values hand-checked on Example A | committed deliberately red; message undercounts (says six, was eight) |
| `e661a0b` | reveal channels in Rust, `SWD_CONTROL_FEATURES` parity, re-pin to `7wd-encoder-7` | bit-exact cross-language agreement with channels live | **one game, one seed** — §4.3 |
| `b2928b8` | probe prints `limits_hit`; research doc §14 | — | doc claims, not code |

---

## 2. Every existing checkpoint now needs `migrate=True`

The encoder signature moved twice today (the widening, then the `-7` version
bump). `extension_7wd/candidate_0085.pt` — the shipped incumbent — is now
refused by any path that does not opt into migration:

    ValueError: checkpoint migration required - encoder signature changed since
    this model was trained (pass migrate=True for an additive warm start)

This was already true after `4daa1c1`; the re-pin makes it permanent and blessed
rather than work-in-progress. Migration itself is sound — the columns are
appended, so they zero-pad and the net computes what it always did — but the
callers are inconsistent about opting in:

* `arena.load_side` defaults `migrate=False` **by design** ("a migrated model is
  not the model that was trained"). Every arena against a pre-widening
  checkpoint now fails until `--migrate` is passed, and then measures the
  migration as much as the checkpoint.
* **`advisor_adapter.py:600` (`_victory_outlook`) calls `load_evaluator` without
  `migrate`, inside `try/except Exception: return None`.** With the shipped
  checkpoint the BGA panel now silently drops its victory outlook — no error, no
  warning, an absent field. The adapter's main evaluator path (line 449) honours
  `_allow_encoder_migration` and surfaces a warning; this second path does
  neither.
* A dozen offline tools (`f4_*`, `chance_cap_quality`, `phase_e` ground truth,
  `target_*`) load without migration and will now raise.

**Questions.** (a) Should `_victory_outlook` share the adapter's migration flag
and evaluator cache instead of loading its own? (b) Is a bare
`except Exception: return None` around a *load* acceptable, given it converts a
schema mismatch into a missing panel field? (c) Should the widening have waited
for a retrained checkpoint, given the advisor is a live consumer?

---

## 3. Assumptions a reviewer should not take on trust

### 3.1 The `minimum_payment` early return (`engine.py:268`)

I return as soon as a candidate buys nothing:

    if not purchased:
        return best

The argument: the comparison key is `(total_coins, purchased)`; with nothing
purchased `trade == 0`, so `total_coins == cost.coins`, and `()` is the smallest
possible `purchased`. **This is minimal only if a rebate can never reduce
printed coins.** I read `_discount_allocations` as allocating over resources
only, and Architecture/Masonry as resource discounts — but that is my reading of
the code and the rules, not a cited source. If any effect reduces the coin part
of a cost, this returns a non-minimal payment and every price is wrong in the
cheap direction.

### 3.2 The coin-only fast path (`engine.py:228`)

`cost.total_resources == 0` returns `Payment(cost.coins, cost.coins, 0, ())`
without consulting rebates or production. Same dependency as 3.1, plus the
assumption that no future effect makes a zero-resource cost discountable.

### 3.3 `PricingContext` is a snapshot with no invalidation (`engine.py:134`)

Built once per `legal_actions`, passed to every price in that call. Nothing
stops a future caller holding one across an `apply_action` and mispricing
silently — no generation counter, no debug assertion, no reference to the state
it was built from. I chose a docstring warning over a mechanism. **Is that the
right trade here?** The comparable failure mode — a stale control table — got a
hard digest check instead.

### 3.4 Action order is part of the codec

Hoisting Wonder pricing out of the slot loop preserves emission order: the
affordable Wonders are still appended per slot. I verified it by comparing 8,607
action tuples element-for-element, but the reasoning deserves an independent
read, because a reordering would silently rewrite action indices in every buffer
and checkpoint.

### 3.5 `COVERING_SLOT_IDS` lives in `game.py`, not `data.py`

It belongs beside the layouts. It cannot go there: `control_table.rule_identity`
hashes `data.py`'s **bytes**, so a comment in that file invalidates the W3
control table and every checkpoint recording its digest. The table is keyed by
`self.age`, matching the previous `TABLEAU_LAYOUTS[self.age]` lookup, so an
out-of-range age raises exactly as before.

### 3.6 Rust derives coverer geometry from the present list, not the layout

`reveal.rs::newly_revealed_counts` finds coverers by scanning present slots for
`row + 1 && |x - ox| == 1`. Python calls `covering_slots(layout, slot)` and then
filters to present ones. These agree **only because `covering_slots` is defined
as that same adjacency test**. If a layout ever encoded a non-adjacent covering
relation the two languages would diverge silently. Should Rust consult
`crate::data::layout` instead, at the cost of building the tuple?

### 3.7 `control.rs`'s flag is now lazily initialised

`ENABLED` went from `static AtomicBool = new(true)` to a `OnceLock` reading
`SWD_CONTROL_FEATURES` on first use. Consequence: **the environment is read once,
at whatever moment the first encode happens.** A test setting the variable after
that silently has no effect. This fixes a real disagreement (Python read the
variable, Rust did not) but trades it for an initialisation-order dependency.
`OnceLock`, or an explicit setter call from Python at import?

### 3.8 The transposition-table soundness argument

I claimed PROVEN at remaining-plies `p` is valid for any `p' >= p` while REFUTED
is valid only for `p' <= p`, and keyed the duplication measurement on
`(state, plies)` accordingly. That key also **excluded `rng` and
`search_barrier`**, on the argument that under the barrier every child is built
from explicit outcomes so the stream is never consumed. If that is wrong the
distinct-position counts are too low. The conclusion (a TT does not pay) only
strengthens under a coarser key, but the reasoning should be checked rather than
inherited.

---

## 4. What is not tested

### 4.1 The engine differentials live in a scratchpad, not the repo

The strongest evidence for `d36ef17` — 1,829,370 prices and 8,607 action lists
identical to the pre-optimisation body — came from two throwaway scripts and is
now unreproducible from the repository. The reference implementation they
compared against exists only in git history. **Recommendation: promote both as
tests**, with the old body vendored into the test file, so the next person to
touch pricing inherits the same guarantee rather than a claim in a commit
message.

### 4.2 `control_certify.py` has no unit tests at all

413 lines of three-valued AND/OR proof whose entire value is that PROVEN can be
trusted, pinned by nothing but a probe run by hand against one table. Missing at
minimum: a constructed forced win that must be PROVEN, a constructed escape that
must be REFUTED, an `age_deal` edge that must be UNKNOWN, and a test that
shrinking the budget never turns UNKNOWN into PROVEN. `fast_clone` has 11 tests
against `deepcopy`, but **nothing tests that `certify()` returns the same verdict
on a `fast_clone` tree as on a `deepcopy` one** — the property the whole
optimisation rests on.

### 4.3 The cross-language reveal gate is one game, one seed

`test_both_languages_agree_with_reveal_on` plays seed 707 with RNG 7071 and
compares every row; it exercised 70 nonzero tokens. Not covered: the 2x2 flag
matrix (control on/off × reveal on/off), multiple seeds, positions reached by
search rather than random play, and states with unusual pools (Mausoleum
revival, late Age III with a nearly empty pool). The pre-existing off-mode test
has the same single-game shape, so this is a house pattern rather than a new
lapse — but the reveal block reads the *unseen pool*, and random play may never
push it into its corners.

### 4.4 Reveal has never run through self-play

Every measurement is on `derive_records_rust`, the replay/training path.
`self_play_many_flat_net` — the actual consumer, and the whole reason the port
exists — has not been run with the channels on. Nothing verifies the flag is
even observed on that path.

### 4.5 The other six rebuild sites

`fab62f9` converted `phase_e` and removed it from `NOT_YET_CONVERTED`.
`ablate_value_head`, `search_gain_probe`, `value_ceiling_probe`, `w0_sizing`,
`w0_sizing_v2` and `weight_decay_probe` still enumerate architecture switches by
hand and would fail on a `control_head` checkpoint exactly as `phase_e` did. The
debt test keeps them enumerated; it does not make them work.

---

## 5. Throughput: what was fixed, what was left, what got slower

### 5.1 Left on the table: the encoder prices the whole pool, twice, per encode

`encoder.py:424` (`_Derived.min_costs`) calls `minimum_payment` for **every
unseen pool card, for both seats, on every encode** — up to ~100 calls, each
rebuilding the four city scans `PricingContext` now exists to share. `d36ef17`
fixed that pattern in `legal_actions` and did not touch the encoder. The same
holds in Rust: `engine.rs::minimum_payment` rebuilds its context per call and
`encoder.rs` calls it per slot per seat, on the self-play hot path.

**This is probably the largest remaining win in the codebase, and it was not
taken.** Porting `PricingContext` into the Rust encoder would speed up every
leaf of every search; the certifier work only sped up an offline tool.

### 5.2 Two contexts per node, not one

`apply_action` validates with `action not in legal_actions(game)`, so every
applied action builds a second `PricingContext` for the same state. Removing
that re-check is worth ~40% of the certifier's remaining runtime and is
deliberately not done: it trades an engine-wide guard for speed. **Should
`apply_action` take a `validated=False` opt-in for callers that decoded from
`legal_action_indices` on the same state?**

### 5.3 Reveal costs 8% of the derive path

0.359 → 0.391 s cpu for 2,868 rows, best of 15, channels on. Encode-only; the
network forward should dominate in self-play, but §4.4 — that has not been
measured. The block is O(unseen pool) per encode, so it is most expensive early
in a game, when the pool is largest.

### 5.4 The certifier cannot go in search, and Rust would not fix it

Row 85 forks 90 ways per action against a 10-card pool; row 86 forks once. A
20-50× port against a 90× per-ply fan-out buys less than one extra ply.
`BOARD_CONTROL_RESEARCH_REQUEST.md` §14 carries the numbers. Symmetry reduction
over revealed cards is the only route that attacks the real constraint, and it
must buy depth as well as width.

---

## 6. Record-keeping defects I already know about

* `4daa1c1`'s message says "six tests red" and names two files. It was eight,
  across three (`test_f4_boundary` was hidden behind pytest's `-x`). Not amended
  because it is no longer the branch tip.
* `data.py` is LF in the working tree while `core.autocrlf=true` would write
  CRLF. Any `git checkout` or `stash pop` touching it changes `rule_identity` and
  invalidates the control table **with no source change at all**. It happened
  once during this session. A `data.py text eol=lf` line in `.gitattributes`
  would close it; not done, because it is a repo-wide config change.
* `rule_identity` hashing raw bytes rather than normalised content is the
  underlying flaw. Fixing it moves the digest, so it needs a table regeneration
  to go with it.

---

## 7. The questions I most want answered

1. §3.1 — can any effect reduce the **coin** part of a cost? Both
   `minimum_payment` shortcuts depend on "no".
2. §2 — is the advisor's silent `_victory_outlook` degradation acceptable until
   a retrained checkpoint exists, or should it migrate-and-warn like the main
   path?
3. §4.1 — should the pricing differentials become permanent tests, with the old
   implementation vendored as the reference?
4. §4.2 — what is the minimum test set that would make PROVEN trustworthy enough
   to gate training on?
5. §5.1 — any reason not to port `PricingContext` into the Rust encoder, given
   it is the hot path and the pattern is now proven in Python?
6. §3.7 — `OnceLock` environment read, or an explicit setter call from Python at
   import?
