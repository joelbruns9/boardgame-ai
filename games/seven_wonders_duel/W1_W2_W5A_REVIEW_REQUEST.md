# W5a throughput, W1 slot identity, W2 tableau graph — review request

**Status: all three BUILT, none TRAINED, no strength evidence of any kind.**
W5a's arm existed before this batch and was repaired for throughput and bf16;
W1 and W2 are new. Every one of them is off by default, and nothing here has
played a game.

The design argument for each lives in `WORLD_CLASS_MODEL_EVOLUTION_PLAN.md`
(Workstreams 1, 2 and 5). This document is the brief: what changed, what is
already gated so a reviewer need not re-verify it by hand, and the specific
questions I could not settle myself.

---

## Scope

| commit | subject | what it covers |
|---|---|---|
| `64cee29` | W5a survives bf16, and stops paying for its candidate axis in Python | the two defects and their fixes |
| `159e812` | record W5a's measured throughput, and why the Rust port is not taken | plan text only |
| `fdbc507` | the tableau learns where it is, and what covers what | W1 and W2 |

| file | change |
|---|---|
| `dataset.py` | vectorised `legal_action_tensors`; new `legal_action_tensors_packed`; scalar version retained as `_legal_action_tensors_scalar` |
| `rust_bridge.py` | flat-batch path routed to the packed builder |
| `net.py` | W1 slot table and index reconstruction; W2 `TableauGraph` / `TableauGraphLayer`; bf16 cast on the W5a residual |
| `slot_identity.py` | **new.** Slot identity, and the static per-Age relation matrices |
| `train.py` | `slot_embedding` / `graph_module` switches; `graph_*` shape in the config; `NEUTRAL_ZERO_PARAMETERS`; graph params exempt from migration zeroing |
| `phase_d.py`, `phase_e.py`, `arena.py` | the switches carried through every rebuild and reporting site |
| `test_action_residual.py`, `test_slot_embedding.py`, `test_tableau_graph.py` | see below |

---

## Already gated — please do not re-verify by hand

| claim | how it is pinned |
|---|---|
| vectorised candidate axis == scalar, packed == generic | element for element and dtype for dtype, 651 real rows, six batch sizes, covering draft / play / pending choices / start-player |
| W1 switch-on is **bit-identical** on every head | `test_slot_embedding.py::test_switching_it_on_reproduces_the_inherited_model_exactly` |
| W2 at `graph_alpha=0` is bit-identical | `test_tableau_graph.py::test_a_zero_gate_is_exactly_inert` |
| W1 slot indices match the layout on a real game | asserted against `TABLEAU_LAYOUTS`, not against the encoder |
| W2 edges match the cover relation | asserted against `data.covering_slots`, the engine's own definition, per Age |
| W2 relation is mirrored (`covers` ⇄ `covered_by`, `descendant_k` ⇄ `ancestor_k`) | per Age, all pairs |
| a message really crosses a cover edge, and a non-edge sends a different one | `test_a_message_crosses_a_cover_edge_and_nothing_else` — the test that fails if the module silently does nothing |
| absent slots send no message | `test_absent_slots_send_nothing` |
| non-tableau tokens pass through untouched | `test_non_tableau_tokens_pass_through_untouched` |
| W1 and W2 are independently switchable | `test_w1_and_w2_are_independently_switchable` |
| fused inference path == per-type loop with both arms on | both test files |
| all three arms survive **bf16**, forward and backward | `test_the_new_arms_survive_bf16` |
| checkpoint round-trip rebuilds the same net | `test_the_switch_survives_a_checkpoint_round_trip` |

Full suite: **1467 passed, 5 skipped** (22:51, `-p no:randomly`).

### Measurements

W5a, search sims/s, 384×8:

| | OFF | ON | cost |
|---|---|---|---|
| CUDA bf16, before | 3,789 | 1,858 | **−51%** |
| CUDA bf16, after | 4,436–4,784 | 4,006–4,325 | **−9%** |
| CPU fp32, before | 529 | 413 | −22% |
| CPU fp32, after | 559 | 543 | **−3%** |

W1 / W2, fused inference forward, d384 L8 pooled, CPU fp32, best decile of 25
interleaved samples per arm:

| arm | 8 rows | 64 rows |
|---|---|---|
| W1 | 1.029× | 0.993× |
| W2 (2 layers) | 1.066× | 1.031× |
| W1 + W2 | 1.060× | 1.044× |
| W2 (3 layers) | 1.089× | 1.061× |

W1 alone, measured on its own against a same-seed baseline: 1.002× / 0.997×,
i.e. not distinguishable from zero at this precision.

---

## Decisions I made that a reviewer should second-guess

### 1. W1's index is derived, not encoded

`row` and `x` are already on the tableau token and the Age one-hot is on GLOBAL,
so `net.TokenEmbedder.slot_ids` reconstructs the slot index rather than the
encoder emitting a 38th `TABLEAU_FEATURES` column.

What that buys: `ENCODER_SIGNATURE` does not move, so every checkpoint and every
materialized buffer stays loadable; no matching change in `encoder.rs`; no
regenerated equivalence corpus.

What it costs: **`net.py` now reads feature columns.** Both lookups are by name,
so an appended feature cannot shift them, and only reordering an existing tuple
could — which the migration rules forbid for their own reasons. It also hard-
codes that **the GLOBAL token is at position 0**, which the encoder guarantees
and `SWDNet` already relied on for its readout.

**Is that trade right, or is the schema the honest place for this?** I think the
coupling is the cheaper of the two, but it is a coupling that did not exist
before, and a reviewer who disagrees should say so now rather than after a run.

### 2. W1 gets no warm-up gate, against the plan's general advice

The plan says exact-zero gates are for equivalence tests and training should
begin at a normalized small nonzero value. That reasoning is about *residual
modules*: at zero the module's own parameters get no gradient. A lookup table is
different — its gradient does not pass through its own value, so every row a
batch touches moves on step one. `test_a_zero_table_still_trains` asserts it.

**Is there a reason to want a gate anyway** — for instance to bound how fast
slot identity can perturb an inherited policy?

### 3. The `neutral` bucket, and a behaviour change to `phase_e`

`phase_e.load_evaluator` refused any migration that zeroed a parameter, on the
grounds the net is then partly random. That is exactly untrue of a zero-
initialized slot table, so `migrate_state_dict` now reports such parameters in a
separate `neutral` bucket and the guard ignores it.

**The part worth a second pair of eyes** is the reload condition beneath it.
Before: `if grown: warn; else: model.load_state_dict(raw)`. A migration whose
only entry was `initialized` — a new `control_scorer` or `action_scorer` — fell
into the `else` and re-loaded a state dict missing those keys, which a strict
load raises on. I widened the condition to skip the reload whenever anything was
grown, neutral, or initialized.

I believe that is a latent bug fixed rather than one introduced: the W3 and W5
arms reach `load_evaluator` with matching architectures, so they never took the
migration path at all. But **nothing tested it before and nothing tests it now**,
and I would rather a reviewer confirm the reasoning than trust mine.

### 4. W2's transitive edges run to distance 6, not the plan's 5

Age III is seven rows deep. Stopping at 5 would give the apex and the bottom row
the `none` relation — the one pair transitive edges exist for. The range is
derived from `TABLEAU_LAYOUTS` rather than written down, so it follows the
geometry if the geometry ever changes.

### 5. W2 uses basis decomposition, and 4 is a guess

Fifteen relation types with a full `d×d` transform each would be fifteen
projections per layer. `W_e = Σ_b a[e,b] V_b` with **four** bases keeps a real
per-relation transform at a quarter of the cost, and the algebra reorders so the
per-relation adjacencies never materialize:

    Σ_e W_e (A_e h) = Σ_b V_b (Σ_e a[e,b] A_e) h

`Σ_e a[e,b] A_e` is one embedding lookup over the static relation matrix.

**Four bases is an unmeasured choice.** With 15 relations it is a 3.75:1
compression, and nothing establishes that the relation set has only four
degrees of freedom. It is a flag (`--graph-bases`), so it is cheap to sweep —
but I have not swept it, and a reviewer may think the first arm should not be
the compressed one.

### 6. Messages are averaged over present slots, not per-relation degree

Per-relation normalization would make a node with one coverer and a node with
two send messages of the same size. "How exposed am I" is exactly that
distinction, so the divisor is the count of present slots.

**Retracted 2026-09-07, on review.** This section previously claimed the cost
was that "magnitudes shrink as the tableau empties". That is wrong, and wrong in
the direction that flattered the choice: the divisor is the number of present
slots, so as the tableau empties each surviving source gets a *larger*
coefficient. The resulting magnitude depends on which vectors and relations
survive and is not monotonic either way. The normalization is still a reasonable
first setting; the claimed behaviour was not a reason for it.

### 7. "No structural relation" is a learned edge type

The plan lists it as an edge, so it is one, which means every node can hear
every other through a shared learned coefficient — effectively a tableau-wide
mean-pool term. Pinning it to zero (`padding_idx=0` on the mix embedding) would
make the module strictly local at no cost. **I followed the plan; I am not sure
the plan is right here**, since the trunk already does global mixing and a
local module might be the cleaner ablation.

### 8. Edges are static; presence is not an edge

A slot two rows down is a distance-2 descendant whether or not the card between
them has been taken. Dynamics live in `accessible` / `coverers` and, exactly, in
W3. Absent slots are masked as message *sources*, so a taken card is not there
rather than being a zero-vector neighbour — but the *topology* never changes.

### 9. The Rust port of W5a's candidate axis: measured, declined

The packed path costs **1.90 ms per search's worth of rows (362), i.e. 5.3
µs/row, ~2.2% of an 85 ms search**. That is the whole addressable prize and Rust
would not recover all of it. Declined because it is a third copy of the
legality-to-token mapping, in a language whose codec constants are generated and
must stay in lockstep — a mapping that has produced bugs before.

**Revisit only if W5a ships and throughput becomes binding.** A reviewer who
thinks 2.2% is worth a third mapping should say so.

---

## What I could not settle, and would most like reviewed

1. **Whether W1 and W2 should share one training run.** The plan's bundling rule
   says yes unless a component has substantial throughput cost. W2 measures
   3–7% on CPU, the same order as the W5a scorer's accepted 9%. I read that as
   "not substantial" and bundled. It is a judgement call about a multi-hour run,
   and the person paying for it should agree with it.

2. **Whether the offline throughput numbers are the right ones.** All the W1/W2
   figures are a single fused forward on a laptop CPU. Generation is CPU per
   process, so I think this is the governing path — but no end-to-end self-play
   or arena throughput has been measured with either arm on, and W5a's history
   in this file is precisely a case where the forward was not where the cost was.

3. **Whether anything else consumes a batch and will now see new work.** The
   graph runs on every forward, including generation, and `slot_ids` does
   elementwise work that `TokenEmbedder.fuse()` does not cache. I traced the
   batch-consuming paths I know of; I would like that list checked rather than
   trusted.

4. **The W5a ambiguity sentinel.** The dense scatter packs entity ids at
   `type * 128 + entity` and knocks duplicates to a sentinel using a parallel
   count. Equivalence against the scalar reference is tested on 651 real rows,
   but real rows are what the game happens to produce; the reviewer should
   decide whether that corpus reaches every action family and every ambiguous-
   source shape, or whether a constructed adversarial case is needed.

---

## Known limitations

- **No training, no strength evidence, no arena** for any of the three.
- W1/W2 throughput is CPU fp32 only. GPU and bf16 throughput are unmeasured
  (bf16 *correctness* is gated).
- `graph_bases = 4` and `graph_layers = 2` are unswept defaults.
- The plan's "later option" for W2 — per-attention-head relation biases in the
  main Transformer — is not built and should not be until the module earns it.
- W5a still has no strength evidence either; the repairs were throughput and a
  crash, not a case for the arm.
- The pre-existing `train_loop` `optimizer_name` blocker recorded in the plan is
  still unfixed and still blocks an offline CLI training run.

---

## How to run the gates

```bash
python -m pytest games/seven_wonders_duel/test_slot_embedding.py \
                 games/seven_wonders_duel/test_tableau_graph.py \
                 games/seven_wonders_duel/test_action_residual.py -q
```

Full suite, ~23 minutes:

```bash
python -m pytest games/seven_wonders_duel -q -p no:randomly
```

Arms are `--slot-embedding`, `--graph-module` (with `--graph-layers`,
`--graph-bases`, `--graph-alpha`), and `--action-residual`, on both `train.py`
and `phase_d.py`. `--graph-alpha 0` is the W2 ablation and is exactly inert.

---

## Review outcome (2026-09-07)

Reviewed against `361a03f`. **Three P2 findings, all accepted and fixed**; all
seven sign-offs granted; one of my explanations retracted (§6 above).

| finding | disposition |
|---|---|
| warm start silently replaces the saved `graph_alpha` | **Fixed.** The three graph flags now default to `None`, so omitted inherits and explicit overrides — including an explicit `0`. A shape flag that disagrees with the inherited weights is refused rather than resolved in either direction. `train.build_arg_parser` / `train.resolve_graph_args` were extracted so this is testable without entering the trainer. |
| evaluator accepts randomly initialized graph weights | **Fixed, and my fix was aimed at the wrong boundary.** See below. |
| `search_gain_probe` cannot rebuild W1/W2 checkpoints | **Fixed, and five more like it.** |

**On the second finding, the reviewer was right and my reasoning in §3 was
wrong in a way worth stating plainly.** I generalized from training-time
migration, where adding a module is deliberate, to `load_evaluator`, where it
never is. That function rebuilds the model from *the checkpoint's own config*,
so a parameter it must invent can only mean the file does not carry the weights
its config declares. The distinction between `zeroed`, `neutral` and
`initialized` matters when training migrates into a new architecture; at the
inference boundary all three are the same fact, and `initialized` is the worst
of them because it serves plausible seed-dependent numbers instead of obvious
nonsense. `load_evaluator` now accepts only `grown` — the encoder-schema case it
was written for — and refuses the rest by name.

**On the third, the finding was narrower than the defect.** `search_gain_probe`
was one of six hand-rebuild sites on a documented debt list whose stated
justification was that a stale rebuild "fails loudly at the tool rather than
mid-run". W1/W2 showed what that is worth: the probe fails loudly, but
`w0_sizing_v2` names *no* architecture switch at all and would have silently
rebuilt a pooled/reply/W5 checkpoint as a plain net. Five modules are converted
to `model_from_config` / `phase_d._model_from_spec`; `w0_sizing` moved to the
allow-list because it builds fresh models from a sizing arm rather than from
saved weights. **`NOT_YET_CONVERTED` is now empty.**

Not adopted, because the review did not ask for it and it would change an arm:
`graph_bases` remains 4 and the `none` relation remains learned, per the
sign-offs. The reviewer's suggestion to describe the module as *relation-aware
mixing* rather than strictly local is adopted in wording.

The reviewer's own measurement is the better one for the cost question and
supersedes mine for the unfused path: **4.51% at 8 rows and 3.25% at 64**, on a
single-threaded unfused CPU forward, which is what `Evaluator` actually
constructs on CPU. My figures forced fusion and four threads.

## Sign-offs requested

1. The derived slot index (§1) — accept the coupling, or move it into the schema.
2. W1 with no warm-up gate (§2).
3. The `phase_e` reload-condition change (§3) — latent bug fixed, or new one.
4. Bundling W1 and W2 into one training run at W2's measured 3–7% (§ questions 1).
5. `graph_bases = 4` as the first arm's value, or sweep before the run (§5).
6. The "none" relation as a learned edge (§7) — keep, or pin to zero.
7. Declining the Rust port of the candidate axis at 2.2% (§9).
