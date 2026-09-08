# W4, its two strength arms, and the searched-outlook backup — review request

**Status: all BUILT, none TRAINED, no strength evidence of any kind.** Every
piece is off by default. Nothing here has played a rated game or trained a
checkpoint.

Covers everything since the last review (`361a03f`), which is five commits: the
fixes made in response to that review, then Workstream 4 in full. The design
argument lives in `WORLD_CLASS_MODEL_EVOLUTION_PLAN.md` (Workstream 4 and its
*Distributional MCTS backup* section); this is the brief.

The previous brief is `W1_W2_W5A_REVIEW_REQUEST.md`, whose *Review outcome*
section records what was conceded and fixed. **Its findings 1–3 are implemented
here and have not themselves been reviewed** — see §1 below.

---

## Scope

| commit | subject |
|---|---|
| `5a4f4ac` | three review findings, and the debt list they came from |
| `30302cc` | W4 says one thing about the winner, and says it off to the side |
| `0ad748b` | let W4's read leave the model, alongside the flat one |
| `078eef5` | give W4 two arms that can actually move strength |
| `fc9ca09` | back up the seven-way outlook, and record it where a target can use it |

| file | change |
|---|---|
| `net.py` | `HierarchicalValue`; `_check_joint7_layout` |
| `train.py` | `hierarchical_value` / `_detach` switches, `hier_value_weight`, `hier_value_replaces_joint7`, the NLL term, `build_arg_parser` / `resolve_graph_args` extraction |
| `inference.py` | `Evaluation.hier_joint7` / `hier_wdl`; `Evaluator.value_source` + `wdl_tensor` |
| `advisor_adapter.py` | `victory_outlook["hierarchical"]` + `flat_disagreement` |
| `rust_bridge.py` | outlook on the flat batch path; routed proxies refuse a mixed `value_source` |
| `buffer.py`, `dataset.py` | `root_outlook` on `MoveRecord` and `Example` |
| `phase_d.py`, `arena.py` | switches carried through every rebuild, spec and reporting site |
| **Rust** `eval.rs` | `LeafOut`, `Outlook`, `outlook_to_p0`, `terminal_outlook_p0`, `extract_rows`, `adapter_outlook` |
| **Rust** `tree.rs` | `descend` returns the leaf outlook; root accumulation; `SearchResult.root_outlook` |
| **Rust** `tree_resumable.rs` | same accumulation in the session the streaming advisor uses |
| **Rust** `self_play.rs`, `lib.rs` | `root_outlook` on the recorded row and the Python dict |
| `search_gain_probe.py`, `w0_sizing_v2.py`, `ablate_value_head.py`, `value_ceiling_probe.py`, `weight_decay_probe.py` | converted to `model_from_config` / `_model_from_spec` |

---

## Already gated — please do not re-verify by hand

| claim | how it is pinned |
|---|---|
| the served heads are **bit-identical** with W4 present | `test_hierarchical_value.py::test_the_served_heads_are_untouched` |
| a detached head cannot move **any** trunk weight | gradients asserted to reach only `hier_value.*` |
| the attached arm really does reach the trunk | asserted separately, so the opt-in is not the safe arm renamed |
| the 7-way distribution and its W/D/L marginal are one object | exact to 1e-6, all four classes |
| the flat heads **do** disagree | the defect, shown rather than described |
| reordering `JOINT7_CLASSES` is refused at import | `_check_joint7_layout` |
| the head trains, and its loss descends | gradient plus a 20-step fit |
| `Evaluation` reports `None` for a model without the head | so "no head" is distinguishable from "head agrees" |
| `exp`, never a second softmax, at both boundaries | sums to 1; marginal matches |
| replacement drops the term but still **reports** it | `test_replacing_joint7_drops_its_term_but_still_reports_it` |
| replacement with a detached head is refused | it would delete supervision, not vary it |
| `value_source` moves `wdl` and leaves `joint7` alone | one variable per arm |
| routed evaluators refuse a mixed `value_source` | a stitched batch cannot read a different head per row |
| a net with no head records no outlook | the short adapter row stays legal |
| the searched outlook is a distribution over 7 classes | sums to 1, non-negative |
| its marginal equals search's value **under the same head** | exact to 2e-6, and asserted to differ under mixed heads |
| selection is unchanged by carrying the vector | same action, same visits, same root value |
| `root_outlook` survives buffer → `Example` | round trip, including `None` rows |
| a buffer written before W4 still loads | `.get`, not `[...]` |

Full suite: **1495 passed, 5 skipped** (22:51, `-p no:randomly`). Rust: `cargo
check --release` clean; extension rebuilt with `maturin develop --release`.

### Measurements

Search throughput, `search_many_flat_net`, d384 L8 pooled, CPU fp32, 1 thread,
4 games × 128 sims, median of 7:

| arm | seconds | ratio |
|---|---|---|
| baseline, no head | 4.749 | 1.000 |
| W4 head + outlook backup | 4.743 | **0.999** |
| W4 + backup + `--value-source hierarchical` | 4.784 | **1.007** |

Forward-only, unfused, 1 thread, d384 L8: W4 head **1.004×** for 3,465
parameters.

---

## Decisions a reviewer should second-guess

### 1. The previous review's fixes, which have not themselves been reviewed

Three P2 findings were accepted and implemented in `5a4f4ac`:

- **`graph_alpha` on warm start.** The three graph flags now default to `None`;
  omitted inherits, explicit overrides including `0`. A shape flag that
  disagrees with inherited weights is **refused** rather than resolved either
  way — that half was mine, not the reviewer's, and is the one to check.
- **`load_evaluator` refusing invented parameters.** I went further than the
  finding asked: because that function rebuilds from *the checkpoint's own
  config*, a parameter it must invent can only mean an incomplete file, so it
  now accepts **only** `grown` (the encoder-schema case) and refuses `zeroed`,
  `neutral` and `initialized` alike. That is stronger than "reject initialized
  graph parameters when their gate is active" and rests on the
  rebuild-from-own-config property. **If that property does not hold somewhere
  I have not looked, this refuses a legitimate load.**
- **The hand-rebuild debt list is now EMPTY.** Five modules converted;
  `w0_sizing` moved to the allow-list because it builds fresh models from a
  sizing arm. Nothing exercises those five tools in CI beyond import.

### 2. Detach is the default, and the default is inert

The head cannot change a served number. Detached it cannot change a trunk weight
either, so its whole cost is 0.4% of a forward. That follows a standing
instruction — the incumbent is the top BGA arena player, so "might help, might
hurt" is not the same bet as "cannot hurt" — but it is worth stating plainly
that **the safe default is also the useless one**: a detached head gives a
consistent distribution to read and no representation benefit.

### 3. Replacement rather than addition is how the parameterisation gets tested

`joint7` and W4 fit the same per-game label. Running both trains two heads on one
observation and mostly re-weights the outcome objective against policy — that
measures the weight, not the structure. `--hier-value-replaces-joint7` drops the
flat term so the comparison holds information and weight fixed.

Consequence worth checking: under that arm the flat `joint7` head is frozen
wherever it was inherited and **its outputs go stale**, while the advisor still
reads them. It is recorded in the checkpoint config; nothing enforces that a
reader honours it.

### 4. The linearity finding is now load-bearing

From the previous review, and correct: `P(win) − P(loss)` is linear in the seven
probabilities, so averaging the vector then collapsing is identical to
collapsing at each leaf then averaging. **The backup therefore cannot change
move selection at all** — not by policy, by arithmetic. This is why the backup
is justified as a data capability and why `--value-source` exists separately as
the arm that *can* move play.

The unexplored corner: a **non-linear** selection rule (penalising lines
carrying military-loss mass, say) would extract something the scalar cannot
represent. Nothing here does that.

### 5. Root-only accumulation, against the plan's per-node design

The plan says "store per-edge probability sums alongside scalar value and
visits". Nothing reads an interior node's outlook — the advisor shows root
moves, every training row is a search root — so the vector is passed up
untouched and summed at the root only. No node grew a field, no serialised tree
changed shape, the resumable tree kept its footprint.

**The cost of this choice is that per-move outlooks do not exist.** The advisor
can show "search thinks this position is 30% science" but not "this move raises
their military win by 12 points", which is arguably the more useful sentence and
is what the plan's per-edge design would have bought.

### 6. Terminals contribute exact victory type

A finished game knows who won and how, so those leaves carry ground truth. This
is the answer to "searched targets are only estimates" for the proven part of
the tree. Mirrors `dataset._joint7_class`, including treating a **shared**
civilian finish as a draw rather than a civilian win.

### 7. The two arms are coupled

The scalar backup averages whichever head `value_source` names; the outlook
averages W4's. Under `flat` they are two different predictions, so search's Q
and its seven-way split need not agree. **A coherent searched panel needs
`--value-source hierarchical`.** Found by asserting agreement and watching it
fail; pinned in both directions.

### 8. The adapter contract accepts two shapes

Rust tries a 3-tuple row and falls back to the 2-tuple, so every existing Python
adapter and stub works untouched. An outlook that does not sum to 1 (±1e-3) is
**refused**, because a caller that sent logits would otherwise poison every sum
with plausible-looking numbers.

### 9. `LeafOut` changed the `Eval` trait — the widest blast radius here

40 compile errors across `eval.rs`, `tree.rs`, `tree_resumable.rs`,
`self_play.rs`, `lib.rs`, `bots.rs`, `solver.rs`. The compiler enumerated them,
so none can have been silently missed, but this touches the resumable tree the
live advisor runs on. **The equivalence corpus was not regenerated** — see
limitations.

---

## What I could not settle

1. **Whether the searched target should blend or replace.** The local precedent
   is split: `root_value` **blends** with the outcome (because cloud3 produced a
   confidently wrong head from a hard fit), while `solver_value` **replaces**
   outright because it is proof. A searched outlook is neither — estimate at net
   leaves, proof at terminals. It may deserve a per-row weight rather than one
   global blend.

2. **The `immediate_value` path silently contributes no outlook** unless the
   node is terminal. Those simulations back up a value but do not increment
   `outlook_visits`, so the outlook mean is over a *subset* of the simulations
   the value mean covers. That is why the two agree only when both come from the
   same head, and it is the assumption I am least sure of. Worth checking
   whether a non-terminal `immediate_value` leaf can occur often enough to bias
   the mean.

3. **Whether accumulating once per simulation is the right weighting** in the
   resumable tree, where a wave dedups leaves. I mirrored exactly what the
   scalar does, on the principle that two means over the same leaves should be
   weighted identically — but "mirrors the scalar" is an argument from symmetry,
   not from what the target should be.

4. **Whether root-only will hold.** If per-move outlooks are wanted later, the
   accumulation has to move down a level. Cheap to add at the root edges;
   genuinely expensive at every node.

---

## Known limitations

- **No training, no strength evidence, no arena** for anything here.
- **No loss consumes `root_outlook`.** That is the offline half, deliberately
  deferred so it can be tuned against real recorded targets.
- The **equivalence corpus was not regenerated** after the `Eval` trait change.
  The Rust/Python engine-parity tests pass, but the corpus is a separate artifact
  and was already stale per `sevenwd_military_offbyone`.
- Throughput is **CPU fp32 single-thread only**. No GPU, no bf16, no end-to-end
  self-play or arena measurement with any arm on.
- `--value-source hierarchical` is measured for **cost**, never for strength; a
  worse-calibrated marginal would make search worse and nothing here would say
  so.
- The extension panel still renders the flat outlook. The head is untrained, so
  displaying it today would render noise.
- The pre-existing `train_loop` `optimizer_name` defect still blocks the offline
  CLI trainer.

---

## How to run the gates

```bash
python -m pytest games/seven_wonders_duel/test_hierarchical_value.py \
                 games/seven_wonders_duel/test_root_outlook.py \
                 games/seven_wonders_duel/test_tableau_graph.py -q
```

Full suite (~23 min): `python -m pytest games/seven_wonders_duel -q -p no:randomly`

Rust: `cd seven_wonders_rust && cargo check --release && maturin develop --release`

Arms: `--hierarchical-value`, `--no-hierarchical-value-detach`,
`--hier-value-replaces-joint7`, `--hier-value-weight`, `--value-source
{flat,hierarchical}`, on both `train.py` and `phase_d.py`.

---

## Sign-offs requested

1. The previous review's three fixes (§1), especially the widened
   `load_evaluator` refusal and the now-empty debt list.
2. Detach as the default, knowing it makes the default arm inert (§2).
3. Replacement as the way to test the parameterisation, and the stale flat head
   it leaves behind (§3).
4. Root-only accumulation, accepting that per-move outlooks do not exist (§5).
5. The `immediate_value` subset question (§ could-not-settle 2) — this is the
   one I would most like a second opinion on.
6. Blend versus replace versus per-row weight for the eventual searched target
   (§ could-not-settle 1).
7. Whether the equivalence corpus must be regenerated before the run, given the
   `Eval` trait change.
