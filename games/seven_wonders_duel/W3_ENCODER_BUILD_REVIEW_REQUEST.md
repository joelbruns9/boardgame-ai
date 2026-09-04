# W3 built into the encoder, both languages — review request

**Status: BUILT, not strength tested.** Exact positional control is now an
encoder input in Python and Rust, so it reaches self-play leaves, the advisor's
searcher, and the replay path. No arm has run; there is no strength evidence of
any kind.

The plan and its three prior reviews are in
`W3_ENCODER_INTEGRATION_REVIEW_REQUEST.md` (revision 4). This document covers
only what was built afterwards, and asks for review of it.

**Deviation from the agreed plan, deliberate and directed.** Revision 4 sequenced
the auxiliary head first and put encoder inputs behind evidence from it. The user
directed building inputs now, on the grounds that control is wanted in the
advisor and self-play regardless of what the auxiliary arm shows. That is a
legitimate second goal, but it means **the cheap experiment no longer gates the
expensive integration**, and this document should be read knowing that.

---

## What was built

| piece | where |
|---|---|
| six control channels per tableau token | `encoder.py` `_tableau_tokens`, `encoder.rs` `tableau_tokens` |
| `control_valid` on GLOBAL | both encoders |
| key derived from the observation | `control_table.control_key_from_observation`, `control.rs` `control_key_word` |
| table handed to Rust as bytes | `control_table.table_blob` → `set_control_table` |
| install at every Rust entry point | `derive_records_rust`, flat batch adapters, advisor searcher |
| auxiliary control head (separate arm) | `net.ControlHead`, loss in `train.compute_losses` |
| control key on the Rust replay wire | `derive.rs`, one `u64` per row |

Schema movement: `ENCODER_VERSION` `7wd-encoder-5` → `-6`, `MAX_FEATURES`
132 → 133, TABLEAU 26 → 32, signature `24e15b12…` → `36b2ffa5…`.

### Feature definition

Three maps — live, my-Theology, their-Theology — each contributing a
reachability flag and a scaled turn count. Only these three earn channels, under
the rule from revision 3: *a counterfactual earns a channel only when the
contingency lies beyond what search will expand.* Spending a Wonder is one legal
action away, so search expands it; Theology may be many plies off and swings up
to ±18 slots.

`UNREACH` is 254 in the table and never reaches the network as a distance: an
unreachable slot is `(0, 0)`. `control_valid` is 0 outside a clean `PLAY_AGE`
state, and then every control channel on every tableau token is 0.

---

## Evidence

| check | result |
|---|---|
| per-slot channels vs the live solver | 2,729 checked, **0 mismatches** |
| Python vs Rust full feature vectors | 205 rows, max abs diff **0.000e+00** |
| Python vs Rust control keys | 184 labelled rows, **0 disagreements**, applicability agrees |
| control-table entries vs the live solver | sampled across ages/masks/tempo, 0 mismatches |
| encode throughput cost | **+1.1% best, +0.4% median** (16,410 → 16,233 rows/s) |
| checkpoint migration | old columns preserved bit-exactly, new columns zero |

Throughput was measured on a quiet machine after a first attempt produced a
`without` build that was *slower* than `with` — a stale test run was competing.
The bad numbers looked plausible, which is the point of recording this.

### Two bugs the build found

**Viewer vs actor.** `encode` is viewer-independent: the same public position
observed from either seat must produce identical tokens. Control was keyed off
`obs.viewer` in Python and `active_player` in Rust. The parity test passed anyway,
because the replay path always encodes for the actor — the divergence was latent,
not absent, and `test_encoding_is_pure_under_hidden_reassignment` found it. Both
sides now key off the actor.

**Out-of-enumeration tempo is not a table gap.** A test hands a seat a fifth
Wonder, producing nine unbuilt, which no legal game reaches. Raising there
conflated "this position is illegal" with "the table is missing". An
out-of-enumeration key is now masked in both languages; an absent table still
panics.

---

## What I want reviewed

1. **The panic on a missing table.** Rust panics rather than emitting zeros,
   because zeros read as "the opponent reaches every slot first" and are
   indistinguishable from a genuine all-unreachable position. Three Python entry
   points install the table, and a pytest fixture installs it for tests that call
   `seven_wonders_rust` directly. **Is the entry-point list complete?** Anything
   that reaches `encode_into` without one of those three will crash at its first
   encode. That is the intended failure, but I would rather it be unreachable.

2. **Handing the table over as bytes.** Rust does not read the artifact; Python
   passes `table_blob()` at install. This makes "both readers agree" true by
   construction, and it means the 8.7 MB is resident per process — relevant if
   self-play forks many workers. Is there a case for a memory-mapped file
   instead, accepting a second loader?

3. **Regenerated goldens.** Four pins moved: the signature and three encoding
   digests. Regenerating a golden is how an unintended encoder change hides. The
   evidence above is recorded in `test_encoder.py` beside the pin, but a reviewer
   should decide whether it is sufficient, and whether the encoding delta is
   *only* the control channels.

4. **The applicability contract, now that it is load-bearing in two languages.**
   `control_valid` is 0 on 16.7% of states. Both encoders must agree on exactly
   which. The end-to-end gate covers real games; it does not cover
   `CHOOSE_NEXT_START_PLAYER`, `WONDER_DRAFT` or pending-choice states
   systematically, because those are masked and therefore compare equal for the
   wrong reason. **A test that both sides mask the same states for the same
   reason is missing.**

5. **Throughput, measured on the wrong path.** +1% was measured on the replay
   derive. Search leaves share `encode_into`, so the per-encode cost is the same,
   but self-play end-to-end was not benchmarked and the advisor was not
   benchmarked at all.

6. **The sequencing consequence.** With inputs built, the four-arm design from
   revision 4 (baseline / auxiliary-only / inputs-only / both) is still the right
   experiment, but the "inputs" arm is no longer cheap to abandon. Is the arm
   design still right, or does having paid the integration cost argue for a
   different first experiment?

---

## Not done

- No strength evidence, and no arm has run.
- `install_rust_table` is called at three entry points; there is no test that the
  set is complete.
- The advisor has not been exercised end to end with control features.
- Self-play has not been run at all with the new width.
- Every existing checkpoint is stale by signature. Migration is verified to
  preserve old columns and zero the new ones, so a migrated model computes what
  it always did — but nothing has been trained.
