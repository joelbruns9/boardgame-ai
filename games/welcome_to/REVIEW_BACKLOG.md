# Welcome To review backlog — 2026-08-28

This is the index of review coverage, not another design document. It was built
from the review records in this directory, the status notes in `RUST_PORT_PLAN.md`,
the Welcome To commit history, and the current working tree at `d8509b3` on
`welcome-to-engine`.

The rule used here is conservative: work counts as reviewed only when the
repository records an external review, its findings, and the disposition of
those findings. A green test suite or a reviewed plan is not treated as review
of a later implementation.

## Already reviewed

| Work | Evidence | Status |
|---|---|---|
| Search steps 1–3 | `SEARCH_STEPS_1_3_PROMPT.md`; external findings fixed in `7ffebe1` | reviewed and remediated |
| Search steps 4–7 | `SEARCH_STEPS_4_7_REVIEW_REQUEST.md`; response recorded in §7; fixes in `c840b56` | reviewed and remediated |
| Rust M3–M6 implementation | `RUST_M3_M6_REVIEW_REQUEST.md` §9; seven findings F1–F7 applied in the working tree | reviewed and remediated |
| S2 generation, inference, playout-cap runtime | `S2_GENERATION_REVIEW_REQUEST.md` §8; G1–G3 applied, G4–G8 reported | reviewed and remediated |
| S2 replay, loader, league, and promotion | `S2_REPLAY_PROMOTION_REVIEW_REQUEST.md` §8; R1–R9 and R12 applied, R10 deferred for measurement, R11 accepted | reviewed and remediated |
| Dense plan/end signal and schema migrations | `PLAN_SIGNAL_REVIEW_REQUEST.md` §9; Q1–Q3 applied, Q4 reporting improved and weight ablation deferred | reviewed and remediated |
| Rust port direction and M0 contracts | `RUST_PORT_PLAN.md` §0; review fixes in `5cb716f`; M0 sign-off in `68c0c48` | plan reviewed before implementation |
| Rust M1/M2 implementation | `RUST_PORT_PLAN.md` M1/M2; post-implementation omissions corrected in `03ae7cf` and the 8,000-game gate rerun | reviewed and signed off |

## Review requests now outstanding

None for the current Welcome To working-tree scope.

All Welcome To changes after `d8509b3` are covered by the completed reviews.
Files such as `self_play.py`, `s2_train.py`, `network.py`, and `samples.rs`
contain more than one concern, so each request names the symbols or sections it
owns. Reviewers should not infer scope from filenames alone.

**Post-review verification snapshot (2026-08-28):** `games/welcome_to/tests` is
**590 passed, 1 skipped, 1 warning**, and the Rust crate **26 passed**, with all
five M3–M6 equivalence gates green and the new ABI v3 postprocess gate green at
2, 3 and 4 seats. The earlier snapshot below predates that
review and its fixes.

**Pre-review snapshot (2026-08-27):** the union of the focused Python
suites named by the four requests is **138 passed, 1 warning** in 88.97 seconds;
the Rust crate is **24 passed**. The warning is a test-only tensor-to-float
conversion in `test_the_policy_loss_ignores_illegal_actions`, not a failed gate.

## Intentionally not claimed as reviewed

No current Welcome To implementation area is intentionally left unreviewed.

The 5090 sweep and a strength run are validation after review, not substitutes
for review. They remain out of scope for these documents.

## Suggested review sequence

M3–M6, generation, replay/promotion, and plan signal are reviewed and
remediated. The code-review backlog is clear for another continuation run.

⚠ **Current strength interpretation:** the S0 bootstrap was weak, but the later
continuation learner is not. Iteration 46 records score 53.28 and 0.690 plans per
seat-game versus GreedyBot's 50.84 and 0.406 on the same mixture. The remaining
gap is narrower and harder: it reliably completes one plan, sometimes two, and
recorded no three-plan games in the reviewed iteration. Slots 0 and 1, whose
estate plans need deliberate long-range fence structure, are the main headroom.

⚠ **F1 changes what a continuation run generates.** `derive_search_seed` no
longer collides across a contiguous seed block, so a rerun of the same
`--seed`/`--games` window produces different (and genuinely independent) search
tapes and root noise. Existing shards stay replayable — replay never re-derives a
search seed — but a run resumed by `skip_seeds` will not reproduce the tapes its
earlier half used.
