# Implementation prompt — Fix denial-search reuse/throughput (root-search + node-TT + measurement)

Context: `games/kingdomino/denial_search.py` and its validation report
`runs/kingdomino/denial_search/validation.json`. Read
`games/kingdomino/NNUE_DENIAL_SEARCH_BUILD_PROMPT.md` for the search's contract, and
[[kingdomino_advisor_throughput_review]] for the known "3× redundant solves / no TT reuse"
history. This is a **throughput / reuse-correctness** task only. It must **NOT** change any emitted
label, policy target, value, or fragility — see the invariant gate below. It does **NOT** retrain
and does **NOT** change the search budget (the separate signal-at-higher-sims re-run is a different
task).

## The problem (already diagnosed — do not re-litigate)

The report shows `leaf_tt_hits: 0`, `leaf_tt_misses: 195424`. This is **expected, not a bug**: the
leaf value cache is keyed on `public_state_key`, which hashes the full board
(`_board_bytes`, `denial_search.py:157`). Every 8-ply horizon leaf is reached by a distinct
pick+placement path, so every leaf board — and therefore every key — is unique. The leaf value
cache **cannot** hit by construction and delivers no benefit. Do not try to "make it hit"; the
redundancy is elsewhere:

1. **`_root_search` runs 3× per validation position on the identical root state** — once for the
   8-ply label (`search_position`, ~`denial_search.py:562`), once for the 4-ply structural ablation
   (the second `search_position` call in `run_validation`), and once inside
   `run_advisor_draft_matrix_baseline` (`denial_search.py:743`). The Rust open-loop MCTS is
   config-independent; recomputing it three times is pure waste and is likely the dominant cost at
   ~36 s/position.
2. **The node TT (`_node_tt`) shares nothing between the 8-ply and 4-ply passes.** `_node_key`
   (`denial_search.py:428`) embeds `pick_plies`, `placement_top_k`, `chance_k`, `seed`, so the
   4-ply ablation — whose first four plies are structurally identical to the 8-ply tree — gets
   entirely fresh keys and re-expands + re-evaluates all of it.
3. **Reuse is unmeasured.** `EvalStats` tracks `policy_cache_hits/misses` and a `_node_tt` exists,
   but the report only emits leaf-cache stats (`denial_search.py:987`). Interior/policy/node reuse
   — the reuse that actually recurs — is invisible, which is why "0 leaf hits" was misread as a
   broken TT.
4. **`public_state_key` is expensive and recomputed 2–3× per leaf** (`denial_search.py:285, 301`,
   then again in `_backup:523`). blake2b over all board arrays plus `repr` of a large tuple, run
   far more often than necessary.

## Required fixes

Implement all four. Keep each behavior-preserving (see invariant gate).

1. **Cache the root search per position.** Add a `public_state_key(state) -> root_result` cache on
   `DenialSearch` (or memoize `_root_search`) so the 8-ply pass, the 4-ply ablation, and the
   advisor baseline reuse ONE root search per distinct root state. The root search does not depend
   on `pick_plies`/`chance_k`/etc., so this is safe. **Watch the seed:** `_root_search` currently
   derives its seed from `self._position_serial`, which increments every `search_position` call, so
   today's three calls already use different seeds. Fix the cache to return the *first* computed
   result for a given root state deterministically (and keep the seed a pure function of the root
   state / a fixed base, not of call order) so results are reproducible and the label pass is
   unaffected. Do NOT cache the trajectory-generation root searches in
   `generate_az_midgame_positions` across different states — only dedupe identical root states.

2. **Let interior structure reuse span passes where it is genuinely identical.** The 4-ply ablation
   and the 8-ply tree share their first four plies. Either (a) run the 4-ply ablation by truncating
   / reading off the already-built 8-ply tree instead of re-searching, or (b) if you keep separate
   searches, remove the config fields from `_node_key` that do not change a node's *value semantics*
   at a given (state, depth, crossings, root_actor) and instead guard reuse by the fields that
   actually do. **Be careful:** `placement_top_k` and `chance_k` DO change a subtree's value, so a
   node computed under one cannot be reused under another — do not collapse those unsoundly. Prefer
   option (a) (derive the 4-ply result from the 8-ply tree) as it is provably identical; only take
   (b) if (a) is infeasible, and prove value-equivalence.

3. **Instrument every reuse channel in the report.** Add to the `throughput` block:
   `policy_tt_hits/misses`, `node_tt_hits/misses` (count `_get_node` returning an existing node vs
   creating one), `root_search_calls` and `root_search_cache_hits`, and keep the (now correctly
   understood) leaf figures. State in the report that leaf-cache reuse is structurally ~0 by design
   and is not the throughput lever, so the number is not misread again.

4. **Memoize `public_state_key`.** Compute it once per `GameState` (e.g. cache on the object or via
   an identity-keyed memo) so `values_p0`/`_backup`/node construction stop recomputing the same
   blake2b+repr. Ensure the memo is invalidated correctly if a state is mutated in place (the search
   uses `.copy()`/`.step()` to produce fresh states, so per-object caching should be safe — verify).

## Invariant gate (this is the whole point — do not skip)

The change is throughput-only. **Emitted labels must be identical before and after**, at fixed
seed/config. Before touching anything, capture a baseline on a small deterministic set
(e.g. `--positions 8 --search-sims 32 --seed 20260716`), save the report, then after the change
re-run with the same args and assert **byte-for-byte-equal** `labels`, `four_ply_labels`, and
`one_round_labels` (compare the parsed JSON for these keys; `throughput` and timing fields are
expected to differ). If any label changes, a "reuse" you introduced was unsound — revert it. Report
this diff check explicitly.

## Validate

- Re-run the full validation (`--positions 50`, same budget as the original report:
  `--search-sims 32 --chance-k 4 --draft-search-sims 50 --draft-budget-seconds 2.0`).
- Report **positions/hour before vs after** and the new reuse stats (root-search cache hits should
  be ~2/3 of root searches eliminated; node-TT hits should be substantial if pass-sharing landed).
- Confirm the invariant gate passed (labels unchanged).
- Update the projection to 10k positions and state whether throughput is now acceptable for
  curriculum scale or whether further work (e.g. larger leaf batching, multi-GPU, or a coarser
  placement delegation) is still needed.

## Tests

Extend `games/kingdomino/tests/test_denial_search.py`:
- Root-search cache returns the identical result for repeated calls on the same root state, and the
  label output is unchanged with the cache on vs off.
- Node-TT / pass-sharing produces byte-identical `four_ply_labels` whether the 4-ply result is
  derived from the 8-ply tree or computed independently (if option (a) is taken).
- `public_state_key` memoization returns values equal to the un-memoized function on a spread of
  states (including post-`step`, post-`copy`, and pre-reveal leaf states).

## Scope / do NOT

- Do not change the search budget, `pick_plies`, the horizon convention, the policy-target math, or
  the label schema (beyond the additive throughput-stat fields).
- Do not retrain, build the training mixture, or run the control/treatment experiment.
- Do not attempt to make the leaf value cache "hit" — it is structurally unable to and that is fine.
