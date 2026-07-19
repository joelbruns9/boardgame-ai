# Implementation prompt — Offline 8-ply pick-denial search + label emitter (build & validate)

Context: `games/kingdomino/NNUE_PROJECT_PLAN.md`, section "Committed direction (2026-07-16): AZ
pick-denial curriculum". Read it first. This builds the **generation engine** for that direction
and validates it on a small set. It does **NOT** retrain anything — the control/treatment
curriculum experiment is the next step after this.

## Goal

An offline search that, at a start-of-round midgame position, finds **denial-corrected pick
policy/value targets** to distill into AlphaZero — so the deployed net eventually plays denial
with no extra search. This extends the mechanic already in the advisor's `_draft_matrix`
(`web_app.py:911`) from current-round to an **8-ply / 2-round / one-chance-node** horizon, and
emits training labels instead of a human-facing table.

## Why this shape (do not re-scope)

- **2p boards are disjoint — placements never interact; ALL interaction is pick/turn-order**
  (advisor docstring). So "delegate placement to AZ, search picks only" is EXACT, not an
  approximation. Do not branch on placement.
- **8 pick-plies = ~2 rounds = crossing exactly ONE chance node** (the next-round draw). This is
  the tractable sweet spot (~2k leaves at k=8, AZ leaf eval affordable). Do NOT attempt
  12-ply/3-chance-node (infeasible with AZ eval, and the extra denial signal is hidden behind
  future draws anyway).
- **The blindspot is prior starvation → the primary target is POLICY, not value.** Forced
  (rooted) exploration of every opponent pick is what defeats starvation; keep that.

## Build

Create an offline module (e.g. `games/kingdomino/denial_search.py`) — request-independent, not
bound to the FastAPI advisor. Reuse the advisor's proven pieces: `_rust_open_loop_search`,
`_opponent_policy_priors`, `_pick_key_of`, `action_to_json`, and the group/representative logic in
`_draft_matrix`. The search must:

1. **Your decision layer:** at the root (you to claim), branch over your pick options (grouped by
   pick, representative = most-visited), descending your own consecutive moves as `_draft_matrix`
   already does (guard the king-order run).
2. **Opponent decision layer (forced exploration):** at the opponent node, branch over ALL their
   pick options — each a rooted mini-search with full budget (rooting defeats starvation).
   Opponent placement delegated to AZ (top-2 prior, take their better — as the advisor does).
3. **Chance node (the new part):** at the round boundary, sample `k` next-round draws from the
   SORTED remaining bag (order-blind), with **common random numbers** reused across sibling pick
   branches. Recurse into the next round's pick layer. Average children (expectiminimax).
4. **Leaf:** AZ value head at the 8-ply horizon (player-0 frame). Enumerate chance exactly when
   `C(remaining,4) <= k` (late/midgame) instead of sampling.
5. **Emit per position:** the per-pick searched values → a **denial-corrected POLICY target**
   over picks (documented temperature/tie/uncertainty handling), a corrected **value**, the
   `fragility = headline - robust`, and provenance (AZ checkpoint sha, sims/budget, `k`,
   enumerated-vs-sampled, completed structure, actor frame, official-cascade version).

### Throughput (the main risk — design for it now)

The advisor spends ~20s per position; a curriculum needs thousands. Build for batch from the
start: **batch the AZ leaf evaluations** (evaluate many expectiminimax leaves in one GPU call, as
self-play does) and **reuse a transposition table** across the tree and across positions where
public state repeats (see [[kingdomino_advisor_throughput_review]]: redundant solves / no TT reuse
were the known problems). Report positions/hour and leaf-eval batch sizes.

## Validate (small set — this is the deliverable, not a retrain)

Run on ~50-100 real **start-of-round midgame** positions sampled from AZ trajectories in the target
band (roughly the mid rounds up to just before the exact-solver frontier; do not sample the
opening or the exact-solvable tail). Produce a report
(`runs/kingdomino/denial_search/validation.json`) establishing:

- **Labels are sane:** policy targets are valid distributions; corrected value in range; no illegal
  picks; forced/sole-pick positions handled; GAME_OVER / near-frontier handled.
- **Denial is actually found:** on positions with high fragility, the corrected policy up-weights a
  pick that AZ's raw prior starved. Report the fragility distribution and how many positions have a
  *material* correction (corrected-best pick ≠ AZ headline pick, by a real margin).
- **Turn-order + chance-crossing are correct:** a couple of hand-checked cases where the denial
  value comes specifically from the next-round turn-order consequence (not just the visible tile),
  confirming the extra round does something.
- **Incremental value over the existing 1-round draft matrix:** run the current advisor
  `_draft_matrix` (current-round only) on the same positions and compare. **Quantify how much the
  8-ply / cross-chance search adds over the 1-round analysis.** If it adds little, that is an
  important finding (the cheaper existing miner may suffice); if it adds a lot, the 8-ply build is
  justified. Report this explicitly.
- **Throughput:** positions/hour at the chosen budget/`k`, and a projection to curriculum scale.

## Invariants / scope

- Order-blind chance (sorted bag, CRN); disjoint-board placement delegation; enumerate late chance
  exactly. Reserved test split stays closed. Use the current-best AZ checkpoint; record its sha.
- Do **NOT** retrain, build the training mixture, or run the control/treatment experiment here.
  Emit labels + the validation report only.
- Add focused tests (extend `games/kingdomino/tests/`): CRN determinism (same samples across
  sibling branches), exact-vs-sampled chance agreement when `C(n,4) <= k`, policy-target validity,
  and forced-pick handling.

## What this routes to (state in the report; don't act on it here)

- Labels sane + material denial found + meaningful gain over the 1-round matrix → proceed to the
  small **control vs treatment curriculum retrain** (equal compute), pre-registered success =
  fragility drop on a frozen held-out set + head-to-head strength.
- Little gain over the 1-round matrix → reconsider: use the cheaper current-round miner, or
  investigate why the extra round is inert, before building the retrain.
