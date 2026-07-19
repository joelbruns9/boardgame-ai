Implement the Kingdomino run7 changes and bank run6's best checkpoint. Work
autonomously; pre-registered decision rules are below so nothing waits on me.
Staying at 80x6 (no capacity change) — the goal is to squeeze the last bit of
strength out of the current net, which is already ~top-20 on BGA.

CONTEXT (state as of 2026-07-08, see also project memory):
- runs/kingdomino/cloud_80x6_run6/ = 86-iteration run: warm-started from
  current_best, soft_gate, HOF diversity (hof_fraction 0.2), sims 4800.
- Diagnosis of run6: (1) promotion NEVER passed — current_best (the run5 net)
  stayed frozen the whole run, so the learner chased an unbeatable target and
  drifted (paired-gate WR 0.52 -> 0.45). (2) HOF is contaminating training:
  ~21% of buffer examples are HOF frontier-seat moves searched at only
  hof_sims=400 (both seats flat), which blurs the policy. The value head is
  fully converged (own/opp loss 0.004); the regressing signal is policy.
- HOF's intended role (per hof.py docstring) is a *diversity opponent* that
  steers the frontier into novel positions; the frontier should learn there
  from its OWN deep search — NOT from shallow 400-sim labels.

Three fixes: (A) bank run6's best into current_best via a high-power gate;
(B) make HOF record deep targets with asymmetric sims; (C) make the promotion
gate a ratchet (promote on a confident LCB, not a 0.53 win-rate wall).

ITEMS (in order):

1. BANK RUN6'S BEST (do this first — it sets run7's warm start).
   Candidates = the three run6 checkpoints with the strongest paired-gate
   signal: iter_0020, iter_0025, iter_0040 (WR 0.52/0.52/0.50, LCB
   0.49/0.49/0.47). Incumbent = runs/kingdomino/best_checkpoint/current_best.pt.
   Evaluate each candidate vs current_best with the NEW high-power settings:
     ~2500 paired seat-swapped games @ sims=300, via
     promotion.evaluate_checkpoint_match (model the runner on
     runs/kingdomino/ablation_h2h.py — detached Start-Process + Monitor;
     background Bash is killed at 10 min).
   Pre-registered decision (candidate = the run6 checkpoint):
     - Promote the candidate with the highest LCB that clears **LCB > 0.51**
       vs current_best. Copy it to runs/kingdomino/best_checkpoint/
       current_best.pt AND add the OLD current_best to the HOF pool
       (add_hof_entry, tag pre_run7). This net becomes run7's warm start.
     - If NONE clears LCB > 0.51: keep current_best; run7 warm-starts from the
       unchanged current_best. Report this as "no bankable gain in run6."

2. HOF ASYMMETRIC DEEP TARGETS (self_play.py).
   Goal: HOF games keep steering the frontier into diverse positions, but the
   frontier seat searches at FULL strength with playout-cap (exactly like
   normal self-play), while the cheap HOF opponent seat stays shallow. Only
   frontier-seat moves are recorded (already the case).
   - In the batched HOF path (play_hof_games_batched / _run_hof_orientation,
     ~L1273-1360): frontier seat searches at n_simulations (4800) with
     playout_cap_randomization + full_search_fraction ON (record only the
     full-search moves, drop fast moves, same as normal self-play); HOF
     opponent seat searches at hof_sims (keep ~400). This likely needs per-seat
     sim control in the two-net batched engine (kingdomino_rust/src/lib.rs) —
     assess feasibility; if the batched engine can't do per-seat sims +
     one-sided playout-cap cleanly, say so and implement the cheapest correct
     alternative rather than guessing.
   - FIX THE PRUNING BUG this exposes: the HOF prune call (~L3247) passes
     total_visits=hof_sims. Once frontier moves are searched at 4800, pass the
     frontier seat's actual sim count, not hof_sims, or targets prune wrong.
   - Pool weighting: prefer near-peer opponents (contested games carry the
     useful diversity; blowouts don't). Either add a competitiveness-weighted
     mode to hof_sample_weights / sample_hof_entry, or curate the run7 pool to
     drop the weakest opponent. Keep it simple; don't over-engineer.

3. PROMOTION RATCHET (launch-script flags only — decide_promotion already
   uses strict LCB comparison, so this is config, not code):
     --promotion_min_win_rate 0.51   (was 0.53)
     --promotion_min_lcb 0.51        (was 0.50 — promote iff confidently >51%)
     --promotion_games 2500          (was 1032 — tighter LCB)
     --promotion_sims 300            (was 100 — closer to real play strength)
   Keep the fixed-suite guard and soft_gate revert unchanged. NOTE the honest
   caveat in the run7 report: this gate change unlocks the ratchet so real
   micro-gains get banked — it does NOT by itself create strength.

4. RUN7 LAUNCH SCRIPT (runs/kingdomino/run7_launch.sh, based on
   run6_launch.sh with the deltas above plus):
     - --warm_start <the Item-1 winner> (new or unchanged current_best).
     - NO --warm_buffer: start the buffer FRESH. Do not import run6's buffer —
       it holds exactly the 400-sim-HOF / drifted-learner data we're removing.
     - --seed 7. Keep sims 4800, playout-cap, lr 1e-4, hof_fraction 0.2,
       everything else as run6 unless a fix above changes it.

5. VERIFY before the full launch: run a tiny job (2-3 iters, few games) and
   confirm from the log/examples that (a) HOF frontier moves are searched at
   ~4800 (not 400), (b) only frontier-seat moves are recorded, (c) prune
   total_visits matches the frontier sim count, (d) the new gate flags are
   in effect. Then start the real run7 detached.

6. REPORT: write runs/kingdomino/run7_prep_report.md — Item-1 H2H table
   (WR/LCB/games per candidate + which was promoted or why none was), the HOF
   change + verification evidence, and the run7 launch command. Commit and push
   code, launch script, and report. NEVER commit checkpoints/buffers (.pt/.pkl
   stay out of git).

CONSTRAINTS:
- All matches: paired seat-swapped via promotion.evaluate_checkpoint_match.
- Long local jobs: detached Start-Process + Monitor, not background Bash.
- Do not touch capacity (stay 80x6). Do not cold-start — run7 warm-starts.
