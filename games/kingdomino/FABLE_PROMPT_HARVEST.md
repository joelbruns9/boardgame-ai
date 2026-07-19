Execute the Kingdomino noise-ball-harvest experiments. Work autonomously;
pre-registered decision rules are below so no judgment calls need to wait for
me. EVERYTHING RUNS LOCALLY on this laptop — do NOT ssh to the cloud box (a
snapshot of run5 has already been downloaded; the box keeps training
undisturbed).

CONTEXT (state as of 2026-07-07, see also project memory):
- runs/kingdomino/cloud_80x6_run5/ holds a LOCAL SNAPSHOT of the in-progress
  cloud run: checkpoints iter_0001..iter_0096 plus nohup.log (full per-
  iteration diagnostics and promotion lines through ~iteration 96; there is
  no training_log.jsonl in the snapshot — parse nohup.log instead).
- run5: 80x6 net, lr 1e-4 -> 5e-5 at iteration 91, sims 2400, soft_gate
  checks every 5 iters (516 games @ 100 sims). It promoted at iteration 5
  (56.1% — harvested run4's data backlog); current_best = run5/iter_0005.
  The 19 checks since: mean ~48.4%, no trend, across both learning rates.
  Two live hypotheses:
  (A) weight noise-ball wander masks small gains -> the checkpoint AVERAGE
      should BEAT current_best;
  (B) self-play data is exhausted for this net -> the average should TIE.
- The laptop GPU is free. The advisor uses the laptop's
  runs/kingdomino/best_checkpoint/current_best.pt.

ITEMS (in order):

1. Average the noise-ball checkpoints (local, CPU, ~1 min):
     python -m games.kingdomino.average_checkpoints \
       --dir runs/kingdomino/cloud_80x6_run5 --first 6 --last 90 \
       --out runs/kingdomino/cloud_80x6_run5/avg_0006_0090.pt

2. Head-to-head — avg_0006_0090 vs iter_0005 (current_best).
   516 paired seat-swapped games @ sims=100 (comparable to run5's promotion
   checks) via promotion.evaluate_checkpoint_match — model the runner on
   runs/kingdomino/ablation_h2h.py (detached Start-Process + monitor pattern;
   background Bash gets killed at 10 min).
   Pre-registered interpretation (candidate = the average):
     >= 53%  -> noise-ball CONFIRMED. Run a confirmation match at sims=400 /
                400 games (strength should survive deeper search). If that
                also holds >= 52%, copy the averaged net to the LAPTOP's
                runs/kingdomino/best_checkpoint/current_best.pt so the advisor
                uses it. NEVER push anything to the box or alter its
                current_best.pt — the running gate owns that file.
     48-52%  -> tie: wander is NOT hiding gains; data exhaustion confirmed;
                run6 = diversity package (see report section below).
     < 47%   -> unexpected (checkpoints not one basin) — investigate before
                concluding: check BN running-stat divergence between iter_0006
                and iter_0090, and retry averaging over 40-90 only.

3. Sims sweep with the H2H winner:
     python -m games.kingdomino.sims_sweep --checkpoint <winner.pt>
   Defaults are right (120 positions, rungs 400..12800, knee threshold 5%).
   ~40 min on the laptop GPU. Deliverables: the knee, the per-deck-size
   table, and a concrete advisor recommendation (flat sims value or a
   deck-size schedule).

4. Preliminary 5e-5 read from the snapshot (analysis only — the snapshot
   freezes at ~iteration 96, so the final verdict at checks 100-110 must wait
   for the next sync; note that clearly in the report):
   Parse runs/kingdomino/cloud_80x6_run5/nohup.log for the promotion lines
   and per-iteration `brier_diag` values. Report: the 19-check series, the
   iteration 85->90 swing (52.9% -> 44.0%), the single post-drop check (95:
   46.6%), and whether brier_diag over iterations 91-96 shows any early
   flattening relative to the 1e-4 trend (0.12 -> ~0.21 over iters 10-90).
   Pre-registered criteria for the eventual verdict, in expected order of
   appearance: (i) check win rates tighten into a 47-51 band; (ii)
   brier_diag stops rising; (iii) any check >= 53%.

5. REPORT: write runs/kingdomino/harvest_session_report.md containing: H2H
   results with games/CI, sims-sweep table + advisor recommendation, the
   snapshot 5e-5 analysis, and a single recommended next step chosen from:
   (a) adopt checkpoint averaging as standard practice + continue current
   recipe, (b) run6 = diversity package on the 80x6 (existing flags only:
   --hof_fraction 0.2 --hof_start_iter 1 --temp_moves 30
   --endgame_oversample 1.0, lr per the 5e-5 evidence), or (c) something the
   data forces. Commit and push the report and any scripts created (never
   commit checkpoints/buffers — .pt/.pkl stay out of git).

CONSTRAINTS:
- Fully local session: no ssh, no box access, nothing written to the box.
- NEVER use the legacy games/kingdomino/elo_anchors.csv / elo_db.json — those
  anchors are old-encoder and incompatible. The 80x6 ladder files
  (elo_anchors_80x6.csv etc.) exist but are not needed for these items.
- All matches: paired seat-swapped via promotion.evaluate_checkpoint_match.
- Long-running local jobs: detached Start-Process + Monitor, not background
  Bash (10-minute kill).
