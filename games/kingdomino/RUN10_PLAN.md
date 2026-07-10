# Run10 planning document (drafted 2026-07-10, during run9)

Theme: **adversarial drafting** — teach the pipeline the interaction dimension
(picks) that self-play equilibrium never taught. Every item below traces to
measured evidence from the kylechu20 loss post-mortem (table 880439726) and
its instrumented rematch win (table 881060676).

## Evidence base (do not relitigate; measurements in campaign memory + tools)

1. **Score heads regress to self-play equilibrium (~135) and are
   under-dispersed on human games**: squeezed player over-projected +25–31,
   blowout winner under-projected −26; both heads ~+11 baseline offset on
   human games. (`runs/kingdomino/bga_score_audit.py` over the game log.)
2. **Policy prior starves squeeze lines, and search inherits it**: the
   game-losing opponent reply had 4.7% prior (their top prior was 40.8%) and
   received 0.62% of 3200 sims. Rooted searches at the same node saw the
   danger (+0.256 their frame) — starvation, not depth, is the failure.
3. **Tactical (board-visible, in-horizon) denial is already learned**;
   strategic multi-round squeezes are not: value labels assume benign
   continuations AND exploration can't propose squeeze lines. Both gaps must
   close together (they feed each other).
4. **Boards are disjoint in 2p — ALL interaction flows through picks**
   (denial + turn order). This collapses opponent-response analysis to ≤3–4
   branches (the advisor draft-matrix insight, validated in the rematch:
   6/8 flagged decisions played max-robust, incl. two headline inversions).
5. Ruled out earlier: capacity (bake-off: 80x6 matched 96x6/80x10),
   search depth (sims sweep), loop hygiene (run8/8b: self-healing verified).

## Package (in priority order)

### 1. Pick-group visit floors in search (the training-native draft matrix)
At opponent decision nodes in the shallow tree (depth ≤ 2), guarantee each
PICK-GROUP a minimum visit share before normal PUCT resumes.
- Pick of a child = `joint_idx % 5` (codec: `placement*5 + pick`) — free.
- Reallocates existing sims (~10–15% of shallow-node budget); ~zero wall-clock.
- Wire through the EXISTING `forced_playout_subtraction` machinery so forced
  visits are removed from policy targets: search explores squeezes, targets
  lift only where values prove out. This breaks the prior→visits→target
  rich-get-richer loop at its narrowest point.
- Flags: `--pick_floor_frac` (per-group min share, ~0.05), `--pick_floor_depth`
  (default 2), off by default.

### 2. Spite personality in the HOF pool (linear-margin maximizer)
The value blend squashes margin through tanh(2m/160), which DEVALUES
large-margin denial. An opponent evaluator with `alpha=1` and a SMALL
margin_gain (~0.2, linear regime) is a pure point-differential maximizer:
denying the learner 8 points = gaining 8. Uses the per-opponent evaluator
knobs shipped in run9 (`hof_alpha_choices` generalized to alpha+gain pairs,
e.g. `--hof_style_choices "0:2.0,1:2.0,1:0.2"`).

### 3. Keep from run9 (whatever its verdict): random openings, recency-
weighted pool, personalities {0,1}, averaged gating k=8, reset-after-2,
STOP file, buffer autosave, gate-skip during warmup.

### 4. Optional, only if 1+2 underdeliver: denial-selection wrapper
Opponent seat picks among searched root children by
`(1−λ)·own_value + λ·(learner's eval drop)` — an explicit spite dial.
More invasive than 1+2; build only with evidence.

## Explicitly deferred (with triggers)

- **Exploiter loop (PSRO-lite)**: train a small net purely to beat the frozen
  best; it is both the fix and the exploitability METRIC (the honest answer
  to "can it stand up to top-10 humans"). Trigger: run10 improves the audit
  but BGA losses to strong players persist. Run11-scale.
- **Distributional score head (quantiles)**: attacks under-dispersion
  architecturally. Trigger: data-side fixes leave the score audit unmoved.
- **BGA-position-seeded self-play**: needs Rust start-from-state; seed pool
  (bga_game_log) still small. Revisit when ~50+ games logged.
- **Offline matrix distillation** (robust values as aux labels on ~1% of
  buffer): ranked below pick-floors (patches points; floors fix everything).

## Pre-registered measurements (all tools exist)

1. **Score audit** (`bga_score_audit.py --checkpoint X` over the game log):
   err_you in losses should shrink from +25–31 toward ±10; game-5-style
   under-projection of blowouts should shrink too.
2. **Row-29 prior check** (the smoking-gun position, table 880439726
   game -1, decision 29): opponent's d34-pick-d8 should no longer sit at
   4.7% prior / 0.6% visits under a run10 net.
3. **Standard gates**: promotions vs the run9/run8 banked average, usual
   ratchet (avg k=8, 0.51/0.50 on 2500@300).
4. Advisor draft-matrix fragility on logged games as a qualitative check
   (fewer "all picks fragile" surprises = search seeing squeezes earlier).

## Sequencing / dependencies

- Run9 verdict first (gates ~55–65; pre-registered: promotions resume =
  diversity thesis pays; parity = 80x6 self-play near ceiling without
  external signal — run10 proceeds either way, but a run9 promotion changes
  the warm start).
- Warm start: best banked artifact at the time (currently run8 avg 4bf07b;
  canonical `best_checkpoint/current_best.pt`).
- Implementation estimate: item 1 ≈ one day (Rust selection + target wiring
  + forced-path verification), item 2 ≈ hours (config + evaluator plumbing).
- Verify pattern as always: unit/forced-path tests locally, tiny run, push;
  launch script with run.pid + STOP instructions; download artifacts the day
  they're minted.
