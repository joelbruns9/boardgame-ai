# Kingdomino contextual open-loop MCTS plan

- **Status:** Proposed offline qualification only. No engine change, self-play
  run, or training run is authorized yet.
- **Date:** 2026-08-13
- **Checkpoint:** `runs/kingdomino/best_checkpoint/current_best.pt`
  (`4bf07b0c...`, 80x6)
- **Primary question:** Can a very small public-information context preserve
  useful post-reveal adaptation without the branch fragmentation that made
  explicit chance modeling uncompetitive?

## 1. Decision summary

The placement audit did not earn a new head or placement curriculum. The
corrected deep-target audit did identify a concentrated development signal, but
it must pass the untouched confirmation split before earning selective
relabeling. The durable data output is a 1,400-state reconstructable
strong-human corpus. If those positions enter training, they should do so
through production-search targets or BGA-seeded self-play, not as unquestioned
one-hot human actions.

The remaining search idea is narrower. Current open-loop MCTS merges all
concrete states reached by the same slot-relative action history. Legal actions
are filtered against each simulation's concrete state, but an edge's visits and
value are still averaged across different revealed rows, dominoes in hand, and
boards. More simulations make that average more precise; they do not make it
observation-conditioned.

Full information-set/chance modeling and progressive exposure of exact reveal
rows have already failed to establish a practical advantage. Most decisively,
the fixed-network progressive Gate 0 scored 48.24% at 800 simulations and
49.46% at 4,800 simulations against ordinary open loop. Contextual open-loop
MCTS and CORAL are therefore not two new levers. They are two choices of
abstraction for one final question:

> Can strategically similar revealed states share conditional Q statistics
> while strategically opposite states stop sharing them?

The next step is an offline abstraction test using already-generated chance
samples. Only a strong, stable result earns an advisor-only Rust implementation.

## 2. Evidence already available

### 2.1 BGA audits

- The corpus contains 1,400 reconstructable public roots from 36 games: 940
  development and 460 confirmation.
- The exact late-placement confirmation found `current_best` no worse than the
  strong-human opponents. Placement supervision is closed.
- Corrected repeated 30,000-sim and matched-pick reanalysis produced a positive
  development lower bound over 4,800-sim choices. Selective relabeling is
  pending confirmation; broad 30,000-sim self-play is not implied.
- The two largest raw-policy/human pick disagreements were first-claim-order
  decisions in Mighty Duel. Both humans later received the model-favored tile.
  One human/model ordering pair was exactly equivalent; the other retained only
  a small non-exact search difference.

Consequently, the BGA corpus is useful as a state-distribution source and
evaluation anchor. It is not evidence that human actions are clean ground-truth
policy labels or that chance aliasing caused the observed game outcomes.

### 2.2 Chance-contingency headroom

The frozen 50-position A-1 audit compared an observation-conditioned one-reveal
reference with aliased alternatives. Depending on sampling arm, the adaptive
reference changed the backed root pick on 2-5 of 50 positions. Three cases were
reasonably stable, with reported adaptive regret around 0.017-0.077 Q.

This established a small local contingency signal, not a stronger search
engine. The audit itself required an A1 equal-compute search comparison and an
A2 paired-game gate before any strength claim.

### 2.3 Practical progressive-chance search probe

The subsequent search probe tested chance exposures `X=0,1,2,4` on the same
family of positions. At 128 simulations on 50 positions, mean actor regret
against the 3,200-sim reference was 0.03936 for `X=0` and 0.04082 for each of
`X=1,2,4`: no improvement from added exposure.

At 4,800 simulations on 12 positions, conclusions depended on which same-family
10,000-sim reference scored the arm. Against the ordinary-open-loop reference,
`X=0` had the lowest mean regret (0.00874 versus 0.01345 for `X=4`). Against the
hybrid reference, `X=4` looked better (0.02829 versus 0.03877). The two deep
references agreed on the pick in 11/12 positions but on the exact top action in
only 7/12.

That reference dependence does not qualify progressive chance modeling as an
improvement. It closes exact/progressive reveal branching at the tested compute
unless an independent objective supplies new evidence.

### 2.4 Fixed-network progressive Gate 0

The later production treatment supplied the independent game-strength test the
small probes lacked. It used balanced, progressively widened chance panels for
deck counts 8 and 12, capped at 16 active outcomes, against ordinary open loop
with the identical `current_best` checkpoint and matched neural work.

| Budget | Games / pairs | Progressive score | One-sided 95% interval | Verdict |
|---:|---:|---:|---:|---|
| 800 | 2,048 / 1,024 | 48.24% | [47.34%, 49.15%] | Fail |
| 4,800 | 4,096 / 2,048 | 49.46% | [48.69%, 50.23%] | Inconclusive |

At 4,800 simulations, mean active and mature widths were 15.98 and 15.93 out
of 16. The null therefore cannot be explained by the treatment failing to
activate. The higher budget converged toward parity rather than a positive
advantage, so the implemented progressive treatment is closed for advisor and
training use. See `CHANCE_PROGRESSIVE_GATE0_FINDINGS.md`.

Relevant artifacts:

- `games/kingdomino/CHANCE_PROGRESSIVE_GATE0_FINDINGS.md`
- `runs/kingdomino/chance_correct_a1/A1_SUMMARY.md`
- `runs/kingdomino/chance_correct_a1/search_probe_n50_s128_ref3200.json`
- `runs/kingdomino/chance_correct_a1/search_probe_dualref_n12_s4800_ref10000.json`

## 3. One continuum, not three independent ideas

Let `R` be the public state after a simulated reveal and `c=f(R)` a context.

| Method | Context function | Sharing behavior |
|---|---|---|
| Current open loop | constant | all reveals share one Q per action |
| Feature-context open loop | compact tile/board bucket | states in one feature bucket share Q |
| CORAL-style open loop | heuristic preferred action, or intent | states with the same intent share Q |
| Full chance/information-set tree | exact observed state or row | no abstraction across distinct reveals |

Contextual open loop estimates `Q(action | context)`. CORAL is a special case
where the context is the action preferred by a heuristic. Faithful CORAL also
uses an observational warm-up, conditional counterfactual bandits, and Thompson
sampling. Those additions are not the first question for Kingdomino. Replacing
PUCT and adding a new bandit simultaneously would make the result impossible to
attribute.

Progressive chance widening is different operationally but targets the same
underlying contingency value. It gradually allocates separate statistics to
exact reveal outcomes. Context abstraction instead pools different outcomes on
purpose. It can only improve on the failed exact-row approach if that failure
came from sample fragmentation and the strategically relevant information is
low-dimensional.

## 4. Information and action contracts

Any candidate must obey all of the following:

1. Context may use only information public at that simulated history: boards,
   current domino, revealed row, claims, actor, phase, and the unordered
   remaining bag.
2. Context must never use the sampled order of unrevealed dominoes.
3. The real public root remains ordinary PUCT. Conditioning begins only after a
   simulated reveal has occurred.
4. Pick rank is already encoded by the slot-relative joint action (`idx % 5`).
   Adding "rank slot" alone supplies no new context.
5. Pick-group choice is the primary experiment. Placement inside a selected
   pick group remains ordinary joint-action PUCT because the exact placement
   audit did not establish placement headroom.
6. Global edge statistics remain available as a fallback. A sparse context may
   not discard all evidence learned in other contexts.

The intended conditional estimate is a shrinkage estimate such as:

```text
Q_used(c,a) = [N(c,a) * Q(c,a) + lambda * Q_global(a)]
              / [N(c,a) + lambda]
```

This makes a new context behave like current open loop until it accumulates
enough visits.

## 5. Phase 0: offline abstraction test

Do not modify MCTS first. Reuse the saved A-1 per-reveal action values if they
contain the required fields. If they do not, rerun only the frozen 50 positions
with additional trace output; do not expand the corpus or simulation ladder.

Compare these abstractions:

1. **Constant:** current open-loop aliasing baseline.
2. **Exact row:** adaptive upper reference, not a deployable early-game arm.
3. **Enriched edge:** base joint action plus public signatures of the current
   domino and the concretely picked domino. At minimum include terrain pair and
   crown distribution; exact domino ID is a diagnostic upper arm.
4. **Intent:** the pick slot preferred by a cheap actor-relative heuristic.
   This is the CORAL-style arm.
5. **Intent plus pressure:** preferred slot plus one coarse class describing
   whether the preference is primarily own-board fit, opponent denial, or turn
   order. Keep the total context count small.

The heuristic may use existing Rust quantities: immediate territory delta,
crowns weighted by owned terrain size, opponent fit/denial, placement count or
discard risk, and row rank. It must not invoke an additional neural-network
evaluation during tree descent.

For each abstraction report:

- value recovered between the constant baseline and exact adaptive reference;
- root pick and exact-action agreement with the adaptive reference;
- regret under one shared reference, never a same-family private reference;
- bucket count, median and lower-tail samples per bucket;
- results by remaining-bag size;
- stability between balanced and matched-IID chance samples; and
- the three previously stable A-1 cases separately, without making them the
  sole gate.

For positions with a positive exact-adaptive gap, define:

```text
recovered_fraction =
    (context_value - constant_value)
    / (exact_adaptive_value - constant_value)
```

The offline gate passes only if one compact abstraction recovers a material
fraction of the adaptive value under both chance-sampling designs, improves the
shared-reference decision metric rather than only Q calibration, and retains
enough samples per active context to be plausible at 4,800 simulations. Freeze
the chosen context and shrinkage rule before any held-out search comparison.

If no compact abstraction passes, close contextual open loop and CORAL together.

## 6. Phase 1: advisor-only contextual PUCT

Only after Phase 0 passes, implement one frozen treatment in the Rust advisor
path. Do not change training self-play yet.

At an eligible post-reveal node maintain:

```text
global:       N[action], W[action]
conditional:  N[context][pick_group], W[context][pick_group]
```

Use conditional PUCT to choose a pick group, followed by existing PUCT to choose
the placement within that group. Prefer a lazy fixed-size table (four live pick
slots; final-placement nodes use the baseline). Do not implement full
action-by-action CORAL or Thompson sampling in this phase.

Compare treatment and baseline at equal:

- network evaluations;
- simulation counts (800 and 4,800); and
- measured wall time.

Primary evaluation is held-out shared-reference regret and stable root-pick
improvement. Also report throughput, memory, missing-child/fallback counts, and
seed sensitivity. A slower method must be compared at equal wall time as well
as equal simulations.

Failure closes the lever. Passing earns a small paired-game A2 evaluation, not
automatic use in self-play or training.

## 7. Phase 2: game and training gates

If the advisor-only treatment passes:

1. Run a small paired, seat-swapped, common-seed match against current open
   loop at equal wall-clock search cost.
2. Require evidence of improvement or at minimum non-inferiority plus a clear
   reference-quality gain. A few move changes alone are insufficient.
3. Only then consider using contextual MCTS for self-play labels.
4. Any training experiment must use a control checkpoint, the BGA anchor, and
   the existing game-clustered evaluation rules. It cannot promote
   `current_best` automatically.

The original CORAL warm-up/Thompson machinery remains deferred. It is worth
testing only if intent-conditioned PUCT passes and there is a specific residual
exploration failure that PUCT cannot address.

## 8. Separate BGA data-distribution action

The BGA corpus remains an independent data lever. Its valid uses are:

- start self-play episodes from reconstructed public states, redeterminizing
  only the unordered hidden bag; or
- add the roots as policy-distillation examples labeled by the production
  search, with an explicit value-target and weighting contract.

Do not insert one-hot human first claims as authoritative labels. In Mighty
Duel, an isolated first claim does not describe the eventual two-tile bundle,
and the largest apparent disagreements demonstrated that failure mode.

The corrected deep-target development audit says that selected 20k-30k labels
may be better, but confirmation is pending and the effect is concentrated.
BGA-seeded self-play is still a distinct state-distribution experiment. A
separate plan should specify root sampling, continuations per root, value
labels, mixture weight, and game-level split before a training run.

## 9. Stop conditions

Close this open-loop branch if any of the following occurs:

- no compact context recovers stable adaptive value offline;
- apparent benefit changes sign between balanced and IID chance samples;
- active buckets are too sparse at a 4,800-sim equivalent budget;
- the treatment improves its own reference but not a shared reference;
- equal-wall-time advisor performance is not better; or
- the paired game gate is negative.

Do not respond to a failure by increasing context cardinality until it becomes
an exact-row tree. That would recreate the already failed progressive chance
experiment under a different name.
