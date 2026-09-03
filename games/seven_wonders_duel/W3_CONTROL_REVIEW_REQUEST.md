# Review request: exact positional control (Workstream 3)

Reviewing commit `c005e7d` plus uncommitted probe changes. Nothing is wired into
the encoder. The decision under review is whether it should be.

## What is being claimed

**The model has a demonstrated blind spot about board control, and an exact
solver can supply the missing fact cheaply.**

The strongest evidence is not the probe. It is table `907773062`, a reviewed game
lost to a forced science victory:

* `SCIENCE_BLOCKING_AND_WONDER_TEMPO_REVIEW.md` records that the checkpoint was
  still POSITIVE at 800 simulations on that position and only crossed negative
  near 2,000.
* The solver computes, in milliseconds, that control of `Observatory` -- the
  card giving the opponent a sixth science symbol -- flips at decision row 85,
  the moment the actor has no unbuilt extra-turn Wonder left. Hold one and it is
  deniable at every Age III row; hold none and it falls in three decisions.

That is a position already lost at the start of Age III, which the model could
not see with 800 simulations of search.

## What was built

`tableau_control.py`, scoped to **positional forcing only** -- the "topology-only
control" option of the two contracts in `WORLD_CLASS_MODEL_EVOLUTION_PLAN.md`,
and a supplement to neural evaluation rather than a replacement.

* Exact memoized minimax over the removal poset. An extra-turn Wonder is
  modelled as the engine does it: the burial removes an accessible slot, then
  the same player moves again.
* Outputs named as topology facts -- `can_take_first`,
  `decisions_until_accessible`, `must_open` -- never as victory claims.
* Public information only. Scrambling every face-down identity leaves answers
  unchanged (tested), which matters because `advisor_scrape` hands the searcher
  a determinization.
* Accessibility gated against the engine's own `is_accessible`.
* Tractable: a full 20-card Age with four extra turns per side is 6,735 nodes
  and 18 ms, so a fresh-Age control map is precomputable.

Fresh Age III, first-mover control by tempo budget:

| my tempo | opp tempo | slots I take first |
|---|---|---|
| 0 | 0 | 70% |
| **1** | **0** | **100%** |
| 0 | 1 | 10% |
| 1 | 1 | 45% |

One unspent extra-turn Wonder against an opponent with none is the difference
between 10% and 100% of Age III.

Affordability is deliberately NOT modelled: the solver states the topological
fact, and the network -- which knows its own coins, production and chains --
learns whether the tempo is attainable. `Theology` is handled as a counterfactual
rather than a mechanic, since it makes every Wonder grant an extra turn.

## The probe, and why it is the weaker evidence

`control_redundancy_probe.py` asks whether the trunk already encodes control:
frozen trunk, small head, group-aware split, 2,049 groups, 8 seeds, probing both
the pooled readout the heads see and the full token set.

| target | pooled | tokens |
|---|---|---|
| `my_tempo` (reference) | 0.86 | 0.92 |
| `their_tempo` (reference) | 0.85 | 0.92 |
| `control_now` | 0.66 | 0.75 |
| `control_if_opponent_takes_theology` | 0.62 | 0.72 |
| `control_with_one_more_tempo` | 0.58 | **0.67** |

Read against the tempo references, which mark what "the model has this" looks
like for this probe rather than 1.0. The counterfactual is consistently the
weakest target on both sources.

**But representation is not use.** A 0.75 R2 on `control_now` says a trained
probe can recover control from the trunk. It does not say the value head uses it
-- and the `907773062` evidence is that the model did NOT act on it, needing
~2,000 simulations to reach a conclusion the solver reaches immediately. A high
probe score would weaken the redundancy argument; it would not explain away the
demonstrated failure.

## Specific questions

1. **Is the probe measuring the right thing at all?** Given that representation
   and use come apart, is there a better test than probe-R2 for "would this
   feature change behaviour"? A value-head-only fine-tune with the feature
   appended, measured on the corpus, would be more direct and more expensive.
2. **Does the solver's contract hold?** In particular the seventh-Wonder rule is
   applied to the tempo BUDGET but not inside the search, so both players
   spending extras during a line could exceed seven builds. How much does that
   matter for the outputs claimed?
3. **Is "both players play optimally for this target" too strong?** It is a bound
   on what is positionally possible, not a prediction. Does that make the feature
   misleading in positions where a player has better things to do?
4. **Is the fresh-Age control map the right shape for the encoder**, or should
   the feature be per-slot rather than an aggregate fraction? Control is a
   property of individual slots; the current features aggregate.
5. **Self-play integration.** These are exact public facts, so they can be
   computed during self-play as well as at inference. Is there a reason not to,
   beyond throughput?

## Known flaws, stated rather than found

* **The leakage I flagged turned out to be negligible, and my claim about it was
  wrong.** `c005e7d`'s message says "every R2 above is inflated"; measured on
  identical data, a position-level split gives 0.77 / 0.68 against the
  group-aware 0.75 / 0.67 -- a 0.01 to 0.02 effect. The reason is that 2,000 of
  the 2,200 positions are self-play games contributing one position each, so
  ~91% of the data has group size one and the two splits are nearly the same
  partition. The leak was real in principle and immaterial in practice at this
  ratio; it would matter more on a corpus-heavy dataset.

  The corollary is that the earlier, lower numbers (0.68 / 0.50 on 700
  positions) were low because of DATA SIZE, not leakage. I changed both at once
  and predicted the wrong direction.
* **The tempo references are an argument, not a bound.** `my_tempo` needs the
  `PLAY_AGAIN` property and the seventh-Wonder rule, so it is the most available
  target rather than a ceiling. Only the ORDERING is meant to carry weight.
* **`must_open` and `decisions_until_accessible` are implemented and tested but
  never validated against the corpus.**
* **The solver is single-Age.** Cross-Age control needs the next-Age starter,
  which depends on conflict position -- outside the topology contract. The
  fresh-Age map partly compensates by being computable in advance.

## Two earlier tests recorded as uninformative, so they are not repeated

* Aggregate control does not correlate with measured action regret (weak, and
  falsified by a position with `control_now = 1.00` beside regret 52.3). Control
  is a precondition rather than a predictor, and regret measured against searches
  by a model blind to control is circular.
* Targeted control after the threat-creating action is CONSTANT across the
  corpus, because `threat_corpus_scan` selects positions on exactly that
  criterion. That test measured the selection filter.

## Context

Four search-side mechanisms have been built and measured this cycle, and all
fall short: chance-sibling bias (null at every gain), Wonder-action factorization
(interior nodes only, no strength evidence), afterstate clustering (cost
unmeasured, both negative controls retracted), and the public tactical leaf
extension (19% of its target error). The threat corpus then measured 9 of 11
positions carrying real action regret, median about 15 points and up to 68.6.

This workstream is the first to attack the problem as a missing FACT rather than
a search-allocation issue.
