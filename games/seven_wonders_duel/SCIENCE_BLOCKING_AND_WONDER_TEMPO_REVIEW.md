# Science Blocking and Wonder Tempo Review

## Purpose and conclusion

This note records the investigation prompted by BGA table `907773062`
(RollwJoel versus Alexmp1), the advisor defects fixed during it, what the
existing checkpoint actually gets wrong, and the smallest defensible learning
plan.

There are three findings with different evidence levels:

1. **Two BGA advisor representation bugs were real and are fixed.** Live BGA
   observations omitted built Wonder tokens and could omit the retired eighth
   Wonder. These were mapper/serving bugs, not neural-network bugs.
2. **The checkpoint recognizes the public post-reveal loss too slowly.** On the
   corrected representation, record 84 of the captured game is still positive
   at 800 simulations, crosses negative near 2,000, and approaches certain loss
   only at much greater depth.
3. **The Mausoleum build is not yet proven to be the ex-ante mistake.** It is
   the earliest irreversible cause on the deal that occurred, but Age III was
   hidden when the action was chosen. A 100,000-simulation search, including
   the subsequent Age III chance node, remains strongly pro-Mausoleum. That is
   evidence about this model's preference, not an independent game-theoretic
   oracle.

The previous version of this note treated item 3 too much like a settled
premise. The implementation plan below now starts with a labeled regression
corpus and separates measured tactical deficiencies from speculative feature
work.

## The critical realized line

At the end of Age II, four Wonders had been constructed. RollwJoel retained The
Sphinx and The Mausoleum; Alexmp1 retained The Appian Way and The Pyramids.

The realized line was:

1. RollwJoel constructed The Mausoleum as the fifth Wonder, using Customs
   House, and revived Statue.
2. Alexmp1 later constructed The Appian Way as the sixth Wonder and took its
   extra turn.
3. At BGA move 55, only The Sphinx and The Pyramids remained unbuilt. The next
   Wonder would be seventh and retire the other.
4. If RollwJoel preserved The Sphinx, Alexmp1 could construct The Pyramids and
   retire it. Constructing The Sphinx was therefore forced by the earlier
   Wonder-count commitment.
5. The Sphinx action and following tableau sequence transferred control of the
   coverers. Observatory was visible and would give Alexmp1 a sixth distinct
   science symbol. The actual reveal left no escape and the science loss became
   forced.

The Mausoleum build is therefore the earliest irreversible cause in the
*realized* line. It is not automatically a legal-information blunder. The
advisor had to value the position over the hidden Age III guild selection,
layout, and later reveals, not condition on the deal that happened.

## Evidence ledger

### Advisor correctness: proven and fixed

The engine and self-play use this invariant:

- `city.wonders` contains every Wonder drafted by that player.
- `city.built_wonders` is the constructed subset.
- `game.retired_wonders` identifies the unbuilt Wonder removed after the
  seventh construction.

The BGA mapper previously emitted only unbuilt Wonders in `city.wonders`.
Because the encoder creates Wonder tokens by iterating that collection, built
Wonder tokens disappeared before live inference. In this game, the live model
received four Wonder tokens instead of eight at the end of Age II and two
instead of eight at move 55.

The mapper now emits all structurally owned Wonders, with built Wonders as a
subset. Draft reconstruction and public advisor summaries were changed to the
same canonical contract.

The mapper also used to hard-code an empty retired-Wonder set. BGA removes the
eighth Wonder from `wondersSituation` after the seventh construction. The
mapper now recovers the retired Wonder and owner from either the still-visible
structural row or the captured game log. If neither source can identify exactly
one retired Wonder and owner, it raises `UnsupportedBgaState` instead of sending
a plausible but incomplete input to the network.

These are advisor implementation bugs. They did not corrupt self-play or
training examples.

### Corrected-input Age II search: Mausoleum remains preferred

The retirement bug did not affect the final Age II choice because no Wonder had
retired yet. Restoring the missing built-Wonder tokens made the checkpoint more
optimistic about Mausoleum in the original 2,000-simulation comparison:

| Input | Mausoleum visits | Mausoleum Q | Root Q |
| --- | ---: | ---: | ---: |
| Former live representation | 1,470 | +0.044 | +0.026 |
| Canonical training representation | 1,570 | +0.169 | +0.152 |

A progressive search on one persistent tree at captured record 72 produced:

| Simulations | Root Q | Mausoleum Q / visits | Discard Q / visits | Sphinx Q / visits |
| ---: | ---: | ---: | ---: | ---: |
| 800 | +0.139 | +0.163 / 552 | +0.116 / 219 | -0.336 / 14 |
| 2,000 | +0.152 | +0.169 / 1,570 | +0.116 / 384 | -0.314 / 23 |
| 10,000 | +0.178 | +0.187 / 9,161 | +0.113 / 742 | -0.310 / 52 |
| 25,008 | +0.184 | +0.190 / 23,615 | +0.119 / 1,241 | -0.305 / 83 |
| 100,000 | +0.190 | +0.193 / 97,219 | +0.119 / 2,479 | -0.302 / 167 |

This is deep-search evidence that the current model consistently prefers
Mausoleum; it is not proof that Mausoleum is objectively best. The leaves still
come from the same checkpoint whose tactical calibration is in question. The
node also contains hidden uncertainty: a determinization chooses the unknown
Age III guild subset, while the search integrates layouts and subsequent chance
events conditional on that pool. A proper corpus entry must average multiple
determinization seeds or enumerate the legal guild subsets.

The correct present conclusion is therefore: **the reviewed game motivates the
hypothesis, but does not yet establish that retirement-tempo encoding caused an
Age II error.**

### Corrected-input post-reveal ladder: measured tactical weakness

At captured record 84, the relevant cards and cover relationship are public.
The same progressive-search protocol produced:

| Simulations | Root Q | Approximate win probability |
| ---: | ---: | ---: |
| 800 | +0.158 | 57.9% |
| 2,006 | -0.107 | 44.6% |
| 5,053 | -0.382 | 30.9% |
| 25,365 | -0.833 | 8.4% |
| 100,738 | -0.936 | 3.2% |
| 250,000 | -0.963 | 1.8% |

This is the clearest failure in the game. The network plus shallow search does
not propagate the forced science sequence quickly enough, even though deeper
search eventually does. The issue is tactical value propagation/search
efficiency, not failure to identify a visible, immediately accessible sixth
symbol.

## What prior science-threat measurements already establish

The `sevenwd-science-threat-prior` study used this checkpoint on a 250-game
mixed corpus. Its results constrain the feature hypotheses:

- The learned board representation already predicts an eventual opponent
  science win with AUC 0.789 in Age I and 0.983 in Age II, outperforming the
  tested hand-built board features.
- When the opponent is one face-up accessible symbol from winning, a
  200-simulation search selects the block 95.7% of the time.
- Science underprediction was concentrated in the `joint7` auxiliary output,
  not specifically in the scalar value head. The value head's seat-0 optimism
  was roughly uniform across threat quintiles, including +0.289 in the safest
  bucket.

Consequences:

- Do not prioritize generic "science-conducive board" features such as another
  distinct-symbol count or broad science-rush flag. The trunk already recovers
  that information well.
- Do not describe the model as generally unable to block science. It usually
  blocks a visible immediate win at shallow search.
- The unmeasured and defensible gap is **tactical control**: symbol-copy
  scarcity, the final coverer of a winning card, reveal risk, preventability,
  Wonder retirement, and multi-turn tableau control.

The 250-game study should be checked into a stable report artifact alongside
the new regression corpus. At present its run label and measurements are the
provenance; they are not discoverable from a small committed result file.

## Checkpoint and replay context

`candidate_0085.pt` uses encoder signature
`24e15b12e1e7cd9f2222a32c1ec022140a42957c42c2c5c5ba352207166a8975`.
Its final validation science-symbol MAE is 0.14365 on a target normalized by six,
or 0.862 symbols. The constant baseline is 0.23372, or 1.402 symbols. The head
therefore removes only about 38.5% of the baseline absolute deviation; one
symbol still decides many science outcomes.

The iteration-85 replay window contains 20,000 games and 371,006 derived policy
examples. Its replay summary records 1,401,979 decisions, so 26.46% of decisions
become policy examples. This is distinct from `buffer_passes = 0.27624`, which
is the fraction of training examples consumed in that iteration. The same
window contains 4,356 science wins, or 21.78%. These figures explain why a
blanket victory-class rebalance is unlikely to target the rare event that
matters: the earlier high-regret blocking decision.

### Law-conditioned self-play audit

A second audit replayed 14,000 candidate-era games from iterations 72 through
85 and split them by the initial progress-token and Wonder setup:

| Setup | Science wins | Rate |
| --- | ---: | ---: |
| Law on the progress-token board | 2,245 / 7,056 | 31.82% |
| Law absent | 825 / 6,944 | 11.88% |
| Law absent, neither player drafted Great Library | 139 / 2,245 | 6.19% |
| Law absent, at least one player drafted Great Library | 686 / 4,699 | 14.60% |

This supports the user's self-play-equilibrium concern. The exact setup in the
reviewed game lies in a regime where the replay population learned that science
wins are uncommon. That does not mean the early economy choices were certainly
wrong: the unconditional strategy must still trade a rare science route against
ordinary economic and civilian value. It does mean that a dangerous no-Law
science line supplies much less training signal and may be systematically
discounted.

The checkpoint is not blind to Law. In diagnostic record-13 and record-19
counterfactuals, replacing Architecture on the initial token board with Law
produced the following raw predictions and policies:

| State | Predicted opponent science win | Most likely action |
| --- | ---: | --- |
| Record 13, actual no-Law setup | 1.10% | Clay Pit, 72.0% |
| Record 13, Law substituted | 4.58% | Pharmacist, 42.0% |
| Record 19, actual no-Law setup | 1.96% | Clay Pool, 76.6% |
| Record 19, Law substituted | 10.23% | Pharmacist, 56.5% |

At record 13 the scalar win estimate also fell from 69.78% to 66.11%; at
record 19 it fell from 65.10% to 60.24%. The model therefore recognizes Law as
a strategic enabler and changes from economy toward science when it is present.
The most likely diagnosis is not missing Law input. It is overgeneralization
from the low base rate of successful no-Law science attacks, combined with weak
credit assignment from a loss dozens of decisions later.

The actual-game outputs show the same slow escalation. At record 13 the raw
head assigned only 1.10% to an opponent science victory and predicted 2.20
final opponent symbols. By record 72 those figures had risen to 19.15% and
5.37 symbols. At the already-public forced-loss record 84 they were still only
14.69% and 5.87 symbols. The threat was represented, but neither its probability
nor its inevitability was calibrated soon enough.

## What the current encoder can distinguish

Encoder version `7wd-encoder-5` already supplies:

- Age, cards remaining, remaining-card parity, and whether military is tied.
- Wonder identity plus per-Wonder built, retired, affordable, cost,
  extra-turn, and shield features.
- Per-player unbuilt-Wonder and unbuilt-extra-turn-Wonder counts.
- Tableau row/x coordinates, accessibility, coverer count, and hidden cards
  covered.
- Exact held science-symbol flags, distinct counts, symbols needed, obtainable
  missing symbols, feasibility, and `gives_sixth_symbol` on revealed cards.

The engine also models end-of-Age timing correctly. An extra-turn Wonder repeats
the turn only while an accessible tableau card remains. If the action empties
Age I or II, the next Age is dealt and the militarily weaker player chooses its
starter; on a military tie, the player who took the final card chooses.

The raw policy at record 72 was not a generic "build any Wonder at the end of
the Age" rule:

| Action | Raw prior |
| --- | ---: |
| Discard Customs House | 39.3% |
| Construct The Mausoleum | 37.4% |
| Construct The Sphinx | 17.5% |
| Construct Customs House | 5.8% |

The network can distinguish Mausoleum from Sphinx and intrinsic extra-turn
status. What it does not receive directly is whether an extra turn is usable
for this card/action, the resulting Wonder build ordinal, which unbuilt Wonder
would be retired next, or explicit tableau adjacency.

The encoder also makes the strategic Law state available. Progress tokens have
learned identities and availability/ownership/candidate features. Science
feasibility includes Law when it is on the board, already owned, or potentially
obtainable through a player's unbuilt Great Library. This last case is a coarse
possibility flag; it does not encode the probability of drawing Law from the
five out-of-play progress tokens.

### Input comparison with ZeusAI

The models differ in both capacity and input philosophy:

| | Current candidate | ZeusAI paper description |
| --- | --- | --- |
| Parameters | 15,808,626 | approximately 92 million |
| Transformer | 8 layers, 6 heads | 12 layers, 12 heads |
| Representation width | 384 | 768 |
| Feed-forward width | 1,536 | 3,072 |
| Component identity | Learned entity and token-type embeddings | Learned component embeddings |
| Rules/effects in input | Many explicit, deterministic rule features | No explicit costs or effects |
| Tableau location | Numeric row/x and cover-summary features | Learned positional information |

Our input is not simply a smaller copy of ZeusAI's. It is more rule-aware:
affordability, effective costs, chain status, sixth-symbol completion, military
crossings, Wonder extra turns, science feasibility, remaining hidden-pool
membership, and similar facts are calculated before the transformer. This
reduces the amount of rules learning demanded from a 15.8M-parameter model.

The likely relative weakness is structural identity. ZeusAI describes each
component as paired with a learned position. Our transformer receives row/x,
accessibility, coverer count, and covered-hidden count, but no learned tableau
slot identity or explicit `covers` attention relation. It can reconstruct the
shape, but it must infer a graph and its removal parity from scalar features.
That is a plausible reason it needs deep tree search to discover a forced
reveal sequence. It is a hypothesis to test against the tactical corpus, not
evidence that copying ZeusAI's 92M architecture is required.

## Focused opponents, not a separate production policy

A strong science-focused network is justified as a training opponent and
evaluation oracle, particularly in the Law-absent/no-Great-Library stratum. It
should not replace the general serving model, and it should not be a scripted
agent that takes green cards regardless of whether it is losing. Train or tune
it on the normal terminal win objective while deliberately oversampling setups
and trajectories in which a science route must be created without Law. It must
remain exploitability-tested against strong general play.

Add that specialist to the opponent league. Train the main model from its own
search targets and the resulting outcomes rather than teaching it to imitate a
possibly distorted specialist policy. This creates the missing defensive
experience: the general model repeatedly encounters an opponent that preserves
tempo Wonders, engineers reveal control, and converts four or five symbols in
the hard setup rather than abandoning the attempt because its mirror image
usually does.

Use the same experimental design for a military-pressure specialist. The
current iteration already uses 15% Hall-of-Fame opposition, so self-play is not
pure latest-model mirroring, but a single old general checkpoint does not
guarantee coverage of either focused style. Evaluate the resulting general
model in a matrix split by Law availability, Great Library ownership, military
starting pressure, victory type, and ordinary general-opponent strength. The
change is successful only if it improves defense and retains the ability to
exploit overcommitted specialists.

## Encoding forced reveals without hidden clairvoyance

A transformer can in principle learn tableau parity and multi-turn removal
sequences. Requiring it to rediscover deterministic public-board analysis from
rare terminal losses is unnecessarily sample-inefficient. The model should be
given consequences of public information, never the identity of an unrevealed
card.

The smallest useful addition is a public tableau-control analysis run for each
currently legal take. It should account for cover edges, whose turn follows,
end-of-Age starter choice, usable extra-turn Wonders, and the fifth-through-
seventh Wonder retirement race. Candidate outputs include:

- cards made accessible immediately and over short forced continuations;
- whether the actor or opponent is forced to reveal or receive the final known
  coverer;
- terminal threat distance in 1, 2, 4, and 8 decisions;
- whether the threat is preventable now or within the next two decisions;
- whether spending or preserving an extra-turn Wonder changes control; and
- for an unknown reveal, its terminal-hazard probability from the legal
  remaining pool and card back rather than its actual hidden identity.

Per-card versions can be appended to tableau tokens. Exact action-pair facts,
such as "take this card with Sphinx and retain control," fit more naturally in
legal-action tokens scored by the policy head. That is a larger architecture
change, but cleaner than asking one pooled state vector and a flat action head
to recreate every card/Wonder conjunction.

The same control calculation must be victory-channel neutral. For science its
terminal resource is a missing distinct symbol; for military it is enough
effective shields to cross the final track space. A shared representation can
emit, for each player and each instant-win channel, `distance`, `forced_in_k`,
`preventable_now`, `must_block_in_k`, and `control_owner`. The current encoder
already handles immediate military wins well, but it can have the same
multi-turn reveal and tempo weakness for future shield cards.

## Regression corpus before feature work

Build a fixed, versioned tactical corpus before changing the encoder. Each row
should preserve both the legal-information state and, when useful, a separate
hindsight branch. Required slices:

1. Age I economy versus unique-symbol denial.
2. Age II fourth/fifth-Wonder timing with an extra-turn Wonder preserved.
3. Sixth-Wonder races where the next build retires the opponent's Wonder.
4. Visible winning science cards with one or two coverers.
5. Positions immediately before and after a decisive reveal.
6. Exact late positions where every action loses.

Every threat row needs labels for:

- `preventable_now`
- `preventable_within_2_decisions`
- `final_known_coverer`
- `immediate_sixth_symbol_visible`
- oracle source and budget
- best-action value, runner-up value, and regret

For record 72, average deep-search pseudo-labels across the legal hidden Age III
pool and deal distribution. Because those labels use the same network at the
leaves, calibrate their reliability on later positions where the exact solver
can answer. Do not call them proofs.

The existing exact corpus is restricted to at most six tableau cards
(`endgame_corpus.py: MAX_PRESENT = 6`). It is valuable for late calibration but
cannot directly label the end-of-Age-II or move-55 decisions. Midgame forced-win
heads therefore cannot depend primarily on exact-solver coverage.

Report by slice:

- Policy top-1/top-k agreement and selected-action regret.
- Value Brier score and calibration.
- Joint winner/type precision and recall.
- Simulations required to reverse an incorrect raw or shallow value.
- Block rate split by preventable, unpreventable, and already-lost threats.

## One additive encoder experiment

Only if the corpus confirms a repeatable gap, make one encoder-version bump
containing the inexpensive deterministic candidates below. **Append every new
column to the end of its existing feature tuple.** `train.migrate_state_dict`
preserves a widened input projection by copying old columns and zero-padding
only trailing columns. Inserting a feature in the middle maps old weights to
the wrong semantics.

All of these additive columns can warm-start from `candidate_0085.pt`. Test
groups by masking their appended columns to zero rather than serializing
multiple schema bumps and cloud retrains.

### Group A: Wonder lifecycle and retirement tempo

Append global actor-relative features such as:

- `my_built_wonders`
- `opp_built_wonders`
- `total_built_wonders`
- `wonder_slots_remaining`
- `next_wonder_is_seventh`

Append per-Wonder features such as:

- one-hot `build_ordinal_if_built_now` for fifth/sixth/seventh
- `retires_if_opponent_builds_next`
- `retires_opponent_wonder_if_built_now`
- `is_only_remaining_retirement_candidate`

These are deterministic public facts and belong in the encoder, not a predicted
head. Their value is reduced sample complexity. The current game alone does not
prove that they improve play.

### Group B: action-side Age and extra-turn tempo

Append per-tableau-card facts:

- `take_ends_age`
- `present_cards_after_take`
- `newly_accessible_count`
- `actor_is_next_age_chooser_if_take_ends_age`

The exact feature `extra_turn_has_legal_follow_up` is action-pair dependent: it
depends on both the selected card and selected Wonder. If the current policy
head cannot consume pairwise action features cleanly, retain the existing
per-Wonder `grants_extra_turn` feature and expose the card-side facts above.
Represent the exact conjunction only in a later action-token design.

This group directly tests the user's end-of-Age concern. The current network
can learn that Sphinx and Mausoleum differ, but it must infer the realized
timing value by combining Wonder identity, parity, phase, and tableau state.

### Group C: tactical terminal scarcity and reveal control

Do not duplicate broad strategic-conduciveness features already ruled out by
the prior study. Make the first experiment symmetric across science and
military, and limit it to unmeasured tactical facts:

- `is_last_obtainable_copy_of_missing_symbol`
- `denies_opponent_last_obtainable_copy`
- `is_final_known_coverer_of_visible_opponent_win`
- probability of exposing an immediate sixth symbol, conditioned on the
  remaining hidden pool and card back
- expected newly exposed cards that add a new opponent symbol
- probability of exposing enough effective shields for an immediate military
  win
- short-horizon science and military `forced_in_k` and `preventable_within_k`

Probability features must be derived without conditioning on hidden identities
the player cannot know. Python/Rust parity tests should include both the feature
values and the remaining-pool calculation.

## Heads and training changes: conditional, not assumed

There should be no hand-coded switch from "maximize win percentage" to "deny
science." A correctly calibrated terminal win value already makes that switch:
a 30-point civilian lead is worth nothing in a science or military loss. In the
reviewed position, blocking the only live loss channel should dominate precisely
because it converts the otherwise likely civilian win into an actual win.

The failure is credit assignment and calibration, not the mathematical
objective. The greedy Mausoleum branch receives immediate economic/civilian
evidence, while the negative target arrives many decisions later through a rare
forced science line. More search eventually propagates that target in the public
late position, but the raw network does not.

The current outcome heads also disagree materially. They produced these raw win
probabilities on the captured game:

| Record | WDL win | Sum of my `joint7` victory modes |
| ---: | ---: | ---: |
| 13 | 69.78% | 72.68% |
| 72 | 56.65% | 64.03% |
| 84 | 74.34% | 82.42% |

At record 84, `joint7` assigned only 14.69% to opponent science despite the
public forced line. Simply replacing WDL with the current decomposed head would
therefore make this case worse.

First make the seven-way outcome head and WDL head probabilistically consistent:
the three "my win" modes should sum to WDL win, the three opponent modes to WDL
loss, and the draw modes should agree. This can be done with one authoritative
seven-way distribution from which WDL is derived, or with an explicit
consistency loss during a transition. Rebalance or focal-weight rare victory
modes only against held-out calibration and action-regret results. Once the
decomposed head is calibrated, deriving the search scalar from it gives science
and military losses a direct path into the value used by MCTS while preserving
the single objective of winning the game.

Counterfactual ranking examples are also more targeted than blanket science
oversampling. Pair a high-regret greedy action with a legal blocking action and
train their value ordering from exact or well-calibrated deep targets. Emphasize
positions where the threat is preventable; teaching an already-lost state to
look alarming does not teach the earlier save.

`joint7` and any other auxiliary head currently do not drive tree selection
directly. Search consumes policy priors and scalar value; an auxiliary head can
help only by shaping the shared trunk during training. Any claim that a head
"teaches urgency" must therefore be supported by an ablation showing improved
searched Q or action regret with the head enabled.

If the corpus still shows a gap after the additive encoder test, try one small
terminal-hazard head predicting science and military victory, per side, within
1, 2, 4, 8, and eventually remaining decisions. Use complete trajectories for
sampled labels and exact late states for calibration. Do not add a midgame
`forced_win` head whose supposed exact labels exist only after the relevant
decisions.

Use existing per-row `policy_weight` and `value_weight` plumbing to emphasize
high-regret, preventable tactical rows. This is cheaper and safer than adding a
new prioritized sampler. Weight decision relevance and regret, not every game
ending in science, to avoid teaching indiscriminate overblocking.

## Graph structure is a later experiment

Coordinates and coverer counts make topology recoverable but indirect. If the
corpus isolates a remaining topology gap after the additive feature experiment,
consider slot embeddings and relative-attention relations for `covers`,
`covered_by`, and `same_branch`.

This is deliberately last. New slot/attention parameters are not trailing input
columns; migration will zero them rather than preserve a neutral equivalent of
the current model. That makes the experiment more expensive and harder to
attribute than the additive groups above.

## Implementation order

1. Ship and monitor the canonical BGA Wonder/retirement fixes and parity tests.
2. Bank the reviewed legal-information states, the corrected-input search
   ladders, the Law-conditioned replay audit, and the prior 250-game
   science-threat report as stable artifacts.
3. Add science-pressure and military-pressure opponents to the training/eval
   league, with setup-stratified reporting and no imitation of specialist
   actions by the general model.
4. Build the tactical regression corpus with explicit preventability and
   victory-channel labels.
5. Settle record 72 across the hidden Age III pool/deal distribution and
   calibrate deep-search pseudo-labels against exact late states.
6. If a repeatable gap remains, land Groups A-C in one appended encoder bump,
   warm-start from `candidate_0085.pt`, and ablate by column masking.
7. Make WDL and `joint7` consistent; only then test a decomposed value as the
   authoritative search value.
8. Add the symmetric hazard head, counterfactual ranking, and row reweighting
   only if the corpus shows that the encoder/league experiments did not close
   the gap.
9. Test action tokens and graph relations last.

## Success criteria

The work succeeds when:

- BGA and self-play emit identical Wonder tokens for equivalent public states,
  including seven-built/one-retired positions.
- The legal-information corpus improves without materially reducing civilian
  or military slices.
- The Law-absent/no-Great-Library science-defense slice improves without
  indiscriminate green-card drafting, and the analogous military-pressure
  slice improves without general strength regression.
- A **public and unpreventable** forced-science line becomes strongly negative
  at raw or shallow-search depth; a preventable threat should instead value the
  blocking action highly, not make the whole state look lost.
- The same public/preventable distinction holds for forced military lines.
- Preventable visible sixth-symbol threats retain or improve the existing 95.7%
  shallow-search block rate.
- The model distinguishes intrinsic Wonder value from realized timing value:
  an extra-turn Wonder is rewarded when its follow-up or Age control is usable,
  and preserving it is valued when a retirement race makes that option scarce.
- Every auxiliary-head gain survives a searched-Q/action-regret ablation rather
  than appearing only in that head's own validation metric.
- WDL and decomposed outcome probabilities agree by construction or within a
  documented calibration tolerance.

## Relevant implementation locations

- `bga_extract.py`: canonical Wonder ownership and retired-Wonder recovery.
- `advisor_scrape.py`: public-observation determinization.
- `encoder.py`: `7wd-encoder-5` feature schemas; append-only requirement.
- `train.py`: per-row weights and additive input-projection migration.
- `endgame_corpus.py`: six-present-card exact-corpus limit.
- `runs/seven_wonders_duel/bga_game_log/table_907773062.jsonl`: captured source
  states used for the ladders; this ignored run artifact should be promoted to
  a stable, minimized fixture before it is relied on as a regression gate.
