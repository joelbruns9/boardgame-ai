# Review request: Kingdomino chance-search follow-ups

Please review commits `a8fc8f7` and `c9e3e27` on
`codex/kingdomino-chance-correct`, following the review of `f0a6db7`.
Self-play integration remains out of scope.

## `a8fc8f7` — measurement and paired-seed probing

This commit adds targeted position selection, paired multi-seed runs, clustered
modal consensus, and realized chance-support coverage diagnostics. It showed
that several original disagreements were search noise and, more importantly,
that materializing a support did not mean every outcome was evaluated.

Please focus on:

- seed pairing and position-level clustering;
- consensus and aggregation logic;
- visited-probability-mass and visits-per-outcome diagnostics; and
- whether repeated seeds are kept distinct from independent-position evidence.

Its reported strength comparisons are now superseded: they used the original
FPU-filled chance estimator corrected by the following commit.

## `c9e3e27` — search-semantics and throughput corrections

This commit responds to the blocking findings from the first review:

- unseen chance outcomes no longer receive FPU; registered probabilities are
  renormalized over outcomes with real evaluations;
- chance nodes receive a full-strength, separately tracked virtual-loss penalty;
- returned child Q uses the current chance estimate instead of stale snapshots;
- reference searches use a paired seed stream disjoint from candidate arms;
- panel mode/size and deck-size-stratified summaries are recorded;
- observation subtrees are allocated lazily, with four-tile row identities
  stored inline; and
- the disabled path has a frozen nontrivial golden-vector test.

Please focus on:

1. Whether visited-mass renormalization is sound while support remains partial.
2. Whether chance-node virtual loss has the correct player-0 framing without
   double-counting observation-node virtual loss.
3. Whether lazy routing by sorted four-tile row is information-set safe for the
   first-reveal-only topology.
4. Whether reported child Q and root value match the quantities used by search.
5. Whether any remaining support scans or allocations threaten eventual
   advisor/self-play throughput.

At one deck=8 root, lazy allocation registered 24,710 support outcomes but
allocated only 1,897 observation subtrees. Wall time remained GPU-dominated:
equal-compute treatment arms were approximately 0–8% slower than the incumbent
at 4,801 NN evaluations. This is a memory/allocation improvement, not yet a
demonstrated throughput win.

Verification: 10 Rust open-loop tests and 55 focused Python tests pass; one
unrelated test is skipped. No further strength experiment should rely on the
old results. After review, A1b will compare IID versus locally balanced chance
traversal and renormalized-Q versus sampled-value backup at matched NN compute.
