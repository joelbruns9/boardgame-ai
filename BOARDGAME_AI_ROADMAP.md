# Board Game AI — General Project Plan
## Goal: Turn two one-off game AIs into a reusable game-modeling lab, then add new games cheaply

---

## Strategic Core

> **The Kingdomino pipeline is the asset, not the Kingdomino model.** Engine → equivalence-tested encoder/codec → batched MCTS → self-play → Elo-gated promotion → Rust acceleration → exact endgame search is a proven playbook. The project now is to (1) extract the game-agnostic parts into a shared core, (2) pick new games that each add exactly one new modeling challenge, and (3) re-run the playbook per game with most of the code reused.

Each new game should be chosen so that ~80% of the stack is reuse and ~20% is a genuinely new capability (hidden information, >2 players, simultaneous moves, …). That keeps every game shippable in weeks while steadily growing what the framework can model.

---

## Guiding Principles (carried over — they earned their place)

- **No human game data** — pure self-play; heuristic/EV bots exist only as baselines and Elo anchors.
- **Correctness before throughput** — equivalence testing found real bugs in every Kingdomino milestone; every port/optimization gets a bit-exactness gate.
- **Measure before building** — benchmark before optimizing; don't Rust-port until the Python profile says game generation is the bottleneck.
- **Promotion gating with confidence** — checkpoints advance only by beating the prior best with statistical confidence.
- **Provenance everywhere** — every checkpoint traceable to code, config, and rules version.
- **The smoke run is the gate** — full-stack proof on small scale before any cloud spend.
- **One new challenge per game** — never take on two new modeling axes at once.

---

## Current Assets (what generalizes, what doesn't)

### From Kingdomino (the complete pipeline)
| Component | Files | Game-agnostic? |
|---|---|---|
| Rules engine + board | `board.py`, `game.py`, `dominoes.py` | No — per game |
| State encoder / action codec | `encoder.py`, `action_codec.py` | Contract yes, impl per game |
| AlphaZero MCTS (batched, open-loop chance) | `mcts_az.py`, `mcts.py` | **Mostly yes** |
| Network (conv trunk, policy/value/score heads) | `network.py` | Trunk per game, head structure yes |
| Self-play loop (batched slots, leaf coalescing) | `self_play.py`, `parallel_self_play.py`, `threaded_self_play.py` | **Mostly yes** |
| Elo ladder + anchors + resolve | `elo_rating.py`, `elo_anchors*.csv` | **Yes** (needs per-game namespacing) |
| Promotion gating + Hall of Fame | `promotion.py`, `hof.py` | **Yes** |
| Run provenance | `run_manifest.py` | **Yes** |
| Rust engine + MCTS (28× leaves/s) | `kingdomino_rust/` | Pattern yes, code per game |
| Exact endgame solver (YBW, TT, policy modes) | `endgame_solver.py` | Pattern yes, move-gen per game |
| Diagnostics / calibration | `diagnostics.py`, `margin_canary.py` | **Mostly yes** |
| Web advisor + BGA recon | `web_app.py`, browser extensions | Pattern yes, per game |

### From Can't Stop
- EV player (exact expectation over dice outcomes), MC rollout player — the template for **strong non-NN baselines**, which every new game needs as Elo anchors.
- Push-your-luck chance handling (large chance fan-out) — informs how the shared MCTS treats stochastic nodes.

### Known debts to fix during extraction
- Elo DB is Kingdomino-only at the repo root; needs `game` field / per-game DBs.
- Two self-play stacks (cantstop's `self_play_cantstop.py` vs kingdomino's) that don't share code.
- No formal `Game` interface — the contract exists only implicitly in what `mcts_az.py` calls.

---

## Phase 0 — Extract the Core (`core/` package)

Define the contracts by **refactoring Kingdomino onto them**, not by designing in the abstract. Kingdomino keeps working (its equivalence tests are the safety net); Can't Stop is optionally back-ported later as validation that the interface isn't Kingdomino-shaped.

### 0.1 The `Game` contract
```python
class Game(Protocol):
    num_players: int            # 2 for now; int, not hardcoded
    def legal_actions(self) -> list[int]        # indices into the codec space
    def apply(self, action: int) -> None
    def is_chance_node(self) -> bool            # chance handled explicitly
    def chance_outcomes(self) -> list[tuple[event, prob]]   # or sample()
    def current_player(self) -> int
    def is_terminal(self) -> bool
    def returns(self) -> np.ndarray             # length num_players (win/margin/score)
    def clone(self) -> Game
    def state_key(self) -> bytes                # for TT / transposition dedup
```
Design decisions locked in now (cheap now, expensive later):
- **`returns()` is a vector**, one entry per player — this is what makes 3–4 player games possible without touching MCTS backup later.
- **Chance nodes are first-class** (Kingdomino draws, Can't Stop dice). Open-loop MCTS stays the default for stochastic games.
- **Hidden information is out of the contract for now** — added in Phase 3 as an `information_state(player)` extension, not baked speculatively.

### 0.2 Encoder / Codec contracts
- `Encoder.encode(game, player) -> np.ndarray` — fixed shape per game, declared in a per-game `spec`.
- `ActionCodec.size`, `encode(action)->int`, `decode(int)->action`, canonical ordering documented per game (the symmetric-domino anchor ambiguity was a real bug — the canonical-order rule becomes part of the contract docs).
- Optional `Symmetries` provider for data augmentation (Kingdomino board symmetries, Can't Stop none).

### 0.3 Shared infrastructure moves to `core/`
- `core/mcts.py` — batched AZ-MCTS parameterized by the Game contract (open-loop chance, leaf coalescing, FPU, Dirichlet, vector-value backup).
- `core/self_play.py`, `core/train.py` — one training loop; per-game config files replace the giant CLI line in `WORKFLOW.md`.
- `core/elo.py` — ladder with `game` namespacing; anchors registered per game.
- `core/promotion.py`, `core/hof.py`, `core/run_manifest.py` — near-verbatim moves.
- Unified CLI: `python -m core.train --game kingdomino --config configs/kingdomino/48x6.yaml`.
- Per-game registry: `games/<name>/__init__.py` exposes `GAME_SPEC` (game class, encoder, codec, network factory, baseline bots, elo anchors).

### 0.4 Exit gate for Phase 0
- Kingdomino trains through the new CLI with **bit-identical** self-play trajectories vs the old path under a fixed seed (the project's own equivalence-testing standard applied to the refactor itself).
- Elo DB migrated with `game: kingdomino` on every record; `--resolve` reproduces current ratings.

---

## The Per-Game Playbook (repeatable milestone template)

This is the Kingdomino sequence, written down as the standard recipe. Each new game is an instance of this template; most games will skip steps (marked ⚑ optional).

| # | Milestone | Gate |
|---|---|---|
| G1 | **Rules engine** (pure Python, readable, slow is fine) | Rules-oracle tests; scripted full games match published rules / BGA replays |
| G2 | **Baselines**: random, greedy heuristic, and one *strong* handcrafted bot (EV/MC style) | Baselines beat random decisively; become Elo anchors |
| G3 | **Encoder + action codec** | Round-trip tests; canonical ordering documented; symmetry tests if applicable |
| G4 | **Hook into core MCTS + network** | MCTS with random net beats greedy baseline at high sims (sanity: search works) |
| G5 | **Smoke training run** (small net, local, few hours) | Elo climbs past strongest handcrafted baseline |
| G6 | **Elo ladder + promotion gating** for the new game | Anchored ladder; gated best-checkpoint |
| G7 | **Scale run** (bigger net, tuned schedule, ⚑ cloud) | New best with confidence; calibration diagnostics healthy |
| G8 | ⚑ **Rust port** — only if profiling shows generation-bound and more scale is wanted | Bit-exact equivalence suite (state, encoder, codec, MCTS), same as Kingdomino M1–M6 |
| G9 | ⚑ **Exact solver** for endgame/subgames where tractable | Solver-vs-search disagreement audit; solved values injected as training targets |
| G10 | ⚑ **Advisor / web app / BGA recon** | Live-play verification against real interface |

Expected effort per game after Phase 0: **G1–G6 in 1–3 weeks** of part-time work, because G4–G6 are configuration, not code.

---

## Choosing New Games: the Axis Map

Games are chosen by which *modeling axis* they add. Current coverage:

| Axis | Can't Stop | Kingdomino |
|---|---|---|
| Stochastic outcomes (chance nodes) | ✓ heavy | ✓ draw order |
| Push-your-luck / stopping decisions | ✓ | — |
| Spatial placement / grid encoding | — | ✓ |
| Drafting / turn-order manipulation | — | ✓ |
| Perfect information (given public state) | ✓ | ✓ |
| 2-player zero-sum-ish | ✓ | ✓ |

Uncovered axes, each a distinct framework capability:
1. **Deterministic perfect-info with adversarial depth** (no chance at all — pure search strength)
2. **Hidden information** (opponent hand) → determinization / IS-MCTS
3. **3–4 players** → vector value heads, non-zero-sum dynamics, kingmaking
4. **Simultaneous action selection** → joint-action or Nash-averaging at nodes
5. **Large/combinatorial action spaces** → action factorization
6. **Long-horizon engine building** → credit assignment over 20+ turns

---

## Recommended Game Progression

### Game 3: **Azul** — consolidation game (new axis: none on purpose; stretch: 3–4 players)
- Tile drafting from factories + pattern-line placement. Public information, modest action space (~180 actions), 2–4 players, huge BGA population for eventual advisor work.
- **Why first:** it is Kingdomino-shaped (drafting + spatial scoring) — the ideal first test that Phase 0's abstractions actually pay off. Run it 2-player through the existing stack essentially unchanged; then use it as the **vehicle for the 3-player extension** (vector returns are already in the contract; MCTS backup and network value head get their first real N>2 exercise).
- Strong heuristic baseline is easy (immediate-scoring greedy + floor-penalty avoidance), so the Elo ladder anchors well.
- Endgame is exactly solvable in the last 1–2 rounds → reuses the solver pattern.

### Game 4: **Schotten Totten / Battle Line** — new axis: hidden information
- 2-player card game: 60-card deck, hidden hands, poker-like formations on 9 flags. Small enough that the *game* is trivial; all the work is the new capability.
- Framework additions: `information_state(player)`, **determinized open-loop MCTS** first (sample opponent hands consistent with observations), IS-MCTS as the follow-up experiment. Belief-state input planes for the network (card-counting is learnable).
- This is deliberately the smallest respectable hidden-info game — the axis gets built without also fighting a big rules engine. (Lost Cities is the even-smaller alternative; Schotten Totten has more interesting inference.)

### Game 5: pick by appetite once 3–4p and hidden info both exist
- **Splendor** — long-horizon engine building, perfect info, 2–4p; tests credit assignment. Very tractable.
- **The Crew** — cooperative trick-taking with communication limits; hidden info + cooperation, a genuinely novel axis.
- **Race for the Galaxy / simultaneous-select game** — simultaneous moves axis; hardest, save for last.
- **Hive or Onitama** — if the deterministic-perfect-info axis appeals (pure search, great solver targets, Onitama is nearly solvable outright).

The rule stays: **one new axis per game.** Azul proves the extraction; Schotten Totten builds hidden info; game 5 is chosen against whichever axis is most interesting at that point.

---

## Phase Plan Summary

| Phase | Content | Exit gate |
|---|---|---|
| **0** | Extract `core/` from Kingdomino; contracts; per-game Elo namespacing; unified CLI | Kingdomino bit-identical through new path; Elo DB migrated |
| **1** | Azul 2p through the playbook (G1–G7) | Azul net beats strong heuristic with confidence on its own ladder |
| **2** | 3–4 player support, exercised on Azul (vector values, multiplayer Elo) | 3p Azul training run healthy; multiplayer Elo methodology documented |
| **3** | Hidden information (determinization → IS-MCTS), exercised on Schotten Totten | Net beats a card-counting heuristic baseline; determinization vs IS-MCTS ablation recorded |
| **4** | Game 5 by chosen axis; ⚑ Rust template crate if any game becomes generation-bound | Playbook doc updated with per-game deltas |
| **∞** | Can't Stop back-port onto `core/` (validation + revival), ongoing Kingdomino scale runs | — |

Phases 1+ can interleave with continued Kingdomino training runs — the ladder and runs/ convention already support parallel tracks.

---

## Repo Layout (target)

```
core/                    # game-agnostic: mcts, self_play, train, elo, promotion, hof, manifest, solver framework
configs/<game>/*.yaml    # training configs (replaces the WORKFLOW.md command template)
games/<game>/            # engine, encoder, codec, network factory, baselines, tests, <game>_rust/ if ported
runs/<game>/<run_name>/  # gitignored artifacts (existing convention, now per-game)
elo/<game>_db.json       # per-game ladders (or one DB with a game field)
docs/playbook.md         # the per-game milestone template, updated with each game's deltas
```

---

## Risks and Standing Decisions

- **Over-abstraction** is the main risk. Mitigation: the interface is whatever Kingdomino + Azul both need — nothing speculative. Hidden info, simultaneous moves, etc. enter the contract only in the phase that uses them.
- **Refactor breakage**: mitigated by the seeded bit-identical gate in Phase 0; the Kingdomino equivalence-test suite is the regression harness.
- **Multiplayer Elo** is genuinely different (pairwise model doesn't directly fit 3–4p results). Decision deferred to Phase 2; candidate: per-game BayesElo-style fit over finish positions, anchors still pairwise.
- **Network trunk per game**: conv trunk for spatial games (Kingdomino, Azul board), flat/transformer trunk for card games (Schotten Totten). The `GAME_SPEC` owns the network factory precisely so this stays a per-game choice.
- **Rust**: never port speculatively. Python-first per game; the Kingdomino crate becomes a template only when a game's scale run is provably generation-bound.

---

## Success Criteria

1. Adding a new 2-player perfect-info game requires **zero changes to `core/`** (Azul is the test).
2. Each game's trained net **beats its strongest handcrafted baseline** with promotion-gate confidence.
3. Kingdomino's training throughput and Elo trajectory are **unchanged** by the extraction.
4. By end of Phase 3 the framework demonstrably covers: chance nodes, spatial encoding, drafting, push-your-luck, 3–4 players, and hidden information.
