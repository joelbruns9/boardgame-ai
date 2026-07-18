# 7 Wonders Duel — Codec, Encoder, and Chance-Interface Specification

Phase A contract (AZ_PROJECT_PLAN.md §3). Everything downstream — `codec.py`,
`encoder.py`, `UnseenPool`, the buffer schema, and later the Rust port — implements
*this* document. Change the spec first, then the code.

Normative sources of truth:

- **Identity order**: `data.py` tuple order (`ALL_BUILDING_CARDS`, `WONDERS`,
  `PROGRESS_TOKENS`). Ids below are frozen; reordering `data.py` is a breaking change
  gated by the id tables in this file.
- **Legality**: `engine.legal_actions` / `game.legal_wonder_choices`. The codec never
  recomputes affordability, chains, or accessibility — it maps engine actions to
  indices and back. Mask exactness is *defined* as agreement with the engine.
- **Observation boundary**: `game.PlayerObservation`. The encoder and `UnseenPool`
  are pure functions of the observation (§5.1), never of hidden `GameState` fields.

---

## 1. Canonical identity tables

### 1.1 Card ids (0–72)

`card_id = index in ALL_BUILDING_CARDS` = Age I (0–22) + Age II (23–45) +
Age III (46–65) + Guilds (66–72).

| id | Age I | id | Age II | id | Age III / Guild |
|---|---|---|---|---|---|
| 0 | Lumber Yard | 23 | Sawmill | 46 | Arsenal |
| 1 | Logging Camp | 24 | Brickyard | 47 | Pretorium |
| 2 | Clay Pool | 25 | Shelf Quarry | 48 | Academy |
| 3 | Clay Pit | 26 | Glass-Blower | 49 | Study |
| 4 | Quarry | 27 | Drying Room | 50 | Chamber of Commerce |
| 5 | Stone Pit | 28 | Walls | 51 | Port |
| 6 | Glassworks | 29 | Forum | 52 | Armory |
| 7 | Press | 30 | Caravansery | 53 | Palace |
| 8 | Guard Tower | 31 | Customs House | 54 | Town Hall |
| 9 | Workshop | 32 | Courthouse | 55 | Obelisk |
| 10 | Apothecary | 33 | Horse Breeders | 56 | Fortifications |
| 11 | Stone Reserve | 34 | Barracks | 57 | Siege Workshop |
| 12 | Clay Reserve | 35 | Archery Range | 58 | Circus |
| 13 | Wood Reserve | 36 | Parade Ground | 59 | University |
| 14 | Stable | 37 | Library | 60 | Observatory |
| 15 | Garrison | 38 | Dispensary | 61 | Gardens |
| 16 | Palisade | 39 | School | 62 | Pantheon |
| 17 | Scriptorium | 40 | Laboratory | 63 | Senate |
| 18 | Pharmacist | 41 | Statue | 64 | Lighthouse |
| 19 | Theater | 42 | Temple | 65 | Arena |
| 20 | Altar | 43 | Aqueduct | 66 | Merchants Guild |
| 21 | Baths | 44 | Rostrum | 67 | Shipowners Guild |
| 22 | Tavern | 45 | Brewery | 68 | Builders Guild |
| | | | | 69 | Magistrates Guild |
| | | | | 70 | Scientists Guild |
| | | | | 71 | Moneylenders Guild |
| | | | | 72 | Tacticians Guild |

### 1.2 Wonder ids (0–11), `WONDERS` order

0 The Appian Way · 1 Circus Maximus · 2 The Colossus · 3 The Great Library ·
4 The Great Lighthouse · 5 The Hanging Gardens · 6 The Mausoleum · 7 Piraeus ·
8 The Pyramids · 9 The Sphinx · 10 The Statue of Zeus · 11 The Temple of Artemis

### 1.3 Progress-token ids (0–9), `PROGRESS_TOKENS` order

0 Agriculture · 1 Architecture · 2 Economy · 3 Law · 4 Masonry · 5 Mathematics ·
6 Philosophy · 7 Strategy · 8 Theology · 9 Urbanism

### 1.4 Back types

| id | Back | Public meaning |
|---|---|---|
| 0 | AGE_I | face-down Age I card |
| 1 | AGE_II | face-down Age II card |
| 2 | AGE_III | face-down Age III non-guild card |
| 3 | GUILD | face-down guild card |

Back type of every dealt slot is **public information** (physically visible card
backs). Age III slots holding a guild show back GUILD. This is an engine-contract
addition (§7): `PublicTableauCard` must carry `back: BackType` for present cards.

### 1.5 Resource order

Wherever a resource-indexed vector appears: `(wood, clay, stone, glass, papyrus)`
(`rules.Resource` order).

---

## 2. Actor-relative frame (hard requirement)

All policy indices, all encoder features, and all value/aux targets are expressed
from the perspective of **the player to act** ("actor"; for a pending choice, the
`pending_choice.player`). Consequences:

- The two city sections of the token schema have **identical dimensions and
  identical feature layout**; "mine" is always the actor's city, "theirs" always the
  opponent's. There is no player-0/player-1 anywhere in the encoding.
- Signed globals are actor-relative: military pawn position `m = +k` means the pawn
  is `k` steps toward the **opponent's** capital (good for actor). Concretely
  `m = conflict_position` if actor is the player favored by positive
  `conflict_position` in the engine, else `-conflict_position` (pin the engine sign
  convention in `codec.py` with a test).
- The NEXT_AGE_STARTER action block is `self` / `opponent`, not player 0/1; the
  codec converts using the actor at decode time.
- Gate (mirror test): construct state `S'` = `S` with cities swapped and
  `conflict_position` negated; assert `encode(S', actor=1−a) == encode(S, actor=a)`
  and that legal-mask index sets match exactly. Run over ≥1k sampled states.

---

## 3. Action codec

Fixed integer space, **N = 1202** actions, identity-indexed.

### 3.1 Layout

| Block | Range | Size | Index formula |
|---|---|---|---|
| WONDER_DRAFT | 0–11 | 12 | `wonder_id` |
| BUILD | 12–84 | 73 | `12 + card_id` |
| DISCARD | 85–157 | 73 | `85 + card_id` |
| CARD_TO_WONDER | 158–1033 | 876 | `158 + card_id*12 + wonder_id` |
| DESTROY | 1034–1106 | 73 | `1034 + card_id` |
| MAUSOLEUM_REVIVE | 1107–1179 | 73 | `1107 + card_id` |
| PROGRESS_BOARD | 1180–1189 | 10 | `1180 + token_id` |
| PROGRESS_LIBRARY | 1190–1199 | 10 | `1190 + token_id` |
| NEXT_AGE_STARTER | 1200–1201 | 2 | `1200` = self starts, `1201` = opponent starts |

### 3.2 Per-block semantics and masks

Masks are generated by mapping `engine.legal_actions(state)` through the encode
direction — the engine is the single source of truth. The rules below are the
*expected* mask content, used for the cross-check gate, not an independent
implementation.

- **WONDER_DRAFT** ↔ `Action(DRAFT_WONDER, wonder_name)`. Mask: current
  `wonder_offer`. Only legal in `Phase.WONDER_DRAFT`.
- **BUILD** ↔ `Action(slot_id, CONSTRUCT_BUILDING)`. Encode: card at `slot_id`
  (accessible ⇒ revealed ⇒ identity known; no duplicates ⇒ bijective). Decode: the
  unique accessible slot holding `card_id`. Mask: accessible ∧ affordable
  (chains included, via `minimum_payment`).
- **DISCARD** ↔ `Action(slot_id, DISCARD_FOR_COINS)`. Mask: accessible.
- **CARD_TO_WONDER** ↔ `Action(slot_id, CONSTRUCT_WONDER, wonder_name)`. Mask:
  accessible slots × actor's unbuilt, unretired, affordable wonders (wonder
  affordability is card-independent, so the mask is a Cartesian product).
- **DESTROY** ↔ `Action(RESOLVE_PENDING_CHOICE, choice=name)` under pending kind
  DESTROY_OPPONENT_BROWN or DESTROY_OPPONENT_GREY. One shared block: only one
  destroy pending can exist at a time; the pending kind (in the global token)
  disambiguates. Mask: `pending_choice.options`.
- **MAUSOLEUM_REVIVE** ↔ pending BUILD_FROM_DISCARD_FREE. Mask: discard pile.
- **PROGRESS_BOARD** ↔ pending CHOOSE_AVAILABLE_PROGRESS (science pair). Mask: the
  ≤5 tokens on the board.
- **PROGRESS_LIBRARY** ↔ pending CHOOSE_UNUSED_PROGRESS (Great Library). Mask: the
  3 tokens drawn by the GREAT_LIBRARY_DRAW chance event (§4). The draw itself is
  **not** an action.
- **NEXT_AGE_STARTER** ↔ `Action(CHOOSE_NEXT_START_PLAYER, starting_player)` with
  `starting_player = actor` for index 1200, `1 − actor` for 1201.

Chained builds need **no codec representation**: a chain build is an ordinary BUILD
action whose payment resolves to free (`Payment.used_chain`); the chain structure
lives in `data.py` (`chain_from`/`chain_to`) and surfaces in the mask (affordable
via chain) and in encoder features (§5.4). Urbanism's +4 is an engine effect,
invisible to the codec.

### 3.3 Gates

1. Round-trip: over ≥10k full games (seeded bots), every engine legal action encodes
   to exactly one index in [0, 1202) and decodes back to an equal `Action`.
2. Mask exactness: on ≥10k sampled states, `mask(state)` as an index set equals
   `{encode(a) for a in legal_actions(state)}`, and every masked index decodes to a
   legal action.
3. Mirror-mask gate (§2).

---

## 4. Chance-event contract and `UnseenPool`

### 4.1 `UnseenPool`

Computed **from an observation only**: for each back type, the set of card ids not
visible anywhere (tableau face-up, either city, discard pile, burials). Additionally
tracks the unseen wonder pool during draft round 0 (12 minus the revealed offer and
picks). The 5 off-board progress tokens are *deducible* (complement of the 5 board
tokens), so they are not "unseen" — but the Great Library draw over them is still
chance.

Consumers (single source of truth for all three):
1. Encoder unseen-pool summary tokens (§5.6).
2. Closed-loop chance children + exact probabilities.
3. Open-loop determinizer: sample a full consistent assignment (per-back-type
   permutation of pool over face-down slots + removed cards; guild backs constrain
   Age III).

### 4.2 Event kinds

| Kind | Trigger | Outcome space | Size | Probability |
|---|---|---|---|---|
| CARD_REVEAL | a face-down slot becomes accessible after a take | one card id from `pool(back)` | ≤ 11 | uniform `1/|pool|` (exchangeability: face-down-elsewhere vs removed is indistinguishable) |
| GREAT_LIBRARY_DRAW | Great Library built | unordered 3-subset of the 5 off-board tokens | **C(5,3) = 10** | uniform 1/10 |
| WONDER_GROUP_REVEAL | 4th pick of draft round 0 | unordered 4-subset of the 8 unseen wonders | C(8,4) = 70 | uniform 1/70 |
| AGE_DEAL | `start_next_age` (and initial Age I deal) | assignment of unseen cards to the new layout | not enumerable | closed mode: **sample k children**, keyed by observable signature (face-up identities + back-type pattern); open mode: per-descent sample |

Notes:
- Multiple cards uncovered by one take emit an **ordered list** of CARD_REVEAL
  events, canonical order ascending `(row, x)`. The searcher treats them as
  sequential chance nodes — never a joint-outcome cap hack.
- GREAT_LIBRARY_DRAW canonical outcome enumeration: subsets as sorted token-id
  triples, listed in lexicographic order. This ordering is part of the contract
  (Rust must match).
- WONDER_GROUP_REVEAL matters only for search *during* draft round 0 (8 of the
  game's decisions). 70 exceeds any per-node cap; closed mode may sample children
  like AGE_DEAL. After the flip, all 8 draft wonders are public and the 4 unused
  stay irrelevant forever.
- AGE_DEAL outcomes are equivalence classes of the observation: two assignments
  that put the same identities face-up with the same back pattern are the same
  child. Search across an age boundary normally defers to the NN at the boundary
  (plan §10); the event exists so open-loop determinization and boundary sampling
  are well-defined.

### 4.3 Engine API additions (the anti-leak contract)

The current engine resolves all chance from the state's seeded RNG **inside**
`apply_action` (Great Library: `rng.sample` in `_resolve_wonder_effects`;
reveals/deals: the deal locked at `GameState.new`). That is correct for the
simulator and for replay, but a searcher stepping a `clone()` would silently read
the true hidden future — `clone()` copies "the exact future RNG stream" by design.
Required additions:

1. `apply_action(state, action, *, chance_outcomes=None) -> StepResult`.
   `StepResult.events` is the ordered tuple of `ResolvedChance(kind, context,
   outcome)` that fired. When `chance_outcomes` is provided, events consume the
   supplied outcomes in order instead of touching state randomness.
2. `enumerate_chance(state, event_context) -> tuple[(outcome, prob), ...]` for the
   enumerable kinds, computed from `UnseenPool`.
3. **Search barrier**: a state flag (set on clones handed to search) under which
   any chance resolution *without* an explicit outcome raises. This makes leak-
   forward a hard error, not a silent bias. Simulator/self-play trajectory replay
   keeps the seeded path.
4. `resample_hidden(state, rng)`: re-randomize every hidden assignment of a
   state clone in place, preserving the visible projection exactly — the
   determinizer for open-loop mode. (Formulated on a clone rather than as
   `determinize(observation) -> GameState` reconstruction: search always starts
   from a clone, and moving only hidden entities gives the same marginals with
   far less machinery.)

### 4.4 Gates

- Statistical: sampled determinizations reproduce closed-loop marginals
  (chi-squared, ≥100k samples on fixed positions) — plan §A3 gate.
- Barrier: searcher stepping a barred clone through an uncovering without supplied
  outcomes raises; with outcomes, the resulting state matches the simulator when
  the supplied outcome equals the true one.
- Great Library: over seeded simulator games, empirical draw frequencies over the
  10 subsets are uniform (chi-squared); `enumerate_chance` lists exactly the 10.

---

## 5. Token schema (encoder)

Sequence of typed tokens, ≈ 60–110 depending on phase. Every token = learned
embedding(entity id) ⊕ its type's feature group. Feature *semantics* are fixed
here; exact offsets live in `encoder.py` and are pinned by an **encoder-signature
hash** (KD export discipline) checked by golden tests and, later, the Rust loader.

Scaling conventions: coin-like quantities encoded raw and `/10`; counts raw; all
flags 0/1; signed military features in [−9, 9] and `/9`.

### 5.1 Purity

The encoder consumes `(PlayerObservation, UnseenPool(observation))` only. Gate:
encode the same observation under ≥10 different consistent hidden assignments
(re-dealt states with identical visible projection) — outputs bit-identical.

### 5.2 Global token (1)

- Phase/decision type one-hot: draft, main turn, each `PendingChoiceKind`,
  next-age-starter, complete.
- Age one-hot (3); cards remaining in current age; face-down count remaining.
- Actor-relative military position `m` (signed, §2); distance-to-military-win pair
  `(9 − m, 9 + m)`; per-side military coin-loss tokens still in play (from
  `military_tokens_remaining`, folded to actor frame); per side, shields from the
  pawn to the next unclaimed token in that direction and that token's coin penalty
  (0/absent when none remain); pending shields.
- Extra-turn pending flag.
- Per-player (actor first, opponent second — identical sub-layout):
  coins; distinct science-symbol count and `6 − count`; per-symbol have-flags (7);
  color counts (7); unbuilt-wonder count; unbuilt extra-turn-wonder count;
  fixed-production vector (5); choice-producer count per resource (5);
  trade price per resource (5, from `minimum_payment` internals: 1 with reserve,
  else 2 + opponent fixed production); discard income (2 + yellow count);
  current score breakdown from `score_player` (blue VP, guild VP, wonder VP,
  progress VP, military VP, coin VP, total).
- Per-player **moonshot clocks** (actor then opponent — the threat/feasibility
  layer; all from public sets, loose upper bounds by design):
  - military: upper-bound additional shields still obtainable = shields on
    face-up present tableau cards + shields over the unseen pools of the current
    and future ages + that player's unbuilt wonder shields (+1/red-card headroom
    if Strategy is still obtainable); `military-win-feasible` flag = bound ≥
    distance-to-win;
  - science: count of symbols the player lacks that are still obtainable
    (face-up tableau, unseen pools, board progress tokens incl. Law, off-board
    tokens if their unbuilt wonders include the Great Library);
    `science-win-feasible` flag = have + obtainable-distinct ≥ 6.
  A dead flag is the license to ignore the threat / convert to civilian; a live
  flag with a short clock is what the value head keys blocking behavior on.

### 5.3 Wonder-draft tokens (variable, draft phase only)

One token per wonder in the current offer: wonder-id embedding + pick-round flag.
Plus the unseen-wonder pool summary (§5.6) during round 0.

### 5.4 Tableau slot tokens (≤ 20)

One per **present** card. Features:

- Face-up: card-id embedding. Face-down: back-type embedding (§1.4).
- Structural: row, x (normalized), face-up-layer flag, accessible flag, number of
  present coverers (0–2), number of face-down cards this card currently covers
  (uncover/chance exposure if taken).
- Computed, **for every face-up present card (not only accessible ones), and for
  both players symmetrically** (actor value then opponent value — the opponent
  column is the deny/hate-draft signal; covered face-up cards get it too because
  planning toward them is exactly the point):
  - affordable-to-build flag; minimum total coin outlay to build (from
    `minimum_payment`; 0 if chain-free);
  - chain-free flag;
  - completes-a-science-pair flag; grants-6th-symbol flag (instant win);
  - shields on card; would-cross-military-token flag; would-win-military flag.
- Chain icon ids (`chain_from` slot: which icon it needs; `chain_to`: which icon it
  grants) as small embeddings — OPTIONAL v1; card-id embeddings can learn this,
  the explicit feature just accelerates small nets.

### 5.5 City, wonder, progress, discard tokens

- **My city / opponent city**: one token per built card: card-id embedding + owner
  flag (actor/opponent). Identical layout both sides (§2).
- **Wonders** (8, all drafted): wonder-id embedding + owner + built flag +
  buried-card-id embedding when built (engine addition §7: burial mapping) +
  affordable-now flag and min coin outlay for its owner + grants-extra-turn flag.
- **Progress tokens**: one token per board token (≤5) + one per owned token (either
  side), with location one-hot (board / mine / theirs); when a PROGRESS_LIBRARY
  pick is pending, the 3 drawn get a candidate flag.
- **Discard pile**: one token per discarded card: card-id embedding + revive-
  candidate flag when Mausoleum pending.

### 5.6 Unseen-pool summary tokens (per back type with a non-empty pool)

Count + pooled embedding (mean of unseen card-id embeddings) per back type with a
non-empty pool; same for the unseen wonder pool during draft round 0. Fed from
`UnseenPool` (§4.1) — the same structure the chance layer uses.

**Future ages are included**: during Age I the AGE_II, AGE_III, and GUILD pools are
full (23 / 20-of-unknown-3-removed / 7-of-unknown-selection) and get summary tokens
too — they are how cross-age economic signal enters the encoding.

Per-pool cost aggregates, for actor and opponent symmetrically: mean and min
build cost over the pool's cards, each cost from `minimum_payment` against that
player's *current* production, trade discounts, chains, and reserves. This is the
explicit carrier of the resource-denial signal: buying a wood producer in Age I
immediately moves the opponent column of the Age III pool aggregates.

### 5.7 Calculated fields — rationale and scope

(KD lesson: computed features bought large sample-efficiency wins.) REQUIRED in v1:
everything listed in §5.2 and §5.4 — they encode the games' three race clocks
(military distance, science distance, VP totals) and the full economy (trade
prices, min payments, both-player affordability, pool cost aggregates). OPTIONAL /
v2 (add only on plateau evidence): chain icon embeddings (§5.4), per-guild
live-value estimates, "resource denial" price deltas (opponent price if actor
takes a producer first), and full **prospect tokens** — one token per unseen card
id with both-player costs (~40–55 extra tokens in Age I; the v1 pool aggregates
(§5.6) plus the 5-dim trade-price vectors (§5.2) carry most of that signal because
card cost factors through per-resource prices).
Nothing here may read hidden information — every computed feature is a function of
the observation (§5.1 gate covers them automatically). Note that unseen-card costs
leak nothing: pool *membership* is public; only the slot assignment is hidden.

### 5.8a Forward compatibility: adding token types mid-run

v2 items (prospect tokens especially) must be addable to a live run without a
restart. Two requirements make that true and are REQUIRED in v1:

1. **Extensible token typing**: every token carries a token-type embedding, and
   each type gets its own input projection. Adding a type later = one new type
   embedding + one new projection, **zero-initialized** so the net's output is
   bit-unchanged at switch-on (warm start from the current checkpoint, no strength
   cliff, SPRT-gated like any change). No other weight changes shape — the set
   transformer is length-agnostic.
2. **Buffer stores no encodings** (already the A4 rule): training re-encodes
   replayed states, so the whole existing buffer is available in the new schema
   immediately — no migration, no mixed-format examples.

Each addition bumps the encoder-signature version (§5). Timing note: after
Phase F, any new token type must also land in the Rust encoder behind a fresh
bit-exactness gate — if evidence for prospect tokens exists before the Rust
encoder port, land them first.

### 5.9 Deliberately excluded features

- **`first_player` flag**: not encoded, in any phase. Given the actor-relative
  frame, everything it carries is derivable where it matters — draft order from
  pick counts + actor, in-age tempo from cards-remaining parity + actor, age-start
  from the loser-chooses rule — and as a bare input it is a pure prior: the value
  head would learn "seat 2 ⇒ losing" as a calibration crutch (ZeusAI measured
  ~66% first-player win rate) without any state-causal content. Second-player
  aggression is learned from the tempo features plus search, not from a seat
  label. Corollary: self-play and eval MUST alternate seats (already standard).
- Player identities 0/1 anywhere (§2).

### 5.8 Gates

Purity (§5.1); mirror symmetry (§2); determinism (same observation ⇒ identical
bytes); golden files on ≥3 scripted games covering draft, all pending kinds, Great
Library, age boundaries, and endgame; encoder-signature hash stable across
refactors unless the spec version bumps.

---

## 6. Buffer record format (A4)

JSONL, one record per game. Bit-exact replay from `(setup.seed, actions)` is the
defining invariant — anything derivable is recoverable (reanalyze, exact
relabeling, trap harvesting).

```json
{
  "schema": 1,
  "spec_version": "codec-1",
  "setup": {"seed": 123, "first_player": 0},
  "agents": {"p0": "run3/iter_0007", "p1": "hof/iter_0004"},
  "result": {"winner": 0, "victory_type": "civilian", "scores": [61, 55]},
  "chance_log": [{"kind": "CARD_REVEAL", "outcome": 41}, ...],
  "moves": [
    {
      "i": 0, "actor": 0, "action": 3,
      "mask_hash": "xxh64:...",
      "visits": {"3": 40, "7": 18, "11": 6},
      "root_value": 0.132, "sims": 64,
      "mode": "closed", "gumbel_topk": [3, 7, 11, 0]
    }
  ]
}
```

- `action` is the codec index (§3). `visits` is sparse, index → count; policy
  targets are derived, not stored.
- `chance_log` is the ordered list of every resolved chance event in trajectory
  order. It is *redundant* given the seed (the simulator RNG is deterministic) and
  that redundancy is the point: the replay gate cross-checks it, so any change to
  engine RNG consumption is caught instead of silently corrupting old buffers.
- Seeded/bot games used for buffer seeding use the same schema with
  `sims: 0, visits: {}` and a `"policy_excluded": true` flag after iteration ~10
  (plan §6 seeding rules).

Gates: replay reproduces every `mask_hash` and the final `setup_fingerprint`-style
state hash; `chance_log` matches; a record survives JSON round-trip byte-stably.

---

## 7. Engine contract additions (implementation checklist)

Gaps found auditing `game.py` / `engine.py` against this spec, in build order.
Items 1–5 SHIPPED (see `test_chance.py`); item 6 lands with `codec.py`.

1. ✅ `PublicTableauCard.back: BackType` — expose the (public) back type of
   present cards (§1.4). `BackType` + `back_type_of` live in `data.py`, along
   with the canonical `CARD_IDS` / `WONDER_IDS` / `PROGRESS_IDS` maps (§1).
2. ✅ Burial mapping: `GameState.wonder_burials` (wonder → buried card) +
   `PlayerObservation.wonder_burials`, alongside the legacy `buried_cards` list.
3. ✅ Great Library refactor: GREAT_LIBRARY_DRAW chance event; options
   canonically sorted by token id; simulator path keeps the seeded `rng.sample`
   stream for replay compatibility (§4.2, §4.3).
4. ✅ `apply_action(..., chance_outcomes=None) -> StepResult` with
   `ResolvedChance` events (CARD_REVEAL, GREAT_LIBRARY_DRAW,
   WONDER_GROUP_REVEAL, AGE_DEAL), supplied-outcome override with
   hidden-card **swap** (world stays consistent — no duplicate/lost cards), and
   `GameState.search_barrier` raising `HiddenInformationError` (§4.3).
5. ✅ `pool.py`: `UnseenPool` from observation only, `enumerate_card_reveal` /
   `enumerate_great_library` / `enumerate_wonder_flip` in canonical order, and
   `resample_hidden(state, rng)` (§4.1, §4.3).
6. ⬜ NEXT_AGE_STARTER stays engine-side as absolute player ids; actor-relative
   conversion is codec-only (§3.2) — no engine change, just a pinned test.

---

## 8. Gate checklist (Phase A exit)

- [ ] Codec round-trip over ≥10k full games (§3.3.1)
- [ ] Mask exactness on ≥10k states (§3.3.2)
- [ ] Mirror gates: encoding + mask (§2)
- [ ] Encoder purity under re-dealt hidden assignments (§5.1)
- [ ] Encoder determinism + golden files + signature hash (§5.8)
- [ ] Chance marginals: determinizer vs closed-loop enumeration, chi-squared (§4.4)
- [ ] Great Library: exactly 10 uniform outcomes, event-driven (§4.4)
- [ ] Search barrier raises on unresolved chance (§4.4)
- [ ] Buffer replay: masks, chance log, final state hash (§6)
