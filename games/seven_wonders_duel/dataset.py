"""Buffer → training-tensor bridge (plan §4 Phase B).

Replays buffer records through the engine, encodes every decision state with
the Phase A encoder, and emits actor-relative training targets. Encodings are
computed here at training time — the buffer stores none (spec §5.8a) — so any
encoder schema change applies to the whole buffer immediately.

Targets (all actor-relative, §2):
- policy: the action played (one-hot CE for bot games) or the recorded visit
  distribution when present; ``policy_excluded`` moves emit no policy target.
- value: W/D/L 3-way from the final result.
- joint7: winner × victory type (my civ/sci/mil, opp civ/sci/mil, draw).
- margin: final civilian score difference (valid only for civilian endings).
- military_final: final pawn position, actor-relative, /9.
- sci_final my/opp: final distinct science symbol counts, /6.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from .buffer import (
    LEGACY_DIGEST_VERSION,
    LOGIC_DIGEST_VERSION,
    SPEC_VERSION,
    GameRecord,
    ReplayMismatchError,
    StaleSpecVersionError,
    archive_policy_seats,
    legal_mask_hash,
    replay,
)
from .codec import NUM_ACTIONS, legal_action_indices
from .data import CARD_IDS, PROGRESS_IDS, WONDER_IDS
from .encoder import _FEATURE_COUNTS, _SCHEMA, Encoding, TokenType, encode
from .engine import _science_symbols
from .game import VictoryType

TOKEN_TYPES = tuple(TokenType)
TYPE_IDS = {token_type: index for index, token_type in enumerate(TOKEN_TYPES)}
FEATURE_COUNTS = tuple(_FEATURE_COUNTS[t] for t in TOKEN_TYPES)
MAX_FEATURES = max(FEATURE_COUNTS)
ENTITY_SPACES = tuple(_SCHEMA["entity_spaces"][t.value] for t in TOKEN_TYPES)
NUM_AUX_CARDS = 74  # 73 card ids + index 0 reserved for "no aux entity"

JOINT7_CLASSES = (
    "my_civilian",
    "my_scientific",
    "my_military",
    "opp_civilian",
    "opp_scientific",
    "opp_military",
    "draw",
)

_VICTORY_OFFSET = {
    VictoryType.CIVILIAN: 0,
    VictoryType.SCIENTIFIC: 1,
    VictoryType.MILITARY: 2,
}


@dataclass(frozen=True, slots=True)
class GameDerivationStats:
    """Final-state observations collected during verified example derivation."""

    ending_age: int
    max_absolute_track: int
    sixth_science_symbol: bool
    progress_tokens: int
    science_pairs: int
    military_tokens_triggered: int
    military_gold_pillaged: int
    wonders_built: int
    wonders_discarded: int


@dataclass(frozen=True, slots=True)
class Example:
    type_ids: np.ndarray  # [T] int8
    entity_ids: np.ndarray  # [T] int16
    aux_ids: np.ndarray  # [T] int16 (0 = none, else card_id + 1)
    features: np.ndarray  # [T, MAX_FEATURES] float16 (exact for the integer-
    # valued raw features < 2048; halves the materialized footprint. True
    # streaming replaces eager materialization in the Phase D loop extraction.)
    legal: np.ndarray  # [L] int16 legal action indices
    policy_target: np.ndarray  # [L] float32 distribution over legal (sums to 1)
    has_policy: bool
    value_class: int  # 0 win / 1 draw / 2 loss
    joint7_class: int
    margin: float
    margin_valid: bool
    military_final: float
    sci_final_my: float
    sci_final_opp: float
    game_key: int  # for game-honest splits
    iteration: int | None  # for iteration-honest splits (Phase D)
    #: The search's backed-up root value at this position, actor-relative in
    #: [-1, +1], or None for rows without a search (bot moves).
    #:
    #: `value_class` is a per-GAME label: all ~16.5 rows of a game carry the same
    #: outcome, which the network can fit by recognising the game and emitting
    #: its result. `root_value` differs at every position -- early ones are
    #: uncertain, late ones decided -- so blending it in makes the target
    #: position-specific and removes that shortcut. See `--value-bootstrap`.
    root_value: float | None = None

    def __post_init__(self) -> None:
        """Make the arrays read-only as well as the fields.

        `frozen=True` stops the *attributes* being rebound, but a numpy array
        reached through a frozen attribute is still writable, so freezing alone
        would leave `example.features[0, 0] = 1.0` working. That matters because
        Phase D's example cache hands the same object to every iteration whose
        replay window contains the game: an in-place write would propagate
        silently into every later iteration.

        Every array here is freshly allocated during derivation (`vectorize`,
        `np.zeros`, `np.asarray` over a new list), so nothing else holds a
        writable alias to them.
        """

        for field in ("type_ids", "entity_ids", "aux_ids", "features", "legal",
                      "policy_target"):
            getattr(self, field).setflags(write=False)


def vectorize(encoding: Encoding) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(encoding.tokens)
    type_ids = np.empty(count, dtype=np.int8)
    entity_ids = np.empty(count, dtype=np.int16)
    aux_ids = np.empty(count, dtype=np.int16)
    features = np.zeros((count, MAX_FEATURES), dtype=np.float16)
    for row, token in enumerate(encoding.tokens):
        type_ids[row] = TYPE_IDS[token.type]
        entity_ids[row] = token.entity_id
        aux_ids[row] = token.aux_id + 1  # -1 (none) -> 0
        values = token.features
        features[row, : len(values)] = values
    return type_ids, entity_ids, aux_ids, features


def _actor_value_class(winner: int | None, actor: int) -> int:
    if winner is None:
        return 1
    return 0 if winner == actor else 2


def _joint7_class(winner: int | None, victory: VictoryType | None, actor: int) -> int:
    if winner is None:
        return 6
    if victory is None or victory is VictoryType.SHARED_CIVILIAN:
        return 6
    offset = _VICTORY_OFFSET[victory]
    return offset if winner == actor else 3 + offset


def is_fast_search_move(move) -> bool:
    """A cheap-search move: searched, but below the full simulation range.

    ``policy_excluded`` alone conflates two different things -- a cheap search
    and a curriculum bot's move. They are distinguished by ``sims``: a bot move
    records none. Only the cheap searches are what KataGo's playout cap
    randomization drops; bot moves are a curriculum device whose value labels
    stay useful.
    """

    return move.policy_excluded and move.sims > 0


def _policy_for_move(move, legal: np.ndarray) -> np.ndarray:
    """Build the sparse-on-legal policy label shared by both derivation paths."""

    policy = np.zeros(len(legal), dtype=np.float32)
    index_of = {int(action): index for index, action in enumerate(legal)}
    if move.policy_target:
        for action, probability in move.policy_target.items():
            if int(action) not in index_of:
                raise ValueError(
                    f"move {move.i}: policy target on illegal action {action}"
                )
            policy[index_of[int(action)]] = probability
        total = float(policy.sum())
        if not 0.999 <= total <= 1.001:
            raise ValueError(f"move {move.i}: policy target sums to {total:.4f}")
        policy /= total
    elif move.visits:
        total = float(sum(move.visits.values()))
        if total <= 0:
            raise ValueError(f"move {move.i}: visit counts sum to zero")
        for action, visits in move.visits.items():
            if int(action) not in index_of:
                raise ValueError(f"move {move.i}: visit on illegal action {action}")
            policy[index_of[int(action)]] = visits / total
    else:
        position = int(np.searchsorted(legal, move.action))
        if position >= len(legal) or int(legal[position]) != move.action:
            raise ValueError(f"move {move.i}: played action is illegal")
        policy[position] = 1.0
    return policy


def examples_from_record(
    record: GameRecord,
    *,
    record_fast_moves: bool = False,
    on_derived: Callable[[GameDerivationStats], None] | None = None,
) -> list[Example]:
    """Replay one game (through the VERIFIED buffer.replay path — mask hashes,
    actors, chance log, trajectory and final digests all checked) and emit an
    example per recorded decision. A stale or tampered buffer raises
    ReplayMismatchError instead of silently training on regenerated states.

    Fast-search moves are replayed but emit no example unless
    ``record_fast_moves``. They cannot be dropped from the *record* -- the
    buffer's defining invariant is that ``replay(record)`` reproduces every
    state from ``(seed, actions)``, so removing a move would break replay for
    every later move in the game. Exclusion belongs here, at the example
    boundary.

    Wu (2020) §3.1: "Only turns with a full search are recorded for training."
    The value target is limited by one noisy result per *game*, so additional
    positions from the same game share that label and add little while
    inflating the buffer. ``on_derived`` receives the W3 observations from this
    same verified replay, avoiding a second stats-only traversal.
    """

    staged: list[tuple[Example, int]] = []  # (example, actor)
    maximum_track = 0
    # Empty except in league games. The archive's seat still contributes value
    # labels as an explicit mixed-policy experiment; only its policy is masked.
    archive_seats = archive_policy_seats(record.agents)

    def featurize(game, move):
        nonlocal maximum_track
        maximum_track = max(maximum_track, abs(game.conflict_position))
        # Replay still steps through this move -- only the example is dropped.
        if not record_fast_moves and is_fast_search_move(move):
            return
        actor = (
            game.pending_choice.player
            if game.pending_choice is not None
            else game.active_player
        )
        encoding = encode(game.observation(actor))
        if encoding.actor != actor:
            raise AssertionError("encoder actor disagrees with replay actor")
        legal = np.asarray(legal_action_indices(game), dtype=np.int16)
        policy = _policy_for_move(move, legal)
        type_ids, entity_ids, aux_ids, features = vectorize(encoding)
        # Inputs only: the outcome labels are unknown until the replay finishes,
        # so the Example is built once, complete, below. It used to be built here
        # with placeholder labels and back-filled in place; `Example` is frozen
        # now that the Phase D cache hands the same object to every iteration
        # whose replay window contains the game.
        staged.append(
            (
                type_ids,
                entity_ids,
                aux_ids,
                features,
                legal,
                policy,
                not move.policy_excluded and actor not in archive_seats,
                actor,
                move.root_value,
            )
        )

    game = replay(record, on_state=featurize)
    maximum_track = max(maximum_track, abs(game.conflict_position))
    final_position = game.conflict_position
    sci_counts = (len(_science_symbols(game, 0)), len(_science_symbols(game, 1)))
    if on_derived is not None:
        on_derived(
            GameDerivationStats(
                ending_age=game.age,
                max_absolute_track=maximum_track,
                sixth_science_symbol=max(sci_counts) >= 6,
                progress_tokens=sum(
                    len(city.progress_tokens) for city in game.cities
                ),
                science_pairs=sum(
                    len(city.claimed_science_pairs) for city in game.cities
                ),
                military_tokens_triggered=4
                - len(game.military_tokens_remaining),
                military_gold_pillaged=14
                - sum(game.military_tokens_remaining.values()),
                wonders_built=sum(
                    len(city.built_wonders) for city in game.cities
                ),
                wonders_discarded=len(game.retired_wonders),
            )
        )
    examples: list[Example] = []
    for (
        type_ids,
        entity_ids,
        aux_ids,
        features,
        legal,
        policy,
        has_policy,
        actor,
        root_value,
    ) in staged:
        if game.final_scores is not None:
            mine, theirs = game.final_scores[actor], game.final_scores[1 - actor]
            margin, margin_valid = (mine - theirs) / 20.0, True
        else:
            margin, margin_valid = 0.0, False
        rel = final_position if actor == 0 else -final_position
        examples.append(
            Example(
                type_ids=type_ids,
                entity_ids=entity_ids,
                aux_ids=aux_ids,
                features=features,
                legal=legal,
                policy_target=policy,
                has_policy=has_policy,
                value_class=_actor_value_class(game.winner, actor),
                root_value=root_value,
                joint7_class=_joint7_class(game.winner, game.victory_type, actor),
                margin=margin,
                margin_valid=margin_valid,
                military_final=rel / 9.0,
                sci_final_my=sci_counts[actor] / 6.0,
                sci_final_opp=sci_counts[1 - actor] / 6.0,
                game_key=record.seed,
                iteration=record.iteration,
            )
        )
    return examples


_CHANCE_KIND_IDS = {
    "card_reveal": 0,
    "great_library_draw": 1,
    "wonder_group_reveal": 2,
    "age_deal": 3,
}
_VICTORY_IDS = {
    "military": 0,
    "scientific": 1,
    "civilian": 2,
    "shared_civilian": 3,
}


def _rust_chance_log(record: GameRecord) -> list[tuple[int, list[int]]]:
    converted: list[tuple[int, list[int]]] = []
    for kind, outcome in record.chance_log:
        try:
            kind_id = _CHANCE_KIND_IDS[kind]
        except KeyError as error:
            raise ReplayMismatchError(f"unknown chance kind {kind!r}") from error
        names = list(outcome) if isinstance(outcome, tuple) else [outcome]
        ids = {
            0: CARD_IDS,
            1: PROGRESS_IDS,
            2: WONDER_IDS,
            3: CARD_IDS,
        }[kind_id]
        try:
            converted.append((kind_id, [ids[name] for name in names]))
        except KeyError as error:
            raise ReplayMismatchError(
                f"unknown {kind} outcome {error.args[0]!r}"
            ) from error
    return converted


def _examples_from_rust_payload(
    record: GameRecord, payload: dict
) -> tuple[list[Example], GameDerivationStats]:
    """Turn one packed Rust replay into the exact public ``Example`` objects."""

    if int(payload["feature_width"]) != MAX_FEATURES:
        raise AssertionError("Rust/Python feature width differs")
    moves = record.moves
    move_offsets = np.frombuffer(payload["move_legal_offsets"], dtype="<u4")
    move_legal = np.frombuffer(payload["move_legal_actions"], dtype="<i2")
    move_actors = np.frombuffer(payload["move_actors"], dtype=np.uint8)
    if len(move_offsets) != len(moves) + 1 or len(move_actors) != len(moves):
        raise ReplayMismatchError("Rust replay move count differs from record")
    for index, move in enumerate(moves):
        if move.i != index:
            raise ReplayMismatchError(
                f"non-contiguous recorded move index {move.i} at position {index}"
            )
        start, stop = int(move_offsets[index]), int(move_offsets[index + 1])
        legal = move_legal[start:stop]
        current_hash = legal_mask_hash([int(action) for action in legal])
        if current_hash != move.mask_hash:
            raise ReplayMismatchError(
                f"move {move.i}: mask hash {current_hash} != recorded {move.mask_hash}"
            )
        if int(move_actors[index]) != move.actor:
            raise ReplayMismatchError(
                f"move {move.i}: actor {int(move_actors[index])} != recorded {move.actor}"
            )

    token_offsets = np.frombuffer(payload["token_offsets"], dtype="<u4")
    row_moves = np.frombuffer(payload["row_move_indices"], dtype="<u4")
    type_ids = np.frombuffer(payload["type_ids"], dtype=np.int8)
    entity_ids = np.frombuffer(payload["entity_ids"], dtype="<i2")
    aux_ids = np.frombuffer(payload["aux_ids"], dtype="<i2")
    features_f64 = np.frombuffer(payload["features_f64"], dtype="<f8")
    if len(token_offsets) != len(row_moves) + 1:
        raise ReplayMismatchError("Rust replay row offsets are not aligned")
    if features_f64.size != type_ids.size * MAX_FEATURES:
        raise ReplayMismatchError("Rust replay feature payload has the wrong size")
    features = features_f64.reshape(-1, MAX_FEATURES).astype(np.float16)
    archive_seats = archive_policy_seats(record.agents)
    stats_payload = payload["stats"]
    final_position = int(stats_payload["final_conflict_position"])
    science_counts = (
        int(stats_payload["science_count_0"]),
        int(stats_payload["science_count_1"]),
    )
    victory = VictoryType(record.victory_type) if record.victory_type else None

    examples: list[Example] = []
    for row, move_index_raw in enumerate(row_moves):
        move_index = int(move_index_raw)
        if move_index >= len(moves):
            raise ReplayMismatchError(f"Rust replay returned bad move {move_index}")
        move = moves[move_index]
        actor = move.actor
        token_start = int(token_offsets[row])
        token_stop = int(token_offsets[row + 1])
        legal_start = int(move_offsets[move_index])
        legal_stop = int(move_offsets[move_index + 1])
        legal = move_legal[legal_start:legal_stop]
        policy = _policy_for_move(move, legal)
        if record.scores is not None:
            mine, theirs = record.scores[actor], record.scores[1 - actor]
            margin, margin_valid = (mine - theirs) / 20.0, True
        else:
            margin, margin_valid = 0.0, False
        relative_position = final_position if actor == 0 else -final_position
        examples.append(
            Example(
                type_ids=type_ids[token_start:token_stop],
                entity_ids=entity_ids[token_start:token_stop],
                aux_ids=aux_ids[token_start:token_stop],
                features=features[token_start:token_stop],
                legal=legal,
                policy_target=policy,
                has_policy=not move.policy_excluded and actor not in archive_seats,
                value_class=_actor_value_class(record.winner, actor),
                root_value=move.root_value,
                joint7_class=_joint7_class(record.winner, victory, actor),
                margin=margin,
                margin_valid=margin_valid,
                military_final=relative_position / 9.0,
                sci_final_my=science_counts[actor] / 6.0,
                sci_final_opp=science_counts[1 - actor] / 6.0,
                game_key=record.seed,
                iteration=record.iteration,
            )
        )

    stats = GameDerivationStats(
        ending_age=int(stats_payload["ending_age"]),
        max_absolute_track=int(stats_payload["max_absolute_track"]),
        sixth_science_symbol=bool(stats_payload["sixth_science_symbol"]),
        progress_tokens=int(stats_payload["progress_tokens"]),
        science_pairs=int(stats_payload["science_pairs"]),
        military_tokens_triggered=int(stats_payload["military_tokens_triggered"]),
        military_gold_pillaged=int(stats_payload["military_gold_pillaged"]),
        wonders_built=int(stats_payload["wonders_built"]),
        wonders_discarded=int(stats_payload["wonders_discarded"]),
    )
    return examples, stats


def derive_records_rust(
    records: list[GameRecord],
    *,
    record_fast_moves: bool = False,
    batch_games: int = 32,
) -> list[tuple[list[Example], GameDerivationStats]]:
    """Rust-default replay/encode path, aligned one result per input game.

    The small game batch bounds transient f64 packing memory during a cold
    40k-game buffer rebuild.  NumPy converts each batch to the durable float16
    representation before the next batch is submitted.

    Rust validates every action/legal mask, actor, resolved chance outcome, game
    completion, recorded result, and current language-neutral digest. Legacy
    records retain RNG-inclusive digests that Rust cannot reproduce, so the Rust
    path validates their structural witnesses but not those two stored digest
    fields. Select ``derive_backend='python'`` for an explicit legacy digest
    audit; warm imports and resumes otherwise remain on the production Rust
    path.
    """

    try:
        import seven_wonders_rust as swr
    except ImportError as error:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "seven_wonders_rust is not installed; use derive_backend='python' "
            "or run maturin develop"
        ) from error
    from .rust_bridge import rust_game_for_record

    if batch_games < 1:
        raise ValueError("batch_games must be positive")
    output: list[tuple[list[Example], GameDerivationStats]] = []
    for batch_start in range(0, len(records), batch_games):
        batch = records[batch_start : batch_start + batch_games]
        for record in batch:
            if record.spec_version != SPEC_VERSION:
                raise StaleSpecVersionError(
                    f"record was written under spec {record.spec_version!r}, this "
                    f"engine is {SPEC_VERSION!r}"
                )
        games = [rust_game_for_record(record) for record in batch]
        actions = [[move.action for move in record.moves] for record in batch]
        actors = [[move.actor for move in record.moves] for record in batch]
        include = [
            [record_fast_moves or not is_fast_search_move(move) for move in record.moves]
            for record in batch
        ]
        chance_logs = [_rust_chance_log(record) for record in batch]
        expected_results = [
            (
                record.winner,
                _VICTORY_IDS[record.victory_type]
                if record.victory_type is not None
                else None,
                record.scores,
            )
            for record in batch
        ]
        expected_digests = []
        for record in batch:
            if record.digest_version == LOGIC_DIGEST_VERSION:
                expected_digests.append(
                    (record.final_digest, record.trajectory_digest)
                )
            elif record.digest_version == LEGACY_DIGEST_VERSION:
                expected_digests.append((None, None))
            else:
                raise ValueError(
                    "unsupported buffer digest version: "
                    f"{record.digest_version!r}"
                )
        try:
            payloads = swr.derive_records(
                games,
                actions,
                actors,
                include,
                chance_logs,
                expected_results,
                expected_digests,
            )
        except ValueError as error:
            raise ReplayMismatchError(str(error)) from error
        if len(payloads) != len(batch):
            raise ReplayMismatchError("Rust derivation returned the wrong game count")
        output.extend(
            _examples_from_rust_payload(record, payload)
            for record, payload in zip(batch, payloads)
        )
    return output


def examples_from_records(records, *, record_fast_moves: bool = False) -> list[Example]:
    out: list[Example] = []
    for record in records:
        out.extend(examples_from_record(record, record_fast_moves=record_fast_moves))
    return out


def collate_inputs(
    vectorized: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    legal_lists: list,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Inputs-only collate for inference: same input keys as :func:`collate`
    (type_ids, entity_ids, aux_ids, features, pad_mask, legal_mask), no
    targets. ``vectorized`` entries come from :func:`vectorize`."""

    size = len(vectorized)
    max_tokens = max(len(v[0]) for v in vectorized)
    type_ids = torch.zeros(size, max_tokens, dtype=torch.long)
    entity_ids = torch.zeros(size, max_tokens, dtype=torch.long)
    aux_ids = torch.zeros(size, max_tokens, dtype=torch.long)
    features = torch.zeros(size, max_tokens, MAX_FEATURES)
    pad_mask = torch.ones(size, max_tokens, dtype=torch.bool)
    legal_mask = torch.zeros(size, NUM_ACTIONS, dtype=torch.bool)
    for row, ((types, entities, auxes, feats), legal) in enumerate(
        zip(vectorized, legal_lists)
    ):
        count = len(types)
        type_ids[row, :count] = torch.from_numpy(types.astype(np.int64))
        entity_ids[row, :count] = torch.from_numpy(entities.astype(np.int64))
        aux_ids[row, :count] = torch.from_numpy(auxes.astype(np.int64))
        features[row, :count] = torch.from_numpy(feats.astype(np.float32))
        pad_mask[row, :count] = False
        legal_mask[row, torch.as_tensor(list(legal), dtype=torch.long)] = True
    tensors = {
        "type_ids": type_ids,
        "entity_ids": entity_ids,
        "aux_ids": aux_ids,
        "features": features,
        "pad_mask": pad_mask,
        "legal_mask": legal_mask,
    }
    if device != "cpu":
        tensors = {k: v.to(device, non_blocking=True) for k, v in tensors.items()}
    return tensors


def collate(batch: list[Example], device: str = "cpu") -> dict[str, torch.Tensor]:
    """Pad a list of examples into batched tensors.

    Policy targets become dense [B, NUM_ACTIONS] distributions with a boolean
    legality mask; padding tokens carry type_id 0 with pad_mask True.
    """

    size = len(batch)
    max_tokens = max(len(e.type_ids) for e in batch)
    type_ids = torch.zeros(size, max_tokens, dtype=torch.long)
    entity_ids = torch.zeros(size, max_tokens, dtype=torch.long)
    aux_ids = torch.zeros(size, max_tokens, dtype=torch.long)
    features = torch.zeros(size, max_tokens, MAX_FEATURES)
    pad_mask = torch.ones(size, max_tokens, dtype=torch.bool)
    legal_mask = torch.zeros(size, NUM_ACTIONS, dtype=torch.bool)
    policy = torch.zeros(size, NUM_ACTIONS)
    has_policy = torch.zeros(size, dtype=torch.bool)
    value_class = torch.zeros(size, dtype=torch.long)
    # Actor-relative win probability implied by the search, as a distribution
    # over (win, draw, loss). Draw mass stays at zero: shared-civilian endings
    # are ~0.1% of games and the search value carries no signal about them, so
    # the outcome term remains their only source of supervision.
    value_soft = torch.zeros((size, 3), dtype=torch.float32)
    value_soft_valid = torch.zeros(size, dtype=torch.bool)
    joint7 = torch.zeros(size, dtype=torch.long)
    margin = torch.zeros(size)
    margin_valid = torch.zeros(size, dtype=torch.bool)
    military_final = torch.zeros(size)
    sci_final = torch.zeros(size, 2)
    for row, example in enumerate(batch):
        count = len(example.type_ids)
        type_ids[row, :count] = torch.from_numpy(example.type_ids.astype(np.int64))
        entity_ids[row, :count] = torch.from_numpy(example.entity_ids.astype(np.int64))
        aux_ids[row, :count] = torch.from_numpy(example.aux_ids.astype(np.int64))
        features[row, :count] = torch.from_numpy(example.features.astype(np.float32))
        pad_mask[row, :count] = False
        legal_indices = torch.from_numpy(example.legal.astype(np.int64))
        legal_mask[row, legal_indices] = True
        # `.astype` copies, matching the lines above: the stored arrays are
        # read-only, and `torch.from_numpy` on a read-only array yields a tensor
        # torch believes it may write.
        policy[row, legal_indices] = torch.from_numpy(
            example.policy_target.astype(np.float32)
        )
        has_policy[row] = example.has_policy
        value_class[row] = example.value_class
        if example.root_value is not None:
            probability = (1.0 + float(example.root_value)) / 2.0
            probability = min(1.0, max(0.0, probability))
            value_soft[row, 0] = probability
            value_soft[row, 2] = 1.0 - probability
            value_soft_valid[row] = True
        joint7[row] = example.joint7_class
        margin[row] = example.margin
        margin_valid[row] = example.margin_valid
        military_final[row] = example.military_final
        sci_final[row, 0] = example.sci_final_my
        sci_final[row, 1] = example.sci_final_opp
    tensors = {
        "type_ids": type_ids,
        "entity_ids": entity_ids,
        "aux_ids": aux_ids,
        "features": features,
        "pad_mask": pad_mask,
        "legal_mask": legal_mask,
        "policy": policy,
        "has_policy": has_policy,
        "value_class": value_class,
        "value_soft": value_soft,
        "value_soft_valid": value_soft_valid,
        "joint7": joint7,
        "margin": margin,
        "margin_valid": margin_valid,
        "military_final": military_final,
        "sci_final": sci_final,
    }
    if device != "cpu":
        tensors = {k: v.to(device, non_blocking=True) for k, v in tensors.items()}
    return tensors
