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

from dataclasses import dataclass

import numpy as np
import torch

from .buffer import GameRecord, replay
from .codec import NUM_ACTIONS, legal_action_indices
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


@dataclass(slots=True)
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


def examples_from_record(record: GameRecord) -> list[Example]:
    """Replay one game (through the VERIFIED buffer.replay path — mask hashes,
    actors, chance log, trajectory and final digests all checked) and emit an
    example per recorded decision. A stale or tampered buffer raises
    ReplayMismatchError instead of silently training on regenerated states."""

    staged: list[tuple[Example, int]] = []  # (example, actor)

    def featurize(game, move):
        actor = (
            game.pending_choice.player
            if game.pending_choice is not None
            else game.active_player
        )
        encoding = encode(game.observation(actor))
        if encoding.actor != actor:
            raise AssertionError("encoder actor disagrees with replay actor")
        legal = np.asarray(legal_action_indices(game), dtype=np.int16)
        policy = np.zeros(len(legal), dtype=np.float32)
        index_of = {int(a): i for i, a in enumerate(legal)}
        if move.policy_target:
            # Preferred: the improved completed-Q distribution from Gumbel
            # search (visits stay as raw evidence for reanalyze).
            for action, probability in move.policy_target.items():
                if int(action) not in index_of:
                    raise ValueError(
                        f"move {move.i}: policy target on illegal action {action}"
                    )
                policy[index_of[int(action)]] = probability
            total = float(policy.sum())
            if not 0.999 <= total <= 1.001:
                raise ValueError(
                    f"move {move.i}: policy target sums to {total:.4f}"
                )
            policy /= total
        elif move.visits:
            total = float(sum(move.visits.values()))
            if total <= 0:
                raise ValueError(f"move {move.i}: visit counts sum to zero")
            for action, visits in move.visits.items():
                if int(action) not in index_of:
                    raise ValueError(
                        f"move {move.i}: visit on illegal action {action}"
                    )
                policy[index_of[int(action)]] = visits / total
        else:
            policy[int(np.searchsorted(legal, move.action))] = 1.0
        type_ids, entity_ids, aux_ids, features = vectorize(encoding)
        staged.append(
            (
                Example(
                    type_ids=type_ids,
                    entity_ids=entity_ids,
                    aux_ids=aux_ids,
                    features=features,
                    legal=legal,
                    policy_target=policy,
                    has_policy=not move.policy_excluded,
                    value_class=0,
                    joint7_class=0,
                    margin=0.0,
                    margin_valid=False,
                    military_final=0.0,
                    sci_final_my=0.0,
                    sci_final_opp=0.0,
                    game_key=record.seed,
                    iteration=record.iteration,
                ),
                actor,
            )
        )

    game = replay(record, on_state=featurize)
    final_position = game.conflict_position
    sci_counts = (len(_science_symbols(game, 0)), len(_science_symbols(game, 1)))
    for example, actor in staged:
        example.value_class = _actor_value_class(game.winner, actor)
        example.joint7_class = _joint7_class(game.winner, game.victory_type, actor)
        if game.final_scores is not None:
            mine, theirs = game.final_scores[actor], game.final_scores[1 - actor]
            example.margin = (mine - theirs) / 20.0
            example.margin_valid = True
        rel = final_position if actor == 0 else -final_position
        example.military_final = rel / 9.0
        example.sci_final_my = sci_counts[actor] / 6.0
        example.sci_final_opp = sci_counts[1 - actor] / 6.0
    return [example for example, _ in staged]


def examples_from_records(records) -> list[Example]:
    out: list[Example] = []
    for record in records:
        out.extend(examples_from_record(record))
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
        policy[row, legal_indices] = torch.from_numpy(example.policy_target)
        has_policy[row] = example.has_policy
        value_class[row] = example.value_class
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
        "joint7": joint7,
        "margin": margin,
        "margin_valid": margin_valid,
        "military_final": military_final,
        "sci_final": sci_final,
    }
    if device != "cpu":
        tensors = {k: v.to(device, non_blocking=True) for k, v in tensors.items()}
    return tensors
