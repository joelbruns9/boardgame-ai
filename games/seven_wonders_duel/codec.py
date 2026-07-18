"""Fixed 1202-action identity-indexed action codec (CODEC_SPEC.md §3).

The codec never recomputes legality — it maps `engine.legal_actions` output to
indices and back. Mask exactness is *defined* as agreement with the engine.

Block layout (spec §3.1; sizes frozen by the id tables in data.py):

    WONDER_DRAFT      0–11     wonder_id
    BUILD             12–84    12 + card_id
    DISCARD           85–157   85 + card_id
    CARD_TO_WONDER    158–1033 158 + card_id*12 + wonder_id
    DESTROY           1034–1106  (Zeus brown / Circus Maximus grey, one block)
    MAUSOLEUM_REVIVE  1107–1179
    PROGRESS_BOARD    1180–1189  (science-pair pick from the board)
    PROGRESS_LIBRARY  1190–1199  (Great Library pick from the 3 drawn)
    NEXT_AGE_STARTER  1200 = self starts, 1201 = opponent starts (actor-relative)
"""

from __future__ import annotations

from .data import (
    ALL_BUILDING_CARDS,
    CARD_IDS,
    PROGRESS_IDS,
    PROGRESS_TOKENS,
    WONDER_IDS,
    WONDERS,
)
from .engine import Action, ActionUse, legal_actions
from .game import GameState, PendingChoiceKind, SlotId

NUM_WONDERS = len(WONDERS)          # 12
NUM_CARDS = len(ALL_BUILDING_CARDS)  # 73
NUM_PROGRESS = len(PROGRESS_TOKENS)  # 10

WONDER_DRAFT_BASE = 0
BUILD_BASE = WONDER_DRAFT_BASE + NUM_WONDERS            # 12
DISCARD_BASE = BUILD_BASE + NUM_CARDS                   # 85
CARD_TO_WONDER_BASE = DISCARD_BASE + NUM_CARDS          # 158
DESTROY_BASE = CARD_TO_WONDER_BASE + NUM_CARDS * NUM_WONDERS  # 1034
MAUSOLEUM_BASE = DESTROY_BASE + NUM_CARDS               # 1107
PROGRESS_BOARD_BASE = MAUSOLEUM_BASE + NUM_CARDS        # 1180
PROGRESS_LIBRARY_BASE = PROGRESS_BOARD_BASE + NUM_PROGRESS  # 1190
NEXT_AGE_BASE = PROGRESS_LIBRARY_BASE + NUM_PROGRESS    # 1200
NUM_ACTIONS = NEXT_AGE_BASE + 2                         # 1202

_CARD_NAMES = tuple(card.name for card in ALL_BUILDING_CARDS)
_WONDER_NAMES = tuple(wonder.name for wonder in WONDERS)
_PROGRESS_NAMES = tuple(token.name for token in PROGRESS_TOKENS)

_PENDING_BLOCK = {
    PendingChoiceKind.DESTROY_OPPONENT_BROWN: DESTROY_BASE,
    PendingChoiceKind.DESTROY_OPPONENT_GREY: DESTROY_BASE,
    PendingChoiceKind.BUILD_FROM_DISCARD_FREE: MAUSOLEUM_BASE,
    PendingChoiceKind.CHOOSE_AVAILABLE_PROGRESS: PROGRESS_BOARD_BASE,
    PendingChoiceKind.CHOOSE_UNUSED_PROGRESS: PROGRESS_LIBRARY_BASE,
}


def _card_at(game: GameState, slot_id: SlotId) -> str:
    card = game.tableau.cards[slot_id]
    if not (card.present and card.revealed):
        raise ValueError(f"slot {slot_id} holds no revealed card")
    return card.card_name


def _slot_for_card(game: GameState, card_name: str) -> SlotId:
    """The unique accessible slot holding ``card_name`` (accessible ⇒ revealed;
    no duplicate cards ⇒ bijective)."""

    for slot_id in game.tableau.accessible_slot_ids():
        if game.tableau.cards[slot_id].card_name == card_name:
            return slot_id
    raise ValueError(f"card is not in an accessible slot: {card_name}")


def encode_action(game: GameState, action: Action) -> int:
    """Map an engine action (in the context of ``game``) to its codec index."""

    if action.use is ActionUse.DRAFT_WONDER:
        if action.wonder_name is None:
            raise ValueError("draft action is missing a wonder")
        return WONDER_DRAFT_BASE + WONDER_IDS[action.wonder_name]

    if action.use is ActionUse.CONSTRUCT_BUILDING:
        if action.slot_id is None:
            raise ValueError("build action is missing a slot")
        return BUILD_BASE + CARD_IDS[_card_at(game, action.slot_id)]

    if action.use is ActionUse.DISCARD_FOR_COINS:
        if action.slot_id is None:
            raise ValueError("discard action is missing a slot")
        return DISCARD_BASE + CARD_IDS[_card_at(game, action.slot_id)]

    if action.use is ActionUse.CONSTRUCT_WONDER:
        if action.slot_id is None or action.wonder_name is None:
            raise ValueError("wonder action is missing a slot or wonder")
        card_id = CARD_IDS[_card_at(game, action.slot_id)]
        return (
            CARD_TO_WONDER_BASE
            + card_id * NUM_WONDERS
            + WONDER_IDS[action.wonder_name]
        )

    if action.use is ActionUse.RESOLVE_PENDING_CHOICE:
        if action.choice is None:
            raise ValueError("pending-choice action is missing a choice")
        pending = game.pending_choice
        if pending is None:
            raise ValueError("no pending choice to encode against")
        base = _PENDING_BLOCK[pending.kind]
        ids = PROGRESS_IDS if base >= PROGRESS_BOARD_BASE else CARD_IDS
        return base + ids[action.choice]

    if action.use is ActionUse.CHOOSE_NEXT_START_PLAYER:
        if action.starting_player is None:
            raise ValueError("next-age action is missing a starting player")
        # Actor-relative (spec §2/§3.2): the chooser is game.active_player.
        return NEXT_AGE_BASE + (0 if action.starting_player == game.active_player else 1)

    raise ValueError(f"cannot encode action: {action}")


def decode_action(game: GameState, index: int) -> Action:
    """Map a codec index back to an engine action in the context of ``game``.

    Every legal (masked) index decodes; indices whose entity is absent from the
    current state raise ValueError.
    """

    if not 0 <= index < NUM_ACTIONS:
        raise ValueError(f"action index out of range: {index}")

    if index < BUILD_BASE:
        return Action(None, ActionUse.DRAFT_WONDER, wonder_name=_WONDER_NAMES[index])

    if index < DISCARD_BASE:
        card_name = _CARD_NAMES[index - BUILD_BASE]
        return Action(_slot_for_card(game, card_name), ActionUse.CONSTRUCT_BUILDING)

    if index < CARD_TO_WONDER_BASE:
        card_name = _CARD_NAMES[index - DISCARD_BASE]
        return Action(_slot_for_card(game, card_name), ActionUse.DISCARD_FOR_COINS)

    if index < DESTROY_BASE:
        card_offset, wonder_id = divmod(index - CARD_TO_WONDER_BASE, NUM_WONDERS)
        card_name = _CARD_NAMES[card_offset]
        return Action(
            _slot_for_card(game, card_name),
            ActionUse.CONSTRUCT_WONDER,
            wonder_name=_WONDER_NAMES[wonder_id],
        )

    if index < NEXT_AGE_BASE:
        if index < MAUSOLEUM_BASE:
            base, choice = DESTROY_BASE, _CARD_NAMES[index - DESTROY_BASE]
        elif index < PROGRESS_BOARD_BASE:
            base, choice = MAUSOLEUM_BASE, _CARD_NAMES[index - MAUSOLEUM_BASE]
        elif index < PROGRESS_LIBRARY_BASE:
            base, choice = (
                PROGRESS_BOARD_BASE,
                _PROGRESS_NAMES[index - PROGRESS_BOARD_BASE],
            )
        else:
            base, choice = (
                PROGRESS_LIBRARY_BASE,
                _PROGRESS_NAMES[index - PROGRESS_LIBRARY_BASE],
            )
        pending = game.pending_choice
        if pending is None or _PENDING_BLOCK[pending.kind] != base:
            raise ValueError(
                f"index {index} does not match the pending choice in this state"
            )
        return Action(None, ActionUse.RESOLVE_PENDING_CHOICE, choice=choice)

    starting_player = (
        game.active_player if index == NEXT_AGE_BASE else 1 - game.active_player
    )
    return Action(
        None, ActionUse.CHOOSE_NEXT_START_PLAYER, starting_player=starting_player
    )


def legal_action_indices(game: GameState) -> tuple[int, ...]:
    """Sorted codec indices of exactly the engine's legal actions."""

    return tuple(sorted(encode_action(game, action) for action in legal_actions(game)))


def legal_action_mask(game: GameState) -> list[bool]:
    mask = [False] * NUM_ACTIONS
    for index in legal_action_indices(game):
        mask[index] = True
    return mask
