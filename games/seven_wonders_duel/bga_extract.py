"""Turn a Board Game Arena 7WD ``gamedatas`` object into the advisor scrape wire.

Division of labour, so nothing fragile lives in the browser:

* The browser extension only grabs ``window.gameui.gamedatas`` verbatim (see
  ``bga_snippet.js``) -- a raw dump with no game knowledge, so a BGA UI change
  can't silently corrupt a mapping.
* Everything game-specific -- seat framing, pyramid geometry, science pairs,
  military tokens -- lives here, next to the engine data it depends on, and is
  unit-tested against a captured real position (``testdata/bga_*.json``).

``wire_from_bga`` emits exactly the dict :meth:`SevenWondersAdvisor.state_from_wire`
already accepts on its scrape branch::

    {"observation": <scrape wire>, "resample_seed": <int>}

so the future flow is: extension POSTs the raw ``gamedatas`` -> a host endpoint
calls ``wire_from_bga`` -> the existing determinizer + Gumbel search. No new
codec, no numeric-id alignment (the advisor keys on the *name strings* BGA
already exposes in ``buildings[id].name`` / ``wonders[id].name`` / token
``type``).

Scope (mirrors the scrape codec it feeds):
  * PLAY_AGE only -- the position a human actually asks about. Wonder draft,
    the between-age start-player choice, and mid-move pending choices raise
    ``UnsupportedBgaState`` rather than emit a wrong position.
  * Base game only. Agora/Pantheon expansions raise (the trained net's action
    space doesn't include them).
  * All three ages. Age III face-down cards are split into guild-back vs
    age-III-back by their card-back sprite (``spriteXY``); see ``_AGE3_BACKS``.

Freshness (the sharp edge). ``window.gameui.gamedatas`` is the *page-load*
payload: BGA patches some fields from its notification stream but leaves the
tableau/buildings/military/tokens stale until the next full load. Reading it
mid-game silently yields an old position. ``_assert_fresh`` catches the common
case by cross-checking each player's science-card count against BGA's own
``scienceSymbolCount``; the browser side must capture on a freshly (re)loaded
page (see ``bga_snippet.js``).

Every gap above fails loudly; the mapper never emits a plausible-but-wrong wire.
"""

from __future__ import annotations

import json
from typing import Any

from .data import CARDS_BY_NAME, PROGRESS_IDS, TABLEAU_LAYOUTS, BackType, back_type_of
from .engine import CardColor, back_type_of_age
from .game import Phase, PendingChoiceKind

# BGA game state where the active player picks/uses an age card -- the main turn.
_MAIN_TURN_STATE = "playerTurn"

# BGA mid-move pending-choice states -> engine PendingChoiceKind. All occur while
# the engine phase is still PLAY_AGE (a pending_choice is set on the same turn),
# so the scrape codec already handles them; only the mapping was missing.
#   chooseProgressToken        science-pair reward: pick a board Progress token
#   chooseOpponentBuilding     destroy an opponent Brown/Grey card (Zeus/Circus)
#   chooseDiscardedBuilding    Mausoleum: build a discarded card for free
#   chooseProgressTokenFromBox Great Library: pick from a random box reveal
# The destroy state serves both colours (resolved per-state); Great Library's
# random reveal is read from BGA private args (see _pending_choice).
_PENDING_STATES = frozenset({
    "chooseProgressToken",
    "chooseOpponentBuilding",
    "chooseDiscardedBuilding",
    "chooseProgressTokenFromBox",
})

# Canonical engine military tokens in BGA slot order (1..4). Engine positions
# ascend with the slot index; positive positions sit on player 0's advantage
# side (player 0 gaining shields pushes the pawn positive and captures the
# opponent's tokens there -- engine._apply_military). BGA zeroes a slot's value
# when its token is captured, so remaining tokens are read from the slot dict,
# not reconstructed from the pawn (which oscillates and can revisit 0).
_MILITARY_SLOT_TO_POS = {1: -7, 2: -4, 3: 4, 4: 7}

# Age III face-down card backs, keyed by BGA card-back sprite (spriteXY). Two
# distinct backs exist; a face-down age-III card is one or the other.
_AGE3_BACKS = {(0, 7): BackType.AGE_III, (1, 7): BackType.GUILD}

# Sign of BGA's conflictPawn relative to engine frame (player 0 == start player).
# VERIFIED on a live off-center position (table 887892216): player 1 leading,
# pawn on player 0's side, BGA conflictPawn == -5; the engine drives
# conflict_position negative when player 1 gains shields, so no flip is needed.
# Also orients token-slot -> engine-position.
_CONFLICT_SIGN = 1


class UnsupportedBgaState(ValueError):
    """The BGA position is outside the scrape codec's supported set (wonder
    draft, a pending mid-move choice, an expansion, or an un-mapped age)."""


class StaleGamedata(ValueError):
    """The BGA ``gamedatas`` snapshot is internally inconsistent -- almost always
    a page-load payload read mid-game before BGA refreshed it. Reload the table
    and capture again."""


def _sid(gamedatas: dict, key: str) -> str:
    """Player ids arrive as either str or int across BGA payloads; normalize."""
    return str(gamedatas[key])


def _seat_order(gamedatas: dict) -> tuple[str, str]:
    """(player0, player1) ids in engine frame. Engine player 0 == the Age-I
    starting player (``new_game(seed, first_player=0)`` is what the determinizer
    rebuilds into), which BGA records as ``startPlayerId``."""
    p0 = _sid(gamedatas, "startPlayerId")
    ids = [str(pid) for pid in gamedatas["players"].keys()]
    others = [pid for pid in ids if pid != p0]
    if len(others) != 1:
        raise UnsupportedBgaState(f"expected 2 players, got {ids}")
    return p0, others[0]


def _require_base_game(gamedatas: dict) -> None:
    if int(gamedatas.get("agora", 0)) or int(gamedatas.get("pantheon", 0)):
        raise UnsupportedBgaState(
            "Agora/Pantheon expansion active; the trained net is base-game only"
        )


def _assert_fresh(gamedatas: dict) -> None:
    """Catch a stale page-load snapshot by an internal-consistency check: each
    player's count of science-bearing buildings must equal BGA's own reported
    ``scienceSymbolCount``. A mid-game read where ``playerBuildings`` hasn't been
    refreshed fails here (e.g. 1 green card owned but 4 symbols reported)."""
    for pid, situation in gamedatas["playersSituation"].items():
        reported = int(situation["scienceSymbolCount"])
        greens = sum(
            CARDS_BY_NAME[b["type"]].science is not None
            for b in gamedatas["playerBuildings"].get(pid, [])
        )
        if greens != reported:
            raise StaleGamedata(
                f"player {pid}: {greens} science buildings in playerBuildings but "
                f"scienceSymbolCount={reported}; gamedatas is stale -- reload the "
                "table before capturing"
            )


def _phase(gamedatas: dict) -> Phase:
    name = gamedatas["gamestate"]["name"]
    if name != _MAIN_TURN_STATE and name not in _PENDING_STATES:
        raise UnsupportedBgaState(
            f"game state {name!r} is not a supported PLAY_AGE decision; "
            "the scrape wire covers the main age-card turn and its mid-move "
            "pending choices only"
        )
    return Phase.PLAY_AGE  # pending choices are resolved within the PLAY_AGE turn


def _destroy_color(gamedatas: dict) -> CardColor:
    """Brown vs Grey for the ``chooseOpponentBuilding`` state. BGA serves both
    from one state, resolving ``${buildingTypeTranslatable}`` into the live
    gamestate; read the colour word back out of it."""
    gs = gamedatas["gamestate"]
    blob = " ".join(
        str(gs.get(k, "")) for k in ("description", "descriptionmyturn", "args")
    ).lower()
    brown, grey = "brown" in blob, ("grey" in blob or "gray" in blob)
    if brown and not grey:
        return CardColor.BROWN
    if grey and not brown:
        return CardColor.GREY
    raise UnsupportedBgaState(
        "could not read the destroy target colour (Brown/Grey) from the live "
        "gamestate; capture this chooseOpponentBuilding position to pin the field"
    )


def _pending_choice(
    gamedatas: dict,
    *,
    chooser_seat: int,
    opponent_pid: str,
    board_tokens: list[str],
    discard_pile: list[str],
) -> dict | None:
    """The ``pending_choice`` wire field, or None on the main age-card turn.

    Options mirror the engine's own construction (engine._apply_wonder_effects /
    _apply_science_building) exactly, derived from public state -- ``legal_actions``
    builds one move per option, so they must match. The chooser is the active
    player; both DESTROY kinds encode to the same codec block but feed distinct
    encoder decision channels, so the colour still matters."""
    name = gamedatas["gamestate"]["name"]
    if name == _MAIN_TURN_STATE:
        return None

    consume_all = False
    if name == "chooseProgressToken":
        kind, options = PendingChoiceKind.CHOOSE_AVAILABLE_PROGRESS, list(board_tokens)
    elif name == "chooseDiscardedBuilding":
        kind, options = PendingChoiceKind.BUILD_FROM_DISCARD_FREE, list(discard_pile)
    elif name == "chooseOpponentBuilding":
        color = _destroy_color(gamedatas)
        kind = (
            PendingChoiceKind.DESTROY_OPPONENT_BROWN
            if color is CardColor.BROWN
            else PendingChoiceKind.DESTROY_OPPONENT_GREY
        )
        options = [
            b["type"]
            for b in gamedatas["playerBuildings"].get(opponent_pid, [])
            if CARDS_BY_NAME[b["type"]].color is color
        ]
    elif name == "chooseProgressTokenFromBox":
        # Great Library: a random reveal of box Progress tokens, not derivable
        # from public state -- read the offered set from BGA's private args (only
        # present on the choosing player's client, which is who the advisor
        # serves). Engine sorts options by token id; consume_all_options=True.
        kind, consume_all = PendingChoiceKind.CHOOSE_UNUSED_PROGRESS, True
        private = (gamedatas["gamestate"].get("args") or {}).get("_private") or {}
        offered = private.get("progressTokensFromBox")
        if not offered:
            raise UnsupportedBgaState(
                "Great Library box tokens live in gamestate.args._private."
                "progressTokensFromBox, absent here -- capture on the choosing "
                "player's client (private info is not sent to spectators/opponent)"
            )
        options = sorted(
            (t["type"] for t in offered.values()), key=PROGRESS_IDS.__getitem__
        )
    else:  # pragma: no cover - _phase already filtered
        return None

    return {
        "kind": kind.value,
        "player": chooser_seat,
        "options": options,
        "consume_all_options": consume_all,
    }


def _science_pairs(building_names: list[str]) -> list[str]:
    """Symbols the player has claimed a progress-token pair for: any science
    symbol they own >=2 copies of (engine._apply_science_building). Returns the
    ScienceSymbol *values* the wire expects, sorted for determinism."""
    counts: dict[str, int] = {}
    for name in building_names:
        card = CARDS_BY_NAME[name]
        if card.science is not None:
            counts[card.science.value] = counts.get(card.science.value, 0) + 1
    return sorted(sym for sym, n in counts.items() if n >= 2)


def _city(gamedatas: dict, pid: str) -> dict:
    situation = gamedatas["playersSituation"][pid]
    buildings = [b["type"] for b in gamedatas["playerBuildings"].get(pid, [])]

    wonders_unbuilt: list[str] = []
    wonders_built: list[str] = []
    wlookup = gamedatas["wonders"]
    for w in gamedatas["wondersSituation"].get(pid, []):
        name = wlookup[str(w["wonder"])]["name"]
        (wonders_built if int(w["constructed"]) else wonders_unbuilt).append(name)

    tokens = [t["type"] for t in gamedatas["progressTokensSituation"].get(pid, [])]

    return {
        "coins": int(situation["coins"]),
        "wonders": wonders_unbuilt,
        "built_wonders": wonders_built,
        "buildings": buildings,
        "progress_tokens": tokens,
        "science_pairs": _science_pairs(buildings),
    }


def _facedown_back(age: int, sprite: Any) -> str:
    """Back type of a face-down card. Ages I/II have a single back; Age III mixes
    age-III and guild backs, told apart by the card-back sprite."""
    if age != 3:
        return back_type_of_age(age).value
    key = tuple(sprite) if sprite is not None else None
    back = _AGE3_BACKS.get(key)
    if back is None:
        raise UnsupportedBgaState(
            f"unrecognized Age III card-back sprite {sprite!r}; expected one of "
            f"{sorted(_AGE3_BACKS)} (age-III vs guild back)"
        )
    return back.value


def _tableau(gamedatas: dict, age: int) -> list[dict]:
    """All slots of the current age's structure, present=False for taken ones.

    BGA lists only cards still on the board, giving each a 1-indexed ``row`` and
    a ``column`` that map to the engine slot ``(row - 1, column)``. A listed card
    is revealed iff it carries a ``building`` id; ``available`` means uncovered
    (accessible). Revealed cards get their true back from the card identity;
    face-down cards from ``_facedown_back`` (age-III guild split via sprite)."""
    blookup = gamedatas["buildings"]

    by_slot: dict[tuple[int, int], dict] = {}
    for card in gamedatas["draftpool"]["cards"]:
        slot = (int(card["row"]) - 1, int(card["column"]))
        revealed = card.get("building") is not None
        if revealed:
            name = blookup[str(card["building"])]["name"]
            back = back_type_of(name).value
        else:
            name = None
            back = _facedown_back(age, card.get("spriteXY"))
        by_slot[slot] = {
            "revealed": revealed,
            "accessible": bool(card.get("available")),
            "card_name": name,
            "back": back,
        }

    out: list[dict] = []
    for slot in TABLEAU_LAYOUTS[age]:
        info = by_slot.get((slot.row, slot.x))
        if info is None:  # slot already emptied (card taken/discarded)
            out.append({
                "slot_id": [slot.row, slot.x],
                "present": False, "revealed": False, "accessible": False,
                "card_name": None, "back": None,
            })
        else:
            out.append({
                "slot_id": [slot.row, slot.x],
                "present": True,
                "revealed": info["revealed"],
                "accessible": info["accessible"],
                "card_name": info["card_name"],
                "back": info["back"],
            })
    return out


def _military(gamedatas: dict) -> tuple[int, list[list[int]]]:
    """(conflict_position, remaining tokens) in engine frame.

    ``conflict_position`` is BGA's signed ``conflictPawn``. Remaining tokens are
    read straight from BGA's slot dict -- BGA zeroes a slot's coin value when its
    token is captured -- rather than reconstructed from the pawn, which can
    oscillate back through positions whose tokens are already gone. Slot->engine
    position and pawn sign share the ``_CONFLICT_SIGN`` orientation."""
    track = gamedatas["militaryTrack"]
    pos = _CONFLICT_SIGN * int(track["conflictPawn"])

    remaining: list[list[int]] = []
    for slot, penalty in track.get("tokens", {}).items():
        penalty = int(penalty)
        if penalty <= 0:
            continue  # captured: BGA zeroes the slot
        engine_pos = _CONFLICT_SIGN * _MILITARY_SLOT_TO_POS[int(slot)]
        remaining.append([engine_pos, penalty])
    remaining.sort()
    return pos, remaining


def wire_from_bga(gamedatas: dict, *, resample_seed: int = 0) -> dict[str, Any]:
    """Map a BGA ``gamedatas`` dict to the advisor scrape-wire envelope.

    Raises :class:`UnsupportedBgaState` for any position the scrape codec does
    not support, or :class:`StaleGamedata` for an unrefreshed snapshot -- never
    emits a plausible-but-wrong wire.
    """
    _require_base_game(gamedatas)
    _assert_fresh(gamedatas)
    phase = _phase(gamedatas)
    p0, p1 = _seat_order(gamedatas)

    active_id = _sid(gamedatas["gamestate"], "active_player")
    active_player = 0 if active_id == p0 else 1

    age = int(gamedatas["draftpool"]["age"])
    board_tokens = [t["type"] for t in gamedatas["progressTokensSituation"].get("board", [])]
    discard_pile = [d["type"] for d in gamedatas.get("discardedBuildings", [])]
    conflict_position, military = _military(gamedatas)

    pending = _pending_choice(
        gamedatas,
        chooser_seat=active_player,
        opponent_pid=p1 if active_id == p0 else p0,
        board_tokens=board_tokens,
        discard_pile=discard_pile,
    )

    observation = {
        "phase": phase.value,
        "active_player": active_player,
        "age": age,
        "cities": [_city(gamedatas, p0), _city(gamedatas, p1)],
        "available_progress_tokens": board_tokens,
        "wonder_offer": [],  # empty outside WONDER_DRAFT
        "tableau": _tableau(gamedatas, age),
        "discard_pile": discard_pile,
        "buried_cards": [],       # Pantheon-only; base game empty
        "wonder_burials": [],     # Agora-only
        "retired_wonders": [],    # Agora-only
        "pending_choice": pending,
        "pending_extra_turn": False,
        "pending_shields": 0,
        "conflict_position": conflict_position,
        "military_tokens_remaining": military,
        "winner": None,
        "victory_type": None,
        "final_scores": None,
    }
    return {"observation": observation, "resample_seed": int(resample_seed)}
