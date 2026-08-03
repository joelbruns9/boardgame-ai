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
  * WONDER_DRAFT and PLAY_AGE, plus the four mid-move pending choices. The
    between-age start-player choice and both expansions raise
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
import re
from typing import Any

from .data import (  # noqa: F401
    AGE_I_CARDS,
    AGE_II_CARDS,
    AGE_III_CARDS,
    CARD_IDS,
    CARDS_BY_NAME,
    GUILD_CARDS,
    PROGRESS_IDS,
    TABLEAU_LAYOUTS,
    BackType,
    back_type_of,
)
from .engine import CardColor, back_type_of_age
from .game import Phase, PendingChoiceKind

# BGA game state where the active player picks/uses an age card -- the main turn.
_MAIN_TURN_STATE = "playerTurn"

# BGA's wonder-draft state (SevenWondersDuel::STATE_SELECT_WONDER_NAME). Not yet
# a supported *position* -- see ADVISOR_IMPLEMENTATION_PLAN.md item A -- but the
# freshness patch has to recognise it, because wondersSituation goes stale here
# and nowhere else.
_DRAFT_STATE = "selectWonder"

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


# BGA title-cases card names ("Chamber Of Commerce") where the engine follows the
# printed card ("Chamber of Commerce"). Verified against a live table's full
# `buildings` material table (892846644): that is the only base-game divergence,
# and the 73 engine names are unique under case folding, so folding aligns them
# unambiguously. Base wonders (12/12) and progress tokens (10/10) match exactly.
_CARDS_BY_FOLDED = {name.casefold(): name for name in CARDS_BY_NAME}

# Card -> age, for cross-checking a log-parsed burial against BGA's structural
# `constructed` age. Guilds report as age 4 (BGA's filterByAge(4)).
_CARD_AGES: dict[str, int] = {}
for _age, _group in ((1, AGE_I_CARDS), (2, AGE_II_CARDS), (3, AGE_III_CARDS), (4, GUILD_CARDS)):
    for _card in _group:
        _CARD_AGES[_card.name] = _age


def _card_name(bga_name: str) -> str:
    """Canonical engine card name for a BGA card name.

    Every BGA name enters the wire through here, so the wire only ever carries
    engine spellings. An unresolvable name is expansion content or a BGA rename:
    fail typed and loud rather than with a bare KeyError deeper in the mapper.
    """
    canonical = _CARDS_BY_FOLDED.get(bga_name.casefold())
    if canonical is None:
        raise UnsupportedBgaState(
            f"unknown BGA card name {bga_name!r}; the mapper covers base-game "
            "cards only (expansion content and renames land here)"
        )
    return canonical


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
    """Catch a stale page-load snapshot by an internal-consistency check.

    ``scienceSymbolCount`` counts a player's science-bearing **cards**, not their
    distinct symbols -- verified on captures where a player holds a duplicated
    symbol (4 cards / 3 distinct, reported as 4). So it should equal the number
    of science cards we can see.

    Only a *shortfall* is an error. That is the stale direction the check exists
    for: ``playerBuildings`` left at its page-load value while ``playersSituation``
    stays fresh, so we see 1 green card against 4 reported.

    A *surplus* is benign and transient. With the DOM patch, buildings are read
    from the board, which BGA updates the instant a card is placed
    (``notif_constructBuilding``), while ``scienceSymbolCount`` only refreshes on
    the next state entry. Right after an opponent builds a green card the DOM
    legitimately leads by one -- observed live as "4 science buildings but
    science count = 3". Refusing that would make the advisor blind for a turn
    every time the opponent takes a science card, and the position it describes
    is the *more* current of the two.
    """
    for pid, situation in gamedatas["playersSituation"].items():
        reported = int(situation["scienceSymbolCount"])
        greens = sum(
            CARDS_BY_NAME[_card_name(b["type"])].science is not None
            for b in gamedatas["playerBuildings"].get(pid, [])
        )
        if greens < reported:
            raise StaleGamedata(
                f"player {pid}: {greens} science buildings visible but "
                f"scienceSymbolCount={reported}; gamedatas is stale -- reload "
                "the table before capturing"
            )


def _phase(gamedatas: dict) -> Phase:
    name = gamedatas["gamestate"]["name"]
    if name == _DRAFT_STATE:
        return Phase.WONDER_DRAFT
    if name != _MAIN_TURN_STATE and name not in _PENDING_STATES:
        raise UnsupportedBgaState(
            f"game state {name!r} is not a supported decision; the scrape wire "
            "covers the wonder draft, the main age-card turn, and its mid-move "
            "pending choices"
        )
    return Phase.PLAY_AGE  # pending choices are resolved within the PLAY_AGE turn


def _wonder_offer(gamedatas: dict) -> list[str]:
    """Wonders still on offer this draft round, in BGA's display order.

    `wondersSituation.selection` holds only `selection{round}` -- BGA never
    reveals the second group during the first round (`Wonders::getSituation`,
    modules/php/Wonders.php:24-30), which is exactly the hidden information the
    determinizer samples.
    """
    wlookup = gamedatas["wonders"]
    selection = gamedatas["wondersSituation"].get("selection") or []
    offer = [wlookup[str(row["id"])]["name"] for row in selection]
    if not offer:
        # Caught between rounds: BGA has emptied #wonder_selection_container but
        # has not rendered the next group yet. The engine would treat this as a
        # non-terminal position with no legal move, which surfaces far away as
        # "batch net row 0 returned a zero mass policy" from the root
        # evaluation. Refuse it here instead -- it is transient, and the
        # extension retries a rejected position.
        raise UnsupportedBgaState(
            "wonder draft has no wonders on offer -- captured between rounds, "
            "before BGA rendered the next group; retry in a moment"
        )
    return offer


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
            name
            for name in (
                _card_name(b["type"])
                for b in gamedatas["playerBuildings"].get(opponent_pid, [])
            )
            if CARDS_BY_NAME[name].color is color
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
    # Canonical card-id order. A city is a *set* of cards -- 7WD has no rule
    # keyed on build order -- but BGA's two sources disagree on ordering
    # (getAllDatas sorts by card_location_arg i.e. build order; the DOM groups by
    # colour column). Sorting makes the wire identical either way.
    buildings = sorted(
        (_card_name(b["type"]) for b in gamedatas["playerBuildings"].get(pid, [])),
        key=CARD_IDS.__getitem__,
    )

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
            name = _card_name(blookup[str(card["building"])]["name"])
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


# "<player> constructed Wonder "<wonder>" for N coin(s) using building "<card>"".
# From the constructWonder notification (Wonder.php:58-73), which carries
# buildingId/buildingName; BGA renders it into the game log, so a single snapshot
# holds every burial. The message is clienttranslate()d, hence the loose middle
# and the anchor on the two curly-quoted names.
_BURIAL_LOG = re.compile(
    r"Wonder\s*[“\"']([^”\"']+)[”\"'].*?"
    r"building\s*[“\"']([^”\"']+)[”\"']"
)


def _wonder_burials(
    gamedatas: dict, log_lines: list[str] | None
) -> tuple[list[tuple[str, str]], tuple[int, ...]]:
    """``(exact (wonder, card) pairs, ages of burials we could not identify)``.

    Constructing a wonder buries an age card under it permanently. Two sources:

    * **Structural (always available).** ``wondersSituation[pid][i]["constructed"]``
      is the buried card's *age* (0 when unbuilt) -- see ``Player::getWondersData``
      and ``Building::getBackSpriteXY``. This alone pins down how many cards of
      each age are out of play, which is what the pool arithmetic needs.
    * **The game log (identity).** Only matters for the *current* age: for a
      finished age a buried card and a box-removed card are indistinguishable --
      both are out of play forever and nothing ever reveals either.

    The log is localized prose, so it is trusted only when it agrees with the
    structural data: same number of burials, and each named card's age equal to
    the wonder's ``constructed`` value. Any disagreement falls back to
    "unidentified", which is blunter but never wrong.
    """
    wlookup = gamedatas["wonders"]
    structural: list[tuple[str, int]] = []
    for pid, rows in gamedatas["wondersSituation"].items():
        if pid == "selection":
            continue
        for row in rows:
            age = int(row.get("constructed") or 0)
            if age:
                structural.append((wlookup[str(row["wonder"])]["name"], age))

    if not structural:
        return [], ()

    parsed: dict[str, str] = {}
    for line in log_lines or []:
        match = _BURIAL_LOG.search(line)
        if match:
            wonder, card = match.group(1).strip(), match.group(2).strip()
            parsed.setdefault(wonder, card)

    exact: list[tuple[str, str]] = []
    unknown: list[int] = []
    for wonder, age in structural:
        card = parsed.get(wonder)
        if card is None:
            unknown.append(age)
            continue
        try:
            canonical = _card_name(card)
        except UnsupportedBgaState:
            unknown.append(age)
            continue
        # Cross-check the prose against the structural age before trusting it.
        if _CARD_AGES.get(canonical) != age:
            unknown.append(age)
            continue
        exact.append((wonder, canonical))

    return sorted(exact), tuple(unknown)


def apply_dom_patch(gamedatas: dict, dom: dict) -> dict:
    """Return ``gamedatas`` with the never-refreshed fields replaced from a DOM
    re-read (``bga_snippet.captureDomPatch``), so a capture taken without an F5
    describes the live board.

    BGA leaves five fields at their page-load values (see the module docstring
    and ADVISOR_IMPLEMENTATION_PLAN.md item B). The browser returns **numeric
    ids only**; the id -> name mapping happens here against ``gamedatas``' own
    material tables, so a BGA DOM change breaks a selector loudly in the page
    rather than yielding a plausible-but-wrong name.

    The result is ordinary ``gamedatas``-shaped data: ``wire_from_bga`` is
    unchanged and its freshness cross-check still runs, now catching a bad patch
    instead of a missing reload.
    """
    patched = dict(gamedatas)
    buildings, tokens = gamedatas["buildings"], gamedatas.get("progressTokens", {})

    def _named(table: dict, ids: list, kind: str) -> list[dict]:
        rows = []
        for index, item_id in enumerate(ids):
            entry = table.get(str(item_id))
            if entry is None:
                raise UnsupportedBgaState(
                    f"DOM patch references unknown {kind} id {item_id!r}; the "
                    "capture and the gamedatas material table disagree"
                )
            rows.append({"id": str(item_id), "type": entry["name"], "location_arg": index})
        return rows

    patched["playerBuildings"] = {
        pid: _named(buildings, ids, "building")
        for pid, ids in dom["playerBuildings"].items()
    }
    patched["discardedBuildings"] = _named(
        buildings, dom["discardedBuildings"], "building"
    )

    progress = {
        "board": [
            {"id": str(tid), "type": tokens[str(tid)]["name"], "location_arg": slot}
            for slot, tid in dom["boardProgressTokens"]
        ]
    }
    for pid, ids in dom["playerProgressTokens"].items():
        progress[pid] = _named(tokens, ids, "progress token")
    patched["progressTokensSituation"] = progress

    patched["militaryTrack"] = {
        "tokens": {str(slot): value for slot, value in dom["militaryTokens"].items()},
        "conflictPawn": dom["conflictPawn"],
    }

    # wondersSituation is stale during the wonder draft only -- outside it, BGA
    # refreshes it via notif_constructWonder and argPlayerTurn. Patch it only in
    # the draft, where nothing is constructed yet, because the DOM capture
    # carries wonder ids but not each wonder's constructed flag.
    if gamedatas["gamestate"]["name"] == _DRAFT_STATE:
        patched["wondersSituation"] = {
            "selection": [
                {"id": str(wid), "type": gamedatas["wonders"][str(wid)]["name"],
                 "location_arg": slot}
                for slot, wid in enumerate(dom["wonderSelection"])
            ],
            **{
                pid: [
                    {"wonder": str(wid), "position": slot, "constructed": False}
                    for slot, wid in enumerate(ids)
                ]
                for pid, ids in dom["playerWonders"].items()
            },
        }
    return patched


def wire_from_bga_payload(payload: dict, *, resample_seed: int | None = None) -> dict[str, Any]:
    """Map the extension's ``{"bga": ..., "args": ..., "dom": ..., "log": [...]}``
    envelope.

    ``dom`` is optional: without it this is exactly ``wire_from_bga`` on a
    freshly loaded page. ``args`` (the current state's server args) is carried
    for cross-checks. ``log`` is the rendered game-log lines, used only to
    identify cards buried under constructed wonders; without it those burials
    are still *counted* (from ``wondersSituation``), just not named.
    """
    gamedatas = payload["bga"]
    if payload.get("dom"):
        gamedatas = apply_dom_patch(gamedatas, payload["dom"])
    if resample_seed is None:
        resample_seed = int(payload.get("resample_seed", 0))
    return wire_from_bga(
        gamedatas, resample_seed=resample_seed, log_lines=payload.get("log")
    )


def wire_from_bga(
    gamedatas: dict, *, resample_seed: int = 0, log_lines: list[str] | None = None
) -> dict[str, Any]:
    """Map a BGA ``gamedatas`` dict to the advisor scrape-wire envelope.

    Raises :class:`UnsupportedBgaState` for any position the scrape codec does
    not support, or :class:`StaleGamedata` for an unrefreshed snapshot -- never
    emits a plausible-but-wrong wire.
    """
    _require_base_game(gamedatas)
    phase = _phase(gamedatas)
    if phase is not Phase.WONDER_DRAFT:
        # Blind during the draft anyway -- it compares science-card counts, and
        # both players have none. Freshness there rests on the DOM patch, which
        # rewrites wondersSituation (stale for the whole draft; see
        # ADVISOR_IMPLEMENTATION_PLAN.md "the fifth stale field").
        _assert_fresh(gamedatas)
    p0, p1 = _seat_order(gamedatas)

    active_id = _sid(gamedatas["gamestate"], "active_player")
    active_player = 0 if active_id == p0 else 1

    # getAllDatas only fills draftpool once the wonder selection is empty
    # (sevenwondersduel.game.php:685-688), so during the draft it arrives as
    # `[]`. That is expected there -- no age has been dealt -- and a hard error
    # anywhere else.
    draftpool = gamedatas["draftpool"]
    drafting = phase is Phase.WONDER_DRAFT
    if drafting:
        age = 1  # the first age is dealt when the draft ends
    elif not isinstance(draftpool, dict):
        raise UnsupportedBgaState(
            "draftpool is empty but the game is past the wonder draft"
        )
    else:
        age = int(draftpool["age"])
    board_tokens = [t["type"] for t in gamedatas["progressTokensSituation"].get("board", [])]
    discard_pile = [_card_name(d["type"]) for d in gamedatas.get("discardedBuildings", [])]
    conflict_position, military = _military(gamedatas)
    # Constructing a wonder buries an age card permanently. Identity from the
    # game log when available; otherwise only the age, which still keeps the
    # unseen-card pool honest. See _wonder_burials.
    burials, unknown_burial_ages = _wonder_burials(gamedatas, log_lines)

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
        "wonder_offer": _wonder_offer(gamedatas) if drafting else [],
        # No age is dealt during the draft, so there is no structure to read.
        "tableau": [] if drafting else _tableau(gamedatas, age),
        "discard_pile": discard_pile,
        "buried_cards": [],       # Pantheon-only; base game empty
        "wonder_burials": [list(pair) for pair in burials],
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
    envelope = {"observation": observation, "resample_seed": int(resample_seed)}
    if unknown_burial_ages:
        envelope["unknown_burial_ages"] = list(unknown_burial_ages)
    return envelope
