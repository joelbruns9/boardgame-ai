"""bga_extract.wire_from_bga proves out end-to-end on a captured real position.

The captured fixture is the trimmed ``gamedatas`` from a live BGA Age I table
(#887892216). The test asserts the mapper's wire is not just well-formed but
*feeds the existing scrape codec*: it parses via observation_from_wire and
determinizes into a full state whose public projection matches the input.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from .advisor_scrape import determinize_observation, observation_from_wire
from .bga_extract import StaleGamedata, UnsupportedBgaState, wire_from_bga
from .game import Phase

_TESTDATA = Path(__file__).parent / "testdata"
_FIXTURE = _TESTDATA / "bga_887892216_agei.json"
_FIXTURE_AGE3 = _TESTDATA / "bga_887892216_ageiii.json"


def _load() -> dict:
    return json.loads(_FIXTURE.read_text())


def _load_age3() -> dict:
    return json.loads(_FIXTURE_AGE3.read_text())


def test_wire_envelope_shape():
    wire = wire_from_bga(_load(), resample_seed=7)
    assert set(wire) == {"observation", "resample_seed"}
    assert wire["resample_seed"] == 7
    obs = wire["observation"]
    assert obs["phase"] == Phase.PLAY_AGE.value
    # startPlayer is "test" (84634030) -> engine seat 0; "me" (RollwJoel) to move
    assert obs["active_player"] == 1
    assert obs["age"] == 1


def test_seat_framing_and_cities():
    obs = wire_from_bga(_load())["observation"]
    p0, p1 = obs["cities"]
    # seat 0 == start player "test": coins 10, has Glassworks/Clay Pool/Scriptorium
    assert p0["coins"] == 10
    assert set(p0["buildings"]) == {"Glassworks", "Clay Pool", "Scriptorium"}
    # seat 1 == "me": coins 7, Press/Pharmacist/Clay Reserve
    assert p1["coins"] == 7
    assert set(p1["buildings"]) == {"Press", "Pharmacist", "Clay Reserve"}
    # nobody has a science pair yet (one green each)
    assert p0["science_pairs"] == [] and p1["science_pairs"] == []
    # all wonders unbuilt, four each
    assert len(p0["wonders"]) == 4 and p0["built_wonders"] == []
    assert "The Statue of Zeus" in p0["wonders"]
    assert "The Sphinx" in p1["wonders"]


def test_board_and_military():
    obs = wire_from_bga(_load())["observation"]
    assert obs["available_progress_tokens"] == [
        "Masonry", "Architecture", "Philosophy", "Agriculture", "Law",
    ]
    assert obs["conflict_position"] == 0
    # pawn at 0 -> all four tokens present, engine positions
    assert sorted(obs["military_tokens_remaining"]) == [
        [-7, 5], [-4, 2], [4, 2], [7, 5],
    ]
    assert set(obs["discard_pile"]) == {
        "Palisade", "Theater", "Workshop", "Altar", "Stable",
    }


def test_tableau_geometry():
    obs = wire_from_bga(_load())["observation"]
    present = {tuple(c["slot_id"]): c for c in obs["tableau"] if c["present"]}
    # Age I structure has 20 slots; 9 remain on the board in this position
    assert len(obs["tableau"]) == 20
    assert len(present) == 9
    # Tavern is the face-up, accessible card at engine slot (2, 5)
    tav = present[(2, 5)]
    assert tav["card_name"] == "Tavern" and tav["revealed"] and tav["accessible"]
    # Garrison face-up accessible at (3, 8); a face-down card at (1, 4)
    assert present[(3, 8)]["card_name"] == "Garrison"
    facedown = present[(1, 4)]
    assert facedown["revealed"] is False and facedown["card_name"] is None
    assert facedown["back"] == "age_i"


def test_feeds_scrape_codec_end_to_end():
    """The whole point: mapper output -> existing codec -> valid full state."""
    wire = wire_from_bga(_load())
    obs = observation_from_wire(wire["observation"])
    assert obs.phase is Phase.PLAY_AGE

    state = determinize_observation(obs, random.Random(0))
    # Public projection of the determinized state must reproduce the input.
    projected = state.observation(0)
    assert projected.cities[0].coins == 10
    assert projected.cities[1].coins == 7
    assert projected.age == 1
    assert projected.active_player == 1
    # determinization is a valid full state: the codec would have raised on any
    # unseen/face-down count mismatch. The 9 cards still in the structure are
    # materialized, and exactly 3 age-I cards were assigned to the removed pile.
    assert len(state.age_decks[1]) == 9
    assert len(state.removed_age_cards[1]) == 3
    # every determinized face-down slot got a concrete, distinct age-I card
    tableau_names = [c.card_name for c in state.tableau.cards.values() if c.present]
    assert all(n is not None for n in tableau_names)


def test_expansion_rejected():
    data = _load()
    data["pantheon"] = 1
    with pytest.raises(UnsupportedBgaState, match="expansion"):
        wire_from_bga(data)


def test_non_play_state_rejected():
    data = _load()
    # Not a real BGA state name -- the draft is "selectWonder". An unknown
    # state must still be refused rather than guessed at.
    data["gamestate"]["name"] = "wonderDraft"
    with pytest.raises(UnsupportedBgaState, match="not a supported decision"):
        wire_from_bga(data)


def test_military_sign_and_capture_off_center():
    # Real off-center capture (table 887892216): player 1 (RollwJoel) leading,
    # pawn on player 0's (test's) side. BGA conflictPawn == -5; slots 2 (-4) and
    # 3 (+4) both captured -- +4 taken earlier when the pawn was positive, -4 on
    # the push to -5. Locks _CONFLICT_SIGN and the oscillation-safe token read.
    data = _load_age3()
    data["gamestate"]["name"] = "playerTurn"
    data["militaryTrack"] = {"tokens": {"1": "5", "2": "0", "3": "0", "4": "5"},
                             "conflictPawn": "-5"}
    obs = wire_from_bga(data)["observation"]
    assert obs["conflict_position"] == -5  # negative == player 1 ahead, engine frame
    assert sorted(obs["military_tokens_remaining"]) == [[-7, 5], [7, 5]]


def test_select_start_player_rejected():
    # The real Age III fixture is captured at the between-age start-player choice,
    # which the scrape codec does not cover -- the mapper must refuse it.
    with pytest.raises(UnsupportedBgaState, match="not a supported decision"):
        wire_from_bga(_load_age3())


def test_stale_snapshot_detected():
    # Simulate the classic stale read: drop the Age II/III green buildings so
    # science-card count no longer matches BGA's reported scienceSymbolCount.
    data = _load_age3()
    data["gamestate"]["name"] = "playerTurn"
    greens = {"Pharmacist", "Library", "Dispensary", "Laboratory", "Scriptorium"}
    for pid, blds in data["playerBuildings"].items():
        data["playerBuildings"][pid] = [b for b in blds if b["type"] not in greens]
    with pytest.raises(StaleGamedata, match="stale"):
        wire_from_bga(data)


def _age3_pending(state_name: str, *, active: str, description: str = "") -> dict:
    data = _load_age3()
    data["gamestate"]["name"] = state_name
    data["gamestate"]["active_player"] = active
    if description:
        data["gamestate"]["description"] = description
    return data


def _resolve_actions(obs: dict):
    """Determinize the pending position and return its legal engine actions."""
    from .codec import decode_action, legal_action_indices
    from .engine import ActionUse

    state = determinize_observation(observation_from_wire(obs), random.Random(0))
    actions = [decode_action(state, i) for i in legal_action_indices(state)]
    assert all(a.use is ActionUse.RESOLVE_PENDING_CHOICE for a in actions)
    return {a.choice for a in actions}


def test_pending_choose_progress_token():
    # Science-pair reward: pick a board Progress token. The high-frequency case.
    data = _age3_pending("chooseProgressToken", active="89146710")
    obs = wire_from_bga(data)["observation"]
    pc = obs["pending_choice"]
    assert pc["kind"] == "choose_available_progress"
    assert pc["player"] == 1  # chooser (RollwJoel) is seat 1
    assert pc["options"] == ["Masonry", "Architecture", "Philosophy", "Agriculture"]
    assert _resolve_actions(obs) == set(pc["options"])


def test_pending_build_from_discard():
    # Mausoleum: build any discarded card for free -> options are the discard pile.
    data = _age3_pending("chooseDiscardedBuilding", active="89146710")
    obs = wire_from_bga(data)["observation"]
    pc = obs["pending_choice"]
    assert pc["kind"] == "build_from_discard_free"
    assert set(pc["options"]) == {
        "Palisade", "Theater", "Workshop", "Altar", "Stable",
        "Rostrum", "School", "Caravansery", "Courthouse", "Customs House",
    }
    assert _resolve_actions(obs) == set(pc["options"])


def test_pending_destroy_opponent_grey():
    # Circus Maximus etc.: destroy an opponent Grey card. Colour read from the
    # live gamestate description; options = opponent's cards of that colour.
    data = _age3_pending(
        "chooseOpponentBuilding", active="89146710",
        description="You must choose one of the opponent's Grey cards to discard",
    )
    obs = wire_from_bga(data)["observation"]
    pc = obs["pending_choice"]
    assert pc["kind"] == "destroy_opponent_grey"
    assert pc["options"] == ["Glassworks"]  # test's only grey card
    assert _resolve_actions(obs) == {"Glassworks"}


def test_pending_destroy_opponent_brown():
    data = _age3_pending(
        "chooseOpponentBuilding", active="89146710",
        description="You must choose one of the opponent's Brown cards to discard",
    )
    obs = wire_from_bga(data)["observation"]
    pc = obs["pending_choice"]
    assert pc["kind"] == "destroy_opponent_brown"
    assert set(pc["options"]) == {
        "Clay Pool", "Stone Pit", "Logging Camp", "Brickyard", "Sawmill",
    }
    assert _resolve_actions(obs) == set(pc["options"])


def test_pending_destroy_color_ambiguous_raises():
    data = _age3_pending("chooseOpponentBuilding", active="89146710",
                         description="choose a card")  # no colour word
    with pytest.raises(UnsupportedBgaState, match="colour"):
        wire_from_bga(data)


def test_pending_great_library():
    # Great Library reveals 3 random box tokens under gamestate.args._private.
    # Shape captured live from table 887928521. Use tokens that are genuinely in
    # this fixture's box (not on the board / not taken) so the codec reconciles.
    data = _age3_pending("chooseProgressTokenFromBox", active="89146710")
    data["gamestate"]["args"] = {"_private": {"progressTokensFromBox": {
        "3": {"id": "3", "type": "Economy"},
        "6": {"id": "6", "type": "Mathematics"},
        "9": {"id": "9", "type": "Theology"},
    }}}
    obs = wire_from_bga(data)["observation"]
    pc = obs["pending_choice"]
    assert pc["kind"] == "choose_unused_progress"
    assert pc["consume_all_options"] is True
    # sorted by progress-token id: Architecture=2 < Economy=3 < Mathematics=6 ...
    assert pc["options"] == ["Economy", "Mathematics", "Theology"]
    assert _resolve_actions(obs) == {"Economy", "Mathematics", "Theology"}


def test_pending_great_library_missing_private_raises():
    # Captured on a spectator/opponent client: no _private -> loud, not wrong.
    data = _age3_pending("chooseProgressTokenFromBox", active="89146710")
    with pytest.raises(UnsupportedBgaState, match="progressTokensFromBox"):
        wire_from_bga(data)


def test_age3_tableau_backs_and_end_to_end():
    # Force the supported main-turn state on the real Age III position (test just
    # picked a starter); everything else is the genuine captured board.
    data = _load_age3()
    data["gamestate"]["name"] = "playerTurn"
    data["gamestate"]["active_player"] = data["startPlayerId"]

    wire = wire_from_bga(data)
    obs = wire["observation"]
    assert obs["age"] == 3
    present = {tuple(c["slot_id"]): c for c in obs["tableau"] if c["present"]}
    # 20-slot Age III structure, all cards still on the board
    assert len(obs["tableau"]) == 20 and len(present) == 20
    # face-down backs split by sprite: 2 guild-backs + 6 age-III-backs
    facedown = [c for c in present.values() if not c["revealed"]]
    assert sum(c["back"] == "guild" for c in facedown) == 2
    assert sum(c["back"] == "age_iii" for c in facedown) == 6
    # the one revealed guild (Builders Guild) carries the guild back too
    revealed_guild = [c for c in present.values() if c["card_name"] == "Builders Guild"]
    assert len(revealed_guild) == 1 and revealed_guild[0]["back"] == "guild"

    # captured military token (+4, slot 3 zeroed) is gone; three remain
    assert sorted(obs["military_tokens_remaining"]) == [[-7, 5], [-4, 2], [7, 5]]

    # feeds the scrape codec: guild/age-III pool split must reconcile
    parsed = observation_from_wire(obs)
    state = determinize_observation(parsed, random.Random(0))
    assert state.age == 3
    # 3 guilds selected this game (1 visible + 2 face-down), 4 guilds unused
    assert len(state.selected_guilds) == 3
    assert len(state.unused_guilds) == 4


# --- Live Age III capture (table 892846644) -------------------------------
#
# Captured mid-game from a real table, in two forms at the same position:
#   *_reference.json  taken right after an F5, so `bga` is a fresh gamedatas
#   *_patched.json    taken with the page open across the whole game, so `bga`
#                     is the stale draft-time payload and `dom` holds the DOM
#                     re-read of the five fields BGA never refreshes
# The pair is the reload-equivalence evidence for the freshness patch, and the
# position is richer than the older fixtures: Age III with both card backs, a
# 15-card discard, and a taken progress token.

_FIXTURE_LIVE_REF = _TESTDATA / "bga_892846644_age3_reference.json"
_FIXTURE_LIVE_PATCHED = _TESTDATA / "bga_892846644_age3_patched.json"


def _load_live_reference() -> dict:
    return json.loads(_FIXTURE_LIVE_REF.read_text(encoding="utf-8"))


def test_bga_title_cased_card_name_is_normalized():
    # BGA spells this "Chamber Of Commerce"; the engine follows the printed card.
    # Before normalization this raised a bare KeyError out of back_type_of.
    data = _load_live_reference()["bga"]
    wire = wire_from_bga(data)
    names = {slot["card_name"] for slot in wire["observation"]["tableau"]}
    names |= set(wire["observation"]["discard_pile"])
    for city in wire["observation"]["cities"]:
        names |= set(city["buildings"])
    assert "Chamber Of Commerce" not in names
    assert "Chamber of Commerce" in names


def test_unknown_card_name_raises_typed():
    data = _load_live_reference()["bga"]
    # Rename a card that is actually in play, so the mapper has to resolve it.
    pid = next(iter(data["playerBuildings"]))
    in_play = data["playerBuildings"][pid][0]
    data["buildings"][str(in_play["id"])]["name"] = "Ministry Of Silly Walks"
    in_play["type"] = "Ministry Of Silly Walks"
    with pytest.raises(UnsupportedBgaState, match="unknown BGA card name"):
        wire_from_bga(data)


def test_live_age3_guild_backs_and_determinization():
    """The Age III deck carries exactly 3 guilds; backs must account for all of
    them, and determinization must preserve the 3-selected/4-unused split."""
    wire = wire_from_bga(_load_live_reference()["bga"])
    obs_wire = wire["observation"]
    tableau = obs_wire["tableau"]
    assert len(tableau) == 20

    guild_slots = [s for s in tableau if s["present"] and s["back"] == "guild"]
    hidden_guilds = [s for s in guild_slots if not s["revealed"]]
    # 2 revealed + 1 face-down == the 3 guilds BGA deals into the Age III deck.
    assert len(guild_slots) == 3
    assert len(hidden_guilds) == 1

    obs = observation_from_wire(obs_wire)
    for seed in range(8):
        state = determinize_observation(obs, random.Random(seed))
        assert state.observation(0) == obs           # public-exact
        assert len(state.selected_guilds) == 3
        assert len(state.unused_guilds) == 4


def test_live_patched_capture_is_stale_without_the_dom_patch():
    """The companion capture proves the freshness problem is real: taken without
    a reload, gamedatas still holds the draft-time board and must fail loudly."""
    patched = json.loads(_FIXTURE_LIVE_PATCHED.read_text(encoding="utf-8"))
    # BGA sends numbers as strings; the mapper's int() casts are load-bearing.
    assert patched["state"] == "playerTurn" and int(patched["age"]) == 3
    # Stale payload: no buildings, no discard -- but BGA reports science symbols.
    assert all(v == [] for v in patched["bga"]["playerBuildings"].values())
    assert patched["bga"]["discardedBuildings"] == []
    assert any(
        int(s["scienceSymbolCount"]) > 0
        for s in patched["bga"]["playersSituation"].values()
    )
    with pytest.raises(StaleGamedata):
        wire_from_bga(patched["bga"])

    # The DOM re-read carries the real board that gamedatas is missing.
    reference = _load_live_reference()
    assert patched["dom"] == reference["dom"]
    for pid, cards in reference["bga"]["playerBuildings"].items():
        assert sorted(int(c["id"]) for c in cards) == sorted(patched["dom"]["playerBuildings"][pid])


def test_dom_patch_reproduces_the_reference_wire():
    """The point of item B: a capture taken WITHOUT a reload, patched from the
    DOM, must yield exactly the wire a freshly loaded page produces."""
    from .bga_extract import wire_from_bga_payload

    patched = json.loads(_FIXTURE_LIVE_PATCHED.read_text(encoding="utf-8"))
    reference = _load_live_reference()
    assert wire_from_bga_payload(patched) == wire_from_bga(reference["bga"])


def test_dom_patch_unknown_id_raises_typed():
    from .bga_extract import wire_from_bga_payload

    patched = json.loads(_FIXTURE_LIVE_PATCHED.read_text(encoding="utf-8"))
    pid = next(iter(patched["dom"]["playerBuildings"]))
    patched["dom"]["playerBuildings"][pid].append(9999)
    with pytest.raises(UnsupportedBgaState, match="unknown building id"):
        wire_from_bga_payload(patched)


def test_wire_from_bga_payload_without_dom_is_plain_mapping():
    from .bga_extract import wire_from_bga_payload

    reference = _load_live_reference()
    assert wire_from_bga_payload({"bga": reference["bga"]}) == wire_from_bga(
        reference["bga"]
    )


def test_adapter_accepts_the_bga_envelope():
    from .advisor_adapter import SevenWondersAdvisor

    patched = json.loads(_FIXTURE_LIVE_PATCHED.read_text(encoding="utf-8"))
    position = SevenWondersAdvisor().state_from_wire(
        {"bga": patched["bga"], "args": patched["args"], "dom": patched["dom"]}
    )
    assert position.game.phase is Phase.PLAY_AGE
    assert position.game.age == 3


# --- Wonder burials -------------------------------------------------------
#
# Constructing a wonder buries an age card under it permanently. gamedatas gives
# only the card's *age* (wondersSituation[...]["constructed"]); the identity
# lives in the constructWonder game-log line. Tier 1 (age only) keeps the pool
# arithmetic correct; tier 2 (log) makes it exact.

_FIXTURE_GREAT_LIBRARY = _TESTDATA / "bga_892846644_greatlibrary.json"


def _load_great_library() -> dict:
    return json.loads(_FIXTURE_GREAT_LIBRARY.read_text(encoding="utf-8"))


def _assert_roundtrip_modulo_midmove_flip(state, obs):
    """``observation(0) == obs`` except for a CARD_REVEAL resolved on rebuild.

    A capture taken mid-move (a pending choice, with the triggering card already
    off the structure) catches BGA between the take and the flip: the slot it
    uncovered is still face-down and marked unavailable. The engine reveals such
    a slot as a CARD_REVEAL chance event fired with that same action
    (``TableauState.take_accessible``), and treats accessible-but-face-down as a
    hard error -- ``encode_action`` raises on it. ``determinize_observation``
    therefore resolves the pending reveal, sampling the identity like any other
    face-down slot. So the rebuilt slot legitimately reads revealed/accessible
    with a card where the raw capture read face-down/None.
    """
    produced = state.observation(0)
    by_slot = {c.slot_id: c for c in obs.tableau}
    resolved = set()
    for card in produced.tableau:
        other = by_slot[card.slot_id]
        if card == other:
            continue
        if (
            card.present and other.present
            and card.revealed and not other.revealed
            and card.accessible and not other.accessible
            and card.card_name is not None and other.card_name is None
        ):
            resolved.add(card.slot_id)
    strip = lambda t: tuple(c for c in t if c.slot_id not in resolved)  # noqa: E731
    assert strip(produced.tableau) == strip(obs.tableau)
    for field in ("cities", "discard_pile", "wonder_burials", "conflict_position",
                  "available_progress_tokens", "pending_choice", "age"):
        assert getattr(produced, field) == getattr(obs, field)
    return resolved


def test_burials_tier2_identified_from_the_game_log():
    from .bga_extract import _wonder_burials

    payload = _load_great_library()
    exact, unknown = _wonder_burials(payload["bga"], payload["log"])
    assert unknown == ()
    assert dict(exact)["The Great Library"] == "Chamber of Commerce"
    assert dict(exact)["Piraeus"] == "Study"
    # One per constructed wonder, no more.
    constructed = sum(
        1
        for pid, rows in payload["bga"]["wondersSituation"].items()
        if pid != "selection"
        for r in rows
        if int(r.get("constructed") or 0)
    )
    assert len(exact) == constructed


def test_burials_tier1_falls_back_to_ages_without_a_log():
    from .bga_extract import _wonder_burials

    payload = _load_great_library()
    exact, unknown = _wonder_burials(payload["bga"], None)
    assert exact == []
    assert sorted(unknown) == [2, 2, 2, 2, 3, 3]


def test_burials_tier1_rejects_a_log_that_disagrees_with_the_age():
    """A mis-parsed / mistranslated line must degrade to 'unidentified', never
    silently seed a wrong card."""
    from .bga_extract import _wonder_burials

    payload = _load_great_library()
    bogus = [
        line.replace("Chamber Of Commerce", "Stone Pit")  # Stone Pit is age 1
        for line in payload["log"]
    ]
    exact, unknown = _wonder_burials(payload["bga"], bogus)
    assert "The Great Library" not in dict(exact)
    assert 3 in unknown


def test_buried_cards_never_occupy_a_facedown_slot_when_identified():
    from .bga_extract import wire_from_bga_payload

    payload = _load_great_library()
    wire = wire_from_bga_payload(payload)
    buried = {card for _w, card in wire["observation"]["wonder_burials"]}
    assert buried  # this position has six
    obs = observation_from_wire(wire["observation"])
    for seed in range(8):
        state = determinize_observation(
            obs,
            random.Random(seed),
            unknown_burial_ages=tuple(wire.get("unknown_burial_ages", ())),
        )
        facedown = {
            c.card_name
            for c in state.tableau.cards.values()
            if c.present and not c.revealed
        }
        assert not (facedown & buried)
        _assert_roundtrip_modulo_midmove_flip(state, obs)


def test_burials_tier1_determinizes_instead_of_raising():
    """Before this fix an Age III burial raised ValueError out of the codec."""
    from .bga_extract import wire_from_bga_payload

    payload = {k: v for k, v in _load_great_library().items() if k != "log"}
    wire = wire_from_bga_payload(payload)
    assert sorted(wire["unknown_burial_ages"]) == [2, 2, 2, 2, 3, 3]
    obs = observation_from_wire(wire["observation"])
    state = determinize_observation(
        obs, random.Random(0), unknown_burial_ages=tuple(wire["unknown_burial_ages"])
    )
    _assert_roundtrip_modulo_midmove_flip(state, obs)


def test_midmove_pending_reveal_is_resolved_on_rebuild():
    """Exactly one slot was uncovered by the move in progress and not yet
    flipped by BGA; the rebuild must resolve it, or encode_action raises
    "slot holds no revealed card" as soon as search descends past the pending
    choice (which is what broke the first real end-to-end run)."""
    from .bga_extract import wire_from_bga_payload

    payload = _load_great_library()
    wire = wire_from_bga_payload(payload)
    obs = observation_from_wire(wire["observation"])
    state = determinize_observation(
        obs, random.Random(0),
        unknown_burial_ages=tuple(wire.get("unknown_burial_ages", ())),
    )
    lag = _assert_roundtrip_modulo_midmove_flip(state, obs)
    assert len(lag) == 1
    slot = next(iter(lag))
    produced = {c.slot_id: c for c in state.observation(0).tableau}[slot]
    assert produced.present and produced.revealed and produced.accessible
    assert produced.card_name is not None
    # The whole point: the codec can now encode moves from this slot.
    from .codec import legal_action_indices
    assert legal_action_indices(state)


def test_buried_cards_are_not_offered_as_card_reveal_outcomes():
    """The searcher enumerates CARD_REVEAL outcomes from unseen_pool, which is a
    different path from the determinizer. A card buried under a wonder can never
    be revealed, so it must not appear as a chance outcome either."""
    from .bga_extract import wire_from_bga_payload
    from .data import BackType
    from .pool import enumerate_card_reveal, unseen_pool

    payload = _load_great_library()
    wire = wire_from_bga_payload(payload)
    obs = observation_from_wire(wire["observation"])
    state = determinize_observation(
        obs, random.Random(0),
        unknown_burial_ages=tuple(wire.get("unknown_burial_ages", ())),
    )
    buried = set(state.wonder_burials.values())
    assert buried

    candidates = {
        name
        for name, _p in enumerate_card_reveal(
            unseen_pool(state.observation(0)), BackType.AGE_III
        )
    }
    assert not (candidates & buried)
    # ...and still covers every card actually face-down on the board.
    facedown = {
        c.card_name for c in state.tableau.cards.values()
        if c.present and not c.revealed
    }
    assert facedown <= candidates


def test_end_to_end_search_runs_on_a_real_capture():
    """The regression the first live run caught: search must descend past the
    pending choice without encode_action raising on an unflipped slot. Uses a
    stub evaluator so the test needs no checkpoint."""
    import threading
    from types import SimpleNamespace

    from .advisor_adapter import SevenWondersAdvisor
    from .inference import Evaluator
    from .train import build_model

    payload = _load_great_library()
    advisor = SevenWondersAdvisor(
        evaluator=Evaluator(build_model("transformer", 32, 1), "cpu")
    )
    position = advisor.state_from_wire(
        {k: payload[k] for k in ("bga", "args", "dom", "log")}
    )
    req = SimpleNamespace(
        engine="nn", seed=0, options={}, max_sims=32,
        checkpoint_path=None, device="cpu", top_k=5, temperature=0.0,
    )
    handle = advisor.open_search(position, req)
    snapshot = handle.advance(32, threading.Event())
    handle.close()
    assert snapshot.sims_done == 32
    assert set(snapshot.entries) == {v.action_id for v in advisor.action_views(position)}


def test_streaming_search_deepens_a_cumulative_tree():
    """Continuous-search contract: a streaming job publishes updated snapshots
    while it runs, and the tree is cumulative (sims accumulate across chunks
    rather than restarting), so displayed win probabilities refine over time."""
    import time

    from games.advisor.contract import RecommendRequest
    from games.advisor.jobs import JobManager

    from .advisor_adapter import SevenWondersAdvisor
    from .inference import Evaluator
    from .train import build_model

    payload = _load_great_library()
    advisor = SevenWondersAdvisor(
        evaluator=Evaluator(build_model("transformer", 32, 1), "cpu")
    )
    state = advisor.state_from_wire(
        {k: payload[k] for k in ("bga", "args", "dom", "log")}
    )
    jobs = JobManager(advisor, chunk_default=20)
    job = jobs.start(
        state,
        RecommendRequest(engine="nn", max_sims=100_000, chunk_sims=20, top_k=8),
    )
    try:
        seen = []
        deadline = time.time() + 10.0
        while time.time() < deadline and len(seen) < 3:
            time.sleep(0.35)
            snap = job.snapshot
            if snap is not None and (not seen or snap.sims_done > seen[-1]):
                seen.append(snap.sims_done)
        assert len(seen) >= 3, f"no streaming progress: {seen}"
        assert seen == sorted(seen) and seen[-1] > seen[0]
        # Cumulative: far more sims than a single chunk.
        assert seen[-1] > 20
    finally:
        job._stop.set()
        deadline = time.time() + 5.0
        while job.active and time.time() < deadline:
            time.sleep(0.05)
    assert not job.active


def test_dom_ahead_of_science_count_is_accepted():
    """The failure a live game hit: "4 science buildings but science count = 3".

    `scienceSymbolCount` counts science CARDS, not distinct symbols -- confirmed
    on captures where a player holds a duplicated symbol (4 cards / 3 distinct,
    reported as 4), so a duplicate cannot cause this. What does: BGA places a
    built card into the DOM immediately (`notif_constructBuilding`) while
    `scienceSymbolCount` refreshes only on the next state entry, so right after
    an opponent builds a green card the DOM legitimately leads by one.

    Refusing that would blind the advisor for a turn every time the opponent
    takes a science card, and the DOM's position is the more current of the two.
    """
    import copy

    from .bga_extract import wire_from_bga_payload

    reference = _load_live_reference()
    payload = copy.deepcopy({"bga": reference["bga"], "dom": reference["dom"]})
    for situation in payload["bga"]["playersSituation"].values():
        situation["scienceSymbolCount"] = int(situation["scienceSymbolCount"]) - 1
    wire = wire_from_bga_payload(payload)  # must not raise
    assert wire["observation"]["cities"]


def test_dom_behind_science_count_is_still_stale():
    """The direction the check exists for must keep failing: gamedatas left at
    its page-load value while playersSituation stays fresh."""
    import copy

    from .bga_extract import wire_from_bga_payload

    reference = _load_live_reference()
    payload = copy.deepcopy({"bga": reference["bga"], "dom": reference["dom"]})
    for pid in payload["dom"]["playerBuildings"]:
        payload["dom"]["playerBuildings"][pid] = []
    with pytest.raises(StaleGamedata):
        wire_from_bga_payload(payload)
