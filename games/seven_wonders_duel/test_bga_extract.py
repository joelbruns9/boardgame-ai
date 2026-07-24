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
    data["gamestate"]["name"] = "wonderDraft"
    with pytest.raises(UnsupportedBgaState, match="not a supported PLAY_AGE"):
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
    with pytest.raises(UnsupportedBgaState, match="not a supported PLAY_AGE"):
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
