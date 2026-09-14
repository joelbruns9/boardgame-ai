"""Mid-move captures with cards BGA has not flipped yet.

BGA flips cards uncovered during a move only when the next turn starts, so at a
mid-move choice they are unknown. The advisor used to fill them with one random
guess and search it as if the cards were face up; on table 915281579 that guess
was ZeusAI's science pair and a 52% position read 24%. Now the position is
searched over several boards whose guesses cover the unseen pool evenly, and
the results are merged with equal weight.

These protect: detection only fires on unflipped uncovered cards; every board is
the captured position with a valid deal; the guesses are stratified, not
independent; the merge is equal-weight and honest about outlook coverage; the
exact solver declines to prove anything about a guess.
"""

from __future__ import annotations

import collections
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from games.advisor import ActionStats, SearchSnapshot

from .advisor_adapter import (
    ADVISOR_REVEAL_WORLDS,
    SevenWondersAdvisor,
    _MergedWorldsHandle,
    merge_world_snapshots,
    reveal_world_count,
    unflipped_uncovered_slots,
)
from .bga_extract import wire_from_bga_payload
from .codec import legal_action_indices
from .inference import Evaluator
from .net import SWDNet

_TESTDATA = Path(__file__).parent / "testdata"


def _capture(name):
    raw = json.loads((_TESTDATA / f"{name}.json").read_text(encoding="utf-8"))
    return {k: raw[k] for k in ("bga", "args", "dom", "log") if k in raw}


def _adapter():
    torch.manual_seed(2)
    return SevenWondersAdvisor(
        evaluator=Evaluator(SWDNet(d_model=32, layers=1, heads=4), "cpu", 64, fuse_embedder=False)
    )


def _request(**options):
    return SimpleNamespace(
        engine="nn", seed=0, options=dict(options), max_sims=160,
        checkpoint_path=None, device="cpu", top_k=8, temperature=0.0,
    )


# --- detection ---------------------------------------------------------------


def test_the_mid_move_capture_has_an_unflipped_card_and_a_normal_one_has_none():
    mid_move = wire_from_bga_payload(_capture("bga_892846644_greatlibrary"))["observation"]
    assert unflipped_uncovered_slots(mid_move) == ((3, 4),)
    normal = wire_from_bga_payload(_capture("bga_892846644_age3_reference"))["observation"]
    assert unflipped_uncovered_slots(normal) == ()


def test_a_covered_face_down_card_is_not_unflipped():
    observation = {
        "age": 2,
        "tableau": [
            {"slot_id": [3, 6], "present": True, "revealed": False},
            {"slot_id": [4, 5], "present": True, "revealed": True},   # still covers (3,6)
            {"slot_id": [4, 7], "present": False, "revealed": False},
        ],
    }
    assert unflipped_uncovered_slots(observation) == ()
    observation["tableau"][1]["present"] = False
    assert unflipped_uncovered_slots(observation) == ((3, 6),)


# --- the boards --------------------------------------------------------------


@pytest.fixture(scope="module")
def mid_move_state():
    adapter = _adapter()
    return adapter, adapter.state_from_wire(_capture("bga_892846644_greatlibrary"))


def test_the_capture_is_searched_over_several_boards(mid_move_state):
    _adapter, state = mid_move_state
    assert state.unflipped_slots == ((3, 4),)
    assert len(state.worlds) >= ADVISOR_REVEAL_WORLDS
    assert state.game is state.worlds[0]


def test_every_board_is_the_captured_position_apart_from_the_guesses(mid_move_state):
    _adapter, state = mid_move_state
    legal = legal_action_indices(state.worlds[0])
    reference = state.worlds[0].observation(0)
    for board in state.worlds:
        assert legal_action_indices(board) == legal
        observation = board.observation(0)
        assert observation.cities == reference.cities
        assert observation.pending_choice == reference.pending_choice
        assert observation.conflict_position == reference.conflict_position
        for mine, theirs in zip(observation.tableau, reference.tableau):
            if mine.slot_id not in state.unflipped_slots:
                assert mine == theirs


def test_every_board_is_a_valid_deal(mid_move_state):
    """The guesses are SWAPS with wherever that card was hidden, so every board
    holds the same cards -- only where they sit differs."""

    _adapter, state = mid_move_state

    def hidden_multiset(board):
        age = board.age
        names = [c.card_name for c in board.tableau.cards.values() if c.present]
        names += list(board.removed_age_cards.get(age, ()))
        if age == 3:
            names += list(board.unused_guilds)
        return collections.Counter(names)

    reference = hidden_multiset(state.worlds[0])
    for board in state.worlds[1:]:
        assert hidden_multiset(board) == reference


def test_the_guesses_cover_the_unseen_pool_evenly(mid_move_state):
    _adapter, state = mid_move_state
    from .advisor_adapter import _hidden_locations, _is_guild_back, _name_at

    (slot,) = state.unflipped_slots
    observation = wire_from_bga_payload(_capture("bga_892846644_greatlibrary"))["observation"]
    backs = {tuple(c["slot_id"]): c.get("back") for c in observation["tableau"]}
    guild = _is_guild_back(backs[slot])
    pool = {
        _name_at(state.worlds[0], loc)
        for loc in _hidden_locations(state.worlds[0], backs, state.unflipped_slots, guild)
    }
    counts = collections.Counter(board.tableau.cards[slot].card_name for board in state.worlds)
    # Round-robin from one shuffled order: as many distinct cards as there are
    # boards (or the whole pool), none repeated more than once beyond the rest.
    assert len(counts) == min(len(state.worlds), len(pool))
    assert max(counts.values()) - min(counts.values()) <= 1


def test_the_board_count_grows_with_the_pool_up_to_a_cap():
    assert reveal_world_count(9, 2) == ADVISOR_REVEAL_WORLDS
    assert reveal_world_count(14, 1) == 14
    assert reveal_world_count(40, 1) == 16


# --- searching them ----------------------------------------------------------


def test_the_boards_are_searched_and_merged(mid_move_state):
    adapter, state = mid_move_state
    handle = adapter.open_search(state, _request())
    try:
        assert isinstance(handle, _MergedWorldsHandle)
        snapshot = handle.advance(160, threading.Event())
    finally:
        handle.close()
    assert snapshot.sims_done >= 160
    assert set(snapshot.entries) == {str(a) for a in legal_action_indices(state.game)}


def test_one_board_can_still_be_asked_for(mid_move_state):
    adapter, state = mid_move_state
    handle = adapter.open_search(state, _request(reveal_worlds=1))
    try:
        assert not isinstance(handle, _MergedWorldsHandle)
    finally:
        handle.close()


def test_a_normal_position_is_one_board():
    adapter = _adapter()
    state = adapter.state_from_wire(_capture("bga_892846644_age3_reference"))
    assert state.worlds == ()
    handle = adapter.open_search(state, _request())
    try:
        assert not isinstance(handle, _MergedWorldsHandle)
    finally:
        handle.close()


def _snap(root, entries, outlook=None):
    return SearchSnapshot(
        sims_done=sum(e.visits for e in entries.values()),
        sims_target=0,
        root_value=root,
        entries=entries,
        root_outlook=outlook,
    )


def test_the_merge_weights_every_board_equally():
    a = _snap(0.2, {"1": ActionStats(90, 0.1, 0.5, outlook={"x": 0.2}), "2": ActionStats(10, -0.3, 0.5)}, {"x": 0.4})
    b = _snap(-0.4, {"1": ActionStats(10, -0.5, 0.7, outlook={"x": 0.6}), "2": ActionStats(0, 0.0, 0.3)}, {"x": 0.8})
    merged = merge_world_snapshots([a, b], target=200)
    assert merged.root_value == pytest.approx(-0.1)
    assert merged.root_outlook == {"x": pytest.approx(0.6)}
    one = merged.entries["1"]
    # Visit-weighted would be -0.04; each board counts once.
    assert (one.visits, one.q_value, one.prior) == (100, pytest.approx(-0.2), pytest.approx(0.6))
    assert one.outlook == {"x": pytest.approx(0.4)}
    two = merged.entries["2"]
    # Unsearched on board b: its Q is board a's alone, and a board that
    # searched the move without an outlook withholds the merged one.
    assert two.q_value == pytest.approx(-0.3)
    assert two.outlook is None


# --- the exact solver --------------------------------------------------------


def test_the_exact_solver_declines_a_position_with_unflipped_cards(mid_move_state):
    from .advisor_endgame import ExactEndgameAnnotator

    _adapter, state = mid_move_state
    result = ExactEndgameAnnotator().annotate(
        state, [], _request(), deadline=None, stop_event=None
    )
    assert result.summary["status"] == "skipped"
    assert result.summary["reason"] == "unflipped_cards"


def test_the_public_state_says_the_advice_is_averaged(mid_move_state):
    adapter, state = mid_move_state
    public = adapter.state_to_public(state)
    assert public["unflipped_cards"] == 1
    assert public["reveal_worlds"] == len(state.worlds)
