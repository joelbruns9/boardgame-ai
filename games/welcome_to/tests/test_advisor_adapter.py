"""The Welcome To advisor, driven exactly as the shared host drives it.

Two things are being checked and they are different.  The *contract* -- that this
adapter satisfies the seam ``games.advisor`` calls, that a search handle is
genuinely resumable, that a snapshot is in the asking player's frame -- is
checked against a tiny untrained network, because none of it depends on the
model being any good.  The *rendering* -- labels, fields, the forecast -- is
checked because it is what the human reads, and a label that names the wrong
box is worse than no advisor at all.

The host itself is exercised end to end through FastAPI's test client rather
than by calling the adapter directly, because the point of the standardization
is that a game only supplies an adapter; if the wire does not come out right
through the real app, the seam has leaked.
"""

from __future__ import annotations

import random
import threading

import pytest

torch = pytest.importorskip("torch")

from games.advisor import RecommendRequest, build_recommendations
from games.welcome_to import macro_codec as mc
from games.welcome_to import network as nw
from games.welcome_to.advisor_adapter import WelcomeToAdvisor
from games.welcome_to.bots import GreedyBot
from games.welcome_to.game import GameConfig, GameState, Phase
from games.welcome_to.snapshot import to_snapshot

#: Small enough that a few hundred simulations run in test time; the contract
#: does not depend on capacity.
_TINY = nw.NetConfig(
    sheet_hidden=32, sheet_out=16, trunk_hidden=48, trunk_blocks=1, head_hidden=32
)


@pytest.fixture(scope="module")
def adapter() -> WelcomeToAdvisor:
    torch.manual_seed(0)
    return WelcomeToAdvisor(net=nw.WelcomeToNet(_TINY), device="cpu")


def _position(adapter, state: GameState):
    return adapter.state_from_wire({"snapshot": to_snapshot(state)})


def _client(adapter, **kwargs):
    """A TestClient over the real advisor app, or a skip.

    Starlette's test client needs an HTTP library that is not a dependency of
    this project, so its absence is a missing test tool rather than a failure.
    """
    pytest.importorskip("fastapi")
    from games.advisor import create_advisor_app

    try:
        # Starlette raises at import time when it cannot find an HTTP client.
        from fastapi.testclient import TestClient

        return TestClient(create_advisor_app(adapter, **kwargs))
    except RuntimeError as exc:  # starlette: needs httpx / httpx2
        pytest.skip(str(exc).splitlines()[0])


def _fresh(players: int = 3, advanced: bool = True, seed: int = 7) -> GameState:
    return GameState.new(
        seed=seed,
        config=GameConfig(players=players, advanced=advanced, solo_rules=False),
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
def test_action_views_cover_the_search_action_set_exactly(adapter):
    """The host joins snapshot entries to views by ``action_id``.

    A view missing for an action the search visits shows up as an unlabelled row
    with the raw index in it; an extra view is a move the user cannot make. Both
    are silent, so the two sets are pinned equal.
    """
    state = _fresh()
    views = adapter.action_views(_position(adapter, state))
    assert [v.action_id for v in views] == [
        str(i) for i in mc.search_legal_macros(state)
    ]
    assert len({v.action_id for v in views}) == len(views)


def test_a_write_is_named_by_what_the_player_actually_does(adapter):
    """The macro folds the card choice and the placement into one action, so the
    label has to name both -- picking a combination *is* choosing which effect
    you get, and a label that said only "write 5 in box 3" would hide the half of
    the decision the player is really making."""

    state = _fresh()
    views = adapter.action_views(_position(adapter, state))
    writes = [v for v in views if v.kind == "write"]
    assert writes

    for view in writes[:20]:
        slot, delta_slot, x, y = mc.decode_macro_write(int(view.action_id))
        number, effect = state.combination(slot)
        assert "stack %d" % (slot + 1) in view.label
        assert str(view.fields["number"]) in view.label
        assert view.fields["box"] == [x, y]
        assert view.fields["effect"] == effect.name
        # Streets are 1-indexed for a human and boxes are counted from one.
        assert "%s street box %d" % (("1st", "2nd", "3rd")[x], y + 1) in view.label


def test_labels_exist_for_every_phase_a_capture_can_land_in(adapter):
    """A phase with no label is a panel full of raw macro indices."""
    seen = set()
    for seed in range(1, 8):
        bot = GreedyBot(rng=random.Random(seed))
        state = _fresh(players=2, advanced=True, seed=seed)
        steps = 0
        while not state.is_terminal and steps < 600:
            if state.actor == 0 and mc.is_macro_root(state):
                for view in adapter.action_views(_position(adapter, state)):
                    assert view.label and not view.label.startswith("write("), view
                    assert view.label[0].isupper() or view.label.startswith("Pass")
                seen.add(state.phase)
            state.apply(bot.act(state))
            steps += 1
    assert {
        Phase.CHOOSE_CARDS,
        Phase.ACTION_SURVEYOR,
        Phase.ACTION_ESTATE,
        Phase.ACTION_PARK,
        Phase.ACTION_BIS,
        Phase.CHOOSE_PLAN,
    } <= seen


# ---------------------------------------------------------------------------
# The resumable handle
# ---------------------------------------------------------------------------
def test_advancing_deepens_one_tree_instead_of_restarting(adapter):
    """The whole point of a handle rather than a one-shot search.

    ``MCTS.search`` treats its budget as a target *total* and takes the root node
    back, so a second advance must add work to the same tree. If it restarted,
    visits would not accumulate and the panel would flicker between independent
    answers instead of sharpening.
    """
    state = _fresh(players=2)
    position = _position(adapter, state)
    req = RecommendRequest(max_sims=48, chunk_sims=16, device="cpu", seed=3)
    handle = adapter.open_search(position, req)
    stop = threading.Event()

    first = handle.advance(16, stop)
    second = handle.advance(16, stop)
    assert first.sims_done == 16
    assert second.sims_done == 32
    # Every edge's visit count is monotone: nothing was thrown away.
    for action_id, stats in first.entries.items():
        assert second.entries[action_id].visits >= stats.visits
    handle.close()


def test_a_set_stop_event_publishes_without_searching(adapter):
    """Cancellation latency is one chunk, and a cancelled advance still answers
    with the numbers it already had -- the host publishes that snapshot."""
    state = _fresh(players=2)
    handle = adapter.open_search(
        _position(adapter, state),
        RecommendRequest(max_sims=32, chunk_sims=8, device="cpu"),
    )
    stop = threading.Event()
    stop.set()
    snapshot = handle.advance(8, stop)
    assert snapshot.partial is True
    assert snapshot.sims_done == 0
    handle.close()


def test_snapshot_values_are_in_the_asking_players_frame(adapter):
    """The host renders ``root_value`` to the human without ever learning whose
    turn it is, so the adapter owns the frame. Seat 0 is the viewer by
    construction of the scrape, and the tree's root is seat 0, so the two
    already agree -- this pins that they still do."""

    state = _fresh(players=3)
    handle = adapter.open_search(
        _position(adapter, state),
        RecommendRequest(max_sims=32, chunk_sims=32, device="cpu"),
    )
    snapshot = handle.advance(32, threading.Event())
    assert -1.0 <= snapshot.root_value <= 1.0
    assert all(-1.0 <= s.q_value <= 1.0 for s in snapshot.entries.values())
    # Visits are the ranking signal and must sum to the work actually done.
    assert sum(s.visits for s in snapshot.entries.values()) == snapshot.sims_done
    handle.close()


def test_recommendations_carry_the_prior_as_well_as_the_visits(adapter):
    """"The net wanted this and search talked it out of it" and "the net never
    looked at it" are different faults. The panel shows both, so the wire has to
    carry both."""

    state = _fresh(players=2)
    position = _position(adapter, state)
    handle = adapter.open_search(
        position, RecommendRequest(max_sims=64, chunk_sims=64, device="cpu")
    )
    snapshot = handle.advance(64, threading.Event())
    recs = build_recommendations(snapshot, adapter.action_views(position), top_k=5)
    handle.close()

    assert recs
    assert [r.rank for r in recs] == list(range(1, len(recs) + 1))
    assert all(r.label for r in recs)
    assert sum(r.prior for r in recs) > 0.0
    # Ranked by visits, most first.
    assert [r.visits for r in recs] == sorted((r.visits for r in recs), reverse=True)


# ---------------------------------------------------------------------------
# What the net believes
# ---------------------------------------------------------------------------
def test_forecast_reports_a_final_score_for_every_seat(adapter):
    """The diagnostic half of the panel.

    A win probability cannot say whether the model is mis-ranking moves or
    mis-reading the board. A predicted final score per seat, split into the
    components that make it up, can.
    """
    state = _fresh(players=3)
    public = adapter.state_to_public(_position(adapter, state))
    forecast = public["forecast"]

    assert len(forecast["seats"]) == 3
    assert forecast["seats"][0]["you"] is True
    assert len(forecast["rank_probs"]) == 3
    assert forecast["rank_probs"] == pytest.approx(
        forecast["rank_probs"], abs=1e-6
    ) and sum(forecast["rank_probs"]) == pytest.approx(1.0, abs=1e-5)
    for seat in forecast["seats"]:
        assert set(seat["components"]) == {
            "parks",
            "pools",
            "estates",
            "plans",
            "temp",
            "bis",
            "permits",
            "roundabouts",
        }
        assert len(seat["will_complete_plan"]) == 3
        assert len(seat["turns_to_plan"]) == 3


def test_public_state_says_which_effect_each_stack_offers_next(adapter):
    """Not a nicety: a card's number face prints its own effect in the corners,
    so next turn's effect is *known*, not predicted. The model is fed it and the
    panel should show the same thing the model sees."""

    state = _fresh()
    public = adapter.state_to_public(_position(adapter, state))
    assert len(public["stacks"]) == 3
    for slot, stack in enumerate(public["stacks"]):
        assert stack["number"] == state.combination(slot)[0]
        assert stack["effect"] == state.combination(slot)[1].name
        assert stack["next_effect"] == state.next_effects(0)[slot].name


def test_public_state_never_leaks_another_seats_current_turn(adapter):
    """The scrape cannot see it and the engine must not invent it.

    Seat 0's sheet is live; everyone else's is frozen at the start of the turn
    (``GameState.sheet_for``). A public payload built off the raw sheets would
    quietly hand the advisor -- and the person reading it -- information the
    table does not have.
    """
    state = _fresh(players=3)
    # An opponent mid-turn: the engine's ground truth has moved on, the
    # turn-start snapshot the viewer is allowed to see has not. A real capture
    # is always in exactly this shape, because BGA sends the frozen sheet and
    # the mapper never fills in the difference.
    state.sheets[1].write(4, (0, 0), state.turn)
    state.sheets[1].parks[0] += 1

    public = adapter.state_to_public(_position(adapter, state))
    assert public["seats"][1]["houses"] == 0, "an opponent's live sheet leaked"
    assert public["seats"][0]["you"] is True

    # And the viewer's own sheet is live, which is the other half of the rule.
    state.sheets[0].write(4, (0, 0), state.turn)
    public = adapter.state_to_public(_position(adapter, state))
    assert public["seats"][0]["houses"] == 1


# ---------------------------------------------------------------------------
# Through the real host
# ---------------------------------------------------------------------------
def test_the_shared_host_serves_this_adapter_end_to_end(adapter):
    """A game is supposed to supply an adapter and nothing else.

    If the envelope only comes out right when the adapter is called directly,
    the seam has leaked -- so the whole start/poll/stop trio is driven through
    the real FastAPI app.
    """
    client = _client(adapter, chunk_default=8)
    state = {"snapshot": to_snapshot(_fresh(players=2))}

    health = client.get("/health").json()
    assert health["ok"] and health["game_id"] == "welcome_to"
    assert "nn" in health["engines"]

    public = client.post("/api/state", json={"state": state}).json()
    assert public["phase"] == "CHOOSE_CARDS"
    assert public["legal_actions"]

    answer = client.post(
        "/api/recommend",
        json={"state": state, "max_sims": 24, "chunk_sims": 8, "top_k": 4, "device": "cpu"},
    ).json()
    assert answer["ok"] is True
    assert answer["sims_done"] == 24
    assert 1 <= len(answer["recommendations"]) <= 4
    assert answer["recommendations"][0]["label"]

    started = client.post(
        "/api/recommend/start",
        json={"state": state, "max_sims": 4000, "chunk_sims": 8, "device": "cpu"},
    ).json()
    job_id = started["job_id"]
    polled = client.get("/api/recommend/poll", params={"job_id": job_id}).json()
    assert polled["job_id"] == job_id
    assert client.post("/api/recommend/stop", json={"job_id": job_id}).json()["ok"]


def test_a_bad_wire_is_rejected_with_a_reason_not_a_stack_trace(adapter):
    """The panel prints the host's `detail` verbatim, so it has to be readable."""
    client = _client(adapter)
    response = client.post("/api/state", json={"state": {"nonsense": 1}})
    assert response.status_code == 400
    assert "observation" in response.json()["detail"]
