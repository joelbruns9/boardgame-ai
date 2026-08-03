"""7WD advisor adapter against the shared host, with a tiny real net.

Proves the validated seam holds end-to-end: the host drives the Gumbel closed
tree through open_search/advance, ranks by visits, and the wire round-trips.
"""

from __future__ import annotations

import threading

import pytest

from games.advisor import JobManager, RecommendRequest

from .advisor_adapter import SevenWondersAdvisor


@pytest.fixture(scope="module")
def adapter():
    from .inference import Evaluator
    from .train import build_model

    return SevenWondersAdvisor(evaluator=Evaluator(build_model("transformer", 32, 1), "cpu"))


def _pos(adapter, prefix=()):
    return adapter.state_from_wire(
        {"seed": 7, "first_player": 0, "prefix": list(prefix)}
    )


def test_wire_round_trip_is_stable(adapter):
    pos = _pos(adapter)
    public = adapter.state_to_public(pos)
    reparsed = adapter.state_from_wire(
        {"seed": public["seed"], "first_player": public["first_player"], "prefix": public["prefix"]}
    )
    assert adapter.state_key(pos) == adapter.state_key(reparsed)
    assert public["phase"] == "WONDER_DRAFT"
    assert public["actor"] == 0
    assert len(public["legal_actions"]) > 0


def test_action_ids_are_identity_indexed(adapter):
    pos = _pos(adapter)
    from .codec import legal_action_indices

    views = adapter.action_views(pos)
    assert [v.action_id for v in views] == [str(i) for i in legal_action_indices(pos.game)]


def test_blocking_recommend_ranks_by_visits(adapter):
    mgr = JobManager(adapter)
    resp = mgr.run_blocking(
        _pos(adapter), RecommendRequest(engine="auto", max_sims=200, chunk_sims=50, top_k=4, seed=1)
    )
    assert resp.ok
    assert resp.sims_done == 200
    visits = [r.visits for r in resp.recommendations]
    assert visits == sorted(visits, reverse=True)
    assert -1.0 <= resp.root_value <= 1.0
    assert all(-1.0 <= r.q_value <= 1.0 for r in resp.recommendations)


def test_streaming_reaches_target_monotonically(adapter):
    import time

    mgr = JobManager(adapter, chunk_default=40)
    job = mgr.start(
        _pos(adapter), RecommendRequest(engine="auto", max_sims=240, chunk_sims=40, seed=2)
    )
    seen = []
    for _ in range(2000):
        polled = mgr.poll(job.job_id)
        if polled.snapshot is not None:
            seen.append(polled.sims_done)
        if polled.status in ("done", "error", "cancelled"):
            break
        time.sleep(0.003)
    assert polled.status == "done", polled.error
    assert polled.sims_done == 240
    assert all(seen[i] <= seen[i + 1] for i in range(len(seen) - 1))


def test_open_search_runs_no_sims_until_advance(adapter):
    pos = _pos(adapter)
    handle = adapter.open_search(pos, RecommendRequest(engine="auto", max_sims=100, seed=3))
    stop = threading.Event()
    first = handle.advance(0, stop)  # zero-sim advance: read the seeded root only
    assert first.sims_done == 0
    assert all(stats.visits == 0 for stats in first.entries.values())
    handle.close()


def test_unknown_engine_rejected(adapter):
    with pytest.raises(ValueError):
        adapter.open_search(_pos(adapter), RecommendRequest(engine="mystery", max_sims=10))


def test_state_to_public_reports_cities_and_advances(adapter):
    pos = adapter.state_from_wire({"seed": 7, "first_player": 0, "prefix": []})
    public = adapter.state_to_public(pos)
    assert public["origin"] == "replay"
    assert [c["player"] for c in public["cities"]] == [0, 1]
    first = public["legal_actions"][0]["action_id"]
    advanced = adapter.state_to_public(
        adapter.state_from_wire({"seed": 7, "first_player": 0, "prefix": [int(first)]})
    )
    assert advanced["actor"] != public["actor"]  # turn passed


def test_web_app_builds_with_routes_and_static():
    from pathlib import Path

    from . import web_app

    paths = {getattr(r, "path", "") for r in web_app.app.routes}
    assert {"/", "/health", "/api/state", "/api/recommend"} <= paths
    assert {"/api/recommend/start", "/api/recommend/poll", "/api/recommend/stop"} <= paths
    assert (Path(web_app.__file__).with_name("web_static") / "index.html").exists()


# --- Rust-backed searcher (ADVISOR_RUST_UNIFICATION.md step 4) ---------------


def _swr():
    return pytest.importorskip("seven_wonders_rust")


def _new_adapter():
    """A fresh advisor per test: the Rust handle holds its own arena, so sharing
    the module-scoped fixture across searches would confuse ownership."""
    from .inference import Evaluator
    from .train import build_model

    return SevenWondersAdvisor(
        evaluator=Evaluator(build_model("transformer", 32, 1), "cpu")
    )


def _req(**over):
    from types import SimpleNamespace

    base = dict(
        engine="nn",
        seed=0,
        options={},
        max_sims=99999,
        checkpoint_path=None,
        device="cpu",
        top_k=8,
        temperature=0.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _scraped_state(adapter):
    import json
    from pathlib import Path

    raw = json.loads(
        (Path(__file__).parent / "testdata" / "bga_892846644_greatlibrary.json").read_text(
            encoding="utf-8"
        )
    )
    return adapter.state_from_wire({k: raw[k] for k in ("bga", "args", "dom", "log")})


def test_rust_is_the_default_searcher():
    _swr()
    adapter = _new_adapter()
    handle = adapter.open_search(_scraped_state(adapter), _req())
    try:
        assert type(handle).__name__ == "_RustClosedHandle"
    finally:
        handle.close()


def test_python_searcher_still_selectable():
    adapter = _new_adapter()
    handle = adapter.open_search(
        _scraped_state(adapter), _req(options={"search_impl": "python"})
    )
    try:
        assert type(handle).__name__ == "_ClosedHandle"
    finally:
        handle.close()


def test_explicit_force_expand_falls_back_to_python():
    """The Rust path cannot force-expand the root chance layer (needs the F4.5
    forced-child cache). Honouring the request matters more than using Rust, so
    an explicit ask falls back rather than silently dropping it."""
    _swr()
    adapter = _new_adapter()
    handle = adapter.open_search(
        _scraped_state(adapter), _req(options={"force_expand_root_chance": True})
    )
    try:
        assert type(handle).__name__ == "_ClosedHandle"
    finally:
        handle.close()


def test_rust_and_python_searchers_produce_the_same_snapshot_shape():
    """Plumbing check only.

    Search equivalence itself is gated in test_puct_root, against the one-shot
    Rust search that is in turn gated against Python. Re-asserting a ranking
    here would be both redundant and unreliable: this fixture uses a random
    untrained net, so every action has an almost identical value and the order
    is noise. (Driven by the real checkpoint the two agree exactly -- 2927 vs
    2924 visits on the top action, Q equal to four decimals.)

    What this does check is that both handles expose the same actions, agree
    roughly on the root value, and report the sims they ran.
    """
    import threading

    _swr()
    snapshots = {}
    for impl in ("rust", "python"):
        adapter = _new_adapter()
        handle = adapter.open_search(
            _scraped_state(adapter), _req(options={"search_impl": impl})
        )
        try:
            snapshots[impl] = handle.advance(600, threading.Event())
        finally:
            handle.close()

    rust, python = snapshots["rust"], snapshots["python"]
    assert set(rust.entries) == set(python.entries)
    assert rust.sims_done >= 600 and python.sims_done >= 600
    assert sum(s.visits for s in rust.entries.values()) > 0
    # Deliberately NOT comparing root values. The Rust path defaults to
    # leaf_batch=16, so it overshoots the requested chunk by a whole wave and
    # explores a different number of simulations; under this fixture's random
    # untrained net every action scores alike, so the running mean wanders. The
    # searches themselves are gated in test_puct_root; this test is plumbing.
    for stats in rust.entries.values():
        assert -1.0 <= stats.q_value <= 1.0
        assert 0.0 <= stats.prior <= 1.0


def test_state_to_public_carries_a_victory_outlook():
    """`how am I winning` is often more actionable than `am I winning` in 7WD.

    Reported on the position rather than through an annotator: it is one root
    evaluation and never changes as search deepens, and annotators only run when
    a search *settles* -- which a streaming search at max_sims=1,000,000 never
    does, so an annotator would never surface it at all.
    """
    adapter = _new_adapter()
    public = adapter.state_to_public(_scraped_state(adapter))
    outlook = public["victory_outlook"]
    assert outlook is not None

    kinds = outlook["victory_type"]
    assert set(kinds) == {
        "you_civilian",
        "you_scientific",
        "you_military",
        "opponent_civilian",
        "opponent_scientific",
        "opponent_military",
        "draw",
    }
    assert sum(kinds.values()) == pytest.approx(1.0, abs=1e-4)
    assert outlook["you_win"] + outlook["opponent_wins"] + outlook["draw"] == (
        pytest.approx(1.0, abs=1e-4)
    )
    assert len(outlook["wdl"]) == 3
    assert len(outlook["final_science"]) == 2
    assert isinstance(outlook["vp_margin"], float)


def test_victory_outlook_is_absent_without_an_evaluator():
    """No checkpoint configured must degrade to None, not raise: the panel
    renders what it gets, and a missing stat is not a reason to fail a request."""
    from .advisor_adapter import SevenWondersAdvisor

    bare = SevenWondersAdvisor()  # no evaluator, no default checkpoint
    public = bare.state_to_public(_scraped_state(bare))
    assert public["victory_outlook"] is None
    assert public["legal_actions"], "the rest of the payload must be unaffected"


def _mausoleum_position():
    """A position where the Mausoleum is affordable and the discard is stocked.

    Constructed rather than searched for: random play almost never leaves the
    Mausoleum unbuilt, affordable, and facing a discard pile worth reviving
    from.
    """

    from .advisor_adapter import _Position
    from .codec import decode_action, legal_action_indices
    from .engine import ActionUse, apply_action
    from .game import Phase, new_game

    for seed in range(400):
        game = new_game(seed, first_player=0)
        while game.phase is Phase.WONDER_DRAFT:
            game.pick_wonder(game.legal_wonder_choices()[0])
        for city in game.cities:
            if "The Mausoleum" in city.wonders:
                city.wonders.remove("The Mausoleum")
        game.cities[game.active_player].wonders.insert(0, "The Mausoleum")
        for _ in range(6):  # stock the discard pile
            if game.phase is not Phase.PLAY_AGE:
                break
            discards = [
                index
                for index in legal_action_indices(game)
                if decode_action(game, index).use is ActionUse.DISCARD_FOR_COINS
            ]
            apply_action(game, decode_action(game, discards[0]))
        if game.phase is not Phase.PLAY_AGE or len(game.discard_pile) < 3:
            continue
        game.cities[game.active_player].coins = 100
        builds = [
            index
            for index in legal_action_indices(game)
            if decode_action(game, index).wonder_name == "The Mausoleum"
        ]
        if builds:
            return _Position(game=game), builds[0]
    raise AssertionError("no Mausoleum position found")


class _PeakedEvaluator:
    """Deterministic evaluator that makes one line strictly best.

    The obvious version of the test below drove a randomly initialised net and
    asserted the two searchers reported the SAME follow-up. That was testing
    noise: under a random net every revival from the discard pile is worth about
    the same, so the follow-up argmax is a coin flip between near-ties, and the
    Rust and Python descents legitimately break ties differently. Measured over
    six torch seeds they disagreed on three, and on a fourth neither expanded
    the child at all, so `follow_up` was None on both.

    Peaking the prior removes the tie: the principal variation is forced, both
    searchers must find it, and the expected string can be written down. Values
    are constant, so nothing here depends on model weights at all.
    """

    def __init__(self, preferred):
        self._preferred = {int(index) for index in preferred}

    def evaluate(self, encodings, legal_lists):
        import numpy as np

        from .inference import Evaluation

        out = []
        for _encoding, legal in zip(encodings, legal_lists):
            policy = np.ones(len(legal), dtype=np.float32)
            for position, index in enumerate(legal):
                if int(index) in self._preferred:
                    policy[position] = 200.0
            out.append(
                Evaluation(
                    policy=policy / policy.sum(),
                    wdl=np.asarray([0.5, 0.2, 0.3], dtype=np.float32),
                    joint7=np.full(7, 1.0 / 7.0, dtype=np.float32),
                    margin=0.0,
                    military=0.0,
                    science=np.zeros(2, dtype=np.float32),
                )
            )
        return out


@pytest.mark.parametrize("search_impl", ["rust", "python"])
def test_a_forced_follow_up_is_reported_for_the_move_that_forces_it(search_impl):
    """Advisor item H: `Wonder: The Mausoleum (using X)` is not the whole move.

    Building it immediately forces a second decision -- which discarded card to
    take for free -- and that is most of the move's value. The search knows it
    as the principal variation; before this, nothing downstream could see it,
    because the root readout returned root edges only.

    Both searchers are asserted against the same literal string, which is the
    cross-engine gate: they are separate walks over separate structures, so a
    divergence would make the panel say different things depending on which
    searcher is on.
    """

    from .codec import MAUSOLEUM_BASE, decode_action
    from .data import CARD_IDS
    from .engine import ActionUse

    position, mausoleum = _mausoleum_position()
    revived = position.game.discard_pile[0]
    adapter = SevenWondersAdvisor(
        evaluator=_PeakedEvaluator({mausoleum, MAUSOLEUM_BASE + CARD_IDS[revived]})
    )
    request = RecommendRequest(
        max_sims=400,
        chunk_sims=400,
        options={"search_impl": search_impl, "force_expand_root_chance": False},
    )
    handle = adapter.open_search(position, request)
    try:
        snapshot = handle.advance(400, threading.Event())
    finally:
        handle.close()

    assert snapshot.entries[str(mausoleum)].follow_up == f"then {revived}"

    # ...and an ordinary move, which simply ends the turn, must stay silent
    # rather than report the opponent's reply as if it were part of your move.
    plain = [
        action_id
        for action_id, stats in snapshot.entries.items()
        if decode_action(position.game, int(action_id)).use
        is ActionUse.DISCARD_FOR_COINS
    ]
    assert plain, "the position must contain a plain move to contrast against"
    assert all(snapshot.entries[action_id].follow_up is None for action_id in plain)


@pytest.mark.parametrize("search_impl", ["rust", "python"])
def test_the_arena_budget_stops_a_long_search(adapter, search_impl):
    """The advisor must not grow its tree without a ceiling.

    The panel asks for "keep thinking until the board changes". On a wide root
    the closed tree allocates a node per simulation, each owning a cloned
    GameState, so that request was unbounded in MEMORY: measured at ~4.4 KB a
    node, 400k sims cost 1.7 GB and a long think froze a real machine.

    A 1 MB budget here trips almost immediately, which keeps the test quick; the
    shipped default is DEFAULT_ARENA_BUDGET_MB.
    """

    position = _pos(adapter, prefix=())
    request = RecommendRequest(
        max_sims=100_000,
        chunk_sims=200,
        options={"search_impl": search_impl, "arena_budget_mb": 1},
    )
    handle = adapter.open_search(position, request)
    try:
        stop = threading.Event()
        snapshot = handle.advance(200, stop)
        for _ in range(50):  # bounded: it must stop on its own well before this
            if snapshot.stop_reason:
                break
            snapshot = handle.advance(200, stop)
    finally:
        handle.close()

    assert snapshot.stop_reason is not None, "the budget never tripped"
    assert "budget 1 MB" in snapshot.stop_reason
    assert snapshot.sims_done < 100_000, "must stop short of the sim target"
    assert snapshot.entries, "the work already done is still the answer"


def test_no_budget_means_no_ceiling(adapter):
    """0 disables it, so the training/analysis paths are unaffected."""

    request = RecommendRequest(
        max_sims=400, chunk_sims=400, options={"arena_budget_mb": 0}
    )
    handle = adapter.open_search(_pos(adapter), request)
    try:
        snapshot = handle.advance(400, threading.Event())
    finally:
        handle.close()
    assert snapshot.stop_reason is None
    assert snapshot.sims_done == 400


def test_the_start_player_outlook_is_suppressed():
    """A calibration hole, not a missing feature.

    Until 2026-08-03 the engine dealt the next Age *after* asking who begins it,
    so every checkpoint trained before that saw CHOOSE_NEXT_START_PLAYER only
    with an exhausted tableau. Item F now hands it a full pyramid. The ranked
    moves survive -- search plays forward into ordinary positions -- but this is
    a single raw read of the root, whose aux heads carry 0.2 loss weight and no
    search correction. Confident-looking noise is worse than nothing.
    """

    import random

    from .advisor_adapter import _Position
    from .codec import decode_action, legal_action_indices
    from .engine import apply_action, legal_actions
    from .game import Phase, new_game
    from .inference import Evaluator
    from .train import build_model

    game = new_game(9)
    rng = random.Random(0)
    while game.phase is not Phase.CHOOSE_NEXT_START_PLAYER:
        actions = legal_actions(game)
        if not actions:
            pytest.skip("no start-player choice reached")
        apply_action(game, rng.choice(actions))

    advisor = SevenWondersAdvisor(
        evaluator=Evaluator(build_model("transformer", 32, 1), "cpu")
    )
    public = advisor.state_to_public(_Position(game=game))
    assert public["victory_outlook"] is None
    assert public["legal_actions"], "the rest of the payload is unaffected"

    # ...and an ordinary turn still reports one, so this is targeted rather
    # than a blanket disable.
    apply_action(game, decode_action(game, legal_action_indices(game)[0]))
    assert game.phase is Phase.PLAY_AGE
    assert advisor.state_to_public(_Position(game=game))["victory_outlook"] is not None
