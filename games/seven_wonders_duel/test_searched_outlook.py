"""Search's own seven-way outlook, per root move, through the advisor.

What these protect:

* every root move gets its own backed-up outlook, and the per-move counts add up
  to the root's (less the root's own expansion, which belongs to no move);
* carrying the vector cannot change what search chooses -- same seed, same
  visits and values with and without it;
* a move shows an outlook only when nearly all of its simulations carried one,
  so a terminals-only mean is never presented as a forecast;
* the advisor backs up W4 only for a checkpoint whose flat joint7 was replaced,
  and the flat head otherwise -- which is what makes it work on today's net;
* the host carries the breakdown to the response untouched.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
swr = pytest.importorskip("seven_wonders_rust")

from games.advisor.ranking import response_from_snapshot

from .advisor_adapter import (
    OUTLOOK_MIN_COVERAGE,
    SevenWondersAdvisor,
    _searched_outlook,
    outlook_source_for,
)
from .inference import Evaluator
from .net import SWDNet


def _evaluator(**kwargs):
    torch.manual_seed(3)
    return Evaluator(SWDNet(d_model=32, layers=1, heads=4, **kwargs), "cpu", 64, fuse_embedder=False)


def _state(adapter):
    from .test_hierarchical_value import _draft_prefix

    return adapter.state_from_wire({"seed": 7, "first_player": 0, "prefix": _draft_prefix(7)})


def _search(adapter, state, sims=200, **options):
    request = SimpleNamespace(
        max_sims=sims, seed=4, options=dict(options), checkpoint_path=None, engine="nn", device=None
    )
    handle = adapter.open_search(state, request)
    snapshot = handle.advance(sims, threading.Event())
    return handle, snapshot


def test_every_searched_move_carries_its_own_outlook():
    adapter = SevenWondersAdvisor(evaluator=_evaluator())
    handle, snapshot = _search(adapter, _state(adapter))
    assert snapshot.root_outlook is not None
    assert sum(snapshot.root_outlook.values()) == pytest.approx(1.0, abs=1e-6)
    searched = [stats for stats in snapshot.entries.values() if stats.visits > 0]
    assert searched
    for stats in searched:
        assert stats.outlook is not None
        assert sum(stats.outlook.values()) == pytest.approx(1.0, abs=1e-6)


def test_per_move_counts_add_up_to_the_root():
    adapter = SevenWondersAdvisor(evaluator=_evaluator())
    handle, snapshot = _search(adapter, _state(adapter))
    root, root_count, edges = handle._search.outlooks()
    visits = {int(a): v for a, v, _s, _p in handle._search.snapshot()[4]}
    # The root's own expansion is counted at the root and belongs to no move.
    assert sum(count for _a, _o, count in edges) == root_count - 1
    for action, outlook, count in edges:
        assert count == visits[action]


def test_carrying_the_outlook_changes_nothing_search_decides(monkeypatch):
    import sys

    # The module the advisor class was actually loaded from: under pytest the
    # package can be importable by two names, and patching the other one is a
    # silent no-op.
    module = sys.modules[SevenWondersAdvisor.__module__]
    adapter = SevenWondersAdvisor(evaluator=_evaluator())
    state = _state(adapter)
    _handle, with_outlook = _search(adapter, state)
    monkeypatch.setattr(module, "outlook_source_for", lambda evaluator: None)
    _handle, without = _search(adapter, state)
    assert without.root_outlook is None
    assert with_outlook.root_value == without.root_value
    for action, stats in with_outlook.entries.items():
        other = without.entries[action]
        assert (stats.visits, stats.q_value) == (other.visits, other.q_value)


def test_a_thinly_covered_move_shows_nothing():
    values = [1 / 7] * 7
    assert _searched_outlook(values, 10, 10) is not None
    assert _searched_outlook(values, int(OUTLOOK_MIN_COVERAGE * 10) - 1, 10) is None
    assert _searched_outlook(None, 10, 10) is None
    assert _searched_outlook(values, 0, 0) is None


def test_the_scalar_bridge_shows_no_searched_outlook():
    adapter = SevenWondersAdvisor(evaluator=_evaluator())
    _handle, snapshot = _search(adapter, _state(adapter), sims=60, leaf_batch=1)
    assert all(stats.outlook is None for stats in snapshot.entries.values())


def test_w4_is_backed_up_only_when_it_replaced_the_flat_head():
    plain = _evaluator()
    assert outlook_source_for(plain) == "flat"
    shadow = _evaluator(hierarchical_value=True)
    assert outlook_source_for(shadow) == "flat"
    replaced = _evaluator(hierarchical_value=True, hierarchical_value_detach=False)
    replaced.model.joint7_replaced = True
    assert outlook_source_for(replaced) == "hierarchical"
    adapter = SevenWondersAdvisor(evaluator=replaced)
    _handle, snapshot = _search(adapter, _state(adapter), sims=60)
    assert snapshot.root_outlook is not None


def test_the_host_carries_the_breakdown_to_the_response():
    adapter = SevenWondersAdvisor(evaluator=_evaluator())
    state = _state(adapter)
    _handle, snapshot = _search(adapter, state, sims=80)
    response = response_from_snapshot(
        snapshot, adapter.action_views(state), engine="nn", top_k=5, search_ms=1
    )
    assert response.root_outlook == snapshot.root_outlook
    for recommendation in response.recommendations:
        assert recommendation.outlook == snapshot.entries[recommendation.action_id].outlook
