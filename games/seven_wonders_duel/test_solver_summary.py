"""The solver's line in the iteration log.

Written after a smoke run in which the solver never fired -- generation was at
one simulation per move -- and nothing in the log distinguished that from a
solver that fired and proved nothing.
"""

from .buffer import MoveRecord
from .phase_d import _summarize_solver


def _move(**kwargs) -> MoveRecord:
    base = dict(i=0, actor=0, action=0, mask_hash="")
    base.update(kwargs)
    return MoveRecord(**base)


def test_silence_is_reported_as_zero_not_omitted():
    summary = _summarize_solver([_move(), _move()])
    assert summary == {"attempted": 0}


def test_attempted_and_masked_are_counted_separately():
    moves = [
        _move(solver_attempted=True, solver_masked=True, solver_regime="exact",
              solver_stop="proved", solver_nodes=1000),
        _move(solver_attempted=True, solver_regime="declined",
              solver_stop="node_cap", solver_nodes=4_500_000),
        _move(),
    ]
    summary = _summarize_solver(moves)
    assert summary["attempted"] == 2
    assert summary["masked"] == 1
    assert summary["masked_fraction"] == 0.5
    assert summary["attempted_fraction"] == 2 / 3
    assert summary["regimes"] == {"declined": 1, "exact": 1}
    assert summary["stops"] == {"node_cap": 1, "proved": 1}
    assert summary["nodes_total"] == 4_501_000
    assert summary["nodes_max"] == 4_500_000
