"""The certifier's safety properties, on positions built without a BGA log.

`control_certify` shipped with no unit tests at all, which the 2026-09-06 review
called out: its whole value is that PROVEN can be trusted, and nothing pinned
that but a probe run by hand against one table. These are the properties that
must hold whatever the position -- an incomplete search must weaken the claim,
never invent one -- plus the two defects the review found.
"""

from __future__ import annotations

import random

import pytest

from .buffer import GameRecorder
from .codec import legal_action_indices
from .control_certify import ABORTED, Verdict, certify
from . import control_certify
from .game import Phase


def _age_three_game(seed=11, rng_seed=1101):
    """Random play into Age III, where reveals make chance edges wide."""

    recorder = GameRecorder(seed, agents={"p0": "t", "p1": "t"})
    rng = random.Random(rng_seed)
    while recorder.game.phase.value != "complete":
        if recorder.game.age == 3 and recorder.game.phase is Phase.PLAY_AGE:
            return recorder.game
        choice = rng.choice(legal_action_indices(recorder.game))
        recorder.play(choice, policy_target={choice: 1.0})
    raise AssertionError("never reached Age III")


def _count_clones(monkeypatch):
    """Count every state the proof builds."""

    built = []
    original = control_certify.fast_clone

    def counting(state):
        built.append(1)
        return original(state)

    monkeypatch.setattr(control_certify, "fast_clone", counting)
    return built


# -- soundness --------------------------------------------------------------


@pytest.mark.parametrize("max_nodes", [1, 2, 5, 50])
def test_a_budget_limited_search_is_never_a_verdict(max_nodes):
    """The property everything else rests on: an incomplete search may only
    return UNKNOWN. An OR node that ran out of budget has NOT established that
    nothing forces the win, and reporting REFUTED there would turn a limit into
    a false negative that the caller cannot tell from a real refutation."""

    game = _age_three_game()
    certificate = certify(
        game, loser=game.active_player, max_nodes=max_nodes, max_plies=14,
        max_secs=300, require_threat=False,
    )
    assert certificate.verdict is Verdict.UNKNOWN
    assert certificate.stopped_by is not None
    assert "nodes" in certificate.limits_hit


def test_shrinking_the_budget_never_turns_unknown_into_proven():
    """UNKNOWN can never be upgraded by an unexamined branch, so a smaller
    budget can only ever weaken the answer."""

    game = _age_three_game()
    verdicts = [
        certify(game, loser=game.active_player, max_nodes=cap, max_plies=14,
                max_secs=300, require_threat=False).verdict
        for cap in (1, 20, 400)
    ]
    assert all(v is Verdict.UNKNOWN for v in verdicts)


# -- the review's findings --------------------------------------------------


def test_an_exhausted_budget_stops_expanding(monkeypatch):
    """Child generation used to continue after the budget was gone.

    At the recorded row 85, `max_nodes=1` still built 270 children: the loop
    called `_children` for every remaining action, and each of those enumerated
    a 90-way Age III reveal, only to hand every child straight back as UNKNOWN.
    Expansion now stops at the limit, and a wide chance edge aborts part-way
    through rather than finishing.
    """

    game = _age_three_game()
    built = _count_clones(monkeypatch)
    certify(game, loser=game.active_player, max_nodes=1, max_plies=14,
            max_secs=300, require_threat=False)
    one_node = len(built)

    built.clear()
    certify(game, loser=game.active_player, max_nodes=200, max_plies=14,
            max_secs=300, require_threat=False)
    many_nodes = len(built)

    # A one-node budget expands one action, not every action.
    assert one_node < many_nodes
    # And the clones it does build are the children of a single edge, plus the
    # root -- not a whole ply of them.
    assert one_node <= many_nodes / 2


def test_children_reports_a_budget_abort_distinctly_from_a_dealt_edge():
    """`None` means "this edge cannot be enumerated at all" and drives an
    `age_deal` note; a budget abort is a different UNKNOWN and must not be
    recorded as one, or `limits_hit` misreports why the proof stopped."""

    assert ABORTED is not None
    game = _age_three_game()
    certificate = certify(
        game, loser=game.active_player, max_nodes=1, max_plies=14, max_secs=300,
        require_threat=False,
    )
    assert "age_deal" not in certificate.limits_hit


def test_victory_type_is_absent_unless_the_proof_supplies_one():
    """It used to be inferred from the ROOT threat gate, which reports which
    sudden deaths are in REACH -- so a seat one symbol short of six that is
    actually lost to an immediate military win was certified 'scientific'.
    Nothing is proven here, so nothing may be named."""

    game = _age_three_game()
    certificate = certify(
        game, loser=game.active_player, max_nodes=50, max_plies=14, max_secs=300,
        require_threat=False,
    )
    assert certificate.verdict is not Verdict.PROVEN
    assert certificate.victory_type is None
