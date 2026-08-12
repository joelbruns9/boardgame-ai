from __future__ import annotations

import random

import pytest

from games.kingdomino_classic import (
    ClassicGameConfig,
    ClassicGameState,
    Phase,
    PickAction,
    TurnAction,
)
from games.kingdomino_classic.board import ClassicBoard, Placement
from games.kingdomino_classic.dominoes import DOMINOES


def _pick_lowest(state: ClassicGameState) -> ClassicGameState:
    action = min(
        (action for action in state.legal_actions() if isinstance(action, PickAction)),
        key=lambda action: action.domino_id,
    )
    return state.step(action)


def test_initial_three_player_draft_discards_the_unclaimed_fourth_domino() -> None:
    state = ClassicGameState.new(
        seed=7, config=ClassicGameConfig(players=3), start_player=1
    )
    initial_row = tuple(state.draft_row)

    assert state.current_actor == 1
    state = _pick_lowest(state)
    assert state.current_actor == 2
    state = _pick_lowest(state)
    assert state.current_actor == 0
    state = _pick_lowest(state)

    assert state.phase == Phase.PLACE_AND_DRAFT
    assert state.unclaimed_discards == [max(initial_row)]
    assert len(state.pending_claims) == 3
    assert [claim.domino_id for claim in state.pending_claims] == sorted(
        initial_row[:3]
    )
    assert len(state.draft_row) == 4


def test_initial_four_player_draft_has_a_forced_final_selection() -> None:
    state = ClassicGameState.new(
        seed=11, config=ClassicGameConfig(players=4), start_player=2
    )

    for expected_actor in (2, 3, 0):
        assert state.current_actor == expected_actor
        state = _pick_lowest(state)

    actions = state.legal_actions()
    assert state.current_actor == 1
    assert len(actions) == 1
    assert isinstance(actions[0], PickAction)
    state = state.step(actions[0])

    assert state.phase == Phase.PLACE_AND_DRAFT
    assert state.unclaimed_discards == []
    assert len(state.pending_claims) == 4


@pytest.mark.parametrize("players", [3, 4])
def test_pending_claim_order_controls_the_next_actor(players: int) -> None:
    state = ClassicGameState.new(
        seed=17, config=ClassicGameConfig(players=players), start_player=0
    )
    picked_by_player: dict[int, int] = {}
    while state.phase == Phase.INITIAL_DRAFT:
        actor = state.current_actor
        action = state.legal_actions()[-1]
        assert isinstance(action, PickAction)
        picked_by_player[actor] = action.domino_id
        state = state.step(action)

    expected_actor = min(picked_by_player, key=picked_by_player.__getitem__)
    assert state.current_actor == expected_actor


def test_place_and_draft_action_advances_both_rows() -> None:
    state = ClassicGameState.new(
        seed=23, config=ClassicGameConfig(players=3), start_player=0
    )
    while state.phase == Phase.INITIAL_DRAFT:
        state = _pick_lowest(state)

    actor = state.current_actor
    current_domino = state.pending_claims[0].domino_id
    future_domino = state.draft_row[0]
    action = next(
        action
        for action in state.legal_actions()
        if isinstance(action, TurnAction)
        and action.pick_domino_id == future_domino
    )
    state = state.step(action)

    assert current_domino in state.boards[actor].placed_domino_ids
    assert future_domino not in state.draft_row
    assert any(
        claim == (actor, future_domino)
        for claim in (
            (claim.player, claim.domino_id) for claim in state.next_claims
        )
    )


@pytest.mark.parametrize("players", [3, 4])
def test_random_complete_games_conserve_inventory_and_finish(players: int) -> None:
    for seed in range(12):
        rng = random.Random(seed + 10_000 * players)
        state = ClassicGameState.new(
            seed=seed,
            config=ClassicGameConfig(players=players),
            start_player=seed % players,
        )
        while not state.is_terminal:
            actions = state.legal_actions()
            assert actions
            if state.phase in (Phase.PLACE_AND_DRAFT, Phase.FINAL_PLACEMENT):
                claim = state.pending_claims[0]
                placements = state.boards[claim.player].legal_placements(
                    DOMINOES[claim.domino_id]
                )
                assert all(
                    isinstance(action, TurnAction)
                    and (action.placement is None) == (not placements)
                    for action in actions
                )
            state = state.step(rng.choice(actions))

        state.assert_invariants()
        assert len(state.history) == 13 * players
        assert len(state.unclaimed_discards) == (12 if players == 3 else 0)
        assert all(
            board.placed_domino_count + len(state.forced_discards[player]) == 12
            for player, board in enumerate(state.boards)
        )
        result = state.result()
        assert len(result.scores) == players
        assert len(result.ranks) == players
        assert sum(result.win_shares) == pytest.approx(1.0)
        for player, breakdown in enumerate(state.score_breakdowns()):
            expected_harmony = 5 if not state.forced_discards[player] else 0
            assert breakdown.harmony_bonus == expected_harmony


def test_nonterminal_result_is_rejected() -> None:
    state = ClassicGameState.new(config=ClassicGameConfig(players=3), seed=1)

    with pytest.raises(ValueError, match="only at game over"):
        state.result()


def test_multiplayer_result_uses_largest_territory_tiebreak() -> None:
    larger_territory = ClassicBoard()
    larger_territory.place(DOMINOES[13], Placement(1, 0, 2, 0))
    larger_territory.place(DOMINOES[19], Placement(1, 1, 2, 1))
    more_crowns = ClassicBoard()
    more_crowns.place(DOMINOES[41], Placement(1, 0, 2, 0))

    state = ClassicGameState.new(
        config=ClassicGameConfig(
            players=3, harmony=False, middle_kingdom=False
        ),
        seed=3,
    )
    state.boards = [larger_territory, more_crowns, ClassicBoard()]
    state.phase = Phase.GAME_OVER

    result = state.result()

    assert result.scores == (2, 2, 0)
    assert result.tiebreak_keys[0] == (2, 2, 1)
    assert result.tiebreak_keys[1] == (2, 1, 2)
    assert result.ranks == (1, 2, 3)
    assert result.winners == (0,)
    assert result.win_shares == (1.0, 0.0, 0.0)


def test_true_multiplayer_tie_splits_win_share() -> None:
    state = ClassicGameState.new(
        config=ClassicGameConfig(
            players=3, harmony=False, middle_kingdom=False
        ),
        seed=5,
    )
    state.phase = Phase.GAME_OVER

    result = state.result()

    assert result.ranks == (1, 1, 1)
    assert result.winners == (0, 1, 2)
    assert result.win_shares == pytest.approx((1 / 3, 1 / 3, 1 / 3))
