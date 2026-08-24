"""Fast M3 samples; the release gate compares at least 400,000 encodings."""

from __future__ import annotations

import numpy as np
import pytest

from games.welcome_to import action_codec as codec
from games.welcome_to import encoder as enc
from games.welcome_to import rust_encode_equiv as eq
from games.welcome_to import snapshot
from games.welcome_to.constants import CARD_TABLE, Effect, NUM_BASE_CARDS
from games.welcome_to.game import GameConfig, GameState
from games.welcome_to.rust_encoder import encode_state as rust_encode_state
from games.welcome_to.rust_equiv import GATE_CONFIGS, constructed_cases

wr = pytest.importorskip(
    "welcome_to_rust",
    reason="the Rust encoder is not built; run maturin develop --release",
)


def _card_with(*, number: int | None = None, effect: Effect | None = None) -> int:
    for card, (candidate_number, candidate_effect) in enumerate(
        CARD_TABLE[:NUM_BASE_CARDS]
    ):
        if (number is None or number == candidate_number) and (
            effect is None or effect is candidate_effect
        ):
            return card
    raise AssertionError(f"no card number={number} effect={effect}")


def test_encoder_abi_is_checked_at_import():
    assert wr.ENCODER_ABI_VERSION == enc.ENCODER_ABI_VERSION
    assert wr.SHEET_PLANES_LEN == int(np.prod(enc.SHEET_PLANES_SHAPE))
    assert wr.SHEET_SCALARS_LEN == enc.MAX_SEATS * enc.NUM_SHEET_SCALAR
    assert wr.VIEWER_PLANE_LEN == int(np.prod(enc.VIEWER_PLANE_SHAPE))
    assert wr.GLOBAL_SCALARS_LEN == enc.NUM_GLOBAL_SCALAR


@pytest.mark.parametrize("config", GATE_CONFIGS)
def test_new_games_are_bit_exact_for_every_viewer(config: GameConfig):
    py = GameState.new(seed=7, config=config)
    rs = wr.RustGameState(
        7,
        players=config.players,
        advanced=config.advanced,
        expert=False,
        solo_rules=False,
    )
    assert eq.compare_state(py, rs, where=f"new {config}") == config.players


@pytest.mark.parametrize("driver", ["random", "no-refusal", "greedy"])
def test_a_complete_game_is_bit_exact(driver: str):
    states, encodings, actions = eq.check_game(104, GATE_CONFIGS[-1], driver)
    assert states == actions + 1
    assert encodings == states * GATE_CONFIGS[-1].players


def test_constructed_rare_positions_are_bit_exact():
    for name, py, _expectation in constructed_cases():
        rs = wr.RustGameState.from_snapshot(snapshot.to_snapshot(py))
        eq.compare_state(py, rs, where=name)


def test_a_midturn_live_sheet_is_hidden_from_an_opponent_in_rust():
    py = GameState.new(seed=21, config=GameConfig(players=2, solo_rules=False))
    py.stack_new[0][0] = _card_with(number=6)
    py.stack_old[0][0] = _card_with(effect=Effect.SURVEYOR)
    py.apply(codec.choose_stack(0))
    opponent_axis = enc.seat_order(py, 1).index(0)

    rust_before = wr.RustGameState.from_snapshot(snapshot.to_snapshot(py))
    before = rust_encode_state(rust_before, 1)
    py.apply(codec.write(0, 0, 3))
    rust_after = wr.RustGameState.from_snapshot(snapshot.to_snapshot(py))
    after = rust_encode_state(rust_after, 1)
    assert np.array_equal(before[0][opponent_axis], after[0][opponent_axis])
    assert np.array_equal(before[1][opponent_axis], after[1][opponent_axis])

    # The owner sees the write; the assertion above is not vacuous.
    own = rust_encode_state(rust_after, 0)
    assert own[0][0, enc.P_WRITTEN, 0, 3] == 1.0


def test_invalid_viewer_is_refused():
    rs = wr.RustGameState(3, players=2, advanced=False, solo_rules=False)
    with pytest.raises(RuntimeError, match="outside 2 seats"):
        rs.encode_state(2)
