"""M5 Gate 2: exact real-network trajectories at identical batch width one."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from games.welcome_to import encoder as enc
from games.welcome_to import macro_codec as mc
from games.welcome_to import mcts
from games.welcome_to import network as nw
from games.welcome_to import rust_search
from games.welcome_to import snapshot
from games.welcome_to.game import GameConfig, GameState
from games.welcome_to.portable_rng import PortableRng, derive_search_seed

import welcome_to_rust as wr


class RecordingNetEvaluator(mcts.NetEvaluator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requests: list[tuple] = []

    def _record(self, kind: int, state: GameState, viewer: int) -> None:
        if kind == 1:
            assert viewer == state.actor
        arrays = enc.encode_state(state, viewer)
        self.requests.append(
            (
                kind,
                viewer,
                state.config.players,
                len(self.requests),
                tuple(mc.legal_macros(state)),
                tuple(array.astype("<f4", copy=False).tobytes() for array in arrays),
            )
        )

    def evaluate(self, state, viewer):
        self._record(0, state, viewer)
        return super().evaluate(state, viewer)

    def policy(self, state, viewer):
        self._record(1, state, viewer)
        return super().policy(state, viewer)


class RecordingMcts(mcts.MCTS):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last = None

    def search(self, *args, **kwargs):
        result = super().search(*args, **kwargs)
        self.last = (result[0].copy(), result[1].copy(), result[2].total.copy())
        return result

    def play(self, *args, **kwargs):
        self.last = None
        return super().play(*args, **kwargs)


def _request_tuple(request) -> tuple:
    return (
        request.kind,
        request.viewer,
        request.seats,
        request.request_id,
        request.legal,
        request.encoding,
    )


def run_game(
    *, game_seed: int, players: int, advanced: bool, simulations: int
) -> dict[str, float]:
    torch.manual_seed(0xA17E)
    net = nw.WelcomeToNet(
        nw.NetConfig(
            sheet_hidden=32,
            sheet_out=16,
            trunk_hidden=48,
            trunk_blocks=1,
            head_hidden=32,
        )
    ).eval()
    config = mcts.SearchConfig(simulations=simulations)
    py_state = GameState.new(
        seed=game_seed,
        config=GameConfig(players=players, advanced=advanced),
        rng_kind="portable",
    )
    rust_state = wr.RustGameState.from_snapshot(snapshot.to_snapshot(py_state))

    py_evaluators = [
        RecordingNetEvaluator(net, torch.device("cpu"), config) for _ in range(players)
    ]
    rust_evaluators = [
        rust_search.PackedNetEvaluator(
            net, torch.device("cpu"), config, record_requests=True
        )
        for _ in range(players)
    ]
    py_searches = [RecordingMcts(py_evaluators[s], config) for s in range(players)]
    rust_searches = [rust_search.native_search(config) for _ in range(players)]

    actions: list[int] = []
    searches = non_forced = 0
    while not py_state.is_terminal:
        seat = py_state.actor
        assert rust_state.actor == seat
        search_seed = derive_search_seed(game_seed, searches)
        py_choice = py_searches[seat].play(
            py_state, seat, PortableRng(search_seed)
        )
        native = rust_searches[seat].play(
            rust_state, rust_evaluators[seat], search_seed, seat
        )
        rust_choice = native["choice"]
        assert rust_choice == py_choice, (
            f"action diverged at decision {searches}: python={py_choice}, "
            f"rust={rust_choice}"
        )
        if py_searches[seat].last is not None:
            non_forced += 1
            py_actions, py_visits, py_total = py_searches[seat].last
            assert native["actions"] == py_actions.tolist()
            assert np.array_equal(native["visits"], py_visits)
            assert np.array_equal(native["total"], py_total)

        mc.apply_macro(py_state, py_choice)
        rust_state.apply_macro(rust_choice)
        differences = snapshot.diff(snapshot.to_snapshot(py_state), rust_state.snapshot())
        assert not differences, f"state diverged after {searches}: {differences[:5]}"
        actions.append(py_choice)
        searches += 1
        if searches > 10_000:
            raise RuntimeError("real-network trajectory did not terminate")

    for seat in range(players):
        python_requests = py_evaluators[seat].requests
        rust_requests = [
            _request_tuple(request) for request in rust_evaluators[seat].requests
        ]
        assert rust_requests == python_requests, f"request tape diverged for seat {seat}"
        assert rust_searches[seat].particle_slots_allocated == 0
    return {
        "decisions": float(searches),
        "searched": float(non_forced),
        "score_sum": float(sum(py_state.scores())),
        "fingerprint": float(hash(tuple(actions))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--players", type=int, default=2, choices=(2, 3, 4))
    parser.add_argument("--advanced", action="store_true")
    parser.add_argument("--simulations", type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    decisions = searched = 0.0
    for game in range(args.games):
        result = run_game(
            game_seed=90_000 + game,
            players=args.players,
            advanced=args.advanced,
            simulations=args.simulations,
        )
        decisions += result["decisions"]
        searched += result["searched"]
    print(
        "M5 real-network gate green: "
        f"{args.games} exact trajectories, {int(decisions)} decisions "
        f"({int(searched)} searched), batch width 1, "
        f"{time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
