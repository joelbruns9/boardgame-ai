"""Evaluator-ABI v3 gate: NumPy postprocessing versus Rust postprocessing.

``S2_GENERATION_REVIEW_REQUEST.md`` §4.1 leaves one question open — whether
moving the segmented policy softmax, the rank masking/softmax and the value
blend from Python into Rust is strength-neutral. The batch-width test compares
v3 to itself, and the one-row oracle only exercises the blocking seam, so
neither of them can see a postprocessing bug.

The comparison this module makes is available without resurrecting dead code,
because **both arithmetics still exist in the tree**:

* :meth:`rust_search.PackedNetEvaluator.evaluate_request` — the blocking M5
  seam. Python does the masked softmax in NumPy over the dense 684-vector and
  the value blend in :func:`mcts.blend_value`. This is the pre-v3 arithmetic.
* :meth:`rust_search.PackedNetEvaluator.forward` — the packed v3 seam. Python
  returns sparse legal logits plus raw rank/score heads, and Rust performs the
  segmented softmax in ``f32``, the masked rank softmax in ``f32``, and
  ``blend_value`` in ``f64``.

Driving one real trajectory through both, on the same fixed tape and the same
network, is therefore exactly the old-postprocess-versus-new-postprocess
comparison the request asks for.

⚠ **The contract asserted here is discrete, not bit-identical, and that is not
a weakening.** The two paths are equal in exact arithmetic but round
differently: NumPy exponentiates a dense masked vector and Rust exponentiates
the compact legal segment, and Torch's ``log_softmax`` is not Rust's
exp-and-normalise. So the gate asserts the tree — root actions, visit counts and
the chosen macro at every decision — and *reports* the largest backup-total
drift rather than asserting it away. A prior difference can in principle flip
PUCT's first-max tie-break; if that ever happens this gate is what says so.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from games.welcome_to import mcts
from games.welcome_to import network as nw
from games.welcome_to import rust_search

import welcome_to_rust as wr

from games.welcome_to.portable_rng import derive_search_seed


def _network(seed: int = 0xA17E) -> nw.WelcomeToNet:
    torch.manual_seed(seed)
    return nw.WelcomeToNet(
        nw.NetConfig(
            sheet_hidden=32,
            sheet_out=16,
            trunk_hidden=48,
            trunk_blocks=1,
            head_hidden=32,
        )
    ).eval()


def run_game(
    *,
    game_seed: int,
    players: int,
    advanced: bool,
    simulations: int,
    net: nw.WelcomeToNet | None = None,
) -> dict[str, float]:
    """Play one game twice over, one decision at a time, and compare the trees."""
    net = net if net is not None else _network()
    config = mcts.SearchConfig(simulations=simulations)
    device = torch.device("cpu")

    # One evaluator object per arm: the two arms must not share request-id
    # counters, and each arm keeps its own profile.
    numpy_eval = rust_search.PackedNetEvaluator(net, device, config)
    rust_eval = rust_search.PackedNetEvaluator(net, device, config)

    state = wr.RustGameState(
        game_seed,
        players=players,
        advanced=advanced,
        expert=False,
        solo_rules=False,
    )
    blocking = [rust_search.native_search(config) for _ in range(players)]
    scheduler = rust_search.native_cloud_scheduler(
        config, capacity=players, workers=1
    )

    decisions = compared = 0
    worst_total = 0.0
    while not state.is_terminal:
        seat = int(state.actor)
        seed = derive_search_seed(game_seed, decisions)
        left = blocking[seat].play(state, numpy_eval, seed, seat)
        right = scheduler.play(
            [state], rust_eval, [seed], roots=[seat], slots=[seat], max_batch=1
        )[0]

        assert left["choice"] == right["choice"], (
            f"seed {game_seed} decision {decisions}: NumPy postprocess chose "
            f"{left['choice']}, Rust postprocess chose {right['choice']}"
        )
        assert list(left["actions"]) == list(right["actions"]), (
            f"seed {game_seed} decision {decisions}: root action lists differ"
        )
        assert list(left["visits"]) == list(right["visits"]), (
            f"seed {game_seed} decision {decisions}: visit counts differ — a "
            "postprocessing difference changed the tree, not just its floats"
        )
        if sum(left["visits"]) > 0:
            compared += 1
            worst_total = max(
                worst_total,
                float(
                    np.max(
                        np.abs(
                            np.asarray(left["total"], dtype=np.float64)
                            - np.asarray(right["total"], dtype=np.float64)
                        )
                    )
                ),
            )

        state.apply_macro(int(left["choice"]))
        decisions += 1
        if decisions > 10_000:
            raise RuntimeError("postprocess parity trajectory did not terminate")

    return {
        "decisions": float(decisions),
        "searched": float(compared),
        "worst_total_drift": worst_total,
    }


def run_gate(
    *,
    games: int = 2,
    players: int = 2,
    advanced: bool = True,
    simulations: int = 16,
    seed0: int = 91_000,
) -> dict[str, float]:
    net = _network()
    decisions = searched = 0.0
    worst = 0.0
    for game in range(games):
        result = run_game(
            game_seed=seed0 + game,
            players=players,
            advanced=advanced,
            simulations=simulations,
            net=net,
        )
        decisions += result["decisions"]
        searched += result["searched"]
        worst = max(worst, result["worst_total_drift"])
    return {
        "games": float(games),
        "decisions": decisions,
        "searched": searched,
        "worst_total_drift": worst,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--players", type=int, default=2, choices=(2, 3, 4))
    parser.add_argument("--base-game", action="store_true")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--seed0", type=int, default=91_000)
    args = parser.parse_args()
    started = time.perf_counter()
    result = run_gate(
        games=args.games,
        players=args.players,
        advanced=not args.base_game,
        simulations=args.simulations,
        seed0=args.seed0,
    )
    print(
        "ABI v3 postprocess gate green: "
        f"{int(result['games'])} trajectories, {int(result['decisions'])} decisions "
        f"({int(result['searched'])} searched roots) — identical trees; "
        f"largest backup-total drift {result['worst_total_drift']:.3e}; "
        f"{time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
