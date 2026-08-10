"""Paired serial search A/B matches for Kingdomino chance-search treatments.

This harness isolates search configuration while keeping the checkpoint fixed.
It deliberately uses the existing serial advisor search because the one-reveal
topology is not yet available in ``BatchedMCTS``.  Every deck seed is played
twice with seats swapped, and match statistics use the same paired-cluster LCB
as checkpoint promotion.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from games.kingdomino.action_codec import decode_action
from games.kingdomino.denial_search import load_checkpoint_network
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.promotion import (
    DEFAULT_CURRENT_BEST,
    MatchStats,
    match_stats_from_pair,
    sha256_file,
    write_json,
)
from games.kingdomino.round_robin_eval import (
    BotProtocol,
    GameResult,
    PairResult,
    play_game,
    update_pair,
)
from games.kingdomino.self_play import make_rust_evaluator


_BACKUP_MODES = frozenset({"sampled", "hajek"})
_TRAVERSAL_MODES = frozenset({"iid", "balanced"})


@dataclass(frozen=True)
class SearchSpec:
    """Immutable serial advisor-search configuration for one match player."""

    sims: int
    chance_exposure: int = 0
    chance_enum_max_rows: int = 12
    chance_backup: str = "hajek"
    chance_traversal: str = "balanced"
    leaf_batch: int = 8
    virtual_loss: int = 1
    cpuct: float = 1.5
    fpu: float = -0.2
    score_scale: float = 160.0
    margin_gain: float = 2.0
    alpha: float = 0.8

    def __post_init__(self) -> None:
        if self.sims <= 0:
            raise ValueError("sims must be > 0")
        if self.chance_exposure < 0:
            raise ValueError("chance_exposure must be >= 0")
        if self.chance_exposure > 0 and self.chance_enum_max_rows <= 0:
            raise ValueError(
                "chance_enum_max_rows must be > 0 when chance exposure is enabled"
            )
        if self.chance_backup not in _BACKUP_MODES:
            raise ValueError(
                f"chance_backup must be one of {sorted(_BACKUP_MODES)}, "
                f"got {self.chance_backup!r}"
            )
        if self.chance_traversal not in _TRAVERSAL_MODES:
            raise ValueError(
                f"chance_traversal must be one of {sorted(_TRAVERSAL_MODES)}, "
                f"got {self.chance_traversal!r}"
            )
        if self.leaf_batch <= 0:
            raise ValueError("leaf_batch must be > 0")
        if self.virtual_loss <= 0:
            raise ValueError("virtual_loss must be > 0")

    @property
    def topology(self) -> str:
        return "open_loop" if self.chance_exposure == 0 else "one_reveal"


@dataclass
class SearchCounters:
    search_calls: int = 0
    nn_evaluations: int = 0
    elapsed_seconds: float = 0.0


SearchFn = Callable[..., tuple[list[tuple[int, int, float, float]], float]]
RustStateFactory = Callable[[Any], Any]
ActionDecoder = Callable[[int, Any], Any]


def _advisor_search(*args, **kwargs):
    import kingdomino_rust as kr

    return kr.advisor_open_loop_search(*args, **kwargs)


class AdvisorSearchBot(BotProtocol):
    """Greedy evaluation bot backed by the Rust serial advisor search."""

    def __init__(
        self,
        evaluator: Callable[..., Any],
        spec: SearchSpec,
        *,
        search_fn: SearchFn = _advisor_search,
        rust_state_factory: RustStateFactory = _rust_state_from_python,
        action_decoder: ActionDecoder = decode_action,
    ) -> None:
        self.spec = spec
        self.counters = SearchCounters()
        self._search_fn = search_fn
        self._rust_state_factory = rust_state_factory
        self._action_decoder = action_decoder

        def counted_evaluator(my_board, opp_board, flat, legal_indices):
            self.counters.nn_evaluations += int(my_board.shape[0])
            return evaluator(my_board, opp_board, flat, legal_indices)

        self._evaluator = counted_evaluator

    def choose_action(
        self,
        state,
        actions: Optional[Sequence[Any]] = None,
        rng: Optional[random.Random] = None,
    ):
        legal = list(actions if actions is not None else state.legal_actions())
        if len(legal) == 1:
            return legal[0]
        if not legal:
            raise ValueError(
                "AdvisorSearchBot received a non-terminal state with no actions"
            )

        rust_state = self._rust_state_factory(state)
        if rust_state is None:
            raise ValueError("Failed to convert Python state to RustGameState")

        py_rng = rng or random.Random()
        search_seed = py_rng.randrange(0, 2**64)
        started = time.perf_counter()
        children, _root_value0 = self._search_fn(
            rust_state,
            self._evaluator,
            int(self.spec.sims),
            dirichlet_alpha=0.3,
            dirichlet_eps=0.0,
            fpu=float(self.spec.fpu),
            cpuct=float(self.spec.cpuct),
            seed=int(search_seed),
            leaf_batch=int(self.spec.leaf_batch),
            virtual_loss=int(self.spec.virtual_loss),
            score_scale=float(self.spec.score_scale),
            margin_gain=float(self.spec.margin_gain),
            alpha=float(self.spec.alpha),
            pick_floor_frac=0.0,
            chance_exposure=int(self.spec.chance_exposure),
            chance_enum_max_rows=int(self.spec.chance_enum_max_rows),
            chance_backup=str(self.spec.chance_backup),
            chance_traversal=str(self.spec.chance_traversal),
        )
        self.counters.elapsed_seconds += time.perf_counter() - started
        self.counters.search_calls += 1
        if not children:
            raise ValueError("Advisor search returned no root children")

        # Match the frozen-position probe: visits first, then prior, then the
        # lowest stable joint index. This is deterministic and adds no move noise.
        top = max(
            children,
            key=lambda row: (int(row[1]), float(row[3]), -int(row[0])),
        )
        return self._action_decoder(int(top[0]), state)


@dataclass
class SearchMatchResult:
    stats: MatchStats
    pair: PairResult
    games: list[GameResult]
    treatment_counters: SearchCounters
    control_counters: SearchCounters

    def to_dict(self) -> dict[str, Any]:
        return {
            "stats": asdict(self.stats),
            "pair": asdict(self.pair),
            "games": [asdict(game) for game in self.games],
            "treatment_counters": asdict(self.treatment_counters),
            "control_counters": asdict(self.control_counters),
        }


def run_paired_search_match(
    treatment_bot: BotProtocol,
    control_bot: BotProtocol,
    *,
    paired_seeds: int,
    seed_start: int,
    treatment_name: str = "treatment",
    control_name: str = "control",
    z: float = 1.96,
    verbose: bool = False,
    play_game_fn: Callable[..., GameResult] = play_game,
) -> tuple[MatchStats, PairResult, list[GameResult]]:
    """Play each deck once in each orientation and score by paired seed."""

    if paired_seeds <= 0:
        raise ValueError("paired_seeds must be > 0")
    if treatment_name == control_name:
        raise ValueError("treatment and control names must differ")

    pair = PairResult(a=treatment_name, b=control_name)
    games: list[GameResult] = []
    started = time.perf_counter()
    for offset in range(int(paired_seeds)):
        seed = int(seed_start) + offset
        oriented = (
            play_game_fn(
                treatment_name,
                treatment_bot,
                control_name,
                control_bot,
                seed=seed,
            ),
            play_game_fn(
                control_name,
                control_bot,
                treatment_name,
                treatment_bot,
                seed=seed,
            ),
        )
        for game in oriented:
            update_pair(pair, game, treatment_name, control_name)
            games.append(game)
        if verbose:
            print(f"paired seed {offset + 1}/{paired_seeds}", flush=True)
    pair.seconds = time.perf_counter() - started
    stats = match_stats_from_pair(
        pair,
        games,
        candidate_name=treatment_name,
        z=float(z),
    )
    return stats, pair, games


def evaluate_serial_search_match(
    net,
    *,
    treatment: SearchSpec,
    control: SearchSpec,
    paired_seeds: int,
    seed_start: int,
    device: str,
    z: float = 1.96,
    verbose: bool = False,
) -> SearchMatchResult:
    """Evaluate two independent search specs using the same frozen network."""

    treatment_evaluator = make_rust_evaluator(
        net,
        device=device,
        margin_gain=treatment.margin_gain,
        alpha=treatment.alpha,
    )
    control_evaluator = make_rust_evaluator(
        net,
        device=device,
        margin_gain=control.margin_gain,
        alpha=control.alpha,
    )
    treatment_bot = AdvisorSearchBot(treatment_evaluator, treatment)
    control_bot = AdvisorSearchBot(control_evaluator, control)
    stats, pair, games = run_paired_search_match(
        treatment_bot,
        control_bot,
        paired_seeds=paired_seeds,
        seed_start=seed_start,
        z=z,
        verbose=verbose,
    )
    return SearchMatchResult(
        stats=stats,
        pair=pair,
        games=games,
        treatment_counters=treatment_bot.counters,
        control_counters=control_bot.counters,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CURRENT_BEST))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--selection-reason",
        required=True,
        help="preregistered reason this search treatment and corpus were selected",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--paired-seeds", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--sims", type=int, default=400,
        help="shared simulation default; per-arm overrides take precedence",
    )
    parser.add_argument("--treatment-sims", type=int)
    parser.add_argument("--control-sims", type=int)
    parser.add_argument("--treatment-exposure", type=int, default=1)
    parser.add_argument("--treatment-enum-max-rows", type=int, default=12)
    parser.add_argument(
        "--treatment-backup", choices=sorted(_BACKUP_MODES), default="hajek"
    )
    parser.add_argument(
        "--treatment-traversal",
        choices=sorted(_TRAVERSAL_MODES),
        default="balanced",
    )
    parser.add_argument("--control-exposure", type=int, default=0)
    parser.add_argument("--control-enum-max-rows", type=int, default=12)
    parser.add_argument(
        "--control-backup", choices=sorted(_BACKUP_MODES), default="hajek"
    )
    parser.add_argument(
        "--control-traversal", choices=sorted(_TRAVERSAL_MODES), default="balanced"
    )
    parser.add_argument("--leaf-batch", type=int, default=8)
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--fpu", type=float, default=-0.2)
    parser.add_argument("--margin-gain", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--z", type=float, default=1.96)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    args = _parse_args(argv)
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite completed match artifact: {output}"
        )
    treatment = SearchSpec(
        sims=args.treatment_sims if args.treatment_sims is not None else args.sims,
        chance_exposure=args.treatment_exposure,
        chance_enum_max_rows=args.treatment_enum_max_rows,
        chance_backup=args.treatment_backup,
        chance_traversal=args.treatment_traversal,
        leaf_batch=args.leaf_batch,
        cpuct=args.cpuct,
        fpu=args.fpu,
        margin_gain=args.margin_gain,
        alpha=args.alpha,
    )
    control = SearchSpec(
        sims=args.control_sims if args.control_sims is not None else args.sims,
        chance_exposure=args.control_exposure,
        chance_enum_max_rows=args.control_enum_max_rows,
        chance_backup=args.control_backup,
        chance_traversal=args.control_traversal,
        leaf_batch=args.leaf_batch,
        cpuct=args.cpuct,
        fpu=args.fpu,
        margin_gain=args.margin_gain,
        alpha=args.alpha,
    )
    net, checkpoint_config = load_checkpoint_network(checkpoint, args.device)
    result = evaluate_serial_search_match(
        net,
        treatment=treatment,
        control=control,
        paired_seeds=args.paired_seeds,
        seed_start=args.seed,
        device=args.device,
        z=args.z,
        verbose=args.verbose,
    )
    payload = {
        "schema_version": "kd-search-ab-v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_config": checkpoint_config,
        "paired_seed_count": int(args.paired_seeds),
        "seed_start": int(args.seed),
        "selection_reason": str(args.selection_reason),
        "treatment": asdict(treatment),
        "control": asdict(control),
        "result": result.to_dict(),
    }
    write_json(output, payload)
    return payload


if __name__ == "__main__":
    main()
