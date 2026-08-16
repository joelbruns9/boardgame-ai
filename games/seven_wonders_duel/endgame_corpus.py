"""The gate for porting the exact endgame solver to Rust.

The port only pays if the Rust solver returns *the same answers* as the Python
one -- an exact solver that is quietly wrong is worse than no solver, because
its output goes straight into training labels as ground truth. So the port is
gated the way the engine, encoder and searcher ports were: a fixed corpus of
real endgame positions, each with the reference answer, and a checker that any
candidate solver must satisfy.

**What a record holds.** Not the position -- the *recipe* for it: a seed, a bot
pairing and a ply, which regenerate it exactly, plus a fingerprint of the state
that recipe lands on. Storing the recipe rather than a serialized state means
the corpus cannot drift out of sync with the engine silently: if the engine
changes what that seed produces, the fingerprint stops matching and the corpus
says so instead of comparing solvers on a position neither would ever see.

Alongside it, the reference answer: regime, root value, and the exact value of
every legal action.

**Both regimes are wanted.** ``exact`` positions have no chance edge left and
are where alpha-beta will do the most work; ``exact_expectimax`` positions still
have face-down cards or a Great Library draw, and are the ones a chance-free
solver would have to hand back. The build reports the split, because a corpus of
only one kind would gate only half the port.

    python -m games.seven_wonders_duel.endgame_corpus --build
    python -m games.seven_wonders_duel.endgame_corpus            # check
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .advisor_endgame import solve_position
from .buffer import logic_fingerprint
from .encoder_audit import DEFAULT_PAIRINGS, make_bot
from .engine import apply_action
from .game import GameState, Phase, new_game

CORPUS = Path(__file__).parent / "testdata" / "endgame_corpus.jsonl"

#: Positions with at most this many tableau cards left are candidates. Above ~6
#: the Python reference itself times out (measured: 5 cards ~11.5k nodes/8s,
#: 6 ~21k/14s, 7 ~37k/29s), and a reference that cannot answer cannot gate.
MAX_PRESENT = 6

#: The reference solve's own budget. Deliberately generous: this runs once, at
#: build time, and a position the reference gives up on is simply not banked.
BUILD_MAX_NODES = 2_000_000
BUILD_MAX_SECS = 60.0


@dataclass(frozen=True)
class Position:
    """A regenerable endgame position."""

    seed: int
    left: str
    right: str
    ply: int
    present: int
    fingerprint: list[int]
    game: GameState | None = None


@dataclass
class CheckReport:
    """What a candidate solver got right, and exactly how it differed."""

    records: int = 0
    checked: int = 0
    skipped_stale: int = 0
    skipped_unsolved: int = 0
    regimes: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        split = ", ".join(f"{k}={v}" for k, v in sorted(self.regimes.items()))
        head = (
            f"{self.checked}/{self.records} positions checked ({split or 'none'})"
        )
        if self.skipped_stale:
            head += f"; {self.skipped_stale} STALE (engine no longer produces them)"
        if self.skipped_unsolved:
            head += f"; {self.skipped_unsolved} the candidate declined"
        if self.problems:
            return head + f"\n!! {len(self.problems)} DISAGREEMENTS\n" + "\n".join(
                f"   {p}" for p in self.problems[:10]
            )
        return head + "\nevery answer matched the reference"


def _actor(game: GameState) -> int:
    return (
        game.pending_choice.player
        if game.pending_choice is not None
        else game.active_player
    )


def harvest(
    seeds: Iterable[int],
    *,
    max_present: int = MAX_PRESENT,
    per_game: int = 2,
) -> list[Position]:
    """Play bot games and keep the endgame positions they pass through."""

    out: list[Position] = []
    for index, seed in enumerate(seeds):
        left, right = DEFAULT_PAIRINGS[index % len(DEFAULT_PAIRINGS)]
        game = new_game(seed)
        bots = (make_bot(left, seed), make_bot(right, seed + 10_000))
        kept = 0
        ply = 0
        while game.phase is not Phase.COMPLETE:
            # Age III only. An earlier age always reaches the next age's deal
            # before terminal, and that edge is sample-only (not enumerable), so
            # the solver refuses the position however few cards are on the board.
            if game.phase is Phase.PLAY_AGE and game.age == 3 and kept < per_game:
                present = sum(1 for c in game.tableau.cards.values() if c.present)
                if 1 <= present <= max_present:
                    out.append(
                        Position(
                            seed=seed,
                            left=left,
                            right=right,
                            ply=ply,
                            present=present,
                            fingerprint=logic_fingerprint(game),
                            game=game.clone(),
                        )
                    )
                    kept += 1
            apply_action(game, bots[_actor(game)].select_action(game))
            ply += 1
    return out


def regenerate(record: dict) -> GameState | None:
    """Replay the recipe. ``None`` when it no longer lands where it did."""

    game = new_game(int(record["seed"]))
    bots = (
        make_bot(record["left"], int(record["seed"])),
        make_bot(record["right"], int(record["seed"]) + 10_000),
    )
    for _ in range(int(record["ply"])):
        if game.phase is Phase.COMPLETE:
            return None
        apply_action(game, bots[_actor(game)].select_action(game))
    if logic_fingerprint(game) != list(record["fingerprint"]):
        return None
    return game


def build(seeds: Iterable[int], *, path: Path = CORPUS) -> CheckReport:
    """Solve every harvested position with the Python reference and bank it."""

    report = CheckReport()
    rows = []
    for position in harvest(seeds):
        started = time.perf_counter()
        solved = solve_position(
            position.game,
            deadline=started + BUILD_MAX_SECS,
            max_nodes=BUILD_MAX_NODES,
        )
        report.records += 1
        if solved is None:
            report.skipped_unsolved += 1
            continue
        report.regimes[solved["regime"]] = report.regimes.get(solved["regime"], 0) + 1
        report.checked += 1
        rows.append(
            {
                "seed": position.seed,
                "left": position.left,
                "right": position.right,
                "ply": position.ply,
                "present": position.present,
                "fingerprint": position.fingerprint,
                "regime": solved["regime"],
                "root_value": solved["root_value"],
                "best_index": solved["best_index"],
                "per_action_value": {
                    str(k): v for k, v in solved["per_action_value"].items()
                },
                "reference_nodes": solved["nodes"],
                "reference_secs": round(time.perf_counter() - started, 3),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return report


def load(path: Path = CORPUS) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check(
    solver: Callable[[GameState], dict | None],
    *,
    path: Path = CORPUS,
    tolerance: float = 1e-9,
) -> CheckReport:
    """Compare a candidate solver against every banked reference answer.

    Every action's value is compared, not just the root: a solver that agrees on
    the root while mispricing the alternatives would still poison a policy label.
    The best-action *set* is compared rather than the single index, because ties
    are common in endgames and which one a solver names is not meaningful --
    which ones it proves optimal is.
    """

    report = CheckReport()
    for record in load(path):
        report.records += 1
        game = regenerate(record)
        if game is None:
            report.skipped_stale += 1
            continue
        answer = solver(game)
        if answer is None:
            report.skipped_unsolved += 1
            continue
        report.checked += 1
        regime = str(answer.get("regime"))
        report.regimes[regime] = report.regimes.get(regime, 0) + 1
        where = f"seed={record['seed']} ply={record['ply']} present={record['present']}"

        if regime != record["regime"]:
            report.problems.append(
                f"{where}: regime {regime} != reference {record['regime']}"
            )
        expected = {int(k): v for k, v in record["per_action_value"].items()}
        got = {int(k): float(v) for k, v in answer["per_action_value"].items()}
        if set(got) != set(expected):
            report.problems.append(
                f"{where}: action set differs "
                f"(missing {sorted(set(expected) - set(got))}, "
                f"extra {sorted(set(got) - set(expected))})"
            )
            continue
        wrong = [
            f"{index}: {got[index]:.9g} != {expected[index]:.9g}"
            for index in sorted(expected)
            if abs(got[index] - expected[index]) > tolerance
        ]
        if wrong:
            report.problems.append(f"{where}: {'; '.join(wrong[:4])}")
            continue
        best_expected = {
            i for i, v in expected.items() if v >= max(expected.values()) - tolerance
        }
        best_got = {i for i, v in got.items() if v >= max(got.values()) - tolerance}
        if best_expected != best_got:
            report.problems.append(
                f"{where}: best-action set {sorted(best_got)} != "
                f"{sorted(best_expected)}"
            )

        # `root_value` and `best_index` are separate fields on the result, and
        # they are the ones callers actually read -- a value target and a policy
        # target. Deriving the verdict from `per_action_value` alone would pass a
        # solver whose per-action table was right while those two were stale.
        root_value = float(answer["root_value"])
        if abs(root_value - float(record["root_value"])) > tolerance:
            report.problems.append(
                f"{where}: root_value {root_value:.9g} != "
                f"{float(record['root_value']):.9g}"
            )
        best_index = int(answer["best_index"])
        # Not the reference's index: ties are common in endgames and which one a
        # solver names is arbitrary. That it names a proven-optimal one is not.
        if best_index not in best_expected:
            report.problems.append(
                f"{where}: best_index {best_index} is not optimal "
                f"(optimal: {sorted(best_expected)})"
            )
    return report


def rust_solver(
    *, max_nodes: int = BUILD_MAX_NODES, max_secs: float = BUILD_MAX_SECS,
    policy_mode: str = "exact",
) -> Callable[[GameState], dict | None]:
    """The Rust solver, on a state injected from the Python position.

    Injection rather than replay on purpose: it is the same boundary the
    advisor crosses, and it is the one that has been caught carrying a
    divergence before (see ``test_rust_state_injection``).
    """

    from .rust_bridge import rust_game_from_state

    def solve(game: GameState) -> dict | None:
        rust_game = rust_game_from_state(game)
        answer = rust_game.solve_endgame(max_nodes, max_secs, policy_mode)
        if answer is None:
            return None
        return {
            "regime": answer["regime"],
            "root_value": answer["root_value"],
            "best_index": answer["best_index"],
            "per_action_value": {
                int(k): float(v) for k, v in answer["per_action_value"].items()
            },
            "nodes": answer["nodes"],
        }

    return solve


def reference_solver(game: GameState) -> dict | None:
    """The Python solver, with a budget generous enough to never be the reason."""

    return solve_position(
        game, deadline=time.perf_counter() + BUILD_MAX_SECS, max_nodes=BUILD_MAX_NODES
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", action="store_true", help="rebuild the corpus")
    parser.add_argument("--games", type=int, default=60, help="games to harvest from")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    seeds = range(args.seed, args.seed + args.games)
    if args.build:
        report = build(seeds)
        print(f"built {CORPUS}")
        print(report)
        return 0 if report.checked else 1

    # Default: check the reference against itself. Worth being able to run --
    # it proves the corpus still regenerates under today's engine, which is the
    # thing that silently rots.
    report = check(reference_solver)
    print(report)
    return 1 if (report.problems or report.skipped_stale) else 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
