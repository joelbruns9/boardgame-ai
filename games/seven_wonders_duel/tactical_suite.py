"""G0 (`MODEL_GROWTH_PLAN.md`): the tactical suite -- the instrument the G1-G4
changes, and G3's sampling, are judged by.

Two commands:

* ``harvest`` walks replay buffers and files positions into classes with
  EXACT labels where the class has one (`tactics.rs` via
  `RustGame.classify_actions` and `RustGame.losing_mass`, and the endgame
  solver's recorded values), plus realized outcomes for the calibration
  classes.
* ``evaluate`` scores a checkpoint on those cases -- the raw network and/or a
  search at a given budget -- per class, by unique game as well as by row.

Classes (one position can be in more than one: `solver` and `ordinary` overlap
the rest):

==============  ==============================================  ===============
class           selection                                        label
==============  ==============================================  ===============
own_win         some action forces a win this turn               winning actions
forced_loss     every action hands the opponent a forced win     value -1
must_block      some actions lose by force, others do not        losing actions
reveal_trap     a revealing action has detected forced-loss      detected loss
                exposure; another action has none detected      exposure
solver          the endgame solver recorded a value              that value
predecessor     the mover's previous decision before walking     realized result
                into a forced loss on the played line
quiet           a win is within reach for either side but        realized result
                nothing is forced (negative control)
ordinary        uniform sample                                   realized result
==============  ==============================================  ===============

Values are actor-relative utilities in [-1, 1] throughout.

**Split.** Whole games are sealed by a hash of (iteration, seed): the default
`--split dev` never reads a sealed game, so design work cannot leak into the
held-out number. Report unique games, not rows -- one endgame yields many
correlated rows.

What the tactical labels do and do not certify. Every `-1` / `+1` action
label and every loss mass is a PROOF under the bounded predicates of
`tactics.py`. The absence of one is not: an unmarked alternative was not shown
to lose, which is not the same as being shown not to lose (an extra turn, or a
loss beyond the detector's horizon, is outside it). So `blunder` and
`trap_pick` mean "chose an action with a proven / detected loss while one
without a detected loss existed" -- an exposure diagnostic, not action regret.

Selection bias, stated once: positions come from the run's own self-play, so a
class measures failures CONDITIONAL on reaching such positions, not their
population frequency (`ordinary` is the population sample).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import time

from . import phase_e as pe
from . import tactics as tc
from .buffer import read_records, replay
from .codec import legal_action_indices
from .game import Phase
from .search import state_actor

CLASSES = (
    "own_win",
    "forced_loss",
    "must_block",
    "reveal_trap",
    "solver",
    "predecessor",
    "quiet",
    "ordinary",
)
#: Classes whose value label is exact (not a realized outcome).
EXACT_VALUE_CLASSES = ("own_win", "forced_loss", "solver")
SEALED_FRACTION = 0.2
SCHEMA = 1


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    cls: str
    split: str
    buffer: str
    iteration: int | None
    seed: int
    move: int
    #: Actor-relative label value: exact for EXACT_VALUE_CLASSES, else the
    #: realized result of the game for the mover.
    value: float
    value_exact: bool
    #: Legal action indices proven winning / losing (own_win, must_block).
    winning: tuple[int, ...] = ()
    losing: tuple[int, ...] = ()
    #: reveal_trap: `{action: losing probability mass}` for revealing actions.
    trap_mass: dict = field(default_factory=dict)
    #: predecessor: move index of the forced-loss position it led into.
    leads_to: int | None = None
    #: Moves from this position to the end of the game as played. Separates
    #: the trivial "the game ends on this move" rows from real tactics.
    plies_to_end: int | None = None


def sealed(iteration: int | None, seed: int, fraction: float = SEALED_FRACTION) -> bool:
    digest = hashlib.sha256(f"g0:{iteration}:{seed}".encode()).hexdigest()
    return int(digest[:8], 16) / 0x1_0000_0000 < fraction


def _realized(record, actor: int) -> float:
    if record.winner is None:
        return 0.0
    return 1.0 if record.winner == actor else -1.0


def _labels(state) -> list[int]:
    """Exact per-action labels; Rust in production, `tactics.py` without it."""

    try:
        from .rust_bridge import rust_game_from_state
    except ImportError:  # pragma: no cover - Python-only environment
        return tc.classify_actions(state)
    return list(rust_game_from_state(state).classify_actions())


def _losing_mass(state) -> list:
    """Per-action `(losing mass, reveals)` or None; Rust in production."""

    try:
        from .rust_bridge import rust_game_from_state
    except ImportError:  # pragma: no cover - Python-only environment
        return tc.losing_mass(state)
    return list(rust_game_from_state(state).losing_mass())


def _motif(state) -> bool:
    """A decisive win is within reach for either side (the quiet-control
    selection: same visible motif as the tactical classes)."""

    mover = state_actor(state)
    return tc._within_reach(state, mover, 1) or tc._within_reach(state, 1 - mover, 1)


def classify_record(
    record,
    buffer: str,
    rng: random.Random,
    *,
    ordinary_rate: float,
    per_game_cap: int,
) -> list[Case]:
    """Every case one game yields."""

    split = "sealed" if sealed(record.iteration, record.seed) else "dev"
    cases: list[Case] = []
    per_class: Counter = Counter()
    forced_loss_moves: list[tuple[int, int]] = []  # (move, actor)
    decisions: list[tuple[int, int, bool]] = []  # (move, actor, lost)

    def add(cls: str, move, actor: int, **fields) -> None:
        if per_class[cls] >= per_game_cap:
            return
        per_class[cls] += 1
        value = fields.pop("value", _realized(record, actor))
        exact = fields.pop("value_exact", False)
        cases.append(
            Case(
                id=f"{record.iteration}:{record.seed}:{move.i}:{cls}",
                cls=cls,
                split=split,
                buffer=buffer,
                iteration=record.iteration,
                seed=record.seed,
                move=move.i,
                value=value,
                value_exact=exact,
                plies_to_end=len(record.moves) - move.i,
                **fields,
            )
        )

    def visit(state, move) -> None:
        if state.phase is not Phase.PLAY_AGE:
            return
        actor = state_actor(state)
        legal = legal_action_indices(state)
        labels = _labels(state)
        winning = tuple(a for a, label in zip(legal, labels) if label == 1)
        losing = tuple(a for a, label in zip(legal, labels) if label == -1)
        lost = bool(legal) and len(losing) == len(legal)
        decisions.append((move.i, actor, lost))
        if winning:
            add("own_win", move, actor, value=1.0, value_exact=True,
                winning=winning, losing=losing)
        elif lost:
            add("forced_loss", move, actor, value=-1.0, value_exact=True)
            forced_loss_moves.append((move.i, actor))
        elif losing:
            add("must_block", move, actor, losing=losing)
        else:
            # A reveal trap: some revealing action hands the opponent a forced
            # win in some worlds (exact over every outcome, `tactics.losing_mass`),
            # and another action has NO detected loss exposure. That second
            # action is not thereby safe -- the detector is bounded -- so this
            # class measures exposure, not regret.
            masses = _losing_mass(state) if state.pending_choice is None else []
            traps = {
                a: m[0] for a, m in zip(legal, masses) if m is not None and m[1] and m[0] > 0.0
            }
            safe = any(m is not None and m[0] == 0.0 for m in masses)
            if traps and safe:
                add("reveal_trap", move, actor, trap_mass=traps)
            elif _motif(state):
                add("quiet", move, actor)
        if move.solver_value is not None:
            add("solver", move, actor, value=float(move.solver_value),
                value_exact=move.solver_regime == "exact")
        if rng.random() < ordinary_rate:
            add("ordinary", move, actor)

    replay(record, on_state=visit)
    moves = {move.i: move for move in record.moves}
    for loss_move, actor in forced_loss_moves:
        # The mover's previous decision on the played line, if it was not
        # already lost itself: the decision that walked into the loss.
        earlier = [
            (i, lost) for i, a, lost in decisions if a == actor and i < loss_move
        ]
        if not earlier:
            continue
        i, already_lost = earlier[-1]
        if already_lost or loss_move - i < 2:
            continue
        add("predecessor", moves[i], actor, leads_to=loss_move)
    return cases


def harvest(
    buffers: list[Path],
    out: Path,
    *,
    ordinary_rate: float = 0.02,
    per_game_cap: int = 3,
    seed: int = 0,
    max_games: int | None = None,
    log=print,
) -> dict:
    rng = random.Random(seed)
    counts: dict = defaultdict(Counter)
    games: dict = defaultdict(set)
    started = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = 0
    with out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": SCHEMA, "kind": "g0_cases"}) + "\n")
        for path in buffers:
            for record in read_records(path):
                if max_games is not None and seen >= max_games:
                    break
                seen += 1
                for case in classify_record(
                    record, str(path), rng,
                    ordinary_rate=ordinary_rate, per_game_cap=per_game_cap,
                ):
                    handle.write(json.dumps(asdict(case)) + "\n")
                    counts[case.split][case.cls] += 1
                    games[(case.split, case.cls)].add((case.iteration, case.seed))
            log(f"{path.name}: {seen} games, {round(time.time() - started)} s")
    summary = {
        "games": seen,
        "rows": {split: dict(c) for split, c in counts.items()},
        "unique_games": {
            f"{split}/{cls}": len(ids) for (split, cls), ids in sorted(games.items())
        },
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def read_cases(path: Path, split: str | None = "dev") -> list[Case]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        if header.get("schema") != SCHEMA:
            raise ValueError(f"{path} is not a schema-{SCHEMA} G0 case file")
        for line in handle:
            raw = json.loads(line)
            raw["winning"] = tuple(raw["winning"])
            raw["losing"] = tuple(raw["losing"])
            raw["trap_mass"] = {int(k): v for k, v in raw["trap_mass"].items()}
            case = Case(**raw)
            if split is None or case.split == split:
                cases.append(case)
    return cases


def load_states(cases: list[Case]) -> dict[str, object]:
    """`{case id: GameState}`, replaying each referenced game once."""

    wanted: dict = defaultdict(lambda: defaultdict(list))
    for case in cases:
        wanted[case.buffer][(case.iteration, case.seed)].append(case)
    states = {}
    for buffer, by_game in wanted.items():
        for record in read_records(Path(buffer)):
            needed = by_game.get((record.iteration, record.seed))
            if not needed:
                continue
            at = defaultdict(list)
            for case in needed:
                at[case.move].append(case.id)

            def grab(state, move, at=at):
                for case_id in at.get(move.i, ()):
                    states[case_id] = state.clone()

            replay(record, on_state=grab)
    missing = [case.id for case in cases if case.id not in states]
    if missing:
        raise ValueError(f"{len(missing)} cases not found in their buffers, e.g. {missing[:3]}")
    return states


@dataclass(slots=True)
class Reading:
    """One model read of one case: actor-relative value, chosen action, and
    the probability mass the model puts on each legal action."""

    value: float
    action: int
    mass: dict


def read_network(evaluator, states: list) -> list[Reading]:
    readings = []
    for start in range(0, len(states), 256):
        chunk = states[start:start + 256]
        for state, ev in zip(chunk, evaluator.evaluate_states(chunk)):
            legal = legal_action_indices(state)
            policy = [float(p) for p in ev.policy]
            best = max(range(len(legal)), key=lambda i: policy[i])
            readings.append(Reading(
                value=float(ev.wdl[0] - ev.wdl[2]),
                action=legal[best],
                mass=dict(zip(legal, policy)),
            ))
    return readings


def read_search(evaluator, states: list, sims: int, *, exact_tactics: bool,
                seed: int = 0, batch_cap: int = 256) -> list[Reading]:
    import seven_wonders_rust as swr

    from .rust_bridge import rust_flat_batch_adapter, rust_game_from_state

    adapter = rust_flat_batch_adapter(evaluator)
    previous = swr.exact_tactics()
    swr.set_exact_tactics(exact_tactics)
    try:
        readings = []
        for start in range(0, len(states), 64):
            chunk = states[start:start + 64]
            results = swr.search_many_flat_net(
                adapter,
                [rust_game_from_state(state) for state in chunk],
                [seed + start + i for i in range(len(chunk))],
                batch_cap, 1, sims, 16,
                force=True, puct_root=True,
            )
            for state, result in zip(chunk, results):
                legal = legal_action_indices(state)
                policy = [float(p) for p in result["policy"]]
                # argmax of the improved policy, as the arena plays.
                best = max(range(len(legal)), key=lambda i: policy[i])
                readings.append(Reading(
                    value=float(result["root_value"]),
                    action=legal[best],
                    mass=dict(zip(legal, policy)),
                ))
        return readings
    finally:
        swr.set_exact_tactics(previous)


def score(cases: list[Case], readings: list[Reading]) -> dict:
    """Per-class metrics. Rows AND unique games are reported: a class with
    300 rows from 40 endgames is 40 pieces of evidence."""

    by_class: dict = defaultdict(list)
    for case, reading in zip(cases, readings):
        by_class[case.cls].append((case, reading))
        # Each class again, split by distance to the end of the game: the
        # last couple of plies are mostly the move that ends it.
        if case.plies_to_end is not None:
            near = "near_end" if case.plies_to_end <= 2 else "deep"
            by_class[f"{case.cls}/{near}"].append((case, reading))
    report = {}
    for cls in [name for c in CLASSES for name in (c, f"{c}/near_end", f"{c}/deep")]:
        rows = by_class.get(cls)
        if not rows:
            continue
        base = cls.split("/")[0]
        errors = [r.value - c.value for c, r in rows]
        entry = {
            "rows": len(rows),
            "unique_games": len({(c.iteration, c.seed) for c, _ in rows}),
            "value_mae": sum(abs(e) for e in errors) / len(errors),
            "value_bias": sum(errors) / len(errors),
            "value_exact": all(c.value_exact for c, _ in rows),
        }
        decisive = [(c, r) for c, r in rows if c.value_exact and abs(c.value) == 1.0]
        if decisive:
            # Confidently on the wrong side of a proven result.
            entry["overconfident_wrong"] = sum(
                1 for c, r in decisive if r.value * c.value < -0.5
            ) / len(decisive)
        if base == "own_win":
            entry["found_win"] = _rate(rows, lambda c, r: r.action in c.winning)
            entry["mass_on_wins"] = _mean(rows, lambda c, r: sum(r.mass.get(a, 0.0) for a in c.winning))
        if base in ("own_win", "must_block"):
            with_losers = [(c, r) for c, r in rows if c.losing]
            if with_losers:
                entry["blunder"] = _rate(with_losers, lambda c, r: r.action in c.losing)
                entry["mass_on_blunders"] = _mean(
                    with_losers, lambda c, r: sum(r.mass.get(a, 0.0) for a in c.losing)
                )
        if base == "reveal_trap":
            entry["trap_pick"] = _rate(rows, lambda c, r: c.trap_mass.get(r.action, 0.0) > 0.0)
            # Detected forced-loss exposure of the chosen action: the
            # probability, over its reveals, that the opponent is then left a
            # forced win. Not regret -- the alternatives are not certified.
            entry["expected_losing_mass"] = _mean(rows, lambda c, r: c.trap_mass.get(r.action, 0.0))
        if base in ("ordinary", "quiet", "predecessor"):
            entry["calibration_ece"] = _ece(rows)
        report[cls] = entry
    return report


def _rate(rows, test) -> float:
    return sum(1 for c, r in rows if test(c, r)) / len(rows)


def _mean(rows, value) -> float:
    return sum(value(c, r) for c, r in rows) / len(rows)


def _ece(rows, bins: int = 10) -> float:
    """Expected calibration error of P(win) = (1 + value) / 2 against the
    realized win, over equal-width bins; draws count as half a win."""

    buckets: dict = defaultdict(list)
    for case, reading in rows:
        p = min(1.0, max(0.0, (1.0 + reading.value) / 2.0))
        buckets[min(bins - 1, int(p * bins))].append((p, (1.0 + case.value) / 2.0))
    total = len(rows)
    return sum(
        len(b) / total * abs(sum(p for p, _ in b) / len(b) - sum(y for _, y in b) / len(b))
        for b in buckets.values()
    )


def evaluate(
    cases_path: Path,
    checkpoint: str,
    *,
    sims: list[int],
    split: str = "dev",
    device: str = "cuda",
    precision: str = "bf16",
    exact_tactics: bool = True,
    max_per_class: int | None = None,
    seed: int = 0,
    readings_out: Path | None = None,
) -> dict:
    cases = read_cases(cases_path, split)
    if max_per_class is not None:
        rng = random.Random(seed)
        grouped: dict = defaultdict(list)
        for case in cases:
            grouped[case.cls].append(case)
        cases = []
        for cls in CLASSES:
            members = grouped.get(cls, [])
            rng.shuffle(members)
            cases.extend(members[:max_per_class])
    states_by_id = load_states(cases)
    states = [states_by_id[case.id] for case in cases]
    evaluator = pe.load_evaluator(checkpoint, device, precision)
    report = {
        "schema": SCHEMA,
        "checkpoint": checkpoint,
        "split": split,
        "exact_tactics": exact_tactics,
        "modes": {},
    }
    for budget in sims:
        started = time.time()
        if budget == 0:
            readings = read_network(evaluator, states)
            mode = "network"
        else:
            readings = read_search(evaluator, states, budget,
                                   exact_tactics=exact_tactics, seed=seed)
            mode = f"search_{budget}"
        report["modes"][mode] = {
            "seconds": round(time.time() - started, 1),
            "classes": score(cases, readings),
        }
        if readings_out is not None:
            write_readings(readings_out, mode, cases, readings, append=budget != sims[0])
    return report


def write_readings(path: Path, mode: str, cases: list[Case], readings: list[Reading],
                   *, append: bool) -> None:
    """One line per (mode, case): what the model read, scored per case, so two
    checkpoints can be compared PAIRED on identical positions (`compare`)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as handle:
        for case, reading in zip(cases, readings):
            row = {
                "mode": mode,
                "id": case.id,
                "cls": case.cls,
                "near_end": case.plies_to_end is not None and case.plies_to_end <= 2,
                "game": [case.iteration, case.seed],
                "value": reading.value,
                "abs_error": abs(reading.value - case.value),
                "action": reading.action,
            }
            if case.cls == "own_win":
                row["found_win"] = reading.action in case.winning
            if case.losing:
                row["blunder"] = reading.action in case.losing
            if case.cls == "reveal_trap":
                row["trap_pick"] = case.trap_mass.get(reading.action, 0.0) > 0.0
            handle.write(json.dumps(row) + "\n")


#: Per-case metrics `compare` pairs: binary ones are compared by discordant
#: pairs (exact McNemar), all by a game-clustered bootstrap of the paired
#: difference.
PAIRED_METRICS = ("found_win", "blunder", "trap_pick", "abs_error")


def _mcnemar(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts."""

    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _cluster_bootstrap(diffs_by_game: dict, *, draws: int, rng: random.Random) -> tuple:
    """95% interval of the mean paired difference, resampling whole GAMES (a
    game's cases are correlated -- the review's point that rows overstate
    evidence)."""

    games = list(diffs_by_game.values())
    means = []
    for _ in range(draws):
        sample = [games[rng.randrange(len(games))] for _ in games]
        total = sum(sum(g) for g in sample)
        count = sum(len(g) for g in sample)
        means.append(total / count)
    means.sort()
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def compare(a_path: Path, b_path: Path, *, draws: int = 2000, seed: int = 0) -> dict:
    """Paired comparison of two `--save-readings` files (A = baseline, B =
    candidate) on the cases they share, per mode, class and near-end/deep
    split. Differences are B - A; for `abs_error` lower is better, for
    `found_win` higher, for `blunder` / `trap_pick` lower."""

    def load(path):
        with Path(path).open(encoding="utf-8") as handle:
            return {(r["mode"], r["id"]): r for r in map(json.loads, handle)}

    a, b = load(a_path), load(b_path)
    shared = sorted(set(a) & set(b))
    rng = random.Random(seed)
    groups: dict = defaultdict(list)
    for key in shared:
        row = a[key]
        groups[(row["mode"], row["cls"])].append(key)
        groups[(row["mode"], f"{row['cls']}/{'near_end' if row['near_end'] else 'deep'}")].append(key)
    report: dict = {"a": str(a_path), "b": str(b_path), "shared_cases": len(shared), "modes": {}}
    for (mode, cls), keys in sorted(groups.items()):
        entry = {"cases": len(keys), "games": len({tuple(a[k]["game"]) for k in keys})}
        for metric in PAIRED_METRICS:
            pairs = [(a[k][metric], b[k][metric], tuple(a[k]["game"]))
                     for k in keys if metric in a[k] and metric in b[k]]
            if not pairs:
                continue
            by_game: dict = defaultdict(list)
            for x, y, game in pairs:
                by_game[game].append(float(y) - float(x))
            low, high = _cluster_bootstrap(by_game, draws=draws, rng=rng)
            stats = {
                "a": sum(float(x) for x, _, _ in pairs) / len(pairs),
                "b": sum(float(y) for _, y, _ in pairs) / len(pairs),
                "diff_ci95": [low, high],
            }
            if metric != "abs_error":
                only_a = sum(1 for x, y, _ in pairs if x and not y)
                only_b = sum(1 for x, y, _ in pairs if y and not x)
                stats.update(only_a=only_a, only_b=only_b, mcnemar_p=_mcnemar(only_a, only_b))
            entry[metric] = stats
        report["modes"].setdefault(mode, {})[cls] = entry
    return report


def _iteration_range(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    h = sub.add_parser("harvest", help="build a case file from replay buffers")
    h.add_argument("--buffers-dir", type=Path, required=True)
    h.add_argument("--iterations", required=True, help="e.g. 61-100 or 60,80,100")
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--ordinary-rate", type=float, default=0.02)
    h.add_argument("--per-game-cap", type=int, default=3)
    h.add_argument("--max-games", type=int, default=None)
    h.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("evaluate", help="score a checkpoint on a case file")
    e.add_argument("--cases", type=Path, required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--sims", default="0,64,800",
                   help="comma-separated budgets; 0 = the raw network")
    e.add_argument("--split", choices=("dev", "sealed"), default="dev")
    e.add_argument("--device", default="cuda")
    e.add_argument("--precision", default="bf16")
    e.add_argument("--exact-tactics", action=argparse.BooleanOptionalAction, default=True)
    e.add_argument("--max-per-class", type=int, default=None)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--save-readings", action="store_true",
                   help="also write per-case readings next to --out "
                   "(<out>.readings.jsonl) for a paired `compare`")
    e.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare", help="paired comparison of two saved readings files")
    c.add_argument("--a", type=Path, required=True, help="baseline readings")
    c.add_argument("--b", type=Path, required=True, help="candidate readings")
    c.add_argument("--draws", type=int, default=2000)
    c.add_argument("--out", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "harvest":
        buffers = [
            args.buffers_dir / f"iter_{i:04d}.jsonl" for i in _iteration_range(args.iterations)
        ]
        summary = harvest(
            buffers, args.out,
            ordinary_rate=args.ordinary_rate, per_game_cap=args.per_game_cap,
            seed=args.seed, max_games=args.max_games,
        )
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "compare":
        report = compare(args.a, args.b, draws=args.draws)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 0
    report = evaluate(
        args.cases, args.checkpoint,
        sims=[int(s) for s in args.sims.split(",")],
        split=args.split, device=args.device, precision=args.precision,
        exact_tactics=args.exact_tactics, max_per_class=args.max_per_class,
        seed=args.seed,
        readings_out=args.out.with_suffix(".readings.jsonl") if args.save_readings else None,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
