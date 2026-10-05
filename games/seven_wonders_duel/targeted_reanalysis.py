"""G8.2 (laptop-sized): targeted reanalysis of run07 buffer positions.

The final-run preparation (`MODEL_GROWTH_PLAN.md`) pretrains on the corrected
run07 buffers. Correction (G1/G2/G2b) fixes what exact checks can see; it
cannot fix a target that is wrong because run07's SEARCH was too shallow or
too fan-out-starved. Re-searching every position would cost as much as
generating the games, so this re-searches a SUBSET with today's search -- G4
exact tactics on, the full budget, root chance enumerated -- and writes the
results as an overlay that derivation applies (`dataset.apply_reanalysis`):

* ``pre_decisive`` -- decisions within ``window`` plies before a position with
  a forced win, forced loss or must_block (the predecessor family G2b cannot
  see: a walk into a loss two or more moves later);
* ``reveal`` -- the played move uncovered a card (run07 read revealing moves
  ~20 pts optimistic in the reviewed games: chance fan-out);
* ``cheap`` -- a sample of the cheap-search moves, whose 100-sim targets taught
  near-end blunders (G3/G2b A/B).

Positions are searched in batches through `rust_coalesced_reanalysis`, which
pools network calls across positions (the self-play coalescer). The overlay is
JSONL keyed by ``(iteration, seed, move)`` and appended as batches finish, so
a stopped run resumes where it left off. G0's sealed games are never selected.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

from .buffer import read_records, replay
from .codec import decode_action, legal_action_indices
from .game import ChanceKind, Phase
from .search import chance_signature, state_actor
from .tactical_suite import sealed

REASONS = ("pre_decisive", "reveal", "cheap")
SCHEMA = 1


@dataclass(frozen=True)
class Target:
    iteration: int | None
    seed: int
    move: int
    reason: str


def _decisive(state) -> bool:
    """The position has a proven per-action result (G0's classes)."""

    from .rust_bridge import rust_game_from_state

    return any(label != 0 for label in rust_game_from_state(state).classify_actions())


def _reveals(state, action_index: int) -> bool:
    specs = chance_signature(state, decode_action(state, action_index))
    return any(spec.kind is ChanceKind.CARD_REVEAL for spec in specs)


def select_record(record, rng: random.Random, *, window: int, per_game_cap: int,
                  cheap_rate: float) -> list[Target]:
    """The targets one game yields, at most `per_game_cap`, pre_decisive first."""

    if sealed(record.iteration, record.seed):
        return []
    decisive: list[int] = []
    reveals: list[int] = []
    cheap: list[int] = []
    playing: list[int] = []

    def visit(state, move) -> None:
        if state.phase is not Phase.PLAY_AGE:
            return
        playing.append(move.i)
        if _decisive(state):
            decisive.append(move.i)
            return
        if _reveals(state, move.action):
            reveals.append(move.i)
        if move.policy_excluded and move.sims > 0 and rng.random() < cheap_rate:
            cheap.append(move.i)

    replay(record, on_state=visit)
    decisive_set = set(decisive)
    pre = sorted({
        i for d in decisive for i in playing
        if d - window <= i < d and i not in decisive_set
    })
    chosen: list[Target] = []
    seen: set[int] = set()
    for reason, moves in (("pre_decisive", pre), ("reveal", reveals), ("cheap", cheap)):
        for i in moves:
            if len(chosen) >= per_game_cap:
                return chosen
            if i in seen:
                continue
            seen.add(i)
            chosen.append(Target(record.iteration, record.seed, i, reason))
    return chosen


def _done_keys(out: Path) -> set[tuple]:
    if not out.exists():
        return set()
    keys = set()
    with out.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("kind") == "header":
                continue
            keys.add((row["iteration"], row["seed"], row["move"]))
    return keys


def run(
    buffers: list[Path],
    out: Path,
    *,
    checkpoint: str,
    sims: int = 1600,
    device: str = "cuda",
    precision: str = "bf16",
    window: int = 4,
    per_game_cap: int = 6,
    cheap_rate: float = 0.05,
    batch_positions: int = 256,
    max_positions: int | None = None,
    seed: int = 0,
    log=print,
) -> dict:
    import seven_wonders_rust as swr

    from .phase_e import load_evaluator
    from .rust_bridge import prepare_reanalysis_position, rust_coalesced_reanalysis_prepared

    done = _done_keys(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.write_text(json.dumps({
            "kind": "header", "schema": SCHEMA, "checkpoint": checkpoint, "sims": sims,
            "window": window, "per_game_cap": per_game_cap, "cheap_rate": cheap_rate,
        }) + "\n", encoding="utf-8")
    evaluator = load_evaluator(checkpoint, device, precision)
    rng = random.Random(seed)
    counts: Counter = Counter()
    started = time.time()
    pending: list[tuple] = []  # (target, rust_game, legal)

    def flush() -> None:
        if not pending:
            return
        previous = swr.exact_tactics()
        swr.set_exact_tactics(True)
        try:
            results, _coalescing = rust_coalesced_reanalysis_prepared(
                evaluator,
                [(game, legal, rng.randrange(2**31)) for _t, game, legal in pending],
                sims=sims, top_k=16, force=True, puct_root=True,
            )
        finally:
            swr.set_exact_tactics(previous)
        with out.open("a", encoding="utf-8") as handle:
            for (target, _game, legal), result in zip(pending, results):
                handle.write(json.dumps({
                    "iteration": target.iteration,
                    "seed": target.seed,
                    "move": target.move,
                    "reason": target.reason,
                    "sims": result.sims,
                    "root_value": result.root_value,
                    "policy": {str(a): float(result.policy_target.get(a, 0.0)) for a in legal},
                }) + "\n")
                counts[target.reason] += 1
        log(f"[g8.2] {sum(counts.values())} positions re-searched "
            f"({dict(counts)}), {round(time.time() - started)} s")
        pending.clear()

    total = 0
    for path in buffers:
        for record in read_records(path):
            targets = [
                t for t in select_record(record, rng, window=window,
                                         per_game_cap=per_game_cap, cheap_rate=cheap_rate)
                if (t.iteration, t.seed, t.move) not in done
            ]
            if not targets:
                continue
            wanted = {t.move: t for t in targets}

            def grab(state, move, wanted=wanted):
                target = wanted.get(move.i)
                if target is not None:
                    pending.append((target, *prepare_reanalysis_position(state)))

            replay(record, on_state=grab)
            total += len(targets)
            if len(pending) >= batch_positions:
                flush()
            if max_positions is not None and total >= max_positions:
                flush()
                return {"positions": dict(counts), "seconds": round(time.time() - started)}
    flush()
    return {"positions": dict(counts), "seconds": round(time.time() - started)}


def load_overlay(paths) -> dict[tuple, dict]:
    """`{(iteration, seed): {move: {"policy": {action: p}, "root_value": v}}}`
    from one or more overlay files; later files win. Keyed by GAME first so
    derivation looks up one record's entries in O(1)."""

    overlay: dict[tuple, dict] = {}
    for path in [paths] if isinstance(paths, (str, Path)) else paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("kind") == "header":
                    continue
                overlay.setdefault((row["iteration"], row["seed"]), {})[row["move"]] = {
                    "policy": {int(a): p for a, p in row["policy"].items()},
                    "root_value": row["root_value"],
                    "reason": row.get("reason"),
                }
    return overlay


def _iterations(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--buffers-dir", type=Path, required=True)
    parser.add_argument("--iterations", required=True, help="e.g. 41-100")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sims", type=int, default=1600)
    parser.add_argument("--window", type=int, default=4,
                        help="plies before a decisive position counted as pre_decisive")
    parser.add_argument("--per-game-cap", type=int, default=6)
    parser.add_argument("--cheap-rate", type=float, default=0.05,
                        help="share of cheap-search moves sampled")
    parser.add_argument("--batch-positions", type=int, default=256)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    buffers = [args.buffers_dir / f"iter_{i:04d}.jsonl" for i in _iterations(args.iterations)]
    summary = run(
        buffers, args.out, checkpoint=args.checkpoint, sims=args.sims, device=args.device,
        precision=args.precision, window=args.window, per_game_cap=args.per_game_cap,
        cheap_rate=args.cheap_rate, batch_positions=args.batch_positions,
        max_positions=args.max_positions, seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
