"""Enhanced Option A: CPU self-play -> replayable-source data buffer (NNUE Step 3).

GPU-free. Self-play with the Rust-hosted expectiminimax searcher (kr.RustSearch)
as the policy, recording the REPLAYABLE SOURCE of each game -- initial deck/seed,
start player, rules config, and the (placement, pick) action trajectory -- plus
the official-cascade outcome as the label. Per-position training features are a
*derived* convenience (replay the source), so a future feature-schema change
needs no new self-play. This is the run10-encoder-lock fix.

"Enhanced" vs raw depth-2 self-play (which is narrow / uneven round awareness):
per-game varied seed, start player, and search depth; epsilon-random exploration
that is higher in the opening; and a few forced-random opening plies. Whole games
are split into train/val/test by seed hash (no position leakage). Provenance +
engine/catalog hash are stored per game so stale buffers fail loudly.

Labels use the official outcome cascade (RustGameState.official_outcome via
SearchEngine); genuine draws stay 0 and remain in the data.

Run:
    PYTHONPATH=. python -m games.kingdomino.nnue.datagen \
        --games 1000 --out runs/kingdomino/nnue_data/pilot --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from games.kingdomino.game import GameState
from games.kingdomino.dominoes import DOMINOES

# Bump if any rules/engine change alters how a (deck, actions) trajectory unfolds,
# so replay of an old buffer against a new engine fails the hash check loudly.
ENGINE_VERSION = 1
# Bump if the on-disk record schema / action serialization changes.
FORMAT_VERSION = 1
GAME_OVER = 3  # RustGameState phase code


def git_provenance() -> dict:
    """Git commit + dirty-state/diff hash of the tree that produced a buffer, so a
    future buffer's exact engine is identifiable even if someone forgets to bump
    ENGINE_VERSION. Best-effort (empty strings if git is unavailable)."""
    here = os.path.dirname(os.path.abspath(__file__))

    def _run(args):
        try:
            return subprocess.run(["git", *args], cwd=here, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace",
                                  timeout=10).stdout.strip()
        except Exception:
            return ""

    commit = _run(["rev-parse", "HEAD"])
    dirty_files = _run(["status", "--porcelain"])
    dirty = bool(dirty_files)
    diff = _run(["diff", "HEAD"]) if dirty else ""
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "git_diff_sha": hashlib.sha256(diff.encode()).hexdigest()[:16] if dirty else "",
    }


def catalog_hash() -> str:
    """Stable hash of the domino catalog + used so a catalog change invalidates
    a buffer (IDs -> (terrain, crowns) composition is what trajectories assume)."""
    items = [
        (did, int(DOMINOES[did].a.terrain), int(DOMINOES[did].a.crowns),
         int(DOMINOES[did].b.terrain), int(DOMINOES[did].b.crowns))
        for did in sorted(DOMINOES)
    ]
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class GenConfig:
    eval: str = "pick_aware"          # RustSearch heuristic (GPU-free teacher)
    nnue_path: Optional[str] = None    # required for an NNUE-backed eval
    nnue_sha256: str = ""              # filled once by generate(), not per game
    move_secs: float = 0.0             # >0 selects operational iterative deepening
    max_depth: int = 8                 # operational cap; depth_choices for fixed search
    selective_width: Optional[int] = None
    selective_root_width: Optional[int] = None
    selective_min_depth: int = 4
    depth_choices: tuple = (2, 3)     # per-game varied depth for diversity
    depth_weights: tuple = (0.85, 0.15)  # depth-3 is ~27x depth-2; keep it a minority
    chance_samples: int = 16
    harmony: bool = True
    middle_kingdom: bool = True
    epsilon_open: float = 0.25        # exploration rate for the first explore_plies
    epsilon_tail: float = 0.05        # exploration rate afterwards
    explore_plies: int = 8            # opening length that gets epsilon_open
    random_opening: int = 2           # forced fully-random opening plies (awkward starts)
    val_frac: float = 0.1
    test_frac: float = 0.1


def _ser_action(a):
    """(placement|None, pick|None) -> JSON-safe [ [x1,y1,x2,y2,flipped]|null, pick|null ]."""
    placement, pick = a
    p = None if placement is None else [int(placement[0]), int(placement[1]),
                                        int(placement[2]), int(placement[3]),
                                        bool(placement[4])]
    return [p, None if pick is None else int(pick)]


def _deser_action(sa):
    """Inverse of _ser_action -> (placement_tuple|None, pick|None) for RustGameState.step."""
    p, pick = sa
    placement = None if p is None else (int(p[0]), int(p[1]), int(p[2]), int(p[3]), bool(p[4]))
    return placement, (None if pick is None else int(pick))


def _epsilon(move_i: int, cfg: GenConfig) -> float:
    return cfg.epsilon_open if move_i < cfg.explore_plies else cfg.epsilon_tail


def play_one_game(seed: int, cfg: GenConfig) -> dict:
    """Play one self-play game and return its replayable-source record (a dict)."""
    import kingdomino_rust as kr

    gs = GameState.new(seed=seed)                 # canonical deck / row / start player
    start_player = int(gs.start_player)
    deck = [int(x) for x in gs.deck]              # post-initial-deal remaining deck (ordered)
    row = [int(x) for x in gs.current_row]        # initial 4-domino row
    rs = kr.RustGameState(start_player, deck, row, cfg.harmony, cfg.middle_kingdom)

    rng = random.Random((seed * 2654435761) & 0xFFFFFFFFFFFF)
    depth = rng.choices(list(cfg.depth_choices), weights=list(cfg.depth_weights))[0]
    search = kr.RustSearch(
        depth=(cfg.max_depth if cfg.move_secs > 0 else depth),
        chance_samples=cfg.chance_samples,
        eval=cfg.eval,
        seed=seed,
        nnue_path=cfg.nnue_path,
    )

    actions = []
    move_i = 0
    while rs.phase != GAME_OVER:
        legal = rs.legal_actions()
        if len(legal) == 1:
            a = legal[0]
        elif move_i < cfg.random_opening or rng.random() < _epsilon(move_i, cfg):
            a = legal[rng.randrange(len(legal))]
        else:
            if cfg.move_secs > 0:
                a = search.choose_action_timed(
                    rs, max_secs=cfg.move_secs, max_depth=cfg.max_depth,
                    selective_width=cfg.selective_width,
                    selective_root_width=cfg.selective_root_width,
                    selective_min_depth=cfg.selective_min_depth,
                ).action
            else:
                a = search.choose_action(
                    rs, (seed ^ (move_i * 0x9E3779B1)) & 0xFFFFFFFFFFFF
                )
        actions.append(_ser_action(a))
        rs = rs.step(a[0], a[1])
        move_i += 1

    s0, s1 = rs.scores()
    outcome = int(kr.SearchEngine(rs).official_outcome())  # +1 P0 / -1 P1 / 0 draw
    return {
        "seed": seed,
        "start_player": start_player,
        "deck": deck,
        "current_row": row,
        "harmony": cfg.harmony,
        "middle_kingdom": cfg.middle_kingdom,
        "actions": actions,
        "final_scores": [int(s0), int(s1)],
        "outcome_p0": outcome,
        "n_positions": len(actions),
        "provenance": {"policy": f"rust_search:{cfg.eval}", "depth": depth,
                       "search_mode": ("operational" if cfg.move_secs > 0 else "fixed_depth"),
                       "move_secs": cfg.move_secs, "max_depth": cfg.max_depth,
                       "selective_width": cfg.selective_width,
                       "selective_root_width": cfg.selective_root_width,
                       "selective_min_depth": cfg.selective_min_depth,
                       "nnue_path": cfg.nnue_path, "nnue_sha256": cfg.nnue_sha256,
                       "chance_samples": cfg.chance_samples,
                       "epsilon_open": cfg.epsilon_open, "epsilon_tail": cfg.epsilon_tail,
                       "explore_plies": cfg.explore_plies,
                       "random_opening": cfg.random_opening, "seed": seed},
        "engine_version": ENGINE_VERSION,
        "format_version": FORMAT_VERSION,
        "catalog_hash": catalog_hash(),
    }


def replay(rec: dict) -> tuple[int, int, int]:
    """Rebuild the game from the stored source and return (s0, s1, outcome_p0).

    Proves the record is genuine replayable source: (deck, actions) alone
    reproduce the exact final scores and official outcome, with no encoded
    features stored (the run10 lock-in is avoided)."""
    import kingdomino_rust as kr

    rs = kr.RustGameState(rec["start_player"], list(rec["deck"]), list(rec["current_row"]),
                          rec["harmony"], rec["middle_kingdom"])
    for sa in rec["actions"]:
        placement, pick = _deser_action(sa)
        rs = rs.step(placement, pick)
    s0, s1 = rs.scores()
    return int(s0), int(s1), int(kr.SearchEngine(rs).official_outcome())


class StaleBufferError(ValueError):
    """A record was produced by a different engine/catalog/format than this code."""


def load_records(path: str, *, strict: bool = True, expect_rules: Optional[dict] = None):
    """Load game records from a .jsonl file or a run directory, hard-failing on any
    stale-buffer signal so an incompatible buffer can never silently train a model.

    Validated per record (strict): engine_version, format_version, and catalog_hash
    must match THIS code; the whole buffer must share one rules config; if
    expect_rules is given, that config must match too. Returns list[dict].
    """
    if os.path.isdir(path):
        files = [os.path.join(path, f"{s}.jsonl") for s in ("train", "val", "test")]
        files = [f for f in files if os.path.exists(f)]
    else:
        files = [path]

    cat = catalog_hash()
    recs, rules_seen = [], set()
    for fp in files:
        with open(fp) as f:
            for ln, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if strict:
                    if rec.get("engine_version") != ENGINE_VERSION:
                        raise StaleBufferError(
                            f"{fp}:{ln}: engine_version {rec.get('engine_version')} "
                            f"!= {ENGINE_VERSION}")
                    if rec.get("format_version") != FORMAT_VERSION:
                        raise StaleBufferError(
                            f"{fp}:{ln}: format_version {rec.get('format_version')} "
                            f"!= {FORMAT_VERSION}")
                    if rec.get("catalog_hash") != cat:
                        raise StaleBufferError(
                            f"{fp}:{ln}: catalog_hash {rec.get('catalog_hash')} != {cat}")
                rules = (bool(rec["harmony"]), bool(rec["middle_kingdom"]))
                rules_seen.add(rules)
                if expect_rules is not None:
                    want = (bool(expect_rules["harmony"]), bool(expect_rules["middle_kingdom"]))
                    if rules != want:
                        raise StaleBufferError(f"{fp}:{ln}: rules {rules} != expected {want}")
                recs.append(rec)
    if strict and len(rules_seen) > 1:
        raise StaleBufferError(f"buffer mixes rules configs: {rules_seen}")
    return recs


def _split_of(seed: int, cfg: GenConfig) -> str:
    """Deterministic whole-game split by seed hash (no position leakage)."""
    h = int(hashlib.sha256(f"split:{seed}".encode()).hexdigest(), 16) % 10_000 / 10_000.0
    if h < cfg.test_frac:
        return "test"
    if h < cfg.test_frac + cfg.val_frac:
        return "val"
    return "train"


def _worker(args):
    seed, cfg = args
    return play_one_game(seed, cfg)


def generate(n_games: int, out_dir: str, cfg: GenConfig, workers: int = 1,
             seed_start: int = 0, verify: bool = True) -> dict:
    if cfg.eval in {"nnue", "sparse_nnue_ref", "sparse_nnue", "sparse_nnue_q"}:
        if not cfg.nnue_path:
            raise ValueError(f"eval={cfg.eval!r} requires nnue_path")
        artifact = Path(cfg.nnue_path).resolve()
        if not artifact.is_file():
            raise FileNotFoundError(f"NNUE artifact not found: {artifact}")
        cfg.nnue_path = str(artifact)
        cfg.nnue_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    elif cfg.nnue_path:
        raise ValueError("nnue_path is only valid with an NNUE-backed eval")
    if (cfg.move_secs < 0 or cfg.max_depth < 1 or cfg.selective_min_depth < 1
            or (cfg.selective_width is not None and cfg.selective_width < 1)
            or (cfg.selective_root_width is not None and cfg.selective_root_width < 1)
            or (cfg.selective_root_width is not None and cfg.selective_width is None)):
        raise ValueError("invalid operational or selective search limit")
    os.makedirs(out_dir, exist_ok=True)
    seeds = list(range(seed_start, seed_start + n_games))
    t0 = time.time()

    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            records = pool.map(_worker, [(s, cfg) for s in seeds], chunksize=8)
    else:
        records = [play_one_game(s, cfg) for s in seeds]

    prov = git_provenance()  # constant for the run; stamped onto every record
    files = {k: open(os.path.join(out_dir, f"{k}.jsonl"), "w") for k in ("train", "val", "test")}
    counts = {"train": 0, "val": 0, "test": 0}
    pos = {"train": 0, "val": 0, "test": 0}
    n_verify_fail = 0
    for rec in records:
        rec.update(prov)
        if verify:
            s0, s1, oc = replay(rec)
            if [s0, s1] != rec["final_scores"] or oc != rec["outcome_p0"]:
                n_verify_fail += 1
        sp = _split_of(rec["seed"], cfg)
        files[sp].write(json.dumps(rec) + "\n")
        counts[sp] += 1
        pos[sp] += rec["n_positions"]
    for f in files.values():
        f.close()

    dt = time.time() - t0
    manifest = {
        "n_games": n_games, "seed_start": seed_start, "counts": counts,
        "positions": pos, "total_positions": sum(pos.values()),
        "config": asdict(cfg), "engine_version": ENGINE_VERSION,
        "format_version": FORMAT_VERSION, "catalog_hash": catalog_hash(),
        "verify_failures": n_verify_fail, "wall_seconds": round(dt, 1),
        "workers": workers, **prov,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--out", default="runs/kingdomino/nnue_data/pilot")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--eval", default="pick_aware")
    ap.add_argument("--nnue-path", default=None)
    ap.add_argument("--move-secs", type=float, default=0.0,
                    help="per-move operational-search budget; 0 uses fixed depth")
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--selective-width", type=int, default=None)
    ap.add_argument("--selective-root-width", type=int, default=None)
    ap.add_argument("--selective-min-depth", type=int, default=4)
    ap.add_argument("--no-verify", action="store_true", help="skip replay verification")
    args = ap.parse_args()

    cfg = GenConfig(eval=args.eval, nnue_path=args.nnue_path,
                    move_secs=args.move_secs, max_depth=args.max_depth,
                    selective_width=args.selective_width,
                    selective_root_width=args.selective_root_width,
                    selective_min_depth=args.selective_min_depth)
    print(f"generating {args.games} games -> {args.out} ({args.workers} workers) ...")
    man = generate(args.games, args.out, cfg, workers=args.workers,
                   seed_start=args.seed_start, verify=not args.no_verify)
    print(json.dumps(man, indent=2))
    if man["verify_failures"]:
        raise SystemExit(f"REPLAY VERIFICATION FAILED on {man['verify_failures']} games")
    print(f"OK: {man['total_positions']:,} positions in {man['wall_seconds']}s "
          f"({man['total_positions'] / max(man['wall_seconds'], 1e-9):.0f} pos/s)")


if __name__ == "__main__":
    main()
