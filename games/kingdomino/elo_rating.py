"""
elo_rating.py — automated Elo rating for Kingdomino AlphaZero checkpoints.

PHASE 1 (minimal viable system): a standalone script that rates any checkpoint
on a fixed anchor ladder using OPEN-LOOP search — the same search the network
was trained with — so the rating matches deployment strength.

ENGINE / ROUTING
────────────────
Network-vs-network games run in a single Rust ``BatchedMCTS`` (open_loop=True)
playing both seats of every game internally.  At each tick we read
``row_search_actors()`` (the SEARCHER/root actor per row — added in lib.rs
specifically for this) and route every leaf to the network of the player whose
move that search is deciding.  This is searcher-owns-network: when it is P0's
turn, P0's net drives the ENTIRE search (it also evaluates P1 nodes), exactly as
in benchmark_vs_rust and in real play.  (``row_actors()`` reports the LEAF actor,
which alternates with depth and is WRONG for two-network rating.)

GREEDYBOT ANCHOR
────────────────
GreedyBot picks moves by a heuristic, not by MCTS visit counts, so it cannot be
driven by BatchedMCTS (feeding it uniform priors would make it a weak uniform
MCTS bot, not GreedyBot).  GreedyBot games therefore run on the SERIAL open-loop
path (round_robin_eval.evaluate_pair with OpenLoopEvalBot for the checkpoint and
GreedyBot for the anchor) — correct GreedyBot behaviour AND the checkpoint's
real open-loop agent.  It is slower; it is one anchor.

Does NOT import evaluation.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import kingdomino_rust

from games.kingdomino.network import KingdominoNet
from games.kingdomino.self_play import make_rust_evaluator
from games.kingdomino.bots import GreedyBot
from games.kingdomino.round_robin_eval import (
    PairResult, GameResult, update_pair,
    checkpoint_state_dict, checkpoint_config, load_checkpoint,
    Participant, OpenLoopEvalBot, evaluate_pair,
)

GREEDY_SENTINEL = "GREEDY_BOT"
LN10_OVER_400 = math.log(10.0) / 400.0


# ─────────────────────────────────────────────────────────────────────────────
# Config / anchor model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EloConfig:
    anchors_csv: str = "games/kingdomino/elo_anchors.csv"
    db_path: str = "elo_db.json"
    games_path: str = "elo_games.jsonl"
    games_per_anchor: int = 32          # paired seeds → 2x this many games per anchor
    sims: int = 400
    device: str = "cuda"
    n_slots: int = 32
    leaf_batch: int = 6
    c_puct: float = 1.5
    fpu: float = -0.2
    margin_gain: float = 2.0
    alpha: float = 0.8
    seed: int = 42
    verbose: bool = False


@dataclass
class AnchorEntry:
    name: str
    path: str                          # checkpoint path, or GREEDY_SENTINEL
    channels: int
    blocks: int
    bilinear_dim: int
    fixed_rating: Optional[float]      # None until bootstrapped
    is_anchor: bool
    # Whether this anchor participates in routine rating games.  Inactive anchors
    # (e.g. greedy_bot) stay in the manifest + db for reference but are never
    # played, eliminating the slow serial GreedyBot leg from rating sessions.
    is_active: bool = True

    @property
    def is_greedy_bot(self) -> bool:
        return self.path == GREEDY_SENTINEL


# ─────────────────────────────────────────────────────────────────────────────
# Anchor manifest I/O
# ─────────────────────────────────────────────────────────────────────────────
def _parse_optional_float(text: str) -> Optional[float]:
    text = (text or "").strip()
    if not text:
        return None
    return float(text)


def _parse_bool(text: str) -> bool:
    return (text or "").strip().lower() in ("1", "true", "yes", "y")


def load_anchors(csv_path: str) -> List[AnchorEntry]:
    """Read the anchor manifest. GREEDY_BOT is kept as a sentinel path; real
    checkpoints store their architecture (channels/blocks/bilinear_dim)."""
    anchors: List[AnchorEntry] = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(row for row in f
                                if row.strip() and not row.lstrip().startswith("#"))
        for row in reader:
            raw_path = (row.get("path") or "").strip()
            # Normalize Windows backslashes to forward slashes (leave the
            # GREEDY_BOT sentinel untouched).  Forward slashes work on both
            # Windows and POSIX; str(Path()) is NOT used here because on POSIX it
            # does not treat '\' as a separator, so a backslash path from a
            # Windows-authored CSV would stay broken on Linux.
            path = (raw_path if raw_path == GREEDY_SENTINEL
                    else raw_path.replace("\\", "/"))
            # Backward compatibility: a missing/blank is_active column (old format)
            # defaults to True, so pre-existing manifests keep all anchors active.
            raw_active = row.get("is_active")
            is_active = (True if raw_active is None or str(raw_active).strip() == ""
                         else _parse_bool(raw_active))
            anchors.append(AnchorEntry(
                name=(row.get("name") or "").strip(),
                path=path,
                channels=int(row.get("channels") or 0),
                blocks=int(row.get("blocks") or 0),
                bilinear_dim=int(row.get("bilinear_dim") or 0),
                fixed_rating=_parse_optional_float(row.get("fixed_rating", "")),
                is_anchor=_parse_bool(row.get("is_anchor", "True")),
                is_active=is_active,
            ))
    return anchors


def write_anchors_csv(csv_path: str, anchors: List[AnchorEntry]) -> None:
    """Rewrite the manifest, persisting newly-assigned fixed_ratings. The path
    column keeps the original sentinel/relative form."""
    fieldnames = ["name", "path", "channels", "blocks", "bilinear_dim",
                  "fixed_rating", "is_anchor", "is_active"]
    out = Path(csv_path)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for a in anchors:
            w.writerow({
                "name": a.name,
                "path": a.path,
                "channels": a.channels,
                "blocks": a.blocks,
                "bilinear_dim": a.bilinear_dim,
                "fixed_rating": "" if a.fixed_rating is None else f"{a.fixed_rating:.2f}",
                "is_anchor": a.is_anchor,
                "is_active": a.is_active,
            })
    tmp.replace(out)


# ─────────────────────────────────────────────────────────────────────────────
# Elo database + game log
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_db_paths(db: dict) -> None:
    """Normalize every stored checkpoint "path" to forward slashes, in place.
    A db written on Windows (backslashes) then read on Linux must still display/
    match consistently; forward slashes are valid on both OSes."""
    for entry in db.get("checkpoints", {}).values():
        p = entry.get("path")
        if isinstance(p, str) and p:
            entry["path"] = p.replace("\\", "/")


def load_db(path: str) -> dict:
    if not os.path.exists(path):
        return {"checkpoints": {}}
    with open(path, "r", encoding="utf-8") as f:
        db = json.load(f)
    db.setdefault("checkpoints", {})
    _normalize_db_paths(db)
    return db


def save_db(db: dict, path: str) -> None:
    """Atomic write (tmp → replace), mirroring ReplayBuffer.save."""
    _normalize_db_paths(db)
    out = Path(path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    tmp.replace(out)


def append_games(games_path: str, records: List[dict]) -> None:
    if not records:
        return
    with open(games_path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_db_entry(db: dict, name: str, *, rating: float, stderr: float,
                    n_games: int, sims: int, path: str,
                    channels: int, blocks: int, bilinear_dim: int,
                    is_anchor: bool, fixed_rating: Optional[float]) -> None:
    db["checkpoints"][name] = {
        "rating": rating,
        "rating_stderr": stderr,
        "n_games": n_games,
        "sims": sims,
        "path": path,
        "channels": channels,
        "blocks": blocks,
        "bilinear_dim": bilinear_dim,
        "timestamp": _now_iso(),
        "is_anchor": is_anchor,
        "fixed_rating": fixed_rating,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Network construction
# ─────────────────────────────────────────────────────────────────────────────
def build_net(path: str, channels: int, blocks: int, bilinear_dim: int,
              device: str) -> KingdominoNet:
    ckpt = load_checkpoint(path, map_location="cpu")
    state = checkpoint_state_dict(ckpt)
    net = KingdominoNet(channels=channels, blocks=blocks, bilinear_dim=bilinear_dim)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


def checkpoint_arch(path: str) -> Tuple[int, int, int]:
    """Architecture (channels, blocks, bilinear_dim) from the checkpoint's stored
    config, falling back to KingdominoNet defaults if absent."""
    ckpt = load_checkpoint(path, map_location="cpu")
    cfg = checkpoint_config(ckpt)
    return (
        int(cfg.get("channels", 96)),
        int(cfg.get("blocks", 8)),
        int(cfg.get("bilinear_dim", 64)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batched two-network game play (the core engine)
# ─────────────────────────────────────────────────────────────────────────────
def _make_batched(n_games: int, seed_start: int, cfg: EloConfig):
    """One open-loop BatchedMCTS for evaluation: no Dirichlet noise (eps=0),
    always greedy move selection (temp_moves=0)."""
    return kingdomino_rust.BatchedMCTS(
        cfg.n_slots,
        int(n_games),
        int(seed_start),
        int(cfg.sims),
        leaf_batch=int(cfg.leaf_batch),
        virtual_loss=1,
        cpuct=float(cfg.c_puct),
        fpu=float(cfg.fpu),
        dirichlet_alpha=0.3,
        dirichlet_eps=0.0,          # evaluation: no root noise
        temp_moves=0,               # evaluation: greedy, no temperature
        open_loop=True,
        margin_gain=float(cfg.margin_gain),
        alpha=float(cfg.alpha),
    )


def _run_batched_orientation(eval_seat0, eval_seat1, n_games: int,
                             seed_start: int, cfg: EloConfig
                             ) -> List[Tuple[int, int, int]]:
    """Play n_games complete open-loop games; seat 0's leaves go to eval_seat0,
    seat 1's to eval_seat1, routed by row_search_actors (searcher-owns-network).
    Returns [(seed, score0, score1)]."""
    batched = _make_batched(n_games, seed_start, cfg)
    results: List[Tuple[int, int, int]] = []
    ticks = 0
    while not batched.done():
        mb, ob, flat, idxs_list = batched.step()
        mb = np.asarray(mb); ob = np.asarray(ob); flat = np.asarray(flat)
        search_actors = np.asarray(batched.row_search_actors(), dtype=np.int64)
        n = mb.shape[0]
        values = np.zeros(n, dtype=np.float32)
        gathered: List[Optional[np.ndarray]] = [None] * n
        for actor_id, evaluator in ((0, eval_seat0), (1, eval_seat1)):
            rows = np.flatnonzero(search_actors == actor_id)
            if rows.size == 0:
                continue
            sub_idxs = [idxs_list[int(r)] for r in rows]
            v, g = evaluator(mb[rows], ob[rows], flat[rows], sub_idxs)
            values[rows] = np.asarray(v, dtype=np.float32)
            for i, r in enumerate(rows):
                gathered[int(r)] = np.asarray(g[i], dtype=np.float32)
        # Every row's searcher is player 0 or 1, so all rows are filled; this
        # guard only protects update() from a None on an unexpected empty row.
        for r in range(n):
            if gathered[r] is None:
                gathered[r] = np.zeros(len(idxs_list[r]), dtype=np.float32)
        for seed, examples, scores in batched.update(values, gathered):
            results.append((int(seed), int(scores[0]), int(scores[1])))
        ticks += 1
        if ticks > 2_000_000:
            raise RuntimeError("Batched rating exceeded tick guard")
    results.sort(key=lambda r: r[0])
    return results


def _winner_by_score(p0: str, p1: str, s0: int, s1: int) -> Optional[str]:
    # The Rust BatchedMCTS path returns only final score totals (no tiebreaker
    # data), so genuine score ties become draws. Rare; worth 0.5 in Elo either way.
    if s0 > s1:
        return p0
    if s1 > s0:
        return p1
    return None


def play_rating_games(net_a, net_b, name_a: str, name_b: str,
                      n_seeds: int, seed_start: int, cfg: EloConfig
                      ) -> Tuple[PairResult, List[GameResult]]:
    """Paired open-loop two-network match (both seats searched by their own net).
    Orientation 0: A is P0; orientation 1: B is P0 — same decks (same base_seed)."""
    assert net_a is not None and net_b is not None, \
        "play_rating_games requires two networks; GreedyBot uses the serial path"
    eval_a = make_rust_evaluator(net_a, device=cfg.device,
                                 margin_gain=cfg.margin_gain, alpha=cfg.alpha)
    eval_b = make_rust_evaluator(net_b, device=cfg.device,
                                 margin_gain=cfg.margin_gain, alpha=cfg.alpha)

    pair = PairResult(a=name_a, b=name_b)
    games: List[GameResult] = []

    # Orientation 0 — A in seat 0, B in seat 1.
    for seed, s0, s1 in _run_batched_orientation(eval_a, eval_b, n_seeds, seed_start, cfg):
        games.append(GameResult(seed, name_a, name_b, s0, s1,
                                _winner_by_score(name_a, name_b, s0, s1), steps=0))
    # Orientation 1 — B in seat 0, A in seat 1 (same decks).
    for seed, s0, s1 in _run_batched_orientation(eval_b, eval_a, n_seeds, seed_start, cfg):
        games.append(GameResult(seed, name_b, name_a, s0, s1,
                                _winner_by_score(name_b, name_a, s0, s1), steps=0))

    for g in games:
        update_pair(pair, g, name_a, name_b)
    return pair, games


def play_rating_games_greedy(ck_net, ck_name: str, n_seeds: int,
                             seed_start: int, cfg: EloConfig
                             ) -> Tuple[PairResult, List[GameResult]]:
    """Serial open-loop match of the checkpoint (OpenLoopEvalBot) vs GreedyBot.
    Uses round_robin_eval.evaluate_pair, so the winner runs through the full
    determine_winner tiebreaker cascade."""
    ck_part = Participant(
        ck_name,
        make_bot=lambda: OpenLoopEvalBot(
            ck_net, device=cfg.device, sims=cfg.sims,
            c_puct=cfg.c_puct, fpu=cfg.fpu, temperature=0.0),
        kind="checkpoint_open_loop",
    )
    greedy_part = Participant("greedy_bot", make_bot=lambda: GreedyBot(), kind="baseline")
    pair, games = evaluate_pair(
        ck_part, greedy_part,
        seed_start=seed_start, seeds_per_pair=n_seeds, verbose=cfg.verbose,
    )
    return pair, games


# ─────────────────────────────────────────────────────────────────────────────
# MLE rating fit
# ─────────────────────────────────────────────────────────────────────────────
def fit_rating_mle(outcomes: List[float],
                   opponent_ratings: List[float]) -> Tuple[float, float]:
    """Maximum-likelihood Elo of one player given per-game outcomes (1.0 win /
    0.5 draw / 0.0 loss) against fixed-rating opponents. Returns (rating, stderr).

    p_i = 1 / (1 + 10^((O_i - R)/400)); R maximizes Σ s·ln p + (1-s)·ln(1-p).
    The NLL is convex in R (one-parameter logistic regression with fixed
    opponents), so a few Newton steps converge globally. stderr from Fisher
    information I(R) = Σ p(1-p)·(ln10/400)^2.
    """
    outcomes = np.asarray(outcomes, dtype=np.float64)
    opp = np.asarray(opponent_ratings, dtype=np.float64)
    n = outcomes.size
    if n == 0:
        return 0.0, 999.0

    mean = float(outcomes.mean())
    # Degenerate likelihoods (R → ±∞): clamp ±800 from the opponent band, flag
    # with a large stderr so the leaderboard shows the estimate is unreliable.
    if mean >= 1.0 - 1e-12:
        return float(opp.max() + 800.0), 999.0
    if mean <= 1e-12:
        return float(opp.min() - 800.0), 999.0

    c = LN10_OVER_400
    R = float(opp.mean())   # warm start at the opponent band centre
    for _ in range(100):
        # p_i = σ(c·(R - O_i)); score Σ(s-p), info I = c²·Σ p(1-p).
        p = 1.0 / (1.0 + np.power(10.0, (opp - R) / 400.0))
        grad = c * float(np.sum(outcomes - p))
        info = (c ** 2) * float(np.sum(p * (1.0 - p)))
        if info <= 0:
            break
        step = grad / info
        R += step
        R = float(np.clip(R, -1000.0, 2000.0))
        if abs(step) < 1e-6:
            break

    p = 1.0 / (1.0 + np.power(10.0, (opp - R) / 400.0))
    info = float(np.sum(p * (1.0 - p)) * (c ** 2))
    stderr = (1.0 / math.sqrt(info)) if info > 0 else 999.0
    return R, stderr


def _subject_outcome(game: GameResult, subject: str) -> float:
    if game.winner is None:
        return 0.5
    return 1.0 if game.winner == subject else 0.0


def _game_record(game: GameResult, checkpoint: str, opponent: str,
                 sims: int, engine: str, cfg: EloConfig) -> dict:
    """One elo_games.jsonl row, from the checkpoint's perspective. The search
    params (alpha/c_puct/margin_gain) are recorded so --resolve can distinguish
    the same checkpoint rated under different sweep settings."""
    if game.p0 == checkpoint:
        orientation, score_ck, score_opp = 0, game.score0, game.score1
    else:
        orientation, score_ck, score_opp = 1, game.score1, game.score0
    return {
        "checkpoint": checkpoint,
        "opponent": opponent,
        "seed": int(game.seed),
        "orientation": orientation,
        "score_checkpoint": int(score_ck),
        "score_opponent": int(score_opp),
        "winner": game.winner if game.winner is not None else "DRAW",
        "sims": sims,
        "engine": engine,
        "routing": "searcher",
        "alpha": cfg.alpha,
        "c_puct": cfg.c_puct,
        "margin_gain": cfg.margin_gain,
        "timestamp": _now_iso(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rating a single checkpoint against the anchor pool
# ─────────────────────────────────────────────────────────────────────────────
def rate_checkpoint(checkpoint_path: str, checkpoint_name: str,
                    cfg: EloConfig, anchors: List[AnchorEntry]
                    ) -> Tuple[float, float, int]:
    """Play the checkpoint against every fixed anchor, append the game log, and
    fit its Elo via MLE. Returns (rating, stderr, n_games)."""
    ch, bl, bd = checkpoint_arch(checkpoint_path)
    ck_net = build_net(checkpoint_path, ch, bl, bd, cfg.device)

    outcomes: List[float] = []
    opp_ratings: List[float] = []
    records: List[dict] = []
    t0 = time.time()
    total_games = 0

    for ai, anchor in enumerate(anchors):
        if anchor.name == checkpoint_name:
            continue
        if not anchor.is_active:
            continue   # inactive anchors (e.g. greedy_bot) are never played
        if anchor.fixed_rating is None:
            raise ValueError(
                f"Anchor {anchor.name!r} has no fixed_rating; bootstrap first.")
        # Disjoint seed band per anchor so anchors never reuse decks.
        seed_start = cfg.seed + ai * 100_000

        if anchor.is_greedy_bot:
            engine = "py_open_loop_vs_greedy"
            pair, games = play_rating_games_greedy(
                ck_net, checkpoint_name, cfg.games_per_anchor, seed_start, cfg)
        else:
            engine = "batched_open_loop"
            anc_net = build_net(anchor.path, anchor.channels, anchor.blocks,
                                anchor.bilinear_dim, cfg.device)
            pair, games = play_rating_games(
                ck_net, anc_net, checkpoint_name, anchor.name,
                cfg.games_per_anchor, seed_start, cfg)

        for g in games:
            outcomes.append(_subject_outcome(g, checkpoint_name))
            opp_ratings.append(anchor.fixed_rating)
            records.append(_game_record(g, checkpoint_name, anchor.name,
                                         cfg.sims, engine, cfg))
        total_games += len(games)

        if cfg.verbose:
            wr = pair.a_win_rate
            print(f"  vs {anchor.name} (Elo {anchor.fixed_rating:.0f}): "
                  f"{pair.a_wins}-{pair.b_wins}-{pair.draws} "
                  f"win_rate={wr:.1%} avg_margin={pair.avg_margin_a:+.2f} "
                  f"[{engine}]", flush=True)

    append_games(cfg.games_path, records)
    rating, stderr = fit_rating_mle(outcomes, opp_ratings)

    if cfg.verbose:
        elapsed = time.time() - t0
        gps = total_games / elapsed if elapsed > 0 else 0.0
        print(f"  rated in {elapsed:.1f}s ({gps:.3f} games/s, {total_games} games)",
              flush=True)
    return rating, stderr, total_games


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap: assign fixed ratings to anchors that don't have one yet
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_anchors(cfg: EloConfig, anchors: List[AnchorEntry], db: dict) -> bool:
    """Assign fixed_ratings to non-greedy anchors lacking one, each rated against
    the anchors already fixed (in manifest order). Persists to db + CSV.
    Returns True if any anchor was bootstrapped."""
    changed = False
    for anchor in anchors:
        if anchor.fixed_rating is not None:
            # Already fixed (e.g. GreedyBot at 0.0); ensure it is in the db.
            if anchor.name not in db["checkpoints"]:
                update_db_entry(
                    db, anchor.name, rating=anchor.fixed_rating, stderr=0.0,
                    n_games=0, sims=cfg.sims, path=anchor.path,
                    channels=anchor.channels, blocks=anchor.blocks,
                    bilinear_dim=anchor.bilinear_dim, is_anchor=True,
                    fixed_rating=anchor.fixed_rating)
                changed = True
            continue

        if not anchor.is_active:
            continue   # inactive anchors without a rating are never played

        # Seed/opponent pool excludes inactive anchors, so the slow serial
        # GreedyBot leg never enters a bootstrap.
        fixed_so_far = [a for a in anchors
                        if a.fixed_rating is not None and a.is_active]
        if not fixed_so_far:
            raise ValueError(
                "Cannot bootstrap: no ACTIVE anchor has a fixed_rating "
                "(seed one active anchor, or use --reanchor).")
        print(f"Bootstrapping anchor {anchor.name} vs "
              f"{[a.name for a in fixed_so_far]} ...", flush=True)
        rating, stderr, n_games = rate_checkpoint(
            anchor.path, anchor.name, cfg, fixed_so_far)
        anchor.fixed_rating = round(rating, 2)
        update_db_entry(
            db, anchor.name, rating=anchor.fixed_rating, stderr=stderr,
            n_games=n_games, sims=cfg.sims, path=anchor.path,
            channels=anchor.channels, blocks=anchor.blocks,
            bilinear_dim=anchor.bilinear_dim, is_anchor=True,
            fixed_rating=anchor.fixed_rating)
        save_db(db, cfg.db_path)
        write_anchors_csv(cfg.anchors_csv, anchors)
        print(f"  -> {anchor.name}: Elo {anchor.fixed_rating:.0f} +/- {stderr:.0f} "
              f"({n_games} games)", flush=True)
        changed = True
    return changed


def reanchor(cfg: EloConfig, anchors: List[AnchorEntry], db: dict) -> None:
    """Re-bootstrap the entire ACTIVE anchor pool, automating the manual two-pass
    that works around the "all-empty has no seed" crash.

      Pass 1: pick the highest-rated active anchor as a temporary FIXED scale
              (or the first alphabetically at 0.0 if none are rated), clear every
              other active anchor, and bootstrap them against it.
      Pass 2: clear the scale anchor and re-rate it against the now-fixed others,
              so all active anchors end up mutually consistent on one scale.

    Inactive anchors (e.g. greedy_bot) are untouched and never played.
    """
    active = [a for a in anchors if a.is_active]
    if len(active) < 2:
        print("[reanchor] need at least 2 active anchors; nothing to do.",
              flush=True)
        return

    rated = [a for a in active if a.fixed_rating is not None]
    if rated:
        scale = max(rated, key=lambda a: a.fixed_rating)
    else:
        scale = sorted(active, key=lambda a: a.name)[0]
        scale.fixed_rating = 0.0
    print(f"[reanchor] scale anchor: {scale.name} fixed at "
          f"{scale.fixed_rating:.1f}", flush=True)

    # Pass 1: clear every active anchor except the scale, bootstrap vs it.
    for a in active:
        if a is not scale:
            a.fixed_rating = None
    print("[reanchor] pass 1: bootstrapping active anchors vs the scale ...",
          flush=True)
    bootstrap_anchors(cfg, anchors, db)

    # Pass 2: clear the scale and re-rate it against the now-fixed others.
    scale.fixed_rating = None
    print(f"[reanchor] pass 2: re-rating {scale.name} vs the rest ...",
          flush=True)
    bootstrap_anchors(cfg, anchors, db)

    save_db(db, cfg.db_path)
    write_anchors_csv(cfg.anchors_csv, anchors)
    print_leaderboard(db)


# ─────────────────────────────────────────────────────────────────────────────
# Ordo-style global re-solve (Bradley-Terry MLE over the full game log)
# ─────────────────────────────────────────────────────────────────────────────
def _solve_bradley_terry(
    games: List[Tuple[str, str, float]],   # (a, b, outcome_for_a)
    fixed: dict,
    n_iter: int = 200,
    tol: float = 1e-8,
) -> dict:
    """Joint MLE of all FREE players' ratings via Newton's method (diagonal
    Hessian, Ordo-style). `fixed` maps name -> held-constant rating. Players in
    the log but absent from `fixed` are free and initialised at 0.0. Returns
    {name: (rating, stderr)} for the free players only."""
    k = math.log(10.0) / 400.0

    players = sorted(set(
        n for a, b, _ in games for n in (a, b) if n not in fixed
    ))
    if not players:
        return {}
    idx = {name: i for i, name in enumerate(players)}
    n = len(players)
    R = np.zeros(n)

    for _ in range(n_iter):
        grad = np.zeros(n)
        hess_diag = np.zeros(n)
        for name_a, name_b, s in games:
            r_a = fixed.get(name_a, R[idx[name_a]] if name_a in idx else 0.0)
            r_b = fixed.get(name_b, R[idx[name_b]] if name_b in idx else 0.0)
            p = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
            if name_a in idx:
                i = idx[name_a]
                grad[i] += k * (s - p)
                hess_diag[i] -= k * k * p * (1.0 - p)
            if name_b in idx:
                j = idx[name_b]
                grad[j] += k * ((1.0 - s) - (1.0 - p))
                hess_diag[j] -= k * k * p * (1.0 - p)
        hess_diag = np.where(np.abs(hess_diag) > 1e-9, hess_diag, -1e-9)
        step = -grad / hess_diag
        # Damp the Newton step and clamp R to a sane band.  Undamped Newton on
        # near-separable logistic data (a player that wins/loses almost all its
        # games) overshoots into the saturated region where p->0/1 and the
        # diagonal Hessian vanishes, sending ratings to +/-1e10.  Capping the
        # per-iteration step keeps it stable; genuinely all-win/all-loss players
        # settle at the band edge and get a large (unreliable) stderr.
        step = np.clip(step, -200.0, 200.0)
        R = np.clip(R + step, -2000.0, 3000.0)
        if np.max(np.abs(step)) < tol:
            break

    results = {}
    for name in players:
        i = idx[name]
        h = 0.0
        for name_a, name_b, _ in games:
            if name != name_a and name != name_b:
                continue
            r_a = fixed.get(name_a, R[idx[name_a]] if name_a in idx else 0.0)
            r_b = fixed.get(name_b, R[idx[name_b]] if name_b in idx else 0.0)
            p = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
            h -= k * k * p * (1.0 - p)
        stderr = 1.0 / math.sqrt(-h) if h < -1e-12 else 999.0
        results[name] = (float(R[i]), stderr)
    return results


def resolve_ladder(games_path: str, db_path: str,
                   fixed: Optional[dict] = None,
                   verbose: bool = False) -> dict:
    """Re-solve all free ratings from the full game log with one joint
    Bradley-Terry fit, holding the anchor pool fixed. Updates elo_db.json in
    place (free entries only). Returns {name: (rating, stderr)}."""
    if not os.path.exists(games_path):
        print(f"[resolve] no game log at {games_path}; nothing to do.")
        return {}
    rows = []
    with open(games_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("[resolve] game log is empty; nothing to do.")
        return {}

    games: List[Tuple[str, str, float]] = []
    for r in rows:
        a = r.get("checkpoint")
        b = r.get("opponent")
        if a is None or b is None:
            continue
        w = r.get("winner", "DRAW")
        s = 1.0 if w == a else (0.0 if w == b else 0.5)
        games.append((a, b, s))

    db = load_db(db_path)
    if fixed is None:
        # Fixed points: every is_anchor entry with a finite fixed_rating, plus
        # greedy_bot=0.0 if present (the conventional zero-point floor).
        fixed = {}
        for name, e in db.get("checkpoints", {}).items():
            if e.get("is_anchor") and e.get("fixed_rating") is not None:
                fixed[name] = float(e["fixed_rating"])
        g = db.get("checkpoints", {}).get("greedy_bot")
        if g is not None and float(g.get("rating", 0.0)) == 0.0:
            fixed["greedy_bot"] = 0.0

    results = _solve_bradley_terry(games, fixed)

    # Count games per free player for the db entry.
    counts: dict = {}
    for a, b, _ in games:
        for nm in (a, b):
            if nm not in fixed:
                counts[nm] = counts.get(nm, 0) + 1

    for name, (rating, stderr) in results.items():
        if name in db["checkpoints"]:
            # Don't touch fixed-anchor entries (they are not in `results`).
            db["checkpoints"][name]["rating"] = round(rating, 2)
            db["checkpoints"][name]["rating_stderr"] = round(stderr, 2)
        else:
            # Player seen in the log but absent from the db — add a fresh entry.
            update_db_entry(
                db, name, rating=round(rating, 2), stderr=round(stderr, 2),
                n_games=counts.get(name, 0), sims=0, path="",
                channels=0, blocks=0, bilinear_dim=0,
                is_anchor=False, fixed_rating=None)
    save_db(db, db_path)
    if verbose:
        print(f"[resolve] solved {len(results)} free rating(s) vs "
              f"{len(fixed)} fixed anchor(s) from {len(games)} games.")
    print_leaderboard(db)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Leaderboard
# ─────────────────────────────────────────────────────────────────────────────
def print_leaderboard(db: dict) -> None:
    entries = list(db.get("checkpoints", {}).items())
    if not entries:
        print("(no rated checkpoints yet)")
        return
    entries.sort(key=lambda kv: kv[1].get("rating", 0.0), reverse=True)
    print("\nElo Leaderboard")
    print("-" * 92)
    print(f"{'#':>2}  {'name':<26} {'rating':>8} {'+/-err':>7} {'games':>6} "
          f"{'sims':>5}  {'path'}")
    print("-" * 92)
    for rank, (name, e) in enumerate(entries, 1):
        tag = " [A]" if e.get("is_anchor") else ""
        path = str(e.get("path", ""))
        if len(path) > 34:
            path = "..." + path[-31:]
        print(f"{rank:>2}  {name + tag:<26} {e.get('rating', 0.0):>8.1f} "
              f"{e.get('rating_stderr', 0.0):>7.1f} {e.get('n_games', 0):>6} "
              f"{e.get('sims', 0):>5}  {path}")
    print("-" * 92)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _cfg_from_args(args: argparse.Namespace) -> EloConfig:
    return EloConfig(
        anchors_csv=args.anchors,
        db_path=args.db,
        games_path=args.games_log,
        games_per_anchor=args.games_per_anchor,
        sims=args.sims,
        device=args.device,
        n_slots=args.n_slots,
        alpha=args.alpha,
        margin_gain=args.margin_gain,
        c_puct=args.c_puct,
        verbose=args.verbose,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Kingdomino automated Elo rating (Phase 1)")
    p.add_argument("--checkpoint", default=None, help="Checkpoint .pt to rate.")
    p.add_argument("--name", default=None, help="Display name (default: file stem).")
    p.add_argument("--anchors", default="games/kingdomino/elo_anchors.csv")
    p.add_argument("--db", default="elo_db.json")
    p.add_argument("--games_log", default="elo_games.jsonl")
    p.add_argument("--games_per_anchor", type=int, default=32)
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_slots", type=int, default=32)
    p.add_argument("--alpha", type=float, default=EloConfig.alpha,
                   help="Weight on margin vs win-probability in the leaf value "
                        "formula (1.0=pure margin, 0.0=pure win prob).")
    p.add_argument("--margin_gain", type=float, default=EloConfig.margin_gain,
                   help="Scaling factor for tanh in the leaf value formula.")
    p.add_argument("--c_puct", type=float, default=EloConfig.c_puct,
                   help="PUCT exploration constant for rating games.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--leaderboard", action="store_true",
                   help="Print the leaderboard and exit (no rating).")
    p.add_argument("--bootstrap", action="store_true",
                   help="Assign fixed_ratings to anchors missing one.")
    p.add_argument("--reanchor", action="store_true",
                   help="Re-bootstrap the whole active anchor pool (two-pass: "
                        "seed the strongest, rate the rest, then re-rate the seed).")
    p.add_argument("--resolve", action="store_true",
                   help="Re-solve all free ratings from the full game log with a "
                        "joint Bradley-Terry fit (no games played; milliseconds).")
    args = p.parse_args()

    cfg = _cfg_from_args(args)

    if args.leaderboard:
        print_leaderboard(load_db(cfg.db_path))
        return

    if args.resolve:
        resolve_ladder(cfg.games_path, cfg.db_path, verbose=args.verbose)
        return

    anchors = load_anchors(cfg.anchors_csv)
    db = load_db(cfg.db_path)

    if args.reanchor:
        reanchor(cfg, anchors, db)
        return

    # Bootstrap when explicitly requested OR when rating needs anchors that are
    # not yet fixed (zero manual intervention).
    needs_bootstrap = any(a.fixed_rating is None for a in anchors)
    if args.bootstrap or (args.checkpoint and needs_bootstrap):
        if bootstrap_anchors(cfg, anchors, db):
            save_db(db, cfg.db_path)
            write_anchors_csv(cfg.anchors_csv, anchors)
        print_leaderboard(db)

    if args.checkpoint:
        name = args.name or Path(args.checkpoint).stem
        print(f"\nRating {name} ({args.checkpoint}) vs "
              f"{len([a for a in anchors if a.name != name])} anchors "
              f"@ sims={cfg.sims}, {cfg.games_per_anchor} paired seeds/anchor ...",
              flush=True)
        rating, stderr, n_games = rate_checkpoint(
            args.checkpoint, name, cfg, anchors)
        ch, bl, bd = checkpoint_arch(args.checkpoint)
        update_db_entry(
            db, name, rating=round(rating, 2), stderr=round(stderr, 2),
            n_games=n_games, sims=cfg.sims, path=str(Path(args.checkpoint)),
            channels=ch, blocks=bl, bilinear_dim=bd, is_anchor=False,
            fixed_rating=None)
        save_db(db, cfg.db_path)
        n_anchors = len([a for a in anchors if a.name != name])
        print(f"\n{name}: Elo {rating:.0f} +/- {stderr:.0f} "
              f"({n_games} games vs {n_anchors} anchors)")
        print_leaderboard(db)
    elif not args.bootstrap:
        p.error("nothing to do: pass --checkpoint, --bootstrap, or --leaderboard")


if __name__ == "__main__":
    main()
