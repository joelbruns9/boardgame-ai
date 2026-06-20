"""
round_robin_eval.py — paired round-robin evaluation for Kingdomino agents.

Evaluates checkpoints and baselines with paired deck seeds:
  A as P0 vs B as P1 on seed S
  B as P0 vs A as P1 on the same seed S

This reduces deck/start-player variance and makes checkpoint comparisons much
more reliable than one-sided matches.

Does NOT import evaluation.py.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from games.kingdomino.bots import GreedyBot, RandomBot
from games.kingdomino.game import (
    GameState, Phase, determine_winner, PickAction, TurnAction,
)
from games.kingdomino.mcts_az import (
    AlphaZeroMCTS,
    make_serial_evaluator,
    run_pimc,
    select_move,
    run_pimc_open_loop,
)
from games.kingdomino.network import KingdominoNet


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def score_total(board) -> int:
    """Return total score for one Board across current ScoreBreakdown variants."""
    score = board.score()
    if hasattr(score, "total"):
        return int(score.total)
    return int(score.territory_score + score.harmony_bonus + score.middle_kingdom_bonus)


def checkpoint_state_dict(ckpt) -> dict:
    """Accept either a raw state_dict or a checkpoint dict containing model_state."""
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")


def checkpoint_config(ckpt) -> dict:
    if isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
        return dict(ckpt["config"])
    return {}


def load_checkpoint(path: str | os.PathLike, map_location: str = "cpu"):
    return torch.load(path, map_location=map_location)


def safe_name_from_path(path: str | os.PathLike) -> str:
    p = Path(path)
    return p.stem


# ─────────────────────────────────────────────────────────────────────────────
# Player wrappers
# ─────────────────────────────────────────────────────────────────────────────

class BotProtocol:
    def choose_action(self, state: GameState, actions=None, rng: Optional[random.Random] = None):
        raise NotImplementedError


class AlphaZeroEvalBot(BotProtocol):
    """Checkpoint-backed AlphaZero player for evaluation.

    Uses PIMC/redeterminization and selects greedily from root visit counts by
    default. No Dirichlet noise is used during evaluation.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        *,
        device: str,
        sims: int,
        determinizations: int = 1,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        fpu: float = 0.0,
        temperature: float = 0.0,
    ) -> None:
        evaluator = make_serial_evaluator(net, device=device)
        self.mcts = AlphaZeroMCTS(
            evaluator,
            c_puct=c_puct,
            n_simulations=sims,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            fpu=fpu,
        )
        self.determinizations = max(1, int(determinizations))
        self.temperature = float(temperature)

    def choose_action(self, state: GameState, actions=None, rng: Optional[random.Random] = None):
        legal = actions if actions is not None else state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        py_rng = rng or random.Random()
        # Derive a NumPy generator from the per-game Python RNG so action sampling
        # and Dirichlet machinery remain deterministic for a given game seed.
        np_rng = np.random.default_rng(py_rng.randrange(0, 2**63 - 1))
        visit_counts, _root_value0 = run_pimc(
            self.mcts,
            state,
            py_rng,
            n_determinizations=self.determinizations,
            add_noise=False,
            np_rng=np_rng,
        )
        return select_move(visit_counts, temperature=self.temperature, rng=np_rng)


class OpenLoopEvalBot(BotProtocol):
    """Open-loop AlphaZero player for evaluation.

    Uses OpenLoopMCTS (resamples deck order per simulation). No Dirichlet
    noise during evaluation. n_determinizations is not a parameter —
    open-loop averages internally.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        *,
        device: str,
        sims: int,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        fpu: float = 0.0,
        temperature: float = 0.0,
    ) -> None:
        from games.kingdomino.mcts_az import OpenLoopMCTS
        evaluator = make_serial_evaluator(net, device=device)
        self.mcts = OpenLoopMCTS(
            evaluator,
            c_puct=c_puct,
            n_simulations=sims,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            fpu=fpu,
        )
        self.temperature = float(temperature)

    def choose_action(
        self, state: GameState, actions=None,
        rng: Optional[random.Random] = None,
    ):
        legal = actions if actions is not None else state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        py_rng = rng or random.Random()
        np_rng = np.random.default_rng(py_rng.randrange(0, 2**63 - 1))
        visit_counts, _ = run_pimc_open_loop(
            self.mcts, state, add_noise=False, rng=np_rng,
        )
        return select_move(visit_counts, temperature=self.temperature,
                           rng=np_rng)


@dataclass
class Participant:
    name: str
    make_bot: Callable[[], BotProtocol]
    kind: str
    source: str = ""
    model_kwargs: Optional[Dict[str, int]] = None
    state_dict: Optional[Dict[str, Any]] = None


@dataclass
class ManifestEntry:
    name: Optional[str]
    path: str
    channels: Optional[int] = None
    blocks: Optional[int] = None
    bilinear_dim: Optional[int] = None


@dataclass
class GameResult:
    seed: int
    p0: str
    p1: str
    score0: int
    score1: int
    winner: Optional[str]
    steps: int
    # Degenerate-play diagnostics (computed in play_game; defaults are the
    # non-degenerate values so paths that don't set them never false-flag).
    discard_rate: float = 0.0       # fraction of placement decisions that discarded
    pick_slot_entropy: float = 1.0  # entropy (nats) of pick-slot choices; 0 = always one slot


@dataclass
class PairResult:
    a: str
    b: str
    games: int = 0
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    a_score_sum: int = 0
    b_score_sum: int = 0
    seconds: float = 0.0

    @property
    def a_points(self) -> float:
        return self.a_wins + 0.5 * self.draws

    @property
    def b_points(self) -> float:
        return self.b_wins + 0.5 * self.draws

    @property
    def a_win_rate(self) -> float:
        return self.a_points / self.games if self.games else 0.0

    @property
    def avg_a_score(self) -> float:
        return self.a_score_sum / self.games if self.games else 0.0

    @property
    def avg_b_score(self) -> float:
        return self.b_score_sum / self.games if self.games else 0.0

    @property
    def avg_margin_a(self) -> float:
        return (self.a_score_sum - self.b_score_sum) / self.games if self.games else 0.0


@dataclass
class Standing:
    name: str
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    score_for: int = 0
    score_against: int = 0

    @property
    def points(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def win_rate(self) -> float:
        return self.points / self.games if self.games else 0.0

    @property
    def avg_margin(self) -> float:
        return (self.score_for - self.score_against) / self.games if self.games else 0.0

    @property
    def avg_score(self) -> float:
        return self.score_for / self.games if self.games else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Participant construction
# ─────────────────────────────────────────────────────────────────────────────

def build_checkpoint_participant(
    path: str,
    *,
    name: Optional[str],
    device: str,
    sims: int,
    determinizations: int,
    c_puct: float,
    temperature: float,
    channels_override: Optional[int],
    blocks_override: Optional[int],
    bilinear_dim_override: Optional[int],
) -> Participant:
    ckpt = load_checkpoint(path, map_location="cpu")
    cfg = checkpoint_config(ckpt)
    model_kwargs = {
        "channels": int(channels_override if channels_override is not None else cfg.get("channels", 96)),
        "blocks": int(blocks_override if blocks_override is not None else cfg.get("blocks", 8)),
        "bilinear_dim": int(
            bilinear_dim_override if bilinear_dim_override is not None else cfg.get("bilinear_dim", 64)
        ),
    }
    state = checkpoint_state_dict(ckpt)

    # Load once now to fail fast on config mismatch. The bot factory creates a
    # fresh net per participant instance so repeated pairings do not share state.
    probe = KingdominoNet(**model_kwargs)
    probe.load_state_dict(state)
    del probe

    participant_name = name or safe_name_from_path(path)

    def make_bot() -> AlphaZeroEvalBot:
        net = KingdominoNet(**model_kwargs)
        net.load_state_dict(state)
        net.eval()
        return AlphaZeroEvalBot(
            net,
            device=device,
            sims=sims,
            determinizations=determinizations,
            c_puct=c_puct,
            temperature=temperature,
        )

    return Participant(
        name=participant_name,
        make_bot=make_bot,
        kind="checkpoint",
        source=str(path),
        model_kwargs=model_kwargs,
        state_dict=state,
    )


def build_open_loop_checkpoint_participant(
    path: str,
    *,
    name: Optional[str],
    device: str,
    sims: int,
    c_puct: float,
    temperature: float,
    channels_override: Optional[int],
    blocks_override: Optional[int],
    bilinear_dim_override: Optional[int],
) -> Participant:
    """Like build_checkpoint_participant but uses OpenLoopEvalBot."""
    ckpt = load_checkpoint(path, map_location="cpu")
    cfg = checkpoint_config(ckpt)
    model_kwargs = {
        "channels": int(channels_override if channels_override is not None else cfg.get("channels", 96)),
        "blocks": int(blocks_override if blocks_override is not None else cfg.get("blocks", 8)),
        "bilinear_dim": int(
            bilinear_dim_override if bilinear_dim_override is not None else cfg.get("bilinear_dim", 64)
        ),
    }
    state = checkpoint_state_dict(ckpt)
    probe = KingdominoNet(**model_kwargs)
    probe.load_state_dict(state)
    del probe

    participant_name = name or safe_name_from_path(path)

    def make_bot() -> OpenLoopEvalBot:
        net = KingdominoNet(**model_kwargs)
        net.load_state_dict(state)
        net.eval()
        return OpenLoopEvalBot(
            net, device=device, sims=sims,
            c_puct=c_puct, temperature=temperature,
        )

    return Participant(
        name=participant_name,
        make_bot=make_bot,
        kind="checkpoint_open_loop",
        source=str(path),
        model_kwargs=model_kwargs,
        state_dict=state,
    )


def optional_int(value: object, field_name: str) -> Optional[int]:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for manifest field {field_name!r}: {value!r}") from exc


def resolve_manifest_path(raw_path: str, manifest_path: Path) -> str:
    path = Path(raw_path.strip())
    if path.is_absolute():
        return str(path)

    # Prefer paths relative to the manifest, but keep repo-root manifests with
    # cwd-relative paths ergonomic.
    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return str(manifest_relative)
    return str(path)


def load_manifest_entries(path: str | os.PathLike) -> List[ManifestEntry]:
    manifest_path = Path(path)
    entries: List[ManifestEntry] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(row for row in f if row.strip() and not row.lstrip().startswith("#"))
        if reader.fieldnames is None:
            raise ValueError(f"Empty manifest: {path}")
        required = {"path"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest {path} is missing required columns: {sorted(missing)}")

        for row_idx, row in enumerate(reader, 2):
            raw_path = (row.get("path") or "").strip()
            if not raw_path:
                raise ValueError(f"Manifest {path} row {row_idx} has an empty path")
            name = (row.get("name") or "").strip() or None
            entries.append(ManifestEntry(
                name=name,
                path=resolve_manifest_path(raw_path, manifest_path),
                channels=optional_int(row.get("channels"), "channels"),
                blocks=optional_int(row.get("blocks"), "blocks"),
                bilinear_dim=optional_int(row.get("bilinear_dim"), "bilinear_dim"),
            ))
    return entries


def build_participants(args: argparse.Namespace) -> List[Participant]:
    participants: List[Participant] = []

    open_loop = bool(getattr(args, "open_loop", False))

    for entry in load_manifest_entries(args.manifest) if args.manifest else []:
        if open_loop:
            participants.append(build_open_loop_checkpoint_participant(
                entry.path,
                name=entry.name,
                device=args.device,
                sims=args.sims,
                c_puct=args.c_puct,
                temperature=args.temperature,
                channels_override=entry.channels,
                blocks_override=entry.blocks,
                bilinear_dim_override=entry.bilinear_dim,
            ))
        else:
            participants.append(build_checkpoint_participant(
                entry.path,
                name=entry.name,
                device=args.device,
                sims=args.sims,
                determinizations=args.determinizations,
                c_puct=args.c_puct,
                temperature=args.temperature,
                channels_override=entry.channels,
                blocks_override=entry.blocks,
                bilinear_dim_override=entry.bilinear_dim,
            ))

    names: List[Optional[str]] = []
    if args.names:
        names = list(args.names)
        if len(names) != len(args.checkpoints):
            raise ValueError("--names must have the same number of entries as --checkpoints")
    else:
        names = [None] * len(args.checkpoints)

    for path, name in zip(args.checkpoints, names):
        if open_loop:
            participants.append(build_open_loop_checkpoint_participant(
                path,
                name=name,
                device=args.device,
                sims=args.sims,
                c_puct=args.c_puct,
                temperature=args.temperature,
                channels_override=args.channels,
                blocks_override=args.blocks,
                bilinear_dim_override=args.bilinear_dim,
            ))
        else:
            participants.append(build_checkpoint_participant(
                path,
                name=name,
                device=args.device,
                sims=args.sims,
                determinizations=args.determinizations,
                c_puct=args.c_puct,
                temperature=args.temperature,
                channels_override=args.channels,
                blocks_override=args.blocks,
                bilinear_dim_override=args.bilinear_dim,
            ))

    if args.include_greedy:
        participants.append(Participant("Greedy", lambda: GreedyBot(), kind="baseline"))
    if args.include_random:
        participants.append(Participant("Random", lambda: RandomBot(), kind="baseline"))

    if len(participants) < 2:
        raise ValueError("Need at least two participants. Add checkpoints and/or --include_greedy/--include_random.")

    # Prevent confusing duplicate names in CSV/standings.
    seen: Dict[str, int] = {}
    for p in participants:
        if p.name not in seen:
            seen[p.name] = 1
        else:
            seen[p.name] += 1
            p.name = f"{p.name}_{seen[p.name]}"
    return participants


# ─────────────────────────────────────────────────────────────────────────────
# Game / match logic
# ─────────────────────────────────────────────────────────────────────────────

def play_game(
    p0_name: str,
    p0_bot: BotProtocol,
    p1_name: str,
    p1_bot: BotProtocol,
    *,
    seed: int,
) -> GameResult:
    state = GameState.new(seed=seed)
    # One per-game RNG object is passed to bots for tie breaks, PIMC shuffles, and
    # temperature sampling. This keeps a game deterministic for a given seed.
    rng = random.Random(seed * 1_000_003 + 17)
    bots = [p0_bot, p1_bot]

    # Degenerate-play tracking (computed before each step, while current_row is
    # still the pre-step row, so a pick's slot index is meaningful).
    n_placements = 0     # TurnActions (PLACE_AND_SELECT + FINAL_PLACEMENT)
    n_discards = 0       # TurnActions with placement=None
    pick_slot_counts: Dict[int, int] = {}

    while state.phase != Phase.GAME_OVER:
        actor = state.current_actor
        actions = state.legal_actions()
        action = bots[actor].choose_action(state, actions, rng=rng)

        # ── degenerate diagnostics ──
        if isinstance(action, TurnAction):
            n_placements += 1
            if action.placement is None:
                n_discards += 1
            picked = action.pick_domino_id          # None in FINAL_PLACEMENT
        elif isinstance(action, PickAction):
            picked = action.domino_id               # INITIAL_SELECTION
        else:
            picked = None
        if picked is not None and picked in state.current_row:
            slot = state.current_row.index(picked)
            pick_slot_counts[slot] = pick_slot_counts.get(slot, 0) + 1

        state = state.step(action)

    score0 = score_total(state.boards[0])
    score1 = score_total(state.boards[1])
    # Authoritative cascade (score -> largest territory -> crowns -> draw), not a
    # raw score comparison: score ties are resolved by the tiebreakers. This is
    # the single point where the winner is decided; update_standings/update_pair
    # consume GameResult.winner rather than re-comparing scores.
    win_idx = determine_winner(state)
    winner = None if win_idx is None else (p0_name, p1_name)[win_idx]

    # discard_rate over placement decisions; pick-slot entropy (nats) over the
    # slots actually chosen.  always-discard ⇒ discard_rate→1; always-first-tile
    # ⇒ entropy 0.
    discard_rate = (n_discards / n_placements) if n_placements else 0.0
    total_picks = sum(pick_slot_counts.values())
    if total_picks > 0:
        pick_slot_entropy = -sum(
            (c / total_picks) * math.log(c / total_picks)
            for c in pick_slot_counts.values()
        )
    else:
        pick_slot_entropy = 1.0   # no picks (shouldn't happen) → non-degenerate
    if discard_rate > 0.8 or pick_slot_entropy < 0.1:
        print(f"    [WARN] degenerate game (seed={seed}, {p0_name} vs {p1_name}): "
              f"discard_rate={discard_rate:.2f} pick_slot_entropy={pick_slot_entropy:.3f}",
              flush=True)

    return GameResult(
        seed=seed,
        p0=p0_name,
        p1=p1_name,
        score0=score0,
        score1=score1,
        winner=winner,
        steps=len(state.history),
        discard_rate=discard_rate,
        pick_slot_entropy=pick_slot_entropy,
    )


def update_standings(standings: Dict[str, Standing], result: GameResult) -> None:
    s0 = standings[result.p0]
    s1 = standings[result.p1]
    s0.games += 1
    s1.games += 1
    s0.score_for += result.score0
    s0.score_against += result.score1
    s1.score_for += result.score1
    s1.score_against += result.score0

    # Consume the cascade-derived winner (set in play_game), not a raw score
    # comparison, so score ties are attributed by the tiebreaker cascade.
    if result.winner == result.p0:
        s0.wins += 1
        s1.losses += 1
    elif result.winner == result.p1:
        s1.wins += 1
        s0.losses += 1
    else:
        s0.draws += 1
        s1.draws += 1


def update_pair(pair: PairResult, result: GameResult, a_name: str, b_name: str) -> None:
    if result.p0 == a_name:
        a_score, b_score = result.score0, result.score1
    else:
        a_score, b_score = result.score1, result.score0
    pair.games += 1
    pair.a_score_sum += a_score
    pair.b_score_sum += b_score
    # Consume the cascade-derived winner (set in play_game), not a raw score
    # comparison, so score ties are attributed by the tiebreaker cascade.
    if result.winner == a_name:
        pair.a_wins += 1
    elif result.winner == b_name:
        pair.b_wins += 1
    else:
        pair.draws += 1


def evaluate_pair(
    a: Participant,
    b: Participant,
    *,
    seed_start: int,
    seeds_per_pair: int,
    verbose: bool,
) -> Tuple[PairResult, List[GameResult]]:
    # Build bots once per pair. This avoids reloading checkpoints every game while
    # keeping pair evaluations isolated from other pairings.
    bot_a = a.make_bot()
    bot_b = b.make_bot()
    pair = PairResult(a=a.name, b=b.name)
    games: List[GameResult] = []
    t0 = time.time()

    for i in range(seeds_per_pair):
        seed = seed_start + i
        # Paired seed: A first, then B first, same deck/start-player seed.
        r1 = play_game(a.name, bot_a, b.name, bot_b, seed=seed)
        r2 = play_game(b.name, bot_b, a.name, bot_a, seed=seed)
        for r in (r1, r2):
            update_pair(pair, r, a.name, b.name)
            games.append(r)

        if verbose and seeds_per_pair >= 10 and (i + 1) % max(1, seeds_per_pair // 5) == 0:
            print(f"    {a.name} vs {b.name}: {i+1}/{seeds_per_pair} paired seeds")

    pair.seconds = time.time() - t0
    return pair, games


def checkpoint_net(participant: Participant, device: str) -> KingdominoNet:
    if participant.model_kwargs is None or participant.state_dict is None:
        raise ValueError(f"Batched eval requires checkpoint participants, got {participant.name!r}")
    net = KingdominoNet(**participant.model_kwargs)
    net.load_state_dict(participant.state_dict)
    net.to(device)
    net.eval()
    return net


def evaluate_rows_by_actor(
    *,
    evaluator0,
    evaluator1,
    mb: np.ndarray,
    ob: np.ndarray,
    flat: np.ndarray,
    idxs_list,
    actors: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    b = int(mb.shape[0])
    values = np.zeros((b,), dtype=np.float64)
    gathered: List[Optional[np.ndarray]] = [None] * b
    idxs = list(idxs_list)

    for actor, evaluator in ((0, evaluator0), (1, evaluator1)):
        rows = np.flatnonzero(actors == actor)
        if len(rows) == 0:
            continue
        sub_idxs = [idxs[int(i)] for i in rows]
        sub_values, sub_gathered = evaluator(mb[rows], ob[rows], flat[rows], sub_idxs)
        for out_i, row in enumerate(rows):
            values[int(row)] = float(sub_values[out_i])
            gathered[int(row)] = np.asarray(sub_gathered[out_i], dtype=np.float64)

    return values, [g if g is not None else np.zeros((0,), dtype=np.float64) for g in gathered]


def evaluate_batched_orientation(
    p0: Participant,
    p1: Participant,
    *,
    seed_start: int,
    n_games: int,
    args: argparse.Namespace,
    verbose: bool,
) -> List[GameResult]:
    if args.determinizations != 1:
        raise ValueError("--engine batched currently requires --determinizations 1")
    if args.temperature != 0.0:
        raise ValueError("--engine batched currently requires --temperature 0")

    import kingdomino_rust
    from games.kingdomino.self_play import make_rust_evaluator

    net0 = checkpoint_net(p0, args.device)
    net1 = checkpoint_net(p1, args.device)
    evaluator0 = make_rust_evaluator(net0, device=args.device, amp=args.amp_inference)
    evaluator1 = make_rust_evaluator(net1, device=args.device, amp=args.amp_inference)

    batched = kingdomino_rust.BatchedMCTS(
        int(args.batch_slots),
        int(n_games),
        int(seed_start),
        int(args.sims),
        leaf_batch=int(args.leaf_batch),
        virtual_loss=1,
        cpuct=float(args.c_puct),
        fpu=0.0,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.0,
        temp_moves=0,
    )

    results: List[GameResult] = []
    ticks = 0
    while not batched.done():
        mb, ob, flat, idxs_list = batched.step()
        actors = np.asarray(batched.row_actors(), dtype=np.int64)
        values, gathered = evaluate_rows_by_actor(
            evaluator0=evaluator0,
            evaluator1=evaluator1,
            mb=np.asarray(mb),
            ob=np.asarray(ob),
            flat=np.asarray(flat),
            idxs_list=idxs_list,
            actors=actors,
        )
        finished = batched.update(values, gathered)
        for seed, examples, scores in finished:
            score0, score1 = int(scores[0]), int(scores[1])
            # The Rust BatchedMCTS path returns only final score totals — no
            # Python GameState or per-board tiebreaker quantities — so the full
            # cascade (determine_winner) can't run here without out-of-scope Rust
            # engine work. Genuine score ties fall through to a draw. The serial
            # engine (default) routes through determine_winner in play_game.
            if score0 > score1:
                winner = p0.name
            elif score1 > score0:
                winner = p1.name
            else:
                winner = None
            results.append(GameResult(
                seed=int(seed),
                p0=p0.name,
                p1=p1.name,
                score0=score0,
                score1=score1,
                winner=winner,
                steps=len(examples),
            ))
        ticks += 1
        if verbose and n_games >= 10 and ticks % 1000 == 0:
            print(f"    {p0.name} as P0 vs {p1.name}: {len(results)}/{n_games} games")
        if ticks > 2_000_000:
            raise RuntimeError("Batched eval exceeded tick guard")

    results.sort(key=lambda g: g.seed)
    return results


def evaluate_pair_batched(
    a: Participant,
    b: Participant,
    *,
    seed_start: int,
    seeds_per_pair: int,
    args: argparse.Namespace,
    verbose: bool,
) -> Tuple[PairResult, List[GameResult]]:
    if a.kind != "checkpoint" or b.kind != "checkpoint":
        raise ValueError("--engine batched only supports checkpoint participants")

    pair = PairResult(a=a.name, b=b.name)
    games: List[GameResult] = []
    t0 = time.time()

    forward = evaluate_batched_orientation(
        a, b, seed_start=seed_start, n_games=seeds_per_pair,
        args=args, verbose=verbose,
    )
    swapped = evaluate_batched_orientation(
        b, a, seed_start=seed_start, n_games=seeds_per_pair,
        args=args, verbose=verbose,
    )
    for r in forward + swapped:
        update_pair(pair, r, a.name, b.name)
        games.append(r)

    pair.seconds = time.time() - t0
    return pair, games


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def pair_row(pair: PairResult) -> dict:
    return {
        "a": pair.a,
        "b": pair.b,
        "games": pair.games,
        "a_wins": pair.a_wins,
        "b_wins": pair.b_wins,
        "draws": pair.draws,
        "a_points": f"{pair.a_points:.1f}",
        "b_points": f"{pair.b_points:.1f}",
        "a_win_rate": f"{pair.a_win_rate:.4f}",
        "avg_a_score": f"{pair.avg_a_score:.3f}",
        "avg_b_score": f"{pair.avg_b_score:.3f}",
        "avg_margin_a": f"{pair.avg_margin_a:.3f}",
        "seconds": f"{pair.seconds:.2f}",
        "games_per_sec": f"{(pair.games / pair.seconds) if pair.seconds > 0 else 0.0:.4f}",
    }


def standing_row(s: Standing) -> dict:
    return {
        "participant": s.name,
        "games": s.games,
        "wins": s.wins,
        "losses": s.losses,
        "draws": s.draws,
        "points": f"{s.points:.1f}",
        "win_rate": f"{s.win_rate:.4f}",
        "avg_score": f"{s.avg_score:.3f}",
        "avg_margin": f"{s.avg_margin:.3f}",
        "score_for": s.score_for,
        "score_against": s.score_against,
    }


def write_csv(path: str | os.PathLike, rows: Sequence[dict]) -> None:
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_leaderboard(standings: Dict[str, Standing]) -> None:
    ordered = sorted(
        standings.values(),
        key=lambda s: (s.points, s.avg_margin, s.avg_score),
        reverse=True,
    )
    print("\nLeaderboard")
    print("-" * 86)
    print(f"{'#':>2}  {'participant':<28} {'games':>5} {'W-L-D':>9} {'pts':>6} {'win%':>7} {'avg_margin':>11} {'avg_score':>10}")
    print("-" * 86)
    for rank, s in enumerate(ordered, 1):
        wld = f"{s.wins}-{s.losses}-{s.draws}"
        print(f"{rank:>2}  {s.name:<28} {s.games:>5} {wld:>9} {s.points:>6.1f} {s.win_rate:>6.1%} {s.avg_margin:>11.2f} {s.avg_score:>10.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kingdomino paired round-robin evaluator")
    p.add_argument("--manifest", default=None,
                   help="CSV with checkpoint rows: name,path,channels,blocks,bilinear_dim. Per-row architecture fields are optional.")
    p.add_argument("--checkpoints", nargs="*", default=[],
                   help="Checkpoint .pt files to evaluate.")
    p.add_argument("--names", nargs="*", default=None,
                   help="Optional display names for checkpoints; must align with --checkpoints.")
    p.add_argument("--include_greedy", action="store_true")
    p.add_argument("--include_random", action="store_true")

    # Evaluation/search settings.
    p.add_argument("--open_loop", action="store_true",
                   help="Use OpenLoopEvalBot for checkpoint participants "
                        "instead of AlphaZeroEvalBot.")
    p.add_argument("--engine", choices=("serial", "batched"), default="serial",
                   help="serial uses Python AlphaZeroMCTS; batched uses Rust BatchedMCTS for checkpoint-only pairs.")
    p.add_argument("--sims", type=int, default=50)
    p.add_argument("--determinizations", type=int, default=1)
    p.add_argument("--batch_slots", type=int, default=86,
                   help="Concurrent Rust BatchedMCTS slots when --engine batched.")
    p.add_argument("--leaf_batch", type=int, default=6,
                   help="Leaf batch per slot when --engine batched.")
    p.add_argument("--amp_inference", action="store_true",
                   help="Use autocast fp16 inference for --engine batched.")
    p.add_argument("--c_puct", type=float, default=1.5)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Move-selection temperature from visit counts; 0 is greedy.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=100_000)
    p.add_argument("--seeds_per_pair", type=int, default=None,
                   help="Paired deck seeds per pairing. Total games per pair = 2 * this.")
    p.add_argument("--games_per_pair", type=int, default=40,
                   help="Total games per pairing; rounded up to an even paired-seed count if --seeds_per_pair is not set.")

    # Optional architecture overrides. If omitted, checkpoint config is used.
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--blocks", type=int, default=None)
    p.add_argument("--bilinear_dim", type=int, default=None)

    # Output.
    p.add_argument("--output", default="eval_results/round_robin_pairs.csv")
    p.add_argument("--leaderboard_output", default=None)
    p.add_argument("--game_log_output", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)

    seeds_per_pair = args.seeds_per_pair
    if seeds_per_pair is None:
        seeds_per_pair = max(1, int(math.ceil(args.games_per_pair / 2)))
    total_games_per_pair = 2 * seeds_per_pair

    participants = build_participants(args)
    print("Round-robin participants:")
    for p in participants:
        src = f" [{p.source}]" if p.source else ""
        print(f"  - {p.name} ({p.kind}){src}")
    print(f"\nPairing settings: {seeds_per_pair} paired seeds = {total_games_per_pair} games/pair, sims={args.sims}, determinizations={args.determinizations}, device={args.device}, engine={args.engine}")
    if args.engine == "batched":
        print(f"Batched settings: batch_slots={args.batch_slots}, leaf_batch={args.leaf_batch}, amp_inference={args.amp_inference}")

    standings: Dict[str, Standing] = {p.name: Standing(p.name) for p in participants}
    pair_results: List[PairResult] = []
    game_results: List[GameResult] = []

    t_all = time.time()
    pair_idx = 0
    for i in range(len(participants)):
        for j in range(i + 1, len(participants)):
            a, b = participants[i], participants[j]
            pair_idx += 1
            pair_seed_start = args.seed + pair_idx * 1_000_000
            print(f"\n[{pair_idx}] {a.name} vs {b.name} ({total_games_per_pair} games)")
            if args.engine == "batched":
                pair, games = evaluate_pair_batched(
                    a,
                    b,
                    seed_start=pair_seed_start,
                    seeds_per_pair=seeds_per_pair,
                    args=args,
                    verbose=args.verbose,
                )
            else:
                pair, games = evaluate_pair(
                    a,
                    b,
                    seed_start=pair_seed_start,
                    seeds_per_pair=seeds_per_pair,
                    verbose=args.verbose,
                )
            for g in games:
                update_standings(standings, g)
            pair_results.append(pair)
            game_results.extend(games)
            print(
                f"  {a.name}: {pair.a_wins}-{pair.b_wins}-{pair.draws} vs {b.name} "
                f"| win_rate={pair.a_win_rate:.1%} avg_margin={pair.avg_margin_a:+.2f} "
                f"| {pair.games / pair.seconds if pair.seconds > 0 else 0.0:.2f} games/sec"
            )

    elapsed = time.time() - t_all
    print_leaderboard(standings)
    print(f"\nTotal: {len(game_results)} games in {elapsed:.1f}s ({len(game_results)/elapsed if elapsed > 0 else 0.0:.3f} games/sec)")

    # Degenerate-play gate: flag (not fail) if >5% of games look degenerate.
    degenerate = [g for g in game_results
                  if g.discard_rate > 0.8 or g.pick_slot_entropy < 0.1]
    n_games = len(game_results)
    frac = len(degenerate) / n_games if n_games else 0.0
    print(f"Degenerate games: {len(degenerate)}/{n_games} ({frac:.1%}) "
          f"[discard_rate>0.8 or pick_slot_entropy<0.1]")
    if frac > 0.05:
        print(f"  ⚠ WARNING: >5% of games are degenerate — investigate "
              f"(always-discard / always-first-tile / stuck policy).")

    pair_rows = [pair_row(p) for p in pair_results]
    write_csv(args.output, pair_rows)
    print(f"Wrote pair results: {args.output}")

    leaderboard_path = args.leaderboard_output
    if leaderboard_path is None:
        out = Path(args.output)
        leaderboard_path = str(out.with_name(out.stem + "_leaderboard.csv"))
    standing_rows = [standing_row(s) for s in sorted(
        standings.values(), key=lambda x: (x.points, x.avg_margin, x.avg_score), reverse=True
    )]
    write_csv(leaderboard_path, standing_rows)
    print(f"Wrote leaderboard: {leaderboard_path}")

    if args.game_log_output:
        rows = [
            {
                "seed": g.seed,
                "p0": g.p0,
                "p1": g.p1,
                "score0": g.score0,
                "score1": g.score1,
                "winner": g.winner or "DRAW",
                "steps": g.steps,
            }
            for g in game_results
        ]
        write_csv(args.game_log_output, rows)
        print(f"Wrote game log: {args.game_log_output}")


if __name__ == "__main__":
    main()
