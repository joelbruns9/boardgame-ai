"""Head-to-head arena: does A actually PLAY better than B?

Every other measurement in this programme is an imitation metric -- policy
top-1, KL against a teacher, corpus regret.  All of them answer "does this model
predict the reference better", which is a different question from "does this
model win more games", and the two have already come apart once (W9's null was a
reporting artefact, not a strength result).

The machinery to play a scored match existed, but only fused into Phase D's
promotion lifecycle: it required a run directory, took both sides' architecture
from ONE run config, gave both sides the same simulation count, and returned a
promote/revert/continue verdict rather than a measurement.  W6.2b's precision
arena had to construct a whole `PhaseDLoop` to borrow it, and could still only
compare a checkpoint against ITSELF.

This module is the plain tool:

    python -m games.seven_wonders_duel.arena \\
        --a runs/cloud6/checkpoints/learner_0090.pt \\
        --b extension_7wd/candidate_0085.pt \\
        --pairs 200 --sims 64 --device cuda --output arena.json

and it answers exactly one question -- A's paired score rate against B, with a
Wilson interval -- while making four things explicit that the fused gate could
not.

**Each side is rebuilt from its own checkpoint.**  `d_model`, `layers`, `heads`,
`pooled_readout`, `reply_head`, `action_residual` and `control_head` all come
from the file being played, through `train.model_from_config`.  So a W5b width
change or a W1 head change can be played against the incumbent; the fused gate
would have loaded both under the run's config and either crashed on shape or --
for `heads`, whose parameter shapes are head-count independent -- silently
computed something the weights were never trained for.

**Equal wall-clock, measured rather than asserted.**  Equal *simulations* is not
a fair fight between architectures of different cost: a wider net that thinks
just as long does strictly more work per move, so an equal-sims win confounds
"better policy" with "more compute".  `--parity time` calibrates each side's
cost per move against simulation count and gives the non-reference side whatever
search it can afford in the time the reference side takes.  Either way the arena
times each side's searches as it plays and reports the ratio it ACHIEVED, so the
parity claim rests on a measurement and not on the calibration's arithmetic.

**Per-side control-feature arms.**  W3's whole point is a baseline that cannot
see exact positional control against a model that can, and `set_control_features`
is a process-wide switch in Python and Rust together -- one process cannot encode
two ways.  Off-mode is defined as zeroing the control channels, so the arena runs
the encoder ON and zeroes those columns for whichever side was trained OFF, which
is bit-identical to what that side saw in training.  Each side's arm comes from
its own checkpoint's `control_features` stamp.

**Fixed N, read once.**  The whole sample is played and the interval is read at
its final size.  There is no early stop, no promotion, no revert, and nothing is
written back into a run.  Optional stopping on a Wilson boundary promotes an
evenly matched candidate 15-19% of the time (W5.5); an arena that stopped early
would report the same inflation as a strength result.

Games are stepped a ply at a time and every search runs wholly under the mover's
own network.  That is not an implementation detail: routing individual leaves by
their acting player let the opponent's net evaluate the interior of a side's own
tree, which INVERTED results (0.225 vs 0.925 against a random-init net).  Each
side here gets its own single-net evaluator and adapter, so the mixing cannot
recur by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from games.az_loop import hardware_identity, wilson_interval

from .encoder import (
    CONTROL_FEATURES,
    ENCODER_SIGNATURE,
    GLOBAL_FEATURES,
    TABLEAU_FEATURES,
    TokenType,
)
from .rust_bridge import rust_flat_batch_adapter, rust_games_for_self_play

#: Token-type ids as Rust packs them (`encoder.rs`: "in `encoder.py::TokenType`
#: declaration order").  Derived from the enum rather than written down, so a
#: reordering breaks here loudly instead of masking the wrong columns.
_TYPE_IDS = {token: index for index, token in enumerate(TokenType)}
_T_GLOBAL = _TYPE_IDS[TokenType.GLOBAL]
_T_TABLEAU = _TYPE_IDS[TokenType.TABLEAU]

#: Which columns "control inputs off" zeroes, in packed-row coordinates.  A
#: token's features occupy `[0, len(schema))` of the shared padded width, so a
#: feature's column is simply its index in that token type's tuple.
_CONTROL_TABLEAU_COLUMNS = tuple(
    TABLEAU_FEATURES.index(name) for name in CONTROL_FEATURES
)
_CONTROL_VALID_COLUMN = GLOBAL_FEATURES.index("control_valid")

#: Seed base for arena pairs.  Distinct from Phase D's gate offsets (50/52M) so
#: an arena never replays the exact deals a promotion gate already scored, and
#: distinct from the calibration offset below.
ARENA_SEED_OFFSET = 61_000_000
CALIBRATION_SEED_OFFSET = 62_000_000

#: Exit codes.  argparse already owns 2 ("could not run"), and an uncaught
#: exception exits 1, so a gate failure gets its own code: a caller that cannot
#: separate "A did not clear the bar" from "the arena crashed" reports crashes
#: as findings, which is what the cloud launcher did before `precision_arena`
#: made the same distinction.
GATE_FAILED_EXIT_CODE = 3


# --- control-feature arms ---------------------------------------------------


class ControlMaskedModel:
    """Wrap a model so it is shown the control channels as zeros.

    `encoder.set_control_features(False)` is process-wide, in Python and Rust
    together, and an arena needs both arms live in one process.  Off-mode is
    defined (encoder.py) as emitting the control channels as zeros with
    `control_valid` 0 -- pure zeroing of known columns -- so applying it to one
    side's rows after packing reproduces exactly what that side saw in training,
    while the other side still receives the real values.

    A plain callable, not an `nn.Module`: the flat adapter only calls
    `evaluator.model(batch)` and reads `.action_residual`, and wrapping after
    `Evaluator.__init__` keeps the embedder fusion `.to()` would have
    invalidated.
    """

    def __init__(self, model):
        self.model = model
        # Read by `_RustFlatBatchAdapter` to decide whether to build the padded
        # legal-action view.  Forwarded, not recomputed.
        self.action_residual = bool(getattr(model, "action_residual", False))

    def __call__(self, batch):
        features = batch["features"]
        type_ids = batch["type_ids"]
        # Clone rather than mutate: the packed batch is the adapter's buffer and
        # a caller may hold it (the cost model's probe drives `build_device_batch`
        # directly).  One [rows, tokens, width] copy per batch is negligible next
        # to the forward it feeds.
        masked = features.clone()
        # Only on the token types that OWN these columns.  A token's features
        # occupy a prefix of the shared padded width, so column 26 is a control
        # channel on a tableau token and an entirely different feature on a pool
        # token (79 wide); zeroing it everywhere would corrupt the input.
        tableau = type_ids == _T_TABLEAU
        for column in _CONTROL_TABLEAU_COLUMNS:
            masked[..., column] = masked[..., column].masked_fill(tableau, 0.0)
        masked[..., _CONTROL_VALID_COLUMN] = masked[
            ..., _CONTROL_VALID_COLUMN
        ].masked_fill(type_ids == _T_GLOBAL, 0.0)
        return self.model({**batch, "features": masked})


def control_arm(checkpoint: dict) -> str:
    """Which W3 arm a checkpoint was trained under.

    An unstamped file predates the stamp, and therefore predates the control
    channels entirely; its encoder signature differs, so it can only be played
    after an additive migration, which zero-initialises the new input columns
    (`train.migrate_state_dict`).  Zero columns consume the control channels as
    exactly nothing, so "off" is not a guess about such a file -- it is what the
    weights do.
    """

    arm = checkpoint.get("control_features")
    if arm in ("on", "off"):
        return arm
    return "off"


# --- sides ------------------------------------------------------------------


@dataclass
class Side:
    """One competitor: a checkpoint, the architecture it names, and its budget."""

    label: str
    source: str
    sha256: str
    name: str
    architecture: dict[str, Any]
    control_arm: str
    precision: str
    sims: int = 0
    migration: dict[str, Any] | None = None
    model: Any = field(default=None, repr=False, compare=False)
    evaluator: Any = field(default=None, repr=False, compare=False)
    adapter: Any = field(default=None, repr=False, compare=False)
    #: Filled by the arena itself: wall-clock this side spent inside
    #: `search_many_flat_net`, and how many of its own moves that bought.
    search_seconds: float = 0.0
    moves: int = 0

    @property
    def seconds_per_move(self) -> float:
        return self.search_seconds / self.moves if self.moves else 0.0

    def mask_control_features(self) -> None:
        """Show this side the control channels as zeros, whatever the encoder emits.

        Applied by the arena, not by `load_side`, because whether it is NEEDED
        depends on the other side: with both sides on the off arm the encoder
        itself is off and the channels are already zero, so masking would buy a
        tensor copy per batch and change nothing.
        """

        if isinstance(self.evaluator.model, ControlMaskedModel):
            return
        self.evaluator.model = ControlMaskedModel(self.evaluator.model)

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source": self.source,
            "sha256": self.sha256,
            "name": self.name,
            "architecture": dict(self.architecture),
            "control_arm": self.control_arm,
            "precision": self.precision,
            "sims": self.sims,
            "migrated": self.migration is not None,
            "migration": _migration_summary(self.migration),
        }


def _migration_summary(migration: dict[str, Any] | None) -> dict[str, Any] | None:
    """Counts, not the key lists: a full report is thousands of names."""

    if migration is None:
        return None
    return {
        key: (len(value) if isinstance(value, list) else value)
        for key, value in migration.items()
    }


def _file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_side(
    label: str,
    path: str | Path,
    *,
    device: str,
    precision: str,
    batch_cap: int,
    migrate: bool = False,
    name: str | None = None,
) -> Side:
    """Rebuild one checkpoint under the architecture IT names.

    `migrate` is off by default and stays a deliberate act.  A migrated model is
    not the model that was trained: new inputs and new heads are zeroed or
    freshly initialised, and playing one in an arena measures the migration as
    much as the checkpoint.  When it is the only way to play an older file the
    report says so on the row.
    """

    import torch

    from .inference import Evaluator
    from .train import load_checkpoint, model_from_config

    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    stored = checkpoint.get("config", {})
    model = model_from_config(stored)
    try:
        load_checkpoint(path, model, migrate=migrate, checkpoint=checkpoint)
    except ValueError as error:
        # Only the migration refusal earns the migration hint.  `load_checkpoint`
        # also raises on a control-table digest mismatch, whose fix is to restore
        # the matching table -- pointing that at --migrate would send the reader
        # to the wrong repair.
        if migrate or "migration required" not in str(error):
            raise ValueError(f"{label}: {path}: {error}") from error
        raise ValueError(
            f"{label}: {path} cannot be played as it stands -- {error}. "
            "Pass --migrate to warm-start it into the current schema, and read "
            "the arena as measuring the migrated model, not the trained one."
        ) from error
    arm = control_arm(checkpoint)
    evaluator = Evaluator(model, device, batch_cap, precision=precision)
    return Side(
        label=label,
        source=str(path.resolve()),
        sha256=_file_digest(path),
        name=name or f"{label}:{path.stem}",
        architecture={
            key: stored.get(key)
            for key in (
                "d_model",
                "layers",
                "heads",
                "pooled_readout",
                "reply_head",
                "action_residual",
                "control_head",
                "iteration",
            )
        },
        control_arm=arm,
        precision=precision,
        migration=checkpoint.get("migration"),
        model=model,
        evaluator=evaluator,
        adapter=rust_flat_batch_adapter(evaluator),
    )


# --- playing ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GameResult:
    seed: int
    first_player: int
    a_seat: int
    winner: int | None
    scores: tuple[int, int] | None
    victory_type: str
    actions: int

    def score_for_a(self) -> float:
        if self.winner is None:
            return 0.5
        return 1.0 if self.winner == self.a_seat else 0.0


def _step_games(
    swr,
    games: list,
    seeds: Sequence[int],
    seat_sides: tuple[Side, Side],
    *,
    batch_cap: int,
    leaf_batch: int,
    top_k: int,
    puct_root: bool,
    force_root_chance: bool,
    age_deal_samples: int,
    max_moves: int,
) -> list[dict]:
    """Play a batch of games a ply at a time, each search under the mover's net.

    `seat_sides[s]` is the side sitting in seat `s`, so a search never mixes
    networks -- the failure that inverted arena results before the arena was
    stepped from Python at all.  Games sharing a mover still batch together, so
    the cost of the split is roughly half the rows per call, not half the speed.

    Timing is accumulated per SIDE here rather than per call, because this is the
    only place that knows both how long a search took and whose it was.  It
    measures the search, not the ply: everything outside the `swr` call is
    bookkeeping shared by both sides.
    """

    live = list(range(len(games)))
    actions_played = [0] * len(games)
    move_index = 0
    while live:
        if move_index >= max_moves:
            raise RuntimeError(
                f"arena game exceeded {max_moves} moves without completing; "
                "raise --max-moves only if the engine legitimately allows it"
            )
        by_seat: dict[int, list[int]] = {0: [], 1: []}
        for slot in live:
            by_seat[games[slot].actor].append(slot)
        for seat, slots in by_seat.items():
            if not slots:
                continue
            side = seat_sides[seat]
            started = time.perf_counter()
            results = swr.search_many_flat_net(
                side.adapter,
                [games[slot] for slot in slots],
                # Distinct per game, per ply, and per seat: two sides searching
                # the same position on the same ply must not share a stream.
                [seeds[slot] + move_index * 1_000_003 + seat * 7_919 for slot in slots],
                batch_cap,
                leaf_batch,
                side.sims,
                top_k,
                force=force_root_chance,
                age_deal_samples=age_deal_samples,
                puct_root=puct_root,
            )
            side.search_seconds += time.perf_counter() - started
            side.moves += len(slots)
            for slot, result in zip(slots, results):
                legal = games[slot].legal_action_indices()
                policy = result["policy"]
                # argmax of the improved policy, never the Gumbel-perturbed
                # `action`: at a small budget that action reduces to a SAMPLE
                # from the prior, and exploration noise has no place in a scored
                # game.
                best = max(range(len(legal)), key=lambda index: policy[index])
                games[slot].apply_index(legal[best])
                actions_played[slot] += 1
        move_index += 1
        live = [slot for slot in live if not games[slot].is_complete()]
    return [
        {
            "winner": game.winner,
            "victory_type": game.victory_type,
            "scores": game.final_scores,
            "actions": actions,
        }
        for game, actions in zip(games, actions_played)
    ]


def play_pairs(
    a: Side,
    b: Side,
    *,
    pairs: int,
    seed: int,
    slots: int,
    batch_cap: int,
    leaf_batch: int,
    top_k: int,
    puct_root: bool,
    force_root_chance: bool,
    age_deal_samples: int,
    max_moves: int,
    seed_offset: int = ARENA_SEED_OFFSET,
    progress=None,
) -> list[GameResult]:
    """Play `pairs` seat-paired games and return every game, in play order.

    A pair is one deal played twice with the seats swapped, so the deal and the
    first-player assignment are held fixed across the two legs and only the
    seating changes.  That is what makes a pair an observation: it cancels the
    deal and cancels first-player advantage within itself, which a per-game
    binomial does not, and it is why the interval below is computed over pairs
    rather than over games.
    """

    import seven_wonders_rust as swr

    results: list[GameResult] = []
    for start in range(0, pairs, slots):
        indices = list(range(start, min(start + slots, pairs)))
        seeds = [seed + seed_offset + index for index in indices]
        # Alternated across pairs as well, so the arena is balanced on who deals
        # first even before the seat swap cancels it within each pair.
        first_players = [index % 2 for index in indices]
        legs = []
        for a_seat in (0, 1):
            seat_sides = (a, b) if a_seat == 0 else (b, a)
            legs.append(
                _step_games(
                    swr,
                    rust_games_for_self_play(seeds, first_players),
                    seeds,
                    seat_sides,
                    batch_cap=batch_cap,
                    leaf_batch=leaf_batch,
                    top_k=top_k,
                    puct_root=puct_root,
                    force_root_chance=force_root_chance,
                    age_deal_samples=age_deal_samples,
                    max_moves=max_moves,
                )
            )
        for offset, (seed_value, first_player) in enumerate(zip(seeds, first_players)):
            for a_seat in (0, 1):
                record = legs[a_seat][offset]
                scores = record["scores"]
                results.append(
                    GameResult(
                        seed=seed_value,
                        first_player=first_player,
                        a_seat=a_seat,
                        winner=record["winner"],
                        scores=tuple(scores) if scores is not None else None,
                        victory_type=record["victory_type"] or "unknown",
                        actions=record["actions"],
                    )
                )
        if progress is not None:
            progress(min(start + slots, pairs), pairs)
    return results


def pair_scores(results: Sequence[GameResult]) -> list[float]:
    """One observation per deal: 1 if A won the pair, 0.5 if it split, 0 if lost.

    Both legs of a pair must be present and must disagree about seating, or the
    "observation" is a single game wearing a pair's clothes.
    """

    if len(results) % 2:
        raise ValueError("arena results must contain complete seat pairs")
    scores: list[float] = []
    for index in range(0, len(results), 2):
        first, second = results[index], results[index + 1]
        if first.seed != second.seed or first.a_seat == second.a_seat:
            raise ValueError(
                f"pair at index {index} is not a seat swap of one deal "
                f"(seeds {first.seed}/{second.seed}, "
                f"A seats {first.a_seat}/{second.a_seat})"
            )
        points = first.score_for_a() + second.score_for_a()
        scores.append(1.0 if points > 1.0 else (0.5 if points == 1.0 else 0.0))
    return scores


# --- wall-clock parity ------------------------------------------------------


def _linear_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Least-squares `y = a + b*x`.  Returns `(a, b)`.

    Ordinary least squares over three or more points rather than a slope through
    two: a search's cost is a fixed per-call overhead plus a per-simulation cost,
    and two points cannot separate them from noise at all -- they just name a
    line through whatever the two measurements happened to be.
    """

    n = len(xs)
    if n < 2:
        raise ValueError("a cost fit needs at least two simulation counts")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance <= 0:
        raise ValueError("a cost fit needs at least two DISTINCT simulation counts")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance
    return mean_y - slope * mean_x, slope


def calibrate_cost(
    side: Side,
    *,
    sims_points: Sequence[int],
    games: int,
    seed: int,
    batch_cap: int,
    leaf_batch: int,
    top_k: int,
    puct_root: bool,
    force_root_chance: bool,
    age_deal_samples: int,
    max_moves: int,
) -> dict[str, Any]:
    """Measure this side's seconds per move against its simulation count.

    The side plays BOTH seats, so the measurement is of the network alone under
    the same batching the arena will use -- not of a matchup.  Under equal-batch
    conditions the two seats of a mirror are the same cost, so the whole call's
    time divided by the whole call's moves is the per-move cost at that budget.

    Returns the fit and the raw points; the caller keeps both, because a fit
    whose points are noisy is a fit that should not be trusted to extrapolate.
    """

    import seven_wonders_rust as swr

    points = []
    for sims in sims_points:
        side.search_seconds = 0.0
        side.moves = 0
        saved = side.sims
        side.sims = int(sims)
        try:
            _step_games(
                swr,
                rust_games_for_self_play(
                    [seed + CALIBRATION_SEED_OFFSET + index for index in range(games)],
                    [index % 2 for index in range(games)],
                ),
                [seed + CALIBRATION_SEED_OFFSET + index for index in range(games)],
                (side, side),
                batch_cap=batch_cap,
                leaf_batch=min(leaf_batch, batch_cap),
                top_k=top_k,
                puct_root=puct_root,
                force_root_chance=force_root_chance,
                age_deal_samples=age_deal_samples,
                max_moves=max_moves,
            )
        finally:
            measured = side.seconds_per_move
            moves = side.moves
            seconds = side.search_seconds
            side.sims = saved
            side.search_seconds = 0.0
            side.moves = 0
        points.append(
            {
                "sims": int(sims),
                "seconds_per_move": measured,
                "moves": moves,
                "seconds": seconds,
            }
        )
    intercept, slope = _linear_fit(
        [point["sims"] for point in points],
        [point["seconds_per_move"] for point in points],
    )
    return {"points": points, "intercept": intercept, "slope": slope}


def sims_for_budget(fit: dict[str, Any], budget_seconds: float, *, floor: int, cap: int) -> int:
    """Invert a cost fit: how many simulations fit in `budget_seconds`?

    Clamped rather than allowed to run away.  A near-zero or negative fitted
    slope means the measurement did not resolve the per-simulation cost (a
    launch-bound side on a fast device, say), and extrapolating through it would
    hand one side an unbounded budget off a measurement that says nothing.  The
    clamp is reported, so a run that hit it is visible as a run that hit it.
    """

    slope = fit["slope"]
    if slope <= 0 or not math.isfinite(slope):
        return cap
    solved = (budget_seconds - fit["intercept"]) / slope
    if not math.isfinite(solved):
        return cap
    return max(floor, min(cap, int(round(solved))))


# --- the arena ---------------------------------------------------------------


def run(
    a_path: str | Path,
    b_path: str | Path,
    *,
    pairs: int,
    device: str = "cuda",
    precision: str = "fp32",
    sims: int = 64,
    sims_a: int | None = None,
    sims_b: int | None = None,
    parity: str = "sims",
    reference: str = "a",
    calibration_games: int = 0,
    calibration_sims: Sequence[int] | None = None,
    parity_tolerance: float = 0.15,
    min_sims: int = 0,
    max_sims: int = 0,
    top_k: int = 16,
    search: str = "gumbel",
    slots: int = 48,
    batch_cap: int = 256,
    leaf_batch: int = 1,
    force_root_chance: bool = True,
    age_deal_samples: int = 32,
    max_moves: int = 512,
    seed: int = 0,
    z: float = 1.96,
    migrate: bool = False,
    min_lcb: float | None = None,
    log=None,
) -> dict[str, Any]:
    """Play A against B and return the report.  Never writes into a run."""

    from .encoder import set_control_features

    log = log or (lambda message: None)
    if pairs <= 0:
        raise ValueError("pairs must be positive")
    if parity not in ("sims", "time"):
        raise ValueError("parity must be 'sims' or 'time'")
    if reference not in ("a", "b"):
        raise ValueError("reference must be 'a' or 'b'")
    if search not in ("gumbel", "puct"):
        raise ValueError("search must be 'gumbel' or 'puct'")

    started = time.monotonic()
    a = load_side("a", a_path, device=device, precision=precision, batch_cap=batch_cap, migrate=migrate)
    b = load_side("b", b_path, device=device, precision=precision, batch_cap=batch_cap, migrate=migrate)

    # One process, one encoder.  Emit the control channels whenever ANY side was
    # trained to read them; a side trained without them is masked back to zeros
    # by its own wrapper, which is what off-mode is.  With neither side on, the
    # encoder skips the table lookups entirely.
    controls_on = "on" in (a.control_arm, b.control_arm)
    set_control_features(controls_on)
    if controls_on:
        for side in (a, b):
            if side.control_arm == "off":
                side.mask_control_features()
    log(
        f"control features: encoder {'on' if controls_on else 'off'}; "
        f"a={a.control_arm} b={b.control_arm}"
    )

    puct_root = search == "puct"
    search_kwargs = dict(
        slots=slots,
        batch_cap=batch_cap,
        leaf_batch=leaf_batch,
        top_k=top_k,
        puct_root=puct_root,
        force_root_chance=force_root_chance,
        age_deal_samples=age_deal_samples,
        max_moves=max_moves,
    )

    calibration: dict[str, Any] | None = None
    if sims_a is not None or sims_b is not None:
        # An explicit budget is the caller's parity claim, not the arena's.
        parity = "manual"
        a.sims = int(sims_a if sims_a is not None else sims)
        b.sims = int(sims_b if sims_b is not None else sims)
    elif parity == "sims":
        a.sims = b.sims = int(sims)
    else:
        # Bounded either way round, so a fit that failed to resolve the
        # per-simulation cost hands one side an 8x handicap rather than an
        # unbounded budget nobody would notice until the arena had run for hours.
        floor = min_sims or max(1, sims // 8)
        cap = max_sims or sims * 8
        points = list(calibration_sims or (max(floor, sims // 2), sims, sims * 2))
        reference_side, other = (a, b) if reference == "a" else (b, a)
        reference_side.sims = int(sims)
        # At the arena's own concurrency: a cost per move measured over 8 games
        # describes 8-game batches, and the arena plays `slots`-game batches on
        # a device whose cost is mostly per batch rather than per row.
        concurrency = calibration_games or min(slots, pairs)
        log(
            f"calibrating cost per move over sims={points} "
            f"at {concurrency} concurrent games ..."
        )
        fits = {}
        for side in (a, b):
            fits[side.label] = calibrate_cost(
                side,
                sims_points=points,
                games=concurrency,
                seed=seed,
                **{k: v for k, v in search_kwargs.items() if k != "slots"},
            )
            log(
                f"  {side.label}: "
                + ", ".join(
                    f"{point['sims']}->{point['seconds_per_move'] * 1000:.1f}ms"
                    for point in fits[side.label]["points"]
                )
            )
        budget = fits[reference_side.label]["intercept"] + (
            fits[reference_side.label]["slope"] * reference_side.sims
        )
        other.sims = sims_for_budget(fits[other.label], budget, floor=floor, cap=cap)
        calibration = {
            "reference": reference,
            "budget_seconds_per_move": budget,
            "concurrent_games": concurrency,
            "floor": floor,
            "cap": cap,
            "fits": fits,
            "clamped": other.sims in (floor, cap),
        }
        log(
            f"  budget {budget * 1000:.1f} ms/move -> "
            f"a={a.sims} sims, b={b.sims} sims"
        )

    log(f"playing {pairs} pairs ({pairs * 2} games): a={a.sims} sims, b={b.sims} sims")

    play_started = time.monotonic()

    def progress(done: int, total: int) -> None:
        elapsed = time.monotonic() - play_started
        log(f"  {done}/{total} pairs, {elapsed / 60:.1f} min of play")

    a.search_seconds = a.moves = 0
    b.search_seconds = b.moves = 0
    results = play_pairs(a, b, pairs=pairs, seed=seed, progress=progress, **search_kwargs)
    # Separated because loading two checkpoints and calibrating a cost model are
    # not the arena's throughput, and a games/hour that quietly included them
    # would misprice every future run sized off this one.
    play_seconds = time.monotonic() - play_started
    elapsed = time.monotonic() - started

    scores = pair_scores(results)
    points_won = sum(scores)
    rate = points_won / len(scores)
    lower, upper = wilson_interval(points_won, len(scores), z=z)

    # The parity that was ACHIEVED, from the arena's own clock.  Normalised per
    # move because the two seats do not move equally often -- 7WD grants extra
    # turns -- so raw seconds would report a turn-count difference as a compute
    # difference.
    ratio = (
        a.seconds_per_move / b.seconds_per_move if b.seconds_per_move > 0 else float("inf")
    )
    within = abs(ratio - 1.0) <= parity_tolerance

    victories: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = victories.setdefault(
            result.victory_type, {"a": 0, "b": 0, "draw": 0}
        )
        if result.winner is None:
            bucket["draw"] += 1
        else:
            bucket["a" if result.winner == result.a_seat else "b"] += 1

    report = {
        "a": a.describe(),
        "b": b.describe(),
        "pairs": len(scores),
        "games": len(results),
        "a_score_rate": rate,
        "wilson": {"lower": lower, "upper": upper, "z": z},
        "null": 0.50,
        "null_inside_interval": lower <= 0.50 <= upper,
        "pair_scores": scores,
        "pair_wins": scores.count(1.0),
        "pair_splits": scores.count(0.5),
        "pair_losses": scores.count(0.0),
        "victory_types": victories,
        "moves_per_game": sum(result.actions for result in results) / len(results),
        "parity": {
            "mode": parity,
            "tolerance": parity_tolerance,
            "measured": {
                "a_seconds_per_move": a.seconds_per_move,
                "b_seconds_per_move": b.seconds_per_move,
                "a_search_seconds": a.search_seconds,
                "b_search_seconds": b.search_seconds,
                "a_moves": a.moves,
                "b_moves": b.moves,
                "ratio_a_over_b": ratio,
                "within_tolerance": within,
            },
            "calibration": calibration,
        },
        "search": {
            "mode": search,
            "top_k": top_k,
            "slots": slots,
            "batch_cap": batch_cap,
            "leaf_batch": leaf_batch,
            "age_deal_samples": age_deal_samples,
            "force_root_chance": force_root_chance,
        },
        "seed": seed,
        "seed_offset": ARENA_SEED_OFFSET,
        "device": device,
        "encoder_signature": ENCODER_SIGNATURE,
        "control_features_encoder": "on" if controls_on else "off",
        "seconds": elapsed,
        "play_seconds": play_seconds,
        "games_per_hour": (
            len(results) / play_seconds * 3600.0 if play_seconds > 0 else 0.0
        ),
        "hardware": hardware_identity(),
    }
    report["verdict"] = _verdict(report)
    if min_lcb is not None:
        report["min_lcb"] = min_lcb
        report["passed"] = lower > min_lcb
    return report


def _verdict(report: dict[str, Any]) -> str:
    lower = report["wilson"]["lower"]
    upper = report["wilson"]["upper"]
    rate = report["a_score_rate"]
    if lower > 0.50:
        headline = (
            f"A is stronger: {rate:.3f} with the interval entirely above 0.500"
        )
    elif upper < 0.50:
        headline = (
            f"A is weaker: {rate:.3f} with the interval entirely below 0.500"
        )
    else:
        headline = (
            f"no strength difference resolved at {report['pairs']} pairs: "
            f"{rate:.3f}, interval [{lower:.3f}, {upper:.3f}] contains 0.500"
        )
    parity = report["parity"]["measured"]
    if not parity["within_tolerance"]:
        headline += (
            f" -- but the sides did NOT play at equal wall-clock "
            f"(A/B = {parity['ratio_a_over_b']:.2f} seconds per move), so this "
            "compares policies at unequal compute"
        )
    return headline


# --- CLI ---------------------------------------------------------------------


def _int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Play two checkpoints head to head and report a Wilson interval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--a", type=Path, required=True, help="checkpoint under test")
    parser.add_argument("--b", type=Path, required=True, help="reference checkpoint")
    parser.add_argument("--pairs", type=int, required=True, help="seat-paired deals; 2 games each")
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="fp32", choices=("fp32", "bf16"))
    parser.add_argument("--sims", type=int, default=64, help="simulations per move")
    parser.add_argument("--sims-a", type=int, help="override A's budget (implies manual parity)")
    parser.add_argument("--sims-b", type=int, help="override B's budget (implies manual parity)")
    parser.add_argument(
        "--parity",
        default="sims",
        choices=("sims", "time"),
        help="equal simulations, or equal measured wall-clock per move",
    )
    parser.add_argument("--reference", default="a", choices=("a", "b"),
                        help="under --parity time, whose --sims sets the budget")
    parser.add_argument("--calibration-games", type=int, default=0,
                    help="concurrent games while calibrating; 0 follows --slots")
    parser.add_argument("--calibration-sims", type=_int_list,
                        help="comma-separated sims to fit the cost model over")
    parser.add_argument("--parity-tolerance", type=float, default=0.15,
                        help="how far the measured seconds/move ratio may sit from 1.0")
    parser.add_argument("--min-sims", type=int, default=0,
                        help="floor on a solved budget; 0 means --sims/8")
    parser.add_argument("--max-sims", type=int, default=0,
                        help="cap on a solved budget; 0 means --sims*8")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--search", default="gumbel", choices=("gumbel", "puct"))
    parser.add_argument("--slots", type=int, default=48, help="pairs played concurrently")
    parser.add_argument("--batch-cap", type=int, default=256)
    parser.add_argument("--leaf-batch", type=int, default=1)
    parser.add_argument("--age-deal-samples", type=int, default=32)
    parser.add_argument("--no-force-root-chance", dest="force_root_chance",
                        action="store_false")
    parser.add_argument("--max-moves", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--z", type=float, default=1.96)
    parser.add_argument("--migrate", action="store_true",
                        help="warm-start a checkpoint from an older schema (see --help)")
    parser.add_argument("--min-lcb", type=float,
                        help="exit 3 unless A's Wilson lower bound exceeds this")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    for label, path in (("--a", args.a), ("--b", args.b)):
        if not path.is_file():
            # argparse exits 2, which is how a caller separates "could not run"
            # from this tool's exit 3, "ran, and A did not clear the bar".
            parser.error(f"{label} {path} does not exist")
    if args.pairs <= 0:
        parser.error("--pairs must be positive")

    def log(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    report = run(
        args.a,
        args.b,
        pairs=args.pairs,
        device=args.device,
        precision=args.precision,
        sims=args.sims,
        sims_a=args.sims_a,
        sims_b=args.sims_b,
        parity=args.parity,
        reference=args.reference,
        calibration_games=args.calibration_games,
        calibration_sims=args.calibration_sims,
        parity_tolerance=args.parity_tolerance,
        min_sims=args.min_sims,
        max_sims=args.max_sims,
        top_k=args.top_k,
        search=args.search,
        slots=args.slots,
        batch_cap=args.batch_cap,
        leaf_batch=args.leaf_batch,
        force_root_chance=args.force_root_chance,
        age_deal_samples=args.age_deal_samples,
        max_moves=args.max_moves,
        seed=args.seed,
        z=args.z,
        migrate=args.migrate,
        min_lcb=args.min_lcb,
        log=log,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(format_summary(report))
    if args.min_lcb is not None and not report["passed"]:
        return GATE_FAILED_EXIT_CODE
    return 0


def format_summary(report: dict[str, Any]) -> str:
    wilson = report["wilson"]
    parity = report["parity"]["measured"]
    lines = [
        f"{report['a']['name']} vs {report['b']['name']}",
        f"  {report['pairs']} pairs ({report['games']} games), "
        f"{report['a']['sims']} vs {report['b']['sims']} sims, "
        f"{report['play_seconds'] / 60:.1f} min of play "
        f"({report['games_per_hour']:.0f} games/hour)",
        f"  A score rate {report['a_score_rate']:.3f} "
        f"[{wilson['lower']:.3f}, {wilson['upper']:.3f}] at z={wilson['z']} "
        f"against a null of 0.500",
        f"  pairs won/split/lost: {report['pair_wins']}/"
        f"{report['pair_splits']}/{report['pair_losses']}",
        f"  measured wall-clock per move: A {parity['a_seconds_per_move'] * 1000:.1f} ms, "
        f"B {parity['b_seconds_per_move'] * 1000:.1f} ms "
        f"(ratio {parity['ratio_a_over_b']:.2f}, "
        f"{'within' if parity['within_tolerance'] else 'OUTSIDE'} tolerance)",
        f"  {report['verdict']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
