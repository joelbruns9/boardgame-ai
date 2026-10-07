"""G12 (`MODEL_GROWTH_PLAN.md`): restart self-play games from archived critical
positions, forcing a branch the history did not take.

Self-play rarely reaches the same critical situation twice, so the lesson of a
decisive moment is learnt from one sample. The archive keeps positions shortly
before decisive moments -- a forced win, a forced loss or a must-block is
available (`classify_actions`) -- and a share of each iteration's games restart
from them:

* the ancestor's moves are replayed from its seed, so the restart reuses its
  HIDDEN DEAL (owner decision: a reshuffle would need a new record format).
  Every restart record still replays from ``(seed, first_player, actions)``;
* the restart position's move is drawn from the search's own target with every
  move already tried there removed (`first_move_excludes`), so each restart
  explores the next-best branch rather than replaying history. Its training
  target is untouched;
* the record carries ``restart_from``: derivation trains only the new moves,
  and on move targets and search values but not the realised result, which
  reuses the ancestor's deal (`dataset.Example.outcome_free`).

The restart point sits ``0..window`` plies before a decisive position, drawn
uniformly: a backward curriculum in the plan's sense (restart before the
decisive choice as well as at it). An entry is restarted at most
``max_restarts`` times, each excluding the moves already tried, and ages out.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random

from .buffer import GameRecord, MoveRecord, record_digests, replay
from .game import Phase

DEFAULT_WINDOW = 8
DEFAULT_PER_GAME = 2
DEFAULT_MAX_RESTARTS = 3
DEFAULT_MAX_AGE = 10


@dataclass
class Entry:
    """One archived restart position."""

    seed: int
    first_player: int
    #: The ancestor's moves before the restart position: `(action, actor, mask_hash)`.
    prefix: list
    #: The ancestor's chance events consumed by those moves, in record form.
    chance_prefix: list
    #: Moves already played at the restart position (history, then restarts).
    tried: list
    born: int
    #: `(ancestor iteration, decisive ply, plies before it)`, for provenance.
    source: list
    restarts: int = 0
    legal_count: int = 0
    #: The ancestor game's winner, so a restart can report whether its forced
    #: branch ended differently.
    ancestor_winner: int | None = None

    @property
    def ply(self) -> int:
        return len(self.prefix)

    @property
    def key(self) -> str:
        actions = ",".join(str(a) for a, _actor, _hash in self.prefix)
        return f"{self.seed}:{self.first_player}:{actions}"


def _decisive_plies(record: GameRecord) -> tuple[list[int], dict]:
    """`(decisive move indices, {move index: (chance events so far, legal count)})`."""

    from .rust_bridge import rust_game_from_state

    from .codec import legal_action_indices

    decisive: list[int] = []
    info: dict = {}
    consumed = {"events": 0}
    start = record.restart_from or 0

    def visit(state, move) -> None:
        info[move.i] = (consumed["events"], len(legal_action_indices(state)))
        if move.i < start or state.phase is Phase.WONDER_DRAFT:
            return
        if any(label != 0 for label in rust_game_from_state(state).classify_actions()):
            decisive.append(move.i)

    def count(_move, events) -> None:
        consumed["events"] += len(events)

    replay(record, on_state=visit, on_events=count)
    return decisive, info


def harvest(
    records,
    iteration: int,
    *,
    window: int = DEFAULT_WINDOW,
    per_game: int = DEFAULT_PER_GAME,
    seed: int = 0,
) -> list[Entry]:
    """Restart positions from one iteration's games. Curriculum-bot games are
    skipped (their moves are not the policy being trained)."""

    out: list[Entry] = []
    for record in records:
        if record.agents.get("opponent_type") == "bot":
            continue
        rng = random.Random(f"{seed}:{record.iteration}:{record.seed}")
        decisive, info = _decisive_plies(record)
        # One restart per decisive STRETCH: consecutive decisive plies share it.
        stretches = [d for k, d in enumerate(decisive) if k == 0 or d - decisive[k - 1] > window]
        rng.shuffle(stretches)
        floor = record.restart_from or 0
        taken = 0
        for d in stretches:
            if taken >= per_game:
                break
            back = rng.randint(0, window)
            ply = max(floor, d - back)
            move = record.moves[ply]
            events, legal = info[ply]
            if legal < 2:
                continue  # one legal move: no branch to explore
            prefix = [(m.action, m.actor, m.mask_hash) for m in record.moves[:ply]]
            chance = [
                [kind, list(outcome) if isinstance(outcome, tuple) else outcome]
                for kind, outcome in record.chance_log[:events]
            ]
            out.append(Entry(
                seed=record.seed, first_player=record.first_player, prefix=prefix,
                chance_prefix=chance, tried=[move.action], born=iteration,
                source=[record.iteration, d, d - ply], legal_count=legal,
                ancestor_winner=record.winner,
            ))
            taken += 1
    return out


@dataclass
class Archive:
    """The run's restart positions, persisted next to the checkpoints."""

    entries: dict = field(default_factory=dict)
    max_restarts: int = DEFAULT_MAX_RESTARTS
    max_age: int = DEFAULT_MAX_AGE

    def add(self, entries) -> int:
        added = 0
        for entry in entries:
            if entry.key not in self.entries:
                self.entries[entry.key] = entry
                added += 1
        return added

    def prune(self, iteration: int) -> int:
        """Drop entries that are spent: restarted ``max_restarts`` times, every
        legal move tried, or older than ``max_age`` iterations."""

        stale = [
            key for key, entry in self.entries.items()
            if entry.restarts >= self.max_restarts
            or (entry.legal_count and len(entry.tried) >= entry.legal_count)
            or iteration - entry.born > self.max_age
        ]
        for key in stale:
            del self.entries[key]
        return len(stale)

    def draw(self, count: int, rng: random.Random) -> list[Entry]:
        """Up to ``count`` distinct entries, uniformly; each is charged one restart."""

        pool = sorted(self.entries)
        chosen = rng.sample(pool, min(count, len(pool)))
        out = []
        for key in chosen:
            entry = self.entries[key]
            entry.restarts += 1
            out.append(entry)
        return out

    def note_played(self, entry: Entry, action: int) -> None:
        """Record the move a restart actually played at the entry's position."""

        if entry.key in self.entries and action not in entry.tried:
            entry.tried.append(action)

    def save(self, path: Path) -> None:
        payload = {
            "max_restarts": self.max_restarts,
            "max_age": self.max_age,
            "entries": [asdict(entry) for entry in self.entries.values()],
        }
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, **defaults) -> "Archive":
        path = Path(path)
        if not path.exists():
            return cls(**defaults)
        payload = json.loads(path.read_text(encoding="utf-8"))
        archive = cls(max_restarts=payload["max_restarts"], max_age=payload["max_age"])
        for raw in payload["entries"]:
            entry = Entry(**raw)
            archive.entries[entry.key] = entry
        return archive


def restart_game(entry: Entry):
    """The Rust game at ``entry``'s position, reached by replaying the
    ancestor's prefix from its seed, with every Great Library draw locked: the
    prefix's own, plus one for the continuation sampled from the replayed
    engine's RNG -- exactly what `rust_game_for_self_play` locks for a fresh
    game, and what the engine would draw if the Library is built later."""

    import seven_wonders_rust as swr

    from .codec import decode_action
    from .data import PROGRESS_IDS
    from .engine import apply_action
    from .game import new_game
    from .rust_bridge import rust_setup

    actions = [action for action, _actor, _hash in entry.prefix]
    fresh = new_game(entry.seed, first_player=entry.first_player)
    setup = rust_setup(fresh)
    replayed = fresh.clone()
    draws: list[list[str]] = []
    for index in actions:
        action = decode_action(replayed, index)
        apply_action(replayed, action)
        if action.wonder_name == "The Great Library" and replayed.pending_choice is not None:
            draws.append(list(replayed.pending_choice.options))
    count = min(3, len(replayed.unused_progress_tokens))
    if count:
        draw = replayed.rng.sample(replayed.unused_progress_tokens, count)
        draws.append(sorted(draw, key=PROGRESS_IDS.__getitem__))
    game = swr.RustGame(library_draws=draws, **setup)
    for index in actions:
        game.apply_index(index)
    return game


def merge_record(entry: Entry, continuation: GameRecord, iteration: int | None) -> GameRecord:
    """One replayable record: the ancestor's prefix (replay-only rows) plus the
    restart's continuation, with ``restart_from`` at the archived position."""

    ply = entry.ply
    prefix_moves = [
        MoveRecord(
            i=i, actor=actor, action=action, mask_hash=mask_hash,
            visits={}, policy_target=None, root_value=None, sims=0,
            mode="restart_prefix", gumbel_topk=None, policy_excluded=True,
        )
        for i, (action, actor, mask_hash) in enumerate(entry.prefix)
    ]
    import dataclasses

    moves = prefix_moves + [dataclasses.replace(m, i=m.i + ply) for m in continuation.moves]
    chance_log = tuple(
        (kind, tuple(outcome) if isinstance(outcome, list) else outcome)
        for kind, outcome in entry.chance_prefix
    ) + tuple(continuation.chance_log)
    final, trajectory = record_digests(
        entry.seed, entry.first_player, [m.action for m in moves],
        continuation.digest_version,
    )
    agents = dict(continuation.agents)
    agents["restart_of"] = f"{entry.source[0]}:{entry.seed}:{ply}"
    return GameRecord(
        seed=entry.seed,
        first_player=entry.first_player,
        agents=agents,
        iteration=iteration,
        winner=continuation.winner,
        victory_type=continuation.victory_type,
        scores=continuation.scores,
        chance_log=chance_log,
        moves=tuple(moves),
        final_digest=final,
        trajectory_digest=trajectory,
        target_version=continuation.target_version,
        digest_version=continuation.digest_version,
        restart_from=ply,
    )
