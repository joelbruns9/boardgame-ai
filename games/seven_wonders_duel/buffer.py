"""Replayable game-buffer schema (CODEC_SPEC.md §6, plan A4).

One JSONL record per game. The defining invariant: ``replay(record)``
reproduces every state bit-exactly from ``(setup.seed, actions)`` — verified
per move against ``mask_hash`` and at the end against ``final_digest`` — so
reanalyze, exact relabeling, and trap harvesting are derived queries, never
migrations. The ``chance_log`` is deliberately redundant with the seed: any
change to engine RNG consumption breaks replay loudly instead of silently
corrupting old buffers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

from .codec import decode_action, legal_action_indices
from .engine import apply_action
from .game import GameState, Phase, ResolvedChance, new_game

SCHEMA_VERSION = 1
SPEC_VERSION = "codec-1"


@dataclass(frozen=True, slots=True)
class MoveRecord:
    i: int
    actor: int
    action: int
    mask_hash: str
    visits: dict[int, int] = field(default_factory=dict)
    root_value: float | None = None
    sims: int = 0
    mode: str = "simulator"
    gumbel_topk: tuple[int, ...] | None = None
    policy_excluded: bool = False


@dataclass(frozen=True, slots=True)
class GameRecord:
    seed: int
    first_player: int
    agents: dict[str, str]
    winner: int | None
    victory_type: str | None
    scores: tuple[int, int] | None
    chance_log: tuple[tuple[str, str | tuple[str, ...]], ...]
    moves: tuple[MoveRecord, ...]
    final_digest: str
    trajectory_digest: str
    """Chained sha256 over the pre-move state digest of every decision plus the
    final state — catches intermediate divergence that leaves the legal mask,
    the chance outcomes, and the final state unchanged."""
    schema: int = SCHEMA_VERSION
    spec_version: str = SPEC_VERSION


class ReplayMismatchError(RuntimeError):
    """A recorded game no longer reproduces under the current engine."""


def mask_hash(game: GameState) -> str:
    payload = json.dumps(legal_action_indices(game)).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()[:16]


def state_digest(game: GameState) -> str:
    """Canonical digest of the complete state — public, hidden deal, draft
    counters, and the RNG stream — so replay divergence anywhere is caught,
    including a change in engine RNG consumption that happens to produce the
    same visible outcomes."""

    cities = tuple(
        (
            city.coins,
            tuple(city.wonders),
            tuple(city.built_wonders),
            tuple(city.buildings),
            tuple(city.progress_tokens),
            tuple(sorted(s.value for s in city.claimed_science_pairs)),
        )
        for city in game.cities
    )
    tableau = tuple(
        (slot_id, card.card_name, card.present, card.revealed)
        for slot_id, card in sorted(game.tableau.cards.items())
    )
    payload = (
        game.phase.value,
        game.first_player,
        game.active_player,
        game.age,
        game.wonder_round,
        game.wonder_pick_index,
        cities,
        game.available_progress_tokens,
        game.unused_progress_tokens,
        game.wonder_groups,
        game.unused_wonders,
        tuple(game.wonder_offer),
        tuple((age, game.age_decks.get(age, ())) for age in (1, 2, 3)),
        tuple((age, game.removed_age_cards.get(age, ())) for age in (1, 2, 3)),
        game.selected_guilds,
        game.unused_guilds,
        tableau,
        tuple(game.discard_pile),
        tuple(game.buried_cards),
        tuple(sorted(game.wonder_burials.items())),
        tuple(sorted(game.retired_wonders)),
        (
            (
                game.pending_choice.kind.value,
                game.pending_choice.player,
                game.pending_choice.options,
                game.pending_choice.consume_all_options,
            )
            if game.pending_choice is not None
            else None
        ),
        game.pending_extra_turn,
        game.pending_shields,
        game.conflict_position,
        tuple(sorted(game.military_tokens_remaining.items())),
        game.winner,
        game.victory_type.value if game.victory_type is not None else None,
        game.final_scores,
        hashlib.sha256(str(game.rng.getstate()).encode()).hexdigest()[:16],
    )
    return "sha256:" + hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def _log_entry(event: ResolvedChance) -> tuple[str, str | tuple[str, ...]]:
    return (event.kind.value, event.outcome)


class GameRecorder:
    """Drives one simulator game while building its buffer record.

    Usage: construct, then repeatedly ``play(action_index, **search_stats)``
    until ``game.phase is COMPLETE``, then ``finish()``.
    """

    def __init__(self, seed: int, first_player: int = 0, agents: dict[str, str] | None = None):
        self.seed = seed
        self.first_player = first_player
        self.agents = dict(agents) if agents is not None else {}
        self.game = new_game(seed, first_player=first_player)
        self._moves: list[MoveRecord] = []
        self._chance_log: list[tuple[str, str | tuple[str, ...]]] = []
        self._trajectory = hashlib.sha256()

    def play(
        self,
        action_index: int,
        *,
        visits: dict[int, int] | None = None,
        root_value: float | None = None,
        sims: int = 0,
        mode: str = "simulator",
        gumbel_topk: tuple[int, ...] | None = None,
        policy_excluded: bool = False,
    ) -> None:
        game = self.game
        actor = (
            game.pending_choice.player
            if game.pending_choice is not None
            else game.active_player
        )
        self._trajectory.update(state_digest(game).encode())
        move = MoveRecord(
            i=len(self._moves),
            actor=actor,
            action=action_index,
            mask_hash=mask_hash(game),
            visits=dict(visits) if visits is not None else {},
            root_value=root_value,
            sims=sims,
            mode=mode,
            gumbel_topk=gumbel_topk,
            policy_excluded=policy_excluded,
        )
        result = apply_action(game, decode_action(game, action_index))
        self._moves.append(move)
        self._chance_log.extend(_log_entry(event) for event in result.events)

    def finish(self) -> GameRecord:
        game = self.game
        if game.phase is not Phase.COMPLETE:
            raise ValueError("cannot finish a record before the game is complete")
        final_digest = state_digest(game)
        self._trajectory.update(final_digest.encode())
        return GameRecord(
            seed=self.seed,
            first_player=self.first_player,
            agents=self.agents,
            winner=game.winner,
            victory_type=game.victory_type.value if game.victory_type else None,
            scores=game.final_scores,
            chance_log=tuple(self._chance_log),
            moves=tuple(self._moves),
            final_digest=final_digest,
            trajectory_digest="sha256:" + self._trajectory.hexdigest(),
        )


def replay(record: GameRecord) -> GameState:
    """Re-run the game from (seed, actions), verifying masks, chance log, and
    the final digest. Raises ReplayMismatchError on any divergence."""

    game = new_game(record.seed, first_player=record.first_player)
    log_position = 0
    trajectory = hashlib.sha256()
    for move in record.moves:
        trajectory.update(state_digest(game).encode())
        current_hash = mask_hash(game)
        if current_hash != move.mask_hash:
            raise ReplayMismatchError(
                f"move {move.i}: mask hash {current_hash} != recorded {move.mask_hash}"
            )
        actor = (
            game.pending_choice.player
            if game.pending_choice is not None
            else game.active_player
        )
        if actor != move.actor:
            raise ReplayMismatchError(
                f"move {move.i}: actor {actor} != recorded {move.actor}"
            )
        result = apply_action(game, decode_action(game, move.action))
        for event in result.events:
            if log_position >= len(record.chance_log):
                raise ReplayMismatchError(
                    f"move {move.i}: chance event beyond recorded log"
                )
            if _log_entry(event) != record.chance_log[log_position]:
                raise ReplayMismatchError(
                    f"move {move.i}: chance event {_log_entry(event)} != "
                    f"recorded {record.chance_log[log_position]}"
                )
            log_position += 1
    if log_position != len(record.chance_log):
        raise ReplayMismatchError("recorded chance log has unconsumed entries")
    if game.phase is not Phase.COMPLETE:
        raise ReplayMismatchError("replayed game did not complete")
    digest = state_digest(game)
    if digest != record.final_digest:
        raise ReplayMismatchError(
            f"final digest {digest} != recorded {record.final_digest}"
        )
    trajectory.update(digest.encode())
    trajectory_digest = "sha256:" + trajectory.hexdigest()
    if trajectory_digest != record.trajectory_digest:
        raise ReplayMismatchError(
            f"trajectory digest {trajectory_digest} != recorded "
            f"{record.trajectory_digest}"
        )
    return game


# --- JSONL serialization ----------------------------------------------------


def to_json_line(record: GameRecord) -> str:
    payload = {
        "schema": record.schema,
        "spec_version": record.spec_version,
        "setup": {"seed": record.seed, "first_player": record.first_player},
        "agents": record.agents,
        "result": {
            "winner": record.winner,
            "victory_type": record.victory_type,
            "scores": list(record.scores) if record.scores is not None else None,
        },
        "chance_log": [
            {"kind": kind, "outcome": list(outcome) if isinstance(outcome, tuple) else outcome}
            for kind, outcome in record.chance_log
        ],
        "moves": [
            {
                "i": move.i,
                "actor": move.actor,
                "action": move.action,
                "mask_hash": move.mask_hash,
                "visits": {str(k): v for k, v in sorted(move.visits.items())},
                "root_value": move.root_value,
                "sims": move.sims,
                "mode": move.mode,
                "gumbel_topk": list(move.gumbel_topk)
                if move.gumbel_topk is not None
                else None,
                "policy_excluded": move.policy_excluded,
            }
            for move in record.moves
        ],
        "final_digest": record.final_digest,
        "trajectory_digest": record.trajectory_digest,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def from_json_line(line: str) -> GameRecord:
    payload = json.loads(line)
    if payload["schema"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported buffer schema: {payload['schema']}")
    result = payload["result"]
    return GameRecord(
        schema=payload["schema"],
        spec_version=payload["spec_version"],
        seed=payload["setup"]["seed"],
        first_player=payload["setup"]["first_player"],
        agents=dict(payload["agents"]),
        winner=result["winner"],
        victory_type=result["victory_type"],
        scores=tuple(result["scores"]) if result["scores"] is not None else None,
        chance_log=tuple(
            (
                entry["kind"],
                tuple(entry["outcome"])
                if isinstance(entry["outcome"], list)
                else entry["outcome"],
            )
            for entry in payload["chance_log"]
        ),
        moves=tuple(
            MoveRecord(
                i=move["i"],
                actor=move["actor"],
                action=move["action"],
                mask_hash=move["mask_hash"],
                visits={int(k): v for k, v in move["visits"].items()},
                root_value=move["root_value"],
                sims=move["sims"],
                mode=move["mode"],
                gumbel_topk=tuple(move["gumbel_topk"])
                if move["gumbel_topk"] is not None
                else None,
                policy_excluded=move["policy_excluded"],
            )
            for move in payload["moves"]
        ),
        final_digest=payload["final_digest"],
        trajectory_digest=payload["trajectory_digest"],
    )


def append_records(path, records) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(to_json_line(record) + "\n")


def read_records(path) -> list[GameRecord]:
    with open(path, "r", encoding="utf-8") as handle:
        return [from_json_line(line) for line in handle if line.strip()]
