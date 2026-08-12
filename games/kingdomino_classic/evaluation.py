"""Seat-balanced multiplayer tournaments for Classic Kingdomino baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Callable, Iterable, Sequence

from .baselines import Bot
from .config import ClassicGameConfig
from .game import ClassicGameState, GameResult


@dataclass(frozen=True, slots=True)
class Participant:
    name: str
    make_bot: Callable[[], Bot]


@dataclass(frozen=True, slots=True)
class GameRecord:
    seed: int
    rotation: int
    start_player: int
    seats: tuple[str, ...]
    scores: tuple[int, ...]
    ranks: tuple[int, ...]
    winners: tuple[int, ...]
    win_shares: tuple[float, ...]
    forced_discards: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Standing:
    name: str
    seat_games: int
    win_share: float
    average_rank: float
    average_score: float
    last_place_rate: float
    forced_discard_rate: float
    seat_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TournamentReport:
    config_key: str
    players: int
    games: int
    records: tuple[GameRecord, ...]
    standings: tuple[Standing, ...]

    def standing(self, name: str) -> Standing:
        for standing in self.standings:
            if standing.name == name:
                return standing
        raise KeyError(name)

    @property
    def equal_strength_win_share(self) -> float:
        """Expected per-seat win share when all seats are equally strong."""

        return 1.0 / self.players

    def win_share_lift(self, name: str) -> float:
        return self.standing(name).win_share - self.equal_strength_win_share

    def to_dict(self) -> dict[str, object]:
        return {
            "config_key": self.config_key,
            "players": self.players,
            "games": self.games,
            "equal_strength_win_share": self.equal_strength_win_share,
            "standings": [asdict(standing) for standing in self.standings],
            "records": [asdict(record) for record in self.records],
        }

    def summary_dict(self) -> dict[str, object]:
        return {
            "config_key": self.config_key,
            "players": self.players,
            "games": self.games,
            "equal_strength_win_share": self.equal_strength_win_share,
            "standings": [asdict(standing) for standing in self.standings],
        }


def play_game(
    participants: Sequence[Participant],
    *,
    config: ClassicGameConfig,
    seed: int,
    start_player: int = 0,
    rotation: int = 0,
) -> GameRecord:
    if len(participants) != config.players:
        raise ValueError(
            f"Expected {config.players} seated participants; got {len(participants)}."
        )
    bots = [participant.make_bot() for participant in participants]
    state = ClassicGameState.new(
        seed=seed, config=config, start_player=start_player
    )
    # Keep tie-breaking reproducible and distinct from deck shuffling.  One
    # stream per seat prevents a bot's tie count from perturbing its rivals.
    rngs = [
        random.Random(
            (seed + 1) * 1_000_003
            + rotation * 97_409
            + (seat + 1) * 19_349_663
        )
        for seat in range(config.players)
    ]
    while not state.is_terminal:
        actor = state.current_actor
        actions = state.legal_actions()
        action = bots[actor].choose_action(state, actions, rng=rngs[actor])
        if action not in actions:
            raise ValueError(
                f"Participant {participants[actor].name!r} returned an illegal action."
            )
        state = state.step(action)

    result: GameResult = state.result()
    return GameRecord(
        seed=seed,
        rotation=rotation,
        start_player=start_player,
        seats=tuple(participant.name for participant in participants),
        scores=result.scores,
        ranks=result.ranks,
        winners=result.winners,
        win_shares=result.win_shares,
        forced_discards=tuple(len(ids) for ids in state.forced_discards),
    )


def _rotated(
    participants: Sequence[Participant], rotation: int
) -> tuple[Participant, ...]:
    size = len(participants)
    return tuple(participants[(seat - rotation) % size] for seat in range(size))


def run_seat_balanced_tournament(
    participants: Sequence[Participant],
    *,
    config: ClassicGameConfig,
    seeds: Iterable[int],
) -> TournamentReport:
    """Play every lineup rotation on each identical seeded deck.

    Seat zero is the starting player.  Rotating the lineup through every seat
    therefore gives each participant one start and one appearance in every seat
    per seed.  Repeated names are aggregated, which supports one-challenger
    versus an otherwise homogeneous field.
    """

    if len(participants) != config.players:
        raise ValueError(
            f"Expected {config.players} lineup entries; got {len(participants)}."
        )
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("A tournament requires at least one seed.")

    records: list[GameRecord] = []
    for seed in seed_values:
        for rotation in range(config.players):
            seated = _rotated(participants, rotation)
            records.append(
                play_game(
                    seated,
                    config=config,
                    seed=seed,
                    start_player=0,
                    rotation=rotation,
                )
            )

    names = sorted({participant.name for participant in participants})
    standings: list[Standing] = []
    for name in names:
        seat_games = 0
        win_share = 0.0
        rank_sum = 0.0
        score_sum = 0.0
        last_places = 0
        forced_discards = 0
        seat_counts = [0] * config.players
        for record in records:
            lowest_key = min(
                (record.scores[seat], -record.ranks[seat])
                for seat in range(config.players)
            )
            for seat, seat_name in enumerate(record.seats):
                if seat_name != name:
                    continue
                seat_games += 1
                seat_counts[seat] += 1
                win_share += record.win_shares[seat]
                rank_sum += record.ranks[seat]
                score_sum += record.scores[seat]
                if (record.scores[seat], -record.ranks[seat]) == lowest_key:
                    last_places += 1
                forced_discards += record.forced_discards[seat]
        if seat_games == 0:
            continue
        standings.append(
            Standing(
                name=name,
                seat_games=seat_games,
                win_share=win_share / seat_games,
                average_rank=rank_sum / seat_games,
                average_score=score_sum / seat_games,
                last_place_rate=last_places / seat_games,
                forced_discard_rate=forced_discards
                / (seat_games * config.dominoes_per_player),
                seat_counts=tuple(seat_counts),
            )
        )
    standings.sort(
        key=lambda standing: (
            -standing.win_share,
            standing.average_rank,
            -standing.average_score,
            standing.name,
        )
    )
    return TournamentReport(
        config_key=config.configuration_key,
        players=config.players,
        games=len(records),
        records=tuple(records),
        standings=tuple(standings),
    )


def run_challenger_tournament(
    challenger: Participant,
    field: Participant,
    *,
    config: ClassicGameConfig,
    seeds: Iterable[int],
) -> TournamentReport:
    lineup = [challenger, *([field] * (config.players - 1))]
    return run_seat_balanced_tournament(lineup, config=config, seeds=seeds)
