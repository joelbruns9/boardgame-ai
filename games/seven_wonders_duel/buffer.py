"""Replayable game-buffer schema (CODEC_SPEC.md §6, plan A4).

One JSONL record per game. The defining invariant: ``replay(record)``
reproduces every game-logic state from ``(setup.seed, actions)`` — verified per
move against ``mask_hash`` and at the end against versioned final/trajectory
digests — so reanalyze, exact relabeling, and trap harvesting are derived
queries, never migrations. The ``chance_log`` is deliberately redundant with
the seed: a change to chance resolution breaks replay loudly instead of
silently corrupting old buffers. Legacy digests additionally include CPython's
RNG internals; current digests use the cross-language logic fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import struct

from .codec import decode_action, legal_action_indices
from .data import CARD_IDS, PROGRESS_IDS, WONDER_IDS, ScienceSymbol
from .engine import apply_action
from .game import (
    GameState,
    PendingChoiceKind,
    Phase,
    ResolvedChance,
    VictoryType,
    new_game,
)

SCHEMA_VERSION = 1
SPEC_VERSION = "codec-2"
LEGACY_DIGEST_VERSION = "python-rng-v1"
LOGIC_DIGEST_VERSION = "logic-sha256-v1"
"""Version of what a state and an action MEAN, for replay.

1. through 2026-08-02.
2. the next Age is dealt when the previous one is exhausted, before the
   military chooser is asked who starts it, instead of as a consequence of that
   choice (``ENGINE_AGE_DEAL_ORDERING.md``).  The AGE_DEAL chance event moves
   one move earlier, so the same seed produces a different game and every
   record written under spec 1 -- including the 84k-game laptop run -- is
   unreplayable rather than merely stale.
"""

TARGET_VERSION = 3
"""Version of the TRAINING TARGET definition, independent of the codec.

``schema``/``spec_version`` cover replay: what a state and an action mean.  They
deliberately do not move when the *labels* change, and that is a real gap --
2026-07-25's sigma fix rescaled completed Q to [0, 1] and dropped ``c_scale``
1.0 -> 0.1, which changes every ``policy_target`` value while leaving the codec
untouched.  Old records therefore stayed loadable and silently mixed two
incompatible definitions of the thing the policy head regresses onto.

1. completed Q with unnormalised sigma, ``c_visit=50``, ``c_scale=1.0``.  Sigma
   spanned +/-50 against log-prior differences of ~1-3, so the prior barely
   contributed to the target.
2. completed Q min-max rescaled to [0, 1] across the root's legal actions, then
   ``(c_visit + max_visits) * c_scale`` with ``c_scale=0.1`` (mctx's
   ``value_scale``).
3. the PUCT root's VISIT DISTRIBUTION, optionally with KataGo policy-target
   pruning applied (2026-08-17).  A different quantity entirely from
   definitions 1-2, not a rescaling of one: completed Q prices every legal
   action including unvisited ones, while visit counts price only what the
   search actually looked at.  Mixing the two in one buffer would train the
   policy head against two incompatible notions of "improved policy" with
   nothing in the row to say which it was holding.

   Rows also carry ``solver_masked`` (proof-masked policy) and, when pruning is
   on, a target whose forced visits have been removed -- both per-row rather
   than per-version, because they are legitimately mixed within a single run.

Bump this whenever the meaning of ``policy_target``, ``root_value`` or the
value label changes.  Replay is unaffected: because ``replay(record)``
reproduces states bit-exactly, a stale record can always be RE-derived rather
than discarded, so this gates training, not reading.
"""


def target_version_for_moves(moves) -> int:
    """The target definition a record's TRAINED rows actually hold.

    Stamping the module constant on every record is wrong in both directions.
    A Gumbel run writes completed-Q targets, which is definition 2 whatever this
    build is capable of -- so version 3 on those rows makes
    ``check_target_versions`` reject pre-bump Gumbel buffers whose targets are
    definitionally identical to today's, while simultaneously letting PUCT and
    Gumbel rows mix undetected, which is the exact thing the version exists to
    make structural rather than documented.

    Read off the rows that will actually be trained on -- searched, full-budget,
    learner-owned -- because the hybrid records BOTH kinds: its cheap moves
    carry Gumbel targets that the example boundary then drops. ``gumbel_topk``
    is the signature: the Gumbel root records its candidate set, the PUCT root
    deliberately records none rather than invent one.
    """

    trained = [
        move
        for move in moves
        if not move.policy_excluded and move.sims > 0 and move.gumbel_topk is not None
    ]
    if not trained:
        return TARGET_VERSION
    return 3 if any(len(move.gumbel_topk) == 0 for move in trained) else 2


@dataclass(frozen=True, slots=True)
class MoveRecord:
    i: int
    actor: int
    action: int
    mask_hash: str
    visits: dict[int, int] = field(default_factory=dict)
    policy_target: dict[int, float] | None = None
    """Improved (completed-Q) policy from Gumbel search — the preferred
    training target; visits are kept as raw search evidence for reanalyze."""
    root_value: float | None = None
    sims: int = 0
    mode: str = "simulator"
    gumbel_topk: tuple[int, ...] | None = None
    policy_excluded: bool = False
    solver_value: float | None = None
    """Exact value of the pre-move position, actor-relative, when the endgame
    solver reached one (``SOLVER_SELF_PLAY_PLAN.md``).  ``root_value`` is left
    alone: it stays the search's estimate, so the two remain the sampled and the
    proven answer to the same question."""
    solver_regime: str | None = None
    """``"exact"`` -- no chance edge was crossed, so ``solver_value`` is exactly
    -1, 0 or +1 and maps onto a W/D/L class -- or ``"exact_expectimax"``, whose
    scalar is ``P(win) - P(loss)`` and does NOT determine one."""
    solver_attempted: bool = False
    """A solve ran at this position.  Without it a missing ``solver_value``
    conflates three different things -- solver off, trigger not selected, solve
    declined -- so the declined positions could not be found in the buffer and
    the cost of failed attempts could not be measured."""
    solver_stop: str | None = None
    """``"unsolvable"`` (a sample-only Age deal, which no budget reaches) or
    ``"budget"``; ``None`` when the solve succeeded."""
    solver_nodes: int = 0
    """Nodes visited, including by an attempt that then declined."""
    solver_masked: bool = False
    """This row's ``policy_target`` has had its provably-losing moves zeroed and
    the survivors renormalised.

    Per-move rather than a ``TARGET_VERSION`` bump on purpose.  The bump would
    invalidate every record ever written in order to say something less precise:
    the definition of ``policy_target`` changed *for these rows only*, and a
    buffer mixing masked and unmasked rows is intended, not a defect.  A run
    that wants only one kind can filter on this flag."""


@dataclass(frozen=True, slots=True)
class GameRecord:
    seed: int
    first_player: int
    agents: dict[str, str]
    iteration: int | None
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
    target_version: int = TARGET_VERSION
    digest_version: str = LOGIC_DIGEST_VERSION
    _source_digest: str | None = field(
        default=None, init=False, compare=False, repr=False
    )
    """SHA-256 of the JSONL payload this object was parsed from.

    This is provenance, not part of the durable schema.  Training rereads the
    replay window every iteration, so retaining the digest computed while
    parsing avoids rebuilding and hashing the complete nested payload merely to
    look it up in the in-memory example cache.  ``dataclasses.replace`` does not
    copy ``init=False`` fields, which deliberately invalidates the provenance
    when any record field is changed.
    """

    @property
    def source_digest(self) -> str | None:
        return self._source_digest


class ReplayMismatchError(RuntimeError):
    """A recorded game no longer reproduces under the current engine."""


class StaleSpecVersionError(ReplayMismatchError):
    """A record predates a change to what a state or an action MEANS.

    Distinct from an ordinary mismatch: nothing is corrupt, the record simply
    describes a different game than this engine plays. Raised before replay
    starts so the reason is the version, not a downstream digest.
    """


OPPONENT_TYPES = ("current_best", "hof", "bot", "hof_bot")

def archive_policy_seats(agents: dict[str, str]) -> frozenset[int]:
    """Defense-in-depth exclusion for an archived net's policy targets.

    The production Rust recorder is the primary enforcement point:
    ``self_play.rs:1461`` in ``finish_move`` sets ``policy_excluded`` whenever
    the actor's network is not network 0 (the learner). This predicate protects
    imported,
    legacy, Python-written, or otherwise retagged records that do not carry that
    recorder-side exclusion.

    Deliberately **narrow**.  The obvious generalisation -- "exclude any seat
    ``agents["pN"]`` does not call ``network``" -- is wrong, and quietly so:
    curriculum-seed games name a scripted bot on *both* seats and record
    ``policy_excluded=False``, because imitating those bots is the entire point
    of the curriculum.  Widening this predicate would delete the curriculum's
    policy signal while every test still passed.  Only an archive assigned by
    ``_tag_league_opponents`` is excluded here, identified by the same three
    fields that function writes.

    Value labels are untouched as an explicit experimental choice. League games
    follow a mixed learner/archive policy, so those outcomes are observed labels
    but not unbiased estimates of current-policy self-play value. The settling
    experiment is all league values versus learner-turn values versus none.
    """

    if agents.get("kind") != "league":
        return frozenset()
    # An archive assigned to a bot-controlled seat never evaluated a move, and
    # `_tag_league_opponents` records that as `league_assignment_used=false`.
    if agents.get("league_assignment_used") != "true":
        return frozenset()
    name = agents.get("league_assignment")
    if name is None:
        return frozenset()
    return frozenset(seat for seat in (0, 1) if agents.get(f"p{seat}") == name)


def resolve_opponent_type(agents: dict[str, str]) -> str:
    """Return the explicit W3 opponent category, with legacy compatibility.

    New records carry ``opponent_type`` because ``kind`` cannot represent a
    HOF-vs-curriculum-bot game. Old buffers remain readable; their historical
    ``kind`` value is used when the explicit field is absent.
    """

    explicit = agents.get("opponent_type")
    if explicit is not None:
        if explicit not in OPPONENT_TYPES:
            raise ValueError(f"unknown opponent_type metadata: {explicit!r}")
        return explicit
    kind = agents.get("kind", "self_play")
    if kind == "league_mixed":
        return "hof_bot"
    if kind == "league":
        return "hof"
    if kind == "mixed" or kind == "bot" or "curriculum" in kind:
        return "bot"
    return "current_best"


def legal_mask_hash(legal: list[int] | tuple[int, ...]) -> str:
    payload = json.dumps(list(legal)).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()[:16]


def mask_hash(game: GameState) -> str:
    return legal_mask_hash(legal_action_indices(game))


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


_SCIENCE_ORDER = {symbol: index for index, symbol in enumerate(ScienceSymbol)}
_PHASE_ORDER = {member: index for index, member in enumerate(Phase)}
_VICTORY_ORDER = {member: index for index, member in enumerate(VictoryType)}
_PENDING_ORDER = {
    member: index for index, member in enumerate(PendingChoiceKind)
}
_PROGRESS_PENDING = {
    PendingChoiceKind.CHOOSE_UNUSED_PROGRESS,
    PendingChoiceKind.CHOOSE_AVAILABLE_PROGRESS,
}


def logic_fingerprint(game: GameState) -> list[int]:
    """Language-neutral integer serialization of all game-logic state.

    This is byte-for-byte identical to Rust ``GameState::fingerprint``. Python's
    RNG internals are deliberately excluded: all resolved randomness is already
    locked by setup plus ``chance_log``, while this surface covers the state that
    can affect subsequent game logic and encoded training positions.
    """

    out: list[int] = []

    def push_list(names, id_map) -> None:
        ids = [id_map[name] for name in names]
        out.append(len(ids))
        out.extend(ids)

    out.extend(
        (
            _PHASE_ORDER[game.phase],
            game.first_player,
            game.active_player,
            game.age,
            game.wonder_round,
            game.wonder_pick_index,
        )
    )
    for city in game.cities:
        out.append(city.coins)
        push_list(city.wonders, WONDER_IDS)
        push_list(city.built_wonders, WONDER_IDS)
        push_list(city.buildings, CARD_IDS)
        push_list(city.progress_tokens, PROGRESS_IDS)
        pairs = sorted(_SCIENCE_ORDER[symbol] for symbol in city.claimed_science_pairs)
        out.append(len(pairs))
        out.extend(pairs)

    push_list(game.available_progress_tokens, PROGRESS_IDS)
    push_list(game.unused_progress_tokens, PROGRESS_IDS)
    push_list(game.wonder_groups[0], WONDER_IDS)
    push_list(game.wonder_groups[1], WONDER_IDS)
    push_list(game.unused_wonders, WONDER_IDS)
    push_list(game.wonder_offer, WONDER_IDS)
    for age in (1, 2, 3):
        push_list(game.age_decks[age], CARD_IDS)
    for age in (1, 2, 3):
        push_list(game.removed_age_cards[age], CARD_IDS)
    push_list(game.selected_guilds, CARD_IDS)
    push_list(game.unused_guilds, CARD_IDS)

    slots = sorted(game.tableau.cards.items())
    out.append(len(slots))
    for (row, x), card in slots:
        out.extend(
            (
                row,
                x,
                CARD_IDS[card.card_name],
                int(card.present),
                int(card.present and card.revealed),
            )
        )
    push_list(game.discard_pile, CARD_IDS)
    push_list(game.buried_cards, CARD_IDS)

    burials = sorted(
        (WONDER_IDS[wonder], CARD_IDS[card])
        for wonder, card in game.wonder_burials.items()
    )
    out.append(len(burials))
    for wonder, card in burials:
        out.extend((wonder, card))
    retired = sorted(WONDER_IDS[wonder] for wonder in game.retired_wonders)
    out.append(len(retired))
    out.extend(retired)

    pending = game.pending_choice
    if pending is None:
        out.append(-1)
    else:
        out.extend(
            (
                _PENDING_ORDER[pending.kind],
                pending.player,
                int(pending.consume_all_options),
            )
        )
        id_map = PROGRESS_IDS if pending.kind in _PROGRESS_PENDING else CARD_IDS
        push_list(pending.options, id_map)
    out.extend((int(game.pending_extra_turn), game.pending_shields))

    out.append(game.conflict_position)
    military = sorted(game.military_tokens_remaining.items())
    out.append(len(military))
    for position, penalty in military:
        out.extend((position, penalty))
    out.append(-1 if game.winner is None else game.winner)
    out.append(
        -1 if game.victory_type is None else _VICTORY_ORDER[game.victory_type]
    )
    if game.final_scores is None:
        out.append(-1)
    else:
        out.extend((1, game.final_scores[0], game.final_scores[1]))
    return out


def _logic_frame(game: GameState) -> bytes:
    fingerprint = logic_fingerprint(game)
    return struct.pack(f"<I{len(fingerprint)}i", len(fingerprint), *fingerprint)


def logic_state_digest(game: GameState) -> str:
    return "sha256:" + hashlib.sha256(_logic_frame(game)).hexdigest()


def _update_trajectory(
    trajectory,
    game: GameState,
    digest_version: str,
) -> None:
    if digest_version == LOGIC_DIGEST_VERSION:
        trajectory.update(_logic_frame(game))
    elif digest_version == LEGACY_DIGEST_VERSION:
        trajectory.update(state_digest(game).encode())
    else:
        raise ValueError(f"unknown buffer digest version: {digest_version!r}")


def _final_digest(game: GameState, digest_version: str) -> str:
    if digest_version == LOGIC_DIGEST_VERSION:
        return logic_state_digest(game)
    if digest_version == LEGACY_DIGEST_VERSION:
        return state_digest(game)
    raise ValueError(f"unknown buffer digest version: {digest_version!r}")


def _log_entry(event: ResolvedChance) -> tuple[str, str | tuple[str, ...]]:
    return (event.kind.value, event.outcome)


class GameRecorder:
    """Drives one simulator game while building its buffer record.

    Usage: construct, then repeatedly ``play(action_index, **search_stats)``
    until ``game.phase is COMPLETE``, then ``finish()``.
    """

    def __init__(
        self,
        seed: int,
        first_player: int = 0,
        agents: dict[str, str] | None = None,
        iteration: int | None = None,
        digest_version: str = LOGIC_DIGEST_VERSION,
    ):
        self.seed = seed
        self.first_player = first_player
        self.agents = dict(agents) if agents is not None else {}
        self.iteration = iteration
        if digest_version not in (LOGIC_DIGEST_VERSION, LEGACY_DIGEST_VERSION):
            raise ValueError(f"unknown buffer digest version: {digest_version!r}")
        self.digest_version = digest_version
        self.game = new_game(seed, first_player=first_player)
        self._moves: list[MoveRecord] = []
        self._chance_log: list[tuple[str, str | tuple[str, ...]]] = []
        self._trajectory = hashlib.sha256()

    def play(
        self,
        action_index: int,
        *,
        visits: dict[int, int] | None = None,
        policy_target: dict[int, float] | None = None,
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
        _update_trajectory(self._trajectory, game, self.digest_version)
        move = MoveRecord(
            i=len(self._moves),
            actor=actor,
            action=action_index,
            mask_hash=mask_hash(game),
            visits=dict(visits) if visits is not None else {},
            policy_target=dict(policy_target) if policy_target is not None else None,
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
        final_digest = _final_digest(game, self.digest_version)
        _update_trajectory(self._trajectory, game, self.digest_version)
        return GameRecord(
            seed=self.seed,
            first_player=self.first_player,
            agents=self.agents,
            iteration=self.iteration,
            winner=game.winner,
            victory_type=game.victory_type.value if game.victory_type else None,
            scores=game.final_scores,
            chance_log=tuple(self._chance_log),
            moves=tuple(self._moves),
            final_digest=final_digest,
            trajectory_digest="sha256:" + self._trajectory.hexdigest(),
            digest_version=self.digest_version,
        )


def replay(record: GameRecord, on_state=None) -> GameState:
    """Re-run the game from (seed, actions), verifying masks, chance log, and
    the final digest. Raises ReplayMismatchError on any divergence.

    ``on_state(game, move)`` is invoked at every decision AFTER that move's
    integrity checks pass and BEFORE the move is applied — the hook consumers
    (featurization, reanalyze) use so they can never read from an unverified
    replay.

    A record written under a different ``spec_version`` is refused up front:
    the digests would fail anyway, but confusingly and only at the end.
    """

    if record.spec_version != SPEC_VERSION:
        raise StaleSpecVersionError(
            f"record was written under spec {record.spec_version!r}, this "
            f"engine is {SPEC_VERSION!r} — the game a seed produces has "
            "changed, so the record cannot be replayed"
        )
    game = new_game(record.seed, first_player=record.first_player)
    log_position = 0
    trajectory = hashlib.sha256()
    for move in record.moves:
        try:
            _update_trajectory(trajectory, game, record.digest_version)
        except ValueError as error:
            raise ReplayMismatchError(str(error)) from error
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
        if on_state is not None:
            on_state(game, move)
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
    try:
        digest = _final_digest(game, record.digest_version)
    except ValueError as error:
        raise ReplayMismatchError(str(error)) from error
    if digest != record.final_digest:
        raise ReplayMismatchError(
            f"final digest {digest} != recorded {record.final_digest}"
        )
    _update_trajectory(trajectory, game, record.digest_version)
    trajectory_digest = "sha256:" + trajectory.hexdigest()
    if trajectory_digest != record.trajectory_digest:
        raise ReplayMismatchError(
            f"trajectory digest {trajectory_digest} != recorded "
            f"{record.trajectory_digest}"
        )
    return game


# --- JSONL serialization ----------------------------------------------------


def _solver_fields(move: MoveRecord) -> dict:
    """The endgame-solver keys for one move, empty when it was never solved."""

    if not move.solver_attempted:
        return {}
    return {
        "solver_value": move.solver_value,
        "solver_regime": move.solver_regime,
        "solver_attempted": True,
        "solver_stop": move.solver_stop,
        "solver_nodes": move.solver_nodes,
        "solver_masked": move.solver_masked,
    }


def to_json_line(record: GameRecord) -> str:
    payload = {
        "schema": record.schema,
        "spec_version": record.spec_version,
        "target_version": record.target_version,
        "digest_version": record.digest_version,
        "setup": {"seed": record.seed, "first_player": record.first_player},
        "agents": record.agents,
        "iteration": record.iteration,
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
            # Solver fields are omitted when the solver did not answer, which is
            # every move of every record written before this feature and every
            # move of a run with it off.  Emitting them unconditionally would
            # change the bytes -- and so the source digest -- of records whose
            # content is identical, invalidating the example cache for a field
            # that carries no information.
            _solver_fields(move) | {
                "i": move.i,
                "actor": move.actor,
                "action": move.action,
                "mask_hash": move.mask_hash,
                "visits": {str(k): v for k, v in sorted(move.visits.items())},
                "policy_target": (
                    {str(k): v for k, v in sorted(move.policy_target.items())}
                    if move.policy_target is not None
                    else None
                ),
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
    serialized = line.rstrip("\r\n")
    payload = json.loads(serialized)
    if payload["schema"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported buffer schema: {payload['schema']}")
    result = payload["result"]
    record = GameRecord(
        schema=payload["schema"],
        spec_version=payload["spec_version"],
        # Absent means the file predates target versioning, which can only be
        # definition 1.  Reading stays permissive on purpose -- replay and
        # reanalyze must be able to open stale buffers to re-derive targets
        # from them.  Enforcement belongs at the training boundary.
        target_version=payload.get("target_version", 1),
        digest_version=payload.get("digest_version", LEGACY_DIGEST_VERSION),
        seed=payload["setup"]["seed"],
        first_player=payload["setup"]["first_player"],
        agents=dict(payload["agents"]),
        iteration=payload.get("iteration"),
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
                policy_target=(
                    {int(k): v for k, v in move["policy_target"].items()}
                    if move.get("policy_target") is not None
                    else None
                ),
                root_value=move["root_value"],
                sims=move["sims"],
                mode=move["mode"],
                gumbel_topk=tuple(move["gumbel_topk"])
                if move["gumbel_topk"] is not None
                else None,
                policy_excluded=move["policy_excluded"],
                solver_value=move.get("solver_value"),
                solver_regime=move.get("solver_regime"),
                solver_attempted=move.get("solver_attempted", False),
                solver_stop=move.get("solver_stop"),
                solver_nodes=move.get("solver_nodes", 0),
                solver_masked=move.get("solver_masked", False),
            )
            for move in payload["moves"]
        ),
        final_digest=payload["final_digest"],
        trajectory_digest=payload["trajectory_digest"],
    )
    object.__setattr__(
        record,
        "_source_digest",
        hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    )
    return record


def append_records(path, records) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(to_json_line(record) + "\n")


def read_records(path) -> list[GameRecord]:
    with open(path, "r", encoding="utf-8") as handle:
        return [from_json_line(line) for line in handle if line.strip()]


class StaleTargetsError(RuntimeError):
    """A buffer holds targets computed under a superseded definition."""


def target_version_census(records) -> dict[int, int]:
    """Count records per target definition."""

    census: dict[int, int] = {}
    for record in records:
        census[record.target_version] = census.get(record.target_version, 0) + 1
    return census


def check_target_versions(records, *, source: str, allow_stale: bool = False):
    """Refuse to train on targets that mean different things.

    Training a single head against two definitions of ``policy_target`` is
    silent corruption: nothing errors, the loss still falls, and the resulting
    net is fit to an average of two objectives.  The failure is invisible in
    every metric the loop reports, which is precisely why it needs to be
    structural rather than documented.

    Returns the census so callers can log what they accepted.
    """

    census = target_version_census(records)
    stale = {version: n for version, n in census.items() if version != TARGET_VERSION}
    if not stale:
        return census
    breakdown = ", ".join(
        f"{n} record(s) at target_version={version}" for version, n in sorted(stale.items())
    )
    if allow_stale:
        print(
            f"WARNING: {source} mixes target definitions ({breakdown}; current "
            f"is {TARGET_VERSION}) and stale targets were explicitly allowed. "
            f"The policy head will be fit to inconsistent labels."
        )
        return census
    raise StaleTargetsError(
        f"{source} contains {breakdown}, but the current target definition is "
        f"{TARGET_VERSION}. These were computed under a superseded rule and "
        f"mixing them trains the policy head on two different objectives. "
        f"Start from a fresh buffer, or pass allow_stale_targets to override "
        f"knowingly. See TARGET_VERSION in buffer.py for what changed."
    )
