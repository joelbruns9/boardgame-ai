"""
Near-completion plan curriculum: restart real games a few turns before the
learner finished a City Plan.

WHY
───
v3_random_01 (2026-09-30 measurement, iterations 7-8): estate plans stay
feasible to the end -- 80% are still reachable but unfinished when the game
ends -- and only 9% are completed.  The learner rarely sees a plan finished, so
the plan heads sit at base rate and the search has nothing to back up.  The
remedy the reviewer proposed is a curriculum that starts close to success:
positions a few useful decisions from a completion, mixed into ordinary games.

WHERE THE POSITIONS COME FROM
────────────────────────────
Not constructed.  A constructed near-complete sheet has to be legal (ascending
streets, fences, estates, the deck it implies) and would still be a
distribution nobody plays.  Instead: take games from the previous iteration in
which the **learner** (seat 0) completed a plan on turn ``C``, rewind to the
start of turn ``C - k`` for ``k`` in :data:`DEFAULT_DISTANCES`, reshuffle the
undrawn deck (:meth:`GameState.redeterminize`), and play on from there with the
current learner and league.  Every start position is one a real game reached,
and the continuation is an ordinary game: nothing forces the plan to complete
again.

Only the learner's completions are used because the learner is always seat 0;
relabelling seats would be a deep engine change.

CALIBRATION
───────────
The start positions are selected (near a completion); the continuations are
not.  So the curriculum shifts which states are sampled but not what happens
from them -- a plan head trained on these rows still predicts completion
*given the state*, which is the quantity it is asked for.  The reviewer's
warning about inflated completion probabilities applies to oversampling
*outcomes*; this oversamples *states*.

WHAT A RESTART RECORDS
──────────────────────
:class:`Restart` on :class:`self_play.SelfPlayTrajectory`: the source game's
engine seed, the decision index ``at`` where the deck was reshuffled, the
reshuffle seed, and (for reporting) the rewind distance and plan slot.  The
trajectory's ``actions`` are the whole sequence -- prefix then continuation --
so both replays (Python :func:`self_play.replay`, Rust capture ``finish``)
reproduce it without the source corpus.  Search targets exist only at or after
``at``: the prefix was played by an older generation.

The trajectory's own ``seed`` stays the unique job identity; the engine is built
from ``source_seed``.  One source game can feed several restarts, and a shared
seed would have given them the same search and opponent streams and collided in
resume.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

from games.welcome_to.portable_rng import PortableRng

#: Rewind distances in turns, sampled uniformly per restart.  1 is "the plan is
#: one turn away", 8 is about a third of a learner game.
DEFAULT_DISTANCES: tuple[int, ...] = (1, 2, 4, 8)

_MASK64 = (1 << 64) - 1
_SELECT_DOMAIN = 0x6375_7272_6963_756C  # "curricul"
_RESHUFFLE_DOMAIN = 0x7265_7368_7566_666C  # "reshuffl"


@dataclass(frozen=True, slots=True)
class Restart:
    """Where a curriculum game left its source game."""

    source_seed: int
    at: int
    reshuffle_seed: int
    distance: int
    slot: int
    #: The source game's forced City Plan deal, or ``()`` if it was natural.
    plan_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.at <= 0:
            raise ValueError("a restart needs a non-empty prefix")
        if self.distance <= 0 or not 0 <= self.slot < 3:
            raise ValueError("restart distance must be positive and slot in 0..2")


@dataclass(frozen=True, slots=True)
class Candidate:
    """A learner plan completion in a source game, rewound ``distance`` turns."""

    source_seed: int
    players: int
    at: int
    distance: int
    slot: int
    prefix: tuple[int, ...]
    plan_ids: tuple[int, ...] = ()


def candidates(
    trajectories: Sequence, distances: Sequence[int] = DEFAULT_DISTANCES
) -> list[Candidate]:
    """Every (learner completion, distance) rewind point in ``trajectories``.

    A rewind point is the first decision of turn ``C - k``, which is always the
    learner's (seat 0 acts first) at a turn start.  Distances that would rewind
    to before the first turn are skipped.  Restart games are not used as
    sources: their prefixes belong to an older generation twice over.
    """
    import welcome_to_rust as wr

    out: list[Candidate] = []
    for trajectory in trajectories:
        if trajectory.restart is not None:
            continue
        state = trajectory.new_rust_state()
        first_decision_of_turn: dict[int, int] = {}
        for decision, action in enumerate(trajectory.actions):
            if state.turn not in first_decision_of_turn:
                first_decision_of_turn[state.turn] = decision
                if state.actor != 0:
                    raise ValueError(
                        f"turn {state.turn} of seed {trajectory.seed} starts with "
                        f"seat {state.actor}, not the learner"
                    )
            state.apply_macro(action)
        if not state.is_terminal:
            raise ValueError(f"source seed {trajectory.seed} does not finish")
        first_turn = min(first_decision_of_turn)
        for slot in range(3):
            completed = [turn for seat, turn in state.plan_turns_for(0, slot) if seat == 0]
            if not completed:
                continue
            for distance in distances:
                turn = completed[0] - distance
                if turn < first_turn or turn not in first_decision_of_turn:
                    continue
                at = first_decision_of_turn[turn]
                if at == 0:
                    continue
                out.append(
                    Candidate(
                        source_seed=trajectory.seed,
                        players=trajectory.players,
                        at=at,
                        distance=distance,
                        slot=slot,
                        prefix=tuple(trajectory.actions[:at]),
                        plan_ids=tuple(trajectory.plan_ids or ()),
                    )
                )
    return out


def plan_restarts(
    jobs: Sequence[tuple[int, int]],
    pool: Sequence[Candidate],
    *,
    fraction: float,
    seed: int,
) -> dict[int, tuple[Restart, tuple[int, ...]]]:
    """Assign restarts to a deterministic ``fraction`` of ``jobs``.

    ``jobs`` is ``(job seed, players)``.  A chosen job keeps its seat count and
    draws a candidate with the same count, so the 60/30/10 seat mix and resume
    validation are untouched; a job with no same-count candidate stays an
    ordinary game.  Returns ``job seed -> (restart, prefix)``.

    Deterministic in ``(jobs, pool, fraction, seed)``, which is what lets a
    resumed generation rebuild the identical plan.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("restart fraction must be in [0, 1]")
    if fraction == 0.0 or not pool:
        return {}
    rng = PortableRng((seed ^ _SELECT_DOMAIN) & _MASK64)
    ordered = sorted(jobs)
    chosen = ordered[:]
    rng.shuffle(chosen)
    chosen = sorted(chosen[: int(len(ordered) * fraction + 0.5)])
    by_players: dict[int, list[Candidate]] = {}
    for candidate in pool:
        by_players.setdefault(candidate.players, []).append(candidate)
    for group in by_players.values():
        group.sort(key=lambda c: (c.source_seed, c.slot, c.distance))
    plan: dict[int, tuple[Restart, tuple[int, ...]]] = {}
    for job_seed, players in chosen:
        group = by_players.get(players)
        if not group:
            continue
        candidate = group[rng.randrange(len(group))]
        reshuffle = PortableRng((job_seed ^ _RESHUFFLE_DOMAIN) & _MASK64).next_u64()
        plan[job_seed] = (
            Restart(
                source_seed=candidate.source_seed,
                at=candidate.at,
                reshuffle_seed=reshuffle,
                distance=candidate.distance,
                slot=candidate.slot,
                plan_ids=candidate.plan_ids,
            ),
            candidate.prefix,
        )
    return plan


def plan_digest(plan: Mapping[int, tuple[Restart, tuple[int, ...]]]) -> str:
    """A stable identity for a restart plan, for the resume manifest."""
    rows = [
        [seed, asdict(restart), list(prefix)]
        for seed, (restart, prefix) in sorted(plan.items())
    ]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def report(trajectories: Sequence, final_states: Sequence) -> dict[str, float]:
    """Curriculum metrics: restart games, and how often the learner finished the
    source plan again, overall and per rewind distance."""
    out: dict[str, float] = {}
    rows: dict[int, list[bool]] = {}
    plans = 0
    for trajectory, state in zip(trajectories, final_states):
        restart: Optional[Restart] = trajectory.restart
        if restart is None:
            continue
        done = 0 in state.plan_turns[restart.slot]
        rows.setdefault(restart.distance, []).append(done)
        plans += sum(1 for slot in state.plan_turns if 0 in slot)
    games = sum(len(v) for v in rows.values())
    out["curriculum_games"] = float(games)
    if games:
        out["curriculum_source_plan_rate"] = sum(sum(v) for v in rows.values()) / games
        out["curriculum_learner_plans_per_game"] = plans / games
        for distance, values in sorted(rows.items()):
            out[f"curriculum_games_d{distance}"] = float(len(values))
            out[f"curriculum_source_plan_rate_d{distance}"] = sum(values) / len(values)
    return out
