"""Offline decision contract shared by turn-table and future rollout backends.

No game RNG is accepted by an evaluator. Force a roll without drawing dice;
call engine.roll with the caller's chosen dice afterward.
"""
from dataclasses import dataclass
import hashlib
from typing import Protocol

from .engine import Phase, apply_move, can_stop, legal_moves, stop
from .portable_rng import PortableRng


@dataclass(frozen=True)
class Action:
    kind: str
    move: tuple[int, ...] = ()

    @property
    def key(self):
        return self.kind + (':' + ','.join(map(str, self.move)) if self.move else '')


def winning_bank(state):
    if not can_stop(state):
        return False
    child = state.clone()
    stop(child)
    return child.game_over


def actions(state):
    """Canonical choices. Omit dominated rolling when banking wins exactly.

    The engine permits declining a winning bank, but the turn solvers represent
    that node as terminal. This interface adopts the same stop-only convention.
    """
    if state.game_over:
        return ()
    if state.phase == Phase.AWAIT_MOVE:
        seen, result = set(), []
        for move in legal_moves(state, state.dice):
            child = state.clone()
            apply_move(child, move)
            effect = tuple(sorted(child.runners.items()))
            if effect not in seen:
                seen.add(effect)
                result.append(Action('move', tuple(move)))
        return tuple(result)
    if state.phase == Phase.AWAIT_DECISION:
        return ((Action('stop'),) if winning_bank(state) else
                ((Action('stop'), Action('roll')) if can_stop(state)
                 else (Action('roll'),)))
    if state.phase == Phase.AWAIT_ROLL:
        return (Action('roll'),)
    raise ValueError('unsupported decision phase')


def force_action(state, action):
    """Return an independent state after a legal choice, before any new dice."""
    if action not in actions(state):
        raise ValueError(f'illegal action: {action}')
    child = state.clone()
    if action.kind == 'move':
        apply_move(child, action.move)
    elif action.kind == 'stop':
        stop(child)
    else:
        child.phase, child.dice = Phase.AWAIT_ROLL, None
    return child


@dataclass(frozen=True)
class ActionValue:
    action: Action
    value: tuple[float, ...]


@dataclass(frozen=True)
class Decision:
    actor: int
    value: tuple[float, ...]
    selected: Action | None
    options: tuple[ActionValue, ...]

    def to_dict(self):
        return {'actor': self.actor, 'value': list(self.value),
                'selected': None if self.selected is None else self.selected.key,
                'options': [{'action': x.action.key, 'value': list(x.value)}
                            for x in self.options]}


class DecisionBackend(Protocol):
    """Future rollout implementations provide this same entry point."""
    def evaluate(self, state) -> Decision: ...


class TurnTableBackend:
    """Adapt baseline or selective search without changing their tie-breaking."""
    def __init__(self, evaluator, *, search_config=None):
        self.evaluator = evaluator
        self.search_config = search_config

    def evaluate(self, state):
        actor = state.active_player
        if state.game_over:
            return Decision(actor, tuple(float(p == state.winner)
                            for p in range(state.rules.num_players)), None, ())
        if self.search_config is None:
            from .rust_solver import RustTurnSolver
            solver = RustTurnSolver(state, self.evaluator)
        else:
            from .turn_search import WholeTurnSearch
            solver = WholeTurnSearch(state, self.evaluator, self.search_config)
        choices = actions(state)
        if not choices:
            raise ValueError('move phase has no legal actions; use engine.roll to resolve bust')
        if state.phase == Phase.AWAIT_MOVE:
            selected = Action('move', solver.choose_move(state))
            values = [solver.value(force_action(state, a)) for a in choices]
        elif state.phase == Phase.AWAIT_DECISION:
            selected = Action('stop' if solver.should_stop(state) else 'roll')
            sv, rv = solver.stop_roll(state)
            values = [sv if a.kind == 'stop' else rv for a in choices]
        else:
            selected = Action('roll')
            values = [solver.value(state)]
        # Values come from the SAME backed-up table. Dice choices retain the
        # table's optimal subsequent stop/roll choice, rather than forcing it.
        options = [ActionValue(a, tuple(map(float, v))) for a, v in zip(choices, values)]
        options.sort(key=lambda x: (-x.value[actor], x.action != selected, x.action.key))
        return Decision(actor, tuple(map(float, solver.value(state))), selected, tuple(options))


def rng_stream(seed, domain, index=0):
    """Stable, independently reproducible game/search/training streams."""
    if domain not in ('game', 'search', 'training'):
        raise ValueError('unknown RNG domain')
    if type(seed) is not int or type(index) is not int or index < 0:
        raise ValueError('integer seed and nonnegative integer index required')
    payload = f'cantstop-decision-rng-v1|{seed}|{domain}|{index}'.encode()
    return PortableRng(int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little'))
