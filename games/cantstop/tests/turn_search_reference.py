"""Offline adaptive whole-turn search; exact dice backups, no offset correction.

Each expansion replaces one NN leaf with an exact next-turn solve. Updated
values propagate to the root before selecting the next expansion. This is
selective expectimax-style search, not MCTS. Depth counts EXTRA player turns.
"""
from dataclasses import dataclass, asdict
import math
import time
import weakref

import numpy as np

from games.cantstop.engine import GameState, Phase
from games.cantstop.rust_solver import _rust, leaf_features
from games.cantstop.snapshot import snapshot, from_snapshot


@dataclass(frozen=True)
class TurnSearchConfig:
    expansions: int = 4
    depth: int = 1
    seconds: float | None = None
    explore_every: int = 4

    def __post_init__(self):
        for name in ('expansions', 'depth', 'explore_every'):
            v = getattr(self, name)
            if type(v) is not int or v < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.expansions and self.depth == 0:
            raise ValueError('positive expansions require depth >= 1')
        if self.seconds is not None and (not math.isfinite(self.seconds) or self.seconds < 0):
            raise ValueError('seconds must be finite and nonnegative')


class _Node:
    def __init__(self, state, evaluator, depth=0, parent=None, leaf=None):
        self.state, self.depth, self.leaf = state.clone(), depth, leaf
        self.parent = weakref.proxy(parent) if parent is not None else None
        self.solver = _rust().TurnSolver(snapshot(state))
        self.children = {}
        self.boards = None
        if hasattr(evaluator, 'evaluate_features'):
            reference = GameState(state.rules)
            reference.active_player = self.solver.leaf_active_player
            values = evaluator.evaluate_features(leaf_features(self.solver), reference)
        else:
            self.boards = self.solver.leaf_snapshots()
            values = evaluator([from_snapshot(s) for s in self.boards])
        self.values = np.array(values, dtype=np.float64, copy=True)
        if (self.values.shape != (self.solver.num_leaves, state.rules.num_players)
                or not np.isfinite(self.values).all()
                or np.any(self.values < 0) or np.any(self.values > 1)
                or not np.allclose(self.values.sum(axis=1), 1, atol=1e-6, rtol=0)):
            raise ValueError('evaluator must return probability vectors in absolute seat order')
        self.solver.set_leaf_values_bytes(self.values.astype('<f8').tobytes())

    def backup(self):
        self.solver.rebackup(self.values.tolist())

    def value(self):
        return np.asarray(self.solver.value(sorted(self.state.runners.items()),
            int(self.state.phase), None if self.state.dice is None else list(self.state.dice)))

    def board(self, leaf):
        if self.boards is None:
            self.boards = self.solver.leaf_snapshots()
        return from_snapshot(self.boards[leaf])


class WholeTurnSearch:
    """Build once at the opening roll; query throughout that same turn.

    Expansion count is a hard bound. Time is a soft bound checked between
    solves: the mandatory root and any in-flight expansion finish atomically.
    Fixed budgets are reproducible; time budgets depend on hardware/load.
    """
    def __init__(self, state, evaluator, config=TurnSearchConfig(), *, clock=time.perf_counter,
                 cancelled=lambda: False):
        self.config = config
        started = clock()
        self.root = _Node(state, evaluator)
        self.nodes = [self.root]
        self.initial_value = self.root.value().copy()
        initial_choice = self._choice(state)
        self.trace = []
        reason = 'expansion_budget'
        for step in range(config.expansions):
            if cancelled():
                reason = 'cancelled'; break
            if config.seconds is not None and clock()-started >= config.seconds:
                reason = 'time_budget'; break
            candidates = []
            def visit(node, influence):
                reach = node.solver.leaf_reach()
                if node.depth < config.depth:
                    for leaf, weight in enumerate(reach):
                        if leaf not in node.children:
                            # Reach is decision visitation, NOT a distribution
                            # over chosen terminal outcomes. Used only to allocate work.
                            candidates.append((influence*weight, float(node.values[leaf, node.state.active_player]), node, leaf))
                for leaf, child in node.children.items():
                    visit(child, influence*reach[leaf])
            visit(self.root, 1.0)
            if not candidates:
                reason = 'frontier_exhausted'; break
            exploratory = bool(config.explore_every and (step+1) % config.explore_every == 0)
            if step == 0:
                # Always check bust: its continuation influences every risky roll.
                chosen = next(c for c in candidates if c[2] is self.root and c[3] == 0)
            elif exploratory:
                # Check attractive alternatives even when the current policy
                # assigns them zero reach. No claim this priority is optimal.
                chosen = max(candidates, key=lambda c: (c[1], c[0]))
            else:
                chosen = max(candidates, key=lambda c: (c[0], c[1]))
            weight, _, parent, leaf = chosen
            child = _Node(parent.board(leaf), evaluator, parent.depth+1, parent, leaf)
            parent.children[leaf] = child
            self.nodes.append(child)
            current = child
            while current.parent is not None:
                current.parent.values[current.leaf] = current.value()
                current.parent.backup()
                current = current.parent
            self.trace.append({'depth':child.depth, 'leaf':leaf, 'priority':weight,
                               'parent_node':self.nodes.index(parent), 'exploratory':exploratory, 'root_value':self.root.value().tolist()})
        self.stats = {'settings':asdict(config), 'expansions':len(self.trace),
                      'solves':len(self.nodes), 'positions':sum(n.solver.num_positions for n in self.nodes),
                      'evaluator_rows':sum(n.solver.num_leaves for n in self.nodes),
                      'elapsed_seconds':clock()-started, 'stop_reason':reason,
                      'initial_value':self.initial_value.tolist(), 'final_value':self.root.value().tolist(),
                      'depth_reached':max(n.depth for n in self.nodes),
                      'initial_choice':initial_choice, 'final_choice':self._choice(state),
                      'root_choice_changed':initial_choice != self._choice(state)}

    def _choice(self, state):
        if state.phase == Phase.AWAIT_MOVE:
            return list(self.choose_move(state))
        if state.phase == Phase.AWAIT_DECISION:
            return 'stop' if self.should_stop(state) else 'roll'
        return None

    def value(self, state):
        return np.asarray(self.root.solver.value(sorted(state.runners.items()), int(state.phase),
            None if state.dice is None else list(state.dice)))

    def choose_move(self, state):
        return tuple(self.root.solver.choose_move(sorted(state.runners.items()), list(state.dice)))

    def should_stop(self, state):
        return state.phase == Phase.AWAIT_DECISION and self.root.solver.should_stop(sorted(state.runners.items()))

    def stop_roll(self, state):
        return self.root.solver.stop_roll(sorted(state.runners.items()))
