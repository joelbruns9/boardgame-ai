"""Staged root-action comparisons with bounded stronger early continuation.

Selective future-turn expansion is approximate; only each individual turn's
chance backup is exact given its frontier. This is not full N-turn expectiminimax.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import json
import math
import time

from .decision_search import Decision, TurnTableBackend
from .rollout_search import RolloutBackend, RolloutConfig
from .rust_solver import RustTurnSolver
from .snapshot import snapshot
from .turn_search import WholeTurnSearch, TurnSearchConfig


@dataclass(frozen=True)
class ProgressiveConfig:
    horizons: tuple = (0, 1)
    samples: tuple = (8, 32)
    margin: float | None = None
    max_candidates: int | None = None
    early_turns: int = 0
    expansions: int = 0
    depth: int = 1
    cache_entries: int = 8
    cache_positions: int = 200000

    def __post_init__(self):
        if not isinstance(self.horizons, tuple) or not isinstance(self.samples, tuple):
            raise ValueError('horizons and samples must be tuples')
        if not self.horizons or len(self.horizons) != len(self.samples):
            raise ValueError('each horizon needs a sample budget')
        last = -1
        for h, n in zip(self.horizons, self.samples):
            if type(n) is not int or n < 1:
                raise ValueError('sample budgets must be positive integers')
            if h is not None and (type(h) is not int or h < 0):
                raise ValueError('invalid horizon')
            level = math.inf if h is None else h
            if level <= last:
                raise ValueError('horizons must increase; full game can appear only last')
            last = level
        if self.margin is not None and (not math.isfinite(self.margin) or self.margin < 0):
            raise ValueError('margin must be finite and nonnegative')
        for name in ('early_turns', 'expansions', 'cache_entries', 'cache_positions'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.max_candidates is not None and (type(self.max_candidates) is not int or self.max_candidates < 1):
            raise ValueError('max_candidates must be positive')
        if bool(self.early_turns) != bool(self.expansions):
            raise ValueError('early_turns and expansions must both be enabled or both zero')
        TurnSearchConfig(expansions=self.expansions, depth=self.depth)


class EarlyTurnPolicy:
    """Per-evaluation LRU: fixed evaluator/config; no cross-checkpoint reuse.

    A cached table may contain several selective-search nodes. Bound their total
    retained position count as well as entry count. Oversize solves are used
    without retention; these are cache bounds, not limits on an in-flight solve.
    """
    def __init__(self, evaluator, config):
        self.evaluator, self.config = evaluator, config
        self.cache = OrderedDict()
        self.positions = 0
        self.stats = {'hits': 0, 'misses': 0, 'strong_solves': 0, 'baseline_solves': 0,
                      'expansions': 0, 'peak_cached_positions': 0, 'peak_cached_entries': 0}

    def __call__(self, state, evaluator, completed):
        if evaluator is not self.evaluator:
            raise ValueError('policy cache belongs to a different evaluator')
        strong = 1 <= completed <= self.config.early_turns
        key = (strong, json.dumps(snapshot(state), separators=(',', ':')))
        if key in self.cache:
            self.stats['hits'] += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.stats['misses'] += 1
        if strong:
            solver = WholeTurnSearch(state, evaluator, TurnSearchConfig(
                expansions=self.config.expansions, depth=self.config.depth))
            self.stats['strong_solves'] += 1
            self.stats['expansions'] += solver.stats['expansions']
        else:
            solver = RustTurnSolver(state, evaluator)
            self.stats['baseline_solves'] += 1
        size = solver.num_positions
        if self.config.cache_entries and size <= self.config.cache_positions:
            while self.cache and (len(self.cache) >= self.config.cache_entries or
                                  self.positions + size > self.config.cache_positions):
                _, old = self.cache.popitem(last=False)
                self.positions -= old.num_positions
            self.cache[key] = solver
            self.positions += size
            self.stats['peak_cached_positions'] = max(self.positions, self.stats['peak_cached_positions'])
            self.stats['peak_cached_entries'] = max(len(self.cache), self.stats['peak_cached_entries'])
        return solver


class ProgressiveBackend:
    def __init__(self, evaluator, config=ProgressiveConfig(), rollout=RolloutConfig()):
        self.evaluator, self.config, self.rollout = evaluator, config, rollout
        self.last_stats = None

    def evaluate(self, state):
        start = time.perf_counter()
        policy = EarlyTurnPolicy(self.evaluator, self.config)
        self.last_stats = {'status': 'running', 'config': asdict(self.config),
                           'rollout': asdict(self.rollout), 'stages': [], 'policy': policy.stats}
        try:
            baseline = TurnTableBackend(self.evaluator).evaluate(state)
            self.last_stats['baseline'] = baseline.to_dict()
            if state.game_over:
                self.last_stats['status'] = 'complete'
                return baseline
            survivors = [o.action for o in baseline.options]
            result = baseline
            for i, (h, samples) in enumerate(zip(self.config.horizons, self.config.samples)):
                if h == 0:
                    # Baseline continuation makes H=0 exactly the root table.
                    options = tuple(o for o in baseline.options if o.action in survivors)
                    result = Decision(baseline.actor, options[0].value, options[0].action, options)
                    stage_stats = {'status':'complete', 'source':'exact_baseline',
                        'actions':[{'action':o.action.key, 'samples':0} for o in options]}
                else:
                    cfg = replace(self.rollout, samples=samples, horizon=h)
                    backend = RolloutBackend(self.evaluator, cfg, policy_factory=policy)
                    result = backend.evaluate(state, candidates=survivors)
                    stage_stats = backend.last_stats
                keep = [o.action for o in result.options]
                # All actions get the first stage. Only same-stage estimates
                # are filtered; survivors all advance to the final horizon.
                if i < len(self.config.horizons)-1:
                    best = result.value[state.active_player]
                    if self.config.margin is not None:
                        keep = [o.action for o in result.options
                                if best-o.value[state.active_player] <= self.config.margin]
                    if self.config.max_candidates is not None:
                        keep = keep[:self.config.max_candidates]
                self.last_stats['stages'].append({'index': i, 'horizon': h,
                    'result': result.to_dict(), 'rollout': stage_stats,
                    'pruned': [a.key for a in survivors if a not in keep],
                    'survivors': [a.key for a in keep]})
                survivors = keep
            self.last_stats['status'] = 'complete'
            self.last_stats['finalists'] = [o.action.key for o in result.options]
            return result
        except BaseException as exc:
            self.last_stats['status'] = 'incomplete'
            self.last_stats['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            self.last_stats['elapsed_seconds'] = time.perf_counter()-start
            policy.cache.clear()  # Evaluator/checkpoint lifecycle ends here.


def audit_filtering(progressive, reference):
    """Compare finalists against a stronger independently seeded evaluation.

    Reference uncertainty is retained; this is estimated regret, not ground truth.
    """
    actor = reference.actor
    values = {o.action: o.value[actor] for o in reference.options}
    finalists = [o.action for o in progressive.options]
    if any(a not in values for a in finalists):
        raise ValueError('reference must cover all finalists')
    best = max(values.values())
    finalist_best = max(values[a] for a in finalists)
    return {'reference_winner_pruned': reference.selected not in finalists,
            'estimated_filtering_regret': best-finalist_best,
            'estimated_selected_regret': best-values[progressive.selected]}
