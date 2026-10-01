"""Batched full-game audit continuation with the serial estimator's semantics.

The root turn stays in Python against one immutable table. Rust receives only
turn-start boards and returns complete trajectories, replayed through the common
RolloutBackend aggregation so raw payoffs, pairing, corrections and digests have
one implementation. This backend supports plain baseline continuation only.
"""
from dataclasses import replace
import hashlib
import json
import time

import numpy as np

from .decision_search import actions, rng_stream
from .encoder import FEATURE_SIZE, decode_features
from .rollout_search import RolloutBackend, RolloutConfig, RolloutLimitExceeded, SharedDice
from .rust_pool import as_feature_evaluator, relative_for, values_for
from .snapshot import from_snapshot, snapshot


class _AuditBoardEvaluator:
    """Keep primitive snapshot types identical for board-level mock evaluators."""
    def __init__(self, evaluate):
        self.evaluate = evaluate

    def evaluate_features(self, features, reference):
        boards = [decode_features(row, reference.active_player) for row in features]
        for board in boards:
            # decode_features counts numpy booleans, yielding numpy integer
            # rule/claim fields. A snapshot-hashing mock observes their repr.
            board.rules = reference.rules
            board.claimed_by = {c: None if p is None else int(p) for c, p in board.claimed_by.items()}
        return self.evaluate(boards)


class PoolRolloutBackend(RolloutBackend):
    def __init__(self, evaluator, config=RolloutConfig(horizon=None, common_random_numbers=True),
                 *, threads=0, in_flight=64, max_rows=1_000_000, **kwargs):
        if config.horizon is not None or not config.common_random_numbers:
            raise ValueError('pool requires full-game horizon and shared dice')
        if any(type(x) is not int for x in (threads, in_flight, max_rows)) or threads < 0 or in_flight < 1 or max_rows < 1:
            raise ValueError('invalid pool resource limits')
        if kwargs.get('policy_factory') is not None:
            raise ValueError('pool supports plain baseline continuation only')
        super().__init__(evaluator, config, **kwargs)
        self.threads, self.in_flight, self.max_rows = threads, in_flight, max_rows
        self._replay = None

    def _trajectory(self, root, action, rng, root_solver, shared=None):
        key, trajectory = next(self._replay)
        if key != action.key:
            raise RuntimeError('pool result order differs from canonical action order')
        return trajectory

    def _evaluate(self, state, subset=None):
        if state.game_over:
            return super()._evaluate(state, subset)
        candidates = actions(state)
        if subset is not None:
            subset = tuple(subset)
            if not subset or len(set(subset)) != len(subset) or any(a not in candidates for a in subset):
                raise ValueError('candidate subset must contain distinct legal actions')
            candidates = tuple(a for a in candidates if a in subset)
        if not candidates:
            raise ValueError('no legal root actions')
        import cantstop_rust
        if not hasattr(cantstop_rust.SelfPlayPool, 'configure_audit'):
            raise RuntimeError('rebuild cantstop_rust for audit pool support')
        root_backend = RolloutBackend(self.evaluator, replace(self.config, horizon=0),
                                     solver_factory=self.solver_factory, cancelled=self.cancelled,
                                     deadline=self.deadline, clock=self.clock)
        table = None
        def root_solver():
            nonlocal table
            if table is None:
                table = root_backend._make_solver(state, 0)
            return table
        trajectories, specs, streams, slots = [], [], [], []
        started = time.perf_counter()
        for action in candidates:
            payload = json.dumps([snapshot(state), 'shared'], separators=(',', ':')).encode()
            domain_seed = self.config.seed ^ int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')
            for i in range(self.config.samples):
                self._check_budget()
                rng = rng_stream(domain_seed, 'search', self.config.sample_offset + i)
                trajectory = root_backend._trajectory(state, action, rng, root_solver, SharedDice(rng.state))
                endpoint, turns, rolls, correction = trajectory
                slot = len(trajectories)
                trajectories.append((action.key, trajectory))
                if not endpoint.game_over:
                    if turns != 1 or endpoint.runners:
                        raise RuntimeError('pool boundary must finish exactly the root turn')
                    slots.append(slot)
                    specs.append((snapshot(endpoint), 0, [0] * state.rules.num_players))
                    streams.append((rng.state, turns, rolls, correction.tolist()))
        costs = {'root_seconds': time.perf_counter() - started, 'rust_seconds': 0.,
                 'inference_seconds': 0., 'rounds': 0, 'rows': 0,
                 'threads': self.threads, 'in_flight': self.in_flight, 'max_rows': self.max_rows}
        if specs:
            evaluator = (as_feature_evaluator(self.evaluator) if hasattr(self.evaluator, 'evaluate_features')
                         else _AuditBoardEvaluator(self.evaluator))
            batched = hasattr(evaluator, 'relative_probs')
            pool = cantstop_rust.SelfPlayPool(specs, self.config.max_turns, self.threads,
                                            self.in_flight, max_rows=self.max_rows)
            pool.configure_audit(streams, self.config.max_rolls, self.config.dice_luck)
            tick = time.perf_counter()
            pool.advance()
            costs['rust_seconds'] += time.perf_counter() - tick
            while pool.running:
                self._check_budget()
                if pool.failure is not None:
                    raise RolloutLimitExceeded(pool.failure)
                tick = time.perf_counter()
                raw, blocks = pool.pending()
                costs['rust_seconds'] += time.perf_counter() - tick
                if not blocks:
                    raise RuntimeError('live audit pool returned no inference blocks')
                features = np.frombuffer(raw, dtype='<f4').reshape(-1, FEATURE_SIZE)
                tick = time.perf_counter()
                if batched:
                    values = relative_for([evaluator], features, blocks)
                else:
                    values = values_for([evaluator], features, blocks, [state.rules] * len(specs))
                costs['inference_seconds'] += time.perf_counter() - tick
                costs['rounds'] += 1
                costs['rows'] += len(features)
                tick = time.perf_counter()
                if batched:
                    pool.resume_relative(values)
                else:
                    pool.resume(values)
                costs['rust_seconds'] += time.perf_counter() - tick
            if pool.failure is not None:
                raise RolloutLimitExceeded(pool.failure)
            results = pool.audit_results()
            if len(results) != len(slots):
                raise RuntimeError('audit pool dropped games')
            for expected_id, (gid, winner, turns, rolls, correction, snap) in enumerate(results):
                endpoint = from_snapshot(snap)
                if gid != expected_id or not endpoint.game_over or endpoint.winner != winner:
                    raise RuntimeError('audit pool returned invalid or reordered terminal outcomes')
                slot = slots[expected_id]
                trajectories[slot] = (trajectories[slot][0], (endpoint, turns, rolls, np.asarray(correction)))
        self._check_budget()
        self._replay = iter(trajectories)
        try:
            result = super()._evaluate(state, subset)
            if next(self._replay, None) is not None:
                raise RuntimeError('unconsumed audit pool games')
            for row in self.last_stats['actions']:
                row['aggregation_seconds'] = row.pop('generation_seconds')
            self.last_stats['backend'] = 'pool'
            self.last_stats['pool'] = costs
            return result
        finally:
            self._replay = None
