"""Equal-budget rollouts with optional dice variance reduction under the baseline turn policy.

H counts additional completed player turns after finishing the current turn.
None means full games. No pruning or adaptive allocation. Variance-reduction switches default off.
"""
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import time

import numpy as np

from .decision_search import ActionValue, Decision, actions, force_action, rng_stream
from .engine import Phase, apply_move, roll, stop
from .rust_solver import RustTurnSolver
from .snapshot import snapshot


@dataclass(frozen=True)
class RolloutConfig:
    samples: int = 32
    horizon: int | None = 1
    seed: int = 20260930
    max_turns: int = 1000
    max_rolls: int = 10000
    endpoint_batch_size: int = 256
    sample_offset: int = 0
    common_random_numbers: bool = False
    dice_luck: bool = False

    def __post_init__(self):
        if type(self.sample_offset) is not int or self.sample_offset < 0:
            raise ValueError('sample_offset must be a nonnegative integer')
        for name in ('common_random_numbers', 'dice_luck'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        for name in ('samples', 'max_turns', 'max_rolls', 'endpoint_batch_size'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.horizon is not None and (type(self.horizon) is not int or self.horizon < 0):
            raise ValueError('horizon must be a nonnegative integer or None')
        if type(self.seed) is not int:
            raise ValueError('seed must be an integer')


class RolloutInterrupted(RuntimeError):
    pass


class RolloutLimitExceeded(RuntimeError):
    pass


def search_dice(rng):
    """Uniform dice via rejection, without changing the arena's legacy RNG."""
    limit = (1 << 64) - ((1 << 64) % 6)
    dice = []
    while len(dice) < 4:
        x = rng.next_u64()
        if x < limit:
            dice.append(x % 6 + 1)
    return tuple(dice)


class SharedDice:
    def __init__(self, seed):
        self.seed = seed

    def at(self, turn, roll_index):
        payload = f'cantstop-shared-dice-v1|{self.seed}|{turn}|{roll_index}'.encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')
        return search_dice(rng_stream(seed, 'search'))


def expected_roll_value(solver, state):
    """g averaged over all dice, before drawing: force the roll phase."""
    pre = state.clone()
    pre.phase, pre.dice = Phase.AWAIT_ROLL, None
    return np.asarray(solver.value(pre), dtype=np.float64)


def dice_luck_delta(expected, solver, rolled_state, busted):
    """Zero-conditional-mean control variate, even if g is inaccurate.

    g is the baseline table's post-roll value, or its original bust leaf.
    expected MUST be computed before drawing using the same immutable table.
    """
    realized = solver.bust_value if busted else solver.value(rolled_state)
    delta = np.asarray(expected, dtype=np.float64) - np.asarray(realized, dtype=np.float64)
    if not np.isfinite(delta).all():
        raise ValueError('nonfinite dice-luck correction')
    return delta


class RolloutBackend:
    def __init__(self, evaluator, config=RolloutConfig(), *, solver_factory=RustTurnSolver, policy_factory=None, retain_samples=False, cancelled=lambda: False, deadline=None, clock=time.perf_counter):
        self.evaluator, self.config = evaluator, config
        self.solver_factory = solver_factory
        self.policy_factory = policy_factory
        self.last_stats = None
        self.last_samples = None
        self.retain_samples = retain_samples
        self.cancelled, self.deadline, self.clock = cancelled, deadline, clock

    def _check_budget(self):
        if self.cancelled():
            raise RolloutInterrupted("cancelled")
        if self.deadline is not None and self.clock() >= self.deadline:
            raise RolloutInterrupted("time_budget")

    def _make_solver(self, state, completed):
        self._check_budget()
        if self.policy_factory is not None:
            return self.policy_factory(state, self.evaluator, completed)
        return self.solver_factory(state, self.evaluator)

    def _trajectory(self, root, action, rng, root_solver, shared=None):
        state = force_action(root, action)
        completed = int(state.game_over or state.active_player != root.active_player)
        rolls = 0
        roll_index = 0
        correction = np.zeros(root.rules.num_players, dtype=np.float64)
        solver = None
        while True:
            self._check_budget()
            if state.game_over:
                return state, completed, rolls, correction
            if self.config.horizon is not None and completed >= self.config.horizon + 1:
                assert state.phase == Phase.AWAIT_ROLL and not state.runners
                return state, completed, rolls, correction
            if completed >= self.config.max_turns:
                raise RolloutLimitExceeded('maximum completed turns reached before rollout finished')
            # A forced move/roll stays inside the root's existing table. On
            # future turns build after observing opening dice. A zero control
            # variate on that opening avoids an expensive extra pre-roll solve.
            if completed == 0 and solver is None:
                solver = root_solver()
            if state.phase in (Phase.AWAIT_ROLL, Phase.AWAIT_DECISION):
                if state.phase == Phase.AWAIT_DECISION:
                    if solver is None:
                        solver = self._make_solver(state, completed)
                    if solver.should_stop(state):
                        stop(state)
                        completed += 1
                        solver = None
                        roll_index = 0
                        continue
                if rolls >= self.config.max_rolls:
                    raise RolloutLimitExceeded('maximum dice rolls reached before rollout finished')
                expected = (expected_roll_value(solver, state)
                            if self.config.dice_luck and solver is not None else None)
                dice = shared.at(completed, roll_index) if shared is not None else search_dice(rng)
                rolls += 1
                roll_index += 1
                busted = not roll(state, dice)
                if expected is not None:
                    correction += dice_luck_delta(expected, solver, state, busted)
                if busted:
                    completed += 1
                    solver = None
                    roll_index = 0
                    continue
            if solver is None:
                solver = self._make_solver(state, completed)
            apply_move(state, solver.choose_move(state))

    def evaluate(self, state, *, candidates=None):
        started = time.perf_counter()
        self.last_samples = None
        self.last_stats = {'status': 'running', 'config': asdict(self.config), 'endpoint_seconds': 0.0, 'actions': []}
        try:
            result = self._evaluate(state, candidates)
            self.last_stats['status'] = 'complete'
            return result
        except BaseException as exc:
            self.last_stats['status'] = 'incomplete'
            self.last_stats['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            self.last_stats['elapsed_seconds'] = time.perf_counter() - started

    def _evaluate(self, state, subset=None):
        n, actor = state.rules.num_players, state.active_player
        if state.game_over:
            return Decision(actor, tuple(float(p == state.winner) for p in range(n)), None, ())
        candidates = actions(state)
        if subset is not None:
            subset = tuple(subset)
            if not subset or len(set(subset)) != len(subset) or any(a not in candidates for a in subset):
                raise ValueError('candidate subset must contain distinct legal actions')
            candidates = tuple(a for a in candidates if a in subset)
        if not candidates:
            raise ValueError('no legal root actions')
        root_table = None
        def root_solver():
            nonlocal root_table
            if root_table is None:
                root_table = self._make_solver(state, 0)
            return root_table
        samples = np.empty((len(candidates), self.config.samples, n), dtype=np.float64)
        corrections = np.zeros_like(samples)
        pending = defaultdict(list)
        digest = hashlib.sha256()

        def flush(key):
            group = pending[key]
            boards = [entry[2] for entry in group]
            batch_started = time.perf_counter()
            self._check_budget()
            values = np.asarray(self.evaluator(boards), dtype=np.float64)
            if (values.shape != (len(group), n) or not np.isfinite(values).all()
                    or (values < 0).any() or (values > 1).any()
                    or not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0)):
                raise ValueError('endpoint evaluator must return absolute-seat probability vectors')
            for (a, i, _), value in zip(group, values):
                samples[a, i] = value
            group.clear()
            self.last_stats['endpoint_seconds'] += time.perf_counter() - batch_started

        for a, action in enumerate(candidates):
            action_started = time.perf_counter()
            stats = {'action': action.key, 'samples': 0, 'terminal_samples': 0,
                     'completed_turns': 0, 'dice_rolls': 0}
            self.last_stats['actions'].append(stats)
            # Independent candidate seeds, or a shared simulation seed when
            # explicit pairing is enabled. Same state/config repeats exactly, independent of
            # evaluation order or previous backend calls.
            payload = json.dumps([snapshot(state), 'shared' if self.config.common_random_numbers else action.key], separators=(',', ':')).encode()
            domain_seed = self.config.seed ^ int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')
            for i in range(self.config.samples):
                rng = rng_stream(domain_seed, 'search', self.config.sample_offset+i)
                shared = SharedDice(rng.state) if self.config.common_random_numbers else None
                endpoint, turns, rolls, correction = self._trajectory(
                    state, action, rng, root_solver, shared)
                corrections[a, i] = correction
                digest.update(json.dumps([a, i, snapshot(endpoint), turns, rolls], separators=(',', ':')).encode())
                stats['samples'] += 1
                stats['completed_turns'] += turns
                stats['dice_rolls'] += rolls
                if endpoint.game_over:
                    samples[a, i] = [float(p == endpoint.winner) for p in range(n)]
                    stats['terminal_samples'] += 1
                else:
                    # NetEvaluator rotates a whole batch using the first
                    # board's actor; mixed-seat batches would corrupt values.
                    key = (endpoint.rules, endpoint.active_player)
                    pending[key].append((a, i, endpoint))
                    if len(pending[key]) >= self.config.endpoint_batch_size:
                        flush(key)
            stats['generation_seconds'] = time.perf_counter() - action_started
        for key in pending:
            if pending[key]:
                flush(key)
        raw_samples = samples.copy()
        samples += corrections  # Deliberately no clipping or renormalization.
        if self.retain_samples:
            self.last_samples = {action.key: {"raw": raw_samples[a].copy(), "adjusted": samples[a].copy()}
                                 for a, action in enumerate(candidates)}
        options = []
        for a, action in enumerate(candidates):
            mean = samples[a].mean(axis=0)
            variance = samples[a].var(axis=0, ddof=1) if self.config.samples > 1 else None
            stats = self.last_stats['actions'][a]
            stats['mean'] = mean.tolist()
            stats['raw_mean'] = raw_samples[a].mean(axis=0).tolist()
            stats['correction_mean'] = corrections[a].mean(axis=0).tolist()
            stats['adjusted_samples_outside_unit_interval'] = int(np.any((samples[a] < 0) | (samples[a] > 1), axis=1).sum())
            stats['sample_variance'] = None if variance is None else variance.tolist()
            stats['standard_error'] = None if variance is None else np.sqrt(variance / self.config.samples).tolist()
            options.append(ActionValue(action, tuple(map(float, mean))))
        # Canonical engine action order breaks exact ties (stop before roll).
        best = max(options, key=lambda x: x.value[actor])
        options.sort(key=lambda x: -x.value[actor])
        self.last_stats['differences'] = []
        for a in range(len(candidates)):
            for b in range(a + 1, len(candidates)):
                difference = samples[a, :, actor] - samples[b, :, actor]
                raw_difference = raw_samples[a, :, actor] - raw_samples[b, :, actor]
                variance = float(difference.var(ddof=1)) if self.config.samples > 1 else None
                self.last_stats['differences'].append({
                    'a': candidates[a].key, 'b': candidates[b].key,
                    'mean': float(difference.mean()), 'sample_variance': variance,
                    'raw_sample_variance': float(raw_difference.var(ddof=1)) if variance is not None else None,
                    'standard_error': None if variance is None else float(np.sqrt(variance / self.config.samples)),
                    'paired': self.config.common_random_numbers})
        self.last_stats['trajectory_sha256'] = digest.hexdigest()
        self.last_stats['uncertainty'] = 'Monte Carlo sampling only; excludes endpoint and continuation-policy error'
        return Decision(actor, best.value, best.action, tuple(options))
