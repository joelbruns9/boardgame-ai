"""
test_parallel_self_play.py — tests for parallel_self_play.py

REWRITE NOTE
────────────
This file previously targeted a bespoke ``InferenceServer(Process)`` plus
``InferenceRequest`` / ``InferenceResponse`` / ``make_batched_evaluator`` /
``_worker_loop`` / ``run_parallel_games`` API.  That whole layer was REPLACED
by the ``RemoteInferenceServer`` + ``inference_service`` IPC architecture (see
the ``parallel_self_play.py`` module docstring).  Six of the seven symbols the
old test imported no longer exist, so the module could not even be imported —
every test in it was silently dead.

The dead-API tests have been removed.  The surviving tests target the CURRENT
architecture:
  1. Example / ReplayBuffer data plumbing (current 11-field Example incl. the
     logging-phase ``iteration``; FLAT_SIZE; mean_age; sample_batch tuple).
  2. The parallel training loop (``run_parallel_self_play_training``) — fixed to
     use the real ``max_wait_ms`` kwarg and the current history schema
     (``value_loss`` removed since Phase 1b; four-head losses + the logging
     keys checked instead).
  3. The correctness oracle (serial == parallel) — the real correctness
     guarantee for the multiprocess path; already targets current code.

Run with:  python -m games.kingdomino.tests.test_parallel_self_play
"""
from __future__ import annotations

import dataclasses
import os
import sys
import unittest

import numpy as np

# The correctness-oracle tests run serial-vs-parallel comparisons that create
# and tear down multiple multiprocessing pools / RemoteInferenceServers in one
# process.  Under Windows ``spawn`` this repeated orchestration hits a kernel
# handle-duplication race (``PermissionError: [WinError 5] Access is denied`` in
# a child's ``rebuild_pipe_connection``) that hangs the parent indefinitely —
# an OS/spawn limitation, not a regression (the lighter single-loop MP test
# below passes).  These oracle tests are skipped on win32 and run on Linux/CI
# (where the cloud training actually runs).  The non-MP oracle test
# (``test_game_rngs_deterministic``) is NOT skipped.
_SKIP_HEAVY_MP = sys.platform == "win32"
_SKIP_HEAVY_MP_REASON = (
    "heavy repeated multiprocessing hangs on Windows spawn "
    "(WinError 5 handle duplication); runs on Linux/CI"
)

# ── make sure project root is importable ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from games.kingdomino.network import KingdominoNet  # noqa: F401 (import-sanity)
from games.kingdomino.self_play import (
    SelfPlayConfig, ReplayBuffer, Example, _new_history,
)
from games.kingdomino.parallel_self_play import run_parallel_self_play_training
from games.kingdomino.encoder import NUM_BOARD_CHANNELS, CANVAS_SIZE, FLAT_SIZE
from games.kingdomino.action_codec import NUM_JOINT_ACTIONS


# ── tiny config for fast tests ────────────────────────────────────────────────
def _tiny_cfg(**overrides) -> SelfPlayConfig:
    defaults = dict(
        channels=8, blocks=1, bilinear_dim=16,
        n_simulations=4, n_determinizations=1,
        c_puct=1.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
        temp_moves=2,
        buffer_capacity=5_000,
        lr=1e-3, weight_decay=0.0,
        batch_size=8,
        value_weight=1.0, policy_weight=1.0, augment=False,
        n_iterations=1, games_per_iteration=2, train_steps_per_iteration=2,
        min_buffer_to_train=1,
        benchmark_every=0, benchmark_seeds=2, benchmark_sims=4,
        device="cpu", seed=42,
        warm_start_path=None, checkpoint_dir=None,
    )
    defaults.update(overrides)
    return SelfPlayConfig(**defaults)


def _dummy_example(iteration: int = 0, *, pidx=(0,), pval=(1.0,), lidx=(0,)) -> Example:
    """A minimal well-formed Example matching the current 11-field schema."""
    return Example(
        my_board=np.zeros((NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE), np.float16),
        opp_board=np.zeros((NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE), np.float16),
        flat=np.zeros(FLAT_SIZE, np.float16),
        policy_idx=np.array(pidx, np.int32),
        policy_val=np.array(pval, np.float32),
        legal_idx=np.array(lidx, np.int32),
        z=0.0, own_score=0.0, opp_score=0.0, win_target=0.5,
        iteration=iteration,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — current Example / ReplayBuffer data plumbing (CPU, no multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────
class TestExampleSchema(unittest.TestCase):

    def test_flat_size(self):
        self.assertEqual(FLAT_SIZE, 261)

    def test_example_fields(self):
        """Example carries the 10 data fields plus the logging-phase iteration."""
        names = [f.name for f in dataclasses.fields(Example)]
        self.assertEqual(names, [
            "my_board", "opp_board", "flat",
            "policy_idx", "policy_val", "legal_idx",
            "z", "own_score", "opp_score", "win_target",
            "iteration",
        ])

    def test_example_iteration_defaults_zero(self):
        ex = Example(
            my_board=np.zeros((NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE), np.float16),
            opp_board=np.zeros((NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE), np.float16),
            flat=np.zeros(FLAT_SIZE, np.float16),
            policy_idx=np.array([0], np.int32),
            policy_val=np.array([1.0], np.float32),
            legal_idx=np.array([0], np.int32),
            z=0.0, own_score=0.0, opp_score=0.0, win_target=0.5,
        )
        self.assertEqual(ex.iteration, 0)

    def test_buffer_mean_age(self):
        buf = ReplayBuffer(capacity=100)
        self.assertEqual(buf.mean_age(5), 0.0)          # empty buffer
        buf.add([_dummy_example(0), _dummy_example(2)])  # mean iteration = 1
        self.assertAlmostEqual(buf.mean_age(5), 4.0)     # 5 - 1

    def test_sample_batch_returns_nine_tuple(self):
        buf = ReplayBuffer(capacity=100)
        buf.add([_dummy_example(0, pidx=(0, 5), pval=(0.5, 0.5), lidx=(0, 5, 9))
                 for _ in range(8)])
        batch = buf.sample_batch(4, np.random.default_rng(0), augment_d4=False)
        # (my_board, opp_board, flat, policy, legal_mask, z, own, opp, win)
        self.assertEqual(len(batch), 9)
        mb, ob, flat, policy, mask, z, own_t, opp_t, win_t = batch
        self.assertEqual(tuple(policy.shape), (4, NUM_JOINT_ACTIONS))
        self.assertEqual(tuple(flat.shape), (4, FLAT_SIZE))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — run_parallel_self_play_training (one tiny iteration)
# ─────────────────────────────────────────────────────────────────────────────
class TestParallelTrainingLoop(unittest.TestCase):

    def test_one_iteration_no_crash(self):
        cfg = _tiny_cfg(
            games_per_iteration=2,
            train_steps_per_iteration=2,
            min_buffer_to_train=1,
            benchmark_every=0,
        )
        result = run_parallel_self_play_training(
            cfg, n_workers=2, max_batch=4, max_wait_ms=20.0, verbose=False
        )
        self.assertIn("net", result)
        self.assertIn("history", result)
        self.assertIn("buffer", result)
        buf: ReplayBuffer = result["buffer"]
        self.assertGreater(len(buf), 0)

        h = result["history"]
        # The full logging-phase schema is present.
        for key in _new_history():
            self.assertIn(key, h, f"history missing key {key!r}")
        # value_loss was removed since Phase 1b (single-head value gone).
        self.assertNotIn("value_loss", h)
        # Losses should have been recorded (buffer > min) and be finite.
        for key in ("policy_loss", "own_loss", "opp_loss", "win_loss"):
            self.assertGreater(len(h[key]), 0, f"{key} not recorded")
            for v in h[key]:
                self.assertTrue(np.isfinite(v), f"{key}={v} not finite")

    def test_games_per_sec_positive(self):
        cfg = _tiny_cfg(games_per_iteration=2, train_steps_per_iteration=1,
                        min_buffer_to_train=1, benchmark_every=0)
        result = run_parallel_self_play_training(
            cfg, n_workers=2, max_batch=4, max_wait_ms=20.0, verbose=False
        )
        gps_list = result["history"]["games_per_sec"]
        self.assertGreater(len(gps_list), 0)
        for gps in gps_list:
            self.assertGreater(gps, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — correctness oracle: parallel reproduces serial data
# ─────────────────────────────────────────────────────────────────────────────
class TestCorrectnessOracle(unittest.TestCase):

    @unittest.skipIf(_SKIP_HEAVY_MP, _SKIP_HEAVY_MP_REASON)
    def test_pipeline_oracle_bit_identical(self):
        """Deterministic batch-independent evaluator: serial == parallel for
        1 and 3 workers, bit-for-bit."""
        from games.kingdomino.correctness_oracle import (
            _oracle_cfg, run_pipeline_oracle,
        )
        cfg = _oracle_cfg(n_simulations=8, temp_moves=4)
        base = cfg.seed * 1_000_003
        seeds = [base + i for i in range(4)]
        ok = run_pipeline_oracle(cfg, seeds, worker_counts=(1, 3), verbose=False)
        self.assertTrue(ok, "parallel pipeline diverged from serial")

    @unittest.skipIf(_SKIP_HEAVY_MP, _SKIP_HEAVY_MP_REASON)
    def test_realnet_oracle_batch1(self):
        """Real network, production InferenceServer, 1 worker, batch-1:
        serial == parallel."""
        from games.kingdomino.correctness_oracle import (
            _oracle_cfg, run_realnet_oracle,
        )
        cfg = _oracle_cfg(channels=8, blocks=1, bilinear_dim=16,
                          n_simulations=6, temp_moves=3)
        base = cfg.seed * 1_000_003
        seeds = [base + i for i in range(2)]
        ok = run_realnet_oracle(cfg, seeds, verbose=False)
        self.assertTrue(ok, "real-net parallel diverged from serial at batch-1")

    def test_game_rngs_deterministic(self):
        """_game_rngs is a pure function of the seed."""
        from games.kingdomino.self_play import _game_rngs
        py1, np1 = _game_rngs(12345)
        py2, np2 = _game_rngs(12345)
        self.assertEqual(py1.random(), py2.random())
        self.assertEqual(np1.random(), np2.random())
        py3, _ = _game_rngs(54321)
        self.assertNotEqual(_game_rngs(12345)[0].random(), py3.random())


if __name__ == "__main__":
    # Run with verbose output
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
