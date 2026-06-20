"""
test_inference_service.py — tests for the unified inference service (A1 + A3)
and the threaded self-play game pool.

Run: python test_inference_service.py
"""
from __future__ import annotations

import os
import sys
import threading
import time
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from games.kingdomino.network import KingdominoNet
from games.kingdomino.encoder import FLAT_SIZE
from games.kingdomino.mcts_az import make_serial_evaluator
from games.kingdomino.action_codec import NUM_JOINT_ACTIONS
from games.kingdomino.self_play import SelfPlayConfig, Example
from games.kingdomino.inference_service import (
    RequestRouter, InferenceResult, InferenceClient,
    LocalInferenceService, RemoteInferenceServer, RemoteInferenceWorkerClient,
    clone_state_dict,
)
from games.kingdomino.threaded_self_play import run_threaded_self_play_games


def _tiny_net():
    torch.manual_seed(0)
    return KingdominoNet(channels=8, blocks=1, bilinear_dim=16)


def _rand_inputs(rng):
    return (rng.standard_normal((9, 13, 13)).astype(np.float32),
            rng.standard_normal((9, 13, 13)).astype(np.float32),
            rng.standard_normal(FLAT_SIZE).astype(np.float32))


# The inference seam gathers LEGAL logits: infer(mb, ob, flat, idxs) returns
# logits[idxs].  Passing every index makes the gather a no-op, so the returned
# logits keep their full (NUM_JOINT_ACTIONS,) shape — what these tests assert.
_ALL_IDX = np.arange(NUM_JOINT_ACTIONS, dtype=np.int64)


# ── TEST 1 — RequestRouter ───────────────────────────────────────────────────
class TestRouter(unittest.TestCase):
    def test_register_complete(self):
        r = RequestRouter()
        p = r.register(7)
        self.assertFalse(p.event.is_set())
        res = InferenceResult(0.5, np.zeros(NUM_JOINT_ACTIONS, np.float32), 1)
        r.complete(7, res)
        self.assertTrue(p.event.is_set())
        self.assertEqual(p.result.value, 0.5)

    def test_cancel_all(self):
        r = RequestRouter()
        p = r.register(1)
        r.cancel_all()
        self.assertTrue(p.event.is_set())
        self.assertTrue(p.cancelled)

    def test_complete_unknown_id_noop(self):
        r = RequestRouter()
        r.complete(999, InferenceResult(0.0, np.zeros(1, np.float32), 0))  # no crash


# ── TEST 2 — InferenceClient roundtrip with a manual backend ─────────────────
class TestClient(unittest.TestCase):
    def test_roundtrip(self):
        router = RequestRouter()
        captured = {}

        def submit(rid, wid, mb, ob, flat, idxs_list, batched):
            captured["rid"] = rid
            # Simulate a backend completing the request from another thread.
            threading.Timer(0.01, lambda: router.complete(
                rid, InferenceResult(0.25, np.ones(NUM_JOINT_ACTIONS, np.float32), 3)
            )).start()

        client = InferenceClient(submit, router, lambda: 3, worker_id=2)
        mb, ob, flat = _rand_inputs(np.random.default_rng(0))
        v, lg = client(mb, ob, flat, _ALL_IDX)
        self.assertEqual(v, 0.25)
        self.assertEqual(lg.shape, (NUM_JOINT_ACTIONS,))
        self.assertEqual(client.model_version, 3)
        # request_id carries the worker_id in the high bits
        self.assertEqual(captured["rid"] >> 40, 2)


# ── TEST 3 — LocalInferenceService matches serial evaluator ──────────────────
class TestLocalService(unittest.TestCase):
    def test_matches_serial_batch1(self):
        net = _tiny_net().eval()
        serial = make_serial_evaluator(net, device="cpu")
        snet = _tiny_net()
        snet.load_state_dict(net.state_dict())
        with LocalInferenceService(snet, "cpu", max_batch=8, max_wait_ms=5) as svc:
            client = svc.make_client()
            rng = np.random.default_rng(1)
            for _ in range(10):
                mb, ob, flat = _rand_inputs(rng)
                vs, ls = serial(mb, ob, flat, _ALL_IDX)
                vc, lc = client(mb, ob, flat, _ALL_IDX)
                self.assertAlmostEqual(vs, vc, places=5)
                self.assertTrue(np.allclose(ls, lc, atol=1e-5))

    def test_concurrent_calls(self):
        """Many threads sharing one client all get correct-shaped results."""
        net = _tiny_net()
        with LocalInferenceService(net, "cpu", max_batch=16, max_wait_ms=5) as svc:
            client = svc.make_client()
            errors = []

            def hammer(seed):
                rng = np.random.default_rng(seed)
                try:
                    for _ in range(15):
                        v, lg = client(*_rand_inputs(rng), _ALL_IDX)
                        assert isinstance(v, float)
                        assert lg.shape == (NUM_JOINT_ACTIONS,)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=hammer, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            s = svc.stats()
            self.assertGreater(s["requests"], 0)
            self.assertGreater(s["mean_batch"], 1.0)  # batching actually happened

    def test_weight_update_and_barrier(self):
        net = _tiny_net()
        with LocalInferenceService(net, "cpu", max_batch=4, max_wait_ms=5) as svc:
            client = svc.make_client()
            client(*_rand_inputs(np.random.default_rng(0)), _ALL_IDX)  # force a batch
            # update_weights returns the requested version; barrier confirms it
            v = svc.update_weights(
                {k: t.clone() for k, t in _tiny_net().state_dict().items()})
            self.assertTrue(svc.wait_for_version(v, timeout_s=10),
                            "wait_for_version timed out")
            self.assertGreaterEqual(svc.stats()["model_version"], v)
            # client reports the applied version
            self.assertGreaterEqual(client.model_version, v)

    def test_clean_shutdown(self):
        net = _tiny_net()
        svc = LocalInferenceService(net, "cpu").start()
        svc.stop()
        self.assertFalse(svc._thread.is_alive())

    def test_owns_independent_net(self):
        """Mutating the trainer's net must not change the service's net."""
        net = _tiny_net()
        with LocalInferenceService(net, "cpu", max_batch=4, max_wait_ms=5) as svc:
            client = svc.make_client()
            mb, ob, flat = _rand_inputs(np.random.default_rng(0))
            v_before, _ = client(mb, ob, flat, _ALL_IDX)
            # Corrupt the trainer net in place; service has its own deepcopy.
            with torch.no_grad():
                for p in net.parameters():
                    p.add_(100.0)
            v_after, _ = client(mb, ob, flat, _ALL_IDX)
            self.assertAlmostEqual(v_before, v_after, places=5,
                                   msg="service net was not independent")

    def test_auto_client_ids_no_collision(self):
        """Two default clients must not share a request-id prefix."""
        net = _tiny_net()
        with LocalInferenceService(net, "cpu", max_batch=4, max_wait_ms=5) as svc:
            c1 = svc.make_client()
            c2 = svc.make_client()
            errors = []

            def run(c, seed):
                rng = np.random.default_rng(seed)
                try:
                    for _ in range(10):
                        v, lg = c(*_rand_inputs(rng), _ALL_IDX)
                        assert lg.shape == (NUM_JOINT_ACTIONS,)
                except Exception as e:
                    errors.append(e)

            t1 = threading.Thread(target=run, args=(c1, 1))
            t2 = threading.Thread(target=run, args=(c2, 2))
            t1.start(); t2.start(); t1.join(); t2.join()
            self.assertEqual(errors, [])

    def test_infer_timeout(self):
        """If the batcher never runs, infer raises TimeoutError, not a hang."""
        net = _tiny_net()
        svc = LocalInferenceService(net, "cpu", max_batch=4, max_wait_ms=5)
        # Deliberately do NOT start the batcher thread.
        client = svc.make_client()
        mb, ob, flat = _rand_inputs(np.random.default_rng(0))
        with self.assertRaises(TimeoutError):
            client(mb, ob, flat, _ALL_IDX, timeout_s=0.3)


# ── TEST: clone_state_dict severs storage sharing ───────────────────────────
class TestCloneStateDict(unittest.TestCase):
    def test_independent_storage(self):
        net = _tiny_net()
        sd = net.state_dict()
        cloned = clone_state_dict(sd)
        with torch.no_grad():
            for p in net.parameters():
                p.add_(1.0)
        # cloned values must be unchanged by the in-place mutation above
        for k in cloned:
            self.assertFalse(torch.equal(cloned[k], net.state_dict()[k].cpu()),
                             f"{k} clone tracked the mutation")


# ── TEST 4 — threaded game pool generates valid games ────────────────────────
class TestThreadedGamePool(unittest.TestCase):
    def test_generates_games(self):
        cfg = SelfPlayConfig(channels=8, blocks=1, bilinear_dim=16,
                             n_simulations=6, temp_moves=2,
                             device="cpu", seed=3)
        net = _tiny_net()
        with LocalInferenceService(net, "cpu", max_batch=4, max_wait_ms=5) as svc:
            examples, scores = run_threaded_self_play_games(
                svc, cfg, n_games=4, game_seed_start=1000, game_threads=4)
        self.assertEqual(len(examples), 4)
        self.assertEqual(len(scores), 4)
        for exs in examples:
            self.assertGreater(len(exs), 0)
            e = exs[0]
            self.assertIsInstance(e, Example)
    def test_fail_fast_on_game_error(self):
        """A failing game aborts the iteration by default rather than training
        on a biased subset."""
        cfg = SelfPlayConfig(channels=8, blocks=1, bilinear_dim=16,
                             n_simulations=6, temp_moves=2, device="cpu", seed=3)
        net = _tiny_net()
        # n_determinizations is an int; passing a non-int via a bad cfg field
        # would raise inside play_selfplay_game. Simpler: monkeypatch the game
        # function to raise for one seed.
        import games.kingdomino.threaded_self_play as tsp
        orig = tsp.play_selfplay_game

        def boom(mcts, *, seed, **kw):
            if seed == 1001:
                raise ValueError("injected failure")
            return orig(mcts, seed=seed, **kw)

        tsp.play_selfplay_game = boom
        try:
            with LocalInferenceService(net, "cpu", max_batch=4, max_wait_ms=5) as svc:
                with self.assertRaises(RuntimeError):
                    run_threaded_self_play_games(
                        svc, cfg, n_games=4, game_seed_start=1000,
                        game_threads=4, fail_fast=True)
        finally:
            tsp.play_selfplay_game = orig

    def test_deterministic_order(self):
        """Examples come back in seed order regardless of completion order."""
        cfg = SelfPlayConfig(channels=8, blocks=1, bilinear_dim=16,
                             n_simulations=6, temp_moves=2, device="cpu", seed=3)
        net = _tiny_net()
        with LocalInferenceService(net, "cpu", max_batch=4, max_wait_ms=5) as svc:
            ex_a, sc_a = run_threaded_self_play_games(
                svc, cfg, n_games=5, game_seed_start=2000, game_threads=5)
            ex_b, sc_b = run_threaded_self_play_games(
                svc, cfg, n_games=5, game_seed_start=2000, game_threads=5)
        # Same seeds, sorted order → identical scores sequence run-to-run
        self.assertEqual(sc_a, sc_b)


# ── TEST 5 — Remote A3: parent server + per-worker client ────────────────────
_MODEL_KWARGS = dict(channels=8, blocks=1, bilinear_dim=16)


class TestRemoteService(unittest.TestCase):
    def test_remote_roundtrip_with_initial_weights(self):
        """Server built from full model_kwargs, started with initial weights,
        served through a worker-side client constructed from queue handles."""
        net = KingdominoNet(**_MODEL_KWARGS)
        server = RemoteInferenceServer(
            n_workers=2, model_kwargs=_MODEL_KWARGS, device="cpu",
            max_batch=4, max_wait_ms=10)
        server.start(initial_state_dict=net.state_dict(),
                     wait_until_loaded=True, timeout_s=20)
        # applied_version should be confirmed before any inference
        self.assertGreaterEqual(server.applied_version, 1)
        try:
            req_q, resp_qs = server.worker_handles()
            wclient = RemoteInferenceWorkerClient(req_q, resp_qs, worker_id=1)
            client = wclient.make_client()
            rng = np.random.default_rng(0)
            for _ in range(5):
                v, lg = client(*_rand_inputs(rng), _ALL_IDX, timeout_s=20)
                self.assertIsInstance(v, float)
                self.assertTrue(np.isfinite(v))
                self.assertEqual(lg.shape, (NUM_JOINT_ACTIONS,))
            # client reports the APPLIED version observed in results
            self.assertGreaterEqual(client.model_version, 1)
            wclient.stop()
        finally:
            server.stop()

    def test_remote_version_consistent_under_rapid_updates(self):
        """Rapid updates may coalesce in the bounded weight queue; wait_for_version
        for the LATEST requested version must still succeed (not hang)."""
        net = KingdominoNet(**_MODEL_KWARGS)
        server = RemoteInferenceServer(
            n_workers=1, model_kwargs=_MODEL_KWARGS, device="cpu",
            max_batch=2, max_wait_ms=10)
        server.start(initial_state_dict=net.state_dict(),
                     wait_until_loaded=True, timeout_s=20)
        try:
            # Fire several updates back-to-back without waiting between them.
            v = 0
            for _ in range(5):
                v = server.update_weights(net.state_dict())
            self.assertEqual(v, server.requested_version)
            # The applied version must reach the latest requested version, even
            # though earlier ones may have been dropped from the queue.
            ok = server.wait_for_version(v, timeout_s=20)
            self.assertTrue(ok, "wait_for_version timed out (version desync)")
            self.assertGreaterEqual(server.applied_version, v)
        finally:
            server.stop()

    def test_remote_clean_shutdown(self):
        server = RemoteInferenceServer(2, _MODEL_KWARGS, device="cpu").start()
        time.sleep(0.3)
        server.stop()
        self.assertFalse(server._proc.is_alive())

    def test_remote_worker_enforces_single_client(self):
        """A second make_client() on one worker raises (shared-client invariant)."""
        server = RemoteInferenceServer(1, _MODEL_KWARGS, device="cpu")
        server.start(initial_state_dict=KingdominoNet(**_MODEL_KWARGS).state_dict(),
                     wait_until_loaded=True, timeout_s=20)
        try:
            req_q, resp_qs = server.worker_handles()
            wclient = RemoteInferenceWorkerClient(req_q, resp_qs, worker_id=0)
            wclient.make_client()
            with self.assertRaises(RuntimeError):
                wclient.make_client()
            wclient.stop()
        finally:
            server.stop()


# ── TEST: RequestRouter counts dropped (late) completions ────────────────────
class TestDroppedCompletions(unittest.TestCase):
    def test_dropped_counter(self):
        r = RequestRouter()
        self.assertEqual(r.dropped_completions, 0)
        # complete() for an id never registered (e.g. a late result after timeout)
        r.complete(123, InferenceResult(0.0, np.zeros(NUM_JOINT_ACTIONS, np.float32), 1))
        self.assertEqual(r.dropped_completions, 1)
        # a registered id does not count as dropped
        p = r.register(5)
        r.complete(5, InferenceResult(0.0, np.zeros(NUM_JOINT_ACTIONS, np.float32), 1))
        self.assertTrue(p.event.is_set())
        self.assertEqual(r.dropped_completions, 1)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(
        __import__(__name__)))
    sys.exit(0 if result.wasSuccessful() else 1)