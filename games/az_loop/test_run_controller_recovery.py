from __future__ import annotations

import json

from .run_controller import RunController


def test_pending_successor_restores_roles_and_rolls_back_adapter_outputs(tmp_path):
    class Adapter:
        rolled_back = []

        def rollback_iteration(self, iteration):
            self.rolled_back.append(iteration)

    checkpoint_dir = tmp_path / "checkpoints"
    recovery = checkpoint_dir / "_recovery"
    recovery.mkdir(parents=True)
    latest = checkpoint_dir / "latest.pt"
    best = checkpoint_dir / "current_best.pt"
    latest.write_bytes(b"interrupted latest")
    best.write_bytes(b"interrupted best")
    (recovery / "latest.pt").write_bytes(b"committed latest")
    (recovery / "current_best.pt").write_bytes(b"committed best")
    pending = checkpoint_dir / "pending_iteration.json"
    pending.write_text(json.dumps({"iteration": 70}), encoding="utf-8")

    controller = object.__new__(RunController)
    controller.pending_path = pending
    controller.recovery_dir = recovery
    controller.latest_path = latest
    controller.current_best_path = best
    controller.adapter = Adapter()

    controller._reconcile_pending([{"iteration": 69}])

    assert latest.read_bytes() == b"committed latest"
    assert best.read_bytes() == b"committed best"
    assert controller.adapter.rolled_back == [70]
    assert not pending.exists()
