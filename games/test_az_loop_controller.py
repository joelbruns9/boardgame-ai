"""Generic controller tests driven by a fake in-memory lifecycle adapter.

These exercise the game-agnostic soft-gate lifecycle without torch, a Rust
engine, or any game.  Checkpoints are tiny byte files whose contents chain the
learner lineage (``init|t0|t1|...``) so cumulative learning and resume can be
asserted by inspecting bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from games.az_loop import (
    AnchorResult,
    BootstrapPolicy,
    ControllerConfig,
    GenerationResult,
    GeneratorMode,
    GeneratorSource,
    PromotionResult,
    ReplayResult,
    RunController,
    TrainingResult,
    artifact_for,
    atomic_write_bytes,
)
from games.az_loop.checkpoint_lifecycle import TRAINED, UNTRAINED
from games.az_loop.contract import (
    AnchorRequest,
    AssembleRequest,
    GenerateRequest,
    PromotionRequest,
    TrainRequest,
)
from games.az_loop.training_control import (
    GeneratorState,
    PromotionAction,
    gate_transition,
    not_scheduled_transition,
)


class MemoryStore:
    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def append_iteration(self, row: dict[str, Any]) -> None:
        self._rows.append(row)

    def iterations(self) -> list[dict[str, Any]]:
        return list(self._rows)


class FakeAdapter:
    """Deterministic byte-file adapter with scripted promotion decisions."""

    name = "fake"

    def __init__(self, checkpoint_dir: Path, decisions: list[str] | None = None):
        self.work = Path(checkpoint_dir)
        self.decisions = list(decisions or [])
        self.generate_calls: list[tuple[int, GeneratorSource, str]] = []
        self.archived: list[str] = []
        self.resets: list[Path] = []
        self.autosaves: list[int] = []
        self.autosave_should_fail = False

    def initialize_learner(self, *, seed: int):
        path = self.work / "init.pt"
        atomic_write_bytes(path, b"init")
        return artifact_for(
            path, role="candidate", iteration=-1, training_state=UNTRAINED
        )

    def generate(self, request: GenerateRequest) -> GenerationResult:
        sha = artifact_for(
            request.generator_checkpoint,
            role="generator",
            iteration=request.iteration,
            training_state=TRAINED,
        ).sha256
        self.generate_calls.append((request.iteration, request.generator_source, sha))
        return GenerationResult(generated_games=10, metrics={"seconds": 0.0})

    def assemble_replay(self, request: AssembleRequest) -> ReplayResult:
        return ReplayResult(training_games=100, payload=None)

    def train(self, request: TrainRequest) -> TrainingResult:
        learner_bytes = Path(request.learner_checkpoint).read_bytes()
        candidate_bytes = learner_bytes + f"|t{request.iteration}".encode()
        path = self.work / f"candidate_{request.iteration:04d}.pt"
        atomic_write_bytes(path, candidate_bytes)
        return TrainingResult(
            candidate=artifact_for(
                path,
                role="candidate",
                iteration=request.iteration,
                training_state=TRAINED,
            ),
            trained=True,
            metrics={"examples": 100},
        )

    def evaluate_promotion(self, request: PromotionRequest) -> PromotionResult:
        if not self.decisions:
            raise AssertionError("evaluate_promotion called with no scripted decision")
        return PromotionResult(decision=self.decisions.pop(0), metrics={"games": 50})

    def evaluate_anchors(self, request: AnchorRequest):
        return AnchorResult(passed=True, metrics={"opponents": 5})

    def archive_best(self, artifact) -> None:
        self.archived.append(artifact.sha256)

    def on_learner_reset(self, best_checkpoint: Path) -> None:
        self.resets.append(Path(best_checkpoint))

    def autosave(self, iteration: int) -> None:
        if self.autosave_should_fail:
            raise OSError("simulated autosave failure")
        self.autosaves.append(iteration)

    def contract(self) -> dict[str, Any]:
        return {"name": self.name}


def _controller(
    tmp_path: Path,
    *,
    mode: GeneratorMode,
    decisions: list[str] | None = None,
    policy: BootstrapPolicy = BootstrapPolicy.AUTO_FIRST_TRAINED,
    iterations: int = 1,
    promotion_every: int = 1,
    revert_reset_after: int = 0,
    store: MemoryStore | None = None,
    anchor_every: int = 0,
    autosave_every: int = 0,
) -> tuple[RunController, FakeAdapter, MemoryStore]:
    ckpt = tmp_path / "checkpoints"
    adapter = FakeAdapter(ckpt, decisions)
    store = store or MemoryStore()
    controller = RunController(
        adapter=adapter,
        store=store,
        checkpoint_dir=ckpt,
        config=ControllerConfig(
            mode=mode,
            bootstrap_policy=policy,
            promotion_every=promotion_every,
            revert_reset_after=revert_reset_after,
            anchor_gate_every_promotions=anchor_every,
            buffer_autosave_every=autosave_every,
            iterations=iterations,
        ),
    )
    return controller, adapter, store


def _sha(path: Path) -> str:
    return artifact_for(
        path, role="x", iteration=0, training_state=TRAINED
    ).sha256


# -- pure state machine -----------------------------------------------------


def _soft_state(**kw) -> GeneratorState:
    base = dict(
        mode=GeneratorMode.SOFT_GATE,
        bootstrap_state=TRAINED,
        generator_source=GeneratorSource.LATEST,
    )
    base.update(kw)
    return GeneratorState(**base)


def test_accept_promotes_and_switches_best_iteration():
    result = gate_transition(_soft_state(), "accept", revert_reset_after=0, iteration=5)
    assert result.action == PromotionAction.PROMOTE
    assert result.replace_best and not result.reset_learner
    assert result.next_state.current_best_iteration == 5
    assert result.next_state.generator_source == GeneratorSource.LATEST


def test_continue_is_probation_and_keeps_best():
    result = gate_transition(_soft_state(), "continue", revert_reset_after=0, iteration=5)
    assert result.action == PromotionAction.PROBATION
    assert not result.replace_best and not result.reset_learner
    assert result.next_state.generator_source == GeneratorSource.LATEST


def test_reject_reverts_generator_to_best_without_reset():
    result = gate_transition(_soft_state(), "reject", revert_reset_after=3, iteration=5)
    assert result.action == PromotionAction.REVERT
    assert not result.replace_best and not result.reset_learner
    assert result.next_state.generator_source == GeneratorSource.CURRENT_BEST
    assert result.next_state.consecutive_reverts == 1


def test_reject_triggers_reset_at_threshold():
    state = _soft_state(consecutive_reverts=2)
    result = gate_transition(state, "reject", revert_reset_after=3, iteration=9)
    assert result.action == PromotionAction.REVERT_RESET
    assert result.reset_learner
    assert result.next_state.consecutive_reverts == 0


def test_not_scheduled_leaves_revert_counter_untouched():
    state = _soft_state(consecutive_reverts=2)
    result = not_scheduled_transition(state, iteration=7)
    assert result.action == PromotionAction.NOT_SCHEDULED
    assert result.next_state.consecutive_reverts == 2
    assert not result.replace_best and not result.reset_learner


def test_continue_resets_revert_counter():
    state = _soft_state(consecutive_reverts=2)
    result = gate_transition(state, "continue", revert_reset_after=3, iteration=7)
    assert result.next_state.consecutive_reverts == 0


# -- controller: checkpoint separation --------------------------------------


def test_bootstrap_installs_learner_as_latest_and_best(tmp_path):
    controller, adapter, store = _controller(
        tmp_path, mode=GeneratorMode.SOFT_GATE, iterations=1
    )
    (row,) = controller.run()
    assert row["promotion_action"] == PromotionAction.BOOTSTRAP_PROMOTE.value
    assert _sha(controller.latest_path) == _sha(controller.current_best_path)
    # Bootstrap did not gate, so no scripted decision was consumed.
    assert row["promotion_scheduled"] is False
    # Untrained init weights must never be archived to HOF.
    assert adapter.archived == []
    assert controller.current_best_path.read_bytes() == b"init|t0"


def test_continue_advances_latest_without_touching_best(tmp_path):
    controller, adapter, store = _controller(
        tmp_path,
        mode=GeneratorMode.SOFT_GATE,
        decisions=["continue"],
        iterations=2,
    )
    rows = controller.run()
    best_bytes = controller.current_best_path.read_bytes()
    latest_bytes = controller.latest_path.read_bytes()
    assert best_bytes == b"init|t0"  # bootstrap best, unchanged by probation
    assert latest_bytes == b"init|t0|t1"  # cumulative lineage on latest
    assert rows[1]["promotion_action"] == PromotionAction.PROBATION.value
    assert rows[1]["latest_sha256"] != rows[1]["current_best_sha256"]


def test_reject_cannot_overwrite_best(tmp_path):
    controller, adapter, store = _controller(
        tmp_path,
        mode=GeneratorMode.SOFT_GATE,
        decisions=["reject"],
        iterations=2,
    )
    controller.run()
    assert controller.current_best_path.read_bytes() == b"init|t0"
    assert adapter.archived == []  # rejected candidate never enters HOF


def test_accept_promotes_and_archives_previous_trained_best(tmp_path):
    controller, adapter, store = _controller(
        tmp_path,
        mode=GeneratorMode.SOFT_GATE,
        decisions=["accept"],
        iterations=2,
    )
    controller.run()
    # iter0 bootstrap best = init|t0 (trained); iter1 accept promotes init|t0|t1
    assert controller.current_best_path.read_bytes() == b"init|t0|t1"
    assert adapter.archived == [_sha_of_bytes(b"init|t0")]


def _sha_of_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


# -- controller: generator modes --------------------------------------------


def test_latest_mode_always_generates_with_latest(tmp_path):
    controller, adapter, store = _controller(
        tmp_path,
        mode=GeneratorMode.LATEST,
        decisions=["reject", "reject"],
        iterations=3,
    )
    controller.run()
    for _iteration, source, _sha256 in adapter.generate_calls:
        assert source == GeneratorSource.LATEST


def test_current_best_mode_always_generates_with_best(tmp_path):
    controller, adapter, store = _controller(
        tmp_path,
        mode=GeneratorMode.CURRENT_BEST,
        decisions=["continue", "continue"],
        iterations=3,
    )
    controller.run()
    for _iteration, source, _sha256 in adapter.generate_calls[1:]:
        assert source == GeneratorSource.CURRENT_BEST


def test_soft_gate_switches_generator_to_best_after_reject(tmp_path):
    controller, adapter, store = _controller(
        tmp_path,
        mode=GeneratorMode.SOFT_GATE,
        decisions=["reject", "continue"],
        iterations=3,
    )
    controller.run()
    # iter0 bootstrap -> latest ; iter1 generates with latest, gate rejects ;
    # iter2 must generate with current_best (recovery data).
    assert adapter.generate_calls[0][1] == GeneratorSource.LATEST
    assert adapter.generate_calls[1][1] == GeneratorSource.LATEST
    assert adapter.generate_calls[2][1] == GeneratorSource.CURRENT_BEST


def test_revert_reset_restores_best_weights_into_latest(tmp_path):
    controller, adapter, store = _controller(
        tmp_path,
        mode=GeneratorMode.SOFT_GATE,
        decisions=["reject", "reject"],
        iterations=3,
        revert_reset_after=2,
    )
    rows = controller.run()
    # Second reject hits the threshold: learner weights reset to current_best.
    assert rows[2]["promotion_action"] == PromotionAction.REVERT_RESET.value
    assert controller.latest_path.read_bytes() == controller.current_best_path.read_bytes()
    assert len(adapter.resets) == 1


# -- controller: promotion cadence ------------------------------------------


def test_promotion_every_two_skips_alternate_iterations(tmp_path):
    controller, adapter, store = _controller(
        tmp_path,
        mode=GeneratorMode.SOFT_GATE,
        decisions=["continue"],  # only one gate should actually run
        iterations=3,
        promotion_every=2,
    )
    rows = controller.run()
    actions = [row["promotion_action"] for row in rows]
    assert actions[0] == PromotionAction.BOOTSTRAP_PROMOTE.value
    # Post-bootstrap eligible ordinals 1,2 -> gate only on the 2nd.
    assert actions[1] == PromotionAction.NOT_SCHEDULED.value
    assert actions[2] == PromotionAction.PROBATION.value
    assert adapter.decisions == []  # exactly one decision consumed


# -- controller: resume ------------------------------------------------------


def test_resume_continues_lineage_and_verifies_hashes(tmp_path):
    store = MemoryStore()
    first, _adapter, _store = _controller(
        tmp_path,
        mode=GeneratorMode.SOFT_GATE,
        decisions=["continue"],
        iterations=2,
        store=store,
    )
    first.run()
    assert first.latest_path.read_bytes() == b"init|t0|t1"

    # A fresh controller + fresh adapter, same store and checkpoint dir.
    second, adapter2, _store2 = _controller(
        tmp_path,
        mode=GeneratorMode.SOFT_GATE,
        decisions=["continue", "continue"],
        iterations=2,
        store=store,
    )
    rows = second.run()
    assert [row["iteration"] for row in rows] == [2, 3]
    assert second.latest_path.read_bytes() == b"init|t0|t1|t2|t3"
    # The generator for the first resumed iteration must be the on-disk latest.
    assert adapter2.generate_calls[0][0] == 2


def test_resume_refuses_hash_mismatch(tmp_path):
    store = MemoryStore()
    first, _adapter, _store = _controller(
        tmp_path, mode=GeneratorMode.SOFT_GATE, decisions=["continue"], iterations=2,
        store=store,
    )
    first.run()
    # Corrupt latest.pt out from under the manifest.
    first.latest_path.write_bytes(b"tampered")
    second, _adapter2, _store2 = _controller(
        tmp_path, mode=GeneratorMode.SOFT_GATE, iterations=1, store=store
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        second.initialize()


def test_resume_refuses_missing_established_checkpoint(tmp_path):
    store = MemoryStore()
    first, _adapter, _store = _controller(
        tmp_path, mode=GeneratorMode.SOFT_GATE, decisions=["continue"], iterations=2,
        store=store,
    )
    first.run()
    first.current_best_path.unlink()
    second, _adapter2, _store2 = _controller(
        tmp_path, mode=GeneratorMode.SOFT_GATE, iterations=1, store=store
    )
    with pytest.raises(FileNotFoundError, match="refusing to substitute random"):
        second.initialize()


def test_resume_refuses_mode_switch(tmp_path):
    store = MemoryStore()
    first, _adapter, _store = _controller(
        tmp_path, mode=GeneratorMode.SOFT_GATE, decisions=["continue"], iterations=2,
        store=store,
    )
    first.run()
    second, _adapter2, _store2 = _controller(
        tmp_path, mode=GeneratorMode.STRICT_GATE, iterations=1, store=store
    )
    with pytest.raises(ValueError, match="cannot resume"):
        second.initialize()


# -- controller: strict-gate compatibility & atomicity ----------------------


def test_strict_gate_generates_with_best_and_promotes_on_accept(tmp_path):
    controller, adapter, store = _controller(
        tmp_path,
        mode=GeneratorMode.STRICT_GATE,
        policy=BootstrapPolicy.GATE,
        decisions=["accept", "reject"],
        iterations=2,
    )
    controller.run()
    for _iteration, source, _sha256 in adapter.generate_calls:
        assert source == GeneratorSource.CURRENT_BEST


def test_no_temp_files_linger_after_run(tmp_path):
    controller, adapter, store = _controller(
        tmp_path,
        mode=GeneratorMode.SOFT_GATE,
        decisions=["accept", "reject"],
        iterations=3,
    )
    controller.run()
    assert list(controller.checkpoint_dir.glob("*.tmp")) == []


# -- Milestone 3: full synthetic transition matrix --------------------------


def test_synthetic_transition_matrix_exact_hashes_and_next_generator(tmp_path):
    """One run through bootstrap -> accept -> continue -> reject -> continue,
    asserting the exact byte identity landing in latest/best and the generator
    identity at every generation call."""

    controller, adapter, store = _controller(
        tmp_path,
        mode=GeneratorMode.SOFT_GATE,
        decisions=["accept", "continue", "reject", "continue"],
        iterations=5,
    )
    rows = controller.run()
    h = _sha_of_bytes

    # iter0 bootstrap: latest and best are the same first trained learner.
    assert rows[0]["promotion_action"] == PromotionAction.BOOTSTRAP_PROMOTE.value
    assert rows[0]["latest_sha256"] == rows[0]["current_best_sha256"] == h(b"init|t0")

    # iter1 accept -> promote: best advances to latest; outgoing best archived.
    assert rows[1]["promotion_action"] == PromotionAction.PROMOTE.value
    assert rows[1]["latest_sha256"] == rows[1]["current_best_sha256"] == h(b"init|t0|t1")
    assert adapter.archived == [h(b"init|t0")]

    # iter2 continue -> probation: latest advances, best frozen.
    assert rows[2]["promotion_action"] == PromotionAction.PROBATION.value
    assert rows[2]["latest_sha256"] == h(b"init|t0|t1|t2")
    assert rows[2]["current_best_sha256"] == h(b"init|t0|t1")

    # iter3 reject -> revert: latest still advances (learner preserved), best frozen.
    assert rows[3]["promotion_action"] == PromotionAction.REVERT.value
    assert rows[3]["latest_sha256"] == h(b"init|t0|t1|t2|t3")
    assert rows[3]["current_best_sha256"] == h(b"init|t0|t1")

    # iter4 continue after revert: generated with best, trained from the
    # PRESERVED latest (recovery continues the rejected learner).
    assert rows[4]["promotion_action"] == PromotionAction.PROBATION.value
    assert rows[4]["latest_sha256"] == h(b"init|t0|t1|t2|t3|t4")

    # Only iter4's generation uses current_best (post-reject recovery).
    sources = [source for _iter, source, _sha in adapter.generate_calls]
    assert sources == [
        GeneratorSource.LATEST,
        GeneratorSource.LATEST,
        GeneratorSource.LATEST,
        GeneratorSource.LATEST,
        GeneratorSource.CURRENT_BEST,
    ]
    # A single accept archived exactly one outgoing best; nothing else entered HOF.
    assert adapter.archived == [h(b"init|t0")]


def test_autosave_fires_on_cadence_only(tmp_path):
    controller, adapter, store = _controller(
        tmp_path,
        mode=GeneratorMode.SOFT_GATE,
        decisions=["continue", "continue", "continue"],
        iterations=4,
        autosave_every=2,
    )
    controller.run()
    # Completed iterations 2 and 4 trigger autosave (iteration indices 1 and 3).
    assert adapter.autosaves == [1, 3]


def test_autosave_disabled_by_default(tmp_path):
    controller, adapter, store = _controller(
        tmp_path,
        mode=GeneratorMode.SOFT_GATE,
        decisions=["continue"],
        iterations=2,
    )
    controller.run()
    assert adapter.autosaves == []


def test_autosave_failure_is_not_fatal(tmp_path, capsys):
    controller, adapter, store = _controller(
        tmp_path,
        mode=GeneratorMode.SOFT_GATE,
        decisions=["continue"],
        iterations=2,
        autosave_every=1,
    )
    adapter.autosave_should_fail = True
    rows = controller.run()  # must complete despite every autosave raising
    assert [row["iteration"] for row in rows] == [0, 1]
    assert "buffer autosave failed" in capsys.readouterr().out


def test_revert_does_not_delete_latest_and_recovery_trains_from_it(tmp_path):
    controller, adapter, store = _controller(
        tmp_path,
        mode=GeneratorMode.SOFT_GATE,
        decisions=["reject", "continue"],
        iterations=3,
    )
    rows = controller.run()
    # After the reject, latest.pt still holds the rejected learner (not deleted,
    # not reset), distinct from the protected best.
    assert controller.latest_path.exists()
    assert rows[1]["latest_sha256"] == _sha_of_bytes(b"init|t0|t1")
    assert rows[1]["current_best_sha256"] == _sha_of_bytes(b"init|t0")
    # iter2 generates with best but the candidate descends from the preserved latest.
    assert rows[2]["latest_sha256"] == _sha_of_bytes(b"init|t0|t1|t2")
