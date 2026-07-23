"""Generator modes, lifecycle state, and the pure soft-gate transition policy.

This module holds the *decisions* of the training lifecycle with no I/O: given
the current :class:`GeneratorState` and a gate outcome, it returns the next
state plus the checkpoint effects the controller must apply.  Keeping it pure
makes the accept/continue/reject/bootstrap/revert-reset matrix exhaustively
unit-testable without touching disk, models, or a game engine.

Design notes tied to the conversion plan:

* The four generator modes mirror Kingdomino's proven ``GENERATOR_MODES``.
* The three soft-gate actions reuse Seven Wonders Duel's paired-SPRT decisions
  (``accept``/``continue``/``reject``) directly rather than adding a second
  raw win-rate decision system.
* ``consecutive_reverts`` is counted in **gate checks, not iterations**.  An
  iteration that runs no gate (``not_scheduled``) never touches the counter --
  it neither increments nor resets it.  Only an actual gate decision moves it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from .checkpoint_lifecycle import TRAINED, UNTRAINED


class GeneratorMode(str, Enum):
    LATEST = "latest"
    CURRENT_BEST = "current_best"
    STRICT_GATE = "strict_gate"
    SOFT_GATE = "soft_gate"


class BootstrapPolicy(str, Enum):
    AUTO_FIRST_TRAINED = "auto_first_trained"
    GATE = "gate"


class GeneratorSource(str, Enum):
    LATEST = "latest"
    CURRENT_BEST = "current_best"


class PromotionAction(str, Enum):
    NOT_SCHEDULED = "not_scheduled"
    BOOTSTRAP_PROMOTE = "bootstrap_promote"
    PROMOTE = "promote"
    PROBATION = "probation"
    REVERT = "revert"
    REVERT_RESET = "revert_reset"


ACCEPT = "accept"
CONTINUE = "continue"
REJECT = "reject"
_GATE_DECISIONS = (ACCEPT, CONTINUE, REJECT)


@dataclass(frozen=True, slots=True)
class GeneratorState:
    """Durable control state threaded across iterations.

    Persisted inside each completed iteration row so a resume can reconstruct
    the exact learner/generator/best relationship from the last row alone.
    """

    mode: GeneratorMode
    bootstrap_state: str = UNTRAINED
    generator_source: GeneratorSource = GeneratorSource.LATEST
    consecutive_reverts: int = 0
    current_best_iteration: int = -1
    last_iteration: int = -1

    def as_row(self) -> dict[str, object]:
        """Self-contained control-state snapshot for a completed iteration row.

        ``next_generator_source`` is the source the *next* iteration will
        generate with; it is deliberately distinct from any top-level
        ``generator_source`` field that records which model produced *this*
        iteration's data.
        """

        return {
            "generator_mode": self.mode.value,
            "bootstrap_state": self.bootstrap_state,
            "next_generator_source": self.generator_source.value,
            "consecutive_reverts": self.consecutive_reverts,
            "current_best_iteration": self.current_best_iteration,
            "last_iteration": self.last_iteration,
        }

    @classmethod
    def from_row(cls, control_state: dict[str, object]) -> "GeneratorState":
        return cls(
            mode=GeneratorMode(str(control_state["generator_mode"])),
            bootstrap_state=str(control_state["bootstrap_state"]),
            generator_source=GeneratorSource(str(control_state["next_generator_source"])),
            consecutive_reverts=int(control_state["consecutive_reverts"]),
            current_best_iteration=int(control_state["current_best_iteration"]),
            last_iteration=int(control_state["last_iteration"]),
        )


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """The checkpoint effects the controller applies for one gate outcome."""

    action: PromotionAction
    next_state: GeneratorState
    replace_best: bool  # install latest -> current_best
    reset_learner: bool  # reset latest weights back to current_best


def initial_state(mode: GeneratorMode) -> GeneratorState:
    """The state a fresh run starts in, before any training has occurred."""

    source = (
        GeneratorSource.CURRENT_BEST
        if mode in (GeneratorMode.CURRENT_BEST, GeneratorMode.STRICT_GATE)
        else GeneratorSource.LATEST
    )
    return GeneratorState(mode=mode, bootstrap_state=UNTRAINED, generator_source=source)


def select_generator_source(state: GeneratorState) -> GeneratorSource:
    """Which checkpoint self-play uses for the *next* generation call.

    Fixed by mode for every mode except ``soft_gate``, where it follows the last
    gate action (``current_best`` only after a reject, otherwise ``latest``).
    Until the first successful training both files hold identical weights, so an
    untrained run generates with ``latest`` regardless.
    """

    if state.mode == GeneratorMode.LATEST:
        return GeneratorSource.LATEST
    if state.mode in (GeneratorMode.CURRENT_BEST, GeneratorMode.STRICT_GATE):
        return GeneratorSource.CURRENT_BEST
    return state.generator_source


def _generator_after(mode: GeneratorMode, action: PromotionAction) -> GeneratorSource:
    if mode == GeneratorMode.LATEST:
        return GeneratorSource.LATEST
    if mode in (GeneratorMode.CURRENT_BEST, GeneratorMode.STRICT_GATE):
        return GeneratorSource.CURRENT_BEST
    # soft_gate: revert switches generation to the protected best for recovery.
    if action in (PromotionAction.REVERT, PromotionAction.REVERT_RESET):
        return GeneratorSource.CURRENT_BEST
    return GeneratorSource.LATEST


def bootstrap_transition(state: GeneratorState, iteration: int) -> TransitionResult:
    """First successful training installs the learner as latest *and* best."""

    action = PromotionAction.BOOTSTRAP_PROMOTE
    return TransitionResult(
        action=action,
        replace_best=True,
        reset_learner=False,
        next_state=replace(
            state,
            bootstrap_state=TRAINED,
            generator_source=_generator_after(state.mode, action),
            consecutive_reverts=0,
            current_best_iteration=iteration,
            last_iteration=iteration,
        ),
    )


def not_scheduled_transition(state: GeneratorState, iteration: int) -> TransitionResult:
    """A trained iteration with no gate this cycle: nothing but the clock moves."""

    return TransitionResult(
        action=PromotionAction.NOT_SCHEDULED,
        replace_best=False,
        reset_learner=False,
        next_state=replace(state, last_iteration=iteration),
    )


def gate_transition(
    state: GeneratorState,
    decision: str,
    *,
    revert_reset_after: int,
    iteration: int,
) -> TransitionResult:
    """Map a paired-SPRT decision onto the soft-gate lifecycle action."""

    if decision not in _GATE_DECISIONS:
        raise ValueError(f"unknown gate decision: {decision!r}")

    if decision == ACCEPT:
        action = PromotionAction.PROMOTE
        return TransitionResult(
            action=action,
            replace_best=True,
            reset_learner=False,
            next_state=replace(
                state,
                bootstrap_state=TRAINED,
                generator_source=_generator_after(state.mode, action),
                consecutive_reverts=0,
                current_best_iteration=iteration,
                last_iteration=iteration,
            ),
        )

    if decision == CONTINUE:
        action = PromotionAction.PROBATION
        return TransitionResult(
            action=action,
            replace_best=False,
            reset_learner=False,
            next_state=replace(
                state,
                generator_source=_generator_after(state.mode, action),
                consecutive_reverts=0,
                last_iteration=iteration,
            ),
        )

    # decision == REJECT
    count = state.consecutive_reverts + 1
    reset = revert_reset_after > 0 and count >= revert_reset_after
    action = PromotionAction.REVERT_RESET if reset else PromotionAction.REVERT
    return TransitionResult(
        action=action,
        replace_best=False,
        reset_learner=reset,
        next_state=replace(
            state,
            generator_source=_generator_after(state.mode, action),
            consecutive_reverts=0 if reset else count,
            last_iteration=iteration,
        ),
    )


def is_bootstrap_eligible(state: GeneratorState, policy: BootstrapPolicy) -> bool:
    """True when the first trained learner should auto-install as best.

    Only the ``auto_first_trained`` policy bootstraps; ``gate`` preserves the
    old behavior of gating the first candidate against untrained weights.
    """

    return (
        state.bootstrap_state == UNTRAINED
        and policy == BootstrapPolicy.AUTO_FIRST_TRAINED
    )


def decide_transition(
    state: GeneratorState,
    *,
    policy: BootstrapPolicy,
    promotion_scheduled: bool,
    gate_decision: str | None,
    revert_reset_after: int,
    iteration: int,
) -> TransitionResult:
    """Single entry point the controller calls after training an iteration.

    ``gate_decision`` must be provided iff a promotion gate actually ran this
    iteration.  Bootstrap short-circuits the gate entirely.
    """

    if is_bootstrap_eligible(state, policy):
        return bootstrap_transition(state, iteration)
    if promotion_scheduled:
        if gate_decision is None:
            raise ValueError("scheduled promotion requires a gate decision")
        return gate_transition(
            state,
            gate_decision,
            revert_reset_after=revert_reset_after,
            iteration=iteration,
        )
    return not_scheduled_transition(state, iteration)
