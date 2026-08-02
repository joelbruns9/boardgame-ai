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
    gate_rung: int = 0
    """Index into ``GateLadder.rungs`` for the *next* gate (W5.8)."""

    consecutive_probations: int = 0
    """Probations since the last decisive gate; drives the ladder step-up."""

    probations_since_decisive: int = 0
    """Probations since the last promote or revert; drives the probation reset.

    Deliberately *not* ``consecutive_probations``, which the ladder zeroes every
    time it steps up (``_ladder_after``).  Sharing one counter would cap this one
    at ``step_up_after - 1`` and the probation reset could never fire once the
    ladder was on.  Same signal, two consumers, two lifetimes.
    """

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
            "gate_rung": self.gate_rung,
            "consecutive_probations": self.consecutive_probations,
            "probations_since_decisive": self.probations_since_decisive,
        }

    @classmethod
    def from_row(cls, control_state: dict[str, object]) -> "GeneratorState":
        # The ladder fields postdate the schema, so a resume of a pre-ladder run
        # starts at the bottom rung rather than refusing to load.
        return cls(
            mode=GeneratorMode(str(control_state["generator_mode"])),
            bootstrap_state=str(control_state["bootstrap_state"]),
            generator_source=GeneratorSource(str(control_state["next_generator_source"])),
            consecutive_reverts=int(control_state["consecutive_reverts"]),
            current_best_iteration=int(control_state["current_best_iteration"]),
            last_iteration=int(control_state["last_iteration"]),
            gate_rung=int(control_state.get("gate_rung", 0)),
            consecutive_probations=int(control_state.get("consecutive_probations", 0)),
            probations_since_decisive=int(
                control_state.get("probations_since_decisive", 0)
            ),
        )


@dataclass(frozen=True, slots=True)
class GateLadder:
    """Scheduled gate sizes (W5.8).

    The gate plays a fixed number of games, decided *before* the match from the
    ladder position and the games clock -- never from the match's own data.  A
    small gate is safe rather than merely cheap: with a confidence-bounded rule
    on both sides, too little evidence produces probation, not a coin flip.  So
    the size trades promotion *latency* against wall time and nothing else.

    Step **up** after ``step_up_after`` consecutive probations (the learner is
    improving by less than the current size can resolve), **down** one rung after
    a promotion (the next candidate is likely to be resolvable again).
    """

    rungs: tuple[int, ...] = (100, 200, 400, 800)
    step_up_after: int = 2
    floor_games: int = 0
    """Games that must exist before the ladder may step up at all.

    Bootstrap gates are noisy and their probations are not evidence of
    stagnation; without a floor they would ladder the run straight to the top
    rung before the learner has said anything.
    """

    def validate(self) -> None:
        if not self.rungs:
            raise ValueError("gate ladder needs at least one rung")
        if any(size <= 0 or size % 2 for size in self.rungs):
            raise ValueError("every gate rung must be a positive even number")
        if list(self.rungs) != sorted(self.rungs):
            raise ValueError("gate rungs must be ascending")
        if self.step_up_after <= 0:
            raise ValueError("step_up_after must be positive")
        if self.floor_games < 0:
            raise ValueError("floor_games must be non-negative")

    @property
    def top(self) -> int:
        return len(self.rungs) - 1

    def games(self, rung: int) -> int:
        return self.rungs[max(0, min(self.top, rung))]

    @staticmethod
    def fixed(games: int) -> "GateLadder":
        """A one-rung ladder -- the pre-W5.8 behaviour of a single gate size."""

        return GateLadder(rungs=(games,))


def _ladder_after(
    state: GeneratorState,
    decision: str,
    *,
    ladder: GateLadder,
    allow_step_up: bool,
) -> dict[str, int]:
    """Rung and probation counter for the *next* gate.

    Only probation moves the ladder up, and only a promotion moves it down.  A
    revert is decisive evidence, so it clears the counter without changing the
    size: the confirming gate that could trigger a reset is worth running at the
    same resolution that produced the first revert.
    """

    if decision == ACCEPT:
        return {"gate_rung": max(0, state.gate_rung - 1), "consecutive_probations": 0}
    if decision == REJECT:
        return {"gate_rung": state.gate_rung, "consecutive_probations": 0}
    if not allow_step_up:
        # Below the floor the ladder does not count, rather than counting into a
        # debt that is paid the moment the floor clears.  Otherwise bootstrap
        # probations still ladder the run up, just later -- which is the thing
        # the floor exists to prevent.
        return {"gate_rung": state.gate_rung, "consecutive_probations": 0}
    probations = state.consecutive_probations + 1
    if probations >= ladder.step_up_after:
        return {
            "gate_rung": min(ladder.top, state.gate_rung + 1),
            "consecutive_probations": 0,
        }
    return {"gate_rung": state.gate_rung, "consecutive_probations": probations}


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
    ladder: GateLadder | None = None,
    allow_step_up: bool = True,
    probation_reset_after: int = 0,
    gate_stop_reason: str | None = None,
) -> TransitionResult:
    """Map a fixed-N gate decision onto the soft-gate lifecycle action.

    Two rules here exist because run 03 spent 45 iterations degrading without
    the reset ever firing, and both were about how *inconclusive* gates are
    treated:

    * A probation no longer clears ``consecutive_reverts``.  A revert is proof
      the candidate is worse; a probation is only "at this many games we could
      not tell", which is the modal outcome and is not evidence of innocence.
      Clearing on it meant three reverts had to land with no near-miss between
      them, and one 0.465 gate wiped a revert that had just fired.
    * ``probation_reset_after`` makes a run of probations decisive in its own
      right.  Sustained probation is the state where nothing moves: the learner
      is not promoted, the generator is not rolled back, and the counter that
      would reset it never advances.  Accumulating them bounds how long that can
      last.  Off (0) by default, since on an underpowered gate probation may
      mean the *gate* cannot resolve real progress rather than that there is
      none -- pair it with a ladder that buys resolution first.
    """

    if decision not in _GATE_DECISIONS:
        raise ValueError(f"unknown gate decision: {decision!r}")
    if probation_reset_after < 0:
        raise ValueError("probation_reset_after must be non-negative")

    if decision == CONTINUE and gate_stop_reason == "revert_suppressed_knot":
        # This gate is explicitly excluded from lifecycle evidence: its apparent
        # loss crossed a known distribution boundary. Keep the visible probation
        # action, but do not move any reset or resolution counter.
        action = PromotionAction.PROBATION
        return TransitionResult(
            action=action,
            replace_best=False,
            reset_learner=False,
            next_state=replace(
                state,
                generator_source=_generator_after(state.mode, action),
                last_iteration=iteration,
            ),
        )

    rungs = _ladder_after(
        state,
        decision,
        ladder=ladder or GateLadder(),
        allow_step_up=allow_step_up,
    )

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
                probations_since_decisive=0,
                current_best_iteration=iteration,
                last_iteration=iteration,
                **rungs,
            ),
        )

    if decision == CONTINUE:
        probations = state.probations_since_decisive + 1
        reset = probation_reset_after > 0 and probations >= probation_reset_after
        action = (
            PromotionAction.REVERT_RESET if reset else PromotionAction.PROBATION
        )
        return TransitionResult(
            action=action,
            replace_best=False,
            reset_learner=reset,
            next_state=replace(
                state,
                generator_source=_generator_after(state.mode, action),
                # A probation is not evidence the candidate is sound, so it
                # leaves the revert tally where it stands rather than clearing
                # it.  Only a decisive gate moves that counter.
                consecutive_reverts=state.consecutive_reverts,
                probations_since_decisive=0 if reset else probations,
                last_iteration=iteration,
                **rungs,
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
            probations_since_decisive=0,
            last_iteration=iteration,
            **rungs,
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
    ladder: GateLadder | None = None,
    allow_step_up: bool = True,
    probation_reset_after: int = 0,
    gate_stop_reason: str | None = None,
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
            ladder=ladder,
            allow_step_up=allow_step_up,
            probation_reset_after=probation_reset_after,
            gate_stop_reason=gate_stop_reason,
        )
    return not_scheduled_transition(state, iteration)
