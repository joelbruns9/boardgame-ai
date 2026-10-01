"""
Baseline policies.

:class:`RandomBot` is a **rules fuzzer, not a baseline.**  Uniform random play is
close to worthless as an opponent in Welcome To, because a single careless write
destroys a whole street: numbers must ascend strictly, so putting a 15 in the
first box of a street takes its remaining capacity from ten houses to two.  A
random player blocks itself out within a few turns and every game ends on three
permit refusals.  Its job here is to exercise every branch of the rules, which it
does well.

:class:`GreedyBot` is the actual floor.  It looks one action ahead on its own
score and breaks the many ties on how much future the move leaves -- remaining
capacity, remaining span, and whether the number landed where it proportionally
belongs.  Those tie-breaks are the whole bot: most of a turn scores nothing
directly and the entire question is where the number goes.  Together they are
worth about twelve points a game over scoring alone (see the class docstring),
and greedy is genuinely non-trivial as a result -- parks, pools and estates all
pay immediately, so it accumulates real points.

What greedy structurally cannot do is set up a City Plan, race anyone for the
first-finisher value, or reason about which cards are left — which is exactly the
headroom a trained network has to find.
"""
from __future__ import annotations

import random
from typing import Optional, Protocol

from games.welcome_to import action_codec as codec
from games.welcome_to.constants import TEMP_DELTAS
from games.welcome_to.game import GameConfig, GameState, Phase
from games.welcome_to.sheet import Sheet


class Bot(Protocol):
    def act(self, state: GameState) -> int:  # pragma: no cover - interface
        ...


class RandomBot:
    def __init__(self, rng: Optional[random.Random] = None, seed: int = 0) -> None:
        self.rng = rng or random.Random(seed)

    def act(self, state: GameState) -> int:
        return self.rng.choice(state.legal_actions())


class GreedyBot:
    """One-ply lookahead on the acting player's own score, with three shaping terms.

    Raw score is flat across most of a turn -- writing a number scores nothing by
    itself -- so what the bot is really doing is choosing between equal-scoring
    moves on how much future they leave.  Three terms do that, and measurement
    (advanced variant, one seat, 150 paired seeds) says all three earn their place:

    ==========================================  =======  ==================
    terms                                        score    vs capacity only
    ==========================================  =======  ==================
    capacity only                                  33.9   --
    + total span                                   42.6   +8.2  (t = 4.1)
    + positional fit                               45.8   +3.2  (t = 2.2)
    ==========================================  =======  ==================

    *Capacity* counts how many houses still fit but saturates once a gap is short.
    *Span* keeps discriminating past that point.  *Positional fit* is the rule a
    human uses -- high numbers belong on the right, low on the left -- and it is
    the only one of the three that looks at the action rather than the resulting
    sheet, so it is applied to the write itself.

    Set any weight to zero to ablate that term; ``SPAN_WEIGHT = POSITION_WEIGHT =
    0`` recovers the plain capacity-greedy baseline.
    """

    #: Points per house of remaining capacity.
    CAPACITY_WEIGHT: float = 0.25
    #: Points per unit of total remaining span.
    SPAN_WEIGHT: float = 0.03
    #: Points per box of distance from a number's proportional home.
    POSITION_WEIGHT: float = 0.5

    def __init__(self, rng: Optional[random.Random] = None, seed: int = 0) -> None:
        self.rng = rng or random.Random(seed)

    def act(self, state: GameState) -> int:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]

        me = state.actor
        writing = state.phase is Phase.WRITE_NUMBER
        sheet = state.sheets[me]

        best_value = float("-inf")
        best: list[int] = []
        for action in legal:
            value = self._evaluate(state.step(action), me)
            if writing and self.POSITION_WEIGHT and action != codec.A_PERMIT_REFUSAL:
                value += self.POSITION_WEIGHT * self._fit(state, sheet, action)
            if value > best_value + 1e-9:
                best_value = value
                best = [action]
            elif value > best_value - 1e-9:
                best.append(action)
        return self.rng.choice(best)

    def _fit(self, state: GameState, sheet: Sheet, action: int) -> float:
        """Positional fit of the write ``action`` would make."""
        delta_slot, x, y = codec.decode_write(action)
        assert state.ctx.number is not None
        number = state.ctx.number + TEMP_DELTAS[delta_slot]
        fit = sheet.positional_fit(number, x, y)
        return fit if fit is not None else -99.0

    def _evaluate(self, state: GameState, me: int) -> float:
        breakdown = state.score_breakdown(me, viewer=me)
        sheet = state.sheets[me]
        return (
            breakdown.total
            + self.CAPACITY_WEIGHT * sum(sheet.placement_capacity())
            + self.SPAN_WEIGHT * sheet.total_span()
        )


def play_match(
    bots: list[Bot],
    seed: int = 0,
    config: Optional[GameConfig] = None,
    max_steps: int = 20000,
) -> GameState:
    """Run one game with ``bots[i]`` in seat ``i`` and return the final state."""
    config = config or GameConfig(players=len(bots))
    if config.players != len(bots):
        raise ValueError("one bot per seat")
    state = GameState.new(seed=seed, config=config)
    for _ in range(max_steps):
        if state.is_terminal:
            return state
        state.apply(bots[state.actor].act(state))
    raise RuntimeError("game did not terminate")


def describe_line(state: GameState, action: int) -> str:
    """One-line trace of a decision, for debugging a bot."""
    return (
        f"t{state.turn} p{state.actor} {state.phase.name:<16} "
        f"{codec.describe(action)}"
    )
