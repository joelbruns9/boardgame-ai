"""The lockstep equivalence harness — ``RUST_PORT_PLAN.md`` M1 and M2.

Drives the Python and Rust engines through the *same* game, action for action,
and compares the complete M0-C snapshot after every action, the macro
vocabulary at every macro root (M2), and the read API, the boundary triple and
``redeterminize`` once per turn.  The gate is 8,000 games at 2/3/4 seats with
the advanced variant on and off; the pytest module runs a small sample of it,
and

    python -m games.welcome_to.rust_equiv --games 8000

runs the gate itself.  What it compares, and why each item is in the list, is
in the plan; the short version is that a census cannot see a card move from the
deck to the discard, and an omitted ``public_sheets`` would let an information
leak pass a green rules gate.

⚠ **Action selection is shared, not mirrored.** Both engines are offered the
same action index, drawn from the *Python* engine's legal list by a separate
``PortableRng``.  If the lists disagree that is itself the failure, and it is
reported before anything is applied.

⚠ **Raw order is what is compared.** PUCT's first-max tie-break depends on the
order ``legal_actions`` returns, so a set comparison would pass a state the
search would play differently.  A set/multiplicity diff is reported *as a
diagnostic* when the ordered lists differ.
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

from games.welcome_to import action_codec as codec
from games.welcome_to import macro_codec as mc
from games.welcome_to import snapshot as sn
from games.welcome_to.bots import GreedyBot
from games.welcome_to.game import GameConfig, GameState
from games.welcome_to.portable_rng import PortableRng

try:  # pragma: no cover - the crate is optional until it is built
    import welcome_to_rust as wr
except ImportError:  # pragma: no cover
    wr = None  # type: ignore[assignment]

#: The gate's configuration matrix: 2/3/4 seats, advanced on and off (M0-A).
GATE_CONFIGS: tuple[GameConfig, ...] = tuple(
    GameConfig(players=p, advanced=adv, expert=False, solo_rules=False)
    for p in (2, 3, 4)
    for adv in (False, True)
)

#: How the shared action is chosen.  **Three of them, because one is not
#: enough**, and that was measured rather than assumed:
#:
#: * ``random`` — uniform over legal actions.  Cheap and wide, but it walks into
#:   permit refusals: 60/60 sampled games ended on the third refusal, City Plans
#:   were validated 9 times in 60 games and the deck reformed once.  A gate made
#:   only of these would barely touch plan scoring or a reshuffle.
#: * ``no-refusal`` — uniform, but never *choosing* a refusal when anything else
#:   is legal.  Longer games at the same cost.
#: * ``greedy`` — :class:`~games.welcome_to.bots.GreedyBot`, ~35× slower per
#:   action but the only driver that reaches deep positions: 12 games gave 32
#:   plan validations and 12 deck reforms against 9 and 3.
#:
#: ⚠ The driver only chooses *which legal action*; both engines are then given
#: the same index.  Nothing about it can hide a divergence, and everything about
#: it changes which states get compared.
DRIVERS: tuple[str, ...] = ("random", "no-refusal", "greedy")

#: How often M2's "every macro applies identically" claim is checked: on the
#: first macro root of every Nth turn.
#:
#: ⚠ **Sampled by root, never by macro.** A root offers up to ~100 macros and
#: each comparison re-serialises two whole states in Python; at every root the
#: 8,000-game gate would take hours.  Checking *all* the macros at fewer roots
#: keeps the property whole — "every macro this root offers applies identically"
#: — where checking some macros at every root would leave a hole in exactly the
#: place a rare macro lives.
MACRO_APPLY_EVERY: int = 3


class Divergence(AssertionError):
    """The two engines disagree.  Carries where, not just that."""


@dataclass
class GameReport:
    seed: int
    config: GameConfig
    driver: str
    steps: int
    scores: tuple[int, ...]


def _config_kwargs(config: GameConfig) -> dict:
    return {
        "players": config.players,
        "advanced": config.advanced,
        "expert": config.expert,
        "solo_rules": config.solo_rules,
    }


def _legal_diagnostic(py_legal: Sequence[int], rs_legal: Sequence[int]) -> str:
    """A set/multiplicity diff, for when the ordered lists differ."""
    left, right = Counter(py_legal), Counter(rs_legal)
    if left == right:
        return "same multiset, different order"
    missing = sorted((left - right).elements())
    extra = sorted((right - left).elements())
    return f"python-only {missing}, rust-only {extra}"


def _compare(py: GameState, rs, seed: int, config: GameConfig, step: int, action: Optional[int]) -> None:
    left = sn.to_snapshot(py)
    right = rs.snapshot()
    lines = sn.diff(left, right)
    if lines:
        head = "\n  ".join(lines[:20])
        more = "" if len(lines) <= 20 else f"\n  ... and {len(lines) - 20} more"
        raise Divergence(
            f"state diverged at step {step} of seed {seed} {config} "
            f"(after action {action}):\n  {head}{more}"
        )


def _check_accessors(py: GameState, rs, where: str) -> None:
    """The read API the encoder and the search will call.

    None of it appears in the snapshot, because none of it is *state* — but
    `next_effects` is what §6.2's certainty is made of, `plan_turns_for` and
    `reshuffle_vote_for` are the information-set filters, and a viewer-scoped
    score is what an opponent model reads. A port can get every field of the
    state right and still answer these differently.
    """

    def fail(what: str, left, right) -> None:
        raise Divergence(f"{what} diverged at {where}:\n  python {left}\n  rust   {right}")

    def effects(values) -> list:
        return [None if v is None else int(v) for v in values]

    for player in range(py.config.players):
        left = [(n, None if e is None else int(e)) for n, e in py.visible_cards(player)]
        right = [(n, e) for n, e in rs.visible_cards(player)]
        if left != right:
            fail(f"visible_cards({player})", left, right)
        if effects(py.next_effects(player)) != effects(rs.next_effects(player)):
            fail(f"next_effects({player})", py.next_effects(player), rs.next_effects(player))
        if list(py.table_cards(player)) != list(rs.table_cards(player)):
            fail(f"table_cards({player})", py.table_cards(player), rs.table_cards(player))
        if list(py.playable_slots(player)) != list(rs.playable_slots(player)):
            fail(f"playable_slots({player})", py.playable_slots(player), rs.playable_slots(player))
        if py.reshuffle_vote_for(player) != rs.reshuffle_vote_for(player):
            fail(f"reshuffle_vote_for({player})", py.reshuffle_vote_for(player), rs.reshuffle_vote_for(player))
        for slot in range(3):
            left_turns = sorted(py.plan_turns_for(player, slot).items())
            right_turns = sorted(rs.plan_turns_for(player, slot))
            if left_turns != right_turns:
                fail(f"plan_turns_for({player}, {slot})", left_turns, right_turns)
        # Viewer-scoped scoring: the same numbers an opponent model would read.
        for viewer in (None, player):
            if list(py.scores(viewer)) != list(rs.scores(viewer)):
                fail(f"scores(viewer={viewer})", py.scores(viewer), rs.scores(viewer))
            if list(py.plan_scores(viewer)) != list(rs.plan_scores(viewer)):
                fail(f"plan_scores(viewer={viewer})", py.plan_scores(viewer), rs.plan_scores(viewer))
            if list(py.temp_scores(viewer)) != list(rs.temp_scores(viewer)):
                fail(f"temp_scores(viewer={viewer})", py.temp_scores(viewer), rs.temp_scores(viewer))
            # The vector methods above cover every target's *total*. Check every
            # component too: two swapped components can preserve the total and
            # would otherwise pass.
            for target in range(py.config.players):
                left_break = py.score_breakdown(target, viewer)
                right_break = rs.score_breakdown(target, viewer)
                for field, value in right_break.items():
                    left_value = (
                        left_break.total
                        if field == "total"
                        else getattr(left_break, field)
                    )
                    if left_value != value:
                        fail(
                            f"score_breakdown({target}, {viewer}).{field}",
                            left_value,
                            value,
                        )

    if list(py.scorable_plan_slots()) != list(rs.scorable_plan_slots()):
        fail("scorable_plan_slots", py.scorable_plan_slots(), rs.scorable_plan_slots())
    if list(py.returns()) != list(rs.returns()):
        fail("returns", py.returns(), rs.returns())
    if py.end_of_game_reason() != rs.end_of_game_reason():
        fail("end_of_game_reason", py.end_of_game_reason(), rs.end_of_game_reason())


def _check_macros(py: GameState, rs, where: str, apply_all: bool) -> None:
    """M2: the macro vocabulary, at one macro root.

    Three separate claims, and the third is the expensive one:

    * ``legal_macros`` agrees **in order** — the same reason raw primitive order
      is compared, one layer up;
    * ``search_legal_macros`` agrees at **both** settings of
      ``prune_roundabout_pass``, because the flag exists to be switched;
    * every macro applies end to end identically (``apply_all``).

    ⚠ **The third claim is sampled by root, not thinned by macro.** A macro root
    offers up to ~100 macros and each comparison re-serialises two whole states
    in Python; measured, checking every root costs 5.4× the M1 gate, against
    2.7× at :data:`MACRO_APPLY_EVERY` = 3.  Applying *all* of them at fewer
    roots keeps the property whole ("every macro this root offers applies
    identically"); dropping some macros at every root would leave the hole
    exactly where a rare macro lives.

    ⚠ **And sampling is safe for a reason, not just cheap.** Given M1 — the same
    primitive applied to the same state gives the same state, over 1.5M actions
    — a macro applying identically *follows* from its primitive sequence being
    identical, and that sequence is compared here at every sampled root, for
    every macro.  The snapshot comparison is the empirical backstop, kept
    because a reduction that turns out to be wrong is exactly what a gate is
    for.
    """
    if not mc.is_macro_root(py):
        # WRITE_NUMBER is inside a macro. Both engines must refuse to decide
        # here — a port that answered would be inventing a decision point.
        if rs.is_macro_root:
            raise Divergence(f"is_macro_root disagrees at {where}: python False, rust True")
        for engine, call in (("python", lambda: mc.legal_macros(py)), ("rust", rs.legal_macros)):
            try:
                call()
            except ValueError:
                continue
            raise Divergence(f"{engine} did not refuse legal_macros at WRITE_NUMBER ({where})")
        return
    if not rs.is_macro_root:
        raise Divergence(f"is_macro_root disagrees at {where}: python True, rust False")

    py_macros = mc.legal_macros(py)
    rs_macros = rs.legal_macros()
    if py_macros != rs_macros:
        raise Divergence(
            f"legal_macros diverged at {where} (phase {py.phase.name}):\n"
            f"  python {py_macros}\n  rust   {rs_macros}\n"
            f"  {_legal_diagnostic(py_macros, rs_macros)}"
        )
    for prune in (True, False):
        left = mc.search_legal_macros(py, prune_roundabout_pass=prune)
        right = rs.search_legal_macros(prune)
        if left != right:
            raise Divergence(
                f"search_legal_macros(prune_roundabout_pass={prune}) diverged at "
                f"{where}:\n  python {left}\n  rust   {right}"
            )

    if not apply_all:
        return
    for index in py_macros:
        if list(mc.primitives_for(index)) != list(wr.macro_primitives(index)):
            raise Divergence(
                f"primitives_for({index}) diverged at {where}: "
                f"{mc.primitives_for(index)} != {wr.macro_primitives(index)}"
            )
        py_next = mc.step_macro(py, index)
        rs_next = rs.step_macro(index)
        lines = sn.diff(sn.to_snapshot(py_next), rs_next.snapshot())
        if lines:
            raise Divergence(
                f"macro {index} ({mc.describe(index)}) applied differently at "
                f"{where}:\n  " + "\n  ".join(lines[:20])
            )


def _check_boundary_triple(py: GameState, rs, seed: int, config: GameConfig, step: int) -> None:
    """The public boundary triple, in lockstep, on copies.

    ``_end_turn`` is *built* from ``prepare`` / ``sample`` / ``apply``, so a game
    played straight through exercises the three of them — but only through the
    private path, and only on the deal the engine happened to make.  A search
    calls them itself, on a state of its choosing, with its own generator. That
    is what this checks: the same afterstate, the same sampled draws from the
    same generator state, the same generator left behind, and the same state
    after applying the outcome.
    """
    py_after = py.copy()
    rs_after = rs.copy()
    py_open = py_after.prepare_turn_boundary()
    rs_open = rs_after.prepare_turn_boundary()
    if py_open != rs_open:
        raise Divergence(
            f"prepare_turn_boundary diverged at step {step} of seed {seed} "
            f"{config}: python {py_open} != rust {rs_open}"
        )
    _compare(py_after, rs_after, seed, config, step, "prepare_turn_boundary")
    if not py_open:
        return

    probe_seed = (seed * 0x9E37_79B9 + step) & ((1 << 64) - 1)
    probe = PortableRng(probe_seed)
    outcome = py_after.sample_boundary_outcome(probe)
    draws, reformed, rs_probe_state = rs_after.sample_boundary_outcome(probe_seed)
    if list(outcome.draws) != list(draws) or outcome.reformed != reformed:
        raise Divergence(
            f"sample_boundary_outcome diverged at step {step} of seed {seed} "
            f"{config}: python {outcome.draws} reformed={outcome.reformed} != "
            f"rust {draws} reformed={reformed}"
        )
    if probe.state != rs_probe_state:
        raise Divergence(
            f"the sampling generator diverged at step {step} of seed {seed} "
            f"{config}: python {probe.state} != rust {rs_probe_state} -- the two "
            "engines drew a different number of times"
        )
    # Sampling must not touch the afterstate: a chance node calls it repeatedly.
    _compare(py_after, rs_after, seed, config, step, "sample_boundary_outcome")

    py_after.apply_boundary_outcome(outcome)
    rs_after.apply_boundary_outcome(list(draws))
    _compare(py_after, rs_after, seed, config, step, "apply_boundary_outcome")


def _check_redeterminize(
    py: GameState, rs, seed: int, config: GameConfig, step: int
) -> None:
    """MCTS's root operation, from one shared search-generator state.

    This belongs in the played-game gate, not only in a standalone unit test:
    redeterminization is sensitive to ``deck_pos`` and to the exact hidden tail,
    so checking one fixed mid-game position cannot support the plan's advertised
    "once per turn" coverage.
    """
    probe_seed = (
        seed * 0xD1B5_4A32_D192_ED03
        + step * 0x9E37_79B9_7F4A_7C15
        + py.turn
    ) & ((1 << 64) - 1)
    probe = PortableRng(probe_seed)
    py_next = py.redeterminize(probe)
    rs_next, rs_probe_state = rs.redeterminize(probe_seed)
    if probe.state != rs_probe_state:
        raise Divergence(
            f"redeterminize's caller RNG diverged at step {step} of seed {seed} "
            f"{config}: python {probe.state} != rust {rs_probe_state}"
        )
    lines = sn.diff(sn.to_snapshot(py_next), rs_next.snapshot())
    if lines:
        raise Divergence(
            f"redeterminize diverged at step {step} of seed {seed} {config}:\n  "
            + "\n  ".join(lines[:20])
        )


def _has_private_midturn_state(state: GameState) -> bool:
    """Whether an information-set accessor has something non-public to hide."""
    return (
        any(live != public for live, public in zip(state.sheets, state.public_sheets))
        or any(
            completed_turn == state.turn
            for slot in state.plan_turns
            for completed_turn in slot.values()
        )
        or bool(state.reshuffle_votes)
    )


def _choose(driver: str, py: GameState, legal: Sequence[int], picker: PortableRng, bot: GreedyBot) -> int:
    """Which legal action to play.  See :data:`DRIVERS`."""
    if driver == "greedy":
        return bot.act(py)
    pool = list(legal)
    if driver == "no-refusal":
        # Only ever *declines to choose* a refusal; when it is the sole legal
        # action the engine still takes it, which is the rule and not a policy.
        pool = [a for a in pool if a != codec.A_PERMIT_REFUSAL] or list(legal)
    return pool[picker.randrange(len(pool))]


def check_game(
    seed: int,
    config: GameConfig,
    driver: str = "random",
    max_steps: int = 20000,
    check_boundaries: bool = True,
    check_macros: bool = True,
    macro_apply_every: int = MACRO_APPLY_EVERY,
) -> GameReport:
    """Play one game in lockstep.  Raises :class:`Divergence` on any mismatch."""
    if wr is None:  # pragma: no cover
        raise RuntimeError("welcome_to_rust is not built; run `maturin develop --release`")
    if driver not in DRIVERS:
        raise ValueError(f"unknown driver {driver!r}; expected one of {DRIVERS}")
    if macro_apply_every < 1:
        raise ValueError("macro_apply_every must be at least 1")

    py = GameState.new(seed=seed, config=config)
    rs = wr.RustGameState(seed, **_config_kwargs(config))
    picker = PortableRng(seed ^ 0x5745_4C43_4F4D_4521)  # "WELCOME!"
    # The bot's own generator only breaks ties between equally-valued actions;
    # it never touches either engine.
    bot = GreedyBot(random.Random(seed))

    _compare(py, rs, seed, config, 0, None)
    checked_turn = -1
    checked_hidden_turn = -1

    for step in range(1, max_steps + 1):
        if py.is_terminal:
            break
        where = f"step {step} of seed {seed} {config}"
        first_of_turn = py.turn != checked_turn
        # Boundary checks alone cannot catch an information leak: live and
        # public sheets are equal there. The first hand-off to a later seat is
        # the cheapest high-value checkpoint -- seat 0 has finished mutating
        # its live sheet (and possibly a plan/vote), while opponents must still
        # read the public snapshot.
        if (
            check_boundaries
            and py.actor > 0
            and py.turn != checked_hidden_turn
            and _has_private_midturn_state(py)
        ):
            _check_accessors(py, rs, where + " (mid-turn information set)")
            checked_hidden_turn = py.turn
        if check_macros:
            # ⚠ Applying every macro is the expensive claim (see _check_macros),
            # so it runs on the first root of every ``macro_apply_every`` turns.
            apply_all = first_of_turn and (py.turn % macro_apply_every == 0)
            _check_macros(py, rs, where, apply_all=apply_all)
        if check_boundaries and first_of_turn:
            _check_accessors(py, rs, where)
            _check_boundary_triple(py, rs, seed, config, step)
            _check_redeterminize(py, rs, seed, config, step)
        if first_of_turn:
            checked_turn = py.turn
        py_legal = py.legal_actions()
        rs_legal = rs.legal_actions()
        if py_legal != rs_legal:
            raise Divergence(
                f"legal_actions diverged at step {step} of seed {seed} {config} "
                f"(phase {py.phase.name}, actor {py.actor}):\n"
                f"  python {py_legal}\n  rust   {rs_legal}\n"
                f"  {_legal_diagnostic(py_legal, rs_legal)}"
            )
        action = _choose(driver, py, py_legal, picker, bot)
        py.apply(action)
        rs.apply(action)
        _compare(py, rs, seed, config, step, action)
    else:  # pragma: no cover - a game that will not end is a rules bug
        raise Divergence(f"game {seed} {config} did not terminate in {max_steps} steps")

    if rs.is_terminal is not py.is_terminal:
        raise Divergence(f"terminality diverged at seed {seed} {config}")
    if list(py.scores()) != list(rs.scores()):
        raise Divergence(
            f"scores diverged at seed {seed} {config}: "
            f"python {py.scores()} != rust {rs.scores()}"
        )
    if list(py.winners()) != list(rs.winners()):
        raise Divergence(
            f"winners diverged at seed {seed} {config}: "
            f"python {py.winners()} != rust {rs.winners()}"
        )
    if list(py.ranking()) != list(rs.ranking()):
        raise Divergence(
            f"ranking diverged at seed {seed} {config}: "
            f"python {py.ranking()} != rust {rs.ranking()}"
        )
    return GameReport(
        seed=seed, config=config, driver=driver, steps=step, scores=tuple(py.scores())
    )


def gate(
    games: int,
    configs: Sequence[GameConfig] = GATE_CONFIGS,
    drivers: Sequence[str] = DRIVERS,
    seed0: int = 0,
    macro_apply_every: int = MACRO_APPLY_EVERY,
    check_macros: bool = True,
) -> Iterator[GameReport]:
    """``games`` games, dealt round-robin across ``configs`` and ``drivers``.

    The driver advances once per *full pass* over the configuration matrix, so
    every configuration meets every driver.  Cycling both on ``i`` would tie a
    seat count to a driver — with six configurations and three drivers, 4-seat
    advanced would have been the greedy one and nothing else ever would.
    """
    for i in range(games):
        yield check_game(
            seed0 + i,
            configs[i % len(configs)],
            drivers[(i // len(configs)) % len(drivers)],
            check_macros=check_macros,
            macro_apply_every=macro_apply_every,
        )


# ──────────────────────────────────────────────────────────────────────────
# Constructed positions — the states play does not reach
# ──────────────────────────────────────────────────────────────────────────
#
# Measured over 60 greedy games: every one ended on the third permit refusal
# except four that filled a sheet, and none completed all three City Plans. So
# two of `isEndOfGame`'s three clauses, the queued reshuffle and the
# exact-empty reform are effectively untested by played games — and rare is
# exactly where a rules divergence survives a gate.
#
# These hand the Rust engine a Python state through the M0-C snapshot, which is
# the direction M0-C exists for.


def _mid_game(seed: int, config: GameConfig, turn: int = 5) -> GameState:
    bot = GreedyBot(random.Random(seed))
    state = GameState.new(seed=seed, config=config)
    while not state.is_terminal and state.turn < turn:
        state.apply(bot.act(state))
    return state


def _fill_sheet(state: GameState, player: int) -> None:
    """Every box written, so ``isEndOfGame``'s first clause fires."""
    sheet = state.sheets[player]
    for x, row in enumerate(sheet.numbers):
        for y in range(len(row)):
            if row[y] is None:
                sheet.write(y, (x, y), state.turn)


#: What each constructed case is *supposed* to do at its boundary, so that a
#: case which silently stopped exercising its rule fails instead of passing.
#: ``open`` is ``prepare_turn_boundary``'s return; ``draws`` and ``reformed``
#: describe the outcome sampled from the afterstate.
Expectation = dict


def constructed_cases() -> Iterator[tuple[str, GameState, Expectation]]:
    """Named positions that a played game does not produce."""
    config = GameConfig(players=2, advanced=True, expert=False, solo_rules=False)

    state = _mid_game(3, config)
    _fill_sheet(state, 0)
    yield ("a sheet with no free box (isEndOfGame clause 1)", state, {"open": False})

    state = _mid_game(4, config)
    for slot in range(3):
        state.plan_turns[slot][0] = state.turn - 1
    yield ("all three plans completed (isEndOfGame clause 2)", state, {"open": False})

    state = _mid_game(5, config)
    state.sheets[0].permits = 3
    yield ("a third permit refusal (isEndOfGame clause 3)", state, {"open": False})

    state = _mid_game(6, config)
    state.reshuffle_next_turn = True
    yield (
        "a queued reshuffle (six draws, two batches)",
        state,
        {"open": True, "draws": 6, "reformed": True},
    )

    state = _mid_game(7, config)
    state.discard.extend(state.deck[state.deck_pos :])
    state.deck_pos = len(state.deck)
    yield (
        "an exact-empty deck (the boundary reforms first)",
        state,
        {"open": True, "draws": 3, "reformed": True},
    )

    state = _mid_game(8, config)
    state.discard.extend(state.deck[state.deck_pos :])
    state.deck_pos = len(state.deck)
    state.reshuffle_next_turn = True
    yield (
        "an empty deck AND a queued reshuffle",
        state,
        {"open": True, "draws": 6, "reformed": True},
    )

    state = _mid_game(9, config, turn=12)
    yield (
        "an ordinary mid-game position, for contrast",
        state,
        {"open": True, "draws": 3, "reformed": False},
    )


def check_constructed(
    name: str, py: GameState, expect: Expectation, check_macros: bool = True
) -> None:
    """Compare a constructed position, its boundary and its scoring."""
    if wr is None:  # pragma: no cover
        raise RuntimeError("welcome_to_rust is not built; run `maturin develop --release`")
    snap = sn.to_snapshot(py)
    rs = wr.RustGameState.from_snapshot(snap)
    lines = sn.diff(snap, rs.snapshot())
    if lines:
        raise Divergence(f"{name}: the snapshot did not survive the hand-off:\n  " + "\n  ".join(lines[:20]))
    if py.legal_actions() != rs.legal_actions():
        raise Divergence(
            f"{name}: legal_actions disagree\n  python {py.legal_actions()}\n"
            f"  rust   {rs.legal_actions()}"
        )
    if list(py.scores()) != list(rs.scores()):
        raise Divergence(f"{name}: scores {py.scores()} != {rs.scores()}")
    if list(py.winners()) != list(rs.winners()):
        raise Divergence(f"{name}: winners {py.winners()} != {rs.winners()}")

    _check_accessors(py, rs, name)
    if check_macros:
        _check_macros(py, rs, name, apply_all=True)
    _check_boundary_triple(py, rs, seed=-1, config=py.config, step=0)
    _check_redeterminize(py, rs, seed=-1, config=py.config, step=0)

    # ⚠ And check the case still does what it was written to do. A constructed
    # position that quietly stopped being a queued reshuffle would keep passing
    # for ever, testing an ordinary boundary twice.
    probe = py.copy()
    opened = probe.prepare_turn_boundary()
    if opened != expect["open"]:
        raise Divergence(
            f"{name}: expected prepare_turn_boundary() == {expect['open']}, got {opened}"
        )
    if not opened:
        return
    outcome = probe.sample_boundary_outcome(PortableRng(1234))
    if "draws" in expect and len(outcome.draws) != expect["draws"]:
        raise Divergence(
            f"{name}: expected {expect['draws']} draws, got {len(outcome.draws)}"
        )
    if "reformed" in expect and outcome.reformed != expect["reformed"]:
        raise Divergence(
            f"{name}: expected reformed={expect['reformed']}, got {outcome.reformed}"
        )


def check_all_constructed(check_macros: bool = True) -> int:
    cases = list(constructed_cases())
    for name, state, expect in cases:
        check_constructed(name, state, expect, check_macros=check_macros)
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description="M1 Python/Rust engine equivalence")
    parser.add_argument("--games", type=int, default=8000, help="games to play (gate: 8000)")
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--progress", type=int, default=100, help="report every N games")
    parser.add_argument(
        "--macro-apply-every",
        type=int,
        default=MACRO_APPLY_EVERY,
        help="apply every macro on the first root of every Nth turn (M2)",
    )
    parser.add_argument(
        "--m1-only",
        action="store_true",
        help="skip M2 macro comparisons; run the corrected engine gate only",
    )
    args = parser.parse_args()

    cases = check_all_constructed(check_macros=not args.m1_only)
    print(f"{cases} constructed positions agree (the ones play does not reach)", flush=True)

    started = time.perf_counter()
    steps = 0
    played = Counter()
    for i, report in enumerate(
        gate(
            args.games,
            seed0=args.seed0,
            macro_apply_every=args.macro_apply_every,
            check_macros=not args.m1_only,
        ),
        start=1,
    ):
        steps += report.steps
        played[(report.config.players, report.config.advanced, report.driver)] += 1
        if args.progress and i % args.progress == 0:
            elapsed = time.perf_counter() - started
            print(
                f"{i}/{args.games} games  {steps} actions  "
                f"{elapsed:.1f}s  {i / elapsed:.1f} games/s",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    print(
        f"OK: {args.games} games, {steps} actions compared, {elapsed:.1f}s. "
        f"Both engines agree after every action."
    )
    # Which games were actually played, so a green gate says what it covered
    # rather than only that it was green.
    for (players, advanced, driver), count in sorted(played.items()):
        print(f"  {players}p advanced={int(advanced)} {driver:11s} {count:5d} games")


if __name__ == "__main__":
    main()
