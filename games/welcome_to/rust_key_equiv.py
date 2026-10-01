"""M4 constructed equivalence-class gate for the Rust information key.

Random states almost never collide, so they cannot validate an observation key.
This module constructs both sides of the contract: families which must collapse
and field mutations which must separate, then compares Python/Rust partitions in
linear time.  It also checks the invariants licensed by a collision.

    python -m games.welcome_to.rust_key_equiv
"""

from __future__ import annotations

import collections
import random
from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np

from games.welcome_to import encoder as enc
from games.welcome_to import deck_knowledge as dk
from games.welcome_to import macro_codec as mc
from games.welcome_to import mcts
from games.welcome_to import snapshot
from games.welcome_to.bots import GreedyBot
from games.welcome_to.constants import (
    CARD_TABLE,
    NUM_BASE_CARDS,
    PARK_BOXES,
    SOLO_CARD_ID,
    Effect,
)
from games.welcome_to.game import GameConfig, GameState, Phase
from games.welcome_to.rust_encoder import encode_state as rust_encode_state

try:  # pragma: no cover - optional until built
    import welcome_to_rust as wr
except ImportError:  # pragma: no cover
    wr = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Case:
    name: str
    state: GameState
    viewer: int


@dataclass(frozen=True)
class GateReport:
    cases: int
    groups: int
    collisions: int
    separations: int
    key_bytes: int


def _played_midturn(seed: int = 23, players: int = 3, turn: int = 8) -> GameState:
    state = GameState.new(
        seed=seed,
        config=GameConfig(players=players, advanced=True, solo_rules=False),
    )
    bots = [GreedyBot(random.Random(seed * 101 + seat)) for seat in range(players)]
    while not state.is_terminal and (state.turn < turn or state.actor == 0):
        state.apply(bots[state.actor].act(state))
    if state.is_terminal or state.actor == 0:
        raise RuntimeError("failed to construct a later-seat mid-turn position")
    return state


def _twins(excluding: tuple | None = None) -> list[int]:
    by_face: dict[tuple, list[int]] = collections.defaultdict(list)
    for card in range(NUM_BASE_CARDS):
        by_face[CARD_TABLE[card]].append(card)
    return next(
        cards
        for face, cards in by_face.items()
        if len(cards) == 2 and face != excluding
    )


def _terminal_state() -> GameState:
    state = GameState.new(
        seed=81,
        config=GameConfig(players=2, advanced=True, solo_rules=False),
    )
    # A constructed full sheet reaches the same terminal path used by the M1
    # rare-position gate, without waiting for normal play to happen to fill one.
    for x, row in enumerate(state.sheets[0].numbers):
        for y, number in enumerate(row):
            if number is None:
                state.sheets[0].write(y, (x, y), state.turn)
    state.public_sheets = [sheet.copy() for sheet in state.sheets]
    assert state.prepare_turn_boundary() is False
    assert state.is_terminal
    return state


def collapse_cases() -> list[Case]:
    """Families whose differences are invisible to the named viewer."""

    base = _played_midturn()
    viewer = base.actor
    out = [Case("base observation", base, viewer)]

    for seed in range(12):
        out.append(
            Case(
                f"redeterminized deck order {seed}",
                base.redeterminize(random.Random(seed)),
                viewer,
            )
        )

    hidden_sheet = base.copy()
    hidden_target = next(player for player in range(base.config.players) if player != viewer)
    hidden_sheet.sheets[hidden_target].parks[0] = (
        hidden_sheet.sheets[hidden_target].parks[0] + 1
    ) % (PARK_BOXES[0] + 1)
    out.append(Case("opponent live sheet mutation", hidden_sheet, viewer))

    other_vote = base.copy()
    other_vote.reshuffle_votes[hidden_target] = True
    out.append(Case("another seat private vote", other_vote, viewer))

    aggregate = base.copy()
    aggregate.reshuffle_next_turn = not aggregate.reshuffle_next_turn
    out.append(Case("private table-wide reshuffle aggregate", aggregate, viewer))

    slot = next(i for i, card in enumerate(base.stack_new[0]) if card is not None)
    original_face = CARD_TABLE[base.stack_new[0][slot]]
    twins = _twins(excluding=original_face)
    for card in twins:
        state = base.copy()
        state.stack_new[0][slot] = card
        out.append(Case(f"printed twin physical id {card}", state, viewer))

    # A viewer who is not the actor must not see the actor's scratch context.
    observer = (base.actor + 1) % base.config.players
    hidden_ctx_a = base.copy()
    hidden_ctx_b = base.copy()
    hidden_ctx_b.ctx.pending_sizes = (2, 5)
    hidden_ctx_b.ctx.chosen_estates = ((0, 0, 1),)
    out.extend(
        [
            Case("other actor context A", hidden_ctx_a, observer),
            Case("other actor context B", hidden_ctx_b, observer),
        ]
    )

    terminal = _terminal_state()
    out.append(Case("terminal observation", terminal, 0))
    for seed in range(4):
        out.append(
            Case(
                f"terminal redeterminized order {seed}",
                terminal.redeterminize(random.Random(900 + seed)),
                0,
            )
        )
    return out


def _mutated(base: GameState, name: str, mutate) -> Case:
    state = base.copy()
    mutate(state)
    return Case(name, state, base.actor)


def separation_cases() -> list[Case]:
    """Base plus one mutation for every visible component of the Python key."""

    base = _played_midturn(seed=41)
    viewer = base.actor
    cases = [Case("visible baseline", base, viewer)]
    add = lambda name, fn: cases.append(_mutated(base, name, fn))

    cases.append(
        Case("viewer identity", base, (viewer + 1) % base.config.players)
    )

    add("turn", lambda s: setattr(s, "turn", s.turn + 1))
    add("actor", lambda s: setattr(s, "actor", (s.actor + 1) % s.config.players))
    add(
        "phase",
        lambda s: setattr(
            s,
            "phase",
            Phase.WRITE_NUMBER if s.phase is not Phase.WRITE_NUMBER else Phase.CHOOSE_CARDS,
        ),
    )
    add(
        "config advanced",
        lambda s: setattr(
            s,
            "config",
            GameConfig(
                players=s.config.players,
                advanced=not s.config.advanced,
                expert=s.config.expert,
                solo_rules=s.config.solo_rules,
            ),
        ),
    )
    add(
        "config solo_rules",
        lambda s: setattr(
            s,
            "config",
            GameConfig(
                players=s.config.players,
                advanced=s.config.advanced,
                expert=s.config.expert,
                solo_rules=not s.config.solo_rules,
            ),
        ),
    )

    def grow_player_count(state: GameState) -> None:
        state.config = GameConfig(
            players=state.config.players + 1,
            advanced=state.config.advanced,
            expert=state.config.expert,
            solo_rules=state.config.solo_rules,
        )
        state.sheets.append(state.sheets[0].copy())
        state.public_sheets.append(state.public_sheets[0].copy())
        state.turn_choice.append(None)
        state.expert_pending.append(None)

    add("config player count", grow_player_count)
    add("raw discard count", lambda s: s.discard.append(SOLO_CARD_ID))

    original = base.stack_new[0][0]
    different = next(
        card for card in range(NUM_BASE_CARDS) if CARD_TABLE[card] != CARD_TABLE[original]
    )
    add("printed table face", lambda s: s.stack_new[0].__setitem__(0, different))

    table_swap = next(
        (left, right)
        for left in range(3)
        for right in range(left + 1, 3)
        if CARD_TABLE[base.stack_new[0][left]] != CARD_TABLE[base.stack_new[0][right]]
    )
    add(
        "printed table position",
        lambda s: (
            s.stack_new[0].__setitem__(table_swap[0], s.stack_new[0][table_swap[1]]),
            s.stack_new[0].__setitem__(table_swap[1], base.stack_new[0][table_swap[0]]),
        ),
    )

    # _SHEET_FIELDS, field for field, on the viewer's live sheet.
    add("sheet numbers", lambda s: s.sheets[viewer].numbers[0].__setitem__(0, 9))
    add(
        "sheet is_bis",
        lambda s: s.sheets[viewer].is_bis[0].__setitem__(
            0, not s.sheets[viewer].is_bis[0][0]
        ),
    )
    add(
        "sheet written_turn",
        lambda s: s.sheets[viewer].written_turn[0].__setitem__(
            0, s.sheets[viewer].written_turn[0][0] + 1
        ),
    )
    add(
        "sheet fences",
        lambda s: s.sheets[viewer].fences[0].__setitem__(
            0, not s.sheets[viewer].fences[0][0]
        ),
    )
    add(
        "sheet top_fences",
        lambda s: s.sheets[viewer].top_fences[0].__setitem__(
            0, not s.sheets[viewer].top_fences[0][0]
        ),
    )
    add("sheet parks", lambda s: s.sheets[viewer].parks.__setitem__(0, s.sheets[viewer].parks[0] + 1))
    add("sheet pools", lambda s: s.sheets[viewer].pools.__setitem__(0, s.sheets[viewer].pools[0] + 1))
    add(
        "sheet estate_marks",
        lambda s: s.sheets[viewer].estate_marks.__setitem__(
            0, s.sheets[viewer].estate_marks[0] + 1
        ),
    )
    for field in ("temps", "bis_marks", "permits", "roundabouts"):
        add(
            f"sheet {field}",
            lambda s, field=field: setattr(
                s.sheets[viewer], field, getattr(s.sheets[viewer], field) + 1
            ),
        )

    opponent = next(player for player in range(base.config.players) if player != viewer)
    add(
        "opponent public snapshot",
        lambda s: s.public_sheets[opponent].parks.__setitem__(
            0, s.public_sheets[opponent].parks[0] + 1
        ),
    )
    add("plan identity", lambda s: setattr(s, "plan_ids", (0, s.plan_ids[1], s.plan_ids[2])))
    add(
        "visible plan race",
        lambda s: s.plan_turns[0].__setitem__(viewer, s.turn),
    )
    add(
        "visible opponent plan race",
        lambda s: s.plan_turns[1].__setitem__(opponent, s.turn - 2),
    )
    add("deck remaining", lambda s: setattr(s, "deck_pos", s.deck_pos + 1))
    add(
        "deck and discard composition",
        lambda s: s.discard.__setitem__(
            0,
            next(card for card in range(NUM_BASE_CARDS) if CARD_TABLE[card] != CARD_TABLE[s.discard[0]]),
        ),
    )

    # Isolate discard composition from all the correlated fields. Both states
    # have the same raw discard length, table and (saturated) deck composition;
    # only the discard histogram differs.
    saturated_card = 0
    discard_a = base.copy()
    discard_b = base.copy()
    discard_a.discard = [saturated_card] * 10 + [SOLO_CARD_ID]
    discard_b.discard = [saturated_card] * 11
    assert len(discard_a.discard) == len(discard_b.discard)
    assert np.array_equal(
        dk.deck_composition(discard_a, viewer),
        dk.deck_composition(discard_b, viewer),
    )
    assert not np.array_equal(
        dk.discard_composition(discard_a, viewer),
        dk.discard_composition(discard_b, viewer),
    )
    cases.extend(
        [
            Case("isolated discard composition A", discard_a, viewer),
            Case("isolated discard composition B", discard_b, viewer),
        ]
    )
    add("solo-card drawn flag", lambda s: setattr(s, "solo_card_drawn", not s.solo_card_drawn))
    add("viewer reshuffle vote", lambda s: s.reshuffle_votes.__setitem__(viewer, True))
    add("viewer turn choice", lambda s: s.turn_choice.__setitem__(viewer, 0))

    add("ctx slot", lambda s: setattr(s.ctx, "slot", 0))
    add("ctx number", lambda s: setattr(s.ctx, "number", 7))
    add("ctx effect", lambda s: setattr(s.ctx, "effect", Effect.PARK))
    add("ctx last_house", lambda s: setattr(s.ctx, "last_house", (0, 0)))
    for field in ("built_roundabout", "roundabout_declined", "refused"):
        add(f"ctx {field}", lambda s, field=field: setattr(s.ctx, field, not getattr(s.ctx, field)))
    add("ctx plan_slot", lambda s: setattr(s.ctx, "plan_slot", 0))
    add("ctx pending_sizes", lambda s: setattr(s.ctx, "pending_sizes", (3,)))
    add("ctx chosen_estates", lambda s: setattr(s.ctx, "chosen_estates", ((0, 0, 1),)))
    return cases


def _rust_state(state: GameState):
    if wr is None:  # pragma: no cover
        raise RuntimeError("welcome_to_rust is not built; run maturin develop --release")
    return wr.RustGameState.from_snapshot(snapshot.to_snapshot(state))


def _labels(keys: Iterable[Hashable]) -> list[int]:
    groups: dict[Hashable, int] = {}
    out = []
    for key in keys:
        if key not in groups:
            groups[key] = len(groups)
        out.append(groups[key])
    return out


def _assert_group_invariants(cases: list[Case], python_keys: list[tuple]) -> None:
    grouped: dict[tuple, list[Case]] = collections.defaultdict(list)
    for case, key in zip(cases, python_keys, strict=True):
        grouped[key].append(case)

    config = mcts.SearchConfig()
    for group in grouped.values():
        if len(group) < 2:
            continue
        reference = group[0]
        reference_encoding = enc.encode_state(reference.state, reference.viewer)
        reference_macros = mc.search_legal_macros(reference.state)
        terminal_values = []
        for case in group:
            encoded = enc.encode_state(case.state, case.viewer)
            assert all(
                np.array_equal(left, right)
                for left, right in zip(reference_encoding, encoded, strict=True)
            ), f"equal key but different encoding: {reference.name!r}, {case.name!r}"
            assert mc.search_legal_macros(case.state) == reference_macros, (
                f"equal key but different macro legality: {reference.name!r}, {case.name!r}"
            )
            rs = _rust_state(case.state)
            assert mc.search_legal_macros(case.state) == rs.search_legal_macros(True)
            rust_encoding = rust_encode_state(rs, case.viewer)
            assert all(
                np.array_equal(left, right)
                for left, right in zip(encoded, rust_encoding, strict=True)
            )
            if case.state.is_terminal:
                terminal_values.append(
                    mcts.terminal_value(case.state, case.viewer, config)
                )
        if terminal_values:
            assert len(terminal_values) == len(group), "terminal and live states merged"
            assert len(set(terminal_values)) == 1, "equal terminal keys have different values"


def run_gate() -> GateReport:
    if wr is None:  # pragma: no cover
        raise RuntimeError("welcome_to_rust is not built; run maturin develop --release")
    assert wr.INFORMATION_KEY_ABI_VERSION == mcts.INFORMATION_KEY_ABI_VERSION

    collapsed = collapse_cases()
    separated = separation_cases()
    cases = collapsed + separated
    python_keys = [mcts.information_key(case.state, case.viewer) for case in cases]
    rust_keys = [
        bytes(_rust_state(case.state).information_key(case.viewer)) for case in cases
    ]
    python_labels = _labels(python_keys)
    rust_labels = _labels(rust_keys)
    assert rust_labels == python_labels, "Python and Rust induce different key partitions"

    collapse_python = [mcts.information_key(case.state, case.viewer) for case in collapsed]
    _assert_group_invariants(collapsed, collapse_python)

    separation_python = [mcts.information_key(case.state, case.viewer) for case in separated]
    assert len(set(separation_python)) == len(separated), (
        "at least one visible-component mutation did not move the Python key"
    )
    separation_rust = [
        bytes(_rust_state(case.state).information_key(case.viewer)) for case in separated
    ]
    assert len(set(separation_rust)) == len(separated), (
        "at least one visible-component mutation did not move the Rust key"
    )

    groups = len(set(python_keys))
    collisions = len(cases) - groups
    return GateReport(
        cases=len(cases),
        groups=groups,
        collisions=collisions,
        separations=len(separated) - 1,
        key_bytes=sum(len(key) for key in rust_keys),
    )


def main() -> None:
    report = run_gate()
    print(
        f"OK: {report.cases} constructed states, {report.groups} observation "
        f"groups, {report.collisions} intentional collisions, "
        f"{report.separations} visible-component separations; Python and Rust "
        f"partitions agree. {report.key_bytes:,} Rust key bytes inspected."
    )


if __name__ == "__main__":
    main()
