"""The state snapshot schema -- one versioned, complete, order-preserving
serialisation of :class:`~games.welcome_to.game.GameState` (``RUST_PORT_PLAN.md``
M0-C).

Three consumers, which is why it is a schema and not a test helper:

* the M1 equivalence gate compares the Python and Rust engines through it;
* cross-engine hand-off (a Rust state opened in Python, or the reverse) needs
  it -- hence :func:`from_snapshot` as well as :func:`to_snapshot`;
* a debugger wants a state it can print, diff and check in as a fixture.

⚠ **Order is preserved everywhere it exists.** The deck and the discard are
ordered lists, not a card census: a census cannot see a card moved from one to
the other, and that move changes both the reveal distribution and the encoder.
The two genuinely unordered fields (``plan_turns``, ``reshuffle_votes``) are
emitted **sorted by key**, because Python dict insertion order is an artefact of
which seat acted first and Rust has no obligation to reproduce it.

⚠ **The field lists are asserted against ``dataclasses.fields``**, not
maintained by hand.  A hand-written list of somebody else's fields rots
silently, and this one is a list of four other dataclasses' fields.
"""

from __future__ import annotations

import random
from dataclasses import fields
from typing import Any

from games.welcome_to.constants import Effect
from games.welcome_to.game import (
    GameConfig,
    GameState,
    Phase,
    TurnCtx,
    rng_kind_of,
)
from games.welcome_to.portable_rng import PortableRng
from games.welcome_to.sheet import Sheet

#: Bump when the shape changes.  Both engines emit it and the gate compares it,
#: so a Rust build against an older schema fails loudly instead of two
#: differently-shaped dictionaries being compared key by key and agreeing.
SNAPSHOT_VERSION: int = 1

_CONFIG_FIELDS: tuple[str, ...] = ("players", "advanced", "expert", "solo_rules")

_SHEET_FIELDS: tuple[str, ...] = (
    "numbers",
    "is_bis",
    "written_turn",
    "fences",
    "top_fences",
    "parks",
    "pools",
    "estate_marks",
    "temps",
    "bis_marks",
    "permits",
    "roundabouts",
)

_CTX_FIELDS: tuple[str, ...] = (
    "slot",
    "number",
    "effect",
    "last_house",
    "built_roundabout",
    "roundabout_declined",
    "refused",
    "plan_slot",
    "pending_sizes",
    "chosen_estates",
)

_STATE_FIELDS: tuple[str, ...] = (
    "config",
    "sheets",
    "public_sheets",
    "deck",
    "deck_pos",
    "discard",
    "stack_new",
    "stack_old",
    "expert_pending",
    "plan_ids",
    "plan_turns",
    "turn",
    "actor",
    "phase",
    "ctx",
    "turn_choice",
    "reshuffle_next_turn",
    "reshuffle_votes",
    "rng",
    "solo_card_drawn",
    "boundary_prepared",
)


def _check_complete(cls, listed: tuple[str, ...]) -> None:
    actual = tuple(f.name for f in fields(cls))
    if actual != listed:
        raise AssertionError(
            f"{cls.__name__} fields changed: the snapshot lists {listed}, the "
            f"dataclass has {actual}.  Update the schema and bump "
            f"SNAPSHOT_VERSION -- a field nobody snapshots is a divergence "
            f"nobody can see."
        )


# Run at import: a field added to any of the four dataclasses must be a loud
# failure everywhere the snapshot is used, not a quiet hole in the M1 gate.
_check_complete(GameConfig, _CONFIG_FIELDS)
_check_complete(Sheet, _SHEET_FIELDS)
_check_complete(TurnCtx, _CTX_FIELDS)
_check_complete(GameState, _STATE_FIELDS)


# ──────────────────────────────────────────────────────────────────────────
# Out
# ──────────────────────────────────────────────────────────────────────────
def sheet_snapshot(sheet: Sheet) -> dict[str, Any]:
    return {
        "numbers": [list(row) for row in sheet.numbers],
        "is_bis": [list(row) for row in sheet.is_bis],
        "written_turn": [list(row) for row in sheet.written_turn],
        "fences": [list(row) for row in sheet.fences],
        "top_fences": [list(row) for row in sheet.top_fences],
        "parks": list(sheet.parks),
        "pools": list(sheet.pools),
        "estate_marks": list(sheet.estate_marks),
        "temps": sheet.temps,
        "bis_marks": sheet.bis_marks,
        "permits": sheet.permits,
        "roundabouts": sheet.roundabouts,
    }


def ctx_snapshot(ctx: TurnCtx) -> dict[str, Any]:
    return {
        "slot": ctx.slot,
        "number": ctx.number,
        "effect": None if ctx.effect is None else int(ctx.effect),
        "last_house": None if ctx.last_house is None else list(ctx.last_house),
        "built_roundabout": ctx.built_roundabout,
        "roundabout_declined": ctx.roundabout_declined,
        "refused": ctx.refused,
        "plan_slot": ctx.plan_slot,
        "pending_sizes": list(ctx.pending_sizes),
        "chosen_estates": [list(e) for e in ctx.chosen_estates],
    }


def rng_snapshot(rng) -> dict[str, Any]:
    """The generator, as far as it is portable.

    A ``PortableRng`` is a single u64 and is snapshotted **exactly**, which is
    what lets the M1 gate catch a divergence in the *number of draws* on the
    step it happens rather than several boundaries later, when the deal
    differs and the cause is long gone.  ``random.Random``'s 625-word state has
    no Rust counterpart, so it is recorded as the kind alone -- a ``cpython``
    game is not cross-engine material in the first place.
    """
    kind = rng_kind_of(rng)
    return {"kind": kind, "state": rng.state if kind == "portable" else None}


def to_snapshot(state: GameState) -> dict[str, Any]:
    """The whole state, in one order-preserving dictionary."""
    return {
        "version": SNAPSHOT_VERSION,
        "config": {
            "players": state.config.players,
            "advanced": state.config.advanced,
            "expert": state.config.expert,
            "solo_rules": state.config.solo_rules,
        },
        "sheets": [sheet_snapshot(s) for s in state.sheets],
        "public_sheets": [sheet_snapshot(s) for s in state.public_sheets],
        "deck": list(state.deck),
        "deck_pos": state.deck_pos,
        "discard": list(state.discard),
        "stack_new": [list(g) for g in state.stack_new],
        "stack_old": [list(g) for g in state.stack_old],
        "expert_pending": list(state.expert_pending),
        "plan_ids": list(state.plan_ids),
        # Sorted by seat: insertion order records which seat acted first, which
        # is an artefact of serialisation rather than state.
        "plan_turns": [sorted(d.items()) for d in state.plan_turns],
        "turn": state.turn,
        "actor": state.actor,
        "phase": int(state.phase),
        "ctx": ctx_snapshot(state.ctx),
        "turn_choice": list(state.turn_choice),
        "reshuffle_next_turn": state.reshuffle_next_turn,
        "reshuffle_votes": sorted(state.reshuffle_votes.items()),
        "rng": rng_snapshot(state.rng),
        "solo_card_drawn": state.solo_card_drawn,
        "boundary_prepared": state.boundary_prepared,
        # Derived, and included deliberately: these are what the rest of the
        # engine reads, so a disagreement is worth catching even when every raw
        # field matches.
        "deck_remaining": state.deck_remaining,
        "is_terminal": state.is_terminal,
    }


# ──────────────────────────────────────────────────────────────────────────
# Back in
# ──────────────────────────────────────────────────────────────────────────
def sheet_from_snapshot(raw: dict[str, Any]) -> Sheet:
    return Sheet(
        numbers=[list(row) for row in raw["numbers"]],
        is_bis=[[bool(v) for v in row] for row in raw["is_bis"]],
        written_turn=[list(row) for row in raw["written_turn"]],
        fences=[[bool(v) for v in row] for row in raw["fences"]],
        top_fences=[[bool(v) for v in row] for row in raw["top_fences"]],
        parks=list(raw["parks"]),
        pools=list(raw["pools"]),
        estate_marks=list(raw["estate_marks"]),
        temps=raw["temps"],
        bis_marks=raw["bis_marks"],
        permits=raw["permits"],
        roundabouts=raw["roundabouts"],
    )


def ctx_from_snapshot(raw: dict[str, Any]) -> TurnCtx:
    return TurnCtx(
        slot=raw["slot"],
        number=raw["number"],
        effect=None if raw["effect"] is None else Effect(raw["effect"]),
        last_house=None if raw["last_house"] is None else tuple(raw["last_house"]),
        built_roundabout=raw["built_roundabout"],
        roundabout_declined=raw["roundabout_declined"],
        refused=raw["refused"],
        plan_slot=raw["plan_slot"],
        pending_sizes=tuple(raw["pending_sizes"]),
        chosen_estates=tuple(tuple(e) for e in raw["chosen_estates"]),
    )


def rng_from_snapshot(raw: dict[str, Any]):
    if raw["kind"] == "portable":
        return PortableRng(raw["state"])
    # A ``cpython`` snapshot carries no state to restore, so what comes back is
    # a *fresh* Mersenne Twister.  Every deterministic field is exact; only
    # future draws differ, which is the price of a generator Rust cannot hold.
    return random.Random()


def from_snapshot(raw: dict[str, Any]) -> GameState:
    """Rebuild a state.  The inverse of :func:`to_snapshot`, up to the RNG."""
    version = raw.get("version")
    if version != SNAPSHOT_VERSION:
        raise ValueError(
            f"snapshot version {version!r} != {SNAPSHOT_VERSION}; the schema "
            "changed and this state was written by a different build"
        )
    cfg = raw["config"]
    return GameState(
        config=GameConfig(
            players=cfg["players"],
            advanced=cfg["advanced"],
            expert=cfg["expert"],
            solo_rules=cfg["solo_rules"],
        ),
        sheets=[sheet_from_snapshot(s) for s in raw["sheets"]],
        public_sheets=[sheet_from_snapshot(s) for s in raw["public_sheets"]],
        deck=list(raw["deck"]),
        deck_pos=raw["deck_pos"],
        discard=list(raw["discard"]),
        stack_new=[list(g) for g in raw["stack_new"]],
        stack_old=[list(g) for g in raw["stack_old"]],
        expert_pending=list(raw["expert_pending"]),
        plan_ids=tuple(raw["plan_ids"]),  # type: ignore[arg-type]
        plan_turns=[{int(p): int(t) for p, t in slot} for slot in raw["plan_turns"]],
        turn=raw["turn"],
        actor=raw["actor"],
        phase=Phase(raw["phase"]),
        ctx=ctx_from_snapshot(raw["ctx"]),
        turn_choice=list(raw["turn_choice"]),
        reshuffle_next_turn=raw["reshuffle_next_turn"],
        reshuffle_votes={int(p): bool(v) for p, v in raw["reshuffle_votes"]},
        rng=rng_from_snapshot(raw["rng"]),
        solo_card_drawn=raw["solo_card_drawn"],
        boundary_prepared=raw["boundary_prepared"],
    )


def diff(left: Any, right: Any, path: str = "") -> list[str]:
    """Every leaf where two snapshots differ, as ``path: left != right`` lines.

    A whole-dictionary ``!=`` says *that* two engines disagree; this says where,
    which is the difference between a two-minute fix and an afternoon.
    """
    out: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                side = "left" if key in left else "right"
                out.append(f"{path}{key}: present in {side} only")
                continue
            out.extend(diff(left[key], right[key], f"{path}{key}."))
        return out
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        here = path.rstrip(".")
        if len(left) != len(right):
            out.append(f"{here}: length {len(left)} != {len(right)}")
            return out
        for i, (a, b) in enumerate(zip(left, right)):
            out.extend(diff(a, b, f"{here}[{i}]."))
        return out
    if left != right:
        out.append(f"{path.rstrip('.')}: {left!r} != {right!r}")
    return out
