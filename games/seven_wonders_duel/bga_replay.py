"""Every decision of a logged BGA game -- both players' -- as engine positions.

The advisor log holds two streams per table (``bga_game_log/table_<id>.jsonl``):

* ``decision`` rows: a full scraped position each time the local player was to
  move. Hidden cards in them are a GUESS (one determinization).
* ``bga_packets`` rows: BGA's own notification stream -- every move by either
  player, with card and wonder names -- but no mid-age card flips.

Neither alone gives the opponent's positions. Together they do, by ANCHORED
replay: start from a decision snapshot, correct its guessed hidden cards to what
was actually revealed by the next snapshot, and apply the packet moves (ours and
theirs) one by one. Every intermediate state is then a real position, including
the one the opponent faced and the instant after each reveal.

What cannot be recovered is said, not guessed: a segment that crosses an Age
boundary stops at it (the new Age's face-down deal is not in the packets), and a
move that maps to no legal action ends its segment with the reason recorded.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .advisor_adapter import _hidden_locations, _is_guild_name, _label, _name_at, _set_name
from .bga_extract import _card_name
from .codec import decode_action, legal_action_indices
from .engine import ActionUse, apply_action
from .game import GameState, Phase

MOVE_TYPES = {
    "wonderSelected",
    "constructBuilding",
    "discardBuilding",
    "constructWonder",
    "progressTokenChosen",
    "opponentDiscardBuilding",
}


@dataclass
class Move:
    player_id: str
    kind: str
    args: dict[str, Any]
    move_id: int


@dataclass
class ReplayedDecision:
    """One decision, with the position the mover faced."""

    state: GameState  # before the move; never mutated after capture
    mover: int  # engine seat
    action_index: int
    label: str
    is_me: bool
    revealed: tuple[tuple[int, int], ...] = ()  # slots this move turned face up
    segment: int = 0


@dataclass
class Replay:
    table_id: str
    my_seat: int
    decisions: list[ReplayedDecision] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


def _rows(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a live log's last line is often partial
    return sorted(out, key=lambda row: row.get("logged_at", 0))


def _events(adapter, rows: list[dict]):
    """Snapshots and moves in arrival order, packets de-duplicated by uid."""

    seen: set[str] = set()
    events: list[tuple[str, Any]] = []
    for row in rows:
        if row.get("kind") == "decision" and row.get("state"):
            try:
                events.append(("snap", adapter.state_from_wire(row["state"]).game))
            except Exception:  # noqa: BLE001 -- a refused capture is just absent
                continue
        elif row.get("kind") == "bga_packets":
            for packet in (row.get("extra") or {}).get("packets") or []:
                for entry in packet.get("data") or []:
                    uid = entry.get("uid") or f"{packet.get('move_id')}:{entry.get('type')}"
                    if uid in seen:
                        continue
                    seen.add(uid)
                    kind = entry.get("type")
                    if kind in MOVE_TYPES:
                        args = entry.get("args") or {}
                        events.append(
                            ("move", Move(str(args.get("playerId")), kind, args, int(packet.get("move_id") or 0)))
                        )
                    elif kind == "nextAgeDraftpoolReveal":
                        events.append(("age", entry.get("args") or {}))
    return events


def _my_player_id(events) -> str | None:
    """The logging player: the one to move at every snapshot."""

    counts: dict[str, int] = {}
    last_move_before: list[str] = []
    for kind, payload in events:
        if kind == "move":
            last_move_before.append(payload.player_id)
        elif kind == "snap" and last_move_before:
            # The player who moved LAST before our snapshot is usually the
            # opponent; the one who moves next is us. Count the next mover.
            pass
    for index, (kind, payload) in enumerate(events):
        if kind != "snap":
            continue
        nxt = next((p for k, p in events[index + 1 :] if k == "move"), None)
        if nxt is not None:
            counts[nxt.player_id] = counts.get(nxt.player_id, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _card_at(game: GameState, slot) -> str | None:
    card = game.tableau.cards.get(tuple(slot)) if slot is not None else None
    return card.card_name if card is not None and card.present else None


def _action_for(game: GameState, move: Move) -> int | None:
    """The legal action index this packet describes, or None."""

    a = move.args
    building = a.get("buildingName")
    building = _card_name(building) if building not in (None, "None", "") else None
    wonder = a.get("wonderName")
    wonder = None if wonder in (None, "None", "") else wonder
    token = a.get("progressTokenName")
    for index in legal_action_indices(game):
        action = decode_action(game, index)
        use = action.use
        if move.kind == "wonderSelected":
            if use is ActionUse.DRAFT_WONDER and action.wonder_name == wonder:
                return index
        elif use is ActionUse.RESOLVE_PENDING_CHOICE:
            # Mid-move choices: a progress token, a destroyed card, a revival.
            want = token if move.kind == "progressTokenChosen" else building
            if action.choice == want:
                return index
        elif move.kind == "constructBuilding":
            if use is ActionUse.CONSTRUCT_BUILDING and _card_at(game, action.slot_id) == building:
                return index
        elif move.kind == "discardBuilding":
            if use is ActionUse.DISCARD_FOR_COINS and _card_at(game, action.slot_id) == building:
                return index
        elif move.kind == "constructWonder":
            if (
                use is ActionUse.CONSTRUCT_WONDER
                and action.wonder_name == wonder
                and _card_at(game, action.slot_id) == building
            ):
                return index
    return None


def _force_card(board: GameState, slot, name: str) -> bool:
    """Make hidden ``slot`` hold ``name`` by swapping it with wherever the
    determinization put ``name``. Keeps the board a valid deal. False if
    ``name`` is not hidden anywhere on this board (the guess cannot hold it)."""

    here = ("slot", tuple(slot))
    if _name_at(board, here) == name:
        return True
    guild = _is_guild_name(name)
    backs = {
        s: ("guild" if _is_guild_name(c.card_name) else "age")
        for s, c in board.tableau.cards.items()
    }
    for there in _hidden_locations(board, backs, (), guild):
        if there != here and _name_at(board, there) == name:
            current = _name_at(board, here)
            _set_name(board, there, current)
            _set_name(board, here, name)
            return True
    return False


def _reconcile(board: GameState, truth: GameState) -> None:
    """Rewrite ``board``'s hidden cards to what ``truth`` later showed face-up."""

    if truth.age != board.age:
        return
    for slot, card in board.tableau.cards.items():
        if not card.present or card.revealed:
            continue
        seen = truth.tableau.cards.get(slot)
        if seen is not None and seen.revealed:
            _force_card(board, slot, seen.card_name)


def _fix_wonder_groups(board: GameState, moves: list[Move]) -> None:
    """During the draft the second group is a guess; the packets name it."""

    if board.phase is not Phase.WONDER_DRAFT:
        return
    picks = [m.args.get("wonderName") for m in moves if m.kind == "wonderSelected"]
    if len(picks) < 8:
        return
    group1 = tuple(picks[4:8])
    known = set(board.wonder_groups[0]) | set(group1)
    box = [w for w in list(board.wonder_groups[1]) + list(board.unused_wonders) if w not in known]
    board.wonder_groups = (board.wonder_groups[0], group1)
    board.unused_wonders = tuple(box[: len(board.unused_wonders)])


def replay_table(adapter, path: str | Path) -> Replay:
    path = Path(path)
    events = _events(adapter, _rows(path))
    me = _my_player_id(events)
    all_moves = [p for k, p in events if k == "move"]
    snaps = [(i, p) for i, (k, p) in enumerate(events) if k == "snap"]
    result = Replay(table_id=path.stem.replace("table_", ""), my_seat=0)

    # Seat of each player id: the logging player is whoever is to move in the
    # snapshots, which the adapter frames from the snapshot itself.
    first = snaps[0][1] if snaps else None
    my_seat = first.active_player if first is not None else 0
    result.my_seat = my_seat
    seat_of = {me: my_seat}

    # Keep the LAST snapshot before each run of moves: a streaming client posts
    # the same decision several times, and a refused partial capture may sit
    # before a good one.
    anchors = []
    for n, (index, game) in enumerate(snaps):
        following = snaps[n + 1][0] if n + 1 < len(snaps) else len(events)
        if any(k == "move" for k, _ in events[index + 1 : following]):
            anchors.append((index, game, following))

    for segment, (index, game, stop) in enumerate(anchors):
        board = copy.deepcopy(game)
        truth = next((g for i, g in snaps if i >= stop), None)
        if truth is not None:
            _reconcile(board, truth)
        _fix_wonder_groups(board, all_moves)
        # Slots whose identity is still the determinizer's guess.
        guessed = {s for s, c in board.tableau.cards.items() if c.present and not c.revealed}
        segment_decisions: list[ReplayedDecision] = []
        for kind, payload in events[index + 1 : stop]:
            if kind == "age":
                result.gaps.append(f"segment {segment}: stopped at the Age {payload.get('ageRoman')} deal")
                break
            if kind != "move":
                continue
            move: Move = payload
            if board.phase is Phase.COMPLETE:
                break
            if board.phase is Phase.CHOOSE_NEXT_START_PLAYER:
                result.gaps.append(f"segment {segment}: stopped at the start-player choice")
                break
            action_index = _action_for(board, move)
            if action_index is None and move.args.get("buildingName") not in (None, "None", ""):
                # The card was face down at the anchor, uncovered by an earlier
                # move in this segment and taken before the next snapshot, so
                # nothing ever showed it face up to reconcile from. It must sit
                # in a slot the anchor had face down and the next snapshot no
                # longer has. Place it there and try again.
                name = _card_name(move.args["buildingName"])
                for index_ in legal_action_indices(board):
                    slot = decode_action(board, index_).slot_id
                    if slot is None or tuple(slot) not in guessed:
                        continue
                    gone = truth is None or not (
                        truth.age == board.age
                        and truth.tableau.cards.get(tuple(slot)) is not None
                        and truth.tableau.cards[tuple(slot)].present
                    )
                    if gone and _force_card(board, slot, name):
                        action_index = _action_for(board, move)
                        if action_index is not None:
                            # The card was there all along: rewrite the earlier
                            # positions of this segment too, or the instant after
                            # the reveal -- the very position reveal luck is read
                            # from -- still shows the old guess.
                            for earlier in segment_decisions:
                                slot_card = earlier.state.tableau.cards.get(tuple(slot))
                                if slot_card is not None and slot_card.present and not slot_card.revealed:
                                    _force_card(earlier.state, slot, name)
                            break
            if action_index is None:
                result.gaps.append(
                    f"segment {segment}: no legal action for {move.kind} "
                    f"{move.args.get('buildingName') or move.args.get('wonderName') or move.args.get('progressTokenName')}"
                )
                break
            mover = board.active_player
            if move.player_id not in seat_of:
                seat_of[move.player_id] = mover
            if seat_of[move.player_id] != mover:
                result.gaps.append(f"segment {segment}: {move.kind} by the player not to move")
                break
            before = copy.deepcopy(board)
            facedown = {s for s, c in board.tableau.cards.items() if c.present and not c.revealed}
            action = decode_action(board, action_index)
            apply_action(board, action)
            revealed = tuple(
                sorted(s for s in facedown if board.tableau.cards.get(s) is not None and board.tableau.cards[s].revealed)
            )
            segment_decisions.append(
                ReplayedDecision(
                    state=before,
                    mover=mover,
                    action_index=action_index,
                    label=_label(action, before),
                    is_me=move.player_id == me,
                    revealed=revealed,
                    segment=segment,
                )
            )
        result.decisions.extend(segment_decisions)
    return result
