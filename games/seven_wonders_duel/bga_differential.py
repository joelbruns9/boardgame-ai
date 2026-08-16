"""Replay real BGA games against our engine and compare BGA's own arithmetic.

This is the differential harness: BGA publishes a number for almost everything
it does, so every one of those numbers is a test our engine either passes or
fails. It is how the military off-by-one was found, and it costs nothing but
the games you were going to play anyway.

What is compared, per real game:

  per construct   ``cost``                       vs ``minimum_payment(...)``
                  ``payment.coinReward``         vs ``_apply_card_coin_effects``
                  ``payment.economyProgressTokenCoins`` vs the Economy transfer
                  ``payment.militaryNewPosition``vs ``_apply_military``
                  ``payment.militaryTokens``     vs the token penalty we charge
  per age boundary and at victory, BGA publishes ``playersSituation``:
                  ``coins``                      vs our tracked treasury
                  ``player_score_*``             vs ``score_player`` per category
  at the end, ``nextPlayerTurnEndGameScoring`` + ``endGameCategoryUpdate``:
                  every final score category, including guild/treasury/military

**Where the numbers come from.** Rows of kind ``bga_packets`` in the advisor's
game log, whose ``extra["packets"]`` holds BGA's notification packets verbatim,
captured by the extension while you play. Only those packets carry BGA's
arithmetic -- a position snapshot says what the board looks like, never what
BGA charged for it. Nothing here touches BGA itself, deliberately: its archive
endpoint refuses to serve the notification log of a finished table
(``Cannot find gamenotifs log file of an archived table``), so a fetch-it-back
design would have rotted within days. ``--packets-json`` replays a raw
``{table_id: [packet, ...]}`` dump for corpora captured some other way.

**Seat framing.** Engine seat 0 is BGA's start player, the same convention
``bga_extract._seat_order`` uses, and the military sign depends on it. The
packets do not carry ``startPlayerId``, but they do carry the wonder draft,
whose first pick is the start player's; that derivation is cross-checked
against the ``startPlayerId`` recorded in the position rows whenever both are
present, so a wrong assumption fails loudly rather than flipping the military
track.

**Gaps.** A missing packet is worse than a missing game: skip one construct and
every later coin comparison is measured against a treasury that silently drifted.
So a table whose ``move_id`` sequence is incomplete is reported and skipped, not
replayed.

Only the two cities and the military track are reconstructed -- that is all any
compared quantity depends on, so no tableau is needed.

    python -m games.seven_wonders_duel.bga_differential
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from games.advisor.game_log import iter_rows, log_dir_for

from .bga_extract import _card_name
from .data import CARDS_BY_NAME, WONDERS_BY_NAME, CardColor
from .engine import (
    _apply_card_coin_effects,
    _apply_military,
    _apply_progress_immediate,
    minimum_payment,
    score_player,
)
from .game import CityState, GameState, Phase, TableauState

#: ``kind`` of the game-log rows carrying BGA notification packets.
PACKET_KIND = "bga_packets"

GAME_ID = "seven_wonders_duel"
STARTING_COINS = 7
# Keyed by each token's band start, matching game.new_game.
INITIAL_TOKENS = {-6: 5, -3: 2, 3: 2, 6: 5}

# Tokens that rebate construction costs. Counted, not asserted: the point of the
# counter is to show whether a real game has ever exercised the rebate path.
REBATE_TOKENS = {"Masonry", "Architecture"}

COUNTER_NAMES = (
    "priced",
    "coin_reward",
    "opponent_coin_loss",
    "with_rebate_token",
    "military",
    "token_penalty",
    "score_checks",
    "coin_checks",
    "economy_transfer",
    "progress_token_coins",
    "final_score_checks",
)


class World:
    """Cities + military track, carried through one replayed game."""

    def __init__(self) -> None:
        self.cities = (CityState(coins=STARTING_COINS), CityState(coins=STARTING_COINS))
        self.conflict = 0
        self.tokens = dict(INITIAL_TOKENS)

    def state(self) -> GameState:
        """A GameState carrying just the fields the compared code reads."""

        return GameState(
            seed=0,
            first_player=0,
            phase=Phase.PLAY_AGE,
            active_player=0,
            age=1,
            cities=self.cities,
            available_progress_tokens=(),
            unused_progress_tokens=(),
            wonder_groups=((), ()),
            unused_wonders=(),
            wonder_offer=[],
            wonder_round=0,
            wonder_pick_index=0,
            age_decks={},
            removed_age_cards={},
            selected_guilds=(),
            unused_guilds=(),
            tableau=TableauState(age=1, cards={}),
            discard_pile=[],
            buried_cards=[],
            retired_wonders=set(),
            pending_choice=None,
            pending_extra_turn=False,
            pending_shields=0,
            conflict_position=self.conflict,
            military_tokens_remaining=self.tokens,
            winner=None,
            victory_type=None,
            final_scores=None,
            rng=random.Random(0),
        )


@dataclass
class TableResult:
    """What one replayed game found."""

    table_id: str
    packets: int = 0
    problems: list[str] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=lambda: dict.fromkeys(COUNTER_NAMES, 0))
    skipped: str | None = None

    def __str__(self) -> str:
        if self.skipped:
            return f"table {self.table_id}: SKIPPED -- {self.skipped}"
        return (
            f"table {self.table_id}: {len(self.problems)} mismatches "
            f"over {self.packets} packets"
        )


# --------------------------------------------------------------------------
# reading packets out of the advisor's game log
# --------------------------------------------------------------------------


def _packet_key(packet: dict[str, Any]) -> tuple[str, str]:
    return str(packet.get("move_id")), str(packet.get("packet_id"))


def _log_files(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    """Every game-log file under ``paths``, which may be dirs, files, or both."""

    if isinstance(paths, (str, Path)):
        paths = [paths]
    files: list[Path] = []
    for entry in paths:
        entry = Path(entry)
        if entry.is_dir():
            files.extend(sorted(entry.glob("*.jsonl")))
        elif entry.is_file():
            files.append(entry)
    return files


def packets_by_table(
    paths: str | Path | Iterable[str | Path],
) -> dict[str, list[dict[str, Any]]]:
    """Collect BGA notification packets per table from advisor game logs.

    Batches overlap -- a reloaded page re-sends history the log already has --
    so packets are deduped on ``(move_id, packet_id)`` and returned in move
    order. Sorting is numeric on both, because BGA writes them as strings and
    ``"10" < "9"`` lexically would reorder a whole game.
    """

    files = _log_files(paths)
    seen: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for path in files:
        for row in iter_rows(path):
            if row.get("kind") != PACKET_KIND:
                continue
            table = str(row.get("table_id") or "unknown")
            bucket = seen.setdefault(table, {})
            for packet in (row.get("extra") or {}).get("packets") or []:
                if not isinstance(packet, dict):
                    continue
                key = _packet_key(packet)
                # On a collision keep the fuller copy. Packets arrive whole, so
                # this should never bite -- but the alternative, silently
                # preferring whichever copy was read first, would drop real
                # entries if a capture ever hands us a partial one.
                previous = bucket.get(key)
                if previous is None or len(packet.get("data") or []) > len(
                    previous.get("data") or []
                ):
                    bucket[key] = packet

    out: dict[str, list[dict[str, Any]]] = {}
    for table, bucket in seen.items():
        out[table] = [
            packet
            for _, packet in sorted(
                bucket.items(), key=lambda kv: (_as_int(kv[0][0]), _as_int(kv[0][1]))
            )
        ]
    return out


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def captured_start_players(
    paths: str | Path | Iterable[str | Path],
) -> dict[str, str]:
    """``startPlayerId`` per table, as recorded in the logged positions."""

    out: dict[str, str] = {}
    for path in _log_files(paths):
        for row in iter_rows(path):
            table = str(row.get("table_id") or "unknown")
            if table in out:
                continue
            start = ((row.get("state") or {}).get("bga") or {}).get("startPlayerId")
            if start:
                out[table] = str(start)
    return out


def start_player_from_packets(packets: Iterable[dict[str, Any]]) -> str | None:
    """The start player, read off the wonder draft.

    Round 1 of the draft goes start-player, opponent, opponent, start-player, so
    whoever picks the first wonder began Age I. Verified against all four games
    captured before this harness existed, whose start player was known
    independently from ``startPlayerId``; :func:`replay` re-checks it on every
    game where both are available.
    """

    for packet in packets:
        for entry in packet.get("data") or []:
            if entry.get("type") == "wonderSelected":
                pid = (entry.get("args") or {}).get("playerId")
                if pid:
                    return str(pid)
    return None


def missing_moves(packets: list[dict[str, Any]]) -> list[int]:
    """Move numbers absent from ``packets``, which make a replay untrustworthy."""

    moves = {_as_int(p.get("move_id"), -1) for p in packets}
    moves.discard(-1)
    if not moves:
        return []
    return [n for n in range(1, max(moves) + 1) if n not in moves]


# --------------------------------------------------------------------------
# the comparison itself
# --------------------------------------------------------------------------


def _apply_economy(
    world: World,
    seat: int,
    payment: dict[str, Any],
    counters: dict[str, int],
    move: str,
    problems: list[str],
) -> None:
    """Economy: the opponent gains what this player spends on trade.

    Trade steps are the ones with no producing item (``itemType`` null) and a
    non-coin resource; their ``cost`` is the coin amount paid to the bank.

    BGA publishes its own figure as ``economyProgressTokenCoins``, so the amount
    derived from the steps is cross-checked against it rather than trusted --
    that keeps this bookkeeping from silently absorbing a real disagreement,
    which is exactly what it did on the first pass.
    """

    steps = payment.get("steps") or []
    trade = sum(
        _as_int(step.get("cost"))
        for step in steps
        if step.get("itemType") is None and step.get("resource") != "coin"
    )
    holder_has_economy = "Economy" in world.cities[1 - seat].progress_tokens
    transferred = trade if holder_has_economy else 0
    published = _as_int(payment.get("economyProgressTokenCoins"))
    if transferred != published:
        problems.append(f"{move} economy transfer ours={transferred} bga={published}")
    if transferred:
        world.cities[1 - seat].coins += transferred
        counters["economy_transfer"] += 1


def _shields_for_card(world: World, seat: int, card) -> int:
    shields = card.shields
    if card.color is CardColor.RED and "Strategy" in world.cities[seat].progress_tokens:
        shields += 1
    return shields


def _check_military(
    world: World,
    seat: int,
    shields: int,
    payment: dict[str, Any],
    move: str,
    problems: list[str],
    counters: dict[str, int],
) -> None:
    """Run our military resolution and compare every figure BGA published."""

    want_new = payment.get("militaryNewPosition")
    tokens = payment.get("militaryTokens") or {}
    want_pays = (
        sum(_as_int(t.get("militaryOpponentPays")) for t in tokens.values())
        if isinstance(tokens, dict)
        else 0
    )
    want_steps = _as_int(payment.get("militarySteps"))

    state = world.state()
    before = world.cities[1 - seat].coins
    _apply_military(state, seat, shields)
    world.conflict = state.conflict_position
    world.tokens = state.military_tokens_remaining
    paid = before - world.cities[1 - seat].coins

    counters["military"] += 1
    if want_steps and shields != want_steps:
        problems.append(f"{move} shields ours={shields} bga={want_steps}")
    if want_new is not None and world.conflict != int(want_new):
        problems.append(
            f"{move} military position ours={world.conflict} bga={int(want_new)}"
        )
    if paid != want_pays:
        problems.append(f"{move} military coin loss ours={paid} bga={want_pays}")
    if paid or want_pays:
        counters["token_penalty"] += 1


def _check_situation(
    world: World,
    situation: dict[str, Any],
    seat_of: dict[str, int],
    move: str,
    problems: list[str],
    counters: dict[str, int],
    *,
    scores: bool,
) -> None:
    """Compare coins (always) and the live score categories (when published)."""

    for pid, row in situation.items():
        seat = seat_of.get(str(pid))
        if seat is None:
            continue
        if scores:
            breakdown = score_player(world.state(), seat)
            expected = {
                "blue": _as_int(row.get("player_score_blue")),
                # purple/coins/military are filled only during end-game scoring;
                # mid-game they read 0 for everyone, so they are excluded here
                # and covered by check_final_scores.
                "wonders": _as_int(row.get("player_score_wonders")),
                "progress": _as_int(row.get("player_score_progresstokens")),
                "total": _as_int(row.get("score")),
            }
            ours = {
                "blue": breakdown.blue_buildings,
                "wonders": breakdown.wonders,
                "progress": breakdown.progress,
                "total": breakdown.total
                - breakdown.treasury
                - breakdown.military
                - breakdown.guild,
            }
            counters["score_checks"] += 1
            for key, want in expected.items():
                if ours[key] != want:
                    problems.append(
                        f"{move} seat{seat} score.{key}: ours={ours[key]} bga={want}"
                    )
        bga_coins = _as_int(row.get("coins"))
        counters["coin_checks"] += 1
        if world.cities[seat].coins != bga_coins:
            problems.append(
                f"{move} seat{seat} coins: ours={world.cities[seat].coins} "
                f"bga={bga_coins}"
            )


def _check_final_scores(
    world: World,
    packets: list[dict[str, Any]],
    seat_of: dict[str, int],
    counters: dict[str, int],
    problems: list[str],
) -> None:
    """Compare our final ``score_player`` against BGA's completed scoring.

    ``nextPlayerTurnEndGameScoring.playersSituation`` carries the card and
    wonder categories; ``endGameCategoryUpdate`` then adds guild (purple),
    treasury (coins) and military. Summing the two is the only way to see BGA's
    real final numbers -- the mid-game snapshots never populate them. A game
    that ended by supremacy or concession never scores, so there is nothing to
    compare and that is not a failure.
    """

    pre: dict[str, Any] | None = None
    added: dict[str, dict[str, int]] = {}
    names: dict[str, str] = {}
    for packet in packets:
        for entry in packet.get("data") or []:
            args = entry.get("args") or {}
            pid, name = args.get("playerId"), args.get("player_name")
            if pid and name:
                names[str(pid)] = str(name)
            if entry.get("type") == "nextPlayerTurnEndGameScoring":
                pre = args.get("playersSituation")
            elif entry.get("type") == "endGameCategoryUpdate":
                name = args.get("player_name")
                category = args.get("category")
                points = args.get("points")
                if name is None or category is None or points is None:
                    continue
                bucket = added.setdefault(str(name), {})
                bucket[str(category)] = bucket.get(str(category), 0) + _as_int(points)
    if pre is None:
        return

    for pid, row in pre.items():
        seat = seat_of.get(str(pid))
        if seat is None:
            continue
        extra = added.get(names.get(str(pid), ""), {})
        expected = {
            "blue": _as_int(row.get("player_score_blue")),
            "guild": _as_int(extra.get("purple")),
            "wonders": _as_int(row.get("player_score_wonders")),
            "progress": _as_int(row.get("player_score_progresstokens")),
            "treasury": _as_int(extra.get("coins")),
            "military": _as_int(extra.get("military")),
            "buildings": (
                _as_int(row.get("player_score_blue"))
                + _as_int(row.get("player_score_green"))
                + _as_int(row.get("player_score_yellow"))
                + _as_int(extra.get("purple"))
            ),
            "total": _as_int(row.get("score")) + sum(extra.values()),
        }
        got = score_player(world.state(), seat)
        ours = {
            "blue": got.blue_buildings,
            "guild": got.guild,
            "wonders": got.wonders,
            "progress": got.progress,
            "treasury": got.treasury,
            "military": got.military,
            "buildings": got.buildings,
            "total": got.total,
        }
        counters["final_score_checks"] += 1
        for key, want in expected.items():
            if ours[key] != want:
                problems.append(f"FINAL seat{seat} {key}: ours={ours[key]} bga={want}")


def replay(
    table_id: str,
    packets: list[dict[str, Any]],
    *,
    start_player: str | None = None,
) -> TableResult:
    """Replay one game's packets, comparing every number BGA published."""

    result = TableResult(table_id=table_id, packets=len(packets))
    if not packets:
        result.skipped = "no packets captured"
        return result

    derived = start_player_from_packets(packets)
    if derived is None:
        result.skipped = "no wonder draft in the packets, so no seat framing"
        return result
    if start_player is not None and derived != start_player:
        # Either the draft-order derivation is wrong or the capture is
        # mismatched; both flip the military sign, so refuse rather than replay.
        result.skipped = (
            f"start player disagrees: draft says {derived}, "
            f"logged position says {start_player}"
        )
        return result
    gaps = missing_moves(packets)
    if gaps:
        shown = ", ".join(str(n) for n in gaps[:10])
        more = "" if len(gaps) <= 10 else f" (+{len(gaps) - 10} more)"
        result.skipped = f"incomplete capture: missing move(s) {shown}{more}"
        return result

    world = World()
    seat_of: dict[str, int] = {}
    problems = result.problems
    counters = result.counters

    for packet in packets:
        pending_situation = None
        pending_scored = None
        applied_action = False
        move = f"m{packet.get('move_id')}"
        for entry in packet.get("data") or []:
            kind = entry.get("type")
            args = entry.get("args") or {}
            pid = str(args.get("playerId") or "")
            if pid and pid not in seat_of:
                seat_of[pid] = 0 if pid == derived else 1
            seat = seat_of.get(pid)
            payment = args.get("payment") or {}

            if kind == "wonderSelected" and seat is not None:
                world.cities[seat].wonders.append(args["wonderName"])

            elif kind == "constructBuilding" and seat is not None:
                applied_action = True
                name = _card_name(args["buildingName"])
                card = CARDS_BY_NAME[name]
                if args.get("wonderName") != "The Mausoleum":  # revival is free
                    priced = minimum_payment(world.state(), seat, card.cost, card=card)
                    theirs = _as_int(args.get("cost"))
                    counters["priced"] += 1
                    if priced.total_coins != theirs:
                        problems.append(
                            f"{move} {name}: cost ours={priced.total_coins} bga={theirs}"
                        )
                    if REBATE_TOKENS & set(world.cities[seat].progress_tokens):
                        counters["with_rebate_token"] += 1
                    _apply_economy(world, seat, payment, counters, move, problems)
                    world.cities[seat].coins -= theirs
                world.cities[seat].buildings.append(name)

                if card.effects:
                    before = world.cities[seat].coins
                    _apply_card_coin_effects(world.state(), seat, card)
                    gained = world.cities[seat].coins - before
                    want = _as_int(payment.get("coinReward"))
                    if gained or want:
                        counters["coin_reward"] += 1
                        if gained != want:
                            problems.append(
                                f"{move} {name}: coinReward ours={gained} bga={want}"
                            )
                    loss = _as_int(payment.get("opponentCoinLoss"))
                    if loss:
                        counters["opponent_coin_loss"] += 1
                        world.cities[1 - seat].coins = max(
                            0, world.cities[1 - seat].coins - loss
                        )

                shields = _shields_for_card(world, seat, card)
                if shields:
                    _check_military(
                        world, seat, shields, payment, move, problems, counters
                    )

            elif kind == "constructWonder" and seat is not None:
                applied_action = True
                wonder_name = args["wonderName"]
                wonder = WONDERS_BY_NAME.get(wonder_name)
                if wonder is None:
                    problems.append(f"{move} unknown wonder {wonder_name!r}")
                    continue
                priced = minimum_payment(world.state(), seat, wonder.cost, is_wonder=True)
                theirs = _as_int(args.get("cost"))
                counters["priced"] += 1
                if priced.total_coins != theirs:
                    problems.append(
                        f"{move} WONDER {wonder_name}: ours={priced.total_coins} "
                        f"bga={theirs}"
                    )
                _apply_economy(world, seat, payment, counters, move, problems)
                world.cities[seat].coins -= theirs
                world.cities[seat].built_wonders.append(wonder_name)
                # Wonders pay out too (Appian Way +3/-3, Temple of Artemis +12,
                # Hanging Gardens +6). Applying them silently left the only
                # opponent_loses_coins source in the game unasserted, and the
                # counter reading 0 hid that a real instance had gone by.
                gained = sum(
                    e.amount for e in wonder.effects if e.kind.value == "immediate_coins"
                )
                loss = sum(
                    e.amount
                    for e in wonder.effects
                    if e.kind.value == "opponent_loses_coins"
                )
                want_gain = _as_int(payment.get("coinReward"))
                want_loss = _as_int(payment.get("opponentCoinLoss"))
                if gained or want_gain:
                    counters["coin_reward"] += 1
                    if gained != want_gain:
                        problems.append(
                            f"{move} WONDER {wonder_name}: coinReward "
                            f"ours={gained} bga={want_gain}"
                        )
                if loss or want_loss:
                    counters["opponent_coin_loss"] += 1
                    if loss != want_loss:
                        problems.append(
                            f"{move} WONDER {wonder_name}: opponentCoinLoss "
                            f"ours={loss} bga={want_loss}"
                        )
                world.cities[seat].coins += gained
                world.cities[1 - seat].coins = max(0, world.cities[1 - seat].coins - loss)
                if wonder.shields:
                    _check_military(
                        world, seat, wonder.shields, payment, move, problems, counters
                    )

            elif kind == "discardBuilding" and seat is not None:
                applied_action = True
                world.cities[seat].coins += _as_int(args.get("gain"))

            elif kind == "progressTokenChosen" and seat is not None:
                applied_action = True
                token = args["progressTokenName"]
                world.cities[seat].progress_tokens.append(token)
                # Several tokens pay out the moment they are taken (Urbanism 6,
                # Agriculture 6). Missing this made the coin trail drift from the
                # token onward, which is how Urbanism surfaced at m69.
                before = world.cities[seat].coins
                _apply_progress_immediate(world.state(), seat, token)
                if world.cities[seat].coins != before:
                    counters["progress_token_coins"] += 1

            elif kind == "opponentDiscardBuilding" and seat is not None:
                victim = 1 - seat
                name = args.get("buildingName")
                if name in world.cities[victim].buildings:
                    world.cities[victim].buildings.remove(name)

            situation = args.get("playersSituation")
            if situation is None and isinstance(args.get("args"), dict):
                situation = args["args"].get("playersSituation")
            if situation:
                # Every published figure in a packet is the value AFTER the move
                # resolves, but the entries carrying them can precede the move in
                # the array. So compare once per packet, once the packet is fully
                # applied: the last coin snapshot, and the last score-bearing one.
                pending_situation = situation
                if "playersSituation" in args:
                    pending_scored = situation

        if pending_scored is not None:
            _check_situation(
                world, pending_scored, seat_of, move, problems, counters, scores=True
            )
        elif pending_situation is not None and applied_action:
            _check_situation(
                world, pending_situation, seat_of, move, problems, counters, scores=False
            )

    _check_final_scores(world, packets, seat_of, counters, problems)
    return result


def run(
    log_dir: str | Path,
    tables: Iterable[str] = (),
) -> list[TableResult]:
    """Replay every captured game (or just ``tables``) found under ``log_dir``."""

    by_table = packets_by_table(log_dir)
    starts = captured_start_players(log_dir)
    wanted = [str(t) for t in tables] or sorted(by_table)
    results = []
    for table in wanted:
        packets = by_table.get(table, [])
        results.append(replay(table, packets, start_player=starts.get(table)))
    return results


def format_report(results: list[TableResult]) -> str:
    """One-command output: per table, then what was and was not exercised."""

    lines = [str(r) for r in results]
    totals = dict.fromkeys(COUNTER_NAMES, 0)
    for result in results:
        for name, count in result.counters.items():
            totals[name] += count

    lines.append("")
    lines.append("checks performed:")
    for name in COUNTER_NAMES:
        flag = "" if totals[name] else "   <- NOT EXERCISED"
        lines.append(f"   {name:<22}{totals[name]:>5}{flag}")

    problems = [f"[{r.table_id}] {p}" for r in results for p in r.problems]
    lines.append("")
    if problems:
        lines.append(f"!! {len(problems)} MISMATCHES")
        lines.extend(f"   {line}" for line in problems)
    elif any(r.counters["priced"] for r in results):
        lines.append("every published BGA quantity matched ours")
    else:
        lines.append("nothing was compared: no game packets captured yet")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tables", nargs="*", help="table ids (default: all captured)")
    parser.add_argument(
        "--log-dir",
        default=None,
        help="advisor game-log directory (default: runs/seven_wonders_duel/bga_game_log)",
    )
    parser.add_argument(
        "--packets-json",
        default=None,
        help=(
            "replay from a raw {table_id: [packet, ...]} JSON dump instead of the "
            "game log, for corpora captured outside the advisor"
        ),
    )
    args = parser.parse_args(argv)

    if args.packets_json:
        raw = json.loads(Path(args.packets_json).read_text(encoding="utf-8"))
        wanted = args.tables or sorted(raw)
        results = [replay(str(t), raw.get(str(t)) or []) for t in wanted]
    else:
        log_dir = args.log_dir or log_dir_for(GAME_ID)
        results = run(log_dir, args.tables)

    print(format_report(results))
    if any(r.problems or r.skipped for r in results):
        return 1
    # Exit 2, not 0: a run that compared nothing is not a clean bill of health,
    # and this is the state the harness is in until captures start arriving.
    if not any(r.counters["priced"] for r in results):
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
