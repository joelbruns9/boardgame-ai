"""Replay a BGA Can't Stop game from its notification packets.

The advisor logs BGA's packet stream as ``bga_packets`` rows
(extension_cantstop/packet_recorder.js). Unlike board captures, which only
see decision screens, the stream is the server's ordered record of every
event, so a game is *read* rather than inferred (games/cantstop/luck.py):

    rollDice        args.player_id, args.dice, args.possibleMoves (BGA's own
                    legality per pairing); a roll whose pairings are all
                    illegal is a bust
    moveToken       one per column advanced: args.column_id and the runner's
                    new BGA height (distance from the top, 0 = top);
                    a double on one column is ONE token ("(3, 6), (4, 5)")
    gameStateChange args.id 10 (diceRoll) after a move = rolled again;
                    args.id 99 = game over
    saveProgress    the player stopped (args.column_list: columns claimed)

The first roll of a game can arrive inside the "history" packet as
``history_history`` with ``args.originalType == "rollDice"`` (no
possibleMoves); it is unwrapped here. Format read off table 925113041
(recon game, 2026-10-02).

Every event is replayed through our engine and checked: the roller is the
seat to move, BGA's legal pairings agree with ``legal_moves``, each token
lands where the engine puts the runner, a stop banks what BGA saved, and the
game ends exactly when the engine says it is won. Any disagreement raises
``PacketMismatch`` -- a game that does not replay is not used.
"""

from .engine import (COLUMN_HEIGHTS, GameState, Phase, apply_move, legal_moves,
                     roll, stop)


class PacketMismatch(ValueError):
    """The packet stream does not replay through the engine."""


DICE_ROLL, GAME_END = 10, 99


def packet_events(rows):
    """(move_id, type, args) in server order from a log's ``bga_packets``
    rows, history entries unwrapped, duplicate packets dropped."""
    seen, packets = set(), []
    for row in rows:
        if row.get("kind") != "bga_packets":
            continue
        for p in (row.get("extra") or {}).get("packets") or []:
            key = (str(p.get("move_id")), str(p.get("packet_id")))
            if key not in seen:
                seen.add(key)
                packets.append(p)
    packets.sort(key=lambda p: (_int(p.get("move_id")), _int(p.get("packet_id"))))
    out = []
    for p in packets:
        for e in p.get("data") or []:
            kind, args = e.get("type"), e.get("args") or {}
            if kind == "history_history" and args.get("originalType"):
                kind = args["originalType"]
            out.append((_int(p.get("move_id")), kind, args))
    return out


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return -1


def starts_at_game_start(events):
    """The stream holds the whole game: move ids run 1, 2, ... with no hole
    and the first roll comes before any token moves."""
    ids = sorted({m for m, _, _ in events})
    if not ids or ids[0] != 1 or ids != list(range(1, ids[-1] + 1)):
        return False
    kinds = [k for _, k, _ in events if k in ("rollDice", "moveToken", "saveProgress")]
    return bool(kinds) and kinds[0] == "rollDice"


def _bga_legal(args):
    """Whether BGA offered any pairing, or None when it did not say."""
    menu = args.get("possibleMoves")
    if menu is None:
        return None
    return any(m.get(k, {}).get("move") for m in menu for k in ("0", "1"))


def replay(events, rules, player_ids):
    """Rebuild the game. Returns (turns, final state); ``turns`` use luck.py's
    Turn/Step types with every move and stop/roll observed."""
    from .luck import Step, Turn, turn_start
    seat_of = {str(p): i for i, p in enumerate(player_ids)}
    state, turns, turn, step, tokens = None, [], None, None, []

    def fail(msg, move_id):
        raise PacketMismatch(f"move {move_id}: {msg}")

    def flush(move_id):
        """Apply the tokens since the last roll as one move."""
        nonlocal tokens
        if not tokens:
            return
        cols = []
        for col, height in tokens:
            pos = COLUMN_HEIGHTS[col] - height
            cols += [col] * (pos - state.position(col))
        move = tuple(sorted(cols))
        if move not in legal_moves(state, state.dice):
            fail(f"tokens {tokens} are not a legal move for {state.dice}", move_id)
        apply_move(state, move)
        for col, height in tokens:
            if state.runners.get(col) != COLUMN_HEIGHTS[col] - height:
                fail(f"runner on {col} disagrees with BGA height {height}", move_id)
        step.move = move
        tokens = []

    for move_id, kind, args in events:
        if kind == "moveToken":
            if state is None or state.phase != Phase.AWAIT_MOVE:
                fail("token move without a roll", move_id)
            tokens.append((int(args["column_id"]), int(args["height"])))
            continue
        if state is not None and tokens:
            flush(move_id)
        if kind == "rollDice":
            seat = seat_of.get(str(args.get("player_id")))
            if seat is None:
                fail(f"unknown player {args.get('player_id')}", move_id)
            if state is None:
                state = GameState(rules)
                state.active_player = seat
            if state.game_over or seat != state.active_player:
                fail(f"seat {seat} rolled, engine expects {state.active_player}", move_id)
            if state.phase == Phase.AWAIT_ROLL:
                turn = Turn(seat, turn_start(state), [], "adjacent" if turns else "first")
                turns.append(turn)
            elif state.phase != Phase.AWAIT_DECISION or (step and step.then != "roll"):
                fail("roll without a roll-again decision", move_id)
            dice = tuple(int(d) for d in args["dice"])
            step = Step(dict(state.runners), dice)
            turn.steps.append(step)
            moves = roll(state, dice)
            offered = _bga_legal(args)
            if offered is not None and offered != bool(moves):
                fail(f"BGA legal={offered}, engine moves={moves} for {dice}", move_id)
            if not moves:
                step.dice = None          # luck.py's bust step
                step.bust_dice = dice
                turn.end = "bust"
        elif kind == "gameStateChange":
            sid = _int(args.get("id"))
            if sid == DICE_ROLL and state is not None and state.phase == Phase.AWAIT_DECISION:
                step.then = "roll"
            elif sid == GAME_END:
                if state is None or not state.game_over:
                    fail("BGA ended the game; engine has no winner", move_id)
        elif kind == "saveProgress":
            if state is None or state.phase != Phase.AWAIT_DECISION:
                fail("stop outside a stop/roll decision", move_id)
            seat = state.active_player
            banked = dict(state.runners)
            stop(state)
            claimed = {int(c) for c in str(args.get("column_list") or "").split(",") if c.strip()}
            topped = {c for c, p in banked.items() if p >= COLUMN_HEIGHTS[c]}
            if claimed != topped:
                fail(f"BGA claimed {sorted(claimed)}, engine topped {sorted(topped)}", move_id)
            for col, height in (args.get("tokens") or {}).items():
                if COLUMN_HEIGHTS[int(col)] - int(height) != banked.get(int(col)):
                    fail(f"saved {col} at height {height}, engine banked {banked}", move_id)
            step.then = "stop"
            turn.end = "win" if state.game_over else "stop"
            if state.game_over and state.winner != seat:
                fail("winner is not the seat that stopped", move_id)
    if state is not None and tokens:
        flush(None)
    if turn is not None and not turn.end:
        turn.end = "log_end"
    return turns, state
