"""Luck, decisions and model surprises in logged BGA Can't Stop games.

Two measures (agreed 2026-10-02):

1. **Dice luck with no model in it.** On every roll whose outcome is known,
   the chance of busting is pure dice arithmetic given the board and the
   runners (``bust_probability``). Busts minus expected busts, and squares
   lost to busts minus expected squares lost, say whether the dice were
   unkind without trusting the net.

2. **A win-probability ledger.** Every observed step moves each seat's win
   probability (the exact turn solve over the net's end-of-turn values), and
   each move is filed under one heading:

     luck      a roll: the value after the dice (best play assumed) minus the
               exact average over all 1296 dice before it. Under fair dice
               this averages zero whatever the net believes, so it cannot
               soak up a systematic losing pattern across games.
     decision  a choice: the value of what the mover did minus the value of
               their best choice (zero when they follow the solver).
     residual  a turn boundary: the exact one-turn look-ahead from the new
               board minus the net's value of that board -- the net
               disagreeing with itself one turn deeper. At the game's end the
               look-ahead meets the real result.
     gap       an unobserved stretch (a missed capture; in older logs, every
               opponent turn).

   The headings add up exactly: final - start = luck + decision + residual
   + gap, per seat.

**The first-roll trap.** The "before" of a turn's first roll must be the
exact average over every first roll (a solve rooted at turn start), never the
net's value of the board. The shortcut folds the net's errors into luck, and
the blind-spot estimate then comes out zero by construction.

**What the logs cannot show.** The advisor captures a position only while
dice are showing, so a roll that busts leaves no capture, and neither does
the pairing chosen just before it. The ledger assumes that pairing was the
best one *for rolling on* (``inferred`` on the entries). This moves value
between "decision" and "luck" on that one step; their sum is exact.

    python -m games.cantstop.luck --log-dir ../boardgame-ai/runs/cantstop/bga_game_log
"""

import argparse
import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .advisor_adapter import DEFAULT_CHECKPOINT, parse_state, wins_on_stop
from .engine import (COLUMNS, GameState, Phase, apply_move, bust, can_stop,
                     legal_moves, stop)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = ROOT / "runs/cantstop/bga_game_log"

# Every sorted 4-dice roll with its number of orderings (126 rolls, 1296 total).
DICE = sorted(Counter(tuple(sorted(d))
                      for d in itertools.product(range(1, 7), repeat=4)).items())
TOL = 1e-9


# ---- dice arithmetic (no model) ----

def bust_probability(board, runners):
    """Chance the next roll busts, for ``board``'s seat to move holding
    ``runners``. Exact over all 1296 rolls."""
    s = board.clone()
    s.runners = dict(runners)
    return sum(w for d, w in DICE if not legal_moves(s, d)) / 1296


def at_risk(board, runners):
    """Squares a bust would lose: runner positions above saved progress."""
    saved = board.progress[board.active_player]
    return sum(pos - saved[c] for c, pos in runners.items())


# ---- loading ----

@dataclass
class Capture:
    state: GameState      # AWAIT_MOVE: saved board, runners before the move, dice
    turn_id: str | None


@dataclass
class Game:
    table_id: str
    player_ids: list
    viewer_seat: int
    opponents_logged: bool
    captures: list
    names: dict = field(default_factory=dict)
    skipped: int = 0


def is_stale(prev, cap):
    """``cap`` shows the board after ``prev``'s move with ``prev``'s dice still
    up: read between the move and the next roll (seen in opponent turns on
    live tables). The next real roll follows ``prev`` directly."""
    a, b = prev.state, cap.state
    return (a.dice == b.dice and a.active_player == b.active_player
            and board_key(a) == board_key(b)
            and any(after_move(a, m).runners == b.runners
                    for m in legal_moves(a, a.dice)))


def load_game(path, viewer_id=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]
    raws = [r["extra"]["capture"] for r in rows if r.get("kind") == "decision"]
    raws = [r for r in raws if r.get("phase") == "diceChoice"]
    if not raws:
        raise ValueError(f"{path}: no dice captures")
    order = [str(p) for p in raws[0]["playerorder"]]
    actives = {str(r["active_player"]) for r in raws}
    viewer = (viewer_id or raws[0].get("viewer_player")
              or (actives.pop() if len(actives) == 1 else None))
    if viewer not in order:
        raise ValueError(f"{path}: cannot tell which seat is the viewer")
    caps, skipped, last = [], 0, None
    for raw in raws:
        try:
            state = parse_state(raw)
        except ValueError:
            skipped += 1
            continue
        key = (repr(state), raw.get("turn_id"))
        if key == last:          # the same position logged twice
            continue
        last = key
        cap = Capture(state, raw.get("turn_id"))
        if caps and is_stale(caps[-1], cap):
            skipped += 1
            continue
        caps.append(cap)
    names = {p: raws[-1]["players"][p].get("name", p) for p in order}
    return Game(str(raws[0]["table_id"]), order, order.index(viewer),
                any(str(r["active_player"]) != viewer for r in raws),
                caps, names, skipped)


# ---- reconstruction: which move, then roll or stop ----

@dataclass
class Step:
    runners: dict | None        # before the roll; None for a bust after an unseen move
    dice: tuple | None          # None: the roll busted
    move: tuple | None = None   # chosen after the roll
    then: str | None = None     # 'roll' | 'stop' | None (not observed)
    candidates: list | None = None  # moves it could have been, before a bust
    p_range: tuple | None = None    # bust odds over those moves (set by the ledger)


@dataclass
class Turn:
    seat: int
    start: GameState            # saved board at turn start, AWAIT_ROLL
    steps: list
    link: str                   # 'first' | 'adjacent' | 'gap' (to the previous turn)
    end: str = ""               # 'stop' | 'bust' | 'win' | 'gap' | 'log_end'


def turn_start(state):
    s = state.clone()
    s.runners, s.dice, s.phase = {}, None, Phase.AWAIT_ROLL
    return s


def after_move(state, move):
    c = state.clone()
    apply_move(c, move)
    return c


def board_key(s):
    return (s.active_player,
            tuple(tuple(sorted((c, p) for c, p in row.items() if p)) for row in s.progress),
            tuple(sorted((c, o) for c, o in s.claimed_by.items() if o is not None)))


def seat_key(s, seat, later=None):
    """``seat``'s saved progress and claims. With ``later``, ignore columns
    another seat has claimed by then: a claim wipes everyone's progress
    there, and in viewer-only logs it can happen in an unseen turn."""
    gone = set() if later is None else {
        c for c, o in later.claimed_by.items() if o is not None and o != seat}
    return (tuple(sorted((c, p) for c, p in s.progress[seat].items() if p and c not in gone)),
            tuple(sorted(s.claimed_columns(seat))))


def classify(a, b):
    """What happened between capture ``a`` and the next capture ``b``."""
    sa = a.state
    seat = sa.active_player
    children = {m: after_move(sa, m) for m in legal_moves(sa, sa.dice)}
    if b is None:
        wins = [m for m, c in children.items() if wins_on_stop(c)]
        return ("win", wins[0], None) if wins else ("log_end", None, None)
    sb = b.state
    same_turn = (a.turn_id == b.turn_id if a.turn_id and b.turn_id
                 else bool(sb.runners))
    if (same_turn and sb.active_player == seat and sb.runners
            and board_key(sa) == board_key(sb)):
        for m, c in children.items():
            if c.runners == sb.runners:
                return "roll", m, None
        return "in_turn_gap", None, None
    for m, c in children.items():
        if can_stop(c) and not wins_on_stop(c):
            end = c.clone()
            stop(end)
            if seat_key(end, seat, sb) == seat_key(sb, seat):
                return "stop", m, end
    if seat_key(sa, seat, sb) == seat_key(sb, seat):
        end = turn_start(sa)
        bust(end)
        return "bust", [m for m, c in children.items() if not wins_on_stop(c)], end
    return "gap", None, None


def reconstruct(game):
    turns, cur, link = [], None, "first"
    caps = game.captures
    for i, a in enumerate(caps):
        b = caps[i + 1] if i + 1 < len(caps) else None
        if cur is None:
            cur = Turn(a.state.active_player, turn_start(a.state), [], link)
        step = Step(dict(a.state.runners), a.state.dice)
        cur.steps.append(step)
        kind, info, end = classify(a, b)
        if kind == "roll":
            step.move, step.then = info, "roll"
            continue
        if kind == "in_turn_gap":
            continue
        if kind in ("win", "stop"):
            step.move, step.then = info, "stop"
        elif kind == "bust":
            step.then, step.candidates = "roll", info
            cur.steps.append(Step(None, None))
        cur.end = kind
        turns.append(cur)
        cur = None
        if end is not None and b is not None and game.opponents_logged:
            busts = first_roll_busts(end, a, b)
            for s in busts:
                turns.append(Turn(s.active_player, s, [Step({}, None)], "adjacent", "bust"))
            if busts:
                end = turn_start(busts[-1])
                bust(end)
        link = ("adjacent" if end is not None and b is not None
                and board_key(end) == board_key(b.state) else "gap")
    return turns


def _turn_counter(turn_id):
    prefix, _, n = (turn_id or "").rpartition(":")
    return (prefix, int(n)) if prefix and n.isdigit() else None


def first_roll_busts(end, a, b):
    """Turns between capture ``a``'s turn (ending on board ``end``) and
    capture ``b`` that left no capture and no mark: a bust on the very first
    roll, which never shows dice to capture. Accepted only when the board is
    unchanged, that bust was possible, and -- when the page's turn counter is
    readable -- exactly that many turns went by. Anything else stays a gap
    (a page reload, or captures missed in a background tab)."""
    out, s = [], end.clone()
    while (s.active_player != b.state.active_player
           and board_key(s)[1:] == board_key(b.state)[1:]
           and bust_probability(turn_start(s), {}) > 0):
        out.append(turn_start(s))
        bust(s)
    if not out or board_key(s) != board_key(b.state):
        return []
    ta, tb = _turn_counter(a.turn_id), _turn_counter(b.turn_id)
    if ta and tb and (ta[0] != tb[0] or tb[1] - ta[1] != len(out) + 1):
        return []
    return out


# ---- the ledger ----

def at(start, runners, phase, dice=None):
    s = start.clone()
    s.runners, s.phase, s.dice = dict(runners), phase, dice
    return s


@dataclass
class Ledger:
    game: Game
    start: np.ndarray
    final: np.ndarray
    outcome: np.ndarray | None  # one-hot when the log reaches the winning stop
    entries: list               # dicts: kind, actor, delta, plus roll details


def build_ledger(game, turns, evaluate, solver_cls=None):
    if solver_cls is None:
        from .rust_solver import RustTurnSolver as solver_cls
    entries, cursor, start = [], None, None

    def add(kind, delta, actor, **info):
        entries.append(dict(kind=kind, actor=actor, delta=np.asarray(delta, float), **info))

    for ti, turn in enumerate(turns):
        solver = solver_cls(turn.start, evaluate)
        roll_value = lambda k: solver.stop_roll(at(turn.start, k, Phase.AWAIT_DECISION))[1]
        stop_value = lambda k: solver.stop_roll(at(turn.start, k, Phase.AWAIT_DECISION))[0]
        v_turn = np.asarray(solver.value(turn.start))  # exact average over first rolls
        if cursor is None:
            start = v_turn
        else:
            add("residual" if turn.link == "adjacent" else "gap", v_turn - cursor,
                turn.seat, turn=ti)
        # Known: the cursor already equals the next roll's "before" value.
        cursor, known, K = v_turn, not turn.steps[0].runners, {}
        for si, step in enumerate(turn.steps):
            board = turn.start
            if step.runners is not None:
                K = step.runners
            pre = v_turn if not K else roll_value(K)
            if known:
                assert np.allclose(pre, cursor, atol=TOL), (turn.start, K)
            else:
                add("gap", pre - cursor, turn.seat, turn=ti)
            roll = dict(turn=ti, runners=dict(K), pre=np.asarray(pre).tolist(),
                        p_bust=bust_probability(board, K),
                        at_risk=at_risk(board, K), inferred=step.runners is None)
            if step.dice is None:
                post = solver.bust_value
                add("luck", post - pre, turn.seat, busted=True, **roll,
                    p_bust_range=step.p_range)
                cursor = post
                continue
            post = np.asarray(solver.value(at(board, K, Phase.AWAIT_MOVE, step.dice)))
            add("luck", post - pre, turn.seat, busted=False, dice=step.dice, **roll)
            cursor, known = post, False
            if step.move is not None:
                child = after_move(at(board, K, Phase.AWAIT_MOVE, step.dice), step.move).runners
                chosen = stop_value(child) if step.then == "stop" else roll_value(child)
                add("decision", chosen - post, turn.seat, turn=ti, then=step.then)
                cursor, known, K = chosen, step.then == "roll", child
            elif step.candidates:
                kids = [after_move(at(board, K, Phase.AWAIT_MOVE, step.dice), m).runners
                        for m in step.candidates]
                values = [roll_value(k) for k in kids]
                best = max(range(len(kids)), key=lambda j: values[j][turn.seat])
                add("decision", values[best] - post, turn.seat, turn=ti, then="roll",
                    inferred=True)
                ps = [bust_probability(board, k) for k in kids]
                cursor, known, K = values[best], True, kids[best]
                turn.steps[si + 1].p_range = (min(ps), max(ps))
    outcome = cursor if turns and turns[-1].end == "win" else None
    return Ledger(game, start, cursor, outcome, entries)


# ---- summaries ----

def seat_summary(ledger, seat):
    """One seat's view of a game: its win-probability ledger by heading, and
    its own rolls' model-free bust record."""
    out = Counter()
    for e in ledger.entries:
        d = float(e["delta"][seat])
        mine = e["actor"] == seat
        if e["kind"] == "luck":
            out["luck_own" if mine else "luck_others"] += d
        elif e["kind"] == "decision":
            out["decision_own" if mine else "decision_others"] += d
        else:
            out[e["kind"]] += d
        if e["kind"] == "luck" and mine:
            p, r = e["p_bust"], e["at_risk"]
            out["rolls"] += 1
            out["busts"] += e["busted"]
            out["busts_expected"] += p
            out["busts_var"] += p * (1 - p)
            out["squares_lost"] += r * e["busted"]
            out["squares_expected"] += p * r
            out["squares_var"] += p * (1 - p) * r * r
            rng = e.get("p_bust_range")
            if rng and rng[1] - rng[0] > 0.01:
                out["ambiguous_busts"] += 1
    out["start"] = float(ledger.start[seat])
    out["final"] = float(ledger.final[seat])
    total = sum(out[k] for k in ("luck_own", "luck_others", "decision_own",
                                 "decision_others", "residual", "gap"))
    assert abs(out["start"] + total - out["final"]) < 1e-6, "ledger does not add up"
    return out


def z(actual, expected, var):
    return (actual - expected) / math.sqrt(var) if var > 0 else 0.0


def mean_ci(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return float(xs.mean()) if len(xs) else float("nan"), float("nan")
    return float(xs.mean()), float(1.96 * xs.std(ddof=1) / math.sqrt(len(xs)))


def pct(x):
    return f"{100 * x:+.1f}"


def report(ledgers):
    lines = []
    lines.append("Per game, viewer's seat (win-probability points; luck own/others, "
                 "decisions own/others, residual, gap):")
    lines.append(f"{'table':>10} {'p':>2} {'opp':>3} {'start':>6} {'result':>6} "
                 f"{'luck':>13} {'decisions':>13} {'resid':>6} {'gap':>6}  "
                 f"busts (expected, z)")
    rows = []
    for L in ledgers:
        g = L.game
        s = seat_summary(L, g.viewer_seat)
        rows.append((L, s))
        result = "?" if L.outcome is None else ("WIN" if L.outcome[g.viewer_seat] > .5 else "loss")
        lines.append(
            f"{g.table_id:>10} {len(g.player_ids):>2} {'yes' if g.opponents_logged else 'no':>3} "
            f"{100 * s['start']:6.1f} {result:>6} "
            f"{pct(s['luck_own']):>6}/{pct(s['luck_others']):>6} "
            f"{pct(s['decision_own']):>6}/{pct(s['decision_others']):>6} "
            f"{pct(s['residual']):>6} {pct(s['gap']):>6}  "
            f"{s['busts']:.0f} ({s['busts_expected']:.1f}, "
            f"{z(s['busts'], s['busts_expected'], s['busts_var']):+.1f})")

    lines.append("")
    lines.append("Model-free dice luck, viewer's own rolls, all games:")
    tot = Counter()
    for _, s in rows:
        tot.update(s)
    lines.append(f"  rolls {tot['rolls']:.0f}; busts {tot['busts']:.0f} vs "
                 f"{tot['busts_expected']:.1f} expected (z {z(tot['busts'], tot['busts_expected'], tot['busts_var']):+.2f}); "
                 f"squares lost {tot['squares_lost']:.0f} vs {tot['squares_expected']:.1f} "
                 f"(z {z(tot['squares_lost'], tot['squares_expected'], tot['squares_var']):+.2f}); "
                 f"busts whose odds depended on an unseen pairing: {tot['ambiguous_busts']:.0f}")

    # Win-rate checks need every result. A viewer-only log shows a result
    # only when the viewer wins (an opponent's winning turn is not logged),
    # so those games would bias the actual win rate upward: left out.
    done = [(L, s) for L, s in rows if L.game.opponents_logged and L.outcome is not None]
    lines.append("")
    lines.append(f"Win rate vs the net's expectation: {len(done)} games with opponents "
                 f"logged and a result ({sum(1 for L, _ in rows if not L.game.opponents_logged)} "
                 f"viewer-only logs left out: their result shows only when the viewer won)")
    if done:
        v = [L.game.viewer_seat for L, _ in done]
        result = [float(L.outcome[i]) for (L, _), i in zip(done, v)]
        startp = [s["start"] for _, s in done]
        luck = [s["luck_own"] + s["luck_others"] for _, s in done]
        dec = [s["decision_own"] + s["decision_others"] for _, s in done]
        for name, xs in (
                ("net's prediction at start", startp),
                ("prediction given how everyone played", [p + d for p, d in zip(startp, dec)]),
                ("actual win rate", result),
                ("luck-adjusted win rate", [r - l for r, l in zip(result, luck)]),
                ("blind-spot estimate (turn-boundary residuals)", [s["residual"] for _, s in done]),
                ("unobserved stretches (gaps)", [s["gap"] for _, s in done])):
            m, ci = mean_ci(xs)
            lines.append(f"  {name:<48} {100 * m:6.1f}  +/- {100 * ci:.1f}")
        own = [s["decision_own"] for _, s in done]
        gift = [s["decision_others"] for _, s in done]
        lines.append(f"  decisions: viewer's own {100 * np.mean(own):+.1f} pt/game, "
                     f"opponents' {100 * np.mean(gift):+.1f} pt/game (positive = gifts to the viewer)")
        lines.append("  luck-adjusted = actual - dice luck = prediction given play + residuals "
                     "+ gaps, game by game")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--viewer", help="BGA player id, when a log does not say")
    ap.add_argument("--json", type=Path, help="write every ledger entry here")
    args = ap.parse_args(argv)

    from .model import NetEvaluator, load_net
    evaluate = NetEvaluator(load_net(args.checkpoint, device=args.device), device=args.device)
    ledgers = []
    for path in sorted(args.log_dir.glob("table_*.jsonl")):
        game = load_game(path, args.viewer)
        turns = reconstruct(game)
        ledgers.append(build_ledger(game, turns, evaluate))
    print(report(ledgers))
    if args.json:
        args.json.write_text(json.dumps([
            {"table_id": L.game.table_id, "players": L.game.player_ids,
             "viewer_seat": L.game.viewer_seat, "start": L.start.tolist(),
             "final": L.final.tolist(),
             "outcome": None if L.outcome is None else L.outcome.tolist(),
             "entries": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in e.items()} for e in L.entries]}
            for L in ledgers], default=list))


if __name__ == "__main__":
    main()
