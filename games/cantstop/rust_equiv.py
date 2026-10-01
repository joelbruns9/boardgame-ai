"""M1 gate: the Rust engine plays the same game as the Python engine.

Both engines are driven in lockstep from here. Python is the reference. After
every transition the two snapshots (``snapshot.py``) must be equal, and so
must everything the engines report on the way: the dice each side's portable
RNG drew (and the RNG state afterwards, so a divergence in the *number* of
draws fails on the draw it happens), the legal moves, ``can_stop`` and
``stop_blocked``.

**A uniform-random driver is not a gate** (the Welcome To port measured it:
uniform play never reached two of the three end conditions). Four drivers
are cycled, each chosen for the rule corners it reaches:

    uniform   random move, stop half the time
    pusher    rarely stops -> long turns, busts, a full runner cap, runners
              parked on opponents' markers (blocking)
    cautious  stops whenever allowed -> short turns, many blocked stops
    climber   prefers moves that top out a column and stops to claim it ->
              claims that clear opponents' markers, and wins

and the counters in ``Coverage`` report whether each corner was actually
reached. Rule corners no driver reaches reliably are covered by constructed
positions in ``tests/test_rust_engine_equiv.py``.

Two optional deeper checks, sampled because they multiply the cost:

    all_dice_every  compare legal moves for all 126 dice multisets
    probe_every     try every transition on clones -- including the illegal
                    ones -- and compare "raised ValueError?" and the result

Run the full gate (thousands of games, all 10 rule sets, all drivers):

    python -m games.cantstop.rust_equiv --games-per-cell 100
"""

import argparse
import itertools
import time
from collections import Counter

from games.cantstop import engine
from games.cantstop.engine import (
    ALL_RULESETS, COLUMN_HEIGHTS, MAX_RUNNERS, GameState, Phase, apply_move,
    bust, can_stop, dice_pairings, legal_moves, random_dice, roll,
    stop, stop_blocked,
)
from games.cantstop.portable_rng import PortableRng
from games.cantstop.snapshot import snapshot

DRIVERS = ("uniform", "pusher", "cautious", "climber")

# Every multiset of four dice, sorted: the whole domain of legal_moves.
ALL_DICE = tuple(itertools.combinations_with_replacement(range(1, 7), 4))

# Driver decisions come from their own stream so they never perturb the dice
# stream the two engines share.
_DRIVER_SALT = 0x9E3779B97F4A7C15

MAX_TURNS = 20_000


class Divergence(AssertionError):
    """The engines disagree. The message names the step and both values."""


def _rust():
    import cantstop_rust
    return cantstop_rust


def new_rust_state(rules):
    return _rust().GameState(rules.num_players, rules.columns_to_win,
                             rules.blocking)


def _moves(rust_moves):
    return [tuple(m) for m in rust_moves]


def _same(what, py_value, rs_value, where):
    if py_value != rs_value:
        raise Divergence(f"{what} differs at {where}: "
                         f"python={py_value!r} rust={rs_value!r}")


def rust_step(fn, where):
    """Run one Rust transition the Python side just made. A refusal here is
    a divergence -- Python accepted the same step -- so report it as one."""
    try:
        return fn()
    except ValueError as exc:
        raise Divergence(f"rust refused a step python took at {where}: "
                         f"{exc}") from exc


def check_state(py, rs, where):
    _same("snapshot", snapshot(py), rs.snapshot(), where)
    if py.phase in (Phase.AWAIT_DECISION, Phase.AWAIT_ROLL):
        _same("stop_blocked", stop_blocked(py), rs.stop_blocked(), where)
        _same("can_stop", can_stop(py), rs.can_stop(), where)


def check_all_dice(py, rs, where):
    for dice in ALL_DICE:
        _same(f"legal_moves{dice}", legal_moves(py, dice),
              _moves(rs.legal_moves(list(dice))), where)


# ---- transition probes (legal and illegal alike) ----

def _attempt(fn):
    try:
        return ("ok", fn())
    except ValueError:
        return ("ValueError", None)


def _probe_ops(py):
    """Every transition worth trying from here, as (name, python-fn,
    rust-fn) builders over clones. Includes the illegal ones: refusal must
    agree too, and a refusal must leave the state untouched."""
    ops = [
        ("stop", stop, lambda s: s.stop()),
        ("bust", bust, lambda s: s.bust()),
        ("roll(1,1,6,6)", lambda s: roll(s, (1, 1, 6, 6)),
         lambda s: _moves(s.roll([1, 1, 6, 6]))),
        ("roll(3,4,3,4)", lambda s: roll(s, (3, 4, 3, 4)),
         lambda s: _moves(s.roll([3, 4, 3, 4]))),
    ]
    candidates = [(2,), (7, 7), (2, 12)]
    if py.dice is not None:
        candidates += legal_moves(py, py.dice)
    for mv in candidates:
        ops.append((f"apply_move{mv}",
                    lambda s, mv=mv: apply_move(s, mv),
                    lambda s, mv=mv: s.apply_move(list(mv))))
    return ops


def probe_transitions(py, rs, where):
    for name, py_fn, rs_fn in _probe_ops(py):
        p, r = py.clone(), rs.clone()
        py_out = _attempt(lambda: py_fn(p))
        rs_out = _attempt(lambda: rs_fn(r))
        _same(f"probe {name}", py_out, rs_out, where)
        _same(f"state after probe {name}", snapshot(p), r.snapshot(), where)


# ---- drivers ----

def _choose_move(driver, state, moves, rng):
    if driver == "climber":
        def score(mv):
            after = dict(state.runners)
            for col in mv:
                after[col] = after.get(col, state.position(col)) + 1
            tops = sum(after[c] >= COLUMN_HEIGHTS[c] for c in set(mv))
            return (tops, len(mv), -min(COLUMN_HEIGHTS[c] - after[c]
                                        for c in mv))
        best = max(score(m) for m in moves)
        moves = [m for m in moves if score(m) == best]
    return moves[rng.randrange(len(moves))]


def _wants_stop(driver, state, rng):
    if driver == "cautious":
        return True
    if driver == "pusher":
        return rng.next_float() < 0.1
    if driver == "climber":
        if any(pos >= COLUMN_HEIGHTS[c] for c, pos in state.runners.items()):
            return True
        return rng.next_float() < 0.25
    return rng.next_float() < 0.5


# ---- coverage ----

class Coverage(Counter):
    """Counts of the rule corners actually reached. Keys:

    games, steps, wins, busts
    decisions             AWAIT_DECISION states seen
    blocked_decisions     ... where stop_blocked was true
    cap_full_rolls        rolls taken with all three runners out
    cap_split_rolls       a pairing whose two sums were each usable but the
                          cap forbade opening both (either may be played)
    capped_doubles        a double offered as one step: one space left
    claims                columns claimed
    claims_clearing       ... that wiped another player's saved marker
    """


def _note_roll(cov, state, dice, moves):
    runners = state.runners
    free = MAX_RUNNERS - len(runners)
    if free == 0:
        cov["cap_full_rolls"] += 1
    if not moves:
        cov["busts"] += 1
        return
    for a, b in dice_pairings(dice):
        if a == b:
            if ((a,) in moves
                    and COLUMN_HEIGHTS[a] - state.position(a) == 1):
                cov["capped_doubles"] += 1
            continue
        ok_a = engine._column_open(state, a) and (a in runners or free > 0)
        ok_b = engine._column_open(state, b) and (b in runners or free > 0)
        if ok_a and ok_b and (a not in runners) + (b not in runners) > free:
            cov["cap_split_rolls"] += 1


def _note_stop(cov, state):
    me = state.active_player
    for col, pos in state.runners.items():
        if pos >= COLUMN_HEIGHTS[col]:
            cov["claims"] += 1
            if any(prog[col] for p, prog in enumerate(state.progress)
                   if p != me):
                cov["claims_clearing"] += 1


# ---- the lockstep game ----

def play_lockstep(rules, seed, driver, *, all_dice_every=0, probe_every=0,
                  coverage=None):
    """Play one seeded game on both engines, comparing after every step.
    Raises ``Divergence`` on the first disagreement. Returns the coverage
    counter (a fresh one unless passed in)."""
    rust = _rust()
    cov = Coverage() if coverage is None else coverage
    py = GameState(rules)
    rs = new_rust_state(rules)
    dice_py, dice_rs = PortableRng(seed), rust.Rng(seed)
    choose = PortableRng(seed ^ _DRIVER_SALT)
    step = 0
    turns = 0

    def where():
        return (f"{rules} seed={seed} driver={driver} step={step} "
                f"turn={turns}\n{py!r}")

    def deep_checks():
        if all_dice_every and step % all_dice_every == 0:
            check_all_dice(py, rs, where())
        if probe_every and step % probe_every == 0:
            probe_transitions(py, rs, where())

    check_state(py, rs, where())
    while not py.game_over:
        deep_checks()
        if py.phase == Phase.AWAIT_DECISION:
            cov["decisions"] += 1
            if stop_blocked(py):
                cov["blocked_decisions"] += 1
            if can_stop(py) and _wants_stop(driver, py, choose):
                _note_stop(cov, py)
                stop(py)
                rust_step(rs.stop, where())
                step += 1
                turns += 1
                check_state(py, rs, where())
                continue

        dice = random_dice(dice_py)
        _same("dice", tuple(dice), tuple(dice_rs.roll_dice()), where())
        _same("rng state", dice_py.state, dice_rs.state, where())
        _note_roll(cov, py, tuple(sorted(dice)), legal_moves(py, dice))
        py_moves = roll(py, dice)
        rs_moves = _moves(rust_step(lambda: rs.roll(list(dice)), where()))
        step += 1
        _same("roll moves", py_moves, rs_moves, where())
        check_state(py, rs, where())
        if not py_moves:
            turns += 1
            if turns > MAX_TURNS:
                raise RuntimeError(f"game did not finish: {where()}")
            continue

        deep_checks()
        move = _choose_move(driver, py, py_moves, choose)
        apply_move(py, move)
        rust_step(lambda: rs.apply_move(list(move)), where())
        step += 1
        check_state(py, rs, where())

    cov["games"] += 1
    cov["wins"] += 1
    cov["steps"] += step
    return cov


def run_gate(games_per_cell, *, first_seed=0, all_dice_every=0,
             probe_every=0, rulesets=ALL_RULESETS, drivers=DRIVERS,
             progress=None):
    """Every rule set x every driver x ``games_per_cell`` seeds. Returns a
    ``{(rules, driver): Coverage}`` table; raises on the first divergence."""
    table = {}
    for rules in rulesets:
        for driver in drivers:
            cov = Coverage()
            for i in range(games_per_cell):
                play_lockstep(rules, first_seed + i, driver,
                              all_dice_every=all_dice_every,
                              probe_every=probe_every, coverage=cov)
            table[(rules, driver)] = cov
            if progress:
                progress(rules, driver, cov)
    return table


def _label(rules):
    return (f"{rules.num_players}p to {rules.columns_to_win}"
            f"{' blk' if rules.blocking else '    '}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games-per-cell", type=int, default=100,
                    help="games per (rule set, driver); 40 cells")
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--all-dice-every", type=int, default=25)
    ap.add_argument("--probe-every", type=int, default=5)
    args = ap.parse_args(argv)

    started = time.perf_counter()
    total = Coverage()

    def report(rules, driver, cov):
        total.update(cov)
        print(f"{_label(rules)} {driver:<8} games={cov['games']:>4} "
              f"steps={cov['steps']:>8} busts={cov['busts']:>6} "
              f"blocked={cov['blocked_decisions']:>5} "
              f"capsplit={cov['cap_split_rolls']:>5} "
              f"claims={cov['claims']:>5} clearing={cov['claims_clearing']:>4}",
              flush=True)

    run_gate(args.games_per_cell, first_seed=args.first_seed,
             all_dice_every=args.all_dice_every,
             probe_every=args.probe_every, progress=report)
    elapsed = time.perf_counter() - started
    print(f"\nGREEN: {total['games']} games, {total['steps']} steps, "
          f"0 divergences, {elapsed:.0f} s")
    print("coverage:", dict(sorted(total.items())))


if __name__ == "__main__":
    main()
