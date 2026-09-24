"""M1 gate: the Rust engine is equivalent to the Python engine.

Three parts:

* a seeded lockstep corpus over all 10 rule sets and all four drivers, with
  coverage floors so a driver that stops reaching a rule corner fails loudly
  instead of quietly passing;
* constructed positions for the corners no driver is trusted to reach, each
  checked against all 126 dice and every transition, legal or not;
* mutation tests proving the harness can fail at all -- a gate that cannot go
  red is not a gate.

The full-size gate is ``python -m games.cantstop.rust_equiv``; this file is
the fast regression version. Skipped, not failed, when the extension is not
built (``cd games/cantstop/cantstop_rust && maturin develop --release``).

Run: python -m pytest games/cantstop/tests/test_rust_engine_equiv.py -q
"""

import pytest

from games.cantstop import engine, rust_equiv
from games.cantstop.engine import (
    ALL_RULESETS, COLUMN_HEIGHTS, GameState, Phase, RuleSet, can_stop,
    legal_moves, stop, stop_blocked,
)
from games.cantstop.rust_equiv import (
    DRIVERS, Coverage, Divergence, check_all_dice, check_state,
    play_lockstep, probe_transitions, run_gate,
)
from games.cantstop.snapshot import from_snapshot, snapshot

rust = pytest.importorskip("cantstop_rust", reason="run maturin develop first")

BASE2 = RuleSet.make(2)
BLOCK2 = RuleSet.make(2, blocking=True)
BLOCK4 = RuleSet.make(4, blocking=True)


def state_with(rules=BASE2, active=0, runners=None, progress=None,
               claimed=None, phase=Phase.AWAIT_DECISION, dice=None):
    s = GameState(rules)
    s.active_player = active
    for p, cols in (progress or {}).items():
        s.progress[p].update(cols)
    for col, p in (claimed or {}).items():
        s.claimed_by[col] = p
    s.runners = dict(runners or {})
    s.phase = phase
    s.dice = dice
    return s


def load_both(py):
    rs = rust.GameState.from_snapshot(snapshot(py))
    return py, rs


def check_everything(py, rs, where="constructed"):
    check_state(py, rs, where)
    check_all_dice(py, rs, where)
    probe_transitions(py, rs, where)


# ---- the corpus ----

@pytest.fixture(scope="module")
def corpus():
    """10 rule sets x 4 drivers x 5 seeds, every deep check on every step."""
    return run_gate(5, first_seed=1000, all_dice_every=1, probe_every=1)


def test_corpus_is_green_in_every_cell(corpus):
    assert len(corpus) == len(ALL_RULESETS) * len(DRIVERS)
    for cov in corpus.values():
        assert cov["games"] == cov["wins"] == 5


def test_corpus_reaches_every_rule_corner(corpus):
    """Floors, not exact counts: the point is that each corner is reached
    in bulk, not how often. Measured at authoring on these seeds, each total
    was several times its floor."""
    total = Coverage()
    for cov in corpus.values():
        total.update(cov)
    floors = {"busts": 200, "blocked_decisions": 500, "cap_full_rolls": 1000,
              "cap_split_rolls": 300, "capped_doubles": 10, "claims": 100,
              "claims_clearing": 80}
    for key, floor in floors.items():
        assert total[key] >= floor, (key, total[key])


def test_blocking_is_reached_in_every_blocking_rule_set(corpus):
    for (rules, driver), cov in corpus.items():
        if rules.blocking:
            assert cov["blocked_decisions"] > 0, (rules, driver)
        else:
            assert cov["blocked_decisions"] == 0, (rules, driver)


def test_the_drivers_really_differ(corpus):
    """Each driver exists for the corners it reaches; if two collapse into
    the same play the cycle is silently thinner than it claims."""
    def total(driver, key):
        return sum(c[key] for (r, d), c in corpus.items() if d == driver)
    assert total("pusher", "busts") > 3 * total("cautious", "busts")
    assert total("pusher", "cap_full_rolls") > \
        3 * total("cautious", "cap_full_rolls")
    steps_per_claim = {d: total(d, "steps") / total(d, "claims")
                       for d in DRIVERS}
    assert steps_per_claim["climber"] < steps_per_claim["pusher"]


# ---- constructed positions ----

CONSTRUCTED = {
    # Blocking: a runner on an opponent's saved marker forbids stopping...
    "blocked stop": state_with(
        BLOCK2, runners={7: 4}, progress={1: {7: 4}}),
    # ...but passing it is fine, and so is sitting on your own marker.
    "runner past marker": state_with(
        BLOCK2, runners={7: 5}, progress={1: {7: 4}, 0: {7: 3}}),
    # The same position without blocking is a normal stop.
    "blocking off": state_with(
        BASE2, runners={7: 4}, progress={1: {7: 4}}),
    # In 4p the blocker is a non-adjacent opponent, found by scanning all.
    "blocked by the third seat": state_with(
        BLOCK4, active=1, runners={5: 2, 9: 6}, progress={3: {9: 6}}),
    # Two runners on markers, one of them blocked.
    "one of two runners blocked": state_with(
        BLOCK4, runners={4: 1, 10: 3}, progress={2: {4: 1}, 1: {10: 2}}),
    # Full cap: only columns with runners can move.
    "runner cap full": state_with(runners={6: 2, 7: 3, 8: 1}),
    # Two runners: a pairing of two new columns can only play one of them.
    "one slot left": state_with(runners={6: 2, 8: 1}),
    # Doubles against the top: two spaces left, one space left, at the top.
    "double near the top": state_with(runners={2: 1, 12: 2}),
    "runner at the top": state_with(runners={12: 3, 7: 12}),
    # Claimed columns are closed to everyone.
    "claimed columns": state_with(
        claimed={7: 1, 6: 0, 2: 1}, runners={8: 3}),
    # Stopping claims the column and wipes every player's marker on it.
    "claim clears markers": state_with(
        BLOCK4, active=2, runners={3: 5}, progress={0: {3: 4}, 1: {3: 2}}),
    # Stopping to claim the winning column.
    "winning stop": state_with(
        claimed={2: 0, 12: 0}, runners={3: 5}, progress={1: {4: 6}}),
    # Mid-roll positions, so apply_move is probed with legal moves too.
    "await move": state_with(
        runners={7: 2}, phase=Phase.AWAIT_MOVE, dice=(1, 3, 4, 6)),
    "await move, capped double": state_with(
        runners={2: 2, 5: 1, 9: 1}, phase=Phase.AWAIT_MOVE, dice=(1, 1, 1, 1)),
    "turn start": state_with(
        BLOCK2, active=1, progress={0: {7: 5}, 1: {6: 3}},
        phase=Phase.AWAIT_ROLL),
    "game over": state_with(
        claimed={2: 1, 3: 1, 4: 1}, phase=Phase.GAME_OVER),
}
CONSTRUCTED["game over"].winner = 1


@pytest.mark.parametrize("name", list(CONSTRUCTED))
def test_constructed_position(name):
    py, rs = load_both(CONSTRUCTED[name].clone())
    check_everything(py, rs, name)


def test_constructed_positions_hit_what_they_claim():
    """Guard the fixtures themselves: a typo that un-blocks the blocked stop
    would leave that test green and useless."""
    c = CONSTRUCTED
    assert stop_blocked(c["blocked stop"])
    assert not stop_blocked(c["runner past marker"])
    assert not stop_blocked(c["blocking off"])
    assert stop_blocked(c["blocked by the third seat"])
    assert stop_blocked(c["one of two runners blocked"])
    assert not legal_moves(c["runner cap full"], (1, 1, 1, 1))  # 2s: no slot
    assert legal_moves(c["one slot left"], (1, 2, 3, 6)) == [
        (3,), (4, 8), (5,), (7,), (9,)]
    assert legal_moves(c["await move, capped double"], (1, 1, 1, 1)) == [(2,)]
    assert legal_moves(c["double near the top"], (6, 6, 6, 6)) == [(12,)]

    s = c["claim clears markers"].clone()
    stop(s)
    assert s.claimed_by[3] == 2
    assert all(p[3] == 0 for p in s.progress)
    s = c["winning stop"].clone()
    stop(s)
    assert s.winner == 0 and s.phase == Phase.GAME_OVER


def test_rust_refuses_like_python():
    """Same exception type, not just 'some error'. The probes compare this
    everywhere; here it is spelled out for the cases a caller will hit."""
    py, rs = load_both(CONSTRUCTED["blocked stop"].clone())
    with pytest.raises(ValueError, match="cannot stop"):
        rs.stop()
    with pytest.raises(ValueError, match="illegal in phase"):
        rs.apply_move([7])
    rs2 = rust.GameState(2, 3, False)
    with pytest.raises(ValueError, match="illegal in phase"):
        rs2.stop()
    with pytest.raises(ValueError, match="dice must be"):
        rs2.roll([0, 1, 2, 3])
    with pytest.raises(ValueError):
        rust.GameState(2, 4, False)        # 2p plays to 3 or 5


def test_snapshot_round_trips_on_both_sides():
    for s in CONSTRUCTED.values():
        snap = snapshot(s)
        assert snapshot(from_snapshot(snap)) == snap
        assert rust.GameState.from_snapshot(snap).snapshot() == snap


@pytest.mark.parametrize("bad", [
    lambda s: s[:1] + (5,) + s[2:],                       # active out of range
    lambda s: s[:4] + ([(7, 14)],) + s[5:],               # runner above top
    lambda s: s[:4] + ([(2, 1), (3, 1), (4, 1), (5, 1)],) + s[5:],
    lambda s: s[:5] + ([1, 2, 3, 4],) + s[6:],            # dice outside AWAIT_MOVE
    lambda s: s[:6] + (7,) + s[7:],                       # no such phase
])
def test_rust_rejects_malformed_snapshots(bad):
    with pytest.raises(ValueError):
        rust.GameState.from_snapshot(bad(snapshot(CONSTRUCTED["turn start"])))


def test_rust_state_clone_is_independent():
    rs = rust.GameState(2, 3, False)
    other = rs.clone()
    rs.roll([1, 2, 3, 4])
    assert other.phase == int(Phase.AWAIT_ROLL)
    assert rs.phase == int(Phase.AWAIT_MOVE)


# ---- the gate can fail ----
#
# Each mutation breaks the Python reference in a way a real porting error
# would, and the lockstep game must notice within a few games. Seeds and
# drivers are fixed so a pass is not luck.

def _gate_catches(rules=BASE2, driver="uniform", games=5):
    with pytest.raises(Divergence):
        for seed in range(games):
            play_lockstep(rules, seed, driver, all_dice_every=5,
                          probe_every=5)


def test_catches_a_wrong_column_height(monkeypatch):
    monkeypatch.setitem(engine.COLUMN_HEIGHTS, 7, 12)
    _gate_catches(driver="climber")


def test_catches_a_wrong_runner_cap(monkeypatch):
    monkeypatch.setattr(engine, "MAX_RUNNERS", 4)
    _gate_catches(driver="pusher")


def test_catches_blocking_ignored(monkeypatch):
    never = lambda state: False  # noqa: E731
    monkeypatch.setattr(engine, "stop_blocked", never)
    monkeypatch.setattr(rust_equiv, "stop_blocked", never)
    _gate_catches(rules=BLOCK2, driver="cautious")


def test_catches_a_claim_that_keeps_opponent_markers(monkeypatch):
    real_stop = engine.stop

    def keeps_markers(state):
        me = state.active_player
        saved = [dict(p) for p in state.progress]
        real_stop(state)
        for p, prog in enumerate(state.progress):
            if p != me:
                for col in prog:
                    if state.claimed_by[col] == me and saved[p][col]:
                        prog[col] = saved[p][col]
    monkeypatch.setattr(rust_equiv, "stop", keeps_markers)
    _gate_catches(driver="climber", games=20)


def test_catches_an_extra_dice_draw(monkeypatch):
    """A divergence in the NUMBER of draws must fail on the draw itself."""
    real = rust_equiv.random_dice

    def five_draws(rng):
        dice = real(rng)
        rng.next_u64()
        return dice
    monkeypatch.setattr(rust_equiv, "random_dice", five_draws)
    with pytest.raises(Divergence, match="rng state"):
        play_lockstep(BASE2, 0, "uniform")


def test_catches_a_move_ordering_difference(monkeypatch):
    real = engine.legal_moves
    monkeypatch.setattr(engine, "legal_moves",
                        lambda s, d: sorted(real(s, d), key=len))
    monkeypatch.setattr(rust_equiv, "legal_moves", engine.legal_moves)
    _gate_catches()
