"""Step 2 depth/speed read: pure-Python `ExpectiminimaxBot` vs the make/unmake
`RustExpectiminimax`, on identical states and configs. Reports nodes/s and wall
time per depth, and the speedup — the early signal for how much the mutable
engine buys before the recursion is hosted in Rust (Step 3).

Run with the RELEASE module built (`maturin develop --release`), else nodes/s is
the debug figure, not representative.
"""
from __future__ import annotations

import random
import time
from math import inf

import kingdomino_rust as kr

from games.kingdomino.game import GameState, Phase
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.expectiminimax import ExpectiminimaxBot, pick_aware_p0
from games.kingdomino.rust_expectiminimax import RustExpectiminimax, pick_aware

CHANCE_SAMPLES = 8
ENUM_CAP = 128


def _pick_state(target_deck=12, seed0=0):
    """A mid/late PLACE_AND_SELECT state with a moderate bag."""
    seed = seed0
    while seed < 400:
        st = GameState.new(seed=seed)
        rng = random.Random(seed * 7 + 3)
        for _ in range(400):
            if st.phase == Phase.GAME_OVER:
                break
            if (
                st.phase == Phase.PLACE_AND_SELECT
                and len(st.deck) <= target_deck
                and len(st.legal_actions()) >= 3
            ):
                return st
            st = st.step(rng.choice(st.legal_actions()))
        seed += 1
    raise RuntimeError("no suitable state found")


def _bench_once(make_bot, run, depths):
    rows = []
    for d in depths:
        bot = make_bot(d)
        t0 = time.perf_counter()
        val = run(bot, d)
        dt = time.perf_counter() - t0
        nodes = bot.nodes
        rows.append((d, nodes, dt, nodes / dt if dt else 0.0, val))
    return rows


def main():
    pystate = _pick_state()
    rs = _rust_state_from_python(pystate)
    eng_proto = kr.SearchEngine(rs)
    print(f"state: phase={pystate.phase.name} deck={len(pystate.deck)} "
          f"actions={len(pystate.legal_actions())}")

    def py_bot(d):
        return ExpectiminimaxBot(depth=d, chance_samples=CHANCE_SAMPLES,
                                 enum_cap=ENUM_CAP, eval_fn=pick_aware_p0)

    def r_bot(d):
        return RustExpectiminimax(depth=d, chance_samples=CHANCE_SAMPLES,
                                  enum_cap=ENUM_CAP, eval_fn=pick_aware)

    depths = [2, 3, 4]
    py_rows = _bench_once(py_bot, lambda b, d: b._value(pystate, d, -inf, inf), depths)
    r_rows = _bench_once(r_bot, lambda b, d: b.value(kr.SearchEngine(rs), d), depths)

    print(f"\n{'depth':>5} | {'python nodes':>12} {'py s':>8} {'py n/s':>10} "
          f"| {'rust nodes':>11} {'rust s':>8} {'rust n/s':>10} | {'speedup':>7}")
    print("-" * 92)
    for (d, pn, pt, pns, pv), (_, rn, rt, rns, rv) in zip(py_rows, r_rows):
        spd = (rns / pns) if pns else 0.0
        match = "ok" if abs(pv - rv) < 1e-9 else f"DIFF {rv-pv:.1e}"
        print(f"{d:>5} | {pn:>12,} {pt:>8.3f} {pns:>10,.0f} "
              f"| {rn:>11,} {rt:>8.3f} {rns:>10,.0f} | {spd:>6.1f}x  [{match}]")


if __name__ == "__main__":
    main()
