"""Step 3 depth/speed read for the Rust-HOSTED `kr.RustSearch` (recursion in Rust).

Two measurements:

1. `value()` vs the Python-hosted `RustExpectiminimax` on one position — the raw
   nodes/s speedup from hosting the recursion in Rust (single root subtree).
2. `choose_action()` — the ACTUAL bot path — timed across several representative
   positions. This costs more than `value()`: choose_action searches every root
   child with a fresh full window (no root-sibling pruning), so it is the honest
   per-move budget for timed play. Root-window reuse / move ordering / iterative
   deepening (future work) would narrow the gap; rayon over root children would
   cut wall-clock further.

Single-threaded. Run with the RELEASE module (`maturin develop --release`).
"""
from __future__ import annotations

import random
import time

import kingdomino_rust as kr

from games.kingdomino.game import GameState, Phase
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.rust_expectiminimax import RustExpectiminimax, pick_aware

CHANCE_SAMPLES = 8
ENUM_CAP = 128


def _pick_state(target_deck=12, seed0=0):
    """A mid/late PLACE_AND_SELECT state with a moderate bag (chance nodes present)."""
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


def _representative_states(n=5):
    """A spread of PLACE_AND_SELECT positions (varying deck size / branching) so the
    choose_action timing reflects a distribution, not one lucky position."""
    out = []
    for i, target in enumerate((16, 14, 12, 10, 8)[:n]):
        st = _pick_state(target_deck=target, seed0=i * 37)
        out.append(st)
    return out


def _value_speedup(pystate):
    rs = _rust_state_from_python(pystate)
    print(f"\n[1] value() speedup on: phase={pystate.phase.name} "
          f"deck={len(pystate.deck)} actions={len(pystate.legal_actions())}  "
          f"(chance_samples={CHANCE_SAMPLES}, enum_cap={ENUM_CAP})")
    ref_depths = [2, 3, 4]
    rust_depths = [2, 3, 4, 5, 6]
    print(f"\n{'depth':>5} | {'py-hosted n/s':>13} {'py s':>8} "
          f"| {'rust-hosted n/s':>15} {'rust s':>8} {'rust nodes':>13} "
          f"| {'speedup':>7} {'match':>10}")
    print("-" * 100)
    for d in rust_depths:
        search = kr.RustSearch(depth=d, chance_samples=CHANCE_SAMPLES,
                               enum_cap=ENUM_CAP, eval="pick_aware")
        t0 = time.perf_counter()
        rv = search.value(rs, d)
        rt = time.perf_counter() - t0
        rns = search.nodes / rt if rt else 0.0
        pns_s = pt_s = spd_s = match_s = ""
        if d in ref_depths:
            ref = RustExpectiminimax(depth=d, chance_samples=CHANCE_SAMPLES,
                                     enum_cap=ENUM_CAP, eval_fn=pick_aware)
            t0 = time.perf_counter()
            pv = ref.value(kr.SearchEngine(rs), d)
            pt = time.perf_counter() - t0
            pns = ref.nodes / pt if pt else 0.0
            pns_s, pt_s = f"{pns:,.0f}", f"{pt:.3f}"
            spd_s = f"{rns / pns:.1f}x" if pns else ""
            match_s = "ok" if abs(pv - rv) < 1e-9 else f"~{rv - pv:+.2e}"
        print(f"{d:>5} | {pns_s:>13} {pt_s:>8} "
              f"| {rns:>15,.0f} {rt:>8.3f} {search.nodes:>13,} "
              f"| {spd_s:>7} {match_s:>10}")


def _choose_action_budget(states, depths=(4, 5, 6)):
    """Time the REAL bot path (choose_action) across positions → the per-move budget."""
    print(f"\n[2] choose_action() per-move wall time across {len(states)} positions "
          f"(the operational bot path):")
    print(f"\n{'depth':>5} | {'mean s':>9} {'max s':>9} {'mean nodes':>13} "
          f"{'max nodes':>13}")
    print("-" * 60)
    rss = [_rust_state_from_python(st) for st in states]
    for d in depths:
        times, nodes = [], []
        for rs in rss:
            search = kr.RustSearch(depth=d, chance_samples=CHANCE_SAMPLES,
                                   enum_cap=ENUM_CAP, eval="pick_aware")
            t0 = time.perf_counter()
            search.choose_action(rs)
            times.append(time.perf_counter() - t0)
            nodes.append(search.nodes)
        print(f"{d:>5} | {sum(times) / len(times):>9.3f} {max(times):>9.3f} "
              f"{sum(nodes) // len(nodes):>13,} {max(nodes):>13,}")
    print("\nNote: choose_action > value() (fresh full window per root child). "
          "depth 5 is the defensible timed-play baseline today; treat depth 6 as "
          "offline/experimental until root-window reuse + a deadline land.")


def main():
    states = _representative_states()
    _value_speedup(states[2])       # the deck-12 mid position, for the speedup table
    _choose_action_budget(states)   # the real per-move budget across the spread


if __name__ == "__main__":
    main()
