"""Step 3 depth/speed read: the Rust-HOSTED `kr.RustSearch` (recursion in Rust)
vs the Python-hosted `RustExpectiminimax` (make/unmake per node, FFI hop per node),
on identical states/configs. Reports nodes/s and wall time per depth and the
Rust-hosted speedup — the measurement of what hosting the recursion in Rust buys,
and whether depth 6 (the vs-AZ target) is now in reach.

Run with the RELEASE module built (`maturin develop --release`).
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


def main():
    pystate = _pick_state()
    rs = _rust_state_from_python(pystate)
    print(f"state: phase={pystate.phase.name} deck={len(pystate.deck)} "
          f"actions={len(pystate.legal_actions())}  "
          f"(chance_samples={CHANCE_SAMPLES}, enum_cap={ENUM_CAP})")

    # Python-hosted reference: only up to depth 4 (depth 5+ is impractical here).
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

        pns_s, pt_s, spd_s, match_s = "", "", "", ""
        if d in ref_depths:
            ref = RustExpectiminimax(depth=d, chance_samples=CHANCE_SAMPLES,
                                     enum_cap=ENUM_CAP, eval_fn=pick_aware)
            t0 = time.perf_counter()
            pv = ref.value(kr.SearchEngine(rs), d)
            pt = time.perf_counter() - t0
            pns = ref.nodes / pt if pt else 0.0
            pns_s = f"{pns:,.0f}"
            pt_s = f"{pt:.3f}"
            spd_s = f"{rns / pns:.1f}x" if pns else ""
            # Enumerated in-horizon → exact; sampled deals may diverge from the
            # Python RNG, so only assert equality when nothing was sampled.
            match_s = "ok" if abs(pv - rv) < 1e-9 else f"~{rv - pv:+.2e}"

        print(f"{d:>5} | {pns_s:>13} {pt_s:>8} "
              f"| {rns:>15,.0f} {rt:>8.3f} {search.nodes:>13,} "
              f"| {spd_s:>7} {match_s:>10}")

    print("\n(match column blank = no Python-hosted reference run at that depth; "
          "'~<delta>' = sampled chance, RNGs differ by design)")


if __name__ == "__main__":
    main()
