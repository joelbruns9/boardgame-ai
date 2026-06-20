"""Micro-benchmark: legal_placements + is_legal_placement, original vs optimized,
on the SAME machine so the ratio is meaningful (absolute numbers are sandbox)."""
import random, time
from games.kingdomino.board import Board as OptBoard, Placement as PlOpt
from games.kingdomino.board_original import Board as OrigBoard, Placement as PlOrig
from games.kingdomino.dominoes import DOMINOES

DOMS = list(DOMINOES.values())

def build_states(BoardCls, PlCls, n_states=400, seed0=0):
    """Random mid-game boards (varied fill levels)."""
    rng = random.Random(seed0)
    states = []
    for s in range(n_states):
        b = BoardCls()
        target = rng.randrange(4, 40)
        for _ in range(target):
            d = DOMS[rng.randrange(len(DOMS))]
            lp = b.legal_placements(d)
            if not lp: break
            p = lp[rng.randrange(len(lp))]
            b.place(d, PlCls(p.x1, p.y1, p.x2, p.y2, p.flipped))
        states.append(b)
    return states

def bench(BoardCls, PlCls, label):
    states = build_states(BoardCls, PlCls, n_states=400, seed0=123)
    test_doms = [DOMINOES[i] for i in (1, 13, 23, 36, 48)]
    t0 = time.time()
    total = 0
    for _ in range(3):                      # repeat for stability
        for b in states:
            for d in test_doms:
                total += len(b.legal_placements(d))
    dt = time.time() - t0
    calls = 3 * len(states) * len(test_doms)
    print(f"  {label:<10} {calls} legal_placements calls in {dt:.3f}s  "
          f"= {calls/dt:,.0f} calls/s  (moves summed={total})")
    return dt

print("legal_placements micro-benchmark (same boards, same machine):")
d_orig = bench(OrigBoard, PlOrig, "original")
d_opt  = bench(OptBoard, PlOpt, "optimized")
print(f"\n  speedup on legal_placements: {d_orig/d_opt:.2f}x")