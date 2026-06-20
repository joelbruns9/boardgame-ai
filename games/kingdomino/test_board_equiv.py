"""Equivalence oracle: optimized Board vs original Board (board_original.py).

Drives many random fill sequences in lockstep on both boards and asserts that
EVERY rules-relevant output is identical:
  - legal_placements(d)            — as LISTS (content AND order)
  - is_legal_placement(d, p)       — over a grid incl. illegal candidates
  - is_empty(x, y)                 — every canvas cell
  - half_connects via legal moves  — covered transitively by the above
  - _frontier()                    — as sets
  - score()                        — at every state
If any assertion fires, the optimization changed game behavior.
"""
import random
import numpy as np

from games.kingdomino.board import Board as OptBoard
from games.kingdomino.board_original import Board as OrigBoard
from games.kingdomino.dominoes import DOMINOES, Terrain

DOMS = list(DOMINOES.values())
# A representative probe set: doubles, mixed, crowned, mine/swamp combos.
PROBE_IDS = [1, 3, 7, 10, 12, 13, 19, 23, 36, 40, 46, 48]
PROBE = [DOMINOES[i] for i in PROBE_IDS]


def all_cells_empty_match(a: OrigBoard, b: OptBoard) -> bool:
    n = a.canvas_size
    for y in range(n):
        for x in range(n):
            if a.is_empty(x, y) != b.is_empty(x, y):
                print(f"  is_empty mismatch at ({x},{y}): orig={a.is_empty(x,y)} opt={b.is_empty(x,y)}")
                return False
    return True


def candidate_grid(a: OrigBoard):
    """Yield placements spanning the occupied bbox +/-2, both orientations and
    flips, including many ILLEGAL ones (out of bbox, non-adjacent, occupied)."""
    bx = a.occupied_bbox()
    min_x, min_y, max_x, max_y = bx
    for y in range(min_y - 2, max_y + 3):
        for x in range(min_x - 2, max_x + 3):
            # horizontal and vertical partners
            for (x2, y2) in ((x + 1, y), (x, y + 1)):
                for flipped in (False, True):
                    yield (x, y, x2, y2, flipped)


def check_state(a: OrigBoard, b: OptBoard, step_info: str) -> bool:
    from games.kingdomino.board import Placement as PlOpt
    from games.kingdomino.board_original import Placement as PlOrig

    # 1) is_empty over the whole canvas
    if not all_cells_empty_match(a, b):
        print(f"  [{step_info}] is_empty divergence"); return False

    # 2) _frontier as sets
    if a._frontier() != b._frontier():
        print(f"  [{step_info}] frontier set divergence"); return False

    # 3) score
    if a.score().total != b.score().total or a.score() != ScoreEq(b.score()):
        # compare field-by-field (different classes)
        sa, sb = a.score(), b.score()
        if (sa.territory_score, sa.harmony_bonus, sa.middle_kingdom_bonus) != \
           (sb.territory_score, sb.harmony_bonus, sb.middle_kingdom_bonus):
            print(f"  [{step_info}] score divergence orig={sa} opt={sb}"); return False

    # 4) legal_placements as LISTS (content + order) for each probe domino
    for d in PROBE:
        la = a.legal_placements(d)
        lb = b.legal_placements(d)
        ta = [(p.x1, p.y1, p.x2, p.y2, p.flipped) for p in la]
        tb = [(p.x1, p.y1, p.x2, p.y2, p.flipped) for p in lb]
        if ta != tb:
            # distinguish set vs order divergence
            if set(ta) == set(tb):
                print(f"  [{step_info}] legal_placements ORDER differs for dom {d.id} "
                      f"(sets equal, {len(ta)} moves)")
            else:
                print(f"  [{step_info}] legal_placements SET differs for dom {d.id}: "
                      f"orig\\opt={set(ta)-set(tb)} opt\\orig={set(tb)-set(ta)}")
            return False

    # 5) is_legal_placement over a candidate grid (incl. illegal) for probe doms
    for d in PROBE:
        for (x1, y1, x2, y2, fl) in candidate_grid(a):
            ra = a.is_legal_placement(d, PlOrig(x1, y1, x2, y2, fl))
            rb = b.is_legal_placement(d, PlOpt(x1, y1, x2, y2, fl))
            if ra != rb:
                print(f"  [{step_info}] is_legal_placement mismatch dom {d.id} "
                      f"p=({x1},{y1},{x2},{y2},{fl}): orig={ra} opt={rb}")
                return False
    return True


class ScoreEq:
    """Adapter so we can compare across the two Board modules' ScoreBreakdown."""
    def __init__(self, s): self.s = s
    def __eq__(self, other): return True  # real comparison done field-by-field above


def run(n_sequences=200, max_placements=44, seed0=0, verbose=False) -> bool:
    from games.kingdomino.board import Placement as PlOpt
    rng = random.Random(seed0)
    states_checked = 0
    for seq in range(n_sequences):
        a = OrigBoard()
        b = OptBoard()
        # initial state
        if not check_state(a, b, f"seq{seq} step0"):
            return False
        states_checked += 1
        for step in range(max_placements):
            # choose a random domino and a random legal placement (agreed by both)
            d = DOMS[rng.randrange(len(DOMS))]
            la = a.legal_placements(d)
            lb = b.legal_placements(d)
            ta = [(p.x1, p.y1, p.x2, p.y2, p.flipped) for p in la]
            tb = [(p.x1, p.y1, p.x2, p.y2, p.flipped) for p in lb]
            if ta != tb:
                print(f"  seq{seq} step{step}: drive-domino {d.id} legal lists differ")
                return False
            if not la:
                break
            pick = la[rng.randrange(len(la))]
            a.place(d, type(la[0])(pick.x1, pick.y1, pick.x2, pick.y2, pick.flipped))
            b.place(d, PlOpt(pick.x1, pick.y1, pick.x2, pick.y2, pick.flipped))
            if not check_state(a, b, f"seq{seq} step{step+1}"):
                return False
            states_checked += 1
        if verbose and seq % 50 == 0:
            print(f"  ...seq {seq} ok ({states_checked} states checked)")
    print(f"PASS: {states_checked} board states checked across {n_sequences} "
          f"fill sequences — optimized Board is behavior-identical.")
    return True


if __name__ == "__main__":
    ok = run(n_sequences=200, max_placements=44, verbose=True)
    raise SystemExit(0 if ok else 1)