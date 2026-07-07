"""Re-cut sims_sweep_results.jsonl: are high-rung top-move flips genuine
refinements (large visit gap after flip, value moves) or coin flips among
near-equal moves (tiny top2 gap)?"""
import json

rows = [json.loads(l) for l in open(
    r"C:\Users\joeld\projects\boardgame-ai\runs\kingdomino\sims_sweep_results.jsonl",
    encoding="utf-8")]
rungs = [400, 800, 1600, 3200, 6400, 12800]

print(f"{'pair':>14} | {'n_chg':>5} | gap@hi when CHANGED (med) | gap@hi when SAME (med) | |dv| chg / same")
for a, b in zip(rungs, rungs[1:]):
    chg_gaps, same_gaps, chg_dv, same_dv = [], [], [], []
    for r in rows:
        ra, rb = r["rungs"][str(a)], r["rungs"][str(b)]
        gap = rb["top2_visit_gap"]
        dv = abs(rb["value"] - ra["value"])
        if ra["top"] != rb["top"]:
            chg_gaps.append(gap); chg_dv.append(dv)
        else:
            same_gaps.append(gap); same_dv.append(dv)
    med = lambda xs: sorted(xs)[len(xs)//2] if xs else float("nan")
    print(f"{a}->{b:>6} | {len(chg_gaps):>5} | {med(chg_gaps):>25.3f} | "
          f"{med(same_gaps):>22.3f} | {med(chg_dv):.3f} / {med(same_dv):.3f}")

# Does the 400-sim move agree with the 12800-sim move? (cheap rung vs deepest)
for base in (400, 800, 1600, 3200, 6400):
    agree = sum(r["rungs"][str(base)]["top"] == r["rungs"]["12800"]["top"] for r in rows)
    # contested only
    ac = [r for r in rows if abs(r["rungs"][str(base)]["value"]) < 0.5]
    agree_c = sum(r["rungs"][str(base)]["top"] == r["rungs"]["12800"]["top"] for r in ac)
    print(f"{base:>5} vs 12800: agree {agree}/{len(rows)} ({agree/len(rows):.1%}); "
          f"contested {agree_c}/{len(ac)} ({agree_c/len(ac):.1%})")

# positions that NEVER settle (top move differs somewhere in last 3 rungs)
unstable = [r for r in rows if len({r["rungs"][str(s)]["top"] for s in (3200, 6400, 12800)}) > 1]
gaps = sorted(r["rungs"]["12800"]["top2_visit_gap"] for r in unstable)
print(f"\nunstable in 3200..12800: {len(unstable)}/120; "
      f"median top2 gap @12800 = {gaps[len(gaps)//2]:.3f}")
stable_gaps = sorted(r["rungs"]["12800"]["top2_visit_gap"] for r in rows if r not in unstable)
print(f"stable positions: median top2 gap @12800 = {stable_gaps[len(stable_gaps)//2]:.3f}")
