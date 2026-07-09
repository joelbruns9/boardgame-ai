"""Run7 peak certification: the missing transitivity link.

Run6's best was banked via a round-robin among run6 checkpoints and never
beat run5's net at high power; run7's peak beat run6's best (55.1%). This
match closes the chain by testing the run7 peak DIRECTLY against run5's
promoted net — plus run3_iter80 as an off-lineage second opinion.

Decision (pre-registered): if the peak clears run5 (LCB > 0.50), run8
warm-starts from the run7 peak; if it loses, run6+run7 were a lineage detour
and run8 warm-starts from run5's net.

Runs on the cloud box from repo root.
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, ".")
from games.kingdomino.promotion import evaluate_checkpoint_match

PEAK = "runs/kingdomino/cloud_80x6_run7/run7_peak_iter25.pt"
OUT = Path("runs/kingdomino/run7_peak_cert_results.json")
MATCHES = [
    ("run5_iter_0005", "runs/kingdomino/cloud_80x6_run5/iter_0005.pt",
     2500, 20260713),
    ("run3_iter_0080", "runs/kingdomino/cloud_80x6_run3/iter_0080.pt",
     1500, 20260714),
]

results = []
for name, base, games, seed in MATCHES:
    if not Path(base).exists():
        # Original run dirs may not be on the box; sha-identical copies are.
        # run5 iter_0005: run6's never-promoted gate baseline, and hof_run7's
        # pre_run7 entry. run3 iter_0080: its HOF copies.
        alts = {
            "run5_iter_0005": [
                "runs/kingdomino/cloud_80x6_run6/current_best.pt",
                *map(str, sorted(Path("runs/kingdomino/hof_run7").glob(
                    "hof_manual_pre_run7_*.pt"))),
            ],
            "run3_iter_0080": list(map(str, sorted(
                Path("runs/kingdomino/hof_run6").glob(
                    "hof_iter_0080_run3_iter80_*.pt")))),
        }.get(name, [])
        base = next((a for a in alts if Path(a).exists()), None)
        if base is None:
            print(f"SKIP {name}: no copy found on this machine", flush=True)
            continue
        print(f"  using sha-identical copy: {base}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] START peak vs {name} "
          f"({games} games @ sims=300)", flush=True)
    t0 = time.time()
    s = evaluate_checkpoint_match(
        PEAK, base, games=games, sims=300, device="cuda",
        batch_slots=86, leaf_batch=6, seed=seed,
    )
    payload = {
        "baseline": name, "baseline_path": base, "games": s.games,
        "sims": 300, "seed": seed,
        "peak_win_rate": s.win_rate, "peak_lcb": s.lower_confidence_bound,
        "wins": s.wins, "losses": s.losses, "draws": s.draws,
        "peak_mean_margin": s.mean_margin,
        "elapsed_min": (time.time() - t0) / 60.0,
    }
    results.append(payload)
    print(f"[{time.strftime('%H:%M:%S')}] RESULT peak vs {name}: "
          f"WR={s.win_rate:.4f} LCB={s.lower_confidence_bound:.4f} "
          f"({s.wins}-{s.losses}-{s.draws}, margin {s.mean_margin:+.2f})",
          flush=True)

OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
print("PEAK CERT DONE", flush=True)
