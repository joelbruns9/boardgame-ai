"""Noise-ball harvest H2H: avg_0006_0090 (candidate) vs iter_0005 (current_best).
Paired seat-swapped match via promotion.evaluate_checkpoint_match.
516 games @ sims=100 mirrors run5's soft_gate promotion checks.
Optional argv: games sims out_path (defaults 516 100 harvest_h2h_result.json).
"""
import json, sys, time

sys.path.insert(0, r"C:\Users\joeld\projects\boardgame-ai")
from games.kingdomino.promotion import evaluate_checkpoint_match

CAND = r"C:\Users\joeld\projects\boardgame-ai\runs\kingdomino\cloud_80x6_run5\avg_0006_0090.pt"
BASE = r"C:\Users\joeld\projects\boardgame-ai\runs\kingdomino\cloud_80x6_run5\iter_0005.pt"

games = int(sys.argv[1]) if len(sys.argv) > 1 else 516
sims = int(sys.argv[2]) if len(sys.argv) > 2 else 100
OUT = sys.argv[3] if len(sys.argv) > 3 else \
    r"C:\Users\joeld\projects\boardgame-ai\runs\kingdomino\harvest_h2h_result.json"

t0 = time.time()
stats = evaluate_checkpoint_match(
    CAND, BASE,
    games=games, sims=sims, device="cuda",
    batch_slots=64, leaf_batch=6, seed=20260707,
)
payload = {
    "candidate": "avg_0006_0090",
    "baseline": "run5_iter_0005",
    "games": stats.games,
    "sims": sims,
    "points": stats.points,
    "win_rate": stats.win_rate,
    "lcb": stats.lower_confidence_bound,
    "wins": stats.wins,
    "losses": stats.losses,
    "draws": stats.draws,
    "mean_margin": stats.mean_margin,
    "elapsed_min": (time.time() - t0) / 60.0,
}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
print(json.dumps(payload, indent=2))
print("H2H DONE")
