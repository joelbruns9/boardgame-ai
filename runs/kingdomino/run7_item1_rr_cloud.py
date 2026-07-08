"""Run7 Item 1 (revised protocol, cloud): round-robin the three run6
candidates against EACH OTHER — no incumbent match. Per user decision
2026-07-08: the candidate with the highest aggregate win% is promoted to
current_best (banking happens back on the laptop).

3 pairings x 2500 paired seat-swapped games @ sims=300 via
promotion.evaluate_checkpoint_match. Each candidate plays 5000 games total.
Runs on the cloud box from repo root (/root/boardgame-ai).

Output: runs/kingdomino/run7_item1_rr_results.json + per-pairing JSONL.
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, ".")
from games.kingdomino.promotion import evaluate_checkpoint_match

RUN6 = Path("runs/kingdomino/cloud_80x6_run6")
OUT_JSONL = Path("runs/kingdomino/run7_item1_rr_results.jsonl")
OUT_JSON = Path("runs/kingdomino/run7_item1_rr_results.json")

GAMES = 2500
SIMS = 300
CANDS = {
    "iter_0020": RUN6 / "iter_0020.pt",
    "iter_0025": RUN6 / "iter_0025.pt",
    "iter_0040": RUN6 / "iter_0040.pt",
}
PAIRS = [
    ("iter_0020", "iter_0025", 20260708),
    ("iter_0020", "iter_0040", 20260709),
    ("iter_0025", "iter_0040", 20260710),
]

points = {k: 0.0 for k in CANDS}
games_played = {k: 0 for k in CANDS}
pairings = []
for a, b, seed in PAIRS:
    print(f"[{time.strftime('%H:%M:%S')}] START {a} vs {b} "
          f"({GAMES} games @ sims={SIMS})", flush=True)
    t0 = time.time()
    stats = evaluate_checkpoint_match(
        str(CANDS[a]), str(CANDS[b]),
        games=GAMES, sims=SIMS, device="cuda",
        batch_slots=86, leaf_batch=6, seed=seed,
    )
    payload = {
        "a": a, "b": b, "games": stats.games, "sims": SIMS, "seed": seed,
        "a_points": stats.points, "a_win_rate": stats.win_rate,
        "a_lcb": stats.lower_confidence_bound,
        "wins": stats.wins, "losses": stats.losses, "draws": stats.draws,
        "a_mean_margin": stats.mean_margin,
        "elapsed_min": (time.time() - t0) / 60.0,
    }
    pairings.append(payload)
    points[a] += stats.points
    points[b] += stats.games - stats.points
    games_played[a] += stats.games
    games_played[b] += stats.games
    with OUT_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] RESULT {a} vs {b}: "
          f"WR={payload['a_win_rate']:.4f} LCB={payload['a_lcb']:.4f} "
          f"({payload['elapsed_min']:.1f} min)", flush=True)

table = sorted(
    ((k, points[k] / games_played[k], points[k], games_played[k]) for k in CANDS),
    key=lambda r: -r[1])
summary = {
    "protocol": "round_robin_3way_no_incumbent",
    "games_per_pairing": GAMES, "sims": SIMS,
    "pairings": pairings,
    "aggregate": [
        {"candidate": k, "win_rate": wr, "points": p, "games": g}
        for k, wr, p, g in table
    ],
    "winner": table[0][0],
}
OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\naggregate standings (points share over 5000 games each):", flush=True)
for k, wr, p, g in table:
    print(f"  {k}: {wr:.4f} ({p:.1f}/{g})", flush=True)
print(f"WINNER {table[0][0]}", flush=True)
print("RR ALL DONE", flush=True)
