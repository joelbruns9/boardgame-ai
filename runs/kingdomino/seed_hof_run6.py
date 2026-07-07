"""Seed the run6 Hall-of-Fame pool with three chain-spread opponents.

These are points along the run1->..->run5 warm-started chain chosen to maximize
trajectory distance (not "best of each run", which would be near-clones):
  - run1/iter_0066        earliest / lowest-sims -> most stylistically different
  - run3/iter_0080        run3 Elo peak (1809; beats run3 final iter_100 by ~36)
  - run5/avg_0006_0090    SWA of run5 iters 6-90: equal strength to current_best,
                          distinct point in weight space (harvest session)

Uniform sampling is intended at train time (--hof_sample_weights uniform), so
insertion order does not affect sampling; added in chain order anyway.

Local-only: writes runs/kingdomino/hof_run6/ (pool + hof_index.jsonl). Sync this
dir to the box alongside the run6 launch and point --hof_dir at it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\joeld\projects\boardgame-ai")
from games.kingdomino.hof import add_hof_entry, read_hof_index

REPO = Path(r"C:\Users\joeld\projects\boardgame-ai")
HOF_DIR = REPO / "runs" / "kingdomino" / "hof_run6"


def relativize_index(hof_dir: Path, repo: Path) -> None:
    """add_hof_entry records absolute paths; rewrite path/source to repo-relative
    POSIX so the pool is portable across machines (e.g. uploaded to the box and
    loaded with cwd=repo root). Without this, load_hof_net fails on the box with
    the laptop's C:\\... paths."""
    index = hof_dir / "hof_index.jsonl"
    rows = [json.loads(x) for x in index.read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        for key in ("path", "source"):
            if r.get(key):
                rel = Path(r[key].replace("\\", "/")).resolve().relative_to(repo.resolve())
                r[key] = rel.as_posix()
    with index.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

SEEDS = [
    (REPO / "runs/kingdomino/cloud_80x6_run1/iter_0066.pt", "run1_iter66", 66),
    (REPO / "runs/kingdomino/cloud_80x6_run3/iter_0080.pt", "run3_iter80", 80),
    (REPO / "runs/kingdomino/cloud_80x6_run5/avg_0006_0090.pt", "run5_avg_0006_0090", None),
]

for src, tag, iteration in SEEDS:
    entry = add_hof_entry(
        src, hof_dir=HOF_DIR, tag=tag, iteration=iteration,
        metadata={"seeded_for": "run6", "chain_role": tag},
    )
    print(f"  + {tag:20s} <- {src.name}  "
          f"[{entry.channels}x{entry.blocks}]  sha256={entry.sha256[:12]}")

relativize_index(HOF_DIR, REPO)

print(f"\nHOF pool: {HOF_DIR}")
for e in read_hof_index(HOF_DIR):
    print(f"  {Path(e.path).name}  tag={e.tag}  iter={e.iteration}  "
          f"arch={e.channels}x{e.blocks}")
print("HOF SEED DONE")
