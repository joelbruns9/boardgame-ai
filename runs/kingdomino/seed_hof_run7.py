"""Seed the run7 Hall-of-Fame pool (curated near-peer opponents only).

Run7 drops run6's weakest opponent, run1/iter_0066 (Elo ~1648 vs run3_iter80's
~1809 and run5_avg's ~current_best level): with the learner now searching HOF
games at full strength, games vs run1_iter66 are blowouts and carry no useful
diversity.  Contested games do.  Pool:
  - run3/iter_0080        run3 Elo peak (1809)
  - run5/avg_0006_0090    SWA of run5 iters 6-90, ties current_best
  - (added by run7_item1_bank.py on promotion) pre_run7 = the outgoing
    current_best, tag pre_run7

Uniform sampling at train time (--hof_sample_weights uniform).  Idempotent:
add_hof_entry dedups by source sha256.  Local-only: writes
runs/kingdomino/hof_run7/ (pool + hof_index.jsonl); sync to the box and point
--hof_dir at it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\joeld\projects\boardgame-ai")
from games.kingdomino.hof import add_hof_entry, read_hof_index

REPO = Path(r"C:\Users\joeld\projects\boardgame-ai")
HOF_DIR = REPO / "runs" / "kingdomino" / "hof_run7"


def relativize_index(hof_dir: Path, repo: Path) -> None:
    """add_hof_entry records absolute paths; rewrite path/source to repo-relative
    POSIX so the pool is portable across machines (box loads with cwd=repo)."""
    index = hof_dir / "hof_index.jsonl"
    rows = [json.loads(x) for x in index.read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        for key in ("path", "source"):
            if r.get(key):
                p = Path(r[key].replace("\\", "/"))
                if p.is_absolute():
                    r[key] = p.resolve().relative_to(repo.resolve()).as_posix()
    with index.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


SEEDS = [
    (REPO / "runs/kingdomino/cloud_80x6_run3/iter_0080.pt", "run3_iter80", 80),
    (REPO / "runs/kingdomino/cloud_80x6_run5/avg_0006_0090.pt", "run5_avg_0006_0090", None),
]

if __name__ == "__main__":
    for src, tag, iteration in SEEDS:
        entry = add_hof_entry(
            src, hof_dir=HOF_DIR, tag=tag, iteration=iteration,
            metadata={"seeded_for": "run7", "chain_role": tag},
        )
        print(f"  + {tag:20s} <- {src.name}  "
              f"[{entry.channels}x{entry.blocks}]  sha256={entry.sha256[:12]}")

    relativize_index(HOF_DIR, REPO)

    print(f"\nHOF pool: {HOF_DIR}")
    for e in read_hof_index(HOF_DIR):
        print(f"  {Path(e.path).name}  tag={e.tag}  iter={e.iteration}  "
              f"arch={e.channels}x{e.blocks}")
    print("HOF SEED DONE")
