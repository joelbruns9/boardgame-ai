"""Run7 Item 1 banking (revised protocol, user decision 2026-07-08): the
round-robin winner among run6 iter_0020/0025/0040 — highest aggregate win%
in run7_item1_rr_results.json — is promoted to current_best unconditionally
(no incumbent match; run6's in-run gates already had iter 20/25 at WR 0.52
vs current_best).

On promotion:
  1. the outgoing current_best is added to the run7 HOF pool (tag pre_run7)
     BEFORE the copy, so the entry is sourced from the outgoing file;
  2. the winner is copied to runs/kingdomino/best_checkpoint/current_best.pt
     via promotion.promote_current_best (timestamped backup + audit log).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\joeld\projects\boardgame-ai")
from games.kingdomino.hof import add_hof_entry
from games.kingdomino.promotion import (
    PromotionDecision,
    promote_current_best,
    promotion_payload,
)
from seed_hof_run7 import HOF_DIR, relativize_index, REPO

RESULTS = REPO / "runs/kingdomino/run7_item1_rr_results.json"
CURRENT_BEST = REPO / "runs/kingdomino/best_checkpoint/current_best.pt"
RUN6 = REPO / "runs/kingdomino/cloud_80x6_run6"

summary = json.loads(RESULTS.read_text(encoding="utf-8"))
print("aggregate standings:")
for row in summary["aggregate"]:
    print(f"  {row['candidate']}: {row['win_rate']:.4f} "
          f"({row['points']}/{row['games']})")
winner = summary["winner"]
winner_path = RUN6 / f"{winner}.pt"
assert winner_path.exists(), winner_path
assert winner == summary["aggregate"][0]["candidate"]

print(f"\nPROMOTING run6 {winner} (round-robin winner)")

# 1. Preserve the outgoing current_best in the run7 HOF pool first.
entry = add_hof_entry(
    CURRENT_BEST, hof_dir=HOF_DIR, tag="pre_run7",
    metadata={"seeded_for": "run7",
              "chain_role": "pre_run7_current_best_run5_iter_0005"},
)
relativize_index(HOF_DIR, REPO)
print(f"  + pre_run7 HOF entry: {Path(entry.path).name}")

# 2. Promote. The decision is recorded verbatim as the revised rule — there is
# no threshold check by design.
decision = PromotionDecision(
    passed=True, bootstrap=False,
    reasons=[
        "revised Item-1 protocol (user, 2026-07-08): 3-way round-robin, "
        "highest aggregate win% promoted unconditionally",
        f"{winner} aggregate {summary['aggregate'][0]['win_rate']:.4f} over "
        f"{summary['aggregate'][0]['games']} games",
    ],
    match=None, fixed_suite=None, min_win_rate=0.0, min_lcb=0.0,
)
payload = promotion_payload(
    candidate=winner_path, current_best=CURRENT_BEST, decision=decision,
    extra={"context": "run7_item1_banking_round_robin",
           "round_robin": summary,
           "hof_pre_run7_entry": entry.path},
)
# best_dir explicitly absolute: the promotion module's defaults are
# repo-relative and would resolve against this script's cwd.
target = promote_current_best(
    winner_path, best_dir=REPO / "runs/kingdomino/best_checkpoint",
    payload=payload)
print(f"  current_best <- {winner_path}")
print(f"  promoted to {target}")
print("ITEM1 BANK DONE")
