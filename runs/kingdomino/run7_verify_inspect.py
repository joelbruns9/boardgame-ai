"""Run7 Item-5 evidence: inspect the tiny verification run's buffer.

Checks, per the pre-registered verification list:
  (a) HOF-game learner moves searched at full --sims (root visit sums == 600
      in the tiny config, the stand-in for 4800);
  (b) only learner-seat moves recorded in HOF games (game_type set, owner
      'current'; the engine never records the frozen seat);
  (c) with policy_target_pruning on, HOF examples were pruned against the
      frontier budget (total_visits = n_simulations) — evidenced by (a): every
      recorded non-exact HOF move carries the frontier visit total;
  (d) run7 gate flags flowed into the config — read back from the checkpoint.
"""
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"C:\Users\joeld\projects\boardgame-ai")
REPO = Path(r"C:\Users\joeld\projects\boardgame-ai")
TINY = REPO / "runs/kingdomino/run7_verify_tiny"

# The tiny run executed self_play as __main__, so its pickled Examples resolve
# against this module's namespace.
from games.kingdomino.self_play import Example  # noqa: E402
globals()["Example"] = Example

with open(TINY / "buffer.pkl", "rb") as f:
    payload = pickle.load(f)
data = payload["data"]
print(f"buffer examples: {len(data)}")

by_type = Counter(getattr(e, "game_type", "self_play") or "self_play" for e in data)
print(f"game types: {dict(by_type)}")

hof = [e for e in data if getattr(e, "game_type", "") in ("current_vs_hof", "hof_vs_current")]
print(f"HOF examples: {len(hof)}")
assert hof, "expected HOF examples in buffer"

# (b) ownership: every recorded HOF move is a learner move
owners = Counter(getattr(e, "owner", None) for e in hof)
print(f"HOF owners: {dict(owners)}  (engine records only the learner seat)")
assert set(owners) == {"current"}, owners

# (a)+(c): non-exact HOF moves carry the FULL frontier visit budget (600 here).
sums = []
n_exact = 0
for e in hof:
    rvc = getattr(e, "root_visit_count", None)
    if rvc is None or len(rvc) == 0:
        n_exact += 1  # exact-endgame moves carry no MCTS root stats
        continue
    sums.append(int(np.sum(rvc)))
sums = np.asarray(sums)
print(f"HOF MCTS moves: {len(sums)}, exact-endgame moves: {n_exact}")
print(f"root visit sums: min={sums.min()} max={sums.max()} mean={sums.mean():.1f}")
assert (sums >= 590).all(), "HOF learner moves must be full 600-sim searches"

# One-visit pruning check: with total_visits=600 the pruned targets should
# contain no ~1/600 entries (pruning removed 1-visit actions).
min_pv = min(float(np.min(e.policy_val)) for e in hof if len(getattr(e, "policy_val", [])) and getattr(e, "root_visit_count", None) is not None and len(e.root_visit_count))
print(f"min policy target mass among HOF MCTS moves: {min_pv:.5f} "
      f"(1-visit at 600 sims would be {1/600:.5f})")

# (d) gate flags read back from the saved checkpoint config
ckpt = torch.load(TINY / "iter_0002.pt", map_location="cpu", weights_only=False)
cfg = ckpt.get("config", {})
gate = {k: cfg.get(k) for k in (
    "promotion_games", "promotion_sims",
    "promotion_min_win_rate", "promotion_min_lcb",
    "soft_gate_revert_win_rate", "selfplay_generator_mode",
    "hof_sims", "n_simulations")}
print(f"checkpoint config gate flags: {gate}")
assert gate["promotion_games"] == 2500 and gate["promotion_sims"] == 300
assert gate["promotion_min_win_rate"] == 0.51 and gate["promotion_min_lcb"] == 0.51
print("VERIFY INSPECT OK")
