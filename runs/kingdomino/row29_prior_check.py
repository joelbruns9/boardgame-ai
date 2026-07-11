"""Row-29 prior check (run10 pre-registered measurement #2).

Reconstructs the opponent's squeeze decision node in the kylechu20 loss
(last segment of table_unknown.jsonl) by stepping row 29's reliable state
through the three known plies, with placements recovered from the row-32
reliable scrape:

  ply 1  viewer:   place d18 (board-0 terrain diff 29->32), pick d4
  ply 2  opponent: place d32 (board-1 domino_id == 32),     pick d33
  ply 3  opponent: <node>  actual reply = place d34, pick d8
                   (historic: 4.7% prior / 0.62% of 3200 sims)

Then reads each checkpoint's policy prior on that reply at the node.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"C:/Users/joeld/projects/boardgame-ai")
sys.path.insert(0, str(REPO))

from games.kingdomino.web_app import state_from_debug_json, _load_nn_evaluator
from games.kingdomino.encoder import encode_state
from games.kingdomino.action_codec import encode_action
from runs.kingdomino.bga_postmortem import segment_games

LOG = REPO / "runs/kingdomino/bga_game_log/table_unknown.jsonl"

rows = [json.loads(l) for l in open(LOG, encoding="utf-8") if l.strip()]
dec = segment_games(rows)[-1]["decisions"]
s29 = state_from_debug_json(dec[29]["state"])
s32 = state_from_debug_json(dec[32]["state"])
assert s29.current_actor == 0 and sorted(s29.current_row) == [4, 8, 33, 48]


def cells_of_id(board, dom_id):
    ys, xs = np.nonzero(board.domino_id == dom_id)
    return sorted(zip(xs.tolist(), ys.tolist()))


def new_terrain_cells(b_from, b_to):
    m = (b_to.terrain > 0) & ~(b_from.terrain > 0)
    ys, xs = np.nonzero(m)
    return sorted(zip(xs.tolist(), ys.tolist()))


def act_by(state, cells, pick):
    hits = [a for a in state.legal_actions()
            if getattr(a, "placement", None) is not None
            and sorted(a.placement.cells) == cells
            and a.pick_domino_id == pick]
    assert hits, f"no legal action places at {cells} picking d{pick}"
    return hits[0]  # symmetric-orientation twins step identically


d18_cells = new_terrain_cells(s29.boards[0], s32.boards[0])
assert len(d18_cells) == 2, f"viewer placed cells: {d18_cells}"
d32_cells = cells_of_id(s32.boards[1], 32)
d34_cells = cells_of_id(s32.boards[1], 34)
assert len(d32_cells) == 2 and len(d34_cells) == 2

child1 = s29.step(act_by(s29, d18_cells, 4))
assert child1.current_actor == 1, f"expected opponent after ply1, got {child1.current_actor}"
node = child1.step(act_by(child1, d32_cells, 33))
assert node.current_actor == 1, f"expected opponent again, got {node.current_actor}"
assert sorted(node.current_row) == [8, 48]

acts = node.legal_actions()
reply = act_by(node, d34_cells, 8)
reply_idx = int(encode_action(reply, node))
print(f"opponent node reconstructed: {len(acts)} legal actions, "
      f"reply = place d34 @ {d34_cells}, pick d8")

for name, ckpt in [
    ("run8 banked avg (baseline)", "runs/kingdomino/best_checkpoint/current_best.pt"),
    ("run10 avg_iter_0050_k8", "runs/kingdomino/cloud_80x6_run10/avg_iter_0050_k8.pt"),
    ("run10 iter_0052", "runs/kingdomino/cloud_80x6_run10/iter_0052.pt"),
]:
    class R:
        checkpoint_path = str(REPO / ckpt)
        device = "cuda"
        channels = blocks = bilinear_dim = None
        nn_sims = 50
    _, net, _p = _load_nn_evaluator(R())
    mb, ob, flat = encode_state(node, node.current_actor)
    device = next(net.parameters()).device
    with torch.inference_mode():
        _own, _opp, _win, logits = net(
            torch.from_numpy(mb).unsqueeze(0).to(device),
            torch.from_numpy(ob).unsqueeze(0).to(device),
            torch.from_numpy(flat).unsqueeze(0).to(device))
    idx_list = [int(encode_action(a, node)) for a in acts]
    uniq = sorted(set(idx_list))
    idxs = torch.tensor(uniq, dtype=torch.long, device=logits.device)
    priors = torch.softmax(logits[0, idxs], dim=0).cpu().numpy()
    pmap = dict(zip(uniq, priors))
    reply_p = pmap[reply_idx]
    order = sorted(pmap.values(), reverse=True)
    rank = order.index(reply_p) + 1
    print(f"{name}:")
    print(f"  squeeze reply prior = {reply_p:.3%}  rank {rank}/{len(uniq)}  "
          f"(top prior = {order[0]:.3%})")

print("ROW29 PRIOR CHECK DONE")
