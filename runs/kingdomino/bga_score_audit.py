"""Score-head calibration audit over logged BGA games.

For every finished logged game, compares the net's own/opp final-score
projections (reliable viewer-turn rows only) against the actual finals.
This is the eval-suite metric that emerged from the 2026-07-10 loss
post-mortem: the score heads regress toward the self-play equilibrium
(~135) and are under-dispersed on human games — the eventual squeezed
player is over-projected by ~+25-30 while winners can be under-projected.
Run against each new checkpoint to see whether diversity training widens
the heads' projected range.

Usage: python runs/kingdomino/bga_score_audit.py --log <jsonl> [--device cuda]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from games.kingdomino.web_app import (  # noqa: E402
    _load_nn_evaluator, _root_trajectory, state_from_debug_json,
)
from runs.kingdomino.bga_postmortem import segment_games  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint", default=None,
                   help="override checkpoint (default: canonical best)")
    args = p.parse_args()

    class R:
        checkpoint_path = args.checkpoint
        device = args.device
        channels = blocks = bilinear_dim = None
        nn_sims = 50
    _, net, ckpt = _load_nn_evaluator(R())
    print(f"net: {ckpt}")

    rows = [json.loads(l) for l in open(args.log, encoding="utf-8") if l.strip()]
    games = segment_games(rows)
    print(f"{'game':>4} {'you(act)':>8} {'opp(act)':>8} "
          f"{'err_you':>8} {'err_opp':>8} {'n':>4}")
    all_you, all_opp = [], []
    for gi, g in enumerate(games):
        if not g["final"]:
            continue
        fin = g["final"]["final"]
        viewer = next((x.get("viewer_id") for x in g["decisions"]
                       if x.get("viewer_id")), None)
        if viewer is None or str(viewer) not in fin["players"]:
            continue
        you_act = fin["players"][str(viewer)]["score"]
        opp_act = next(p["score"] for pid, p in fin["players"].items()
                       if pid != str(viewer))
        ey, eo = [], []
        for rec in g["decisions"]:
            if str(rec.get("active_player")) != str(viewer):
                continue
            if rec["state"].get("board_reconstruction_warning"):
                continue
            try:
                state = state_from_debug_json(rec["state"])
            except Exception:
                continue
            ri = _root_trajectory(net, state, args.device)
            ey.append(ri["own_score_est"] - you_act)
            eo.append(ri["opp_score_est"] - opp_act)
        if len(ey) < 5:
            continue
        all_you.append(statistics.mean(ey))
        all_opp.append(statistics.mean(eo))
        print(f"{gi:>4} {you_act:>8} {opp_act:>8} "
              f"{statistics.mean(ey):>+8.0f} {statistics.mean(eo):>+8.0f} "
              f"{len(ey):>4}")
    if all_you:
        print(f"\nmean err_you {statistics.mean(all_you):+.1f}  "
              f"mean err_opp {statistics.mean(all_opp):+.1f}  "
              f"(over {len(all_you)} games)")
    print("SCORE AUDIT DONE")


if __name__ == "__main__":
    main()
