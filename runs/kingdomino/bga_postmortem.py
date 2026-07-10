"""Post-mortem a logged BGA game: evaluation trajectory + biggest swings.

Reads a bga_game_log JSONL (which may contain several games back-to-back —
table_id sometimes scrapes as null), segments it on 'final' records, and for
the chosen game runs every decision state through the CURRENT best net's
root inference, printing the win%/margin trajectory from the VIEWER's
perspective. The largest adverse swings are then deep-analyzed through the
full advisor (search or exact+swindle as eligible) to show what the engine
would have played.

Swing attribution: a drop AFTER your move is (mostly) your move; a drop
after the opponent's move means the previous eval was optimistic about the
position (their resource was underrated) — both are worth seeing.

Usage:
  python runs/kingdomino/bga_postmortem.py \
      --log runs/kingdomino/bga_game_log/table_unknown.jsonl \
      --game -1 --deep 3 --deep_sims 3200 --device cuda
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from games.kingdomino.web_app import (  # noqa: E402
    RecommendRequest,
    _load_nn_evaluator,
    _root_trajectory,
    recommend,
    state_from_debug_json,
)


def segment_games(rows):
    """Split the record stream into games: decisions accumulate, a 'final'
    closes the segment. Trailing decisions without a final become a segment."""
    games, cur = [], []
    for r in rows:
        if r.get("kind") == "decision":
            cur.append(r)
        elif r.get("kind") == "final":
            if cur:
                games.append({"decisions": cur, "final": r})
                cur = []
    if cur:
        games.append({"decisions": cur, "final": None})
    return games


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", required=True)
    p.add_argument("--game", type=int, default=-1,
                   help="segment index (default -1 = most recent)")
    p.add_argument("--deep", type=int, default=3,
                   help="deep-analyze this many worst swings")
    p.add_argument("--deep_sims", type=int, default=3200)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.log, encoding="utf-8")
            if l.strip()]
    games = segment_games(rows)
    print(f"{len(games)} game segment(s) in {args.log}")
    g = games[args.game]
    fin = g["final"]
    if fin:
        players = fin["final"]["players"]
        order = fin["final"].get("playerorder") or list(players)
        print("final:", " vs ".join(
            f"{players[pid]['name']} {players[pid]['score']}" for pid in order))
    decisions = g["decisions"]
    viewer = next((d.get("viewer_id") for d in decisions if d.get("viewer_id")),
                  None)
    print(f"decisions: {len(decisions)}, viewer id: {viewer}")

    # Load the canonical best net once (autodiscovered current_best).
    class _NNReq:  # minimal duck-type for _load_nn_evaluator
        checkpoint_path = None
        device = args.device
        channels = blocks = bilinear_dim = None
        nn_sims = 50
    _, net, ckpt = _load_nn_evaluator(_NNReq())
    print(f"net: {ckpt}\n")

    # RELIABILITY: during the OPPONENT's turn the viewer's board is not
    # authoritative in the scrape (BGA exposes the kingdom grid for the active
    # player only) — those evals can be wildly off. Swings are therefore
    # differenced between consecutive RELIABLE VIEWER-TURN rows only; other
    # rows still print (marked '!') for context.
    traj = []
    prev_reliable = None
    print(f"{'#':>3} {'mover':>5} {'phase':<18} {'deck':>4} "
          f"{'win%you':>8} {'margin':>7} {'swing':>7}")
    for i, rec in enumerate(decisions):
        try:
            state = state_from_debug_json(rec["state"])
        except Exception as e:
            print(f"{i:>3}  [unparseable state: {e}]")
            continue
        ri = _root_trajectory(net, state, args.device)
        actor_is_viewer = (rec.get("active_player") is not None
                           and str(rec.get("active_player")) == str(viewer))
        warned = bool(rec["state"].get("board_reconstruction_warning"))
        reliable = actor_is_viewer and not warned
        win_you = ri["win_prob"] if actor_is_viewer else 1.0 - ri["win_prob"]
        margin_you = (ri["score_margin_est"] if actor_is_viewer
                      else -ri["score_margin_est"])
        swing = None
        if reliable and prev_reliable is not None:
            swing = win_you - prev_reliable["win_you"]
        row = {"i": i, "rec": rec, "win_you": win_you,
               "margin_you": margin_you, "swing": swing,
               "reliable": reliable,
               "prev_reliable_i": (prev_reliable["i"] if prev_reliable else None),
               "mover": "You" if actor_is_viewer else "Opp"}
        traj.append(row)
        if reliable:
            prev_reliable = row
        flag = " " if reliable else "!"
        print(f"{i:>3} {row['mover']:>5} "
              f"{str(rec['state'].get('phase', '?')):<18} "
              f"{str(rec['state'].get('deck_count', '?')):>4} "
              f"{win_you:>7.1%} {margin_you:>+7.1f} "
              f"{('' if swing is None else f'{swing:+.1%}'):>7}{flag}")

    # Worst swings between consecutive viewer-turn evals: each covers YOUR
    # move at the earlier row plus the opponent's reply (and any reveal), so
    # deep-analyze the EARLIER decision — that's where a better move existed.
    swings = sorted((t for t in traj if t["swing"] is not None),
                    key=lambda t: t["swing"])[: max(0, args.deep)]
    for t in swings:
        j = t["prev_reliable_i"]
        prev = next((x for x in traj if x["i"] == j), None)
        if prev is None:
            continue
        rec = prev["rec"]
        print(f"\n=== swing {t['swing']:+.1%} between your rows {j} -> {t['i']} "
              f"— analyzing your decision at row {j} "
              f"(win%you before: {prev['win_you']:.1%}) ===")
        req = RecommendRequest(
            engine="auto", state=rec["state"], top_k=5,
            nn_sims=args.deep_sims, num_simulations=args.deep_sims,
            device=args.device, exact_max_secs=60.0, seed=0,
            swindle_budget_secs=30.0,
        )
        try:
            t0 = time.time()
            out = recommend(req)
            print(f"engine={out.get('engine')} "
                  f"({time.time() - t0:.0f}s)"
                  + (f" swindle_mode" if out.get("swindle_mode") else ""))
            for r in out.get("recommendations", [])[:5]:
                bits = [f"rank {r['rank']}"]
                if r.get("placement"):
                    pl = r["placement"]
                    bits.append(f"place d{r.get('domino_id')} "
                                f"({pl.get('x1')},{pl.get('y1')})-"
                                f"({pl.get('x2')},{pl.get('y2')})")
                if r.get("pick_domino_id") is not None:
                    bits.append(f"pick d{r['pick_domino_id']}")
                if r.get("q_win_prob") is not None:
                    bits.append(f"win%={r['q_win_prob']:.1%}")
                if r.get("exact_margin_pts") is not None:
                    bits.append(f"margin={r['exact_margin_pts']:+.1f}")
                if r.get("visit_frac") is not None:
                    bits.append(f"visits={r['visit_frac']:.1%}")
                print("   " + "  ".join(bits))
        except Exception as e:
            print(f"   deep analysis failed: {e}")

    print("\nPOSTMORTEM DONE")


if __name__ == "__main__":
    main()
