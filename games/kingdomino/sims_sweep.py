"""Advisor decision-stability sweep: find the sims knee.

Measures where additional simulations stop changing the ADVICE — the practical
"ideal sims" for the advisor. (Divergence from the raw prior grows ~log(sims)
forever; that is search working, not a stopping signal. The stopping signal is
successive-rung stability: if 2N sims picks the same move as N sims almost
always, the extra latency buys nothing.)

Method:
  1. Generate real positions by playing games with the CURRENT checkpoint
     driving both sides via the Rust advisor search at a light budget
     (--gen_sims). Snapshot positions with deck > 4 (deck <= 4 belongs to the
     exact solver; sims are irrelevant there).
  2. For each position, run the Rust advisor open-loop search at each rung of
     --rungs, with a per-position seed shared across rungs (common random
     numbers: the determinization stream prefix is identical, so rung-to-rung
     differences are due to the added sims, not resampled decks).
  3. Report, per successive rung pair (N -> 2N):
       - top-move change rate (overall, by deck-size bucket, and split into
         contested |v|<0.5 vs decided |v|>=0.5 positions)
       - mean |value drift|
       - mean latency per search
     The knee is the first pair whose contested change rate drops below
     --knee_threshold.

Writes per-position results to --out (JSONL) so the table can be re-cut
without re-searching. Run on the laptop; does not touch training.

Usage:
  python -m games.kingdomino.sims_sweep --checkpoint runs/kingdomino/best_checkpoint/current_best.pt
  python -m games.kingdomino.sims_sweep --n_positions 60 --rungs 400,800,1600,3200
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import kingdomino_rust as kr
from games.kingdomino.game import GameState, Phase
from games.kingdomino.action_codec import encode_action
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.self_play import KingdominoNet, make_rust_evaluator


def load_net(path: str, device: str) -> KingdominoNet:
    ck = torch.load(path, map_location="cpu")
    cfg = ck.get("config", {})
    net = KingdominoNet(
        channels=int(cfg.get("channels", 80)),
        blocks=int(cfg.get("blocks", 6)),
        bilinear_dim=int(cfg.get("bilinear_dim", 64)),
        score_scale=float(cfg.get("score_scale", 160.0)),
    )
    net.load_state_dict(ck["model_state"])
    return net.to(device).eval()


def default_checkpoint() -> str:
    best = Path("runs/kingdomino/best_checkpoint/current_best.pt")
    if best.exists():
        return str(best)
    cands = sorted(Path("runs/kingdomino").glob("*/iter_*.pt"))
    if not cands:
        raise SystemExit("No checkpoint found; pass --checkpoint.")
    return str(cands[-1])


def top_move(children) -> int:
    """children: [(joint_idx, visits, value_sum, prior)] -> joint idx of the
    visit-count argmax (ties -> lowest joint index, matching greedy select)."""
    return int(max(children, key=lambda c: (c[1], -c[0]))[0])


def generate_positions(evaluator, n_positions: int, gen_sims: int, seed: int,
                       min_deck: int = 8) -> list[GameState]:
    """Self-play games with the checkpoint driving both sides (light search);
    snapshot one position per (game, deck_len) with deck >= min_deck so the
    sample spans phases without near-duplicate consecutive roots."""
    rng = random.Random(seed)
    positions: list[GameState] = []
    game = 0
    while len(positions) < n_positions:
        st = GameState.new(seed=seed + game * 977)
        game += 1
        seen_decklens: set[int] = set()
        while st.phase != Phase.GAME_OVER:
            if (st.phase == Phase.PLACE_AND_SELECT
                    and len(st.deck) >= min_deck
                    and len(st.deck) not in seen_decklens
                    and len(st.legal_actions()) >= 2):
                seen_decklens.add(len(st.deck))
                positions.append(st)
                if len(positions) >= n_positions:
                    break
            rs = _rust_state_from_python(st)
            if st.phase == Phase.INITIAL_SELECTION or len(st.deck) <= 4:
                # Trivial or solver territory: play a uniform-random legal move
                # (sweep never measures these positions).
                st = st.step(rng.choice(st.legal_actions()))
                continue
            children, _v = kr.advisor_open_loop_search(
                rs, evaluator, gen_sims, dirichlet_eps=0.0, cpuct=1.5,
                seed=rng.getrandbits(63), leaf_batch=8, alpha=0.5)
            idx_to_action = {int(encode_action(a, st)): a for a in st.legal_actions()}
            st = st.step(idx_to_action[top_move(children)])
    return positions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None,
                   help="Net to sweep (default: best_checkpoint/current_best.pt, "
                        "else newest iter_*.pt).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_positions", type=int, default=120)
    p.add_argument("--rungs", default="400,800,1600,3200,6400,12800",
                   help="Comma-separated sims rungs (ascending).")
    p.add_argument("--gen_sims", type=int, default=200,
                   help="Sims for position-generation games (play quality only).")
    p.add_argument("--seed", type=int, default=20260707)
    p.add_argument("--knee_threshold", type=float, default=0.05,
                   help="Contested change rate below which a rung pair is 'stable'.")
    p.add_argument("--out", default="runs/kingdomino/sims_sweep_results.jsonl")
    args = p.parse_args()

    rungs = [int(x) for x in args.rungs.split(",")]
    assert rungs == sorted(rungs) and len(rungs) >= 2, "--rungs must ascend, >=2 entries"

    ckpt = args.checkpoint or default_checkpoint()
    print(f"checkpoint: {ckpt}")
    net = load_net(ckpt, args.device)
    evaluator = make_rust_evaluator(net, device=args.device, alpha=0.5)

    print(f"generating {args.n_positions} positions (gen_sims={args.gen_sims}) ...")
    positions = generate_positions(evaluator, args.n_positions, args.gen_sims,
                                   args.seed)
    from collections import Counter
    print("deck sizes:", dict(sorted(Counter(len(s.deck) for s in positions).items())))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, st in enumerate(positions):
            rs = _rust_state_from_python(st)
            pos_seed = (args.seed * 1_000_003 + i) & 0xFFFF_FFFF_FFFF_FFFF
            rec = {"i": i, "deck": len(st.deck),
                   "n_legal": len(st.legal_actions()), "rungs": {}}
            for n_sims in rungs:
                t0 = time.perf_counter()
                children, v0 = kr.advisor_open_loop_search(
                    rs, evaluator, n_sims, dirichlet_eps=0.0, cpuct=1.5,
                    seed=pos_seed, leaf_batch=8, alpha=0.5)
                elapsed = time.perf_counter() - t0
                ranked = sorted(children, key=lambda c: -c[1])
                gap = (ranked[0][1] - ranked[1][1]) / max(1, n_sims) if len(ranked) > 1 else 1.0
                rec["rungs"][str(n_sims)] = {
                    "top": top_move(children), "value": v0,
                    "top2_visit_gap": gap, "ms": round(elapsed * 1000, 1),
                }
            fh.write(json.dumps(rec) + "\n")
            rows.append(rec)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(positions)} positions swept", flush=True)

    # ── report ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print(f"DECISION-STABILITY SWEEP  ({len(rows)} positions, ckpt={Path(ckpt).name})")
    print("=" * 74)
    print(f"{'rungs':>14} | {'chg all':>7} {'chg contested':>13} {'chg decided':>11} | "
          f"{'|dv|':>6} | {'ms/search':>9}")
    knee = None
    for a, b in zip(rungs, rungs[1:]):
        pair = f"{a}->{b}"
        changed, chg_c, n_c, chg_d, n_d, dv, ms = 0, 0, 0, 0, 0, 0.0, 0.0
        for r in rows:
            ra, rb = r["rungs"][str(a)], r["rungs"][str(b)]
            ch = ra["top"] != rb["top"]
            changed += ch
            dv += abs(rb["value"] - ra["value"])
            ms += rb["ms"]
            if abs(ra["value"]) < 0.5:
                n_c += 1
                chg_c += ch
            else:
                n_d += 1
                chg_d += ch
        n = len(rows)
        rate_c = chg_c / n_c if n_c else float("nan")
        print(f"{pair:>14} | {changed / n:7.1%} {rate_c:13.1%} "
              f"{(chg_d / n_d if n_d else float('nan')):11.1%} | "
              f"{dv / n:6.3f} | {ms / n:8.0f}ms")
        if knee is None and n_c and rate_c < args.knee_threshold:
            knee = a
    print("\nby deck size (change rate, all rung pairs pooled):")
    bucket: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        chs = [r["rungs"][str(a)]["top"] != r["rungs"][str(b)]["top"]
               for a, b in zip(rungs, rungs[1:])]
        bucket[r["deck"]].append(sum(chs))
    for deck in sorted(bucket):
        vals = bucket[deck]
        pairs = len(rungs) - 1
        print(f"  deck={deck:>2}: {sum(vals) / (len(vals) * pairs):6.1%}  (n={len(vals)})")
    if knee is not None:
        print(f"\nknee (contested change rate < {args.knee_threshold:.0%}): "
              f"~{knee} sims — rungs beyond this change contested advice rarely.")
    else:
        print(f"\nno knee found below {rungs[-1]} sims at threshold "
              f"{args.knee_threshold:.0%} — contested positions still benefit "
              f"from more search; consider extending --rungs.")
    print(f"per-position detail: {out_path}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
