"""Margin-regret canary — does the TRAINED net actually fight for margin in
positions it has already won?

This is the empirical acceptance test for the win-gated leaf value
(mcts_az: leaf = (1-B)*win + B*win^4*margin). The sweep validates the FORMULA;
this validates the LEARNED POLICY end-to-end — a well-shaped value can still fail
to change play if the priors/search don't propagate it or the score head is
miscalibrated.

Method (rigorous, no rollouts):
  * Reach no-chance endgame positions (deck in {0,4}) that the mover has WON
    under optimal play (exact solve value >= --win_threshold in the mover frame).
  * By minimax, the exact ROOT value already equals the BEST achievable value
    over the mover's moves, so:
        regret = root_exact_value(mover)  -  exact_value(mover's chosen child)
    is >= 0, and is exactly the margin the net's search left on the table.
  * The net's move = argmax visit count from OpenLoopMCTS(net) at temperature 0.
  * We report how often regret > 0 (lazy-play rate), and its size in both
    value units and approximate SCORE POINTS.

Because the ground truth is an exact solve, this is a NECESSARY condition: a net
that can't maximize margin when the answer is exactly computable is certainly
lazy in the (unsolvable) midgame too. Run it at iter ~5-10 of the from-scratch
run once a 333-flat checkpoint exists:

    python -m games.kingdomino.margin_canary --checkpoint runs/.../iter_008.pt \
        --positions 100 --sims 400

With no --checkpoint it builds a RANDOM net (smoke test): the harness runs and
prints a (meaningless, ~random) regret — use it only to confirm the plumbing.
"""
from __future__ import annotations

import argparse
import math
import random
import sys

import numpy as np
import torch

from games.kingdomino.game import GameState, Phase
from games.kingdomino.network import KingdominoNet
from games.kingdomino.endgame_solver import exact_endgame_value
from games.kingdomino.mcts_az import (
    OpenLoopMCTS,
    make_serial_evaluator,
    select_move,
)


def _is_solvable_endgame(state: GameState) -> bool:
    """No-chance endgame (matches the Rust is_no_chance_endgame_state contract)."""
    if state.phase == Phase.PLACE_AND_SELECT:
        return len(state.deck) in (0, 4)
    if state.phase == Phase.FINAL_PLACEMENT:
        return len(state.deck) == 0
    return False


def _mover_frame(value_p0: float, mover: int) -> float:
    """Convert a player-0-frame value to the mover's frame."""
    return value_p0 if mover == 0 else -value_p0


def _value_to_points(delta_value: float, B: float, margin_gain: float,
                     score_scale: float) -> float:
    """Approximate the score-point size of a value gap between two WON positions.

    For a decided win, value = (1-B) + B*tanh(m_norm*margin_gain), m_norm = pts/scale.
    We invert at the top margin to get a local pts-per-value slope and scale the gap
    by it — a rough but interpretable translation (exact only for small gaps)."""
    # local slope d(value)/d(pts) at a representative 20-pt lead
    m0 = 20.0 / score_scale
    dv_dpts = B * margin_gain * (1.0 - math.tanh(m0 * margin_gain) ** 2) / score_scale
    return delta_value / dv_dpts if dv_dpts > 0 else float("nan")


def build_net(args):
    """Return (net, margin_gain, alpha, score_scale). Loads arch+leaf params from
    the checkpoint config when available; random-init for smoke otherwise."""
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        channels = cfg.get("channels", args.channels)
        blocks = cfg.get("blocks", args.blocks)
        bilinear = cfg.get("bilinear_dim", args.bilinear_dim)
        score_scale = float(cfg.get("score_scale", args.score_scale))
        margin_gain = float(cfg.get("margin_gain", args.margin_gain))
        alpha = float(cfg.get("alpha", args.alpha))
        net = KingdominoNet(channels=channels, blocks=blocks,
                            bilinear_dim=bilinear, score_scale=score_scale)
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        net.load_state_dict(state)
        print(f"loaded checkpoint {args.checkpoint} "
              f"(channels={channels} blocks={blocks} bilinear={bilinear} "
              f"margin_gain={margin_gain} alpha/B={alpha})")
    else:
        score_scale, margin_gain, alpha = args.score_scale, args.margin_gain, args.alpha
        net = KingdominoNet(channels=args.channels, blocks=args.blocks,
                            bilinear_dim=args.bilinear_dim, score_scale=score_scale)
        print(f"NO checkpoint — RANDOM net (SMOKE TEST; regret is meaningless). "
              f"margin_gain={margin_gain} alpha/B={alpha}")
    return net.eval(), margin_gain, alpha, score_scale


def collect_won_endgames(args, margin_gain, alpha, score_scale):
    """Yield (state, mover, root_mover_value) for won no-chance endgame positions."""
    rng = random.Random(args.seed)
    found = 0
    seed = args.seed
    while found < args.positions and seed < args.seed + args.max_seed_scan:
        st = GameState.new(seed=seed)
        seed += 1
        r = random.Random(seed * 7919)
        while st.phase != Phase.GAME_OVER:
            if _is_solvable_endgame(st) and len(st.legal_actions()) >= 2:
                v_p0, solved = exact_endgame_value(
                    st, max_secs=args.solve_secs, rng=rng,
                    score_scale=score_scale, margin_gain=margin_gain, alpha=alpha)
                if solved:
                    mover = st.current_actor
                    v_mover = _mover_frame(v_p0, mover)
                    if v_mover >= args.win_threshold:
                        yield st, mover, v_mover
                        found += 1
                break  # one position per game (the first solvable node)
            st = st.step(r.choice(st.legal_actions()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="", help="333-flat .pt; omit for smoke test")
    p.add_argument("--positions", type=int, default=60)
    p.add_argument("--sims", type=int, default=400, help="MCTS simulations per position")
    p.add_argument("--win_threshold", type=float, default=0.55,
                   help="min exact mover-frame value to count a position as WON")
    p.add_argument("--regret_eps", type=float, default=0.02,
                   help="value regret above this counts as a lazy (margin-losing) move")
    p.add_argument("--solve_secs", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--max_seed_scan", type=int, default=20000)
    # arch fallbacks (used only when a checkpoint omits them / smoke test)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--bilinear_dim", type=int, default=32)
    p.add_argument("--score_scale", type=float, default=160.0)
    p.add_argument("--margin_gain", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.5)
    args = p.parse_args()

    net, margin_gain, alpha, score_scale = build_net(args)
    evaluator = make_serial_evaluator(net, margin_gain=margin_gain, alpha=alpha)

    regrets = []       # value units, mover frame, >= 0
    lost_the_win = 0   # chosen move actually crossed from win to non-win
    print(f"\nscanning for {args.positions} won endgame positions "
          f"(win_threshold={args.win_threshold})...")
    for i, (st, mover, root_v) in enumerate(
            collect_won_endgames(args, margin_gain, alpha, score_scale)):
        mcts = OpenLoopMCTS(evaluator, n_simulations=args.sims,
                            margin_gain=margin_gain, alpha=alpha)
        vc, _ = mcts.search(st, rng=np.random.default_rng(args.seed + i))
        chosen = select_move(vc, temperature=0.0, rng=np.random.default_rng(i))
        child = st.step(chosen)
        cv_p0, solved = exact_endgame_value(
            child, max_secs=args.solve_secs, rng=random.Random(i),
            score_scale=score_scale, margin_gain=margin_gain, alpha=alpha)
        if not solved:
            continue
        chosen_v = _mover_frame(cv_p0, mover)
        regret = max(0.0, root_v - chosen_v)   # >=0 by minimax
        regrets.append(regret)
        if chosen_v <= 0.5 < root_v:           # gave back the win entirely
            lost_the_win += 1

    if not regrets:
        print("no solvable won positions found — widen --max_seed_scan or lower "
              "--win_threshold.")
        return

    regrets = np.array(regrets)
    lazy = regrets > args.regret_eps
    pts = np.array([_value_to_points(r, alpha, margin_gain, score_scale) for r in regrets])
    print("\n" + "=" * 70)
    print(f"MARGIN-REGRET CANARY  ({len(regrets)} won positions, {args.sims} sims)")
    print("=" * 70)
    print(f"lazy-play rate (regret > {args.regret_eps}):  "
          f"{100*lazy.mean():.1f}%  ({lazy.sum()}/{len(regrets)})")
    print(f"mean regret:   {regrets.mean():.4f} value  (~{pts.mean():.1f} pts)")
    print(f"median regret: {np.median(regrets):.4f} value  (~{np.median(pts):.1f} pts)")
    print(f"p90 regret:    {np.percentile(regrets,90):.4f} value  "
          f"(~{np.percentile(pts,90):.1f} pts)")
    print(f"max regret:    {regrets.max():.4f} value  (~{pts.max():.1f} pts)")
    print(f"gave back the win entirely: {lost_the_win}")
    print("\nHealthy net: low lazy-rate and small regret — search walks into the "
          "highest-margin winning line. A high lazy-rate says raise B, sharpen the "
          "gate (n=6), or check score-head calibration.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
