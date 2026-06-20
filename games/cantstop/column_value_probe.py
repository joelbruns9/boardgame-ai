# games/cantstop/column_value_probe.py
#
# Probe the value head's per-column preference and check it against the truth.
#
# What it does
# ------------
# Builds minimal positions that differ only in:
#   (a) which column the active player has one step of progress on (swept 2..12)
#   (b) the score context: even / ahead / behind
# then reads the RAW value head (averaged over many dice rolls to wash out the
# current-roll noise), and optionally plays each position to completion to get
# the empirical win rate under the model's own policy.
#
# It answers the two questions from our discussion in one shot:
#   1. The shape of the value-head column curve, and whether it changes with the
#      score — does the model's love of 2/12 cool off when it's ahead (calibrated,
#      variance-when-behind) or persist (the same red flag as pushing while ahead)?
#   2. Whether the head OVER-rates any column relative to actual rollout outcomes
#      (the suspected 2/12 overvaluation that would feed the aggression).
#
# The value head and rollouts are both from the ACTIVE player's perspective
# (P(active player wins)), so they are directly comparable.
#
# Run from the repo root, same pattern as the advisor:
#   python -m games.cantstop.column_value_probe \
#       --model models/cantstop/self_play_60h_s1200/best_model.pt \
#       --rollouts 300
#
# Outputs a printed table, a JSON dump, and (if matplotlib is present) a PNG plot.

import os
import sys
import json
import math
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, COLUMN_HEIGHTS,
    get_valid_moves, apply_move, stop_turn, bust_turn,
)
from games.cantstop.features import (
    extract_features, get_legal_action_mask, action_to_move_decision,
)
from games.cantstop.evaluate import load_model, nn_player

COLUMNS = list(range(2, 13))
CONTEXTS = ('even', 'ahead', 'behind')


# ----------------------------------------------------------------------------
# Position construction
# ----------------------------------------------------------------------------

def build_state(column, steps, context, claim_cols, use_runner):
    """
    Active player (index 0) has `steps` of progress on `column`, nothing else.
    Score context is set by giving claimed columns to player 0 (ahead) or
    player 1 (behind). Dice are left empty; callers set them as needed.
    Returns None if the probe column collides with a claimed column.
    """
    if context in ('ahead', 'behind') and column in claim_cols:
        return None  # can't have progress on a claimed column

    s = GameState(num_players=2)
    s.active_player = 0

    steps = max(1, min(steps, COLUMN_HEIGHTS[column] - 1))  # keep it un-claimed
    if use_runner:
        s.runners = {column: steps}
    else:
        s.progress[0] = {column: steps}

    if context == 'ahead':
        s.claimed[0] = set(claim_cols)
        s.all_claimed = set(claim_cols)
    elif context == 'behind':
        s.claimed[1] = set(claim_cols)
        s.all_claimed = set(claim_cols)

    return s


# ----------------------------------------------------------------------------
# Raw value head
# ----------------------------------------------------------------------------

@torch.no_grad()
def value_head(model, state, device):
    """
    Raw value-head P(active player wins) for a state that already has dice.
    The value head is independent of the action mask, so on bust rolls (no legal
    move) we substitute an all-true mask purely to keep the policy path
    well-defined — the returned value is unchanged either way.
    """
    valid = get_valid_moves(state)
    mask_arr = get_legal_action_mask(valid)
    if not mask_arr.any():
        mask_arr = np.ones_like(mask_arr)
    feats = torch.tensor(extract_features(state, valid),
                         dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.tensor(mask_arr, dtype=torch.bool).unsqueeze(0).to(device)
    value, _logits = model(feats, mask)
    return float(value.item())


def dice_avg_value(model, base_state, device, n_rolls, rng):
    """
    Average the value head over `n_rolls` uniform 4d6 rolls. This marginalizes
    out the current roll so we read the *position's* value, not one roll's.
    Bust rolls (no legal move) are included — they're part of the position value.
    """
    vals = []
    for _ in range(n_rolls):
        s = base_state.clone()
        s.dice = list(rng.choices((1, 2, 3, 4, 5, 6), k=4))
        vals.append(value_head(model, s, device))
    return float(np.mean(vals)), float(np.std(vals))


# ----------------------------------------------------------------------------
# Rollout ground truth
# ----------------------------------------------------------------------------

def play_to_end(start_state, model, device, cap):
    """
    Play `start_state` to completion with the model driving both players.
    Mirrors the engine's reference game loop (re-rolls fresh dice each step),
    so the model's own policy decides every move + stop/continue.
    Returns the winning player index, or None if the game hit the step cap.
    """
    s = start_state.clone()
    s.dice = []  # let the loop roll the first decision's dice
    steps = 0
    while not s.game_over and steps < cap:
        steps += 1
        s.roll_dice()
        valid = get_valid_moves(s)
        if not valid:
            bust_turn(s)
            continue
        move, decision = nn_player(s, model, device)
        if move is None:
            bust_turn(s)
            continue
        apply_move(s, move)
        if decision == 'stop':
            stop_turn(s)
    return s.winner


def rollout_winrate(base_state, model, device, n_games, cap):
    """Empirical P(active player of base_state wins), over n_games playouts."""
    me = base_state.active_player
    wins = 0
    decided = 0
    for _ in range(n_games):
        w = play_to_end(base_state, model, device, cap)
        if w is None:
            continue
        decided += 1
        if w == me:
            wins += 1
    if decided == 0:
        return float('nan'), float('nan'), 0
    p = wins / decided
    # 95% binomial confidence half-width.
    ci = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / decided)
    return p, ci, decided


# ----------------------------------------------------------------------------
# Policy-head column scan
# ----------------------------------------------------------------------------
# Reads the runner-placement preference DIRECTLY from the policy head, separate
# from the value head. From a neutral board (no progress/runners; score set by
# claimed columns), sample many rolls and, for each column, measure the average
# total policy mass placed on moves that advance it GIVEN it was on offer — the
# column's "grab rate". Availability (how often a column is offerable) is tracked
# separately so high-frequency columns don't look preferred just for showing up.

@torch.no_grad()
def policy_probs(model, state, valid, device):
    """Full policy distribution over the action space (illegal ~0), as nn_player sees it."""
    feats = torch.tensor(extract_features(state, valid),
                         dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.tensor(get_legal_action_mask(valid),
                        dtype=torch.bool).unsqueeze(0).to(device)
    _value, logits = model(feats, mask)
    return F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()


def policy_column_scan(model, context, claim_cols, device, n_rolls, rng):
    """Per-column {avail_frac, grab_rate} from a neutral board in `context`."""
    avail = {c: 0 for c in COLUMNS}
    grab = {c: 0.0 for c in COLUMNS}
    used = 0
    for _ in range(n_rolls):
        s = GameState(num_players=2)
        s.active_player = 0
        if context == 'ahead':
            s.claimed[0] = set(claim_cols); s.all_claimed = set(claim_cols)
        elif context == 'behind':
            s.claimed[1] = set(claim_cols); s.all_claimed = set(claim_cols)
        s.dice = list(rng.choices((1, 2, 3, 4, 5, 6), k=4))

        valid = get_valid_moves(s)
        if not valid:
            continue
        used += 1

        offered = set()
        for mv in valid:
            offered.update(mv)
        for c in offered:
            if c in avail:
                avail[c] += 1

        probs = policy_probs(model, s, valid, device)
        for a in np.nonzero(probs > 1e-6)[0]:
            mv, _dec = action_to_move_decision(int(a))
            pa = float(probs[a])
            for c in set(mv):   # dedup: a (7,7) double touches column 7 once
                if c in grab:
                    grab[c] += pa

    out = {}
    for c in COLUMNS:
        out[c] = {
            'avail_frac': (avail[c] / used) if used else float('nan'),
            'grab_rate': (grab[c] / avail[c]) if avail[c] else float('nan'),
        }
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Value-head column-preference probe.")
    ap.add_argument('--model', required=True, help='Path to model checkpoint (.pt).')
    ap.add_argument('--device', default=None, help='cuda / cpu (default: auto).')
    ap.add_argument('--dice-samples', type=int, default=64,
                    help='Dice rolls to average each value-head read over.')
    ap.add_argument('--steps', type=int, default=1,
                    help='Steps of progress on the probe column (default 1).')
    ap.add_argument('--probe', choices=('saved', 'runner'), default='saved',
                    help="Probe banked progress ('saved') or a current-turn "
                         "runner ('runner'). Default: saved.")
    ap.add_argument('--claim-cols', type=int, nargs=2, default=(4, 10),
                    metavar=('C1', 'C2'),
                    help='Columns claimed to create the ahead/behind contexts. '
                         'Pick ambivalent ones; those columns are skipped in '
                         'those contexts. Default: 4 10.')
    ap.add_argument('--rollouts', type=int, default=200,
                    help='Playouts per position for the ground-truth win rate '
                         '(0 disables rollouts). Default 200.')
    ap.add_argument('--rollout-contexts', default='even',
                    help="Comma list of contexts to roll out, or 'all'. "
                         "Default: even (cheapest, answers overvaluation).")
    ap.add_argument('--rollout-cap', type=int, default=400,
                    help='Max rolls per playout before calling it undecided.')
    ap.add_argument('--policy-scan', type=int, default=2000, metavar='N',
                    help='Rolls per context for the policy-head column scan '
                         '(0 disables). Reads runner-placement preference '
                         'directly from the policy. Default 2000.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='column_value_probe',
                    help='Output basename for the .json and .png.')
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    claim_cols = tuple(args.claim_cols)
    use_runner = (args.probe == 'runner')
    if args.rollout_contexts.strip().lower() == 'all':
        rollout_contexts = set(CONTEXTS)
    else:
        rollout_contexts = {c.strip() for c in args.rollout_contexts.split(',') if c.strip()}

    print(f"Device: {device}")
    model = load_model(args.model, device)
    print(f"Probe: {args.probe} progress = {args.steps} step(s) on each column")
    print(f"Score contexts via claimed columns {claim_cols} "
          f"(skipped in ahead/behind)\n")

    val_rng = random.Random(args.seed)        # for dice-averaging
    random.seed(args.seed)                     # for rollouts (engine uses module random)

    results = {c: {} for c in CONTEXTS}

    for context in CONTEXTS:
        do_rollout = (args.rollouts > 0 and context in rollout_contexts)
        tag = f"[{context}] value-head sweep"
        if do_rollout:
            tag += f" + {args.rollouts} rollouts/col (slow part)"
        print(tag + " ...", flush=True)
        for col in COLUMNS:
            base = build_state(col, args.steps, context, claim_cols, use_runner)
            if base is None:
                results[context][col] = None
                print(f"    col {col:>2}: (claimed — skipped)", flush=True)
                continue
            v_mean, v_std = dice_avg_value(model, base, device,
                                           args.dice_samples, val_rng)
            entry = {'value_head': v_mean, 'value_head_std': v_std}
            if do_rollout:
                p, ci, n = rollout_winrate(base, model, device,
                                           args.rollouts, args.rollout_cap)
                entry.update({'rollout_winrate': p, 'rollout_ci95': ci,
                              'rollout_games': n})
                print(f"    col {col:>2}: head {v_mean:.3f}  |  "
                      f"rollout {p:.3f} ±{ci:.3f} (n={n})", flush=True)
            else:
                print(f"    col {col:>2}: head {v_mean:.3f}", flush=True)
            results[context][col] = entry

    # ---- Policy-head column scan ----
    policy_results = {}
    if args.policy_scan > 0:
        pol_rng = random.Random(args.seed + 1)
        for context in CONTEXTS:
            print(f"[{context}] policy scan ({args.policy_scan} rolls) ...", flush=True)
            policy_results[context] = policy_column_scan(
                model, context, claim_cols, device, args.policy_scan, pol_rng)

    # ---- Print table ----
    print("=" * 72)
    hdr = f"{'col':>3} {'h':>3} | " + " | ".join(f"{c:>16}" for c in CONTEXTS)
    print(hdr)
    print(f"{'':>3} {'':>3} | " + " | ".join(f"{'head    rollout':>16}" for _ in CONTEXTS))
    print("-" * 72)
    for col in COLUMNS:
        cells = []
        for context in CONTEXTS:
            e = results[context][col]
            if e is None:
                cells.append(f"{'(claimed)':>16}")
            elif 'rollout_winrate' in e and not math.isnan(e['rollout_winrate']):
                cells.append(f"{e['value_head']:.3f} {e['rollout_winrate']:.3f}±{e['rollout_ci95']:.2f}".ljust(16))
            else:
                cells.append(f"{e['value_head']:.3f}{'':>11}")
        print(f"{col:>3} {COLUMN_HEIGHTS[col]:>3} | " + " | ".join(cells))
    print("=" * 72)
    print("head = dice-averaged value head;  rollout = empirical win rate (±95% CI)")
    print("Reading guide: a head value well ABOVE its rollout = overvaluation.")
    print("Compare the head curve's 2/12 bump across contexts: shrinks when")
    print("ahead = calibrated; persists = the same red flag as pushing while ahead.")

    # ---- Print policy table ----
    if policy_results:
        print("\n" + "=" * 72)
        print("POLICY column scan — when a column is offered, avg policy mass on")
        print("moves advancing it (the 'grab rate'). 'offered' = how often it's playable.")
        print("-" * 72)
        print(f"{'col':>3} {'h':>3} {'offered':>8} | "
              + " | ".join(f"grab {c:>6}" for c in CONTEXTS))
        for col in COLUMNS:
            off = policy_results['even'][col]['avail_frac']
            off_s = f"{off*100:>6.0f}%" if not math.isnan(off) else f"{'--':>7}"
            cells = []
            for context in CONTEXTS:
                g = policy_results[context][col]['grab_rate']
                cells.append(f"{'(claimed)':>11}" if math.isnan(g) else f"{g:>11.3f}")
            print(f"{col:>3} {COLUMN_HEIGHTS[col]:>3} {off_s:>8} | " + " | ".join(cells))
        print("=" * 72)
        print("High grab on 6/7/8 = the runner-placement love, now read straight from")
        print("the policy. Watch whether it cools (toward even grab) when ahead.")

    # ---- Dump JSON ----
    out_json = f"{args.out}.json"
    with open(out_json, 'w') as f:
        json.dump({
            'model': args.model, 'probe': args.probe, 'steps': args.steps,
            'claim_cols': list(claim_cols), 'dice_samples': args.dice_samples,
            'rollouts': args.rollouts, 'policy_scan': args.policy_scan,
            'results': results, 'policy_results': policy_results,
        }, f, indent=2)
    print(f"\nWrote {out_json}")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5.5))
        styles = {'even': ('-', 'o'), 'ahead': ('--', 's'), 'behind': (':', '^')}
        for context in CONTEXTS:
            xs, ys = [], []
            for col in COLUMNS:
                e = results[context][col]
                if e is None:
                    continue
                xs.append(col)
                ys.append(e['value_head'])
            ls, mk = styles[context]
            ax.plot(xs, ys, ls, marker=mk, label=f'value head ({context})')

        # Rollout ground truth (with CI bars) for whichever contexts have it.
        for context in CONTEXTS:
            xs, ys, es = [], [], []
            for col in COLUMNS:
                e = results[context][col]
                if e is None or 'rollout_winrate' not in e:
                    continue
                if math.isnan(e['rollout_winrate']):
                    continue
                xs.append(col)
                ys.append(e['rollout_winrate'])
                es.append(e['rollout_ci95'])
            if xs:
                ax.errorbar(xs, ys, yerr=es, fmt='x', capsize=3,
                            label=f'rollout truth ({context})', color='black', alpha=0.7)

        ax.set_xlabel('column')
        ax.set_ylabel('P(active player wins)')
        ax.set_title(f"Value-head column preference vs. rollout truth "
                     f"({args.probe} +{args.steps})")
        ax.set_xticks(COLUMNS)
        ax.axhline(0.5, color='grey', lw=0.6)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        out_png = f"{args.out}.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=130)
        print(f"Wrote {out_png}")

        if policy_results:
            fig2, ax2 = plt.subplots(figsize=(9, 5.5))
            for context in CONTEXTS:
                xs, ys = [], []
                for col in COLUMNS:
                    g = policy_results[context][col]['grab_rate']
                    if math.isnan(g):
                        continue
                    xs.append(col)
                    ys.append(g)
                ls, mk = styles[context]
                ax2.plot(xs, ys, ls, marker=mk, label=f'grab rate ({context})')
            xs = [c for c in COLUMNS
                  if not math.isnan(policy_results['even'][c]['avail_frac'])]
            ys = [policy_results['even'][c]['avail_frac'] for c in xs]
            ax2.plot(xs, ys, color='grey', lw=0.8, alpha=0.6, label='offered (even)')
            ax2.set_xlabel('column')
            ax2.set_ylabel('policy grab rate  /  availability')
            ax2.set_title('Policy-head column preference (grab rate when offered)')
            ax2.set_xticks(COLUMNS)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8)
            out_png2 = f"{args.out}_policy.png"
            fig2.tight_layout()
            fig2.savefig(out_png2, dpi=130)
            print(f"Wrote {out_png2}")
    except ImportError:
        print("(matplotlib not available — skipped plot; numbers are in the JSON.)")


if __name__ == "__main__":
    main()