"""
play_vs_model.py
Play Can't Stop against a trained model from the terminal.

Goal: a calibration tool. Automated evals tell you whether the model
beats other models. They don't tell you how the model plays at an
absolute level. Playing a few games yourself does.

Usage (from repo root):

    python -m games.cantstop.scripts.play_vs_model \\
        --model models/cantstop/self_play/model_iter_009_accepted.pt \\
        --sims 200

Optional flags:
    --sims N         MCTS simulations per AI move. Default 200. Higher
                     = stronger AI, slower per move. 50 is fast and
                     weak; 800 is slow and strong.
    --human-goes-first    Make yourself player 0. Default is alternating.
    --device cpu|cuda     Device for the model. Default cpu (fine for
                          one-game-at-a-time play).
    --show-ai-thinking    Print the AI's value estimate and top moves
                          before it plays. Useful for understanding
                          what the AI thinks the position is worth.
"""

import os
import sys
import argparse
import random

# Path: scripts/ -> cantstop/ -> games/ -> project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from games.cantstop.engine import (
    GameState, COLUMN_HEIGHTS, COLUMNS_TO_WIN,
    get_valid_moves, apply_move, stop_turn, bust_turn,
)
from games.cantstop.features import action_to_move_decision
from games.cantstop.evaluate import load_model
from games.cantstop.mcts import MCTS


# ============================================================
# Board rendering
# ============================================================

def render_board(state, human_player):
    """Pretty-print the current board state from the human's perspective."""
    me = human_player
    ai = 1 - me

    # Column heights (2-12). The Can't Stop board:
    #   2:3  3:5  4:7  5:9  6:11  7:13  8:11  9:9  10:7  11:5  12:3
    print()
    print("=" * 72)
    print(f"  Active player: {'YOU' if state.active_player == me else 'AI'}    "
          f"Dice: {state.dice if state.dice else '(not rolled)'}")
    if state.runners:
        print(f"  Current-turn runners: " +
              ", ".join(f"col {c}@{state.runners[c]}"
                        for c in sorted(state.runners)))
    print()
    print("  Col:       " + "  ".join(f"{c:>2}" for c in sorted(COLUMN_HEIGHTS)))
    print("  Height:    " + "  ".join(f"{COLUMN_HEIGHTS[c]:>2}"
                                       for c in sorted(COLUMN_HEIGHTS)))
    print("  Your prog: " + "  ".join(
        _format_progress(state, me, c) for c in sorted(COLUMN_HEIGHTS)))
    print("  AI prog:   " + "  ".join(
        _format_progress(state, ai, c) for c in sorted(COLUMN_HEIGHTS)))
    print()
    print(f"  Your claimed columns:  {sorted(state.claimed[me])} "
          f"({len(state.claimed[me])}/{COLUMNS_TO_WIN})")
    print(f"  AI claimed columns:    {sorted(state.claimed[ai])} "
          f"({len(state.claimed[ai])}/{COLUMNS_TO_WIN})")
    print("=" * 72)


def _format_progress(state, player, col):
    """Format the progress for one column."""
    if col in state.claimed[player]:
        return " ✓"
    p = state.progress[player].get(col, 0)
    return f"{p:>2}" if p > 0 else " ."


# ============================================================
# Human move selection
# ============================================================

def choose_move_human(state, valid_moves):
    """
    Prompt the human to pick a move. Returns the move tuple.

    Move tuples look like:
      (6, 8)   advance two columns by 1 each
      (7, 7)   advance column 7 by 2
      (6,)     partial — advance just one column by 1 (other not playable)
    """
    # Sort for stable display order.
    sorted_moves = sorted(valid_moves, key=lambda m: (len(m), m))

    print("\n  Your valid moves:")
    for i, mv in enumerate(sorted_moves):
        print(f"    [{i}] {_format_move(mv)}")

    while True:
        choice = input("\n  Pick move number: ").strip()
        try:
            idx = int(choice)
            if 0 <= idx < len(sorted_moves):
                return sorted_moves[idx]
        except ValueError:
            pass
        print(f"    Invalid. Enter a number 0..{len(sorted_moves)-1}.")


def choose_stop_or_continue_human(state):
    """
    Prompt the human to decide STOP (lock in runners) or CONTINUE
    (risk a bust to advance more). Only called when stop is legal —
    i.e., the player has at least one runner.
    """
    print("\n  After this move:")
    print(f"    Current-turn runners: " +
          ", ".join(f"col {c}@{state.runners[c]}"
                    for c in sorted(state.runners)))
    print("    [s] STOP    — save these runner positions, end your turn")
    print("    [c] CONTINUE — keep rolling, but lose everything if you bust")
    while True:
        ch = input("  Stop or continue? [s/c]: ").strip().lower()
        if ch in ('s', 'stop'):
            return 'stop'
        if ch in ('c', 'continue'):
            return 'continue'
        print("    Enter 's' or 'c'.")


def _format_move(mv):
    """Human-readable move description."""
    if len(mv) == 1:
        return f"advance column {mv[0]} by 1 (partial — only one column playable)"
    if mv[0] == mv[1]:
        return f"advance column {mv[0]} by 2 (true double)"
    return f"advance columns {mv[0]} and {mv[1]} by 1 each"


# ============================================================
# AI move selection
# ============================================================

def choose_action_ai(mcts, state, num_simulations, show_thinking=False,
                     selection='visits', debug_actions=8):
    """Have the AI pick (move, decision) via MCTS."""
    action_idx, move, decision, mcts_policy, mcts_value = mcts.get_action(
        state,
        num_simulations=num_simulations,
        temperature=0.0,   # greedy — always pick the most-visited action
        # Human play should use deployment-style search, not self-play
        # exploration. Disable root Dirichlet noise.
        dirichlet_epsilon=0.0,
    )

    root_stats = getattr(mcts, 'last_root_stats', None)
    if selection == 'value' and root_stats and root_stats.get('actions'):
        # Diagnostic/deployment alternative: choose max root child Q rather
        # than robust child. This is useful when low-sim PUCT exploration makes
        # a risky action get more visits even though its estimated value is not
        # best. Training should still use visit-count policies.
        best = max(root_stats['actions'], key=lambda row: (row['Q'], row['N']))
        action_idx = int(best['action_idx'])
        move, decision = action_to_move_decision(action_idx)

    if show_thinking:
        # mcts_value is from the active player's perspective.
        # Higher = AI thinks it's winning. ~0.5 = even. ~0 = losing.
        print(f"\n  AI value estimate: {mcts_value:.3f} "
              f"(>0.5 = AI ahead, <0.5 = AI behind)")
        if root_stats:
            print(f"  Root: N={root_stats.get('root_N')} "
                  f"Q={root_stats.get('root_Q', 0.0):.3f} "
                  f"NN={root_stats.get('root_value_initial', 0.0):.3f}")

        print("  AI considered actions:")
        if root_stats and root_stats.get('actions'):
            for row in root_stats['actions'][:max(1, int(debug_actions))]:
                mv, dec = action_to_move_decision(int(row['action_idx']))
                chosen = " <-- chosen" if int(row['action_idx']) == action_idx else ""
                print(f"    {row['visit_frac']*100:5.1f}% "
                      f"N={row['N']:4d} Q={row['Q']:.3f} "
                      f"P={row['prior']:.3f} — {_format_move(mv)} "
                      f"+ {dec.upper()}{chosen}")
        else:
            top_indices = mcts_policy.argsort()[-3:][::-1]
            for ai_idx in top_indices:
                if mcts_policy[ai_idx] > 0:
                    mv, dec = action_to_move_decision(int(ai_idx))
                    print(f"    {mcts_policy[ai_idx]*100:5.1f}% — "
                          f"{_format_move(mv)} + {dec.upper()}")

    return move, decision


# ============================================================
# Game loop
# ============================================================

def play_one_game(model_path, num_simulations, human_first, device,
                  show_thinking, mcts_kwargs=None, selection='visits',
                  debug_actions=8):
    """Play a single game. Returns winner (0 = human wins by default,
    or whichever player number the human played as)."""
    model = load_model(model_path, device)
    mcts = MCTS(model, device, **(mcts_kwargs or {}))

    state = GameState(2)
    human_player = 0 if human_first else 1
    ai_player = 1 - human_player

    print(f"\n{'#' * 72}")
    print(f"# NEW GAME")
    print(f"# You are Player {human_player}.  AI is Player {ai_player}.")
    print(f"# AI strength: {num_simulations} MCTS simulations per move.")
    print(f"# Model: {model_path}")
    print(f"{'#' * 72}")

    turn_number = 0
    while not state.game_over:
        turn_number += 1
        if turn_number > 500:  # safety
            print("\n  Game ran too long, calling it a draw.")
            return None

        if not state.dice:
            state.roll_dice()

        valid = get_valid_moves(state)

        if not valid:
            # Bust — no legal move on this roll.
            active_name = "YOU" if state.active_player == human_player else "AI"
            print(f"\n  >> {active_name} BUSTED on dice {state.dice}. "
                  f"Lost all current-turn progress. <<")
            if state.active_player == human_player:
                # Show what the human had before busting.
                if state.runners:
                    print(f"     (lost runners: " +
                          ", ".join(f"col {c}@{state.runners[c]}"
                                    for c in sorted(state.runners)) + ")")
            bust_turn(state)
            state.dice = []
            input("  Press Enter to continue...")
            continue

        # ---- Active player's turn ----
        if state.active_player == human_player:
            render_board(state, human_player)
            move = choose_move_human(state, valid)
            apply_move(state, move)

            # If player has runners and it's not game-ending, they choose
            # whether to stop or continue. apply_move can win the game if
            # this move claimed the 3rd column for them — check that.
            if state.game_over:
                decision = 'stop'  # forced; game's already over
            elif state.runners:
                decision = choose_stop_or_continue_human(state)
            else:
                # No runners means no stop decision needed; treat as
                # "stop" so play passes. (This branch is unusual but
                # defensive.)
                decision = 'stop'

            if decision == 'stop':
                stop_turn(state)
                state.dice = []
                print("\n  You stopped. Turn passes to AI.")
            else:
                state.dice = []  # continue: spend dice, roll fresh next loop

        else:
            # AI's turn. Run search and apply.
            render_board(state, human_player)
            print("  AI is thinking...")
            move, decision = choose_action_ai(
                mcts, state, num_simulations, show_thinking,
                selection=selection, debug_actions=debug_actions)
            print(f"  AI plays: {_format_move(move)} + {decision.upper()}")
            apply_move(state, move)

            if decision == 'stop':
                stop_turn(state)
                state.dice = []
            else:
                state.dice = []

            input("  Press Enter to continue...")

    render_board(state, human_player)
    if state.winner == human_player:
        print("\n  🎉 YOU WIN! 🎉")
    else:
        print("\n  AI wins. Better luck next game.")
    return state.winner


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Play Can't Stop against a trained model"
    )
    parser.add_argument('--model', required=True,
                        help='Path to model .pt checkpoint.')
    parser.add_argument('--sims', type=int, default=200,
                        help='MCTS simulations per AI move (default 200). '
                             '50 = weak/fast, 200 = balanced, 800 = strong/slow.')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='Device for the model (default cpu).')
    parser.add_argument('--human-goes-first', action='store_true',
                        dest='human_first',
                        help='Always have you play as P0. Default alternates.')
    parser.add_argument('--show-ai-thinking', action='store_true',
                        dest='show_thinking',
                        help='Print AI value estimate and top moves each turn.')
    parser.add_argument('--chance-mode', type=str, default='sampled',
                        choices=['sampled', 'hybrid-exact', 'exact-all'],
                        dest='chance_mode',
                        help='Chance-node handling mode for the AI search.')
    parser.add_argument('--exact-bust-threshold', type=float, default=0.05,
                        dest='exact_bust_threshold',
                        help='Hybrid exact-chance gate: enable exact chance '
                             'when bust probability is at least this value.')
    parser.add_argument('--exact-max-outcomes', type=int, default=32,
                        dest='exact_max_outcomes',
                        help='Maximum canonical outcome count eligible for '
                             'exact chance.')
    parser.add_argument('--exact-cache-size', type=int, default=4096,
                        dest='exact_cache_size',
                        help='LRU cache entries for exact chance outcome '
                             'enumeration. Set 0 to disable.')
    parser.add_argument('--target-inflight', type=int, default=1,
                        dest='target_inflight',
                        help='MCTS concurrency for interactive play. Default 1 '
                             'uses fully synchronous search, which is the safest '
                             'quality setting.')
    parser.add_argument('--warmup-sims', type=int, default=0,
                        dest='warmup_sims',
                        help='Sequential warmup simulations before async mode. '
                             'Ignored when --target-inflight 1.')
    parser.add_argument('--selection', type=str, default='visits',
                        choices=['visits', 'value'],
                        help='Interactive AI action selection. visits = robust '
                             'child / AlphaZero-style. value = max root child Q '
                             'for diagnosing risky push/stop mistakes.')
    parser.add_argument('--debug-actions', type=int, default=8,
                        dest='debug_actions',
                        help='When --show-ai-thinking is set, print this many '
                             'root actions with visit share, Q, prior, and N.')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        parser.error(f"Model file not found: {args.model}")

    print(f"\nLoading model from {args.model}...")
    print(f"(One-time setup — game will start in a moment.)")

    # Track results across multiple games.
    games_played = 0
    human_wins = 0
    ai_wins = 0

    # Alternate who goes first across games (unless --human-goes-first).
    game_idx = 0
    while True:
        if args.human_first:
            human_first = True
        else:
            human_first = (game_idx % 2 == 0)

        winner = play_one_game(
            model_path=args.model,
            num_simulations=args.sims,
            human_first=human_first,
            device=args.device,
            show_thinking=args.show_thinking,
            selection=args.selection,
            debug_actions=args.debug_actions,
            mcts_kwargs=dict(
                chance_mode=args.chance_mode,
                exact_bust_threshold=args.exact_bust_threshold,
                exact_max_outcomes=args.exact_max_outcomes,
                exact_cache_size=args.exact_cache_size,
                target_inflight=args.target_inflight,
                warmup_sims=args.warmup_sims,
            ),
        )

        games_played += 1
        human_player_this_game = 0 if human_first else 1
        if winner == human_player_this_game:
            human_wins += 1
        elif winner is not None:
            ai_wins += 1

        print(f"\n  Series score: YOU {human_wins} – {ai_wins} AI "
              f"({games_played} games)")

        again = input("\n  Play another game? [y/N]: ").strip().lower()
        if again not in ('y', 'yes'):
            break
        game_idx += 1

    print(f"\n  Final score: YOU {human_wins} – {ai_wins} AI "
          f"across {games_played} games.")
    if games_played > 0:
        wr = human_wins / games_played
        print(f"  Your win rate: {wr*100:.0f}%")
        if games_played < 5:
            print(f"  (Small sample — try 10+ games to get a real read.)")


if __name__ == "__main__":
    main()