# evaluate.py
# Evaluates the trained neural network against other players.

import os
import sys
import torch
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, get_valid_moves, apply_move,
    stop_turn, bust_turn
)
from games.cantstop.features import (
    extract_features, get_legal_action_mask,
    action_to_move_decision, move_to_action
)
from games.cantstop.model import CantStopNet
from games.cantstop.ev_player import ev_player, run_tournament, play_game
from games.cantstop.mc_player import mc_player


# ---- NEURAL NETWORK PLAYER ----

def nn_player(state, model, device='cuda'):
    """
    Makes decisions using the trained neural network.
    Uses policy head for move selection.
    Uses value head to decide stop vs continue.
    """
    valid = get_valid_moves(state)
    if not valid:
        return None, "bust"

    model.eval()
    with torch.no_grad():
        features = extract_features(state, valid)
        features_t = torch.tensor(features, dtype=torch.float32)\
                         .unsqueeze(0).to(device)
        mask = get_legal_action_mask(valid)
        mask_t = torch.tensor(mask, dtype=torch.bool)\
                     .unsqueeze(0).to(device)

        value, logits = model(features_t, mask_t)

        import torch.nn.functional as F
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu()

        # Pick highest probability legal action
        best_action = probs.argmax().item()
        move, decision = action_to_move_decision(best_action)

    return move, decision


# ---- LOAD MODEL ----

def load_model(path, device='cuda'):
    """Load a trained model from checkpoint."""
    model = CantStopNet().to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Loaded model from {path}")
    print(f"  Trained epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        if isinstance(metrics, dict) and 'val' in metrics:
            print(f"  Val loss: {metrics['val']['loss']:.4f}")
            print(f"  Policy acc: {metrics['val']['policy_acc']:.3f}")
    return model


# ---- TOURNAMENT ----

def run_nn_tournament(model, opponent_fn, opponent_name,
                      games=1000, device='cuda'):
    """
    Run a tournament between the neural network and an opponent.
    Alternates who goes first to remove first-mover advantage.
    """
    nn_fn = lambda s: nn_player(s, model, device)

    wins = {0: 0, 1: 0, None: 0}

    for i in range(games):
        if i % 100 == 0:
            print(f"\r  Game {i}/{games}...", end="", flush=True)

        if random.random() < 0.5:
            # NN plays as player 0
            winner = play_game(nn_fn, opponent_fn)
            if winner == 0:   wins[0] += 1
            elif winner == 1: wins[1] += 1
            else:             wins[None] += 1
        else:
            # NN plays as player 1
            winner = play_game(opponent_fn, nn_fn)
            if winner == 0:   wins[1] += 1
            elif winner == 1: wins[0] += 1
            else:             wins[None] += 1

    print()
    total = games - wins[None]
    nn_pct = 100 * wins[0] / total if total > 0 else 0

    print(f"\n{'='*45}")
    print(f"  Neural Network vs {opponent_name}")
    print(f"  {games:,} games played")
    print(f"{'='*45}")
    print(f"  Neural Network  {wins[0]:>5} wins  ({nn_pct:.1f}%)")
    print(f"  {opponent_name:<20} {wins[1]:>5} wins  ({100-nn_pct:.1f}%)")
    if wins[None]:
        print(f"  No result      {wins[None]:>5}")

    return wins


# ---- MAIN ----

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="models/cantstop/best_model.pt")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nEvaluating neural network")
    print(f"Device: {device}")
    print(f"Model:  {args.model}\n")

    model = load_model(args.model, device)

    # Tournament 1: NN vs Random
    rand_fn = lambda s: (random.choice(get_valid_moves(s))
                         if get_valid_moves(s) else None,
                         random.choice(["stop", "continue"]))

    run_nn_tournament(model, rand_fn, "Random", args.games, device)

    # Tournament 2: NN vs EV Player
    ev_fn = lambda s: ev_player(s, use_corrected=False)
    run_nn_tournament(model, ev_fn, "EV Player", args.games, device)

    print("\nDone!")