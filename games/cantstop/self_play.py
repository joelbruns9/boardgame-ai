# self_play.py
# Self-play training loop for the Can't Stop neural network.
#
# Improvements over v1:
#   1. Replay buffer — prevents catastrophic forgetting
#   2. Larger evaluation (500 games) — reduces acceptance noise
#   3. Phase-based temperature — precision in endgame
#   4. Soft targets — KL divergence over full distributions
#
# Each iteration:
#   1. Generate games using current network (phase-based temperature)
#   2. Add to replay buffer
#   3. Train new model on sample from replay buffer (soft targets)
#   4. Evaluate new vs old (500 games, 55% threshold)
#   5. Accept or reject new model

import os
import sys
import json
import time
import random
import argparse
import numpy as np
from collections import deque
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, get_valid_moves, apply_move,
    stop_turn, bust_turn, COLUMN_HEIGHTS
)
from games.cantstop.features import (
    extract_features, get_legal_action_mask,
    move_to_action, action_to_move_decision,
    FEATURE_SIZE, ACTION_SPACE
)
from games.cantstop.model import CantStopNet, combined_loss
from games.cantstop.evaluate import load_model, nn_player
from games.cantstop.ev_player import play_game


# ---- PHASE-BASED TEMPERATURE ----

def get_temperature(step_index, state, global_temp_multiplier=1.0):
    """
    Dynamic temperature based on game phase.

    Early game:  high temp → explore widely
    Mid game:    medium temp → balanced
    Endgame:     low temp → precision

    global_temp_multiplier decays over iterations (1.2 → 0.8)
    to shift from exploration to exploitation as network matures.
    """
    max_score = max(
        len(state.claimed[0]),
        len(state.claimed[1])
    )

    if max_score >= 2:
        base_temp = 1.0   # was 0.1 — raised for better exploration
    elif max_score == 1 or step_index > 15:
        base_temp = 2.0   # was 0.5
    else:
        base_temp = 3.0   # was 1.2

    return base_temp * global_temp_multiplier


# ---- SOFT TARGET SAMPLING ----

def get_action_with_soft_target(model, state, valid_moves, device,
                                 temperature):
    """
    Run one forward pass, sample action with temperature,
    and return both the sampled action AND the full soft
    probability distribution for training.

    Returns:
        action_idx:  int — sampled action index
        move:        tuple — the move
        decision:    str — stop or continue
        soft_target: np.array shape (ACTION_SPACE,) — full distribution
    """
    features = extract_features(state, valid_moves)
    features_t = torch.tensor(features, dtype=torch.float32)\
                     .unsqueeze(0).to(device)

    mask = get_legal_action_mask(valid_moves)
    mask_t = torch.tensor(mask, dtype=torch.bool)\
                 .unsqueeze(0).to(device)

    with torch.no_grad():
        value, logits = model(features_t, mask_t)

        # Apply temperature to logits before softmax
        scaled_logits = logits.squeeze(0).clone()
        scaled_logits[~mask_t.squeeze(0)] = float('-inf')

        if temperature <= 0.01:
            # Effectively greedy
            action_idx = scaled_logits.argmax().item()
            # Still create soft target as near one-hot
            soft = torch.zeros(ACTION_SPACE)
            soft[action_idx] = 1.0
        else:
            scaled_logits_temp = scaled_logits / temperature
            probs = F.softmax(scaled_logits_temp, dim=-1)

            # Sample action
            action_idx = torch.multinomial(probs, num_samples=1).item()

            # Soft target = full probability distribution
            # This preserves uncertainty for training
            soft = probs.cpu()

        # Ensure soft target is normalized and legal only
        soft_np = soft.numpy().astype(np.float32)
        soft_np[~mask] = 0.0
        total = soft_np.sum()
        if total > 0:
            soft_np /= total

    move, decision = action_to_move_decision(action_idx)
    value_scalar = float(value.item())

    return action_idx, move, decision, soft_np, features, mask, value_scalar


# ---- SELF-PLAY GAME ----

def play_self_play_game(model, device, global_temp_multiplier=1.0):
    """
    Play one complete self-play game.
    Records full soft target distributions for each decision.

    Returns list of records with outcome filled in.
    """
    state = GameState(2)
    records = []
    step_index = 0
    max_turns = 200

    for _ in range(max_turns):
        if state.game_over:
            break

        state.roll_dice()
        valid = get_valid_moves(state)

        if not valid:
            bust_turn(state)
            continue

        player = state.active_player

        # Phase-based temperature
        temp = get_temperature(step_index, state, global_temp_multiplier)

        # Forward pass — get action and soft target
        model.eval()
        action_idx, move, decision, soft_target, features, mask, value = \
            get_action_with_soft_target(model, state, valid, device, temp)

        if move is None:
            bust_turn(state)
            continue

        records.append({
            'features':    features,      # np.array (74,)
            'mask':        mask,          # np.array (154,) bool
            'soft_target': soft_target,   # np.array (154,) float — KEY
            'action_idx':  action_idx,    # int — for accuracy tracking
            'player':      player,
            'step_index':  step_index,
            'value_pred':  value,         # network's win probability estimate
        })

        step_index += 1

        apply_move(state, move)
        if decision == "stop":
            stop_turn(state)

    # Fill in outcomes
    winner = state.winner
    labeled = []
    for rec in records:
        rec['outcome'] = 1.0 if rec['player'] == winner else 0.0
        labeled.append(rec)

    return labeled


# ---- REPLAY BUFFER ----

class ReplayBuffer:
    """
    Rolling replay buffer that stores past self-play records.
    Prevents catastrophic forgetting by mixing old and new experience.

    Acts like a circular queue — oldest records are dropped when full.
    """

    def __init__(self, max_size=500_000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, records):
        """Add new records to buffer."""
        self.buffer.extend(records)

    def size(self):
        return len(self.buffer)

    def sample(self, n):
        """Sample n records randomly from buffer."""
        n = min(n, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def to_tensors(self, records=None):
        """
        Convert buffer (or sample) to training tensors.
        Uses soft targets for policy training.
        """
        if records is None:
            records = list(self.buffer)

        n = len(records)
        features_t      = torch.zeros(n, FEATURE_SIZE,   dtype=torch.float32)
        masks_t         = torch.zeros(n, ACTION_SPACE,    dtype=torch.bool)
        soft_targets_t  = torch.zeros(n, ACTION_SPACE,    dtype=torch.float32)
        value_targets_t = torch.zeros(n,                  dtype=torch.float32)
        action_idx_t    = torch.zeros(n,                  dtype=torch.long)

        for i, rec in enumerate(records):
            features_t[i]     = torch.tensor(rec['features'],    dtype=torch.float32)
            masks_t[i]        = torch.tensor(rec['mask'],        dtype=torch.bool)
            soft_targets_t[i] = torch.tensor(rec['soft_target'], dtype=torch.float32)
            value_targets_t[i]= float(rec['outcome'])
            action_idx_t[i]   = rec['action_idx']

        return features_t, masks_t, soft_targets_t, value_targets_t, action_idx_t


# ---- SOFT TARGET LOSS ----

def soft_policy_loss(logits, soft_targets, masks, entropy_weight=0.01):
    """
    Numerically stable soft target loss with entropy regularization.
    Guards against all-illegal rows causing NaN in softmax.
    """
    # Guard: verify every row has at least one legal action
    valid_rows = masks.any(dim=1)
    if not valid_rows.all():
        # Filter out invalid rows
        logits       = logits[valid_rows]
        soft_targets = soft_targets[valid_rows]
        masks        = masks[valid_rows]

    if masks.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Apply mask
    masked_logits = logits.masked_fill(~masks, -1e9)  # use -1e9 not -inf

    log_probs = F.log_softmax(masked_logits, dim=-1)
    log_probs = log_probs.clamp(min=-100)

    # Normalize soft targets
    soft = soft_targets * masks.float()
    row_sums = soft.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    soft = soft / row_sums

    ce_loss = -(soft * log_probs).sum(dim=-1).mean()

    # Entropy regularization
    probs = F.softmax(masked_logits, dim=-1).clamp(min=1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()
    entropy_loss = -entropy_weight * entropy

    return ce_loss + entropy_loss


def combined_loss_soft(value_pred, value_target, logits, soft_targets,
                        masks, value_weight=1.0, policy_weight=1.0,
                        entropy_weight=0.01):
    v_loss = F.binary_cross_entropy(value_pred, value_target)
    p_loss = soft_policy_loss(logits, soft_targets, masks, entropy_weight)
    total = value_weight * v_loss + policy_weight * p_loss
    return total, v_loss, p_loss


# ---- TRAIN ON REPLAY BUFFER ----

def train_on_buffer(model, replay_buffer, device,
                    epochs=5, batch_size=512, lr=1e-4,
                    sample_size=None):
    """
    Train model on a sample from the replay buffer.
    Uses soft targets and KL divergence loss.

    sample_size: how many records to train on (None = all in buffer)
    """
    # Sample from buffer
    if sample_size and replay_buffer.size() > sample_size:
        records = replay_buffer.sample(sample_size)
    else:
        records = list(replay_buffer.buffer)

    features, masks, soft_targets, value_targets, action_idxs = \
        replay_buffer.to_tensors(records)

    n = len(records)
    val_size = int(n * 0.1)
    train_size = n - val_size

    # Shuffle
    perm = torch.randperm(n)
    features       = features[perm]
    masks          = masks[perm]
    soft_targets   = soft_targets[perm]
    value_targets  = value_targets[perm]
    action_idxs    = action_idxs[perm]

    # Split
    def split(t):
        return t[:train_size], t[train_size:]

    train_feat,  val_feat  = split(features)
    train_mask,  val_mask  = split(masks)
    train_soft,  val_soft  = split(soft_targets)
    train_val,   val_val   = split(value_targets)
    train_act,   val_act   = split(action_idxs)

    train_ds = TensorDataset(train_feat, train_mask, train_soft,
                              train_val, train_act)
    val_ds   = TensorDataset(val_feat,   val_mask,   val_soft,
                              val_val,   val_act)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n  Training on {train_size:,} records "
          f"({replay_buffer.size():,} in buffer) for {epochs} epochs...")

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        # ---- TRAIN ----
        model.train()
        total_loss = total_samples = 0

        for batch in train_loader:
            feat, msk, soft, val_t, act = [b.to(device) for b in batch]
            optimizer.zero_grad()

            val_pred, logits = model(feat, msk)
            loss, v_loss, p_loss = combined_loss_soft(
                val_pred, val_t, logits, soft, msk
            )

            loss.backward()
            
            # Check for NaN gradients
            has_nan = any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in model.parameters()
            )
            if has_nan:
                print("\n  WARNING: NaN gradients detected, skipping batch")
                optimizer.zero_grad()
                continue
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss   += loss.item() * feat.shape[0]
            total_samples += feat.shape[0]

        # ---- VALIDATE ----
        model.eval()
        val_loss_total = val_correct = val_total = entropy_sum = 0

        with torch.no_grad():
            for batch in val_loader:
                feat, msk, soft, val_t, act = [b.to(device) for b in batch]

                val_pred, logits = model(feat, msk)
                loss, _, _ = combined_loss_soft(
                    val_pred, val_t, logits, soft, msk
                )

                val_loss_total += loss.item() * feat.shape[0]

                # Policy accuracy — argmax of masked logits vs sampled action
                masked_logits = logits.masked_fill(~msk, -1e9)
                preds = masked_logits.argmax(1)
                val_correct += (preds == act).sum().item()
                val_total   += feat.shape[0]

                # Entropy — monitor for collapse
                probs = F.softmax(masked_logits, dim=-1).clamp(min=1e-8)
                entropy = -(probs * probs.log()).sum(dim=-1).mean()
                entropy_sum += entropy.item() * feat.shape[0]

        train_loss = total_loss    / total_samples
        val_loss   = val_loss_total / val_total
        val_acc    = val_correct   / val_total
        avg_entropy= entropy_sum   / val_total

        print(f"    Epoch {epoch}/{epochs} | "
              f"Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"Policy acc: {val_acc:.3f} | "
              f"Entropy: {avg_entropy:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    return best_val_loss


# ---- EVALUATION ----

def evaluate_networks(new_model, old_model, device, num_games=500):
    """
    Evaluate new model against old model.
    Returns win rate of new model.
    500 games gives ±2.2% margin of error at 55%.
    """
    new_fn = lambda s: nn_player(s, new_model, device)
    old_fn = lambda s: nn_player(s, old_model, device)

    wins_new = wins_old = draws = 0

    print(f"\n  Evaluating new vs old ({num_games} games)...")

    for i in range(num_games):
        if (i + 1) % 100 == 0:
            print(f"\r    Game {i+1}/{num_games}...", end="", flush=True)

        if random.random() < 0.5:
            winner = play_game(new_fn, old_fn)
            if winner == 0:   wins_new += 1
            elif winner == 1: wins_old += 1
            else:             draws += 1
        else:
            winner = play_game(old_fn, new_fn)
            if winner == 0:   wins_old += 1
            elif winner == 1: wins_new += 1
            else:             draws += 1

    print()
    total = wins_new + wins_old
    win_rate = wins_new / total if total > 0 else 0.5

    print(f"  New model: {wins_new}/{total} ({win_rate:.1%})")
    print(f"  Old model: {wins_old}/{total} ({1-win_rate:.1%})")

    return win_rate


# ---- MAIN SELF-PLAY LOOP ----

def self_play_loop(
    initial_model_path,
    output_dir='models/cantstop/self_play',
    iterations=10,
    games_per_iter=5000,
    train_epochs=5,
    eval_games=500,
    acceptance_threshold=0.55,
    buffer_size=500_000,
    sample_size=200_000,
    initial_temp_mult=1.0,
    final_temp_mult=0.8,
    device=None,
):
    """
    Main self-play training loop with all four improvements:
      1. Replay buffer (buffer_size records)
      2. Larger evaluation (eval_games)
      3. Phase-based temperature (+ global multiplier decay)
      4. Soft targets (KL divergence)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*55}")
    print(f"  Can't Stop Self-Play Training v2")
    print(f"{'='*55}")
    print(f"  Device:        {device}")
    if device == 'cuda':
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
    print(f"  Iterations:    {iterations}")
    print(f"  Games/iter:    {games_per_iter:,}")
    print(f"  Buffer size:   {buffer_size:,}")
    print(f"  Sample size:   {sample_size:,}")
    print(f"  Train epochs:  {train_epochs}")
    print(f"  Eval games:    {eval_games}")
    print(f"  Accept thresh: {acceptance_threshold:.0%}")
    print(f"{'='*55}\n")

    # Load initial model
    current_model = load_model(initial_model_path, device)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=buffer_size)

    history = []
    accepted = 0
    rejected = 0

    # Seed replay buffer with supervised training data
    # This prevents catastrophic forgetting of EV player knowledge
    print("Seeding replay buffer with supervised data...")
    import json
    seed_path = "data/cantstop/training_data_20260509_235133.jsonl"
    seed_records = []
    with open(seed_path) as f:
        for i, line in enumerate(f):
            if i >= 50000:  # seed with 50K supervised records
                break
            try:
                rec = json.loads(line)
                # Convert supervised record to self-play format
                from games.cantstop.train import _record_to_state
                state = _record_to_state(rec)
                valid = [tuple(m) for m in rec['valid_moves']]
                features = extract_features(state, valid)
                mask = get_legal_action_mask(valid)
                
                # Hard target converted to soft (near one-hot)
                soft = mask.astype(np.float32)
                move = tuple(rec['move'])
                decision = rec['decision']
                try:
                    action_idx = move_to_action(move, decision)
                    soft_target = np.zeros(ACTION_SPACE, dtype=np.float32)
                    soft_target[action_idx] = 1.0
                except ValueError:
                    continue
                    
                seed_records.append({
                    'features':   features,
                    'mask':       mask,
                    'soft_target': soft_target,
                    'action_idx': action_idx,
                    'player':     rec['active_player'],
                    'step_index': rec.get('step_index', 0),
                    'outcome':    float(rec['outcome']),
                    'value_pred': 0.5,
                })
            except:
                continue

    replay_buffer.add(seed_records)
    print(f"Seeded buffer with {len(seed_records):,} supervised records")

    for iteration in range(1, iterations + 1):
        iter_start = time.time()

        # Global temperature multiplier decays over iterations
        temp_mult = initial_temp_mult - (
            (initial_temp_mult - final_temp_mult) *
            (iteration - 1) / max(iterations - 1, 1)
        )

        print(f"\n{'─'*55}")
        print(f"  Iteration {iteration}/{iterations} | "
              f"Temp mult: {temp_mult:.2f} | "
              f"Buffer: {replay_buffer.size():,}")
        print(f"{'─'*55}")

        # ---- STAGE 1: GENERATE SELF-PLAY GAMES ----
        print(f"\n  Generating {games_per_iter:,} self-play games...")
        gen_start = time.time()
        all_new_records = []

        for i in range(games_per_iter):
            if (i + 1) % 500 == 0:
                elapsed = time.time() - gen_start
                rate = len(all_new_records) / elapsed if elapsed > 0 else 0
                print(f"\r    Game {i+1:,}/{games_per_iter:,} | "
                      f"Records: {len(all_new_records):,} | "
                      f"Rate: {rate:,.0f}/s",
                      end="", flush=True)

            records = play_self_play_game(
                current_model, device, temp_mult
            )
            all_new_records.extend(records)

        print()
        gen_time = time.time() - gen_start
        print(f"  Generated {len(all_new_records):,} records "
              f"in {gen_time:.1f}s "
              f"({len(all_new_records)/gen_time:,.0f}/s)")

        # Add to replay buffer
        replay_buffer.add(all_new_records)
        print(f"  Buffer size: {replay_buffer.size():,} / {buffer_size:,}")

        # ---- STAGE 2: TRAIN NEW MODEL ----
        new_model = CantStopNet().to(device)
        new_model.load_state_dict(current_model.state_dict())

        val_loss = train_on_buffer(
            new_model, replay_buffer, device,
            epochs=train_epochs,
            sample_size=sample_size,
            lr=3e-5,
        )

        # ---- STAGE 3: EVALUATE ----
        win_rate = evaluate_networks(
            new_model, current_model, device, eval_games
        )

        iter_time = time.time() - iter_start

        # ---- ACCEPT OR REJECT ----
        if win_rate >= acceptance_threshold:
            print(f"\n  ✓ ACCEPTED ({win_rate:.1%} >= {acceptance_threshold:.0%})")
            current_model = new_model
            accepted += 1

            save_path = os.path.join(
                output_dir,
                f'model_iter_{iteration:03d}_accepted.pt'
            )
            torch.save({
                'iteration':   iteration,
                'win_rate':    win_rate,
                'val_loss':    float(val_loss),
                'temp_mult':   temp_mult,
                'model_state': new_model.state_dict(),
            }, save_path)
            print(f"  Saved: {save_path}")

        else:
            print(f"\n  ✗ REJECTED ({win_rate:.1%} < {acceptance_threshold:.0%})")
            rejected += 1

        history.append({
            'iteration':  iteration,
            'win_rate':   float(win_rate),
            'val_loss':   float(val_loss),
            'temp_mult':  float(temp_mult),
            'buffer_size':replay_buffer.size(),
            'accepted':   win_rate >= acceptance_threshold,
            'time':       float(iter_time),
        })

        print(f"\n  Time: {iter_time:.1f}s | "
              f"Accepted: {accepted} | Rejected: {rejected}")

    # ---- FINAL SUMMARY ----
    print(f"\n{'='*55}")
    print(f"  Self-Play Complete!")
    print(f"  Iterations:  {iterations}")
    print(f"  Accepted:    {accepted}")
    print(f"  Rejected:    {rejected}")
    print(f"\n  Win rate progression:")
    for h in history:
        status = "✓" if h['accepted'] else "✗"
        print(f"    Iter {h['iteration']:2d}: {h['win_rate']:.1%} {status} "
              f"(buffer: {h['buffer_size']:,})")
    print(f"{'='*55}\n")

    # Save final model
    final_path = os.path.join(output_dir, f'final_{timestamp}.pt')
    torch.save({
        'model_state': current_model.state_dict(),
        'history':     history,
    }, final_path)
    print(f"  Final model: {final_path}")

    return current_model, history


# ---- ENTRY POINT ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-play training for Can't Stop"
    )
    parser.add_argument("--model",      type=str,
                        default="models/cantstop/best_model.pt")
    parser.add_argument("--output",     type=str,
                        default="models/cantstop/self_play")
    parser.add_argument("--iterations", type=int,   default=10)
    parser.add_argument("--games",      type=int,   default=5000)
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--eval",       type=int,   default=500)
    parser.add_argument("--threshold",  type=float, default=0.55)
    parser.add_argument("--buffer",     type=int,   default=500_000)
    parser.add_argument("--sample",     type=int,   default=200_000)
    parser.add_argument("--temp_start", type=float, default=1.0)
    parser.add_argument("--temp_end",   type=float, default=0.8)
    parser.add_argument("--device",     type=str,   default=None)
    args = parser.parse_args()

    self_play_loop(
        initial_model_path=args.model,
        output_dir=args.output,
        iterations=args.iterations,
        games_per_iter=args.games,
        train_epochs=args.epochs,
        eval_games=args.eval,
        acceptance_threshold=args.threshold,
        buffer_size=args.buffer,
        sample_size=args.sample,
        initial_temp_mult=args.temp_start,
        final_temp_mult=args.temp_end,
        device=args.device,
    )