# self_play.py
# MCTS self-play training loop for Can't Stop.
#
# Key design:
#   - MCTS provides training targets (visit counts + values)
#   - Workers generate games on CPU in parallel
#   - Main process trains on GPU
#   - Replay buffer prevents catastrophic forgetting
#   - Acceptance gating prevents regressions

import os
import sys
import time
import random
import argparse
import numpy as np
from collections import deque
from datetime import datetime
import multiprocessing as mp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, get_valid_moves, apply_move,
    stop_turn, bust_turn
)
from games.cantstop.features import (
    extract_features, get_legal_action_mask,
    move_to_action, action_to_move_decision,
    FEATURE_SIZE, ACTION_SPACE
)
from games.cantstop.model import CantStopNet
from games.cantstop.mcts import MCTS
from games.cantstop.evaluate import load_model, nn_player
from games.cantstop.ev_player import play_game


# ---- PHASE-BASED TEMPERATURE ----

def get_temperature(step_index, state, global_temp_mult=1.0):
    """
    Phase-based temperature for MCTS action selection.
    Early game: more random → diverse positions
    Late game:  near-greedy → precise endgame play
    """
    max_score = max(
        len(state.claimed[0]),
        len(state.claimed[1])
    )
    if max_score >= 2:
        base_temp = 0.3
    elif max_score == 1 or step_index > 15:
        base_temp = 0.7
    else:
        base_temp = 1.0
    return base_temp * global_temp_mult


# ---- SELF-PLAY GAME WITH MCTS ----

def play_mcts_game(mcts, num_simulations=20, global_temp_mult=1.0):
    """
    Play one complete self-play game using MCTS for decisions.

    For each position:
      1. Run MCTS (guided by network)
      2. Record: features, MCTS visit counts (policy target),
                 MCTS value estimate (value target)
      3. Sample action from MCTS distribution with temperature

    Returns list of labeled training records.
    """
    state = GameState(2)
    records = []
    step_index = 0
    max_turns = 200

    for _ in range(max_turns):
        if state.game_over:
            break

        if not state.dice:
            state.roll_dice()

        valid = get_valid_moves(state)

        if not valid:
            bust_turn(state)
            state.dice = []
            continue

        player = state.active_player
        temp = get_temperature(step_index, state, global_temp_mult)

        # MCTS search — this is the key signal
        action_idx, move, decision, mcts_policy, mcts_value = \
            mcts.get_action(
                state,
                num_simulations=num_simulations,
                temperature=temp
            )

        if move is None:
            bust_turn(state)
            state.dice = []
            continue

        # Record position with MCTS targets
        features = extract_features(state, valid)
        mask     = get_legal_action_mask(valid)

        records.append({
            'features':    features,
            'mask':        mask,
            'mcts_policy': mcts_policy,
            'mcts_value':  mcts_value,
            'action_idx':  action_idx,
            'player':      player,
            'step_index':  step_index,
        })

        step_index += 1

        apply_move(state, move)
        if decision == "stop":
            stop_turn(state)
            state.dice = []

    # Fill in value targets — blend MCTS estimate with final outcome
    winner = state.winner
    lambda_blend = 0.7
    labeled = []
    for rec in records:
        final_outcome = 1.0 if rec['player'] == winner else 0.0
        rec['value_target'] = (
            lambda_blend * final_outcome +
            (1 - lambda_blend) * rec['mcts_value']
        )
        labeled.append(rec)

    return labeled


# ---- PARALLEL WORKER FUNCTIONS ----
# Must be top-level for multiprocessing pickling on Windows

_worker_mcts = None


def _init_worker(model_path):
    """
    Initialize worker process with CPU-based model and MCTS.
    Called once per worker — avoids reloading for each batch.
    Workers use CPU to avoid CUDA multiprocessing issues on Windows.
    Network inference is only 3% of time so CPU is fine.
    """
    global _worker_mcts
    worker_device = 'cpu'
    model = load_model(model_path, worker_device)
    _worker_mcts = MCTS(model, worker_device)


def _worker_generate_batch(args):
    """
    Generate a batch of MCTS games in a worker process.
    Uses pre-initialized _worker_mcts instance.
    Returns [] on any error so the pool doesn't hang overnight.
    """
    num_games, num_simulations, temp_mult, seed = args
    random.seed(seed)
    np.random.seed(seed)

    all_records = []
    for i in range(num_games):
        try:
            records = play_mcts_game(
                _worker_mcts,
                num_simulations=num_simulations,
                global_temp_mult=temp_mult,
            )
            all_records.extend(records)
        except Exception as e:
            import traceback
            print(f"  [worker] Game {i+1}/{num_games} failed: {e}", flush=True)
            traceback.print_exc()
            continue  # skip bad game, keep going

    return all_records


def generate_games_parallel(model_path, num_games, num_simulations,
                             temp_mult, num_workers=None):
    """
    Generate MCTS games across multiple CPU worker processes.
    Each worker has its own model and MCTS instance.
    Main process keeps GPU for training.
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)

    # Distribute games evenly
    games_per_worker = num_games // num_workers
    leftover = num_games % num_workers

    rng = random.Random(42)
    batch_args = []
    for i in range(num_workers):
        n = games_per_worker + (1 if i < leftover else 0)
        if n > 0:
            batch_args.append((
                n,
                num_simulations,
                temp_mult,
                rng.randint(0, 2**31)
            ))

    actual_workers = len(batch_args)

    with mp.Pool(
        processes=actual_workers,
        initializer=_init_worker,
        initargs=(model_path,)
    ) as pool:
        results = pool.map(_worker_generate_batch, batch_args)

    all_records = []
    for batch in results:
        all_records.extend(batch)
    return all_records


# ---- REPLAY BUFFER ----

class ReplayBuffer:
    """Rolling replay buffer — prevents catastrophic forgetting."""

    def __init__(self, max_size=500_000):
        self.max_size = max_size
        self.buffer   = deque(maxlen=max_size)

    def add(self, records):
        self.buffer.extend(records)

    def size(self):
        return len(self.buffer)

    def sample(self, n):
        n = min(n, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def to_tensors(self, records):
        """Convert records to training tensors."""
        n = len(records)
        features_t       = torch.zeros(n, FEATURE_SIZE,  dtype=torch.float32)
        masks_t          = torch.zeros(n, ACTION_SPACE,   dtype=torch.bool)
        policy_targets_t = torch.zeros(n, ACTION_SPACE,   dtype=torch.float32)
        value_targets_t  = torch.zeros(n,                 dtype=torch.float32)
        action_idx_t     = torch.zeros(n,                 dtype=torch.long)

        for i, rec in enumerate(records):
            features_t[i]       = torch.tensor(rec['features'],    dtype=torch.float32)
            masks_t[i]          = torch.tensor(rec['mask'],        dtype=torch.bool)
            policy_targets_t[i] = torch.tensor(rec['mcts_policy'], dtype=torch.float32)
            value_targets_t[i]  = float(rec['value_target'])
            action_idx_t[i]     = rec['action_idx']

        return features_t, masks_t, policy_targets_t, value_targets_t, action_idx_t


# ---- LOSS FUNCTIONS ----

def policy_loss_mcts(logits, policy_targets, masks):
    """
    Cross entropy loss with MCTS visit count targets.
    Numerically stable — guards against all-illegal rows.
    Includes entropy regularization to prevent policy collapse.
    """
    # Guard against all-illegal rows
    valid_rows = masks.any(dim=1)
    if not valid_rows.all():
        logits         = logits[valid_rows]
        policy_targets = policy_targets[valid_rows]
        masks          = masks[valid_rows]

    if masks.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    masked_logits = logits.masked_fill(~masks, -1e9)
    log_probs     = F.log_softmax(masked_logits, dim=-1).clamp(min=-100)

    # Normalize targets over legal actions
    targets  = policy_targets * masks.float()
    row_sums = targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    targets  = targets / row_sums

    ce_loss = -(targets * log_probs).sum(dim=-1).mean()

    # Entropy regularization
    probs   = F.softmax(masked_logits, dim=-1).clamp(min=1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()

    return ce_loss - 0.01 * entropy


def combined_loss_mcts(value_pred, value_target, logits,
                        policy_targets, masks):
    """Combined value + MCTS policy loss."""
    v_loss = F.binary_cross_entropy(value_pred, value_target)
    p_loss = policy_loss_mcts(logits, policy_targets, masks)
    total  = v_loss + p_loss
    return total, v_loss, p_loss


# ---- TRAIN ON BUFFER ----

def train_on_buffer(model, replay_buffer, device,
                    epochs=5, batch_size=512, lr=3e-5,
                    sample_size=None):
    """Train model on sample from replay buffer using MCTS targets."""
    if sample_size and replay_buffer.size() > sample_size:
        records = replay_buffer.sample(sample_size)
    else:
        records = list(replay_buffer.buffer)

    features, masks, policy_targets, value_targets, action_idxs = \
        replay_buffer.to_tensors(records)

    n        = len(records)
    val_size = int(n * 0.1)
    tr_size  = n - val_size

    perm           = torch.randperm(n)
    features       = features[perm]
    masks          = masks[perm]
    policy_targets = policy_targets[perm]
    value_targets  = value_targets[perm]
    action_idxs    = action_idxs[perm]

    def split(t): return t[:tr_size], t[tr_size:]

    tr_feat, vl_feat = split(features)
    tr_mask, vl_mask = split(masks)
    tr_pol,  vl_pol  = split(policy_targets)
    tr_val,  vl_val  = split(value_targets)
    tr_act,  vl_act  = split(action_idxs)

    train_ds = TensorDataset(tr_feat, tr_mask, tr_pol, tr_val, tr_act)
    val_ds   = TensorDataset(vl_feat, vl_mask, vl_pol, vl_val, vl_act)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n  Training on {tr_size:,} records "
          f"({replay_buffer.size():,} in buffer) for {epochs} epochs...")

    best_val_loss = float('inf')
    best_state    = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = total_samples = 0
        nan_batches = total_batches = 0

        for batch in train_loader:
            feat, msk, pol, val_t, act = [b.to(device) for b in batch]
            optimizer.zero_grad()
            total_batches += 1

            val_pred, logits = model(feat, msk)
            loss, v_loss, p_loss = combined_loss_mcts(
                val_pred, val_t, logits, pol, msk
            )

            if torch.isnan(loss):
                optimizer.zero_grad()
                nan_batches += 1
                continue

            loss.backward()

            has_nan = any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in model.parameters()
            )
            if has_nan:
                optimizer.zero_grad()
                nan_batches += 1
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss    += loss.item() * feat.shape[0]
            total_samples += feat.shape[0]

        if nan_batches > 0:
            nan_pct = nan_batches / max(total_batches, 1) * 100
            warn = "⚠️  WARNING: high NaN rate" if nan_pct > 1.0 else "note"
            print(f"    [{warn}] Skipped {nan_batches}/{total_batches} "
                  f"batches ({nan_pct:.1f}%) due to NaN loss/grads")

        model.eval()
        vl_loss_total = vl_correct = vl_total = entropy_sum = 0

        with torch.no_grad():
            for batch in val_loader:
                feat, msk, pol, val_t, act = [b.to(device) for b in batch]
                val_pred, logits = model(feat, msk)
                loss, _, _       = combined_loss_mcts(
                    val_pred, val_t, logits, pol, msk
                )
                vl_loss_total += loss.item() * feat.shape[0]

                masked_logits  = logits.masked_fill(~msk, -1e9)
                preds          = masked_logits.argmax(1)
                vl_correct    += (preds == act).sum().item()
                vl_total      += feat.shape[0]

                probs   = F.softmax(masked_logits, dim=-1).clamp(min=1e-8)
                entropy = -(probs * probs.log()).sum(dim=-1).mean()
                entropy_sum += entropy.item() * feat.shape[0]

        tr_loss     = total_loss    / max(total_samples, 1)
        vl_loss     = vl_loss_total / max(vl_total, 1)
        vl_acc      = vl_correct    / max(vl_total, 1)
        avg_entropy = entropy_sum   / max(vl_total, 1)

        print(f"    Epoch {epoch}/{epochs} | "
              f"Train: {tr_loss:.4f} | "
              f"Val: {vl_loss:.4f} | "
              f"Policy acc: {vl_acc:.3f} | "
              f"Entropy: {avg_entropy:.3f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state    = {k: v.clone()
                            for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    return best_val_loss


# ---- EVALUATION ----

def evaluate_networks(new_model, old_model, device, num_games=500):
    """Evaluate new vs old model. Returns new model win rate."""
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
            else:             draws    += 1
        else:
            winner = play_game(old_fn, new_fn)
            if winner == 0:   wins_old += 1
            elif winner == 1: wins_new += 1
            else:             draws    += 1

    print()
    total    = wins_new + wins_old
    win_rate = wins_new / total if total > 0 else 0.5

    print(f"  New model: {wins_new}/{total} ({win_rate:.1%})")
    print(f"  Old model: {wins_old}/{total} ({1-win_rate:.1%})")

    return win_rate


# ---- MAIN SELF-PLAY LOOP ----

def self_play_loop(
    initial_model_path,
    output_dir='models/cantstop/self_play',
    iterations=10,
    games_per_iter=1000,
    num_simulations=20,
    train_epochs=5,
    eval_games=500,
    acceptance_threshold=0.55,
    buffer_size=200_000,
    sample_size=100_000,
    initial_temp_mult=1.0,
    final_temp_mult=0.7,
    num_workers=None,
    device=None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*55}")
    print(f"  Can't Stop MCTS Self-Play")
    print(f"{'='*55}")
    print(f"  Device:        {device}")
    if device == 'cuda':
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
    print(f"  Workers:       {num_workers} (CPU)")
    print(f"  Iterations:    {iterations}")
    print(f"  Games/iter:    {games_per_iter:,}")
    print(f"  MCTS sims:     {num_simulations}")
    print(f"  Buffer size:   {buffer_size:,}")
    print(f"  Sample size:   {sample_size:,}")
    print(f"  Train epochs:  {train_epochs}")
    print(f"  Eval games:    {eval_games}")
    print(f"  Accept thresh: {acceptance_threshold:.0%}")
    print(f"{'='*55}\n")

    # Load model — track path for workers
    current_model      = load_model(initial_model_path, device)
    current_model_path = initial_model_path

    replay_buffer = ReplayBuffer(max_size=buffer_size)

    history  = []
    accepted = 0
    rejected = 0

    for iteration in range(1, iterations + 1):
        iter_start = time.time()

        temp_mult = initial_temp_mult - (
            (initial_temp_mult - final_temp_mult) *
            (iteration - 1) / max(iterations - 1, 1)
        )

        print(f"\n{'─'*55}")
        print(f"  Iteration {iteration}/{iterations} | "
              f"Temp: {temp_mult:.2f} | "
              f"Sims: {num_simulations} | "
              f"Buffer: {replay_buffer.size():,}")
        print(f"{'─'*55}")

        # ---- STAGE 1: PARALLEL GAME GENERATION ----
        print(f"\n  Generating {games_per_iter:,} MCTS games "
              f"({num_simulations} sims/move, "
              f"{num_workers} workers)...")
        gen_start = time.time()

        all_new_records = generate_games_parallel(
            model_path=current_model_path,
            num_games=games_per_iter,
            num_simulations=num_simulations,
            temp_mult=temp_mult,
            num_workers=num_workers,
        )

        gen_time = time.time() - gen_start
        print(f"  Generated {len(all_new_records):,} records "
              f"in {gen_time:.1f}s "
              f"({games_per_iter/gen_time:.2f} games/s)")

        replay_buffer.add(all_new_records)
        print(f"  Buffer: {replay_buffer.size():,} / {buffer_size:,}")

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
            print(f"\n  ✓ ACCEPTED ({win_rate:.1%} >= "
                  f"{acceptance_threshold:.0%})")
            current_model = new_model

            save_path = os.path.join(
                output_dir,
                f'model_iter_{iteration:03d}_accepted.pt'
            )
            torch.save({
                'iteration':   iteration,
                'win_rate':    float(win_rate),
                'val_loss':    float(val_loss),
                'temp_mult':   float(temp_mult),
                'model_state': new_model.state_dict(),
            }, save_path)
            current_model_path = save_path  # workers use new model
            accepted += 1
            print(f"  Saved: {save_path}")

        else:
            print(f"\n  ✗ REJECTED ({win_rate:.1%} < "
                  f"{acceptance_threshold:.0%})")
            rejected += 1

        history.append({
            'iteration':   iteration,
            'win_rate':    float(win_rate),
            'val_loss':    float(val_loss),
            'temp_mult':   float(temp_mult),
            'buffer_size': replay_buffer.size(),
            'accepted':    win_rate >= acceptance_threshold,
            'time':        float(iter_time),
        })

        print(f"\n  Time: {iter_time:.1f}s | "
              f"Accepted: {accepted} | Rejected: {rejected}")

    # ---- SUMMARY ----
    print(f"\n{'='*55}")
    print(f"  MCTS Self-Play Complete!")
    print(f"  Iterations: {iterations}")
    print(f"  Accepted:   {accepted}")
    print(f"  Rejected:   {rejected}")
    print(f"\n  Win rate progression:")
    for h in history:
        status = "✓" if h['accepted'] else "✗"
        print(f"    Iter {h['iteration']:2d}: "
              f"{h['win_rate']:.1%} {status} "
              f"({h['time']:.0f}s)")
    print(f"{'='*55}\n")

    final_path = os.path.join(output_dir, f'final_{timestamp}.pt')
    torch.save({
        'model_state': current_model.state_dict(),
        'history':     history,
    }, final_path)
    print(f"  Final model: {final_path}")

    return current_model, history


# ---- ENTRY POINT ----

if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows

    parser = argparse.ArgumentParser(
        description="MCTS self-play for Can't Stop"
    )
    parser.add_argument("--model",      type=str,
                        default="models/cantstop/best_model.pt")
    parser.add_argument("--output",     type=str,
                        default="models/cantstop/self_play")
    parser.add_argument("--iterations", type=int,   default=10)
    parser.add_argument("--games",      type=int,   default=1000)
    parser.add_argument("--sims",       type=int,   default=20)
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--eval",       type=int,   default=500)
    parser.add_argument("--threshold",  type=float, default=0.55)
    parser.add_argument("--buffer",     type=int,   default=200_000)
    parser.add_argument("--sample",     type=int,   default=100_000)
    parser.add_argument("--temp_start", type=float, default=1.0)
    parser.add_argument("--temp_end",   type=float, default=0.7)
    parser.add_argument("--workers",    type=int,   default=None)
    parser.add_argument("--device",     type=str,   default=None)
    args = parser.parse_args()

    self_play_loop(
        initial_model_path=args.model,
        output_dir=args.output,
        iterations=args.iterations,
        games_per_iter=args.games,
        num_simulations=args.sims,
        train_epochs=args.epochs,
        eval_games=args.eval,
        acceptance_threshold=args.threshold,
        buffer_size=args.buffer,
        sample_size=args.sample,
        initial_temp_mult=args.temp_start,
        final_temp_mult=args.temp_end,
        num_workers=args.workers,
        device=args.device,
    )