# train.py
# Training pipeline for the Can't Stop neural network.
#
# Loads training data from .jsonl files, converts to tensors,
# and trains the dual-head (value + policy) network.
#
# Usage:
#   python games/cantstop/train.py --data data/cantstop/training_data_*.jsonl
#   python games/cantstop/train.py --data data/cantstop/training_data_*.jsonl --epochs 10

import os
import sys
import json
import time
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.features import (
    FEATURE_SIZE, ACTION_SPACE, COLUMNS,
    extract_features, get_legal_action_mask,
    move_to_action, action_to_move_decision,
    BUST_CACHE, ROLL_FREQUENCY, DIFFICULTY_WT,
    MAX_THREAT, COL_INDEX
)
from games.cantstop.engine import GameState, COLUMN_HEIGHTS
from games.cantstop.model import CantStopNet, combined_loss


# ---- DATASET ----

class CantStopDataset(Dataset):
    """
    Memory-efficient dataset — stores masks as move lists,
    reconstructs bool tensor on demand in __getitem__.
    Reduces RAM usage by ~3.4GB for 21M records.
    """

    def __init__(self, jsonl_paths, max_records=None, skip_exploration=False):
        print(f"Loading and preprocessing training data...")

        raw_records = []
        for path in jsonl_paths:
            print(f"  Reading {path}...")
            loaded = skipped = 0
            with open(path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if skip_exploration and record.get('is_exploration', False):
                        skipped += 1
                        continue

                    raw_records.append(record)
                    loaded += 1

                    if max_records and len(raw_records) >= max_records:
                        break

            print(f"    Loaded {loaded:,} records (skipped {skipped:,})")
            if max_records and len(raw_records) >= max_records:
                break

        import gc
        n = len(raw_records)
        print(f"\n  Preprocessing {n:,} records...")
        random.shuffle(raw_records)

        # Pre-allocate compact arrays
        # Features: float32 — unavoidable
        self.features       = np.zeros((n, FEATURE_SIZE), dtype=np.float32)
        # Masks: packed bits — 154 bits = 20 bytes per record vs 154 bytes
        self.masks_packed   = np.zeros((n, 20),            dtype=np.uint8)
        self.value_targets  = np.zeros(n,                  dtype=np.float32)
        self.policy_targets = np.zeros(n,                  dtype=np.int32)

        errors = 0
        for i, rec in enumerate(raw_records):
            try:
                state = _record_to_state(rec)
                valid_moves = [tuple(m) for m in rec['valid_moves']]

                self.features[i] = extract_features(state, valid_moves)

                # Pack mask into bits
                mask = get_legal_action_mask(valid_moves)
                self.masks_packed[i] = np.packbits(mask, bitorder='little')

                self.value_targets[i]  = float(rec['outcome'])

                move = tuple(rec['move'])
                decision = rec['decision']
                try:
                    action_idx = move_to_action(move, decision)
                except ValueError:
                    action_idx = int(mask.nonzero()[0])
                self.policy_targets[i] = action_idx

            except Exception as e:
                errors += 1
                continue

            if (i + 1) % 500_000 == 0:
                print(f"    Preprocessed {i+1:,} / {n:,}...")

        # Free raw records immediately
        del raw_records
        gc.collect()

        if errors:
            print(f"    Warning: {errors} records failed")

        # Report memory usage
        feature_mb = self.features.nbytes / 1024**2
        mask_mb    = self.masks_packed.nbytes / 1024**2
        total_mb   = (self.features.nbytes + self.masks_packed.nbytes +
                     self.value_targets.nbytes + self.policy_targets.nbytes) / 1024**2
        print(f"  Memory usage:")
        print(f"    Features: {feature_mb:.0f}MB")
        print(f"    Masks:    {mask_mb:.0f}MB")
        print(f"    Total:    {total_mb:.0f}MB")
        print(f"  Done. {n:,} records ready.\n")

    def __len__(self):
        return len(self.value_targets)

    def __getitem__(self, idx):
        # Unpack bits back to bool tensor
        mask = np.unpackbits(
            self.masks_packed[idx], count=ACTION_SPACE, bitorder='little'
        ).astype(bool)

        return (
            torch.tensor(self.features[idx],       dtype=torch.float32),
            torch.tensor(mask,                     dtype=torch.bool),
            torch.tensor(self.value_targets[idx],  dtype=torch.float32),
            torch.tensor(self.policy_targets[idx], dtype=torch.long),
        )


def _record_to_state(rec):
    """
    Reconstruct a GameState from a training record.
    Used for feature extraction during dataset loading.
    """
    state = GameState(2)
    player   = rec['active_player']
    opponent = 1 - player

    state.active_player = player
    state.dice = list(rec['dice'])

    # Progress — convert string keys back to int
    state.progress[player]   = {int(k): v for k, v in rec['progress_active'].items()}
    state.progress[opponent] = {int(k): v for k, v in rec['progress_opponent'].items()}

    # Claimed columns
    state.claimed[player]   = set(rec['claimed_active'])
    state.claimed[opponent] = set(rec['claimed_opponent'])
    state.all_claimed       = state.claimed[0] | state.claimed[1]

    # Runners
    state.runners = {int(k): v for k, v in rec['runners'].items()} \
        if rec['runners'] else {}

    return state


# ---- TRAINING LOOP ----

def train_epoch(model, loader, optimizer, device, epoch):
    """Run one full pass through the training data."""
    model.train()

    total_loss   = 0.0
    total_v_loss = 0.0
    total_p_loss = 0.0
    total_samples = 0
    correct_policy = 0

    start = time.time()

    for batch_idx, (features, masks, value_targets, policy_targets) in enumerate(loader):
        features       = features.to(device)
        masks          = masks.to(device)
        value_targets  = value_targets.to(device)
        policy_targets = policy_targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        value_pred, policy_logits = model(features, masks)

        # Loss
        loss, v_loss, p_loss = combined_loss(
            value_pred, value_targets,
            policy_logits, policy_targets
        )

        # Backward pass
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Metrics
        batch_size = features.shape[0]
        total_loss   += loss.item()   * batch_size
        total_v_loss += v_loss.item() * batch_size
        total_p_loss += p_loss.item() * batch_size
        total_samples += batch_size

        # Policy accuracy — did we predict the chosen action?
        predicted = policy_logits.argmax(dim=1)
        correct_policy += (predicted == policy_targets).sum().item()

        # Progress update every 100 batches
        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - start
            samples_per_sec = total_samples / elapsed
            avg_loss = total_loss / total_samples

            print(
                f"\r  Epoch {epoch} | "
                f"Batch {batch_idx+1}/{len(loader)} | "
                f"Loss: {avg_loss:.4f} | "
                f"Speed: {samples_per_sec:,.0f} samples/s",
                end="", flush=True
            )

    print()  # newline after progress

    n = total_samples
    return {
        'loss':          total_loss   / n,
        'value_loss':    total_v_loss / n,
        'policy_loss':   total_p_loss / n,
        'policy_acc':    correct_policy / n,
    }


def evaluate(model, loader, device):
    """Evaluate model on validation set."""
    model.eval()

    total_loss   = 0.0
    total_v_loss = 0.0
    total_p_loss = 0.0
    total_samples = 0
    correct_policy = 0
    value_errors = []

    with torch.no_grad():
        for features, masks, value_targets, policy_targets in loader:
            features       = features.to(device)
            masks          = masks.to(device)
            value_targets  = value_targets.to(device)
            policy_targets = policy_targets.to(device)

            value_pred, policy_logits = model(features, masks)

            loss, v_loss, p_loss = combined_loss(
                value_pred, value_targets,
                policy_logits, policy_targets
            )

            batch_size = features.shape[0]
            total_loss   += loss.item()   * batch_size
            total_v_loss += v_loss.item() * batch_size
            total_p_loss += p_loss.item() * batch_size
            total_samples += batch_size

            predicted = policy_logits.argmax(dim=1)
            correct_policy += (predicted == policy_targets).sum().item()

            # Value MAE — mean absolute error in win probability
            mae = (value_pred - value_targets).abs().mean().item()
            value_errors.append(mae)

    n = total_samples
    return {
        'loss':          total_loss   / n,
        'value_loss':    total_v_loss / n,
        'policy_loss':   total_p_loss / n,
        'policy_acc':    correct_policy / n,
        'value_mae':     np.mean(value_errors),
    }


# ---- CHECKPOINT ----

def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save model checkpoint."""
    torch.save({
        'epoch':      epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'metrics':    metrics,
    }, path)
    print(f"  Saved checkpoint: {path}")


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    print(f"  Loaded checkpoint from epoch {epoch}: {path}")
    return epoch, metrics


# ---- MAIN TRAINING FUNCTION ----

def train(
    data_paths,
    output_dir='models/cantstop',
    epochs=20,
    batch_size=512,
    lr=1e-3,
    val_fraction=0.1,
    max_records=None,
    resume=None,
    device=None,
):
    """
    Main training function.

    Parameters:
        data_paths:    list of .jsonl file paths
        output_dir:    where to save checkpoints and final model
        epochs:        number of training epochs
        batch_size:    samples per gradient update
        lr:            learning rate
        val_fraction:  fraction of data held out for validation
        max_records:   cap on training records (None = all)
        resume:        path to checkpoint to resume from
        device:        'cuda', 'cpu', or None (auto-detect)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*55}")
    print(f"  Can't Stop Neural Network Training")
    print(f"{'='*55}")
    print(f"  Device:      {device}")
    if device == 'cuda':
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  LR:          {lr}")
    print(f"  Output:      {output_dir}")
    print(f"{'='*55}\n")

    # ---- LOAD DATA ----
    dataset = CantStopDataset(data_paths, max_records=max_records)

    # Train/validation split
    val_size   = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == 'cuda'),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == 'cuda'),
    )

    print(f"  Training samples:   {train_size:,}")
    print(f"  Validation samples: {val_size:,}\n")

    # ---- MODEL ----
    model = CantStopNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )

    start_epoch = 1
    best_val_loss = float('inf')
    history = []

    # Resume from checkpoint if provided
    if resume:
        start_epoch, _ = load_checkpoint(model, optimizer, resume, device)
        start_epoch += 1

    # ---- TRAINING LOOP ----
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 45)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        # Learning rate schedule
        scheduler.step(val_metrics['loss'])

        epoch_time = time.time() - epoch_start

        # Print metrics
        print(f"  Train loss:    {train_metrics['loss']:.4f}"
              f"  (value: {train_metrics['value_loss']:.4f}"
              f"  policy: {train_metrics['policy_loss']:.4f})")
        print(f"  Val loss:      {val_metrics['loss']:.4f}"
              f"  (value: {val_metrics['value_loss']:.4f}"
              f"  policy: {val_metrics['policy_loss']:.4f})")
        print(f"  Policy acc:    train={train_metrics['policy_acc']:.3f}"
              f"  val={val_metrics['policy_acc']:.3f}")
        print(f"  Value MAE:     {val_metrics['value_mae']:.4f}")
        print(f"  Time:          {epoch_time:.1f}s")

        # Save history
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val':   val_metrics,
            'time':  epoch_time,
        })

        # Save checkpoint every epoch
        ckpt_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch:03d}.pt')
        save_checkpoint(model, optimizer, epoch,
                       {'train': train_metrics, 'val': val_metrics},
                       ckpt_path)

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = os.path.join(output_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch,
                           {'train': train_metrics, 'val': val_metrics},
                           best_path)
            print(f"  ★ New best model! Val loss: {best_val_loss:.4f}")

    # ---- SAVE FINAL MODEL ----
    final_path = os.path.join(output_dir, f'final_model_{timestamp}.pt')
    save_checkpoint(model, optimizer, epochs,
                   history[-1], final_path)

    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best model:    {os.path.join(output_dir, 'best_model.pt')}")
    print(f"{'='*55}\n")

    return model, history


# ---- ENTRY POINT ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Can't Stop neural network")
    parser.add_argument("--data",    nargs="+", required=True,
                        help="Path(s) to .jsonl training data files")
    parser.add_argument("--epochs",  type=int,   default=20)
    parser.add_argument("--batch",   type=int,   default=512)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--val",     type=float, default=0.1,
                        help="Validation fraction (default 0.1)")
    parser.add_argument("--records", type=int,   default=None,
                        help="Max records to load (default: all)")
    parser.add_argument("--output",  type=str,   default="models/cantstop")
    parser.add_argument("--resume",  type=str,   default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device",  type=str,   default=None,
                        help="cuda or cpu (default: auto)")
    args = parser.parse_args()

    train(
        data_paths=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        val_fraction=args.val,
        max_records=args.records,
        resume=args.resume,
        device=args.device,
    )