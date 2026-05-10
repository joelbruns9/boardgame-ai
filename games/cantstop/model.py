# model.py
# Neural network for Can't Stop.
#
# Architecture:
#   - Shared column encoder (processes each column's 4 features identically)
#   - Dice encoder (processes dice availability features)
#   - Context encoder (processes game context features)
#   - Trunk (combines all encodings)
#   - Value head (predicts win probability)
#   - Policy head (predicts action distribution over 154 actions)
#
# This is an AlphaZero-style dual-head network adapted for Can't Stop.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.features import (
    FEATURE_SIZE, ACTION_SPACE, NUM_COLUMNS,
    extract_features, get_legal_action_mask,
    get_feature_names, move_to_action, action_to_move_decision
)


# ---- ARCHITECTURE CONSTANTS ----
# Tunable without changing the overall design

COL_FEATURES    = 4    # features per column (claimed, saved, runner, opp_saved)
DICE_FEATURES   = 2    # features per column from dice (multiplicity, roll_freq)
CONTEXT_FEATURES = 8   # game context features

COL_EMBED_SIZE   = 16  # output size of shared column encoder
DICE_EMBED_SIZE  = 16  # output size of dice encoder
CONTEXT_EMBED    = 16  # output size of context encoder

# Trunk input = all embeddings concatenated
TRUNK_INPUT = (NUM_COLUMNS * COL_EMBED_SIZE) + \
              (NUM_COLUMNS * DICE_EMBED_SIZE) + \
              CONTEXT_EMBED

TRUNK_HIDDEN = 256     # trunk hidden layer size
VALUE_HIDDEN = 64      # value head hidden size
POLICY_HIDDEN = 128    # policy head hidden size


# ---- SHARED COLUMN ENCODER ----
class ColumnEncoder(nn.Module):
    """
    Processes each column's features through the same small network.
    Shared weights mean the network learns one universal
    'what does column state mean?' function.

    Input:  (batch, num_columns, col_features) = (B, 11, 4)
    Output: (batch, num_columns, embed_size)   = (B, 11, 16)
    """
    def __init__(self, in_features=COL_FEATURES, embed_size=COL_EMBED_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, embed_size),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch, num_columns, col_features)
        # Apply same network to each column independently
        return self.net(x)


# ---- DICE ENCODER ----
class DiceEncoder(nn.Module):
    """
    Processes dice availability features per column.
    Separate from column encoder since dice features have
    different semantics (current roll vs board state).

    Input:  (batch, num_columns, dice_features) = (B, 11, 2)
    Output: (batch, num_columns, embed_size)    = (B, 11, 16)
    """
    def __init__(self, in_features=DICE_FEATURES, embed_size=DICE_EMBED_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, embed_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# ---- CONTEXT ENCODER ----
class ContextEncoder(nn.Module):
    """
    Processes scalar game context features.

    Input:  (batch, context_features) = (B, 8)
    Output: (batch, embed_size)       = (B, 16)
    """
    def __init__(self, in_features=CONTEXT_FEATURES, embed_size=CONTEXT_EMBED):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, embed_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# ---- TRUNK ----
class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TRUNK_INPUT, TRUNK_HIDDEN),
            nn.LayerNorm(TRUNK_HIDDEN),
            nn.ReLU(),
            nn.Linear(TRUNK_HIDDEN, TRUNK_HIDDEN),
            nn.LayerNorm(TRUNK_HIDDEN),
            nn.ReLU(),
            nn.Linear(TRUNK_HIDDEN, TRUNK_HIDDEN // 2),
            nn.LayerNorm(TRUNK_HIDDEN // 2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# ---- VALUE HEAD ----
class ValueHead(nn.Module):
    """
    Predicts win probability from the shared representation.

    Output: scalar in [0, 1] — probability of winning
    Loss:   binary cross entropy against actual outcome
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TRUNK_HIDDEN // 2, VALUE_HIDDEN),
            nn.ReLU(),
            nn.Linear(VALUE_HIDDEN, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,)


# ---- POLICY HEAD ----
class PolicyHead(nn.Module):
    """
    Predicts action distribution over all 154 (move, decision) pairs.

    Output: logits of shape (B, 154) — softmax applied after masking
    Loss:   cross entropy against chosen action

    Masking: illegal actions are set to -inf before softmax
    so they get zero probability.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TRUNK_HIDDEN // 2, POLICY_HIDDEN),
            nn.ReLU(),
            nn.Linear(POLICY_HIDDEN, ACTION_SPACE),
        )

    def forward(self, x):
        return self.net(x)  # (B, 154) — raw logits


# ---- FULL MODEL ----
class CantStopNet(nn.Module):
    """
    Complete Can't Stop neural network.

    Architecture:
        Column features → ColumnEncoder (shared weights across 11 cols)
        Dice features   → DiceEncoder   (shared weights across 11 cols)
        Context         → ContextEncoder
        All embeddings  → Trunk → shared representation
                               ├→ ValueHead  → win probability
                               └→ PolicyHead → action logits

    Usage:
        model = CantStopNet()
        value, policy_logits = model(features, action_mask)

        # value: (B,) win probabilities
        # policy_logits: (B, 154) masked action logits
    """
    def __init__(self):
        super().__init__()

        self.col_encoder     = ColumnEncoder()
        self.dice_encoder    = DiceEncoder()
        self.context_encoder = ContextEncoder()
        self.trunk           = Trunk()
        self.value_head      = ValueHead()
        self.policy_head     = PolicyHead()

    def forward(self, features, action_mask=None):
        """
        Parameters:
            features:    (B, 74) float tensor — extracted features
            action_mask: (B, 154) bool tensor — True = legal action
                         If None, all actions treated as legal.

        Returns:
            value:         (B,) float tensor — win probabilities
            policy_logits: (B, 154) float tensor — masked logits
        """
        B = features.shape[0]

        # ---- SPLIT FEATURES ----
        # Column features: indices 0-43  → reshape to (B, 11, 4)
        col_feats = features[:, :44].reshape(B, NUM_COLUMNS, COL_FEATURES)

        # Dice features: indices 44-65 → reshape to (B, 11, 2)
        dice_feats = features[:, 44:66].reshape(B, NUM_COLUMNS, DICE_FEATURES)

        # Context features: indices 66-73 → (B, 8)
        ctx_feats = features[:, 66:74]

        # ---- ENCODE ----
        col_embed  = self.col_encoder(col_feats)     # (B, 11, 16)
        dice_embed = self.dice_encoder(dice_feats)   # (B, 11, 16)
        ctx_embed  = self.context_encoder(ctx_feats) # (B, 16)

        # ---- FLATTEN AND CONCATENATE ----
        col_flat  = col_embed.reshape(B, -1)   # (B, 176)
        dice_flat = dice_embed.reshape(B, -1)  # (B, 176)

        combined = torch.cat([col_flat, dice_flat, ctx_embed], dim=1)
        # combined: (B, 176 + 176 + 16) = (B, 368)

        # ---- TRUNK ----
        shared = self.trunk(combined)  # (B, 128)

        # ---- HEADS ----
        value = self.value_head(shared)           # (B,)
        policy_logits = self.policy_head(shared)  # (B, 154)

        # ---- MASK ILLEGAL ACTIONS ----
        if action_mask is not None:
            # Set illegal action logits to -inf so softmax gives 0
            policy_logits = policy_logits.masked_fill(~action_mask, float('-inf'))

        return value, policy_logits

    def predict(self, state, valid_moves=None, device='cpu'):
        """
        Convenience method for inference on a single game state.

        Returns:
            value: float — win probability
            action_probs: dict mapping (move, decision) → probability
        """
        self.eval()
        with torch.no_grad():
            from games.cantstop.engine import get_valid_moves

            if valid_moves is None:
                valid_moves = get_valid_moves(state)

            # Extract features
            features = extract_features(state, valid_moves)
            features_t = torch.tensor(features, dtype=torch.float32)\
                             .unsqueeze(0).to(device)

            # Build action mask
            mask = get_legal_action_mask(valid_moves)
            mask_t = torch.tensor(mask, dtype=torch.bool)\
                         .unsqueeze(0).to(device)

            # Forward pass
            value, logits = self(features_t, mask_t)

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            # Build readable output
            action_probs = {}
            for idx in mask.nonzero()[0]:
                move, decision = action_to_move_decision(int(idx))
                action_probs[(move, decision)] = float(probs[idx])

        return float(value.item()), action_probs

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---- LOSS FUNCTIONS ----

def value_loss(predicted, actual):
    """
    Binary cross entropy between predicted win probability and actual outcome.

    predicted: (B,) float tensor in [0,1]
    actual:    (B,) float tensor in {0,1}
    """
    return F.binary_cross_entropy(predicted, actual)


def policy_loss(logits, target_actions):
    """
    Cross entropy between predicted action distribution and chosen action.

    logits:         (B, 154) float tensor — raw logits (already masked)
    target_actions: (B,) long tensor — index of chosen action
    """
    return F.cross_entropy(logits, target_actions)


def combined_loss(value_pred, value_target, policy_logits, policy_target,
                  value_weight=1.0, policy_weight=1.0):
    """
    Combined value + policy loss.

    value_weight and policy_weight control the tradeoff.
    Start with 1.0 / 1.0 and tune if one head dominates.
    """
    v_loss = value_loss(value_pred, value_target)
    p_loss = policy_loss(policy_logits, policy_target)
    total = value_weight * v_loss + policy_weight * p_loss
    return total, v_loss, p_loss


# ---- SELF TEST ----
if __name__ == "__main__":
    import time
    from games.cantstop.engine import GameState, get_valid_moves

    print("\nTesting CantStopNet...\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = CantStopNet().to(device)
    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"Expected: ~100,000-200,000\n")

    # ---- Test 1: Single forward pass ----
    print("Test 1: Single forward pass")
    state = GameState(2)
    state.dice = [3, 4, 3, 4]
    valid = get_valid_moves(state)

    features = extract_features(state, valid)
    features_t = torch.tensor(features, dtype=torch.float32)\
                     .unsqueeze(0).to(device)
    mask = get_legal_action_mask(valid)
    mask_t = torch.tensor(mask, dtype=torch.bool)\
                 .unsqueeze(0).to(device)

    value, logits = model(features_t, mask_t)
    probs = F.softmax(logits, dim=-1)

    print(f"  Value output: {value.item():.4f} (expect ~0.5 random init)")
    print(f"  Policy shape: {logits.shape} (expect [1, 154])")
    print(f"  Legal action probs sum: {probs[mask_t].sum().item():.4f} (expect 1.0)")
    print(f"  Illegal action probs:   {probs[~mask_t].sum().item():.6f} (expect 0.0)")
    assert abs(probs[mask_t].sum().item() - 1.0) < 1e-5
    print("  PASS\n")

    # ---- Test 2: Batch forward pass ----
    print("Test 2: Batch forward pass (B=32)")
    batch_features = features_t.repeat(32, 1)
    batch_masks    = mask_t.repeat(32, 1)

    values, batch_logits = model(batch_features, batch_masks)
    print(f"  Values shape: {values.shape} (expect [32])")
    print(f"  Logits shape: {batch_logits.shape} (expect [32, 154])")
    assert values.shape == (32,)
    assert batch_logits.shape == (32, 154)
    print("  PASS\n")

    # ---- Test 3: Loss computation ----
    print("Test 3: Loss computation")
    value_targets  = torch.randint(0, 2, (32,)).float().to(device)
    # Use first legal action as target
    first_legal = int(mask_t[0].nonzero()[0])
    policy_targets = torch.full((32,), first_legal, dtype=torch.long).to(device)

    total, v_loss, p_loss = combined_loss(
        values, value_targets, batch_logits, policy_targets
    )
    print(f"  Total loss:  {total.item():.4f}")
    print(f"  Value loss:  {v_loss.item():.4f}")
    print(f"  Policy loss: {p_loss.item():.4f}")
    assert not torch.isnan(total)
    print("  PASS\n")

    # ---- Test 4: Predict convenience method ----
    print("Test 4: Predict method")
    win_prob, action_probs = model.predict(state, valid, device=device)
    print(f"  Win probability: {win_prob:.4f}")
    print(f"  Legal actions: {len(action_probs)}")
    print(f"  Prob sum: {sum(action_probs.values()):.4f} (expect 1.0)")
    print(f"  Top actions:")
    for (move, dec), prob in sorted(
        action_probs.items(), key=lambda x: -x[1]
    )[:4]:
        print(f"    {move} {dec}: {prob:.4f}")
    assert abs(sum(action_probs.values()) - 1.0) < 1e-5
    print("  PASS\n")

    # ---- Test 5: Gradient flow ----
    print("Test 5: Gradient flow")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    values2, logits2 = model(batch_features, batch_masks)
    loss, _, _ = combined_loss(
        values2, value_targets, logits2, policy_targets
    )
    loss.backward()

    # Check gradients exist and are finite
    grad_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    print(f"  All gradients finite: {grad_ok}")
    assert grad_ok
    print("  PASS\n")

    # ---- Test 6: Performance benchmark ----
    print("Test 6: Inference performance")
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            model(features_t, mask_t)

        start = time.time()
        for _ in range(1000):
            model(features_t, mask_t)
        elapsed = time.time() - start

    print(f"  1000 inferences: {elapsed:.3f}s")
    print(f"  Per inference: {elapsed/1000*1000:.3f}ms")
    print(f"  (Target: <1ms for real-time use)")

    if device == 'cuda':
        # GPU batch throughput
        big_batch = features_t.repeat(512, 1)
        big_mask  = mask_t.repeat(512, 1)
        with torch.no_grad():
            for _ in range(5):
                model(big_batch, big_mask)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                model(big_batch, big_mask)
            torch.cuda.synchronize()
            elapsed = time.time() - start
        throughput = 512 * 100 / elapsed
        print(f"\n  GPU batch throughput (B=512): {throughput:,.0f} samples/sec")

    print(f"\n{'='*45}")
    print(f"  All tests passed!")
    print(f"  Model ready for training.")
    print(f"{'='*45}\n")