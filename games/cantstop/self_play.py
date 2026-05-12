# self_play.py
# MCTS self-play training loop for Can't Stop.
#
# Key design:
#   - MCTS provides training targets (visit counts + values)
#   - Workers generate games on CPU in parallel
#   - Main process trains on GPU
#   - Replay buffer prevents catastrophic forgetting
#   - Always-accept-with-floor acceptance (>=45% vs current)
#   - Best model checkpoint maintained at a stable path
#   - Evaluation uses MCTS at low sim count (matches deployment behavior)

import os
import sys
import time
import random
import argparse
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
import multiprocessing as mp

import torch
import torch.nn.functional as F

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
from games.cantstop.evaluate import load_model


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

# Module-level counter for assigning unique game IDs from a single worker.
# Each worker process maintains its own counter; we combine with worker
# seed to make IDs globally unique across the pool.
_worker_game_counter = 0
_worker_id_prefix = 0


def play_mcts_game(mcts, num_simulations=20, global_temp_mult=1.0,
                   game_id=None):
    """
    Play one complete self-play game using MCTS for decisions.

    For each position:
      1. Run MCTS (guided by network)
      2. Record: features, MCTS visit counts (policy target),
                 MCTS value estimate (value target)
      3. Sample action from MCTS distribution with temperature

    Returns list of labeled training records, each tagged with
    `game_id` so the trainer can split train/val by game (avoids
    correlated-position leakage across the split).
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

        # MCTS search — this is the key training signal.
        action_idx, move, decision, mcts_policy, mcts_value = \
            mcts.get_action(
                state,
                num_simulations=num_simulations,
                temperature=temp
            )

        # Record position with MCTS targets.
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
            'game_id':     game_id,
        })

        step_index += 1

        apply_move(state, move)
        if decision == "stop":
            stop_turn(state)
            state.dice = []
        else:
            # CONTINUE → the current dice are spent. Clear them so the
            # next iteration rolls fresh. (Original code left dice set,
            # which made the next iteration's get_valid_moves operate on
            # stale dice — a silent correctness bug.)
            state.dice = []

    # Fill in value targets — blend MCTS estimate with final outcome.
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
# Must be top-level for multiprocessing pickling on Windows.

_worker_mcts = None


def _init_worker(model_path, iteration_seed):
    """
    Initialize worker process with CPU-based model and MCTS.
    Called once per worker — avoids reloading for each batch.
    Workers use CPU to avoid CUDA multiprocessing issues on Windows.
    Network inference is only 3% of time so CPU is fine.

    Sets up a per-worker game_id prefix combining the iteration seed
    (varies across iterations) with os.getpid() (varies across workers
    within a pool). Together these give unique game IDs across all
    games ever generated.
    """
    global _worker_mcts, _worker_game_counter, _worker_id_prefix
    worker_device = 'cpu'
    model = load_model(model_path, worker_device)
    _worker_mcts = MCTS(model, worker_device)
    _worker_game_counter = 0
    # Combine iteration seed (16 bits) + pid (16 bits) for the prefix.
    # The counter occupies the low 32 bits, leaving room for ~4B games
    # per worker per iteration before overflow (way beyond practical).
    iter_bits = (int(iteration_seed) & 0xFFFF) << 16
    pid_bits  = (os.getpid() & 0xFFFF)
    _worker_id_prefix = iter_bits | pid_bits


def _worker_generate_batch(args):
    """
    Generate a batch of MCTS games in a worker process.
    Uses pre-initialized _worker_mcts instance.
    Returns [] on any error so the pool doesn't hang overnight.
    """
    global _worker_game_counter
    num_games, num_simulations, temp_mult, seed = args
    random.seed(seed)
    np.random.seed(seed)

    all_records = []
    for i in range(num_games):
        try:
            # Assign a globally-unique game_id.
            # Prefix is iteration_seed[16] | pid[16] in upper 32 bits;
            # counter occupies lower 32 bits.
            game_id = (_worker_id_prefix << 32) | (_worker_game_counter & 0xFFFFFFFF)
            _worker_game_counter += 1

            records = play_mcts_game(
                _worker_mcts,
                num_simulations=num_simulations,
                global_temp_mult=temp_mult,
                game_id=game_id,
            )
            all_records.extend(records)
        except Exception as e:
            import traceback
            print(f"  [worker] Game {i+1}/{num_games} failed: {e}", flush=True)
            traceback.print_exc()
            continue  # skip bad game, keep going

    return all_records


def generate_games_parallel(model_path, num_games, num_simulations,
                             temp_mult, num_workers=None,
                             iteration_seed=None):
    """
    Generate MCTS games across multiple CPU worker processes.
    Each worker has its own model and MCTS instance.
    Main process keeps GPU for training.

    `iteration_seed` ensures different iterations produce different
    dice sequences. The original code used a fixed seed=42 inside this
    function, which meant every iteration generated statistically
    identical games — a silent diversity bug. We now thread the
    iteration through so each iteration explores fresh randomness.
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)

    # Distribute games evenly
    games_per_worker = num_games // num_workers
    leftover = num_games % num_workers

    if iteration_seed is None:
        # Non-reproducible default (time-based) — better than fixed 42.
        iteration_seed = int(time.time() * 1000) & 0x7FFFFFFF

    rng = random.Random(iteration_seed)
    batch_args = []
    for i in range(num_workers):
        n = games_per_worker + (1 if i < leftover else 0)
        if n > 0:
            seed = rng.randint(0, 2**31 - 1)
            batch_args.append((n, num_simulations, temp_mult, seed))

    actual_workers = len(batch_args)

    # Each worker initializer needs the model path AND its own seed
    # for the game_id prefix. mp.Pool can only pass one initargs tuple
    # to ALL workers, so we use a trick: pass model_path globally and
    # let each worker derive its seed from its first batch_args entry.
    # Simpler: pass a wrapper that uses the seed from os.getpid + time.
    # We use os.getpid to give each worker a stable, unique prefix.
    with mp.Pool(
        processes=actual_workers,
        initializer=_init_worker,
        initargs=(model_path, iteration_seed),
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
        # Convert to list once for O(1) indexing during sampling
        # (deque indexing is O(N), which would make random.sample
        # quadratic). The copy is shallow — only references move.
        return random.sample(list(self.buffer), n)

    def to_tensors(self, records):
        """
        Convert records to training tensors.

        Vectorized: stacks numpy arrays into a single buffer per field,
        then converts to a tensor in one shot. The original record-by-
        record loop was 10x+ slower at 100k+ records because each call
        to torch.tensor allocates and copies individually.
        """
        n = len(records)
        if n == 0:
            empty_f = torch.zeros(0, FEATURE_SIZE,  dtype=torch.float32)
            empty_m = torch.zeros(0, ACTION_SPACE,  dtype=torch.bool)
            empty_p = torch.zeros(0, ACTION_SPACE,  dtype=torch.float32)
            empty_v = torch.zeros(0,                dtype=torch.float32)
            empty_a = torch.zeros(0,                dtype=torch.long)
            return empty_f, empty_m, empty_p, empty_v, empty_a

        # Pre-allocate numpy buffers — single allocation per field.
        features_np      = np.empty((n, FEATURE_SIZE), dtype=np.float32)
        masks_np         = np.empty((n, ACTION_SPACE), dtype=np.bool_)
        policies_np      = np.empty((n, ACTION_SPACE), dtype=np.float32)
        values_np        = np.empty(n,                 dtype=np.float32)
        actions_np       = np.empty(n,                 dtype=np.int64)

        for i, rec in enumerate(records):
            features_np[i]  = rec['features']
            masks_np[i]     = rec['mask']
            policies_np[i]  = rec['mcts_policy']
            values_np[i]    = rec['value_target']
            actions_np[i]   = rec['action_idx']

        # from_numpy avoids the extra copy that torch.tensor does.
        features_t       = torch.from_numpy(features_np)
        masks_t          = torch.from_numpy(masks_np)
        policy_targets_t = torch.from_numpy(policies_np)
        value_targets_t  = torch.from_numpy(values_np)
        action_idx_t     = torch.from_numpy(actions_np)

        return features_t, masks_t, policy_targets_t, value_targets_t, action_idx_t


# ---- LOSS FUNCTIONS ----

def policy_loss_mcts(logits, policy_targets, masks):
    """
    Cross entropy loss with MCTS visit count targets.
    Numerically stable — guards against all-illegal rows.
    Includes entropy regularization to prevent policy collapse.
    """
    # Guard against all-illegal rows.
    valid_rows = masks.any(dim=1)
    if not valid_rows.all():
        logits         = logits[valid_rows]
        policy_targets = policy_targets[valid_rows]
        masks          = masks[valid_rows]

    if masks.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    masked_logits = logits.masked_fill(~masks, -1e9)
    log_probs     = F.log_softmax(masked_logits, dim=-1).clamp(min=-100)

    # Normalize targets over legal actions.
    targets  = policy_targets * masks.float()
    row_sums = targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    targets  = targets / row_sums

    ce_loss = -(targets * log_probs).sum(dim=-1).mean()

    # Entropy regularization.
    probs   = F.softmax(masked_logits, dim=-1).clamp(min=1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()

    return ce_loss - 0.01 * entropy


def combined_loss_mcts(value_pred, value_target, logits,
                        policy_targets, masks):
    """Combined value + MCTS policy loss."""
    # Ensure value_pred has same shape as value_target. Model may
    # return (B, 1) or (B,) depending on implementation; we squeeze
    # defensively to avoid silent broadcasting in BCE.
    if value_pred.dim() > value_target.dim():
        value_pred = value_pred.squeeze(-1)
    v_loss = F.binary_cross_entropy(value_pred, value_target)
    p_loss = policy_loss_mcts(logits, policy_targets, masks)
    total  = v_loss + p_loss
    return total, v_loss, p_loss


# ---- TRAIN/VAL SPLIT (BY GAME) ----

def split_records_by_game(records, val_frac=0.1):
    """
    Split records into train/val without leaking positions from the
    same game across the split.

    Records lacking a game_id (legacy) are treated as one unique game
    each, falling back to per-record split — equivalent to the old
    behavior. New records always have a real game_id.
    """
    by_game = defaultdict(list)
    no_id_records = []
    for rec in records:
        gid = rec.get('game_id')
        if gid is None:
            no_id_records.append(rec)
        else:
            by_game[gid].append(rec)

    game_ids = list(by_game.keys())
    random.shuffle(game_ids)

    # Ensure at least one game in val when we have at least 2 games.
    # int(N * val_frac) rounds to 0 for small N, which left val empty.
    if len(game_ids) >= 2:
        n_val_games = max(1, int(len(game_ids) * val_frac))
    else:
        n_val_games = 0
    val_game_ids = set(game_ids[:n_val_games])

    train_records = []
    val_records = []
    for gid, recs in by_game.items():
        if gid in val_game_ids:
            val_records.extend(recs)
        else:
            train_records.extend(recs)

    # Records without a game_id: shuffle and split per-record.
    if no_id_records:
        random.shuffle(no_id_records)
        n_no_id_val = int(len(no_id_records) * val_frac)
        val_records.extend(no_id_records[:n_no_id_val])
        train_records.extend(no_id_records[n_no_id_val:])

    return train_records, val_records


# ---- TRAIN ON BUFFER ----

def _estimate_gpu_data_bytes(n_records):
    """Estimate per-split GPU memory for tensorized data (rough)."""
    # features (f32) + mask (bool) + policy (f32) + value (f32) + action (i64)
    per_record = (FEATURE_SIZE * 4) + ACTION_SPACE + (ACTION_SPACE * 4) + 4 + 8
    return n_records * per_record


def _shuffle_indices(n, device):
    """Random permutation of [0..n) on the given device."""
    return torch.randperm(n, device=device)


def train_on_buffer(model, replay_buffer, device,
                    epochs=5, batch_size=512, lr=3e-5,
                    sample_size=None,
                    gpu_data_cap_bytes=2_000_000_000):
    """
    Train model on sample from replay buffer using MCTS targets.

    Performance pattern (works at any scale):
      - Tensorize sampled records ONCE per training call.
      - Transfer all training+val tensors to `device` ONCE.
      - Iterate batches via index slicing into on-device tensors
        (no DataLoader, no per-batch host→device copies).
      - Fresh random permutation each epoch for shuffling.

    Falls back to per-batch transfer if estimated GPU footprint exceeds
    `gpu_data_cap_bytes` (default 2 GB). At 100k records / Can't Stop
    feature size this is ~108 MB so the fast path is always used; for
    larger games with bigger feature tensors the fallback protects
    against OOM.

    Other improvements vs. the original:
      - Vectorized tensorization (np.stack-then-from_numpy).
      - Train/val split by GAME ID — no correlated-position leakage.
      - Policy accuracy compares NN argmax to MCTS argmax (the policy
        training target), not to the sampled action.
    """
    if sample_size and replay_buffer.size() > sample_size:
        records = replay_buffer.sample(sample_size)
    else:
        records = list(replay_buffer.buffer)

    # ---- Split by game BEFORE tensorizing ----
    tr_records, vl_records = split_records_by_game(records, val_frac=0.1)
    tr_size = len(tr_records)
    vl_size = len(vl_records)

    # Tensorize each split independently (CPU-side, single allocation).
    tr_feat, tr_mask, tr_pol, tr_val, tr_act = \
        replay_buffer.to_tensors(tr_records)
    vl_feat, vl_mask, vl_pol, vl_val, vl_act = \
        replay_buffer.to_tensors(vl_records)

    # ---- Decide: all-on-device fast path, or per-batch fallback ----
    est_bytes = _estimate_gpu_data_bytes(tr_size + vl_size)
    use_fast_path = (est_bytes <= gpu_data_cap_bytes)

    if use_fast_path:
        # Single host→device transfer of the entire training+val data.
        tr_feat = tr_feat.to(device); tr_mask = tr_mask.to(device)
        tr_pol  = tr_pol.to(device);  tr_val  = tr_val.to(device)
        tr_act  = tr_act.to(device)
        vl_feat = vl_feat.to(device); vl_mask = vl_mask.to(device)
        vl_pol  = vl_pol.to(device);  vl_val  = vl_val.to(device)
        vl_act  = vl_act.to(device)
        path_note = "on-device"
    else:
        path_note = (f"per-batch (data ~{est_bytes/1e9:.1f} GB > "
                     f"cap {gpu_data_cap_bytes/1e9:.1f} GB)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n  Training on {tr_size:,} records "
          f"(val: {vl_size:,} | buffer: {replay_buffer.size():,}) "
          f"for {epochs} epochs [{path_note}]...")

    best_val_loss = float('inf')
    best_state    = None

    def _move_batch(batch_tensors):
        """Move a batch of (possibly CPU) tensors to device."""
        return tuple(t.to(device, non_blocking=True) for t in batch_tensors)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        nan_batches = total_batches = 0

        # Fresh shuffle each epoch. On-device permutation if fast path,
        # CPU permutation if fallback.
        if use_fast_path:
            perm = _shuffle_indices(tr_size, device=device)
        else:
            perm = _shuffle_indices(tr_size, device='cpu')

        for start in range(0, tr_size, batch_size):
            end = min(start + batch_size, tr_size)
            idx = perm[start:end]

            feat = tr_feat[idx]
            msk  = tr_mask[idx]
            pol  = tr_pol[idx]
            val_t = tr_val[idx]
            act  = tr_act[idx]

            if not use_fast_path:
                feat, msk, pol, val_t, act = _move_batch(
                    (feat, msk, pol, val_t, act)
                )

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

            bs = end - start
            total_loss    += loss.item() * bs
            total_samples += bs

        if nan_batches > 0:
            nan_pct = nan_batches / max(total_batches, 1) * 100
            warn = "⚠️  WARNING: high NaN rate" if nan_pct > 1.0 else "note"
            print(f"    [{warn}] Skipped {nan_batches}/{total_batches} "
                  f"batches ({nan_pct:.1f}%) due to NaN loss/grads")

        # ---- Validation ----
        model.eval()
        vl_loss_total = 0.0
        vl_correct_vs_mcts = 0
        vl_total = 0
        entropy_sum = 0.0

        with torch.no_grad():
            for start in range(0, vl_size, batch_size):
                end = min(start + batch_size, vl_size)

                feat  = vl_feat[start:end]
                msk   = vl_mask[start:end]
                pol   = vl_pol[start:end]
                val_t = vl_val[start:end]
                act   = vl_act[start:end]

                if not use_fast_path:
                    feat, msk, pol, val_t, act = _move_batch(
                        (feat, msk, pol, val_t, act)
                    )

                val_pred, logits = model(feat, msk)
                loss, _, _ = combined_loss_mcts(
                    val_pred, val_t, logits, pol, msk
                )
                bs = end - start
                vl_loss_total += loss.item() * bs

                masked_logits = logits.masked_fill(~msk, -1e9)
                nn_preds      = masked_logits.argmax(1)

                # Policy match: NN argmax vs MCTS argmax (the policy
                # training target — not the sampled action).
                mcts_preds = pol.argmax(1)
                vl_correct_vs_mcts += (nn_preds == mcts_preds).sum().item()

                vl_total += bs

                probs   = F.softmax(masked_logits, dim=-1).clamp(min=1e-8)
                entropy = -(probs * probs.log()).sum(dim=-1).mean()
                entropy_sum += entropy.item() * bs

        tr_loss     = total_loss    / max(total_samples, 1)
        vl_loss     = vl_loss_total / max(vl_total, 1)
        vl_acc      = vl_correct_vs_mcts / max(vl_total, 1)
        avg_entropy = entropy_sum   / max(vl_total, 1)

        print(f"    Epoch {epoch}/{epochs} | "
              f"Train: {tr_loss:.4f} | "
              f"Val: {vl_loss:.4f} | "
              f"Policy match (vs MCTS argmax): {vl_acc:.3f} | "
              f"Entropy: {avg_entropy:.3f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state    = {k: v.detach().clone()
                             for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    return best_val_loss


# ---- EVALUATION (MCTS vs MCTS) ----

def _play_mcts_eval_game(mcts_p0, mcts_p1, num_simulations,
                         max_turns=200):
    """
    Play one game between two MCTS instances. Returns winner (0 or 1)
    or None if the game hit max_turns without a winner.

    Both MCTS instances use temperature=0 (greedy at the policy level —
    pick the most-visited move). This is the correct deployment-style
    comparison: no exploration noise, just "which model's search is
    stronger?"
    """
    state = GameState(2)
    for _ in range(max_turns):
        if state.game_over:
            return state.winner

        if not state.dice:
            state.roll_dice()

        valid = get_valid_moves(state)
        if not valid:
            bust_turn(state)
            state.dice = []
            continue

        active = state.active_player
        searcher = mcts_p0 if active == 0 else mcts_p1

        _, move, decision, _, _ = searcher.get_action(
            state,
            num_simulations=num_simulations,
            temperature=0.0,
        )

        apply_move(state, move)
        if decision == "stop":
            stop_turn(state)
            state.dice = []
        else:
            # CONTINUE → spend the current roll, force a fresh roll.
            state.dice = []

    return None  # draw / timeout


# ---- Parallel eval worker ----
# Worker globals (one set per process).
_eval_mcts_new = None
_eval_mcts_old = None


def _init_eval_worker(new_model_path, old_model_path):
    """Load both models on CPU, build an MCTS for each."""
    global _eval_mcts_new, _eval_mcts_old
    new_model = load_model(new_model_path, 'cpu')
    old_model = load_model(old_model_path, 'cpu')
    _eval_mcts_new = MCTS(new_model, 'cpu')
    _eval_mcts_old = MCTS(old_model, 'cpu')


def _eval_worker_run(args):
    """
    Play a chunk of eval games. Half with new as P0, half with old as P0.
    Returns (wins_new, wins_old, draws).
    """
    n_games, num_simulations, seed = args
    random.seed(seed)
    np.random.seed(seed)

    wins_new = wins_old = draws = 0
    for i in range(n_games):
        try:
            # Alternate within the worker so each worker is self-balanced.
            if i % 2 == 0:
                winner = _play_mcts_eval_game(
                    _eval_mcts_new, _eval_mcts_old, num_simulations
                )
                if winner == 0:   wins_new += 1
                elif winner == 1: wins_old += 1
                else:             draws    += 1
            else:
                winner = _play_mcts_eval_game(
                    _eval_mcts_old, _eval_mcts_new, num_simulations
                )
                if winner == 0:   wins_old += 1
                elif winner == 1: wins_new += 1
                else:             draws    += 1
        except Exception as e:
            import traceback
            print(f"  [eval worker] game {i} failed: {e}", flush=True)
            traceback.print_exc()
            # Skip the failed game — it's safer to under-count than crash.
            continue

    return wins_new, wins_old, draws


def _save_model_to_tmp(model, output_dir, name='_eval_new_tmp.pt'):
    """Write a model's state dict to a temp file for worker loading."""
    tmp_path = os.path.join(output_dir, name)
    torch.save({'model_state': model.state_dict()}, tmp_path)
    return tmp_path


def evaluate_networks(new_model, old_model_path, num_games=500,
                      eval_sims=20, num_workers=None,
                      output_dir='.'):
    """
    Evaluate new vs old via MCTS-vs-MCTS at eval_sims simulations,
    using a worker pool.

    Each worker loads both models from disk (CPU), builds an MCTS
    instance for each, and plays a chunk of games alternating which
    model is P0 (to remove first-mover bias). Workers report
    (wins_new, wins_old, draws); main aggregates.

    Why parallel:
      - Sequential eval at 500 games × ~3s = ~25 min was the iteration
        bottleneck. With 8 workers that's ~3 min.
      - Allows tighter measurement (more games at the same wallclock
        cost) → tighter acceptance criteria possible.

    Args:
      new_model: in-memory model object (will be saved to tmp file
                 so workers can load it).
      old_model_path: path to the previous accepted model on disk.
      num_games: total eval games (split across workers).
      eval_sims: MCTS sims per move during eval.
      num_workers: defaults to min(cpu_count(), 8).
      output_dir: where to write the temporary new-model checkpoint.

    Returns:
      win_rate of the new model (draws excluded from denominator).
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)

    # Stash the new model to disk so workers can load it.
    new_model_tmp_path = _save_model_to_tmp(new_model, output_dir)

    # Split games across workers. Each worker gets an even number where
    # possible so its internal P0-alternation stays balanced.
    games_per_worker = num_games // num_workers
    leftover = num_games % num_workers
    seed_rng = random.Random(int(time.time() * 1000) & 0x7FFFFFFF)
    batch_args = []
    for i in range(num_workers):
        n = games_per_worker + (1 if i < leftover else 0)
        if n > 0:
            batch_args.append((n, eval_sims, seed_rng.randint(0, 2**31 - 1)))
    actual_workers = len(batch_args)

    print(f"\n  Evaluating new vs old "
          f"({num_games} MCTS games @ {eval_sims} sims, "
          f"{actual_workers} workers)...")

    eval_start = time.time()
    with mp.Pool(
        processes=actual_workers,
        initializer=_init_eval_worker,
        initargs=(new_model_tmp_path, old_model_path),
    ) as pool:
        results = pool.map(_eval_worker_run, batch_args)
    eval_elapsed = time.time() - eval_start

    wins_new = sum(r[0] for r in results)
    wins_old = sum(r[1] for r in results)
    draws    = sum(r[2] for r in results)

    total    = wins_new + wins_old
    win_rate = wins_new / total if total > 0 else 0.5

    print(f"  Eval done in {eval_elapsed:.1f}s "
          f"({num_games/max(eval_elapsed,1e-6):.1f} games/s aggregate)")
    print(f"  New model: {wins_new}/{total} ({win_rate:.1%})  "
          f"draws: {draws}")
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
    eval_sims=20,
    accept_floor=0.50,
    buffer_size=200_000,
    sample_size=100_000,
    initial_temp_mult=1.0,
    final_temp_mult=0.7,
    num_workers=None,
    device=None,
):
    """
    Main training loop.

    Acceptance policy: always-accept-with-floor.
      - win_rate >= accept_floor (default 0.50) → accept new model
      - win_rate <  accept_floor              → reject (regression)

    The previous >55% threshold rejected genuinely-improving models
    because greedy eval (no MCTS) was a noisy proxy for the wrong
    capability — search-guidance quality. With MCTS eval at low sim
    count, win_rate ≈ 0.50 against the same baseline genuinely
    indicates a comparable model; we accept rather than stall.

    Floor of 0.50 (rather than e.g. 0.45) means we accept ties — which
    avoids the "stuck at 50-52%" failure mode — but never accept a
    losing record. Anything below 50% is more likely to be a real
    regression than a different-but-equal model.

    The strongest accepted model is always written to
    `<output_dir>/best_model.pt` after every acceptance, so a crash
    or stop can never lose the best model.
    """
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
    print(f"  Eval games:    {eval_games} @ {eval_sims} sims (CPU MCTS)")
    print(f"  Accept floor:  {accept_floor:.0%}")
    print(f"{'='*55}\n")

    # Load model — track path for workers.
    current_model      = load_model(initial_model_path, device)
    current_model_path = initial_model_path

    # Stable "best" checkpoint path — overwritten on every acceptance.
    best_model_path = os.path.join(output_dir, 'best_model.pt')

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

        # Per-iteration seed ensures different games each iteration.
        iter_seed = (int(time.time() * 1000) ^ (iteration * 2654435761)) & 0x7FFFFFFF

        all_new_records = generate_games_parallel(
            model_path=current_model_path,
            num_games=games_per_iter,
            num_simulations=num_simulations,
            temp_mult=temp_mult,
            num_workers=num_workers,
            iteration_seed=iter_seed,
        )

        gen_time = time.time() - gen_start
        print(f"  Generated {len(all_new_records):,} records "
              f"in {gen_time:.1f}s "
              f"({games_per_iter/max(gen_time, 1e-6):.2f} games/s)")

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

        # ---- STAGE 3: EVALUATE (MCTS vs MCTS, parallel workers) ----
        win_rate = evaluate_networks(
            new_model,
            old_model_path=current_model_path,
            num_games=eval_games,
            eval_sims=eval_sims,
            num_workers=num_workers,
            output_dir=output_dir,
        )

        iter_time = time.time() - iter_start

        # ---- ACCEPT OR REJECT ----
        if win_rate >= accept_floor:
            print(f"\n  ✓ ACCEPTED ({win_rate:.1%} >= "
                  f"{accept_floor:.0%})")
            current_model = new_model

            save_path = os.path.join(
                output_dir,
                f'model_iter_{iteration:03d}_accepted.pt'
            )
            checkpoint = {
                'iteration':   iteration,
                'win_rate':    float(win_rate),
                'val_loss':    float(val_loss),
                'temp_mult':   float(temp_mult),
                'model_state': new_model.state_dict(),
            }
            torch.save(checkpoint, save_path)

            # ALSO save to the stable best_model.pt path, so the
            # strongest accepted model is always at a known location
            # regardless of crashes or stops.
            torch.save(checkpoint, best_model_path)

            current_model_path = save_path  # workers use new model
            accepted += 1
            print(f"  Saved iteration checkpoint: {save_path}")
            print(f"  Updated best model:         {best_model_path}")

        else:
            print(f"\n  ✗ REJECTED ({win_rate:.1%} < "
                  f"{accept_floor:.0%}) — keeping current model")
            rejected += 1

        history.append({
            'iteration':   iteration,
            'win_rate':    float(win_rate),
            'val_loss':    float(val_loss),
            'temp_mult':   float(temp_mult),
            'buffer_size': replay_buffer.size(),
            'accepted':    bool(win_rate >= accept_floor),
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
    print(f"  Best model:  {best_model_path}")

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
    parser.add_argument("--eval-sims",  type=int,   default=20,
                        dest="eval_sims",
                        help="MCTS simulations per move during eval")
    parser.add_argument("--floor",      type=float, default=0.50,
                        help="Win-rate floor for acceptance "
                             "(below this, reject). Default 0.50 "
                             "accepts ties; never accepts a losing "
                             "record.")
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
        eval_sims=args.eval_sims,
        accept_floor=args.floor,
        buffer_size=args.buffer,
        sample_size=args.sample,
        initial_temp_mult=args.temp_start,
        final_temp_mult=args.temp_end,
        num_workers=args.workers,
        device=args.device,
    )