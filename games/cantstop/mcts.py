# mcts.py
# Monte Carlo Tree Search for Can't Stop.
#
# Implements AlphaZero-style MCTS:
#   - Network provides prior probabilities (policy head)
#   - Network provides position value (value head)
#   - UCB formula balances exploration vs exploitation
#   - Visit counts become policy targets for training
#   - Average value becomes value target for training
#
# This replaces the raw network sampling in self_play.py
# with a proper tree search that corrects network mistakes.

import math
import torch
import torch.nn.functional as F
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, get_valid_moves, apply_move,
    stop_turn, bust_turn
)
from games.cantstop.features import (
    extract_features, get_legal_action_mask,
    move_to_action, action_to_move_decision,
    ACTION_SPACE
)


# ---- UCB CONSTANT ----
# Controls exploration vs exploitation tradeoff.
# Higher = more exploration. AlphaZero uses ~1.0-2.0.
C_PUCT = 1.5


# ---- MCTS NODE ----

class MCTSNode:
    """
    Represents one position in the MCTS tree.

    Each node stores:
        state:    GameState at this position
        parent:   parent MCTSNode (None for root)
        action:   action that led here from parent
        children: dict mapping action_idx → MCTSNode
        N:        visit count
        W:        total value accumulated
        Q:        mean value = W / N
        P:        prior probability from network policy head
        is_terminal: whether game is over at this node
    """

    __slots__ = [
        'state', 'parent', 'action', 'prior',
        'children', 'N', 'W',
        'valid_moves', 'mask', 'is_terminal',
        'is_expanded', 'virtual_loss',
    ]

    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state    = state
        self.parent   = parent
        self.action   = action       # action_idx that led here
        self.prior    = prior        # P(action) from parent's policy

        self.children    = {}        # action_idx → MCTSNode
        self.N           = 0         # visit count
        self.W           = 0.0       # total value
        self.is_terminal = state.game_over
        self.is_expanded = False
        self.virtual_loss = 0

        # Cache valid moves and mask
        if not self.is_terminal:
            self.valid_moves = get_valid_moves(state)
            self.mask = get_legal_action_mask(self.valid_moves)
        else:
            self.valid_moves = []
            self.mask = np.zeros(ACTION_SPACE, dtype=bool)

    @property
    def Q(self):
        """Mean value — exploitation term in UCB."""
        return self.W / self.N if self.N > 0 else 0.0

    def ucb_score(self, child_action_idx, child):
        """
        Upper Confidence Bound score for child selection.

        UCB = Q(child) + C_PUCT * P(child) * sqrt(N(parent)) / (1 + N(child))

        Q term:   exploitation — prefer high-value nodes
        P term:   prior — prefer actions network thinks are good
        N term:   exploration — prefer less-visited nodes
        """
        exploitation = child.Q
        exploration  = C_PUCT * child.prior * math.sqrt(self.N) / (1 + child.N)
        return exploitation + exploration

    def select_child(self):
        """Select child with highest UCB score."""
        return max(
            self.children.values(),
            key=lambda c: self.ucb_score(c.action, c)
        )

    def is_leaf(self):
        """Node is a leaf if it hasn't been expanded yet."""
        return not self.is_expanded


# ---- MCTS ----

class MCTS:
    """
    Monte Carlo Tree Search with batched network evaluation.
    
    Key optimization: instead of evaluating leaf nodes one at a time
    during simulation, we collect all leaves that need evaluation
    and run ONE batched forward pass per search call.
    
    This reduces 20 sequential GPU calls to 1 batched call — ~15x faster.
    """

    def __init__(self, model, device):
        self.model  = model
        self.device = device

    def evaluate_batch(self, nodes):
        """
        Evaluate multiple nodes in a single batched forward pass.
        
        Input:  list of MCTSNode objects needing evaluation
        Output: list of (value, policy_probs) tuples
        
        This is the key optimization — one GPU call instead of N.
        """
        if not nodes:
            return []

        self.model.eval()
        with torch.no_grad():
            # Build batch tensors
            batch_features = []
            batch_masks    = []

            for node in nodes:
                features = extract_features(node.state, node.valid_moves)
                mask     = node.mask
                batch_features.append(features)
                batch_masks.append(mask)

            # Stack into single tensors
            features_t = torch.tensor(
                np.array(batch_features), dtype=torch.float32
            ).to(self.device)
            masks_t = torch.tensor(
                np.array(batch_masks), dtype=torch.bool
            ).to(self.device)

            # ONE batched forward pass
            values, logits = self.model(features_t, masks_t)

            # Extract results per node
            results = []
            probs_all = F.softmax(logits, dim=-1).cpu().numpy()
            values_np = values.cpu().numpy()

            for i, node in enumerate(nodes):
                probs = probs_all[i] * node.mask
                total = probs.sum()
                if total > 0:
                    probs /= total
                else:
                    probs = node.mask.astype(np.float32)
                    probs /= probs.sum()

                results.append((float(values_np[i]), probs))

        return results

    def evaluate(self, state, valid_moves, mask):
        """Single node evaluation — delegates to evaluate_batch."""
        node = MCTSNode(state.clone())
        value, probs = self.evaluate_batch([node])[0]
        return value, probs

    def expand_no_eval(self, node, value, priors):
        """
        Expand node using pre-computed value and priors.
        Separates tree expansion from network evaluation.
        """
        for action_idx in node.mask.nonzero()[0]:
            move, decision = action_to_move_decision(int(action_idx))
            prior = float(priors[action_idx])

            next_state = node.state.clone()
            apply_move(next_state, move)

            if decision == "stop":
                stop_turn(next_state)
                if not next_state.game_over:
                    next_state.roll_dice()
            else:
                if not next_state.game_over:
                    next_state.roll_dice()
                    new_valid = get_valid_moves(next_state)
                    if not new_valid:
                        bust_turn(next_state)
                        if not next_state.game_over:
                            next_state.roll_dice()

            child = MCTSNode(
                state=next_state,
                parent=node,
                action=int(action_idx),
                prior=prior,
            )
            node.children[int(action_idx)] = child

        node.is_expanded = True

    def backpropagate(self, node, value):
        """Update visit counts and values up the tree."""
        while node is not None:
            node.N += 1
            node.W += value
            value   = 1.0 - value
            node    = node.parent

    def search(self, state, num_simulations=50,
           dirichlet_alpha=0.5, dirichlet_epsilon=0.25):
        """
        Batched MCTS search.
        
        Strategy:
        1. Run num_simulations tree traversals to find leaf nodes
        2. Batch evaluate ALL leaves in one GPU call
        3. Expand leaves and backpropagate
        
        This gives ~15x speedup over sequential evaluation.
        """
        if not state.dice:
            state.roll_dice()

        root = MCTSNode(state.clone())

        if not root.valid_moves:
            return np.zeros(ACTION_SPACE, dtype=np.float32), 0.0

        # ---- PHASE 1: Evaluate and expand root ----
        root_value, root_priors = self.evaluate(
            root.state, root.valid_moves, root.mask
        )
        self.expand_no_eval(root, root_value, root_priors)
        root.N = 1
        root.W = root_value

        # Add Dirichlet noise to root priors
        if root.children and dirichlet_epsilon > 0:
            noise = np.random.dirichlet(
                [dirichlet_alpha] * len(root.children)
            )
            for i, child in enumerate(root.children.values()):
                child.prior = (
                    (1 - dirichlet_epsilon) * child.prior +
                    dirichlet_epsilon * noise[i]
                )

        # ---- PHASE 2: Batched simulations ----
        # Run simulations in batches — collect leaves, evaluate together
        remaining = num_simulations - 1
        batch_size = min(remaining, 8)  # evaluate 8 leaves at once

        while remaining > 0:
            current_batch = min(batch_size, remaining)
            leaves_to_eval = []
            paths = []  # (node, value_if_terminal) for backprop

            # Collect a batch of leaf nodes via tree traversal
            for _ in range(current_batch):
                node = root
                traversal_path = []

                # Selection — traverse to leaf, applying virtual loss
                while not node.is_leaf() and not node.is_terminal:
                    if not node.children:
                        break
                    node.N += 1
                    node.W -= 1
                    node.virtual_loss += 1
                    traversal_path.append(node)
                    node = node.select_child()

                if node.is_terminal:
                    # Terminal — value is known
                    winner = node.state.winner
                    acting = node.state.active_player
                    val = 1.0 if winner == acting else 0.0
                    paths.append((node, val, None, traversal_path))
                elif not node.valid_moves:
                    paths.append((node, 0.0, None, traversal_path))
                else:
                    leaves_to_eval.append(node)
                    paths.append((node, None, len(leaves_to_eval) - 1, traversal_path))

            # Batch evaluate all non-terminal leaves
            if leaves_to_eval:
                batch_results = self.evaluate_batch(leaves_to_eval)

                # Expand and backpropagate
                for node, term_val, leaf_idx, traversal_path in paths:
                    # Remove virtual loss before real backprop
                    for vnode in traversal_path:
                        vnode.N -= 1
                        vnode.W += 1
                        vnode.virtual_loss -= 1
                    if leaf_idx is not None:
                        value, priors = batch_results[leaf_idx]
                        self.expand_no_eval(node, value, priors)
                        self.backpropagate(node, value)
                    else:
                        self.backpropagate(node, term_val)
            else:
                # All terminal
                for node, term_val, _, traversal_path in paths:
                    for vnode in traversal_path:
                        vnode.N -= 1
                        vnode.W += 1
                        vnode.virtual_loss -= 1
                    self.backpropagate(node, term_val)

            remaining -= current_batch

        # ---- Extract results ----
        visits = np.zeros(ACTION_SPACE, dtype=np.float32)
        for action_idx, child in root.children.items():
            visits[action_idx] = child.N

        total = visits.sum()
        policy = visits / total if total > 0 else \
                 root.mask.astype(np.float32) / root.mask.sum()

        return policy, root.Q

    def get_action(self, state, num_simulations=50, temperature=1.0):
        """Run MCTS and return chosen action with training targets."""
        policy, value = self.search(state, num_simulations)

        # Training target is always raw normalized visit counts
        train_policy = policy  # already normalized in search()

        if temperature <= 0.01:
            action_idx = policy.argmax()
        else:
            visits_temp = policy ** (1.0 / temperature)
            total = visits_temp.sum()
            if total > 0:
                visits_temp /= total
            action_idx = np.random.choice(ACTION_SPACE, p=visits_temp)

        move, decision = action_to_move_decision(int(action_idx))
        return int(action_idx), move, decision, train_policy, value

# ---- SELF-TEST ----

if __name__ == "__main__":
    import time
    from games.cantstop.model import CantStopNet
    from games.cantstop.evaluate import load_model

    print("Testing MCTS...\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = load_model('models/cantstop/best_model.pt', device)
    mcts = MCTS(model, device)

    # Test 1 — single position search
    state = GameState(2)
    state.roll_dice()
    valid = get_valid_moves(state)

    print(f"Dice: {state.dice}")
    print(f"Valid moves: {valid}\n")

    start = time.time()
    policy, value = mcts.search(state, num_simulations=50)
    elapsed = time.time() - start

    print(f"MCTS (50 sims): {elapsed:.2f}s")
    print(f"Position value: {value:.3f}")
    print(f"Policy (legal actions only):")

    legal_indices = policy.nonzero()[0]
    for idx in legal_indices:
        move, decision = action_to_move_decision(int(idx))
        print(f"  {move} {decision}: {policy[idx]:.3f}")

    # Test 2 — timing at different simulation counts
    print("\nTiming at different simulation counts:")
    for n_sims in [10, 20, 50, 100]:
        state2 = GameState(2)
        state2.roll_dice()
        start = time.time()
        for _ in range(5):
            mcts.search(state2.clone(), num_simulations=n_sims)
        elapsed = (time.time() - start) / 5
        print(f"  {n_sims:3d} sims: {elapsed:.3f}s per decision")

    # Test 3 — action selection
    print("\nAction selection test:")
    action_idx, move, decision, policy, value = mcts.get_action(
        state, num_simulations=50, temperature=1.0
    )
    print(f"  Chosen: {move} {decision} (action {action_idx})")
    print(f"  Value:  {value:.3f}")
    print(f"  Policy entropy: "
          f"{-(policy[policy>0] * np.log(policy[policy>0])).sum():.3f}")

    print("\nMCTS tests complete!")