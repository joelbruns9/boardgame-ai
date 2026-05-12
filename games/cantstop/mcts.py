# mcts.py
# Monte Carlo Tree Search for Can't Stop — lazy chance node version.
#
# Tree structure:
#   DecisionNode  — a player has dice and must choose a (pair, stop/continue)
#                   action. Neural network is evaluated here.
#   ChanceNode    — nature must roll dice. No player choice, no NN evaluation.
#                   Children are populated LAZILY by sampled dice outcome.
#
# Edges:
#   DecisionNode --(stop, non-winning)--> ChanceNode   (opponent pre-roll)
#   DecisionNode --(stop, winning)------> terminal DecisionNode
#   DecisionNode --(continue)-----------> ChanceNode   (same player pre-roll)
#   ChanceNode   --(normal outcome)-----> DecisionNode (same player, post-roll)
#   ChanceNode   --(BUST outcome)-------> ChanceNode   (opponent pre-roll)
#
# Perspective flip rule (in backprop):
#   Each child node stores `flip_from_parent` set at creation time by comparing
#   parent's active_player to its own active_player.
#       After STOP non-winning: player changes → flip
#       After CONTINUE:          player same   → no flip
#       Across ChanceNode normal outcome: player same → no flip
#       Across ChanceNode BUST outcome: player changes → flip (child is a
#                                                              nested ChanceNode)
#       Terminal STOP win: player same (winner == active_player) → no flip
#
# Note on BUST handling: a BUST outcome means the rolling player has no
# legal moves; the turn passes to the opponent. We model that as a
# nested ChanceNode (opponent's pre-roll), NOT as a DecisionNode — this
# way the opponent's first dice roll is also a fresh sample on each
# traversal, preserving correct expected-value semantics. All BUST
# outcomes from a chance node collapse to a single shared canonical
# key (empty tuple), since the resulting state depends only on the
# pre-roll state (no info from the busting dice carries forward).
#
# Virtual loss is applied only at DecisionNodes during selection.
# Dirichlet noise is applied only at the root DecisionNode.
# Neural network is evaluated only at DecisionNodes — batching preserved.

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


# ---- NODE TYPES ----

class DecisionNode:
    """
    A node where the active player must choose an action.

    Invariants when non-terminal:
        - state.dice is populated (dice already rolled)
        - valid_moves is non-empty
        - actions are integer indices into ACTION_SPACE encoding (pair, stop/continue)
        - children[action_idx] is either:
              - a terminal DecisionNode (only possible after STOP that wins game)
              - a ChanceNode (otherwise)
        - NN is evaluated here

    Backprop semantics:
        - N, W track per-DecisionNode statistics
        - W is accumulated in this node's active_player perspective
        - flip_from_parent indicates whether the value coming up from this node
          should be flipped before adding to the parent's W
    """

    __slots__ = [
        'state', 'parent', 'parent_action', 'prior',
        'children', 'N', 'W',
        'valid_moves', 'mask', 'is_terminal',
        'is_expanded', 'virtual_N', 'flip_from_parent',
    ]

    def __init__(self, state, parent=None, parent_action=None,
                 prior=0.0, flip_from_parent=False):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.prior = prior
        self.flip_from_parent = flip_from_parent

        self.children = {}          # action_idx -> ChanceNode or terminal DecisionNode
        # Real statistics — modified ONLY by backprop. Never touched by
        # virtual loss bookkeeping. This split (vs. mutating N/W and
        # reversing) means real stats can't be left corrupt by an
        # exception mid-simulation.
        self.N = 0
        self.W = 0.0
        # Virtual visit count — incremented on traversal-through,
        # decremented on backprop arrival or explicit cleanup. UCB sees
        # effective_N = N + virtual_N and effective_W = W - virtual_N,
        # which discourages parallel workers from converging on the
        # same path.
        self.virtual_N = 0
        self.is_terminal = state.game_over
        self.is_expanded = False

        if not self.is_terminal:
            self.valid_moves = get_valid_moves(state)
            self.mask = get_legal_action_mask(self.valid_moves)
        else:
            self.valid_moves = []
            self.mask = np.zeros(ACTION_SPACE, dtype=bool)

    @property
    def Q(self):
        """Real Q — used as the training value target. Excludes VL."""
        return self.W / self.N if self.N > 0 else 0.0

    @property
    def effective_Q(self):
        """Q including virtual loss — used by UCB during traversal."""
        eff_N = self.N + self.virtual_N
        if eff_N == 0:
            return 0.0
        return (self.W - self.virtual_N) / eff_N

    @property
    def effective_N(self):
        """Visit count including virtual visits — used by UCB."""
        return self.N + self.virtual_N

    def ucb_score(self, child):
        """
        UCB score for a child of this DecisionNode.

        The child is either a ChanceNode (whose Q is the visit-weighted EV
        over its sampled outcomes) or a terminal DecisionNode (Q is the
        terminal value). In both cases child.Q is in the *child's*
        active_player perspective, so we must flip it back to this node's
        perspective if flip_from_parent is True.

        UCB uses effective_N/effective_Q so that virtual loss applied
        elsewhere influences selection. Only DecisionNodes carry VL
        statistics; ChanceNodes are queried for their normal Q.
        """
        child_q = (
            child.effective_Q if isinstance(child, DecisionNode) else child.Q
        )
        if child.flip_from_parent:
            child_q = 1.0 - child_q

        child_N = (
            child.effective_N if isinstance(child, DecisionNode) else child.N
        )
        parent_N = self.effective_N

        exploration = (
            C_PUCT * child.prior * math.sqrt(max(parent_N, 1)) / (1 + child_N)
        )
        return child_q + exploration

    def select_child(self):
        """Select child with highest UCB."""
        return max(self.children.values(), key=lambda c: self.ucb_score(c))

    def is_leaf(self):
        return not self.is_expanded


class ChanceNode:
    """
    A node where nature rolls dice. The state stored here is PRE-ROLL:
    state.dice is empty. Each traversal rolls fresh dice into a clone and
    looks up or creates the child DecisionNode for the canonicalized outcome.

    Children are keyed by a canonical outcome = sorted tuple of valid moves
    given the rolled dice and current runner/claimed state. Two rolls that
    produce the same set of legal moves are equivalent for MCTS purposes.
    An empty tuple represents a BUST outcome.

    Backprop semantics:
        - N, W accumulate over all sampled outcomes
        - Q = W/N is exactly the visit-weighted Monte Carlo expected value
          of this chance node (in this node's active_player perspective)

    Prior is inherited from the action that led here (parent's policy prior
    on the (pair, stop/continue) action). This makes UCB at the parent
    DecisionNode work correctly without any special-casing for chance nodes.
    """

    __slots__ = [
        'state', 'parent', 'parent_action', 'prior',
        'children_by_outcome', 'N', 'W', 'flip_from_parent',
    ]

    def __init__(self, state, parent=None, parent_action=None,
                 prior=0.0, flip_from_parent=False):
        # state stored here must have dice cleared; we never roll into it
        # directly — we clone and roll fresh on each traversal.
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.prior = prior
        self.flip_from_parent = flip_from_parent

        self.children_by_outcome = {}   # canonical_outcome (tuple) -> DecisionNode
        self.N = 0
        self.W = 0.0

    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0


# ---- HELPERS ----

def _make_child_after_action(parent_decision_node, action_idx, prior):
    """
    Build the child node resulting from taking `action_idx` at
    parent_decision_node.

    Returns either:
        - a terminal DecisionNode (if STOP ended the game)
        - a ChanceNode (otherwise — dice need to be rolled before next decision)

    flip_from_parent is set based on whether the resulting active_player
    differs from the parent's active_player.
    """
    move, decision = action_to_move_decision(int(action_idx))

    next_state = parent_decision_node.state.clone()
    apply_move(next_state, move)

    if decision == "stop":
        stop_turn(next_state)
        # stop_turn either: (a) wins the game (winner == prior active player,
        # active_player unchanged), or (b) passes turn to opponent.
        if next_state.game_over:
            # Terminal — model as a terminal DecisionNode. No further chance.
            flip = (next_state.active_player !=
                    parent_decision_node.state.active_player)
            terminal = DecisionNode(
                state=next_state,
                parent=parent_decision_node,
                parent_action=int(action_idx),
                prior=prior,
                flip_from_parent=flip,
            )
            return terminal
        # Non-terminal stop: opponent's turn, needs to roll.
        next_state.dice = []  # ensure pre-roll
        flip = (next_state.active_player !=
                parent_decision_node.state.active_player)
        return ChanceNode(
            state=next_state,
            parent=parent_decision_node,
            parent_action=int(action_idx),
            prior=prior,
            flip_from_parent=flip,
        )

    # decision == "continue": same player keeps going, needs to roll again.
    next_state.dice = []  # pre-roll
    # active_player unchanged → no flip
    return ChanceNode(
        state=next_state,
        parent=parent_decision_node,
        parent_action=int(action_idx),
        prior=prior,
        flip_from_parent=False,
    )


def _sample_chance_outcome(chance_node):
    """
    Roll fresh dice on a clone of the chance node's state.

    Returns:
        (kind, child_state, canonical_outcome, flip)

    where:
        kind = 'decision' for normal outcomes — child_state has dice
               rolled and valid moves available; canonical is the sorted
               tuple of valid moves
        kind = 'bust'     for BUST outcomes — child_state is post-bust
               with dice empty (opponent needs to roll); canonical is ()
               (a single shared key); flip is whether active_player
               actually changed (almost always True; could be False if
               opponent also busts back, but that resolution happens via
               nested chance sampling, not here).

    This split means BUST outcomes lead to a ChanceNode (opponent's
    pre-roll), not a DecisionNode — preserving the same lazy-chance
    semantics for the opponent's first dice roll after the bust.
    """
    rolled_state = chance_node.state.clone()
    rolled_state.roll_dice()
    valid = get_valid_moves(rolled_state)

    if valid:
        canonical = tuple(sorted(valid))
        return 'decision', rolled_state, canonical, False

    # BUST: pass turn to opponent. Clear dice so the resulting ChanceNode
    # represents "opponent needs to roll."
    bust_turn(rolled_state)
    rolled_state.dice = []
    flip = (rolled_state.active_player != chance_node.state.active_player)
    # canonical = () is the single shared BUST key — all bust outcomes
    # collapse to the same nested ChanceNode for the opponent.
    return 'bust', rolled_state, (), flip


# ---- MCTS ----

class MCTS:
    """
    Monte Carlo Tree Search with batched network evaluation and
    lazy chance nodes for proper handling of dice variance.

    External interface (unchanged):
        MCTS(model, device)
        MCTS.search(state, num_simulations, dirichlet_alpha, dirichlet_epsilon)
        MCTS.get_action(state, num_simulations, temperature)
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    # ---- BATCHED NETWORK EVALUATION ----

    def evaluate_batch(self, nodes):
        """
        Evaluate multiple DecisionNodes in a single batched forward pass.
        Only DecisionNodes are evaluated by the network — never ChanceNodes.
        """
        if not nodes:
            return []

        self.model.eval()
        with torch.no_grad():
            batch_features = []
            batch_masks = []

            for node in nodes:
                features = extract_features(node.state, node.valid_moves)
                mask = node.mask
                batch_features.append(features)
                batch_masks.append(mask)

            features_t = torch.tensor(
                np.array(batch_features), dtype=torch.float32
            ).to(self.device)
            masks_t = torch.tensor(
                np.array(batch_masks), dtype=torch.bool
            ).to(self.device)

            values, logits = self.model(features_t, masks_t)

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
                    s = probs.sum()
                    if s > 0:
                        probs /= s
                results.append((float(values_np[i]), probs))

        return results

    def evaluate(self, state, valid_moves, mask):
        """Single-node evaluation — used for the root."""
        tmp = DecisionNode(state.clone())
        value, probs = self.evaluate_batch([tmp])[0]
        return value, probs

    # ---- EXPANSION ----

    def expand_decision_node(self, node, priors):
        """
        Expand a DecisionNode: create child node per legal action.
        Each child is either a ChanceNode (non-terminal) or terminal
        DecisionNode (only when STOP wins the game).

        priors: full ACTION_SPACE-length array of network priors.
        """
        for action_idx in node.mask.nonzero()[0]:
            prior = float(priors[action_idx])
            child = _make_child_after_action(node, int(action_idx), prior)
            node.children[int(action_idx)] = child
        node.is_expanded = True

    # ---- BACKPROPAGATION ----

    def backpropagate(self, leaf_node, leaf_value, decision_path):
        """
        Propagate value up the tree, applying per-edge perspective flips.

        leaf_value is in leaf_node.state.active_player perspective.

        decision_path: list of DecisionNodes traversed during selection
                       (NOT including the leaf). Each had virtual_N
                       incremented during traversal; we decrement it here.

        Real N/W statistics are updated independently of virtual_N,
        so an exception between traversal and backprop leaves at worst
        a phantom virtual_N count, never corrupted real statistics.

        Strategy: walk from leaf upward via parent pointers, tracking
        the running value in the current node's perspective. At each
        step, flip the value if the child reported flip_from_parent.
        ChanceNodes on the path also get N/W updated automatically by
        the walk-up — no separate path list needed for them.
        """
        # Decrement virtual_N on every DecisionNode that had it applied.
        # This does NOT touch real N/W — those are updated by the walk-up
        # below, independently.
        for dn in decision_path:
            dn.virtual_N -= 1

        # Walk up: update each ancestor's real N and W with the value
        # in *its* perspective.
        node = leaf_node
        value = leaf_value
        while node is not None:
            node.N += 1
            node.W += value
            parent = node.parent
            if parent is None:
                break
            # Flip value when moving from node into parent if the edge
            # crossed a player change.
            if node.flip_from_parent:
                value = 1.0 - value
            node = parent

    # ---- TRAVERSAL ----

    def _traverse(self, root):
        """
        Single-simulation tree traversal from root to a leaf.

        Returns one of:
            ('terminal', leaf_decision_node, value, decision_path)
            ('eval',     leaf_decision_node, None,  decision_path)

        Virtual loss is applied to every interior DecisionNode visited
        (NOT to chance nodes). The caller must undo virtual loss during
        backprop.

        leaf_decision_node is always a DecisionNode (terminal or to-be-
        evaluated). ChanceNodes are never returned as leaves — instead,
        when we reach an unseen outcome at a ChanceNode, we create the
        corresponding new DecisionNode and return that as the leaf.
        """
        decision_path = []
        node = root
        # Safety cap on consecutive ChanceNode traversals within a single
        # simulation. A bust chain (P0 busts → P1 busts → P0 busts → ...)
        # is theoretically unbounded in pathological end-game states.
        # In practice this almost never triggers, but the cap prevents
        # any single simulation from hanging during overnight runs.
        chance_depth = 0
        CHANCE_DEPTH_CAP = 20

        while True:
            if isinstance(node, DecisionNode):
                # Reset chance-depth counter — we're back at a decision.
                chance_depth = 0
                # Terminal? — stop and return.
                if node.is_terminal:
                    winner = node.state.winner
                    acting = node.state.active_player
                    val = 1.0 if winner == acting else 0.0
                    return ('terminal', node, val, decision_path)

                # Unexpanded DecisionNode → this is our leaf for NN eval.
                if not node.is_expanded:
                    return ('eval', node, None, decision_path)

                # Defensive: if expanded but no children, treat as leaf.
                if not node.children:
                    return ('eval', node, None, decision_path)

                # Apply virtual loss and descend.
                # Real N/W are NEVER touched here — only virtual_N is
                # incremented. UCB sees the effect via effective_N /
                # effective_Q. This separation means an exception
                # mid-traversal can corrupt at most virtual_N, never
                # the real visit/value statistics.
                node.virtual_N += 1
                decision_path.append(node)
                node = node.select_child()

            else:
                # ChanceNode: sample a fresh outcome.
                # NO virtual loss applied here (would corrupt EV estimate).
                chance_depth += 1
                if chance_depth > CHANCE_DEPTH_CAP:
                    # Safety valve: extremely rare bust-chain. Return a
                    # neutral value (0.5) in this node's perspective. The
                    # outer code expects a DecisionNode leaf for 'terminal'
                    # / 'eval' returns, so we synthesize one by treating
                    # this chance node itself as the leaf. We use the
                    # 'terminal'-style return with value 0.5 so backprop
                    # walks up from `node` (a ChanceNode) — backpropagate
                    # handles both node types uniformly via parent
                    # pointers, so this works.
                    return ('terminal', node, 0.5, decision_path)

                kind, child_state, canonical, flip = (
                    _sample_chance_outcome(node)
                )

                existing = node.children_by_outcome.get(canonical)
                if existing is None:
                    # Lazy creation: child type depends on the outcome.
                    # NOTE on prior=0.0: the children of a ChanceNode are
                    # never selected via UCB (chance nodes use sampling,
                    # not policy-guided selection), so the prior field
                    # on these children is dead and unused. We set 0.0
                    # for clarity. UCB exploration into a chance node is
                    # driven by the *chance node's own prior*, which was
                    # inherited from the policy head at the parent
                    # DecisionNode when the chance node was created.
                    if kind == 'bust':
                        # BUST → opponent's pre-roll ChanceNode.
                        # Continue traversal into it on the next iteration
                        # of this same simulation (so we don't waste a sim
                        # on an intermediate node with no value).
                        new_chance = ChanceNode(
                            state=child_state,
                            parent=node,
                            parent_action=None,
                            prior=0.0,   # dead — never read by UCB
                            flip_from_parent=flip,
                        )
                        node.children_by_outcome[canonical] = new_chance
                        node = new_chance
                        continue
                    else:
                        # Normal outcome → new DecisionNode (leaf).
                        new_decision = DecisionNode(
                            state=child_state,
                            parent=node,
                            parent_action=None,
                            prior=0.0,   # dead — never read by UCB
                            flip_from_parent=flip,
                        )
                        node.children_by_outcome[canonical] = new_decision

                        if new_decision.is_terminal:
                            winner = new_decision.state.winner
                            acting = new_decision.state.active_player
                            val = 1.0 if winner == acting else 0.0
                            return ('terminal', new_decision, val,
                                    decision_path)
                        return ('eval', new_decision, None, decision_path)

                # Existing child for this outcome — continue selection.
                node = existing

    # ---- SEARCH ----

    def search(self, state, num_simulations=50,
               dirichlet_alpha=0.5, dirichlet_epsilon=0.25):
        """
        Run batched MCTS from `state` and return:
            (policy, value)
        where policy is the normalized visit-count distribution over
        ACTION_SPACE at the root DecisionNode, and value is the root's
        Q (estimated win probability for the root's active player).
        """
        # Root must be a DecisionNode with dice present.
        if not state.dice:
            state.roll_dice()

        root_state = state.clone()
        root = DecisionNode(state=root_state, parent=None,
                            parent_action=None, prior=0.0,
                            flip_from_parent=False)

        if not root.valid_moves:
            # No legal moves at root — busted root. Caller is expected to
            # have rolled into a position with valid moves, but handle
            # defensively.
            return np.zeros(ACTION_SPACE, dtype=np.float32), 0.0

        # ---- Evaluate and expand root ----
        root_value, root_priors = self.evaluate(
            root.state, root.valid_moves, root.mask
        )
        self.expand_decision_node(root, root_priors)
        root.N = 1
        root.W = root_value

        # Dirichlet noise on root priors (only over legal actions).
        if root.children and dirichlet_epsilon > 0:
            noise = np.random.dirichlet(
                [dirichlet_alpha] * len(root.children)
            )
            for i, child in enumerate(root.children.values()):
                child.prior = (
                    (1 - dirichlet_epsilon) * child.prior +
                    dirichlet_epsilon * noise[i]
                )

        # ---- Batched simulation loop ----
        remaining = num_simulations - 1
        batch_size_cap = 8

        while remaining > 0:
            current_batch = min(batch_size_cap, remaining)

            traversals = []
            leaves_for_eval = []   # parallel list, only non-terminal entries

            for _ in range(current_batch):
                kind, leaf, val, dpath = self._traverse(root)
                if kind == 'terminal':
                    traversals.append(('terminal', leaf, val, dpath))
                else:
                    leaves_for_eval.append(leaf)
                    traversals.append(('eval', leaf, None, dpath))

            # Batched NN evaluation for all non-terminal leaves.
            eval_results = (
                self.evaluate_batch(leaves_for_eval)
                if leaves_for_eval else []
            )
            eval_iter = iter(eval_results)

            for kind, leaf, val, dpath in traversals:
                if kind == 'terminal':
                    self.backpropagate(leaf, val, dpath)
                else:
                    value, priors = next(eval_iter)
                    # Expand the leaf with its priors.
                    self.expand_decision_node(leaf, priors)
                    self.backpropagate(leaf, value, dpath)

            remaining -= current_batch

        # ---- Extract policy from root visit counts ----
        visits = np.zeros(ACTION_SPACE, dtype=np.float32)
        for action_idx, child in root.children.items():
            visits[action_idx] = child.N

        total = visits.sum()
        if total > 0:
            policy = visits / total
        else:
            # Fallback — shouldn't happen with num_simulations > 0.
            mask_f = root.mask.astype(np.float32)
            policy = mask_f / mask_f.sum() if mask_f.sum() > 0 else mask_f

        return policy, root.Q

    # ---- ACTION SELECTION ----

    def get_action(self, state, num_simulations=50, temperature=1.0):
        """
        Run MCTS and return:
            (action_idx, move, decision, train_policy, value)

        train_policy is the raw normalized visit-count distribution from
        the root DecisionNode — this is the policy training target.
        Temperature is applied only to action sampling, never to the
        training target.
        """
        policy, value = self.search(state, num_simulations)
        train_policy = policy  # raw normalized visit counts

        if temperature <= 0.01:
            action_idx = int(policy.argmax())
        else:
            visits_temp = policy ** (1.0 / temperature)
            total = visits_temp.sum()
            if total > 0:
                visits_temp = visits_temp / total
                action_idx = int(np.random.choice(ACTION_SPACE, p=visits_temp))
            else:
                action_idx = int(policy.argmax())

        move, decision = action_to_move_decision(int(action_idx))
        return int(action_idx), move, decision, train_policy, value


# ---- SELF-TEST ----

if __name__ == "__main__":
    import time
    from games.cantstop.model import CantStopNet
    from games.cantstop.evaluate import load_model

    print("Testing MCTS (lazy chance nodes)...\n")

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