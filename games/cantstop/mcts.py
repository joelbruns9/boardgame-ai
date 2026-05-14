# mcts.py
# Monte Carlo Tree Search for Can't Stop — lazy chance node + async version.
#
# Tree structure:
#   DecisionNode  — a player has dice and must choose a (pair, stop/continue)
#                   action. Neural network is evaluated here.
#   ChanceNode    — nature must roll dice. No player choice, no NN evaluation.
#                   Children are populated LAZILY by sampled dice outcome.
#
# Execution model:
#   ASYNC (default): execution has two phases.
#     Phase A — Sequential warmup. The first `warmup_sims` simulations
#     run one at a time. Each sees the previous sim's backprop fully
#     reflected in tree statistics before starting. This is necessary
#     because async sims that start before tree statistics exist will
#     all descend the same path (UCB sees identical visit counts and
#     priors), wasting compute and biasing visit-count training targets.
#
#     Phase B — Async batched. Remaining sims run with `target_inflight`
#     concurrency. Each sim advances down the tree until it hits an
#     unexpanded leaf, then parks (virtual loss already applied along
#     its path). When all in-flight sims are parked (or no sim can
#     advance), parked leaves are sent to the network in ONE batch.
#     Each sim resumes: leaf is expanded with returned priors, backprop
#     happens with returned value.
#
#   SYNC (legacy, kept for testing): one sim at a time, batch_size_cap=8.
#     Use target_inflight=1 to force this path.
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


# ---- ASYNC SIMULATION STATE MACHINE ----
#
# Each in-flight simulation is a PendingSim. The scheduler advances
# them in three phases per cycle:
#
#   Phase 1 (descend): for each RUNNING sim, call _step_sim() in a loop
#     until the sim transitions out of RUNNING. Sims transition to:
#       AWAITING_EVAL — hit an unexpanded leaf; needs NN evaluation.
#                       Virtual loss has already been applied along its path.
#       READY_BACKPROP — hit a terminal node or the chance-depth cap.
#                       Value is already known; just walk it up the tree.
#
#   Phase 2 (batch eval): collect all AWAITING_EVAL sims, call
#     evaluate_batch() ONCE with all their leaves. Iterate results,
#     expand each leaf (if not already expanded — see "duplicate leaf"
#     below), store the returned value on each sim, transition to
#     READY_BACKPROP.
#
#   Phase 3 (backprop): for each READY_BACKPROP sim, call backpropagate()
#     with its stored leaf and value. Transition to DONE.
#
# Duplicate leaves: if sims A and B both park at the same DecisionNode X
# (rare — virtual loss usually steers B elsewhere — but possible at
# shallow depths), evaluate_batch() will evaluate X twice. Both calls
# yield the same priors/value (model is deterministic at eval()). When
# resuming, A expands X; B sees X already expanded and skips re-expansion.
# Both sims backprop with the same value — which double-counts that
# leaf's value in the tree statistics, but does so symmetrically and is
# corrected by subsequent visits. The alternative (dedup leaves before
# evaluation) requires identity-tracking across sims and complicates the
# scheduler with negligible benefit.

# Sim states — small integers for cheap comparison.
_SIM_RUNNING        = 0
_SIM_AWAITING_EVAL  = 1
_SIM_READY_BACKPROP = 2
_SIM_DONE           = 3


class PendingSim:
    """
    A single in-flight MCTS simulation. Replaces the implicit state
    that _traverse() used to keep in its local variables.

    Lifecycle:
        Created in RUNNING state pointing at the root.
        _step_sim() advances `node` one selection step. May descend
            through multiple ChanceNode samples in a single call (chance
            sampling is cheap CPU work and doesn't need batching).
        On hitting an unexpanded DecisionNode → AWAITING_EVAL, parks.
        On hitting a terminal → READY_BACKPROP with value set.
        On chance-depth-cap exceeded → READY_BACKPROP with value 0.5.
        After backprop → DONE.
    """

    __slots__ = [
        'state',           # one of _SIM_* constants
        'node',            # current node during descent
        'decision_path',   # list of DecisionNodes traversed (VL applied)
        'chance_depth',    # consecutive ChanceNode samples in this sim
        'leaf',            # the DecisionNode that needs eval or is terminal
        'leaf_value',      # known value (terminal) or NN-returned value
    ]

    def __init__(self, root):
        self.state = _SIM_RUNNING
        self.node = root
        self.decision_path = []
        self.chance_depth = 0
        self.leaf = None
        self.leaf_value = None


CHANCE_DEPTH_CAP = 20  # also referenced by _traverse (kept consistent)


# ---- MCTS ----

class MCTS:
    """
    Monte Carlo Tree Search with batched network evaluation and
    lazy chance nodes for proper handling of dice variance.

    External interface (unchanged):
        MCTS(model_or_client, device)
        MCTS.search(state, num_simulations, dirichlet_alpha, dirichlet_epsilon)
        MCTS.get_action(state, num_simulations, temperature)

    The first constructor argument may be either:
      - a torch.nn.Module (local model on `device`) — original behavior
      - an inference_server.InferenceClient — sends inference requests
        to a separate GPU server process. Workers use this path so they
        never touch CUDA.

    The branch is selected by duck-typing on .infer(): if the object
    has an .infer(features_np, masks_np) method, it's treated as a
    client; otherwise as a model.
    """

    def __init__(self, model, device, target_inflight=16, warmup_sims=16):
        """
        Args:
            model: either a torch.nn.Module or an InferenceClient.
            device: target device for local inference. Ignored when model
                    is an InferenceClient (the server owns the device).
            target_inflight: how many simulations to run concurrently
                in the async scheduler. Larger values give bigger GPU
                batches (good for throughput) but increase the chance
                that many in-flight sims pile up on the same path before
                tree statistics differentiate them (bad for training
                signal). 16 is a balanced default — see the sweep test
                for tuning. target_inflight=1 forces the legacy sync
                path entirely.
            warmup_sims: number of simulations to run SEQUENTIALLY
                (inflight=1) at the start of each search before
                switching to the async batched scheduler. Sequential
                warmup guarantees that the first N sims see real
                tree statistics for the *previous* sim's backprop, so
                MCTS visit counts reflect search quality rather than
                arrival order. Set to 0 to disable warmup (matches the
                buggy original async behavior — kept available for A/B
                testing). For typical Can't Stop positions, 16 covers
                the worst-case branching factor.
        """
        self.device = device
        self.target_inflight = max(1, int(target_inflight))
        self.warmup_sims = max(0, int(warmup_sims))
        # Duck-type detection: InferenceClient has .infer(); a torch
        # model does not. (Importing InferenceClient explicitly would
        # create a cycle, since inference_server may want to import
        # MCTS internals in the future.)
        if hasattr(model, 'infer') and callable(model.infer):
            self.client = model
            self.model = None
        else:
            self.client = None
            self.model = model

    # ---- BATCHED NETWORK EVALUATION ----

    def evaluate_batch(self, nodes):
        """
        Evaluate multiple DecisionNodes in a single batched forward pass.
        Only DecisionNodes are evaluated by the network — never ChanceNodes.

        If a remote inference client was provided at construction time,
        features/masks are sent over IPC and the GPU server returns
        results. Otherwise the local model is used directly.
        """
        if not nodes:
            return []

        # Build batch on the worker (CPU) side. Same code path for both
        # local and remote inference — only the forward pass differs.
        batch_features = []
        batch_masks = []
        for node in nodes:
            features = extract_features(node.state, node.valid_moves)
            mask = node.mask
            batch_features.append(features)
            batch_masks.append(mask)

        features_np = np.array(batch_features, dtype=np.float32)
        masks_np    = np.array(batch_masks,    dtype=np.bool_)

        if self.client is not None:
            # Remote path: server returns already-softmaxed probs.
            values_np, probs_all = self.client.infer(features_np, masks_np)
        else:
            # Local path (original behavior, preserved exactly).
            self.model.eval()
            with torch.no_grad():
                features_t = torch.from_numpy(features_np).to(self.device)
                masks_t    = torch.from_numpy(masks_np).to(self.device)
                values, logits = self.model(features_t, masks_t)
                probs_t = F.softmax(logits, dim=-1)
                values_np = values.detach().cpu().numpy().astype(np.float32)
                probs_all = probs_t.detach().cpu().numpy().astype(np.float32)

        # ---- Mask + renormalize per-node ----
        # Whether softmax happened on GPU server or local model, we still
        # explicitly mask and renormalize to guard against numerical
        # leakage onto illegal moves.
        results = []
        for i, node in enumerate(nodes):
            probs = probs_all[i] * node.mask
            total = probs.sum()
            if total > 0:
                probs = probs / total
            else:
                probs = node.mask.astype(np.float32)
                s = probs.sum()
                if s > 0:
                    probs = probs / s
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

    # ---- ASYNC STEP FUNCTION ----

    def _step_sim(self, sim):
        """
        Advance a single PendingSim by one selection step.

        Returns nothing; mutates `sim` in place. After this call, `sim`
        is in one of:
            _SIM_RUNNING       — needs more steps (will be called again).
            _SIM_AWAITING_EVAL — parked at a leaf; needs NN evaluation.
                                 sim.leaf points at the unexpanded
                                 DecisionNode; sim.decision_path holds
                                 the VL-applied ancestors.
            _SIM_READY_BACKPROP — hit terminal or chance-depth-cap.
                                  sim.leaf and sim.leaf_value are set.

        This mirrors one iteration of the inner `while True` in
        _traverse(), with the explicit state machine making "park" a
        first-class outcome rather than a return value.

        IMPORTANT: when at a DecisionNode, this method may need to
        chase through several ChanceNode samples before reaching the
        next leaf or DecisionNode. Chance sampling is pure CPU work
        (no batching benefit) so we keep going within one call until
        we either (a) descend through a DecisionNode again, (b) park
        at a leaf, or (c) hit terminal/cap. This keeps the scheduler
        cycle count low while preserving batching at the leaves.
        """
        node = sim.node

        while True:
            if isinstance(node, DecisionNode):
                # Reset chance-depth counter — we're at a decision.
                sim.chance_depth = 0

                # Terminal?
                if node.is_terminal:
                    winner = node.state.winner
                    acting = node.state.active_player
                    sim.leaf = node
                    sim.leaf_value = 1.0 if winner == acting else 0.0
                    sim.node = node
                    sim.state = _SIM_READY_BACKPROP
                    return

                # Unexpanded DecisionNode → park here for NN eval.
                # Defensive: if expanded but no children somehow, also park.
                if not node.is_expanded or not node.children:
                    sim.leaf = node
                    sim.node = node
                    sim.state = _SIM_AWAITING_EVAL
                    return

                # Apply virtual loss and descend.
                # Real N/W are NEVER touched here — only virtual_N is
                # incremented. UCB sees the effect via effective_N /
                # effective_Q. This is what spreads parallel sims onto
                # different paths.
                node.virtual_N += 1
                sim.decision_path.append(node)
                node = node.select_child()
                # Loop and process the new node (might be ChanceNode).
                continue

            # ChanceNode: sample a fresh outcome. NO virtual loss applied
            # here (would corrupt EV estimate over sampled outcomes).
            sim.chance_depth += 1
            if sim.chance_depth > CHANCE_DEPTH_CAP:
                # Safety valve: extremely rare bust-chain. Use neutral
                # value 0.5 in this node's perspective. The backprop
                # walk handles ChanceNode parents uniformly, so treating
                # the chance node itself as the leaf is fine here.
                sim.leaf = node
                sim.leaf_value = 0.5
                sim.node = node
                sim.state = _SIM_READY_BACKPROP
                return

            kind, child_state, canonical, flip = (
                _sample_chance_outcome(node)
            )

            existing = node.children_by_outcome.get(canonical)
            if existing is None:
                # Lazy creation. See _traverse for the prior=0.0
                # rationale (ChanceNode children are sampled, not
                # UCB-selected, so their prior field is unused).
                if kind == 'bust':
                    new_chance = ChanceNode(
                        state=child_state,
                        parent=node,
                        parent_action=None,
                        prior=0.0,
                        flip_from_parent=flip,
                    )
                    node.children_by_outcome[canonical] = new_chance
                    node = new_chance
                    continue
                else:
                    new_decision = DecisionNode(
                        state=child_state,
                        parent=node,
                        parent_action=None,
                        prior=0.0,
                        flip_from_parent=flip,
                    )
                    node.children_by_outcome[canonical] = new_decision

                    if new_decision.is_terminal:
                        winner = new_decision.state.winner
                        acting = new_decision.state.active_player
                        sim.leaf = new_decision
                        sim.leaf_value = (
                            1.0 if winner == acting else 0.0
                        )
                        sim.node = new_decision
                        sim.state = _SIM_READY_BACKPROP
                        return

                    # Fresh DecisionNode that needs NN eval — park.
                    sim.leaf = new_decision
                    sim.node = new_decision
                    sim.state = _SIM_AWAITING_EVAL
                    return

            # Existing child for this outcome — keep descending.
            node = existing

    # ---- ASYNC SCHEDULER ----

    def _run_one_sim(self, root):
        """
        Run exactly ONE simulation end-to-end (descend → evaluate →
        expand → backprop). Used by the warmup phase before the async
        loop kicks in.

        Returns nothing — root's statistics are mutated in place.

        This is the "atomic unit of MCTS" that the rest of the
        scheduler is built around. By running these sequentially at
        the start, we guarantee that each subsequent simulation sees
        all previous simulations' backpropagated statistics — exactly
        like classical single-threaded MCTS. Once a few of these have
        landed, the tree has real statistics that UCB can use to
        diversify parallel sims correctly.
        """
        sim = PendingSim(root)
        while sim.state == _SIM_RUNNING:
            self._step_sim(sim)

        if sim.state == _SIM_AWAITING_EVAL:
            # Single-item batch — evaluate just this one leaf.
            results = self.evaluate_batch([sim.leaf])
            value, priors = results[0]
            if not sim.leaf.is_expanded:
                self.expand_decision_node(sim.leaf, priors)
            sim.leaf_value = value
            sim.state = _SIM_READY_BACKPROP

        # sim is now READY_BACKPROP (either path)
        self.backpropagate(sim.leaf, sim.leaf_value, sim.decision_path)
        sim.state = _SIM_DONE

    def _run_async_simulations(self, root, num_simulations):
        """
        Run `num_simulations` MCTS sims through `root`.

        Phase A (warmup): run `min(warmup_sims, num_simulations)` sims
        sequentially. Each one sees the prior sim's backprop fully
        reflected in tree statistics before starting. This solves the
        "ramp-up" problem where N parallel sims started at the root
        with identical statistics would all descend down the same path
        (a real issue, observed: 38-point win rate gap vs sync mode
        without warmup).

        Phase B (async): remaining sims run with `target_inflight`
        concurrency, using the batched scheduler. Each cycle:
          1. Top up in-flight pool to target_inflight.
          2. Advance every RUNNING sim until it parks or finishes.
          3. If any sims are AWAITING_EVAL, send them all to the NN
             in ONE batch. Expand each leaf, set leaf_value, transition
             to READY_BACKPROP.
          4. Backprop every READY_BACKPROP sim. Mark DONE.
        """
        # ---- Phase A: sequential warmup ----
        warmup = min(self.warmup_sims, num_simulations)
        for _ in range(warmup):
            self._run_one_sim(root)
        async_remaining = num_simulations - warmup

        if async_remaining <= 0:
            return

        # ---- Phase B: async batched simulation ----
        target = self.target_inflight
        sims_launched = 0
        sims_completed = 0

        # In-flight pool — sims that are not yet DONE.
        inflight = []

        while sims_completed < async_remaining:
            # ---- Phase 0: top up ----
            # Refill the in-flight pool from sims_launched up to
            # whatever's left. Each new sim starts at the root in
            # RUNNING state.
            while (len(inflight) < target and
                   sims_launched < async_remaining):
                inflight.append(PendingSim(root))
                sims_launched += 1

            if not inflight:
                break  # nothing left to do

            # ---- Phase 1: advance every RUNNING sim ----
            # _step_sim may transition a sim to AWAITING_EVAL or
            # READY_BACKPROP; the inner while-loop keeps calling it
            # until it parks or completes.
            for sim in inflight:
                while sim.state == _SIM_RUNNING:
                    self._step_sim(sim)

            # ---- Phase 2: batched NN evaluation ----
            awaiting = [s for s in inflight if s.state == _SIM_AWAITING_EVAL]
            if awaiting:
                leaves = [s.leaf for s in awaiting]
                results = self.evaluate_batch(leaves)
                # results is a list of (value, probs) in the same order
                # as `leaves`. Some sims may share a leaf (duplicate
                # parking); evaluate_batch returns one (value, probs)
                # per request, so duplicate leaves get duplicate
                # results. That's fine — the returned values are
                # identical (model is deterministic in eval mode), and
                # we only call expand_decision_node on the first
                # occurrence per leaf.
                for sim, (value, priors) in zip(awaiting, results):
                    if not sim.leaf.is_expanded:
                        self.expand_decision_node(sim.leaf, priors)
                    sim.leaf_value = value
                    sim.state = _SIM_READY_BACKPROP

            # ---- Phase 3: backprop ----
            new_inflight = []
            for sim in inflight:
                if sim.state == _SIM_READY_BACKPROP:
                    self.backpropagate(
                        sim.leaf, sim.leaf_value, sim.decision_path
                    )
                    sim.state = _SIM_DONE
                    sims_completed += 1
                else:
                    new_inflight.append(sim)
            inflight = new_inflight

            # Safety: if no sim made forward progress this cycle, we'd
            # loop forever. This shouldn't happen — every sim either
            # parks (→ batch eval transitions it) or finishes (→
            # backprop transitions it) — but the assertion is cheap.
            if (len(inflight) == target and not awaiting and
                    sims_launched >= async_remaining):
                # All in-flight sims are RUNNING (no awaiting, no
                # ready), and we can't launch more. _step_sim must have
                # left them in RUNNING with no transition possible —
                # this is a bug. Bail to avoid hanging.
                break

    def _run_sync_simulations(self, root, num_simulations):
        """
        Legacy synchronous simulation loop. Kept for the
        target_inflight=1 path. Functionally identical to the
        pre-async behavior: small batches of up to 8 traversals,
        evaluated together, then backpropagated.
        """
        remaining = num_simulations
        batch_size_cap = 8

        while remaining > 0:
            current_batch = min(batch_size_cap, remaining)

            traversals = []
            leaves_for_eval = []

            for _ in range(current_batch):
                kind, leaf, val, dpath = self._traverse(root)
                if kind == 'terminal':
                    traversals.append(('terminal', leaf, val, dpath))
                else:
                    leaves_for_eval.append(leaf)
                    traversals.append(('eval', leaf, None, dpath))

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
                    self.expand_decision_node(leaf, priors)
                    self.backpropagate(leaf, value, dpath)

            remaining -= current_batch

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

        # ---- Simulation loop ----
        # Root counts as 1 sim already (evaluated and expanded above).
        remaining = num_simulations - 1

        if remaining > 0:
            if self.target_inflight <= 1:
                # Legacy sync path. Kept for testing / debugging /
                # very-small-num_simulations cases where the async
                # scheduler's bookkeeping overhead would dominate.
                self._run_sync_simulations(root, remaining)
            else:
                self._run_async_simulations(root, remaining)

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

    print("Testing MCTS (lazy chance nodes + async scheduler)...\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = load_model('models/cantstop/best_model.pt', device)

    # Async MCTS with warmup — the new defaults are
    # target_inflight=16, warmup_sims=16.
    mcts = MCTS(model, device, target_inflight=16, warmup_sims=16)

    # ---- Test 1: single position search ----
    state = GameState(2)
    state.roll_dice()
    valid = get_valid_moves(state)

    print(f"\nDice: {state.dice}")
    print(f"Valid moves: {valid}\n")

    start = time.time()
    policy, value = mcts.search(state, num_simulations=50)
    elapsed = time.time() - start

    print(f"MCTS (50 sims, async warmup=16 inflight=16): {elapsed:.2f}s")
    print(f"Position value: {value:.3f}")
    print(f"Policy (legal actions only):")

    legal_indices = policy.nonzero()[0]
    for idx in legal_indices:
        move, decision = action_to_move_decision(int(idx))
        print(f"  {move} {decision}: {policy[idx]:.3f}")

    # Sanity check: policy should NOT collapse to 100% on one move
    # when multiple moves are legal. (The pre-warmup async version
    # did this; warmup fixes it.)
    max_p = policy[legal_indices].max()
    if len(legal_indices) > 1 and max_p > 0.95:
        print(f"  ⚠ Policy concentrated {max_p:.1%} on one move — "
              f"warmup may not be helping enough")
    else:
        print(f"  ✓ Policy distributes across moves (max = {max_p:.1%})")

    # ---- Test 2: async vs sync correctness ----
    # Both schedulers should produce comparable policies on the same
    # state. We can't assert equality because Dirichlet noise + chance
    # node sampling make them stochastic — but they should agree on
    # the rough ranking of legal actions over enough sims. We disable
    # Dirichlet noise + fix the global RNG to make this comparison
    # more deterministic.
    print("\nAsync vs sync consistency check:")
    mcts_sync = MCTS(model, device, target_inflight=1)

    # Disable Dirichlet by passing epsilon=0
    np.random.seed(123)
    state_t = GameState(2)
    state_t.dice = [3, 4, 3, 4]  # fixed dice for repeatability
    pol_async, val_async = mcts.search(
        state_t.clone(), num_simulations=200, dirichlet_epsilon=0.0
    )
    np.random.seed(123)
    pol_sync, val_sync = mcts_sync.search(
        state_t.clone(), num_simulations=200, dirichlet_epsilon=0.0
    )

    # Pick the top action under each
    top_async = int(pol_async.argmax())
    top_sync  = int(pol_sync.argmax())
    move_async, dec_async = action_to_move_decision(top_async)
    move_sync, dec_sync = action_to_move_decision(top_sync)
    print(f"  Sync  top action: {move_sync} {dec_sync} "
          f"({pol_sync[top_sync]:.3f}) value={val_sync:.3f}")
    print(f"  Async top action: {move_async} {dec_async} "
          f"({pol_async[top_async]:.3f}) value={val_async:.3f}")
    # Most-visited action should usually match. Don't hard-assert
    # equality because of chance sampling, but flag if Q values
    # disagree wildly (a sign of broken backprop or VL).
    q_delta = abs(val_async - val_sync)
    if q_delta > 0.15:
        print(f"  ⚠ Q-value disagreement = {q_delta:.3f} — investigate")
    else:
        print(f"  ✓ Q values agree within {q_delta:.3f}")

    # ---- Test 3: timing comparison ----
    print("\nTiming sync vs async (per-search wallclock):")
    for n_sims in [50, 100, 200, 400]:
        # Sync
        state_t = GameState(2)
        state_t.roll_dice()
        start = time.time()
        for _ in range(3):
            mcts_sync.search(state_t.clone(), num_simulations=n_sims)
        t_sync = (time.time() - start) / 3

        # Async
        start = time.time()
        for _ in range(3):
            mcts.search(state_t.clone(), num_simulations=n_sims)
        t_async = (time.time() - start) / 3

        speedup = t_sync / max(t_async, 1e-6)
        print(f"  {n_sims:4d} sims:  sync={t_sync*1000:7.1f} ms  "
              f"async={t_async*1000:7.1f} ms  "
              f"speedup={speedup:.2f}x")

    # ---- Test 4: action selection ----
    print("\nAction selection (async):")
    action_idx, move, decision, policy, value = mcts.get_action(
        state, num_simulations=100, temperature=1.0
    )
    print(f"  Chosen: {move} {decision} (action {action_idx})")
    print(f"  Value:  {value:.3f}")
    print(f"  Policy entropy: "
          f"{-(policy[policy>0] * np.log(policy[policy>0])).sum():.3f}")

    # ---- Test 5: virtual loss hygiene ----
    # After search, every node's virtual_N must be zero. If async
    # scheduling missed a decrement (bug), virtual_N would persist.
    print("\nVirtual loss hygiene (every node's virtual_N == 0):")
    state_t = GameState(2)
    state_t.roll_dice()
    # Build a root we can inspect by patching search to retain it.
    # Easiest path: re-run search and walk the root via a hook.
    # Since we don't expose the root from search(), we instead create
    # a private root here and walk it manually using the same code.
    root_state = state_t.clone()
    if not root_state.dice:
        root_state.roll_dice()
    root = DecisionNode(state=root_state, parent=None, parent_action=None,
                        prior=0.0, flip_from_parent=False)
    root_value, root_priors = mcts.evaluate(
        root.state, root.valid_moves, root.mask
    )
    mcts.expand_decision_node(root, root_priors)
    root.N = 1
    root.W = root_value
    mcts._run_async_simulations(root, 200)

    # BFS walk: every DecisionNode reachable should have virtual_N == 0
    bad = 0
    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, DecisionNode):
            if node.virtual_N != 0:
                bad += 1
            for child in node.children.values():
                stack.append(child)
        else:
            for child in node.children_by_outcome.values():
                stack.append(child)
    if bad == 0:
        print(f"  ✓ All {len(seen)} reachable nodes have virtual_N == 0")
    else:
        print(f"  ⚠ {bad}/{len(seen)} nodes have nonzero virtual_N — bug!")

    print("\nMCTS tests complete!")