"""
mcts_az.py — AlphaZero-style PUCT search for Kingdomino.

This is the self-play / evaluation search.  It is deliberately a SEPARATE
class from the legacy UCB+heuristic MCTS:
  - No import of evaluation.py.  The heuristic never touches training data.
  - Leaf value comes ONLY from the network's value head.
  - Action priors come ONLY from the network's policy head.
  - No rollout policy, no rollouts.
A grep for "evaluation" in this file should find nothing.

PUCT SELECTION
At each node, pick a = argmax_a [ Q(a) + c_puct · P(a) · √N(s) / (1 + N(a)) ]
where P(a) is the network policy prior, N(s) the node visit count, N(a) the
edge visit count, and Q(a) the mean action value.

VALUE PERSPECTIVE — fixed player-0 frame
Every value stored in the tree is expressed from player 0's perspective.
  - Network leaf value is from state.current_actor's view → negate if actor==1.
  - Leaf value comes from the network's three value-relevant outputs:
        margin_value = tanh((own_norm − opp_norm) × MARGIN_GAIN)
        win_value    = 2 × win_prob − 1
        leaf_value   = ALPHA × margin_value + (1 − ALPHA) × win_value
    Both components are in the encoded player's frame; _postprocess flips to
    player-0 frame.
  - Terminal value = terminal_search_value(state, player=0, ...) using the SAME
    mixed formula as non-terminal leaves:
        alpha * tanh((own-opp)/score_scale * margin_gain) + (1-alpha) * win_value,
    where win_value is ±1/0 from final scores.  (Replaces compute_target_z, whose
    tanh(margin/30) scale was inconsistent with the non-terminal estimates.)
  - Backup adds the player-0 value to every node on the path; NO sign flips.
  - At selection, a node maximises from ITS actor's view: q = child.Q if
    actor==0 else -child.Q.
This avoids per-node negation bugs during backup (the classic 2-player MCTS
footgun) and is correct because the formula is zero-sum by construction:
own/opp are swapped for the opponent, so leaf_value(player 0) =
−leaf_value(player 1).

IMPERFECT INFORMATION — determinization is the CALLER'S job
`search(root_state)` treats root_state's deck as a fixed, known world (closed
-loop PUCT, perfect information from here).  For information-set-safe play the
caller must pass a *determinized* root — sample the hidden deck with
`encoder.redeterminize` first.  `run_pimc` does this and aggregates visit
counts over several determinizations (Perfect Information Monte Carlo).

Why this is safe: the encoder computes features from public information only
(it never reads deck order), so the network sees identical inputs regardless
of which determinization the engine is stepping through.  The determinization
only decides which tiles get revealed in future rounds during simulation; it
never leaks into what the network evaluates.

EVALUATOR SEAM
  Evaluator: (mb, ob, flat, idxs) -> (value, gathered_legal_logits)
    - mb, ob    (9,13,13) float32 board arrays
    - flat       (261,)   float32 flat features
    - idxs       (n,)     int64  legal joint-action indices (encode_action order)
    - value      float    in (-1,1), leaf_value = α·margin + (1-α)·win,
                          from the ENCODED player's perspective
    - logits     (n,)     float32  network logits gathered at idxs, in idxs order
  MCTS softmaxes over legal actions only.  The evaluator need never materialise
  the full 3390-entry logit vector across a process boundary; remote backends
  ship just len(idxs) floats per leaf, keeping IPC cheap.

LEAF PARALLELIZATION (the throughput lever)
`search(..., leaf_batch=N)` collects N leaves per simulation step using virtual
loss, evaluates the unique non-terminal ones in ONE batched network call via a
BatchedEvaluator, then backs up real values.

  - leaf_batch=1 (default) routes through the UNCHANGED serial `_simulate`, so
    it is bit-identical to the pre-leaf-parallel search.  Existing data, the
    correctness oracle, and the policy_compare variance floor all stay valid.
  - leaf_batch>1 uses `_simulate_batch` with virtual loss in the player-0 frame.
    This is an approximation of serial (collisions + virtual-loss bias) whose
    policy divergence must be validated with policy_compare against the floor.

Virtual loss sign in the fixed player-0 frame: a child N chosen at parent P is
discouraging its chooser (P's actor).  P's actor views N's value as +Q0 (actor
0) or -Q0 (actor 1).  To lower that view, we push N.value_sum DOWN if actor==0,
UP if actor==1: vl_value0 = -1 if chooser==0 else +1.  Root gets a visit-count
bump only (no parent defines a sign; root value is never used in selection).

OPEN-LOOP MCTS (OpenLoopMCTS — Phase 3)
Resample deck order PER SIMULATION rather than per search. Tree nodes
are keyed on action sequences; no concrete state is stored in nodes.
Each simulation reconstructs its concrete state by replaying the path
of actions from root onto a freshly drawn determinization. The root
receives the public state at search time. This is the mechanism that
makes future-conditional value (tempo, flexibility) learnable: by
averaging over many sampled futures, the search correctly prices moves
that hedge across deck draws rather than committing to one.

See OpenLoopNode for the node structure and OpenLoopMCTS (Phase 3b)
for the simulation loop.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from games.kingdomino.game import GameState, Phase
from games.kingdomino.encoder import encode_state, compute_target_z, redeterminize
from games.kingdomino.endgame_solver import exact_endgame_value
from games.kingdomino.action_codec import (
    NUM_JOINT_ACTIONS, encode_action, decode_action,
)


# Leaf-value combination hyperparameters.  leaf_value mixes a dense score-margin
# signal with the win-probability signal:
#     margin_value = tanh((own_norm − opp_norm) × MARGIN_GAIN)
#     win_value    = 2 × win_prob − 1
#     leaf_value   = ALPHA × margin_value + (1 − ALPHA) × win_value
# own/opp are already normalized by score_scale (the network head outputs
# normalized scores directly), so MARGIN_GAIN operates on normalized scores
# (typically in [-1, 1]).  MARGIN_GAIN=2.0 puts a 0.1 normalized-score difference
# (= 10 points at score_scale=100) through tanh(0.2) ≈ 0.197 — a meaningful but
# non-saturating signal.
#
# MARGIN_GAIN and ALPHA are module-level DEFAULTS ONLY.  They are bound into
# AlphaZeroMCTS/OpenLoopMCTS at construction (margin_gain/alpha params) and into
# the evaluator closures (make_serial/batched_evaluator), so changing these
# module values after construction has NO effect on a search already built.  The
# training loop forwards cfg.margin_gain / cfg.alpha explicitly through
# make_serial_evaluator / make_mcts / make_open_loop_mcts — the old
# `mcts_az.MARGIN_GAIN = cfg.margin_gain` global-override pattern is removed.
MARGIN_GAIN: float = 2.0
ALPHA: float = 0.8         # weight on margin_value; (1-ALPHA) on win_value


def terminal_search_value(
    state: GameState,
    player: int = 0,
    *,
    score_scale: float,
    margin_gain: float,
    alpha: float,
) -> float:
    """Terminal backup value in PLAYER-0 frame, using the SAME mixed formula as
    non-terminal leaves (replaces compute_target_z inside MCTS, whose tanh(margin/30)
    scale was inconsistent with the non-terminal estimates it is compared against).

      own_norm  = score0 / score_scale
      opp_norm  = score1 / score_scale
      margin    = tanh((own_norm - opp_norm) * margin_gain)
      win_value = +1.0 win / 0.0 draw / -1.0 loss   (exact, from final scores)
      result    = alpha * margin + (1 - alpha) * win_value

    win_value uses the score-only cascade (same limitation as Rust finalize_move
    — no territory/crowns tiebreaker); a score tie → 0.0.  Always returns the
    player-0 frame value (positive = good for player 0), matching compute_target_z's
    convention and the player-0-frame backup logic; `player` is accepted for call-
    site symmetry but the result is not reframed (caller negates if needed).
    """
    scores = state.scores()
    s0, s1 = float(scores[0]), float(scores[1])
    own_norm = s0 / score_scale
    opp_norm = s1 / score_scale
    margin_value = math.tanh((own_norm - opp_norm) * margin_gain)
    if s0 > s1:
        win_value = 1.0
    elif s1 > s0:
        win_value = -1.0
    else:
        win_value = 0.0        # score tie → neutral (no tiebreaker cascade)
    return alpha * margin_value + (1.0 - alpha) * win_value


class Node:
    """A search-tree node. Edge statistics are stored in the child nodes."""
    __slots__ = ("state", "prior", "visit_count", "value_sum",
                 "children", "is_expanded")

    def __init__(self, prior: float):
        self.state: Optional[GameState] = None  # set lazily when descended into
        self.prior: float = prior               # P(parent → this), from policy head
        self.visit_count: int = 0
        self.value_sum: float = 0.0             # PLAYER-0 frame
        self.children: Dict[object, "Node"] = {}
        self.is_expanded: bool = False

    @property
    def value(self) -> float:
        """Mean value in player-0 frame (0.0 if unvisited)."""
        return self.value_sum / self.visit_count if self.visit_count else 0.0


# Sentinel for OpenLoopNode.exact_value's "Unsolvable" state (solver timed out).
# A distinct object so it is never confused with a real solved float value.
_EXACT_UNSOLVABLE = object()


class OpenLoopNode:
    """A search-tree node for open-loop MCTS.

    KEY DESIGN DIFFERENCE FROM Node
    ────────────────────────────────
    Open-loop nodes are STATELESS. No concrete GameState is stored here.
    Instead, each simulation reconstructs a concrete state by replaying
    the path of actions from the root onto a freshly determinized deck,
    sampling a new deck order per simulation. This means:

      - `Node.state` (a stored GameState) does not exist here.
      - The concrete state at this node is only available within a
        simulation, as a local variable passed down the call stack.
      - Children are keyed by ACTION (same as Node), not by state.

    WHY STATELESS?
    Each simulation draws a fresh deck order (determinization), so the
    concrete state at any non-root node depends on WHICH simulation is
    visiting it. Storing one state per node would either overwrite it
    on each visit (losing other simulations' context) or require one
    state per simulation per node (memory O(sims × nodes)). Stateless
    replay costs O(depth) per simulation in CPU but is correct and
    memory-efficient.

    PICK ENCODING AND DEEP NODES
    Action keys use slot-relative pick indices (see action_codec.py
    _encode_pick). At deep nodes (future rounds after the current public
    row), the concrete domino at slot k varies across determinizations —
    this is correct and intentional. Training targets are extracted only
    at the root, where current_row is public.

    VALUE CONVENTION
    value_sum / visit_count is in the PLAYER-0 frame, matching Node.
    _postprocess handles the per-actor perspective flip, same as before.
    """
    __slots__ = ("action", "depth", "prior", "visit_count",
                 "value_sum", "children", "is_expanded", "exact_value")

    def __init__(self, action: object, depth: int, prior: float):
        self.action: object = action          # action that led to this node
                                              # (None for root)
        self.depth: int = depth               # depth from root (root = 0)
        self.prior: float = prior             # P(parent → this), from policy
        self.visit_count: int = 0
        self.value_sum: float = 0.0           # player-0 frame
        self.children: Dict[object, "OpenLoopNode"] = {}
        self.is_expanded: bool = False
        # OPT-1 solve-once cache, three states (cf. Rust ExactCache):
        #   None                -> Unsolved: not yet attempted this search.
        #   float               -> Solved: exact player-0 value; node stays
        #                          unexpanded so later sims return it in O(1).
        #   _EXACT_UNSOLVABLE   -> solver timed out; the node was expanded once
        #                          via the network and is never re-solved.
        # Only set when exact solving is "active" for the search (root
        # terminal-adjacent ⇒ this node's concrete state is the same for every
        # determinization, so a single cached value is correct for all).
        self.exact_value: Union[float, object, None] = None

    @property
    def value(self) -> float:
        """Mean value in player-0 frame (0.0 if unvisited)."""
        return self.value_sum / self.visit_count if self.visit_count else 0.0


# Evaluator seam: (mb, ob, flat, legal_idxs) -> (value, legal_logits).
# legal_idxs is the int64 array of legal joint-action indices (engine order);
# legal_logits is the network's logits GATHERED at those indices, in the same
# order.  MCTS softmaxes over legal actions only, so the evaluator need never
# materialise the full action vector across a process boundary — the remote
# backend ships just len(legal_idxs) floats per leaf.
Evaluator = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[float, np.ndarray]
]

# BatchedEvaluator: stacked inputs + per-leaf legal indices → (values, gathered).
#     (mbs (K,9,13,13), obs (K,9,13,13), flats (K,261), idxs_list[K])
#       -> (values (K,), [gathered_legal_logits])  (each (n_legal_i,))
# Mirrors the single Evaluator's 4-arg/gathered contract, just batched.  Returning
# gathered (not full) logits keeps the IPC response small; the in-process backend
# gathers cheaply too.  With leaf_batch=1 unused; only _simulate_batch calls it.
BatchedEvaluator = Callable[
    [np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]],
    Tuple[np.ndarray, List[np.ndarray]]
]


def make_serial_evaluator(
    network: torch.nn.Module,
    device: str = "cpu",
    *,
    margin_gain: float = MARGIN_GAIN,
    alpha: float = ALPHA,
) -> Evaluator:
    """Wrap a network into a batch-1 Evaluator (eval mode, inference_mode, on
    device).  margin_gain/alpha are bound into the closure here (Fix 2) — no
    module-global reads at search time."""
    network = network.to(device).eval()
    mg = float(margin_gain)
    al = float(alpha)

    def evaluator(mb: np.ndarray, ob: np.ndarray, flat: np.ndarray,
                  idxs: np.ndarray):
        with torch.inference_mode():
            mb_t = torch.from_numpy(mb).unsqueeze(0).to(device)
            ob_t = torch.from_numpy(ob).unsqueeze(0).to(device)
            flat_t = torch.from_numpy(flat).unsqueeze(0).to(device)
            own, opp, win_prob, logits = network(mb_t, ob_t, flat_t)
            # Full leaf_value: convex mix of score-margin and win signals, in the
            # encoded player's frame (own/opp/win_prob all from that player's view).
            # _postprocess flips to the player-0 frame.
            own_n = own.item()
            opp_n = opp.item()
            margin_value = math.tanh((own_n - opp_n) * mg)
            win_value = 2.0 * float(win_prob.item()) - 1.0
            leaf_value = al * margin_value + (1.0 - al) * win_value
            # GPU-side gather (Fix 3): transfer only the legal logits, not all 3390.
            idx_t = torch.as_tensor(idxs, device=device, dtype=torch.long)
            gathered = logits[0].index_select(0, idx_t).detach().cpu().numpy()
        return leaf_value, gathered

    return evaluator


def make_batched_evaluator(
    network: torch.nn.Module,
    device: str = "cpu",
    *,
    margin_gain: float = MARGIN_GAIN,
    alpha: float = ALPHA,
) -> BatchedEvaluator:
    """Wrap a network into a BatchedEvaluator: ONE forward over K positions, then
    gather each leaf's legal logits.  This is the in-process counterpart of the
    IPC batched evaluator; both honor the same (mbs, obs, flats, idxs_list) ->
    (values, [gathered]) contract so _simulate_batch is backend-agnostic.
    margin_gain/alpha bound in the closure (Fix 2).

    With a batch of 1, numerically equal to make_serial_evaluator up to FP
    accumulation order (~1e-6) — the same cuDNN noise already in the floor.
    """
    network = network.to(device).eval()
    mg = float(margin_gain)
    al = float(alpha)

    def evaluator(mbs: np.ndarray, obs: np.ndarray, flats: np.ndarray,
                  idxs_list: List[np.ndarray]):
        with torch.inference_mode():
            mb_t = torch.from_numpy(np.ascontiguousarray(mbs)).to(device)
            ob_t = torch.from_numpy(np.ascontiguousarray(obs)).to(device)
            flat_t = torch.from_numpy(np.ascontiguousarray(flats)).to(device)
            own, opp, win_prob, logits = network(mb_t, ob_t, flat_t)
            # Full leaf_value (vectorized): convex mix of score-margin and win
            # signals, per leaf, in each encoded player's frame.
            own_n = own.reshape(-1).detach().cpu().numpy()       # (K,)
            opp_n = opp.reshape(-1).detach().cpu().numpy()       # (K,)
            win_p = win_prob.reshape(-1).detach().cpu().numpy()  # (K,)
            margin_values = np.tanh((own_n - opp_n) * mg)
            win_values = 2.0 * win_p - 1.0
            values = al * margin_values + (1.0 - al) * win_values  # (K,)
            # GPU-side gather (Fix 3): per leaf, transfer only its legal logits.
            gathered = []
            for i in range(len(idxs_list)):
                idx_t = torch.as_tensor(idxs_list[i], device=device, dtype=torch.long)
                gathered.append(
                    logits[i].index_select(0, idx_t).detach().cpu().numpy())
        return values, gathered

    return evaluator


class AlphaZeroMCTS:
    def __init__(
        self,
        evaluator: Evaluator,
        *,
        batched_evaluator: Optional[BatchedEvaluator] = None,
        c_puct: float = 1.5,
        n_simulations: int = 100,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        fpu: float = 0.0,
        virtual_loss: int = 1,
        score_scale: float = 100.0,
        margin_gain: float = 2.0,
        alpha: float = 0.8,
    ):
        self.evaluator = evaluator
        # Terminal-value formula params, bound at construction (Fix 1/2): the
        # GAME_OVER backup uses terminal_search_value with these, matching the
        # non-terminal leaf-value scale.
        self._score_scale = float(score_scale)
        self._margin_gain = float(margin_gain)
        self._alpha = float(alpha)
        # Used ONLY by leaf_batch>1 path.  If None, _evaluate_batch loops the
        # single evaluator — correct but with no GPU-batching gain (fine for the
        # oracle and policy_compare divergence tests; supply make_batched_evaluator
        # for throughput).
        self._batched_evaluator: Optional[BatchedEvaluator] = batched_evaluator
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.fpu = fpu
        self.virtual_loss = int(virtual_loss)

    # ── public ──
    def search(
        self,
        root_state: GameState,
        *,
        add_noise: bool = False,
        rng: Optional[np.random.Generator] = None,
        leaf_batch: int = 1,
    ) -> Tuple[Dict[object, int], Node]:
        """Run n_simulations of PUCT from `root_state`.

        Returns (visit_counts, root):
            visit_counts : {action: edge visit count} at the root
            root         : the root Node (for value / diagnostics)

        leaf_batch:
            1 (default) → serial search, bit-identical to the pre-LP code.
            >1          → leaf parallelization: collect up to `leaf_batch` leaves
                          per step, evaluate unique non-terminal ones in one
                          batched call, back up real values.  This is an
                          approximation of serial; validate divergence with
                          policy_compare against the variance floor before use.

        Total leaf budget is exactly n_simulations regardless of leaf_batch
        (last chunk shrinks to fit), so divergence comparisons hold sims fixed.
        """
        if root_state.phase == Phase.GAME_OVER:
            raise ValueError("Cannot search from a terminal state.")

        root = Node(prior=1.0)
        root.state = root_state
        root_value0 = self._expand(root)
        root.visit_count = 1
        root.value_sum = root_value0

        if add_noise:
            self._add_dirichlet_noise(root, rng)

        if leaf_batch <= 1:
            # Untouched serial path — bit-identical to pre-LP code.
            for _ in range(self.n_simulations):
                self._simulate(root)
        else:
            remaining = self.n_simulations
            while remaining > 0:
                b = min(leaf_batch, remaining)
                self._simulate_batch(root, b)
                remaining -= b

        visit_counts = {a: c.visit_count for a, c in root.children.items()}
        return visit_counts, root

    # ── descent: shared by serial and batched ──────────────────────────────
    def _descend(self, root: Node) -> List[Node]:
        """Descend by PUCT until an unexpanded node or a terminal.

        Returns the full path [root, ..., leaf].  Sets child.state lazily on
        the way down (functional step()), exactly as the serial loop does.
        No virtual loss is applied here — it is applied AFTER the descent
        completes, so VL from earlier descents in the same batch affects the
        PUCT scores read here but never the descent code itself.
        """
        path = [root]
        node = root
        while node.is_expanded and not (node.state.phase == Phase.GAME_OVER):
            action, child = self._select_child(node)
            if child.state is None:
                child.state = node.state.step(action)
            path.append(child)
            node = child
        return path

    # ── serial simulation (leaf_batch == 1) ────────────────────────────────
    def _simulate(self, root: Node) -> None:
        path = [root]
        node = root
        # Descend until an unexpanded node or a terminal state.
        while node.is_expanded and not (node.state.phase == Phase.GAME_OVER):
            action, child = self._select_child(node)
            if child.state is None:
                child.state = node.state.step(action)
            path.append(child)
            node = child

        if node.state.phase == Phase.GAME_OVER:
            v0 = terminal_search_value(
                node.state, player=0, score_scale=self._score_scale,
                margin_gain=self._margin_gain, alpha=self._alpha)
        else:
            v0 = self._expand(node)

        # Backup — all values in player-0 frame, no sign flips.
        for n in path:
            n.visit_count += 1
            n.value_sum += v0

    # ── leaf-parallel simulation (leaf_batch > 1) ───────────────────────────
    def _simulate_batch(self, root: Node, batch_size: int) -> None:
        """Collect `batch_size` leaves with virtual loss, evaluate the unique
        non-terminal ones in one batched call, remove VL, back up real values.

        Each collected path counts as one simulation toward the budget.  A
        collision (two descents reaching the same unexpanded leaf) backs that
        leaf's value up twice — exactly as if two simulations hit it.  Virtual
        loss keeps collisions rare.
        """
        pending: List[Tuple[List[Node], Node]] = []
        for _ in range(batch_size):
            path = self._descend(root)
            pending.append((path, path[-1]))
            self._apply_virtual_loss(path, +1)

        # Unique non-terminal leaves needing a network evaluation.
        # Dedup by node identity: expand each leaf exactly once (a second
        # expansion would overwrite freshly-created child stats).
        unique_leaves: List[Node] = []
        seen: set = set()
        for _, leaf in pending:
            if leaf.state.phase == Phase.GAME_OVER:
                continue
            if id(leaf) in seen:
                continue
            seen.add(id(leaf))
            unique_leaves.append(leaf)

        leaf_v0: Dict[int, float] = {}
        if unique_leaves:
            results = self._evaluate_batch([lf.state for lf in unique_leaves])
            for lf, (v0, priors) in zip(unique_leaves, results):
                if not lf.is_expanded:
                    for action, p in priors.items():
                        lf.children[action] = Node(prior=p)
                    lf.is_expanded = True
                leaf_v0[id(lf)] = v0

        # Remove VL over the EXACT same paths — exact additive inverse.
        for path, _ in pending:
            self._apply_virtual_loss(path, -1)

        # Real backup — player-0 frame, no sign flips.
        for path, leaf in pending:
            if leaf.state.phase == Phase.GAME_OVER:
                v0 = terminal_search_value(
                    leaf.state, player=0, score_scale=self._score_scale,
                    margin_gain=self._margin_gain, alpha=self._alpha)
            else:
                v0 = leaf_v0[id(leaf)]
            for n in path:
                n.visit_count += 1
                n.value_sum += v0

    def _apply_virtual_loss(self, path: Sequence[Node], sign: int) -> None:
        """Apply (sign=+1) or remove (sign=-1) virtual loss along `path`.

        Virtual loss makes the just-collected path look pessimistic to the
        actors who chose it, so the next descent in the batch explores elsewhere.

        In the fixed player-0 frame, child N at parent P is scored by P's actor
        as +N.value (actor 0) or -N.value (actor 1).  To lower that score we
        push N.value_sum DOWN if actor==0, UP if actor==1:
            vl_value0 = -1 if chooser==0 else +1

        Root (index 0) has no chooser, so it gets a visit-count bump only —
        that inflates sqrt(N_root) in the next descent's u-term uniformly, the
        standard mild diversifier; root.value_sum adjustment is irrelevant since
        no parent selects the root.

        Apply with +1 during collection and remove with -1 over the same path
        list.  Since each edit is path-scoped and additive, removal is an exact
        inverse even when paths share prefixes or collide.
        """
        n_vl = self.virtual_loss
        if n_vl <= 0:
            return
        for i, node in enumerate(path):
            node.visit_count += sign * n_vl
            if i > 0:
                chooser = path[i - 1].state.current_actor
                vl_value0 = -1.0 if chooser == 0 else 1.0
                node.value_sum += sign * n_vl * vl_value0

    # ── evaluation ──────────────────────────────────────────────────────────
    def _select_child(self, node: Node) -> Tuple[object, Node]:
        actor = node.state.current_actor
        sqrt_n = math.sqrt(node.visit_count)
        best_score = -math.inf
        best = None
        for action, child in node.children.items():
            if child.visit_count > 0:
                q0 = child.value_sum / child.visit_count   # player-0 frame
                q = q0 if actor == 0 else -q0               # → actor's frame
            else:
                q = self.fpu
            u = self.c_puct * child.prior * sqrt_n / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best = (action, child)
        return best

    def _expand(self, node: Node) -> float:
        """Evaluate the network at node.state, create child edges with priors.

        Returns the leaf value in PLAYER-0 frame.
        """
        state = node.state
        value0, priors = self._evaluate(state)
        for action, p in priors.items():
            node.children[action] = Node(prior=p)
        node.is_expanded = True
        return value0

    def _evaluate(self, state: GameState) -> Tuple[float, Dict[object, float]]:
        """Single network evaluation (4-arg evaluator seam).
        Returns (value_player0, {action: prior}).
        """
        legal = state.legal_actions()
        if not legal:
            raise ValueError(
                f"_evaluate received a non-terminal state with no legal "
                f"actions (phase={state.phase.name})."
            )
        mb, ob, flat = encode_state(state, state.current_actor)
        # Legal action indices (engine order), computed BEFORE inference so the
        # evaluator returns only these logits.
        idxs = np.fromiter((encode_action(a, state) for a in legal),
                           dtype=np.int64, count=len(legal))
        # Defensive (Fix 5): duplicate joint indices would silently corrupt priors.
        assert len(set(idxs.tolist())) == len(idxs), (
            f"Duplicate legal joint indices in state phase={state.phase.name}: "
            f"{sorted(c for c in idxs.tolist() if idxs.tolist().count(c) > 1)}"
        )
        v, legal_logits = self.evaluator(mb, ob, flat, idxs)

        if not np.isfinite(legal_logits).all() or not math.isfinite(v):
            raise FloatingPointError(
                "Evaluator returned NaN/Inf in MCTS evaluation."
            )
        return self._postprocess(state, v, legal_logits, legal)

    def _evaluate_batch(
        self, states: Sequence[GameState]
    ) -> List[Tuple[float, Dict[object, float]]]:
        """Batched network evaluation over N non-terminal states.

        If a BatchedEvaluator was supplied: one forward pass over all N
        positions, then gather per-sample legal logits and softmax.

        Otherwise: fall back to looping _evaluate (correct, no GPU-batching win;
        useful for correctness checks and policy_compare divergence tests).

        Returns a list aligned with `states`: (value_player0, {action: prior}).
        """
        if self._batched_evaluator is None:
            # Fallback: loop the single evaluator — correct, just not batched.
            return [self._evaluate(s) for s in states]

        legals = []
        idxs_list = []
        encs = []
        for s in states:
            legal = s.legal_actions()
            if not legal:
                raise ValueError(
                    f"_evaluate_batch: non-terminal state has no legal "
                    f"actions (phase={s.phase.name})."
                )
            legals.append(legal)
            idxs = np.fromiter((encode_action(a, s) for a in legal),
                               dtype=np.int64, count=len(legal))
            # Defensive (Fix 5): duplicate joint indices would corrupt priors.
            assert len(set(idxs.tolist())) == len(idxs), (
                f"Duplicate legal joint indices in _evaluate_batch at state "
                f"phase={s.phase.name}"
            )
            idxs_list.append(idxs)
            encs.append(encode_state(s, s.current_actor))

        mbs = np.stack([e[0] for e in encs])
        obs = np.stack([e[1] for e in encs])
        flats = np.stack([e[2] for e in encs])

        # One batched call — returns per-leaf values and GATHERED legal logits
        # (the evaluator/server does the gather, keeping IPC responses small).
        values, gathered = self._batched_evaluator(mbs, obs, flats, idxs_list)

        out: List[Tuple[float, Dict[object, float]]] = []
        for i, s in enumerate(states):
            legal_logits = gathered[i]
            if not np.isfinite(legal_logits).all() or not math.isfinite(float(values[i])):
                raise FloatingPointError(
                    "BatchedEvaluator returned NaN/Inf in MCTS evaluation."
                )
            out.append(self._postprocess(s, float(values[i]), legal_logits, legals[i]))
        return out

    def _postprocess(
        self, state: GameState, v: float, legal_logits: np.ndarray, legal,
    ) -> Tuple[float, Dict[object, float]]:
        """Shared post-processing: stable softmax over gathered legal logits,
        player-0 framing of the value.

        `legal_logits` is already gathered in `legal` order — no further
        indexing needed.  Used by both _evaluate (single path) and
        _evaluate_batch (batched path), so they cannot drift numerically.
        """
        legal_logits = legal_logits - legal_logits.max()  # stable softmax
        exp = np.exp(legal_logits)
        probs = exp / exp.sum()
        priors = {a: float(p) for a, p in zip(legal, probs)}
        value0 = v if state.current_actor == 0 else -v
        return value0, priors

    def _add_dirichlet_noise(self, root: Node, rng: Optional[np.random.Generator]) -> None:
        if rng is None:
            rng = np.random.default_rng()
        actions = list(root.children.keys())
        if not actions:
            return
        noise = rng.dirichlet([self.dirichlet_alpha] * len(actions))
        eps = self.dirichlet_epsilon
        for action, n in zip(actions, noise):
            child = root.children[action]
            child.prior = (1 - eps) * child.prior + eps * float(n)


class OpenLoopMCTS:
    """Open-loop MCTS for imperfect-information Kingdomino.

    OPEN-LOOP VS CLOSED-LOOP
    ────────────────────────
    Closed-loop (AlphaZeroMCTS): one deck order is fixed at the start of
    search (via redeterminize at the root). All simulations share that
    deck. The tree stores concrete GameStates in nodes.

    Open-loop (this class): the deck order is resampled PER SIMULATION.
    Nodes store no concrete state — each simulation reconstructs its
    concrete state by replaying the action path from root onto a freshly
    drawn deck. This averages over many futures instead of committing to
    one, making future-conditional value (tempo, flexibility, blocking)
    visible to the search.

    CHILDREN ARE KEYED BY SLOT-RELATIVE JOINT INDEX (not action objects)
    ───────────────────────────────────────────────────────────────────
    This is the one substantive implementation difference from the Phase 3b
    pseudocode, and it is REQUIRED for correctness. An engine action object
    (TurnAction) carries a CONCRETE pick_domino_id. At a deep node the
    concrete domino in a given pick slot differs across determinizations, so
    a child's stored action object cannot be re-encoded or replayed in a
    different simulation's state (its domino_id is not in that row).

    The joint index, by contrast, is determinization-invariant: its pick
    component is the SLOT (0..3), not the domino (see action_codec
    `_encode_pick`), and its placement component is board coordinates
    (public). So children are keyed by joint index — a stable, slot-relative
    identifier valid in every determinization — and each simulation calls
    `decode_action(idx, concrete_state)` to obtain the concrete action to
    step. At the root, children indices are converted back to engine action
    objects (root current_row is public) so the returned visit_counts match
    AlphaZeroMCTS's {action: count} contract.

    INTERFACE COMPATIBILITY
    ───────────────────────
    search() returns (visit_counts, root) with the same types as
    AlphaZeroMCTS.search() — visit_counts keyed by engine action objects —
    so callers (visit_counts_to_policy, select_move, play_selfplay_game)
    work unchanged. run_pimc's outer determinization loop is redundant for
    open-loop (it averages internally); use run_pimc_open_loop and set
    n_determinizations=1.

    EVALUATOR SEAM
    ──────────────
    Same as AlphaZeroMCTS: Evaluator = (mb, ob, flat, idxs) ->
    (value, gathered_logits). The leaf_value is in the encoded player's
    frame; the value is flipped to the player-0 frame for backup.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        c_puct: float = 1.5,
        n_simulations: int = 100,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        fpu: float = 0.0,
        virtual_loss: int = 1,
        score_scale: float = 100.0,
        margin_gain: float = 2.0,
        alpha: float = 0.8,
        exact_endgame_max_secs: float = 3.0,
        exact_endgame_max_hidden_tiles: int = 4,
        exact_endgame_enabled: bool = True,
    ):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.fpu = fpu
        # virtual_loss: in serial Python OpenLoopMCTS, VL is applied and then
        # immediately undone within a SINGLE simulation — no second simultaneous
        # descent observes it, so it has no effect on search quality here.  It is
        # implemented correctly (parent/chooser actor sign, linear magnitude) for
        # when this logic is ported to the batched Rust path, where concurrent
        # descents DO observe it.  Default 1 also hid a squared-magnitude bug at
        # VL>1 (now fixed: linear self.virtual_loss * vl_value0).
        self.virtual_loss = int(virtual_loss)
        # Terminal-value formula params, bound at construction (Fix 1/2).
        self._score_scale = float(score_scale)
        self._margin_gain = float(margin_gain)
        self._alpha = float(alpha)
        self._exact_endgame_max_secs = float(exact_endgame_max_secs)
        self._exact_endgame_max_hidden_tiles = int(exact_endgame_max_hidden_tiles)
        self._exact_endgame_enabled = bool(exact_endgame_enabled)
        # Whether exact endgame solving is active for the CURRENT search. Set per
        # search() from the root: only true when the root is terminal-adjacent
        # (deck small enough to solve), which is also the condition under which a
        # solved leaf's value is determinization-independent and therefore safe
        # to cache (OPT-1). See search().
        self._exact_endgame_active = False
        self._exact_solve_count = 0      # leaves solved exactly this search
        self._exact_cache_hits = 0       # leaves served from OPT-1 cache
        self._exact_fallback_count = 0   # solver timed out this search (≤1: see _simulate)
        # Diagnostic: number of times _select_child found no stored child legal
        # in the current determinization (a deep-node divergence). Reset per
        # search(); expected to be rare. See _select_child.
        self._fallback_count = 0

    # ── public ──
    def search(
        self,
        root_state: GameState,
        *,
        add_noise: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[Dict[object, int], "OpenLoopNode"]:
        """Run n_simulations of open-loop PUCT from `root_state`.

        `root_state` is the PUBLIC state — boards, current_row, claims,
        phase are known; deck order is hidden and will be resampled.

        Returns (visit_counts, root):
            visit_counts : {action: edge visit count} at the root
            root         : the root OpenLoopNode

        RNG: a single np.random.Generator drives both Dirichlet noise
        (numpy) and deck resampling (derived Python random.Random).
        """
        if root_state.phase == Phase.GAME_OVER:
            raise ValueError("Cannot search from a terminal state.")

        if rng is None:
            rng = np.random.default_rng()

        # Derive a Python RNG for redeterminize from the numpy RNG. One
        # derivation per search call — py_rng then drives ALL per-simulation
        # deck shuffles within this search, keeping the sequence deterministic
        # given a fixed np rng seed.
        py_rng = random.Random(int(rng.integers(0, 2**63 - 1)))
        self._fallback_count = 0
        self._exact_solve_count = 0
        self._exact_cache_hits = 0
        self._exact_fallback_count = 0

        # Exact endgame solving is enabled for this search only when the ROOT is
        # terminal-adjacent (deck <= max_hidden_tiles). In that regime the only
        # hidden information is the ORDER of the few remaining deck tiles, which
        # is irrelevant — they are dealt as a sorted row — so every node's
        # concrete state is identical across determinizations. That makes the
        # exact solver's value (a) correct to back up and (b) safe to cache once
        # per node (OPT-1). From a non-terminal-adjacent root, a deck=4 leaf's
        # board would depend on which tiles a determinization happened to deal,
        # so a single cached value would be wrong; we fall back to the network
        # there and let the exact grounding happen later, when the game actually
        # reaches a terminal-adjacent position and that search solves it.
        self._exact_endgame_active = (
            self._exact_endgame_enabled
            and len(root_state.deck) <= self._exact_endgame_max_hidden_tiles
        )

        # Expand root using a fresh determinization. redeterminize preserves
        # the public current_row, so the root's child joint indices are the
        # same regardless of which determinization is drawn here.
        root = OpenLoopNode(action=None, depth=0, prior=1.0)
        det = redeterminize(root_state, py_rng)
        root_value0 = self._expand(root, det)
        root.visit_count = 1
        root.value_sum = root_value0

        if add_noise:
            self._add_dirichlet_noise(root, rng)

        for _ in range(self.n_simulations):
            det = redeterminize(root_state, py_rng)
            self._simulate(root, root_state, det)

        # Convert slot-relative joint-index keys back to engine action objects
        # using the PUBLIC root state (current_row is public at the root, so
        # this is deterministic and determinization-independent).
        visit_counts = {
            decode_action(idx, root_state): c.visit_count
            for idx, c in root.children.items()
        }
        return visit_counts, root

    # ── simulation ──
    def _simulate(
        self,
        root: "OpenLoopNode",
        root_state: GameState,
        det: GameState,
    ) -> None:
        """One open-loop simulation.

        Descends the tree by PUCT, reconstructing the concrete state at each
        node by stepping `det` forward with the selected (decoded) actions.
        Expands and evaluates the leaf, applies virtual loss during descent,
        then undoes the virtual loss and backs up the real value.

        `root_state` is the public state (used only to anchor depth-0; not
        stepped). `det` is the determinized copy for THIS simulation.
        """
        path: List["OpenLoopNode"] = [root]
        vl_values: List[float] = [0.0]   # root (index 0) gets no virtual loss
        node = root
        state = det   # concrete state, stepped as we descend

        # ── descent ──
        dead_end = False
        while node.is_expanded and state.phase != Phase.GAME_OVER:
            sel = self._select_child(node, state)
            if sel is None:
                # No stored child is legal in THIS determinization (deep-node
                # divergence). Stop descending and evaluate the current
                # concrete state as the leaf — we cannot step an action that is
                # illegal here, and the stale children belong to a different
                # future. Rare; counted in self._fallback_count.
                dead_end = True
                break
            idx, child = sel
            # Fix 4: the chooser is the PARENT actor — the one selecting the
            # child — read BEFORE stepping.  (The old code read
            # state.current_actor AFTER step = the CHILD actor, which is wrong
            # whenever the two differ.)  The parent is always non-terminal here,
            # so the chooser is always well-defined even if the child is terminal.
            chooser = state.current_actor
            action = decode_action(idx, state)   # concrete action for THIS sim
            state = state.step(action)
            # Virtual loss in the player-0 frame (same purpose as AlphaZeroMCTS:
            # discourages re-selection by the chooser).  vl_value0 ∈ {-1,+1};
            # the magnitude is applied LINEARLY (self.virtual_loss * vl_value0),
            # not squared.  Stored per node so the undo before backup is an exact
            # inverse.
            vl_value0 = -1.0 if chooser == 0 else 1.0
            if self.virtual_loss:
                child.visit_count += self.virtual_loss
                child.value_sum += self.virtual_loss * vl_value0
            path.append(child)
            vl_values.append(vl_value0)
            node = child

        # ── evaluate leaf ──
        if state.phase == Phase.GAME_OVER:
            value0 = terminal_search_value(
                state, player=0, score_scale=self._score_scale,
                margin_gain=self._margin_gain, alpha=self._alpha)
        elif dead_end:
            # Stale node already expanded for a different future; do NOT
            # re-expand (that would mix determinizations' children). Just use
            # the network's value estimate for the current concrete state.
            value0, _ = self._evaluate(state)
        elif isinstance(node.exact_value, float):
            # OPT-1 solve-once cache (Solved): this leaf was already solved exactly
            # in a previous simulation. The value is determinization-independent
            # (see search()'s _exact_endgame_active note), so return it directly.
            # The node stays unexpanded, so PUCT descent never goes below it.
            value0 = node.exact_value
            self._exact_cache_hits += 1
        else:
            # node.exact_value is None (Unsolved) — an Unsolvable node was expanded
            # on its timeout, so descent never stops on it again.
            if node.exact_value is None and self._should_exact_solve(state):
                # The exact solver is deterministic and public-bag exhaustive;
                # the seed is stable documentation of this simulation's
                # determinized hidden bag if future tie-breaking ever needs it.
                seed = sum((i + 1) * int(d) for i, d in enumerate(state.deck))
                value0, solved = exact_endgame_value(
                    state,
                    max_secs=self._exact_endgame_max_secs,
                    rng=random.Random(seed),
                    score_scale=self._score_scale,
                    margin_gain=self._margin_gain,
                    alpha=self._alpha,
                )
                if solved:
                    # Cache on the node and leave it unexpanded: every later
                    # simulation reaching this leaf takes the O(1) branch above.
                    node.exact_value = value0
                    self._exact_solve_count += 1
                else:
                    # Timed out: mark this node Unsolvable, expand it once via the
                    # network, and give up exact solving for the REST of this
                    # search. The position is too hard within budget, so other
                    # endgame leaves would just time out too — re-attempting them
                    # is the retry storm. (Mirrors the Rust per-game sentinel.)
                    node.exact_value = _EXACT_UNSOLVABLE
                    self._exact_fallback_count += 1
                    self._exact_endgame_active = False
                    value0 = self._expand_and_evaluate(node, state)
            else:
                value0 = self._expand_and_evaluate(node, state)

        # ── undo virtual loss, then backup (player-0 frame, no sign flips) ──
        for i, (n, vl) in enumerate(zip(path, vl_values)):
            if self.virtual_loss and i > 0:
                # Root (i == 0) never received virtual loss; undo only the
                # children that did, reversing the exact amounts applied.
                n.visit_count -= self.virtual_loss
                n.value_sum -= vl * self.virtual_loss
            n.visit_count += 1
            n.value_sum += value0

    # ── expansion / evaluation ──
    def _expand_node(self, node: "OpenLoopNode", state: GameState) -> float:
        """Evaluate `state` with the network and create children keyed by the
        legal slot-relative joint indices. Returns value in player-0 frame."""
        value0, priors = self._evaluate(state)   # priors: {joint_idx: prior}
        for idx, p in priors.items():
            node.children[idx] = OpenLoopNode(
                action=idx, depth=node.depth + 1, prior=p)
        node.is_expanded = True
        return value0

    def _expand(self, node: "OpenLoopNode", state: GameState) -> float:
        """Expand the root at search start. See _expand_node."""
        return self._expand_node(node, state)

    def _expand_and_evaluate(self, node: "OpenLoopNode", state: GameState) -> float:
        """Expand a leaf node during simulation and return value0. See _expand_node."""
        return self._expand_node(node, state)

    def _should_exact_solve(self, state: GameState) -> bool:
        """Whether to try public-bag exact endgame solving for this leaf.

        Requires `_exact_endgame_active` (set per search() from the root): exact
        solving only runs when the root is terminal-adjacent, which guarantees
        this leaf's concrete state is determinization-independent. The per-leaf
        deck check is then always satisfied (deck only shrinks while descending),
        but is kept as a cheap, explicit guard.
        """
        return (
            self._exact_endgame_active
            and state.phase in (Phase.PLACE_AND_SELECT, Phase.FINAL_PLACEMENT)
            and len(state.deck) <= self._exact_endgame_max_hidden_tiles
        )

    def _evaluate(self, state: GameState) -> Tuple[float, Dict[int, float]]:
        """Network evaluation. Returns (value_player0, {joint_idx: prior}).

        Same logic as AlphaZeroMCTS._evaluate / _postprocess (encode, gather
        legal logits, stable softmax, flip value to player-0 frame), but the
        prior dict is keyed by the slot-relative joint INDEX rather than the
        concrete action object — see the class docstring for why.
        """
        legal = state.legal_actions()
        if not legal:
            raise ValueError(
                f"_evaluate received a non-terminal state with no legal "
                f"actions (phase={state.phase.name})."
            )
        mb, ob, flat = encode_state(state, state.current_actor)
        idxs = np.fromiter((encode_action(a, state) for a in legal),
                           dtype=np.int64, count=len(legal))
        # Defensive (Fix 5): duplicate joint indices would collide in the prior
        # dict and silently drop children.
        assert len(set(idxs.tolist())) == len(idxs), (
            f"Duplicate legal joint indices in OpenLoop _evaluate at state "
            f"phase={state.phase.name}"
        )
        v, legal_logits = self.evaluator(mb, ob, flat, idxs)
        if not np.isfinite(legal_logits).all() or not math.isfinite(v):
            raise FloatingPointError(
                "Evaluator returned NaN/Inf in MCTS evaluation."
            )
        legal_logits = legal_logits - legal_logits.max()   # stable softmax
        exp = np.exp(legal_logits)
        probs = exp / exp.sum()
        priors = {int(i): float(p) for i, p in zip(idxs, probs)}
        value0 = v if state.current_actor == 0 else -v
        return value0, priors

    def _select_child(
        self, node: "OpenLoopNode", state: GameState
    ) -> Optional[Tuple[int, "OpenLoopNode"]]:
        """Select a child by PUCT score among children legal in `state`.

        LEGAL ACTION FILTERING AT DEEP NODES
        ─────────────────────────────────────
        Children are keyed by slot-relative joint index. At depth > 0 the
        concrete current_row (and the domino-in-hand, which sets legal
        placements) differ across determinizations, so a child index created
        in one determinization may not be legal in THIS one. We therefore
        intersect the node's child indices with the set of legal joint indices
        in this simulation's concrete state and select only among those.

        Returns (joint_idx, child), or None if NO stored child is legal in
        this determinization. The caller (_simulate) treats None as a
        dead-end: it stops descending and evaluates the current state, since
        no stored action can be legally replayed here. This is theoretically
        reachable (a future row whose legal placements are disjoint from the
        stored children) but rare — placement legality is mostly independent
        of which pick slot was taken. Counted in self._fallback_count.
        """
        legal = state.legal_actions()
        # encode_action against the CONCRETE state: pick component is the slot
        # in THIS row, placement is board coords — yields the legal joint-index
        # set for this determinization.
        legal_set = {encode_action(a, state) for a in legal}

        legal_children = {
            idx: c for idx, c in node.children.items() if idx in legal_set
        }
        if not legal_children:
            self._fallback_count += 1
            return None

        sqrt_n = math.sqrt(max(1, node.visit_count))
        actor = state.current_actor
        best_score = -math.inf
        best_idx = None
        best_child = None
        for idx, child in legal_children.items():
            if child.visit_count > 0:
                q0 = child.value            # player-0 frame
                q = q0 if actor == 0 else -q0
            else:
                q = self.fpu
            u = self.c_puct * child.prior * sqrt_n / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_idx = idx
                best_child = child
        return best_idx, best_child

    def _add_dirichlet_noise(
        self, root: "OpenLoopNode", rng: Optional[np.random.Generator]
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        actions = list(root.children.keys())
        if not actions:
            return
        noise = rng.dirichlet([self.dirichlet_alpha] * len(actions))
        eps = self.dirichlet_epsilon
        for action, n in zip(actions, noise):
            child = root.children[action]
            child.prior = (1 - eps) * child.prior + eps * float(n)


# ─── helpers ──────────────────────────────────────────────────────────────
def root_value_for_actor(root: Node, state: GameState) -> float:
    """Root value in the current actor's frame (player-0 stored frame → actor)."""
    v0 = root.value
    return v0 if state.current_actor == 0 else -v0


def visit_counts_to_policy(
    visit_counts: Dict[object, int],
    state: GameState,
    temperature: float = 1.0,
) -> np.ndarray:
    """Convert visit counts to a (NUM_JOINT_ACTIONS,) policy target."""
    policy = np.zeros(NUM_JOINT_ACTIONS, dtype=np.float32)
    if not visit_counts:
        raise ValueError("Cannot build policy target from empty visit_counts.")
    actions = list(visit_counts.keys())
    counts = np.array([visit_counts[a] for a in actions], dtype=np.float64)
    if counts.sum() <= 0:
        raise ValueError("All root visit counts are zero; increase n_simulations.")

    encoded = [encode_action(a, state) for a in actions]
    # Defensive (Fix 5): two actions mapping to the same joint index would
    # overwrite each other in the policy vector, losing visit mass.
    assert len(set(encoded)) == len(encoded), (
        f"Duplicate joint indices in visit_counts_to_policy: "
        f"phase={state.phase.name}"
    )

    if temperature <= 1e-6:
        policy[encoded[int(counts.argmax())]] = 1.0
        return policy

    weights = counts ** (1.0 / temperature)
    weights /= weights.sum()
    for idx, w in zip(encoded, weights):
        policy[idx] = w
    return policy


def select_move(
    visit_counts: Dict[object, int],
    temperature: float,
    rng: np.random.Generator,
) -> object:
    """Pick a move from visit counts. temperature=0 → greedy."""
    actions = list(visit_counts.keys())
    if not actions:
        raise ValueError("Cannot select a move from empty visit_counts.")
    counts = np.array([visit_counts[a] for a in actions], dtype=np.float64)
    if counts.sum() <= 0:
        raise ValueError("All root visit counts are zero; increase n_simulations.")
    if temperature <= 1e-6:
        return actions[int(counts.argmax())]
    weights = counts ** (1.0 / temperature)
    weights /= weights.sum()
    return actions[int(rng.choice(len(actions), p=weights))]


def run_pimc(
    mcts: AlphaZeroMCTS,
    public_state: GameState,
    py_rng: random.Random,
    *,
    n_determinizations: int = 1,
    add_noise: bool = False,
    np_rng: Optional[np.random.Generator] = None,
    leaf_batch: int = 1,
) -> Tuple[Dict[object, int], float]:
    """Perfect Information Monte Carlo over several determinizations.

    Samples `n_determinizations` worlds, runs PUCT on each, aggregates root
    visit counts.  Returns (aggregated_visit_counts, mean_root_value in
    player-0 frame).

    `leaf_batch` is threaded into each per-determinization search call.
    leaf_batch=1 is the serial reference (bit-identical to pre-LP).
    """
    agg: Dict[object, float] = defaultdict(float)
    total_value0 = 0.0
    for _ in range(n_determinizations):
        det_state = redeterminize(public_state, py_rng)
        visit_counts, root = mcts.search(
            det_state, add_noise=add_noise, rng=np_rng, leaf_batch=leaf_batch)
        for a, c in visit_counts.items():
            agg[a] += c
        total_value0 += root.value
    return dict(agg), total_value0 / n_determinizations


def run_pimc_open_loop(
    mcts: OpenLoopMCTS,
    public_state: GameState,
    *,
    add_noise: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[object, int], float, "OpenLoopNode"]:
    """Single open-loop MCTS search (no outer determinization loop needed).

    Open-loop MCTS internally averages over many deck orders, so the outer
    PIMC loop of run_pimc is redundant. This wrapper provides a compatible
    interface for callers that currently use run_pimc.

    Exact endgame search is active through OpenLoopMCTS constructor flags
    (exact_endgame_enabled=True by default). Advisor callers should keep that
    default unless deliberately benchmarking network-only leaf values.

    Returns (visit_counts, root_value_player0, root). The root node is exposed
    so callers can read per-child priors and Q values (its children are keyed
    by slot-relative joint index — use encode_action to look one up).
    """
    if rng is None:
        rng = np.random.default_rng()
    visit_counts, root = mcts.search(
        public_state, add_noise=add_noise, rng=rng)
    return visit_counts, root.value, root


if __name__ == "__main__":
    from games.kingdomino.network import KingdominoNet
    net = KingdominoNet(channels=48, blocks=4, bilinear_dim=32)
    mcts = AlphaZeroMCTS(make_serial_evaluator(net), n_simulations=50)
    state = GameState.new(seed=0)
    rng = random.Random(0)
    while state.phase != Phase.PLACE_AND_SELECT:
        state = state.step(rng.choice(state.legal_actions()))
    vc, root = mcts.search(state)
    print(f"phase={state.phase.name} legal={len(state.legal_actions())} "
          f"visited={len(vc)} total_visits={sum(vc.values())} "
          f"root_value0={root.value:+.3f}")

    # ── Open-loop MCTS smoke ──
    ol_mcts = OpenLoopMCTS(make_serial_evaluator(net), n_simulations=20)
    np_rng = np.random.default_rng(1)
    ol_vc, ol_root = ol_mcts.search(state, add_noise=True, rng=np_rng)
    # Count total nodes in the open-loop tree (root + all descendants).
    def _count_nodes(n):
        return 1 + sum(_count_nodes(c) for c in n.children.values())
    total_nodes = _count_nodes(ol_root)
    print(f"open-loop: visited={len(ol_vc)} "
          f"total_visits={sum(ol_vc.values())} "
          f"root_value0={ol_root.value:+.3f}")
    print(f"open-loop: total_nodes={total_nodes} "
          f"root.visit_count={ol_root.visit_count} (expect {20 + 1}) "
          f"fallbacks={ol_mcts._fallback_count}")
    assert ol_root.visit_count == 20 + 1, \
        f"root.visit_count={ol_root.visit_count}, expected 21"

    # ── Determinism check: same seed → identical visit_counts ──
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    vc_a, _ = ol_mcts.search(state, rng=rng_a)
    vc_b, _ = ol_mcts.search(state, rng=rng_b)
    assert vc_a == vc_b, f"Non-deterministic: {vc_a} != {vc_b}"
    print("Determinism check: PASS")
