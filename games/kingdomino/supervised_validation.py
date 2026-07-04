"""
DIAGNOSTIC — Supervised validation harness for the value head.

Used during early development to validate the encoder → network →
training pipeline before self-play was working. Not needed for routine
training. Kept for reference.

────────────────────────────────────────────────────────────────────────
Original module docstring follows:

supervised_validation.py — value-head supervised training and validation.

WHY THIS EXISTS
This harness validates the entire pipeline — encoder, network, training loop,
augmentation, and MCTS integration — on a SIMPLER objective than full self-play.
Self-play is a moving-target RL loop; debugging "why won't it converge" is
much easier when three of the suspects (encoder correctness, network learning
capability, training-loop plumbing) have already been ruled out via a fixed
regression task.

THE TASK
1. Generate games with existing bots (Greedy here; extensible to MCTS).
2. For each non-terminal position, record the encoded state from the current
   actor's perspective, labelled with compute_target_z(terminal, current_actor).
   Same-game positions get alternating-sign labels naturally — no rebalancing.
3. Train the network's win head (full forward pass, BCE on win_prob — z is
   rescaled to (0,1) as the target; all heads receive gradient).
4. Validate two ways:
     (a) Held-out MSE  — does the value head predict outcomes on unseen
         positions?  (basic ML sanity)
     (b) Gameplay      — does it beat random rollouts as the leaf evaluator
         in an otherwise-identical UCB MCTS?  (the question that matters,
         and the one your Run 3 diagnostic flagged as the bottleneck)

NOT IMPORTED: evaluation.py.  The trained value head will eventually replace
the heuristic; the harness must not be allowed to launder it back in.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from games.kingdomino.action_codec import NUM_JOINT_ACTIONS
from games.kingdomino.augmentation import NUM_D4_TRANSFORMS, augment
from games.kingdomino.bots import GreedyBot, RandomBot
from games.kingdomino.encoder import compute_target_z, encode_state
from games.kingdomino.game import GameState, Phase, determine_winner
from games.kingdomino.network import KingdominoNet


# ─── 1. Position record + game generation ────────────────────────────────
@dataclass
class Position:
    """One labelled training example."""
    my_board: np.ndarray   # (9, 13, 13) float32
    opp_board: np.ndarray  # (9, 13, 13) float32
    flat: np.ndarray       # (FLAT_SIZE,) float32
    z_target: float        # in [-1, 1], from the current actor's perspective


def play_one_game(
    bot_a, bot_b, seed: int,
) -> Tuple[List[Position], Tuple[int, int]]:
    """Play one game; record one Position per non-terminal decision.
    Returns (positions, (final_score_0, final_score_1)).
    """
    state = GameState.new(seed=seed)
    rng = random.Random(seed * 31 + 7)

    # Collect (state-snapshot, current_actor) for every non-terminal decision
    decisions: List[Tuple[GameState, int]] = []
    while state.phase != Phase.GAME_OVER:
        actor = state.current_actor
        decisions.append((state, actor))
        actions = state.legal_actions()
        bot = bot_a if actor == 0 else bot_b
        action = bot.choose_action(state, actions, rng=rng)
        state = state.step(action)

    # Now label every recorded decision with the terminal z from its actor's view
    positions: List[Position] = []
    for s, actor in decisions:
        mb, ob, flat = encode_state(s, actor)
        z = compute_target_z(state, actor)
        positions.append(Position(mb, ob, flat, float(z)))

    final_scores = (state.boards[0].score().total,
                    state.boards[1].score().total)
    return positions, final_scores


def generate_game_records(
    bot_a, bot_b, n_games: int, seed: int = 0, verbose: bool = True,
) -> List[Tuple[List[Position], Tuple[int, int]]]:
    """Generate `n_games` games, keeping each game's positions grouped.

    Grouping matters for an honest holdout split: every position in a game
    shares the same terminal outcome, so a position-level split would leak
    near-identical, identically-labelled states across the train/holdout
    boundary and flatter the holdout MSE.  Split by GAME, then flatten.
    """
    games: List[Tuple[List[Position], Tuple[int, int]]] = []
    t0 = time.time()
    score_diffs: List[int] = []
    for i in range(n_games):
        positions, scores = play_one_game(bot_a, bot_b, seed=seed + i)
        games.append((positions, scores))
        score_diffs.append(scores[0] - scores[1])
        if verbose and (i + 1) % max(1, n_games // 10) == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            n_pos = sum(len(p) for p, _ in games)
            print(f"  game {i+1:4d}/{n_games}  ({rate:5.1f} games/sec)  "
                  f"positions={n_pos}")
    if verbose:
        print(f"  score diff: mean={np.mean(score_diffs):+.2f}, "
              f"std={np.std(score_diffs):.2f} "
              f"(used for tuning sigma in compute_target_z if needed)")
    return games


def generate_games(
    bot_a, bot_b, n_games: int, seed: int = 0, verbose: bool = True,
) -> List[Position]:
    """Generate `n_games` games, return all positions flattened.

    Convenience wrapper; prefer generate_game_records when you need a
    leak-free by-game train/holdout split.
    """
    games = generate_game_records(bot_a, bot_b, n_games, seed, verbose)
    return [p for positions, _ in games for p in positions]


# ─── 2. Dataset (with optional D4 augmentation) ──────────────────────────
class ValueDataset(Dataset):
    """Holds Position records.  Optionally applies a random D4 transform per
    __getitem__.  Augmentation transforms only the spatial features (boards);
    flat features and z are invariant under D4.
    """
    def __init__(self, positions: List[Position], augment_d4: bool = True):
        self.positions = positions
        self.augment_d4 = augment_d4
        # Placeholder policy for augment() — augmented output is discarded.
        self._dummy_policy = np.zeros(NUM_JOINT_ACTIONS, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int):
        pos = self.positions[idx]
        if self.augment_d4:
            t = int(np.random.randint(0, NUM_D4_TRANSFORMS))
            # augment now returns 8 items; this harness only uses z.  The
            # four-head scalar targets are DIAGNOSTIC PLACEHOLDERS here (this is a
            # z-only validation harness, not a four-head training path), passed
            # explicitly because augment() now requires them.
            mb, ob, flat, _, z, _own, _opp, _win = augment(
                pos.my_board, pos.opp_board, pos.flat,
                self._dummy_policy, pos.z_target, t,
                own_score=0.0, opp_score=0.0, win_target=0.5,
            )
        else:
            mb, ob, flat, z = pos.my_board, pos.opp_board, pos.flat, pos.z_target
        return (
            torch.from_numpy(mb).float(),
            torch.from_numpy(ob).float(),
            torch.from_numpy(flat).float(),
            torch.tensor(z, dtype=torch.float32),
        )


# ─── 3. Training loop ────────────────────────────────────────────────────
def compute_holdout_bce(net: KingdominoNet, loader: DataLoader, device: str) -> float:
    """Mean BCE of win_prob predictions on the held-out set.

    win_prob ∈ (0,1) is the network's win-probability head; the z target in
    [-1,1] is rescaled to (0,1) to match it.  (forward_value was removed in
    Phase 1a; win_prob is the closest interpretable scalar.)
    """
    net.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for mb, ob, flat, z in loader:
            mb, ob = mb.to(device), ob.to(device)
            flat, z = flat.to(device), z.to(device)
            own, opp, win_prob, logits = net(mb, ob, flat)
            z_01 = (z + 1.0) / 2.0
            total += F.binary_cross_entropy(win_prob, z_01, reduction="sum").item()
            count += mb.shape[0]
    return total / max(1, count)


def train_value_head(
    net: KingdominoNet,
    train_ds: ValueDataset,
    holdout_ds: ValueDataset,
    *,
    n_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[KingdominoNet, Dict[str, List[float]]]:
    """Train net's win head.  Returns (net with best-holdout weights, history).

    Implementation notes:
      - Uses net.forward() and trains on win_prob (BCE with z rescaled to
        (0,1)).  All heads receive gradients (the full forward pass is used);
        for this diagnostic harness that is acceptable.
      - Best-checkpoint selection by holdout BCE.  Restored before return.
    """
    net = net.to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    holdout_loader = DataLoader(holdout_ds, batch_size=batch_size, shuffle=False,
                                num_workers=0)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_bce": [], "holdout_bce": []}
    best_holdout = math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, n_epochs + 1):
        net.train()
        train_sum, train_count = 0.0, 0
        for mb, ob, flat, z in train_loader:
            mb, ob = mb.to(device), ob.to(device)
            flat, z = flat.to(device), z.to(device)
            own, opp, win_prob, logits = net(mb, ob, flat)
            z_01 = (z + 1.0) / 2.0           # [-1,1] → (0,1) to match win_prob
            loss = F.binary_cross_entropy(win_prob, z_01)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_sum += loss.item() * mb.shape[0]
            train_count += mb.shape[0]
        train_bce = train_sum / max(1, train_count)
        holdout_bce = compute_holdout_bce(net, holdout_loader, device)

        history["train_bce"].append(train_bce)
        history["holdout_bce"].append(holdout_bce)

        if holdout_bce < best_holdout:
            best_holdout = holdout_bce
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}

        if verbose:
            print(f"  epoch {epoch:3d}/{n_epochs}  "
                  f"train_bce={train_bce:.4f}  holdout_bce={holdout_bce:.4f}"
                  f"{'  *best' if holdout_bce == best_holdout else ''}")

    if best_state is not None:
        net.load_state_dict(best_state)
    return net, history


# ─── 4. Network-leaf UCB MCTS for gameplay validation ────────────────────
class _Node:
    __slots__ = ("state", "N", "W", "is_expanded", "children")

    def __init__(self, state: Optional[GameState]):
        self.state = state                # set lazily for children
        self.N = 0                        # visit count
        self.W = 0.0                      # value sum, PLAYER-0 frame
        self.is_expanded = False
        self.children: Dict[object, "_Node"] = {}

    def expand(self, legal_actions) -> None:
        for a in legal_actions:
            self.children[a] = _Node(state=None)
        self.is_expanded = True


# Leaf evaluator: callable(state, player_perspective, rng) -> float in [-1, 1].
# The rng is threaded through so stochastic evaluators (random rollouts) are
# reproducible; deterministic evaluators (the network) ignore it.
LeafEvaluator = Callable[[GameState, int, random.Random], float]


class UCBMCTS:
    """Minimal UCB MCTS with a configurable leaf evaluator.

    Used for the validation comparison: the SAME MCTS is instantiated with
    two different leaf evaluators (random rollout vs trained network value)
    so the only experimental variable is the leaf signal — exactly what the
    Run 3 diagnostic identified as the bottleneck.

    Values stored in player-0 frame to match mcts_az.py's convention;
    perspective is converted at selection time.
    """
    def __init__(
        self, leaf_evaluator: LeafEvaluator, *,
        c_ucb: float = 1.4, n_simulations: int = 25,
    ):
        self.leaf_eval = leaf_evaluator
        self.c_ucb = c_ucb
        self.n_simulations = n_simulations

    def choose_action(self, state, actions, rng=None):
        """Bot interface — matches GreedyBot / RandomBot."""
        rng = rng or random
        if state.phase == Phase.GAME_OVER:
            raise ValueError("Cannot choose action in terminal state.")
        root = _Node(state)
        root.expand(actions)
        for _ in range(self.n_simulations):
            self._simulate(root, rng)
        # Robust-child selection, but tie-broken by actor-frame mean value.
        # With few simulations relative to the legal-action count, many
        # children share N=1 (or 0); a pure max-by-visits collapses into
        # dict-insertion order and ignores the value signal we're validating.
        actor = state.current_actor

        def root_score(a):
            c = root.children[a]
            if c.N == 0:
                return (-1, -math.inf)
            q0 = c.W / c.N
            q = q0 if actor == 0 else -q0
            return (c.N, q)

        return max(root.children, key=root_score)

    def _simulate(self, root: _Node, rng) -> None:
        path = [root]
        node = root
        while node.is_expanded and node.state.phase != Phase.GAME_OVER:
            action = self._select_action(node, rng)
            child = node.children[action]
            if child.state is None:
                child.state = node.state.step(action)
            path.append(child)
            node = child

        if node.state.phase == Phase.GAME_OVER:
            v0 = compute_target_z(node.state, player=0)
        else:
            node.expand(node.state.legal_actions())
            v0 = self.leaf_eval(node.state, 0, rng)  # player-0 perspective

        for n in path:
            n.N += 1
            n.W += v0

    def _select_action(self, node: _Node, rng):
        actor = node.state.current_actor
        # Try unvisited children first (standard UCB initialisation).
        unvisited = [a for a, c in node.children.items() if c.N == 0]
        if unvisited:
            return rng.choice(unvisited)
        # All visited — UCB1 from the actor's perspective.
        log_n = math.log(max(1, node.N))
        best_score, best_action = -math.inf, None
        for a, child in node.children.items():
            q0 = child.W / child.N
            q = q0 if actor == 0 else -q0
            u = self.c_ucb * math.sqrt(log_n / child.N)
            score = q + u
            if score > best_score:
                best_score, best_action = score, a
        return best_action


# ─── 5. Leaf evaluators ──────────────────────────────────────────────────
def random_rollout_evaluator(state: GameState, player: int,
                             rng: Optional[random.Random] = None) -> float:
    """Play a uniform-random game to terminal; return tanh(margin/30) from
    `player`'s perspective.  This is the Run 3 baseline leaf signal."""
    rng = rng or random
    bot = RandomBot()
    s = state
    while s.phase != Phase.GAME_OVER:
        s = s.step(bot.choose_action(s, s.legal_actions(), rng=rng))
    return compute_target_z(s, player)


def make_network_evaluator(net: KingdominoNet, device: str = "cpu") -> LeafEvaluator:
    """Wrap a trained network into a leaf-evaluator callable.

    The network's win head outputs P(win) from the encoded perspective
    (state.current_actor), in (0,1).  We remap it to [-1,1] (the LeafEvaluator
    contract) and convert to the caller's `player` perspective by negating when
    they differ.  (forward_value was removed in Phase 1a; win_prob is used.)
    """
    net = net.to(device).eval()

    def evaluator(state: GameState, player: int,
                  rng: Optional[random.Random] = None) -> float:
        mb, ob, flat = encode_state(state, state.current_actor)
        mb_t = torch.from_numpy(mb).unsqueeze(0).to(device)
        ob_t = torch.from_numpy(ob).unsqueeze(0).to(device)
        flat_t = torch.from_numpy(flat).unsqueeze(0).to(device)
        with torch.no_grad():
            own, opp, win_prob, logits = net(mb_t, ob_t, flat_t)
        v = 2.0 * float(win_prob.squeeze().item()) - 1.0   # (0,1) → [-1,1]
        return v if player == state.current_actor else -v

    return evaluator


# ─── 6. Head-to-head evaluation ──────────────────────────────────────────
def head_to_head(
    bot_a, bot_b, n_seeds: int, *, seed: int = 0, verbose: bool = True,
) -> Dict[str, float]:
    """Play paired games of bot_a vs bot_b.

    Each deck seed is played TWICE — once with bot_a as player 0, once as
    player 1 — so deck luck and first-player advantage are both controlled
    rather than merely cancelled in expectation.  Total games = 2 * n_seeds.
    Returns win counts for a/b/draws.
    """
    a_wins = b_wins = draws = 0
    games_played = 0
    for i in range(n_seeds):
        for a_is_p0 in (True, False):
            p0_bot, p1_bot = (bot_a, bot_b) if a_is_p0 else (bot_b, bot_a)
            state = GameState.new(seed=seed + i)  # SAME deck for both assignments
            rng = random.Random(seed * 1000 + i * 2 + int(a_is_p0))
            while state.phase != Phase.GAME_OVER:
                actor = state.current_actor
                bot = p0_bot if actor == 0 else p1_bot
                action = bot.choose_action(state, state.legal_actions(), rng=rng)
                state = state.step(action)

            # Route through the authoritative tiebreaker cascade rather than a
            # raw score comparison (score ties are resolved by determine_winner).
            winner = determine_winner(state)
            if winner is None:
                draws += 1
            elif winner == (0 if a_is_p0 else 1):
                a_wins += 1
            else:
                b_wins += 1
            games_played += 1

        if verbose and (i + 1) % max(1, n_seeds // 10) == 0:
            print(f"  seed {i+1:3d}/{n_seeds}  ({games_played} games)  "
                  f"a={a_wins}  draws={draws}  b={b_wins}")

    return {
        "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
        "n_games": games_played,
        "a_win_rate": a_wins / max(1, games_played),
    }


# ─── 7. Full pipeline ────────────────────────────────────────────────────
def run_pipeline(
    *,
    n_training_games: int = 500,
    n_eval_games: int = 20,
    n_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    n_simulations: int = 25,
    holdout_frac: float = 0.1,
    device: str = "cpu",
    seed: int = 0,
    channels: int = 96,
    blocks: int = 8,
    bilinear_dim: int = 64,
    verbose: bool = True,
) -> Dict:
    """End-to-end: generate games → train value head → MSE + gameplay validation."""
    # Reproducibility: seed every RNG the pipeline touches (Python random for
    # game/shuffle, NumPy for augmentation sampling, Torch for init + training).
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def header(s):
        if verbose: print(f"\n{'=' * 60}\n{s}\n{'=' * 60}")

    header("Step 1: Generating training games (Greedy vs Greedy)")
    bot = GreedyBot()
    games = generate_game_records(bot, bot, n_training_games, seed=seed, verbose=verbose)
    n_positions = sum(len(p) for p, _ in games)
    if verbose:
        print(f"\nTotal positions: {n_positions}  "
              f"(≈ {n_positions/max(1,n_training_games):.1f} per game)")

    # Split by GAME, not by position, so no game straddles the train/holdout
    # boundary (every position in a game shares the same terminal label).
    rng = random.Random(seed)
    rng.shuffle(games)
    n_holdout_games = max(1, int(len(games) * holdout_frac))
    holdout_games = games[:n_holdout_games]
    train_games = games[n_holdout_games:]
    holdout_positions = [p for positions, _ in holdout_games for p in positions]
    train_positions = [p for positions, _ in train_games for p in positions]
    if verbose:
        print(f"Train: {len(train_positions)} positions / {len(train_games)} games   "
              f"Holdout: {len(holdout_positions)} positions / {len(holdout_games)} games")

    train_ds = ValueDataset(train_positions, augment_d4=True)
    holdout_ds = ValueDataset(holdout_positions, augment_d4=False)

    header("Step 2: Training value head")
    net = KingdominoNet(channels=channels, blocks=blocks, bilinear_dim=bilinear_dim)
    if verbose:
        n_params = sum(p.numel() for p in net.parameters())
        print(f"Network: {n_params:,} parameters")
    net, history = train_value_head(
        net, train_ds, holdout_ds,
        n_epochs=n_epochs, batch_size=batch_size, lr=lr,
        device=device, verbose=verbose,
    )

    header("Step 3: Gameplay validation — network value vs random rollouts")
    network_mcts = UCBMCTS(make_network_evaluator(net, device=device),
                           n_simulations=n_simulations)
    rollout_mcts = UCBMCTS(random_rollout_evaluator,
                           n_simulations=n_simulations)
    # n_eval_games is interpreted as DECK SEEDS; each is played twice (paired
    # sides), so the actual game count is 2 * n_eval_games.
    stats = head_to_head(network_mcts, rollout_mcts, n_eval_games,
                         seed=seed + 10_000, verbose=verbose)

    if verbose:
        print(f"\nNetwork-value MCTS:  {stats['a_wins']:3d} wins")
        print(f"Random-rollout MCTS: {stats['b_wins']:3d} wins")
        print(f"Draws:               {stats['draws']:3d}")
        print(f"(over {stats['n_games']} games = {n_eval_games} paired deck seeds)")
        print(f"\nNetwork win rate: {stats['a_win_rate']:.1%}")
        verdict = ("✓ PASS — value head provides a stronger leaf signal than rollouts"
                   if stats['a_win_rate'] > 0.5 else
                   "✗ FAIL — value head did not beat the rollout baseline")
        print(f"\n{verdict}")

    return {
        "net": net,
        "history": history,
        "eval_stats": stats,
        "win_rate": stats["a_win_rate"],
    }


# ─── 8. CLI entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n_games", type=int, default=200,
                   help="number of training games to generate")
    p.add_argument("--n_eval_games", type=int, default=10,
                   help="deck seeds for validation; each played twice "
                        "(paired sides), so total games = 2x this")
    p.add_argument("--n_epochs", type=int, default=8)
    p.add_argument("--n_sims", type=int, default=25,
                   help="MCTS simulations per move in the gameplay test")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--channels", type=int, default=96)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--save_path", default=None,
                   help="where to save trained value head (omit to skip)")
    args = p.parse_args()

    result = run_pipeline(
        n_training_games=args.n_games,
        n_eval_games=args.n_eval_games,
        n_epochs=args.n_epochs,
        n_simulations=args.n_sims,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        channels=args.channels,
        blocks=args.blocks,
    )

    if args.save_path:
        torch.save({
            "model_state": result["net"].state_dict(),
            "kind": "value_only_supervised_validation",
            "policy_head_trained": False,
            "history": result["history"],
            "eval_stats": result["eval_stats"],
        }, args.save_path)
        print(f"\nSaved trained network to {args.save_path}")
        print("  (metadata marks this value-only; the policy head is at "
              "initialisation — load model_state as a warm-start trunk/value "
              "path, not as a full AlphaZero model.)")
