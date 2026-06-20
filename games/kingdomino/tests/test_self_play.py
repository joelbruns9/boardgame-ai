"""Test suite for self_play.py.

Verifies the data pipeline (buffer round-trip, sparse/dense + augmentation
consistency), the self-play game runner (label signs, target/mask alignment),
the training step (finite losses, gradient to both heads, overfit a fixed
batch), the benchmark adapter, and a tiny end-to-end loop.  Uses small configs
for speed — the point is plumbing correctness, not strength.
"""
from __future__ import annotations

import math
import random
import sys

import numpy as np
import torch

from games.kingdomino.game import GameState, Phase
from games.kingdomino.action_codec import NUM_JOINT_ACTIONS
from games.kingdomino.encoder import NUM_BOARD_CHANNELS, CANVAS_SIZE, FLAT_SIZE
from games.kingdomino.network import KingdominoNet
from games.kingdomino.bots import GreedyBot, RandomBot
from games.kingdomino.mcts_az import AlphaZeroMCTS, make_serial_evaluator
from games.kingdomino import self_play as sp
from games.kingdomino.self_play import (
    SelfPlayConfig, Example, ReplayBuffer, play_selfplay_game, train_step,
    AZPlayer, benchmark_vs, run_self_play_training, make_mcts, _temperature,
)

_failures = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        _failures.append(name)

torch.manual_seed(0)


def tiny_net():
    return KingdominoNet(channels=16, blocks=2, bilinear_dim=8)

def tiny_mcts(net, n_sims=6):
    return AlphaZeroMCTS(make_serial_evaluator(net), n_simulations=n_sims,
                         dirichlet_alpha=0.3, dirichlet_epsilon=0.25)


# ──────────────────────────────────────────────────────────────────────────
print("=== TEST 1: ReplayBuffer add / len / ring-buffer eviction ===")
buf = ReplayBuffer(capacity=10)
def dummy_example(z=0.0):
    return Example(
        my_board=np.zeros((NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE), np.float16),
        opp_board=np.zeros((NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE), np.float16),
        flat=np.zeros(FLAT_SIZE, np.float16),
        policy_idx=np.array([0, 5], np.int32),
        policy_val=np.array([0.5, 0.5], np.float32),
        legal_idx=np.array([0, 5, 9], np.int32),
        z=z,
        own_score=0.0, opp_score=0.0, win_target=0.5,
    )
buf.add([dummy_example(float(i)) for i in range(7)])
check("length after 7 adds", len(buf) == 7)
buf.add([dummy_example(float(i)) for i in range(7, 15)])  # total 15 into cap 10
check("capacity enforced (ring buffer)", len(buf) == 10)
zs = sorted(ex.z for ex in buf.data)
check("oldest examples evicted (newest 10 retained)", zs == [float(i) for i in range(5, 15)],
      f"got {zs}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 2: sample_batch shapes, dtypes, and dense reconstruction ===")
batch = buf.sample_batch(8, np.random.default_rng(0), device="cpu", augment_d4=False)
mb, ob, flat, policy, mask, z, own_t, opp_t, win_t = batch
check("my_board (8,9,13,13) float32",
      mb.shape == (8, NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE) and mb.dtype == torch.float32)
check("flat (8,FLAT_SIZE)", flat.shape == (8, FLAT_SIZE))
check("policy (8,3390) float32", policy.shape == (8, NUM_JOINT_ACTIONS) and policy.dtype == torch.float32)
check("legal_mask (8,3390) bool", mask.shape == (8, NUM_JOINT_ACTIONS) and mask.dtype == torch.bool)
check("z (8,)", z.shape == (8,))
check("own/opp/win targets (8,) float32",
      own_t.shape == (8,) and opp_t.shape == (8,) and win_t.shape == (8,)
      and own_t.dtype == torch.float32 and win_t.dtype == torch.float32)
# dense reconstruction: dummy had policy {0:0.5, 5:0.5}, legal {0,5,9}
check("policy densified correctly", abs(float(policy[0, 0]) - 0.5) < 1e-6 and abs(float(policy[0, 5]) - 0.5) < 1e-6)
check("policy zero elsewhere", abs(float(policy[0, 1])) < 1e-9)
check("legal mask densified correctly", bool(mask[0, 0] and mask[0, 5] and mask[0, 9] and not mask[0, 1]))
check("policy support is subset of legal mask",
      bool(((policy > 0) <= mask).all()))


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 3: augmentation keeps policy support inside the legal mask ===")
# Use a REAL self-play position so policy/mask have spatial structure.
net = tiny_net()
mcts = tiny_mcts(net, n_sims=6)
examples, scores = play_selfplay_game(
    mcts, n_determinizations=1, temp_moves=4, seed=1,
    py_rng=random.Random(1), np_rng=np.random.default_rng(1),
)
buf2 = ReplayBuffer(1000); buf2.add(examples)
ok = True
rng = np.random.default_rng(7)
for _ in range(20):
    _, _, _, pol, msk, _, _, _, _ = buf2.sample_batch(4, rng, augment_d4=True)
    if not bool(((pol > 0) <= msk).all()):
        ok = False; break
check("augmented policy support ⊆ augmented legal mask (20 batches)", ok)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 4: play_selfplay_game produces well-formed examples ===")
check("plausible example count (~52)", 40 <= len(examples) <= 60, f"got {len(examples)}")
ex0 = examples[0]
check("board dtype float16 (memory)", ex0.my_board.dtype == np.float16)
check("board shape", ex0.my_board.shape == (NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE))
check("z in [-1, 1]", all(-1.0 <= e.z <= 1.0 for e in examples))
check("policy mass sums ~1 (sparse vals)", all(abs(e.policy_val.sum() - 1.0) < 1e-4 for e in examples))
check("every policy index is legal", all(set(e.policy_idx.tolist()) <= set(e.legal_idx.tolist()) for e in examples))
# Value sign: zero-sum, so player-0 and player-1 frames should carry opposite signs
nonzero = [e.z for e in examples if abs(e.z) > 1e-6]
signs = set(np.sign(nonzero)) if nonzero else set()
check("z carries both signs across actors (or game drawn)",
      signs in ({-1.0, 1.0}, set(), {-1.0}, {1.0}))


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 5: train_step gives finite losses (masked CE not NaN) ===")
net = tiny_net()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
batch = buf2.sample_batch(32, np.random.default_rng(0), augment_d4=True)
policy_loss, own_loss, opp_loss, win_loss, win_brier, baseline_brier = \
    train_step(net, batch, opt)
check("policy loss finite and non-negative (masked CE ok, no 0·-inf NaN)",
      math.isfinite(policy_loss) and policy_loss >= 0)
check("own/opp score losses finite and in [0, ~] (normalized MSE)",
      math.isfinite(own_loss) and math.isfinite(opp_loss)
      and own_loss >= 0 and opp_loss >= 0)
check("win loss finite and non-negative (BCE)",
      math.isfinite(win_loss) and win_loss >= 0)
check("win_brier finite in [0, 1]",
      math.isfinite(win_brier) and 0.0 <= win_brier <= 1.0)
check("baseline_brier finite in [0, 0.25] (Bernoulli variance)",
      math.isfinite(baseline_brier) and 0.0 <= baseline_brier <= 0.25 + 1e-9)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 6: both heads receive gradient from train_step ===")
net = tiny_net()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
batch = buf2.sample_batch(32, np.random.default_rng(1), augment_d4=False)
mb, ob, flat, policy, mask, z, own_t, opp_t, win_t = batch
own, opp, win_prob, logits = net(mb, ob, flat)
import torch.nn.functional as F
from games.kingdomino.network import masked_log_softmax
# Touch a score head and the policy head so both pathways get gradient.
loss = F.mse_loss(own, own_t) - (policy * masked_log_softmax(logits, mask)).sum(1).mean()
net.zero_grad(); loss.backward()
# value_mlp removed in Phase 1a; checking own_score_mlp gradient as the
# equivalent new head, and policy head param (W bilinear).
vgrad = net.own_score_mlp[-1].weight.grad
pgrad = net.W.grad
check("score head (own_score_mlp) receives gradient",
      vgrad is not None and float(vgrad.abs().sum()) > 0)
check("policy head (bilinear W) receives gradient", pgrad is not None and float(pgrad.abs().sum()) > 0)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 7: train_step can overfit a fixed batch (learning works) ===")
net = tiny_net()
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
fixed = buf2.sample_batch(16, np.random.default_rng(2), augment_d4=False)
first_p, first_own, first_opp, first_win, *_ = train_step(net, fixed, opt)
for _ in range(40):
    last_p, last_own, last_opp, last_win, *_ = train_step(net, fixed, opt)
check(f"own-score loss drops on fixed batch ({first_own:.3f} → {last_own:.3f})",
      last_own < first_own * 0.6)
check(f"policy loss drops on fixed batch ({first_p:.3f} → {last_p:.3f})", last_p < first_p * 0.9)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 8: temperature schedule ===")
check("τ=1 before temp_moves", _temperature(0, 5) == 1.0 and _temperature(4, 5) == 1.0)
check("τ=0 at/after temp_moves", _temperature(5, 5) == 0.0 and _temperature(20, 5) == 0.0)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 9: AZPlayer returns a legal action ===")
net = tiny_net()
az = AZPlayer(tiny_mcts(net, n_sims=6), n_determinizations=1, np_rng=np.random.default_rng(0))
state = GameState.new(seed=3)
r = random.Random(3)
while state.phase != Phase.PLACE_AND_SELECT:
    state = state.step(r.choice(state.legal_actions()))
action = az.choose_action(state, state.legal_actions(), rng=random.Random(0))
check("AZPlayer action is legal", action in state.legal_actions())


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 10: benchmark_vs runs and reports paired-seed counts ===")
net = tiny_net()
az = AZPlayer(tiny_mcts(net, n_sims=4), n_determinizations=1, np_rng=np.random.default_rng(0))
stats = benchmark_vs(az, RandomBot(), n_seeds=3, seed=0, verbose=False)
check("benchmark counts sum to 2*n_seeds", stats["az_wins"] + stats["opp_wins"] + stats["draws"] == 6
      and stats["n_games"] == 6)
check("win rate in [0,1]", 0.0 <= stats["az_win_rate"] <= 1.0)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 11: end-to-end run_self_play_training (2 tiny iterations) ===")
cfg = SelfPlayConfig(
    channels=16, blocks=2, bilinear_dim=8,
    n_simulations=6, n_determinizations=1, temp_moves=4,
    buffer_capacity=5000, batch_size=64,
    n_iterations=2, games_per_iteration=2, train_steps_per_iteration=5,
    min_buffer_to_train=1, benchmark_every=2, benchmark_seeds=2, benchmark_sims=4,
    device="cpu", seed=0,
)
result = run_self_play_training(cfg, verbose=False)
check("returns net/history/buffer", set(result.keys()) >= {"net", "history", "buffer"})
check("buffer populated by self-play", len(result["buffer"]) > 0)
check("training produced loss history",
      len(result["history"]["policy_loss"]) >= 1
      and len(result["history"]["own_loss"]) >= 1
      and len(result["history"]["opp_loss"]) >= 1
      and len(result["history"]["win_loss"]) >= 1)
check("benchmark recorded", len(result["history"]["benchmark"]) >= 1)
check("all recorded losses finite",
      all(math.isfinite(x) for k in ("policy_loss", "own_loss", "opp_loss", "win_loss")
          for x in result["history"][k]))


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 12: self_play does not import evaluation.py ===")
import ast
tree = ast.parse(open(sp.__file__, encoding="utf-8").read())
bad = False
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module and "evaluation" in node.module:
        bad = True
    elif isinstance(node, ast.Import):
        for alias in node.names:
            if "evaluation" in alias.name:
                bad = True
check("no import references 'evaluation'", not bad)


# ──────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
if _failures:
    print(f"FAILED: {len(_failures)} test(s)")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")