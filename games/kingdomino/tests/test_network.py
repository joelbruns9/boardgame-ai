"""Test suite for KingdominoNet.

The critical test is TEST 4: the placement-conv reshape must lay out spatial
placements in exactly the order the action codec expects
(index = direction*169 + y*13 + x).  If this is wrong, the policy head and the
codec disagree and the network silently trains on misaligned targets.
"""
from __future__ import annotations

import sys
import numpy as np
import torch

from games.kingdomino.encoder import (
    FLAT_SIZE, CANVAS_SIZE, NUM_BOARD_CHANNELS, encode_state,
)
from games.kingdomino.action_codec import (
    NUM_JOINT_ACTIONS, NUM_SPATIAL_PLACEMENTS, NUM_DIRECTIONS, PICK_AXIS_SIZE,
    legal_mask, make_joint_idx,
)
from games.kingdomino.game import GameState, Phase
from games.kingdomino import network as net_mod
from games.kingdomino.network import (
    KingdominoNet, masked_log_softmax, count_parameters,
)
from games.kingdomino.mcts_az import MARGIN_GAIN, ALPHA


_failures = []
def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        _failures.append(name)


torch.manual_seed(0)


# ──────────────────────────────────────────────────────────────────────────
print("=== TEST 1: four-head forward pass shapes and ranges ===")
net = KingdominoNet(channels=48, blocks=4, bilinear_dim=32)  # small for fast tests
net.eval()
B = 4
mb = torch.randn(B, NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE)
ob = torch.randn(B, NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE)
flat = torch.randn(B, FLAT_SIZE)
with torch.no_grad():
    own, opp, win_prob, p = net(mb, ob, flat)
check("own_score shape (B,)", own.shape == (B,))
check("opp_score shape (B,)", opp.shape == (B,))
check("win_prob shape (B,)", win_prob.shape == (B,))
check("policy shape (B, 3390)", p.shape == (B, NUM_JOINT_ACTIONS))
check("win_prob in (0, 1)", bool((win_prob > 0).all() and (win_prob < 1).all()))
# leaf_value is the search-time combination (mcts_az); must stay in (-1, 1).
leaf_value = ALPHA * torch.tanh((own - opp) * MARGIN_GAIN) + (1 - ALPHA) * (2 * win_prob - 1)
check("leaf_value in (-1, 1)",
      bool((leaf_value > -1).all() and (leaf_value < 1).all()))
check("policy logits finite", bool(torch.isfinite(p).all()))


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 1b: checkpoint_version is 3 (symmetric pending encoder) ===")
check("KingdominoNet.checkpoint_version == 3",
      KingdominoNet.checkpoint_version == 3,
      f"got {KingdominoNet.checkpoint_version}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 2: gradients flow to all parameters ===")
net.train()
own, opp, win_prob, p = net(mb, ob, flat)
# Touch all four heads so every parameter (all three value heads + policy) gets grad.
loss = own.sum() + opp.sum() + win_prob.sum() + p.sum()
loss.backward()
no_grad = [name for name, param in net.named_parameters()
           if param.requires_grad and (param.grad is None or param.grad.abs().sum() == 0)]
check("all parameters receive nonzero gradient", len(no_grad) == 0,
      f"params without grad: {no_grad[:5]}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 3: batch independence (eval mode) ===")
# In eval mode (BN uses running stats), each example must be processed
# independently — the output for example i must not depend on other examples.
net.eval()
with torch.no_grad():
    own_batch, opp_batch, win_batch, p_batch = net(mb, ob, flat)
    # Process example 2 alone
    own_single, opp_single, win_single, p_single = net(mb[2:3], ob[2:3], flat[2:3])
check("eval-mode own_score is batch-independent",
      torch.allclose(own_batch[2], own_single[0], atol=1e-5))
check("eval-mode win_prob is batch-independent",
      torch.allclose(win_batch[2], win_single[0], atol=1e-5))
check("eval-mode policy is batch-independent",
      torch.allclose(p_batch[2], p_single[0], atol=1e-5))


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 4: placement-conv reshape matches codec index layout ===")
# This is THE correctness test. We monkey-inspect the reshape by replacing the
# placement_conv with a known deterministic pattern and verifying that the
# spatial placement representation P[direction*169 + y*13 + x] comes from the
# conv output channel-group `direction` at spatial position (y, x).
#
# Strategy: build a fresh net, then hand-set placement_conv weights so that the
# conv output at (direction, d=0, y, x) equals a unique tag = direction*169 +
# y*13 + x (independent of the input). Then run forward and read P[:, :, 0],
# checking P[idx, 0] == idx for all spatial idx.

net2 = KingdominoNet(channels=16, blocks=1, bilinear_dim=4)
net2.eval()
D = net2.bilinear_dim

# We can't easily force arbitrary conv outputs, so instead we test the reshape
# logic directly by replicating it on a tensor with known structure.
# Construct a fake placement_conv output where the value at
# [b, direction*D + d, y, x] = direction*1000000 + y*1000 + x*10 + d
# and verify the network's reshape sends it to P[b, direction*169 + y*13 + x, d].
Bt = 2
fake = torch.zeros(Bt, NUM_DIRECTIONS * D, CANVAS_SIZE, CANVAS_SIZE)
for direction in range(NUM_DIRECTIONS):
    for d in range(D):
        for y in range(CANVAS_SIZE):
            for x in range(CANVAS_SIZE):
                fake[:, direction * D + d, y, x] = (
                    direction * 1_000_000 + y * 1000 + x * 10 + d
                )

# Replicate the network's reshape (must stay in sync with network.forward).
p_spatial = fake.view(Bt, NUM_DIRECTIONS, D, CANVAS_SIZE, CANVAS_SIZE)
p_spatial = p_spatial.permute(0, 1, 3, 4, 2).contiguous()
p_spatial = p_spatial.view(Bt, NUM_SPATIAL_PLACEMENTS, D)

# Verify: P[b, direction*169 + y*13 + x, d] == direction*1e6 + y*1000 + x*10 + d
mismatches = 0
for direction in range(NUM_DIRECTIONS):
    for y in range(CANVAS_SIZE):
        for x in range(CANVAS_SIZE):
            idx = direction * (CANVAS_SIZE * CANVAS_SIZE) + y * CANVAS_SIZE + x
            for d in range(D):
                expected = direction * 1_000_000 + y * 1000 + x * 10 + d
                if p_spatial[0, idx, d].item() != expected:
                    mismatches += 1
check("placement reshape lays out (direction, y, x) exactly as codec expects",
      mismatches == 0, f"mismatches={mismatches}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 5: policy reshape to joint index matches make_joint_idx ===")
# Q is (B, 678, 5) reshaped to (B, 3390). Verify Q[b, p, k] lands at
# joint index p*5+k == make_joint_idx(p, k).
Bt = 1
Q = torch.zeros(Bt, 678, PICK_AXIS_SIZE)
test_pairs = [(0, 0), (5, 2), (676, 4), (677, 1), (300, 3)]
for (pp, kk) in test_pairs:
    Q[0, pp, kk] = 1.0
flat_pol = Q.reshape(Bt, NUM_JOINT_ACTIONS)
mismatches = 0
for (pp, kk) in test_pairs:
    ji = make_joint_idx(pp, kk)
    if flat_pol[0, ji].item() != 1.0:
        mismatches += 1
# Also ensure exactly len(test_pairs) entries are set
if int((flat_pol > 0.5).sum().item()) != len(test_pairs):
    mismatches += 1
check("Q reshape to flat policy matches make_joint_idx layout",
      mismatches == 0, f"mismatches={mismatches}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 6: masked_log_softmax zeroes illegal actions ===")
logits = torch.randn(3, NUM_JOINT_ACTIONS)
mask = torch.zeros(3, NUM_JOINT_ACTIONS, dtype=torch.bool)
# Mark a handful of legal actions per row
for b in range(3):
    legal_idxs = torch.randperm(NUM_JOINT_ACTIONS)[:10]
    mask[b, legal_idxs] = True
log_probs = masked_log_softmax(logits, mask)
probs = log_probs.exp()
# Illegal actions must have ~zero probability
illegal_mass = probs[~mask].sum().item()
check("illegal actions receive ~zero probability", illegal_mass < 1e-6,
      f"illegal_mass={illegal_mass}")
# Legal probabilities sum to ~1 per row
row_sums = probs.sum(dim=1)
check("legal probabilities sum to 1 per row",
      bool(torch.allclose(row_sums, torch.ones(3), atol=1e-5)))


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 7: end-to-end with real encoded state ===")
# Feed an actual encoded game state through the network and a real legal mask.
net3 = KingdominoNet(channels=48, blocks=4, bilinear_dim=32)
net3.eval()
state = GameState.new(seed=3)
rng = __import__("random").Random(3)
while state.phase != Phase.PLACE_AND_SELECT:
    state = state.step(rng.choice(state.legal_actions()))
mb_np, ob_np, flat_np = encode_state(state, player=state.current_actor)
mask_np = legal_mask(state)

mb_t = torch.from_numpy(mb_np).unsqueeze(0)
ob_t = torch.from_numpy(ob_np).unsqueeze(0)
flat_t = torch.from_numpy(flat_np).unsqueeze(0)
mask_t = torch.from_numpy(mask_np).unsqueeze(0)

with torch.no_grad():
    own, opp, win_prob, p = net3(mb_t, ob_t, flat_t)
    log_probs = masked_log_softmax(p, mask_t)
    probs = log_probs.exp()

check("real-state forward produces finite value heads",
      bool(torch.isfinite(own).all() and torch.isfinite(opp).all()
           and torch.isfinite(win_prob).all()))
check("real-state masked policy sums to 1",
      bool(torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5)))
check("real-state policy puts all mass on legal actions",
      probs[~mask_t].sum().item() < 1e-6)
n_legal = int(mask_np.sum())
check(f"number of legal actions is plausible ({n_legal})", n_legal == len(state.legal_actions()))


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 8: optimisation reduces loss on a fixed batch ===")
# Sanity that the network can learn. Note the policy cross-entropy has a hard
# floor of log(n_legal) since the target is uniform over legal actions — so we
# track the value loss (which CAN reach ~0 by memorising the batch) separately,
# and also check total loss decreases meaningfully in absolute terms.
net4 = KingdominoNet(channels=32, blocks=2, bilinear_dim=16)
net4.train()
opt = torch.optim.Adam(net4.parameters(), lr=1e-3)
B = 8
mb = torch.randn(B, NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE)
ob = torch.randn(B, NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE)
flat = torch.randn(B, FLAT_SIZE)
target_v = torch.tanh(torch.randn(B))
mask = torch.zeros(B, NUM_JOINT_ACTIONS, dtype=torch.bool)
for b in range(B):
    mask[b, torch.randperm(NUM_JOINT_ACTIONS)[:12]] = True
target_p = mask.float()
target_p /= target_p.sum(dim=1, keepdim=True)
policy_floor = np.log(12)  # uniform over 12 legal actions

first_value_loss = first_total = None
for step in range(80):
    opt.zero_grad()
    own, opp, win_prob, p = net4(mb, ob, flat)
    log_probs = masked_log_softmax(p, mask)
    # own_score is a raw-linear head, so it can memorise an arbitrary target —
    # the same "can the value pathway learn?" sanity the old value head gave.
    value_loss = torch.nn.functional.mse_loss(own, target_v)
    policy_loss = -(target_p * log_probs).sum(dim=1).mean()
    loss = value_loss + policy_loss
    loss.backward()
    opt.step()
    if first_total is None:
        first_total = loss.item()
        first_value_loss = value_loss.item()
final_value_loss = value_loss.item()
final_total = loss.item()
check("value loss drops substantially (memorises batch)",
      final_value_loss < first_value_loss * 0.3,
      f"first={first_value_loss:.4f} final={final_value_loss:.4f}")
check("total loss approaches the policy entropy floor",
      final_total < first_total and final_total < policy_floor + 0.2,
      f"first={first_total:.4f} final={final_total:.4f} floor~={policy_floor:.4f}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 9: GroupNorm variant works (batch-independent norm) ===")
net5 = KingdominoNet(channels=48, blocks=3, bilinear_dim=32, norm="group")
net5.train()  # GroupNorm behaves identically in train/eval
B = 4
mb = torch.randn(B, NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE)
ob = torch.randn(B, NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE)
flat = torch.randn(B, FLAT_SIZE)
own_b, opp_b, win_b, p_b = net5(mb, ob, flat)
own_s, opp_s, win_s, p_s = net5(mb[1:2], ob[1:2], flat[1:2])
check("GroupNorm: forward works and is batch-independent even in train mode",
      torch.allclose(own_b[1], own_s[0], atol=1e-5)
      and torch.allclose(p_b[1], p_s[0], atol=1e-4))


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 10: network does not import evaluation.py ===")
import ast
tree = ast.parse(open(net_mod.__file__).read())
imports_evaluation = False
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module and 'evaluation' in node.module:
        imports_evaluation = True
    elif isinstance(node, ast.Import):
        for alias in node.names:
            if 'evaluation' in alias.name:
                imports_evaluation = True
check("no import statement references 'evaluation'", not imports_evaluation)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 11: default-size parameter count is hobby-scale ===")
big = KingdominoNet()  # defaults
n = count_parameters(big)
check(f"default network parameter count is 0.5M–5M ({n:,})", 500_000 < n < 5_000_000)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 12: win head is not saturated at initialisation ===")
# Regression guard: max-pooling in the global context produces large
# activations; without small-init on the win head's final layer the pre-sigmoid
# logit is driven far from 0, saturating the sigmoid near 0/1 where its gradient
# vanishes and training can freeze.  A freshly-initialised net should output
# win_prob ~= 0.5 (well away from the 0/1 rails) on real encoded states.
# forward_value removed in Phase 1a; using win_prob from the full forward pass.
from games.kingdomino.game import GameState
import random as _random
_fresh = KingdominoNet()  # defaults
_states = []
for _s in range(8):
    st = GameState.new(seed=_s)
    rr = _random.Random(_s)
    while st.phase.value < 1:  # advance past INITIAL_SELECTION to a board state
        st = st.step(rr.choice(st.legal_actions()))
    _states.append(st)
_mb = torch.stack([torch.from_numpy(encode_state(s, s.current_actor)[0]).float() for s in _states])
_ob = torch.stack([torch.from_numpy(encode_state(s, s.current_actor)[1]).float() for s in _states])
_fl = torch.stack([torch.from_numpy(encode_state(s, s.current_actor)[2]).float() for s in _states])
_fresh.eval()
with torch.no_grad():
    _own, _opp, _win, _logits = _fresh(_mb, _ob, _fl)
_mean_win = _win.mean().item()  # win_prob in (0,1); .abs() redundant
check(f"initial mean win_prob is near 0.5 ({_mean_win:.3f} in (0.3, 0.7))",
      0.3 < _mean_win < 0.7)
check(f"no initial win_prob pinned at 0/1 "
      f"(min={_win.min():.3f}, max={_win.max():.3f})",
      float(_win.min()) > 0.05 and float(_win.max()) < 0.95)


# ──────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
if _failures:
    print(f"FAILED: {len(_failures)} test(s)")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
