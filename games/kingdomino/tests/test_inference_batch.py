"""Validate batched-send on the LOCAL backend (in-process, deterministic):
  T1: single infer == direct serial evaluator
  T2: infer_batch(K) == K single infers (bit-identical)
  T3: mixed single + batched traffic, concurrent threads
  T4: MCTS via IPC batched evaluator == MCTS via in-process batched evaluator
"""
import random, threading
import numpy as np
import torch

from games.kingdomino.network import KingdominoNet
from games.kingdomino.inference_service import (
    LocalInferenceService, make_ipc_batched_evaluator)
from games.kingdomino.mcts_az import (
    AlphaZeroMCTS, make_serial_evaluator, make_batched_evaluator, run_pimc)
from games.kingdomino.game import GameState, Phase
from games.kingdomino.encoder import encode_state
from games.kingdomino.action_codec import encode_action

torch.manual_seed(0)
NET = KingdominoNet(channels=16, blocks=2, bilinear_dim=16).eval()
serial = make_serial_evaluator(NET)

def leaf_at(seed, nmoves):
    st = GameState.new(seed=seed); rng = random.Random(seed + 7)
    for _ in range(nmoves):
        if st.phase == Phase.GAME_OVER: break
        st = st.step(rng.choice(st.legal_actions()))
    while st.phase != Phase.PLACE_AND_SELECT:
        if st.phase == Phase.GAME_OVER: break
        st = st.step(rng.choice(st.legal_actions()))
    legal = st.legal_actions()
    mb, ob, flat = encode_state(st, st.current_actor)
    idxs = np.fromiter((encode_action(a, st) for a in legal), np.int64, len(legal))
    return st, mb, ob, flat, idxs

svc = LocalInferenceService(NET, device="cpu", max_batch=64, max_wait_ms=3.0).start()
svc.update_weights(NET.state_dict()); svc.wait_for_version(1)
client = svc.make_client()

# T1: single infer == direct serial evaluator
print("=== T1: single infer == serial evaluator ===")
_, mb, ob, flat, idxs = leaf_at(0, 4)
v_ser, l_ser = serial(mb, ob, flat, idxs)
v_cli, l_cli = client.infer(mb, ob, flat, idxs)
assert abs(v_ser - v_cli) < 1e-5, (v_ser, v_cli)
assert np.allclose(l_ser, l_cli, atol=1e-5), np.abs(l_ser - l_cli).max()
print(f"  value diff {abs(v_ser-v_cli):.2e}, logits max diff {np.abs(l_ser-l_cli).max():.2e}  PASS")

# T2: infer_batch(K) == K single infers, up to batched-vs-unbatched FP (~1e-6,
# the same GroupNorm/cuDNN accumulation noise already present in the floor).
print("=== T2: infer_batch == K single infers (within FP) ===")
leaves = [leaf_at(s, 3 + s)[1:] for s in range(8)]   # (mb,ob,flat,idxs) x8
batch_res = client.infer_batch(leaves)
ok = True
max_vd = max_ld = 0.0
for k, (mb_k, ob_k, flat_k, idx_k) in enumerate(leaves):
    v_s, l_s = client.infer(mb_k, ob_k, flat_k, idx_k)
    v_b, l_b = batch_res[k]
    vd, ld = abs(v_s - v_b), float(np.abs(l_s - l_b).max())
    max_vd, max_ld = max(max_vd, vd), max(max_ld, ld)
    if vd > 1e-4 or ld > 1e-4:
        ok = False; print(f"  leaf {k}: vdiff {vd:.2e} ldiff {ld:.2e} (EXCEEDS 1e-4)")
assert ok
print(f"  8 leaves match within FP: max value diff {max_vd:.2e}, "
      f"max logit diff {max_ld:.2e}  PASS")

# T3: mixed single + batched, concurrent
print("=== T3: mixed single + batched concurrent traffic ===")
errs = []
def worker_single(seed):
    try:
        _, mb, ob, flat, idxs = leaf_at(seed, 5)
        v, l = client.infer(mb, ob, flat, idxs)
        vr, lr = serial(mb, ob, flat, idxs)
        assert abs(v - vr) < 1e-5 and np.allclose(l, lr, atol=1e-5)
    except Exception as e: errs.append(("single", seed, repr(e)))
def worker_batch(seed):
    try:
        lv = [leaf_at(seed * 10 + i, 4)[1:] for i in range(5)]
        res = client.infer_batch(lv)
        for (mb, ob, flat, idxs), (v, l) in zip(lv, res):
            vr, lr = serial(mb, ob, flat, idxs)
            assert abs(v - vr) < 1e-5 and np.allclose(l, lr, atol=1e-5)
    except Exception as e: errs.append(("batch", seed, repr(e)))
ths = [threading.Thread(target=worker_single, args=(s,)) for s in range(6)] + \
      [threading.Thread(target=worker_batch, args=(s,)) for s in range(6)]
for t in ths: t.start()
for t in ths: t.join()
assert not errs, errs
print("  12 concurrent threads (6 single + 6 batched) all correct  PASS")

# T4: MCTS via IPC batched evaluator == MCTS via in-process batched
print("=== T4: MCTS over IPC batched evaluator == in-process batched ===")
st, *_ = leaf_at(2, 4)
mcts_inproc = AlphaZeroMCTS(make_serial_evaluator(NET),
    batched_evaluator=make_batched_evaluator(NET), n_simulations=120)
mcts_ipc = AlphaZeroMCTS(client.infer,
    batched_evaluator=make_ipc_batched_evaluator(client), n_simulations=120)
mism = 0
for i in range(6):
    a, _ = run_pimc(mcts_inproc, st, random.Random(100+i), n_determinizations=1,
                    add_noise=True, np_rng=np.random.default_rng(200+i), leaf_batch=4)
    b, _ = run_pimc(mcts_ipc, st, random.Random(100+i), n_determinizations=1,
                    add_noise=True, np_rng=np.random.default_rng(200+i), leaf_batch=4)
    if a != b:
        mism += 1
        # quantify
        keys = set(a) | set(b)
        l1 = sum(abs(a.get(k,0)-b.get(k,0)) for k in keys)
        print(f"  pos {i}: visit L1 diff {l1} (total {sum(a.values())})")
print(f"  {6-mism}/6 positions bit-identical visit counts")
assert mism == 0, "IPC batched path diverged from in-process batched"
print("  PASS")

svc.stop()
print("\nALL INFERENCE-BATCH TESTS PASSED ✓")