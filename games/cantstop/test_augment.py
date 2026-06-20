# games/cantstop/test_augment.py
#
# Fast integration test for mirror augmentation — no model, no self-play run,
# no checkpoints touched. Builds synthetic records matching the self-play
# record schema, pushes them through the REAL augment_records and
# ReplayBuffer.to_tensors, and checks the mirrored half is consistent and
# tensorizes correctly. Seconds to run.
#
#   python -m games.cantstop.test_augment

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.features import FEATURE_SIZE, ACTION_SPACE
from games.cantstop.symmetry import (
    augment_records, reflect_record, is_symmetric,
)
from games.cantstop.self_play import ReplayBuffer


def _make_record(rng, game_id):
    """A synthetic record with the same schema as play_mcts_game emits."""
    k = int(rng.integers(1, 6))                       # a few legal actions
    legal = rng.choice(ACTION_SPACE, size=k, replace=False)
    mask = np.zeros(ACTION_SPACE, dtype=np.bool_)
    mask[legal] = True
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    w = rng.random(k).astype(np.float32); w /= w.sum()
    policy[legal] = w                                 # supported on legal only
    return {
        'features':     rng.random(FEATURE_SIZE).astype(np.float32),
        'mask':         mask,
        'mcts_policy':  policy,
        'mcts_value':   float(rng.random()),
        'action_idx':   int(rng.choice(legal)),
        'player':       int(rng.integers(0, 2)),
        'step_index':   int(rng.integers(0, 20)),
        'game_id':      game_id,
        'value_target': float(rng.random()),
    }


def main():
    assert is_symmetric(), "base game should report symmetric"

    rng = np.random.default_rng(0)
    N = 50
    records = [_make_record(rng, game_id=i % 7) for i in range(N)]

    aug = augment_records(records)
    assert len(aug) == 2 * N, f"expected {2*N} records, got {len(aug)}"
    assert aug[0] is records[0], "originals should be kept in place"

    for i in range(N):
        mirror = aug[N + i]
        ref = reflect_record(records[i])
        assert np.array_equal(mirror['features'], ref['features'])
        assert np.array_equal(mirror['mask'], ref['mask'])
        assert np.allclose(mirror['mcts_policy'], ref['mcts_policy'])
        assert mirror['action_idx'] == ref['action_idx']
        # value + metadata must be untouched by mirroring
        assert mirror['value_target'] == records[i]['value_target']
        assert mirror['player'] == records[i]['player']
        assert mirror['game_id'] == records[i]['game_id']
        # policy must stay supported on the (mirrored) legal mask
        assert mirror['mcts_policy'][~mirror['mask']].sum() < 1e-6
        # the chosen action must be legal under the mirrored mask
        assert mirror['mask'][mirror['action_idx']]

    # double reflection is the identity
    rr = reflect_record(reflect_record(records[0]))
    assert np.array_equal(rr['features'], records[0]['features'])
    assert rr['action_idx'] == records[0]['action_idx']

    # tensorize the augmented batch through the real buffer code
    rb = ReplayBuffer(max_size=10_000)
    f, m, p, v, a = rb.to_tensors(aug)
    assert tuple(f.shape) == (2 * N, FEATURE_SIZE)
    assert tuple(m.shape) == (2 * N, ACTION_SPACE)
    assert tuple(p.shape) == (2 * N, ACTION_SPACE)
    assert tuple(v.shape) == (2 * N,)
    assert tuple(a.shape) == (2 * N,)

    import torch
    assert not torch.isnan(f).any()
    assert not torch.isnan(p).any()
    assert not torch.isnan(v).any()

    print(f"PASS: {N} records -> {len(aug)} after mirror augmentation; "
          f"mirrored half consistent (features/mask/policy/action), value and "
          f"metadata preserved, policy stays on legal mask; tensorized to "
          f"{tuple(f.shape)} / {tuple(p.shape)} with no NaNs.")


if __name__ == "__main__":
    main()