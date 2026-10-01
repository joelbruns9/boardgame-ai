"""Review diagnostic: exact maximum chance to win in the next turn.

Run using the main venv from boardgame-ai-cantstop. No source/model edits.
This is a lower bound on achievable eventual win probability for the
player about to move, not an independently solved full-game value.
"""
import json
from pathlib import Path

import cantstop_rust
import numpy as np

from games.cantstop.encoder import decode_features, PER_SEAT, MAX_SEATS
from games.cantstop.engine import RuleSet, Phase
from games.cantstop.model import load_net, NetEvaluator
from games.cantstop.portable_rng import PortableRng
from games.cantstop.rust_pool import run_pool, game_seeds
from games.cantstop.snapshot import snapshot
from games.cantstop.value_accuracy import uniform_indices


def main():
    checkpoint = 'runs/td0_personas/iter_0120.pt'
    ev = NetEvaluator(load_net(checkpoint, device='cuda'), device='cuda')
    rules = RuleSet.make(2)
    rng = PortableRng(260926)
    games = run_pool([rules] * 32, game_seeds(rng, 32), [ev],
                     in_flight=4, threads=2)
    pool = [(g, row, (r.winner - int(slot)) % 2)
            for g, r in enumerate(games)
            for row, slot in zip(r.features, r.winner_slots)]
    indices = uniform_indices(len(pool), min(120, len(pool)), rng)
    features = np.stack([pool[i][1] for i in indices])
    probs = ev.relative_probs(features)[:, 0]
    mirrored = features.copy()
    for seat in range(MAX_SEATS):
        start = seat * PER_SEAT
        mirrored[:, start:start + 11] = features[:, start:start + 11][:, ::-1]
        mirrored[:, start + 11:start + 22] = features[:, start + 11:start + 22][:, ::-1]
    mirror_probs = ev.relative_probs(mirrored)[:, 0]
    mirror_gap = np.abs(probs - mirror_probs)
    rows = []
    for i, pred in zip(indices, probs):
        game, row, active = pool[i]
        state = decode_features(row, active)
        solver = cantstop_rust.TurnSolver(snapshot(state))
        values = np.zeros((solver.num_leaves, 2), dtype='<f8')
        values[:, 1 - active] = 1.0
        solver.set_leaf_values_bytes(values.tobytes())
        immediate = solver.root_value(int(Phase.AWAIT_ROLL), None)[active]
        rows.append(dict(source_game=game, prediction=float(pred),
                         immediate_win_bound=immediate,
                         violation=immediate - float(pred),
                         snapshot=snapshot(state)))
    gaps = np.array([x['violation'] for x in rows])
    report = dict(checkpoint=checkpoint, seed=260926, source_games=32,
                  pool_rows=len(pool), sampled_boards=len(rows),
                  violations_over_1pp=int((gaps > .01).sum()),
                  violations_over_3pp=int((gaps > .03).sum()),
                  max_violation=float(gaps.max()),
                  mean_positive_violation=float(np.maximum(gaps, 0).mean()),
                  mirror_mean_abs_gap=float(mirror_gap.mean()),
                  mirror_max_abs_gap=float(mirror_gap.max()),
                  mirror_boards_over_1pp=int((mirror_gap > .01).sum()),
                  worst=sorted(rows, key=lambda x: -x['violation'])[:5])
    out = Path('C:/Users/joeld/projects/boardgame-ai/reviews/cantstop-phase4-threat-probe.json')
    out.write_text(json.dumps(report, indent=2, default=lambda x: x.item()), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'worst'}, indent=2))
    print('Worst board:', json.dumps(report['worst'][0], default=lambda x: x.item()))


if __name__ == '__main__':
    main()
