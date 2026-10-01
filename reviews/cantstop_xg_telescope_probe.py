import numpy as np
from games.cantstop.rollout_search import RolloutBackend, RolloutConfig
from games.cantstop.decision_search import TurnTableBackend
from games.cantstop.rust_solver import hashed_evaluator, sample_positions
from games.cantstop.engine import ALL_RULESETS, Phase
from games.cantstop.solver import ProgressHeuristic
ev = ProgressHeuristic()
for rules in ALL_RULESETS[:4]:
    for st in sample_positions(rules, 3, 6):
        if st.phase == Phase.AWAIT_ROLL: continue
        base = TurnTableBackend(ev).evaluate(st)
        bv = {o.action.key: np.array(o.value) for o in base.options}
        for H in (0, 1):
            rb = RolloutBackend(ev, RolloutConfig(samples=24, horizon=H, dice_luck=True), retain_samples=True)
            rb.evaluate(st)
            for k, s in rb.last_samples.items():
                a = st.active_player
                adj, raw = s['adjusted'][:, a], s['raw'][:, a]
                print(rules.num_players, int(st.phase), 'H', H, k, 'adjSD %.2e rawSD %.3f  adjmean-base %+.2e' % (adj.std(ddof=1), raw.std(ddof=1), adj.mean()-bv[k][a]))
