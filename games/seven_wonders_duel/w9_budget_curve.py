"""Discovery-vs-budget, with the chance partition removed.

Searches the opponent's reply node directly as a root, at a ladder of budgets.
That is the counterfactual the Workstream 9 hypothesis rests on: if the reply
were NOT split across ten chance worlds, how much budget does the refutation
need before PUCT funds it past the ~8 visits the trace says it needs to correct?

Reports, per budget and per world:
  * visits the refutation received
  * whether it cleared its own baseline (the correction threshold)
  * its final rank

Run alongside the partitioned numbers, this separates two effects that the first
Workstream 9 experiment conflated: the partition divides the budget at the reply
node, and the prior decides how much of that budget the refutation gets.
"""

import json
import statistics
import sys

from games.seven_wonders_duel.w9_reference_case import (
    DEFAULT_DECISION_ROW,
    REPO_ROOT,
    build_mcts,
    load_evaluator_for,
    load_position,
    parse_args,
    resolve_action,
    revealed_slots,
    root_ranking,
)
from games.seven_wonders_duel.codec import decode_action, legal_action_indices
from games.seven_wonders_duel.engine import apply_action
from games.seven_wonders_duel.game import Phase
from games.seven_wonders_duel.search import (
    chance_signature,
    enumerate_chains,
    state_actor,
)

BUDGETS = (165, 400, 1000, 2000, 4000, 6000)
WORLDS = 4  # first four enumerated worlds, same set at every budget

pos = load_position(
    REPO_ROOT / "runs/seven_wonders_duel/bga_game_log/table_908370787.jsonl",
    DEFAULT_DECISION_ROW,
    0,
)
evaluator, _ = load_evaluator_for(parse_args([]))
act = resolve_action(pos, "Discard for coins: Caravansery")
slots = {tuple(s) for s in revealed_slots(pos.game, act["index"])}
chains = enumerate_chains(
    pos.game, chance_signature(pos.game, decode_action(pos.game, act["index"]))
)[:WORLDS]


def reply_node_state(outcomes):
    """The opponent's reply position for one chance world."""
    child = pos.game.clone()
    child.search_barrier = True
    apply_action(
        child, decode_action(child, act["index"]), chance_outcomes=outcomes
    )
    while state_actor(child) == pos.actor and child.phase is not Phase.COMPLETE:
        legal = legal_action_indices(child)
        if len(legal) != 1:
            break
        forced = decode_action(child, legal[0])
        if chance_signature(child, forced):
            break
        apply_action(child, forced)
    return child


rows = []
print(f"{'budget':>7} | " + " | ".join(f"{c[2][0][:11]:>11}" for c in chains))
print("-" * (9 + 14 * len(chains)))
for budget in BUDGETS:
    cells, record = [], {"budget": budget, "worlds": []}
    for outcomes, _p, key in chains:
        child = reply_node_state(outcomes)
        mcts = build_mcts(evaluator, parse_args([]))
        root = mcts.make_root(child)
        for _ in range(budget):
            mcts.descend(root)
        sign = 1.0 if root.actor == 0 else -1.0
        ranking = root_ranking(root, sign, child)
        exact = next(
            (
                r
                for r in ranking
                if (a := decode_action(child, r["index"])).wonder_name
                == "The Temple of Artemis"
                and a.slot_id is not None
                and tuple(a.slot_id) in slots
            ),
            None,
        )
        visits = exact["visits"] if exact else 0
        q = exact["q"] if exact else None
        cleared = q is not None and q > sign * root.value_p0
        cells.append(f"{visits:>5}v {'OK' if cleared else '--'} r{exact['rank'] if exact else '-':<2}")
        record["worlds"].append(
            {
                "revealed": key[0],
                "visits": visits,
                "rank": exact["rank"] if exact else None,
                "q": q,
                "cleared_baseline": cleared,
                "prior": exact["prior"] if exact else None,
            }
        )
    rows.append(record)
    print(f"{budget:>7} | " + " | ".join(cells))

out = REPO_ROOT / "runs/seven_wonders_duel/w9_reference/budget_curve.json"
out.write_text(
    json.dumps(
        {
            "note": "opponent reply node searched directly as root; no chance partition",
            "action": act["label"],
            "budgets": list(BUDGETS),
            "rows": rows,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"\nwrote {out}", file=sys.stderr)

# Where does it cross?
for record in rows:
    got = [w["visits"] for w in record["worlds"]]
    ok = sum(1 for w in record["worlds"] if w["cleared_baseline"])
    print(
        f"  budget {record['budget']:>5}: median visits {statistics.median(got):>6.0f}"
        f"   cleared {ok}/{len(got)}"
    )
