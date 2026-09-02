"""Go/no-go for afterstate aliasing across chance siblings.

The proposal: keep the chance node and its per-world action sets, but SHARE the
node reached after an action whose consequence does not depend on the revealed
card's identity -- burying it under a Wonder, where the card is inert. Then one
strategic line is searched once with N visits instead of ten times with N/10.

That is only sound if those post-burial positions really are interchangeable.
They are not identical: the buried card leaves the unseen pool, so what can be
revealed later differs. This measures whether that residual matters.

Three readings per world, at increasing depth, so the divergence can be seen to
grow (or not) with horizon:

  raw    -- the network's own value of the post-burial position, no search
  N=200  -- shallow search
  N=1200 -- deeper search, where pool differences have room to compound

Tight clustering at depth  -> aliasing is sound, build it.
Wide clustering at depth   -> aliasing averages genuinely different futures.

Run against BOTH burial targets. Burying the exposed slot is the case we want to
alias; burying Aqueduct is a control -- there the buried card is the SAME in
every world, so any spread there is baseline noise from the reveal itself rather
than from the aliasing.
"""

import json
import statistics

from games.seven_wonders_duel.w9_reference_case import (
    DEFAULT_DECISION_ROW,
    REPO_ROOT,
    build_mcts,
    load_evaluator_for,
    load_position,
    parse_args,
    resolve_action,
    revealed_slots,
    win_pct,
)
from games.seven_wonders_duel.codec import decode_action, legal_action_indices
from games.seven_wonders_duel.engine import apply_action
from games.seven_wonders_duel.game import Phase
from games.seven_wonders_duel.search import (
    chance_signature,
    enumerate_chains,
    state_actor,
)

pos = load_position(
    REPO_ROOT / "runs/seven_wonders_duel/bga_game_log/table_908370787.jsonl",
    DEFAULT_DECISION_ROW,
    0,
)
evaluator, _ = load_evaluator_for(parse_args([]))
act = resolve_action(pos, "Discard for coins: Caravansery")
exposed = {tuple(s) for s in revealed_slots(pos.game, act["index"])}
chains = enumerate_chains(
    pos.game, chance_signature(pos.game, decode_action(pos.game, act["index"]))
)

BUDGETS = (0, 200, 1200)
report = {"action": act["label"], "budgets": list(BUDGETS), "targets": {}}

# Control: the reply nodes THEMSELVES, before any action. That is how much
# these worlds differ anyway -- the reveal leaves a different card on the board,
# which changes options regardless of aliasing. If the post-burial spread is no
# larger than this, aliasing is removing noise rather than signal.
for target_name, want_exposed in (
    ("CONTROL: reply node, no action", None),
    ("bury EXPOSED slot", True),
):
    print(f"\n=== after Artemis, {target_name} ===")
    print(f"  {'revealed':<15}" + "".join(f"{'raw' if b == 0 else f'N={b}':>10}" for b in BUDGETS))
    rows = []
    for outcomes, _p, key in chains:
        # opponent's reply position for this world
        child = pos.game.clone()
        child.search_barrier = True
        apply_action(child, decode_action(child, act["index"]), chance_outcomes=outcomes)
        while state_actor(child) == pos.actor and child.phase is not Phase.COMPLETE:
            legal = legal_action_indices(child)
            if len(legal) != 1:
                break
            forced = decode_action(child, legal[0])
            if chance_signature(child, forced):
                break
            apply_action(child, forced)

        if want_exposed is None:
            after = child  # the control: measure the world as it stands
        else:
            chosen = None
            for index in legal_action_indices(child):
                a = decode_action(child, index)
                if a.wonder_name != "The Temple of Artemis" or a.slot_id is None:
                    continue
                if (tuple(a.slot_id) in exposed) == want_exposed:
                    chosen = index
                    break
            if chosen is None:
                print(f"  {key[0]:<15} (not available)")
                continue
            after = child.clone()
            after.search_barrier = True
            apply_action(after, decode_action(after, chosen))

        cells, values = [], {}
        for budget in BUDGETS:
            mcts = build_mcts(evaluator, parse_args([]))
            root = mcts.make_root(after)  # expansion alone gives the raw value
            for _ in range(budget):
                mcts.descend(root)
            # deciding player's frame, so numbers compare to the panel
            pct = win_pct(pos.sign * root.value_p0)
            values[str(budget)] = round(pct, 2)
            cells.append(f"{pct:>10.1f}")
        print(f"  {key[0]:<15}" + "".join(cells))
        rows.append({"revealed": key[0], "win_pct": values})

    report["targets"][target_name] = rows
    print(f"  {'-' * 15}" + "-" * (10 * len(BUDGETS)))
    for budget in BUDGETS:
        vals = [r["win_pct"][str(budget)] for r in rows]
        label = "raw" if budget == 0 else f"N={budget}"
        print(
            f"  {label:<15}spread={max(vals) - min(vals):>6.1f}  "
            f"sd={statistics.pstdev(vals):>5.2f}  mean={statistics.fmean(vals):>6.1f}"
        )

out = REPO_ROOT / "runs/seven_wonders_duel/w9_reference/afterstate_clustering.json"
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"\nwrote {out}")
