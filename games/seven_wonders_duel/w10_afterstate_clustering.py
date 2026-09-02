"""Is a post-burial afterstate cluster a sound abstraction? (Workstream 10)

**These are NOT transpositions.** After the revealed card is buried under a
Wonder its effect is inert, but its identity is not irrelevant: it is known to be
out of the unseen pool, so future reveal supports and conditional values differ
between members. `ClosedNode` holds one concrete `GameState`, and both
`sample_outcomes` and `enumerate_chains` read `unseen_pool(node.state...)`, so
sharing a node would make later chance enumeration use whichever member built it
and would replace per-world conditional values with an aggregate. Any cross-world
object must therefore be an explicitly APPROXIMATE cluster, never a transposition.

This script measures whether that approximation is defensible. Scalar value
spread alone cannot license it -- a 2-3 point deviation is harmless against a
20-point root gap and decisive against a one-point action margin -- so it also
reports what aliasing would do to the POLICY:

  * best-action agreement across the cluster's members;
  * cluster regret: what each member loses by playing the cluster's majority
    action instead of its own best;
  * the worst member error relative to that member's own action margin.

Three arms, and the third is the one that matters:

  reply node        the ten worlds before anyone acts -- how much they differ
                    anyway. Aliasing must remove less than this to be removing
                    noise rather than signal.
  bury EXPOSED      the case we want to alias. The revealed card goes under the
                    Wonder, so its identity is discarded by the game itself.
  bury AQUEDUCT     the control. Same Wonder, same extra turn, but the revealed
                    card STAYS on the board. If the spread contracts here too,
                    the contraction comes from advancing into a constrained
                    extra-turn afterstate, not from discarding the identity, and
                    the whole premise fails.

Burying Aqueduct itself uncovers slots and fires a reveal, so that arm is
averaged over sampled reveal outcomes, probability-weighted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

from .w9_reference_case import (
    DEFAULT_DECISION_ROW,
    REPO_ROOT,
    build_mcts,
    checkpoint_fingerprint,
    load_evaluator_for,
    load_position,
    parse_args as ref_parse_args,
    resolve_action,
    revealed_slots,
    root_ranking,
    win_pct,
)
from .codec import decode_action, legal_action_indices
from .engine import apply_action
from .game import Phase
from .search import chance_signature, enumerate_chains, state_actor


def reply_node(position, action_index, outcomes):
    """The opponent's reply position for one chance world, at the common ply."""

    child = position.game.clone()
    child.search_barrier = True
    apply_action(child, decode_action(child, action_index), chance_outcomes=outcomes)
    while state_actor(child) == position.actor and child.phase is not Phase.COMPLETE:
        legal = legal_action_indices(child)
        if len(legal) != 1:
            break
        forced = decode_action(child, legal[0])
        if chance_signature(child, forced):
            break
        apply_action(child, forced)
    return child


def artemis_action(state, exposed, *, on_exposed: bool):
    """The Artemis build burying (or not burying) a slot this move exposed."""

    for index in legal_action_indices(state):
        action = decode_action(state, index)
        if action.wonder_name != "The Temple of Artemis" or action.slot_id is None:
            continue
        if (tuple(action.slot_id) in exposed) == on_exposed:
            return index
    return None


def search_state(state, evaluator, args, seed, sims):
    """(value in decider frame, ranking) after `sims` simulations."""

    cfg = ref_parse_args(["--seed", str(seed)])
    mcts = build_mcts(evaluator, cfg)
    root = mcts.make_root(state)
    for _ in range(sims):
        mcts.descend(root)
    sign = 1.0 if root.actor == 0 else -1.0
    return root, sign, root_ranking(root, sign, state)


def policy_metrics(members: list[dict]) -> dict:
    """What aliasing would cost the POLICY, not just the value.

    `members` each carry {best, q_by_action, margin}. The cluster would play one
    action for every member; regret is what a member loses by taking the
    cluster's majority action instead of its own best, and it is only meaningful
    against that member's own margin between its top two actions.
    """

    if not members:
        return {}
    votes = Counter(m["best"] for m in members)
    majority, agreement = votes.most_common(1)[0]
    regrets, ratios = [], []
    for member in members:
        own = member["q_by_action"].get(member["best"])
        shared = member["q_by_action"].get(majority)
        if own is None or shared is None:
            continue  # majority action not legal here -- cluster is unsound
        regret = own - shared
        regrets.append(regret)
        if member["margin"] > 1e-9:
            ratios.append(regret / member["margin"])
    return {
        "members": len(members),
        "best_action_agreement": f"{agreement}/{len(members)}",
        "distinct_best_actions": len(votes),
        "majority_action_illegal_in": sum(
            1 for m in members if majority not in m["q_by_action"]
        ),
        "cluster_regret_mean": round(statistics.fmean(regrets), 4) if regrets else None,
        "cluster_regret_max": round(max(regrets), 4) if regrets else None,
        "regret_over_margin_max": round(max(ratios), 3) if ratios else None,
    }


def run_arm(name, builder, position, evaluator, args, log) -> dict:
    """One arm: build each world's state, search it at every seed, summarise."""

    log(f"\n=== {name} ===")
    worlds, raw_vals, searched_vals = [], [], []
    members: list[dict] = []
    for outcomes, _p, key in position.chains:
        built = builder(outcomes, key)
        if built is None:
            log(f"  {key[0]:<15} (not available)")
            continue
        # `built` is a list of (state, weight): the Aqueduct arm fires its own
        # reveal, so its afterstate is a distribution rather than one state.
        raw_seeds, search_seeds = [], []
        per_seed_rank = None
        for seed in range(args.seeds):
            raw = sum(
                w * win_pct(position.sign * search_state(s, evaluator, args, seed, 0)[0].value_p0)
                for s, w in built
            )
            acc, rank_for_metrics = 0.0, None
            for s, w in built:
                root, sign, ranking = search_state(s, evaluator, args, seed, args.sims)
                acc += w * win_pct(position.sign * root.value_p0)
                if rank_for_metrics is None:
                    rank_for_metrics = (ranking, sign)
            raw_seeds.append(raw)
            search_seeds.append(acc)
            if per_seed_rank is None:
                per_seed_rank = rank_for_metrics
        raw_mean = statistics.fmean(raw_seeds)
        searched_mean = statistics.fmean(search_seeds)
        raw_vals.append(raw_mean)
        searched_vals.append(searched_mean)
        ranking = per_seed_rank[0]
        by_action = {r["index"]: r["q"] for r in ranking}
        top = sorted(ranking, key=lambda r: -r["visits"])
        margin = (top[0]["q"] - top[1]["q"]) if len(top) > 1 else 0.0
        members.append(
            {"best": top[0]["index"], "q_by_action": by_action, "margin": abs(margin)}
        )
        worlds.append(
            {
                "revealed": key[0],
                "raw": round(raw_mean, 2),
                "searched": round(searched_mean, 2),
                "seed_spread": round(max(search_seeds) - min(search_seeds), 2),
                "best_action": top[0]["index"],
                "best_label": top[0]["label"],
                "margin": round(abs(margin), 4),
            }
        )
        log(
            f"  {key[0]:<15} raw={raw_mean:>6.1f}  N={args.sims}:{searched_mean:>6.1f}"
            f"  seed-spread={worlds[-1]['seed_spread']:>4.1f}  best={top[0]['label'][:34]}"
        )

    def spread(vals):
        return {
            "spread": round(max(vals) - min(vals), 2),
            "sd": round(statistics.pstdev(vals), 2),
            "mean": round(statistics.fmean(vals), 2),
        }

    summary = {
        "worlds": worlds,
        "value_raw": spread(raw_vals) if raw_vals else {},
        "value_searched": spread(searched_vals) if searched_vals else {},
        "policy": policy_metrics(members),
    }
    log(f"  {'-' * 60}")
    log(f"  raw       {summary['value_raw']}")
    log(f"  searched  {summary['value_searched']}")
    log(f"  policy    {summary['policy']}")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--table", default="908370787")
    parser.add_argument("--decision-row", type=int, default=DEFAULT_DECISION_ROW)
    parser.add_argument("--resample-seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="extension_7wd/candidate_0085.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-migration", action="store_true")
    parser.add_argument("--sims", type=int, default=1200)
    parser.add_argument("--seeds", type=int, default=2, help="independent searches per world")
    parser.add_argument("--reveal-samples", type=int, default=2,
                        help="reveal outcomes sampled for the Aqueduct arm")
    parser.add_argument("--walk-action", default="Discard for coins: Caravansery")
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    log = (lambda *_: None) if args.quiet else (
        lambda m: print(m, file=sys.stderr, flush=True)
    )

    log_path = REPO_ROOT / f"runs/seven_wonders_duel/bga_game_log/table_{args.table}.jsonl"
    position = load_position(log_path, args.decision_row, args.resample_seed)
    evaluator, checkpoint_path = load_evaluator_for(args)
    act = resolve_action(position, args.walk_action)
    exposed = {tuple(s) for s in revealed_slots(position.game, act["index"])}
    position.chains = enumerate_chains(
        position.game,
        chance_signature(position.game, decode_action(position.game, act["index"])),
    )
    log(f"{len(position.chains)} worlds behind {act['label']!r}; exposed slots {sorted(exposed)}")

    def control(outcomes, key):
        return [(reply_node(position, act["index"], outcomes), 1.0)]

    def bury(on_exposed):
        def build(outcomes, key):
            child = reply_node(position, act["index"], outcomes)
            index = artemis_action(child, exposed, on_exposed=on_exposed)
            if index is None:
                return None
            action = decode_action(child, index)
            specs = chance_signature(child, action)
            if not specs:
                after = child.clone()
                after.search_barrier = True
                apply_action(after, decode_action(after, index))
                return [(after, 1.0)]
            # Burying a card that itself uncovers slots fires a reveal, so this
            # afterstate is a distribution. Sample it, probability-weighted and
            # renormalised, rather than picking one outcome.
            chains = enumerate_chains(child, specs)
            picked = random.Random(args.resample_seed).sample(
                chains, min(args.reveal_samples, len(chains))
            )
            mass = sum(p for _o, p, _k in picked) or 1.0
            built = []
            for sub_outcomes, prob, _k in picked:
                after = child.clone()
                after.search_barrier = True
                apply_action(after, decode_action(after, index),
                             chance_outcomes=sub_outcomes)
                built.append((after, prob / mass))
            return built
        return build

    report = {
        "harness": "w10_afterstate_clustering",
        "note": "APPROXIMATE cluster, not a transposition -- members' unseen pools differ",
        "position": {
            "table": args.table,
            "decision_row": args.decision_row,
            "observation_sha256": position.observation_sha256,
            "action": act["label"],
            "worlds": len(position.chains),
        },
        "checkpoint": checkpoint_fingerprint(checkpoint_path),
        "search": {"sims": args.sims, "seeds": args.seeds,
                   "reveal_samples": args.reveal_samples},
        "code_version": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()[:16],
        "arms": {},
    }
    for name, builder in (
        ("CONTROL: reply node, no action", control),
        ("bury EXPOSED slot", bury(True)),
        ("CONTROL: bury AQUEDUCT (identity survives)", bury(False)),
    ):
        report["arms"][name] = run_arm(name, builder, position, evaluator, args, log)

    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        log(f"\nwrote {out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
