"""Is a post-burial afterstate cluster a sound abstraction? (Workstream 10)

**These are NOT transpositions.** After the revealed card is buried under a
Wonder its effect is inert, but its identity is not irrelevant: it is known to be
out of the unseen pool, so members differ in future reveal support and in
conditional values. `ClosedNode` holds one concrete `GameState`, and both
`sample_outcomes` and `enumerate_chains` read `unseen_pool(node.state...)`, so
sharing a node would make later chance enumeration use whichever member built it
and would replace per-world conditional values with an aggregate. Any cross-world
object must be an explicitly APPROXIMATE cluster, never a transposition.

This measures whether that approximation is defensible, on three axes, because
the cheap one is not sufficient:

* **value spread** -- necessary, nowhere near sufficient. A 2-3 point deviation
  is harmless against a 20-point root gap and decisive at a one-point margin.
* **policy structure** -- do members agree on a best action, and is the cluster's
  action even LEGAL in every member? An illegal majority action means a shared
  policy is undefined, not merely lossy.
* **reference regret** -- what a member actually loses by playing the cluster's
  action instead of its own best, measured against that member's own margin.

Regret is computed from **dedicated per-action reference searches**, not from the
member's own visit distribution. Ranking by visits while differencing Q mixes two
orderings and admits negative "regret"; an earlier version of this file did that
and reported -0.0131.

Every public state is its own member. A secondary reveal fired by the burial is
public and would remain in any honest cluster key, so those outcomes are separate
members rather than averaged into one.

Arms:

  reply node       the worlds before anyone acts -- how much they differ anyway.
  bury EXPOSED     the case we want to cluster: the revealed card goes under the
                   Wonder, so the game itself discards its identity.
  bury AQUEDUCT    the control. Same Wonder, same extra turn, but the revealed
                   card STAYS on the board. Clustering these by a key that drops
                   the revealed identity SHOULD fail, and the arm exists to show
                   that the metrics detect it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .codec import decode_action, legal_action_indices
from .engine import apply_action
from .game import Phase
from .search import chance_signature, enumerate_chains, state_actor
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


def find_burial(state, wonder: str, *, exposed, card: str | None):
    """The Wonder build burying a slot this move exposed, or a NAMED card.

    Naming the card matters: "the first Artemis action not on an exposed slot"
    happens to be Aqueduct in this position, but that is codec iteration order,
    not a definition, and this harness takes a CLI.
    """

    for index in legal_action_indices(state):
        action = decode_action(state, index)
        if action.wonder_name != wonder or action.slot_id is None:
            continue
        on_exposed = tuple(action.slot_id) in exposed
        if card is None:
            if on_exposed:
                return index
        elif not on_exposed:
            slot = state.tableau.cards.get(tuple(action.slot_id))
            if slot is not None and slot.card_name == card:
                return index
    return None


def search(state, evaluator, seed, sims):
    """(root, sign, ranking) after `sims` simulations from `state`."""

    mcts = build_mcts(evaluator, ref_parse_args(["--seed", str(seed)]))
    root = mcts.make_root(state)
    for _ in range(sims):
        mcts.descend(root)
    sign = 1.0 if root.actor == 0 else -1.0
    return root, sign, root_ranking(root, sign, state)


def reference_q(state, action_index, evaluator, args):
    """A dedicated deeper value for ONE action, in the mover's frame.

    Searches the child directly rather than reading the parent's edge Q: that Q
    is whatever the parent's allocation happened to buy, which is the quantity
    under suspicion.

    If the action fires chance, the value is the **probability-weighted
    expectation over its support**, and it is averaged over `--ref-seeds`
    independent searches. An earlier version picked ONE outcome uniformly and
    called it the reference -- discarding the probabilities the enumerator had
    just handed it, at a single seed.
    """

    mover_sign = 1.0 if state_actor(state) == 0 else -1.0
    specs = chance_signature(state, decode_action(state, action_index))
    if specs:
        chains = enumerate_chains(state, specs)
        if args.ref_max_outcomes and len(chains) > args.ref_max_outcomes:
            # Stratified subset, renormalised: exhaustive support is preferred
            # and used whenever it fits the budget.
            picked = random.Random(args.resample_seed).sample(
                chains, args.ref_max_outcomes
            )
        else:
            picked = chains
        exhaustive = len(picked) == len(chains)
    else:
        picked, exhaustive = [([], 1.0, ())], True

    mass = sum(p for _o, p, _k in picked) or 1.0
    total = 0.0
    for outcomes, probability, _key in picked:
        child = state.clone()
        child.search_barrier = True
        apply_action(
            child, decode_action(child, action_index),
            chance_outcomes=outcomes or None,
        )
        seeded = []
        for offset in range(args.ref_seeds):
            root, _sign, _ranking = search(
                child, evaluator, args.seed_base + offset, args.ref_sims
            )
            seeded.append(mover_sign * root.value_p0)
        total += (probability / mass) * statistics.fmean(seeded)
    return total, exhaustive


def information_state_key(state, buried_identity):
    """Normalized public key, omitting ONLY the validated buried identity.

    Not a full `GameState` fingerprint: that carries determinized hidden
    placements, which differ for reasons the abstraction has nothing to say
    about. This keeps every PUBLIC fact -- including any secondary reveal the
    burial fired, which stays on the board and therefore stays in the key -- and
    removes the buried card's identity from each public place it appears.

    Approximate by construction: members sharing a key still differ in their
    remaining unseen pools. That residual is what the regret measurement prices.
    """

    tableau = sorted(
        (slot, c.card_name if c.revealed else "?")
        for slot, c in state.tableau.cards.items() if c.present
    )
    cities = []
    for city in state.cities:
        cities.append((
            tuple(sorted(city.buildings)),
            tuple(sorted(city.built_wonders)),
            tuple(sorted(city.wonders)),
            tuple(sorted(city.progress_tokens)),
            city.coins,
        ))
    buried = tuple(sorted(
        b for b in state.buried_cards if b != buried_identity
    ))
    raw = json.dumps({
        "age": state.age,
        "actor": state_actor(state),
        "tableau": tableau,
        "cities": cities,
        "conflict": state.conflict_position,
        "tokens": tuple(sorted(state.available_progress_tokens)),
        "buried_count": len(state.buried_cards),
        "buried_others": buried,
        "discard": tuple(sorted(state.discard_pile)),
    }, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_members(position, act, exposed, builder, args):
    """Every public state in this arm, as its own member.

    A secondary reveal fired by the burial is public information and would stay
    in any honest cluster key, so it gets its own member instead of being
    averaged away.
    """

    members = []
    for outcomes, _p, key in position.chains:
        world = key[0] if isinstance(key, tuple) else key
        child = reply_node(position, act["index"], outcomes)
        built = builder(child)
        if built is None:
            continue
        for state, secondary, buried in built:
            members.append({
                "world": world, "secondary": secondary, "state": state,
                "cluster_key": information_state_key(state, buried),
            })
    return members


def analyse(members, evaluator, args, log) -> dict:
    """Value spread, policy stability across seeds, and reference regret."""

    # -- per-member search, every seed ---------------------------------------
    for m in members:
        raw = win_pct(args.sign * search(m["state"], evaluator, args.seed_base, 0)[0].value_p0)
        vals, bests, topk_union = [], [], set()
        for offset in range(args.seeds):
            root, _sign, ranking = search(
                m["state"], evaluator, args.seed_base + offset, args.sims
            )
            vals.append(win_pct(args.sign * root.value_p0))
            bests.append(max(ranking, key=lambda r: r["visits"])["index"])
            topk_union.update(
                r["index"]
                for r in sorted(ranking, key=lambda r: -r["visits"])[: args.topk]
            )
        m["raw"] = raw
        m["searched"] = statistics.fmean(vals)
        m["seed_spread"] = max(vals) - min(vals)
        m["visit_best_by_seed"] = bests
        m["visit_best"] = Counter(bests).most_common(1)[0][0]
        # Top-k from EVERY seed, unioned. Taking it from the last seed's ranking
        # alone can drop a low-prior correct action by construction -- which is
        # the exact failure this whole investigation is about.
        m["visit_topk"] = sorted(topk_union)
        m["seed_stable"] = len(set(bests)) == 1
        m["legal"] = set(legal_action_indices(m["state"]))

    # -- reference Q on the candidate set ------------------------------------
    # Only the actions that could be the cluster's choice or a member's own best
    # need reference values; scoring all ~12 actions per member is wasted work.
    candidates = {index for m in members for index in m["visit_topk"]}
    log(f"  reference-scoring {len(candidates)} candidate action(s) per member")
    for m in members:
        scored = {
            index: reference_q(m["state"], index, evaluator, args)
            for index in candidates
            if index in m["legal"]
        }
        m["ref_q"] = {index: value for index, (value, _ex) in scored.items()}
        m["ref_exhaustive"] = all(ex for _v, ex in scored.values())
        if m["ref_q"]:
            ordered = sorted(m["ref_q"].items(), key=lambda kv: -kv[1])
            m["ref_best"] = ordered[0][0]
            m["ref_margin"] = (
                ordered[0][1] - ordered[1][1] if len(ordered) > 1 else None
            )
        else:
            m["ref_best"], m["ref_margin"] = None, None

    # -- the cluster's action, by reference Q, and what it costs --------------
    votes = Counter(m["ref_best"] for m in members if m["ref_best"] is not None)
    cluster_action, agree = votes.most_common(1)[0] if votes else (None, 0)
    regrets, ratios, illegal = [], [], 0
    for m in members:
        if cluster_action not in m["legal"]:
            illegal += 1
            continue
        if m["ref_best"] is None:
            continue
        regret = m["ref_q"][m["ref_best"]] - m["ref_q"][cluster_action]
        regrets.append(regret)
        if m["ref_margin"]:
            ratios.append(regret / abs(m["ref_margin"]))

    vals = [m["searched"] for m in members]
    raws = [m["raw"] for m in members]
    return {
        "members": len(members),
        "value_raw": {
            "spread": round(max(raws) - min(raws), 2),
            "sd": round(statistics.pstdev(raws), 2),
            "mean": round(statistics.fmean(raws), 2),
        },
        "value_searched": {
            "spread": round(max(vals) - min(vals), 2),
            "sd": round(statistics.pstdev(vals), 2),
            "mean": round(statistics.fmean(vals), 2),
        },
        "policy": {
            "visit_best_distinct": len({m["visit_best"] for m in members}),
            "seed_stable_members": sum(1 for m in members if m["seed_stable"]),
            "max_seed_value_spread": round(max(m["seed_spread"] for m in members), 2),
            "ref_best_distinct": len(votes),
            "ref_best_agreement": f"{agree}/{len(members)}",
            "cluster_action_illegal_in": illegal,
        },
        "reference_regret": {
            # Non-negative by construction now: best and comparison come from the
            # same reference ordering.
            "mean": round(statistics.fmean(regrets), 4) if regrets else None,
            "max": round(max(regrets), 4) if regrets else None,
            "over_margin_max": round(max(ratios), 3) if ratios else None,
            "scored_members": len(regrets),
        },
        "worlds": [
            {
                "world": m["world"],
                "secondary": m["secondary"],
                "raw": round(m["raw"], 2),
                "searched": round(m["searched"], 2),
                "seed_spread": round(m["seed_spread"], 2),
                "seed_stable": m["seed_stable"],
                "ref_best": m["ref_best"],
                "ref_margin": round(m["ref_margin"], 4) if m["ref_margin"] else None,
            }
            for m in members
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--table", default="908370787")
    parser.add_argument("--decision-row", type=int, default=DEFAULT_DECISION_ROW)
    parser.add_argument("--resample-seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="extension_7wd/candidate_0085.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-migration", action="store_true")
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--ref-sims", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--reveal-samples", type=int, default=2)
    parser.add_argument("--ref-seeds", type=int, default=2,
                        help="independent searches per reference action value")
    parser.add_argument("--ref-max-outcomes", type=int, default=0,
                        help="cap on chance outcomes per reference value; "
                             "0 means exhaustive support (preferred)")
    parser.add_argument(
        "--topk", type=int, default=3,
        help="actions per member entering the reference-scored candidate set; "
             "must exceed 1 or margins are undefined when members agree",
    )
    parser.add_argument("--walk-action", default="Discard for coins: Caravansery")
    parser.add_argument("--wonder", default="The Temple of Artemis")
    parser.add_argument("--control-card", default="Aqueduct",
                        help="burial target for the negative control, BY NAME")
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
    args.sign = position.sign
    log(f"{len(position.chains)} worlds behind {act['label']!r}; exposed {sorted(exposed)}")

    def burial(card):
        def build(child):
            index = find_burial(child, args.wonder, exposed=exposed, card=card)
            if index is None:
                return None
            action = decode_action(child, index)
            slot = child.tableau.cards.get(tuple(action.slot_id))
            buried = slot.card_name if slot is not None and slot.revealed else None
            specs = chance_signature(child, action)
            if not specs:
                after = child.clone()
                after.search_barrier = True
                apply_action(after, decode_action(after, index))
                return [(after, None, buried)]
            chains = enumerate_chains(child, specs)
            picked = random.Random(args.resample_seed).sample(
                chains, min(args.reveal_samples, len(chains))
            )
            built = []
            for sub, _p, sub_key in picked:
                after = child.clone()
                after.search_barrier = True
                apply_action(after, decode_action(after, index), chance_outcomes=sub)
                built.append((
                    after,
                    list(sub_key) if isinstance(sub_key, tuple) else [sub_key],
                    buried,
                ))
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
        "search": {
            "sims": args.sims, "ref_sims": args.ref_sims, "seeds": args.seeds,
            "seed_base": args.seed_base, "reveal_samples": args.reveal_samples,
        },
        "code_version": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
        "arms": {},
    }
    arms = (
        ("CONTROL: reply node, no action", lambda child: [(child, None, None)]),
        (f"bury EXPOSED slot ({args.wonder})", burial(None)),
        (f"CONTROL: bury {args.control_card} (identity survives)",
         burial(args.control_card)),
    )
    for name, builder in arms:
        log(f"\n=== {name} ===")
        members = build_members(position, act, exposed, builder, args)
        if not members:
            log("  no members -- skipped")
            continue
        # Partition by the key the design actually proposes. Analysing a whole
        # arm as one cluster measured something no implementation would do: the
        # Aqueduct arm's public secondary reveals stay in the key, so those
        # states would never share a cluster to begin with.
        groups = defaultdict(list)
        for member in members:
            groups[member["cluster_key"]].append(member)
        log(f"  {len(members)} member(s) -> {len(groups)} cluster(s) by info-state key")
        clusters = []
        for key, group in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            summary = analyse(group, evaluator, args, log)
            summary["cluster_key"] = key
            clusters.append(summary)
            log(f"    {key} n={summary['members']} "
                f"agree={summary['policy']['ref_best_agreement']} "
                f"illegal={summary['policy']['cluster_action_illegal_in']} "
                f"regret_max={summary['reference_regret']['max']}")
        multi = [c for c in clusters if c["members"] > 1]
        report["arms"][name] = {
            "members": len(members),
            "clusters": len(clusters),
            "singleton_clusters": len(clusters) - len(multi),
            "largest_cluster": max((c["members"] for c in clusters), default=0),
            "per_cluster": clusters,
            "note": (
                "a singleton cluster shares nothing and proves nothing; only "
                "multi-member clusters bear on whether aliasing is sound"
            ),
        }

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
