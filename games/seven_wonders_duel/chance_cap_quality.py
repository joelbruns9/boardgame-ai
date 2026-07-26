"""Chance-cap approximation quality (CHANCE_ENUMERATION_PLAN.md Step 4).

Throughput is the reason to cap chance enumeration; approximation quality is
what decides whether capping is allowed to ship. This measures the second on the
real self-play position distribution, at two levels:

**Level A -- edge Q error, analytic.** For every pure double card-reveal edge on
a sampled root, evaluate the WHOLE outcome space once with the net, then compare
the exact probability-weighted Q against the balanced ``n * X`` support's Q. No
search, no seed noise: this is the approximation error by itself, broken down by
pool size and by same-/mixed-back signature. The same pass measures catastrophe
coverage -- terminal children dropped, and how much of the worst-case tail the
retained support still sees.

**Level B -- decision impact, through the real searcher.** Run the production
Rust search at the same roots with and without the cap and compare what the
search actually decides: selected action, policy-target KL, visited-action
survivors.

Level B needs a control and reports one. A capped edge draws one uniform per
descent where an uncapped edge draws one per reveal, so the two runs' RNG
streams diverge even where the approximation is irrelevant. Re-running the
UNCAPPED search under a different seed measures how much disagreement that
divergence alone produces; capped-vs-exhaustive numbers mean nothing except
against that floor.

Usage::

    python -m games.seven_wonders_duel.chance_cap_quality \\
        --buffer runs/laptop_training_10h_02/buffer_final.jsonl \\
        --checkpoint runs/laptop_training_10h_02/checkpoints/latest.pt \\
        --roots 200 --offsets 1 2 3 --out chance_cap_quality.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

from .codec import decode_action, legal_action_indices
from .encoder import encode
from .engine import apply_action
from .game import ChanceKind, Phase
from .pool import enumerate_card_reveal, unseen_pool
from .search import (
    balanced_double_reveal_chains,
    chance_signature,
    enumerate_chains,
    state_actor,
)

CHEAP_SIMS_MAX = 24
"""Only cheap moves are capped (Step 3), so only cheap roots are measured."""


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------


def iter_roots(buffer_path: Path, games: int, stride: int):
    """(seed, first_player, prefix) for cheap searched decisions in the buffer.

    Bot moves and full-search moves are skipped: the first are not searched at
    all, the second keep exhaustive chance by design."""

    with buffer_path.open(encoding="utf-8") as handle:
        for game_index, line in enumerate(handle):
            if game_index >= games:
                return
            record = json.loads(line)
            setup = record["setup"]
            prefix: list[int] = []
            for move in record["moves"]:
                sims = move.get("sims", 0)
                if 0 < sims <= CHEAP_SIMS_MAX and len(prefix) % stride == 0:
                    yield setup["seed"], setup["first_player"], list(prefix)
                prefix.append(move["action"])


def root_seed(search_seed: int, root_index: int, replicate: int = 0) -> int:
    """A distinct deterministic seed per root (and per paired replicate).

    Reusing one seed for every root, as the first version did, correlates the
    offset draws and the Gumbel stream across the whole corpus -- unlike
    self-play, where every move gets a fresh seed. Both arms of a comparison
    still share the seed, so the pairing survives."""

    mixed = (search_seed * 0x9E3779B1 + root_index * 0x85EBCA77 + replicate * 0xC2B2AE3D)
    return mixed & ((1 << 63) - 1)


def pure_double_reveal_edges(state):
    """(action_index, specs) for every legal action whose chance signature is
    exactly two card reveals -- the only edges the cap touches."""

    out = []
    for index in legal_action_indices(state):
        specs = chance_signature(state, decode_action(state, index))
        if len(specs) == 2 and all(
            spec.kind is ChanceKind.CARD_REVEAL for spec in specs
        ):
            out.append((index, specs))
    return out


# --------------------------------------------------------------------------
# Level A -- exact edge-Q error and catastrophe coverage
# --------------------------------------------------------------------------


def child_values_p0(evaluator, states) -> list[float]:
    """Net value (player-0 relative) for each state, in one batched call."""

    values: list[float] = [0.0] * len(states)
    pending, slots = [], []
    for index, state in enumerate(states):
        if state.phase is Phase.COMPLETE:
            values[index] = (
                0.0 if state.winner is None else (1.0 if state.winner == 0 else -1.0)
            )
            continue
        pending.append(state)
        slots.append(index)
    if pending:
        actors = [state_actor(state) for state in pending]
        evaluations = evaluator.evaluate(
            [encode(state.observation(actor)) for state, actor in zip(pending, actors)],
            [legal_action_indices(state) for state in pending],
        )
        for slot, actor, evaluation in zip(slots, actors, evaluations):
            value_actor = float(evaluation.wdl[0] - evaluation.wdl[2])
            values[slot] = value_actor if actor == 0 else -value_actor
    return values


def every_support(names, offsets):
    """Every offset subset the construction could draw, as lists of keys.

    A single seed-derived realization measures one draw, not the estimator. With
    `n <= 11` the whole space is `C(n-1, X)` — at most 45 subsets for X=2 — so
    the exact distribution is affordable once the children are evaluated, and no
    seed choice can flatter or slander the result."""

    n = len(names)
    modulus = n - 1
    if offsets <= 0 or offsets >= modulus:
        return []
    supports = []
    for chosen in combinations(range(modulus), offsets):
        supports.append(
            [
                (names[i], names[(i + 1 + offset) % n])
                for i in range(n)
                for offset in chosen
            ]
        )
    return supports


def measure_edge(state, index, specs, evaluator, offsets_list, search_seed):
    """One edge: exact Q over the full outcome space against (a) the support the
    production seed would draw and (b) EVERY support the construction can draw."""

    chains = enumerate_chains(state, specs)
    children = []
    for outcomes, _, _ in chains:
        clone = state.clone()
        clone.search_barrier = True
        apply_action(clone, decode_action(clone, index), chance_outcomes=outcomes)
        children.append(clone)
    values = child_values_p0(evaluator, children)
    by_key = {key: value for (_, _, key), value in zip(chains, values)}
    exact_q = sum(
        probability * value for (_, probability, _), value in zip(chains, values)
    )

    actor = state_actor(state)
    sign = 1.0 if actor == 0 else -1.0
    # The catastrophe is the outcome that hurts the player who is choosing.
    worst = min(sign * value for value in values)
    terminal_keys = {
        key for (_, _, key), child in zip(chains, children) if child.phase is Phase.COMPLETE
    }

    same_back = specs[0].context[1] == specs[1].context[1]
    row = {
        "outcomes": len(chains),
        "same_back": same_back,
        "pool": int(round(len(chains) ** 0.5)) + 1 if same_back else None,
        "exact_q": exact_q,
        "terminal_children": len(terminal_keys),
        "caps": {},
    }
    names = [name for name, _ in enumerate_card_reveal(
        unseen_pool(state.observation(state.active_player)), specs[0].context[1]
    )]
    for offsets in offsets_list:
        support = balanced_double_reveal_chains(state, specs, offsets, search_seed)
        if support is None:  # the cap does not bite here; exhaustive is kept
            continue
        keys = [key for _, _, key in support]
        weight = support[0][1]
        capped_q = sum(weight * by_key[key] for key in keys)
        retained_worst = min(sign * by_key[key] for key in keys)

        # The estimator's whole distribution, not one draw.
        errors, gaps, missed_terminals = [], [], 0
        for pairs in every_support(names, offsets):
            support_weight = 1.0 / len(pairs)
            error = sum(support_weight * by_key[key] for key in pairs) - exact_q
            errors.append(error)
            gaps.append(min(sign * by_key[key] for key in pairs) - worst)
            missed_terminals += len(terminal_keys - set(pairs))

        row["caps"][offsets] = {
            "retained": len(keys),
            "q_error": capped_q - exact_q,
            "terminal_retained": len(terminal_keys & set(keys)),
            # How much of the worst case the retained support still sees. 0.0
            # means the true catastrophe is in the support.
            "worst_gap": retained_worst - worst,
            # Over ALL supports the construction can draw:
            "supports": len(errors),
            "mean_abs_error": mean(abs(e) for e in errors) if errors else 0.0,
            "signed_bias": mean(errors) if errors else 0.0,
            "worst_support_error": max((abs(e) for e in errors), default=0.0),
            "all_errors": errors,
            "worst_gap_mean_over_supports": mean(gaps) if gaps else 0.0,
            "worst_case_covered_fraction": (
                sum(1 for gap in gaps if gap == 0.0) / len(gaps) if gaps else 0.0
            ),
            "terminal_drops_over_supports": missed_terminals,
        }
    return row


# --------------------------------------------------------------------------
# Level B -- decision impact through the production searcher
# --------------------------------------------------------------------------


def kl_divergence(p, q) -> float:
    total = 0.0
    for left, right in zip(p, q):
        if left > 0.0:
            total += left * math.log(left / max(right, 1e-12))
    return total


def l1_distance(p, q) -> float:
    return sum(abs(left - right) for left, right in zip(p, q))


def js_divergence(p, q) -> float:
    mid = [(left + right) / 2.0 for left, right in zip(p, q)]
    return 0.5 * kl_divergence(p, mid) + 0.5 * kl_divergence(q, mid)


def sigma(completed, c_visit: float, c_scale: float, max_visits: int):
    """`GumbelMCTS._sigma` / `tree.rs::sigma_vector`: the completed Q values are
    MIN-MAX NORMALISED before scaling.

    This is the step the review request got wrong. Sigma is
    `scale * (q - low) / span`, so a Q perturbation lands as
    `scale * dq / span` — divided by the completed-Q span, not multiplied by
    scale alone. On a root where every action is nearly tied, span is small and
    the same absolute Q error produces a much larger logit perturbation. That is
    why the span, the endpoints, and dsigma are measured here rather than
    argued."""

    low, high = min(completed), max(completed)
    span = max(high - low, 1e-8)
    scale = (c_visit + max_visits) * c_scale
    return [scale * (q - low) / span for q in completed], low, high, span


def sigma_diagnostics(baseline, other, *, c_visit=50.0, c_scale=0.1) -> dict:
    """How far the cap moved the Gumbel logits, and whether it moved an endpoint
    of the normalisation window (which rescales EVERY action's sigma)."""

    qa, qb = list(baseline["completed_q"]), list(other["completed_q"])
    if not qa or len(qa) != len(qb):
        return {}
    max_visits = max(max(baseline["visits"], default=0), max(other["visits"], default=0))
    sa, low_a, high_a, span_a = sigma(qa, c_visit, c_scale, max_visits)
    sb, _, _, span_b = sigma(qb, c_visit, c_scale, max_visits)
    deltas = [abs(x - y) for x, y in zip(sa, sb)]
    # Logit margin between the best two actions, baseline side: how much slack a
    # sigma perturbation has before it can reorder the top of the policy.
    logits = sorted(sa, reverse=True)
    margin = logits[0] - logits[1] if len(logits) > 1 else float("inf")
    return {
        "q_span": span_a,
        "q_span_capped": span_b,
        "max_delta_sigma": max(deltas),
        "rms_delta_sigma": math.sqrt(sum(d * d for d in deltas) / len(deltas)),
        "endpoint_moved": float(
            qa.index(low_a) != qb.index(min(qb)) or qa.index(high_a) != qb.index(max(qb))
        ),
        "logit_margin": margin,
        "delta_sigma_exceeds_margin": float(max(deltas) > margin),
    }


def compare_search(baseline, other) -> dict:
    policy_a, policy_b = list(baseline["policy"]), list(other["policy"])
    best_a = max(range(len(policy_a)), key=policy_a.__getitem__)
    best_b = max(range(len(policy_b)), key=policy_b.__getitem__)
    visited_a = {i for i, v in enumerate(baseline["visits"]) if v > 0}
    visited_b = {i for i, v in enumerate(other["visits"]) if v > 0}
    union = visited_a | visited_b
    # What a disagreement COSTS, judged by the exhaustive run's own completed Q:
    # picking a different action matters only if that action is worse. A
    # disagreement rate on its own cannot distinguish a blunder from a coin
    # flip between two moves the baseline rates identically.
    completed = list(baseline["completed_q"])
    regret = completed[best_a] - completed[best_b] if completed else float("nan")

    # Per-ROUND survivors. "Visited at all" cannot detect an elimination: a
    # candidate visited in round 1 keeps visits > 0 whatever happens later, so
    # the visited-set Jaccard below is ~always 1.0 and is kept only as a floor.
    rounds_a = [set(r) for r in baseline.get("survivors", [])]
    rounds_b = [set(r) for r in other.get("survivors", [])]
    round_changes = sum(1 for x, y in zip(rounds_a, rounds_b) if x != y)
    final_changed = float(
        bool(rounds_a) and bool(rounds_b) and rounds_a[-1] != rounds_b[-1]
    )
    return {
        "action_disagreement": float(best_a != best_b),
        "action_regret": regret,
        "policy_kl": kl_divergence(policy_a, policy_b),
        "policy_l1": l1_distance(policy_a, policy_b),
        "policy_js": js_divergence(policy_a, policy_b),
        "survivor_rounds": len(rounds_a),
        "survivor_round_changes": round_changes,
        "survivor_any_round_changed": float(round_changes > 0),
        "survivor_final_changed": final_changed,
        "survivor_jaccard": (
            len(visited_a & visited_b) / len(union) if union else 1.0
        ),
        "topk_identical": float(list(baseline["topk"]) == list(other["topk"])),
        **sigma_diagnostics(baseline, other),
    }


def run_searches(adapter, games, seeds, sims, top_k, offsets, batch_cap):
    import seven_wonders_rust as swr

    return swr.search_many_flat_net(
        adapter,
        games,
        seeds,
        batch_cap,
        1,
        sims,
        top_k,
        force=True,
        double_reveal_offsets=offsets,
    )


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def percentile(values, q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    at = q * (len(ordered) - 1)
    low, high = math.floor(at), math.ceil(at)
    if low == high:
        return ordered[int(at)]
    return ordered[low] * (high - at) + ordered[high] * (at - low)


def summarise_level_a(rows, offsets_list) -> dict:
    summary = {}
    for offsets in offsets_list:
        errors, gaps, retained, dropped_terminals, by_pool, by_back = (
            [],
            [],
            [],
            0,
            defaultdict(list),
            defaultdict(list),
        )
        for row in rows:
            cap = row["caps"].get(offsets)
            if cap is None:
                continue
            error = abs(cap["q_error"])
            errors.append(error)
            gaps.append(cap["worst_gap"])
            retained.append(cap["retained"] / row["outcomes"])
            dropped_terminals += row["terminal_children"] - cap["terminal_retained"]
            by_pool[row["pool"]].append(error)
            by_back["same" if row["same_back"] else "mixed"].append(error)
        if not errors:
            continue
        # Over EVERY support the construction can draw, not just the one the
        # production seed happened to pick.
        all_errors = [
            abs(e)
            for row in rows
            if offsets in row["caps"]
            for e in row["caps"][offsets]["all_errors"]
        ]
        signed = [
            row["caps"][offsets]["signed_bias"] for row in rows if offsets in row["caps"]
        ]
        worst_per_edge = [
            row["caps"][offsets]["worst_support_error"]
            for row in rows
            if offsets in row["caps"]
        ]
        covered = [
            row["caps"][offsets]["worst_case_covered_fraction"]
            for row in rows
            if offsets in row["caps"]
        ]
        summary[offsets] = {
            "edges": len(errors),
            # --- the support the production seed drew (one realization) -------
            "q_mae": mean(errors),
            "q_p95": percentile(errors, 0.95),
            "q_max": max(errors),
            "retained_fraction": mean(retained),
            "worst_gap_mean": mean(gaps),
            "worst_case_covered": sum(1 for gap in gaps if gap == 0.0) / len(gaps),
            "terminal_children_dropped": dropped_terminals,
            "q_mae_by_pool": {
                str(pool): mean(values) for pool, values in sorted(by_pool.items(), key=lambda kv: (kv[0] is None, kv[0]))
            },
            "q_mae_by_back": {key: mean(values) for key, values in by_back.items()},
            # --- over the estimator's WHOLE support distribution -------------
            "supports_enumerated": len(all_errors),
            "all_q_mae": mean(all_errors) if all_errors else 0.0,
            "all_q_p95": percentile(all_errors, 0.95),
            "all_q_p99": percentile(all_errors, 0.99),
            "all_q_max": max(all_errors, default=0.0),
            "signed_bias_mean": mean(signed) if signed else 0.0,
            "signed_bias_max_abs": max((abs(v) for v in signed), default=0.0),
            "worst_support_error_mean": mean(worst_per_edge) if worst_per_edge else 0.0,
            "worst_support_error_max": max(worst_per_edge, default=0.0),
            "worst_case_covered_over_supports": mean(covered) if covered else 0.0,
            "terminal_drops_over_supports": sum(
                row["caps"][offsets]["terminal_drops_over_supports"]
                for row in rows
                if offsets in row["caps"]
            ),
        }
    return summary


def summarise_level_b(comparisons) -> dict:
    out = {}
    for label, rows in comparisons.items():
        if not rows:
            continue
        regrets = [
            r["action_regret"] for r in rows if r["action_disagreement"] and not math.isnan(r["action_regret"])
        ]
        kls = [r["policy_kl"] for r in rows]
        deltas = [r["max_delta_sigma"] for r in rows if "max_delta_sigma" in r]
        spans = [r["q_span"] for r in rows if "q_span" in r]
        margins = [r["logit_margin"] for r in rows if "logit_margin" in r]
        out[label] = {
            "positions": len(rows),
            "action_disagreement": mean(r["action_disagreement"] for r in rows),
            "regret_when_disagreeing_mean": mean(regrets) if regrets else 0.0,
            "regret_when_disagreeing_p95": percentile(regrets, 0.95),
            "policy_kl_mean": mean(kls),
            "policy_kl_p95": percentile(kls, 0.95),
            "policy_kl_p99": percentile(kls, 0.99),
            "policy_kl_max": max(kls),
            "policy_l1_mean": mean(r["policy_l1"] for r in rows),
            "policy_l1_p95": percentile([r["policy_l1"] for r in rows], 0.95),
            "policy_js_mean": mean(r["policy_js"] for r in rows),
            "policy_js_p95": percentile([r["policy_js"] for r in rows], 0.95),
            # Sequential-halving eliminations, per round -- the visited-set
            # Jaccard below cannot see these (see `compare_search`).
            "survivor_any_round_changed": mean(
                r["survivor_any_round_changed"] for r in rows
            ),
            "survivor_final_changed": mean(r["survivor_final_changed"] for r in rows),
            "survivor_round_changes_mean": mean(
                r["survivor_round_changes"] for r in rows
            ),
            "survivor_jaccard": mean(r["survivor_jaccard"] for r in rows),
            "topk_identical": mean(r["topk_identical"] for r in rows),
            # Sigma is scale * (q - low) / span, so the perturbation depends on
            # the completed-Q span, not on scale alone.
            "q_span_mean": mean(spans) if spans else 0.0,
            "q_span_p05": percentile(spans, 0.05),
            "q_span_min": min(spans, default=0.0),
            "max_delta_sigma_mean": mean(deltas) if deltas else 0.0,
            "max_delta_sigma_p95": percentile(deltas, 0.95),
            "max_delta_sigma_max": max(deltas, default=0.0),
            "rms_delta_sigma_mean": mean(r["rms_delta_sigma"] for r in rows if "rms_delta_sigma" in r)
            if deltas
            else 0.0,
            "endpoint_moved": mean(r["endpoint_moved"] for r in rows if "endpoint_moved" in r)
            if deltas
            else 0.0,
            "logit_margin_mean": mean(margins) if margins else 0.0,
            "delta_sigma_exceeds_margin": mean(
                r["delta_sigma_exceeds_margin"] for r in rows if "delta_sigma_exceeds_margin" in r
            )
            if deltas
            else 0.0,
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=150)
    parser.add_argument("--roots", type=int, default=200)
    parser.add_argument(
        "--stride", type=int, default=3, help="sample every Nth cheap decision"
    )
    parser.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sims", type=int, default=20, help="cheap-move budget")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--search-seed", type=int, default=20260725)
    parser.add_argument("--control-seed", type=int, default=20260726)
    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
        help="paired seed replicates per root: both arms share each seed, so "
        "pairing survives while tail sensitivity to the offset draw shows up",
    )
    parser.add_argument("--batch-cap", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    from .phase_e import load_evaluator
    from .rust_bridge import rust_flat_batch_adapter, rust_game_from_prefix

    evaluator = load_evaluator(args.checkpoint, args.device)

    rows, rust_games, skipped = [], [], 0
    for seed, first_player, prefix in iter_roots(
        Path(args.buffer), args.games, args.stride
    ):
        if len(rust_games) >= args.roots:
            break
        state, rust_game = rust_game_from_prefix(seed, first_player, prefix)
        edges = pure_double_reveal_edges(state)
        if not edges:
            skipped += 1
            continue
        # Every root gets its OWN seed, as it would in self-play.
        seed_here = root_seed(args.search_seed, len(rust_games))
        rows.extend(
            measure_edge(state, index, specs, evaluator, args.offsets, seed_here)
            for index, specs in edges
        )
        rust_games.append(rust_game)

    report = {
        "corpus": {
            "buffer": args.buffer,
            "checkpoint": args.checkpoint,
            "roots_with_double_reveal": len(rust_games),
            "roots_without": skipped,
            "edges": len(rows),
            # Different-back edges are deliberately left exhaustive, so they
            # appear here but never in the Level A caps.
            "edges_same_back": sum(1 for row in rows if row["same_back"]),
            "edges_mixed_back": sum(1 for row in rows if not row["same_back"]),
            "sims": args.sims,
        },
        "level_a_edge_q": summarise_level_a(rows, args.offsets),
    }

    if not args.skip_search and rust_games:
        adapter = rust_flat_batch_adapter(evaluator)
        comparisons: dict[str, list] = defaultdict(list)
        forced_rows: dict[str, int] = defaultdict(int)
        for replicate in range(max(1, args.replicates)):
            # One seed per root per replicate; both arms share it, so a
            # comparison never confounds the cap with a different Gumbel stream.
            seeds = [
                root_seed(args.search_seed, index, replicate)
                for index in range(len(rust_games))
            ]
            control_seeds = [
                root_seed(args.control_seed, index, replicate)
                for index in range(len(rust_games))
            ]
            baseline = run_searches(
                adapter, rust_games, seeds, args.sims, args.top_k, 0, args.batch_cap
            )
            control = run_searches(
                adapter, rust_games, control_seeds, args.sims, args.top_k, 0, args.batch_cap
            )
            comparisons["control_seed_only"].extend(
                compare_search(a, b) for a, b in zip(baseline, control)
            )
            forced_rows["0"] += sum(r["nn_work"]["forced_rows"] for r in baseline)
            for offsets in args.offsets:
                capped = run_searches(
                    adapter, rust_games, seeds, args.sims, args.top_k, offsets, args.batch_cap
                )
                comparisons[f"offsets_{offsets}"].extend(
                    compare_search(a, b) for a, b in zip(baseline, capped)
                )
                forced_rows[str(offsets)] += sum(
                    r["nn_work"]["forced_rows"] for r in capped
                )
                # Not every root has an edge the cap can bite (small pools
                # fall back), but the arm as a whole must be capping something.
                assert any(r["nn_work"]["fixed_support_edges"] > 0 for r in capped), (
                    f"offsets={offsets} capped nothing"
                )
            assert all(r["nn_work"]["fixed_support_edges"] == 0 for r in baseline), (
                "uncapped arm must hold no approximate support"
            )
        report["replicates"] = max(1, args.replicates)
        report["forced_rows"] = dict(forced_rows)
        report["level_b_decisions"] = summarise_level_b(comparisons)

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True), "utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
