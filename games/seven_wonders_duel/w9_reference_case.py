"""The Workstream 9 reference case: chance fan-out on a chance-independent reply.

Baseline item 12 of ``WORLD_CLASS_MODEL_EVOLUTION_PLAN.md``. Reproduces the BGA
table ``908370787`` Age II decision on the frozen ``candidate_0085.pt`` and
emits one JSON artifact holding the three measurements every Workstream 9 gate
re-runs:

1. **Tree walk** -- the root edges at a fixed budget, and for one edge the
   per-chance-child breakdown: which card each world revealed, how many visits
   that world holds, its top reply, and where the tracked refutation sits inside
   it. This is the measurement that showed one strategic reply partitioned
   across ten statistically independent copies.
2. **Simulation ladder** -- the same single tree read at successive budgets, so
   the rank flip at the root and the refutation's per-world discovery are
   visible as functions of the budget rather than as one snapshot.
3. **Single-world probes** -- the same reply searched with the chance node
   collapsed to one known outcome, where no splitting occurs. The contrast
   between (1) and (3) is the whole finding: the reply is easy to find once, and
   near-impossible to find forty times.

Optionally (``--stages`` includes ``ref-values``) it also measures deep-search
reference values for each root action. Those are dedicated searches at the child
position repeated across chance worlds. **They are not solved game-theoretic
values** -- they are deep estimates, sufficient to expose a large advisor error
and nothing more.

The position is loaded from the passively captured BGA game log, mapped through
the same extractor and determinizer the live advisor uses, so this measures the
advisor's real input rather than a hand-built fixture.

Nothing here is a gate on its own. It produces the numbers a gate compares. Run
it once before touching the searcher, once after, and diff the summaries.

Usage
-----
Baseline (walk + ladder + probes), roughly the advisor's own budget::

    python -m games.seven_wonders_duel.w9_reference_case \\
        --out runs/seven_wonders_duel/w9_reference/baseline.json

Cheap smoke, to check plumbing before paying for the real budgets::

    python -m games.seven_wonders_duel.w9_reference_case --smoke --out /tmp/smoke.json

Everything, including the deep reference values (hours)::

    python -m games.seven_wonders_duel.w9_reference_case \\
        --stages walk,ladder,probes,ref-values \\
        --ladder 1000,2000,3000,45000 \\
        --out runs/seven_wonders_duel/w9_reference/full.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# The reference case's own coordinates. Defaults, not constants: the harness is
# meant to be pointed at other actor-created-threat positions as the corpus in
# the plan's baseline item grows.
# ---------------------------------------------------------------------------

DEFAULT_TABLE = "908370787"
DEFAULT_DECISION_ROW = 17
"""Age II, the human's move 31. Decision rows are the ``kind == "decision"``
subsequence of the log, which is what the plan's row number counts."""

DEFAULT_WALK_ACTION = "Discard for coins: Caravansery"
"""The played move. Removing ``Caravansery`` opens the cover chain
``r4c8 -> Caravansery -> r2c10 -> School`` and fires a CARD_REVEAL on r2c10, so
this is the edge whose reply is split across worlds."""

DEFAULT_TRACKED = "The Temple of Artemis"
"""The refutation. The opponent's unbuilt extra-turn Wonder takes r2c10 and
``School`` in a single turn, completing the science pair that yields
``Theology``. Its identity is public; only the buried card is not."""

EXPECTED_LEGAL = (
    "Build: Aqueduct",
    "Discard for coins: Caravansery",
    "Discard for coins: Aqueduct",
    "Wonder: Circus Maximus (using Caravansery)",
    "Wonder: Circus Maximus (using Aqueduct)",
)
"""Guards against a silently different position: a re-scrape, a re-indexed log,
or an extractor change would otherwise be measured as a search regression."""


def win_pct(q: float) -> float:
    """Actor-frame edge in [-1, 1] -> the percentage the advisor panel shows.

    ``extension_7wd/content.js`` renders ``(q + 1) / 2 * 100``; the plan's
    percentages are in that frame, so comparisons stay in it too.
    """

    return (q + 1.0) / 2.0 * 100.0


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Position:
    game: Any
    actor: int
    sign: float
    observation_sha256: str
    legal: list[dict]
    meta: dict


def load_position(
    log_path: Path, decision_row: int, resample_seed: int, *, strict: bool = True
) -> Position:
    """Rebuild the searchable state exactly as the live advisor would.

    ``wire_from_bga_payload`` -> ``observation_from_wire`` ->
    ``determinize_observation`` is the same chain ``advisor_adapter`` runs on a
    scrape, so what comes back here is the advisor's real input, hidden
    information included only as a valid determinization.
    """

    from .advisor_adapter import _label, state_actor
    from .advisor_scrape import determinize_observation, observation_from_wire
    from .bga_extract import wire_from_bga_payload
    from .codec import decode_action, legal_action_indices
    from .game import Phase

    rows = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    decisions = [row for row in rows if row.get("kind") == "decision"]
    if not 0 <= decision_row < len(decisions):
        raise SystemExit(
            f"decision row {decision_row} out of range: {log_path.name} holds "
            f"{len(decisions)} decision rows"
        )

    payload = wire_from_bga_payload(decisions[decision_row]["state"])
    if "observation" not in payload:
        raise SystemExit("extractor returned no observation wire for this row")
    observation = payload["observation"]
    obs = observation_from_wire(observation)
    game = determinize_observation(
        obs,
        random.Random(resample_seed),
        unknown_burial_ages=tuple(
            int(age) for age in payload.get("unknown_burial_ages", ())
        ),
    )
    actor = state_actor(game)
    legal = [
        {"index": int(index), "label": _label(decode_action(game, index), game)}
        for index in legal_action_indices(game)
    ]
    labels = tuple(entry["label"] for entry in legal)
    if strict and labels != EXPECTED_LEGAL:
        raise SystemExit(
            "position does not match the reference case -- refusing to measure "
            "a different decision as if it were this one.\n"
            f"  expected: {EXPECTED_LEGAL}\n"
            f"  found:    {labels}\n"
            "Pass --no-verify-position to measure it anyway."
        )
    if game.phase is not Phase.PLAY_AGE:
        raise SystemExit(f"expected a PLAY_AGE decision, found {game.phase.name}")

    return Position(
        game=game,
        actor=actor,
        sign=1.0 if actor == 0 else -1.0,
        observation_sha256=hashlib.sha256(
            json.dumps(observation, sort_keys=True, default=str).encode()
        ).hexdigest(),
        legal=legal,
        meta={
            "log": str(log_path.relative_to(REPO_ROOT)),
            "decision_row": decision_row,
            "decision_rows_total": len(decisions),
            "age": int(game.age),
            "phase": game.phase.name,
            "actor": actor,
            "resample_seed": resample_seed,
            # The determinization only fills hidden slots; every number below is
            # a function of the public position plus the searcher's own chance
            # resampling, so the seed is recorded rather than swept.
            "determinization_note": (
                "hidden identities are one valid determinization; the closed "
                "searcher resamples chance itself and never reads them"
            ),
        },
    )


def resolve_action(position: Position, wanted: str) -> dict:
    """Find one legal action by exact label, else by unique substring."""

    exact = [entry for entry in position.legal if entry["label"] == wanted]
    if len(exact) == 1:
        return exact[0]
    partial = [entry for entry in position.legal if wanted in entry["label"]]
    if len(partial) == 1:
        return partial[0]
    raise SystemExit(
        f"{'ambiguous' if partial else 'no'} action for {wanted!r}; legal here: "
        + ", ".join(entry["label"] for entry in position.legal)
    )


# ---------------------------------------------------------------------------
# Tree reading
# ---------------------------------------------------------------------------


def root_ranking(root, sign: float, game) -> list[dict]:
    """The root edges as the advisor would rank them: by visits, descending."""

    from .advisor_adapter import _label
    from .codec import decode_action

    rows = [
        {
            "index": int(edge.action_index),
            "label": _label(decode_action(game, edge.action_index), game),
            "visits": int(edge.visits),
            "prior": round(float(edge.prior), 6),
            "q": round(sign * edge.q_p0, 6),
            "win_pct": round(win_pct(sign * edge.q_p0), 2),
            "probability_weighted": bool(edge.probability_weighted),
            "chance_children": len(edge.children),
        }
        for edge in root.edges
    ]
    rows.sort(key=lambda row: (-row["visits"], -row["q"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def revealed_slots(state, action_index: int) -> list:
    """Tableau slots the action uncovers -- the burial targets that matter.

    The refutation is the Wonder burying the slot this very move exposes: doing
    so removes the coverer AND buys the extra turn, so the terminal card falls
    in one turn. Burying any other card buys the turn and leaves the coverer
    standing, which hands the terminal card back. They are different moves and
    must not be summed.
    """

    from .codec import decode_action
    from .search import ChanceKind, chance_signature

    specs = chance_signature(state, decode_action(state, action_index))
    return [
        spec.context[0] for spec in specs if spec.kind is ChanceKind.CARD_REVEAL
    ]


def world_breakdown(
    edge, tracked: str, *, top_replies: int = 3, exact_slots=None
) -> dict:
    """Per-chance-child statistics for one root edge.

    Each key of ``edge.children`` is the chain of revealed card names, so a
    "world" is named by what its CARD_REVEAL turned up. Within a world the
    opponent's replies are ordinary action edges, and the tracked refutation is
    looked up among them by label. ``child.samples`` is the descent count --
    the visits that world actually received.
    """

    from .advisor_adapter import _label
    from .codec import decode_action

    worlds: list[dict] = []
    for key, child in edge.children.items():
        node = child.node
        reply_sign = 1.0 if node.actor == 0 else -1.0
        replies = [
            {
                "index": int(reply.action_index),
                "label": _label(decode_action(node.state, reply.action_index), node.state),
                "visits": int(reply.visits),
                "prior": round(float(reply.prior), 6),
                "q": round(reply_sign * reply.q_p0, 6),
            }
            for reply in node.edges
        ]
        replies.sort(key=lambda row: (-row["visits"], -row["q"]))
        matched = [row for row in replies if tracked in row["label"]]
        # Wonder construction is exposed as one action per burial target, so the
        # tracked Wonder is a GROUP of edges. The group total is reported for
        # mechanism 2's sake -- a healthy-looking total can still be many edges
        # of four visits each -- but it is NOT the refutation and must never be
        # read as one: only the variant burying a slot this move exposes
        # produces the extra-turn-then-terminal-card sequence. Splitting these
        # was a review finding of 2026-09-01; the earlier harness summed them
        # and reported the wrong action as "best variant" in 4 of 10 worlds.
        group_visits = sum(row["visits"] for row in matched)
        best = max(matched, key=lambda row: row["visits"], default=None)
        exact = None
        if exact_slots:
            wanted = {tuple(slot) for slot in exact_slots}
            for row in matched:
                action = decode_action(node.state, row["index"])
                if action.slot_id is not None and tuple(action.slot_id) in wanted:
                    exact = row
                    break
        worlds.append(
            {
                "revealed": list(key) if isinstance(key, tuple) else [key],
                "visits": int(child.samples),
                "probability": (
                    round(float(child.probability), 8)
                    if child.probability is not None
                    else None
                ),
                "node_visits": int(node.visits),
                "top_replies": replies[:top_replies],
                # THE measurement. Everything about promotion, funding and
                # revision should be read from here, not from the group below.
                "refutation": {
                    "action": exact,
                    "examined": bool(exact and exact["visits"] > 0),
                    "is_top_reply": bool(
                        exact and replies and replies[0]["index"] == exact["index"]
                    ),
                    "rank": (
                        next(
                            (i for i, row in enumerate(replies, 1)
                             if row["index"] == exact["index"]),
                            None,
                        )
                        if exact else None
                    ),
                },
                # The Wonder group, for mechanism 2 only. Not the refutation.
                "tracked_group": {
                    "variants": len(matched),
                    "group_visits": group_visits,
                    "group_prior": round(sum(row["prior"] for row in matched), 6),
                    "best_variant": best,
                    "best_variant_is_refutation": bool(
                        exact and best and best["index"] == exact["index"]
                    ),
                },
            }
        )
    worlds.sort(key=lambda world: -world["visits"])

    examined = [world for world in worlds if world["refutation"]["examined"]]
    promoted = [world for world in worlds if world["refutation"]["is_top_reply"]]
    per_world = [world["visits"] for world in worlds]
    return {
        "edge_visits": int(edge.visits),
        "edge_prior": round(float(edge.prior), 6),
        "worlds": worlds,
        "rollup": {
            "world_count": len(worlds),
            "visits_per_world_mean": (
                round(statistics.fmean(per_world), 1) if per_world else 0.0
            ),
            "visits_per_world_min": min(per_world, default=0),
            "visits_per_world_max": max(per_world, default=0),
            # The refutation -- the Wonder burying a slot this move exposes.
            "worlds_examining_refutation": len(examined),
            "worlds_where_refutation_is_top": len(promoted),
            "refutation_visits_total": sum(
                (world["refutation"]["action"] or {}).get("visits", 0)
                for world in worlds
            ),
            "worlds_missing_refutation_action": sum(
                1 for world in worlds if world["refutation"]["action"] is None
            ),
            # The Wonder group, which is NOT the refutation. Retained for
            # mechanism 2, and to show how much of the group's funding went to
            # a strategically different burial target.
            "group_visits_total": sum(
                world["tracked_group"]["group_visits"] for world in worlds
            ),
            "worlds_whose_best_variant_is_wrong": sum(
                1 for world in worlds
                if world["tracked_group"]["best_variant"] is not None
                and not world["tracked_group"]["best_variant_is_refutation"]
            ),
            "variants_per_world": max(
                (world["tracked_group"]["variants"] for world in worlds), default=0
            ),
        },
    }


# ---------------------------------------------------------------------------
# Search construction
# ---------------------------------------------------------------------------


def build_mcts(evaluator, args):
    """The advisor's own Python search configuration.

    ``advisor_adapter._open_nn_search`` builds exactly this for
    ``search_impl='python'``: closed mode, PUCT descent under an expanded root,
    root chance force-expanded. Matching it is the point -- a harness that
    configured search differently would measure a different engine from the one
    that gave the advice.
    """

    from .search import GumbelMCTS, SearchConfig

    return GumbelMCTS(
        evaluator,
        SearchConfig(
            mode="closed",
            seed=int(args.seed),
            c_puct=float(args.c_puct),
            force_expand_root_chance=bool(args.force_expand_root_chance),
            wonder_group_selection=bool(args.wonder_group_selection),
            chance_sibling_bias=float(args.chance_sibling_bias),
            chance_sibling_bias_cap=float(args.chance_sibling_bias_cap),
            chance_sibling_bias_positive_only=bool(
                args.chance_sibling_bias_positive_only
            ),
        ),
    )


ARMS = {
    # The plan's five-arm diagnostic, minus the open-loop control: that arm is
    # a different searcher (`mode="open"`) rather than a flag, and it is
    # included as its own arm below so the mechanisms are never confounded with
    # an architecture change.
    "closed": {},
    "closed+sibling": {"chance_sibling_bias": 1.0},
    "closed+wonder": {"wonder_group_selection": True},
    "closed+both": {"chance_sibling_bias": 1.0, "wonder_group_selection": True},
    # The original signed formulation, kept as an arm so the positive-only
    # clamp's effect is attributable rather than bundled into the default.
    "closed+sibling-signed": {
        "chance_sibling_bias": 1.0,
        "chance_sibling_bias_positive_only": False,
    },
}


def load_evaluator_for(args):
    from .phase_e import load_evaluator

    path = Path(args.checkpoint)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}")
    return load_evaluator(str(path), args.device, migrate=bool(args.allow_migration)), path


def checkpoint_fingerprint(path: Path) -> dict:
    from .encoder import ENCODER_SIGNATURE

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {
        "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "sha256": digest.hexdigest(),
        "live_encoder_signature": ENCODER_SIGNATURE,
    }


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def run_ladder_and_walk(position: Position, evaluator, args, log) -> dict:
    """Crank ONE tree and read it at each rung.

    A tree per rung would be a different sample at every budget and the rank
    flip would be confounded with seed noise. The advisor deepens one tree, so
    the ladder does too, and each rung is a strictly deeper read of the same
    search.
    """

    walk_action = resolve_action(position, args.walk_action)
    # The slots this move exposes. Only a Wonder burying one of these is the
    # refutation; the same Wonder burying anything else is a different move.
    exact_slots = revealed_slots(position.game, walk_action["index"])
    rungs = sorted({int(rung) for rung in args.ladder})
    mcts = build_mcts(evaluator, args)

    started = time.perf_counter()
    root = mcts.make_root(position.game)
    expand_seconds = time.perf_counter() - started
    log(
        f"root expanded in {expand_seconds:.1f}s "
        f"({mcts.closed_nodes_created} nodes; force_expand="
        f"{bool(args.force_expand_root_chance)})"
    )

    walk_edge = next(
        edge for edge in root.edges if edge.action_index == walk_action["index"]
    )

    ladder: list[dict] = []
    tree_walk: dict | None = None
    done = 0
    for rung in rungs:
        rung_started = time.perf_counter()
        while done < rung:
            mcts.descend(root)
            done += 1
        elapsed = time.perf_counter() - rung_started
        ranking = root_ranking(root, position.sign, position.game)
        breakdown = world_breakdown(walk_edge, args.tracked, exact_slots=exact_slots)
        ladder.append(
            {
                "sims": done,
                "seconds": round(elapsed, 2),
                "root_value": round(position.sign * root.value_p0, 6),
                "root_win_pct": round(win_pct(position.sign * root.value_p0), 2),
                "ranking": ranking,
                "walk_edge_rollup": breakdown["rollup"],
            }
        )
        top = ranking[0]
        log(
            f"  {done:>6} sims  {elapsed:>6.1f}s  top={top['label']!r} "
            f"({top['visits']} visits, {top['win_pct']:.1f}%)  "
            f"refutation top in {breakdown['rollup']['worlds_where_refutation_is_top']}"
            f"/{breakdown['rollup']['world_count']} worlds"
        )
        if args.walk_at is not None and done >= int(args.walk_at) and tree_walk is None:
            tree_walk = {
                "sims": done,
                "action": walk_action,
                "tracked": args.tracked,
                **breakdown,
            }
    if tree_walk is None:  # --walk-at above every rung, or unset: use the last
        tree_walk = {
            "sims": done,
            "action": walk_action,
            "tracked": args.tracked,
            **world_breakdown(walk_edge, args.tracked, exact_slots=exact_slots),
        }

    return {
        "ladder": ladder,
        "tree_walk": tree_walk,
        "nodes_created": int(mcts.closed_nodes_created),
        "root_expand_seconds": round(expand_seconds, 2),
    }


def run_single_world_probes(position: Position, evaluator, args, log) -> dict:
    """Search the same reply with the chance node collapsed to one outcome.

    Each probe applies the walked action with an explicit CARD_REVEAL outcome
    and searches the resulting child directly, so the opponent's whole budget
    goes to one world. This is the control the tree walk is compared against:
    the same reply, the same network, the same budget, no partition.
    """

    from .codec import decode_action
    from .engine import apply_action
    from .search import chance_signature, enumerate_chains

    walk_action = resolve_action(position, args.walk_action)
    exact_slots = revealed_slots(position.game, walk_action["index"])
    action = decode_action(position.game, walk_action["index"])
    specs = chance_signature(position.game, action)
    if not specs:
        raise SystemExit(
            f"{walk_action['label']!r} fires no chance event; there is nothing "
            "to collapse and the probe would duplicate the tree walk"
        )
    chains = enumerate_chains(position.game, specs)
    log(f"{len(chains)} enumerable worlds behind {walk_action['label']!r}")

    # Deterministic subset: enumerate_chains is in canonical CARD_IDS order, so
    # an evenly spaced slice samples the pool without depending on the rng.
    wanted = min(int(args.probe_worlds), len(chains))
    step = max(1, len(chains) // wanted)
    selected = chains[:: step][:wanted]

    probes = []
    for outcomes, probability, key in selected:
        child = position.game.clone()
        child.search_barrier = True
        apply_action(child, decode_action(child, walk_action["index"]),
                     chance_outcomes=outcomes)
        mcts = build_mcts(evaluator, args)
        started = time.perf_counter()
        root = mcts.make_root(child)
        for _ in range(int(args.probe_sims)):
            mcts.descend(root)
        elapsed = time.perf_counter() - started
        sign = 1.0 if root.actor == 0 else -1.0
        ranking = root_ranking(root, sign, child)
        matched = [row for row in ranking if args.tracked in row["label"]]
        wanted = {tuple(slot) for slot in exact_slots}
        exact = None
        for row in matched:
            slot = decode_action(child, row["index"]).slot_id
            if slot is not None and tuple(slot) in wanted:
                exact = row
                break
        probes.append(
            {
                "revealed": list(key) if isinstance(key, tuple) else [key],
                "probability": round(float(probability), 8),
                "sims": int(args.probe_sims),
                "seconds": round(elapsed, 2),
                "top_reply": ranking[0] if ranking else None,
                "refutation": {
                    "action": exact,
                    "is_top_reply": bool(
                        exact and ranking and ranking[0]["index"] == exact["index"]
                    ),
                },
                "tracked_group": {
                    "variants": len(matched),
                    "group_visits": sum(row["visits"] for row in matched),
                },
            }
        )
        log(
            f"  world {probes[-1]['revealed']}: top={ranking[0]['label']!r} "
            f"({ranking[0]['visits']} visits) refutation="
            f"{(exact or {}).get('visits', 0)} visits"
        )

    found = [probe for probe in probes if probe["refutation"]["is_top_reply"]]
    visits = [
        (probe["refutation"]["action"] or {}).get("visits", 0) for probe in probes
    ]
    return {
        "action": walk_action,
        "tracked": args.tracked,
        "worlds_enumerable": len(chains),
        "worlds_probed": len(probes),
        "probes": probes,
        "rollup": {
            "worlds_where_refutation_is_top": len(found),
            "refutation_visits_min": min(visits, default=0),
            "refutation_visits_max": max(visits, default=0),
            "sims_per_probe": int(args.probe_sims),
        },
    }


def run_reference_values(position: Position, evaluator, args, log) -> dict:
    """Deep-search reference values for every root action.

    For each legal action, search the position it leads to, once per sampled
    chance world, and read the child's root value back in the DECIDING player's
    frame. Across worlds this gives the range the plan's cost-of-the-error table
    reports.

    These are deep estimates, not solved values. They are strong enough to show
    that one displayed number is ~20 points wrong and are not evidence of
    anything finer than that.
    """

    from .codec import decode_action
    from .engine import apply_action
    from .search import chance_signature, enumerate_chains

    results = []
    for entry in position.legal:
        action = decode_action(position.game, entry["index"])
        specs = chance_signature(position.game, action)
        if specs:
            chains = enumerate_chains(position.game, specs)
            wanted = min(int(args.ref_worlds), len(chains))
            step = max(1, len(chains) // wanted)
            selected = chains[::step][:wanted]
        else:
            selected = [([], 1.0, ())]

        values = []
        for outcomes, _probability, key in selected:
            child = position.game.clone()
            child.search_barrier = True
            apply_action(
                child,
                decode_action(child, entry["index"]),
                chance_outcomes=outcomes or None,
            )
            mcts = build_mcts(evaluator, args)
            started = time.perf_counter()
            root = mcts.make_root(child)
            for _ in range(int(args.ref_sims)):
                mcts.descend(root)
            # Read in the DECIDING player's frame, not the child actor's, so the
            # numbers are directly comparable across actions and to the panel.
            value = position.sign * root.value_p0
            values.append(
                {
                    "revealed": list(key) if isinstance(key, tuple) else [key],
                    "win_pct": round(win_pct(value), 2),
                    "seconds": round(time.perf_counter() - started, 2),
                }
            )
        pcts = [row["win_pct"] for row in values]
        results.append(
            {
                **entry,
                "worlds": values,
                "win_pct_min": round(min(pcts), 2),
                "win_pct_max": round(max(pcts), 2),
                "win_pct_mean": round(statistics.fmean(pcts), 2),
            }
        )
        log(
            f"  {entry['label']!r}: {results[-1]['win_pct_min']:.1f}-"
            f"{results[-1]['win_pct_max']:.1f}% over {len(values)} world(s)"
        )

    return {
        "sims_per_world": int(args.ref_sims),
        "worlds_per_action": int(args.ref_worlds),
        "caveat": (
            "deep-search estimates, not solved game-theoretic values; "
            "sufficient only to expose a large advisor error"
        ),
        "actions": sorted(results, key=lambda row: -row["win_pct_mean"]),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def build_summary(report: dict, args) -> dict:
    """The handful of numbers a Workstream 9 gate actually diffs."""

    summary: dict[str, Any] = {"tracked": args.tracked}

    ladder = report.get("ladder") or []
    if ladder:
        summary["root_rank_by_sims"] = {
            str(rung["sims"]): [row["label"] for row in rung["ranking"][:3]]
            for rung in ladder
        }
        walked = args.walk_action
        summary["walked_action_rank_by_sims"] = {
            str(rung["sims"]): next(
                (row["rank"] for row in rung["ranking"] if walked in row["label"]), None
            )
            for rung in ladder
        }
        summary["walked_action_win_pct_by_sims"] = {
            str(rung["sims"]): next(
                (row["win_pct"] for row in rung["ranking"] if walked in row["label"]),
                None,
            )
            for rung in ladder
        }
        # The headline Workstream 9 metric: how much budget it takes before the
        # refutation is the best reply in ANY world, and in half of them. `null`
        # means it never was inside the budget measured -- which is the finding.
        worlds_top = [
            (rung["sims"], rung["walk_edge_rollup"]["worlds_where_refutation_is_top"],
             rung["walk_edge_rollup"]["world_count"])
            for rung in ladder
        ]
        summary["sims_to_promote_refutation"] = {
            "in_any_world": next(
                (sims for sims, top, _ in worlds_top if top >= 1), None
            ),
            "in_half_of_worlds": next(
                (sims for sims, top, total in worlds_top if total and top * 2 >= total),
                None,
            ),
            "in_all_worlds": next(
                (sims for sims, top, total in worlds_top if total and top == total), None
            ),
            "max_sims_measured": max(sims for sims, _, _ in worlds_top),
        }

    walk = report.get("tree_walk")
    if walk:
        summary["tree_walk"] = {"sims": walk["sims"], **walk["rollup"]}

    probes = report.get("single_world_probes")
    if probes:
        summary["single_world"] = probes["rollup"]
        if walk:
            # The contrast that names the defect: the same reply, found readily
            # with one world's budget, missed in most worlds when partitioned.
            summary["partition_penalty"] = {
                "worlds_where_refutation_is_top_partitioned": (
                    f"{walk['rollup']['worlds_where_refutation_is_top']}"
                    f"/{walk['rollup']['world_count']}"
                ),
                "worlds_where_refutation_is_top_isolated": (
                    f"{probes['rollup']['worlds_where_refutation_is_top']}"
                    f"/{probes['worlds_probed']}"
                ),
                # Ten exact refutation edges, one per world, each competing
                # inside a two-way Wonder-target branch. NOT twenty copies of
                # the same reply -- the other branch is a different move.
                "exact_refutation_edges": walk["rollup"]["world_count"],
                "wonder_target_branching": max(
                    walk["rollup"]["variants_per_world"], 1
                ),
                "buckets_per_idea": (
                    walk["rollup"]["world_count"]
                    * max(walk["rollup"]["variants_per_world"], 1)
                ),
            }

    refs = report.get("reference_values")
    if refs:
        summary["reference_values"] = {
            row["label"]: f"{row['win_pct_min']:.1f}-{row['win_pct_max']:.1f}%"
            for row in refs["actions"]
        }

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    position = parser.add_argument_group("position")
    position.add_argument("--table", default=DEFAULT_TABLE)
    position.add_argument("--log-dir", default="runs/seven_wonders_duel/bga_game_log")
    position.add_argument("--decision-row", type=int, default=DEFAULT_DECISION_ROW)
    position.add_argument(
        "--resample-seed", type=int, default=0,
        help="determinization seed; hidden identities only (default: 0)",
    )
    position.add_argument(
        "--no-verify-position", dest="verify_position", action="store_false",
        help="skip the legal-action guard (use when pointing at another position)",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--checkpoint", default="extension_7wd/candidate_0085.pt")
    model.add_argument("--device", default="cpu")
    model.add_argument(
        "--allow-migration", action="store_true",
        help="serve a checkpoint whose encoder signature predates the live one",
    )

    search = parser.add_argument_group("search")
    search.add_argument("--seed", type=int, default=0)
    search.add_argument("--c-puct", type=float, default=1.5)
    search.add_argument(
        "--no-force-expand", dest="force_expand_root_chance", action="store_false",
        help="do not force-expand the root chance layer (the advisor does)",
    )
    search.add_argument(
        "--wonder-group-selection", action="store_true",
        help="Workstream 9 mechanism 2: pick the Wonder, then the burial target",
    )
    search.add_argument(
        "--chance-sibling-bias", type=float, default=0.0,
        help="Workstream 9 mechanism 1 coefficient; 0 is off (default: 0)",
    )
    search.add_argument("--chance-sibling-bias-cap", type=float, default=1.0)
    search.add_argument(
        "--signed-sibling-bias", dest="chance_sibling_bias_positive_only",
        action="store_false",
        help="allow a negative sibling advantage (the original formulation)",
    )
    search.add_argument(
        "--arm", default=None, choices=sorted(ARMS),
        help="preset flag combination; overrides the individual mechanism flags",
    )

    stages = parser.add_argument_group("stages")
    stages.add_argument(
        "--stages", default="walk,ladder,probes",
        help="comma-separated: walk, ladder, probes, ref-values "
             "(walk and ladder share one tree and are run together)",
    )
    stages.add_argument(
        "--ladder", default="1000,2000,3000",
        help="root sim counts to read the tree at (default: 1000,2000,3000)",
    )
    stages.add_argument(
        "--walk-at", type=int, default=3000,
        help="which rung carries the full per-world tree walk (default: 3000)",
    )
    stages.add_argument("--walk-action", default=DEFAULT_WALK_ACTION)
    stages.add_argument("--tracked", default=DEFAULT_TRACKED)
    stages.add_argument("--probe-sims", type=int, default=6000)
    stages.add_argument("--probe-worlds", type=int, default=6)
    stages.add_argument("--ref-sims", type=int, default=6000)
    stages.add_argument("--ref-worlds", type=int, default=3)
    stages.add_argument(
        "--smoke", action="store_true",
        help="tiny budgets for a plumbing check; the numbers mean nothing",
    )

    parser.add_argument(
        "--sweep-arms", default=None,
        help="comma-separated arms to run in sequence, each with its own report "
             f"(one of {sorted(ARMS)}, or 'all'). Writes <out stem>.<arm>.json "
             "per arm plus a comparison block. Ignores --arm.",
    )
    parser.add_argument("--out", default=None, help="write the full JSON report here")
    parser.add_argument(
        "--summary-out", default=None,
        help="write only the summary block here. The full report carries every "
             "per-world reply and is large enough to belong under runs/, which "
             "is gitignored; the summary is the small diffable part and is what "
             "a later search change is compared against.",
    )
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)
    if args.arm:
        args.wonder_group_selection = False
        args.chance_sibling_bias = 0.0
        args.chance_sibling_bias_positive_only = True
        for flag, value in ARMS[args.arm].items():
            setattr(args, flag, value)
    if args.smoke:
        args.ladder, args.walk_at = "40,80", 80
        args.probe_sims, args.probe_worlds = 60, 2
        args.ref_sims, args.ref_worlds = 60, 1
    args.ladder = [int(rung) for rung in str(args.ladder).split(",") if rung.strip()]
    args.stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    unknown = args.stages - {"walk", "ladder", "probes", "ref-values"}
    if unknown:
        raise SystemExit(f"unknown stage(s): {', '.join(sorted(unknown))}")
    return args


def run_one(position: Position, evaluator, fingerprint: dict, args, log) -> dict:
    """Every stage for ONE flag combination, as a complete report."""

    report: dict[str, Any] = {
        "harness": "w9_reference_case",
        "harness_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "position": {
            **position.meta,
            "table": args.table,
            "observation_sha256": position.observation_sha256,
            "legal": position.legal,
        },
        "checkpoint": fingerprint,
        "search": {
            "impl": "python",
            "mode": "closed",
            "root": "PUCT descent under an expanded root (the advisor's path)",
            "seed": args.seed,
            "c_puct": args.c_puct,
            "force_expand_root_chance": bool(args.force_expand_root_chance),
            "arm": args.arm or "custom",
            "wonder_group_selection": bool(args.wonder_group_selection),
            "chance_sibling_bias": float(args.chance_sibling_bias),
            "chance_sibling_bias_cap": float(args.chance_sibling_bias_cap),
            "chance_sibling_bias_positive_only": bool(
                args.chance_sibling_bias_positive_only
            ),
        },
        "stages_run": sorted(args.stages),
        "smoke": bool(args.smoke),
    }

    if args.stages & {"walk", "ladder"}:
        log("stage: tree walk + simulation ladder")
        report.update(run_ladder_and_walk(position, evaluator, args, log))
        if "ladder" not in args.stages:
            report.pop("ladder", None)
        if "walk" not in args.stages:
            report.pop("tree_walk", None)

    if "probes" in args.stages:
        log("stage: single-world probes")
        report["single_world_probes"] = run_single_world_probes(
            position, evaluator, args, log
        )

    if "ref-values" in args.stages:
        log("stage: deep-search reference values")
        report["reference_values"] = run_reference_values(position, evaluator, args, log)

    report["summary"] = build_summary(report, args)
    return report


def summary_of(report: dict) -> dict:
    """The small, diffable half: provenance plus the gate numbers."""

    return {
        "harness": report["harness"],
        "harness_version": report["harness_version"],
        "generated_at": report["generated_at"],
        "position": {
            key: report["position"][key]
            for key in ("table", "decision_row", "age", "observation_sha256")
        },
        "checkpoint": report["checkpoint"],
        "search": report["search"],
        "stages_run": report["stages_run"],
        "smoke": report["smoke"],
        "summary": report["summary"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = (lambda *_: None) if args.quiet else (
        lambda message: print(message, file=sys.stderr, flush=True)
    )

    log_path = REPO_ROOT / args.log_dir / f"table_{args.table}.jsonl"
    if not log_path.exists():
        raise SystemExit(f"game log not found: {log_path}")

    position = load_position(
        log_path, args.decision_row, args.resample_seed,
        strict=args.verify_position,
    )
    log(
        f"position: table {args.table} decision row {args.decision_row} -- Age "
        f"{position.meta['age']}, {len(position.legal)} legal actions, "
        f"actor {position.actor}"
    )

    evaluator, checkpoint_path = load_evaluator_for(args)
    fingerprint = checkpoint_fingerprint(checkpoint_path)
    log(f"checkpoint: {fingerprint['path']} sha256={fingerprint['sha256'][:16]}...")

    def write(target: str, obj: Any) -> None:
        path = Path(target)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
        log(f"wrote {path}")

    if not args.sweep_arms:
        report = run_one(position, evaluator, fingerprint, args, log)
        if args.out:
            write(args.out, report)
        elif not args.summary_out:
            print(json.dumps(report, indent=2))
        if args.summary_out:
            write(args.summary_out, summary_of(report))
        if not args.quiet:
            print(json.dumps(report["summary"], indent=2), file=sys.stderr)
        return 0

    # Sweep: one model load, one position, one report per arm. Sharing the
    # evaluator is what makes the arms comparable -- a per-arm reload would put
    # a different object (and, on some devices, different kernels) behind each.
    arms = sorted(ARMS) if args.sweep_arms == "all" else [
        arm.strip() for arm in args.sweep_arms.split(",") if arm.strip()
    ]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arm(s): {', '.join(sorted(unknown))}")

    comparison = {}
    for arm in arms:
        log(f"=== arm: {arm} ===")
        args.arm = arm
        args.wonder_group_selection = False
        args.chance_sibling_bias = 0.0
        args.chance_sibling_bias_positive_only = True
        for flag, value in ARMS[arm].items():
            setattr(args, flag, value)
        report = run_one(position, evaluator, fingerprint, args, log)
        comparison[arm] = report["summary"]
        if args.out:
            stem = Path(args.out)
            write(str(stem.with_suffix("")) + f".{arm}.json", report)

    payload = {
        "harness": "w9_reference_case",
        "harness_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "position": {
            "table": args.table,
            "decision_row": args.decision_row,
            "observation_sha256": position.observation_sha256,
        },
        "checkpoint": fingerprint,
        "smoke": bool(args.smoke),
        "arms": {arm: ARMS[arm] for arm in arms},
        "comparison": comparison,
    }
    if args.summary_out:
        write(args.summary_out, payload)
    elif not args.out:
        print(json.dumps(payload, indent=2))
    if not args.quiet:
        print(
            json.dumps(
                {
                    arm: {
                        "sims_to_promote": summary.get("sims_to_promote_refutation"),
                        "tree_walk": summary.get("tree_walk"),
                    }
                    for arm, summary in comparison.items()
                },
                indent=2,
            ),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
