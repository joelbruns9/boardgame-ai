"""
Pool rescue (2026-10-03): can this player complete pool City Plans at all, and
are they worth it, if they are kept alive?

The pool check (2026-10-03, 768 positions) found the learner does not protect
pool plans -- it plays plan-killing placements more often than their share of
legal moves -- and that option A made its value head indifferent, because its
playouts never complete a pool plan. Before building features or discovery for
pool plans, this measures whether they are reachable and valuable for this
player when protected by rule.

Two arms on the same deals, one frozen checkpoint, T = 0, seat 0 searching and
the others on argmax policy (the hygiene-rescue setup):

    normal   no assistance
    pool     the learner's decisions follow the pool rule all game

THE POOL RULE (learner only, while it has a live pool plan -- "all pools in two
streets", "parks + pools in one street", "complete a street"):

* **Writes.** If a legal write lets the learner build a pool (or, for
  park+pool and complete-street plans, a park) in a street the plan still
  needs, without killing any live pool plan, play it -- pools first, then the
  most placement capacity. Otherwise, if the chosen write would kill a live
  pool plan, replace it with a write that does not: same card and delta first,
  then same card, then any; ties by capacity.
* **Pool / park prompts.** With an uncompleted pool plan dealt, never pass on
  building a pool or park: it is free points and consumes nothing.

Diagnostic only: a hand-written rule never becomes a training signal.

    python -m games.welcome_to.pool_rescue --checkpoint <ckpt> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

from games.welcome_to import action_codec as codec
from games.welcome_to import hygiene_rescue as hr
from games.welcome_to import macro_codec as mc
from games.welcome_to import placement_assist as pa
from games.welcome_to import plans as pl
from games.welcome_to import self_play, snapshot
from games.welcome_to.constants import NUM_STREETS
from games.welcome_to.game import GameState, Phase

ARMS: tuple[str, ...] = ("normal", "pool")
POOL_PLAN_IDS: frozenset[int] = frozenset(
    p.id
    for p in pl.PLANS
    if p.kind is pl.PlanKind.COMPLETE_STREET
    or (p.kind is pl.PlanKind.DECORATIVE and p.params[0] != "park")
)
_BUILD_POOL = mc.from_primitive(codec.A_POOL_BUILD)
_PASS_POOL = mc.from_primitive(codec.A_PASS_POOL)
_PASS_PARK = mc.from_primitive(codec.A_PASS_PARK)


def _alive_pool_boxes(sheet, x: int) -> int:
    return pl._pool_boxes_alive(sheet, x)


def needed_streets(state: GameState, seat: int = 0) -> tuple[set[int], set[int], list[int]]:
    """(streets needing pools, streets needing parks, live pool-plan slots)."""
    sheet = state.sheets[seat]
    pools, parks, live = set(), set(), []
    for slot, plan_id in enumerate(state.plan_ids):
        plan = pl.PLANS[plan_id]
        if plan_id not in POOL_PLAN_IDS or seat in state.plan_turns[slot]:
            continue
        if not pl.feasible(plan, sheet):
            continue
        live.append(slot)
        complete_pools = sheet.street_pools_complete()
        complete_parks = sheet.street_parks_complete()
        if plan.kind is pl.PlanKind.DECORATIVE and plan.params[0] == "pool":
            streets = [x for x in range(NUM_STREETS) if sheet.pools[x] + _alive_pool_boxes(sheet, x) >= 3]
            pools |= {x for x in streets if not complete_pools[x]}
        elif plan.kind is pl.PlanKind.DECORATIVE:  # pool&park in street params[1]
            x = plan.params[1]
            if not complete_pools[x]:
                pools.add(x)
            if not complete_parks[x]:
                parks.add(x)
        else:  # COMPLETE_STREET: any street that can still finish pools
            for x in range(NUM_STREETS):
                if sheet.pools[x] + _alive_pool_boxes(sheet, x) >= 3:
                    if not complete_pools[x]:
                        pools.add(x)
                    if not complete_parks[x]:
                        parks.add(x)
    return pools, parks, live


def _resolved(state: GameState, macro: int) -> tuple[GameState, Optional[str]]:
    """State after ``macro`` and, if it opens a pool/park prompt, after taking
    it -- so a progress move is not misread as a kill. Returns which prompt."""
    after = mc.step_macro(state, macro)
    if after.phase is Phase.ACTION_POOL and after._pool_available():
        after.apply(codec.A_POOL_BUILD)
        return after, "pool"
    if after.phase is Phase.ACTION_PARK and after._park_streets():
        after.apply(codec.park_street(after._park_streets()[0]))
        return after, "park"
    return after, None


def _kills(state: GameState, after: GameState, slots: list[int]) -> bool:
    sheet = after.sheets[state.actor]
    return any(not pl.feasible(pl.PLANS[state.plan_ids[s]], sheet) for s in slots)


def pool_choice(state: GameState, choice: int) -> int:
    """The pool rule for the acting seat (see module docstring)."""
    # Prompts first. At a pool/park prompt the house is already written but the
    # pool or park is not, so the street can look unable to finish; building is
    # free points and never consumes anything, so with any uncompleted pool
    # plan dealt to this seat, never pass.
    if state.phase in (Phase.ACTION_POOL, Phase.ACTION_PARK) and choice in (_PASS_POOL, _PASS_PARK):
        pending = any(
            pid in POOL_PLAN_IDS and state.actor not in state.plan_turns[slot]
            for slot, pid in enumerate(state.plan_ids)
        )
        if pending and state.phase is Phase.ACTION_POOL and state._pool_available():
            return _BUILD_POOL
        if pending and state.phase is Phase.ACTION_PARK and state._park_streets():
            return mc.from_primitive(codec.park_street(state._park_streets()[0]))
        return choice
    pools, parks, live = needed_streets(state, state.actor)
    if not live:
        return choice
    if not pa.is_write(choice):
        return choice
    writes = [m for m in mc.legal_macros(state) if pa.is_write(m)]
    resolved = {m: _resolved(state, m) for m in writes}
    safe = [m for m in writes if not _kills(state, resolved[m][0], live)]

    def street_of(m):
        return mc.decode_macro_write(m)[2]

    def capacity(m):
        return pa.hygiene_key(state, m)

    progress = [
        m
        for m in safe
        if (resolved[m][1] == "pool" and street_of(m) in pools)
        or (resolved[m][1] == "park" and street_of(m) in parks)
    ]
    if progress:
        return max(progress, key=lambda m: (resolved[m][1] == "pool", capacity(m), -m))
    if choice in safe or not safe:
        return choice
    slot, delta, _x, _y = mc.decode_macro_write(choice)
    for keep in (
        lambda m: mc.decode_macro_write(m)[:2] == (slot, delta),
        lambda m: mc.decode_macro_write(m)[0] == slot,
        lambda m: True,
    ):
        options = [m for m in safe if keep(m)]
        if options:
            return max(options, key=lambda m: (capacity(m), -m))
    return choice


class PoolAssistant:
    """``move_override`` applying the pool rule to seat 0, all game."""

    def __init__(self) -> None:
        self.decisions = 0
        self.changed = 0

    def __call__(self, rust_state, choice: int) -> int:
        if int(rust_state.actor) != 0:
            return choice
        state = snapshot.from_snapshot(rust_state.snapshot())
        prompt = state.phase in (Phase.ACTION_POOL, Phase.ACTION_PARK)
        if not prompt and not needed_streets(state, 0)[2]:
            return choice
        picked = pool_choice(state, choice)
        self.decisions += 1
        self.changed += picked != choice
        return picked


def plan_rows(trajectory: self_play.SelfPlayTrajectory) -> dict:
    """Per-game plan facts beyond hygiene_rescue.game_metrics."""
    state = trajectory.new_python_state()
    for action in trajectory.actions:
        mc.apply_macro(state, action)
    pool_slots = [s for s, pid in enumerate(state.plan_ids) if pid in POOL_PLAN_IDS]
    return {
        "has_pool_plan": bool(pool_slots),
        "pool_plan_done": float(any(0 in state.plan_turns[s] for s in pool_slots)),
        "learner_all_three": float(all(0 in state.plan_turns[s] for s in range(3))),
        "learner_pools": float(sum(state.sheets[0].pools)),
        "learner_pool_points": float(state.score_breakdown(0).pools),
        "plan_ids": list(state.plan_ids),
    }


def run(checkpoint, out, *, games=600, simulations=200, seed=995_000, inflight=256, workers=8, device="cuda", load=None) -> dict:
    from games.welcome_to import s2_promotion, s2_train

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    net = (load or (lambda path: s2_train.load_training_checkpoint(path, device)[0]))(checkpoint)
    config = self_play.SelfPlayConfig(
        games=games, inflight=min(inflight, games), max_batch=min(inflight, games),
        scheduler_workers=workers, seed=seed,
        opening_temperature_turns=0, opening_temperature=0.0, late_temperature=0.0,
    )
    search = s2_promotion.gate_search_config(simulations)
    opponents = (self_play.Opponent("checkpoint_policy", net),)
    per_arm = {}
    results = {"checkpoint": str(Path(checkpoint).resolve()), "games": games, "arms": {}}
    for name in ARMS:
        path, meta = out / f"{name}.jsonl", out / f"{name}.meta.json"
        if path.exists() and meta.exists():
            trajectories = [self_play.SelfPlayTrajectory.from_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            info = json.loads(meta.read_text(encoding="utf-8"))
        else:
            assistant = PoolAssistant()
            started = time.perf_counter()
            trajectories, _ = self_play.generate(
                net, config=config, search_config=search, opponents=opponents, device=device,
                move_override=None if name == "normal" else assistant,
            )
            info = {"seconds": time.perf_counter() - started, "assisted_decisions": assistant.decisions, "changed_decisions": assistant.changed}
            path.write_text("".join(t.to_json() + "\n" for t in trajectories), encoding="utf-8")
            meta.write_text(json.dumps(info, indent=2), encoding="utf-8")
        trajectories = sorted(trajectories, key=lambda t: t.seed)
        rows = [{**hr.game_metrics(t), **plan_rows(t)} for t in trajectories]
        per_arm[name] = rows
        pool_rows = [r for r in rows if r["has_pool_plan"]]
        results["arms"][name] = {
            **info,
            "pool_plan_games": len(pool_rows),
            "pool_games": hr.summarise([{k: v for k, v in r.items() if isinstance(v, float)} for r in pool_rows]),
            "all_games": hr.summarise([{k: v for k, v in r.items() if isinstance(v, float)} for r in rows]),
        }
        # completion per pool plan id
        by_plan = {}
        for r in pool_rows:
            for slot, pid in enumerate(r["plan_ids"]):
                if pid in POOL_PLAN_IDS:
                    by_plan.setdefault(pid, []).append(r["pool_plan_done"])
        results["arms"][name]["pool_plan_completion"] = {str(pid): sum(v) / len(v) for pid, v in by_plan.items()}
    normal = [r for r in per_arm["normal"] if r["has_pool_plan"]]
    pool = [r for r in per_arm["pool"] if r["has_pool_plan"]]
    names = sorted(k for k, v in normal[0].items() if isinstance(v, float))
    results["pool_vs_normal_on_pool_deals"] = {n: hr.paired_delta(pool, normal, n) for n in names}
    results["pool_vs_normal_all_deals"] = {n: hr.paired_delta(per_arm["pool"], per_arm["normal"], n) for n in names}
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--games", type=int, default=600)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=995_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    results = run(args.checkpoint, args.out, games=args.games, simulations=args.simulations, seed=args.seed, device=args.device)
    deltas = results["pool_vs_normal_on_pool_deals"]
    print(f"pool-plan deals: {results['arms']['pool']['pool_plan_games']}; "
          f"changed {results['arms']['pool']['changed_decisions']}/{results['arms']['pool']['assisted_decisions']} learner decisions")
    for metric in ("pool_plan_done", "learner_plans", "learner_all_three", "learner_score", "learner_margin",
                   "learner_pool_points", "learner_refusals", "end_turn"):
        d = deltas.get(metric)
        if d:
            print(f"  {metric:24s} {d['mean']:+.3f} [{d['lower']:+.3f}, {d['upper']:+.3f}]")
    for name in ARMS:
        print(name, "pool-plan completion by id:", results["arms"][name]["pool_plan_completion"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
