"""
Hygiene rescue (reviewer test 1, built 2026-09-30): does fixing sheet hygiene
extend games and unlock City Plans?

Three arms play the **same deals** (seeds and seat counts) with one frozen
checkpoint, at temperature 0 and without root noise.  Seat 0 searches as in
generation; the other seats play the checkpoint's argmax policy.

    normal   no assistance                                   (reference)
    focal    seat 0's placements assisted through turn T
    all      every seat's placements assisted through turn T

**Assistance** reranks a placement *within the chosen card*: when a seat picks
``WRITE(slot, delta, box)``, it is replaced by the write of the same stack slot
-- same number and effect, any temp delta and box -- that leaves that seat the
most placement capacity, then the most span.  Ties keep the seat's own choice.
Refusals, roundabouts and every effect decision are untouched, and from turn
``T + 1`` on everyone plays normally.

The heuristic is a **diagnostic instrument**, never a training signal: it is
the "sheet hygiene" GreedyBot-style term the project does not train on.

Reading it (the reviewer's decision rule):

* ``all`` extends games and raises plan completion a lot -> hygiene is causally
  important, and the training levers that target it are worth more.
* games extend but plans stay rare -> plan pursuit is a separate bottleneck.
* ``focal`` vs ``all`` separates the seat's own sheet from game length set by
  the opponents' sheets (any seat's third refusal ends the game).

Paired deltas against ``normal`` are per-deal differences with a normal-approx
95% interval.

    python -m games.welcome_to.hygiene_rescue --checkpoint <ckpt> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from games.welcome_to import macro_codec as mc
from games.welcome_to import placement_assist
from games.welcome_to import plans as pl
from games.welcome_to import self_play, training
from games.welcome_to.game import GameState

ARMS: tuple[str, ...] = ("normal", "focal", "all")
DEFAULT_ASSIST_THROUGH = 16
#: Hygiene is read at the start of this turn, as in the 2026-09-26 diagnosis.
MEASURE_TURN = 16


# The rule lives in placement_assist, shared with S2 generation.
_is_write = placement_assist.is_write
hygiene_key = placement_assist.hygiene_key
assisted_choice = placement_assist.assisted_choice
Assistant = placement_assist.Assistant


def game_metrics(trajectory: self_play.SelfPlayTrajectory) -> dict[str, float]:
    """Per-deal outcomes, from a Python replay of the macro sequence."""
    state = GameState.new(
        seed=trajectory.engine_seed, config=trajectory.config, rng_kind=trajectory.rng
    )
    log = training.ReplayLog(state)
    hygiene: Optional[list[tuple[float, int]]] = None
    for action in trajectory.actions:
        if hygiene is None and state.turn == MEASURE_TURN:
            hygiene = []
            for sheet in state.sheets:
                empty = sum(1 for row in sheet.numbers for n in row if n is None)
                spans = sheet.box_spans()
                dead = sum(
                    1
                    for x, row in enumerate(sheet.numbers)
                    for y, n in enumerate(row)
                    if n is None and spans[x][y] == 0
                )
                hygiene.append((sum(sheet.placement_capacity()) / max(empty, 1), dead))
        log.observe(state, action)
        mc.apply_macro(state, action)
    history = log.history(state)
    seats = trajectory.players
    scores = state.scores()
    plans = [sum(1 for slot in state.plan_turns if seat in slot) for seat in range(seats)]
    out = {
        "end_turn": float(state.turn),
        "learner_score": float(scores[0]),
        "learner_margin": float(scores[0] - max(scores[1:])),
        "learner_plans": float(plans[0]),
        "learner_refusals": float(state.sheets[0].permits),
        "learner_forced_refusals": float(len(history.forced_refusals[0])),
        "learner_plan_deaths": float(sum(d is not None for d in history.plan_deaths[0])),
        "learner_estate_plans": float(
            sum(
                1
                for slot, plan_id in enumerate(state.plan_ids)
                if pl.PLANS[plan_id].kind is pl.PlanKind.ESTATE and 0 in state.plan_turns[slot]
            )
        ),
        "seat_plans": sum(plans) / seats,
        "seat_refusals": sum(sheet.permits for sheet in state.sheets) / seats,
        "seat_plan_deaths": sum(
            d is not None for deaths in history.plan_deaths for d in deaths
        )
        / seats,
        "plan_ending": float("completed all three plans" in (state.end_of_game_reason() or "")),
    }
    if hygiene is not None:
        out["learner_cap_per_empty_t16"] = hygiene[0][0]
        out["learner_dead_boxes_t16"] = float(hygiene[0][1])
        out["seat_cap_per_empty_t16"] = sum(h[0] for h in hygiene) / seats
        out["seat_dead_boxes_t16"] = sum(h[1] for h in hygiene) / seats
    return out


def paired_delta(
    arm: Sequence[dict[str, float]], reference: Sequence[dict[str, float]], name: str
) -> Optional[dict[str, float]]:
    """Mean per-deal difference ``arm - reference`` with a 95% interval, over
    deals where both arms have the metric."""
    diffs = [a[name] - b[name] for a, b in zip(arm, reference) if name in a and name in b]
    if len(diffs) < 2:
        return None
    mean = statistics.fmean(diffs)
    stderr = statistics.stdev(diffs) / math.sqrt(len(diffs))
    return {
        "mean": mean,
        "stderr": stderr,
        "lower": mean - 1.96 * stderr,
        "upper": mean + 1.96 * stderr,
        "n": float(len(diffs)),
    }


def summarise(per_game: Sequence[dict[str, float]]) -> dict[str, float]:
    names = sorted({name for game in per_game for name in game})
    return {
        name: statistics.fmean(game[name] for game in per_game if name in game)
        for name in names
    }


def run(
    checkpoint: str | Path,
    out: str | Path,
    *,
    games: int = 300,
    simulations: int = 200,
    assist_through: int = DEFAULT_ASSIST_THROUGH,
    seed: int = 990_000,
    inflight: int = 256,
    workers: int = 8,
    device: str = "cuda",
    arms: Sequence[str] = ARMS,
    load: Optional[Callable] = None,
) -> dict:
    """Play every arm (resumably: a finished arm's games are reused), then
    write ``results.json`` with per-arm means and paired deltas vs normal."""
    from games.welcome_to import s2_promotion, s2_train

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    net = (load or (lambda path: s2_train.load_training_checkpoint(path, device)[0]))(checkpoint)
    search = s2_promotion.gate_search_config(simulations)
    config = self_play.SelfPlayConfig(
        games=games,
        inflight=min(inflight, games),
        max_batch=min(inflight, games),
        scheduler_workers=workers,
        seed=seed,
        opening_temperature_turns=0,
        opening_temperature=0.0,
        late_temperature=0.0,
    )
    opponents = (self_play.Opponent("checkpoint_policy", net),)
    seats_for = {"normal": frozenset(), "focal": frozenset({0}), "all": None}

    results: dict = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "games": games,
        "simulations": simulations,
        "assist_through": assist_through,
        "seed": seed,
        "arms": {},
    }
    per_arm: dict[str, list[dict[str, float]]] = {}
    for name in arms:
        path = out / f"{name}.jsonl"
        meta = out / f"{name}.meta.json"
        if path.exists() and meta.exists():
            trajectories = [
                self_play.SelfPlayTrajectory.from_json(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            info = json.loads(meta.read_text(encoding="utf-8"))
        else:
            assistant = Assistant(seats_for[name], assist_through)
            started = time.perf_counter()
            trajectories, _ = self_play.generate(
                net,
                config=config,
                search_config=search,
                opponents=opponents,
                device=device,
                move_override=None if name == "normal" else assistant,
            )
            info = {
                "seconds": time.perf_counter() - started,
                "assisted_decisions": assistant.decisions,
                "changed_decisions": assistant.changed,
            }
            path.write_text(
                "".join(t.to_json() + "\n" for t in trajectories), encoding="utf-8"
            )
            meta.write_text(json.dumps(info, indent=2), encoding="utf-8")
        trajectories = sorted(trajectories, key=lambda t: t.seed)
        per_arm[name] = [game_metrics(t) for t in trajectories]
        results["arms"][name] = {**info, **summarise(per_arm[name])}
        print(f"{name}: {json.dumps(results['arms'][name], default=float)}", flush=True)

    if "normal" in per_arm:
        reference = per_arm["normal"]
        names = sorted({n for game in reference for n in game})
        for name in per_arm:
            if name == "normal":
                continue
            results["arms"][name]["vs_normal"] = {
                metric: paired_delta(per_arm[name], reference, metric) for metric in names
            }
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--assist-through", type=int, default=DEFAULT_ASSIST_THROUGH)
    parser.add_argument("--seed", type=int, default=990_000)
    parser.add_argument("--inflight", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    results = run(
        args.checkpoint,
        args.out,
        games=args.games,
        simulations=args.simulations,
        assist_through=args.assist_through,
        seed=args.seed,
        inflight=args.inflight,
        workers=args.workers,
        device=args.device,
    )
    for name, arm in results["arms"].items():
        deltas = arm.get("vs_normal")
        if not deltas:
            continue
        print(f"\n{name} vs normal (changed {arm['changed_decisions']}/{arm['assisted_decisions']} placements):")
        for metric in (
            "end_turn",
            "learner_plans",
            "seat_plans",
            "learner_estate_plans",
            "learner_plan_deaths",
            "learner_refusals",
            "learner_margin",
            "learner_cap_per_empty_t16",
            "seat_cap_per_empty_t16",
        ):
            d = deltas.get(metric)
            if d:
                print(f"  {metric:28s} {d['mean']:+.3f} [{d['lower']:+.3f}, {d['upper']:+.3f}]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
