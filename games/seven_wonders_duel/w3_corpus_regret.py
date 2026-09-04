"""Regret of each A/B arm on the threat corpus, against a COMMON reference.

The comparison `threat_corpus_measure` cannot make on its own. Run per arm, it
evaluates every action with that arm's own search, so two arms produce two
self-assessments on different scales: one says "Build: Library, 87.1%", the
other "Wonder: Appian Way, 84.3%", and nothing in either output says which is
right. A table of those reads like a comparison and is not one.

Regret needs one yardstick:

    reference pass   ONE model (the incumbent) evaluates EVERY legal action at
                     each position -- this is what `threat_corpus_measure`
                     already produces, and it is the expensive part.
    per arm          only the action the arm would PLAY. One search per
                     position, not a sweep over every action and world, so it
                     is roughly two orders of magnitude cheaper.
    regret           reference value of the reference's best action
                     minus the reference value of the arm's choice.

Every number then comes from the same evaluator, so arms are directly
comparable and a difference means something.

    python -m games.seven_wonders_duel.w3_corpus_regret \\
        --reference-dir runs/seven_wonders_duel/threat_corpus/w3_baseline \\
        --arm baseline=runs/.../w3_offline_ab/baseline_seed20260904.pt \\
        --arm inputs=runs/.../w3_offline_ab/inputs_seed20260904.pt

The reference is a model, not an oracle. It is the strongest evaluator we have,
not ground truth, so a small regret difference between arms is evidence about
agreement with the incumbent's search -- not proof either arm plays better.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def reference_positions(reference_dir: Path) -> list[dict]:
    """Every position the reference pass measured, with its action values."""

    out = []
    for path in sorted(reference_dir.glob("*.json")):
        if path.name.startswith("summary") or path.name == "triage_report.json":
            continue
        if "_recheck" in path.name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        actions = payload.get("reference_values", {}).get("actions")
        position = payload.get("position")
        if not actions or not position:
            continue
        out.append({
            "artifact": path.name,
            "log": position["log"],
            "decision_row": int(position["decision_row"]),
            "resample_seed": int(position.get("resample_seed", 0)),
            "checkpoint": payload.get("checkpoint", {}).get("path"),
            "values": {int(a["index"]): float(a["win_pct_weighted"]) for a in actions},
            "labels": {int(a["index"]): a["label"] for a in actions},
        })
    return out


def arm_choice(game, evaluator, sims: int, seed: int) -> int:
    """The action this arm would actually play. One search, not a sweep."""

    from .search import GumbelMCTS, SearchConfig

    searcher = GumbelMCTS(
        evaluator,
        SearchConfig(
            mode="closed",
            sims=sims,
            seed=seed,
            force_expand_root_chance=True,
        ),
    )
    return int(searcher.search(game).action_index)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reference-dir", required=True,
                        help="a threat_corpus_measure output directory: the "
                             "COMMON yardstick every arm is scored against")
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=PATH",
                        help="repeatable; the arm's checkpoint")
    parser.add_argument("--sims", type=int, default=600,
                        help="search budget for the arm's CHOICE. Fixed across "
                             "arms: a budget difference would be the effect.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out",
                        default="runs/seven_wonders_duel/w3_corpus_regret.json")
    args = parser.parse_args(argv)

    arms = {}
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        arms[name] = path
    if not arms:
        raise SystemExit("no --arm given")

    reference_dir = Path(args.reference_dir)
    if not reference_dir.is_absolute():
        reference_dir = REPO_ROOT / reference_dir
    positions = reference_positions(reference_dir)
    if args.limit:
        positions = positions[: args.limit]
    if not positions:
        raise SystemExit(f"no reference artifacts in {reference_dir}")

    refs = {p["checkpoint"] for p in positions}
    print(f"{len(positions)} reference positions from {reference_dir}")
    print(f"reference model(s): {refs}")
    if len(refs) > 1:
        print("WARNING: the reference pass mixes checkpoints; regret across "
              "positions is then measured against different yardsticks.")

    from .phase_e import load_evaluator
    from .w9_reference_case import load_position

    rows = []
    for name, path in arms.items():
        print(f"\n=== arm {name}: {path}")
        evaluator = load_evaluator(str(path), args.device, migrate=True)
        started = time.perf_counter()
        for index, position in enumerate(positions):
            log_path = Path(position["log"])
            if not log_path.is_absolute():
                log_path = REPO_ROOT / log_path
            # strict=False: the verifier compares against the ONE hardcoded
            # reference case and refuses anything else, which is right for that
            # harness and wrong for a corpus sweep.
            state = load_position(
                log_path, position["decision_row"], position["resample_seed"],
                strict=False,
            )
            game = getattr(state, "game", state)
            chosen = arm_choice(game, evaluator, args.sims, args.seed)
            values = position["values"]
            best = max(values.values())
            # An action the reference never scored cannot be given a regret.
            # Recorded rather than dropped: silently skipping them would bias
            # the mean toward the arms that stay inside the reference's set.
            got = values.get(chosen)
            rows.append({
                "arm": name,
                "artifact": position["artifact"],
                "chosen_index": chosen,
                "chosen_label": position["labels"].get(chosen),
                "reference_best": round(best, 3),
                "reference_value_of_choice": None if got is None else round(got, 3),
                "regret": None if got is None else round(best - got, 3),
                "agrees_with_reference_best": (
                    None if got is None else abs(best - got) < 1e-9
                ),
            })
            print(f"  [{index + 1}/{len(positions)}] {position['artifact']}: "
                  f"{position['labels'].get(chosen, chosen)} "
                  f"regret {rows[-1]['regret']}")
        print(f"  arm {name} took {(time.perf_counter() - started) / 60:.1f} min")

    report = {
        "harness": "w3_corpus_regret",
        "reference_dir": str(reference_dir),
        "reference_models": sorted(str(r) for r in refs),
        "sims": args.sims,
        "seed": args.seed,
        "arms": arms,
        "rows": rows,
        "note": (
            "Regret is measured against the reference model's own action "
            "values, so it reports agreement with the incumbent's search "
            "rather than ground truth. The reference is the strongest "
            "evaluator available, not an oracle."
        ),
    }
    summary = {}
    for row in rows:
        entry = summary.setdefault(
            row["arm"], {"n": 0, "scored": 0, "regret": 0.0, "agree": 0, "unscored": 0}
        )
        entry["n"] += 1
        if row["regret"] is None:
            entry["unscored"] += 1
            continue
        entry["scored"] += 1
        entry["regret"] += row["regret"]
        entry["agree"] += int(bool(row["agrees_with_reference_best"]))
    for name, entry in summary.items():
        entry["mean_regret"] = (
            round(entry["regret"] / entry["scored"], 3) if entry["scored"] else None
        )
        entry["agreement"] = (
            round(entry["agree"] / entry["scored"], 3) if entry["scored"] else None
        )
    report["summary"] = summary

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"{'arm':<12}{'positions':>10}{'scored':>8}{'mean regret':>13}{'agreement':>11}")
    print("-" * 66)
    for name, entry in summary.items():
        print(f"{name:<12}{entry['n']:>10}{entry['scored']:>8}"
              f"{entry['mean_regret']!s:>13}{entry['agreement']!s:>11}")
    print("\nLower regret is better. Both columns are agreement with the "
          "reference model's search, not with ground truth.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
