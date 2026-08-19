"""Do cheap moves break the plans full-search moves start? (plan §B2)

A full-search move's row carries the GAME OUTCOME as its value label. If that
search sees a three-move plan and plays move 1, but moves 2 and 3 are cheap and
never see it, the plan dies and the outcome records that the plan was bad. The
row that did the right thing is punished for what the following cheap moves
failed to do. That is label contamination, and it is separate from the diversity
question it is usually confused with.

The bias runs in the worst possible direction. Cheap search is prior-guided, not
random, so it reliably executes the plans the prior ALREADY knows and drops the
ones it does not -- exactly the plans worth learning. The mechanism
preferentially suppresses novel strategy while leaving known strategy intact.

This measures it without a training run. Games are played normally under the
production cheap/full schedule; at every CHEAP move a full search is also run,
its answer recorded, and then **discarded** -- the cheap move is still what gets
played, so the distribution of positions is the one production actually visits.
Running the full answer instead would measure a different game.

Four numbers, in increasing order of how much they should move a decision:

1. **Disagreement rate.** How often cheap and full pick different actions. High
   agreement (~95%) means plans rarely break and none of §B2's fixes are worth
   buying.
2. **Disagreement on LOW-PRIOR moves.** The cases where the full search is
   discovering something the net does not already believe -- the only cases where
   a broken chain costs anything. Disagreement concentrated here confirms the
   mechanism; disagreement on moves the prior already liked does not.
3. **KL(cheap || full)** at the same position, as a magnitude rather than a
   count: two searches can pick the same action while disagreeing markedly about
   everything else.
4. **Provably-losing move rate**, at positions the endgame solver settles. This
   is ground truth rather than mutual agreement, and it is the only one of the
   four that says how much strategy the cheap path actually *drops* rather than
   how often the two differ.

Read it as a gate on how much to spend, not as a yes/no.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import torch

from .buffer import new_game
from .codec import decode_action, legal_action_indices
from .engine import apply_action
from .game import Phase
from .inference import Evaluator
from .search import GumbelMCTS, SearchConfig
from .train import load_checkpoint, model_from_config


def _kl(p: dict[int, float], q: dict[int, float]) -> float:
    """KL(p || q) over the shared action support.

    `q` is floored rather than skipped: an action the full search never visited
    has probability 0, and dropping those terms would report a smaller
    divergence exactly when the two searches disagree most.
    """

    total = 0.0
    for action, weight in p.items():
        if weight <= 0.0:
            continue
        total += weight * math.log(weight / max(q.get(action, 0.0), 1e-12))
    return total


def _prior_rank(prior: dict[int, float], action: int) -> float:
    """Where `action` sits in the prior, as a fraction (0 = the net's favourite).

    A rank, not a probability: "low-prior" has to mean the same thing at a
    position with 4 legal moves and one with 40, and a raw threshold on
    probability does not.
    """

    ordered = sorted(prior, key=lambda a: -prior.get(a, 0.0))
    if not ordered:
        return 0.0
    return ordered.index(action) / max(1, len(ordered) - 1)


def probe_game(
    seed: int,
    evaluator: Evaluator,
    *,
    cheap: SearchConfig,
    full: SearchConfig,
    full_fraction: float,
    solver_nodes: int,
    solver_secs: float,
    solver_max_cards: int,
) -> list[dict[str, Any]]:
    """One game, recording a cheap/full comparison at every cheap move."""

    from .rust_bridge import rust_game_from_state

    rng = torch.Generator().manual_seed(seed)
    game = new_game(seed, first_player=seed % 2)
    rows: list[dict[str, Any]] = []
    move_index = 0
    while game.phase is not Phase.COMPLETE:
        is_full = float(torch.rand(1, generator=rng)) < full_fraction
        move_seed = seed * 977 + move_index

        if is_full:
            # A full move is PLAYED with the full search, exactly as production
            # does. An earlier version ran the schedule but played the cheap
            # answer at every move, so the whole game was an all-cheap game and
            # the positions measured were not the ones production visits -- which
            # is the one property the probe's design turns on.
            result = GumbelMCTS(
                evaluator, dataclasses.replace(full, seed=move_seed)
            ).search(game)
        else:
            cheap_result = GumbelMCTS(
                evaluator, dataclasses.replace(cheap, seed=move_seed)
            ).search(game)
            result = cheap_result
            # The full search here is the counterfactual: recorded, compared, and
            # thrown away. Playing it would turn every move into a full move.
            #
            # ADJACENT seed, not the same one. The two searches therefore differ
            # in their noise as well as in their depth, so some disagreement is
            # sampling rather than mechanism and the rate is an OVER-estimate.
            # That is the safe direction for the decision this feeds -- a low
            # disagreement rate rules the concern out, a high one only fails to
            # rule it out -- but do not read the rate as an effect size. Common
            # random numbers (the same seed) would isolate depth, at the cost of
            # making cheap a strict prefix of full at Gumbel roots.
            full_result = GumbelMCTS(
                evaluator, dataclasses.replace(full, seed=move_seed + 1)
            ).search(game)
            prior = evaluator.evaluate_states([game])[0]
            legal = legal_action_indices(game)
            prior_map = {a: float(p) for a, p in zip(legal, prior.policy)}

            cheap_action = max(cheap_result.policy_target, key=cheap_result.policy_target.get)
            full_action = max(full_result.policy_target, key=full_result.policy_target.get)
            row = {
                "seed": seed,
                "move": move_index,
                "cards_left": sum(
                    1 for c in game.tableau.cards.values() if c.present
                ),
                "agree": cheap_action == full_action,
                "full_prior_rank": _prior_rank(prior_map, full_action),
                "kl_cheap_full": _kl(cheap_result.policy_target, full_result.policy_target),
            }

            # Ground truth where it exists. Cheap enough to ask only when the
            # solver can actually settle the position.
            if (
                game.phase is Phase.PLAY_AGE
                and game.age == 3
                and row["cards_left"] <= solver_max_cards
            ):
                answer = rust_game_from_state(game).solve_endgame(
                    solver_nodes, solver_secs, "exact", "star1"
                )
                # `exact_per_action` is a FLAG saying every action was priced on
                # a full window; `per_action_value` holds the values. Reading the
                # flag as the values is silent -- it is truthy -- so the guard
                # checks the flag and the lookup uses the map. A declined solve
                # carries the flag as False, so this also covers the refusal.
                if answer.get("exact_per_action"):
                    per_action = {
                        int(k): float(v) for k, v in answer["per_action_value"].items()
                    }
                    best = max(per_action.values())
                    # An action missing from an exact pricing is a contract
                    # violation, not a move that happens to be fine. Defaulting
                    # it to `best` records "not losing", which is the answer
                    # that hides the problem.
                    for name, action in (("cheap", cheap_action), ("full", full_action)):
                        if action not in per_action:
                            raise AssertionError(
                                f"exact solve did not price the {name} action "
                                f"{action}; per_action has {sorted(per_action)}"
                            )
                    row["cheap_loses"] = per_action[cheap_action] < best - 1e-9
                    row["full_loses"] = per_action[full_action] < best - 1e-9
            rows.append(row)

        played = max(result.policy_target, key=result.policy_target.get)
        apply_action(game, decode_action(game, played))
        move_index += 1
    return rows


def summarise(rows: list[dict], low_prior_rank: float) -> dict[str, Any]:
    if not rows:
        return {"cheap_moves": 0}
    disagreements = [row for row in rows if not row["agree"]]
    low_prior = [row for row in rows if row["full_prior_rank"] >= low_prior_rank]
    low_prior_disagree = [row for row in low_prior if not row["agree"]]
    solved = [row for row in rows if "cheap_loses" in row]
    kls = sorted(row["kl_cheap_full"] for row in rows)
    return {
        "cheap_moves": len(rows),
        "disagreement_rate": len(disagreements) / len(rows),
        "low_prior_moves": len(low_prior),
        # The number that decides whether the mechanism is real: disagreement
        # concentrated where the full search is finding something the net does
        # not already believe.
        "disagreement_rate_on_low_prior": (
            len(low_prior_disagree) / len(low_prior) if low_prior else float("nan")
        ),
        "kl_median": kls[len(kls) // 2],
        "kl_p90": kls[int(0.9 * len(kls))],
        "solved_positions": len(solved),
        "cheap_provably_losing": (
            sum(row["cheap_loses"] for row in solved) / len(solved) if solved else float("nan")
        ),
        "full_provably_losing": (
            sum(row["full_loses"] for row in solved) / len(solved) if solved else float("nan")
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=20260818)
    parser.add_argument("--cheap-sims", type=int, default=20)
    parser.add_argument("--full-sims", type=int, default=96)
    parser.add_argument("--full-search-fraction", type=float, default=0.25)
    parser.add_argument("--search-mode", default="puct", choices=["puct", "gumbel"])
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument(
        "--low-prior-rank",
        type=float,
        default=0.5,
        help="prior rank at or above which the full search's choice counts as "
        "something the net did not already believe (0 = the net's favourite)",
    )
    parser.add_argument("--solver-nodes", type=int, default=4_500_000)
    parser.add_argument("--solver-secs", type=float, default=30.0)
    parser.add_argument("--solver-max-cards", type=int, default=9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_config(payload.get("config", {}))
    load_checkpoint(args.checkpoint, model, checkpoint=payload)
    evaluator = Evaluator(model, args.device)

    root = "puct" if args.search_mode == "puct" else "gumbel"
    cheap = SearchConfig(sims=args.cheap_sims, top_k=args.top_k, root_selection=root)
    full = SearchConfig(sims=args.full_sims, top_k=args.top_k, root_selection=root)

    rows: list[dict] = []
    # Appended per game, not held to the end. This probe is hours long and has
    # twice been stopped part-way; the first time it lost 72 games of solves,
    # because the rows only existed in memory. A partial run is still a usable
    # sample -- the statistics are means over independent games -- so losing it
    # to a kill is pure waste.
    rows_path = args.out.with_suffix(".rows.jsonl") if args.out else None
    if rows_path is not None:
        rows_path.write_text("", encoding="utf-8")

    for index in range(args.games):
        fresh = probe_game(
            args.seed_base + index,
            evaluator,
            cheap=cheap,
            full=full,
            full_fraction=args.full_search_fraction,
            solver_nodes=args.solver_nodes,
            solver_secs=args.solver_secs,
            solver_max_cards=args.solver_max_cards,
        )
        rows.extend(fresh)
        if rows_path is not None:
            with rows_path.open("a", encoding="utf-8") as handle:
                for row in fresh:
                    handle.write(json.dumps(row) + "\n")
        # The running summary makes a stopped run readable without rerunning
        # anything, and shows the conditional rate settling (or not) as games
        # accumulate -- which is the number the decision turns on.
        if (index + 1) % 10 == 0 or index + 1 == args.games:
            running = summarise(rows, args.low_prior_rank)
            print(
                f"game {index + 1}/{args.games}: {len(rows)} cheap moves | "
                f"disagree {running['disagreement_rate']:.3f} | "
                f"low-prior {running['disagreement_rate_on_low_prior']:.3f} "
                f"(n={running['low_prior_moves']})",
                flush=True,
            )
        else:
            print(f"game {index + 1}/{args.games}: {len(rows)} cheap moves", flush=True)

    summary = summarise(rows, args.low_prior_rank)
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
