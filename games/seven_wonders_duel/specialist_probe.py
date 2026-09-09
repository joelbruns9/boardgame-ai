"""W7 S0a: does the leaf bias actually fund attacking continuations?

**S0 is two experiments, and this is the first of them.** With frozen weights it
answers utility correctness and attack *coverage*: does a biased search visit and
fund the attacking lines, or does the incumbent's prior keep excluding them?
That needs no training. Defence can only improve through training and held-out
evaluation, so S0b -- a small general-training A/B -- is a separate run and not
this script.

**Acceptance is not "attempted attacks went up".** An agent can attempt more
*unsound* rushes and teach the general nothing. This reports:

* ``moved`` -- how often the bias changed the move at all. If this is ~0 the
  mechanism did nothing and every downstream null is uninterpretable.
* ``credible`` -- of the moves it changed, how often the LAMBDA-ZERO search
  still rates the new move within ``--credible-gap`` of its own best. This is
  the plan's "attacks a lambda-zero search still rates as reasonable".
* ``q_cost`` -- the mean unbiased completed-Q the biased choice gives up.
  Ordinary playing strength that has not collapsed.
* ``prior_of_new_choice`` and ``visit_share_of_new_choice`` -- the two failure
  modes a null has to distinguish: the search never funded the move, or the
  prior never offered it.
* ``own_type_outlook`` -- the searched probability of the specialist's own
  victory type, biased against unbiased. This is the pursuit measurement, and it
  is read off the search's own seven-way outlook rather than inferred from
  realised endings.

A null here does NOT reject the workstream. Biasing an unchanged net can fail
because that net undervalues attacking lines, because it assigns attack moves
tiny priors, or because search cannot find their continuations inside the
budget -- all of which fine-tuning is meant to change. The three columns above
exist so a null says *which*.

Usage::

    python -m games.seven_wonders_duel.specialist_probe \\
        --checkpoint runs/cloud6/checkpoints/current_best.pt \\
        --buffer runs/cloud6/buffers/iter_0085.jsonl \\
        --lambda 0.5 --victory scientific --positions 200 --sims 256 \\
        --out probe_science_0.5.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
from typing import Any

VICTORY_INDEX = {"civilian": 0, "scientific": 1, "military": 2}


def _positions(records, wanted: int, seed: int, min_move: int):
    """Mid-game positions, one per game, in a deterministic order.

    One per game rather than many: consecutive plies of one game are nearly the
    same position, and a probe whose sample is dominated by a handful of games
    measures those games rather than the policy.
    """

    from .buffer import replay

    rng = random.Random(seed)
    picked: list[tuple[Any, int, int]] = []
    for record in records:
        # `sims > 0` excludes bot moves, which record none; `mode` excludes
        # them by name as well, since a curriculum game's rows are not the
        # policy under test.
        eligible = [
            move.i
            for move in record.moves
            if move.i >= min_move and move.sims > 0 and move.mode != "bot"
        ]
        if not eligible:
            continue
        target = rng.choice(eligible)
        captured: list[Any] = []

        def keep(game, move, target=target, captured=captured):
            if move.i == target:
                captured.append(game.clone())

        replay(record, on_state=keep)
        if captured:
            picked.append((captured[0], record.seed, target))
        if len(picked) >= wanted:
            break
    return picked


def _search(
    adapter, games, seeds, *, sims, top_k, bias, batch_cap, leaf_batch, puct_root
):
    import seven_wonders_rust as swr

    lam, victory, seats, symmetric = bias
    if lam <= 0.0:
        return swr.search_many_flat_net(
            adapter,
            games,
            seeds,
            batch_cap,
            leaf_batch,
            sims,
            top_k,
            puct_root=puct_root,
        )
    # The seat is the SEARCHER's, and it differs per position, so a biased sweep
    # is one call per seat rather than one call for everything. Grouping keeps
    # the batching that makes the flat boundary worth using.
    results: list[Any] = [None] * len(games)
    for seat in (0, 1):
        rows = [index for index, value in enumerate(seats) if value == seat]
        if not rows:
            continue
        batch = swr.search_many_flat_net(
            adapter,
            [games[index] for index in rows],
            [seeds[index] for index in rows],
            batch_cap,
            leaf_batch,
            sims,
            top_k,
            puct_root=puct_root,
            specialist_lambda=lam,
            specialist_victory=victory,
            specialist_seat=seat,
            specialist_symmetric=symmetric,
        )
        for index, row in zip(rows, batch):
            results[index] = row
    return results


def probe(
    checkpoint: str | Path,
    records,
    *,
    lam: float,
    victory: str,
    positions: int = 200,
    sims: int = 256,
    top_k: int = 8,
    seed: int = 20260908,
    credible_gap: float = 0.05,
    min_move: int = 8,
    device: str = "cpu",
    batch_cap: int = 256,
    leaf_batch: int = 8,
    root_selection: str = "puct",
    symmetric: bool = False,
) -> dict[str, Any]:
    import torch

    from .inference import Evaluator
    from .rust_bridge import rust_flat_batch_adapter, rust_game_from_state
    from .search import state_actor
    from .train import load_checkpoint, model_from_config

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = model_from_config(payload.get("config", {}))
    load_checkpoint(checkpoint, model, checkpoint=payload)
    model.to(device).eval()
    if getattr(model, "hier_value", None) is None:
        # The bias reads W4's seven-way outlook; without the head every leaf
        # would raise, and a probe that reports a null because the head is
        # missing is worse than no probe.
        raise SystemExit(
            "this checkpoint has no hierarchical value head, so it cannot "
            "supply the outlook a biased search reads"
        )
    evaluator = Evaluator(model, device, batch_cap)
    adapter = rust_flat_batch_adapter(evaluator)

    sampled = _positions(records, positions, seed, min_move)
    if not sampled:
        raise SystemExit("no eligible positions in the supplied records")
    games = [rust_game_from_state(game) for game, _seed, _move in sampled]
    seats = [state_actor(game) for game, _seed, _move in sampled]
    # One search seed per position, fixed, so the biased and unbiased searches
    # differ only in the bias.
    seeds = [
        (seed + 1_000_003 * game_seed + 17 * move_index) & ((1 << 63) - 1)
        for _game, game_seed, move_index in sampled
    ]

    if root_selection not in ("puct", "gumbel"):
        raise SystemExit("root_selection must be 'puct' or 'gumbel'")
    puct_root = root_selection == "puct"
    if puct_root and leaf_batch > 1:
        # A PUCT root with a wide leaf wave selects under virtual loss,
        # which is a different algorithm AT THE ROOT -- and the root's own
        # distribution is exactly what this probe measures. Batching still
        # happens ACROSS positions in one flat call, so the cost is small.
        leaf_batch = 1
    base = _search(
        adapter,
        games,
        seeds,
        sims=sims,
        top_k=top_k,
        bias=(0.0, victory, seats, symmetric),
        batch_cap=batch_cap,
        leaf_batch=leaf_batch,
        puct_root=puct_root,
    )
    biased = _search(
        adapter,
        games,
        seeds,
        sims=sims,
        top_k=top_k,
        bias=(lam, victory, seats, symmetric),
        batch_cap=batch_cap,
        leaf_batch=leaf_batch,
        puct_root=puct_root,
    )

    offset = VICTORY_INDEX[victory]
    rows: list[dict[str, Any]] = []
    for index, (state, game_seed, move_index) in enumerate(sampled):
        from .codec import legal_action_indices

        legal = legal_action_indices(state)
        b, x = base[index], biased[index]
        base_visits, biased_visits = list(b["visits"]), list(x["visits"])
        # THE SEARCH'S OWN ANSWER, not argmax visits.
        #
        # Under PUCT the two coincide, but under a Gumbel root they do not: the
        # returned action comes from the Gumbel candidate scores and the
        # deterministic policy from completed Q, and on 16 mock searches max
        # visits disagreed with the returned action 5 times and with the
        # policy-target argmax 11. Defining "the move" by visits therefore
        # measured a move the searched player would not play.
        base_best = legal.index(int(b["action"]))
        biased_best = legal.index(int(x["action"]))
        completed = list(b["completed_q"])
        best_q = max(completed) if completed else 0.0
        q_cost = best_q - completed[biased_best] if completed else 0.0
        # Credibility is only EVIDENCE where the lambda-zero search actually
        # looked. `completed_q` falls back to the root value for an unvisited
        # candidate, so counting that as "the unbiased search rates this move
        # fine" would report the absence of an opinion as an endorsement.
        verified = base_visits[biased_best] > 0
        total_biased = sum(biased_visits) or 1
        base_outlook = b.get("root_outlook")
        biased_outlook = x.get("root_outlook")
        rows.append(
            {
                "game_seed": game_seed,
                "move": move_index,
                "seat": seats[index],
                "moved": biased_best != base_best,
                "q_cost": q_cost,
                "credible": verified and q_cost <= credible_gap,
                "credibility_verified": verified,
                "prior_of_new_choice": list(b["prior"])[biased_best],
                "visit_share_of_new_choice": biased_visits[biased_best] / total_biased,
                "base_visit_share_of_new_choice": (
                    base_visits[biased_best] / (sum(base_visits) or 1)
                ),
                "root_value_base": b["root_value"],
                "root_value_biased": x["root_value"],
                "root_value_unshaped": x.get("root_value_unshaped"),
                "own_type_base": None if base_outlook is None else base_outlook[offset],
                "own_type_biased": (
                    None if biased_outlook is None else biased_outlook[offset]
                ),
                "legal": len(legal),
            }
        )

    moved = [row for row in rows if row["moved"]]

    def mean(values):
        values = [value for value in values if value is not None]
        return statistics.fmean(values) if values else None

    summary = {
        "positions": len(rows),
        "lambda": lam,
        "victory": victory,
        # WHICH FORM. `own` alone against `own - other`. Measured on this net,
        # the difference discriminates between sibling moves 2.6x better for
        # science and 2.1x better for military, so the same lambda has that
        # much more steering authority under the symmetric form. Reporting a
        # lambda without the form it was measured under is meaningless.
        "symmetric": symmetric,
        "sims": sims,
        "credible_gap": credible_gap,
        "moved_fraction": len(moved) / len(rows),
        # Of the moves the bias changed, how many an UNBIASED search still
        # rates as reasonable. This, not the move count, is the acceptance.
        "root_selection": root_selection,
        # Of the moves the bias changed, how many the lambda-zero search VISITED
        # and still rated within `credible_gap` of its own best.
        "credible_fraction_of_moved": (
            sum(1 for row in moved if row["credible"]) / len(moved) if moved else None
        ),
        # ... and how many it never looked at, where credibility is unknown
        # rather than absent. A large number here is itself the S0a finding:
        # the incumbent's search is not funding the attacking continuations.
        "unverified_fraction_of_moved": (
            sum(1 for row in moved if not row["credibility_verified"]) / len(moved)
            if moved
            else None
        ),
        "mean_q_cost_of_moved": mean(row["q_cost"] for row in moved),
        "max_q_cost_of_moved": max((row["q_cost"] for row in moved), default=None),
        # The two failure modes a null has to separate.
        "mean_prior_of_new_choice": mean(row["prior_of_new_choice"] for row in moved),
        "mean_visit_share_of_new_choice": mean(
            row["visit_share_of_new_choice"] for row in moved
        ),
        "mean_base_visit_share_of_new_choice": mean(
            row["base_visit_share_of_new_choice"] for row in moved
        ),
        # Pursuit, read off the search's own outlook rather than from realised
        # endings -- which a frozen-weights probe never reaches.
        "own_type_outlook_base": mean(row["own_type_base"] for row in rows),
        "own_type_outlook_biased": mean(row["own_type_biased"] for row in rows),
        "root_value_shift": mean(
            row["root_value_biased"] - row["root_value_base"] for row in rows
        ),
        # The unshaped root must stay a win probability whatever the bias did.
        "unshaped_root_within_unit_interval": all(
            row["root_value_unshaped"] is None
            or -1.0 <= row["root_value_unshaped"] <= 1.0
            for row in rows
        ),
    }
    return {"summary": summary, "rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--buffer", required=True, help="JSONL of GameRecords to draw positions from"
    )
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="bias on (own - opponent) rather than own alone. Measured on a "
        "cloud2-trained net, the difference separates sibling moves 2.6x "
        "better for science and 2.1x better for military, so a given lambda "
        "steers that much harder. In 7WD science the two are not rival "
        "strategies -- taking a symbol advances you AND denies them -- so this "
        "is a better-conditioned reading of the same intent, not a different "
        "one. Lambda is only meaningful together with this flag.",
    )
    parser.add_argument(
        "--victory", choices=sorted(VICTORY_INDEX), default="scientific"
    )
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument("--sims", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument(
        "--credible-gap",
        type=float,
        default=0.05,
        help="how much unbiased completed Q an attack may give up and still "
        "count as credible",
    )
    parser.add_argument("--min-move", type=int, default=8)
    parser.add_argument(
        "--root-selection",
        choices=("puct", "gumbel"),
        default="puct",
        help="root rule for BOTH searches. Defaults to puct because that is "
        "what self-play runs on the moves it records, so the probe measures "
        "the decision the training data would carry",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--leaf-batch", type=int, default=8)
    parser.add_argument("--batch-cap", type=int, default=256)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    from .buffer import read_records

    records = read_records(args.buffer)
    report = probe(
        args.checkpoint,
        records,
        lam=args.lam,
        victory=args.victory,
        positions=args.positions,
        sims=args.sims,
        top_k=args.top_k,
        seed=args.seed,
        credible_gap=args.credible_gap,
        min_move=args.min_move,
        device=args.device,
        batch_cap=args.batch_cap,
        leaf_batch=args.leaf_batch,
        symmetric=args.symmetric,
        root_selection=args.root_selection,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
