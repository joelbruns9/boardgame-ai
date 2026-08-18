"""What predicts whether an endgame position will solve, and how cheaply.

The production trigger is "Age III and at most N cards left". N was set from a
reach table measured on a handful of positions, which is enough to be safe and
not enough to be efficient: a sharper rule wastes fewer attempts, and every
attempt not wasted is budget available to search DEEPER.

Measure once, simulate every budget
-----------------------------------
The node count a position needs is a fixed property of that position. So one
expensive pass with a generous budget yields the economics of *every* smaller
budget exactly, with no re-solving: under budget B, a position needing n nodes
either completes (spending n) or declines (spending exactly B). Positions that
decline even under the study budget are right-censored and reported as such
rather than quietly dropped -- they are the expensive tail, which is the part a
trigger most needs to exclude.

Measure the decline rate BEFORE building a predictor
----------------------------------------------------
The node budget is already a self-limiting trigger: a position that would cost
100M nodes does not cost 100M, it costs `max_nodes` and then declines. So a
perfect predictor saves only `decline_rate x max_nodes` per triggered position.
If declines are rare at the depth you want, no predictor is worth building; if
they dominate, one is. That single number gates the rest of this study, and
`--report` prints it first.

Distribution
------------
Trigger thresholds must be calibrated on SELF-PLAY BY A TRAINED NET. Bot games
are ~3x cheaper and far narrower at equal card count (a rush bot has spent the
wonders and options that make a position expensive), and advisor captures are
human games, a third distribution again. Neither predicts the positions
self-play will actually reach.

Recorded self-play from earlier runs cannot be used on this branch: every game
in cloud6's buffers fails replay under the corrected engine (measured: 0 of 100
in iter_0010), so positions must be generated rather than loaded. Until a net
exists that matches the live encoder, `--migrate` gives the closest available
approximation -- a real net's self-play, played somewhat weakly. Thresholds
from it will be OPTIMISTIC, because weaker play reaches narrower, more decided
endgames; treat them as a lower bound on cost and re-run after the retrain.

Usage:

    python -m games.seven_wonders_duel.endgame_trigger_study \
        --checkpoint <path> --migrate --games 40 --max-cards 12 \
        --study-nodes 50000000 --out trigger_study.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from .buffer import ReplayMismatchError, read_records, replay
from .codec import legal_action_indices
from .data import WONDERS_BY_NAME, back_type_of
from .encoder import _Derived
from .engine import _science_symbols, minimum_payment, score_player
from .game import Phase
from .pool import unseen_pool
from .phase_e import load_evaluator
from .rust_bridge import (
    phase_d_record_from_rust,
    rust_flat_batch_adapter,
    rust_game_from_state,
    rust_games_for_self_play,
)

#: Cards for which a solve is even considered. Deliberately wider than any
#: production trigger: the study has to see the positions a trigger would
#: exclude in order to say whether excluding them was right.
DEFAULT_MAX_CARDS = 12

#: Wonders whose presence changes the SHAPE of the search rather than its size.
#: The Great Library draws from the progress pool (a chance edge), and the
#: Mausoleum revives from the discard (an option set that grows with the pile).
_CHANCE_WONDERS = ("The Great Library",)
_REVIVE_WONDERS = ("The Mausoleum",)


def _unbuilt(game, names) -> int:
    return sum(
        1
        for city in game.cities
        for wonder in city.wonders
        if wonder in names and wonder not in city.built_wonders
    )


def position_features(game) -> dict:
    """Everything cheap enough to compute before deciding whether to solve.

    Cheap is the constraint that makes this a trigger and not a search: each of
    these is O(board), and the alternative it gates costs millions of nodes.
    """

    cards = game.tableau.cards
    present = [card for card in cards.values() if card.present]
    actor = (
        game.pending_choice.player
        if game.pending_choice is not None
        else game.active_player
    )
    # One observation and one pool per position: `_Derived` caches the
    # reachability sets several of these features share, and rebuilding it per
    # feature would dominate the cost of computing them.
    observation = game.observation(actor)
    pool = unseen_pool(observation)
    derived = _Derived(observation, pool, actor)
    accessible = [
        slot for slot in cards if game.tableau.is_accessible(slot)
    ]
    unbuilt_wonders = sum(
        len(city.wonders) - len(city.built_wonders) for city in game.cities
    )
    return {
        # The current trigger's only variable.
        "cards_left": len(present),
        # Chance sources: each face-down card still on the board becomes a
        # reveal, and a reveal is the chance edge that makes a subtree an
        # expectimax average instead of a minimax.
        "unrevealed": sum(1 for card in present if not card.revealed),
        "accessible": len(accessible),
        # Once believed to be the strongest single correlate of node count
        # (+0.65, ahead of cards remaining); this study measured +0.478 against
        # cards' +0.938 and corrected `solver.rs`. It still earns its place --
        # its INTERACTION with depth, `cards_x_logleg`, is the only term that
        # beat the additive baseline.
        "legal": len(legal_action_indices(game)),
        # Why human and strong-net positions cost ~3x a rush bot's at equal card
        # count: the options a weak player has already spent.
        "unbuilt_wonders": unbuilt_wonders,
        "chance_wonders": _unbuilt(game, _CHANCE_WONDERS),
        "revive_wonders": _unbuilt(game, _REVIVE_WONDERS),
        "discard": len(game.discard_pile),
        "coins": sum(city.coins for city in game.cities),
        # --- decidedness ---------------------------------------------------
        # Alpha-beta can only cut a branch it can already call, so a position
        # whose result is settled collapses fast and a close one does not. That
        # makes closeness a cost predictor -- and, by the same mechanism, a
        # VALUE predictor: the positions where many moves flip the result are
        # both the expensive ones and the ones where the mask carries the most
        # information. One cause, two consequences.
        #
        # Each victory route needs its own term, because they are not
        # substitutes: a game can be dead level on points while one side is one
        # symbol from an instant science win, and `vp_gap` alone would call that
        # position close for the wrong reason.
        "military": abs(game.conflict_position),
        # Civilian: how close the scoring race is, the direct reading of "is
        # this game still live".
        "vp_gap": abs(score_player(game, 0).total - score_player(game, 1).total),
        # Scientific: 6 distinct symbols ends the game immediately, so the
        # leader's count is a distance-to-instant-win, not a score.
        "science_max": max(len(_science_symbols(game, p)) for p in (0, 1)),
        # One symbol away is a forcing threat: it collapses the opponent's
        # options to "block or lose", which is a different search shape from
        # merely being ahead on symbols.
        "science_threat": int(
            max(len(_science_symbols(game, p)) for p in (0, 1)) >= 5
        ),
        # Military: 9 is an instant win, so this is the mirror of `science_max`
        # for the other instant-win route.
        "military_to_win": 9 - abs(game.conflict_position),
        # An instant win one ply away TRUNCATES the tree, so these are predicted
        # to correlate with CHEAPER solves -- the opposite sign from `vp_gap`,
        # which is why they cannot be folded into one "closeness" number without
        # the two effects cancelling. Taken from the encoder rather than
        # rebuilt: `military_bound` and `science_missing_obtainable` route
        # through `reachable_cards`, which is what the 2026-08-14 fix corrected
        # when the discard pile was invisible to both.
        "mil_win_feasible": max(
            int(derived.military_bound(seat) >= 9 - derived.rel_position(seat))
            for seat in (0, 1)
        ),
        "sci_win_feasible": max(
            int(
                len(derived.symbols[seat])
                + derived.science_missing_obtainable(seat)
                >= 6
            )
            for seat in (0, 1)
        ),
        # --- branching ------------------------------------------------------
        # The real expectimax fan-out. A CardReveal enumerates one outcome per
        # unseen card of that back, and `chance::expand` takes the CARTESIAN
        # PRODUCT when a single take exposes several slots at once. Pool sizes
        # run 5-20, so counting face-down cards (`unrevealed`) throws away a 4x
        # spread on a multiplicative term. Summing logs is the form that matches
        # log(nodes) ~ sum of log(branching), which is what the model fits.
        "chance_fanout": sum(
            math.log10(max(1, len(pool.cards[back_type_of(card.card_name)])))
            for card in present
            if not card.revealed
        ),
        "chance_fanout_max": max(
            (
                len(pool.cards[back_type_of(card.card_name)])
                for card in present
                if not card.revealed
            ),
            default=0,
        ),
        # A live pending choice is a branch with one child per option, so the
        # count is the branching and the old 0/1 flag was throwing it away.
        "pending_options": (
            len(game.pending_choice.options) if game.pending_choice is not None else 0
        ),
        "pending_choice": int(game.pending_choice is not None),
        # A wonder nobody can pay for is not a branch.
        "affordable_wonders": sum(
            1
            for seat in (0, 1)
            for name in game.cities[seat].wonders
            if name not in game.cities[seat].built_wonders
            and name not in game.retired_wonders
            and minimum_payment(
                game, seat, WONDERS_BY_NAME[name].cost, is_wonder=True
            ).total_coins
            <= game.cities[seat].coins
        ),
        # --- interactions the additive model cannot express -----------------
        # Node count is branching^depth, so the product is the theoretically
        # correct term and the additive pair only approximates it.
        "cards_x_logleg": len(present)
        * math.log10(max(1, len(legal_action_indices(game)))),
        # The discard only creates branching for a seat holding an unbuilt
        # Mausoleum, which is why raw `discard` measured ~0.011: the effect is
        # conditional, and an unconditional term averages it away.
        "discard_x_revive": len(game.discard_pile)
        * _unbuilt(game, _REVIVE_WONDERS),
    }


def collect(
    *,
    evaluator,
    games: int,
    seed_base: int,
    max_cards: int,
    study_nodes: int,
    study_secs: float,
    full_sims: tuple[int, int],
    top_k: int,
    slots: int,
    batch_cap: int,
) -> list[dict]:
    """Play games, then price every Age III position at or under `max_cards`."""

    import seven_wonders_rust as swr

    # Solver OFF during generation: the study must observe the positions
    # ordinary self-play reaches, and a mask would change the moves played and
    # therefore the very distribution being measured.
    swr.set_endgame_solver(0)
    swr.set_cheap_top_k(0)
    seeds = [seed_base + index for index in range(games)]
    records, _ = swr.self_play_many_flat_net(
        adapter=rust_flat_batch_adapter(evaluator),
        games=rust_games_for_self_play(seeds, [i % 2 for i in range(games)]),
        game_seeds=seeds,
        global_batch_cap=batch_cap,
        leaf_batch=1,
        cheap_sims_min=16,
        cheap_sims_max=24,
        full_sims_min=full_sims[0],
        full_sims_max=full_sims[1],
        full_search_fraction=1.0,
        top_k=top_k,
        draft_prior=0.0,
        iteration=-1,
        force=True,
        age_deal_samples=32,
        cheap_double_reveal_offsets=0,
        max_active_slots=slots,
    )

    return price_records(
        (phase_d_record_from_rust(raw, validate=False) for raw in records),
        max_cards=max_cards,
        study_nodes=study_nodes,
        study_secs=study_secs,
    )


def records_from_buffers(paths, limit: int | None = None):
    """Yield replayable records from existing buffer files.

    Written for the cloud runs, which are the only large corpus of endgames
    produced by something stronger than a bot -- and which predate the military
    off-by-one fix, so they were played under different rules. About a third of
    them diverge mid-game under the corrected engine: the old engine offered a
    legal move the new one does not, and every recorded action after that point
    was chosen for a position that no longer exists. Those are dropped whole
    rather than truncated, because a prefix that stops before Age III carries no
    endgame anyway.

    Games that replay with every mask and actor matching are kept even though
    their FINAL digest differs: the divergence is in the terminal score, which
    no position before it depends on. The positions are therefore strong-play
    positions re-derived under corrected rules, not bit-copies of what the cloud
    net saw -- which is the right thing to price a solver on, and the wrong
    thing to call a reproduction.
    """

    from .cloud_position_salvage import divergence_point

    kept = dropped = 0
    for path in paths:
        for record in read_records(path):
            if limit is not None and kept >= limit:
                return
            first, _ = divergence_point(record)
            if first is not None:
                dropped += 1
                continue
            kept += 1
            yield record
    print(f"  buffer records: {kept} replayable, {dropped} diverged and dropped")


def price_records(
    records,
    *,
    max_cards: int,
    study_nodes: int,
    study_secs: float,
    allow_final_digest_drift: bool = False,
):
    """Price every Age III position at or under `max_cards` in each record.

    `allow_final_digest_drift` exists for the pre-military-fix cloud buffers,
    whose terminal SCORE differs under the corrected engine while every position
    along the way reconstructs exactly. Every position priced here is captured
    before the terminal state, so the mismatch cannot reach one -- but it is
    opt-in rather than swallowed, because for freshly generated records the same
    mismatch would mean the engine disagrees with itself.
    """

    rows: list[dict] = []
    for record_index, record in enumerate(records):
        positions: list[tuple[int, object]] = []

        def capture(game, move, positions=positions):
            if (
                game.phase is Phase.PLAY_AGE
                and game.age == 3
                and sum(1 for c in game.tableau.cards.values() if c.present)
                <= max_cards
            ):
                positions.append((move.i, game.clone()))

        try:
            replay(record, on_state=capture)
        except ReplayMismatchError as error:
            if not (allow_final_digest_drift and "final digest" in str(error)):
                raise
        total_moves = len(record.moves)
        for move_index, game in positions:
            features = position_features(game)
            started = time.perf_counter()
            answer = rust_game_from_state(game).solve_endgame(
                study_nodes, study_secs, "exact", "star1"
            )
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "game": record_index,
                    "move": move_index,
                    "moves_from_end": total_moves - move_index - 1,
                    **features,
                    # None when the study budget itself was not enough: the
                    # observation is right-censored at `study_nodes`, not
                    # missing, and the analysis must not treat it as either
                    # cheap or absent.
                    "nodes": None if answer is None else int(answer["nodes"]),
                    "censored": answer is None,
                    "regime": None if answer is None else answer["regime"],
                    "seconds": elapsed,
                    # Is an EXPENSIVE position a more VALUABLE one? The intuition
                    # says yes -- a close game with many live options is both
                    # where alpha-beta cannot cut early and where the right move
                    # decides the result -- and if it holds, a cost-weighted
                    # trigger is wrong: deep solves would be worth more each, so
                    # the optimum cap sits deeper than counting solves equally
                    # suggests. Recorded so that is measured rather than assumed.
                    "root_value": None if answer is None else float(answer["root_value"]),
                    "optimal": (
                        None
                        if answer is None
                        else sum(
                            1
                            for v in answer["per_action_value"].values()
                            if float(v) >= max(map(float, answer["per_action_value"].values())) - 1e-9
                        )
                    ),
                }
            )
    return rows


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation, which is what a threshold rule actually cares about:
    whether the feature ORDERS positions by cost, not whether it does so
    linearly."""

    n = len(xs)
    if n < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if dx == 0 or dy == 0 else num / (dx * dy)


FEATURES = (
    "cards_left",
    "unrevealed",
    "accessible",
    "legal",
    "unbuilt_wonders",
    "chance_wonders",
    "revive_wonders",
    "discard",
    "coins",
    "military",
    "vp_gap",
    "science_max",
    "science_threat",
    "military_to_win",
    "mil_win_feasible",
    "sci_win_feasible",
    "chance_fanout",
    "chance_fanout_max",
    "pending_options",
    "affordable_wonders",
    "cards_x_logleg",
    "discard_x_revive",
    "moves_from_end",
)


def simulate_budget(rows: list[dict], budget: int, rule=None) -> dict:
    """Economics of one production budget under one trigger rule, exactly.

    No re-solving: a position needing n nodes completes for n when n <= budget
    and otherwise burns exactly `budget`. Censored rows are known to need MORE
    than the study budget, so under any smaller budget they certainly decline.
    """

    attempted = [row for row in rows if rule is None or rule(row)]
    solved = spent = wasted = 0
    for row in attempted:
        needed = row["nodes"]
        if needed is not None and needed <= budget:
            solved += 1
            spent += needed
        else:
            wasted += budget
    return {
        "attempted": len(attempted),
        "solved": solved,
        "declined": len(attempted) - solved,
        "decline_rate": (len(attempted) - solved) / len(attempted) if attempted else None,
        "nodes_useful": spent,
        "nodes_wasted": wasted,
        "waste_fraction": wasted / (spent + wasted) if (spent + wasted) else None,
        # The number a trigger is trying to raise.
        "solves_per_million_nodes": (
            solved / ((spent + wasted) / 1e6) if (spent + wasted) else None
        ),
    }


def report(rows: list[dict], budgets: list[int]) -> dict:
    priced = [row for row in rows if row["nodes"] is not None]
    correlations = {}
    if priced:
        costs = [math.log10(max(1, row["nodes"])) for row in priced]
        for name in FEATURES:
            correlations[name] = _spearman([row[name] for row in priced], costs)

    by_cards: dict[str, dict] = {}
    for row in rows:
        by_cards.setdefault(str(row["cards_left"]), {"n": 0, "censored": 0, "nodes": []})
        bucket = by_cards[str(row["cards_left"])]
        bucket["n"] += 1
        if row["nodes"] is None:
            bucket["censored"] += 1
        else:
            bucket["nodes"].append(row["nodes"])
    for bucket in by_cards.values():
        nodes = sorted(bucket.pop("nodes"))
        bucket["median_nodes"] = nodes[len(nodes) // 2] if nodes else None
        bucket["p90_nodes"] = nodes[min(len(nodes) - 1, int(0.9 * len(nodes)))] if nodes else None

    return {
        "positions": len(rows),
        "censored": sum(1 for row in rows if row["nodes"] is None),
        "rank_correlation_with_log_nodes": correlations,
        "by_cards_left": {k: by_cards[k] for k in sorted(by_cards, key=int)},
        "budgets": {
            str(budget): {
                "all_positions": simulate_budget(rows, budget),
                "by_cards_cap": {
                    str(cap): simulate_budget(
                        rows, budget, rule=lambda r, cap=cap: r["cards_left"] <= cap
                    )
                    for cap in sorted({row["cards_left"] for row in rows})
                },
            }
            for budget in budgets
        },
    }


def _print(summary: dict) -> None:
    print(f"{summary['positions']} positions priced, {summary['censored']} censored")
    print("\n  rank correlation with log10(nodes) -- what orders positions by cost:")
    ordered = sorted(
        (
            (abs(v), k, v)
            for k, v in summary["rank_correlation_with_log_nodes"].items()
            if v is not None
        ),
        reverse=True,
    )
    for _, name, value in ordered:
        print(f"    {name:<18} {value:+.3f}")
    print("\n  cost by cards left:")
    print("    cards    n  censored   median      p90")
    for cards, bucket in summary["by_cards_left"].items():
        median = bucket["median_nodes"]
        p90 = bucket["p90_nodes"]
        print(
            f"    {cards:>5} {bucket['n']:>4} {bucket['censored']:>9} "
            f"{median if median is None else f'{median:>8,}'} "
            f"{p90 if p90 is None else f'{p90:>8,}'}"
        )
    for budget, block in summary["budgets"].items():
        allp = block["all_positions"]
        if not allp["attempted"]:
            continue
        print(f"\n  at a {int(budget):,}-node budget, over every position seen:")
        print(
            f"    decline rate {allp['decline_rate']:.0%}, "
            f"{allp['waste_fraction']:.0%} of nodes wasted, "
            f"{allp['solves_per_million_nodes']:.2f} solves/Mnode"
        )
        print("    cards cap  attempted  solved  decline  waste  solves/Mnode")
        for cap, stats in block["by_cards_cap"].items():
            if not stats["attempted"]:
                continue
            print(
                f"    {cap:>9} {stats['attempted']:>10} {stats['solved']:>7} "
                f"{stats['decline_rate']:>8.0%} {stats['waste_fraction']:>6.0%} "
                f"{stats['solves_per_million_nodes']:>13.2f}"
            )


def measure_node_rate(samples: int = 8) -> float:
    """This machine's single-thread solver rate, in nodes/second.

    The only machine-dependent quantity in the whole calibration. Everything
    else -- how many nodes a position needs, how the cost is distributed across
    card counts -- is a property of the POSITIONS, identical on every box. That
    asymmetry is what makes calibrating a rented machine cheap: price the
    positions once, anywhere, then measure only this on the box you rent.

    Timed on real 9-10 card endgames rather than a synthetic loop, because the
    rate is not constant -- chance-heavy subtrees cost more per node than
    minimax ones, and a benchmark over the wrong shape would misprice the
    budget in exactly the positions the budget has to cover.
    """

    from .encoder_audit import DEFAULT_PAIRINGS, make_bot
    from .engine import apply_action
    from .game import new_game

    rates: list[float] = []
    for index in range(60):
        left, right = DEFAULT_PAIRINGS[index % len(DEFAULT_PAIRINGS)]
        game = new_game(index)
        bots = (make_bot(left, index), make_bot(right, index + 10_000))
        while game.phase is not Phase.COMPLETE:
            if game.phase is Phase.PLAY_AGE and game.age == 3:
                present = sum(1 for c in game.tableau.cards.values() if c.present)
                if present in (9, 10):
                    started = time.perf_counter()
                    answer = rust_game_from_state(game.clone()).solve_endgame(
                        40_000_000, 60.0, "exact", "star1"
                    )
                    elapsed = time.perf_counter() - started
                    if answer and answer["nodes"] > 500_000 and elapsed > 0:
                        rates.append(answer["nodes"] / elapsed)
                    break
            actor = (
                game.pending_choice.player
                if game.pending_choice is not None
                else game.active_player
            )
            apply_action(game, bots[actor].select_action(game))
        if len(rates) >= samples:
            break
    if not rates:
        raise RuntimeError("could not time a single solve")
    rates.sort()
    return rates[len(rates) // 2]


def calibrate(rows: list[dict], rate: float, seconds_per_game: float, games: int) -> dict:
    """Pick the deepest cap and budget that fit a per-game time allowance.

    The knob a run actually has is "how much wall time per game may the solver
    take", because that is what trades against generation throughput. Cap and
    node budget are downstream of it, and both depend on the box only through
    `rate`.
    """

    options = []
    for cap in sorted({row["cards_left"] for row in rows}):
        for budget in (500_000, 1_000_000, 2_000_000, 4_500_000, 9_000_000, 20_000_000):
            stats = simulate_budget(rows, budget, rule=lambda r, c=cap: r["cards_left"] <= c)
            if not stats["attempted"]:
                continue
            nodes_per_game = (stats["nodes_useful"] + stats["nodes_wasted"]) / games
            options.append(
                {
                    "cap": cap,
                    "budget": budget,
                    "seconds_per_game": nodes_per_game / rate,
                    "solves_per_game": stats["solved"] / games,
                    "decline_rate": stats["decline_rate"],
                    "waste_fraction": stats["waste_fraction"],
                }
            )
    affordable = [o for o in options if o["seconds_per_game"] <= seconds_per_game]
    # Most solves per game inside the allowance; ties to the cheaper option, so
    # a budget that buys nothing extra is never chosen over one that does.
    best = max(
        affordable,
        key=lambda o: (o["solves_per_game"], -o["seconds_per_game"]),
        default=None,
    )
    return {
        "node_rate": rate,
        "seconds_per_game_allowance": seconds_per_game,
        "recommended": best,
        "frontier": sorted(options, key=lambda o: o["seconds_per_game"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--from-buffer",
        type=Path,
        nargs="+",
        help="price positions from existing buffer files instead of generating "
        "games. The cloud runs are the only large corpus of endgames reached by "
        "a trained net rather than a bot, which is the distribution the trigger "
        "will actually meet. Needs no --checkpoint: the games are already "
        "played. Records that do not replay under the current engine are "
        "dropped, and the count is printed.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--inference-precision", default="fp32")
    parser.add_argument("--migrate", action="store_true")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=20260816)
    parser.add_argument("--max-cards", type=int, default=DEFAULT_MAX_CARDS)
    parser.add_argument("--study-nodes", type=int, default=50_000_000)
    parser.add_argument("--study-secs", type=float, default=120.0)
    parser.add_argument("--full-sims", type=int, nargs=2, default=[64, 128])
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--global-batch-cap", type=int, default=512)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[500_000, 2_000_000, 5_000_000, 20_000_000],
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--calibrate",
        type=Path,
        help="skip generation and calibrate THIS machine against a cost profile "
        "written by an earlier --out run. Node counts are machine-independent, "
        "so the expensive study is done once and a rented box only needs its "
        "solver rate measured -- about a minute.",
    )
    parser.add_argument(
        "--seconds-per-game",
        type=float,
        default=2.0,
        help="wall-clock the solver may spend per self-play game. The cap and "
        "node budget are derived from it and this machine's measured rate.",
    )
    return parser


def _print_calibration(result: dict) -> None:
    print(f"measured solver rate: {result['node_rate']/1e6:.2f}M nodes/s")
    print(f"allowance: {result['seconds_per_game_allowance']:.2f}s per game\n")
    best = result["recommended"]
    if best is None:
        print("  nothing fits the allowance; raise --seconds-per-game")
        return
    print(
        f"  RECOMMENDED: --endgame-solver-max-cards {best['cap']} "
        f"--endgame-solver-max-nodes {best['budget']}"
    )
    print(
        f"    {best['solves_per_game']:.2f} solves/game at "
        f"{best['seconds_per_game']:.2f}s/game, {best['decline_rate']:.0%} declines\n"
    )
    print("  frontier (what more time would buy):")
    print("    cap    budget   sec/game  solves/game  declines")
    seen = -1.0
    for option in result["frontier"]:
        if option["solves_per_game"] <= seen:
            continue  # dominated: costs more, solves no more
        seen = option["solves_per_game"]
        print(
            f"    {option['cap']:>3} {option['budget']:>9,} {option['seconds_per_game']:>10.2f} "
            f"{option['solves_per_game']:>12.2f} {option['decline_rate']:>9.0%}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.calibrate:
        profile = json.loads(args.calibrate.read_text(encoding="utf-8"))
        result = calibrate(
            profile["rows"],
            measure_node_rate(),
            args.seconds_per_game,
            profile["games"],
        )
        _print_calibration(result)
        if args.out:
            args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"\nwrote {args.out}")
        return 0
    if args.from_buffer:
        rows = price_records(
            records_from_buffers(args.from_buffer, limit=args.games),
            max_cards=args.max_cards,
            study_nodes=args.study_nodes,
            study_secs=args.study_secs,
            allow_final_digest_drift=True,
        )
        summary = report(rows, args.budgets)
        _print(summary)
        if args.out:
            args.out.write_text(
                json.dumps(
                    {
                        "source": [str(path) for path in args.from_buffer],
                        "games": args.games,
                        "study_nodes": args.study_nodes,
                        "summary": summary,
                        "rows": rows,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"wrote {args.out}")
        return 0
    if args.checkpoint is None:
        parser = build_parser()
        parser.error("one of --checkpoint or --from-buffer is required")
    evaluator = load_evaluator(
        str(args.checkpoint),
        args.device,
        precision=args.inference_precision,
        migrate=args.migrate,
    )
    if args.migrate:
        print(
            "WARNING: --migrate. Off-distribution net, so it plays weaker than a "
            "trained one and reaches narrower, more decided endgames. Thresholds "
            "from this run are OPTIMISTIC; re-run after the retrain.\n"
        )
    rows = collect(
        evaluator=evaluator,
        games=args.games,
        seed_base=args.seed_base,
        max_cards=args.max_cards,
        study_nodes=args.study_nodes,
        study_secs=args.study_secs,
        full_sims=tuple(args.full_sims),
        top_k=args.top_k,
        slots=args.slots,
        batch_cap=args.global_batch_cap,
    )
    summary = report(rows, args.budgets)
    _print(summary)
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "migrate": args.migrate,
                    "games": args.games,
                    "study_nodes": args.study_nodes,
                    "summary": summary,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
