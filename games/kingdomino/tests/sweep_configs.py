"""Sweep harness for tuning the Kingdomino MCTS bot.

Plays MCTS configurations against a fixed baseline (GreedyBot) and prints a
comparison table. Two things keep the comparison fair and low-variance:

  * Each deck seed is played twice -- once with MCTS as P0 and once as P1 --
    so first-player advantage cancels out.
  * Every configuration is evaluated on the SAME set of deck seeds (common
    random numbers), so score differences reflect the configuration rather
    than which decks happened to come up.

Run:
    python -m games.kingdomino.sweep

Scale cost with the constants below. Rough cost ~ N_SEEDS * 2 * SIMS * moves.
Start small (N_SEEDS=4, SIMS=25) to sanity check, then scale up.
"""

import random
import statistics
import time

from games.kingdomino.game import GameState
from games.kingdomino.bots import GreedyBot
from games.kingdomino.mcts import MCTSBot
from games.kingdomino.bot_match import (
    legal_actions,
    apply_action,
    is_terminal,
    current_player,
    total_score,
)

# ----------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------
N_SEEDS = 6          # distinct decks; each is played from both seats
SIMS = 50            # default simulation budget (a config may override it)

# Each entry: (name, MCTSBot kwargs). "simulations" may be overridden per config.
DEFAULT_CONFIGS = [
    # --- Phase A: rollout policy, progressive-widening defaults -------------
    ("random rollout d8", dict(rollout_policy="random", rollout_depth_limit=8)),
    ("evaluate leaf",     dict(rollout_policy="evaluate")),
    ("greedy rollout d8", dict(rollout_policy="greedy", rollout_depth_limit=8)),

    # --- Phase B: progressive-widening grid (cheap evaluate leaf) -----------
    ("eval pw_c=1 a=0.4", dict(rollout_policy="evaluate", pw_c=1.0, pw_alpha=0.4)),
    ("eval pw_c=2 a=0.4", dict(rollout_policy="evaluate", pw_c=2.0, pw_alpha=0.4)),
    ("eval pw_c=2 a=0.6", dict(rollout_policy="evaluate", pw_c=2.0, pw_alpha=0.6)),
    ("eval pw_c=4 a=0.5", dict(rollout_policy="evaluate", pw_c=4.0, pw_alpha=0.5)),

    # --- Baseline: effectively no widening (old fan-out-first behavior) -----
    ("no widening (old)", dict(rollout_policy="evaluate", pw_c=1e9, pw_alpha=1.0)),
]


def play_game(bot0, bot1, deck_seed, rng):
    """Play one full game; return (scores, move_time_per_seat, move_count_per_seat)."""
    state = GameState.new(seed=deck_seed)
    bots = (bot0, bot1)
    move_time = [0.0, 0.0]
    move_count = [0, 0]

    while not is_terminal(state):
        p = current_player(state)
        actions = legal_actions(state)
        t0 = time.perf_counter()
        action = bots[p].choose_action(state, actions, rng=rng)
        move_time[p] += time.perf_counter() - t0
        move_count[p] += 1
        state = apply_action(state, action)

    scores = [total_score(b.score()) for b in state.boards]
    return scores, move_time, move_count


def eval_config(name, kwargs, seeds, sims, opponent_factory):
    kwargs = dict(kwargs)
    sims = kwargs.pop("simulations", sims)

    wins = draws = losses = 0
    margins = []
    my_scores = []
    mcts_time = 0.0
    mcts_moves = 0

    for deck_seed in seeds:
        for mcts_seat in (0, 1):
            # Deterministic per (deck, seat): seed the global RNG (used by the
            # tree's tie-breaking) and a dedicated stream for everything else.
            stream_seed = (deck_seed * 1000003) ^ (mcts_seat * 0x9E3779B1)
            random.seed(stream_seed)
            game_rng = random.Random(stream_seed)

            mcts = MCTSBot(simulations=sims, seed=stream_seed, **kwargs)
            opp = opponent_factory()
            bot0, bot1 = (mcts, opp) if mcts_seat == 0 else (opp, mcts)

            scores, move_time, move_count = play_game(bot0, bot1, deck_seed, game_rng)

            mine = scores[mcts_seat]
            theirs = scores[1 - mcts_seat]
            my_scores.append(mine)
            margins.append(mine - theirs)
            if mine > theirs:
                wins += 1
            elif mine == theirs:
                draws += 1
            else:
                losses += 1

            mcts_time += move_time[mcts_seat]
            mcts_moves += move_count[mcts_seat]

    games = len(seeds) * 2
    return {
        "name": name,
        "games": games,
        "record": f"{wins}-{losses}-{draws}",
        "win_rate": (wins + 0.5 * draws) / games,
        "avg_score": statistics.mean(my_scores),
        "avg_margin": statistics.mean(margins),
        "ms_move": 1000.0 * mcts_time / max(1, mcts_moves),
    }


def print_table(rows):
    header = ("config", "games", "W-L-T", "win%", "avg_score", "avg_margin", "ms/move")
    widths = (22, 6, 9, 7, 10, 11, 9)

    def fmt_row(vals):
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))

    print(fmt_row(header))
    print(fmt_row(tuple("-" * w for w in widths)))
    for r in rows:
        print(fmt_row((
            r["name"],
            r["games"],
            r["record"],
            f"{100 * r['win_rate']:.0f}%",
            f"{r['avg_score']:.1f}",
            f"{r['avg_margin']:+.1f}",
            f"{r['ms_move']:.0f}",
        )))


def run_sweep(configs=DEFAULT_CONFIGS, n_seeds=N_SEEDS, sims=SIMS,
              opponent_factory=GreedyBot, seed_base=10_000):
    seeds = [seed_base + i for i in range(n_seeds)]
    print(f"Opponent: {opponent_factory().__class__.__name__} | "
          f"decks: {n_seeds} (x2 seats = {2 * n_seeds} games/config) | "
          f"default sims: {sims}\n")

    rows = []
    for name, kwargs in configs:
        t0 = time.perf_counter()
        result = eval_config(name, kwargs, seeds, sims, opponent_factory)
        result["_wall"] = time.perf_counter() - t0
        rows.append(result)
        print(f"  done: {name:22s}  win% {100*result['win_rate']:4.0f}  "
              f"({result['_wall']:.0f}s)")

    rows.sort(key=lambda r: (r["win_rate"], r["avg_margin"]), reverse=True)
    print()
    print_table(rows)
    print("\n(win% counts ties as half; avg_margin and avg_score are from the "
          "MCTS player's perspective, averaged over both seats)")


# ----------------------------------------------------------------------------
# Round-robin ladder: every config plays every other config directly.
# ----------------------------------------------------------------------------
def _mcts_factory(kwargs):
    kwargs = dict(kwargs)

    def make(sims, seed):
        local = dict(kwargs)
        sims_used = local.pop("simulations", sims)
        return MCTSBot(simulations=sims_used, seed=seed, **local)

    return make


def _greedy_factory():
    def make(sims, seed):
        return GreedyBot()

    return make


def run_ladder(configs=DEFAULT_CONFIGS, n_seeds=N_SEEDS, sims=SIMS,
               include_greedy=True, seed_base=20_000):
    """Round-robin: each participant plays every other, both seats, shared decks."""
    participants = [(name, _mcts_factory(kw)) for name, kw in configs]
    if include_greedy:
        participants.append(("Greedy", _greedy_factory()))

    seeds = [seed_base + i for i in range(n_seeds)]
    games_per_pair = 2 * n_seeds

    stand = {
        name: dict(w=0, l=0, d=0, pts=0.0, margin=0.0, score=0.0,
                   games=0, time=0.0, moves=0)
        for name, _ in participants
    }
    # h2h[a][b] = [points a took from b, games between them]
    h2h = {a: {b: [0.0, 0] for b, _ in participants} for a, _ in participants}

    n = len(participants)
    print(f"Ladder: {n} participants, {n * (n - 1) // 2} pairings | "
          f"decks: {n_seeds} (x2 seats = {games_per_pair} games/pair) | "
          f"default sims: {sims}\n")

    for i in range(n):
        for j in range(i + 1, n):
            name_a, fac_a = participants[i]
            name_b, fac_b = participants[j]
            t0 = time.perf_counter()

            for deck_seed in seeds:
                for a_seat in (0, 1):
                    stream_seed = (deck_seed * 1000003) ^ (a_seat * 0x9E3779B1) \
                        ^ (hash((name_a, name_b)) & 0xFFFFFFFF)
                    random.seed(stream_seed)
                    game_rng = random.Random(stream_seed)

                    a = fac_a(sims, stream_seed)
                    b = fac_b(sims, stream_seed ^ 0x55555555)
                    bot0, bot1 = (a, b) if a_seat == 0 else (b, a)

                    scores, mtime, mcount = play_game(bot0, bot1, deck_seed, game_rng)
                    sa, sb = scores[a_seat], scores[1 - a_seat]

                    ea, eb = stand[name_a], stand[name_b]
                    ea["games"] += 1
                    eb["games"] += 1
                    ea["margin"] += sa - sb
                    eb["margin"] += sb - sa
                    ea["score"] += sa
                    eb["score"] += sb
                    ea["time"] += mtime[a_seat]
                    ea["moves"] += mcount[a_seat]
                    eb["time"] += mtime[1 - a_seat]
                    eb["moves"] += mcount[1 - a_seat]
                    h2h[name_a][name_b][1] += 1
                    h2h[name_b][name_a][1] += 1

                    if sa > sb:
                        ea["w"] += 1; eb["l"] += 1; ea["pts"] += 1
                        h2h[name_a][name_b][0] += 1
                    elif sa == sb:
                        ea["d"] += 1; eb["d"] += 1
                        ea["pts"] += 0.5; eb["pts"] += 0.5
                        h2h[name_a][name_b][0] += 0.5
                        h2h[name_b][name_a][0] += 0.5
                    else:
                        eb["w"] += 1; ea["l"] += 1; eb["pts"] += 1
                        h2h[name_b][name_a][0] += 1

            print(f"  {name_a:20s} vs {name_b:20s}  ({time.perf_counter() - t0:.0f}s)")

    order = sorted(stand, key=lambda nm: (stand[nm]["pts"], stand[nm]["margin"]),
                   reverse=True)
    print()
    _print_standings(order, stand)
    print()
    _print_h2h_matrix(order, h2h)
    print("\n(win% = points / games, ties as half; avg_margin from each "
          "participant's own perspective, averaged over all opponents and seats)")


def _print_standings(order, stand):
    header = ("#", "participant", "games", "W-L-T", "pts", "win%", "avg_margin", "ms/move")
    widths = (3, 22, 6, 9, 6, 6, 11, 9)

    def row(vals):
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))

    print(row(header))
    print(row(tuple("-" * w for w in widths)))
    for rank, nm in enumerate(order, 1):
        e = stand[nm]
        g = max(1, e["games"])
        print(row((
            rank, nm, e["games"], f"{e['w']}-{e['l']}-{e['d']}",
            f"{e['pts']:.1f}", f"{100 * e['pts'] / g:.0f}%",
            f"{e['margin'] / g:+.1f}", f"{1000 * e['time'] / max(1, e['moves']):.0f}",
        )))


def _print_h2h_matrix(order, h2h):
    tags = {nm: f"C{i + 1}" for i, nm in enumerate(order)}
    print("Head-to-head win% (row vs column):")
    cell_w = 6
    head = " " * 6 + "".join(tags[nm].ljust(cell_w) for nm in order)
    print(head)
    for nm in order:
        line = tags[nm].ljust(6)
        for opp in order:
            if nm == opp:
                line += "-".ljust(cell_w)
            else:
                pts, games = h2h[nm][opp]
                line += (f"{100 * pts / games:.0f}%" if games else "·").ljust(cell_w)
        print(line)
    print("\nlegend: " + "  ".join(f"{tags[nm]}={nm}" for nm in order))


if __name__ == "__main__":
    # Default entry point runs the round-robin ladder. For the
    # vs-Greedy-only table instead, call run_sweep().
    run_ladder()