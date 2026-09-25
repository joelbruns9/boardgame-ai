"""Head-to-head matches between evaluators.

An "player" here is just an ``evaluate(boards) -> (N, num_players)`` callable:
a trained net (``NetEvaluator``), an earlier checkpoint, or the
``ProgressHeuristic``. They all drive the same turn solver, so a match
compares *leaf evaluation quality* with search held constant.

Seats are rotated so every player spends an equal number of games in every
seat. Can't Stop has a real first-player advantage, and with 2-4 seats an
unrotated match would measure it instead of the players.
"""

import math
import random

from .engine import GameState
from .portable_rng import PortableRng
from .self_play import DEFAULT_MAX_TURNS, PLAIN, TurnLimitExceeded, play_turn


def play_match_game(rules, seating, rng, max_turns=DEFAULT_MAX_TURNS,
                    search_seating=None):
    """Play one game where ``seating[seat]`` evaluates for that seat, and
    ``search_seating[seat]`` (a ``self_play.Search``; plain when omitted) is
    how that seat searches.

    Returns the winning *seat*, which the caller maps back to a player.
    """
    state = GameState(rules)
    turns = 0
    while not state.game_over:
        if turns >= max_turns:
            raise TurnLimitExceeded(
                f"{rules} reached {max_turns} turns with no winner")
        a = state.active_player
        play_turn(state, seating[a], rng,
                  search=search_seating[a] if search_seating else PLAIN)
        turns += 1
    return state.winner


def seating_for_game(num_players, game_index):
    """Seat index for each player in game ``game_index``.

    ``seating_for_game(n, i)[p]`` is where player ``p`` sits. Rotating by the
    game index means that over any run of ``n`` games each player occupies
    each seat exactly once, which is what keeps the first-player advantage
    out of the result.
    """
    return [(p + game_index) % num_players for p in range(num_players)]


def player_of_seat(seat_of, seat):
    """Invert a seating: which player sat at ``seat``.

    Split out because crediting a win to the winning *seat* instead of the
    player who occupied it is invisible in any single game and inverts the
    whole match once seats rotate.
    """
    return seat_of.index(seat)


def play_match(rules, players, games, rng=None, max_turns=DEFAULT_MAX_TURNS,
               backend="auto", threads=0, in_flight=None, searches=None):
    """Play ``games`` games between ``players`` and count wins per player.

    ``players`` must have exactly ``rules.num_players`` entries. Game *i*
    seats player *p* at seat ``(p + i) % num_players``, so over a multiple of
    ``num_players`` games every player sits in every seat equally often.
    ``searches`` gives each player its own ``self_play.Search`` (plain for
    all when omitted) -- how the lookahead is measured: the same net at
    depth 2 against itself at depth 1.

    Each game rolls from its own ``PortableRng`` seeded from ``rng`` up
    front, so the ``"rust"`` pool and the ``"python"`` loop play the same
    games (``"auto"``: rust when built).
    """
    n = rules.num_players
    if len(players) != n:
        raise ValueError(
            f"{rules.num_players}-player rules need {n} players, got "
            f"{len(players)}")
    rng = rng or random.Random()
    from . import rust_pool
    if backend == "auto":
        backend = "rust" if rust_pool.rust_available() else "python"
    if backend == "rust":
        return rust_pool.play_match(
            rules, players, games, rng, max_turns=max_turns, threads=threads,
            in_flight=in_flight or rust_pool.DEFAULT_IN_FLIGHT,
            searches=searches)
    if backend != "python":
        raise ValueError(f"unknown backend {backend!r}")
    seeds = rust_pool.game_seeds(rng, games)

    wins = [0] * n
    for i in range(games):
        seat_of = seating_for_game(n, i)
        seating = [None] * n
        search_seating = [PLAIN] * n
        for player, seat in enumerate(seat_of):
            seating[seat] = players[player]
            if searches is not None:
                search_seating[seat] = searches[player]
        winning_seat = play_match_game(rules, seating,
                                       PortableRng(seeds[i]), max_turns,
                                       search_seating)
        wins[player_of_seat(seat_of, winning_seat)] += 1
    return wins


def win_rate(wins, player=0):
    total = sum(wins)
    return wins[player] / total if total else 0.0


def wilson_interval(successes, total, z=1.96):
    """95% Wilson score interval for a win rate.

    Reported instead of a bare rate because MVP matches are short: 60 games
    at 55% does not distinguish a better net from noise, and the interval
    makes that visible rather than inviting a conclusion the sample cannot
    support.
    """
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt(p * (1 - p) / total
                           + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def verdict(wins, num_players, player=0):
    """Summarize a match: rate, interval, and whether it is actually better.

    Separate from ``compare`` so the significance rule can be tested without
    playing games. "Better" is the interval's lower bound clearing an even
    match, never the bare rate -- at MVP match sizes a rate above even is
    routinely noise.
    """
    total = sum(wins)
    lo, hi = wilson_interval(wins[player], total)
    even = 1.0 / num_players
    return {
        "games": total,
        "wins": list(wins),
        "win_rate": win_rate(wins, player),
        "ci95": (lo, hi),
        "even_match": even,
        "better": lo > even,
    }


def compare(rules, challenger, incumbent, games, rng=None,
            max_turns=DEFAULT_MAX_TURNS, backend="auto", threads=0,
            in_flight=None, search=PLAIN):
    """Match one challenger against copies of one incumbent.

    Returns the challenger's win rate, its Wilson interval, and the raw wins.
    A challenger is only credibly better when the interval's lower bound
    clears the ``1 / num_players`` an even match would produce.
    """
    n = rules.num_players
    players = [challenger] + [incumbent] * (n - 1)
    wins = play_match(rules, players, games, rng, max_turns, backend,
                      threads, in_flight, [search] * n)
    return verdict(wins, n)
