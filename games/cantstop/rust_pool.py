"""Many games at once on the Rust pool, with one batched forward per round.

Phase 3 M3 of ``VARIANT_SOLVER_PLAN.md``. The Rust ``SelfPlayPool`` plays
every game's turns -- dice, enumeration, backup, moves, row recording -- and
stops each game where its turn needs leaf values. Python's only job is the
evaluator:

    pool.advance()
    while pool.running:
        feats, blocks = pool.pending()          # every waiting game's leaves
        pool.resume(values_for(feats, blocks))  # one forward per evaluator

Evaluators follow the solver's contract with one addition, ``evaluate_
features(features, reference)`` -> absolute-seat values; ``NetEvaluator`` and
``ProgressHeuristic`` both have it. A ``NetEvaluator`` additionally exposes
``relative_probs``, and then all of its blocks go through **one** forward
and are rotated back per block -- that coalescing is the point of M3.

Seeding is per game: game *i* rolls from ``PortableRng(seeds[i])``, the same
stream in Python and Rust, so ``self_play.play_game(rules, evaluate,
PortableRng(seed))`` is the single-game reference the M3 gate replays.
"""

import time

import numpy as np

from .encoder import FEATURE_SIZE, to_absolute
from .engine import GameState
from .portable_rng import PortableRng
from .self_play import (DEFAULT_MAX_TURNS, PLAIN, GameResult,
                        TurnLimitExceeded)
from .snapshot import snapshot


def _rust():
    import cantstop_rust
    return cantstop_rust


def rust_available():
    try:
        _rust()
    except ImportError:
        return False
    return True


def game_seeds(rng, count):
    """Per-game seeds, drawn up front in schedule order, so a game's dice do
    not depend on how many games before it happened to run."""
    return [rng.randrange(2 ** 64) for _ in range(count)]


def _reference(rules_by_game, block):
    game, _rows, _ev, _n, leaf_active = block[:5]
    ref = GameState(rules_by_game[game])
    ref.active_player = leaf_active
    return ref


def values_for(evaluators, features, blocks, rules_by_game):
    """Absolute-seat values for a pending buffer, as float64 LE bytes in
    block order. Blocks sharing a batched evaluator go through one call."""
    starts = np.cumsum([0] + [b[1] for b in blocks])
    out = [None] * len(blocks)
    by_eval = {}
    for i, b in enumerate(blocks):
        by_eval.setdefault(b[2], []).append(i)
    for ev_id, idxs in by_eval.items():
        ev = evaluators[ev_id]
        if hasattr(ev, "relative_probs"):
            rows = np.concatenate([features[starts[i]:starts[i + 1]]
                                   for i in idxs])
            probs = ev.relative_probs(rows)
            at = 0
            for i in idxs:
                n = blocks[i][1]
                out[i] = to_absolute(probs[at:at + n],
                                     _reference(rules_by_game, blocks[i]))
                at += n
        else:
            for i in idxs:
                out[i] = ev.evaluate_features(
                    features[starts[i]:starts[i + 1]],
                    _reference(rules_by_game, blocks[i]))
    for i, b in enumerate(blocks):
        want = (b[1], b[3])
        if np.shape(out[i]) != want:
            raise ValueError(f"evaluator {b[2]} returned shape "
                             f"{np.shape(out[i])}, expected {want}")
    return np.concatenate([np.asarray(v, dtype="<f8").ravel()
                           for v in out]).tobytes()


def relative_for(evaluators, features, blocks):
    """Seat-relative probabilities (rows, 4) float32 LE bytes in block order,
    when every evaluator is a batched net (has ``relative_probs``). Rust then
    rotates each block to absolute seats itself (``resume_relative``). One
    forward per evaluator per round; self-play needs no regrouping at all."""
    ids = {b[2] for b in blocks}
    if len(ids) == 1:
        probs = evaluators[ids.pop()].relative_probs(features)
        return np.ascontiguousarray(probs, dtype="<f4").tobytes()
    starts = np.cumsum([0] + [b[1] for b in blocks])
    out = np.empty((len(features), 4), dtype="<f4")
    for ev_id in sorted(ids):
        idxs = [i for i, b in enumerate(blocks) if b[2] == ev_id]
        rows = np.concatenate([features[starts[i]:starts[i + 1]]
                               for i in idxs])
        probs = evaluators[ev_id].relative_probs(rows)
        at = 0
        for i in idxs:
            n = blocks[i][1]
            out[starts[i]:starts[i + 1]] = probs[at:at + n]
            at += n
    return out.tobytes()


class BoardEvaluator:
    """Adapter: any ``evaluate(boards)`` callable as a feature evaluator.

    The pool hands out encoded features only. Decoding is exact
    (``encoder.decode_features``; gated in ``test_rust_pool``), so the
    wrapped callable sees the same end-of-turn boards the Python solver
    would have given it. Added after review: ``backend="auto"`` used to pick
    the pool for any evaluator and then fail on plain callables."""

    def __init__(self, evaluate):
        self.evaluate = evaluate

    def evaluate_features(self, features, reference):
        from .encoder import decode_features
        return self.evaluate([decode_features(row, reference.active_player)
                              for row in features])


def as_feature_evaluator(evaluate):
    if hasattr(evaluate, "evaluate_features"):
        return evaluate
    if not callable(evaluate):
        raise TypeError(f"not an evaluator: {evaluate!r}")
    return BoardEvaluator(evaluate)


class PoolStats:
    """What a pool run cost: rounds (= forwards per evaluator), rows, and
    where the wall time went."""

    def __init__(self):
        self.rounds = 0
        self.rows = 0
        self.rust_seconds = 0.0
        self.eval_seconds = 0.0
        self.wall_seconds = 0.0

    def as_dict(self):
        return dict(vars(self))


# Games live at once. Enough to keep every core busy and each forward
# large; finished games are replaced from the schedule so the batch stays
# full until the schedule runs out.
DEFAULT_IN_FLIGHT = 64

# Leaf rows per round (one forward per evaluator). ~400 MB of features;
# bounds host and GPU memory once lookahead multiplies each game's rows.
DEFAULT_MAX_ROWS = 1_000_000


def run_pool(rules_list, seeds, evaluators, seating=None,
             max_turns=DEFAULT_MAX_TURNS, threads=0, stats=None,
             in_flight=DEFAULT_IN_FLIGHT, search=PLAIN, searches=None,
             search_seating=None, max_rows=DEFAULT_MAX_ROWS, starts=None,
             allow_unfinished=False):
    """Play one game per entry of ``rules_list`` on the Rust pool.

    ``seating[i][seat]`` is the evaluator index for that seat of game *i*
    (all zeros when omitted: self-play with ``evaluators[0]``). Returns one
    ``GameResult`` per game, in order. Raises ``TurnLimitExceeded`` like
    ``play_game`` if any game runs out of turns.

    ``search`` (a ``self_play.Search``) applies to every seat; or give
    ``searches`` plus ``search_seating[i][seat]`` indices, as evaluators.
    """
    if searches is None:
        searches = [search]
    rust = _rust()
    stats = stats if stats is not None else PoolStats()
    if seating is None:
        seating = [[0] * r.num_players for r in rules_list]
    # A plain zip would silently truncate the schedule to its shortest
    # input (found in review: two games with no seeds returned []).
    if not len(rules_list) == len(seeds) == len(seating):
        raise ValueError(
            f"schedule lengths differ: {len(rules_list)} rule sets, "
            f"{len(seeds)} seeds, {len(seating)} seatings")
    evaluators = [as_feature_evaluator(e) for e in evaluators]
    used = {i for seats in seating for i in seats}
    if used and max(used) >= len(evaluators):
        raise ValueError(f"seating uses evaluator {max(used)} but only "
                         f"{len(evaluators)} were given")
    # ``starts``: turn-start positions to play from instead of empty boards
    # (rollouts, one-turn solves); ``allow_unfinished`` returns games that
    # hit ``max_turns`` (winner -1) instead of raising.
    if starts is None:
        starts = [GameState(r) for r in rules_list]
    elif len(starts) != len(rules_list) or any(
            st.rules != r for st, r in zip(starts, rules_list)):
        raise ValueError("starts must match rules_list one to one")
    specs = [(snapshot(st), s, list(seats))
             for st, s, seats in zip(starts, seeds, seating, strict=True)]
    batched = all(hasattr(e, "relative_probs") for e in evaluators)
    started = time.perf_counter()
    pool = rust.SelfPlayPool(
        specs, max_turns, threads, in_flight,
        [(s.exact_root, s.lookahead_k, s.lookahead_offset, s.stop_bias)
         for s in searches],
        search_seating, max_rows)

    t = time.perf_counter()
    pool.advance()
    stats.rust_seconds += time.perf_counter() - t
    while pool.running:
        t = time.perf_counter()
        raw, blocks = pool.pending()
        stats.rust_seconds += time.perf_counter() - t
        # A writable bytearray Rust filled in place: no copy here.
        features = np.frombuffer(raw, dtype="<f4").reshape(-1, FEATURE_SIZE)
        t = time.perf_counter()
        if batched:
            values = relative_for(evaluators, features, blocks)
        else:
            values = values_for(evaluators, features, blocks, rules_list)
        stats.eval_seconds += time.perf_counter() - t
        stats.rounds += 1
        stats.rows += len(features)
        t = time.perf_counter()
        if batched:
            pool.resume_relative(values)
        else:
            pool.resume(values)
        stats.rust_seconds += time.perf_counter() - t
    stats.wall_seconds += time.perf_counter() - started

    if pool.failure is not None and not allow_unfinished:
        raise TurnLimitExceeded(pool.failure)
    all_features = np.frombuffer(bytearray(pool.features()),
                                 dtype="<f4").reshape(-1, FEATURE_SIZE)
    results, at = [], 0
    for (gid, winner, rows, slots, turns, solves, ev_rows, lengths,
         values), rules in zip(pool.results(), rules_list, strict=True):
        feats = all_features[at:at + rows]
        at += rows
        results.append(GameResult(
            rules=rules,
            winner=winner,
            features=(feats.copy() if rows else
                      np.zeros((0, 0), dtype=np.float32)),
            winner_slots=np.asarray(slots, dtype=np.int64),
            turns=turns,
            solves=solves,
            evaluator_rows=ev_rows,
            turn_lengths=list(lengths),
            turn_values=[None if v is None else list(v) for v in values],
        ))
    return results


def generate(rule_sets, games_per_ruleset, evaluate, rng,
             max_turns=DEFAULT_MAX_TURNS, threads=0, stats=None,
             in_flight=DEFAULT_IN_FLIGHT, search=PLAIN):
    """``train.generate`` on the pool: same schedule, same seeds."""
    rules_list = [r for r in rule_sets for _ in range(games_per_ruleset)]
    seeds = game_seeds(rng, len(rules_list))
    return run_pool(rules_list, seeds, [evaluate], max_turns=max_turns,
                    threads=threads, stats=stats, in_flight=in_flight,
                    search=search)


def play_match(rules, players, games, rng, max_turns=DEFAULT_MAX_TURNS,
               threads=0, stats=None, in_flight=DEFAULT_IN_FLIGHT,
               searches=None):
    """``arena.play_match`` on the pool: game *i* seats player *p* at
    ``(p + i) % n`` and wins are credited to the player, not the seat."""
    from .arena import player_of_seat, seating_for_game
    n = rules.num_players
    if len(players) != n:
        raise ValueError(f"{n}-player rules need {n} players, "
                         f"got {len(players)}")
    # One evaluator slot per distinct object: ``compare`` seats n-1 copies
    # of the incumbent, and they should share one forward per round.
    evaluators, slot_of_player = [], []
    for p in players:
        for j, e in enumerate(evaluators):
            if e is p:
                slot_of_player.append(j)
                break
        else:
            evaluators.append(p)
            slot_of_player.append(len(evaluators) - 1)
    searches = list(searches) if searches is not None else [PLAIN] * n
    search_slots = list(dict.fromkeys(searches))      # distinct, in order
    seat_maps = [seating_for_game(n, i) for i in range(games)]
    seating, search_seating = [], []
    for seat_of in seat_maps:
        ev_of_seat, search_of_seat = [0] * n, [0] * n
        for player, seat in enumerate(seat_of):
            ev_of_seat[seat] = slot_of_player[player]
            search_of_seat[seat] = search_slots.index(searches[player])
        seating.append(ev_of_seat)
        search_seating.append(search_of_seat)
    seeds = game_seeds(rng, games)
    results = run_pool([rules] * games, seeds, evaluators, seating,
                       max_turns=max_turns, threads=threads, stats=stats,
                       in_flight=in_flight, searches=search_slots,
                       search_seating=search_seating)
    wins = [0] * n
    for seat_of, r in zip(seat_maps, results):
        wins[player_of_seat(seat_of, r.winner)] += 1
    return wins
