"""Offline, fixed-size arena: one deeper-search seat vs ordinary-search seats.

All players share ONE checkpoint. A zero-expansion control still uses the exact
current-turn solver. No advisor/training changes. Serial driver, Rust turn solves.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import time
import zlib

from .arena import verdict
from .engine import GameState, RuleSet, apply_move, can_stop, random_dice, roll, stop
from .experiment import identity, write_json, file_sha256
from .portable_rng import PortableRng
from .rust_pool import game_seeds
from .self_play import DEFAULT_MAX_TURNS, TurnLimitExceeded
from .turn_search import TurnSearchConfig, WholeTurnSearch


def play_game(rules, evaluator, seed, challenger_seat, config, max_turns=DEFAULT_MAX_TURNS, start=None):
    state = start.clone() if start is not None else GameState(rules)
    if state.rules != rules or not 0 <= challenger_seat < rules.num_players:
        raise ValueError('invalid arena configuration')
    rng = PortableRng(seed)
    costs = {'search':[], 'control':[]}
    turns = 0
    while not state.game_over:
        if turns >= max_turns:
            raise TurnLimitExceeded(f'game seed {seed} exceeded {max_turns} turns')
        actor = state.active_player
        mode = 'search' if actor == challenger_seat else 'control'
        turns += 1
        if not roll(state, random_dice(rng)):
            continue
        solver = WholeTurnSearch(state, evaluator, config if mode == 'search' else TurnSearchConfig(expansions=0))
        costs[mode].append(solver.stats)
        while True:
            apply_move(state, solver.choose_move(state))
            if can_stop(state) and solver.should_stop(state):
                stop(state); break
            if not roll(state, random_dice(rng)):
                break
    return {'seed':seed, 'challenger_seat':challenger_seat, 'winner_seat':state.winner,
            'challenger_won':state.winner == challenger_seat, 'turns':turns, 'costs':costs}


def summarize(records, players):
    if not records:
        return None
    # Player zero is the challenger; other identities rotate with it.
    wins = [0]*players
    for r in records:
        wins[(r['winner_seat']-r['challenger_seat']) % players] += 1
    result = verdict(wins, players)
    result['interval_method'] = 'Approximate Wilson 95%; fixed game count, balanced seats, independent game seeds'
    result['by_seat'] = [{'seat':s, 'games':sum(r['challenger_seat']==s for r in records),
        'wins':sum(r['challenger_seat']==s and r['challenger_won'] for r in records)} for s in range(players)]
    result['costs'] = {}
    for mode in ('search', 'control'):
        rows = [s for r in records for s in r['costs'][mode]]
        seconds = sorted(s['elapsed_seconds'] for s in rows)
        result['costs'][mode] = {'turns_solved':len(rows),
            'seconds_total':sum(seconds), 'seconds_mean':sum(seconds)/len(rows) if rows else 0,
            'seconds_p95':seconds[min(len(seconds)-1, int(.95*len(seconds)))] if rows else 0,
            'expansions':sum(s['expansions'] for s in rows),
            'evaluator_rows':sum(s['evaluator_rows'] for s in rows),
            'root_choices_changed':sum(s['root_choice_changed'] for s in rows),
            'peak_positions_per_turn':max((s['positions'] for s in rows), default=0)}
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--players', type=int, choices=(2,3,4), default=2)
    ap.add_argument('--extended', action='store_true')
    ap.add_argument('--blocking', action='store_true')
    ap.add_argument('--games', type=int, default=200)
    ap.add_argument('--expansions', type=int, default=4)
    ap.add_argument('--depth', type=int, default=1)
    ap.add_argument('--seconds', type=float)
    ap.add_argument('--explore-every', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20260929)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    if args.games <= 0 or args.games % args.players:
        ap.error('--games must be positive and a multiple of --players')
    if Path(args.out).exists():
        ap.error('output already exists; choose a new --out')
    try:
        config = TurnSearchConfig(args.expansions, args.depth, args.seconds, args.explore_every)
    except ValueError as e:
        ap.error(str(e))
    from .model import NetEvaluator, load_net
    rules = RuleSet.make(args.players, args.extended, args.blocking)
    evaluator = NetEvaluator(load_net(args.checkpoint, device=args.device), device=args.device)
    seed = zlib.crc32(f'turn-search-arena-v1|{args.seed}'.encode())
    seeds = game_seeds(PortableRng(seed), args.games)
    report = {'meta':identity(nets={'shared':args.checkpoint}, rules=str(rules),
              search=asdict(config), control=asdict(TurnSearchConfig(expansions=0)),
              seed=seed, requested_seed=args.seed, planned_games=args.games, device=args.device,
              source_sha256={name:file_sha256(Path(__file__).parent/name) for name in
                  ('turn_search.py', 'search_arena.py', 'cantstop_rust/src/lib.rs')}),
              'status':'running', 'games':[], 'result':None}
    write_json(args.out, report)
    started = time.perf_counter()
    try:
        # Warm up inference/allocator before measuring either competitor.
        warmup_state = GameState(rules)
        roll(warmup_state, (1,2,3,4))
        WholeTurnSearch(warmup_state, evaluator, TurnSearchConfig(expansions=0))
        report['warmup_seconds'] = time.perf_counter()-started
        started = time.perf_counter()
        for i, game_seed in enumerate(seeds):
            record = play_game(rules, evaluator, game_seed, i % args.players, config)
            report['games'].append(record)
            report['elapsed_seconds'] = time.perf_counter()-started
            write_json(args.out, report)
            print(f'{i+1}/{args.games} games complete ({report["elapsed_seconds"]:.1f}s)', flush=True)
    except BaseException as e:
        report['status'] = 'incomplete'
        report['error'] = f'{type(e).__name__}: {e}'
        write_json(args.out, report)
        raise
    report['status'] = 'complete'
    report['result'] = summarize(report['games'], args.players)
    write_json(args.out, report)
    r = report['result']
    print(f'Search win rate {r["win_rate"]:.2%}; 95% CI {r["ci95"]}; even {1/args.players:.2%}')


if __name__ == '__main__':
    main()
