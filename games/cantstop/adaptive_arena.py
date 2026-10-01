"""Seat-balanced arena for adaptive decisions versus the unchanged baseline."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

from .adaptive_search import AdaptiveBackend,AdaptiveConfig,atomic_json
from .arena import verdict
from .decision_compare import load_suite
from .engine import GameState,RuleSet,Phase,roll,random_dice,apply_move,stop
from .experiment import identity
from .portable_rng import PortableRng
from .progressive_search import ProgressiveConfig,ProgressiveBackend,EarlyTurnPolicy
from .rollout_search import RolloutConfig,RolloutBackend
from .rust_solver import RustTurnSolver
from .snapshot import from_snapshot


class ArenaEvaluationFailure(RuntimeError):
    """Preserve partial-game diagnostics without inventing a game result."""
    def __init__(self, message, game):
        super().__init__(message)
        self.game = game


def finalize_report(report):
    """Withhold strength results for failed, cancelled, or incomplete runs."""
    from collections import Counter
    decisions = [d for game in report['games'] for d in game['decisions']]
    decisions += report.get('failed_game', {}).get('decisions', [])
    reasons = Counter(d['stop_reason'] for d in decisions)
    fallbacks = Counter(d['stop_reason'] for d in decisions if d.get('fallback'))
    report['decision_summary'] = {'decisions':len(decisions),
        'stop_reasons':dict(reasons), 'fallbacks_by_reason':dict(fallbacks),
        'evaluation_failures':reasons['evaluation_failure']}
    report['result'] = None
    if reasons['evaluation_failure'] or report.get('failed_game'):
        report['status'] = 'incomplete'
        report['verdict_withheld'] = 'evaluation_failure'
    elif report['status'] != 'complete':
        report['verdict_withheld'] = 'incomplete_run'
    elif reasons['cancelled']:
        report['status'] = 'incomplete'
        report['verdict_withheld'] = 'cancelled'
    elif report['meta']['start_fixture'] is not None:
        report['verdict_withheld'] = 'fixture_smoke'
    else:
        players = report['meta']['rules']['num_players']
        wins = [0]*players
        for game in report['games']:
            wins[(game['winner_seat']-game['challenger_seat'])%players] += 1
        report['result'] = verdict(wins,players)
        report.pop('verdict_withheld',None)


def play_game(rules,evaluator,seed,challenger,config,rollout,continuation=ProgressiveConfig(),
              *,start=None,seconds=None,max_turns=1000,backend_factory=None):
    state=GameState(rules) if start is None else start.clone()
    if state.rules!=rules: raise ValueError('start fixture rules differ')
    if not 0<=challenger<rules.num_players: raise ValueError('invalid challenger seat')
    rng=PortableRng(seed); decisions=[]; turns=0
    backend=(AdaptiveBackend(evaluator,config,rollout,continuation) if backend_factory is None else backend_factory())
    while not state.game_over:
        if turns>=max_turns: raise RuntimeError('arena turn limit exceeded')
        actor=state.active_player; turns+=1
        baseline=None
        while not state.game_over and state.active_player==actor:
            if state.phase==Phase.AWAIT_ROLL:
                if not roll(state,random_dice(rng)): break
            if actor==challenger:
                try:
                    decision=(backend.evaluate(state,seconds=seconds) if isinstance(backend,AdaptiveBackend) else backend.evaluate(state))
                except Exception as exc:
                    stats=backend.last_stats or {}
                    decisions.append({'phase':int(state.phase),'action':None,
                        'seconds':stats.get('elapsed_seconds',0),'stop_reason':'evaluation_failure',
                        'fallback':False,'samples':sum(stats.get('committed_samples',{}).values()),
                        'error':f'{type(exc).__name__}: {exc}'})
                    raise ArenaEvaluationFailure(str(exc),{'seed':seed,'challenger_seat':challenger,
                        'turns':turns,'decisions':decisions}) from exc
                stats=backend.last_stats
                if isinstance(backend,AdaptiveBackend):
                    samples=sum(stats['committed_samples'].values())
                elif isinstance(backend,ProgressiveBackend):
                    samples=sum(a['samples'] for stage in stats['stages'] for a in stage['rollout']['actions'])
                else:
                    samples=sum(a['samples'] for a in stats['actions'])
                decisions.append({'phase':int(state.phase),'action':decision.selected.key,
                    'seconds':stats['elapsed_seconds'],'stop_reason':stats.get('stop_reason',stats['status']),
                    'fallback':stats.get('fallback',False),'samples':samples,
                    'error':stats.get('error')})
                if stats.get('stop_reason') == 'evaluation_failure':
                    raise ArenaEvaluationFailure(stats.get('error','evaluation failure'),
                        {'seed':seed,'challenger_seat':challenger,'turns':turns,'decisions':decisions})
                action=decision.selected
                if action.kind=='move': apply_move(state,action.move)
                elif action.kind=='stop': stop(state)
                elif not roll(state,random_dice(rng)): break
            else:
                if baseline is None: baseline=RustTurnSolver(state,evaluator)
                if state.phase==Phase.AWAIT_MOVE: apply_move(state,baseline.choose_move(state))
                elif baseline.should_stop(state): stop(state)
                elif not roll(state,random_dice(rng)): break
    return {'seed':seed,'challenger_seat':challenger,'winner_seat':state.winner,
            'challenger_won':state.winner==challenger,'turns':turns,'decisions':decisions}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--players',type=int,choices=(2,3,4),default=2)
    p.add_argument('--extended',action='store_true'); p.add_argument('--blocking',action='store_true')
    p.add_argument('--backend',choices=('adaptive','progressive','rollout'),default='adaptive')
    p.add_argument('--samples',type=int,default=32)
    p.add_argument('--stage-horizons',type=int,nargs='+',default=[0,1])
    p.add_argument('--stage-samples',type=int,nargs='+',default=[8,32])
    p.add_argument('--margin',type=float)
    p.add_argument('--max-candidates',type=int)
    p.add_argument('--games',type=int,default=200)
    p.add_argument('--confidence',choices=('paired_t','hoeffding'),default='paired_t',
                   help='paired_t: approximate corrected-sample intervals; hoeffding: bounded raw intervals')
    p.add_argument('--budgets',type=int,nargs='+',default=[32,128,512])
    p.add_argument('--horizon',type=int,default=1)
    p.add_argument('--dice-luck',action=argparse.BooleanOptionalAction,default=True); p.add_argument('--common-random-numbers',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--early-turns',type=int,default=0); p.add_argument('--early-expansions',type=int,default=0)
    p.add_argument('--seconds',type=float)
    p.add_argument('--seed',type=int,default=20260930)
    p.add_argument('--start-fixture',help='Integration smoke only: disables strength verdict')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    if args.out.exists(): p.error('output exists')
    if args.seconds is not None and args.backend!='adaptive': p.error('--seconds requires adaptive backend')
    if args.games<=0 or args.games%args.players: p.error('games must be a positive multiple of players')
    rules=RuleSet.make(args.players,args.extended,args.blocking)
    start=None
    if args.start_fixture:
        matches=[r for r in load_suite() if r['id']==args.start_fixture]
        if not matches: p.error('unknown fixture')
        start=from_snapshot(matches[0]['snapshot'])
        if start.rules!=rules: p.error('fixture rules differ from requested arena')
    cfg=AdaptiveConfig(tuple(args.budgets),confidence=args.confidence)
    rc=RolloutConfig(samples=args.samples,horizon=args.horizon,seed=args.seed,dice_luck=args.dice_luck,common_random_numbers=args.common_random_numbers)
    continuation=ProgressiveConfig(early_turns=args.early_turns,expansions=args.early_expansions,
        horizons=tuple(args.stage_horizons),samples=tuple(args.stage_samples),margin=args.margin,max_candidates=args.max_candidates)
    from .model import NetEvaluator,load_net
    evaluator=NetEvaluator(load_net(args.checkpoint,device=args.device),device=args.device)
    rng=PortableRng(args.seed)
    def factory():
        if args.backend=='progressive': return ProgressiveBackend(evaluator,continuation,rc)
        if args.backend=='rollout': return RolloutBackend(evaluator,rc,policy_factory=EarlyTurnPolicy(evaluator,continuation))
        return AdaptiveBackend(evaluator,cfg,rc,continuation)
    report={'status':'running','meta':identity(nets={'shared':args.checkpoint},rules=asdict(rules),
        backend=args.backend,adaptive=asdict(cfg),rollout=asdict(rc),continuation=asdict(continuation),
        seconds=args.seconds,planned_games=args.games,start_fixture=args.start_fixture),
        'games':[],'result':None}
    atomic_json(args.out,report)
    began=time.perf_counter()
    try:
        for i in range(args.games):
            report['games'].append(play_game(rules,evaluator,rng.next_u64(),i%args.players,cfg,rc,continuation,
                                            start=start,seconds=args.seconds,backend_factory=factory))
            atomic_json(args.out,report)
            print(f'{i+1}/{args.games} games complete',flush=True)
        report['status']='complete'
    except BaseException as exc:
        if isinstance(exc,ArenaEvaluationFailure): report['failed_game']=exc.game
        report['status']='incomplete'; report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        finalize_report(report)
        report['elapsed_seconds']=time.perf_counter()-began
        atomic_json(args.out,report)


if __name__=='__main__': main()
