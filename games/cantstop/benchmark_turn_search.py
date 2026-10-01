"""CPU-only benchmark against the frozen original turn-search controller.

Use fixed budgets. Verifies identical traces/values before reporting timings.
The heuristic isolates controller cost without sharing a running match's GPU.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

from .advisor_adapter import parse_state
from .experiment import file_sha256, write_json
from .solver import ProgressHeuristic
from .turn_search import WholeTurnSearch, TurnSearchConfig
from .tests.turn_search_reference import WholeTurnSearch as Reference
from .tests.turn_search_numpy_reference import WholeTurnSearch as NumpyReference


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', default=str(Path(__file__).parent/'tests/fixtures/bga_923128580_stale_scores.json'))
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    if args.repeats < 1: ap.error('--repeats must be positive')
    if Path(args.out).exists(): ap.error('choose a new output path')
    state = parse_state(json.loads(Path(args.capture).read_text()))
    evaluator = ProgressHeuristic()
    report = {'evaluator':'CPU ProgressHeuristic, not NN', 'capture':args.capture,
              'capture_sha256':file_sha256(args.capture), 'repeats':args.repeats,
              'source_sha256':{name:file_sha256(Path(__file__).parent/name) for name in
                  ('turn_search.py','tests/turn_search_reference.py',
                   'tests/turn_search_numpy_reference.py','cantstop_rust/src/lib.rs')}, 'results':[]}
    # Untimed warmup of the shared native code.
    WholeTurnSearch(state, evaluator, TurnSearchConfig(expansions=0))
    for k, depth in [(4,1),(8,1),(8,2)]:
        config = TurnSearchConfig(expansions=k, depth=depth)
        samples = {'original':[], 'numpy':[], 'optimized':[]}
        expected = None
        for repeat in range(args.repeats):
            order = [('original',Reference),('numpy',NumpyReference),('optimized',WholeTurnSearch)]
            if repeat % 2: order.reverse()
            for name, cls in order:
                gc.collect()
                start = time.perf_counter()
                search = cls(state, evaluator, config)
                elapsed = time.perf_counter()-start
                signature = (search.trace, search.value(state).tolist(), search.choose_move(state))
                if expected is None: expected = signature
                assert signature == expected, 'controller changed search behavior'
                samples[name].append(elapsed)
                del search
        medians = {name:statistics.median(v) for name,v in samples.items()}
        result = {'expansions':k, 'depth':depth, 'seconds':samples, 'median_seconds':medians,
                  'speedup':medians['original']/medians['optimized'],
                  'speedup_over_numpy':medians['numpy']/medians['optimized'], 'exact_equivalence':True}
        report['results'].append(result)
        print(f'k={k} depth={depth}: {medians["original"]:.3f}s -> {medians["optimized"]:.3f}s ({result["speedup"]:.2f}x original; {result["speedup_over_numpy"]:.2f}x NumPy)', flush=True)
    write_json(args.out, report)


if __name__ == '__main__':
    main()
