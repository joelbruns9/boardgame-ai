# self_play.py
# MCTS self-play training loop for Can't Stop.
#
# Key design:
#   - MCTS provides training targets (visit counts + values)
#   - Workers generate games on CPU in parallel
#   - Main process trains on GPU
#   - Replay buffer prevents catastrophic forgetting
#   - Always-accept-with-floor acceptance (>=45% vs current)
#   - Best model checkpoint maintained at a stable path
#   - Evaluation uses MCTS at low sim count (matches deployment behavior)

import os
import sys
import time
import random
import argparse
import numpy as np

# Windows + redirected stdout defaults to cp1252, which can't encode the
# unicode box-drawing characters (─ ✓ ✗ ⚠ ◐) used in our logs. Reconfigure
# stdout/stderr to UTF-8 with error replacement so log files always work.
# Has no effect on terminals that already use UTF-8.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    # reconfigure() was added in Python 3.7; older versions skip silently.
    pass

from collections import deque, defaultdict
from datetime import datetime
import multiprocessing as mp

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, get_valid_moves, apply_move,
    stop_turn, bust_turn
)
from games.cantstop.features import (
    extract_features, get_legal_action_mask,
    move_to_action, action_to_move_decision,
    FEATURE_SIZE, ACTION_SPACE
)
from games.cantstop.model import CantStopNet
from games.cantstop.mcts import MCTS, BatchedSyncSearch
from games.cantstop.evaluate import load_model
from games.cantstop.inference_server import (
    InferenceServerManager, InferenceClient, ThreadSafeInferenceClient,
)


# ---- PHASE-BASED TEMPERATURE ----

def get_temperature(step_index, state, global_temp_mult=1.0):
    """
    Phase-based temperature for MCTS action selection.
    Early game: more random → diverse positions
    Late game:  near-greedy → precise endgame play
    """
    max_score = max(
        len(state.claimed[0]),
        len(state.claimed[1])
    )
    if max_score >= 2:
        base_temp = 0.3
    elif max_score == 1 or step_index > 15:
        base_temp = 0.7
    else:
        base_temp = 1.0
    return base_temp * global_temp_mult


# ---- SELF-PLAY GAME WITH MCTS ----


def play_mcts_game(mcts, num_simulations=20, global_temp_mult=1.0,
                   game_id=None):
    """
    Play one complete self-play game using MCTS for decisions.

    For each position:
      1. Run MCTS (guided by network)
      2. Record: features, MCTS visit counts (policy target),
                 MCTS value estimate (value target)
      3. Sample action from MCTS distribution with temperature

    Returns list of labeled training records, each tagged with
    `game_id` so the trainer can split train/val by game (avoids
    correlated-position leakage across the split).
    """
    state = GameState(2)
    records = []
    step_index = 0
    max_turns = 200

    for _ in range(max_turns):
        if state.game_over:
            break

        if not state.dice:
            state.roll_dice()

        valid = get_valid_moves(state)

        if not valid:
            bust_turn(state)
            state.dice = []
            continue

        player = state.active_player
        temp = get_temperature(step_index, state, global_temp_mult)

        # MCTS search — this is the key training signal.
        action_idx, move, decision, mcts_policy, mcts_value = \
            mcts.get_action(
                state,
                num_simulations=num_simulations,
                temperature=temp
            )

        # Record position with MCTS targets.
        features = extract_features(state, valid)
        mask     = get_legal_action_mask(valid)

        records.append({
            'features':    features,
            'mask':        mask,
            'mcts_policy': mcts_policy,
            'mcts_value':  mcts_value,
            'action_idx':  action_idx,
            'player':      player,
            'step_index':  step_index,
            'game_id':     game_id,
        })

        step_index += 1

        apply_move(state, move)
        if decision == "stop":
            stop_turn(state)
            state.dice = []
        else:
            # CONTINUE → the current dice are spent. Clear them so the
            # next iteration rolls fresh. (Original code left dice set,
            # which made the next iteration's get_valid_moves operate on
            # stale dice — a silent correctness bug.)
            state.dice = []

    # Fill in value targets — blend MCTS estimate with final outcome.
    winner = state.winner
    lambda_blend = 0.7
    labeled = []
    for rec in records:
        final_outcome = 1.0 if rec['player'] == winner else 0.0
        rec['value_target'] = (
            lambda_blend * final_outcome +
            (1 - lambda_blend) * rec['mcts_value']
        )
        labeled.append(rec)

    return labeled




# ---- FIX C: BATCHED SYNC SELF-PLAY GAME POOL ----

class _BatchedGameRunner:
    """
    One live self-play game managed by the Fix C worker coordinator.

    The runner owns exactly one GameState and at most one current
    BatchedSyncSearch for the position where a decision is needed. The
    coordinator keeps many runners alive at once and batches the pending
    eval chunks from all their current searches into one inference call.
    """

    __slots__ = [
        'mcts', 'num_simulations', 'global_temp_mult', 'game_id',
        'state', 'records', 'step_index', 'turns', 'max_turns', 'done',
        'search', 'pending_player', 'pending_temp', 'pending_valid',
        'pending_features', 'pending_mask',
    ]

    def __init__(self, mcts, num_simulations, global_temp_mult, game_id):
        self.mcts = mcts
        self.num_simulations = num_simulations
        self.global_temp_mult = global_temp_mult
        self.game_id = game_id

        self.state = GameState(2)
        self.records = []
        self.step_index = 0
        self.turns = 0
        self.max_turns = 200
        self.done = False

        self.search = None
        self.pending_player = None
        self.pending_temp = None
        self.pending_valid = None
        self.pending_features = None
        self.pending_mask = None

    def advance_until_search_needed(self):
        """
        Advance cheap game mechanics until this runner either needs MCTS
        for a legal decision, already has an active search, or finishes.
        """
        while not self.done and self.search is None:
            if self.state.game_over or self.turns >= self.max_turns:
                self.done = True
                return

            self.turns += 1

            if not self.state.dice:
                self.state.roll_dice()

            valid = get_valid_moves(self.state)
            if not valid:
                bust_turn(self.state)
                self.state.dice = []
                continue

            self.pending_player = self.state.active_player
            self.pending_temp = get_temperature(
                self.step_index, self.state, self.global_temp_mult
            )
            self.pending_valid = valid
            self.pending_features = extract_features(self.state, valid)
            self.pending_mask = get_legal_action_mask(valid)
            self.search = BatchedSyncSearch(
                self.mcts,
                self.state,
                num_simulations=self.num_simulations,
                # Match normal self-play MCTS.get_action/search defaults.
                dirichlet_alpha=0.5,
                dirichlet_epsilon=0.25,
                batch_size_cap=8,
            )
            return

    def search_done(self):
        return self.search is not None and self.search.is_done()

    def finish_search_and_apply(self):
        """Consume a completed search, record target row, and apply move."""
        if self.search is None or not self.search.is_done():
            return

        mcts_policy, mcts_value = self.search.result()
        temp = self.pending_temp

        if temp <= 0.01:
            action_idx = int(mcts_policy.argmax())
        else:
            visits_temp = mcts_policy ** (1.0 / temp)
            total = visits_temp.sum()
            if total > 0:
                visits_temp = visits_temp / total
                action_idx = int(np.random.choice(ACTION_SPACE, p=visits_temp))
            else:
                action_idx = int(mcts_policy.argmax())

        move, decision = action_to_move_decision(int(action_idx))

        self.records.append({
            'features':    self.pending_features,
            'mask':        self.pending_mask,
            'mcts_policy': mcts_policy,
            'mcts_value':  mcts_value,
            'action_idx':  action_idx,
            'player':      self.pending_player,
            'step_index':  self.step_index,
            'game_id':     self.game_id,
        })

        self.step_index += 1
        apply_move(self.state, move)
        if decision == "stop":
            stop_turn(self.state)
            self.state.dice = []
        else:
            self.state.dice = []

        self.search = None
        self.pending_player = None
        self.pending_temp = None
        self.pending_valid = None
        self.pending_features = None
        self.pending_mask = None

    def labeled_records(self):
        """Return records with final blended value targets filled in."""
        winner = self.state.winner
        lambda_blend = 0.7
        labeled = []
        for rec in self.records:
            final_outcome = 1.0 if rec['player'] == winner else 0.0
            rec['value_target'] = (
                lambda_blend * final_outcome +
                (1 - lambda_blend) * rec['mcts_value']
            )
            labeled.append(rec)
        return labeled


def play_mcts_games_batched_sync(mcts, num_games, num_simulations=20,
                                 global_temp_mult=1.0,
                                 batch_sync_searches=8):
    """
    Fix C: play many self-play games with per-search sync MCTS semantics,
    but batch NN eval chunks across independent active games/searches.

    Unlike within-tree async, this never has multiple concurrent sims inside
    the same MCTS root. Each BatchedSyncSearch preserves the legacy
    target_inflight=1 chunk semantics. The only batching gain comes from
    concatenating eval leaves from several independent search roots.
    """
    width = max(1, int(batch_sync_searches))
    records_out = []
    runners = []
    games_started = 0
    games_finished = 0

    # Lightweight diagnostics for this worker. Printed only once per worker
    # batch so small tests can verify Fix C is really coordinating searches.
    coord_calls = 0
    coord_samples = 0
    coord_searches = 0
    coord_max_samples = 0

    def _start_runner():
        nonlocal games_started
        gid = _next_game_id()
        runner = _BatchedGameRunner(
            mcts=mcts,
            num_simulations=num_simulations,
            global_temp_mult=global_temp_mult,
            game_id=gid,
        )
        games_started += 1
        return runner

    while games_finished < num_games:
        # Keep a pool of live independent games. This is the core Fix C
        # behavior missing from the previous attempt.
        while len(runners) < width and games_started < num_games:
            runners.append(_start_runner())

        # Advance every runner until it has an active search, is done, or is
        # waiting for the next coordinator eval batch.
        for r in runners:
            r.advance_until_search_needed()

        # Consume completed searches and immediately advance those games to
        # their next decision point if possible. This keeps the pool full.
        progressed = True
        while progressed:
            progressed = False
            for r in runners:
                if r.search_done():
                    r.finish_search_and_apply()
                    r.advance_until_search_needed()
                    progressed = True

        # Remove finished games and replace them below.
        still_live = []
        for r in runners:
            if r.done:
                records_out.extend(r.labeled_records())
                games_finished += 1
            else:
                still_live.append(r)
        runners = still_live

        if games_finished >= num_games:
            break
        if not runners:
            continue

        # Ask each active search for its next legacy sync eval chunk, then
        # merge all chunks into one remote inference call.
        all_nodes = []
        chunks = []  # (search, n_nodes)
        for r in runners:
            if r.search is None or r.search.is_done():
                continue
            nodes = r.search.prepare_next_eval()
            if nodes:
                chunks.append((r.search, len(nodes)))
                all_nodes.extend(nodes)

        # Some prepare_next_eval() calls may have finished terminal-only
        # chunks without needing NN eval. Loop back to consume them.
        if not all_nodes:
            continue

        results = mcts.evaluate_batch(all_nodes)
        coord_calls += 1
        coord_samples += len(all_nodes)
        coord_searches += len(chunks)
        if len(all_nodes) > coord_max_samples:
            coord_max_samples = len(all_nodes)

        offset = 0
        for search, n in chunks:
            search.apply_eval_results(results[offset:offset + n])
            offset += n

    coord_stats = {
        'batch_sync_calls': coord_calls,
        'batch_sync_samples': coord_samples,
        'batch_sync_searches': coord_searches,
        'batch_sync_max_samples': coord_max_samples,
        'batch_sync_width': width,
    }

    return records_out, coord_stats


# ---- PARALLEL WORKER FUNCTIONS ----
# Must be top-level for multiprocessing pickling on Windows.
#
# Workers no longer load a model. Instead, each worker holds a shared
# ThreadSafeInferenceClient wired to the GPU inference server. The
# server is owned and managed by the main process and runs on CUDA.
# Each worker can run N game threads concurrently sharing one client
# (see --games-per-worker). When one thread is blocked on inference,
# other threads can use the CPU.


# Worker-process globals — populated by _init_worker, consumed by
# _worker_generate_batch. Per-process, not per-thread.
_worker_client = None             # ThreadSafeInferenceClient (shared by threads)
_worker_mcts_kwargs = None        # dict of MCTS kwargs for per-thread instances
_worker_game_counter = 0          # atomic via _worker_game_counter_lock below
_worker_game_counter_lock = None  # threading.Lock — created in _init_worker
_worker_id_prefix = 0


def _init_worker(request_queue, response_queues, worker_counter,
                 iteration_seed, target_inflight, warmup_sims):
    """
    Initialize worker process.

    Atomically claims a worker_id (0..num_workers-1) via the shared
    `worker_counter` Value. Builds ONE ThreadSafeInferenceClient bound to:
      - the shared request_queue (all workers push to one queue)
      - this worker's private response_queue (response_queues[worker_id])

    The client is shared across N game threads (set by --games-per-worker).
    The client owns a dispatcher thread that routes responses by request_id
    so threads can call infer() concurrently without colliding.

    Each game thread gets its OWN MCTS instance pointing at this shared
    client — separate trees, separate per-search statistics, no cross-
    thread MCTS interaction. Quality of MCTS targets is unchanged.

    `target_inflight` and `warmup_sims` are stored for game-thread use.
    """
    global _worker_client, _worker_mcts_kwargs
    global _worker_game_counter, _worker_game_counter_lock
    global _worker_id_prefix

    # Atomically claim a worker_id. The shared Value is created by the
    # parent so this is safe across worker processes.
    with worker_counter.get_lock():
        worker_id = worker_counter.value
        worker_counter.value += 1

    if worker_id >= len(response_queues):
        # Defensive: if pool spawns more workers than expected (e.g.
        # maxtasksperchild causing recycling), we'd run out of response
        # queues. This shouldn't happen with our config, but bail
        # cleanly if it does.
        raise RuntimeError(
            f"Worker_id {worker_id} exceeds response_queues count "
            f"{len(response_queues)}. Pool worker recycling not supported."
        )

    # ONE thread-safe client per worker, shared across game threads.
    _worker_client = ThreadSafeInferenceClient(
        request_queue=request_queue,
        response_queue=response_queues[worker_id],
        worker_id=worker_id,
    )
    _worker_client.start()

    # MCTS kwargs — used to build a fresh MCTS in each game thread.
    _worker_mcts_kwargs = dict(
        target_inflight=target_inflight,
        warmup_sims=warmup_sims,
    )

    # Per-thread game ID counter, with a lock for atomic increment.
    import threading as _t
    _worker_game_counter = 0
    _worker_game_counter_lock = _t.Lock()

    # game_id prefix: iteration_seed[16] | worker_id[16]
    iter_bits = (int(iteration_seed) & 0xFFFF) << 16
    wid_bits  = (worker_id & 0xFFFF)
    _worker_id_prefix = iter_bits | wid_bits


def _next_game_id():
    """Atomically allocate the next game_id within this worker."""
    global _worker_game_counter
    with _worker_game_counter_lock:
        gid = (_worker_id_prefix << 32) | (_worker_game_counter & 0xFFFFFFFF)
        _worker_game_counter += 1
    return gid


def _game_thread_body(num_games, num_simulations, temp_mult, seed,
                       result_list, result_lock, error_box,
                       coord_stats_list, batch_sync_searches=0):
    """
    Body for one game thread inside a worker process.

    Each thread:
      - Seeds its own RNGs (so different threads see different dice).
        Note: random and numpy RNGs are NOT thread-safe globally, so
        we seed per-thread. Multiple threads sharing the same generator
        without locks would corrupt internal state. The seeding here
        relies on the fact that random.seed() in Python is thread-local
        via the random module's _inst — but numpy.random.seed() sets a
        global. For our purposes, the dice rolls happen via random.choices
        (in engine.roll_dice) and numpy is used by mcts/features; the
        risk of cross-thread state corruption is real but minor for
        training data quality. If we ever need stricter isolation, swap
        to thread-local Generator instances.

      - Builds its OWN MCTS instance pointing at the shared client.
        Per-search statistics, virtual loss, warmup are all per-tree.

      - Plays games sequentially until its quota is done.

      - Appends records to result_list under result_lock.

      - On any fatal error, stores into error_box and exits.
    """
    try:
        import random as _r
        import numpy as _np

        # Per-thread RNG seeding. (See caveat above.)
        _r.seed(seed)
        _np.random.seed(seed & 0xFFFFFFFF)

        mcts = MCTS(_worker_client, 'cpu', **_worker_mcts_kwargs)

        local_records = []
        local_coord_stats = None
        if batch_sync_searches and batch_sync_searches > 1:
            # Fix C path: this thread owns a pool of active games and batches
            # their legacy-sync MCTS eval chunks together.
            records, local_coord_stats = play_mcts_games_batched_sync(
                mcts,
                num_games=num_games,
                num_simulations=num_simulations,
                global_temp_mult=temp_mult,
                batch_sync_searches=batch_sync_searches,
            )
            local_records.extend(records)
        else:
            # Original path: one complete game at a time.
            for i in range(num_games):
                try:
                    game_id = _next_game_id()
                    records = play_mcts_game(
                        mcts,
                        num_simulations=num_simulations,
                        global_temp_mult=temp_mult,
                        game_id=game_id,
                    )
                    local_records.extend(records)
                except Exception as e:
                    import traceback
                    print(f"  [worker thread] Game {i+1}/{num_games} "
                          f"failed: {e}", flush=True)
                    traceback.print_exc()
                    continue  # skip bad game, keep going

        # Splice into shared result list atomically.
        with result_lock:
            result_list.extend(local_records)
            if local_coord_stats is not None:
                coord_stats_list.append(local_coord_stats)

    except Exception as e:
        # Fatal thread error — surface to main thread via error_box.
        import traceback
        error_box.append((e, traceback.format_exc()))


def _worker_generate_batch(args):
    """
    Generate a batch of MCTS games in a worker process.

    With games_per_worker=N, spawns N threads that share one
    ThreadSafeInferenceClient. Each thread runs its share of games
    sequentially. When one thread is blocked on inference, others
    can use the CPU — overlapping the CPU/GPU phases that previously
    serialized.

    Returns a tuple (records, stats):
        records: list of training records from all games
        stats:   dict with timing breakdown (per-worker totals)

    Returns ([], {}) on fatal error so the pool doesn't hang.
    """
    import time as _t
    import threading as _th

    (num_games, num_simulations, temp_mult, seed,
     games_per_worker, batch_sync_searches) = args

    worker_t0 = _t.perf_counter()

    # Distribute games among threads. Threads get a contiguous range
    # of game IDs each; if the work doesn't divide evenly, earlier
    # threads get one extra game.
    games_per_worker = max(1, int(games_per_worker))
    base = num_games // games_per_worker
    leftover = num_games % games_per_worker
    thread_game_counts = [
        base + (1 if i < leftover else 0)
        for i in range(games_per_worker)
    ]

    # Per-thread seeds derived from the worker seed — independent
    # dice across threads.
    thread_seeds = [(seed * 1000003 + i * 9176) & 0x7FFFFFFF
                    for i in range(games_per_worker)]

    # Shared collection state.
    result_list = []
    result_lock = _th.Lock()
    error_box = []   # thread errors surface here
    coord_stats_list = []

    threads = []
    for i in range(games_per_worker):
        if thread_game_counts[i] <= 0:
            continue
        t = _th.Thread(
            target=_game_thread_body,
            args=(thread_game_counts[i], num_simulations,
                  temp_mult, thread_seeds[i],
                  result_list, result_lock, error_box,
                  coord_stats_list, batch_sync_searches),
            name=f"game-w{_worker_id_prefix & 0xFFFF}-t{i}",
            daemon=False,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    if error_box:
        # At least one thread had a fatal error — report and continue.
        for exc, tb in error_box:
            print(f"  [worker] game thread fatal: {exc}\n{tb}",
                  flush=True)

    worker_t1 = _t.perf_counter()

    batch_sync_calls = sum(s.get('batch_sync_calls', 0) for s in coord_stats_list)
    batch_sync_samples = sum(s.get('batch_sync_samples', 0) for s in coord_stats_list)
    batch_sync_searches_total = sum(s.get('batch_sync_searches', 0) for s in coord_stats_list)
    batch_sync_max_samples = max(
        [s.get('batch_sync_max_samples', 0) for s in coord_stats_list] or [0]
    )
    batch_sync_width = max(
        [s.get('batch_sync_width', 0) for s in coord_stats_list] or [0]
    )

    # Pull final stats from the shared client. Stats are accumulated
    # across all calls from all threads in this worker.
    if _worker_client is not None and hasattr(_worker_client, 'get_stats'):
        cs = _worker_client.get_stats()
        stats = {
            'wall_s':        worker_t1 - worker_t0,
            'blocked_s':     cs['blocked_s'],
            'put_s':         cs['put_s'],
            'n_calls':       cs['n_calls'],
            'total_samples': cs['total_samples'],
            'records':       len(result_list),
            'games_per_worker': games_per_worker,
            'batch_sync_searches': batch_sync_searches,
            'batch_sync_calls': batch_sync_calls,
            'batch_sync_samples': batch_sync_samples,
            'batch_sync_searches_total': batch_sync_searches_total,
            'batch_sync_max_samples': batch_sync_max_samples,
            'batch_sync_width': batch_sync_width,
        }
    else:
        stats = {
            'wall_s':        worker_t1 - worker_t0,
            'blocked_s':     0.0,
            'put_s':         0.0,
            'n_calls':       0,
            'total_samples': 0,
            'records':       len(result_list),
            'games_per_worker': games_per_worker,
            'batch_sync_searches': batch_sync_searches,
            'batch_sync_calls': batch_sync_calls,
            'batch_sync_samples': batch_sync_samples,
            'batch_sync_searches_total': batch_sync_searches_total,
            'batch_sync_max_samples': batch_sync_max_samples,
            'batch_sync_width': batch_sync_width,
        }

    # Shut down the client's dispatcher thread cleanly. The worker
    # process is about to exit (pool.map returns), but explicit shutdown
    # is cleaner and surfaces any lingering errors.
    if _worker_client is not None and hasattr(_worker_client, 'shutdown'):
        try:
            _worker_client.shutdown(timeout=2.0)
        except Exception:
            pass  # best-effort; pool exits anyway

    return result_list, stats


def generate_games_parallel(server_manager, num_games, num_simulations,
                             temp_mult, num_workers=None,
                             iteration_seed=None,
                             target_inflight=16, warmup_sims=16,
                             games_per_worker=2,
                             batch_sync_searches=0):
    """
    Generate MCTS games across multiple CPU worker processes.
    Workers send inference requests to the GPU server held by the
    server_manager (main process owns it).

    The server must already be running when this is called. The caller
    (self_play_loop) is responsible for server lifecycle — start before
    the first iteration, update_model() between iterations, stop after
    the last iteration.

    `iteration_seed` ensures different iterations produce different
    dice sequences and unique game_id prefixes.

    `target_inflight` and `warmup_sims` are forwarded to each worker's
    MCTS instance. See _init_worker for the rationale.

    `games_per_worker` controls how many games each worker process plays
    CONCURRENTLY using threads. When >1, the worker holds N game threads
    that share one ThreadSafeInferenceClient. While one thread is
    blocked waiting on inference, others can use the CPU — overlapping
    the CPU/GPU phases that previously serialized. Default 2; the
    measurement data showed ~78% of worker wall time was blocked on
    inference, so 2 threads should recover most of that.
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)

    if num_workers > len(server_manager.response_queues):
        raise ValueError(
            f"num_workers ({num_workers}) exceeds server manager's "
            f"configured response_queues ({len(server_manager.response_queues)}). "
            f"Recreate the server with a larger num_workers."
        )

    # Quota — how many games each worker process plays in total.
    # Worker then splits its quota across its N game threads.
    quota_per_worker = num_games // num_workers
    leftover = num_games % num_workers

    if iteration_seed is None:
        iteration_seed = int(time.time() * 1000) & 0x7FFFFFFF

    rng = random.Random(iteration_seed)
    batch_args = []
    for i in range(num_workers):
        n = quota_per_worker + (1 if i < leftover else 0)
        if n > 0:
            seed = rng.randint(0, 2**31 - 1)
            batch_args.append(
                (n, num_simulations, temp_mult, seed, games_per_worker,
                 batch_sync_searches)
            )

    actual_workers = len(batch_args)

    # Worker counter — atomic across worker processes — assigns each
    # worker its unique worker_id during _init_worker.
    ctx = server_manager.ctx
    worker_counter = ctx.Value('i', 0)

    with ctx.Pool(
        processes=actual_workers,
        initializer=_init_worker,
        initargs=(
            server_manager.request_queue,
            server_manager.response_queues,
            worker_counter,
            iteration_seed,
            target_inflight,
            warmup_sims,
        ),
    ) as pool:
        results = pool.map(_worker_generate_batch, batch_args)

    all_records = []
    worker_stats = []
    for batch in results:
        # New protocol: workers return (records, stats). Be tolerant if
        # we somehow receive an old-style bare list.
        if isinstance(batch, tuple) and len(batch) == 2:
            recs, stats = batch
        else:
            recs, stats = batch, {}
        all_records.extend(recs)
        if stats:
            worker_stats.append(stats)

    # ---- Aggregate timing breakdown ----
    if worker_stats:
        n_workers = len(worker_stats)
        wall_total    = sum(s['wall_s']    for s in worker_stats)
        blocked_total = sum(s['blocked_s'] for s in worker_stats)
        put_total     = sum(s['put_s']     for s in worker_stats)
        n_calls       = sum(s['n_calls']   for s in worker_stats)
        total_samples = sum(s['total_samples'] for s in worker_stats)

        # When a worker has multiple game threads, blocked_total
        # accumulates time across ALL threads (the wrapper's stats
        # don't know which thread called). Average per-thread blocked
        # time is what tells us about contention; the overall worker
        # CPU utilization comes from a different formula entirely.
        gpw_values = [s.get('games_per_worker', 1) for s in worker_stats]
        avg_gpw = sum(gpw_values) / n_workers if n_workers > 0 else 1
        n_threads_total = sum(gpw_values)  # total game threads across all workers

        # Per-worker wall time.
        avg_wall = wall_total / n_workers
        avg_put  = put_total  / n_workers

        # Per-THREAD blocked time. Normalize blocked_total by number
        # of threads, not workers. With N threads per worker each
        # spending B fraction of time blocked, blocked_total ≈ N × wall × B.
        avg_thread_blocked = blocked_total / max(n_threads_total, 1)
        if avg_wall > 0:
            pct_thread_blocked = 100.0 * avg_thread_blocked / avg_wall
        else:
            pct_thread_blocked = 0.0

        # Aggregate CPU utilization: how much CPU work did we do, across
        # all threads, as a fraction of the wall-clock × thread-count
        # budget? "100%" would mean every thread did CPU work for the
        # entire wall time (impossible — must wait on GPU at least
        # sometimes). With 1 thread/worker and 79% blocked, util = 21%.
        # With 2 threads/worker and equal blocking, util might be 30-50%
        # depending on how well threads cover each others' waits.
        thread_time_budget = wall_total * 1.0   # wall × (avg threads / workers)
        # Actually we want: total CPU time / total wall × thread-count budget
        # Total CPU = thread_count × wall − blocked_total (only true when
        # each thread runs to completion within wall time, which is the case)
        total_thread_seconds = sum(s['wall_s'] * s.get('games_per_worker', 1)
                                    for s in worker_stats)
        total_cpu_seconds    = total_thread_seconds - blocked_total
        if total_thread_seconds > 0:
            pct_cpu_util = 100.0 * total_cpu_seconds / total_thread_seconds
        else:
            pct_cpu_util = 0.0

        # Per-call averages.
        if n_calls > 0:
            avg_call_blocked_us = (blocked_total / n_calls) * 1e6
            avg_samples_per_call = total_samples / n_calls
        else:
            avg_call_blocked_us = 0.0
            avg_samples_per_call = 0.0

        print(f"\n  ---- Worker timing breakdown "
              f"({n_workers} workers × {avg_gpw:.0f} threads) ----")
        print(f"    Per-worker wall avg:   {avg_wall:7.2f}s")
        print(f"    Per-thread blocked:    {avg_thread_blocked:7.2f}s "
              f"({pct_thread_blocked:5.1f}% of wall) — "
              f"how much each thread waits")
        print(f"    Aggregate CPU util:    {pct_cpu_util:5.1f}% "
              f"(across {n_threads_total} game threads, total)")
        print(f"    Per-worker put time:   {avg_put:7.3f}s "
              f"(negligible if << wall)")
        print(f"    Inference calls/worker: {n_calls // n_workers:,}")
        print(f"    Mean samples/call:     {avg_samples_per_call:6.2f}")
        print(f"    Mean blocked/call:     {avg_call_blocked_us:6.0f} µs")

        bs_calls = sum(s.get('batch_sync_calls', 0) for s in worker_stats)
        if bs_calls > 0:
            bs_samples = sum(s.get('batch_sync_samples', 0) for s in worker_stats)
            bs_searches = sum(s.get('batch_sync_searches_total', 0) for s in worker_stats)
            bs_max = max(s.get('batch_sync_max_samples', 0) for s in worker_stats)
            bs_width = max(s.get('batch_sync_width', 0) for s in worker_stats)
            print(f"    Batch-sync width:      {bs_width:6d}")
            print(f"    Batch-sync calls:      {bs_calls:6,}")
            print(f"    Mean searches/call:    {bs_searches / bs_calls:6.2f}")
            print(f"    Mean coord samples:    {bs_samples / bs_calls:6.2f}")
            print(f"    Max coord samples:     {bs_max:6d}")

        print(f"  --------------------------------------------------------")

        # Sanity-check verdict — now uses per-thread blocked %.
        if pct_thread_blocked > 60:
            if avg_gpw == 1:
                print(f"  ⇒ Threads are BLOCKED majority of wall time. "
                      f"Try --games-per-worker 2+ to overlap CPU/GPU.")
            else:
                print(f"  ⇒ Threads are still blocked majority of wall "
                      f"time. Try higher --games-per-worker or "
                      f"investigate server-side bottlenecks.")
        elif pct_thread_blocked > 40:
            print(f"  ⇒ Threads are moderately blocked. Configuration "
                  f"is reasonable; marginal gain from more threads.")
        else:
            print(f"  ⇒ Threads are mostly CPU-bound. GPU/IPC overlap "
                  f"is good — further parallelism won't help much.")

    return all_records


# ---- REPLAY BUFFER ----

class ReplayBuffer:
    """Rolling replay buffer — prevents catastrophic forgetting."""

    def __init__(self, max_size=500_000):
        self.max_size = max_size
        self.buffer   = deque(maxlen=max_size)

    def add(self, records):
        self.buffer.extend(records)

    def size(self):
        return len(self.buffer)

    def sample(self, n):
        n = min(n, len(self.buffer))
        # Convert to list once for O(1) indexing during sampling
        # (deque indexing is O(N), which would make random.sample
        # quadratic). The copy is shallow — only references move.
        return random.sample(list(self.buffer), n)

    def to_tensors(self, records):
        """
        Convert records to training tensors.

        Vectorized: stacks numpy arrays into a single buffer per field,
        then converts to a tensor in one shot. The original record-by-
        record loop was 10x+ slower at 100k+ records because each call
        to torch.tensor allocates and copies individually.
        """
        n = len(records)
        if n == 0:
            empty_f = torch.zeros(0, FEATURE_SIZE,  dtype=torch.float32)
            empty_m = torch.zeros(0, ACTION_SPACE,  dtype=torch.bool)
            empty_p = torch.zeros(0, ACTION_SPACE,  dtype=torch.float32)
            empty_v = torch.zeros(0,                dtype=torch.float32)
            empty_a = torch.zeros(0,                dtype=torch.long)
            return empty_f, empty_m, empty_p, empty_v, empty_a

        # Pre-allocate numpy buffers — single allocation per field.
        features_np      = np.empty((n, FEATURE_SIZE), dtype=np.float32)
        masks_np         = np.empty((n, ACTION_SPACE), dtype=np.bool_)
        policies_np      = np.empty((n, ACTION_SPACE), dtype=np.float32)
        values_np        = np.empty(n,                 dtype=np.float32)
        actions_np       = np.empty(n,                 dtype=np.int64)

        for i, rec in enumerate(records):
            features_np[i]  = rec['features']
            masks_np[i]     = rec['mask']
            policies_np[i]  = rec['mcts_policy']
            values_np[i]    = rec['value_target']
            actions_np[i]   = rec['action_idx']

        # from_numpy avoids the extra copy that torch.tensor does.
        features_t       = torch.from_numpy(features_np)
        masks_t          = torch.from_numpy(masks_np)
        policy_targets_t = torch.from_numpy(policies_np)
        value_targets_t  = torch.from_numpy(values_np)
        action_idx_t     = torch.from_numpy(actions_np)

        return features_t, masks_t, policy_targets_t, value_targets_t, action_idx_t


# ---- LOSS FUNCTIONS ----

def policy_loss_mcts(logits, policy_targets, masks):
    """
    Cross entropy loss with MCTS visit count targets.
    Numerically stable — guards against all-illegal rows.
    Includes entropy regularization to prevent policy collapse.
    """
    # Guard against all-illegal rows.
    valid_rows = masks.any(dim=1)
    if not valid_rows.all():
        logits         = logits[valid_rows]
        policy_targets = policy_targets[valid_rows]
        masks          = masks[valid_rows]

    if masks.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    masked_logits = logits.masked_fill(~masks, -1e9)
    log_probs     = F.log_softmax(masked_logits, dim=-1).clamp(min=-100)

    # Normalize targets over legal actions.
    targets  = policy_targets * masks.float()
    row_sums = targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    targets  = targets / row_sums

    ce_loss = -(targets * log_probs).sum(dim=-1).mean()

    # Entropy regularization.
    probs   = F.softmax(masked_logits, dim=-1).clamp(min=1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()

    return ce_loss - 0.01 * entropy


def combined_loss_mcts(value_pred, value_target, logits,
                        policy_targets, masks):
    """Combined value + MCTS policy loss."""
    # Ensure value_pred has same shape as value_target. Model may
    # return (B, 1) or (B,) depending on implementation; we squeeze
    # defensively to avoid silent broadcasting in BCE.
    if value_pred.dim() > value_target.dim():
        value_pred = value_pred.squeeze(-1)
    v_loss = F.binary_cross_entropy(value_pred, value_target)
    p_loss = policy_loss_mcts(logits, policy_targets, masks)
    total  = v_loss + p_loss
    return total, v_loss, p_loss


# ---- TRAIN/VAL SPLIT (BY GAME) ----

def split_records_by_game(records, val_frac=0.1):
    """
    Split records into train/val without leaking positions from the
    same game across the split.

    Records lacking a game_id (legacy) are treated as one unique game
    each, falling back to per-record split — equivalent to the old
    behavior. New records always have a real game_id.
    """
    by_game = defaultdict(list)
    no_id_records = []
    for rec in records:
        gid = rec.get('game_id')
        if gid is None:
            no_id_records.append(rec)
        else:
            by_game[gid].append(rec)

    game_ids = list(by_game.keys())
    random.shuffle(game_ids)

    # Ensure at least one game in val when we have at least 2 games.
    # int(N * val_frac) rounds to 0 for small N, which left val empty.
    if len(game_ids) >= 2:
        n_val_games = max(1, int(len(game_ids) * val_frac))
    else:
        n_val_games = 0
    val_game_ids = set(game_ids[:n_val_games])

    train_records = []
    val_records = []
    for gid, recs in by_game.items():
        if gid in val_game_ids:
            val_records.extend(recs)
        else:
            train_records.extend(recs)

    # Records without a game_id: shuffle and split per-record.
    if no_id_records:
        random.shuffle(no_id_records)
        n_no_id_val = int(len(no_id_records) * val_frac)
        val_records.extend(no_id_records[:n_no_id_val])
        train_records.extend(no_id_records[n_no_id_val:])

    return train_records, val_records


# ---- TRAIN ON BUFFER ----

def _estimate_gpu_data_bytes(n_records):
    """Estimate per-split GPU memory for tensorized data (rough)."""
    # features (f32) + mask (bool) + policy (f32) + value (f32) + action (i64)
    per_record = (FEATURE_SIZE * 4) + ACTION_SPACE + (ACTION_SPACE * 4) + 4 + 8
    return n_records * per_record


def _shuffle_indices(n, device):
    """Random permutation of [0..n) on the given device."""
    return torch.randperm(n, device=device)


def train_on_buffer(model, replay_buffer, device,
                    epochs=5, batch_size=512, lr=3e-5,
                    sample_size=None,
                    gpu_data_cap_bytes=2_000_000_000):
    """
    Train model on sample from replay buffer using MCTS targets.

    Performance pattern (works at any scale):
      - Tensorize sampled records ONCE per training call.
      - Transfer all training+val tensors to `device` ONCE.
      - Iterate batches via index slicing into on-device tensors
        (no DataLoader, no per-batch host→device copies).
      - Fresh random permutation each epoch for shuffling.

    Falls back to per-batch transfer if estimated GPU footprint exceeds
    `gpu_data_cap_bytes` (default 2 GB). At 100k records / Can't Stop
    feature size this is ~108 MB so the fast path is always used; for
    larger games with bigger feature tensors the fallback protects
    against OOM.

    Other improvements vs. the original:
      - Vectorized tensorization (np.stack-then-from_numpy).
      - Train/val split by GAME ID — no correlated-position leakage.
      - Policy accuracy compares NN argmax to MCTS argmax (the policy
        training target), not to the sampled action.
    """
    if sample_size and replay_buffer.size() > sample_size:
        records = replay_buffer.sample(sample_size)
    else:
        records = list(replay_buffer.buffer)

    # ---- Split by game BEFORE tensorizing ----
    tr_records, vl_records = split_records_by_game(records, val_frac=0.1)
    tr_size = len(tr_records)
    vl_size = len(vl_records)

    # Tensorize each split independently (CPU-side, single allocation).
    tr_feat, tr_mask, tr_pol, tr_val, tr_act = \
        replay_buffer.to_tensors(tr_records)
    vl_feat, vl_mask, vl_pol, vl_val, vl_act = \
        replay_buffer.to_tensors(vl_records)

    # ---- Decide: all-on-device fast path, or per-batch fallback ----
    est_bytes = _estimate_gpu_data_bytes(tr_size + vl_size)
    use_fast_path = (est_bytes <= gpu_data_cap_bytes)

    if use_fast_path:
        # Single host→device transfer of the entire training+val data.
        tr_feat = tr_feat.to(device); tr_mask = tr_mask.to(device)
        tr_pol  = tr_pol.to(device);  tr_val  = tr_val.to(device)
        tr_act  = tr_act.to(device)
        vl_feat = vl_feat.to(device); vl_mask = vl_mask.to(device)
        vl_pol  = vl_pol.to(device);  vl_val  = vl_val.to(device)
        vl_act  = vl_act.to(device)
        path_note = "on-device"
    else:
        path_note = (f"per-batch (data ~{est_bytes/1e9:.1f} GB > "
                     f"cap {gpu_data_cap_bytes/1e9:.1f} GB)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n  Training on {tr_size:,} records "
          f"(val: {vl_size:,} | buffer: {replay_buffer.size():,}) "
          f"for {epochs} epochs [{path_note}]...")

    best_val_loss = float('inf')
    best_state    = None

    def _move_batch(batch_tensors):
        """Move a batch of (possibly CPU) tensors to device."""
        return tuple(t.to(device, non_blocking=True) for t in batch_tensors)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        nan_batches = total_batches = 0

        # Fresh shuffle each epoch. On-device permutation if fast path,
        # CPU permutation if fallback.
        if use_fast_path:
            perm = _shuffle_indices(tr_size, device=device)
        else:
            perm = _shuffle_indices(tr_size, device='cpu')

        for start in range(0, tr_size, batch_size):
            end = min(start + batch_size, tr_size)
            idx = perm[start:end]

            feat = tr_feat[idx]
            msk  = tr_mask[idx]
            pol  = tr_pol[idx]
            val_t = tr_val[idx]
            act  = tr_act[idx]

            if not use_fast_path:
                feat, msk, pol, val_t, act = _move_batch(
                    (feat, msk, pol, val_t, act)
                )

            optimizer.zero_grad()
            total_batches += 1

            val_pred, logits = model(feat, msk)
            loss, v_loss, p_loss = combined_loss_mcts(
                val_pred, val_t, logits, pol, msk
            )

            if torch.isnan(loss):
                optimizer.zero_grad()
                nan_batches += 1
                continue

            loss.backward()

            has_nan = any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in model.parameters()
            )
            if has_nan:
                optimizer.zero_grad()
                nan_batches += 1
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = end - start
            total_loss    += loss.item() * bs
            total_samples += bs

        if nan_batches > 0:
            nan_pct = nan_batches / max(total_batches, 1) * 100
            warn = "⚠️  WARNING: high NaN rate" if nan_pct > 1.0 else "note"
            print(f"    [{warn}] Skipped {nan_batches}/{total_batches} "
                  f"batches ({nan_pct:.1f}%) due to NaN loss/grads")

        # ---- Validation ----
        model.eval()
        vl_loss_total = 0.0
        vl_correct_vs_mcts = 0
        vl_total = 0
        entropy_sum = 0.0

        with torch.no_grad():
            for start in range(0, vl_size, batch_size):
                end = min(start + batch_size, vl_size)

                feat  = vl_feat[start:end]
                msk   = vl_mask[start:end]
                pol   = vl_pol[start:end]
                val_t = vl_val[start:end]
                act   = vl_act[start:end]

                if not use_fast_path:
                    feat, msk, pol, val_t, act = _move_batch(
                        (feat, msk, pol, val_t, act)
                    )

                val_pred, logits = model(feat, msk)
                loss, _, _ = combined_loss_mcts(
                    val_pred, val_t, logits, pol, msk
                )
                bs = end - start
                vl_loss_total += loss.item() * bs

                masked_logits = logits.masked_fill(~msk, -1e9)
                nn_preds      = masked_logits.argmax(1)

                # Policy match: NN argmax vs MCTS argmax (the policy
                # training target — not the sampled action).
                mcts_preds = pol.argmax(1)
                vl_correct_vs_mcts += (nn_preds == mcts_preds).sum().item()

                vl_total += bs

                probs   = F.softmax(masked_logits, dim=-1).clamp(min=1e-8)
                entropy = -(probs * probs.log()).sum(dim=-1).mean()
                entropy_sum += entropy.item() * bs

        tr_loss     = total_loss    / max(total_samples, 1)
        vl_loss     = vl_loss_total / max(vl_total, 1)
        vl_acc      = vl_correct_vs_mcts / max(vl_total, 1)
        avg_entropy = entropy_sum   / max(vl_total, 1)

        print(f"    Epoch {epoch}/{epochs} | "
              f"Train: {tr_loss:.4f} | "
              f"Val: {vl_loss:.4f} | "
              f"Policy match (vs MCTS argmax): {vl_acc:.3f} | "
              f"Entropy: {avg_entropy:.3f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state    = {k: v.detach().clone()
                             for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    return best_val_loss


# ---- EVALUATION (MCTS vs MCTS) ----

def _play_mcts_eval_game(mcts_p0, mcts_p1, num_simulations,
                         max_turns=200):
    """
    Play one game between two MCTS instances. Returns winner (0 or 1)
    or None if the game hit max_turns without a winner.

    Both MCTS instances use temperature=0 (greedy at the policy level —
    pick the most-visited move). This is the correct deployment-style
    comparison: no exploration noise, just "which model's search is
    stronger?"
    """
    state = GameState(2)
    for _ in range(max_turns):
        if state.game_over:
            return state.winner

        if not state.dice:
            state.roll_dice()

        valid = get_valid_moves(state)
        if not valid:
            bust_turn(state)
            state.dice = []
            continue

        active = state.active_player
        searcher = mcts_p0 if active == 0 else mcts_p1

        _, move, decision, _, _ = searcher.get_action(
            state,
            num_simulations=num_simulations,
            temperature=0.0,
        )

        apply_move(state, move)
        if decision == "stop":
            stop_turn(state)
            state.dice = []
        else:
            # CONTINUE → spend the current roll, force a fresh roll.
            state.dice = []

    return None  # draw / timeout


# ---- Parallel eval worker ----
# Worker globals (one set per process).
_eval_mcts_new = None
_eval_mcts_old = None


def _init_eval_worker(new_model_path, old_model_path):
    """Load both models on CPU, build an MCTS for each."""
    global _eval_mcts_new, _eval_mcts_old
    new_model = load_model(new_model_path, 'cpu')
    old_model = load_model(old_model_path, 'cpu')
    _eval_mcts_new = MCTS(new_model, 'cpu')
    _eval_mcts_old = MCTS(old_model, 'cpu')


def _eval_worker_run(args):
    """
    Play a chunk of eval games. Half with new as P0, half with old as P0.
    Returns (wins_new, wins_old, draws).
    """
    n_games, num_simulations, seed = args
    random.seed(seed)
    np.random.seed(seed)

    wins_new = wins_old = draws = 0
    for i in range(n_games):
        try:
            # Alternate within the worker so each worker is self-balanced.
            if i % 2 == 0:
                winner = _play_mcts_eval_game(
                    _eval_mcts_new, _eval_mcts_old, num_simulations
                )
                if winner == 0:   wins_new += 1
                elif winner == 1: wins_old += 1
                else:             draws    += 1
            else:
                winner = _play_mcts_eval_game(
                    _eval_mcts_old, _eval_mcts_new, num_simulations
                )
                if winner == 0:   wins_old += 1
                elif winner == 1: wins_new += 1
                else:             draws    += 1
        except Exception as e:
            import traceback
            print(f"  [eval worker] game {i} failed: {e}", flush=True)
            traceback.print_exc()
            # Skip the failed game — it's safer to under-count than crash.
            continue

    return wins_new, wins_old, draws


def _save_model_to_tmp(model, output_dir, name='_eval_new_tmp.pt'):
    """Write a model's state dict to a temp file for worker loading."""
    tmp_path = os.path.join(output_dir, name)
    torch.save({'model_state': model.state_dict()}, tmp_path)
    return tmp_path


def evaluate_networks(new_model, old_model_path, num_games=500,
                      eval_sims=20, num_workers=None,
                      output_dir='.'):
    """
    Evaluate new vs old via MCTS-vs-MCTS at eval_sims simulations,
    using a worker pool.

    Each worker loads both models from disk (CPU), builds an MCTS
    instance for each, and plays a chunk of games alternating which
    model is P0 (to remove first-mover bias). Workers report
    (wins_new, wins_old, draws); main aggregates.

    Why parallel:
      - Sequential eval at 500 games × ~3s = ~25 min was the iteration
        bottleneck. With 8 workers that's ~3 min.
      - Allows tighter measurement (more games at the same wallclock
        cost) → tighter acceptance criteria possible.

    Args:
      new_model: in-memory model object (will be saved to tmp file
                 so workers can load it).
      old_model_path: path to the previous accepted model on disk.
      num_games: total eval games (split across workers).
      eval_sims: MCTS sims per move during eval.
      num_workers: defaults to min(cpu_count(), 8).
      output_dir: where to write the temporary new-model checkpoint.

    Returns:
      win_rate of the new model (draws excluded from denominator).
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)

    # Stash the new model to disk so workers can load it.
    new_model_tmp_path = _save_model_to_tmp(new_model, output_dir)

    # Split games across workers. Each worker gets an even number where
    # possible so its internal P0-alternation stays balanced.
    games_per_worker = num_games // num_workers
    leftover = num_games % num_workers
    seed_rng = random.Random(int(time.time() * 1000) & 0x7FFFFFFF)
    batch_args = []
    for i in range(num_workers):
        n = games_per_worker + (1 if i < leftover else 0)
        if n > 0:
            batch_args.append((n, eval_sims, seed_rng.randint(0, 2**31 - 1)))
    actual_workers = len(batch_args)

    print(f"\n  Evaluating new vs old "
          f"({num_games} MCTS games @ {eval_sims} sims, "
          f"{actual_workers} workers)...")

    eval_start = time.time()
    with mp.Pool(
        processes=actual_workers,
        initializer=_init_eval_worker,
        initargs=(new_model_tmp_path, old_model_path),
    ) as pool:
        results = pool.map(_eval_worker_run, batch_args)
    eval_elapsed = time.time() - eval_start

    wins_new = sum(r[0] for r in results)
    wins_old = sum(r[1] for r in results)
    draws    = sum(r[2] for r in results)

    total    = wins_new + wins_old
    win_rate = wins_new / total if total > 0 else 0.5

    print(f"  Eval done in {eval_elapsed:.1f}s "
          f"({num_games/max(eval_elapsed,1e-6):.1f} games/s aggregate)")
    print(f"  New model: {wins_new}/{total} ({win_rate:.1%})  "
          f"draws: {draws}")
    print(f"  Old model: {wins_old}/{total} ({1-win_rate:.1%})")

    return win_rate


# ---- MAIN SELF-PLAY LOOP ----

def self_play_loop(
    initial_model_path,
    output_dir='models/cantstop/self_play',
    iterations=10,
    games_per_iter=1000,
    num_simulations=20,
    train_epochs=5,
    eval_games=500,
    eval_sims=20,
    accept_floor=0.50,
    buffer_size=200_000,
    sample_size=100_000,
    initial_temp_mult=1.0,
    final_temp_mult=0.7,
    num_workers=None,
    device=None,
    target_inflight=16,
    warmup_sims=16,
    games_per_worker=2,
    batch_sync_searches=0,
):
    """
    Main training loop.

    Acceptance policy: always-accept-with-floor.
      - win_rate >= accept_floor (default 0.50) → accept new model
      - win_rate <  accept_floor              → reject (regression)

    The previous >55% threshold rejected genuinely-improving models
    because greedy eval (no MCTS) was a noisy proxy for the wrong
    capability — search-guidance quality. With MCTS eval at low sim
    count, win_rate ≈ 0.50 against the same baseline genuinely
    indicates a comparable model; we accept rather than stall.

    Floor of 0.50 (rather than e.g. 0.45) means we accept ties — which
    avoids the "stuck at 50-52%" failure mode — but never accept a
    losing record. Anything below 50% is more likely to be a real
    regression than a different-but-equal model.

    The strongest accepted model is always written to
    `<output_dir>/best_model.pt` after every acceptance, so a crash
    or stop can never lose the best model.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*55}")
    print(f"  Can't Stop MCTS Self-Play")
    print(f"{'='*55}")
    print(f"  Device:        {device}")
    if device == 'cuda':
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
    print(f"  Workers:       {num_workers} (CPU)")
    print(f"  Iterations:    {iterations}")
    print(f"  Games/iter:    {games_per_iter:,}")
    print(f"  MCTS sims:     {num_simulations}")
    print(f"  Inflight:      {target_inflight} (async scheduler)")
    print(f"  Warmup:        {warmup_sims} sequential sims/search")
    print(f"  Games/worker:  {games_per_worker} concurrent threads")
    if batch_sync_searches and batch_sync_searches > 1:
        print(f"  Batch-sync:    {batch_sync_searches} active sync searches per worker")
    print(f"  Buffer size:   {buffer_size:,}")
    print(f"  Sample size:   {sample_size:,}")
    print(f"  Train epochs:  {train_epochs}")
    print(f"  Eval games:    {eval_games} @ {eval_sims} sims (CPU MCTS)")
    print(f"  Accept floor:  {accept_floor:.0%}")
    print(f"{'='*55}\n")

    # Load model — track path for workers.
    current_model      = load_model(initial_model_path, device)
    current_model_path = initial_model_path

    # Stable "best" checkpoint path — overwritten on every acceptance.
    best_model_path = os.path.join(output_dir, 'best_model.pt')

    replay_buffer = ReplayBuffer(max_size=buffer_size)

    # ---- INFERENCE SERVER ----
    # Owns the GPU during game generation. Stays alive across iterations
    # so we don't pay model-load + CUDA-init overhead each time. Between
    # iterations we send a control-queue message to reload weights.
    #
    # During training, the server is idle (workers aren't sending
    # requests); its model occupies ~1MB of VRAM which is negligible
    # next to ~300MB of training data on the same GPU.
    server_device = device if device == 'cuda' else 'cpu'
    server_manager = InferenceServerManager(
        model_path=current_model_path,
        device=server_device,
        num_workers=num_workers,
        mp_context=mp.get_context('spawn'),
    )
    print(f"\n  Starting inference server (device={server_device}, "
          f"num_workers={num_workers})...")
    server_manager.start()
    # Give the server time to load the model + initialize CUDA. If we
    # ship the first request before this is done it'll just wait, but
    # explicit wait surfaces startup failures cleanly.
    time.sleep(2.0)
    if not server_manager.is_alive():
        raise RuntimeError(
            "Inference server failed to start. Check the model path "
            "and CUDA availability."
        )

    history  = []
    accepted = 0
    rejected = 0

    try:
        for iteration in range(1, iterations + 1):
            iter_start = time.time()

            temp_mult = initial_temp_mult - (
                (initial_temp_mult - final_temp_mult) *
                (iteration - 1) / max(iterations - 1, 1)
            )

            print(f"\n{'─'*55}")
            print(f"  Iteration {iteration}/{iterations} | "
                  f"Temp: {temp_mult:.2f} | "
                  f"Sims: {num_simulations} | "
                  f"Buffer: {replay_buffer.size():,}")
            print(f"{'─'*55}")

            # ---- STAGE 0: SYNC INFERENCE SERVER ----
            # Push the current model weights to the server. On iter 1
            # the server already has them (loaded at start()), so this
            # is a no-op cost (server reloads same file). From iter 2
            # onward it picks up the latest accepted model.
            server_manager.update_model(current_model_path)
            # Small grace period for the server to pick up the new
            # weights before workers send requests. The control queue
            # is checked at the top of each server loop iteration; with
            # SERVER_POLL_INTERVAL=50ms this is more than enough.
            time.sleep(0.5)

            # ---- STAGE 1: PARALLEL GAME GENERATION ----
            print(f"\n  Generating {games_per_iter:,} MCTS games "
                  f"({num_simulations} sims/move, "
                  f"{num_workers} workers, GPU inference)...")
            gen_start = time.time()

            # Per-iteration seed ensures different games each iteration.
            iter_seed = (int(time.time() * 1000) ^ (iteration * 2654435761)) & 0x7FFFFFFF

            if not server_manager.is_alive():
                raise RuntimeError(
                    "Inference server died during training. "
                    "Workers cannot generate games without it."
                )

            all_new_records = generate_games_parallel(
                server_manager=server_manager,
                num_games=games_per_iter,
                num_simulations=num_simulations,
                temp_mult=temp_mult,
                num_workers=num_workers,
                iteration_seed=iter_seed,
                target_inflight=target_inflight,
                warmup_sims=warmup_sims,
                games_per_worker=games_per_worker,
                batch_sync_searches=batch_sync_searches,
            )

            gen_time = time.time() - gen_start
            print(f"  Generated {len(all_new_records):,} records "
                  f"in {gen_time:.1f}s "
                  f"({games_per_iter/max(gen_time, 1e-6):.2f} games/s)")

            replay_buffer.add(all_new_records)
            print(f"  Buffer: {replay_buffer.size():,} / {buffer_size:,}")

            # ---- STAGE 2: TRAIN NEW MODEL ----
            new_model = CantStopNet().to(device)
            new_model.load_state_dict(current_model.state_dict())

            val_loss = train_on_buffer(
                new_model, replay_buffer, device,
                epochs=train_epochs,
                sample_size=sample_size,
                lr=3e-5,
            )

            # ---- STAGE 3: EVALUATE (MCTS vs MCTS, parallel workers) ----
            # Eval continues to use CPU MCTS for both new and old models.
            # The server only holds one model at a time and eval needs
            # both simultaneously; spinning up a second GPU server isn't
            # worth the complexity for this size of game.
            win_rate = evaluate_networks(
                new_model,
                old_model_path=current_model_path,
                num_games=eval_games,
                eval_sims=eval_sims,
                num_workers=num_workers,
                output_dir=output_dir,
            )

            iter_time = time.time() - iter_start

            # ---- ACCEPT OR REJECT ----
            if win_rate >= accept_floor:
                print(f"\n  ✓ ACCEPTED ({win_rate:.1%} >= "
                      f"{accept_floor:.0%})")
                current_model = new_model

                save_path = os.path.join(
                    output_dir,
                    f'model_iter_{iteration:03d}_accepted.pt'
                )
                checkpoint = {
                    'iteration':   iteration,
                    'win_rate':    float(win_rate),
                    'val_loss':    float(val_loss),
                    'temp_mult':   float(temp_mult),
                    'model_state': new_model.state_dict(),
                }
                torch.save(checkpoint, save_path)

                # ALSO save to the stable best_model.pt path, so the
                # strongest accepted model is always at a known location
                # regardless of crashes or stops.
                torch.save(checkpoint, best_model_path)

                current_model_path = save_path  # workers use new model
                accepted += 1
                print(f"  Saved iteration checkpoint: {save_path}")
                print(f"  Updated best model:         {best_model_path}")

            else:
                print(f"\n  ✗ REJECTED ({win_rate:.1%} < "
                      f"{accept_floor:.0%}) — keeping current model")
                rejected += 1

            history.append({
                'iteration':   iteration,
                'win_rate':    float(win_rate),
                'val_loss':    float(val_loss),
                'temp_mult':   float(temp_mult),
                'buffer_size': replay_buffer.size(),
                'accepted':    bool(win_rate >= accept_floor),
                'time':        float(iter_time),
            })

            print(f"\n  Time: {iter_time:.1f}s | "
                  f"Accepted: {accepted} | Rejected: {rejected}")
    finally:
        # Always stop the server, even if an iteration raised.
        print("\n  Stopping inference server...")
        server_manager.stop()

    # ---- SUMMARY ----
    print(f"\n{'='*55}")
    print(f"  MCTS Self-Play Complete!")
    print(f"  Iterations: {iterations}")
    print(f"  Accepted:   {accepted}")
    print(f"  Rejected:   {rejected}")
    print(f"\n  Win rate progression:")
    for h in history:
        status = "✓" if h['accepted'] else "✗"
        print(f"    Iter {h['iteration']:2d}: "
              f"{h['win_rate']:.1%} {status} "
              f"({h['time']:.0f}s)")
    print(f"{'='*55}\n")

    final_path = os.path.join(output_dir, f'final_{timestamp}.pt')
    torch.save({
        'model_state': current_model.state_dict(),
        'history':     history,
    }, final_path)
    print(f"  Final model: {final_path}")
    print(f"  Best model:  {best_model_path}")

    return current_model, history


# ---- ENTRY POINT ----

if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows

    parser = argparse.ArgumentParser(
        description="MCTS self-play for Can't Stop"
    )
    parser.add_argument("--model",      type=str,
                        default="models/cantstop/best_model.pt")
    parser.add_argument("--output",     type=str,
                        default="models/cantstop/self_play")
    parser.add_argument("--iterations", type=int,   default=10)
    parser.add_argument("--games",      type=int,   default=1000)
    parser.add_argument("--sims",       type=int,   default=20)
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--eval",       type=int,   default=500)
    parser.add_argument("--eval-sims",  type=int,   default=20,
                        dest="eval_sims",
                        help="MCTS simulations per move during eval")
    parser.add_argument("--floor",      type=float, default=0.50,
                        help="Win-rate floor for acceptance "
                             "(below this, reject). Default 0.50 "
                             "accepts ties; never accepts a losing "
                             "record.")
    parser.add_argument("--buffer",     type=int,   default=200_000)
    parser.add_argument("--sample",     type=int,   default=100_000)
    parser.add_argument("--temp_start", type=float, default=1.0)
    parser.add_argument("--temp_end",   type=float, default=0.7)
    parser.add_argument("--workers",    type=int,   default=None)
    parser.add_argument("--device",     type=str,   default=None)
    parser.add_argument("--inflight",   type=int,   default=16,
                        help="Concurrent in-flight MCTS sims per "
                             "worker in the async scheduler. Default "
                             "16. Higher values give bigger GPU "
                             "batches but worse training signal "
                             "without proportionally more warmup. "
                             "Set to 1 to force the legacy sync path.")
    parser.add_argument("--warmup",     type=int,   default=16,
                        help="Sequential warmup sims at the start of "
                             "each MCTS search. Default 16. Required "
                             "for usable training targets when "
                             "--inflight > 1 — covers the worst-case "
                             "branching factor at the root.")
    parser.add_argument("--games-per-worker", type=int, default=2,
                        dest="games_per_worker",
                        help="Concurrent games per worker process "
                             "(threads sharing one inference client). "
                             "Default 2. When >1, threads cover each "
                             "others' inference waits — recovers the "
                             "78%% of wall time previously spent "
                             "blocked. Set to 1 for the legacy single-"
                             "threaded behavior.")
    parser.add_argument("--batch-sync-searches", type=int, default=0,
                        dest="batch_sync_searches",
                        help="Fix C experimental path: number of active "
                             "independent games/searches per worker thread "
                             "whose legacy-sync MCTS eval chunks are "
                             "merged into one inference request. Set 0 or 1 "
                             "to disable.")
    args = parser.parse_args()

    self_play_loop(
        initial_model_path=args.model,
        output_dir=args.output,
        iterations=args.iterations,
        games_per_iter=args.games,
        num_simulations=args.sims,
        train_epochs=args.epochs,
        eval_games=args.eval,
        eval_sims=args.eval_sims,
        accept_floor=args.floor,
        buffer_size=args.buffer,
        sample_size=args.sample,
        initial_temp_mult=args.temp_start,
        final_temp_mult=args.temp_end,
        num_workers=args.workers,
        device=args.device,
        target_inflight=args.inflight,
        warmup_sims=args.warmup,
        games_per_worker=args.games_per_worker,
        batch_sync_searches=args.batch_sync_searches,
    )