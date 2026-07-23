"""Phase E Tier-1 trap suite (plan §7): harvest, ground truth, A/B measurement.

Pipeline (each stage resumable; ``all`` runs them in order):

1. **harvest** — scan replayable buffer games (plus optional fresh bot games)
   for positions where an uncovering action has >=1 consistent reveal that
   hands the opponent an immediate win AND a safe alternative exists. The
   detector is purely mechanical (engine + chance enumeration, no nets).
2. **groundtruth** — full-enumeration shallow expectimax with net leaves over
   each harvested position (``expand_exhaustive`` + ``closed_root_exact_value``),
   yielding exact per-action Q and the exact root value.
3. **evaluate** — run each search variant (closed / open / closed_forced) at
   each sims budget over ~20 seeds per position; record the chosen action,
   trap picks, and |root Q - exact| error.
4. **report** — aggregate trap-pick rate and selected-action Q error per
   (variant, sims).

Definition of "immediate win" (deliberately mechanical, documented limits):
the opponent, on the move after the trap action resolves, has some action
whose every consistent chance outcome ends the game in their favour —
including wins delivered through their own consecutive pending choices
(science pair -> progress token -> Law, Great Library, etc.). Wins that need
an extra-turn second move are NOT counted (they are two decisions deep, and
the same approximation is applied symmetrically to safety), and a chain after
which the trap actor retains the move (own pending resolved best-case, or an
extra turn) is classified safe.

The harvest prefilter is conservative: a trap requires the opponent to be
within mechanical reach of a military (position + shields swing <=
``MILITARY_REACH``) or scientific (>= ``SCIENCE_REACH`` distinct symbols) win,
neither of which the trap actor's own move can improve for them.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import zlib

from .bots import (
    GreedyBot,
    MilitaryAggressiveBot,
    MilitaryEconomyBot,
    ScienceAggressiveBot,
    ScienceEconomyBot,
)
from .buffer import (
    GameRecord,
    GameRecorder,
    ReplayMismatchError,
    read_records,
    replay,
)
from .codec import decode_action, encode_action, legal_action_indices
from .data import CARDS_BY_NAME
from .engine import ActionUse, apply_action, _science_symbols
from .game import ChanceKind, GameState, HiddenInformationError, Phase, new_game
from .search import (
    GumbelMCTS,
    SearchConfig,
    chance_signature,
    closed_root_exact_value,
    enumerate_chains,
    expand_exhaustive,
    state_actor,
)


@contextmanager
def _stage_lock(run_dir: Path, stage: str):
    """OS-released single-writer lock for append-only resumable stage files."""

    path = run_dir / f".{stage}.lock"
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"another {stage} writer holds {path}") from exc
    try:
        yield
    finally:
        if os.name == "nt":
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

MILITARY_REACH = 5  # max plausible one-action shield swing (3 shields +
# Strategy +1, with one square of slack)
SCIENCE_REACH = 4  # 4 distinct symbols can reach 6 in one action: new-symbol
# green that also completes a pair -> progress choice -> Law
PENDING_DEPTH = 6
CURRICULUM_BOT_TYPES = (
    ScienceAggressiveBot,
    ScienceEconomyBot,
    MilitaryAggressiveBot,
    MilitaryEconomyBot,
)
VARIANTS = {
    # name -> (search mode, force_expand_root_chance)
    "closed": ("closed", False),
    "open": ("open", False),
    "closed_forced": ("closed", True),
}
RESULT_SCHEMA = 2  # selected-action Q/error replaced the invalid root-mean metric


# --------------------------------------------------------------------------
# Mechanical immediate-win predicate
# --------------------------------------------------------------------------


def threat_possible(state: GameState, player: int) -> bool:
    """Cheap necessary condition for `player` to win within one action (plus
    own pending choices). Sound because an opponent's move never increases
    `player`'s coins, symbols, or conflict progress, and never lowers their
    trade costs — only a reveal can add opportunity, which is the point."""

    need = 9 - state.conflict_position if player == 0 else 9 + state.conflict_position
    if need <= MILITARY_REACH:
        return True
    return len(_science_symbols(state, player)) >= SCIENCE_REACH


def _pending_forced_win(state: GameState, player: int, depth: int) -> bool:
    """`player` holds the pending choice: can some option chain end the game
    in their favour without the opponent moving?"""

    if state.winner is not None:
        return state.winner == player
    pending = state.pending_choice
    if pending is None or pending.player != player or depth <= 0:
        return False
    for index in legal_action_indices(state):
        clone = state.clone()
        clone.search_barrier = True
        try:
            apply_action(clone, decode_action(clone, index))
        except HiddenInformationError:
            continue
        if _pending_forced_win(clone, player, depth - 1):
            return True
    return False


def _winning_move_candidates(state: GameState) -> list[int]:
    """Actions that could conceivably end the game for the actor this move:
    constructing a shield or science building, or any wonder. Discards,
    takes-without-build, and start-player choices cannot win."""

    candidates = []
    for index in legal_action_indices(state):
        action = decode_action(state, index)
        if action.use is ActionUse.CONSTRUCT_WONDER:
            candidates.append(index)
        elif action.use is ActionUse.CONSTRUCT_BUILDING:
            card = CARDS_BY_NAME[state.tableau.cards[action.slot_id].card_name]
            if card.shields > 0 or card.science is not None:
                candidates.append(index)
        elif action.use is ActionUse.RESOLVE_PENDING_CHOICE:
            candidates.append(index)
    return candidates


def guaranteed_win_now(state: GameState, depth: int = PENDING_DEPTH) -> bool:
    """The actor of `state` has an action guaranteed to end the game in their
    favour across ALL consistent chance outcomes of that action (pending
    chains included). Extra-turn two-move wins are out of scope by design."""

    if state.phase is Phase.COMPLETE:
        return False
    player = state_actor(state)
    if not threat_possible(state, player):
        return False
    if state.pending_choice is not None:
        return _pending_forced_win(state, player, depth)
    for index in _winning_move_candidates(state):
        action = decode_action(state, index)
        specs = chance_signature(state, action)
        if any(spec.kind is ChanceKind.AGE_DEAL for spec in specs):
            continue
        chains = enumerate_chains(state, specs)
        wins = True
        for outcomes, _probability, _key in chains:
            clone = state.clone()
            clone.search_barrier = True
            try:
                apply_action(
                    clone, decode_action(clone, index), chance_outcomes=outcomes or None
                )
            except HiddenInformationError:
                wins = False
                break
            if clone.winner == player:
                continue
            if not _pending_forced_win(clone, player, depth):
                wins = False
                break
        if wins:
            return True
    return False


def _chain_is_losing(child: GameState, actor: int, depth: int = PENDING_DEPTH) -> bool:
    """After one root-action chance chain resolved: does this outcome hand the
    opponent an immediate win? Chains where the trap actor keeps the move
    (extra turn, or a pending they can steer to safety) count as safe."""

    if child.winner is not None:
        return child.winner != actor
    next_actor = state_actor(child)
    if next_actor == actor:
        if child.pending_choice is not None:
            for index in legal_action_indices(child):
                clone = child.clone()
                clone.search_barrier = True
                try:
                    apply_action(clone, decode_action(clone, index))
                except HiddenInformationError:
                    continue
                if not _chain_is_losing(clone, actor, depth - 1):
                    return False
            return True
        return False  # extra turn: actor retains the initiative
    return guaranteed_win_now(child, depth)


def analyze_position(state: GameState) -> dict | None:
    """Full mechanical trap analysis of one decision state. Returns None
    unless the position qualifies: >=1 reveal-bearing action with a losing
    chain AND >=1 fully safe action."""

    if (
        state.phase is not Phase.PLAY_AGE
        or state.pending_choice is not None
        or state.winner is not None
    ):
        return None
    actor = state.active_player
    actions = []
    for index in legal_action_indices(state):
        action = decode_action(state, index)
        specs = chance_signature(state, action)
        if any(spec.kind is ChanceKind.AGE_DEAL for spec in specs):
            return None  # cannot enumerate; filtered upstream by card count
        has_reveal = any(spec.kind is ChanceKind.CARD_REVEAL for spec in specs)
        chains = enumerate_chains(state, specs)
        losing_mass = 0.0
        n_losing = 0
        for outcomes, probability, _key in chains:
            clone = state.clone()
            clone.search_barrier = True
            apply_action(
                clone, decode_action(clone, index), chance_outcomes=outcomes or None
            )
            if _chain_is_losing(clone, actor):
                losing_mass += probability
                n_losing += 1
        actions.append(
            {
                "action": index,
                "n_chains": len(chains),
                "n_losing": n_losing,
                "losing_mass": losing_mass,
                "has_reveal": has_reveal,
            }
        )
    traps = [a for a in actions if a["n_losing"] and a["has_reveal"]]
    safe = [a["action"] for a in actions if not a["n_losing"]]
    unsafe_other = [
        a["action"] for a in actions if a["n_losing"] and not a["has_reveal"]
    ]
    if not traps or not safe:
        return None
    return {
        "actor": actor,
        "age": state.age,
        "conflict": state.conflict_position,
        "present_cards": sum(
            1 for card in state.tableau.cards.values() if card.present
        ),
        "n_legal": len(actions),
        "actions": actions,
        "traps": [a["action"] for a in traps],
        "safe": safe,
        "unsafe_other": unsafe_other,
    }


def harvest_prefilter(state: GameState) -> bool:
    """Cheap gate run at every replayed decision before full analysis."""

    if (
        state.phase is not Phase.PLAY_AGE
        or state.pending_choice is not None
        or state.winner is not None
    ):
        return False
    present = sum(1 for card in state.tableau.cards.values() if card.present)
    if present < 3:  # keeps the depth-2 subtree clear of AGE_DEAL boundaries
        return False
    return threat_possible(state, 1 - state.active_player)


# --------------------------------------------------------------------------
# Harvest
# --------------------------------------------------------------------------


def fresh_bot_records(games: int, seed: int) -> list[GameRecord]:
    records = []
    for i in range(games):
        rush = CURRICULUM_BOT_TYPES[(i // 2) % len(CURRICULUM_BOT_TYPES)](
            seed=(seed + i) ^ 0xA5A5
        )
        greedy = GreedyBot()
        bots = (rush, greedy) if i % 2 == 0 else (greedy, rush)
        recorder = GameRecorder(
            seed + 40_000_000 + i,
            first_player=(i // 2) % 2,
            agents={"p0": bots[0].name, "p1": bots[1].name, "kind": "phase_e_fresh"},
        )
        while recorder.game.phase is not Phase.COMPLETE:
            actor = state_actor(recorder.game)
            action = bots[actor].select_action(recorder.game)
            recorder.play(encode_action(recorder.game, action))
        records.append(recorder.finish())
    return records


def harvest_records(
    records: list[GameRecord],
    source: str,
    *,
    quota: int,
    per_game_cap: int,
    found: list[dict],
    seen_ids: set[str],
    stats: dict,
    on_found=None,
) -> None:
    """Scan `records`, appending qualifying positions to `found` until quota."""

    for game_index, record in enumerate(records):
        if len(found) >= quota:
            return
        game_prefix = f"{source}:{game_index}:"
        banked_this_game = sum(
            position_id.startswith(game_prefix) for position_id in seen_ids
        )
        if banked_this_game >= per_game_cap:
            continue
        game_hits: list[dict] = []
        game_hit_ids: set[str] = set()

        def on_state(game, move, _gi=game_index, _rec=record):
            stats["positions"] += 1
            if (
                len(found) + len(game_hits) >= quota
                or banked_this_game + len(game_hits) >= per_game_cap
            ):
                return
            if not harvest_prefilter(game):
                return
            stats["candidates"] += 1
            analysis = analyze_position(game)
            if analysis is None:
                return
            position_id = f"{source}:{_gi}:{move.i}"
            if position_id in seen_ids or position_id in game_hit_ids:
                return
            game_hit_ids.add(position_id)
            game_hits.append(
                {
                    "id": position_id,
                    "source": source,
                    "game_seed": _rec.seed,
                    "first_player": _rec.first_player,
                    "move_index": move.i,
                    "prefix": [m.action for m in _rec.moves[: move.i]],
                    **analysis,
                }
            )

        try:
            replay(record, on_state=on_state)
            stats["games"] += 1
        except ReplayMismatchError:
            stats["replay_mismatches"] += 1
            continue
        for row in game_hits:
            if len(found) >= quota:
                break
            seen_ids.add(row["id"])
            found.append(row)
            if on_found is not None:
                on_found(row)


def run_harvest(args, run_dir: Path) -> list[dict]:
    positions_path = run_dir / "positions.jsonl"
    found: list[dict] = []
    seen_ids: set[str] = set()
    if positions_path.exists():
        for line in positions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                found.append(row)
                seen_ids.add(row["id"])
    if len(found) >= args.quota:
        print(f"harvest: {len(found)} positions already banked, skipping scan")
        return found

    stats = {"games": 0, "positions": 0, "candidates": 0, "replay_mismatches": 0}
    start = time.time()
    with positions_path.open("a", encoding="utf-8", newline="\n") as handle:
        def persist(row):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()

        for buffer_path in args.buffers:
            if len(found) >= args.quota:
                break
            source = Path(buffer_path).stem
            records = read_records(buffer_path)
            harvest_records(
                records,
                source,
                quota=args.quota,
                per_game_cap=args.per_game_cap,
                found=found,
                seen_ids=seen_ids,
                stats=stats,
                on_found=persist,
            )
            print(
                f"harvest: {source}: total {len(found)}/{args.quota} "
                f"({stats['candidates']} candidates / {stats['positions']} positions, "
                f"{time.time() - start:.0f}s)"
            )
        fresh_round = 0
        while len(found) < args.quota and args.fresh_games:
            batch = min(args.fresh_games, 200)
            records = fresh_bot_records(batch, args.seed + fresh_round * 1_000_000)
            harvest_records(
                records,
                f"fresh_{fresh_round}",
                quota=args.quota,
                per_game_cap=args.per_game_cap,
                found=found,
                seen_ids=seen_ids,
                stats=stats,
                on_found=persist,
            )
            fresh_round += 1
            print(f"harvest: fresh round {fresh_round}: total {len(found)}/{args.quota}")
            if fresh_round * batch >= args.fresh_games:
                break
    print(
        f"harvest: {len(found)} positions from {stats['games']} games "
        f"({stats['replay_mismatches']} replay mismatches skipped)"
    )
    return found


# --------------------------------------------------------------------------
# Reconstruction & evaluator
# --------------------------------------------------------------------------


def reconstruct(position: dict) -> GameState:
    state = new_game(position["game_seed"], first_player=position["first_player"])
    for index in position["prefix"]:
        apply_action(state, decode_action(state, index))
    if state_actor(state) != position["actor"]:
        raise RuntimeError(f"reconstruction diverged for {position['id']}")
    return state


def load_evaluator(checkpoint_path: str, device: str):
    import torch

    from .inference import Evaluator
    from .train import build_model, load_checkpoint

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = build_model(
        config.get("model", "transformer"),
        config.get("d_model", 128),
        config.get("layers", 4),
    )
    load_checkpoint(checkpoint_path, model)
    return Evaluator(model, device=device)


# --------------------------------------------------------------------------
# Ground truth: full-enumeration shallow expectimax with net leaves
# --------------------------------------------------------------------------


class _StructureMCTS(GumbelMCTS):
    """Tree builder with a free dummy evaluation: expand_exhaustive needs
    priors only structurally, and every frontier value is overwritten by one
    batched pass of the real net afterwards."""

    def __init__(self):
        super().__init__(evaluator=None, config=SearchConfig())

    def _evaluate(self, state):
        return 0.0, {}


def _walk(node):
    yield node
    for edge in node.edges:
        for child in edge.children.values():
            yield from _walk(child.node)


def exact_ground_truth(
    state: GameState, evaluator, *, depth: int, frontier_cap: int
) -> dict | None:
    """Depth-limited full-enumeration expectimax from `state` with net leaf
    values. Returns exact per-action Q (root-actor perspective) or None when
    the enumerated tree would exceed `frontier_cap` frontier nodes."""

    mcts = _StructureMCTS()
    root_state = state.clone()
    root_state.search_barrier = True
    root = mcts._make_closed_node(root_state)
    expand_exhaustive(mcts, root, depth=1)
    if depth >= 2:
        estimate = 0
        for edge in root.edges:
            for child in edge.children.values():
                node = child.node
                if node.terminal:
                    continue
                if not node.edges:
                    mcts._expand_closed(node)
                estimate += sum(
                    len(enumerate_chains(node.state, deeper.specs))
                    for deeper in node.edges
                )
        if estimate > frontier_cap:
            return None
    expand_exhaustive(mcts, root, depth=depth)

    leaves = [
        node
        for node in _walk(root)
        if not node.terminal and not node.edges and node.visits
    ]
    evaluations = evaluator.evaluate_states([node.state for node in leaves])
    for node, evaluation in zip(leaves, evaluations, strict=True):
        actor = state_actor(node.state)
        value = float(evaluation.wdl[0] - evaluation.wdl[2])
        node.value_sum_p0 = value if actor == 0 else -value
        node.visits = 1

    sign = 1.0 if root.actor == 0 else -1.0
    exact_q = {}
    for edge in root.edges:
        children = list(edge.children.values())
        mass = sum(child.probability for child in children)
        if abs(mass - 1.0) > 1e-9:
            raise RuntimeError(f"edge mass {mass} != 1 in ground truth")
        value_p0 = sum(
            child.probability * closed_root_exact_value(child.node)
            for child in children
        )
        exact_q[edge.action_index] = sign * value_p0
    exact_best = max(exact_q, key=exact_q.__getitem__)
    return {
        "exact_q": {str(a): q for a, q in exact_q.items()},
        "exact_root": exact_q[exact_best],
        "exact_best": exact_best,
        "n_frontier": len(leaves),
        "depth": depth,
    }


_GROUND_TRUTH_EVALUATOR = None
_GROUND_TRUTH_DEPTH = 0
_GROUND_TRUTH_FRONTIER_CAP = 0


def _ground_truth_process_init(checkpoint: str, device: str, depth: int, frontier_cap: int):
    global _GROUND_TRUTH_EVALUATOR, _GROUND_TRUTH_DEPTH, _GROUND_TRUTH_FRONTIER_CAP
    _GROUND_TRUTH_EVALUATOR = load_evaluator(checkpoint, device)
    _GROUND_TRUTH_DEPTH = depth
    _GROUND_TRUTH_FRONTIER_CAP = frontier_cap


def _ground_truth_row(position: dict, evaluator, depth: int, frontier_cap: int) -> dict:
    state = reconstruct(position)
    truth = exact_ground_truth(
        state,
        evaluator,
        depth=depth,
        frontier_cap=frontier_cap,
    )
    row = {"id": position["id"], "skipped": truth is None}
    if truth is not None:
        row.update(truth)
        trap_q = max(truth["exact_q"][str(a)] for a in position["traps"])
        row["trap_gap"] = truth["exact_root"] - trap_q
    return row


def _ground_truth_process_row(position: dict) -> dict:
    if _GROUND_TRUTH_EVALUATOR is None:
        raise RuntimeError("ground-truth process evaluator was not initialized")
    return _ground_truth_row(
        position,
        _GROUND_TRUTH_EVALUATOR,
        _GROUND_TRUTH_DEPTH,
        _GROUND_TRUTH_FRONTIER_CAP,
    )


def run_ground_truth(args, run_dir: Path, positions: list[dict], evaluator) -> dict:
    path = run_dir / "ground_truth.jsonl"
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    start = time.time()

    def evaluate_position(position):
        return _ground_truth_row(
            position,
            evaluator,
            args.gt_depth,
            args.gt_frontier_cap,
        )

    pending = [position for position in positions if position["id"] not in done]
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        if args.groundtruth_process_workers:
            pool = ProcessPoolExecutor(
                max_workers=args.groundtruth_process_workers,
                initializer=_ground_truth_process_init,
                initargs=(
                    str(args.checkpoint),
                    args.device,
                    args.gt_depth,
                    args.gt_frontier_cap,
                ),
            )
            mapper = _ground_truth_process_row
        else:
            pool = ThreadPoolExecutor(max_workers=args.groundtruth_workers)
            mapper = evaluate_position
        with pool:
            futures = [pool.submit(mapper, position) for position in pending]
            for completed, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                done[row["id"]] = row
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                if completed % 10 == 0 or completed == len(pending):
                    print(
                        f"groundtruth: {len(done)}/{len(positions)} "
                        f"({time.time() - start:.0f}s)"
                    )
    skipped = sum(1 for row in done.values() if row.get("skipped"))
    if skipped:
        print(f"groundtruth: {skipped} positions skipped (frontier cap)")
    return done


# --------------------------------------------------------------------------
# Search evaluation
# --------------------------------------------------------------------------


def _search_seed(position_id: str, sims: int, seed_index: int) -> int:
    # Same stream for every variant at the same (position, sims, seed):
    # paired comparisons share Gumbel noise where the candidate sets align.
    paired = f"{position_id}|{sims}|{seed_index}"
    return (zlib.crc32(paired.encode()) << 8) | (seed_index & 0xFF)


def run_evaluate(
    args, run_dir: Path, positions: list[dict], truths: dict, evaluator
) -> list[dict]:
    path = run_dir / "results.jsonl"
    done: set[tuple] = set()
    rows: list[dict] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("schema") != RESULT_SCHEMA:
                    continue
                rows.append(row)
                done.add((row["id"], row["variant"], row["sims"], row["seed"]))
    start = time.time()
    usable = [
        p for p in positions if truths.get(p["id"]) and not truths[p["id"]].get("skipped")
    ]
    total = len(usable) * len(args.variants) * len(args.sims) * args.seeds
    print(
        f"evaluate: {total} searches over {len(usable)} positions "
        f"({len(done)} already done); Gumbel search is ~10-20ms/sim on CPU — "
        f"expect roughly {total * sum(args.sims) / len(args.sims) * 0.012 / 3600:.1f}h "
        "at defaults (closed_forced adds a constant per search)"
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for position in usable:
            truth = truths[position["id"]]
            trap_set = set(position["traps"])
            unsafe_set = trap_set | set(position["unsafe_other"])
            state = reconstruct(position)
            for variant in args.variants:
                mode, forced = VARIANTS[variant]
                for sims in args.sims:
                    for seed_index in range(args.seeds):
                        key = (position["id"], variant, sims, seed_index)
                        if key in done:
                            continue
                        config = SearchConfig(
                            sims=sims,
                            top_k=args.top_k,
                            mode=mode,
                            seed=_search_seed(position["id"], sims, seed_index),
                            force_expand_root_chance=forced,
                        )
                        t0 = time.time()
                        result = GumbelMCTS(evaluator, config).search(state)
                        row = {
                            "schema": RESULT_SCHEMA,
                            "id": position["id"],
                            "variant": variant,
                            "sims": sims,
                            "seed": seed_index,
                            "action": result.action_index,
                            "trap_pick": result.action_index in trap_set,
                            "unsafe_pick": result.action_index in unsafe_set,
                            "root_value": result.root_value,
                            "search_q": result.action_value,
                            "exact_action_q": truth["exact_q"][
                                str(result.action_index)
                            ],
                            "action_q_error": abs(
                                result.action_value
                                - truth["exact_q"][str(result.action_index)]
                            ),
                            "action_regret": truth["exact_root"]
                            - truth["exact_q"][str(result.action_index)],
                            "ms": round((time.time() - t0) * 1000.0, 1),
                        }
                        rows.append(row)
                        done.add(key)
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            elapsed = time.time() - start
            print(
                f"evaluate: {len(done)}/{total} searches "
                f"({elapsed:.0f}s, {position['id']})"
            )
    return rows


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


CONSEQUENTIAL_GAP = 0.25


def run_report(
    run_dir: Path, positions: list[dict], rows: list[dict], truths: dict | None = None
) -> dict:
    by_position = {p["id"]: p for p in positions}
    consequential = {
        pid
        for pid, truth in (truths or {}).items()
        if truth.get("trap_gap", 0.0) >= CONSEQUENTIAL_GAP
    }

    def aggregate(selected_rows):
        cells: dict[tuple, dict] = {}
        for row in selected_rows:
            if row["id"] not in by_position:
                continue
            cell = cells.setdefault(
                (row["variant"], row["sims"]),
                {"n": 0, "trap": 0, "unsafe": 0, "q_err": [], "ms": []},
            )
            cell["n"] += 1
            cell["trap"] += row["trap_pick"]
            cell["unsafe"] += row["unsafe_pick"]
            cell["q_err"].append(row["action_q_error"])
            cell["ms"].append(row["ms"])
        return cells

    def median(values):
        ordered = sorted(values)
        mid = len(ordered) // 2
        if not ordered:
            return 0.0
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    summary = {
        "positions": len(positions),
        "consequential_positions": len(consequential),
        "consequential_gap": CONSEQUENTIAL_GAP,
        "segments": {},
    }
    lines = []
    segments = [("all", rows)]
    if consequential:
        segments.append(
            (
                f"consequential (trap_gap >= {CONSEQUENTIAL_GAP})",
                [row for row in rows if row["id"] in consequential],
            )
        )
    for segment_name, segment_rows in segments:
        header = (
            f"{'variant':<14}{'sims':>6}{'rows':>7}{'trap%':>8}{'unsafe%':>9}"
            f"{'|dQ(a)| mean':>13}{'|dQ(a)| med':>12}{'ms/search':>11}"
        )
        lines += [f"== {segment_name} ==", header, "-" * len(header)]
        entries = []
        for (variant, sims), cell in sorted(aggregate(segment_rows).items()):
            entry = {
                "variant": variant,
                "sims": sims,
                "rows": cell["n"],
                "trap_pick_rate": cell["trap"] / cell["n"],
                "unsafe_pick_rate": cell["unsafe"] / cell["n"],
                "action_q_error_mean": sum(cell["q_err"]) / cell["n"],
                "action_q_error_median": median(cell["q_err"]),
                "ms_mean": sum(cell["ms"]) / cell["n"],
            }
            entries.append(entry)
            lines.append(
                f"{variant:<14}{sims:>6}{cell['n']:>7}"
                f"{100 * entry['trap_pick_rate']:>7.1f}%"
                f"{100 * entry['unsafe_pick_rate']:>8.1f}%"
                f"{entry['action_q_error_mean']:>13.4f}"
                f"{entry['action_q_error_median']:>12.4f}"
                f"{entry['ms_mean']:>11.0f}"
            )
        lines.append("")
        summary["segments"][segment_name] = entries
    report_text = "\n".join(lines)
    print(report_text)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "summary.txt").write_text(report_text + "\n", encoding="utf-8")
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "stage", choices=("harvest", "groundtruth", "evaluate", "report", "all")
    )
    parser.add_argument("--run-dir", default="runs/seven_wonders_duel/phase_e")
    parser.add_argument("--buffers", nargs="*", default=[])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quota", type=int, default=120)
    parser.add_argument("--per-game-cap", type=int, default=2)
    parser.add_argument("--fresh-games", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gt-depth", type=int, default=2)
    parser.add_argument("--gt-frontier-cap", type=int, default=60000)
    parser.add_argument("--groundtruth-workers", type=int, default=1)
    parser.add_argument("--groundtruth-process-workers", type=int, default=0)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["closed", "open", "closed_forced"],
        choices=sorted(VARIANTS),
    )
    parser.add_argument("--sims", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=16)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    def load_positions():
        path = run_dir / "positions.jsonl"
        if not path.exists():
            raise SystemExit("no positions.jsonl — run the harvest stage first")
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def load_truths():
        path = run_dir / "ground_truth.jsonl"
        if not path.exists():
            raise SystemExit("no ground_truth.jsonl — run the groundtruth stage first")
        return {
            row["id"]: row
            for row in (
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }

    needs_net = args.stage in ("groundtruth", "evaluate", "all") and not (
        args.stage == "groundtruth" and args.groundtruth_process_workers
    )
    evaluator = None
    if needs_net:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for this stage")
        evaluator = load_evaluator(args.checkpoint, args.device)

    if args.stage in ("harvest", "all"):
        with _stage_lock(run_dir, "harvest"):
            positions = run_harvest(args, run_dir)
        if len(positions) < 100:
            print(
                f"WARNING: only {len(positions)} positions (plan asks >=100); "
                "raise --fresh-games or add more --buffers"
            )
    else:
        positions = load_positions()

    if args.stage in ("groundtruth", "all"):
        with _stage_lock(run_dir, "groundtruth"):
            truths = run_ground_truth(args, run_dir, positions, evaluator)
    elif args.stage in ("evaluate", "report"):
        truths = load_truths()

    if args.stage in ("evaluate", "all"):
        with _stage_lock(run_dir, "evaluate"):
            rows = run_evaluate(args, run_dir, positions, truths, evaluator)
    elif args.stage == "report":
        path = run_dir / "results.jsonl"
        if not path.exists():
            raise SystemExit("no results.jsonl — run the evaluate stage first")
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("schema") == RESULT_SCHEMA:
                    rows.append(row)
        if not rows:
            raise SystemExit(
                "results.jsonl contains no current-schema rows — rerun evaluate"
            )

    if args.stage in ("report", "all"):
        run_report(run_dir, positions, rows, truths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
