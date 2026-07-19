"""Offline pick-denial search and curriculum-label emitter for Kingdomino.

The live advisor's draft matrix established the useful decomposition: boards are
private, so the only adversarial interaction is the sequence of picks.  This
module turns that decomposition into a request-independent, batch-oriented
generation tool.  Placements are delegated to AlphaZero, while every pick is
expanded explicitly.  The resulting expectiminimax values are emitted as policy
and value targets; this module never trains a network.

Horizon convention
------------------
``pick_plies=8`` applies eight PLACE_AND_SELECT actions.  A start-of-round root
therefore has one *interior* deal (after pick four).  The deal after pick eight
is the horizon boundary, not a second chance expansion.  At that boundary the
leaf is represented as the public pre-reveal information state: the completed
claims are retained, ``current_row`` is empty, and the sorted pre-deal bag is
restored.  This is the only order-blind way to ask the AZ value head for the
value at exactly eight picks without silently leaking the engine's hidden deck
order or expanding a forbidden second chance node.

Policy target
-------------
Values are converted to actor-frame pick logits with a documented robust
softmax.  For pick ``i`` the effective gap from the best is reduced by one
combined Monte-Carlo standard error; gaps within ``tie_tolerance`` are zero.
The remaining non-negative gaps are divided by ``temperature`` and softmaxed.
This makes exact/uncertain ties share mass, prevents sampling noise from making
spurious one-hot labels, and produces a deterministic valid distribution.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from games.kingdomino.action_codec import encode_action
from games.kingdomino.encoder import encode_state
from games.kingdomino.game import (
    GameState,
    Phase,
    PickAction,
    TurnAction,
    determine_winner,
)
from games.kingdomino.promotion import DEFAULT_CURRENT_BEST, sha256_file


CASCADE_VERSION = "official-score-largest-territory-crowns-draw-v1"
ACTOR_FRAME = "root-current-actor"


def _stable_seed(base: int, sorted_bag: Sequence[int], drew: int = 4) -> int:
    """Cross-version-stable seed for common random numbers."""
    h = hashlib.blake2b(digest_size=8)
    h.update(int(base).to_bytes(8, "little", signed=True))
    h.update(int(drew).to_bytes(2, "little", signed=False))
    for domino_id in sorted_bag:
        h.update(int(domino_id).to_bytes(2, "little", signed=False))
    return int.from_bytes(h.digest(), "little", signed=False)


def chance_rows(
    remaining_bag: Sequence[int],
    k: int,
    *,
    seed: int,
    drew: int = 4,
) -> tuple[list[tuple[int, ...]], str]:
    """Return canonical order-blind draw rows and ``enumerated|sampled`` mode.

    Enumeration is exact precisely when ``C(n, drew) <= k``.  Otherwise rows
    are sampled with replacement from the distribution of combinations.  The
    stable seed depends only on the sorted public bag, so sibling branches use
    identical rows (common random numbers), irrespective of hidden deck order.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    bag = tuple(sorted(int(x) for x in remaining_bag))
    if len(bag) < drew:
        return [], "enumerated"
    count = math.comb(len(bag), drew)
    if count <= k:
        return list(itertools.combinations(bag, drew)), "enumerated"
    rng = random.Random(_stable_seed(seed, bag, drew))
    return [tuple(sorted(rng.sample(bag, drew))) for _ in range(k)], "sampled"


def denial_policy_target(
    actor_values: Sequence[float],
    stderrs: Optional[Sequence[float]] = None,
    *,
    temperature: float = 0.10,
    tie_tolerance: float = 1e-6,
    uncertainty_z: float = 1.0,
) -> list[float]:
    """Turn searched actor-frame pick values into a robust policy target."""
    values = np.asarray(actor_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("actor_values must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all():
        raise ValueError("actor_values must all be finite")
    if values.size == 1:
        return [1.0]
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    errors = np.zeros_like(values) if stderrs is None else np.asarray(stderrs, dtype=np.float64)
    if errors.shape != values.shape or not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("stderrs must be finite, non-negative, and aligned with values")
    best_idx = int(np.argmax(values))
    raw_gap = values[best_idx] - values
    combined = np.sqrt(errors[best_idx] ** 2 + errors ** 2)
    gap = np.maximum(0.0, raw_gap - float(uncertainty_z) * combined)
    gap[raw_gap <= float(tie_tolerance)] = 0.0
    logits = -gap / float(temperature)
    logits -= float(logits.max())
    probs = np.exp(logits)
    probs /= float(probs.sum())
    return [float(x) for x in probs]


def _pick_key(action: PickAction | TurnAction) -> Optional[int]:
    if isinstance(action, PickAction):
        return int(action.domino_id)
    if action.pick_domino_id is not None:
        return int(action.pick_domino_id)
    return None


def _placement_json(action: PickAction | TurnAction) -> Optional[dict[str, Any]]:
    if not isinstance(action, TurnAction) or action.placement is None:
        return None
    p = action.placement
    return {"x1": int(p.x1), "y1": int(p.y1), "x2": int(p.x2), "y2": int(p.y2),
            "flipped": bool(p.flipped)}


def action_json(state: GameState, action: PickAction | TurnAction) -> dict[str, Any]:
    """Small stable action record suitable for emitted labels."""
    return {
        "action_idx": int(encode_action(action, state)),
        "kind": "pick" if isinstance(action, PickAction) else "turn",
        "pick_domino_id": _pick_key(action),
        "placement": _placement_json(action),
    }


def _board_bytes(state: GameState) -> bytes:
    chunks: list[bytes] = []
    for board in state.boards:
        chunks.extend((board.terrain.tobytes(), board.crowns.tobytes(), board.domino_id.tobytes()))
    return b"".join(chunks)


def _public_state_key_uncached(state: GameState) -> str:
    """Compute the public-state hash without consulting the per-object memo."""
    h = hashlib.blake2b(digest_size=20)
    h.update(_board_bytes(state))
    payload = (
        tuple(sorted(int(x) for x in state.deck)),
        tuple(sorted(int(x) for x in state.current_row)),
        tuple((int(c.player), int(c.domino_id)) for c in state.pending_claims),
        tuple((int(c.player), int(c.domino_id)) for c in state.next_claims),
        int(state.phase), int(state.actor_index), int(state.initial_pick_count),
        int(state.start_player), tuple(int(x) for x in state.discards),
    )
    h.update(repr(payload).encode("ascii"))
    return h.hexdigest()


def public_state_key(state: GameState) -> str:
    """Hash the public information state; hidden deck order is excluded.

    Search states are immutable after construction: ``step`` and every chance
    helper create a fresh ``GameState`` before changing it.  Cache the digest on
    that object so node construction, evaluator lookup, and backup do not each
    repeat the board-array blake2b.  ``GameState.copy`` intentionally does not
    copy dynamic attributes, so a subsequently mutated copy starts uncached.
    """
    cached = getattr(state, "_denial_public_state_key", None)
    if cached is None:
        cached = _public_state_key_uncached(state)
        setattr(state, "_denial_public_state_key", cached)
    return str(cached)


def _replace_draw(child: GameState, pre_deal_bag: Sequence[int], row: Sequence[int]) -> GameState:
    out = child.copy()
    remaining = list(int(x) for x in pre_deal_bag)
    for domino_id in row:
        remaining.remove(int(domino_id))
    out.current_row = sorted(int(x) for x in row)
    out.deck = sorted(remaining)
    return out


def _as_pre_reveal_leaf(child: GameState, pre_deal_bag: Sequence[int]) -> GameState:
    out = child.copy()
    out.current_row = []
    out.deck = sorted(int(x) for x in pre_deal_bag)
    return out


def _terminal_value_p0(state: GameState) -> float:
    winner = determine_winner(state)
    return 0.0 if winner is None else (1.0 if winner == 0 else -1.0)


@dataclass
class EvalStats:
    policy_batch_sizes: list[int] = field(default_factory=list)
    leaf_batch_sizes: list[int] = field(default_factory=list)
    leaf_cache_hits: int = 0
    leaf_cache_misses: int = 0
    policy_cache_hits: int = 0
    policy_cache_misses: int = 0
    node_tt_hits: int = 0
    node_tt_misses: int = 0
    node_tt_pass_hits: int = 0
    root_search_calls: int = 0
    root_search_cache_hits: int = 0


class AZBatchEvaluator:
    """Batched AZ policy/value evaluator with a cross-position public-state TT."""

    def __init__(self, net, *, device: str, batch_size: int = 512,
                 margin_gain: float = 2.0, alpha: float = 0.5,
                 max_policy_cache: int = 250_000,
                 max_leaf_cache: int = 1_000_000):
        import torch
        self.torch = torch
        self.net = net.to(device).eval()
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.margin_gain = float(margin_gain)
        self.alpha = float(alpha)
        self.max_policy_cache = max(1, int(max_policy_cache))
        self.max_leaf_cache = max(1, int(max_leaf_cache))
        self.policy_cache: dict[str, dict[int, float]] = {}
        self.leaf_cache: dict[str, float] = {}
        self.stats = EvalStats()

    def _forward(self, states: Sequence[GameState]):
        torch = self.torch
        mbs, obs, flats = [], [], []
        for state in states:
            actor = 0 if state.phase == Phase.GAME_OVER else int(state.current_actor)
            mb, ob, flat = encode_state(state, actor)
            mbs.append(mb); obs.append(ob); flats.append(flat)
        mb_t = torch.from_numpy(np.ascontiguousarray(mbs)).to(self.device)
        ob_t = torch.from_numpy(np.ascontiguousarray(obs)).to(self.device)
        fl_t = torch.from_numpy(np.ascontiguousarray(flats)).to(self.device)
        with torch.inference_mode():
            own, opp, win, logits = self.net(mb_t, ob_t, fl_t)
            margin = torch.tanh((own - opp) * self.margin_gain)
            win_v = 2.0 * win - 1.0
            actor_values = ((1.0 - self.alpha) * win_v
                            + self.alpha * win_v.pow(4) * margin).reshape(-1)
        return actor_values.float().cpu().numpy(), logits.float().cpu().numpy()

    def policies(self, states: Iterable[GameState]) -> None:
        if len(self.policy_cache) >= self.max_policy_cache:
            self.policy_cache.clear()
        missing: dict[str, GameState] = {}
        for state in states:
            key = public_state_key(state)
            if key in self.policy_cache:
                self.stats.policy_cache_hits += 1
            else:
                missing.setdefault(key, state)
        items = list(missing.items())
        self.stats.policy_cache_misses += len(items)
        for start in range(0, len(items), self.batch_size):
            chunk = items[start:start + self.batch_size]
            states_chunk = [x[1] for x in chunk]
            self.stats.policy_batch_sizes.append(len(chunk))
            _values, logits = self._forward(states_chunk)
            for (key, state), row in zip(chunk, logits):
                actions = state.legal_actions()
                idxs = np.asarray([encode_action(a, state) for a in actions], dtype=np.int64)
                legal = row[idxs].astype(np.float64)
                legal -= float(legal.max())
                probs = np.exp(legal); probs /= float(probs.sum())
                self.policy_cache[key] = {int(i): float(p) for i, p in zip(idxs, probs)}

    def policy(self, state: GameState) -> dict[int, float]:
        self.policies([state])
        return self.policy_cache[public_state_key(state)]

    def values_p0(self, states: Iterable[GameState]) -> dict[str, float]:
        if len(self.leaf_cache) >= self.max_leaf_cache:
            self.leaf_cache.clear()
        requested = list(states)
        missing: dict[str, GameState] = {}
        for state in requested:
            key = public_state_key(state)
            if state.phase == Phase.GAME_OVER:
                self.leaf_cache.setdefault(key, _terminal_value_p0(state))
            if key in self.leaf_cache:
                self.stats.leaf_cache_hits += 1
            else:
                missing.setdefault(key, state)
        items = list(missing.items())
        self.stats.leaf_cache_misses += len(items)
        for start in range(0, len(items), self.batch_size):
            chunk = items[start:start + self.batch_size]
            self.stats.leaf_batch_sizes.append(len(chunk))
            actor_values, _logits = self._forward([x[1] for x in chunk])
            for (key, state), actor_value in zip(chunk, actor_values):
                actor = int(state.current_actor)
                self.leaf_cache[key] = float(actor_value if actor == 0 else -actor_value)
        return {public_state_key(s): self.leaf_cache[public_state_key(s)] for s in requested}


@dataclass
class Edge:
    pick: int
    action: PickAction | TurnAction
    action_record: dict[str, Any]
    children: list[tuple["Node", float]]
    chance_mode: Optional[str] = None
    value: float = 0.0
    stderr: float = 0.0


@dataclass
class Node:
    state: GameState
    depth: int
    chance_crossings: int
    key: tuple[Any, ...]
    edges: list[Edge] = field(default_factory=list)
    value: Optional[float] = None
    stderr: float = 0.0


@dataclass
class SearchConfig:
    pick_plies: int = 8
    chance_k: int = 8
    seed: int = 0
    placement_top_k: int = 2
    root_search_sims: int = 128
    policy_temperature: float = 0.10
    tie_tolerance: float = 1e-6
    uncertainty_z: float = 1.0


class DenialSearch:
    """Reusable offline searcher.  Evaluator caches survive across positions."""

    def __init__(self, evaluator: AZBatchEvaluator, *, checkpoint_path: str,
                 config: Optional[SearchConfig] = None):
        self.evaluator = evaluator
        self.checkpoint_path = str(Path(checkpoint_path))
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self.config = config or SearchConfig()
        if self.config.pick_plies < 1 or self.config.chance_k < 1:
            raise ValueError("pick_plies and chance_k must be >= 1")
        self._chance_cache: dict[tuple[Any, ...], tuple[list[tuple[int, ...]], str]] = {}
        self._rust_evaluator = None
        self._root_search_cache: dict[tuple[str, str], Any] = {}
        self._node_tt: dict[tuple[Any, ...], Node] = {}
        self._node_tt_max = 250_000

    def _chance_rows(self, bag: Sequence[int]) -> tuple[list[tuple[int, ...]], str]:
        key = (tuple(sorted(int(x) for x in bag)), int(self.config.chance_k),
               int(self.config.seed), 4)
        if key not in self._chance_cache:
            self._chance_cache[key] = chance_rows(
                bag, self.config.chance_k, seed=self.config.seed, drew=4)
        return self._chance_cache[key]

    def _root_seed(self, state: GameState) -> int:
        """Call-order-independent seed derived from the public root and base."""
        digest = bytes.fromhex(public_state_key(state))
        mixed = int.from_bytes(hashlib.blake2b(digest, digest_size=8).digest(), "little")
        return (int(self.config.seed) + mixed) & 0xFFFF_FFFF_FFFF_FFFF

    def clear_root_search_cache(self) -> None:
        self._root_search_cache.clear()

    def _root_search(
        self,
        state: GameState,
        *,
        cache_namespace: str = "primary",
        seed_override: Optional[int] = None,
        use_cache: bool = True,
    ):
        """Memoized advisor-equivalent Rust open-loop root search.

        Normal generation uses the ``primary`` public-state cache and the stable
        root-derived seed.  Validation may use a second, explicitly seeded
        compatibility namespace because the pre-fix 4-ply report serialized a
        different call-order-seeded Q; keeping that auxiliary result is required
        by the strict no-label-change gate.
        """
        if self.config.root_search_sims <= 0:
            return None
        state_key = public_state_key(state)
        cache_key = (str(cache_namespace), state_key)
        if use_cache and cache_key in self._root_search_cache:
            self.evaluator.stats.root_search_cache_hits += 1
            return self._root_search_cache[cache_key]
        seed = self._root_seed(state) if seed_override is None else int(seed_override)
        result = self._root_search_compute(state, seed)
        if use_cache:
            self._root_search_cache[cache_key] = result
        return result

    def _root_search_compute(self, state: GameState, seed: int):
        self.evaluator.stats.root_search_calls += 1
        import kingdomino_rust as kr
        from games.kingdomino.endgame_solver import _rust_state_from_python
        from games.kingdomino.self_play import make_rust_evaluator
        if self._rust_evaluator is None:
            self._rust_evaluator = make_rust_evaluator(
                self.evaluator.net, device=self.evaluator.device,
                margin_gain=self.evaluator.margin_gain, alpha=self.evaluator.alpha)
        children, value0 = kr.advisor_open_loop_search(
            _rust_state_from_python(state), self._rust_evaluator,
            int(self.config.root_search_sims), dirichlet_eps=0.0, cpuct=1.5,
            seed=int(seed) & 0xFFFF_FFFF_FFFF_FFFF,
            leaf_batch=8, alpha=self.evaluator.alpha,
        )
        by_idx = {int(encode_action(a, state)): a for a in state.legal_actions()}
        visits: dict[int, float] = {}
        info: dict[int, tuple[float, Optional[float]]] = {}
        actor = int(state.current_actor)
        for idx, n, value_sum, prior in children:
            if int(idx) not in by_idx:
                continue
            visits[int(idx)] = float(n)
            q_actor = None
            if n:
                q0 = float(value_sum) / float(n)
                q_actor = q0 if actor == 0 else -q0
            info[int(idx)] = (float(prior), q_actor)
        return visits, float(value0), info

    def _root_candidates(self, state: GameState, root_result):
        policy = self.evaluator.policy(state)
        groups: dict[int, list[PickAction | TurnAction]] = {}
        for action in state.legal_actions():
            pick = _pick_key(action)
            groups.setdefault(-1 if pick is None else pick, []).append(action)
        if root_result is None:
            visits: dict[int, float] = {}
            info = {idx: (prior, None) for idx, prior in policy.items()}
        else:
            visits, _value0, info = root_result
        reps: dict[int, list[PickAction | TurnAction]] = {}
        rows: dict[int, dict[str, Any]] = {}
        total_visits = sum(visits.values()) or 1.0
        for pick, actions in groups.items():
            def rank(a):
                idx = int(encode_action(a, state))
                return (visits.get(idx, 0.0), info.get(idx, (policy.get(idx, 0.0), None))[0], -idx)
            rep = max(actions, key=rank)
            rep_idx = int(encode_action(rep, state))
            group_idxs = [int(encode_action(a, state)) for a in actions]
            reps[pick] = [rep]
            rows[pick] = {
                "pick_domino_id": None if pick == -1 else int(pick),
                "representative": action_json(state, rep),
                "group_visits": float(sum(visits.get(i, 0.0) for i in group_idxs)),
                "group_visit_fraction": float(sum(visits.get(i, 0.0) for i in group_idxs) / total_visits),
                "raw_prior": float(sum(policy.get(i, 0.0) for i in group_idxs)),
                "headline_edge": info.get(rep_idx, (0.0, None))[1],
            }
        return reps, rows

    def _node_key(self, state: GameState, depth: int, crossings: int, root_actor: int):
        return (public_state_key(state), int(depth), int(crossings), int(root_actor),
                int(self.config.pick_plies), int(self.config.placement_top_k),
                int(self.config.chance_k), int(self.config.seed))

    def _get_node(self, state: GameState, depth: int, crossings: int,
                  root_actor: int) -> Node:
        key = self._node_key(state, depth, crossings, root_actor)
        node = self._node_tt.get(key)
        if node is None:
            node = Node(state, depth, crossings, key)
            self._node_tt[key] = node
            self.evaluator.stats.node_tt_misses += 1
        else:
            self.evaluator.stats.node_tt_hits += 1
        return node

    def _expand(self, root: Node, root_candidates: dict[int, list]) -> tuple[list[Node], dict[str, Any]]:
        levels: dict[int, list[Node]] = {0: [root]}
        seen_level: dict[int, set[tuple[Any, ...]]] = {0: {root.key}}
        chance_modes = {"enumerated": 0, "sampled": 0}
        pre_reveal_leaves = 0
        for depth in range(self.config.pick_plies):
            nodes = levels.get(depth, [])
            live = [n for n in nodes if n.state.phase != Phase.GAME_OVER and not n.edges]
            self.evaluator.policies(n.state for n in live)
            for node in live:
                state = node.state
                policy = self.evaluator.policy_cache[public_state_key(state)]
                groups: dict[int, list[PickAction | TurnAction]] = {}
                for action in state.legal_actions():
                    pick = _pick_key(action)
                    groups.setdefault(-1 if pick is None else int(pick), []).append(action)
                selected: dict[int, list[PickAction | TurnAction]] = {}
                if depth == 0:
                    selected = root_candidates
                else:
                    # The opponent gets its proven top-2 placement delegation;
                    # our own deeper placements use the top prior representative.
                    top_k = self.config.placement_top_k if int(state.current_actor) != int(root.state.current_actor) else 1
                    for pick, actions in groups.items():
                        selected[pick] = sorted(
                            actions,
                            key=lambda a: (-policy.get(int(encode_action(a, state)), 0.0),
                                           int(encode_action(a, state))),
                        )[:max(1, top_k)]
                for pick, actions in selected.items():
                    for action in actions:
                        child = state.step(action)
                        next_depth = depth + 1
                        dealt = len(state.deck) - len(child.deck) == 4
                        children: list[tuple[Node, float]] = []
                        mode: Optional[str] = None
                        if dealt and next_depth >= self.config.pick_plies:
                            leaf_state = _as_pre_reveal_leaf(child, state.deck)
                            target = self._get_node(leaf_state, next_depth, node.chance_crossings,
                                                    int(root.state.current_actor))
                            children = [(target, 1.0)]
                            pre_reveal_leaves += 1
                        elif dealt:
                            if node.chance_crossings >= 1:
                                raise RuntimeError("8-ply denial search attempted a second interior chance node")
                            rows, mode = self._chance_rows(state.deck)
                            chance_modes[mode] += 1
                            weight = 1.0 / max(1, len(rows))
                            for row in rows:
                                drawn = _replace_draw(child, state.deck, row)
                                target = self._get_node(drawn, next_depth, node.chance_crossings + 1,
                                                        int(root.state.current_actor))
                                children.append((target, weight))
                        else:
                            target = self._get_node(child, next_depth, node.chance_crossings,
                                                    int(root.state.current_actor))
                            children = [(target, 1.0)]
                        node.edges.append(Edge(
                            int(pick), action, action_json(state, action), children, mode))
                        for target, _weight in children:
                            bucket = levels.setdefault(next_depth, [])
                            keys = seen_level.setdefault(next_depth, set())
                            if target.key not in keys:
                                bucket.append(target); keys.add(target.key)
        all_nodes = [n for depth in sorted(levels) for n in levels[depth]]
        structure = {
            "nodes": len(all_nodes),
            "edges": sum(len(n.edges) for n in all_nodes),
            "leaves": sum(1 for n in all_nodes if not n.edges),
            "chance_events": chance_modes,
            "pre_reveal_horizon_leaves": int(pre_reveal_leaves),
            "max_depth": max((n.depth for n in all_nodes), default=0),
            "completed": all(n.state.phase == Phase.GAME_OVER or n.edges
                             for n in all_nodes if n.depth < self.config.pick_plies),
        }
        return all_nodes, structure

    def _backup(self, nodes: Sequence[Node], root: Node) -> None:
        leaves = [n for n in nodes if not n.edges]
        values = self.evaluator.values_p0(n.state for n in leaves)
        for leaf in leaves:
            leaf.value = values[public_state_key(leaf.state)]
            leaf.stderr = 0.0
        for node in sorted(nodes, key=lambda n: n.depth, reverse=True):
            if not node.edges:
                continue
            by_pick: dict[int, list[Edge]] = {}
            for edge in node.edges:
                child_values = np.asarray([float(c.value) for c, _w in edge.children])
                weights = np.asarray([float(w) for _c, w in edge.children])
                edge.value = float(np.dot(child_values, weights))
                propagated_var = sum((float(w) * float(c.stderr)) ** 2 for c, w in edge.children)
                sample_var = 0.0
                if edge.chance_mode == "sampled" and len(child_values) > 1:
                    sample_var = float(np.var(child_values, ddof=1) / len(child_values))
                edge.stderr = math.sqrt(max(0.0, propagated_var + sample_var))
                by_pick.setdefault(edge.pick, []).append(edge)
            actor = int(node.state.current_actor)
            pick_edges: list[Edge] = []
            for candidates in by_pick.values():
                choose = max if actor == 0 else min
                pick_edges.append(choose(candidates, key=lambda e: e.value))
            choose = max if actor == 0 else min
            selected = choose(pick_edges, key=lambda e: e.value)
            node.value, node.stderr = selected.value, selected.stderr

    def search_position(self, state: GameState, *, root_result=None) -> dict[str, Any]:
        started = time.perf_counter()
        # The graph TT intentionally spans positions, but curriculum generation
        # must have bounded memory.  Leaf/policy TTs remain available after this
        # structural reset and hold the compact, reusable inference results.
        if len(self._node_tt) >= self._node_tt_max:
            self._node_tt.clear()
        if state.phase == Phase.GAME_OVER:
            value0 = _terminal_value_p0(state)
            return {"status": "game_over", "policy_target": [],
                    "corrected_value_player0": value0, "corrected_value_actor": None,
                    "provenance": self._provenance({"completed": True}, time.perf_counter() - started)}
        root_actor = int(state.current_actor)
        if root_result is None:
            root_result = self._root_search(state)
        root_candidates, root_rows = self._root_candidates(state, root_result)
        root = self._get_node(state.copy(), 0, 0, root_actor)
        # A root key may repeat across positions.  Its completed graph/value is
        # safe to reuse because all configuration fields are part of the key.
        if root.edges and root.value is not None:
            nodes = list(self._reachable(root))
            structure = {"nodes": len(nodes), "edges": sum(len(n.edges) for n in nodes),
                         "leaves": sum(not n.edges for n in nodes), "completed": True,
                         "tt_root_reuse": True, "chance_events": {"enumerated": 0, "sampled": 0},
                         "pre_reveal_horizon_leaves": 0, "max_depth": max(n.depth for n in nodes)}
        else:
            nodes, structure = self._expand(root, root_candidates)
            self._backup(nodes, root)
            structure["tt_root_reuse"] = False

        root_edges: dict[int, Edge] = {}
        for edge in root.edges:
            current = root_edges.get(edge.pick)
            if current is None or (edge.value > current.value if root_actor == 0 else edge.value < current.value):
                root_edges[edge.pick] = edge
        picks = sorted(root_edges)
        p0_values = [root_edges[p].value for p in picks]
        actor_values = [v if root_actor == 0 else -v for v in p0_values]
        errors = [root_edges[p].stderr for p in picks]
        target = denial_policy_target(
            actor_values, errors, temperature=self.config.policy_temperature,
            tie_tolerance=self.config.tie_tolerance, uncertainty_z=self.config.uncertainty_z)
        best_i = int(np.argmax(actor_values))
        corrected_actor = float(actor_values[best_i])
        corrected_p0 = corrected_actor if root_actor == 0 else -corrected_actor

        headline_pick = max(
            picks,
            key=lambda p: (root_rows[p]["group_visits"], root_rows[p]["raw_prior"], -p),
        )
        for pick, p0v, av, err, prob in zip(picks, p0_values, actor_values, errors, target):
            row = root_rows[pick]
            row.update({
                "searched_value_player0": float(p0v),
                "searched_value_actor": float(av),
                "mc_standard_error": float(err),
                "policy_target": float(prob),
                "fragility": (None if row["headline_edge"] is None
                              else float(row["headline_edge"] - av)),
            })
        headline_row = root_rows[headline_pick]
        output = {
            "status": "ok",
            "state_key": public_state_key(state),
            "actor": root_actor,
            "legal_pick_ids": [None if p == -1 else int(p) for p in picks],
            "per_pick": [root_rows[p] for p in picks],
            "policy_target": [float(x) for x in target],
            "corrected_best_pick": None if picks[best_i] == -1 else int(picks[best_i]),
            "headline_pick": None if headline_pick == -1 else int(headline_pick),
            "corrected_value_player0": float(corrected_p0),
            "corrected_value_actor": float(corrected_actor),
            "fragility": headline_row["fragility"],
            "correction_margin": float(corrected_actor - root_edges[headline_pick].value
                                       * (1.0 if root_actor == 0 else -1.0)),
            "structure": structure,
            "provenance": self._provenance(structure, time.perf_counter() - started),
        }
        return output

    def derive_four_ply_position(self, state: GameState, *, root_result=None) -> dict[str, Any]:
        """Read the exact four-ply cutoff from an already-built eight-ply tree.

        The first four action layers are reused verbatim.  Only the public
        pre-reveal cutoff states are constructed and batch-evaluated, matching
        the independent four-ply search's horizon convention.  If a supplied
        compatibility root result selects a different representative action,
        fall back to an independent four-ply pass rather than reuse unsafely.
        """
        if self.config.pick_plies != 8:
            raise ValueError("derive_four_ply_position requires an active 8-ply configuration")
        started = time.perf_counter()
        root_actor = int(state.current_actor)
        root = self._node_tt.get(self._node_key(state, 0, 0, root_actor))
        if root is None or root.value is None or not root.edges:
            raise RuntimeError("eight-ply root must be searched before deriving four plies")
        root_candidates, root_rows = self._root_candidates(state, root_result)
        built_reps = {edge.pick: int(edge.action_record["action_idx"]) for edge in root.edges}
        wanted_reps = {
            pick: int(encode_action(actions[0], state))
            for pick, actions in root_candidates.items()
        }
        if built_reps != wanted_reps:
            old = self.config
            self.config = SearchConfig(
                pick_plies=4, chance_k=old.chance_k, seed=old.seed,
                placement_top_k=old.placement_top_k,
                root_search_sims=old.root_search_sims,
                policy_temperature=old.policy_temperature,
                tie_tolerance=old.tie_tolerance,
                uncertainty_z=old.uncertainty_z,
            )
            try:
                return self.search_position(state, root_result=root_result)
            finally:
                self.config = old

        interior: dict[tuple[Any, ...], Node] = {}
        stack = [root]
        while stack:
            node = stack.pop()
            if node.key in interior or node.depth >= 4:
                continue
            interior[node.key] = node
            if node.depth < 3:
                stack.extend(child for edge in node.edges for child, _w in edge.children)
        self.evaluator.stats.node_tt_pass_hits += len(interior)

        cutoff_by_edge: dict[int, GameState] = {}
        cutoff_unique: dict[str, GameState] = {}
        pre_reveal_count = 0
        for node in interior.values():
            if node.depth != 3 or node.state.phase == Phase.GAME_OVER:
                continue
            for edge in node.edges:
                child = node.state.step(edge.action)
                dealt = len(node.state.deck) - len(child.deck) == 4
                cutoff = (_as_pre_reveal_leaf(child, node.state.deck) if dealt else child)
                if dealt:
                    pre_reveal_count += 1
                cutoff_by_edge[id(edge)] = cutoff
                cutoff_unique.setdefault(public_state_key(cutoff), cutoff)
        cutoff_values = self.evaluator.values_p0(cutoff_unique.values())

        node_values: dict[tuple[Any, ...], tuple[float, float]] = {}
        edge_values: dict[int, tuple[float, float]] = {}

        def value_node(node: Node) -> tuple[float, float]:
            cached = node_values.get(node.key)
            if cached is not None:
                return cached
            if node.state.phase == Phase.GAME_OVER:
                result = (_terminal_value_p0(node.state), 0.0)
                node_values[node.key] = result
                return result
            by_pick: dict[int, list[tuple[Edge, float, float]]] = {}
            for edge in node.edges:
                if node.depth == 3:
                    cutoff = cutoff_by_edge[id(edge)]
                    value, stderr = cutoff_values[public_state_key(cutoff)], 0.0
                else:
                    values = [(value_node(child), weight) for child, weight in edge.children]
                    value = sum(v * weight for (v, _se), weight in values)
                    stderr = math.sqrt(sum((weight * se) ** 2
                                           for (_v, se), weight in values))
                edge_values[id(edge)] = (float(value), float(stderr))
                by_pick.setdefault(edge.pick, []).append((edge, float(value), float(stderr)))
            choose = max if int(node.state.current_actor) == 0 else min
            pick_winners = [choose(items, key=lambda item: item[1]) for items in by_pick.values()]
            selected = choose(pick_winners, key=lambda item: item[1])
            result = (selected[1], selected[2])
            node_values[node.key] = result
            return result

        value_node(root)
        root_edges: dict[int, tuple[float, float]] = {}
        for edge in root.edges:
            value, stderr = edge_values[id(edge)]
            current = root_edges.get(edge.pick)
            if current is None or (value > current[0] if root_actor == 0 else value < current[0]):
                root_edges[edge.pick] = (value, stderr)
        picks = sorted(root_edges)
        p0_values = [root_edges[p][0] for p in picks]
        actor_values = [v if root_actor == 0 else -v for v in p0_values]
        errors = [root_edges[p][1] for p in picks]
        target = denial_policy_target(
            actor_values, errors, temperature=self.config.policy_temperature,
            tie_tolerance=self.config.tie_tolerance, uncertainty_z=self.config.uncertainty_z)
        best_i = int(np.argmax(actor_values))
        corrected_actor = float(actor_values[best_i])
        corrected_p0 = corrected_actor if root_actor == 0 else -corrected_actor
        headline_pick = max(
            picks,
            key=lambda p: (root_rows[p]["group_visits"], root_rows[p]["raw_prior"], -p),
        )
        for pick, p0v, av, err, probability in zip(
                picks, p0_values, actor_values, errors, target):
            row = root_rows[pick]
            row.update({
                "searched_value_player0": float(p0v),
                "searched_value_actor": float(av),
                "mc_standard_error": float(err),
                "policy_target": float(probability),
                "fragility": (None if row["headline_edge"] is None
                              else float(row["headline_edge"] - av)),
            })
        terminal_interior = sum(1 for node in interior.values()
                                if node.state.phase == Phase.GAME_OVER)
        structure = {
            "nodes": len(interior) + len(cutoff_unique),
            "edges": sum(len(node.edges) for node in interior.values()),
            "leaves": len(cutoff_unique) + terminal_interior,
            "chance_events": {"enumerated": 0, "sampled": 0},
            "pre_reveal_horizon_leaves": int(pre_reveal_count),
            "max_depth": 4,
            "completed": True,
            "tt_root_reuse": False,
        }
        provenance = self._provenance(structure, time.perf_counter() - started)
        provenance["pick_plies"] = 4
        headline_value = root_edges[headline_pick][0]
        headline_actor_value = headline_value if root_actor == 0 else -headline_value
        return {
            "status": "ok",
            "state_key": public_state_key(state),
            "actor": root_actor,
            "legal_pick_ids": [None if p == -1 else int(p) for p in picks],
            "per_pick": [root_rows[p] for p in picks],
            "policy_target": [float(x) for x in target],
            "corrected_best_pick": None if picks[best_i] == -1 else int(picks[best_i]),
            "headline_pick": None if headline_pick == -1 else int(headline_pick),
            "corrected_value_player0": float(corrected_p0),
            "corrected_value_actor": float(corrected_actor),
            "fragility": root_rows[headline_pick]["fragility"],
            "correction_margin": float(corrected_actor - headline_actor_value),
            "structure": structure,
            "provenance": provenance,
        }

    @staticmethod
    def _reachable(root: Node):
        stack, seen = [root], set()
        while stack:
            node = stack.pop()
            if node.key in seen:
                continue
            seen.add(node.key); yield node
            stack.extend(c for e in node.edges for c, _w in e.children)

    def _provenance(self, structure: dict[str, Any], elapsed: float) -> dict[str, Any]:
        return {
            "engine": "offline-pick-denial-expectiminimax-v1",
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "root_search_sims": int(self.config.root_search_sims),
            "pick_plies": int(self.config.pick_plies),
            "chance_k": int(self.config.chance_k),
            "chance_handling": "enumerate iff C(remaining,4)<=k; otherwise CRN sample",
            "horizon_boundary": "order-blind-pre-reveal",
            "placement_delegation": f"AZ policy; opponent top-{self.config.placement_top_k}, takes better subtree",
            "actor_frame": ACTOR_FRAME,
            "value_storage_frame": "player-0",
            "official_cascade_version": CASCADE_VERSION,
            "policy_target": {
                "temperature": float(self.config.policy_temperature),
                "tie_tolerance": float(self.config.tie_tolerance),
                "uncertainty_z": float(self.config.uncertainty_z),
            },
            "completed_structure": bool(structure.get("completed", False)),
            "elapsed_seconds": float(elapsed),
        }


def load_checkpoint_network(path: str | Path, device: str):
    import torch
    from games.kingdomino.network import KingdominoNet
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    if not isinstance(config, dict):
        config = {}
    net = KingdominoNet(
        channels=int(config.get("channels", 96)),
        blocks=int(config.get("blocks", 8)),
        bilinear_dim=int(config.get("bilinear_dim", 64)),
    )
    net.load_state_dict(state_dict)
    return net.to(device).eval(), config


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    return None if not values else float(np.percentile(np.asarray(values), q))


def _labels_equal_ignoring_elapsed(left: Sequence[dict], right: Sequence[dict]) -> bool:
    """Strict parsed-label equality with only embedded wall time removed."""
    def clean(items):
        out = copy.deepcopy(list(items))
        for label in out:
            label.get("provenance", {}).pop("elapsed_seconds", None)
        return out
    return clean(left) == clean(right)


def generate_az_midgame_positions(
    search: DenialSearch,
    *,
    count: int,
    seed: int,
    min_deck: int = 8,
    max_deck: int = 28,
) -> list[tuple[GameState, dict[str, Any]]]:
    """Generate real greedy AZ-MCTS trajectories and sample round starts."""
    out: list[tuple[GameState, dict[str, Any]]] = []
    game_seed = int(seed)
    while len(out) < count:
        state = GameState.new(seed=game_seed)
        ply = 0
        while state.phase != Phase.GAME_OVER:
            # No later position from this game can re-enter the requested band.
            # Avoid spending trajectory search on the exact-solvable tail.
            if len(state.deck) < min_deck:
                break
            if (state.phase == Phase.PLACE_AND_SELECT and state.actor_index == 0
                    and min_deck <= len(state.deck) <= max_deck):
                out.append((state.copy(), {"game_seed": game_seed, "trajectory_ply": ply,
                                           "deck_count": len(state.deck)}))
                if len(out) >= count:
                    break
            # Pre-fix trajectory generation used the fixed base seed because
            # _position_serial advanced only in label search.  Keep that exact
            # trajectory while memoizing identical states in its own namespace.
            result = search._root_search(
                state, cache_namespace="trajectory",
                seed_override=search.config.seed)
            if result is None:
                policy = search.evaluator.policy(state)
                action = max(state.legal_actions(), key=lambda a: policy[int(encode_action(a, state))])
            else:
                visits, _v0, info = result
                action = max(
                    state.legal_actions(),
                    key=lambda a: (visits.get(int(encode_action(a, state)), 0.0),
                                   info.get(int(encode_action(a, state)), (0.0, None))[0]),
                )
            state = state.step(action); ply += 1
        game_seed += 1
    return out[:count]


def run_advisor_draft_matrix_baseline(
    search: DenialSearch,
    state: GameState,
    *,
    sims: int,
    budget_seconds: float,
    seed: int,
    root_result=None,
) -> dict[str, Any]:
    """Run the existing live-advisor ``_draft_matrix`` on an offline state.

    This adapter is deliberately confined to validation.  The label engine
    above has no FastAPI/request dependency; only the requested incremental
    comparison instantiates the advisor's request model.
    """
    from games.kingdomino.web_app import RecommendRequest, _draft_matrix

    actions = state.legal_actions()
    root = root_result if root_result is not None else search._root_search(state)
    if root is None:
        policy = search.evaluator.policy(state)
        visits_idx = {idx: p for idx, p in policy.items()}
        info_idx = {idx: (p, None) for idx, p in policy.items()}
    else:
        visits_idx, _value0, raw_info = root
        # _draft_matrix consumes (prior, q_win_probability), whereas the
        # offline helper stores q in actor-edge form.
        info_idx = {
            idx: (prior, None if q_actor is None else (float(q_actor) + 1.0) / 2.0)
            for idx, (prior, q_actor) in raw_info.items()
        }
    visits = {a: visits_idx.get(int(encode_action(a, state)), 0.0) for a in actions}
    info = {a: info_idx.get(int(encode_action(a, state)), (0.0, None)) for a in actions}
    req = RecommendRequest(
        engine="nn", checkpoint_path=search.checkpoint_path,
        device=search.evaluator.device, seed=int(seed),
        nn_sims=max(1, int(search.config.root_search_sims)),
        draft_matrix=True, draft_search_sims=max(50, int(sims)),
        draft_budget_secs=float(budget_seconds),
    )
    matrix = _draft_matrix(
        state, req, net=search.evaluator.net,
        checkpoint_path=search.checkpoint_path,
        actions=actions, visit_counts=visits, priors_by_action=info,
    )
    if not matrix or not matrix.get("rows"):
        return {"status": "unavailable", "partial": True, "per_pick": [],
                "policy_target": [], "corrected_best_pick": None,
                "corrected_value_actor": None, "raw": matrix}
    usable = [r for r in matrix["rows"] if r.get("robust_edge") is not None]
    if not usable:
        return {"status": "partial", "partial": True, "per_pick": matrix["rows"],
                "policy_target": [], "corrected_best_pick": None,
                "corrected_value_actor": None, "raw": matrix}
    usable.sort(key=lambda r: int(r["pick_domino_id"]))
    values = [float(r["robust_edge"]) for r in usable]
    policy = denial_policy_target(
        values, temperature=search.config.policy_temperature,
        tie_tolerance=search.config.tie_tolerance,
        uncertainty_z=search.config.uncertainty_z)
    best = int(np.argmax(values))
    for row, probability in zip(usable, policy):
        row["policy_target"] = float(probability)
    return {
        "status": "ok" if not matrix.get("partial") else "partial",
        "partial": bool(matrix.get("partial")),
        "per_pick": usable,
        "policy_target": policy,
        "corrected_best_pick": int(usable[best]["pick_domino_id"]),
        "corrected_value_actor": float(values[best]),
        "search_sims": int(matrix.get("search_sims", sims)),
        "raw": matrix,
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint)
    net, checkpoint_cfg = load_checkpoint_network(checkpoint, args.device)
    evaluator = AZBatchEvaluator(
        net, device=args.device, batch_size=args.leaf_batch_size,
        margin_gain=float(checkpoint_cfg.get("margin_gain", 2.0)),
        alpha=float(checkpoint_cfg.get("alpha", 0.5)))
    config = SearchConfig(
        pick_plies=8, chance_k=args.chance_k, seed=args.seed,
        placement_top_k=2, root_search_sims=args.search_sims,
        policy_temperature=args.policy_temperature)
    search = DenialSearch(evaluator, checkpoint_path=str(checkpoint), config=config)
    positions = generate_az_midgame_positions(
        search, count=args.positions, seed=args.seed,
        min_deck=args.min_deck, max_deck=args.max_deck)

    # The throughput timer excludes trajectory acquisition.  Reset counters at
    # the same boundary so reuse metrics describe label/ablation/advisor work,
    # while retaining the warmed cache contents themselves.
    evaluator.stats = EvalStats()
    started = time.perf_counter()
    labels = []
    one_round_labels = []
    pre_chance_labels = []
    for index, (state, source) in enumerate(positions):
        # Preserve the pre-fix serialized labels exactly: the old 8-ply pass
        # used serial 2*i+1, while the old 4-ply/advisor pair shared serial
        # 2*i+2.  These explicit deterministic compatibility seeds are keyed by
        # public state in separate caches; normal search uses the root-derived
        # call-order-independent seed.
        primary_seed = args.seed + 104729 * (2 * index + 1)
        auxiliary_seed = args.seed + 104729 * (2 * index + 2)
        primary_root = search._root_search(
            state, cache_namespace="validation_primary",
            seed_override=primary_seed)
        label = search.search_position(state, root_result=primary_root)
        label["position_index"] = index; label["source"] = source
        labels.append(label)
        auxiliary_root = search._root_search(
            state, cache_namespace="validation_auxiliary",
            seed_override=auxiliary_seed)
        pre_chance = search.derive_four_ply_position(
            state, root_result=auxiliary_root)
        pre_chance["position_index"] = index
        pre_chance_labels.append(pre_chance)
        auxiliary_root = search._root_search(
            state, cache_namespace="validation_auxiliary",
            seed_override=auxiliary_seed)
        one = run_advisor_draft_matrix_baseline(
            search, state, sims=args.draft_search_sims,
            budget_seconds=args.draft_budget_seconds,
            seed=args.seed + index, root_result=auxiliary_root)
        one["position_index"] = index
        one_round_labels.append(one)
        print(f"denial validation {index + 1}/{len(positions)}: "
              f"best={label.get('corrected_best_pick')} fragility={label.get('fragility')}")
    elapsed = time.perf_counter() - started

    ok = [x for x in labels if x["status"] == "ok"]
    policy_valid = [abs(sum(x["policy_target"]) - 1.0) <= 1e-6
                    and all(0.0 <= p <= 1.0 for p in x["policy_target"]) for x in ok]
    values_valid = [-1.000001 <= x["corrected_value_player0"] <= 1.000001 for x in ok]
    picks_valid = [set(x["legal_pick_ids"]) == {r["pick_domino_id"] for r in x["per_pick"]}
                   for x in ok]
    fragilities = [float(x["fragility"]) for x in ok if x.get("fragility") is not None]
    material = [x for x in ok if x["corrected_best_pick"] != x["headline_pick"]
                and x["correction_margin"] >= args.material_margin]
    starvation = []
    for x in ok:
        by_pick = {r["pick_domino_id"]: r for r in x["per_pick"]}
        row = by_pick.get(x["corrected_best_pick"])
        if row and row["raw_prior"] <= args.starved_prior and row["policy_target"] > row["raw_prior"]:
            starvation.append(x)
    high_fragility_threshold = 0.20
    high_fragility = [x for x in ok if x.get("fragility") is not None
                      and float(x["fragility"]) >= high_fragility_threshold]
    high_fragility_starvation = []
    for x in high_fragility:
        row = next((r for r in x["per_pick"]
                    if r["pick_domino_id"] == x["corrected_best_pick"]), None)
        if row and row["raw_prior"] <= args.starved_prior and row["policy_target"] > row["raw_prior"]:
            high_fragility_starvation.append(x)
    high_fragility_material = [x for x in high_fragility
                               if x["corrected_best_pick"] != x["headline_pick"]
                               and x["correction_margin"] >= args.material_margin]

    comparisons = []
    for eight, one in zip(labels, one_round_labels):
        eight_policy = {r["pick_domino_id"]: float(r["policy_target"])
                        for r in eight.get("per_pick", [])}
        one_policy = {r["pick_domino_id"]: float(r.get("policy_target", 0.0))
                      for r in one.get("per_pick", [])}
        picks = set(eight_policy) | set(one_policy)
        one_value = one.get("corrected_value_actor")
        comparisons.append({
            "position_index": eight["position_index"],
            "eight_ply_best": eight.get("corrected_best_pick"),
            "one_round_best": one.get("corrected_best_pick"),
            "one_round_status": one.get("status"),
            "best_changed": eight.get("corrected_best_pick") != one.get("corrected_best_pick"),
            "value_delta_actor": (None if one_value is None else
                                  float(eight.get("corrected_value_actor", 0.0) - one_value)),
            "policy_l1": (None if not one_policy else
                          float(sum(abs(eight_policy.get(p, 0.0) - one_policy.get(p, 0.0))
                                    for p in picks))),
        })
    complete_comparisons = [c for c in comparisons if c["value_delta_actor"] is not None]
    changed = [c for c in complete_comparisons if c["best_changed"]]
    cross_chance = []
    for eight, before in zip(labels, pre_chance_labels):
        cross_chance.append({
            "position_index": eight["position_index"],
            "eight_ply_best": eight.get("corrected_best_pick"),
            "pre_chance_best": before.get("corrected_best_pick"),
            "best_changed": eight.get("corrected_best_pick") != before.get("corrected_best_pick"),
            "value_delta_actor": float(eight["corrected_value_actor"]
                                       - before["corrected_value_actor"]),
        })
    hand_cases = sorted(cross_chance,
                        key=lambda c: abs(c["value_delta_actor"]), reverse=True)[:2]
    for case in hand_cases:
        eight = labels[case["position_index"]]
        case["trace_check"] = {
            "interior_chance_events": eight["structure"]["chance_events"],
            "max_depth": eight["structure"]["max_depth"],
            "pre_reveal_leaf": eight["structure"]["pre_reveal_horizon_leaves"] > 0,
            "interpretation": "This is an 8-ply versus identical 4-ply structural ablation. The only added public event is the shared next-round draw followed by claim-order play, so the delta cannot come only from the currently visible tile.",
        }

    leaf_batches = evaluator.stats.leaf_batch_sizes
    root_requests = evaluator.stats.root_search_calls + evaluator.stats.root_search_cache_hits
    invariant_gate = None
    if args.invariant_baseline:
        baseline_path = Path(args.invariant_baseline)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        n_gate = len(baseline.get("labels", []))
        checks = {
            "labels": _labels_equal_ignoring_elapsed(
                baseline.get("labels", []), labels[:n_gate]),
            "four_ply_labels": _labels_equal_ignoring_elapsed(
                baseline.get("four_ply_labels", []), pre_chance_labels[:n_gate]),
            "one_round_labels": _labels_equal_ignoring_elapsed(
                baseline.get("one_round_labels", []), one_round_labels[:n_gate]),
        }
        invariant_gate = {
            "baseline_path": str(baseline_path),
            "positions": n_gate,
            "seed": args.seed,
            "ignored_fields": ["provenance.elapsed_seconds"],
            "checks": checks,
            "passed": all(checks.values()),
        }
        if not invariant_gate["passed"]:
            raise RuntimeError(f"throughput invariant gate failed: {checks}")
    report = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "scope": "labels and validation only; no retraining or mixture construction",
        "invariant_gate": invariant_gate,
        "configuration": vars(args),
        "provenance": {
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_config": checkpoint_cfg, "actor_frame": ACTOR_FRAME,
            "official_cascade_version": CASCADE_VERSION,
            "trajectory": "greedy current-best AZ open-loop MCTS",
            "target_band": {"min_deck": args.min_deck, "max_deck": args.max_deck,
                            "opening_excluded": True, "exact_tail_excluded": True},
        },
        "sanity": {
            "positions": len(labels), "policy_targets_valid": all(policy_valid),
            "corrected_values_in_range": all(values_valid), "legal_picks_only": all(picks_valid),
            "forced_pick_contract": "covered by games/kingdomino/tests/test_denial_search.py",
            "game_over_contract": "terminal returns outcome and an empty policy",
            "near_frontier_positions": sum(x["source"]["deck_count"] <= 8 for x in labels),
            "completed_structures": sum(bool(x["structure"]["completed"]) for x in ok),
        },
        "denial": {
            "fragility": {"count": len(fragilities), "min": min(fragilities, default=None),
                           "median": _percentile(fragilities, 50), "p90": _percentile(fragilities, 90),
                           "max": max(fragilities, default=None)},
            "material_margin": args.material_margin,
            "material_corrections": len(material),
            "starved_prior_threshold": args.starved_prior,
            "starved_picks_upweighted": len(starvation),
            "high_fragility_threshold": high_fragility_threshold,
            "high_fragility_positions": len(high_fragility),
            "high_fragility_material_corrections": len(high_fragility_material),
            "high_fragility_starved_picks_upweighted": len(high_fragility_starvation),
            "finding": ("High-fragility prior-starvation correction established"
                        if high_fragility_starvation else
                        "High-fragility prior-starvation correction NOT established at this validation budget"),
        },
        "chance_and_turn_order": {
            "order_blind_sorted_bag": True, "common_random_numbers": True,
            "exact_rule": "C(remaining,4) <= k", "interior_chance_nodes": 1,
            "four_ply_structural_ablation": cross_chance,
            "corrected_best_changed_vs_four_ply": sum(c["best_changed"] for c in cross_chance),
            "mean_abs_value_delta_vs_four_ply": statistics.fmean(
                abs(c["value_delta_actor"]) for c in cross_chance),
            "hand_checked_cases": hand_cases,
        },
        "incremental_over_one_round": {
            "baseline": "existing games.kingdomino.web_app._draft_matrix",
            "draft_search_sims": args.draft_search_sims,
            "draft_budget_seconds": args.draft_budget_seconds,
            "positions_requested": len(comparisons),
            "positions_compared": len(complete_comparisons),
            "partial_or_unavailable": len(comparisons) - len(complete_comparisons),
            "corrected_best_changed": len(changed),
            "corrected_best_changed_rate": len(changed) / max(1, len(complete_comparisons)),
            "mean_abs_value_delta_actor": (statistics.fmean(abs(c["value_delta_actor"])
                                                             for c in complete_comparisons)
                                            if complete_comparisons else None),
            "mean_policy_l1": (statistics.fmean(c["policy_l1"] for c in complete_comparisons
                                                  if c["policy_l1"] is not None)
                               if any(c["policy_l1"] is not None for c in complete_comparisons)
                               else None),
            "comparisons": comparisons,
        },
        "throughput": {
            "elapsed_seconds": elapsed,
            "positions_per_hour": len(labels) * 3600.0 / max(elapsed, 1e-9),
            "projection_10000_positions_hours": (10000.0 * elapsed / max(1, len(labels))) / 3600.0,
            "before_positions_per_hour": 99.03305083186183,
            "before_projection_10000_positions_hours": 100.97639036666644,
            "speedup_vs_original_validation": ((len(labels) * 3600.0 / max(elapsed, 1e-9))
                                                 / 99.03305083186183),
            "leaf_eval_batches": len(leaf_batches),
            "leaf_eval_batch_sizes": leaf_batches,
            "mean_leaf_eval_batch_size": statistics.fmean(leaf_batches) if leaf_batches else 0.0,
            "max_leaf_eval_batch_size": max(leaf_batches, default=0),
            "leaf_tt_hits": evaluator.stats.leaf_cache_hits,
            "leaf_tt_misses": evaluator.stats.leaf_cache_misses,
            "leaf_tt_note": "Horizon boards are path-unique, so leaf-cache reuse is structurally approximately zero and is not the throughput lever.",
            "policy_tt_hits": evaluator.stats.policy_cache_hits,
            "policy_tt_misses": evaluator.stats.policy_cache_misses,
            "node_tt_hits": evaluator.stats.node_tt_hits,
            "node_tt_misses": evaluator.stats.node_tt_misses,
            "node_tt_pass_hits": evaluator.stats.node_tt_pass_hits,
            "root_search_calls": evaluator.stats.root_search_calls,
            "root_search_cache_hits": evaluator.stats.root_search_cache_hits,
            "root_search_requests": root_requests,
            "root_search_cache_hit_rate": (evaluator.stats.root_search_cache_hits
                                           / max(1, root_requests)),
            "root_search_compatibility_note": "Strict preservation of legacy 4-ply/advisor Q metadata requires one auxiliary root seed, so validation computes two roots and eliminates the third solve; normal repeated search uses one public-state-keyed primary result.",
            "policy_eval_batch_sizes": evaluator.stats.policy_batch_sizes,
        },
        "routing": {
            "if_successful": "Proceed to equal-compute control-vs-treatment curriculum retrain; pre-register fragility drop on frozen held-out positions plus head-to-head strength.",
            "if_little_incremental_gain": "Prefer the cheaper current-round miner or investigate why the extra round is inert before retraining.",
            "observed_decision": ("proceed_to_control_treatment" if high_fragility_starvation
                                  and material and changed else
                                  "hold_retrain_and_investigate_signal_or_budget"),
            "action_taken": "none",
        },
        "labels": labels,
        "four_ply_labels": pre_chance_labels,
        "one_round_labels": one_round_labels,
    }
    return report


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CURRENT_BEST))
    parser.add_argument("--output", default="runs/kingdomino/denial_search/validation.json")
    parser.add_argument("--positions", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--search-sims", type=int, default=64)
    parser.add_argument("--chance-k", type=int, default=8)
    parser.add_argument("--leaf-batch-size", type=int, default=512)
    parser.add_argument("--draft-search-sims", type=int, default=50)
    parser.add_argument("--draft-budget-seconds", type=float, default=2.0)
    parser.add_argument("--policy-temperature", type=float, default=0.10)
    parser.add_argument("--material-margin", type=float, default=0.03)
    parser.add_argument("--starved-prior", type=float, default=0.10)
    parser.add_argument("--min-deck", type=int, default=8)
    parser.add_argument("--max-deck", type=int, default=28)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--invariant-baseline", default="")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    report = run_validation(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output} ({report['sanity']['positions']} positions, "
          f"{report['throughput']['positions_per_hour']:.1f} positions/hour)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
