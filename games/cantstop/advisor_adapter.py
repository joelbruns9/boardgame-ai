"""Can't Stop serving adapter for games.advisor; no alternate search policy."""
import hashlib
import json
import os
from pathlib import Path
import threading

from games.advisor import ActionStats, ActionView, EngineSpec, SearchSnapshot
from .engine import (COLUMNS, COLUMN_HEIGHTS, GameState, Phase, RuleSet,
                     apply_move, can_stop, legal_moves)
from .snapshot import snapshot

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "runs/p4_pilot/iter_0080.pt"
PHASES = {"diceChoice": Phase.AWAIT_MOVE, "continueChoice": Phase.AWAIT_DECISION}


def wins_on_stop(state):
    """A legal bank would finish the game, so no roll value is needed."""
    return can_stop(state) and (
        len(state.claimed_columns(state.active_player))
        + sum(pos >= COLUMN_HEIGHTS[col] for col, pos in state.runners.items())
        >= state.rules.columns_to_win
    )


def integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def positions(raw, name):
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be an object")
    out = {}
    for key, value in raw.items():
        c = int(key)
        if str(c) != str(key) or c not in COLUMNS or c in out:
            raise ValueError(f"invalid/duplicate column in {name}")
        p = integer(value, name)
        if not 0 <= p <= COLUMN_HEIGHTS[c]:
            raise ValueError(f"invalid height in {name}, column {c}")
        out[c] = p
    return out


def parse_state(payload):
    if payload.get("format") == "bga-cantstop-v1":
        from .bga_extract import normalize_capture
        payload = normalize_capture(payload)
    rules = payload["rules"]
    if type(rules["blocking"]) is not bool:
        raise ValueError("blocking must be an explicit boolean")
    r = RuleSet(integer(rules["num_players"], "num_players"),
                integer(rules["columns_to_win"], "columns_to_win"), rules["blocking"])
    state = GameState(r)
    state.active_player = integer(payload["active_player"], "active_player")
    if not 0 <= state.active_player < r.num_players:
        raise ValueError("active player is not seated")
    if payload["phase"] not in PHASES:
        raise ValueError("only diceChoice and continueChoice are supported")
    state.phase = PHASES[payload["phase"]]
    progress, claimed = payload["progress"], payload["claimed"]
    if len(progress) != r.num_players or len(claimed) != r.num_players:
        raise ValueError("progress/claims must include every player")
    for player, row in enumerate(progress):
        state.progress[player].update(positions(row, "progress"))
    for player, cols in enumerate(claimed):
        for col in cols:
            col = integer(col, "claimed column")
            if col not in COLUMNS or state.claimed_by[col] is not None:
                raise ValueError("invalid or duplicate column claim")
            state.claimed_by[col] = player
    for col in COLUMNS:
        saved = [row[col] for row in state.progress]
        if state.claimed_by[col] is not None:
            if any(saved):
                raise ValueError("claimed columns must have zero saved progress")
        elif any(p == COLUMN_HEIGHTS[col] for p in saved):
            raise ValueError("saved top marker must be a claim")
        if r.blocking and len([p for p in saved if p]) != len(set(p for p in saved if p)):
            raise ValueError("saved markers overlap under blocking rules")
    if any(len(state.claimed_columns(p)) >= r.columns_to_win for p in range(r.num_players)):
        raise ValueError("game already won; decision capture is stale")
    state.runners = positions(payload["runners"], "runners")
    if len(state.runners) > 3:
        raise ValueError("more than three runners")
    for c, pos in state.runners.items():
        if state.claimed_by[c] is not None or pos <= state.progress[state.active_player][c]:
            raise ValueError("runner must advance saved progress on an unclaimed column")
    dice = payload.get("dice", [])
    if state.phase == Phase.AWAIT_MOVE:
        if not isinstance(dice, list) or len(dice) != 4:
            raise ValueError("diceChoice needs four dice")
        if any(type(d) is not int or not 1 <= d <= 6 for d in dice):
            raise ValueError("invalid dice")
        state.dice = tuple(sorted(dice))
        if not legal_moves(state, state.dice):
            raise ValueError("diceChoice has no legal moves; capture may be stale")
    else:
        if dice:
            raise ValueError("continueChoice must not reuse consumed dice")
        if not state.runners:
            raise ValueError("continueChoice needs at least one runner")
    return state


class CantStopAdvisor:
    game_id = "cantstop"

    def __init__(self, checkpoint=None, evaluator=None, cache_entries=4, cache_positions=100_000, cache_ttl=900):
        self.checkpoint = Path(checkpoint or os.environ.get("CANTSTOP_ADVISOR_CHECKPOINT", DEFAULT_CHECKPOINT)).resolve()
        self._evaluator_override = evaluator
        self._cache = {}
        self._lock = threading.Lock()
        from .advisor_cache import TurnCache
        self.turn_cache = TurnCache(self._lock, cache_entries, cache_positions, cache_ttl)

    state_from_wire = staticmethod(parse_state)

    def state_to_public(self, state):
        return {"rules": {"num_players": state.rules.num_players,
                          "columns_to_win": state.rules.columns_to_win,
                          "blocking": state.rules.blocking},
                "active_player": state.active_player,
                "phase": next(k for k, v in PHASES.items() if v == state.phase),
                "progress": [{str(c): p for c, p in row.items() if p} for row in state.progress],
                "claimed": [sorted(state.claimed_columns(p)) for p in range(state.rules.num_players)],
                "runners": {str(c): p for c, p in state.runners.items()},
                "dice": list(state.dice or [])}

    def state_key(self, state):
        return hashlib.sha256(json.dumps(snapshot(state)).encode()).hexdigest()

    def action_views(self, state):
        if state.phase == Phase.AWAIT_MOVE:
            views = []
            for move in sorted(legal_moves(state, state.dice)):
                child = state.clone()
                apply_move(child, move)
                for decision in ("stop", "roll"):
                    if decision == "roll" and wins_on_stop(child):
                        continue
                    if decision == "stop" and not can_stop(child):
                        continue
                    views.append(ActionView("move:" + ",".join(map(str, move)) + "|" + decision,
                                            "Advance " + " + ".join(map(str, move)),
                                            "move", {"columns": list(move),
                                                     "decision": decision,
                                                     "stop_legal": can_stop(child),
                                                     "wins_game": wins_on_stop(child),
                                                     "after_move": self.state_to_public(child)}))
            return views
        if wins_on_stop(state):
            return [ActionView("stop", "Stop and win", "stop", {"wins_game": True})]
        actions = [ActionView("roll", "Roll again", "roll")]
        if can_stop(state):
            actions.append(ActionView("stop", "Stop and bank", "stop"))
        return actions

    def engines(self):
        return {"auto": EngineSpec("auto", "Learned turn solver", needs_checkpoint=True,
                                   default_sims=1, streaming=False),
                "heuristic": EngineSpec("heuristic", "Heuristic turn solver",
                                        default_sims=1, streaming=False)}

    def evaluator(self, req):
        if req.engine == "heuristic":
            from .solver import ProgressHeuristic
            return ProgressHeuristic()
        if req.engine != "auto":
            raise ValueError("unknown engine")
        if self._evaluator_override is not None:
            return self._evaluator_override
        from .model import NetEvaluator, load_net
        path = Path(req.checkpoint_path).resolve() if req.checkpoint_path else self.checkpoint
        device = os.environ.get("CANTSTOP_ADVISOR_DEVICE", req.device)
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, device)
        if key not in self._cache:
            self._cache = {key: NetEvaluator(load_net(path, device=device), device=device)}
        return self._cache[key]

    def solver_for(self, state, req):
        """Called under the adapter lock; no network work occurs on a cache hit."""
        from .rust_solver import RustTurnSolver
        table, turn = req.options.get("table_id"), req.options.get("turn_id")
        scope = (table, turn) if all(isinstance(x, str) and 0 < len(x) <= 256 for x in (table, turn)) else None
        if req.engine == "heuristic":
            model = ("heuristic",)
        elif req.engine == "auto" and self._evaluator_override is not None:
            model = ("override", id(self._evaluator_override))
        elif req.engine == "auto":
            path = Path(req.checkpoint_path).resolve() if req.checkpoint_path else self.checkpoint
            stat = path.stat()
            model = (str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size,
                     os.environ.get("CANTSTOP_ADVISOR_DEVICE", req.device))
        else:
            raise ValueError("unknown engine")
        # Fixed saved board, absolute player order and rules must match exactly.
        base = json.dumps(snapshot(state)[:4])
        key = (table, turn, base, model) if scope else None
        solver = self.turn_cache.get(key, state) if key else None
        if solver is not None:
            return solver, True
        solver = RustTurnSolver(state, self.evaluator(req))
        self.turn_cache.put(key, state, solver)
        return solver, False

    def open_search(self, state, req):
        return _Handle(self, state.clone(), req)

    def annotators(self):
        return []

    def contract(self):
        from .experiment import file_sha256
        return {"game_id": self.game_id, "wire_version": 1,
                "checkpoint": str(self.checkpoint), "checkpoint_exists": self.checkpoint.is_file(),
                "checkpoint_sha256": file_sha256(self.checkpoint) if self.checkpoint.is_file() else None,
                "search": "Exact current-turn solve with learned continuation values",
                "probability": "actor probability encoded as 2*p-1",
                "ranking": "value; visits are zero (not MCTS)",
                "turn_cache": self.turn_cache.stats()}


class _Handle:
    def __init__(self, adapter, state, req):
        self.adapter, self.state, self.req = adapter, state, req
        self.result = None

    def advance(self, chunk_sims, stop_event):
        if self.result is not None:
            return self.result
        if stop_event.is_set():
            return SearchSnapshot(0, 1, 0, {}, partial=True)
        with self.adapter._lock:
            if stop_event.is_set():
                return SearchSnapshot(0, 1, 0, {}, partial=True)
            if wins_on_stop(self.state):
                self.result = SearchSnapshot(1, 1, 1.0,
                    {"stop": ActionStats(0, 1.0, 0, follow_up="Stop and win")},
                    stop_reason="Stopping now wins the game with certainty.")
                return self.result
            solver, cache_hit = self.adapter.solver_for(self.state, self.req)
            entries = {}
            actor = self.state.active_player
            for action in self.adapter.action_views(self.state):
                follow_up = None
                if action.kind == "move":
                    child = self.state.clone()
                    apply_move(child, tuple(action.fields["columns"]))
                    stop_value, roll_value = solver.stop_roll(child)
                    decision = action.fields["decision"]
                    probability = float((stop_value if decision == "stop" else roll_value)[actor])
                    follow_up = "then stop and bank" if decision == "stop" else "then roll again"
                    if decision == "roll" and not can_stop(child):
                        follow_up += " (stopping is blocked)"
                else:
                    stop_value, roll_value = solver.stop_roll(self.state)
                    probability = float((stop_value if action.kind == "stop" else roll_value)[actor])
                entries[action.action_id] = ActionStats(0, 2*probability-1, 0, follow_up=follow_up)
            self.result = SearchSnapshot(1, 1, max(e.q_value for e in entries.values()), entries,
                                        stop_reason=("Cached turn lookup complete; " if cache_hit else "Current-turn solve complete; ")
                                        + "continuation values are model estimates.")
        return self.result

    def close(self):
        self.result = None
