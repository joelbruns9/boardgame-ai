"""
web_app.py — local Kingdomino web lab and future advisor API reference client.

Copy this file to:
    games/kingdomino/web_app.py

Copy the static files to:
    games/kingdomino/web_static/

Run from the project root:
    python -m pip install fastapi uvicorn
    uvicorn games.kingdomino.web_app:app --reload --port 8000

Open:
    http://127.0.0.1:8000

Design goal
-----------
This is intentionally more than a local play UI.  It defines a stable JSON
surface that a future BGA/Firefox advisor extension can reuse:

    browser/local client -> public Kingdomino state JSON -> /api/recommend

The local UI obtains state from the Python engine.  A future BGA extension will
obtain the same public-state JSON by scraping Board Game Arena.  Recommendation
responses should therefore be rendered by both clients with minimal changes.

Current scope
-------------
MVP game lab:
  - create a new game from a seed
  - inspect both boards, current row, claims, scores, phase, actor
  - list legal actions
  - apply legal actions
  - preview legal actions without committing
  - export/import a debug state snapshot
  - expose /api/recommend with a deterministic heuristic placeholder response
  - undo and jump through the current session timeline

The placeholder recommender is intentionally simple and model-free.  Later, keep
its response shape and replace the scoring function with NN/MCTS advisor output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4
import json
import os
import random
import glob
import re
import sys

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - runtime setup hint
    raise RuntimeError(
        "FastAPI UI dependencies are missing. Install with: "
        "python -m pip install fastapi uvicorn"
    ) from exc

from games.kingdomino.board import Placement
from games.kingdomino.dominoes import DOMINOES, Terrain
from games.kingdomino.game import Claim, GameConfig, GameState, Phase, PickAction, TurnAction

try:
    from games.kingdomino.action_codec import encode_action
except Exception:  # pragma: no cover - optional debug field
    encode_action = None


# ─────────────────────────────────────────────────────────────────────────────
# App state
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Kingdomino Web Lab", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    # Local advisor server: accept BGA page origins and extension origins.
    # Firefox/Chrome extension contexts do not always use the same Origin as
    # the page, so strict origin matching makes local testing brittle.
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).with_name("web_static")
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# In-memory sessions are fine for a development lab.  Export/import keeps
# positions reproducible across server restarts.
_SESSIONS: dict[str, GameState] = {}
# Timeline snapshots for undo/jump in the local lab.  This is intentionally
# separate from the public advisor protocol; a future BGA extension should send
# its current observed state rather than rely on server-side session history.
_SESSION_TIMELINES: dict[str, list[GameState]] = {}

_NN_EVALUATOR_CACHE: dict[tuple[str, str, int, int, int], Any] = {}
_EXACT_ADVISOR_VALUE_CACHE: dict[tuple[str, float, float, float], float] = {}
_EXACT_ADVISOR_CACHE_MAX = 200_000


# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────
class NewGameRequest(BaseModel):
    seed: Optional[int] = None
    start_player: Optional[int] = None
    board_size: int = 7
    canvas_size: int = 15
    harmony: bool = True
    middle_kingdom: bool = True
    mighty_duel: bool = True


class SessionRequest(BaseModel):
    session_id: str


class ApplyActionRequest(BaseModel):
    session_id: str
    legal_index: Optional[int] = Field(default=None, description="Index into current legal action list")
    action_id: Optional[str] = Field(default=None, description="Stable action id from /api/legal-actions")


class UndoActionRequest(BaseModel):
    session_id: str
    steps: int = Field(default=1, ge=1, le=200)


class JumpToStepRequest(BaseModel):
    session_id: str
    step: int = Field(description="Timeline step to restore. 0 is the initial position.")


class ImportStateRequest(BaseModel):
    session_id: Optional[str] = None
    state: dict[str, Any]


class AdvisorProbeSaveRequest(BaseModel):
    filename: Optional[str] = None
    probe: dict[str, Any]


class RecommendRequest(BaseModel):
    session_id: Optional[str] = None
    state: Optional[dict[str, Any]] = None
    engine: str = Field(default="greedy", description="greedy/heuristic, random, nn, exact, or auto")
    requested_engine: Optional[str] = None
    num_simulations: int = Field(default=50, ge=0, le=20000)
    top_k: int = Field(default=8, ge=1, le=100)
    checkpoint_path: Optional[str] = None
    nn_sims: int = Field(default=50, ge=1, le=20000)
    # Unused by the open-loop NN path (OpenLoopMCTS averages over deck orders
    # internally, one search per request).  Kept to preserve the API surface.
    determinizations: int = Field(default=1, ge=1, le=16)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    device: str = "cuda"
    # Architecture is normally read from the checkpoint's stored config.  These
    # are optional overrides, only used as a fallback for old checkpoints that
    # predate the saved config dict.  Leave them None to trust the checkpoint.
    channels: Optional[int] = None
    blocks: Optional[int] = None
    bilinear_dim: Optional[int] = None
    seed: int = 0
    # Exact endgame advisor. exact_threads=0 leaves Rayon on its default global
    # pool size, which is all logical CPUs unless RAYON_NUM_THREADS was set
    # before the Rust extension initialized.
    # Advisor policy: always solve — default to the ceiling; requests may lower it.
    exact_max_secs: float = Field(default=3600.0, ge=0.0, le=3600.0)
    exact_threads: int = Field(default=0, ge=0, le=128)
    # Swindle analysis (exact engine, losing/drawn roots): enumerate opponent
    # replies to the top candidate moves and exact-solve each, identifying
    # moves that maximize the chance an imperfect opponent errs. None = auto
    # (run whenever the root is not winning); True/False force on/off.
    swindle: Optional[bool] = None
    swindle_budget_secs: float = Field(default=60.0, ge=0.0, le=600.0)


class BotActionRequest(BaseModel):
    session_id: str
    mode: str = Field(default="greedy", description="random, greedy, or nn")
    apply: bool = True
    # NN/MCTS options.  If checkpoint_path is omitted, the newest common
    # Kingdomino iter_*.pt checkpoint is used when available.
    checkpoint_path: Optional[str] = None
    nn_sims: int = Field(default=50, ge=1, le=20000)
    # Unused by the open-loop NN path (OpenLoopMCTS averages over deck orders
    # internally, one search per request).  Kept to preserve the API surface.
    determinizations: int = Field(default=1, ge=1, le=16)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    device: str = "cuda"
    # Optional architecture overrides; normally read from the checkpoint config.
    channels: Optional[int] = None
    blocks: Optional[int] = None
    bilinear_dim: Optional[int] = None
    seed: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Serialization helpers
# ─────────────────────────────────────────────────────────────────────────────
def _terrain_name(value: int) -> str:
    try:
        return Terrain(int(value)).name
    except Exception:
        return str(value)


def _score_to_json(score) -> dict[str, int]:
    return {
        "territory_score": int(score.territory_score),
        "harmony_bonus": int(score.harmony_bonus),
        "middle_kingdom_bonus": int(score.middle_kingdom_bonus),
        "total": int(score.total),
    }


def _claim_to_json(claim: Claim) -> dict[str, int]:
    return {"player": int(claim.player), "domino_id": int(claim.domino_id)}


def _domino_to_json(domino_id: int) -> dict[str, Any]:
    d = DOMINOES[int(domino_id)]
    return {
        "id": int(d.id),
        "a": {"terrain": d.a.terrain.name, "terrain_id": int(d.a.terrain), "crowns": int(d.a.crowns)},
        "b": {"terrain": d.b.terrain.name, "terrain_id": int(d.b.terrain), "crowns": int(d.b.crowns)},
        "total_crowns": int(d.a.crowns + d.b.crowns),
    }


def _placement_to_json(p: Optional[Placement]) -> Optional[dict[str, Any]]:
    if p is None:
        return None
    return {
        "x1": int(p.x1), "y1": int(p.y1),
        "x2": int(p.x2), "y2": int(p.y2),
        "flipped": bool(p.flipped),
        "cells": [[int(p.x1), int(p.y1)], [int(p.x2), int(p.y2)]],
    }


def _board_to_json(board, *, harmony: bool, middle_kingdom: bool, include_grids: bool = True) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for x, y in board.occupied_cells():
        terrain_id = int(board.terrain[y, x])
        cells.append({
            "x": int(x),
            "y": int(y),
            "terrain": _terrain_name(terrain_id),
            "terrain_id": terrain_id,
            "crowns": int(board.crowns[y, x]),
            "domino_id": int(board.domino_id[y, x]),
        })

    out: dict[str, Any] = {
        "canvas_size": int(board.canvas_size),
        "castle_pos": [int(board.castle_pos[0]), int(board.castle_pos[1])],
        "bbox": list(map(int, board.occupied_bbox() or (0, 0, 0, 0))),
        "occupied_count": int(len(board.occupied_cells())),
        "cells": cells,
        "score": _score_to_json(board.score(harmony, middle_kingdom)),
    }
    if include_grids:
        out["terrain_grid"] = [[int(v) for v in row] for row in board.terrain.tolist()]
        out["crowns_grid"] = [[int(v) for v in row] for row in board.crowns.tolist()]
        out["domino_grid"] = [[int(v) for v in row] for row in board.domino_id.tolist()]
    return out


def _history_item_to_json(action: object) -> dict[str, Any]:
    if isinstance(action, PickAction):
        return {"kind": "pick", "domino_id": int(action.domino_id)}
    if isinstance(action, TurnAction):
        return {
            "kind": "turn",
            "placement": _placement_to_json(action.placement),
            "pick_domino_id": None if action.pick_domino_id is None else int(action.pick_domino_id),
        }
    return {"kind": type(action).__name__, "repr": repr(action)}


def action_to_json(state: GameState, action: PickAction | TurnAction, index: int) -> dict[str, Any]:
    action_idx: Optional[int] = None
    if encode_action is not None:
        try:
            action_idx = int(encode_action(action, state))
        except Exception:
            action_idx = None

    if isinstance(action, PickAction):
        action_id = f"pick:{action.domino_id}"
        label = f"Pick domino {action.domino_id}"
        return {
            "legal_index": int(index),
            "action_id": action_id,
            "action_idx": action_idx,
            "kind": "pick",
            "label": label,
            "domino_id": int(action.domino_id),
            "domino": _domino_to_json(action.domino_id),
            "placement": None,
            "pick_domino_id": int(action.domino_id),
        }

    claim = state.pending_claims[state.actor_index]
    current_domino_id = int(claim.domino_id)
    if action.placement is None:
        placement_part = "discard"
        label = f"Discard domino {current_domino_id}"
    else:
        p = action.placement
        placement_part = f"p:{p.x1},{p.y1},{p.x2},{p.y2},{int(p.flipped)}"
        label = (
            f"Place domino {current_domino_id} at "
            f"({p.x1},{p.y1}) → ({p.x2},{p.y2})"
            f"{' flipped' if p.flipped else ''}"
        )
    if action.pick_domino_id is not None:
        label += f"; pick {action.pick_domino_id}"
        pick_part = f"pick:{action.pick_domino_id}"
    else:
        pick_part = "nopick"

    return {
        "legal_index": int(index),
        "action_id": f"turn:{current_domino_id}:{placement_part}:{pick_part}",
        "action_idx": action_idx,
        "kind": "turn",
        "label": label,
        "domino_id": current_domino_id,
        "domino": _domino_to_json(current_domino_id),
        "placement": _placement_to_json(action.placement),
        "pick_domino_id": None if action.pick_domino_id is None else int(action.pick_domino_id),
    }


def legal_actions_json(state: GameState) -> list[dict[str, Any]]:
    return [action_to_json(state, a, i) for i, a in enumerate(state.legal_actions())]


def _current_task_json(state: GameState) -> dict[str, Any]:
    if state.phase == Phase.GAME_OVER:
        scores = state.scores()
        if scores[0] > scores[1]:
            winner = 0
        elif scores[1] > scores[0]:
            winner = 1
        else:
            winner = None
        return {
            "kind": "game_over",
            "title": "Game over",
            "detail": "Tie game" if winner is None else f"Player {winner} wins",
            "current_domino_id": None,
            "requires_pick": False,
            "requires_placement": False,
        }
    actor = int(state.current_actor)
    if state.phase == Phase.INITIAL_SELECTION:
        return {
            "kind": "initial_pick",
            "title": f"Player {actor}: choose a domino",
            "detail": "Click a domino in the current row. Opening pick order is P1, P2, P2, P1.",
            "current_domino_id": None,
            "requires_pick": True,
            "requires_placement": False,
        }
    claim = state.pending_claims[state.actor_index]
    if state.phase == Phase.PLACE_AND_SELECT:
        return {
            "kind": "place_and_pick",
            "title": f"Player {actor}: place domino {claim.domino_id}, then pick a future domino",
            "detail": "Select a future domino from the row, then click two adjacent cells on your board to place the current domino.",
            "current_domino_id": int(claim.domino_id),
            "current_domino": _domino_to_json(claim.domino_id),
            "requires_pick": True,
            "requires_placement": True,
        }
    return {
        "kind": "final_place",
        "title": f"Player {actor}: final placement for domino {claim.domino_id}",
        "detail": "Click two adjacent cells on your board. No future domino is selected in the final round.",
        "current_domino_id": int(claim.domino_id),
        "current_domino": _domino_to_json(claim.domino_id),
        "requires_pick": False,
        "requires_placement": True,
    }


def state_to_public_json(state: GameState, *, include_debug: bool = False) -> dict[str, Any]:
    cfg = state.config
    phase = state.phase.name
    game_over = state.phase == Phase.GAME_OVER
    current_actor = None if game_over else int(state.current_actor)
    current_claim = None
    if state.phase in (Phase.PLACE_AND_SELECT, Phase.FINAL_PLACEMENT):
        current_claim = _claim_to_json(state.pending_claims[state.actor_index])

    out: dict[str, Any] = {
        "game": "kingdomino",
        "rules": {
            "players": int(cfg.players),
            "board_size": int(cfg.board_size),
            "canvas_size": int(cfg.canvas_size),
            "harmony": bool(cfg.harmony),
            "middle_kingdom": bool(cfg.middle_kingdom),
            "mighty_duel": bool(cfg.mighty_duel),
        },
        "phase": phase,
        "phase_id": int(state.phase),
        "game_over": bool(game_over),
        "current_actor": current_actor,
        "current_claim": current_claim,
        "current_claim_tile": None if current_claim is None else _domino_to_json(current_claim["domino_id"]),
        "current_task": _current_task_json(state),
        "legal_action_count": 0 if game_over else len(state.legal_actions()),
        "actor_index": int(state.actor_index),
        "initial_pick_count": int(state.initial_pick_count),
        "start_player": int(state.start_player),
        "current_row": [int(d) for d in state.current_row],
        "current_row_tiles": [_domino_to_json(d) for d in state.current_row],
        "pending_claims": [_claim_to_json(c) for c in state.pending_claims],
        "next_claims": [_claim_to_json(c) for c in state.next_claims],
        "deck_count": int(len(state.deck)),
        "boards": [
            _board_to_json(b, harmony=cfg.harmony, middle_kingdom=cfg.middle_kingdom)
            for b in state.boards
        ],
        "scores": [int(s) for s in state.scores()],
        "score_breakdowns": [
            _score_to_json(b.score(cfg.harmony, cfg.middle_kingdom)) for b in state.boards
        ],
        "history_len": int(len(state.history)),
        "visible_history": [_history_item_to_json(a) for a in state.history],
    }
    if include_debug:
        out["debug"] = {
            "deck": [int(d) for d in state.deck],
            "history": [_history_item_to_json(a) for a in state.history],
        }
    return out


def state_to_debug_json(state: GameState) -> dict[str, Any]:
    # Full internal-state export for local reproduction.  The BGA advisor should
    # not depend on this because it includes hidden deck order.
    return state_to_public_json(state, include_debug=True)


def state_from_debug_json(data: dict[str, Any]) -> GameState:
    rules = data.get("rules", {})
    cfg = GameConfig(
        players=int(rules.get("players", 2)),
        board_size=int(rules.get("board_size", 7)),
        canvas_size=int(rules.get("canvas_size", 15)),
        harmony=bool(rules.get("harmony", True)),
        middle_kingdom=bool(rules.get("middle_kingdom", True)),
        mighty_duel=bool(rules.get("mighty_duel", True)),
    )
    boards = []
    from games.kingdomino.board import Board
    for bdata in data.get("boards", []):
        castle = tuple(bdata.get("castle_pos", [cfg.canvas_size // 2, cfg.canvas_size // 2]))
        b = Board(cfg.canvas_size, castle_pos=castle)  # includes castle
        # Clear and rebuild so import is exact, including shifted castle if any.
        b.terrain[:, :] = 0
        b.crowns[:, :] = 0
        b.domino_id[:, :] = 0
        b._occupied.clear()
        b._cell.clear()
        b._min_x = b._min_y = cfg.canvas_size
        b._max_x = b._max_y = -1
        for cell in bdata.get("cells", []):
            x, y = int(cell["x"]), int(cell["y"])
            terrain_id = int(cell["terrain_id"])
            b.terrain[y, x] = terrain_id
            b.crowns[y, x] = int(cell.get("crowns", 0))
            b.domino_id[y, x] = int(cell.get("domino_id", 0))
            b._occupied.add((x, y))
            b._cell[(x, y)] = terrain_id
            b._min_x = min(b._min_x, x); b._max_x = max(b._max_x, x)
            b._min_y = min(b._min_y, y); b._max_y = max(b._max_y, y)
        if not b._occupied:
            cx, cy = b.castle_pos
            b.terrain[cy, cx] = Terrain.CASTLE
            b.domino_id[cy, cx] = -1
            b._occupied.add((cx, cy))
            b._cell[(cx, cy)] = int(Terrain.CASTLE)
            b._min_x = b._max_x = cx; b._min_y = b._max_y = cy
        boards.append(b)
    while len(boards) < cfg.players:
        boards.append(Board(cfg.canvas_size))

    debug = data.get("debug", {})
    deck = [int(d) for d in debug.get("deck", [])]
    current_row = [int(d) for d in data.get("current_row", [])]
    pending_claims = [Claim(int(c["player"]), int(c["domino_id"])) for c in data.get("pending_claims", [])]
    next_claims = [Claim(int(c["player"]), int(c["domino_id"])) for c in data.get("next_claims", [])]
    phase = Phase[data.get("phase", "INITIAL_SELECTION")]

    # History import is only for display/debug.  Reconstructing exact action
    # instances is not required for continuing from the current state.
    history = list(debug.get("history", data.get("visible_history", [])))

    return GameState(
        config=cfg,
        boards=boards,
        deck=deck,
        current_row=current_row,
        pending_claims=pending_claims,
        next_claims=next_claims,
        phase=phase,
        actor_index=int(data.get("actor_index", 0)),
        initial_pick_count=int(data.get("initial_pick_count", 0)),
        start_player=int(data.get("start_player", 0)),
        history=history,
    )


def _get_state(session_id: str) -> GameState:
    try:
        return _SESSIONS[session_id]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown session_id {session_id!r}")


def _ensure_timeline(session_id: str) -> list[GameState]:
    state = _get_state(session_id)
    timeline = _SESSION_TIMELINES.setdefault(session_id, [state])
    if not timeline:
        timeline.append(state)
    if timeline[-1] is not state:
        # This can happen after an older import/export path or manual server
        # mutation.  Keep the visible current state as the newest timeline step.
        timeline.append(state)
    return timeline


def _session_meta(session_id: str) -> dict[str, Any]:
    timeline = _ensure_timeline(session_id)
    step = len(timeline) - 1
    return {
        "session_id": session_id,
        "timeline_step": step,
        "timeline_length": len(timeline),
        "can_undo": step > 0,
        "can_redo": False,
    }


def _response_for_session(session_id: str, *, include_debug: bool = False) -> dict[str, Any]:
    state = _get_state(session_id)
    return {
        "ok": True,
        "session_id": session_id,
        "session": _session_meta(session_id),
        "state": state_to_public_json(state, include_debug=include_debug),
        "legal_actions": legal_actions_json(state),
    }


def _select_legal_action(state: GameState, legal_index: Optional[int], action_id: Optional[str]):
    actions = state.legal_actions()
    if legal_index is not None:
        if not 0 <= legal_index < len(actions):
            raise HTTPException(status_code=400, detail=f"legal_index {legal_index} out of range")
        return actions[legal_index]
    if action_id is not None:
        for i, action in enumerate(actions):
            if action_to_json(state, action, i)["action_id"] == action_id:
                return action
        raise HTTPException(status_code=400, detail=f"action_id {action_id!r} is not legal in current state")
    raise HTTPException(status_code=400, detail="Provide legal_index or action_id")


def _choose_random_action(state: GameState, seed: int = 0):
    actions = state.legal_actions()
    if not actions:
        raise HTTPException(status_code=400, detail="No legal actions available.")
    rng = random.Random(int(seed) + 1009 * len(state.history))
    return rng.choice(actions), {"engine": "random", "score": None}


def _choose_greedy_action(state: GameState):
    actions = state.legal_actions()
    if not actions:
        raise HTTPException(status_code=400, detail="No legal actions available.")
    scored = []
    for action in actions:
        score, parts = _heuristic_action_score(state, action)
        scored.append((score, action, parts))
    scored.sort(key=lambda x: x[0], reverse=True)
    score, action, parts = scored[0]
    return action, {"engine": "greedy-placeholder", "score": float(score), "debug": parts}


def _exact_thread_meta(requested_threads: int) -> dict[str, Any]:
    requested = int(requested_threads or 0)
    rust_already_loaded = "kingdomino_rust" in sys.modules
    if requested > 0 and not rust_already_loaded:
        os.environ["RAYON_NUM_THREADS"] = str(requested)
    env_threads = os.environ.get("RAYON_NUM_THREADS")
    return {
        "threads_requested": requested,
        "threads_effective": requested if requested > 0 else int(os.cpu_count() or 1),
        "rayon_num_threads_env": env_threads,
        "thread_pool_already_initialized": bool(rust_already_loaded),
        "thread_note": (
            "exact_threads only changes Rayon before the Rust extension initializes"
            if requested > 0 and rust_already_loaded
            else None
        ),
    }


def _exact_state_key(state: GameState) -> str:
    data = state_to_debug_json(state)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _exact_supported_detail(state: GameState, request_state: Optional[dict[str, Any]]) -> Optional[str]:
    if request_state is not None:
        deck_count = request_state.get("deck_count")
        debug_deck = request_state.get("debug", {}).get("deck") if isinstance(request_state.get("debug"), dict) else None
        if deck_count == 4 and not debug_deck:
            return (
                "Exact deck=4 solving requires the hidden deck identities in state.debug.deck. "
                "The BGA capture reported deck_count=4 but did not include debug.deck."
            )

    if state.phase == Phase.GAME_OVER:
        return None
    if state.phase == Phase.PLACE_AND_SELECT and len(state.deck) in (0, 4):
        return None
    if state.phase == Phase.FINAL_PLACEMENT and len(state.deck) == 0:
        return None
    return (
        "Exact advisor is only available for PLACE_AND_SELECT with deck length 4 or 0, "
        "FINAL_PLACEMENT with deck length 0, or GAME_OVER. "
        f"Current phase={state.phase.name}, deck length={len(state.deck)}."
    )


def _cached_exact_value(
    state: GameState,
    *,
    max_secs: float,
    score_scale: float,
    margin_gain: float,
    alpha: float,
    seed: int,
) -> tuple[float, bool, bool]:
    key = (_exact_state_key(state), float(score_scale), float(margin_gain), float(alpha))
    cached = _EXACT_ADVISOR_VALUE_CACHE.get(key)
    if cached is not None:
        return float(cached), True, True

    from games.kingdomino.endgame_solver import exact_endgame_value

    value0, solved = exact_endgame_value(
        state,
        max_secs=float(max_secs),
        rng=random.Random(int(seed)),
        score_scale=float(score_scale),
        margin_gain=float(margin_gain),
        alpha=float(alpha),
    )
    if not solved:
        return float(value0), False, False
    if len(_EXACT_ADVISOR_VALUE_CACHE) >= _EXACT_ADVISOR_CACHE_MAX:
        _EXACT_ADVISOR_VALUE_CACHE.clear()
    _EXACT_ADVISOR_VALUE_CACHE[key] = float(value0)
    return float(value0), False, True


def _exact_cache_stats_for(
    state: GameState,
    *,
    max_secs: float,
    score_scale: float,
    margin_gain: float,
    alpha: float,
    seed: int,
) -> tuple[float, int, int]:
    value, hit, solved = _cached_exact_value(
        state,
        max_secs=max_secs,
        score_scale=score_scale,
        margin_gain=margin_gain,
        alpha=alpha,
        seed=seed,
    )
    if not solved:
        raise HTTPException(status_code=504, detail="Exact solver did not finish within exact_max_secs.")
    return value, int(hit), int(not hit)


def _opponent_policy_priors(req: RecommendRequest, child: GameState) -> Optional[dict]:
    """NN policy priors over the OPPONENT's legal replies in `child` (their
    turn). Used to weight swindle traps by how likely a strong-but-imperfect
    opponent is to walk into them. Returns None when no checkpoint resolves —
    swindle then falls back to uniform weighting."""
    try:
        import torch
        from games.kingdomino.encoder import encode_state

        _evaluator, net, _path = _load_nn_evaluator(req)
        acts = child.legal_actions()
        mb, ob, flat = encode_state(child, child.current_actor)
        device = next(net.parameters()).device
        with torch.inference_mode():
            _own, _opp, _win, logits = net(
                torch.from_numpy(mb).unsqueeze(0).to(device),
                torch.from_numpy(ob).unsqueeze(0).to(device),
                torch.from_numpy(flat).unsqueeze(0).to(device),
            )
        idxs = torch.tensor([int(encode_action(a, child)) for a in acts],
                            dtype=torch.long, device=logits.device)
        legal_logits = logits[0, idxs]
        priors = torch.softmax(legal_logits, dim=0).cpu().numpy()
        return {i: float(p) for i, p in enumerate(priors)}
    except Exception:
        return None


def _swindle_for_move(
    child: GameState,
    child_value_actor: float,
    actor: int,
    req: RecommendRequest,
    *,
    score_scale: float,
    margin_gain: float,
    margin_probe_gain: float,
    deadline: float,
) -> Optional[dict[str, Any]]:
    """One-ply trap analysis of `child` (opponent to move) from the ACTOR's
    perspective. Exact-solves every opponent reply and reports how many of
    them improve the actor's game-theoretic outcome — i.e. are mistakes.

    Correctness invariant (deck <= 4 is chance-free: the final-row reveal is
    deterministic): the opponent's BEST reply must reproduce the child's own
    minimax value. A violation means solver/state inconsistency; the move's
    swindle stats are dropped and a warning logged rather than shown wrong.

    Returns None when the deadline is hit before finishing (partial results
    are never shown) or when the child is terminal."""
    import time as _time

    # Kingdomino does NOT strictly alternate: pick order can give the same
    # player consecutive moves. Descend through the ACTOR's own follow-up
    # moves along the exact-optimal line (value-preserving, so the invariant
    # below still checks against child_value_actor) until it is genuinely the
    # opponent's decision — that is where traps live.
    node = child
    guard = 0
    while node.phase != Phase.GAME_OVER and int(node.current_actor) == int(actor):
        best_v = None
        best_g = None
        for a in node.legal_actions():
            if _time.perf_counter() > deadline:
                return None
            g = node.step(a)
            v0, _hit, solved = _cached_exact_value(
                g, max_secs=float(req.exact_max_secs), score_scale=score_scale,
                margin_gain=margin_gain, alpha=0.0, seed=int(req.seed))
            if not solved:
                return None
            v_act = float(v0 if actor == 0 else -v0)
            if best_v is None or v_act > best_v:
                best_v, best_g = v_act, g
        if best_g is None:
            return None
        node = best_g
        guard += 1
        if guard > 8:
            return None  # defensive: no legal Kingdomino line has this many consecutive moves

    if node.phase == Phase.GAME_OVER:
        return {"replies": 0, "flips_win": 0, "flips_draw": 0,
                "uniform_rate": 0.0, "weighted_rate": None,
                "expected_points_uniform": None, "expected_points_weighted": None,
                "trap_payoff_pts": None, "note": "terminal after this move"}

    child = node
    replies = child.legal_actions()
    priors = _opponent_policy_priors(req, child)
    flips_win = flips_draw = 0
    weighted_improving = 0.0
    pts_uniform_sum = 0.0
    pts_weighted_sum = 0.0
    best_payoff: Optional[float] = None
    min_reply_value = None
    for ri, r in enumerate(replies):
        if _time.perf_counter() > deadline:
            return None
        g = child.step(r)
        v0, _hit, solved = _cached_exact_value(
            g, max_secs=float(req.exact_max_secs), score_scale=score_scale,
            margin_gain=margin_gain, alpha=0.0, seed=int(req.seed))
        if not solved:
            return None
        v_act = float(v0 if actor == 0 else -v0)
        min_reply_value = v_act if min_reply_value is None else min(min_reply_value, v_act)
        pts = 1.0 if v_act > 1e-9 else (0.5 if v_act > -1e-9 else 0.0)
        w = priors.get(ri, 0.0) if priors is not None else 1.0 / len(replies)
        pts_uniform_sum += pts / len(replies)
        pts_weighted_sum += pts * w
        if v_act > child_value_actor + 1e-9:
            if v_act > 1e-9:
                flips_win += 1
            else:
                flips_draw += 1
            weighted_improving += w
            m0, _mh, m_solved = _cached_exact_value(
                g, max_secs=float(req.exact_max_secs), score_scale=score_scale,
                margin_gain=margin_probe_gain, alpha=1.0, seed=int(req.seed))
            if m_solved:
                payoff = float(m0 if actor == 0 else -m0) * score_scale / margin_probe_gain
                best_payoff = payoff if best_payoff is None else max(best_payoff, payoff)

    # Invariant: opponent's best reply == child's minimax value.
    if min_reply_value is None or abs(min_reply_value - child_value_actor) > 1e-6:
        print(f"WARNING: swindle invariant violated: min reply value "
              f"{min_reply_value} != child value {child_value_actor}; "
              f"dropping swindle stats for this move")
        return None

    improving = flips_win + flips_draw
    return {
        "replies": len(replies),
        "flips_win": flips_win,
        "flips_draw": flips_draw,
        "uniform_rate": improving / len(replies),
        "weighted_rate": (float(weighted_improving) if priors is not None else None),
        "expected_points_uniform": pts_uniform_sum,
        "expected_points_weighted": (pts_weighted_sum if priors is not None else None),
        "trap_payoff_pts": best_payoff,
    }


def _recommend_exact(
    state: GameState,
    req: RecommendRequest,
    *,
    request_state: Optional[dict[str, Any]],
    started_at: float,
) -> dict[str, Any]:
    import time

    detail = _exact_supported_detail(state, request_state)
    if detail is not None:
        raise HTTPException(status_code=400, detail=detail)

    thread_meta = _exact_thread_meta(req.exact_threads)
    score_scale = 160.0
    margin_gain = 2.0
    win_alpha = 0.0
    # Ranking frame matches training (win-gated B=0.5; was the pre-win-gate
    # 0.8). Rankings are identical for any alpha>0 — the value is a monotone
    # transform of the raw margin — so this is consistency, not behavior.
    rank_alpha = 0.5
    # Margin probe: with alpha=1 the exact leaf is win_gate*tanh(g*m/scale); a
    # tiny gain keeps tanh in its linear region (|x| <= ~0.008 for |m| <= 120),
    # so the chance-node EXPECTATION divides back out to expected margin in
    # actual points: E[m] ~= value * scale / g. The win gate only zeroes exact
    # draws, where m = 0 anyway.
    margin_probe_gain = 0.01

    def _exact_margin_pts(child_state) -> tuple[float, int, int]:
        value0, hit, miss = _exact_cache_stats_for(
            child_state,
            max_secs=float(req.exact_max_secs),
            score_scale=score_scale,
            margin_gain=margin_probe_gain,
            alpha=1.0,
            seed=int(req.seed),
        )
        return float(value0) * score_scale / margin_probe_gain, hit, miss
    root_value0, root_hit, root_solved = _cached_exact_value(
        state,
        max_secs=float(req.exact_max_secs),
        score_scale=score_scale,
        margin_gain=margin_gain,
        alpha=win_alpha,
        seed=int(req.seed),
    )
    if not root_solved:
        raise HTTPException(status_code=504, detail="Exact solver did not finish the root position within exact_max_secs.")
    if state.phase == Phase.GAME_OVER:
        return {
            "ok": True,
            "engine": "exact",
            "value": None,
            "root_value_player0": float(root_value0),
            "search_ms": int(round((time.perf_counter() - started_at) * 1000)),
            "num_simulations": 0,
            "exact": {
                "solved": True,
                "label": "exact",
                "deck_count": int(len(state.deck)),
                "cache_hits": int(root_hit),
                "cache_misses": int(not root_hit),
                "cache_size": int(len(_EXACT_ADVISOR_VALUE_CACHE)),
                "max_secs": float(req.exact_max_secs),
                **thread_meta,
            },
            "recommendations": [],
        }

    actor = int(state.current_actor)
    rows = []
    cache_hits = int(root_hit)
    cache_misses = int(not root_hit)
    root_rank_value0, rank_hit, rank_miss = _exact_cache_stats_for(
        state,
        max_secs=float(req.exact_max_secs),
        score_scale=score_scale,
        margin_gain=margin_gain,
        alpha=rank_alpha,
        seed=int(req.seed),
    )
    cache_hits += rank_hit
    cache_misses += rank_miss
    actions = state.legal_actions()
    id_to_index = {
        action_to_json(state, action, i)["action_id"]: i
        for i, action in enumerate(actions)
    }
    for action in actions:
        child = state.step(action)
        value0, hit, solved = _cached_exact_value(
            child,
            max_secs=float(req.exact_max_secs),
            score_scale=score_scale,
            margin_gain=margin_gain,
            alpha=win_alpha,
            seed=int(req.seed),
        )
        if not solved:
            raise HTTPException(
                status_code=504,
                detail=f"Exact solver did not finish legal action {action_to_json(state, action, -1).get('action_id')} within exact_max_secs.",
            )
        cache_hits += int(hit)
        cache_misses += int(not hit)
        rank_value0, rank_hit, rank_miss = _exact_cache_stats_for(
            child,
            max_secs=float(req.exact_max_secs),
            score_scale=score_scale,
            margin_gain=margin_gain,
            alpha=rank_alpha,
            seed=int(req.seed),
        )
        cache_hits += rank_hit
        cache_misses += rank_miss
        margin0_pts, margin_hit, margin_miss = _exact_margin_pts(child)
        cache_hits += margin_hit
        cache_misses += margin_miss
        value_actor = float(value0 if actor == 0 else -value0)
        rank_value_actor = float(rank_value0 if actor == 0 else -rank_value0)
        margin_actor_pts = float(margin0_pts if actor == 0 else -margin0_pts)
        q_win_prob = max(0.0, min(1.0, (value_actor + 1.0) / 2.0))
        aj = action_to_json(state, action, -1)
        idx = id_to_index.get(aj["action_id"], -1)
        if idx >= 0:
            aj = action_to_json(state, actions[idx], idx)
        rows.append((value_actor, rank_value_actor, q_win_prob, idx, aj, value0,
                     rank_value0, hit, margin_actor_pts, child))

    rows.sort(key=lambda item: (item[0], item[1]), reverse=True)

    # ── Swindle analysis (losing/drawn roots): identify moves that maximize
    # the chance an imperfect opponent errs. One-ply: exact-solve every
    # opponent reply to the top candidates, in rank order, under a time
    # budget. Among equally-valued moves the ranking then prefers high trap
    # rates over minimal losing margins — a simplifying "least bad" move that
    # leaves the opponent nothing to get wrong is worth less against a human
    # than a slightly worse move with a trap their natural reply walks into.
    root_value_actor = float(root_value0 if actor == 0 else -root_value0)
    swindle_mode = (req.swindle is True
                    or (req.swindle is None and root_value_actor <= 1e-9))
    swindle_truncated = False
    swindle_results: list[Optional[dict[str, Any]]] = [None] * len(rows)
    if swindle_mode and rows:
        deadline = time.perf_counter() + float(req.swindle_budget_secs)
        analyze_n = min(len(rows), max(int(req.top_k), 8))
        for pos in range(analyze_n):
            if time.perf_counter() > deadline:
                swindle_truncated = True
                break
            row = rows[pos]
            res = _swindle_for_move(
                row[9], row[0], actor, req,
                score_scale=score_scale, margin_gain=margin_gain,
                margin_probe_gain=margin_probe_gain, deadline=deadline)
            if res is None:
                if time.perf_counter() > deadline:
                    swindle_truncated = True
                    break
                continue  # invariant violation — stats dropped for this move
            swindle_results[pos] = res
        # Re-rank the ANALYZED head among itself: (game value, trap score,
        # margin). Unanalyzed tail keeps its original order below.
        analyzed = [(rows[i], swindle_results[i]) for i in range(analyze_n)]
        tail = [(rows[i], None) for i in range(analyze_n, len(rows))]

        def _trap_score(s: Optional[dict[str, Any]]) -> float:
            if not s:
                return -1.0
            if s.get("weighted_rate") is not None:
                return float(s["weighted_rate"])
            return float(s.get("uniform_rate") or 0.0)

        analyzed.sort(key=lambda rs: (rs[0][0], _trap_score(rs[1]), rs[0][1]),
                      reverse=True)
        paired = analyzed + tail
    else:
        paired = [(row, None) for row in rows]

    recs = []
    for rank, ((value_actor, rank_value_actor, q_win_prob, idx, aj, value0,
                rank_value0, hit, margin_actor_pts, _child), swindle) in enumerate(
            paired[: max(1, int(req.top_k))], start=1):
        rec = {
            "rank": rank,
            **aj,
            "prior": None,
            "visit_frac": None,
            "q_win_prob": float(q_win_prob),
            "q_value": float(value_actor),
            "q_rank_value": float(rank_value_actor),
            "exact_margin_pts": float(margin_actor_pts),
            "is_legal": idx >= 0,
            "debug": {
                "engine": "exact",
                "label": "exact",
                "legal_index_resolved": idx,
                "value_player0": float(value0),
                "rank_value_player0": float(rank_value0),
                "cache_hit": bool(hit),
            },
        }
        if swindle is not None:
            rec["swindle"] = swindle
        recs.append(rec)

    root_margin0_pts, margin_hit, margin_miss = _exact_margin_pts(state)
    cache_hits += margin_hit
    cache_misses += margin_miss
    value_actor = root_value_actor
    rank_value_actor = float(root_rank_value0 if actor == 0 else -root_rank_value0)
    return {
        "ok": True,
        "engine": "exact",
        "value": value_actor,
        "root_win_prob": max(0.0, min(1.0, (value_actor + 1.0) / 2.0)),
        "root_rank_value": rank_value_actor,
        "root_margin_pts": float(root_margin0_pts if actor == 0 else -root_margin0_pts),
        "swindle_mode": bool(swindle_mode),
        "swindle_truncated": bool(swindle_truncated),
        "root_value_player0": float(root_value0),
        "search_ms": int(round((time.perf_counter() - started_at) * 1000)),
        "num_simulations": 0,
        "exact": {
            "solved": True,
            "label": "exact",
            "deck_count": int(len(state.deck)),
            "cache_hits": int(cache_hits),
            "cache_misses": int(cache_misses),
            "cache_size": int(len(_EXACT_ADVISOR_VALUE_CACHE)),
            "max_secs": float(req.exact_max_secs),
            "score_scale": score_scale,
            "margin_gain": margin_gain,
            "alpha": win_alpha,
            "rank_alpha": rank_alpha,
            **thread_meta,
        },
        "recommendations": recs,
    }


def _newest_checkpoint_path() -> Optional[str]:
    # 1) Canonical best model for the BGA advisor.  Copy the promoted checkpoint
    #    here as current_best.pt; the server loads it automatically when no
    #    checkpoint_path is supplied.
    canonical = Path("runs/kingdomino/best_checkpoint/current_best.pt")
    if canonical.exists():
        return str(canonical)

    # 2) Otherwise fall back to the highest-iteration checkpoint under any run.
    #    Rank by iteration number parsed from the filename (iter_NNNN.pt) so we
    #    do not have to torch.load every candidate; break ties by mtime.
    best: Optional[tuple[int, float, str]] = None
    for path in glob.glob("runs/kingdomino/**/iter_*.pt", recursive=True):
        m = re.search(r"iter_(\d+)\.pt$", path)
        iteration = int(m.group(1)) if m else -1
        try:
            mtime = Path(path).stat().st_mtime
        except Exception:
            mtime = 0.0
        item = (iteration, mtime, path)
        if best is None or item > best:
            best = item
    return None if best is None else best[2]


def _load_nn_evaluator(req: BotActionRequest):
    checkpoint_path = req.checkpoint_path or _newest_checkpoint_path()
    if not checkpoint_path:
        raise HTTPException(
            status_code=400,
            detail="NN bot needs a checkpoint_path, runs/kingdomino/best_checkpoint/current_best.pt, or an iter_*.pt checkpoint under runs/kingdomino/.",
        )
    path = str(Path(checkpoint_path))
    # Cache key includes the request-supplied overrides so a forced re-arch of
    # the same path/device does not return a stale net.  The resolved arch is
    # otherwise a pure function of the file, so (path, device) keys the common
    # case where overrides are None.
    key = (path, req.device, req.channels, req.blocks, req.bilinear_dim)
    cached = _NN_EVALUATOR_CACHE.get(key)
    if cached is not None:
        return cached["evaluator"], cached["net"], path

    try:
        import torch
        from games.kingdomino.network import KingdominoNet
        from games.kingdomino.mcts_az import make_serial_evaluator
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not import NN dependencies: {exc}") from exc

    if not Path(path).exists():
        raise HTTPException(status_code=400, detail=f"Checkpoint not found: {path}")
    try:
        # Load on CPU first so we can read the stored config and build the net at
        # the checkpoint's own architecture before moving it to the device.  The
        # server is the single source of truth for architecture: it reads
        # channels/blocks/bilinear_dim from checkpoint["config"] (saved by
        # self_play.py's save_checkpoint).  Request fields are only fallbacks for
        # legacy checkpoints that predate the config dict; if both are absent we
        # fall back to KingdominoNet's own defaults.
        ck = torch.load(path, map_location="cpu")
        sd = ck.get("model_state", ck) if isinstance(ck, dict) else ck
        cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
        if not isinstance(cfg, dict):
            cfg = {}

        # Precedence per field: checkpoint config → request override → the
        # KingdominoNet constructor defaults (96/8/64), matching
        # elo_rating.checkpoint_arch().
        def _arch(field: str, override: Optional[int], default: int) -> int:
            if field in cfg and cfg[field] is not None:
                return int(cfg[field])
            if override is not None:
                return int(override)
            return default

        channels = _arch("channels", req.channels, 96)
        blocks = _arch("blocks", req.blocks, 8)
        bilinear_dim = _arch("bilinear_dim", req.bilinear_dim, 64)

        net = KingdominoNet(
            channels=channels,
            blocks=blocks,
            bilinear_dim=bilinear_dim,
        ).to(req.device)
        net.load_state_dict(sd)
        net.eval()
        # The advisor searches with the TRAINING value frame: the win-gated form
        # (1-B)*win + B*win^4*margin at B=alpha=0.5 (2026-07-06 change; was
        # alpha=0.0 pure win probability).  The win^4 gate means margin only
        # influences the search once the win is essentially decided — fixing the
        # observed "sloppy play above 95% win prob" without ever trading win
        # probability for points in contested positions.  Q values are therefore
        # the win-gated search value, not a calibrated win probability; (Q+1)/2
        # is still surfaced as a [0,1] score (margin-tinted only when decided),
        # and the net's own win head (root_inference.win_prob) is the calibrated
        # probability readout.
        evaluator = make_serial_evaluator(net, device=req.device, alpha=0.5)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load NN checkpoint {path}: {exc}") from exc

    _NN_EVALUATOR_CACHE[key] = {"evaluator": evaluator, "net": net}
    return evaluator, net, path


_RUST_ADVISOR_EVAL_CACHE: dict[Any, Any] = {}
# Wave size for the Rust advisor search's leaf batching (virtual loss). 8 keeps
# VL distortion negligible at advisor sim counts while giving the GPU real
# batches instead of the serial path's batch-of-1 forwards.
_RUST_ADVISOR_LEAF_BATCH = 8


def _rust_open_loop_search(state: GameState, net, req_key, device: str,
                           sims: int, seed: int):
    """Rust-engine advisor search. Returns (visit_counts, value0, action_info)
    where visit_counts maps Python action -> visits, value0 is the search-mean
    root value (player-0 frame), and action_info maps action -> (prior,
    q_win_prob | None). Raises on any failure — callers fall back to the
    Python OpenLoopMCTS path.

    Only called for deck > 4 roots: terminal-adjacent roots stay on the Python
    path, whose exact-endgame hook solves them perfectly (the Rust search has
    no exact hook by design — see advisor_open_loop_search's docstring).
    """
    import kingdomino_rust as kr
    from games.kingdomino.action_codec import encode_action
    from games.kingdomino.endgame_solver import _rust_state_from_python

    ev = _RUST_ADVISOR_EVAL_CACHE.get(req_key)
    if ev is None:
        from games.kingdomino.self_play import make_rust_evaluator
        # Training value frame (win-gated, B=0.5) — same rationale as the
        # serial evaluator in _load_nn_evaluator.
        ev = make_rust_evaluator(net, device=device, alpha=0.5)
        if len(_RUST_ADVISOR_EVAL_CACHE) >= 4:
            _RUST_ADVISOR_EVAL_CACHE.clear()
        _RUST_ADVISOR_EVAL_CACHE[req_key] = ev

    rs = _rust_state_from_python(state)
    children, value0 = kr.advisor_open_loop_search(
        rs, ev, int(sims),
        dirichlet_eps=0.0,
        cpuct=1.5,
        seed=int(seed) & 0xFFFF_FFFF_FFFF_FFFF,
        leaf_batch=_RUST_ADVISOR_LEAF_BATCH,
        alpha=0.5,
    )
    actor = int(state.current_actor)
    idx_to_action = {int(encode_action(a, state)): a for a in state.legal_actions()}
    visit_counts: dict[Any, float] = {}
    action_info: dict[Any, tuple[float, Optional[float]]] = {}
    for idx, visits, value_sum, prior in children:
        action = idx_to_action.get(int(idx))
        if action is None:
            continue  # root child from a stale index — impossible at the root, defensive
        visit_counts[action] = float(visits)
        if visits > 0:
            q0 = value_sum / visits
            q_actor = q0 if actor == 0 else -q0
            q_win_prob = (q_actor + 1.0) / 2.0
        else:
            q_win_prob = None
        action_info[action] = (float(prior), q_win_prob)
    if not visit_counts:
        raise RuntimeError("rust advisor search returned no legal root children")
    return visit_counts, float(value0), action_info


def _root_trajectory(net, state: GameState, device: str) -> dict[str, Any]:
    """One forward pass on the root state — the network's pre-search trajectory
    estimate (projected final scores + win probability).

    This is a SINGLE root forward pass, NOT search-averaged: it reflects the
    network's prior belief before any tree exploration.  (The per-action Q
    values in the recommendations ARE search-updated, and are win probabilities
    under the advisor's win-gated B=0.5 search — margin-tinted only in
    decided positions.)  Surfacing the un-searched root
    readout is a known, accepted limitation.

    The own/opp heads are normalized by score_scale=100 and are from the CURRENT
    ACTOR's perspective (own = the actor's projected final score, opp = the
    opponent's), so they are multiplied back to point estimates here.
    """
    import torch
    from games.kingdomino.encoder import encode_state

    mb, ob, flat = encode_state(state, state.current_actor)
    with torch.inference_mode():
        mb_t = torch.from_numpy(mb).unsqueeze(0).to(device)
        ob_t = torch.from_numpy(ob).unsqueeze(0).to(device)
        flat_t = torch.from_numpy(flat).unsqueeze(0).to(device)
        own, opp, win_prob, _logits = net(mb_t, ob_t, flat_t)
    return {
        "own_score_est": float(own.item() * 160.0),
        "opp_score_est": float(opp.item() * 160.0),
        "score_margin_est": float((own.item() - opp.item()) * 160.0),
        "win_prob": float(win_prob.item()),
    }


def _choose_nn_action(state: GameState, req: BotActionRequest):
    if state.phase == Phase.GAME_OVER:
        raise HTTPException(status_code=400, detail="Game is already over.")
    # Bot-action path ignores the net (no trajectory/prior readout needed here).
    evaluator, net, checkpoint_path = _load_nn_evaluator(req)
    try:
        import numpy as np
        from games.kingdomino.mcts_az import OpenLoopMCTS, run_pimc_open_loop, select_move
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not import MCTS dependencies: {exc}") from exc

    np_rng = np.random.default_rng(int(req.seed) + 104729 * (len(state.history) + 1))
    engine_name = "nn-mcts"
    visit_counts = value0 = None
    # Rust open-loop search for deck > 4 roots: same search semantics
    # (per-simulation deck redeterminization), Rust tree ops + batched leaf
    # evals instead of Python batch-1 forwards. Terminal-adjacent roots stay on
    # the Python engine, whose exact-endgame hook solves them perfectly.
    if len(state.deck) > 4:
        try:
            key = (checkpoint_path, req.device)
            visit_counts, value0, _info = _rust_open_loop_search(
                state, net, key, req.device, int(req.nn_sims),
                int(req.seed) + 104729 * (len(state.history) + 1))
            engine_name = "nn-mcts-rust"
        except Exception as exc:  # any failure → Python path (never a 500)
            print(f"[advisor] rust search unavailable ({exc!r}); using Python engine")
            visit_counts = None
    if visit_counts is None:
        # Open-loop MCTS resamples the deck on every simulation, averaging over
        # deck uncertainty — the correct search for a live advisory context.
        mcts = OpenLoopMCTS(
            evaluator,
            c_puct=1.5,
            n_simulations=int(req.nn_sims),
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            # Training value frame for terminal/exact backups (win-gated B=0.5;
            # the class default 0.8 is the pre-win-gate training value).
            score_scale=160.0,
            margin_gain=2.0,
            alpha=0.5,
            # Advisor policy: always solve reachable endgames exactly. 3600s is a
            # hung-request safeguard, not a budget — with the within-solve TT the
            # worst measured real position solves in well under a minute.
            exact_endgame_max_secs=3600.0,
        )
        visit_counts, value0, _root = run_pimc_open_loop(
            mcts,
            state,
            add_noise=False,
            rng=np_rng,
        )
    action = select_move(visit_counts, temperature=float(req.temperature), rng=np_rng)
    total_visits = sum(float(v) for v in visit_counts.values()) or 1.0
    chosen_visits = float(visit_counts.get(action, 0.0))
    value_actor = float(value0 if state.current_actor == 0 else -value0)
    return action, {
        "engine": engine_name,
        "checkpoint_path": checkpoint_path,
        "nn_sims": int(req.nn_sims),
        "temperature": float(req.temperature),
        "root_value_player0": float(value0),
        "root_value_actor": value_actor,
        "chosen_visit_frac": chosen_visits / total_visits,
        "visited_actions": len(visit_counts),
        "total_visits": total_visits,
    }


def _choose_bot_action(state: GameState, req: BotActionRequest):
    mode = req.mode.strip().lower()
    if mode == "random":
        return _choose_random_action(state, req.seed)
    if mode == "greedy":
        return _choose_greedy_action(state)
    if mode in ("nn", "model", "az", "alphazero"):
        return _choose_nn_action(state, req)
    raise HTTPException(status_code=400, detail=f"Unknown bot mode {req.mode!r}; expected random, greedy, or nn.")


def _heuristic_action_score(state: GameState, action) -> tuple[float, dict[str, Any]]:
    """Tiny model-free placeholder for /api/recommend.

    This is not meant to be strong.  It gives the frontend a stable advisor
    response shape before NN/MCTS is wired in.
    """
    actor = None if state.phase == Phase.GAME_OVER else state.current_actor
    before = state.scores()[actor] if actor is not None else 0
    try:
        nxt = state.step(action)
        after = nxt.scores()[actor] if actor is not None else before
        score_delta = float(after - before)
    except Exception:
        score_delta = -999.0

    pick_bonus = 0.0
    pick_id = getattr(action, "pick_domino_id", None)
    if isinstance(action, PickAction):
        pick_id = action.domino_id
    if pick_id is not None:
        d = DOMINOES[int(pick_id)]
        pick_bonus = 5.0 * float(d.a.crowns + d.b.crowns) + 0.02 * float(pick_id)

    discard_penalty = -20.0 if isinstance(action, TurnAction) and action.placement is None else 0.0
    total = score_delta + pick_bonus + discard_penalty
    return total, {"score_delta": score_delta, "pick_bonus": pick_bonus, "discard_penalty": discard_penalty}


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="web_static/index.html not found")
    return FileResponse(index_path)


@app.get("/api/schema")
def schema() -> dict[str, Any]:
    return {
        "protocol_version": "kingdomino-advisor-v0.1",
        "state": "Public GameState JSON. Future BGA extension should produce this shape.",
        "recommend_request": RecommendRequest.model_json_schema(),
        "recommend_response_fields": [
            "ok", "value", "search_ms", "num_simulations", "recommendations",
        ],
        "local_lab_endpoints": ["/api/undo-action", "/api/jump-to-step"],
    }


@app.post("/api/new-game")
def new_game(req: NewGameRequest) -> dict[str, Any]:
    cfg = GameConfig(
        players=2,
        board_size=req.board_size,
        canvas_size=req.canvas_size,
        harmony=req.harmony,
        middle_kingdom=req.middle_kingdom,
        mighty_duel=req.mighty_duel,
    )
    state = GameState.new(seed=req.seed, config=cfg, start_player=req.start_player)
    session_id = uuid4().hex
    _SESSIONS[session_id] = state
    _SESSION_TIMELINES[session_id] = [state]
    return _response_for_session(session_id)


@app.get("/api/state")
def get_state(session_id: str = Query(...), debug: bool = False) -> dict[str, Any]:
    return _response_for_session(session_id, include_debug=debug)


@app.get("/api/legal-actions")
def get_legal_actions(session_id: str = Query(...)) -> dict[str, Any]:
    state = _get_state(session_id)
    return {"ok": True, "session_id": session_id, "session": _session_meta(session_id), "legal_actions": legal_actions_json(state)}


@app.post("/api/apply-action")
def apply_action(req: ApplyActionRequest) -> dict[str, Any]:
    state = _get_state(req.session_id)
    action = _select_legal_action(state, req.legal_index, req.action_id)
    new_state = state.step(action)
    _SESSIONS[req.session_id] = new_state
    timeline = _ensure_timeline(req.session_id)
    timeline.append(new_state)
    return _response_for_session(req.session_id)




@app.post("/api/bot-action")
def bot_action(req: BotActionRequest) -> dict[str, Any]:
    """Choose a bot action for the current session, optionally applying it.

    This is intentionally part of the same web/advisor protocol surface used by
    the local UI.  The future BGA extension should use /api/recommend, not this
    local-lab helper, but the action JSON returned here has the same shape.
    """
    import time
    t0 = time.perf_counter()
    state = _get_state(req.session_id)
    if state.phase == Phase.GAME_OVER:
        return {"ok": True, "session_id": req.session_id, "game_over": True, "state": state_to_public_json(state), "legal_actions": []}

    action, info = _choose_bot_action(state, req)
    legal = state.legal_actions()
    try:
        legal_index = next(i for i, a in enumerate(legal) if a == action)
    except StopIteration:
        # Some canonicalised symmetric actions may not compare equal by object
        # identity.  Fall back to action_id matching.
        target_id = action_to_json(state, action, -1)["action_id"]
        legal_index = next(
            (i for i, a in enumerate(legal) if action_to_json(state, a, i)["action_id"] == target_id),
            -1,
        )
    action_json = action_to_json(state, action, legal_index)
    elapsed_ms = int(round((time.perf_counter() - t0) * 1000))

    if not req.apply:
        return {
            "ok": True,
            "session_id": req.session_id,
            "applied": False,
            "bot": info,
            "elapsed_ms": elapsed_ms,
            "action": action_json,
            "state": state_to_public_json(state),
            "legal_actions": legal_actions_json(state),
        }

    new_state = state.step(action)
    _SESSIONS[req.session_id] = new_state
    timeline = _ensure_timeline(req.session_id)
    timeline.append(new_state)
    resp = _response_for_session(req.session_id)
    resp.update({
        "applied": True,
        "bot": info,
        "elapsed_ms": elapsed_ms,
        "action": action_json,
    })
    return resp


@app.post("/api/undo-action")
def undo_action(req: UndoActionRequest) -> dict[str, Any]:
    timeline = _ensure_timeline(req.session_id)
    if len(timeline) <= 1:
        return _response_for_session(req.session_id)
    steps = min(int(req.steps), len(timeline) - 1)
    for _ in range(steps):
        timeline.pop()
    _SESSIONS[req.session_id] = timeline[-1]
    return _response_for_session(req.session_id)


@app.post("/api/jump-to-step")
def jump_to_step(req: JumpToStepRequest) -> dict[str, Any]:
    timeline = _ensure_timeline(req.session_id)
    step = int(req.step)
    if not 0 <= step < len(timeline):
        raise HTTPException(status_code=400, detail=f"step {step} out of range 0..{len(timeline)-1}")
    # Keep the selected state and discard future timeline states.  This makes
    # subsequent play deterministic and avoids a redo stack for now.
    del timeline[step + 1:]
    _SESSIONS[req.session_id] = timeline[-1]
    return _response_for_session(req.session_id)


@app.post("/api/preview-action")
def preview_action(req: ApplyActionRequest) -> dict[str, Any]:
    state = _get_state(req.session_id)
    action = _select_legal_action(state, req.legal_index, req.action_id)
    preview = state.step(action)
    return {"ok": True, "session_id": req.session_id, "state": state_to_public_json(preview), "action": action_to_json(state, action, req.legal_index or 0)}


@app.get("/api/export-state")
def export_state(session_id: str = Query(...)) -> dict[str, Any]:
    state = _get_state(session_id)
    return {"ok": True, "session_id": session_id, "session": _session_meta(session_id), "state": state_to_debug_json(state)}


@app.post("/api/import-state")
def import_state(req: ImportStateRequest) -> dict[str, Any]:
    session_id = req.session_id or uuid4().hex
    state = state_from_debug_json(req.state)
    _SESSIONS[session_id] = state
    _SESSION_TIMELINES[session_id] = [state]
    return _response_for_session(session_id)


@app.post("/api/state/render")
def render_state(req: ImportStateRequest) -> dict[str, Any]:
    # Stateless helper for a future extension: submit a public/debug state JSON
    # and get the normalized render/action JSON back.
    state = state_from_debug_json(req.state)
    return {"ok": True, "state": state_to_public_json(state), "legal_actions": legal_actions_json(state)}


def _safe_probe_filename(name: Optional[str]) -> str:
    raw = (name or f"kingdomino-advisor-probe-{uuid4().hex}.json").replace("\\", "/").split("/")[-1]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    if not safe:
        safe = f"kingdomino-advisor-probe-{uuid4().hex}.json"
    if not safe.lower().endswith(".json"):
        safe += ".json"
    return safe


@app.post("/api/advisor-probe/save")
def save_advisor_probe(req: AdvisorProbeSaveRequest) -> dict[str, Any]:
    out_dir = Path("runs/kingdomino/advisor_probes")
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = _safe_probe_filename(req.filename)
    path = out_dir / filename
    if path.exists():
        stem = path.stem
        suffix = path.suffix
        for i in range(1, 10000):
            candidate = out_dir / f"{stem}-{i}{suffix}"
            if not candidate.exists():
                path = candidate
                break
    with path.open("w", encoding="utf-8") as f:
        json.dump(req.probe, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return {"ok": True, "path": str(path), "filename": path.name}


class GameLogAppendRequest(BaseModel):
    table_id: Optional[str] = None
    record: dict[str, Any]


# Per-table hash of the last appended record's CORE (state/final only — the
# captured_at timestamp would defeat dedupe) so extension reload / page reload
# can't append the same position twice in a row.
_GAME_LOG_LAST_KEY: dict[str, str] = {}


@app.post("/api/game-log/append")
def append_game_log(req: GameLogAppendRequest) -> dict[str, Any]:
    """Append one passively-captured BGA game record (a decision state or the
    final result) to a per-table JSONL under runs/kingdomino/bga_game_log/.

    Purpose: an off-distribution eval suite now (value/policy calibration on
    human games), and a seed pool for position-seeded self-play later. Human
    MOVES are deliberately not training targets — see the run7 post-mortem
    discussion; positions and outcomes are what's worth keeping."""
    import hashlib

    table = "".join(
        c for c in str(req.table_id or "unknown") if c.isalnum() or c in "-_"
    ) or "unknown"
    core = {k: req.record.get(k)
            for k in ("kind", "state", "final", "gamestate_name", "active_player")}
    key = hashlib.sha256(
        json.dumps(core, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if _GAME_LOG_LAST_KEY.get(table) == key:
        return {"ok": True, "appended": False, "reason": "duplicate"}
    out_dir = Path("runs/kingdomino/bga_game_log")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"table_{table}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(req.record, sort_keys=True, ensure_ascii=False) + "\n")
    _GAME_LOG_LAST_KEY[table] = key
    return {"ok": True, "appended": True, "path": str(path)}


@app.post("/api/recommend")
def recommend(req: RecommendRequest) -> dict[str, Any]:
    """Return advisor recommendations in the shared UI/BGA protocol shape.

    Engines:
      - greedy / heuristic: fast deterministic model-free scoring
      - random: useful sanity check for UI rendering
      - nn / mcts / alphazero: OpenLoopMCTS (deck resampled per simulation) using a checkpoint
      - exact / solver: exhaustive no-chance endgame solve for deck=4 or deck=0 states
      - auto: exact when eligible, otherwise NN/MCTS

    The local web UI and the future BGA extension should both consume this
    endpoint.  Bot-play helpers may apply a move; /api/recommend only reports.
    """
    import time
    t0 = time.perf_counter()
    if req.session_id:
        state = _get_state(req.session_id)
    elif req.state is not None:
        state = state_from_debug_json(req.state)
    else:
        raise HTTPException(status_code=400, detail="Provide session_id or state")

    mode = req.engine.strip().lower()
    if mode in ("exact", "solver", "exact-solver"):
        return _recommend_exact(state, req, request_state=req.state, started_at=t0)
    if mode in ("auto", "exact-auto", "auto-endgame"):
        if _exact_supported_detail(state, req.state) is None:
            return _recommend_exact(state, req, request_state=req.state, started_at=t0)
        mode = "nn"

    actions = state.legal_actions()
    if not actions:
        return {
            "ok": True,
            "engine": req.engine,
            "value": None,
            "search_ms": int(round((time.perf_counter() - t0) * 1000)),
            "num_simulations": 0,
            "recommendations": [],
        }

    top_k = max(1, int(req.top_k))

    # ── Random advisor: exposes protocol with a deliberately uninformative rank.
    if mode == "random":
        rng = random.Random(int(req.seed) + 31 * (len(state.history) + 1))
        shuffled = list(enumerate(actions))
        rng.shuffle(shuffled)
        recs = []
        denom = float(min(top_k, len(shuffled))) or 1.0
        for rank, (i, action) in enumerate(shuffled[:top_k], start=1):
            recs.append({
                "rank": rank,
                **action_to_json(state, action, i),
                "prior": None,
                "visit_frac": 1.0 / denom,
                "q_value": None,
                "is_legal": True,
                "debug": {"engine": "random"},
            })
        return {
            "ok": True,
            "engine": "random",
            "value": None,
            "search_ms": int(round((time.perf_counter() - t0) * 1000)),
            "num_simulations": 0,
            "recommendations": recs,
        }

    # ── NN/MCTS advisor: same network/search path as the NN bot, but returns top-k.
    if mode in ("nn", "model", "az", "alphazero", "mcts", "nn-mcts"):
        # Reuse the BotActionRequest loader by building a compatible request.
        bot_req = BotActionRequest(
            session_id=req.session_id or "",
            mode="nn",
            apply=False,
            checkpoint_path=req.checkpoint_path,
            nn_sims=int(req.nn_sims or req.num_simulations or 50),
            determinizations=int(req.determinizations),
            temperature=float(req.temperature),
            device=req.device,
            # Pass architecture overrides through untouched (may be None, in
            # which case the loader reads arch from the checkpoint config).
            channels=req.channels,
            blocks=req.blocks,
            bilinear_dim=req.bilinear_dim,
            seed=int(req.seed),
        )
        evaluator, net, checkpoint_path = _load_nn_evaluator(bot_req)
        try:
            import numpy as np
            from games.kingdomino.mcts_az import OpenLoopMCTS, run_pimc_open_loop
            from games.kingdomino.action_codec import encode_action
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not import MCTS dependencies: {exc}") from exc

        # Single root forward pass (pre-search trajectory estimate); surfaced as
        # "root_inference" below.  Done before the search so it reflects the
        # network's prior belief, independent of tree exploration.
        root_traj = _root_trajectory(net, state, req.device)

        sims = int(bot_req.nn_sims)
        np_rng = np.random.default_rng(int(req.seed) + 104729 * (len(state.history) + 1))
        engine_name = "nn-mcts"
        visit_counts = value0 = root = rust_info = None
        # Rust open-loop search for deck > 4 roots (see _choose_nn_action).
        if len(state.deck) > 4:
            try:
                key = (checkpoint_path, req.device)
                visit_counts, value0, rust_info = _rust_open_loop_search(
                    state, net, key, req.device, sims,
                    int(req.seed) + 104729 * (len(state.history) + 1))
                engine_name = "nn-mcts-rust"
            except Exception as exc:
                print(f"[advisor] rust search unavailable ({exc!r}); using Python engine")
                visit_counts = rust_info = None
        if visit_counts is None:
            # Open-loop MCTS resamples the deck per simulation, so raising sim
            # counts genuinely explores varied futures rather than deepening one
            # fixed deck.
            mcts = OpenLoopMCTS(
                evaluator,
                c_puct=1.5,
                n_simulations=sims,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25,
                # Training value frame (win-gated B=0.5); see _choose_nn_action.
                score_scale=160.0,
                margin_gain=2.0,
                alpha=0.5,
                # Advisor policy: always solve reachable endgames exactly (see
                # _choose_nn_action for rationale).
                exact_endgame_max_secs=3600.0,
            )
            try:
                visit_counts, value0, root = run_pimc_open_loop(
                    mcts,
                    state,
                    add_noise=False,
                    rng=np_rng,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": str(exc),
                        "phase": state.phase.name,
                        "current_actor": None if state.phase == Phase.GAME_OVER else int(state.current_actor),
                        "current_row": [int(d) for d in state.current_row],
                        "pending_claims": [_claim_to_json(c) for c in state.pending_claims],
                        "next_claims": [_claim_to_json(c) for c in state.next_claims],
                        "deck_count": int(len(state.deck)),
                        "legal_action_count": int(len(actions)),
                    },
                ) from exc
        total_visits = sum(float(v) for v in visit_counts.values()) or 1.0
        # Map actions to their current legal-index by stable action_id, because
        # determinization can produce equivalent action objects that do not share
        # identity with the public state's legal_actions() list.
        id_to_index = {
            action_to_json(state, a, i)["action_id"]: i
            for i, a in enumerate(actions)
        }
        actor = int(state.current_actor)
        rows = []
        for action, count in visit_counts.items():
            aj = action_to_json(state, action, -1)
            idx = id_to_index.get(aj["action_id"], -1)
            if idx >= 0:
                aj = action_to_json(state, actions[idx], idx)
            if rust_info is not None:
                # Rust path: per-action (prior, q_win_prob) computed in
                # _rust_open_loop_search from the OL root children.
                prior, q_win_prob = rust_info.get(action, (None, None))
            else:
                # Open-loop root children are keyed by slot-relative joint index,
                # not action objects, so re-encode the action to find its child.
                child = root.children.get(encode_action(action, state))
                prior = float(child.prior) if child is not None else None
                # child.value is player-0 frame; flip to the acting player's
                # frame, then map the [-1, 1] value to a [0, 1] win probability.
                # Under the win-gated B=0.5 search this Q is the search value
                # (margin-tinted only when decided), squashed to [0,1];
                # root_inference.win_prob is the calibrated probability.
                if child is not None and child.visit_count > 0:
                    q_actor = child.value if actor == 0 else -child.value
                    q_win_prob = (q_actor + 1.0) / 2.0
                else:
                    q_win_prob = None
            rows.append((float(count), idx, aj, prior, q_win_prob))
        rows.sort(key=lambda x: x[0], reverse=True)
        recs = []
        for rank, (count, idx, aj, prior, q_win_prob) in enumerate(rows[:top_k], start=1):
            recs.append({
                "rank": rank,
                **aj,
                "prior": prior,
                "visit_frac": count / total_visits,
                "q_win_prob": q_win_prob,
                "q_value": None,
                "is_legal": idx >= 0,
                "debug": {"engine": engine_name, "legal_index_resolved": idx},
            })
        value_actor = float(value0 if state.current_actor == 0 else -value0)
        return {
            "ok": True,
            "engine": engine_name,
            "checkpoint_path": checkpoint_path,
            "value": value_actor,
            "root_win_prob": max(0.0, min(1.0, (value_actor + 1.0) / 2.0)),
            "root_value_player0": float(value0),
            # Network's pre-search trajectory readout (own/opp/win_prob); a single
            # root forward pass, not search-averaged.  See _root_trajectory.
            "root_inference": root_traj,
            "search_ms": int(round((time.perf_counter() - t0) * 1000)),
            "num_simulations": sims,
            "total_visits": total_visits,
            "recommendations": recs,
        }

    # ── Greedy/heuristic advisor: fast default and current placeholder behavior.
    if mode not in ("greedy", "heuristic", "heuristic-placeholder"):
        raise HTTPException(status_code=400, detail=f"Unknown advisor engine {req.engine!r}; expected greedy, random, nn, exact, or auto.")

    scored = []
    for i, action in enumerate(actions):
        score, parts = _heuristic_action_score(state, action)
        aj = action_to_json(state, action, i)
        scored.append((score, aj, parts))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    denom = sum(max(0.0, s[0] - (top[-1][0] if top else 0.0) + 1.0) for s in top) or 1.0
    recs = []
    for rank, (score, aj, parts) in enumerate(top, start=1):
        pseudo_visit = max(0.0, score - top[-1][0] + 1.0) / denom if top else 0.0
        recs.append({
            "rank": rank,
            **aj,
            "prior": None,
            "visit_frac": pseudo_visit,
            "q_value": score,
            "is_legal": True,
            "debug": parts,
        })
    search_ms = int(round((time.perf_counter() - t0) * 1000))
    return {
        "ok": True,
        "engine": "greedy-heuristic",
        "value": None,
        "search_ms": search_ms,
        "num_simulations": 0,
        "recommendations": recs,
    }
