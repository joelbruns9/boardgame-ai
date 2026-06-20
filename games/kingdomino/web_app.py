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
import random
import glob

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


class RecommendRequest(BaseModel):
    session_id: Optional[str] = None
    state: Optional[dict[str, Any]] = None
    engine: str = Field(default="greedy", description="greedy/heuristic, random, or nn")
    num_simulations: int = Field(default=50, ge=0, le=5000)
    top_k: int = Field(default=8, ge=1, le=100)
    checkpoint_path: Optional[str] = None
    nn_sims: int = Field(default=50, ge=1, le=5000)
    determinizations: int = Field(default=1, ge=1, le=16)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    device: str = "cpu"
    channels: int = 64
    blocks: int = 6
    bilinear_dim: int = 64
    seed: int = 0


class BotActionRequest(BaseModel):
    session_id: str
    mode: str = Field(default="greedy", description="random, greedy, or nn")
    apply: bool = True
    # NN/MCTS options.  If checkpoint_path is omitted, the newest common
    # Kingdomino iter_*.pt checkpoint is used when available.
    checkpoint_path: Optional[str] = None
    nn_sims: int = Field(default=50, ge=1, le=5000)
    determinizations: int = Field(default=1, ge=1, le=16)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    device: str = "cpu"
    channels: int = 64
    blocks: int = 6
    bilinear_dim: int = 64
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


def _newest_checkpoint_path() -> Optional[str]:
    promoted = Path("checkpoints_best/kingdomino_current_best.pt")
    if promoted.exists():
        return str(promoted)

    patterns = [
        "checkpoints_scratch_s800_64x6/iter_*.pt",
        "checkpoints*/iter_*.pt",
        "**/iter_*.pt",
    ]
    best: Optional[tuple[int, float, str]] = None
    for pat in patterns:
        for path in glob.glob(pat, recursive=True):
            try:
                import torch
                ck = torch.load(path, map_location="cpu")
                iteration = int(ck.get("iteration", -1)) if isinstance(ck, dict) else -1
            except Exception:
                iteration = -1
            try:
                mtime = Path(path).stat().st_mtime
            except Exception:
                mtime = 0.0
            item = (iteration, mtime, path)
            if best is None or item > best:
                best = item
        if best is not None:
            break
    return None if best is None else best[2]


def _load_nn_evaluator(req: BotActionRequest):
    checkpoint_path = req.checkpoint_path or _newest_checkpoint_path()
    if not checkpoint_path:
        raise HTTPException(
            status_code=400,
            detail="NN bot needs a checkpoint_path, checkpoints_best/kingdomino_current_best.pt, or an iter_*.pt checkpoint in a checkpoints* folder.",
        )
    path = str(Path(checkpoint_path))
    key = (path, req.device, int(req.channels), int(req.blocks), int(req.bilinear_dim))
    cached = _NN_EVALUATOR_CACHE.get(key)
    if cached is not None:
        return cached, path

    try:
        import torch
        from games.kingdomino.network import KingdominoNet
        from games.kingdomino.mcts_az import make_serial_evaluator
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not import NN dependencies: {exc}") from exc

    if not Path(path).exists():
        raise HTTPException(status_code=400, detail=f"Checkpoint not found: {path}")
    try:
        ck = torch.load(path, map_location=req.device)
        sd = ck.get("model_state", ck) if isinstance(ck, dict) else ck
        net = KingdominoNet(
            channels=int(req.channels),
            blocks=int(req.blocks),
            bilinear_dim=int(req.bilinear_dim),
        ).to(req.device)
        net.load_state_dict(sd)
        net.eval()
        evaluator = make_serial_evaluator(net, device=req.device)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load NN checkpoint {path}: {exc}") from exc

    _NN_EVALUATOR_CACHE[key] = evaluator
    return evaluator, path


def _choose_nn_action(state: GameState, req: BotActionRequest):
    if state.phase == Phase.GAME_OVER:
        raise HTTPException(status_code=400, detail="Game is already over.")
    evaluator, checkpoint_path = _load_nn_evaluator(req)
    try:
        import numpy as np
        from games.kingdomino.mcts_az import AlphaZeroMCTS, run_pimc, select_move
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not import MCTS dependencies: {exc}") from exc

    mcts = AlphaZeroMCTS(
        evaluator,
        c_puct=1.5,
        n_simulations=int(req.nn_sims),
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )
    py_rng = random.Random(int(req.seed) + 9176 * (len(state.history) + 1))
    np_rng = np.random.default_rng(int(req.seed) + 104729 * (len(state.history) + 1))
    visit_counts, value0 = run_pimc(
        mcts,
        state,
        py_rng,
        n_determinizations=int(req.determinizations),
        add_noise=False,
        np_rng=np_rng,
    )
    action = select_move(visit_counts, temperature=float(req.temperature), rng=np_rng)
    total_visits = sum(float(v) for v in visit_counts.values()) or 1.0
    chosen_visits = float(visit_counts.get(action, 0.0))
    value_actor = float(value0 if state.current_actor == 0 else -value0)
    return action, {
        "engine": "nn-mcts",
        "checkpoint_path": checkpoint_path,
        "nn_sims": int(req.nn_sims),
        "determinizations": int(req.determinizations),
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


@app.post("/api/recommend")
def recommend(req: RecommendRequest) -> dict[str, Any]:
    """Return advisor recommendations in the shared UI/BGA protocol shape.

    Engines:
      - greedy / heuristic: fast deterministic model-free scoring
      - random: useful sanity check for UI rendering
      - nn / mcts / alphazero: AlphaZeroMCTS + PIMC using a checkpoint

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

    mode = req.engine.strip().lower()
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
            channels=int(req.channels),
            blocks=int(req.blocks),
            bilinear_dim=int(req.bilinear_dim),
            seed=int(req.seed),
        )
        evaluator, checkpoint_path = _load_nn_evaluator(bot_req)
        try:
            import numpy as np
            from games.kingdomino.mcts_az import AlphaZeroMCTS, run_pimc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not import MCTS dependencies: {exc}") from exc

        sims = int(bot_req.nn_sims)
        mcts = AlphaZeroMCTS(
            evaluator,
            c_puct=1.5,
            n_simulations=sims,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        )
        py_rng = random.Random(int(req.seed) + 9176 * (len(state.history) + 1))
        np_rng = np.random.default_rng(int(req.seed) + 104729 * (len(state.history) + 1))
        try:
            visit_counts, value0 = run_pimc(
                mcts,
                state,
                py_rng,
                n_determinizations=int(bot_req.determinizations),
                add_noise=False,
                np_rng=np_rng,
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
        rows = []
        for action, count in visit_counts.items():
            aj = action_to_json(state, action, -1)
            idx = id_to_index.get(aj["action_id"], -1)
            if idx >= 0:
                aj = action_to_json(state, actions[idx], idx)
            rows.append((float(count), idx, aj))
        rows.sort(key=lambda x: x[0], reverse=True)
        recs = []
        for rank, (count, idx, aj) in enumerate(rows[:top_k], start=1):
            recs.append({
                "rank": rank,
                **aj,
                "prior": None,
                "visit_frac": count / total_visits,
                "visit_count": int(count),
                "q_value": None,
                "is_legal": idx >= 0,
                "debug": {"engine": "nn-mcts", "legal_index_resolved": idx},
            })
        value_actor = float(value0 if state.current_actor == 0 else -value0)
        return {
            "ok": True,
            "engine": "nn-mcts",
            "checkpoint_path": checkpoint_path,
            "value": value_actor,
            "root_value_player0": float(value0),
            "search_ms": int(round((time.perf_counter() - t0) * 1000)),
            "num_simulations": sims,
            "determinizations": int(bot_req.determinizations),
            "total_visits": total_visits,
            "recommendations": recs,
        }

    # ── Greedy/heuristic advisor: fast default and current placeholder behavior.
    if mode not in ("greedy", "heuristic", "heuristic-placeholder"):
        raise HTTPException(status_code=400, detail=f"Unknown advisor engine {req.engine!r}; expected greedy, random, or nn.")

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
