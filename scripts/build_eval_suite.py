from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from games.kingdomino.endgame_solver import exact_endgame_value
from games.kingdomino.game import GameState, Phase
from games.kingdomino.print_model_contract import ruleset_hash
from games.kingdomino.web_app import state_to_public_json


DEFAULT_OUT = Path("data/kingdomino/eval_suite_v1.jsonl")


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def public_state_for_suite(state: GameState) -> dict[str, Any]:
    """Serializable public state with hidden bag membership, not true order.

    The existing importer expects `debug.deck` when reconstructing a state. For
    fixed-suite positions this is deliberately sorted bag membership so the
    suite is reproducible and does not preserve the self-play deck order.
    """
    payload = state_to_public_json(state, include_debug=True)
    payload["debug"]["deck"] = sorted(int(d) for d in state.deck)
    payload["debug"]["history"] = []
    payload["visible_history"] = []
    for board in payload.get("boards", []):
        board.pop("terrain_grid", None)
        board.pop("crowns_grid", None)
        board.pop("domino_grid", None)
    return payload


def position_id(public_state: dict[str, Any], source: str, index: int) -> str:
    raw = canonical_json({"source": source, "index": index, "state": public_state})
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def phase_name(state: GameState) -> str:
    placed = sum(len(b.occupied_cells()) - 1 for b in state.boards)
    progress = placed / 96.0
    if progress < 0.25:
        return "opening"
    if progress < 0.75:
        return "midgame"
    return "endgame"


def legal_multi_action(state: GameState) -> bool:
    return state.phase != Phase.GAME_OVER and len(state.legal_actions()) > 1


def random_decision_positions(seed: int, n_games: int) -> list[GameState]:
    out: list[GameState] = []
    for g in range(n_games):
        rng = random.Random(seed + 1009 * g)
        state = GameState.new(seed=seed + g)
        while state.phase != Phase.GAME_OVER:
            if legal_multi_action(state):
                out.append(state)
            actions = state.legal_actions()
            if not actions:
                break
            state = state.step(rng.choice(actions))
    return out


def select_evenly(states: list[GameState], count: int) -> list[GameState]:
    if not states or count <= 0:
        return []
    if len(states) <= count:
        return list(states)
    if count == 1:
        return [states[len(states) // 2]]
    idxs = [round(i * (len(states) - 1) / (count - 1)) for i in range(count)]
    return [states[int(i)] for i in idxs]


def exact_value_if_available(state: GameState, max_secs: float) -> float | None:
    if state.phase == Phase.GAME_OVER:
        return None
    if len(state.deck) not in (0, 4):
        return None
    value, solved = exact_endgame_value(
        state,
        max_secs=max_secs,
        rng=random.Random(0),
        score_scale=160.0,
        margin_gain=2.0,
        alpha=0.8,
    )
    return float(value) if solved else None


def build_suite(
    *,
    seed: int,
    n_games: int,
    per_phase: int,
    endgames: int,
    exact_max_secs: float,
) -> list[dict[str, Any]]:
    positions = random_decision_positions(seed, n_games)
    by_phase = {
        "opening": [s for s in positions if phase_name(s) == "opening"],
        "midgame": [s for s in positions if phase_name(s) == "midgame"],
        "endgame": [s for s in positions if phase_name(s) == "endgame"],
    }

    selected: list[tuple[str, str, GameState]] = []
    for phase in ("opening", "midgame", "endgame"):
        for state in select_evenly(by_phase[phase], per_phase):
            selected.append(("phase_representative", phase, state))

    exact_candidates = [
        s for s in positions
        if s.phase in (Phase.PLACE_AND_SELECT, Phase.FINAL_PLACEMENT)
        and len(s.deck) in (0, 4)
        and legal_multi_action(s)
    ]
    for state in select_evenly(exact_candidates, endgames):
        selected.append(("endgame_exact", "endgame", state))

    records: list[dict[str, Any]] = []
    seen_states: set[str] = set()
    for i, (source, phase, state) in enumerate(selected):
        public_state = public_state_for_suite(state)
        key = canonical_json(public_state)
        if key in seen_states:
            continue
        seen_states.add(key)
        exact = exact_value_if_available(state, exact_max_secs)
        records.append({
            "position_id": position_id(public_state, source, i),
            "ruleset_hash": ruleset_hash(),
            "public_state": public_state,
            "source": source,
            "phase": phase,
            "tiles_remaining": int(len(state.deck)),
            "expected_exact_value": exact,
            "notes": None,
        })
    return records


def write_jsonl(records: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(canonical_json(rec) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Kingdomino fixed eval suite v1.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--per_phase", type=int, default=4)
    parser.add_argument("--endgames", type=int, default=6)
    parser.add_argument("--exact_max_secs", type=float, default=3.0)
    args = parser.parse_args()

    records = build_suite(
        seed=args.seed,
        n_games=args.games,
        per_phase=args.per_phase,
        endgames=args.endgames,
        exact_max_secs=args.exact_max_secs,
    )
    if not records:
        raise SystemExit("No eval-suite records generated.")
    write_jsonl(records, args.out)
    n_exact = sum(1 for r in records if r["expected_exact_value"] is not None)
    phases = {p: sum(1 for r in records if r["phase"] == p)
              for p in ("opening", "midgame", "endgame")}
    print(f"Wrote {len(records)} positions to {args.out}")
    print(f"Exact-valued positions: {n_exact}")
    print(f"Phase counts: {phases}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
