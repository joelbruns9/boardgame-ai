from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from games.kingdomino.dominoes import DOMINOES, Terrain
from games.kingdomino.web_app import (
    RecommendRequest,
    legal_actions_json,
    recommend,
    state_from_debug_json,
    state_to_public_json,
)


# ─────────────────────────────────────────────────────────────────────────────
# Illegal-move probe (domino 37 lake/grassland+crown)
# ─────────────────────────────────────────────────────────────────────────────
#
# We saw the advisor recommend an ILLEGAL placement that the engine's
# legal-action generator produced.  This probe reconstructs that exact position
# THROUGH THE REAL IMPORT PATH (state_from_debug_json) and prints engine-vs-truth
# side by side so we can tell which of three hypotheses is occurring:
#
#   (A) Import bug    — the engine board's terrain at some cell differs from the
#                       true list below (e.g. an [x,y] vs [y,x] transpose, or a
#                       stale Board._cell cache that was not rebuilt on import).
#                       The engine then "sees" a lake where there is grassland,
#                       so the move looks legal.
#   (B) Legality bug  — terrain is imported correctly, but half_connects /
#                       is_legal_placement is wrong (e.g. it accepts adjacency to
#                       ANY occupied tile regardless of terrain, or it matches the
#                       wrong half's terrain).
#   (C) Decoding bug  — the action's reported cells (10,4)/(10,5) are not the
#                       cells the engine actually evaluated (codec/coord mismatch).
#
# The task uses "real-world" terrain names; map them to engine Terrain enum:
#   swamp=SWAMP, grassland=GRASS, lake=WATER, field=WHEAT, forest=FOREST.
_PROBE_TERRAIN = {
    "swamp": Terrain.SWAMP,
    "grassland": Terrain.GRASS,
    "lake": Terrain.WATER,
    "field": Terrain.WHEAT,
    "forest": Terrain.FOREST,
    "castle": Terrain.CASTLE,
}

# Active player's board in engine coords; castle at (7,7).  This is the exact
# failing position.  (x, y) -> terrain name.
_PROBE_CELLS: dict[tuple[int, int], str] = {
    (6, 6): "swamp",
    (7, 6): "swamp",
    (10, 6): "grassland",
    (6, 7): "lake",
    (7, 7): "castle",
    (8, 7): "field",
    (9, 7): "grassland",
    (10, 7): "grassland",
    (6, 8): "lake",
    (7, 8): "forest",
    (8, 8): "field",
}

# Opponent's board (board 1), castle at (7,7). A small distinct layout used to
# exercise SYMMETRIC two-board reconstruction + readback: the extension now
# reconstructs the opponent's board the same way as the active player's, so the
# import path must round-trip both boards with zero mismatches.
_PROBE_OPP_CELLS: dict[tuple[int, int], str] = {
    (7, 7): "castle",
    (8, 7): "field",
    (9, 7): "forest",
    (7, 8): "lake",
    (6, 7): "grassland",
}

# Active domino and the illegal placement the engine recommended.
_PROBE_DOMINO_ID = 37  # WATER(lake) 0c / GRASS(grassland) 1c
# grassland+crown half at (10,4), lake half at (10,5)
_PROBE_GRASS_CELL = (10, 4)
_PROBE_LAKE_CELL = (10, 5)


def _probe_state_json() -> dict[str, Any]:
    """Build a debug-state JSON for the failing position.

    Uses FINAL_PLACEMENT so the only legal actions are placements of domino 37
    (no pick combinatorics), isolating placement legality.  The active player
    (board 0) holds the claim on domino 37.
    """
    cells = []
    for (x, y), name in _PROBE_CELLS.items():
        terr = int(_PROBE_TERRAIN[name])
        crowns = 0
        domino_id = -1 if name == "castle" else 0
        # Mark the two grassland-with-crown tiles? The pre-existing board has no
        # crowns relevant to legality; keep crowns=0 for placed terrain.
        cells.append({
            "x": x,
            "y": y,
            "terrain": _PROBE_TERRAIN[name].name,
            "terrain_id": terr,
            "crowns": crowns,
            "domino_id": domino_id,
        })
    board0 = {"castle_pos": [7, 7], "cells": cells}
    # Board 1: opponent layout, built through the same cell shape as board 0.
    opp_cells = []
    for (x, y), name in _PROBE_OPP_CELLS.items():
        terr = int(_PROBE_TERRAIN[name])
        opp_cells.append({
            "x": x,
            "y": y,
            "terrain": _PROBE_TERRAIN[name].name,
            "terrain_id": terr,
            "crowns": 0,
            "domino_id": -1 if name == "castle" else 0,
        })
    board1 = {"castle_pos": [7, 7], "cells": opp_cells}
    return {
        "rules": {
            "players": 2, "board_size": 7, "canvas_size": 15,
            "harmony": True, "middle_kingdom": True, "mighty_duel": True,
        },
        "phase": "FINAL_PLACEMENT",
        "actor_index": 0,
        "initial_pick_count": 0,
        "start_player": 0,
        "current_row": [],
        "pending_claims": [{"player": 0, "domino_id": _PROBE_DOMINO_ID}],
        "next_claims": [],
        "boards": [board0, board1],
        "debug": {"deck": [], "history": []},
    }


def _terrain_label(value: int) -> str:
    try:
        return Terrain(int(value)).name
    except Exception:
        return f"?{value}"


def _engine_neighbors(board, x: int, y: int) -> list[str]:
    """Orthogonal neighbours of (x,y) as the engine's _cell cache sees them."""
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        t = board._cell.get((nx, ny))
        if t is None:
            out.append(f"({nx},{ny})=empty")
        else:
            out.append(f"({nx},{ny})={_terrain_label(t)}")
    return out


def _truth_legal_for_cells(
    grass_cell: tuple[int, int], lake_cell: tuple[int, int]
) -> bool:
    """Independent (engine-free) legality check for placing domino 37.

    GRASS half at grass_cell, WATER half at lake_cell.  Legal iff at least one
    half is orthogonally adjacent to a PRE-EXISTING matching-terrain tile or the
    castle, using the hardcoded _PROBE_CELLS truth board.
    """
    truth = {xy: _PROBE_TERRAIN[name] for xy, name in _PROBE_CELLS.items()}

    def half_ok(cell: tuple[int, int], want: Terrain) -> bool:
        cx, cy = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            t = truth.get((cx + dx, cy + dy))
            if t is not None and (t == Terrain.CASTLE or t == want):
                return True
        return False

    return half_ok(grass_cell, Terrain.GRASS) or half_ok(lake_cell, Terrain.WATER)


def _display_map_for_placement(domino, placement) -> dict[tuple[int, int], Terrain]:
    """Replicate the extension's CORRECTED half->cell display mapping.

    Mirrors the fixed JS in extension_kingdomino/content.js (miniBoardEl):
        h1, h2 = (b, a) if flipped else (a, b)   # h1 -> (x1,y1), h2 -> (x2,y2)
    Returns {(x,y): Terrain} for the two placed cells, i.e. what the advisor
    would *display* at each cell. Regression guard against the flipped-swap bug.
    """
    h1, h2 = (domino.b, domino.a) if placement.flipped else (domino.a, domino.b)
    return {
        (placement.x1, placement.y1): h1.terrain,
        (placement.x2, placement.y2): h2.terrain,
    }


def _readback_board(board, truth_cells: dict[tuple[int, int], str], label: str) -> int:
    """Print a terrain[y,x] vs _cell vs expected table for one board.

    Returns the number of mismatched cells (ARRAY!=EXPECTED or CACHE!=ARRAY).
    Shared by the active and opponent boards so both use the identical readback.
    """
    print(label)
    print("-" * len(label))
    print(f"{'cell':>10} | {'expected':>10} | {'terrain[y,x]':>12} | {'_cell':>8} | flags")
    mismatches = 0
    for (x, y), name in sorted(truth_cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        expected = _PROBE_TERRAIN[name]
        arr_val = int(board.terrain[y, x])
        cache_val = board._cell.get((x, y))
        arr_name = _terrain_label(arr_val)
        cache_name = "MISSING" if cache_val is None else _terrain_label(cache_val)
        flags = []
        if arr_val != int(expected):
            flags.append("ARRAY!=EXPECTED")
        if cache_val is None or int(cache_val) != arr_val:
            flags.append("CACHE!=ARRAY")
        if flags:
            mismatches += 1
        print(
            f"({x:>2},{y:>2})   | {expected.name:>10} | {arr_name:>12} | "
            f"{cache_name:>8} | {','.join(flags) if flags else 'ok'}"
        )
    return mismatches


def run_probe_illegal_37() -> None:
    print("=" * 78)
    print("PROBE: illegal domino-37 placement (lake/grassland+crown)")
    print("=" * 78)
    d = DOMINOES[_PROBE_DOMINO_ID]
    print(
        f"active domino {_PROBE_DOMINO_ID}: "
        f"a={d.a.terrain.name}/{d.a.crowns}c  b={d.b.terrain.name}/{d.b.crowns}c"
    )
    print(
        f"engine recommended: grassland half at {_PROBE_GRASS_CELL}, "
        f"lake half at {_PROBE_LAKE_CELL}"
    )
    print()

    # Build the state through the SAME path the server uses.
    state_json = _probe_state_json()
    state = state_from_debug_json(state_json)
    board = state.boards[0]

    # ── Board readback: terrain array AND _cell cache, flag any disagreement. ──
    # Both boards are reconstructed from the SAME captured state and read back
    # with the SAME pattern — mirroring the extension fix that now rebuilds the
    # opponent's board the same authoritative way as the active player's.
    mismatches = _readback_board(
        board, _PROBE_CELLS, "Board readback (active player, board 0)"
    )
    print(f"\nreadback mismatches: {mismatches} "
          f"(>0 strongly implies hypothesis A: import bug)")

    opp_board = state.boards[1]
    print()
    opp_mismatches = _readback_board(
        opp_board, _PROBE_OPP_CELLS, "Board readback (opponent, board 1)"
    )
    print(f"\nopponent readback mismatches: {opp_mismatches}")

    # Symmetric two-board contract: each board must round-trip with 0 mismatches.
    assert mismatches == 0, (
        f"active board (board 0) reconstruction has {mismatches} mismatch(es)"
    )
    assert opp_mismatches == 0, (
        f"opponent board (board 1) reconstruction has {opp_mismatches} mismatch(es)"
    )
    print("\nBoth boards reconstructed with 0 mismatches (active + opponent).")

    # Spot-check the cells the task called out.
    print("\nSpot checks:")
    for xy, want in [((10, 6), "grassland"), ((6, 7), "lake"), ((6, 8), "lake"),
                     ((9, 7), "grassland"), ((10, 7), "grassland")]:
        x, y = xy
        got = _terrain_label(int(board.terrain[y, x]))
        print(f"  ({x},{y}) expected={_PROBE_TERRAIN[want].name:>10} got={got:>10}"
              f"  {'OK' if got == _PROBE_TERRAIN[want].name else 'MISMATCH'}")

    # ── Enumerate legal actions for domino 37. ──
    print("\nLegal actions for domino 37")
    print("---------------------------")
    actions = state.legal_actions()
    placements = []
    for a in actions:
        p = getattr(a, "placement", None)
        if p is not None:
            placements.append(p)
    print(f"legal action count: {len(actions)} (placement actions: {len(placements)})")

    probe_cells = {_PROBE_GRASS_CELL, _PROBE_LAKE_CELL}
    probe_in_legal = False
    # Track where the engine actually puts each half in the matched placement, so
    # we can compare its orientation against the orientation the advisor reported.
    engine_grass_cell: tuple[int, int] | None = None
    engine_lake_cell: tuple[int, int] | None = None
    matched_placement = None  # the engine Placement covering the probe cells
    for p in placements:
        cell_set = {(p.x1, p.y1), (p.x2, p.y2)}
        is_probe = cell_set == probe_cells
        if is_probe:
            probe_in_legal = True
            matched_placement = p
        # Resolve which half lands where, given flipped.
        h1, h2 = (d.b, d.a) if p.flipped else (d.a, d.b)
        marker = "  <<< PROBE cells (10,4)/(10,5)" if is_probe else ""
        # Only spam neighbour detail for the probe action (and any near it),
        # otherwise the list is long.
        if is_probe:
            for (hx, hy), h in (((p.x1, p.y1), h1), ((p.x2, p.y2), h2)):
                if h.terrain == Terrain.GRASS:
                    engine_grass_cell = (hx, hy)
                elif h.terrain == Terrain.WATER:
                    engine_lake_cell = (hx, hy)
            print(
                f"  ({p.x1},{p.y1})={h1.terrain.name} / "
                f"({p.x2},{p.y2})={h2.terrain.name} flipped={p.flipped}{marker}"
            )
            print(f"      half1 ({p.x1},{p.y1})={h1.terrain.name} neighbors: "
                  f"{_engine_neighbors(board, p.x1, p.y1)}")
            print(f"      half2 ({p.x2},{p.y2})={h2.terrain.name} neighbors: "
                  f"{_engine_neighbors(board, p.x2, p.y2)}")
            print(f"      engine half_connects(half1)="
                  f"{board.half_connects(p.x1, p.y1, h1)}  "
                  f"half_connects(half2)={board.half_connects(p.x2, p.y2, h2)}")

    # If the probe action was NOT in the legal set, still print the neighbour
    # view the engine would use for those exact cells, so we can see what it sees.
    if not probe_in_legal:
        print("  (probe action not in legal set; showing engine's neighbour view "
              "for the probe cells anyway)")
        gx, gy = _PROBE_GRASS_CELL
        lx, ly = _PROBE_LAKE_CELL
        print(f"      grass half ({gx},{gy}) neighbors: {_engine_neighbors(board, gx, gy)}")
        print(f"      lake  half ({lx},{ly}) neighbors: {_engine_neighbors(board, lx, ly)}")
        print(f"      engine half_connects(grass@{_PROBE_GRASS_CELL})="
              f"{board.half_connects(gx, gy, DOMINOES[37].b)}  "
              f"half_connects(lake@{_PROBE_LAKE_CELL})="
              f"{board.half_connects(lx, ly, DOMINOES[37].a)}")

    # Truth-legality for the orientation the ADVISOR REPORTED (grass@10,4 /
    # lake@10,5) vs. the orientation the ENGINE ACTUALLY EVALUATED.
    reported_legal = _truth_legal_for_cells(_PROBE_GRASS_CELL, _PROBE_LAKE_CELL)
    engine_orient_legal = None
    if engine_grass_cell is not None and engine_lake_cell is not None:
        engine_orient_legal = _truth_legal_for_cells(engine_grass_cell, engine_lake_cell)

    print()
    print(f"PROBE (10,4)/(10,5) in legal set: {probe_in_legal}")
    print(f"PROBE expected-legal: {reported_legal}   "
          f"(orientation as REPORTED: grass@{_PROBE_GRASS_CELL}, "
          f"lake@{_PROBE_LAKE_CELL})")
    if engine_grass_cell is not None:
        print(f"ENGINE's actual orientation: grass@{engine_grass_cell}, "
              f"lake@{engine_lake_cell}  -> truth-legal: {engine_orient_legal}")
    print()

    # ── Regression test for the extension display fix ─────────────────────────
    # The flipped-swap bug lived in the extension's placement->display mapping.
    # Replicate the CORRECTED mapping here and assert it places each half on the
    # cell the engine validated as legal: GRASS at (10,5), WATER at (10,4).
    print("Display-mapping regression test (extension half->cell)")
    print("------------------------------------------------------")
    assert matched_placement is not None, (
        "probe placement not in legal set; cannot run display regression"
    )
    disp = _display_map_for_placement(d, matched_placement)
    for (x, y), terr in sorted(disp.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"  display ({x},{y}) = {terr.name}")
    got_105 = disp.get((10, 5))
    got_104 = disp.get((10, 4))
    assert got_105 == Terrain.GRASS, (
        f"FAIL: display at (10,5) should be GRASS, got "
        f"{got_105.name if got_105 else None} — flipped-swap bug present"
    )
    assert got_104 == Terrain.WATER, (
        f"FAIL: display at (10,4) should be WATER, got "
        f"{got_104.name if got_104 else None} — flipped-swap bug present"
    )
    # The corrected orientation must agree with the engine's legality.
    display_legal = _truth_legal_for_cells((10, 5), (10, 4))
    print(f"  PASS: display (10,5)=GRASS, (10,4)=WATER")
    print(f"  PROBE expected-legal (corrected orientation): {display_legal}")
    assert display_legal is True, "corrected orientation should be legal"
    print()

    # ── Verdict ──────────────────────────────────────────────────────────────
    if mismatches:
        print("=> Hypothesis A (IMPORT BUG): board readback disagrees with truth.")
    elif probe_in_legal and engine_orient_legal and not reported_legal:
        print("=> Hypothesis C (DECODE/REPORT BUG): the engine's legal placement is")
        print("   genuinely LEGAL (grass half adjacent to grassland), but its two")
        print("   halves were reported in the SWAPPED orientation. The displayed")
        print("   'grass@(10,4), lake@(10,5)' is illegal; the engine actually means")
        print("   'grass@(10,5), lake@(10,4)'. Bug is in placement orientation")
        print("   reporting/decoding, NOT in the legality check or the import.")
    elif probe_in_legal and not engine_orient_legal:
        print("=> Hypothesis B (LEGALITY-CHECK BUG): engine accepted a placement")
        print("   that is illegal even in the orientation the engine evaluated.")
    elif not probe_in_legal and not reported_legal:
        print("=> Engine correctly EXCLUDES the reported orientation; reproduction")
        print("   via state_from_debug_json did not surface the bug — re-check the")
        print("   real capture's terrain_ids / coords.")
    else:
        print("=> Unexpected combination; inspect the printout above.")


def _load_state(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if "state" in data and isinstance(data["state"], dict):
        return data["state"]
    return data


def _tile_label(domino_id: int) -> str:
    d = DOMINOES[int(domino_id)]
    return (
        f"{domino_id}: {d.a.terrain.name}/{d.a.crowns}c + "
        f"{d.b.terrain.name}/{d.b.crowns}c"
    )


def _pick_id(rec: dict[str, Any]) -> int | None:
    v = rec.get("pick_domino_id")
    return None if v is None else int(v)


def _placement_key(rec: dict[str, Any]) -> str:
    p = rec.get("placement")
    if p is None:
        return "discard"
    return f"({p['x1']},{p['y1']})->({p['x2']},{p['y2']}) flipped={p.get('flipped')}"


def _print_state_summary(state_json: dict[str, Any]) -> None:
    state = state_from_debug_json(state_json)
    print("State summary")
    print("-------------")
    print(f"phase: {state.phase.name}")
    print(f"current_actor: {state.current_actor}")
    print(f"actor_index: {state.actor_index}")
    print(f"current_row: {state.current_row}")
    for d in state.current_row:
        print(f"  row tile {_tile_label(d)}")
    print(f"pending_claims: {[(c.player, c.domino_id) for c in state.pending_claims]}")
    for c in state.pending_claims:
        print(f"  player {c.player} owns/current {_tile_label(c.domino_id)}")
    print(f"next_claims: {[(c.player, c.domino_id) for c in state.next_claims]}")
    print(f"hidden deck count: {len(state.deck)}")
    public = state_to_public_json(state)
    print("boards:")
    for i, board in enumerate(public["boards"]):
        active = " (active)" if i == state.current_actor else ""
        print(f"  player {i}{active}:")
        for cell in sorted(board["cells"], key=lambda c: (c["y"], c["x"], c["domino_id"])):
            print(
                f"    ({cell['x']},{cell['y']}) "
                f"{cell['terrain']} crowns={cell['crowns']} domino={cell['domino_id']}"
            )
    debug = state_json.get("debug") or {}
    placements = debug.get("reconstructed_placements") or []
    if placements:
        print("BGA reconstructed placements:")
        for p in placements:
            print(
                f"  domino {p.get('domino_id')} owner_bga={p.get('owner_bga')} "
                f"player={p.get('player')} rotation={p.get('rotation')} "
                f"offset={p.get('offset')} anchor_bga={p.get('anchor_bga')}"
            )
            for c in p.get("cells", []):
                print(
                    f"    {c.get('side')}: bga={c.get('bga')} "
                    f"engine={c.get('engine')} {c.get('terrain')} crowns={c.get('crowns')}"
                )
    actions = legal_actions_json(state)
    print(f"legal action count: {len(actions)}")
    by_pick: dict[int | None, int] = defaultdict(int)
    for a in actions:
        by_pick[a.get("pick_domino_id")] += 1
    print(f"legal actions by pick: {dict(sorted(by_pick.items(), key=lambda kv: (-1 if kv[0] is None else kv[0])))}")
    print()


def _run_recommendation(
    state_json: dict[str, Any],
    *,
    engine: str,
    sims: int,
    top_k: int,
    checkpoint: str | None,
    device: str,
    channels: int,
    blocks: int,
    bilinear_dim: int,
    seed: int,
) -> dict[str, Any]:
    req = RecommendRequest(
        state=state_json,
        engine=engine,
        num_simulations=sims,
        nn_sims=max(1, sims),
        top_k=top_k,
        checkpoint_path=checkpoint,
        device=device,
        channels=channels,
        blocks=blocks,
        bilinear_dim=bilinear_dim,
        seed=seed,
    )
    return recommend(req)


def _print_recommendation(title: str, out: dict[str, Any], top_k: int) -> None:
    print(title)
    print("-" * len(title))
    print(
        f"engine={out.get('engine')} value={out.get('value')} "
        f"search_ms={out.get('search_ms')} sims={out.get('num_simulations')}"
    )
    recs = out.get("recommendations", [])
    by_pick: dict[int | None, float] = defaultdict(float)
    for rec in recs:
        by_pick[_pick_id(rec)] += float(rec.get("visit_frac") or 0.0)
    if by_pick:
        print("top-k visit share by pick:")
        for pick, frac in sorted(by_pick.items(), key=lambda kv: kv[1], reverse=True):
            label = "none" if pick is None else _tile_label(pick)
            print(f"  {label}: {frac:.3f}")
    print("top actions:")
    for rec in recs[:top_k]:
        pick = _pick_id(rec)
        pick_label = "none" if pick is None else _tile_label(pick)
        print(
            f"  #{rec.get('rank')}: visit={float(rec.get('visit_frac') or 0.0):.3f} "
            f"N={rec.get('visit_count')} legal={rec.get('legal_index')} "
            f"pick={pick_label} placement={_placement_key(rec)} "
            f"label={rec.get('label')}"
        )
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Audit a copied BGA Kingdomino advisor capture.")
    p.add_argument(
        "--probe_illegal_37",
        action="store_true",
        help="Run the self-contained illegal-move diagnostic (no capture needed).",
    )
    p.add_argument("capture_json", type=Path, nargs="?", default=None)
    p.add_argument("--checkpoint", default="checkpoints_m9_warm_currentbest_32x4_s1600_b100k_t300_i60/iter_0060.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--bilinear_dim", type=int, default=64)
    p.add_argument("--sims", default="200,800,1600")
    p.add_argument("--top_k", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include_greedy", action="store_true")
    args = p.parse_args()

    if args.probe_illegal_37:
        run_probe_illegal_37()
        return

    if args.capture_json is None:
        p.error("capture_json is required unless --probe_illegal_37 is given")

    state_json = _load_state(args.capture_json)
    _print_state_summary(state_json)

    if args.include_greedy:
        out = _run_recommendation(
            state_json,
            engine="greedy",
            sims=0,
            top_k=args.top_k,
            checkpoint=None,
            device="cpu",
            channels=args.channels,
            blocks=args.blocks,
            bilinear_dim=args.bilinear_dim,
            seed=args.seed,
        )
        _print_recommendation("Greedy heuristic", out, args.top_k)

    for sims_text in args.sims.split(","):
        sims = int(sims_text.strip())
        out = _run_recommendation(
            state_json,
            engine="nn",
            sims=sims,
            top_k=args.top_k,
            checkpoint=args.checkpoint,
            device=args.device,
            channels=args.channels,
            blocks=args.blocks,
            bilinear_dim=args.bilinear_dim,
            seed=args.seed,
        )
        _print_recommendation(f"NN/MCTS sims={sims}", out, args.top_k)


if __name__ == "__main__":
    main()
