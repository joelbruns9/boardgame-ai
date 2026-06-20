"""
test_rust_game_equiv.py — verify RustGameState steps in lockstep with the
Python GameState across many random games.

This is the gatekeeping validation for Milestone 1 (the Rust engine), before
the encoder / codec / MCTS tree are built on top of it.  For each seed we mirror
a fresh Python GameState into a RustGameState (same deck order, same row, same
start player), then drive BOTH with the same random sequence of legal actions.
At every ply we assert the full state matches:

  - phase, current actor, actor_index, initial_pick_count
  - current_row, deck (exact order)
  - pending_claims, next_claims (player + domino id)
  - both boards' terrain and crowns maps
  - the SET of legal actions (canonicalised, order-independent) and its size

At game end we assert the final scores match.  We also verify the hard-coded
Rust domino table against Python's DOMINOES.

The fiddly correctness spots this is designed to catch (flagged in game.py):
opening pick order, claim sorting on round boundaries, and the two-phase
place+pick turn structure.

Run (PowerShell, one line):
  python -m games.kingdomino.test_rust_game_equiv
"""
from __future__ import annotations

import random
import sys

from games.kingdomino.game import GameState, Phase, PickAction, TurnAction
from games.kingdomino.dominoes import DOMINOES

import kingdomino_rust


# ─── mirroring + translation ────────────────────────────────────────────────
def _rust_from_python(py: GameState) -> "kingdomino_rust.RustGameState":
    """Build a RustGameState that starts identically to a fresh Python state.

    `py.deck` is already the post-deal deck (the four current_row tiles removed),
    which is exactly what RustGameState.new expects.
    """
    return kingdomino_rust.RustGameState(
        py.start_player, list(py.deck), list(py.current_row),
        py.config.harmony, py.config.middle_kingdom,
    )


def _translate(action) -> tuple:
    """Convert a Python engine action to RustGameState.step(*args)."""
    if isinstance(action, PickAction):
        return (None, action.domino_id)
    if isinstance(action, TurnAction):
        p = action.placement
        ptuple = None if p is None else (p.x1, p.y1, p.x2, p.y2, p.flipped)
        return (ptuple, action.pick_domino_id)
    raise TypeError(f"Unexpected action type: {type(action).__name__}")


# ─── canonical legal-action keys (order- and flip-independent) ──────────────
def _placement_key(ptuple, domino_id):
    """Canonical physical key for a placement: the two (cell, terrain, crowns)
    halves, sorted so symmetric/flipped encodings collapse to one key."""
    if ptuple is None:
        return None
    x1, y1, x2, y2, flipped = ptuple
    dom = DOMINOES[domino_id]
    h1, h2 = (dom.b, dom.a) if flipped else (dom.a, dom.b)
    c1 = (x1, y1, int(h1.terrain), h1.crowns)
    c2 = (x2, y2, int(h2.terrain), h2.crowns)
    return (c1, c2) if c1 <= c2 else (c2, c1)


def _py_action_key(action, state: GameState):
    if isinstance(action, PickAction):
        return ("P", action.domino_id)
    p = action.placement
    ptuple = None if p is None else (p.x1, p.y1, p.x2, p.y2, p.flipped)
    domino_id = state.pending_claims[state.actor_index].domino_id
    return ("T", _placement_key(ptuple, domino_id), action.pick_domino_id)


def _rust_action_key(rust_action, domino_id):
    """Canonical key for a Rust (placement, pick) action.

    `domino_id` is the tile being placed this turn, or None in
    INITIAL_SELECTION — which disambiguates pick-only actions (None, Some(d))
    from turn actions (placement-or-None, pick).
    """
    ptuple, pick = rust_action
    if domino_id is None:
        return ("P", pick)
    return ("T", _placement_key(ptuple, domino_id), pick)


# ─── per-ply full-state comparison ──────────────────────────────────────────
def _claims(py_claims):
    return [(c.player, c.domino_id) for c in py_claims]


def _state_mismatch(py: GameState, rs) -> str | None:
    """Return a description of the first state mismatch, or None if identical."""
    if int(py.phase) != rs.phase:
        return f"phase: py={py.phase.name}({int(py.phase)}) rust={rs.phase}"
    if int(py.actor_index) != rs.actor_index:
        return f"actor_index: py={py.actor_index} rust={rs.actor_index}"
    if int(py.initial_pick_count) != rs.initial_pick_count:
        return f"initial_pick_count: py={py.initial_pick_count} rust={rs.initial_pick_count}"
    if list(py.current_row) != rs.current_row():
        return f"current_row: py={list(py.current_row)} rust={rs.current_row()}"
    if list(py.deck) != rs.deck():
        return f"deck: py={list(py.deck)[:8]}... rust={rs.deck()[:8]}..."
    if _claims(py.pending_claims) != rs.pending_claims():
        return f"pending_claims: py={_claims(py.pending_claims)} rust={rs.pending_claims()}"
    if _claims(py.next_claims) != rs.next_claims():
        return f"next_claims: py={_claims(py.next_claims)} rust={rs.next_claims()}"
    if py.phase != Phase.GAME_OVER and py.current_actor != rs.current_actor():
        return f"current_actor: py={py.current_actor} rust={rs.current_actor()}"
    for player in (0, 1):
        # board_terrain/board_crowns return Rust Vec<u8>, which pyo3 maps to
        # Python `bytes`; wrap in list() to compare against the int lists.
        py_terr = py.boards[player].terrain.reshape(-1).tolist()
        if py_terr != list(rs.board_terrain(player)):
            return f"board[{player}].terrain differs"
        py_cr = py.boards[player].crowns.reshape(-1).tolist()
        if py_cr != list(rs.board_crowns(player)):
            return f"board[{player}].crowns differs"
    return None


def _legal_mismatch(py: GameState, rs) -> str | None:
    """Compare canonical legal-action sets; return a description or None."""
    py_legal = py.legal_actions()
    rust_legal = rs.legal_actions()

    turn_phase = py.phase in (Phase.PLACE_AND_SELECT, Phase.FINAL_PLACEMENT)
    domino_id = py.pending_claims[py.actor_index].domino_id if turn_phase else None

    py_keys = [_py_action_key(a, py) for a in py_legal]
    rust_keys = [_rust_action_key(a, domino_id) for a in rust_legal]
    py_set, rust_set = set(py_keys), set(rust_keys)

    if py_set != rust_set:
        only_py = sorted(map(str, py_set - rust_set))[:3]
        only_rust = sorted(map(str, rust_set - py_set))[:3]
        return (f"legal set differs (py={len(py_set)} rust={len(rust_set)}); "
                f"only_py={only_py} only_rust={only_rust}")
    if len(py_keys) != len(rust_keys):
        return f"legal multiplicity differs: py={len(py_keys)} rust={len(rust_keys)}"
    return None


# ─── domino table check ─────────────────────────────────────────────────────
def _check_domino_table() -> int:
    mismatches = 0
    for did, dom in DOMINOES.items():
        rust = kingdomino_rust.domino_halves(did)
        expected = (int(dom.a.terrain), dom.a.crowns, int(dom.b.terrain), dom.b.crowns)
        if rust != expected:
            mismatches += 1
            if mismatches <= 5:
                print(f"  DOMINO TABLE MISMATCH id={did}: rust={rust} expected={expected}")
    return mismatches


# ─── driver ─────────────────────────────────────────────────────────────────
def run_equiv(n_games: int = 2000, seed: int = 0, verbose: bool = False) -> bool:
    table_mismatches = _check_domino_table()

    state_mismatches = 0
    legal_mismatches = 0
    score_mismatches = 0
    games_played = 0

    for g in range(n_games):
        py = GameState.new(seed=seed + g)
        rs = _rust_from_python(py)
        rng = random.Random((seed + g) * 2654435761 & 0xFFFFFFFF)

        ply = 0
        game_ok = True
        while py.phase != Phase.GAME_OVER:
            sm = _state_mismatch(py, rs)
            if sm is not None:
                state_mismatches += 1
                game_ok = False
                if verbose and state_mismatches <= 5:
                    print(f"  [game {g} ply {ply}] STATE {sm}")
                break

            lm = _legal_mismatch(py, rs)
            if lm is not None:
                legal_mismatches += 1
                game_ok = False
                if verbose and legal_mismatches <= 5:
                    print(f"  [game {g} ply {ply}] {lm}")
                break

            action = rng.choice(py.legal_actions())
            py = py.step(action)
            rs = rs.step(*_translate(action))
            ply += 1

        if game_ok:
            # Terminal: both game-over, scores equal.
            if rs.phase != int(Phase.GAME_OVER):
                state_mismatches += 1
                if verbose and state_mismatches <= 5:
                    print(f"  [game {g}] rust not GAME_OVER at end: phase={rs.phase}")
            else:
                py_scores = tuple(py.scores())
                rust_scores = rs.scores()
                if py_scores != rust_scores:
                    score_mismatches += 1
                    if verbose and score_mismatches <= 5:
                        print(f"  [game {g}] SCORE py={py_scores} rust={rust_scores}")
            games_played += 1

    print(f"\n=== RustGameState equivalence: {n_games} games "
          f"({games_played} completed) ===")
    print(f"  domino table mismatches: {table_mismatches}")
    print(f"  state mismatches:        {state_mismatches}")
    print(f"  legal-set mismatches:    {legal_mismatches}")
    print(f"  score mismatches:        {score_mismatches}")
    ok = (table_mismatches == 0 and state_mismatches == 0
          and legal_mismatches == 0 and score_mismatches == 0)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    n = 2000
    for a in sys.argv[1:]:
        if a.startswith("--games="):
            n = int(a.split("=", 1)[1])
    ok = run_equiv(n_games=n, verbose=verbose)
    sys.exit(0 if ok else 1)
