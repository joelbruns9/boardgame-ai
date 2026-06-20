"""
test_rust_action_equiv.py — verify the Rust action codec matches Python exactly,
in both VALUE and ORDER, across many random game states.

Milestone 3 gate.  Two independent checks at every state in M1's lockstep walk:

  1. ORDERED-SEQUENCE EQUALITY.  Python's legal_actions() is now sorted by joint
     index (the canonical order), so [encode_action(a, state) for a in
     legal_actions()] is the canonical ascending sequence.  Rust's
     legal_action_indices() must equal it element-for-element — same set AND
     same order.  M4's PUCT tie-breaking depends on this ordering being
     identical between the two engines.

  2. PER-ACTION INDEX EQUALITY (codec correctness, independent of ordering).
     For every legal Python action, encode_action(action, state) must equal
     RustGameState.encode_action(translate(action)).  This catches index
     arithmetic bugs directly, action by action.

Both must be green before Milestone 4.

Run (PowerShell, one line):
  python -m games.kingdomino.test_rust_action_equiv
"""
from __future__ import annotations

import random
import sys

from games.kingdomino.game import GameState, Phase
from games.kingdomino.action_codec import encode_action, NUM_JOINT_ACTIONS
from games.kingdomino.test_rust_game_equiv import _rust_from_python, _translate

import kingdomino_rust


def run_equiv(n_games: int = 2000, seed: int = 0, verbose: bool = False) -> bool:
    order_mismatches = 0       # Rust ordered index sequence != Python's
    peraction_mismatches = 0   # Rust encode_action(a) != Python encode_action(a)
    not_ascending = 0          # Python canonical sequence not strictly ascending
    out_of_range = 0           # any index outside [0, NUM_JOINT_ACTIONS)
    states_checked = 0
    actions_checked = 0

    for g in range(n_games):
        py = GameState.new(seed=seed + g)
        rs = _rust_from_python(py)
        rng = random.Random((seed + g) * 2654435761 & 0xFFFFFFFF)

        while py.phase != Phase.GAME_OVER:
            py_actions = py.legal_actions()
            py_seq = [encode_action(a, py) for a in py_actions]
            rust_seq = rs.legal_action_indices()
            states_checked += 1

            # Sanity: canonical sequence must be strictly ascending and in range.
            if py_seq != sorted(py_seq) or len(set(py_seq)) != len(py_seq):
                not_ascending += 1
                if verbose and not_ascending <= 5:
                    print(f"  [g{g}] python sequence not strictly ascending: {py_seq[:12]}")
            if any(not (0 <= i < NUM_JOINT_ACTIONS) for i in rust_seq):
                out_of_range += 1
                if verbose and out_of_range <= 5:
                    bad = [i for i in rust_seq if not (0 <= i < NUM_JOINT_ACTIONS)]
                    print(f"  [g{g}] rust index out of range: {bad[:8]}")

            # 1. Ordered-sequence equality.
            if py_seq != rust_seq:
                order_mismatches += 1
                if verbose and order_mismatches <= 5:
                    _report_seq(g, py.phase, py_seq, rust_seq)

            # 2. Per-action index equality.
            for a, py_idx in zip(py_actions, py_seq):
                placement, pick = _translate(a)
                rust_idx = rs.encode_action(placement, pick)
                actions_checked += 1
                if rust_idx != py_idx:
                    peraction_mismatches += 1
                    if verbose and peraction_mismatches <= 5:
                        print(f"  [g{g} {py.phase.name}] per-action: action={a} "
                              f"py={py_idx} rust={rust_idx} translate={(placement, pick)}")

            action = rng.choice(py_actions)
            py = py.step(action)
            rs = rs.step(*_translate(action))

    print(f"\n=== Rust action-codec equivalence: {n_games} games, "
          f"{states_checked} states, {actions_checked} actions ===")
    print(f"  ordered-sequence mismatches: {order_mismatches}")
    print(f"  per-action index mismatches: {peraction_mismatches}")
    print(f"  python non-ascending states: {not_ascending}")
    print(f"  rust out-of-range indices:   {out_of_range}")
    ok = (order_mismatches == 0 and peraction_mismatches == 0
          and not_ascending == 0 and out_of_range == 0)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def _report_seq(g, phase, py_seq, rust_seq):
    if len(py_seq) != len(rust_seq):
        print(f"  [g{g} {phase.name}] length differs: py={len(py_seq)} rust={len(rust_seq)}")
        print(f"      py={py_seq[:12]}\n      rust={rust_seq[:12]}")
        return
    first = next(i for i in range(len(py_seq)) if py_seq[i] != rust_seq[i])
    print(f"  [g{g} {phase.name}] order differs at pos {first}: "
          f"py={py_seq[first]} rust={rust_seq[first]} "
          f"(py_set==rust_set: {set(py_seq) == set(rust_seq)})")


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    n = 2000
    for a in sys.argv[1:]:
        if a.startswith("--games="):
            n = int(a.split("=", 1)[1])
    ok = run_equiv(n_games=n, verbose=verbose)
    sys.exit(0 if ok else 1)
