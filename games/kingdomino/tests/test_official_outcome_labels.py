"""Official-outcome cascade parity.

The self-play buffer labels (win_target) now route through the Rust official
outcome cascade (finalize_move / exact-endgame plan -> RustGameState::
official_outcome_i8), replacing the old score-only labeling that mislabeled every
tiebreak-decided game as a draw (the run10 label bug).

This cross-checks that cascade against game.py's determine_winner -- the single
source of truth -- on real terminal states, and specifically REQUIRES coverage of
games decided by the tiebreak (equal total score, decided by largest territory or
total crowns), so the cascade path is genuinely exercised rather than trivially
agreeing on score-decided games.
"""
import random

import pytest

from games.kingdomino.game import GameState, Phase, determine_winner
from games.kingdomino.endgame_solver import _rust_state_from_python

kr = pytest.importorskip("kingdomino_rust")


def _play_to_end(seed: int) -> GameState:
    st = GameState.new(seed=seed)
    rng = random.Random(seed * 7 + 3)
    while st.phase != Phase.GAME_OVER:
        st = st.step(rng.choice(st.legal_actions()))
    return st


def _rust_outcome(st: GameState) -> int:
    """Rust official outcome in P0 frame: +1 P0 / -1 P1 / 0 draw."""
    rs = _rust_state_from_python(st)
    assert rs is not None, "could not build Rust state from terminal Python state"
    return int(kr.SearchEngine(rs).official_outcome())


def _py_outcome(st: GameState) -> int:
    w = determine_winner(st)
    return 0 if w is None else (1 if w == 0 else -1)


def test_rust_cascade_matches_determine_winner_incl_tiebreaks():
    n = 0
    n_score_tie = 0
    n_tiebreak_decided = 0
    n_true_draw = 0
    for seed in range(600):
        st = _play_to_end(seed)
        rust_out = _rust_outcome(st)
        py_out = _py_outcome(st)
        assert rust_out == py_out, (
            f"seed {seed}: rust official_outcome {rust_out} != "
            f"determine_winner {py_out} (scores {st.scores()})"
        )
        n += 1
        s0, s1 = st.scores()
        if s0 == s1:
            n_score_tie += 1
            if rust_out != 0:
                n_tiebreak_decided += 1  # equal score, decided by the cascade
            else:
                n_true_draw += 1

    assert n > 100, f"too few completed games ({n})"
    # The whole point of the fix: equal-score games decided by the tiebreak must
    # occur and must be labeled decisively (not 0.5). If none appear the cascade
    # path is untested -> fail loudly rather than pass vacuously.
    assert n_tiebreak_decided > 0, (
        f"no score-tied-but-tiebreak-decided game in {n} games "
        f"(score ties: {n_score_tie}); cascade path not exercised"
    )
    print(f"games={n} score_ties={n_score_tie} "
          f"tiebreak_decided={n_tiebreak_decided} true_draws={n_true_draw}")


if __name__ == "__main__":
    test_rust_cascade_matches_determine_winner_incl_tiebreaks()
    print("PASS: Rust official-outcome cascade matches determine_winner, tiebreaks covered.")
