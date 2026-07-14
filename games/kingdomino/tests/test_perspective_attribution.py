"""End-to-end perspective attribution in the native Rust batched self-play path.

The bug class we chased: for every recorded training example, the player whose
PERSPECTIVE the input is encoded from must be the SAME player whose final outcome
fills (own_score, opp_score, win_target, z). Non-alternating turn order and round
boundaries make this easy to get subtly wrong, and it silently poisons labels.

Static reading shows all three record sites (full-search, recorded fast-move,
exact-plan) bind one `actor = real_state.actor()` used for encode + policy frame +
outcome fill. This asserts it empirically on generated data, exercising all three
record sites at once (mixed full/fast moves + exact endgames).
"""
import math

import numpy as np
import pytest

from games.kingdomino.network import KingdominoNet
from games.kingdomino.self_play import SelfPlayConfig, play_selfplay_games_batched
from games.kingdomino.encoder import FLAT_LAYOUT

_af = FLAT_LAYOUT["actor_flag"]
ACTOR_FLAG_OFF = _af.start if isinstance(_af, slice) else int(np.ravel(_af)[0])


def _score_rule_win(own, opp):
    if own > opp:
        return 1.0
    if opp > own:
        return 0.0
    return 0.5


def _generate(n_games=12, seed_start=500000):
    net = KingdominoNet(channels=16, blocks=2, bilinear_dim=16, score_scale=160.0).eval()
    cfg = SelfPlayConfig()
    cfg.engine = "batched_open_loop"      # info-set-safe det=1 production path
    cfg.n_determinizations = 1
    cfg.n_simulations = 24
    cfg.batch_slots = 8
    cfg.leaf_batch = 4
    cfg.temp_moves = 6
    cfg.score_scale = 160.0
    cfg.exact_endgame_max_secs = 3.0      # exercise the exact-plan record site
    cfg.playout_cap_randomization = True  # mix full + fast moves
    cfg.full_search_fraction = 0.5
    cfg.record_fast_moves = True          # exercise the recorded-fast record site
    cfg.fast_move_sims = 6
    all_examples, all_scores, _stats = play_selfplay_games_batched(
        net, cfg, n_games=n_games, game_seed_start=seed_start)
    return all_examples, all_scores


def test_perspective_attribution_end_to_end():
    pytest.importorskip("kingdomino_rust")
    all_examples, all_scores = _generate()

    n_ex = n_actor0 = n_actor1 = 0
    for exs, (s0, s1) in zip(all_examples, all_scores):
        seen = {0: None, 1: None}
        for ex in exs:
            n_ex += 1
            flat = np.asarray(ex.flat, dtype=np.float32)
            own, opp, wt, z = (float(ex.own_score), float(ex.opp_score),
                               float(ex.win_target), float(ex.z))

            # A. every training example is encoded from the mover
            assert abs(float(flat[ACTOR_FLAG_OFF]) - 1.0) < 1e-3

            # B. targets are the true final scores in ONE player's frame
            if (own, opp) == (float(s0), float(s1)):
                actor = 0
                n_actor0 += 1
            elif (own, opp) == (float(s1), float(s0)):
                actor = 1
                n_actor1 += 1
            else:
                pytest.fail(f"(own,opp)=({own},{opp}) matches neither ({s0},{s1})")

            # C. win_target is consistent with the official outcome in this frame.
            # When the two final scores differ, the official cascade agrees with the
            # raw-score rule; on a score TIE the cascade (largest territory -> total
            # crowns) decides, so win_target may be a decisive 0.0/1.0, not 0.5.
            if own != opp:
                assert abs(wt - _score_rule_win(own, opp)) < 1e-6
            else:
                assert wt in (0.0, 0.5, 1.0)
            # D. z shares the frame: z == tanh((own-opp)/30)
            assert abs(z - math.tanh((own - opp) / 30.0)) < 1e-4

            if seen[actor] is None:
                seen[actor] = wt
            else:
                assert abs(seen[actor] - wt) < 1e-6  # same actor, same result

        # E. both actors present, win_targets complementary
        if seen[0] is not None and seen[1] is not None:
            assert abs((seen[0] + seen[1]) - 1.0) < 1e-6

    assert n_ex > 0
    assert n_actor0 > 0 and n_actor1 > 0  # both perspectives recorded


if __name__ == "__main__":
    test_perspective_attribution_end_to_end()
    print("PASS: perspective attribution correct end to end.")
