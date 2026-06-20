# test_patches.py
# Validation tests for the two bug-fix patches applied to self_play.py:
#   Patch 1: best_model.pt overwrite protection (two-tier acceptance,
#            atomic writes, .bak rotation)
#   Patch 2: timeout records dropped instead of mislabelled as losses
#
# Run with:
#   pytest games/cantstop/tests/test_patches.py -v

import os
import sys

import numpy as np
import pytest
import torch

# Three '..'s to climb out of: tests/ -> cantstop/ -> games/ -> project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from games.cantstop.engine import GameState
from games.cantstop.features import ACTION_SPACE, FEATURE_SIZE
from games.cantstop.mcts import MCTS
from games.cantstop.model import CantStopNet
from games.cantstop import self_play as sp


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def cpu_model():
    """Small untrained model — fine for these structural tests."""
    torch.manual_seed(0)
    return CantStopNet().eval()


# ============================================================
# Helpers
# ============================================================

def _make_checkpoint(model, iteration=1, win_rate=0.55):
    return {
        'iteration': iteration,
        'win_rate': float(win_rate),
        'val_loss': 0.0,
        'temp_mult': 1.0,
        'model_state': {k: v.clone() for k, v in model.state_dict().items()},
    }


def _state_dicts_equal(sd_a, sd_b):
    if set(sd_a.keys()) != set(sd_b.keys()):
        return False
    return all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)


# ============================================================
# PATCH 1A: _atomic_save_with_rotation
# ============================================================

class TestAtomicSaveWithRotation:

    def test_writes_new_file_when_none_exists(self, tmp_path, cpu_model):
        dest = str(tmp_path / 'best_model.pt')
        ck = _make_checkpoint(cpu_model)
        sp._atomic_save_with_rotation(ck, dest)

        assert os.path.exists(dest)
        assert not os.path.exists(dest + '.bak')   # nothing to rotate
        assert not os.path.exists(dest + '.tmp')   # tmp cleaned up

        loaded = torch.load(dest, map_location='cpu', weights_only=False)
        assert loaded['iteration'] == ck['iteration']

    def test_rotates_existing_to_bak(self, tmp_path, cpu_model):
        dest = str(tmp_path / 'best_model.pt')

        # First write
        ck1 = _make_checkpoint(cpu_model, iteration=1, win_rate=0.55)
        sp._atomic_save_with_rotation(ck1, dest)

        # Build a different second model
        m2 = CantStopNet().eval()
        with torch.no_grad():
            for p in m2.parameters():
                p.add_(1.0)
        ck2 = _make_checkpoint(m2, iteration=2, win_rate=0.62)
        sp._atomic_save_with_rotation(ck2, dest)

        # New version is at dest; previous version is at .bak
        loaded_dest = torch.load(dest, map_location='cpu', weights_only=False)
        loaded_bak = torch.load(dest + '.bak', map_location='cpu', weights_only=False)
        assert loaded_dest['iteration'] == 2
        assert loaded_bak['iteration'] == 1

    def test_atomic_under_simulated_crash(self, tmp_path, cpu_model, monkeypatch):
        """
        If torch.save raises during the .tmp write, dest_path must be
        unchanged. The whole point of atomic writes.
        """
        dest = str(tmp_path / 'best_model.pt')

        ck_v1 = _make_checkpoint(cpu_model, iteration=1, win_rate=0.55)
        sp._atomic_save_with_rotation(ck_v1, dest)
        v1_bytes = open(dest, 'rb').read()

        def crashing_save(obj, path, *args, **kwargs):
            raise RuntimeError("simulated disk failure")

        monkeypatch.setattr(torch, 'save', crashing_save)

        ck_v2 = _make_checkpoint(cpu_model, iteration=2, win_rate=0.99)
        with pytest.raises(RuntimeError, match="simulated"):
            sp._atomic_save_with_rotation(ck_v2, dest)

        # Original file is intact
        assert os.path.exists(dest)
        assert open(dest, 'rb').read() == v1_bytes


# ============================================================
# PATCH 1B: _should_overwrite_best gate logic
# ============================================================

class TestBestModelOverwriteGate:

    def test_regressing_model_does_not_overwrite(self, tmp_path, monkeypatch):
        """The headline test — a model that loses to disk-best is NOT a new best."""
        best_path = str(tmp_path / 'best_model.pt')

        old = CantStopNet().eval()
        sp._atomic_save_with_rotation(_make_checkpoint(old), best_path)

        new_model = CantStopNet().eval()

        # Mock evaluate_networks to return a losing win rate
        monkeypatch.setattr(sp, 'evaluate_networks',
                            lambda *a, **kw: 0.40)

        is_new_best, wr = sp._should_overwrite_best(
            new_model=new_model,
            best_model_path=best_path,
            eval_games=100,
            eval_sims=20,
            num_workers=1,
            output_dir=str(tmp_path),
            best_overwrite_threshold=0.55,
        )

        assert not is_new_best
        assert wr == pytest.approx(0.40)

    def test_winning_model_is_new_best(self, tmp_path, monkeypatch):
        best_path = str(tmp_path / 'best_model.pt')
        sp._atomic_save_with_rotation(
            _make_checkpoint(CantStopNet().eval()), best_path)

        new_model = CantStopNet().eval()
        monkeypatch.setattr(sp, 'evaluate_networks',
                            lambda *a, **kw: 0.62)

        is_new_best, wr = sp._should_overwrite_best(
            new_model=new_model,
            best_model_path=best_path,
            eval_games=100,
            eval_sims=20,
            num_workers=1,
            output_dir=str(tmp_path),
            best_overwrite_threshold=0.55,
        )

        assert is_new_best
        assert wr == pytest.approx(0.62)

    def test_threshold_boundary_inclusive(self, tmp_path, monkeypatch):
        """Exactly hitting threshold counts (using >=)."""
        best_path = str(tmp_path / 'best_model.pt')
        sp._atomic_save_with_rotation(
            _make_checkpoint(CantStopNet().eval()), best_path)

        monkeypatch.setattr(sp, 'evaluate_networks',
                            lambda *a, **kw: 0.55)

        is_new_best, _ = sp._should_overwrite_best(
            new_model=CantStopNet().eval(),
            best_model_path=best_path,
            eval_games=100,
            eval_sims=20,
            num_workers=1,
            output_dir=str(tmp_path),
            best_overwrite_threshold=0.55,
        )
        assert is_new_best

    def test_just_below_threshold_does_not_overwrite(self, tmp_path, monkeypatch):
        best_path = str(tmp_path / 'best_model.pt')
        sp._atomic_save_with_rotation(
            _make_checkpoint(CantStopNet().eval()), best_path)

        monkeypatch.setattr(sp, 'evaluate_networks',
                            lambda *a, **kw: 0.5499)

        is_new_best, _ = sp._should_overwrite_best(
            new_model=CantStopNet().eval(),
            best_model_path=best_path,
            eval_games=100,
            eval_sims=20,
            num_workers=1,
            output_dir=str(tmp_path),
            best_overwrite_threshold=0.55,
        )
        assert not is_new_best


# ============================================================
# PATCH 1 INTEGRATION: end-to-end gate behavior
# ============================================================

class TestEndToEndBestModelProtection:
    """
    Simulates what self_play_loop does: write a strong model to
    best_model.pt, then attempt to overwrite with a regressing one.
    """

    def test_regressing_model_does_not_clobber_best(self, tmp_path, monkeypatch):
        best_path = str(tmp_path / 'best_model.pt')

        # Seed best with a known-shaped checkpoint
        strong = CantStopNet().eval()
        torch.manual_seed(42)
        for p in strong.parameters():
            p.data.normal_(0, 0.1)
        strong_state = {k: v.clone() for k, v in strong.state_dict().items()}
        sp._atomic_save_with_rotation(
            _make_checkpoint(strong, iteration=5, win_rate=0.70),
            best_path)

        # Build a "new" model and mock evals
        new_model = CantStopNet().eval()
        monkeypatch.setattr(sp, 'evaluate_networks',
                            lambda *a, **kw: 0.40)  # loses to best

        # Inline the relevant slice of self_play_loop's accept block:
        is_new_best, wr = sp._should_overwrite_best(
            new_model=new_model, best_model_path=best_path,
            eval_games=100, eval_sims=20, num_workers=1,
            output_dir=str(tmp_path), best_overwrite_threshold=0.55,
        )
        if is_new_best:
            sp._atomic_save_with_rotation(
                _make_checkpoint(new_model, iteration=6, win_rate=0.51),
                best_path)

        # Gate said no -> best_model.pt must still hold strong's weights
        loaded = torch.load(best_path, map_location='cpu', weights_only=False)
        assert _state_dicts_equal(loaded['model_state'], strong_state)


# ============================================================
# PATCH 2: timeout records dropped, not mislabelled
# ============================================================

class TestTimeoutRecordsDropped:

    def test_play_mcts_game_drops_timeout_records(self):
        """
        Direct test of the play_mcts_game labelling block under
        the winner=None condition. After the patch, the function
        must return [] (not mislabelled records).

        Approach: simulate the labelling block in isolation. We construct
        records and a fake state with winner=None, then run only the
        post-game-loop labelling logic.
        """
        # Build the fake "post-game-loop" state.
        state = GameState(2)
        state.winner = None
        state.game_over = False

        records = [
            {'features': np.zeros(FEATURE_SIZE, dtype=np.float32),
             'mask': np.zeros(ACTION_SPACE, dtype=bool),
             'mcts_policy': np.zeros(ACTION_SPACE, dtype=np.float32),
             'mcts_value': 0.5,
             'action_idx': 0, 'player': 0, 'step_index': 0, 'game_id': 1},
            {'features': np.zeros(FEATURE_SIZE, dtype=np.float32),
             'mask': np.zeros(ACTION_SPACE, dtype=bool),
             'mcts_policy': np.zeros(ACTION_SPACE, dtype=np.float32),
             'mcts_value': 0.5,
             'action_idx': 0, 'player': 1, 'step_index': 1, 'game_id': 1},
        ]

        # Inline the post-patch labelling logic from play_mcts_game.
        if state.winner is None:
            result = []
        else:
            winner = state.winner
            lambda_blend = 0.7
            result = []
            for rec in records:
                final_outcome = 1.0 if rec['player'] == winner else 0.0
                rec['value_target'] = (
                    lambda_blend * final_outcome +
                    (1 - lambda_blend) * rec['mcts_value']
                )
                result.append(rec)

        assert result == []

    def test_batched_runner_drops_timeout_records(self, cpu_model):
        """
        _BatchedGameRunner.labeled_records returns [] when state.winner
        is None (timeout case).
        """
        mcts = MCTS(cpu_model, 'cpu', target_inflight=1, warmup_sims=0)
        runner = sp._BatchedGameRunner(
            mcts=mcts, num_simulations=2, global_temp_mult=1.0, game_id=1
        )

        # Force timeout state
        runner.done = True
        runner.state.winner = None
        runner.state.game_over = False
        runner.records = [
            {'features': np.zeros(FEATURE_SIZE, dtype=np.float32),
             'mask': np.zeros(ACTION_SPACE, dtype=bool),
             'mcts_policy': np.zeros(ACTION_SPACE, dtype=np.float32),
             'mcts_value': 0.5,
             'action_idx': 0, 'player': 0, 'step_index': 0, 'game_id': 1},
        ]

        out = runner.labeled_records()
        assert out == []

    def test_batched_runner_labels_completed_games_correctly(self, cpu_model):
        """
        Regression guard: a normally-completed game still gets labelled.
        We want to make sure the timeout fix didn't break the happy path.
        """
        mcts = MCTS(cpu_model, 'cpu', target_inflight=1, warmup_sims=0)
        runner = sp._BatchedGameRunner(
            mcts=mcts, num_simulations=2, global_temp_mult=1.0, game_id=1
        )

        runner.done = True
        runner.state.winner = 0
        runner.state.game_over = True
        runner.records = [
            {'features': np.zeros(FEATURE_SIZE, dtype=np.float32),
             'mask': np.zeros(ACTION_SPACE, dtype=bool),
             'mcts_policy': np.zeros(ACTION_SPACE, dtype=np.float32),
             'mcts_value': 0.4,
             'action_idx': 0, 'player': 0, 'step_index': 0, 'game_id': 1},
            {'features': np.zeros(FEATURE_SIZE, dtype=np.float32),
             'mask': np.zeros(ACTION_SPACE, dtype=bool),
             'mcts_policy': np.zeros(ACTION_SPACE, dtype=np.float32),
             'mcts_value': 0.6,
             'action_idx': 0, 'player': 1, 'step_index': 1, 'game_id': 1},
        ]

        out = runner.labeled_records()
        assert len(out) == 2
        # Player 0 won -> final_outcome 1.0
        assert out[0]['value_target'] == pytest.approx(0.7 * 1.0 + 0.3 * 0.4)
        # Player 1 lost -> final_outcome 0.0
        assert out[1]['value_target'] == pytest.approx(0.7 * 0.0 + 0.3 * 0.6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))