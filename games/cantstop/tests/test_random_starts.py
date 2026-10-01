"""Random-prefix starts and variant emphasis for the Phase 4 loop."""

import json
import random

import pytest
import torch

from games.cantstop import phase4
from games.cantstop.engine import COLUMNS, Phase, RuleSet
from games.cantstop.random_starts import pick_random_starts, random_start
from games.cantstop.rust_pool import game_seeds
from games.cantstop.rust_pool_equiv import MOCKS
from games.cantstop.schedule import RowSchedule, rules_key
from games.cantstop.portable_rng import PortableRng
from games.cantstop.train import _starts, generate

pytest.importorskip("cantstop_rust", reason="run maturin develop first")

R2, R3 = RuleSet.make(2), RuleSet.make(3, extended=True, blocking=True)


def test_random_start_is_a_live_turn_start_and_reproducible():
    boards = [random_start(r, s) for r in (R2, R3) for s in range(200)]
    for b in boards:
        assert not b.game_over and b.phase == Phase.AWAIT_ROLL
        assert not b.runners and b.dice is None
        for p in range(b.rules.num_players):
            assert len(b.claimed_columns(p)) < b.rules.columns_to_win
    assert repr(random_start(R3, 7)) == repr(random_start(R3, 7))
    # Varied: many distinct boards, some with claimed columns.
    assert len({repr(b) for b in boards}) > 350
    assert any(any(b.claimed_by[c] is not None for c in COLUMNS) for b in boards)


def test_pick_fraction_and_off_switch():
    seeds = game_seeds(PortableRng(3), 4000)
    assert pick_random_starts(seeds, 0.0) == [False] * 4000
    share = sum(pick_random_starts(seeds, 0.25)) / 4000
    assert 0.22 < share < 0.28
    with pytest.raises(ValueError):
        pick_random_starts(seeds, 1.5)


def test_fraction_zero_reproduces_the_old_generation_exactly():
    a = generate([R2], {R2: 6}, MOCKS["hashed"], random.Random(4))
    b = generate([R2], {R2: 6}, MOCKS["hashed"], random.Random(4),
                 random_start_fraction=0.0)
    assert [(r.winner, r.turns, r.features.tobytes()) for r in a] == \
           [(r.winner, r.turns, r.features.tobytes()) for r in b]
    assert _starts([R2], [1], 0.0, 8) is None


def test_random_starts_change_only_the_picked_games():
    rng_a, rng_b = random.Random(9), random.Random(9)
    plain = generate([R2], {R2: 40}, MOCKS["hashed"], rng_a)
    mixed = generate([R2], {R2: 40}, MOCKS["hashed"], rng_b,
                     random_start_fraction=0.5)
    seeds = game_seeds(random.Random(9), 40)
    picks = pick_random_starts(seeds, 0.5)
    assert 5 < sum(picks) < 35
    for pick, p, m in zip(picks, plain, mixed):
        same = (p.turns, p.features.tobytes()) == (m.turns, m.features.tobytes())
        assert same != pick          # unpicked identical, picked different


def test_variant_weights_scale_the_row_target():
    plain = RowSchedule((R2, R3), 1000).games()
    heavy = RowSchedule((R2, R3), 1000, weights={rules_key(R3): 2.0}).games()
    assert heavy[R2] == plain[R2]
    assert abs(heavy[R3] - 2 * plain[R3]) <= 1
    with pytest.raises(ValueError):
        RowSchedule((R2,), 1000, weights={rules_key(R3): 2.0})
    with pytest.raises(ValueError):
        RowSchedule((R2,), 1000, weights={rules_key(R2): 0.0})
    assert phase4.parse_weights(["3:4:b=2", "2:3=1.5"]) == {
        (3, 4, True): 2.0, (2, 3, False): 1.5}


KW = dict(rule_sets=(R2, R3), rows_per_variant=30, hidden=(16, 16),
          batch_size=32, replay_window=2, eval_every=0, seed=5, device="cpu",
          probes_per_variant=3)


def test_loop_runs_with_emphasis_and_random_starts_and_resumes(tmp_path):
    phase4.run(tmp_path, iterations=1, **KW)
    phase4.run(tmp_path, iterations=2, resume=True,
               variant_weights={rules_key(R3): 2.0},
               random_start_fraction=0.5, **KW)
    rows = [json.loads(l) for l in open(tmp_path / "run.jsonl")]
    assert rows[0]["random_start_fraction"] == 0.0
    assert rows[1]["random_start_fraction"] == 0.5
    assert rows[1]["variant_weights"] == {str(rules_key(R3)): 2.0}
    assert rows[1]["games"][str(R3)] > rows[0]["games"][str(R3)]
    assert torch.load(tmp_path / "state.pt", weights_only=False)[
        "config"]["random_start_fraction"] == 0.5


MIX = dict(variant_weights={rules_key(R3): 2.0}, random_start_fraction=0.5,
           random_start_turns=4)


def test_resume_without_mix_flags_keeps_the_saved_mix(tmp_path):
    """Review finding: omitted flags used to reset the mix to defaults.
    Uninterrupted == resumed with the flags omitted."""
    a, b = tmp_path / "a", tmp_path / "b"
    phase4.run(a, iterations=2, **MIX, **KW)
    phase4.run(b, iterations=1, **MIX, **KW)
    phase4.run(b, iterations=2, resume=True, **KW)
    sa = torch.load(a / "iter_0002.pt", weights_only=False)["state_dict"]
    sb = torch.load(b / "iter_0002.pt", weights_only=False)["state_dict"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    strip = lambda r: {k: v for k, v in r.items() if not k.endswith("seconds")}
    la = [strip(json.loads(l)) for l in open(a / "run.jsonl")]
    lb = [strip(json.loads(l)) for l in open(b / "run.jsonl")]
    assert la == lb


def test_resume_applies_an_explicit_override_only(tmp_path):
    phase4.run(tmp_path, iterations=1, **MIX, **KW)
    phase4.run(tmp_path, iterations=2, resume=True, random_start_fraction=0.0,
               **KW)
    config = torch.load(tmp_path / "state.pt", weights_only=False)["config"]
    assert config["random_start_fraction"] == 0.0
    assert config["random_start_turns"] == 4
    assert config["variant_weights"] == {str(rules_key(R3)): 2.0}
