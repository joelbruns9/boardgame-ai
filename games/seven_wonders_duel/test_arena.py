"""Tests for the standalone head-to-head arena."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from .arena import (
    ARENA_SEED_OFFSET,
    GATE_FAILED_EXIT_CODE,
    ControlMaskedModel,
    GameResult,
    _CONTROL_TABLEAU_COLUMNS,
    _CONTROL_VALID_COLUMN,
    _T_GLOBAL,
    _T_TABLEAU,
    _linear_fit,
    _step_games,
    _verdict,
    control_arm,
    format_summary,
    load_side,
    main,
    pair_scores,
    play_pairs,
    run,
    sims_for_budget,
)
from .rust_bridge import rust_games_for_self_play
from .train import build_model, make_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- pair scoring -----------------------------------------------------------


def _game(seed: int, a_seat: int, winner: int | None) -> GameResult:
    return GameResult(
        seed=seed,
        first_player=0,
        a_seat=a_seat,
        winner=winner,
        scores=(10, 9),
        victory_type="civilian",
        actions=40,
    )


def test_pair_scores_reads_a_pair_as_one_observation():
    results = [
        # A wins both legs of deal 1.
        _game(1, 0, 0),
        _game(1, 1, 1),
        # Split on deal 2: each side wins the seat it held.
        _game(2, 0, 0),
        _game(2, 1, 0),
        # A loses both legs of deal 3.
        _game(3, 0, 1),
        _game(3, 1, 0),
    ]
    assert pair_scores(results) == [1.0, 0.5, 0.0]


def test_pair_scores_counts_a_draw_as_half_a_leg():
    # One drawn leg plus one won leg is 1.5 points, which is a pair win.
    assert pair_scores([_game(1, 0, None), _game(1, 1, 1)]) == [1.0]
    # Two drawn legs split the pair.
    assert pair_scores([_game(1, 0, None), _game(1, 1, None)]) == [0.5]


def test_pair_scores_refuses_an_incomplete_pair():
    with pytest.raises(ValueError, match="complete seat pairs"):
        pair_scores([_game(1, 0, 0)])


def test_pair_scores_refuses_legs_that_are_not_a_seat_swap():
    """A 'pair' of two games in the SAME seat cancels neither deal nor seat.

    Silently accepting it would report per-game binomial noise as a paired
    observation, which is the whole reason the interval is computed over pairs.
    """

    with pytest.raises(ValueError, match="not a seat swap"):
        pair_scores([_game(1, 0, 0), _game(1, 0, 1)])
    with pytest.raises(ValueError, match="not a seat swap"):
        pair_scores([_game(1, 0, 0), _game(2, 1, 1)])


# --- control-feature arms ---------------------------------------------------


def test_control_arm_reads_the_stamp_and_defaults_off():
    assert control_arm({"control_features": "on"}) == "on"
    assert control_arm({"control_features": "off"}) == "off"
    # Unstamped predates the channels; its migrated input columns are zeroed, so
    # "off" describes what the weights do rather than guessing.
    assert control_arm({}) == "off"
    assert control_arm({"control_features": "unknown"}) == "off"


def _batch(rows: int = 3, tokens: int = 4, width: int = 133):
    features = torch.arange(
        rows * tokens * width, dtype=torch.float32
    ).reshape(rows, tokens, width) + 1.0
    type_ids = torch.full((rows, tokens), _T_TABLEAU, dtype=torch.long)
    type_ids[:, 0] = _T_GLOBAL
    # A pool token, whose columns 26..31 are ordinary pool features.
    pool_type = max(_T_GLOBAL, _T_TABLEAU) + 5
    type_ids[:, -1] = pool_type
    return {"features": features, "type_ids": type_ids}, pool_type


def test_control_mask_zeroes_exactly_the_control_columns():
    captured = {}

    def inner(batch):
        captured["features"] = batch["features"]
        return {"policy": torch.zeros(1), "value": torch.zeros(1)}

    batch, pool_type = _batch()
    original = batch["features"].clone()
    ControlMaskedModel(inner)(batch)
    masked = captured["features"]
    type_ids = batch["type_ids"]

    tableau = type_ids == _T_TABLEAU
    for column in _CONTROL_TABLEAU_COLUMNS:
        assert torch.all(masked[..., column][tableau] == 0.0)
        # ... and nowhere else: the same column on a pool token is a different
        # feature entirely.
        other = ~tableau
        assert torch.equal(
            masked[..., column][other], original[..., column][other]
        )
    is_global = type_ids == _T_GLOBAL
    assert torch.all(masked[..., _CONTROL_VALID_COLUMN][is_global] == 0.0)
    assert torch.equal(
        masked[..., _CONTROL_VALID_COLUMN][~is_global],
        original[..., _CONTROL_VALID_COLUMN][~is_global],
    )
    assert (type_ids == pool_type).any()


def test_control_mask_leaves_every_other_column_alone():
    captured = {}

    def inner(batch):
        captured["features"] = batch["features"]
        return {}

    batch, _ = _batch()
    original = batch["features"].clone()
    ControlMaskedModel(inner)(batch)
    masked = captured["features"]
    touched = set(_CONTROL_TABLEAU_COLUMNS) | {_CONTROL_VALID_COLUMN}
    for column in range(original.shape[-1]):
        if column in touched:
            continue
        assert torch.equal(masked[..., column], original[..., column])


def test_control_mask_does_not_mutate_the_adapters_buffer():
    """The packed batch belongs to the adapter; masking must not edit it."""

    batch, _ = _batch()
    original = batch["features"].clone()
    ControlMaskedModel(lambda payload: {})(batch)
    assert torch.equal(batch["features"], original)


def test_control_mask_forwards_action_residual():
    class Inner:
        action_residual = True

        def __call__(self, batch):
            return {}

    assert ControlMaskedModel(Inner()).action_residual is True
    assert ControlMaskedModel(lambda batch: {}).action_residual is False


# --- cost fitting -----------------------------------------------------------


def test_linear_fit_recovers_a_known_line():
    intercept, slope = _linear_fit([1.0, 2.0, 3.0, 4.0], [3.0, 5.0, 7.0, 9.0])
    assert intercept == pytest.approx(1.0)
    assert slope == pytest.approx(2.0)


def test_linear_fit_refuses_a_degenerate_sample():
    with pytest.raises(ValueError, match="at least two"):
        _linear_fit([8.0], [1.0])
    with pytest.raises(ValueError, match="DISTINCT"):
        _linear_fit([8.0, 8.0], [1.0, 2.0])


def test_sims_for_budget_inverts_the_fit():
    fit = {"intercept": 0.010, "slope": 0.001}
    # 0.010 + 0.001 * 54 = 0.064
    assert sims_for_budget(fit, 0.064, floor=1, cap=10_000) == 54


def test_sims_for_budget_clamps_rather_than_extrapolating_through_noise():
    """A flat or negative fitted slope did not resolve the per-sim cost.

    Solving through it hands one side an unbounded (or negative) budget off a
    measurement that says nothing about simulation cost.
    """

    assert sims_for_budget({"intercept": 0.0, "slope": 0.0}, 1.0, floor=4, cap=99) == 99
    assert sims_for_budget({"intercept": 0.0, "slope": -1e-6}, 1.0, floor=4, cap=99) == 99
    assert sims_for_budget({"intercept": 0.0, "slope": float("nan")}, 1.0, floor=4, cap=99) == 99
    # A budget below the fixed overhead solves negative; the floor holds.
    assert sims_for_budget({"intercept": 1.0, "slope": 0.001}, 0.1, floor=4, cap=99) == 4
    assert sims_for_budget({"intercept": 0.0, "slope": 1e-12}, 1.0, floor=4, cap=99) == 99


# --- stepping ---------------------------------------------------------------


class _StubSide:
    """Enough of a `Side` for `_step_games`: a budget, an adapter, and counters."""

    def __init__(self, label: str, sims: int):
        self.label = label
        self.sims = sims
        self.adapter = f"adapter-{label}"
        self.search_seconds = 0.0
        self.moves = 0


class _StubSearcher:
    """Applies the FIRST legal action, and records what each call was asked for."""

    def __init__(self):
        self.calls: list[dict] = []

    def search_many_flat_net(self, adapter, batch, seeds, *args, **kwargs):
        self.calls.append(
            {
                "adapter": adapter,
                "games": len(batch),
                "sims": args[2],
                "seeds": list(seeds),
            }
        )
        results = []
        for game in batch:
            legal = game.legal_action_indices()
            policy = [0.0] * len(legal)
            # argmax at index 0, but a LATER index reported as `action`, so a
            # caller that plays the Gumbel action is visible.
            policy[0] = 1.0
            results.append({"action": legal[-1], "policy": policy})
        return results


def _step_kwargs(**overrides):
    kwargs = dict(
        batch_cap=16,
        leaf_batch=1,
        top_k=4,
        puct_root=False,
        force_root_chance=True,
        age_deal_samples=0,
        max_moves=512,
    )
    kwargs.update(overrides)
    return kwargs


def test_step_games_gives_each_seat_its_own_budget_and_adapter():
    """Per-side simulation counts are the whole point of wall-clock parity.

    The fused promotion gate handed both sides `gate_sims`, so a wider net
    "thinking as long" as a narrow one was in fact doing strictly more work.
    """

    searcher = _StubSearcher()
    a, b = _StubSide("a", 7), _StubSide("b", 33)
    records = _step_games(
        searcher,
        rust_games_for_self_play([4242], [0]),
        [4242],
        (a, b),
        **_step_kwargs(),
    )

    assert len(records) == 1
    assert records[0]["actions"] > 0
    assert searcher.calls, "expected at least one search"
    for call in searcher.calls:
        if call["adapter"] == "adapter-a":
            assert call["sims"] == 7
        else:
            assert call["adapter"] == "adapter-b"
            assert call["sims"] == 33
    # Every move is attributed to exactly one side.
    assert a.moves + b.moves == records[0]["actions"]
    assert a.moves > 0 and b.moves > 0
    assert a.search_seconds >= 0.0 and b.search_seconds >= 0.0


def test_step_games_plays_the_improved_policy_argmax():
    """Never the Gumbel-perturbed action: at a small budget it is a SAMPLE."""

    applied: list[int] = []

    class _Recording(_StubSearcher):
        def search_many_flat_net(self, adapter, batch, seeds, *args, **kwargs):
            for game in batch:
                applied.append(game.legal_action_indices()[0])
            return super().search_many_flat_net(adapter, batch, seeds, *args, **kwargs)

    games = rust_games_for_self_play([909], [0])
    first_legal = games[0].legal_action_indices()[0]
    _step_games(_Recording(), games, [909], (_StubSide("a", 2), _StubSide("b", 2)),
                **_step_kwargs())
    assert applied[0] == first_legal


def test_step_games_seeds_differ_by_ply_and_by_seat():
    searcher = _StubSearcher()
    _step_games(
        searcher,
        rust_games_for_self_play([11], [0]),
        [11],
        (_StubSide("a", 2), _StubSide("b", 2)),
        **_step_kwargs(),
    )
    seeds = [call["seeds"][0] for call in searcher.calls]
    assert len(seeds) == len(set(seeds)), "search seeds repeated across plies/seats"


def test_step_games_refuses_a_game_that_will_not_end():
    class _NeverEnds:
        actor = 0

        @staticmethod
        def legal_action_indices():
            return [0]

        def apply_index(self, index):
            return None

        @staticmethod
        def is_complete():
            return False

    with pytest.raises(RuntimeError, match="exceeded 3 moves"):
        _step_games(
            _StubSearcher(),
            [_NeverEnds()],
            [1],
            (_StubSide("a", 2), _StubSide("b", 2)),
            **_step_kwargs(max_moves=3),
        )


# --- verdict ----------------------------------------------------------------


def _report(rate: float, lower: float, upper: float, ratio: float = 1.0) -> dict:
    return {
        "a_score_rate": rate,
        "pairs": 100,
        "wilson": {"lower": lower, "upper": upper, "z": 1.96},
        "parity": {
            "measured": {
                "ratio_a_over_b": ratio,
                "within_tolerance": abs(ratio - 1.0) <= 0.15,
            }
        },
    }


def test_verdict_names_which_side_is_stronger():
    assert "A is stronger" in _verdict(_report(0.60, 0.53, 0.67))
    assert "A is weaker" in _verdict(_report(0.40, 0.33, 0.47))
    assert "no strength difference resolved" in _verdict(_report(0.51, 0.44, 0.58))


def test_verdict_says_so_when_the_sides_did_not_play_at_equal_wall_clock():
    """A strength claim at unequal compute is a claim about compute."""

    verdict = _verdict(_report(0.60, 0.53, 0.67, ratio=1.9))
    assert "A is stronger" in verdict
    assert "did NOT play at equal wall-clock" in verdict
    assert "1.90" in verdict


# --- end to end -------------------------------------------------------------


def _write_checkpoint(path: Path, *, bias: list[float], seed: int) -> Path:
    """A tiny, current-schema checkpoint with a pinned value head."""

    torch.manual_seed(seed)
    model = build_model("transformer", 32, 1)
    model.heads.value.bias.data.copy_(torch.tensor(bias))
    payload = make_checkpoint(model, {"d_model": 32, "layers": 1, "iteration": seed})
    torch.save(payload, path)
    return path


@pytest.fixture(scope="module")
def tiny_checkpoints(tmp_path_factory):
    directory = tmp_path_factory.mktemp("arena_checkpoints")
    return (
        _write_checkpoint(directory / "a.pt", bias=[1.0, 0.0, -1.0], seed=1),
        _write_checkpoint(directory / "b.pt", bias=[-1.0, 0.0, 1.0], seed=2),
    )


def test_load_side_rebuilds_from_the_checkpoints_own_config(tiny_checkpoints):
    a_path, _ = tiny_checkpoints
    side = load_side("a", a_path, device="cpu", precision="fp32", batch_cap=16)
    assert side.architecture["d_model"] == 32
    assert side.architecture["layers"] == 1
    assert side.control_arm in ("on", "off")
    assert side.migration is None
    assert len(side.sha256) == 64
    described = side.describe()
    assert described["migrated"] is False
    assert described["source"].endswith("a.pt")


def test_load_side_refuses_an_older_schema_without_migrate(tmp_path):
    """A migrated model is not the model that was trained.

    Playing one measures the migration as much as the checkpoint, so it has to
    be asked for.
    """

    path = _write_checkpoint(tmp_path / "stale.pt", bias=[0.0, 0.0, 0.0], seed=3)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["encoder_signature"] = "0" * 64
    torch.save(payload, path)

    with pytest.raises(ValueError, match="--migrate"):
        load_side("a", path, device="cpu", precision="fp32", batch_cap=16)

    side = load_side(
        "a", path, device="cpu", precision="fp32", batch_cap=16, migrate=True
    )
    assert side.migration is not None
    assert side.describe()["migrated"] is True


def test_load_side_masks_only_the_side_trained_without_control_inputs(
    tiny_checkpoints, tmp_path
):
    """The W3 arena in miniature: one arm per side, in one process.

    `set_control_features` is process-wide, so the encoder runs ON and the
    off-arm side is masked back to zeros -- which is what off-mode is defined as.
    """

    a_path, _ = tiny_checkpoints
    on_path = tmp_path / "on.pt"
    off_path = tmp_path / "off.pt"
    payload = torch.load(a_path, map_location="cpu", weights_only=False)
    payload["control_features"] = "on"
    torch.save(payload, on_path)
    payload["control_features"] = "off"
    torch.save(payload, off_path)

    on_side = load_side("a", on_path, device="cpu", precision="fp32", batch_cap=16)
    off_side = load_side("b", off_path, device="cpu", precision="fp32", batch_cap=16)
    assert on_side.control_arm == "on"
    assert off_side.control_arm == "off"
    # Loading masks nothing: whether the mask is needed depends on the OTHER
    # side, which is a fact only the arena has.
    assert not isinstance(off_side.evaluator.model, ControlMaskedModel)

    off_side.mask_control_features()
    assert isinstance(off_side.evaluator.model, ControlMaskedModel)
    inner = off_side.evaluator.model.model
    # Idempotent: a second call must not stack a mask on a mask.
    off_side.mask_control_features()
    assert off_side.evaluator.model.model is inner


@pytest.mark.slow
def test_arena_masks_only_the_off_arm_side(tiny_checkpoints, tmp_path):
    """A mixed-arm arena is what W3 needs, and it must run in one process."""

    a_path, b_path = tiny_checkpoints
    paths = {}
    for label, source, arm in (("on", a_path, "on"), ("off", b_path, "off")):
        payload = torch.load(source, map_location="cpu", weights_only=False)
        payload["control_features"] = arm
        paths[label] = tmp_path / f"{label}.pt"
        torch.save(payload, paths[label])

    report = run(
        paths["on"],
        paths["off"],
        pairs=1,
        device="cpu",
        sims=2,
        top_k=2,
        slots=1,
        batch_cap=16,
        age_deal_samples=0,
    )
    # The encoder emits the real channels, and the off side is masked back to
    # the zeros it was trained on.
    assert report["control_features_encoder"] == "on"
    assert report["a"]["control_arm"] == "on"
    assert report["b"]["control_arm"] == "off"


@pytest.mark.slow
def test_arena_leaves_both_sides_unmasked_when_neither_reads_control(
    tiny_checkpoints, tmp_path
):
    """With the encoder off the channels are already zero; masking is a copy."""

    a_path, b_path = tiny_checkpoints
    paths = []
    for index, source in enumerate((a_path, b_path)):
        payload = torch.load(source, map_location="cpu", weights_only=False)
        payload["control_features"] = "off"
        paths.append(tmp_path / f"off_{index}.pt")
        torch.save(payload, paths[-1])

    report = run(
        paths[0],
        paths[1],
        pairs=1,
        device="cpu",
        sims=2,
        top_k=2,
        slots=1,
        batch_cap=16,
        age_deal_samples=0,
    )
    assert report["control_features_encoder"] == "off"


@pytest.mark.slow
def test_arena_plays_a_real_match_and_reports_a_paired_interval(tiny_checkpoints):
    """End to end: two real checkpoints, real search, a real interval.

    Two simulations and two pairs -- enough to exercise every seam (loading,
    stepping, seat pairing, timing, the report) without turning the suite into
    an arena run.
    """

    a_path, b_path = tiny_checkpoints
    report = run(
        a_path,
        b_path,
        pairs=2,
        device="cpu",
        sims=2,
        top_k=2,
        slots=2,
        batch_cap=16,
        age_deal_samples=0,
    )

    assert report["pairs"] == 2
    assert report["games"] == 4
    assert 0.0 <= report["a_score_rate"] <= 1.0
    assert len(report["pair_scores"]) == 2
    assert all(score in (0.0, 0.5, 1.0) for score in report["pair_scores"])
    assert report["wilson"]["lower"] <= report["a_score_rate"] <= report["wilson"]["upper"]
    assert report["pair_wins"] + report["pair_splits"] + report["pair_losses"] == 2
    assert report["a"]["sims"] == report["b"]["sims"] == 2
    assert report["seed_offset"] == ARENA_SEED_OFFSET
    assert report["moves_per_game"] > 0
    # Both sides are timed, and every move belongs to exactly one of them.
    measured = report["parity"]["measured"]
    assert measured["a_moves"] > 0 and measured["b_moves"] > 0
    assert measured["a_seconds_per_move"] > 0
    assert measured["b_seconds_per_move"] > 0
    assert math.isfinite(measured["ratio_a_over_b"])
    # Serialisable, because the point of the tool is a file someone re-reads.
    json.dumps(report)
    assert "vs" in format_summary(report)


@pytest.mark.slow
def test_arena_pairs_the_same_deal_in_both_seats(tiny_checkpoints):
    """The pairing contract the interval depends on, checked on real games."""

    a_path, b_path = tiny_checkpoints
    a = load_side("a", a_path, device="cpu", precision="fp32", batch_cap=16)
    b = load_side("b", b_path, device="cpu", precision="fp32", batch_cap=16)
    a.sims = b.sims = 2
    results = play_pairs(
        a,
        b,
        pairs=2,
        seed=0,
        slots=2,
        batch_cap=16,
        leaf_batch=1,
        top_k=2,
        puct_root=False,
        force_root_chance=True,
        age_deal_samples=0,
        max_moves=512,
    )
    assert len(results) == 4
    for index in range(0, 4, 2):
        first, second = results[index], results[index + 1]
        assert first.seed == second.seed
        assert first.first_player == second.first_player
        assert {first.a_seat, second.a_seat} == {0, 1}
    # `pair_scores` accepts what `play_pairs` produces -- the two halves of the
    # contract meet here rather than in a caller.
    assert len(pair_scores(results)) == 2


@pytest.mark.slow
def test_manual_budgets_give_each_side_a_different_simulation_count(tiny_checkpoints):
    a_path, b_path = tiny_checkpoints
    report = run(
        a_path,
        b_path,
        pairs=1,
        device="cpu",
        sims=2,
        sims_a=2,
        sims_b=5,
        top_k=2,
        slots=1,
        batch_cap=16,
        age_deal_samples=0,
    )
    assert report["parity"]["mode"] == "manual"
    assert report["a"]["sims"] == 2
    assert report["b"]["sims"] == 5


@pytest.mark.slow
def test_time_parity_calibrates_a_budget_and_reports_what_it_achieved(
    tiny_checkpoints,
):
    """The `--parity time` path end to end, structure not magnitude.

    Two 32-wide models at 4 simulations cannot resolve a per-simulation cost on
    a laptop CPU, so the solved budget may well clamp -- which is exactly why
    the clamp is reported and why the verdict rests on the ratio the arena
    MEASURED rather than on this fit.
    """

    a_path, b_path = tiny_checkpoints
    report = run(
        a_path,
        b_path,
        pairs=1,
        device="cpu",
        sims=4,
        parity="time",
        reference="a",
        calibration_games=2,
        calibration_sims=(2, 4),
        top_k=2,
        slots=1,
        batch_cap=16,
        age_deal_samples=0,
    )
    calibration = report["parity"]["calibration"]
    assert calibration is not None
    assert calibration["reference"] == "a"
    assert calibration["budget_seconds_per_move"] > 0
    assert set(calibration["fits"]) == {"a", "b"}
    for fit in calibration["fits"].values():
        assert [point["sims"] for point in fit["points"]] == [2, 4]
        assert all(point["moves"] > 0 for point in fit["points"])
    # The reference side keeps the budget it was asked for; the other side's is
    # solved, and bounded either way round.
    assert report["a"]["sims"] == 4
    assert calibration["floor"] <= report["b"]["sims"] <= calibration["cap"]
    assert isinstance(calibration["clamped"], bool)
    # Calibration is not throughput: the arena's own games/hour excludes it.
    assert report["play_seconds"] < report["seconds"]


@pytest.mark.slow
def test_min_lcb_turns_the_arena_into_a_gate(tiny_checkpoints, tmp_path):
    """Exit 3, not 1: a crash and 'A did not clear the bar' are opposite reads."""

    a_path, b_path = tiny_checkpoints
    output = tmp_path / "report.json"
    argv = [
        "--a", str(a_path),
        "--b", str(b_path),
        "--pairs", "1",
        "--sims", "2",
        "--top-k", "2",
        "--slots", "1",
        "--batch-cap", "16",
        "--age-deal-samples", "0",
        "--device", "cpu",
        "--output", str(output),
        "--quiet",
        # Unreachable at one pair, so the gate must fail whoever wins.
        "--min-lcb", "0.99",
    ]
    assert main(argv) == GATE_FAILED_EXIT_CODE
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["min_lcb"] == 0.99


def test_cli_separates_could_not_run_from_did_not_pass(tmp_path):
    """A missing checkpoint exits 2 (argparse), never the gate's 3."""

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "games.seven_wonders_duel.arena",
            "--a", str(tmp_path / "missing.pt"),
            "--b", str(tmp_path / "also_missing.pt"),
            "--pairs", "1",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 2
    assert "does not exist" in result.stderr


def test_run_rejects_impossible_settings(tiny_checkpoints):
    a_path, b_path = tiny_checkpoints
    with pytest.raises(ValueError, match="pairs must be positive"):
        run(a_path, b_path, pairs=0, device="cpu")
    with pytest.raises(ValueError, match="parity must be"):
        run(a_path, b_path, pairs=2, parity="whatever", device="cpu")
    with pytest.raises(ValueError, match="reference must be"):
        run(a_path, b_path, pairs=2, parity="time", reference="c", device="cpu")
    with pytest.raises(ValueError, match="search must be"):
        run(a_path, b_path, pairs=2, search="mcts", device="cpu")


def test_repeated_searches_reuse_their_inference_threads(tiny_checkpoints):
    """The leak that took a laptop down: a thread per search, never reclaimed.

    Rust used to `thread::spawn` a worker for every `search_many_flat_net` call
    and call the adapter from it.  Torch keeps ~16 MB of per-thread state that
    outlives the thread, so the arena grew ~3.2 GB per `run` and the process'
    OS thread count climbed about 7 per search without ever falling -- 35 GB
    and an out-of-memory shutdown partway through a match.  `eval.rs` pools
    those threads now; this is the arena-side guard on that.

    Thread count is the assertion rather than resident memory: it is the thing
    that actually has to stay bounded, and it is exact where a memory reading is
    a moving target on a shared machine.
    """

    psutil = pytest.importorskip("psutil")
    import seven_wonders_rust as swr

    a_path, _ = tiny_checkpoints
    side = load_side("a", a_path, device="cpu", precision="fp32", batch_cap=16)
    process = psutil.Process()

    def one_search(index: int) -> None:
        swr.search_many_flat_net(
            side.adapter,
            rust_games_for_self_play([9_000 + index], [0]),
            [7 * index],
            16, 1, 2, 2,
            force=True,
            age_deal_samples=0,
            puct_root=False,
        )

    # Warm up first: the pinned thread and Torch's own pool are built on the
    # first forward, and counting that one-time cost as growth would make the
    # test pass for the wrong reason.
    for index in range(4):
        one_search(index)
    settled = process.num_threads()

    for index in range(4, 16):
        one_search(index)

    # A couple of threads of slack for anything the runtime parks on its own;
    # the bug this guards grew the count by ~7 for each of the 12 searches.
    assert process.num_threads() <= settled + 2
