"""W7 S1-S4 + S2b: the specialist league around the biased search.

`test_specialist_bias.py` gates the search change. This gates everything that
turns a biased searcher into a training arrangement: the opponent draw and its
shares arithmetic, per-model target routing, the value-leak contract, per-model
step sizing, the archive and collapse floor, and the S2b selection.
"""

from __future__ import annotations

from dataclasses import replace
import shutil
from pathlib import Path
import random

import pytest

from .buffer import (
    GameRecord,
    MoveRecord,
    archive_policy_seats,
    from_json_line,
    to_json_line,
)
from .dataset import (
    GENERAL_ROUTE,
    bootstrap_root_value,
    move_target_route,
)
from .phase_d import LeagueAssignment, _tag_league_opponents
from .specialist import (
    CLASS_IDS,
    SpecialistConfig,
    SpecialistLineage,
    assert_no_shaped_bootstrap,
    cap_reanalysis,
    class_of_route,
    collapse_verdict,
    draw_opponent_class,
    inflow_census,
    league_game_count,
    league_share,
    parse_specialists,
    reanalysis_candidates,
    route_for,
    shaped_share,
    steps_for_inflow,
)

SCIENCE = SpecialistConfig(name="science", lambda_=0.5, share=0.15)
MILITARY = SpecialistConfig(name="military", lambda_=0.4, share=0.10)


# --------------------------------------------------------------------------
# S1: opponent classes and the shares arithmetic
# --------------------------------------------------------------------------


def test_parse_specialists_round_trips_and_validates():
    parsed = parse_specialists("science:0.15:0.5,military:0.10:0.4:3")
    assert [config.name for config in parsed] == ["science", "military"]
    assert parsed[0].share == 0.15 and parsed[0].lambda_ == 0.5
    assert parsed[1].train_every == 3
    assert parsed[0].route == "specialist:1"
    assert parse_specialists("") == ()

    for bad in (
        "science:0.15",  # too few fields
        "economic:0.1:0.5",  # unknown class
        "science:0.1:0.0",  # a specialist needs a positive lambda
        "science:0.1:0.5,science:0.1:0.5",  # duplicate class
        "science:0.7:0.5,military:0.7:0.5",  # shares over 1
    ):
        with pytest.raises(ValueError):
            parse_specialists(bad)


def test_route_names_round_trip_and_never_collide_with_the_general():
    assert route_for("science") == "specialist:1"
    assert class_of_route(route_for("military")) == "military"
    assert class_of_route(GENERAL_ROUTE) is None
    assert class_of_route("none") is None
    # 0 is the learner at the Rust boundary; no specialist may claim it.
    assert 0 not in CLASS_IDS.values()


def test_the_draw_delivers_the_INTENDED_shares_not_shares_times_L():
    """The one arithmetic error this design invites.

    Shares are fractions of ALL games; ``L`` is their sum and is the fraction of
    each iteration that is league play. Drawing on the raw shares would deliver
    ``share * L`` -- 6% where 15% was intended at ``L = 0.4``. The draw must
    renormalise, and the whole league allocation must go to the drawn class.
    """

    hof_share = 0.15
    specialists = (SCIENCE, MILITARY)
    games_per_iteration = 200
    trials = 20_000
    rng = random.Random(4)
    played = {"hof": 0, "science": 0, "military": 0}
    for _ in range(trials):
        drawn = draw_opponent_class(rng, hof_share, specialists)
        played[drawn] += league_game_count(
            games_per_iteration, hof_share, specialists
        )
    total_games = trials * games_per_iteration
    assert played["hof"] / total_games == pytest.approx(0.15, abs=0.01)
    assert played["science"] / total_games == pytest.approx(0.15, abs=0.01)
    assert played["military"] / total_games == pytest.approx(0.10, abs=0.01)
    assert league_share(hof_share, specialists) == pytest.approx(0.40)


def test_no_shares_means_no_league_at_all():
    rng = random.Random(0)
    assert draw_opponent_class(rng, 0.0, ()) is None
    assert league_game_count(100, 0.0, ()) == 0


def test_a_class_with_a_zero_share_is_never_drawn():
    rng = random.Random(1)
    zero = SpecialistConfig(name="military", lambda_=0.4, share=0.0)
    drawn = {draw_opponent_class(rng, 0.2, (SCIENCE, zero)) for _ in range(500)}
    assert "military" not in drawn


def test_the_assignment_name_groups_by_class_and_keeps_the_hof_spelling():
    hof = LeagueAssignment(
        checkpoint="a.pt",
        sha256="0123456789abcdef",
        iteration_added=42,
        nets_p0=(1,),
        nets_p1=(0,),
    )
    # Unchanged for an archive: existing stats keyed on this name still group.
    assert hof.name == "hof_iter_0042_0123456789ab"
    assert not hof.is_specialist
    specialist = replace(
        hof, opponent_class="science", specialist_lambda=0.5, specialist_class_id=1
    )
    assert specialist.name == "science_iter_0042_0123456789ab"
    assert specialist.is_specialist
    assert specialist.name != hof.name


def test_tagging_records_the_class_the_lambda_and_the_route():
    league = LeagueAssignment(
        checkpoint="s.pt",
        sha256="abc123abc123abc",
        iteration_added=7,
        nets_p0=(0, 1),
        nets_p1=(0, 0),
        opponent_class="science",
        specialist_lambda=0.5,
        specialist_victory="scientific",
        specialist_class_id=1,
        opponent_route="specialist:1",
    )
    records = [_record(agents={"p0": "network", "p1": "network"}) for _ in range(2)]
    tagged = _tag_league_opponents(records, league)
    assert "opponent_class" not in tagged[0].agents  # pure self-play game
    agents = tagged[1].agents
    assert agents["opponent_class"] == "science"
    assert agents["opponent_lambda"] == "0.5"
    assert agents["opponent_route"] == "specialist:1"
    # `opponent_type` keeps its existing vocabulary so W3's consumers are
    # untouched; the class is additive.
    assert agents["opponent_type"] == "hof"


# --------------------------------------------------------------------------
# S2: routing, and the value-leak contract
# --------------------------------------------------------------------------


def _move(i, actor, **kwargs):
    base = dict(
        i=i,
        actor=actor,
        action=0,
        mask_hash="sha256:0",
        visits={0: 1},
        policy_target={0: 1.0},
        root_value=0.2,
        sims=32,
        gumbel_topk=(),
        policy_excluded=False,
    )
    base.update(kwargs)
    return MoveRecord(**base)


def _record(*, agents=None, moves=(), **kwargs):
    base = dict(
        seed=1,
        first_player=0,
        agents=agents or {"p0": "network", "p1": "network"},
        iteration=3,
        winner=0,
        victory_type="scientific",
        scores=(50, 40),
        chance_log=(),
        moves=tuple(moves),
        final_digest="sha256:0",
        trajectory_digest="sha256:0",
    )
    base.update(kwargs)
    return GameRecord(**base)


def test_an_unbiased_move_is_readable_by_every_model():
    move = _move(0, 0)
    assert move_target_route(move) == GENERAL_ROUTE
    value, shaped = bootstrap_root_value(move)
    assert value == 0.2 and shaped is False


def test_a_shaped_root_never_becomes_a_value_target_for_ANYONE():
    """Including the model whose own lambda produced it.

    `value_soft` trains a W/D/L PROBABILITY head, and that head is what the
    next search reads back as its leaf value. Teaching the bonus there makes
    the search add it twice, and it poisons `root_value_unshaped` -- which is
    computed from those same net outputs -- so the leak reaches the general by
    the one channel the quarantine exists to close.

    Routing a utility to its owner does not make it a probability, and there is
    no separate utility head to route it to.
    """

    move = _move(
        0,
        1,
        root_value=0.9,
        root_value_unshaped=0.3,
        search_lambda=0.5,
        search_victory="scientific",
        target_route="specialist:1",
    )
    for _model in ("specialist:1", GENERAL_ROUTE):
        value, shaped = bootstrap_root_value(move)
        assert value == 0.3, "the unshaped root is the only bootstrap source"
        assert shaped is False


def test_a_biased_move_with_no_unshaped_root_carries_no_soft_target_at_all():
    """A missing value costs a soft target; a shaped one teaches a wrong one."""

    move = _move(
        0,
        1,
        root_value=0.9,
        root_value_unshaped=None,
        search_lambda=0.5,
        target_route="specialist:1",
    )
    value, shaped = bootstrap_root_value(move)
    assert value is None and shaped is False


def test_archive_policy_seats_keeps_a_learning_specialists_seat():
    """A specialist is not an archive: its targets are kept and routed."""

    archive_agents = {
        "kind": "league",
        "league_assignment_used": "true",
        "league_assignment": "hof_iter_0007_abc",
        "p1": "hof_iter_0007_abc",
    }
    assert archive_policy_seats(archive_agents) == frozenset({1})
    specialist_agents = dict(
        archive_agents,
        league_assignment="science_iter_0007_abc",
        p1="science_iter_0007_abc",
        opponent_route="specialist:1",
    )
    assert archive_policy_seats(specialist_agents) == frozenset()


def test_the_new_record_fields_round_trip_and_stay_absent_when_inert():
    plain = _record(moves=[_move(0, 0)])
    line = to_json_line(plain)
    # Inert provenance is omitted, so a record's BYTES -- and therefore the
    # example cache's key -- are unchanged by the feature existing.
    assert "search_lambda" not in line
    assert "target_route" not in line
    assert from_json_line(line).moves[0].search_lambda == 0.0
    assert from_json_line(line).moves[0].target_route == GENERAL_ROUTE

    biased = _record(
        moves=[
            _move(
                0,
                1,
                root_value=0.9,
                root_value_unshaped=0.3,
                search_lambda=0.5,
                search_victory="scientific",
                target_route="specialist:1",
                reanalysis=False,
            )
        ]
    )
    restored = from_json_line(to_json_line(biased)).moves[0]
    assert restored.search_lambda == 0.5
    assert restored.search_victory == "scientific"
    assert restored.root_value_unshaped == 0.3
    assert restored.target_route == "specialist:1"


# --------------------------------------------------------------------------
# S3: step counts follow measured inflow
# --------------------------------------------------------------------------


class _Row:
    """An Example stand-in. `target_route` is the SOURCE move's route;
    `derived_for` is the model the row was derived for, and the two differ on
    exactly the rows that carry value across models."""

    def __init__(
        self,
        route,
        has_policy=True,
        search_lambda=0.0,
        shaped=False,
        derived_for=GENERAL_ROUTE,
    ):
        self.target_route = route
        self.has_policy = has_policy
        self.search_lambda = search_lambda
        self.root_value_shaped = shaped
        self.derived_for = derived_for
        self.root_value = 0.1


def test_inflow_census_counts_only_policy_eligible_rows_per_model():
    rows = [
        _Row(GENERAL_ROUTE),
        _Row(GENERAL_ROUTE),
        _Row(GENERAL_ROUTE, has_policy=False),
        _Row("specialist:1"),
        _Row("none", has_policy=False),
    ]
    census = inflow_census(rows)
    assert census[GENERAL_ROUTE] == 2
    assert census["specialist:1"] == 1
    # Present with a zero count, so "this model got nothing" is distinguishable
    # from "this model was not in the buffer at all".
    assert census["none"] == 0


def test_steps_scale_to_the_models_own_inflow():
    """cloud2 ran ~5.2 samples per new position. A specialist at a 15% opponent
    share sees ~7.5% of the general's positions, so the general's step count
    would put it near 65 -- memorising the buffer within a couple of iterations.
    """

    assert steps_for_inflow(1000, 10_000, 750) == 75
    assert steps_for_inflow(1000, 10_000, 10_000) == 1000
    # No inflow means no step, not a step on stale data.
    assert steps_for_inflow(1000, 10_000, 0) == 0
    assert steps_for_inflow(1000, 0, 500) == 0
    # A tiny but nonzero inflow still gets at least one step rather than
    # silently never training.
    assert steps_for_inflow(1000, 10_000, 1) >= 1


def test_step_counts_follow_BANKED_ROWS_not_elapsed_iterations():
    """The defect this replaces: `iterations_since_train + 1` multiplied the
    NEWEST iteration's count, so inflows of 0, 0, 100 earned 75 steps where 100
    rows warrant 25. One opponent class is drawn per iteration, so an iteration
    that supplies a given specialist nothing is the ordinary case.
    """

    assert steps_for_inflow(100, 400, 100) == 25
    # Four iterations of the same inflow banked together earn four times the
    # steps, because four times the ROWS arrived -- not because four iterations
    # elapsed.
    assert steps_for_inflow(100, 400, 400) == 100
    # ... and elapsed time with no rows earns nothing.
    assert steps_for_inflow(100, 400, 0) == 0


def test_banking_accumulates_rows_across_empty_iterations(tmp_path: Path):
    lineage = SpecialistLineage(tmp_path, replace(SCIENCE, train_every=3))
    lineage.bootstrap(_checkpoint(tmp_path / "g.pt", b"general"), 0)
    assert lineage.bank_inflow(0).banked_inflow == 0
    assert lineage.bank_inflow(0).banked_inflow == 0
    assert lineage.bank_inflow(100).banked_inflow == 100
    # Spending the bank empties it, so the same rows cannot be spent twice.
    lineage.accept(_checkpoint(tmp_path / "c.pt", b"v1"), 3, steps=25)
    assert lineage.load().banked_inflow == 0


def test_shaped_share_measures_what_a_buffer_actually_holds():
    rows = [_Row(GENERAL_ROUTE), _Row("specialist:1", search_lambda=0.5)]
    assert shaped_share(rows) == pytest.approx(0.5)
    assert shaped_share([]) == 0.0


def test_the_isolation_assertion_catches_a_shaped_root_in_ANY_buffer():
    """The configuration that makes this dangerous is the one cloud2 ran."""

    clean = [_Row(GENERAL_ROUTE), _Row("specialist:1", search_lambda=0.5)]
    assert_no_shaped_bootstrap(clean, GENERAL_ROUTE)

    leaked = _Row("specialist:1", search_lambda=0.5, shaped=True)
    with pytest.raises(AssertionError, match="probability head"):
        assert_no_shaped_bootstrap([leaked], GENERAL_ROUTE)
    # ... and it is NOT legitimate in its owner's buffer either. The specialist
    # reads its own value head back as a leaf value, so a shaped target there
    # makes it count the bonus twice.
    leaked.derived_for = "specialist:1"
    with pytest.raises(AssertionError, match="probability head"):
        assert_no_shaped_bootstrap([leaked], "specialist:1")


def test_the_isolation_assertion_also_catches_a_mislabelled_derivation():
    """Provenance and derivation must agree, or the shaped check above is
    reading a field that means something else."""

    stray = _Row("specialist:1", derived_for="specialist:1")
    with pytest.raises(AssertionError, match="provenance disagrees"):
        assert_no_shaped_bootstrap([stray], GENERAL_ROUTE)


# --------------------------------------------------------------------------
# S4: lineage, archive, collapse floor
# --------------------------------------------------------------------------


def _checkpoint(path: Path, payload: bytes = b"weights") -> Path:
    path.write_bytes(payload)
    return path


def test_bootstrap_archives_the_frozen_reference_and_it_never_moves(tmp_path: Path):
    """The first accepted specialist doubles as the frozen attacker.

    Fixed at the start of the run and never rebuilt: scoring an improving
    general against an improving specialist cannot separate "defence got
    stronger" from "the attacks got weaker".
    """

    lineage = SpecialistLineage(tmp_path, SCIENCE)
    source = _checkpoint(tmp_path / "current_best.pt", b"general")
    lineage.bootstrap(source, iteration=10)
    frozen = lineage.frozen_reference()
    assert frozen is not None
    assert Path(frozen).read_bytes() == b"general"

    lineage.accept(_checkpoint(tmp_path / "cand.pt", b"trained"), 11, steps=40)
    lineage.accept(_checkpoint(tmp_path / "cand2.pt", b"trained2"), 12, steps=40)
    assert lineage.frozen_reference() == frozen
    assert Path(frozen).read_bytes() == b"general"
    # Archived from the FIRST accepted specialist, not later: attacking styles
    # are forgotten the same way general strategies are.
    assert len(lineage.archive.entries()) == 3


def test_accept_advances_without_a_promotion_gate(tmp_path: Path):
    """A specialist is sparring equipment, not the product.

    Deliberately not the general's soft gate: it produced 0 promotions over 38k
    games in cloud6, and a specialist population that silently never advances is
    a full run wasted before anyone notices.
    """

    lineage = SpecialistLineage(tmp_path, SCIENCE)
    lineage.bootstrap(_checkpoint(tmp_path / "g.pt", b"general"), 0)
    state = lineage.accept(_checkpoint(tmp_path / "c.pt", b"v1"), 1, steps=30)
    assert state.update_count == 30
    assert lineage.latest_path.read_bytes() == b"v1"
    state = lineage.accept(_checkpoint(tmp_path / "c2.pt", b"v2"), 2, steps=30)
    assert state.update_count == 60
    assert state.iterations_trained == 2


def test_the_floor_reverts_to_the_last_good_checkpoint(tmp_path: Path):
    lineage = SpecialistLineage(tmp_path, SCIENCE)
    lineage.bootstrap(_checkpoint(tmp_path / "g.pt", b"general"), 0)
    lineage.accept(_checkpoint(tmp_path / "c.pt", b"healthy"), 1, steps=10)
    lineage.mark_good(1, 0.42)
    lineage.accept(_checkpoint(tmp_path / "c2.pt", b"broken"), 2, steps=10)
    assert lineage.latest_path.read_bytes() == b"broken"

    healthy, _ = collapse_verdict(0.42, SCIENCE)
    assert healthy
    broken, reason = collapse_verdict(0.02, SCIENCE)
    assert not broken and "collapse floor" in reason

    state = lineage.revert(2, 0.02)
    assert lineage.latest_path.read_bytes() == b"healthy"
    assert state.reverts == 1
    assert any(entry["event"] == "revert" for entry in state.history)


def test_a_specialist_scoring_below_the_general_is_not_a_failure():
    """Specialists are EXPECTED to score worse overall, by design."""

    healthy, _ = collapse_verdict(0.30, SCIENCE)
    assert healthy


def test_lineage_state_survives_a_fresh_process(tmp_path: Path):
    lineage = SpecialistLineage(tmp_path, SCIENCE)
    lineage.bootstrap(_checkpoint(tmp_path / "g.pt", b"general"), 0)
    lineage.accept(_checkpoint(tmp_path / "c.pt", b"v1"), 1, steps=25)
    reopened = SpecialistLineage(tmp_path, SCIENCE).load()
    assert reopened.update_count == 25
    assert reopened.lambda_ == SCIENCE.lambda_
    assert reopened.victory == "scientific"
    assert reopened.bootstrapped_from is not None


def test_two_specialists_do_not_share_a_directory(tmp_path: Path):
    """The lifecycles must not be able to reset each other by sharing a path."""

    science = SpecialistLineage(tmp_path, SCIENCE)
    military = SpecialistLineage(tmp_path, MILITARY)
    assert science.directory != military.directory
    science.bootstrap(_checkpoint(tmp_path / "g.pt", b"general"), 0)
    assert not military.latest_path.exists()
    assert military.frozen_reference() is None


def test_idle_iterations_bank_toward_train_every(tmp_path: Path):
    lineage = SpecialistLineage(tmp_path, replace(SCIENCE, train_every=3))
    lineage.bootstrap(_checkpoint(tmp_path / "g.pt", b"general"), 0)
    lineage.note_idle_iteration()
    lineage.note_idle_iteration()
    assert lineage.load().iterations_since_train == 2
    lineage.accept(_checkpoint(tmp_path / "c.pt", b"v1"), 3, steps=90)
    assert lineage.load().iterations_since_train == 0


# --------------------------------------------------------------------------
# S2b: reanalysis selection
# --------------------------------------------------------------------------


def test_reanalysis_selects_where_lambda_moved_the_valuation():
    moves = [
        # unbiased -- never a candidate
        _move(0, 0),
        # biased but the bias barely moved the valuation
        _move(
            1,
            1,
            root_value=0.31,
            root_value_unshaped=0.30,
            search_lambda=0.5,
            search_victory="scientific",
            target_route="specialist:1",
        ),
        # biased and it moved a lot
        _move(
            2,
            1,
            root_value=0.80,
            root_value_unshaped=0.30,
            search_lambda=0.5,
            search_victory="scientific",
            target_route="specialist:1",
        ),
    ]
    record = _record(moves=moves, winner=1, victory_type="civilian")
    assert reanalysis_candidates(record, min_gap=0.05) == [2]


def test_every_specialist_move_of_a_type_win_is_a_candidate():
    """A rush that WORKED is the line the general should learn to play, and the
    per-move gap can be small all along a plan that only pays off at the end.
    """

    moves = [
        _move(
            i,
            1,
            root_value=0.31,
            root_value_unshaped=0.30,
            search_lambda=0.5,
            search_victory="scientific",
            target_route="specialist:1",
        )
        for i in range(3)
    ]
    won = _record(moves=moves, winner=1, victory_type="scientific")
    assert reanalysis_candidates(won, min_gap=0.5) == [0, 1, 2]
    lost = _record(moves=moves, winner=0, victory_type="scientific")
    assert reanalysis_candidates(lost, min_gap=0.5) == []


def test_cheap_searches_are_never_reanalysed():
    move = _move(
        0,
        1,
        root_value=0.9,
        root_value_unshaped=0.1,
        search_lambda=0.5,
        target_route="specialist:1",
        policy_excluded=True,
    )
    record = _record(moves=[move], winner=1, victory_type="scientific")
    assert reanalysis_candidates(record) == []


def test_the_reanalysis_share_is_capped_and_trimmed_round_robin():
    """So one long game cannot consume the whole allowance."""

    selected = [[0, 1, 2, 3], [10, 11], [20]]
    capped = cap_reanalysis(selected, general_inflow=100, cap=0.04)
    # 4% of 100 policy rows is a budget of 4, spread one per game before any
    # game gets a second row.
    assert capped == [[0], [10], [20], [1]][:3] or capped == [[0, 1], [10], [20]]
    assert sum(len(entry) for entry in capped) == 4
    assert capped[1] == [10] and capped[2] == [20]
    # An uncapped call is the identity.
    assert cap_reanalysis(selected, 100, 1.0) == selected
    # A cap that rounds to nothing selects nothing rather than everything.
    assert cap_reanalysis(selected, 10, 0.01) == [[], [], []]


# --------------------------------------------------------------------------
# End to end: a specialist really does search biased, and its targets really
# are routed
# --------------------------------------------------------------------------


def _two_net_records(
    specialist_lambda, *, games=4, symmetric=False, solve_endgames=False
):
    """One production generation call with network 1 as the (maybe) specialist.

    Real nets, the production flat boundary and the production scheduler --
    everything between `PhaseDLoop.generate_iteration` and the recorded rows
    except the loop itself. A mock evaluator could not be used here: the bias
    reads W4's outlook head, and refusing to bias without one is the point.
    """

    torch = pytest.importorskip("torch")
    swr = pytest.importorskip("seven_wonders_rust")

    from .inference import Evaluator
    from .net import SWDNet
    from .rust_bridge import (
        phase_d_records_from_rust,
        rust_games_for_self_play,
        rust_searcher_routed_flat_batch_adapter,
    )

    def make(seed):
        torch.manual_seed(seed)
        return SWDNet(d_model=32, layers=2, heads=4, hierarchical_value=True)

    evaluators = tuple(
        Evaluator(make(seed), "cpu", 64, fuse_embedder=False) for seed in (0, 1)
    )
    adapter = rust_searcher_routed_flat_batch_adapter(evaluators)
    seeds = list(range(4242, 4242 + games))
    first = [(index // 2) % 2 for index in range(games)]
    # Network 1 on seat 1 for every game, so every game has a specialist seat.
    raw, _metrics = swr.self_play_many_flat_net(
        adapter=adapter,
        games=rust_games_for_self_play(seeds, first),
        game_seeds=seeds,
        global_batch_cap=32,
        leaf_batch=1,
        cheap_sims_min=2,
        cheap_sims_max=3,
        full_sims_min=6,
        full_sims_max=8,
        full_search_fraction=0.5,
        top_k=3,
        draft_prior=0.0,
        iteration=5,
        max_moves=256,
        nets_p0=[0] * games,
        nets_p1=[1] * games,
        specialist_lambda=specialist_lambda,
        specialist_victory="scientific" if specialist_lambda else None,
        specialist_symmetric=symmetric,
        specialist_class_id=1 if specialist_lambda else 0,
        solve_endgames=solve_endgames,
    )
    return phase_d_records_from_rust(raw, validate=False)


def test_a_specialist_seat_records_its_bias_and_the_learner_seat_does_not():
    records = _two_net_records(0.5)
    specialist_moves = [
        move for record in records for move in record.moves if move.actor == 1
    ]
    learner_moves = [
        move for record in records for move in record.moves if move.actor == 0
    ]
    searched_specialist = [m for m in specialist_moves if m.sims > 0]
    assert searched_specialist, "no specialist move was searched"

    for move in searched_specialist:
        assert move.search_lambda == 0.5
        assert move.search_victory == "scientific"
        assert move.search_symmetric is False
        assert move.target_route == "specialist:1"
        # The separately accumulated lambda-zero root, which is what every
        # other model may bootstrap from.
        assert move.root_value_unshaped is not None
        assert -1.0 <= move.root_value_unshaped <= 1.0
    for move in (m for m in learner_moves if m.sims > 0):
        assert move.search_lambda == 0.0
        assert move.search_victory is None
        assert move.root_value_unshaped is None
        assert move.target_route == GENERAL_ROUTE


def test_a_learning_specialists_full_moves_are_no_longer_policy_excluded():
    """The S2 inversion: an archive's targets are dropped, a specialist's are
    kept and routed. `policy_excluded` still carries "cheap search"."""

    records = _two_net_records(0.5)
    full = [
        move
        for record in records
        for move in record.moves
        if move.actor == 1 and move.sims > 0 and not move.policy_excluded
    ]
    assert full, "a learning specialist's full moves must keep their targets"
    assert all(move.target_route == "specialist:1" for move in full)
    cheap = [
        move
        for record in records
        for move in record.moves
        if move.actor == 1 and move.sims > 0 and move.policy_excluded
    ]
    # Cheap moves are still excluded, for the reason they always were.
    assert all(move.policy_target is not None for move in cheap)


def test_lambda_zero_generation_is_byte_identical_to_no_specialist_at_all():
    """The plumbing is inert until it is asked for, on the production path."""

    plain = _two_net_records(0.0)
    lines = [to_json_line(record) for record in plain]
    again = [to_json_line(record) for record in _two_net_records(0.0)]
    assert lines == again
    for record in plain:
        for move in record.moves:
            assert move.search_lambda == 0.0
            assert move.root_value_unshaped is None
            # Network 1 with no specialist spec is an ARCHIVE: it learns
            # nothing, so its policy label goes nowhere -- which is exactly the
            # behaviour that existed before W7, now spelled out in the record.
            expected = GENERAL_ROUTE if move.actor == 0 else "none"
            assert move.target_route == expected


def test_a_live_lambda_changes_the_games_that_are_generated():
    """A null result must not be explicable by "the flag did nothing"."""

    unbiased = [to_json_line(r) for r in _two_net_records(0.0)]
    biased = [to_json_line(r) for r in _two_net_records(1.0)]
    assert unbiased != biased


def test_the_general_never_bootstraps_from_a_shaped_root_end_to_end():
    """The isolation test the plan asks for, over real generated records.

    Run with `value_bootstrap > 0` in mind: `Example.root_value` is exactly what
    `dataset.collate` turns into `value_soft`, so this asserts on the quantity
    that would carry the leak.
    """

    from .dataset import examples_from_records

    records = _two_net_records(0.5)
    general = examples_from_records(records, derived_for=GENERAL_ROUTE)
    assert general
    assert_no_shaped_bootstrap(general, GENERAL_ROUTE)
    # The general still learns VALUE from the specialist's positions -- that is
    # the point of the arrangement -- so those rows must be present, and their
    # root must be the unshaped one.
    from_specialist = [
        example for example in general if example.target_route == "specialist:1"
    ]
    assert from_specialist, "the general saw none of the specialist's positions"
    assert all(not example.has_policy for example in from_specialist)
    assert all(not example.root_value_shaped for example in from_specialist)

    own = examples_from_records(records, derived_for="specialist:1")
    mine = [e for e in own if e.target_route == "specialist:1"]
    assert any(example.has_policy for example in mine)
    # Its POLICY label is its own; its VALUE target is still the unshaped root,
    # because the value head is a probability head and its own search adds the
    # bonus itself.
    assert all(not example.root_value_shaped for example in mine)
    assert_no_shaped_bootstrap(own, "specialist:1")


# --------------------------------------------------------------------------
# The loop: seeding, drawing, and the two lifecycles staying apart
# --------------------------------------------------------------------------


def _loop(tmp_path: Path, **overrides):
    from .phase_d import PhaseDConfig, PhaseDLoop
    from .test_league_generation import add_archive, write_iterations

    config = PhaseDConfig(
        run_dir=str(tmp_path / "run"),
        seed_games=0,
        d_model=32,
        layers=1,
        device="cpu",
        games_per_iteration=500,
        hof_opponent_fraction=0.15,
        hof_start_games=10_000,
        # A biased search reads W4's outlook head, so a league run must carry
        # one; the config refuses to launch without it.
        hierarchical_value=True,
        hier_value_weight=0.5,
        **{"specialists": "science:0.15:0.5,military:0.10:0.4", **overrides},
    )
    loop = PhaseDLoop(config)
    write_iterations(loop, 20, 500)  # 10,000 games: past the threshold
    add_archive(loop, 5)
    return loop


def _promoted_general(loop, *, hierarchical=True):
    """A real checkpoint at `current_best`, which is what a specialist seeds from.

    It carries W4's head by default: a biased search reads the outlook from it,
    so a specialist seeded from a checkpoint without one could never search.
    """

    import torch

    from .net import SWDNet
    from .train import make_checkpoint

    loop.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3)
    model = SWDNet(
        d_model=32, layers=1, heads=4, hierarchical_value=hierarchical
    )
    torch.save(
        make_checkpoint(
            model,
            {
                "model": "transformer",
                "d_model": 32,
                "layers": 1,
                "heads": 4,
                "hierarchical_value": hierarchical,
            },
        ),
        loop.current_best,
    )
    return loop.current_best


def test_seeding_refuses_a_general_without_the_outlook_head(tmp_path: Path):
    """The config check covers the FLAGS; this covers the WEIGHTS.

    A resumed run whose `current_best` predates `--hierarchical-value` would
    otherwise seed a specialist that dies at its first biased leaf.
    """

    loop = _loop(tmp_path)
    _promoted_general(loop, hierarchical=False)
    with pytest.raises(ValueError, match="no hierarchical value head"):
        loop.bootstrap_specialists(20)


def test_an_unseeded_specialist_falls_back_to_the_archive(tmp_path: Path):
    """An iteration with NO opponent is a silently different curriculum from one
    with an archive, so the fallback is deliberate rather than a skip."""

    loop = _loop(tmp_path)
    drew = [loop.league_assignment(iteration, 500) for iteration in range(20, 40)]
    assert all(assignment is not None for assignment in drew)
    assert {assignment.opponent_class for assignment in drew} == {"hof"}
    assert all(assignment.specialist_lambda == 0.0 for assignment in drew)


def test_seeding_freezes_the_general_anchor_exactly_once(tmp_path: Path):
    loop = _loop(tmp_path)
    _promoted_general(loop)
    assert loop.bootstrap_specialists(20) == ["science", "military"]
    anchor = loop.general_anchor_path
    assert anchor.is_file()
    stamp = anchor.stat().st_mtime_ns
    # Idempotent: a second call seeds nothing and must not rebuild the anchor,
    # which every measurement in the workstream is read against.
    assert loop.bootstrap_specialists(21) == []
    assert anchor.stat().st_mtime_ns == stamp


def test_once_seeded_the_draw_reaches_the_specialists_with_their_bias(
    tmp_path: Path,
):
    loop = _loop(tmp_path)
    _promoted_general(loop)
    loop.bootstrap_specialists(20)
    drew = [loop.league_assignment(iteration, 500) for iteration in range(20, 80)]
    classes = {assignment.opponent_class for assignment in drew}
    assert classes == {"hof", "science", "military"}
    science = next(a for a in drew if a.opponent_class == "science")
    assert science.specialist_lambda == 0.5
    assert science.specialist_victory == "scientific"
    assert science.opponent_route == "specialist:1"
    assert science.specialist_class_id == 1
    military = next(a for a in drew if a.opponent_class == "military")
    assert military.specialist_victory == "military"
    assert military.opponent_route == "specialist:2"
    hof = next(a for a in drew if a.opponent_class == "hof")
    assert hof.specialist_lambda == 0.0 and hof.opponent_route == "none"
    # 40% of 500 games go to whichever class was drawn.
    assert all(assignment.games == 200 for assignment in drew)


def test_the_draw_stays_resume_stable_with_specialists_configured(tmp_path: Path):
    loop = _loop(tmp_path)
    _promoted_general(loop)
    loop.bootstrap_specialists(20)
    first = loop.league_assignment(33, 500)
    second = loop.league_assignment(33, 500)
    assert first.opponent_class == second.opponent_class
    assert first.checkpoint == second.checkpoint
    assert first.nets_p0 == second.nets_p0


def test_an_unseeded_specialist_is_skipped_rather_than_trained(tmp_path: Path):
    loop = _loop(tmp_path)
    rows = loop.run_specialist_iteration([], 20, general_inflow=1000)
    assert [row["class"] for row in rows] == ["science", "military"]
    assert all(row["trained"] is False for row in rows)
    assert all(row["skipped"] == "not seeded" for row in rows)


def test_a_specialist_with_no_inflow_takes_no_step(tmp_path: Path):
    """No inflow means no step, not a step on stale data."""

    loop = _loop(tmp_path)
    _promoted_general(loop)
    loop.bootstrap_specialists(20)
    rows = loop.run_specialist_iteration([], 21, general_inflow=1000)
    assert all(row["trained"] is False for row in rows)
    assert all(row["skipped"] == "no inflow" for row in rows)
    # Skipping banks the iteration rather than losing it.
    assert loop.specialist_lineages["science"].load().iterations_since_train == 1


def test_specialists_are_off_unless_configured(tmp_path: Path):
    loop = _loop(tmp_path, specialists="")
    assert loop.specialist_configs == ()
    assert loop.specialist_lineages == {}
    assert loop.bootstrap_specialists(20) == []
    assert loop.run_specialist_iteration([], 20, general_inflow=1000) == []
    assignment = loop.league_assignment(20, 500)
    assert assignment.opponent_class == "hof"
    assert assignment.games == 75  # 15% of 500, exactly as before W7


def test_a_league_without_the_outlook_head_is_refused_at_launch(tmp_path: Path):
    """The right error at the right moment.

    The search raises at the first leaf with no outlook, which is correct but
    would kill a run minutes into generation rather than at launch.
    """

    from .phase_d import PhaseDConfig

    config = PhaseDConfig(
        run_dir=str(tmp_path / "a"),
        specialists="science:0.15:0.5",
        hierarchical_value=False,
    )
    with pytest.raises(ValueError, match="hierarchical-value"):
        config.validate()


def test_the_league_composition_is_part_of_the_schedule_identity(tmp_path: Path):
    """Changing a share mid-run changes which opponent every later iteration
    draws, so a resume must not be able to do it silently."""

    from .phase_d import PhaseDConfig

    base = PhaseDConfig(
        run_dir=str(tmp_path / "a"),
        specialists="science:0.15:0.5",
        hierarchical_value=True,
        hier_value_weight=0.5,
    )
    changed = PhaseDConfig(
        run_dir=str(tmp_path / "a"),
        specialists="science:0.20:0.5",
        hierarchical_value=True,
        hier_value_weight=0.5,
    )
    assert base.schedule_identity() != changed.schedule_identity()


def test_shares_that_exceed_every_game_are_rejected(tmp_path: Path):
    from .phase_d import PhaseDConfig

    config = PhaseDConfig(
        run_dir=str(tmp_path / "a"),
        hof_opponent_fraction=0.6,
        specialists="science:0.3:0.5,military:0.3:0.5",
        hierarchical_value=True,
        hier_value_weight=0.5,
    )
    with pytest.raises(ValueError, match="exceed every game"):
        config.validate()


# --------------------------------------------------------------------------
# The solver policy under a live bias
# --------------------------------------------------------------------------


def _records_with_solver(specialist_lambda, *, games=4):
    """As `_two_net_records`, with the exact endgame solver enabled.

    The solver's eligibility and budget are process globals, so this saves and
    restores them: a leaked solver setting would silently change every later
    test in the process, which is exactly the class of defect the
    `solve_endgames` per-call flag exists to prevent.
    """

    swr = pytest.importorskip("seven_wonders_rust")

    saved = swr.endgame_solver()
    swr.set_endgame_solver(200_000, 2.0, 8, True)
    try:
        return _two_net_records(specialist_lambda, games=games, solve_endgames=True)
    finally:
        # The "off" state is `max_nodes = 0`, and the setter refuses a
        # non-positive `max_secs`, so restoring the default means disabling it
        # explicitly rather than replaying the tuple it handed back.
        if saved[0] == 0:
            swr.set_endgame_solver(0, 1.0, 0, False)
        else:
            swr.set_endgame_solver(*saved)


def test_a_biased_searcher_does_not_take_the_solver_shortcut():
    """The stated policy for the solver boundary.

    The endgame mask is a proof of UNBIASED optimality. Applied to a shaped
    search it would delete exactly the attacking continuations the bias funded
    whenever they are provably a shade worse -- silently turning the specialist
    back into the general at every solved endgame. The general's seat is
    unaffected, which is what makes this a policy and not an accident.
    """

    biased = _records_with_solver(0.5)
    specialist_solves = sum(
        1
        for record in biased
        for move in record.moves
        if move.actor == 1 and move.solver_attempted
    )
    learner_solves = sum(
        1
        for record in biased
        for move in record.moves
        if move.actor == 0 and move.solver_attempted
    )
    assert specialist_solves == 0
    assert learner_solves > 0, "the solver never ran at all -- test proves nothing"

    # An ARCHIVE on the same seat still solves: the exclusion is keyed on the
    # bias, not on the seat or the network id.
    unbiased = _records_with_solver(0.0)
    archive_solves = sum(
        1
        for record in unbiased
        for move in record.moves
        if move.actor == 1 and move.solver_attempted
    )
    assert archive_solves > 0


# --------------------------------------------------------------------------
# Exploration: "any net that is training", not "network 0"
# --------------------------------------------------------------------------


def _seat1_only_records(specialist_lambda, dirichlet_epsilon, *, games=4):
    """Games where ONLY seat 1 searches, so its exploration is observable.

    Seat 0 is a scripted bot. With both seats searching, the learner's own root
    noise perturbs the shared trajectory and nothing about seat 1 could be
    isolated from it.
    """

    torch = pytest.importorskip("torch")
    swr = pytest.importorskip("seven_wonders_rust")

    from .inference import Evaluator
    from .net import SWDNet
    from .rust_bridge import (
        phase_d_records_from_rust,
        rust_games_for_self_play,
        rust_searcher_routed_flat_batch_adapter,
    )

    def make(seed):
        torch.manual_seed(seed)
        return SWDNet(d_model=32, layers=2, heads=4, hierarchical_value=True)

    evaluators = tuple(
        Evaluator(make(seed), "cpu", 64, fuse_embedder=False) for seed in (0, 1)
    )
    seeds = list(range(7100, 7100 + games))
    raw, _metrics = swr.self_play_many_flat_net(
        adapter=rust_searcher_routed_flat_batch_adapter(evaluators),
        games=rust_games_for_self_play(seeds, [0] * games),
        game_seeds=seeds,
        global_batch_cap=32,
        leaf_batch=1,
        cheap_sims_min=2,
        cheap_sims_max=3,
        full_sims_min=6,
        full_sims_max=8,
        full_search_fraction=1.0,
        top_k=3,
        draft_prior=0.0,
        iteration=5,
        max_moves=256,
        puct_root=True,
        bots_p0=["science_aggressive/v1"] * games,
        bots_p1=[None] * games,
        nets_p0=[0] * games,
        nets_p1=[1] * games,
        dirichlet_epsilon=dirichlet_epsilon,
        specialist_lambda=specialist_lambda,
        specialist_victory="scientific" if specialist_lambda else None,
        specialist_class_id=1 if specialist_lambda else 0,
    )
    return [to_json_line(record) for record in phase_d_records_from_rust(raw, validate=False)]


def test_an_archived_opponent_still_plays_noise_free():
    """Unchanged by W7, and it must be: handicapping an archive inflates the
    learner's league win rate, and league games are a sixth of the data."""

    assert _seat1_only_records(0.0, 0.0) == _seat1_only_records(0.0, 0.25)


def test_a_learning_specialist_explores_like_the_learner_does():
    """The predicate becomes "any net that is TRAINING", not "network 0".

    Root noise was gated on network 0 because network 1 was always a frozen
    archive. A specialist that learns needs exploration for exactly the reason
    the learner does, and without it its policy never discovers whether the
    moves it declined were good.
    """

    assert _seat1_only_records(0.5, 0.0) != _seat1_only_records(0.5, 0.25)


def test_projection_equals_a_full_derivation_for_the_same_model():
    """The optimization that keeps one derivation per record, not one per model.

    Replay/encode/vectorize is the same work for every model; only four scalars
    per row differ. This asserts the cheap path and the expensive path agree on
    every field, so the saving is free rather than approximate.
    """

    from .dataset import examples_from_record, project_examples

    for record in _two_net_records(0.5, games=3):
        for route in (GENERAL_ROUTE, "specialist:1"):
            derived = examples_from_record(record, derived_for=route)
            projected = project_examples(
                examples_from_record(record, derived_for=GENERAL_ROUTE),
                record,
                route,
            )
            assert len(projected) == len(derived)
            for want, got in zip(derived, projected):
                assert want.has_policy == got.has_policy
                assert want.root_value == got.root_value
                assert want.root_value_shaped == got.root_value_shaped
                assert want.derived_for == got.derived_for
                assert want.target_route == got.target_route
                assert want.value_class == got.value_class
                assert want.joint7_class == got.joint7_class
                assert want.move_index == got.move_index
                assert (want.policy_target == got.policy_target).all()
                assert (want.legal == got.legal).all()
                assert (want.features == got.features).all()


# --------------------------------------------------------------------------
# The training paths actually run
# --------------------------------------------------------------------------


def _biased_run(tmp_path: Path, **overrides):
    """A loop with a promoted general, seeded specialists, and biased records."""

    loop = _loop(
        tmp_path, **{"train_steps": 2, "train_batch_size": 4, **overrides}
    )
    _promoted_general(loop)
    loop.bootstrap_specialists(20)
    records = _two_net_records(0.5, games=6)
    return loop, records


def test_a_specialist_train_step_runs_and_advances_its_lineage(tmp_path: Path):
    """The largest untested path in the build: an actual specialist update."""

    loop, records = _biased_run(tmp_path)
    lineage = loop.specialist_lineages["science"]
    before = lineage.latest_path.read_bytes()
    rows = loop.run_specialist_iteration(records, 21, general_inflow=200)
    science = next(row for row in rows if row["class"] == "science")
    assert science["trained"] is True, science
    assert science["inflow"] > 0
    assert science["steps"] >= 1
    assert lineage.load().update_count == science["steps"]
    assert lineage.latest_path.read_bytes() != before
    # Its own optimizer state, beside the general's rather than shared with it.
    assert lineage.optimizer_path.is_file()
    assert lineage.optimizer_path != loop.optimizer_state_path

    # The military specialist appears in no game here, so it has no inflow and
    # must take no step rather than a step on the science specialist's data.
    military = next(row for row in rows if row["class"] == "military")
    assert military["trained"] is False


def test_a_specialists_step_count_is_smaller_than_the_generals(tmp_path: Path):
    """S3: a specialist sees a fraction of the general's positions, and giving
    it the general's step count memorises its buffer within a couple of
    iterations."""

    loop, records = _biased_run(tmp_path, train_steps=1000)
    rows = loop.run_specialist_iteration(records, 21, general_inflow=10_000)
    science = next(row for row in rows if row["class"] == "science")
    assert science["trained"] is True
    assert 0 < science["steps"] < loop.config.train_steps


def test_reanalysis_produces_general_rows_from_the_specialists_positions(
    tmp_path: Path,
):
    """S2b end to end: selection, re-search at lambda zero, and routing."""

    loop, records = _biased_run(tmp_path, specialist_reanalysis=True)
    extra, stats = loop.reanalysis_for_general(records, 21, general_inflow=400)
    if stats["positions"] == 0:
        pytest.skip("no position in this sample moved the valuation far enough")
    assert extra
    assert stats["examples"] == len(extra)
    assert stats["share"] <= stats["cap"] + 1e-9
    for example in extra:
        # Targets for the GENERAL, produced by an unbiased search.
        assert example.reanalysis is True
        assert example.has_policy is True
        assert example.derived_for == GENERAL_ROUTE
        assert example.target_route == GENERAL_ROUTE
        assert example.search_lambda == 0.0
        assert example.root_value_shaped is False
        assert -1.0 <= example.root_value <= 1.0
        assert example.policy_target.sum() == pytest.approx(1.0, abs=1e-6)
    assert_no_shaped_bootstrap(extra, GENERAL_ROUTE)


def test_outcomes_are_reported_by_opponent_class_not_as_one_aggregate():
    """"Did we beat the science attacker" and "did we beat the archive" are
    different questions; pooling them hides both."""

    from .phase_d import summarize_records

    league = LeagueAssignment(
        checkpoint="s.pt",
        sha256="abc123abc123abc",
        iteration_added=7,
        nets_p0=(0, 1),
        nets_p1=(0, 0),
        opponent_class="science",
        specialist_lambda=0.5,
        specialist_victory="scientific",
        specialist_class_id=1,
        opponent_route="specialist:1",
    )
    # Game 1 is the league game; the specialist sits on seat 0 and loses.
    records = _tag_league_opponents(
        [
            _record(winner=0, victory_type="civilian"),
            _record(winner=1, victory_type="scientific"),
        ],
        league,
    )
    summary = summarize_records(records)["opponent_classes"]
    assert set(summary) == {"science"}
    assert summary["science"]["games"] == 1
    # The learner held seat 1 and won, so it scores a full point.
    assert summary["science"]["learner_score_rate"] == pytest.approx(1.0)
    assert summary["science"]["victory_types"] == {"scientific": 1}
    # A pure self-play game contributes to no class, rather than pulling every
    # class toward 0.5.
    assert sum(entry["games"] for entry in summary.values()) == 1


def test_the_collapse_floor_runs_a_real_match_and_can_revert(tmp_path: Path):
    """The floor is measured against the FROZEN anchor, not the current
    generator: a floor defined against the thing it protects moves under it."""

    loop, _records = _biased_run(tmp_path, anchor_games=2, gate_backend="rust")
    config = next(c for c in loop.specialist_configs if c.name == "science")
    lineage = loop.specialist_lineages["science"]
    result = loop.specialist_floor_check(config, 21, games=2)
    assert result["measured"] is True
    assert result["biased"] is True, (
        "the floor must measure the OPPONENT THAT PLAYS -- these weights with "
        "the bonus switched off are a different player"
    )
    assert 0.0 <= result["score_rate"] <= 1.0
    assert result["floor"] == config.collapse_floor
    state = lineage.load()
    assert state.last_score == pytest.approx(result["score_rate"])
    assert state.last_score_iteration == 21
    # Healthy or not, the general's own lifecycle is untouched by this.
    assert loop.current_best.is_file()


# --------------------------------------------------------------------------
# Regressions from the 2026-09-08 implementation review
# --------------------------------------------------------------------------


def test_a_rollback_takes_the_rejected_specialist_out_of_GENERATION(tmp_path: Path):
    """R1. Restoring `latest.pt` is not a rollback on its own.

    Generation samples the ARCHIVE, and `accept` archives every candidate, so
    the rejected checkpoint was the newest entry and stayed the usual opponent
    after a collapse. The live branch must resolve from the live file, and
    rejected entries must be quarantined against archive draws too.
    """

    lineage = SpecialistLineage(tmp_path, SCIENCE)
    lineage.bootstrap(_checkpoint(tmp_path / "g.pt", b"general"), 0)
    lineage.accept(_checkpoint(tmp_path / "c.pt", b"healthy"), 1, steps=10)
    lineage.mark_good(1, 0.42)
    lineage.accept(_checkpoint(tmp_path / "c2.pt", b"broken"), 2, steps=10)
    lineage.revert(2, 0.02)

    live = lineage.live_entry()
    assert Path(live.path).read_bytes() == b"healthy"
    # ... and no archive draw can reach the rejected weights either.
    for seed in range(50):
        sampled = lineage.sample_archive(random.Random(seed))
        if sampled is not None:
            assert Path(sampled.path).read_bytes() != b"broken"


def test_a_rollback_also_restores_the_optimizer_state(tmp_path: Path):
    """R1, second half: carrying the rejected update's momentum into the next
    step re-applies the update the floor just rejected."""

    lineage = SpecialistLineage(tmp_path, SCIENCE)
    lineage.bootstrap(_checkpoint(tmp_path / "g.pt", b"general"), 0)
    lineage.accept(_checkpoint(tmp_path / "c.pt", b"healthy"), 1, steps=10)
    lineage.optimizer_path.write_bytes(b"moments-good")
    lineage.mark_good(1, 0.42)
    lineage.accept(_checkpoint(tmp_path / "c2.pt", b"broken"), 2, steps=10)
    lineage.optimizer_path.write_bytes(b"moments-bad")

    lineage.revert(2, 0.02)
    assert lineage.optimizer_path.read_bytes() == b"moments-good"


def test_a_rollback_with_no_good_optimizer_snapshot_starts_cold(tmp_path: Path):
    """Absence is recoverable; the rejected moments are not."""

    lineage = SpecialistLineage(tmp_path, SCIENCE)
    lineage.bootstrap(_checkpoint(tmp_path / "g.pt", b"general"), 0)
    lineage.accept(_checkpoint(tmp_path / "c.pt", b"broken"), 1, steps=10)
    lineage.optimizer_path.write_bytes(b"moments-bad")
    lineage.revert(1, 0.01)
    assert not lineage.optimizer_path.exists()


def test_retrying_an_interrupted_iteration_does_not_train_twice(tmp_path: Path):
    """R4. Specialist updates commit inside `adapter.train`, before the
    controller commits the iteration, and the rollback hook restores GENERAL
    artifacts only. A retry re-runs generation from the same seeds, so the
    records are the same records -- and they have already been spent.
    """

    loop, records = _biased_run(tmp_path)
    first = loop.run_specialist_iteration(records, 21, general_inflow=200)
    science = next(row for row in first if row["class"] == "science")
    assert science["trained"] is True
    banked = loop.specialist_lineages["science"].load().update_count

    retry = loop.run_specialist_iteration(records, 21, general_inflow=200)
    again = next(row for row in retry if row["class"] == "science")
    assert again["trained"] is False
    assert again["skipped"] == "already trained this iteration"
    assert loop.specialist_lineages["science"].load().update_count == banked

    # A LATER iteration is not blocked by the guard.
    later = loop.run_specialist_iteration(records, 22, general_inflow=200)
    assert next(r for r in later if r["class"] == "science")["trained"] is True


def test_disabling_specialists_leaves_HOF_sampling_byte_identical(tmp_path: Path):
    """R7. `draw_opponent_class` consumed a random value even when HOF was the
    only possible class, shifting the stream `hof.sample` then reads -- so an
    existing HOF-enabled run would have selected different archived opponents at
    the same seed and iteration.
    """

    from .test_league_generation import add_archive, write_iterations

    loop = _loop(tmp_path, specialists="")
    for iteration in (3, 5, 8):
        add_archive(loop, iteration)

    for iteration in range(20, 40):
        got = loop.league_assignment(iteration, 500)
        # The pre-W7 path: one generator, keyed the same way, consumed first by
        # `hof.sample`.
        rng = random.Random(loop.config.seed + iteration * 100_003)
        expected = loop.hof.sample(rng, mode=loop.config.hof_sampling_mode)
        assert got.checkpoint == expected.path, f"iteration {iteration}"
        assert got.opponent_class == "hof"


def test_reanalysis_selection_does_not_depend_on_a_fresh_process(tmp_path: Path):
    """R8. The cap read `last_training_stats`, which holds the PREVIOUS
    iteration's number and on a fresh process holds nothing at all -- so the
    same inputs selected 25 rows on one run and 250 on a resume, both reporting
    themselves within a 25% cap.
    """

    loop, records = _biased_run(tmp_path, specialist_reanalysis=True)
    warm = loop.reanalysis_for_general(records, 21, general_inflow=40)[1]
    loop.last_training_stats = {}
    cold = loop.reanalysis_for_general(records, 21, general_inflow=40)[1]
    assert warm["positions"] == cold["positions"]
    assert warm["examples"] == cold["examples"]
    assert warm["general_inflow"] == cold["general_inflow"] == 40
    assert warm["examples"] <= int(40 * warm["cap"])


def test_the_reanalysis_teacher_is_the_learner_being_continued(tmp_path: Path):
    """Under the soft gate `train_candidate` fine-tunes from `latest.pt` for
    many iterations without a promotion; teaching the general from the protected
    best is then a stale target."""

    loop, records = _biased_run(tmp_path, specialist_reanalysis=True)
    latest = loop.checkpoint_dir / "latest.pt"
    shutil.copy2(loop.current_best, latest)
    stats = loop.reanalysis_for_general(
        records, 21, general_inflow=40, teacher=latest
    )[1]
    assert Path(stats["teacher"]).name == "latest.pt"


def test_a_zero_inflow_reanalysis_selects_nothing(tmp_path: Path):
    """R8, second half: `cap >= 1.0` and a zero inflow both short-circuited to
    "everything", so the two configurations that most needed a bound had none.
    """

    loop, records = _biased_run(tmp_path, specialist_reanalysis=True)
    _extra, stats = loop.reanalysis_for_general(records, 21, general_inflow=0)
    assert stats["examples"] == 0


def test_a_specialists_value_target_means_win_probability_not_utility(
    tmp_path: Path,
):
    """R2, at the point that matters: what `collate` actually hands the head.

    `value_soft` is a (win, draw, loss) distribution and the head that fits it
    is the one `search.py::_evaluate` reads back as `wdl[0] - wdl[2]`. A shaped
    root there teaches the bonus as win probability and the leaf bias then adds
    it again -- and because `root_value_unshaped` is computed from those same
    outputs, the distortion would reach the general too.
    """

    from .dataset import collate, examples_from_records

    loop, records = _biased_run(tmp_path)
    del loop
    own = examples_from_records(records, derived_for="specialist:1")
    mine = [
        e for e in own if e.target_route == "specialist:1" and e.root_value is not None
    ]
    assert mine, "no specialist row carried a bootstrap value at all"
    batch = collate(mine[:8])
    for row, example in enumerate(mine[:8]):
        if not batch["value_soft_valid"][row]:
            continue
        probability = float(batch["value_soft"][row, 0])
        assert 0.0 <= probability <= 1.0
        assert probability == pytest.approx(
            (1.0 + example.root_value) / 2.0, abs=1e-6
        )
        # The unshaped root is a win probability, so it lies in [-1, 1]; a
        # shaped one can exceed that and would clamp silently.
        assert -1.0 <= example.root_value <= 1.0
    assert all(not e.root_value_shaped for e in mine)


def test_reanalysis_reports_whether_it_funded_the_specialists_move(
    tmp_path: Path,
):
    """The plan wants the specialist's candidates given enough coverage that the
    general's weak prior cannot exclude them again. This measures whether that
    is a live problem before a mechanism is built for it: 7WD's median branching
    is 4 against a default top-k of 16.
    """

    loop, records = _biased_run(tmp_path, specialist_reanalysis=True)
    _extra, stats = loop.reanalysis_for_general(records, 21, general_inflow=400)
    if stats["positions"] == 0:
        pytest.skip("no position in this sample was selected")
    assert stats["coverage_positions"] > 0
    fraction = stats["specialist_move_visited_fraction"]
    assert fraction is not None and 0.0 <= fraction <= 1.0


def test_the_league_reports_what_it_costs(tmp_path: Path):
    """The plan's contract: measure end-to-end cost rather than calling the
    league free. Specialist training and the floor match get their own phases
    -- folded into `gate` the floor would read as the gate getting slower.
    """

    loop, records = _biased_run(tmp_path, anchor_games=2, gate_backend="rust")
    loop.run_specialist_iteration(records, 21, general_inflow=200)
    assert loop.phase_seconds["specialist_training"] > 0.0
    config = next(c for c in loop.specialist_configs if c.name == "science")
    result = loop.specialist_floor_check(config, 21, games=2)
    assert loop.phase_seconds["specialist_floor"] > 0.0
    assert result["seconds"] > 0.0
    # The general's own gate is a separate number, not inflated by the floor.
    assert "specialist_floor" != "gate"
    assert loop.phase_seconds.get("gate", 0.0) == 0.0


def test_seeding_earlier_than_league_play_is_refused():
    """The trap the first end-to-end smoke fell into.

    `specialist_bootstrap_games` gates seeding; `hof_start_games` gates
    `league_assignment` entirely, upstream of it. They coincide in the shipped
    config because the former defaults to following the latter, so the conflict
    is invisible until someone lowers one alone -- which is what "games before a
    specialist is seeded" invites.

    The failure is silent and expensive: lineage directories, an archived seed
    checkpoint, both specialists listed in the manifest with their lambdas, and
    not one biased game.
    """

    import pytest

    from .phase_d import PhaseDConfig

    common = dict(
        specialists="science:0.25:3",
        hierarchical_value=True,
        # The head is shadow-only, so an unweighted one is refused separately.
        hier_value_weight=0.1,
        hof_start_games=10_000,
    )
    with pytest.raises(ValueError, match="never be drawn as an opponent"):
        PhaseDConfig(specialist_bootstrap_games=24, **common).validate()

    # 0 means "follow hof_start_games" and is the shipped configuration.
    PhaseDConfig(specialist_bootstrap_games=0, **common).validate()
    # Equal or later is coherent: the specialist seeds once league play exists.
    PhaseDConfig(specialist_bootstrap_games=10_000, **common).validate()
    PhaseDConfig(specialist_bootstrap_games=20_000, **common).validate()
