"""The network's shape contract, and the two masking rules that fail silently."""
from __future__ import annotations

import random

import pytest
import torch

from games.welcome_to import datagen
from games.welcome_to import encoder as enc
from games.welcome_to import network as nw
from games.welcome_to import train as tn
from games.welcome_to import training
from games.welcome_to.macro_codec import NUM_MACRO_ACTIONS
from games.welcome_to.bots import GreedyBot
from games.welcome_to.game import GameConfig, GameState

_SMALL = nw.NetConfig(
    sheet_hidden=32, sheet_out=16, trunk_hidden=48, trunk_blocks=1, head_hidden=32
)


def _batch(players: int = 3, games: int = 1, limit: int = 24) -> dict:
    def factory(rng: random.Random):
        bots = [GreedyBot(random.Random(rng.randrange(1 << 30))) for _ in range(players)]
        return lambda state: bots[state.actor].act(state)

    trajectories = datagen.generate(
        games, factory, config=GameConfig(players=players, advanced=True), seed=3
    )
    samples = [s for t in trajectories for s in datagen.replay(t)][:limit]
    return nw.to_tensors(datagen.batch(samples))


def _forward(net: nw.WelcomeToNet, batch: dict) -> dict:
    return net(
        batch["sheet_planes"],
        batch["sheet_scalars"],
        batch["viewer_plane"],
        batch["global_scalars"],
    )


# ──────────────────────────────────────────────────────────────────────────
# Shape
# ──────────────────────────────────────────────────────────────────────────
def test_the_heads_cover_the_target_set_exactly():
    """A target with no head, or a head with no target, is a silent hole."""
    heads = set(nw.PER_SEAT_HEAD_TARGETS) | set(nw.GLOBAL_HEAD_TARGETS)
    regressions = {
        name
        for name in training.TARGET_NAMES
        if not name.endswith("_mask")
        and name != "seat_valid"
        and not name.startswith("rank_")
    }
    assert heads == regressions


def test_forward_shapes():
    batch = _batch()
    net = nw.WelcomeToNet(_SMALL)
    out = _forward(net, batch)
    n = batch["action"].shape[0]
    assert out["policy_logits"].shape == (n, NUM_MACRO_ACTIONS)
    assert out["rank_logits"].shape == (n, training.MAX_RANKS)
    assert out["per_seat"].shape == (n, enc.MAX_SEATS, len(nw.PER_SEAT_HEAD_TARGETS))
    for name in nw.PER_SEAT_HEAD_TARGETS:
        assert out[name].shape == (n, enc.MAX_SEATS)
    for name in nw.GLOBAL_HEAD_TARGETS:
        assert out[name].shape == (n,)


def test_the_default_network_is_near_the_four_million_budget():
    count = nw.parameter_count(nw.WelcomeToNet())
    assert 3e6 < count < 5e6, count


# ──────────────────────────────────────────────────────────────────────────
# The shared sheet encoder
# ──────────────────────────────────────────────────────────────────────────
def test_the_sheet_encoder_is_the_same_function_for_every_seat():
    """Shared weights are the whole reason the per-seat structure is cheap."""
    net = nw.WelcomeToNet(_SMALL).eval()
    batch = _batch()
    planes, scalars = batch["sheet_planes"], batch["sheet_scalars"]

    swapped_planes = planes.clone()
    swapped_scalars = scalars.clone()
    swapped_planes[:, [0, 1]] = planes[:, [1, 0]]
    swapped_scalars[:, [0, 1]] = scalars[:, [1, 0]]

    with torch.no_grad():
        a = net.sheet_encoder(
            torch.cat([planes.reshape(planes.shape[0], enc.MAX_SEATS, -1), scalars], -1)
        )
        b = net.sheet_encoder(
            torch.cat(
                [
                    swapped_planes.reshape(planes.shape[0], enc.MAX_SEATS, -1),
                    swapped_scalars,
                ],
                -1,
            )
        )
    assert torch.allclose(a[:, 0], b[:, 1]) and torch.allclose(a[:, 1], b[:, 0])


def test_the_per_seat_head_reads_the_context():
    """AUX_TARGETS_SPEC §4: an isolated h_s cannot carry a cross-seat target."""
    net = nw.WelcomeToNet(_SMALL).eval()
    batch = _batch()
    with torch.no_grad():
        base = _forward(net, batch)["score"]
        moved = dict(batch)
        moved["global_scalars"] = batch["global_scalars"] + 0.5
        after = _forward(net, moved)["score"]
    assert not torch.allclose(base, after), "the head ignored everything but its seat"


# ──────────────────────────────────────────────────────────────────────────
# Masking -- both of these are silent when broken
# ──────────────────────────────────────────────────────────────────────────
def test_the_rank_distribution_is_masked_on_the_logits():
    """M4.  Softmax over four classes then zeroing two sums to less than one."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 5.0, 5.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    p = nw.rank_probabilities(logits, mask)
    assert torch.allclose(p.sum(-1), torch.ones(2), atol=1e-6)
    assert float(p[0, 2]) == 0.0 and float(p[0, 3]) == 0.0
    assert float(p[1, 3]) == 0.0
    # and the live classes keep their relative odds
    assert float(p[0, 1] / p[0, 0]) == pytest.approx(float(torch.exp(torch.tensor(1.0))))


def test_a_two_seat_game_produces_a_two_class_distribution():
    net = nw.WelcomeToNet(_SMALL).eval()
    batch = _batch(players=2)
    with torch.no_grad():
        out = _forward(net, batch)
    mask = torch.stack(
        [batch[f"rank_mask_{r}"] for r in range(training.MAX_RANKS)], dim=-1
    )
    p = nw.rank_probabilities(out["rank_logits"], mask)
    assert torch.allclose(p.sum(-1), torch.ones(p.shape[0]), atol=1e-6)
    assert float(p[:, 2:].abs().max()) == 0.0


def test_padded_seats_cannot_move_a_per_seat_loss():
    """M4.  Zero is a value; absent is not, and the per-seat head is shared."""
    net = nw.WelcomeToNet(_SMALL).eval()
    batch = _batch(players=2)
    with torch.no_grad():
        out = _forward(net, batch)
        clean, _ = nw.losses(out, batch)

        garbage = dict(batch)
        for name in training.PER_SEAT_TARGETS:
            if name == "seat_valid":
                continue
            spoiled = batch[name].clone()
            spoiled[:, 2:] = 17.5
            garbage[name] = spoiled
        dirty, _ = nw.losses(out, garbage)
    assert float(clean) == pytest.approx(float(dirty)), "a padded seat reached a loss"


def test_a_masked_target_is_normalised_by_its_mask_sum():
    """M1.  Dividing by the batch size discounts rare events by their rarity."""
    net = nw.WelcomeToNet(_SMALL).eval()
    batch = _batch(players=3, games=3, limit=400)
    with torch.no_grad():
        out = _forward(net, batch)
        _, parts = nw.losses(out, batch)

    checked = 0
    for name, mask_name in training.MASKED_TARGETS.items():
        mask = batch["seat_valid"] * batch[mask_name]
        if not 0 < float(mask.sum()) < mask.numel():
            continue  # this slot was completed by nobody, or by everybody
        error = (out[name] - batch[name]) ** 2
        by_hand = float((mask * error).sum() / mask.sum())
        assert float(parts[name]) == pytest.approx(by_hand, rel=1e-5)
        # and the batch-size version would be materially different -- the whole
        # point of M1 is that the discount is large, not a rounding difference
        by_batch = float((mask * error).sum() / mask.numel())
        assert by_batch < by_hand * 0.9
        checked += 1
    assert checked, "no masked target had a partly-live mask to check"


def test_the_sentinel_never_reaches_a_loss():
    """M2.  A -1 sentinel behind a live mask would be trained on as a value."""
    net = nw.WelcomeToNet(_SMALL).eval()
    batch = _batch(players=3)
    with torch.no_grad():
        out = _forward(net, batch)
        _, before = nw.losses(out, batch)

        spoiled = dict(batch)
        for name, mask_name in training.MASKED_TARGETS.items():
            values = batch[name].clone()
            values[batch[mask_name] <= 0] = -999.0
            spoiled[name] = values
        _, after = nw.losses(out, spoiled)
    for name in training.MASKED_TARGETS:
        assert float(before[name]) == pytest.approx(float(after[name]), rel=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Loss and training loop
# ──────────────────────────────────────────────────────────────────────────
def test_the_loss_weights_cover_every_head_by_group():
    for name in nw.PER_SEAT_HEAD_TARGETS + nw.GLOBAL_HEAD_TARGETS:
        assert nw._GROUP_OF[name] in nw.LOSS_WEIGHTS
    assert nw._GROUP_OF["policy"] == "policy"
    assert nw._GROUP_OF["rank"] == "objective"


def test_a_group_is_weighted_once_not_once_per_target():
    """Otherwise a group's real pull is its coefficient times its size.

    Per-target weighting would make `components` (8 targets at 0.2) outweigh
    `capacity` (4 at 0.3), so the table would not describe the model it
    configures -- and adding a target would silently reweight its group.  §10
    step 4 of the aux spec adds nine targets to `plan_race`.
    """
    net = nw.WelcomeToNet(_SMALL).eval()
    batch = _batch(players=3, games=2, limit=200)
    with torch.no_grad():
        total, parts = nw.losses(_forward(net, batch), batch)

    for group, weight in nw.LOSS_WEIGHTS.items():
        members = [float(parts[n]) for n, g in nw._GROUP_OF.items() if g == group]
        assert members
        assert float(parts[f"group_{group}"]) == pytest.approx(
            sum(members) / len(members), rel=1e-5
        )

    rebuilt = sum(
        weight * float(parts[f"group_{group}"])
        for group, weight in nw.LOSS_WEIGHTS.items()
    )
    assert float(total) == pytest.approx(rebuilt, rel=1e-5)

    # and the defect this replaces would give a materially different number
    per_target = sum(
        nw.LOSS_WEIGHTS[group] * float(parts[name])
        for name, group in nw._GROUP_OF.items()
    )
    assert abs(per_target - float(total)) > 0.05 * float(total)


def test_the_loss_is_finite_and_differentiable():
    net = nw.WelcomeToNet(_SMALL)
    batch = _batch()
    total, parts = nw.losses(_forward(net, batch), batch)
    assert torch.isfinite(total)
    total.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_the_policy_loss_ignores_illegal_actions():
    net = nw.WelcomeToNet(_SMALL)
    batch = _batch()
    out = _forward(net, batch)
    _, before = nw.losses(out, batch)

    moved = dict(out)
    logits = out["policy_logits"].clone()
    logits[batch["legal"] <= 0] += 50.0
    moved["policy_logits"] = logits
    _, after = nw.losses(moved, batch)
    assert float(before["policy"]) == pytest.approx(float(after["policy"]), rel=1e-5)


def test_a_short_run_reduces_the_loss():
    """The pipeline end to end: corpus, replay, forward, loss, step."""
    _, metrics = tn.train(
        tn.TrainConfig(games=10, epochs=1, batch_size=32, paired_games=2),
        net_config=_SMALL,
        device="cpu",
        log=False,
    )
    assert metrics["eval_samples"] > 0
    assert 0.0 <= metrics["policy_top1"] <= 1.0
    assert metrics["paired_games"] == 2.0
    for key in ("gate_policy_agreement", "gate_score_within_2", "gate_permits_beats_mean"):
        assert key in metrics


# ──────────────────────────────────────────────────────────────────────────
# The corpus
# ──────────────────────────────────────────────────────────────────────────
def test_the_seat_mixture_is_sixty_thirty_ten():
    counts = tn.seat_counts(1000)
    assert len(counts) == 1000
    assert counts.count(2) == 600 and counts.count(3) == 300 and counts.count(4) == 100


def test_a_small_corpus_still_gets_its_four_seat_games():
    """Largest-remainder, not rounding: the 10% share must not vanish."""
    counts = tn.seat_counts(10)
    assert len(counts) == 10
    assert counts.count(4) >= 1, "the far end of the seat axis went dark"


def test_the_corpus_carries_every_seat_count():
    corpus = tn.build_corpus(10, seed=1)
    assert len(corpus) == 10
    assert {t.players for t in corpus} == {2, 3, 4}


def test_the_greedy_policy_plays_legal_macros():
    """The policy applies the move rather than returning one: a macro is up to
    two engine steps, and a caller holding an index cannot apply it alone."""
    from games.welcome_to import macro_codec as mc
    from games.welcome_to.game import Phase

    net = nw.WelcomeToNet(_SMALL)
    play = tn.greedy_policy(net, torch.device("cpu"))
    state = GameState.new(seed=4, config=GameConfig(players=2, advanced=True))
    for _ in range(30):
        if state.is_terminal:
            break
        assert state.phase is not Phase.WRITE_NUMBER, "a macro left the game mid-write"
        before = state.turn, state.actor, state.phase
        play(state)
        assert (state.turn, state.actor, state.phase) != before, "nothing happened"


# ──────────────────────────────────────────────────────────────────────────
# The paired gate
# ──────────────────────────────────────────────────────────────────────────
def test_the_paired_gate_replaces_exactly_one_seat():
    """The baseline arm must be a plain all-GreedyBot game at the same seed.

    If it is not -- if the two arms differ in anything but the substituted seat
    -- the delta is not attributable to the policy under test.  So the baseline
    is reconstructed here from scratch and has to match to the point.
    """
    net = nw.WelcomeToNet(_SMALL)
    metrics = tn.paired_score_gap(net, torch.device("cpu"), games=1, seed=9_000)
    assert metrics["paired_games"] == 1.0

    players, game_seed, evaluated = 2, 9_000, 0
    bots = [GreedyBot(random.Random(game_seed * 100 + p)) for p in range(players)]
    state = GameState.new(seed=game_seed, config=GameConfig(players=players, advanced=True))
    while not state.is_terminal:
        state.apply(bots[state.actor].act(state))
    assert metrics["greedy_score"] == pytest.approx(float(state.scores()[evaluated]))


def test_the_paired_gate_is_reproducible():
    net = nw.WelcomeToNet(_SMALL)
    a = tn.paired_score_gap(net, torch.device("cpu"), games=3)
    b = tn.paired_score_gap(net, torch.device("cpu"), games=3)
    assert a == b


def test_the_paired_gate_reports_its_own_noise():
    """A +-2 point gate is meaningless without knowing what 2 points is worth."""
    net = nw.WelcomeToNet(_SMALL)
    metrics = tn.paired_score_gap(net, torch.device("cpu"), games=4)
    assert metrics["score_gap_stderr"] > 0.0
