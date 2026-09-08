"""Workstream 5b: the action's consequence, not just its identity.

W5a scores an action from the contextual token of the card it acts on. That
says "this is a build of the Sawmill". It does not say "...and it uncovers
those two slots", which is the sentence every reviewed failure in the plan is
in -- table 908370787 is a burial that uncovered a threat.

The edge is reached through geometry already in the repo rather than a new
encoder feature: W5a knows the action's source token, W1 knows that token's
slot, and `slot_identity.covered_slots` knows which slots a slot covers. So it
costs two gathers, no schema change, and no Rust change.

The test that matters here is `test_it_finds_the_slots_the_layout_says_it
_should`. A branch that selected nothing would pass every neutrality and
gradient assertion in this file.
"""

import random

import pytest

torch = pytest.importorskip("torch")

from games.seven_wonders_duel.buffer import GameRecorder
from games.seven_wonders_duel.codec import legal_action_indices
from games.seven_wonders_duel.data import TABLEAU_LAYOUTS
from games.seven_wonders_duel.dataset import (
    TOKEN_TYPES,
    collate,
    examples_from_record,
)
from games.seven_wonders_duel.encoder import TABLEAU_FEATURES, TokenType
from games.seven_wonders_duel.game import Phase
from games.seven_wonders_duel.net import SWDNet
from games.seven_wonders_duel.slot_identity import (
    MAX_COVERED,
    AGE_SLOT_IDS,
    covered_slots,
)
from games.seven_wonders_duel.train import migrate_state_dict

_TABLEAU = TOKEN_TYPES.index(TokenType.TABLEAU)
_COVERERS = TABLEAU_FEATURES.index("coverers")
_SLOT_COORDINATES = {value: key for key, value in AGE_SLOT_IDS.items()}


@pytest.fixture(scope="module")
def batch():
    recorder = GameRecorder(11, agents={"p0": "random", "p1": "random"})
    rng = random.Random(7)
    while recorder.game.phase is not Phase.COMPLETE:
        recorder.play(rng.choice(legal_action_indices(recorder.game)))
    return collate(examples_from_record(recorder.finish()), contextual_actions=True)


def _model(**kwargs):
    torch.manual_seed(0)
    return SWDNet(d_model=32, layers=2, heads=4, action_residual=True, **kwargs)


# --- it finds the right slots -----------------------------------------------


def test_it_finds_the_slots_the_layout_says_it_should(batch):
    """Checked against `slot_identity.covered_slots`, which is itself checked
    against `data.covering_slots` -- so this compares the tensor path to the
    geometry, not to a second copy of itself."""

    model = _model(action_exposes=True)
    with torch.no_grad():
        tokens = model.embedder(batch)
        slot_ids = model.embedder.slot_ids(batch)
        covered_token, uncovers = model.action_scorer.uncovered_tokens(
            tokens, {**batch, "slot_ids": slot_ids}, slot_ids
        )

    checked = 0
    rows = batch["type_ids"].shape[0]
    for row in range(rows):
        sources = batch["action_source_indices"][row]
        present = batch["action_source_present"][row].bool()
        for action in range(sources.shape[0]):
            token = int(sources[action])
            if not bool(present[action]) or int(slot_ids[row, token]) == 0:
                # Not a tableau source: nothing can be uncovered by it, and the
                # branch must contribute nothing.
                assert not bool(uncovers[row, action].any())
                continue
            age, source_row, source_x = _SLOT_COORDINATES[
                int(slot_ids[row, token]) - 1
            ]
            layout = TABLEAU_LAYOUTS[age]
            within = layout.index(
                next(s for s in layout if (s.row, s.x) == (source_row, source_x))
            )
            expected = {j for j in covered_slots(age)[within] if j >= 0}

            for k in range(MAX_COVERED):
                if not bool(uncovers[row, action, k]):
                    continue
                found = int(covered_token[row, action, k])
                found_age, found_row, found_x = _SLOT_COORDINATES[
                    int(slot_ids[row, found]) - 1
                ]
                assert found_age == age
                found_within = layout.index(
                    next(s for s in layout if (s.row, s.x) == (found_row, found_x))
                )
                assert found_within in expected
                # And it is genuinely an UNCOVERING: this action's removal is
                # what makes that slot reachable.
                assert round(float(batch["features"][row, found, _COVERERS])) == 1
                checked += 1
    assert checked > 0, "no action in a whole game uncovered anything"


def test_a_doubly_covered_slot_is_not_counted(batch):
    """Two coverers and the card stays buried.

    The action is a step towards uncovering it, not an uncovering -- and
    treating those alike is the difference between "this hands them the sixth
    symbol" and "this might, eventually".
    """

    model = _model(action_exposes=True)
    with torch.no_grad():
        tokens = model.embedder(batch)
        slot_ids = model.embedder.slot_ids(batch)
        covered_token, uncovers = model.action_scorer.uncovered_tokens(
            tokens, {**batch, "slot_ids": slot_ids}, slot_ids
        )

    coverers = batch["features"][..., _COVERERS]
    selected = coverers.gather(
        1, covered_token.reshape(covered_token.shape[0], -1)
    ).reshape_as(covered_token)
    assert not bool((uncovers & (selected.round() != 1)).any())

    # And the mask is doing work: some covered slot in a whole game has two.
    assert bool((coverers.round() == 2).any())


# --- it is switch-neutral, and it trains ------------------------------------


def test_switching_it_on_reproduces_the_w5a_scorer(batch):
    plain = _model().eval()
    with torch.no_grad():
        before = plain(batch)

    exposed = _model(action_exposes=True)
    report = migrate_state_dict(plain.state_dict(), exposed)
    assert report["initialized"] == ["action_scorer.exposed.weight"]
    assert not exposed.action_scorer.exposed.weight.any()
    exposed.eval()
    with torch.no_grad():
        after = exposed(batch)
    for key, value in before.items():
        assert torch.equal(value, after[key]), key


def test_the_branch_trains_from_a_zero_output_projection(batch):
    """Zeroing the OUTPUT projection is neutral without being dead.

    A zeroed gate multiplying the whole branch would freeze it; a zeroed output
    projection still receives gradient, because the branch's own inputs and the
    error signal reaching it are both non-zero.
    """

    model = _model(action_exposes=True)
    model.train()
    model(batch)["action_policy"].square().mean().backward()
    grad = model.action_scorer.exposed.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_it_needs_the_action_residual_it_extends():
    """W5b is a branch of W5a's scorer, not a scorer of its own."""

    model = SWDNet(d_model=32, layers=2, heads=4, action_exposes=True)
    assert model.action_exposes is False
    assert model.action_scorer is None
