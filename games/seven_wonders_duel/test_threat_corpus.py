"""The actor-created-threat corpus: rules agreement, and honest counting.

Written before the repairs they gate. Every defect in this pipeline shipped
because no test existed -- including two introduced by fixes for earlier defects
-- so these assert against the ENGINE rather than against the scanner's own
reconstruction of the rules, and against the two tables the plan already names.

`908370787` is the Workstream 9 reference case: `Discard for coins: Caravansery`
uncovers `r2c10`, and `School` sits one further removal away, which is why the
opponent's unbuilt extra-turn Wonder is what makes the threat immediate.
`907773062` is the forced-science review game.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from . import threat_corpus_episodes as episodes_mod
from . import threat_corpus_scan as scan_mod
from .advisor_scrape import determinize_observation, observation_from_wire
from .bga_extract import wire_from_bga_payload
from .data import CARDS_BY_NAME, CardColor
from .engine import _science_symbols, military_token_band
from .game import Phase

REFERENCE_TABLE, REFERENCE_ROW = "908370787", 17
SCIENCE_TABLE = "907773062"
LOG_DIR = scan_mod.REPO_ROOT / "runs/seven_wonders_duel/bga_game_log"

pytestmark = pytest.mark.skipif(
    not LOG_DIR.exists(), reason="BGA game logs are not present"
)


def load(table: str, row: int):
    path = LOG_DIR / f"table_{table}.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    decisions = [r for r in rows if r.get("kind") == "decision"]
    payload = wire_from_bga_payload(decisions[row]["state"])
    obs = observation_from_wire(payload["observation"])
    return determinize_observation(
        obs,
        random.Random(0),
        unknown_burial_ages=tuple(int(a) for a in payload.get("unknown_burial_ages", ())),
    )


# -- rules agreement --------------------------------------------------------


def test_science_symbols_include_progress_tokens():
    """`Law` grants a science symbol, so counting buildings alone can miss a
    sixth-symbol threat. Constructed rather than sampled: no logged city in the
    reference position holds `Law`, so a pure agreement check would pass
    vacuously against a scanner that ignored tokens entirely.
    """

    from .data import PROGRESS_BY_NAME

    game = load(REFERENCE_TABLE, REFERENCE_ROW)
    law = PROGRESS_BY_NAME["Law"]
    assert law.science is not None, "Law is the token that grants a symbol"

    city = game.cities[1]
    before = scan_mod.science_symbols(game, 1)
    assert law.science not in before, "pick a player without that symbol"
    city.progress_tokens.append("Law")
    after = scan_mod.science_symbols(game, 1)
    city.progress_tokens.remove("Law")

    assert after == before | {law.science}, (
        "science_symbols must count progress tokens, not just buildings"
    )
    # And it must be the engine's own answer, not a parallel implementation.
    city.progress_tokens.append("Law")
    assert scan_mod.science_symbols(game, 1) == _science_symbols(game, 1)
    city.progress_tokens.remove("Law")


def test_science_pair_respects_claimed_pairs_and_token_supply():
    """A second copy of a symbol only threatens if the pair is UNCLAIMED and a
    progress token is still available to take."""

    game = load(REFERENCE_TABLE, REFERENCE_ROW)
    opponent = 1 - scan_mod.state_actor(game)
    city = game.cities[opponent]

    green = next(
        c for c in CARDS_BY_NAME.values()
        if c.science is not None and c.science in {
            CARDS_BY_NAME[n].science for n in city.buildings
            if CARDS_BY_NAME[n].science is not None
        }
    ) if any(CARDS_BY_NAME[n].science for n in city.buildings) else None
    if green is None:
        pytest.skip("no repeatable symbol in this city")

    # Already claimed -> not a threat.
    city.claimed_science_pairs.add(green.science)
    assert scan_mod.threat_of(green.name, game, opponent) != "science_pair"
    city.claimed_science_pairs.discard(green.science)

    # No tokens left to take -> not a threat.
    saved, game.available_progress_tokens = game.available_progress_tokens, ()
    assert scan_mod.threat_of(green.name, game, opponent) != "science_pair"
    game.available_progress_tokens = saved


def test_military_uses_engine_bands_and_strategy():
    """Bands begin at 3 AND 6, only count while their token remains, and a red
    card is worth one more shield to a builder holding `Strategy`."""

    game = load(REFERENCE_TABLE, REFERENCE_ROW)
    opponent = 1 - scan_mod.state_actor(game)

    red = next(c for c in CARDS_BY_NAME.values()
               if c.color is CardColor.RED and c.shields >= 1)
    game.conflict_position = 0
    plain = scan_mod.shields_if_built(red, game, opponent)
    game.cities[opponent].progress_tokens.append("Strategy")
    assert scan_mod.shields_if_built(red, game, opponent) == plain + 1
    game.cities[opponent].progress_tokens.remove("Strategy")

    # The 3-band exists and the scanner must not treat only |6| as a band.
    assert military_token_band(3) is not None
    assert military_token_band(-3) is not None
    # A band whose token is gone is not a threat.
    direction = 1 if opponent == 0 else -1
    game.conflict_position = 2 * direction
    game.military_tokens_remaining.clear()
    assert scan_mod.threat_of(red.name, game, opponent) is None


# -- topology ---------------------------------------------------------------


def test_uncovered_matches_engine_accessibility():
    """`uncovered_by` must agree with the engine's own accessibility rule after
    the same removal -- duplicated logic is where divergence hides."""

    game = load(REFERENCE_TABLE, REFERENCE_ROW)
    tableau = game.tableau
    for slot_id, card in list(tableau.cards.items()):
        if not card.present or not tableau.is_accessible(slot_id):
            continue
        predicted = set(scan_mod.uncovered_by(tableau, slot_id))
        probe = game.clone()
        probe.tableau.cards[slot_id].present = False
        actual = {
            other
            for other, c in probe.tableau.cards.items()
            if c.present and probe.tableau.is_accessible(other)
        } - {o for o, c in tableau.cards.items() if c.present and tableau.is_accessible(o)}
        assert predicted == actual, f"removal of {slot_id}"


def test_reference_case_has_the_expected_shape():
    """The plan's own description, asserted: removing `Caravansery` uncovers
    `r2c10`, and `School` is one further removal beyond it."""

    game = load(REFERENCE_TABLE, REFERENCE_ROW)
    caravansery = next(
        slot for slot, c in game.tableau.cards.items()
        if c.present and c.revealed and c.card_name == "Caravansery"
    )
    uncovered = scan_mod.uncovered_by(game.tableau, caravansery)
    assert (1, 10) in uncovered

    reachable = dict(scan_mod.reach(game.tableau, (1, 10), 3))
    school = next(
        slot for slot, c in game.tableau.cards.items()
        if c.present and c.revealed and c.card_name == "School"
    )
    assert reachable.get(school) == 1, "School should sit one removal past r2c10"


# -- the chain / extra-turn rule --------------------------------------------


@pytest.mark.parametrize(
    "distance,needs",
    [(0, False), (1, True), (2, True), (3, True)],
)
def test_extra_turn_requirement_follows_distance_not_threat_class(distance, needs):
    """Distance 0: the opponent simply takes it next turn, no extra turn needed.
    Distance 1: one extra-turn Wonder uncovers AND takes. Distance 2+: one is
    not enough. This is independent of whether the threat is terminal -- an
    earlier version keyed it on threat class and got both ends backwards.
    """

    for threat in ("science_win", "military_win", "science_pair", "military_band"):
        episode = {"threat": threat, "distance": distance}
        assert episodes_mod.needs_extra_turn(episode) is needs


def test_distance_two_or_more_is_not_reachable_in_one_extra_turn():
    episode = {"threat": "science_win", "distance": 2}
    assert episodes_mod.resolvable_with_one_extra_turn(episode) is False
    assert episodes_mod.resolvable_with_one_extra_turn(
        {"threat": "science_win", "distance": 1}
    ) is True


# -- counting ---------------------------------------------------------------


def test_one_physical_run_is_one_episode():
    """Snapshots at different chain distances belong to ONE episode. Emitting
    them as separate entries inflated 82 runs into 134 reported episodes."""

    positions = [
        {
            "table": "T", "decision_row": row, "age": 2, "actor": 0,
            "opponent_unbuilt_extra_turn": ["The Sphinx"],
            "public_threats": [{
                "action_index": 1, "action_use": "discard_for_coins",
                "removes": [3, 9], "target": [1, 10], "target_card": "School",
                "distance": distance, "face_up": True, "threat": "science_pair",
            }],
            "hidden_slots_in_reach": 0,
        }
        for row, distance in ((10, 3), (11, 2), (12, 1))
    ]
    built = episodes_mod.build_episodes(positions)
    assert len(built) == 1, "three rows of one standoff are one episode"
    assert {s["distance"] for s in built[0]["snapshots"]} == {1, 2, 3}
    assert built[0]["episode_id"]


def test_duplicate_observations_are_collapsed():
    """The logs repeat identical observations; counting them twice inflates the
    corpus. `907773062` is the known offender."""

    path = LOG_DIR / f"table_{SCIENCE_TABLE}.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    decisions = [r for r in rows if r.get("kind") == "decision"]
    digests = [scan_mod.observation_digest(r["state"]) for r in decisions]
    assert len(set(digests)) < len(digests), "expected duplicates in this table"


def test_episodes_retain_exact_action_identity():
    """A later regret search needs the codec index and the card/Wonder, not just
    `action_use` -- different targets under one use are different moves."""

    positions = [{
        "table": "T", "decision_row": 5, "age": 2, "actor": 0,
        "opponent_unbuilt_extra_turn": [],
        "public_threats": [{
            "action_index": 742, "action_use": "construct_wonder",
            "action_label": "Wonder: The Sphinx (using Brewery)",
            "removes": [3, 9], "target": [1, 10], "target_card": "School",
            "distance": 0, "face_up": True, "threat": "science_pair",
        }],
        "hidden_slots_in_reach": 0,
    }]
    built = episodes_mod.build_episodes(positions)
    actions = built[0]["snapshots"][0]["creating_actions"]
    assert actions and actions[0]["action_index"] == 742
    assert "Sphinx" in actions[0]["action_label"]
