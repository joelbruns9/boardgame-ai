"""Falsify the encoder's feature claims against games that were actually played.

The engine is now checked against an authority that is not us (see
``bga_differential.py``). The encoder is not. Its ~65 derived features are
*claims about the position*, and the discard-pile bug -- a card revivable with
an unbuilt Mausoleum was invisible to "can this player still win by science" --
was found by accident, from one game that happened to be lost in a memorable
way. That is not a method. This is the method.

**The idea.** Most features are claims, and a claim can be contradicted. Three
kinds of contradiction, in increasing power:

1. *What the game went on to do.* ``sci_missing_obtainable`` says at most N
   more distinct science symbols are still gettable; a player who then collects
   N+1 has falsified it. Likewise ``mil_shields_obtainable``, and the two
   ``*_win_feasible`` flags against how the game actually ended.
2. *What the engine actually did on the next move.* The per-card claims
   (``affordable``, ``eff_shields``, ``would_cross_token``,
   ``would_win_military``) predict the consequence of taking that card, so
   taking it settles them.
3. *What a search can reach.* The bounds are claims about every continuation,
   so a directed line that beats one has disproved it -- and because the claim
   covers all continuations, the opponent is allowed to cooperate, which makes
   the search cheap and its counterexamples real lines rather than estimates.

**Why (3) exists.** (1) was the obvious design and it is far too weak. With the
original discard-pile bug deliberately reintroduced, 60 games and 7,396
retrospective checks found *nothing*: a science victory needs a chain of luck
on top of the wrong claim, so the claim survives. Hunting the bound directly
finds it from 20 games. ``test_encoder_audit.py`` keeps that honest by putting
the bug back and requiring the audit to rediscover it.

**On circularity.** Re-running the same formula on the same state proves
nothing. The encoder never sees the ``GameState`` -- it rebuilds a stub from
the observation -- so the checks here run the *engine's own helpers* on the
stub and on the real state and compare (``card_cost``, ``stub_economy``,
``score_block``). There is no second formula to disagree with: any difference
is the reconstruction losing something, which is exactly the class of bug the
discard-pile one belonged to. What this audit does not test is a rule the
engine gets wrong in the first place; that is the BGA differential harness.

**Coverage is the weak point, not soundness.** A claim is only falsified if
something contradicts it, so the corpus is played by bots chosen to reach the
endings that matter: science and military rushes, which random play almost
never produces (measured: 1 military and 0 science wins in 46 random games).
The report says how many games ended each way and how many checks of each kind
ran, so a run that never tested science victories cannot be mistaken for one
that did.

    python -m games.seven_wonders_duel.encoder_audit --games 200 --hunt 6
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .bots import (
    GreedyBot,
    MilitaryAggressiveBot,
    MilitaryEconomyBot,
    RandomBot,
    ScienceAggressiveBot,
    ScienceEconomyBot,
)
from .data import (
    CARD_IDS,
    CARDS_BY_NAME,
    PROGRESS_BY_NAME,
    WONDERS_BY_NAME,
    CardColor,
    EffectKind,
    ScienceSymbol,
)
from .encoder import (
    GLOBAL_FEATURES,
    TABLEAU_FEATURES,
    TokenType,
    _stub_state,  # deliberately: the stub IS what this audit is testing
    encode,
)
from .engine import (
    Action,
    ActionUse,
    _choice_producers,
    _fixed_production,
    _opponent_trade_production,
    _trade_discounts,
    apply_action,
    legal_actions,
    minimum_payment,
    score_player,
)
from .game import GameState, Phase, VictoryType, new_game
from .rules import discard_income

_GLOBAL_INDEX = {name: i for i, name in enumerate(GLOBAL_FEATURES)}
_TABLEAU_INDEX = {name: i for i, name in enumerate(TABLEAU_FEATURES)}
_CARD_BY_ID = {index: name for name, index in CARD_IDS.items()}


def make_bot(name: str, seed: int) -> Any:
    """Build a bot by name. Some take a seed, the deterministic ones do not."""

    factory = BOT_FACTORIES[name]
    try:
        return factory(seed)
    except TypeError:
        return factory()


BOT_FACTORIES: dict[str, Callable[..., Any]] = {
    "random": RandomBot,
    "greedy": GreedyBot,
    "science_aggressive": ScienceAggressiveBot,
    "science_economy": ScienceEconomyBot,
    "military_aggressive": MilitaryAggressiveBot,
    "military_economy": MilitaryEconomyBot,
}

# Pairings the corpus is played with. Science and military rushes are here to
# reach the endings the feasibility claims are about; the mirrors matter too,
# because a rush that is *contested* is what drives a player to the edge of
# still-possible without going over.
DEFAULT_PAIRINGS: tuple[tuple[str, str], ...] = (
    ("science_aggressive", "greedy"),
    ("greedy", "science_aggressive"),
    ("science_aggressive", "science_economy"),
    ("military_aggressive", "greedy"),
    ("greedy", "military_aggressive"),
    ("military_aggressive", "military_economy"),
    ("science_aggressive", "military_aggressive"),
    ("greedy", "random"),
)


@dataclass
class Violation:
    """One falsified claim, with enough to reproduce it."""

    check: str
    seed: int
    ply: int
    seat: int
    detail: str

    def __str__(self) -> str:
        return f"[{self.check}] seed={self.seed} ply={self.ply} seat={self.seat}: {self.detail}"


@dataclass
class AuditReport:
    games: int = 0
    plies: int = 0
    checks: Counter = field(default_factory=Counter)
    violations: list[Violation] = field(default_factory=list)
    endings: Counter = field(default_factory=Counter)

    def record(self, check: str, ok: bool, **kwargs) -> None:
        self.checks[check] += 1
        if not ok:
            self.violations.append(Violation(check=check, **kwargs))


CHECKS = (
    "sci_win_feasible",
    "mil_win_feasible",
    "sci_missing_obtainable",
    "mil_shields_obtainable",
    "score_block",
    "global_block",
    "stub_economy",
    "card_cost",
    "card_affordable",
    "card_eff_shields",
    "card_would_win_military",
    "card_would_cross_token",
    "hunt_sci_missing_obtainable",
    "hunt_mil_shields_obtainable",
)


# --------------------------------------------------------------------------
# reading the encoding back out
# --------------------------------------------------------------------------


def _global_features(encoding) -> dict[str, float]:
    token = next(t for t in encoding.tokens if t.type is TokenType.GLOBAL)
    return dict(zip(GLOBAL_FEATURES, token.features))


def _tableau_features(encoding) -> dict[str, dict[str, float]]:
    """Per-card tableau features, keyed by card name (face-down cards have none)."""

    out: dict[str, dict[str, float]] = {}
    for token in encoding.tokens:
        if token.type is not TokenType.TABLEAU:
            continue
        name = _CARD_BY_ID.get(token.entity_id)
        if name is not None:  # a face-down card encodes no per-card claims
            out[name] = dict(zip(TABLEAU_FEATURES, token.features))
    return out


def _seat_prefix(seat: int, actor: int) -> str:
    return "my_" if seat == actor else "opp_"


# --------------------------------------------------------------------------
# what the real game did, measured from the real GameState
# --------------------------------------------------------------------------


def science_symbols(game: GameState, seat: int) -> frozenset:
    """Distinct science symbols held, from buildings and progress tokens."""

    found = set()
    city = game.cities[seat]
    for name in city.buildings:
        symbol = CARDS_BY_NAME[name].science
        if symbol is not None:
            found.add(symbol)
    for name in city.progress_tokens:
        symbol = PROGRESS_BY_NAME[name].science
        if symbol is not None:
            found.add(symbol)
    return frozenset(found)


def added_shields(game: GameState, seat: int, card_name: str) -> int:
    """Shields a card gives this seat, counting Strategy as the encoder does."""

    card = CARDS_BY_NAME[card_name]
    shields = card.shields
    if card.color is CardColor.RED and "Strategy" in game.cities[seat].progress_tokens:
        shields += 1
    return shields


def relative_position(game: GameState, seat: int) -> int:
    return game.conflict_position if seat == 0 else -game.conflict_position


# --------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------


@dataclass
class _Claim:
    """One position's claims, checked once the rest of the game is known."""

    ply: int
    seat: int
    sci_win_feasible: float
    mil_win_feasible: float
    sci_missing_obtainable: float
    mil_shields_obtainable: float
    symbols_then: frozenset


def audit_game(
    seed: int,
    bots: tuple[Any, Any],
    report: AuditReport,
    *,
    max_plies: int = 400,
    hunt_attempts: int = 0,
    hunt_positions: int = 2,
) -> None:
    """Play one game, checking every claim the encoder makes along the way."""

    game = new_game(seed)
    claims: list[_Claim] = []
    # Shields each seat adds from here on, indexed by ply, filled as we go.
    shields_at_ply: list[tuple[int, int, int]] = []  # (ply, seat, shields)
    # Positions where an impossibility was claimed, kept for the hunt. Ranked
    # rather than sampled evenly: a claim made when the player is one symbol or
    # one shield short is the one most likely to be wrong, and the cheapest to
    # disprove if it is.
    hunt_candidates: list[tuple[float, int, GameState, dict, int]] = []
    ply = 0

    while game.phase is not Phase.COMPLETE and ply < max_plies:
        actor = (
            game.pending_choice.player
            if game.pending_choice is not None
            else game.active_player
        )
        observation = game.observation(actor)
        encoding = encode(observation)
        globals_ = _global_features(encoding)
        cards = _tableau_features(encoding)
        _check_stub_pricing(game, observation, cards, actor, seed, ply, report)
        _check_stub_economy(game, observation, seed, ply, report)
        _check_global_block(game, globals_, actor, seed, ply, report)

        for seat in (0, 1):
            prefix = _seat_prefix(seat, actor)
            claims.append(
                _Claim(
                    ply=ply,
                    seat=seat,
                    sci_win_feasible=globals_[f"{prefix}sci_win_feasible"],
                    mil_win_feasible=globals_[f"{prefix}mil_win_feasible"],
                    sci_missing_obtainable=globals_[f"{prefix}sci_missing_obtainable"],
                    mil_shields_obtainable=globals_[f"{prefix}mil_shields_obtainable"],
                    symbols_then=science_symbols(game, seat),
                )
            )
            _check_score_block(game, globals_, prefix, seed, ply, seat, report)

        if hunt_attempts:
            # Rank by how tight the claim is: a bound of 1 is disproved by one
            # lucky card, a bound of 5 needs a whole run of them. Taken over the
            # BEST seat, since one falsified claim is enough and the hunt covers
            # both seats anyway; nudged toward positions where the discard pile
            # holds something wanted, the route the known bug was blind to.
            tightness = max(
                -min(
                    globals_[f"{_seat_prefix(seat, actor)}sci_missing_obtainable"],
                    globals_[f"{_seat_prefix(seat, actor)}mil_shields_obtainable"],
                )
                + (0.5 if _discard_holds_wanted(game, seat, "scientific") else 0.0)
                for seat in (0, 1)
            )
            hunt_candidates.append(
                (tightness, ply, game.clone(), dict(globals_), actor)
            )

        action = bots[actor].select_action(game)
        _check_card_claims(game, cards, actor, action, seed, ply, report)

        # Measured before the move, and counting wonders as well as cards: the
        # bound this feeds is over everything the seat can still add, and
        # reading it afterwards would price the card under a Strategy token the
        # same move may just have granted.
        gained_shields = _shields_from_action(game, actor, action)
        before = _Snapshot.of(game, actor)
        apply_action(game, action)
        after = _Snapshot.of(game, actor)
        _check_transition(
            game, cards, actor, action, before, after, seed, ply, report
        )
        if gained_shields:
            shields_at_ply.append((ply, actor, gained_shields))
        ply += 1

    report.games += 1
    report.plies += ply
    report.endings[game.victory_type.value if game.victory_type else "unfinished"] += 1
    _check_feasibility_against_outcome(game, claims, shields_at_ply, seed, report)

    if hunt_attempts and hunt_candidates:
        rng = random.Random(seed)
        hunt_candidates.sort(key=lambda row: -row[0])
        for _, hunt_ply, position, globals_, actor in hunt_candidates[:hunt_positions]:
            hunt_claimed_bounds(
                position,
                globals_,
                actor,
                seed,
                hunt_ply,
                report,
                attempts=hunt_attempts,
                rng=rng,
            )


@dataclass(frozen=True)
class _Snapshot:
    """The handful of quantities a transition check compares."""

    coins: tuple[int, int]
    conflict: int
    tokens_remaining: frozenset
    victory: VictoryType | None
    winner: int | None
    slot_names: dict

    @classmethod
    def of(cls, game: GameState, actor: int) -> "_Snapshot":
        return cls(
            coins=(game.cities[0].coins, game.cities[1].coins),
            conflict=game.conflict_position,
            tokens_remaining=frozenset(game.military_tokens_remaining),
            victory=game.victory_type,
            winner=game.winner,
            slot_names={
                slot_id: card.card_name
                for slot_id, card in game.tableau.cards.items()
                if card.card_name is not None
            },
        )


def _check_score_block(
    game: GameState,
    globals_: dict[str, float],
    prefix: str,
    seed: int,
    ply: int,
    seat: int,
    report: AuditReport,
) -> None:
    """The score features are computed off a stub rebuilt from the observation.

    Comparing them to ``score_player`` on the REAL state is the check that the
    reconstruction kept everything the score depends on -- the failure mode a
    recomputation on the same stub could never show.
    """

    real = score_player(game, seat)
    expected = {
        "score_military": real.military,
        "score_buildings": real.buildings,
        "score_guild": real.guild,
        "score_wonders": real.wonders,
        "score_progress": real.progress,
        "score_treasury": real.treasury,
        "score_total": real.total,
        "score_blue": real.blue_buildings,
    }
    wrong = [
        f"{name}: encoder={globals_[prefix + name]:g} real={want}"
        for name, want in expected.items()
        if globals_[prefix + name] != want
    ]
    report.record(
        "score_block",
        not wrong,
        seed=seed,
        ply=ply,
        seat=seat,
        detail="; ".join(wrong),
    )


def _check_stub_economy(
    game: GameState,
    observation,
    seed: int,
    ply: int,
    report: AuditReport,
) -> None:
    """Production and trade prices, computed on the stub and for real.

    Same argument as the pricing check and the same power: these are engine
    helpers, so running both sides through them compares *states*, not
    formulas. They are worth their own check because every price in the game
    is downstream of them -- a stub that loses a brown card makes every cost
    feature wrong at once, in a way that still looks internally consistent.
    """

    stub = _stub_state(observation)
    for seat in (0, 1):
        wrong = []
        for label, helper in (
            ("fixed_production", _fixed_production),
            ("choice_producers", _choice_producers),
            ("opponent_trade_production", _opponent_trade_production),
            ("trade_discounts", _trade_discounts),
        ):
            on_stub, for_real = helper(stub, seat), helper(game, seat)
            if on_stub != for_real:
                wrong.append(f"{label}: stub={on_stub} real={for_real}")
        report.record(
            "stub_economy",
            not wrong,
            seed=seed,
            ply=ply,
            seat=seat,
            detail="; ".join(wrong),
        )


def _check_global_block(
    game: GameState,
    globals_: dict[str, float],
    actor: int,
    seed: int,
    ply: int,
    report: AuditReport,
) -> None:
    """The plainly-readable per-player features, against the real state.

    These are simple counts and flags, so this is not a deep check of a rule --
    it is a check that the observation and the stub still carry what the feature
    describes. That is the failure this whole audit is about: the encoder never
    sees the ``GameState``, only a reconstruction of it.
    """

    for seat in (0, 1):
        prefix = _seat_prefix(seat, actor)
        city = game.cities[seat]
        symbols = science_symbols(game, seat)
        colors = Counter(CARDS_BY_NAME[name].color for name in city.buildings)
        remaining = dict(game.military_tokens_remaining)
        sign = 1 if seat == 0 else -1

        expected = {
            f"{prefix}coins": city.coins,
            f"{prefix}sci_distinct": len(symbols),
            f"{prefix}sci_to_win": max(0, 6 - len(symbols)),
            f"{prefix}token_2coin_remaining": 1.0 if sign * 3 in remaining else 0.0,
            f"{prefix}token_5coin_remaining": 1.0 if sign * 6 in remaining else 0.0,
            f"{prefix}discard_income": discard_income(colors[CardColor.YELLOW]),
        }
        for symbol in ScienceSymbol:
            expected[f"{prefix}sym_{symbol.value}"] = 1.0 if symbol in symbols else 0.0
        for color in CardColor:
            expected[f"{prefix}color_{color.value}"] = colors[color]

        wrong = [
            f"{name}: encoder={globals_[name]:g} real={want:g}"
            for name, want in expected.items()
            if name in globals_ and globals_[name] != want
        ]
        report.record(
            "global_block",
            not wrong,
            seed=seed,
            ply=ply,
            seat=seat,
            detail="; ".join(wrong),
        )

    # Actor-framed track features: military is signed toward the actor, and the
    # two distances are what the feasibility flags are compared against.
    military = relative_position(game, actor)
    track = {
        "military": military,
        "dist_my_mil_win": 9 - military,
        "dist_opp_mil_win": 9 + military,
        "pending_shields": game.pending_shields,
        "pending_extra_turn": 1.0 if game.pending_extra_turn else 0.0,
    }
    wrong = [
        f"{name}: encoder={globals_[name]:g} real={want:g}"
        for name, want in track.items()
        if globals_[name] != want
    ]
    report.record(
        "global_block",
        not wrong,
        seed=seed,
        ply=ply,
        seat=actor,
        detail="; ".join(wrong),
    )


def _check_stub_pricing(
    game: GameState,
    observation,
    cards: dict[str, dict[str, float]],
    actor: int,
    seed: int,
    ply: int,
    report: AuditReport,
) -> None:
    """Price every visible card twice: off the encoder's stub, and for real.

    The encoder does not read the ``GameState``. It rebuilds a stub from the
    observation and calls the engine's own pricing on that, so anything the
    observation drops or the stub fails to carry comes out as a wrong price --
    quietly, in a feature the net then trains on. Running the *same helper* on
    both states is what makes this check formula-free: it cannot be fooled by a
    shared mistake in a formula, because there is no second formula, and any
    disagreement is squarely the stub's.

    This is the same class of bug as the discard-pile one: not a rule the engine
    got wrong, but a fact the encoder could not see.
    """

    stub = _stub_state(observation)
    for name in cards:
        card = CARDS_BY_NAME[name]
        for seat in (0, 1):
            on_stub = minimum_payment(stub, seat, card.cost, card=card)
            for_real = minimum_payment(game, seat, card.cost, card=card)
            wrong = []
            if on_stub.total_coins != for_real.total_coins:
                wrong.append(
                    f"cost stub={on_stub.total_coins} real={for_real.total_coins}"
                )
            if on_stub.used_chain != for_real.used_chain:
                wrong.append(
                    f"chain stub={on_stub.used_chain} real={for_real.used_chain}"
                )
            # The feature the net actually sees, as opposed to the pricing call.
            prefix = _seat_prefix(seat, actor)
            claimed = cards[name][f"{prefix}affordable"]
            really = 1.0 if game.cities[seat].coins >= for_real.total_coins else 0.0
            if claimed != really:
                wrong.append(f"affordable encoder={claimed:g} real={really:g}")
            report.record(
                "card_cost",
                not wrong,
                seed=seed,
                ply=ply,
                seat=seat,
                detail=f"{name}: " + "; ".join(wrong),
            )


def _check_card_claims(
    game: GameState,
    cards: dict[str, dict[str, float]],
    actor: int,
    action: Action,
    seed: int,
    ply: int,
    report: AuditReport,
) -> None:
    """``affordable`` claims about the card the game is about to build."""

    if action.use is not ActionUse.CONSTRUCT_BUILDING or action.slot_id is None:
        return
    card = game.tableau.cards.get(action.slot_id)
    name = None if card is None else card.card_name
    if name is None or name not in cards:
        return
    features = cards[name]
    prefix = _seat_prefix(actor, actor)
    # The engine only offers legal actions, so a build that happens is a build
    # the actor could pay for. `affordable` claiming otherwise is a real
    # disagreement about this seat's purchasing power.
    report.record(
        "card_affordable",
        features[f"{prefix}affordable"] == 1.0,
        seed=seed,
        ply=ply,
        seat=actor,
        detail=f"{name}: affordable=0 but the engine allowed the build",
    )


def _check_transition(
    game: GameState,
    cards: dict[str, dict[str, float]],
    actor: int,
    action: Action,
    before: _Snapshot,
    after: _Snapshot,
    seed: int,
    ply: int,
    report: AuditReport,
) -> None:
    """Per-card predictions against what the engine actually did."""

    if action.use is not ActionUse.CONSTRUCT_BUILDING or action.slot_id is None:
        return
    name = before.slot_names.get(action.slot_id)
    if name is None or name not in cards:
        return
    features = cards[name]
    prefix = _seat_prefix(actor, actor)

    # Shields: the pawn moves by exactly what the card was claimed to be worth.
    claimed_shields = features[f"{prefix}eff_shields"]
    moved = abs(after.conflict - before.conflict)
    # The pawn stops at 9; a claim beyond the end of the track is not wrong.
    capped = min(claimed_shields, 9 - relative_position_from(before.conflict, actor))
    report.record(
        "card_eff_shields",
        moved == capped,
        seed=seed,
        ply=ply,
        seat=actor,
        detail=f"{name}: claimed {claimed_shields:g} shields, pawn moved {moved}",
    )

    # "This card wins the game by military" -- checkable the moment it is played.
    claims_win = features[f"{prefix}would_win_military"] == 1.0
    really_won = (
        after.victory is VictoryType.MILITARY
        and after.winner == actor
        and before.victory is None
    )
    report.record(
        "card_would_win_military",
        claims_win == really_won,
        seed=seed,
        ply=ply,
        seat=actor,
        detail=(
            f"{name}: would_win_military={int(claims_win)} but the move "
            f"{'did' if really_won else 'did not'} end the game by military"
        ),
    )

    # "This card crosses the next military token", which is what makes the
    # opponent pay. A token leaving the board is the unambiguous evidence: the
    # coin loss itself can read as 0 against an opponent who has no coins.
    claims_cross = features[f"{prefix}would_cross_token"] == 1.0
    token_taken = bool(before.tokens_remaining - after.tokens_remaining)
    report.record(
        "card_would_cross_token",
        claims_cross == token_taken,
        seed=seed,
        ply=ply,
        seat=actor,
        detail=(
            f"{name}: would_cross_token={int(claims_cross)}, "
            f"tokens taken={sorted(before.tokens_remaining - after.tokens_remaining)}"
        ),
    )


def relative_position_from(conflict: int, seat: int) -> int:
    return conflict if seat == 0 else -conflict


def _check_feasibility_against_outcome(
    game: GameState,
    claims: list[_Claim],
    shields_at_ply: list[tuple[int, int, int]],
    seed: int,
    report: AuditReport,
) -> None:
    """The retrospective half: what the rest of the game says about each claim.

    Every claim here is about the future, so it can only be judged once the
    future has happened. ``*_feasible = 0`` asserts impossibility over every
    continuation, so the game contradicting it once is enough; the two
    ``obtainable`` features are upper bounds, contradicted by the player going
    on to collect more than the bound allowed.
    """

    final_symbols = {seat: science_symbols(game, seat) for seat in (0, 1)}
    for claim in claims:
        seat = claim.seat
        won_science = (
            game.victory_type is VictoryType.SCIENTIFIC and game.winner == seat
        )
        won_military = game.victory_type is VictoryType.MILITARY and game.winner == seat
        report.record(
            "sci_win_feasible",
            not (claim.sci_win_feasible == 0.0 and won_science),
            seed=seed,
            ply=claim.ply,
            seat=seat,
            detail=(
                "encoder said a science win was impossible; the player then "
                "won by science"
            ),
        )
        report.record(
            "mil_win_feasible",
            not (claim.mil_win_feasible == 0.0 and won_military),
            seed=seed,
            ply=claim.ply,
            seat=seat,
            detail=(
                "encoder said a military win was impossible; the player then "
                "won by military"
            ),
        )
        gained = len(final_symbols[seat] - claim.symbols_then)
        report.record(
            "sci_missing_obtainable",
            gained <= claim.sci_missing_obtainable,
            seed=seed,
            ply=claim.ply,
            seat=seat,
            detail=(
                f"bound said at most {claim.sci_missing_obtainable:g} new symbols "
                f"were still obtainable; the player gained {gained}"
            ),
        )
        later = sum(
            shields for ply, s, shields in shields_at_ply if s == seat and ply >= claim.ply
        )
        report.record(
            "mil_shields_obtainable",
            later <= claim.mil_shields_obtainable,
            seed=seed,
            ply=claim.ply,
            seat=seat,
            detail=(
                f"bound said at most {claim.mil_shields_obtainable:g} shields were "
                f"still obtainable; the player added {later}"
            ),
        )


# --------------------------------------------------------------------------
# active falsification: hunt a line that beats what the encoder said was possible
# --------------------------------------------------------------------------
#
# Waiting for a game to contradict a claim only tests the claims that game
# happened to touch, and that turned out to be far too weak: with the original
# discard-pile bug deliberately reintroduced, 60 games and 7,396 retrospective
# checks found nothing. The reason is that requiring a science *victory* asks
# for a whole chain of luck, while the wrong claim underneath it is much
# smaller -- a bound on what the player can still obtain.
#
# So the hunt targets the bound directly. ``sci_missing_obtainable`` says at
# most N more distinct symbols are still gettable; a line that gets N+1 has
# falsified it, whatever the game goes on to do. Same for
# ``mil_shields_obtainable``. These are the two features the feasibility flags
# are computed from, so a wrong flag is a wrong bound one step earlier, and the
# bound is falsifiable orders of magnitude more often.
#
# The opponent cooperates, which is legitimate: the bound is a claim about
# every continuation, so the largest search space is the right one. A hunt that
# finds nothing proves nothing -- it is a search, not a proof. A hunt that finds
# something is a real line, played by the real engine, and the claim is wrong.


def _missing_symbols(game: GameState, seat: int) -> set:
    return set(ScienceSymbol) - science_symbols(game, seat)


def _discard_holds_wanted(game: GameState, seat: int, kind: str) -> bool:
    """Is there something in the discard pile this seat would want revived?

    This is the route the original bug was blind to, so the hunt is told about
    it explicitly: the point is to test the encoder, not to admire the search.
    """

    if kind == "scientific":
        lacking = _missing_symbols(game, seat)
        return any(CARDS_BY_NAME[n].science in lacking for n in game.discard_pile)
    return any(CARDS_BY_NAME[n].shields for n in game.discard_pile)


def _wonder_opens_a_route(game: GameState, seat: int, wonder_name: str, kind: str) -> bool:
    """Wonders that reach cards or tokens ordinary play cannot."""

    wonder = WONDERS_BY_NAME.get(wonder_name)
    if wonder is None:
        return False
    for effect in wonder.effects:
        if effect.kind is EffectKind.BUILD_FROM_DISCARD_FREE:
            return _discard_holds_wanted(game, seat, kind)
        if effect.kind is EffectKind.CHOOSE_UNUSED_PROGRESS and kind == "scientific":
            return True  # the Great Library can still reach Law
    return False


def _card_gain(game: GameState, seat: int, card_name: str | None, kind: str) -> float:
    if card_name is None:
        return 0.0
    card = CARDS_BY_NAME[card_name]
    if kind == "scientific":
        return 100.0 if card.science in _missing_symbols(game, seat) else 0.0
    return 100.0 * card.shields


def _target_score(game: GameState, seat: int, action: Action, kind: str) -> float:
    """How much this action advances ``seat``'s haul of symbols or shields."""

    if action.use is ActionUse.CONSTRUCT_BUILDING and action.slot_id is not None:
        card = game.tableau.cards.get(action.slot_id)
        gain = _card_gain(game, seat, None if card is None else card.card_name, kind)
        if gain:
            return gain
        return 1.0
    if action.use is ActionUse.RESOLVE_PENDING_CHOICE and action.choice is not None:
        # Covers both shapes of choice: a card revived out of the discard, and a
        # progress token drawn from the box.
        if action.choice in CARDS_BY_NAME:
            gain = _card_gain(game, seat, action.choice, kind)
            if gain:
                return gain
        if kind == "scientific" and action.choice == "Law":
            return 100.0  # Law is a science symbol in progress-token form
        return 1.0
    if action.use is ActionUse.CONSTRUCT_WONDER and action.wonder_name is not None:
        return 60.0 if _wonder_opens_a_route(game, seat, action.wonder_name, kind) else 5.0
    return 1.0


def _helper_score(game: GameState, target: int, action: Action, kind: str) -> float:
    """The opponent cooperating: get out of the way, and do not end the game.

    Cooperation is legitimate here precisely because the claim under test says
    *no* continuation exists. It also makes the hunt far stronger than adversarial
    play would: it searches the largest space the claim rules out.
    """

    helper = 1 - target
    name = None
    if action.slot_id is not None and action.use is ActionUse.CONSTRUCT_BUILDING:
        card = game.tableau.cards.get(action.slot_id)
        name = None if card is None else card.card_name

    if name is not None:
        card = CARDS_BY_NAME[name]
        if kind == "scientific":
            if card.science is not None and card.science in _missing_symbols(game, target):
                return -100.0  # taking what the target still needs
        elif card.shields:
            # Shields for the helper push the pawn the wrong way and can end the
            # game early, before the target ever completes its own route.
            return -50.0
    if action.use is ActionUse.DISCARD_FOR_COINS and action.slot_id is not None:
        discarded = game.tableau.cards.get(action.slot_id)
        wanted = None if discarded is None else CARDS_BY_NAME[discarded.card_name].science
        if kind == "scientific" and wanted is not None and wanted in _missing_symbols(game, target):
            # Discarding is not always removal: a Mausoleum holder can build it
            # back out of the discard, which is the exact route the original bug
            # was blind to. Mildly discouraged, not forbidden.
            return -10.0
    if relative_position(game, helper) >= 7:
        return -25.0  # close to a military win of its own; stay off the track
    return 1.0


def hunt_bound(
    game: GameState,
    seat: int,
    kind: str,
    *,
    attempts: int = 8,
    rng: random.Random | None = None,
    max_plies: int = 200,
) -> tuple[int, str]:
    """Best haul found for ``seat``: new distinct symbols, or shields added.

    Returns ``(best, description)``. Directed cooperative play with restarts --
    a lower bound on what is really obtainable, which is all a falsifier needs:
    anything it actually achieves, the encoder's upper bound had better allow.
    """

    rng = rng or random.Random(0)
    start_symbols = science_symbols(game, seat)
    best = 0
    best_line = ""
    for attempt in range(attempts):
        trial = game.clone()
        shields = 0
        plies = 0
        while trial.phase is not Phase.COMPLETE and plies < max_plies:
            actor = (
                trial.pending_choice.player
                if trial.pending_choice is not None
                else trial.active_player
            )
            actions = legal_actions(trial)
            if not actions:
                break
            scored = [
                (
                    (
                        _target_score(trial, seat, action, kind)
                        if actor == seat
                        else _helper_score(trial, seat, action, kind)
                    )
                    # Enough noise to explore, not enough to drown the direction.
                    + rng.random() * 3.0,
                    index,
                )
                for index, action in enumerate(actions)
            ]
            _, index = max(scored)
            action = actions[index]
            if actor == seat:
                shields += _shields_from_action(trial, seat, action)
            apply_action(trial, action)
            plies += 1

        got = (
            len(science_symbols(trial, seat) - start_symbols)
            if kind == "scientific"
            else shields
        )
        if got > best:
            best = got
            unit = "new symbols" if kind == "scientific" else "shields"
            best_line = (
                f"a line exists giving seat {seat} {got} {unit} "
                f"({plies} plies, attempt {attempt + 1})"
            )
    return best, best_line


def _shields_from_action(game: GameState, seat: int, action: Action) -> int:
    """Shields this action gives ``seat``, counted as the encoder's bound does."""

    if action.use is ActionUse.CONSTRUCT_BUILDING and action.slot_id is not None:
        card = game.tableau.cards.get(action.slot_id)
        if card is not None and card.card_name is not None:
            return added_shields(game, seat, card.card_name)
    elif action.use is ActionUse.RESOLVE_PENDING_CHOICE and action.choice in CARDS_BY_NAME:
        return added_shields(game, seat, action.choice)
    elif action.use is ActionUse.CONSTRUCT_WONDER and action.wonder_name is not None:
        wonder = WONDERS_BY_NAME.get(action.wonder_name)
        return 0 if wonder is None else wonder.shields
    return 0


def hunt_claimed_bounds(
    game: GameState,
    globals_: dict[str, float],
    actor: int,
    seed: int,
    ply: int,
    report: AuditReport,
    *,
    attempts: int,
    rng: random.Random,
) -> None:
    """Try to beat the two obtainable bounds the encoder claims at this position."""

    for seat in (0, 1):
        prefix = _seat_prefix(seat, actor)
        for kind, feature in (
            ("scientific", "sci_missing_obtainable"),
            ("military", "mil_shields_obtainable"),
        ):
            claimed = globals_[f"{prefix}{feature}"]
            got, line = hunt_bound(game, seat, kind, attempts=attempts, rng=rng)
            report.record(
                f"hunt_{feature}",
                got <= claimed,
                seed=seed,
                ply=ply,
                seat=seat,
                detail=f"bound said {claimed:g}, but {line}",
            )


def run(
    games: int,
    pairings: Iterable[tuple[str, str]] = DEFAULT_PAIRINGS,
    *,
    seed0: int = 0,
    hunt_attempts: int = 0,
    hunt_positions: int = 2,
) -> AuditReport:
    report = AuditReport()
    pairings = list(pairings)
    for index in range(games):
        left, right = pairings[index % len(pairings)]
        seed = seed0 + index
        bots = (make_bot(left, seed), make_bot(right, seed + 10_000))
        audit_game(
            seed,
            bots,
            report,
            hunt_attempts=hunt_attempts,
            hunt_positions=hunt_positions,
        )
    return report


def format_report(report: AuditReport) -> str:
    lines = [
        f"{report.games} games, {report.plies} decision points",
        "endings: "
        + ", ".join(f"{kind}={count}" for kind, count in sorted(report.endings.items())),
        "",
        "claims checked:",
    ]
    for check in CHECKS:
        count = report.checks[check]
        bad = sum(1 for v in report.violations if v.check == check)
        flag = "   <- NOT EXERCISED" if not count else (f"   <- {bad} FALSIFIED" if bad else "")
        lines.append(f"   {check:<26}{count:>8}{flag}")

    lines.append("")
    if report.violations:
        lines.append(f"!! {len(report.violations)} FALSIFIED CLAIMS")
        seen: Counter = Counter()
        for violation in report.violations:
            seen[violation.check] += 1
            if seen[violation.check] <= 3:  # a bug repeats; three is enough to fix it
                lines.append(f"   {violation}")
        for check, count in seen.items():
            if count > 3:
                lines.append(f"   ... and {count - 3} more [{check}]")
    else:
        lines.append("no claim was contradicted by any game played")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument(
        "--hunt",
        type=int,
        default=0,
        help=(
            "attempts per hunted claim (0 = retrospective checks only). A hunt "
            "searches for a continuation contradicting an 'impossible' claim"
        ),
    )
    parser.add_argument(
        "--hunt-positions",
        type=int,
        default=2,
        help="positions per game to hunt, the closest calls first",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bots",
        default=None,
        help=(
            "pairing as left:right (default: a rotation over science and "
            "military rushes, which are the endings the claims are about)"
        ),
    )
    args = parser.parse_args(argv)

    if args.bots:
        left, right = args.bots.split(":", 1)
        pairings = [(left.strip(), right.strip())]
    else:
        pairings = list(DEFAULT_PAIRINGS)

    report = run(
        args.games,
        pairings,
        seed0=args.seed,
        hunt_attempts=args.hunt,
        hunt_positions=args.hunt_positions,
    )
    print(format_report(report))
    return 1 if report.violations else 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
