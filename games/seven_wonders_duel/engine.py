"""Base-game action resolution, victory conditions, and final scoring."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from itertools import product

from .data import (
    CARDS_BY_NAME,
    PROGRESS_BY_NAME,
    PROGRESS_IDS,
    TABLEAU_LAYOUTS,
    WONDER_IDS,
    WONDERS_BY_NAME,
    BackType,
    CardColor,
    CardData,
    Cost,
    EffectKind,
    ScienceSymbol,
    back_type_of,
)
from .game import (
    ChanceKind,
    GameState,
    HiddenInformationError,
    PendingChoice,
    PendingChoiceKind,
    Phase,
    ResolvedChance,
    SlotId,
    StepResult,
    TableauState,
    VictoryType,
)
from .rules import Resource, discard_income, normal_trade_unit_cost


class ActionUse(str, Enum):
    DRAFT_WONDER = "draft_wonder"
    CONSTRUCT_BUILDING = "construct_building"
    DISCARD_FOR_COINS = "discard_for_coins"
    CONSTRUCT_WONDER = "construct_wonder"
    RESOLVE_PENDING_CHOICE = "resolve_pending_choice"
    CHOOSE_NEXT_START_PLAYER = "choose_next_start_player"


@dataclass(frozen=True, slots=True)
class Action:
    slot_id: SlotId | None
    use: ActionUse
    wonder_name: str | None = None
    choice: str | None = None
    starting_player: int | None = None


@dataclass(frozen=True, slots=True)
class Payment:
    total_coins: int
    printed_coins: int
    trade_coins: int
    purchased: tuple[tuple[Resource, int], ...]
    used_chain: bool = False


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    military: int
    buildings: int  # all card VP including guilds
    wonders: int
    progress: int
    treasury: int
    total: int
    blue_buildings: int
    guild: int = 0  # guild-card VP alone (subset of buildings)


_RESOURCES = tuple(Resource)


def _cards_for_city(game: GameState, player: int) -> tuple[CardData, ...]:
    return tuple(CARDS_BY_NAME[name] for name in game.cities[player].buildings)


def _fixed_production(game: GameState, player: int) -> Counter[Resource]:
    produced: Counter[Resource] = Counter()
    for card in _cards_for_city(game, player):
        produced.update(card.fixed_production)
    return produced


def _choice_producers(game: GameState, player: int) -> tuple[tuple[Resource, ...], ...]:
    choices = [card.choice_production for card in _cards_for_city(game, player) if card.choice_production]
    choices.extend(
        WONDERS_BY_NAME[name].choice_production
        for name in game.cities[player].built_wonders
        if WONDERS_BY_NAME[name].choice_production
    )
    return tuple(choices)


def _opponent_trade_production(game: GameState, player: int) -> Counter[Resource]:
    produced: Counter[Resource] = Counter()
    for card in _cards_for_city(game, 1 - player):
        if card.color in (CardColor.BROWN, CardColor.GREY):
            produced.update(card.fixed_production)
    return produced


def _trade_discounts(game: GameState, player: int) -> frozenset[Resource]:
    discounted: set[Resource] = set()
    for card in _cards_for_city(game, player):
        discounted.update(card.trade_discount)
    return frozenset(discounted)


def _chain_is_free(game: GameState, player: int, card: CardData) -> bool:
    return bool(
        card.chain_from
        and any(existing.chain_to == card.chain_from for existing in _cards_for_city(game, player))
    )


def _discount_allocations(cost: Cost, rebate: int):
    counts = tuple(cost.resource_count(resource) for resource in _RESOURCES)
    for allocation in product(*(range(count + 1) for count in counts)):
        if sum(allocation) <= rebate:
            yield allocation


def minimum_payment(
    game: GameState,
    player: int,
    cost: Cost,
    *,
    card: CardData | None = None,
    is_wonder: bool = False,
) -> Payment:
    """Find the cheapest legal payment across rebates and flexible production."""

    if card is not None and _chain_is_free(game, player, card):
        return Payment(0, 0, 0, (), used_chain=True)

    rebate = 0
    tokens = set(game.cities[player].progress_tokens)
    if is_wonder and "Architecture" in tokens:
        rebate = 2
    elif card is not None and card.color is CardColor.BLUE and "Masonry" in tokens:
        rebate = 2

    fixed = _fixed_production(game, player)
    choice_producers = _choice_producers(game, player)
    opponent = _opponent_trade_production(game, player)
    discounts = _trade_discounts(game, player)
    best: Payment | None = None

    allocations = _discount_allocations(cost, rebate)
    choice_assignments = product(*choice_producers) if choice_producers else [()]
    # Materialize because it is traversed once per possible rebate allocation.
    choice_assignments = tuple(choice_assignments)
    for allocation in allocations:
        requirements = {
            resource: cost.resource_count(resource) - reduction
            for resource, reduction in zip(_RESOURCES, allocation, strict=True)
        }
        for assignment in choice_assignments:
            produced = fixed.copy()
            produced.update(assignment)
            purchased = tuple(
                (resource, max(0, requirements[resource] - produced[resource]))
                for resource in _RESOURCES
                if requirements[resource] > produced[resource]
            )
            trade = sum(
                quantity
                * (1 if resource in discounts else normal_trade_unit_cost(opponent[resource]))
                for resource, quantity in purchased
            )
            candidate = Payment(
                total_coins=cost.coins + trade,
                printed_coins=cost.coins,
                trade_coins=trade,
                purchased=purchased,
            )
            if best is None or (candidate.total_coins, candidate.purchased) < (
                best.total_coins,
                best.purchased,
            ):
                best = candidate
    if best is None:
        raise AssertionError("payment search produced no candidate")
    return best


def _can_afford(game: GameState, player: int, payment: Payment) -> bool:
    return game.cities[player].coins >= payment.total_coins


def _unbuilt_wonders(game: GameState, player: int) -> tuple[str, ...]:
    city = game.cities[player]
    return tuple(
        name
        for name in city.wonders
        if name not in city.built_wonders and name not in game.retired_wonders
    )


def legal_actions(game: GameState) -> tuple[Action, ...]:
    if game.pending_choice is not None:
        return tuple(
            Action(None, ActionUse.RESOLVE_PENDING_CHOICE, choice=option)
            for option in game.pending_choice.options
        )
    if game.phase is Phase.WONDER_DRAFT:
        return tuple(
            Action(None, ActionUse.DRAFT_WONDER, wonder_name=name)
            for name in game.legal_wonder_choices()
        )
    if game.phase is Phase.CHOOSE_NEXT_START_PLAYER:
        return (
            Action(None, ActionUse.CHOOSE_NEXT_START_PLAYER, starting_player=0),
            Action(None, ActionUse.CHOOSE_NEXT_START_PLAYER, starting_player=1),
        )
    if game.phase is not Phase.PLAY_AGE:
        return ()
    player = game.active_player
    actions: list[Action] = []
    for slot_id in game.tableau.accessible_slot_ids():
        card = CARDS_BY_NAME[game.tableau.cards[slot_id].card_name]
        payment = minimum_payment(game, player, card.cost, card=card)
        if _can_afford(game, player, payment):
            actions.append(Action(slot_id, ActionUse.CONSTRUCT_BUILDING))
        actions.append(Action(slot_id, ActionUse.DISCARD_FOR_COINS))
        for wonder_name in _unbuilt_wonders(game, player):
            wonder = WONDERS_BY_NAME[wonder_name]
            if wonder.cost is None:
                raise AssertionError(f"missing Wonder cost: {wonder_name}")
            wonder_payment = minimum_payment(game, player, wonder.cost, is_wonder=True)
            if _can_afford(game, player, wonder_payment):
                actions.append(Action(slot_id, ActionUse.CONSTRUCT_WONDER, wonder_name))
    return tuple(actions)


class _ChanceCtx:
    """Collects chance events for one action and mediates outcome resolution.

    Three modes (CODEC_SPEC.md §4.3): simulator (no outcomes supplied, no
    barrier) resolves from the locked deal / seeded RNG; search-with-outcomes
    consumes the supplied outcomes in event order; barred (search_barrier set,
    nothing supplied) raises HiddenInformationError instead of leaking.
    """

    __slots__ = ("game", "supplied", "events")

    def __init__(self, game: GameState, chance_outcomes) -> None:
        self.game = game
        self.supplied = None if chance_outcomes is None else list(chance_outcomes)
        self.events: list[ResolvedChance] = []

    def draw(self, kind: ChanceKind):
        """Return the externally supplied outcome for the next event, or None
        when the simulator's own randomness is allowed to resolve it."""

        if self.supplied is not None:
            if not self.supplied:
                raise HiddenInformationError(
                    f"chance_outcomes exhausted before {kind.value} event"
                )
            return self.supplied.pop(0)
        if self.game.search_barrier:
            raise HiddenInformationError(
                f"search barrier: {kind.value} requires an explicit outcome"
            )
        return None

    def result(self) -> StepResult:
        if self.supplied:
            raise ValueError(
                f"{len(self.supplied)} unconsumed chance outcome(s) supplied"
            )
        return StepResult(events=tuple(self.events))


def _override_reveal(game: GameState, slot_id: SlotId, new_name: str) -> None:
    """Assign a supplied reveal outcome to a slot, swapping the previously
    locked card into the outcome card's hidden location so the complete state
    stays a consistent world (no duplicate, no lost card)."""

    slot_card = game.tableau.cards[slot_id]
    old_name = slot_card.card_name
    if new_name == old_name:
        return
    if new_name not in CARDS_BY_NAME:
        raise ValueError(f"unknown card in reveal outcome: {new_name}")
    if back_type_of(new_name) is not back_type_of(old_name):
        raise ValueError(
            f"reveal outcome {new_name} does not match back type of slot {slot_id}"
        )
    for candidate in game.tableau.cards.values():
        if candidate.present and not candidate.revealed and candidate.card_name == new_name:
            candidate.card_name = old_name
            slot_card.card_name = new_name
            return
    removed = game.removed_age_cards.get(game.age, ())
    if new_name in removed:
        game.removed_age_cards[game.age] = tuple(
            old_name if name == new_name else name for name in removed
        )
        slot_card.card_name = new_name
        return
    if new_name in game.unused_guilds:
        game.unused_guilds = tuple(
            old_name if name == new_name else name for name in game.unused_guilds
        )
        game.selected_guilds = tuple(
            new_name if name == old_name else name for name in game.selected_guilds
        )
        slot_card.card_name = new_name
        return
    raise ValueError(f"reveal outcome is not in the unseen pool: {new_name}")


def _process_reveals(ctx: _ChanceCtx, newly_accessible: tuple[SlotId, ...]) -> None:
    """Resolve reveals sequentially (spec §4.2). Each slot stays unrevealed
    until its own event fires, so a supplied outcome can swap with a card
    locked in a sibling slot exposed by the same take, and a barrier violation
    raises before any hidden identity is marked visible."""

    game = ctx.game
    for slot_id in newly_accessible:
        back = back_type_of(game.tableau.cards[slot_id].card_name)
        supplied = ctx.draw(ChanceKind.CARD_REVEAL)
        if supplied is not None:
            _override_reveal(game, slot_id, supplied)
        game.tableau.reveal(slot_id)
        ctx.events.append(
            ResolvedChance(
                kind=ChanceKind.CARD_REVEAL,
                context=(slot_id, back),
                outcome=game.tableau.cards[slot_id].card_name,
            )
        )


def _override_wonder_flip(game: GameState, supplied) -> None:
    outcome = tuple(supplied)
    pool = tuple(game.wonder_groups[1]) + tuple(game.unused_wonders)
    if len(outcome) != 4 or len(set(outcome)) != 4 or any(w not in pool for w in outcome):
        raise ValueError(f"invalid wonder-flip outcome: {outcome}")
    game.wonder_offer = list(outcome)
    game.wonder_groups = (game.wonder_groups[0], outcome)
    game.unused_wonders = tuple(w for w in pool if w not in outcome)


def _pay(game: GameState, player: int, payment: Payment) -> None:
    city = game.cities[player]
    if not _can_afford(game, player, payment):
        raise ValueError("player cannot afford construction")
    city.coins -= payment.total_coins
    opponent = game.cities[1 - player]
    if payment.trade_coins and "Economy" in opponent.progress_tokens:
        opponent.coins += payment.trade_coins


def _count_color(game: GameState, player: int, color: CardColor) -> int:
    return sum(CARDS_BY_NAME[name].color is color for name in game.cities[player].buildings)


def _apply_card_coin_effects(game: GameState, player: int, card: CardData) -> None:
    city = game.cities[player]
    for effect in card.effects:
        if effect.kind is EffectKind.IMMEDIATE_COINS:
            city.coins += effect.amount
        elif effect.kind is EffectKind.COINS_PER_OWN_COLOR:
            if effect.color is None:
                raise AssertionError("color-count effect is missing a color")
            city.coins += effect.amount * _count_color(game, player, effect.color)
        elif effect.kind is EffectKind.COINS_PER_OWN_WONDER:
            city.coins += effect.amount * len(city.built_wonders)
        elif effect.kind is EffectKind.COINS_PER_MOST_COLOR:
            if effect.color is None:
                raise AssertionError("guild color effect is missing a color")
            city.coins += effect.amount * max(
                _count_color(game, 0, effect.color), _count_color(game, 1, effect.color)
            )
        elif effect.kind is EffectKind.COINS_PER_MOST_BROWN_GREY:
            counts = [
                _count_color(game, p, CardColor.BROWN) + _count_color(game, p, CardColor.GREY)
                for p in (0, 1)
            ]
            city.coins += effect.amount * max(counts)


def _apply_progress_immediate(game: GameState, player: int, token_name: str) -> None:
    token = PROGRESS_BY_NAME[token_name]
    for effect in token.effects:
        if effect.kind is EffectKind.IMMEDIATE_COINS:
            game.cities[player].coins += effect.amount


def _science_symbols(game: GameState, player: int) -> set[ScienceSymbol]:
    symbols: set[ScienceSymbol] = set()
    for name in game.cities[player].buildings:
        symbol = CARDS_BY_NAME[name].science
        if symbol is not None:
            symbols.add(symbol)
    for name in game.cities[player].progress_tokens:
        symbol = PROGRESS_BY_NAME[name].science
        if symbol is not None:
            symbols.add(symbol)
    return symbols


def _declare_victory(game: GameState, player: int, victory_type: VictoryType) -> None:
    game.winner = player
    game.victory_type = victory_type
    game.phase = Phase.COMPLETE


def _check_scientific_victory(game: GameState, player: int) -> bool:
    if len(_science_symbols(game, player)) >= 6:
        _declare_victory(game, player, VictoryType.SCIENTIFIC)
        return True
    return False


def _apply_science_building(game: GameState, player: int, card: CardData) -> None:
    symbol = card.science
    if symbol is None or _check_scientific_victory(game, player):
        return
    city = game.cities[player]
    copies = sum(CARDS_BY_NAME[name].science is symbol for name in city.buildings)
    if copies >= 2 and symbol not in city.claimed_science_pairs:
        city.claimed_science_pairs.add(symbol)
        _set_pending_if_options(
            game,
            PendingChoiceKind.CHOOSE_AVAILABLE_PROGRESS,
            player,
            game.available_progress_tokens,
        )


def _apply_military(game: GameState, player: int, shields: int) -> None:
    direction = 1 if player == 0 else -1
    opponent = game.cities[1 - player]
    for _ in range(shields):
        game.conflict_position += direction
        penalty = game.military_tokens_remaining.pop(game.conflict_position, None)
        if penalty is not None:
            opponent.coins = max(0, opponent.coins - penalty)
        if abs(game.conflict_position) == 9:
            _declare_victory(game, player, VictoryType.MILITARY)
            return


def _after_building_constructed(game: GameState, player: int, card: CardData) -> None:
    _apply_card_coin_effects(game, player, card)
    shields = card.shields
    if card.color is CardColor.RED and "Strategy" in game.cities[player].progress_tokens:
        shields += 1
    if shields:
        _apply_military(game, player, shields)
    if game.phase is not Phase.COMPLETE and card.science is not None:
        _apply_science_building(game, player, card)


def _finish_turn(game: GameState, player: int, extra_turn: bool) -> None:
    if game.phase is Phase.COMPLETE:
        return
    if game.pending_choice is not None:
        game.pending_extra_turn = extra_turn
        return
    if game.tableau.accessible_slot_ids():
        game.active_player = player if extra_turn else 1 - player
    elif game.age == 3:
        _resolve_civilian_endgame(game)
    else:
        game.phase = Phase.CHOOSE_NEXT_START_PLAYER
        if game.conflict_position > 0:
            game.active_player = 1
        elif game.conflict_position < 0:
            game.active_player = 0
        else:
            game.active_player = player


def _set_pending_if_options(
    game: GameState,
    kind: PendingChoiceKind,
    player: int,
    options: tuple[str, ...],
    *,
    consume_all_options: bool = False,
) -> None:
    if options:
        game.pending_choice = PendingChoice(kind, player, options, consume_all_options)


def _resolve_wonder_effects(
    game: GameState, player: int, wonder_name: str, ctx: _ChanceCtx
) -> bool:
    wonder = WONDERS_BY_NAME[wonder_name]
    extra_turn = False
    for effect in wonder.effects:
        if effect.kind is EffectKind.IMMEDIATE_COINS:
            game.cities[player].coins += effect.amount
        elif effect.kind is EffectKind.OPPONENT_LOSES_COINS:
            opponent = game.cities[1 - player]
            opponent.coins = max(0, opponent.coins - effect.amount)
        elif effect.kind is EffectKind.PLAY_AGAIN:
            extra_turn = True
        elif effect.kind is EffectKind.DESTROY_OPPONENT_BROWN:
            options = tuple(
                name
                for name in game.cities[1 - player].buildings
                if CARDS_BY_NAME[name].color is CardColor.BROWN
            )
            _set_pending_if_options(game, PendingChoiceKind.DESTROY_OPPONENT_BROWN, player, options)
        elif effect.kind is EffectKind.DESTROY_OPPONENT_GREY:
            options = tuple(
                name
                for name in game.cities[1 - player].buildings
                if CARDS_BY_NAME[name].color is CardColor.GREY
            )
            _set_pending_if_options(game, PendingChoiceKind.DESTROY_OPPONENT_GREY, player, options)
        elif effect.kind is EffectKind.BUILD_FROM_DISCARD_FREE:
            _set_pending_if_options(
                game,
                PendingChoiceKind.BUILD_FROM_DISCARD_FREE,
                player,
                tuple(game.discard_pile),
            )
        elif effect.kind is EffectKind.CHOOSE_UNUSED_PROGRESS:
            count = min(effect.amount, len(game.unused_progress_tokens))
            if count:
                supplied = ctx.draw(ChanceKind.GREAT_LIBRARY_DRAW)
                if supplied is not None:
                    drawn = tuple(supplied)
                    if (
                        len(drawn) != count
                        or len(set(drawn)) != count
                        or any(t not in game.unused_progress_tokens for t in drawn)
                    ):
                        raise ValueError(f"invalid Great Library outcome: {drawn}")
                else:
                    drawn = tuple(game.rng.sample(game.unused_progress_tokens, count))
                options = tuple(sorted(drawn, key=PROGRESS_IDS.__getitem__))
                ctx.events.append(
                    ResolvedChance(
                        kind=ChanceKind.GREAT_LIBRARY_DRAW, context=(), outcome=options
                    )
                )
                _set_pending_if_options(
                    game,
                    PendingChoiceKind.CHOOSE_UNUSED_PROGRESS,
                    player,
                    options,
                    consume_all_options=True,
                )
    return extra_turn


def apply_action(
    game: GameState, action: Action, *, chance_outcomes=None
) -> StepResult:
    """Apply one action; return the chance events that fired (CODEC_SPEC.md §4.3).

    ``chance_outcomes``: optional ordered sequence consumed by chance events as
    they fire (searcher-controlled resolution). Without it, the simulator's
    locked deal / seeded RNG resolves chance — unless ``game.search_barrier``
    is set, in which case unresolved chance raises HiddenInformationError.
    """

    if action not in legal_actions(game):
        raise ValueError(f"illegal action: {action}")
    ctx = _ChanceCtx(game, chance_outcomes)
    if action.use is ActionUse.DRAFT_WONDER:
        if action.wonder_name is None:
            raise AssertionError("malformed Wonder draft action")
        flipped_second_group = game.pick_wonder(action.wonder_name)
        if flipped_second_group:
            supplied = ctx.draw(ChanceKind.WONDER_GROUP_REVEAL)
            if supplied is not None:
                _override_wonder_flip(game, supplied)
            ctx.events.append(
                ResolvedChance(
                    kind=ChanceKind.WONDER_GROUP_REVEAL,
                    context=(),
                    outcome=tuple(sorted(game.wonder_offer, key=WONDER_IDS.__getitem__)),
                )
            )
        if game.phase is Phase.PLAY_AGE:
            # The eighth pick ends the draft and makes the Age I structure
            # observable for the first time — an AGE_DEAL chance event
            # (spec §4.2: "start_next_age and the initial Age I deal").
            supplied = ctx.draw(ChanceKind.AGE_DEAL)
            if supplied is not None:
                deal = _validated_age_deal(game, supplied)
                game.tableau = TableauState.from_deck(1, deal)
            else:
                deal = game.age_decks[1]
            ctx.events.append(
                ResolvedChance(kind=ChanceKind.AGE_DEAL, context=(1,), outcome=tuple(deal))
            )
        return ctx.result()
    if action.use is ActionUse.RESOLVE_PENDING_CHOICE:
        if action.choice is None:
            raise AssertionError("malformed pending-choice action")
        resolve_pending_choice(game, action.choice)
        return ctx.result()
    if action.use is ActionUse.CHOOSE_NEXT_START_PLAYER:
        if action.starting_player is None:
            raise AssertionError("malformed next-Age action")
        start_next_age(game, action.starting_player, _ctx=ctx)
        return ctx.result()
    if action.slot_id is None:
        raise AssertionError("primary Age action is missing a tableau slot")
    player = game.active_player
    card_name = game.tableau.cards[action.slot_id].card_name
    card = CARDS_BY_NAME[card_name]

    if action.use is ActionUse.DISCARD_FOR_COINS:
        _, newly_revealed = game.tableau.take_accessible(action.slot_id)
        _process_reveals(ctx, newly_revealed)
        game.discard_pile.append(card_name)
        yellow_count = _count_color(game, player, CardColor.YELLOW)
        game.cities[player].coins += discard_income(yellow_count)
        _finish_turn(game, player, False)
        return ctx.result()

    if action.use is ActionUse.CONSTRUCT_BUILDING:
        payment = minimum_payment(game, player, card.cost, card=card)
        _pay(game, player, payment)
        _, newly_revealed = game.tableau.take_accessible(action.slot_id)
        _process_reveals(ctx, newly_revealed)
        game.cities[player].buildings.append(card_name)
        _after_building_constructed(game, player, card)
        if payment.used_chain and "Urbanism" in game.cities[player].progress_tokens:
            game.cities[player].coins += 4
        _finish_turn(game, player, False)
        return ctx.result()

    if action.use is not ActionUse.CONSTRUCT_WONDER or action.wonder_name is None:
        raise AssertionError("malformed Wonder action")
    wonder = WONDERS_BY_NAME[action.wonder_name]
    if wonder.cost is None:
        raise AssertionError(f"missing Wonder cost: {wonder.name}")
    payment = minimum_payment(game, player, wonder.cost, is_wonder=True)
    _pay(game, player, payment)
    _, newly_revealed = game.tableau.take_accessible(action.slot_id)
    _process_reveals(ctx, newly_revealed)
    game.buried_cards.append(card_name)
    game.wonder_burials[wonder.name] = card_name
    game.cities[player].built_wonders.append(wonder.name)

    if sum(len(city.built_wonders) for city in game.cities) == 7:
        remaining = [
            name
            for city in game.cities
            for name in city.wonders
            if name not in city.built_wonders and name not in game.retired_wonders
        ]
        if len(remaining) != 1:
            raise AssertionError("seventh Wonder must leave exactly one unbuilt Wonder")
        game.retired_wonders.add(remaining[0])

    extra_turn = _resolve_wonder_effects(game, player, wonder.name, ctx)
    if "Theology" in game.cities[player].progress_tokens:
        extra_turn = True
    if game.pending_choice is not None:
        game.pending_shields = wonder.shields
    elif wonder.shields:
        _apply_military(game, player, wonder.shields)
    _finish_turn(game, player, extra_turn)
    return ctx.result()


def resolve_pending_choice(game: GameState, choice: str) -> None:
    pending = game.pending_choice
    if pending is None:
        raise ValueError("there is no pending choice")
    if choice not in pending.options:
        raise ValueError(f"invalid pending choice: {choice}")
    player = pending.player
    game.pending_choice = None
    extra_turn = game.pending_extra_turn
    game.pending_extra_turn = False
    pending_shields = game.pending_shields
    game.pending_shields = 0

    if pending.kind in (
        PendingChoiceKind.DESTROY_OPPONENT_BROWN,
        PendingChoiceKind.DESTROY_OPPONENT_GREY,
    ):
        opponent = game.cities[1 - player]
        opponent.buildings.remove(choice)
        game.discard_pile.append(choice)
    elif pending.kind is PendingChoiceKind.BUILD_FROM_DISCARD_FREE:
        game.discard_pile.remove(choice)
        card = CARDS_BY_NAME[choice]
        game.cities[player].buildings.append(choice)
        _after_building_constructed(game, player, card)
    elif pending.kind in (
        PendingChoiceKind.CHOOSE_UNUSED_PROGRESS,
        PendingChoiceKind.CHOOSE_AVAILABLE_PROGRESS,
    ):
        game.cities[player].progress_tokens.append(choice)
        if pending.consume_all_options:
            consumed = set(pending.options)
            game.unused_progress_tokens = tuple(
                name for name in game.unused_progress_tokens if name not in consumed
            )
        else:
            game.available_progress_tokens = tuple(
                name for name in game.available_progress_tokens if name != choice
            )
        _apply_progress_immediate(game, player, choice)
        _check_scientific_victory(game, player)
    else:
        raise AssertionError(f"unhandled pending choice kind: {pending.kind}")

    if game.phase is Phase.COMPLETE:
        return
    if pending_shields:
        _apply_military(game, player, pending_shields)
    if game.phase is Phase.COMPLETE:
        return
    _finish_turn(game, player, extra_turn)


def _military_victory_points(game: GameState, player: int) -> int:
    position = game.conflict_position
    if position == 0 or (position > 0) != (player == 0):
        return 0
    distance = abs(position)
    if distance <= 3:
        return 2
    if distance <= 6:
        return 5
    return 10


def _guild_victory_points(game: GameState, player: int) -> int:
    points = 0
    for name in game.cities[player].buildings:
        card = CARDS_BY_NAME[name]
        for effect in card.effects:
            if effect.kind is EffectKind.VP_PER_MOST_COLOR:
                if effect.color is None:
                    raise AssertionError("guild VP effect is missing a color")
                points += effect.amount * max(
                    _count_color(game, 0, effect.color),
                    _count_color(game, 1, effect.color),
                )
            elif effect.kind is EffectKind.VP_PER_MOST_BROWN_GREY:
                counts = [
                    _count_color(game, p, CardColor.BROWN)
                    + _count_color(game, p, CardColor.GREY)
                    for p in (0, 1)
                ]
                points += effect.amount * max(counts)
            elif effect.kind is EffectKind.VP_PER_MOST_WONDER:
                points += effect.amount * max(
                    len(game.cities[0].built_wonders),
                    len(game.cities[1].built_wonders),
                )
            elif effect.kind is EffectKind.VP_PER_RICHEST_COIN_SET:
                points += effect.amount * (max(city.coins for city in game.cities) // 3)
    return points


def score_player(game: GameState, player: int) -> ScoreBreakdown:
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    city = game.cities[player]
    cards = [CARDS_BY_NAME[name] for name in city.buildings]
    military = _military_victory_points(game, player)
    guild = _guild_victory_points(game, player)
    buildings = sum(card.victory_points for card in cards) + guild
    wonders = sum(WONDERS_BY_NAME[name].victory_points for name in city.built_wonders)
    progress = sum(PROGRESS_BY_NAME[name].victory_points for name in city.progress_tokens)
    for token_name in city.progress_tokens:
        for effect in PROGRESS_BY_NAME[token_name].effects:
            if effect.kind is EffectKind.VP_PER_PROGRESS:
                progress += effect.amount * len(city.progress_tokens)
    treasury = city.coins // 3
    blue_buildings = sum(
        card.victory_points for card in cards if card.color is CardColor.BLUE
    )
    total = military + buildings + wonders + progress + treasury
    return ScoreBreakdown(
        military=military,
        buildings=buildings,
        wonders=wonders,
        progress=progress,
        treasury=treasury,
        total=total,
        blue_buildings=blue_buildings,
        guild=guild,
    )


def _resolve_civilian_endgame(game: GameState) -> None:
    scores = (score_player(game, 0), score_player(game, 1))
    game.final_scores = (scores[0].total, scores[1].total)
    game.phase = Phase.COMPLETE
    if scores[0].total != scores[1].total:
        game.winner = 0 if scores[0].total > scores[1].total else 1
        game.victory_type = VictoryType.CIVILIAN
    elif scores[0].blue_buildings != scores[1].blue_buildings:
        game.winner = 0 if scores[0].blue_buildings > scores[1].blue_buildings else 1
        game.victory_type = VictoryType.CIVILIAN
    else:
        game.winner = None
        game.victory_type = VictoryType.SHARED_CIVILIAN


def _validated_age_deal(game: GameState, supplied) -> tuple[str, ...]:
    """Validate a searcher-supplied AGE_DEAL arrangement and update the setup
    records (removed cards, guild selection) to stay a consistent world."""

    deal = tuple(supplied)
    age = game.age
    layout_size = len(TABLEAU_LAYOUTS[age])
    if len(deal) != layout_size or len(set(deal)) != layout_size:
        raise ValueError(f"Age {age} deal must be {layout_size} distinct cards")
    allowed_backs = (
        {BackType.AGE_III, BackType.GUILD} if age == 3 else {back_type_of_age(age)}
    )
    visible = set(game.discard_pile) | set(game.buried_cards)
    for city in game.cities:
        visible |= set(city.buildings)
    guilds_in_deal = []
    for name in deal:
        if name not in CARDS_BY_NAME:
            raise ValueError(f"unknown card in Age deal: {name}")
        back = back_type_of(name)
        if back not in allowed_backs:
            raise ValueError(f"card {name} cannot appear in an Age {age} deal")
        if name in visible:
            raise ValueError(f"card {name} is already visible and cannot be dealt")
        if back is BackType.GUILD:
            guilds_in_deal.append(name)
    if age == 3 and len(guilds_in_deal) != 3:
        raise ValueError("an Age III deal must contain exactly 3 guild cards")

    age_universe = tuple(
        card.name
        for card in CARDS_BY_NAME.values()
        if back_type_of(card.name) is back_type_of_age(age)
    )
    game.removed_age_cards[age] = tuple(
        name for name in age_universe if name not in deal and name not in visible
    )
    if age == 3:
        game.selected_guilds = tuple(guilds_in_deal)
        game.unused_guilds = tuple(
            card.name
            for card in CARDS_BY_NAME.values()
            if back_type_of(card.name) is BackType.GUILD
            and card.name not in guilds_in_deal
        )
    game.age_decks[age] = deal
    return deal


def back_type_of_age(age: int) -> BackType:
    return {1: BackType.AGE_I, 2: BackType.AGE_II, 3: BackType.AGE_III}[age]


def start_next_age(
    game: GameState,
    starting_player: int,
    *,
    chance_outcomes=None,
    _ctx: _ChanceCtx | None = None,
) -> StepResult:
    """Prepare the next Age after the military chooser is resolved.

    Dealing the new tableau is an AGE_DEAL chance event: the simulator uses the
    deck locked at setup; a searcher supplies a full arrangement (from
    ``pool.resample_hidden`` / a determinizer) via ``chance_outcomes``.
    """

    if game.phase is not Phase.CHOOSE_NEXT_START_PLAYER:
        raise ValueError("the current Age is not complete")
    if starting_player not in (0, 1):
        raise ValueError("starting_player must be 0 or 1")
    ctx = _ctx if _ctx is not None else _ChanceCtx(game, chance_outcomes)
    game.age += 1
    supplied = ctx.draw(ChanceKind.AGE_DEAL)
    if supplied is not None:
        deal = _validated_age_deal(game, supplied)
    else:
        deal = game.age_decks[game.age]
    game.tableau = TableauState.from_deck(game.age, deal)
    ctx.events.append(
        ResolvedChance(kind=ChanceKind.AGE_DEAL, context=(game.age,), outcome=deal)
    )
    game.active_player = starting_player
    game.phase = Phase.PLAY_AGE
    return StepResult(events=tuple(ctx.events)) if _ctx is None else StepResult()
