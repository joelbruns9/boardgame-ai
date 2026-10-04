"""G4 exact tactics: the Python reference for `seven_wonders_rust/src/tactics.rs`.

Two predicates over one decision state, both EXACT (no network, no sampling):

* `forced_win` -- the mover can force a win before the opponent moves again:
  by military or science, through their own pending choices (Mausoleum
  retrieval, a science-pair token, Law), through ONE extra turn (a replay
  wonder, or any wonder with Theology), or on points by taking the last card of
  Age III. Every consistent chance outcome of every step must win.
* `forced_loss` -- every action of the mover, under every chance outcome,
  leaves the opponent a `forced_win`.

`phase_e.guaranteed_win_now` is the narrower predicate the trap suite was built
on (no extra turns, no civilian endings) and stays as it is; this module is
what search uses.

Proof discipline: anything the predicate cannot see through -- an Age deal it
cannot enumerate, an outcome it cannot apply -- counts AGAINST the claim being
proven. A win proof skips such an action; a loss proof treats it as an escape.
"""

from __future__ import annotations

from .codec import decode_action, legal_action_indices
from .data import CARDS_BY_NAME, WONDERS_BY_NAME, EffectKind
from .engine import ActionUse, _science_symbols, apply_action
from .game import ChanceKind, GameState, HiddenInformationError, Phase
from .search import chance_signature, enumerate_chains, state_actor

#: One action's largest plausible shield swing (3 shields + Strategy, + slack).
MILITARY_REACH = 5
#: A new-symbol green that completes a pair -> token -> Law: +2 symbols.
SCIENCE_REACH = 4
#: Steps of the mover's own pending choices followed within one turn.
PENDING_DEPTH = 6
#: Extra turns followed. One covers every replay wonder; Theology chains of
#: two wonders in a row are rare enough to leave to search.
EXTRA_TURNS = 1


def _present_cards(state: GameState) -> int:
    return sum(1 for card in state.tableau.cards.values() if card.present)


def _within_reach(state: GameState, player: int, turns: int) -> bool:
    """Necessary condition for `player` to win within `turns` own actions.

    The opponent never adds to `player`'s shields or symbols, so only
    `player`'s own actions (and reveals, which the reaches allow for) can.
    Civilian: the game ends on points only when Age III runs out of cards.
    """

    need = 9 - state.conflict_position if player == 0 else 9 + state.conflict_position
    if need <= MILITARY_REACH * turns:
        return True
    if len(_science_symbols(state, player)) >= 6 - 2 * turns:
        return True
    return state.age == 3 and _present_cards(state) <= turns


def _chains(state: GameState, index: int):
    """The chance outcomes of one action, or None when they cannot be
    enumerated (an Age deal)."""

    specs = chance_signature(state, decode_action(state, index))
    if any(spec.kind is ChanceKind.AGE_DEAL for spec in specs):
        return None
    return [outcomes for outcomes, _probability, _key in enumerate_chains(state, specs)]


def _child(state: GameState, index: int, outcomes) -> GameState | None:
    """One consistent child, or None when it cannot be applied without
    reading hidden state."""

    clone = state.clone()
    clone.search_barrier = True
    try:
        apply_action(clone, decode_action(clone, index), chance_outcomes=outcomes or None)
    except HiddenInformationError:
        return None
    return clone


def _every_child(state: GameState, index: int, test) -> bool:
    """`test` holds for EVERY consistent child of the action. Children are
    built one at a time and the walk stops at the first failure -- most
    candidate actions fail in their first world, and building every world
    first was ~50x the cost. Unenumerable or inapplicable counts as failure."""

    chains = _chains(state, index)
    if not chains:
        return False
    for outcomes in chains:
        child = _child(state, index, outcomes)
        if child is None or not test(child):
            return False
    return True


def _can_replay(state: GameState, player: int) -> bool:
    """`player` still holds an unbuilt wonder that would grant an extra turn
    (a play-again wonder, or any wonder with Theology). A necessary condition
    for the extra-turn route, checked from the city alone."""

    city = state.cities[player]
    unbuilt = [w for w in city.wonders if w not in city.built_wonders]
    if "Theology" in city.progress_tokens:
        return bool(unbuilt)
    return any(
        e.kind is EffectKind.PLAY_AGAIN
        for w in unbuilt
        for e in WONDERS_BY_NAME[w].effects
    )


def _replay_reach(state: GameState, player: int, extra: int) -> bool:
    """Necessary condition for a win that NEEDS the extra turn.

    A play-again wonder brings no shields and no symbols -- only coins, a
    resource or points -- so it can make a winning card affordable but cannot
    bring a military or science win nearer than one action already reaches.
    Only Theology (any wonder replays: Colossus' shields, the Great Library's
    tokens, the Mausoleum's greens) widens those reaches; the civilian ending
    widens with every extra card taken.
    """

    if extra <= 0 or not _can_replay(state, player):
        return False
    if "Theology" in state.cities[player].progress_tokens:
        return _within_reach(state, player, 1 + extra)
    return state.age == 3 and _present_cards(state) <= 1 + extra


def _replay_wonders(state: GameState) -> list[int]:
    """Wonder builds that hand the mover an extra turn: a play-again wonder,
    or any wonder once they hold Theology."""

    player = state_actor(state)
    theology = "Theology" in state.cities[player].progress_tokens
    out = []
    for index in legal_action_indices(state):
        action = decode_action(state, index)
        if action.use is not ActionUse.CONSTRUCT_WONDER:
            continue
        effects = WONDERS_BY_NAME[action.wonder_name].effects
        if theology or any(e.kind is EffectKind.PLAY_AGAIN for e in effects):
            out.append(index)
    return out


def _candidates(state: GameState) -> list[int]:
    """Actions that could end the game for the mover, or hand them a turn
    that can: shield and science builds, every wonder, every pending option,
    and -- when it may be the last card of Age III -- everything."""

    legal = legal_action_indices(state)
    if state.age == 3 and _present_cards(state) <= 1:
        return legal
    out = []
    for index in legal:
        action = decode_action(state, index)
        if action.use in (ActionUse.CONSTRUCT_WONDER, ActionUse.RESOLVE_PENDING_CHOICE):
            out.append(index)
        elif action.use is ActionUse.CONSTRUCT_BUILDING:
            card = CARDS_BY_NAME[state.tableau.cards[action.slot_id].card_name]
            if card.shields > 0 or card.science is not None:
                out.append(index)
    return out


def _won(child: GameState, player: int, depth: int, extra: int) -> bool:
    """After one of `player`'s actions resolved: is the win now forced?"""

    if child.phase is Phase.COMPLETE:
        return child.winner == player
    if state_actor(child) != player:
        return False
    if child.pending_choice is not None:
        return _pending_win(child, player, depth, extra)  # same turn: no extra spent
    return extra > 0 and forced_win(child, depth, extra - 1)


def _pending_win(state: GameState, player: int, depth: int, extra: int = 0) -> bool:
    if depth <= 0:
        return False
    return any(
        _every_child(state, index, lambda c: _won(c, player, depth - 1, extra))
        for index in legal_action_indices(state)
    )


def forced_win(
    state: GameState, depth: int = PENDING_DEPTH, extra: int = EXTRA_TURNS
) -> bool:
    if state.phase is Phase.COMPLETE:
        return False
    player = state_actor(state)
    now = _within_reach(state, player, 1)
    if not (now or _replay_reach(state, player, extra)):
        return False
    if state.pending_choice is not None:
        return _pending_win(state, player, depth, extra)
    # Out of reach this action: only an extra turn can get there.
    candidates = _candidates(state) if now else _replay_wonders(state)
    return any(
        _every_child(state, index, lambda c: _won(c, player, depth, extra))
        for index in candidates
    )


def _lost(child: GameState, actor: int, depth: int) -> bool:
    """After `actor`'s action resolved: is `actor` now certainly lost?"""

    if child.phase is Phase.COMPLETE:
        return child.winner is not None and child.winner != actor
    if state_actor(child) == actor:
        # Own pending choice: lost only if EVERY option is. An extra turn keeps
        # the initiative and is never proven lost here.
        if child.pending_choice is None or depth <= 0:
            return False
        return all(
            _every_child(child, index, lambda g: _lost(g, actor, depth - 1))
            for index in legal_action_indices(child)
        )
    return forced_win(child, depth)


def forced_loss(state: GameState, depth: int = PENDING_DEPTH) -> bool:
    if state.phase is Phase.COMPLETE:
        return False
    actor = state_actor(state)
    # The mover's own action brings the opponent no shields or symbols (bar a
    # reveal, which the reaches allow for), but it does take a card -- one
    # step nearer the civilian ending.
    opponent = 1 - actor
    # The mover's own action takes a card, so every civilian reach is one
    # card wider here.
    if not (
        _within_reach(state, opponent, 1)
        or _replay_reach(state, opponent, EXTRA_TURNS)
        or (state.age == 3 and _present_cards(state) <= 2)
        or (
            state.age == 3
            and _present_cards(state) <= 2 + EXTRA_TURNS
            and _can_replay(state, opponent)
        )
    ):
        return False
    legal = legal_action_indices(state)
    return bool(legal) and all(
        _every_child(state, index, lambda c: _lost(c, actor, depth)) for index in legal
    )
