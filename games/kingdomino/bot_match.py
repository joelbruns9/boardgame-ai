import random

from games.kingdomino.bots import RandomBot, GreedyBot
from games.kingdomino.game import GameState, Phase, determine_winner


def total_score(score):
    return score.territory_score + score.harmony_bonus + score.middle_kingdom_bonus


def new_state(seed):
    random.seed(seed)

    if hasattr(GameState, "new"):
        return GameState.new(seed=seed)

    if hasattr(GameState, "new_game"):
        return GameState.new_game(seed=seed)

    try:
        return GameState(seed=seed)
    except TypeError:
        return GameState()


def is_terminal(state):
    if hasattr(state, "is_terminal"):
        return state.is_terminal()

    if hasattr(state, "terminal"):
        return state.terminal()

    if hasattr(state, "is_game_over"):
        return state.is_game_over()

    return state.phase == Phase.GAME_OVER


def legal_actions(state):
    if hasattr(state, "legal_actions"):
        return state.legal_actions()

    if hasattr(state, "get_legal_actions"):
        return state.get_legal_actions()

    if hasattr(state, "actions"):
        return state.actions()

    raise AttributeError("Could not find legal action method on GameState")


def apply_action(state, action):
    if hasattr(state, "step"):
        result = state.step(action)
        return state if result is None else result

    if hasattr(state, "apply"):
        result = state.apply(action)
        return state if result is None else result

    if hasattr(state, "apply_action"):
        result = state.apply_action(action)
        return state if result is None else result

    raise AttributeError("Could not find action application method on GameState")


def current_player(state):
    current_actor = getattr(state, "current_actor", None)

    if callable(current_actor):
        actor = current_actor()
        return actor.player

    if isinstance(current_actor, int):
        return current_actor

    if state.phase == Phase.INITIAL_SELECTION:
        if state.initial_pick_count in (0, 3):
            return state.start_player
        return 1 - state.start_player

    claim = state.pending_claims[state.actor_index]
    return claim.player


def play_bot_game(seed=0, bot0=None, bot1=None):
    rng = random.Random(seed)
    bots = [bot0 or RandomBot(), bot1 or RandomBot()]

    state = new_state(seed)

    while not is_terminal(state):
        actions = legal_actions(state)
        player = current_player(state)
        action = bots[player].choose_action(state, actions, rng=rng)
        state = apply_action(state, action)

    scores = [total_score(board.score()) for board in state.boards]
    return scores, state


def run_match(label, bot0, bot1, games=100, seed_offset=0, verbose=False):
    wins = [0, 0]
    ties = 0
    score_sum = [0, 0]
    step_counts = []

    for i in range(games):
        seed = seed_offset + i
        scores, state = play_bot_game(seed=seed, bot0=bot0, bot1=bot1)

        score_sum[0] += scores[0]
        score_sum[1] += scores[1]
        step_counts.append(len(state.history))

        # Authoritative cascade (score -> largest territory -> crowns -> draw),
        # not a raw score comparison: score ties are resolved by tiebreakers.
        winner = determine_winner(state)
        if winner == 0:
            wins[0] += 1
        elif winner == 1:
            wins[1] += 1
        else:
            ties += 1

        if verbose:
            print(f"seed={seed:03d} scores={scores} steps={len(state.history)}")

    print()
    print(label)
    print("-" * len(label))
    print(f"games: {games}")
    print(f"P0 wins: {wins[0]}")
    print(f"P1 wins: {wins[1]}")
    print(f"ties: {ties}")
    print(f"avg P0 score: {score_sum[0] / games:.2f}")
    print(f"avg P1 score: {score_sum[1] / games:.2f}")
    print(f"min steps: {min(step_counts)}")
    print(f"max steps: {max(step_counts)}")

    return {
        "label": label,
        "games": games,
        "wins": wins,
        "ties": ties,
        "avg_scores": [score_sum[0] / games, score_sum[1] / games],
    }


def main():
    games = 100

    run_match(
        label="Greedy as P0 vs Random as P1",
        bot0=GreedyBot(),
        bot1=RandomBot(),
        games=games,
        seed_offset=0,
    )

    run_match(
        label="Random as P0 vs Greedy as P1",
        bot0=RandomBot(),
        bot1=GreedyBot(),
        games=games,
        seed_offset=10_000,
    )

    run_match(
        label="Greedy as P0 vs Greedy as P1",
        bot0=GreedyBot(),
        bot1=GreedyBot(),
        games=games,
        seed_offset=20_000,
    )


if __name__ == "__main__":
    main()