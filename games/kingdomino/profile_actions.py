"""
DIAGNOSTIC — Profiling script for the pre-Rust A3 worker architecture.
The bottlenecks identified here (IPC overhead, Python tree work) were
addressed by the Rust BatchedMCTS engine in self_play.py.

Not relevant for current training pipeline. Kept for historical reference.
"""
import random
from collections import defaultdict

from games.kingdomino.bots import RandomBot, GreedyBot
from games.kingdomino.bot_match import (
    new_state,
    is_terminal,
    legal_actions,
    apply_action,
    current_player,
)


def percentile(values, p):
    if not values:
        return None

    sorted_values = sorted(values)
    index = int(round((p / 100) * (len(sorted_values) - 1)))
    return sorted_values[index]


def is_true_discard(action):
    """
    True discard means this is a TurnAction where placement is None.

    PickAction in the initial selection phase does not have a placement field,
    so it should not be counted as a discard.
    """
    return hasattr(action, "placement") and action.placement is None


def has_placement(action):
    return hasattr(action, "placement") and action.placement is not None


def has_pick(action):
    return getattr(action, "pick_domino_id", None) is not None or getattr(action, "domino_id", None) is not None


def placement_key(action):
    placement = getattr(action, "placement", None)
    if placement is None:
        return None

    return (
        placement.x1,
        placement.y1,
        placement.x2,
        placement.y2,
        placement.flipped,
    )


def pick_key(action):
    if hasattr(action, "pick_domino_id"):
        return action.pick_domino_id
    if hasattr(action, "domino_id"):
        return action.domino_id
    return None


def summarize_values(values):
    return {
        "turns": len(values),
        "avg": sum(values) / len(values) if values else 0.0,
        "min": min(values) if values else 0,
        "p50": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else 0,
    }


def print_summary(name, values, indent="  "):
    s = summarize_values(values)
    print(f"{indent}{name}:")
    print(f"{indent}  turns: {s['turns']}")
    print(f"{indent}  avg: {s['avg']:.2f}")
    print(f"{indent}  min: {s['min']}")
    print(f"{indent}  p50: {s['p50']}")
    print(f"{indent}  p75: {s['p75']}")
    print(f"{indent}  p90: {s['p90']}")
    print(f"{indent}  p95: {s['p95']}")
    print(f"{indent}  p99: {s['p99']}")
    print(f"{indent}  max: {s['max']}")


def profile_games(bot0, bot1, games=100, seed_offset=0):
    bots = [bot0, bot1]

    action_counts_by_phase = defaultdict(list)
    placement_counts_by_phase = defaultdict(list)
    pick_counts_by_phase = defaultdict(list)
    unique_placement_counts_by_phase = defaultdict(list)
    unique_pick_counts_by_phase = defaultdict(list)
    true_discard_counts_by_phase = defaultdict(list)

    max_action_context = None
    max_actions = -1

    max_unique_placement_context = None
    max_unique_placements = -1

    total_turns = 0
    total_actions_seen = 0
    total_true_discard_actions = 0
    chosen_true_discards = 0

    for i in range(games):
        seed = seed_offset + i
        state = new_state(seed)
        local_rng = random.Random(seed)

        while not is_terminal(state):
            actions = legal_actions(state)
            phase_name = state.phase.name
            player = current_player(state)

            action_count = len(actions)
            placement_count = sum(1 for action in actions if has_placement(action))
            pick_count = sum(1 for action in actions if has_pick(action))
            true_discard_count = sum(1 for action in actions if is_true_discard(action))

            unique_placements = {
                placement_key(action)
                for action in actions
                if placement_key(action) is not None
            }
            unique_picks = {
                pick_key(action)
                for action in actions
                if pick_key(action) is not None
            }

            unique_placement_count = len(unique_placements)
            unique_pick_count = len(unique_picks)

            action_counts_by_phase[phase_name].append(action_count)
            placement_counts_by_phase[phase_name].append(placement_count)
            pick_counts_by_phase[phase_name].append(pick_count)
            unique_placement_counts_by_phase[phase_name].append(unique_placement_count)
            unique_pick_counts_by_phase[phase_name].append(unique_pick_count)
            true_discard_counts_by_phase[phase_name].append(true_discard_count)

            total_turns += 1
            total_actions_seen += action_count
            total_true_discard_actions += true_discard_count

            if action_count > max_actions:
                max_actions = action_count
                max_action_context = {
                    "seed": seed,
                    "step": len(state.history),
                    "phase": phase_name,
                    "player": player,
                    "actions": action_count,
                    "placements": placement_count,
                    "unique_placements": unique_placement_count,
                    "unique_picks": unique_pick_count,
                    "true_discards": true_discard_count,
                }

            if unique_placement_count > max_unique_placements:
                max_unique_placements = unique_placement_count
                max_unique_placement_context = {
                    "seed": seed,
                    "step": len(state.history),
                    "phase": phase_name,
                    "player": player,
                    "actions": action_count,
                    "unique_placements": unique_placement_count,
                    "unique_picks": unique_pick_count,
                    "true_discards": true_discard_count,
                }

            chosen = bots[player].choose_action(state, actions, rng=local_rng)

            if is_true_discard(chosen):
                chosen_true_discards += 1

            state = apply_action(state, chosen)

    print("Action-space profile")
    print("--------------------")
    print(f"games: {games}")
    print(f"total turns: {total_turns}")
    print(f"total legal actions seen: {total_actions_seen}")
    print(f"total true discard actions available: {total_true_discard_actions}")
    print(f"chosen true discards: {chosen_true_discards}")
    print(f"max legal actions: {max_actions}")
    print(f"max action context: {max_action_context}")
    print(f"max unique placements: {max_unique_placements}")
    print(f"max unique placement context: {max_unique_placement_context}")
    print()

    for phase_name in sorted(action_counts_by_phase):
        print(f"{phase_name}:")
        print_summary("legal actions", action_counts_by_phase[phase_name])
        print_summary("placement actions", placement_counts_by_phase[phase_name])
        print_summary("pick actions", pick_counts_by_phase[phase_name])
        print_summary("unique placements", unique_placement_counts_by_phase[phase_name])
        print_summary("unique picks", unique_pick_counts_by_phase[phase_name])
        print_summary("true discard actions", true_discard_counts_by_phase[phase_name])
        print()


def main():
    games = 100

    print("Profiling Greedy vs Greedy")
    profile_games(
        bot0=GreedyBot(),
        bot1=GreedyBot(),
        games=games,
        seed_offset=30_000,
    )

    print()
    print("Profiling Random vs Random")
    profile_games(
        bot0=RandomBot(),
        bot1=RandomBot(),
        games=games,
        seed_offset=40_000,
    )


if __name__ == "__main__":
    main()