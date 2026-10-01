"""Varied start positions for self-play: a short random-play prefix.

A share of self-play games starts from a board reached by a few turns of
random play instead of the empty board. Random play is a deliberately poor,
board-blind policy -- random column choices, banking on a coin flip -- so it
reaches boards our self-play never does (odd columns pushed, lopsided
progress, early claims of short columns), the kind a human opponent can
create. Every start is REACHABLE by legal play, unlike a uniform random
board, which would mostly show the net positions no game produces.

The prefix only builds the board: its turns are not recorded and produce no
training rows. Rows come from the self-play that follows, with the usual
targets. A start never ends the game; if the prefix would, the board before
that turn is used.
"""

from .engine import (GameState, apply_move, can_stop, legal_moves,
                     random_dice, roll, stop)
from .portable_rng import PortableRng

# Mixed into each game's seed so the prefix draws its own stream: switching
# random starts on or off never shifts any other game's dice.
_SALT = 0x5EED_0F_5_7A27


def random_turn(state, rng, stop_prob):
    """One board-blind turn: random legal move, bank with ``stop_prob``."""
    while True:
        moves = roll(state, random_dice(rng))
        if not moves:
            return
        apply_move(state, rng.choice(legal_moves(state, state.dice)))
        if can_stop(state) and rng.next_float() < stop_prob:
            stop(state)
            return


def random_start(rules, seed, turns_per_player=8, stop_prob=0.35):
    """A turn-start board after 1..``turns_per_player * players`` random
    turns, from the game's own ``seed`` (salted)."""
    rng = PortableRng((seed ^ _SALT) & (2**64 - 1))
    state = GameState(rules)
    turns = rng.randint(1, turns_per_player * rules.num_players)
    for _ in range(turns):
        before = state.clone()
        random_turn(state, rng, stop_prob)
        if state.game_over:
            return before
    return state


def pick_random_starts(seeds, fraction):
    """Which games start from a random prefix: a per-game coin from the
    game's own seed, so the choice draws nothing from the run's RNG."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("random start fraction must be in [0, 1]")
    if fraction == 0.0:
        return [False] * len(seeds)
    return [PortableRng((s ^ (_SALT << 1)) & (2**64 - 1)).next_float() < fraction
            for s in seeds]
