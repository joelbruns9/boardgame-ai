"""Row-balanced variant schedule for the Phase 4 generalist.

Rows per game span 5x across the ten rule sets (2p base ~10, 3p 4-column
blocking ~50), so equal GAMES would train mostly on the long variants. The
schedule instead targets equal ROWS per variant each iteration: games per
variant = ceil(target / rows-per-game), with rows-per-game tracked as an
exponential moving average of what the previous iterations produced (games
lengthen or shorten as the net learns). The measured cost per row is
nearly flat across variants, so this also roughly equalises compute.
"""

import math

from .engine import ALL_RULESETS

# Measured 2026-09-26 (VARIANT_SOLVER_PLAN.md, Phase 4 sizing table), keyed
# by (players, columns to win, blocking). Only a starting point: the moving
# average takes over after the first iteration.
SEED_ROWS_PER_GAME = {
    (2, 3, False): 9.8, (2, 3, True): 9.8,
    (2, 5, False): 20.5, (2, 5, True): 20.1,
    (3, 3, False): 29.0, (3, 3, True): 29.3,
    (3, 4, False): 41.2, (3, 4, True): 50.0,
    (4, 3, False): 29.1, (4, 3, True): 28.2,
}


def rules_key(rules):
    return (rules.num_players, rules.columns_to_win, rules.blocking)


class RowSchedule:
    """Games per variant for a per-variant row target."""

    def __init__(self, rule_sets=ALL_RULESETS, rows_per_variant=4000,
                 smoothing=0.5, min_games=1, weights=None):
        self.rule_sets = tuple(rule_sets)
        self.rows_per_variant = rows_per_variant
        # Emphasis: a variant's row target is rows_per_variant x its weight
        # (default 1). Configuration, not state: a resume takes the CLI's.
        self.weights = {rules_key(r): 1.0 for r in self.rule_sets}
        for k, w in (weights or {}).items():
            if k not in self.weights or not w > 0:
                raise ValueError(f"bad variant weight {k}: {w}")
            self.weights[k] = float(w)
        self.smoothing = smoothing          # weight kept on the old estimate
        self.min_games = min_games
        self.rows_per_game = {rules_key(r): SEED_ROWS_PER_GAME[rules_key(r)]
                              for r in self.rule_sets}

    def games(self):
        """{rules: games} for the next iteration, in rule-set order."""
        return {r: max(self.min_games,
                       math.ceil(self.rows_per_variant
                                 * self.weights[rules_key(r)]
                                 / self.rows_per_game[rules_key(r)]))
                for r in self.rule_sets}

    def update(self, results):
        """Fold in one iteration's measured rows per game."""
        rows, games = {}, {}
        for res in results:
            k = rules_key(res.rules)
            rows[k] = rows.get(k, 0) + len(res)
            games[k] = games.get(k, 0) + 1
        for k, g in games.items():
            measured = rows[k] / g
            self.rows_per_game[k] = (self.smoothing * self.rows_per_game[k]
                                     + (1 - self.smoothing) * measured)

    def state(self):
        return {"rows_per_variant": self.rows_per_variant,
                "smoothing": self.smoothing,
                "rows_per_game": {str(k): v for k, v
                                  in self.rows_per_game.items()}}

    def load(self, state):
        self.rows_per_variant = state["rows_per_variant"]
        self.smoothing = state["smoothing"]
        by_str = {str(k): k for k in self.rows_per_game}
        for ks, v in state["rows_per_game"].items():
            self.rows_per_game[by_str[ks]] = v
