"""Per-variant evaluation and the probe-set monitor for the Phase 4 run.

**Matches** (`evaluate_variants`): in every variant, the current net
against n-1 copies of each opponent (the heuristic, and any frozen
reference nets), seats rotated over COMPLETE seat cycles, even = 1/n,
Wilson interval. These are coarse monitors (200 games is about +/-7 pts);
strength claims go through `confirm.py`.

**Probe monitor** (`ProbeSet`): a fixed set of boards per variant, drawn
uniformly once from heuristic self-play and kept for the whole run. Each
measurement compares the net's value V(b) for the side to move with one
exact turn of search from b using the same net, T(V)(b). The residual
V - T(V) is a SELF-CONSISTENCY monitor (plan review): small does not mean
accurate, and growth does not uniquely diagnose harmful drift -- but a
residual rising past the pre-declared threshold (``alert``) is the λ=0
bootstrap-drift signal to look at.
"""

import math
import zlib

import numpy as np

from .arena import verdict
from .confirm import seat_cycle_games
from .encoder import decode_features, encode_batch, to_absolute
from .portable_rng import PortableRng
from .rust_pool import game_seeds, play_match, run_pool
from .schedule import rules_key
from .self_play import PLAIN, Search
from .snapshot import from_snapshot, snapshot
from .solver import ProgressHeuristic


def evaluate_variants(net, rule_sets, games, seed, references=None,
                      search=PLAIN, threads=0, in_flight=None):
    """{variant: {"vs_heuristic": verdict, "vs_<ref>": verdict, ...}}.
    ``references``: {name: evaluator} frozen nets."""
    opponents = {"heuristic": ProgressHeuristic(), **(references or {})}
    out = {}
    for rules in rule_sets:
        n = rules.num_players
        g = seat_cycle_games(games, n)
        row = {}
        for name, opp in opponents.items():
            rng = PortableRng(zlib.crc32(
                f"eval|{rules_key(rules)}|{name}|{seed}".encode()))
            kw = {"in_flight": in_flight} if in_flight else {}
            wins = play_match(rules, [net] + [opp] * (n - 1), g, rng,
                              threads=threads, searches=[search] * n, **kw)
            row[f"vs_{name}"] = verdict(wins, n)
        out[str(rules)] = row
    return out


class ProbeSet:
    """Fixed boards per variant, and the V-vs-T(V) residual over them."""

    def __init__(self, boards):
        self.boards = boards                # {str(rules): [snapshot, ...]}

    @classmethod
    def build(cls, rule_sets, per_variant=50, seed=0, games=24):
        """Uniform draws from heuristic self-play boards, per variant."""
        h = ProgressHeuristic()
        boards = {}
        for i, rules in enumerate(rule_sets):
            res = run_pool([rules] * games,
                           game_seeds(PortableRng(seed * 101 + i), games), [h])
            pool = []
            n = rules.num_players
            for r in res:
                for row, slot in zip(r.features, r.winner_slots):
                    pool.append(decode_features(row, (r.winner - int(slot)) % n))
            order = list(range(len(pool)))
            PortableRng(seed * 211 + i).shuffle(order)
            boards[str(rules)] = [snapshot(pool[j])
                                  for j in sorted(order[:per_variant])]
        return cls(boards)

    def measure(self, net, alert=0.05, in_flight=64):
        """Per variant: RMSE, bias and max |V - T(V)| for the side to move,
        and whether RMSE exceeds ``alert``."""
        out = {}
        for key, snaps in self.boards.items():
            boards = [from_snapshot(s) for s in snaps]
            movers = [b.active_player for b in boards]
            probs = net.relative_probs(encode_batch(boards))
            v = np.array([to_absolute(probs[i], b)[m]
                          for i, (b, m) in enumerate(zip(boards, movers))])
            res = run_pool([b.rules for b in boards], list(range(len(boards))),
                           [net], max_turns=1, allow_unfinished=True,
                           starts=boards, search=Search(exact_root=True),
                           in_flight=in_flight)
            t = np.array([r.turn_values[0][m] for r, m in zip(res, movers)])
            d = v - t
            rmse = float(math.sqrt(np.mean(d ** 2)))
            out[key] = {"rmse": rmse, "bias": float(d.mean()),
                        "max": float(np.abs(d).max()), "alert": rmse > alert}
        return out

    def state(self):
        return {"boards": self.boards}

    @classmethod
    def from_state(cls, state):
        return cls(state["boards"])
