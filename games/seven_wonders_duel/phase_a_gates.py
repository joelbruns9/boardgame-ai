"""Full-scale Phase A statistical gates (CODEC_SPEC.md §4.4, §8).

Run: python -m games.seven_wonders_duel.phase_a_gates

Chi-squared acceptance without scipy: statistic < df + 4*sqrt(2*df)
(normal approximation, roughly p > 3e-4 — same convention as the Kingdomino
trainer's no-scipy checks). Failures print FAIL and exit nonzero.
"""

from __future__ import annotations

import math
import random
import sys
import time

from .data import BackType
from .engine import Action, ActionUse, apply_action
from .game import ChanceKind, Phase, new_game
from .pool import enumerate_card_reveal, resample_hidden, unseen_pool

FAILURES: list[str] = []


def _chi2(observed: dict, expected_each: float, label: str) -> None:
    df = len(observed) - 1
    stat = sum((n - expected_each) ** 2 / expected_each for n in observed.values())
    threshold = df + 4 * math.sqrt(2 * df)
    verdict = "ok" if stat < threshold else "FAIL"
    if verdict == "FAIL":
        FAILURES.append(label)
    print(f"  {label}: chi2={stat:.1f} df={df} threshold={threshold:.1f} -> {verdict}")


def _playing_game(seed=30):
    game = new_game(seed)
    while game.phase is Phase.WONDER_DRAFT:
        game.pick_wonder(game.legal_wonder_choices()[0])
    return game


def gate_determinizer_single_marginal(samples=100_000):
    """resample_hidden must reproduce the closed-loop 1/11 reveal marginal."""

    print(f"[1/3] determinizer single-slot marginal ({samples:,} samples)")
    game = _playing_game(30)
    apply_action(game, Action((4, 1), ActionUse.DISCARD_FOR_COINS))
    pool = unseen_pool(game.observation(0))
    expected_names = {name for name, _ in enumerate_card_reveal(pool, BackType.AGE_I)}
    counts: dict[str, int] = {name: 0 for name in expected_names}
    start = time.time()
    for i in range(samples):
        clone = game.clone()
        resample_hidden(clone, random.Random(i))
        counts[clone.tableau.cards[(3, 2)].card_name] += 1
    print(f"  sampled in {time.time() - start:.0f}s; support={len(counts)}")
    assert set(counts) == expected_names, "support mismatch vs closed-loop pool"
    _chi2(counts, samples / len(expected_names), "single-slot marginal")


def gate_determinizer_joint_marginal(samples=100_000):
    """Conditioned multi-reveal: the (3,2)x(3,4) joint must be uniform over
    ordered pairs of distinct pool cards (sequential chance-node semantics)."""

    print(f"[2/3] determinizer joint double-reveal marginal ({samples:,} samples)")
    game = _playing_game(30)
    apply_action(game, Action((4, 1), ActionUse.DISCARD_FOR_COINS))
    apply_action(game, Action((4, 5), ActionUse.DISCARD_FOR_COINS))
    pool = unseen_pool(game.observation(0))
    names = {name for name, _ in enumerate_card_reveal(pool, BackType.AGE_I)}
    pairs = {(a, b): 0 for a in names for b in names if a != b}
    for i in range(samples):
        clone = game.clone()
        resample_hidden(clone, random.Random(1_000_000 + i))
        key = (
            clone.tableau.cards[(3, 2)].card_name,
            clone.tableau.cards[(3, 4)].card_name,
        )
        pairs[key] += 1
    assert all(isinstance(k, tuple) for k in pairs)
    _chi2(pairs, samples / len(pairs), f"joint marginal over {len(pairs)} pairs")


def gate_great_library_uniformity(samples=20_000):
    """Empirical simulator draws must be uniform over the C(5,3)=10 subsets."""

    print(f"[3/3] Great Library empirical uniformity ({samples:,} samples)")
    base = _playing_game(400)
    for city in base.cities:
        if "The Great Library" in city.wonders:
            city.wonders.remove("The Great Library")
    base.cities[0].wonders[0:0] = ["The Great Library"]
    base.cities[0].coins = 100
    slot = base.tableau.accessible_slot_ids()[0]
    action = Action(slot, ActionUse.CONSTRUCT_WONDER, "The Great Library")
    counts: dict[frozenset, int] = {}
    for i in range(samples):
        clone = base.clone()
        clone.rng = random.Random(i)
        result = apply_action(clone, action)
        draw = next(
            e for e in result.events if e.kind is ChanceKind.GREAT_LIBRARY_DRAW
        )
        key = frozenset(draw.outcome)
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 10, f"expected 10 subsets, saw {len(counts)}"
    _chi2(counts, samples / 10, "Great Library subsets")


def main() -> int:
    gate_determinizer_single_marginal()
    gate_determinizer_joint_marginal()
    gate_great_library_uniformity()
    if FAILURES:
        print(f"FAILED gates: {FAILURES}")
        return 1
    print("All Phase A statistical gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
