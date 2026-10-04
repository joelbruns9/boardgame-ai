"""
Plan-deal curriculum (NEXT_TOY_RUN_PLAN.md §3.3, built 2026-10-03).

WHY
───
No self-play game has ever ended on three completed City Plans
(``plan_ending_fraction`` 0.0 at every iteration of v3_curriculum_01), while
high-level human games almost always do. The model never experiences a plan
ending, its value, or the race for it -- and it cannot, while most deals include
a plan it is unable to finish.

WHAT
────
A share of each iteration's games deal plans weighted toward the ones the
learner already completes: per stack, each legal plan is drawn with probability
proportional to its completion rate last iteration plus a floor. Easy plans
dominate at first; a plan enters the mix as soon as it starts completing.
Deals stay legal -- one plan per stack (owner, 2026-10-03).

It changes which positions are played, never a label: the plans are part of the
input, so values stay conditional on the deal. It is a **helper**: its share
follows the run's helper schedule down to zero (``s2_run``
``--helpers-end-iteration``), and forced-deal games are excluded from the
strength metrics.
"""
from __future__ import annotations

import hashlib
import json
from typing import Collection, Mapping, Optional, Sequence

from games.welcome_to.plans import available_plan_ids
from games.welcome_to.portable_rng import PortableRng

#: Added to every plan's completion rate, so a never-completed plan still
#: appears now and then and can start being learned.
DEFAULT_FLOOR = 0.03
_MASK64 = (1 << 64) - 1
_SELECT_DOMAIN = 0x6465_616C_7365_6C65  # "dealsele"
_DRAW_DOMAIN = 0x6465_616C_6472_6177  # "dealdraw"


def plan_weights(
    completion: Mapping[str, float], floor: float = DEFAULT_FLOOR
) -> dict[int, float]:
    """Per plan id: last iteration's learner completion rate plus ``floor``."""
    return {int(pid): float(rate) + floor for pid, rate in completion.items()}


def _weighted_choice(rng: PortableRng, items: Sequence[int], weights: Sequence[float]) -> int:
    total = sum(weights)
    target = rng.next_float() * total
    running = 0.0
    for item, weight in zip(items, weights):
        running += weight
        if target < running:
            return item
    return items[-1]


def plan_deals(
    jobs: Sequence[tuple[int, int]],
    completion: Mapping[str, float],
    *,
    fraction: float,
    seed: int,
    exclude: Collection[int] = (),
    floor: float = DEFAULT_FLOOR,
) -> dict[int, tuple[int, int, int]]:
    """``job seed -> (plan from stack 1, 2, 3)`` for a deterministic
    ``fraction`` of the jobs not in ``exclude``.

    A plan missing from ``completion`` (never dealt last iteration) gets the
    floor alone. Deterministic in its arguments, so a resumed generation
    rebuilds the identical deals.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("deal fraction must be in [0, 1]")
    if fraction == 0.0 or not completion:
        return {}
    weights = plan_weights(completion, floor)
    excluded = frozenset(exclude)
    eligible = sorted(job for job, _players in jobs if job not in excluded)
    picker = PortableRng((seed ^ _SELECT_DOMAIN) & _MASK64)
    chosen = eligible[:]
    picker.shuffle(chosen)
    chosen = sorted(chosen[: int(len(eligible) * fraction + 0.5)])
    deals: dict[int, tuple[int, int, int]] = {}
    for job in chosen:
        rng = PortableRng((job ^ _DRAW_DOMAIN) & _MASK64)
        plans = []
        for stack in (1, 2, 3):
            options = available_plan_ids(stack, True)
            plans.append(_weighted_choice(rng, options, [weights.get(p, floor) for p in options]))
        deals[job] = tuple(plans)
    return deals


def deal_digest(deals: Mapping[int, Sequence[int]]) -> str:
    rows = [[seed, list(plans)] for seed, plans in sorted(deals.items())]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def completion_from_metrics(path) -> Optional[dict[str, float]]:
    """``plan_completion_by_id`` from a generation metrics file, if present."""
    from pathlib import Path

    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("plan_completion_by_id")
