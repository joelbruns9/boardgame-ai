"""
Search — a learner-only semi-MDP over the macro vocabulary.

THE ROOT-PLAYER CONTRACT — binding, all four clauses
────────────────────────────────────────────────────
An earlier wording ("no minimax negation; back up the learner's own score") was
underspecified in a way that produces a real bug: ``encode_state`` defaults to
``state.actor``, so a leaf reached while seat 1 is to move silently yields *seat
1's* value, and backing that up as seat 0's is simply wrong.  So, explicitly:

1. **Tree nodes belong only to the root player ``r``.**  No node in this tree is
   ever an opponent's decision.
2. **Every other seat is sampled forward** — one policy sample per
   determinization, no nested search — until ``r`` is to act again.  An
   opponent's turn is a *transition*, not a node.
3. **Leaves are evaluated as ``encode_state(state, r)``**, never
   ``state.actor``.
4. **The backed-up scalar stays in ``r``'s frame** for the whole path.  No
   negation, no reframing.

This is legitimate because Welcome To is near-solitaire: the stacks are shared
and never consumed, so an opponent's concurrent choice cannot change your legal
moves or your immediate scoring.  Sampling them approximates an expectation, and
no equilibrium reasoning is required.

⚠ **S3 needs more than this.** Once several seats are learners, a single scalar
tree that changes viewpoint is not a fixable variant — it is the wrong object.
Run a separate root-player search per acting seat, or move to a vector-valued
Maxⁿ design.  Budget it as real work, not as a flag.

CHANCE IS KEYED, NOT MERGED
───────────────────────────
Open loop: every simulation starts from its own ``redeterminize(search_rng)``.
Keying nodes on the action sequence alone — Kingdomino's ``OpenLoopMCTS`` shape —
would let one node aggregate simulations in which **different cards were
revealed**, mixing distinct observable states into one set of priors, children
and Q.  The sharp consequence is a *selection bias*: an action that is only
**legal** under favourable reveals accumulates Q solely from the simulations
where it was legal, so it is systematically overvalued.

So a child is keyed on ``(action, observed cards)``.  Chance enters only at turn
boundaries, so the branching this adds is confined and shallow.

**⚠ The key is the viewer information state, and that includes what an opponent
made public.**  An earlier version keyed on raw card **ids** and nothing else —
no opponent sheets, no race state — with the reasoning that opponent samples are
the opponent model rather than chance, so averaging over them is the expectation
wanted.  Both halves of that were wrong in the same direction: ids are *too fine*
(15 of the 66 printed types have two physical copies, so identical-looking
reveals key apart) and the table alone is *too coarse* (two transitions differing
only in what an opponent published would merge).

Under §7.1a's progressive widening the averaging happens in the **weights** —
`count/samples` over merged children — not by collapsing distinguishable outcomes
into one child, so a distinction the viewer can see belongs in the key.

**Measured neutral to make:** over 3,840 ``(node, action, observation)`` edges at
128–256 simulations, the id key and the information key induce **exactly the same
partition** — 0 edges differ.  Reveals are near-unique, so the table already
separated every boundary crossing.  The change costs ~31 µs per simulation step
(+7% of search wall clock today, and a larger share once step 6 makes the network
cheaper) and buys a key that is still correct when children are retained.

WHAT THE SEARCH SPENDS BUDGET ON
────────────────────────────────
Three savings, all of them exact — none of them changes which position the tree
believes it is in.  ``SEARCH_SPEC.md`` §5.1 and §12 steps 1–3.

1. **Dominated actions are pruned from the search's action set**, via
   ``macro_codec.search_legal_macros``.  The park, pool and estate passes only
   forgo a strictly-increasing track, so no line that takes them can beat the
   line that does not.  ⚠ This is a *search* mask: the engine's
   ``legal_actions()``, the 684 indices and ``datagen``'s replay masks are
   untouched, and they have to be — the reference policy takes those passes
   1,853 times in the recorded corpus.  ⚠ ``PASS_ROUNDABOUT`` is pruned too, but
   behind ``SearchConfig.prune_roundabout_pass``, because it is the one that
   interacts with the *bootstrap* prior — SEARCH_SPEC §5.1a, and read
   roundabouts per game off the S0 checkpoint.
2. **Forced decisions are collapsed**, at the external root (as before) and now
   inside every simulation.  A node with one action has nothing to decide, and
   the two savings compound: after §5.1 the park and pool nodes are forced and
   so disappear entirely.  Measured over 50 GreedyBot seat-games of two-seat
   advanced play, decisions per turn fall from **2.45 to 2.14** — every one of
   the 280 park and 37 pool nodes gone, 100% of them.  The estate node narrows
   but never collapses (0 of 320); §4 has the phase-by-phase counts.
3. **The tree is re-rooted within a turn**, exactly, because no card is revealed
   between two of ``r``'s own decisions.  Measured, it preserves 8.7% of the
   budget under flat priors and **47.7% under an S0-shaped one** — the
   difference is entirely how concentrated the prior is.  ⚠ It is discarded at
   the turn boundary, where a reveal makes the child's statistics a sample from
   the wrong distribution.

THE LEAF VALUE
──────────────
``AUX_TARGETS_SPEC.md`` §5.  The rank head predicts a distribution over finishing
positions; the utility is applied *afterwards*, so the objective can be changed
on a frozen checkpoint instead of in a second training run::

    rank_value   = 2 · Σ p_r·u_r − 1
    margin_value = tanh((score_me − best opponent) · MARGIN_GAIN)
    spread       = 1 − 4·Var[u]
    floor_n      = 1 − (n+1) / (3·(n−1))          0, 1/3, 4/9 for n = 2, 3, 4
    confidence   = max(0, (spread − floor_n) / (1 − floor_n)) ** k
    leaf_value   = (1 − α)·rank_value + α·confidence·margin_value

The gate is the **variance** of the utility, not ``win_value ** 4``.  With three
or more outcomes an expectation cannot carry its own variance — a certain second
place and a coin flip between first and third have the same mean — so a
mean-based gate is a category error, and it goes gradient-dead exactly where it
matters: a 4p seat locked into second has a flat rank *and* margin suppressed by
98%, which is a common position in the last third of every game.

**No auxiliary head enters the leaf value.**  Deciding a permit is worth 0.3
points would be hand-tuning a valuation, and it is redundant: if predicted
permits are informative about score, the score head already uses them.
"""
from __future__ import annotations

import collections
import hashlib
import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import torch

from games.welcome_to import deck_knowledge as dk
from games.welcome_to import encoder as enc
from games.welcome_to import macro_codec as mc
from games.welcome_to import network as nw
from games.welcome_to import training
from games.welcome_to.constants import CARD_TABLE
from games.welcome_to.game import GameState

#: What the caller may condition on between its own decisions: the viewer
#: information state of :func:`information_key`.  Terminal states included --
#: two different endings are two different outcomes.
Observation = tuple

class Ask(IntEnum):
    """What a suspended search wants computed.

    ⚠ **Two kinds, and both must go through the same suspension point**, or the
    batching only reaches half the work.  Measured on a 32-search wave before
    ``POLICY`` was added: leaves batched perfectly at 32.0 rows per call, and
    opponent sampling was still **53% of rows and 97% of calls**, every one of
    them a batch of one, because ``_advance`` called the network directly.
    """

    #: A tree leaf.  Answered with ``(priors, value)``.
    LEAF = 0
    #: An opponent's move inside a transition.  Answered with priors alone --
    #: no value, no tree (clause 2).
    POLICY = 1


#: A suspended network request: what is wanted, of what state, for which viewer.
#: For ``LEAF`` the viewer is always the *root* player -- clause 3.
Request = tuple[Ask, GameState, int]
#: What comes back: ``(priors, value)`` for ``LEAF``, bare priors for ``POLICY``.
Result = object


@dataclass(frozen=True, slots=True)
class SearchConfig:
    simulations: int = 128
    c_puct: float = 1.5
    #: Leaf-value blend, AUX_TARGETS_SPEC §5.2.  ``k = 1`` is the starting
    #: default; ``k = 2`` is KD parity and suppresses margin to ~2% of leaf
    #: value at turn 12, which discards the better signal for most of the game.
    alpha: float = 0.5
    margin_gain: float = 2.0
    confidence_power: float = 1.0
    #: Whether the search also prunes ``PASS_ROUNDABOUT`` (SEARCH_SPEC §5.1a).
    #:
    #: On, like the other three dominated passes -- and a switch only because it
    #: is the one that interacts with a **bootstrap** prior.  With the pass gone,
    #: ``ROUNDABOUT_OPEN`` means "build one, for -3 or -8 points", and an S0 net
    #: cloned from GreedyBot arrives with ~30% of its mass there.
    #:
    #: ⚠ **That 30% is an artifact of the teacher, not a strategy.**  Opening the
    #: prompt does not touch the sheet, so GreedyBot's one-ply evaluation scores
    #: it *identically* to every other score-neutral move: measured over 1,418
    #: offers it is in the tied-best set **100%** of the time and strictly best
    #: **4%**, then chosen by a uniform pick over ~3.5 tied actions.  A
    #: behaviour-cloned prior faithfully reproduces a coin flip on a no-op.
    #:
    #: Against a synthetic 0.8 clone of that prior, pruning pins roundabouts at
    #: ``ROUNDABOUT_BOXES`` in every game, and 4x the budget does not recover it.
    #: **Do not read that as a verdict on the pruning** -- it measures the
    #: artifact, on bot games, with a stand-in value head.  A player worth having
    #: never opens the dialog without meaning to build, and for such a policy
    #: "open" and "build" are one decision and the pruning is free.  What it does
    #: say is concrete:
    #: **read roundabouts per game off the S0 checkpoint before trusting it**,
    #: and if it is near the cap, fix the prior (or turn this off for the
    #: bootstrap only), because the search itself will not out-visit 0.8 of
    #: prior mass at any budget tried.
    prune_roundabout_pass: bool = True
    # ── The chance edge: progressive widening (SEARCH_SPEC §7.1a, C) ────
    #
    #: Widening constant ``C`` in ``ceil(C * samples ** alpha)``.  ``None`` is
    #: **off**, which is the open-loop control arm: every distinct outcome gets
    #: its own child, for ever, and the tree never revisits one.
    #:
    #: ⚠ Turning this on is what makes the chance edge *finite*, and it is the
    #: only mechanism here that can produce depth.  §6.1 measured mean leaf depth
    #: 1.59, unmoved by budget or by prior sharpness, because every boundary
    #: crossing drew a key never seen before and expanded a fresh leaf.
    chance_widening: Optional[float] = None
    #: ``alpha``.  Not a free parameter: §7.6 fixes ``K ≈ N**(1/H)`` from a
    #: target depth ``H``, so ``alpha = 1/H`` and depth 2 gives 0.5.  With
    #: ``C = 1`` that is ~8 outcomes at 64 samples and ~16 at 256, which is the
    #: schedule §7.6 wrote down before anyone noticed it was widening.
    chance_widening_alpha: float = 0.5
    #: States kept per outcome, to resume a descent from when the edge is closed.
    #:
    #: ⚠ **This is the particle collection**, and its size is the belief's
    #: resolution.  One state per outcome is the collapse §13's POMCP reference
    #: warns about: samples that share a viewer information state but differ in
    #: hidden detail — most often the opponents' live current-turn sheets —
    #: would all be represented by whichever one happened to arrive first.
    max_particles: int = 4
    # ── Root exploration noise.  All off by default; S2 turns it on. ────
    #
    # SEARCH_SPEC §7.8.  Dirichlet noise perturbs the *root* prior so that a
    # confidently-wrong policy still explores: without it, an action the network
    # dislikes is never searched, so it never appears in the training data, so
    # the network goes on disliking it.  Root only, and self-play only -- at an
    # internal node it would corrupt the search's own estimates rather than
    # diversify the data it produces.
    #
    #: Absolute concentration, the form KD (0.3) and 7WD (1.8) use.
    #:
    #: ⚠ **For Welcome To this is the wrong knob, and the measurement says so.**
    #: A search root here ranges from 2 actions (`ASK_RESHUFFLE`, `CHOOSE_PLAN`)
    #: to 331 (`CHOOSE_CARDS`), mean 35.1 over 7,807 roots (§4).  One alpha
    #: across a 2-to-331 range either drowns the narrow nodes or does nothing to
    #: the wide ones.  Prefer :attr:`dirichlet_concentration`.
    dirichlet_alpha: Optional[float] = None
    #: ``alpha = dirichlet_concentration / len(legal actions)`` -- the scaled
    #: form, and the one to use.  Overrides :attr:`dirichlet_alpha` when set.
    #:
    #: **10.0 is AlphaZero's own rule, not a guess**: its published constants are
    #: ~10 / branching factor -- Go 0.03 at ~250 moves, chess 0.3 at ~35, shogi
    #: 0.15 at ~70.  Here it gives 0.20 at `CHOOSE_CARDS`, 0.37 at the surveyor,
    #: 1.9 at estate and 5.0 at reshuffle, which is the spread the game has.
    dirichlet_concentration: Optional[float] = None
    dirichlet_weight: float = 0.25
    #: Fraction of ``simulations`` a **newly noised** root must run fresh, even
    #: if re-rooting already met the budget.
    #:
    #: ⚠ **Not cosmetic: at 0 the noise is provably inert.** ``simulations`` is a
    #: target *total*, so a fully-reused root would apply noise and then run
    #: nothing -- prior perturbed, no simulation ever selecting against it, visit
    #: counts and therefore the policy target bit-identical to the un-noised
    #: search.  A fraction rather than a count because it has to track the budget
    #: for the same reason ``K`` does (§7.6): the same checkpoint searches at
    #: different budgets in self-play, arena and analysis.
    #:
    #: **1.0 is the safe default** -- a noised root pays for a fresh search, as
    #: plain AlphaZero does -- but it forfeits re-rooting's saving whenever noise
    #: is on, and that is 47.7% of the budget against an S0-shaped prior (§4).
    #: §7.8 recommends **0.25** for S2 and says why it is a starting point to be
    #: measured rather than a derived constant.
    noise_fresh_fraction: float = 1.0
    #: Visit-count temperature for :func:`play`.  0 plays the argmax.
    temperature: float = 0.0


@dataclass(slots=True)
class Outcome:
    """One sampled result of a chance edge -- ``SEARCH_SPEC.md`` §7.1a.

    ``count / samples`` is the weight: the **empirical** mass of this outcome,
    never an enumerated probability.  That is the whole of why resolution C
    dissolves §7.1a's blocker rather than answering it — an empirical weight
    never needs ``P(transition)``, which could not be computed, because it
    estimates it instead, opponent randomness and all.
    """

    #: How many sampled transitions landed on this viewer information state.
    count: int = 0
    #: Concrete states to resume a descent from: the **particle collection**.
    #: Several, because samples agreeing on what the viewer can see may still
    #: differ in what it cannot, and one canonical state would narrow the belief.
    particles: list = field(default_factory=list)
    #: Set instead of a child node when this transition ended the game.
    terminal_value: Optional[float] = None


class Node:
    """One decision by the root player ``r``."""

    __slots__ = (
        "actions", "prior", "visits", "total", "children", "expanded", "noised",
        "edge_visits", "outcomes", "edge_exact",
    )

    def __init__(self, actions: np.ndarray, prior: np.ndarray) -> None:
        self.actions = actions
        self.prior = prior
        self.visits = np.zeros(len(actions), dtype=np.float64)
        self.total = np.zeros(len(actions), dtype=np.float64)
        self.children: dict[tuple[int, Observation], "Node"] = {}
        self.expanded = True
        #: Whether root noise has already been mixed into :attr:`prior`.  A
        #: retained subtree becomes a root later in the same turn, and noising a
        #: node twice would compound the perturbation on exactly the positions
        #: re-rooting is meant to search *more* carefully.
        self.noised = False
        #: Chance bookkeeping, per action index.  Empty and unused unless
        #: ``SearchConfig.chance_widening`` is set; the control arm allocates
        #: nothing and behaves exactly as it did.
        self.edge_visits: dict[int, int] = {}
        self.outcomes: dict[int, dict[Observation, "Outcome"]] = {}
        #: Action indices whose transition is **proven** deterministic, so their
        #: support is exactly one and no widening target can ever exceed it.
        self.edge_exact: set[int] = set()

    def select(self, c_puct: float) -> int:
        """PUCT.  Returns an index into :attr:`actions`, not a macro index."""
        total_visits = self.visits.sum()
        q = np.divide(
            self.total, self.visits, out=np.zeros_like(self.total), where=self.visits > 0
        )
        exploration = c_puct * self.prior * math.sqrt(max(total_visits, 1.0)) / (
            1.0 + self.visits
        )
        return int(np.argmax(q + exploration))


# ──────────────────────────────────────────────────────────────────────────
# The leaf value
# ──────────────────────────────────────────────────────────────────────────
def confidence_floor(num_seats: int) -> float:
    """``1 − (n+1) / (3·(n−1))`` — the seat-count correction of §5.1a.

    ``1 − 4·Var[u]`` is not comparable across table sizes: a *uniform*
    distribution over three finishing positions already scores higher than one
    over two, so an uncorrected gate would call a 4p position confident merely
    for having four outcomes.  This is that uniform baseline, subtracted off.
    """
    if num_seats < 2:
        return 0.0
    return 1.0 - (num_seats + 1) / (3.0 * (num_seats - 1))


def blend_value(
    rank_probs: Sequence[float],
    scores: Sequence[float],
    num_seats: int,
    config: SearchConfig,
) -> tuple[float, dict[str, float]]:
    """The leaf scalar, in the root player's frame, plus its parts for logging.

    ``rank_probs`` and ``scores`` are the network's outputs **on the seat axis**,
    so index 0 is the root player by construction of ``encode_state(state, r)``.
    ``scores`` are the head's own units (points ÷ 80), which is why the margin
    needs no second division.
    """
    utility = training.rank_utility(num_seats)
    probs = list(rank_probs[:num_seats])
    mean_u = sum(p * u for p, u in zip(probs, utility))
    var_u = sum(p * u * u for p, u in zip(probs, utility)) - mean_u * mean_u
    rank_value = 2.0 * mean_u - 1.0

    if num_seats < 2:
        margin = 0.0
    else:
        margin = math.tanh(
            (scores[0] - max(scores[1:num_seats])) * config.margin_gain
        )

    spread = 1.0 - 4.0 * var_u
    floor = confidence_floor(num_seats)
    confidence = max(0.0, (spread - floor) / (1.0 - floor)) ** config.confidence_power
    value = (1.0 - config.alpha) * rank_value + config.alpha * confidence * margin
    return value, {
        "rank_value": rank_value,
        "margin": margin,
        "var_u": var_u,
        "confidence": confidence,
    }


def terminal_value(state: GameState, root: int, config: SearchConfig) -> float:
    """The same blend, on a finished game, where the distribution is a point mass.

    Computed rather than looked up so that a terminal leaf and an evaluated leaf
    are on one scale — a terminal node valued differently from its own predicted
    value would put a step discontinuity at the end of every line.
    """
    seats = state.config.players
    order = enc.seat_order(state, root)
    distribution = training.rank_distributions(state)[root]
    scores = state.scores()
    ordered = [scores[s] / training.SCORE_SCALE for s in order]
    return blend_value(distribution, ordered, min(seats, enc.MAX_SEATS), config)[0]


# ──────────────────────────────────────────────────────────────────────────
# The evaluator
# ──────────────────────────────────────────────────────────────────────────
class NetEvaluator:
    """Wraps the network: one forward gives priors, the leaf value and a policy.

    Every call is ``encode_state(state, viewer)`` with an explicit viewer —
    clause 3 of the contract.  The default-to-``state.actor`` behaviour is never
    used here, and that is the whole point of passing it.
    """

    def __init__(
        self,
        net: nw.WelcomeToNet,
        device: Optional[torch.device] = None,
        config: Optional[SearchConfig] = None,
    ) -> None:
        self.net = net
        self.device = device or next(net.parameters()).device
        self.config = config or SearchConfig()
        #: Forward passes made.  ``THROUGHPUT_LEVERS.md`` §2.1's vocabulary, kept
        #: apart on purpose: ``calls`` is evaluator invocations, ``rows`` is
        #: leaves, and ``batch_widths`` is the distribution of rows per call.
        #: Reporting a batch-width improvement as a throughput win is the named
        #: failure mode there, and it needs both numbers to avoid.
        self.calls = 0
        self.rows = 0
        self.batch_widths: collections.Counter[int] = collections.Counter()

    @torch.no_grad()
    def _forward_many(self, requests: Sequence[Request]) -> dict[str, torch.Tensor]:
        """One network call for ``requests``.  The only place the net is run."""
        self.net.eval()
        self.calls += 1
        self.rows += len(requests)
        self.batch_widths[len(requests)] += 1
        columns = zip(*(enc.encode_state(state, viewer) for state, viewer in requests))
        tensors = [
            torch.as_tensor(np.stack(column)).float().to(self.device)
            for column in columns
        ]
        return self.net(*tensors)

    def answer_batch(self, requests: Sequence[Request]) -> list[Result]:
        """Answer a **mixed** batch of requests from one forward pass.

        ⚠ **Mixed on purpose.** A leaf wants ``(priors, value)`` and an opponent
        wants priors alone, but both come out of the *same* heads of the *same*
        forward, so splitting the wave by kind would just halve the batch width.
        Measured: splitting gave a mean batch of 12.1 where the wave had 32
        searches live; answering them together gives the full width.

        One interpretation, shared by the batched and single paths, so a batch of
        1 and a batch of 32 cannot drift apart in how they read the heads — a
        second reading that agrees today is a parity bug waiting to happen.

        ⚠ **This does not batch the encoder**, and cannot: ``encode_state`` is
        per-row Python and stays per-row. Measured, it is 30% of search wall
        clock against the forward's 30%, so it becomes the larger share exactly
        when batching starts working (``THROUGHPUT_LEVERS.md`` §4.2).
        """
        if not requests:
            return []
        rows = [(state, viewer) for _, state, viewer in requests]
        out = self._forward_many(rows)
        logits = out["policy_logits"].cpu().numpy()

        wants_value = [i for i, (kind, _, _) in enumerate(requests) if kind is Ask.LEAF]
        values: dict[int, float] = {}
        if wants_value:
            seats = [
                min(requests[i][1].config.players, enc.MAX_SEATS) for i in wants_value
            ]
            mask = np.zeros((len(wants_value), training.MAX_RANKS), dtype=np.float32)
            for row, count in enumerate(seats):
                mask[row, :count] = 1.0
            rank_probs = nw.rank_probabilities(
                out["rank_logits"][wants_value],
                torch.as_tensor(mask).to(self.device),
            ).cpu().numpy()
            scores = out["score"][wants_value].cpu().numpy()
            for row, index in enumerate(wants_value):
                values[index], _ = blend_value(
                    rank_probs[row], scores[row], seats[row], self.config
                )

        results: list[Result] = []
        for index, (kind, state, _) in enumerate(requests):
            priors = _masked_softmax(logits[index], mc.legal_mask(state))
            results.append((priors, values[index]) if kind is Ask.LEAF else priors)
        return results

    def evaluate_batch(self, rows: Sequence[tuple[GameState, int]]) -> list[Result]:
        """``(priors, value)`` per row, from one forward pass."""
        return self.answer_batch([(Ask.LEAF, state, viewer) for state, viewer in rows])

    def evaluate(self, state: GameState, viewer: int) -> tuple[np.ndarray, float]:
        """``(priors over the legal macros, leaf value in ``viewer``'s frame)``.

        Kept as the single-leaf seam that tests and variant evaluators override.
        """
        return self.evaluate_batch([(state, viewer)])[0]

    def policy_batch(
        self, rows: Sequence[tuple[GameState, int]]
    ) -> list[np.ndarray]:
        """Bare policies for several states at once.  No value, no tree."""
        return self.answer_batch([(Ask.POLICY, state, viewer) for state, viewer in rows])

    def policy(self, state: GameState, viewer: int) -> np.ndarray:
        """The bare policy, for sampling an opponent forward.  No value, no tree."""
        return self.policy_batch([(state, viewer)])[0]


def _masked_softmax(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    masked = np.where(legal, logits, -np.inf)
    shifted = masked - masked.max()
    exp = np.exp(shifted, where=legal, out=np.zeros_like(shifted))
    total = exp.sum()
    if total <= 0.0:  # every legal logit underflowed
        out = legal.astype(np.float32)
        return out / max(out.sum(), 1.0)
    return (exp / total).astype(np.float32)


#: An opponent model: given a state and the seat to move, apply exactly one
#: move, **yielding** whatever it needs computed on the way.  A generator rather
#: than a plain callable so that opponent sampling suspends at the same point a
#: leaf does and pools into the same wave (:class:`Ask`).
OpponentPolicy = Callable[[GameState, int, random.Random], Iterator[Request]]


def sampling_opponent() -> OpponentPolicy:
    """Sample one move from the network's own policy — clause 2, no nested search.

    The priors are masked by ``legal_mask``, not ``search_legal_macros``: an
    opponent is a model of what the other seat will *actually* do, and §5.1's
    dominance pruning is a statement about how the search spends its budget, not
    about what other people play.
    """

    def move(state: GameState, seat: int, rng: random.Random) -> Iterator[Request]:
        probs = yield (Ask.POLICY, state, seat)
        index = rng.choices(range(len(probs)), weights=probs, k=1)[0]
        mc.apply_macro(state, index)

    return move


# ──────────────────────────────────────────────────────────────────────────
# Driving searches -- one, or a wave of them
#
# ``THROUGHPUT_LEVERS.md`` §2.1's vocabulary, since it is what these two are
# about and the words drift between projects:
#
#   simulation   one root-to-leaf descent
#   leaf         a descent that needs a network evaluation -- not every
#                simulation produces one, because a terminal descent is scored
#                by the rules and an already-expanded subtree produces nothing
#   batch width  leaves summed **across games** in one evaluator call.  The only
#                one of these the device ever sees, and the only one these two
#                functions move
#
# ⚠ **Batch width is not throughput** (§4.7).  Measured on this search: the
# fixed cost of a forward is worth ~19 rows, and the forward is 30% of search
# wall clock, so driving it to zero is worth **+43% at the very most** and a
# batch past ~32 buys almost nothing.  ``encode_state`` is the other 30% and
# does not batch away at all.  Report leaves per second next to any batch-width
# number, or the win will not be there.
# ──────────────────────────────────────────────────────────────────────────
def _answer(request: Request, evaluator: NetEvaluator) -> Result:
    kind, state, viewer = request
    if kind is Ask.LEAF:
        return evaluator.evaluate(state, viewer)
    return evaluator.policy(state, viewer)


def drive(generator: Iterator[Request], evaluator: NetEvaluator):
    """Run one search generator to completion, one request at a time.

    The un-batched path, and the one every existing caller takes.  It goes
    through ``evaluate`` and ``policy`` rather than their batch forms so that the
    single-request seams subclasses override stay the seams.
    """
    try:
        request = next(generator)
        while True:
            request = generator.send(_answer(request, evaluator))
    except StopIteration as done:
        return done.value


def run_searches(
    generators: Sequence[Iterator[Request]],
    evaluator: NetEvaluator,
    max_batch: int = 32,
) -> list:
    """Run several search generators together, pooling their leaves into batches.

    Each round advances **every** live generator to its next leaf, then evaluates
    those leaves together.  So batch width is the number of searches still
    running, capped by ``max_batch`` — which is a ceiling, not a target.

    ⚠ **This is a class A lever** (``THROUGHPUT_LEVERS.md`` §1): it changes when
    work happens, never what work happens.  Each generator is advanced in a fixed
    order and receives exactly its own result, so the sequence of values a search
    sees is the sequence it would have seen alone.

    ⚠ **With one caveat, and it is measured rather than assumed.** A batched
    forward is not bit-identical to a single one — float reductions run in a
    different order — so priors and values differ by ~1e-7 between the two
    paths. That is far too small to matter as a value, and it *can* in principle
    flip a PUCT comparison that was tied to seven digits. Fingerprint discrete
    trajectories, never float targets (§A), and see
    ``test_a_wave_of_searches_agrees_with_running_them_one_at_a_time``.

    ``max_batch`` defaults to 32 because the fixed per-call cost measured worth
    ~19 rows; there is no reason to build for hundreds.
    """
    pending: list[Optional[Result]] = [None] * len(generators)
    results: list = [None] * len(generators)
    live = list(range(len(generators)))
    started = [False] * len(generators)

    while live:
        requests: list[tuple[int, Request]] = []
        still_live: list[int] = []
        for index in live:
            generator = generators[index]
            try:
                if started[index]:
                    request = generator.send(pending[index])
                else:
                    request = next(generator)
                    started[index] = True
            except StopIteration as done:
                results[index] = done.value
                continue
            requests.append((index, request))
            still_live.append(index)
        live = still_live
        if not requests:
            break
        # leaves and opponent policies go in the SAME call: they read the same
        # heads of the same forward, so splitting by kind would only halve the
        # batch width (measured 12.1 against 32 live searches)
        for start in range(0, len(requests), max_batch):
            chunk = requests[start : start + max_batch]
            answers = evaluator.answer_batch([request for _, request in chunk])
            for (index, _), answer in zip(chunk, answers):
                pending[index] = answer
    return results


def trajectory_fingerprint(actions: Sequence[int], visits: Sequence[Sequence[float]] = ()) -> str:
    """A digest of the **discrete** outputs of a run, for class A verification.

    ``THROUGHPUT_LEVERS.md`` §A: two geometry configurations must produce
    identical games from identical seeds, and the way to check that is to
    fingerprint actions and visit counts — **never the float targets**, which
    legitimately drift when a batch shape changes the order of float reductions.
    Visit counts are integers held in a float array, so they digest safely; the
    priors and values they were computed from do not.
    """
    hasher = hashlib.blake2b(digest_size=16)
    for action in actions:
        hasher.update(int(action).to_bytes(4, "little", signed=True))
    for row in visits:
        hasher.update(b"|")
        for count in row:
            hasher.update(int(count).to_bytes(4, "little", signed=True))
    return hasher.hexdigest()


# ──────────────────────────────────────────────────────────────────────────
# Re-rooting: proving a retained subtree belongs to the state being searched
# ──────────────────────────────────────────────────────────────────────────
#: Every field of :class:`~games.welcome_to.sheet.Sheet`, in order.
#:
#: ⚠ **Held against ``dataclasses.fields(Sheet)`` by a test**, because this was a
#: hand-written list and a hand-written list of somebody else's fields rots
#: silently. Add a field to ``Sheet`` and both keys built from this would simply
#: stop distinguishing on it: ``_position_key`` would re-root onto a subtree from
#: a different position and ``information_key`` would merge two positions under
#: widening, with nothing failing. The steps 1-3 review found one instance of
#: exactly that shape; this closes the class.
_SHEET_FIELDS: tuple[str, ...] = (
    "numbers",
    "is_bis",
    "written_turn",
    "fences",
    "top_fences",
    "parks",
    "pools",
    "estate_marks",
    "temps",
    "bis_marks",
    "permits",
    "roundabouts",
)


def _freeze(value):
    return tuple(_freeze(item) for item in value) if isinstance(value, list) else value


def _sheet_key(sheet) -> tuple:
    """Every field of a player sheet, as something hashable."""
    return tuple(_freeze(getattr(sheet, name)) for name in _SHEET_FIELDS)


def _position_key(state: GameState, root: int) -> tuple:
    """A full identity for the position a retained subtree was gathered at.

    ⚠ **Not** the viewer information-state key of ``SEARCH_SPEC.md`` §12.1, and
    it must not be mistaken for one or grow into one.  That key has to be
    *viewer-relative* because it labels chance children built from
    determinizations; this one only ever compares two of the caller's **own real
    game states**, so reading hidden fields is not just harmless, it is what
    makes the check total: a key that hid something could match two positions
    that differ, which is the one failure mode re-rooting must not have.

    Deck *order* is excluded on purpose — the unrevealed order is re-determinized
    per simulation anyway, so requiring it to match would refuse every legitimate
    re-root the moment a caller re-seeded.  Deck **composition** is emphatically
    *not* excluded, and an earlier version of this function got that wrong: it
    carried only ``deck_pos`` and ``len(discard)``, so swapping an undrawn card
    with a discarded one of a different printed type left the key unchanged.
    Those are different positions — different reveal distribution, different
    ``deck_composition`` in the encoder, so different priors and values — and
    they must not share a subtree.  Sorted physical ids are stricter than the
    printed-type histogram that would be semantically sufficient, and cost a
    sort of ~80 ints against a whole search.
    """
    ctx = state.ctx
    return (
        root,
        state.turn,
        state.actor,
        int(state.phase),
        state.deck_pos,
        tuple(sorted(state.deck[state.deck_pos :])),
        tuple(sorted(state.discard)),
        state.plan_ids,
        tuple(tuple(sorted(turns.items())) for turns in state.plan_turns),
        tuple(state.table_cards(root)),
        tuple(_sheet_key(sheet) for sheet in state.sheets),
        tuple(_sheet_key(sheet) for sheet in state.public_sheets),
        tuple(state.turn_choice),
        state.reshuffle_next_turn,
        tuple(sorted(state.reshuffle_votes.items())),
        (
            ctx.slot,
            ctx.number,
            ctx.effect,
            ctx.last_house,
            ctx.built_roundabout,
            ctx.roundabout_declined,
            ctx.refused,
            ctx.plan_slot,
            ctx.pending_sizes,
            ctx.chosen_estates,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────
# The viewer information-state key -- SEARCH_SPEC.md §12.1
# ──────────────────────────────────────────────────────────────────────────
def _printed(card: Optional[int]) -> Optional[tuple[int, int]]:
    """A card's printed face — ``(number, effect)`` — never its physical id.

    ⚠ **This is the whole point of the type mapping.** 15 of the 66 printed
    types have two physical copies, so keying on ids splits two reveals that
    look identical to every player at the table into different children. It
    costs nothing while children are never reused — measured, 0 spurious splits
    in 60 samples — and becomes wrong the moment §7's retained chance children
    exist, which is why §6.1 called fixing it a prerequisite for §7.
    """
    if card is None:
        return None
    number, effect = CARD_TABLE[card]
    return int(number), int(effect)


def information_key(state: GameState, viewer: int) -> tuple:
    """What ``viewer`` knows, and **nothing** it does not — §12.1's key.

    This is the key a chance child is merged on under §7.1a's progressive
    widening: two sampled transitions are the same child exactly when this
    agrees. It therefore has to be right in *both* directions. Too coarse and
    distinct positions merge, which biases the empirical weights. Too fine — and
    this is the failure that actually happened, with raw card ids — and
    identical-looking outcomes split, so ``count/samples`` never converges
    because nothing ever collides.

    ⚠ **Do not confuse this with** :func:`_position_key`, immediately above.
    They look alike and are opposites. ``_position_key`` compares two of the
    caller's *own real states* to guard re-rooting, so it reads hidden fields
    deliberately and totality is its correctness condition. This one labels
    determinizations, so reading a hidden field is a **clairvoyance bug**: the
    tree would key on something the player cannot see, and the trained policy
    would inherit the cheat.

    Included, per §12.1: the viewer's **live** sheet; every opponent's
    **turn-start public snapshot**; the **printed types** on the table; the plan
    validations and race state the viewer may see; the visible deck and discard
    composition; the viewer's **own** reshuffle vote; turn, phase and the
    within-turn context that belongs to the viewer.

    Excluded, and each for a reason that has bitten something before:

    - **future deck order** — §7.3's non-anticipativity rule; a key carrying it
      is textbook strategy fusion;
    - **the opponents' live current-turn sheets** — ``sheet_for`` returns the
      public snapshot for everyone but the viewer, which is what
      ``Houses::getOfPlayer`` does;
    - **``reshuffle_next_turn``**, the table-wide OR — private mid-turn
      (``ENCODER_V2_SPEC.md`` §9.3a): reading it would tell a later serial actor
      that an earlier one voted yes, which nobody knows in the concurrent game.
      The viewer's own vote goes in; the aggregate does not;
    - **the turn context of an actor who is not the viewer** — ``ctx`` holds the
      slot taken, the number and where it went, which is precisely this turn's
      hidden write. In the search this never arises, because a child is keyed
      only where the root player is to act; it is excluded anyway, because a key
      that is only safe when its caller behaves is not safe.
    """
    ctx = state.ctx
    own_turn = state.actor == viewer
    return (
        viewer,
        state.turn,
        state.actor,
        int(state.phase),
        # ⚠ **The whole frozen config, not just the seat count.** This carried
        # `players` alone, and the encoder also reads `advanced`, `expert` and
        # `solo` -- so flipping `advanced` left the key identical and the
        # encoding different, breaking the containment §3.1 of the review brief
        # claimed. Dormant while a search never changes configuration mid-flight,
        # and false as stated, which is worse: the invariant is what licenses
        # sharing one node's priors across a whole particle collection.
        state.config,
        # raw length as well as the histogram: `discard_composition` returns
        # zeros in expert mode, where the discard is not attributable, so the
        # count is not recoverable from it
        len(state.discard),
        # the table, as printed faces
        tuple(_printed(card) for card in state.table_cards(viewer)),
        # sheets: live for the viewer, turn-start snapshots for everyone else
        tuple(
            _sheet_key(state.sheet_for(viewer, player))
            for player in range(state.config.players)
        ),
        # the race, as this viewer may see it
        state.plan_ids,
        tuple(
            tuple(sorted(state.plan_turns_for(viewer, slot).items()))
            for slot in range(3)
        ),
        # the deck, as composition -- never as order
        state.deck_remaining,
        _composition_key(dk.deck_composition(state, viewer)),
        _composition_key(dk.discard_composition(state, viewer)),
        state.solo_card_drawn,
        # the viewer's own private bookkeeping, and nobody else's
        state.reshuffle_vote_for(viewer),
        state.turn_choice[viewer],
        (
            ctx.slot,
            ctx.number,
            None if ctx.effect is None else int(ctx.effect),
            ctx.last_house,
            ctx.built_roundabout,
            ctx.roundabout_declined,
            ctx.refused,
            ctx.plan_slot,
            ctx.pending_sizes,
            ctx.chosen_estates,
        )
        if own_turn
        else None,
    )


def _composition_key(counts: np.ndarray) -> bytes:
    """A ``(15, 6)`` count matrix as something hashable, cheaply.

    ``tobytes`` rather than a tuple of ints: this runs once per simulation step,
    and 90 cells x 2 matrices is 180 ``int()`` calls that measured as most of the
    key's cost outside numpy.  Counts are small non-negative integers, so int16
    is exact.
    """
    return counts.astype(np.int16).tobytes()


@dataclass(frozen=True, slots=True)
class _Retained:
    """A subtree kept between two of the root player's decisions in one turn."""

    node: Node
    root: int
    key: tuple


# ──────────────────────────────────────────────────────────────────────────
# Search
# ──────────────────────────────────────────────────────────────────────────
class MCTS:
    def __init__(
        self,
        evaluator: NetEvaluator,
        config: Optional[SearchConfig] = None,
        opponent: Optional[OpponentPolicy] = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or evaluator.config
        self.opponent = opponent or sampling_opponent()
        self._retained: Optional[_Retained] = None
        #: Counters, for the measurements SEARCH_SPEC §4 and §12 record.
        #: ``simulations_reused`` is the work re-rooting saved: visits a root
        #: already had when :meth:`search` was entered.
        self.simulations_run = 0
        self.simulations_reused = 0
        self.reroots = 0

    def _fresh_after_noise(self) -> int:
        """How many simulations a newly-noised root must run, however much it reused.

        ⚠ **Without a floor here, root noise can be completely inert.** The
        budget is a target *total*, so a retained root whose visits already meet
        ``simulations`` runs zero fresh simulations — the prior is perturbed and
        then nothing ever selects against it, leaving the visit counts, and so
        the policy target, bit-identical to the un-noised search.  Reproduced
        before it was fixed: ``prior_changed True, fresh_simulations 0,
        visits_changed False``.  Partially-reused roots have the same problem in
        weaker form, since how much exploration a decision gets then depends on
        how concentrated the *previous* decision's tree happened to be.
        """
        fraction = max(0.0, min(1.0, self.config.noise_fresh_fraction))
        return int(math.ceil(self.config.simulations * fraction))

    def _search_actions(self, state: GameState) -> list[int]:
        """The search's action set at ``state`` — §5.1 pruning, under config.

        Every place that asks "what may the search do here?" goes through this
        one method, because :meth:`_collapse_forced`, :meth:`play` and
        :meth:`_within_turn_successor` must agree exactly on which decisions are
        forced; if they disagreed, a retained subtree would sit one move away
        from the position it is compared against and re-rooting would silently
        stop working.
        """
        return mc.search_legal_macros(state, self.config.prune_roundabout_pass)

    # ── the transition: everything between two of r's decisions ──────────
    def _advance(
        self, state: GameState, root: int, rng: random.Random
    ) -> Iterator[Request]:
        """Sample every other seat forward until ``root`` is to act, or the end.

        Clause 2.  An opponent's turn is a transition, not a node, so nothing
        here is stored and nothing here is searched.
        """
        guard = 0
        while not state.is_terminal and state.actor != root:
            yield from self.opponent(state, state.actor, rng)
            guard += 1
            if guard > 5000:  # pragma: no cover - a stuck engine, not a rules case
                raise RuntimeError("opponents did not yield the turn")
        # ⚠ **Terminal states are keyed like any other, and must be.** This
        # returned a bare ``()`` for every ending, which is harmless in the
        # open-loop arm -- a terminal transition stores no child, so the key is
        # never looked up -- and silently wrong under §7.1a's widening, where the
        # observation *is* the outcome key. Measured before the fix: 255
        # distinct endings of one edge merged into a single outcome carrying
        # whichever final score happened to be computed last. Different endings
        # have different scores; collapsing them is precisely the belief
        # collapse particles exist to prevent.
        return information_key(state, root)

    def _leaf(self, state: GameState, root: int) -> tuple[Optional[Node], float]:
        """Blocking :meth:`_leaf_gen`, for callers with nothing to batch against."""
        return drive(self._leaf_gen(state, root), self.evaluator)

    def _leaf_gen(self, state: GameState, root: int) -> Iterator[Request]:
        """Expand one leaf.  **Yields** its evaluation request, receives the result.

        ⚠ This ``yield`` is the whole point of the generator shape, and it is
        deliberately the *only* one in the search.  It is what lets a driver hold
        several searches at their leaves at once and evaluate them in one call
        (:func:`run_searches`), without the search knowing whether it is being
        run alone or in a wave.  Without it, cross-game batching means rewriting
        the descent; with it, it is a driver.
        """
        if state.is_terminal:
            return None, terminal_value(state, root, self.config)
        priors, value = yield (Ask.LEAF, state, root)
        # ``search_legal_macros``, not ``legal_macros``: the dominated passes of
        # SEARCH_SPEC §5.1 are pruned from the *search's* action set only.  The
        # priors themselves stay masked by the full legal set and are
        # renormalised over what survives, so a prior mass the network puts on a
        # pruned pass is redistributed rather than silently kept.
        actions = np.asarray(self._search_actions(state), dtype=np.int64)
        prior = priors[actions]
        total = prior.sum()
        prior = prior / total if total > 0 else np.full(len(actions), 1.0 / len(actions))
        return Node(actions, prior.astype(np.float64)), value

    def _collapse_forced(
        self,
        state: GameState,
        root: int,
        rng: random.Random,
        turn: int,
        observation: Observation,
    ) -> Iterator[Request]:
        """Apply every forced decision in a row, storing no node for any of them.

        A node with one action has nothing to decide: PUCT over a one-element
        array always returns index 0, and the network call that built its priors
        bought nothing.  ``MCTS.play`` has always short-circuited this at the
        external root; this is the same saving *inside* the tree, and after §5.1
        pruning it is where park and pool nodes go -- their pass is dominated and
        the build is the only action left, so the node disappears entirely.

        The forced action is still **applied**; only the node is skipped.  Its
        statistics would be a copy of its single child's, and the backup path
        must therefore not contain it.

        ⚠ **Collapsing stops at the turn boundary**, hence the ``turn`` guard.
        Within a turn nothing is revealed, so every state between two of ``r``'s
        decisions is deterministic given the action and no information is lost by
        not keying it.  Across a boundary a reveal intervenes: two forced steps
        that each crossed one would put *two* chance outcomes behind a child
        keyed on the second alone, merging distinct observable states into one
        node -- precisely the selection bias this file's header exists to
        prevent.  So the caller's own action may end the turn (its reveal is
        keyed, as before), but no forced action is skipped past that point.
        """
        while (
            state.turn == turn
            and not state.is_terminal
            and state.actor == root
            and mc.is_macro_root(state)
        ):
            forced = self._search_actions(state)
            if len(forced) != 1:
                break
            mc.apply_macro(state, forced[0])
            observation = yield from self._advance(state, root, rng)
        return observation

    # ── the chance edge (SEARCH_SPEC §7.1a, resolution C) ────────────────
    def _edge_is_closed(self, node: Node, index: int) -> bool:
        """Whether this edge has as many distinct outcomes as it is yet allowed.

        ``ceil(C · traversals**alpha)``.  Closed means "stop sampling new futures
        and start re-using the ones you have", which is what turns an unbounded
        chance fan-out into a finite set of children that a descent can revisit.

        ⚠ **A cap alone cannot close an edge whose support is smaller than the
        cap**, and an earlier version claimed otherwise. ``len(outcomes)`` stops
        at the real support while ``ceil(C·n**alpha)`` keeps growing, so the
        predicate goes false for ever and the edge re-samples on almost every
        traversal. Measured before the fix: support-one edges reused **6 of 77**
        traversals (7.8%), against 287 of 427 (67%) on multi-outcome edges —
        while the docstring said a within-turn edge "reaches its cap immediately
        and every later descent resumes from a particle".

        So determinism is **proven, not inferred from collisions**: a transition
        that did not change the turn and did not end the game consumed no
        randomness at all, so its support is exactly one, and the edge is closed
        permanently. That is knowable here; general finite support is not.
        Ordinary progressive widening cannot detect exhaustion from collisions
        alone, and this does not pretend to.
        """
        if self.config.chance_widening is None:
            return False
        outcomes = node.outcomes.get(index)
        if not outcomes:
            return False
        if index in node.edge_exact:
            return True
        # ⚠ The cap counts **traversals**, while a weight counts **fresh draws**.
        # They must be different numbers.  If reuse fed the cap's counter the
        # edge would never widen again; if reuse fed a weight's counter the
        # weights would be a Polya urn -- sampling in proportion to a count that
        # the sampling itself increments amplifies whichever outcome happened to
        # arrive first, and converges to a random limit rather than the truth.
        visits = node.edge_visits.get(index, 0)
        allowed = math.ceil(
            self.config.chance_widening * visits ** self.config.chance_widening_alpha
        )
        return len(outcomes) >= max(allowed, 1)

    def _pick_outcome(
        self, node: Node, index: int, rng: random.Random
    ) -> tuple[Observation, Outcome]:
        """Sample a retained outcome **by its empirical weight**, ``count/draws``.

        A chance node averages; it never PUCTs.  ``count`` is the number of times
        this outcome came out of a *fresh* draw from the real transition, and is
        never touched by reuse, so ``count / sum(counts)`` is an unbiased
        estimate of its mass.  Sampling descents in that proportion is what makes
        the visit distribution over children match the transition distribution,
        so the ordinary running mean at the parent action *is* the
        weight-weighted mean and no special backup is needed.

        With near-unique reveals every draw is its own outcome, all counts are 1,
        and this is uniform over the retained set -- §7.1's ``count/K`` collapsing
        to ``1/K``.  With a three-card deck the draws collide onto the six real
        reveals and the counts become the distribution.
        """
        outcomes = node.outcomes[index]
        keys = list(outcomes)
        weights = [outcomes[key].count for key in keys]
        chosen = rng.choices(keys, weights=weights, k=1)[0]
        return chosen, outcomes[chosen]

    def _record_outcome(
        self,
        node: Node,
        index: int,
        observation: Observation,
        state: GameState,
        rng: random.Random,
        exact: bool,
    ) -> Optional[Outcome]:
        """Merge a freshly sampled transition into this edge, or start a child.

        Merging is the mechanism, not an optimisation: it is what makes
        ``count/draws`` converge on the true mass.

        ``exact`` says this transition consumed no randomness — the turn did not
        change and the game did not end — which **proves** the edge's support is
        one and closes it for good.

        ⚠ **Particles are replaced by reservoir sampling, not dropped.** Keeping
        the first ``max_particles`` and discarding every later sample is unbiased
        in expectation but freezes the conditional belief on whichever
        determinizations happened to arrive first; the collection never improves
        however long the edge is searched. Reservoir replacement keeps a uniform
        sample of *all* fresh draws for the same memory.
        """
        if self.config.chance_widening is None:
            return None
        node.edge_visits[index] = node.edge_visits.get(index, 0) + 1
        outcomes = node.outcomes.setdefault(index, {})
        outcome = outcomes.get(observation)
        if outcome is None:
            outcome = outcomes[observation] = Outcome()
        outcome.count += 1
        if exact and not state.is_terminal:
            node.edge_exact.add(index)
        if not state.is_terminal:
            cap = self.config.max_particles
            if len(outcome.particles) < cap:
                outcome.particles.append(state.copy())
            else:
                # reservoir: the k-th draw replaces a uniform slot with p = cap/k
                slot = rng.randrange(outcome.count)
                if slot < cap:
                    outcome.particles[slot] = state.copy()
        return outcome

    def _simulate(self, state: GameState, node: Node, root: int, rng: random.Random) -> Iterator[Request]:
        """One root-to-leaf descent.  A generator, via :meth:`_leaf_gen`."""
        path: list[tuple[Node, int]] = []
        while True:
            index = node.select(self.config.c_puct)
            path.append((node, index))
            action = int(node.actions[index])

            if self._edge_is_closed(node, index):
                # ⚠ The saving *and* the depth: no transition is sampled, so no
                # opponent is evaluated and no card is drawn.  The descent
                # resumes from a particle of an outcome already seen, which is
                # the only way a chance child is ever revisited (§7.1a).
                observation, outcome = self._pick_outcome(node, index, rng)
                node.edge_visits[index] = node.edge_visits.get(index, 0) + 1
                if outcome.terminal_value is not None:
                    value = outcome.terminal_value
                    break
                state = rng.choice(outcome.particles).copy()
                node = node.children[(action, observation)]
                continue

            turn = state.turn
            mc.apply_macro(state, action)
            observation = yield from self._advance(state, root, rng)
            observation = yield from self._collapse_forced(
                state, root, rng, turn, observation
            )
            outcome = self._record_outcome(
                node,
                index,
                observation,
                state,
                rng,
                exact=state.turn == turn and not state.is_terminal,
            )

            key = (action, observation)
            child = node.children.get(key)
            if child is None:
                child, value = yield from self._leaf_gen(state, root)
                if child is not None:
                    node.children[key] = child
                elif outcome is not None:
                    outcome.terminal_value = value
                break
            node = child

        for parent, index in path:
            parent.visits[index] += 1.0
            parent.total[index] += value  # clause 4: r's frame, never negated
        return value

    def search(
        self,
        state: GameState,
        root: Optional[int] = None,
        rng: Optional[random.Random] = None,
        node: Optional[Node] = None,
    ) -> tuple[np.ndarray, np.ndarray, Node]:
        """``(macro indices, visit counts, root node)``.

        ``state`` is not modified.  Each simulation gets its own
        ``redeterminize(rng)``, and ``rng`` is advanced by the call — passing the
        state's own generator would return the identical shuffle every time,
        because ``copy`` clones the RNG state exactly.

        ``node`` is a **retained subtree** whose statistics were gathered at this
        very state (see :meth:`play`).  The budget is then a *target total*:
        ``simulations`` counts the visits the root ends with, not the ones this
        call adds, so re-rooting spends the saved simulations on depth rather
        than on repeating work.  Passing a node that does not belong to ``state``
        is a correctness bug, not a heuristic loss — :meth:`play` is the only
        caller that should build one, and it verifies the correspondence.
        """
        return drive(self.search_gen(state, root, rng, node), self.evaluator)

    def search_gen(
        self,
        state: GameState,
        root: Optional[int] = None,
        rng: Optional[random.Random] = None,
        node: Optional[Node] = None,
    ) -> Iterator[Request]:
        """:meth:`search` as a generator, for :func:`run_searches` to interleave.

        Identical work in an identical order; the only difference is that the
        leaf evaluations leave through a ``yield`` instead of a method call, so
        somebody else may decide when and with what else they are computed.
        """
        root = state.actor if root is None else root
        rng = rng or random.Random()
        if state.actor != root:
            raise ValueError("search starts at a state the root player is to act in")

        if node is None:
            node, _ = yield from self._leaf_gen(state, root)
            if node is None:
                raise ValueError("cannot search a finished game")
        noised = self._apply_root_noise(node, rng)

        reused = int(node.visits.sum())
        budget = max(self.config.simulations - reused, 0)
        if noised:
            budget = max(budget, self._fresh_after_noise())
        self.simulations_reused += reused
        self.simulations_run += budget
        for _ in range(budget):
            yield from self._simulate(state.redeterminize(rng), node, root, rng)
        return node.actions, node.visits, node

    def _root_alpha(self, width: int) -> Optional[float]:
        """The Dirichlet concentration for a root of ``width`` legal actions.

        ``dirichlet_concentration`` wins when both are set, because the scaled
        form is the correct one here and an absolute alpha is the escape hatch,
        not the other way round.
        """
        if self.config.dirichlet_concentration is not None:
            return self.config.dirichlet_concentration / width
        return self.config.dirichlet_alpha

    def _apply_root_noise(self, node: Node, rng: random.Random) -> bool:
        """Mix in root noise, once per node.  Returns whether it fired *now*."""
        if node.noised:  # a retained subtree that was noised as a root already
            return False
        if len(node.actions) < 2:
            return False
        alpha = self._root_alpha(len(node.actions))
        if alpha is None:
            return False
        noise = np.random.default_rng(rng.randrange(1 << 32)).dirichlet(
            [alpha] * len(node.actions)
        )
        weight = self.config.dirichlet_weight
        node.prior = (1.0 - weight) * node.prior + weight * noise
        node.noised = True
        return True

    # ── playing ──────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Drop the retained subtree.  Call between games; cheap and always safe.

        :meth:`play` also drops it by itself whenever the position it predicted
        is not the one it is handed, so this is belt-and-braces rather than a
        precondition — but an explicit reset says what is meant.
        """
        self._retained = None

    def play(
        self,
        state: GameState,
        root: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> int:
        """Choose one macro for the acting seat, and return it.

        A single legal move skips the search entirely.  Half of all decisions in
        this game have two options or fewer, so this is not a micro-optimisation
        — it is most of the budget.  The retained subtree is deliberately **left
        alone** on that path: :meth:`_collapse_forced` skipped the same forced
        decision inside every simulation, so the subtree kept here is already the
        one for the next *real* decision, several forced moves further on.
        """
        root = state.actor if root is None else root
        legal = self._search_actions(state)
        if len(legal) == 1:
            return legal[0]

        node = self._take_retained(state, root)
        actions, visits, node = self.search(state, root, rng, node=node)
        if self.config.temperature <= 0.0:
            choice = int(actions[int(np.argmax(visits))])
        else:
            weights = visits ** (1.0 / self.config.temperature)
            if weights.sum() <= 0.0:  # pragma: no cover - every simulation failed
                choice = int(actions[int(np.argmax(visits))])
            else:
                rng = rng or random.Random()
                choice = int(
                    actions[rng.choices(range(len(actions)), weights=weights, k=1)[0]]
                )
        self._retain(state, root, node, choice)
        return choice

    # ── deterministic within-turn re-rooting ─────────────────────────────
    def _retain(
        self, state: GameState, root: int, node: Node, choice: int
    ) -> None:
        """Keep the subtree under ``choice`` for the next decision **this turn**.

        ⚠ **Re-rooting is exact only within a turn, and must not be generalised
        to cross one.**  Inside a turn the transition is deterministic — no card
        is revealed between two of ``root``'s own decisions — so the child's
        statistics were gathered under exactly the position that is about to be
        searched, and reusing them changes nothing but the budget.  Across a turn
        boundary a reveal intervenes and those statistics were gathered under
        determinizations that no longer apply; the child would then be a sample
        from the wrong distribution, and the error would be invisible because the
        tree still *looks* well-formed.  This is the kind of optimisation someone
        extends later without noticing, which is why the guard is a hard
        precondition here and not a heuristic.
        """
        successor = self._within_turn_successor(state, root, choice)
        if successor is None:
            self._retained = None
            return
        # the same key ``_advance`` stored it under -- within a turn no opponent
        # moves and no card is revealed, so the successor's information state is
        # exactly what every simulation saw down this edge
        child = node.children.get((choice, information_key(successor, root)))
        if child is None:  # every simulation ended before reaching this child
            self._retained = None
            return
        self._retained = _Retained(
            node=child, root=root, key=_position_key(successor, root)
        )

    def _within_turn_successor(
        self, state: GameState, root: int, choice: int
    ) -> Optional[GameState]:
        """The position ``choice`` leads to, or ``None`` if it leaves the turn.

        Mirrors :meth:`_collapse_forced` exactly — the tree's child sits *after*
        the forced decisions, so the position to compare against must too.
        ``_advance`` is not run: it samples opponents, and the real game's
        opponents will not make the sampled moves, which is another way of saying
        the same thing as "this stops at the turn boundary".
        """
        turn = state.turn
        nxt = mc.step_macro(state, choice)
        while (
            nxt.turn == turn
            and not nxt.is_terminal
            and nxt.actor == root
            and mc.is_macro_root(nxt)
        ):
            forced = self._search_actions(nxt)
            if len(forced) != 1:
                break
            mc.apply_macro(nxt, forced[0])
        if nxt.is_terminal or nxt.turn != turn or nxt.actor != root:
            return None
        if not mc.is_macro_root(nxt):  # pragma: no cover - no macro ends mid-write
            return None
        return nxt

    def _take_retained(self, state: GameState, root: int) -> Optional[Node]:
        """The retained subtree if it is provably this state's, else ``None``."""
        retained = self._retained
        self._retained = None
        if retained is None or retained.root != root:
            return None
        if retained.key != _position_key(state, root):
            return None
        self.reroots += 1
        return retained.node


def visit_policy(actions: np.ndarray, visits: np.ndarray) -> np.ndarray:
    """The visit distribution as a full-width vector — S2's policy target."""
    policy = np.zeros(mc.NUM_MACRO_ACTIONS, dtype=np.float32)
    total = visits.sum()
    if total > 0:
        policy[actions] = visits / total
    return policy


def arena_player(search: MCTS):
    """Adapt a search to :mod:`games.welcome_to.arena`'s ``Player`` protocol.

    The seat is passed explicitly and used as the search root -- the arena
    substitutes one seat, so the root is that seat and never ``state.actor`` by
    default.  Clause 3 again, one layer up.

    One ``Player`` closure plays many games, and nothing here can see a game
    boundary.  It does not need to: :meth:`MCTS.play` verifies that the position
    it is handed is the one its retained subtree was gathered at, so a tree left
    over from the previous game is dropped rather than reused.  Callers that
    *can* see the boundary should still call :meth:`MCTS.reset` there, because
    saying it beats relying on it.
    """

    def move(state: GameState, seat: int, rng: random.Random) -> None:
        mc.apply_macro(state, search.play(state, root=seat, rng=rng))

    return move
