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

**Opponent samples are deliberately *not* in the key.**  They are not chance —
they are the opponent model, and averaging over them is exactly the expectation
wanted.  They also cannot create the legality-driven bias above, because nothing
an opponent does changes what ``r`` may play.

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

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from games.welcome_to import encoder as enc
from games.welcome_to import macro_codec as mc
from games.welcome_to import network as nw
from games.welcome_to import training
from games.welcome_to.game import GameState

#: What the caller observes between its own decisions: every card on the table.
#: All of them are fully identified, so this is public, and it is exactly the
#: chance outcome a turn boundary reveals.
Observation = tuple


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
    #: Root exploration noise.  Off by default; S2 turns it on.
    dirichlet_alpha: Optional[float] = None
    dirichlet_weight: float = 0.25
    #: Visit-count temperature for :func:`play`.  0 plays the argmax.
    temperature: float = 0.0


class Node:
    """One decision by the root player ``r``."""

    __slots__ = (
        "actions", "prior", "visits", "total", "children", "expanded", "noised"
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
        self.calls = 0

    @torch.no_grad()
    def _forward(self, state: GameState, viewer: int) -> dict[str, torch.Tensor]:
        self.net.eval()
        self.calls += 1
        arrays = enc.encode_state(state, viewer)
        tensors = [
            torch.as_tensor(a).unsqueeze(0).float().to(self.device) for a in arrays
        ]
        return self.net(*tensors)

    def evaluate(self, state: GameState, viewer: int) -> tuple[np.ndarray, float]:
        """``(priors over the legal macros, leaf value in ``viewer``'s frame)``."""
        out = self._forward(state, viewer)
        seats = min(state.config.players, enc.MAX_SEATS)

        mask = np.zeros(training.MAX_RANKS, dtype=np.float32)
        mask[:seats] = 1.0
        rank_probs = nw.rank_probabilities(
            out["rank_logits"], torch.as_tensor(mask).to(self.device).unsqueeze(0)
        )[0].cpu().numpy()
        scores = out["score"][0].cpu().numpy()
        value, _ = blend_value(rank_probs, scores, seats, self.config)

        legal = mc.legal_mask(state)
        logits = out["policy_logits"][0].cpu().numpy()
        priors = _masked_softmax(logits, legal)
        return priors, value

    def policy(self, state: GameState, viewer: int) -> np.ndarray:
        """The bare policy, for sampling an opponent forward.  No value, no tree."""
        out = self._forward(state, viewer)
        logits = out["policy_logits"][0].cpu().numpy()
        return _masked_softmax(logits, mc.legal_mask(state))


def _masked_softmax(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    masked = np.where(legal, logits, -np.inf)
    shifted = masked - masked.max()
    exp = np.exp(shifted, where=legal, out=np.zeros_like(shifted))
    total = exp.sum()
    if total <= 0.0:  # every legal logit underflowed
        out = legal.astype(np.float32)
        return out / max(out.sum(), 1.0)
    return (exp / total).astype(np.float32)


#: An opponent model: given a state and the seat to move, apply one move.
OpponentPolicy = Callable[[GameState, int, random.Random], None]


def sampling_opponent(evaluator: NetEvaluator) -> OpponentPolicy:
    """Sample one move from the network's own policy — clause 2, no nested search."""

    def move(state: GameState, seat: int, rng: random.Random) -> None:
        probs = evaluator.policy(state, seat)
        index = rng.choices(range(len(probs)), weights=probs, k=1)[0]
        mc.apply_macro(state, index)

    return move


# ──────────────────────────────────────────────────────────────────────────
# Re-rooting: proving a retained subtree belongs to the state being searched
# ──────────────────────────────────────────────────────────────────────────
def _sheet_key(sheet) -> tuple:
    """Every mutable field of a player sheet, as something hashable."""
    return (
        tuple(tuple(row) for row in sheet.numbers),
        tuple(tuple(row) for row in sheet.is_bis),
        tuple(tuple(row) for row in sheet.written_turn),
        tuple(tuple(row) for row in sheet.fences),
        tuple(tuple(row) for row in sheet.top_fences),
        tuple(sheet.parks),
        tuple(sheet.pools),
        tuple(sheet.estate_marks),
        sheet.temps,
        sheet.bis_marks,
        sheet.permits,
        sheet.roundabouts,
    )


def _position_key(state: GameState, root: int) -> tuple:
    """A full identity for the position a retained subtree was gathered at.

    ⚠ **Not** the viewer information-state key of ``SEARCH_SPEC.md`` §12.1, and
    it must not be mistaken for one or grow into one.  That key has to be
    *viewer-relative* because it labels chance children built from
    determinizations; this one only ever compares two of the caller's **own real
    game states**, so reading hidden fields is not just harmless, it is what
    makes the check total: a key that hid something could match two positions
    that differ, which is the one failure mode re-rooting must not have.

    Deck *order* is excluded on purpose — ``deck_pos`` and the discard pile pin
    down how much has been drawn, and the unrevealed order is re-determinized per
    simulation anyway, so requiring it to match would refuse every legitimate
    re-root the moment a caller re-seeded.
    """
    ctx = state.ctx
    return (
        root,
        state.turn,
        state.actor,
        int(state.phase),
        state.deck_pos,
        len(state.discard),
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
        self.opponent = opponent or sampling_opponent(evaluator)
        self._retained: Optional[_Retained] = None
        #: Counters, for the measurements SEARCH_SPEC §4 and §12 record.
        #: ``simulations_reused`` is the work re-rooting saved: visits a root
        #: already had when :meth:`search` was entered.
        self.simulations_run = 0
        self.simulations_reused = 0
        self.reroots = 0

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
    ) -> Observation:
        """Sample every other seat forward until ``root`` is to act, or the end.

        Clause 2.  An opponent's turn is a transition, not a node, so nothing
        here is stored and nothing here is searched.
        """
        guard = 0
        while not state.is_terminal and state.actor != root:
            self.opponent(state, state.actor, rng)
            guard += 1
            if guard > 5000:  # pragma: no cover - a stuck engine, not a rules case
                raise RuntimeError("opponents did not yield the turn")
        # ⚠ UNDER-SPECIFIED, deliberately recorded rather than quietly fixed.
        # This is raw card **IDs**, and it carries neither the opponents' now
        # public sheets nor the race state.  It costs nothing while reveals are
        # near-unique -- measured, 0 spurious splits in 60 samples -- because
        # nothing is ever reused.  It becomes wrong the moment chance children
        # are deliberately retained, which is what SELF_PLAY_PLAN.md's sparse
        # chance design does: 15 of the 66 printed card types have two physical
        # copies, so identical-looking reveals would key to different children.
        return tuple(state.table_cards(root))

    def _leaf(self, state: GameState, root: int) -> tuple[Optional[Node], float]:
        if state.is_terminal:
            return None, terminal_value(state, root, self.config)
        priors, value = self.evaluator.evaluate(state, root)
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
    ) -> Observation:
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
            observation = self._advance(state, root, rng)
        return observation

    def _simulate(self, state: GameState, node: Node, root: int, rng: random.Random) -> float:
        path: list[tuple[Node, int]] = []
        while True:
            index = node.select(self.config.c_puct)
            path.append((node, index))
            turn = state.turn
            mc.apply_macro(state, int(node.actions[index]))
            observation = self._advance(state, root, rng)
            observation = self._collapse_forced(state, root, rng, turn, observation)

            key = (int(node.actions[index]), observation)
            child = node.children.get(key)
            if child is None:
                child, value = self._leaf(state, root)
                if child is not None:
                    node.children[key] = child
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
        root = state.actor if root is None else root
        rng = rng or random.Random()
        if state.actor != root:
            raise ValueError("search starts at a state the root player is to act in")

        if node is None:
            node, _ = self._leaf(state, root)
            if node is None:
                raise ValueError("cannot search a finished game")
        self._apply_root_noise(node, rng)

        reused = int(node.visits.sum())
        budget = max(self.config.simulations - reused, 0)
        self.simulations_reused += reused
        self.simulations_run += budget
        for _ in range(budget):
            self._simulate(state.redeterminize(rng), node, root, rng)
        return node.actions, node.visits, node

    def _apply_root_noise(self, node: Node, rng: random.Random) -> None:
        if node.noised:  # a retained subtree that was noised as a root already
            return
        if self.config.dirichlet_alpha is None or len(node.actions) < 2:
            return
        noise = np.random.default_rng(rng.randrange(1 << 32)).dirichlet(
            [self.config.dirichlet_alpha] * len(node.actions)
        )
        weight = self.config.dirichlet_weight
        node.prior = (1.0 - weight) * node.prior + weight * noise
        node.noised = True

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
        child = node.children.get((choice, tuple(successor.table_cards(root))))
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
