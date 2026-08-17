"""Dual-mode MCTS with a Gumbel root (plan §5, Phase C Python reference).

Structure mirrors the proven Kingdomino `mcts_az.py` dual-tree design
(closed `AlphaZeroMCTS` + `OpenLoopMCTS`), rewritten for what 7WD needs that
Kingdomino's tree does not have:

- **Gumbel root**: top-k by Gumbel + log-prior, sequential halving over the
  sims budget, completed-Q improved policy target (the §2 lever vs ZeusAI).
- **Explicit chance layer** (closed mode): the searcher PREDICTS each action's
  chance events from public information (`chance_signature`), samples outcomes
  from `UnseenPool` enumerations, and steps barred clones with explicit
  `chance_outcomes` — the locked deal is never read (HiddenInformationError
  otherwise). Chance edges use exact probability-weighted expectation over
  expanded children (Star-style), so a fully expanded tree reproduces
  expectimax to float precision (`closed_root_exact_value`, the §5 gate).
- **Open mode**: nodes keyed by action path; each descent re-determinizes the
  root clone via `resample_hidden` and walks it in simulator mode; legality is
  re-masked per world; priors cached at first expansion (the known weakness
  the Phase E A/B measures).
- **Per-node actors**: extra turns and pending choices break strict
  alternation, so values are stored player-0-relative and converted per node.

The actor sequence along a path is deterministic given the actions (reveals
change identities, never who acts next), which is what makes path-keyed open
nodes and closed-tree actors well-defined.

Known divergence from the plan's architecture, kept deliberately: multi-event
chance CHAINS are stored flattened per action edge (children keyed by the full
outcome tuple with correct sequential-conditional probabilities) rather than
as nested per-event chance nodes. Sampling and expectation are semantically
identical; what is lost is prefix sharing across sibling outcomes, which
matters for Star pruning — the Rust searcher (Phase F) implements first-class
sequential chance layers per the shared-crate design, and this reference
stays the simpler equivalent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .codec import decode_action, legal_action_indices
from .data import (
    CARD_IDS,
    TABLEAU_LAYOUTS,
    BackType,
    back_type_of,
    covering_slots,
)
from .encoder import encode
from .engine import Action, ActionUse, apply_action
from .game import ChanceKind, GameState, Phase
from .inference import Evaluator
from .portable_rng import PortableRng
from .pool import (
    enumerate_card_reveal,
    enumerate_great_library,
    enumerate_wonder_flip,
    unseen_pool,
)

AGE_BACKS = {1: BackType.AGE_I, 2: BackType.AGE_II, 3: BackType.AGE_III}


# --------------------------------------------------------------------------
# Chance signature: public prediction of the events an action will fire.
# Gated against actual engine StepResult events in test_search.py.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChanceSpec:
    kind: ChanceKind
    context: tuple = ()


def _newly_accessible_after_take(observation, taken) -> list[tuple]:
    """(slot_id, back) of face-down cards a take would expose. Consumes ONLY
    the observation: coverage topology from the printed layout and
    `PublicTableauCard.back` — this function is the anti-leak foundation and
    never touches hidden identities."""

    layout = {(slot.row, slot.x): slot for slot in TABLEAU_LAYOUTS[observation.age]}
    cards = {card.slot_id: card for card in observation.tableau}
    present = {
        slot_id for slot_id, card in cards.items() if card.present and slot_id != taken
    }
    exposed = []
    for slot_id, card in cards.items():
        if slot_id == taken or not card.present or card.revealed:
            continue
        coverers = covering_slots(
            tuple(layout.values()), layout[slot_id]
        )
        if not any((c.row, c.x) in present for c in coverers):
            exposed.append((slot_id, card.back))
    return sorted(exposed)


def _exhausts_the_age(state: GameState, action: Action) -> bool:
    """Does this action empty the tableau, so the next Age is dealt?

    Decided by applying the action to a throwaway clone rather than by
    re-deriving the rules here: whether the last take actually ends the Age
    depends on victories and deferred choices that only the engine knows, and
    duplicating that in the searcher would be a second copy to keep in sync.
    A cheap public precondition gates the clone, so it happens only on the one
    take per Age that can empty the pyramid.

    The result is still a function of public information — the last card is
    face-up, and so is everything that decides whether taking it ends the
    game — which is what the leak-free contract requires. The clone is
    unbarred so its own chance draws resolve from the locked deal; nothing is
    read back from it but the phase.
    """

    if state.age >= 3:
        return False
    present = sum(1 for card in state.tableau.cards.values() if card.present)
    if action.use is ActionUse.RESOLVE_PENDING_CHOICE:
        if present:
            return False
    elif present != 1:
        return False
    clone = state.clone()
    clone.search_barrier = False
    apply_action(clone, action)
    return clone.phase is Phase.CHOOSE_NEXT_START_PLAYER


def chance_signature(state: GameState, action: Action) -> tuple[ChanceSpec, ...]:
    """Predict the chance events an action fires, from public information only
    (implemented against the observation; gated exactly vs engine events)."""

    observation = state.observation(state_actor(state))
    if action.use is ActionUse.DRAFT_WONDER:
        picked = sum(len(city.wonders) for city in observation.cities)
        specs = []
        if picked == 3:
            specs.append(ChanceSpec(ChanceKind.WONDER_GROUP_REVEAL))
        if picked == 7:
            specs.append(ChanceSpec(ChanceKind.AGE_DEAL, (1,)))
        return tuple(specs)
    if action.use is ActionUse.CHOOSE_NEXT_START_PLAYER:
        # The Age was dealt when the previous one ran out; the choice itself
        # fires nothing.
        return ()
    if action.use is ActionUse.RESOLVE_PENDING_CHOICE:
        if _exhausts_the_age(state, action):
            return (ChanceSpec(ChanceKind.AGE_DEAL, (observation.age + 1,)),)
        return ()
    specs = [
        ChanceSpec(ChanceKind.CARD_REVEAL, (slot_id, back))
        for slot_id, back in _newly_accessible_after_take(observation, action.slot_id)
    ]
    if action.use is ActionUse.CONSTRUCT_WONDER and action.wonder_name == "The Great Library":
        offboard = unseen_pool(observation).offboard_progress
        if offboard:
            specs.append(ChanceSpec(ChanceKind.GREAT_LIBRARY_DRAW))
    if _exhausts_the_age(state, action):
        specs.append(ChanceSpec(ChanceKind.AGE_DEAL, (observation.age + 1,)))
    return tuple(specs)


def age_deal_key(age: int, deal) -> tuple:
    """Observable signature of an Age deal (spec §4.2): face-up identities plus
    the public back pattern, in layout order. Two hidden arrangements with the
    same signature are the same chance child."""

    layout = TABLEAU_LAYOUTS[age]
    return tuple(
        name if slot.face_up else back_type_of(name)
        for slot, name in zip(layout, deal, strict=True)
    )


def sample_outcomes(
    state: GameState, specs, rng
) -> tuple[list, float | None, tuple]:
    """Sample one outcome per spec. Returns (outcomes, joint probability or
    None when any event is sample-only, hashable child key). Sequential
    CARD_REVEALs condition on earlier outcomes (same-back pools shrink)."""

    pool = unseen_pool(state.observation(state.active_player))
    used: set[str] = set()
    outcomes: list = []
    probability: float | None = 1.0
    for spec in specs:
        if spec.kind is ChanceKind.CARD_REVEAL:
            back = spec.context[1]
            names = [
                name
                for name, _ in enumerate_card_reveal(pool, back)
                if name not in used
            ]
            choice = names[rng.randrange(len(names))]
            used.add(choice)
            outcomes.append(choice)
            if probability is not None:
                probability *= 1.0 / len(names)
        elif spec.kind is ChanceKind.GREAT_LIBRARY_DRAW:
            subsets = enumerate_great_library(pool)
            subset, p = subsets[rng.randrange(len(subsets))]
            outcomes.append(subset)
            if probability is not None:
                probability *= p
        elif spec.kind is ChanceKind.WONDER_GROUP_REVEAL:
            flips = enumerate_wonder_flip(pool)
            subset, p = flips[rng.randrange(len(flips))]
            outcomes.append(subset)
            if probability is not None:
                probability *= p
        elif spec.kind is ChanceKind.AGE_DEAL:
            age = spec.context[0]
            names = sorted(pool.cards[AGE_BACKS[age]])
            rng.shuffle(names)
            if age == 3:
                guilds = sorted(pool.cards[BackType.GUILD])
                rng.shuffle(guilds)
                deal = names[:17] + guilds[:3]
                rng.shuffle(deal)
            else:
                deal = names[: len(TABLEAU_LAYOUTS[age])]
            outcomes.append(tuple(deal))
            probability = None  # sample-only event (spec §4.2)
        else:  # pragma: no cover
            raise AssertionError(spec.kind)
    key_parts = []
    for spec, outcome in zip(specs, outcomes):
        if spec.kind is ChanceKind.AGE_DEAL:
            # Children keyed by OBSERVABLE signature (spec §4.2): equivalent
            # hidden arrangements coalesce into one node.
            key_parts.append(age_deal_key(spec.context[0], outcome))
        else:
            key_parts.append(outcome if isinstance(outcome, (str, tuple)) else tuple(outcome))
    return outcomes, probability, tuple(key_parts)


# --------------------------------------------------------------------------
# Tree structures (values stored player-0-relative; converted per node actor)
# --------------------------------------------------------------------------


def fixed_support_index(weights, target: float) -> int:
    """Index of the child a fixed-support draw lands on: the first whose
    cumulative weight exceeds `target`.

    The last child absorbs the float residue — with mass 1 by construction the
    residue is ~1e-16, and a draw past the final cumulative sum must still
    resolve to an existing child rather than fall through. Rust's
    `tree::fixed_support_index` mirrors this exactly (same fold order, same
    strict `<`), so the two languages pick the same child from the same draw."""

    weights = list(weights)
    if not weights:
        raise ValueError("fixed-support edge has no children to sample")
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if target < cumulative:
            return index
    return len(weights) - 1


def state_actor(state: GameState) -> int:
    return (
        state.pending_choice.player
        if state.pending_choice is not None
        else state.active_player
    )


def _terminal_value_p0(state: GameState) -> float:
    if state.winner is None:
        return 0.0
    return 1.0 if state.winner == 0 else -1.0


@dataclass(slots=True)
class _Child:
    probability: float | None  # None for sample-only chance
    node: "ClosedNode"
    samples: int = 0  # descent count (Monte Carlo weight for sample-only)


@dataclass(slots=True)
class _Edge:
    action_index: int
    prior: float
    specs: tuple
    children: dict = field(default_factory=dict)  # key -> _Child
    visits: int = 0
    value_sum_p0: float = 0.0  # visit-weighted running mean (selection Q)
    probability_weighted: bool = False
    """Use the current exact chance expectation for Q. Enabled only after
    every enumerable child has been materialized (forced root expansion)."""
    fixed_support: bool = False
    """The children are an APPROXIMATE, re-normalised subset of the outcome
    space, and the edge is closed against growth.

    Three edge classes must stay distinct (CHANCE_ENUMERATION_PLAN.md):

    | class                   | support               | later descent           |
    |-------------------------|-----------------------|-------------------------|
    | `probability_weighted`  | exhaustive, exact     | always finds a child    |
    | `fixed_support`         | retained subset       | samples only among them |
    | ordinary sampled        | grows lazily          | may materialize a child |

    An approximate edge MUST be closed: ordinary descent samples from the
    COMPLETE chance distribution and appends any outcome it cannot find,
    carrying that outcome's original probability. On a truncated edge that
    would push the mass above 1 and `q_p0` would then return a weighted sum
    over more than unit mass with nothing raising. Closing the support is
    what keeps the invariant true for the life of the tree."""

    def close_fixed_support(self) -> None:
        """Mark an approximate edge closed. The retained children must already
        carry re-normalised weights summing to 1 — that is what makes their
        probability-weighted Q a proper (stratified) expectation."""

        mass = sum(child.probability or 0.0 for child in self.children.values())
        if not self.children or abs(mass - 1.0) > 1e-9:
            raise ValueError(
                f"fixed-support edge holds re-normalised mass {mass:.6f} != 1"
            )
        self.probability_weighted = True
        self.fixed_support = True

    @property
    def q_p0(self) -> float:
        if self.probability_weighted:
            mass = sum(child.probability or 0.0 for child in self.children.values())
            if abs(mass - 1.0) > 1e-9:
                raise ValueError(
                    f"probability-weighted edge holds mass {mass:.6f} != 1"
                )
            return sum(
                child.probability * child.node.value_p0
                for child in self.children.values()
            )
        return self.value_sum_p0 / self.visits if self.visits else 0.0


@dataclass(slots=True)
class ClosedNode:
    state: GameState  # barred clone
    actor: int
    terminal: bool
    edges: list = field(default_factory=list)  # [_Edge] aligned to legal
    legal: tuple = ()
    visits: int = 0
    value_sum_p0: float = 0.0

    @property
    def value_p0(self) -> float:
        return self.value_sum_p0 / self.visits if self.visits else 0.0


@dataclass(slots=True)
class OpenNode:
    actor: int | None = None
    priors: dict | None = None  # action_index -> prior, cached at 1st expansion
    children: dict = field(default_factory=dict)  # action_index -> OpenNode
    visits: int = 0
    value_sum_p0: float = 0.0
    edge_visits: dict = field(default_factory=dict)  # action_index -> int
    edge_value_p0: dict = field(default_factory=dict)

    @property
    def value_p0(self) -> float:
        return self.value_sum_p0 / self.visits if self.visits else 0.0


def prune_policy_target(
    visits: list[int],
    priors: list[float],
    q: list[float],
    c_puct: float,
    k: float,
) -> "list[float] | None":
    """KataGo policy-target pruning (§3.2) -- the other half of forced playouts.

    Forced playouts put visits on children PUCT would not have chosen. Those
    belong in the search and in the trajectory, and emphatically not in the
    label: a target built from raw visit counts teaches the forcing.

    For every child but the most-visited, subtract up to ``sqrt(k * P(c) * N)``
    visits, stopping as soon as one more removal would raise that child's PUCT
    score above the best child's. That guard makes the rule self-calibrating --
    a child PUCT genuinely chose already sits at parity, so nothing is taken from
    it -- and it is the part a flat ``sqrt(kPN)`` subtraction gets wrong.

    ``priors`` are the NOISED priors the search descended under. The clean prior
    asks a different question ("would the net have chosen this?") and would take
    back visits the search made on its own terms.

    Returns ``None`` when there is nothing to do, so callers fall back to the raw
    distribution rather than record a fabricated one. A free function, not a
    method, so ``tree::prune_policy_target`` can be gated directly against it.
    """

    total = sum(visits)
    if k <= 0.0 or total <= 0:
        return None
    best = max(range(len(visits)), key=lambda j: visits[j])
    root_sqrt = math.sqrt(total)

    def puct(j: int, kept_j: float) -> float:
        return q[j] + c_puct * priors[j] * root_sqrt / (1.0 + kept_j)

    best_puct = puct(best, visits[best])
    kept = [float(v) for v in visits]
    for j, count in enumerate(visits):
        if j == best or count == 0:
            continue
        forced = math.sqrt(k * priors[j] * total)
        removed = 0.0
        while removed < forced and kept[j] > 0 and puct(j, kept[j] - 1) <= best_puct:
            kept[j] -= 1
            removed += 1
        if kept[j] <= 1:
            kept[j] = 0.0
    mass = sum(kept)
    if mass <= 0:
        return None  # a zero label is worse than an unpruned one
    return [v / mass for v in kept]


@dataclass(frozen=True, slots=True)
class SearchResult:
    action_index: int
    action_value: float  # selected-action Q, root-actor perspective
    root_value: float  # root-actor perspective
    visits: dict  # action_index -> visit count
    policy_target: dict  # action_index -> improved (completed-Q) probability
    gumbel_topk: tuple  # the initial Gumbel top-k candidate set (buffer field)
    sims: int  # always <= config.sims
    mode: str
    # action_index -> completed Q (root-actor perspective).  `action_value` is
    # only the Gumbel-selected action's entry; a caller that plays a different
    # action (evaluation plays argmax(policy_target)) needs that action's Q.
    completed_q: dict
    training_policy: dict | None = None
    """The distribution to RECORD, when it differs from `policy_target`.

    Under PUCT with policy-target pruning they differ by design: the move is
    played from the raw visit distribution, keeping forced exploration in the
    trajectory, while the label has those forced visits taken back out. `None`
    means the two are the same, which is every Gumbel search and every PUCT
    search with pruning off.
    """


@dataclass(slots=True)
class SearchConfig:
    sims: int = 64
    top_k: int = 16
    mode: str = "closed"  # or "open"
    root_selection: str = "gumbel"  # or "puct"
    """How the ROOT picks which action to simulate next.

    ``gumbel`` is Gumbel top-k + sequential halving: the training-target
    generator, and what self-play must keep using.  ``puct`` selects at the
    root by PUCT like every other node and plays argmax visits -- the search
    the advisor runs, and therefore what evaluation should measure.
    """

    c_puct: float = 1.5
    c_visit: float = 50.0
    # mctx's `value_scale`; pairs with the [0, 1] min-max rescale in `_sigma`.
    # Was 1.0 against a raw q in [-1, 1], which made sigma swamp the prior.
    c_scale: float = 0.1

    forced_playout_k: float = 0.0
    """KataGo forced playouts at the PUCT root (paper §3.2). Zero is off.

    Dirichlet noise raises a child's prior, but PUCT can still decline to spend
    a single simulation there, so the noise changes nothing and the policy never
    learns whether the move was good.  Forcing at least
    ``sqrt(k * P(c) * N)`` visits guarantees the look.  KataGo uses ``k = 2``.

    Pairs with policy-target pruning, which removes these visits from the
    LABEL again: the forced look belongs in the search and in the trajectory,
    never in the target.  Enabling this without pruning teaches the forcing.

    Applies at the root only, and only under ``root_selection="puct"`` -- the
    Gumbel root already guarantees every candidate a share of the budget.
    """

    dirichlet_epsilon: float = 0.0
    """Root exploration noise, applied ONLY under ``root_selection="puct"``.

    Zero (the default) is off. The Gumbel root carries its own exploration in
    the Gumbel keys, so noise there would double up; PUCT has no such source and
    without it self-play collapses toward deterministic lines. Kingdomino's
    settled values are the reference: 0.25 for the learner's full-search moves,
    0.0 for evaluation, 0.0 for fast moves, 0.0 for archived HOF opponents.

    Noise perturbs which actions get SEARCHED. It never reaches the recorded
    policy target, which ``_puct_root`` builds from visit counts, nor the
    recorded ``prior``, which is snapshotted before the blend.
    """

    dirichlet_alpha: float = 1.8
    """Concentration. The AlphaZero convention is ``alpha ~ 10 / branching``:
    0.3 for chess (~35 moves), 0.15 for shogi (~92), 0.03 for Go (~250).

    7WD measures a MEDIAN of 4 legal actions and a mean of 5.6, which puts it
    near 1.8 -- six times Kingdomino's 0.3. Copying 0.3 here would be a real
    error, not a nitpick: at alpha well below 1 over ~5 actions the draw is
    extremely peaked, dumping nearly all the noise mass on one arbitrary action
    instead of spreading a mild perturbation across the root.
    """

    seed: int = 0
    force_expand_root_chance: bool = False
    """Closed mode: exhaustively materialize and evaluate every enumerable
    chance child of the root's edges before searching (plan §5 catastrophe
    coverage; AGE_DEAL edges stay sampled)."""
    round_robin_candidates: bool = False
    """Interleave each sequential-halving round instead of blocking it.

    Sequential halving fixes only *how many* simulations each surviving
    candidate receives in a round, not the order they are issued in, so
    ``c0, c1, .., ck, c0, c1, ..`` is as faithful as ``c0 x per_action,
    c1 x per_action, ..``. The blocked order below is the original choice and
    the one every recorded gate is anchored to.

    It matters for throughput because the Rust searcher batches leaves across
    root candidates: under the blocked order the next simulation repeats the
    current candidate whenever ``per_action >= 2``, which holds realized leaf
    wave width at 1.19 (THROUGHPUT_ACTION_PLAN.md, Phase 2). Interleaving lets a
    wave hold one simulation per candidate.

    This changes which leaves a round visits and therefore every search output.
    It is a different, equally valid sample -- not a refactor."""
    double_reveal_offsets: int = 0
    """Offsets per first-reveal stratum on a PURE double card-reveal root edge.

    Zero (the default) keeps forced expansion exhaustive. A positive `X` keeps
    the balanced `n * X` support instead of all `n * (n - 1)` directed pairs —
    every hidden card leads exactly one stratum, so it appears exactly `X` times
    in the first revealed slot and exactly `X` times in the second, weighted
    `1 / (n * X)`, closed against growth. Those edges are 54.5% of all forced children
    (CHANCE_ENUMERATION_PLAN.md). Everything else stays exhaustive; the cap is
    ignored once `X` would retain the full space anyway."""


class GumbelMCTS:
    """One search instance per move. `search(state)` returns the chosen action
    plus the improved policy target (buffer §6 fields)."""

    def __init__(self, evaluator: Evaluator, config: SearchConfig | None = None):
        self.evaluator = evaluator
        self.config = config or SearchConfig()
        self.rng = PortableRng(self.config.seed)
        self.closed_nodes_created = 0

    # ---- shared -----------------------------------------------------------

    def _evaluate(self, state: GameState) -> tuple[float, dict]:
        """(value_p0, priors dict over legal indices) from the net."""

        if state.phase is Phase.COMPLETE:
            return _terminal_value_p0(state), {}
        actor = state_actor(state)
        legal = legal_action_indices(state)
        evaluation = self.evaluator.evaluate(
            [encode(state.observation(actor))], [legal]
        )[0]
        value_actor = float(evaluation.wdl[0] - evaluation.wdl[2])
        value_p0 = value_actor if actor == 0 else -value_actor
        priors = {index: float(p) for index, p in zip(legal, evaluation.policy)}
        return value_p0, priors

    def _sigma(self, completed: dict, max_visits: int) -> dict:
        """Gumbel-AlphaZero sigma over MIN-MAX NORMALISED completed Q values.

        The (c_visit + max_visits) * c_scale * q form is from the paper, but the
        paper (and mctx's ``qtransform_completed_by_mix_value``) applies it to a
        Q rescaled to [0, 1] across the root's actions.  Applied to a raw
        actor-relative q in [-1, 1] with c_scale=1.0, sigma spanned +/-50 while
        log-prior differences are ~1-3, so the prior contributed essentially
        nothing to the improved policy and the search ignored the network's
        move preferences.

        Rescaling per call (rather than once) is required: completed Q changes
        as simulations accumulate, so the normalisation window has to track it.
        When every action shares a value -- no search information yet -- the
        span guard collapses sigma to zero and the improved policy is exactly
        the prior, which is the correct degenerate case.
        """

        values = list(completed.values())
        low = min(values)
        span = max(max(values) - low, 1e-8)
        scale = (self.config.c_visit + max_visits) * self.config.c_scale
        return {a: scale * (q - low) / span for a, q in completed.items()}

    def search(self, state: GameState) -> SearchResult:
        if self.config.mode == "closed":
            return self._search_closed(state)
        if self.config.mode == "open":
            return self._search_open(state)
        raise ValueError(f"unknown mode: {self.config.mode}")

    # ---- incremental (advisor) API ---------------------------------------
    # `search()` runs the whole Gumbel sequential-halving budget in one shot,
    # whose schedule is coupled to the total sim count -- so it cannot be
    # resumed chunk-by-chunk. The advisor host wants the opposite: a tree it
    # can crank one sim at a time and read between cranks (monotone refinement,
    # streaming display, cancel-any-time). The Gumbel root is a training-target
    # device; plain PUCT descent underneath it is naturally incremental, gives
    # the visits+Q a recommendation needs, and is exposed here.

    def make_root(self, state: GameState) -> ClosedNode:
        """Build and expand a closed-mode root for incremental search.

        Same root setup as ``_search_closed`` -- barrier clone, expand, seed
        the root visit, optional forced chance expansion -- but runs NO
        simulations. Callers drive :meth:`descend` to add sims one at a time.
        """

        if self.config.mode != "closed":
            raise ValueError("make_root/descend support closed mode only")
        root_state = state.clone()
        root_state.search_barrier = True
        root = self._make_closed_node(root_state)
        root_value_p0 = self._expand_closed(root)
        root.visits += 1
        root.value_sum_p0 += root_value_p0
        if self.config.force_expand_root_chance and not root.terminal:
            self._force_expand_root(root)
        return root

    def descend(self, root: ClosedNode) -> None:
        """Run one PUCT simulation from an existing root. Monotone: successive
        calls deepen the same tree rather than restart it."""

        self._descend_closed(root, None)

    def _gumbel_root(
        self, legal, priors, simulate, root_value, root_actor, initial_q=None
    ):
        """Gumbel top-k + sequential halving. `simulate(action_index)` runs one
        descent through that root action and returns its running Q (root-actor
        perspective) and visit count."""

        config = self.config
        if config.sims < 1 or config.top_k < 1:
            raise ValueError("sims and top_k must be positive")
        log_prior = {a: math.log(max(priors.get(a, 1e-12), 1e-12)) for a in legal}
        # Portable Gumbel keys (one per legal action, sorted order) so the Rust
        # searcher reproduces the top-k and halving bit-for-bit — see portable_rng.
        gumbel = {a: self.rng.gumbel() for a in legal}
        candidates = sorted(
            legal, key=lambda a: gumbel[a] + log_prior[a], reverse=True
        )[: min(config.top_k, len(legal))]
        topk = tuple(candidates)

        budget = config.sims
        sims_used = 0
        rounds_total = max(1, math.ceil(math.log2(max(len(candidates), 2))))
        round_index = 0
        q_hat: dict = {}
        visits: dict = {a: 0 for a in legal}
        initial_q = initial_q or {}

        def completed_q(action):
            return q_hat.get(action, initial_q.get(action, root_value))

        while sims_used < budget:
            rounds_remaining = max(1, rounds_total - round_index)
            per_action = max(
                1, (budget - sims_used) // (rounds_remaining * len(candidates))
            )
            # Blocked (default) issues `per_action` consecutive simulations per
            # candidate; interleaved cycles the candidate list `per_action`
            # times. Same allocation, different order -- see
            # `SearchConfig.round_robin_candidates`.
            schedule = (
                [action for _ in range(per_action) for action in candidates]
                if self.config.round_robin_candidates
                else [action for action in candidates for _ in range(per_action)]
            )
            for action in schedule:
                if sims_used >= budget:
                    break
                q, n = simulate(action)
                q_hat[action] = q
                visits[action] = n
                sims_used += 1
            if len(candidates) > 1:
                max_visits = max(visits.values()) if visits else 0
                sigma = self._sigma({a: completed_q(a) for a in legal}, max_visits)
                candidates = sorted(
                    candidates,
                    key=lambda a: gumbel[a] + log_prior[a] + sigma[a],
                    reverse=True,
                )[: max(1, len(candidates) // 2)]
            round_index += 1

        max_visits = max(visits.values()) if visits else 0
        # Improved policy over ALL legal actions: completed Q (an exact forced
        # chance expectation when available, otherwise the root value for an
        # unvisited action) — the Gumbel policy target.  Sigma normalises over
        # this same full-legal window, so the halving key, the played action and
        # the target all share one scale.
        completed = {a: completed_q(a) for a in legal}
        sigma = self._sigma(completed, max_visits)
        best = max(candidates, key=lambda a: gumbel[a] + log_prior[a] + sigma[a])
        logits = {a: log_prior[a] + sigma[a] for a in legal}
        peak = max(logits.values())
        weights = {a: math.exp(v - peak) for a, v in logits.items()}
        total = sum(weights.values())
        policy_target = {a: w / total for a, w in weights.items()}
        return best, completed[best], visits, policy_target, sims_used, topk, completed

    # ---- closed mode ------------------------------------------------------

    def _make_closed_node(self, state: GameState) -> ClosedNode:
        # Counted because every node owns a cloned GameState, so on a wide root
        # the tree costs memory linearly in simulations. The advisor reads this
        # to stop deepening; nothing else depends on it. (Rust measures its
        # arena in bytes directly -- `arena_deep_bytes` -- which is exact; this
        # side has only the count.)
        self.closed_nodes_created += 1
        terminal = state.phase is Phase.COMPLETE
        node = ClosedNode(
            state=state,
            actor=state_actor(state) if not terminal else 0,
            terminal=terminal,
        )
        if not terminal:
            node.legal = legal_action_indices(state)
        return node

    def _closed_child(self, node: ClosedNode, edge: _Edge) -> ClosedNode:
        """Descend one edge: sample the chance chain, materialize/reuse the
        child. Never touches the locked deal (barred clones + explicit
        outcomes)."""

        if edge.fixed_support:
            # Closed support: draw among the RETAINED children by their
            # re-normalised weights and never materialize a new one (an omitted
            # outcome re-entering the tree would break the mass invariant).
            children = list(edge.children.values())
            index = fixed_support_index(
                (child.probability for child in children), self.rng.next_float()
            )
            child = children[index]
            child.samples += 1
            return child.node
        if edge.specs:
            outcomes, probability, key = sample_outcomes(
                node.state, edge.specs, self.rng
            )
        else:
            outcomes, probability, key = None, 1.0, ()
        child = edge.children.get(key)
        if child is None:
            clone = node.state.clone()
            clone.search_barrier = True
            apply_action(
                clone,
                decode_action(clone, edge.action_index),
                chance_outcomes=outcomes,
            )
            child = _Child(probability=probability, node=self._make_closed_node(clone))
            edge.children[key] = child
        child.samples += 1
        return child.node

    def _expand_closed(self, node: ClosedNode) -> float:
        value_p0, priors = self._evaluate(node.state)
        if not node.terminal:
            node.edges = [
                _Edge(
                    action_index=index,
                    prior=priors.get(index, 0.0),
                    specs=chance_signature(
                        node.state, decode_action(node.state, index)
                    ),
                )
                for index in node.legal
            ]
        return value_p0

    def _select_closed(self, node: ClosedNode) -> _Edge:
        sign = 1.0 if node.actor == 0 else -1.0
        total = math.sqrt(max(1, node.visits))
        best, best_score = None, -math.inf
        for edge in node.edges:
            q = sign * edge.q_p0
            score = q + self.config.c_puct * edge.prior * total / (1 + edge.visits)
            if score > best_score:
                best, best_score = edge, score
        return best

    def _forced_playout_edge(self, root: ClosedNode) -> "_Edge | None":
        """A root child owed a forced playout, or None.

        ``N`` is the sum of CHILD visits, not ``node.visits``: the root counts
        its own expansion, and the three implementations of this rule (here,
        ``tree.rs``, ``tree_resumable.rs``) have to agree on the quantity
        exactly or the equivalence gate fails on a boundary case nobody looks
        at. Summing edges is unambiguous in all three.

        Among children below quota, take the best PUCT score, so forcing
        interleaves with the ordinary search instead of front-loading every
        forced visit. Strict ``>`` takes the first on a tie, matching
        ``_select_closed``.
        """

        k = self.config.forced_playout_k
        if k <= 0.0:
            return None
        total = sum(edge.visits for edge in root.edges)
        if total <= 0:
            return None  # nothing to scale a quota against yet
        sign = 1.0 if root.actor == 0 else -1.0
        root_sqrt = math.sqrt(max(1, root.visits))
        best, best_score = None, -math.inf
        for edge in root.edges:
            if edge.visits >= math.sqrt(k * edge.prior * total):
                continue
            q = sign * edge.q_p0
            score = q + self.config.c_puct * edge.prior * root_sqrt / (1 + edge.visits)
            if score > best_score:
                best, best_score = edge, score
        return best

    def _prune_policy_target(self, root: ClosedNode, sign: float, visits: dict) -> dict:
        """KataGo policy-target pruning (§3.2) -- the other half of forcing.

        Forced playouts put visits on children PUCT would not have chosen. Those
        visits belong in the search and in the trajectory, and emphatically not
        in the label: a target built from raw visit counts teaches the forcing.

        For every child but the most-visited, subtract up to
        ``sqrt(k * P(c) * N)`` visits, stopping as soon as one more removal
        would raise that child's PUCT score above the best child's. The guard is
        what makes this self-calibrating -- a child PUCT genuinely chose already
        sits at parity, so nothing is taken from it -- and it is what a flat
        ``sqrt(kPN)`` subtraction gets wrong.

        Mirrors ``tree::prune_policy_target``; the two are gated against each
        other. Priors here are the NOISED ones the search descended under.
        """

        edges = {edge.action_index: edge for edge in root.edges}
        order = list(visits)
        pruned = prune_policy_target(
            [visits[a] for a in order],
            [edges[a].prior for a in order],
            [sign * edges[a].q_p0 for a in order],
            self.config.c_puct,
            self.config.forced_playout_k,
        )
        return None if pruned is None else dict(zip(order, pruned))

    def _descend_closed(self, node: ClosedNode, forced_edge: _Edge | None) -> float:
        """One simulation from `node`; returns the leaf value (p0 terms)."""

        if node.terminal:
            value = _terminal_value_p0(node.state)
            node.visits += 1
            node.value_sum_p0 += value
            return value
        if not node.edges:  # unexpanded leaf
            value = self._expand_closed(node)
            node.visits += 1
            node.value_sum_p0 += value
            return value
        edge = forced_edge if forced_edge is not None else self._select_closed(node)
        child = self._closed_child(node, edge)
        value = self._descend_closed(child, None)
        edge.visits += 1
        edge.value_sum_p0 += value
        node.visits += 1
        node.value_sum_p0 += value
        return value

    def _add_dirichlet_noise(self, root: ClosedNode) -> None:
        """Blend Dirichlet noise into the root edges' priors, in place.

        ``prior <- (1 - eps) * prior + eps * noise`` over the root's edges, the
        AlphaZero form and a direct port of Kingdomino's ``_add_dirichlet_noise``
        (``mcts_az.py:729``) -- with one deliberate difference. Kingdomino draws
        from numpy, whose values Rust cannot reproduce, so its equivalence gate
        has to run at eps=0 and its production self-play is not bit-comparable
        across the two implementations. 7WD's Gumbel keys come from
        ``PortableRng``, so its self-play IS bit-comparable *with exploration on*
        today; drawing this noise from anywhere else would give that up. Hence
        ``PortableRng.dirichlet``, sharing the searcher's existing stream.

        No-op unless the PUCT root is selected and eps > 0. Applied after the
        clean prior is snapshotted, so only search behaviour changes.
        """

        config = self.config
        if config.root_selection != "puct" or config.dirichlet_epsilon <= 0.0:
            return
        if not root.edges:
            return
        epsilon = config.dirichlet_epsilon
        noise = self.rng.dirichlet(config.dirichlet_alpha, len(root.edges))
        # Standard AlphaZero blend; priors are expected normalised (production
        # normalises in `blend_priors`). Scaling by observed mass would hide a
        # malformed prior rather than surface it.
        for edge, sample in zip(root.edges, noise):
            edge.prior = (1.0 - epsilon) * edge.prior + epsilon * sample

    def _puct_root(self, root: ClosedNode, sign: float, root_value: float):
        """Plain PUCT from the root — the search the advisor actually runs.

        The Gumbel root exists to make a small, fixed simulation budget yield an
        unbiased policy-improvement *target*.  Neither premise holds in
        evaluation: the budget is not being spent to build a training target,
        and Gumbel keys are exploration noise that perturb which candidates get
        searched at all.  Competitive play wants the best move, so the root
        selects by PUCT like every node below it and plays argmax visits.

        Returns the same tuple shape as ``_gumbel_root`` so ``_search_closed``
        can treat the two interchangeably.  ``gumbel_topk`` comes back empty:
        there is no Gumbel top-k, and recording a fake one would let a buffer
        row claim a candidate set that never existed.
        """

        for _ in range(self.config.sims):
            self._descend_closed(root, forced_edge=self._forced_playout_edge(root))
        visits = {edge.action_index: edge.visits for edge in root.edges}
        completed = {
            edge.action_index: (
                sign * edge.q_p0
                if edge.visits or edge.probability_weighted
                else root_value
            )
            for edge in root.edges
        }
        total = sum(visits.values())
        if total > 0:
            policy_target = {a: v / total for a, v in visits.items()}
        else:  # every simulation hit a terminal root edge
            mass = sum(edge.prior for edge in root.edges) or 1.0
            policy_target = {e.action_index: e.prior / mass for e in root.edges}
        # `max` over the legal order takes the FIRST maximum; Rust iterates the
        # same order with a strict `>` so the two agree on ties.
        best = max(root.legal, key=lambda a: visits.get(a, 0))
        return (
            best,
            completed[best],
            visits,
            policy_target,
            self.config.sims,
            (),
            completed,
            self._prune_policy_target(root, sign, visits),
        )

    def _force_expand_root(self, root: ClosedNode) -> None:
        """Materialize + evaluate every enumerable chance child of the root's
        edges (catastrophe coverage: rare instant-loss reveals are guaranteed
        probability-weighted, never unsampled). AGE_DEAL edges stay sampled.

        With `double_reveal_offsets` set, a pure double card-reveal edge keeps
        the balanced `n * X` support instead and is CLOSED — its children are a
        re-normalised subset, so descent must never re-open it."""

        for edge in root.edges:
            if not edge.specs or any(
                spec.kind is ChanceKind.AGE_DEAL for spec in edge.specs
            ):
                continue
            if edge.fixed_support:
                # Re-entry with a different seed would append members of a
                # SECOND support and only notice at the closing mass check, by
                # which point the tree is already mutated. Refuse before that.
                raise RuntimeError(
                    "forced expansion re-entered an already-closed edge; a "
                    "fixed support cannot be extended or replaced in place"
                )
            balanced = balanced_double_reveal_chains(
                root.state,
                edge.specs,
                self.config.double_reveal_offsets,
                self.config.seed,
            )
            chains = (
                balanced
                if balanced is not None
                else enumerate_chains(root.state, edge.specs)
            )
            # Validate the support BEFORE materializing any of it: an evaluator
            # failure part-way through then cannot leave a half-built edge whose
            # mass is neither 1 nor recoverable.
            held = sum(child.probability or 0.0 for child in edge.children.values())
            incoming = sum(
                probability
                for _, probability, key in chains
                if key not in edge.children
            )
            if abs(held + incoming - 1.0) > 1e-9:
                raise RuntimeError(
                    f"forced root edge would hold probability mass "
                    f"{held + incoming:.6f} != 1"
                )
            for outcomes, probability, key in chains:
                if key in edge.children:
                    continue
                clone = root.state.clone()
                clone.search_barrier = True
                apply_action(
                    clone,
                    decode_action(clone, edge.action_index),
                    chance_outcomes=outcomes,
                )
                child_node = self._make_closed_node(clone)
                value_p0, _ = self._evaluate(clone)
                child_node.visits = 1
                child_node.value_sum_p0 = value_p0
                edge.children[key] = _Child(probability=probability, node=child_node)
            if balanced is not None:
                edge.close_fixed_support()
                continue
            mass = sum(child.probability for child in edge.children.values())
            if abs(mass - 1.0) > 1e-9:
                raise RuntimeError(
                    f"forced root edge holds probability mass {mass:.6f} != 1"
                )
            edge.probability_weighted = True

    def _search_closed(self, state: GameState) -> SearchResult:
        root_state = state.clone()
        root_state.search_barrier = True
        root = self._make_closed_node(root_state)
        root_value_p0 = self._expand_closed(root)
        root.visits += 1
        root.value_sum_p0 += root_value_p0
        if self.config.force_expand_root_chance and not root.terminal:
            self._force_expand_root(root)
        sign = 1.0 if root.actor == 0 else -1.0
        edges_by_action = {edge.action_index: edge for edge in root.edges}
        # Snapshot BEFORE any noise: this is the network's opinion, and it is
        # what `_gumbel_root` scores against and what the buffer records as
        # `prior`. Blending noise into it would corrupt every KL diagnostic and
        # silently redefine what a recorded row means.
        priors = {edge.action_index: edge.prior for edge in root.edges}
        self._add_dirichlet_noise(root)

        def simulate(action_index: int):
            edge = edges_by_action[action_index]
            self._descend_closed(root, forced_edge=edge)
            return sign * edge.q_p0, edge.visits

        forced_q = {
            edge.action_index: sign * edge.q_p0
            for edge in root.edges
            if edge.probability_weighted
        }
        training_policy = None
        if self.config.root_selection == "puct":
            (
                best,
                action_value,
                visits,
                policy_target,
                sims,
                topk,
                completed,
                training_policy,
            ) = self._puct_root(root, sign, sign * root_value_p0)
        else:
            (
                best,
                action_value,
                visits,
                policy_target,
                sims,
                topk,
                completed,
            ) = self._gumbel_root(
                root.legal,
                priors,
                simulate,
                sign * root_value_p0,
                root.actor,
                initial_q=forced_q,
            )
        self._closed_root = root  # exposed for gates/inspection
        return SearchResult(
            action_index=best,
            action_value=action_value,
            root_value=sign * root.value_p0,
            visits=visits,
            policy_target=policy_target,
            gumbel_topk=topk,
            sims=sims,
            mode="closed",
            completed_q=completed,
            training_policy=training_policy,
        )

    # ---- open mode --------------------------------------------------------

    def _descend_open(
        self, node: OpenNode, world: GameState, forced_action: int | None
    ) -> float:
        if world.phase is Phase.COMPLETE:
            value = _terminal_value_p0(world)
            node.visits += 1
            node.value_sum_p0 += value
            return value
        actor = state_actor(world)
        if node.actor is None:
            node.actor = actor
        legal = legal_action_indices(world)  # per-world masking
        if node.priors is None:
            value, priors = self._evaluate(world)
            node.priors = priors  # cached at first expansion (open-loop flaw)
            node.visits += 1
            node.value_sum_p0 += value
            return value
        if forced_action is not None:
            action = forced_action
        else:
            sign = 1.0 if actor == 0 else -1.0
            total = math.sqrt(max(1, node.visits))
            prior_sum = sum(node.priors.get(a, 0.0) for a in legal) or 1.0

            def score(a):
                q = sign * (
                    node.edge_value_p0.get(a, 0.0) / node.edge_visits[a]
                    if node.edge_visits.get(a)
                    else 0.0
                )
                prior = node.priors.get(a, 0.0) / prior_sum
                return q + self.config.c_puct * prior * total / (
                    1 + node.edge_visits.get(a, 0)
                )

            action = max(legal, key=score)
        child = node.children.get(action)
        if child is None:
            child = node.children[action] = OpenNode()
        apply_action(world, decode_action(world, action))
        value = self._descend_open(child, world, None)
        node.edge_visits[action] = node.edge_visits.get(action, 0) + 1
        node.edge_value_p0[action] = node.edge_value_p0.get(action, 0.0) + value
        node.visits += 1
        node.value_sum_p0 += value
        return value

    def _search_open(self, state: GameState) -> SearchResult:
        from .pool import resample_hidden

        root = OpenNode()
        root_value_p0, priors = self._evaluate(state)
        root.priors = priors
        root.actor = state_actor(state)
        root.visits += 1
        root.value_sum_p0 += root_value_p0
        legal = legal_action_indices(state)
        sign = 1.0 if root.actor == 0 else -1.0

        def simulate(action_index: int):
            world = state.clone()
            world.search_barrier = False
            resample_hidden(world, self.rng)
            self._descend_open(root, world, forced_action=action_index)
            n = root.edge_visits.get(action_index, 0)
            q = sign * (root.edge_value_p0.get(action_index, 0.0) / n) if n else 0.0
            return q, n

        (
            best,
            action_value,
            visits,
            policy_target,
            sims,
            topk,
            completed,
        ) = self._gumbel_root(
            legal, priors, simulate, sign * root_value_p0, root.actor
        )
        self._open_root = root
        return SearchResult(
            action_index=best,
            action_value=action_value,
            root_value=sign * root.value_p0,
            visits=visits,
            policy_target=policy_target,
            gumbel_topk=topk,
            sims=sims,
            mode="open",
            completed_q=completed,
        )


# --------------------------------------------------------------------------
# Balanced double-reveal support — an approximate, exactly-weighted SUBSET of a
# two-card-reveal edge's outcome space (CHANCE_ENUMERATION_PLAN.md, Step 2).
# --------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
# Domain separation: the offsets must be a function of the POSITION and the
# search seed, and must not consume the main search RNG stream. Drawing them
# from the shared stream is the bug class that bit the PUCT root port, where
# Gumbel keys offset every downstream chance sample.
_OFFSET_DOMAIN_TAG = 0x0FF5_E75E_ED01_7D0B

_BACK_IDS = {back: index for index, back in enumerate(BackType)}
_KIND_IDS = {kind: index for index, kind in enumerate(ChanceKind)}


def _mix64(accumulator: int, value: int) -> int:
    """One FNV-1a round over a 64-bit word (mirrored by Rust's `mix64`)."""

    return ((accumulator ^ (value & _MASK64)) * _FNV_PRIME) & _MASK64


def double_reveal_offset_seed(search_seed: int, specs, pools) -> int:
    """Domain-separated seed for one edge's offset draw.

    Built from the search seed, the chance signature, and the reveal pools (the
    part of the public state the support is defined over) — deliberately NOT
    from the action index, so two root edges sharing a chance signature draw the
    same offsets. Common random numbers: comparing two actions over a common
    support cancels the offset noise out of the comparison."""

    accumulator = _mix64(_FNV_OFFSET, _OFFSET_DOMAIN_TAG)
    accumulator = _mix64(accumulator, search_seed)
    for spec in specs:
        accumulator = _mix64(accumulator, _KIND_IDS[spec.kind])
        (row, x), back = spec.context
        for value in (row, x, _BACK_IDS[back]):
            accumulator = _mix64(accumulator, value)
    for names in pools:
        accumulator = _mix64(accumulator, len(names))
        for name in names:
            accumulator = _mix64(accumulator, CARD_IDS[name])
    return accumulator


def distinct_offsets(modulus: int, count: int, seed: int) -> list[int]:
    """`count` distinct offsets in `[0, modulus)`, uniform over subsets, from a
    partial Fisher-Yates draw on a private stream. Returned ascending so the
    support does not depend on draw order."""

    rng = PortableRng(seed)
    values = list(range(modulus))
    for k in range(count):
        j = k + rng.randrange(modulus - k)
        values[k], values[j] = values[j], values[k]
    return sorted(values[:count])


def balanced_double_reveal_chains(
    state: GameState, specs, offsets: int, search_seed: int
) -> list[tuple[list, float, tuple]] | None:
    """The balanced `n * offsets` support of a PURE double card-reveal edge, in
    the same `(outcomes, probability, key)` shape as `enumerate_chains`.

    Returns None when the construction does not apply — a different chance
    signature, two DIFFERENT backs, `offsets <= 0`, or an `offsets` large enough
    that the full space is no bigger — and the caller must then enumerate
    exhaustively.

    Construction (`n` unseen cards of the shared back, `X` offsets):

    * **Stratify on the first reveal.** Every hidden card is the first reveal in
      exactly one stratum. That is the marginal-coverage guarantee, and it is
      what makes this strictly better than IID sampling at equal budget.
    * **Second reveal by cyclic block.** Stratum `i` takes seconds
      `first[(i + 1 + t) % n]` for `X` offsets `t` in `[0, n - 1)`, i.e.
      directed-pair distances `1..n-1`. Each fixed distance is a bijection on
      the pool, so every card appears exactly `X` times in second position, and
      a self-pair is unreachable by construction (distance 0 is excluded).
    * **Weight `1 / (n * X)`** — mass exactly 1, so the retained children carry
      a proper stratified expectation rather than a conditional one.

    **Different backs stay exhaustive by choice, not by oversight.** Those pools
    are disjoint (back types partition the card universe), so the outcome space
    is the full `n1 * n2` grid and a cyclic support over the second pool would
    still be unbiased — but only its FIRST margin could be exact, since a subset
    balanced in both margins needs a size divisible by `lcm(n1, n2)`, and those
    pool sizes are usually coprime (the grid itself). It would also make the
    retained count depend on which slot happens to be listed first, which is
    board position, not anything meaningful. Measured on the corpus those edges
    are 5.3% of double reveals, 3.2% of their outcomes and **2.9% of the cap's
    saving**, while carrying 3-4x the Q error — not a trade worth a second
    construction with a weaker guarantee.

    The coverage claim is **marginal**: every hidden card in every revealed
    slot. It is not coverage of every dangerous PAIR interaction."""

    if offsets <= 0 or len(specs) != 2:
        return None
    if any(spec.kind is not ChanceKind.CARD_REVEAL for spec in specs):
        return None
    if specs[0].context[1] != specs[1].context[1]:
        return None
    pool = unseen_pool(state.observation(state.active_player))
    names = [name for name, _ in enumerate_card_reveal(pool, specs[0].context[1])]
    n = len(names)
    modulus = n - 1  # directed-pair distances 1..n-1, never a self-pair
    if offsets >= modulus:
        return None  # the balanced support would be the whole space (or larger)
    chosen = distinct_offsets(
        modulus, offsets, double_reveal_offset_seed(search_seed, specs, (names, names))
    )
    weight = 1.0 / (n * offsets)
    chains = []
    for i, name in enumerate(names):
        for offset in chosen:
            outcomes = [name, names[(i + 1 + offset) % n]]
            chains.append((outcomes, weight, tuple(outcomes)))
    return chains


def enumerate_chains(state: GameState, specs) -> list[tuple[list, float, tuple]]:
    """All (outcomes, joint probability, key) chains for enumerable specs —
    sequential CARD_REVEALs condition later pools on earlier outcomes. Used by
    exhaustive expansion (gates) and root force-expansion. AGE_DEAL is
    sample-only and unsupported here by design."""

    pool = unseen_pool(state.observation(state.active_player))

    def expand(index: int, used: frozenset):
        if index == len(specs):
            return [([], 1.0)]
        spec = specs[index]
        results = []
        if spec.kind is ChanceKind.CARD_REVEAL:
            back = spec.context[1]
            names = [
                name for name, _ in enumerate_card_reveal(pool, back) if name not in used
            ]
            for name in names:
                for tail, p in expand(index + 1, used | {name}):
                    results.append(([name, *tail], p / len(names)))
        elif spec.kind is ChanceKind.GREAT_LIBRARY_DRAW:
            for subset, p in enumerate_great_library(pool):
                for tail, tail_p in expand(index + 1, used):
                    results.append(([subset, *tail], p * tail_p))
        elif spec.kind is ChanceKind.WONDER_GROUP_REVEAL:
            for subset, p in enumerate_wonder_flip(pool):
                for tail, tail_p in expand(index + 1, used):
                    results.append(([subset, *tail], p * tail_p))
        else:
            raise ValueError(f"cannot enumerate {spec.kind}")
        return results

    chains = expand(0, frozenset())
    return [
        (outcomes, probability, tuple(outcomes)) for outcomes, probability in chains
    ]


def expand_exhaustive(
    mcts: GumbelMCTS, node: ClosedNode, depth: int | None = None
) -> None:
    """Fully expand a closed subtree, materializing every chance outcome with
    exact probabilities. ``depth=None`` runs to terminal; a finite depth stops
    there and evaluates the frontier with the net (the §5 net-leaves gate).
    Gate/verifier utility for small positions — raises on AGE_DEAL, and on an
    approximate edge: appending the omitted outcomes to a closed support would
    push its re-normalised mass past 1 and corrupt the tree in place, several
    steps before `closed_root_exact_value` got the chance to refuse it."""

    if node.terminal:
        return
    if depth is not None and depth <= 0:
        if node.visits == 0:
            value_p0, _ = mcts._evaluate(node.state)
            node.visits = 1
            node.value_sum_p0 = value_p0
        return
    if not node.edges:
        mcts._expand_closed(node)
    next_depth = None if depth is None else depth - 1
    for edge in node.edges:
        if edge.fixed_support:
            raise ValueError(
                "exhaustive expansion reached an approximate fixed-support edge; "
                "its support is closed and cannot be completed in place"
            )
        for outcomes, probability, key in enumerate_chains(node.state, edge.specs):
            if key not in edge.children:
                clone = node.state.clone()
                clone.search_barrier = True
                apply_action(
                    clone,
                    decode_action(clone, edge.action_index),
                    chance_outcomes=outcomes or None,
                )
                edge.children[key] = _Child(
                    probability=probability, node=mcts._make_closed_node(clone)
                )
        for child in edge.children.values():
            expand_exhaustive(mcts, child.node, next_depth)


# --------------------------------------------------------------------------
# Exact recomputation over a (fully expanded) closed tree — the §5 gate hook.
# --------------------------------------------------------------------------


def closed_root_exact_value(node: ClosedNode) -> float:
    """Recursive exact value (p0 terms): max over edges at decision nodes,
    probability-weighted expectation at chance edges, net value at evaluated
    frontier leaves (depth-limited trees). The contract is strict: enumerable
    chance edges must carry their FULL probability mass — a partial tree
    raises instead of silently returning a conditional value."""

    if node.terminal:
        return _terminal_value_p0(node.state)
    if not node.edges:
        if node.visits:
            return node.value_p0  # evaluated frontier leaf
        raise ValueError("exact value reached an unexpanded, unevaluated node")
    sign = 1.0 if node.actor == 0 else -1.0
    best = -math.inf
    for edge in node.edges:
        if not edge.children:
            raise ValueError("exact value requires every edge expanded")
        if edge.fixed_support:
            raise ValueError(
                "exact value reached an approximate fixed-support edge — its "
                "children are a re-normalised subset, not the full outcome space"
            )
        if any(child.probability is None for child in edge.children.values()):
            # Sample-only chance (AGE_DEAL): Monte Carlo mean over samples.
            weight = sum(child.samples for child in edge.children.values())
            if weight == 0:
                raise ValueError("sample-only edge has no sampled descents")
            value = (
                sum(
                    child.samples * closed_root_exact_value(child.node)
                    for child in edge.children.values()
                )
                / weight
            )
        else:
            mass = sum(child.probability for child in edge.children.values())
            if abs(mass - 1.0) > 1e-9:
                raise ValueError(
                    f"chance edge holds probability mass {mass:.6f} != 1 — "
                    "missing outcomes; expand exhaustively before exact_value"
                )
            value = sum(
                child.probability * closed_root_exact_value(child.node)
                for child in edge.children.values()
            )
        best = max(best, sign * value)
    return sign * best
