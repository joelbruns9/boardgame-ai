"""W7 S1-S4: specialist opponents inside the training loop.

A specialist is the same architecture as the general net, fine-tuned from a
promoted general checkpoint, whose SEARCH leaf value carries a bonus for its
intended victory type (``search.LeafBias`` / ``eval.rs::LeafBias``). It gets no
self-play campaign of its own: it plays the opponent seat inside the ordinary
loop, exactly where an archived HOF checkpoint sits today.

This module owns everything about that arrangement which is *not* a search
change -- the opponent draw, the persisted lineage, the archive, the collapse
floor and the step-count arithmetic -- so that ``phase_d`` gains hooks rather
than a second lifecycle.

Three invariants are load-bearing and are each tested:

* **Lifecycles are separate.** A specialist must not be reset because the
  general's soft gate rejected the general's own candidate, and the general must
  not be reset because a specialist collapsed.
* **The floor is measured against a FROZEN reference.** A floor defined against
  the current generator moves under the thing it is meant to protect.
* **Step counts follow measured inflow.** At an opponent share of *f* a
  specialist sees roughly *f*/2 of the positions the general sees; giving it the
  general's step count memorises its buffer within a couple of iterations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
import shutil
from typing import Any

from games.az_loop.hof import HallOfFame, HOFEntry, _sha256

#: Victory types a specialist can be built for. Civilian is deliberately absent:
#: a "civilian specialist" is a general that avoids the two attacking endings,
#: which is not an attacker at all.
SPECIALIST_CLASSES = ("science", "military")

#: Buffer/class ids. 0 is reserved for the general, so a specialist id is never
#: confusable with "the learner" at the Rust boundary, where network 0 is the
#: learner by definition.
CLASS_IDS: dict[str, int] = {name: index + 1 for index, name in enumerate(SPECIALIST_CLASSES)}

GENERAL_ROUTE = "general"
NO_ROUTE = "none"

#: The victory class each specialist biases toward, in `search.VICTORY_OFFSETS`
#: spelling.
CLASS_VICTORY = {"science": "scientific", "military": "military"}


def route_for(class_name: str) -> str:
    """``"specialist:<id>"`` -- the route string a record and an example carry."""

    return f"specialist:{CLASS_IDS[class_name]}"


def class_of_route(route: str) -> str | None:
    """Inverse of :func:`route_for`; ``None`` for the general and for no route."""

    if not route.startswith("specialist:"):
        return None
    wanted = int(route.split(":", 1)[1])
    for name, class_id in CLASS_IDS.items():
        if class_id == wanted:
            return name
    raise ValueError(f"unknown specialist class id in route {route!r}")


@dataclass(frozen=True, slots=True)
class SpecialistConfig:
    """One specialist type's knobs.

    ``share`` is a fraction of ALL games, matching the plan's shares arithmetic:
    with 40% league games drawn HOF/science/military at 37.5/37.5/25, the
    expected game shares are 15/15/10.
    """

    name: str
    lambda_: float
    share: float
    symmetric: bool = False
    #: Train this specialist every Nth iteration. The alternative to scaling
    #: step counts down; both are supported because which is right depends on
    #: the measured inflow, and that is a run-time number.
    train_every: int = 1
    #: Score rate against the frozen anchor below which the specialist is
    #: reverted to its last good checkpoint and the event logged loudly.
    collapse_floor: float = 0.15
    #: Cap on how large a share of the general's policy inflow may come from
    #: lambda-zero reanalysis of this specialist's positions (S2b).
    reanalysis_share_cap: float = 0.25

    def __post_init__(self) -> None:
        if self.name not in SPECIALIST_CLASSES:
            raise ValueError(f"unknown specialist class {self.name!r}")
        if not 0.0 <= self.share <= 1.0:
            raise ValueError("specialist share must lie in [0, 1]")
        if self.lambda_ <= 0.0:
            raise ValueError("a specialist needs a positive lambda")
        if self.train_every < 1:
            raise ValueError("train_every must be at least 1")
        if not 0.0 <= self.collapse_floor < 0.5:
            # At or above 0.5 the floor would demand the specialist beat the
            # general, which is the opposite of what a specialist is for.
            raise ValueError("collapse_floor must lie in [0, 0.5)")
        if not 0.0 <= self.reanalysis_share_cap <= 1.0:
            raise ValueError("reanalysis_share_cap must lie in [0, 1]")

    @property
    def victory(self) -> str:
        return CLASS_VICTORY[self.name]

    @property
    def class_id(self) -> int:
        return CLASS_IDS[self.name]

    @property
    def route(self) -> str:
        return route_for(self.name)


def parse_specialists(spec: str) -> tuple[SpecialistConfig, ...]:
    """``"science:0.15:0.5,military:0.10:0.5"`` -> configs.

    Fields are ``name:share:lambda`` with optional ``:train_every``. A string
    form rather than a dict so the whole league fits in one CLI flag and lands
    verbatim in the run manifest, where a schedule change has to be visible.
    """

    if not spec.strip():
        return ()
    out: list[SpecialistConfig] = []
    for chunk in spec.split(","):
        parts = [piece.strip() for piece in chunk.split(":")]
        if len(parts) not in (3, 4):
            raise ValueError(
                f"specialist spec {chunk!r} must be name:share:lambda[:train_every]"
            )
        name, share, lam = parts[0], float(parts[1]), float(parts[2])
        every = int(parts[3]) if len(parts) == 4 else 1
        out.append(
            SpecialistConfig(name=name, share=share, lambda_=lam, train_every=every)
        )
    names = [config.name for config in out]
    if len(set(names)) != len(names):
        raise ValueError("a specialist class may appear only once")
    total = sum(config.share for config in out)
    if total > 1.0:
        raise ValueError(f"specialist shares sum to {total:.3f} > 1")
    return tuple(out)


def draw_opponent_class(
    rng: random.Random,
    hof_share: float,
    specialists: tuple[SpecialistConfig, ...],
) -> str | None:
    """Draw this iteration's opponent class, or ``None`` for pure self-play.

    ONE class per iteration, not a per-game mix. ``_SearcherRoutedModel.forward``
    hard-rejects any net id outside ``{0, 1}``, so a single generation call can
    carry at most two networks; rotation also keeps one opponent model cached on
    the device for the whole call, which is the reason the class already
    documents for choosing one archive per iteration.

    **The shares arithmetic, which is easy to get wrong by a factor of L.**
    Shares are fractions of ALL games. Let ``L`` be their sum. Each iteration
    plays a fraction ``L`` of its games as league games (see
    :func:`league_game_count`) against ONE drawn class, so the expected share of
    all games played against class ``c`` is ``P(draw c) * L``. Setting that equal
    to ``share_c`` gives ``P(draw c) = share_c / L`` -- the shares RENORMALISED
    among the classes, not used as draw probabilities directly. Drawing on the
    raw shares would deliver ``share_c * L``: 15% intended, 6% delivered at
    ``L = 0.4``.
    """

    weights = [("hof", hof_share)] + [
        (config.name, config.share) for config in specialists
    ]
    weights = [(name, share) for name, share in weights if share > 0.0]
    total = sum(share for _, share in weights)
    if total <= 0.0:
        return None
    if total > 1.0 + 1e-9:
        raise ValueError(f"opponent shares sum to {total:.3f} > 1")
    if len(weights) == 1:
        # ONE class: the answer is determined, so DO NOT consume a draw.
        #
        # `rng` is the same seeded generator the caller then hands to
        # `hof.sample`, and before W7 that sample was its first consumer.
        # Spending a value here shifts the stream and makes an existing
        # HOF-enabled run select different archived opponents at the same seed
        # and iteration -- a silent change to a running experiment, in the
        # configuration W7 is supposed to leave alone.
        return weights[0][0]
    draw = rng.random() * total
    cumulative = 0.0
    for name, share in weights:
        cumulative += share
        if draw < cumulative:
            return name
    return weights[-1][0]


def league_share(hof_share: float, specialists: tuple[SpecialistConfig, ...]) -> float:
    """``L``: the fraction of every iteration's games that are league games."""

    return hof_share + sum(config.share for config in specialists)


def league_game_count(
    games: int, hof_share: float, specialists: tuple[SpecialistConfig, ...]
) -> int:
    """How many of ``games`` this iteration plays against the drawn opponent.

    ``round(games * L)`` -- the WHOLE league allocation goes to the one class
    drawn, which is what makes the expected per-class share come out at
    ``share_c``. See :func:`draw_opponent_class`.
    """

    share = league_share(hof_share, specialists)
    if share <= 0.0:
        return 0
    return min(games, int(round(games * share)))


def steps_for_inflow(
    base_steps: int,
    general_inflow: int,
    model_inflow: int,
    *,
    minimum: int = 1,
) -> int:
    """Scale a model's train steps to ITS OWN measured inflow.

    cloud2 ran ``samples_per_new_position`` around 5.2. A specialist at an
    opponent share of 15% sees roughly 7.5% of the general's positions -- one
    seat of the games it appears in -- so the general's step count would put it
    near 65 samples per new position and memorise the buffer within a couple of
    iterations.

    ``model_inflow`` is DATA, not elapsed time: it is the count of unconsumed
    policy rows this model may learn from, banked across however many iterations
    supplied them. An earlier version multiplied the newest iteration's count by
    ``iterations_since_train + 1`` instead, which invented inflow whenever an
    iteration supplied none -- and with one opponent class per iteration that is
    the ordinary case, not an edge one. Inflows of 0, 0, 100 earned 75 steps
    where 100 rows warranted 25.

    Measured, never assumed: this is not ``share/2 * general_inflow`` once cheap
    moves, bot games and route exclusions are taken out.

    The arithmetic is samples-per-new-position matching. The general does
    ``base_steps`` on ``general_inflow`` rows; the specialist does
    ``base_steps * banked / general_inflow`` on ``banked`` rows, which is the
    same reuse. Banking over N iterations therefore scales the step count up on
    its own, with no separate iteration term to get wrong.
    """

    if base_steps < 0:
        raise ValueError("base_steps must be non-negative")
    if general_inflow <= 0 or model_inflow <= 0:
        return 0
    return max(minimum, int(round(base_steps * (model_inflow / general_inflow))))


@dataclass
class LineageState:
    """The persisted identity of one specialist, beside the general's own.

    Everything a resume needs that is not the weights: which checkpoint is live,
    which one to fall back to, how many updates it has taken, the lambda it was
    trained under, and the iteration its assignment stream is keyed on. Kept in
    a JSON file rather than inferred from the checkpoint directory so a partial
    iteration cannot leave the run guessing.
    """

    class_name: str
    lambda_: float
    victory: str
    symmetric: bool = False
    update_count: int = 0
    iterations_trained: int = 0
    #: Iterations elapsed since the last train step -- the ``train_every``
    #: CADENCE, and nothing else. It is deliberately not a data measure; see
    #: `banked_inflow`.
    iterations_since_train: int = 0
    #: Unconsumed policy rows this specialist may still learn from.
    #:
    #: The quantity `steps_for_inflow` is sized against. Persisted because an
    #: iteration that supplies no specialist data is ordinary (one opponent
    #: class per iteration), and a step count derived from elapsed iterations
    #: rather than banked rows invents training signal that does not exist.
    banked_inflow: int = 0
    #: The last iteration whose data was consumed by a train step.
    #:
    #: Specialist updates are committed inside `adapter.train`, before the
    #: controller commits the iteration, and the controller's rollback hook
    #: restores general artifacts only. Without a completion identity, retrying
    #: an interrupted iteration would apply a second update from the same
    #: source data. `None` means "never trained".
    last_trained_iteration: int | None = None
    latest: str | None = None
    last_good: str | None = None
    bootstrapped_from: str | None = None
    #: sha256s admitted to the archive since the last measurement above the
    #: floor. A rollback quarantines them: without this the rejected checkpoint
    #: stays the newest archive entry and keeps playing.
    quarantined: list[str] = field(default_factory=list)
    accepted_since_good: list[str] = field(default_factory=list)
    #: Score rate against the frozen anchor at the last measurement, and the
    #: iteration it was taken at.
    last_score: float | None = None
    last_score_iteration: int | None = None
    reverts: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


class SpecialistLineage:
    """One specialist's directory: weights, optimizer state, archive, journal.

    ``root/specialists/<class>/`` holds ``latest.pt``, ``last_good.pt``,
    ``optimizer.pt``, ``state.json`` and ``archive/``. Separate from the
    general's tree on purpose -- the two lifecycles must not be able to reset
    each other by sharing a path.
    """

    def __init__(self, root: str | Path, config: SpecialistConfig):
        self.config = config
        self.directory = Path(root) / "specialists" / config.name
        self.archive = HallOfFame(self.directory / "archive")
        self.state_path = self.directory / "state.json"

    # ---- persistence ------------------------------------------------------

    def load(self) -> LineageState:
        if not self.state_path.exists():
            return LineageState(
                class_name=self.config.name,
                lambda_=self.config.lambda_,
                victory=self.config.victory,
                symmetric=self.config.symmetric,
            )
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        return LineageState(**payload)

    def save(self, state: LineageState) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(asdict(state), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @property
    def latest_path(self) -> Path:
        return self.directory / "latest.pt"

    @property
    def last_good_path(self) -> Path:
        return self.directory / "last_good.pt"

    @property
    def optimizer_path(self) -> Path:
        return self.directory / "optimizer.pt"

    @property
    def last_good_optimizer_path(self) -> Path:
        return self.directory / "optimizer_last_good.pt"

    # ---- lifecycle --------------------------------------------------------

    def bootstrap(self, general_checkpoint: str | Path, iteration: int) -> LineageState:
        """Seed a specialist from a PROMOTED general checkpoint.

        The first accepted specialist of each type doubles as the frozen
        reference every measurement is read against, so it is archived here and
        never rebuilt (see :meth:`frozen_reference`).
        """

        source = Path(general_checkpoint)
        if not source.is_file():
            raise FileNotFoundError(source)
        self.directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, self.latest_path)
        shutil.copy2(source, self.last_good_path)
        state = self.load()
        state.latest = str(self.latest_path.resolve())
        state.last_good = str(self.last_good_path.resolve())
        state.bootstrapped_from = str(source.resolve())
        state.history.append(
            {"event": "bootstrap", "iteration": iteration, "source": str(source)}
        )
        self.save(state)
        self.archive.add(
            self.latest_path,
            iteration=iteration,
            tag=f"{self.config.name}_seed",
            metadata={"lambda": self.config.lambda_, "victory": self.config.victory},
        )
        # The seed is the frozen reference and is never quarantined: it is the
        # thing a rollback rolls back TO.
        return state

    def accept(self, candidate: str | Path, iteration: int, *, steps: int) -> LineageState:
        """Take a trained candidate as the live specialist.

        Deliberately NOT the general's soft-gate controller: that gate produced
        0 promotions over 38k games in cloud6, and a specialist population that
        silently never advances is a full run wasted before anyone notices. The
        floor below is the only thing that can reject a specialist, and it
        rejects divergence rather than an absence of improvement.
        """

        source = Path(candidate)
        if not source.is_file():
            raise FileNotFoundError(source)
        self.directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, self.latest_path)
        state = self.load()
        state.latest = str(self.latest_path.resolve())
        state.update_count += steps
        state.iterations_trained += 1
        state.iterations_since_train = 0
        # The banked rows have now been spent.
        state.banked_inflow = 0
        state.last_trained_iteration = iteration
        state.history.append(
            {"event": "accept", "iteration": iteration, "steps": steps}
        )
        # Archive from the FIRST accepted specialist, not later: attacking
        # styles are forgotten the same way general strategies are, and a
        # specialist that drifts takes its earlier style with it.
        entry = self.archive.add(
            self.latest_path,
            iteration=iteration,
            tag=self.config.name,
            metadata={"lambda": self.config.lambda_, "victory": self.config.victory},
        )
        # Provisional until a floor measurement clears it: a rollback has to be
        # able to take this entry back out of circulation.
        state.accepted_since_good.append(entry.sha256)
        self.save(state)
        return state

    def bank_inflow(self, rows: int) -> LineageState:
        """Record policy rows this specialist has not trained on yet."""

        state = self.load()
        state.banked_inflow += max(0, int(rows))
        self.save(state)
        return state

    def already_trained(self, iteration: int) -> bool:
        """Has this iteration's data already been consumed?

        The idempotence guard for a retried iteration. A retry re-runs
        generation from the same seeds, so the records are the same records; the
        specialist has already spent them and must not spend them again.
        """

        return self.load().last_trained_iteration == iteration

    def mark_good(self, iteration: int, score: float) -> LineageState:
        """Record a measurement above the floor, and pin the rollback target."""

        state = self.load()
        state.last_score = score
        state.last_score_iteration = iteration
        if self.latest_path.is_file():
            shutil.copy2(self.latest_path, self.last_good_path)
            state.last_good = str(self.last_good_path.resolve())
        if self.optimizer_path.is_file():
            # Snapshot the moments alongside the weights. Restoring weights
            # while keeping the optimizer state from the REJECTED update would
            # re-apply that update's momentum on the next step.
            shutil.copy2(self.optimizer_path, self.last_good_optimizer_path)
        # Everything admitted since the last clear measurement is now cleared.
        state.accepted_since_good = []
        self.save(state)
        return state

    def revert(self, iteration: int, score: float) -> LineageState:
        """Fall through the floor: restore the last good checkpoint, loudly."""

        state = self.load()
        if self.last_good_path.is_file():
            shutil.copy2(self.last_good_path, self.latest_path)
        # QUARANTINE every archive entry admitted since the last clear
        # measurement. Restoring `latest.pt` alone is not a rollback: generation
        # samples this archive, and the rejected checkpoint is its newest entry,
        # so a collapsed specialist would keep playing.
        state.quarantined.extend(state.accepted_since_good)
        state.accepted_since_good = []
        # Restore the optimizer to the moments that produced the last good
        # weights, or drop it. `_load_optimizer_state` treats absence as a cold
        # start, which is recoverable; carrying the rejected update's momentum
        # into the next step is not.
        if self.last_good_optimizer_path.is_file():
            shutil.copy2(self.last_good_optimizer_path, self.optimizer_path)
        else:
            self.optimizer_path.unlink(missing_ok=True)
        state.last_score = score
        state.last_score_iteration = iteration
        state.reverts += 1
        state.history.append(
            {"event": "revert", "iteration": iteration, "score": score}
        )
        self.save(state)
        return state

    def live_entry(self):
        """The checkpoint generation should actually play: `latest.pt`.

        Resolved from the LIVE file rather than from the newest archive entry.
        The two diverge after a rollback -- `latest.pt` is restored, the archive
        still ends with the rejected weights -- and reading the archive there put
        the collapsed specialist straight back into generation.
        """

        if not self.latest_path.is_file():
            return None
        checksum = _sha256(self.latest_path)
        state = self.load()
        for entry in self.archive.entries():
            if entry.sha256 == checksum:
                return entry
        return HOFEntry(
            path=str(self.latest_path.resolve()),
            sha256=checksum,
            source=str(self.latest_path.resolve()),
            iteration=state.iterations_trained,
            tag=self.config.name,
            created_at_utc="",
            metadata={},
        )

    def sample_archive(self, rng: random.Random):
        """An archived entry that is not quarantined, or `None`."""

        quarantined = set(self.load().quarantined)
        live = [
            entry for entry in self.archive.entries()
            if entry.sha256 not in quarantined
        ]
        return rng.choice(live) if live else None

    def note_idle_iteration(self) -> LineageState:
        """One iteration passed without a train step.

        CADENCE only. Any rows this iteration produced were banked by
        `bank_inflow`; incrementing a counter here does not create data, and
        conflating the two is what made an inflow of zero earn steps.
        """

        state = self.load()
        state.iterations_since_train += 1
        self.save(state)
        return state

    def frozen_reference(self) -> str | None:
        """The FIRST accepted specialist of this type -- the frozen attacker.

        Fixed at the start of the run and never rebuilt. Scoring an improving
        general against an improving specialist cannot separate "defence got
        stronger" from "the attacks got weaker", and that ambiguity would make
        the workstream unfalsifiable.
        """

        entries = self.archive.entries()
        return entries[0].path if entries else None


def collapse_verdict(
    score: float, config: SpecialistConfig
) -> tuple[bool, str]:
    """``(healthy, reason)`` for one specialist's score against the anchor.

    Specialists are EXPECTED to score worse overall than the general, by design.
    That is not a failure; falling through the floor is. A low overall score
    with a reproducible exploit on the fixed panel is still worth keeping, which
    is why this reads one number and the panel is reported separately rather
    than folded in here.
    """

    if score < config.collapse_floor:
        return False, (
            f"{config.name} specialist scored {score:.3f} against the frozen "
            f"anchor, below its collapse floor of {config.collapse_floor:.3f}"
        )
    return True, f"{config.name} specialist healthy at {score:.3f}"


def inflow_census(examples) -> dict[str, int]:
    """Policy-eligible rows per model, from one derivation pass.

    "Generation is not free, and specialist games replace general games" is not
    cost-neutral for the general: replacing games preserves its GAME count, not
    its policy-target inflow, because an opponent seat's moves yield the general
    no policy label. This is that number, tracked separately from the shared
    value rows so the S3 step rule runs on a measurement.
    """

    census: dict[str, int] = {}
    for example in examples:
        route = getattr(example, "target_route", GENERAL_ROUTE)
        census.setdefault(route, 0)
        if example.has_policy:
            census[route] += 1
    return census


def shaped_share(examples) -> float:
    """Fraction of rows whose SOURCE search carried a nonzero lambda.

    Reported per buffer so "how much of what this model trained on came from a
    shaped search" is a measurement rather than an inference from the config.
    """

    if not examples:
        return 0.0
    shaped = sum(1 for e in examples if getattr(e, "search_lambda", 0.0))
    return shaped / len(examples)


def assert_no_shaped_bootstrap(examples, derived_for: str = GENERAL_ROUTE) -> None:
    """No example anywhere carries a shaped root into a value target.

    ``value_bootstrap`` blends ``Example.root_value`` into ``value_soft``, which
    trains a W/D/L PROBABILITY head -- the same head every search reads back as
    its leaf value. A shaped root there teaches the lambda bonus as win
    probability, and the search then adds the bonus a second time.

    This is asserted for the model that produced the bias as well as for
    everyone else. Routing a utility to its owner does not make it a
    probability, and there is no separate utility head to route it to; the
    shaped root is recorded for auditing and trains nothing. cloud2 ran
    ``--value-bootstrap 0.5``, so the configuration that makes this matter is
    the one that actually shipped.

    Reads ``Example.root_value_shaped``, a recorded fact, rather than
    re-deriving ``dataset.bootstrap_root_value``'s rule -- a refactor there
    cannot make this pass silently.
    """

    for example in examples:
        if getattr(example, "root_value_shaped", False):
            raise AssertionError(
                "a shaped search's root_value reached a value target "
                f"(route {getattr(example, 'target_route', '?')!r}, derived for "
                f"{getattr(example, 'derived_for', '?')!r}); the value head is a "
                "probability head and the search adds the bonus itself"
            )
        if getattr(example, "derived_for", GENERAL_ROUTE) != derived_for:
            raise AssertionError(
                "example provenance disagrees with the derivation it came from"
            )


# --------------------------------------------------------------------------
# S2b: transfer of attacking knowledge (lambda-zero reanalysis)
# --------------------------------------------------------------------------
#
# Without this, the general learns only to DEFEND. The specialist's attacking
# policy targets train the specialist; the general's own seat was defending.
#
# Retain the positions where the specialist's attack was live, re-search them at
# lambda = 0, and route the resulting targets to the general. That teaches the
# general to play SOUND attacks -- what the unbiased search makes of the position
# the biased agent steered into -- without teaching it to trade wins for a
# preferred victory type.


#: Default gap, in utility units, above which lambda is judged to have mattered
#: at a position.
#:
#: The plan calls for the flag "lambda changed the chosen move". That fact is not
#: recoverable from a record: the played action and the visit distribution are
#: both the BIASED search's, and a lambda-zero argmax over them cannot be
#: reconstructed after the fact without re-searching -- which is the expensive
#: thing this selection exists to avoid. What the record does hold is both root
#: utilities, so this uses the size of the gap between them as the proxy: the
#: bias moved this position's valuation by at least this much. It is a strictly
#: weaker signal, and it is named as one rather than presented as the flag the
#: plan asked for.
DEFAULT_REANALYSIS_GAP = 0.05


def reanalysis_candidates(
    record,
    *,
    min_gap: float = DEFAULT_REANALYSIS_GAP,
    include_type_wins: bool = True,
) -> list[int]:
    """Move indices worth re-searching at lambda = 0, for one game.

    Reanalysis is search compute, so this deliberately does not select
    everything. Two sources:

    * moves where the bias moved the search's own valuation by at least
      ``min_gap`` -- see :data:`DEFAULT_REANALYSIS_GAP` for what this is and is
      not;
    * every full-budget specialist move in a game the specialist WON BY ITS OWN
      TYPE, whether or not any single position shows a large gap. A rush that
      worked is exactly the line the general should learn to play, and the
      per-move gap can be small all along a plan that only pays off at the end.
    """

    victory = getattr(record, "victory_type", None)
    winner = getattr(record, "winner", None)
    selected: list[int] = []
    type_win_seats: set[int] = set()
    if include_type_wins and winner is not None and victory is not None:
        for move in record.moves:
            lam = getattr(move, "search_lambda", 0.0)
            if not lam or move.actor != winner:
                continue
            wanted = getattr(move, "search_victory", None)
            if wanted is not None and str(victory).lower().endswith(wanted.lower()):
                type_win_seats.add(move.actor)
    for move in record.moves:
        if not getattr(move, "search_lambda", 0.0):
            continue
        if getattr(move, "policy_excluded", False):
            # A cheap search is not a position worth spending a full re-search
            # on; its own target was never good enough to train on either.
            continue
        unshaped = getattr(move, "root_value_unshaped", None)
        gap = (
            abs(move.root_value - unshaped)
            if move.root_value is not None and unshaped is not None
            else 0.0
        )
        if gap >= min_gap or move.actor in type_win_seats:
            selected.append(move.i)
    return selected


def cap_reanalysis(
    selected_per_record: list[list[int]],
    general_inflow: int,
    cap: float,
) -> list[list[int]]:
    """Trim the selection so reanalysis cannot dominate the general's buffer.

    The share is a cap on the general's POLICY inflow, which is the quantity
    reanalysis rows compete for -- not on the buffer, most of which is value
    rows the reanalysis does not touch. Trimming is round-robin across games so
    a single long game cannot consume the whole allowance.
    """

    # A count allowance across the WHOLE cap range. `cap >= 1.0` used to short
    # circuit to "everything", and a zero inflow to the same thing, so the two
    # configurations that most need a bound were the two that had none.
    if general_inflow <= 0:
        return [[] for _ in selected_per_record]
    budget = int(general_inflow * min(cap, 1.0))
    if budget <= 0:
        return [[] for _ in selected_per_record]
    total = sum(len(entry) for entry in selected_per_record)
    if total <= budget:
        return selected_per_record
    kept: list[list[int]] = [[] for _ in selected_per_record]
    cursor = 0
    remaining = budget
    while remaining > 0:
        progressed = False
        for index, entry in enumerate(selected_per_record):
            if cursor < len(entry):
                kept[index].append(entry[cursor])
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
        cursor += 1
    return kept


def reanalysis_examples(
    record,
    move_indices,
    search_factory,
    *,
    derived_for: str = GENERAL_ROUTE,
):
    """Re-search the selected positions and emit GENERAL examples.

    Returns ``(examples, coverage)``; ``coverage`` counts how often the
    unbiased re-search visited the move the specialist actually played.

    ``search_factory(move_index) -> object with .search(state)`` supplies a
    lambda-ZERO searcher driven by the GENERAL's own current net. These are
    targets for the general, not a second opinion from the specialist.

    **Re-searched from the actor's observation, not the realised deal.** The
    recorded pre-move state carries no hidden card identities -- reveals are
    chance events the search samples for itself -- so replaying to a position and
    searching it sees exactly what the player saw. A reanalysis that read the
    realised deal would produce targets no player could have computed, and would
    train the general on clairvoyant play.
    """

    import numpy as np

    from .buffer import replay
    from .codec import legal_action_indices
    from .dataset import (
        Example,
        _actor_value_class,
        _joint7_class,
        vectorize,
    )
    from .encoder import encode
    from .game import VictoryType

    wanted = set(move_indices)
    if not wanted:
        return [], {"specialist_move_visited": 0, "positions": 0}
    staged: list[tuple] = []
    coverage: list[bool] = []

    def visit(game, move):
        if move.i not in wanted:
            return
        actor = (
            game.pending_choice.player
            if game.pending_choice is not None
            else game.active_player
        )
        legal = np.asarray(legal_action_indices(game), dtype=np.int16)
        result = search_factory(move.i).search(game)
        # Did the unbiased re-search fund the move the SPECIALIST actually
        # played? The plan asks that the specialist's candidates get enough
        # coverage that the general's weak prior cannot simply exclude them
        # again. This measures whether that is a live problem here before any
        # mechanism is built for it: at 7WD's median branching of 4 against a
        # top-k of 16, the candidate set should already contain it.
        coverage.append(bool(result.visits.get(move.action, 0)))
        policy = np.zeros(len(legal), dtype=np.float32)
        for position, action in enumerate(legal):
            policy[position] = float(result.policy_target.get(int(action), 0.0))
        mass = float(policy.sum())
        if mass <= 0.0:
            return
        policy /= mass
        encoding = encode(game.observation(actor))
        staged.append((vectorize(encoding), legal, policy, actor, result))

    game = replay(record, on_state=visit)
    victory = (
        VictoryType(record.victory_type) if record.victory_type is not None else None
    )
    scores = record.scores
    final_position = game.conflict_position
    from .dataset import _science_symbols

    sci_counts = (len(_science_symbols(game, 0)), len(_science_symbols(game, 1)))
    examples = []
    for (tokens, legal, policy, actor, result) in staged:
        type_ids, entity_ids, aux_ids, features = tokens
        if scores is not None:
            margin, margin_valid = (scores[actor] - scores[1 - actor]) / 20.0, True
        else:
            margin, margin_valid = 0.0, False
        relative = final_position if actor == 0 else -final_position
        examples.append(
            Example(
                type_ids=type_ids,
                entity_ids=entity_ids,
                aux_ids=aux_ids,
                features=features,
                legal=legal,
                policy_target=policy,
                has_policy=True,
                value_class=_actor_value_class(record.winner, actor),
                joint7_class=_joint7_class(record.winner, victory, actor),
                margin=margin,
                margin_valid=margin_valid,
                military_final=relative / 9.0,
                sci_final_my=sci_counts[actor] / 6.0,
                sci_final_opp=sci_counts[1 - actor] / 6.0,
                game_key=record.seed,
                iteration=record.iteration,
                # The re-search ran at lambda = 0, so its root IS a calibrated
                # win probability and needs no quarantine.
                root_value=result.root_value,
                root_value_shaped=False,
                target_route=derived_for,
                derived_for=derived_for,
                search_lambda=0.0,
                search_victory=None,
                reanalysis=True,
            )
        )
    return examples, {
        "specialist_move_visited": sum(coverage),
        "positions": len(coverage),
    }
