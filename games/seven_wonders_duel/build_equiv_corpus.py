"""Regenerate the committed engine-equivalence corpus (W6.2).

A corpus record is ``(seed, first_player, action indices, chance log)``, so it
only describes a real game under the engine that produced it. The age-deal
reordering (``ENGINE_AGE_DEAL_ORDERING.md``, ``SPEC_VERSION`` codec-2) moved the
Age deal ahead of the start-player choice, which changes the chance stream of
every game a seed produces. Records written under codec-1 still *drive* both
engines identically -- which is why the equivalence gate kept passing -- but the
trajectories they walk are no longer ones this engine generates, so the gate
that runs on a rented box before training would be checking parity over a
distribution the run will never see.

This script rebuilds the corpus under the current engine, in three strata that
mirror what a real run puts in its buffers:

* ``curriculum_seed`` -- bot-vs-bot seed games, exactly ``phase_d``'s seed step;
* ``selfplay_early``  -- searched self-play from an untrained net, with the
  launch curriculum mix and a full draft prior, i.e. iteration 0;
* ``selfplay_late``   -- searched self-play from a trained checkpoint, no bot
  mix and no draft prior, i.e. a run past its anneal.

The late stratum needs a real checkpoint and there is no substitute: filling it
from an untrained net would leave the corpus with no late-game distribution at
all while still reporting 50 games, so a missing checkpoint is an error rather
than a fallback.

Coverage is asserted before anything is written -- all 9 decision branches and
all 9 token types, the property ``test_encode_corpus_equivalent`` relies on --
and every record is replayed through ``buffer.replay``, which verifies masks,
the chance log and the final digest against this engine.

Usage (from the repo root)::

    python -m games.seven_wonders_duel.build_equiv_corpus
    python -m games.seven_wonders_duel.build_equiv_corpus --check   # no writes

Provenance of the committed files is recorded in ``testdata/equiv_corpus/
PROVENANCE.md``; re-running with the same flags and checkpoint reproduces them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from games.az_loop import GameJob

from . import phase_d as pd
from .buffer import GameRecord, replay, to_json_line
from .encoder import TokenType, encode
from .inference import Evaluator
from .train import build_model, heads_from_config, load_checkpoint

CORPUS_DIR = Path(__file__).resolve().parent / "testdata" / "equiv_corpus"
DEFAULT_LATE_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "runs"
    / "laptop_training_03_w7"
    / "checkpoints"
    / "current_best.pt"
)

# 50 games total: the floor `test_rust_engine_equiv.CORPUS_MIN_GAMES` enforces.
CURRICULUM_GAMES = 20
EARLY_GAMES = 15
LATE_GAMES = 15

# Distinct seed bases so the three strata never replay the same game, and so a
# stratum can be regenerated alone without colliding with the others.
CURRICULUM_SEED_BASE = 700_000
EARLY_SEED_BASE = 800_000
LATE_SEED_BASE = 900_000

ALL_TOKEN_TYPES = frozenset(range(len(TokenType)))
ALL_DECISIONS = frozenset(range(9))
_TOKEN_TYPE_INDEX = {t: i for i, t in enumerate(TokenType)}


def _curriculum_records(count: int) -> list[GameRecord]:
    return [
        pd._bot_seed_game(
            GameJob(
                index=index,
                seed=CURRICULUM_SEED_BASE + index,
                kind="curriculum_seed",
            )
        )
        for index in range(count)
    ]


def _selfplay_records(
    count: int,
    *,
    seed_base: int,
    evaluator: Evaluator,
    config: pd.PhaseDConfig,
    iteration: int,
    curriculum_mix_fraction: float,
    draft_prior: float,
) -> list[GameRecord]:
    schedules = pd.ResolvedSchedules(
        curriculum_mix_fraction=curriculum_mix_fraction,
        draft_prior=draft_prior,
    )
    return [
        pd._self_play_game(
            GameJob(index=index, seed=seed_base + index),
            evaluator,
            config,
            iteration,
            schedules,
        )
        for index in range(count)
    ]


def _evaluator_for(
    checkpoint_path: Path | None, config: pd.PhaseDConfig, *, seed: int
) -> Evaluator:
    """An evaluator over a trained checkpoint, or over a seeded untrained net.

    The untrained net is seeded so the early stratum is reproducible; without
    that, "regenerate the corpus" would produce different games every run and
    the committed files could never be checked against the script.
    """

    if checkpoint_path is None:
        torch.manual_seed(seed)
        model = build_model("transformer", config.d_model, config.layers, None)
    else:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved = payload.get("config", {})
        model = build_model(
            "transformer",
            int(saved.get("d_model", config.d_model)),
            int(saved.get("layers", config.layers)),
            heads_from_config(saved),
        )
        load_checkpoint(checkpoint_path, model, checkpoint=payload)
    return Evaluator(model, config.device, precision="fp32")


def _coverage(records: list[GameRecord]) -> tuple[set[int], set[int], int]:
    """Replay every record and collect encoder coverage.

    ``replay`` is the verification: it re-runs the game from ``(seed, actions)``
    and raises unless every mask, chance event and the final digest match this
    engine. A record that survives it is one this engine really produces.
    """

    token_types: set[int] = set()
    decisions: set[int] = set()
    states = 0

    def observe(game):
        nonlocal states
        states += 1
        tokens = encode(game.observation(0)).tokens
        token_types.update(_TOKEN_TYPE_INDEX[tok.type] for tok in tokens)
        # The GLOBAL token is always tokens[0] and carries the decision one-hot
        # in its leading 9 features.
        decisions.add(list(tokens[0].features[:9]).index(1.0))

    for record in records:
        final = replay(record, on_state=lambda game, _move: observe(game))
        # The terminal COMPLETE state is a decision branch of its own and is not
        # visited by `on_state`; `test_encode_corpus_equivalent` encodes it too.
        observe(final)
    return token_types, decisions, states


def _write(path: Path, records: list[GameRecord]) -> None:
    path.write_text(
        "".join(to_json_line(record) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )


def build(args) -> int:
    config = pd.PhaseDConfig(
        device=args.device,
        cheap_sims_min=args.cheap_sims_min,
        cheap_sims_max=args.cheap_sims_max,
        full_sims_min=args.full_sims_min,
        full_sims_max=args.full_sims_max,
    )

    late_checkpoint = Path(args.late_checkpoint)
    if not late_checkpoint.is_file():
        raise SystemExit(
            f"late-stratum checkpoint not found: {late_checkpoint}\n"
            "The corpus needs a trained net for its late self-play stratum; an "
            "untrained one would leave the corpus with no late distribution "
            "while still reporting 50 games. Pass --late-checkpoint."
        )

    print(f"curriculum_seed: {CURRICULUM_GAMES} bot games")
    curriculum = _curriculum_records(CURRICULUM_GAMES)

    print(f"selfplay_early: {EARLY_GAMES} games from an untrained net")
    early = _selfplay_records(
        EARLY_GAMES,
        seed_base=EARLY_SEED_BASE,
        evaluator=_evaluator_for(None, config, seed=args.untrained_seed),
        config=config,
        iteration=0,
        # The launch curriculum mix and a full draft prior: iteration 0.
        curriculum_mix_fraction=config.opponent_fraction,
        draft_prior=1.0,
    )

    print(f"selfplay_late: {LATE_GAMES} games from {late_checkpoint}")
    late = _selfplay_records(
        LATE_GAMES,
        seed_base=LATE_SEED_BASE,
        evaluator=_evaluator_for(late_checkpoint, config, seed=args.untrained_seed),
        config=config,
        iteration=args.late_iteration,
        # Past both anneals: no bot mix, no draft prior.
        curriculum_mix_fraction=0.0,
        draft_prior=0.0,
    )

    strata = {
        "curriculum_seed": curriculum,
        "selfplay_early": early,
        "selfplay_late": late,
    }
    every = curriculum + early + late

    token_types, decisions, states = _coverage(every)
    missing_types = ALL_TOKEN_TYPES - token_types
    missing_decisions = ALL_DECISIONS - decisions
    if missing_types or missing_decisions:
        raise SystemExit(
            "corpus does not cover the encoder: "
            f"missing token types {sorted(missing_types)}, "
            f"missing decisions {sorted(missing_decisions)}. "
            "Nothing was written."
        )

    print(
        f"verified {len(every)} games / {states} states: "
        f"decisions={sorted(decisions)} token_types={sorted(token_types)}"
    )

    if args.check:
        print("--check: nothing written")
        return 0

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for name, records in strata.items():
        path = CORPUS_DIR / f"{name}.jsonl"
        _write(path, records)
        print(f"wrote {path} ({len(records)} games)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--late-checkpoint",
        default=str(DEFAULT_LATE_CHECKPOINT),
        help="trained checkpoint for the late self-play stratum",
    )
    parser.add_argument(
        "--late-iteration",
        type=int,
        default=60,
        help="iteration recorded on the late stratum's records",
    )
    parser.add_argument(
        "--untrained-seed",
        type=int,
        default=20260803,
        help="torch seed for the early stratum's untrained net",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--cheap-sims-min", type=int, default=16)
    parser.add_argument("--cheap-sims-max", type=int, default=24)
    parser.add_argument("--full-sims-min", type=int, default=64)
    parser.add_argument("--full-sims-max", type=int, default=128)
    parser.add_argument(
        "--check",
        action="store_true",
        help="generate and verify coverage without writing the corpus",
    )
    return build(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
