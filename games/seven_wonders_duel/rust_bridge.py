"""Cold-path helpers for constructing the Rust 7WD engine from Python data.

F4 keeps setup/checkpoint orchestration in Python.  This module centralizes the
fully locked constructor boundary so quality/benchmark tools do not import test
helpers or subtly disagree about setup ordering.
"""

from __future__ import annotations

import time

from .buffer import GameRecord, GameRecorder, replay
from .codec import decode_action, legal_action_indices
from .data import PROGRESS_IDS
from .encoder import Encoding, Token, TokenType
from .engine import apply_action
from .game import new_game


def rust_setup(game) -> dict:
    """Constructor kwargs for ``seven_wonders_rust.RustGame`` from a fresh game."""

    return {
        "first_player": game.first_player,
        "available_progress": list(game.available_progress_tokens),
        "unused_progress": list(game.unused_progress_tokens),
        "wonder_group0": list(game.wonder_groups[0]),
        "wonder_group1": list(game.wonder_groups[1]),
        "unused_wonders": list(game.unused_wonders),
        "age1": list(game.age_decks[1]),
        "age2": list(game.age_decks[2]),
        "age3": list(game.age_decks[3]),
        "removed1": list(game.removed_age_cards[1]),
        "removed2": list(game.removed_age_cards[2]),
        "removed3": list(game.removed_age_cards[3]),
        "selected_guilds": list(game.selected_guilds),
        "unused_guilds": list(game.unused_guilds),
    }


def rust_game_from_prefix(seed: int, first_player: int, prefix: list[int]):
    """Return ``(python_state, rust_state)`` after replaying an action prefix.

    Great Library draws are discovered in a Python prepass and supplied to Rust
    up front. Search-time chance outcomes are explicitly sampled/materialized by
    the searcher, so no additional hidden RNG crosses the boundary.
    """

    try:
        import seven_wonders_rust as swr
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "seven_wonders_rust is not installed; run maturin develop in seven_wonders_rust/"
        ) from exc

    fresh = new_game(seed, first_player=first_player)
    setup = rust_setup(fresh)
    python_state = fresh.clone()
    library_draws: list[list[str]] = []
    for index in prefix:
        action = decode_action(python_state, index)
        apply_action(python_state, action)
        if action.wonder_name == "The Great Library":
            pending = python_state.pending_choice
            # If no unused progress tokens remain, the effect is a deliberate
            # no-op and neither engine consumes a draw.
            if pending is not None:
                library_draws.append(list(pending.options))

    rust_state = swr.RustGame(library_draws=library_draws, **setup)
    for index in prefix:
        rust_state.apply_index(index)
    return python_state, rust_state


def rust_game_for_self_play(seed: int, first_player: int = 0):
    """Construct a fresh Rust game with its one play-time RNG event locked.

    Setup consumes all Python RNG draws except the possible Great Library draw.
    Precomputing that sample is therefore equivalent to drawing it when the
    Wonder is built, and lets the complete hot path remain inside Rust.
    """

    try:
        import seven_wonders_rust as swr
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "seven_wonders_rust is not installed; run maturin develop in seven_wonders_rust/"
        ) from exc

    game = new_game(seed, first_player=first_player)
    count = min(3, len(game.unused_progress_tokens))
    draws: list[list[str]] = []
    if count:
        draw = game.rng.sample(game.unused_progress_tokens, count)
        draws.append(sorted(draw, key=PROGRESS_IDS.__getitem__))
    return swr.RustGame(library_draws=draws, **rust_setup(game))


def rust_games_for_self_play(
    seeds: list[int], first_players: list[int]
) -> list:
    """Build locked cooperative-scheduler inputs in deterministic job order."""

    if len(seeds) != len(first_players):
        raise ValueError("seeds and first_players must have equal length")
    return [
        rust_game_for_self_play(seed, first_player)
        for seed, first_player in zip(seeds, first_players)
    ]


def rust_global_batch_adapter(evaluator):
    """Adapt the current evaluator to F4.4's one-call global batch contract.

    F4.5 replaces these Python ``Token`` objects with reusable flat buffers; the
    row ownership and gathered-legal-policy contract introduced here stays the
    same.
    """

    token_types = list(TokenType)

    def adapter(rows):
        encodings = []
        legal_lists = []
        for tokens, actor, legal in rows:
            encodings.append(
                Encoding(
                    actor=actor,
                    tokens=tuple(
                        Token(
                            token_types[type_id],
                            entity_id,
                            aux_id,
                            tuple(features),
                        )
                        for type_id, entity_id, aux_id, features in tokens
                    ),
                )
            )
            legal_lists.append(list(legal))
        results = evaluator.evaluate(encodings, legal_lists)
        if len(results) != len(rows):
            raise ValueError(
                f"evaluator returned {len(results)} rows for {len(rows)} inputs"
            )
        return [
            (
                float(result.wdl[0] - result.wdl[2]),
                [float(probability) for probability in result.policy],
            )
            for result in results
        ]

    return adapter


class _RustFlatBatchAdapter:
    """F4.5 packed-buffer transformer boundary with compact result transfer."""

    def __init__(
        self, evaluator, *, diagnostic_sync: bool = False, pinned_memory: bool = False
    ):
        self.evaluator = evaluator
        self.diagnostic_sync = diagnostic_sync
        self.pinned_memory = pinned_memory
        self.last_metrics: dict[str, float | int] = {}
        self.total_metrics: dict[str, float | int] = {
            "batches": 0,
            "rows": 0,
            "tokens": 0,
            "tensor_seconds": 0.0,
            "h2d_seconds": 0.0,
            "forward_seconds": 0.0,
            "gather_seconds": 0.0,
            "d2h_seconds": 0.0,
        }
        self.batch_rows: list[int] = []
        self.batch_tokens: list[int] = []
        self.batch_padded_tokens: list[int] = []

    def _sync(self):
        if self.diagnostic_sync and str(self.evaluator.device).startswith("cuda"):
            import torch

            torch.cuda.synchronize(self.evaluator.device)

    def __call__(self, payload):
        import torch

        rows = int(payload["rows"])
        tokens = int(payload["tokens"])
        width = int(payload["feature_width"])
        if rows <= 0 or rows > self.evaluator.max_batch:
            raise ValueError(
                f"flat batch rows {rows} outside evaluator max {self.evaluator.max_batch}"
            )

        tensor_start = time.perf_counter()
        token_offsets = torch.frombuffer(payload["token_offsets"], dtype=torch.int32).long()
        lengths = token_offsets[1:] - token_offsets[:-1]
        if len(lengths) != rows or int(lengths.sum()) != tokens:
            raise ValueError("flat token offsets do not align")
        max_tokens = int(payload["max_tokens"])
        pin_memory = self.pinned_memory and self.evaluator.device != "cpu"
        row_ids = torch.repeat_interleave(torch.arange(rows), lengths)
        starts = torch.repeat_interleave(token_offsets[:-1], lengths)
        columns = torch.arange(tokens) - starts

        type_ids = torch.zeros(rows, max_tokens, dtype=torch.long, pin_memory=pin_memory)
        entity_ids = torch.zeros(rows, max_tokens, dtype=torch.long, pin_memory=pin_memory)
        aux_ids = torch.zeros(rows, max_tokens, dtype=torch.long, pin_memory=pin_memory)
        features = torch.zeros(
            rows, max_tokens, width, dtype=torch.float32, pin_memory=pin_memory
        )
        pad_mask = torch.ones(rows, max_tokens, dtype=torch.bool, pin_memory=pin_memory)
        type_ids[row_ids, columns] = torch.frombuffer(
            payload["type_ids"], dtype=torch.uint8
        ).long()
        entity_ids[row_ids, columns] = torch.frombuffer(
            payload["entity_ids"], dtype=torch.int16
        ).long()
        aux_ids[row_ids, columns] = torch.frombuffer(
            payload["aux_ids"], dtype=torch.int16
        ).long()
        # Match dataset.vectorize's float16 storage before the model consumes
        # float32 tensors; this keeps the new boundary checkpoint-equivalent.
        packed_features = torch.frombuffer(
            payload["features"], dtype=torch.float32
        ).reshape(tokens, width)
        features[row_ids, columns] = packed_features.to(torch.float16).to(torch.float32)
        pad_mask[row_ids, columns] = False
        legal_offsets = torch.frombuffer(
            payload["legal_offsets"], dtype=torch.int32
        ).long()
        legal_lengths = legal_offsets[1:] - legal_offsets[:-1]
        legal_actions = torch.frombuffer(
            payload["legal_actions"], dtype=torch.uint16
        ).long()
        if len(legal_lengths) != rows or int(legal_lengths.sum()) != len(legal_actions):
            raise ValueError("flat legal offsets do not align")
        tensor_seconds = time.perf_counter() - tensor_start

        h2d_start = time.perf_counter()
        batch = {
            "type_ids": type_ids,
            "entity_ids": entity_ids,
            "aux_ids": aux_ids,
            "features": features,
            "pad_mask": pad_mask,
            "actors": torch.frombuffer(payload["actors"], dtype=torch.uint8).long(),
        }
        if self.evaluator.device != "cpu":
            batch = {
                key: value.to(self.evaluator.device, non_blocking=True)
                for key, value in batch.items()
            }
        self._sync()
        h2d_seconds = time.perf_counter() - h2d_start

        forward_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.evaluator.model(batch)
        self._sync()
        forward_seconds = time.perf_counter() - forward_start

        gather_start = time.perf_counter()
        device = outputs["policy"].device
        legal_rows = torch.repeat_interleave(
            torch.arange(rows, device=device), legal_lengths.to(device)
        )
        compact_logits = outputs["policy"][legal_rows, legal_actions.to(device)]
        compact_policy = []
        offset = 0
        for count in legal_lengths.tolist():
            compact_policy.append(torch.softmax(compact_logits[offset : offset + count], dim=0))
            offset += count
        compact_policy_tensor = torch.cat(compact_policy)
        wdl = torch.softmax(outputs["value"], dim=-1)
        value_actor = wdl[:, 0] - wdl[:, 2]
        self._sync()
        gather_seconds = time.perf_counter() - gather_start

        d2h_start = time.perf_counter()
        policy_cpu = compact_policy_tensor.float().cpu()
        value_cpu = value_actor.float().cpu()
        self._sync()
        d2h_seconds = time.perf_counter() - d2h_start

        legal_counts = legal_lengths.tolist()
        result = []
        offset = 0
        for row, count in enumerate(legal_counts):
            result.append(
                (
                    float(value_cpu[row]),
                    [float(value) for value in policy_cpu[offset : offset + count]],
                )
            )
            offset += count

        current = {
            "batches": 1,
            "rows": rows,
            "tokens": tokens,
            "tensor_seconds": tensor_seconds,
            "h2d_seconds": h2d_seconds,
            "forward_seconds": forward_seconds,
            "gather_seconds": gather_seconds,
            "d2h_seconds": d2h_seconds,
        }
        self.last_metrics = current
        for key, value in current.items():
            self.total_metrics[key] += value
        self.batch_rows.append(rows)
        self.batch_tokens.append(tokens)
        self.batch_padded_tokens.append(rows * max_tokens)
        return result


def rust_flat_batch_adapter(
    evaluator, *, diagnostic_sync: bool = False, pinned_memory: bool = False
):
    """Return the F4.5 flat-buffer adapter for the current Torch evaluator."""

    return _RustFlatBatchAdapter(
        evaluator,
        diagnostic_sync=diagnostic_sync,
        pinned_memory=pinned_memory,
    )


def rust_seat_routed_flat_batch_adapter(
    evaluators, *, diagnostic_sync: bool = False, pinned_memory: bool = False
):
    """Route packed Rust rows to a different evaluator model for each seat.

    Encodings and values remain actor-relative; the packed ``actors`` byte
    selects which checkpoint evaluates each row. This keeps model-vs-model
    arena games on the same Rust engine/search/coalescer used by self-play.
    """

    import torch

    if len(evaluators) != 2:
        raise ValueError("seat-routed evaluation requires exactly two evaluators")
    devices = {str(evaluator.device) for evaluator in evaluators}
    if len(devices) != 1:
        raise ValueError("seat-routed evaluators must use the same device")

    class _SeatRoutedModel(torch.nn.Module):
        def __init__(self, models):
            super().__init__()
            self.models = torch.nn.ModuleList(models)

        def forward(self, batch):
            actors = batch["actors"]
            if torch.any((actors < 0) | (actors > 1)):
                raise ValueError("packed actor ids must be 0 or 1")
            combined = None
            for seat, model in enumerate(self.models):
                indices = torch.nonzero(actors == seat, as_tuple=False).flatten()
                if not len(indices):
                    continue
                seat_batch = {
                    key: value.index_select(0, indices)
                    for key, value in batch.items()
                    if key != "actors"
                }
                outputs = model(seat_batch)
                if combined is None:
                    combined = {
                        key: value.new_empty((len(actors), *value.shape[1:]))
                        for key, value in outputs.items()
                    }
                for key, value in outputs.items():
                    combined[key].index_copy_(0, indices, value)
            if combined is None:
                raise ValueError("seat-routed batch cannot be empty")
            return combined

    class _EvaluatorProxy:
        pass

    proxy = _EvaluatorProxy()
    proxy.device = evaluators[0].device
    proxy.max_batch = min(evaluator.max_batch for evaluator in evaluators)
    proxy.model = _SeatRoutedModel([evaluator.model for evaluator in evaluators])
    proxy.model.to(proxy.device).eval()
    return _RustFlatBatchAdapter(
        proxy,
        diagnostic_sync=diagnostic_sync,
        pinned_memory=pinned_memory,
    )


_CHANCE_KIND = {
    0: "card_reveal",
    1: "great_library_draw",
    2: "wonder_group_reveal",
    3: "age_deal",
}


def phase_d_record_from_rust(raw: dict, *, validate: bool = True) -> GameRecord:
    """Materialize a Phase-D ``GameRecord`` from one completed Rust game.

    This is deliberately cold-path work: Rust has already selected and applied
    every move and recorded all search/chance data. Python replays the finished
    action list once to compute the existing RNG-inclusive digests and mask
    hashes, preserving buffer schema 1 without putting Python between moves.
    """

    if raw.get("schema") != 1 or raw.get("spec_version") != "codec-1":
        raise ValueError("unsupported Rust self-play record schema")
    recorder = GameRecorder(
        int(raw["seed"]),
        first_player=int(raw["first_player"]),
        agents=dict(raw["agents"]),
        iteration=raw.get("iteration"),
    )
    expected_events: dict[int, list[tuple[str, str | tuple[str, ...]]]] = {}
    for event in raw["chance_log"]:
        kind = _CHANCE_KIND[int(event["kind_id"])]
        names = list(event["outcome"])
        outcome: str | tuple[str, ...]
        outcome = names[0] if kind == "card_reveal" else tuple(names)
        expected_events.setdefault(int(event["move_index"]), []).append((kind, outcome))

    for row in raw["moves"]:
        i = int(row["i"])
        if i != len(recorder._moves):
            raise ValueError(f"non-contiguous Rust move index {i}")
        legal = list(legal_action_indices(recorder.game))
        if legal != list(row["legal"]):
            raise ValueError(f"Rust/Python legal mask diverged at move {i}")
        visits = {action: int(count) for action, count in zip(legal, row["visits"])}
        policy = (
            {
                action: float(probability)
                for action, probability in zip(legal, row["policy_target"])
            }
            if row["policy_target"] is not None
            else None
        )
        before_events = len(recorder._chance_log)
        recorder.play(
            int(row["action"]),
            visits=visits,
            policy_target=policy,
            root_value=(
                float(row["root_value"]) if row["root_value"] is not None else None
            ),
            sims=int(row["sims"]),
            mode=str(row["mode"]),
            gumbel_topk=(
                tuple(int(x) for x in row["gumbel_topk"])
                if row["gumbel_topk"] is not None
                else None
            ),
            policy_excluded=bool(row["policy_excluded"]),
        )
        actual_events = recorder._chance_log[before_events:]
        expected = expected_events.pop(i, [])
        if actual_events != expected:
            raise ValueError(
                f"Rust/Python chance log diverged at move {i}: "
                f"{actual_events!r} != {expected!r}"
            )
    if expected_events:
        raise ValueError(f"Rust chance log has unconsumed move entries: {sorted(expected_events)}")

    record = recorder.finish()
    if (
        record.winner != raw["winner"]
        or record.victory_type != raw["victory_type"]
        or record.scores != (tuple(raw["scores"]) if raw["scores"] is not None else None)
    ):
        raise ValueError("Rust/Python final result diverged")
    if validate:
        replay(record)
    return record


def phase_d_records_from_rust(
    raw_records: list[dict], *, validate: bool = True
) -> list[GameRecord]:
    """Convert cooperative output without changing its deterministic order."""

    return [
        phase_d_record_from_rust(raw, validate=validate) for raw in raw_records
    ]
