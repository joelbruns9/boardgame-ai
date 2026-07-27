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
    """F4.5 packed-buffer transformer boundary with compact result transfer.

    Every ``*_seconds`` counter here is a **host** timer: it measures how long
    the calling thread spent in that section, which for CUDA work is enqueue
    time, not device execution. With ``cuda_events=True`` the same four sections
    are additionally bracketed by CUDA events, giving true device durations; the
    events are queried lazily (:meth:`drain_events`) so reading them does not
    itself synchronise the pipeline. Phase 0 of the throughput plan exists
    because the earlier measurements conflated these two.
    """

    #: Query and release recorded events once this many have accumulated. They
    #: belong to long-finished batches by then, so draining costs no stall.
    _EVENT_DRAIN_THRESHOLD = 4096

    def __init__(
        self,
        evaluator,
        *,
        diagnostic_sync: bool = False,
        pinned_memory: bool = False,
        cuda_events: bool = False,
        vectorized_gather: bool = True,
    ):
        self.evaluator = evaluator
        self.diagnostic_sync = diagnostic_sync
        self.pinned_memory = pinned_memory
        #: Phase 3: replace the per-row softmax loop with a segmented softmax, a
        #: single D2H transfer, and one bulk `tolist()`. All three of those costs
        #: scale per row while the per-batch cost is flat, so they are what makes
        #: a wider batch stop paying. Arithmetically the same computation in a
        #: different reduction order (~1e-7); on by default since Phase 3, which
        #: measured +33.6% end to end with identical scheduler work.
        self.vectorized_gather = bool(vectorized_gather)
        self.cuda_events = bool(cuda_events) and str(evaluator.device).startswith("cuda")
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
            # Host time spent inside explicit synchronisation, so it can be
            # subtracted from the sections above on --diagnostic-sync runs.
            "sync_seconds": 0.0,
            # Device time, populated only when cuda_events is on.
            "device_h2d_seconds": 0.0,
            "device_forward_seconds": 0.0,
            "device_gather_seconds": 0.0,
            "device_d2h_seconds": 0.0,
        }
        self.batch_rows: list[int] = []
        self.batch_tokens: list[int] = []
        self.batch_padded_tokens: list[int] = []
        #: Per-batch device forward duration in milliseconds, in batch order.
        self.batch_device_forward_ms: list[float] = []
        self._events: list[tuple[str, object, object]] = []

    def _sync(self):
        if self.diagnostic_sync and str(self.evaluator.device).startswith("cuda"):
            import torch

            started = time.perf_counter()
            torch.cuda.synchronize(self.evaluator.device)
            self.total_metrics["sync_seconds"] += time.perf_counter() - started

    def _begin_event(self):
        """Record a start event on the current stream, or return ``None``."""

        if not self.cuda_events:
            return None
        import torch

        start = torch.cuda.Event(enable_timing=True)
        start.record()
        return start

    def _end_event(self, name: str, start):
        if start is None:
            return
        import torch

        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._events.append((name, start, end))
        if len(self._events) >= self._EVENT_DRAIN_THRESHOLD:
            self.drain_events()

    def drain_events(self) -> None:
        """Fold recorded CUDA events into the device totals and release them.

        Safe to call at any time; call it once after a run before reading
        ``total_metrics``. Each event is synchronised individually, which is
        free for batches that have already completed.
        """

        for name, start, end in self._events:
            end.synchronize()
            seconds = start.elapsed_time(end) / 1000.0
            self.total_metrics[f"device_{name}_seconds"] += seconds
            if name == "forward":
                self.batch_device_forward_ms.append(seconds * 1000.0)
        self._events.clear()

    def build_device_batch(self, payload):
        """Unpack `payload` into the model's input tensors on the target device.

        Split out of :meth:`__call__` so instrumentation (the cost model's
        queue-depth probe) can drive the model with exactly the tensors
        production builds, rather than an approximation of them. Returns the
        batch, the per-row legal lengths and actions, and the host time spent in
        tensor construction and in the host-to-device enqueue.
        """

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
        h2d_event = self._begin_event()
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
        self._end_event("h2d", h2d_event)
        self._sync()
        h2d_seconds = time.perf_counter() - h2d_start
        return batch, legal_lengths, legal_actions, tensor_seconds, h2d_seconds

    def __call__(self, payload):
        import torch

        rows = int(payload["rows"])
        tokens = int(payload["tokens"])
        max_tokens = int(payload["max_tokens"])
        (
            batch,
            legal_lengths,
            legal_actions,
            tensor_seconds,
            h2d_seconds,
        ) = self.build_device_batch(payload)

        forward_start = time.perf_counter()
        forward_event = self._begin_event()
        with torch.no_grad():
            outputs = self.evaluator.model(batch)
        self._end_event("forward", forward_event)
        self._sync()
        forward_seconds = time.perf_counter() - forward_start

        gather_start = time.perf_counter()
        gather_event = self._begin_event()
        device = outputs["policy"].device
        legal_rows = torch.repeat_interleave(
            torch.arange(rows, device=device), legal_lengths.to(device)
        )
        compact_logits = outputs["policy"][legal_rows, legal_actions.to(device)]
        if self.vectorized_gather:
            # Segmented softmax: one set of kernels for the whole batch instead of
            # one `torch.softmax` launch per row. `legal_rows` already labels each
            # compacted logit with its row, so the row boundaries need no loop.
            # Same formula `torch.softmax` uses (subtract the segment max, then
            # normalise), so agreement is to float reduction order.
            segment_max = torch.full(
                (rows,),
                float("-inf"),
                device=device,
                dtype=compact_logits.dtype,
            )
            segment_max.scatter_reduce_(
                0, legal_rows, compact_logits, reduce="amax", include_self=True
            )
            shifted = (compact_logits - segment_max[legal_rows]).exp()
            denominator = torch.zeros(
                rows, device=device, dtype=shifted.dtype
            ).index_add_(0, legal_rows, shifted)
            # Rows with no legal actions (terminal leaves) leave a zero
            # denominator, but `legal_rows` never refers to them, so the division
            # never reads it.
            compact_policy_tensor = shifted / denominator[legal_rows]
        else:
            compact_policy = []
            offset = 0
            for count in legal_lengths.tolist():
                compact_policy.append(
                    torch.softmax(compact_logits[offset : offset + count], dim=0)
                )
                offset += count
            compact_policy_tensor = torch.cat(compact_policy)
        wdl = torch.softmax(outputs["value"], dim=-1)
        value_actor = wdl[:, 0] - wdl[:, 2]
        self._end_event("gather", gather_event)
        self._sync()
        gather_seconds = time.perf_counter() - gather_start

        # The first blocking `.cpu()` is where an asynchronous pipeline actually
        # waits for the device, so this host timer is a stall measurement, not a
        # transfer measurement; the paired device event separates the two.
        d2h_start = time.perf_counter()
        d2h_event = self._begin_event()
        if self.vectorized_gather:
            # One transfer, not two: values and policies are concatenated on the
            # device and split again on the host.
            merged = torch.cat(
                (value_actor.float(), compact_policy_tensor.float())
            ).cpu()
            value_cpu = merged[:rows]
            policy_cpu = merged[rows:]
        else:
            policy_cpu = compact_policy_tensor.float().cpu()
            value_cpu = value_actor.float().cpu()
        self._end_event("d2h", d2h_event)
        self._sync()
        d2h_seconds = time.perf_counter() - d2h_start

        legal_counts = legal_lengths.tolist()
        result = []
        offset = 0
        if self.vectorized_gather:
            # `tolist()` converts in one C call; indexing element by element cost
            # a Python `float()` per legal action, millions of them per run.
            values = value_cpu.tolist()
            policies = policy_cpu.tolist()
            for row, count in enumerate(legal_counts):
                result.append((values[row], policies[offset : offset + count]))
                offset += count
        else:
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
    evaluator,
    *,
    diagnostic_sync: bool = False,
    pinned_memory: bool = False,
    cuda_events: bool = False,
    vectorized_gather: bool = True,
):
    """Return the F4.5 flat-buffer adapter for the current Torch evaluator."""

    return _RustFlatBatchAdapter(
        evaluator,
        diagnostic_sync=diagnostic_sync,
        pinned_memory=pinned_memory,
        cuda_events=cuda_events,
        vectorized_gather=vectorized_gather,
    )


def rust_seat_routed_flat_batch_adapter(
    evaluators,
    *,
    diagnostic_sync: bool = False,
    pinned_memory: bool = False,
    cuda_events: bool = False,
    vectorized_gather: bool = True,
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
    # `.to()` runs `_apply` on every child, which invalidates each embedder's
    # fused cache by design. Re-fuse, or seat-routed arena play would silently
    # fall back to the per-type loop.
    from .net import fuse_for_inference

    for model in proxy.model.models:
        fuse_for_inference(model)
    return _RustFlatBatchAdapter(
        proxy,
        diagnostic_sync=diagnostic_sync,
        pinned_memory=pinned_memory,
        cuda_events=cuda_events,
        vectorized_gather=vectorized_gather,
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
