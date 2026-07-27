"""Phase 0 item 5: the controlled batch-size / token-length cost model.

Everything the throughput plan wants to decide — whether widening batches pays,
whether launch overhead dominates, where the per-row gather overtakes the
forward — reduces to one question: what does a batch of ``R`` rows and ``T``
tokens actually cost, on the host and on the device, separately?

Self-play cannot answer that. Its batch shapes are whatever the scheduler
happened to produce, its rows and tokens are correlated, and its timers are all
host timers. So this module measures the boundary directly:

* a corpus of **real** encoder outputs is collected by random-walking games, so
  token contents, type mixtures and legal-action counts are representative
  (``TokenEmbedder`` branches per token type, so synthetic tokens would not be);
* rows are drawn from one token-length bucket at a time, which decorrelates the
  row and token axes that self-play confounds;
* each configuration is replayed through the production flat adapter with CUDA
  events, giving host and device time for the same call;
* a least-squares fit then reports the model the plan asks for — a fixed cost
  per batch, a marginal cost per row, and a marginal cost per token.

Run without a search anywhere in the loop::

    python -m games.seven_wonders_duel.f4_cost_model \
        --checkpoint runs/.../best.pt --device cuda --output runs/cost_model

The fitted ``fixed_ms`` is the number Phase 0's branch point turns on: if the
device fit's fixed term carries the ~7.3 ms, batch widening is the lever; if the
host fit carries it while the device fit is near zero, the work is launch
overhead and Phase 3b (CUDA graphs / compile / static shapes) comes first.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import struct
import time
from pathlib import Path

import torch

from .phase_e import load_evaluator
from .rust_bridge import rust_flat_batch_adapter, rust_game_for_self_play


SCHEMA = "f4-cost-model-1"
FEATURE_WIDTH = 130
DEFAULT_ROWS = (1, 8, 27, 64, 128, 256)


def collect_corpus(games: int, seed: int, stride: int = 3) -> list[dict]:
    """Random-walk `games` games, keeping every `stride`-th decision.

    Rows carry exactly what the flat boundary needs: the encoder token list, the
    acting seat, and the legal action indices.
    """

    rng = random.Random(seed)
    corpus: list[dict] = []
    for index in range(games):
        game = rust_game_for_self_play(seed + index, index % 2)
        position = 0
        while not game.is_complete():
            legal = game.legal_action_indices()
            if not legal:
                break
            if position % stride == 0:
                corpus.append(
                    {
                        "tokens": game.encode(),
                        "actor": game.actor,
                        "legal": list(legal),
                    }
                )
            game.apply_index(rng.choice(legal))
            position += 1
    return corpus


def build_payload(rows: list[dict]) -> dict:
    """Pack rows into the byte layout ``eval.rs::FlatBatchBuilder`` produces.

    Kept byte-compatible on purpose: the adapter under measurement is the same
    object self-play calls, so a divergence here would measure a different
    boundary than the one being tuned.
    """

    token_offsets = bytearray()
    legal_offsets = bytearray()
    type_ids = bytearray()
    entity_ids = bytearray()
    aux_ids = bytearray()
    features = bytearray()
    actors = bytearray()
    legal_actions = bytearray()
    token_offsets += struct.pack("<i", 0)
    legal_offsets += struct.pack("<i", 0)
    tokens = 0
    legal_total = 0
    max_tokens = 0
    for row in rows:
        actors.append(int(row["actor"]))
        max_tokens = max(max_tokens, len(row["tokens"]))
        for type_id, entity_id, aux_id, values in row["tokens"]:
            type_ids.append(int(type_id))
            entity_ids += struct.pack("<h", int(entity_id))
            # Rust stores aux_id + 1, reserving 0 for "no aux entity".
            aux_ids += struct.pack("<h", int(aux_id) + 1)
            padded = list(values[:FEATURE_WIDTH])
            padded += [0.0] * (FEATURE_WIDTH - len(padded))
            features += struct.pack(f"<{FEATURE_WIDTH}f", *padded)
            tokens += 1
        token_offsets += struct.pack("<i", tokens)
        for action in row["legal"]:
            legal_actions += struct.pack("<H", int(action))
        legal_total += len(row["legal"])
        legal_offsets += struct.pack("<i", legal_total)
    return {
        "rows": len(rows),
        "tokens": tokens,
        "max_tokens": max_tokens,
        "feature_width": FEATURE_WIDTH,
        "token_offsets": token_offsets,
        "type_ids": type_ids,
        "entity_ids": entity_ids,
        "aux_ids": aux_ids,
        "features": features,
        "actors": actors,
        "legal_offsets": legal_offsets,
        "legal_actions": legal_actions,
    }


def token_length_buckets(corpus: list[dict], count: int) -> list[dict]:
    """Split the corpus into `count` equal-population token-length buckets.

    Batching within a bucket keeps padding low and makes the token axis vary
    independently of the row axis, which is what the fit needs.
    """

    ordered = sorted(corpus, key=lambda row: len(row["tokens"]))
    if not ordered:
        return []
    size = max(1, len(ordered) // count)
    buckets = []
    for index in range(count):
        start = index * size
        end = len(ordered) if index == count - 1 else (index + 1) * size
        members = ordered[start:end]
        if not members:
            continue
        lengths = [len(row["tokens"]) for row in members]
        buckets.append(
            {
                "label": f"tokens_{lengths[0]}_{lengths[-1]}",
                "min_tokens": lengths[0],
                "max_tokens": lengths[-1],
                "median_tokens": statistics.median(lengths),
                "rows": members,
            }
        )
    return buckets


def measure_cell(adapter, payload: dict, repetitions: int, warmup: int) -> dict:
    """Time one (rows, token-bucket) configuration.

    Host time is wall time in the adapter call; device time comes from the CUDA
    events the adapter recorded, drained after the reps so the measurement never
    synchronises the thing it measures mid-flight.
    """

    for _ in range(warmup):
        adapter(payload)
    adapter.drain_events()
    before = dict(adapter.total_metrics)
    host_ms = []
    for _ in range(repetitions):
        started = time.perf_counter()
        adapter(payload)
        host_ms.append((time.perf_counter() - started) * 1000.0)
    adapter.drain_events()
    delta = {
        key: float(adapter.total_metrics[key]) - float(before.get(key, 0.0))
        for key in adapter.total_metrics
    }
    per_call = {
        f"{key.removesuffix('_seconds')}_ms": value * 1000.0 / repetitions
        for key, value in delta.items()
        if key.endswith("_seconds")
    }
    cell = {
        "rows": payload["rows"],
        "tokens": payload["tokens"],
        "padded_tokens": payload["rows"] * payload["max_tokens"],
        "max_tokens": payload["max_tokens"],
        "repetitions": repetitions,
        "host_total_ms_mean": statistics.mean(host_ms),
        "host_total_ms_median": statistics.median(host_ms),
        "host_total_ms_min": min(host_ms),
    } | per_call
    device_parts = [
        value for key, value in cell.items() if key.startswith("device_") and key.endswith("_ms")
    ]
    if device_parts:
        cell["device_total_ms"] = sum(device_parts)
    return cell


def queue_depth_probe(adapter, payload: dict, depth: int) -> dict:
    """Decide whether the forward is launch-bound or device-bound.

    A CUDA event span around a forward is *not* device-busy time: if the host
    cannot enqueue kernels fast enough, the span includes the device's idle gaps
    and looks exactly like device work. This probe separates the two by
    enqueuing `depth` forwards back to back with no synchronisation in between
    and timing the host loop and the device span independently.

    * host loop ≈ device span  → the host could not get ahead.
    * host loop ≪ device span  → the host ran ahead of a saturated device.

    **This ratio alone cannot distinguish the two regimes**, and reading it as if
    it could is a trap: when the device is genuinely saturated the host *also*
    blocks, because the CUDA launch queue fills, so a device-bound run reports
    `host ≈ device` exactly like a launch-bound one. The ratio is only conclusive
    in the launch-bound direction when the queue never fills — i.e. at small
    models and narrow batches.

    The sound discriminator is `device_ms_per_row` across widths, which
    :func:`launch_bound_verdict` computes from a set of probes: launch-bound work
    has a large fixed cost per batch, so its per-row cost collapses as the batch
    widens; device-bound work pays per row and stays flat.
    """

    batch, _, _, _, _ = adapter.build_device_batch(payload)
    model = adapter.evaluator.model
    if not str(adapter.evaluator.device).startswith("cuda"):
        return {"rows": payload["rows"], "supported": False}
    with torch.no_grad():
        for _ in range(3):
            model(batch)
    torch.cuda.synchronize(adapter.evaluator.device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    host_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(depth):
            model(batch)
    host_ms = (time.perf_counter() - host_start) * 1000.0
    end.record()
    torch.cuda.synchronize(adapter.evaluator.device)
    device_ms = start.elapsed_time(end)
    rows = payload["rows"]
    return {
        "rows": rows,
        "supported": True,
        "depth": depth,
        "host_enqueue_ms_per_forward": host_ms / depth,
        "device_span_ms_per_forward": device_ms / depth,
        # 1.0 means the host never got ahead — of dispatch cost OR of a full
        # launch queue. See the docstring: not a regime verdict on its own.
        "host_pacing_ratio": host_ms / device_ms if device_ms > 0 else 0.0,
        "device_us_per_row": (device_ms / depth) * 1000.0 / max(1, rows),
    }


def launch_bound_verdict(probes: list[dict]) -> dict:
    """Decide the regime from how device cost per row scales with batch width.

    Launch-bound work carries a large fixed cost per batch, so widening the batch
    amortises it and the per-row cost collapses. Device-bound work pays per row,
    so the per-row cost barely moves. This is what the throughput levers hinge on:
    fewer/wider batches only help in the first regime, because batching changes
    the number of batches, never the number of rows.
    """

    usable = sorted(
        (probe for probe in probes if probe.get("supported")),
        key=lambda probe: probe["rows"],
    )
    if len(usable) < 2:
        return {"decided": False, "reason": "need probes at two or more widths"}
    narrow, wide = usable[0], usable[-1]
    drop = (
        narrow["device_us_per_row"] / wide["device_us_per_row"]
        if wide["device_us_per_row"] > 0
        else 0.0
    )
    if drop > 4.0:
        regime = "launch_bound"
    elif drop > 1.5:
        regime = "mixed"
    else:
        regime = "device_bound"
    return {
        "decided": True,
        "regime": regime,
        "narrow_rows": narrow["rows"],
        "wide_rows": wide["rows"],
        "device_us_per_row_narrow": narrow["device_us_per_row"],
        "device_us_per_row_wide": wide["device_us_per_row"],
        "per_row_cost_drop": drop,
        "implication": {
            "launch_bound": "wider/fewer batches pay: raise slots, batch leaves",
            "mixed": "some fixed cost remains; measure before widening further",
            "device_bound": "batching buys ~nothing; only fewer ROWS or a faster "
            "device helps",
        }[regime],
    }


def fit_cost_model(cells: list[dict], target: str) -> dict:
    """Least-squares fit of ``t = fixed + per_row * R + per_token * T``.

    Reported with R² and the residual scale so a bad fit is visible rather than
    quoted. Two-point extrapolations are exactly what Phase 0 exists to replace.
    """

    usable = [cell for cell in cells if target in cell]
    if len(usable) < 3:
        return {"target": target, "fitted": False, "reason": "need at least 3 cells"}
    design = torch.tensor(
        [[1.0, float(cell["rows"]), float(cell["padded_tokens"])] for cell in usable],
        dtype=torch.float64,
    )
    observed = torch.tensor(
        [[float(cell[target])] for cell in usable], dtype=torch.float64
    )
    solution = torch.linalg.lstsq(design, observed).solution.flatten()
    predicted = (design @ solution.unsqueeze(1)).flatten()
    residual = observed.flatten() - predicted
    total = observed.flatten() - observed.mean()
    r_squared = float(
        1.0 - (residual.pow(2).sum() / total.pow(2).sum()).item()
        if float(total.pow(2).sum()) > 0
        else 0.0
    )
    return {
        "target": target,
        "fitted": True,
        "cells": len(usable),
        "fixed_ms": float(solution[0]),
        "per_row_ms": float(solution[1]),
        "per_padded_token_ms": float(solution[2]),
        "r_squared": r_squared,
        "residual_ms_max": float(residual.abs().max()),
    }


def aggregate_passes(passes: list[list[dict]]) -> list[dict]:
    """Median each cell across passes, and record the observed spread.

    A single pass over the grid drifts: GPU clocks, other tenants on the device
    and thermal state move the per-row terms by more than the effect being
    measured. Fitting one pass produces a confident-looking model that the next
    pass contradicts, so the fit runs on medians and the spread is reported
    beside them.
    """

    grouped: dict[tuple, list[dict]] = {}
    for cells in passes:
        for cell in cells:
            grouped.setdefault((cell["bucket"], cell["rows"]), []).append(cell)
    merged = []
    for key, group in grouped.items():
        first = group[0]
        cell = {
            name: value
            for name, value in first.items()
            if not isinstance(value, (int, float)) or isinstance(value, bool)
        }
        cell |= {"bucket": key[0], "rows": key[1], "passes": len(group)}
        for name, value in first.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            observations = [float(item[name]) for item in group if name in item]
            cell[name] = statistics.median(observations)
            if len(observations) > 1 and cell[name] != 0:
                cell[f"{name}_spread"] = (
                    max(observations) - min(observations)
                ) / abs(cell[name])
        merged.append(cell)
    return sorted(merged, key=lambda cell: (cell["bucket"], cell["rows"]))


def run(args) -> dict:
    corpus = collect_corpus(args.corpus_games, args.seed, stride=args.corpus_stride)
    if not corpus:
        raise RuntimeError("corpus collection produced no positions")
    buckets = token_length_buckets(corpus, args.token_buckets)
    evaluator = load_evaluator(str(args.checkpoint), args.device)
    evaluator.max_batch = max(args.rows)
    adapter = rust_flat_batch_adapter(evaluator, cuda_events=args.cuda_events)
    rng = random.Random(args.seed)

    all_passes = []
    for pass_index in range(args.passes):
        pass_cells = []
        for bucket in buckets:
            for rows in sorted(args.rows):
                members = bucket["rows"]
                sample = [rng.choice(members) for _ in range(rows)]
                payload = build_payload(sample)
                cell = measure_cell(adapter, payload, args.repetitions, args.warmup)
                cell |= {
                    "bucket": bucket["label"],
                    "bucket_median_tokens": bucket["median_tokens"],
                    "pass": pass_index,
                }
                pass_cells.append(cell)
                print(
                    f"cost-model: pass={pass_index} rows={rows:>4} "
                    f"bucket={bucket['label']:<18} "
                    f"host={cell['host_total_ms_mean']:.3f} ms "
                    f"device_forward={cell.get('device_forward_ms', 0.0):.3f} ms",
                    flush=True,
                )
        all_passes.append(pass_cells)
    cells = aggregate_passes(all_passes)
    probes = []
    if buckets:
        widest = max(buckets, key=lambda bucket: bucket["median_tokens"])
        for rows in sorted(args.rows):
            sample = [rng.choice(widest["rows"]) for _ in range(rows)]
            probe = queue_depth_probe(
                adapter, build_payload(sample), args.queue_depth
            )
            probes.append(probe)
            if probe["supported"]:
                print(
                    f"queue-probe: rows={rows:>4} "
                    f"host={probe['host_enqueue_ms_per_forward']:.3f} ms "
                    f"device_span={probe['device_span_ms_per_forward']:.3f} ms "
                    f"device={probe['device_us_per_row']:.1f} us/row",
                    flush=True,
                )
        verdict = launch_bound_verdict(probes)
        if verdict.get("decided"):
            print(
                f"queue-probe: {verdict['regime'].upper()} -- device cost per row "
                f"{verdict['device_us_per_row_narrow']:.1f} -> "
                f"{verdict['device_us_per_row_wide']:.1f} us "
                f"({verdict['per_row_cost_drop']:.2f}x drop). "
                f"{verdict['implication']}",
                flush=True,
            )

    summary = {
        "schema": SCHEMA,
        "device": args.device,
        "checkpoint": str(args.checkpoint),
        "torch_version": torch.__version__,
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "corpus_positions": len(corpus),
        "passes": args.passes,
        "repetitions": args.repetitions,
        "buckets": [
            {key: value for key, value in bucket.items() if key != "rows"}
            for bucket in buckets
        ],
        "cells": cells,
        "queue_depth_probes": probes,
        "regime": launch_bound_verdict(probes),
        "fits": [
            fit_cost_model(cells, target)
            for target in (
                "host_total_ms_mean",
                "device_forward_ms",
                "device_total_ms",
                "forward_ms",
                "gather_ms",
                "tensor_ms",
            )
            if any(target in cell for cell in cells)
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cost_model.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=list(DEFAULT_ROWS),
        help="batch widths to measure (default: 1 8 27 64 128 256)",
    )
    parser.add_argument("--token-buckets", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument(
        "--passes",
        type=int,
        default=3,
        help="times the whole grid is repeated; cells are fitted on the median "
        "across passes, because one pass drifts more than the effects measured",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--corpus-games", type=int, default=8)
    parser.add_argument("--corpus-stride", type=int, default=3)
    parser.add_argument(
        "--queue-depth",
        type=int,
        default=50,
        help="forwards enqueued back to back in the launch-bound probe",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--no-cuda-events",
        dest="cuda_events",
        action="store_false",
        help="skip CUDA-event timing (host timings only)",
    )
    parser.set_defaults(cuda_events=True)
    args = parser.parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "fits": summary["fits"],
                "queue_depth_probes": summary["queue_depth_probes"],
                "regime": summary["regime"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
