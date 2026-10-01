"""CUDA-only production-geometry sweep for S2 inflight games and Rust workers."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
from pathlib import Path
from typing import Optional, Sequence

import torch

from games.welcome_to import self_play
from games.welcome_to import train


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile needs at least one value")
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


class _GpuSampler:
    """Continuous in-process NVML sampler; unavailable is never reported as 0."""

    def __init__(self, device: str, hz: float) -> None:
        self.device = torch.device(device)
        self.interval = 1.0 / hz if hz > 0 else 0.0
        self.samples: list[dict[str, Optional[float]]] = []
        self.available = False
        self.error: Optional[str] = None
        self._pynvml = None
        self._handle = None
        self._stop: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _resolve_handle(pynvml, index: int):
        """The NVML handle for CUDA device ``index``, or an exception saying why not.

        ⚠ **An NVML index is not a CUDA index.** NVML always enumerates by PCI
        bus id; CUDA orders by compute capability unless ``CUDA_DEVICE_ORDER``
        says otherwise, and ``CUDA_VISIBLE_DEVICES`` remaps CUDA's indices while
        leaving NVML's alone. So ``nvmlDeviceGetHandleByIndex`` is only the same
        device by coincidence, and on a multi-GPU host it silently samples a
        different card while still reporting ``nvml_available: true`` -- the one
        failure mode that makes a sweep's power and utilisation numbers wrong
        rather than missing.

        ⚠ ``str(torch_properties.uuid)`` is the bare UUID; NVML wants it with a
        ``GPU-`` prefix. Measured on this repository's machine: the bare form
        raises ``NVMLError_NotFound``, so the previous spelling took the index
        fallback **every time** rather than as a rare last resort.

        The handle is verified by reading its UUID back. Index is used only when
        exactly one GPU is visible, where there is nothing to confuse it with.
        """
        properties = torch.cuda.get_device_properties(index)
        wanted = str(getattr(properties, "uuid", "") or "")
        if wanted:
            if not wanted.lower().startswith("gpu-"):
                wanted = f"GPU-{wanted}"
            handle = pynvml.nvmlDeviceGetHandleByUUID(wanted.encode())
            found = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(found, bytes):
                found = found.decode()
            if found.lower() != wanted.lower():
                raise RuntimeError(
                    f"NVML resolved {wanted} to {found}; refusing to sample it"
                )
            return handle
        if pynvml.nvmlDeviceGetCount() != 1:
            raise RuntimeError(
                "torch exposed no device UUID and this host has more than one "
                "GPU; an NVML index would not reliably name the CUDA device"
            )
        return pynvml.nvmlDeviceGetHandleByIndex(0)

    def __enter__(self):
        if self.interval <= 0:
            self.error = "monitor frequency is disabled"
            return self
        try:
            import pynvml

            pynvml.nvmlInit()
            index = self.device.index
            if index is None:
                index = torch.cuda.current_device()
            self._handle = self._resolve_handle(pynvml, index)
            self._pynvml = pynvml
            self.available = True
        except Exception as error:  # pragma: no cover - hardware/driver specific
            self.error = f"{type(error).__name__}: {error}"
            return self
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        assert self._stop is not None and self._pynvml is not None
        pynvml = self._pynvml
        while not self._stop.is_set():
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)

                def optional(call) -> Optional[float]:
                    try:
                        return float(call())
                    except Exception:
                        return None

                self.samples.append(
                    {
                        "gpu_utilization_percent": float(utilization.gpu),
                        "memory_utilization_percent": float(utilization.memory),
                        "memory_used_mib": float(memory.used) / 2**20,
                        "power_w": (
                            value / 1000.0
                            if (
                                value := optional(
                                    lambda: pynvml.nvmlDeviceGetPowerUsage(self._handle)
                                )
                            )
                            is not None
                            else None
                        ),
                        "temperature_c": optional(
                            lambda: pynvml.nvmlDeviceGetTemperature(
                                self._handle, pynvml.NVML_TEMPERATURE_GPU
                            )
                        ),
                        "sm_clock_mhz": optional(
                            lambda: pynvml.nvmlDeviceGetClockInfo(
                                self._handle, pynvml.NVML_CLOCK_SM
                            )
                        ),
                    }
                )
            except Exception as error:  # pragma: no cover - driver lost mid-run
                self.error = f"{type(error).__name__}: {error}"
                self.available = False
                return
            if self._stop.wait(self.interval):
                return

    def __exit__(self, *_exc) -> bool:
        alive = False
        if self._thread is not None and self._stop is not None:
            self._stop.set()
            self._thread.join(timeout=5.0)
            alive = self._thread.is_alive()
        if self._pynvml is not None and not alive:
            # Only shut NVML down once the sampler has actually stopped calling
            # it; tearing the library down underneath a live thread is a crash,
            # not a leak, and leaking one handle for the rest of a sweep process
            # is the cheaper failure.
            try:
                self._pynvml.nvmlShutdown()
            except Exception:  # pragma: no cover
                pass
        if alive:
            self.error = self.error or "NVML sampler thread did not stop in 5s"
        return False

    def summary(self) -> dict[str, object]:
        utilization = [
            float(row["gpu_utilization_percent"])
            for row in self.samples
            if row["gpu_utilization_percent"] is not None
        ]
        result: dict[str, object] = {
            "nvml_available": bool(self.available and self.samples),
            "nvml_error": self.error,
            "nvml_samples": len(self.samples),
            "nvml_utilization_mean": (
                statistics.fmean(utilization) if utilization else None
            ),
            "nvml_utilization_p95": (
                _percentile(utilization, 0.95) if utilization else None
            ),
        }
        for name in (
            "memory_utilization_percent",
            "memory_used_mib",
            "power_w",
            "temperature_c",
            "sm_clock_mhz",
        ):
            values = [float(row[name]) for row in self.samples if row[name] is not None]
            result[f"nvml_{name}_mean"] = (
                statistics.fmean(values) if values else None
            )
            result[f"nvml_{name}_max"] = max(values) if values else None
        return result


def sweep(
    checkpoint: str | Path,
    *,
    inflight: Sequence[int] = (8, 16, 32),
    scheduler_workers: Sequence[int] = (8,),
    games: int = 32,
    simulations: int = 200,
    seed: int = 15_000,
    device: str = "cuda",
    monitor_hz: float = 10.0,
    require_gpu_telemetry: bool = False,
) -> list[dict]:
    """Run identical games at each scheduler width and retain no corpus."""
    if device != "cuda":
        raise ValueError("the S2 in-flight sweep is intentionally CUDA-only")
    if not math.isfinite(monitor_hz) or monitor_hz < 0:
        raise ValueError("monitor_hz must be finite and non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if games <= 0 or simulations <= 0 or not inflight or not scheduler_workers:
        raise ValueError(
            "games, simulations, in-flight arms, and worker arms must be positive"
        )
    if any(width <= 0 or width > games for width in inflight):
        raise ValueError("each in-flight arm must be in [1, games]")
    if any(workers <= 0 for workers in scheduler_workers):
        raise ValueError("each scheduler-worker arm must be positive")

    net = train.load(checkpoint, device).eval()
    reference: Optional[dict[int, str]] = None
    results: list[dict] = []
    for workers, width in (
        (workers, width) for workers in scheduler_workers for width in inflight
    ):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with _GpuSampler(device, monitor_hz) as monitor:
            trajectories, metrics = self_play.generate(
                net,
                config=self_play.SelfPlayConfig(
                    games=games,
                    inflight=width,
                    max_batch=width,
                    scheduler_workers=workers,
                    seed=seed,
                ),
                search_config=self_play.default_search_config(simulations),
                device=device,
                cuda_events=True,
            )
        torch.cuda.synchronize()
        gpu_metrics = monitor.summary()
        if require_gpu_telemetry and not gpu_metrics["nvml_available"]:
            raise RuntimeError(
                f"GPU telemetry was required but unavailable: {gpu_metrics['nvml_error']}"
            )
        fingerprints = {
            trajectory.seed: trajectory.to_json() for trajectory in trajectories
        }
        if reference is None:
            reference = fingerprints
            agreement = 1.0
            mismatches: list[int] = []
        else:
            mismatches = sorted(
                game_seed
                for game_seed in reference
                if fingerprints.get(game_seed) != reference[game_seed]
            )
            agreement = (games - len(mismatches)) / games
        row = {
            "inflight": float(width),
            "max_batch": float(width),
            "scheduler_workers": float(workers),
            "simulations": float(simulations),
            "trajectory_agreement": agreement,
            "trajectory_mismatches": mismatches,
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            **gpu_metrics,
            **metrics,
        }
        results.append(row)
        print(
            f"workers={workers:>2} inflight={width:>3}  "
            f"{metrics['games_per_hour']:.1f} games/h  "
            f"{metrics['evaluator_rows_per_second']:.1f} rows/s  "
            f"batch={metrics['mean_batch']:.2f} "
            f"p90={metrics['batch_p90']:.0f}  "
            f"VRAM={row['cuda_peak_allocated_mib']:.0f} MiB  "
            f"agreement={agreement:.3f}",
            flush=True,
        )
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inflight", default="8,16,32")
    parser.add_argument("--scheduler-workers", default="8")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=15_000)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--monitor-hz", type=float, default=10.0)
    parser.add_argument("--require-gpu-telemetry", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    widths = tuple(int(value) for value in args.inflight.split(",") if value)
    worker_counts = tuple(
        int(value) for value in args.scheduler_workers.split(",") if value
    )
    results = sweep(
        args.checkpoint,
        inflight=widths,
        scheduler_workers=worker_counts,
        games=args.games,
        simulations=args.simulations,
        seed=args.seed,
        device=args.device,
        monitor_hz=args.monitor_hz,
        require_gpu_telemetry=args.require_gpu_telemetry,
    )
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(results, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        print(f"wrote sweep to {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
