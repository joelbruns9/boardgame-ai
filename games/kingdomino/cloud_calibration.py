"""Cloud calibration runner for Kingdomino training hardware.

Runs the Phase 3 benchmark sequence from CLOUD_RUN.md and writes:

* results.csv: one structured row per benchmark command
* summary.json: selected settings and hardware metadata
* summary.md: human-readable review checklist
* logs/*.log and logs/*.cmd: raw command output and exact commands

The runner intentionally shells out to the existing benchmark modules so their
behavior remains the single source of truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


CSV_FIELDS = [
    "timestamp",
    "phase",
    "test",
    "preset",
    "status",
    "elapsed_sec",
    "device",
    "channels",
    "blocks",
    "batch_size",
    "amp_inference",
    "compile",
    "channels_last",
    "batch_slots",
    "leaf_batch",
    "sims",
    "games",
    "game_cpus",
    "solver_cpus",
    "async_solve",
    "exact_endgame_max_secs",
    "full_search_fraction",
    "games_per_sec",
    "recorded_positions_per_sec",
    "recorded_positions_per_game",
    "total_evals_per_sec",
    "requests_per_sec",
    "mean_batch",
    "fill_ratio",
    "max_batch_seen",
    "eval_h2d_sec",
    "eval_forward_sec",
    "eval_readback_sec",
    "exact_solve_count",
    "exact_tree_solve_count",
    "exact_cache_hit_count",
    "exact_fallback_count",
    "exact_attempt_deck4_initial_count",
    "exact_attempt_deck4_retry_count",
    "exact_attempt_deck0_count",
    "exact_fallback_deck4_initial_count",
    "exact_fallback_deck4_retry_count",
    "exact_fallback_deck0_count",
    "exact_solver_secs",
    "fast_move_count",
    "full_move_count",
    "recorded_fast_move_count",
    "recorded_full_move_count",
    "exact_recorded_move_count",
    "forward_peak_evals_per_sec",
    "forward_peak_batch",
    "compile_verdict",
    "double_buffer_verdict",
    "log_path",
]


@dataclass
class RunContext:
    args: argparse.Namespace
    out: Path
    logs: Path
    rows: list[dict[str, Any]] = field(default_factory=list)
    hardware: dict[str, Any] = field(default_factory=dict)

    def command_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _format_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _write_csv(ctx: RunContext) -> None:
    path = ctx.out / "results.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in ctx.rows:
            writer.writerow({k: _format_value(row.get(k)) for k in CSV_FIELDS})


def _append_row(ctx: RunContext, row: dict[str, Any]) -> None:
    base = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preset": ctx.args.preset,
        "device": ctx.args.device,
        "status": "ok",
    }
    base.update(row)
    ctx.rows.append(base)
    _write_csv(ctx)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _run_command(
    ctx: RunContext,
    name: str,
    cmd: list[str],
    *,
    env_extra: dict[str, str] | None = None,
) -> tuple[int, float, str, Path]:
    log_path = ctx.logs / f"{_slug(name)}.log"
    cmd_path = ctx.logs / f"{_slug(name)}.cmd"
    env = ctx.command_env()
    if env_extra:
        env.update(env_extra)
    cmd_path.write_text(" ".join(_quote_arg(x) for x in cmd) + "\n", encoding="utf-8")
    print(f"\n==> {name}", flush=True)
    print("    " + " ".join(_quote_arg(x) for x in cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(Path.cwd()),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    elapsed = time.perf_counter() - t0
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    print(proc.stdout[-4000:], end="" if proc.stdout.endswith("\n") else "\n")
    if proc.returncode != 0:
        print(f"    command failed with exit code {proc.returncode}", flush=True)
    return proc.returncode, elapsed, proc.stdout, log_path


def _quote_arg(arg: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=,+-]+", arg):
        return arg
    return json.dumps(arg)


def _hardware_snapshot(device: str) -> dict[str, Any]:
    data: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        import torch

        data["torch"] = torch.__version__
        data["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            data["gpu"] = torch.cuda.get_device_name(0)
            data["capability"] = list(torch.cuda.get_device_capability(0))
            data["arch_list"] = list(torch.cuda.get_arch_list())
    except Exception as exc:  # pragma: no cover - hardware dependent
        data["torch_error"] = str(exc)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        data["nvidia_smi_gpu"] = out
    except Exception as exc:  # pragma: no cover - hardware dependent
        data["nvidia_smi_error"] = str(exc)
    return data


def _parse_forward(text: str) -> dict[str, Any]:
    m = re.search(r"peak forward throughput\s*:\s*([0-9,]+)\s+evals/s at batch\s+(\d+)", text)
    if not m:
        return {}
    return {
        "forward_peak_evals_per_sec": float(m.group(1).replace(",", "")),
        "forward_peak_batch": int(m.group(2)),
    }


def _parse_compile_verdict(text: str) -> str:
    if "USE --compile" in text and "DO NOT USE --compile" not in text:
        return "use"
    if "DO NOT USE --compile" in text:
        return "do_not_use"
    return ""


def _parse_double_buffer_verdict(text: str) -> str:
    if "USE --double_buffer" in text and "DO NOT USE --double_buffer" not in text:
        return "use"
    if "DO NOT USE --double_buffer" in text:
        return "do_not_use"
    return ""


def _read_training_row(run_dir: Path) -> dict[str, Any]:
    log_path = run_dir / "training_log.jsonl"
    if not log_path.exists():
        return {}
    last: dict[str, Any] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            last = json.loads(line)
    return last


def _selfplay_metrics(row: dict[str, Any]) -> dict[str, Any]:
    played = float(row.get("selfplay_games", row.get("games_per_iteration", 0)) or 0)
    elapsed = float(row.get("selfplay_sec", row.get("elapsed_sec", 0)) or 0)
    buffer_size = float(row.get("buffer_size", 0) or 0)
    games_per_sec = row.get("games_per_sec")
    requests_per_sec = row.get("requests_per_sec")
    out = {
        "games_per_sec": games_per_sec if games_per_sec is not None else (
            played / elapsed if elapsed > 0 else None),
        "recorded_positions_per_game": buffer_size / played if played > 0 else None,
        "recorded_positions_per_sec": (
            buffer_size / elapsed if elapsed > 0 else (
                (buffer_size / played) * float(games_per_sec)
                if played > 0 and games_per_sec is not None else None)),
        "total_evals_per_sec": requests_per_sec,
    }
    for key in [
        "requests_per_sec",
        "mean_batch",
        "fill_ratio",
        "max_batch_seen",
        "eval_h2d_sec",
        "eval_forward_sec",
        "eval_readback_sec",
        "exact_solve_count",
        "exact_tree_solve_count",
        "exact_cache_hit_count",
        "exact_fallback_count",
        "exact_attempt_deck4_initial_count",
        "exact_attempt_deck4_retry_count",
        "exact_attempt_deck0_count",
        "exact_fallback_deck4_initial_count",
        "exact_fallback_deck4_retry_count",
        "exact_fallback_deck0_count",
        "exact_solver_secs",
        "fast_move_count",
        "full_move_count",
        "recorded_fast_move_count",
        "recorded_full_move_count",
        "exact_recorded_move_count",
    ]:
        if key in row:
            out[key] = row[key]
    return out


def run_forward(ctx: RunContext) -> None:
    for channels in ctx.args.channels:
        for amp in [False, True]:
            name = f"forward_{channels}x{ctx.args.blocks}_{'amp' if amp else 'eager'}"
            cmd = [
                sys.executable,
                "-m",
                "games.kingdomino.forward_bench",
                "--device",
                ctx.args.device,
                "--channels",
                str(channels),
                "--blocks",
                str(ctx.args.blocks),
                "--batches",
                ctx.args.forward_batches,
                "--cudnn_benchmark",
                "--legal_counts",
                ctx.args.legal_counts,
            ]
            if amp:
                cmd.append("--amp_inference")
            rc, elapsed, text, log_path = _run_command(ctx, name, cmd)
            parsed = _parse_forward(text)
            _append_row(ctx, {
                "phase": "3.1_forward",
                "test": name,
                "status": "ok" if rc == 0 else "failed",
                "elapsed_sec": elapsed,
                "channels": channels,
                "blocks": ctx.args.blocks,
                "amp_inference": amp,
                "log_path": str(log_path),
                **parsed,
            })


def run_compile(ctx: RunContext) -> None:
    name = f"compile_{ctx.args.primary_channels}x{ctx.args.blocks}"
    cmd = [
        sys.executable,
        "-m",
        "games.kingdomino.bench_compile",
        "--device",
        ctx.args.device,
        "--sims",
        str(ctx.args.sims),
        "--games",
        str(ctx.args.compile_games),
        "--channels",
        str(ctx.args.primary_channels),
        "--blocks",
        str(ctx.args.blocks),
        "--batch_slots",
        str(ctx.args.default_batch_slots),
        "--leaf_batch",
        str(ctx.args.leaf_batch),
    ]
    rc, elapsed, text, log_path = _run_command(
        ctx, name, cmd, env_extra={"TORCH_LOGS": "recompiles"})
    _append_row(ctx, {
        "phase": "3.7_compile",
        "test": name,
        "status": "ok" if rc == 0 else "failed",
        "elapsed_sec": elapsed,
        "channels": ctx.args.primary_channels,
        "blocks": ctx.args.blocks,
        "batch_slots": ctx.args.default_batch_slots,
        "leaf_batch": ctx.args.leaf_batch,
        "sims": ctx.args.sims,
        "games": ctx.args.compile_games,
        "compile_verdict": _parse_compile_verdict(text),
        "log_path": str(log_path),
    })


def run_double_buffer(ctx: RunContext) -> None:
    name = f"doublebuffer_{ctx.args.primary_channels}x{ctx.args.blocks}"
    cmd = [
        sys.executable,
        "-m",
        "games.kingdomino.bench_doublebuffer",
        "--device",
        ctx.args.device,
        "--sims",
        str(ctx.args.sims),
        "--games",
        str(ctx.args.doublebuffer_games),
        "--channels",
        str(ctx.args.primary_channels),
        "--blocks",
        str(ctx.args.blocks),
        "--batch_slots",
        str(ctx.args.default_batch_slots),
        "--leaf_batch",
        str(ctx.args.leaf_batch),
    ]
    rc, elapsed, text, log_path = _run_command(ctx, name, cmd)
    _append_row(ctx, {
        "phase": "3.7_double_buffer",
        "test": name,
        "status": "ok" if rc == 0 else "failed",
        "elapsed_sec": elapsed,
        "channels": ctx.args.primary_channels,
        "blocks": ctx.args.blocks,
        "batch_slots": ctx.args.default_batch_slots,
        "leaf_batch": ctx.args.leaf_batch,
        "sims": ctx.args.sims,
        "games": ctx.args.doublebuffer_games,
        "double_buffer_verdict": _parse_double_buffer_verdict(text),
        "log_path": str(log_path),
    })


def run_selfplay_case(
    ctx: RunContext,
    *,
    phase: str,
    name: str,
    batch_slots: int | None = None,
    game_cpus: int | None = None,
    exact_secs: float = 0.0,
    full_search_fraction: float | None = None,
    profile_eval_timing: bool = False,
    playout_cap_randomization: bool = False,
) -> None:
    batch_slots = int(batch_slots if batch_slots is not None else ctx.args.default_batch_slots)
    full_search_fraction = float(
        ctx.args.full_search_fraction
        if full_search_fraction is None
        else full_search_fraction)
    run_dir = ctx.out / "selfplay" / _slug(name)
    cmd = [
        sys.executable,
        "-m",
        "games.kingdomino.self_play",
        "--engine",
        "batched_open_loop",
        "--device",
        ctx.args.device,
        "--channels",
        str(ctx.args.primary_channels),
        "--blocks",
        str(ctx.args.blocks),
        "--leaf_batch",
        str(ctx.args.leaf_batch),
        "--batch_slots",
        str(batch_slots),
        "--sims",
        str(ctx.args.sims),
        "--iterations",
        "1",
        "--games_per_iter",
        str(ctx.args.selfplay_games),
        "--train_steps",
        "0",
        "--min_buffer",
        "999999999",
        "--exact_endgame_max_secs",
        str(exact_secs),
        "--full_search_fraction",
        str(full_search_fraction),
        "--benchmark_every",
        "0",
        "--elo_every",
        "0",
        "--checkpoint_dir",
        str(run_dir),
        "--seed",
        "0",
    ]
    if profile_eval_timing:
        cmd.append("--profile_eval_timing")
    if playout_cap_randomization:
        cmd.extend([
            "--playout_cap_randomization",
            "--fast_move_sims",
            "100",
            "--fast_move_dirichlet_epsilon",
            "0.0",
            "--fast_move_temp_moves",
            "0",
            "--policy_target_pruning",
        ])
    if game_cpus is not None:
        cmd.extend(["--async_solve", "--game_cpus", str(game_cpus)])
    elif exact_secs > 0:
        cmd.append("--async_solve")
    rc, elapsed, _text, log_path = _run_command(ctx, name, cmd)
    train_row = _read_training_row(run_dir)
    metrics = _selfplay_metrics(train_row)
    _append_row(ctx, {
        "phase": phase,
        "test": name,
        "status": "ok" if rc == 0 else "failed",
        "elapsed_sec": elapsed,
        "channels": ctx.args.primary_channels,
        "blocks": ctx.args.blocks,
        "batch_slots": batch_slots,
        "leaf_batch": ctx.args.leaf_batch,
        "sims": ctx.args.sims,
        "games": ctx.args.selfplay_games,
        "game_cpus": game_cpus,
        "async_solve": bool(game_cpus is not None or exact_secs > 0),
        "exact_endgame_max_secs": exact_secs,
        "full_search_fraction": full_search_fraction,
        "log_path": str(log_path),
        **metrics,
    })


def run_batch_slots(ctx: RunContext) -> None:
    for batch_slots in ctx.args.batch_slots:
        run_selfplay_case(
            ctx,
            phase="3.2_batch_slots",
            name=f"batch_slots_{batch_slots}_{ctx.args.primary_channels}x{ctx.args.blocks}",
            batch_slots=batch_slots,
            exact_secs=0.0,
            full_search_fraction=1.0,
            profile_eval_timing=True,
        )


def run_full_sweeps(ctx: RunContext) -> None:
    for game_cpus in ctx.args.game_cpus:
        run_selfplay_case(
            ctx,
            phase="3.3_cpu_split",
            name=f"game_cpus_{game_cpus}",
            game_cpus=game_cpus,
            exact_secs=ctx.args.cpu_split_exact_secs,
            profile_eval_timing=False,
        )
    for exact_secs in ctx.args.exact_secs:
        run_selfplay_case(
            ctx,
            phase="3.4_exact_cap",
            name=f"exact_cap_{exact_secs:g}s",
            game_cpus=ctx.args.selected_game_cpus,
            exact_secs=exact_secs,
            profile_eval_timing=False,
        )
    for fraction in ctx.args.full_search_fractions:
        run_selfplay_case(
            ctx,
            phase="3.5_full_search_fraction",
            name=f"full_search_{fraction:g}",
            game_cpus=ctx.args.selected_game_cpus,
            exact_secs=ctx.args.selected_exact_secs,
            full_search_fraction=fraction,
            profile_eval_timing=False,
            playout_cap_randomization=True,
        )


def _best_row(rows: Iterable[dict[str, Any]], key: str) -> dict[str, Any] | None:
    vals = [r for r in rows if r.get("status") == "ok" and r.get(key) not in ("", None)]
    if not vals:
        return None
    return max(vals, key=lambda r: float(r[key]))


def write_summary(ctx: RunContext) -> None:
    forward_best = _best_row(ctx.rows, "forward_peak_evals_per_sec")
    batch_best = _best_row(
        [r for r in ctx.rows if r.get("phase") == "3.2_batch_slots"],
        "games_per_sec")
    compile_rows = [r for r in ctx.rows if r.get("compile_verdict")]
    db_rows = [r for r in ctx.rows if r.get("double_buffer_verdict")]
    summary = {
        "hardware": ctx.hardware,
        "preset": ctx.args.preset,
        "rows": len(ctx.rows),
        "best_forward": forward_best,
        "best_batch_slots": batch_best,
        "compile_verdict": compile_rows[-1].get("compile_verdict") if compile_rows else "",
        "double_buffer_verdict": db_rows[-1].get("double_buffer_verdict") if db_rows else "",
    }
    (ctx.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Cloud Calibration Summary",
        "",
        f"Preset: `{ctx.args.preset}`",
        f"Rows: `{len(ctx.rows)}`",
        "",
    ]
    if forward_best:
        lines.append(
            f"- Best forward: `{forward_best.get('channels')}x{forward_best.get('blocks')}` "
            f"amp=`{forward_best.get('amp_inference')}` at "
            f"`{forward_best.get('forward_peak_evals_per_sec')}` evals/sec "
            f"(batch `{forward_best.get('forward_peak_batch')}`).")
    if batch_best:
        lines.append(
            f"- Best batch_slots: `{batch_best.get('batch_slots')}` at "
            f"`{batch_best.get('games_per_sec')}` games/sec, "
            f"mean_batch `{batch_best.get('mean_batch')}`.")
    if compile_rows:
        lines.append(f"- Compile verdict: `{compile_rows[-1].get('compile_verdict')}`.")
    if db_rows:
        lines.append(f"- Double-buffer verdict: `{db_rows[-1].get('double_buffer_verdict')}`.")
    lines.extend([
        "",
        "Raw logs are in `logs/`; structured rows are in `results.csv`.",
        "",
    ])
    (ctx.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["bootstrap", "full"], default="bootstrap")
    p.add_argument("--out", default="runs/kingdomino/cloud_calibration")
    p.add_argument("--device", default="cuda")
    p.add_argument("--channels", default="80,96",
                   help="comma-list of channel counts for forward ceiling tests")
    p.add_argument("--primary_channels", type=int, default=80,
                   help="model width for self-play and feature tests")
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--forward_batches",
                   default="64,96,128,160,192,224,256,320,384,512")
    p.add_argument("--legal_counts", default="20,50,100")
    p.add_argument("--batch_slots", default="32,48,64,80,96,128")
    p.add_argument("--leaf_batch", type=int, default=6)
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--selfplay_games", type=int, default=30)
    p.add_argument("--compile_games", type=int, default=20)
    p.add_argument("--doublebuffer_games", type=int, default=50)
    p.add_argument("--default_batch_slots", type=int, default=32)
    p.add_argument("--game_cpus", default="2,4,6,8")
    p.add_argument("--selected_game_cpus", type=int, default=2)
    p.add_argument("--cpu_split_exact_secs", type=float, default=3.0)
    p.add_argument("--exact_secs", default="3,5,7,10")
    p.add_argument("--selected_exact_secs", type=float, default=3.0)
    p.add_argument("--full_search_fractions", default="0.25,0.35,0.45")
    p.add_argument("--full_search_fraction", type=float, default=1.0,
                   help="default fraction for non-fraction self-play rows")
    p.add_argument("--skip_forward", action="store_true")
    p.add_argument("--skip_compile", action="store_true")
    p.add_argument("--skip_doublebuffer", action="store_true")
    p.add_argument("--skip_batch_slots", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.channels = _parse_csv_ints(args.channels)
    args.batch_slots = _parse_csv_ints(args.batch_slots)
    args.game_cpus = _parse_csv_ints(args.game_cpus)
    args.exact_secs = _parse_csv_floats(args.exact_secs)
    args.full_search_fractions = _parse_csv_floats(args.full_search_fractions)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(args=args, out=out, logs=logs)
    ctx.hardware = _hardware_snapshot(args.device)
    (out / "hardware.json").write_text(
        json.dumps(ctx.hardware, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Writing calibration output to {out}")
    _append_row(ctx, {
        "phase": "hardware",
        "test": "hardware",
        "status": "ok",
        "elapsed_sec": 0.0,
    })

    if not args.skip_forward:
        run_forward(ctx)
    if not args.skip_compile:
        run_compile(ctx)
    if not args.skip_doublebuffer:
        run_double_buffer(ctx)
    if not args.skip_batch_slots:
        run_batch_slots(ctx)
    if args.preset == "full":
        run_full_sweeps(ctx)

    write_summary(ctx)
    print(f"\nSummary written to {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
