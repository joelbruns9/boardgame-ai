"""
profile_forward.py - PyTorch profiler for Kingdomino inference forward passes.

This is a diagnostic companion to forward_bench.py. It answers "what is inside
the forward time?" rather than "how many evals/s can this GPU sustain?"

Typical first run:

  python -m games.kingdomino.profile_forward --device cuda --channels 48 \
      --blocks 6 --batch 137 --mode both --cudnn_benchmark

Optional TensorBoard trace:

  python -m games.kingdomino.profile_forward --trace_dir runs/prof_forward

Then view:

  tensorboard --logdir runs/prof_forward
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from games.kingdomino.encoder import encode_state
from games.kingdomino.game import GameState
from games.kingdomino.network import KingdominoNet


NUM_JOINT_ACTIONS = 3390


def _input_shapes():
    mb, ob, flat = encode_state(GameState.new(seed=0), 0)
    return mb.shape, ob.shape, flat.shape[0]


def _make_inputs(
    *,
    batch: int,
    device: str,
    legal_count: int,
    seed: int,
    channels_last: bool,
):
    mb_shape, ob_shape, flat_size = _input_shapes()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mb = torch.rand(batch, *mb_shape, generator=gen).to(device)
    ob = torch.rand(batch, *ob_shape, generator=gen).to(device)
    if channels_last:
        mb = mb.contiguous(memory_format=torch.channels_last)
        ob = ob.contiguous(memory_format=torch.channels_last)
    flat = torch.rand(batch, flat_size, generator=gen).to(device)
    legal_idx = torch.randint(
        0, NUM_JOINT_ACTIONS, (batch, legal_count), generator=gen
    ).to(device)
    return mb, ob, flat, legal_idx


def _configure_torch(args) -> None:
    if str(args.device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
        torch.backends.cudnn.allow_tf32 = not args.no_tf32
        if not args.no_tf32 and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _activities(device: str):
    acts = [ProfilerActivity.CPU]
    if str(device).startswith("cuda") and torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    return acts


def _make_trace_handler(trace_dir: str | None):
    if not trace_dir:
        return None
    from torch.profiler import tensorboard_trace_handler

    Path(trace_dir).mkdir(parents=True, exist_ok=True)
    return tensorboard_trace_handler(trace_dir)


def _profile_one(
    *,
    name: str,
    fn: Callable[[], object],
    device: str,
    steps: int,
    wait: int,
    warmup: int,
    active: int,
    repeat: int,
    row_limit: int,
    trace_dir: str | None,
) -> None:
    total_steps = wait + warmup + active * repeat
    if steps < total_steps:
        raise ValueError(
            f"--steps={steps} is too small for schedule wait={wait}, "
            f"warmup={warmup}, active={active}, repeat={repeat}; need at least "
            f"{total_steps}"
        )

    handler = _make_trace_handler(
        str(Path(trace_dir) / name) if trace_dir else None
    )
    print("\n" + "=" * 78)
    print(f"Profiling {name}: steps={steps}, schedule="
          f"{wait}/{warmup}/{active}x{repeat}")
    print("=" * 78)

    with profile(
        activities=_activities(device),
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=handler,
    ) as prof:
        with torch.inference_mode():
            for _ in range(steps):
                fn()
                prof.step()

    _sync(device)
    print("\nTop operators by CUDA time:")
    try:
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=row_limit
        ))
    except Exception as exc:
        print(f"(CUDA table unavailable: {exc})")

    print("\nTop operators by CPU time:")
    print(prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=row_limit
    ))

    if trace_dir:
        print(f"\nTrace written under: {Path(trace_dir) / name}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Profile KingdominoNet forward operator costs."
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--channels", type=int, default=48)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--bilinear_dim", type=int, default=64)
    p.add_argument("--norm", choices=["group", "batch"], default="group",
                   help="normalization layer in the residual trunk")
    p.add_argument("--batch", type=int, default=137)
    p.add_argument("--legal_count", type=int, default=50)
    p.add_argument("--mode", choices=["full", "legal", "both"], default="full")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--wait", type=int, default=5)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--active", type=int, default=10)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--row_limit", type=int, default=30)
    p.add_argument("--trace_dir", default=None,
                   help="optional TensorBoard trace directory")
    p.add_argument("--cudnn_benchmark", action="store_true")
    p.add_argument("--no_tf32", action="store_true")
    p.add_argument("--amp_inference", action="store_true")
    p.add_argument("--channels_last", action="store_true",
                   help="use channels-last memory format for board tensors and "
                        "conv weights")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")

    _configure_torch(args)
    net = KingdominoNet(
        channels=args.channels,
        blocks=args.blocks,
        bilinear_dim=args.bilinear_dim,
        norm=args.norm,
    ).to(args.device).eval()
    if args.channels_last:
        net = net.to(memory_format=torch.channels_last)
    mb, ob, flat, legal_idx = _make_inputs(
        batch=args.batch,
        device=args.device,
        legal_count=args.legal_count,
        seed=args.seed,
        channels_last=args.channels_last,
    )
    use_amp = bool(args.amp_inference and str(args.device).startswith("cuda"))

    def full_forward():
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=use_amp):
            return net(mb, ob, flat)

    def legal_forward():
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=use_amp):
            return net.forward_legal(mb, ob, flat, legal_idx)

    print(f"device={args.device}  net={args.channels}ch/{args.blocks}b "
          f"norm={args.norm}  "
          f"batch={args.batch}  legal_count={args.legal_count}  "
          f"tf32={not args.no_tf32}  amp={args.amp_inference}  "
          f"cudnn_benchmark={args.cudnn_benchmark}  "
          f"channels_last={args.channels_last}")

    _sync(args.device)
    if args.mode in ("full", "both"):
        _profile_one(
            name="full_forward",
            fn=full_forward,
            device=args.device,
            steps=args.steps,
            wait=args.wait,
            warmup=args.warmup,
            active=args.active,
            repeat=args.repeat,
            row_limit=args.row_limit,
            trace_dir=args.trace_dir,
        )
    if args.mode in ("legal", "both"):
        _profile_one(
            name=f"legal_forward_L{args.legal_count}",
            fn=legal_forward,
            device=args.device,
            steps=args.steps,
            wait=args.wait,
            warmup=args.warmup,
            active=args.active,
            repeat=args.repeat,
            row_limit=args.row_limit,
            trace_dir=args.trace_dir,
        )


if __name__ == "__main__":
    main()
