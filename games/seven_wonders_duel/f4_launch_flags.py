"""W6.3: translate a measured F4 production manifest into Phase D launch flags.

The throughput bench and the training loop name the same four settings
differently -- the bench takes ``--slots``, Phase D takes ``--rust-slots`` -- so
the sweep's answer cannot be pasted into the launch. argparse rejects the wrong
spelling loudly, which is better than silence, but the fix has always been a
manual re-typing step in the middle of the one workflow where a transcription
error is both easy and expensive: the numbers look plausible either way, and the
run is 24 hours long.

This module is that translation, written down once and tested.  Anything the
bench measures that Phase D cannot accept is listed explicitly in
:data:`NOT_TRANSLATED` with the reason, so a new bench field fails here rather
than being dropped on the floor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PHASE_D_FLAGS: dict[str, str] = {
    "slots": "--rust-slots",
    "global_batch_cap": "--rust-global-batch-cap",
    "max_inflight_batches": "--rust-max-inflight-batches",
    "scheduler_workers": "--rust-scheduler-workers",
}
"""Bench manifest field -> Phase D flag. The whole point of this module."""

NOT_TRANSLATED: dict[str, str] = {
    "pinned_memory": (
        "bench-only: the loop's Rust boundary owns its own staging buffers"
    ),
    "torch_compile": (
        "bench-only: Phase D compiles via its own evaluator construction"
    ),
    "leaf_batch": "generation-schedule setting, not a throughput knob",
    "device": "supplied by the launch, not by the sweep",
    "inference_precision": "set by --precision, which is pinned per run",
}
"""Measured fields that deliberately do not become launch flags."""

_ENVIRONMENT_FIELDS = frozenset(
    {
        "contract_schema_version",
        "contract_sha256",
        "quality_lock_sha256",
        "checkpoint_sha256",
        "git_commit",
        "dirty_worktree",
        "torch_version",
        "cuda_version",
        "cpu_model",
        "gpu_model",
        "games",
        "seed",
    }
)
"""Provenance recorded by the bench; never a launch flag."""


def launch_flags(manifest: dict) -> list[str]:
    """Phase D flags for a measured manifest, in a stable order.

    Raises when a tuned field is missing rather than silently launching on a
    default: an unflagged ``--rust-scheduler-workers`` is 1, and a sweep that
    chose 4 would be quietly discarded.
    """

    missing = [field for field in PHASE_D_FLAGS if field not in manifest]
    if missing:
        raise KeyError(
            "production manifest is missing tuned fields "
            f"{sorted(missing)}; launching without them would silently fall "
            "back to Phase D defaults and discard the sweep"
        )
    flags: list[str] = []
    for field, flag in PHASE_D_FLAGS.items():
        value = manifest[field]
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{field} must be an int, got {value!r}")
        if value <= 0:
            raise ValueError(f"{field} must be positive, got {value}")
        flags.extend([flag, str(value)])
    return flags


def unknown_fields(manifest: dict) -> list[str]:
    """Measured fields this translation neither maps nor knowingly ignores."""

    known = set(PHASE_D_FLAGS) | set(NOT_TRANSLATED) | _ENVIRONMENT_FIELDS
    return sorted(field for field in manifest if field not in known)


def production_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "production_manifest" in payload:
        return payload["production_manifest"]
    if "manifest" in payload:
        return payload["manifest"]
    if "winner" in payload:
        return payload["winner"]["manifest"]
    raise KeyError(
        f"{path} has no production_manifest/manifest/winner block; pass the "
        "output of f4_cloud_finalize.py"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "production",
        type=Path,
        help="f4_cloud_finalize.py output (or any file with a manifest block)",
    )
    parser.add_argument(
        "--strict-unknown",
        action="store_true",
        help="fail when the manifest holds a field this translation does not "
        "know about, instead of reporting it",
    )
    args = parser.parse_args(argv)
    manifest = production_manifest(args.production)
    unknown = unknown_fields(manifest)
    if unknown and args.strict_unknown:
        raise SystemExit(
            f"unmapped manifest fields: {unknown}; add them to PHASE_D_FLAGS or "
            "to NOT_TRANSLATED with a reason"
        )
    print(" ".join(launch_flags(manifest)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
