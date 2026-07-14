"""One-generation, restartable NNUE bootstrap loop.

Pipeline:
  incumbent self-play -> replayable shard -> train on old+new train splits
  (frozen validation source) -> export -> paired candidate gate -> local promote

The reserved test split is never opened.  This MVP uses honest final outcomes as
targets; deep-search value/policy relabeling is deliberately a later phase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = proc.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def _copy_triplet(base: Path, target_base: Path) -> None:
    target_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".knnue", ".manifest.json"):
        src = base.with_suffix(suffix)
        dst = target_base.with_suffix(suffix)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(src, tmp)
        tmp.replace(dst)


def run_generation(args) -> dict:
    import kingdomino_rust as kr

    if not hasattr(kr.RustSearch, "choose_action_timed"):
        raise RuntimeError(
            "the active kingdomino_rust extension predates operational search; "
            "run with the isolated release extension or rebuild/install it"
        )

    root = Path(args.run_dir).resolve()
    generation = root / f"generation_{args.generation:03d}"
    generation.mkdir(parents=True, exist_ok=True)
    state_path = generation / "state.json"
    run_config = {
        "games": args.games,
        "workers": args.workers,
        "seed_start": args.seed_start,
        "selfplay_move_secs": args.selfplay_move_secs,
        "max_depth": args.max_depth,
        "chance_samples": args.chance_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "train_seed": args.train_seed,
        "gate_move_secs": args.gate_move_secs,
        "paired_seeds": args.paired_seeds,
        "gate_seed_start": args.gate_seed_start,
        "min_win_rate": args.min_win_rate,
        "min_lcb": args.min_lcb,
    }
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema": "kingdomino_nnue_generation_v1",
        "generation": args.generation,
        "status": "started",
        "target_kind": "final_official_outcome",
        "reserved_test_opened": False,
        "phases": {},
        "run_config": run_config,
    }
    if state.get("run_config") != run_config:
        raise ValueError(
            "generation already exists with different run arguments; "
            "use a new --generation number"
        )

    incumbent_ckpt = Path(args.incumbent_ckpt).resolve()
    incumbent_knnue = Path(args.incumbent_knnue).resolve()
    base_sources = [str(Path(p).resolve()) for p in args.base_source]
    inputs = {
        "incumbent_ckpt": str(incumbent_ckpt),
        "incumbent_ckpt_sha256": _sha256(incumbent_ckpt),
        "incumbent_knnue": str(incumbent_knnue),
        "incumbent_knnue_sha256": _sha256(incumbent_knnue),
        "base_sources": base_sources,
    }
    if state.get("inputs") is not None and state["inputs"] != inputs:
        raise ValueError(
            "generation already exists with different incumbent/data inputs; "
            "use a new --generation number"
        )
    state["inputs"] = inputs
    _write_json(state_path, state)

    data_dir = generation / "selfplay"
    if not (data_dir / "manifest.json").exists():
        command = [
            sys.executable, "-m", "games.kingdomino.nnue.datagen",
            "--games", str(args.games),
            "--out", str(data_dir),
            "--workers", str(args.workers),
            "--seed-start", str(args.seed_start),
            "--eval", "sparse_nnue_q",
            "--nnue-path", str(incumbent_knnue),
            "--move-secs", str(args.selfplay_move_secs),
            "--max-depth", str(args.max_depth),
        ]
        _run(command, generation / "01_datagen.log")
    data_manifest = json.loads((data_dir / "manifest.json").read_text())
    if data_manifest.get("verify_failures"):
        raise RuntimeError("self-play replay verification failed")
    state["phases"]["selfplay"] = data_manifest
    _write_json(state_path, state)

    candidate_ckpt = generation / "candidate.pt"
    if not candidate_ckpt.exists():
        command = [
            sys.executable, "-m", "games.kingdomino.nnue.train_sparse",
            "--cache-dir", str(generation / "packed"),
            "--init-ckpt", str(incumbent_ckpt),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--device", args.device,
            "--seed", str(args.train_seed),
            "--out", str(candidate_ckpt),
        ]
        for source in [*base_sources, str(data_dir)]:
            command += ["--source", source]
        # Validation stays frozen on the explicitly supplied base corpus.  New
        # validation and all test JSONL remain unopened during model selection.
        for source in base_sources:
            command += ["--val-source", source]
        _run(command, generation / "02_train.log")
    state["phases"]["train"] = {
        "candidate": str(candidate_ckpt),
        "sha256": _sha256(candidate_ckpt),
    }
    _write_json(state_path, state)

    candidate_base = generation / "candidate"
    if not candidate_base.with_suffix(".knnue").exists():
        _run([
            sys.executable, "-m", "games.kingdomino.nnue.sparse_export",
            "--ckpt", str(candidate_ckpt), "--out", str(candidate_base),
        ], generation / "03_export.log")
    state["phases"]["export"] = {
        "knnue": str(candidate_base.with_suffix(".knnue")),
        "sha256": _sha256(candidate_base.with_suffix(".knnue")),
    }
    _write_json(state_path, state)

    match_path = generation / "promotion_match.json"
    if not match_path.exists():
        _run([
            sys.executable, "-m", "games.kingdomino.nnue.match", "nnue",
            "--candidate", str(candidate_base.with_suffix(".knnue")),
            "--incumbent", str(incumbent_knnue),
            "--move-secs", str(args.gate_move_secs),
            "--max-depth", str(args.max_depth),
            "--chance-samples", str(args.chance_samples),
            "--paired-seeds", str(args.paired_seeds),
            "--seed-start", str(args.gate_seed_start),
            "--out", str(match_path),
        ], generation / "04_match.log")
    match = json.loads(match_path.read_text())
    pair = match["pair"]
    passed = (
        pair["a_points_rate"] >= args.min_win_rate
        and pair["a_points_lcb_95"] > args.min_lcb
    )
    decision = {
        "passed": passed,
        "min_win_rate": args.min_win_rate,
        "min_lcb": args.min_lcb,
        "points_rate": pair["a_points_rate"],
        "lcb_95": pair["a_points_lcb_95"],
        "match": str(match_path),
    }
    state["phases"]["promotion"] = decision

    current_base = root / "current_best"
    if passed:
        _copy_triplet(candidate_base, current_base)
        ckpt_dst = current_base.with_suffix(".pt")
        tmp = ckpt_dst.with_suffix(ckpt_dst.suffix + ".tmp")
        shutil.copy2(candidate_ckpt, tmp)
        tmp.replace(ckpt_dst)
        decision["promoted_to"] = str(current_base)
    state["status"] = "complete"
    _write_json(state_path, state)
    _write_json(generation / "decision.json", decision)
    print(json.dumps(decision, indent=2))
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--generation", type=int, default=0)
    ap.add_argument("--incumbent-ckpt", required=True)
    ap.add_argument("--incumbent-knnue", required=True)
    ap.add_argument("--base-source", action="append", required=True)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=1_000_000)
    ap.add_argument("--selfplay-move-secs", type=float, default=0.1)
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--chance-samples", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--gate-move-secs", type=float, default=0.5)
    ap.add_argument("--paired-seeds", type=int, default=64)
    ap.add_argument("--gate-seed-start", type=int, default=2_000_000)
    ap.add_argument("--min-win-rate", type=float, default=0.55)
    ap.add_argument("--min-lcb", type=float, default=0.50)
    args = ap.parse_args()
    run_generation(args)


if __name__ == "__main__":
    main()
