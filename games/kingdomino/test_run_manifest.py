from __future__ import annotations

import json
from pathlib import Path

import torch

from games.kingdomino.network import KingdominoNet
from games.kingdomino.run_manifest import (
    ARTIFACTS,
    initialize_run_manifest,
    record_checkpoint,
)
from games.kingdomino.self_play import SelfPlayConfig, save_checkpoint


def _tiny_cfg(tmp_path: Path) -> SelfPlayConfig:
    return SelfPlayConfig(
        channels=8,
        blocks=1,
        bilinear_dim=8,
        checkpoint_dir=str(tmp_path),
        log_path=str(tmp_path / "training_log.jsonl"),
        lr_schedule="0:0.001,5:0.0003",
        sims_schedule="0:16,10:32",
        alpha_schedule="0:0.8,20:0.5",
    )


def test_initialize_run_manifest_writes_expected_artifacts(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path)
    net = KingdominoNet(
        channels=cfg.channels,
        blocks=cfg.blocks,
        bilinear_dim=cfg.bilinear_dim,
    )

    metadata = initialize_run_manifest(
        cfg,
        tmp_path,
        log_path=cfg.log_path or tmp_path / "training_log.jsonl",
        net=net,
    )

    for name in ARTIFACTS.values():
        assert (tmp_path / name).exists(), name

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((tmp_path / "model_contract.json").read_text(encoding="utf-8"))
    schedules = json.loads((tmp_path / "schedule_config.json").read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == 1
    assert manifest["run_id"] == tmp_path.name
    assert manifest["artifacts"]["model_contract"] == "model_contract.json"
    assert contract["channels"] == 8
    assert contract["blocks"] == 1
    assert contract["policy_head_size"] > 0
    assert schedules["step_indexing"] == "zero_based"
    assert schedules["schedules"]["n_simulations"]["parsed"] == [[0, 16], [10, 32]]
    assert metadata["ruleset_hash"]
    assert metadata["run_manifest_path"].endswith("run_manifest.json")


def test_checkpoint_embeds_manifest_metadata_and_records_checkpoint(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path)
    net = KingdominoNet(
        channels=cfg.channels,
        blocks=cfg.blocks,
        bilinear_dim=cfg.bilinear_dim,
    )
    metadata = initialize_run_manifest(
        cfg,
        tmp_path,
        log_path=cfg.log_path or tmp_path / "training_log.jsonl",
        net=net,
    )

    checkpoint_path = tmp_path / "iter_0001.pt"
    save_checkpoint(str(checkpoint_path), net, cfg, 1, {"benchmark": []}, metadata)
    record_checkpoint(tmp_path, checkpoint_path, 1)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert ckpt["run_manifest"]["run_id"] == tmp_path.name
    assert ckpt["run_manifest"]["ruleset_hash"] == metadata["ruleset_hash"]
    assert ckpt["run_manifest"]["model_contract_path"].endswith("model_contract.json")

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["last_checkpoint"]["name"] == "iter_0001.pt"
    assert manifest["checkpoints"] == [manifest["last_checkpoint"]]
