from __future__ import annotations

import copy
import json

import pytest

from games.kingdomino.chance_corpus_manifest import (
    DEFAULT_CONFIG,
    _assert_hidden_order_invariant,
    build_inventory_report,
    inventory_sources,
    load_split,
    validate_config,
    validate_entries,
)
from games.kingdomino.denial_signal_sweep import load_frozen_positions


def _config():
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _entry(source_row, *, position_id="p0", split="tuning"):
    return {
        "position_id": position_id,
        "source_id": source_row["source_id"],
        "source_index": source_row["source_index"],
        "public_state_key": source_row["public_state_key"],
        "chance_public_state_key_hex": source_row["chance_public_state_key_hex"],
        "bag_size": source_row["bag_size"],
        "split": split,
        "tags": ["ordinary_self_play"],
    }


def test_inventory_reports_available_coverage_and_missing_strata():
    report = build_inventory_report(DEFAULT_CONFIG)
    assert report["available_unique_positions"] == 50
    assert report["available_bag_size_counts"] == {
        "8": 8,
        "12": 8,
        "16": 8,
        "20": 8,
        "24": 9,
        "28": 9,
    }
    assert report["missing_bag_sizes"] == [44, 40, 36, 32, 4]
    assert report["missing_required_tags"] == [
        "advisor_or_bga_loss",
        "defensive_blocking",
        "flexibility_or_draft_order",
    ]
    assert report["split_validation"]["sealed_and_usable"] is False


def test_public_identity_is_hidden_deck_order_invariant():
    positions = load_frozen_positions(
        DEFAULT_CONFIG.resolve().parents[3]
        / "runs/kingdomino/denial_search/signal_positions.jsonl"
    )
    _assert_hidden_order_invariant(positions[0][0])


def test_duplicate_public_identity_is_rejected():
    config, inventory = inventory_sources(DEFAULT_CONFIG)
    source_row = inventory[("denial_signal_positions_v1", 0)]
    config = copy.deepcopy(config)
    config["entries"] = [
        _entry(source_row, position_id="first"),
        _entry(source_row, position_id="second", split="confirmation"),
    ]
    with pytest.raises(ValueError, match="duplicate public-state"):
        validate_entries(config, inventory)


def test_public_identity_drift_is_rejected_before_assignment():
    config, inventory = inventory_sources(DEFAULT_CONFIG)
    source_row = inventory[("denial_signal_positions_v1", 0)]
    config = copy.deepcopy(config)
    row = _entry(source_row)
    row["public_state_key"] = "stale"
    config["entries"] = [row]
    with pytest.raises(ValueError, match="public identity drift"):
        validate_entries(config, inventory)


def test_unsealed_manifest_blocks_tuning_and_confirmation_access():
    with pytest.raises(ValueError, match="unsealed"):
        load_split(DEFAULT_CONFIG, "tuning")
    with pytest.raises(ValueError, match="unsealed"):
        load_split(DEFAULT_CONFIG, "confirmation", allow_confirmation=True)


def test_sealed_manifest_requires_exact_counts_before_use():
    config, inventory = inventory_sources(DEFAULT_CONFIG)
    config = copy.deepcopy(config)
    config["status"] = "sealed"
    with pytest.raises(ValueError, match="exact 120/120"):
        validate_entries(config, inventory)


def test_bga_sources_require_explicit_authorization_provenance():
    config = _config()
    source = config["sources"][0]
    source["source_kind"] = "bga_loss"
    source["provenance"] = {"source_artifact": "game.json"}
    with pytest.raises(ValueError, match="BGA source missing"):
        validate_config(config)
