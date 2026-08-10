"""Inventory and validate the frozen 240-position chance-search corpus.

This scaffold intentionally refuses tuning access until a manifest is sealed
with 120 tuning and 120 untouched confirmation positions.  Inventorying an
available source does not silently promote its positions into either split.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from games.kingdomino.denial_search import (
    chance_public_state_key_v1,
    public_state_key,
)
from games.kingdomino.denial_signal_sweep import file_sha256, load_frozen_positions


SCHEMA_VERSION = "kd-chance-position-corpus-config-v1"
INVENTORY_SCHEMA = "kd-chance-position-corpus-inventory-v1"
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "chance_position_corpus_v1.json"
DEFAULT_INVENTORY = Path(
    "runs/kingdomino/chance_correct_a1/chance_position_corpus_inventory_v1.json"
)
VALID_SPLITS = {"tuning", "confirmation"}
VALID_SOURCE_KINDS = {
    "ordinary_self_play",
    "advisor_loss",
    "bga_loss",
    "flexibility_draft_order",
    "defensive_blocking",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed or unreadable corpus JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"corpus JSON must be an object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _repo_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name != "configs":
        raise ValueError("corpus config must live in games/kingdomino/configs")
    return resolved.parents[3]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("chance corpus schema mismatch")
    if config.get("status") not in {"inventory-only-unsealed", "sealed"}:
        raise ValueError("chance corpus status must be inventory-only-unsealed or sealed")
    target = config.get("target", {})
    if int(target.get("total_positions", 0)) != 240 or target.get(
        "split_counts"
    ) != {"tuning": 120, "confirmation": 120}:
        raise ValueError("chance corpus must freeze a 120/120 split")
    bag_sizes = [int(value) for value in target.get("reachable_pre_reveal_bag_sizes", [])]
    if bag_sizes != list(range(44, 3, -4)):
        raise ValueError("reachable pre-reveal bag-size strata are incomplete")
    required_tags = target.get("required_tags", [])
    if len(required_tags) != 4 or len(set(required_tags)) != 4:
        raise ValueError("four distinct strategic source/tag families are required")
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("at least one inventory source must be declared")
    source_ids = [str(row["source_id"]) for row in sources]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("duplicate corpus source IDs")
    for source in sources:
        if source.get("source_kind") not in VALID_SOURCE_KINDS:
            raise ValueError(f"invalid corpus source kind: {source.get('source_kind')}")
        digest = str(source.get("expected_sha256", ""))
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("corpus source requires a lowercase SHA-256")
        if source.get("source_kind") == "bga_loss":
            provenance = source.get("provenance", {})
            for field in ("source_artifact", "collection_basis", "authorization"):
                if not provenance.get(field):
                    raise ValueError(f"BGA source missing explicit provenance field {field}")
    if not isinstance(config.get("entries"), list):
        raise ValueError("corpus entries must be a list")


def _assert_hidden_order_invariant(state: Any) -> None:
    public_key = public_state_key(state)
    chance_key = chance_public_state_key_v1(state)
    variant = state.copy()
    variant.deck = list(reversed(variant.deck))
    if public_state_key(variant) != public_key or chance_public_state_key_v1(
        variant
    ) != chance_key:
        raise ValueError("public-state identity leaks hidden deck order")


def inventory_sources(
    config_path: str | Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    validate_config(config)
    root = _repo_root(config_path)
    records: dict[tuple[str, int], dict[str, Any]] = {}
    source_reports = []
    for source in config["sources"]:
        source_id = str(source["source_id"])
        source_path = _resolve(root, str(source["path"]))
        actual_hash = file_sha256(source_path)
        if actual_hash != source["expected_sha256"]:
            raise ValueError(f"corpus source hash mismatch: {source_path}")
        positions = load_frozen_positions(source_path)
        bag_counts: Counter[int] = Counter()
        for index, (state, metadata) in enumerate(positions):
            _assert_hidden_order_invariant(state)
            bag_size = len(state.deck)
            declared_bag = metadata.get("deck_count")
            if declared_bag is not None and int(declared_bag) != bag_size:
                raise ValueError(
                    f"source metadata bag count mismatch at {source_id}:{index}"
                )
            bag_counts[bag_size] += 1
            records[(source_id, index)] = {
                "source_id": source_id,
                "source_index": index,
                "source_kind": source["source_kind"],
                "source_path": str(source_path),
                "source_sha256": actual_hash,
                "bag_size": bag_size,
                "public_state_key": public_state_key(state),
                "chance_public_state_key_hex": chance_public_state_key_v1(state).hex(),
                "default_tags": list(source.get("default_tags", [])),
                "metadata": metadata,
                "hidden_order_invariance_verified": True,
            }
        source_reports.append(
            {
                "source_id": source_id,
                "source_kind": source["source_kind"],
                "path": str(source_path),
                "sha256": actual_hash,
                "positions": len(positions),
                "bag_size_counts": {
                    str(key): value for key, value in sorted(bag_counts.items())
                },
                "default_tags": list(source.get("default_tags", [])),
            }
        )
    return config, records | {("__reports__", -1): {"sources": source_reports}}


def validate_entries(
    config: dict[str, Any], inventory: dict[tuple[str, int], dict[str, Any]]
) -> dict[str, Any]:
    entries = config["entries"]
    position_ids = [str(row.get("position_id", "")) for row in entries]
    if any(not value for value in position_ids) or len(set(position_ids)) != len(
        position_ids
    ):
        raise ValueError("corpus entries require unique nonempty position IDs")
    public_keys = [str(row.get("public_state_key", "")) for row in entries]
    if len(set(public_keys)) != len(public_keys):
        raise ValueError("duplicate public-state identity in corpus entries")
    split_counts: Counter[str] = Counter()
    split_bags: dict[str, Counter[int]] = {
        "tuning": Counter(),
        "confirmation": Counter(),
    }
    split_tags: dict[str, Counter[str]] = {
        "tuning": Counter(),
        "confirmation": Counter(),
    }
    for entry in entries:
        split = str(entry.get("split", ""))
        if split not in VALID_SPLITS:
            raise ValueError(f"invalid corpus split {split}")
        key = (str(entry.get("source_id", "")), int(entry.get("source_index", -1)))
        source_row = inventory.get(key)
        if source_row is None:
            raise ValueError(f"corpus entry references unknown source row {key}")
        if (
            entry.get("public_state_key") != source_row["public_state_key"]
            or entry.get("chance_public_state_key_hex")
            != source_row["chance_public_state_key_hex"]
            or int(entry.get("bag_size", -1)) != source_row["bag_size"]
        ):
            raise ValueError(f"corpus entry public identity drift: {entry['position_id']}")
        tags = {str(value) for value in entry.get("tags", [])}
        if not tags:
            raise ValueError(f"corpus entry lacks strategic tags: {entry['position_id']}")
        if source_row["source_kind"] == "bga_loss":
            provenance = entry.get("bga_provenance", {})
            for field in ("game_id", "source_artifact", "authorization"):
                if not provenance.get(field):
                    raise ValueError(
                        f"BGA-derived entry {entry['position_id']} lacks {field} provenance"
                    )
        split_counts[split] += 1
        split_bags[split][source_row["bag_size"]] += 1
        split_tags[split].update(tags)

    sealed = config["status"] == "sealed"
    if sealed:
        if dict(split_counts) != config["target"]["split_counts"]:
            raise ValueError("sealed corpus does not have exact 120/120 counts")
        required_bags = set(config["target"]["reachable_pre_reveal_bag_sizes"])
        required_tags = set(config["target"]["required_tags"])
        for split in VALID_SPLITS:
            if set(split_bags[split]) != required_bags:
                raise ValueError(f"sealed {split} split lacks reachable bag-size strata")
            if not required_tags.issubset(split_tags[split]):
                raise ValueError(f"sealed {split} split lacks required strategic tags")
    elif entries:
        raise ValueError("unsealed inventory config must not assign tuning/confirmation entries")
    return {
        "status": config["status"],
        "entry_count": len(entries),
        "split_counts": dict(split_counts),
        "sealed_and_usable": sealed,
    }


def build_inventory_report(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config, inventory = inventory_sources(config_path)
    entry_validation = validate_entries(config, inventory)
    source_reports = inventory[("__reports__", -1)]["sources"]
    available_bags: Counter[int] = Counter()
    available_tags: Counter[str] = Counter()
    for key, row in inventory.items():
        if key == ("__reports__", -1):
            continue
        available_bags[row["bag_size"]] += 1
        available_tags.update(row["default_tags"])
    required_bags = set(config["target"]["reachable_pre_reveal_bag_sizes"])
    required_tags = set(config["target"]["required_tags"])
    return {
        "schema_version": INVENTORY_SCHEMA,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "status": config["status"],
        "sources": source_reports,
        "available_unique_positions": len(inventory) - 1,
        "available_bag_size_counts": {
            str(key): value for key, value in sorted(available_bags.items())
        },
        "missing_bag_sizes": sorted(required_bags - set(available_bags), reverse=True),
        "available_default_tag_counts": dict(sorted(available_tags.items())),
        "missing_required_tags": sorted(required_tags - set(available_tags)),
        "split_validation": entry_validation,
        "decision": "inventory only; collection and immutable split assignment remain required before tuning",
    }


def load_split(
    config_path: str | Path, split: str, *, allow_confirmation: bool = False
) -> list[dict[str, Any]]:
    config, inventory = inventory_sources(config_path)
    validation = validate_entries(config, inventory)
    if not validation["sealed_and_usable"]:
        raise ValueError("corpus is unsealed; tuning and confirmation access are blocked")
    if split not in VALID_SPLITS:
        raise ValueError(f"invalid corpus split {split}")
    if split == "confirmation" and not allow_confirmation:
        raise PermissionError("confirmation split is sealed from tuning access")
    return [row for row in config["entries"] if row["split"] == split]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_inventory_report(args.config)
    if not args.validate_only:
        _atomic_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
