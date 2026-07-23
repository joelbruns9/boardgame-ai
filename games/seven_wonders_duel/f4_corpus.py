"""Build the deterministic stratified position corpus for the F4.2 quality gate.

The output combines every supplied Phase-E trap fixture/truth with replayable
production-buffer states until each preregistered phase stratum is populated.
It never manufactures ground truth for generic states; consequential trap quotas
remain an explicit manifest failure until Phase-E harvesting/ground-truth work has
actually produced enough fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .buffer import read_records, replay
from .codec import decode_action, legal_action_indices
from .game import ChanceKind, Phase
from .phase_e import chance_signature, reconstruct, state_actor
from .f4_quality import CONTRACT_PATH, REQUIRED_PHASE_STRATA


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def position_stratum(state) -> str | None:
    if state.pending_choice is not None:
        return "pending_choice"
    if state.phase is Phase.WONDER_DRAFT:
        return "wonder_draft"
    if state.phase is Phase.CHOOSE_NEXT_START_PLAYER:
        return "between_ages"
    if state.phase is Phase.PLAY_AGE and state.age in (1, 2, 3):
        return f"age_{state.age}"
    return None


def chance_tags(state) -> list[str]:
    tags = set()
    names = {
        ChanceKind.CARD_REVEAL: "card_reveal",
        ChanceKind.GREAT_LIBRARY_DRAW: "great_library_draw",
        ChanceKind.WONDER_GROUP_REVEAL: "wonder_group_reveal",
        ChanceKind.AGE_DEAL: "age_deal",
    }
    for index in legal_action_indices(state):
        for spec in chance_signature(state, decode_action(state, index)):
            tags.add(names[spec.kind])
    return sorted(tags)


def build(args) -> dict:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    corpus_cfg = contract["fast_search_quality_gate"]["corpus"]
    per_phase = corpus_cfg["minimum_positions_per_game_phase"]
    target_total = corpus_cfg["minimum_total_positions"]
    trap_positions_path = args.phase_e / "positions.jsonl"
    trap_truths_path = args.phase_e / "ground_truth.jsonl"
    trap_positions = _read_jsonl(trap_positions_path)
    truths = {row["id"]: row for row in _read_jsonl(trap_truths_path)}

    positions: list[dict] = []
    seen = set()
    phase_counts = {stratum: 0 for stratum in REQUIRED_PHASE_STRATA}
    chance_counts = {
        "card_reveal": 0,
        "great_library_draw": 0,
        "wonder_group_reveal": 0,
        "age_deal": 0,
    }

    def add(row: dict, state) -> bool:
        if row["id"] in seen:
            return False
        stratum = position_stratum(state)
        if stratum is None:
            return False
        tags = chance_tags(state)
        normalized = {
            **row,
            "stratum": stratum,
            "age": state.age,
            "actor": state_actor(state),
            "chance_tags": tags,
            "n_legal": len(legal_action_indices(state)),
        }
        positions.append(normalized)
        seen.add(row["id"])
        phase_counts[stratum] += 1
        for tag in tags:
            chance_counts[tag] += 1
        return True

    # Preserve the complete *consequential* Phase-E bank first.  Harvested
    # positions are candidates, not established traps: carrying every rejected
    # candidate into the paired 32-seed sweep makes the calibration arbitrarily
    # large without adding evidence required by the contract.
    consequential_gap = corpus_cfg["consequential_gap_minimum"]
    for position in trap_positions:
        if truths.get(position["id"], {}).get("trap_gap", 0.0) >= consequential_gap:
            add(position, reconstruct(position))

    for buffer_path in args.buffers:
        if all(count >= per_phase for count in phase_counts.values()) and len(positions) >= target_total:
            break
        source = buffer_path.stem
        for game_index, record in enumerate(read_records(buffer_path)):
            if all(count >= per_phase for count in phase_counts.values()) and len(positions) >= target_total:
                break
            prefix: list[int] = []

            def on_state(state, move, _record=record, _game_index=game_index):
                stratum = position_stratum(state)
                need_phase = stratum is not None and phase_counts[stratum] < per_phase
                need_total = len(positions) < target_total
                if not (need_phase or need_total):
                    prefix.append(move.action)
                    return
                row = {
                    "id": f"f4:{source}:{_game_index}:{move.i}",
                    "source": source,
                    "game_seed": _record.seed,
                    "first_player": _record.first_player,
                    "move_index": move.i,
                    "prefix": list(prefix),
                    "traps": [],
                    "safe": [],
                    "unsafe_other": [],
                }
                add(row, state)
                prefix.append(move.action)

            replay(record, on_state=on_state)

    consequential = [
        position
        for position in positions
        if truths.get(position["id"], {}).get("trap_gap", 0.0) >= consequential_gap
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    positions_out = args.output / "positions.jsonl"
    truths_out = args.output / "ground_truth.jsonl"
    positions_out.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in positions),
        encoding="utf-8",
    )
    truths_out.write_text(
        "".join(
            json.dumps(truths[position["id"]], sort_keys=True, allow_nan=False) + "\n"
            for position in positions
            if position["id"] in truths
        ),
        encoding="utf-8",
    )
    manifest = {
        "contract_sha256": _sha256(CONTRACT_PATH),
        "positions": len(positions),
        "phase_counts": phase_counts,
        "chance_counts": chance_counts,
        "consequential_positions": len(consequential),
        "requirements": {
            "minimum_total_positions": target_total,
            "minimum_positions_per_game_phase": per_phase,
            "minimum_consequential_positions": corpus_cfg["minimum_consequential_positions"],
        },
        "structural_eligible": (
            len(positions) >= target_total
            and all(count >= per_phase for count in phase_counts.values())
            and len(consequential) >= corpus_cfg["minimum_consequential_positions"]
        ),
        "sources": {
            "phase_e_positions": {
                "path": str(trap_positions_path.resolve()),
                "sha256": _sha256(trap_positions_path),
            },
            "phase_e_truths": {
                "path": str(trap_truths_path.resolve()),
                "sha256": _sha256(trap_truths_path),
            },
            "buffers": [
                {"path": str(path.resolve()), "sha256": _sha256(path)} for path in args.buffers
            ],
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-e", type=Path, required=True)
    parser.add_argument("--buffers", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
