"""Placement-only audit tooling for the Kingdomino BGA corpus.

Phase 1 reconstructs placement decisions from consecutive reliable board
snapshots and score-verifies terminal suffixes against authoritative BGA final
scores.  It deliberately does not evaluate a network or optimize non-terminal
future placements: the output is a screened input corpus for those later steps.

Example::

    python -m games.kingdomino.placement_headroom_audit reconstruct --split all
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
from pathlib import Path
import time
from typing import Any, Iterable

from games.kingdomino.board import Board, Placement
from games.kingdomino.dominoes import DOMINOES
from games.kingdomino.game import Claim, GameState, Phase, TurnAction
from games.kingdomino.web_app import state_from_debug_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("placement_audit_corpus_v1.json")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "kingdomino" / "placement_audit"
FEASIBILITY_PROBE_V1 = (
    "881199380",
    "881159658",
    "881651578",
    "881170142",
    "883077657",
)


@dataclass(frozen=True)
class ReconstructionRow:
    table_id: str
    split: str
    source_decision_index: int
    target_decision_index: int | None
    player: int
    player_id: str | None
    player_name: str | None
    phase: str
    deck_count: int
    domino_id: int | None
    next_pick_domino_id: int | None
    placement: dict[str, Any] | None
    status: str
    drop_reason: str | None
    source_captured_at: str | None
    target_captured_at: str | None
    remaining_domino_ids: list[int] | None = None
    actual_final_score: int | None = None
    best_final_score: int | None = None
    final_regret: int | None = None
    matching_final_sequences: int | None = None
    scoring_harmony: bool | None = None
    scoring_middle_kingdom: bool | None = None
    matching_scoring_rules: list[str] | None = None
    first_action_unique: bool | None = None

    @property
    def kept(self) -> bool:
        return self.status in {
            "reconstructed",
            "forced_discard",
            "score_verified_terminal",
        }


@dataclass(frozen=True)
class CompletenessRow:
    table_id: str
    split: str
    player: int
    player_id: str | None
    player_name: str | None
    final_score: int | None
    ordered_domino_ids: list[int]
    dominoes_by_layer: list[list[int]]
    tile_count: int
    complete_layers: int
    sequence_complete: bool
    scoring_rule_candidates: list[str]
    scoring_rules_unique: bool
    whole_game_score_eligible: bool
    captured_placement_decisions: int
    verified_placement_decisions: int
    decision_regret_eligible: int
    issues: list[str]


@dataclass(frozen=True)
class BeamLayerStats:
    layer: int
    domino_id: int
    incoming_states: int
    expanded_actions: int
    unique_states: int
    kept_states: int
    forced_discard_parents: int
    best_partial_score: int
    elapsed_seconds: float


@dataclass(frozen=True)
class FixedSequenceSolveResult:
    domino_ids: list[int]
    beam_width: int | None
    best_score: int
    territory_score: int
    harmony_bonus: int
    middle_kingdom_bonus: int
    discards: int
    placements: list[dict[str, Any] | None]
    layer_stats: list[BeamLayerStats]
    elapsed_seconds: float
    backend: str = "python"


@dataclass
class _BeamNode:
    board: Board
    discards: int
    placements: tuple[Placement | None, ...]


class ExactStateLimitExceeded(RuntimeError):
    def __init__(self, layer: int, unique_states: int, limit: int):
        self.layer = layer
        self.unique_states = unique_states
        self.limit = limit
        super().__init__(
            f"exact state limit {limit} exceeded at layer {layer} "
            f"({unique_states} unique states observed)"
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "kingdomino-placement-corpus/v1":
        raise ValueError(f"Unsupported placement corpus manifest: {path}")
    return manifest


def _resolve_game_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Versioned manifests live below the repository, while corpus paths are
    # repository-relative.  Falling back to the manifest directory also makes
    # small test manifests convenient.
    repo_candidate = REPO_ROOT / path
    return repo_candidate if repo_candidate.exists() else manifest_path.parent / path


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(
            f"Corpus hash mismatch for {path}: expected {expected}, got {digest}"
        )


def _warning_players(state_json: dict[str, Any]) -> set[int] | None:
    """Return warned players, or ``None`` when an unscoped warning is present."""
    warning = state_json.get("board_reconstruction_warning")
    if not warning:
        return set()
    warnings = warning if isinstance(warning, list) else [warning]
    players: set[int] = set()
    for item in warnings:
        if not isinstance(item, dict) or item.get("player") is None:
            return None
        players.add(int(item["player"]))
    return players


def _board_cells(state_json: dict[str, Any], player: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    boards = state_json.get("boards", [])
    if player < 0 or player >= len(boards):
        raise ValueError("missing_board")
    cells: dict[tuple[int, int], tuple[int, int, int]] = {}
    for cell in boards[player].get("cells", []):
        coord = (int(cell["x"]), int(cell["y"]))
        if coord in cells:
            raise ValueError("duplicate_board_coordinate")
        cells[coord] = (
            int(cell["terrain_id"]),
            int(cell.get("crowns", 0)),
            int(cell.get("domino_id", 0)),
        )
    return cells


def _board_is_reliable(state_json: dict[str, Any], player: int) -> bool:
    warned = _warning_players(state_json)
    if warned is None or player in warned:
        return False
    notes = state_json.get("debug", {}).get("notes", [])
    board_notes = [
        str(note) for note in notes
        if str(note).startswith(f"Player {player} board ")
    ]
    if board_notes:
        # The capture can also rebuild an inactive player's board from rendered
        # DOM tiles.  That fallback is useful for live advice but its coordinate
        # frame sometimes differs from the authoritative BGA kingdom grid, so it
        # is not reliable enough for consecutive-snapshot deltas.
        authoritative = (
            f"Player {player} board built from authoritative BGA kingdom grid."
        )
        if authoritative not in board_notes:
            return False
    try:
        cells = _board_cells(state_json, player)
    except (KeyError, TypeError, ValueError):
        return False
    castle_count = sum(terrain == 1 for terrain, _crowns, _domino in cells.values())
    return bool(cells) and castle_count == 1


def _board_fingerprint(board: Board) -> dict[tuple[int, int], tuple[int, int, int]]:
    return {
        (x, y): (
            int(board.terrain[y, x]),
            int(board.crowns[y, x]),
            int(board.domino_id[y, x]),
        )
        for x, y in board.occupied_cells()
    }


def _observable_cells(
    cells: dict[tuple[int, int], tuple[int, int, int]],
) -> dict[tuple[int, int], tuple[int, int]]:
    """Strip cell domino IDs, which authoritative BGA grids report as ``-2``."""
    return {coord: (terrain, crowns) for coord, (terrain, crowns, _domino) in cells.items()}


def _observable_board_fingerprint(board: Board) -> dict[tuple[int, int], tuple[int, int]]:
    return _observable_cells(_board_fingerprint(board))


def _claim_domino(state_json: dict[str, Any], actor: int) -> int:
    claims = state_json.get("pending_claims", [])
    actor_index = int(state_json.get("actor_index", 0))
    if actor_index < 0 or actor_index >= len(claims):
        raise ValueError("missing_current_claim")
    claim = claims[actor_index]
    if int(claim["player"]) != actor:
        raise ValueError("current_claim_player_mismatch")
    return int(claim["domino_id"])


def _infer_next_pick(
    source: dict[str, Any], subsequent_states: Iterable[dict[str, Any]], actor: int
) -> int:
    available = {int(value) for value in source.get("current_row", [])}
    for state in subsequent_states:
        claims = state.get("pending_claims", []) + state.get("next_claims", [])
        claimed = {
            int(claim["domino_id"])
            for claim in claims
            if int(claim["player"]) == actor
        }
        candidates = sorted(available & claimed)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            break
    raise ValueError("next_pick_not_unique")


def _player_metadata(
    records: list[dict[str, Any]], decisions: list[dict[str, Any]]
) -> dict[int, tuple[str | None, str | None]]:
    bga_to_engine: dict[str, int] = {}
    for record in decisions:
        active_player = record.get("active_player")
        actor = record.get("state", {}).get("current_actor")
        if active_player is not None and actor is not None:
            bga_to_engine[str(active_player)] = int(actor)

    final = next((record.get("final") for record in records if record.get("kind") == "final"), None)
    players = final.get("players", {}) if isinstance(final, dict) else {}
    out: dict[int, tuple[str | None, str | None]] = {}
    for player_id, engine_index in bga_to_engine.items():
        detail = players.get(player_id, {})
        out[engine_index] = (player_id, detail.get("name"))
    return out


def _final_scores_by_engine(
    records: list[dict[str, Any]],
    metadata: dict[int, tuple[str | None, str | None]],
) -> dict[int, int]:
    final = next((record.get("final") for record in records if record.get("kind") == "final"), None)
    players = final.get("players", {}) if isinstance(final, dict) else {}
    scores: dict[int, int] = {}
    for engine_index, (player_id, _name) in metadata.items():
        if player_id is not None and player_id in players and players[player_id].get("score") is not None:
            scores[engine_index] = int(players[player_id]["score"])
    return scores


def _remaining_score_audit_dominoes(
    source: dict[str, Any], actor: int
) -> tuple[list[int], int | None]:
    actor_index = int(source.get("actor_index", 0))
    pending = source.get("pending_claims", [])[actor_index:]
    current_and_pending = [
        int(claim["domino_id"])
        for claim in pending
        if int(claim["player"]) == actor
    ]
    if source.get("phase") == Phase.FINAL_PLACEMENT.name:
        return current_and_pending, None
    if (
        source.get("phase") == Phase.PLACE_AND_SELECT.name
        and int(source.get("deck_count", -1)) == 0
        and len(source.get("current_row", [])) == 1
    ):
        next_pick = int(source["current_row"][0])
        next_actor_claims = [
            int(claim["domino_id"])
            for claim in source.get("next_claims", [])
            if int(claim["player"]) == actor
        ]
        return current_and_pending + sorted(next_actor_claims + [next_pick]), next_pick
    raise ValueError("source_is_not_terminal_score_auditable")


def _is_terminal_score_source(source: dict[str, Any]) -> bool:
    return source.get("phase") == Phase.FINAL_PLACEMENT.name or (
        source.get("phase") == Phase.PLACE_AND_SELECT.name
        and int(source.get("deck_count", -1)) == 0
        and len(source.get("current_row", [])) == 1
    )


def _final_score_sequences(
    source: dict[str, Any], actor: int, *, harmony: bool, middle_kingdom: bool
) -> tuple[list[int], int | None, list[tuple[int, TurnAction]]]:
    """Enumerate final scores and first actions for one player's remaining tiles."""
    remaining, next_pick = _remaining_score_audit_dominoes(source, actor)
    if not remaining:
        raise ValueError("missing_remaining_final_claims")

    source_state = state_from_debug_json(source)
    audit_state = GameState(
        config=source_state.config,
        boards=[board.copy() for board in source_state.boards],
        deck=[],
        current_row=[],
        pending_claims=[Claim(actor, domino_id) for domino_id in remaining],
        next_claims=[],
        phase=Phase.FINAL_PLACEMENT,
        actor_index=0,
        start_player=source_state.start_player,
        discards=list(source_state.discards),
    )
    outcomes: list[tuple[int, TurnAction]] = []

    def visit(state: GameState, first_action: TurnAction | None) -> None:
        if state.phase == Phase.GAME_OVER:
            if first_action is None:
                raise ValueError("terminal_sequence_has_no_action")
            outcomes.append(
                (
                    state.boards[actor].score(
                        harmony=harmony,
                        middle_kingdom=middle_kingdom,
                    ).total,
                    first_action,
                )
            )
            return
        for action in state.legal_actions():
            if not isinstance(action, TurnAction):
                raise TypeError("expected_final_turn_action")
            visit(state.step(action), action if first_action is None else first_action)

    visit(audit_state, None)
    return remaining, next_pick, outcomes


def _infer_game_scoring_rules(
    decisions: list[dict[str, Any]], final_scores: dict[int, int]
) -> set[tuple[bool, bool]]:
    """Infer BGA bonus options from terminal boards and authoritative scores."""
    candidates = {(harmony, middle) for harmony in (False, True) for middle in (False, True)}
    evidence = 0
    for actor, final_score in final_scores.items():
        source: dict[str, Any] | None = None
        for index, record in enumerate(decisions):
            state = record.get("state", {})
            if (
                _is_terminal_score_source(state)
                and state.get("current_actor") is not None
                and int(state["current_actor"]) == actor
                and _board_is_reliable(state, actor)
                and not any(
                    _board_is_reliable(later["state"], actor)
                    for later in decisions[index + 1:]
                )
            ):
                source = state
        if source is None:
            continue
        player_candidates: set[tuple[bool, bool]] = set()
        for harmony, middle in candidates:
            try:
                _remaining, _next_pick, outcomes = _final_score_sequences(
                    source,
                    actor,
                    harmony=harmony,
                    middle_kingdom=middle,
                )
            except (IndexError, KeyError, TypeError, ValueError):
                continue
            if any(score == final_score for score, _first_action in outcomes):
                player_candidates.add((harmony, middle))
        if player_candidates:
            candidates &= player_candidates
            evidence += 1
    return candidates if evidence else set()


def _score_verify_terminal(
    *, table_id: str, split: str, source_index: int, actor: int,
    player_id: str | None, player_name: str | None,
    source_record: dict[str, Any], final_score: int,
    scoring_rules: set[tuple[bool, bool]],
) -> ReconstructionRow:
    source = source_record["state"]
    domino_id: int | None = None
    try:
        domino_id = _claim_domino(source, actor)
        verified: list[
            tuple[
                tuple[bool, bool],
                list[int],
                int | None,
                list[TurnAction],
                int,
            ]
        ] = []
        for harmony, middle_kingdom in scoring_rules:
            remaining, next_pick, outcomes = _final_score_sequences(
                source,
                actor,
                harmony=harmony,
                middle_kingdom=middle_kingdom,
            )
            matching = [
                first_action for score, first_action in outcomes if score == final_score
            ]
            if matching:
                verified.append(
                    (
                        (harmony, middle_kingdom),
                        remaining,
                        next_pick,
                        matching,
                        max(score for score, _first_action in outcomes),
                    )
                )
        if not verified:
            raise ValueError("final_score_not_reachable")
        regrets = {best_score - final_score for _rules, _remaining, _pick, _matches, best_score in verified}
        if len(regrets) != 1:
            raise ValueError("terminal_regret_depends_on_ambiguous_rules")
        remaining = verified[0][1]
        next_pick = verified[0][2]
        best_score = final_score + next(iter(regrets))
        unique_first_actions = {
            action
            for _rules, _remaining, _pick, matching, _best in verified
            for action in matching
        }
        placement = None
        if len(unique_first_actions) == 1:
            first_action = next(iter(unique_first_actions))
            if first_action.placement is not None:
                placement = asdict(first_action.placement)
        matched_rules = {rules for rules, _remaining, _pick, _matching, _best in verified}
        harmony_values = {rules[0] for rules in matched_rules}
        middle_values = {rules[1] for rules in matched_rules}
        harmony = next(iter(harmony_values)) if len(harmony_values) == 1 else None
        middle_kingdom = next(iter(middle_values)) if len(middle_values) == 1 else None
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return _drop_row(
            table_id=table_id, split=split, source_index=source_index,
            target_index=None, actor=actor, player_id=player_id,
            player_name=player_name, source_record=source_record,
            target_record=None, domino_id=domino_id, next_pick=None,
            reason=str(exc),
        )

    return ReconstructionRow(
        table_id=table_id,
        split=split,
        source_decision_index=source_index,
        target_decision_index=None,
        player=actor,
        player_id=player_id,
        player_name=player_name,
        phase=str(source.get("phase")),
        deck_count=int(source.get("deck_count", -1)),
        domino_id=domino_id,
        next_pick_domino_id=next_pick,
        placement=placement,
        status="score_verified_terminal",
        drop_reason=None,
        source_captured_at=source_record.get("captured_at"),
        target_captured_at=None,
        remaining_domino_ids=remaining,
        actual_final_score=final_score,
        best_final_score=best_score,
        final_regret=best_score - final_score,
        matching_final_sequences=sum(len(item[3]) for item in verified),
        scoring_harmony=harmony,
        scoring_middle_kingdom=middle_kingdom,
        matching_scoring_rules=[
            f"harmony={harmony_value},middle_kingdom={middle_value}"
            for harmony_value, middle_value in sorted(matched_rules)
        ],
        first_action_unique=len(unique_first_actions) == 1,
    )


def _drop_row(
    *, table_id: str, split: str, source_index: int, target_index: int | None,
    actor: int, player_id: str | None, player_name: str | None,
    source_record: dict[str, Any], target_record: dict[str, Any] | None,
    domino_id: int | None, next_pick: int | None, reason: str,
) -> ReconstructionRow:
    state_json = source_record["state"]
    return ReconstructionRow(
        table_id=table_id,
        split=split,
        source_decision_index=source_index,
        target_decision_index=target_index,
        player=actor,
        player_id=player_id,
        player_name=player_name,
        phase=str(state_json.get("phase")),
        deck_count=int(state_json.get("deck_count", -1)),
        domino_id=domino_id,
        next_pick_domino_id=next_pick,
        placement=None,
        status="dropped",
        drop_reason=reason,
        source_captured_at=source_record.get("captured_at"),
        target_captured_at=None if target_record is None else target_record.get("captured_at"),
    )


def reconstruct_decision(
    *, table_id: str, split: str, decisions: list[dict[str, Any]],
    source_index: int, player_metadata: dict[int, tuple[str | None, str | None]] | None = None,
    final_scores: dict[int, int] | None = None,
    scoring_rules: set[tuple[bool, bool]] | None = None,
) -> ReconstructionRow:
    """Reconstruct one placement and verify it through ``GameState.step``."""
    source_record = decisions[source_index]
    source = source_record["state"]
    actor = int(source["current_actor"])
    player_id, player_name = (player_metadata or {}).get(actor, (None, None))
    domino_id: int | None = None
    next_pick: int | None = None

    if not _board_is_reliable(source, actor):
        return _drop_row(
            table_id=table_id, split=split, source_index=source_index,
            target_index=None, actor=actor, player_id=player_id,
            player_name=player_name, source_record=source_record,
            target_record=None, domino_id=None, next_pick=None,
            reason="source_board_unreliable",
        )

    target_index = next(
        (
            index
            for index in range(source_index + 1, len(decisions))
            if _board_is_reliable(decisions[index]["state"], actor)
        ),
        None,
    )
    if target_index is None:
        if (
            _is_terminal_score_source(source)
            and actor in (final_scores or {})
            and bool(scoring_rules)
        ):
            return _score_verify_terminal(
                table_id=table_id,
                split=split,
                source_index=source_index,
                actor=actor,
                player_id=player_id,
                player_name=player_name,
                source_record=source_record,
                final_score=(final_scores or {})[actor],
                scoring_rules=scoring_rules or set(),
            )
        return _drop_row(
            table_id=table_id, split=split, source_index=source_index,
            target_index=None, actor=actor, player_id=player_id,
            player_name=player_name, source_record=source_record,
            target_record=None, domino_id=None, next_pick=None,
            reason="no_later_reliable_snapshot",
        )
    target_record = decisions[target_index]
    target = target_record["state"]

    try:
        domino_id = _claim_domino(source, actor)
        if source.get("phase") == Phase.PLACE_AND_SELECT.name:
            next_pick = _infer_next_pick(
                source,
                (decisions[index]["state"] for index in range(source_index + 1, target_index + 1)),
                actor,
            )
        source_cells = _board_cells(source, actor)
        target_cells = _board_cells(target, actor)
    except (KeyError, TypeError, ValueError) as exc:
        return _drop_row(
            table_id=table_id, split=split, source_index=source_index,
            target_index=target_index, actor=actor, player_id=player_id,
            player_name=player_name, source_record=source_record,
            target_record=target_record, domino_id=domino_id, next_pick=next_pick,
            reason=str(exc),
        )

    source_observable = _observable_cells(source_cells)
    target_observable = _observable_cells(target_cells)
    if any(target_observable.get(coord) != value for coord, value in source_observable.items()):
        return _drop_row(
            table_id=table_id, split=split, source_index=source_index,
            target_index=target_index, actor=actor, player_id=player_id,
            player_name=player_name, source_record=source_record,
            target_record=target_record, domino_id=domino_id, next_pick=next_pick,
            reason="non_additive_board_delta",
        )
    added = set(target_cells) - set(source_cells)
    if len(added) not in {0, 2}:
        reason = "ambiguous_multi_action_delta" if len(added) > 2 else "invalid_cell_delta"
        return _drop_row(
            table_id=table_id, split=split, source_index=source_index,
            target_index=target_index, actor=actor, player_id=player_id,
            player_name=player_name, source_record=source_record,
            target_record=target_record, domino_id=domino_id, next_pick=next_pick,
            reason=reason,
        )

    try:
        state = state_from_debug_json(source)
        if int(state.current_actor) != actor:
            raise ValueError("engine_actor_mismatch")
        if domino_id not in DOMINOES:
            raise ValueError("unknown_domino")
        legal_placements = state.boards[actor].legal_placements(DOMINOES[domino_id])
        if not added:
            if legal_placements:
                raise ValueError("zero_delta_but_placement_available")
            action = TurnAction(None, next_pick)
            child = state.step(action)
            if _observable_board_fingerprint(child.boards[actor]) != target_observable:
                raise ValueError("step_board_mismatch")
            return ReconstructionRow(
                table_id=table_id, split=split,
                source_decision_index=source_index, target_decision_index=target_index,
                player=actor, player_id=player_id, player_name=player_name,
                phase=str(source.get("phase")), deck_count=int(source.get("deck_count", -1)),
                domino_id=domino_id, next_pick_domino_id=next_pick,
                placement=None, status="forced_discard", drop_reason=None,
                source_captured_at=source_record.get("captured_at"),
                target_captured_at=target_record.get("captured_at"),
            )

        if any(target_cells[coord][2] > 0 and target_cells[coord][2] != domino_id for coord in added):
            raise ValueError("delta_domino_id_mismatch")
        matches: list[tuple[Placement, Any]] = []
        for placement in legal_placements:
            action = TurnAction(placement, next_pick)
            child = state.step(action)
            if _observable_board_fingerprint(child.boards[actor]) == target_observable:
                matches.append((placement, child))
        if not matches:
            raise ValueError("no_legal_action_matches_snapshot")
        if len(matches) != 1:
            raise ValueError("ambiguous_legal_placement")
        placement = matches[0][0]
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return _drop_row(
            table_id=table_id, split=split, source_index=source_index,
            target_index=target_index, actor=actor, player_id=player_id,
            player_name=player_name, source_record=source_record,
            target_record=target_record, domino_id=domino_id, next_pick=next_pick,
            reason=str(exc),
        )

    return ReconstructionRow(
        table_id=table_id,
        split=split,
        source_decision_index=source_index,
        target_decision_index=target_index,
        player=actor,
        player_id=player_id,
        player_name=player_name,
        phase=str(source.get("phase")),
        deck_count=int(source.get("deck_count", -1)),
        domino_id=domino_id,
        next_pick_domino_id=next_pick,
        placement=asdict(placement),
        status="reconstructed",
        drop_reason=None,
        source_captured_at=source_record.get("captured_at"),
        target_captured_at=target_record.get("captured_at"),
    )


def reconstruct_game(path: Path, table_id: str, split: str) -> list[ReconstructionRow]:
    records = _read_jsonl(path)
    decisions = [record for record in records if record.get("kind") == "decision"]
    metadata = _player_metadata(records, decisions)
    final_scores = _final_scores_by_engine(records, metadata)
    scoring_rules = _infer_game_scoring_rules(decisions, final_scores)
    rows: list[ReconstructionRow] = []
    for index, record in enumerate(decisions):
        state = record.get("state", {})
        if state.get("phase") not in {Phase.PLACE_AND_SELECT.name, Phase.FINAL_PLACEMENT.name}:
            continue
        if state.get("current_actor") is None:
            continue
        rows.append(
            reconstruct_decision(
                table_id=table_id,
                split=split,
                decisions=decisions,
                source_index=index,
                player_metadata=metadata,
                final_scores=final_scores,
                scoring_rules=scoring_rules,
            )
        )
    return rows


def _placement_layer(state: dict[str, Any]) -> int | None:
    phase = state.get("phase")
    if phase == Phase.FINAL_PLACEMENT.name:
        return 11
    if phase != Phase.PLACE_AND_SELECT.name:
        return None
    deck_count = int(state.get("deck_count", -1))
    if 0 <= deck_count <= 40 and deck_count % 4 == 0:
        return (40 - deck_count) // 4
    return None


def _claim_sequence_evidence(
    decisions: list[dict[str, Any]],
) -> tuple[
    dict[int, set[int]],
    dict[int, set[int]],
    dict[int, set[int]],
    list[str],
]:
    """Collect layer tiles and tile-owner evidence from all public claims."""
    layer_tiles: dict[int, set[int]] = defaultdict(set)
    tile_layers: dict[int, set[int]] = defaultdict(set)
    tile_owners: dict[int, set[int]] = defaultdict(set)
    issues: list[str] = []

    def observe(tile: Any, layer: int, owner: Any | None = None) -> None:
        domino_id = int(tile)
        if domino_id not in DOMINOES:
            issues.append(f"unknown_domino:{domino_id}")
            return
        if not 0 <= layer < 12:
            issues.append(f"invalid_layer:{layer}:d{domino_id}")
            return
        layer_tiles[layer].add(domino_id)
        tile_layers[domino_id].add(layer)
        if owner is not None:
            tile_owners[domino_id].add(int(owner))

    for record in decisions:
        state = record.get("state", {})
        phase = state.get("phase")
        if (
            phase == Phase.INITIAL_SELECTION.name
            and int(state.get("deck_count", -1)) == 44
        ):
            for domino_id in state.get("current_row", []):
                observe(domino_id, 0)
            for claim in state.get("next_claims", []):
                observe(claim["domino_id"], 0, claim["player"])
            continue

        layer = _placement_layer(state)
        if layer is None:
            continue
        for claim in state.get("pending_claims", []):
            observe(claim["domino_id"], layer, claim["player"])
        if phase == Phase.PLACE_AND_SELECT.name and layer < 11:
            for domino_id in state.get("current_row", []):
                observe(domino_id, layer + 1)
            for claim in state.get("next_claims", []):
                observe(claim["domino_id"], layer + 1, claim["player"])

    return layer_tiles, tile_layers, tile_owners, sorted(set(issues))


def _resolve_claim_sequences(
    decisions: list[dict[str, Any]],
) -> tuple[dict[int, list[list[int]]], dict[int, list[str]], list[str]]:
    layer_tiles, tile_layers, tile_owners, global_issues = _claim_sequence_evidence(decisions)
    missing_before_resolution = sorted(set(DOMINOES) - set(tile_layers))
    deficient_layers = [
        (layer, 4 - len(layer_tiles.get(layer, set())))
        for layer in range(12)
        if len(layer_tiles.get(layer, set())) < 4
    ]
    # If exactly one canonical tile and one layer slot were missed before the
    # logger attached, 48-tile conservation identifies that tile and layer
    # without using placement results or scores.
    if (
        len(missing_before_resolution) == 1
        and len(deficient_layers) == 1
        and deficient_layers[0][1] == 1
    ):
        missing_tile = missing_before_resolution[0]
        missing_layer = deficient_layers[0][0]
        layer_tiles[missing_layer].add(missing_tile)
        tile_layers[missing_tile].add(missing_layer)
    owner_by_tile: dict[int, int] = {}
    for domino_id, owners in tile_owners.items():
        if len(owners) == 1:
            owner_by_tile[domino_id] = next(iter(owners))
        elif len(owners) > 1:
            global_issues.append(
                f"owner_conflict:d{domino_id}:{','.join(map(str, sorted(owners)))}"
            )
    for domino_id, layers in tile_layers.items():
        if len(layers) > 1:
            global_issues.append(
                f"layer_conflict:d{domino_id}:{','.join(map(str, sorted(layers)))}"
            )

    # A Mighty Duel layer contains exactly two claims per player.  Infer an
    # unobserved owner only when the two-per-player constraint forces it.
    for layer in range(12):
        tiles = layer_tiles.get(layer, set())
        if len(tiles) != 4:
            continue
        counts = Counter(owner_by_tile[tile] for tile in tiles if tile in owner_by_tile)
        unknown = sorted(tile for tile in tiles if tile not in owner_by_tile)
        need = {player: 2 - counts[player] for player in (0, 1)}
        if any(value < 0 for value in need.values()) or sum(need.values()) != len(unknown):
            continue
        if need[0] == 0:
            owner_by_tile.update({tile: 1 for tile in unknown})
        elif need[1] == 0:
            owner_by_tile.update({tile: 0 for tile in unknown})
        elif len(unknown) == 1:
            forced = 0 if need[0] == 1 else 1
            owner_by_tile[unknown[0]] = forced

    by_player = {0: [[] for _ in range(12)], 1: [[] for _ in range(12)]}
    player_issues = {0: [], 1: []}
    for layer in range(12):
        tiles = layer_tiles.get(layer, set())
        if len(tiles) != 4:
            for player in (0, 1):
                player_issues[player].append(f"layer_{layer}_tile_count:{len(tiles)}")
        for player in (0, 1):
            owned = sorted(tile for tile in tiles if owner_by_tile.get(tile) == player)
            by_player[player][layer] = owned
            if len(owned) != 2:
                player_issues[player].append(f"layer_{layer}_owner_count:{len(owned)}")

    missing_tiles = sorted(set(DOMINOES) - set(tile_layers))
    if missing_tiles:
        global_issues.append("missing_tiles:" + ",".join(map(str, missing_tiles)))
    extra_layer_tiles = sum(len(tiles) for tiles in layer_tiles.values())
    if extra_layer_tiles != 48:
        global_issues.append(f"observed_layer_tile_total:{extra_layer_tiles}")
    return by_player, player_issues, sorted(set(global_issues))


def audit_game_completeness(
    path: Path, table_id: str, split: str
) -> list[CompletenessRow]:
    records = _read_jsonl(path)
    decisions = [record for record in records if record.get("kind") == "decision"]
    metadata = _player_metadata(records, decisions)
    final_scores = _final_scores_by_engine(records, metadata)
    scoring_rules = _infer_game_scoring_rules(decisions, final_scores)
    sequences, player_issues, global_issues = _resolve_claim_sequences(decisions)
    reconstruction = reconstruct_game(path, table_id, split)

    rule_labels = [
        f"harmony={harmony},middle_kingdom={middle}"
        for harmony, middle in sorted(scoring_rules)
    ]
    rows: list[CompletenessRow] = []
    for player in (0, 1):
        layers = sequences[player]
        ordered = [domino_id for layer in layers for domino_id in layer]
        issues = list(global_issues) + list(player_issues[player])
        if len(ordered) != 24:
            issues.append(f"player_tile_count:{len(ordered)}")
        if len(set(ordered)) != len(ordered):
            issues.append("duplicate_player_tiles")
        if player not in final_scores:
            issues.append("missing_final_score")
        if len(scoring_rules) != 1:
            issues.append(f"scoring_rules_candidate_count:{len(scoring_rules)}")
        sequence_complete = (
            len(ordered) == 24
            and len(set(ordered)) == 24
            and all(len(layer) == 2 for layer in layers)
            and not global_issues
        )
        eligible = sequence_complete and player in final_scores and len(scoring_rules) == 1
        player_reconstruction = [row for row in reconstruction if row.player == player]
        verified = sum(row.kept for row in player_reconstruction)
        player_id, player_name = metadata.get(player, (None, None))
        rows.append(
            CompletenessRow(
                table_id=table_id,
                split=split,
                player=player,
                player_id=player_id,
                player_name=player_name,
                final_score=final_scores.get(player),
                ordered_domino_ids=ordered,
                dominoes_by_layer=layers,
                tile_count=len(ordered),
                complete_layers=sum(len(layer) == 2 for layer in layers),
                sequence_complete=sequence_complete,
                scoring_rule_candidates=rule_labels,
                scoring_rules_unique=len(scoring_rules) == 1,
                whole_game_score_eligible=eligible,
                captured_placement_decisions=len(player_reconstruction),
                verified_placement_decisions=verified,
                decision_regret_eligible=verified if eligible else 0,
                issues=sorted(set(issues)),
            )
        )
    return rows


def _solver_board_key(board: Board) -> tuple[tuple[int, int, int, int], ...]:
    """Behavioral board identity; domino IDs do not affect legality or score."""
    return tuple(
        (
            x,
            y,
            int(board.terrain[y, x]),
            int(board.crowns[y, x]),
        )
        for x, y in board.occupied_cells()
    )


def _beam_rank(
    node: _BeamNode,
    *,
    final_layer: bool,
    harmony: bool,
    middle_kingdom: bool,
) -> tuple[float, int, int]:
    board = node.board
    score = board.score(
        harmony=harmony if final_layer else False,
        middle_kingdom=middle_kingdom if final_layer else False,
    )
    if final_layer:
        return float(score.total), -node.discards, len(board._frontier())

    min_x, min_y, max_x, max_y = board.occupied_bbox() or (7, 7, 7, 7)
    bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
    holes = bbox_area - len(board._occupied)
    frontier = len(board._frontier())
    # Territory score is the strongest available non-clairvoyant partial
    # signal.  Compactness and frontier size break ties toward boards that
    # retain placement flexibility; both are deliberately small relative to a
    # score point so beam-width convergence, not the heuristic, is the arbiter.
    rank = float(score.territory_score) - 0.01 * holes + 0.001 * frontier
    return rank, -node.discards, frontier


def _solve_fixed_tile_sequence_rust(
    sequence: list[int],
    *,
    harmony: bool,
    middle_kingdom: bool,
    beam_width: int,
) -> FixedSequenceSolveResult | None:
    """Use the production Rust board engine when the audit entry point exists."""
    try:
        import kingdomino_rust
    except ImportError:
        return None
    if not hasattr(kingdomino_rust, "fixed_sequence_beam"):
        return None

    (
        best_score,
        territory_score,
        harmony_bonus,
        middle_bonus,
        discards,
        raw_placements,
        raw_stats,
        elapsed_seconds,
    ) = kingdomino_rust.fixed_sequence_beam(
        sequence,
        harmony,
        middle_kingdom,
        beam_width,
    )
    placements = [
        None
        if placement is None
        else {
            "x1": int(placement[0]),
            "y1": int(placement[1]),
            "x2": int(placement[2]),
            "y2": int(placement[3]),
            "flipped": bool(placement[4]),
        }
        for placement in raw_placements
    ]
    layer_stats = [
        BeamLayerStats(
            layer=int(stats[0]),
            domino_id=int(stats[1]),
            incoming_states=int(stats[2]),
            expanded_actions=int(stats[3]),
            unique_states=int(stats[4]),
            kept_states=int(stats[5]),
            forced_discard_parents=int(stats[6]),
            best_partial_score=int(stats[7]),
            elapsed_seconds=float(stats[8]),
        )
        for stats in raw_stats
    ]
    return FixedSequenceSolveResult(
        domino_ids=sequence,
        beam_width=beam_width,
        best_score=int(best_score),
        territory_score=int(territory_score),
        harmony_bonus=int(harmony_bonus),
        middle_kingdom_bonus=int(middle_bonus),
        discards=int(discards),
        placements=placements,
        layer_stats=layer_stats,
        elapsed_seconds=float(elapsed_seconds),
        backend="rust",
    )


def solve_fixed_tile_sequence(
    domino_ids: Iterable[int],
    *,
    harmony: bool,
    middle_kingdom: bool,
    beam_width: int | None,
    exact_state_limit: int | None = None,
    backend: str = "auto",
) -> FixedSequenceSolveResult:
    """Optimize placements for a fixed ordered tile sequence.

    ``beam_width=None`` retains every unique board state and is exact unless
    ``exact_state_limit`` raises :class:`ExactStateLimitExceeded`.
    """
    sequence = [int(domino_id) for domino_id in domino_ids]
    if not sequence:
        raise ValueError("domino_ids must not be empty")
    if any(domino_id not in DOMINOES for domino_id in sequence):
        raise ValueError("domino_ids contains an unknown tile")
    if beam_width is not None and beam_width <= 0:
        raise ValueError("beam_width must be positive or None")
    if exact_state_limit is not None and exact_state_limit <= 0:
        raise ValueError("exact_state_limit must be positive or None")
    if backend not in {"auto", "python", "rust"}:
        raise ValueError("backend must be auto, python, or rust")

    if beam_width is not None and backend != "python":
        rust_result = _solve_fixed_tile_sequence_rust(
            sequence,
            harmony=harmony,
            middle_kingdom=middle_kingdom,
            beam_width=beam_width,
        )
        if rust_result is not None:
            return rust_result
        if backend == "rust":
            raise RuntimeError("kingdomino_rust.fixed_sequence_beam is unavailable")

    started = time.perf_counter()
    initial = _BeamNode(Board(), 0, ())
    nodes: dict[tuple[tuple[int, int, int, int], ...], _BeamNode] = {
        _solver_board_key(initial.board): initial
    }
    layer_stats: list[BeamLayerStats] = []

    for layer_index, domino_id in enumerate(sequence):
        layer_started = time.perf_counter()
        incoming = len(nodes)
        expanded_actions = 0
        forced_discard_parents = 0
        next_nodes: dict[tuple[tuple[int, int, int, int], ...], _BeamNode] = {}
        domino = DOMINOES[domino_id]
        for node in nodes.values():
            placements = node.board.legal_placements(domino)
            if not placements:
                forced_discard_parents += 1
                expanded_actions += 1
                child = _BeamNode(
                    board=node.board,
                    discards=node.discards + 1,
                    placements=node.placements + (None,),
                )
                next_nodes.setdefault(_solver_board_key(child.board), child)
            else:
                expanded_actions += len(placements)
                for placement in placements:
                    board = node.board.copy()
                    board.place(domino, placement)
                    child = _BeamNode(
                        board=board,
                        discards=node.discards,
                        placements=node.placements + (placement,),
                    )
                    next_nodes.setdefault(_solver_board_key(board), child)
            if (
                beam_width is None
                and exact_state_limit is not None
                and len(next_nodes) > exact_state_limit
            ):
                raise ExactStateLimitExceeded(
                    layer=layer_index + 1,
                    unique_states=len(next_nodes),
                    limit=exact_state_limit,
                )

        unique_states = len(next_nodes)
        final_layer = layer_index == len(sequence) - 1
        if beam_width is not None and unique_states > beam_width:
            kept = heapq.nlargest(
                beam_width,
                next_nodes.values(),
                key=lambda node: _beam_rank(
                    node,
                    final_layer=final_layer,
                    harmony=harmony,
                    middle_kingdom=middle_kingdom,
                ),
            )
            nodes = {_solver_board_key(node.board): node for node in kept}
        else:
            nodes = next_nodes
        best_partial = max(
            node.board.score(harmony=False, middle_kingdom=False).territory_score
            for node in nodes.values()
        )
        layer_stats.append(
            BeamLayerStats(
                layer=layer_index + 1,
                domino_id=domino_id,
                incoming_states=incoming,
                expanded_actions=expanded_actions,
                unique_states=unique_states,
                kept_states=len(nodes),
                forced_discard_parents=forced_discard_parents,
                best_partial_score=best_partial,
                elapsed_seconds=time.perf_counter() - layer_started,
            )
        )

    best = max(
        nodes.values(),
        key=lambda node: (
            node.board.score(harmony=harmony, middle_kingdom=middle_kingdom).total,
            -node.discards,
        ),
    )
    score = best.board.score(harmony=harmony, middle_kingdom=middle_kingdom)
    return FixedSequenceSolveResult(
        domino_ids=sequence,
        beam_width=beam_width,
        best_score=score.total,
        territory_score=score.territory_score,
        harmony_bonus=score.harmony_bonus,
        middle_kingdom_bonus=score.middle_kingdom_bonus,
        discards=best.discards,
        placements=[None if placement is None else asdict(placement) for placement in best.placements],
        layer_stats=layer_stats,
        elapsed_seconds=time.perf_counter() - started,
    )


def _group_summary(
    rows: Iterable[ReconstructionRow], *, expected_decisions: int | None = None
) -> dict[str, Any]:
    materialized = list(rows)
    kept = sum(row.kept for row in materialized)
    reasons = Counter(row.drop_reason for row in materialized if row.drop_reason)
    statuses = Counter(row.status for row in materialized)
    reference_available = len(materialized) - reasons["no_later_reliable_snapshot"]
    summary = {
        "attempted": len(materialized),
        "kept": kept,
        "yield": kept / len(materialized) if materialized else None,
        "reference_available": reference_available,
        "yield_given_reference": kept / reference_available if reference_available else None,
        "statuses": dict(sorted(statuses.items())),
        "drop_reasons": dict(sorted(reasons.items())),
    }
    if expected_decisions is not None:
        summary.update(
            {
                "expected_decisions": expected_decisions,
                "source_capture_rate": (
                    len(materialized) / expected_decisions if expected_decisions else None
                ),
                "end_to_end_yield": kept / expected_decisions if expected_decisions else None,
            }
        )
    return summary


def build_summary(rows: list[ReconstructionRow]) -> dict[str, Any]:
    by_game: dict[str, list[ReconstructionRow]] = defaultdict(list)
    by_game_player: dict[tuple[str, int], list[ReconstructionRow]] = defaultdict(list)
    by_player: dict[int, list[ReconstructionRow]] = defaultdict(list)
    by_split: dict[str, list[ReconstructionRow]] = defaultdict(list)
    for row in rows:
        by_game[row.table_id].append(row)
        by_game_player[(row.table_id, row.player)].append(row)
        by_player[row.player].append(row)
        by_split[row.split].append(row)
    terminal_rows = [row for row in rows if row.status == "score_verified_terminal"]

    def terminal_summary(values: list[ReconstructionRow]) -> dict[str, Any]:
        regrets = [int(row.final_regret) for row in values if row.final_regret is not None]
        return {
            "score_verified": len(values),
            "zero_regret": sum(regret == 0 for regret in regrets),
            "positive_regret": sum(regret > 0 for regret in regrets),
            "mean_regret": sum(regrets) / len(regrets) if regrets else None,
            "max_regret": max(regrets) if regrets else None,
            "unique_first_action": sum(row.first_action_unique is True for row in values),
        }

    return {
        "schema": "kingdomino-placement-reconstruction-summary/v1",
        "overall": _group_summary(rows, expected_decisions=48 * len(by_game)),
        "by_split": {
            key: _group_summary(
                value,
                expected_decisions=48 * len({row.table_id for row in value}),
            )
            for key, value in sorted(by_split.items())
        },
        "by_game": {
            key: _group_summary(value, expected_decisions=48)
            for key, value in sorted(by_game.items())
        },
        "by_player": {
            f"p{player}": _group_summary(
                value,
                expected_decisions=24 * len({row.table_id for row in value}),
            )
            for player, value in sorted(by_player.items())
        },
        "by_game_player": {
            f"{table_id}:p{player}": _group_summary(value, expected_decisions=24)
            for (table_id, player), value in sorted(by_game_player.items())
        },
        "terminal_score_verification": {
            "overall": terminal_summary(terminal_rows),
            "by_player": {
                f"p{player}": terminal_summary(
                    [row for row in terminal_rows if row.player == player]
                )
                for player in sorted({row.player for row in terminal_rows})
            },
        },
    }


def run_reconstruction(
    manifest_path: Path,
    output_dir: Path,
    split: str,
    verify_hashes: bool = True,
) -> tuple[list[ReconstructionRow], dict[str, Any]]:
    manifest = _load_manifest(manifest_path)
    games = manifest.get("games", [])
    selected = games if split == "all" else [game for game in games if game["split"] == split]
    rows: list[ReconstructionRow] = []
    for game in selected:
        path = _resolve_game_path(manifest_path, game["path"])
        if verify_hashes:
            _verify_sha256(path, game["sha256"])
        rows.extend(reconstruct_game(path, str(game["table_id"]), str(game["split"])))

    summary = build_summary(rows)
    summary.update(
        {
            "manifest": str(manifest_path),
            "corpus_id": manifest.get("corpus_id"),
            "selected_split": split,
            "games": len(selected),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / f"reconstruction_decisions_{split}.jsonl"
    decisions_path.write_text(
        "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_path = output_dir / f"reconstruction_summary_{split}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows, summary


def build_completeness_summary(rows: list[CompletenessRow]) -> dict[str, Any]:
    games = sorted({row.table_id for row in rows})
    issue_counts = Counter(issue for row in rows for issue in row.issues)

    def group(values: list[CompletenessRow]) -> dict[str, Any]:
        return {
            "players": len(values),
            "sequence_complete": sum(row.sequence_complete for row in values),
            "unique_scoring_rules": sum(row.scoring_rules_unique for row in values),
            "whole_game_score_eligible": sum(
                row.whole_game_score_eligible for row in values
            ),
            "captured_placement_decisions": sum(
                row.captured_placement_decisions for row in values
            ),
            "verified_placement_decisions": sum(
                row.verified_placement_decisions for row in values
            ),
            "decision_regret_eligible": sum(
                row.decision_regret_eligible for row in values
            ),
        }

    by_player = {
        f"p{player}": group([row for row in rows if row.player == player])
        for player in sorted({row.player for row in rows})
    }
    by_split = {
        split: group([row for row in rows if row.split == split])
        for split in sorted({row.split for row in rows})
    }
    paired_eligible = [
        table_id
        for table_id in games
        if all(
            row.whole_game_score_eligible
            for row in rows
            if row.table_id == table_id
        )
    ]
    opponent_eligible = [
        row.table_id
        for row in rows
        if row.player == 1 and row.whole_game_score_eligible
    ]
    development_paired = [
        table_id
        for table_id in paired_eligible
        if any(row.table_id == table_id and row.split == "development" for row in rows)
    ]
    feasibility_ranked = sorted(
        development_paired,
        key=lambda table_id: hashlib.sha256(
            f"placement-solver-feasibility-v1:{table_id}".encode("utf-8")
        ).hexdigest(),
    )
    return {
        "schema": "kingdomino-placement-completeness-summary/v1",
        "games": len(games),
        "players": len(rows),
        "overall": group(rows),
        "by_player": by_player,
        "by_split": by_split,
        "paired_whole_game_score_eligible_games": paired_eligible,
        "paired_whole_game_score_eligible_count": len(paired_eligible),
        "opponent_whole_game_score_eligible_games": opponent_eligible,
        "opponent_whole_game_score_eligible_count": len(opponent_eligible),
        "feasibility_probe_v1": {
            "selection": (
                "Lowest five SHA-256 ranks of "
                "'placement-solver-feasibility-v1:' + table_id among paired, "
                "development-eligible games"
            ),
            "table_ids": feasibility_ranked[:5],
        },
        "issue_counts": dict(sorted(issue_counts.items())),
    }


def run_completeness_audit(
    manifest_path: Path,
    output_dir: Path,
    split: str,
    verify_hashes: bool = True,
) -> tuple[list[CompletenessRow], dict[str, Any]]:
    manifest = _load_manifest(manifest_path)
    games = manifest.get("games", [])
    selected = games if split == "all" else [game for game in games if game["split"] == split]
    rows: list[CompletenessRow] = []
    for game in selected:
        path = _resolve_game_path(manifest_path, game["path"])
        if verify_hashes:
            _verify_sha256(path, game["sha256"])
        rows.extend(audit_game_completeness(path, str(game["table_id"]), str(game["split"])))

    summary = build_completeness_summary(rows)
    summary.update(
        {
            "manifest": str(manifest_path),
            "corpus_id": manifest.get("corpus_id"),
            "selected_split": split,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / f"sequence_completeness_{split}.jsonl"
    rows_path.write_text(
        "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_path = output_dir / f"sequence_completeness_summary_{split}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows, summary


def run_solver_probe(
    manifest_path: Path,
    output_path: Path,
    table_ids: Iterable[str],
    players: Iterable[int],
    beam_widths: Iterable[int],
    exact_state_limit: int | None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    table_ids = [str(table_id) for table_id in table_ids]
    players = [int(player) for player in players]
    manifest = _load_manifest(manifest_path)
    games_by_id = {str(game["table_id"]): game for game in manifest.get("games", [])}
    widths = sorted(set(int(width) for width in beam_widths))
    if not widths or any(width <= 0 for width in widths):
        raise ValueError("beam_widths must contain positive integers")

    cases: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_checkpoint() -> None:
        checkpoint = {
            "schema": "kingdomino-fixed-sequence-solver-probe/v1",
            "manifest": str(manifest_path),
            "table_ids": table_ids,
            "players": players,
            "beam_widths": widths,
            "exact_state_limit": exact_state_limit,
            "cases": cases,
            "completed_cases": len(cases),
            "expected_cases": len(table_ids) * len(players),
            "all_widest_two_agree": bool(cases)
            and all(case["widest_two_agree"] for case in cases),
        }
        output_path.write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    for table_id in table_ids:
        if table_id not in games_by_id:
            raise ValueError(f"table {table_id} is not in the frozen corpus manifest")
        game = games_by_id[table_id]
        path = _resolve_game_path(manifest_path, game["path"])
        if verify_hashes:
            _verify_sha256(path, game["sha256"])
        completeness = audit_game_completeness(path, table_id, str(game["split"]))
        by_player = {row.player: row for row in completeness}
        for player in players:
            row = by_player[int(player)]
            if not row.whole_game_score_eligible:
                raise ValueError(
                    f"table {table_id} player {player} is not whole-game-score eligible: "
                    + "; ".join(row.issues)
                )
            rule = row.scoring_rule_candidates[0]
            harmony = "harmony=True" in rule
            middle_kingdom = "middle_kingdom=True" in rule
            case: dict[str, Any] = {
                "table_id": table_id,
                "split": row.split,
                "player": player,
                "player_id": row.player_id,
                "player_name": row.player_name,
                "actual_score": row.final_score,
                "domino_ids": row.ordered_domino_ids,
                "harmony": harmony,
                "middle_kingdom": middle_kingdom,
                "exact_dp": None,
                "beams": [],
            }
            if exact_state_limit is not None:
                print(
                    f"solver-probe table={table_id} player={player} exact_limit="
                    f"{exact_state_limit}",
                    flush=True,
                )
                exact_started = time.perf_counter()
                try:
                    exact = solve_fixed_tile_sequence(
                        row.ordered_domino_ids,
                        harmony=harmony,
                        middle_kingdom=middle_kingdom,
                        beam_width=None,
                        exact_state_limit=exact_state_limit,
                    )
                    case["exact_dp"] = {
                        "status": "completed",
                        "result": asdict(exact),
                    }
                except ExactStateLimitExceeded as exc:
                    case["exact_dp"] = {
                        "status": "state_limit_exceeded",
                        "limit": exc.limit,
                        "layer": exc.layer,
                        "unique_states": exc.unique_states,
                        "elapsed_seconds": time.perf_counter() - exact_started,
                    }
            for width in widths:
                print(
                    f"solver-probe table={table_id} player={player} beam={width}",
                    flush=True,
                )
                solved = solve_fixed_tile_sequence(
                    row.ordered_domino_ids,
                    harmony=harmony,
                    middle_kingdom=middle_kingdom,
                    beam_width=width,
                )
                solved_json = asdict(solved)
                solved_json["headroom_vs_actual"] = (
                    None if row.final_score is None else solved.best_score - row.final_score
                )
                case["beams"].append(solved_json)
            widest_score = case["beams"][-1]["best_score"]
            for beam in case["beams"]:
                beam["score_gap_to_widest"] = widest_score - beam["best_score"]
            case["widest_score"] = widest_score
            case["widest_two_agree"] = (
                len(case["beams"]) >= 2
                and case["beams"][-2]["best_score"] == widest_score
            )
            cases.append(case)
            write_checkpoint()
            print(
                f"solver-probe completed table={table_id} player={player} "
                f"scores={[beam['best_score'] for beam in case['beams']]}",
                flush=True,
            )

    report = {
        "schema": "kingdomino-fixed-sequence-solver-probe/v1",
        "manifest": str(manifest_path),
        "table_ids": table_ids,
        "players": players,
        "beam_widths": widths,
        "exact_state_limit": exact_state_limit,
        "cases": cases,
        "all_widest_two_agree": all(case["widest_two_agree"] for case in cases),
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    reconstruct = subparsers.add_parser(
        "reconstruct",
        help="verify placements from BGA snapshots and terminal suffixes from final scores",
    )
    reconstruct.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    reconstruct.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    reconstruct.add_argument(
        "--split", choices=("development", "confirmation", "all"), default="development"
    )
    reconstruct.add_argument(
        "--no-verify-hashes", action="store_true", help="skip source-corpus SHA-256 checks"
    )
    completeness = subparsers.add_parser(
        "completeness",
        help="audit complete claimed-tile sequences and score-analysis eligibility",
    )
    completeness.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    completeness.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    completeness.add_argument(
        "--split", choices=("development", "confirmation", "all"), default="development"
    )
    completeness.add_argument(
        "--no-verify-hashes", action="store_true", help="skip source-corpus SHA-256 checks"
    )
    probe = subparsers.add_parser(
        "solver-probe",
        help="run exact-state feasibility and a beam-width sweep on frozen development games",
    )
    probe.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    probe.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "solver_feasibility_probe_v1.json",
    )
    probe.add_argument("--table-ids", nargs="*", default=list(FEASIBILITY_PROBE_V1))
    probe.add_argument("--beam-widths", default="256,1024,4096,16384")
    probe.add_argument("--exact-state-limit", type=int, default=50_000)
    probe.add_argument(
        "--players", choices=("opponent", "p0", "p1", "both"), default="opponent"
    )
    probe.add_argument(
        "--no-verify-hashes", action="store_true", help="skip source-corpus SHA-256 checks"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "reconstruct":
        _rows, summary = run_reconstruction(
            args.manifest,
            args.output_dir,
            args.split,
            verify_hashes=not args.no_verify_hashes,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "completeness":
        _rows, summary = run_completeness_audit(
            args.manifest,
            args.output_dir,
            args.split,
            verify_hashes=not args.no_verify_hashes,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "solver-probe":
        player_map = {
            "opponent": [1],
            "p0": [0],
            "p1": [1],
            "both": [0, 1],
        }
        widths = [int(value) for value in args.beam_widths.split(",") if value.strip()]
        report = run_solver_probe(
            args.manifest,
            args.output,
            args.table_ids,
            player_map[args.players],
            widths,
            None if args.exact_state_limit <= 0 else args.exact_state_limit,
            verify_hashes=not args.no_verify_hashes,
        )
        compact = {
            "schema": report["schema"],
            "output": str(args.output),
            "all_widest_two_agree": report["all_widest_two_agree"],
            "cases": [
                {
                    "table_id": case["table_id"],
                    "player": case["player"],
                    "actual_score": case["actual_score"],
                    "exact_dp": case["exact_dp"],
                    "beam_scores": {
                        str(beam["beam_width"]): beam["best_score"]
                        for beam in case["beams"]
                    },
                    "widest_two_agree": case["widest_two_agree"],
                }
                for case in report["cases"]
            ],
        }
        print(json.dumps(compact, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
