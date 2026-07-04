from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from games.kingdomino.encoder import encode_state
from games.kingdomino.game import Phase
from games.kingdomino.web_app import (
    BotActionRequest,
    RecommendRequest,
    action_to_json,
    _load_nn_evaluator,
    _root_trajectory,
    recommend,
    state_from_debug_json,
)


DEFAULT_PROBE_DIR = Path("runs/kingdomino/advisor_probes")
DEFAULT_AUDIT_CSV = Path("runs/kingdomino/advisor_probes/phase1_audit.csv")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def _probe_parts(data: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    capture = data.get("capture") if isinstance(data.get("capture"), dict) else data
    payload = data.get("advisor_payload") or data.get("payload")
    response = data.get("advisor_response") or data.get("response")
    if not isinstance(payload, dict):
        payload = None
    if not isinstance(response, dict):
        response = None
    return capture, payload, response


def _state_json_from_probe(data: dict[str, Any]) -> dict[str, Any]:
    capture, payload, _response = _probe_parts(data)
    if payload and isinstance(payload.get("state"), dict):
        return payload["state"]
    if capture and isinstance(capture.get("state"), dict):
        return capture["state"]
    if isinstance(data.get("state"), dict):
        return data["state"]
    raise ValueError("Probe has no normalized state at advisor_payload.state, capture.state, or state")


def _fmt_score(score: dict[str, Any]) -> str:
    return (
        f"total={score.get('total')} territory={score.get('territory_score')} "
        f"harmony={score.get('harmony_bonus')} middle={score.get('middle_kingdom_bonus')}"
    )


def _board_summary(state_public: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    boards = state_public.get("boards") or []
    breakdowns = state_public.get("score_breakdowns") or []
    for i, board in enumerate(boards):
        bbox = board.get("bbox")
        occupied = board.get("occupied_count")
        score = breakdowns[i] if i < len(breakdowns) else board.get("score", {})
        width = height = None
        if isinstance(bbox, list) and len(bbox) == 4:
            width = int(bbox[2]) - int(bbox[0]) + 1
            height = int(bbox[3]) - int(bbox[1]) + 1
        missing_full = 49 - int(occupied or 0)
        lines.append(
            f"  P{i}: cells={occupied} bbox={bbox} size={width}x{height} "
            f"missing_full={missing_full} score({_fmt_score(score)})"
        )
    return lines


def _top_recommendations(response: dict[str, Any] | None, limit: int) -> list[str]:
    if not response:
        return ["  <no saved advisor response>"]
    recs = response.get("recommendations")
    if not isinstance(recs, list) or not recs:
        return ["  <no recommendations>"]
    lines = []
    for rec in recs[:limit]:
        label = rec.get("label") or rec.get("action_id") or "<unknown action>"
        bits = []
        if isinstance(rec.get("visit_frac"), (int, float)):
            bits.append(f"visit={float(rec['visit_frac']):.3f}")
        if isinstance(rec.get("q_win_prob"), (int, float)):
            bits.append(f"win={float(rec['q_win_prob']):.3f}")
        if isinstance(rec.get("q_rank_value"), (int, float)):
            bits.append(f"rank={float(rec['q_rank_value']):.3f}")
        lines.append(f"  #{rec.get('rank', '?')}: {label} {' '.join(bits)}")
    return lines


def _recompute(state_json: dict[str, Any], engine: str, top_k: int, exact_max_secs: float) -> dict[str, Any]:
    req = RecommendRequest(
        engine=engine,
        state=state_json,
        top_k=top_k,
        exact_max_secs=exact_max_secs,
    )
    return recommend(req)


def _checkpoint_from_probe(payload: dict[str, Any] | None, response: dict[str, Any] | None) -> str | None:
    for src in (payload, response):
        if not isinstance(src, dict):
            continue
        for key in ("checkpoint_path", "checkpoint"):
            value = src.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _raw_value_for_player0(root_traj: dict[str, Any], actor: int) -> float:
    win_prob = float(root_traj["win_prob"])
    value_actor = 2.0 * win_prob - 1.0
    return value_actor if actor == 0 else -value_actor


def _trajectory_for_player(net, state, player: int, device: str) -> dict[str, Any]:
    import torch

    mb, ob, flat = encode_state(state, player)
    with torch.inference_mode():
        mb_t = torch.from_numpy(mb).unsqueeze(0).to(device)
        ob_t = torch.from_numpy(ob).unsqueeze(0).to(device)
        flat_t = torch.from_numpy(flat).unsqueeze(0).to(device)
        own, opp, win_prob, _logits = net(mb_t, ob_t, flat_t)
    return {
        "own_score_est": float(own.item() * 160.0),
        "opp_score_est": float(opp.item() * 160.0),
        "score_margin_est": float((own.item() - opp.item()) * 160.0),
        "win_prob": float(win_prob.item()),
    }


def _root_actor_edge_from_traj(traj: dict[str, Any], encoded_player: int, root_actor: int) -> float:
    value_encoded = 2.0 * float(traj["win_prob"]) - 1.0
    value0 = value_encoded if encoded_player == 0 else -value_encoded
    return value0 if root_actor == 0 else -value0


def _phase1_audit_row(
    path: Path,
    *,
    net,
    device: str,
) -> dict[str, Any]:
    data = _load_json(path)
    capture, payload, response = _probe_parts(data)
    state_json = _state_json_from_probe(data)
    state = state_from_debug_json(state_json)
    warning = state_json.get("board_reconstruction_warning")
    row: dict[str, Any] = {
        "path": str(path),
        "phase": state.phase.name,
        "current_actor": None if state.phase == Phase.GAME_OVER else int(state.current_actor),
        "deck_count": len(state.deck),
        "legal_actions": len(state.legal_actions()),
        "scores": "/".join(str(s) for s in state.scores()),
        "has_board_warning": bool(warning),
        "saved_value": None if not response else response.get("value"),
        "saved_root_win_prob": None if not response else response.get("root_win_prob"),
        "saved_engine": None if not response else response.get("engine"),
        "saved_sims": None if not payload else payload.get("nn_sims"),
    }
    if state.phase == Phase.GAME_OVER:
        row.update({
            "p0_win": None,
            "p1_win": None,
            "win_symmetry_err": None,
            "margin_symmetry_err": None,
            "root_actor_raw_edge": None,
            "best_child_root_edge": None,
            "worst_child_root_edge": None,
            "mean_child_root_edge": None,
            "root_minus_best_child": None,
            "root_sign_disagrees_all_children": False,
            "max_child_actor_jump": None,
            "transition_errors": 0,
            "flags": "game_over",
        })
        return row

    root_actor = int(state.current_actor)
    traj0 = _trajectory_for_player(net, state, 0, device)
    traj1 = _trajectory_for_player(net, state, 1, device)
    p0_win = float(traj0["win_prob"])
    p1_win = float(traj1["win_prob"])
    p0_margin = float(traj0["score_margin_est"])
    p1_margin = float(traj1["score_margin_est"])
    root_actor_traj = traj0 if root_actor == 0 else traj1
    root_actor_raw_edge = _root_actor_edge_from_traj(root_actor_traj, root_actor, root_actor)

    child_edges: list[float] = []
    child_actor_jumps: list[float] = []
    transition_errors = 0
    for action in state.legal_actions():
        try:
            child = state.step(action)
        except Exception:
            transition_errors += 1
            continue
        if child.phase == Phase.GAME_OVER:
            scores = child.scores()
            if scores[0] > scores[1]:
                child_value0 = 1.0
            elif scores[1] > scores[0]:
                child_value0 = -1.0
            else:
                child_value0 = 0.0
            child_root_edge = child_value0 if root_actor == 0 else -child_value0
        else:
            child_actor = int(child.current_actor)
            child_traj = _trajectory_for_player(net, child, child_actor, device)
            child_root_edge = _root_actor_edge_from_traj(child_traj, child_actor, root_actor)
            child_actor_edge = _root_actor_edge_from_traj(child_traj, child_actor, child_actor)
            child_actor_jumps.append(abs(root_actor_raw_edge - child_actor_edge))
        child_edges.append(child_root_edge)

    best_child = max(child_edges) if child_edges else None
    worst_child = min(child_edges) if child_edges else None
    mean_child = (sum(child_edges) / len(child_edges)) if child_edges else None
    flags: list[str] = []
    win_symmetry_err = abs((p0_win + p1_win) - 1.0)
    margin_symmetry_err = abs(p0_margin + p1_margin)
    if win_symmetry_err > 0.20:
        flags.append("win_symmetry")
    if margin_symmetry_err > 10.0:
        flags.append("margin_symmetry")
    if best_child is not None:
        root_minus_best = root_actor_raw_edge - best_child
        if abs(root_minus_best) > 0.50:
            flags.append("root_child_gap")
        if root_actor_raw_edge > 0.0 and best_child < 0.0:
            flags.append("root_positive_all_children_negative")
        if root_actor_raw_edge < 0.0 and worst_child > 0.0:
            flags.append("root_negative_all_children_positive")
    else:
        root_minus_best = None
    if transition_errors:
        flags.append("transition_error")
    if warning:
        flags.append("board_warning")

    row.update({
        "p0_win": p0_win,
        "p1_win": p1_win,
        "p0_margin": p0_margin,
        "p1_margin": p1_margin,
        "win_symmetry_err": win_symmetry_err,
        "margin_symmetry_err": margin_symmetry_err,
        "root_actor_raw_edge": root_actor_raw_edge,
        "best_child_root_edge": best_child,
        "worst_child_root_edge": worst_child,
        "mean_child_root_edge": mean_child,
        "root_minus_best_child": root_minus_best,
        "root_sign_disagrees_all_children": (
            best_child is not None and (
                (root_actor_raw_edge > 0.0 and best_child < 0.0)
                or (root_actor_raw_edge < 0.0 and worst_child is not None and worst_child > 0.0)
            )
        ),
        "max_child_actor_jump": max(child_actor_jumps) if child_actor_jumps else None,
        "transition_errors": transition_errors,
        "flags": ",".join(flags) if flags else "ok",
    })
    return row


def run_phase1_audit(
    paths: list[Path],
    *,
    device: str,
    checkpoint_path: str | None,
    output_csv: Path | None,
) -> str:
    if not paths:
        return f"No probe files found. Pass a file path or put JSON files in {DEFAULT_PROBE_DIR}."

    first_data = _load_json(paths[0])
    _capture, payload, response = _probe_parts(first_data)
    ckpt = checkpoint_path or _checkpoint_from_probe(payload, response)
    bot_req = BotActionRequest(
        session_id="advisor-probe-phase1-audit",
        mode="nn",
        checkpoint_path=ckpt,
        device=device,
        nn_sims=1,
        seed=0,
    )
    _evaluator, net, loaded_checkpoint = _load_nn_evaluator(bot_req)

    rows = [_phase1_audit_row(path, net=net, device=device) for path in paths]
    fieldnames = [
        "path", "phase", "current_actor", "deck_count", "legal_actions", "scores",
        "has_board_warning", "saved_engine", "saved_sims", "saved_value", "saved_root_win_prob",
        "p0_win", "p1_win", "p0_margin", "p1_margin",
        "win_symmetry_err", "margin_symmetry_err", "root_actor_raw_edge",
        "best_child_root_edge", "worst_child_root_edge", "mean_child_root_edge",
        "root_minus_best_child", "root_sign_disagrees_all_children",
        "max_child_actor_jump", "transition_errors", "flags",
    ]
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    flagged = [r for r in rows if r.get("flags") != "ok"]
    severe_gap = sorted(
        [
            r for r in rows
            if isinstance(r.get("root_minus_best_child"), (int, float))
        ],
        key=lambda r: abs(float(r["root_minus_best_child"])),
        reverse=True,
    )[:8]

    lines = [
        "phase1 audit:",
        f"  probes: {len(rows)}",
        f"  checkpoint: {loaded_checkpoint}",
        f"  csv: {output_csv if output_csv is not None else '<not written>'}",
        f"  flagged: {len(flagged)}",
    ]
    flag_counts: dict[str, int] = {}
    for row in flagged:
        for flag in str(row.get("flags") or "").split(","):
            if flag and flag != "ok":
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
    if flag_counts:
        lines.append("  flag counts:")
        for flag, count in sorted(flag_counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"    {flag}: {count}")
    if severe_gap:
        lines.append("  largest root/best-child gaps:")
        for row in severe_gap:
            lines.append(
                "    "
                f"{Path(str(row['path'])).name}: "
                f"actor=P{row['current_actor']} phase={row['phase']} "
                f"root={float(row['root_actor_raw_edge']):+.3f} "
                f"best_child={float(row['best_child_root_edge']):+.3f} "
                f"gap={float(row['root_minus_best_child']):+.3f} "
                f"flags={row['flags']}"
            )
    return "\n".join(lines)


def _child_eval_diagnostic(
    state_json: dict[str, Any],
    payload: dict[str, Any] | None,
    response: dict[str, Any] | None,
    *,
    top_k: int,
    device: str,
    checkpoint_path: str | None,
) -> list[str]:
    state = state_from_debug_json(state_json)
    if state.phase == Phase.GAME_OVER:
        return ["child-eval: game is over"]

    ckpt = checkpoint_path or _checkpoint_from_probe(payload, response)
    bot_req = BotActionRequest(
        session_id="advisor-probe-child-eval",
        mode="nn",
        checkpoint_path=ckpt,
        device=device,
        nn_sims=1,
        seed=0,
    )
    _evaluator, net, loaded_checkpoint = _load_nn_evaluator(bot_req)

    root_actor = int(state.current_actor)
    root_traj = _root_trajectory(net, state, device)
    root_value0 = _raw_value_for_player0(root_traj, root_actor)
    root_actor_edge = root_value0 if root_actor == 0 else -root_value0

    rows: list[dict[str, Any]] = []
    for idx, action in enumerate(state.legal_actions()):
        child = state.step(action)
        if child.phase == Phase.GAME_OVER:
            # Terminal child has no current actor. Use exact score result in a
            # pure win-value frame so it remains comparable to alpha=0 advisor
            # search values.
            scores = child.scores()
            if scores[0] > scores[1]:
                child_value0 = 1.0
            elif scores[1] > scores[0]:
                child_value0 = -1.0
            else:
                child_value0 = 0.0
            child_actor = None
            child_traj = {
                "own_score_est": None,
                "opp_score_est": None,
                "score_margin_est": scores[0] - scores[1],
                "win_prob": None,
            }
        else:
            child_actor = int(child.current_actor)
            child_traj = _root_trajectory(net, child, device)
            child_value0 = _raw_value_for_player0(child_traj, child_actor)
        root_actor_edge_after_child = child_value0 if root_actor == 0 else -child_value0
        child_actor_edge = None if child_actor is None else (child_value0 if child_actor == 0 else -child_value0)
        rows.append({
            "idx": idx,
            "action": action,
            "action_json": action_to_json(state, action, idx),
            "child_actor": child_actor,
            "child_traj": child_traj,
            "child_actor_edge": child_actor_edge,
            "value_player0": child_value0,
            "root_actor_edge": root_actor_edge_after_child,
            "scores": child.scores(),
        })

    rows.sort(key=lambda r: float(r["root_actor_edge"]), reverse=True)

    lines = [
        "child-eval diagnostic:",
        f"  checkpoint: {loaded_checkpoint}",
        f"  root_actor: P{root_actor}",
        (
            "  root raw: "
            f"actor_win={float(root_traj['win_prob']):.3f} "
            f"actor_edge={root_actor_edge:+.3f} "
            f"score_est={float(root_traj['own_score_est']):.1f}/{float(root_traj['opp_score_est']):.1f} "
            f"margin={float(root_traj['score_margin_est']):+.1f}"
        ),
        "  children sorted by root_actor_edge (pure win-head frame):",
    ]
    for rank, row in enumerate(rows[:top_k], start=1):
        traj = row["child_traj"]
        child_actor = row["child_actor"]
        child_actor_txt = "GAME_OVER" if child_actor is None else f"P{child_actor}"
        win_txt = "-" if traj.get("win_prob") is None else f"{float(traj['win_prob']):.3f}"
        child_edge_txt = "-" if row["child_actor_edge"] is None else f"{float(row['child_actor_edge']):+.3f}"
        own = traj.get("own_score_est")
        opp = traj.get("opp_score_est")
        score_txt = f"terminal_scores={row['scores']}" if own is None or opp is None else f"score_est={float(own):.1f}/{float(opp):.1f}"
        lines.append(
            f"  #{rank} legal={row['idx']} root_edge={float(row['root_actor_edge']):+.3f} "
            f"value_p0={float(row['value_player0']):+.3f} child_actor={child_actor_txt} "
            f"child_win={win_txt} child_actor_edge={child_edge_txt} {score_txt} "
            f"| {row['action_json'].get('label')}"
        )
    return lines


def inspect_probe(
    path: Path,
    *,
    recompute_engine: str | None,
    top_k: int,
    exact_max_secs: float,
    child_eval: bool,
    device: str,
    checkpoint_path: str | None,
) -> str:
    data = _load_json(path)
    capture, payload, response = _probe_parts(data)
    state_json = _state_json_from_probe(data)
    state = state_from_debug_json(state_json)

    from games.kingdomino.web_app import state_to_public_json

    public = state_to_public_json(state)
    lines = [
        f"== {path} ==",
        f"schema: {data.get('schema', '<raw-capture>')}",
        f"url: {data.get('url') or (capture or {}).get('url')}",
        f"phase: {state.phase.name} actor: {None if state.phase == Phase.GAME_OVER else state.current_actor}",
        f"deck: count={len(state.deck)} ids={list(state.deck)}",
        f"current_row: {list(state.current_row)}",
        f"pending_claims: {[(int(c.player), int(c.domino_id)) for c in state.pending_claims]}",
        f"next_claims: {[(int(c.player), int(c.domino_id)) for c in state.next_claims]}",
        f"legal_actions: {len(state.legal_actions())}",
        f"scores: {public.get('scores')}",
        "boards:",
        *_board_summary(public),
    ]

    debug = state_json.get("debug") if isinstance(state_json.get("debug"), dict) else {}
    if debug:
        lines.append("capture/debug:")
        for key in ("bga_state_name", "bga_locations", "displayed_deck_count", "inferred_hidden_deck"):
            if key in debug:
                lines.append(f"  {key}: {debug[key]}")
        notes = debug.get("notes")
        if isinstance(notes, list):
            for note in notes:
                lines.append(f"  note: {note}")

    warning = state_json.get("board_reconstruction_warning")
    if warning:
        lines.append(f"board_reconstruction_warning: {warning}")

    try:
        actor = state.current_actor if state.phase != Phase.GAME_OVER else 0
        my_board, opp_board, flat = encode_state(state, actor)
        lines.append(
            "encoder: "
            f"my_occupied={float(my_board[8].sum()):.0f} "
            f"opp_occupied={float(opp_board[8].sum()):.0f} "
            f"flat_size={flat.shape[0]}"
        )
    except Exception as exc:
        lines.append(f"encoder: ERROR {exc}")

    if payload:
        lines.append(
            f"saved payload: engine={payload.get('engine')} requested={payload.get('requested_engine')} "
            f"nn_sims={payload.get('nn_sims')} exact_max_secs={payload.get('exact_max_secs')}"
        )
    if response:
        lines.append(
            f"saved response: engine={response.get('engine')} value={response.get('value')} "
            f"root_win_prob={response.get('root_win_prob')} search_ms={response.get('search_ms')}"
        )
        lines.extend(_top_recommendations(response, top_k))

    if recompute_engine:
        lines.append(f"recompute {recompute_engine}:")
        recomputed = _recompute(state_json, recompute_engine, top_k, exact_max_secs)
        lines.append(
            f"  engine={recomputed.get('engine')} value={recomputed.get('value')} "
            f"root_win_prob={recomputed.get('root_win_prob')} search_ms={recomputed.get('search_ms')}"
        )
        lines.extend(_top_recommendations(recomputed, top_k))

    if child_eval:
        lines.extend(_child_eval_diagnostic(
            state_json,
            payload,
            response,
            top_k=top_k,
            device=device,
            checkpoint_path=checkpoint_path,
        ))

    if state.legal_actions():
        lines.append("first legal actions:")
        for i, action in enumerate(state.legal_actions()[: min(top_k, 8)]):
            lines.append(f"  {i}: {action_to_json(state, action, i).get('label')}")

    return "\n".join(lines)


def _paths_from_args(args: argparse.Namespace) -> list[Path]:
    if args.paths:
        return [Path(p) for p in args.paths]
    if not DEFAULT_PROBE_DIR.exists():
        return []
    return sorted(DEFAULT_PROBE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Inspect downloaded Kingdomino advisor probe JSON files.")
    parser.add_argument("paths", nargs="*", help="Probe JSON files. Defaults to runs/kingdomino/advisor_probes/*.json")
    parser.add_argument("--recompute", choices=["greedy", "exact", "auto", "nn"], default=None)
    parser.add_argument("--child-eval", action="store_true", help="Evaluate raw NN on every child state and print root-actor-framed values.")
    parser.add_argument("--phase1-audit", action="store_true", help="Run perspective/root-child invariant audit across probes and write a CSV.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--exact-max-secs", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--audit-csv", default=str(DEFAULT_AUDIT_CSV))
    args = parser.parse_args()

    paths = _paths_from_args(args)
    if not paths:
        print(f"No probe files found. Pass a file path or put JSON files in {DEFAULT_PROBE_DIR}.")
        return

    if args.phase1_audit:
        output_csv = Path(args.audit_csv) if args.audit_csv else None
        print(run_phase1_audit(
            paths,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            output_csv=output_csv,
        ))
        return

    for idx, path in enumerate(paths):
        if idx:
            print()
        print(inspect_probe(
            path,
            recompute_engine=args.recompute,
            top_k=args.top_k,
            exact_max_secs=args.exact_max_secs,
            child_eval=args.child_eval,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
        ))


if __name__ == "__main__":
    main()
