"""BGA denial anchor (Variant 1) — policy prior assigned to the picks strong
human opponents actually played.

Non-circular: uses the POLICY head only, never the value head. For each opponent
pick decision in the BGA logs, reconstruct the game state, run the net's policy,
group mass by pick (summing over placements, exactly as _root_candidates does),
and record the prior the net assigned to the domino the human actually took.

A left-heavy pile near zero = the policy starves moves that top-30 humans play
= the reply-starvation blindspot, proven without the value head.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from games.kingdomino.action_codec import encode_action
from games.kingdomino.denial_search import (
    AZBatchEvaluator, _pick_key, load_checkpoint_network, public_state_key,
)
from games.kingdomino.promotion import DEFAULT_CURRENT_BEST
from games.kingdomino.web_app import state_from_debug_json

LOG_DIR = Path(r"runs/kingdomino/bga_game_log")


def _claims_by(state_json: dict) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for c in state_json.get("pending_claims", []) + state_json.get("next_claims", []):
        out.setdefault(int(c["player"]), set()).add(int(c["domino_id"]))
    return out


def _hero_player_index(records: list[dict], hero_viewer: str) -> dict[str, int]:
    """Map BGA active_player id -> engine player index, from observed pairings."""
    mapping: dict[str, int] = {}
    for r in records:
        ap = r.get("active_player")
        ca = r.get("state", {}).get("current_actor")
        if ap is not None and ca is not None:
            mapping[str(ap)] = int(ca)
    return mapping


def _pick_events(decisions: list[dict]):
    """Yield (state_json, actor_index, picked_domino) for CLEAN single-pick steps."""
    for t in range(len(decisions) - 1):
        s0 = decisions[t]["state"]
        s1 = decisions[t + 1]["state"]
        actor = s0.get("current_actor")
        if actor is None:
            continue
        actor = int(actor)
        row0 = set(int(x) for x in s0.get("current_row", []))
        before = _claims_by(s0).get(actor, set())
        after = _claims_by(s1).get(actor, set())
        newly = [d for d in (after - before) if d in row0]
        if len(newly) == 1:                       # drop multi-pick (init/duel) deltas
            yield s0, actor, newly[0]


def analyse(checkpoint: str, device: str, min_deck: int, max_deck: int, out: Path):
    net, _cfg = load_checkpoint_network(checkpoint, device)
    ev = AZBatchEvaluator(net, device=device)

    files = sorted(glob.glob(str(LOG_DIR / "table_*.jsonl")))
    finals = {}  # table -> opponent name
    rows = []
    stats = dict(tables=0, opp_picks=0, clean=0, dropped_multi=0, illegal=0)

    for fn in files:
        if Path(fn).name == "table_unknown.jsonl":
            continue                              # quarantined: no advisor, unknown provenance
        recs = [json.loads(l) for l in open(fn, encoding="utf-8")]
        decisions = [r for r in recs if r.get("kind") == "decision"]
        final = next((r for r in recs if r.get("kind") == "final"), None)
        if not decisions or not final:
            continue
        stats["tables"] += 1
        players = final["final"]["players"]
        # hero = viewer_id (the logging user)
        hero_vid = str(decisions[0].get("viewer_id"))
        ap_to_idx = _hero_player_index(decisions, hero_vid)
        hero_idx = ap_to_idx.get(hero_vid)
        opp_vid = next((p for p in players if p != hero_vid), None)
        opp_name = players.get(opp_vid, {}).get("name") if opp_vid else None
        opp_idx = ap_to_idx.get(str(opp_vid)) if opp_vid else None
        if hero_idx is None or opp_idx is None:
            continue

        # count multi-pick drops for reporting
        for t in range(len(decisions) - 1):
            s0 = decisions[t]["state"]; s1 = decisions[t + 1]["state"]
            a = s0.get("current_actor")
            if a is None or int(a) != opp_idx:
                continue
            row0 = set(int(x) for x in s0.get("current_row", []))
            newly = [d for d in (_claims_by(s1).get(int(a), set()) - _claims_by(s0).get(int(a), set())) if d in row0]
            if len(newly) > 1:
                stats["dropped_multi"] += 1

        for s0, actor, picked in _pick_events(decisions):
            if actor != opp_idx:
                continue                          # opponent picks only (clean, uncontaminated)
            stats["opp_picks"] += 1
            deck = int(s0.get("deck_count", -1))
            if not (min_deck <= deck <= max_deck):
                continue
            try:
                state = state_from_debug_json(s0)
                actions = state.legal_actions()
            except Exception:
                continue
            # group policy mass by pick
            pol = ev.policy(state)
            by_pick: dict[int, float] = {}
            for a in actions:
                pid = _pick_key(a)
                if pid is None:
                    continue
                idx = int(encode_action(a, state))
                by_pick[pid] = by_pick.get(pid, 0.0) + float(pol.get(idx, 0.0))
            if picked not in by_pick:
                stats["illegal"] += 1             # state/pick mismatch -> drop
                continue
            stats["clean"] += 1
            priors = sorted(by_pick.values(), reverse=True)
            rank = sorted(by_pick, key=lambda d: -by_pick[d]).index(picked) + 1
            rows.append(dict(
                table=Path(fn).stem.replace("table_", ""),
                opponent=opp_name, deck=deck, n_choices=len(by_pick),
                prior_on_played=by_pick[picked], rank=rank,
                top_prior=priors[0],
            ))

    P = np.array([r["prior_on_played"] for r in rows])
    ranks = np.array([r["rank"] for r in rows])
    report = dict(
        checkpoint=str(checkpoint), n=len(rows), stats=stats,
        deck_band=[min_deck, max_deck],
        prior_on_played=dict(
            mean=float(P.mean()), median=float(np.median(P)),
            p10=float(np.percentile(P, 10)), p90=float(np.percentile(P, 90)),
            frac_under_05=float((P < 0.05).mean()),
            frac_under_10=float((P < 0.10).mean()),
            frac_under_02=float((P < 0.02).mean()),
        ),
        rank_of_played={int(k): int((ranks == k).sum()) for k in range(1, 6)},
        top1_match=float((ranks == 1).mean()),
    )
    out.write_text(json.dumps(dict(report=report, rows=rows), indent=1), encoding="utf-8")
    return report, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CURRENT_BEST))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-deck", type=int, default=0)
    ap.add_argument("--max-deck", type=int, default=48)
    ap.add_argument("--out", type=Path, default=Path("runs/kingdomino/bga_game_log/denial_anchor.json"))
    a = ap.parse_args()
    report, rows = analyse(a.checkpoint, a.device, a.min_deck, a.max_deck, a.out)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
