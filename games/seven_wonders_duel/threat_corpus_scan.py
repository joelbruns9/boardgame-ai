"""Find positions where the actor's own move uncovers a threat to itself.

Stage 0 of `WORLD_CLASS_MODEL_EVOLUTION_PLAN.md` asks for a corpus of
actor-created-threat positions. Table `908370787` is one instance; this finds the
rest across the captured BGA logs, unblocking everything that currently stalls on
"we only have one position": the regret-over-margin measurement for Workstream
10, any strength claim for Workstream 9's mechanism 2, the trigger definition for
a tactical search extension, and any targeted training correction.

**Every rule question is delegated to the engine.** An earlier version
reconstructed the science and military rules here and got both wrong -- it
counted science symbols from buildings only, missing the one `Law` grants; it
ignored `claimed_science_pairs` and whether any progress token remained to take;
it invented a single `|6|` military band where the engine has bands at 3 and 6
whose tokens are claimed once; and it ignored the extra shield `Strategy` gives a
red card. Those are not details -- they decide whether a position enters the
corpus. Nothing here re-derives a rule that `engine.py` already owns.

**Actions are applied, not assumed.** The scan builds the afterstate for each
candidate action and asks the engine who moves next, because an action that
retains the turn, ends the Age, or ends the game does not hand the opponent the
reply this corpus is about.

The shape, generalised from the reference case rather than copied from it:

  1. It is the actor's turn in an Age.
  2. A legal action removes a card, uncovering one or more slots.
  3. After that action the OPPONENT is to move.
  4. Within `--reach` further removals sits a card that threatens the actor if
     the opponent takes it.
  5. Whether the opponent can act on it *immediately* depends on chain distance
     and on holding a legal, affordable extra-turn Wonder -- recorded here,
     judged in `threat_corpus_episodes`.

Threat classes, judged for the OPPONENT against engine state:

  science_win     the card's symbol completes six distinct (tokens included)
  science_pair    completes an UNCLAIMED pair while a progress token remains
  military_band   the shields enter a band whose token is still there
  military_win    the shields reach the opponent's capital

This scan is structural: it decides shape, not cost. Whether a threat is worth
anything is an action-regret question, answered by
`w9_reference_case.py --stages ref-values` at minutes per position rather than
milliseconds. Keeping the two apart is what makes 2,899 rows affordable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from .codec import decode_action, legal_action_indices
from .data import (
    CARDS_BY_NAME,
    TABLEAU_LAYOUTS,
    WONDERS_BY_NAME,
    CardColor,
    EffectKind,
    covering_slots,
)
from .engine import (
    _can_afford,
    _science_symbols,
    apply_action,
    military_token_band,
    minimum_payment,
)
from .game import Phase
from .search import state_actor

REPO_ROOT = Path(__file__).resolve().parents[2]

EXTRA_TURN_WONDERS = frozenset(
    w.name for w in WONDERS_BY_NAME.values()
    if any(e.kind is EffectKind.PLAY_AGAIN for e in w.effects)
)

MILITARY_CAPITAL = 9
"""`engine._apply_military` declares a military victory at |position| == 9."""


# -- rules, all delegated ---------------------------------------------------


def science_symbols(game, player):
    """Distinct science symbols, buildings AND progress tokens (e.g. `Law`)."""

    return _science_symbols(game, player)


def shields_if_built(card, game, player):
    """Shields the card would contribute, including `Strategy`'s extra.

    Mirrors `engine._after_building_constructed`.
    """

    shields = card.shields
    if card.color is CardColor.RED and "Strategy" in game.cities[player].progress_tokens:
        shields += 1
    return shields


def threat_of(card_name, game, opponent):
    """How `card_name` would threaten the actor if `opponent` built it."""

    card = CARDS_BY_NAME.get(card_name)
    if card is None:
        return None
    city = game.cities[opponent]

    if card.science is not None:
        symbols = science_symbols(game, opponent)
        if card.science not in symbols and len(symbols) + 1 >= 6:
            return "science_win"
        # A pair only pays if it is unclaimed AND a token is left to take.
        copies = sum(
            CARDS_BY_NAME[n].science is card.science for n in city.buildings
        )
        if (
            copies >= 1
            and card.science not in city.claimed_science_pairs
            and game.available_progress_tokens
        ):
            return "science_pair"

    shields = shields_if_built(card, game, opponent)
    if shields:
        direction = 1 if opponent == 0 else -1
        position = game.conflict_position
        for _ in range(shields):
            position += direction
            if abs(position) == MILITARY_CAPITAL:
                return "military_win"
            band = military_token_band(position)
            if band is not None and band in game.military_tokens_remaining:
                return "military_band"
    return None


# -- topology ---------------------------------------------------------------


def uncovered_by(tableau, slot_id):
    """Slots that become accessible when `slot_id` is removed.

    Gated against `TableauState.is_accessible` by
    `test_uncovered_matches_engine_accessibility`; the duplication is deliberate
    (this asks a hypothetical the engine has no API for) and the test is what
    keeps it from drifting.
    """

    layout = TABLEAU_LAYOUTS[tableau.age]
    out = []
    for other_id, card in tableau.cards.items():
        if not card.present or other_id == slot_id:
            continue
        coverers = [(c.row, c.x) for c in covering_slots(layout, card.slot)]
        if slot_id not in coverers:
            continue
        if all(
            not tableau.cards[c].present
            for c in coverers
            if c != slot_id and c in tableau.cards
        ):
            out.append(other_id)
    return out


def reach(tableau, slot_id, depth):
    """(slot, distance) reachable by removing up to `depth` further cards."""

    seen, frontier, out = {slot_id}, [(slot_id, 0)], []
    while frontier:
        current, dist = frontier.pop(0)
        if dist >= depth:
            continue
        for nxt in uncovered_by(tableau, current):
            if nxt in seen:
                continue
            seen.add(nxt)
            out.append((nxt, dist + 1))
            frontier.append((nxt, dist + 1))
    return out


def affordable_extra_turn_wonders(game, player):
    """Unbuilt, un-retired extra-turn Wonders this player could actually build.

    Ownership is not enough: the threat requires the opponent to PLAY the
    Wonder, so a retired one, or one it cannot pay for, does not make the threat
    immediate.

    Affordability is priced with the engine's own `minimum_payment` /
    `_can_afford`, which take an explicit player and so work while it is the
    other player's turn -- unlike `legal_action_indices`, which only describes
    the mover. An earlier version fell back to ownership whenever the player was
    not to move, which meant EVERY corpus row reported
    `extra_turn_legality_verified: false` and the corpus recorded a weaker claim
    than it appeared to.

    The seventh-Wonder rule is honoured: once seven are built across both
    cities, no eighth can be constructed.
    """

    city = game.cities[player]
    retired = set(getattr(game, "retired_wonders", ()) or ())
    owned = [
        w for w in city.wonders
        if w not in city.built_wonders
        and w in EXTRA_TURN_WONDERS
        and w not in retired
    ]
    if not owned:
        return [], True

    built_total = sum(len(c.built_wonders) for c in game.cities)
    if built_total >= 7:
        return [], True  # no eighth Wonder is ever constructed

    affordable = []
    for name in owned:
        wonder = WONDERS_BY_NAME[name]
        try:
            payment = minimum_payment(game, player, wonder.cost, is_wonder=True)
        except Exception:
            continue  # unpriceable here; omit rather than overclaim
        if _can_afford(game, player, payment):
            affordable.append(name)
    return sorted(affordable), True


def observation_digest(state_payload) -> str:
    """Stable digest of a logged observation, for de-duplication.

    The logs repeat identical observations -- the same position captured twice
    -- and counting those twice inflates every corpus statistic.
    """

    return hashlib.sha256(
        json.dumps(state_payload, sort_keys=True, default=str).encode()
    ).hexdigest()


# -- the scan ---------------------------------------------------------------


def scan_row(game, reach_depth, *, apply_actions=True):
    """Every (action, threat) pair this position exposes. Network-free."""

    actor = state_actor(game)
    opponent = 1 - actor
    hits, skipped = [], Counter()

    for index in legal_action_indices(game):
        action = decode_action(game, index)
        if action.slot_id is None:
            continue
        slot_id = tuple(action.slot_id)
        if slot_id not in game.tableau.cards:
            continue

        after = None
        if apply_actions:
            # Who actually moves next? An action granting an extra turn, ending
            # the Age, or ending the game does not hand the opponent this reply.
            try:
                after = game.clone()
                after.search_barrier = False
                apply_action(after, decode_action(after, index))
            except Exception:
                skipped["apply_failed"] += 1
                continue
            if after.phase is Phase.COMPLETE:
                skipped["ends_game"] += 1
                continue
            if state_actor(after) != opponent:
                skipped["actor_retains_turn"] += 1
                continue
            if after.tableau.age != game.tableau.age:
                skipped["age_changed"] += 1
                continue

        board = after.tableau if after is not None else game.tableau
        threat_state = after if after is not None else game
        label = f"{action.use.value}"
        card_here = game.tableau.cards.get(slot_id)
        if card_here is not None and card_here.revealed:
            label += f":{card_here.card_name}"
        if action.wonder_name:
            label += f" [{action.wonder_name}]"

        for exposed in uncovered_by(game.tableau, slot_id):
            targets = [(exposed, 0)] + reach(board, exposed, reach_depth)
            for target, distance in targets:
                card = board.cards.get(target)
                if card is None or not card.present:
                    continue
                if not card.revealed:
                    hits.append({
                        "action_index": int(index), "action_use": action.use.value,
                        "action_label": label, "removes": list(slot_id),
                        "target": list(target), "distance": distance,
                        "face_up": False, "threat": None,
                    })
                    continue
                kind = threat_of(card.card_name, threat_state, opponent)
                if kind is None:
                    continue
                hits.append({
                    "action_index": int(index), "action_use": action.use.value,
                    "action_label": label, "removes": list(slot_id),
                    "target": list(target), "target_card": card.card_name,
                    "distance": distance, "face_up": True, "threat": kind,
                })
    return hits, skipped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--log-dir", default="runs/seven_wonders_duel/bga_game_log")
    parser.add_argument("--reach", type=int, default=3)
    parser.add_argument("--resample-seed", type=int, default=0)
    parser.add_argument("--no-apply-actions", dest="apply_actions",
                        action="store_false",
                        help="classify from the pre-action state (topology only)")
    parser.add_argument("--limit-tables", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    log = (lambda *_: None) if args.quiet else (
        lambda m: print(m, file=sys.stderr, flush=True)
    )

    from .advisor_scrape import determinize_observation, observation_from_wire
    from .bga_extract import wire_from_bga_payload

    paths = sorted((REPO_ROOT / args.log_dir).glob("table_*.jsonl"))
    if args.limit_tables:
        paths = paths[: args.limit_tables]
    log(f"scanning {len(paths)} tables, reach={args.reach}, "
        f"apply_actions={args.apply_actions}")

    found, stats = [], Counter()
    for path in paths:
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        decisions = [r for r in rows if r.get("kind") == "decision"]
        seen_digests = set()
        for row_index, row in enumerate(decisions):
            stats["rows"] += 1
            digest = observation_digest(row["state"])
            if digest in seen_digests:
                stats["duplicate_observation"] += 1
                continue
            seen_digests.add(digest)
            try:
                payload = wire_from_bga_payload(row["state"])
                if "observation" not in payload:
                    stats["no_observation"] += 1
                    continue
                obs = observation_from_wire(payload["observation"])
                if obs.phase is not Phase.PLAY_AGE:
                    stats["not_play_age"] += 1
                    continue
                game = determinize_observation(
                    obs, random.Random(args.resample_seed),
                    unknown_burial_ages=tuple(
                        int(a) for a in payload.get("unknown_burial_ages", ())
                    ),
                )
            except Exception as exc:
                stats[f"skip:{type(exc).__name__}"] += 1
                continue

            hits, skipped = scan_row(game, args.reach, apply_actions=args.apply_actions)
            stats.update({f"action_skip:{k}": v for k, v in skipped.items()})
            public = [h for h in hits if h["face_up"] and h["threat"]]
            if not public:
                stats["no_public_threat"] += 1
                continue
            opponent = 1 - state_actor(game)
            extra, verified = affordable_extra_turn_wonders(game, opponent)
            stats["hits"] += 1
            for h in public:
                stats[f"threat:{h['threat']}"] += 1
            found.append({
                "table": path.stem.replace("table_", ""),
                "decision_row": row_index,
                "observation_digest": digest[:16],
                "age": int(game.age),
                "actor": state_actor(game),
                "opponent_unbuilt_extra_turn": extra,
                "extra_turn_legality_verified": verified,
                "public_threats": public,
                "hidden_slots_in_reach": sum(1 for h in hits if not h["face_up"]),
            })
            log(
                f"  {path.stem.replace('table_', ''):<10} row {row_index:<3} "
                f"age {game.age} -- {len(public)} threat(s) "
                f"{sorted({h['threat'] for h in public})}"
                f"{'  extra-turn: ' + ','.join(extra) if extra else ''}"
            )

    log("\n" + json.dumps(dict(stats), indent=2))
    report = {
        "harness": "threat_corpus_scan",
        "note": "structural shape only; action regret is measured separately",
        "params": {
            "reach": args.reach, "resample_seed": args.resample_seed,
            "apply_actions": bool(args.apply_actions),
        },
        "extra_turn_wonders": sorted(EXTRA_TURN_WONDERS),
        "stats": dict(stats),
        "positions": found,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        log(f"wrote {out}  ({len(found)} positions)")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
