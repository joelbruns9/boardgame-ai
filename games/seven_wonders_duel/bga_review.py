"""Post-game review of a logged BGA game: play, the opponent, and luck, apart.

    python -m games.seven_wonders_duel.bga_review TABLE_JSONL \
        --checkpoint extension_7wd/run07_iter35.pt --device cuda --sims 3200

Every decision of both players comes from :mod:`bga_replay`. Each is searched,
and every value below is the LOGGING player's win probability.

For a move ``m`` in position ``s`` the swing to the next position splits into:

* **choice**   ``Q(best) - Q(m)`` in ``s``, from the mover's side: what picking
  ``m`` over the search's favourite cost (the mover's "regret").
* **optimism** ``Q(m) - E[V(after m)]``: how far the search in ``s`` overrated
  ``m`` against a direct evaluation of the positions it actually leads to. This
  is where a reply the search under-funded shows up -- the chance fan-out blind
  spot measured it at ~20 points on some reveal moves.
* **luck**     ``V(after m, actual reveal) - E[V(after m)]``, nonzero only when
  ``m`` turned cards face up. ``E`` is the mean over ``--worlds`` alternative
  reveals: the same move with other cards the hidden pool could have produced.

A move that reveals nothing has ``E[V(after m)] = V(after m)`` and no luck.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from .advisor_adapter import (
    SevenWondersAdvisor,
    _Position,
    _hidden_locations,
    _is_guild_name,
    _label,
    _name_at,
    _set_name,
)
from .bga_replay import replay_table
from .codec import decode_action
from .engine import apply_action
from .game import GameState, Phase


@dataclass
class Reviewed:
    n: int
    who: str
    age: int
    move: str
    best: str
    v_before: float  # my win prob at the position, searched
    q_move: float  # my win prob of the move played, from that search
    q_best: float  # my win prob of the mover's best move
    choice: float  # mover's regret in the MOVER's win prob: Q(played) - Q(best), <= 0 up to noise
    visits_move: int
    e_after: float | None = None  # mean over reveal worlds
    v_after: float | None = None  # actual reveal
    optimism: float | None = None  # mover's side: how far its search overrated its move
    luck: float | None = None
    worlds: int = 0
    age_end: bool = False  # the move closed an Age: 'after' holds a GUESSED next deal


class Searcher:
    def __init__(self, checkpoint: str, device: str, sims: int, leaf_batch: int = 16):
        self.ad = SevenWondersAdvisor(default_checkpoint=checkpoint, device=device)
        self.device, self.sims, self.leaf = device, sims, leaf_batch
        self.calls = 0

    def search(self, game: GameState, seed: int = 0):
        from games.advisor.contract import RecommendRequest

        req = RecommendRequest(
            max_sims=self.sims, chunk_sims=self.sims, device=self.device, seed=seed,
            options={"leaf_batch": self.leaf},
        )
        handle = self.ad.open_search(_Position(game=game, first_player=game.first_player), req)
        snap = handle.advance(self.sims, threading.Event())
        handle.close()
        self.calls += 1
        return snap

    def value(self, game: GameState, me: int, seed: int = 0) -> float:
        """My win probability at ``game``, searched (exact when finished)."""

        if game.phase is Phase.COMPLETE:
            return 1.0 if game.winner == me else (0.5 if game.winner is None else 0.0)
        snap = self.search(game, seed)
        v = snap.root_value if game.active_player == me else -snap.root_value
        return (v + 1) / 2


def _alternative_reveals(state: GameState, action_index: int, revealed, count: int, seed: int):
    """``state`` after the move, with the uncovered cards redrawn from the
    hidden pool: one board per alternative, each still a valid deal."""

    rng = random.Random(seed)
    backs = {
        s: ("guild" if _is_guild_name(c.card_name) else "age") for s, c in state.tableau.cards.items()
    }
    boards = []
    for k in range(count):
        board = copy.deepcopy(state)
        for slot in revealed:
            here = ("slot", tuple(slot))
            guild = _is_guild_name(_name_at(board, here))
            places = [loc for loc in _hidden_locations(board, backs, (), guild) if loc != here]
            # Only cards still hidden at this moment can come up.
            places = [loc for loc in places if loc[0] != "slot" or loc[1] not in revealed]
            if not places:
                continue
            there = places[(k * 7919 + rng.randrange(len(places))) % len(places)]
            current = _name_at(board, here)
            _set_name(board, here, _name_at(board, there))
            _set_name(board, there, current)
        apply_action(board, decode_action(board, action_index))
        boards.append(board)
    return boards


def review(path: str | Path, searcher: Searcher, worlds: int = 8) -> dict:
    rep = replay_table(searcher.ad, path)
    me = rep.my_seat
    out: list[Reviewed] = []
    for n, d in enumerate(rep.decisions):
        s = d.state
        snap = searcher.search(s, seed=n)
        sign = 1.0 if d.mover == me else -1.0
        mine = lambda q: (sign * q + 1) / 2
        ent = snap.entries
        best_id = max(ent, key=lambda k: ent[k].visits)
        played = ent.get(str(d.action_index))
        row = Reviewed(
            n=n,
            who="you" if d.is_me else "ZeusAI",
            age=s.age,
            move=d.label,
            best=_label(decode_action(s, int(best_id)), s),
            v_before=mine(snap.root_value),
            q_move=mine(played.q_value) if played else float("nan"),
            q_best=mine(ent[best_id].q_value),
            choice=(sign * (mine(played.q_value) - mine(ent[best_id].q_value))) if played else float("nan"),
            visits_move=played.visits if played else 0,
        )
        after = copy.deepcopy(s)
        apply_action(after, decode_action(after, d.action_index))
        row.v_after = searcher.value(after, me, seed=10_000 + n)
        row.age_end = after.age != s.age or after.phase is Phase.CHOOSE_NEXT_START_PLAYER
        if d.revealed:
            alts = _alternative_reveals(s, d.action_index, d.revealed, worlds, seed=n)
            vals = [searcher.value(b, me, seed=20_000 + 97 * n + k) for k, b in enumerate(alts)]
            row.e_after = statistics.fmean(vals + [row.v_after])
            row.worlds = len(vals) + 1
            row.luck = row.v_after - row.e_after
        else:
            row.e_after = row.v_after
            row.luck = 0.0
        row.optimism = sign * (row.q_move - row.e_after) if played and not row.age_end else None
        if row.age_end:
            row.luck = None
        out.append(row)
        print(
            f"{n:3d} A{row.age} {row.who:6s} {row.move[:34]:34s} win {row.v_before:.2f}"
            f" | choice {row.choice:+.2f} | after {row.v_after:.2f}"
            + (f" luck {row.luck:+.2f} ({row.worlds} worlds)" if d.revealed else "")
            + (f" | opt {row.optimism:+.2f}" if row.optimism is not None else ""),
            flush=True,
        )
    return {"table": rep.table_id, "my_seat": me, "gaps": rep.gaps, "rows": [asdict(r) for r in out]}


def summarize(result: dict) -> str:
    rows = result["rows"]
    def tot(who, key):
        keep = lambda r: r[key] is not None and r[key] == r[key] and not r.get("age_end")
        if key in ("choice", "optimism"):  # both read the move's tree value, which is noise when thin
            keep = lambda r, k=keep: k(r) and r["visits_move"] >= 100
        return sum(r[key] for r in rows if r["who"] == who and keep(r))
    lines = [f"table {result['table']}: {len(rows)} decisions reviewed"]
    for who in ("you", "ZeusAI"):
        lines.append(
            f"  {who:6s} move-choice cost (own side) {tot(who, 'choice'):+.2f}"
            f"  |  reveal luck on its moves (YOUR side) {tot(who, 'luck'):+.2f}"
            f"  |  search overrating of its own moves {tot(who, 'optimism'):+.2f}"
        )
    worst = sorted((r for r in rows if not r.get("age_end")), key=lambda r: (r["v_after"] or 0) - r["v_before"])[:6]
    lines.append("  biggest single drops in your win probability:")
    for r in worst:
        lines.append(
            f"    #{r['n']} A{r['age']} {r['who']} {r['move'][:40]}: {r['v_before']:.2f} -> {r['v_after']:.2f}"
            f" (move choice {r['choice']:+.2f}, search overrating {r['optimism'] if r['optimism'] is not None else 0:+.2f},"
            f" reveal luck {r['luck'] or 0:+.2f})"
        )
    for gap in result["gaps"]:
        lines.append(f"  not replayed: {gap}")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sims", type=int, default=3200)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = review(args.table, Searcher(args.checkpoint, args.device, args.sims), args.worlds)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(summarize(result))


if __name__ == "__main__":
    main()
