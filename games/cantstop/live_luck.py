"""Dice luck so far in a live game, for the advisor panel.

Reads the table's game log -- the BGA packet stream the extension records
(``bga_packets`` rows) -- replays it, and runs the luck ledger over every
turn so far (games/cantstop/luck.py). Per player:

    busts / busts_expected   model-free: actual busts vs the exact bust odds
                             of every roll that player made
    dice_pts                 win-probability points this player gained or
                             lost to ALL dice (their rolls and everyone
                             else's), each roll measured against the exact
                             average over the 1296 possible rolls
    own_rolls_pts            the part from their own rolls

Only a stream that covers the game from its first roll is used: a partial
one cannot place the board, and guessing would put wrong numbers in front of
the player. Solves are cached per turn-start board, so a call late in a game
re-solves nothing it has already solved.
"""

import json
import threading
from pathlib import Path

from games.advisor.game_log import GameLogWriter, log_dir_for

from .luck import build_ledger, load_game, read_turns, seat_summary, z
from .snapshot import snapshot


class LiveLuck:
    def __init__(self, advisor, log_dir=None, max_solvers=400):
        self.advisor = advisor
        self.writer = GameLogWriter(log_dir or log_dir_for(advisor.game_id))
        self.max_solvers = max_solvers
        self._solvers, self._table = {}, None
        self._lock = threading.Lock()

    def summary(self, table_id, device="cuda"):
        path = self.writer.path_for(table_id)
        if not Path(path).is_file():
            return {"available": False, "reason": "nothing logged for this table yet"}
        try:
            game = load_game(path)
        except ValueError as exc:
            return {"available": False, "reason": str(exc)}
        turns = read_turns(path, game)
        if game.source != "packets":
            return {"available": False,
                    "reason": "the game record does not reach back to the first roll "
                              "(reload the BGA tab: BGA re-sends the history)"}
        from games.advisor.contract import RecommendRequest
        evaluate = self.advisor.evaluator(RecommendRequest(engine="auto", device=device))
        with self._lock:
            if self._table != table_id:
                self._solvers, self._table = {}, table_id
            ledger = build_ledger(game, turns, evaluate, solver_cls=self._solver)
        players = []
        for seat, pid in enumerate(game.player_ids):
            s = seat_summary(ledger, seat)
            players.append({
                "player_id": pid, "name": game.names.get(pid, pid),
                "rolls": int(s["rolls"]), "busts": int(s["busts"]),
                "busts_expected": round(s["busts_expected"], 2),
                "busts_z": round(z(s["busts"], s["busts_expected"], s["busts_var"]), 2),
                "dice_pts": round(100 * (s["luck_own"] + s["luck_others"]), 1),
                "own_rolls_pts": round(100 * s["luck_own"], 1)})
        return {"available": True, "turns": len(turns),
                "game_over": bool(turns and turns[-1].end == "win"), "players": players}

    def _solver(self, start, evaluate):
        """Turn-start solves, cached by board (under ``self._lock``)."""
        from .rust_solver import RustTurnSolver
        key = json.dumps(snapshot(start))
        solver = self._solvers.get(key)
        if solver is None:
            if len(self._solvers) >= self.max_solvers:
                self._solvers.pop(next(iter(self._solvers)))
            with self.advisor._lock:          # the GPU evaluator is shared
                solver = self._solvers[key] = RustTurnSolver(start, evaluate)
        return solver
