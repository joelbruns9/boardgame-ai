"""Phase-0 baseline: depth-2 expectiminimax (trivial margin eval) vs GreedyBot,
both seats. Slow in pure Python (~1s/searcher-move); run in the background."""
from games.kingdomino.bot_match import run_match
from games.kingdomino.bots import GreedyBot
from games.kingdomino.expectiminimax import ExpectiminimaxBot

GAMES = 16

run_match("EMM(d2) P0 vs Greedy P1", ExpectiminimaxBot(depth=2), GreedyBot(),
          games=GAMES, seed_offset=1000)
run_match("Greedy P0 vs EMM(d2) P1", GreedyBot(), ExpectiminimaxBot(depth=2),
          games=GAMES, seed_offset=2000)
print("\nDONE")
