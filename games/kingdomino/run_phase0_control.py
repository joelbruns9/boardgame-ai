"""Phase-0 control (review finding 3): does cheap PICK-AWARENESS close the gap to
GreedyBot, or is a learned eval specifically needed?

Paired: pick-blind and pick-aware evals are run on the SAME seed set and BOTH
seats, so the only variable is the leaf evaluator. Depth 2 throughout."""
from games.kingdomino.bot_match import run_match
from games.kingdomino.bots import GreedyBot
from games.kingdomino.expectiminimax import ExpectiminimaxBot, pick_aware_p0

GAMES = 12
CS = 8  # chance_samples (smaller for speed; both configs identical)


def blind():
    return ExpectiminimaxBot(depth=2, chance_samples=CS)  # tanh_margin default


def aware():
    return ExpectiminimaxBot(depth=2, chance_samples=CS, eval_fn=pick_aware_p0)


for name, make in (("BLIND", blind), ("AWARE", aware)):
    run_match(f"EMM({name},d2) P0 vs Greedy P1", make(), GreedyBot(),
              games=GAMES, seed_offset=0)
    run_match(f"Greedy P0 vs EMM({name},d2) P1", GreedyBot(), make(),
              games=GAMES, seed_offset=0)
print("\nDONE")
