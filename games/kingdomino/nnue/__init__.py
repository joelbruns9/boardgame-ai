"""NNUE evaluation pipeline for the alpha-beta searcher.

Generalization note (see games/kingdomino/NNUE_PROJECT_PLAN.md): the *net* and
*trainer* here are game-agnostic — they consume a flat feature matrix + two label
vectors (official outcome, score margin) and know nothing about Kingdomino. Only
the data loader (`data.py`) is game-specific (it reads the Kingdomino self-play
buffer). When a second game arrives, it supplies its own loader and reuses `net.py`
+ the training loop unchanged.
"""
