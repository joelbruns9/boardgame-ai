"""Vector-valued multiplayer MCTS for classic Kingdomino.

Backups preserve one value per present player.  Selection at a decision node
uses the component belonging to that node's actor; there is no two-player sign
flip in this package.
"""
