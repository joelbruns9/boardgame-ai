"""Stable action encoding for classic Kingdomino.

The codec will expose a fixed four-slot draft axis.  The fourth slot is masked
for three-player games so one network and replay schema can serve both player
counts.
"""
