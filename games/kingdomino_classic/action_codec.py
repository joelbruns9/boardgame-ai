"""Stable action encoding for classic Kingdomino.

The codec exposes a fixed four-slot draft axis in both player counts.  In 3p,
all four slots can initially be selected and the sole unclaimed domino is
discarded after the third selection.  In 4p, the fourth selection is forced.
"""
