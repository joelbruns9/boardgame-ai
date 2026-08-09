"""Preconfigured A1b sampled/Hájek × IID/balanced probe matrix.

The shared runner keeps the incumbent x0 arm singular, crosses both backup
estimators and both chance-routing policies for every positive exposure, and
holds simulations/NN evaluations fixed. Command-line values supplied after
these defaults may override them.
"""

from __future__ import annotations

import sys

from games.kingdomino.chance_correct_search_probe import main

A1B_DEFAULT_ARGS = [
    "--backup-modes", "sampled,hajek",
    "--traversal-modes", "iid,balanced",
    "--reference-backups", "sampled,hajek",
    "--reference-traversal", "balanced",
    "--enum-max-rows", "12",
    "--sims", "4800",
    "--reference-sims", "10000",
    "--min-realized-mass", "0.5",
    "--output", "runs/kingdomino/chance_correct_a1/a1b_probe.json",
]


if __name__ == "__main__":
    raise SystemExit(main([*A1B_DEFAULT_ARGS, *sys.argv[1:]]))
