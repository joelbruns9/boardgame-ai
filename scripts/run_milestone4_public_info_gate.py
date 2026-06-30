from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


COMMANDS = [
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "games/kingdomino/test_milestone4_public_info_gate.py",
    ],
    [sys.executable, "-m", "games.kingdomino.tests.test_encoder"],
    [sys.executable, "-m", "games.kingdomino.tests.test_open_loop_mcts"],
    [sys.executable, "-m", "games.kingdomino.test_rust_mcts_equiv"],
    [sys.executable, "-m", "pytest", "-q", "games/kingdomino/test_endgame_exact.py"],
]


def main() -> int:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    for cmd in COMMANDS:
        print("\n=== " + " ".join(cmd) + " ===", flush=True)
        result = subprocess.run(cmd, cwd=ROOT, env=env)
        if result.returncode != 0:
            print(f"\nMilestone 4 gate FAILED: {' '.join(cmd)}", flush=True)
            return result.returncode
    print("\nMilestone 4 public-info gate PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
