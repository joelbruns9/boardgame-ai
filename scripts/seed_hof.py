from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from games.kingdomino.hof import DEFAULT_HOF_DIR, add_hof_entry
from games.kingdomino.promotion import DEFAULT_CURRENT_BEST


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy a promoted or manually approved checkpoint into the Kingdomino HOF pool."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_CURRENT_BEST)
    parser.add_argument("--hof_dir", type=Path, default=DEFAULT_HOF_DIR)
    parser.add_argument("--tag", default="current_best")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    metadata = {"note": args.note} if args.note else None
    entry = add_hof_entry(
        args.source,
        hof_dir=args.hof_dir,
        tag=args.tag,
        iteration=args.iteration,
        metadata=metadata,
    )
    print(json.dumps(entry.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
