"""Is `value_acc` a bottleneck, or is it 7WD's intrinsic uncertainty?

The value head has sat at ~0.715-0.74 across five runs regardless of
regularisation, and it is upstream of everything: value quality bounds search
quality, which bounds target quality, which bounds the policy. But a plateau is
only a bottleneck if a better number was available. 7WD is high variance -- if a
mid-game position genuinely only determines the winner ~75% of the time, then
0.74 is near-optimal and the value head is innocent.

Two references make that decidable without any ground truth, both computed on
the same positions:

  ply stratification    Late positions are nearly decided. If accuracy climbs to
                        ~0.97 by the final moves, the head answers knowable
                        questions correctly and the early uncertainty is the
                        game. If it stalls at ~0.80 into a decided endgame, it
                        is failing on positions whose answer is available, and
                        the encoder is genuinely implicated.

  search vs raw net     `root_value` is the same network's value backed up over
                        a full search. It is what the head could be if it could
                        compute. The gap is the value-side analogue of the
                        policy improvement operator: wide means search knows
                        things the head has not absorbed; nil means the head has
                        already extracted everything the search does.

Runs on captured buffers. One forward pass per position, no search, no rental.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .buffer import from_json_line
from .dataset import collate, examples_from_record, is_fast_search_move
from .train import build_model, heads_from_config

# Example.value_class: 0 win / 1 draw / 2 loss, actor-relative.
WIN, DRAW, LOSS = 0, 1, 2


def _rows(buffers: Path, games: int, from_iteration: int):
    """Yield (example, ply) pairs, replaying each game through the verified path.

    `examples_from_record` emits one example per recorded decision in move
    order, skipping cheap searches. Pairing against the same filter recovers the
    ply, and the assert makes a silent misalignment impossible rather than
    quietly mislabelling every stratum.
    """

    seen = 0
    paths = [
        p for p in sorted(buffers.glob("iter_*.jsonl"))
        if int(p.stem.split("_")[1]) >= from_iteration
    ]
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if seen >= games:
                    return
                record = from_json_line(line)
                examples = examples_from_record(record)
                plies = [m.i for m in record.moves if not is_fast_search_move(m)]
                assert len(examples) == len(plies), (
                    f"{len(examples)} examples against {len(plies)} recorded "
                    "decisions -- the ply pairing is not 1:1"
                )
                total = len(record.moves)
                for example, ply in zip(examples, plies):
                    yield example, ply, total
                seen += 1


def run(checkpoint: Path, buffers: Path, *, games: int, device: str, batch: int, from_iteration: int = 0) -> dict:
    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = stored.get("config", {})
    model = build_model(
        "transformer",
        int(config.get("d_model", 384)),
        int(config.get("layers", 8)),
        heads_from_config(config),
    )
    model.load_state_dict(stored["model_state"])
    model.to(device).eval()

    # Ten buckets of game progress. Absolute ply would mix a 55-move military
    # rush with a 76-move civilian game at the same index.
    buckets = [
        {"n": 0, "net": 0, "search": 0, "search_n": 0, "certain": 0}
        for _ in range(10)
    ]
    pending: list[tuple] = []

    def flush():
        if not pending:
            return
        batched = collate([p[0] for p in pending], device=device)
        with torch.no_grad():
            predicted = model(batched)["value"].argmax(dim=-1).cpu()
        for (example, ply, total), guess in zip(pending, predicted.tolist()):
            slot = buckets[min(9, int(10 * ply / max(total, 1)))]
            slot["n"] += 1
            slot["net"] += int(guess == example.value_class)
            if example.root_value is not None:
                slot["search_n"] += 1
                implied = WIN if example.root_value > 0 else LOSS
                slot["search"] += int(implied == example.value_class)
                slot["certain"] += int(abs(example.root_value) > 0.9)
        pending.clear()

    for row in _rows(buffers, games, from_iteration):
        pending.append(row)
        if len(pending) >= batch:
            flush()
    flush()

    report = {
        "checkpoint": str(checkpoint.resolve()),
        "buffers": str(buffers.resolve()),
        "games": games,
        "from_iteration": from_iteration,
        "positions": sum(b["n"] for b in buckets),
        "by_progress": [],
    }
    print(f"{'game progress':>14} {'n':>7} {'net value acc':>14} {'search acc':>11} {'|rv|>0.9':>9}")
    for index, slot in enumerate(buckets):
        if not slot["n"]:
            continue
        net = slot["net"] / slot["n"]
        search = slot["search"] / slot["search_n"] if slot["search_n"] else None
        certain = slot["certain"] / slot["search_n"] if slot["search_n"] else None
        report["by_progress"].append(
            {
                "bucket": f"{index * 10}-{index * 10 + 10}%",
                "n": slot["n"],
                "net_value_acc": net,
                "search_value_acc": search,
                "share_root_value_confident": certain,
            }
        )
        print(
            f"{f'{index * 10}-{index * 10 + 10}%':>14} {slot['n']:>7,} {net:>13.3f} "
            f"{(f'{search:.3f}' if search is not None else '-'):>11} "
            f"{(f'{certain:.3f}' if certain is not None else '-'):>9}"
        )

    overall = sum(b["net"] for b in buckets) / max(sum(b["n"] for b in buckets), 1)
    late = buckets[-2]["net"] + buckets[-1]["net"]
    late_n = buckets[-2]["n"] + buckets[-1]["n"]
    report["overall_net_value_acc"] = overall
    report["final_20pct_net_value_acc"] = late / late_n if late_n else None
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--buffers", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument(
        "--from-iteration",
        type=int,
        default=0,
        help="skip earlier buffers; positions from iteration 0 were generated by the seed net, not the checkpoint under test",
    )
    args = parser.parse_args(argv)

    report = run(
        args.checkpoint,
        args.buffers,
        games=args.games,
        device=args.device,
        batch=args.batch,
        from_iteration=args.from_iteration,
    )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"\noverall {report['overall_net_value_acc']:.3f} | "
        f"final 20% of the game {report['final_20pct_net_value_acc']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
