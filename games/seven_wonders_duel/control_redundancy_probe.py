"""Does the network already know positional control? (Workstream 3 go/no-go)

The question that decides whether control facts belong in the encoder is NOT
whether they correlate with action regret. Control is a precondition rather than
a predictor -- losing needs control conceded AND a threatening card present, so
neither correlates alone -- and regret measured against searches by the CURRENT
model is circular anyway, since that model cannot see control.

The question is: **is this information already in the representation?**

Freeze the trunk, attach a small head, and train it to predict the control
features from the pooled readout the network already produces. Then compare
against a baseline that sees only the label distribution.

  probe recovers control well   -> the trunk already encodes it; adding explicit
                                   features spends width on something the model
                                   infers for itself
  probe recovers it poorly      -> the information is genuinely absent, and the
                                   feature adds something the network cannot
                                   currently represent

Trunk weights never receive gradient, so this measures what the existing
`candidate_0085` representation contains, not what it could learn to contain.

Targets are the exact solver's outputs on each position (`tableau_control`), all
of them public-information topology facts. Positions come from the threat corpus
plus ordinary self-play positions, because a probe trained only on threat
positions would measure recall on a narrow slice rather than the representation
in general.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import torch
from torch import nn

from .advisor_scrape import determinize_observation, observation_from_wire
from .bga_extract import wire_from_bga_payload
from .codec import decode_action, legal_action_indices
from .encoder import encode
from .engine import apply_action
from .game import Phase, new_game
from .search import state_actor
from .tableau_control import control_features
from .threat_corpus_scan import REPO_ROOT

# The control facts worth asking about. Fractions in [0, 1] and small counts.
TARGETS = (
    "control_now",
    "control_with_one_more_tempo",
    "control_if_opponent_takes_theology",
    "my_tempo",
    "their_tempo",
)
_COUNT_SCALE = 4.0  # tempo budgets are 0..~4; keep every target on a like scale


def label_vector(features: dict) -> list[float] | None:
    if not features:
        return None
    out = []
    for name in TARGETS:
        value = features.get(name)
        if value is None:
            return None
        out.append(
            float(value) if name.startswith("control")
            else float(value) / _COUNT_SCALE
        )
    return out


def corpus_positions(limit: int | None):
    """Threat-corpus positions: the slice the feature is meant to serve."""

    path = REPO_ROOT / "runs/seven_wonders_duel/threat_corpus/episodes.json"
    if not path.exists():
        return []
    corpus = json.loads(path.read_text(encoding="utf-8"))
    seen, out = set(), []
    for episode in corpus["episodes"]:
        for snapshot in episode["snapshots"]:
            key = (episode["table"], snapshot["decision_row"])
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
            if limit and len(out) >= limit:
                return out
    return out


def load_logged(table: str, row: int):
    path = REPO_ROOT / f"runs/seven_wonders_duel/bga_game_log/table_{table}.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    decisions = [r for r in rows if r.get("kind") == "decision"]
    payload = wire_from_bga_payload(decisions[row]["state"])
    obs = observation_from_wire(payload["observation"])
    return determinize_observation(
        obs, random.Random(0),
        unknown_burial_ages=tuple(
            int(a) for a in payload.get("unknown_burial_ages", ())
        ),
    )


def selfplay_positions(count: int, seed: int):
    """Ordinary positions, so the probe measures the representation broadly.

    A probe trained only where threats exist would report recall on a narrow
    slice and say little about whether the trunk encodes control generally.
    """

    rng = random.Random(seed)
    out = []
    for game_seed in range(count * 3):
        if len(out) >= count:
            break
        game = new_game(game_seed, first_player=game_seed % 2)
        # A new game starts in the Wonder draft, so play THROUGH whatever phase
        # is current rather than stopping at the first non-PLAY_AGE one.
        for _ in range(rng.randrange(10, 55)):
            if game.phase is Phase.COMPLETE:
                break
            legal = legal_action_indices(game)
            if not legal:
                break
            apply_action(game, decode_action(game, rng.choice(legal)))
        if game.phase is Phase.PLAY_AGE and game.tableau.accessible_slot_ids():
            out.append(game.clone())
    return out


def build_dataset(args, log):
    from .dataset import collate_inputs, vectorize

    games = []
    for table, row in corpus_positions(args.corpus_limit):
        try:
            games.append(load_logged(table, row))
        except Exception:
            continue
    log(f"  {len(games)} corpus positions")
    extra = selfplay_positions(args.selfplay, args.seed)
    log(f"  {len(extra)} self-play positions")
    games.extend(extra)

    encodings, labels, legal = [], [], []
    for game in games:
        features = control_features(game, state_actor(game))
        vector = label_vector(features)
        if vector is None:
            continue
        encoding = encode(game.observation(state_actor(game)))
        encodings.append(vectorize(encoding))
        legal.append(legal_action_indices(game))
        labels.append(vector)
    log(f"  {len(labels)} usable examples")
    batch = collate_inputs(encodings, legal, device="cpu")
    return batch, torch.tensor(labels, dtype=torch.float32)


def token_features(model, batch, chunk=64):
    """Masked mean+max over ALL trunk tokens, not just the pooled readout.

    Control is a spatial property of the tableau, and the pooled readout is a
    bottleneck the value and policy heads see. Probing both separates two
    different questions:

      pooled  -- can the HEADS use control? (what matters for the current net)
      tokens  -- is control in the REPRESENTATION at all? (what matters for
                 whether a new feature adds information, or whether a better
                 readout would suffice)

    If control is recoverable from tokens but not from the pooled readout, the
    fix is the readout, not an encoder feature.
    """

    model.eval()
    outs = []
    total = batch["pad_mask"].shape[0]
    with torch.no_grad():
        for start in range(0, total, chunk):
            piece = {k: v[start:start + chunk] for k, v in batch.items()}
            tokens = model.embedder(piece)
            encoded = model.encoder(tokens, src_key_padding_mask=piece["pad_mask"])
            normed = model.final_norm(encoded)
            real = ~piece["pad_mask"]
            counts = real.sum(1, keepdim=True).clamp(min=1)
            weights = real.unsqueeze(-1)
            mean = (normed * weights).sum(1) / counts
            maxed = normed.masked_fill(~weights, float("-inf")).max(1).values
            outs.append(torch.cat([normed[:, 0], mean, maxed], dim=-1))
    return torch.cat(outs)


def fit_probe(features, labels, args, seed):
    """Train one probe with early stopping on a validation split."""

    torch.manual_seed(seed)
    order = torch.randperm(len(labels))
    n_test = int(len(labels) * args.holdout)
    n_val = max(20, int(len(labels) * 0.15))
    test, val, train = order[:n_test], order[n_test:n_test + n_val], order[n_test + n_val:]

    probe = nn.Sequential(
        nn.Linear(features.shape[1], args.hidden), nn.ReLU(), nn.Dropout(args.dropout),
        nn.Linear(args.hidden, args.hidden), nn.ReLU(), nn.Dropout(args.dropout),
        nn.Linear(args.hidden, len(TARGETS)),
    )
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val, best_state, stale = float("inf"), None, 0
    for _ in range(args.epochs):
        probe.train()
        optimizer.zero_grad()
        nn.functional.mse_loss(probe(features[train]), labels[train]).backward()
        optimizer.step()
        probe.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(probe(features[val]), labels[val]).item()
        if v < best_val - 1e-6:
            best_val, stale = v, 0
            best_state = {k: t.clone() for k, t in probe.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        predicted, actual = probe(features[test]), labels[test]
    r2 = []
    for i in range(len(TARGETS)):
        ss_res = ((predicted[:, i] - actual[:, i]) ** 2).sum().item()
        ss_tot = ((actual[:, i] - actual[:, i].mean()) ** 2).sum().item()
        r2.append(1 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan"))
    return r2


def readout(model, batch, chunk=64):
    """The pooled representation the heads see. Trunk frozen throughout."""

    model.eval()
    outs = []
    total = batch["pad_mask"].shape[0]
    with torch.no_grad():
        for start in range(0, total, chunk):
            piece = {k: v[start:start + chunk] for k, v in batch.items()}
            tokens = model.embedder(piece)
            encoded = model.encoder(tokens, src_key_padding_mask=piece["pad_mask"])
            normed = model.final_norm(encoded)
            if model.readout_proj is None:
                outs.append(normed[:, 0])
            else:
                real = ~piece["pad_mask"]
                counts = real.sum(1, keepdim=True).clamp(min=1)
                weights = real.unsqueeze(-1)
                mean = (normed * weights).sum(1) / counts
                maxed = normed.masked_fill(~weights, float("-inf")).max(1).values
                outs.append(model.readout_proj(
                    torch.cat([normed[:, 0], mean, maxed], dim=-1)
                ))
    return torch.cat(outs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default="extension_7wd/candidate_0085.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-migration", action="store_true")
    parser.add_argument("--corpus-limit", type=int, default=120)
    parser.add_argument("--selfplay", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    log = (lambda *_: None) if args.quiet else (
        lambda m: print(m, file=sys.stderr, flush=True)
    )

    from .phase_e import load_evaluator

    path = Path(args.checkpoint)
    if not path.is_absolute():
        path = REPO_ROOT / path
    evaluator = load_evaluator(str(path), args.device, migrate=args.allow_migration)
    model = evaluator.model

    log("building dataset")
    batch, labels = build_dataset(args, log)
    if len(labels) < 40:
        raise SystemExit("too few examples to probe")

    log("extracting frozen readout")
    features = readout(model, batch)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    sources = {"pooled": features, "tokens": token_features(model, batch)}
    report = {"harness": "control_redundancy_probe", "examples": len(labels),
              "checkpoint": str(path.name), "seeds": args.seeds,
              "probe": {"hidden": args.hidden, "layers": 2,
                        "dropout": args.dropout, "early_stopping": True},
              "sources": {}}

    for source_name, source in sources.items():
        runs = [fit_probe(source, labels, args, args.seed + k) for k in range(args.seeds)]
        log("")
        log(f"--- probe source: {source_name} (dim {source.shape[1]}) ---")
        log(f"{'target':<40}{'R2 mean':>9}{'min':>8}{'max':>8}")
        per = {}
        for i, name in enumerate(TARGETS):
            vals = [r[i] for r in runs if r[i] == r[i]]
            if not vals:
                continue
            per[name] = {"r2_mean": round(statistics.fmean(vals), 3),
                         "r2_min": round(min(vals), 3), "r2_max": round(max(vals), 3)}
            log(f"{name:<40}{per[name]['r2_mean']:>9.2f}"
                f"{per[name]['r2_min']:>8.2f}{per[name]['r2_max']:>8.2f}")
        report["sources"][source_name] = per

    log("")
    log("Read the ORDERING against my_tempo/their_tempo, the most directly")
    log("derivable targets here -- they mark what 'the model has this' looks")
    log("like for this probe, which is not 1.0.")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
