"""
Phase-sliced calibration diagnostics for the Kingdomino AlphaZero training loop.

All functions take a list of Example objects (CPU, float16 boards) and a
KingdominoNet (eval mode, on device). They return plain Python dicts of scalar
metrics suitable for JSON serialisation.

Phase definitions (from flat[FLAT_LAYOUT['game_progress']] = placed_cells / 96):
  opening:  game_progress < 0.33  (~tiles 1-10 of 12 rounds, placed < 32 cells)
  midgame:  0.33 <= game_progress < 0.75  (~placed 32-72 cells)
  endgame:  game_progress >= 0.75  (~placed 72+ cells; deck=0 or deck=4 positions)

The endgame slice overlaps exactly with positions where the exact endgame solver
fires (deck in {0,4}). This is intentional — endgame calibration is the primary
target of Milestone 2.

This module has NO side effects on import and does not import self_play at
runtime (only for type-checking), so self_play can import it without a cycle.
"""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np
import torch

from games.kingdomino.encoder import FLAT_LAYOUT

if TYPE_CHECKING:                       # avoid a runtime import cycle
    from games.kingdomino.self_play import Example
    from games.kingdomino.network import KingdominoNet


# Minimum slice size below which a phase metric is reported as None (insufficient
# sample to be meaningful / to avoid mean-of-empty NaNs).
_MIN_SLICE = 10

PHASE_THRESHOLDS = {
    "opening": (0.0,  0.33),
    "midgame": (0.33, 0.75),
    "endgame": (0.75, 1.01),  # 1.01 to include game_progress=1.0
}
_PHASES = ("opening", "midgame", "endgame")


# ─── phase slicing ─────────────────────────────────────────────────────────
def _progress(examples: List["Example"]) -> np.ndarray:
    """game_progress scalar for each example (float32 array)."""
    prog_slice = FLAT_LAYOUT["game_progress"]
    return np.array([float(ex.flat[prog_slice][0]) for ex in examples],
                    dtype=np.float64)


def _phase_mask(examples: List["Example"], phase: str) -> np.ndarray:
    """Boolean mask over examples selecting the given phase."""
    lo, hi = PHASE_THRESHOLDS[phase]
    progress = _progress(examples)
    return (progress >= lo) & (progress < hi)


# ─── shared network forward helpers ────────────────────────────────────────
def _collate(batch: List["Example"], device: str):
    """Stack the board/flat arrays of `batch` into float32 device tensors."""
    mb = np.stack([ex.my_board.astype(np.float32) for ex in batch])
    ob = np.stack([ex.opp_board.astype(np.float32) for ex in batch])
    flat = np.stack([ex.flat.astype(np.float32) for ex in batch])
    to = lambda a: torch.from_numpy(a).to(device)
    return to(mb), to(ob), to(flat)


def _forward_heads(examples: List["Example"], net, device: str,
                   batch_size: int):
    """Run the scalar heads over all examples in minibatches.

    Returns (own_pred, opp_pred, win_prob) as float64 numpy arrays of shape (N,).
    own_pred/opp_pred are the network's normalized score outputs (raw_score /
    score_scale); win_prob is in (0, 1).
    """
    net.eval()
    own_chunks, opp_chunks, win_chunks = [], [], []
    with torch.no_grad():
        for s in range(0, len(examples), batch_size):
            batch = examples[s:s + batch_size]
            mb, ob, flat = _collate(batch, device)
            own, opp, win, _ = net(mb, ob, flat)
            own_chunks.append(own.detach().cpu().numpy().astype(np.float64))
            opp_chunks.append(opp.detach().cpu().numpy().astype(np.float64))
            win_chunks.append(win.detach().cpu().numpy().astype(np.float64))
    return (np.concatenate(own_chunks), np.concatenate(opp_chunks),
            np.concatenate(win_chunks))


def _policy_stats(examples: List["Example"], net, device: str,
                  batch_size: int):
    """Per-example policy comparison between the MCTS target and the raw network.

    Returns (net_top1, mcts_top1, kl) as numpy arrays of shape (N,):
      net_top1  : joint action index the network assigns highest masked logit
      mcts_top1 : joint action index with most MCTS visit mass (argmax policy_val)
      kl        : KL(mcts_policy || net_policy) over legal actions, per example
    """
    net.eval()
    net_top1 = np.empty(len(examples), dtype=np.int64)
    mcts_top1 = np.empty(len(examples), dtype=np.int64)
    kl = np.empty(len(examples), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(examples), batch_size):
            batch = examples[s:s + batch_size]
            mb, ob, flat = _collate(batch, device)
            _, _, _, logits = net(mb, ob, flat)
            logits = logits.detach().cpu().numpy().astype(np.float64)
            for j, ex in enumerate(batch):
                i = s + j
                legal = np.asarray(ex.legal_idx, dtype=np.int64)
                pidx = np.asarray(ex.policy_idx, dtype=np.int64)
                pval = np.asarray(ex.policy_val, dtype=np.float64)
                ll = logits[j, legal]
                # stable log-softmax denominator over legal actions
                m = ll.max()
                logZ = m + np.log(np.exp(ll - m).sum())
                net_logp_pol = logits[j, pidx] - logZ
                net_top1[i] = int(legal[int(np.argmax(ll))])
                mcts_top1[i] = int(pidx[int(np.argmax(pval))])
                # KL(p||q) = Σ p (log p − log q), summed over support of p only.
                kl[i] = float(np.sum(pval * (np.log(pval) - net_logp_pol)))
    return net_top1, mcts_top1, kl


def _pred_z(own_pred: np.ndarray, opp_pred: np.ndarray, win_prob: np.ndarray,
            alpha: float, margin_gain: float) -> np.ndarray:
    """Reconstruct the network's combined value target z from its heads.

    z = alpha*tanh((own_norm - opp_norm)*margin_gain) + (1-alpha)*(2*win - 1)
    own_pred/opp_pred are already normalized network outputs (own_score/scale).
    """
    margin = np.tanh((own_pred - opp_pred) * margin_gain)
    return alpha * margin + (1.0 - alpha) * (2.0 * win_prob - 1.0)


# ─── Metric 1: win Brier by phase ──────────────────────────────────────────
def win_brier_by_phase(examples: List["Example"], net, device: str,
                       batch_size: int = 512) -> dict:
    """MSE of win_prob vs win_target, sliced by phase.

    win_brier_endgame is the primary signal for the alpha transition trigger;
    baseline_brier_endgame enables the ratio win_brier_endgame /
    baseline_brier_endgame.
    """
    _, _, win_prob = _forward_heads(examples, net, device, batch_size)
    win_t = np.array([float(ex.win_target) for ex in examples], dtype=np.float64)

    result: dict = {}
    for phase in _PHASES:
        mask = _phase_mask(examples, phase)
        n = int(mask.sum())
        result[f"n_{phase}"] = n
        if n < _MIN_SLICE:
            result[f"win_brier_{phase}"] = None
        else:
            result[f"win_brier_{phase}"] = float(
                np.mean((win_prob[mask] - win_t[mask]) ** 2))

    end_mask = _phase_mask(examples, "endgame")
    if int(end_mask.sum()) >= _MIN_SLICE:
        base = float(win_t[end_mask].mean())
        result["baseline_brier_endgame"] = float(base * (1.0 - base))
    else:
        result["baseline_brier_endgame"] = None
    return result


# ─── Metric 2: margin MAE by phase ─────────────────────────────────────────
def margin_mae_by_phase(examples: List["Example"], net, device: str,
                        batch_size: int = 512,
                        score_scale: float = 100.0) -> dict:
    """MAE of the own_score head vs actual own_score (in raw score points)."""
    own_pred, _, _ = _forward_heads(examples, net, device, batch_size)
    own_score = np.array([float(ex.own_score) for ex in examples],
                         dtype=np.float64)
    pred_pts = own_pred * score_scale

    result: dict = {}
    for phase in _PHASES:
        mask = _phase_mask(examples, phase)
        if int(mask.sum()) < _MIN_SLICE:
            result[f"margin_mae_{phase}"] = None
        else:
            result[f"margin_mae_{phase}"] = float(
                np.mean(np.abs(pred_pts[mask] - own_score[mask])))
    return result


# ─── Metric 3: value calibration curve ─────────────────────────────────────
def value_calibration_curve(examples: List["Example"], net, device: str,
                            n_bins: int = 10, batch_size: int = 512) -> dict:
    """Calibration of win_prob: actual win rate vs predicted, in equal-width bins.

    cal_ece = Σ_b (n_b / N) * |actual_rate_b - pred_mean_b|.
    """
    _, _, win_prob = _forward_heads(examples, net, device, batch_size)
    win_t = np.array([float(ex.win_target) for ex in examples], dtype=np.float64)
    N = len(examples)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize → bin index in [0, n_bins-1]; clip the win_prob==1.0 edge case.
    bin_idx = np.clip(np.digitize(win_prob, edges) - 1, 0, n_bins - 1)

    pred_mean: List[Optional[float]] = []
    actual_rate: List[Optional[float]] = []
    counts: List[int] = []
    ece = 0.0
    for b in range(n_bins):
        sel = bin_idx == b
        c = int(sel.sum())
        counts.append(c)
        if c == 0:
            pred_mean.append(None)
            actual_rate.append(None)
            continue
        pm = float(win_prob[sel].mean())
        ar = float(win_t[sel].mean())
        pred_mean.append(pm)
        actual_rate.append(ar)
        if N > 0:
            ece += (c / N) * abs(ar - pm)

    return {
        "cal_bin_edges": [float(e) for e in edges],
        "cal_pred_mean": pred_mean,
        "cal_actual_rate": actual_rate,
        "cal_n": counts,
        "cal_ece": float(ece),
    }


# ─── Metric 4: root value vs final margin (signed bias) ────────────────────
def root_value_vs_final_margin(examples: List["Example"], net, device: str,
                               batch_size: int = 512,
                               alpha: float = 0.8,
                               margin_gain: float = 2.0) -> dict:
    """Mean signed error (pred_z - z) overall and per phase, plus MAE.

    Positive bias = the value head overestimates; negative = underestimates.
    """
    own_pred, opp_pred, win_prob = _forward_heads(examples, net, device,
                                                  batch_size)
    pred_z = _pred_z(own_pred, opp_pred, win_prob, alpha, margin_gain)
    z = np.array([float(ex.z) for ex in examples], dtype=np.float64)
    err = pred_z - z

    result = {
        "value_bias_overall": float(err.mean()) if len(err) else None,
        "value_mae_overall": float(np.abs(err).mean()) if len(err) else None,
    }
    for phase in _PHASES:
        mask = _phase_mask(examples, phase)
        if int(mask.sum()) < _MIN_SLICE:
            result[f"value_bias_{phase}"] = None
        else:
            result[f"value_bias_{phase}"] = float(err[mask].mean())
    end_mask = _phase_mask(examples, "endgame")
    result["value_mae_endgame"] = (
        float(np.abs(err[end_mask]).mean())
        if int(end_mask.sum()) >= _MIN_SLICE else None)
    return result


# ─── Metric 5: exact endgame value error (proxy = endgame slice) ───────────
def exact_endgame_value_error(examples: List["Example"], net, device: str,
                              batch_size: int = 512,
                              alpha: float = 0.8,
                              margin_gain: float = 2.0) -> dict:
    """pred_z vs ex.z error on the endgame slice (proxy for exact-solver targets).

    Uses the endgame slice (game_progress >= 0.75), which overlaps with where
    the exact solver fires. Becomes precise once Example carries an is_exact flag.
    """
    end_mask = _phase_mask(examples, "endgame")
    n = int(end_mask.sum())
    if n < _MIN_SLICE:
        return {"exact_value_mae": None, "exact_value_bias": None,
                "n_endgame": n}

    end_examples = [ex for ex, m in zip(examples, end_mask) if m]
    own_pred, opp_pred, win_prob = _forward_heads(end_examples, net, device,
                                                  batch_size)
    pred_z = _pred_z(own_pred, opp_pred, win_prob, alpha, margin_gain)
    z = np.array([float(ex.z) for ex in end_examples], dtype=np.float64)
    err = pred_z - z
    return {
        "exact_value_mae": float(np.abs(err).mean()),
        "exact_value_bias": float(err.mean()),
        "n_endgame": n,
    }


# ─── Metric 6: MCTS lift rate (net top-1 vs MCTS top-1) ────────────────────
def mcts_lift_rate(examples: List["Example"], net, device: str,
                   batch_size: int = 512) -> dict:
    """Fraction of positions where MCTS top-1 differs from raw-network top-1.

    High lift = search substantially corrects the prior; low lift = the prior
    already matches the search policy.
    """
    net_top1, mcts_top1, _ = _policy_stats(examples, net, device, batch_size)
    lift = (net_top1 != mcts_top1).astype(np.float64)

    result = {
        "mcts_lift_rate_overall": float(lift.mean()) if len(lift) else None,
    }
    for phase in _PHASES:
        mask = _phase_mask(examples, phase)
        if int(mask.sum()) < _MIN_SLICE:
            result[f"mcts_lift_rate_{phase}"] = None
        else:
            result[f"mcts_lift_rate_{phase}"] = float(lift[mask].mean())
    return result


# ─── Metric 7: policy KL by phase ──────────────────────────────────────────
def policy_kl_by_phase(examples: List["Example"], net, device: str,
                       batch_size: int = 512) -> dict:
    """Mean KL(mcts_policy || net_policy) overall and per phase.

    High KL = network prior far from the MCTS policy; decreasing over training
    means the network is internalising the search signal.
    """
    _, _, kl = _policy_stats(examples, net, device, batch_size)

    result = {
        "policy_kl_overall": float(kl.mean()) if len(kl) else None,
    }
    for phase in _PHASES:
        mask = _phase_mask(examples, phase)
        if int(mask.sum()) < _MIN_SLICE:
            result[f"policy_kl_{phase}"] = None
        else:
            result[f"policy_kl_{phase}"] = float(kl[mask].mean())
    return result


# ─── diagnostic batch construction ─────────────────────────────────────────
def build_diag_batch(examples: List["Example"], n: int = 1024,
                     seed: int = 42) -> List["Example"]:
    """Sample a fixed subset of examples for network-inference diagnostics.

    Same seed → same positions each iteration (as the buffer rotates the sample
    changes gradually, which is fine).
    """
    rng = np.random.default_rng(seed)
    if len(examples) <= n:
        return list(examples)
    idx = rng.choice(len(examples), size=n, replace=False)
    return [examples[i] for i in idx]


# ─── main entry point ──────────────────────────────────────────────────────
def compute_all_diagnostics(examples: List["Example"], net, device: str,
                            score_scale: float = 100.0,
                            margin_gain: float = 2.0,
                            alpha: float = 0.8,
                            diag_n: int = 1024,
                            batch_size: int = 512) -> dict:
    """Run all diagnostics; return a flat, JSON-serialisable dict of metrics.

    Network-inference metrics run on a fixed subsample of size diag_n. Missing
    values (insufficient data for a slice) are represented as None.
    """
    diag = build_diag_batch(examples, n=diag_n)

    result: dict = {}
    result.update(win_brier_by_phase(diag, net, device, batch_size))
    result.update(margin_mae_by_phase(diag, net, device, batch_size,
                                       score_scale=score_scale))
    result.update(value_calibration_curve(diag, net, device,
                                           batch_size=batch_size))
    result.update(root_value_vs_final_margin(diag, net, device, batch_size,
                                              alpha=alpha,
                                              margin_gain=margin_gain))
    result.update(exact_endgame_value_error(diag, net, device, batch_size,
                                             alpha=alpha,
                                             margin_gain=margin_gain))
    result.update(mcts_lift_rate(diag, net, device, batch_size))
    result.update(policy_kl_by_phase(diag, net, device, batch_size))
    return result


# ─── alpha transition trigger (stub) ───────────────────────────────────────
def check_alpha_transition(log_rows: List[dict], window: int = 5,
                           threshold: float = 0.5) -> bool:
    """Empirical trigger for the alpha transition (alpha=0.8 → alpha=0.0).

    Fires when win_brier_endgame / baseline_brier_endgame < threshold sustained
    over `window` consecutive diagnostic iterations. threshold=0.5 means the win
    head is >50% better than a naive constant predictor on endgame positions.

    Stub: the training loop logs the result but does not act on it (Milestone 5).
    """
    recent = [r for r in log_rows[-window * 5:]
              if r.get("win_brier_endgame") is not None
              and r.get("baseline_brier_endgame") is not None
              and r["baseline_brier_endgame"] > 0]
    if len(recent) < window:
        return False
    recent = recent[-window:]
    ratios = [r["win_brier_endgame"] / r["baseline_brier_endgame"]
              for r in recent]
    return all(ratio < threshold for ratio in ratios)
