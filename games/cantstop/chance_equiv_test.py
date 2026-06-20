#!/usr/bin/env python
"""
chance_equiv_test.py — Gate 2 for gated exact chance nodes.

The premise of exact chance is that exact-Q (low sims) computes the value
that pure-sampled search converges to. This harness checks that directly,
at the search's ROOT VALUE on cap decision states:

    v_exact  = root value with EXACT on,  at low_sims   (e.g. 1200)
    v_oracle = root value with EXACT off, at high_sims   (e.g. 6000)

If exact is correct, |v_exact - v_oracle| is within sampling noise, with NO
systematic offset. A consistent signed bias is the fingerprint of a
perspective-flip or weighting bug — exactly what the isolated self-tests
can't see. Run this AFTER integrating ChanceNode into mcts.py.

Set the tolerance from the noise floor: two independent sampled@high runs on
the same positions disagree by some amount purely from sampling; exact-vs-
oracle error should be no worse than that. measure_noise_floor() does this.

WIRING: make_root_value_fn() is wired to your MCTS (MCTS.search -> (policy,
value)). If your tree renamed things, the only lines to touch are the two
imports and the engine.search call.
Then:  python -m games.cantstop.chance_equiv_test --model <path>
Without --model it runs the offline self-tests (logic + position collection).
"""

import argparse
import random
import statistics

from games.cantstop.engine import get_valid_moves, MAX_RUNNERS
from games.cantstop.chance_exact_sketch import collect_cap_positions


# ============================================================
# Adapter — the ONLY MCTS-touching surface. Wire to your tree.
# ============================================================

def make_root_value_fn(model_path, device="cpu",
                       target_inflight=1, warmup_sims=16):
    """
    Return root_value(state, exact: bool, sims: int) -> float in [0, 1]:
    the search's value estimate for `state` in its active player's
    perspective.

    Wired to the real MCTS API:
        MCTS(model, device, target_inflight, warmup_sims)
        .search(state, num_simulations, dirichlet_alpha, dirichlet_epsilon)
            -> (policy, value);  value == root.Q (win prob for root's player).

    Dirichlet root noise is disabled (epsilon=0) so we compare the *pure*
    search value, not exploration jitter. target_inflight=1 + warmup keeps
    the estimate clean (no async pile-up on a single path).

    If your tree renamed things, the only lines to touch are the two imports
    and the engine.search call.
    """
    from games.cantstop import mcts as M
    from games.cantstop.mcts import MCTS
    from games.cantstop.evaluate import load_model

    model = load_model(model_path, device)

    def root_value(state, exact, sims):
        M.EXACT_CHANCE_ENABLED = exact
        engine = MCTS(model, device,
                      target_inflight=target_inflight, warmup_sims=warmup_sims)
        _policy, value = engine.search(
            state.clone(), num_simulations=sims, dirichlet_epsilon=0.0,
        )
        return value

    return root_value


# ============================================================
# Comparison + summary (pure; unit-tested offline)
# ============================================================

def summarize_equiv(signed_errors, tol):
    """signed_errors: list of (v_exact - v_oracle). Returns error stats and a
    pass flag. A nonzero mean (bias) is the flip/weight-bug fingerprint."""
    n = len(signed_errors)
    if n == 0:
        return {}
    abs_errs = [abs(e) for e in signed_errors]
    abs_errs_sorted = sorted(abs_errs)
    p95 = abs_errs_sorted[min(n - 1, int(round(0.95 * (n - 1))))]
    bias = statistics.fmean(signed_errors)
    return {
        "n": n,
        "mean_abs": statistics.fmean(abs_errs),
        "p95_abs": p95,
        "max_abs": max(abs_errs),
        "bias": bias,                              # signed mean
        "frac_within_tol": sum(1 for e in abs_errs if e <= tol) / n,
        "pass": (p95 <= tol and abs(bias) <= tol / 2),
    }


def run_equivalence(root_value, num_positions=20, low_sims=1200,
                    high_sims=6000, tol=0.03, seed=0):
    """
    Two questions on the same cap positions and oracle:
      correctness — does exact@low match the converged oracle? (bias ~ 0)
      payoff      — does exact@low track the oracle BETTER than sampled@low,
                    i.e. is the variance win real at the budget you train at?
    """
    rng = random.Random(seed)
    positions = collect_cap_positions(2000, rng, num_positions)
    total = len(positions) * (2 * low_sims + high_sims)
    print(f"\n  Q-equivalence + variance: {len(positions)} positions, "
          f"exact@{low_sims} & sampled@{low_sims} vs oracle@{high_sims} "
          f"(~{total:,} sims total)")
    exact_err, sampled_err = [], []
    for i, st in enumerate(positions, 1):
        v_oracle = root_value(st.clone(), False, high_sims)
        v_exact = root_value(st.clone(), True, low_sims)
        v_samp = root_value(st.clone(), False, low_sims)
        exact_err.append(v_exact - v_oracle)
        sampled_err.append(v_samp - v_oracle)
        print(f"    [{i:>3}/{len(positions)}] exact={v_exact:.3f} "
              f"sampled={v_samp:.3f} oracle={v_oracle:.3f}  "
              f"|d_ex|={abs(v_exact - v_oracle):.3f} "
              f"|d_sa|={abs(v_samp - v_oracle):.3f}")

    se = summarize_equiv(exact_err, tol)
    ss = summarize_equiv(sampled_err, tol)
    print(f"\n  Correctness (exact@{low_sims} vs oracle, tol {tol}):")
    print(f"    bias {se['bias']:+.4f}  (nonzero => perspective/weight bug) | "
          f"{'PASS' if se['pass'] else 'FAIL'}")
    print(f"\n  Value error vs oracle@{high_sims}:")
    print(f"    exact@{low_sims}    mean|err| {se['mean_abs']:.4f} | "
          f"p95 {se['p95_abs']:.4f} | max {se['max_abs']:.4f}")
    print(f"    sampled@{low_sims}  mean|err| {ss['mean_abs']:.4f} | "
          f"p95 {ss['p95_abs']:.4f} | max {ss['max_abs']:.4f}")
    if se['mean_abs'] > 1e-9:
        ratio = ss['mean_abs'] / se['mean_abs']
        if ratio >= 1.5:
            print(f"    => exact cuts value error ~{ratio:.1f}x at {low_sims} "
                  f"sims — the variance win is real at your training budget.")
        else:
            print(f"    => exact ~ sampled at {low_sims} sims ({ratio:.1f}x). "
                  f"Dice variance is largely resolved at this budget; the gain")
            print(f"       lives at LOWER budgets (e.g. the advisor's 500), not "
                  f"the {low_sims} you train at.")
    return {'exact': se, 'sampled': ss}


def measure_noise_floor(root_value, num_positions=20, high_sims=6000,
                        tol=0.03, seed=0):
    """Sampled@high vs sampled@high (different seeds) — sets the realistic
    tolerance. exact-vs-oracle error should be no worse than this."""
    rng = random.Random(seed)
    positions = collect_cap_positions(2000, rng, num_positions)
    print(f"\n  Noise floor: {len(positions)} positions, "
          f"sampled@{high_sims} vs itself "
          f"(~{len(positions) * 2 * high_sims:,} sims total)")
    signed = []
    for i, st in enumerate(positions, 1):
        a = root_value(st.clone(), False, high_sims)
        b = root_value(st.clone(), False, high_sims)
        signed.append(a - b)
        print(f"    [{i:>3}/{len(positions)}] {a:.3f} vs {b:.3f}  "
              f"d={a - b:+.3f}")
    s = summarize_equiv(signed, tol)
    print(f"\n  Noise floor (sampled@{high_sims} vs itself): "
          f"mean|err| {s['mean_abs']:.4f} | p95 {s['p95_abs']:.4f}")
    return s


# ============================================================
# Offline self-tests (no model needed)
# ============================================================

def _selftest():
    # 1. cap positions are valid crunch states.
    pos = collect_cap_positions(200, random.Random(0), 50)
    assert pos and all(len(p.runners) == MAX_RUNNERS for p in pos)
    assert all(get_valid_moves(p) for p in pos)
    print(f"  cap collector: {len(pos)} valid crunch states  OK")

    # 2. summary passes on small unbiased noise.
    rng = random.Random(1)
    small = [rng.uniform(-0.02, 0.02) for _ in range(300)]
    s = summarize_equiv(small, tol=0.03)
    assert s["pass"], f"expected pass, got {s}"
    print(f"  summary: small unbiased noise -> PASS "
          f"(p95 {s['p95_abs']:.3f}, bias {s['bias']:+.3f})  OK")

    # 3. summary FAILS on a systematic bias (the flip-bug signature).
    biased = [0.08 + rng.uniform(-0.01, 0.01) for _ in range(300)]
    s2 = summarize_equiv(biased, tol=0.03)
    assert not s2["pass"] and s2["bias"] > 0.03
    print(f"  summary: systematic bias -> FAIL (bias {s2['bias']:+.3f})  OK")

    # 4. adapter is concretely wired to MCTS.search (not a stub).
    import inspect
    src = inspect.getsource(make_root_value_fn)
    assert "NotImplementedError" not in src, "adapter still stubbed"
    assert "engine.search" in src, "adapter must call MCTS.search"
    print("  adapter wired to MCTS.search (no stub)  OK")

    print("All equivalence-harness self-tests passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gate 2: exact/sampled Q-equivalence")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--positions", type=int, default=20)
    ap.add_argument("--low-sims", type=int, default=1200, dest="low_sims")
    ap.add_argument("--high-sims", type=int, default=6000, dest="high_sims")
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--noise-floor", action="store_true", dest="noise_floor",
                    help="also measure the sampled-vs-sampled noise floor")
    args = ap.parse_args()

    if args.model:
        rv = make_root_value_fn(args.model, args.device)
        if args.noise_floor:
            measure_noise_floor(rv, args.positions, args.high_sims, args.tol)
        run_equivalence(rv, args.positions, args.low_sims, args.high_sims, args.tol)
    else:
        _selftest()