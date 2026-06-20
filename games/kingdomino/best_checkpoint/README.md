# Best Checkpoints

## best_32x4.pt
- **Architecture:** 32 channels, 4 residual blocks (32ch/4b)
- **Training:** 4 local runs totaling ~300 training iterations
- **Elo rating:** ~952 (sims=400, vs anchor pool)
- **vs GreedyBot:** 100% win rate, +74 score margin
- **Notes:** Policy loss 1.869 at end of run 4 (lr=3e-4, fpu=-0.2).
  Not fully saturated — still improving at time of archival.
  warm_start from this checkpoint for continued 32ch/4b training
  or as initialization for 48ch/6b scale-up.

Future entries will be added here as the model scales up.
