# Rules-oracle coverage

The suite is organized in layers so failures identify whether the defect is in
transcribed data, setup, action resolution, or game outcomes.

| Test module | Contract covered |
| --- | --- |
| `test_data.py` | Component counts, unique names, science pairs, chains, and all three tableau layouts |
| `test_rules.py` | Trade arithmetic, discard income, and treasury conversion |
| `test_game.py` | Seeded setup, Wonder draft order, hidden information, accessibility, and reveals |
| `test_engine.py` | Payments, choice production, discounts, chains, the three card uses, pending choices, Economy, and the seven-Wonder cap |
| `test_outcomes.py` | Military boundaries, science/Progress timing, supremacy, Age changes, scoring bands, tie-breaking, and shared victories |
| `test_effect_matrix.py` | Dynamic commercial cards, every Guild formula, all Wonders, replay/destruction ordering, resource Wonders, and Progress timing/scoring |
| `test_full_game.py` | Reproducible 8-pick Wonder drafts and complete 60-card, three-Age games through `legal_actions()` and `apply_action()` only |
| `test_bots.py` | Clone independence, seeded decisions, non-mutating greedy selection, terminal matches, and series accounting |

The scripted full-game oracle intentionally discards every Age card. That makes
its expected result simple and independent of shuffle order: each player takes
30 turns, finishes with 67 coins, scores 22 treasury points, and shares the
civilian victory. Rich card combinations remain covered by focused effect tests,
where failures are easier to diagnose.

Run the complete game suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest games\seven_wonders_duel -q
```
