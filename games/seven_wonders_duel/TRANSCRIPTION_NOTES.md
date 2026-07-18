# Base-game transcription notes

## Sources

The authoritative source for rules, Building cards, Guild cards, Progress
tokens, Wonder effects, chain links, and Age layouts is the [official English
rulebook](https://cdn.svc.asmodee.net/production-unboxnowcom/uploads/2022/02/7du-rules-us-15990558193s5I6.pdf).

The rulebook does not contain a Wonder-cost table. Its page 6 setup photograph
shows eight Wonder faces clearly enough to verify their costs. The other four
costs - Great Library, Hanging Gardens, Mausoleum, and Sphinx - were visually
checked against [this photograph of all twelve physical Wonder
cards](https://artemisgames.co.nz/cdn/shop/products/81RMtBue-pL._AC_SL1500_1024x1024@2x.jpg?v=1612693854).
The photograph is a transcription reference only and is not stored in the
project.

## Source inconsistencies normalized in code

- The physical Guild card is named `Merchants Guild`, while the effect glossary
  calls it `Traders Guild`. `Merchants Guild` is canonical and the other wording
  is retained as a source alias.
- The printed card list spells `Pretorium` without the historically usual `ae`.
  The physical spelling is canonical and `Praetorium` is retained as an alias.
- The card face uses `Glass-Blower`; `Glassblower` is retained as an alias.

## Data conventions

- Production is capacity and does not accumulate or get consumed.
- `fixed_production` affects an opponent's normal trade price only for brown and
  grey cards. `choice_production` from yellow cards or Wonders does not.
- A chain is stored as the printed symbol identifier. `chain_to` grants a symbol;
  `chain_from` consumes that prerequisite to make construction free.
- Guild coin effects occur once at construction. Guild victory-point effects are
  evaluated at final scoring and may select a different qualifying city.
- Tableau `x` values use integer half-columns. A card is covered by a card in the
  next row whose coordinate differs by one.

## Remaining engineering work

- Randomized invariant/fuzz games beyond the deterministic oracle scenarios.
- Baseline bots, fixed action codec, encoder, search integration, and self-play.

Observation-state handling is now implemented in `game.py`. Player observations
omit future Wonder offers during the draft, all setup-removed cards, all future
Age decks, unused Guilds and Progress tokens, and identities of face-down tableau
cards until they become accessible.

Primary and outcome resolution is implemented in `engine.py`. It includes optimal
payments across fixed and choice production, commercial discounts, Architecture
and Masonry rebates, Economy transfers, free chains and Urbanism, Building and
Guild coin effects, discarding, Wonder construction, the seven-Wonder cap, and
public pending choices for destructive, discard-recovery, and Great Library
effects, military movement and one-time coin penalties, science pairs, Progress
selection, military and scientific supremacy, Age changes, and civilian scoring.

The military track uses signed positions `-9..9`: zero is neutral, positive
positions favor player 0, and either endpoint is the opposing capital. The 2-coin
tokens trigger on first entry to `-4` and `4`; the 5-coin tokens trigger on first
entry to `-7` and `7`. Civilian military points are 2 for distances 1-3, 5 for
distances 4-6, and 10 for distances 7-8.

Wonder drafting, primary Age actions, pending choices, and next-Age starter
selection all use the same structured `Action` API. This is the canonical action
contract for the upcoming codec.
