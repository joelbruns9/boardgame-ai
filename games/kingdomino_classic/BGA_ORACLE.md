# BGA implementation oracle

The base-game rules oracle is the checked-in BGA implementation under:

```text
BGA Files/kingdomino/kingdomino.game.php
BGA Files/kingdomino/material.inc.php
BGA Files/kingdomino/gameoptions.json
```

The K0 engine intentionally excludes Lost Treasures. The following BGA
behaviors are authoritative for the current scope:

| Behavior | BGA implementation |
|---|---|
| Use all 48 dominoes for 3p and 4p | `setupNewGame` |
| Draw four sorted dominoes each row | `drawDominoes` |
| Discard the ownerless fourth domino after three 3p claims | `drawDominoes`, `remainsFutureDomino` |
| Claim in table order during initial setup | `activateOwnerOfNextKing` |
| Later acting order follows ascending claimed domino number | `getOwnerOfNextCurrentDomino` |
| A turn places the current domino and claims from the future row | `placeDomino`, `routeAfterPlacement` |
| A claimed domino may be discarded only when no placement exists | `discardDomino`, `dominoCanBePlaced` |
| Placement requires empty cells, a terrain/castle connection, and a 5x5 fit | `dominoFitsInPosition` |
| Score connected terrain size times crowns | `getKingdomTerritories`, `getKingdomScore` |
| Rank by score, largest territory, then total crowns | `finalScoring`, `gameinfos.inc.php` |
| Harmony is +5 when none of that player's claimed dominoes were discarded | `finalScoring` |
| Middle Kingdom is +10 when the castle is centered in the occupied bounds | `finalScoring` |

Tests should cite the corresponding behavior in their names or comments when
they are direct equivalence anchors. Any intentional departure from these
semantics must be documented here before changing the engine.
