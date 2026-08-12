# K2 Player-Equivariant Encoder and Action Codec

Encoder version: `1`. Action codec version: `1`.

The encoder represents players on one padded axis. It does not create `self`,
`left opponent`, `right opponent`, or seat-specific feature blocks. Actor,
initial-pick priority, and pending-claim priority are attributes attached to a
player slot. Consequently, relabeling players only permutes that axis.

The rules state stores the initial draft order explicitly rather than deriving
it from numeric player IDs. This lets arbitrary S3/S4 relabelings move turn
priority with the player, including partway through the initial draft.

## Tensor contract

| Tensor | Shape | Role |
|---|---:|---|
| `boards` | `[4, 8, 9, 9]` | Shared castle, terrain, and crown planes |
| `player_features` | `[4, 18]` | Presence, actor/order roles, score and board summaries |
| `player_dominoes` | `[4, 48, 4]` | Placed, pending, next, and forced-discard ownership |
| `domino_features` | `[48, 18]` | Ordered half material, number, and public global zone |
| `global_features` | `[8]` | Phase, player count, Harmony, and Middle Kingdom |

The 9x9 board is centred on the castle and covers coordinates `-4..4` on both
axes. Every legal 5x5 kingdom fits without translation or cropping. The crown
plane is scaled by three; terrain and castle planes are one-hot.

Every domino occupies exactly one public inventory zone: deck membership
without deck order, current draft, unclaimed discard, or one player's placed,
pending, next, or forced-discard bank. The hidden shuffled deck order is never
encoded.

In 3p, slot 3 is zero in all three player-axis tensors. The global player-count
feature is the only explicit non-player-axis indicator of the player count.

## Exact symmetry contract

For any real-player permutation `P` in S3 or S4:

```text
encode(state, player_order=P) == P(encode(state))
```

For any board transform `D` in D4:

```text
encode(D(state)) == D(encode(state))
```

Player permutations and D4 transforms commute exactly, and each D4 transform
has a byte-exact inverse. Tests exercise every S3/S4 permutation and every D4
element over initial, middle, final-placement, and terminal trajectory states.

## Network requirement

The encoder makes exact equivariance possible; K3 must preserve it. Player
boards and player features must pass through shared weights. Cross-player
interaction must use a permutation-equivariant operation with the presence
mask, and value outputs must retain the player axis. Flattening the four player
slots into position-specific dense weights is prohibited. Policy output is
invariant under player relabeling because the acting-player flag moves with the
actor; vector value output is equivariant.

## Action contract

The policy has 2,170 fixed entries:

```text
434 placement entries x 5 pick entries
```

The placement axis contains 432 spatial entries, forced discard, and initial
draft/no-placement. Spatial entries are the 144 undirected adjacent edges on
the 9x9 canvas times three half-assignment modes: A at the canonical endpoint,
B at the canonical endpoint, or identical halves. The identical mode removes
duplicate moves while remaining fixed under endpoint reversal; the other modes
swap when a D4 transform reverses an edge.

The pick axis contains the four original sorted draft-row slots plus `NO_PICK`
for final placement. Already claimed dominoes remain part of the reconstructed
row, so available tile indices never shift during sequential drafting. This
gives 3p two legal slots before its third claim and one forced original slot
before the 4p fourth claim.

Legal actions round-trip exactly, legal masks commute with every D4 transform,
and the eight policy transforms are bijective with byte-exact inverses.
