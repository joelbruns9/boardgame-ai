# 7 Wonders Duel AI

Base-game rules engine and self-play AI for the two-player game by Antoine Bauza
and Bruno Cathala.

## Authoritative source

- [Official English rulebook (Asmodee/Repos Production, PDF)](https://cdn.svc.asmodee.net/production-unboxnowcom/uploads/2022/02/7du-rules-us-15990558193s5I6.pdf)
- Rules version used here: English base-game rulebook, copyright 2015, retrieved
  2026-07-10.

The implementation should contain only the structured facts needed to model the
game. Do not add publisher artwork, card scans, or a repackaged copy of the
rulebook. Pantheon, Agora, and solo mode are out of scope until the base engine
passes its rules-oracle tests.

## Rules in one page

Two players develop cities over three Ages. Each Age uses a twenty-card tableau
containing face-up and face-down cards. On a turn, the player selects any card
not covered by another card, reveals newly uncovered cards, and uses the selected
card in exactly one of three ways:

1. Construct its Building and apply its effect.
2. Discard it for `2 + number of yellow Buildings in the player's city` coins.
3. Use it face down to construct one of the player's unbuilt Wonders, paying the
   Wonder's cost rather than the selected card's cost.

Resources are production capacity, not inventory: construction does not consume
them. Each missing resource can be bought from the bank. Its ordinary unit price
is `2 + the opponent's matching production on brown/grey Buildings`. Yellow-card
and Wonder production do not increase that price; certain commercial Buildings
instead fix selected purchase prices at one coin. A chain symbol can make a later
Building free.

The game ends immediately if a player reaches the opponent's capital on the
military track or collects six different science symbols. Otherwise it ends after
Age III and victory points decide the winner. Scoring includes military position,
Buildings, Wonders, Progress tokens, and one point per complete set of three
coins. A tie is broken by points printed on blue Buildings; a second tie is shared.

Important setup and flow constraints:

- Each player starts with seven coins and drafts four of the twelve Wonders.
- Three cards are removed unseen from every Age deck.
- Three of seven Guilds are shuffled into Age III.
- Five of ten Progress tokens are available on the board.
- Only seven total Wonders can be constructed; the eighth becomes unavailable.
- Between Ages, the militarily weaker player chooses the next starting player.
  If the pawn is centered, the player who took the last turn chooses.

## Modeling decisions

- Model setup randomness and face-down tableau cards explicitly. The complete
  simulation state may know shuffled identities, but player observations must not
  expose them before reveal.
- Treat compound effects as pending decisions rather than hiding choices inside
  `apply()`. Examples include destructive Wonders, Mausoleum, Great Library, and
  choosing who starts the next Age.
- Resolve military and science supremacy immediately after each effect that could
  trigger them, before allowing another action.
- Use one structured `Action` type for Wonder drafting, tableau-card uses,
  pending targets, and next-Age starter selection. Add a fixed integer codec only
  after this readable contract passes its engine tests.

## Delivery milestones

1. **Complete:** transcribe card, Wonder, Progress-token, chain, and
   tableau-layout data into typed, auditable tables.
2. **Complete:** build deterministic setup from a seed, the two-round Wonder
   draft, tableau reveal logic, and an observation-safe game state.
3. **Complete:** implement structured legal actions, minimum-cost payments,
   chains, discarding, Building construction, Wonder construction, and pending
   target choices.
4. **Complete:** implement military movement, science pairs, Progress effects,
   immediate victories, Age transitions, civilian scoring, and tie-breaking.
5. **Complete:** rules-oracle matrix for effects plus reproducible scripted
   three-Age games through the public action API.
6. **Complete:** seeded random bot, deterministic one-ply greedy bot, and
   alternating-seat match runner.
7. Add the fixed action codec and state encoder, then integrate MCTS.

The current foundation consists of `rules.py` for core arithmetic, `data.py` for
typed component data, `game.py` for seeded setup and observation-safe state, and
`engine.py` for legal actions and primary action resolution. Source decisions
and normalized naming differences are recorded in `TRANSCRIPTION_NOTES.md`.
Test-layer coverage is summarized in `RULES_ORACLE.md`.
Baseline definitions and the reproducible greedy-versus-random result are in
`BASELINES.md`.
