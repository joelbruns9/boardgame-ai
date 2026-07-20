"""Phase F1 equivalence gates for the Rust 7WD engine (`seven_wonders_rust`).

Method is the Kingdomino `test_rust_*_equiv` discipline (AZ_PROJECT_PLAN §2b,
PHASE_F.md): drive the Rust engine and the Python reference from the same action
sequence and assert bit-for-bit agreement of a language-neutral fingerprint at
*every* decision, not just the end.

- **F1a** — replay agreement: at each move the Rust fingerprint and legal-action
  mask equal Python's; the final state matches. Corpus = real buffer games (all
  phases, all effects) plus random-policy games for breadth.
- **F1b** — make/unmake: `RustGame.roundtrip_ok(index)` (apply then undo)
  restores the fingerprint at every decision.

`logic_fingerprint` mirrors `state.rs::GameState::fingerprint` exactly: the same
field order, numeric-id sorts, and length prefixes, and it excludes Python's RNG
internal state (Rust models no RNG — see PHASE_F.md). Both sides map component
names to ids through the identical `data.py` tables, so id agreement is implied.
"""

from __future__ import annotations

import os
import random

from .buffer import from_json_line
from .codec import decode_action, legal_action_indices
from .data import BackType, CARD_IDS, PROGRESS_IDS, WONDER_IDS, ScienceSymbol
from .engine import apply_action
from .pool import unseen_pool as py_unseen_pool
from .game import (
    ChanceKind,
    Phase,
    PendingChoiceKind,
    VictoryType,
    new_game,
)

import seven_wonders_rust as swr

_SCIENCE_ORDER = {s: i for i, s in enumerate(ScienceSymbol)}
_PHASE_ORD = {
    Phase.WONDER_DRAFT: 0,
    Phase.PLAY_AGE: 1,
    Phase.CHOOSE_NEXT_START_PLAYER: 2,
    Phase.COMPLETE: 3,
}
_VICTORY_ORD = {
    VictoryType.MILITARY: 0,
    VictoryType.SCIENTIFIC: 1,
    VictoryType.CIVILIAN: 2,
    VictoryType.SHARED_CIVILIAN: 3,
}
_PENDING_ORD = {
    PendingChoiceKind.DESTROY_OPPONENT_BROWN: 0,
    PendingChoiceKind.DESTROY_OPPONENT_GREY: 1,
    PendingChoiceKind.BUILD_FROM_DISCARD_FREE: 2,
    PendingChoiceKind.CHOOSE_UNUSED_PROGRESS: 3,
    PendingChoiceKind.CHOOSE_AVAILABLE_PROGRESS: 4,
}
_PROGRESS_PENDING = {
    PendingChoiceKind.CHOOSE_UNUSED_PROGRESS,
    PendingChoiceKind.CHOOSE_AVAILABLE_PROGRESS,
}


def logic_fingerprint(game) -> list[int]:
    """Language-neutral integer fingerprint of all game-logic state.

    Byte-for-byte identical to `state.rs::GameState::fingerprint`.
    """

    out: list[int] = []

    def push_list(names, id_map):
        ids = [id_map[n] for n in names]
        out.append(len(ids))
        out.extend(ids)

    out.append(_PHASE_ORD[game.phase])
    out.append(game.first_player)
    out.append(game.active_player)
    out.append(game.age)
    out.append(game.wonder_round)
    out.append(game.wonder_pick_index)

    for city in game.cities:
        out.append(city.coins)
        push_list(city.wonders, WONDER_IDS)
        push_list(city.built_wonders, WONDER_IDS)
        push_list(city.buildings, CARD_IDS)
        push_list(city.progress_tokens, PROGRESS_IDS)
        pairs = sorted(_SCIENCE_ORDER[s] for s in city.claimed_science_pairs)
        out.append(len(pairs))
        out.extend(pairs)

    push_list(game.available_progress_tokens, PROGRESS_IDS)
    push_list(game.unused_progress_tokens, PROGRESS_IDS)
    push_list(game.wonder_groups[0], WONDER_IDS)
    push_list(game.wonder_groups[1], WONDER_IDS)
    push_list(game.unused_wonders, WONDER_IDS)
    push_list(game.wonder_offer, WONDER_IDS)
    for age in (1, 2, 3):
        push_list(game.age_decks[age], CARD_IDS)
    for age in (1, 2, 3):
        push_list(game.removed_age_cards[age], CARD_IDS)
    push_list(game.selected_guilds, CARD_IDS)
    push_list(game.unused_guilds, CARD_IDS)

    slots = sorted(game.tableau.cards.items())  # by (row, x)
    out.append(len(slots))
    for (row, x), card in slots:
        out.append(row)
        out.append(x)
        out.append(CARD_IDS[card.card_name])
        out.append(int(card.present))
        out.append(int(card.present and card.revealed))

    push_list(game.discard_pile, CARD_IDS)
    push_list(game.buried_cards, CARD_IDS)

    burials = sorted(
        (WONDER_IDS[w], CARD_IDS[c]) for w, c in game.wonder_burials.items()
    )
    out.append(len(burials))
    for w, c in burials:
        out.append(w)
        out.append(c)

    retired = sorted(WONDER_IDS[w] for w in game.retired_wonders)
    out.append(len(retired))
    out.extend(retired)

    pending = game.pending_choice
    if pending is None:
        out.append(-1)
    else:
        out.append(_PENDING_ORD[pending.kind])
        out.append(pending.player)
        out.append(int(pending.consume_all_options))
        id_map = PROGRESS_IDS if pending.kind in _PROGRESS_PENDING else CARD_IDS
        ids = [id_map[o] for o in pending.options]
        out.append(len(ids))
        out.extend(ids)
    out.append(int(game.pending_extra_turn))
    out.append(game.pending_shields)

    out.append(game.conflict_position)
    mil = sorted(game.military_tokens_remaining.items())
    out.append(len(mil))
    for pos, pen in mil:
        out.append(pos)
        out.append(pen)

    out.append(-1 if game.winner is None else game.winner)
    out.append(-1 if game.victory_type is None else _VICTORY_ORD[game.victory_type])
    if game.final_scores is None:
        out.append(-1)
    else:
        out.append(1)
        out.append(game.final_scores[0])
        out.append(game.final_scores[1])
    return out


def extract_setup(game) -> dict:
    """Constructor kwargs for `RustGame` from a fresh Python `GameState`."""

    return {
        "first_player": game.first_player,
        "available_progress": list(game.available_progress_tokens),
        "unused_progress": list(game.unused_progress_tokens),
        "wonder_group0": list(game.wonder_groups[0]),
        "wonder_group1": list(game.wonder_groups[1]),
        "unused_wonders": list(game.unused_wonders),
        "age1": list(game.age_decks[1]),
        "age2": list(game.age_decks[2]),
        "age3": list(game.age_decks[3]),
        "removed1": list(game.removed_age_cards[1]),
        "removed2": list(game.removed_age_cards[2]),
        "removed3": list(game.removed_age_cards[3]),
        "selected_guilds": list(game.selected_guilds),
        "unused_guilds": list(game.unused_guilds),
    }


_BACK_ORDER = (BackType.AGE_I, BackType.AGE_II, BackType.AGE_III, BackType.GUILD)


def _expected_pool(game):
    """Python `unseen_pool` mapped to the sorted id lists the Rust `unseen_pool`
    returns: (age1, age2, age3, guild, wonders, offboard_progress)."""

    up = py_unseen_pool(game.observation(game.active_player))
    cards = tuple(
        tuple(sorted(CARD_IDS[n] for n in up.cards[back])) for back in _BACK_ORDER
    )
    return cards + (
        tuple(sorted(WONDER_IDS[n] for n in up.wonders)),
        tuple(sorted(PROGRESS_IDS[n] for n in up.offboard_progress)),
    )


def compare_game(
    seed,
    first_player,
    action_indices,
    library_draws,
    *,
    check_roundtrip=True,
    deep_every=None,
    check_pool=False,
):
    """Drive Python and Rust from the same action sequence; assert agreement.

    Returns the number of decisions compared. Raises AssertionError with a
    localized message on the first divergence. When ``deep_every`` is set, every
    ``deep_every``-th decision also runs the exhaustive F1b audit
    (`roundtrip_all_ok`): full-state undo + apply determinism over *every* legal
    action to depth 2, not just the trajectory action.
    """

    py = new_game(seed, first_player=first_player)
    setup = extract_setup(py)

    py_fps: list[list[int]] = []
    py_masks: list[list[int]] = []
    py_pools: list[tuple] = []
    for idx in action_indices:
        py_fps.append(logic_fingerprint(py))
        py_masks.append(list(legal_action_indices(py)))
        if check_pool:
            py_pools.append(_expected_pool(py))
        apply_action(py, decode_action(py, idx))
    assert py.phase is Phase.COMPLETE, f"seed {seed}: python game did not complete"
    py_final = logic_fingerprint(py)

    rg = swr.RustGame(library_draws=[list(d) for d in library_draws], **setup)
    for i, idx in enumerate(action_indices):
        rust_fp = rg.fingerprint()
        if rust_fp != py_fps[i]:
            _diff_and_raise(seed, i, py_fps[i], rust_fp)
        rust_mask = rg.legal_action_indices()
        assert rust_mask == py_masks[i], (
            f"seed {seed} move {i}: legal mask mismatch\n"
            f"  python: {py_masks[i]}\n  rust:   {rust_mask}"
        )
        if check_roundtrip:
            assert rg.roundtrip_ok(idx), (
                f"seed {seed} move {i}: make/unmake did not restore state (F1b)"
            )
        if deep_every and i % deep_every == 0:
            assert rg.roundtrip_all_ok(2), (
                f"seed {seed} move {i}: exhaustive make/unmake audit failed (F1b)"
            )
        if check_pool:
            rust_pool = tuple(tuple(lst) for lst in rg.unseen_pool())
            assert rust_pool == py_pools[i], (
                f"seed {seed} move {i}: unseen-pool mismatch (F2.1)\n"
                f"  python: {py_pools[i]}\n  rust:   {rust_pool}"
            )
        rg.apply_index(idx)

    rust_final = rg.fingerprint()
    if rust_final != py_final:
        _diff_and_raise(seed, len(action_indices), py_final, rust_final)
    assert rg.is_complete(), f"seed {seed}: rust game did not complete"
    return len(action_indices)


def _diff_and_raise(seed, move, py_fp, rust_fp):
    where = "final" if move == "final" else f"move {move}"
    n = min(len(py_fp), len(rust_fp))
    first = next((k for k in range(n) if py_fp[k] != rust_fp[k]), n)
    raise AssertionError(
        f"seed {seed} {where}: fingerprint mismatch at index {first} "
        f"(py len {len(py_fp)}, rust len {len(rust_fp)})\n"
        f"  py[{first}:{first + 8}]   = {py_fp[first:first + 8]}\n"
        f"  rust[{first}:{first + 8}] = {rust_fp[first:first + 8]}"
    )


def random_game(seed, first_player):
    """Play a full game under a seeded random-legal policy; collect the action
    sequence and the ordered Great Library draws."""

    game = new_game(seed, first_player=first_player)
    rng = random.Random((seed << 1) ^ 0x9E3779B9 ^ first_player)
    actions: list[int] = []
    library: list[list[str]] = []
    while game.phase is not Phase.COMPLETE:
        legal = list(legal_action_indices(game))
        idx = rng.choice(legal)
        actions.append(idx)
        result = apply_action(game, decode_action(game, idx))
        for ev in result.events:
            if ev.kind is ChanceKind.GREAT_LIBRARY_DRAW:
                library.append(list(ev.outcome))
    return first_player, actions, library


import glob

import pytest

BUFFER_DIR = os.path.join(os.path.dirname(__file__), "runs", "phase_d_toy", "buffers")
# F1a corpus size. Default is a fast-but-real multi-file subset for routine CI;
# the documented ≥10k acceptance gate is `SWR_F1A_GAMES=0 pytest -k buffer`
# (0 = every game in every buffer file). See PHASE_F.md.
F1A_GAMES = int(os.environ.get("SWR_F1A_GAMES", "400"))
F1A_DEEP_GAMES = 4  # games that additionally get the exhaustive depth-2 audit


def iter_buffer_records(limit):
    """Yield decoded records from every buffer .jsonl (sorted), up to `limit`
    games total (`limit <= 0` means all)."""

    paths = sorted(glob.glob(os.path.join(BUFFER_DIR, "*.jsonl")))
    n = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if limit > 0 and n >= limit:
                    return
                if line.strip():
                    yield n, from_json_line(line)
                    n += 1


def _library_draws(record):
    return [
        outcome
        for kind, outcome in record.chance_log
        if kind == ChanceKind.GREAT_LIBRARY_DRAW.value
    ]


# --- pytest entry points ------------------------------------------------------


def test_random_games_equivalent():
    total = 0
    for seed in range(60):
        fp = seed % 2
        first_player, actions, library = random_game(seed, fp)
        # Exhaustively audit make/unmake on a few of these games.
        deep_every = 4 if seed < F1A_DEEP_GAMES else None
        total += compare_game(
            seed, first_player, actions, library, deep_every=deep_every
        )
    assert total > 0


def test_buffer_games_equivalent():
    """F1a: byte-exact replay over the buffer corpus.

    Skips (does not silently pass) when buffers are absent — they live under the
    gitignored ``runs/`` and are present on the gate box, not a fresh checkout.
    Runs ``SWR_F1A_GAMES`` games across all buffer files (default 400; set 0 for
    the full ≥10k acceptance gate) and asserts the corpus was non-empty.
    """

    if not os.path.isdir(BUFFER_DIR) or not glob.glob(
        os.path.join(BUFFER_DIR, "*.jsonl")
    ):
        pytest.skip(f"no buffer corpus under {BUFFER_DIR} (F1a needs replay buffers)")

    n = 0
    for index, record in iter_buffer_records(F1A_GAMES):
        deep_every = 25 if index < F1A_DEEP_GAMES else None
        compare_game(
            record.seed,
            record.first_player,
            [m.action for m in record.moves],
            _library_draws(record),
            deep_every=deep_every,
        )
        n += 1
    assert n > 0, "buffer corpus present but yielded no games"
    if F1A_GAMES > 0:
        assert n == min(F1A_GAMES, n), "fewer games compared than requested"


def test_unseen_pool_equivalent():
    """F2.1: Rust `unseen_pool` (encoder foundation) matches Python's public
    projection at every decision, across all phases (random games span draft →
    all three ages → endgame)."""

    total = 0
    for seed in range(40):
        first_player, actions, library = random_game(seed, seed % 2)
        total += compare_game(
            seed, first_player, actions, library, check_pool=True
        )
    assert total > 0


def test_apply_index_rejects_illegal_action():
    """The public apply boundary must reject a non-legal index rather than
    mutating state through an unchecked decode (e.g. an unowned wonder)."""

    py = new_game(0, first_player=0)
    rg = swr.RustGame(library_draws=[], **extract_setup(py))
    legal = set(rg.legal_action_indices())
    illegal = next(i for i in range(swr.num_actions()) if i not in legal)
    with pytest.raises(ValueError):
        rg.apply_index(illegal)
    # State is unchanged and a legal action still applies.
    assert rg.legal_action_indices() == sorted(legal)
    rg.apply_index(next(iter(legal)))


def test_generated_rust_data_matches_python():
    """`src/data_gen.rs` must equal a fresh `export_rust_data.generate()` — makes
    "cannot drift" enforced, not just documented."""

    import importlib.util

    gen_py = os.path.join(
        os.path.dirname(__file__), "seven_wonders_rust", "export_rust_data.py"
    )
    spec = importlib.util.spec_from_file_location("swr_export_rust_data", gen_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    gen_rs = os.path.join(
        os.path.dirname(__file__), "seven_wonders_rust", "src", "data_gen.rs"
    )
    with open(gen_rs, "r", encoding="utf-8") as fh:
        on_disk = fh.read()
    fresh = module.generate()
    assert fresh.replace("\r\n", "\n") == on_disk.replace("\r\n", "\n"), (
        "src/data_gen.rs is stale — regenerate with "
        "`python -m games.seven_wonders_duel.seven_wonders_rust.export_rust_data`"
    )
