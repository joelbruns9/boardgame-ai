"""Precomputed exact positional control (Workstream 3), as a shipped table.

`tableau_control.ControlSolver` is exact but far too slow to run at encode time:
`control_features()` costs 816 ms mean and 5.8 s worst on real positions, against
a leaf evaluation of order 1 ms. It does not need to run there. The solver reads
only the removal poset, turn order and the shared Wonder pool, so its answer is a
function of PUBLIC STRUCTURE alone -- and that structure has a small, closed key
space:

    (age, present_mask, who_moves, tempo_state)

The pyramid is far more constrained than ``2**20`` suggests. Only 428 / 428 / 132
masks are reachable per age, so the whole table is under ten megabytes and takes
minutes to build. Encode time becomes an array read, the solver never enters the
training loop, and Rust needs a table reader rather than a port of the search.

**This module owns the CONTRACT, not just the bytes.** A table is only
interchangeable with a checkpoint if the feature meaning behind it is identical,
so the manifest pins the schema version, the layout identity, the rule-data
identity, the tempo enumeration and a digest of the contents. A git commit is
provenance, not a compatibility check: a no-op refactor that produces identical
bytes must not invalidate a checkpoint, and a solver change that alters one entry
must.

Cell values (uint8), per slot:

    0-97    attacker turns to force the take
    ABSENT  the slot is not present in this mask
    UNREACH the slot is present and the attacker cannot force it

`UNREACH` is deliberately NOT a large distance. `_INF` is 99, and feeding that as
a turn count would swamp every other input; the encoder is expected to emit a
reachable flag and a scaled distance that is zero when the flag is zero.

Build:

    python -m games.seven_wonders_duel.control_table build --jobs 12

Read:

    table = ControlTable.load()
    cells = table.lookup(age, mask, who_moves, tempo)   # 20 uint8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .tableau_control import (
    ATTACKER,
    DEFENDER,
    ControlSolver,
    Layout,
    _INF,
    _as_theology,
    tempo_state,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "games/seven_wonders_duel/testdata/control_table"

#: Bump when the MEANING of a cell changes. Checkpoints record this.
SCHEMA_VERSION = 1

ABSENT = 255
UNREACH = 254
MAX_TURNS = 97          # anything above this would collide with the sentinels

AGES = (1, 2, 3)


# --------------------------------------------------------------------------
# The key space
# --------------------------------------------------------------------------


def reachable_masks(age: int) -> list[int]:
    """Every present-mask reachable from a full pyramid by legal removals.

    This is the whole reason the table is small: the cover relation is a partial
    order, so the reachable set is its down-set, not the power set.
    """

    layout = Layout.for_age(age)
    full = (1 << len(layout.slots)) - 1
    seen = {full}
    frontier = [full]
    while frontier:
        nxt = []
        for mask in frontier:
            accessible = layout.accessible(mask)
            while accessible:
                bit = accessible & -accessible
                accessible ^= bit
                child = mask ^ bit
                if child not in seen:
                    seen.add(child)
                    nxt.append(child)
        frontier = nxt
    return sorted(seen)


def tempo_states() -> list[tuple]:
    """Every tempo state a real game can hold, in a fixed canonical order.

    Each player drafts four Wonders, so an unbuilt count runs 0..4 split between
    ordinary and extra-turn. Seven of the eight get built, which is why
    ``builds_left == total_unbuilt - 1`` -- an identity checked against 289 real
    BGA positions, not assumed. States with no builds left are excluded: the
    pool is closed, every counter is zero, and one all-zero entry would be
    duplicated 220 times.

    The set is CLOSED under `_as_theology` (which only moves ordinary counts
    into extra ones), so the Theology counterfactual maps natural states to
    natural states and adds no keys.
    """

    out = []
    for ord_a in range(5):
        for ext_a in range(5 - ord_a):
            for ord_d in range(5):
                for ext_d in range(5 - ord_d):
                    total = ord_a + ext_a + ord_d + ext_d
                    if not total:
                        continue
                    builds = min(7, total - 1)
                    if builds <= 0:
                        continue
                    out.append((builds, ord_a, ext_a, ord_d, ext_d))
    out = sorted(set(out))
    # The closure claim is load-bearing for the key count; assert it here rather
    # than in a comment.
    known = set(out)
    for state in out:
        for player in (ATTACKER, DEFENDER):
            if _as_theology(state, player) not in known:
                raise AssertionError(
                    f"Theology closure escapes the natural set: {state} -> "
                    f"{_as_theology(state, player)}"
                )
    return out


TEMPO_STATES = tempo_states()
TEMPO_INDEX = {state: i for i, state in enumerate(TEMPO_STATES)}


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def _solve_one_tempo(args) -> tuple:
    """All masks x both movers, for one tempo state. One shard of work.

    Sharding on tempo rather than on masks is deliberate: within a shard the
    solver's memo is keyed on `(present, target, to_move, tempo)`, so holding
    tempo fixed lets deeper masks reuse the work done for shallower ones. That
    is the only direction reuse pays -- see `test_reuse_saves_work`.
    """

    age, tempo_i = args
    tempo = TEMPO_STATES[tempo_i]
    layout = Layout.for_age(age)
    masks = reachable_masks(age)
    width = len(layout.slots)
    out = np.full((len(masks), 2, width), ABSENT, dtype=np.uint8)

    for who in (0, 1):
        first = ATTACKER if who == 0 else DEFENDER
        solver = ControlSolver(age)
        for mask_i, mask in enumerate(masks):
            for slot_i, slot in enumerate(layout.slots):
                if not (mask >> slot_i) & 1:
                    continue
                turns = solver.solve(mask, slot, first, tempo)
                if turns >= _INF:
                    out[mask_i, who, slot_i] = UNREACH
                elif turns > MAX_TURNS:
                    raise AssertionError(f"turn count {turns} collides with sentinels")
                else:
                    out[mask_i, who, slot_i] = turns
    return age, tempo_i, out


def rule_identity() -> str:
    """Digest of the code and data the table's MEANING depends on.

    Not a git commit: a refactor that leaves every byte identical must not
    invalidate a checkpoint, and a solver change that alters one entry must.
    The digest covers the solver and the layout/wonder data it reads.
    """

    digest = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in ("tableau_control.py", "data.py"):
        digest.update((here / name).read_bytes())
    return digest.hexdigest()


def build(out_dir: Path, jobs: int = 1, log=print, *,
          ages: tuple = AGES, tempo_indices: list | None = None) -> dict:
    """Generate the table and its manifest. Returns the manifest.

    `ages` and `tempo_indices` exist so a SUBSET table can be built cheaply --
    the test suite generates one so that a clean checkout still exercises
    agreement with the solver, rather than skipping the whole file because an
    8.7 MB artifact is absent.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tempo_indices = (list(range(len(TEMPO_STATES))) if tempo_indices is None
                     else list(tempo_indices))
    tasks = [(age, i) for age in ages for i in tempo_indices]
    planes: dict[int, np.ndarray] = {}
    masks_by_age = {age: reachable_masks(age) for age in ages}
    slot_of = {t: i for i, t in enumerate(tempo_indices)}
    for age in ages:
        width = len(Layout.for_age(age).slots)
        planes[age] = np.full(
            (len(masks_by_age[age]), 2, len(tempo_indices), width),
            ABSENT, dtype=np.uint8,
        )

    total_keys = sum(len(m) for m in masks_by_age.values()) * 2 * len(tempo_indices)
    log(f"{len(tasks)} shards, {total_keys:,} keys "
        f"({', '.join(f'age {a}: {len(m)} masks' for a, m in masks_by_age.items())})")

    done = 0
    if jobs > 1:
        import multiprocessing as mp

        with mp.Pool(jobs) as pool:
            for age, tempo_i, plane in pool.imap_unordered(
                _solve_one_tempo, tasks, chunksize=1
            ):
                planes[age][:, :, slot_of[tempo_i], :] = plane
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    log(f"  {done}/{len(tasks)} shards "
                        f"({(time.perf_counter() - started) / 60:.1f} min)")
    else:
        for task in tasks:
            age, tempo_i, plane = _solve_one_tempo(task)
            planes[age][:, :, slot_of[tempo_i], :] = plane
            done += 1
            if done % 50 == 0 or done == len(tasks):
                log(f"  {done}/{len(tasks)} shards "
                    f"({(time.perf_counter() - started) / 60:.1f} min)")

    elapsed = time.perf_counter() - started
    digest = hashlib.sha256()
    nbytes = 0
    arrays = {}
    for age in ages:
        arr = np.ascontiguousarray(planes[age])
        arrays[f"age{age}"] = arr
        digest.update(arr.tobytes())
        nbytes += arr.nbytes
    # Compressed: control maps are extremely redundant (8.7 MB -> 0.34 MB), so
    # the artifact is small enough to ship rather than regenerate. That matters
    # now that the ENCODER reads it -- a missing table would otherwise be a
    # missing feature, and a zero-filled control channel is a confident lie.
    np.savez_compressed(out_dir / "table.npz", **arrays)

    manifest = {
        "harness": "control_table",
        "schema_version": SCHEMA_VERSION,
        "rule_identity": rule_identity(),
        "content_digest": digest.hexdigest(),
        "ages": list(ages),
        "complete": list(ages) == list(AGES) and len(tempo_indices) == len(TEMPO_STATES),
        "masks": {str(age): masks_by_age[age] for age in ages},
        "tempo_states": [list(TEMPO_STATES[i]) for i in tempo_indices],
        "tempo_definition": (
            "each player 0..4 unbuilt Wonders split ordinary/extra; "
            "builds_left == total_unbuilt - 1, capped at 7; "
            "closed under _as_theology"
        ),
        "cell_encoding": {
            "turns": f"0..{MAX_TURNS}", "UNREACH": UNREACH, "ABSENT": ABSENT,
        },
        "axes": "[mask, who_moves(0=attacker first), tempo, slot]",
        "keys": total_keys,
        "bytes": nbytes,
        "build_seconds": round(elapsed, 1),
        "build_jobs": jobs,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    log(f"\nwrote {nbytes / 1e6:.1f} MB, {total_keys:,} keys, "
        f"{elapsed / 60:.1f} min wall ({elapsed * jobs / 3600:.1f} core-hours)")
    return manifest


# --------------------------------------------------------------------------
# The key, derived from what the ENCODER sees
# --------------------------------------------------------------------------


def observation_present_mask(obs, layout: Layout) -> int:
    """Present-mask from the observation, not from the hidden game state.

    Derived from the observation deliberately: the encoder sees the observation,
    so a key built from anything else could diverge from the features it labels.
    Identities are never read, only presence, which is why this is safe on a
    determinized state.
    """

    mask = 0
    for card in obs.tableau:
        if card.present and card.slot_id in layout.index:
            mask |= 1 << layout.index[card.slot_id]
    return mask


def control_key_from_observation(obs):
    """`control_key` for the observation's own viewer, from the observation alone.

    The encoder holds an observation, not a GameState, and so does the Rust
    encoder. Deriving the key from exactly what the encoder sees is what makes
    the feature and its label describe the same position -- and it is why the
    observation carries `phase`, `active_player` and `pending_choice` at all.
    """

    # Compared by VALUE, not identity. This package is importable under two
    # names (`games.seven_wonders_duel` and `seven_wonders_duel`), and in a
    # process holding both, `obs.phase is Phase.PLAY_AGE` is False for states
    # built through the other tree. The failure is silent and total -- every
    # position masks, so control simply vanishes -- which is exactly the class
    # of error the applicability flag exists to make visible.
    if obs.phase.value != "play_age" or obs.pending_choice is not None:
        return None
    if not obs.tableau:
        return None
    layout = Layout.for_age(max(obs.age, 1))
    mask = observation_present_mask(obs, layout)
    if not mask:
        return None
    # Oriented to the ACTOR, never to the viewer. `encode` is viewer-independent
    # -- the same public position observed from either seat must produce the same
    # tokens -- and keying control off `obs.viewer` broke that, while Rust keyed
    # off `active_player`. The two agreed only because the replay path always
    # encodes for the actor, so the divergence was latent rather than absent.
    # In a clean PLAY_AGE state with no pending choice the actor IS the next
    # tableau mover, which is why `who_moves` is unconditionally true here.
    tempo = tempo_state(obs, obs.active_player)
    if tempo[0] <= 0:
        return None
    # A tempo outside the enumeration is not a table gap, it is a position no
    # legal game reaches -- eight Wonders are drafted, so nine unbuilt cannot
    # happen. Constructed test states and hand-built positions can still produce
    # one, and for those the honest answer is "no control answer", the same as
    # any other inapplicable position. A genuinely missing table is a different
    # failure and still raises.
    if tempo not in TEMPO_INDEX:
        return None
    return (max(obs.age, 1), mask, True, tempo)


#: Process-wide cached table. The encoder reads it on every position, so the
#: 8.7 MB decompression must happen once, not per call.
_DEFAULT_TABLE = None


def default_table() -> "ControlTable":
    global _DEFAULT_TABLE
    if _DEFAULT_TABLE is None:
        table = ControlTable.load()
        # Validated HERE, once per process, because this is the only path the
        # encoder uses. A table generated by an older solver still parses, still
        # looks complete, and silently answers with cells the current code would
        # not produce -- and every training row would carry them.
        table.check_contract()
        _DEFAULT_TABLE = table
    return _DEFAULT_TABLE


def table_content_digest() -> str:
    """Identity of the table's CONTENTS, for a checkpoint to record.

    `ENCODER_SIGNATURE` pins the feature-name schema, so adding a channel makes
    old checkpoints declare themselves incompatible. It does not pin the data
    behind those names: two tables with the same schema and different cells hash
    the same, so a checkpoint trained against one can be served against the other
    with no complaint while `control_now_turns_s` quietly means something else.
    Folding this into the signature itself would force the encoder module to load
    an 8.7 MB artifact merely to know its own name, so the checkpoint records it
    instead.
    """

    return default_table().manifest["content_digest"]


def control_key(game, seat: int, obs=None):
    """`(age, mask, who_moves_is_attacker, tempo)` for `seat`, or None.

    None means **this position has no control label**, and the caller must mask
    it rather than substitute a default. The applicability rule is the one in
    `W3_ENCODER_INTEGRATION_REVIEW_REQUEST.md`:

    * Outside `PLAY_AGE`, or with a pending choice, the current decision-maker
      is not the next tableau mover -- `state_actor` returns
      `pending_choice.player` and `_finish_turn` defers `pending_extra_turn`, so
      "who moves" would be wrong on 16.7% of rows. Resolving a progress token
      can also restate the tempo half of the key.
    * `WONDER_DRAFT` additionally emits **no tableau tokens at all** (measured:
      0 tokens against 20 present cards), so there is nothing to label.
    * A closed Wonder pool has one all-zero tempo state that the table does not
      carry.

    In a clean `PLAY_AGE` state the next tableau mover is `active_player`, and
    tableau tokens are emitted in `sorted(present)` order, which is exactly
    `Layout.slots` order -- verified over 1,235 states.
    """

    return control_key_from_observation(
        game.observation(seat) if obs is None else obs
    )


# --------------------------------------------------------------------------
# Wire format
#
# The Rust self-play path builds its own encoding and holds its own game state,
# so it -- not Python -- is the only place that can derive a key for those rows.
# Rather than ship the 8.7 MB table into Rust just to LABEL data, Rust emits the
# key and Python does the lookup at collate time. That keeps the table, and the
# whole parity surface it implies, on one side of the boundary until control is
# wanted as an INPUT rather than as a target.
#
# One u64 per example row. Zero means "no key" -- the same masked, never
# defaulted, contract as a Python row outside a clean PLAY_AGE state.
# --------------------------------------------------------------------------

_PACK_VALID = 1 << 63


def pack_key(key) -> int:
    """Pack `(age, mask, who_moves, tempo)` into one u64. 0 means no key."""

    if key is None:
        return 0
    age, mask, who_moves, tempo = key
    builds, ord_a, ext_a, ord_d, ext_d = tempo
    if not 1 <= age <= 3:
        raise ValueError(f"age {age} does not fit the wire format")
    if mask >> 20:
        raise ValueError(f"mask 0x{mask:x} exceeds 20 slots")
    for name, value, width in (
        ("builds_left", builds, 7), ("ord_a", ord_a, 7), ("ext_a", ext_a, 7),
        ("ord_d", ord_d, 7), ("ext_d", ext_d, 7),
    ):
        if not 0 <= value <= width:
            raise ValueError(f"{name}={value} does not fit three bits")
    return (
        _PACK_VALID
        | (age & 0x3)
        | (mask & 0xFFFFF) << 2
        | (1 if who_moves else 0) << 22
        | (builds & 0x7) << 23
        | (ord_a & 0x7) << 26
        | (ext_a & 0x7) << 29
        | (ord_d & 0x7) << 32
        | (ext_d & 0x7) << 35
    )


def unpack_key(word: int):
    """Inverse of `pack_key`. Returns None for 0."""

    if not word:
        return None
    if not word & _PACK_VALID:
        raise ValueError(f"control key word 0x{word:x} has no valid bit")
    return (
        word & 0x3,
        (word >> 2) & 0xFFFFF,
        bool((word >> 22) & 0x1),
        (
            (word >> 23) & 0x7, (word >> 26) & 0x7, (word >> 29) & 0x7,
            (word >> 32) & 0x7, (word >> 35) & 0x7,
        ),
    )


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


class MissingControlEntry(LookupError):
    """A key the shipped table does not cover.

    Raised rather than asserted, and never silently defaulted: a zero-filled
    control feature is indistinguishable from 'the opponent gets everything',
    which is a confident lie. Callers in self-play and advisor inference must
    fail over to a known-good model/encoder pair, not to zeros -- and must never
    fall back to solving live, which costs seconds.
    """


class ControlTable:
    """Read-side of the precomputed table."""

    def __init__(self, planes, mask_index, manifest):
        self._planes = planes
        self._mask_index = mask_index
        self.manifest = manifest
        # Driven by the manifest rather than by TEMPO_STATES: the table's own
        # contract decides what it covers, so a subset table reads correctly and
        # a stale global cannot silently reindex a shipped artifact.
        self._tempo_index = {
            tuple(state): i for i, state in enumerate(manifest["tempo_states"])
        }

    @classmethod
    def load(cls, out_dir: Path | None = None) -> "ControlTable":
        out_dir = Path(out_dir) if out_dir else DEFAULT_DIR
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise MissingControlEntry(
                f"table schema {manifest['schema_version']} != "
                f"expected {SCHEMA_VERSION} ({out_dir})"
            )
        with np.load(out_dir / "table.npz") as bundle:
            planes = {int(a): bundle[f"age{a}"] for a in manifest["ages"]}
        mask_index = {
            int(age): {mask: i for i, mask in enumerate(masks)}
            for age, masks in manifest["masks"].items()
        }
        return cls(planes, mask_index, manifest)

    def check_contract(self, *, rule_identity_expected: str | None = None) -> None:
        """Startup validation. Cheap, and the only place a mismatch is loud."""

        want = rule_identity_expected or rule_identity()
        if self.manifest["rule_identity"] != want:
            raise MissingControlEntry(
                "control table was built from different solver/rule code "
                f"({self.manifest['rule_identity'][:12]} != {want[:12]}); "
                "regenerate the table or restore the matching checkpoint"
            )

    def lookup(self, age: int, mask: int, who_moves_is_attacker: bool,
               tempo: tuple) -> np.ndarray:
        """The 20 per-slot cells for one key. Raises on a miss; never defaults."""

        try:
            mask_i = self._mask_index[age][mask]
        except KeyError:
            raise MissingControlEntry(
                f"unreachable present_mask 0x{mask:05x} for age {age} "
                f"(digest {self.manifest['content_digest'][:12]})"
            ) from None
        try:
            tempo_i = self._tempo_index[tuple(tempo)]
        except KeyError:
            raise MissingControlEntry(
                f"tempo state {tuple(tempo)} is outside the generated set "
                f"(digest {self.manifest['content_digest'][:12]})"
            ) from None
        return self._planes[age][mask_i, 0 if who_moves_is_attacker else 1, tempo_i]


# --------------------------------------------------------------------------


def table_blob(table: "ControlTable") -> bytes:
    """Serialize the table for the Rust encoder.

    Rust cannot read a compressed `.npz`, and shipping a second on-disk artifact
    would invite the two drifting apart. Instead Python hands Rust the bytes it
    is itself using, so "the readers agree" is true by construction rather than
    by a test. The digest travels with them, and Rust checks it against the
    manifest it is told to expect.

    Layout (little-endian):

        magic  b"SWDCTL1"      7 bytes
        digest                32 bytes (sha256 of the content, as in the manifest)
        n_tempo u32, then n_tempo * 5 u8   -- the tempo enumeration, in order
        n_ages  u32
        per age: age u8, width u8, n_masks u32,
                 n_masks * u32 masks (ascending),
                 plane bytes [mask][who][tempo][slot]
    """

    import struct

    out = bytearray()
    out += b"SWDCTL1"
    out += bytes.fromhex(table.manifest["content_digest"])
    tempos = [tuple(t) for t in table.manifest["tempo_states"]]
    out += struct.pack("<I", len(tempos))
    for tempo in tempos:
        out += bytes(tempo)
    ages = [int(a) for a in table.manifest["ages"]]
    out += struct.pack("<I", len(ages))
    for age in ages:
        masks = [int(m) for m in table.manifest["masks"][str(age)]]
        plane = table._planes[age]
        out += struct.pack("<BBI", age, plane.shape[3], len(masks))
        for mask in masks:
            out += struct.pack("<I", mask)
        out += plane.tobytes(order="C")
    return bytes(out)


_RUST_INSTALLED = False


def ensure_rust_table() -> bool:
    """Install the table into Rust once per process. Cheap after the first call.

    Every entry point that reaches the Rust encoder calls this: the replay
    derive, the flat batch adapters, and the advisor's searcher. Rust panics
    rather than emitting zeros if it is missing, so the cost of forgetting is a
    crash at the first encode -- loud, but only after the run has started.
    Calling it here makes that unreachable in normal use.

    Guarded because `table_blob` materializes 8.7 MB; doing that per batch would
    dwarf the 1% the feature itself costs.
    """

    global _RUST_INSTALLED
    if _RUST_INSTALLED:
        return True
    _RUST_INSTALLED = install_rust_table()
    return _RUST_INSTALLED


def install_rust_table(module=None, table: "ControlTable | None" = None) -> bool:
    """Hand the Rust encoder the table. Returns False when Rust is unavailable.

    Called before any Rust encode. The Rust side refuses to encode control
    channels without it rather than emitting zeros: a zero-filled channel reads
    as "the opponent reaches everything first", so a missing table must be a
    crash, not a quiet downgrade.
    """

    if module is None:
        try:
            import seven_wonders_rust as module
        except ImportError:
            return False
    if not hasattr(module, "set_control_table"):
        return False
    table = table if table is not None else default_table()
    module.set_control_table(table_blob(table))
    # Confirm what Rust actually holds. `set_control_table` rejects a
    # CONFLICTING table, but this also catches the case where some other code
    # installed first and ours was accepted as a no-op.
    check = getattr(module, "control_table_digest", None)
    if check is not None:
        installed = check()
        want = table.manifest["content_digest"]
        if installed != want:
            raise MissingControlEntry(
                f"Rust holds control table {installed} but Python is using {want}"
            )
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out-dir", default=str(DEFAULT_DIR))
    b.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p = sub.add_parser("plan", help="key-space and size, without building")
    p.add_argument("--out-dir", default=str(DEFAULT_DIR))
    args = parser.parse_args(argv)

    if args.cmd == "plan":
        total = 0
        for age in AGES:
            masks = reachable_masks(age)
            width = len(Layout.for_age(age).slots)
            keys = len(masks) * 2 * len(TEMPO_STATES)
            total += keys
            print(f"age {age}: {len(masks):4d} masks x 2 x {len(TEMPO_STATES)} tempo "
                  f"= {keys:8,d} keys, {keys * width / 1e6:5.1f} MB")
        print(f"total: {total:,} keys")
        return 0

    build(Path(args.out_dir), jobs=args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
