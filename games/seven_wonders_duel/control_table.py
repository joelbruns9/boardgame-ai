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
    for age in ages:
        arr = np.ascontiguousarray(planes[age])
        np.save(out_dir / f"age{age}.npy", arr)
        digest.update(arr.tobytes())
        nbytes += arr.nbytes

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
        planes = {int(a): np.load(out_dir / f"age{a}.npy") for a in manifest["ages"]}
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
