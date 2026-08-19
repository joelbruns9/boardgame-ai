"""Inventory of the places that rebuild a saved model.

Adding `pooled_readout` / `reply_head` to the model broke six separate rebuild
sites, each of which had enumerated by hand the config keys it happened to know
about. The failure never raised at the call site: it surfaced as a strict load
error hours into a run, or -- worse, for head count -- as a model that loaded
cleanly and computed something else.

`train.model_from_config` is the fix, but converting every caller at once was
more churn than the task at hand justified. This test makes the remainder
*enumerated* rather than silent: a module that both reads a checkpoint's head
count and builds a model must either be converted or appear below with a reason.

Shrink `NOT_YET_CONVERTED`. Do not grow it without one.
"""

from __future__ import annotations

import re
from pathlib import Path

PACKAGE = Path(__file__).parent

#: Reading a checkpoint's head count AND building a model is the signature of a
#: hand rebuild. Deliberately a coarse heuristic with an explicit allow-list
#: rather than a cleverer pattern: two earlier attempts at a precise regex were
#: wrong in both directions -- one flagged a file that only validates a head
#: count, the next missed every module that assigns it to a local first.
_READS_HEADS = re.compile(r"(?<!def )heads_from_config\s*\(")
_BUILDS = re.compile(r"build_model\s*\(")

#: Converted, or legitimately not a rebuild.
ALLOWED = {
    # Defines the helpers.
    "train.py",
    # Rebuilds via model_from_config. It still reads the stored head count, to
    # CHECK it against this run's config and refuse a mismatched resume -- a
    # different question from rebuilding the checkpoint faithfully.
    "phase_d.py",
}

#: Modules that still rebuild by hand. All are offline analysis tools -- probes,
#: benches and sweeps -- none on the training or gate path, so a stale rebuild
#: there fails loudly at the tool rather than mid-run.
NOT_YET_CONVERTED = {
    "ablate_value_head.py",
    "phase_e.py",
    "search_gain_probe.py",
    "value_ceiling_probe.py",
    "w0_sizing.py",
    "w0_sizing_v2.py",
    "weight_decay_probe.py",
}

#: A separate, UNAUDITED class this test does not cover: modules that read a
#: checkpoint's head count and pass it onward into a config dict or a subprocess
#: argument rather than building a model themselves -- f4_phase_d_ab,
#: f4_phase_d_sweep, precision_arena, w5_gate_bench, w5_gate_slots_sweep. They
#: can carry the same staleness one step further away, and a regex cannot follow
#: it. Named here so the gap is known rather than implied by silence.



def _rebuild_sites() -> set[str]:
    found = set()
    for path in PACKAGE.glob("*.py"):
        if path.name.startswith("test_"):
            continue
        text = path.read_text(encoding="utf-8")
        if _READS_HEADS.search(text) and _BUILDS.search(text):
            found.add(path.name)
    return found


def test_no_unlisted_module_rebuilds_a_model_by_hand():
    unlisted = _rebuild_sites() - NOT_YET_CONVERTED - ALLOWED
    assert not unlisted, (
        f"{sorted(unlisted)} rebuild a saved model without train.model_from_config. "
        "Use the helper, or add the module to NOT_YET_CONVERTED with a reason."
    )


def test_the_debt_list_does_not_rot():
    """A converted module must leave the list, or the list lies about the debt."""

    stale = NOT_YET_CONVERTED - _rebuild_sites()
    assert not stale, (
        f"{sorted(stale)} no longer match the hand-rebuild shape; remove them "
        "from NOT_YET_CONVERTED so the remaining debt stays accurate."
    )


def test_the_training_and_gate_paths_carry_no_debt():
    """The paths that matter are converted.

    phase_d writes and reloads every checkpoint a run produces, and
    build_equiv_corpus is the engine-parity gate; a stale rebuild in either is
    the expensive kind, discovered hours into a rented box rather than at a tool.
    """

    for name in ("phase_d.py", "build_equiv_corpus.py"):
        assert name not in NOT_YET_CONVERTED, f"{name} must use model_from_config"
