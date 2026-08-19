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

#: Building a model is the signature. Keyed on `build_model` ALONE, with an
#: allow-list, rather than on "reads a head count AND builds": the conjunction
#: let `cloud_preflight` through, which builds a model to size a rented box and
#: was silently omitting `pooled_readout` / `reply_head` -- understating every
#: disk figure for the run those switches were added for. A site that builds a
#: model is a site that has to know the architecture, whether or not it happens
#: to mention heads.
_BUILDS = re.compile(r"build_model\s*\(")

#: Converted, or legitimately not a rebuild of saved weights.
ALLOWED = {
    # Defines the helpers.
    "train.py",
    # Rebuilds via model_from_config. It still reads the stored head count, to
    # CHECK it against this run's config and refuse a mismatched resume -- a
    # different question from rebuilding the checkpoint faithfully.
    "phase_d.py",
    # Rebuilds via model_from_config.
    "build_equiv_corpus.py",
    # Builds FRESH models from an explicit spec, never from saved weights, so
    # there is nothing for a config to be stale against. They still have to
    # take the architecture switches, which is what `parameter_count` is
    # asserted on below.
    "cloud_preflight.py",
    "phase_b_gate.py",
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
        if _BUILDS.search(text):
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
        # Not merely absent from the debt list -- actually going through the
        # helper. The list version of this assertion would pass if either file
        # reverted to a hand rebuild that stopped mentioning a head count.
        text = (PACKAGE / name).read_text(encoding="utf-8")
        assert "model_from_config" in text, f"{name} must use model_from_config"


def test_a_fresh_model_builder_still_takes_the_architecture_switches():
    """The allow-list is not a blind spot.

    `cloud_preflight` builds a fresh model rather than reloading one, so it is
    legitimately allow-listed -- but it sizes the box the run will be rented on,
    and a size computed without `pooled_readout` / `reply_head` is wrong in the
    direction that matters. This pins the thing the allow-list stops checking.
    """

    from .cloud_preflight import parameter_count

    plain = parameter_count(64, 2, 4)
    pooled = parameter_count(64, 2, 4, pooled_readout=True)
    replied = parameter_count(64, 2, 4, reply_head=True)
    assert pooled > plain, "pooled readout adds a projection and must be counted"
    assert replied > plain, "the reply head adds parameters and must be counted"
