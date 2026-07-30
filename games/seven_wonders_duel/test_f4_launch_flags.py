"""W6.3: the sweep's answer must reach the launch without being re-typed."""

from __future__ import annotations


import json

import pytest

from .f4_launch_flags import (
    NOT_TRANSLATED,
    PHASE_D_FLAGS,
    launch_flags,
    production_manifest,
    unknown_fields,
)


TUNED = {
    "slots": 48,
    "global_batch_cap": 256,
    "max_inflight_batches": 2,
    "scheduler_workers": 4,
}


def test_every_tuned_field_becomes_its_prefixed_phase_d_flag():
    assert launch_flags(TUNED) == [
        "--rust-slots",
        "48",
        "--rust-global-batch-cap",
        "256",
        "--rust-max-inflight-batches",
        "2",
        "--rust-scheduler-workers",
        "4",
    ]


def test_the_flags_are_accepted_by_the_phase_d_parser():
    """The check that actually matters: these must parse, and land where meant.

    A translation producing plausible-looking flags Phase D rejects fails at
    launch; one producing flags Phase D silently *ignores* is worse, because the
    run proceeds on defaults and the sweep is quietly discarded.
    """

    from .phase_d import build_parser

    args = build_parser().parse_args(
        ["--run-dir", "unused", *launch_flags(TUNED)]
    )
    assert args.rust_slots == TUNED["slots"]
    assert args.rust_global_batch_cap == TUNED["global_batch_cap"]
    assert args.rust_max_inflight_batches == TUNED["max_inflight_batches"]
    assert args.rust_scheduler_workers == TUNED["scheduler_workers"]


def test_every_translated_flag_exists_on_the_parser():
    """Catches a Phase D rename that would otherwise be found on the cloud box."""

    from .phase_d import build_parser

    known = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }
    missing = sorted(set(PHASE_D_FLAGS.values()) - known)
    assert not missing, f"Phase D has no such flags: {missing}"


def test_a_missing_tuned_field_fails_instead_of_falling_back_to_a_default():
    incomplete = dict(TUNED)
    del incomplete["scheduler_workers"]
    with pytest.raises(KeyError, match="scheduler_workers"):
        launch_flags(incomplete)


def test_nonsense_values_are_refused():
    with pytest.raises(ValueError, match="positive"):
        launch_flags({**TUNED, "slots": 0})
    with pytest.raises(TypeError, match="int"):
        launch_flags({**TUNED, "slots": "48"})
    with pytest.raises(TypeError, match="int"):
        launch_flags({**TUNED, "slots": True})


def test_bench_only_fields_are_ignored_on_purpose_not_by_accident():
    manifest = {**TUNED, "pinned_memory": True, "torch_compile": False}
    assert unknown_fields(manifest) == []
    assert "pinned_memory" in NOT_TRANSLATED
    flags = launch_flags(manifest)
    assert "--pinned-memory" not in flags


def test_an_unrecognised_measured_field_is_reported():
    assert unknown_fields({**TUNED, "some_new_knob": 3}) == ["some_new_knob"]


def test_translation_and_ignore_lists_do_not_overlap():
    assert not set(PHASE_D_FLAGS) & set(NOT_TRANSLATED)


def test_the_finalize_output_shape_is_what_gets_read(tmp_path):
    path = tmp_path / "production.json"
    path.write_text(json.dumps({"production_manifest": TUNED}), encoding="utf-8")
    assert production_manifest(path) == TUNED

    path.write_text(json.dumps({"winner": {"manifest": TUNED}}), encoding="utf-8")
    assert production_manifest(path) == TUNED

    path.write_text(json.dumps({"nothing": 1}), encoding="utf-8")
    with pytest.raises(KeyError, match="no production_manifest"):
        production_manifest(path)
