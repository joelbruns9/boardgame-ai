"""Buried defaults, made visible.

The most expensive recurring defect in this project is a value "configured" in
one place and silently decided by argparse in another: --train-steps at 8x the
sample reuse the loop is tuned for, --weight-decay 5,000x off, --leaf-batch
decided by an A/B and then never passed. A default is not the problem; an
INVISIBLE default is.
"""

from __future__ import annotations

import pytest

from .phase_d import CRITICAL_FLAGS, build_parser, report_unspecified_flags


def _argv(**flags) -> list[str]:
    argv = []
    for name, value in flags.items():
        argv += [f"--{name.replace('_', '-')}", str(value)]
    return argv


def _complete() -> list[str]:
    return [token for flag in CRITICAL_FLAGS for token in (flag, "1")]


def test_a_missing_critical_flag_refuses_the_run():
    """Not a warning. A wrong value in this set fails quietly for hours."""

    with pytest.raises(SystemExit) as excinfo:
        report_unspecified_flags(build_parser(), ["--run-dir", "x"],
                                 critical=CRITICAL_FLAGS)
    message = str(excinfo.value)
    for flag in CRITICAL_FLAGS:
        assert flag in message, f"{flag} is required but not named in the error"
    assert "nobody made" in message


def test_stating_every_critical_flag_allows_the_run():
    unspecified = report_unspecified_flags(
        build_parser(), _complete(), critical=CRITICAL_FLAGS
    )
    for flag in CRITICAL_FLAGS:
        assert flag not in unspecified


def test_the_report_names_what_was_not_passed():
    unspecified = report_unspecified_flags(
        build_parser(), _complete(), critical=CRITICAL_FLAGS
    )
    assert "--learning-rate" in unspecified, "an unpassed setting must be listed"
    assert "--train-steps" not in unspecified


def test_a_boolean_counts_as_stated_in_either_polarity():
    """BooleanOptionalAction registers --x and --no-x. Passing --no-reply-head
    is a decision, and reporting it as defaulted would be wrong."""

    argv = _complete() + ["--no-reply-head"]
    unspecified = report_unspecified_flags(build_parser(), argv,
                                           critical=CRITICAL_FLAGS)
    assert "--reply-head" not in unspecified


def test_equals_form_counts_as_stated():
    argv = [f"{flag}=1" for flag in CRITICAL_FLAGS]
    unspecified = report_unspecified_flags(build_parser(), argv,
                                           critical=CRITICAL_FLAGS)
    for flag in CRITICAL_FLAGS:
        assert flag not in unspecified


def test_each_setting_is_counted_once_not_once_per_option_string():
    """`--help` is not a setting, and --x/--no-x is one decision. Counting
    option strings would report roughly twice as many defaults as there are
    choices, which makes the list easy to dismiss."""

    unspecified = report_unspecified_flags(build_parser(), _complete(),
                                           critical=CRITICAL_FLAGS)
    assert "--help" not in unspecified
    assert not [flag for flag in unspecified if flag.startswith("--no-")]


def test_the_critical_list_stays_short_and_real():
    """It is a list of flags that fail QUIETLY when wrong, not a list of
    important flags. Long lists get bypassed."""

    options = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }
    assert len(CRITICAL_FLAGS) <= 8, "a long required list becomes a nuisance"
    for flag in CRITICAL_FLAGS:
        assert flag in options, f"{flag} is required but is not a real flag"
