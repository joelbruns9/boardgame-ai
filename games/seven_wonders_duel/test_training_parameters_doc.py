"""`training_parameters.md` is the page people configure runs from.

Nothing checked it, so it silently fell six workstreams behind: 33 flags
undocumented, and a recommended command that had been corrupted into something
that would not parse. These tests are cheap and catch both.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest


DOC = Path(__file__).with_name("training_parameters.md")

# Referenced deliberately as historical or hypothetical, not as usable flags.
PROSE_ONLY = {"--train-epochs", "--anchor-sims", "--revert-win-rate", "--no-"}


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def parser_options() -> set[str]:
    from .phase_d import build_parser

    return {
        option
        for action in build_parser()._actions
        for option in action.option_strings
        if option.startswith("--")
    } - {"--help"}


def _mentioned(doc: str) -> set[str]:
    return set(re.findall(r"(?<![\w-])--[a-z0-9][a-z0-9-]+", doc))


def test_every_flag_is_documented(doc, parser_options):
    missing = sorted(parser_options - _mentioned(doc))
    assert not missing, (
        "flags exist but training_parameters.md does not mention them: "
        f"{missing}"
    )


def test_no_flag_is_documented_that_does_not_exist(doc, parser_options):
    phantom = sorted(_mentioned(doc) - parser_options - PROSE_ONLY)
    assert not phantom, (
        "training_parameters.md documents flags the parser does not accept: "
        f"{phantom}. If one is a deliberate historical reference, add it to "
        "PROSE_ONLY."
    )


def test_every_documented_command_actually_parses(doc, parser_options):
    """The failure that matters: a command someone pastes and runs.

    The previous version of this page had a recommended command truncated
    mid-path, which no amount of prose review would have caught.
    """

    from .phase_d import build_parser

    blocks = [
        block
        for block in re.findall(r"```powershell\r?\n(.*?)```", doc, re.S)
        if "phase_d" in block
    ]
    assert blocks, "no phase_d command blocks found; has the format changed?"
    for block in blocks:
        body = re.sub(r"`\s*\n", " ", block.split("phase_d", 1)[1])
        tokens = [token for token in body.split() if token != "`"]
        assert not any("\\" in token for token in tokens), (
            "use forward slashes in documented paths; PowerShell accepts them "
            "and backslashes get mangled when the command is round-tripped"
        )
        build_parser().parse_args(tokens)  # raises SystemExit on a bad flag


def test_the_gate_section_describes_the_shipped_rule(doc):
    """Guards against the page drifting back to the SPRT-era advice."""

    assert "--promotion-min-lcb" in doc and "--revert-max-ucb" in doc
    assert "no mid-match stopping" in doc.lower()
    # The old page argued gates were unaffordable at ~90 minutes each.
    assert "90 minutes" not in doc or "not 90" in doc
