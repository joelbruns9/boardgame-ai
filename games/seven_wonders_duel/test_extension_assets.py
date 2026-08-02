"""The extension ships a copy of ``bga_snippet.js``; keep the two in step.

A browser extension has to be self-contained, so ``extension_7wd/`` carries its
own copy of the capture code. That copy is what actually runs against a live
table, while ``games/seven_wonders_duel/bga_snippet.js`` is the one referenced by
the plan, the tests and the console workflow. Silent drift between them is the
failure this guards: the extension would keep reading the board the old way
while every doc and test described the new way.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_CANONICAL = Path(__file__).parent / "bga_snippet.js"
_EXTENSION = _REPO / "extension_7wd"


def test_extension_snippet_matches_canonical():
    shipped = _EXTENSION / "bga_snippet.js"
    assert shipped.exists(), "extension is missing its bga_snippet.js copy"
    assert shipped.read_text(encoding="utf-8") == _CANONICAL.read_text(
        encoding="utf-8"
    ), (
        "extension_7wd/bga_snippet.js has drifted from "
        "games/seven_wonders_duel/bga_snippet.js -- re-copy it"
    )


def test_manifest_is_cross_browser_mv3():
    manifest = json.loads((_EXTENSION / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 3
    # No background key at all: Chrome MV3 wants a service_worker and Firefox
    # wants background.scripts, so the portable answer is to need neither.
    assert "background" not in manifest
    # Firefox needs an explicit add-on id to load unsigned.
    assert manifest["browser_specific_settings"]["gecko"]["id"]
    hosts = manifest["host_permissions"]
    assert any("boardgamearena" in h for h in hosts)
    assert any("127.0.0.1:8000" in h for h in hosts)
    # The page-world files must be reachable via runtime.getURL().
    war = manifest["web_accessible_resources"][0]["resources"]
    assert {"bga_snippet.js", "page_bridge.js"} <= set(war)


def _code_only(source: str) -> str:
    """Strip // and /* */ comments, so prose *about* a banned API does not read
    as a use of it."""
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return "\n".join(re.sub(r"//.*$", "", line) for line in source.splitlines())


@pytest.mark.parametrize("name", ["content.js", "page_bridge.js", "bga_snippet.js"])
def test_extension_scripts_declare_no_browser_specific_globals(name):
    """Firefox-only or Chrome-only globals would break the other browser."""
    code = _code_only((_EXTENSION / name).read_text(encoding="utf-8"))
    assert "wrappedJSObject" not in code, "Firefox-only; breaks Chrome"
    if name != "content.js":
        # Only the isolated-world script may touch extension APIs at all.
        assert not re.search(r"\bchrome\.runtime\b|\bbrowser\.runtime\b", code)


def test_page_bridge_states_match_the_mapper():
    """The bridge only wakes on states bga_extract actually supports, so a
    routine BGA transition never reaches Python just to be rejected."""
    from .bga_extract import _MAIN_TURN_STATE, _PENDING_STATES

    source = (_EXTENSION / "page_bridge.js").read_text(encoding="utf-8")
    block = source.split("const ADVISABLE = new Set([", 1)[1].split("]);", 1)[0]
    declared = set(re.findall(r'"([^"]+)"', block))
    assert declared == {_MAIN_TURN_STATE} | set(_PENDING_STATES)
