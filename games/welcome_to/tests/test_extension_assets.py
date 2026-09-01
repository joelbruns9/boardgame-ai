"""The extension ships a copy of ``bga_snippet.js``; keep the two in step.

A browser extension has to be self-contained, so ``extension_welcome_to/``
carries its own copy of the capture code.  That copy is what actually runs
against a live table, while ``games/welcome_to/bga_snippet.js`` is the one the
tests and the console workflow reference.  Silent drift between them is the
failure this guards: the extension would keep reading the board the old way
while every doc and test described the new way.

The rest of this file pins the four properties that are invisible until they
fail on a live table -- the network boundary, the state list, the browser
globals, and the panel actually consuming what the host sends.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_CANONICAL = Path(__file__).resolve().parents[1] / "bga_snippet.js"
_EXTENSION = _REPO / "extension_welcome_to"


def test_extension_snippet_matches_canonical():
    shipped = _EXTENSION / "bga_snippet.js"
    assert shipped.exists(), "extension is missing its bga_snippet.js copy"
    assert shipped.read_text(encoding="utf-8") == _CANONICAL.read_text(
        encoding="utf-8"
    ), (
        "extension_welcome_to/bga_snippet.js has drifted from "
        "games/welcome_to/bga_snippet.js -- re-copy it"
    )


def test_manifest_is_cross_browser_mv3():
    manifest = json.loads((_EXTENSION / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 3
    # Both background forms are declared on purpose: Chrome MV3 reads
    # service_worker and ignores scripts, Firefox MV3 the reverse. A background
    # context is REQUIRED, not optional: the page is https and the advisor is
    # http://127.0.0.1, so a content-script fetch is blocked as mixed content.
    background = manifest["background"]
    assert background["service_worker"] == "background.js"
    assert background["scripts"] == ["background.js"]
    # Firefox needs an explicit add-on id to load unsigned.
    assert manifest["browser_specific_settings"]["gecko"]["id"]
    hosts = manifest["host_permissions"]
    assert any("boardgamearena" in h for h in hosts)
    # Port 8001, not 8000: the 7WD advisor owns 8000 and both hosts get left
    # running. A shared port would serve one game's extension from the other's
    # model, which fails as a confusing 400 rather than as a wrong port.
    assert any("127.0.0.1:8001" in h for h in hosts)
    assert not any(":8000" in h for h in hosts)
    # The page-world files must be reachable via runtime.getURL().
    war = manifest["web_accessible_resources"][0]["resources"]
    assert {"bga_snippet.js", "page_bridge.js"} <= set(war)


def _code_only(source: str) -> str:
    """Strip // and /* */ comments, so prose *about* a banned API does not read
    as a use of it."""
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return "\n".join(re.sub(r"//.*$", "", line) for line in source.splitlines())


@pytest.mark.parametrize(
    "name", ["content.js", "page_bridge.js", "bga_snippet.js", "background.js"]
)
def test_extension_scripts_declare_no_browser_specific_globals(name):
    """Firefox-only or Chrome-only globals would break the other browser."""
    code = _code_only((_EXTENSION / name).read_text(encoding="utf-8"))
    assert "wrappedJSObject" not in code, "Firefox-only; breaks Chrome"
    if name not in ("content.js", "background.js"):
        # Page-world scripts must not touch extension APIs; they do not have
        # them. content.js and background.js are extension contexts and may.
        assert not re.search(r"\bchrome\.runtime\b|\bbrowser\.runtime\b", code)


def test_only_the_background_script_makes_network_calls():
    """Mixed content: the page is https, the advisor is http://127.0.0.1.

    A content-script fetch is blocked before it leaves the browser -- this
    failed on a live 7WD table as a bare "host offline" while the host was
    healthy. All network must go through the background context, which is not a
    page and so has no such rule.
    """
    content = _code_only((_EXTENSION / "content.js").read_text(encoding="utf-8"))
    assert "fetch(" not in content, "content.js must proxy via runtime.sendMessage"
    bridge = _code_only((_EXTENSION / "page_bridge.js").read_text(encoding="utf-8"))
    assert "fetch(" not in bridge, "the page world must never reach the advisor"
    snippet = _code_only((_EXTENSION / "bga_snippet.js").read_text(encoding="utf-8"))
    assert "fetch(" not in snippet, "the page world must never reach the advisor"
    background = _code_only((_EXTENSION / "background.js").read_text(encoding="utf-8"))
    assert "fetch(" in background


def test_page_bridge_states_match_the_mapper():
    """The bridge only wakes on states bga_extract supports, so a routine BGA
    transition never reaches Python just to be rejected."""
    from games.welcome_to.bga_extract import PHASE_OF_STATE

    source = (_EXTENSION / "page_bridge.js").read_text(encoding="utf-8")
    block = source.split("const ADVISABLE = new Set([", 1)[1].split("]);", 1)[0]
    declared = set(re.findall(r'"([^"]+)"', block))
    assert declared == set(PHASE_OF_STATE)


def test_panel_shows_the_diagnostics_the_host_sends():
    """The advisor exists to expose the model's weaknesses, so the panel has to
    render the fields that do that -- not just the top move.

    Each of these is something a "best move" overlay would drop, and each was
    added to answer a specific question: what did the net believe before search,
    what does it think the game ends at, and which of its heads are meaningless
    in this checkpoint.
    """
    content = _code_only((_EXTENSION / "content.js").read_text(encoding="utf-8"))
    assert "r.prior" in content, "the raw policy is half the diagnosis"
    assert "pub.forecast" in content
    assert "final_score" in content
    assert "seat.components" in content
    assert "will_complete_plan" in content
    assert "untrained_heads" in content
    assert "pub.warnings" in content
    assert "next_effect" in content, "the known next effect is public information"


#: Every extension in this repo that injects a script into the BGA page world.
_PAGE_WORLD_EXTENSIONS = ("extension_7wd", "extension_welcome_to")

_DECLARATION = re.compile(
    r"^(?:function|const|let|var|class)\s+([A-Za-z_$][\w$]*)", re.M
)


def _page_world_globals(path: Path) -> set:
    """Top-level declarations in a classic script, i.e. what lands on ``window``."""
    return set(_DECLARATION.findall(_code_only(path.read_text(encoding="utf-8"))))


def test_the_welcome_to_page_scripts_add_exactly_one_global():
    """Two advisors, one page. Anything else is a name collision waiting to fire.

    Both extensions match ``*://*.boardgamearena.com/*`` and both inject their
    capture script into the MAIN world, where a top-level ``function foo()``
    becomes ``window.foo``. Whichever content script runs second wins.

    This was not hypothetical: ``extension_7wd/bga_snippet.js`` declares
    ``findGameWindow``, ``findGameWindows``, ``seatOf``, ``_required`` and
    ``captureForAdvisor``, and so did this one -- so with both extensions loaded,
    the 7WD panel silently stopped waking on 7WD tables, because it was calling
    Welcome To's ``findGameWindow``, which looks for ``constructionCards``,
    throws, and is swallowed as "not the board frame".

    So the Welcome To page-world scripts publish exactly one name and reach
    everything else through it.
    """
    exported = _page_world_globals(_EXTENSION / "bga_snippet.js")
    assert exported == set(), (
        "bga_snippet.js must declare nothing at the top level of the page world; "
        "found %s" % sorted(exported)
    )
    bridge = _page_world_globals(_EXTENSION / "page_bridge.js")
    assert bridge == set(), (
        "page_bridge.js must stay inside its IIFE; found %s" % sorted(bridge)
    )
    source = (_EXTENSION / "bga_snippet.js").read_text(encoding="utf-8")
    assert "window.__WTO_ADVISOR__ = {" in source
    assert "__WTO_ADVISOR__" in (_EXTENSION / "page_bridge.js").read_text(
        encoding="utf-8"
    )


def test_no_two_bga_extensions_share_a_page_world_name():
    """The general form of the rule above, for whatever gets added next."""
    seen: dict = {}
    for name in _PAGE_WORLD_EXTENSIONS:
        directory = _REPO / name
        if not directory.exists():
            continue
        for script in ("bga_snippet.js", "page_bridge.js"):
            path = directory / script
            if path.exists():
                seen.setdefault(name, set()).update(_page_world_globals(path))
    names = sorted(seen)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            clash = seen[a] & seen[b]
            assert not clash, (
                "%s and %s both declare %s in the BGA page world; whichever "
                "content script loads second silently replaces the other's"
                % (a, b, sorted(clash))
            )


def _run_node(script: str) -> str:
    """Execute a Node snippet, or skip when Node is unavailable."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; browser-shaped cases not exercised")
    done = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


_HARNESS = r"""
const fs = require("fs");
const src = fs.readFileSync(%(snippet)s, "utf8").replace(/\r/g, "");
const grab = (name) => {
  const start = src.indexOf("function " + name + "(");
  let depth = 0;
  for (let j = src.indexOf("{", start); j < src.length; j++) {
    if (src[j] === "{") depth++;
    else if (src[j] === "}") { depth--; if (!depth) return src.slice(start, j + 1); }
  }
};
const fns = ["findGameWindows", "seatOf", "findGameWindow"].map(grab).join("\n");
// A frame shaped the way BGA builds one. `player_id` and `isSpectator` are the
// FRAMEWORK's own fields (bga-framework.d.ts:890, 897); `gamedatas` carries no
// me_id, because welcometo.game.php's getAllDatas does not send one -- which is
// exactly the bug this harness now covers.
const win = (url, me, players, frames = []) => ({
  frames,
  location: { href: url },
  gameui: me === null ? undefined
    : {
        player_id: me === "spectator" ? undefined : me,
        isSpectator: me === "spectator",
        gamedatas: { players, constructionCards: [[], [], []] },
      },
});
const PLAYERS = { "111": {}, "222": {} };
const run = (top) => {
  const find = new Function("window", fns + "; return findGameWindow;")({ top });
  try { return "seat:" + (find().gameui.player_id || ""); }
  catch (e) { return (e.wtoAmbiguous ? "loud" : "quiet"); }
};
const U = "https://boardgamearena.com/9/welcometo?table=1";
console.log([
  run(win(U, "111", PLAYERS)),
  run(win(U, "111", PLAYERS, [win(U + "&testuser=222", "222", PLAYERS)])),
  run(win(U, "111", PLAYERS, [win(U + "&x=2", "222", PLAYERS)])),
  run(win(U, "999", PLAYERS)),
  run(win("https://example.com", null, {})),
  run(win(U + "&testuser=222", "222", PLAYERS)),
  run(win(U, "spectator", PLAYERS)),
  run(win(U, 111, PLAYERS)),
].join(","));
"""


def test_frame_selection_picks_the_seat_and_is_loud_when_it_cannot():
    """Which seat is this, and which frame is it in.

    Two hazards, and the first one bit on the very first live table. **The seat
    comes from ``gameui.player_id``, the BGA framework's field, not from
    ``gamedatas.me_id``** -- that is a field a game's own ``getAllDatas`` chooses
    to send, 7 Wonders Duel sends one and Welcome To does not, and reading it
    produced "not seated at the table (me_id missing)" on a perfectly ordinary
    table. The harness therefore builds frames with no ``me_id`` at all.

    The second is that a table can expose more than one ``gameui``: a nested
    ``...&testuser=<id>`` frame renders another seat. Here the whole position IS
    one player's sheet, so reading the wrong frame does not degrade the advice --
    it advises somebody else's game while looking perfectly healthy. Convention
    handles the observed case (drop ``testuser=`` frames); the rest raise a
    tagged error rather than silently taking the shallowest. "quiet" is correct
    for a page with no board, because the bridge runs in every frame and most of
    them are not the game.
    """
    script = _HARNESS % {"snippet": json.dumps(str(_CANONICAL))}
    assert _run_node(script).strip().split(",") == [
        "seat:111",  # ordinary table: one board frame
        "seat:111",  # the observed hazard: opponent frame ignored
        "loud",      # two unmarked frames on different seats: unknowable
        "loud",      # seated as somebody who is not at this table
        "quiet",     # not a game page at all
        "loud",      # ONLY impersonated frames: refuse, do not fall back
        "loud",      # spectator: reads every sheet, has no turn to advise
        "seat:111",  # player_id arrives as a NUMBER; players keys are strings
    ]
