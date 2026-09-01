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
    # Both background forms are declared on purpose: Chrome MV3 reads
    # service_worker and ignores scripts, Firefox MV3 the reverse. That keeps a
    # single manifest. A background context is REQUIRED (not optional): the page
    # is https and the advisor is http://127.0.0.1, so a content-script fetch is
    # blocked as mixed content -- observed live as a bare "host offline".
    background = manifest["background"]
    assert background["service_worker"] == "background.js"
    assert background["scripts"] == ["background.js"]
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


def test_page_bridge_states_match_the_mapper():
    """The bridge only wakes on states bga_extract actually supports, so a
    routine BGA transition never reaches Python just to be rejected."""
    from .bga_extract import (
        _DRAFT_STATE,
        _MAIN_TURN_STATE,
        _PENDING_STATES,
        _START_PLAYER_STATE,
    )

    source = (_EXTENSION / "page_bridge.js").read_text(encoding="utf-8")
    block = source.split("const ADVISABLE = new Set([", 1)[1].split("]);", 1)[0]
    declared = set(re.findall(r'"([^"]+)"', block))
    assert declared == {
        _DRAFT_STATE,
        _MAIN_TURN_STATE,
        _START_PLAYER_STATE,
    } | set(_PENDING_STATES)


def test_only_the_background_script_makes_network_calls():
    """Mixed content: the page is https, the advisor is http://127.0.0.1. A
    content-script fetch is blocked before it leaves the browser -- this failed
    on a live table as a bare "host offline". All network must go through the
    background context, which is not a page and so has no such rule.
    """
    content = _code_only((_EXTENSION / "content.js").read_text(encoding="utf-8"))
    assert "fetch(" not in content, "content.js must proxy via runtime.sendMessage"
    bridge = _code_only((_EXTENSION / "page_bridge.js").read_text(encoding="utf-8"))
    assert "fetch(" not in bridge, "the page world must never reach the advisor"
    background = _code_only((_EXTENSION / "background.js").read_text(encoding="utf-8"))
    assert "fetch(" in background


def test_panel_renders_exact_endgame_annotations_and_ties():
    """The host answer is useful only if the shipped panel consumes it."""

    content = _code_only((_EXTENSION / "content.js").read_text(encoding="utf-8"))
    assert "snap.summary.exact_endgame" in content
    assert "r.annotations.exact_endgame" in content
    assert "guaranteed " in content
    assert "best (ties included)" in content
    assert "exact solver · solving concurrently…" in content
    assert "exact solver · skipped · estimate " in content
    assert "exact solver · timed out" in content
    assert "exact solver · disabled in host" in content
    assert 'call("/health"' in content


def _run_node(script: str) -> str:
    """Execute a Node snippet, or skip when Node is unavailable.

    The frame-selection logic can only run in a browser-shaped world, which is
    exactly why its one real bug was found on a live table rather than here.
    Node is close enough: `findGameWindow` touches only `frames`,
    `location.href` and `gameui.gamedatas`, all of which a plain object can
    provide.
    """
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; frame-selection cases not exercised")
    done = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


_FRAME_HARNESS = r"""
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
const fns = grab("findGameWindows") + "\n" + grab("findGameWindow") + "\n";
const win = (url, me, players, frames = []) => ({
  frames,
  location: { href: url },
  gameui: me === null ? undefined
    : { gamedatas: { me_id: me, players, wondersSituation: {} } },
});
const PLAYERS = { "111": {}, "222": {} };
const run = (top) => {
  const find = new Function("window", fns + "; return findGameWindow;")({ top });
  try { return "seat:" + find().gameui.gamedatas.me_id; }
  catch (e) { return (e.swdAmbiguous ? "loud" : "quiet"); }
};
const U = "https://boardgamearena.com/9/sevenwondersduel?table=1";
console.log([
  run(win(U, "111", PLAYERS)),
  run(win(U, "111", PLAYERS, [win(U + "&testuser=222", "222", PLAYERS)])),
  run(win(U, "111", PLAYERS, [win(U + "&x=2", "222", PLAYERS)])),
  run(win(U, "999", PLAYERS)),
  run(win("https://example.com", null, {})),
  run(win(U + "&testuser=222", "222", PLAYERS)),
].join(","));
"""


_RECORDER_HARNESS = r"""
const snippet = require(%(snippet)s);
const { installPacketRecorder, drainPackets, captureCurrentStateArgs } = snippet;

// A window shaped like the one BGA builds. Verified against a live replay page
// on 2026-08-15: gameui.notifqueue.onNotification(packet) is where every
// incoming packet arrives, whole, and it also accepts a JSON string.
const makeWindow = (opts = {}) => {
  const seen = [];
  const w = {
    location: { href: "https://boardgamearena.com/9/sevenwondersduel?table=77" },
    frames: [],
    g_gamelogs: opts.preloaded || undefined,
  };
  if (!opts.noGameui) {
    w.gameui = { notifqueue: { onNotification: (p) => { seen.push(p); return "dispatched"; } } };
  }
  w.seen = seen;
  return w;
};
const packet = (over) => Object.assign(
  { channel: "/table/t77", table_id: 77, packet_id: "1", move_id: "1",
    data: [{ type: "constructBuilding", args: {} }] }, over);
const keys = (list) => list.map((p) => p.move_id + "/" + p.packet_id).join(" ");
const out = {};

// 1. seeding, hooking, pass-through
const w = makeWindow({ preloaded: [packet({ move_id: "0", packet_id: "0" })] });
const store = installPacketRecorder(w);
out.hooked = store.hooked;
out.seeded = store.seeded;
out.passThrough = w.gameui.notifqueue.onNotification(packet());
out.dispatchedToBga = w.seen.length;

// 2. what is kept and what is not
w.gameui.notifqueue.onNotification(JSON.stringify(packet({ move_id: "2", packet_id: "2" })));
w.gameui.notifqueue.onNotification(packet({ move_id: "2", packet_id: "2" }));  // dup
w.gameui.notifqueue.onNotification(packet({ move_id: "3", packet_id: "3",
  data: [{ type: "tablechat", args: {} }] }));
w.gameui.notifqueue.onNotification(packet({ move_id: "4", packet_id: "4",
  channel: "/chat/global" }));
w.gameui.notifqueue.onNotification(packet({ move_id: "5", packet_id: "5", table_id: 999 }));
w.gameui.notifqueue.onNotification(packet({ move_id: "6", packet_id: "6",
  channel: "/player/p42" }));
out.kept = keys(drainPackets(w));
out.drainedTwice = drainPackets(w).length;

// 3. installing twice must not double-record or re-wrap
const again = installPacketRecorder(w);
again === store || (out.storeChanged = true);
w.gameui.notifqueue.onNotification(packet({ move_id: "7", packet_id: "7" }));
out.afterReinstall = keys(drainPackets(w));

// 4. installed before BGA built gameui: the hook must still take later
const early = makeWindow({ noGameui: true });
out.earlyHooked = installPacketRecorder(early).hooked;
early.gameui = { notifqueue: { onNotification: () => "dispatched" } };
out.lateHooked = installPacketRecorder(early).hooked;
early.gameui.notifqueue.onNotification(packet({ move_id: "8", packet_id: "8" }));
out.lateKept = keys(drainPackets(early));

// 5. gamedatas can carry stale/empty args even though its state id is current.
// The raw state-change packet is authoritative for per-state fields such as
// chooseOpponentBuilding's Brown/Grey discriminator.
const stateW = makeWindow();
stateW.gameui.gamedatas = { gamestate: { id: 65, args: {} } };
installPacketRecorder(stateW);
stateW.gameui.notifqueue.onNotification(packet({ move_id: "9", packet_id: "9",
  data: [{ type: "gameStateChange", args: { id: 65, args: {
    buildingType: "Brown", buildingTypeTranslatable: "Brown",
    i18n: ["buildingTypeTranslatable"]
  } } }] }));
out.currentStateArgs = captureCurrentStateArgs(stateW, stateW.gameui.gamedatas);

// Never search back to the old id-65 transition if the newest state change no
// longer matches the live state. A dropped packet must degrade to the explicit
// gamedatas fallback, not silently reuse an earlier Wonder's colour.
stateW.gameui.gamedatas.gamestate.args = { source: "fallback" };
stateW.gameui.notifqueue.onNotification(packet({ move_id: "10", packet_id: "10",
  data: [{ type: "gameStateChange", args: { id: 30, args: {} } }] }));
out.noStaleReuse = captureCurrentStateArgs(stateW, stateW.gameui.gamedatas);

console.log(JSON.stringify(out));
"""


def test_packet_recorder_captures_the_oracle_without_disturbing_the_table():
    """The differential harness's whole input comes through this hook.

    Three failures it has to make impossible. Recording must not swallow or
    alter BGA's own dispatch -- a throw here would break the live table. The
    capture must be complete for this game, because the harness refuses to
    replay a game with holes in its move sequence, so a dropped packet costs
    the whole game rather than one move. And installing must stay idempotent
    while remaining able to hook late: the bridge can run before BGA has built
    `gameui`, and an install that counted that attempt as done would leave the
    recorder permanently deaf, silently.
    """

    script = _RECORDER_HARNESS % {"snippet": json.dumps(str(_CANONICAL))}
    out = json.loads(_run_node(script))

    assert out["hooked"] is True
    assert out["seeded"] == 1  # replay/archive pages preload the history
    # BGA's own handler still runs, still returns its value, exactly once.
    assert out["passThrough"] == "dispatched" and out["dispatchedToBga"] == 1
    # Kept: this table's /table packets, a JSON-string packet, our own /player
    # stream. Dropped: duplicates, chat, other channels, other tables.
    assert out["kept"] == "0/0 1/1 2/2 6/6"
    assert out["drainedTwice"] == 0  # a drain hands each packet over once
    assert "storeChanged" not in out
    assert out["afterReinstall"] == "7/7"  # re-install did not double-wrap
    # Hooking late is the case that would otherwise fail silently.
    assert out["earlyHooked"] is False and out["lateHooked"] is True
    assert out["lateKept"] == "8/8"
    assert out["currentStateArgs"]["buildingType"] == "Brown"
    assert out["noStaleReuse"] == {"source": "fallback"}


def test_content_script_posts_packets_to_the_game_log():
    """The capture is only worth anything if it reaches the log the harness reads."""

    from .bga_differential import PACKET_KIND

    content = _code_only((_EXTENSION / "content.js").read_text(encoding="utf-8"))
    assert f'kind: "{PACKET_KIND}"' in content
    bridge = _code_only((_EXTENSION / "page_bridge.js").read_text(encoding="utf-8"))
    # Packets arrive on the opponent's turns and through end-game scoring too,
    # so the pump must not sit behind the "is it my turn" gate.
    assert "installPacketRecorder" in bridge and "drainPackets" in bridge


def test_frame_selection_picks_the_seat_and_is_loud_when_it_cannot():
    """The wrong-seat hazard, which cost a Great Library position on a live table.

    A BGA table can expose more than one `gameui`: a nested `...&testuser=<id>`
    frame renders the OPPONENT's seat. Public data is identical in both, so the
    reconstructed position looks fine -- but per-seat private args are not sent
    to the wrong frame, and the Great Library's box tokens live there. Picking
    wrong therefore surfaces far away as an `UnsupportedBgaState` refusal.

    Dropping `testuser=` frames handles the observed case. The two additions
    here are about the cases that convention CANNOT resolve: they now raise a
    tagged error instead of silently taking the shallowest frame. "quiet" is
    correct for a page with no board, because this bridge runs in every frame
    and most of them are not the game.
    """
    script = _FRAME_HARNESS % {"snippet": json.dumps(str(_CANONICAL))}
    assert _run_node(script).strip().split(",") == [
        "seat:111",  # ordinary table: one board frame
        "seat:111",  # the observed hazard: opponent frame ignored
        "loud",      # two unmarked frames on different seats: unknowable
        "loud",      # spectator: reads the board, gets no private args
        "quiet",     # not a game page at all
        "loud",      # ONLY impersonated frames: refuse, do not fall back
    ]
