# advisor_server.py
#
# Local HTTP server for the Can't Stop game advisor.
#
# Purpose
# -------
# Listen on localhost. Accept a JSON-encoded game state. Run MCTS using
# the loaded model. Return a ranked list of recommended actions with
# value estimates, visit counts, and priors.
#
# Architecture
# ------------
# This is the "smart" half of the eventual browser-extension setup:
#   [BGA tab + browser ext]  →  HTTP  →  [this server]  →  recommendation
#
# For the prototype phase, the "browser extension" half is replaced by
# a manual entry web UI also served by this process. Same JSON contract,
# different source.
#
# Endpoints
# ---------
#   GET  /                — serves the manual entry HTML page
#   GET  /health          — returns model info + readiness
#   POST /recommend       — game-state JSON in, recommendation JSON out
#   GET  /static/*        — CSS/JS assets for the UI
#
# Run from project root:
#   python -m games.cantstop.advisor_server --model models/cantstop/.../best_model.pt
#
# Then open http://127.0.0.1:8765/ in any browser.

import os
import sys
import json
import time
import argparse
import threading
import http.server
import socketserver
from urllib.parse import urlparse

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import GameState, get_valid_moves, bust_turn
from games.cantstop.features import action_to_move_decision, ACTION_SPACE
from games.cantstop.evaluate import load_model
from games.cantstop.mcts import MCTS


# ============================================================
# JSON game-state schema
# ============================================================
# Wire format for the request body of POST /recommend:
#
# {
#   "active_player": 0,                    # 0 = me, 1 = opponent
#   "scores":   [1, 2],                    # claimed-column count per player
#   "claimed":  [[7, 12], [2, 8, 9]],      # which columns each player has won
#   "progress": [{"5": 2, "8": 3}, {"6": 1}],
#                                          # locked-in (saved) progress per player
#   "runners":  {"4": 1, "9": 2},          # active player's temporary markers
#   "dice":     [1, 3, 4, 6],              # current dice — empty if not rolling yet
#   "num_simulations": 500                 # optional override; default in server cfg
# }
#
# Response shape from /recommend:
#
# {
#   "ok": true,
#   "value":           0.62,               # win probability from this position
#   "search_ms":       142,                # wall-clock cost of the MCTS search
#   "recommendations": [
#     {
#       "rank": 1,
#       "move":   [3, 6],                  # tuple of column targets
#       "decision": "continue",            # "continue" | "stop"
#       "action_idx":  42,                 # internal action index (debug)
#       "visits":      214,                # raw MCTS visit count
#       "visit_frac":  0.43,               # share of total visits (= policy target)
#       "q_value":     0.58,               # MCTS Q estimate for this child
#       "prior":       0.31,               # raw network policy for this action
#     },
#     ...                                  # all legal actions, ranked by visits
#   ],
#   "warnings":        []                  # any state-parse issues that didn't fatal
# }


# ============================================================
# Server state
# ============================================================
# Owned by the main process. Workers in the inference server (if used)
# would be in separate processes but for the advisor we run inference
# locally on the model since requests are sparse (1 per turn, manual).

# Globals populated by main():
_mcts = None
_model = None
_model_path = None
_default_sims = 500
_lock = threading.Lock()   # serializes /recommend (MCTS shares state)


# ============================================================
# JSON ↔ GameState conversion
# ============================================================

def json_to_state(payload, warnings):
    """
    Build a GameState from the wire format. Tolerant of small format
    variations (string vs int keys in dicts) since the eventual
    extension may serialize differently than the form.

    Mutates `warnings` (a list) with non-fatal issues for the client.
    Raises ValueError on fatal schema problems.
    """
    state = GameState(num_players=2)

    if 'active_player' not in payload:
        raise ValueError("missing required field: active_player")
    state.active_player = int(payload['active_player'])
    if state.active_player not in (0, 1):
        raise ValueError(f"active_player must be 0 or 1, got "
                         f"{state.active_player}")

    # Claimed columns per player.
    claimed_payload = payload.get('claimed', [[], []])
    if len(claimed_payload) != 2:
        raise ValueError("claimed must have exactly 2 sublists")
    for p in (0, 1):
        cols = set(int(c) for c in claimed_payload[p])
        state.claimed[p] = cols
        state.all_claimed.update(cols)

    # Saved progress per player.
    progress_payload = payload.get('progress', [{}, {}])
    if len(progress_payload) != 2:
        raise ValueError("progress must have exactly 2 dicts")
    for p in (0, 1):
        # JSON keys are strings — convert to int for engine.
        prog = {int(col): int(amount)
                for col, amount in progress_payload[p].items()}
        state.progress[p] = prog

    # Active player's runners — small dict of col → step count.
    runners_payload = payload.get('runners', {})
    state.runners = {int(col): int(amount)
                     for col, amount in runners_payload.items()}

    # Dice (may be empty if it's a fresh turn pre-roll).
    dice_payload = payload.get('dice', [])
    state.dice = [int(d) for d in dice_payload]
    for d in state.dice:
        if not (1 <= d <= 6):
            raise ValueError(f"dice values must be 1-6, got {d}")
    if state.dice and len(state.dice) != 4:
        warnings.append(f"dice should have exactly 4 values; got "
                        f"{len(state.dice)}")

    # Sanity: claimed columns should not also have runners or progress.
    for p in (0, 1):
        for col in state.claimed[p]:
            if col in state.progress[p]:
                warnings.append(f"player {p} has both claimed and "
                                f"progress on column {col}; ignoring "
                                f"progress")
                del state.progress[p][col]
        if p == state.active_player:
            for col in state.claimed[p]:
                if col in state.runners:
                    warnings.append(f"active player claimed column "
                                    f"{col} but also has a runner there; "
                                    f"ignoring runner")
                    del state.runners[col]

    return state


# ============================================================
# Recommendation: run MCTS and extract rich output
# ============================================================

def make_recommendations(state, num_simulations):
    """
    Run MCTS on `state`, return the response dict.

    Unlike mcts.search() which returns (policy_vector, value), we need
    per-child breakdown: visits, Q, prior, action_idx. The cleanest way
    is to replicate the structure of MCTS.search() but stop short of
    collapsing children into a policy vector.
    """
    from games.cantstop.mcts import DecisionNode

    # Validate up front so we don't run MCTS and then realize the
    # position is broken.
    if state.game_over:
        return {
            'ok': False,
            'error': 'game is already over',
            'winner': state.winner,
        }

    # If dice haven't been rolled yet, we can't make a recommendation —
    # there's nothing to decide. Surface this clearly.
    if not state.dice:
        return {
            'ok': False,
            'error': 'no dice present — roll first, then ask for a '
                     'recommendation',
        }

    valid = get_valid_moves(state)
    if not valid:
        # Bust position. The "advice" here is just "you busted; click
        # whatever BGA wants to confirm the bust."
        return {
            'ok': True,
            'value': 0.0,
            'search_ms': 0,
            'recommendations': [],
            'note': "bust — no legal moves; turn ends without scoring",
        }

    t0 = time.perf_counter()

    # Replicate MCTS.search()'s setup so we can keep `root` around to
    # inspect children. We can't reuse search() because it doesn't
    # return the tree.
    root = DecisionNode(
        state=state.clone(),
        parent=None,
        parent_action=None,
        prior=0.0,
        flip_from_parent=False,
    )

    # Initial expand — same as search() does for the root.
    root_value, root_priors = _mcts.evaluate(
        root.state, root.valid_moves, root.mask
    )
    _mcts.expand_decision_node(root, root_priors)
    root.N = 1
    root.W = root_value

    # Run the simulations. Reuse the existing scheduler (sync, since
    # we configured target_inflight=1 on the local advisor MCTS).
    if num_simulations - 1 > 0:
        # Disable Dirichlet for advisor — we want sharp recommendations,
        # not exploration. dirichlet_epsilon=0 in the underlying calls
        # would be cleaner but those happen inside search(); since we
        # bypass search(), we just don't apply noise here.
        _mcts._run_sync_simulations(root, num_simulations - 1)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Build per-child recommendations.
    total_visits = sum(child.N for child in root.children.values())
    recs = []
    for action_idx, child in root.children.items():
      move, decision = action_to_move_decision(int(action_idx))
      visits = int(child.N)

      # child.Q is in the child's active_player perspective. When the
      # turn flips across this edge (the "stop" decision passes control
      # to the opponent), invert to express it from the root player's
      # perspective. Matches the convention in DecisionNode.ucb_score.
      child_q = float(child.Q)
      if child.flip_from_parent:
          child_q = 1.0 - child_q

      recs.append({
          'rank':       0,   # filled in after sort
          'move':       list(move),
          'decision':   decision,
          'action_idx': int(action_idx),
          'visits':     visits,
          'visit_frac': (visits / total_visits) if total_visits > 0
                        else 0.0,
          'q_value':    child_q,
          'prior':      float(child.prior),
      })

    # Sort by visits descending — this is what AlphaZero takes as the
    # "best move" criterion. Q would be an alternative but visits are
    # more robust to single-sim noise.
    recs.sort(key=lambda r: (-r['visits'], -r['q_value']))
    for i, r in enumerate(recs):
        r['rank'] = i + 1

    return {
        'ok':              True,
        'value':           float(root.Q),
        'search_ms':       elapsed_ms,
        'num_simulations': num_simulations,
        'recommendations': recs,
    }


# ============================================================
# HTTP request handler
# ============================================================

class AdvisorHandler(http.server.BaseHTTPRequestHandler):

    # Silence the default request logging — too noisy for an
    # interactive prototype.
    def log_message(self, fmt, *args):
        return

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        # Allow same-origin requests + any localhost origin. The eventual
        # browser extension may need this — for the prototype it's only
        # used by the served page.
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        # CORS preflight — respond with the allowed headers/methods.
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            self._send_html(_HTML_PAGE)
            return
        if path == '/health':
            self._send_json(200, {
                'ok':                True,
                'model_path':        _model_path,
                'default_sims':      _default_sims,
            })
            return
        # Anything else: 404
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b'not found')

    def do_POST(self):
        path = urlparse(self.path).path
        if path != '/recommend':
            self.send_response(404)
            self.end_headers()
            return

        # Read body.
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length <= 0:
            self._send_json(400, {'ok': False,
                                  'error': 'empty request body'})
            return
        try:
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode('utf-8'))
        except Exception as e:
            self._send_json(400, {'ok': False,
                                  'error': f'invalid JSON: {e}'})
            return

        # Convert to GameState.
        warnings = []
        try:
            state = json_to_state(payload, warnings)
        except ValueError as e:
            self._send_json(400, {'ok': False,
                                  'error': str(e),
                                  'warnings': warnings})
            return

        # Determine sim count.
        num_simulations = int(payload.get('num_simulations',
                                          _default_sims))
        num_simulations = max(10, min(num_simulations, 5000))

        # Run MCTS under the lock — MCTS internals reuse state on the
        # current implementation. The lock keeps two browser tabs from
        # corrupting each other if you accidentally have it open twice.
        with _lock:
            try:
                response = make_recommendations(state, num_simulations)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json(500, {'ok': False,
                                      'error': f'MCTS failed: {e}'})
                return

        if warnings:
            response['warnings'] = warnings
        self._send_json(200, response)


# ============================================================
# Embedded HTML / CSS / JS
# ============================================================
# Kept inline so the prototype is a single file. When this graduates to
# a real product, split these out into static files served from disk.

_HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Can't Stop Advisor</title>
  <style>
    :root {
      --bg: #0f1419;
      --panel: #1a2330;
      --panel-soft: #232f40;
      --text: #e6edf3;
      --text-dim: #8b98a5;
      --accent: #58a6ff;
      --good: #56d364;
      --bad:  #f85149;
      --warn: #d29922;
      --border: #30363d;
    }
    body { background: var(--bg); color: var(--text);
           font-family: ui-sans-serif, -apple-system, system-ui, sans-serif;
           margin: 0; padding: 24px; }
    h1 { font-size: 20px; margin: 0 0 16px; font-weight: 600; }
    h2 { font-size: 14px; margin: 16px 0 8px; color: var(--text-dim);
         text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
    .layout { display: grid; grid-template-columns: 1fr 1fr;
              gap: 24px; max-width: 1400px; margin: 0 auto; }
    .panel { background: var(--panel); border: 1px solid var(--border);
             border-radius: 8px; padding: 20px; }
    label { display: block; font-size: 13px; color: var(--text-dim);
            margin-bottom: 4px; }
    input[type="text"], input[type="number"] {
      background: var(--panel-soft); color: var(--text);
      border: 1px solid var(--border); border-radius: 4px;
      padding: 6px 10px; font-size: 14px; width: 100%; box-sizing: border-box;
      font-family: ui-monospace, monospace;
    }
    .field-row { display: grid; grid-template-columns: 1fr 1fr;
                 gap: 12px; margin-bottom: 12px; }
    .field-row label { margin-bottom: 4px; }
    button { background: var(--accent); color: var(--bg); border: 0;
             border-radius: 4px; padding: 10px 18px; font-size: 14px;
             font-weight: 600; cursor: pointer; }
    button:hover { filter: brightness(1.1); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .dice-input { display: flex; gap: 8px; }
    .dice-input input { width: 50px; text-align: center;
                        font-size: 18px; font-weight: 600; }
    .hint { font-size: 12px; color: var(--text-dim); margin-top: 4px; }
    .response-meta { display: flex; gap: 24px; font-size: 13px;
                     color: var(--text-dim); margin-bottom: 12px; }
    .response-meta strong { color: var(--text); }
    .rec-list { display: flex; flex-direction: column; gap: 8px; }
    .rec {
      background: var(--panel-soft); border: 1px solid var(--border);
      border-radius: 6px; padding: 12px;
      display: grid; grid-template-columns: 36px 1fr auto; gap: 14px;
      align-items: center;
    }
    .rec.top { border-color: var(--good);
               box-shadow: 0 0 0 1px var(--good) inset; }
    .rank { font-size: 20px; font-weight: 700; color: var(--text-dim);
            text-align: center; }
    .rec.top .rank { color: var(--good); }
    .rec-main { display: flex; flex-direction: column; gap: 4px; }
    .rec-move { font-size: 15px; font-weight: 600; }
    .rec-decision { display: inline-block; font-size: 11px;
                    padding: 1px 6px; border-radius: 3px;
                    background: var(--panel); color: var(--text-dim);
                    margin-left: 8px; text-transform: uppercase;
                    letter-spacing: 0.04em; }
    .rec-decision.continue { color: var(--accent); }
    .rec-decision.stop { color: var(--warn); }
    .rec-stats { display: flex; gap: 16px; font-size: 12px;
                 color: var(--text-dim); }
    .rec-stats strong { color: var(--text); font-weight: 600; }
    .rec-right { text-align: right; }
    .rec-q { font-size: 20px; font-weight: 700; }
    .rec-q.good { color: var(--good); }
    .rec-q.bad  { color: var(--bad); }
    .rec-q-label { font-size: 11px; color: var(--text-dim);
                   text-transform: uppercase; letter-spacing: 0.04em; }
    .empty { color: var(--text-dim); font-style: italic; padding: 40px 0;
             text-align: center; }
    .error { background: rgba(248,81,73,0.1); color: var(--bad);
             border: 1px solid var(--bad); border-radius: 4px;
             padding: 10px 14px; font-size: 13px; }
    .warn-box { background: rgba(210,153,34,0.1); color: var(--warn);
                border: 1px solid var(--warn); border-radius: 4px;
                padding: 8px 12px; font-size: 12px; margin-bottom: 10px; }
    .actions { display: flex; gap: 10px; margin-top: 16px; align-items: center; }
    .sim-input { width: 80px !important; }
    code { background: var(--panel); padding: 1px 5px; border-radius: 3px;
           font-size: 12px; }
  </style>
</head>
<body>
  <h1>Can't Stop Advisor — Manual Position Entry</h1>
  <div class="layout">
    <!-- LEFT: position input -->
    <div class="panel">
      <h2>Your position (player 0)</h2>
      <div class="field-row">
        <div>
          <label>Claimed columns</label>
          <input type="text" id="p0_claimed" placeholder="e.g. 7, 12" />
          <div class="hint">Columns you've already won. Comma-separated.</div>
        </div>
        <div>
          <label>Saved progress</label>
          <input type="text" id="p0_progress"
                 placeholder="e.g. 5:2, 8:3" />
          <div class="hint">column:steps from finished previous turns.</div>
        </div>
      </div>

      <h2>Opponent's position (player 1)</h2>
      <div class="field-row">
        <div>
          <label>Claimed columns</label>
          <input type="text" id="p1_claimed" placeholder="e.g. 2, 8, 9" />
        </div>
        <div>
          <label>Saved progress</label>
          <input type="text" id="p1_progress"
                 placeholder="e.g. 6:1" />
        </div>
      </div>

      <h2>Current turn (it's your turn, mid-roll)</h2>
      <div class="field-row">
        <div>
          <label>Your runners (this turn so far)</label>
          <input type="text" id="runners" placeholder="e.g. 4:1, 9:2" />
          <div class="hint">Temporary markers placed this turn.</div>
        </div>
        <div>
          <label>Dice (4 values, 1-6)</label>
          <div class="dice-input">
            <input type="number" min="1" max="6" id="d0" />
            <input type="number" min="1" max="6" id="d1" />
            <input type="number" min="1" max="6" id="d2" />
            <input type="number" min="1" max="6" id="d3" />
          </div>
        </div>
      </div>

      <div class="actions">
        <button id="go">Recommend move</button>
        <label style="margin: 0;">Sims:</label>
        <input type="number" id="sims" class="sim-input" value="500"
               min="10" max="5000" step="50" />
        <span style="color: var(--text-dim); font-size: 12px;"
              id="status"></span>
      </div>
    </div>

    <!-- RIGHT: recommendations -->
    <div class="panel">
      <h2>Recommendation</h2>
      <div id="result"><div class="empty">Enter a position on the left,
      then click <strong>Recommend move</strong>.</div></div>
    </div>
  </div>

<script>
"use strict";

function parseColumnList(s) {
  if (!s || !s.trim()) return [];
  return s.split(",").map(x => parseInt(x.trim(), 10))
          .filter(x => !isNaN(x));
}

function parseColumnDict(s) {
  // "5:2, 8:3" → { "5": 2, "8": 3 }
  if (!s || !s.trim()) return {};
  const out = {};
  for (const piece of s.split(",")) {
    const parts = piece.split(":");
    if (parts.length !== 2) continue;
    const k = parseInt(parts[0].trim(), 10);
    const v = parseInt(parts[1].trim(), 10);
    if (!isNaN(k) && !isNaN(v)) out[String(k)] = v;
  }
  return out;
}

function collectPayload() {
  const p0_claimed  = parseColumnList(document.getElementById('p0_claimed').value);
  const p1_claimed  = parseColumnList(document.getElementById('p1_claimed').value);
  const p0_progress = parseColumnDict(document.getElementById('p0_progress').value);
  const p1_progress = parseColumnDict(document.getElementById('p1_progress').value);
  const runners     = parseColumnDict(document.getElementById('runners').value);

  const dice = [];
  for (let i = 0; i < 4; i++) {
    const v = parseInt(document.getElementById('d' + i).value, 10);
    if (!isNaN(v)) dice.push(v);
  }

  const sims = parseInt(document.getElementById('sims').value, 10) || 500;

  return {
    active_player: 0,    // advisor always asks about "me", which is player 0
    claimed:  [p0_claimed,  p1_claimed],
    progress: [p0_progress, p1_progress],
    runners:  runners,
    dice:     dice,
    num_simulations: sims,
  };
}

function renderError(msg, warnings) {
  let html = `<div class="error">${msg}</div>`;
  if (warnings && warnings.length) {
    for (const w of warnings) {
      html += `<div class="warn-box">⚠ ${w}</div>`;
    }
  }
  document.getElementById('result').innerHTML = html;
}

function renderRecommendation(resp) {
  let html = '';

  if (resp.warnings && resp.warnings.length) {
    for (const w of resp.warnings) {
      html += `<div class="warn-box">⚠ ${w}</div>`;
    }
  }

  if (resp.note) {
    html += `<div class="warn-box">${resp.note}</div>`;
  }

  const valuePct = (resp.value * 100).toFixed(1);
  const valueColor = resp.value >= 0.5 ? 'good' : 'bad';
  html += `<div class="response-meta">
    <span>Position value: <strong class="${valueColor}">${valuePct}%</strong></span>
    <span>Search: <strong>${resp.search_ms}ms</strong></span>
    <span>Sims: <strong>${resp.num_simulations}</strong></span>
  </div>`;

  if (!resp.recommendations || resp.recommendations.length === 0) {
    html += `<div class="empty">No legal moves — turn ends.</div>`;
    document.getElementById('result').innerHTML = html;
    return;
  }

  html += `<div class="rec-list">`;
  for (const r of resp.recommendations) {
    const qpct = (r.q_value * 100).toFixed(1);
    const qclass = r.q_value >= 0.5 ? 'good' : 'bad';
    const topClass = r.rank === 1 ? ' top' : '';
    const moveStr = r.move.length === 1
                    ? `column ${r.move[0]}`
                    : `columns ${r.move[0]} + ${r.move[1]}`;
    const decClass = r.decision === 'stop' ? 'stop' : 'continue';
    html += `<div class="rec${topClass}">
      <div class="rank">#${r.rank}</div>
      <div class="rec-main">
        <div>
          <span class="rec-move">${moveStr}</span>
          <span class="rec-decision ${decClass}">${r.decision}</span>
        </div>
        <div class="rec-stats">
          <span><strong>${r.visits}</strong> visits
            (${(r.visit_frac*100).toFixed(0)}%)</span>
          <span>prior <strong>${(r.prior*100).toFixed(1)}%</strong></span>
        </div>
      </div>
      <div class="rec-right">
        <div class="rec-q ${qclass}">${qpct}%</div>
        <div class="rec-q-label">est. win</div>
      </div>
    </div>`;
  }
  html += `</div>`;

  document.getElementById('result').innerHTML = html;
}

document.getElementById('go').addEventListener('click', async () => {
  const btn    = document.getElementById('go');
  const status = document.getElementById('status');
  btn.disabled = true;
  status.textContent = 'thinking...';

  const payload = collectPayload();
  try {
    const r = await fetch('/recommend', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await r.json();
    if (data.ok) {
      renderRecommendation(data);
      status.textContent = '';
    } else {
      renderError(data.error || 'unknown error', data.warnings);
      status.textContent = 'error';
    }
  } catch (e) {
    renderError('Network error: ' + e.message);
    status.textContent = 'error';
  } finally {
    btn.disabled = false;
  }
});

// Enter key in any field triggers recommend.
document.querySelectorAll('input').forEach(el => {
  el.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') document.getElementById('go').click();
  });
});
</script>
</body>
</html>
"""


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Local Can't Stop advisor server."
    )
    parser.add_argument('--model', required=True,
                        help='Path to model checkpoint (.pt file).')
    parser.add_argument('--port', type=int, default=8765,
                        help='Listen port (default 8765).')
    parser.add_argument('--host', default='127.0.0.1',
                        help='Listen host (default 127.0.0.1, '
                             'localhost only).')
    parser.add_argument('--sims', type=int, default=500,
                        help='Default MCTS sims per request (default '
                             '500). Client can override per-request.')
    parser.add_argument('--device', default=None,
                        help='Inference device (default: cuda if '
                             'available, else cpu).')
    args = parser.parse_args()

    global _mcts, _model, _model_path, _default_sims
    _model_path = args.model
    _default_sims = args.sims

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model: {args.model}")
    print(f"Device:        {device}")
    _model = load_model(args.model, device)

    # Sync MCTS — no GPU server. Single request at a time means async
    # scheduling adds bookkeeping without payoff. We do NOT need warmup
    # either for the same reason.
    _mcts = MCTS(_model, device,
                 target_inflight=1, warmup_sims=0)

    addr = (args.host, args.port)
    httpd = socketserver.ThreadingTCPServer(addr, AdvisorHandler)
    print()
    print("=" * 60)
    print(f"  Can't Stop Advisor")
    print("=" * 60)
    print(f"  Model:        {args.model}")
    print(f"  Device:       {device}")
    print(f"  Default sims: {args.sims}")
    print(f"  Listening:    http://{args.host}:{args.port}/")
    print()
    print(f"  Open http://{args.host}:{args.port}/ in your browser.")
    print(f"  Ctrl-C to stop.")
    print("=" * 60)
    print()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        httpd.server_close()


if __name__ == "__main__":
    main()