let sessionId = null;
let sessionMeta = null;
let currentState = null;
let currentLegalActions = [];
let selectedPickDominoId = null;
let selectedFirstCell = null;
let previewAction = null;
let pendingMoveAction = null;
let showCroppedBoards = false;
let swapHalves = false;
let autoPlayRunning = false;

const $ = (id) => document.getElementById(id);

function setStatus(msg, isError = false) {
  const el = $('status');
  el.textContent = msg;
  el.style.color = isError ? 'var(--danger)' : 'var(--muted)';
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.ok === false) {
    throw new Error(data.detail || data.error || `${res.status} ${res.statusText}`);
  }
  return data;
}

function updateSessionMeta(data) {
  if (!data) return;
  if (data.session_id) sessionId = data.session_id;
  if (data.session) sessionMeta = data.session;
  renderTimelineControls();
}

function renderTimelineControls() {
  const step = sessionMeta?.timeline_step ?? 0;
  const length = sessionMeta?.timeline_length ?? 1;
  const canUndo = Boolean(sessionMeta?.can_undo);
  const undoBtn = $('undoBtn');
  const jumpBtn = $('jumpBtn');
  const jumpInput = $('jumpStepInput');
  const timeline = $('timelineInfo');
  if (undoBtn) undoBtn.disabled = !canUndo;
  if (jumpBtn) jumpBtn.disabled = length <= 1;
  if (jumpInput) {
    jumpInput.max = Math.max(0, length - 1);
    if (document.activeElement !== jumpInput) jumpInput.value = String(step);
  }
  if (timeline) timeline.textContent = `Step ${step} / ${Math.max(0, length - 1)}`;
}

function terrainClass(id) { return `t${Number(id || 0)}`; }
function crownText(n) { return n > 0 ? '♛'.repeat(n) : ''; }
function halfLabel(h, label = '') { return `${label}${h.terrain}${h.crowns ? ' +' + h.crowns : ''}`; }
function currentDomino() { return currentTask().current_domino || null; }
function halfShort(h) { return h ? `${h.terrain}${h.crowns ? ' +' + h.crowns : ''}` : '—'; }
function placementHalfAssignments(action) {
  if (!action?.placement || !action?.domino) return [];
  const p = action.placement;
  const h1 = p.flipped ? action.domino.b : action.domino.a;
  const h2 = p.flipped ? action.domino.a : action.domino.b;
  return [
    { label: p.flipped ? 'B' : 'A', half: h1, x: p.x1, y: p.y1 },
    { label: p.flipped ? 'A' : 'B', half: h2, x: p.x2, y: p.y2 },
  ];
}
function moveSummaryHtml(action) {
  if (!action) return '<p class="hint">No placement selected yet.</p>';
  const parts = placementHalfAssignments(action).map(c =>
    `<div><strong>${c.label}</strong> ${halfShort(c.half)} at (${c.x},${c.y})</div>`
  ).join('');
  const pick = action.pick_domino_id !== null && action.pick_domino_id !== undefined
    ? `<div><strong>Pick</strong> domino #${action.pick_domino_id}</div>`
    : '<div><strong>Pick</strong> none</div>';
  return `<div class="pending-move-summary">${parts}${pick}<small>${action.action_id}</small></div>`;
}
function samePlacement(a, b) {
  if (!a || !b) return false;
  return a.x1 === b.x1 && a.y1 === b.y1 && a.x2 === b.x2 && a.y2 === b.y2 && Boolean(a.flipped) === Boolean(b.flipped);
}

function renderTile(domino, opts = {}) {
  if (!domino) return '';
  const selected = opts.selected ? ' selected' : '';
  const clickable = opts.clickable ? ' clickable' : '';
  const subtitle = opts.subtitle ? `<div class="tile-subtitle">${opts.subtitle}</div>` : '';
  return `<div class="tile${selected}${clickable}" data-domino-id="${domino.id}">
    <div class="tile-id">#${domino.id}</div>
    <div class="tile-halves">
      <div class="tile-half ${terrainClass(domino.a.terrain_id)}"><span>A</span>${halfLabel(domino.a)}</div>
      <div class="tile-half ${terrainClass(domino.b.terrain_id)}"><span>B</span>${halfLabel(domino.b)}</div>
    </div>
    ${subtitle}
  </div>`;
}

function currentActorBoardIndex() {
  return currentState && currentState.current_actor !== null ? Number(currentState.current_actor) : null;
}

function currentTask() {
  return currentState?.current_task || {};
}

function currentBotMode() {
  const actor = currentActorBoardIndex();
  if (actor === null) return 'human';
  const el = $(`botMode${actor}`);
  return el ? el.value : 'human';
}

function botRequestPayload(mode, apply = true) {
  return {
    session_id: sessionId,
    mode,
    apply,
    checkpoint_path: $('nnCheckpointInput')?.value?.trim() || null,
    nn_sims: Number($('nnSimsInput')?.value || 50),
    determinizations: 1,
    temperature: 0,
    device: $('nnDeviceSelect')?.value || 'cpu',
    channels: 64,
    blocks: 6,
    bilinear_dim: 64,
    seed: Number($('seedInput')?.value || 0),
  };
}

function renderBotStatus() {
  const actor = currentActorBoardIndex();
  const mode = currentBotMode();
  const el = $('botInfo');
  if (!el) return;
  if (actor === null) {
    el.innerHTML = '<span class="hint">Game over.</span>';
    return;
  }
  el.innerHTML = `Current actor P${actor}: <strong>${mode}</strong>${mode === 'human' ? ' — make a move manually.' : ' — bot can move now.'}`;
}

function getSelectedPickActionCount() {
  if (selectedPickDominoId === null) return 0;
  return currentLegalActions.filter(a => a.pick_domino_id === selectedPickDominoId).length;
}

function defaultPickIfNeeded() {
  const task = currentTask();
  if (!task.requires_pick || currentState.phase === 'INITIAL_SELECTION') return;
  if (selectedPickDominoId !== null) {
    const stillLegal = currentLegalActions.some(a => a.pick_domino_id === selectedPickDominoId);
    if (stillLegal) return;
  }
  const ids = [...new Set(currentLegalActions.map(a => a.pick_domino_id).filter(x => x !== null && x !== undefined))];
  selectedPickDominoId = ids.length === 1 ? ids[0] : null;
}

function resetPlacementSelection() {
  selectedFirstCell = null;
  previewAction = null;
  pendingMoveAction = null;
}

function resetSelectionsForNewState() {
  selectedPickDominoId = null;
  selectedFirstCell = null;
  previewAction = null;
  defaultPickIfNeeded();
}

function instructionHtml(state) {
  const task = state.current_task || {};
  const pickMsg = task.requires_pick && state.phase !== 'INITIAL_SELECTION'
    ? (selectedPickDominoId ? `Future pick selected: #${selectedPickDominoId}` : 'Select a future domino from the row first')
    : '';
  const placeMsg = task.requires_placement
    ? (pendingMoveAction ? 'Move ready. Review the preview, then click Apply selected move.'
      : (selectedFirstCell ? `First cell selected: (${selectedFirstCell.x},${selectedFirstCell.y}). Click an adjacent cell for the other half.` : 'Click two adjacent cells on the current actor board to place.'))
    : '';
  return `<div class="task-banner">
    <div>
      <div class="task-title">${task.title || 'Kingdomino'}</div>
      <div class="task-detail">${task.detail || ''}</div>
    </div>
    <div class="task-status">
      ${pickMsg ? `<div>${pickMsg}</div>` : ''}
      ${placeMsg ? `<div>${placeMsg}</div>` : ''}
    </div>
  </div>`;
}

function renderSummary(state) {
  const cur = state.current_actor === null ? '—' : `P${state.current_actor}`;
  $('task').innerHTML = instructionHtml(state);
  $('summary').innerHTML = `<div class="kv">
    <div>Session</div><div>${sessionId || '—'}</div>
    <div>Phase</div><div>${state.phase}</div>
    <div>Current actor</div><div>${cur}</div>
    <div>Current claim</div><div>${state.current_claim ? `P${state.current_claim.player} / #${state.current_claim.domino_id}` : '—'}</div>
    <div>Legal actions</div><div>${state.legal_action_count ?? currentLegalActions.length}</div>
    <div>Scores</div><div>P0 ${state.scores[0]} — P1 ${state.scores[1]}</div>
    <div>Deck count</div><div>${state.deck_count}</div>
    <div>History</div><div>${state.history_len} actions</div>
    <div>Timeline</div><div>${sessionMeta ? `step ${sessionMeta.timeline_step} / ${sessionMeta.timeline_length - 1}` : '—'}</div>
  </div>`;

  const task = state.current_task || {};
  $('currentDomino').innerHTML = task.current_domino
    ? renderTile(task.current_domino, { subtitle: 'Current domino to place' })
    : '<p class="hint">No domino to place in this phase.</p>';

  $('currentRow').innerHTML = state.current_row_tiles.map(d => renderTile(d, {
    clickable: true,
    selected: selectedPickDominoId === d.id,
    subtitle: state.phase === 'INITIAL_SELECTION' ? 'Click to pick now' : 'Future pick option',
  })).join('') || '<p>No current row.</p>';
  document.querySelectorAll('#currentRow .tile.clickable').forEach(tile => {
    tile.addEventListener('click', () => onRowTileClick(Number(tile.dataset.dominoId)));
  });

  $('claims').innerHTML = `
    <div><strong>Pending</strong>${state.pending_claims.map(c => `<div class="claim ${state.current_claim && c.player === state.current_claim.player && c.domino_id === state.current_claim.domino_id ? 'active' : ''}">P${c.player}: #${c.domino_id}</div>`).join('') || '<div class="claim">—</div>'}</div>
    <div><strong>Next</strong>${state.next_claims.map(c => `<div class="claim">P${c.player}: #${c.domino_id}</div>`).join('') || '<div class="claim">—</div>'}</div>
  `;

  $('placementHelp').innerHTML = task.requires_placement ? `
    <div class="placement-controls">
      <label><input id="swapHalves" type="checkbox" ${swapHalves ? 'checked' : ''}/> Prefer B first when both half-orders are legal</label>
      <button id="clearPlacementBtn" class="secondary">Clear placement</button>
    </div>
    <div class="pending-move ${pendingMoveAction ? 'ready' : ''}">
      <h4>Selected move</h4>
      ${moveSummaryHtml(pendingMoveAction)}
      <div class="pending-buttons">
        <button id="applyPendingMoveBtn" ${pendingMoveAction ? '' : 'disabled'}>Apply selected move</button>
        <button id="clearPendingMoveBtn" class="secondary" ${selectedFirstCell || pendingMoveAction ? '' : 'disabled'}>Clear</button>
      </div>
    </div>
    <p class="hint">Choose a future pick if needed, then click two adjacent cells. The UI now auto-matches either A-first or B-first legal placements; the checkbox only sets your preferred half order when both are available.</p>
  ` : '<p class="hint">Placement controls appear when a domino must be placed.</p>';
  const swap = $('swapHalves');
  if (swap) swap.addEventListener('change', (e) => { swapHalves = Boolean(e.target.checked); pendingMoveAction = null; previewAction = null; renderAll(currentState, currentLegalActions, false); });
  const clear = $('clearPlacementBtn');
  if (clear) clear.addEventListener('click', () => { resetPlacementSelection(); renderAll(currentState, currentLegalActions, false); });
  const clearPending = $('clearPendingMoveBtn');
  if (clearPending) clearPending.addEventListener('click', () => { resetPlacementSelection(); renderAll(currentState, currentLegalActions, false); });
  const applyPending = $('applyPendingMoveBtn');
  if (applyPending) applyPending.addEventListener('click', applyPendingMove);
}

function boardViewBounds(board) {
  if (!showCroppedBoards) return { minX: 0, minY: 0, maxX: board.canvas_size - 1, maxY: board.canvas_size - 1 };
  const b = board.bbox || [0, 0, board.canvas_size - 1, board.canvas_size - 1];
  return {
    minX: Math.max(0, b[0] - 2),
    minY: Math.max(0, b[1] - 2),
    maxX: Math.min(board.canvas_size - 1, b[2] + 2),
    maxY: Math.min(board.canvas_size - 1, b[3] + 2),
  };
}

function cellTerrainName(board, x, y) {
  return board.cells.find(c => c.x === x && c.y === y)?.terrain || 'EMPTY';
}

function actionPickMatches(a) {
  if (currentState.phase === 'PLACE_AND_SELECT') return a.pick_domino_id === selectedPickDominoId;
  return true;
}

function sameUnorderedCells(p, x1, y1, x2, y2) {
  return (
    (p.x1 === x1 && p.y1 === y1 && p.x2 === x2 && p.y2 === y2) ||
    (p.x1 === x2 && p.y1 === y2 && p.x2 === x1 && p.y2 === y1)
  );
}

function clickedCellLabelForAction(action, x, y) {
  const assignment = placementHalfAssignments(action).find(c => c.x === x && c.y === y);
  return assignment ? assignment.label : null;
}

function findActionForPlacement(x1, y1, x2, y2) {
  const candidates = currentLegalActions.filter(a =>
    a && a.placement && actionPickMatches(a) && sameUnorderedCells(a.placement, x1, y1, x2, y2)
  );
  if (!candidates.length) return null;

  // The user clicked cells in a visual order.  Engine placements have their own
  // canonical x1/y1/x2/y2 order plus a flipped flag, so requiring an exact
  // A-first/B-first match makes some legal moves feel unavailable.  Prefer the
  // requested half order, but fall back to any legal action using the same two
  // cells and pick.  The pending-move panel then shows the actual half mapping.
  const preferredFirstLabel = swapHalves ? 'B' : 'A';
  return candidates.find(a => clickedCellLabelForAction(a, x1, y1) === preferredFirstLabel) || candidates[0];
}

function candidateSecondCells(board, player, x, y) {
  const actor = currentActorBoardIndex();
  if (actor !== player || !currentTask().requires_placement) return false;
  if (!selectedFirstCell) return false;
  const dx = Math.abs(selectedFirstCell.x - x);
  const dy = Math.abs(selectedFirstCell.y - y);
  if (dx + dy !== 1) return false;
  if (currentState.phase === 'PLACE_AND_SELECT' && selectedPickDominoId === null) return false;
  return Boolean(findActionForPlacement(selectedFirstCell.x, selectedFirstCell.y, x, y));
}

function previewClassForCell(x, y, player) {
  // Placement previews must only appear on the board that will actually receive
  // the domino.  Coordinates can overlap between player boards, so checking only
  // x/y makes the same ghost placement appear on both boards.
  const actor = currentActorBoardIndex();
  if (actor === null || player !== actor) return '';
  const p = previewAction?.placement;
  if (!p) return '';
  if (p.x1 === x && p.y1 === y) return p.flipped ? ' preview-b' : ' preview-a';
  if (p.x2 === x && p.y2 === y) return p.flipped ? ' preview-a' : ' preview-b';
  return '';
}

function renderBoards(state) {
  const actor = currentActorBoardIndex();
  $('boards').innerHTML = state.boards.map((board, player) => {
    const bounds = boardViewBounds(board);
    const cols = bounds.maxX - bounds.minX + 1;
    const rows = bounds.maxY - bounds.minY + 1;
    let cells = '';
    for (let y = bounds.minY; y <= bounds.maxY; y++) {
      for (let x = bounds.minX; x <= bounds.maxX; x++) {
        const t = board.terrain_grid[y][x];
        const crowns = board.crowns_grid[y][x];
        const isFirst = selectedFirstCell && selectedFirstCell.player === player && selectedFirstCell.x === x && selectedFirstCell.y === y;
        const candidate = candidateSecondCells(board, player, x, y);
        const previewCls = previewClassForCell(x, y, player);
        const title = `P${player} (${x},${y}) ${cellTerrainName(board, x, y)}`;
        cells += `<button class="cell ${terrainClass(t)} ${isFirst ? 'selected-cell' : ''} ${candidate ? 'candidate-cell' : ''} ${previewCls}" data-player="${player}" data-x="${x}" data-y="${y}" title="${title}">${t === 1 ? 'K' : crownText(crowns)}</button>`;
      }
    }
    const s = board.score;
    return `<section class="panel board-wrap ${actor === player ? 'active-board' : ''}">
      <h2>Player ${player}${actor === player ? ' · current actor' : ''}</h2>
      <div class="kv score">
        <div>Score</div><div>${s.total}</div>
        <div>Territory</div><div>${s.territory_score}</div>
        <div>Harmony</div><div>${s.harmony_bonus}</div>
        <div>Middle</div><div>${s.middle_kingdom_bonus}</div>
      </div>
      <div class="board" style="grid-template-columns: repeat(${cols}, 24px); grid-template-rows: repeat(${rows}, 24px);">${cells}</div>
    </section>`;
  }).join('');
  document.querySelectorAll('.cell').forEach(btn => {
    btn.addEventListener('click', () => onBoardCellClick(Number(btn.dataset.player), Number(btn.dataset.x), Number(btn.dataset.y)));
  });
}

function renderActions(actions) {
  currentLegalActions = actions || [];
  defaultPickIfNeeded();
  const grouped = groupActions(currentLegalActions);
  $('actions').innerHTML = grouped || '<p>No legal actions.</p>';
  document.querySelectorAll('.action').forEach(btn => {
    btn.addEventListener('mouseenter', () => {
      previewAction = currentLegalActions[Number(btn.dataset.index)] || null;
      renderBoards(currentState);
    });
    btn.addEventListener('mouseleave', () => {
      previewAction = pendingMoveAction;
      renderBoards(currentState);
    });
    btn.addEventListener('click', () => applyAction(Number(btn.dataset.index)));
  });
}


function placedCellEntriesForAction(action) {
  if (!action?.placement || !action?.domino) return [];
  const p = action.placement;
  const hAtCell1 = p.flipped ? action.domino.b : action.domino.a;
  const hAtCell2 = p.flipped ? action.domino.a : action.domino.b;
  return [
    { x: p.x1, y: p.y1, terrain_id: hAtCell1.terrain_id, terrain: hAtCell1.terrain, crowns: hAtCell1.crowns },
    { x: p.x2, y: p.y2, terrain_id: hAtCell2.terrain_id, terrain: hAtCell2.terrain, crowns: hAtCell2.crowns },
  ];
}

function transformRel(dx, dy, t) {
  // D4 around the castle: four rotations plus their reflected versions.
  switch (t) {
    case 0: return [ dx,  dy];      // identity
    case 1: return [-dy,  dx];      // rot90
    case 2: return [-dx, -dy];      // rot180
    case 3: return [ dy, -dx];      // rot270
    case 4: return [-dx,  dy];      // horizontal flip
    case 5: return [-dy, -dx];      // rot90 + flip
    case 6: return [ dx, -dy];      // rot180 + flip
    case 7: return [ dy,  dx];      // rot270 + flip
    default: return [dx, dy];
  }
}

function canonicalStateAfterActionKey(action) {
  if (!action?.placement || !currentState) {
    return action?.placement ? `placement:${action.action_id}` : `nonplacement:${action.action_id}`;
  }
  const actor = currentActorBoardIndex();
  if (actor === null) return action.action_id;

  // Strategic placement grouping is intentionally based on ONLY the active
  // player's resulting board.  The two players' boards do not share a physical
  // orientation, so an asymmetric opponent board should not split equivalent
  // "straight out" / "sideways" choices on the current player's board.
  const board = currentState.boards[actor];
  const entries = board.cells.map(c => ({
    x: c.x,
    y: c.y,
    terrain_id: c.terrain_id,
    crowns: c.crowns,
  }));
  for (const pc of placedCellEntriesForAction(action)) {
    entries.push({
      x: pc.x,
      y: pc.y,
      terrain_id: pc.terrain_id,
      crowns: pc.crowns,
    });
  }

  const keys = [];
  const [cx, cy] = board.castle_pos;
  for (let t = 0; t < 8; t++) {
    const parts = [];
    for (const e of entries) {
      const [tx, ty] = transformRel(e.x - cx, e.y - cy, t);
      parts.push(`${tx},${ty}:${e.terrain_id}:${e.crowns}`);
    }
    parts.sort();
    keys.push(parts.join('|'));
  }
  return keys.sort()[0];
}

function terrainAtCastleTouch(action) {
  if (!action?.placement || !currentState) return null;
  const actor = currentActorBoardIndex();
  if (actor === null) return null;
  const board = currentState.boards[actor];
  const [cx, cy] = board.castle_pos;
  for (const c of placedCellEntriesForAction(action)) {
    if (Math.abs(c.x - cx) + Math.abs(c.y - cy) === 1) return c.terrain;
  }
  return null;
}

function initialPlacementShapeLabel(action) {
  if (!action?.placement || !currentState) return null;
  const actor = currentActorBoardIndex();
  if (actor === null) return null;
  const board = currentState.boards[actor];
  // Initial placement on this board: only the castle is occupied.
  if (board.occupied_count !== 1) return null;
  const [cx, cy] = board.castle_pos;
  const p = action.placement;
  const cells = [{ x: p.x1, y: p.y1 }, { x: p.x2, y: p.y2 }];
  const touching = cells.find(c => Math.abs(c.x - cx) + Math.abs(c.y - cy) === 1);
  const other = cells.find(c => c !== touching);
  if (!touching || !other) return null;
  const dx1 = touching.x - cx;
  const dy1 = touching.y - cy;
  const dx2 = other.x - touching.x;
  const dy2 = other.y - touching.y;
  const shape = (dx1 === dx2 && dy1 === dy2) ? 'Straight out from castle' : 'Sideways around castle';
  const terrain = terrainAtCastleTouch(action);
  return terrain ? `${shape} · ${terrain} touches K` : shape;
}

function symmetryGroupLabel(actions) {
  if (!actions.length) return 'Equivalent placements';
  const initial = initialPlacementShapeLabel(actions[0]);
  if (initial) return initial;
  if (!actions[0].placement) return 'Discard / no placement';
  return `Equivalent placements (${actions.length})`;
}

function groupBySymmetry(actions) {
  const groups = new Map();
  for (const a of actions) {
    const key = canonicalStateAfterActionKey(a);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(a);
  }
  return [...groups.values()].sort((a, b) => a[0].legal_index - b[0].legal_index);
}

function groupActions(actions) {
  if (!actions.length) return '';
  if (currentState.phase === 'INITIAL_SELECTION') {
    return actions.map(actionButtonHtml).join('');
  }
  const byPick = new Map();
  for (const a of actions) {
    const key = a.pick_domino_id === null || a.pick_domino_id === undefined ? 'No pick' : `Pick #${a.pick_domino_id}`;
    if (!byPick.has(key)) byPick.set(key, []);
    byPick.get(key).push(a);
  }
  return [...byPick.entries()].map(([key, vals]) => {
    const symGroups = groupBySymmetry(vals);
    const openPick = selectedPickDominoId && key === `Pick #${selectedPickDominoId}`;
    const body = symGroups.map((group, i) => {
      const label = symmetryGroupLabel(group);
      const open = symGroups.length <= 4 || i === 0;
      return `<details class="symmetry-group" ${open ? 'open' : ''}>
        <summary>${label} <span>${group.length} physical move${group.length === 1 ? '' : 's'}</span></summary>
        ${group.slice(0, 80).map(actionButtonHtml).join('')}
        ${group.length > 80 ? `<p class="hint">Showing first 80 of ${group.length} equivalent physical moves.</p>` : ''}
      </details>`;
    }).join('');
    return `<details class="action-group" ${openPick ? 'open' : ''}>
      <summary>${key} <span>${vals.length} actions · ${symGroups.length} strategic group${symGroups.length === 1 ? '' : 's'}</span></summary>
      ${body}
    </details>`;
  }).join('');
}

function actionButtonHtml(a) {
  return `<button class="action" data-index="${a.legal_index}">
    <strong>${a.legal_index}.</strong> ${a.label}
    <small>${a.action_id}${a.action_idx !== null && a.action_idx !== undefined ? ` · idx ${a.action_idx}` : ''}</small>
  </button>`;
}

function renderAll(state, legalActions, reset = true) {
  currentState = state;
  if (reset) defaultPickIfNeeded();
  renderSummary(state);
  renderBoards(state);
  renderActions(legalActions || currentLegalActions);
  renderTimelineControls();
  renderBotStatus();
}

async function onRowTileClick(dominoId) {
  if (!currentState) return;
  if (currentState.phase === 'INITIAL_SELECTION') {
    const action = currentLegalActions.find(a => a.kind === 'pick' && a.domino_id === dominoId);
    if (action) return applyAction(action.legal_index);
  }
  selectedPickDominoId = dominoId;
  resetPlacementSelection();
  renderAll(currentState, currentLegalActions, false);
  setStatus(`Selected future pick #${dominoId}. Now place the current domino.`);
}

async function onBoardCellClick(player, x, y) {
  if (!currentState || !currentTask().requires_placement) return;
  const actor = currentActorBoardIndex();
  if (player !== actor) {
    setStatus(`Place on Player ${actor}'s board.`, true);
    return;
  }
  if (currentState.phase === 'PLACE_AND_SELECT' && selectedPickDominoId === null) {
    setStatus('Select a future domino from the current row before placing.', true);
    return;
  }
  if (!selectedFirstCell || pendingMoveAction) {
    selectedFirstCell = { player, x, y };
    pendingMoveAction = null;
    previewAction = null;
    renderAll(currentState, currentLegalActions, false);
    return;
  }
  if (selectedFirstCell.player !== player) {
    selectedFirstCell = { player, x, y };
    renderAll(currentState, currentLegalActions, false);
    return;
  }
  const dx = Math.abs(selectedFirstCell.x - x);
  const dy = Math.abs(selectedFirstCell.y - y);
  if (dx + dy !== 1) {
    selectedFirstCell = { player, x, y };
    setStatus('Second cell must be adjacent. Restarted with this cell as the first half.', true);
    renderAll(currentState, currentLegalActions, false);
    return;
  }
  const action = findActionForPlacement(selectedFirstCell.x, selectedFirstCell.y, x, y);
  if (!action) {
    setStatus('That placement is not legal for the current domino/pick. Choose highlighted adjacent cells or a different future pick.', true);
    resetPlacementSelection();
    renderAll(currentState, currentLegalActions, false);
    return;
  }
  pendingMoveAction = action;
  previewAction = action;
  const firstLabel = clickedCellLabelForAction(action, selectedFirstCell.x, selectedFirstCell.y);
  const halfMsg = firstLabel ? ` First click matched half ${firstLabel}.` : '';
  setStatus(`Move selected.${halfMsg} Review the board preview, then click Apply selected move.`);
  renderAll(currentState, currentLegalActions, false);
}

async function newGame() {
  try {
    setStatus('Creating game...');
    const seed = Number($('seedInput').value || 0);
    const data = await api('/api/new-game', { method: 'POST', body: JSON.stringify({ seed }) });
    updateSessionMeta(data);
    resetSelectionsForNewState();
    renderAll(data.state, data.legal_actions);
    $('advisor').innerHTML = '';
    setStatus(`New game seed ${seed}`);
  } catch (err) { setStatus(err.message, true); }
}

async function refresh() {
  if (!sessionId) return newGame();
  try {
    const st = await api(`/api/state?session_id=${encodeURIComponent(sessionId)}`);
    updateSessionMeta(st);
    const la = await api(`/api/legal-actions?session_id=${encodeURIComponent(sessionId)}`);
    updateSessionMeta(la);
    renderAll(st.state, la.legal_actions);
    setStatus('Refreshed');
  } catch (err) { setStatus(err.message, true); }
}


async function applyPendingMove() {
  if (!pendingMoveAction) {
    setStatus('No selected move to apply.', true);
    return;
  }
  await applyAction(pendingMoveAction.legal_index);
}

async function applyAction(index) {
  try {
    setStatus(`Applying action ${index}...`);
    const data = await api('/api/apply-action', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, legal_index: index }),
    });
    updateSessionMeta(data);
    resetSelectionsForNewState();
    renderAll(data.state, data.legal_actions);
    $('advisor').innerHTML = '';
    setStatus(`Applied action ${index}`);
  } catch (err) { setStatus(err.message, true); }
}

function advisorRequestPayload() {
  const engine = $('advisorEngineSelect')?.value || 'greedy';
  const topK = Number($('advisorTopKInput')?.value || 8);
  const sims = Number($('advisorSimsInput')?.value || $('nnSimsInput')?.value || 50);
  return {
    session_id: sessionId,
    engine,
    top_k: topK,
    num_simulations: sims,
    nn_sims: sims,
    determinizations: 1,
    temperature: 0,
    checkpoint_path: $('nnCheckpointInput')?.value?.trim() || null,
    device: $('nnDeviceSelect')?.value || 'cpu',
    channels: 64,
    blocks: 6,
    bilinear_dim: 64,
    seed: Number($('seedInput')?.value || 0),
  };
}

function formatAdvisorMeta(r) {
  const parts = [];
  if (r.visit_count !== undefined) parts.push(`visits=${r.visit_count}`);
  if (r.visit_frac !== undefined && r.visit_frac !== null) parts.push(`visit=${(100 * Number(r.visit_frac || 0)).toFixed(1)}%`);
  if (r.q_value !== undefined && r.q_value !== null) parts.push(`q=${Number(r.q_value).toFixed(3)}`);
  if (r.prior !== undefined && r.prior !== null) parts.push(`prior=${(100 * Number(r.prior || 0)).toFixed(1)}%`);
  parts.push(r.action_id);
  return parts.join(' · ');
}

async function recommend() {
  if (!sessionId) return;
  try {
    const payload = advisorRequestPayload();
    setStatus(`Running ${payload.engine} advisor...`);
    const data = await api('/api/recommend', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    const header = `<div class="advisor-summary"><strong>${data.engine || payload.engine}</strong>${data.search_ms !== undefined ? ` · ${data.search_ms} ms` : ''}${data.value !== null && data.value !== undefined ? ` · value ${Number(data.value).toFixed(3)}` : ''}${data.checkpoint_path ? `<br/><small>${data.checkpoint_path}</small>` : ''}</div>`;
    $('advisor').innerHTML = header + (data.recommendations.map(r => {
      const idx = Number(r.legal_index);
      const disabled = Number.isFinite(idx) && idx >= 0 ? '' : 'disabled';
      return `<div class="rec" data-index="${idx}">
        <strong>#${r.rank}</strong> ${r.label}<br/>
        <small>${formatAdvisorMeta(r)}</small>
        <div><button class="apply-rec" data-index="${idx}" ${disabled}>Apply</button></div>
      </div>`;
    }).join('') || '<p>No recommendation.</p>');
    document.querySelectorAll('.rec').forEach(el => {
      el.addEventListener('mouseenter', () => {
        const idx = Number(el.dataset.index);
        previewAction = Number.isFinite(idx) && idx >= 0 ? (currentLegalActions[idx] || null) : null;
        renderBoards(currentState);
      });
      el.addEventListener('mouseleave', () => { previewAction = pendingMoveAction; renderBoards(currentState); });
    });
    document.querySelectorAll('.apply-rec').forEach(btn => btn.addEventListener('click', () => applyAction(Number(btn.dataset.index))));
    setStatus(`Advisor returned ${data.recommendations.length} moves`);
  } catch (err) { setStatus(err.message, true); }
}

async function applyBotMove(mode = null) {
  if (!sessionId || !currentState || currentState.game_over) return;
  const actor = currentActorBoardIndex();
  const chosenMode = mode || currentBotMode();
  if (!chosenMode || chosenMode === 'human') {
    setStatus(`Player ${actor} is set to Human.`, true);
    return;
  }
  try {
    setStatus(`Running ${chosenMode} bot for Player ${actor}...`);
    const data = await api('/api/bot-action', {
      method: 'POST',
      body: JSON.stringify(botRequestPayload(chosenMode, true)),
    });
    updateSessionMeta(data);
    resetSelectionsForNewState();
    renderAll(data.state, data.legal_actions);
    $('advisor').innerHTML = '';
    const bot = data.bot || {};
    const actionLabel = data.action?.label || 'bot action';
    $('botInfo').innerHTML = `<div><strong>${chosenMode}</strong> applied: ${actionLabel}</div><small>${bot.engine || ''}${data.elapsed_ms !== undefined ? ` · ${data.elapsed_ms} ms` : ''}${bot.chosen_visit_frac !== undefined ? ` · visit ${(100 * bot.chosen_visit_frac).toFixed(1)}%` : ''}</small>`;
    setStatus(`${chosenMode} bot moved for Player ${actor}.`);
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function autoPlayToHuman() {
  if (autoPlayRunning) return;
  autoPlayRunning = true;
  const btn = $('autoToHumanBtn');
  if (btn) btn.disabled = true;
  try {
    for (let i = 0; i < 200; i++) {
      if (!currentState || currentState.game_over) break;
      const actor = currentActorBoardIndex();
      const mode = currentBotMode();
      if (!mode || mode === 'human') {
        setStatus(`Stopped at human turn: Player ${actor}.`);
        break;
      }
      await applyBotMove(mode);
      await new Promise(resolve => setTimeout(resolve, 60));
    }
  } finally {
    autoPlayRunning = false;
    if (btn) btn.disabled = false;
  }
}

async function exportState() {
  if (!sessionId) return;
  try {
    const data = await api(`/api/export-state?session_id=${encodeURIComponent(sessionId)}`);
    updateSessionMeta(data);
    $('jsonBox').value = JSON.stringify(data.state, null, 2);
    setStatus('Exported debug state JSON');
  } catch (err) { setStatus(err.message, true); }
}

async function importState() {
  try {
    const parsed = JSON.parse($('jsonBox').value);
    const data = await api('/api/import-state', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, state: parsed }),
    });
    updateSessionMeta(data);
    resetSelectionsForNewState();
    renderAll(data.state, data.legal_actions);
    $('advisor').innerHTML = '';
    setStatus('Imported state');
  } catch (err) { setStatus(err.message, true); }
}


async function undoAction() {
  if (!sessionId) return;
  try {
    setStatus('Undoing last action...');
    const data = await api('/api/undo-action', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, steps: 1 }),
    });
    updateSessionMeta(data);
    resetSelectionsForNewState();
    renderAll(data.state, data.legal_actions);
    $('advisor').innerHTML = '';
    setStatus('Undid last action');
  } catch (err) { setStatus(err.message, true); }
}

async function jumpToStep() {
  if (!sessionId) return;
  try {
    const step = Number($('jumpStepInput').value || 0);
    setStatus(`Jumping to step ${step}...`);
    const data = await api('/api/jump-to-step', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, step }),
    });
    updateSessionMeta(data);
    resetSelectionsForNewState();
    renderAll(data.state, data.legal_actions);
    $('advisor').innerHTML = '';
    setStatus(`Restored step ${data.session.timeline_step}`);
  } catch (err) { setStatus(err.message, true); }
}

$('newGameBtn').addEventListener('click', newGame);
$('refreshBtn').addEventListener('click', refresh);
$('undoBtn').addEventListener('click', undoAction);
$('jumpBtn').addEventListener('click', jumpToStep);
$('exportBtn').addEventListener('click', exportState);
$('importBtn').addEventListener('click', importState);
const recommendButton = $('recommendBtn2') || $('recommendBtn');
if (recommendButton) recommendButton.addEventListener('click', recommend);
$('botMoveBtn').addEventListener('click', () => applyBotMove());
$('autoToHumanBtn').addEventListener('click', autoPlayToHuman);
['botMode0', 'botMode1'].forEach(id => $(id).addEventListener('change', renderBotStatus));
$('clearJsonBtn').addEventListener('click', () => { $('jsonBox').value = ''; });
$('cropToggle').addEventListener('change', (e) => { showCroppedBoards = Boolean(e.target.checked); renderBoards(currentState); });

newGame();
