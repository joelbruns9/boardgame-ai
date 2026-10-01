// Match the entire validated position; object insertion order is irrelevant.
function sameAdvisorState(a, b) {
  function canonical(value) {
    if (Array.isArray(value)) return value.map(canonical);
    if (value && typeof value === "object")
      return Object.fromEntries(Object.keys(value).sort().map(k => [k, canonical(value[k])]));
    return value;
  }
  return JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
}

// Never reuse advice across tables or for a changed board/player/rule set.
function findAdvisorContinuation(result, sourceTable, table, state) {
  if (!result || sourceTable !== table || state.phase !== "continueChoice") return null;
  return (result.recommendations || []).find(r =>
    r.fields?.after_move && sameAdvisorState(r.fields.after_move, state)) || null;
}
