// Observe ALL phases, including other players and bust/stop transitions.
function createCantStopTurnTracker(session) {
  let table = null, actor = null, sequence = 0, closed = true;
  return observation => {
    const decision = ["diceChoice", "continueChoice"].includes(observation.phase);
    if (table !== observation.table_id || actor !== observation.active_player || (closed && decision)) {
      table = observation.table_id; actor = observation.active_player; sequence++;
      closed = false;
    }
    if (["endTurn", "failConfirm", "saveProgress", "nextPlayer", "gameEnd"].includes(observation.phase))
      closed = true;
    return session + ":" + sequence;
  };
}
