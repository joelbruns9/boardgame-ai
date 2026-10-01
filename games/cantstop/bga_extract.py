"""Normalize raw BGA markers using the existing extension's top-distance convention.

BGA cantstop.js version 260212-1713 labels 1 as Jump and 2 as Forced movement.
Flag 0 is standard (also observed in table 923012474). Reject unsupported flags.
"""
from .engine import COLUMNS, COLUMN_HEIGHTS


def blocking_from_bga(flag):
    if type(flag) not in (int, str):
        raise ValueError("missing or invalid BGA movement variant")
    if flag in (0, "0"):
        return False
    if flag in (2, "2"):
        return True
    if flag in (1, "1"):
        raise ValueError("BGA Jump variant is not supported by this model")
    raise ValueError(f"unknown BGA movement variant: {flag!r}")


def normalize_capture(raw):
    blocking = blocking_from_bga(raw.get("movement_variant_raw"))
    if "blocking" in raw and (type(raw["blocking"]) is not bool or raw["blocking"] != blocking):
        raise ValueError("blocking setting conflicts with BGA movement variant")
    order = [str(p) for p in raw["playerorder"]]
    if not 2 <= len(order) <= 4 or len(set(order)) != len(order):
        raise ValueError("invalid player order")
    players = raw["players"]
    colors = {str(players[p]["color"]).lower().lstrip("#"): i for i, p in enumerate(order)}
    if len(colors) != len(order) or "000000" in colors:
        raise ValueError("ambiguous player colors")
    progress = [{} for _ in order]
    claimed = [[] for _ in order]
    runners = {}
    seen = set()
    for marker in raw["markers"]:
        c, h = marker["column"], marker["height"]
        if type(c) is not int or c not in COLUMNS or type(h) is not int or not 0 <= h < COLUMN_HEIGHTS[c]:
            raise ValueError("invalid BGA marker coordinate")
        color = str(marker["color"]).lower().lstrip("#")
        key = (color, c)
        if key in seen:
            raise ValueError("duplicate marker; board animation may be in progress")
        seen.add(key)
        if color == "000000":
            runners[str(c)] = COLUMN_HEIGHTS[c] - h
        elif color in colors:
            p = colors[color]
            if h == 0:
                claimed[p].append(c)
            else:
                progress[p][str(c)] = COLUMN_HEIGHTS[c] - h
        else:
            raise ValueError("unknown marker color")
    # BGA can retain inert opponent markers on a column that has been claimed.
    for cols in claimed:
        for c in cols:
            for row in progress:
                row.pop(str(c), None)
    # gamedatas.players[*].score can remain at its initial value after claims
    # (live table 923128580). DOM claim markers are the source of truth; the
    # metadata score is diagnostic only, not a freshness/reachability check.
    active = str(raw["active_player"])
    if active not in order:
        raise ValueError("active player not in player order")
    return {"rules": {"num_players": len(order),
                       "columns_to_win": raw["required_column_count"],
                       "blocking": blocking},
            "active_player": order.index(active), "phase": raw["phase"],
            "progress": progress, "claimed": claimed, "runners": runners,
            "dice": raw.get("dice", [])}
