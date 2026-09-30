"""Bounded turn-solver cache. Caller serializes access with the adapter lock."""
from collections import OrderedDict
from dataclasses import dataclass
import threading
import time


@dataclass
class Entry:
    solver: object
    runners: dict
    deadline: float
    timer: object = None


class TurnCache:
    def __init__(self, lock, max_entries=4, max_positions=100_000, ttl=900):
        self.lock = lock
        self.max_entries = max_entries
        self.max_positions = max_positions
        self.ttl = ttl
        self.entries = OrderedDict()
        self.hits = self.misses = self.builds = 0

    def remove(self, key):
        entry = self.entries.pop(key, None)
        if entry and entry.timer:
            entry.timer.cancel()

    def get(self, key, state):
        now = time.monotonic()
        for old in list(self.entries):
            if self.entries[old].deadline <= now or (old[0] == key[0] and old != key):
                self.remove(old)
        entry = self.entries.get(key)
        if entry:
            # Runners never disappear or move backwards within an ongoing turn.
            forward = all(state.runners.get(c, -1) >= p for c, p in entry.runners.items())
            if forward:
                try:
                    entry.solver.value(state)  # verifies the queried subtree exists
                except KeyError:
                    forward = False
            if forward:
                self.entries.move_to_end(key)
                entry.runners = dict(state.runners)
                self.hits += 1
                return entry.solver
            self.remove(key)
        self.misses += 1
        return None

    def put(self, key, state, solver):
        self.builds += 1
        if not key or self.max_entries <= 0 or solver.num_positions > self.max_positions:
            return
        self.remove(key)
        while self.entries and (len(self.entries) >= self.max_entries or
                sum(e.solver.num_positions for e in self.entries.values()) + solver.num_positions > self.max_positions):
            self.remove(next(iter(self.entries)))
        entry = Entry(solver, dict(state.runners), time.monotonic() + self.ttl)
        self.entries[key] = entry
        # Expire even when no further requests arrive. Fixed lifetime avoids
        # retaining a solver indefinitely for an abandoned or very long turn.
        def expire():
            with self.lock:
                if self.entries.get(key) is entry:
                    self.remove(key)
        entry.timer = threading.Timer(self.ttl, expire)
        entry.timer.daemon = True
        entry.timer.start()

    def close(self):
        with self.lock:
            for key in list(self.entries):
                self.remove(key)

    def stats(self):
        with self.lock:
            return {"hits": self.hits, "misses": self.misses, "builds": self.builds,
                    "entries": len(self.entries),
                    "positions": sum(e.solver.num_positions for e in self.entries.values()),
                    "max_entries": self.max_entries, "max_positions": self.max_positions,
                    "ttl_seconds": self.ttl}
