"""v2.7 priority scheduler (PRIORITY_SCHEDULER): shared compute instead of an equal wall-clock budget per game.

All games of a run are started at once (one thread each) and wait here; at most `slots` of them play at a time.
A game keeps its slot until its context is trimmed (its cached prompt prefix is gone at that point anyway) or until
it has held the slot for `stretch_cap_s`; it then re-enters the queue and the free slot goes to the waiting game with
the highest priority (the same game again if it is still the best). If nobody is waiting, a game simply keeps playing.

Order of the queue (deterministic):
  1. games that have never played, in dispatch order (every game gets one first stretch before any game resumes);
  2. games that have played, by
         priority = L / (N (N+1) / 2) * 0.5 ** ((output tokens on the current level / tau) ** 2)
     L = level being played (1-based), N = number of levels of the game: the score weight of the level the game is
     on, halved after `tau` output tokens spent on that level without completing it (and ~1/16 after 2 tau);
  3. ties by dispatch order.
Every game shares one deadline (the end of the run). A game waiting for a slot when the deadline passes is woken
and finishes through the normal deadline path (replay written, results recorded).

Nothing here is visible to the model. With the flag off this module is never configured and nothing changes.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

_SCHED: "PriorityScheduler | None" = None
_TLS = threading.local()


def configure(slots: int, deadline: float | None, keys: list[Any], *, tau_tokens: float, stretch_cap_s: float,
              default_levels: int = 8, log: Callable[[str], None] | None = None) -> "PriorityScheduler":
    """Create the run's scheduler (call once, before the game threads start)."""
    global _SCHED
    _SCHED = PriorityScheduler(slots, deadline, keys, tau_tokens=tau_tokens, stretch_cap_s=stretch_cap_s,
                               default_levels=default_levels, log=log)
    return _SCHED


def get() -> "PriorityScheduler | None":
    return _SCHED


def reset() -> None:
    global _SCHED
    _SCHED = None


def set_thread_key(key: Any) -> None:
    """The runner names the game run played by this thread (e.g. (pass_no, game_id)); default: the game id."""
    _TLS.key = key


def thread_key(default: Any) -> Any:
    return getattr(_TLS, "key", None) or default


class PriorityScheduler:
    def __init__(self, slots: int, deadline: float | None, keys: list[Any], *, tau_tokens: float,
                 stretch_cap_s: float, default_levels: int = 8, log: Callable[[str], None] | None = None):
        self.slots = max(1, int(slots))
        self.deadline = deadline
        self.tau = max(1.0, float(tau_tokens))
        self.stretch_cap_s = max(60.0, float(stretch_cap_s))
        self.default_levels = max(1, int(default_levels))
        self._log = log or (lambda s: print(s, flush=True))
        self._cond = threading.Condition()
        self._free = self.slots
        self._order = {k: i for i, k in enumerate(keys)}
        self._waiting: dict[Any, tuple] = {}
        self._granted: set[Any] = set()
        self._stats: dict[Any, dict[str, Any]] = {}
        self.t0 = time.time()

    # ---- priority -------------------------------------------------------------------------------------------
    def priority(self, level: int, n_levels: int | None, tokens_on_level: float) -> float:
        n = int(n_levels) if n_levels and int(n_levels) > 0 else self.default_levels
        lvl = min(max(1, int(level or 1)), n)
        return lvl / (n * (n + 1) / 2.0) * 0.5 ** ((max(0.0, float(tokens_on_level)) / self.tau) ** 2)

    # ---- internals ------------------------------------------------------------------------------------------
    def _st(self, key: Any) -> dict[str, Any]:
        if key not in self._order:
            self._order[key] = len(self._order)
        return self._stats.setdefault(key, {"active_s": 0.0, "stretches": 0, "granted_at": None, "waited_s": 0.0,
                                            "last_priority": None, "yields": 0})

    def _name(self, key: Any) -> str:
        if isinstance(key, tuple):
            gid = str(key[-1]).split("-", 1)[0]
            return f"{gid}#{key[0]}" if len(key) > 1 and key[0] not in (None, 1) else gid
        return str(key).split("-", 1)[0]

    def _mins(self) -> str:
        return f"{(time.time() - self.t0) / 60:6.1f}m"

    def _pump(self) -> None:
        while self._free > 0 and self._waiting:
            key = max(self._waiting, key=lambda k: self._waiting[k])
            del self._waiting[key]
            self._granted.add(key)
            self._free -= 1
        self._cond.notify_all()

    def _wait_for_slot(self, key: Any, band: int, value: float) -> bool:
        """Queue `key` and block until it holds a slot (True) or the deadline passes (False). Caller holds the lock."""
        self._waiting[key] = (band, float(value), -self._order[key])
        self._pump()
        t_wait = time.time()
        immediate = key in self._granted   # still the best game: it simply keeps playing (not a new stretch)
        while key not in self._granted:
            remaining = (self.deadline - time.time()) if self.deadline else None
            if remaining is not None and remaining <= 0:
                self._waiting.pop(key, None)
                self._st(key)["waited_s"] += time.time() - t_wait
                return False
            self._cond.wait(timeout=5.0 if remaining is None else max(0.05, min(5.0, remaining)))
        st = self._st(key)
        st["waited_s"] += time.time() - t_wait
        st["granted_at"] = time.time()
        if not immediate or band == 2:
            st["stretches"] += 1
        return True

    def _release_locked(self, key: Any) -> None:
        st = self._st(key)
        if key in self._granted:
            self._granted.discard(key)
            self._free += 1
            if st["granted_at"] is not None:
                st["active_s"] += time.time() - st["granted_at"]
            st["granted_at"] = None
        self._waiting.pop(key, None)

    # ---- API used by the game loop ----------------------------------------------------------------------------
    def start(self, key: Any) -> bool:
        """Block until this game may start (never-played games go first, in dispatch order)."""
        with self._cond:
            self._st(key)
            t = time.time()
            ok = self._wait_for_slot(key, 2, 0.0)
            waiting = len(self._waiting)
        if ok:
            self._log(f"[sched {self._mins()}] START  {self._name(key):6s} (waited {(time.time() - t) / 60:.1f} min; "
                      f"{waiting} waiting)")
        return ok

    def should_yield(self, key: Any, trimmed: bool) -> str | None:
        """Reason to give the slot up now ("trim" / "cap"), or None. Only when another game is waiting."""
        with self._cond:
            st = self._st(key)
            if key not in self._granted or not self._waiting:
                return None
            if trimmed:
                return "trim"
            if st["granted_at"] is not None and time.time() - st["granted_at"] >= self.stretch_cap_s:
                return "cap"
            return None

    def handover(self, key: Any, value: float, reason: str, info: str = "") -> bool:
        """Give the slot up and queue again with priority `value`. Returns False if the deadline passed while waiting."""
        with self._cond:
            st = self._st(key)
            held = (time.time() - st["granted_at"]) if st["granted_at"] is not None else 0.0
            st["last_priority"] = float(value)
            self._release_locked(key)
            t = time.time()
            n_before = st["stretches"]
            ok = self._wait_for_slot(key, 1, value)
            waited = time.time() - t
            paused = (st["stretches"] > n_before) or not ok
            if paused:
                st["yields"] += 1
            n_wait = len(self._waiting)
        if paused:
            self._log(f"[sched {self._mins()}] {'RESUME' if ok else 'ENDED '} {self._name(key):6s} after waiting "
                      f"{waited / 60:.1f} min (paused at {reason} after a {held / 60:.1f}-min stretch; {info}; "
                      f"priority {value:.4f}; {n_wait} waiting)")
        return ok

    def finish(self, key: Any) -> None:
        """The game ended (won, deadline, error): free its slot for good."""
        with self._cond:
            self._release_locked(key)
            self._pump()

    def active_seconds(self, key: Any) -> float:
        with self._cond:
            st = self._st(key)
            return st["active_s"] + ((time.time() - st["granted_at"]) if st["granted_at"] is not None else 0.0)

    def stats(self, key: Any) -> dict[str, Any]:
        with self._cond:
            st = dict(self._st(key))
        st["active_s"] = self.active_seconds(key)
        st.pop("granted_at", None)
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in st.items()}

    def describe(self) -> str:
        return (f"priority scheduler: {self.slots} slots, {len(self._order)} game runs, tau {self.tau / 1000:.0f}k output "
                f"tokens on a level, stretch cap {self.stretch_cap_s / 60:.0f} min, run ends in "
                + (f"{(self.deadline - time.time()) / 60:.1f} min" if self.deadline else "never"))

    def summary(self) -> str:
        with self._cond:
            keys = sorted(self._order, key=lambda k: self._order[k])
        lines = ["[sched] active minutes per game run (stretches, times paused for another game):"]
        for k in keys:
            s = self.stats(k)
            lines.append(f"  {self._name(k):8s} {s['active_s'] / 60:6.1f} min  ({s['stretches']} stretches, {s['yields']} pauses, "
                         f"waited {s['waited_s'] / 60:.1f} min)")
        return "\n".join(lines)
