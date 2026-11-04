"""Level-weighted dynamic time allocation. Enabled on Kaggle with LEVEL_WEIGHTED_ALLOCATION=1.

Drop-in replacement for allocation.TimeAllocator with the same worker API
(get_deadline / report_progress / report_finished / summary), no LLM estimate
tool needed.

Rule: a game that has completed L levels holds an allowance of
    a * w(L),   w(L) = sum(i ** p for i in 1..L+1)
so every level-up extends its deadline (p=1: level k gets k units; p=0: every
level gets one unit).  The unit `a` (seconds) is recomputed continuously so that
the whole queue of games exactly consumes the remaining slot-time:
    a * (sum_running w_g + n_unstarted * w_bar) = remaining_global * workers + sum_running elapsed_g
where w_bar is the expected final w of a not-yet-started game, estimated from
finished games (blended with a prior).  When no games remain unstarted, every
running game simply gets the global deadline (a cut slot could not be reused).
"""
from __future__ import annotations

import os
import threading
from time import time as wall_time


class LevelWeightedAllocator:
    def __init__(self, global_deadline: float, game_ids: list[str], num_workers: int, *,
                 level_power: float | None = None, prior_wbar: float | None = None,
                 prior_weight: float | None = None, min_minutes: float | None = None,
                 max_share: float | None = None, log_every_s: float | None = None,
                 now=None) -> None:
        env = os.environ.get
        self.p = float(env("ALLOC_LEVEL_POWER", "1.0")) if level_power is None else level_power
        self.prior_wbar = float(env("ALLOC_PRIOR_WBAR", "3.5")) if prior_wbar is None else prior_wbar
        self.prior_weight = float(env("ALLOC_PRIOR_WEIGHT", "12")) if prior_weight is None else prior_weight
        self.min_share = float(env("ALLOC_MIN_SHARE", "0.55")) if min_minutes is None else None
        self._min_minutes = min_minutes
        self.max_share = float(env("ALLOC_MAX_SHARE", "4.0")) if max_share is None else max_share
        self.log_every_s = float(env("ALLOC_LOG_EVERY_S", "300")) if log_every_s is None else log_every_s
        self.now = now or wall_time  # overridable clock (simulation)
        self._lock = threading.Lock()
        self._deadline = float(global_deadline)
        self._workers = max(1, int(num_workers))
        self._games = {g: dict(start=None, levels=0, finished=False, kind="", w_final=None) for g in game_ids}
        self._equal_share = max(1.0, (self._deadline - self.now()) * self._workers / max(1, len(game_ids)))
        # floor on any game's allowance: a fraction of the uniform per-game share (or explicit minutes)
        self.min_s = (60.0 * self._min_minutes) if self._min_minutes is not None else self.min_share * self._equal_share
        self._a = self._equal_share / self.prior_wbar
        self._last_log = 0.0
        self._n_realloc = 0

    # ── weights ──────────────────────────────────────────────────────────
    def w(self, levels_completed: int) -> float:
        return float(sum(i ** self.p for i in range(1, int(levels_completed) + 2)))

    def _wbar(self) -> float:
        fin = [g["w_final"] for g in self._games.values() if g["finished"] and g["w_final"] is not None]
        return (self.prior_weight * self.prior_wbar + sum(fin)) / (self.prior_weight + len(fin))

    # ── core: solve for the unit `a` (lock held) ─────────────────────────
    def _recompute(self) -> None:
        now = self.now()
        remaining = max(0.0, self._deadline - now)
        running = [g for g in self._games.values() if g["start"] is not None and not g["finished"]]
        unstarted = sum(1 for g in self._games.values() if g["start"] is None)
        supply = remaining * self._workers + sum(now - g["start"] for g in running)
        ws = [self.w(g["levels"]) for g in running] + [self._wbar()] * unstarted
        if ws and supply > 0:
            a = self._a
            for _ in range(4):  # fixed point: floors/caps in units of `a`
                lo, hi = self.min_s / a, self.max_share * self._equal_share / a
                demand = sum(min(max(w, lo), hi) for w in ws)
                a = supply / demand if demand > 0 else a
            self._a = a
        self._n_realloc += 1
        if now - self._last_log >= self.log_every_s:
            self._last_log = now
            fin = sum(1 for g in self._games.values() if g["finished"])
            lv = sorted((g["levels"] for g in running), reverse=True)
            print(f"[ALLOC] unit={self._a/60:.1f} min  w_bar={self._wbar():.2f}  running={len(running)} "
                  f"unstarted={unstarted} finished={fin}  remaining={remaining/60:.0f} min  levels={lv}", flush=True)

    def _deadline_for(self, g: dict) -> float:
        unstarted = any(x["start"] is None for x in self._games.values())
        if not unstarted:
            return self._deadline  # tail: nobody could use a freed slot
        allow = self._a * self.w(g["levels"])
        allow = min(max(allow, self.min_s), self.max_share * self._equal_share)
        return min(g["start"] + allow, self._deadline)

    # ── worker API (same as TimeAllocator) ───────────────────────────────
    def get_deadline(self, game_id: str) -> float:
        with self._lock:
            g = self._games.get(game_id)
            if g is None:
                return self._deadline
            if g["start"] is None:
                g["start"] = self.now()
            self._recompute()
            return self._deadline_for(g)

    def report_progress(self, game_id: str, *, current_level: int, levels_completed: int,
                        messages: int = 0, actions_taken: int = 0,
                        finish_percent=None, median_turns=None) -> None:
        with self._lock:
            g = self._games.get(game_id)
            if g is None or g["finished"]:
                return
            if g["start"] is None:
                g["start"] = self.now()
            new = max(0, int(levels_completed))
            if new > g["levels"]:
                g["levels"] = new
                self._recompute()
                print(f"[ALLOC] {game_id.split('-')[0]} completed level {new}: allowance now "
                      f"{self._a * self.w(new) / 60:.0f} min (unit {self._a/60:.1f} min)", flush=True)

    def report_finished(self, game_id: str, *, kind: str, levels_completed: int) -> None:
        with self._lock:
            g = self._games.get(game_id)
            if g is None or g["finished"]:
                return
            g["finished"] = True
            g["kind"] = kind
            g["levels"] = max(g["levels"], int(levels_completed))
            g["w_final"] = self.w(g["levels"])
            self._recompute()

    def describe(self) -> str:
        return (f"p={self.p}, min_share={self.min_share if self._min_minutes is None else None}, "
                f"floor={self.min_s/60:.1f} min, max_share={self.max_share}, prior_wbar={self.prior_wbar} "
                f"(prior_weight {self.prior_weight:.0f}), equal_share={self._equal_share/60:.1f} min, "
                f"initial unit={self._a/60:.1f} min, games={len(self._games)}, workers={self._workers}")

    def summary(self) -> str:
        with self._lock:
            fin = [g for g in self._games.values() if g["finished"]]
            lv = sum(g["levels"] for g in fin)
            return (f"[ALLOC] done: {len(fin)} finished, {lv} levels, final unit {self._a/60:.1f} min, "
                    f"w_bar {self._wbar():.2f}, reallocations {self._n_realloc}")
