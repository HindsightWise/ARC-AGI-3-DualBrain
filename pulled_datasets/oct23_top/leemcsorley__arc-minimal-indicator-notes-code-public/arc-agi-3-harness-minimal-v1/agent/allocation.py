"""Smart time allocation across concurrent game workers.

When SMART_ALLOCATION is enabled (off by default), workers report their
AI-estimated chance of finishing the current level and median turns needed.
The allocator uses these to compute expected-score-per-second for each game
and dynamically extends/shrinks deadlines so that the highest-EV games get
the most remaining compute.

Integration:
  - allocation.py  – this module (TimeAllocator + estimate tool spec/parsing)
  - llm.py         – adds estimate_progress tool to tool specs
  - agent.py       – workers call allocator.report_progress / get_deadline
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from time import time as wall_time
from typing import Any


import os as _os
SMART_ALLOCATION: bool = _os.environ.get("SMART_ALLOCATION", "0").lower() in ("1", "true", "yes")


# ── Estimate tool spec (added alongside other tool specs when enabled) ──────

ESTIMATE_TOOL_SPEC: dict[str, Any] = {
    "name": "estimate_progress",
    "description": (
        "Report your estimated chance of finishing the CURRENT level and "
        "how many more turns you think it will take. Call this EVERY turn "
        "alongside your other tool calls. Be honest — overconfidence wastes "
        "time on hopeless games, underconfidence starves promising ones."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "finish_percent": {
                "type": "integer",
                "description": (
                    "Your estimated probability (0-100) that you will eventually "
                    "finish the current level given enough turns."
                ),
                "minimum": 0,
                "maximum": 100,
            },
            "median_turns": {
                "type": "integer",
                "description": (
                    "Your median estimate of how many MORE turns you need to "
                    "finish the current level (assuming you do finish it)."
                ),
                "minimum": 1,
            },
        },
        "required": ["finish_percent", "median_turns"],
        "additionalProperties": False,
    },
}


def parse_estimate_from_commands(commands: list[Any]) -> tuple[int | None, int | None]:
    """Extract (finish_percent, median_turns) from parsed commands list.

    Returns (None, None) if no estimate_progress command was found.
    Each command is expected to have .kind and .content attributes.
    """
    for cmd in commands:
        if cmd.kind != "estimate_progress":
            continue
        raw = cmd.content.strip()
        if not raw:
            return None, None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None, None
        fp = data.get("finish_percent")
        mt = data.get("median_turns")
        if isinstance(fp, (int, float)):
            fp = max(0, min(100, int(fp)))
        else:
            fp = None
        if isinstance(mt, (int, float)):
            mt = max(1, int(mt))
        else:
            mt = None
        return fp, mt
    return None, None


# ── Per-game state ──────────────────────────────────────────────────────────

@dataclass
class _GameState:
    """Mutable state tracked per game by the allocator."""
    game_id: str
    initial_budget: float  # seconds originally allocated
    start_time: float = 0.0
    current_level: int = 0
    levels_completed: int = 0
    messages: int = 0
    actions_taken: int = 0
    finish_percent: int = 50  # AI-estimated chance to finish current level
    median_turns: int = 20   # AI-estimated turns remaining
    finished: bool = False
    finish_kind: str = ""    # "success", "game_over", "deadline", etc.
    avg_seconds_per_turn: float = 0.0  # rolling estimate


# ── Allocator ───────────────────────────────────────────────────────────────

class TimeAllocator:
    """Dynamically distributes a fixed time budget across concurrent game workers.

    Thread-safe — multiple workers call methods concurrently.

    Scoring model:
        Each unfinished game has an *expected score rate*:
            EV = (finish_percent / 100) * score_gain / estimated_seconds_to_finish
        Games with higher EV get more of the remaining budget.
    """

    def __init__(
        self,
        global_deadline: float,
        game_ids: list[str],
        num_workers: int,
        *,
        score_per_level: float = 1.0,
        min_budget: float = 120.0,
    ) -> None:
        self._lock = threading.Lock()
        self._global_deadline = global_deadline
        self._num_workers = num_workers
        self._score_per_level = score_per_level
        self._min_budget = min_budget

        now = wall_time()
        total_available = max(0.0, global_deadline - now)
        equal_budget = max(min_budget, total_available * num_workers / len(game_ids)) if game_ids else 0.0

        self._games: dict[str, _GameState] = {}
        for gid in game_ids:
            self._games[gid] = _GameState(
                game_id=gid,
                initial_budget=equal_budget,
                start_time=now,
            )

        # Deadline overrides computed by the scoring model.
        # Maps game_id -> absolute timestamp.  If absent, fall back to
        # min(start + initial_budget, global_deadline).
        self._deadline_overrides: dict[str, float] = {}

    # ── Worker API ──────────────────────────────────────────────────────

    def get_deadline(self, game_id: str) -> float:
        """Return the current deadline for *game_id*.  Called each turn."""
        with self._lock:
            gs = self._games.get(game_id)
            if gs is None:
                return self._global_deadline

            override = self._deadline_overrides.get(game_id)
            if override is not None:
                return min(override, self._global_deadline)

            default = min(gs.start_time + gs.initial_budget, self._global_deadline)
            return default

    def report_progress(
        self,
        game_id: str,
        *,
        current_level: int,
        levels_completed: int,
        messages: int,
        actions_taken: int,
        finish_percent: int | None = None,
        median_turns: int | None = None,
    ) -> None:
        """Worker reports its current state.  Triggers reallocation."""
        with self._lock:
            gs = self._games.get(game_id)
            if gs is None or gs.finished:
                return

            gs.current_level = current_level
            gs.levels_completed = levels_completed
            gs.messages = messages
            gs.actions_taken = actions_taken

            if finish_percent is not None:
                gs.finish_percent = max(0, min(100, finish_percent))
            if median_turns is not None:
                gs.median_turns = max(1, median_turns)

            elapsed = wall_time() - gs.start_time
            if messages > 0:
                gs.avg_seconds_per_turn = elapsed / messages

            self._reallocate()

    def report_finished(
        self,
        game_id: str,
        *,
        kind: str,
        levels_completed: int,
    ) -> None:
        """Worker finished — free its remaining budget for reallocation."""
        with self._lock:
            gs = self._games.get(game_id)
            if gs is None:
                return
            gs.finished = True
            gs.finish_kind = kind
            gs.levels_completed = levels_completed
            self._reallocate()

    # ── Internal scoring & reallocation ─────────────────────────────────

    def _reallocate(self) -> None:
        """Redistribute remaining global time among unfinished games.

        Must be called with self._lock held.

        Strategy:
            1. Compute EV-rate for each unfinished game
            2. Distribute remaining time proportionally to EV-rate
            3. Clamp each game to [min_budget_remaining, global_deadline]
        """
        now = wall_time()
        remaining_global = max(0.0, self._global_deadline - now)
        if remaining_global <= 0:
            return

        active: list[tuple[str, _GameState]] = [
            (gid, gs) for gid, gs in self._games.items() if not gs.finished
        ]
        if not active:
            return

        # Compute EV-rate for each active game
        ev_rates: dict[str, float] = {}
        for gid, gs in active:
            ev_rates[gid] = self._ev_rate(gs)

        total_ev = sum(ev_rates.values())

        if total_ev <= 0:
            # No game has positive EV — split equally
            equal_share = remaining_global * self._num_workers / len(active)
            for gid, gs in active:
                self._deadline_overrides[gid] = now + max(self._min_budget_remaining(gs), equal_share)
            return

        # Distribute proportionally to EV rate
        for gid, gs in active:
            fraction = ev_rates[gid] / total_ev
            share = remaining_global * self._num_workers * fraction
            share = max(share, self._min_budget_remaining(gs))
            self._deadline_overrides[gid] = now + share

    def _ev_rate(self, gs: _GameState) -> float:
        """Expected score rate: EV / estimated_seconds_to_finish.

        Higher = more worth investing time in.
        """
        prob = gs.finish_percent / 100.0
        score_gain = self._score_per_level

        # Estimate seconds to finish current level
        spt = gs.avg_seconds_per_turn if gs.avg_seconds_per_turn > 0 else 30.0
        seconds_to_finish = gs.median_turns * spt

        if seconds_to_finish <= 0:
            seconds_to_finish = 1.0

        ev = prob * score_gain
        ev_rate = ev / seconds_to_finish

        return max(ev_rate, 0.0)

    def _min_budget_remaining(self, gs: _GameState) -> float:
        """Minimum remaining seconds — don't yank a game below this."""
        elapsed = wall_time() - gs.start_time
        total_min = max(self._min_budget, 60.0)
        return max(0.0, total_min - elapsed)

    # ── Diagnostics ─────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of current allocations."""
        with self._lock:
            now = wall_time()
            lines = ["[Allocator] Game status:"]
            for gid, gs in sorted(self._games.items()):
                status = gs.finish_kind if gs.finished else "active"
                dl = self._deadline_overrides.get(gid)
                if dl is None:
                    dl = min(gs.start_time + gs.initial_budget, self._global_deadline)
                remaining = max(0.0, dl - now)
                ev = self._ev_rate(gs) if not gs.finished else 0.0
                lines.append(
                    f"  {gid[:8]:8s} L{gs.levels_completed} "
                    f"p={gs.finish_percent:3d}% t~{gs.median_turns:3d} "
                    f"EV={ev:.4f} "
                    f"rem={remaining/60:.1f}m "
                    f"[{status}]"
                )
            return "\n".join(lines)
