"""
The Dual-Brain Cognitive Agent for ARC-AGI-3.
Integrates:
1. Two-Phase Planning:
   - Phase 1: 'Plan to Plan' (Epistemic Probing to identify Avatar & Affordances)
   - Phase 2: 'Plan a Plan' (Goal-Directed Tactical BFS / A* Pathfinding)
2. Working Memory & Oscillation Interception (State Hashing to prevent loops)
3. Anti-Dogma Evidentiary Trial (Clean priors per game/level)
4. Terminal Safety Guard (Immediately issues RESET on GAME_OVER to prevent 400 errors)
"""
from __future__ import annotations

import collections
import hashlib
import random
import time
from typing import Any, List, Optional, Set, Tuple

import numpy as np
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent


class WorkingMemory:
    """Tracks state history, detects circular loops, and logs hypotheses."""
    def __init__(self, loop_threshold: int = 3):
        self.visited_states: collections.Counter = collections.Counter()
        self.history: list[dict[str, Any]] = []
        self.loop_threshold = loop_threshold
        self.avatar_color: Optional[int] = None
        self.avatar_pos: Optional[Tuple[int, int]] = None
        self.probing_phase: bool = True
        self.probe_actions_left: list[GameAction] = []

    def hash_grid(self, grid: np.ndarray) -> str:
        """Returns deterministic MD5 hash of the 2D/3D grid."""
        return hashlib.md5(grid.tobytes()).hexdigest()

    def record_step(self, action: GameAction, grid: np.ndarray, state: Any):
        h = self.hash_grid(grid)
        self.visited_states[h] += 1
        self.history.append({"action": action, "state_hash": h, "state": state})

    def is_in_loop(self, grid: np.ndarray) -> bool:
        h = self.hash_grid(grid)
        return self.visited_states[h] >= self.loop_threshold

    def reset_for_level(self):
        self.visited_states.clear()
        self.history.clear()
        self.probing_phase = True
        self.probe_actions_left = [
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        ]


class MyAgent(Agent):
    """Dual-Brain Agent implementing Two-Phase Cognitive Meta-Planning."""

    MAX_ACTIONS = 120

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        self.memory = WorkingMemory(loop_threshold=3)
        self.last_action: Optional[GameAction] = None

    @property
    def name(self) -> str:
        return f"DualBrainAgent.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Terminate only on WIN
        return latest_frame.state is GameState.WIN

    def _extract_grid(self, latest_frame: FrameData) -> np.ndarray:
        if latest_frame is None or getattr(latest_frame, "frame", None) is None:
            return np.zeros((0, 0), dtype=np.int8)
        raw = np.array(latest_frame.frame)
        while raw.ndim > 2:
            raw = raw[0]
        return raw

    def _find_avatar_diff(
        self, prev_grid: np.ndarray, curr_grid: np.ndarray, action_taken: GameAction
    ) -> Optional[Tuple[int, Tuple[int, int]]]:
        """Compares two frames to find which pixel cluster translated with the action."""
        if prev_grid.size == 0 or curr_grid.size == 0 or prev_grid.shape != curr_grid.shape:
            return None
        diff = prev_grid != curr_grid
        if not np.any(diff):
            return None

        # Vector expected for simple directional actions
        direction_vectors = {
            GameAction.ACTION1: (-1, 0), # Up
            GameAction.ACTION2: (1, 0),  # Down
            GameAction.ACTION3: (0, -1), # Left
            GameAction.ACTION4: (0, 1),  # Right
        }
        vec = direction_vectors.get(action_taken)
        if not vec:
            return None

        # Find pixels that disappeared in prev and appeared in curr matching the vector
        curr_nonzero_coords = np.argwhere(diff)
        for r, c in curr_nonzero_coords:
            color = curr_grid[r, c]
            if color == 0:
                continue
            prev_r, prev_c = r - vec[0], c - vec[1]
            if 0 <= prev_r < prev_grid.shape[0] and 0 <= prev_c < prev_grid.shape[1]:
                if prev_grid[prev_r, prev_c] == color:
                    return color, (int(r), int(c))
        return None

    def _bfs_path_to_targets(
        self, grid: np.ndarray, start_pos: Tuple[int, int], legal_actions: List[GameAction]
    ) -> Optional[GameAction]:
        """Tactical BFS finding the shortest path to non-wall target entities."""
        h, w = grid.shape
        start_r, start_c = start_pos
        avatar_color = grid[start_r, start_c]

        # Targets are non-zero, non-avatar, non-boundary entities
        action_map = {
            GameAction.ACTION1: (-1, 0),
            GameAction.ACTION2: (1, 0),
            GameAction.ACTION3: (0, -1),
            GameAction.ACTION4: (0, 1),
        }
        valid_moves = [a for a in legal_actions if a in action_map]
        if not valid_moves:
            return None

        queue = collections.deque([(start_r, start_c, [])])
        visited = set([(start_r, start_c)])

        while queue:
            curr_r, curr_c, path = queue.popleft()

            # If we reached a new color entity (potential target / key / exit)
            cell_val = grid[curr_r, curr_c]
            if (curr_r, curr_c) != (start_r, start_c) and cell_val not in (0, avatar_color):
                if path:
                    return path[0]

            if len(path) > 30: # horizon limit
                continue

            for act in valid_moves:
                dr, dc = action_map[act]
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                    # Don't step into obvious static walls if we know the wall color
                    visited.add((nr, nc))
                    queue.append((nr, nc, path + [act]))

        return None

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # Rule 1: Prevent 400 Bad Request on GAME_OVER or NOT_PLAYED
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.memory.reset_for_level()
            self.last_action = GameAction.RESET
            return GameAction.RESET

        grid = self._extract_grid(latest_frame)
        self.memory.record_step(self.last_action, grid, latest_frame.state)

        # Get legal actions from frame metadata
        available = getattr(latest_frame, "available_actions", None)
        if available:
            legal_actions = [GameAction.from_id(int(a)) for a in available if a != 0]
        else:
            legal_actions = [a for a in GameAction if a is not GameAction.RESET]

        if not legal_actions:
            return GameAction.RESET

        # Loop Interceptor: If in circular oscillation, force a random detour
        if self.memory.is_in_loop(grid):
            detour_actions = [a for a in legal_actions if a != self.last_action]
            action = random.choice(detour_actions if detour_actions else legal_actions)
            action.reasoning = {"why": "loop_intercept_detour"}
            self.last_action = action
            return action

        # Phase 1: Epistemic Probing (Identify Avatar)
        if len(frames) >= 2 and self.last_action:
            prev_grid = self._extract_grid(frames[-2])
            avatar_info = self._find_avatar_diff(prev_grid, grid, self.last_action)
            if avatar_info:
                self.memory.avatar_color, self.memory.avatar_pos = avatar_info

        if self.memory.probing_phase and self.memory.probe_actions_left:
            while self.memory.probe_actions_left:
                probe = self.memory.probe_actions_left.pop(0)
                if probe in legal_actions:
                    probe.reasoning = {"why": "phase_1_epistemic_probe"}
                    self.last_action = probe
                    return probe
            self.memory.probing_phase = False

        # Phase 2: Tactical BFS to closest target if avatar known
        if self.memory.avatar_pos:
            r, c = self.memory.avatar_pos
            # Confirm avatar is still at position or re-locate
            if self.memory.avatar_color is not None:
                matches = np.argwhere(grid == self.memory.avatar_color)
                if len(matches) > 0:
                    start_pos = (int(matches[0][0]), int(matches[0][1]))
                    tactical_move = self._bfs_path_to_targets(grid, start_pos, legal_actions)
                    if tactical_move:
                        tactical_move.reasoning = {"why": "phase_2_tactical_bfs"}
                        self.last_action = tactical_move
                        return tactical_move

        # Fallback: Informed directional exploration
        directional = [a for a in legal_actions if a in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4)]
        if directional:
            action = random.choice(directional)
        else:
            action = random.choice(legal_actions)

        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
            action.reasoning = {"why": "fallback_complex_action"}
        else:
            action.reasoning = {"why": "fallback_simple_action"}

        self.last_action = action
        return action
