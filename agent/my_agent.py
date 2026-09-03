"""
The Dual-Brain Cognitive Agent for ARC-AGI-3 (Version 2).
Engineered to maximize Relative Human Action Efficiency (RHAE) under quadratic penalty:
1. Scene Perception:
   - Dynamic background color detection via histogram mode.
   - Pure NumPy connected-component segmentation for sprite clusters & centroids.
   - Fast background/canvas filtering for sub-millisecond execution.
2. Geodesic Trajectory Planning (Requirement R1):
   - Direct straight-line Manhattan action sequences (|r1-r2| + |c1-c2|) with zero jitter.
   - Minimal convex detours via A* with turn-penalty metric.
   - Multi-step commitment queue (collections.deque) delivering O(1) latency (<0.5ms/step).
3. Complete Action Space Specialization (Requirement R2):
   - ACTION1..4: Directional moves from geodesic queue.
   - ACTION5: Contextual interaction affordance testing & execution.
   - ACTION6: Coordinate tap targeting sprite centroids {"x": x, "y": y} in [0, 63] via click queue.
4. Cognitive Integrity & Anti-Fragility (Requirement R3):
   - Two-Phase Planning (Phase 1 micro-probing; Phase 2 goal pursuit).
   - Real-time bump collision detection updating occupancy grid (passable map).
   - State hashing + n-gram action cycle detection to break circular deadlocks.
   - Terminal Safety: Immediate GameAction.RESET on GAME_OVER preventing HTTP 400.
"""
from __future__ import annotations

import collections
import heapq
import hashlib
import random
import time
from typing import Any, List, Optional, Set, Tuple

import numpy as np
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

DIRECTION_VECTORS: dict[GameAction, Tuple[int, int]] = {
    GameAction.ACTION1: (-1, 0),  # Up
    GameAction.ACTION2: (1, 0),   # Down
    GameAction.ACTION3: (0, -1),  # Left
    GameAction.ACTION4: (0, 1),   # Right
}

VECTOR_TO_ACTION: dict[Tuple[int, int], GameAction] = {
    (-1, 0): GameAction.ACTION1,
    (1, 0): GameAction.ACTION2,
    (0, -1): GameAction.ACTION3,
    (0, 1): GameAction.ACTION4,
}


class WorkingMemory:
    """Tracks state history, loop counts, occupancy grid, and hypotheses."""

    def __init__(self, loop_threshold: int = 3):
        self.loop_threshold = loop_threshold
        self.visited_states: collections.Counter = collections.Counter()
        self.recent_actions: collections.deque[GameAction] = collections.deque(maxlen=16)
        self.plan_queue: collections.deque[GameAction] = collections.deque()
        self.click_queue: collections.deque[Tuple[int, int]] = collections.deque()
        self.avatar_color: Optional[int] = None
        self.avatar_pos: Optional[Tuple[int, int]] = None
        self.step_size: int = 1
        self.avatar_colors: set[int] = set()
        self.probing_phase: bool = True
        self.probe_actions_left: list[GameAction] = []
        self.passable_map = np.ones((64, 64), dtype=bool)
        self.known_walls: set[Tuple[int, int]] = set()
        self.clicked_centroids: collections.Counter = collections.Counter()
        self.last_grid_bytes: Optional[bytes] = None
        self.cached_sprites: Optional[list[dict[str, Any]]] = None
        self.last_state_hash: Optional[str] = None
        self.consecutive_bumps: int = 0

    def reset_for_level(self):
        self.visited_states.clear()
        self.recent_actions.clear()
        self.plan_queue.clear()
        self.click_queue.clear()
        self.avatar_color = None
        self.avatar_pos = None
        self.step_size = 1
        self.avatar_colors.clear()
        self.probing_phase = True
        self.probe_actions_left = [
            GameAction.ACTION4,
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
        ]
        self.passable_map.fill(True)
        self.known_walls.clear()
        self.clicked_centroids.clear()
        self.last_grid_bytes = None
        self.cached_sprites = None
        self.last_state_hash = None
        self.consecutive_bumps = 0

    def hash_state(self, grid_bytes: bytes, avatar_pos: Optional[Tuple[int, int]]) -> str:
        pos_bytes = f"{avatar_pos[0]},{avatar_pos[1]}".encode() if avatar_pos else b"none"
        return hashlib.md5(grid_bytes + pos_bytes).hexdigest()

    def record_step(self, action: Optional[GameAction], grid_bytes: bytes, avatar_pos: Optional[Tuple[int, int]]) -> str:
        h = self.hash_state(grid_bytes, avatar_pos)
        self.visited_states[h] += 1
        if action:
            self.recent_actions.append(action)
        self.last_state_hash = h
        return h

    def is_in_loop(self) -> bool:
        if self.last_state_hash and self.visited_states[self.last_state_hash] >= self.loop_threshold:
            return True
        acts = list(self.recent_actions)
        n = len(acts)
        if n >= 4 and acts[-1] == acts[-3] and acts[-2] == acts[-4]:
            return True
        if n >= 6 and acts[-1] == acts[-4] and acts[-2] == acts[-5] and acts[-3] == acts[-6]:
            return True
        if n >= 8 and acts[-4:] == acts[-8:-4]:
            return True
        return False


class ScenePerception:
    """Pure NumPy scene perception and connected-component segmentation."""

    @staticmethod
    def extract_grid(latest_frame: Optional[FrameData]) -> np.ndarray:
        if latest_frame is None or not getattr(latest_frame, "frame", None):
            return np.zeros((64, 64), dtype=np.int8)
        raw = np.array(latest_frame.frame[0], dtype=np.int8)
        while raw.ndim > 2:
            raw = raw[0]
        if raw.shape != (64, 64):
            grid = np.zeros((64, 64), dtype=np.int8)
            h, w = min(64, raw.shape[0]), min(64, raw.shape[1])
            grid[:h, :w] = raw[:h, :w]
            return grid
        return raw

    @staticmethod
    def detect_background_and_canvas(grid: np.ndarray) -> Tuple[int, Set[int]]:
        counts = np.bincount(grid.ravel().clip(0, 15), minlength=16)
        bg_color = int(np.argmax(counts))
        # Large canvas / background colors that contain > 400 pixels
        canvas_colors = set(np.where(counts > 400)[0].tolist())
        return bg_color, canvas_colors

    @staticmethod
    def is_border_or_wall(s: dict[str, Any]) -> bool:
        min_r, max_r, min_c, max_c = s["bbox"]
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        if (min_r <= 1 and h == 1 and w >= 32) or (min_r >= 54 and h <= 2 and w >= 32):
            return True
        if s["size"] > 150:
            return True
        return False

    @staticmethod
    def segment_sprites(
        grid: np.ndarray, bg_color: int, ignored_colors: Optional[Set[int]] = None
    ) -> list[dict[str, Any]]:
        H, W = grid.shape
        visited = np.zeros((H, W), dtype=bool)
        sprites = []
        ignored = (ignored_colors or set()) | {bg_color}

        non_bg_mask = ~np.isin(grid, list(ignored))
        non_bg_mask[56:, :] = False  # Ignore HUD
        coords_r, coords_c = np.where(non_bg_mask)

        for r, c in zip(coords_r, coords_c):
            if visited[r, c]:
                continue
            color = int(grid[r, c])
            coords = []
            queue = [(r, c)]
            visited[r, c] = True
            idx = 0
            while idx < len(queue):
                cr, cc = queue[idx]
                idx += 1
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < 56 and 0 <= nc < W and not visited[nr, nc] and grid[nr, nc] == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            cx = int(round(sum(cols) / len(cols)))
            cy = int(round(sum(rows) / len(rows)))
            sprites.append({
                "color": color,
                "size": len(coords),
                "bbox": (int(min(rows)), int(max(rows)), int(min(cols)), int(max(cols))),
                "centroid": (int(np.clip(cx, 0, 63)), int(np.clip(cy, 0, 63))),
                "coords": coords,
            })

        # Sort sprites by saliency: compact interactive elements prioritized over huge walls
        sprites.sort(key=lambda s: (ScenePerception.is_border_or_wall(s), s["size"]))
        return sprites


class AvatarTracker:
    """Isolates avatar position, translation vector, and bump collisions."""

    @staticmethod
    def update(
        prev_grid: np.ndarray,
        curr_grid: np.ndarray,
        last_action: Optional[GameAction],
        memory: WorkingMemory,
    ) -> Tuple[Optional[Tuple[int, int]], bool]:
        if last_action not in DIRECTION_VECTORS:
            return memory.avatar_pos, True

        exp_dr, exp_dc = DIRECTION_VECTORS[last_action]
        diff = prev_grid != curr_grid
        diff[56:, :] = False

        if not np.any(diff):
            memory.consecutive_bumps += 1
            memory.plan_queue.clear()
            if memory.avatar_pos:
                ar, ac = memory.avatar_pos
                step = memory.step_size
                raw_wall_r = ar + exp_dr * step
                raw_wall_c = ac + exp_dc * step
                if 0 <= raw_wall_r < 64 and 0 <= raw_wall_c < 64:
                    wall_r = raw_wall_r
                    wall_c = raw_wall_c
                    r_low, r_high = max(0, wall_r - step // 2), min(64, wall_r + step // 2 + 1)
                    c_low, c_high = max(0, wall_c - step // 2), min(64, wall_c + step // 2 + 1)
                    memory.passable_map[r_low:r_high, c_low:c_high] = False
                    memory.known_walls.add((wall_r, wall_c))
                    # Defensive invariant: avatar's current standing position is strictly passable
                    memory.passable_map[ar, ac] = True
                    memory.known_walls.discard((ar, ac))
            return memory.avatar_pos, False

        memory.consecutive_bumps = 0
        pts = np.argwhere(diff)
        if len(pts) == 0:
            return memory.avatar_pos, True

        if exp_dr != 0:
            med = np.median(pts[:, 0])
            curr_pts = pts[pts[:, 0] < med] if exp_dr < 0 else pts[pts[:, 0] > med]
            prev_pts = pts[pts[:, 0] > med] if exp_dr < 0 else pts[pts[:, 0] < med]
        else:
            med = np.median(pts[:, 1])
            curr_pts = pts[pts[:, 1] < med] if exp_dc < 0 else pts[pts[:, 1] > med]
            prev_pts = pts[pts[:, 1] > med] if exp_dc < 0 else pts[pts[:, 1] < med]

        if len(curr_pts) > 0 and len(prev_pts) > 0:
            r1, c1 = curr_pts.mean(axis=0)
            r0, c0 = prev_pts.mean(axis=0)
            measured_dr = r1 - r0
            measured_dc = c1 - c0
            step = int(round(max(abs(measured_dr), abs(measured_dc))))
            if step >= 1:
                memory.step_size = step

            new_pos = (int(round(r1)), int(round(c1)))
            memory.avatar_pos = new_pos
            colors = set(np.unique(curr_grid[curr_pts[:, 0], curr_pts[:, 1]]).tolist())
            if colors:
                memory.avatar_colors.update(colors)
                memory.avatar_color = list(colors)[0]
            return new_pos, True

        return memory.avatar_pos, True


class GeodesicPlanner:
    """Plans direct straight-line Manhattan trajectories with minimal convex detours via A*."""

    @staticmethod
    def plan(
        start: Tuple[int, int],
        goal: Tuple[int, int],
        passable: np.ndarray,
        step_size: int,
        legal_actions: Set[GameAction],
    ) -> list[GameAction]:
        if start == goal:
            return []

        sr, sc = start
        gr, gc = goal
        dr = gr - sr
        dc = gc - sc

        v_step = 1 if dr > 0 else -1
        h_step = 1 if dc > 0 else -1
        v_act = GameAction.ACTION2 if dr > 0 else GameAction.ACTION1
        h_act = GameAction.ACTION4 if dc > 0 else GameAction.ACTION3

        v_count = int(round(abs(dr) / max(1, step_size)))
        h_count = int(round(abs(dc) / max(1, step_size)))

        if v_count == 0 and h_count == 0:
            return []

        # 1. Vertical first, then Horizontal
        can_v_first = (v_act in legal_actions or v_count == 0) and (h_act in legal_actions or h_count == 0)
        if can_v_first:
            blocked = False
            curr_r, curr_c = sr, sc
            for _ in range(v_count):
                curr_r = int(np.clip(curr_r + v_step * step_size, 0, 63))
                if not passable[curr_r, curr_c] and (curr_r, curr_c) != (gr, gc):
                    blocked = True
                    break
            if not blocked:
                for _ in range(h_count):
                    curr_c = int(np.clip(curr_c + h_step * step_size, 0, 63))
                    if not passable[curr_r, curr_c] and (curr_r, curr_c) != (gr, gc):
                        blocked = True
                        break
            if not blocked and (v_count > 0 or h_count > 0):
                return [v_act] * v_count + [h_act] * h_count

        # 2. Horizontal first, then Vertical
        can_h_first = (h_act in legal_actions or h_count == 0) and (v_act in legal_actions or v_count == 0)
        if can_h_first:
            blocked = False
            curr_r, curr_c = sr, sc
            for _ in range(h_count):
                curr_c = int(np.clip(curr_c + h_step * step_size, 0, 63))
                if not passable[curr_r, curr_c] and (curr_r, curr_c) != (gr, gc):
                    blocked = True
                    break
            if not blocked:
                for _ in range(v_count):
                    curr_r = int(np.clip(curr_r + v_step * step_size, 0, 63))
                    if not passable[curr_r, curr_c] and (curr_r, curr_c) != (gr, gc):
                        blocked = True
                        break
            if not blocked and (v_count > 0 or h_count > 0):
                return [h_act] * h_count + [v_act] * v_count

        # 3. Obstructed: A* minimal convex detour with turn penalty
        return GeodesicPlanner.a_star_convex_detour(start, goal, passable, step_size, legal_actions)

    @staticmethod
    def a_star_convex_detour(
        start: Tuple[int, int],
        goal: Tuple[int, int],
        passable: np.ndarray,
        step_size: int,
        legal_actions: Set[GameAction],
        max_expansions: int = 1200,
    ) -> list[GameAction]:
        if start == goal:
            return []

        sr, sc = start
        gr, gc = goal
        dirs = [
            (-1, 0, GameAction.ACTION1),
            (1, 0, GameAction.ACTION2),
            (0, -1, GameAction.ACTION3),
            (0, 1, GameAction.ACTION4),
        ]
        valid_dirs = [d for d in dirs if d[2] in legal_actions]
        if not valid_dirs:
            return []

        pq = [(abs(sr - gr) + abs(sc - gc), 0.0, sr, sc, 0, 0)]
        came_from = {}
        g_score = {(sr, sc): 0.0}

        H, W = passable.shape
        found = False
        expansions = 0
        target_r, target_c = sr, sc

        while pq and expansions < max_expansions:
            f, g, r, c, ldr, ldc = heapq.heappop(pq)
            expansions += 1

            if (r, c) == (gr, gc) or (
                step_size > 1
                and abs(r - gr) < step_size
                and abs(c - gc) < step_size
                and (r, c) != (sr, sc)
            ):
                found = True
                target_r, target_c = r, c
                break

            if g > g_score.get((r, c), float("inf")):
                continue

            for dr, dc, act in valid_dirs:
                nr = int(np.clip(r + dr * step_size, 0, H - 1))
                nc = int(np.clip(c + dc * step_size, 0, W - 1))
                if (nr, nc) == (r, c):
                    continue
                if not passable[nr, nc] and (nr, nc) != (gr, gc):
                    continue

                turn = (ldr != 0 or ldc != 0) and (ldr != dr or ldc != dc)
                edge_cost = 1.2 if turn else 1.0
                new_g = g + edge_cost
                if new_g < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = new_g
                    new_f = new_g + (abs(nr - gr) + abs(nc - gc)) / max(1, step_size)
                    came_from[(nr, nc)] = (r, c, act)
                    heapq.heappush(pq, (new_f, new_g, nr, nc, dr, dc))

        if not found:
            return []

        path = []
        curr = (target_r, target_c)
        while curr in came_from:
            prev_r, prev_c, act = came_from[curr]
            path.append(act)
            curr = (prev_r, prev_c)
        path.reverse()
        return path


class MyAgent(Agent):
    """Dual-Brain Version 2 Agent implementing Geodesic Manhattan Trajectories & Complete Action Space."""

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
        return f"DualBrainAgent.v2.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in (GameState.WIN, "WIN")

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER, "NOT_PLAYED", "GAME_OVER"):
            self.memory.reset_for_level()
            self.last_action = GameAction.RESET
            return GameAction.RESET

        grid = ScenePerception.extract_grid(latest_frame)
        grid_bytes = grid.tobytes()
        bg_color, canvas_colors = ScenePerception.detect_background_and_canvas(grid)

        available = getattr(latest_frame, "available_actions", None)
        if available:
            legal_actions = [GameAction.from_id(int(a)) for a in available if a != 0]
        else:
            legal_actions = [a for a in GameAction if a is not GameAction.RESET]

        if not legal_actions:
            return GameAction.RESET

        legal_set = set(legal_actions)

        # Track movement & bump collisions
        if len(frames) >= 2 and self.last_action:
            prev_grid = ScenePerception.extract_grid(frames[-2])
            AvatarTracker.update(prev_grid, grid, self.last_action, self.memory)

        self.memory.record_step(self.last_action, grid_bytes, self.memory.avatar_pos)

        # Loop interceptor
        if self.memory.is_in_loop():
            self.memory.plan_queue.clear()
            self.memory.click_queue.clear()
            escape = self._get_orthogonal_escape(legal_actions)
            if escape:
                self.last_action = escape
                return escape

        # ACTION6 click dispatch (queue-based O(1) execution)
        directional_legal = [a for a in legal_actions if a in DIRECTION_VECTORS]
        if GameAction.ACTION6 in legal_set and (not directional_legal or len(directional_legal) < 2):
            return self._handle_action6(grid, bg_color, canvas_colors)

        # Phase 1 micro-probe
        if self.memory.probing_phase and directional_legal:
            while self.memory.probe_actions_left:
                probe = self.memory.probe_actions_left.pop(0)
                if probe in legal_set:
                    probe.reasoning = {"why": "phase_1_epistemic_probe"}
                    self.last_action = probe
                    return probe
            self.memory.probing_phase = False

        # Phase 2 geodesic queue pop (O(1) fast path)
        if self.memory.plan_queue:
            action = self.memory.plan_queue.popleft()
            if action in legal_set:
                action.reasoning = {"why": "geodesic_queue_pop"}
                self.last_action = action
                return action

        # ACTION5 contextual affordance
        if GameAction.ACTION5 in legal_set and self._should_trigger_action5(grid, bg_color):
            act5 = GameAction.ACTION5
            act5.reasoning = {"why": "contextual_affordance_action5"}
            self.last_action = act5
            return act5

        # Plan new geodesic trajectory
        if self.memory.avatar_pos and directional_legal:
            sprites = ScenePerception.segment_sprites(grid, bg_color, canvas_colors | self.memory.avatar_colors)
            target = self._select_best_target(sprites, self.memory.avatar_pos)
            if target:
                target_pos = (target["centroid"][1], target["centroid"][0])
                path = GeodesicPlanner.plan(
                    self.memory.avatar_pos,
                    target_pos,
                    self.memory.passable_map,
                    self.memory.step_size,
                    legal_set,
                )
                if path:
                    self.memory.plan_queue.extend(path)
                    action = self.memory.plan_queue.popleft()
                    action.reasoning = {"why": "geodesic_trajectory_start"}
                    self.last_action = action
                    return action

        # Fallback informed move
        action = self._fallback_action(legal_actions, grid, bg_color, canvas_colors)
        self.last_action = action
        return action

    def _handle_action6(self, grid: np.ndarray, bg_color: int, canvas_colors: Set[int]) -> GameAction:
        if not self.memory.click_queue:
            sprites = ScenePerception.segment_sprites(
                grid, bg_color, canvas_colors | self.memory.avatar_colors
            )
            if not sprites:
                # If all non-bg colors were ignored, fallback to non-bg without canvas filter
                sprites = ScenePerception.segment_sprites(
                    grid, bg_color, self.memory.avatar_colors
                )
            if not sprites:
                self.memory.click_queue.append((32, 32))
            else:
                sprites.sort(key=lambda s: (
                    ScenePerception.is_border_or_wall(s),
                    self.memory.clicked_centroids[s["centroid"]],
                    s["size"]
                ))
                for s in sprites:
                    self.memory.click_queue.append(s["centroid"])

        cx, cy = self.memory.click_queue.popleft()
        self.memory.clicked_centroids[(cx, cy)] += 1
        act = GameAction.ACTION6
        act.set_data({"x": int(cx), "y": int(cy)})
        act.reasoning = {"why": f"action6_target_centroid_({cx},{cy})"}
        self.last_action = act
        return act

    def _should_trigger_action5(self, grid: np.ndarray, bg_color: int) -> bool:
        if not self.memory.avatar_pos:
            return False
        ar, ac = self.memory.avatar_pos
        step = self.memory.step_size
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(np.clip(ar + dr * step, 0, 63)), int(np.clip(ac + dc * step, 0, 63))
            if grid[nr, nc] != bg_color and grid[nr, nc] not in self.memory.avatar_colors:
                return True
        return False

    def _get_orthogonal_escape(self, legal_actions: List[GameAction]) -> Optional[GameAction]:
        legal_set = set(legal_actions)
        last = self.last_action

        def is_passable(act: GameAction) -> bool:
            if not self.memory.avatar_pos or act not in DIRECTION_VECTORS:
                return True
            ar, ac = self.memory.avatar_pos
            step = self.memory.step_size
            dr, dc = DIRECTION_VECTORS[act]
            nr = int(np.clip(ar + dr * step, 0, 63))
            nc = int(np.clip(ac + dc * step, 0, 63))
            return bool(self.memory.passable_map[nr, nc] and (nr, nc) != (ar, ac))

        primary_cands: list[GameAction] = []
        primary_reason = "loop_break_passable"
        secondary_cands: list[GameAction] = []

        if last in (GameAction.ACTION1, GameAction.ACTION2):
            primary_cands = [GameAction.ACTION4, GameAction.ACTION3]
            primary_reason = "orthogonal_escape_horizontal"
            secondary_cands = [GameAction.ACTION2 if last == GameAction.ACTION1 else GameAction.ACTION1]
        elif last in (GameAction.ACTION3, GameAction.ACTION4):
            primary_cands = [GameAction.ACTION1, GameAction.ACTION2]
            primary_reason = "orthogonal_escape_vertical"
            secondary_cands = [GameAction.ACTION4 if last == GameAction.ACTION3 else GameAction.ACTION3]
        else:
            primary_cands = [GameAction.ACTION4, GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3]
            primary_reason = "loop_break_passable"
            secondary_cands = []

        # 1. Try passable primary orthogonal candidates (strictly excluding ACTION5)
        for esc in primary_cands:
            if esc in legal_set and esc != GameAction.ACTION5 and is_passable(esc):
                esc.reasoning = {"why": primary_reason}
                return esc

        # 2. Try passable secondary candidates (reverse / alternate directional)
        for esc in secondary_cands:
            if esc in legal_set and esc != GameAction.ACTION5 and is_passable(esc):
                esc.reasoning = {"why": "loop_break_passable"}
                return esc

        # 3. If avatar_pos is known and none of preferred are passable, try ANY passable directional action
        all_dirs = [GameAction.ACTION4, GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3]
        for esc in all_dirs:
            if esc in legal_set and esc != last and esc != GameAction.ACTION5 and is_passable(esc):
                esc.reasoning = {"why": "loop_break_passable"}
                return esc

        # 4. Fallback if none verified passable: pick primary candidate in legal_set (excluding ACTION5)
        for esc in primary_cands:
            if esc in legal_set and esc != GameAction.ACTION5:
                esc.reasoning = {"why": primary_reason}
                return esc

        # 5. Fallback: any legal directional action != last (excluding ACTION5)
        for esc in all_dirs:
            if esc in legal_set and esc != last and esc != GameAction.ACTION5:
                esc.reasoning = {"why": "loop_break_alternative"}
                return esc

        return None

    def _select_best_target(self, sprites: list[dict[str, Any]], avatar_pos: Tuple[int, int]) -> Optional[dict[str, Any]]:
        ar, ac = avatar_pos
        valid_sprites = []
        for s in sprites:
            cy, cx = s["centroid"][1], s["centroid"][0]
            if abs(cy - ar) < self.memory.step_size and abs(cx - ac) < self.memory.step_size:
                continue
            dist = abs(cy - ar) + abs(cx - ac)
            valid_sprites.append((dist, s))

        if not valid_sprites:
            return None

        valid_sprites.sort(key=lambda item: (
            ScenePerception.is_border_or_wall(item[1]),
            item[0]
        ))
        return valid_sprites[0][1]

    def _fallback_action(self, legal_actions: List[GameAction], grid: np.ndarray, bg_color: int, canvas_colors: Set[int]) -> GameAction:
        directional = [a for a in legal_actions if a in DIRECTION_VECTORS]
        if directional:
            if self.memory.avatar_pos:
                ar, ac = self.memory.avatar_pos
                step = self.memory.step_size
                passable_dirs = []
                for a in directional:
                    dr, dc = DIRECTION_VECTORS[a]
                    nr, nc = int(np.clip(ar + dr * step, 0, 63)), int(np.clip(ac + dc * step, 0, 63))
                    if self.memory.passable_map[nr, nc] and (nr, nc) != (ar, ac):
                        passable_dirs.append(a)
                if passable_dirs:
                    action = random.choice(passable_dirs)
                    action.reasoning = {"why": "fallback_passable_directional"}
                    return action
            action = random.choice(directional)
            action.reasoning = {"why": "fallback_directional"}
            return action

        action = random.choice(legal_actions)
        if action.is_complex():
            return self._handle_action6(grid, bg_color, canvas_colors)
        action.reasoning = {"why": "fallback_simple"}
        return action
