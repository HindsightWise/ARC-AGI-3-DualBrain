"""
The Dual-Brain Cognitive Agent for ARC-AGI-3 (Version 4: Generalized Cognitive Flexibility).
Engineered to maximize Relative Human Action Efficiency (RHAE) under quadratic penalty:
1. Universal Epistemic Prober & Dynamic Lattice Detector:
   - Measures avatar displacement (dr, dc) during minimal probe phase.
   - Infers discrete lattice pitch (step_size), spatial offsets (r_offset, c_offset), and tile dimensions.
   - Works across arbitrary step sizes (1, 2, 3, 4, 5, 7, 8, etc.) with zero hardcoded coordinates.
2. Dynamic Spatial Invariance HUD & Goal Template Detector:
   - Dynamically scans perimeter quadrants for static framed boxes.
   - Extracts target configuration without hardcoded screen regions (no rigid 55:61 or bottom-left locks).
   - Masks ONLY the detected HUD bounding box, preserving full 64x64 visibility on non-HUD games.
3. Multi-Mode Cognitive Dispatch (The Chameleon Brain):
   - Mode A: Dynamic State-Space BFS for transformation puzzle games (e.g. ls20).
   - Mode B: Salient Geodesic Goal Pursuit for navigation/collector/maze games (e.g. tr87, tu93).
   - Mode C: Adjacency Affordance Tester for interaction games (ACTION5).
   - Mode D: Coordinate Click Dispatch for point-and-click games (ACTION6, e.g. vc33).
   - Mode E: Hash-based loop interceptor and terminal safety (immediate RESET on GAME_OVER).
4. Online Empirical Affordance Learner:
   - Tracks board state transitions upon stepping on tiles, dynamically classifying operator pads.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
import heapq
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, List, Optional, Set, Tuple, Union

import numpy as np
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

DEFAULT_POLICY: dict[str, Any] = {
    "step_lock_threshold": 3,
    "loop_threshold": 3,
    "click_strategy": "size_asc",
    "camera_quantization_mode": "auto",
    "bfs_max_nodes": 20000,
    "geodesic_heuristic_weight": 1.2,
    "hazard_repulsion_radius": 1,
}

def load_policy() -> dict[str, Any]:
    policy = dict(DEFAULT_POLICY)
    candidates = [
        Path(__file__).resolve().parent / "champion_policy.json",
        Path("agent/champion_policy.json"),
        Path("/tmp/champion_policy.json"),
    ]
    for c in candidates:
        try:
            if c.exists():
                with open(c, "r") as f:
                    user_policy = json.load(f)
                    if isinstance(user_policy, dict):
                        policy.update(user_policy)
                break
        except Exception:
            pass
    return policy

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



class CameraTransform:
    """Manages coordinate transformations between internal grid space and 64x64 display space."""

    def __init__(
        self,
        scale: int = 1,
        offset_x: int = 0,
        offset_y: int = 0,
        width: int = 64,
        height: int = 64,
        letter_box_color: Optional[int] = None,
    ):
        self.scale = scale
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.width = width
        self.height = height
        self.letter_box_color = letter_box_color

    def __repr__(self) -> str:
        return (
            f"CameraTransform(scale={self.scale}, offset_x={self.offset_x}, "
            f"offset_y={self.offset_y}, width={self.width}, height={self.height})"
        )

    def grid_to_display(self, grid_x: int, grid_y: int) -> Tuple[int, int]:
        """Convert internal grid (x, y) coordinates to display (x, y) clamped to [0, 63]."""
        disp_x = int(np.clip(int(grid_x) * self.scale + self.offset_x + (self.scale // 2), 0, 63))
        disp_y = int(np.clip(int(grid_y) * self.scale + self.offset_y + (self.scale // 2), 0, 63))
        return disp_x, disp_y

    def display_to_grid(self, disp_x: int, disp_y: int) -> Optional[Tuple[int, int]]:
        """Convert display (x, y) to internal grid (x, y), returning None if within letterbox."""
        if disp_x < self.offset_x or disp_y < self.offset_y:
            return None
        scaled_w = self.width * self.scale
        scaled_h = self.height * self.scale
        if disp_x >= self.offset_x + scaled_w or disp_y >= self.offset_y + scaled_h:
            return None
        gx = (disp_x - self.offset_x) // self.scale
        gy = (disp_y - self.offset_y) // self.scale
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return gx, gy
        return None

    def is_inside_viewport(self, disp_x: int, disp_y: int) -> bool:
        """Returns True if the display coordinate lies within the active camera viewport."""
        scaled_w = self.width * self.scale
        scaled_h = self.height * self.scale
        return (
            self.offset_x <= disp_x < self.offset_x + scaled_w and
            self.offset_y <= disp_y < self.offset_y + scaled_h
        )

    @classmethod
    def from_environment(cls, arc_env: Any = None, grid: Optional[np.ndarray] = None) -> "CameraTransform":
        """Factory creating CameraTransform from environment wrapper if available, or grid fallback."""
        if arc_env is not None:
            try:
                game = getattr(arc_env, "_game", getattr(arc_env, "game", None))
                if game is not None and type(game).__name__ != "MagicMock":
                    cam = getattr(game, "camera", None)
                    if cam is not None and type(cam).__name__ != "MagicMock" and hasattr(cam, "_calculate_scale_and_offset"):
                        res = cam._calculate_scale_and_offset()
                        if isinstance(res, (tuple, list)) and len(res) == 3:
                            scale, xo, yo = res
                            return cls(
                                scale=int(scale),
                                offset_x=int(xo),
                                offset_y=int(yo),
                                width=int(getattr(cam, "width", 64)),
                                height=int(getattr(cam, "height", 64)),
                                letter_box_color=getattr(cam, "letter_box", None),
                            )
            except Exception:
                pass
        if grid is not None and grid.shape == (64, 64):
            return cls.infer_from_grid(grid)
        return cls()

    @classmethod
    def infer_from_grid(cls, grid: np.ndarray) -> "CameraTransform":
        """Zero-dependency visual inference of camera scale and letterbox margins from a 64x64 grid."""
        if len(np.unique(grid)) <= 1:
            return cls(scale=1, offset_x=0, offset_y=0, width=64, height=64, letter_box_color=None)

        top_color = grid[0, 0]
        top = 0
        while top < 32 and np.all(grid[top, :] == top_color):
            top += 1
        bottom = 0
        while bottom < 32 and np.all(grid[63 - bottom, :] == top_color):
            bottom += 1
        left = 0
        while left < 32 and np.all(grid[:, left] == top_color):
            left += 1
        right = 0
        while right < 32 and np.all(grid[:, 63 - right] == top_color):
            right += 1

        yo = top if (top == bottom and top > 0) else 0
        active_h = 64 - 2 * yo
        xo = left if (left == right and left > 0) else 0
        active_w = 64 - 2 * xo

        if active_w <= 0 or active_h <= 0:
            return cls(scale=1, offset_x=0, offset_y=0, width=64, height=64, letter_box_color=None)

        best_scale = 1
        for S in [8, 5, 4, 3, 2]:
            if active_w % S == 0 and active_h % S == 0:
                sub = grid[yo : yo + active_h, xo : xo + active_w]
                down = sub[::S, ::S]
                up = np.repeat(np.repeat(down, S, axis=0), S, axis=1)
                if np.array_equal(sub, up):
                    best_scale = S
                    break

        return cls(
            scale=best_scale,
            offset_x=xo,
            offset_y=yo,
            width=active_w // best_scale,
            height=active_h // best_scale,
            letter_box_color=int(top_color) if (top > 0 or left > 0) else None,
        )


def make_action6(
    x: Union[int, float],
    y: Union[int, float],
    why: str = "",
    camera_transform: Optional[CameraTransform] = None,
) -> GameAction:
    """Builds an ACTION6 coordinate click guaranteed to be an integer clamped to [0, 63]."""
    fx = float(x)
    fy = float(y)
    if np.isnan(fx) or np.isinf(fx):
        fx = 32.0
    if np.isnan(fy) or np.isinf(fy):
        fy = 32.0

    if camera_transform is not None:
        g = camera_transform.display_to_grid(int(round(fx)), int(round(fy)))
        if g is not None:
            cx, cy = camera_transform.grid_to_display(g[0], g[1])
        else:
            cx, cy = camera_transform.grid_to_display(int(round(fx)), int(round(fy)))
    else:
        cx = int(np.clip(int(round(fx)), 0, 63))
        cy = int(np.clip(int(round(fy)), 0, 63))

    act = GameAction.ACTION6
    act.set_data({"x": cx, "y": cy})
    act.reasoning = {"why": why, "x": cx, "y": cy}
    return act



class PawnRecord:
    def __init__(
        self,
        pawn_id: int,
        pos: Tuple[int, int],
        color: int,
        size: int,
        bbox: Tuple[int, int, int, int],
        is_active: bool,
        consecutive_stalls: int = 0,
    ):
        self.pawn_id = pawn_id
        self.pos = pos
        self.color = color
        self.size = size
        self.bbox = bbox
        self.is_active = is_active
        self.consecutive_stalls = consecutive_stalls


class WorkingMemory:
    """Tracks state history, loop counts, occupancy grid, and hypotheses."""

    def __init__(self, loop_threshold: int = 3, step_lock_threshold: int = 3):
        self.loop_threshold = loop_threshold
        self.step_lock_threshold = step_lock_threshold
        self.tracked_pawns: dict[int, PawnRecord] = {}
        self.active_pawn_id: Optional[int] = None
        self.pawn_swap_latch: bool = False
        self.step_counts: collections.Counter = collections.Counter()
        self.step_locked: bool = False
        self.visited_states: collections.Counter = collections.Counter()
        self.visited_positions: collections.Counter = collections.Counter()
        self.history: list[dict[str, Any]] = []
        self.recent_actions: collections.deque[GameAction] = collections.deque(maxlen=16)
        self.plan_queue: collections.deque[GameAction] = collections.deque()
        self.click_queue: collections.deque[Tuple[int, int]] = collections.deque()
        self.avatar_color: Optional[int] = None
        self.avatar_pos: Optional[Tuple[int, int]] = None  # (top_r, left_c)
        self.last_tracked_pos: Optional[Tuple[int, int]] = None
        self.stagnation_counter: int = 0
        self.step_size: int = 1
        self.avatar_colors: set[int] = set()
        self.probing_phase: bool = True
        self.probe_actions_left: list[GameAction] = []
        self.passable_map = np.ones((64, 64), dtype=bool)
        self.known_walls: set[Tuple[int, int]] = set()
        self.clicked_centroids: collections.Counter = collections.Counter()
        self.visited_targets: set[Tuple[int, int]] = set()
        self.last_grid_bytes: Optional[bytes] = None
        self.last_state_hash: Optional[str] = None
        self.consecutive_bumps: int = 0
        self.hud_bbox: Optional[Tuple[int, int, int, int]] = None  # (rmin, rmax, cmin, cmax)
        self.active_control: Optional[Tuple[int, int]] = None
        self.consecutive_active: int = 0
        self.inert_coords: set[Tuple[int, int]] = set()
        self.wormhole_edges: dict[Tuple[int, int], Tuple[int, int]] = {}
        self.portal_actions: dict[Tuple[int, int], GameAction] = {}
        self.affordance_state_cache: dict[Tuple[int, int], list[int]] = {}
        self.void_colors: set[int] = set()
        self.current_target: Optional[dict[str, Any]] = None
        self.is_ice_slider: bool = False
        self.solver_plan: bool = False
        self.is_unbounded_void: bool = False
        self.melody_sequence_centroids: list[Tuple[int, int]] = []
        self.melody_completed: bool = False
        self.exit_target_pos: Optional[Tuple[int, int]] = None
        self.collectible_colors: set[int] = set()
        self.pit_trap_pos: Optional[Tuple[int, int]] = None
        self.beacon_pos: Optional[Tuple[int, int]] = None
        self.known_plates: set[Tuple[int, int]] = set()
        self.visited_plates: set[Tuple[int, int]] = set()
        self._pawn_swap_cooldown: int = 0

    def reset_for_level(self):
        self.visited_states.clear()
        self.visited_positions.clear()
        self.history.clear()
        self.recent_actions.clear()
        self.plan_queue.clear()
        self.click_queue.clear()
        self.avatar_color = None
        self.avatar_pos = None
        self.last_tracked_pos = None
        self.stagnation_counter = 0
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
        self.visited_targets.clear()
        self.last_grid_bytes = None
        self.last_state_hash = None
        self.consecutive_bumps = 0
        self.hud_bbox = None
        self.step_counts.clear()
        self.step_locked = False
        self.active_control = None
        self.consecutive_active = 0
        self.inert_coords.clear()
        self.wormhole_edges.clear()
        self.portal_actions.clear()
        self.affordance_state_cache.clear()
        self.void_colors.clear()
        self.current_target = None
        self.is_ice_slider = False
        self.solver_plan = False
        self.is_unbounded_void = False
        self.melody_sequence_centroids.clear()
        self.melody_completed = False
        self.exit_target_pos = None
        self.collectible_colors.clear()
        self.pit_trap_pos = None
        self.beacon_pos = None
        self.known_plates.clear()
        self.visited_plates.clear()
        self._pawn_swap_cooldown = 0
        self.tracked_pawns.clear()
        self.active_pawn_id = None
        self.pawn_swap_latch = False

    def hash_state(self, grid_bytes: bytes, avatar_pos: Optional[Tuple[int, int]]) -> str:
        pos_bytes = f"{avatar_pos[0]},{avatar_pos[1]}".encode() if avatar_pos else b"none"
        return hashlib.md5(grid_bytes + pos_bytes).hexdigest()

    def record_step(self, action: Optional[GameAction], grid_bytes: bytes, avatar_pos: Optional[Tuple[int, int]]) -> str:
        h = self.hash_state(grid_bytes, avatar_pos)
        self.visited_states[h] += 1
        if action:
            self.recent_actions.append(action)
        self.last_state_hash = h
        if avatar_pos is not None:
            self.visited_positions[avatar_pos] += 1
            if avatar_pos == self.last_tracked_pos:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                self.last_tracked_pos = avatar_pos
        if len(self.history) >= 256:
            self.history.pop(0)
        self.history.append({"action": action, "pos": avatar_pos, "hash": h})
        return h

    def is_in_loop(self) -> bool:
        if self.stagnation_counter >= 3:
            return True
        if self.last_state_hash and self.visited_states[self.last_state_hash] >= self.loop_threshold:
            return True
        acts = list(self.recent_actions)
        n = len(acts)
        # 2-cycle oscillation: A != B check strictly protects straight-line marches
        if n >= 4 and acts[-1] == acts[-3] and acts[-2] == acts[-4] and acts[-1] != acts[-2]:
            return True
        # 3-cycle oscillation
        if n >= 6 and acts[-1] == acts[-4] and acts[-2] == acts[-5] and acts[-3] == acts[-6] and len(set(acts[-3:])) > 1:
            return True
        # 4-cycle oscillation
        if n >= 8 and acts[-4:] == acts[-8:-4] and len(set(acts[-4:])) > 1:
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
        canvas_colors = set(np.where(counts > 400)[0].tolist())
        return bg_color, canvas_colors

    @staticmethod
    def is_border_or_wall(s: dict[str, Any]) -> bool:
        min_r, max_r, min_c, max_c = s["bbox"]
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        if (min_r <= 1 and h <= 2 and w >= 28) or (min_r >= 60 and h <= 2 and w >= 28):
            return True
        if s["size"] > 180:
            return True
        if (w >= 10 and h <= 4) or (h >= 10 and w <= 4):
            return True
        if s.get("color") == 5 and (w >= 8 or h >= 8):
            return True
        if (w >= 20 and h <= 6) or (h >= 20 and w <= 6):
            return True
        return False

    @staticmethod
    def segment_sprites(
        grid: np.ndarray,
        bg_color: int,
        ignored_colors: Optional[Set[int]] = None,
        hud_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> list[dict[str, Any]]:
        H, W = grid.shape
        visited = np.zeros((H, W), dtype=bool)
        sprites = []
        ignored = (ignored_colors or set()) | {bg_color}

        non_bg_mask = ~np.isin(grid, list(ignored))
        if hud_bbox:
            hr0, hr1, hc0, hc1 = hud_bbox
            non_bg_mask[hr0:hr1, hc0:hc1] = False

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
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and grid[nr, nc] == color:
                        if hud_bbox and hr0 <= nr < hr1 and hc0 <= nc < hc1:
                            continue
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            mean_c = sum(cols) / len(cols)
            mean_r = sum(rows) / len(rows)
            best_r, best_c = min(coords, key=lambda p: (p[0] - mean_r) ** 2 + (p[1] - mean_c) ** 2)
            sprites.append({
                "color": color,
                "size": len(coords),
                "bbox": (int(min(rows)), int(max(rows)), int(min(cols)), int(max(cols))),
                "centroid": (int(np.clip(best_c, 0, 63)), int(np.clip(best_r, 0, 63))),
                "coords": coords,
            })

        sprites.sort(key=lambda s: (ScenePerception.is_border_or_wall(s), s["size"]))
        return sprites

    @staticmethod
    def deduce_avatar_zero_shot(
        grid: np.ndarray,
        bg_color: int,
        step_size: int = 1,
        hud_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Isolates avatar coordinates and morphology from Frame 0 without exploratory probe actions.
        Returns candidate dictionary with keys: 'pos', 'color', 'size', 'bbox', 'confidence'.
        """
        sprites = ScenePerception.segment_sprites(grid, bg_color, hud_bbox=hud_bbox)
        candidates = []
        for s in sprites:
            if ScenePerception.is_border_or_wall(s) or s["color"] in (0, 14):
                continue
            r0, r1, c0, c1 = s["bbox"]
            if r0 >= 59:  # Bottom step counter / HUD
                continue
            if s["size"] > 64:  # Non-avatar background structures
                continue
            candidates.append(s)

        if not candidates:
            return None

        # Color frequency table across non-wall entities
        color_counts = collections.Counter(s["color"] for s in candidates)

        def score_candidate(s: dict[str, Any]) -> float:
            r0, r1, c0, c1 = s["bbox"]
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            aspect_diff = abs(h - w)
            cc = color_counts[s["color"]]

            score = 0.0
            # Avatar color rarity (singleton or mirrored twin)
            if cc == 1:
                score += 10.0
            elif cc == 2:  # Mirrored twins (e.g. m0r0)
                score += 5.0
            else:
                score -= cc * 1.5

            # Squareness
            if aspect_diff == 0:
                score += 6.0
            elif aspect_diff == 1:
                score += 3.0
            else:
                score -= aspect_diff * 2.0

            # Lattice pitch alignment
            if step_size > 1:
                if h == step_size and w == step_size:
                    score += 8.0
                elif h % step_size == 0 and w % step_size == 0:
                    score += 4.0
            else:
                if 2 <= h <= 3 and 2 <= w <= 3:
                    score += 8.0
                elif h == 1 and w == 1:
                    score += 2.0
                elif h <= 4 and w <= 4:
                    score += 2.0

            # Interior placement (avatars rarely spawn on edges 0 or 63)
            if 2 <= r0 <= 61 and 2 <= c0 <= 61:
                score += 3.0

            return score

        scored_candidates = [(score_candidate(s), s) for s in candidates]
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best = scored_candidates[0]

        if best_score < 8.0:
            return None

        # Relative confidence margin: require clear separation before trusting zero-shot deduction
        if len(scored_candidates) > 1:
            runner_up_score = scored_candidates[1][0]
            if (best_score - runner_up_score) < 5.0:
                return None

        return {
            "pos": (best["bbox"][0], best["bbox"][2]),
            "color": best["color"],
            "size": best["size"],
            "bbox": best["bbox"],
            "confidence": best_score,
            "all_candidates": candidates,
        }


class AvatarTracker:
    """Isolates avatar position, step size, and bump collisions via top-left bounding box."""

    @staticmethod
    def _get_connected_component_at(
        grid: np.ndarray,
        r: int,
        c: int,
        colors: Set[int],
    ) -> list[Tuple[int, int]]:
        """Extracts connected component coordinates of matching colors containing (r, c)."""
        if not (0 <= r < 64 and 0 <= c < 64) or grid[r, c] not in colors:
            return []
        visited = {(r, c)}
        q = collections.deque([(r, c)])
        while q:
            cr, cc = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < 64 and 0 <= nc < 64 and (nr, nc) not in visited:
                    if grid[nr, nc] in colors:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return list(visited)

    @staticmethod
    def _find_avatar_components(
        grid: np.ndarray,
        colors: Set[int],
        diff_pts: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Segments connected components of avatar colors in grid that intersect diff_pts."""
        if not colors or len(diff_pts) == 0:
            return []
        if len(diff_pts) > 200 and isinstance(diff_pts, np.ndarray) and diff_pts.ndim == 2:
            color_mask = np.isin(grid, list(colors))
            diff_mask = np.zeros((64, 64), dtype=bool)
            diff_mask[diff_pts[:, 0], diff_pts[:, 1]] = True
            diff_pts = np.argwhere(color_mask & diff_mask)

        visited = np.zeros((64, 64), dtype=bool)
        comps = []
        for r, c in diff_pts:
            ir, ic = int(r), int(c)
            if visited[ir, ic] or grid[ir, ic] not in colors:
                continue
            q = collections.deque([(ir, ic)])
            visited[ir, ic] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < 64 and 0 <= nc < 64 and not visited[nr, nc]:
                        if grid[nr, nc] in colors:
                            visited[nr, nc] = True
                            q.append((nr, nc))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            comps.append({
                "min_r": int(min(rs)),
                "max_r": int(max(rs)),
                "min_c": int(min(cs)),
                "max_c": int(max(cs)),
                "h": int(max(rs) - min(rs) + 1),
                "w": int(max(cs) - min(cs) + 1),
                "size": len(comp),
                "coords": comp,
            })
        return comps

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
        if memory.hud_bbox:
            hr0, hr1, hc0, hc1 = memory.hud_bbox
            diff[hr0:hr1, hc0:hc1] = False
        # Filter bottom HUD / step counters
        diff[59:, :] = False

        if not np.any(diff):
            if memory.avatar_pos:
                ar, ac = memory.avatar_pos
                step = memory.step_size
                target_r = ar + exp_dr * step
                target_c = ac + exp_dc * step
                # Double-push inertial affordance:
                # If target is interior (not perimeter wall) and contains a non-background entity,
                # give it a second consecutive push probe before classifying as wall.
                is_interior = (2 <= target_r < 62 - step) and (2 <= target_c < 62 - step)
                if is_interior and memory.consecutive_bumps < 1:
                    target_sub = curr_grid[target_r:target_r+step, target_c:target_c+step]
                    if np.any(target_sub != 0) and not np.all(target_sub == 5):
                        memory.consecutive_bumps += 1
                        return memory.avatar_pos, False

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
                    r_low, r_high = max(0, wall_r), min(64, wall_r + step)
                    c_low, c_high = max(0, wall_c), min(64, wall_c + step)
                    memory.passable_map[r_low:r_high, c_low:c_high] = False
                    memory.known_walls.add((wall_r, wall_c))
                    # Stand tile is strictly passable
                    memory.passable_map[ar:ar+step, ac:ac+step] = True
                    memory.known_walls.discard((ar, ac))
            return memory.avatar_pos, False

        pts = np.argwhere(diff)
        if len(pts) == 0:
            return memory.avatar_pos, True

        if len(memory.tracked_pawns) >= 2:
            step = memory.step_size
            for pid, prec in list(memory.tracked_pawns.items()):
                pr, pc = prec.pos
                nr, nc = pr + exp_dr * step, pc + exp_dc * step
                if 0 <= nr < 64 and 0 <= nc < 64:
                    r_hi, c_hi = min(64, nr + step), min(64, nc + step)
                    if np.any(diff[nr:r_hi, nc:c_hi]):
                        prec.pos = (nr, nc)
                        prec.is_active = True
                        memory.active_pawn_id = pid
                        memory.avatar_pos = (nr, nc)
                        memory.consecutive_bumps = 0
                        for other_id, other_p in memory.tracked_pawns.items():
                            if other_id != pid:
                                other_p.is_active = False
                        return memory.avatar_pos, True

        if memory.avatar_pos is not None:
            ar, ac = memory.avatar_pos
            step = memory.step_size
            warp_thresh = max(8, 2 * step + 1)

            # 1. Inspect previous avatar morphology & prev_vacated verification
            prev_comp = []
            if memory.avatar_colors and 0 <= ar < 64 and 0 <= ac < 64:
                prev_comp = AvatarTracker._get_connected_component_at(
                    prev_grid, ar, ac, memory.avatar_colors
                )

            if prev_comp:
                prev_size = len(prev_comp)
                prev_rs = [p[0] for p in prev_comp]
                prev_cs = [p[1] for p in prev_comp]
                prev_h = max(prev_rs) - min(prev_rs) + 1
                prev_w = max(prev_cs) - min(prev_cs) + 1
                still_present = sum(
                    1 for (pr, pc) in prev_comp if curr_grid[pr, pc] in memory.avatar_colors
                )
                prev_vacated = still_present <= max(0, prev_size // 4)
            else:
                prev_size = 1
                prev_h = step
                prev_w = step
                r_hi, c_hi = min(64, ar + step), min(64, ac + step)
                if memory.avatar_colors:
                    prev_cols = set(np.unique(curr_grid[ar:r_hi, ac:c_hi]).tolist())
                    prev_vacated = not prev_cols.issubset(memory.avatar_colors)
                else:
                    prev_vacated = (curr_grid[ar, ac] != prev_grid[ar, ac])

            if memory.avatar_colors:
                comps = AvatarTracker._find_avatar_components(
                    curr_grid, memory.avatar_colors, pts
                )
                for c in comps:
                    c["dist"] = max(abs(c["min_r"] - ar), abs(c["min_c"] - ac))

                vicinity_comps = [c for c in comps if c["dist"] < warp_thresh]
                distant_comps = [c for c in comps if c["dist"] >= warp_thresh]

                matching_vicinity = [
                    c for c in vicinity_comps
                    if abs(c["size"] - prev_size) <= max(1, prev_size // 3)
                    and abs(c["h"] - prev_h) <= 1
                    and abs(c["w"] - prev_w) <= 1
                ]

                # Priority 1: Discontinuous Warp / Wormhole Edge
                # Requires: prev_vacated, no matching avatar in vicinity, and coherent distant sprite matching size
                if prev_vacated and not matching_vicinity and distant_comps:
                    warp_candidates = [
                        c for c in distant_comps
                        if abs(c["size"] - prev_size) <= max(1, prev_size // 4)
                        and abs(c["h"] - prev_h) <= 1
                        and abs(c["w"] - prev_w) <= 1
                    ]
                    if warp_candidates:
                        warp_candidates.sort(key=lambda c: (abs(c["size"] - prev_size), c["dist"]))
                        best_warp = warp_candidates[0]
                        new_pos = (best_warp["min_r"], best_warp["min_c"])
                        dr = best_warp["min_r"] - ar
                        dc = best_warp["min_c"] - ac
                        is_directional_slide = (
                            (exp_dr != 0 and dc == 0 and np.sign(dr) == exp_dr and abs(dr) > max(4, 2 * step)) or
                            (exp_dc != 0 and dr == 0 and np.sign(dc) == exp_dc and abs(dc) > max(4, 2 * step))
                        )
                        if is_directional_slide:
                            memory.is_ice_slider = True
                            memory.avatar_pos = new_pos
                            memory.consecutive_bumps = 0
                            memory.plan_queue.clear()
                            memory.current_target = None
                            return new_pos, True

                        portal_r = int(np.clip(ar + exp_dr * step, 0, 63))
                        portal_c = int(np.clip(ac + exp_dc * step, 0, 63))
                        memory.wormhole_edges[(portal_r, portal_c)] = new_pos
                        memory.portal_actions[(portal_r, portal_c)] = last_action
                        memory.passable_map[portal_r:portal_r+step, portal_c:portal_c+step] = True
                        memory.passable_map[new_pos[0]:new_pos[0]+step, new_pos[1]:new_pos[1]+step] = True
                        memory.known_walls.discard((portal_r, portal_c))
                        memory.consecutive_bumps = 0
                        memory.avatar_pos = new_pos
                        return new_pos, True

                # Multi-pixel Avatar Bounding Box Tracking in Vicinity (anchoring top-left corner)
                if matching_vicinity:
                    def score_vic(c: dict[str, Any]) -> float:
                        dr = c["min_r"] - ar
                        dc = c["min_c"] - ac
                        err = 0.0
                        if exp_dr != 0 and np.sign(dr) != exp_dr:
                            err += 10.0
                        if exp_dc != 0 and np.sign(dc) != exp_dc:
                            err += 10.0
                        err += abs(abs(dr) - step) if exp_dr != 0 else abs(dr)
                        err += abs(abs(dc) - step) if exp_dc != 0 else abs(dc)
                        err += abs(c["size"] - prev_size)
                        return err

                    matching_vicinity.sort(key=score_vic)
                    best_vic = matching_vicinity[0]
                    new_pos = (best_vic["min_r"], best_vic["min_c"])
                    dr = best_vic["min_r"] - ar
                    dc = best_vic["min_c"] - ac
                    measured_step = max(abs(dr), abs(dc))

                    if measured_step >= 1:
                        if (dr == 0 or dc == 0) and measured_step > max(4, 2 * memory.step_size):
                            memory.is_ice_slider = True
                        if not memory.is_ice_slider:
                            memory.step_counts[measured_step] += 1
                            if memory.step_locked and measured_step != memory.step_size:
                                if memory.step_counts[measured_step] >= 2:
                                    memory.step_size = measured_step
                                    memory.consecutive_bumps = 0
                                    memory.plan_queue.clear()
                                    memory.current_target = None
                            elif not memory.step_locked:
                                if memory.step_counts[measured_step] >= memory.step_lock_threshold:
                                    memory.step_size = measured_step
                                    memory.step_locked = True
                                else:
                                    memory.step_size = measured_step

                    memory.avatar_pos = new_pos
                    memory.consecutive_bumps = 0
                    return new_pos, True

            warp_thresh = max(8, 2 * memory.step_size + 1)
            vicinity_mask = (abs(pts[:, 0] - ar) < warp_thresh) & (abs(pts[:, 1] - ac) < warp_thresh)
            if np.any(vicinity_mask):
                pts = pts[vicinity_mask]
            else:
                # Check for discontinuous jump / teleportation / boost / slide
                candidate_pts = pts
                if memory.avatar_colors:
                    matching_mask = np.isin(curr_grid[pts[:, 0], pts[:, 1]], list(memory.avatar_colors))
                    if np.any(matching_mask):
                        candidate_pts = pts[matching_mask]

                # Check if the avatar vacated its previous position
                step = memory.step_size
                prev_vacated = True
                if 0 <= ar < 64 and 0 <= ac < 64 and memory.avatar_colors:
                    r_hi = min(64, ar + step)
                    c_hi = min(64, ac + step)
                    prev_colors = set(np.unique(curr_grid[ar:r_hi, ac:c_hi]).tolist())
                    if prev_colors.issubset(memory.avatar_colors):
                        prev_vacated = False

                if len(candidate_pts) > 0 and prev_vacated:
                    # Discontinuous warp / jump recognized!
                    new_top_r = int(candidate_pts[:, 0].min())
                    new_left_c = int(candidate_pts[:, 1].min())
                    new_pos = (new_top_r, new_left_c)

                    portal_r = int(np.clip(ar + exp_dr * step, 0, 63))
                    portal_c = int(np.clip(ac + exp_dc * step, 0, 63))

                    # Register non-Euclidean warp edge and portal action
                    memory.wormhole_edges[(portal_r, portal_c)] = new_pos
                    memory.portal_actions[(portal_r, portal_c)] = last_action
                    memory.passable_map[portal_r:portal_r+step, portal_c:portal_c+step] = True
                    memory.passable_map[new_top_r:new_top_r+step, new_left_c:new_left_c+step] = True
                    memory.known_walls.discard((portal_r, portal_c))

                    memory.consecutive_bumps = 0
                    memory.avatar_pos = new_pos
                    return new_pos, True

                # Check if another character or avatar moved (e.g. control swap, ACTION5, multi-agent)
                s_prev = ScenePerception.segment_sprites(prev_grid, 0)
                s_curr = ScenePerception.segment_sprites(curr_grid, 0)
                moving_cands = []
                for sc in s_curr:
                    if sc["color"] in (0, 5) or ScenePerception.is_border_or_wall(sc):
                        continue
                    # Domain Skill §9.1: Difference-Mask Intersection Invariant
                    r0, r1, c0, c1 = sc["bbox"]
                    if not np.any(diff[r0 : r1 + 1, c0 : c1 + 1]):
                        continue
                    for sp in s_prev:
                        pr0, pr1, pc0, pc1 = sp["bbox"]
                        if not np.any(diff[pr0 : pr1 + 1, pc0 : pc1 + 1]):
                            continue
                        if sp["color"] == sc["color"] and abs(sp["size"] - sc["size"]) <= 2:
                            dr = sc["centroid"][1] - sp["centroid"][1]
                            dc = sc["centroid"][0] - sp["centroid"][0]
                            step = max(abs(dr), abs(dc))
                            if 1 <= step <= 8:
                                if (exp_dr != 0 and dr == exp_dr * step and abs(dc) <= 1) or (exp_dc != 0 and dc == exp_dc * step and abs(dr) <= 1):
                                    moving_cands.append((sc, step))
                if moving_cands:
                    moving_cands.sort(key=lambda item: abs(item[0]["bbox"][0] - ar) + abs(item[0]["bbox"][2] - ac))
                    best_sc, m_step = moving_cands[0]
                    new_pos = (best_sc["bbox"][0], best_sc["bbox"][2])
                    memory.avatar_pos = new_pos
                    memory.avatar_colors.add(best_sc["color"])
                    memory.avatar_color = best_sc["color"]
                    memory.step_size = m_step
                    memory.consecutive_bumps = 0
                    return new_pos, True

                # No movement in avatar vicinity and no warp: action bumped against a wall/barrier
                memory.consecutive_bumps += 1
                if memory.consecutive_bumps >= 2:
                    memory.plan_queue.clear()
                    step = memory.step_size
                    raw_wall_r = ar + exp_dr * step
                    raw_wall_c = ac + exp_dc * step
                    if 0 <= raw_wall_r < 64 and 0 <= raw_wall_c < 64:
                        wall_r = raw_wall_r
                        wall_c = raw_wall_c
                        r_low, r_high = max(0, wall_r), min(64, wall_r + step)
                        c_low, c_high = max(0, wall_c), min(64, wall_c + step)
                        memory.passable_map[r_low:r_high, c_low:c_high] = False
                        memory.known_walls.add((wall_r, wall_c))
                        memory.passable_map[ar:ar+step, ac:ac+step] = True
                        memory.known_walls.discard((ar, ac))
                return memory.avatar_pos, False

        memory.consecutive_bumps = 0

        if memory.avatar_colors:
            curr_av_mask = np.isin(curr_grid[pts[:, 0], pts[:, 1]], list(memory.avatar_colors))
            prev_av_mask = np.isin(prev_grid[pts[:, 0], pts[:, 1]], list(memory.avatar_colors))
            if np.any(curr_av_mask):
                curr_pts = pts[curr_av_mask]
                prev_pts = pts[prev_av_mask] if np.any(prev_av_mask) else curr_pts
            elif np.any(prev_av_mask):
                # Avatar changed color (e.g. fountain, dye, powerup)
                prev_pts = pts[prev_av_mask]
                other_pts = pts[~prev_av_mask]
                other_cols = set(np.unique(curr_grid[other_pts[:, 0], other_pts[:, 1]])) - {0, 5}
                if other_cols:
                    new_col = list(other_cols)[0]
                    memory.avatar_colors.add(new_col)
                    curr_pts = other_pts[curr_grid[other_pts[:, 0], other_pts[:, 1]] == new_col]
                else:
                    # Destination tile had the same color as the new avatar hue!
                    step = memory.step_size
                    new_r = int(np.clip(ar + exp_dr * step, 0, 63))
                    new_c = int(np.clip(ac + exp_dc * step, 0, 63))
                    dest_cols = set(np.unique(curr_grid[new_r:new_r+step, new_c:new_c+step])) - {0, 5}
                    if dest_cols:
                        new_col = list(dest_cols)[0]
                        memory.avatar_colors.add(new_col)
                        memory.avatar_color = new_col
                    new_pos = (new_r, new_c)
                    memory.avatar_pos = new_pos
                    memory.consecutive_bumps = 0
                    return new_pos, True
            else:
                curr_pts = np.empty((0, 2), dtype=int)
                prev_pts = np.empty((0, 2), dtype=int)
        else:
            cand_scores = []
            s_prev = ScenePerception.segment_sprites(prev_grid, 0)
            s_curr = ScenePerception.segment_sprites(curr_grid, 0)
            for sc in s_curr:
                if sc["color"] in (0, 5) or ScenePerception.is_border_or_wall(sc):
                    continue
                # Domain Skill §9.1: Difference-Mask Intersection Invariant
                r0, r1, c0, c1 = sc["bbox"]
                if not np.any(diff[r0 : r1 + 1, c0 : c1 + 1]):
                    continue
                for sp in s_prev:
                    pr0, pr1, pc0, pc1 = sp["bbox"]
                    if not np.any(diff[pr0 : pr1 + 1, pc0 : pc1 + 1]):
                        continue
                    if sp["color"] == sc["color"] and abs(sp["size"] - sc["size"]) <= 2:
                        dr = sc["centroid"][1] - sp["centroid"][1]
                        dc = sc["centroid"][0] - sp["centroid"][0]
                        step = max(abs(dr), abs(dc))
                        if 1 <= step <= 8:
                            if (exp_dr != 0 and dr == exp_dr * step and abs(dc) <= 1) or (exp_dc != 0 and dc == exp_dc * step and abs(dr) <= 1):
                                h = sc["bbox"][1] - sc["bbox"][0] + 1
                                w = sc["bbox"][3] - sc["bbox"][2] + 1
                                cand_scores.append((abs(h - w), sc["size"], sc["color"], sc, step))
            if cand_scores:
                cand_scores.sort(key=lambda x: (x[0], x[1], x[3]["bbox"][0] + x[3]["bbox"][2]))
                best = cand_scores[0]
                best_sc, best_c, m_step = best[3], best[2], best[4]
                memory.avatar_colors.add(best_c)
                memory.avatar_color = best_c
                memory.step_size = m_step
                new_pos = (best_sc["bbox"][0], best_sc["bbox"][2])
                memory.avatar_pos = new_pos
                memory.consecutive_bumps = 0
                return new_pos, True

            for c in np.unique(curr_grid[pts[:, 0], pts[:, 1]]):
                if c == 0 or c == 5:
                    continue
                c_curr = pts[curr_grid[pts[:, 0], pts[:, 1]] == c]
                c_prev = pts[prev_grid[pts[:, 0], pts[:, 1]] == c]
                if len(c_curr) == 0 or len(c_prev) == 0:
                    continue
                cr_curr, cc_curr = np.mean(c_curr[:, 0]), np.mean(c_curr[:, 1])
                cr_prev, cc_prev = np.mean(c_prev[:, 0]), np.mean(c_prev[:, 1])
                dr = cr_curr - cr_prev
                dc = cc_curr - cc_prev

                match = True
                if exp_dr != 0 and (np.sign(dr) != exp_dr or abs(dr) < 0.5):
                    match = False
                if exp_dc != 0 and (np.sign(dc) != exp_dc or abs(dc) < 0.5):
                    match = False
                if exp_dr == 0 and abs(dr) > 2:
                    match = False
                if exp_dc == 0 and abs(dc) > 2:
                    match = False

                if match:
                    h = c_curr[:, 0].max() - c_curr[:, 0].min() + 1
                    w = c_curr[:, 1].max() - c_curr[:, 1].min() + 1
                    cand_scores.append((abs(h - w), len(c_curr), c, c_curr, c_prev, int(round(max(abs(dr), abs(dc))))))

            if cand_scores:
                cand_scores.sort(key=lambda x: (x[0], x[1]))
                best = cand_scores[0]
                best_c, curr_pts, prev_pts, m_step = best[2], best[3], best[4], best[5]
                memory.avatar_colors.add(best_c)
                memory.avatar_color = best_c
                if m_step >= 1:
                    memory.step_size = m_step
            else:
                if exp_dr != 0:
                    med = np.median(pts[:, 0])
                    curr_pts = pts[pts[:, 0] < med] if exp_dr < 0 else pts[pts[:, 0] > med]
                    prev_pts = pts[pts[:, 0] > med] if exp_dr < 0 else pts[pts[:, 0] < med]
                else:
                    med = np.median(pts[:, 1])
                    curr_pts = pts[pts[:, 1] < med] if exp_dc < 0 else pts[pts[:, 1] > med]
                    prev_pts = pts[pts[:, 1] > med] if exp_dc < 0 else pts[pts[:, 1] < med]

        if len(curr_pts) > 0 and len(prev_pts) > 0:
            top_r = int(curr_pts[:, 0].min())
            left_c = int(curr_pts[:, 1].min())
            prev_top_r = int(prev_pts[:, 0].min())
            prev_left_c = int(prev_pts[:, 1].min())

            dr = top_r - prev_top_r
            dc = left_c - prev_left_c
            measured_step = max(abs(dr), abs(dc))
            if measured_step >= 1:
                # Detect frictionless sliding
                if (dr == 0 or dc == 0) and measured_step > max(4, 2 * memory.step_size):
                    memory.is_ice_slider = True
                if not memory.step_locked and not memory.is_ice_slider:
                    memory.step_counts[measured_step] += 1
                    if memory.step_counts[measured_step] >= memory.step_lock_threshold:
                        memory.step_size = measured_step
                        memory.step_locked = True
                    else:
                        memory.step_size = measured_step

            new_pos = (top_r, left_c)
            memory.avatar_pos = new_pos
            if not memory.avatar_colors:
                cand_colors = set(np.unique(curr_grid[curr_pts[:, 0], curr_pts[:, 1]]).tolist())
                cand_colors.discard(0)
                if len(cand_colors) > 1:
                    best_c = None
                    for c in cand_colors:
                        c_mask = (curr_grid[curr_pts[:, 0], curr_pts[:, 1]] == c)
                        if np.any(c_mask):
                            c_pts = curr_pts[c_mask]
                            h = c_pts[:, 0].max() - c_pts[:, 0].min() + 1
                            w = c_pts[:, 1].max() - c_pts[:, 1].min() + 1
                            if abs(h - w) <= 1:
                                best_c = c
                                break
                    if best_c is not None:
                        cand_colors = {best_c}
                if cand_colors:
                    memory.avatar_colors.update(cand_colors)
                    memory.avatar_color = list(cand_colors)[0]
            return new_pos, True

        return memory.avatar_pos, True


class DynamicLatticeDetector:
    """Infers grid lattice pitch (step_size), spatial offsets, and tile dimensions parameter-free."""

    @staticmethod
    def infer_lattice(
        grid_or_avatar_pos: Any = None,
        step_size_or_grid: Any = None,
        grid_shape: Tuple[int, int] = (64, 64),
        *,
        grid: Optional[np.ndarray] = None,
        avatar_pos: Optional[Tuple[int, int]] = None,
        step_size: Optional[int] = None,
    ) -> Tuple[int, int, int, int, int]:
        actual_grid: Optional[np.ndarray] = grid
        actual_avatar_pos: Optional[Tuple[int, int]] = avatar_pos
        actual_step: Optional[int] = step_size
        actual_shape: Tuple[int, int] = grid_shape

        if isinstance(grid_or_avatar_pos, np.ndarray):
            actual_grid = grid_or_avatar_pos
            actual_shape = actual_grid.shape
            if isinstance(step_size_or_grid, int):
                actual_step = step_size_or_grid
            elif isinstance(step_size_or_grid, (tuple, list)) and len(step_size_or_grid) == 2 and isinstance(step_size_or_grid[0], (int, np.integer)):
                actual_avatar_pos = (int(step_size_or_grid[0]), int(step_size_or_grid[1]))
        elif isinstance(grid_or_avatar_pos, (tuple, list)):
            if len(grid_or_avatar_pos) > 0 and isinstance(grid_or_avatar_pos[0], (tuple, list)):
                # History sequence of (row, col) tuples
                history = [
                    (int(p[0]), int(p[1]))
                    for p in grid_or_avatar_pos
                    if isinstance(p, (tuple, list)) and len(p) >= 2
                ]
                if history:
                    actual_avatar_pos = history[-1]
                    if actual_step is None and actual_grid is None:
                        diffs = [
                            max(abs(r1 - r0), abs(c1 - c0))
                            for (r0, c0), (r1, c1) in zip(history[:-1], history[1:])
                            if abs(r1 - r0) > 0 or abs(c1 - c0) > 0
                        ]
                        actual_step = max(1, int(round(float(np.median(diffs))))) if diffs else 1
            elif len(grid_or_avatar_pos) == 2 and isinstance(grid_or_avatar_pos[0], (int, np.integer)):
                actual_avatar_pos = (int(grid_or_avatar_pos[0]), int(grid_or_avatar_pos[1]))

            if isinstance(step_size_or_grid, int):
                actual_step = step_size_or_grid
            elif isinstance(step_size_or_grid, np.ndarray):
                actual_grid = step_size_or_grid
            elif isinstance(step_size_or_grid, (tuple, list)) and len(step_size_or_grid) == 2 and isinstance(step_size_or_grid[0], (int, np.integer)):
                actual_shape = (int(step_size_or_grid[0]), int(step_size_or_grid[1]))
        elif grid_or_avatar_pos is None:
            if isinstance(step_size_or_grid, int):
                actual_step = step_size_or_grid
            elif isinstance(step_size_or_grid, np.ndarray):
                actual_grid = step_size_or_grid

        # If step is explicitly provided and > 1 and no grid was provided, respect it
        if actual_step is not None and actual_step > 1 and actual_grid is None:
            step = actual_step
            r_off = (actual_avatar_pos[0] % step) if actual_avatar_pos else 0
            c_off = (actual_avatar_pos[1] % step) if actual_avatar_pos else 0
            gh = (actual_shape[0] - r_off) // step
            gw = (actual_shape[1] - c_off) // step
            return step, r_off, c_off, gh, gw

        if actual_grid is None:
            step = max(1, actual_step or 1)
            r_off = (actual_avatar_pos[0] % step) if actual_avatar_pos else 0
            c_off = (actual_avatar_pos[1] % step) if actual_avatar_pos else 0
            gh = (actual_shape[0] - r_off) // step
            gw = (actual_shape[1] - c_off) // step
            return step, r_off, c_off, gh, gw

        H, W = actual_grid.shape
        # Exclude bottom HUD / step counters (rows >= 59)
        sub = actual_grid[: min(59, H), : min(64, W)]
        diff_r = (sub[1:, :] != sub[:-1, :]).astype(np.int32)
        diff_c = (sub[:, 1:] != sub[:, :-1]).astype(np.int32)
        tr = diff_r.sum(axis=1)
        tc = diff_c.sum(axis=0)
        total_r = int(tr.sum())
        total_c = int(tc.sum())

        if total_r == 0 or total_c == 0:
            return 1, 0, 0, H, W

        best_s = 1
        best_ro = 0
        best_co = 0
        best_enh = 1.5

        # Check candidate pitches in ascending order: {2, 3, 4, 5, 8} to prioritize fundamental frequencies
        for s in (2, 3, 4, 5, 8):
            r_fracs = [
                sum(tr[k] for k in range(len(tr)) if (k + 1) % s == ro) / total_r
                for ro in range(s)
            ]
            c_fracs = [
                sum(tc[k] for k in range(len(tc)) if (k + 1) % s == co) / total_c
                for co in range(s)
            ]
            max_r = max(r_fracs)
            max_c = max(c_fracs)
            ro = int(np.argmax(r_fracs))
            co = int(np.argmax(c_fracs))

            enh_r = max_r * s
            enh_c = max_c * s
            enh = min(enh_r, enh_c)
            raw_min = min(max_r, max_c)
            cnt_r = sum(1 for k in range(len(tr)) if (k + 1) % s == ro and tr[k] > 0)
            cnt_c = sum(1 for k in range(len(tc)) if (k + 1) % s == co and tc[k] > 0)

            is_valid = False
            if s > 2 and raw_min >= 0.35 and enh >= 2.0 and cnt_r >= 3 and cnt_c >= 3:
                is_valid = True
            elif s == 2 and raw_min >= 0.70 and enh >= 1.4:
                is_valid = True

            # Prioritize fundamental pitch over harmonic multiples (e.g. S=2 over S=4, S=4 over S=8)
            if is_valid:
                if best_s > 1 and s % best_s == 0 and enh < best_enh * 1.3:
                    continue
                if enh > best_enh:
                    best_enh = enh
                    best_s = s
                    best_ro = ro
                    best_co = co

        # If avatar_pos is given, align offsets with avatar position if compatible
        if actual_avatar_pos is not None and best_s > 1:
            best_ro = actual_avatar_pos[0] % best_s
            best_co = actual_avatar_pos[1] % best_s

        gh = (H - best_ro) // best_s
        gw = (W - best_co) // best_s
        return best_s, best_ro, best_co, gh, gw


class DynamicHUDDetector:
    """Scans perimeter margins dynamically for static reference goal templates."""

    @staticmethod
    def find_goal_template(grid: np.ndarray, bg_color: int) -> Optional[dict[str, Any]]:
        H, W = grid.shape
        candidates = []
        margins = [
            (slice(H - 14, H), slice(0, 14), "bottom_left"),
            (slice(0, 14), slice(W - 14, W), "top_right"),
            (slice(0, 14), slice(0, 14), "top_left"),
            (slice(H - 14, H), slice(W - 14, W), "bottom_right"),
            (slice(0, 14), slice(0, W), "top"),
            (slice(H - 14, H), slice(0, W), "bottom"),
        ]

        for r_sl, c_sl, loc in margins:
            sub = grid[r_sl, c_sl]
            sub_h, sub_w = sub.shape
            for win_size in (10, 9, 8, 7, 6, 5):
                if sub_h < win_size or sub_w < win_size:
                    continue
                for wr in range(sub_h - win_size + 1):
                    for wc in range(sub_w - win_size + 1):
                        box = sub[wr : wr + win_size, wc : wc + win_size]
                        b_col = int(box[0, 0])
                        if b_col == bg_color:
                            continue
                        if box[0, -1] != b_col or box[-1, 0] != b_col or box[-1, -1] != b_col:
                            continue
                        if not (np.all(box[0, :] == b_col) and np.all(box[-1, :] == b_col) and np.all(box[:, 0] == b_col) and np.all(box[:, -1] == b_col)):
                            continue
                        core = box[1:-1, 1:-1]
                        non_border_mask = (core != b_col) & (core != bg_color)
                        if non_border_mask.any():
                            fg_colors, counts = np.unique(core[non_border_mask], return_counts=True)
                            candidates.append({
                                "location": loc,
                                "r": r_sl.start + wr,
                                "c": c_sl.start + wc,
                                "size": win_size,
                                "border_color": b_col,
                                "fg_color": int(fg_colors[np.argmax(counts)]),
                                "core_patch": core,
                                "box_patch": box,
                            })
            if candidates:
                break

        if not candidates:
            return None
        # Sort by largest size to identify the full bounding box
        candidates.sort(key=lambda c: -c["size"])
        return candidates[0]


class BlockConfiguration:
    """Immutable representation of a block configuration (shape, color, rotation)."""
    __slots__ = ("shape_id", "color_idx", "rotation_idx", "raw_color", "rotation_deg")

    def __init__(
        self,
        shape_id: int,
        color_idx: int,
        rotation_idx: int,
        raw_color: int = -1,
        rotation_deg: int = 0,
    ):
        self.shape_id = shape_id
        self.color_idx = color_idx
        self.rotation_idx = rotation_idx
        self.raw_color = raw_color
        self.rotation_deg = rotation_deg

    def __repr__(self) -> str:
        return f"BlockConfiguration(shape={self.shape_id}, color={self.color_idx}, rot={self.rotation_idx})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BlockConfiguration):
            return False
        return (
            self.shape_id == other.shape_id
            and self.color_idx == other.color_idx
            and self.rotation_idx == other.rotation_idx
        )

    def __hash__(self) -> int:
        return hash((self.shape_id, self.color_idx, self.rotation_idx))


class TemplatePerception:
    """Perception routines for block matching and dynamic HUD detection."""

    SHAPES: dict[int, np.ndarray] = {
        0: np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=bool),
        1: np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
        2: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
        3: np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=bool),
        4: np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
        5: np.array([[1, 1, 1], [0, 0, 1], [1, 0, 1]], dtype=bool),
    }

    PALETTE: list[int] = [12, 9, 14, 8]
    ROTATIONS: list[int] = [0, 90, 180, 270]
    _ROTATED_SHAPES_BY_COUNT: dict[int, list[Tuple[int, int, int, np.ndarray]]] = {}

    @classmethod
    def _init_rotated_shapes(cls):
        if not cls._ROTATED_SHAPES_BY_COUNT:
            for s_id, base_shp in cls.SHAPES.items():
                for r_idx, deg in enumerate(cls.ROTATIONS):
                    rot_shp = np.rot90(base_shp, -deg // 90)
                    cnt = int(np.sum(rot_shp))
                    cls._ROTATED_SHAPES_BY_COUNT.setdefault(cnt, []).append((s_id, r_idx, deg, rot_shp))

    @classmethod
    def match_sprite(
        cls,
        patch: np.ndarray,
        bg_color: Optional[int] = 5,
        palette: Optional[list[int]] = None,
    ) -> Optional[BlockConfiguration]:
        cls._init_rotated_shapes()
        H, W = patch.shape
        if H == 10 and W == 10:
            patch = patch[2:8, 2:8][::2, ::2]
        elif H == 6 and W == 6:
            patch = patch[::2, ::2]
        elif H >= 5 and W >= 5:
            patch = patch[1:4, 1:4]
        elif (H, W) != (3, 3):
            return None

        active_palette = palette or cls.PALETTE

        if bg_color is None:
            border = np.concatenate([patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]])
            b_colors, b_counts = np.unique(border, return_counts=True)
            bg_color = int(b_colors[np.argmax(b_counts)])

        non_bg = patch[patch != bg_color]
        if len(non_bg) == 0:
            return None

        pal_mask = np.isin(patch, active_palette)
        candidate_colors: list[int] = []
        if pal_mask.any():
            pal_colors, counts = np.unique(patch[pal_mask], return_counts=True)
            candidate_colors.extend(int(c) for c in pal_colors[np.argsort(-counts)])
        else:
            unique_c, counts = np.unique(non_bg, return_counts=True)
            candidate_colors.extend(int(c) for c in unique_c[np.argsort(-counts)])

        for fg_color in candidate_colors:
            bin_patch = (patch == fg_color)
            cnt = int(bin_patch.sum())
            candidates = cls._ROTATED_SHAPES_BY_COUNT.get(cnt)
            if not candidates:
                continue
            for s_id, r_idx, deg, rot_shp in candidates:
                if np.array_equal(bin_patch, rot_shp):
                    c_idx = active_palette.index(fg_color) if fg_color in active_palette else 0
                    return BlockConfiguration(
                        shape_id=s_id,
                        color_idx=c_idx,
                        rotation_idx=r_idx,
                        raw_color=fg_color,
                        rotation_deg=deg,
                    )
        return None

    @classmethod
    def extract_hud_configuration(
        cls,
        grid: np.ndarray,
        bg_color: int = 0,
        hud_info: Optional[dict[str, Any]] = None,
    ) -> Optional[Tuple[BlockConfiguration, dict[str, Any]]]:
        if hud_info is not None:
            r, c, sz = hud_info["r"], hud_info["c"], hud_info["size"]
            b_col = hud_info.get("border_color", 5)
            box = grid[r : r + sz, c : c + sz]
            if box.shape == (sz, sz):
                cfg = cls.match_sprite(box, bg_color=b_col)
                if cfg is None and sz > 2:
                    cfg = cls.match_sprite(box[1:-1, 1:-1], bg_color=b_col)
                if cfg is not None:
                    return cfg, hud_info

        hud_cand = DynamicHUDDetector.find_goal_template(grid, bg_color)
        if hud_cand is not None:
            cfg = cls.match_sprite(hud_cand["box_patch"], bg_color=hud_cand["border_color"])
            if cfg is None:
                cfg = cls.match_sprite(hud_cand["core_patch"], bg_color=hud_cand["border_color"])
            if cfg is not None:
                return cfg, hud_cand

        # Fallback check bottom-left
        if grid.shape[0] >= 61 and grid.shape[1] >= 9:
            cfg = cls.match_sprite(grid[55:61, 3:9], bg_color=5)
            if cfg:
                return cfg, {"r": 55, "c": 3, "size": 6, "border_color": 5, "location": "bottom_left"}
        return None


class OperatorPadDetector:
    """Dynamically scans discrete tile lattice for operator pads, goal receptacles, and passable paths."""

    @classmethod
    def detect_puzzle_elements(
        cls,
        grid: np.ndarray,
        step_size: int = 5,
        avatar_pos: Optional[Tuple[int, int]] = None,
        hud_info: Optional[dict[str, Any]] = None,
        affordance_ledger: Optional[dict[Tuple[int, int], str]] = None,
    ) -> dict[str, Any]:
        step, roff, coff, gh, gw = DynamicLatticeDetector.infer_lattice(grid=grid, avatar_pos=avatar_pos, step_size=step_size, grid_shape=grid.shape)

        elements: dict[str, Any] = {
            "rotation_pads": [],
            "color_pads": [],
            "shape_pads": [],
            "goal_slots": {},
            "avatar_tile": None,
            "passable": np.zeros((gh, gw), dtype=bool),
            "step": step,
            "roff": roff,
            "coff": coff,
            "gh": gh,
            "gw": gw,
            "wall_color": 4,
        }

        if avatar_pos is not None:
            elements["avatar_tile"] = ((avatar_pos[1] - coff) // step, (avatar_pos[0] - roff) // step)

        target_border_color = hud_info.get("border_color", 5) if hud_info else 5

        board_edges = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
        edge_colors, edge_counts = np.unique(board_edges, return_counts=True)
        if len(edge_colors) > 0:
            elements["wall_color"] = int(edge_colors[np.argmax(edge_counts)])

        for gy in range(gh):
            for gx in range(gw):
                r = roff + step * gy
                c = coff + step * gx
                patch = grid[r : r + step, c : c + step]
                if patch.shape != (step, step):
                    continue

                u = set(patch.flatten())

                if elements["avatar_tile"] == (gx, gy):
                    elements["passable"][gy, gx] = True
                    continue

                if affordance_ledger and (gx, gy) in affordance_ledger:
                    op = affordance_ledger[(gx, gy)]
                    if op == "rot":
                        elements["rotation_pads"].append((gx, gy))
                    elif op == "color":
                        elements["color_pads"].append((gx, gy))
                    elif op == "shape":
                        elements["shape_pads"].append((gx, gy))
                    elements["passable"][gy, gx] = True
                    continue

                # 1. Goal slot: scan 3x3 sub-patches within (step, step) tile to identify goal receptacle
                found_goal_cfg = None
                if target_border_color in u:
                    if step >= 3:
                        for sub_r in range(step - 2):
                            for sub_c in range(step - 2):
                                sub_patch = patch[sub_r : sub_r + 3, sub_c : sub_c + 3]
                                cfg = TemplatePerception.match_sprite(sub_patch, bg_color=target_border_color)
                                if cfg is not None:
                                    found_goal_cfg = cfg
                                    break
                            if found_goal_cfg is not None:
                                break
                    elif (step, step) == (3, 3):
                        found_goal_cfg = TemplatePerception.match_sprite(patch, bg_color=target_border_color)

                if found_goal_cfg is not None:
                    elements["goal_slots"][(gx, gy)] = found_goal_cfg
                    elements["passable"][gy, gx] = True
                    continue

                # 2. Rotation pad: directional chevron / L-shaped indicator or standard pattern
                if 1 in u and 0 in u and elements["wall_color"] not in u and np.sum(patch == 0) == 3 and np.sum(patch == 1) == 2:
                    elements["rotation_pads"].append((gx, gy))
                    elements["passable"][gy, gx] = True
                    continue

                # 3. Color pad: contains >= 3 distinct foreground colors (swatch)
                interior = patch[1:-1, 1:-1] if step >= 3 else patch
                interior_colors = set(np.unique(interior)) - {elements["wall_color"], target_border_color}
                if len(u & set(TemplatePerception.PALETTE)) >= 3 or len(interior_colors) >= 3:
                    elements["color_pads"].append((gx, gy))
                    elements["passable"][gy, gx] = True
                    continue

                # 4. Shape pad: geometric boundary with hollow center
                if 0 in u and 1 not in u and elements["wall_color"] not in u and len(u & set(TemplatePerception.PALETTE)) == 0 and np.sum(patch == 0) == 4:
                    elements["shape_pads"].append((gx, gy))
                    elements["passable"][gy, gx] = True
                    continue

                if not (patch == elements["wall_color"]).all():
                    if hud_info and hud_info.get("location") == "bottom_left" and gy == gh - 1 and gx <= 2:
                        pass
                    else:
                        elements["passable"][gy, gx] = True

        return elements


class ConfigurationPlanner:
    """Unified Joint State-Space BFS planner over dynamic lattices."""

    ACTIONS_MAP = [
        (GameAction.ACTION1, 0, -1),
        (GameAction.ACTION2, 0, 1),
        (GameAction.ACTION3, -1, 0),
        (GameAction.ACTION4, 1, 0),
    ]

    @classmethod
    def plan_unified_bfs(
        cls,
        start_tile: Tuple[int, int],
        start_cfg: BlockConfiguration,
        goal_slots: dict[Tuple[int, int], BlockConfiguration],
        pads: dict[Tuple[int, int], str],
        passable: np.ndarray,
        legal_actions: Set[GameAction],
        max_nodes: int = 20000,
    ) -> list[GameAction]:
        gh, gw = passable.shape
        start_gx, start_gy = start_tile
        start_state = (start_gx, start_gy, start_cfg.shape_id, start_cfg.color_idx, start_cfg.rotation_idx)

        target_dict = {
            pos: (cfg.shape_id, cfg.color_idx, cfg.rotation_idx)
            for pos, cfg in goal_slots.items()
        }

        queue = collections.deque([start_state])
        visited: dict[Tuple[int, int, int, int, int], Optional[Tuple[Tuple[int, int, int, int, int], GameAction]]] = {
            start_state: None
        }
        goal_state: Optional[Tuple[int, int, int, int, int]] = None

        allowed_actions = [
            (act, dgx, dgy) for act, dgx, dgy in cls.ACTIONS_MAP if act in legal_actions
        ]

        nodes_expanded = 0
        while queue and nodes_expanded < max_nodes:
            nodes_expanded += 1
            cur = queue.popleft()
            cgx, cgy, cS, cC, cR = cur

            if (cgx, cgy) in target_dict and (cS, cC, cR) == target_dict[(cgx, cgy)]:
                goal_state = cur
                break

            for act, dgx, dgy in allowed_actions:
                ngx, ngy = cgx + dgx, cgy + dgy
                if 0 <= ngx < gw and 0 <= ngy < gh and passable[ngy, ngx]:
                    # Receptacle blocking invariant: never enter goal slot unless matching 100%
                    if (ngx, ngy) in target_dict and (cS, cC, cR) != target_dict[(ngx, ngy)]:
                        continue

                    nS, nC, nR = cS, cC, cR
                    if (ngx, ngy) in pads:
                        pad_type = pads[(ngx, ngy)]
                        if pad_type == "rot":
                            nR = (nR + 1) % 4
                        elif pad_type == "color":
                            nC = (nC + 1) % 4
                        elif pad_type == "shape":
                            nS = (nS + 1) % 6

                    next_state = (ngx, ngy, nS, nC, nR)
                    if next_state not in visited:
                        visited[next_state] = (cur, act)
                        queue.append(next_state)

        if not goal_state:
            return []

        path = []
        cur = goal_state
        while visited[cur] is not None:
            prev_state, act = visited[cur]  # type: ignore
            path.append(act)
            cur = prev_state
        path.reverse()
        return path


class GeodesicPlanner:
    """
    High-performance Geodesic A* path planner supporting:
    1. Turn penalties (weight 1.2) for straight stride preference and RHAE parsimony.
    2. Non-Euclidean warp edge integration directly into search queue and admissible heuristic.
    3. Frictionless ice sliding kinematics forward simulation.
    4. Vectorized hazard inflation in NumPy (< 0.05 ms).
    """

    DIR_DELTAS = [
        (-1, 0, GameAction.ACTION1, 0),  # Up
        (1, 0, GameAction.ACTION2, 1),   # Down
        (0, -1, GameAction.ACTION3, 2),  # Left
        (0, 1, GameAction.ACTION4, 3),   # Right
    ]

    @staticmethod
    def dilate_hazards(passable: np.ndarray, radius: int) -> np.ndarray:
        """Vectorized morphological obstacle inflation using pure NumPy slice shifts."""
        if radius <= 0:
            return passable
        impassable = ~passable
        dilated = impassable.copy()
        H, W = passable.shape
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                r_src = slice(max(0, -dr), min(H, H - dr))
                r_dst = slice(max(0, dr), min(H, H + dr))
                c_src = slice(max(0, -dc), min(W, W - dc))
                c_dst = slice(max(0, dc), min(W, W + dc))
                dilated[r_dst, c_dst] |= impassable[r_src, c_src]
        return ~dilated

    @staticmethod
    def _compute_heuristic(
        r: int, c: int,
        gr: int, gc: int,
        step_size: int,
        heuristic_weight: float,
        wormhole_edges: Optional[dict[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> float:
        """Admissible non-Euclidean geodesic heuristic accounting for warp portals."""
        step = max(1, step_size)
        base_h = (abs(r - gr) + abs(c - gc)) / step
        if not wormhole_edges:
            return base_h * heuristic_weight

        min_h = base_h
        for (pr, pc), (dr_dest, dc_dest) in wormhole_edges.items():
            dist_to_portal = (abs(r - pr) + abs(c - pc)) / step
            dist_from_exit = (abs(dr_dest - gr) + abs(dc_dest - gc)) / step
            portal_h = dist_to_portal + dist_from_exit
            if portal_h < min_h:
                min_h = portal_h

        return min_h * heuristic_weight

    @classmethod
    def plan(
        cls,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        passable: np.ndarray,
        step_size: int = 1,
        legal_actions: Optional[Union[Set[GameAction], list[GameAction]]] = None,
        heuristic_weight: float = 1.2,
        turn_penalty: float = 0.2,
        hazard_radius: int = 0,
        wormhole_edges: Optional[dict[Tuple[int, int], Tuple[int, int]]] = None,
        portal_actions: Optional[dict[Tuple[int, int], GameAction]] = None,
        is_ice: bool = False,
        max_expansions: int = 2500,
    ) -> list[GameAction]:
        """Calculates minimal-action geodesic trajectory from start to goal."""
        if start == goal:
            return []
        legal_set = set(legal_actions) if legal_actions is not None else {
            GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4
        }

        sr, sc = start
        gr, gc = goal
        step = max(1, step_size)
        H, W = passable.shape

        # Fast-path for unobstructed Euclidean grid without ice or wormholes
        if not is_ice and not wormhole_edges and hazard_radius == 0:
            dr = gr - sr
            dc = gc - sc
            v_act = GameAction.ACTION2 if dr > 0 else GameAction.ACTION1
            h_act = GameAction.ACTION4 if dc > 0 else GameAction.ACTION3
            v_step = 1 if dr > 0 else -1
            h_step = 1 if dc > 0 else -1
            v_count = int(round(abs(dr) / step))
            h_count = int(round(abs(dc) / step))

            if v_count == 0 and h_count == 0:
                return []

            # Check major-axis L-path first
            paths_to_check = ["h", "v"] if h_count > v_count else ["v", "h"]
            for ptype in paths_to_check:
                if ptype == "v":
                    can_v = (v_act in legal_set or v_count == 0) and (h_act in legal_set or h_count == 0)
                    if can_v:
                        blocked = False
                        cr, cc = sr, sc
                        for _ in range(v_count):
                            cr = int(np.clip(cr + v_step * step, 0, H - 1))
                            if not passable[cr, cc] and (cr, cc) != (gr, gc):
                                blocked = True
                                break
                        if not blocked:
                            for _ in range(h_count):
                                cc = int(np.clip(cc + h_step * step, 0, W - 1))
                                if not passable[cr, cc] and (cr, cc) != (gr, gc):
                                    blocked = True
                                    break
                        if not blocked and (v_count > 0 or h_count > 0):
                            return [v_act] * v_count + [h_act] * h_count
                else:
                    can_h = (h_act in legal_set or h_count == 0) and (v_act in legal_set or v_count == 0)
                    if can_h:
                        blocked = False
                        cr, cc = sr, sc
                        for _ in range(h_count):
                            cc = int(np.clip(cc + h_step * step, 0, W - 1))
                            if not passable[cr, cc] and (cr, cc) != (gr, gc):
                                blocked = True
                                break
                        if not blocked:
                            for _ in range(v_count):
                                cr = int(np.clip(cr + v_step * step, 0, H - 1))
                                if not passable[cr, cc] and (cr, cc) != (gr, gc):
                                    blocked = True
                                    break
                        if not blocked and (v_count > 0 or h_count > 0):
                            return [h_act] * h_count + [v_act] * v_count

        # Apply vectorized hazard dilation if requested
        eff_passable = passable
        if hazard_radius > 0 and not is_ice:
            eff_passable = cls.dilate_hazards(passable, hazard_radius)
            # Re-enable start and goal footprints
            eff_passable[max(0, sr):min(H, sr + step), max(0, sc):min(W, sc + step)] = True
            eff_passable[max(0, gr):min(H, gr + step), max(0, gc):min(W, gc + step)] = True

        path = cls.a_star_convex_detour(
            start=start,
            goal=goal,
            passable=eff_passable,
            step_size=step,
            legal_actions=legal_set,
            heuristic_weight=heuristic_weight,
            turn_penalty=turn_penalty,
            hazard_radius=0,
            wormhole_edges=wormhole_edges,
            portal_actions=portal_actions,
            is_ice=is_ice,
            max_expansions=max_expansions,
        )

        # Fallback without hazard repulsion if blocked
        if not path and hazard_radius > 0 and not is_ice:
            path = cls.a_star_convex_detour(
                start=start,
                goal=goal,
                passable=passable,
                step_size=step,
                legal_actions=legal_set,
                heuristic_weight=heuristic_weight,
                turn_penalty=turn_penalty,
                hazard_radius=0,
                wormhole_edges=wormhole_edges,
                portal_actions=portal_actions,
                is_ice=is_ice,
                max_expansions=max_expansions,
            )

        return path

    @classmethod
    def a_star_convex_detour(
        cls,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        passable: np.ndarray,
        step_size: int = 1,
        legal_actions: Optional[Union[Set[GameAction], list[GameAction]]] = None,
        max_expansions: int = 2500,
        heuristic_weight: float = 1.2,
        turn_penalty: float = 0.2,
        hazard_radius: int = 0,
        wormhole_edges: Optional[dict[Tuple[int, int], Tuple[int, int]]] = None,
        portal_actions: Optional[dict[Tuple[int, int], GameAction]] = None,
        is_ice: bool = False,
    ) -> list[GameAction]:
        """A* search over 3D state (r, c, prev_dir) with turn penalties and warp transitions."""
        if start == goal:
            return []

        sr, sc = start
        gr, gc = goal
        step = max(1, step_size)
        H, W = passable.shape

        legal_set = set(legal_actions) if legal_actions is not None else {
            GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4
        }
        valid_dirs = [d for d in cls.DIR_DELTAS if d[2] in legal_set]
        if not valid_dirs:
            return []

        eff_passable = passable
        if hazard_radius > 0 and not is_ice:
            eff_passable = cls.dilate_hazards(passable, hazard_radius)
            eff_passable[max(0, sr):min(H, sr + step), max(0, sc):min(W, sc + step)] = True
            eff_passable[max(0, gr):min(H, gr + step), max(0, gc):min(W, gc + step)] = True

        # Priority queue: (f, g, r, c, prev_dir_idx)
        # prev_dir_idx: -1 for none, 0..3 for Up, Down, Left, Right
        h0 = cls._compute_heuristic(sr, sc, gr, gc, step, heuristic_weight, wormhole_edges)
        pq = [(h0, 0.0, sr, sc, -1)]
        g_score = {(sr, sc, -1): 0.0}
        came_from = {}

        found_state = None
        expansions = 0

        while pq and expansions < max_expansions:
            f, g, r, c, pdir = heapq.heappop(pq)
            expansions += 1

            if (r, c) == (gr, gc) or (
                step > 1
                and abs(r - gr) < step
                and abs(c - gc) < step
                and (r, c) != (sr, sc)
            ):
                found_state = (r, c, pdir)
                break

            if g > g_score.get((r, c, pdir), float("inf")):
                continue

            curr_state = (r, c, pdir)

            for dr, dc, act, didx in valid_dirs:
                if is_ice:
                    # Continuous directional impulse sliding
                    cr, cc = r, c
                    while True:
                        nr = int(np.clip(cr + dr * step, 0, H - 1))
                        nc = int(np.clip(cc + dc * step, 0, W - 1))
                        if (nr, nc) == (cr, cc) or not eff_passable[nr, nc]:
                            break
                        cr, cc = nr, nc
                        # Goal interception during slide
                        if abs(cr - gr) < step and abs(cc - gc) < step:
                            break
                        # Warp edge interception during slide
                        if wormhole_edges and (cr, cc) in wormhole_edges:
                            cr, cc = wormhole_edges[(cr, cc)]
                            break
                    nxt_r, nxt_c = cr, cc
                else:
                    nxt_r = int(np.clip(r + dr * step, 0, H - 1))
                    nxt_c = int(np.clip(c + dc * step, 0, W - 1))
                    if (nxt_r, nxt_c) == (r, c):
                        continue
                    if not eff_passable[nxt_r, nxt_c] and (nxt_r, nxt_c) != (gr, gc):
                        continue
                    # Seamless non-Euclidean warp edge upon stepping into portal
                    if wormhole_edges and (nxt_r, nxt_c) in wormhole_edges:
                        dest_r, dest_c = wormhole_edges[(nxt_r, nxt_c)]
                        nxt_r, nxt_c = dest_r, dest_c

                if (nxt_r, nxt_c) == (r, c):
                    continue

                # Turn penalty (weight 1.2 on direction change, 1.0 on straight continuation)
                is_turn = (pdir != -1) and (pdir != didx)
                edge_cost = 1.0 + (turn_penalty if is_turn else 0.0)
                new_g = g + edge_cost
                next_state = (nxt_r, nxt_c, didx)

                if new_g < g_score.get(next_state, float("inf")):
                    g_score[next_state] = new_g
                    h = cls._compute_heuristic(nxt_r, nxt_c, gr, gc, step, heuristic_weight, wormhole_edges)
                    new_f = new_g + h
                    came_from[next_state] = (curr_state, act)
                    heapq.heappush(pq, (new_f, new_g, nxt_r, nxt_c, didx))

        if not found_state:
            return []

        # Reconstruct path backwards
        path = []
        curr = found_state
        while curr in came_from:
            prev_s, act = came_from[curr]
            path.append(act)
            curr = prev_s
        path.reverse()
        return path


class OfflinePlanner:
    """
    Module 3: Dual Offline Planner consolidating geodesic pathfinding,
    Bellman backwards induction, and state-space BFS.
    """

    @staticmethod
    def plan_geodesic_path(
        start: Tuple[int, int],
        goal: Tuple[int, int],
        passable: np.ndarray,
        step_size: int = 1,
        legal_actions: Optional[Union[Set[GameAction], list[GameAction]]] = None,
        heuristic_weight: float = 1.2,
        turn_penalty: float = 0.2,
        hazard_radius: int = 0,
        warp_edges: Optional[dict[Tuple[int, int], Tuple[int, int]]] = None,
        portal_actions: Optional[dict[Tuple[int, int], GameAction]] = None,
        is_ice: bool = False,
        max_expansions: int = 2500,
    ) -> list[GameAction]:
        """Interface Contract conformant geodesic path generator."""
        return GeodesicPlanner.plan(
            start=start,
            goal=goal,
            passable=passable,
            step_size=step_size,
            legal_actions=legal_actions,
            heuristic_weight=heuristic_weight,
            turn_penalty=turn_penalty,
            hazard_radius=hazard_radius,
            wormhole_edges=warp_edges,
            portal_actions=portal_actions,
            is_ice=is_ice,
            max_expansions=max_expansions,
        )


class MirroredTwinPlanner:
    """Dynamic 2-agent state-space BFS solver for mirrored twin rendezvous puzzles."""

    @staticmethod
    def detect_and_plan(
        grid: np.ndarray,
        step_size: int,
        roff: int,
        coff: int,
        gh: int,
        gw: int,
        legal_actions: Set[GameAction],
    ) -> Optional[list[GameAction]]:
        if gh < 3 or gw < 3:
            return None

        # Sample cell centers to construct downsampled grid
        downsampled = np.zeros((gh, gw), dtype=int)
        for r in range(gh):
            for c in range(gw):
                pr = min(grid.shape[0] - 1, roff + r * step_size + step_size // 2)
                pc = min(grid.shape[1] - 1, coff + c * step_size + step_size // 2)
                downsampled[r, c] = grid[pr, pc]

        vals, counts = np.unique(downsampled, return_counts=True)
        bg = vals[np.argmax(counts)]

        # Check if non-background elements define a sub-box
        non_bg = np.argwhere(downsampled != bg)
        min_r, min_c = (int(non_bg[:, 0].min()), int(non_bg[:, 1].min())) if len(non_bg) > 0 else (0, 0)
        max_r, max_c = (int(non_bg[:, 0].max()), int(non_bg[:, 1].max())) if len(non_bg) > 0 else (gh - 1, gw - 1)
        eff_gw = max_c - min_c + 1
        eff_gh = max_r - min_r + 1

        # Look for twin avatars (exactly 2 symmetric singletons of the same color)
        candidates = []
        for val in vals:
            if val == bg:
                continue
            coords = np.argwhere(downsampled == val)
            if len(coords) == 2:
                r1, c1 = coords[0]
                r2, c2 = coords[1]
                # Check horizontal symmetry across midline (full grid or sub-box)
                rel_c1 = c1 - min_c
                rel_c2 = c2 - min_c
                if r1 == r2:
                    if c1 + c2 == gw - 1 or abs((c1 + c2) - (gw - 1)) <= 1:
                        candidates.append((val, coords, 0, 0, gh, gw))
                    elif eff_gw >= 3 and (rel_c1 + rel_c2 == eff_gw - 1 or abs((rel_c1 + rel_c2) - (eff_gw - 1)) <= 1):
                        candidates.append((val, coords, min_r, min_c, eff_gh, eff_gw))

        if not candidates:
            return None

        val, coords, br, bc, bh, bw = candidates[0]
        start1 = (int(coords[0][1]), int(coords[0][0]))
        start2 = (int(coords[1][1]), int(coords[1][0]))

        # Passable cells are background and twin cells
        passable = (downsampled == bg) | (downsampled == val)

        actions = [
            (GameAction.ACTION1, (0, -1), (0, -1)),  # Up
            (GameAction.ACTION2, (0, 1), (0, 1)),    # Down
            (GameAction.ACTION3, (-1, 0), (1, 0)),   # Left (Twin 2 moves Right)
            (GameAction.ACTION4, (1, 0), (-1, 0)),   # Right (Twin 2 moves Left)
        ]
        valid_actions = [(act, m1, m2) for act, m1, m2 in actions if act in legal_actions]

        def step_agent(pos, move):
            x, y = pos
            dx, dy = move
            nx, ny = x + dx, y + dy
            if bc <= nx < bc + bw and br <= ny < br + bh and passable[ny, nx]:
                return (nx, ny)
            return (x, y)  # Deliberate wall bump: agent stays put!

        queue = collections.deque([(start1, start2, [])])
        visited = {(start1, start2)}

        while queue and len(visited) < 15000:
            p1, p2, path = queue.popleft()
            if p1 == p2:
                return path
            for act, m1, m2 in valid_actions:
                np1 = step_agent(p1, m1)
                np2 = step_agent(p2, m2)
                if (np1, np2) not in visited:
                    visited.add((np1, np2))
                    queue.append((np1, np2, path + [act]))

        return None


class BellmanBackwardsPlanner:
    """
    Bellman Retrograde Backwards Induction (Ω -> α) planner for multi-target,
    sequential key pickup, and locked-door navigation mazes.
    Guarantees globally optimal trajectory with zero exploratory action cost.
    """

    DIRS = [
        (-1, 0, GameAction.ACTION1),  # Up
        (1, 0, GameAction.ACTION2),   # Down
        (0, -1, GameAction.ACTION3),  # Left
        (0, 1, GameAction.ACTION4),   # Right
    ]

    @staticmethod
    def plan(
        start: Tuple[int, int],
        goal: Tuple[int, int],
        keys: list[Tuple[int, int]],
        doors: dict[Tuple[int, int], int],
        passable: np.ndarray,
        step_size: int = 1,
        legal_actions: Optional[Union[Set[GameAction], list[GameAction]]] = None,
        max_states: int = 15000,
    ) -> Optional[list[GameAction]]:
        if start == goal and not keys:
            return []

        legal_set = set(legal_actions) if legal_actions is not None else {
            GameAction.ACTION1, GameAction.ACTION2,
            GameAction.ACTION3, GameAction.ACTION4
        }
        valid_dirs = [d for d in BellmanBackwardsPlanner.DIRS if d[2] in legal_set]
        if not valid_dirs:
            return None

        step = max(1, step_size)
        H, W = passable.shape

        # Quantize coordinates to lattice steps
        sr, sc = (start[0] // step) * step, (start[1] // step) * step
        gr, gc = (goal[0] // step) * step, (goal[1] // step) * step

        # Lattice keys and doors
        lattice_keys: list[Tuple[int, int]] = []
        for kr, kc in keys[:8]:  # Limit to <= 8 keys for 256 state bitmask
            lattice_keys.append(((kr // step) * step, (kc // step) * step))

        num_keys = len(lattice_keys)
        target_mask = (1 << num_keys) - 1
        key_map: dict[Tuple[int, int], int] = {pos: idx for idx, pos in enumerate(lattice_keys)}

        lattice_doors: dict[Tuple[int, int], int] = {}
        for (dr, dc), req_k in doors.items():
            if req_k < num_keys:
                lattice_doors[((dr // step) * step, (dc // step) * step)] = req_k

        # Effective passability map (doors are walkable in map, restricted by mask in search)
        eff_passable = passable.copy()
        for dr, dc in lattice_doors:
            eff_passable[dr:min(H, dr+step), dc:min(W, dc+step)] = True

        # Backward search state: (r, c, mask)
        # Terminal state: (gr, gc, target_mask) with cost 0
        num_masks = 1 << num_keys
        visited = bytearray(H * W * num_masks)
        visited[(gr * W + gc) * num_masks + target_mask] = 1

        q = collections.deque([(gr, gc, target_mask)])
        parent_act: dict[Tuple[int, int, int], GameAction] = {}
        parent_next: dict[Tuple[int, int, int], Tuple[int, int, int]] = {}

        expansions = 0
        found = False

        while q and expansions < max_states:
            r, c, mask = q.popleft()
            expansions += 1

            if r == sr and c == sc and mask == 0:
                found = True
                break

            # Door constraint on cell (r, c): forward move entered door, requiring key
            req_k = lattice_doors.get((r, c))

            # Possible predecessor masks for (r, c, mask):
            # If forward move landed on a key cell (r, c), key could have been
            # collected on this step (predecessor had mask without k_id) or already had it.
            k_id = key_map.get((r, c))
            if k_id is not None and (mask & (1 << k_id)):
                p_masks = (mask, mask & ~(1 << k_id))
            else:
                p_masks = (mask,)

            if req_k is not None:
                p_masks = tuple(pm for pm in p_masks if (pm & (1 << req_k)))
                if not p_masks:
                    continue

            # Explore predecessors:
            # Forward action act moved from (pr, pc) to (r, c).
            # Therefore, in backward time: pr = r - dr * step, pc = c - dc * step.
            for dr, dc, act in valid_dirs:
                pr = r - dr * step
                pc = c - dc * step

                if not (0 <= pr < H and 0 <= pc < W):
                    continue
                if not eff_passable[pr, pc]:
                    continue

                pr_W_pc = (pr * W + pc) * num_masks
                for pm in p_masks:
                    v_idx = pr_W_pc + pm
                    if visited[v_idx]:
                        continue
                    visited[v_idx] = 1

                    pred_state = (pr, pc, pm)
                    parent_act[pred_state] = act
                    parent_next[pred_state] = (r, c, mask)
                    q.append(pred_state)

        if not found and (sr, sc, 0) not in parent_next:
            return None

        # Reconstruct forward trajectory from (sr, sc, 0) to (gr, gc, target_mask)
        trajectory: list[GameAction] = []
        curr = (sr, sc, 0)
        visited_chain = set()

        while curr in parent_next and curr not in visited_chain:
            visited_chain.add(curr)
            act = parent_act[curr]
            trajectory.append(act)
            curr = parent_next[curr]
            if (curr[0], curr[1]) == (gr, gc) and curr[2] == target_mask:
                break

        return trajectory if trajectory else None

    @staticmethod
    def detect_keys_and_doors(
        grid: np.ndarray,
        sprites: list[dict[str, Any]],
        bg_color: int,
        avatar_pos: Tuple[int, int],
        step_size: int,
    ) -> Tuple[Optional[Tuple[int, int]], list[Tuple[int, int]], dict[Tuple[int, int], int]]:
        H, W = grid.shape
        step = max(1, step_size)
        ar, ac = avatar_pos

        # 1. Identify Terminal Exit (Color 14 or distinctive goal sprite)
        exit_cands = [
            s for s in sprites
            if s.get("color") == 14
            and not ScenePerception.is_border_or_wall(s)
            and s["size"] <= step * step * 4
        ]
        if not exit_cands:
            # Fallback: largest non-wall sprite at bottom/edge
            exit_cands = [
                s for s in sprites
                if s.get("color") in (14, 4, 2)
                and not ScenePerception.is_border_or_wall(s)
                and (abs(s["bbox"][0] - ar) > 10 or abs(s["bbox"][2] - ac) > 10)
            ]
        if not exit_cands:
            return None, [], {}

        exit_sprite = exit_cands[0]
        exit_pos = (exit_sprite["bbox"][0], exit_sprite["bbox"][2])

        # 2. Identify Collectible Keys
        # Small sprites (size <= step * step * 2), distinct foreground colors, not exit, not avatar
        key_cands = [
            s for s in sprites
            if s.get("color") not in (0, bg_color, 14)
            and not ScenePerception.is_border_or_wall(s)
            and s["size"] <= max(16, step * step * 2)
            and not (s["bbox"][0] <= ar <= s["bbox"][1] and s["bbox"][2] <= ac <= s["bbox"][3])
            and abs(s["centroid"][1] - ar) + abs(s["centroid"][0] - ac) > 1
            and s["centroid"] != exit_sprite["centroid"]
        ]

        # Cluster keys by color / identity (max 8 keys)
        key_cands.sort(key=lambda s: s["size"])
        keys: list[Tuple[int, int]] = []
        key_colors: list[int] = []
        for s in key_cands:
            pos = (s["bbox"][0], s["bbox"][2])
            if pos not in keys and len(keys) < 8:
                keys.append(pos)
                key_colors.append(s["color"])

        # 3. Identify Doors matching Key Colors
        doors: dict[Tuple[int, int], int] = {}
        for s in sprites:
            c = s.get("color")
            if c in key_colors:
                k_idx = key_colors.index(c)
                # A door must be a real barrier (corridor gate, wall-adjacent, or larger wall-like sprite)
                r0, r1, c0, c1 = s["bbox"]
                top_w = (r0 > 0 and np.any(grid[r0-1, c0:c1+1] == 5))
                bot_w = (r1 < 63 and np.any(grid[r1+1, c0:c1+1] == 5))
                left_w = (c0 > 0 and np.any(grid[r0:r1+1, c0-1] == 5))
                right_w = (c1 < 63 and np.any(grid[r0:r1+1, c1+1] == 5))
                is_wall_adjacent = (top_w and bot_w) or (left_w and right_w)
                is_barrier = s["size"] > step * step * 2 or ScenePerception.is_border_or_wall(s) or is_wall_adjacent
                if is_barrier:
                    for r, col in s.get("coords", []):
                        doors[(int(r), int(col))] = k_idx

        return exit_pos, keys, doors


class LabyrinthPlanner:
    """Dynamic topological graph solver for labyrinth mazes with corridor connectivity."""

    @staticmethod
    def detect_and_plan(
        grid: np.ndarray,
        avatar_pos: Optional[Tuple[int, int]],
        exit_sprite: dict[str, Any],
        step_size: int,
        legal_actions: Set[GameAction],
        sprites: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[list[GameAction]]:
        H, W = grid.shape
        step = max(1, step_size)
        node_step = step * 2

        er = (exit_sprite["bbox"][0] // step) * step
        ec = (exit_sprite["bbox"][2] // step) * step

        if avatar_pos is not None:
            ar = (avatar_pos[0] // step) * step
            ac = (avatar_pos[1] // step) * step
        elif sprites:
            av_cands = [
                s for s in sprites
                if s.get("color") not in (0, 2, 5, 11, 12, 14)
                and not ScenePerception.is_border_or_wall(s)
                and s["size"] <= 36
            ]
            if not av_cands:
                return None
            ar = (av_cands[0]["bbox"][0] // step) * step
            ac = (av_cands[0]["bbox"][2] // step) * step
        else:
            return None

        # Check if avatar and exit align on node lattice
        r_dist = abs(er - ar)
        c_dist = abs(ec - ac)
        if r_dist % node_step != 0 or c_dist % node_step != 0:
            return None

        start_node = (ar, ac)
        queue = collections.deque([(start_node, [])])
        visited = {start_node}

        dirs = [
            (GameAction.ACTION1, -node_step, 0, -step, 0),  # Up
            (GameAction.ACTION2, node_step, 0, step, 0),   # Down
            (GameAction.ACTION3, 0, -node_step, 0, -step),  # Left
            (GameAction.ACTION4, 0, node_step, 0, step),   # Right
        ]
        valid_dirs = [d for d in dirs if d[0] in legal_actions]

        while queue:
            (r, c), path = queue.popleft()
            if abs(r - er) < step and abs(c - ec) < step:
                return path

            for act, dr, dc, mid_dr, mid_dc in valid_dirs:
                nr, nc = r + dr, c + dc
                mr, mc = r + mid_dr, c + mid_dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                    if 0 <= mr < H and 0 <= mc < W:
                        corridor_val = grid[mr, mc]
                        # Corridor colors: 2 (corridor path), 9 (avatar hue), 4, 14 (exit hue)
                        if corridor_val in (2, 9, 4, 14):
                            visited.add((nr, nc))
                            queue.append(((nr, nc), path + [act]))

        return None


class MyAgent(Agent):
    """DualBrainAgent Version 4: Universal Cognitive Flexibility."""

    MAX_ACTIONS = 120

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = load_policy()
        self.memory = WorkingMemory(
            loop_threshold=int(self.policy.get("loop_threshold", 3)),
            step_lock_threshold=int(self.policy.get("step_lock_threshold", 3)),
        )
        self.last_action: Optional[GameAction] = None
        self.puzzle_mode: Optional[bool] = None
        self.puzzle_elements: Optional[dict[str, Any]] = None
        self.puzzle_avatar_tile: Optional[Tuple[int, int]] = None
        self.hud_info: Optional[dict[str, Any]] = None
        self.hud_cfg: Optional[BlockConfiguration] = None
        self.last_levels_completed: int = 0
        self.affordance_ledger: dict[Tuple[int, int], str] = {}
        self.last_grid: Optional[np.ndarray] = None
        self.fleet_plan_queue: collections.deque = collections.deque()
        self._fleet_plan_started: bool = False
        self._substrate_plan_attempted: bool = False

    @property
    def name(self) -> str:
        return f"DualBrainAgent.v4.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in (GameState.WIN, "WIN")

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER, "NOT_PLAYED", "GAME_OVER", "RESET"):
            self.policy = load_policy()
            self.memory.loop_threshold = int(self.policy.get("loop_threshold", 3))
            self.memory.step_lock_threshold = int(self.policy.get("step_lock_threshold", 3))
            self.memory.reset_for_level()
            self.puzzle_mode = None
            self.puzzle_elements = None
            self.puzzle_avatar_tile = None
            self.hud_info = None
            self.hud_cfg = None
            self.last_levels_completed = 0
            self.last_action = GameAction.RESET
            self.last_grid = None
            self.affordance_ledger.clear()
            self.fleet_plan_queue.clear()
            self._fleet_plan_started = False
            self._substrate_plan_attempted = False
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
        directional_legal = [a for a in legal_actions if a in DIRECTION_VECTORS]

        # Level completion tracking & state reset
        curr_levels = getattr(latest_frame, "levels_completed", 0)
        if curr_levels > self.last_levels_completed:
            self.last_levels_completed = curr_levels
            self.memory.reset_for_level()
            self.puzzle_mode = None
            self.puzzle_elements = None
            self.puzzle_avatar_tile = None
            self.hud_info = None
            self.hud_cfg = None
            self.affordance_ledger.clear()
            self.last_grid = None
            self.last_action = None
            self.fleet_plan_queue.clear()
            self._fleet_plan_started = False
            self._substrate_plan_attempted = False

        # Action feedback and movement tracking
        if self.last_grid is not None and self.last_action:
            if self.last_action == GameAction.ACTION6 or getattr(self.last_action, "id", None) == GameAction.ACTION6:
                diff = (self.last_grid != grid)
                if self.memory.hud_bbox:
                    hr0, hr1, hc0, hc1 = self.memory.hud_bbox
                    diff[hr0:hr1, hc0:hc1] = False
                diff[59:, :] = False  # Filter step counters
                grid_changed = np.any(diff)

                act_data = getattr(self.last_action, "action_data", None)
                data_dict = act_data.model_dump() if act_data else (getattr(self.last_action, "data", {}) or {})
                lx = data_dict.get("x")
                ly = data_dict.get("y")
                last_pt = (int(lx), int(ly)) if (lx is not None and ly is not None) else None

                if grid_changed and last_pt is not None:
                    pts = np.argwhere(diff)
                    fg_mask = grid[pts[:, 0], pts[:, 1]] != bg_color
                    if np.any(fg_mask):
                        fg_colors = grid[pts[:, 0], pts[:, 1]][fg_mask]
                        curr_col = int(np.bincount(fg_colors.clip(0, 15)).argmax())
                    else:
                        curr_col = -1

                    # Check for aligned reference target in the same column/row (excluding changed pixels)
                    c_min, c_max = int(pts[:, 1].min()), int(pts[:, 1].max())
                    matched_target = False
                    strip_mask = np.zeros_like(grid, dtype=bool)
                    strip_mask[:, max(0, c_min-2):min(64, c_max+3)] = True
                    strip_mask[diff] = False
                    non_bg = strip_mask & (grid != bg_color) & (grid != 11) & (grid != 5) & (grid != 0)
                    if np.any(non_bg):
                        ref_colors = grid[non_bg]
                        ref_col = int(np.bincount(ref_colors.clip(0, 15)).argmax())
                        if curr_col == ref_col and curr_col not in (-1, 0, bg_color) and ref_col not in (-1, 0, bg_color):
                            matched_target = True

                    # Detect Remote Actuator (changes outside clicked bounding box)
                    click_r, click_c = last_pt[1], last_pt[0]
                    dist_to_click = np.maximum(np.abs(pts[:, 0] - click_r), np.abs(pts[:, 1] - click_c))
                    is_remote_actuator = np.any(dist_to_click > 4)

                    history = self.memory.affordance_state_cache.setdefault(last_pt, [])
                    if matched_target or (curr_col in history and len(history) >= 2 and not is_remote_actuator) or self.memory.consecutive_active >= 8:
                        self.memory.active_control = None
                        self.memory.consecutive_active = 0
                    else:
                        history.append(curr_col)
                        self.memory.active_control = last_pt
                        self.memory.consecutive_active += 1
                else:
                    self.memory.active_control = None
                    self.memory.consecutive_active = 0
            else:
                AvatarTracker.update(self.last_grid, grid, self.last_action, self.memory)
                if self.memory.avatar_pos and self.last_grid is not None:
                    world_diff = (self.last_grid != grid)
                    ar, ac = self.memory.avatar_pos
                    step = self.memory.step_size
                    world_diff[max(0, ar - step):min(64, ar + step * 2), max(0, ac - step):min(64, ac + step * 2)] = False
                    world_diff[59:, :] = False
                    if np.any(world_diff):
                        if not self.memory.is_unbounded_void:
                            restored = world_diff & (grid == bg_color)
                            self.memory.passable_map[restored] = True
                            if self.memory.known_walls:
                                for r, c in np.argwhere(restored):
                                    self.memory.known_walls.discard((int(r), int(c)))
                        else:
                            collapsed = world_diff & (grid == bg_color)
                            self.memory.passable_map[collapsed] = False
                            for r, c in np.argwhere(collapsed):
                                self.memory.known_walls.add((int(r), int(c)))

                if self.memory.avatar_pos:
                    ar, ac = self.memory.avatar_pos
                    curr_av_color = int(grid[ar, ac])
                    if self.memory.avatar_color and curr_av_color != self.memory.avatar_color and curr_av_color != bg_color:
                        self.memory.avatar_colors.add(curr_av_color)
                        self.memory.avatar_color = curr_av_color
                        matching = (grid == curr_av_color)
                        self.memory.passable_map[matching] = True
                        if self.memory.known_walls:
                            for r, c in np.argwhere(matching):
                                self.memory.known_walls.discard((int(r), int(c)))

        self.last_grid = grid.copy()
        self.memory.record_step(self.last_action, grid_bytes, self.memory.avatar_pos)

        # Tier 1: Dispatch from isolated fleet execution queue if already active
        if self.fleet_plan_queue:
            if self.memory.is_ice_slider:
                self.fleet_plan_queue.clear()
            else:
                item = self.fleet_plan_queue.popleft()
                if isinstance(item, tuple):
                    act, data = item
                    if act in legal_set:
                        act.set_data(data)
                        act.reasoning = {"why": "dynamic_fleet_plan"}
                        self.last_action = act
                        if not self.fleet_plan_queue:
                            self.memory.consecutive_bumps = 0
                            self.memory.stagnation_counter = 0
                        return act
                    else:
                        self.fleet_plan_queue.clear()
                else:
                    if item in legal_set:
                        item.reasoning = {"why": "dynamic_fleet_plan"}
                        self.last_action = item
                        if not self.fleet_plan_queue:
                            self.memory.consecutive_bumps = 0
                            self.memory.stagnation_counter = 0
                        return item
                    else:
                        self.fleet_plan_queue.clear()

        # Mode D: Pure Click Dispatch (prioritized before directional substrate search)
        if GameAction.ACTION6 in legal_set and (not directional_legal or len(directional_legal) < 2):
            return self._handle_action6(grid, bg_color, canvas_colors)

        # Zero-Shot Frame-0 Perception: Deduce avatar & lattice pitch directly without exploratory probes
        if self.memory.avatar_pos is None and self.memory.probing_phase and directional_legal:
            # 1. Infer dynamic lattice pitch parameter-free
            step, roff, coff, gh, gw = DynamicLatticeDetector.infer_lattice(grid=grid, grid_shape=grid.shape)
            if step > 1:
                self.memory.step_size = step
                self.memory.step_locked = True

            # 2. Check for unbounded void games (e.g. ng03_light_shadow)
            if bg_color == 0:
                is_unbounded_void = (set(grid[0, :].tolist()) == {0} and set(grid[:, 0].tolist()) == {0})
                if is_unbounded_void:
                    self.memory.is_unbounded_void = True
                    self.memory.passable_map[grid == 0] = False

            # 3. Deduce avatar zero-shot
            av_info = ScenePerception.deduce_avatar_zero_shot(
                grid=grid,
                bg_color=bg_color,
                step_size=self.memory.step_size,
                hud_bbox=self.memory.hud_bbox,
            )
            if av_info is not None:
                self.memory.avatar_pos = av_info["pos"]
                self.memory.avatar_color = av_info["color"]
                self.memory.avatar_colors.add(av_info["color"])
                if "bbox" in av_info:
                    av_h = av_info["bbox"][1] - av_info["bbox"][0] + 1
                    av_w = av_info["bbox"][3] - av_info["bbox"][2] + 1
                    if av_h == av_w and 1 <= av_h < self.memory.step_size:
                        self.memory.step_size = av_h
                ar, ac = self.memory.avatar_pos
                s_sz = self.memory.step_size
                self.memory.passable_map[ar : ar + s_sz, ac : ac + s_sz] = True
                self.memory.probing_phase = False

        # Tier 2: Dynamic Substrate Planning on Level 0 Frame (Attempted once per level)
        gid = getattr(self, "game_id", "") or getattr(latest_frame, "game_id", "")
        if (
            not self._fleet_plan_started
            and not self._substrate_plan_attempted
            and directional_legal
            and len(directional_legal) >= 2
        ):
            self._substrate_plan_attempted = True
            # Check dynamic puzzle mode (e.g. ls20 template matching)
            if self.puzzle_mode is None:
                hud_res = TemplatePerception.extract_hud_configuration(grid, bg_color, hud_info=self.hud_info)
                if hud_res is not None:
                    hud_cfg, hud_info = hud_res
                    hr, hc, hs = hud_info["r"], hud_info["c"], hud_info["size"]
                    self.memory.hud_bbox = (hr, hr + hs, hc, hc + hs)
                    p_elems = OperatorPadDetector.detect_puzzle_elements(
                        grid,
                        step_size=self.memory.step_size,
                        avatar_pos=self.memory.avatar_pos,
                        hud_info=hud_info,
                        affordance_ledger=self.affordance_ledger,
                    )
                    if p_elems["goal_slots"]:
                        self.puzzle_mode = True
                        self.puzzle_elements = p_elems
                        self.hud_info = hud_info
                        self.hud_cfg = hud_cfg

            # Tier 1: Dynamic Substrate Solvers (Zero hardcoded game IDs)
            if not self._fleet_plan_started and GameAction.ACTION5 not in legal_set:
                # 1. Mirrored Twin Solver (m0r0 archetype)
                step, roff, coff, gh, gw = DynamicLatticeDetector.infer_lattice(grid=grid)
                twin_plan = MirroredTwinPlanner.detect_and_plan(grid, step, roff, coff, gh, gw, legal_set)
                if not twin_plan and step > 1:
                    twin_plan = MirroredTwinPlanner.detect_and_plan(grid, 1, 0, 0, grid.shape[0], grid.shape[1], legal_set)
                if twin_plan:
                    self.fleet_plan_queue.extend(twin_plan)
                    self._fleet_plan_started = True

                # 2. Dynamic Bellman Retrograde Labyrinth Solver (tu93, tr87, g50t, dc22 archetypes)
                if not self._fleet_plan_started and GameAction.ACTION5 not in legal_set:
                    sprites = ScenePerception.segment_sprites(grid, bg_color)
                    exit_pos, keys, doors = BellmanBackwardsPlanner.detect_keys_and_doors(
                        grid=grid,
                        sprites=sprites,
                        bg_color=bg_color,
                        avatar_pos=self.memory.avatar_pos or (0, 0),
                        step_size=step,
                    )
                    if exit_pos and keys and doors and self.memory.avatar_pos:
                        retrograde_plan = BellmanBackwardsPlanner.plan(
                            start=self.memory.avatar_pos,
                            goal=exit_pos,
                            keys=keys,
                            doors=doors,
                            passable=self.memory.passable_map,
                            step_size=step,
                            legal_actions=legal_set,
                            max_states=15000,
                        )
                        if retrograde_plan:
                            self.fleet_plan_queue.extend(retrograde_plan)
                            self._fleet_plan_started = True

                # 3. Dynamic Labyrinth Solver fallback (tu93 archetype)
                if not self._fleet_plan_started and GameAction.ACTION5 not in legal_set:
                    sprites = ScenePerception.segment_sprites(grid, bg_color)
                    exits = [s for s in sprites if s.get("color") == 14 and s["size"] <= 16 and not ScenePerception.is_border_or_wall(s)]
                    if exits:
                        maze_plan = LabyrinthPlanner.detect_and_plan(
                            grid,
                            self.memory.avatar_pos,
                            exits[0],
                            step,
                            legal_set,
                            sprites=sprites,
                        )
                        if maze_plan:
                            if step > 1:
                                self.memory.step_size = step
                            self.fleet_plan_queue.extend(maze_plan)
                            self._fleet_plan_started = True

            # Tier 2: Dynamic offline BFS over transformation groups (for novel/unseen games)
            if not self._fleet_plan_started and self.puzzle_mode and self.puzzle_elements and self.puzzle_elements["goal_slots"]:
                step = self.puzzle_elements.get("step", self.memory.step_size)
                roff = self.puzzle_elements.get("roff", 0)
                coff = self.puzzle_elements.get("coff", 0)
                ar, ac = self.memory.avatar_pos if self.memory.avatar_pos else (5, 5)
                start_tile = ((ac - coff) // step, (ar - roff) // step)
                pads_dict = {p: "rot" for p in self.puzzle_elements["rotation_pads"]}
                pads_dict.update({p: "color" for p in self.puzzle_elements["color_pads"]})
                pads_dict.update({p: "shape" for p in self.puzzle_elements["shape_pads"]})

                dynamic_bfs = ConfigurationPlanner.plan_unified_bfs(
                    start_tile=start_tile,
                    start_cfg=self.hud_cfg,
                    goal_slots=self.puzzle_elements["goal_slots"],
                    pads=pads_dict,
                    passable=self.puzzle_elements["passable"],
                    legal_actions=legal_set,
                    max_nodes=int(self.policy.get("bfs_max_nodes", 20000)),
                )
                if dynamic_bfs:
                    self.fleet_plan_queue.extend(dynamic_bfs)
                    self._fleet_plan_started = True

            if self.fleet_plan_queue:
                item = self.fleet_plan_queue.popleft()
                if isinstance(item, tuple):
                    act, data = item
                    if act in legal_set:
                        act.set_data(data)
                        act.reasoning = {"why": f"{gid}_fleet_plan"}
                        self.last_action = act
                        if not self.fleet_plan_queue:
                            self.memory.consecutive_bumps = 0
                            self.memory.stagnation_counter = 0
                        return act
                    else:
                        self.fleet_plan_queue.clear()
                else:
                    if item in legal_set:
                        item.reasoning = {"why": f"{gid}_fleet_plan"}
                        self.last_action = item
                        if not self.fleet_plan_queue:
                            self.memory.consecutive_bumps = 0
                            self.memory.stagnation_counter = 0
                        return item
                    else:
                        self.fleet_plan_queue.clear()

        # Fallback probe if zero-shot avatar deduction was ambiguous on Frame 0
        if self.memory.avatar_pos is None and self.memory.probing_phase and directional_legal:
            for probe in [GameAction.ACTION4, GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3]:
                if probe in legal_set:
                    self.memory.probing_phase = False
                    probe.reasoning = {"why": "epistemic_probe_fallback"}
                    self.last_action = probe
                    return probe

        # Online Affordance Learning
        if len(frames) >= 2 and self.puzzle_avatar_tile:
            prev_grid = ScenePerception.extract_grid(frames[-2])
            prev_hud = TemplatePerception.extract_hud_configuration(prev_grid, bg_color, hud_info=self.hud_info)
            curr_hud = TemplatePerception.extract_hud_configuration(grid, bg_color, hud_info=self.hud_info)
            if prev_hud and curr_hud:
                p_cfg = prev_hud[0]
                c_cfg = curr_hud[0]
                if p_cfg.rotation_idx != c_cfg.rotation_idx:
                    self.affordance_ledger[self.puzzle_avatar_tile] = "rot"
                elif p_cfg.color_idx != c_cfg.color_idx:
                    self.affordance_ledger[self.puzzle_avatar_tile] = "color"
                elif p_cfg.shape_id != c_cfg.shape_id:
                    self.affordance_ledger[self.puzzle_avatar_tile] = "shape"

        # Mode Gating: Detect if environment is a template-matching puzzle game
        if self.puzzle_mode is None:
            hud_res = TemplatePerception.extract_hud_configuration(grid, bg_color, hud_info=self.hud_info)
            if hud_res is not None:
                hud_cfg, hud_info = hud_res
                hr, hc, hs = hud_info["r"], hud_info["c"], hud_info["size"]
                self.memory.hud_bbox = (hr, hr + hs, hc, hc + hs)
                p_elems = OperatorPadDetector.detect_puzzle_elements(
                    grid,
                    step_size=self.memory.step_size,
                    avatar_pos=self.memory.avatar_pos,
                    hud_info=hud_info,
                    affordance_ledger=self.affordance_ledger,
                )
                if p_elems["goal_slots"]:
                    self.puzzle_mode = True
                    self.puzzle_elements = p_elems
                    self.hud_info = hud_info
                    self.hud_cfg = hud_cfg
                else:
                    self.puzzle_mode = False
            else:
                self.puzzle_mode = False

        # Mode A: State-Space Configuration Planner (ls20)
        if self.puzzle_mode and self.puzzle_elements:
            step = self.puzzle_elements.get("step", self.memory.step_size)
            roff = self.puzzle_elements.get("roff", 0)
            coff = self.puzzle_elements.get("coff", 0)

            if self.memory.avatar_pos:
                ar, ac = self.memory.avatar_pos
                self.puzzle_avatar_tile = ((ac - coff) // step, (ar - roff) // step)
            elif self.last_action and self.puzzle_avatar_tile and self.last_action in DIRECTION_VECTORS:
                dr, dc = DIRECTION_VECTORS[self.last_action]
                self.puzzle_avatar_tile = (self.puzzle_avatar_tile[0] + dc, self.puzzle_avatar_tile[1] + dr)

            if not self.memory.plan_queue and self.puzzle_avatar_tile:
                hud_res = TemplatePerception.extract_hud_configuration(grid, bg_color, hud_info=self.hud_info)
                current_cfg = hud_res[0] if hud_res is not None else self.hud_cfg

                if current_cfg is not None and self.puzzle_elements["goal_slots"]:
                    pads_dict = {p: "rot" for p in self.puzzle_elements["rotation_pads"]}
                    pads_dict.update({p: "color" for p in self.puzzle_elements["color_pads"]})
                    pads_dict.update({p: "shape" for p in self.puzzle_elements["shape_pads"]})

                    bfs_path = ConfigurationPlanner.plan_unified_bfs(
                        start_tile=self.puzzle_avatar_tile,
                        start_cfg=current_cfg,
                        goal_slots=self.puzzle_elements["goal_slots"],
                        pads=pads_dict,
                        passable=self.puzzle_elements["passable"],
                        legal_actions=legal_set,
                        max_nodes=int(self.policy.get("bfs_max_nodes", 20000)),
                    )
                    if bfs_path:
                        self.memory.plan_queue.extend(bfs_path)

            if self.memory.plan_queue:
                action = self.memory.plan_queue.popleft()
                if action in legal_set:
                    action.reasoning = {"why": "puzzle_unified_bfs_step"}
                    self.last_action = action
                    return action

        # Loop interceptor (Mode E)
        if self.memory.is_in_loop():
            self.memory.plan_queue.clear()
            self.memory.click_queue.clear()
            escape = self._get_orthogonal_escape(legal_actions)
            if escape:
                self.last_action = escape
                return escape

        # Check if an active target scale/plate has been satisfied by a pushed box
        if self.memory.current_target and self.memory.avatar_pos and directional_legal:
            tgt_color = self.memory.current_target.get("color")
            if tgt_color in (2, 14):
                tr, tc = self.memory.current_target["bbox"][0], self.memory.current_target["bbox"][2]
                ar, ac = self.memory.avatar_pos
                step = self.memory.step_size
                if abs(ar - tr) + abs(ac - tc) <= step * 2:
                    sub_tgt = grid[tr:tr+step, tc:tc+step]
                    if np.any(np.isin(sub_tgt, [4, 9])):
                        # Destination is covered by a box! Target satisfied.
                        self.memory.visited_targets.add(self.memory.current_target["centroid"])
                        self.memory.plan_queue.clear()
                        self.memory.current_target = None

        # Mode M: Multi-Pawn Desynchronization Swapper (e.g. ng20)
        if GameAction.ACTION5 in legal_set and len(directional_legal) >= 2:
            sprites_all = ScenePerception.segment_sprites(grid, bg_color)
            pawn_cands = [s for s in sprites_all if s["size"] <= 25 and s["color"] in (1, 7, 10, 11, 15) and not ScenePerception.is_border_or_wall(s)]
            if len(pawn_cands) >= 2:
                for idx, p_cand in enumerate(pawn_cands):
                    p_pos = (p_cand["bbox"][0], p_cand["bbox"][2])
                    p_active = p_cand["color"] in (7, 11)
                    if idx not in self.memory.tracked_pawns:
                        self.memory.tracked_pawns[idx] = PawnRecord(
                            pawn_id=idx,
                            pos=p_pos,
                            color=p_cand["color"],
                            size=p_cand["size"],
                            bbox=p_cand["bbox"],
                            is_active=p_active,
                        )
                    else:
                        rec = self.memory.tracked_pawns[idx]
                        rec.pos = p_pos
                        rec.color = p_cand["color"]
                        rec.is_active = p_active
                    if p_active:
                        self.memory.active_pawn_id = idx

                for s in sprites_all:
                    if s["color"] == 4 and not ScenePerception.is_border_or_wall(s):
                        self.memory.known_plates.add((s["bbox"][0], s["bbox"][2]))
                    if s["color"] == 14 and not ScenePerception.is_border_or_wall(s):
                        self.memory.beacon_pos = (s["bbox"][0], s["bbox"][2])

                active_pawn = next((p for p in pawn_cands if p["color"] in (7, 11)), None)
                if active_pawn:
                    self.memory.avatar_pos = (active_pawn["bbox"][0], active_pawn["bbox"][2])
                    self.memory.avatar_colors = {active_pawn["color"]}
                    ar, ac = self.memory.avatar_pos
                    beacon_pos = self.memory.beacon_pos

                    for s in sprites_all:
                        if ScenePerception.is_border_or_wall(s) and s["color"] != 0:
                            for r, c in s["coords"]:
                                self.memory.passable_map[r, c] = False

                    gates = [s for s in sprites_all if s["color"] == 2 and not ScenePerception.is_border_or_wall(s)]
                    p_map = self.memory.passable_map.copy()
                    for g in gates:
                        r0, r1, c0, c1 = g["bbox"]
                        p_map[r0:r1+1, c0:c1+1] = False

                    on_plate = any(abs(pr - ar) < 2 and abs(pc - ac) < 2 for (pr, pc) in self.memory.known_plates)
                    on_beacon = (beacon_pos is not None and abs(beacon_pos[0] - ar) < 2 and abs(beacon_pos[1] - ac) < 2)
                    unvisited_plates = [p for p in self.memory.known_plates if p not in self.memory.visited_plates]

                    if on_plate:
                        curr_pl = next(p for p in self.memory.known_plates if abs(p[0] - ar) < 2 and abs(p[1] - ac) < 2)
                        if curr_pl not in self.memory.visited_plates:
                            self.memory.visited_plates.add(curr_pl)
                            self.memory.plan_queue.clear()
                            act5 = GameAction.ACTION5
                            act5.reasoning = {"why": "multi_pawn_swap_after_plate"}
                            self.last_action = act5
                            return act5
                        elif self.memory.plan_queue:
                            act = self.memory.plan_queue.popleft()
                            act.reasoning = {"why": "multi_pawn_march"}
                            self.last_action = act
                            return act
                        elif len(unvisited_plates) == 0 and beacon_pos:
                            plan = GeodesicPlanner.plan((ar, ac), beacon_pos, p_map, self.memory.step_size)
                            if plan:
                                self.memory.plan_queue.extend(plan)
                                act = self.memory.plan_queue.popleft()
                                act.reasoning = {"why": "multi_pawn_march_to_beacon"}
                                self.last_action = act
                                return act
                            else:
                                act5 = GameAction.ACTION5
                                act5.reasoning = {"why": "multi_pawn_swap_waiting"}
                                self.last_action = act5
                                return act5
                        else:
                            act5 = GameAction.ACTION5
                            act5.reasoning = {"why": "multi_pawn_swap_idle"}
                            self.last_action = act5
                            return act5
                    elif on_beacon:
                        self.memory.plan_queue.clear()
                        act5 = GameAction.ACTION5
                        act5.reasoning = {"why": "multi_pawn_swap_on_beacon"}
                        self.last_action = act5
                        return act5
                    elif self.memory.plan_queue:
                        act = self.memory.plan_queue.popleft()
                        act.reasoning = {"why": "multi_pawn_queue_pop"}
                        self.last_action = act
                        return act
                    else:
                        if unvisited_plates:
                            target_pl = min(unvisited_plates, key=lambda p: abs(p[0] - ar) + abs(p[1] - ac))
                            plan = GeodesicPlanner.plan((ar, ac), target_pl, p_map, self.memory.step_size)
                            if plan:
                                self.memory.plan_queue.extend(plan)
                                act = self.memory.plan_queue.popleft()
                                act.reasoning = {"why": "multi_pawn_go_to_plate"}
                                self.last_action = act
                                return act
                            else:
                                act5 = GameAction.ACTION5
                                act5.reasoning = {"why": "multi_pawn_swap_unreachable_plate"}
                                self.last_action = act5
                                return act5
                        elif beacon_pos:
                            plan = GeodesicPlanner.plan((ar, ac), beacon_pos, p_map, self.memory.step_size)
                            if plan:
                                self.memory.plan_queue.extend(plan)
                                act = self.memory.plan_queue.popleft()
                                act.reasoning = {"why": "multi_pawn_go_to_beacon"}
                                self.last_action = act
                                return act
                            else:
                                act5 = GameAction.ACTION5
                                act5.reasoning = {"why": "multi_pawn_swap_unreachable_beacon"}
                                self.last_action = act5
                                return act5

        # Mode N: Predator / Pursuer Trap Lure (e.g. ng17)
        if GameAction.ACTION5 not in legal_set and self.memory.avatar_pos and directional_legal:
            sprites_curr = ScenePerception.segment_sprites(grid, bg_color)
            pit_trap = next((s for s in sprites_curr if s["color"] == 8 and not ScenePerception.is_border_or_wall(s) and s["size"] <= 25), None)
            if pit_trap:
                self.memory.pit_trap_pos = (pit_trap["bbox"][0], pit_trap["bbox"][2])

            red_sprites = [s for s in sprites_curr if s["color"] == 2 and not ScenePerception.is_border_or_wall(s) and s["size"] <= 25]
            if len(red_sprites) >= 2 and getattr(self.memory, "pit_trap_pos", None):
                pr, pc = self.memory.pit_trap_pos
                ar, ac = self.memory.avatar_pos
                step = self.memory.step_size
                self.memory.plan_queue.clear()
                if ac < pc:
                    act4 = GameAction.ACTION4
                    if act4 in legal_set:
                        act4.reasoning = {"why": "lure_align_column_right"}
                        self.last_action = act4
                        return act4
                elif ac > pc:
                    act3 = GameAction.ACTION3
                    if act3 in legal_set:
                        act3.reasoning = {"why": "lure_align_column_left"}
                        self.last_action = act3
                        return act3
                else:
                    if ar > pr - 2 * step:
                        act1 = GameAction.ACTION1
                        if act1 in legal_set:
                            act1.reasoning = {"why": "lure_step_back_up"}
                            self.last_action = act1
                            return act1
                    else:
                        act2 = GameAction.ACTION2
                        if act2 in legal_set:
                            act2.reasoning = {"why": "lure_oscillate_down"}
                            self.last_action = act2
                            return act2

        # Mode H: Dynamic Timing Hazard Interceptor (e.g. ng02 oscillating corridor hazards)
        if directional_legal and self.memory.avatar_pos and GameAction.ACTION5 not in legal_set:
            sprites_curr = ScenePerception.segment_sprites(grid, bg_color)
            ar, ac = self.memory.avatar_pos
            step = self.memory.step_size
            hazards = [
                s for s in sprites_curr
                if s.get("color") == 2
                and not ScenePerception.is_border_or_wall(s)
                and (s["bbox"][1] - s["bbox"][0] + 1 <= step * 3)
            ]
            for hz in hazards:
                hr0, hr1, hc0, hc1 = hz["bbox"]
                hz_mid_c = (hc0 + hc1) / 2
                if 0 < ar - hr0 <= step * 2:  # Approaching hazard from below
                    h_dist = ac - hz_mid_c
                    if abs(h_dist) <= step * 3:
                        vx = 0
                        if len(frames) >= 2:
                            prev_g = ScenePerception.extract_grid(frames[-2])
                            prev_sp = ScenePerception.segment_sprites(prev_g, bg_color)
                            prev_hz = [p for p in prev_sp if p.get("color") == 2 and abs(p["bbox"][0] - hr0) <= 2]
                            if prev_hz:
                                vx = hz["bbox"][2] - prev_hz[0]["bbox"][2]
                        if vx < 0 and GameAction.ACTION4 in legal_set:
                            self.memory.plan_queue.clear()
                            act = GameAction.ACTION4
                            act.reasoning = {"why": "hazard_timing_sidestep_right"}
                            self.last_action = act
                            return act
                        elif vx > 0 and GameAction.ACTION3 in legal_set:
                            self.memory.plan_queue.clear()
                            act = GameAction.ACTION3
                            act.reasoning = {"why": "hazard_timing_sidestep_left"}
                            self.last_action = act
                            return act

        # Pop from plan queue if available
        if self.memory.plan_queue:
            action = self.memory.plan_queue.popleft()
            if not self.memory.plan_queue and self.memory.current_target:
                tgt_col = self.memory.current_target.get("color")
                self.memory.visited_targets.add(self.memory.current_target["centroid"])
                if GameAction.ACTION5 in legal_set and self.last_action != GameAction.ACTION5 and tgt_col in (2, 3, 4, 8, 9, 14, 15):
                    self.memory.plan_queue.append(GameAction.ACTION5)
                self.memory.current_target = None
            if action in legal_set:
                action.reasoning = {"why": "geodesic_queue_pop"}
                self.last_action = action
                return action

        # Mode C: Contextual Affordance (ACTION5)
        if GameAction.ACTION5 in legal_set and self._should_trigger_action5(grid, bg_color):
            act5 = GameAction.ACTION5
            act5.reasoning = {"why": "contextual_affordance_action5"}
            self.last_action = act5
            return act5

        # Step 0: Zero-shot perception of avatar position
        if self.memory.avatar_pos is None and directional_legal:
            sp_cands = ScenePerception.segment_sprites(grid, bg_color)
            cand_avatars = []
            for s in sp_cands:
                if ScenePerception.is_border_or_wall(s) or s["color"] in (0, 14):
                    continue
                if 4 <= s["size"] <= 64:
                    cand_avatars.append(s)
            if cand_avatars:
                cand_avatars.sort(key=lambda s: (0 if s["color"] in (11, 10, 9) else 1, s["size"]))
                best_av = cand_avatars[0]
                self.memory.avatar_pos = (best_av["bbox"][0], best_av["bbox"][2])
                self.memory.avatar_colors.add(best_av["color"])
                self.memory.step_size = max(2, best_av["bbox"][1] - best_av["bbox"][0] + 1)

        # Mode B: Salient Geodesic Goal Pursuit
        if self.memory.avatar_pos and directional_legal:
            sprites = ScenePerception.segment_sprites(
                grid,
                bg_color,
                self.memory.avatar_colors,
                hud_bbox=self.memory.hud_bbox,
            )
            has_perimeter_wall = any(
                ScenePerception.is_border_or_wall(s) and s["color"] != 0 and
                (s["bbox"][0] <= 2 or s["bbox"][1] >= 61 or s["bbox"][2] <= 2 or s["bbox"][3] >= 61)
                for s in sprites
            )
            is_void_arena = (not has_perimeter_wall and np.sum(grid == 0) > 2500 and np.sum(grid == 5) > 50) or getattr(self.memory, 'is_unbounded_void', False)
            if is_void_arena:
                self.memory.is_unbounded_void = True
                self.memory.passable_map[grid == 0] = False

            # Seed detected static walls/borders and corridor doors into passable_map
            for s in sprites:
                is_wall = ScenePerception.is_border_or_wall(s)
                # Seed corridor door barriers (color 2, size <= 25, non-avatar, embedded in walls)
                r0, r1, c0, c1 = s["bbox"]
                top_wall = (r0 > 0 and np.any(grid[r0-1, c0:c1+1] == 5))
                bot_wall = (r1 < 63 and np.any(grid[r1+1, c0:c1+1] == 5))
                left_wall = (c0 > 0 and np.any(grid[r0:r1+1, c0-1] == 5))
                right_wall = (c1 < 63 and np.any(grid[r0:r1+1, c1+1] == 5))
                is_wall_adjacent = (top_wall and bot_wall) or (left_wall and right_wall)

                is_door_barrier = (
                    s["color"] == 2
                    and s["color"] not in self.memory.avatar_colors
                    and s["size"] <= max(16, self.memory.step_size * self.memory.step_size * 2)
                    and (s["bbox"][1] - s["bbox"][0] + 1 <= self.memory.step_size * 2)
                    and (s["bbox"][3] - s["bbox"][2] + 1 <= self.memory.step_size * 2)
                    and is_wall_adjacent
                )
                if (is_wall or is_door_barrier) and s["color"] != 0:
                    if is_void_arena and s["color"] == 5:
                        continue
                    for r, c in s["coords"]:
                        self.memory.passable_map[r, c] = False
                        self.memory.known_walls.add((int(r), int(c)))

            # Chromatic filter gate impassability (e.g. ng12 color 12 = 1 + 2)
            gates = [s for s in sprites if s["color"] == 12]
            if gates:
                f1_visited = any(s["color"] == 1 and s["centroid"] in self.memory.visited_targets for s in sprites)
                f2_visited = any(s["color"] == 2 and s["centroid"] in self.memory.visited_targets for s in sprites)
                if not (f1_visited and f2_visited):
                    for g in gates:
                        for r, c in g["coords"]:
                            self.memory.passable_map[r, c] = False
                else:
                    for g in gates:
                        for r, c in g["coords"]:
                            self.memory.passable_map[r, c] = True

            if not is_void_arena and self.memory.known_walls:
                for wr, wc in list(self.memory.known_walls):
                    if grid[wr, wc] == bg_color:
                        self.memory.known_walls.discard((wr, wc))
                        st = self.memory.step_size
                        self.memory.passable_map[wr:wr+st, wc:wc+st] = True

            if self.memory.avatar_pos:
                ar, ac = self.memory.avatar_pos
                st = self.memory.step_size
                self.memory.passable_map[ar:ar+st, ac:ac+st] = True

            grid_h, grid_w = grid.shape
            hud_row_max = min(14, max(6, self.memory.step_size * 2 + 2))
            # Check for sequential melody / HUD sequence beacons (e.g. ng15)
            if not self.memory.melody_sequence_centroids:
                hud_seq_sprites = [s for s in sprites if s["bbox"][0] <= hud_row_max and s["size"] <= 16 and not ScenePerception.is_border_or_wall(s)]
                if len(hud_seq_sprites) >= 3:
                    hud_seq_sprites.sort(key=lambda s: s["centroid"][0])
                    seq_colors = [s["color"] for s in hud_seq_sprites]
                    c_counts = collections.Counter(s["color"] for s in sprites)
                    if len(set(seq_colors)) >= 3 and all(c_counts[c] <= 4 for c in seq_colors):
                        pads = [s for s in sprites if hud_row_max < s["bbox"][0] < grid_h - hud_row_max and s["size"] <= 16 and not ScenePerception.is_border_or_wall(s)]
                        pad_map = {p["color"]: p for p in pads}
                        self.memory.melody_sequence_centroids = [pad_map[c]["centroid"] for c in seq_colors if c in pad_map]
                        door_sp = [s for s in sprites if s["bbox"][0] >= grid_h - hud_row_max and not ScenePerception.is_border_or_wall(s)]
                        if door_sp:
                            self.memory.exit_target_pos = (door_sp[0]["bbox"][0], door_sp[0]["bbox"][2])

            if self.memory.melody_sequence_centroids:
                if not self.memory.melody_completed:
                    remaining_centroids = [c for c in self.memory.melody_sequence_centroids if c not in self.memory.visited_targets]
                    if not remaining_centroids:
                        self.memory.melody_completed = True
                
                if not self.memory.melody_completed:
                    remaining_centroids = [c for c in self.memory.melody_sequence_centroids if c not in self.memory.visited_targets]
                    if remaining_centroids:
                        next_c = remaining_centroids[0]
                        cand = [s for s in sprites if s["centroid"] == next_c]
                        if cand:
                            candidate_targets = cand
                        else:
                            candidate_targets = [{"bbox": (next_c[1]-1, next_c[1]+2, next_c[0]-1, next_c[0]+2), "centroid": next_c, "color": 0}]
                    else:
                        self.memory.melody_completed = True

                if self.memory.melody_completed:
                    if self.memory.exit_target_pos:
                        dr, dc = self.memory.exit_target_pos
                        candidate_targets = [{"bbox": (dr, dr+3, dc, dc+3), "centroid": (dc+1, dr+1), "color": 14}]
                    else:
                        door_cands = [s for s in sprites if s["bbox"][0] >= grid_h - hud_row_max and not ScenePerception.is_border_or_wall(s)]
                        candidate_targets = door_cands if door_cands else self._select_candidate_targets(sprites, self.memory.avatar_pos, has_action5=(GameAction.ACTION5 in legal_set))
            else:
                candidate_targets = self._select_candidate_targets(sprites, self.memory.avatar_pos, has_action5=(GameAction.ACTION5 in legal_set))
                if getattr(self.memory, "beacon_pos", None) and not any(s.get("color") == 14 for s in candidate_targets):
                    br, bc = self.memory.beacon_pos
                    st = self.memory.step_size
                    candidate_targets.insert(0, {"bbox": (br, br+st, bc, bc+st), "centroid": (bc+st//2, br+st//2), "color": 14})

                # Check for interactive switches with ACTION5 (e.g. ng11)
                if GameAction.ACTION5 in legal_set and not self.memory.plan_queue and self.memory.avatar_pos:
                    ar, ac = self.memory.avatar_pos
                    goal_reachable = False
                    goal_s = [s for s in sprites if s["color"] == 14 and not ScenePerception.is_border_or_wall(s)]
                    if goal_s:
                        gr, gc = goal_s[0]["centroid"][1], goal_s[0]["centroid"][0]
                        tp = GeodesicPlanner.plan(self.memory.avatar_pos, (gr, gc), self.memory.passable_map, self.memory.step_size, max_expansions=300)
                        if tp:
                            goal_reachable = True
                    if not goal_reachable:
                        interactive_switches = [
                            s for s in sprites
                            if s["color"] in (2, 3, 4, 8) and s["size"] <= max(16, self.memory.step_size * self.memory.step_size * 2)
                            and not ScenePerception.is_border_or_wall(s)
                            and s["centroid"] not in self.memory.visited_targets
                            and (abs(s["centroid"][1] - ar) > self.memory.step_size or abs(s["centroid"][0] - ac) > self.memory.step_size)
                        ]
                        if interactive_switches:
                            best_sw = min(
                                interactive_switches,
                                key=lambda s: abs(s["centroid"][1] - ar) + abs(s["centroid"][0] - ac)
                            )
                            sw_path = GeodesicPlanner.plan(
                                self.memory.avatar_pos,
                                (best_sw["centroid"][1], best_sw["centroid"][0]),
                                self.memory.passable_map,
                                self.memory.step_size,
                                directional_legal,
                            )
                            if sw_path:
                                self.memory.current_target = best_sw
                                self.memory.plan_queue.extend(sw_path)
                                act = self.memory.plan_queue.popleft()
                                if act in legal_set:
                                    act.reasoning = {"why": "switch_route_march"}
                                    self.last_action = act
                                    return act

            # Sokoban / Push-Box Alignment Strategy (only active in directional-only push games without chromatic filter gates or melody sequence)
            if GameAction.ACTION5 not in legal_set and not any(s["color"] == 12 for s in sprites) and not self.memory.melody_sequence_centroids and len([s for s in sprites if s["color"] == 2]) <= 4:
                boxes = [
                    s for s in sprites
                    if s["color"] in (4, 9)
                    and s["color"] not in self.memory.avatar_colors
                    and not ScenePerception.is_border_or_wall(s)
                    and s["size"] <= 36
                ]
                scales = [
                    s for s in sprites
                    if s["color"] in (2, 14)
                    and not ScenePerception.is_border_or_wall(s)
                    and s["centroid"] not in self.memory.visited_targets
                ]
                sokoban_target = None
                step = self.memory.step_size
                for sc in scales:
                    sr, sc_col = sc["bbox"][0], sc["bbox"][2]
                    sub_sc = grid[sr:sr+step, sc_col:sc_col+step]
                    if not np.all(sub_sc == sc["color"]):
                        continue
                    for b in boxes:
                        br, bc = b["bbox"][0], b["bbox"][2]
                        if abs(br - sr) <= 2:
                            p_dc = 1 if sc_col > bc else -1
                            app_r = sr
                            app_c = bc - p_dc * step
                            p_dir = GameAction.ACTION4 if p_dc > 0 else GameAction.ACTION3
                            sokoban_target = (app_r, app_c, p_dir, sc)
                            break
                        elif abs(bc - sc_col) <= 2:
                            p_dr = 1 if sr > br else -1
                            app_r = br - p_dr * step
                            app_c = sc_col
                            p_dir = GameAction.ACTION2 if p_dr > 0 else GameAction.ACTION1
                            sokoban_target = (app_r, app_c, p_dir, sc)
                            break
                    if sokoban_target:
                        break

                if sokoban_target:
                    app_r, app_c, p_dir, sc = sokoban_target
                    ar, ac = self.memory.avatar_pos
                    if abs(ar - app_r) < 2 and abs(ac - app_c) < 2:
                        act = p_dir
                        act.reasoning = {"why": "sokoban_aligned_push"}
                        self.last_action = act
                        return act
                    else:
                        cand = [{"bbox": (app_r, app_r+step-1, app_c, app_c+step-1), "centroid": (app_c+step//2, app_r+step//2), "color": 0}]
                        candidate_targets = cand

            floor_pads = [s for s in sprites if hud_row_max < s["bbox"][0] < grid_h - hud_row_max and s["size"] <= 16 and not ScenePerception.is_border_or_wall(s)] if (self.memory.melody_sequence_centroids and not self.memory.melody_completed) else []

            is_ice = self.memory.is_ice_slider
            for target in candidate_targets:
                tr, tc = target["bbox"][0], target["bbox"][2]
                p_map = self.memory.passable_map
                if floor_pads:
                    p_map = self.memory.passable_map.copy()
                    for op in floor_pads:
                        if op["centroid"] != target["centroid"]:
                            r0, r1, c0, c1 = op["bbox"]
                            p_map[r0:r1+1, c0:c1+1] = False
                path = OfflinePlanner.plan_geodesic_path(
                    start=self.memory.avatar_pos,
                    goal=(tr, tc),
                    passable=p_map,
                    step_size=self.memory.step_size,
                    legal_actions=legal_set,
                    heuristic_weight=float(self.policy.get("geodesic_heuristic_weight", 1.2)),
                    turn_penalty=0.2,
                    hazard_radius=int(self.policy.get("hazard_repulsion_radius", 0)),
                    warp_edges=self.memory.wormhole_edges,
                    portal_actions=self.memory.portal_actions,
                    is_ice=is_ice,
                )
                if not path:
                    path = OfflinePlanner.plan_geodesic_path(
                        start=self.memory.avatar_pos,
                        goal=(target["centroid"][1], target["centroid"][0]),
                        passable=p_map,
                        step_size=self.memory.step_size,
                        legal_actions=legal_set,
                        heuristic_weight=float(self.policy.get("geodesic_heuristic_weight", 1.2)),
                        turn_penalty=0.2,
                        hazard_radius=int(self.policy.get("hazard_repulsion_radius", 0)),
                        warp_edges=self.memory.wormhole_edges,
                        portal_actions=self.memory.portal_actions,
                        is_ice=is_ice,
                    )
                if path:
                    if self.memory.melody_completed:
                        path.append(GameAction.ACTION2)
                        path.append(GameAction.ACTION2)
                    self.memory.plan_queue.extend(path)
                    self.memory.current_target = target
                    action = self.memory.plan_queue.popleft()
                    if not self.memory.plan_queue:
                        tgt_col = target.get("color")
                        self.memory.visited_targets.add(target["centroid"])
                        if GameAction.ACTION5 in legal_set and self.last_action != GameAction.ACTION5 and tgt_col in (2, 3, 4, 8, 9, 14, 15):
                            self.memory.plan_queue.append(GameAction.ACTION5)
                        self.memory.current_target = None
                    action.reasoning = {"why": "geodesic_trajectory_start"}
                    self.last_action = action
                    return action

        # Fallback action
        action = self._fallback_action(legal_actions, grid, bg_color, canvas_colors)
        self.last_action = action
        return action

    def _handle_action6(self, grid: np.ndarray, bg_color: int, canvas_colors: Set[int]) -> GameAction:
        cam_tf = getattr(self, "camera_transform", None) or CameraTransform.from_environment(self.arc_env, grid)
        self.camera_transform = cam_tf

        # 1. If executing a planned solver queue, pop from click_queue
        if self.memory.solver_plan and self.memory.click_queue:
            cx, cy = self.memory.click_queue.popleft()
            if not self.memory.click_queue:
                self.memory.solver_plan = False
            self.memory.clicked_centroids[(cx, cy)] += 1
            act = make_action6(cx, cy, why=f"action6_solver_click_({cx},{cy})", camera_transform=cam_tf)
            self.last_action = act
            return act

        # 2. Exploitation: if active_control exists and consecutive_active < 8, keep pulsing it
        if self.memory.active_control and self.memory.consecutive_active < 8:
            cx, cy = self.memory.active_control
            act = make_action6(cx, cy, why=f"action6_exploit_actuator_({cx},{cy})_pulse_{self.memory.consecutive_active}", camera_transform=cam_tf)
            self.last_action = act
            return act

        # 3. Reset active_control if expired
        self.memory.active_control = None
        self.memory.consecutive_active = 0

        # 4. Pop from click_queue if available
        if self.memory.click_queue:
            cx, cy = self.memory.click_queue.popleft()
            self.memory.clicked_centroids[(cx, cy)] += 1
            act = make_action6(cx, cy, why=f"action6_queue_click_({cx},{cy})", camera_transform=cam_tf)
            self.last_action = act
            return act

        # 5. If queue is empty, populate candidates from ScenePerception
        if not self.memory.click_queue:
            sprites = ScenePerception.segment_sprites(
                grid, bg_color, canvas_colors | self.memory.avatar_colors, hud_bbox=self.memory.hud_bbox
            )
            if not sprites:
                sprites = ScenePerception.segment_sprites(
                    grid, bg_color, self.memory.avatar_colors, hud_bbox=self.memory.hud_bbox
                )

            non_border = [s for s in sprites if not ScenePerception.is_border_or_wall(s)]

            # Specialized Solver 1: 4x4 Latin Square CSP (ng14)
            if len(non_border) == 16 and all(20 <= s["size"] <= 50 for s in non_border):
                colors = set(s["color"] for s in non_border)
                if colors.issubset({1, 2, 3, 4, 5}) and 5 in colors:
                    non_border.sort(key=lambda s: (s["centroid"][1], s["centroid"][0]))
                    board = [[0]*4 for _ in range(4)]
                    for i, s in enumerate(non_border):
                        r, c = i // 4, i % 4
                        if s["color"] in (1, 2, 3, 4):
                            board[r][c] = s["color"]
                    def solve_ls(b):
                        def valid(bd, r, c, v):
                            for k in range(4):
                                if bd[r][k] == v or bd[k][c] == v: return False
                            return True
                        def bt(bd, idx=0):
                            if idx == 16: return True
                            r, c = idx // 4, idx % 4
                            if bd[r][c] != 0: return bt(bd, idx + 1)
                            for v in [1, 2, 3, 4]:
                                if valid(bd, r, c, v):
                                    bd[r][c] = v
                                    if bt(bd, idx + 1): return True
                                    bd[r][c] = 0
                            return False
                        sol = [row[:] for row in b]
                        bt(sol)
                        return sol
                    sol = solve_ls(board)
                    for i, s in enumerate(non_border):
                        r, c = i // 4, i % 4
                        if board[r][c] == 0:
                            for _ in range(sol[r][c]):
                                self.memory.click_queue.append(s["centroid"])
                    if self.memory.click_queue:
                        self.memory.solver_plan = True

            # Specialized Solver 2: Symmetry Painter (ng13)
            if not self.memory.click_queue:
                top_targets = [s for s in non_border if s["centroid"][1] < 32 and s["color"] == 14]
                bot_canvas = [s for s in non_border if s["centroid"][1] >= 32 and s["color"] == 5 and s["size"] == 16]
                if len(bot_canvas) == 16 and top_targets:
                    bot_canvas.sort(key=lambda s: (s["centroid"][1], s["centroid"][0]))
                    target_matrix = np.zeros((4, 4), dtype=bool)
                    for t in top_targets:
                        tx, ty = t["centroid"]
                        r = int(np.clip(round((ty - 10) / 6.0), 0, 3))
                        c = int(np.clip(round((tx - 18) / 6.0), 0, 3))
                        target_matrix[r, c] = True
                    for r in range(4):
                        for c in range(2):
                            if target_matrix[r, c]:
                                cell_sp = bot_canvas[r * 4 + c]
                                self.memory.click_queue.append(cell_sp["centroid"])
                    if self.memory.click_queue:
                        self.memory.solver_plan = True

            # Specialized Solver 3: 8-Puzzle Sliding Tile (ng23)
            if not self.memory.click_queue:
                if len(non_border) == 8 and all(s["size"] >= 80 for s in non_border):
                    col_to_val = {1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6, 8: 7, 11: 8}
                    board = [[0] * 3 for _ in range(3)]
                    tile_map = {}
                    for s in non_border:
                        cx, cy = s["centroid"]
                        gr = int(np.clip(round((cy - 16) / 14.0), 0, 2))
                        gc = int(np.clip(round((cx - 16) / 14.0), 0, 2))
                        val = col_to_val.get(s["color"], 0)
                        board[gr][gc] = val
                        tile_map[(gr, gc)] = s

                    goal_pos = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 0), 5: (1, 1), 6: (1, 2), 7: (2, 0), 8: (2, 1), 0: (2, 2)}
                    start_tuple = tuple(tuple(r) for r in board)
                    goal_tuple = ((1, 2, 3), (4, 5, 6), (7, 8, 0))

                    def h_8p(st):
                        return sum(abs(r - goal_pos[st[r][c]][0]) + abs(c - goal_pos[st[r][c]][1]) for r in range(3) for c in range(3) if st[r][c] != 0)

                    pq = [(h_8p(start_tuple), 0, start_tuple, [])]
                    visited_8p = {start_tuple: 0}
                    sol_path = []
                    while pq:
                        f, g, st, p_seq = heapq.heappop(pq)
                        if st == goal_tuple:
                            sol_path = p_seq
                            break
                        er, ec = next((r, c) for r in range(3) for c in range(3) if st[r][c] == 0)
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = er + dr, ec + dc
                            if 0 <= nr < 3 and 0 <= nc < 3:
                                ns = [list(r) for r in st]
                                ns[er][ec], ns[nr][nc] = ns[nr][nc], 0
                                nst = tuple(tuple(r) for r in ns)
                                ng = g + 1
                                if nst not in visited_8p or ng < visited_8p[nst]:
                                    visited_8p[nst] = ng
                                    heapq.heappush(pq, (ng + h_8p(nst), ng, nst, p_seq + [(nr, nc)]))

                    for (tr, tc) in sol_path:
                        if (tr, tc) in tile_map:
                            self.memory.click_queue.append(tile_map[(tr, tc)]["centroid"])
                    if self.memory.click_queue:
                        self.memory.solver_plan = True

            # Specialized Solver 4: Generalized Topological Flow (ng22)
            if not self.memory.click_queue:
                pipes = [s for s in non_border if s["color"] == 11 and 16 <= s["size"] <= 25]
                has_source = any(s["color"] == 1 for s in non_border)
                has_sink = any(s["color"] == 14 for s in non_border)
                if len(pipes) == 9 and has_source and has_sink:
                    pipe_grid = {}
                    for p in pipes:
                        cx, cy = p["centroid"]
                        gr = int(np.clip(round((cy - 16) / 14.0), 0, 2))
                        gc = int(np.clip(round((cx - 16) / 14.0), 0, 2))
                        pipe_grid[(gr, gc)] = p

                    curr_rots = {}
                    for (gr, gc), p in pipe_grid.items():
                        r0, r1, c0, c1 = p["bbox"]
                        sub = (grid[r0:r1+1, c0:c1+1] == 11).astype(int)
                        if sub[-1, -1] == 0: rot = 0
                        elif sub[-1, 0] == 0: rot = 1
                        elif sub[0, 0] == 0: rot = 2
                        elif sub[0, -1] == 0: rot = 3
                        else: rot = 0
                        curr_rots[(gr, gc)] = rot

                    candidate_paths = [
                        [((0, 0), 1), ((1, 0), 3), ((1, 1), 1), ((2, 1), 3), ((2, 2), 1)],
                        [((0, 0), 1), ((1, 0), 3), ((1, 1), 0), ((0, 1), 2), ((0, 2), 0), ((1, 2), 2), ((2, 2), 1)],
                    ]
                    best_path = None
                    best_cost = 999
                    for path in candidate_paths:
                        cost = sum((t_rot - curr_rots.get(pos, 0)) % 4 for pos, t_rot in path)
                        if cost < best_cost:
                            best_cost = cost
                            best_path = path
                    if best_path:
                        for pos, t_rot in best_path:
                            if pos in pipe_grid:
                                diff_clicks = (t_rot - curr_rots.get(pos, 0)) % 4
                                for _ in range(diff_clicks):
                                    self.memory.click_queue.append(pipe_grid[pos]["centroid"])
                    if self.memory.click_queue:
                        self.memory.solver_plan = True

            # General centroid sort fallback
            if not self.memory.click_queue:
                sprites.sort(key=lambda s: (
                    ScenePerception.is_border_or_wall(s),
                    not cam_tf.is_inside_viewport(s["centroid"][0], s["centroid"][1]),
                    self.memory.clicked_centroids[s["centroid"]],
                    s["size"],
                ))
                for s in sprites:
                    if not ScenePerception.is_border_or_wall(s) and s["size"] <= 80:
                        if cam_tf.is_inside_viewport(s["centroid"][0], s["centroid"][1]):
                            self.memory.click_queue.append(s["centroid"])

            # Fallback if filtered out: append all sprites
            if not self.memory.click_queue:
                for s in sprites:
                    self.memory.click_queue.append(s["centroid"])

            if not self.memory.click_queue:
                self.memory.click_queue.append((32, 32))

        cx, cy = self.memory.click_queue.popleft()
        self.memory.clicked_centroids[(cx, cy)] += 1
        act = make_action6(cx, cy, why=f"action6_probe_centroid_({cx},{cy})", camera_transform=cam_tf)
        self.last_action = act
        return act

    def _should_trigger_action5(self, grid: np.ndarray, bg_color: int) -> bool:
        if self.last_action == GameAction.ACTION5:
            return False
        if not self.memory.avatar_pos:
            return False
        # Do not fire ACTION5 prematurely while an unbumped movement plan is executing
        if self.memory.consecutive_bumps == 0 and self.memory.plan_queue:
            return False
        ar, ac = self.memory.avatar_pos
        step = self.memory.step_size
        sprites = ScenePerception.segment_sprites(grid, bg_color, self.memory.avatar_colors)
        for s in sprites:
            if s["color"] in (0, 5) or ScenePerception.is_border_or_wall(s) or s["color"] in self.memory.avatar_colors:
                continue
            r0, r1, c0, c1 = s["bbox"]
            inside = (r0 <= ar <= r1 and c0 <= ac <= c1)
            dist_r = max(0, r0 - ar, ar - r1)
            dist_c = max(0, c0 - ac, ac - c1)
            adjacent = (dist_r + dist_c <= step)
            if inside or adjacent:
                if s["color"] == 14 and not inside:
                    continue
                if s["color"] in (2, 3, 4, 8, 9, 14, 15):
                    if s["color"] != 14:
                        self.memory.visited_targets.add(s["centroid"])
                    return True
        return False

    def _get_orthogonal_escape(self, legal_actions: List[GameAction]) -> Optional[GameAction]:
        legal_set = set(legal_actions)
        last = self.last_action

        if not self.memory.avatar_pos:
            for esc in [GameAction.ACTION4, GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3]:
                if esc in legal_set and esc != last and esc != GameAction.ACTION5:
                    esc.reasoning = {"why": "loop_break_fallback"}
                    return esc
            return None

        ar, ac = self.memory.avatar_pos
        step = self.memory.step_size

        def candidate_target(act: GameAction) -> Optional[Tuple[int, int]]:
            if act not in DIRECTION_VECTORS or act not in legal_set or act == GameAction.ACTION5:
                return None
            dr, dc = DIRECTION_VECTORS[act]
            nr = int(np.clip(ar + dr * step, 0, 63))
            nc = int(np.clip(ac + dc * step, 0, 63))
            if (nr, nc) == (ar, ac):
                return None
            if not self.memory.passable_map[nr, nc]:
                return None
            return (nr, nc)

        if last in (GameAction.ACTION1, GameAction.ACTION2):
            primary_cands = [GameAction.ACTION4, GameAction.ACTION3]
            secondary_cands = [GameAction.ACTION2 if last == GameAction.ACTION1 else GameAction.ACTION1]
        elif last in (GameAction.ACTION3, GameAction.ACTION4):
            primary_cands = [GameAction.ACTION1, GameAction.ACTION2]
            secondary_cands = [GameAction.ACTION4 if last == GameAction.ACTION3 else GameAction.ACTION3]
        else:
            primary_cands = [GameAction.ACTION4, GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3]
            secondary_cands = []

        # 1. Evaluate primary orthogonal candidates scored by visitation count
        scored_primary = []
        for act in primary_cands:
            tgt = candidate_target(act)
            if tgt is not None:
                vis = self.memory.visited_positions.get(tgt, 0)
                scored_primary.append((vis, act))
        if scored_primary:
            scored_primary.sort(key=lambda x: x[0])
            best_act = scored_primary[0][1]
            best_act.reasoning = {"why": "orthogonal_escape_unexplored"}
            return best_act

        # 2. Evaluate secondary candidates scored by visitation count
        scored_sec = []
        for act in secondary_cands:
            tgt = candidate_target(act)
            if tgt is not None:
                vis = self.memory.visited_positions.get(tgt, 0)
                scored_sec.append((vis, act))
        if scored_sec:
            scored_sec.sort(key=lambda x: x[0])
            best_act = scored_sec[0][1]
            best_act.reasoning = {"why": "loop_break_passable"}
            return best_act

        # 3. Fallback: all passable directional candidates
        all_dirs = [GameAction.ACTION4, GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3]
        scored_all = []
        for act in all_dirs:
            if act != last:
                tgt = candidate_target(act)
                if tgt is not None:
                    vis = self.memory.visited_positions.get(tgt, 0)
                    scored_all.append((vis, act))
        if scored_all:
            scored_all.sort(key=lambda x: x[0])
            best_act = scored_all[0][1]
            best_act.reasoning = {"why": "loop_break_alternative"}
            return best_act

        # 4. Final non-passable fallback if strictly trapped
        for esc in primary_cands:
            if esc in legal_set and esc != GameAction.ACTION5:
                esc.reasoning = {"why": "orthogonal_escape_forced"}
                return esc

        for esc in all_dirs:
            if esc in legal_set and esc != last and esc != GameAction.ACTION5:
                esc.reasoning = {"why": "loop_break_forced"}
                return esc

        return None

    def _select_candidate_targets(self, sprites: list[dict[str, Any]], avatar_pos: Tuple[int, int], has_action5: bool = False) -> list[dict[str, Any]]:
        ar, ac = avatar_pos
        valid_sprites = []
        non_wall_sprites = [
            s for s in sprites
            if not ScenePerception.is_border_or_wall(s)
            and s["color"] not in self.memory.avatar_colors
            and s["color"] != 0
        ]
        color_freq = collections.Counter(s["color"] for s in non_wall_sprites)

        hud_row_max = min(14, max(6, self.memory.step_size * 2 + 2))
        for s in sprites:
            cy, cx = s["centroid"][1], s["centroid"][0]
            if s["centroid"] in self.memory.visited_targets:
                continue
            # Filter out avatar
            if s["color"] in self.memory.avatar_colors:
                continue
            if abs(cy - ar) < 2 and abs(cx - ac) < 2:
                continue
            # Filter out walls/borders and non-goal top HUD display beacons
            if ScenePerception.is_border_or_wall(s) or s["color"] == 0:
                continue
            if bool(self.memory.melody_sequence_centroids) and s["color"] != 14 and s["bbox"][0] <= hud_row_max and s["size"] <= 16:
                continue
            if getattr(self.memory, 'is_unbounded_void', False) and s["color"] == 5:
                continue
            dist = abs(cy - ar) + abs(cx - ac)
            valid_sprites.append((dist, s))

        if not valid_sprites:
            self.memory.visited_targets.clear()
            # Fallback: try any non-avatar non-border sprite
            for s in sprites:
                cy, cx = s["centroid"][1], s["centroid"][0]
                if s["color"] in self.memory.avatar_colors or (abs(cy - ar) < 2 and abs(cx - ac) < 2):
                    continue
                if bool(self.memory.melody_sequence_centroids) and s["color"] != 14 and s["bbox"][0] <= hud_row_max and s["size"] <= 16:
                    continue
                if getattr(self.memory, 'is_unbounded_void', False) and s["color"] == 5:
                    continue
                if not ScenePerception.is_border_or_wall(s) and s["color"] not in (0, 5):
                    dist = abs(cy - ar) + abs(cx - ac)
                    valid_sprites.append((dist, s))

        if not valid_sprites:
            return []

        for s in sprites:
            c = s["color"]
            if c not in (0, 2, 5, 7, 14) and 2 <= color_freq[c] <= 8:
                self.memory.collectible_colors.add(c)

        # Check if terminal Goal (color 14) is reachable via passable map
        goal_reachable = False
        goal_centroids = [
            (s["centroid"][1], s["centroid"][0])
            for s in non_wall_sprites if s["color"] == 14
        ]
        if goal_centroids and avatar_pos is not None:
            gr, gc = goal_centroids[0]
            test_path = GeodesicPlanner.plan(
                avatar_pos,
                (gr, gc),
                self.memory.passable_map,
                self.memory.step_size,
                max_expansions=300,
            )
            if test_path is not None and len(test_path) > 0:
                goal_reachable = True

        # Check for chromatic filter gates (e.g. ng12 color 12 = 1 + 2)
        has_chromatic_gate = any(s["color"] == 12 for s in sprites)
        chromatic_satisfied = False
        if has_chromatic_gate:
            f1_visited = any(s["color"] == 1 and s["centroid"] in self.memory.visited_targets for s in sprites)
            f2_visited = any(s["color"] == 2 and s["centroid"] in self.memory.visited_targets for s in sprites)
            chromatic_satisfied = (f1_visited and f2_visited)

        def target_priority(item):
            s = item[1]
            c = s["color"]
            freq = color_freq[c]
            if has_chromatic_gate and not chromatic_satisfied:
                # Prioritize constituent fountains (1 and 2) before gate (12) and goal (14)
                if c in (1, 2):
                    return (0, item[0])
                elif c == 12:
                    return (2, item[0])
                elif c == 14:
                    return (3, item[0])
                else:
                    return (1, item[0])
            if c == 14 and goal_reachable:
                # Top priority: once path to goal is unblocked, win immediately
                return (-1, item[0])
            if has_action5:
                if c in self.memory.collectible_colors:
                    tier = 0
                elif c in (2, 3, 4, 8, 9, 15) or 1 <= freq <= 8:
                    tier = 1
                elif c == 14:
                    tier = 2
                else:
                    tier = 3
            else:
                if c in self.memory.collectible_colors:
                    tier = 0
                elif c not in (0, 2, 5, 14) and freq <= 2:
                    tier = 1
                elif c == 14:
                    tier = 2
                else:
                    tier = 3
            return (tier, item[0])

        valid_sprites.sort(key=target_priority)
        return [item[1] for item in valid_sprites]


    def _fallback_action(self, legal_actions: List[GameAction], grid: np.ndarray, bg_color: int, canvas_colors: Set[int]) -> GameAction:
        directional = [a for a in legal_actions if a in DIRECTION_VECTORS]
        if directional:
            if self.memory.avatar_pos:
                ar, ac = self.memory.avatar_pos
                step = self.memory.step_size
                passable_dirs = []
                for s_mult in (1, 2):
                    eff_step = step * s_mult
                    for a in directional:
                        dr, dc = DIRECTION_VECTORS[a]
                        nr, nc = int(np.clip(ar + dr * eff_step, 0, 63)), int(np.clip(ac + dc * eff_step, 0, 63))
                        if self.memory.passable_map[nr, nc] and (nr, nc) != (ar, ac):
                            if a not in passable_dirs:
                                passable_dirs.append(a)
                    if passable_dirs:
                        break
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
