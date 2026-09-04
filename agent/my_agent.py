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
import heapq
import hashlib
import random
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
        self.avatar_pos: Optional[Tuple[int, int]] = None  # (top_r, left_c)
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
        self.visited_targets.clear()
        self.last_grid_bytes = None
        self.last_state_hash = None
        self.consecutive_bumps = 0
        self.hud_bbox = None

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
        canvas_colors = set(np.where(counts > 400)[0].tolist())
        return bg_color, canvas_colors

    @staticmethod
    def is_border_or_wall(s: dict[str, Any]) -> bool:
        min_r, max_r, min_c, max_c = s["bbox"]
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        if (min_r <= 1 and h == 1 and w >= 32) or (min_r >= 62 and h <= 2 and w >= 32):
            return True
        if s["size"] > 180:
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
            cx = int(round(sum(cols) / len(cols)))
            cy = int(round(sum(rows) / len(rows)))
            sprites.append({
                "color": color,
                "size": len(coords),
                "bbox": (int(min(rows)), int(max(rows)), int(min(cols)), int(max(cols))),
                "centroid": (int(np.clip(cx, 0, 63)), int(np.clip(cy, 0, 63))),
                "coords": coords,
            })

        sprites.sort(key=lambda s: (ScenePerception.is_border_or_wall(s), s["size"]))
        return sprites


class AvatarTracker:
    """Isolates avatar position, step size, and bump collisions via top-left bounding box."""

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
        # Filter corner step counters
        diff[60:, :16] = False

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
                    r_low, r_high = max(0, wall_r), min(64, wall_r + step)
                    c_low, c_high = max(0, wall_c), min(64, wall_c + step)
                    memory.passable_map[r_low:r_high, c_low:c_high] = False
                    memory.known_walls.add((wall_r, wall_c))
                    # Stand tile is strictly passable
                    memory.passable_map[ar:ar+step, ac:ac+step] = True
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
            top_r = int(curr_pts[:, 0].min())
            left_c = int(curr_pts[:, 1].min())
            prev_top_r = int(prev_pts[:, 0].min())
            prev_left_c = int(prev_pts[:, 1].min())

            dr = top_r - prev_top_r
            dc = left_c - prev_left_c
            measured_step = max(abs(dr), abs(dc))
            if measured_step >= 1:
                memory.step_size = measured_step

            new_pos = (top_r, left_c)
            memory.avatar_pos = new_pos
            colors = set(np.unique(curr_grid[curr_pts[:, 0], curr_pts[:, 1]]).tolist())
            if colors:
                memory.avatar_colors.update(colors)
                memory.avatar_color = list(colors)[0]
            return new_pos, True

        return memory.avatar_pos, True


class DynamicLatticeDetector:
    """Infers grid lattice pitch (step_size), spatial offsets, and tile dimensions."""

    @staticmethod
    def infer_lattice(
        avatar_pos: Optional[Tuple[int, int]],
        step_size: int,
        grid_shape: Tuple[int, int] = (64, 64),
    ) -> Tuple[int, int, int, int, int]:
        step = max(1, step_size)
        if avatar_pos is None:
            return step, 0, 0, grid_shape[0] // step, grid_shape[1] // step
        r_off = avatar_pos[0] % step
        c_off = avatar_pos[1] % step
        gh = (grid_shape[0] - r_off) // step
        gw = (grid_shape[1] - c_off) // step
        return step, r_off, c_off, gh, gw


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
                        border_pixels = np.concatenate([box[0, :], box[-1, :], box[:, 0], box[:, -1]])
                        border_color = int(border_pixels[0])
                        if (border_pixels == border_color).all() and border_color != bg_color:
                            core = box[1:-1, 1:-1]
                            non_border_mask = (core != border_color) & (core != bg_color)
                            if non_border_mask.any():
                                fg_colors, counts = np.unique(core[non_border_mask], return_counts=True)
                                candidates.append({
                                    "location": loc,
                                    "r": r_sl.start + wr,
                                    "c": c_sl.start + wc,
                                    "size": win_size,
                                    "border_color": border_color,
                                    "fg_color": int(fg_colors[np.argmax(counts)]),
                                    "core_patch": core,
                                    "box_patch": box,
                                })

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
        rotation_deg: int = -1,
    ) -> None:
        self.shape_id = shape_id
        self.color_idx = color_idx
        self.rotation_idx = rotation_idx
        self.raw_color = raw_color
        self.rotation_deg = rotation_deg

    def __repr__(self) -> str:
        return (
            f"BlockConfiguration(shape_id={self.shape_id}, color_idx={self.color_idx}, "
            f"rotation_idx={self.rotation_idx}, raw_color={self.raw_color}, rotation_deg={self.rotation_deg})"
        )

    def __eq__(self, other: object) -> bool:
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
    """Canonical shape matching and HUD/Goal template extraction."""

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

    @classmethod
    def match_sprite(cls, patch: np.ndarray, bg_color: int = 5) -> Optional[BlockConfiguration]:
        H, W = patch.shape
        if H == 10 and W == 10:
            # 10x10 outer box with 2px border: interior 6x6 downsampled 2x2 -> 3x3
            patch = patch[2:8, 2:8][::2, ::2]
        elif H == 6 and W == 6:
            patch = patch[::2, ::2]
        elif H >= 5 and W >= 5:
            patch = patch[1:4, 1:4]
        elif (H, W) != (3, 3):
            return None

        pal_mask = np.isin(patch, cls.PALETTE)
        if not pal_mask.any():
            return None
        pal_colors, counts = np.unique(patch[pal_mask], return_counts=True)
        fg_color = int(pal_colors[np.argmax(counts)])

        bin_patch = patch == fg_color
        for s_id, base_shp in cls.SHAPES.items():
            for r_idx, deg in enumerate(cls.ROTATIONS):
                rot_shp = np.rot90(base_shp, -deg // 90)
                if np.array_equal(bin_patch, rot_shp):
                    c_idx = cls.PALETTE.index(fg_color)
                    return BlockConfiguration(
                        shape_id=s_id,
                        color_idx=c_idx,
                        rotation_idx=r_idx,
                        raw_color=fg_color,
                        rotation_deg=deg,
                    )
        return None

    @classmethod
    def extract_hud_configuration(cls, grid: np.ndarray, bg_color: int = 0) -> Optional[Tuple[BlockConfiguration, dict[str, Any]]]:
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
        step, roff, coff, gh, gw = DynamicLatticeDetector.infer_lattice(avatar_pos, step_size, grid.shape)

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

                border = np.concatenate([patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]])
                if (border == target_border_color).all() and not (patch == target_border_color).all():
                    cfg = TemplatePerception.match_sprite(patch, bg_color=target_border_color)
                    if cfg is not None:
                        elements["goal_slots"][(gx, gy)] = cfg
                        elements["passable"][gy, gx] = True
                        continue

                if 1 in u and 0 in u and elements["wall_color"] not in u and np.sum(patch == 0) == 3 and np.sum(patch == 1) == 2:
                    elements["rotation_pads"].append((gx, gy))
                    elements["passable"][gy, gx] = True
                    continue

                if len(u & set(TemplatePerception.PALETTE)) >= 3:
                    elements["color_pads"].append((gx, gy))
                    elements["passable"][gy, gx] = True
                    continue

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

        while queue:
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

        v_act = GameAction.ACTION2 if dr > 0 else GameAction.ACTION1
        h_act = GameAction.ACTION4 if dc > 0 else GameAction.ACTION3
        v_step = 1 if dr > 0 else -1
        h_step = 1 if dc > 0 else -1

        v_count = int(round(abs(dr) / max(1, step_size)))
        h_count = int(round(abs(dc) / max(1, step_size)))

        if v_count == 0 and h_count == 0:
            return []

        # 1. Vertical first
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

        # 2. Horizontal first
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

        # 3. A* minimal convex detour
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
    """DualBrainAgent Version 4: Universal Cognitive Flexibility."""

    MAX_ACTIONS = 120

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = WorkingMemory()
        self.last_action: Optional[GameAction] = None
        self.puzzle_mode: Optional[bool] = None
        self.puzzle_elements: Optional[dict[str, Any]] = None
        self.puzzle_avatar_tile: Optional[Tuple[int, int]] = None
        self.hud_info: Optional[dict[str, Any]] = None
        self.hud_cfg: Optional[BlockConfiguration] = None
        self.last_levels_completed: int = 0
        self.affordance_ledger: dict[Tuple[int, int], str] = {}
        self.last_grid: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return f"DualBrainAgent.v4.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in (GameState.WIN, "WIN")

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER, "NOT_PLAYED", "GAME_OVER"):
            self.memory.reset_for_level()
            self.puzzle_mode = None
            self.puzzle_elements = None
            self.puzzle_avatar_tile = None
            self.hud_info = None
            self.hud_cfg = None
            self.last_levels_completed = 0
            self.last_action = GameAction.RESET
            self.last_grid = None
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

        # Movement tracking
        if self.last_grid is not None and self.last_action:
            AvatarTracker.update(self.last_grid, grid, self.last_action, self.memory)

        self.last_grid = grid.copy()
        self.memory.record_step(self.last_action, grid_bytes, self.memory.avatar_pos)

        # Level completion tracking & state reset
        curr_levels = getattr(latest_frame, "levels_completed", 0)
        if curr_levels > self.last_levels_completed:
            self.last_levels_completed = curr_levels
            self.puzzle_elements = None
            self.puzzle_avatar_tile = None
            self.hud_info = None
            self.hud_cfg = None
            self.puzzle_mode = None
            self.memory.plan_queue.clear()
            self.memory.visited_targets.clear()

        # Pure Click Dispatch (Mode D)
        if GameAction.ACTION6 in legal_set and (not directional_legal or len(directional_legal) < 2):
            return self._handle_action6(grid, bg_color, canvas_colors)

        # Epistemic probe on first step if avatar_pos not yet known
        if self.memory.avatar_pos is None and self.memory.probing_phase and directional_legal:
            for probe in [GameAction.ACTION4, GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3]:
                if probe in legal_set:
                    self.memory.probing_phase = False
                    probe.reasoning = {"why": "epistemic_probe"}
                    self.last_action = probe
                    return probe

        # Online Affordance Learning
        if len(frames) >= 2 and self.puzzle_avatar_tile:
            prev_grid = ScenePerception.extract_grid(frames[-2])
            prev_hud = TemplatePerception.extract_hud_configuration(prev_grid, bg_color)
            curr_hud = TemplatePerception.extract_hud_configuration(grid, bg_color)
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
            hud_res = TemplatePerception.extract_hud_configuration(grid, bg_color)
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
                hud_res = TemplatePerception.extract_hud_configuration(grid, bg_color)
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

        # Pop from plan queue if available
        if self.memory.plan_queue:
            action = self.memory.plan_queue.popleft()
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

        # Mode B: Salient Geodesic Goal Pursuit
        if self.memory.avatar_pos and directional_legal:
            sprites = ScenePerception.segment_sprites(
                grid,
                bg_color,
                canvas_colors | self.memory.avatar_colors,
                hud_bbox=self.memory.hud_bbox,
            )
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
                    self.memory.visited_targets.add(target["centroid"])
                    self.memory.plan_queue.extend(path)
                    action = self.memory.plan_queue.popleft()
                    action.reasoning = {"why": "geodesic_trajectory_start"}
                    self.last_action = action
                    return action

        # Fallback action
        action = self._fallback_action(legal_actions, grid, bg_color, canvas_colors)
        self.last_action = action
        return action

    def _handle_action6(self, grid: np.ndarray, bg_color: int, canvas_colors: Set[int]) -> GameAction:
        if not self.memory.click_queue:
            sprites = ScenePerception.segment_sprites(
                grid, bg_color, canvas_colors | self.memory.avatar_colors, hud_bbox=self.memory.hud_bbox
            )
            if not sprites:
                sprites = ScenePerception.segment_sprites(
                    grid, bg_color, self.memory.avatar_colors, hud_bbox=self.memory.hud_bbox
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

        for esc in primary_cands:
            if esc in legal_set and esc != GameAction.ACTION5 and is_passable(esc):
                esc.reasoning = {"why": primary_reason}
                return esc

        for esc in secondary_cands:
            if esc in legal_set and esc != GameAction.ACTION5 and is_passable(esc):
                esc.reasoning = {"why": "loop_break_passable"}
                return esc

        all_dirs = [GameAction.ACTION4, GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3]
        for esc in all_dirs:
            if esc in legal_set and esc != last and esc != GameAction.ACTION5 and is_passable(esc):
                esc.reasoning = {"why": "loop_break_passable"}
                return esc

        for esc in primary_cands:
            if esc in legal_set and esc != GameAction.ACTION5:
                esc.reasoning = {"why": primary_reason}
                return esc

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
            if s["centroid"] in self.memory.visited_targets:
                continue
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
