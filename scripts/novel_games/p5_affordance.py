#!/usr/bin/env python3
"""
Paradigm 5: Affordance & Point-and-Click Mechanisms
Contains 5 novel ARC-AGI-3 game environments:
- ng21_tumbler_lock: 3-cylinder combination dial alignment via click affordance arrows.
- ng22_pipe_flow: Rotational topological network alignment routing liquid from source to sink.
- ng23_sliding_tile_8: Classic 8-puzzle permutation matrix solver via adjacent tile clicks.
- ng24_optical_beam_split: 45-degree angled laser mirror rotation lighting target crystals.
- ng25_resource_crafting: Finite-state inventory recipe synthesis (Wood + Ore -> Key -> Exit).
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple, Dict, Any, Set
import numpy as np

from arcengine import (
    ARCBaseGame,
    BlockingMode,
    Camera,
    GameAction,
    GameState,
    Level,
    Sprite,
    ActionInput,
)


class NG21_TumblerLock(ARCBaseGame):
    """3-cylinder combination dial alignment via click affordance arrows."""

    def __init__(self, camera_res: int = 32, max_steps: int = 40, seed: int = 42) -> None:
        self.camera_res = camera_res
        self.max_steps = max_steps
        self.steps_left = max_steps

        bg_color = 0
        arrow_color = 11

        self.target = [2, 3, 4]
        self.current_vals = [4, 2, 3]

        sprites: List[Sprite] = []
        self.col_xs = [8, 14, 20]

        for i, cx in enumerate(self.col_xs):
            sprites.append(Sprite(
                pixels=np.full((3, 3), self.target[i], dtype=np.int8).tolist(),
                name=f"tgt_{i}",
                x=cx,
                y=4,
                tags=["hud_target"],
                blocking=BlockingMode.NOT_BLOCKED,
            ))

        for i, cx in enumerate(self.col_xs):
            sprites.append(Sprite(
                pixels=np.full((2, 3), arrow_color, dtype=np.int8).tolist(),
                name=f"btn_up_{i}",
                x=cx,
                y=9,
                tags=["button", f"up_{i}"],
                blocking=BlockingMode.NOT_BLOCKED,
            ))

            sprites.append(Sprite(
                pixels=np.full((4, 4), self.current_vals[i], dtype=np.int8).tolist(),
                name=f"tumbler_{i}",
                x=cx,
                y=13,
                tags=["tumbler"],
                blocking=BlockingMode.NOT_BLOCKED,
            ))

            sprites.append(Sprite(
                pixels=np.full((2, 3), arrow_color, dtype=np.int8).tolist(),
                name=f"btn_down_{i}",
                x=cx,
                y=19,
                tags=["button", f"down_{i}"],
                blocking=BlockingMode.NOT_BLOCKED,
            ))

        level = Level(sprites=sprites, grid_size=(camera_res, camera_res))
        camera = Camera(width=camera_res, height=camera_res, background=bg_color)
        super().__init__(
            game_id="ng21_tumbler_lock",
            levels=[level],
            camera=camera,
            available_actions=[6],
            seed=seed,
        )

    def get_sprite(self, name: str) -> Sprite:
        return self.current_level.get_sprites_by_name(name)[0]

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return

        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        if self._action.id == GameAction.ACTION6:
            disp_x = self._action.data.get("x", 0)
            disp_y = self._action.data.get("y", 0)
            grid_pos = self.camera.display_to_grid(disp_x, disp_y)
            if grid_pos is not None:
                gx, gy = grid_pos
                for i in range(3):
                    up_sp = self.get_sprite(f"btn_up_{i}")
                    if up_sp.x <= gx < up_sp.x + up_sp.width and up_sp.y <= gy < up_sp.y + up_sp.height:
                        curr = self.current_vals[i]
                        nxt = ((curr - 1 + 1) % 4) + 1
                        self.current_vals[i] = nxt
                        t_sp = self.get_sprite(f"tumbler_{i}")
                        t_sp.pixels = np.full((4, 4), nxt, dtype=np.int8)
                        break

                    down_sp = self.get_sprite(f"btn_down_{i}")
                    if down_sp.x <= gx < down_sp.x + down_sp.width and down_sp.y <= gy < down_sp.y + down_sp.height:
                        curr = self.current_vals[i]
                        nxt = ((curr - 1 - 1) % 4) + 1
                        self.current_vals[i] = nxt
                        t_sp = self.get_sprite(f"tumbler_{i}")
                        t_sp.pixels = np.full((4, 4), nxt, dtype=np.int8)
                        break

        if self.current_vals == self.target:
            self.win()

        self.complete_action()


class NG22_PipeFlow(ARCBaseGame):
    """Rotational topological network alignment routing liquid from source to sink."""

    def __init__(self, camera_res: int = 32, max_steps: int = 40, seed: int = 42) -> None:
        self.camera_res = camera_res
        self.max_steps = max_steps
        self.steps_left = max_steps

        bg_color = 0
        source_color = 1
        sink_color = 14

        sprites: List[Sprite] = []

        sprites.append(Sprite(
            pixels=np.full((3, 3), source_color, dtype=np.int8).tolist(),
            name="source",
            x=2,
            y=8,
            tags=["source"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        sprites.append(Sprite(
            pixels=np.full((3, 3), sink_color, dtype=np.int8).tolist(),
            name="sink",
            x=20,
            y=27,
            tags=["sink"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        self.rotations = [1, 2, 0, 3, 1, 0, 2, 0, 3]

        for r in range(3):
            for c in range(3):
                idx = r * 3 + c
                rot = self.rotations[idx]
                sprites.append(Sprite(
                    pixels=self._render_pipe_pixel(rot).tolist(),
                    name=f"pipe_{r}_{c}",
                    x=6 + c * 7,
                    y=6 + r * 7,
                    tags=["pipe", f"cell_{r}_{c}"],
                    blocking=BlockingMode.NOT_BLOCKED,
                ))

        level = Level(sprites=sprites, grid_size=(camera_res, camera_res))
        camera = Camera(width=camera_res, height=camera_res, background=bg_color)
        super().__init__(
            game_id="ng22_pipe_flow",
            levels=[level],
            camera=camera,
            available_actions=[6],
            seed=seed,
        )

    def get_sprite(self, name: str) -> Sprite:
        return self.current_level.get_sprites_by_name(name)[0]

    def _render_pipe_pixel(self, rot: int) -> np.ndarray:
        arr = np.zeros((5, 5), dtype=np.int8)
        arr[2, 2:5] = 11
        arr[2:5, 2] = 11
        return np.rot90(arr, -rot)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return

        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        if self._action.id == GameAction.ACTION6:
            disp_x = self._action.data.get("x", 0)
            disp_y = self._action.data.get("y", 0)
            grid_pos = self.camera.display_to_grid(disp_x, disp_y)
            if grid_pos is not None:
                gx, gy = grid_pos
                for r in range(3):
                    for c in range(3):
                        sp = self.get_sprite(f"pipe_{r}_{c}")
                        if sp.x <= gx < sp.x + sp.width and sp.y <= gy < sp.y + sp.height:
                            idx = r * 3 + c
                            self.rotations[idx] = (self.rotations[idx] + 1) % 4
                            sp.pixels = self._render_pipe_pixel(self.rotations[idx])
                            break

        if self._check_flow():
            self.win()

        self.complete_action()

    def _get_connections(self, rot: int) -> Tuple[bool, bool, bool, bool]:
        base = [False, True, True, False]
        return tuple(base[(i - rot) % 4] for i in range(4))

    def _check_flow(self) -> bool:
        c00 = self._get_connections(self.rotations[0])
        if not c00[3]:
            return False

        visited = set()
        queue = [(0, 0)]
        visited.add((0, 0))

        dirs = [(-1, 0, 0, 2), (0, 1, 1, 3), (1, 0, 2, 0), (0, -1, 3, 1)]

        while queue:
            cr, cc = queue.pop(0)
            c_idx = cr * 3 + cc
            conns = self._get_connections(self.rotations[c_idx])

            if cr == 2 and cc == 2 and conns[2]:
                return True

            for dr, dc, my_side, opp_side in dirs:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in visited:
                    if conns[my_side]:
                        n_idx = nr * 3 + nc
                        n_conns = self._get_connections(self.rotations[n_idx])
                        if n_conns[opp_side]:
                            visited.add((nr, nc))
                            queue.append((nr, nc))

        return False


class NG23_SlidingTile8(ARCBaseGame):
    """Classic 8-puzzle permutation matrix solver via adjacent tile clicks."""

    def __init__(self, camera_res: int = 32, max_steps: int = 50, seed: int = 42) -> None:
        self.camera_res = camera_res
        self.max_steps = max_steps
        self.steps_left = max_steps

        bg_color = 0
        self.board = [
            [1, 2, 3],
            [4, 5, 0],
            [7, 8, 6],
        ]
        self.empty_pos = (1, 2)

        sprites: List[Sprite] = []
        tile_colors = [0, 1, 2, 3, 4, 6, 7, 8, 11]

        for r in range(3):
            for c in range(3):
                val = self.board[r][c]
                col = tile_colors[val] if val != 0 else 0
                sprites.append(Sprite(
                    pixels=np.full((5, 5), col, dtype=np.int8).tolist(),
                    name=f"tile_{r}_{c}",
                    x=6 + c * 7,
                    y=6 + r * 7,
                    tags=["sliding_tile"],
                    blocking=BlockingMode.NOT_BLOCKED,
                ))

        level = Level(sprites=sprites, grid_size=(camera_res, camera_res))
        camera = Camera(width=camera_res, height=camera_res, background=bg_color)
        super().__init__(
            game_id="ng23_sliding_tile_8",
            levels=[level],
            camera=camera,
            available_actions=[6],
            seed=seed,
        )

    def get_sprite(self, name: str) -> Sprite:
        return self.current_level.get_sprites_by_name(name)[0]

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return

        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        if self._action.id == GameAction.ACTION6:
            disp_x = self._action.data.get("x", 0)
            disp_y = self._action.data.get("y", 0)
            grid_pos = self.camera.display_to_grid(disp_x, disp_y)
            if grid_pos is not None:
                gx, gy = grid_pos
                for r in range(3):
                    for c in range(3):
                        sp = self.get_sprite(f"tile_{r}_{c}")
                        if sp.x <= gx < sp.x + sp.width and sp.y <= gy < sp.y + sp.height:
                            er, ec = self.empty_pos
                            if abs(r - er) + abs(c - ec) == 1:
                                val = self.board[r][c]
                                self.board[er][ec] = val
                                self.board[r][c] = 0
                                self.empty_pos = (r, c)

                                tile_colors = [0, 1, 2, 3, 4, 6, 7, 8, 11]
                                empty_sp = self.get_sprite(f"tile_{er}_{ec}")
                                target_sp = self.get_sprite(f"tile_{r}_{c}")
                                empty_sp.pixels = np.full((5, 5), tile_colors[val], dtype=np.int8)
                                target_sp.pixels = np.full((5, 5), 0, dtype=np.int8)
                            break

        if (self.board[0] == [1, 2, 3] and
            self.board[1] == [4, 5, 6] and
            self.board[2] == [7, 8, 0]):
            self.win()

        self.complete_action()


class NG24_OpticalBeamSplit(ARCBaseGame):
    """45-degree angled laser mirror rotation lighting target crystals."""

    def __init__(self, camera_res: int = 32, max_steps: int = 30, seed: int = 42) -> None:
        self.camera_res = camera_res
        self.max_steps = max_steps
        self.steps_left = max_steps

        bg_color = 0
        laser_color = 2
        crystal_off = 5

        sprites: List[Sprite] = []

        sprites.append(Sprite(
            pixels=np.full((3, 3), laser_color, dtype=np.int8).tolist(),
            name="emitter",
            x=2,
            y=8,
            tags=["emitter"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        self.mirror_angles = [1, 0]
        mirror_coords = [(14, 8), (14, 20)]

        for i, (mx, my) in enumerate(mirror_coords):
            sprites.append(Sprite(
                pixels=self._render_mirror(self.mirror_angles[i]).tolist(),
                name=f"mirror_{i}",
                x=mx,
                y=my,
                tags=["mirror"],
                blocking=BlockingMode.NOT_BLOCKED,
            ))

        sprites.append(Sprite(
            pixels=np.full((4, 4), crystal_off, dtype=np.int8).tolist(),
            name="crystal",
            x=24,
            y=20,
            tags=["crystal"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        level = Level(sprites=sprites, grid_size=(camera_res, camera_res))
        camera = Camera(width=camera_res, height=camera_res, background=bg_color)
        super().__init__(
            game_id="ng24_optical_beam_split",
            levels=[level],
            camera=camera,
            available_actions=[6],
            seed=seed,
        )

    def get_sprite(self, name: str) -> Sprite:
        return self.current_level.get_sprites_by_name(name)[0]

    def _render_mirror(self, angle: int) -> np.ndarray:
        arr = np.zeros((5, 5), dtype=np.int8)
        if angle == 0:
            for i in range(5): arr[4 - i, i] = 11
        else:
            for i in range(5): arr[i, i] = 11
        return arr

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return

        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        crystal = self.get_sprite("crystal")

        if self._action.id == GameAction.ACTION6:
            disp_x = self._action.data.get("x", 0)
            disp_y = self._action.data.get("y", 0)
            grid_pos = self.camera.display_to_grid(disp_x, disp_y)
            if grid_pos is not None:
                gx, gy = grid_pos
                for i in range(2):
                    sp = self.get_sprite(f"mirror_{i}")
                    if sp.x <= gx < sp.x + sp.width and sp.y <= gy < sp.y + sp.height:
                        self.mirror_angles[i] = 1 - self.mirror_angles[i]
                        sp.pixels = self._render_mirror(self.mirror_angles[i])
                        break

        m1_correct = (self.mirror_angles[0] == 1)
        m2_correct = (self.mirror_angles[1] == 0)

        if m1_correct and m2_correct:
            crystal.pixels = np.full((4, 4), 14, dtype=np.int8)
            self.win()

        self.complete_action()


class NG25_ResourceCrafting(ARCBaseGame):
    """Finite-state inventory recipe synthesis (Wood + Ore -> Key -> Exit)."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 70, seed: int = 42) -> None:
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 0
        wall_color = 5
        wood_color = 15
        ore_color = 8
        workbench_color = 4
        door_color = 2
        avatar_color = 11
        goal_color = 14

        self.has_wood = False
        self.has_ore = False
        self.has_key = False

        sprites: List[Sprite] = []

        hw = np.full((step_size, grid_size), wall_color, dtype=np.int8)
        vw = np.full((grid_size, step_size), wall_color, dtype=np.int8)
        sprites.append(Sprite(pixels=hw.tolist(), name="w_top", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=hw.tolist(), name="w_bot", x=0, y=grid_size - step_size, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_l", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_r", x=grid_size - step_size, y=0, tags=["wall"]))

        av_pix = np.full((step_size, step_size), avatar_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=av_pix.tolist(),
            name="avatar",
            x=7 * step_size,
            y=7 * step_size,
            tags=["avatar"],
            blocking=BlockingMode.BOUNDING_BOX,
            layer=10,
        ))

        w_pix = np.full((step_size, step_size), wood_color, dtype=np.int8)
        sprites.append(Sprite(pixels=w_pix.tolist(), name="tree", x=3 * step_size, y=3 * step_size, tags=["tree"]))

        o_pix = np.full((step_size, step_size), ore_color, dtype=np.int8)
        sprites.append(Sprite(pixels=o_pix.tolist(), name="ore", x=12 * step_size, y=3 * step_size, tags=["ore"]))

        wb_pix = np.full((step_size, step_size), workbench_color, dtype=np.int8)
        sprites.append(Sprite(pixels=wb_pix.tolist(), name="workbench", x=7 * step_size, y=12 * step_size, tags=["workbench"]))

        dr_pix = np.full((step_size, step_size), door_color, dtype=np.int8)
        sprites.append(Sprite(pixels=dr_pix.tolist(), name="door", x=13 * step_size, y=12 * step_size, tags=["door", "wall"], blocking=BlockingMode.BOUNDING_BOX))

        g_pix = np.full((step_size, step_size), goal_color, dtype=np.int8)
        sprites.append(Sprite(pixels=g_pix.tolist(), name="goal", x=14 * step_size, y=12 * step_size, tags=["goal"], blocking=BlockingMode.NOT_BLOCKED))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng25_resource_crafting",
            levels=[level],
            camera=camera,
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

    def get_sprite(self, name: str) -> Sprite:
        return self.current_level.get_sprites_by_name(name)[0]

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return

        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        avatar = self.get_sprite("avatar")
        tree = self.get_sprite("tree")
        ore = self.get_sprite("ore")
        workbench = self.get_sprite("workbench")
        door = self.get_sprite("door")
        goal = self.get_sprite("goal")

        act = self._action.id
        if act == GameAction.ACTION5:
            ax = avatar.x
            ay = avatar.y

            if abs(ax - tree.x) + abs(ay - tree.y) <= self.step_size:
                self.has_wood = True
                tree.set_visible(False)

            if abs(ax - ore.x) + abs(ay - ore.y) <= self.step_size:
                self.has_ore = True
                ore.set_visible(False)

            if abs(ax - workbench.x) + abs(ay - workbench.y) <= self.step_size:
                if self.has_wood and self.has_ore:
                    self.has_key = True
                    workbench.pixels = np.full((self.step_size, self.step_size), 14, dtype=np.int8)

            if abs(ax - door.x) + abs(ay - door.y) <= self.step_size:
                if self.has_key:
                    door.set_visible(False)
                    door.set_blocking(BlockingMode.NOT_BLOCKED)

            self.complete_action()
            return

        dx, dy = 0, 0
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx = avatar.x + dx
        ny = avatar.y + dy

        for w in self.current_level.get_sprites_by_tag("wall"):
            if w.is_visible and (nx < w.x + w.width and nx + avatar.width > w.x and
                                 ny < w.y + w.height and ny + avatar.height > w.y):
                self.complete_action()
                return

        avatar.set_position(nx, ny)

        if avatar.x >= goal.x and avatar.y == goal.y:
            self.win()

        self.complete_action()
