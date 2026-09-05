#!/usr/bin/env python3
"""
Paradigm 3: Pattern Matching & Logic Gates
Contains 5 novel ARC-AGI-3 game environments:
- ng11_logic_circuit: Interactive binary switches feeding boolean logic gates (AND/OR/XOR).
- ng12_color_mixer: Primary color fountains blending avatar hue to pass chromatic filter gates.
- ng13_symmetry_painter: Point-and-click reflection canvas replicating target patterns.
- ng14_sudoku_light: 4x4 Latin Square chromatic logic constraint puzzle via ACTION6.
- ng15_sequence_melody: Temporal memory sequence matching floor beacons to HUD melody order.
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


class NG11_LogicCircuit(ARCBaseGame):
    """Interactive binary switches feeding boolean logic gates (AND/OR/XOR)."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 0
        wall_color = 5
        avatar_color = 11
        goal_color = 14

        sprites: List[Sprite] = []

        hw = np.full((step_size, grid_size), wall_color, dtype=np.int8)
        vw = np.full((grid_size, step_size), wall_color, dtype=np.int8)
        sprites.append(Sprite(pixels=hw.tolist(), name="w_top", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=hw.tolist(), name="w_bot", x=0, y=grid_size - step_size, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_l", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_r", x=grid_size - step_size, y=0, tags=["wall"]))

        mid_y = (grid_size // (2 * step_size)) * step_size
        door_x = 7 * step_size
        if door_x > 0:
            hw_l = np.full((step_size, door_x), wall_color, dtype=np.int8)
            sprites.append(Sprite(pixels=hw_l.tolist(), name="w_mid_l", x=0, y=mid_y, tags=["wall"]))
        right_w = grid_size - (door_x + step_size)
        if right_w > 0:
            hw_r = np.full((step_size, right_w), wall_color, dtype=np.int8)
            sprites.append(Sprite(pixels=hw_r.tolist(), name="w_mid_r", x=door_x + step_size, y=mid_y, tags=["wall"]))

        self.switch_a_state = False
        self.switch_b_state = False

        sw_pix = np.full((step_size, step_size), 2, dtype=np.int8)
        sprites.append(Sprite(
            pixels=sw_pix.tolist(),
            name="sw_a",
            x=4 * step_size,
            y=4 * step_size,
            tags=["switch", "sw_a"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))
        sprites.append(Sprite(
            pixels=sw_pix.tolist(),
            name="sw_b",
            x=10 * step_size,
            y=4 * step_size,
            tags=["switch", "sw_b"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        gate_pix = np.full((step_size, step_size), 8, dtype=np.int8)
        sprites.append(Sprite(
            pixels=gate_pix.tolist(),
            name="gate",
            x=7 * step_size,
            y=4 * step_size,
            tags=["gate"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        door_pix = np.full((step_size, step_size), 2, dtype=np.int8)
        sprites.append(Sprite(
            pixels=door_pix.tolist(),
            name="door",
            x=7 * step_size,
            y=mid_y,
            tags=["door", "wall"],
            blocking=BlockingMode.BOUNDING_BOX,
        ))

        av_pix = np.full((step_size, step_size), avatar_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=av_pix.tolist(),
            name="avatar",
            x=7 * step_size,
            y=2 * step_size,
            tags=["avatar"],
            blocking=BlockingMode.BOUNDING_BOX,
            layer=10,
        ))

        g_pix = np.full((step_size, step_size), goal_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=g_pix.tolist(),
            name="goal",
            x=7 * step_size,
            y=12 * step_size,
            tags=["goal"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng11_logic_circuit",
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
        sw_a = self.get_sprite("sw_a")
        sw_b = self.get_sprite("sw_b")
        door = self.get_sprite("door")
        goal = self.get_sprite("goal")

        act = self._action.id
        if act == GameAction.ACTION5:
            if abs(avatar.x - sw_a.x) <= self.step_size and abs(avatar.y - sw_a.y) <= self.step_size:
                self.switch_a_state = not self.switch_a_state
                col = 14 if self.switch_a_state else 2
                sw_a.pixels = np.full((self.step_size, self.step_size), col, dtype=np.int8)
            elif abs(avatar.x - sw_b.x) <= self.step_size and abs(avatar.y - sw_b.y) <= self.step_size:
                self.switch_b_state = not self.switch_b_state
                col = 14 if self.switch_b_state else 2
                sw_b.pixels = np.full((self.step_size, self.step_size), col, dtype=np.int8)

            door_open = (self.switch_a_state != self.switch_b_state)
            if door_open:
                door.set_visible(False)
                door.set_blocking(BlockingMode.NOT_BLOCKED)
            else:
                door.set_visible(True)
                door.set_blocking(BlockingMode.BOUNDING_BOX)

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

        if avatar.x == goal.x and avatar.y == goal.y:
            self.win()

        self.complete_action()


class NG12_ColorMixer(ARCBaseGame):
    """Primary color fountains blending avatar hue to pass chromatic filter gates."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 0
        wall_color = 5
        self.color_neutral = 10
        self.red_color = 2
        self.blue_color = 1
        self.purple_color = 12
        goal_color = 14

        self.has_red = False
        self.has_blue = False

        sprites: List[Sprite] = []

        hw = np.full((step_size, grid_size), wall_color, dtype=np.int8)
        vw = np.full((grid_size, step_size), wall_color, dtype=np.int8)
        sprites.append(Sprite(pixels=hw.tolist(), name="w_top", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=hw.tolist(), name="w_bot", x=0, y=grid_size - step_size, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_l", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_r", x=grid_size - step_size, y=0, tags=["wall"]))

        r_pix = np.full((step_size, step_size), self.red_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=r_pix.tolist(),
            name="f_red",
            x=3 * step_size,
            y=3 * step_size,
            tags=["fountain"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        b_pix = np.full((step_size, step_size), self.blue_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=b_pix.tolist(),
            name="f_blue",
            x=12 * step_size,
            y=3 * step_size,
            tags=["fountain"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        for wx in range(grid_size // step_size):
            if wx != 8:
                sprites.append(Sprite(
                    pixels=r_pix.tolist(),
                    name=f"mid_w_{wx}",
                    x=wx * step_size,
                    y=8 * step_size,
                    tags=["wall"],
                    blocking=BlockingMode.BOUNDING_BOX,
                ))

        p_pix = np.full((step_size, step_size), self.purple_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=p_pix.tolist(),
            name="gate_purple",
            x=8 * step_size,
            y=8 * step_size,
            tags=["filter_gate"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        av_pix = np.full((step_size, step_size), self.color_neutral, dtype=np.int8)
        sprites.append(Sprite(
            pixels=av_pix.tolist(),
            name="avatar",
            x=8 * step_size,
            y=2 * step_size,
            tags=["avatar"],
            blocking=BlockingMode.BOUNDING_BOX,
            layer=10,
        ))

        g_pix = np.full((step_size, step_size), goal_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=g_pix.tolist(),
            name="goal",
            x=8 * step_size,
            y=12 * step_size,
            tags=["goal"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng12_color_mixer",
            levels=[level],
            camera=camera,
            available_actions=[1, 2, 3, 4],
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
        f_red = self.get_sprite("f_red")
        f_blue = self.get_sprite("f_blue")
        gate_purple = self.get_sprite("gate_purple")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx = avatar.x + dx
        ny = avatar.y + dy

        for w in self.current_level.get_sprites_by_tag("wall"):
            if (nx < w.x + w.width and nx + avatar.width > w.x and
                ny < w.y + w.height and ny + avatar.height > w.y):
                self.complete_action()
                return

        if nx == gate_purple.x and ny == gate_purple.y:
            if not (self.has_red and self.has_blue):
                self.complete_action()
                return

        avatar.set_position(nx, ny)

        if avatar.x == f_red.x and avatar.y == f_red.y:
            self.has_red = True
        elif avatar.x == f_blue.x and avatar.y == f_blue.y:
            self.has_blue = True

        if self.has_red and self.has_blue:
            col = self.purple_color
        elif self.has_red:
            col = self.red_color
        elif self.has_blue:
            col = self.blue_color
        else:
            col = self.color_neutral
        avatar.pixels = np.full((self.step_size, self.step_size), col, dtype=np.int8)

        if avatar.x == goal.x and avatar.y == goal.y:
            self.win()

        self.complete_action()


class NG13_SymmetryPainter(ARCBaseGame):
    """Point-and-click reflection canvas replicating target patterns."""

    def __init__(self, camera_res: int = 32, max_steps: int = 30, seed: int = 42) -> None:
        rng = random.Random(seed)
        self.camera_res = camera_res
        self.max_steps = max_steps
        self.steps_left = max_steps

        bg_color = 0
        target_color = 14

        sprites: List[Sprite] = []

        self.target_matrix = np.zeros((4, 4), dtype=bool)
        for r in range(4):
            for c in range(2):
                if rng.random() < 0.5:
                    self.target_matrix[r, c] = True
                    self.target_matrix[r, 3 - c] = True

        if np.sum(self.target_matrix) == 0:
            self.target_matrix[1, 1] = True
            self.target_matrix[1, 2] = True

        t_pix = np.full((2, 2), target_color, dtype=np.int8)
        for r in range(4):
            for c in range(4):
                if self.target_matrix[r, c]:
                    sprites.append(Sprite(
                        pixels=t_pix.tolist(),
                        name=f"tgt_{r}_{c}",
                        x=8 + c * 3,
                        y=4 + r * 3,
                        tags=["target"],
                        blocking=BlockingMode.NOT_BLOCKED,
                    ))

        self.canvas_state = np.zeros((4, 4), dtype=bool)
        c_off_pix = np.full((2, 2), 5, dtype=np.int8)

        for r in range(4):
            for c in range(4):
                sprites.append(Sprite(
                    pixels=c_off_pix.tolist(),
                    name=f"cnv_{r}_{c}",
                    x=8 + c * 3,
                    y=18 + r * 3,
                    tags=["canvas", f"cell_{r}_{c}"],
                    blocking=BlockingMode.NOT_BLOCKED,
                ))

        level = Level(sprites=sprites, grid_size=(camera_res, camera_res))
        camera = Camera(width=camera_res, height=camera_res, background=bg_color)
        super().__init__(
            game_id="ng13_symmetry_painter",
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
                for r in range(4):
                    for c in range(4):
                        sp = self.get_sprite(f"cnv_{r}_{c}")
                        if sp.x <= gx < sp.x + sp.width and sp.y <= gy < sp.y + sp.height:
                            self.canvas_state[r, c] = not self.canvas_state[r, c]
                            self.canvas_state[r, 3 - c] = self.canvas_state[r, c]

                            for col_idx in [c, 3 - c]:
                                cell_sp = self.get_sprite(f"cnv_{r}_{col_idx}")
                                val = self.canvas_state[r, col_idx]
                                pix_col = 11 if val else 5
                                cell_sp.pixels = np.full((2, 2), pix_col, dtype=np.int8)

        if np.array_equal(self.canvas_state, self.target_matrix):
            self.win()

        self.complete_action()


class NG14_SudokuLight(ARCBaseGame):
    """4x4 Latin Square chromatic logic constraint puzzle via ACTION6."""

    def __init__(self, camera_res: int = 32, max_steps: int = 40, seed: int = 42) -> None:
        self.camera_res = camera_res
        self.max_steps = max_steps
        self.steps_left = max_steps

        bg_color = 0
        self.solution = np.array([
            [1, 2, 3, 4],
            [3, 4, 1, 2],
            [4, 1, 2, 3],
            [2, 3, 4, 1],
        ], dtype=int)

        self.fixed_mask = np.zeros((4, 4), dtype=bool)
        fixed_coords = [(0, 0), (0, 3), (1, 1), (2, 2), (3, 0), (3, 3)]
        for r, c in fixed_coords:
            self.fixed_mask[r, c] = True

        self.board = np.zeros((4, 4), dtype=int)
        for r in range(4):
            for c in range(4):
                if self.fixed_mask[r, c]:
                    self.board[r, c] = self.solution[r, c]

        sprites: List[Sprite] = []

        for r in range(4):
            for c in range(4):
                val = self.board[r, c]
                col = val if val > 0 else 5
                sprites.append(Sprite(
                    pixels=np.full((3, 3), col, dtype=np.int8).tolist(),
                    name=f"sudoku_{r}_{c}",
                    x=6 + c * 5,
                    y=6 + r * 5,
                    tags=["sudoku_cell"],
                    blocking=BlockingMode.NOT_BLOCKED,
                ))

        level = Level(sprites=sprites, grid_size=(camera_res, camera_res))
        camera = Camera(width=camera_res, height=camera_res, background=bg_color)
        super().__init__(
            game_id="ng14_sudoku_light",
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
                for r in range(4):
                    for c in range(4):
                        if not self.fixed_mask[r, c]:
                            sp = self.get_sprite(f"sudoku_{r}_{c}")
                            if sp.x <= gx < sp.x + sp.width and sp.y <= gy < sp.y + sp.height:
                                curr = self.board[r, c]
                                nxt = (curr % 4) + 1
                                self.board[r, c] = nxt
                                sp.pixels = np.full((3, 3), nxt, dtype=np.int8)

        is_solved = True
        for r in range(4):
            if set(self.board[r, :]) != {1, 2, 3, 4}:
                is_solved = False
                break
        for c in range(4):
            if set(self.board[:, c]) != {1, 2, 3, 4}:
                is_solved = False
                break

        if is_solved:
            self.win()

        self.complete_action()


class NG15_SequenceMelody(ARCBaseGame):
    """Temporal memory sequence matching floor beacons to HUD melody order."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        rng = random.Random(seed)
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 0
        wall_color = 5
        avatar_color = 11

        available_colors = [1, 2, 3, 4]
        self.melody_sequence = rng.sample(available_colors, 4)
        self.current_idx = 0

        sprites: List[Sprite] = []

        hw = np.full((step_size, grid_size), wall_color, dtype=np.int8)
        vw = np.full((grid_size, step_size), wall_color, dtype=np.int8)
        sprites.append(Sprite(pixels=hw.tolist(), name="w_top", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=hw.tolist(), name="w_bot", x=0, y=grid_size - step_size, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_l", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_r", x=grid_size - step_size, y=0, tags=["wall"]))

        for idx, col in enumerate(self.melody_sequence):
            hud_pix = np.full((step_size, step_size), col, dtype=np.int8)
            sprites.append(Sprite(
                pixels=hud_pix.tolist(),
                name=f"hud_{idx}",
                x=(4 + idx * 2) * step_size,
                y=step_size * 2,
                tags=["hud"],
                blocking=BlockingMode.NOT_BLOCKED,
            ))

        self.pad_positions = {
            1: (3, 6),
            2: (11, 6),
            3: (3, 11),
            4: (11, 11),
        }
        for col, (px, py) in self.pad_positions.items():
            p_pix = np.full((step_size, step_size), col, dtype=np.int8)
            sprites.append(Sprite(
                pixels=p_pix.tolist(),
                name=f"pad_{col}",
                x=px * step_size,
                y=py * step_size,
                tags=["pad"],
                blocking=BlockingMode.NOT_BLOCKED,
            ))

        av_pix = np.full((step_size, step_size), avatar_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=av_pix.tolist(),
            name="avatar",
            x=7 * step_size,
            y=8 * step_size,
            tags=["avatar"],
            blocking=BlockingMode.BOUNDING_BOX,
            layer=10,
        ))

        dr_pix = np.full((step_size, step_size), 2, dtype=np.int8)
        sprites.append(Sprite(
            pixels=dr_pix.tolist(),
            name="door",
            x=7 * step_size,
            y=grid_size - step_size * 2,
            tags=["door", "wall"],
            blocking=BlockingMode.BOUNDING_BOX,
        ))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng15_sequence_melody",
            levels=[level],
            camera=camera,
            available_actions=[1, 2, 3, 4],
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
        door = self.get_sprite("door")

        dx, dy = 0, 0
        act = self._action.id
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

        if self.current_idx < len(self.melody_sequence):
            for col in self.pad_positions:
                sp = self.get_sprite(f"pad_{col}")
                if avatar.x == sp.x and avatar.y == sp.y:
                    expected = self.melody_sequence[self.current_idx]
                    if col == expected:
                        self.current_idx += 1
                        sp.pixels = np.full((self.step_size, self.step_size), 14, dtype=np.int8)
                        if self.current_idx >= len(self.melody_sequence):
                            door.set_visible(False)
                            door.set_blocking(BlockingMode.NOT_BLOCKED)
                    else:
                        self.current_idx = 0
                        for c_reset in self.pad_positions:
                            r_sp = self.get_sprite(f"pad_{c_reset}")
                            r_sp.pixels = np.full((self.step_size, self.step_size), c_reset, dtype=np.int8)

        if self.current_idx >= len(self.melody_sequence):
            if avatar.y >= self.grid_size - self.step_size * 2:
                self.win()

        self.complete_action()
