#!/usr/bin/env python3
"""
Procedural ARC Game Synthesizer for ARC-AGI-3.
Generates synthetic, compliant ARC-AGI-3 games across three primary archetypes:
1. SyntheticMazeGame: Discrete lattice navigation with walls, variable step sizes, and exit targets.
2. SyntheticClickPuzzle: Coordinate-targeting puzzle with downsampled camera viewports and state toggles.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple, Dict, Any
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


class SyntheticMazeGame(ARCBaseGame):
    """Lattice navigation maze with avatar, walls, step sizes, and goal exit."""

    def __init__(
        self,
        grid_w: int = 32,
        grid_h: int = 32,
        step_size: int = 3,
        num_obstacles: int = 15,
        max_steps: int = 50,
        seed: int = 42,
    ) -> None:
        rng = random.Random(seed)
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_w = grid_w
        self.grid_h = grid_h

        bg_color = 0
        wall_color = 4
        avatar_color = 11
        exit_color = 14

        # Avatar sprite
        avatar_pixels = np.full((step_size, step_size), avatar_color, dtype=np.int8)
        self.avatar_sprite = Sprite(
            pixels=avatar_pixels.tolist(),
            name="avatar",
            x=step_size,
            y=step_size,
            tags=["avatar"],
            blocking=BlockingMode.BOUNDING_BOX,
        )

        # Exit sprite
        exit_pixels = np.full((step_size, step_size), exit_color, dtype=np.int8)
        exit_x = (grid_w // step_size - 2) * step_size
        exit_y = (grid_h // step_size - 2) * step_size
        self.exit_sprite = Sprite(
            pixels=exit_pixels.tolist(),
            name="exit",
            x=exit_x,
            y=exit_y,
            tags=["exit", "goal"],
            blocking=BlockingMode.NOT_BLOCKED,
        )

        sprites: List[Sprite] = [self.avatar_sprite, self.exit_sprite]
        occupied = {(1, 1), (grid_w // step_size - 2, grid_h // step_size - 2)}

        # Outer border walls
        h_wall = np.full((step_size, grid_w), wall_color, dtype=np.int8)
        sprites.append(Sprite(pixels=h_wall.tolist(), name="wall_top", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=h_wall.tolist(), name="wall_bottom", x=0, y=grid_h - step_size, tags=["wall"]))
        v_wall = np.full((grid_h, step_size), wall_color, dtype=np.int8)
        sprites.append(Sprite(pixels=v_wall.tolist(), name="wall_left", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=v_wall.tolist(), name="wall_right", x=grid_w - step_size, y=0, tags=["wall"]))

        # Internal obstacles
        for i in range(num_obstacles):
            gx = rng.randint(2, max(2, grid_w // step_size - 3))
            gy = rng.randint(2, max(2, grid_h // step_size - 3))
            if (gx, gy) not in occupied:
                occupied.add((gx, gy))
                obs_pixels = np.full((step_size, step_size), wall_color, dtype=np.int8)
                sprites.append(Sprite(
                    pixels=obs_pixels.tolist(),
                    name=f"obs_{i}",
                    x=gx * step_size,
                    y=gy * step_size,
                    tags=["wall", "obstacle"],
                    blocking=BlockingMode.BOUNDING_BOX,
                ))

        level = Level(sprites=sprites, grid_size=(grid_w, grid_h))
        camera = Camera(width=grid_w, height=grid_h, background=bg_color)
        super().__init__(
            game_id=f"synth_maze_s{step_size}",
            levels=[level],
            camera=camera,
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return

        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        act = self._action.id
        dx, dy = 0, 0
        if act == GameAction.ACTION1:  # Up
            dy = -self.step_size
        elif act == GameAction.ACTION2:  # Down
            dy = self.step_size
        elif act == GameAction.ACTION3:  # Left
            dx = -self.step_size
        elif act == GameAction.ACTION4:  # Right
            dx = self.step_size

        new_x = self.avatar_sprite.x + dx
        new_y = self.avatar_sprite.y + dy

        # Collision check with walls
        can_move = True
        for sp in self.current_level.get_sprites_by_tag("wall"):
            if (
                new_x < sp.x + sp.width
                and new_x + self.avatar_sprite.width > sp.x
                and new_y < sp.y + sp.height
                and new_y + self.avatar_sprite.height > sp.y
            ):
                can_move = False
                break

        if can_move:
            self.avatar_sprite.set_position(new_x, new_y)

        # Check win
        if (
            self.avatar_sprite.x == self.exit_sprite.x
            and self.avatar_sprite.y == self.exit_sprite.y
        ):
            self.next_level()

        self.complete_action()


class SyntheticClickPuzzle(ARCBaseGame):
    """Point-and-click button puzzle with camera viewport downsampling."""

    def __init__(
        self,
        camera_res: int = 32,
        num_buttons: int = 3,
        max_steps: int = 30,
        seed: int = 42,
    ) -> None:
        rng = random.Random(seed)
        self.camera_res = camera_res
        self.max_steps = max_steps
        self.steps_left = max_steps
        bg_color = 3
        active_color = 9
        solved_color = 14

        sprites: List[Sprite] = []
        self.button_sprites: List[Sprite] = []
        self.button_states: List[bool] = [False] * num_buttons

        for i in range(num_buttons):
            bx = rng.randint(4, camera_res - 8)
            by = rng.randint(4, camera_res - 8)
            btn_pixels = np.full((3, 3), active_color, dtype=np.int8)
            sp = Sprite(
                pixels=btn_pixels.tolist(),
                name=f"btn_{i}",
                x=bx,
                y=by,
                tags=["sys_click", f"button_{i}"],
            )
            sprites.append(sp)
            self.button_sprites.append(sp)

        level = Level(sprites=sprites, grid_size=(camera_res, camera_res))
        camera = Camera(width=camera_res, height=camera_res, background=bg_color)
        super().__init__(
            game_id=f"synth_click_c{camera_res}",
            levels=[level],
            camera=camera,
            available_actions=[6],
            seed=seed,
        )

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
                for idx, btn in enumerate(self.button_sprites):
                    if (
                        btn.x <= gx < btn.x + btn.width
                        and btn.y <= gy < btn.y + btn.height
                    ):
                        if not self.button_states[idx]:
                            self.button_states[idx] = True
                            btn.pixels = np.full((btn.height, btn.width), 14, dtype=np.int8)

        if all(self.button_states):
            self.next_level()

        self.complete_action()


def generate_game(archetype: str = "maze", seed: int = 42) -> ARCBaseGame:
    """Factory creating configured synthetic ARC environments."""
    if archetype == "maze":
        step = random.choice([1, 2, 3, 5])
        return SyntheticMazeGame(grid_w=32, grid_h=32, step_size=step, seed=seed)
    elif archetype == "click":
        cam_res = random.choice([16, 32, 64])
        return SyntheticClickPuzzle(camera_res=cam_res, num_buttons=3, seed=seed)
    else:
        return SyntheticMazeGame(grid_w=32, grid_h=32, step_size=3, seed=seed)


if __name__ == "__main__":
    print("Testing SyntheticMazeGame...")
    maze = SyntheticMazeGame(grid_w=32, grid_h=32, step_size=3, seed=123)
    frame = maze.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    print("Maze initial frame shape:", np.array(frame.frame[0]).shape)

    print("\nTesting SyntheticClickPuzzle...")
    click_game = SyntheticClickPuzzle(camera_res=32, seed=456)
    cframe = click_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    print("Click game initial frame shape:", np.array(cframe.frame[0]).shape)
    print("All synthetic environments initialized successfully!")
