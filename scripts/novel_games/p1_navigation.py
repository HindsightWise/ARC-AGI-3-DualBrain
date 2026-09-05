#!/usr/bin/env python3
"""
Paradigm 1: Spatial & Topological Navigation
Contains 5 novel ARC-AGI-3 game environments:
- ng01_wormhole_tunnel: Paired non-Euclidean teleportation portals across sealed chambers.
- ng02_moving_hazard: Dynamic timing obstacle corridors with oscillating hazards.
- ng03_light_shadow: Vision-dependent pathfinding on illuminated tiles (void collapse).
- ng04_gravity_wells: Directional booster pads catapulting across wide chasms.
- ng05_collapsing_path: Hamiltonian path where stepped tiles turn into voids upon leaving.
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


class NG01_WormholeTunnel(ARCBaseGame):
    """Teleportation portals linking sealed chambers."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 0
        wall_color = 5
        avatar_color = 11
        portal_color = 8
        goal_color = 14

        mid = (grid_size // (2 * step_size)) * step_size
        sprites: List[Sprite] = []

        # Avatar in Chamber A (left side)
        av_pixels = np.full((step_size, step_size), avatar_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=av_pixels.tolist(),
            name="avatar",
            x=step_size * 2,
            y=step_size * 2,
            tags=["avatar"],
            blocking=BlockingMode.BOUNDING_BOX,
            layer=10,
        ))

        # Perimeter walls
        hw = np.full((step_size, grid_size), wall_color, dtype=np.int8)
        vw = np.full((grid_size, step_size), wall_color, dtype=np.int8)
        sprites.append(Sprite(pixels=hw.tolist(), name="w_top", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=hw.tolist(), name="w_bottom", x=0, y=grid_size - step_size, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_left", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_right", x=grid_size - step_size, y=0, tags=["wall"]))

        # Central divider wall
        sprites.append(Sprite(pixels=vw.tolist(), name="w_mid", x=mid, y=0, tags=["wall"]))

        # Wormhole A (in Chamber A)
        p_pix = np.full((step_size, step_size), portal_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=p_pix.tolist(),
            name="portal_a",
            x=step_size * 3,
            y=grid_size - step_size * 4,
            tags=["portal", "portal_a"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        # Wormhole B (in Chamber B, right side)
        sprites.append(Sprite(
            pixels=p_pix.tolist(),
            name="portal_b",
            x=mid + step_size * 2,
            y=step_size * 3,
            tags=["portal", "portal_b"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        # Goal in Chamber B
        goal_x = grid_size - step_size * 3
        goal_y = grid_size - step_size * 3
        g_pix = np.full((step_size, step_size), goal_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=g_pix.tolist(),
            name="goal",
            x=goal_x,
            y=goal_y,
            tags=["goal", "exit"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng01_wormhole_tunnel",
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
        portal_a = self.get_sprite("portal_a")
        portal_b = self.get_sprite("portal_b")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx = avatar.x + dx
        ny = avatar.y + dy

        # Wall collisions
        can_move = True
        for wall in self.current_level.get_sprites_by_tag("wall"):
            if (nx < wall.x + wall.width and nx + avatar.width > wall.x and
                ny < wall.y + wall.height and ny + avatar.height > wall.y):
                can_move = False
                break

        if can_move:
            avatar.set_position(nx, ny)

        # Wormhole teleportation check
        if avatar.x == portal_a.x and avatar.y == portal_a.y:
            avatar.set_position(portal_b.x, portal_b.y)
        elif avatar.x == portal_b.x and avatar.y == portal_b.y:
            avatar.set_position(portal_a.x, portal_a.y)

        # Goal check
        if avatar.x == goal.x and avatar.y == goal.y:
            self.win()

        self.complete_action()


class NG02_MovingHazard(ARCBaseGame):
    """Dynamic timing corridors with horizontally oscillating hazard blocks."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 0
        wall_color = 5
        avatar_color = 11
        hazard_color = 2
        goal_color = 14

        sprites: List[Sprite] = []

        av_pixels = np.full((step_size, step_size), avatar_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=av_pixels.tolist(),
            name="avatar",
            x=grid_size // 2,
            y=grid_size - step_size * 3,
            tags=["avatar"],
            blocking=BlockingMode.BOUNDING_BOX,
            layer=10,
        ))

        g_pixels = np.full((step_size, step_size), goal_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=g_pixels.tolist(),
            name="goal",
            x=grid_size // 2,
            y=step_size * 2,
            tags=["goal"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        # Perimeter
        hw = np.full((step_size, grid_size), wall_color, dtype=np.int8)
        vw = np.full((grid_size, step_size), wall_color, dtype=np.int8)
        sprites.append(Sprite(pixels=hw.tolist(), name="w_top", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=hw.tolist(), name="w_bottom", x=0, y=grid_size - step_size, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_left", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_right", x=grid_size - step_size, y=0, tags=["wall"]))

        # 3 Hazards oscillating horizontally
        self.hazard_dx = [step_size, -step_size, step_size]
        hazard_rows = [grid_size // 4, grid_size // 2, (3 * grid_size) // 4]
        for idx, row in enumerate(hazard_rows):
            hz_pix = np.full((step_size, step_size * 2), hazard_color, dtype=np.int8)
            sprites.append(Sprite(
                pixels=hz_pix.tolist(),
                name=f"hazard_{idx}",
                x=step_size * 2 if idx % 2 == 0 else grid_size - step_size * 4,
                y=row,
                tags=["hazard"],
                blocking=BlockingMode.NOT_BLOCKED,
            ))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng02_moving_hazard",
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
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx = avatar.x + dx
        ny = avatar.y + dy

        # Wall collision
        can_move = True
        for wall in self.current_level.get_sprites_by_tag("wall"):
            if (nx < wall.x + wall.width and nx + avatar.width > wall.x and
                ny < wall.y + wall.height and ny + avatar.height > wall.y):
                can_move = False
                break

        if can_move:
            avatar.set_position(nx, ny)

        # Update hazards
        for idx in range(3):
            hz_sp = self.get_sprite(f"hazard_{idx}")
            nx_hz = hz_sp.x + self.hazard_dx[idx]
            if nx_hz <= self.step_size or nx_hz >= self.grid_size - self.step_size * 3:
                self.hazard_dx[idx] = -self.hazard_dx[idx]
                nx_hz = hz_sp.x + self.hazard_dx[idx]
            hz_sp.set_position(nx_hz, hz_sp.y)

            # Hazard hit check
            if (avatar.x < hz_sp.x + hz_sp.width and avatar.x + avatar.width > hz_sp.x and
                avatar.y < hz_sp.y + hz_sp.height and avatar.y + avatar.height > hz_sp.y):
                self.lose()
                self.complete_action()
                return

        # Win check
        if (avatar.x < goal.x + goal.width and avatar.x + avatar.width > goal.x and
            avatar.y < goal.y + goal.height and avatar.y + avatar.height > goal.y):
            self.win()

        self.complete_action()


class NG03_LightShadow(ARCBaseGame):
    """Vision-dependent pathfinding on illuminated tiles (stepping into dark void loses)."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        rng = random.Random(seed)
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 0  # Dark void
        path_color = 5  # Illuminated light path
        avatar_color = 11
        goal_color = 14

        self.safe_tiles: Set[Tuple[int, int]] = set()
        cx, cy = 2, 2
        gx, gy = grid_size // step_size - 2, grid_size // step_size - 2
        self.safe_tiles.add((cx, cy))

        while (cx, cy) != (gx, gy):
            candidates = []
            if cx < gx: candidates.append((cx + 1, cy))
            if cy < gy: candidates.append((cx, cy + 1))
            if cx > 1 and rng.random() < 0.2: candidates.append((cx - 1, cy))
            if cy > 1 and rng.random() < 0.2: candidates.append((cx, cy - 1))
            cx, cy = rng.choice(candidates)
            self.safe_tiles.add((cx, cy))

        sprites: List[Sprite] = []

        # Illuminated path tiles
        p_pix = np.full((step_size, step_size), path_color, dtype=np.int8)
        for (tx, ty) in self.safe_tiles:
            sprites.append(Sprite(
                pixels=p_pix.tolist(),
                name=f"tile_{tx}_{ty}",
                x=tx * step_size,
                y=ty * step_size,
                tags=["path"],
                blocking=BlockingMode.NOT_BLOCKED,
            ))

        # Avatar
        av_pixels = np.full((step_size, step_size), avatar_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=av_pixels.tolist(),
            name="avatar",
            x=2 * step_size,
            y=2 * step_size,
            tags=["avatar"],
            blocking=BlockingMode.NOT_BLOCKED,
            layer=10,
        ))

        # Goal
        g_pixels = np.full((step_size, step_size), goal_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=g_pixels.tolist(),
            name="goal",
            x=gx * step_size,
            y=gy * step_size,
            tags=["goal"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng03_light_shadow",
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
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx = avatar.x + dx
        ny = avatar.y + dy

        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
            avatar.set_position(nx, ny)
        else:
            self.lose()
            self.complete_action()
            return

        tx = nx // self.step_size
        ty = ny // self.step_size
        if (tx, ty) not in self.safe_tiles:
            self.lose()
            self.complete_action()
            return

        if avatar.x == goal.x and avatar.y == goal.y:
            self.win()

        self.complete_action()


class NG04_GravityWells(ARCBaseGame):
    """Directional boost pads catapulting avatar across chasms."""

    def __init__(self, grid_size: int = 64, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 0
        island_color = 5
        avatar_color = 11
        boost_color = 8
        goal_color = 14

        sprites: List[Sprite] = []
        self.walkable_tiles: Set[Tuple[int, int]] = set()

        islands = [
            (2, 6, 2, 6),
            (12, 16, 2, 6),
            (12, 16, 12, 16),
            (22, 26, 12, 16),
        ]
        isl_pix = np.full((step_size, step_size), island_color, dtype=np.int8)
        for (x1, x2, y1, y2) in islands:
            for x in range(x1, x2):
                for y in range(y1, y2):
                    self.walkable_tiles.add((x, y))
                    sprites.append(Sprite(
                        pixels=isl_pix.tolist(),
                        name=f"isl_{x}_{y}",
                        x=x * step_size,
                        y=y * step_size,
                        tags=["floor"],
                        blocking=BlockingMode.NOT_BLOCKED,
                    ))

        self.boost_pads: Dict[Tuple[int, int], Tuple[int, int]] = {
            (5, 4): (7, 0),
            (14, 5): (0, 7),
            (15, 14): (7, 0),
        }
        b_pix = np.full((step_size, step_size), boost_color, dtype=np.int8)
        for (bx, by) in self.boost_pads:
            sprites.append(Sprite(
                pixels=b_pix.tolist(),
                name=f"boost_{bx}_{by}",
                x=bx * step_size,
                y=by * step_size,
                tags=["boost"],
                blocking=BlockingMode.NOT_BLOCKED,
            ))

        av_pixels = np.full((step_size, step_size), avatar_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=av_pixels.tolist(),
            name="avatar",
            x=3 * step_size,
            y=3 * step_size,
            tags=["avatar"],
            blocking=BlockingMode.NOT_BLOCKED,
            layer=10,
        ))

        g_pixels = np.full((step_size, step_size), goal_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=g_pixels.tolist(),
            name="goal",
            x=24 * step_size,
            y=14 * step_size,
            tags=["goal"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng04_gravity_wells",
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
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx = avatar.x + dx
        ny = avatar.y + dy
        tx = nx // self.step_size
        ty = ny // self.step_size

        if (tx, ty) not in self.walkable_tiles and (tx, ty) not in self.boost_pads:
            self.lose()
            self.complete_action()
            return

        avatar.set_position(nx, ny)

        if (tx, ty) in self.boost_pads:
            bdx, bdy = self.boost_pads[(tx, ty)]
            boost_tx = tx + bdx
            boost_ty = ty + bdy
            avatar.set_position(boost_tx * self.step_size, boost_ty * self.step_size)

        if avatar.x == goal.x and avatar.y == goal.y:
            self.win()

        self.complete_action()


class NG05_CollapsingPath(ARCBaseGame):
    """Hamiltonian path where stepped tiles turn into voids upon leaving."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 0
        floor_color = 5
        key_color = 12
        goal_color = 14
        avatar_color = 11

        sprites: List[Sprite] = []
        self.arena_origin = 4
        self.active_floor_coords: Set[Tuple[int, int]] = set()

        fl_pix = np.full((step_size, step_size), floor_color, dtype=np.int8)
        for ax in range(6):
            for ay in range(6):
                sprites.append(Sprite(
                    pixels=fl_pix.tolist(),
                    name=f"fl_{ax}_{ay}",
                    x=(self.arena_origin + ax * 2) * step_size,
                    y=(self.arena_origin + ay * 2) * step_size,
                    tags=["floor"],
                    blocking=BlockingMode.NOT_BLOCKED,
                ))
                self.active_floor_coords.add((ax, ay))

        k_pix = np.full((step_size, step_size), key_color, dtype=np.int8)
        self.key_coords = {(1, 1), (4, 1), (1, 4)}
        for kx, ky in self.key_coords:
            sprites.append(Sprite(
                pixels=k_pix.tolist(),
                name=f"key_{kx}_{ky}",
                x=(self.arena_origin + kx * 2) * step_size,
                y=(self.arena_origin + ky * 2) * step_size,
                tags=["key"],
                blocking=BlockingMode.NOT_BLOCKED,
            ))

        self.curr_tile = (0, 0)
        av_pixels = np.full((step_size, step_size), avatar_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=av_pixels.tolist(),
            name="avatar",
            x=self.arena_origin * step_size,
            y=self.arena_origin * step_size,
            tags=["avatar"],
            blocking=BlockingMode.NOT_BLOCKED,
            layer=10,
        ))

        g_pixels = np.full((step_size, step_size), goal_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=g_pixels.tolist(),
            name="goal",
            x=(self.arena_origin + 5 * 2) * step_size,
            y=(self.arena_origin + 5 * 2) * step_size,
            tags=["goal"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng05_collapsing_path",
            levels=[level],
            camera=camera,
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )

    def get_sprite(self, name: str) -> Optional[Sprite]:
        res = self.current_level.get_sprites_by_name(name)
        return res[0] if res else None

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

        d_ax, d_ay = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: d_ay = -1
        elif act == GameAction.ACTION2: d_ay = 1
        elif act == GameAction.ACTION3: d_ax = -1
        elif act == GameAction.ACTION4: d_ax = 1

        n_ax = self.curr_tile[0] + d_ax
        n_ay = self.curr_tile[1] + d_ay

        if (n_ax, n_ay) not in self.active_floor_coords:
            self.lose()
            self.complete_action()
            return

        # Old floor tile disappears
        self.active_floor_coords.discard(self.curr_tile)
        old_sp = self.get_sprite(f"fl_{self.curr_tile[0]}_{self.curr_tile[1]}")
        if old_sp:
            old_sp.set_visible(False)

        self.curr_tile = (n_ax, n_ay)
        avatar.set_position(
            (self.arena_origin + n_ax * 2) * self.step_size,
            (self.arena_origin + n_ay * 2) * self.step_size,
        )

        if self.curr_tile in self.key_coords:
            self.key_coords.discard(self.curr_tile)
            k_sp = self.get_sprite(f"key_{self.curr_tile[0]}_{self.curr_tile[1]}")
            if k_sp:
                k_sp.set_visible(False)

        if len(self.key_coords) == 0 and self.curr_tile == (5, 5):
            self.win()

        self.complete_action()
