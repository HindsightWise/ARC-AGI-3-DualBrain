#!/usr/bin/env python3
"""
Paradigm 4: Multi-Agent & Desynchronization
Contains 5 novel ARC-AGI-3 game environments:
- ng16_quad_mirrors: 4 avatars across 4 quadrants with symmetric/asymmetric control reflections.
- ng17_predator_prey: Active heuristic chaser enemy that must be lured into a neutralization pit trap.
- ng18_convoy_escort: Autonomous VIP convoy pacing forward; player clears hazards along route.
- ng19_shadow_clone: Delayed temporal ghost trailing player by 4 steps and leaving hazardous footprints.
- ng20_split_merge: Dual-pawn control swapping (ACTION5) to coordinate pressure plates and merge at goal.
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


class NG16_QuadMirrors(ARCBaseGame):
    """4 avatars across 4 quadrants with symmetric/asymmetric control reflections."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 0
        wall_color = 5
        av_color = 11
        goal_color = 14

        sprites: List[Sprite] = []

        hw = np.full((step_size, grid_size), wall_color, dtype=np.int8)
        vw = np.full((grid_size, step_size), wall_color, dtype=np.int8)
        sprites.append(Sprite(pixels=hw.tolist(), name="w_top", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=hw.tolist(), name="w_bot", x=0, y=grid_size - step_size, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_l", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_r", x=grid_size - step_size, y=0, tags=["wall"]))

        mid = (grid_size // (2 * step_size)) * step_size
        sprites.append(Sprite(pixels=hw.tolist(), name="w_mid_h", x=0, y=mid, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_mid_v", x=mid, y=0, tags=["wall"]))

        av_pix = np.full((step_size, step_size), av_color, dtype=np.int8)
        sprites.append(Sprite(pixels=av_pix.tolist(), name="av_tl", x=3 * step_size, y=3 * step_size, tags=["avatar"]))
        sprites.append(Sprite(pixels=av_pix.tolist(), name="av_tr", x=12 * step_size, y=3 * step_size, tags=["avatar"]))
        sprites.append(Sprite(pixels=av_pix.tolist(), name="av_bl", x=3 * step_size, y=12 * step_size, tags=["avatar"]))
        sprites.append(Sprite(pixels=av_pix.tolist(), name="av_br", x=12 * step_size, y=12 * step_size, tags=["avatar"]))

        g_pix = np.full((step_size, step_size), goal_color, dtype=np.int8)
        sprites.append(Sprite(pixels=g_pix.tolist(), name="g_tl", x=5 * step_size, y=5 * step_size, tags=["goal"]))
        sprites.append(Sprite(pixels=g_pix.tolist(), name="g_tr", x=10 * step_size, y=5 * step_size, tags=["goal"]))
        sprites.append(Sprite(pixels=g_pix.tolist(), name="g_bl", x=5 * step_size, y=10 * step_size, tags=["goal"]))
        sprites.append(Sprite(pixels=g_pix.tolist(), name="g_br", x=10 * step_size, y=10 * step_size, tags=["goal"]))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng16_quad_mirrors",
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

        av1 = self.get_sprite("av_tl")
        av2 = self.get_sprite("av_tr")
        av3 = self.get_sprite("av_bl")
        av4 = self.get_sprite("av_br")

        g1 = self.get_sprite("g_tl")
        g2 = self.get_sprite("g_tr")
        g3 = self.get_sprite("g_bl")
        g4 = self.get_sprite("g_br")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        moves = [
            (av1, dx, dy),
            (av2, -dx, dy),
            (av3, dx, -dy),
            (av4, -dx, -dy),
        ]

        for av, mx, my in moves:
            nx = av.x + mx
            ny = av.y + my
            can_move = True
            for w in self.current_level.get_sprites_by_tag("wall"):
                if (nx < w.x + w.width and nx + av.width > w.x and
                    ny < w.y + w.height and ny + av.height > w.y):
                    can_move = False
                    break
            if can_move:
                av.set_position(nx, ny)

        w1 = (av1.x == g1.x and av1.y == g1.y)
        w2 = (av2.x == g2.x and av2.y == g2.y)
        w3 = (av3.x == g3.x and av3.y == g3.y)
        w4 = (av4.x == g4.x and av4.y == g4.y)

        if w1 and w2 and w3 and w4:
            self.win()

        self.complete_action()


class NG17_PredatorPrey(ARCBaseGame):
    """Active heuristic chaser enemy that must be lured into a neutralization pit trap."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 0
        wall_color = 5
        avatar_color = 11
        predator_color = 2
        pit_color = 8
        goal_color = 14

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
            x=3 * step_size,
            y=3 * step_size,
            tags=["avatar"],
            blocking=BlockingMode.BOUNDING_BOX,
            layer=10,
        ))

        pr_pix = np.full((step_size, step_size), predator_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=pr_pix.tolist(),
            name="predator",
            x=12 * step_size,
            y=12 * step_size,
            tags=["predator"],
            blocking=BlockingMode.BOUNDING_BOX,
        ))
        self.predator_active = True

        pt_pix = np.full((step_size, step_size), pit_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=pt_pix.tolist(),
            name="pit",
            x=8 * step_size,
            y=8 * step_size,
            tags=["pit"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        dr_pix = np.full((step_size, step_size), 2, dtype=np.int8)
        sprites.append(Sprite(
            pixels=dr_pix.tolist(),
            name="door",
            x=13 * step_size,
            y=8 * step_size,
            tags=["door", "wall"],
            blocking=BlockingMode.BOUNDING_BOX,
        ))

        g_pix = np.full((step_size, step_size), goal_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=g_pix.tolist(),
            name="goal",
            x=14 * step_size,
            y=8 * step_size,
            tags=["goal"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng17_predator_prey",
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
        predator = self.get_sprite("predator")
        pit = self.get_sprite("pit")
        door = self.get_sprite("door")
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
            if w.is_visible and (nx < w.x + w.width and nx + avatar.width > w.x and
                                 ny < w.y + w.height and ny + avatar.height > w.y):
                self.complete_action()
                return

        avatar.set_position(nx, ny)

        if self.predator_active:
            p_dx = 0
            p_dy = 0
            if predator.x < avatar.x: p_dx = self.step_size
            elif predator.x > avatar.x: p_dx = -self.step_size
            elif predator.y < avatar.y: p_dy = self.step_size
            elif predator.y > avatar.y: p_dy = -self.step_size

            p_nx = predator.x + p_dx
            p_ny = predator.y + p_dy
            predator.set_position(p_nx, p_ny)

            if predator.x == avatar.x and predator.y == avatar.y:
                self.lose()
                self.complete_action()
                return

            if predator.x == pit.x and predator.y == pit.y:
                self.predator_active = False
                predator.set_visible(False)
                door.set_visible(False)
                door.set_blocking(BlockingMode.NOT_BLOCKED)

        if avatar.x >= goal.x and avatar.y == goal.y:
            self.win()

        self.complete_action()


class NG18_ConvoyEscort(ARCBaseGame):
    """Autonomous VIP convoy pacing forward; player clears hazards along route."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 0
        wall_color = 5
        convoy_color = 12
        hazard_rock = 9
        avatar_color = 11
        goal_color = 14

        sprites: List[Sprite] = []

        hw = np.full((step_size, grid_size), wall_color, dtype=np.int8)
        vw = np.full((grid_size, step_size), wall_color, dtype=np.int8)
        sprites.append(Sprite(pixels=hw.tolist(), name="w_top", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=hw.tolist(), name="w_bot", x=0, y=grid_size - step_size, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_l", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_r", x=grid_size - step_size, y=0, tags=["wall"]))

        cv_pix = np.full((step_size, step_size), convoy_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=cv_pix.tolist(),
            name="convoy",
            x=2 * step_size,
            y=8 * step_size,
            tags=["convoy"],
            blocking=BlockingMode.BOUNDING_BOX,
        ))
        self.convoy_step_timer = 0

        self.rock_coords = {(6, 8), (10, 8)}
        hr_pix = np.full((step_size, step_size), hazard_rock, dtype=np.int8)
        for rx, ry in self.rock_coords:
            sprites.append(Sprite(
                pixels=hr_pix.tolist(),
                name=f"rock_{rx}_{ry}",
                x=rx * step_size,
                y=ry * step_size,
                tags=["hazard_rock"],
                blocking=BlockingMode.BOUNDING_BOX,
            ))

        av_pix = np.full((step_size, step_size), avatar_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=av_pix.tolist(),
            name="avatar",
            x=4 * step_size,
            y=4 * step_size,
            tags=["avatar"],
            blocking=BlockingMode.BOUNDING_BOX,
            layer=10,
        ))

        g_pix = np.full((step_size, step_size), goal_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=g_pix.tolist(),
            name="dest",
            x=13 * step_size,
            y=8 * step_size,
            tags=["goal"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng18_convoy_escort",
            levels=[level],
            camera=camera,
            available_actions=[1, 2, 3, 4, 5],
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
        convoy = self.get_sprite("convoy")
        dest = self.get_sprite("dest")

        act = self._action.id
        if act == GameAction.ACTION5:
            px = avatar.x // self.step_size
            py = avatar.y // self.step_size
            for adj in [(px, py), (px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)]:
                if adj in self.rock_coords:
                    self.rock_coords.discard(adj)
                    r_sp = self.get_sprite(f"rock_{adj[0]}_{adj[1]}")
                    if r_sp:
                        r_sp.set_visible(False)
                        r_sp.set_blocking(BlockingMode.NOT_BLOCKED)
                    break
        else:
            dx, dy = 0, 0
            if act == GameAction.ACTION1: dy = -self.step_size
            elif act == GameAction.ACTION2: dy = self.step_size
            elif act == GameAction.ACTION3: dx = -self.step_size
            elif act == GameAction.ACTION4: dx = self.step_size

            nx = avatar.x + dx
            ny = avatar.y + dy

            can_move = True
            for w in self.current_level.get_sprites_by_tag("wall"):
                if (nx < w.x + w.width and nx + avatar.width > w.x and
                    ny < w.y + w.height and ny + avatar.height > w.y):
                    can_move = False
                    break
            if can_move:
                avatar.set_position(nx, ny)

        self.convoy_step_timer += 1
        if self.convoy_step_timer % 2 == 0:
            c_nx = convoy.x + self.step_size
            c_tx = c_nx // self.step_size
            c_ty = convoy.y // self.step_size

            if (c_tx, c_ty) in self.rock_coords:
                self.lose()
                self.complete_action()
                return

            convoy.set_position(c_nx, convoy.y)

            if convoy.x >= dest.x:
                self.win()

        self.complete_action()


class NG19_ShadowClone(ARCBaseGame):
    """Delayed temporal ghost trailing player by 4 steps and leaving hazardous footprints."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 0
        wall_color = 5
        avatar_color = 11
        shadow_color = 6
        goal_color = 14

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
            x=3 * step_size,
            y=3 * step_size,
            tags=["avatar"],
            blocking=BlockingMode.BOUNDING_BOX,
            layer=10,
        ))

        sh_pix = np.full((step_size, step_size), shadow_color, dtype=np.int8)
        sh_sp = Sprite(
            pixels=sh_pix.tolist(),
            name="shadow",
            x=3 * step_size,
            y=3 * step_size,
            tags=["shadow"],
            blocking=BlockingMode.NOT_BLOCKED,
        )
        sh_sp.set_visible(False)
        sprites.append(sh_sp)

        g_pix = np.full((step_size, step_size), goal_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=g_pix.tolist(),
            name="goal",
            x=12 * step_size,
            y=12 * step_size,
            tags=["goal"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        self.history: List[Tuple[int, int]] = [(3 * step_size, 3 * step_size)]
        self.toxic_tiles: Set[Tuple[int, int]] = set()

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng19_shadow_clone",
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
        shadow = self.get_sprite("shadow")
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

        if (nx, ny) in self.toxic_tiles or (shadow.is_visible and nx == shadow.x and ny == shadow.y):
            self.lose()
            self.complete_action()
            return

        avatar.set_position(nx, ny)
        self.history.append((nx, ny))

        delay = 4
        if len(self.history) > delay:
            shadow.set_visible(True)
            old_pos = self.history[-delay]
            shadow.set_position(old_pos[0], old_pos[1])
            self.toxic_tiles.add(old_pos)

            tr_pix = np.full((self.step_size, self.step_size), 2, dtype=np.int8)
            self.current_level.add_sprite(Sprite(
                pixels=tr_pix.tolist(),
                name=f"toxic_{len(self.toxic_tiles)}",
                x=old_pos[0],
                y=old_pos[1],
                tags=["toxic"],
                blocking=BlockingMode.NOT_BLOCKED,
            ))

        if avatar.x == goal.x and avatar.y == goal.y:
            self.win()

        self.complete_action()


class NG20_SplitMerge(ARCBaseGame):
    """Dual-pawn control swapping (ACTION5) to coordinate pressure plates and merge at goal."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 0
        wall_color = 5
        pawn1_active = 11
        pawn2_inactive = 15
        gate_color = 2
        goal_color = 14

        sprites: List[Sprite] = []

        hw = np.full((step_size, grid_size), wall_color, dtype=np.int8)
        vw = np.full((grid_size, step_size), wall_color, dtype=np.int8)
        sprites.append(Sprite(pixels=hw.tolist(), name="w_top", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=hw.tolist(), name="w_bot", x=0, y=grid_size - step_size, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_l", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_r", x=grid_size - step_size, y=0, tags=["wall"]))

        w_tile = np.full((step_size, step_size), wall_color, dtype=np.int8)
        for x_idx in range(grid_size // step_size):
            if x_idx not in [6, 10]:
                sprites.append(Sprite(
                    pixels=w_tile.tolist(),
                    name=f"mid_{x_idx}",
                    x=x_idx * step_size,
                    y=8 * step_size,
                    tags=["wall"],
                    blocking=BlockingMode.BOUNDING_BOX,
                ))

        g_pix = np.full((step_size, step_size), gate_color, dtype=np.int8)
        sprites.append(Sprite(pixels=g_pix.tolist(), name="gate1", x=6 * step_size, y=8 * step_size, tags=["gate", "wall"], blocking=BlockingMode.BOUNDING_BOX))
        sprites.append(Sprite(pixels=g_pix.tolist(), name="gate2", x=10 * step_size, y=8 * step_size, tags=["gate", "wall"], blocking=BlockingMode.BOUNDING_BOX))

        pl_pix = np.full((step_size, step_size), 4, dtype=np.int8)
        sprites.append(Sprite(pixels=pl_pix.tolist(), name="plate1", x=3 * step_size, y=4 * step_size, tags=["plate"]))
        sprites.append(Sprite(pixels=pl_pix.tolist(), name="plate2", x=12 * step_size, y=4 * step_size, tags=["plate"]))

        p1_pix = np.full((step_size, step_size), pawn1_active, dtype=np.int8)
        sprites.append(Sprite(pixels=p1_pix.tolist(), name="pawn1", x=3 * step_size, y=6 * step_size, tags=["avatar", "pawn1"], layer=10))
        p2_pix = np.full((step_size, step_size), pawn2_inactive, dtype=np.int8)
        sprites.append(Sprite(pixels=p2_pix.tolist(), name="pawn2", x=12 * step_size, y=6 * step_size, tags=["avatar", "pawn2"], layer=10))

        self.active_pawn = 1

        m_pix = np.full((step_size, step_size), goal_color, dtype=np.int8)
        sprites.append(Sprite(pixels=m_pix.tolist(), name="beacon", x=8 * step_size, y=12 * step_size, tags=["goal"]))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng20_split_merge",
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

        pawn1 = self.get_sprite("pawn1")
        pawn2 = self.get_sprite("pawn2")
        plate1 = self.get_sprite("plate1")
        plate2 = self.get_sprite("plate2")
        gate1 = self.get_sprite("gate1")
        gate2 = self.get_sprite("gate2")
        beacon = self.get_sprite("beacon")

        act = self._action.id
        if act == GameAction.ACTION5:
            self.active_pawn = 2 if self.active_pawn == 1 else 1
            if self.active_pawn == 1:
                pawn1.pixels = np.full((self.step_size, self.step_size), 11, dtype=np.int8)
                pawn2.pixels = np.full((self.step_size, self.step_size), 15, dtype=np.int8)
            else:
                pawn1.pixels = np.full((self.step_size, self.step_size), 1, dtype=np.int8)
                pawn2.pixels = np.full((self.step_size, self.step_size), 7, dtype=np.int8)
            self.complete_action()
            return

        curr_pawn = pawn1 if self.active_pawn == 1 else pawn2
        dx, dy = 0, 0
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx = curr_pawn.x + dx
        ny = curr_pawn.y + dy

        for w in self.current_level.get_sprites_by_tag("wall"):
            if w.is_visible and (nx < w.x + w.width and nx + curr_pawn.width > w.x and
                                 ny < w.y + w.height and ny + curr_pawn.height > w.y):
                self.complete_action()
                return

        curr_pawn.set_position(nx, ny)

        if pawn1.x == plate1.x and pawn1.y == plate1.y:
            self.gate2_open = True
        if pawn2.x == plate2.x and pawn2.y == plate2.y:
            self.gate1_open = True

        if getattr(self, "gate2_open", False):
            gate2.set_visible(False)
            gate2.set_blocking(BlockingMode.NOT_BLOCKED)
        if getattr(self, "gate1_open", False):
            gate1.set_visible(False)
            gate1.set_blocking(BlockingMode.NOT_BLOCKED)

        if (pawn1.x == beacon.x and pawn1.y == beacon.y and
            pawn2.x == beacon.x and pawn2.y == beacon.y):
            self.win()

        self.complete_action()
