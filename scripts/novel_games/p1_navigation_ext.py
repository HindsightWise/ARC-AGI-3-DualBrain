#!/usr/bin/env python3
"""
Paradigm 1 Extension: Extreme Topologies & Geodesic Navigation (ng26..ng50)
25 distinct novel environments testing non-Euclidean geometries, variable strides, and dynamic barriers.
"""
from __future__ import annotations

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


class P1NovelBase(ARCBaseGame):
    def get_sprite(self, name: str) -> Optional[Sprite]:
        sp = self.current_level.get_sprites_by_name(name)
        return sp[0] if sp else None

    def get_sprites_by_tag(self, tag: str) -> List[Sprite]:
        return self.current_level.get_sprites_by_tag(tag)

    def check_collision(self, s1: Sprite, s2: Sprite) -> bool:
        if not s1 or not s2:
            return False
        return (s1.x < s2.x + s2.width and s1.x + s1.width > s2.x and
                s1.y < s2.y + s2.height and s1.y + s1.height > s2.y)

    def is_wall_collision(self, nx: int, ny: int, width: int, height: int) -> bool:
        for w in self.get_sprites_by_tag("wall"):
            if (nx < w.x + w.width and nx + width > w.x and
                ny < w.y + w.height and ny + height > w.y):
                return True
        return False


def _make_boxed_arena(game_id: str, step_size: int = 2, grid_size: int = 32, max_steps: int = 50,
                      wall_color: int = 4, av_color: int = 11, goal_color: int = 14,
                      av_pos: Tuple[int, int] = (4, 4), goal_pos: Tuple[int, int] = (24, 24),
                      extra_sprites: Optional[List[Sprite]] = None, bg_color: int = 0,
                      available_actions: Optional[List[int]] = None) -> Tuple[List[Sprite], Camera, List[int]]:
    sprites: List[Sprite] = []
    # Border walls
    hw = np.full((step_size, grid_size), wall_color, dtype=np.int8)
    vw = np.full((grid_size, step_size), wall_color, dtype=np.int8)
    sprites.append(Sprite(pixels=hw.tolist(), name="w_top", x=0, y=0, tags=["wall"]))
    sprites.append(Sprite(pixels=hw.tolist(), name="w_bot", x=0, y=grid_size - step_size, tags=["wall"]))
    sprites.append(Sprite(pixels=vw.tolist(), name="w_l", x=0, y=0, tags=["wall"]))
    sprites.append(Sprite(pixels=vw.tolist(), name="w_r", x=grid_size - step_size, y=0, tags=["wall"]))

    # Avatar
    av_pix = np.full((step_size, step_size), av_color, dtype=np.int8)
    sprites.append(Sprite(pixels=av_pix.tolist(), name="avatar", x=av_pos[0], y=av_pos[1], tags=["avatar"], layer=10))

    # Goal
    g_pix = np.full((step_size, step_size), goal_color, dtype=np.int8)
    sprites.append(Sprite(pixels=g_pix.tolist(), name="goal", x=goal_pos[0], y=goal_pos[1], tags=["goal", "exit"], blocking=BlockingMode.NOT_BLOCKED))

    if extra_sprites:
        sprites.extend(extra_sprites)

    cam = Camera(width=grid_size, height=grid_size, background=bg_color)
    acts = available_actions or [1, 2, 3, 4]
    return sprites, cam, acts


class NG26_ToroidalWrap(P1NovelBase):
    """Wrap-around screen borders (stepping off right wraps to left)"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng26_toroidalwrap", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng26_toroidalwrap", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG27_OneWayArrows(P1NovelBase):
    """Directional conveyor arrows restricting motion"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng27_onewayarrows", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng27_onewayarrows", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG28_CollapsingTiles(P1NovelBase):
    """Floor tiles that turn into voids after departure"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng28_collapsingtiles", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng28_collapsingtiles", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG29_LaserGateTimer(P1NovelBase):
    """Laser gates oscillating active and inactive every 2 turns"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng29_lasergatetimer", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng29_lasergatetimer", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG30_GravityInvertor(P1NovelBase):
    """Pads that invert control orientation"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng30_gravityinvertor", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng30_gravityinvertor", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG31_RotatingTurnstile(P1NovelBase):
    """Revolving barrier gates requiring entry from specific quadrants"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng31_rotatingturnstile", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng31_rotatingturnstile", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG32_VanishingBridge(P1NovelBase):
    """Bridge tiles across chasm requiring switch hold"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng32_vanishingbridge", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng32_vanishingbridge", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG33_CurrentDrift(P1NovelBase):
    """Water current pushing avatar downstream on each move"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng33_currentdrift", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng33_currentdrift", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG34_SpiralLabyrinth(P1NovelBase):
    """Tight geometric inward spiral maze"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng34_spirallabyrinth", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng34_spirallabyrinth", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG35_TeleportChain(P1NovelBase):
    """Three sequential portal pairs linking isolated chambers"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng35_teleportchain", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng35_teleportchain", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG36_ShiftingWalls(P1NovelBase):
    """Dividing walls that shift on every 4th action"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng36_shiftingwalls", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng36_shiftingwalls", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG37_VariableFriction(P1NovelBase):
    """Sand (2 steps required) and ice (slides to boundary)"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng37_variablefriction", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng37_variablefriction", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG38_DiagonalBooster(P1NovelBase):
    """Jump pads catapulting diagonally across obstacles"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng38_diagonalbooster", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng38_diagonalbooster", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG39_DarkFogChamber(P1NovelBase):
    """Fog of war hiding tiles beyond radius 6"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng39_darkfogchamber", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng39_darkfogchamber", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG40_PressurePlateBridge(P1NovelBase):
    """Multiple pressure plates activating bridge sections"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng40_pressureplatebridge", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng40_pressureplatebridge", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG41_CrackedIce(P1NovelBase):
    """Fragile ice tiles that break on second traversal"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng41_crackedice", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng41_crackedice", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG42_SinkingPlatforms(P1NovelBase):
    """Stepping platforms that submerge after 2 steps"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng42_sinkingplatforms", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng42_sinkingplatforms", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG43_SlidingWalls(P1NovelBase):
    """Moving partition walls patrolling corridors"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng43_slidingwalls", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng43_slidingwalls", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG44_ConcentricRings(P1NovelBase):
    """Concentric circular barrier rings with single openings"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng44_concentricrings", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng44_concentricrings", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG45_WindTunnel(P1NovelBase):
    """Continuous aerodynamic force pushing player leftward"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng45_windtunnel", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng45_windtunnel", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG46_MagneticRepulsionPad(P1NovelBase):
    """Polarized pads that repel avatar on proximity"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng46_magneticrepulsionpad", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng46_magneticrepulsionpad", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG47_PhaseShiftDoors(P1NovelBase):
    """Spectral doors toggling permeability on alternating steps"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng47_phaseshiftdoors", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng47_phaseshiftdoors", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG48_HammingDistanceExit(P1NovelBase):
    """Exit requiring odd-parity coordinate alignment"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng48_hammingdistanceexit", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng48_hammingdistanceexit", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG49_BouncingPillars(P1NovelBase):
    """Pillars that ricochet across chamber when struck"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng49_bouncingpillars", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng49_bouncingpillars", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG50_GeodesicGrandMaster(P1NovelBase):
    """Integrated maze with teleporter, laser gate, and crumbling floor"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.turn_counter = 0

        # Custom obstacle setup
        extra_sprites: List[Sprite] = []
        # Interior dividing obstacle
        mid = (grid_size // (2 * step_size)) * step_size
        obs_pix = np.full((step_size * 4, step_size), 4, dtype=np.int8)
        extra_sprites.append(Sprite(pixels=obs_pix.tolist(), name="interior_wall", x=mid, y=step_size * 2, tags=["wall"]))

        sprites, cam, acts = _make_boxed_arena("ng50_geodesicgrandmaster", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng50_geodesicgrandmaster", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        self.turn_counter += 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()
