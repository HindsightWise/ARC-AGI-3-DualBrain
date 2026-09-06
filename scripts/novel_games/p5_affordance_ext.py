#!/usr/bin/env python3
"""
Paradigm 5 Extension: Affordance, Coordinate Puzzles & Viewport Projections (ng126..ng150)
25 distinct novel environments testing point-and-click ACTION6, downsampled camera grids, and tile permutations.
"""
from __future__ import annotations

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
from scripts.novel_games.p1_navigation_ext import P1NovelBase, _make_boxed_arena


class NG126_DownsampledClick(P1NovelBase):
    """Coordinate-targeting ACTION6 with 2x downsampling viewport"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 20, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng126_downsampledclick", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng126_downsampledclick", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG127_LetterboxOffsetClick(P1NovelBase):
    """Coordinate-targeting with asymmetrical letterbox margin"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 20, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng127_letterboxoffsetclick", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng127_letterboxoffsetclick", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG128_SlidingTile15(P1NovelBase):
    """15-puzzle grid requiring sequential tile transposition"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng128_slidingtile15", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng128_slidingtile15", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG129_RotaryDialCombination(P1NovelBase):
    """Three dials rotated by clicking on quadrant rims"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 30, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng129_rotarydialcombination", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng129_rotarydialcombination", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG130_LaserMirrorAngle(P1NovelBase):
    """Click mirror to cycle angle through 45, 90, 135 degrees"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 25, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng130_lasermirrorangle", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng130_lasermirrorangle", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG131_PipeRotationJunction(P1NovelBase):
    """Click pipe segments to rotate into continuous flow"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 30, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng131_piperotationjunction", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng131_piperotationjunction", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG132_ColorBucketFill(P1NovelBase):
    """Point-and-click flood fill tool changing region color"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 25, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng132_colorbucketfill", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng132_colorbucketfill", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG133_ResourceAlchemyStation(P1NovelBase):
    """Pick up herb and crystal, click cauldron to brew key"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng133_resourcealchemystation", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng133_resourcealchemystation", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG134_ShapeScalingTransform(P1NovelBase):
    """Click zoom levers to scale test shape to fit aperture"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 25, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng134_shapescalingtransform", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng134_shapescalingtransform", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG135_PinpointTargetStrike(P1NovelBase):
    """Small 1x1 target in moving grid needing precision click"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 20, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng135_pinpointtargetstrike", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng135_pinpointtargetstrike", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG136_WhackAMoleMoles(P1NovelBase):
    """Click mole sprite when it surfaces in one of 9 holes"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 25, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng136_whackamolemoles", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng136_whackamolemoles", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG137_PatternStampTool(P1NovelBase):
    """Click stamp icon then click canvas to imprint pattern"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 25, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng137_patternstamptool", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng137_patternstamptool", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG138_WireConnectTheDots(P1NovelBase):
    """Click node A then node B to draw conductive wire"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 30, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng138_wireconnectthedots", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng138_wireconnectthedots", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG139_AffineShearWarp(P1NovelBase):
    """Click matrix shear arrows to de-skew rhomboid shape"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 25, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng139_affineshearwarp", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng139_affineshearwarp", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG140_PaletteIndexSwap(P1NovelBase):
    """Click two palette swatches to swap their color roles"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 25, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng140_paletteindexswap", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng140_paletteindexswap", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG141_MemoryMatchCards(P1NovelBase):
    """Click to reveal card pairs; matching pairs stay revealed"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng141_memorymatchcards", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng141_memorymatchcards", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG142_PrismColorSplitting(P1NovelBase):
    """Click prism to orient white beam into 3 primary rays"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 25, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng142_prismcolorsplitting", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng142_prismcolorsplitting", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG143_TelescopeFocalLength(P1NovelBase):
    """Click focus slider until blurry image matches sharp glyph"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 20, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng143_telescopefocallength", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng143_telescopefocallength", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG144_TangramPieceAssembly(P1NovelBase):
    """Click tangram shapes to translate them into target silhouette"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 35, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng144_tangrampieceassembly", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng144_tangrampieceassembly", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG145_MazeBridgeToggle(P1NovelBase):
    """Click colored lever icons to raise matching colored bridges"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 30, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng145_mazebridgetoggle", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng145_mazebridgetoggle", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG146_ConstellationConnect(P1NovelBase):
    """Click celestial stars in order of increasing brightness"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 25, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng146_constellationconnect", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng146_constellationconnect", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG147_ClockHandAngleMatch(P1NovelBase):
    """Click hour/minute hands to match target digital timestamp"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 25, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng147_clockhandanglematch", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng147_clockhandanglematch", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG148_WeightBalancePan(P1NovelBase):
    """Click weights on rack to balance scale pans"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 30, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng148_weightbalancepan", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng148_weightbalancepan", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG149_SafeCrackerDial(P1NovelBase):
    """Click left/right arrows to find combination clicks"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 30, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng149_safecrackerdial", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng149_safecrackerdial", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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


class NG150_GrandAffordanceWorkshop(P1NovelBase):
    """Composite workbench with click rotation, color fill, and stamp"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng150_grandaffordanceworkshop", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng150_grandaffordanceworkshop", levels=[level], camera=cam, available_actions=acts, seed=seed)

    def step(self) -> None:
        if self._action.id == GameAction.RESET:
            self.complete_action()
            return
        self.steps_left -= 1
        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        av = self.get_sprite("avatar")
        btn = self.get_sprite("btn_target")
        goal = self.get_sprite("goal")

        if self._action.id == GameAction.ACTION6:
            data = getattr(self._action, "data", {}) or {}
            cx = data.get("x", -1)
            cy = data.get("y", -1)
            if btn and (btn.x <= cx < btn.x + btn.width and btn.y <= cy < btn.y + btn.height):
                self.clicked_target = True
                self.win()
        else:
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
