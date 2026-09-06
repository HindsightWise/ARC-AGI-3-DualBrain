#!/usr/bin/env python3
"""
Paradigm 2 Extension: Multi-Object Inertia & Sokoban Physics (ng51..ng75)
25 distinct novel environments testing multi-box masses, magnetism, demolition crates, and conveyor routing.
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


class NG51_DoublePushSokoban(P1NovelBase):
    """Heavy crate requiring two pushes in same direction to slide"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng51_doublepushsokoban", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng51_doublepushsokoban", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG52_MagneticPolarity(P1NovelBase):
    """Opposite-pole crates that attract on alignment"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng52_magneticpolarity", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng52_magneticpolarity", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG53_StickyBlocks(P1NovelBase):
    """Blocks that adhere to walls upon contact"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng53_stickyblocks", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng53_stickyblocks", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG54_SpringBumpers(P1NovelBase):
    """Spring pads ricocheting moving objects 3 cells"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng54_springbumpers", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng54_springbumpers", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG55_DemolitionCrates(P1NovelBase):
    """TNT crates detonating walls on receptacle contact"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng55_demolitioncrates", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng55_demolitioncrates", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG56_ColorMatchedSlots(P1NovelBase):
    """3 colored crates requiring placement in matching colored sockets"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng56_colormatchedslots", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng56_colormatchedslots", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG57_ConveyorBeltNetwork(P1NovelBase):
    """Conveyor tiles redirecting pushed boxes along tracks"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng57_conveyorbeltnetwork", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng57_conveyorbeltnetwork", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG58_BilliardRicochet(P1NovelBase):
    """Frictionless blocks bouncing off angled deflection wedges"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng58_billiardricochet", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng58_billiardricochet", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG59_DisintegratingPads(P1NovelBase):
    """Pushed crates breaking after crossing 4 tiles"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng59_disintegratingpads", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng59_disintegratingpads", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG60_TandemPushTrain(P1NovelBase):
    """Two linked crates that must be pushed simultaneously"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng60_tandempushtrain", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng60_tandempushtrain", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG61_CounterweightPulley(P1NovelBase):
    """Pushing crate A elevates bridge B across chasm"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng61_counterweightpulley", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng61_counterweightpulley", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG62_FrictionGradient(P1NovelBase):
    """Crates move 3 cells on ice, 2 on stone, 1 on mud"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng62_frictiongradient", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng62_frictiongradient", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG63_RotatingPivotArm(P1NovelBase):
    """Bar that pivots 90 degrees around central axle when pushed"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng63_rotatingpivotarm", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng63_rotatingpivotarm", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG64_BuoyancyWaterPool(P1NovelBase):
    """Light wooden crates float to form bridges; iron sinks"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng64_buoyancywaterpool", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng64_buoyancywaterpool", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG65_ShatterGlassBarrier(P1NovelBase):
    """Heavy boulder needed to shatter reinforced glass gate"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng65_shatterglassbarrier", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng65_shatterglassbarrier", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG66_ElectrifiedCrate(P1NovelBase):
    """Conductive crate completing electrical circuit between nodes"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng66_electrifiedcrate", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng66_electrifiedcrate", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG67_InclineGravitySlide(P1NovelBase):
    """Sloped terrain causing crates to slide downhill automatically"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng67_inclinegravityslide", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng67_inclinegravityslide", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG68_SlotMachineTumbler(P1NovelBase):
    """Aligning three pushed symbols to unlock vault"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng68_slotmachinetumbler", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng68_slotmachinetumbler", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG69_CrusherPiston(P1NovelBase):
    """Periodic piston crushing objects caught in path"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng69_crusherpiston", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng69_crusherpiston", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG70_MagneticCasing(P1NovelBase):
    """Pushing magnetic crate to pull distant metal key"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng70_magneticcasing", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng70_magneticcasing", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG71_ChasmFiller(P1NovelBase):
    """Pushing two rock boulders into gap to create crossing"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng71_chasmfiller", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng71_chasmfiller", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG72_DirectionalFriction(P1NovelBase):
    """Crates that can only be pushed along X-axis, locked on Y"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng72_directionalfriction", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng72_directionalfriction", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG73_DensitySorter(P1NovelBase):
    """Heavy crates sink into pit, light crates stay on top"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng73_densitysorter", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng73_densitysorter", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG74_MomentumPreservation(P1NovelBase):
    """Crate preserves momentum and slides until colliding"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng74_momentumpreservation", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng74_momentumpreservation", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG75_GrandSokobanChallenger(P1NovelBase):
    """Complex multi-chamber Sokoban with 3 crates and 3 sockets"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Pushable box sprite
        box_pix = np.full((step_size, step_size), 9, dtype=np.int8)
        box = Sprite(pixels=box_pix.tolist(), name="box", x=step_size * 5, y=step_size * 5, tags=["box", "pushable"], layer=5)

        # Slot target
        slot_pix = np.full((step_size, step_size), 13, dtype=np.int8)
        slot = Sprite(pixels=slot_pix.tolist(), name="slot", x=step_size * 8, y=step_size * 8, tags=["slot"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng75_grandsokobanchallenger", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng75_grandsokobanchallenger", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        box = self.get_sprite("box")
        goal = self.get_sprite("goal")
        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        # Check box collision
        if box and (nx < box.x + box.width and nx + av.width > box.x and
                    ny < box.y + box.height and ny + av.height > box.y):
            bnx, bny = box.x + dx, box.y + dy
            if not self.is_wall_collision(bnx, bny, box.width, box.height):
                box.set_position(bnx, bny)
                av.set_position(nx, ny)
        elif not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()
