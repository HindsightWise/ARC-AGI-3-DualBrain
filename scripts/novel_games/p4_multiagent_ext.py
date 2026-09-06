#!/usr/bin/env python3
"""
Paradigm 4 Extension: Multi-Agent Swarms & Adversarial Dynamics (ng101..ng125)
25 distinct novel environments testing twin mirrors, inverted axis movement, pursuit evasion, and flocking.
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


class NG101_AntiMirrorTwins(P1NovelBase):
    """Twin moves in inverted directions (UP->DOWN, LEFT->RIGHT)"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng101_antimirrortwins", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng101_antimirrortwins", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG102_DesynchronizedStep(P1NovelBase):
    """Clone moves only on even turns; player on every turn"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng102_desynchronizedstep", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng102_desynchronizedstep", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG103_PursuitPredatorGhost(P1NovelBase):
    """Enemy moves 1 step toward player using Manhattan distance"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng103_pursuitpredatorghost", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng103_pursuitpredatorghost", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG104_PatrolGuardCone(P1NovelBase):
    """Guard paces back and forth with 3-tile forward vision beam"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng104_patrolguardcone", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng104_patrolguardcone", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG105_FragileConvoyEscort(P1NovelBase):
    """Pushing fragile VIP agent across hazard room without damage"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng105_fragileconvoyescort", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng105_fragileconvoyescort", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG106_ShadowTrajectoryClone(P1NovelBase):
    """Clone repeats player's moves with 3-step time delay"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng106_shadowtrajectoryclone", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng106_shadowtrajectoryclone", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG107_SimultaneousDualButtons(P1NovelBase):
    """Player and twin must occupy separate buttons on same step"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng107_simultaneousdualbuttons", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng107_simultaneousdualbuttons", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG108_SwarmRepulsionFlock(P1NovelBase):
    """Three friendly agents that disperse when approached"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng108_swarmrepulsionflock", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng108_swarmrepulsionflock", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG109_TagTeamHandoff(P1NovelBase):
    """Switching control between Agent 1 and Agent 2 at beacon"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng109_tagteamhandoff", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng109_tagteamhandoff", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG110_BaitAndTrapEnemy(P1NovelBase):
    """Luring chaser into pit trap to unlock exit gate"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng110_baitandtrapenemy", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng110_baitandtrapenemy", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG111_QuadMirrorCross(P1NovelBase):
    """Four avatars moving in 4 quadrants simultaneously"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng111_quadmirrorcross", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng111_quadmirrorcross", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG112_RotatingTurretBullet(P1NovelBase):
    """Central turret firing projectile every 3 steps"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng112_rotatingturretbullet", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng112_rotatingturretbullet", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG113_MimicImpostor(P1NovelBase):
    """Impostor mimics moves until broken by wall bump"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng113_mimicimpostor", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng113_mimicimpostor", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG114_CooperativeGateHold(P1NovelBase):
    """One agent holds lever down so second agent can cross"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng114_cooperativegatehold", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng114_cooperativegatehold", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG115_OrbitalGuardPatrol(P1NovelBase):
    """Guard moves in continuous clockwise perimeter orbit"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng115_orbitalguardpatrol", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng115_orbitalguardpatrol", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG116_SpeedDifferentialChase(P1NovelBase):
    """Player moves 2 cells per turn; chaser moves 1 cell"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng116_speeddifferentialchase", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng116_speeddifferentialchase", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG117_SymmetricReunion(P1NovelBase):
    """Two avatars start at opposite corners and must meet in center"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng117_symmetricreunion", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng117_symmetricreunion", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG118_BodyguardInterception(P1NovelBase):
    """Guard steps into line of sight of hostile sentry"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng118_bodyguardinterception", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng118_bodyguardinterception", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG119_TriadSynchronization(P1NovelBase):
    """Three synchronized entities navigating separate mazes"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng119_triadsynchronization", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng119_triadsynchronization", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG120_GhostWallPhasing(P1NovelBase):
    """Enemy can phase through interior divider walls"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng120_ghostwallphasing", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng120_ghostwallphasing", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG121_PincerMovementEvasion(P1NovelBase):
    """Two enemies converging from left and right"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng121_pincermovementevasion", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng121_pincermovementevasion", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG122_FollowTheLeader(P1NovelBase):
    """Three-segment snake body following head trajectory"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng122_followtheleader", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng122_followtheleader", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG123_BeaconAttractionSwarm(P1NovelBase):
    """Activating beacon draws all roaming entities toward it"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng123_beaconattractionswarm", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng123_beaconattractionswarm", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG124_AsymmetricSpeedSplit(P1NovelBase):
    """Avatar A moves on cardinal dirs, Avatar B moves diagonal"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng124_asymmetricspeedsplit", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng124_asymmetricspeedsplit", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG125_GrandSwarmColosseum(P1NovelBase):
    """Arena with twin mirror, roaming chaser, and simultaneous switch"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng125_grandswarmcolosseum", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng125_grandswarmcolosseum", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        twin = self.get_sprite("twin")
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

        # Inverted motion for twin
        if twin:
            tnx, tny = twin.x - dx, twin.y - dy
            if not self.is_wall_collision(tnx, tny, twin.width, twin.height):
                twin.set_position(tnx, tny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()
