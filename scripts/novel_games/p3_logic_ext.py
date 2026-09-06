#!/usr/bin/env python3
"""
Paradigm 3 Extension: Logic Gates, Latin Squares & Cellular Automata (ng76..ng100)
25 distinct novel environments testing boolean networks, parity puzzles, cellular automata, and chromatic algebra.
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


class NG76_LightsOutParity(P1NovelBase):
    """3x3 toggle grid where stepping flips adjacent tile parity"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng76_lightsoutparity", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng76_lightsoutparity", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG77_SudokuLatin4x4(P1NovelBase):
    """Fill 4x4 Latin Square so no duplicate colors exist per row/col"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng77_sudokulatin4x4", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng77_sudokulatin4x4", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG78_CellularAutomatonRule30(P1NovelBase):
    """1D cellular automaton step evolution matching target ribbon"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng78_cellularautomatonrule30", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng78_cellularautomatonrule30", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG79_ConwayLifeStep2D(P1NovelBase):
    """Conway Game of Life single-generation forward prediction"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng79_conwaylifestep2d", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng79_conwaylifestep2d", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG80_SequenceArithmetic(P1NovelBase):
    """Extend arithmetic/geometric progression to open exit"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng80_sequencearithmetic", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng80_sequencearithmetic", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG81_SubtractiveColorMix(P1NovelBase):
    """Combine Cyan, Magenta, Yellow to unlock Black gate"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng81_subtractivecolormix", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng81_subtractivecolormix", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG82_AdditiveRGBMix(P1NovelBase):
    """Combine Red, Green, Blue to create White illumination key"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng82_additivergbmix", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng82_additivergbmix", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG83_FourWaySymmetry(P1NovelBase):
    """Paint quadrant to match 4-way rotational symmetry"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng83_fourwaysymmetry", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng83_fourwaysymmetry", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG84_BooleanANDGate(P1NovelBase):
    """Activate two switches simultaneously to power AND gate door"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng84_booleanandgate", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng84_booleanandgate", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG85_BooleanXORSwitch(P1NovelBase):
    """Either switch A or switch B, but not both, unlocks path"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng85_booleanxorswitch", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng85_booleanxorswitch", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG86_ParityBitVerifier(P1NovelBase):
    """Check row bit sums; correct parity bit unlocks vault"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng86_paritybitverifier", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng86_paritybitverifier", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG87_HammingWeightMatcher(P1NovelBase):
    """Turn on exactly 5 bits across 8-switch register"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng87_hammingweightmatcher", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng87_hammingweightmatcher", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG88_GrayCodeSequence(P1NovelBase):
    """Traverse state pads in Gray code order (1 bit change per step)"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng88_graycodesequence", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng88_graycodesequence", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG89_BinaryCounterRegister(P1NovelBase):
    """Increment 4-bit binary counter by pushing clock pulse"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng89_binarycounterregister", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng89_binarycounterregister", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG90_GraphColoring3Nodes(P1NovelBase):
    """Color 3 adjacent nodes with 3 distinct non-conflicting colors"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng90_graphcoloring3nodes", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng90_graphcoloring3nodes", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG91_TuringTapeMovement(P1NovelBase):
    """Read symbol, write symbol, and transition state head"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng91_turingtapemovement", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng91_turingtapemovement", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG92_TruthTableInduction(P1NovelBase):
    """Induce 2-input boolean operator from truth table display"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng92_truthtableinduction", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng92_truthtableinduction", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG93_FiniteStateMachine(P1NovelBase):
    """Follow state transition diagram to accept string pattern"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng93_finitestatemachine", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng93_finitestatemachine", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG94_CellularSheafConsistency(P1NovelBase):
    """Harmonize boundary values across overlapping patches"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng94_cellularsheafconsistency", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng94_cellularsheafconsistency", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG95_FibonacciSequenceRungs(P1NovelBase):
    """Step on stones whose numbers follow Fibonacci sequence"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng95_fibonaccisequencerungs", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng95_fibonaccisequencerungs", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG96_PrimeNumberFilter(P1NovelBase):
    """Only prime-indexed tiles are safe to step on"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng96_primenumberfilter", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng96_primenumberfilter", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG97_ModuloClockArithmetic(P1NovelBase):
    """Rotate 12-hour clock face to match target modular residue"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng97_moduloclockarithmetic", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng97_moduloclockarithmetic", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG98_MirrorInversionMatrix(P1NovelBase):
    """Reflect input matrix along diagonal axis"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng98_mirrorinversionmatrix", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng98_mirrorinversionmatrix", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG99_KarnaughMapMinimizer(P1NovelBase):
    """Identify minimal boolean cover on 4-cell map"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng99_karnaughmapminimizer", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng99_karnaughmapminimizer", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG100_GrandLogicChallenger(P1NovelBase):
    """Composite boolean logic network with XOR, AND, and parity"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 55, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng100_grandlogicchallenger", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng100_grandlogicchallenger", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        sw = self.get_sprite("switch")
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

        if sw and self.check_collision(av, sw):
            self.state_val ^= 1

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()
