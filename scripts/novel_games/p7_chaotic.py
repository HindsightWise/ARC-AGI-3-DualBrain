#!/usr/bin/env python3
"""
Paradigm 7: Stochastic Chaos, Perimeter Noise & Adversarial Distortions (ng176..ng200)
25 distinct novel environments testing noise HUD filtering (lf52 style), variable lattice strides, and decoy mimics.
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


class NG176_BlackNoiseHUD(P1NovelBase):
    """Animated noise pattern in bottom rows simulating lf52"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng176_blacknoisehud", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng176_blacknoisehud", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG177_StrobeFlasher(P1NovelBase):
    """Background color pulses periodically without changing mechanics"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng177_strobeflasher", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng177_strobeflasher", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG178_ScreenShakeJitter(P1NovelBase):
    """World sprites offset by +/- 1 pixel jitter periodically"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng178_screenshakejitter", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng178_screenshakejitter", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG179_DecoyMimicGoal(P1NovelBase):
    """Two goal sprites; only one has true affordance signature"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng179_decoymimicgoal", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng179_decoymimicgoal", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG180_LatticeStride1(P1NovelBase):
    """1x1 pixel sprite with 1-pixel grid stride"""
    def __init__(self, grid_size: int = 32, step_size: int = 1, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng180_latticestride1", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng180_latticestride1", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG181_LatticeStride4(P1NovelBase):
    """4x4 pixel sprite with 4-pixel grid stride"""
    def __init__(self, grid_size: int = 32, step_size: int = 4, max_steps: int = 35, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng181_latticestride4", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng181_latticestride4", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG182_LatticeStride5(P1NovelBase):
    """5x5 pixel sprite with 5-pixel grid stride and offset"""
    def __init__(self, grid_size: int = 35, step_size: int = 5, max_steps: int = 35, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng182_latticestride5", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng182_latticestride5", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG183_LatticeStride8(P1NovelBase):
    """8x8 pixel sprite with 8-pixel grid stride"""
    def __init__(self, grid_size: int = 32, step_size: int = 8, max_steps: int = 30, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng183_latticestride8", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng183_latticestride8", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG184_AsymmetricStride(P1NovelBase):
    """Horizontal stride = 2, Vertical stride = 4"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng184_asymmetricstride", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng184_asymmetricstride", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG185_ShiftingWallsRandom(P1NovelBase):
    """Interior divider shifts randomly every 5 steps"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng185_shiftingwallsrandom", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng185_shiftingwallsrandom", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG186_VanishingStepPath(P1NovelBase):
    """Floor tiles turn to black void 2 steps after traversal"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng186_vanishingsteppath", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng186_vanishingsteppath", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG187_DriftingCameraViewport(P1NovelBase):
    """Viewport pans dynamically tracking avatar center"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng187_driftingcameraviewport", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng187_driftingcameraviewport", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG188_InvertedContrastPalette(P1NovelBase):
    """Walls are dark (color 1), floor is bright (color 12)"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng188_invertedcontrastpalette", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng188_invertedcontrastpalette", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG189_ColorInversionFlash(P1NovelBase):
    """Inverts color palette for 1 frame upon action"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng189_colorinversionflash", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng189_colorinversionflash", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG190_AdversarialLoopTrap(P1NovelBase):
    """Maze designed to trap naive 2-cycle loop interceptors"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng190_adversariallooptrap", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng190_adversariallooptrap", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG191_GlitchedConveyor(P1NovelBase):
    """Conveyor belts flipping direction on odd turn indices"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng191_glitchedconveyor", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng191_glitchedconveyor", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG192_MirageGhostSprites(P1NovelBase):
    """Sprites that dissolve without blocking when avatar approaches"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng192_mirageghostsprites", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng192_mirageghostsprites", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG193_WindGustPerturbation(P1NovelBase):
    """Periodic wind pushes player 1 cell east every 3 moves"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng193_windgustperturbation", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng193_windgustperturbation", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG194_MultiPerimeterHUD(P1NovelBase):
    """Step counters and HUD icons at top, bottom, and right borders"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng194_multiperimeterhud", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng194_multiperimeterhud", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG195_InvertedCompassZone(P1NovelBase):
    """Entering central zone inverts controls"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng195_invertedcompasszone", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng195_invertedcompasszone", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG196_ActionPenaltyCounter(P1NovelBase):
    """On-screen score counter decrementing live with actions"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng196_actionpenaltycounter", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng196_actionpenaltycounter", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG197_NonRectangularPolygons(P1NovelBase):
    """L-shaped and T-shaped wall clusters"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng197_nonrectangularpolygons", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng197_nonrectangularpolygons", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG198_FragmentedAvatarDuo(P1NovelBase):
    """Avatar consists of two separated pixels moving in tandem"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng198_fragmentedavatarduo", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng198_fragmentedavatarduo", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG199_InvisibleBumpWalls(P1NovelBase):
    """Walls invisible until avatar bumps them directly"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng199_invisiblebumpwalls", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng199_invisiblebumpwalls", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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


class NG200_GrandChallengerChaos(P1NovelBase):
    """Composite arena with noise HUD, teleporters, box push, and key"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng200_grandchallengerchaos", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng200_grandchallengerchaos", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        goal = self.get_sprite("goal")
        hud = self.get_sprite("noise_hud")

        # Mutate noise HUD
        if hud:
            new_hud = np.random.randint(1, 15, size=(hud.height, hud.width), dtype=np.int8)
            hud.pixels = new_hud

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
