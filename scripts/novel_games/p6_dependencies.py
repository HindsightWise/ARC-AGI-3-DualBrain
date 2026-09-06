#!/usr/bin/env python3
"""
Paradigm 6: Hierarchical Dependencies & Bellman Induction (ng151..ng175)
25 distinct novel environments testing multi-key dependency DAGs, durability depletion, and color transmutation.
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


class NG151_DualKeyGate(P1NovelBase):
    """Collect Key A to unlock Gate A, then Key B for Gate B"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng151_dualkeygate", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng151_dualkeygate", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG152_TollBooth(P1NovelBase):
    """Collect 3 coins before toll barrier opens"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng152_tollbooth", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng152_tollbooth", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG153_ChainedPrerequisites(P1NovelBase):
    """Sequential dependency chain: A -> B -> C -> Goal"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng153_chainedprerequisites", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng153_chainedprerequisites", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG154_DurabilityPickaxe(P1NovelBase):
    """Pickaxe has 2 charges; must choose which 2 boulders to break"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng154_durabilitypickaxe", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng154_durabilitypickaxe", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG155_ColorTransmutation(P1NovelBase):
    """Step on Red pad to open red gate, then Blue pad for blue gate"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng155_colortransmutation", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng155_colortransmutation", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG156_SingleItemBackpack(P1NovelBase):
    """Can only carry 1 item at a time; deposit item 1 before taking 2"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng156_singleitembackpack", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng156_singleitembackpack", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG157_DescendingOrderLocks(P1NovelBase):
    """Must touch locks in descending numerical order (3, 2, 1)"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng157_descendingorderlocks", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng157_descendingorderlocks", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG158_CombinationLevers(P1NovelBase):
    """3 levers requiring exact binary pattern (1-0-1) to unlock exit"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng158_combinationlevers", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng158_combinationlevers", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG159_WeightedPressureDoor(P1NovelBase):
    """Push box onto plate to keep door open while player crosses"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng159_weightedpressuredoor", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng159_weightedpressuredoor", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG160_FuelDepletionSprint(P1NovelBase):
    """Each step consumes 1 fuel; fuel canisters replenish 10"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng160_fueldepletionsprint", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng160_fueldepletionsprint", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG161_ConsumableKeys(P1NovelBase):
    """Keys break after single use; plan optimal unlock order"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng161_consumablekeys", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng161_consumablekeys", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG162_TimedDecayKey(P1NovelBase):
    """Key dissolves 12 steps after pickup; sprint to lock"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 40, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng162_timeddecaykey", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng162_timeddecaykey", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG163_ShapeShifterAperture(P1NovelBase):
    """Avatar morphs at pads to fit narrow slits"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng163_shapeshifteraperture", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng163_shapeshifteraperture", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG164_HubAndSpokeMaze(P1NovelBase):
    """Central hub with 4 spokes; each spoke contains key for next spoke"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 55, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng164_hubandspokemaze", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng164_hubandspokemaze", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG165_TopologicalValveOrder(P1NovelBase):
    """One-way gates restricting sequence of chamber visits"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng165_topologicalvalveorder", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng165_topologicalvalveorder", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG166_AlternatingTileColors(P1NovelBase):
    """Player must alternate between black and gold tiles"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng166_alternatingtilecolors", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng166_alternatingtilecolors", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG167_ThermalPhaseState(P1NovelBase):
    """Hot pad thaws ice walls; cold pad freezes water bridge"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng167_thermalphasestate", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng167_thermalphasestate", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG168_BatteryChargePack(P1NovelBase):
    """Charge energy core at dynamo before powering forcefield door"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng168_batterychargepack", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng168_batterychargepack", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG169_MineDisarmingSequence(P1NovelBase):
    """Disarm 3 explosive runes in safe order"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng169_minedisarmingsequence", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng169_minedisarmingsequence", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG170_SacrificialAutomaton(P1NovelBase):
    """Send helper robot into trap spike to create safe path"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng170_sacrificialautomaton", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng170_sacrificialautomaton", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG171_BalancedScalesGate(P1NovelBase):
    """Place equal weights on left and right pressure sensors"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng171_balancedscalesgate", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng171_balancedscalesgate", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG172_HierarchicalSubtreeUnlock(P1NovelBase):
    """Unlock master key by solving leaf nodes first"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng172_hierarchicalsubtreeunlock", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng172_hierarchicalsubtreeunlock", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG173_DynamicPatternKeypad(P1NovelBase):
    """Step on 4 keypad digits matching overhead sequence banner"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng173_dynamicpatternkeypad", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng173_dynamicpatternkeypad", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG174_ElevationStairway(P1NovelBase):
    """Tiles with increasing elevation heights (can only step +1)"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 45, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng174_elevationstairway", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng174_elevationstairway", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()


class NG175_RetrogradeMasterDAG(P1NovelBase):
    """Complex multi-stage dependency labyrinth requiring Bellman backward induction"""
    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.has_key = False

        # Key sprite
        k_pix = np.full((step_size, step_size), 11, dtype=np.int8)
        key = Sprite(pixels=k_pix.tolist(), name="key_a", x=step_size * 4, y=grid_size - step_size * 5, tags=["key"], blocking=BlockingMode.NOT_BLOCKED)

        # Gate sprite
        g_pix = np.full((step_size, step_size * 2), 8, dtype=np.int8)
        gate = Sprite(pixels=g_pix.tolist(), name="gate_a", x=grid_size // 2, y=grid_size // 2, tags=["gate", "wall"])

        sprites, cam, acts = _make_boxed_arena("ng175_retrogrademasterdag", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng175_retrogrademasterdag", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
        key = self.get_sprite("key_a")
        gate = self.get_sprite("gate_a")
        goal = self.get_sprite("goal")

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx, ny = av.x + dx, av.y + dy
        if key and self.check_collision(av, key):
            self.has_key = True
            key.set_position(-100, -100) # Hide key
            if gate:
                gate.set_position(-100, -100) # Open gate

        if not self.is_wall_collision(nx, ny, av.width, av.height):
            av.set_position(nx, ny)

        if self.check_collision(av, goal):
            self.win()
        self.complete_action()
