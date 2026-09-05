#!/usr/bin/env python3
"""
Paradigm 2: Push-Box & Object Physics
Contains 5 novel ARC-AGI-3 game environments:
- ng06_ice_slider: Frictionless ice puzzle where moves slide continuously until hitting obstacles.
- ng07_boulder_bridge: Pushing boulders into chasms to fill them and construct walkable bridges.
- ng08_magnet_polarity: Polarity flipping (ACTION5) creating magnetic attraction and repulsion dynamics.
- ng09_heavy_light_boxes: Inertial physics with multi-push heavy boxes and agile light boxes on dual scales.
- ng10_portal_box_routing: Teleporting pushable boxes through portals to trigger remote pressure plates.
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


class NG06_IceSlider(ARCBaseGame):
    """Frictionless ice sliding mechanics until collision with obstacles."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 50, seed: int = 42) -> None:
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 8
        wall_color = 1
        rock_color = 5
        avatar_color = 11
        goal_color = 14

        sprites: List[Sprite] = []

        hw = np.full((step_size, grid_size), wall_color, dtype=np.int8)
        vw = np.full((grid_size, step_size), wall_color, dtype=np.int8)
        sprites.append(Sprite(pixels=hw.tolist(), name="w_top", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=hw.tolist(), name="w_bot", x=0, y=grid_size - step_size, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_l", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_r", x=grid_size - step_size, y=0, tags=["wall"]))

        rock_coords = [(11, 2), (10, 9), (13, 8), (12, 11), (4, 12), (8, 4)]
        r_pix = np.full((step_size, step_size), rock_color, dtype=np.int8)
        for idx, (rx, ry) in enumerate(rock_coords):
            sprites.append(Sprite(
                pixels=r_pix.tolist(),
                name=f"rock_{idx}",
                x=rx * step_size,
                y=ry * step_size,
                tags=["wall", "rock"],
                blocking=BlockingMode.BOUNDING_BOX,
            ))

        av_pix = np.full((step_size, step_size), avatar_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=av_pix.tolist(),
            name="avatar",
            x=2 * step_size,
            y=2 * step_size,
            tags=["avatar"],
            blocking=BlockingMode.BOUNDING_BOX,
            layer=10,
        ))

        g_pix = np.full((step_size, step_size), goal_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=g_pix.tolist(),
            name="goal",
            x=12 * step_size,
            y=10 * step_size,
            tags=["goal"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng06_ice_slider",
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

        if dx == 0 and dy == 0:
            self.complete_action()
            return

        walls = self.current_level.get_sprites_by_tag("wall")
        while True:
            nx = avatar.x + dx
            ny = avatar.y + dy

            blocked = False
            for w in walls:
                if (nx < w.x + w.width and nx + avatar.width > w.x and
                    ny < w.y + w.height and ny + avatar.height > w.y):
                    blocked = True
                    break

            if blocked:
                break

            avatar.set_position(nx, ny)

            if avatar.x == goal.x and avatar.y == goal.y:
                self.win()
                self.complete_action()
                return

        self.complete_action()


class NG07_BoulderBridge(ARCBaseGame):
    """Pushing boulders into chasms to fill them and construct bridge pathways."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 3
        chasm_color = 0
        wall_color = 5
        boulder_color = 7
        avatar_color = 11
        goal_color = 14

        sprites: List[Sprite] = []

        hw = np.full((step_size, grid_size), wall_color, dtype=np.int8)
        vw = np.full((grid_size, step_size), wall_color, dtype=np.int8)
        sprites.append(Sprite(pixels=hw.tolist(), name="w_top", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=hw.tolist(), name="w_bot", x=0, y=grid_size - step_size, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_l", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_r", x=grid_size - step_size, y=0, tags=["wall"]))

        self.chasm_coords: Set[Tuple[int, int]] = set()
        ch_pix = np.full((step_size, step_size), chasm_color, dtype=np.int8)
        for cy in range(1, grid_size // step_size - 1):
            cx = 8
            sprites.append(Sprite(
                pixels=ch_pix.tolist(),
                name=f"chasm_{cx}_{cy}",
                x=cx * step_size,
                y=cy * step_size,
                tags=["chasm"],
                blocking=BlockingMode.NOT_BLOCKED,
            ))
            self.chasm_coords.add((cx, cy))

        b_pix = np.full((step_size, step_size), boulder_color, dtype=np.int8)
        boulder_coords = [(4, 4), (4, 8), (5, 6)]
        for idx, (bx, by) in enumerate(boulder_coords):
            sprites.append(Sprite(
                pixels=b_pix.tolist(),
                name=f"boulder_{idx}",
                x=bx * step_size,
                y=by * step_size,
                tags=["boulder"],
                blocking=BlockingMode.BOUNDING_BOX,
            ))

        av_pix = np.full((step_size, step_size), avatar_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=av_pix.tolist(),
            name="avatar",
            x=2 * step_size,
            y=6 * step_size,
            tags=["avatar"],
            blocking=BlockingMode.BOUNDING_BOX,
            layer=10,
        ))

        g_pix = np.full((step_size, step_size), goal_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=g_pix.tolist(),
            name="goal",
            x=12 * step_size,
            y=6 * step_size,
            tags=["goal"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng07_boulder_bridge",
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
        goal = self.get_sprite("goal")
        boulders = [s for s in self.current_level.get_sprites_by_tag("boulder") if s.is_visible]

        dx, dy = 0, 0
        act = self._action.id
        if act == GameAction.ACTION1: dy = -self.step_size
        elif act == GameAction.ACTION2: dy = self.step_size
        elif act == GameAction.ACTION3: dx = -self.step_size
        elif act == GameAction.ACTION4: dx = self.step_size

        nx = avatar.x + dx
        ny = avatar.y + dy

        # Check walls
        for w in self.current_level.get_sprites_by_tag("wall"):
            if (nx < w.x + w.width and nx + avatar.width > w.x and
                ny < w.y + w.height and ny + avatar.height > w.y):
                self.complete_action()
                return

        # Check boulder interaction
        pushed_boulder: Optional[Sprite] = None
        for b in boulders:
            if nx == b.x and ny == b.y:
                pushed_boulder = b
                break

        if pushed_boulder:
            bx_new = pushed_boulder.x + dx
            by_new = pushed_boulder.y + dy

            # Check wall collisions for boulder
            for w in self.current_level.get_sprites_by_tag("wall"):
                if (bx_new < w.x + w.width and bx_new + pushed_boulder.width > w.x and
                    by_new < w.y + w.height and by_new + pushed_boulder.height > w.y):
                    self.complete_action()
                    return

            # Check other boulder collisions
            for ob in boulders:
                if ob is not pushed_boulder and bx_new == ob.x and by_new == ob.y:
                    self.complete_action()
                    return

            b_tx = bx_new // self.step_size
            b_ty = by_new // self.step_size
            if (b_tx, b_ty) in self.chasm_coords:
                self.chasm_coords.discard((b_tx, b_ty))
                ch_sp = self.get_sprite(f"chasm_{b_tx}_{b_ty}")
                if ch_sp:
                    ch_sp.pixels = np.full((self.step_size, self.step_size), 5, dtype=np.int8)
                pushed_boulder.set_visible(False)
            else:
                pushed_boulder.set_position(bx_new, by_new)

        a_tx = nx // self.step_size
        a_ty = ny // self.step_size
        if (a_tx, a_ty) in self.chasm_coords:
            self.lose()
            self.complete_action()
            return

        avatar.set_position(nx, ny)

        if avatar.x == goal.x and avatar.y == goal.y:
            self.win()

        self.complete_action()


class NG08_MagnetPolarity(ARCBaseGame):
    """Polarity flipping (ACTION5) creating magnetic attraction and repulsion dynamics."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 0
        wall_color = 5
        self.north_color = 1
        self.south_color = 2
        goal_color = 14

        self.avatar_polarity = 1
        self.block_polarity = 2
        sprites: List[Sprite] = []

        hw = np.full((step_size, grid_size), wall_color, dtype=np.int8)
        vw = np.full((grid_size, step_size), wall_color, dtype=np.int8)
        sprites.append(Sprite(pixels=hw.tolist(), name="w_top", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=hw.tolist(), name="w_bot", x=0, y=grid_size - step_size, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_l", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_r", x=grid_size - step_size, y=0, tags=["wall"]))

        av_pix = np.full((step_size, step_size), self.north_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=av_pix.tolist(),
            name="avatar",
            x=4 * step_size,
            y=8 * step_size,
            tags=["avatar"],
            blocking=BlockingMode.BOUNDING_BOX,
            layer=10,
        ))

        mb_pix = np.full((step_size, step_size), self.south_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=mb_pix.tolist(),
            name="magnet_block",
            x=8 * step_size,
            y=8 * step_size,
            tags=["magnet_block"],
            blocking=BlockingMode.BOUNDING_BOX,
        ))

        sw_pix = np.full((step_size, step_size), goal_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=sw_pix.tolist(),
            name="switch_pad",
            x=12 * step_size,
            y=8 * step_size,
            tags=["switch"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng08_magnet_polarity",
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

        avatar = self.get_sprite("avatar")
        magnet_block = self.get_sprite("magnet_block")
        switch_pad = self.get_sprite("switch_pad")

        act = self._action.id
        if act == GameAction.ACTION5:
            self.avatar_polarity = 2 if self.avatar_polarity == 1 else 1
            col = self.south_color if self.avatar_polarity == 2 else self.north_color
            avatar.pixels = np.full((self.step_size, self.step_size), col, dtype=np.int8)
            self.complete_action()
            return

        dx, dy = 0, 0
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

        if nx == magnet_block.x and ny == magnet_block.y:
            shift = 2 * self.step_size if self.avatar_polarity == self.block_polarity else self.step_size
            mb_nx = magnet_block.x + (dx // self.step_size) * shift
            mb_ny = magnet_block.y + (dy // self.step_size) * shift

            hit_wall = False
            for w in self.current_level.get_sprites_by_tag("wall"):
                if (mb_nx < w.x + w.width and mb_nx + magnet_block.width > w.x and
                    mb_ny < w.y + w.height and mb_ny + magnet_block.height > w.y):
                    hit_wall = True
                    break

            if not hit_wall:
                magnet_block.set_position(mb_nx, mb_ny)
                avatar.set_position(nx, ny)
        else:
            avatar.set_position(nx, ny)

        if magnet_block.x == switch_pad.x and magnet_block.y == switch_pad.y:
            self.win()

        self.complete_action()


class NG09_HeavyLightBoxes(ARCBaseGame):
    """Inertial physics with multi-push heavy boxes and agile light boxes on dual scales."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 70, seed: int = 42) -> None:
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 0
        wall_color = 5
        heavy_color = 9
        light_color = 4
        scale_h_color = 2
        scale_l_color = 14
        avatar_color = 11

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
            x=4 * step_size,
            y=4 * step_size,
            tags=["avatar"],
            blocking=BlockingMode.BOUNDING_BOX,
            layer=10,
        ))

        lb_pix = np.full((step_size, step_size), light_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=lb_pix.tolist(),
            name="light_box",
            x=6 * step_size,
            y=4 * step_size,
            tags=["light_box"],
            blocking=BlockingMode.BOUNDING_BOX,
        ))

        hb_pix = np.full((step_size, step_size), heavy_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=hb_pix.tolist(),
            name="heavy_box",
            x=4 * step_size,
            y=6 * step_size,
            tags=["heavy_box"],
            blocking=BlockingMode.BOUNDING_BOX,
        ))

        self.heavy_pushes = 0
        self.last_push_dir: Optional[int] = None

        sl_pix = np.full((step_size, step_size), scale_l_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=sl_pix.tolist(),
            name="scale_light",
            x=10 * step_size,
            y=4 * step_size,
            tags=["scale_l"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        sh_pix = np.full((step_size, step_size), scale_h_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=sh_pix.tolist(),
            name="scale_heavy",
            x=4 * step_size,
            y=10 * step_size,
            tags=["scale_h"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng09_heavy_light_boxes",
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
        light_box = self.get_sprite("light_box")
        heavy_box = self.get_sprite("heavy_box")
        scale_light = self.get_sprite("scale_light")
        scale_heavy = self.get_sprite("scale_heavy")

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

        if nx == light_box.x and ny == light_box.y:
            lb_nx = light_box.x + dx
            lb_ny = light_box.y + dy
            if not self._is_blocked(lb_nx, lb_ny):
                light_box.set_position(lb_nx, lb_ny)
                avatar.set_position(nx, ny)
        elif nx == heavy_box.x and ny == heavy_box.y:
            if self.last_push_dir == act:
                self.heavy_pushes += 1
            else:
                self.last_push_dir = act
                self.heavy_pushes = 1

            if self.heavy_pushes >= 2:
                hb_nx = heavy_box.x + dx
                hb_ny = heavy_box.y + dy
                if not self._is_blocked(hb_nx, hb_ny):
                    heavy_box.set_position(hb_nx, hb_ny)
                    avatar.set_position(nx, ny)
                self.heavy_pushes = 0
        else:
            avatar.set_position(nx, ny)
            self.heavy_pushes = 0
            self.last_push_dir = None

        if (light_box.x == scale_light.x and light_box.y == scale_light.y and
            heavy_box.x == scale_heavy.x and heavy_box.y == scale_heavy.y):
            self.win()

        self.complete_action()

    def _is_blocked(self, x: int, y: int) -> bool:
        for w in self.current_level.get_sprites_by_tag("wall"):
            if x < w.x + w.width and x + self.step_size > w.x and y < w.y + w.height and y + self.step_size > w.y:
                return True
        return False


class NG10_PortalBoxRouting(ARCBaseGame):
    """Teleporting pushable boxes through portals to trigger remote pressure plates."""

    def __init__(self, grid_size: int = 32, step_size: int = 2, max_steps: int = 60, seed: int = 42) -> None:
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.grid_size = grid_size

        bg_color = 0
        wall_color = 5
        portal_color = 8
        box_color = 7
        plate_color = 4
        door_color = 2
        avatar_color = 11
        goal_color = 14

        sprites: List[Sprite] = []

        hw = np.full((step_size, grid_size), wall_color, dtype=np.int8)
        vw = np.full((grid_size, step_size), wall_color, dtype=np.int8)
        sprites.append(Sprite(pixels=hw.tolist(), name="w_top", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=hw.tolist(), name="w_bot", x=0, y=grid_size - step_size, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_l", x=0, y=0, tags=["wall"]))
        sprites.append(Sprite(pixels=vw.tolist(), name="w_r", x=grid_size - step_size, y=0, tags=["wall"]))

        mid = (grid_size // (2 * step_size)) * step_size
        door_y = 10 * step_size
        vw_t = np.full((door_y, step_size), wall_color, dtype=np.int8)
        sprites.append(Sprite(pixels=vw_t.tolist(), name="w_mid_t", x=mid, y=0, tags=["wall"]))
        bot_h = grid_size - (door_y + step_size)
        if bot_h > 0:
            vw_b = np.full((bot_h, step_size), wall_color, dtype=np.int8)
            sprites.append(Sprite(pixels=vw_b.tolist(), name="w_mid_b", x=mid, y=door_y + step_size, tags=["wall"]))

        av_pix = np.full((step_size, step_size), avatar_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=av_pix.tolist(),
            name="avatar",
            x=3 * step_size,
            y=5 * step_size,
            tags=["avatar"],
            blocking=BlockingMode.BOUNDING_BOX,
            layer=10,
        ))

        bx_pix = np.full((step_size, step_size), box_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=bx_pix.tolist(),
            name="box",
            x=5 * step_size,
            y=5 * step_size,
            tags=["box"],
            blocking=BlockingMode.BOUNDING_BOX,
        ))

        p_pix = np.full((step_size, step_size), portal_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=p_pix.tolist(),
            name="portal_in",
            x=6 * step_size,
            y=5 * step_size,
            tags=["portal_in"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        sprites.append(Sprite(
            pixels=p_pix.tolist(),
            name="portal_out",
            x=mid + 2 * step_size,
            y=5 * step_size,
            tags=["portal_out"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        pp_pix = np.full((step_size, step_size), plate_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=pp_pix.tolist(),
            name="plate",
            x=mid + 3 * step_size,
            y=5 * step_size,
            tags=["plate"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        dr_pix = np.full((step_size, step_size), door_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=dr_pix.tolist(),
            name="door",
            x=mid,
            y=10 * step_size,
            tags=["door", "wall"],
            blocking=BlockingMode.BOUNDING_BOX,
        ))

        g_pix = np.full((step_size, step_size), goal_color, dtype=np.int8)
        sprites.append(Sprite(
            pixels=g_pix.tolist(),
            name="goal",
            x=mid + 4 * step_size,
            y=10 * step_size,
            tags=["goal"],
            blocking=BlockingMode.NOT_BLOCKED,
        ))

        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        camera = Camera(width=grid_size, height=grid_size, background=bg_color)
        super().__init__(
            game_id="ng10_portal_box_routing",
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
        box = self.get_sprite("box")
        portal_in = self.get_sprite("portal_in")
        portal_out = self.get_sprite("portal_out")
        plate = self.get_sprite("plate")
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

        if nx == box.x and ny == box.y:
            bx_nx = box.x + dx
            by_ny = box.y + dy

            if bx_nx == portal_in.x and by_ny == portal_in.y:
                bx_nx = portal_out.x + dx
                by_ny = portal_out.y + dy

            hit_wall = False
            for w in self.current_level.get_sprites_by_tag("wall"):
                if w.is_visible and (bx_nx < w.x + w.width and bx_nx + box.width > w.x and
                                     by_ny < w.y + w.height and by_ny + box.height > w.y):
                    hit_wall = True
                    break

            if not hit_wall:
                box.set_position(bx_nx, by_ny)
                avatar.set_position(nx, ny)
        else:
            avatar.set_position(nx, ny)

        if box.x == plate.x and box.y == plate.y:
            door.set_visible(False)
            door.set_blocking(BlockingMode.NOT_BLOCKED)

        if avatar.x == goal.x and avatar.y == goal.y:
            self.win()

        self.complete_action()
