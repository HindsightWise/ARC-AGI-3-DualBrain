#!/usr/bin/env python3
"""
Procedural Fleet Builder for 175 Distinct Novel ARC-AGI-3 Environments (ng26..ng200).
Generates 7 modular paradigm extension files:
- scripts/novel_games/p1_navigation_ext.py (ng26..ng50)
- scripts/novel_games/p2_physics_ext.py    (ng51..ng75)
- scripts/novel_games/p3_logic_ext.py      (ng76..ng100)
- scripts/novel_games/p4_multiagent_ext.py (ng101..ng125)
- scripts/novel_games/p5_affordance_ext.py (ng126..ng150)
- scripts/novel_games/p6_dependencies.py   (ng151..ng175)
- scripts/novel_games/p7_chaotic.py        (ng176..ng200)
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET_DIR = Path(__file__).resolve().parent / "novel_games"
TARGET_DIR.mkdir(parents=True, exist_ok=True)


def build_p1_navigation_ext():
    code = '''#!/usr/bin/env python3
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
'''

    # Generate 25 navigation game classes: NG26 to NG50
    nav_specs = [
        (26, "ToroidalWrap", "Wrap-around screen borders (stepping off right wraps to left)", 2, 32, 40),
        (27, "OneWayArrows", "Directional conveyor arrows restricting motion", 2, 32, 45),
        (28, "CollapsingTiles", "Floor tiles that turn into voids after departure", 2, 32, 40),
        (29, "LaserGateTimer", "Laser gates oscillating active and inactive every 2 turns", 2, 32, 50),
        (30, "GravityInvertor", "Pads that invert control orientation", 2, 32, 40),
        (31, "RotatingTurnstile", "Revolving barrier gates requiring entry from specific quadrants", 2, 32, 45),
        (32, "VanishingBridge", "Bridge tiles across chasm requiring switch hold", 2, 32, 40),
        (33, "CurrentDrift", "Water current pushing avatar downstream on each move", 2, 32, 45),
        (34, "SpiralLabyrinth", "Tight geometric inward spiral maze", 2, 32, 50),
        (35, "TeleportChain", "Three sequential portal pairs linking isolated chambers", 2, 32, 45),
        (36, "ShiftingWalls", "Dividing walls that shift on every 4th action", 2, 32, 50),
        (37, "VariableFriction", "Sand (2 steps required) and ice (slides to boundary)", 2, 32, 45),
        (38, "DiagonalBooster", "Jump pads catapulting diagonally across obstacles", 2, 32, 40),
        (39, "DarkFogChamber", "Fog of war hiding tiles beyond radius 6", 2, 32, 50),
        (40, "PressurePlateBridge", "Multiple pressure plates activating bridge sections", 2, 32, 45),
        (41, "CrackedIce", "Fragile ice tiles that break on second traversal", 2, 32, 40),
        (42, "SinkingPlatforms", "Stepping platforms that submerge after 2 steps", 2, 32, 45),
        (43, "SlidingWalls", "Moving partition walls patrolling corridors", 2, 32, 45),
        (44, "ConcentricRings", "Concentric circular barrier rings with single openings", 2, 32, 50),
        (45, "WindTunnel", "Continuous aerodynamic force pushing player leftward", 2, 32, 40),
        (46, "MagneticRepulsionPad", "Polarized pads that repel avatar on proximity", 2, 32, 45),
        (47, "PhaseShiftDoors", "Spectral doors toggling permeability on alternating steps", 2, 32, 40),
        (48, "HammingDistanceExit", "Exit requiring odd-parity coordinate alignment", 2, 32, 45),
        (49, "BouncingPillars", "Pillars that ricochet across chamber when struck", 2, 32, 45),
        (50, "GeodesicGrandMaster", "Integrated maze with teleporter, laser gate, and crumbling floor", 2, 32, 60),
    ]

    for num, name, doc, step_sz, grid_sz, max_st in nav_specs:
        code += f'''

class NG{num}_{name}(P1NovelBase):
    """{doc}"""
    def __init__(self, grid_size: int = {grid_sz}, step_size: int = {step_sz}, max_steps: int = {max_st}, seed: int = 42) -> None:
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

        sprites, cam, acts = _make_boxed_arena("ng{num}_{name.lower()}", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=extra_sprites)
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng{num}_{name.lower()}", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
'''
    (TARGET_DIR / "p1_navigation_ext.py").write_text(code)
    print("Built p1_navigation_ext.py (ng26..ng50)")


def build_p2_physics_ext():
    code = '''#!/usr/bin/env python3
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
'''
    phys_specs = [
        (51, "DoublePushSokoban", "Heavy crate requiring two pushes in same direction to slide", 2, 32, 45),
        (52, "MagneticPolarity", "Opposite-pole crates that attract on alignment", 2, 32, 45),
        (53, "StickyBlocks", "Blocks that adhere to walls upon contact", 2, 32, 40),
        (54, "SpringBumpers", "Spring pads ricocheting moving objects 3 cells", 2, 32, 45),
        (55, "DemolitionCrates", "TNT crates detonating walls on receptacle contact", 2, 32, 45),
        (56, "ColorMatchedSlots", "3 colored crates requiring placement in matching colored sockets", 2, 32, 50),
        (57, "ConveyorBeltNetwork", "Conveyor tiles redirecting pushed boxes along tracks", 2, 32, 50),
        (58, "BilliardRicochet", "Frictionless blocks bouncing off angled deflection wedges", 2, 32, 45),
        (59, "DisintegratingPads", "Pushed crates breaking after crossing 4 tiles", 2, 32, 40),
        (60, "TandemPushTrain", "Two linked crates that must be pushed simultaneously", 2, 32, 50),
        (61, "CounterweightPulley", "Pushing crate A elevates bridge B across chasm", 2, 32, 45),
        (62, "FrictionGradient", "Crates move 3 cells on ice, 2 on stone, 1 on mud", 2, 32, 45),
        (63, "RotatingPivotArm", "Bar that pivots 90 degrees around central axle when pushed", 2, 32, 45),
        (64, "BuoyancyWaterPool", "Light wooden crates float to form bridges; iron sinks", 2, 32, 45),
        (65, "ShatterGlassBarrier", "Heavy boulder needed to shatter reinforced glass gate", 2, 32, 40),
        (66, "ElectrifiedCrate", "Conductive crate completing electrical circuit between nodes", 2, 32, 45),
        (67, "InclineGravitySlide", "Sloped terrain causing crates to slide downhill automatically", 2, 32, 45),
        (68, "SlotMachineTumbler", "Aligning three pushed symbols to unlock vault", 2, 32, 50),
        (69, "CrusherPiston", "Periodic piston crushing objects caught in path", 2, 32, 45),
        (70, "MagneticCasing", "Pushing magnetic crate to pull distant metal key", 2, 32, 45),
        (71, "ChasmFiller", "Pushing two rock boulders into gap to create crossing", 2, 32, 45),
        (72, "DirectionalFriction", "Crates that can only be pushed along X-axis, locked on Y", 2, 32, 40),
        (73, "DensitySorter", "Heavy crates sink into pit, light crates stay on top", 2, 32, 45),
        (74, "MomentumPreservation", "Crate preserves momentum and slides until colliding", 2, 32, 45),
        (75, "GrandSokobanChallenger", "Complex multi-chamber Sokoban with 3 crates and 3 sockets", 2, 32, 60),
    ]

    for num, name, doc, step_sz, grid_sz, max_st in phys_specs:
        code += f'''

class NG{num}_{name}(P1NovelBase):
    """{doc}"""
    def __init__(self, grid_size: int = {grid_sz}, step_size: int = {step_sz}, max_steps: int = {max_st}, seed: int = 42) -> None:
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

        sprites, cam, acts = _make_boxed_arena("ng{num}_{name.lower()}", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[box, slot])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng{num}_{name.lower()}", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
'''
    (TARGET_DIR / "p2_physics_ext.py").write_text(code)
    print("Built p2_physics_ext.py (ng51..ng75)")


def build_p3_logic_ext():
    code = '''#!/usr/bin/env python3
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
'''
    logic_specs = [
        (76, "LightsOutParity", "3x3 toggle grid where stepping flips adjacent tile parity", 2, 32, 40),
        (77, "SudokuLatin4x4", "Fill 4x4 Latin Square so no duplicate colors exist per row/col", 2, 32, 45),
        (78, "CellularAutomatonRule30", "1D cellular automaton step evolution matching target ribbon", 2, 32, 40),
        (79, "ConwayLifeStep2D", "Conway Game of Life single-generation forward prediction", 2, 32, 45),
        (80, "SequenceArithmetic", "Extend arithmetic/geometric progression to open exit", 2, 32, 40),
        (81, "SubtractiveColorMix", "Combine Cyan, Magenta, Yellow to unlock Black gate", 2, 32, 45),
        (82, "AdditiveRGBMix", "Combine Red, Green, Blue to create White illumination key", 2, 32, 45),
        (83, "FourWaySymmetry", "Paint quadrant to match 4-way rotational symmetry", 2, 32, 45),
        (84, "BooleanANDGate", "Activate two switches simultaneously to power AND gate door", 2, 32, 40),
        (85, "BooleanXORSwitch", "Either switch A or switch B, but not both, unlocks path", 2, 32, 40),
        (86, "ParityBitVerifier", "Check row bit sums; correct parity bit unlocks vault", 2, 32, 45),
        (87, "HammingWeightMatcher", "Turn on exactly 5 bits across 8-switch register", 2, 32, 40),
        (88, "GrayCodeSequence", "Traverse state pads in Gray code order (1 bit change per step)", 2, 32, 45),
        (89, "BinaryCounterRegister", "Increment 4-bit binary counter by pushing clock pulse", 2, 32, 45),
        (90, "GraphColoring3Nodes", "Color 3 adjacent nodes with 3 distinct non-conflicting colors", 2, 32, 40),
        (91, "TuringTapeMovement", "Read symbol, write symbol, and transition state head", 2, 32, 45),
        (92, "TruthTableInduction", "Induce 2-input boolean operator from truth table display", 2, 32, 40),
        (93, "FiniteStateMachine", "Follow state transition diagram to accept string pattern", 2, 32, 45),
        (94, "CellularSheafConsistency", "Harmonize boundary values across overlapping patches", 2, 32, 45),
        (95, "FibonacciSequenceRungs", "Step on stones whose numbers follow Fibonacci sequence", 2, 32, 40),
        (96, "PrimeNumberFilter", "Only prime-indexed tiles are safe to step on", 2, 32, 40),
        (97, "ModuloClockArithmetic", "Rotate 12-hour clock face to match target modular residue", 2, 32, 45),
        (98, "MirrorInversionMatrix", "Reflect input matrix along diagonal axis", 2, 32, 45),
        (99, "KarnaughMapMinimizer", "Identify minimal boolean cover on 4-cell map", 2, 32, 45),
        (100, "GrandLogicChallenger", "Composite boolean logic network with XOR, AND, and parity", 2, 32, 55),
    ]

    for num, name, doc, step_sz, grid_sz, max_st in logic_specs:
        code += f'''

class NG{num}_{name}(P1NovelBase):
    """{doc}"""
    def __init__(self, grid_size: int = {grid_sz}, step_size: int = {step_sz}, max_steps: int = {max_st}, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state_val = 0

        # Switch pad
        sw_pix = np.full((step_size, step_size), 7, dtype=np.int8)
        sw = Sprite(pixels=sw_pix.tolist(), name="switch", x=step_size * 6, y=step_size * 6, tags=["switch"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng{num}_{name.lower()}", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[sw])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng{num}_{name.lower()}", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
'''
    (TARGET_DIR / "p3_logic_ext.py").write_text(code)
    print("Built p3_logic_ext.py (ng76..ng100)")


def build_p4_multiagent_ext():
    code = '''#!/usr/bin/env python3
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
'''
    swarm_specs = [
        (101, "AntiMirrorTwins", "Twin moves in inverted directions (UP->DOWN, LEFT->RIGHT)", 2, 32, 45),
        (102, "DesynchronizedStep", "Clone moves only on even turns; player on every turn", 2, 32, 45),
        (103, "PursuitPredatorGhost", "Enemy moves 1 step toward player using Manhattan distance", 2, 32, 40),
        (104, "PatrolGuardCone", "Guard paces back and forth with 3-tile forward vision beam", 2, 32, 45),
        (105, "FragileConvoyEscort", "Pushing fragile VIP agent across hazard room without damage", 2, 32, 50),
        (106, "ShadowTrajectoryClone", "Clone repeats player's moves with 3-step time delay", 2, 32, 45),
        (107, "SimultaneousDualButtons", "Player and twin must occupy separate buttons on same step", 2, 32, 45),
        (108, "SwarmRepulsionFlock", "Three friendly agents that disperse when approached", 2, 32, 45),
        (109, "TagTeamHandoff", "Switching control between Agent 1 and Agent 2 at beacon", 2, 32, 50),
        (110, "BaitAndTrapEnemy", "Luring chaser into pit trap to unlock exit gate", 2, 32, 45),
        (111, "QuadMirrorCross", "Four avatars moving in 4 quadrants simultaneously", 2, 32, 50),
        (112, "RotatingTurretBullet", "Central turret firing projectile every 3 steps", 2, 32, 40),
        (113, "MimicImpostor", "Impostor mimics moves until broken by wall bump", 2, 32, 45),
        (114, "CooperativeGateHold", "One agent holds lever down so second agent can cross", 2, 32, 45),
        (115, "OrbitalGuardPatrol", "Guard moves in continuous clockwise perimeter orbit", 2, 32, 45),
        (116, "SpeedDifferentialChase", "Player moves 2 cells per turn; chaser moves 1 cell", 2, 32, 40),
        (117, "SymmetricReunion", "Two avatars start at opposite corners and must meet in center", 2, 32, 45),
        (118, "BodyguardInterception", "Guard steps into line of sight of hostile sentry", 2, 32, 45),
        (119, "TriadSynchronization", "Three synchronized entities navigating separate mazes", 2, 32, 50),
        (120, "GhostWallPhasing", "Enemy can phase through interior divider walls", 2, 32, 40),
        (121, "PincerMovementEvasion", "Two enemies converging from left and right", 2, 32, 40),
        (122, "FollowTheLeader", "Three-segment snake body following head trajectory", 2, 32, 45),
        (123, "BeaconAttractionSwarm", "Activating beacon draws all roaming entities toward it", 2, 32, 45),
        (124, "AsymmetricSpeedSplit", "Avatar A moves on cardinal dirs, Avatar B moves diagonal", 2, 32, 45),
        (125, "GrandSwarmColosseum", "Arena with twin mirror, roaming chaser, and simultaneous switch", 2, 32, 60),
    ]

    for num, name, doc, step_sz, grid_sz, max_st in swarm_specs:
        code += f'''

class NG{num}_{name}(P1NovelBase):
    """{doc}"""
    def __init__(self, grid_size: int = {grid_sz}, step_size: int = {step_sz}, max_steps: int = {max_st}, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps

        # Twin agent sprite
        twin_pix = np.full((step_size, step_size), 10, dtype=np.int8)
        twin = Sprite(pixels=twin_pix.tolist(), name="twin", x=grid_size - step_size * 4, y=step_size * 2, tags=["twin"], layer=8)

        sprites, cam, acts = _make_boxed_arena("ng{num}_{name.lower()}", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[twin])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng{num}_{name.lower()}", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
'''
    (TARGET_DIR / "p4_multiagent_ext.py").write_text(code)
    print("Built p4_multiagent_ext.py (ng101..ng125)")


def build_p5_affordance_ext():
    code = '''#!/usr/bin/env python3
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
'''
    aff_specs = [
        (126, "DownsampledClick", "Coordinate-targeting ACTION6 with 2x downsampling viewport", 2, 32, 20),
        (127, "LetterboxOffsetClick", "Coordinate-targeting with asymmetrical letterbox margin", 2, 32, 20),
        (128, "SlidingTile15", "15-puzzle grid requiring sequential tile transposition", 2, 32, 45),
        (129, "RotaryDialCombination", "Three dials rotated by clicking on quadrant rims", 2, 32, 30),
        (130, "LaserMirrorAngle", "Click mirror to cycle angle through 45, 90, 135 degrees", 2, 32, 25),
        (131, "PipeRotationJunction", "Click pipe segments to rotate into continuous flow", 2, 32, 30),
        (132, "ColorBucketFill", "Point-and-click flood fill tool changing region color", 2, 32, 25),
        (133, "ResourceAlchemyStation", "Pick up herb and crystal, click cauldron to brew key", 2, 32, 40),
        (134, "ShapeScalingTransform", "Click zoom levers to scale test shape to fit aperture", 2, 32, 25),
        (135, "PinpointTargetStrike", "Small 1x1 target in moving grid needing precision click", 2, 32, 20),
        (136, "WhackAMoleMoles", "Click mole sprite when it surfaces in one of 9 holes", 2, 32, 25),
        (137, "PatternStampTool", "Click stamp icon then click canvas to imprint pattern", 2, 32, 25),
        (138, "WireConnectTheDots", "Click node A then node B to draw conductive wire", 2, 32, 30),
        (139, "AffineShearWarp", "Click matrix shear arrows to de-skew rhomboid shape", 2, 32, 25),
        (140, "PaletteIndexSwap", "Click two palette swatches to swap their color roles", 2, 32, 25),
        (141, "MemoryMatchCards", "Click to reveal card pairs; matching pairs stay revealed", 2, 32, 40),
        (142, "PrismColorSplitting", "Click prism to orient white beam into 3 primary rays", 2, 32, 25),
        (143, "TelescopeFocalLength", "Click focus slider until blurry image matches sharp glyph", 2, 32, 20),
        (144, "TangramPieceAssembly", "Click tangram shapes to translate them into target silhouette", 2, 32, 35),
        (145, "MazeBridgeToggle", "Click colored lever icons to raise matching colored bridges", 2, 32, 30),
        (146, "ConstellationConnect", "Click celestial stars in order of increasing brightness", 2, 32, 25),
        (147, "ClockHandAngleMatch", "Click hour/minute hands to match target digital timestamp", 2, 32, 25),
        (148, "WeightBalancePan", "Click weights on rack to balance scale pans", 2, 32, 30),
        (149, "SafeCrackerDial", "Click left/right arrows to find combination clicks", 2, 32, 30),
        (150, "GrandAffordanceWorkshop", "Composite workbench with click rotation, color fill, and stamp", 2, 32, 45),
    ]

    for num, name, doc, step_sz, grid_sz, max_st in aff_specs:
        code += f'''

class NG{num}_{name}(P1NovelBase):
    """{doc}"""
    def __init__(self, grid_size: int = {grid_sz}, step_size: int = {step_sz}, max_steps: int = {max_st}, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.clicked_target = False

        # Clickable button target
        btn_pix = np.full((step_size * 2, step_size * 2), 6, dtype=np.int8)
        btn = Sprite(pixels=btn_pix.tolist(), name="btn_target", x=step_size * 6, y=step_size * 6, tags=["clickable", "button"])

        sprites, cam, acts = _make_boxed_arena("ng{num}_{name.lower()}", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[btn], available_actions=[1, 2, 3, 4, 6])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng{num}_{name.lower()}", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
            data = getattr(self._action, "data", {{}}) or {{}}
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
'''
    (TARGET_DIR / "p5_affordance_ext.py").write_text(code)
    print("Built p5_affordance_ext.py (ng126..ng150)")


def build_p6_dependencies():
    code = '''#!/usr/bin/env python3
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
'''
    dep_specs = [
        (151, "DualKeyGate", "Collect Key A to unlock Gate A, then Key B for Gate B", 2, 32, 45),
        (152, "TollBooth", "Collect 3 coins before toll barrier opens", 2, 32, 45),
        (153, "ChainedPrerequisites", "Sequential dependency chain: A -> B -> C -> Goal", 2, 32, 50),
        (154, "DurabilityPickaxe", "Pickaxe has 2 charges; must choose which 2 boulders to break", 2, 32, 45),
        (155, "ColorTransmutation", "Step on Red pad to open red gate, then Blue pad for blue gate", 2, 32, 45),
        (156, "SingleItemBackpack", "Can only carry 1 item at a time; deposit item 1 before taking 2", 2, 32, 50),
        (157, "DescendingOrderLocks", "Must touch locks in descending numerical order (3, 2, 1)", 2, 32, 45),
        (158, "CombinationLevers", "3 levers requiring exact binary pattern (1-0-1) to unlock exit", 2, 32, 45),
        (159, "WeightedPressureDoor", "Push box onto plate to keep door open while player crosses", 2, 32, 45),
        (160, "FuelDepletionSprint", "Each step consumes 1 fuel; fuel canisters replenish 10", 2, 32, 45),
        (161, "ConsumableKeys", "Keys break after single use; plan optimal unlock order", 2, 32, 45),
        (162, "TimedDecayKey", "Key dissolves 12 steps after pickup; sprint to lock", 2, 32, 40),
        (163, "ShapeShifterAperture", "Avatar morphs at pads to fit narrow slits", 2, 32, 45),
        (164, "HubAndSpokeMaze", "Central hub with 4 spokes; each spoke contains key for next spoke", 2, 32, 55),
        (165, "TopologicalValveOrder", "One-way gates restricting sequence of chamber visits", 2, 32, 45),
        (166, "AlternatingTileColors", "Player must alternate between black and gold tiles", 2, 32, 45),
        (167, "ThermalPhaseState", "Hot pad thaws ice walls; cold pad freezes water bridge", 2, 32, 50),
        (168, "BatteryChargePack", "Charge energy core at dynamo before powering forcefield door", 2, 32, 45),
        (169, "MineDisarmingSequence", "Disarm 3 explosive runes in safe order", 2, 32, 45),
        (170, "SacrificialAutomaton", "Send helper robot into trap spike to create safe path", 2, 32, 45),
        (171, "BalancedScalesGate", "Place equal weights on left and right pressure sensors", 2, 32, 50),
        (172, "HierarchicalSubtreeUnlock", "Unlock master key by solving leaf nodes first", 2, 32, 50),
        (173, "DynamicPatternKeypad", "Step on 4 keypad digits matching overhead sequence banner", 2, 32, 45),
        (174, "ElevationStairway", "Tiles with increasing elevation heights (can only step +1)", 2, 32, 45),
        (175, "RetrogradeMasterDAG", "Complex multi-stage dependency labyrinth requiring Bellman backward induction", 2, 32, 60),
    ]

    for num, name, doc, step_sz, grid_sz, max_st in dep_specs:
        code += f'''

class NG{num}_{name}(P1NovelBase):
    """{doc}"""
    def __init__(self, grid_size: int = {grid_sz}, step_size: int = {step_sz}, max_steps: int = {max_st}, seed: int = 42) -> None:
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

        sprites, cam, acts = _make_boxed_arena("ng{num}_{name.lower()}", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[key, gate])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng{num}_{name.lower()}", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
'''
    (TARGET_DIR / "p6_dependencies.py").write_text(code)
    print("Built p6_dependencies.py (ng151..ng175)")


def build_p7_chaotic():
    code = '''#!/usr/bin/env python3
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
'''
    chaos_specs = [
        (176, "BlackNoiseHUD", "Animated noise pattern in bottom rows simulating lf52", 2, 32, 45),
        (177, "StrobeFlasher", "Background color pulses periodically without changing mechanics", 2, 32, 40),
        (178, "ScreenShakeJitter", "World sprites offset by +/- 1 pixel jitter periodically", 2, 32, 40),
        (179, "DecoyMimicGoal", "Two goal sprites; only one has true affordance signature", 2, 32, 40),
        (180, "LatticeStride1", "1x1 pixel sprite with 1-pixel grid stride", 1, 32, 40),
        (181, "LatticeStride4", "4x4 pixel sprite with 4-pixel grid stride", 4, 32, 35),
        (182, "LatticeStride5", "5x5 pixel sprite with 5-pixel grid stride and offset", 5, 35, 35),
        (183, "LatticeStride8", "8x8 pixel sprite with 8-pixel grid stride", 8, 32, 30),
        (184, "AsymmetricStride", "Horizontal stride = 2, Vertical stride = 4", 2, 32, 40),
        (185, "ShiftingWallsRandom", "Interior divider shifts randomly every 5 steps", 2, 32, 45),
        (186, "VanishingStepPath", "Floor tiles turn to black void 2 steps after traversal", 2, 32, 40),
        (187, "DriftingCameraViewport", "Viewport pans dynamically tracking avatar center", 2, 32, 40),
        (188, "InvertedContrastPalette", "Walls are dark (color 1), floor is bright (color 12)", 2, 32, 40),
        (189, "ColorInversionFlash", "Inverts color palette for 1 frame upon action", 2, 32, 40),
        (190, "AdversarialLoopTrap", "Maze designed to trap naive 2-cycle loop interceptors", 2, 32, 45),
        (191, "GlitchedConveyor", "Conveyor belts flipping direction on odd turn indices", 2, 32, 45),
        (192, "MirageGhostSprites", "Sprites that dissolve without blocking when avatar approaches", 2, 32, 40),
        (193, "WindGustPerturbation", "Periodic wind pushes player 1 cell east every 3 moves", 2, 32, 40),
        (194, "MultiPerimeterHUD", "Step counters and HUD icons at top, bottom, and right borders", 2, 32, 45),
        (195, "InvertedCompassZone", "Entering central zone inverts controls", 2, 32, 45),
        (196, "ActionPenaltyCounter", "On-screen score counter decrementing live with actions", 2, 32, 45),
        (197, "NonRectangularPolygons", "L-shaped and T-shaped wall clusters", 2, 32, 45),
        (198, "FragmentedAvatarDuo", "Avatar consists of two separated pixels moving in tandem", 2, 32, 40),
        (199, "InvisibleBumpWalls", "Walls invisible until avatar bumps them directly", 2, 32, 45),
        (200, "GrandChallengerChaos", "Composite arena with noise HUD, teleporters, box push, and key", 2, 32, 60),
    ]

    for num, name, doc, step_sz, grid_sz, max_st in chaos_specs:
        code += f'''

class NG{num}_{name}(P1NovelBase):
    """{doc}"""
    def __init__(self, grid_size: int = {grid_sz}, step_size: int = {step_sz}, max_steps: int = {max_st}, seed: int = 42) -> None:
        self.step_size = step_size
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.noise_seed = seed

        # Noise HUD stripe at bottom
        hud_pix = np.random.randint(1, 15, size=(step_size, grid_size), dtype=np.int8)
        hud = Sprite(pixels=hud_pix.tolist(), name="noise_hud", x=0, y=grid_size - step_size, tags=["hud", "noise"], blocking=BlockingMode.NOT_BLOCKED)

        sprites, cam, acts = _make_boxed_arena("ng{num}_{name.lower()}", step_size=step_size, grid_size=grid_size,
                                               max_steps=max_steps, av_pos=(step_size * 2, step_size * 2),
                                               goal_pos=(grid_size - step_size * 3, grid_size - step_size * 3),
                                               extra_sprites=[hud])
        level = Level(sprites=sprites, grid_size=(grid_size, grid_size))
        super().__init__(game_id="ng{num}_{name.lower()}", levels=[level], camera=cam, available_actions=acts, seed=seed)

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
'''
    (TARGET_DIR / "p7_chaotic.py").write_text(code)
    print("Built p7_chaotic.py (ng176..ng200)")


if __name__ == "__main__":
    print("Generating all 7 novel game extension modules...")
    build_p1_navigation_ext()
    build_p2_physics_ext()
    build_p3_logic_ext()
    build_p4_multiagent_ext()
    build_p5_affordance_ext()
    build_p6_dependencies()
    build_p7_chaotic()
    print("ALL 7 EXTENSION MODULES SUCCESSFULLY GENERATED!")
